//! The buffer pool: fixed frames, partitioned page table, clock sweep (spec 7.1 to 7.7).

use crate::{BufError, Smgr, WalFlush};
use jamsil_page::page::{self, PageBuf, PageMut, PageRef};
use jamsil_page::walrec::WalPermit;
use jamsil_page::*;
use jamsil_vfs::sync::atomic::{AtomicU64, Ordering};
use jamsil_vfs::sync::latch::{self, Class};
use jamsil_vfs::sync::{thread, Arc, Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::time::Duration;

const PARTITIONS: usize = 128;

/// One lock per fork, so that two extensions of the same fork are serialised (spec 7.8).
type ExtendLocks = HashMap<(RelFileId, ForkId), Arc<Mutex<()>>>;

const PIN_MASK: u64 = (1 << 18) - 1;
const USAGE_SHIFT: u64 = 18;
const USAGE_MASK: u64 = 0xF << USAGE_SHIFT;
const MAX_USAGE: u64 = 5;
const VALID: u64 = 1 << 22;
const DIRTY: u64 = 1 << 23;
const IO_IN_PROGRESS: u64 = 1 << 24;
const JUST_DIRTIED: u64 = 1 << 25;
const CHECKPOINT_NEEDED: u64 = 1 << 26;
const IO_ERROR: u64 = 1 << 27;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    Normal,
    /// Redo: a missing or all-zero page becomes a fresh zero page; damaged pages still fail.
    ZeroOnMissing,
    /// Redo of a record with a full-page image: do not read the disk at all.
    Init,
    /// Repair tools: no verification.
    NoVerify,
}

/// The frame's dirty flag, the half of the hint/checkpoint protocol that lives in the buffer pool
/// (spec 7.5, 7.6, 9.2). Both operations are `SeqCst` on purpose: a page change sets the flag and
/// then reads the checkpoint's published redo point, and a checkpoint publishes the redo point and
/// then scans for the flag. Each side stores before it loads, and only `SeqCst` guarantees that at
/// least one of them sees the other. The loom model `tests/loom_dirty_redo.rs` of `jamsil-engine`
/// runs exactly these functions together with `jamsil_wal::redo_point`.
pub mod dirty_flag {
    use super::{DIRTY, JUST_DIRTIED};
    use jamsil_vfs::sync::atomic::{AtomicU64, STORE_BEFORE_LOAD};

    /// Marks the frame dirty (the store of the protocol).
    pub fn set(state: &AtomicU64) {
        state.fetch_or(DIRTY | JUST_DIRTIED, STORE_BEFORE_LOAD);
        // Loom treats a `SeqCst` access as `AcqRel` (its README: "false alarms"), but models
        // `fence(SeqCst)` exactly. The fence adds nothing to the `SeqCst` store above, so under
        // loom it lets the model check the same guarantee; production does not pay for it.
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
    }

    /// True if the frame is dirty (the load of the protocol: the checkpoint's scan).
    pub fn is_set(state: &AtomicU64) -> bool {
        state.load(STORE_BEFORE_LOAD) & DIRTY != 0
    }
}

struct Frame {
    state: AtomicU64,
    tag: Mutex<Option<PageTag>>,
    content: RwLock<PageBuf>,
    io_mx: Mutex<()>,
    io_cv: Condvar,
}

#[derive(Default)]
pub struct BufMetrics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub backend_writes: AtomicU64,
    pub checkpoint_pages: AtomicU64,
}

#[derive(Default)]
pub struct CheckpointCtx {
    /// Sleep between page writes (pacing, spec 7.7).
    pub pace: Option<Duration>,
}

pub struct BufferPool {
    frames: Vec<Frame>,
    table: Vec<Mutex<HashMap<PageTag, usize>>>,
    clock: AtomicU64,
    smgr: Arc<Smgr>,
    wal: Arc<dyn WalFlush>,
    extend_locks: Mutex<ExtendLocks>,
    pub metrics: BufMetrics,
}

fn partition(tag: &PageTag) -> usize {
    let mut h = tag.rel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (tag.block as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= (tag.fork as u64) << 56;
    h ^= h >> 29;
    (h as usize) % PARTITIONS
}

impl BufferPool {
    pub fn new(n_frames: usize, smgr: Arc<Smgr>, wal: Arc<dyn WalFlush>) -> Arc<Self> {
        assert!(n_frames >= 8);
        let frames = (0..n_frames)
            .map(|_| Frame {
                state: AtomicU64::new(0),
                tag: Mutex::new(None),
                content: RwLock::new([0u8; PAGE_SIZE]),
                io_mx: Mutex::new(()),
                io_cv: Condvar::new(),
            })
            .collect();
        Arc::new(Self {
            frames,
            table: (0..PARTITIONS).map(|_| Mutex::new(HashMap::new())).collect(),
            clock: AtomicU64::new(0),
            smgr,
            wal,
            extend_locks: Mutex::new(HashMap::new()),
            metrics: BufMetrics::default(),
        })
    }

    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }
    pub fn smgr(&self) -> &Arc<Smgr> {
        &self.smgr
    }

    // ---- pin / unpin ------------------------------------------------------------------

    fn try_pin_existing(&self, idx: usize) {
        let st = &self.frames[idx].state;
        let mut cur = st.load(Ordering::SeqCst);
        loop {
            let usage = ((cur & USAGE_MASK) >> USAGE_SHIFT).min(MAX_USAGE - 1) + 1;
            let new = (cur & !USAGE_MASK & !PIN_MASK) | (usage << USAGE_SHIFT) | ((cur & PIN_MASK) + 1);
            match st.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }

    fn unpin(&self, idx: usize) {
        let st = &self.frames[idx].state;
        let mut cur = st.load(Ordering::SeqCst);
        loop {
            debug_assert!(cur & PIN_MASK > 0);
            let mut new = cur - 1;
            if new & PIN_MASK == 0 && new & IO_ERROR != 0 {
                new = 0;
            }
            match st.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }

    fn set_flags(&self, idx: usize, flags: u64) {
        self.frames[idx].state.fetch_or(flags, Ordering::SeqCst);
    }
    fn clear_flags(&self, idx: usize, flags: u64) {
        self.frames[idx].state.fetch_and(!flags, Ordering::SeqCst);
    }

    fn wait_io(&self, idx: usize) {
        let f = &self.frames[idx];
        let mut g = f.io_mx.lock();
        while f.state.load(Ordering::SeqCst) & IO_IN_PROGRESS != 0 {
            f.io_cv.wait_for(&mut g, Duration::from_millis(50));
        }
    }

    fn finish_io(&self, idx: usize) {
        let f = &self.frames[idx];
        let _g = f.io_mx.lock();
        f.state.fetch_and(!IO_IN_PROGRESS, Ordering::SeqCst);
        f.io_cv.notify_all();
    }

    /// Pins the page, reading it from disk if needed. Never returns an unverified page
    /// (except in `NoVerify` mode).
    pub fn pin(&self, tag: PageTag, mode: ReadMode) -> Result<PinnedPage<'_>, BufError> {
        loop {
            let part = &self.table[partition(&tag)];
            let hit = {
                let t = part.lock();
                match t.get(&tag) {
                    Some(&idx) => {
                        self.try_pin_existing(idx);
                        Some(idx)
                    }
                    None => None,
                }
            };
            if let Some(idx) = hit {
                self.wait_io(idx);
                if self.frames[idx].state.load(Ordering::SeqCst) & IO_ERROR != 0 {
                    self.unpin(idx);
                    // The loader failed; retry, which re-reads and reports its own error.
                    continue;
                }
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(PinnedPage { pool: PoolRef::Borrowed(self), idx, tag });
            }
            // Miss: get a victim frame.
            let idx = self.claim_victim()?;
            {
                let mut t = part.lock();
                if t.contains_key(&tag) {
                    // Lost a race to load the same page: free our frame and retry as a hit.
                    drop(t);
                    *self.frames[idx].tag.lock() = None;
                    self.frames[idx].state.store(0, Ordering::SeqCst);
                    continue;
                }
                t.insert(tag, idx);
                *self.frames[idx].tag.lock() = Some(tag);
                // pin 1 (ours), usage 1, loading.
                self.frames[idx].state.store(1 | (1 << USAGE_SHIFT) | IO_IN_PROGRESS, Ordering::SeqCst);
            }
            self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            match self.load(idx, tag, mode) {
                Ok(()) => {
                    self.set_flags(idx, VALID);
                    self.finish_io(idx);
                    return Ok(PinnedPage { pool: PoolRef::Borrowed(self), idx, tag });
                }
                Err(e) => {
                    {
                        let mut t = part.lock();
                        t.remove(&tag);
                        *self.frames[idx].tag.lock() = None;
                    }
                    self.set_flags(idx, IO_ERROR);
                    self.clear_flags(idx, VALID);
                    self.finish_io(idx);
                    self.unpin(idx);
                    return Err(e);
                }
            }
        }
    }

    /// Like `pin`, but the pin owns a handle of the pool, so it can be kept in a structure that outlives
    /// the call: a commit pins its pages at its gate and keeps the pins until step 6 (spec 12.11).
    pub fn pin_owned(self: &Arc<Self>, tag: PageTag, mode: ReadMode) -> Result<PinnedPage<'static>, BufError> {
        let p = self.pin(tag, mode)?;
        let (idx, tag) = (p.idx, p.tag);
        // The pin moves into the owned one: it is not released here, and the new one releases it.
        std::mem::forget(p);
        Ok(PinnedPage { pool: PoolRef::Owned(self.clone()), idx, tag })
    }

    fn load(&self, idx: usize, tag: PageTag, mode: ReadMode) -> Result<(), BufError> {
        let mut g = self.frames[idx].content.write();
        let buf: &mut PageBuf = &mut g;
        if mode == ReadMode::Init {
            buf.fill(0);
            return Ok(());
        }
        let mut attempts = 0;
        loop {
            attempts += 1;
            let present = self.smgr.read(tag, buf)?;
            if !present {
                if mode == ReadMode::ZeroOnMissing {
                    buf.fill(0);
                    return Ok(());
                }
                return Err(BufError::Corrupt(format!("block beyond end of file: {tag:?}")));
            }
            if mode == ReadMode::NoVerify {
                return Ok(());
            }
            if page::is_all_zero(buf) {
                if mode == ReadMode::ZeroOnMissing {
                    return Ok(());
                }
                if attempts >= 2 {
                    return Err(BufError::Corrupt(format!("zero page at {tag:?}")));
                }
                continue;
            }
            let check = page::verify(buf, &tag).and_then(|_| {
                let lsn = PageRef(&buf[..]).lsn();
                if lsn > self.wal.insert_lsn() {
                    corrupt(format!("page_lsn {lsn} beyond WAL position at {tag:?}"))
                } else {
                    Ok(())
                }
            });
            match check {
                Ok(()) => return Ok(()),
                Err(e) if attempts >= 2 => return Err(BufError::Corrupt(e.0)),
                Err(_) => continue, // re-read once
            }
        }
    }

    // ---- eviction ---------------------------------------------------------------------

    /// Returns a frame that is pinned by the caller, holds no mapping and is reset except for pin.
    fn claim_victim(&self) -> Result<usize, BufError> {
        let n = self.frames.len();
        let mut steps = 0usize;
        let mut full_sweeps_pinned = 0usize;
        loop {
            let idx = (self.clock.fetch_add(1, Ordering::Relaxed) % n as u64) as usize;
            steps += 1;
            if steps > n * (MAX_USAGE as usize + 2) {
                // Every frame stayed pinned through several sweeps.
                full_sweeps_pinned += 1;
                if full_sweeps_pinned > 3 {
                    return Err(BufError::NoFreeBuffer);
                }
                steps = 0;
                thread::sleep(Duration::from_millis(1));
            }
            let st = &self.frames[idx].state;
            let cur = st.load(Ordering::SeqCst);
            if cur & PIN_MASK != 0 {
                continue;
            }
            if cur & USAGE_MASK != 0 {
                let new = cur - (1 << USAGE_SHIFT);
                let _ = st.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst);
                continue;
            }
            // pin 0, usage 0: try to claim.
            if st.compare_exchange(cur, (cur & !PIN_MASK) | 1, Ordering::SeqCst, Ordering::SeqCst).is_err() {
                continue;
            }
            let old_tag = *self.frames[idx].tag.lock();
            let Some(old_tag) = old_tag else {
                st.store(1, Ordering::SeqCst);
                return Ok(idx);
            };
            if cur & DIRTY != 0 {
                // Write first, by the protocol of 7.5; if it was re-dirtied or pinned, give up.
                if self.write_frame(idx, old_tag, false).is_err() {
                    st.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
                self.metrics.backend_writes.fetch_add(1, Ordering::Relaxed);
            }
            let part = &self.table[partition(&old_tag)];
            let mut t = part.lock();
            let now = st.load(Ordering::SeqCst);
            if now & PIN_MASK != 1 || now & DIRTY != 0 {
                drop(t);
                st.fetch_sub(1, Ordering::SeqCst);
                continue;
            }
            t.remove(&old_tag);
            *self.frames[idx].tag.lock() = None;
            st.store(1, Ordering::SeqCst);
            self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
            return Ok(idx);
        }
    }

    // ---- writing pages out (spec 7.5, mirror image) -----------------------------------

    /// Writes one frame. The caller holds a pin. `wait_latch` is for the checkpointer.
    fn write_frame(&self, idx: usize, tag: PageTag, _wait_latch: bool) -> Result<(), BufError> {
        let f = &self.frames[idx];
        let mut copy = page::new_page_buf();
        {
            let g = f.content.read();
            // Clear JUST_DIRTIED under the shared latch: any later change sets it again.
            self.clear_flags(idx, JUST_DIRTIED);
            copy.copy_from_slice(&g[..]);
        }
        if f.state.load(Ordering::SeqCst) & DIRTY == 0 {
            return Ok(());
        }
        let lsn = PageRef(&copy[..]).lsn();
        self.wal.flush(lsn)?;
        page::set_checksum(&mut copy[..], &tag);
        self.smgr.write(tag, &copy[..])?;
        // Clear DIRTY unless the page changed again meanwhile.
        let st = &f.state;
        let mut cur = st.load(Ordering::SeqCst);
        loop {
            if cur & JUST_DIRTIED != 0 {
                break;
            }
            let new = cur & !(DIRTY | CHECKPOINT_NEEDED);
            match st.compare_exchange(cur, new, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
        Ok(())
    }

    /// Marks every dirty frame `CHECKPOINT_NEEDED`. Call right after the redo point is taken, and
    /// pass it: the scan is the second half of the protocol of `dirty_flag`, so it must run after
    /// the point was published, or a hint that read the old point could be missed. A debug build
    /// checks that the point is published (spec 7.6, 9.2).
    pub fn mark_checkpoint_needed(&self, redo: Lsn) -> usize {
        debug_assert!(
            self.wal.redo_lsn() >= redo,
            "the checkpoint scan ran before its redo point {redo} was published (it is {})",
            self.wal.redo_lsn()
        );
        let mut n = 0;
        for i in 0..self.frames.len() {
            let st = self.frames[i].state.load(Ordering::SeqCst);
            // The scan of the protocol in `dirty_flag`: a hint that has not seen the new redo point
            // has set this flag before the point was published, so it is found here.
            if dirty_flag::is_set(&self.frames[i].state) && st & VALID != 0 {
                self.set_flags(i, CHECKPOINT_NEEDED);
                n += 1;
            }
        }
        n
    }

    /// Checkpointer only: writes every frame that was dirty when `mark_checkpoint_needed`
    /// ran, sorted by (relfile, fork, block). Does not sync; the caller drains `smgr.sync_all`.
    pub fn flush_all(&self, ctx: &CheckpointCtx) -> Result<(), BufError> {
        let mut work: Vec<(PageTag, usize)> = Vec::new();
        for i in 0..self.frames.len() {
            let st = self.frames[i].state.load(Ordering::SeqCst);
            if st & CHECKPOINT_NEEDED != 0 {
                if let Some(t) = *self.frames[i].tag.lock() {
                    work.push((t, i));
                }
            }
        }
        work.sort();
        for (tag, idx) in work {
            // Re-resolve through the page table: the frame may have been evicted and reused.
            let pinned = {
                let t = self.table[partition(&tag)].lock();
                match t.get(&tag) {
                    Some(&i) if i == idx => {
                        self.try_pin_existing(i);
                        Some(i)
                    }
                    _ => None,
                }
            };
            let Some(i) = pinned else { continue };
            self.wait_io(i);
            let st = self.frames[i].state.load(Ordering::SeqCst);
            let r = if st & DIRTY != 0 && st & CHECKPOINT_NEEDED != 0 {
                let r = self.write_frame(i, tag, true);
                if r.is_ok() {
                    self.metrics.checkpoint_pages.fetch_add(1, Ordering::Relaxed);
                }
                r
            } else {
                Ok(())
            };
            self.clear_flags(i, CHECKPOINT_NEEDED);
            self.unpin(i);
            r?;
            if let Some(d) = ctx.pace {
                thread::sleep(d);
            }
        }
        Ok(())
    }

    /// Writes up to `max` dirty, unpinned, cold frames (background writer, spec 7.7).
    pub fn bg_write(&self, max: usize) -> usize {
        let mut written = 0;
        let n = self.frames.len();
        for _ in 0..n {
            if written >= max {
                break;
            }
            let idx = (self.clock.load(Ordering::Relaxed) as usize + written + 1) % n;
            let st = self.frames[idx].state.load(Ordering::SeqCst);
            if st & DIRTY == 0 || st & PIN_MASK != 0 || st & USAGE_MASK != 0 || st & VALID == 0 {
                continue;
            }
            let Some(tag) = *self.frames[idx].tag.lock() else { continue };
            let pinned = {
                let t = self.table[partition(&tag)].lock();
                match t.get(&tag) {
                    Some(&i) if i == idx => {
                        self.try_pin_existing(i);
                        true
                    }
                    _ => false,
                }
            };
            if pinned {
                if self.write_frame(idx, tag, false).is_ok() {
                    written += 1;
                }
                self.unpin(idx);
            }
        }
        written
    }

    /// Discards all frames of a dropped relation without writing them.
    pub fn drop_relation(&self, rel: RelFileId) {
        for idx in 0..self.frames.len() {
            let tag = *self.frames[idx].tag.lock();
            let Some(tag) = tag else { continue };
            if tag.rel != rel {
                continue;
            }
            let mut t = self.table[partition(&tag)].lock();
            let st = &self.frames[idx].state;
            let cur = st.load(Ordering::SeqCst);
            if cur & PIN_MASK != 0 {
                continue; // caller error: a dropped relation cannot be pinned
            }
            if t.get(&tag) == Some(&idx) {
                t.remove(&tag);
                *self.frames[idx].tag.lock() = None;
                st.store(0, Ordering::SeqCst);
            }
        }
    }

    /// Discards the frames of one fork below `block` without writing them: the pages of a commit log
    /// whose segments were unlinked (spec 12.2). A pinned frame is skipped (a caller error).
    pub fn drop_blocks_below(&self, rel: RelFileId, fork: ForkId, block: BlockNo) {
        for idx in 0..self.frames.len() {
            let tag = *self.frames[idx].tag.lock();
            let Some(tag) = tag else { continue };
            if tag.rel != rel || tag.fork != fork || tag.block >= block {
                continue;
            }
            let mut t = self.table[partition(&tag)].lock();
            let st = &self.frames[idx].state;
            if st.load(Ordering::SeqCst) & PIN_MASK != 0 {
                continue;
            }
            if t.get(&tag) == Some(&idx) {
                t.remove(&tag);
                *self.frames[idx].tag.lock() = None;
                st.store(0, Ordering::SeqCst);
            }
        }
    }

    /// Whether the page is in the pool and dirty (tests).
    pub fn is_dirty(&self, tag: PageTag) -> bool {
        let t = self.table[partition(&tag)].lock();
        t.get(&tag).is_some_and(|&idx| dirty_flag::is_set(&self.frames[idx].state))
    }

    /// Number of dirty frames (diagnostics and tests).
    pub fn dirty_count(&self) -> usize {
        self.frames.iter().filter(|f| f.state.load(Ordering::SeqCst) & DIRTY != 0).count()
    }

    /// Drops every unpinned frame without writing (used by simulated crashes and generation switch).
    pub fn discard_all(&self) {
        for idx in 0..self.frames.len() {
            let tag = *self.frames[idx].tag.lock();
            if let Some(tag) = tag {
                let mut t = self.table[partition(&tag)].lock();
                t.remove(&tag);
            }
            *self.frames[idx].tag.lock() = None;
            self.frames[idx].state.store(0, Ordering::SeqCst);
        }
    }

    // ---- extension (spec 7.8) ---------------------------------------------------------

    /// Extends a fork by `n` pages. Every fork is extended the same way: the new pages are
    /// formatted in the pool with a valid header of kind 0 (no special area, zero body) and marked
    /// dirty; `log` then inserts the extend record and returns its end LSN, which is set on every
    /// new frame. The kind, special area and content come from the first record that uses a page,
    /// which carries `WILL_INIT` (spec 7.8, answer A9). Returns the first new block number.
    pub fn extend(
        &self,
        rel: RelFileId,
        fork: ForkId,
        n: u32,
        log: impl FnOnce(u32, u32) -> Result<Lsn, BufError>,
    ) -> Result<BlockNo, BufError> {
        assert!(n >= 1 && (n as usize) <= (self.frames.len() / 32).max(1), "extension chunk too large");
        let lock = self.extend_locks.lock().entry((rel, fork)).or_default().clone();
        let _g = lock.lock();
        let first = self.smgr.nblocks(rel, fork)?;
        self.smgr.reserve(rel, fork, first, n)?;
        // The length rises before the first new frame is dirtied: a write-out of one of them is
        // part of the extension, and may not run ahead of the known length (spec 6.7).
        self.smgr.ensure_len(rel, fork, first + n)?;
        let mut pins = Vec::new();
        for b in first..first + n {
            let p = self.pin(PageTag::new(rel, fork, b), ReadMode::Init)?;
            {
                let mut w = p.write();
                PageMut(&mut w[..]).init(page::PageKind::Unused, 0);
                w.mark_dirty();
            }
            pins.push(p);
        }
        let lsn = log(first, first + n)?;
        for p in &pins {
            p.write().set_lsn(lsn);
        }
        Ok(first)
    }
}

/// How a pin reaches its pool: by a borrow, or by a handle of its own (`pin_owned`).
enum PoolRef<'a> {
    Borrowed(&'a BufferPool),
    Owned(Arc<BufferPool>),
}

impl std::ops::Deref for PoolRef<'_> {
    type Target = BufferPool;
    fn deref(&self) -> &BufferPool {
        match self {
            PoolRef::Borrowed(p) => p,
            PoolRef::Owned(p) => p,
        }
    }
}

pub struct PinnedPage<'a> {
    pool: PoolRef<'a>,
    idx: usize,
    tag: PageTag,
}

impl Drop for PinnedPage<'_> {
    fn drop(&mut self) {
        self.pool.unpin(self.idx);
    }
}

impl<'a> PinnedPage<'a> {
    pub fn tag(&self) -> PageTag {
        self.tag
    }

    /// Marks the frame dirty without the latch. The flag protocol needs none: a write-out that is in
    /// flight clears `JUST_DIRTIED` before it copies the page and keeps `DIRTY` if this call set it again
    /// meanwhile, and one that finished before leaves the flag for this call to set. A writer that has
    /// to change several pages under one record calls it for each page just before it inserts the record,
    /// so that every frame is dirty before the record has a position in the log (spec 7.5), also when the
    /// page was unlatched, and so possibly written out, since it was first marked.
    pub fn mark_dirty(&self) {
        dirty_flag::set(&self.pool.frames[self.idx].state);
    }

    fn ticket(&self, class: Class) -> latch::Ticket {
        latch::acquire(class, (self.tag.rel, self.tag.block))
    }

    /// Shared latch, unchecked by the latch-order checker (tests and raw pages).
    pub fn read(&self) -> PageRead<'_> {
        self.read_as(Class::Unordered)
    }
    /// Exclusive latch, unchecked by the latch-order checker.
    pub fn write(&self) -> PageWrite<'_> {
        self.write_as(Class::Unordered)
    }
    /// Shared latch of a given class; debug builds check the order of spec 13.6.
    pub fn read_as(&self, class: Class) -> PageRead<'_> {
        let t = self.ticket(class);
        PageRead { g: self.pool.frames[self.idx].content.read(), _t: t }
    }
    /// Exclusive latch of a given class.
    pub fn write_as(&self, class: Class) -> PageWrite<'_> {
        let t = self.ticket(class);
        PageWrite { g: self.pool.frames[self.idx].content.write(), pool: &self.pool, idx: self.idx, tag: self.tag, _t: t }
    }
    /// Never blocks.
    pub fn try_write(&self) -> Option<PageWrite<'_>> {
        self.try_write_as(Class::Unordered)
    }

    /// Sets a hint bit, if it is cheap to do so (spec 7.6). Never blocks and never fails: it takes
    /// the latch with `try_write`, asks the log for a permit for one page image, and only then runs
    /// `set` and `mark_dirty_hint`. If the latch is contended or the log does not accept records,
    /// `set` does not run and the hint is skipped; a hint is always optional, so heap and B-tree
    /// code never has to pass the gate of spec 8.8 for one. Returns whether the hint was set.
    pub fn try_hint(&self, set: impl FnOnce(&mut PageBuf)) -> bool {
        self.try_hint_as(Class::Unordered, set)
    }
    pub fn try_hint_as(&self, class: Class, set: impl FnOnce(&mut PageBuf)) -> bool {
        let Some(mut w) = self.try_write_as(class) else { return false };
        let Some(permit) = self.pool.wal.permit_page_image() else { return false };
        set(&mut w);
        w.mark_dirty_hint(permit);
        true
    }
    pub fn try_write_as(&self, class: Class) -> Option<PageWrite<'_>> {
        let t = self.ticket(class);
        self.pool.frames[self.idx]
            .content
            .try_write()
            .map(|g| PageWrite { g, pool: &self.pool, idx: self.idx, tag: self.tag, _t: t })
    }
}

pub struct PageRead<'a> {
    g: RwLockReadGuard<'a, PageBuf>,
    _t: latch::Ticket,
}

impl Deref for PageRead<'_> {
    type Target = PageBuf;
    fn deref(&self) -> &PageBuf {
        &self.g
    }
}

pub struct PageWrite<'a> {
    g: RwLockWriteGuard<'a, PageBuf>,
    pool: &'a BufferPool,
    idx: usize,
    tag: PageTag,
    _t: latch::Ticket,
}

impl Deref for PageWrite<'_> {
    type Target = PageBuf;
    fn deref(&self) -> &PageBuf {
        &self.g
    }
}
impl DerefMut for PageWrite<'_> {
    fn deref_mut(&mut self) -> &mut PageBuf {
        &mut self.g
    }
}

impl PageWrite<'_> {
    /// Call after changing the bytes and BEFORE inserting the WAL record.
    pub fn mark_dirty(&mut self) {
        dirty_flag::set(&self.pool.frames[self.idx].state);
    }
    /// Call with the record's end LSN, before dropping the guard.
    pub fn set_lsn(&mut self, lsn: Lsn) {
        PageMut(&mut self.g[..]).set_lsn(lsn);
    }
    /// For hint-only changes (spec 7.6), and only through `PinnedPage::try_hint`, which has taken
    /// the permit. The rule lives here so no caller has to remember it: if the page has not been
    /// logged since the current redo point, a torn write of the hint could destroy the page, so a
    /// standalone image record (hint included) is inserted and the page LSN advanced. Otherwise the
    /// frame is only marked dirty.
    ///
    /// The order is what closes the race with a checkpoint, without a lock: set the dirty flag
    /// first, then read the redo point. A checkpoint publishes its redo point before it scans the
    /// frames, so both sides store before they load and, with `SeqCst`, at least one sees the
    /// other. Either the scan finds the frame dirty and writes the page before the checkpoint can
    /// complete, or this call reads the new redo point and logs the image (answer A5).
    pub(crate) fn mark_dirty_hint(&mut self, permit: WalPermit) {
        self.mark_dirty();
        let lsn = PageRef(&self.g[..]).lsn();
        if lsn <= self.pool.wal.redo_lsn() {
            let end = self.pool.wal.insert_page_image(permit, self.tag, &self.g[..]);
            self.set_lsn(end);
        }
    }
}
