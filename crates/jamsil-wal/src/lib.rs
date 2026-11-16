//! Write-ahead log (spec section 8): one append-only stream of checksummed redo records.
//!
//! Deviations from the spec text (see Question.md): insertion is serialized by one mutex rather
//! than 8 slots, and the buffer is unbounded. Group commit is leader-based: a waiter that finds no
//! flush in flight performs the write and sync for everyone (spec 8.5, answer B2). A WAL writer
//! thread, started with [`Wal::start_writer`], covers the case that has no waiter.
#![forbid(unsafe_code)]

mod reader;

pub use reader::{ReadStop, WalReader};

use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use jamsil_vfs::sync::{thread, Arc, Condvar, Mutex, Weak};
use jamsil_vfs::{IoError, IoErrorKind, IoResult, OpenOpts, Vfs, VfsFile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Spare segments kept ahead of the write position (spec 8.1, 5.5).
const SPARES: u64 = 2;
/// Bytes of unflushed log that wake the writer thread (a quarter of the nominal buffer, spec 8.5).
/// `storage.wal_buffer_mib`, default 64 (spec 8.4, 18.1).
pub const DEFAULT_WAL_BUFFER: u64 = 64 << 20;
/// Interval of the WAL writer thread (spec 8.5).
pub const WRITER_INTERVAL: Duration = Duration::from_millis(200);

/// What clears a quarantine (spec 17.2): the two kinds are told apart because they are cleared
/// differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarantineKind {
    /// Corruption or divergence: only a snapshot install clears it. A restart would meet the same damage.
    Corruption,
    /// A full disk, a panicked background thread, a resource failure that did not clear: a restart
    /// clears it, without a snapshot.
    Restart,
}

#[derive(Debug, thiserror::Error)]
pub enum WalError {
    #[error("wal disk full: {0}")]
    DiskFull(String),
    #[error("wal io: {0}")]
    Io(String),
    /// The log was quarantined (`JDB-3001`, spec 5.5) and this record never reached the file: it
    /// lies beyond everything that was handed over before the quarantine. Definitely not durable.
    #[error("JDB-3001 wal quarantined: {0}")]
    Quarantined(String),
    /// Some of the bytes this caller waited for may have reached the file before the failure, and
    /// may survive a restart although never synced. A commit in flight must be reported with an
    /// unknown outcome, never as aborted (answer A10); recovery settles it (spec 9.9).
    #[error("outcome unknown: {0}")]
    OutcomeUnknown(String),
    /// A record whose size depends on user data would exceed the limit (spec 8.1). SQLSTATE 54000.
    #[error("record of {len} bytes exceeds the limit of {max}")]
    RecordTooLarge { len: usize, max: usize },
    /// The WAL writer thread died of a panic. The log may be in an unknown state: a clean shutdown
    /// must not be declared on top of it.
    #[error("the WAL writer thread panicked")]
    WriterPanicked,
}

/// The published redo point, the half of the hint/checkpoint protocol that lives in the WAL (spec
/// 7.6, 9.2). A checkpoint publishes its new redo point and then scans the frames for the dirty
/// flag; a page change sets the flag and then reads the point. Both operations are `SeqCst` on
/// purpose: each side stores before it loads, and only `SeqCst` guarantees that at least one of them
/// sees the other. The loom model of the dirty flag against the redo point (`jamsil-engine`,
/// `tests/loom_dirty_redo.rs`) runs exactly these functions together with
/// `jamsil_bufmgr::dirty_flag`.
pub mod redo_point {
    use jamsil_page::Lsn;
    use jamsil_vfs::sync::atomic::{AtomicU64, STORE_BEFORE_LOAD};

    /// Publishes the new redo point (the store of the protocol).
    pub fn publish(point: &AtomicU64, lsn: Lsn) {
        point.store(lsn, STORE_BEFORE_LOAD);
        // See `jamsil_bufmgr::dirty_flag::set`: loom models a `SeqCst` fence, not a `SeqCst` access.
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
    }

    /// Reads the current redo point (the load of the protocol).
    pub fn read(point: &AtomicU64) -> Lsn {
        point.load(STORE_BEFORE_LOAD)
    }
}

/// Largest record the log accepts: 1 MiB and a quarter of a segment, whichever is smaller (8.1).
pub fn max_record_len(segment_size: u64) -> u64 {
    (MAX_RECORD_LEN as u64).min(segment_size / 4)
}

impl From<IoError> for WalError {
    fn from(e: IoError) -> Self {
        if e.kind == IoErrorKind::DiskFull {
            WalError::DiskFull(e.msg)
        } else {
            WalError::Io(e.msg)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LsnRange {
    pub start: Lsn,
    pub end: Lsn,
}

#[derive(Clone)]
pub struct WalConfig {
    pub segment_size: u64,
    pub system_id: u64,
    /// Size of the WAL buffer: the most that may be inserted and not yet handed to the file. An
    /// inserter that would pass it waits for a flush (spec 8.4). The writer wakes when the buffer
    /// is a quarter full (spec 8.5).
    pub buffer_bytes: u64,
}

impl WalConfig {
    pub fn new(system_id: u64) -> Self {
        Self { segment_size: WAL_SEGMENT_SIZE, system_id, buffer_bytes: DEFAULT_WAL_BUFFER }
    }
}

/// Where the log continues.
#[derive(Debug, Clone, Copy)]
pub enum StartPos {
    /// Brand-new log.
    Fresh,
    /// After recovery: `end` is the end of the valid log, `last` the start of the last record.
    At { end: Lsn, last: Lsn },
}

type Fatal = dyn Fn(&str) + Send + Sync;

struct Inner {
    reserved: Lsn,
    last_start: Lsn,
    /// The bytes of `[buf_start, reserved)` not yet handed to the file, except the gaps. The buffer
    /// starts at the position the writer has reached, so it maps onto the stream front to back.
    buf: Vec<u8>,
    buf_start: Lsn,
    /// Stream ranges inside `[buf_start, reserved)` that are not in `buf`: the bytes between a
    /// `SWITCH` and the end of its segment. They are never written (spec 8.1): the segment was
    /// zero-filled when it was created and is never reused, so the gap is zero already, and the
    /// written position jumps over it. Sorted, and every range lies inside one segment.
    gaps: Vec<(Lsn, Lsn)>,
    redo_lsn: Lsn,
    marker_end: Lsn,
}

pub struct Wal {
    vfs: Arc<dyn Vfs>,
    dir: PathBuf,
    cfg: WalConfig,
    inner: Mutex<Inner>,
    /// The published redo point. Written with `SeqCst` before a checkpoint scans the frames, read
    /// with `SeqCst` by `mark_dirty_hint` after it sets the dirty flag: both sides store before they
    /// load, so at least one sees the other (spec 7.6, answer A5).
    redo_pub: AtomicU64,
    flush_lock: Mutex<FlushState>,
    durable: AtomicU64,
    written: AtomicU64,
    fatal: Mutex<Arc<Fatal>>,
    /// Set when a fatal failure was reported: nothing may be flushed afterwards.
    poisoned: AtomicBool,
    /// Set when the log could not be extended (spec 5.5).
    /// Message and the end of the flush that hit the failure: bytes below it may have been handed
    /// to the file.
    quarantined: Mutex<Option<(String, Lsn, QuarantineKind)>>,
    /// A spare could not be created (spec 5.5, answer A10). New write statements must fail with
    /// `53100`; records already admitted go on into the remaining spares.
    space_low: AtomicBool,
    writer: Arc<WriterCtl>,
    writer_thread: Mutex<Option<thread::JoinHandle>>,
}

/// The answer of `try_place` when the buffer has no room.
#[derive(Debug)]
struct Full {
    /// The insert position to flush to.
    upto: Lsn,
    /// The record is larger than the whole buffer, so waiting for room cannot help.
    oversized: bool,
}

/// Shared with the writer thread, which only holds a weak reference to the log itself.
struct WriterCtl {
    stop: AtomicBool,
    kick: Mutex<bool>,
    cv: Condvar,
}

impl WriterCtl {
    fn wake(&self) {
        *self.kick.lock() = true;
        self.cv.notify_all();
    }
}

struct FlushState {
    files: HashMap<u64, Arc<dyn VfsFile>>,
    /// Highest segment known to exist on disk.
    created_upto: u64,
}

pub fn segment_path(dir: &Path, segno: u64) -> PathBuf {
    dir.join(format!("{segno:016x}.wal"))
}

pub fn parse_segment_name(name: &str) -> Option<u64> {
    let stem = name.strip_suffix(".wal")?;
    if stem.len() != 16 {
        return None;
    }
    u64::from_str_radix(stem, 16).ok()
}

impl Wal {
    pub fn open(vfs: Arc<dyn Vfs>, dir: &Path, cfg: WalConfig, start: StartPos) -> Result<Arc<Self>, WalError> {
        assert!(cfg.segment_size >= 4096 && cfg.segment_size % 8 == 0);
        vfs.create_dir_all(dir)?;
        let seg = cfg.segment_size;
        let (reserved, last_start, buf_start) = match start {
            StartPos::Fresh => (seg, 0, seg),
            StartPos::At { end, last } => (end, last, end),
        };
        let buffer_room = (cfg.buffer_bytes + max_record_len(cfg.segment_size) + 2 * SEG_HEADER_LEN as u64) as usize;
        let wal = Arc::new(Wal {
            vfs,
            dir: dir.to_path_buf(),
            cfg,
            inner: Mutex::new(Inner {
                reserved,
                last_start,
                // The buffer is fixed (spec 8.4): room for its capacity and for one more record, which
                // is all that a record placed with room, or into an empty buffer, can add. Nothing in
                // an insert allocates. Only inserts that go past the capacity by design (after a
                // failed flush) can make it grow.
                buf: Vec::with_capacity(buffer_room),
                gaps: Vec::with_capacity(8),
                buf_start,
                redo_lsn: 0,
                marker_end: 0,
            }),
            redo_pub: AtomicU64::new(0),
            flush_lock: Mutex::new(FlushState { files: HashMap::new(), created_upto: 0 }),
            durable: AtomicU64::new(reserved),
            written: AtomicU64::new(reserved),
            fatal: Mutex::new(Arc::new(|msg: &str| {
                eprintln!("fatal WAL failure: {msg}");
                std::process::exit(75);
            })),
            poisoned: AtomicBool::new(false),
            quarantined: Mutex::new(None),
            space_low: AtomicBool::new(false),
            writer: Arc::new(WriterCtl { stop: AtomicBool::new(false), kick: Mutex::new(false), cv: Condvar::new() }),
            writer_thread: Mutex::new(None),
        });
        {
            let mut inner = wal.inner.lock();
            if inner.reserved % seg == 0 {
                // At a segment boundary (fresh log, or a log that ended exactly at a segment end).
                wal.enter_segment(&mut inner);
            }
        }
        Ok(wal)
    }

    /// Replaces the fatal-failure policy (default: exit code 75). The handler should not return;
    /// if it does, the calling thread panics.
    pub fn set_fatal_handler(&self, f: Arc<Fatal>) {
        *self.fatal.lock() = f;
    }

    fn fatal(&self, msg: &str) -> ! {
        self.poisoned.store(true, Ordering::SeqCst);
        let h = self.fatal.lock().clone();
        h(msg);
        panic!("fatal WAL failure: {msg}");
    }

    pub fn segment_size(&self) -> u64 {
        self.cfg.segment_size
    }
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    pub fn system_id(&self) -> u64 {
        self.cfg.system_id
    }

    /// Starts a new segment in the stream: appends the 32-byte segment header.
    fn enter_segment(&self, inner: &mut Inner) {
        let segno = inner.reserved / self.cfg.segment_size;
        let h = encode_seg_header(self.cfg.system_id, segno);
        inner.buf.extend_from_slice(&h);
        inner.reserved += SEG_HEADER_LEN as u64;
    }

    pub fn insert_lsn(&self) -> Lsn {
        self.inner.lock().reserved
    }
    pub fn durable_lsn(&self) -> Lsn {
        self.durable.load(Ordering::SeqCst)
    }
    pub fn written_lsn(&self) -> Lsn {
        self.written.load(Ordering::SeqCst)
    }
    pub fn redo_lsn(&self) -> Lsn {
        redo_point::read(&self.redo_pub)
    }
    pub fn last_record_start(&self) -> Lsn {
        self.inner.lock().last_start
    }

    /// Checkpoint step 1: the image threshold becomes the current insert position. Returns it.
    pub fn begin_checkpoint(&self) -> Lsn {
        let mut inner = self.inner.lock();
        inner.redo_lsn = inner.reserved;
        redo_point::publish(&self.redo_pub, inner.redo_lsn);
        inner.reserved
    }

    pub fn set_redo_lsn(&self, lsn: Lsn) {
        self.inner.lock().redo_lsn = lsn;
        redo_point::publish(&self.redo_pub, lsn);
    }

    /// Inserts one record. `build` receives the current image threshold (`redo_lsn`) and
    /// returns the record; it is called again if the threshold moves while inserting (spec 8.3).
    pub fn insert_with<'a>(&self, permit: WalPermit, build: impl Fn(Lsn) -> RecordBuilder<'a>) -> LsnRange {
        let mut redo = self.inner.lock().redo_lsn;
        let mut force = false;
        loop {
            let rec = build(redo);
            permit.assert_covers(&rec);
            let mut sealed = assemble(&rec);
            let mut inner = self.inner.lock();
            if inner.redo_lsn != redo {
                redo = inner.redo_lsn;
                continue;
            }
            match self.try_place(&mut inner, &mut sealed, force) {
                Ok(range) => return range,
                Err(full) => {
                    drop(inner);
                    force = self.make_room(full);
                }
            }
        }
    }

    /// Inserts a record, waiting for a flush first if the WAL buffer has no room for it (spec 8.4).
    /// The permit comes from `check_size`, taken before the critical section, and is consumed
    /// (spec 8.8). The insert itself cannot fail: if the flush it waits for fails (a full disk, a
    /// quarantine), the record is placed anyway and the failure surfaces at the callers that flush.
    pub fn insert(&self, permit: WalPermit, rec: RecordBuilder<'_>) -> LsnRange {
        permit.assert_covers(&rec);
        let mut sealed = assemble(&rec);
        let mut force = false;
        loop {
            let mut inner = self.inner.lock();
            match self.try_place(&mut inner, &mut sealed, force) {
                Ok(range) => return range,
                Err(full) => {
                    drop(inner);
                    force = self.make_room(full);
                }
            }
        }
    }

    /// An insert that never waits for buffer space and needs no permit. For the records the log
    /// writes itself: the durability marker, which `flush` inserts while it holds the flush lock
    /// (waiting for a flush there would wait for itself), and the `SWITCH`. They are a few bytes, so
    /// the buffer overshoots by that much.
    fn insert_forced(&self, parts: &RecordBuilder<'_>) -> LsnRange {
        let mut sealed = assemble(parts);
        let mut inner = self.inner.lock();
        self.try_place(&mut inner, &mut sealed, true).expect("a forced insert always places")
    }

    /// Waits for the buffer to drain: flushes up to the insert position at which the buffer was
    /// found full. A waiter may do the write and the sync itself (spec 8.5). Returns whether the
    /// caller should now place its record whether or not there is room: because the flush failed
    /// and no later one is in sight, or because the record could never fit the buffer, and every
    /// flush leaves a durability marker behind, so the buffer is never empty to wait for.
    fn make_room(&self, full: Full) -> bool {
        self.flush(full.upto).is_err() || full.oversized
    }

    /// How many bytes placing a record of `total` bytes adds to the buffer: the record, a segment
    /// header if it starts a segment, and, if it does not fit in the current one, the `SWITCH`
    /// record and the next segment's header. The gap after a `SWITCH` is not in the buffer.
    fn room_needed(&self, inner: &Inner, total: u64) -> u64 {
        let seg = self.cfg.segment_size;
        let hdr = SEG_HEADER_LEN as u64;
        let at = inner.reserved;
        let start = if at % seg == 0 { at + hdr } else { at };
        let seg_end = (start / seg + 1) * seg;
        let left_after = seg_end.saturating_sub(start + total);
        let fits = start + total <= seg_end;
        let switch = !fits || (left_after != 0 && left_after < REC_HEADER_LEN as u64);
        if switch {
            (start - at) + REC_HEADER_LEN as u64 + hdr + total
        } else {
            (start - at) + total
        }
    }

    /// Places a sealed record if the buffer has room: it is empty, or the record fits below its
    /// capacity. `force` skips the check. `Err` says no room and what to flush.
    fn try_place(&self, inner: &mut Inner, sealed: &mut SealedPayload, force: bool) -> Result<LsnRange, Full> {
        if !force && !inner.buf.is_empty() {
            let need = self.room_needed(inner, sealed.total_len() as u64);
            if inner.buf.len() as u64 + need > self.cfg.buffer_bytes {
                return Err(Full { upto: inner.reserved, oversized: need > self.cfg.buffer_bytes });
            }
        }
        Ok(self.place(inner, sealed))
    }

    fn place(&self, inner: &mut Inner, sealed: &mut SealedPayload) -> LsnRange {
        let seg = self.cfg.segment_size;
        let total = sealed.total_len() as u64;
        assert!(total <= max_record_len(seg), "record of {total} bytes exceeds the limit of {}", max_record_len(seg));
        if inner.reserved % seg == 0 {
            self.enter_segment(inner);
        }
        let seg_end = (inner.reserved / seg + 1) * seg;
        let left_after = seg_end.saturating_sub(inner.reserved + total);
        let fits = inner.reserved + total <= seg_end;
        if !fits || (left_after != 0 && left_after < REC_HEADER_LEN as u64) {
            // Pad with a SWITCH record and continue in the next segment.
            let mut sw = assemble(&RecordBuilder::new(rmgr::XLOG, xlog::SWITCH));
            let start = inner.reserved;
            sw.seal(start, inner.last_start);
            inner.buf.extend_from_slice(&sw.bytes);
            inner.last_start = start;
            // The rest of the segment is a gap: it is already zero, and nobody writes it (spec 8.1).
            let sw_end = start + sw.bytes.len() as u64;
            if sw_end < seg_end {
                inner.gaps.push((sw_end, seg_end));
            }
            inner.reserved = seg_end;
            self.enter_segment(inner);
        }
        let start = inner.reserved;
        sealed.seal(start, inner.last_start);
        inner.buf.extend_from_slice(&sealed.bytes);
        inner.last_start = start;
        inner.reserved = start + total;
        if inner.buf.len() as u64 >= self.cfg.buffer_bytes / 4 {
            self.writer.wake();
        }
        LsnRange { start, end: inner.reserved }
    }

    /// Bytes inserted and not yet handed to the file. Bounded by `buffer_bytes` except for one
    /// record that is larger than the buffer, the durability markers, and inserts that went on
    /// after a failed flush.
    pub fn buffered_bytes(&self) -> u64 {
        self.inner.lock().buf.len() as u64
    }

    /// Opens a segment, creating it if needed. A new segment is created under its final name at
    /// full size, zero-filled except for its header, and synced together with the directory before
    /// first use (spec 5.3 rule 5, 8.1). This is also what a spare looks like.
    fn segment_file(&self, fs: &mut FlushState, segno: u64) -> Result<Arc<dyn VfsFile>, WalError> {
        if let Some(f) = fs.files.get(&segno) {
            return Ok(f.clone());
        }
        let path = segment_path(&self.dir, segno);
        let f: Arc<dyn VfsFile> = if self.vfs.exists(&path) {
            Arc::from(self.vfs.open(&path, OpenOpts::read_write())?)
        } else {
            let f: Arc<dyn VfsFile> = Arc::from(self.vfs.open(&path, OpenOpts::create())?);
            let made = f
                .allocate(0, self.cfg.segment_size)
                .and_then(|_| f.write_at(&encode_seg_header(self.cfg.system_id, segno), 0));
            if let Err(e) = made {
                // Do not leave a half-made segment under its final name.
                let _ = self.vfs.unlink(&path);
                return Err(e.into());
            }
            if let Err(e) = f.sync_data().and_then(|_| self.vfs.sync_dir(&self.dir)) {
                self.fatal(&format!("sync of new WAL segment failed: {e}"));
            }
            f
        };
        fs.files.insert(segno, f.clone());
        fs.created_upto = fs.created_upto.max(segno);
        Ok(f)
    }

    /// Blocks until `durable_lsn >= upto`. A failed sync ends the process (exit code 75).
    pub fn flush(&self, upto: Lsn) -> Result<(), WalError> {
        if self.durable_lsn() >= upto {
            return Ok(());
        }
        let mut fs = self.flush_lock.lock();
        if self.durable_lsn() >= upto {
            return Ok(());
        }
        self.flush_locked(&mut fs, upto)
    }

    /// Writes and syncs everything inserted so far.
    pub fn flush_all(&self) -> Result<(), WalError> {
        let upto = self.insert_lsn();
        let mut fs = self.flush_lock.lock();
        self.flush_locked(&mut fs, upto)
    }

    /// One pass of the WAL writer: flush whatever is unflushed. The writer thread's loop and the
    /// deterministic tests both run exactly this, so a simulated tick takes the same path as the
    /// real thread (answer A9). A full disk is retried on the next pass.
    pub fn writer_tick(&self) {
        if self.has_unflushed() {
            let _ = self.flush_all();
        }
    }

    /// The gate before every critical section (spec 8.8). Inside a critical section an insert cannot
    /// fail, so everything that can be known beforehand is checked here:
    /// - the log accepts records: a quarantined log fails with the quarantine error, so that no
    ///   critical section starts only to add records that never reach the file. This is what bounds
    ///   the overshoot of the buffer after a failed flush (spec 8.4) to one critical section per
    ///   thread;
    /// - the record can fit: it fails with `54000` if even the worst case, a full image for each
    ///   block it references, exceeds the record limit, or if a block payload is beyond the 16-bit
    ///   length field (`len` and `max` then name the payload);
    /// - the insert inside will not allocate: the thread's assembly buffer is made large enough.
    pub fn check_size(&self, rec: &RecordBuilder<'_>) -> Result<WalPermit, WalError> {
        if let Some((why, _, _)) = self.quarantined.lock().clone() {
            return Err(WalError::Quarantined(why));
        }
        let (len, max) = (rec.worst_case_len(), max_record_len(self.cfg.segment_size) as usize);
        if len > max {
            return Err(WalError::RecordTooLarge { len, max });
        }
        if let Some(big) = rec.blocks().map(|b| b.data.len()).find(|n| *n > MAX_BLOCK_DATA) {
            return Err(WalError::RecordTooLarge { len: big, max: MAX_BLOCK_DATA });
        }
        reserve_assembly_buffer(len);
        Ok(WalPermit::issue(len))
    }

    /// True while a spare segment could not be created. Cleared when both spares exist again.
    pub fn space_low(&self) -> bool {
        self.space_low.load(Ordering::SeqCst)
    }

    /// True if there are bytes other than the trailing durability marker that are not durable.
    pub fn has_unflushed(&self) -> bool {
        let inner = self.inner.lock();
        inner.reserved > self.durable_lsn() && inner.reserved > inner.marker_end
    }

    fn flush_locked(&self, fs: &mut FlushState, upto: Lsn) -> Result<(), WalError> {
        if self.poisoned.load(Ordering::SeqCst) {
            self.fatal("WAL used after a fatal failure");
        }
        if let Some((why, handed_to, _)) = self.quarantined.lock().clone() {
            // What lies below `handed_to` may have reached the file before the quarantine.
            return Err(if upto <= handed_to { WalError::OutcomeUnknown(why) } else { WalError::Quarantined(why) });
        }
        let seg = self.cfg.segment_size;
        let (data, gaps, from, to) = {
            let inner = self.inner.lock();
            let from = self.written.load(Ordering::SeqCst);
            let to = inner.reserved;
            if to == from {
                return Ok(());
            }
            // The buffer starts at the write position: it is drained only after a successful write.
            debug_assert_eq!(from, inner.buf_start);
            (inner.buf.clone(), inner.gaps.clone(), from, to)
        };
        let gap_bytes: u64 = gaps.iter().map(|g| g.1 - g.0).sum();
        debug_assert_eq!(data.len() as u64 + gap_bytes, to - from);
        let mut pos = from;
        let mut idx = 0usize; // into `data`, which holds the stream without its gaps
        let mut cur_seg: Option<(u64, Arc<dyn VfsFile>)> = None;
        while pos < to {
            // The written position jumps over a gap (spec 8.1).
            if let Some(&(_, gap_end)) = gaps.iter().find(|g| g.0 == pos) {
                pos = gap_end;
                continue;
            }
            let segno = pos / seg;
            let seg_end = (segno + 1) * seg;
            let next_gap = gaps.iter().map(|g| g.0).filter(|g| *g > pos).min().unwrap_or(u64::MAX);
            let n = (to.min(seg_end).min(next_gap) - pos) as usize;
            let file = match &cur_seg {
                Some((s, f)) if *s == segno => f.clone(),
                _ => {
                    if let Some((_, prev)) = cur_seg.take() {
                        // A segment is synced before the first byte of the next one is written.
                        if let Err(e) = prev.sync_data() {
                            self.fatal(&format!("fdatasync failed: {e}"));
                        }
                    }
                    let f = match self.segment_file(fs, segno) {
                        Ok(f) => f,
                        Err(e) => return Err(self.quarantine_on_error(e, to)),
                    };
                    cur_seg = Some((segno, f.clone()));
                    f
                }
            };
            let seg_off = pos - segno * seg;
            if let Err(e) = file.write_at(&data[idx..idx + n], seg_off) {
                if e.kind == IoErrorKind::DiskFull {
                    // A write into a segment that exists does not need new space on the supported
                    // file systems (ext4, XFS), so this means the environment is broken. Retrying at
                    // the next flush would leave a node that fails every commit and says nothing: it
                    // quarantines like a missing segment (spec 5.5). Part of this batch may be in
                    // the file already, so the outcome of its records stays unknown.
                    return Err(self.quarantine_on_error(WalError::DiskFull(e.msg), to));
                }
                self.fatal(&format!("WAL write failed: {e}"));
            }
            pos += n as u64;
            idx += n;
        }
        debug_assert_eq!(idx, data.len());
        self.written.store(to, Ordering::SeqCst);
        if let Some((_, f)) = &cur_seg {
            if let Err(e) = f.sync_data() {
                self.fatal(&format!("fdatasync failed: {e}"));
            }
        }
        // Keep two spares ahead of the write position. A failure here is not yet a failure of the
        // log: it raises the space-low flag and the records already admitted go on into the spares
        // that exist. The quarantine starts only when a record needs a segment and none exists.
        let current = (to.saturating_sub(1)) / seg;
        let mut all_spares = true;
        for next in current + 1..=current + SPARES {
            if self.segment_file(fs, next).is_err() {
                all_spares = false;
                break;
            }
        }
        self.space_low.store(!all_spares, Ordering::SeqCst);
        self.durable.store(to, Ordering::SeqCst);
        {
            let mut inner = self.inner.lock();
            inner.buf.drain(..data.len());
            inner.gaps.retain(|g| g.1 > to);
            inner.buf_start = to;
        }
        // Durability marker (8.5): evidence for recovery that bytes below `to` were synced.
        let r = self.insert_forced(&RecordBuilder::new(rmgr::XLOG, xlog::DURABILITY_MARKER).main(&to.to_le_bytes()));
        self.inner.lock().marker_end = r.end;
        Ok(())
    }

    /// A needed segment that cannot be created quarantines the log (spec 5.5). `handed_to` is the
    /// end of the batch being flushed: its bytes may be in the file already, so the caller's
    /// outcome is unknown, not "failed".
    fn quarantine_on_error(&self, e: WalError, handed_to: Lsn) -> WalError {
        match e {
            WalError::DiskFull(m) => {
                // A full disk is cleared by a restart (spec 17.2).
                *self.quarantined.lock() = Some((m.clone(), handed_to, QuarantineKind::Restart));
                self.space_low.store(true, Ordering::SeqCst);
                WalError::OutcomeUnknown(m)
            }
            other => other,
        }
    }

    /// Quarantines the log (spec 5.5, 17.2). The kind says what clears it. After a panic of the writer
    /// thread the log may have been in the middle of a flush, so everything inserted so far may have
    /// reached the file: flushes of those positions report an unknown outcome, and later ones fail
    /// definitely. The first quarantine stays: a later one does not replace its reason.
    pub fn quarantine(&self, kind: QuarantineKind, msg: &str) {
        let handed_to = self.inner.lock().reserved;
        let mut q = self.quarantined.lock();
        if q.is_none() {
            *q = Some((msg.to_string(), handed_to, kind));
        }
    }

    /// Why the log is quarantined and what clears it, for the status (spec 17.2), or `None`.
    pub fn quarantine_reason(&self) -> Option<(QuarantineKind, String)> {
        self.quarantined.lock().as_ref().map(|(why, _, kind)| (*kind, why.clone()))
    }

    /// True once the log refuses all further flushes.
    pub fn is_quarantined(&self) -> bool {
        self.quarantined.lock().is_some()
    }

    /// Starts the WAL writer thread (spec 4.2, 8.5). It runs in every configuration: it flushes
    /// every `interval` and whenever the buffer passes a quarter of its size, which is what bounds
    /// the loss window under NORMAL and drains a log nobody waits on. `on_panic` receives the
    /// message if the thread dies (a failed sync under a panicking test handler, for instance).
    pub fn start_writer(self: &Arc<Self>, interval: Duration, on_panic: impl FnOnce(String) + Send + 'static) {
        let mut slot = self.writer_thread.lock();
        if slot.is_some() {
            return;
        }
        let ctl = self.writer.clone();
        let weak: Weak<Wal> = Arc::downgrade(self);
        // A panic in the writer quarantines the log (spec 18.5) before the caller's handler hears of
        // it: without the writer the bound of NORMAL is gone, and the state it left is unknown.
        let on_panic = {
            let weak = weak.clone();
            move |msg: String| {
                if let Some(w) = weak.upgrade() {
                    w.quarantine(QuarantineKind::Restart, &msg);
                }
                on_panic(msg);
            }
        };
        let handle = thread::spawn_named("jamsil-wal-writer", on_panic, move || loop {
            {
                let mut kicked = ctl.kick.lock();
                if !*kicked && !ctl.stop.load(Ordering::SeqCst) {
                    ctl.cv.wait_for(&mut kicked, interval);
                }
                *kicked = false;
            }
            if ctl.stop.load(Ordering::SeqCst) {
                return;
            }
            let Some(w) = weak.upgrade() else { return };
            w.writer_tick();
        })
        .expect("spawn wal writer");
        *slot = Some(handle);
    }

    /// Stops the writer thread and waits for it, without flushing. After this returns no thread of
    /// this log touches the files, which is what a simulated crash needs. `Err(WriterPanicked)` if
    /// the thread had died of a panic, so a shutdown can tell a clean stop from a failed one.
    pub fn stop(&self) -> Result<(), WalError> {
        self.writer.stop.store(true, Ordering::SeqCst);
        self.writer.wake();
        let h = self.writer_thread.lock().take();
        if let Some(h) = h {
            // The last reference to the log may be dropped by the writer thread itself.
            if h.id() != thread::current_id() {
                return h.join().map_err(|_| WalError::WriterPanicked);
            }
        }
        Ok(())
    }

    /// True once a fatal failure was reported through the handler (the process normally exits then).
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
    }

    pub fn reader(&self, from: Lsn) -> WalReader {
        WalReader::new(self.vfs.clone(), self.dir.clone(), self.cfg.clone(), from)
    }

    /// Unlinks segments wholly below `retain` (spec 8.10). Never renames for reuse.
    pub fn remove_segments_below(&self, retain: Lsn) -> IoResult<usize> {
        let keep_from = retain / self.cfg.segment_size;
        let mut fs = self.flush_lock.lock();
        let mut n = 0;
        for e in self.vfs.list(&self.dir)? {
            if let Some(s) = parse_segment_name(&e.name) {
                if s < keep_from {
                    fs.files.remove(&s);
                    self.vfs.unlink(&self.dir.join(&e.name))?;
                    n += 1;
                }
            }
        }
        if n > 0 {
            self.vfs.sync_dir(&self.dir)?;
        }
        Ok(n)
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        // Nobody is left to tell: whoever cared called `stop` and saw the result.
        let _ = self.stop();
    }
}

/// True if the segment file is a spare: full length, a header valid for its number, and zeros
/// everywhere else (spec 8.9 rule 4, answer A6).
fn is_spare(vfs: &dyn Vfs, path: &Path, cfg: &WalConfig, segno: u64) -> IoResult<bool> {
    let f = vfs.open(path, OpenOpts::read_only())?;
    if f.len()? != cfg.segment_size {
        return Ok(false);
    }
    let mut head = [0u8; SEG_HEADER_LEN];
    if f.read_at(&mut head, 0)? != SEG_HEADER_LEN || check_seg_header(&head, cfg.system_id, segno).is_err() {
        return Ok(false);
    }
    let mut chunk = vec![0u8; 1 << 20];
    let mut off = SEG_HEADER_LEN as u64;
    while off < cfg.segment_size {
        let want = ((cfg.segment_size - off) as usize).min(chunk.len());
        let n = f.read_at(&mut chunk[..want], off)?;
        if n == 0 || chunk[..n].iter().any(|&b| b != 0) {
            return Ok(false);
        }
        off += n as u64;
    }
    Ok(true)
}

/// Recovery step 6 (spec 9.3, 8.9 rule 4, answer A6): zero everything from `end` to the end of its
/// segment and sync. Later segments that are spares (a valid header and nothing else) stay; every
/// other later segment is removed. Non-zero discarded bytes go to a side file.
pub fn truncate_tail(vfs: &dyn Vfs, dir: &Path, cfg: &WalConfig, end: Lsn) -> IoResult<()> {
    let seg = cfg.segment_size;
    let segno = end / seg;
    let off = end - segno * seg;
    let path = segment_path(dir, segno);
    if vfs.exists(&path) && off != 0 {
        let f = vfs.open(&path, OpenOpts::read_write())?;
        let mut tail = vec![0u8; (seg - off) as usize];
        let n = f.read_at(&mut tail, off)?;
        tail.truncate(n);
        if tail.iter().any(|&b| b != 0) {
            let side = dir.join(format!("{segno:016x}.discarded"));
            let sf = vfs.open(&side, OpenOpts::create())?;
            sf.truncate(0)?;
            sf.write_at(&tail, 0)?;
            sf.sync_data()?;
            f.write_at(&vec![0u8; tail.len()], off)?;
        }
        f.sync_data()?;
    }
    let mut removed = false;
    for e in vfs.list(dir)? {
        if let Some(s) = parse_segment_name(&e.name) {
            // `end` itself at a boundary: the segment that starts there holds nothing valid either.
            if s > segno || (s == segno && off == 0) {
                let p = dir.join(&e.name);
                if !is_spare(vfs, &p, cfg, s)? {
                    vfs.unlink(&p)?;
                    removed = true;
                }
            }
        }
    }
    if removed {
        vfs.sync_dir(dir)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_limited_to_1_mib_and_a_quarter_of_a_segment() {
        assert_eq!(max_record_len(64 << 20), 1 << 20);
        assert_eq!(max_record_len(1 << 20), 256 * 1024);
        assert_eq!(max_record_len(256 * 1024), 64 * 1024);
    }
}
