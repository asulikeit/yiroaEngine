//! Relation file manager: (RelFileId, ForkId, BlockNo) -> file and offset (spec 6.7).
//!
//! Two rules from the answers to Question.md:
//! - File geometry is a table keyed by `RelFileId` (A12). Ids 1 and 2 are the commit log and the
//!   CSN log, in `xact/` with 256 pages per segment; every other id lives in `base/` with
//!   131,072 pages per segment.
//! - A descriptor with writes that no successful `sync_data` has covered is never closed (B7,
//!   spec 5.3 rule 3). The cache evicts clean descriptors only and grows past its bound when none
//!   is clean, so a write-back error cannot go unreported by closing the file.

use crate::BufError;
use jamsil_page::*;
use jamsil_vfs::sync::atomic::{AtomicU64, Ordering};
use jamsil_vfs::sync::{Arc, Mutex};
use jamsil_vfs::{IoResult, OpenOpts, Vfs, VfsFile};
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// 1 GiB segments for ordinary relations.
pub const PAGES_PER_SEGMENT: u32 = 131_072;
/// 2 MiB segments for the commit log and the CSN log, so old segments can be unlinked.
pub const LOG_PAGES_PER_SEGMENT: u32 = 256;
const MAX_OPEN_FILES: usize = 512;

type FileKey = (RelFileId, ForkId, u32);

/// How a relation's forks map to files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub pages_per_segment: u32,
    /// Subdirectory of the generation directory.
    pub dir: &'static str,
    /// `Some(stem)`: a single-fork log named `<stem>.<segment:06>`. `None`: the ordinary
    /// `<relfile:016x>.<fork>.<segment>` name.
    pub log_stem: Option<&'static str>,
}

impl Geometry {
    pub const RELATION: Geometry = Geometry { pages_per_segment: PAGES_PER_SEGMENT, dir: "base", log_stem: None };
    pub const CLOG: Geometry =
        Geometry { pages_per_segment: LOG_PAGES_PER_SEGMENT, dir: "xact", log_stem: Some("clog") };
    pub const CSNLOG: Geometry =
        Geometry { pages_per_segment: LOG_PAGES_PER_SEGMENT, dir: "xact", log_stem: Some("csnlog") };
}

/// An open file plus the bookkeeping B7 needs. A write is counted when the descriptor is handed
/// out for writing (under the cache lock) and again when it finishes; a sync covers exactly the
/// writes that had finished before it started.
struct Desc {
    file: Arc<dyn VfsFile>,
    started: AtomicU64,
    finished: AtomicU64,
    synced: AtomicU64,
}

impl Desc {
    fn new(file: Arc<dyn VfsFile>) -> Arc<Self> {
        Arc::new(Self { file, started: AtomicU64::new(0), finished: AtomicU64::new(0), synced: AtomicU64::new(0) })
    }
    fn is_clean(&self) -> bool {
        self.started.load(Ordering::SeqCst) == self.synced.load(Ordering::SeqCst)
    }
}

pub struct Smgr {
    vfs: Arc<dyn Vfs>,
    gen_dir: PathBuf,
    files: Mutex<HashMap<FileKey, Arc<Desc>>>,
    max_open: usize,
    geometry: Mutex<HashMap<RelFileId, Geometry>>,
    /// Logical fork lengths. May run ahead of the files (spec 7.8).
    lens: Mutex<HashMap<(RelFileId, ForkId), u32>>,
    /// Writes at or beyond the known length that were not preceded by an extension. Always zero in
    /// a correct engine; a release build raises the length and counts the violation here.
    length_raised_by_write: AtomicU64,
    sync_queue: Mutex<BTreeSet<FileKey>>,
    dirty_dirs: Mutex<BTreeSet<&'static str>>,
}

impl Smgr {
    /// `gen_dir` is the generation directory; relation files live in its `base/` and the commit
    /// logs in its `xact/`.
    pub fn new(vfs: Arc<dyn Vfs>, gen_dir: &Path) -> IoResult<Self> {
        Self::with_max_open(vfs, gen_dir, MAX_OPEN_FILES)
    }

    pub fn with_max_open(vfs: Arc<dyn Vfs>, gen_dir: &Path, max_open: usize) -> IoResult<Self> {
        for d in [Geometry::RELATION.dir, Geometry::CLOG.dir] {
            vfs.create_dir_all(&gen_dir.join(d))?;
        }
        let mut geometry = HashMap::new();
        geometry.insert(CLOG_RELFILE, Geometry::CLOG);
        geometry.insert(CSNLOG_RELFILE, Geometry::CSNLOG);
        Ok(Self {
            vfs,
            gen_dir: gen_dir.to_path_buf(),
            files: Mutex::new(HashMap::new()),
            max_open: max_open.max(1),
            geometry: Mutex::new(geometry),
            lens: Mutex::new(HashMap::new()),
            length_raised_by_write: AtomicU64::new(0),
            sync_queue: Mutex::new(BTreeSet::new()),
            dirty_dirs: Mutex::new(BTreeSet::new()),
        })
    }

    pub fn vfs(&self) -> &Arc<dyn Vfs> {
        &self.vfs
    }

    pub fn geometry(&self, rel: RelFileId) -> Geometry {
        self.geometry.lock().get(&rel).copied().unwrap_or(Geometry::RELATION)
    }

    pub fn path(&self, rel: RelFileId, fork: ForkId, seg: u32) -> PathBuf {
        let g = self.geometry(rel);
        let name = match g.log_stem {
            Some(stem) => {
                debug_assert_eq!(fork, ForkId::Main, "a commit log has no forks");
                format!("{stem}.{seg:06}")
            }
            None => format!("{rel:016x}.{}.{seg}", fork.name()),
        };
        self.gen_dir.join(g.dir).join(name)
    }

    /// Number of descriptors currently cached (diagnostics and tests).
    pub fn open_descriptors(&self) -> usize {
        self.files.lock().len()
    }

    /// Returns the descriptor. For a write the pending-write count is raised under the cache lock,
    /// so eviction can never see the descriptor as clean while a write is on its way.
    fn desc(&self, key: FileKey, create: bool, for_write: bool) -> IoResult<Option<Arc<Desc>>> {
        {
            let files = self.files.lock();
            if let Some(d) = files.get(&key) {
                if for_write {
                    d.started.fetch_add(1, Ordering::SeqCst);
                }
                return Ok(Some(d.clone()));
            }
        }
        let path = self.path(key.0, key.1, key.2);
        let exists = self.vfs.exists(&path);
        if !exists && !create {
            return Ok(None);
        }
        let f: Arc<dyn VfsFile> =
            Arc::from(self.vfs.open(&path, if create { OpenOpts::create() } else { OpenOpts::read_write() })?);
        if !exists {
            self.dirty_dirs.lock().insert(self.geometry(key.0).dir);
        }
        let mut files = self.files.lock();
        // Someone may have opened it meanwhile: share that descriptor.
        let d = files.entry(key).or_insert_with(|| Desc::new(f)).clone();
        if for_write {
            d.started.fetch_add(1, Ordering::SeqCst);
        }
        if files.len() > self.max_open {
            // Evict clean descriptors only (B7). If none is clean the cache grows past its bound.
            let victims: Vec<FileKey> = files
                .iter()
                .filter(|(k, v)| **k != key && v.is_clean() && Arc::strong_count(v) == 1)
                .map(|(k, _)| *k)
                .take(files.len() - self.max_open)
                .collect();
            for k in victims {
                files.remove(&k);
            }
        }
        Ok(Some(d))
    }

    fn finish_write(d: &Desc) {
        d.finished.fetch_add(1, Ordering::SeqCst);
    }

    /// The fork's logical length. The in-memory value if there is one; otherwise it is derived from
    /// the files and remembered, the first time the fork is touched after a start and whatever call
    /// touches it first (spec 6.7). Every update of the table starts from this, never from zero:
    /// after a restart the files may be longer than what this process has written so far.
    fn known_len(&self, rel: RelFileId, fork: ForkId) -> IoResult<u32> {
        if let Some(n) = self.lens.lock().get(&(rel, fork)) {
            return Ok(*n);
        }
        let n = self.disk_nblocks(rel, fork)?;
        Ok(*self.lens.lock().entry((rel, fork)).or_insert(n))
    }

    pub fn create_fork(&self, rel: RelFileId, fork: ForkId) -> IoResult<()> {
        self.desc((rel, fork, 0), true, false)?;
        self.known_len(rel, fork)?;
        Ok(())
    }

    /// Carries out a file operation of an `SMGR` record again during redo, and queues the file and
    /// its directory for sync, also when the fork already exists (spec 9.3 "What redo may rely
    /// on"). A killed process may have created the file without syncing it or its directory, and
    /// this process has no record of that: the end-of-recovery checkpoint syncs everything queued
    /// here before the redo point moves.
    pub fn recreate_fork(&self, rel: RelFileId, fork: ForkId) -> IoResult<()> {
        self.create_fork(rel, fork)?;
        self.sync_queue.lock().insert((rel, fork, 0));
        self.dirty_dirs.lock().insert(self.geometry(rel).dir);
        Ok(())
    }

    /// True if the fork's first file exists. A length entry alone does not make a fork exist:
    /// redo raises the length of a page it restores before the file is there (see `ensure_len`).
    pub fn fork_exists(&self, rel: RelFileId, fork: ForkId) -> bool {
        self.vfs.exists(&self.path(rel, fork, 0))
    }

    /// Reads a page. `Ok(false)` if the block lies beyond the end of the file.
    pub fn read(&self, tag: PageTag, buf: &mut [u8]) -> Result<bool, BufError> {
        let pps = self.geometry(tag.rel).pages_per_segment;
        let seg = tag.block / pps;
        let off = (tag.block % pps) as u64 * PAGE_SIZE as u64;
        let Some(d) = self.desc((tag.rel, tag.fork, seg), false, false)? else { return Ok(false) };
        let mut done = 0;
        while done < PAGE_SIZE {
            let n = d.file.read_at(&mut buf[done..PAGE_SIZE], off + done as u64)?;
            if n == 0 {
                break;
            }
            done += n;
        }
        if done == 0 {
            return Ok(false);
        }
        if done < PAGE_SIZE {
            // A partial page can only come from a torn extension; the caller's checksum catches it.
            buf[done..PAGE_SIZE].fill(0);
        }
        Ok(true)
    }

    /// Writes a page, growing the file as needed. Adds the segment to the sync queue.
    ///
    /// No call may write a block at or beyond the known length of its fork, except extension
    /// (spec 6.7): extension and redo raise the length first, with `ensure_len`. A debug build
    /// asserts that. A release build raises the length instead, so no page is lost, but a caller
    /// that gets here has broken the rule.
    pub fn write(&self, tag: PageTag, buf: &[u8]) -> Result<(), BufError> {
        let known = self.known_len(tag.rel, tag.fork)?;
        if tag.block >= known {
            // Refusing the write would block the write-out of a legitimate page, so a release
            // build raises the length below. It does not do so unnoticed: the violation is
            // counted and reported as an error. A debug build stops here.
            self.length_raised_by_write.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "jamsil-bufmgr: error: write of block {} of {:?} at or beyond the known length {known}: \
                 only extension may raise it (spec 6.7)",
                tag.block,
                (tag.rel, tag.fork)
            );
            debug_assert!(
                false,
                "write of block {} of {:?} at or beyond the known length {known}: only extension may raise it",
                tag.block,
                (tag.rel, tag.fork)
            );
        }
        let pps = self.geometry(tag.rel).pages_per_segment;
        let seg = tag.block / pps;
        let off = (tag.block % pps) as u64 * PAGE_SIZE as u64;
        let key = (tag.rel, tag.fork, seg);
        let d = self.desc(key, true, true)?.expect("created");
        let r = d.file.write_at(&buf[..PAGE_SIZE], off);
        // Queue the file before the write is counted finished, so a sync that sees the finished
        // count also finds the key.
        self.sync_queue.lock().insert(key);
        Self::finish_write(&d);
        r?;
        let mut lens = self.lens.lock();
        let e = lens.entry((tag.rel, tag.fork)).or_insert(known);
        *e = (*e).max(tag.block + 1);
        Ok(())
    }

    /// How many writes broke rule 3 of spec 6.7 (see `write`). An engine that is correct keeps it 0.
    pub fn length_raised_by_write(&self) -> u64 {
        self.length_raised_by_write.load(Ordering::Relaxed)
    }

    /// Raises the known length of a fork to at least `len` pages and never lowers it. This is how
    /// the length grows: extension calls it before it dirties the new pages, and redo calls it
    /// before it applies an extend record or restores a page the files may not hold yet (a torn
    /// trailing page is not counted, see `disk_nblocks`).
    pub fn ensure_len(&self, rel: RelFileId, fork: ForkId, len: u32) -> IoResult<()> {
        let known = self.known_len(rel, fork)?;
        let mut lens = self.lens.lock();
        let e = lens.entry((rel, fork)).or_insert(known);
        *e = (*e).max(len);
        Ok(())
    }

    /// Length of a fork in pages: the in-memory value if known, else derived from the files.
    pub fn nblocks(&self, rel: RelFileId, fork: ForkId) -> IoResult<u32> {
        self.known_len(rel, fork)
    }

    /// Length derived from the files: the number of whole pages. A trailing partial page is a torn
    /// write or a torn extension. It is not counted, and redo formats it again (an extend record
    /// covers it) or restores it from an image (spec 6.7).
    pub fn disk_nblocks(&self, rel: RelFileId, fork: ForkId) -> IoResult<u32> {
        let g = self.geometry(rel);
        if g.log_stem.is_some() {
            // A commit log loses its front segments (spec 12.2, 14.6), so its length is derived from
            // its highest segment, not from the sum of its files.
            let Some(&top) = self.log_segments(rel)?.last() else { return Ok(0) };
            let pages = match self.desc((rel, fork, top), false, false)? {
                Some(d) => d.file.len()? / PAGE_SIZE as u64,
                None => 0,
            };
            return Ok((top as u64 * g.pages_per_segment as u64 + pages) as u32);
        }
        let pps = g.pages_per_segment as u64;
        let mut seg = 0u32;
        let mut total = 0u64;
        while let Some(d) = self.desc((rel, fork, seg), false, false)? {
            let pages = d.file.len()? / PAGE_SIZE as u64;
            total += pages;
            if pages < pps {
                break;
            }
            seg += 1;
        }
        Ok(total as u32)
    }

    pub fn set_nblocks(&self, rel: RelFileId, fork: ForkId, n: u32) {
        self.lens.lock().insert((rel, fork), n);
    }

    /// Claims disk space for `[from, from+n)` without changing file lengths.
    pub fn reserve(&self, rel: RelFileId, fork: ForkId, from: u32, n: u32) -> IoResult<()> {
        let pps = self.geometry(rel).pages_per_segment as u64;
        let mut b = from as u64;
        let end = from as u64 + n as u64;
        while b < end {
            let seg = (b / pps) as u32;
            let upto = end.min((seg as u64 + 1) * pps);
            let d = self.desc((rel, fork, seg), true, false)?.expect("created");
            d.file.reserve((b - seg as u64 * pps) * PAGE_SIZE as u64, (upto - b) * PAGE_SIZE as u64)?;
            b = upto;
        }
        Ok(())
    }

    pub fn truncate_fork(&self, rel: RelFileId, fork: ForkId, nblocks: u32) -> IoResult<()> {
        let pps = self.geometry(rel).pages_per_segment as u64;
        let mut seg = 0u32;
        loop {
            let seg_first = seg as u64 * pps;
            if seg_first >= nblocks as u64 && seg > 0 {
                if !self.vfs.exists(&self.path(rel, fork, seg)) {
                    break;
                }
                self.files.lock().remove(&(rel, fork, seg));
                self.sync_queue.lock().remove(&(rel, fork, seg));
                self.vfs.unlink(&self.path(rel, fork, seg))?;
                self.dirty_dirs.lock().insert(self.geometry(rel).dir);
            } else {
                let key = (rel, fork, seg);
                let Some(d) = self.desc(key, false, true)? else { break };
                let keep = (nblocks as u64).saturating_sub(seg_first).min(pps);
                let r = d.file.truncate(keep * PAGE_SIZE as u64);
                self.sync_queue.lock().insert(key);
                Self::finish_write(&d);
                r?;
            }
            seg += 1;
        }
        self.lens.lock().insert((rel, fork), nblocks);
        Ok(())
    }

    /// Unlinks the segments of a commit log below `first_segment` (spec 14.6). Durable after the
    /// next `sync_all`.
    pub fn unlink_log_segments_below(&self, rel: RelFileId, first_segment: u32) -> IoResult<usize> {
        let mut n = 0;
        for seg in 0..first_segment {
            let key = (rel, ForkId::Main, seg);
            let p = self.path(rel, ForkId::Main, seg);
            if self.vfs.exists(&p) {
                self.files.lock().remove(&key);
                self.sync_queue.lock().remove(&key);
                self.vfs.unlink(&p)?;
                n += 1;
            }
        }
        if n > 0 {
            self.dirty_dirs.lock().insert(self.geometry(rel).dir);
        }
        Ok(n)
    }

    /// Removes every file of a relation. Durable after the next `sync_all`.
    pub fn unlink_relation(&self, rel: RelFileId) -> IoResult<()> {
        let dir = self.geometry(rel).dir;
        for fork in [ForkId::Main, ForkId::Fsm, ForkId::Vm, ForkId::Overflow] {
            let mut seg = 0;
            loop {
                let p = self.path(rel, fork, seg);
                if !self.vfs.exists(&p) {
                    break;
                }
                self.files.lock().remove(&(rel, fork, seg));
                self.sync_queue.lock().remove(&(rel, fork, seg));
                self.vfs.unlink(&p)?;
                seg += 1;
            }
            self.lens.lock().remove(&(rel, fork));
        }
        self.dirty_dirs.lock().insert(dir);
        Ok(())
    }

    /// Syncs every queued file, then the directories whose entries changed. A failure here is
    /// fatal for the caller (spec 5.3 rule 3).
    pub fn sync_all(&self) -> IoResult<()> {
        let keys: Vec<FileKey> = std::mem::take(&mut *self.sync_queue.lock()).into_iter().collect();
        for k in keys {
            // Descriptors with unsynced writes are never evicted, so a missing one was unlinked.
            let d = self.files.lock().get(&k).cloned();
            if let Some(d) = d {
                // The sync covers the writes that had finished before it started.
                let covered = d.finished.load(Ordering::SeqCst);
                d.file.sync_data()?;
                d.synced.fetch_max(covered, Ordering::SeqCst);
            }
        }
        let dirs = std::mem::take(&mut *self.dirty_dirs.lock());
        for d in dirs {
            self.vfs.sync_dir(&self.gen_dir.join(d))?;
        }
        Ok(())
    }

    /// Number of files with writes not yet covered by a sync (diagnostics and tests).
    pub fn unsynced_files(&self) -> usize {
        self.files.lock().values().filter(|d| !d.is_clean()).count()
    }

    /// Relation file ids present in `base/` (for orphan removal).
    pub fn list_relfiles(&self) -> IoResult<BTreeSet<RelFileId>> {
        let mut out = BTreeSet::new();
        for e in self.vfs.list(&self.gen_dir.join(Geometry::RELATION.dir))? {
            if let Some(h) = e.name.split('.').next() {
                if h.len() == 16 {
                    if let Ok(id) = u64::from_str_radix(h, 16) {
                        out.insert(id);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Segment numbers present for a commit log.
    pub fn log_segments(&self, rel: RelFileId) -> IoResult<Vec<u32>> {
        let g = self.geometry(rel);
        let Some(stem) = g.log_stem else { return Ok(Vec::new()) };
        let mut out = Vec::new();
        for e in self.vfs.list(&self.gen_dir.join(g.dir))? {
            if let Some(n) = e.name.strip_prefix(stem).and_then(|r| r.strip_prefix('.')) {
                if let Ok(s) = n.parse::<u32>() {
                    out.push(s);
                }
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    pub fn forget_all(&self) {
        self.files.lock().clear();
        self.lens.lock().clear();
        self.sync_queue.lock().clear();
    }
}
