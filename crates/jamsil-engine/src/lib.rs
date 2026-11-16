//! `NativeEngine`, the facade of spec 17.1, for the durable-page layer (milestone M1):
//! create, open and recover, checkpoint, page-level logged updates, commit records.
//!
//! The heap, B-tree, transaction manager and SQL layers sit above this and are not built yet.
#![forbid(unsafe_code)]

mod recovery;

use jamsil_bufmgr::{BufError, BufferPool, CheckpointCtx, PinnedPage, ReadMode, Smgr, WalFlush};
use jamsil_page::control::{ControlData, DbState, SLOT_LEN};
use jamsil_page::page::{self, PageKind, PageMut, PageRef};
use jamsil_page::txnrec::RaftPos;
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::env::{Clock, OsRng, Rng, SystemClock};
use jamsil_vfs::sync::latch;
use jamsil_vfs::sync::{Arc, Mutex};
use jamsil_vfs::{read_exact_at, IoError, IoErrorKind, OpenOpts, Vfs};
use jamsil_txn::{CommitRefused, PoolStore, Recovered, Txn, TxnError, TxnManager, TxnOptions};
use jamsil_vfs::sync::atomic::{AtomicU64, Ordering};
use jamsil_wal::{LsnRange, StartPos, Wal, WalConfig, WalError};
use std::collections::HashMap;
use jamsil_page::walrec::WalPermit;
use std::path::{Path, PathBuf};

pub use recovery::{LogEnd, RecoveryReport};

pub use jamsil_txn::XID_BATCH;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Permanent configuration or format failure: exit code 78.
    #[error("format/configuration mismatch: {0}")]
    FormatMismatch(String),
    /// `JDB-3001`: corruption proven by recovery or a page read; the node is quarantined.
    #[error("JDB-3001 corruption: {0}")]
    Corruption(String),
    #[error("{0}")]
    Buf(#[from] BufError),
    #[error("{0}")]
    Wal(#[from] WalError),
    #[error("io: {0}")]
    Io(#[from] IoError),
    #[error("disk full")]
    DiskFull,
    /// `open` found no database. The adapter calls [`NativeEngine::create`] and opens again.
    #[error("no database at this location")]
    NotInitialized,
    #[error("a database already exists at this location")]
    AlreadyInitialized,
    /// The commit record was inserted but the flush failed: it may or may not survive a restart.
    /// Never to be reported to a client as "aborted" (spec 9.9, 17.2).
    #[error("commit outcome unknown: {0}")]
    OutcomeUnknown(String),
    /// SQLSTATE 54000: a record whose size depends on user data is over the limit (spec 8.1).
    #[error("54000 record of {len} bytes exceeds the limit of {max}")]
    RecordTooLarge { len: usize, max: usize },
    /// A failure of the transaction layer that has no case of its own here.
    #[error("{0}")]
    Txn(String),
}

impl From<TxnError> for EngineError {
    fn from(e: TxnError) -> Self {
        match e {
            TxnError::Quarantined(m) => EngineError::Wal(WalError::Quarantined(m)),
            TxnError::OutcomeUnknown(m) => EngineError::OutcomeUnknown(m),
            TxnError::RecordTooLarge { len, max } => EngineError::RecordTooLarge { len, max },
            other => EngineError::Txn(other.to_string()),
        }
    }
}

impl EngineError {
    /// SQLSTATE of spec 18.2, where the error has one of its own.
    pub fn sqlstate(&self) -> Option<&'static str> {
        match self {
            EngineError::RecordTooLarge { .. } => Some("54000"),
            // Transaction resolution unknown (spec 5.5, 18.2). Never to be read as "aborted".
            EngineError::OutcomeUnknown(_) => Some("08007"),
            EngineError::DiskFull => Some("53100"),
            EngineError::Corruption(_) => Some("XX001"),
            EngineError::Buf(b) => Some(b.sqlstate()),
            _ => None,
        }
    }

    /// Process exit code the supervisor expects for a start-up failure (spec 5.3, 6.6).
    pub fn exit_code(&self) -> Option<i32> {
        match self {
            EngineError::FormatMismatch(_) => Some(78),
            // A fatal I/O failure, a failed fdatasync among them, ends the process (spec 5.3 rule 3,
            // 5.6) and recovery runs from the WAL on the next start.
            EngineError::Io(e) if e.kind == IoErrorKind::Fatal => Some(75),
            _ => None,
        }
    }
}

pub type EResult<T> = Result<T, EngineError>;

/// What the engine does on a fatal failure (spec 5.3 rule 3): it should not return.
pub type FatalHandler = Arc<dyn Fn(&str) + Send + Sync>;

/// The default fatal-failure policy: exit with code 75 (spec 5.3 rule 3).
pub(crate) fn exit_on_fatal() -> FatalHandler {
    Arc::new(|msg: &str| {
        eprintln!("fatal: {msg}");
        std::process::exit(75);
    })
}

/// Segment sizes are powers of two from 256 KiB to 1 GiB (spec 6.6).
pub const MIN_WAL_SEGMENT: u64 = 256 * 1024;
pub const MAX_WAL_SEGMENT: u64 = 1 << 30;

fn check_segment_size(n: u64) -> EResult<()> {
    if n.is_power_of_two() && (MIN_WAL_SEGMENT..=MAX_WAL_SEGMENT).contains(&n) {
        Ok(())
    } else {
        Err(EngineError::FormatMismatch(format!("wal segment size {n} is not a power of two in 256 KiB..1 GiB")))
    }
}

#[derive(Clone)]
pub struct EngineOptions {
    pub buffer_frames: usize,
    /// `None`: use the value in the control file (64 MiB for a new database). `Some` must equal the
    /// control file's value, or start-up fails with exit code 78 (spec 6.6, 18.1).
    pub wal_segment_size: Option<u64>,
    pub cluster_id: Vec<u8>,
    pub wal_consistency_check: bool,
    /// `storage.synchronous`: true = FULL, false = NORMAL.
    pub synchronous_full: bool,
    /// `storage.wal_buffer_mib` in bytes (spec 18.1, default 64 MiB): how much WAL may be inserted
    /// and not yet written before an inserter waits for a flush (spec 8.4).
    pub wal_buffer_bytes: u64,
    pub checkpoint_pace: Option<std::time::Duration>,
    /// Start the WAL writer thread. Always true in production (spec 8.5: the 200 ms bound needs a
    /// flusher that has no waiter); the deterministic simulator sets it to false and drives
    /// `wal.flush_all()` itself.
    pub background_threads: bool,
    pub clock: Arc<dyn Clock>,
    /// Source of `system_id` at creation (spec 18.5).
    pub rng: Arc<dyn Rng>,
    /// Fatal-failure policy from the start of recovery on. `None`: exit with code 75 (spec 5.3).
    /// Tests that kill the process at every I/O boundary of recovery install a handler that
    /// panics. It should not return; if it does, the calling thread panics.
    pub fatal_handler: Option<FatalHandler>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            buffer_frames: 8192,
            wal_segment_size: None,
            cluster_id: b"test-cluster".to_vec(),
            wal_consistency_check: false,
            synchronous_full: true,
            wal_buffer_bytes: jamsil_wal::DEFAULT_WAL_BUFFER,
            checkpoint_pace: None,
            background_threads: true,
            clock: Arc::new(SystemClock),
            rng: Arc::new(OsRng),
            fatal_handler: None,
        }
    }
}

/// Bridges the buffer pool to the WAL. Before recovery has opened the WAL, `flush` is a silent
/// no-op and the `page_lsn` bound is not checked (spec 7.3: the insert position does not exist
/// during redo). The no-op is safe only because recovery syncs every segment it is about to read
/// before it reads them (spec 9.3 step 4, `sync_log_from`): everything redo can apply is already
/// durable, so no page written out during redo runs ahead of the durable log. The restructure that
/// removes this type (spec 8.8: the `Wal` is built first and switches to inserting) must keep that
/// sync. `redo_never_runs_ahead_of_the_durable_log` fails without it.
pub(crate) struct LateWal {
    wal: Mutex<Option<Arc<Wal>>>,
}

impl LateWal {
    fn get(&self) -> Option<Arc<Wal>> {
        self.wal.lock().clone()
    }
}

impl WalFlush for LateWal {
    fn flush(&self, upto: Lsn) -> Result<(), BufError> {
        match self.get() {
            Some(w) => w.flush(upto).map_err(|e| BufError::Wal(e.to_string())),
            None => Ok(()),
        }
    }
    fn insert_lsn(&self) -> Lsn {
        match self.get() {
            Some(w) => w.insert_lsn(),
            None => u64::MAX,
        }
    }
    fn redo_lsn(&self) -> Lsn {
        self.get().map(|w| w.redo_lsn()).unwrap_or(0)
    }
    fn permit_page_image(&self) -> Option<WalPermit> {
        // Before the WAL is attached (redo) a hint is never set, so there is nothing to permit.
        let w = self.get()?;
        w.check_size(&RecordBuilder::new(rmgr::XLOG, xlog::PAGE_IMAGE).block(BlockRef::new(PageTag::new(0, ForkId::Main, 0)))).ok()
    }
    fn insert_page_image(&self, permit: WalPermit, tag: PageTag, page: &[u8]) -> Lsn {
        let w = self.get().expect("page images are only logged once the WAL is open");
        w.insert(permit, RecordBuilder::new(rmgr::XLOG, xlog::PAGE_IMAGE).block(BlockRef::new(tag).image(page))).end
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AppliedPos {
    pub index: u64,
    pub term: u64,
}

pub struct NativeEngine {
    vfs: Arc<dyn Vfs>,
    opts: EngineOptions,
    gen_dir: PathBuf,
    root: PathBuf,
    _lock: Box<dyn Send + Sync>,
    pub wal: Arc<Wal>,
    pub pool: Arc<BufferPool>,
    pub smgr: Arc<Smgr>,
    control: Mutex<(ControlData, usize)>,
    checkpoint_mx: Mutex<()>,
    /// The transaction layer (spec 12): xid, oid and relfile allocation in logged batches, the commit
    /// log and the CSN log on the buffer pool, the commit and abort paths, the four positions.
    txn: TxnManager,
    /// The commit log and the CSN log on the buffer pool; `txn` holds the other handle.
    clog: Arc<PoolStore>,
    /// The commit log below this xid was truncated (spec 12.2); the control file records it.
    oldest_clog_xid: AtomicU64,
    /// The commit log may be truncated below this xid (spec 14.6): vacuum raises it, and the next
    /// checkpoint does the truncating.
    clog_horizon: AtomicU64,
    /// Truncations that failed after a completed checkpoint (spec 14.6).
    clog_truncate_failures: AtomicU64,
    /// Transactions begun through the xid-based calls below (`alloc_xid`, `commit`, `abort`), which
    /// stand in for the callers that `jamsil-heap` and `jamsil-exec` will be.
    open_txns: Mutex<HashMap<Xid, Txn>>,
    fatal_handler: Arc<Mutex<FatalHandler>>,
    pub recovery: RecoveryReport,
}

const INIT_DIR: &str = "init.tmp";

enum RootState {
    /// No database and no trace of an interrupted creation that `open` could finish.
    Empty,
    Ready,
    /// `CURRENT` names generation 1 but `init.tmp/` was not yet renamed.
    FinishCreate,
}

fn gen_name(n: u64) -> String {
    format!("gen-{n:06}")
}

fn parse_gen(name: &str) -> Option<u64> {
    name.strip_prefix("gen-")?.parse().ok()
}

impl NativeEngine {
    /// Opens the database at `root` (the `rdb/` directory) and runs crash recovery (spec 9.3).
    ///
    /// Never creates a database: an empty or missing location returns `NotInitialized` and
    /// changes nothing (spec 6.1). A creation interrupted after `CURRENT` was written is finished
    /// here; any other state with a `gen-*` directory but no `CURRENT` is a permanent failure.
    pub fn open(vfs: Arc<dyn Vfs>, root: &Path, opts: EngineOptions) -> EResult<Arc<NativeEngine>> {
        if let Some(w) = opts.wal_segment_size {
            check_segment_size(w)?;
        }
        if !vfs.exists(root) {
            return Err(EngineError::NotInitialized);
        }
        let lock = vfs.lock_exclusive(&root.join("LOCK"))?;
        match Self::classify_root(&*vfs, root)? {
            RootState::Empty => return Err(EngineError::NotInitialized),
            RootState::Ready => {}
            RootState::FinishCreate => Self::finish_create(&*vfs, root)?,
        }
        recovery::cleanup_generations(&*vfs, root)?;
        let gen_no = Self::read_current(&*vfs, root)?;
        let gen_dir = root.join(gen_name(gen_no));
        recovery::open_and_recover(vfs, root, gen_dir, opts, lock)
    }

    /// The table of spec 6.1: what the directory looks like at start-up.
    fn classify_root(vfs: &dyn Vfs, root: &Path) -> EResult<RootState> {
        let entries = vfs.list(root)?;
        let has_gen = entries.iter().any(|e| parse_gen(&e.name).is_some());
        if !vfs.exists(&root.join("CURRENT")) {
            // Creation renames CURRENT before it names the generation directory, so it can never
            // leave a generation without a CURRENT.
            if has_gen {
                return Err(EngineError::FormatMismatch("gen-* directory present but CURRENT is absent".into()));
            }
            return Ok(RootState::Empty);
        }
        let n = Self::read_current(vfs, root)?;
        if vfs.exists(&root.join(gen_name(n))) {
            return Ok(RootState::Ready);
        }
        if n == 1 && vfs.exists(&root.join(INIT_DIR)) {
            return Ok(RootState::FinishCreate);
        }
        Err(EngineError::FormatMismatch(format!("CURRENT names {} which does not exist", gen_name(n))))
    }

    fn finish_create(vfs: &dyn Vfs, root: &Path) -> EResult<()> {
        vfs.rename(&root.join(INIT_DIR), &root.join(gen_name(1)))?;
        vfs.sync_dir(root)?;
        Ok(())
    }

    pub(crate) fn read_current(vfs: &dyn Vfs, root: &Path) -> EResult<u64> {
        let f = vfs.open(&root.join("CURRENT"), OpenOpts::read_only())?;
        let mut buf = vec![0u8; f.len()? as usize];
        read_exact_at(&*f, &mut buf, 0)?;
        let s = String::from_utf8_lossy(&buf);
        parse_gen(s.trim()).ok_or_else(|| EngineError::FormatMismatch("bad CURRENT".into()))
    }

    /// `open`, creating the database first if there is none: what the adapter does on the first
    /// start of a node.
    pub fn open_or_create(vfs: Arc<dyn Vfs>, root: &Path, opts: EngineOptions) -> EResult<Arc<NativeEngine>> {
        match Self::open(vfs.clone(), root, opts.clone()) {
            Err(EngineError::NotInitialized) => {
                Self::create(vfs.clone(), root, &opts)?;
                Self::open(vfs, root, opts)
            }
            r => r,
        }
    }

    /// Creates generation 1 (the first start of a node). Crash-safe: everything is built in
    /// `init.tmp/` and synced, `CURRENT` is written and renamed, and only then does `init.tmp/`
    /// become `gen-000001/` (spec 6.1). `open` finishes a creation interrupted in between.
    pub fn create(vfs: Arc<dyn Vfs>, root: &Path, opts: &EngineOptions) -> EResult<()> {
        let seg = opts.wal_segment_size.unwrap_or(WAL_SEGMENT_SIZE);
        check_segment_size(seg)?;
        vfs.create_dir_all(root)?;
        let _lock = vfs.lock_exclusive(&root.join("LOCK"))?;
        match Self::classify_root(&*vfs, root)? {
            RootState::Ready => return Err(EngineError::AlreadyInitialized),
            RootState::FinishCreate => return Self::finish_create(&*vfs, root),
            RootState::Empty => {}
        }
        let init = root.join(INIT_DIR);
        if vfs.exists(&init) {
            recovery::remove_tree(&*vfs, &init)?;
        }
        for d in ["wal", "base", "xact", "tmp"] {
            vfs.create_dir_all(&init.join(d))?;
        }
        let mut cd = ControlData::new(opts.rng.next_u64() | 1, &opts.cluster_id);
        cd.wal_segment_size = seg as u32;
        cd.state = DbState::ShutDown;

        // Step 2: the log, written and synced by the code that writes it at run time: the first
        // segment and the two spares. (The first commit-log page and the system tables of spec 15.1
        // arrive with `jamsil-txn` and `jamsil-catalog`.)
        let wcfg = WalConfig { segment_size: seg, system_id: cd.system_id, buffer_bytes: opts.wal_buffer_bytes };
        let wal = Wal::open(vfs.clone(), &init.join("wal"), wcfg, StartPos::Fresh)?;
        if let Some(h) = &opts.fatal_handler {
            wal.set_fatal_handler(h.clone());
        }
        // Step 3: a shutdown checkpoint record. Its redo point is its own start, and the control
        // file names it, so that every database starts through the one recovery path.
        let redo = wal.begin_checkpoint();
        let mut main = Vec::with_capacity(64);
        for v in [redo, cd.next_xid, cd.next_oid as u64, cd.next_relfile, cd.next_xid, 0, 0, cd.schema_version] {
            main.extend_from_slice(&v.to_le_bytes());
        }
        let ck = RecordBuilder::new(rmgr::XLOG, xlog::CHECKPOINT).main(&main);
        let permit = wal.check_size(&ck)?;
        let rec = wal.insert(permit, ck);
        wal.flush(rec.end)?;
        drop(wal);
        cd.generation = 1;
        cd.checkpoint_lsn = rec.start;
        cd.redo_lsn = redo;
        let enc = cd.encode();
        let cf = vfs.open(&init.join("control"), OpenOpts::create())?;
        let mut file = vec![0u8; 2 * SLOT_LEN];
        file[..SLOT_LEN].copy_from_slice(&enc);
        file[SLOT_LEN..].copy_from_slice(&enc);
        cf.write_at(&file, 0)?;
        cf.sync_data()?;
        // Step 4: every file is synced (the log's segments by the flush above), and so is every
        // directory.
        for d in ["wal", "base", "xact", "tmp", ""] {
            vfs.sync_dir(&init.join(d))?;
        }
        // Step 5.
        let tmp = root.join("CURRENT.tmp");
        let c = vfs.open(&tmp, OpenOpts::create())?;
        c.truncate(0)?;
        c.write_at(format!("{}\n", gen_name(1)).as_bytes(), 0)?;
        c.sync_data()?;
        vfs.rename(&tmp, &root.join("CURRENT"))?;
        vfs.sync_dir(root)?;
        Self::finish_create(&*vfs, root)
    }

    pub(crate) fn wal_dir(gen_dir: &Path) -> PathBuf {
        gen_dir.join("wal")
    }

    pub fn options(&self) -> &EngineOptions {
        &self.opts
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn vfs(&self) -> &Arc<dyn Vfs> {
        &self.vfs
    }

    // ---- positions ---------------------------------------------------------------------

    /// The Raft position of the last commit or applied-advance record inserted, published or not
    /// (`logged_position`, spec 12.8).
    pub fn applied(&self) -> AppliedPos {
        let l = self.txn.logged();
        AppliedPos { index: l.logged_position, term: l.logged_term }
    }
    pub fn durable_applied_index(&self) -> u64 {
        self.txn.on_wal_durable(self.wal.durable_lsn());
        self.txn.durable_applied_index()
    }
    pub fn control_snapshot(&self) -> ControlData {
        self.control.lock().0.clone()
    }

    /// The transaction manager (spec 12.11).
    pub fn txn(&self) -> &TxnManager {
        &self.txn
    }

    // ---- allocation --------------------------------------------------------------------

    /// Begins a transaction, gives it an xid, and remembers it under that xid until `commit` or
    /// `abort`. Xids come from logged batches of 4,096 (spec 8.6, 9.3); this fails only when the log
    /// no longer accepts records and the batch is used up.
    pub fn alloc_xid(&self) -> EResult<Xid> {
        let mut txn = self.txn.begin();
        let xid = self.txn.assign_xid(&mut txn)?;
        self.open_txns.lock().insert(xid, txn);
        Ok(xid)
    }

    /// Marks the open transaction of `xid` as one that changed the schema: its commit record carries
    /// `schema_version` (spec 8.7, 12.8). Stands in for what `jamsil-catalog` will do.
    pub fn set_schema_version(&self, xid: Xid, schema_version: u64) {
        if let Some(t) = self.open_txns.lock().get_mut(&xid) {
            t.schema_version = Some(schema_version);
        }
    }

    /// Vacuum has hinted every tuple of every xid below `xid`, so the commit log is no longer needed
    /// below it (spec 14.6). This only raises the horizon: the checkpoint truncates, after it has
    /// completed, and never beyond the `oldest_active_xid` that it captured itself. Every xid below that
    /// value had finished before that checkpoint's redo point, so no record that redo can replay names a
    /// page of a removed segment.
    pub fn raise_clog_horizon(&self, xid: Xid) {
        self.clog_horizon.fetch_max(xid, Ordering::SeqCst);
    }

    pub fn alloc_relfile(&self) -> EResult<RelFileId> {
        Ok(self.txn.alloc_relfile()?)
    }

    // ---- relation files and page changes (the write protocol of spec 7.5) ----------------

    pub fn create_relation(&self, rel: RelFileId, fork: ForkId) -> EResult<()> {
        let mut main = [0u8; 9];
        wr_u64(&mut main, 0, rel);
        main[8] = fork as u8;
        let rec = RecordBuilder::new(rmgr::SMGR, smgr::CREATE).main(&main);
        let permit = self.gate(&rec)?;
        self.smgr.create_fork(rel, fork)?;
        self.wal.insert(permit, rec);
        Ok(())
    }

    /// Extends a fork by `n` formatted pages of kind 0, in chunks of at most 1/32 of the buffer
    /// pool (spec 7.8). The first record that uses a page formats it for its real purpose.
    /// Returns the first new block.
    pub fn extend_relation(&self, rel: RelFileId, fork: ForkId, n: u32) -> EResult<BlockNo> {
        let chunk = (self.pool.n_frames() / 32).max(1) as u32;
        let mut first_block = None;
        let mut left = n;
        while left > 0 {
            let this = left.min(chunk);
            let permit = self.gate(&RecordBuilder::new(rmgr::SMGR, smgr::EXTEND).main(&[0u8; 17]))?;
            let wal = self.wal.clone();
            let first = self.pool.extend(rel, fork, this, |old, new| {
                let mut main = [0u8; 17];
                wr_u64(&mut main, 0, rel);
                main[8] = fork as u8;
                wr_u32(&mut main, 9, old);
                wr_u32(&mut main, 13, new);
                Ok(wal.insert(permit, RecordBuilder::new(rmgr::SMGR, smgr::EXTEND).main(&main)).end)
            })?;
            first_block.get_or_insert(first);
            left -= this;
        }
        Ok(first_block.unwrap_or(0))
    }

    pub fn pin(&self, tag: PageTag) -> EResult<PinnedPage<'_>> {
        Ok(self.pool.pin(tag, ReadMode::Normal)?)
    }

    /// Logged write of `bytes` at `offset` of a page (generic test resource manager).
    /// Follows spec 7.5: latch, change, mark dirty, insert record, set LSN.
    pub fn write_bytes(&self, xid: Xid, tag: PageTag, offset: u16, bytes: &[u8]) -> EResult<LsnRange> {
        self.write_bytes_impl(xid, tag, offset, bytes, false)
    }

    /// Formats `n` pages from `first` as `Raw` pages (`WILL_INIT`, never needs an image).
    pub fn init_raw_pages(&self, xid: Xid, rel: RelFileId, fork: ForkId, first: BlockNo, n: u32) -> EResult<()> {
        for b in first..first + n {
            self.init_and_write_bytes(xid, PageTag::new(rel, fork, b), page::HEADER_SIZE as u16, &[])?;
        }
        Ok(())
    }

    /// Like `write_bytes` but the page is first formatted (`WILL_INIT`, never needs an image).
    pub fn init_and_write_bytes(&self, xid: Xid, tag: PageTag, offset: u16, bytes: &[u8]) -> EResult<LsnRange> {
        self.write_bytes_impl(xid, tag, offset, bytes, true)
    }

    /// Sets one byte as a hint-only change (spec 7.6), through `PinnedPage::try_hint`: the engine
    /// writes no record of its own for it, so the durability of the change is the buffer pool's
    /// `mark_dirty_hint` rule. Test helper standing in for the hint bits of heap and B-tree pages.
    /// A hint is optional: it is skipped, and this returns `false`, if the page latch is contended
    /// or the log does not accept records. It never fails for that reason.
    pub fn set_hint_byte(&self, tag: PageTag, offset: u16, value: u8) -> EResult<bool> {
        let pinned = self.pin(tag)?;
        Ok(pinned.try_hint(|page| page[offset as usize] = value))
    }

    fn write_bytes_impl(&self, xid: Xid, tag: PageTag, offset: u16, bytes: &[u8], init: bool) -> EResult<LsnRange> {
        assert!(offset as usize >= page::HEADER_SIZE && offset as usize + bytes.len() <= PAGE_SIZE);
        // Before the critical section: the payload (nothing below may allocate) and the gate (spec 8.8).
        let mut payload = Vec::with_capacity(2 + bytes.len());
        payload.extend_from_slice(&offset.to_le_bytes());
        payload.extend_from_slice(bytes);
        let op = if init { testop::INIT_SET_BYTES } else { testop::SET_BYTES };
        let permit = self.gate(&RecordBuilder::new(rmgr::TEST, op).xid(xid).block(BlockRef::new(tag).data(&payload)))?;
        let pinned = if init { self.pool.pin(tag, ReadMode::Init)? } else { self.pin(tag)? };
        let mut w = pinned.write();
        if init {
            PageMut(&mut w[..]).init(PageKind::Raw, 0);
        } else {
            assert_eq!(w[20], PageKind::Raw as u8, "write_bytes needs a page formatted by init_and_write_bytes");
        }
        // Critical section: nothing below may fail, allocate or block on I/O.
        w[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes);
        w.mark_dirty();
        let check = self.opts.wal_consistency_check;
        let range = self.wal.insert_with(permit, |redo| {
            let page_lsn = PageRef(&w[..]).lsn();
            let mut b = BlockRef::new(tag).data(&payload);
            if init {
                b = b.flags(BLK_WILL_INIT);
            } else if page_lsn <= redo {
                b = b.image(&w[..]);
            } else if check {
                b = b.image(&w[..]).flags(BLK_FORCED_IMAGE);
            }
            RecordBuilder::new(rmgr::TEST, op).xid(xid).block(b)
        });
        w.set_lsn(range.end);
        Ok(range)
    }

    // ---- commit ------------------------------------------------------------------------

    /// Commit record carrying the Raft position (spec 8.7). Flushes under FULL. `xid` was returned by
    /// `alloc_xid`. The commit follows the six steps of spec 12.8 inside the transaction manager.
    ///
    /// The record's size depends on user data (the relation files to unlink), so it is measured
    /// before the critical section and fails with `54000` if it is too large (answer B18). If the gate
    /// refuses, nothing was logged and the transaction is still the caller's, still open: it can be
    /// committed again or aborted (answer B2). Once the record is inserted, a failed flush is reported
    /// as `OutcomeUnknown`, never as an abort.
    pub fn commit(&self, xid: Xid, pos: Option<AppliedPos>, unlink: &[RelFileId]) -> EResult<Csn> {
        latch::assert_none_held("commit flush");
        let mut txn = self.open_txns.lock().remove(&xid).expect("commit of an xid that alloc_xid returned and that is still open");
        txn.unlink = unlink.to_vec();
        let ticket = match self.txn.commit(txn, pos.map(|p| RaftPos { index: p.index, term: p.term })) {
            Ok(t) => t,
            Err(CommitRefused { txn, error }) => {
                self.open_txns.lock().insert(xid, txn);
                return Err(error.into());
            }
        };
        let csn = ticket.wait()?;
        self.txn.on_wal_durable(self.wal.durable_lsn());
        for r in unlink {
            self.pool.drop_relation(*r);
            self.smgr.unlink_relation(*r)?;
        }
        Ok(csn)
    }

    /// Applied-advance record (Raft no-ops, empty transactions).
    pub fn applied_advance(&self, pos: AppliedPos) -> EResult<Csn> {
        latch::assert_none_held("commit flush");
        let csn = self.txn.applied_advance_sync(RaftPos { index: pos.index, term: pos.term })?;
        self.txn.on_wal_durable(self.wal.durable_lsn());
        Ok(csn)
    }

    /// Aborts the transaction of `xid`. An abort is not flushed, and a crash aborts every xid that has
    /// no commit record (spec 9.3); on a log that no longer accepts records the manager remembers the
    /// abort in memory (answer B2).
    pub fn abort(&self, xid: Xid) {
        let txn = self.open_txns.lock().remove(&xid);
        if let Some(txn) = txn {
            self.txn.abort(txn);
        }
    }

    /// The gate before every critical section (spec 8.8): the log accepts records, the record
    /// fits even with a full image for each block, and the insert will not allocate. Every
    /// function of the engine that inserts a record while it holds a latch or a lock calls it
    /// first, with a builder that names the record's blocks and carries its payload.
    fn gate(&self, rec: &RecordBuilder<'_>) -> EResult<WalPermit> {
        self.wal.check_size(rec).map_err(Self::size_error)
    }

    fn size_error(e: WalError) -> EngineError {
        match e {
            WalError::RecordTooLarge { len, max } => EngineError::RecordTooLarge { len, max },
            other => other.into(),
        }
    }

    /// True while the WAL could not create a spare segment. The adapter and `jamsil-exec` use it
    /// to fail new write statements with `53100` (spec 5.5); records already admitted go on.
    pub fn space_low(&self) -> bool {
        self.wal.space_low()
    }

    // ---- checkpoint (spec 9.2) -----------------------------------------------------------

    pub fn checkpoint(&self) -> EResult<()> {
        self.checkpoint_inner(DbState::InProduction)
    }

    pub(crate) fn checkpoint_inner(&self, state: DbState) -> EResult<()> {
        // A checkpoint on a log that no longer accepts records would write pages it can never log.
        let permit = self.gate(&RecordBuilder::new(rmgr::XLOG, xlog::CHECKPOINT).main(&[0u8; 64]))?;
        let _only_one = self.checkpoint_mx.lock();
        // Steps 1 and 2 under the commit mutex: no commit record is half-logged. The manager reads the
        // logged position and the logged schema version in the same hold in which the closure takes
        // `redo_lsn`, and the allocation high-water marks after it (spec 9.2).
        let cap = self.txn.checkpoint_capture(|| self.wal.begin_checkpoint());
        let redo = cap.redo;
        let applied = AppliedPos { index: cap.logged.logged_position, term: cap.logged.logged_term };
        let schema = cap.logged.logged_schema_version;
        let (next_xid, next_oid, next_relfile) = (cap.alloc.next_xid.max(cap.published.next_xid), u64::from(cap.alloc.next_oid), cap.alloc.next_relfile);
        // Step 3: dirty frames present at step 1.
        self.pool.mark_checkpoint_needed(redo);
        let ctx = CheckpointCtx { pace: self.opts.checkpoint_pace };
        self.pool.flush_all(&ctx)?;
        // Step 4.
        if let Err(e) = self.smgr.sync_all() {
            self.fatal(&format!("fdatasync failed during checkpoint: {e}"));
        }
        // Step 5.
        let mut main = Vec::with_capacity(64);
        for v in [redo, next_xid, next_oid, next_relfile, next_xid, applied.index, applied.term, schema] {
            main.extend_from_slice(&v.to_le_bytes());
        }
        let rec = self.wal.insert(permit, RecordBuilder::new(rmgr::XLOG, xlog::CHECKPOINT).main(&main));
        self.wal.flush(rec.end)?;
        // Step 6.
        {
            let mut c = self.control.lock();
            let mut cd = c.0.clone();
            cd.generation += 1;
            cd.state = state;
            cd.checkpoint_lsn = rec.start;
            cd.redo_lsn = redo;
            cd.next_xid = next_xid;
            cd.oldest_clog_xid = self.oldest_clog_xid.load(Ordering::SeqCst);
            cd.next_oid = next_oid as u32;
            cd.next_relfile = next_relfile;
            cd.applied_index = applied.index;
            cd.applied_term = applied.term;
            cd.schema_version = schema;
            let slot = 1 - c.1;
            self.write_control_slot(&cd, slot);
            *c = (cd, slot);
        }
        // Step 7.
        self.wal.remove_segments_below(redo)?;
        // Truncation of the commit log, after the checkpoint has completed, clamped by what it captured
        // (spec 14.6). The checkpoint is complete: a truncation that fails does not fail it.
        self.truncate_clog_after_checkpoint(cap.published.oldest_active_xid);
        Ok(())
    }

    /// Truncates the commit log up to the horizon vacuum raised, clamped by the `oldest_active_xid` that
    /// the checkpoint captured. A failure is logged and counted and the next checkpoint tries again; a
    /// fatal I/O error ends the process (spec 14.6). Public so that a test can give it a captured value
    /// that an engine of a few xids does not reach.
    #[doc(hidden)]
    pub fn truncate_clog_after_checkpoint(&self, captured_oldest_active: Xid) {
        let upto = self.clog_horizon.load(Ordering::SeqCst).min(captured_oldest_active);
        match self.clog.truncate_clog(upto) {
            Ok(kept) => {
                self.oldest_clog_xid.fetch_max(kept, Ordering::SeqCst);
            }
            Err(e) => {
                self.clog_truncate_failures.fetch_add(1, Ordering::SeqCst);
                eprintln!("jamsil-engine: error: commit-log truncation below xid {upto} failed after a completed checkpoint: {e}");
            }
        }
    }

    /// How many truncations failed after a checkpoint (spec 14.6).
    pub fn clog_truncate_failures(&self) -> u64 {
        self.clog_truncate_failures.load(Ordering::SeqCst)
    }

    fn write_control_slot(&self, cd: &ControlData, slot: usize) {
        let r = (|| -> Result<(), IoError> {
            let f = self.vfs.open(&self.gen_dir.join("control"), OpenOpts::read_write())?;
            f.write_at(&cd.encode(), (slot * SLOT_LEN) as u64)?;
            f.sync_data()
        })();
        if let Err(e) = r {
            self.fatal(&format!("control file write failed: {e}"));
        }
    }

    /// Replaces the fatal-failure policy (default: exit code 75) for the engine and its WAL.
    /// The handler should not return; if it does the calling thread panics.
    pub fn set_fatal_handler(&self, f: Arc<dyn Fn(&str) + Send + Sync>) {
        self.wal.set_fatal_handler(f.clone());
        *self.fatal_handler.lock() = f;
    }

    pub(crate) fn fatal(&self, msg: &str) -> ! {
        let h = self.fatal_handler.lock().clone();
        h(msg);
        panic!("fatal: {msg}");
    }

    /// Stops the background threads without flushing or checkpointing (what dropping does).
    /// `Err` if the WAL writer had died of a panic.
    pub fn halt(&self) -> EResult<()> {
        Ok(self.wal.stop()?)
    }

    /// Clean shutdown: final checkpoint, state "shut down cleanly". The writer thread is stopped
    /// first: if it had died of a panic the log is suspect, and the control file must not say
    /// "shut down cleanly" on top of it. The caller's own flush does the rest.
    pub fn shutdown(&self) -> EResult<()> {
        self.wal.stop()?;
        self.checkpoint_inner(DbState::ShutDown)
    }

    pub fn wal_segment_size(&self) -> u64 {
        self.wal.segment_size()
    }

    /// Verifies every page of a fork by reading it (test and tooling helper).
    pub fn verify_fork(&self, rel: RelFileId, fork: ForkId) -> EResult<u32> {
        let n = self.smgr.nblocks(rel, fork)?;
        for b in 0..n {
            let p = self.pool.pin(PageTag::new(rel, fork, b), ReadMode::Normal)?;
            let _ = p.read();
        }
        Ok(n)
    }
}

impl Drop for NativeEngine {
    /// Waits for the WAL writer thread; never flushes. A crash is dropping the engine and then
    /// crashing the file system underneath it.
    fn drop(&mut self) {
        let _ = self.wal.stop();
    }
}
