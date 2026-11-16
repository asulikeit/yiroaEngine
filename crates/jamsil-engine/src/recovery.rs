//! Crash recovery (spec 9.3, 9.4): redo from the last checkpoint, end-of-log classification.

use crate::*;
use jamsil_bufmgr::PageWrite;
use jamsil_page::txnrec::{decode_applied_advance, AbortRecord, AllocBatch, CommitView};
use jamsil_page::walrec::testop;
use jamsil_heap::{redo_heap, HeapRedoError, RedoPages};
use jamsil_txn::{redo_alloc_batch, redo_clog_truncate, redo_clog_zero_page, redo_xact_bits, PoolStore, RedoError, WalBridge};
use jamsil_wal::{truncate_tail, ReadStop, WalReader};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LogEnd {
    /// Zeros to the end of the segment, or no further segment.
    #[default]
    Normal,
    /// Non-durable garbage after the last valid record; the valid prefix was kept.
    TornTail,
}

#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    pub records_replayed: u64,
    pub redo_start: Lsn,
    pub end_lsn: Lsn,
    pub log_end: LogEnd,
    pub applied: AppliedPos,
    pub restored_images: u64,
    /// The control file did not say "shut down cleanly" (spec 9.3 step 2). Redo runs either way;
    /// the state only decides how the start is reported.
    pub crash_recovery: bool,
}

pub(crate) fn cleanup_generations(vfs: &dyn Vfs, root: &Path) -> EResult<()> {
    let cur = NativeEngine::read_current(vfs, root)?;
    let tmp = root.join("CURRENT.tmp");
    if vfs.exists(&tmp) {
        vfs.unlink(&tmp)?;
    }
    // A creation that finished (CURRENT names an existing generation) leaves nothing behind
    // (spec 6.1 table, row 3).
    let init = root.join("init.tmp");
    if vfs.exists(&init) {
        remove_tree(vfs, &init)?;
    }
    for e in vfs.list(root)? {
        if let Some(g) = parse_gen(&e.name) {
            if g != cur {
                remove_tree(vfs, &root.join(&e.name))?;
            }
        }
    }
    vfs.sync_dir(root)?;
    Ok(())
}

/// Calls `sync_dir` on `wal/`, `base/` and `xact/` (spec 9.3 step 4).
fn sync_dirs(vfs: &dyn Vfs, gen_dir: &Path) -> EResult<()> {
    for sub in ["wal", "base", "xact"] {
        vfs.sync_dir(&gen_dir.join(sub))?;
    }
    Ok(())
}

/// Calls `sync_data` on every WAL segment from the one that holds `from` onward (spec 9.3 step 4).
fn sync_log_from(vfs: &dyn Vfs, wal_dir: &Path, cfg: &WalConfig, from: Lsn) -> EResult<()> {
    let first = from / cfg.segment_size;
    let mut segs: Vec<u64> = vfs
        .list(wal_dir)?
        .iter()
        .filter_map(|e| jamsil_wal::parse_segment_name(&e.name))
        .filter(|s| *s >= first)
        .collect();
    segs.sort_unstable();
    for s in segs {
        vfs.open(&jamsil_wal::segment_path(wal_dir, s), OpenOpts::read_write())?.sync_data()?;
    }
    Ok(())
}

pub(crate) fn remove_tree(vfs: &dyn Vfs, dir: &Path) -> EResult<()> {
    for e in vfs.list(dir)? {
        let p = dir.join(&e.name);
        if e.is_dir {
            remove_tree(vfs, &p)?;
        } else {
            vfs.unlink(&p)?;
        }
    }
    vfs.remove_dir(dir)?;
    Ok(())
}

pub(crate) fn open_and_recover(
    vfs: Arc<dyn Vfs>,
    root: &Path,
    gen_dir: PathBuf,
    opts: EngineOptions,
    lock: Box<dyn Send + Sync>,
) -> EResult<Arc<NativeEngine>> {
    // Step 1: control file and format checks.
    let cf = vfs.open(&gen_dir.join("control"), OpenOpts::read_only())?;
    let mut raw = vec![0u8; 2 * SLOT_LEN];
    let n = cf.read_at(&mut raw, 0)?;
    raw.truncate(n);
    let (cd, slot) = ControlData::pick(&raw).map_err(|e| EngineError::Corruption(e.0))?;
    if cd.format_version != control::FORMAT_VERSION {
        return Err(EngineError::FormatMismatch(format!("format_version {}", cd.format_version)));
    }
    if cd.page_size as usize != PAGE_SIZE {
        return Err(EngineError::FormatMismatch(format!("page_size {}", cd.page_size)));
    }
    let mut want = [0u8; 64];
    let k = opts.cluster_id.len().min(64);
    want[..k].copy_from_slice(&opts.cluster_id[..k]);
    if want != cd.cluster_id {
        return Err(EngineError::FormatMismatch("cluster_id".into()));
    }
    let seg = cd.wal_segment_size as u64;
    check_segment_size(seg).map_err(|_| EngineError::FormatMismatch(format!("control file wal_segment_size {seg}")))?;
    if let Some(want) = opts.wal_segment_size {
        if want != seg {
            return Err(EngineError::FormatMismatch(format!(
                "wal_segment_mib {} differs from the control file's {} (fixed when the database is created)",
                want, seg
            )));
        }
    }
    let wal_dir = NativeEngine::wal_dir(&gen_dir);
    vfs.create_dir_all(&wal_dir)?;
    // tmp/ is emptied at start.
    let tmp_dir = gen_dir.join("tmp");
    vfs.create_dir_all(&tmp_dir)?;
    for e in vfs.list(&tmp_dir)? {
        vfs.unlink(&tmp_dir.join(&e.name))?;
    }

    let smgr = Arc::new(Smgr::new(vfs.clone(), &gen_dir)?);
    let late = Arc::new(LateWal { wal: Mutex::new(None) });
    let pool = BufferPool::new(opts.buffer_frames, smgr.clone(), late.clone());
    // The CSN log is emptied at every start (spec 12.2): after a restart every earlier xid precedes
    // every new snapshot, and the commit log alone decides.
    PoolStore::reset_csn_log(&pool).map_err(|e| EngineError::Txn(e.to_string()))?;
    let wcfg = WalConfig { segment_size: seg, system_id: cd.system_id, buffer_bytes: opts.wal_buffer_bytes };

    let mut report = RecoveryReport { crash_recovery: cd.state != DbState::ShutDown, ..Default::default() };
    let mut ctl = cd.clone();
    // `create` writes a shutdown checkpoint record (spec 6.1 step 3), so a control file that names
    // none is damaged.
    if cd.checkpoint_lsn == 0 {
        return Err(EngineError::Corruption("the control file names no checkpoint record".into()));
    }
    // Step 3: the checkpoint record.
    let mut rd = WalReader::new(vfs.clone(), wal_dir.clone(), wcfg.clone(), cd.checkpoint_lsn);
    let ck = rd
        .next_record()
        .map_err(|e| EngineError::Corruption(format!("checkpoint record unreadable: {e:?}")))?;
    if ck.rmgr != rmgr::XLOG || ck.info != xlog::CHECKPOINT || ck.main.len() < 64 {
        return Err(EngineError::Corruption("control file does not name a checkpoint record".into()));
    }
    let ckpt_end = ck.end_lsn;
    // The allocation marks start at the checkpoint's (spec 9.3 "Xid counter"); the control file
    // holds the same values, and a record in the log can only raise them.
    let alloc = AllocBatch {
        next_xid: cd.next_xid.max(rd_u64(&ck.main, 8)),
        next_oid: cd.next_oid.max(rd_u64(&ck.main, 16) as u32),
        next_relfile: cd.next_relfile.max(rd_u64(&ck.main, 24)).max(FIRST_RELFILE),
    };
    let mut applied = AppliedPos { index: rd_u64(&ck.main, 40), term: rd_u64(&ck.main, 48) };

    // Steps 4 to 6.
    let mut redo = Redo {
        pool: &pool,
        smgr: &smgr,
        check: opts.wal_consistency_check,
        missing: HashSet::new(),
        dropped: HashSet::new(),
        applied,
        schema_version: rd_u64(&ck.main, 56),
        alloc,
        highest_xid: 0,
        oldest_clog_xid: cd.oldest_clog_xid,
        records: 0,
        images: 0,
    };
    report.redo_start = cd.redo_lsn;
    // Step 4, first half (spec 9.3), at every start whatever the state field says. After a process
    // kill, file creations and log records can be visible and still not durable, and the new
    // process has no record of which. Redo relies on nothing it has not made durable itself.
    //
    // The log: every record that redo can read must be durable before it is applied. A page
    // written out during redo would otherwise carry a `page_lsn` beyond the durable log, and a power
    // loss at that point would break invariant 2 of spec 7.10. The pool has no WAL to flush through
    // yet (`LateWal`), so this sync is what makes that no-op safe.
    sync_log_from(&*vfs, &wal_dir, &wcfg, cd.redo_lsn)?;
    // The directories: a spare that the dead process created without a `sync_dir` would otherwise
    // be used with no durable entry, and the recovery checkpoint would move the redo point past the
    // `CREATE` record of a data file whose entry is still in the page cache. File contents need no
    // sync of their own, and neither does a sync of every data file (spec 2.1: the cost of a
    // restart does not depend on the size of the database): the two rules of `Redo` leave every
    // page and file operation that a replayed record names dirty or queued, and the end-of-recovery
    // checkpoint writes and syncs them before the redo point moves.
    sync_dirs(&*vfs, &gen_dir)?;
    let mut rd = WalReader::new(vfs.clone(), wal_dir.clone(), wcfg.clone(), cd.redo_lsn);
    let stop = loop {
        match rd.next_record() {
            Ok(rec) => redo.apply(&rec)?,
            Err(stop) => break stop,
        }
    };
    let verdict = classify(&vfs, &wal_dir, &wcfg, &stop, ckpt_end, rd.valid_end())?;
    let end = rd.valid_end();
    let torn = matches!(verdict, Verdict::TornTail);
    truncate_tail(&*vfs, &wal_dir, &wcfg, end)?;
    report.log_end = if torn { LogEnd::TornTail } else { LogEnd::Normal };
    // Every unresolved reference to a missing relation must have been dropped later.
    for rel in &redo.missing {
        if !redo.dropped.contains(rel) {
            return Err(EngineError::Corruption(format!("redo references missing relation {rel}")));
        }
    }
    // The restart value of `next_xid` is the highest of three: the checkpoint's, the end of the last
    // allocation batch in the log, and one past the highest xid or sub-xid in any replayed record
    // (spec 9.3 "Xid counter"). An xid is therefore never handed out twice, even one whose transaction
    // left no other trace.
    let mut alloc = redo.alloc;
    alloc.next_xid = alloc.next_xid.max(redo.highest_xid);
    let oldest_clog_xid = redo.oldest_clog_xid;
    applied = redo.applied;
    let schema_version = redo.schema_version;
    report.records_replayed = redo.records;
    report.restored_images = redo.images;
    report.end_lsn = end;
    let start = StartPos::At { end, last: rd.last_start().unwrap_or(0) };


    // Step 7 (crash horizon) is a lookup-time rule: nothing is written. The manager applies it: every
    // xid below `alloc.next_xid` that has no commit record is aborted.
    let wal = Wal::open(vfs.clone(), &wal_dir, wcfg, start)?;
    *late.wal.lock() = Some(wal.clone());
    if let Some(h) = &opts.fatal_handler {
        wal.set_fatal_handler(h.clone());
    }
    if opts.background_threads {
        wal.start_writer(jamsil_wal::WRITER_INTERVAL, |m| eprintln!("fatal: {m}"));
    }
    ctl.state = DbState::InRecovery;
    report.applied = applied;

    let fatal_handler = Arc::new(Mutex::new(opts.fatal_handler.clone().unwrap_or_else(exit_on_fatal)));
    let store_fatal = {
        let h = fatal_handler.clone();
        Arc::new(move |m: &str| {
            let f = h.lock().clone();
            f(m)
        })
    };
    // A commit-log page that fails verification quarantines the store (a restart would meet the same
    // page again); a fatal I/O error ends the process (spec 5.6).
    let quarantine = {
        let w = wal.clone();
        Arc::new(move |kind: jamsil_wal::QuarantineKind, m: &str| w.quarantine(kind, m))
    };
    let clog = Arc::new(PoolStore::new(pool.clone(), wal.clone(), store_fatal, quarantine));
    let txn = TxnManager::new(
        clog.clone(),
        Arc::new(WalBridge(wal.clone())),
        opts.clock.clone(),
        // No sequencer admits entries in log order yet (spec 12.8 step 1), and the engine's callers
        // still choose their own Raft positions.
        TxnOptions { synchronous_full: opts.synchronous_full, check_continuity: false, ..Default::default() },
        Recovered {
            next_xid: alloc.next_xid,
            next_oid: alloc.next_oid,
            next_relfile: alloc.next_relfile,
            // Every CSN is the end LSN of a commit record, so the end of the log is above all of them
            // and a snapshot at it sees every commit of the valid log.
            csn: end,
            applied: applied.index,
            applied_term: applied.term,
            schema_version,
        },
    );
    let engine = Arc::new(NativeEngine {
        vfs,
        opts,
        gen_dir,
        root: root.to_path_buf(),
        _lock: lock,
        wal,
        pool,
        smgr,
        control: Mutex::new((ctl, slot)),
        checkpoint_mx: Mutex::new(()),
        txn,
        clog,
        oldest_clog_xid: AtomicU64::new(oldest_clog_xid),
        clog_horizon: AtomicU64::new(oldest_clog_xid),
        clog_truncate_failures: AtomicU64::new(0),
        open_txns: Mutex::new(HashMap::new()),
        fatal_handler,
        recovery: report,
    });
    // Step 9: end-of-recovery record and checkpoint; the state becomes "in production".
    let end = RecordBuilder::new(rmgr::XLOG, xlog::END_OF_RECOVERY);
    let permit = engine.wal.check_size(&end)?;
    engine.wal.insert(permit, end);
    engine.checkpoint_inner(DbState::InProduction)?;
    Ok(engine)
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Normal,
    TornTail,
}

/// The table of spec 9.3 "End of log or corruption". Corruption stops recovery and changes
/// nothing on disk.
fn classify(
    vfs: &Arc<dyn Vfs>,
    dir: &Path,
    cfg: &WalConfig,
    stop: &ReadStop,
    ckpt_end: Lsn,
    valid_end: Lsn,
) -> EResult<Verdict> {
    let seg = cfg.segment_size;
    let corrupt = |m: String| Err(EngineError::Corruption(m));
    let pos = stop.pos();
    // Row 1: the failure lies below the end of the checkpoint record named by the control file.
    if valid_end < ckpt_end {
        return corrupt(format!("log ends at {valid_end}, before checkpoint record end {ckpt_end}: {stop:?}"));
    }
    // A missing next segment is the ordinary end of the log.
    if let ReadStop::NoSegment { .. } = stop {
        return Ok(Verdict::Normal);
    }
    let segno = pos / seg;
    // Row 2: a later segment holds at least one valid record.
    for e in vfs.list(dir)? {
        if let Some(s) = jamsil_wal::parse_segment_name(&e.name) {
            if s > segno {
                let mut r = WalReader::new(vfs.clone(), dir.to_path_buf(), cfg.clone(), s * seg);
                if r.next_record().is_ok() {
                    return corrupt(format!("invalid record at {pos} but segment {s} holds a valid record"));
                }
            }
        }
    }
    // Read the rest of the segment from the failed position (or from the segment start when the
    // header itself failed).
    let scan_from = if pos % seg == jamsil_page::walrec::SEG_HEADER_LEN as u64 { segno * seg } else { pos };
    let path = jamsil_wal::segment_path(dir, segno);
    let f = vfs.open(&path, OpenOpts::read_only())?;
    let len = (segno + 1) * seg - scan_from;
    let mut tail = vec![0u8; len as usize];
    let n = f.read_at(&mut tail, scan_from - segno * seg)?;
    tail.truncate(n);
    // Row 3: zeros to the end of the segment.
    if tail.iter().all(|&b| b == 0) {
        return Ok(Verdict::Normal);
    }
    // Row 4: a valid durability marker above the position proves these bytes were synced.
    let mut off = 0usize;
    // Records are 8-byte aligned relative to the segment start.
    while off + 48 <= tail.len() {
        let abs = scan_from + off as u64;
        if abs % 8 == 0 && rd_u32(&tail, off) == 48 && tail[off + 32] == rmgr::XLOG && tail[off + 33] == xlog::DURABILITY_MARKER {
            let prev = rd_u64(&tail, off + 16);
            if let Ok(rec) = decode(&tail[off..off + 48], abs, prev) {
                if rec.main.len() == 8 && rd_u64(&rec.main, 0) > pos {
                    return corrupt(format!(
                        "invalid record at {pos}, but a durability marker at {abs} proves the log was durable up to {}",
                        rd_u64(&rec.main, 0)
                    ));
                }
            }
        }
        off += 8;
    }
    // Row 5: torn tail.
    Ok(Verdict::TornTail)
}

struct Redo<'a> {
    pool: &'a Arc<BufferPool>,
    smgr: &'a Arc<Smgr>,
    check: bool,
    missing: HashSet<RelFileId>,
    dropped: HashSet<RelFileId>,
    applied: AppliedPos,
    /// The schema version of the last commit record replayed that carried one (spec 9.2: the
    /// checkpoint's value is the starting point).
    schema_version: u64,
    /// The allocation high-water marks: the checkpoint's, raised by every batch record replayed.
    alloc: AllocBatch,
    /// One past the highest xid or sub-xid in any replayed record.
    highest_xid: Xid,
    /// The commit log below this xid was truncated (spec 12.2); records for such xids are skipped.
    oldest_clog_xid: Xid,
    records: u64,
    images: u64,
}

fn redo_error(e: RedoError) -> EngineError {
    match e {
        RedoError::Corrupt(m) => EngineError::Corruption(m),
        RedoError::Buf(b) => b.into(),
        RedoError::Io(i) => i.into(),
    }
}

enum BlockState<'a> {
    /// The page was rebuilt from the record's image; nothing more to do.
    Restored,
    /// The page already reflects this record.
    UpToDate,
    /// The relation does not exist (yet); the reference is remembered.
    Missing,
    Redo(PinnedPage<'a>),
}

impl Redo<'_> {
    fn apply(&mut self, rec: &WalRecord) -> EResult<()> {
        self.records += 1;
        if rec.xid != 0 {
            self.highest_xid = self.highest_xid.max(rec.xid + 1);
        }
        match rec.rmgr {
            rmgr::XLOG => match rec.info {
                xlog::CHECKPOINT if rec.main.len() >= 64 => {
                    self.alloc.raise(AllocBatch {
                        next_xid: rd_u64(&rec.main, 8),
                        next_oid: rd_u64(&rec.main, 16) as u32,
                        next_relfile: rd_u64(&rec.main, 24),
                    });
                }
                xlog::ALLOC_BATCH => {
                    // Strict (spec 8.6): a batch that does not decode is corruption.
                    let b = redo_alloc_batch(rec).map_err(redo_error)?;
                    self.alloc.raise(b);
                }
                xlog::PAGE_IMAGE => {
                    if let Some(b) = rec.blocks.first() {
                        self.restore(rec, b)?;
                    }
                }
                _ => {}
            },
            rmgr::XACT => match rec.info {
                // The decoders are strict (spec 8.7): a record of a transaction that does not decode is
                // corruption, not something to skip (spec 18.2).
                xact::COMMIT => {
                    let v = CommitView::parse(&rec.main).map_err(|e| EngineError::Corruption(e.0))?;
                    if let Some(p) = v.raft() {
                        self.applied = AppliedPos { index: p.index, term: p.term };
                    }
                    if let Some(sv) = v.schema_version() {
                        self.schema_version = sv;
                    }
                    for s in v.sub_xids() {
                        self.highest_xid = self.highest_xid.max(s + 1);
                    }
                    // The commit-log bits of the xid and its sub-xids (spec 12.2).
                    redo_xact_bits(self.pool, rec, self.oldest_clog_xid).map_err(redo_error)?;
                    for rel in v.unlink() {
                        self.pool.drop_relation(rel);
                        let _ = self.smgr.unlink_relation(rel);
                        self.dropped.insert(rel);
                    }
                }
                xact::APPLIED_ADVANCE => {
                    let p = decode_applied_advance(&rec.main).map_err(|e| EngineError::Corruption(e.0))?;
                    self.applied = AppliedPos { index: p.index, term: p.term };
                }
                xact::ABORT => {
                    let a = AbortRecord::decode(&rec.main).map_err(|e| EngineError::Corruption(e.0))?;
                    for s in &a.sub_xids {
                        self.highest_xid = self.highest_xid.max(s + 1);
                    }
                    redo_xact_bits(self.pool, rec, self.oldest_clog_xid).map_err(redo_error)?;
                }
                _ => {}
            },
            rmgr::CLOG => match rec.info {
                clog::ZERO_PAGE => redo_clog_zero_page(self.pool, rec).map_err(redo_error)?,
                clog::TRUNCATE => {
                    let oldest = redo_clog_truncate(self.pool, rec).map_err(redo_error)?;
                    self.oldest_clog_xid = self.oldest_clog_xid.max(oldest);
                }
                _ => {}
            },
            rmgr::HEAP => redo_heap(self, rec).map_err(|e| match e {
                HeapRedoError::Corrupt(m) => EngineError::Corruption(m),
                HeapRedoError::Pages(m) => EngineError::Txn(m),
            })?,
            rmgr::SMGR => self.redo_smgr(rec)?,
            rmgr::TEST => self.redo_test(rec)?,
            _ => {}
        }
        Ok(())
    }

    fn restore(&mut self, rec: &WalRecord, b: &BlockData) -> EResult<()> {
        self.smgr.ensure_len(b.tag.rel, b.tag.fork, b.tag.block + 1)?;
        let p = self.pool.pin(b.tag, ReadMode::Init)?;
        let mut w = p.write();
        b.restore_image(&mut w[..]).map_err(|e| EngineError::Corruption(e.0))?;
        w.mark_dirty();
        w.set_lsn(rec.end_lsn);
        self.images += 1;
        Ok(())
    }

    /// `RedoCtx::block(i)` of spec 8.6: four states.
    fn block<'p>(&'p mut self, rec: &WalRecord, i: usize) -> EResult<BlockState<'p>> {
        let b = &rec.blocks[i];
        let forced = b.flags & BLK_FORCED_IMAGE != 0;
        if b.image.is_some() && !forced {
            self.restore(rec, b)?;
            return Ok(BlockState::Restored);
        }
        if b.will_init() {
            self.smgr.ensure_len(b.tag.rel, b.tag.fork, b.tag.block + 1)?;
            let p = self.pool.pin(b.tag, ReadMode::Init)?;
            return Ok(BlockState::Redo(p));
        }
        if !self.smgr.fork_exists(b.tag.rel, b.tag.fork) {
            self.missing.insert(b.tag.rel);
            return Ok(BlockState::Missing);
        }
        self.smgr.ensure_len(b.tag.rel, b.tag.fork, b.tag.block + 1)?;
        let p = self.pool.pin(b.tag, ReadMode::ZeroOnMissing)?;
        let lsn = PageRef(&p.read()[..]).lsn();
        if lsn >= rec.end_lsn {
            // Redo relies on nothing it has not made durable itself (spec 9.3 "What redo may rely
            // on"): the page may have been written by a killed process and never synced, so it is
            // left dirty even though this record is skipped. The end-of-recovery checkpoint writes
            // and syncs it before the redo point moves.
            p.write().mark_dirty();
            return Ok(BlockState::UpToDate);
        }
        Ok(BlockState::Redo(p))
    }

    fn redo_smgr(&mut self, rec: &WalRecord) -> EResult<()> {
        let m = &rec.main;
        if m.len() < 9 {
            return Ok(());
        }
        let rel = rd_u64(m, 0);
        let Some(fork) = ForkId::from_u8(m[8]) else { return Ok(()) };
        match rec.info {
            smgr::CREATE => {
                // Carried out again and queued for sync, also if the fork exists (spec 9.3).
                self.smgr.recreate_fork(rel, fork)?;
            }
            smgr::EXTEND if m.len() >= 17 => {
                let (old, new) = (rd_u32(m, 9), rd_u32(m, 13));
                self.smgr.recreate_fork(rel, fork)?;
                // The length rises first: the pages below are written out as part of the extension.
                // The current length may already be larger, because pages were written before the
                // crash; it is never lowered (spec 7.8).
                self.smgr.ensure_len(rel, fork, new)?;
                for blk in old..new {
                    // Every page of the range is formatted again, without reading it, and left
                    // dirty (spec 7.8): a page that a killed process wrote but never synced is not
                    // relied on, and nothing is lost, because every later use of these pages lies
                    // in this redo range and rebuilds their content, starting with the `WILL_INIT`
                    // record of the first use.
                    let p = self.pool.pin(PageTag::new(rel, fork, blk), ReadMode::Init)?;
                    let mut w = p.write();
                    PageMut(&mut w[..]).init(PageKind::Unused, 0);
                    w.mark_dirty();
                    w.set_lsn(rec.end_lsn);
                }
            }
            smgr::TRUNCATE if m.len() >= 13 => {
                self.smgr.truncate_fork(rel, fork, rd_u32(m, 9))?;
            }
            _ => {}
        }
        Ok(())
    }

    fn redo_test(&mut self, rec: &WalRecord) -> EResult<()> {
        let check = self.check;
        let image = rec.blocks[0].image.clone();
        let forced = rec.blocks[0].flags & BLK_FORCED_IMAGE != 0;
        let init = rec.info == testop::INIT_SET_BYTES;
        let end = rec.end_lsn;
        let data = rec.blocks[0].data.clone();
        let st = self.block(rec, 0)?;
        let BlockState::Redo(p) = st else { return Ok(()) };
        let mut w = p.write();
        if check && forced {
            // Redo-consistency check (9.6): run the redo logic on the page as it is, then compare
            // with the image the original operation produced.
            let mut copy = page::new_page_buf();
            copy.copy_from_slice(&w[..]);
            apply_test(&mut copy[..], init, &data);
            let mut img = page::new_page_buf();
            if let Some((ho, hl, bytes)) = &image {
                page::restore_hole(*ho, *hl, bytes, &mut img[..]).map_err(|e| EngineError::Corruption(e.0))?;
            }
            mask(&mut copy[..]);
            mask(&mut img[..]);
            if copy[..] != img[..] {
                panic!("wal_consistency_check: redo of record at {} diverges from the logged image", rec.lsn);
            }
        }
        apply_test(&mut w[..], init, &data);
        w.mark_dirty();
        w.set_lsn(end);
        Ok(())
    }
}

/// What the heap's redo needs from the pages (spec 8.6, 9.6): the four states of `RedoCtx::block`, the
/// redo consistency check on a forced image, and the dirty flag and the LSN of a page that was redone.
impl RedoPages for Redo<'_> {
    fn apply_block(
        &mut self,
        rec: &WalRecord,
        i: usize,
        f: &mut dyn FnMut(&mut [u8; PAGE_SIZE]) -> Result<(), HeapRedoError>,
    ) -> Result<(), HeapRedoError> {
        let (check, end) = (self.check, rec.end_lsn);
        let forced = rec.blocks[i].flags & BLK_FORCED_IMAGE != 0;
        let image = rec.blocks[i].image.clone();
        let st = self.block(rec, i).map_err(|e| HeapRedoError::Pages(e.to_string()))?;
        let BlockState::Redo(p) = st else { return Ok(()) };
        let mut w = p.write();
        if check && forced {
            // Redo-consistency check (9.6): run the redo logic on the page as it is, then compare with the
            // image the original operation produced.
            let mut copy = page::new_page_buf();
            copy.copy_from_slice(&w[..]);
            f(&mut copy)?;
            let mut img = page::new_page_buf();
            if let Some((ho, hl, bytes)) = &image {
                page::restore_hole(*ho, *hl, bytes, &mut img[..]).map_err(|e| HeapRedoError::Corrupt(e.0))?;
            }
            mask(&mut copy[..]);
            mask(&mut img[..]);
            if copy[..] != img[..] {
                panic!("wal_consistency_check: redo of heap record at {} diverges from the logged image of block {i}", rec.lsn);
            }
        }
        f(&mut w)?;
        w.mark_dirty();
        w.set_lsn(end);
        Ok(())
    }
}

fn apply_test(page: &mut [u8], init: bool, data: &[u8]) {
    if init {
        PageMut(page).init(PageKind::Raw, 0);
    }
    let off = rd_u16(data, 0) as usize;
    let bytes = &data[2..];
    page[off..off + bytes.len()].copy_from_slice(bytes);
}

/// Masks bytes that may legitimately differ after redo: lsn, checksum and the unused gap.
fn mask(page: &mut [u8]) {
    page[..12].fill(0);
    let (lower, upper) = (rd_u16(page, 14) as usize, rd_u16(page, 16) as usize);
    if lower < upper && upper <= PAGE_SIZE {
        page[lower..upper].fill(0);
    }
}

#[allow(dead_code)]
type _Unused<'a> = PageWrite<'a>;
