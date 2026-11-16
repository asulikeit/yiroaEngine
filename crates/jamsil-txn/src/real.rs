//! The commit log and the CSN log on the buffer pool, and the log sink over the real WAL (spec 12.2,
//! 8.6, 8.7, 8.8). These are the production implementations of the two seams of `store`.
//!
//! Commit-log pages (relfile 1) are WAL-logged like any page. A `CLOG` zero-page record creates each
//! one, and the first change to a page after a checkpoint is preceded by a standalone image of the
//! page as it was, inserted under the page latch (spec 12.2). CSN-log pages (relfile 2) are created
//! with `ReadMode::Init`, never logged, and their files are removed at every start.

use crate::store::{ClogGate, Staged, TxnStore, WalSink};
use crate::TxnError;
use jamsil_bufmgr::{BufError, BufferPool, ReadMode, WalFlush, LOG_PAGES_PER_SEGMENT};
use jamsil_page::page::{PageKind, PageRef};
use jamsil_page::txnrec::*;
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::sync::latch::Class;
use jamsil_vfs::sync::Arc;
use jamsil_wal::{QuarantineKind, Wal, WalError};

/// What the store does when a page of the commit log cannot be read or fails its checks: media
/// damage that no caller can recover from (spec 9.4). It should not return.
pub type Fatal = Arc<dyn Fn(&str) + Send + Sync>;

/// What the store does when a page fails verification: quarantines the node, saying what clears it.
pub type Quarantine = Arc<dyn Fn(QuarantineKind, &str) + Send + Sync>;

pub fn wal_error(e: WalError) -> TxnError {
    match e {
        WalError::RecordTooLarge { len, max } => TxnError::RecordTooLarge { len, max },
        WalError::Quarantined(m) => TxnError::Quarantined(m),
        other => TxnError::Quarantined(other.to_string()),
    }
}

/// A flush that fails after the record was inserted: only a quarantine that began before the record
/// was handed to the file is a definite failure (spec 5.5).
fn flush_error(e: WalError) -> TxnError {
    match e {
        WalError::Quarantined(m) => TxnError::Quarantined(m),
        other => TxnError::OutcomeUnknown(other.to_string()),
    }
}

/// The WAL, as the buffer pool and the commit path see it.
#[derive(Clone)]
pub struct WalBridge(pub Arc<Wal>);

impl WalFlush for WalBridge {
    fn flush(&self, upto: Lsn) -> Result<(), BufError> {
        self.0.flush(upto).map_err(|e| BufError::Wal(e.to_string()))
    }
    fn insert_lsn(&self) -> Lsn {
        self.0.insert_lsn()
    }
    fn redo_lsn(&self) -> Lsn {
        self.0.redo_lsn()
    }
    fn permit_page_image(&self) -> Option<WalPermit> {
        self.0.check_size(&image_builder(PageTag::new(0, ForkId::Main, 0))).ok()
    }
    fn insert_page_image(&self, permit: WalPermit, tag: PageTag, page: &[u8]) -> Lsn {
        self.0.insert(permit, RecordBuilder::new(rmgr::XLOG, xlog::PAGE_IMAGE).block(BlockRef::new(tag).image(page))).end
    }
}

fn image_builder(tag: PageTag) -> RecordBuilder<'static> {
    RecordBuilder::new(rmgr::XLOG, xlog::PAGE_IMAGE).block(BlockRef::new(tag))
}

impl WalSink for WalBridge {
    fn gate_commit(&self, rec: &CommitRecord) -> Result<Staged, TxnError> {
        let bytes = rec.encode();
        let permit = self.0.check_size(&RecordBuilder::new(rmgr::XACT, xact::COMMIT).main(&bytes)).map_err(wal_error)?;
        Ok(Staged { permit, bytes })
    }
    fn gate_applied_advance(&self) -> Result<WalPermit, TxnError> {
        self.0.check_size(&RecordBuilder::new(rmgr::XACT, xact::APPLIED_ADVANCE).main(&[0u8; APPLIED_ADVANCE_LEN])).map_err(wal_error)
    }
    fn gate_abort(&self, rec: &AbortRecord) -> Result<Staged, TxnError> {
        let bytes = rec.encode();
        let permit = self.0.check_size(&RecordBuilder::new(rmgr::XACT, xact::ABORT).main(&bytes)).map_err(wal_error)?;
        Ok(Staged { permit, bytes })
    }
    fn insert_commit(&self, staged: Staged, xid: Xid) -> Csn {
        self.0.insert(staged.permit, RecordBuilder::new(rmgr::XACT, xact::COMMIT).xid(xid).main(&staged.bytes)).end
    }
    fn insert_applied_advance(&self, permit: WalPermit, pos: RaftPos) -> Csn {
        let main = encode_applied_advance(pos);
        self.0.insert(permit, RecordBuilder::new(rmgr::XACT, xact::APPLIED_ADVANCE).main(&main)).end
    }
    fn insert_abort(&self, staged: Staged, xid: Xid) -> Lsn {
        self.0.insert(staged.permit, RecordBuilder::new(rmgr::XACT, xact::ABORT).xid(xid).main(&staged.bytes)).end
    }
    fn log_alloc_batch(&self, batch: AllocBatch) -> Result<(), TxnError> {
        let main = batch.encode();
        let rec = RecordBuilder::new(rmgr::XLOG, xlog::ALLOC_BATCH).main(&main);
        let permit = self.0.check_size(&rec).map_err(wal_error)?;
        self.0.insert(permit, rec);
        Ok(())
    }
    fn flush(&self, upto: Lsn) -> Result<(), TxnError> {
        self.0.flush(upto).map_err(flush_error)
    }
}

/// The commit log and the CSN log on the buffer pool.
pub struct PoolStore {
    pool: Arc<BufferPool>,
    wal: Arc<Wal>,
    fatal: Fatal,
    quarantine: Quarantine,
}

fn clog_tag(block: BlockNo) -> PageTag {
    PageTag::new(CLOG_RELFILE, ForkId::Main, block)
}

fn csn_tag(block: BlockNo) -> PageTag {
    PageTag::new(CSNLOG_RELFILE, ForkId::Main, block)
}

impl PoolStore {
    /// `fatal` ends the process (a fatal I/O error, exit code 75); `quarantine` quarantines the store
    /// (a page that fails verification, `XX001`, spec 5.6): a restart would meet the same page, so
    /// ending the process would loop.
    pub fn new(pool: Arc<BufferPool>, wal: Arc<Wal>, fatal: Fatal, quarantine: Quarantine) -> Self {
        Self { pool, wal, fatal, quarantine }
    }

    /// The CSN log is emptied at every start (spec 12.2): after a restart every earlier xid precedes
    /// every new snapshot, and the commit log alone decides. Removes its files and forgets its pages.
    pub fn reset_csn_log(pool: &BufferPool) -> Result<(), TxnError> {
        let smgr = pool.smgr();
        pool.drop_relation(CSNLOG_RELFILE);
        let segs = smgr.log_segments(CSNLOG_RELFILE).map_err(|e| TxnError::Buffer(e.to_string()))?;
        if let Some(top) = segs.last() {
            smgr.unlink_log_segments_below(CSNLOG_RELFILE, top + 1).map_err(|e| TxnError::Buffer(e.to_string()))?;
        }
        smgr.set_nblocks(CSNLOG_RELFILE, ForkId::Main, 0);
        Ok(())
    }

    fn fail(&self, what: &str) -> ! {
        (self.fatal)(what);
        panic!("fatal: {what}");
    }

    /// What a failed page access means (spec 5.6): a fatal I/O error ends the process; a page that
    /// fails verification quarantines the store and fails the statement with `XX001`; anything else
    /// (no free buffer, a disk that is full) fails the statement.
    fn page_error(&self, what: &str, e: BufError) -> TxnError {
        match e {
            BufError::Io(io) if io.kind == jamsil_vfs::IoErrorKind::Fatal => self.fail(&format!("{what}: {io}")),
            BufError::Corrupt(m) => {
                let msg = format!("JDB-3001 {what}: {m}");
                // Corruption: only a snapshot install clears it, because a restart meets the same page.
                (self.quarantine)(QuarantineKind::Corruption, &msg);
                TxnError::Corrupt(msg)
            }
            other => TxnError::Buffer(format!("{what}: {other}")),
        }
    }

    /// Truncation of the commit log (spec 12.2, 14.6): every xid below `oldest_xid` has been hinted
    /// everywhere. A `CLOG` truncate record is inserted, then the segments wholly below the xid are
    /// unlinked and their pages forgotten. Returns the first xid whose segment is kept. The caller
    /// clamps `oldest_xid` to what a completed checkpoint captured (spec 14.6): this function does not.
    pub fn truncate_clog(&self, oldest_xid: Xid) -> Result<Xid, TxnError> {
        let Some(block) = clog_block(oldest_xid) else { return Ok(oldest_xid) };
        let seg = block / LOG_PAGES_PER_SEGMENT;
        if seg == 0 {
            return Ok(0);
        }
        let main = encode_clog_truncate(oldest_xid);
        let rec = RecordBuilder::new(rmgr::CLOG, clog::TRUNCATE).main(&main);
        let permit = self.wal.check_size(&rec).map_err(wal_error)?;
        self.wal.insert(permit, rec);
        self.unlink_below(seg).map_err(|e| {
            if e.kind == jamsil_vfs::IoErrorKind::Fatal {
                self.fail(&format!("commit-log truncation: {e}"));
            }
            TxnError::Buffer(e.to_string())
        })?;
        Ok(u64::from(seg * LOG_PAGES_PER_SEGMENT) * XIDS_PER_CLOG_PAGE)
    }

    fn unlink_below(&self, seg: u32) -> jamsil_vfs::IoResult<()> {
        self.pool.drop_blocks_below(CLOG_RELFILE, ForkId::Main, seg * LOG_PAGES_PER_SEGMENT);
        self.pool.smgr().unlink_log_segments_below(CLOG_RELFILE, seg)?;
        Ok(())
    }
}

impl TxnStore for PoolStore {
    fn clog_get(&self, xid: Xid) -> Result<ClogStatus, TxnError> {
        let Some(block) = clog_block(xid) else { return Ok(ClogStatus::InProgress) };
        match self.pool.smgr().nblocks(CLOG_RELFILE, ForkId::Main) {
            Ok(n) if block < n => {}
            Ok(_) => return Ok(ClogStatus::InProgress), // the page was never created
            Err(e) => return Err(self.page_error("commit log", e.into())),
        }
        let pin = self.pool.pin(clog_tag(block), ReadMode::Normal).map_err(|e| self.page_error(&format!("commit-log page {block}"), e))?;
        let page = pin.read_as(Class::Aux);
        clog_get(&page[..], xid).map_err(|e| self.page_error(&format!("commit-log page {block}"), BufError::Corrupt(e.0)))
    }

    fn clog_gate(&self, xids: &[Xid], csn: bool) -> Result<ClogGate, TxnError> {
        let mut blocks: Vec<BlockNo> = xids.iter().map(|x| clog_block(*x).expect("an addressable xid")).collect();
        blocks.sort_unstable();
        blocks.dedup();
        let mut permits = Vec::with_capacity(blocks.len());
        let mut clog_pins = Vec::with_capacity(blocks.len());
        for b in &blocks {
            // The pin is kept to step 6: nothing is pinned, read or allocated under the commit mutex.
            clog_pins.push(self.pool.pin_owned(clog_tag(*b), ReadMode::Normal).map_err(|e| self.page_error(&format!("commit-log page {b}"), e))?);
            permits.push(self.wal.check_size(&image_builder(clog_tag(*b))).map_err(wal_error)?);
        }
        let mut csn_blocks: Vec<BlockNo> = Vec::new();
        let mut csn_pins = Vec::new();
        if csn {
            csn_blocks = xids.iter().map(|x| csn_block(*x).expect("an addressable xid")).collect();
            csn_blocks.sort_unstable();
            csn_blocks.dedup();
            csn_pins.reserve(csn_blocks.len());
            for b in &csn_blocks {
                // An ordinary read: a page that was evicted is read back here, outside every lock.
                csn_pins.push(self.pool.pin_owned(csn_tag(*b), ReadMode::Normal).map_err(|e| self.page_error(&format!("CSN-log page {b}"), e))?);
            }
        }
        Ok(ClogGate::new(permits, blocks, clog_pins, csn_blocks, csn_pins))
    }

    fn clog_set_logged(&self, gate: &mut ClogGate, set: &[(Xid, ClogStatus)], log: &mut dyn FnMut() -> Lsn) -> Lsn {
        let parts = gate.parts();
        // Only latches of pages that are already pinned, and nothing allocated, in here. First every page
        // is marked dirty and, if it is the first change since the redo point, preceded by a standalone
        // image of the page as it is (spec 12.2): redo of the record then sets the bits unconditionally,
        // and a torn write can be repaired. The order is that of the write protocol (spec 7.5, 7.6): mark
        // dirty, then read the redo point.
        for (b, pin) in parts.blocks.iter().zip(parts.clog_pins) {
            let mut p = pin.write_as(Class::Aux);
            p.mark_dirty();
            if PageRef(&p[..]).lsn() <= self.wal.redo_lsn() {
                let permit = parts.permits.pop().expect("one permit for each page");
                let end = self.wal.insert(permit, RecordBuilder::new(rmgr::XLOG, xlog::PAGE_IMAGE).block(BlockRef::new(clog_tag(*b)).image(&p[..]))).end;
                p.set_lsn(end);
            }
        }
        // The pages were unlatched since they were marked, and a write-out may have cleaned them: mark
        // them again, before the record has a position in the log (spec 7.5). Named points for the
        // scenarios that put a write-out between the passes and ask what the frame looks like (19.7).
        jamsil_vfs::sched::point("clog:between_passes");
        for pin in parts.clog_pins {
            pin.mark_dirty();
        }
        jamsil_vfs::sched::point("clog:before_record");
        let lsn = log();
        // The bits, after the record. The page never reaches disk before the record that covers it
        // (invariant 4 of spec 12.12): the pool flushes the log up to the page LSN before it writes.
        for (b, pin) in parts.blocks.iter().zip(parts.clog_pins) {
            let mut p = pin.write_as(Class::Aux);
            for (xid, status) in set {
                if clog_block(*xid) == Some(*b) {
                    clog_set(&mut p[..], *xid, *status);
                }
            }
            p.mark_dirty();
            if PageRef(&p[..]).lsn() < lsn {
                p.set_lsn(lsn);
            }
        }
        lsn
    }

    fn clog_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        for block in first..=last {
            let tag = clog_tag(block);
            let rec = RecordBuilder::new(rmgr::CLOG, clog::ZERO_PAGE).block(BlockRef::new(tag).flags(BLK_WILL_INIT));
            let permit = self.wal.check_size(&rec).map_err(wal_error)?;
            // The length rises first, then the page is formatted and dirtied, then the record is
            // inserted (spec 7.8): the record is both the extension and the first use of the page.
            self.pool.smgr().ensure_len(CLOG_RELFILE, ForkId::Main, block + 1).map_err(|e| TxnError::Buffer(e.to_string()))?;
            let pin = self.pool.pin(tag, ReadMode::Init).map_err(|e| self.page_error(&format!("commit-log page {block}"), e))?;
            let mut page = pin.write_as(Class::Aux);
            init_clog_page(&mut page[..]);
            page.mark_dirty();
            let end = self.wal.insert(permit, rec).end;
            page.set_lsn(end);
        }
        Ok(())
    }

    fn csn_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        for block in first..=last {
            // Created in the pool first, so a reader that sees the new length finds the page; never
            // logged (spec 12.2). The pin is dropped at the end of the iteration.
            let pin = self.pool.pin(csn_tag(block), ReadMode::Init).map_err(|e| self.page_error(&format!("CSN-log page {block}"), e))?;
            let mut page = pin.write_as(Class::Aux);
            init_csn_page(&mut page[..]);
            page.mark_dirty();
            drop(page);
            self.pool.smgr().ensure_len(CSNLOG_RELFILE, ForkId::Main, block + 1).map_err(|e| TxnError::Buffer(e.to_string()))?;
        }
        Ok(())
    }

    fn csn_get(&self, xid: Xid) -> Result<Csn, TxnError> {
        let Some(block) = csn_block(xid) else { return Ok(0) };
        match self.pool.smgr().nblocks(CSNLOG_RELFILE, ForkId::Main) {
            Ok(n) if block < n => {}
            Ok(_) => return Ok(0),
            Err(e) => return Err(self.page_error("CSN log", e.into())),
        }
        let pin = self.pool.pin(csn_tag(block), ReadMode::Normal).map_err(|e| self.page_error(&format!("CSN-log page {block}"), e))?;
        let page = pin.read_as(Class::Aux);
        csn_get(&page[..], xid).map_err(|e| self.page_error(&format!("CSN-log page {block}"), BufError::Corrupt(e.0)))
    }

    fn csn_set(&self, gate: &ClogGate, xid: Xid, csn: Csn) {
        let Some(block) = csn_block(xid) else { return };
        // The gate pinned the page: this takes its latch and does no I/O, under the publication lock.
        let i = gate.csn_blocks().binary_search(&block).unwrap_or_else(|_| panic!("the commit of xid {xid} did not pin CSN-log page {block} at its gate"));
        let pin = &gate.csn_pin(i);
        let mut page = pin.write_as(Class::Aux);
        csn_set(&mut page[..], xid, csn);
        page.mark_dirty();
    }
}

// ---- redo (spec 9.3) ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RedoError {
    #[error("{0}")]
    Corrupt(String),
    #[error("{0}")]
    Buf(#[from] BufError),
    #[error("{0}")]
    Io(#[from] jamsil_vfs::IoError),
}

impl From<Corrupt> for RedoError {
    fn from(c: Corrupt) -> Self {
        RedoError::Corrupt(c.0)
    }
}

/// Redo of a commit or abort record: the bits of its xid, and of its sub-xids, are set
/// unconditionally (spec 12.2: the standalone image before the first change after a checkpoint is
/// what makes that safe). The header xid of an abort record is the sub-xid of a `ROLLBACK TO`, or the
/// top-level xid. The page is left dirty, also when it already holds the bits (spec 9.3 "What redo may
/// rely on"). An xid below `oldest_clog_xid` is a fatal inconsistency (spec 14.6).
pub fn redo_xact_bits(pool: &BufferPool, rec: &WalRecord, oldest_clog_xid: Xid) -> Result<(), RedoError> {
    let (status, subs): (ClogStatus, Vec<Xid>) = match rec.info {
        xact::COMMIT => (ClogStatus::Committed, CommitView::parse(&rec.main)?.sub_xids().collect()),
        xact::ABORT => (ClogStatus::Aborted, AbortRecord::decode(&rec.main)?.sub_xids),
        _ => return Ok(()),
    };
    for xid in std::iter::once(rec.xid).chain(subs) {
        // A checkpoint truncates only below the `oldest_active_xid` it captured (spec 14.6), so no
        // record that redo replays can name an xid of a removed segment. One that does shows a bug,
        // and skipping it would hide it.
        if xid < oldest_clog_xid {
            return Err(RedoError::Corrupt(format!(
                "fatal inconsistency: a replayed record names xid {xid}, below oldest_clog_xid {oldest_clog_xid}"
            )));
        }
        let block = clog_block(xid).ok_or_else(|| RedoError::Corrupt(format!("xid {xid} has no commit-log page")))?;
        let pin = pool.pin(clog_tag(block), ReadMode::Normal)?;
        let mut page = pin.write_as(Class::Aux);
        if PageRef(&page[..]).kind() != PageKind::Clog {
            return Err(RedoError::Corrupt(format!("commit-log page {block} has the wrong kind")));
        }
        clog_set(&mut page[..], xid, status);
        page.mark_dirty();
        if PageRef(&page[..]).lsn() < rec.end_lsn {
            page.set_lsn(rec.end_lsn);
        }
    }
    Ok(())
}

/// Redo of a `CLOG` zero-page record: the page is created with every xid in progress, without
/// reading it (spec 8.6).
pub fn redo_clog_zero_page(pool: &BufferPool, rec: &WalRecord) -> Result<(), RedoError> {
    let [b] = &rec.blocks[..] else { return Err(RedoError::Corrupt("a zero-page record has one block reference".into())) };
    if !b.will_init() || b.tag.rel != CLOG_RELFILE || b.tag.fork != ForkId::Main || !rec.main.is_empty() {
        return Err(RedoError::Corrupt("malformed commit-log zero-page record".into()));
    }
    pool.smgr().ensure_len(b.tag.rel, b.tag.fork, b.tag.block + 1)?;
    let pin = pool.pin(b.tag, ReadMode::Init)?;
    let mut page = pin.write_as(Class::Aux);
    init_clog_page(&mut page[..]);
    page.mark_dirty();
    page.set_lsn(rec.end_lsn);
    Ok(())
}

/// Redo of a `CLOG` truncate record: unlinks the segments wholly below `oldest_xid`. Returns it.
pub fn redo_clog_truncate(pool: &BufferPool, rec: &WalRecord) -> Result<Xid, RedoError> {
    let oldest = decode_clog_truncate(&rec.main)?;
    if let Some(block) = clog_block(oldest) {
        let seg = block / LOG_PAGES_PER_SEGMENT;
        pool.drop_blocks_below(CLOG_RELFILE, ForkId::Main, seg * LOG_PAGES_PER_SEGMENT);
        pool.smgr().unlink_log_segments_below(CLOG_RELFILE, seg)?;
    }
    Ok(oldest)
}

/// Redo of an allocation batch: the three marks (spec 8.6). The caller raises its counters to them.
pub fn redo_alloc_batch(rec: &WalRecord) -> Result<AllocBatch, RedoError> {
    Ok(AllocBatch::decode(&rec.main)?)
}
