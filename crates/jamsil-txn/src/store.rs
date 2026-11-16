//! The two seams of the transaction layer: where the commit log and the CSN log live, and where the
//! commit records go. `jamsil-txn::real` implements them over the buffer pool and the WAL; the tests of
//! this crate implement them in memory, which is what lets every interleaving run under the
//! deterministic scheduler (spec 19.2).

use crate::TxnError;
use jamsil_bufmgr::PinnedPage;
use jamsil_page::txnrec::{clog_block, csn_block, AbortRecord, AllocBatch, ClogStatus, CommitRecord, RaftPos};
use jamsil_page::walrec::WalPermit;
use jamsil_page::*;
use jamsil_vfs::sync::Mutex;
use std::collections::{BTreeSet, HashMap};

/// What a commit takes at its gate, before the commit mutex, and keeps until step 6 (spec 12.11): the
/// permits for the standalone page images that setting commit-log bits may need (spec 8.8, 12.2), and
/// the pins of the commit-log and CSN-log pages of its xid and sub-xids. Nothing is pinned, read or
/// allocated under the commit mutex or the publication lock: inside them the writer only takes the
/// latches of pages that are already pinned (spec 7.5). Dropping the gate releases the pins.
#[derive(Default)]
pub struct ClogGate {
    permits: Vec<WalPermit>,
    blocks: Vec<BlockNo>,
    clog_pins: Vec<PinnedPage<'static>>,
    csn_blocks: Vec<BlockNo>,
    csn_pins: Vec<PinnedPage<'static>>,
}

impl std::fmt::Debug for ClogGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClogGate").field("clog_blocks", &self.blocks).field("csn_blocks", &self.csn_blocks).field("permits", &self.permits.len()).finish()
    }
}

/// The parts of a gate, for the store that built it.
pub struct GateParts<'a> {
    pub permits: &'a mut Vec<WalPermit>,
    pub blocks: &'a [BlockNo],
    pub clog_pins: &'a [PinnedPage<'static>],
    pub csn_blocks: &'a [BlockNo],
    pub csn_pins: &'a [PinnedPage<'static>],
}

impl ClogGate {
    /// `blocks` are the distinct commit-log pages the change touches, in ascending order, with one pin
    /// and one permit for each; `csn_blocks` and `csn_pins` the same for the CSN log (empty for an abort,
    /// which writes no CSN).
    pub fn new(
        permits: Vec<WalPermit>,
        blocks: Vec<BlockNo>,
        clog_pins: Vec<PinnedPage<'static>>,
        csn_blocks: Vec<BlockNo>,
        csn_pins: Vec<PinnedPage<'static>>,
    ) -> Self {
        debug_assert!(blocks.len() == clog_pins.len() && csn_blocks.len() == csn_pins.len());
        Self { permits, blocks, clog_pins, csn_blocks, csn_pins }
    }
    pub fn blocks(&self) -> &[BlockNo] {
        &self.blocks
    }
    pub fn csn_blocks(&self) -> &[BlockNo] {
        &self.csn_blocks
    }
    /// The pin of the `i`-th CSN-log page of the gate.
    pub fn csn_pin(&self, i: usize) -> &PinnedPage<'static> {
        &self.csn_pins[i]
    }
    pub fn parts(&mut self) -> GateParts<'_> {
        GateParts { permits: &mut self.permits, blocks: &self.blocks, clog_pins: &self.clog_pins, csn_blocks: &self.csn_blocks, csn_pins: &self.csn_pins }
    }
}

/// A record that passed the gate: the permit and the encoded main payload, made before the critical
/// section, so that the insert inside it allocates nothing (spec 8.8).
#[derive(Debug)]
pub struct Staged {
    pub permit: WalPermit,
    pub bytes: Vec<u8>,
}

/// The commit log and the CSN log (spec 12.2), addressed by xid.
pub trait TxnStore: Send + Sync {
    /// How the xid ended. Callers never ask for an xid that is still in the active set
    /// (invariant 7 of spec 12.12): `TxnManager::status` checks the active set first. A page that
    /// was never created answers "in progress". A page that fails verification is `Corrupt`, and
    /// quarantines the store (spec 5.6).
    fn clog_get(&self, xid: Xid) -> Result<ClogStatus, TxnError>;
    /// At the gate, before any critical section: takes the permits that `clog_set_logged` will need
    /// for these xids and pins their commit-log pages, and, if `csn` is set, the CSN-log pages that the
    /// barrier will write; or fails with the quarantine error when the log no longer accepts
    /// records. May wait for buffer space in the log and read pages from disk. The pins live as long as
    /// the gate does.
    fn clog_gate(&self, xids: &[Xid], csn: bool) -> Result<ClogGate, TxnError>;
    /// Sets the bits of `set` in the same critical section as the record that covers them. `log`
    /// inserts that record (it cannot fail: its permit was taken before) and returns its end LSN; an
    /// implementation marks the pages dirty and inserts the standalone image of each page that needs
    /// one before the record, and sets the bits after it, taking only the latches of the pages the gate
    /// pinned: it pins, reads and allocates nothing. Every page's LSN becomes the record's end, so a
    /// page never reaches disk before its record (invariant 4 of spec 12.12). Step 3 of the commit
    /// calls it under the commit mutex, so a checkpoint can never persist the page without the bit and
    /// start redo after the record (spec 12.8).
    fn clog_set_logged(&self, gate: &mut ClogGate, set: &[(Xid, ClogStatus)], log: &mut dyn FnMut() -> Lsn) -> Lsn;
    /// Creates the commit-log pages `first..=last`, each by a `CLOG` zero-page record that is both the
    /// extension and the first use of the page (spec 12.2). Called on the slow path of xid
    /// allocation, never under the publication lock or the commit mutex (spec 8.6).
    fn clog_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError>;
    /// Creates the CSN-log pages `first..=last` with `Init`, never logged, and unpins them (spec 12.2).
    /// The same slow path as `clog_create_pages`, so that two commits can never initialise one page.
    fn csn_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError>;
    /// The CSN of a committed xid, or 0 if none was written (never, or the log was emptied).
    fn csn_get(&self, xid: Xid) -> Result<Csn, TxnError>;
    /// Step 5 of the commit calls it inside the publication barrier, on a page that the gate pinned:
    /// it does no I/O.
    fn csn_set(&self, gate: &ClogGate, xid: Xid, csn: Csn);
}

/// The log, as the commit path sees it. Records are inserted under the commit mutex, after a permit
/// was taken before it (spec 8.8).
pub trait WalSink: Send + Sync {
    /// The gate before the critical section for a commit record of this size, with the record encoded.
    fn gate_commit(&self, rec: &CommitRecord) -> Result<Staged, TxnError>;
    /// The gate for an applied-advance record.
    fn gate_applied_advance(&self) -> Result<WalPermit, TxnError>;
    /// The gate for an abort record. An abort is not flushed, and a crash aborts every xid without a
    /// commit record (spec 9.3), so a refusal does not stop the abort: the manager remembers it in
    /// memory (spec 12.8).
    fn gate_abort(&self, rec: &AbortRecord) -> Result<Staged, TxnError>;
    /// Inserts the commit record of `xid` and returns its end LSN, which is the commit sequence
    /// number. Called under the commit mutex; allocates nothing.
    fn insert_commit(&self, staged: Staged, xid: Xid) -> Csn;
    /// Inserts an applied-advance record and returns its end LSN.
    fn insert_applied_advance(&self, permit: WalPermit, pos: RaftPos) -> Csn;
    /// Inserts an abort record whose header xid is `xid` and returns its end LSN. Not flushed.
    fn insert_abort(&self, staged: Staged, xid: Xid) -> Lsn;
    /// Gates and inserts an allocation-batch record (spec 8.6). The slow path of allocation calls
    /// it, under the mutex of the allocator, never under the publication lock. Not flushed: any
    /// durable trace of an id lies after its batch record in the log.
    fn log_alloc_batch(&self, batch: AllocBatch) -> Result<(), TxnError>;
    /// Blocks until the log is durable up to `upto`.
    fn flush(&self, upto: Lsn) -> Result<(), TxnError>;
}

#[derive(Default)]
struct MemStoreState {
    clog: HashMap<Xid, ClogStatus>,
    csn: HashMap<Xid, Csn>,
    pages: BTreeSet<BlockNo>,
    csn_pages: BTreeSet<BlockNo>,
    created: Vec<(BlockNo, BlockNo)>,
    csn_created: Vec<(BlockNo, BlockNo)>,
}

/// An in-memory commit log and CSN log. It enforces the rule of spec 8.6: the page of an xid exists
/// before the xid is handed out, so setting the bits of an xid whose page was not created panics. The
/// same holds for the CSN log (spec 12.2).
#[derive(Default)]
pub struct MemStore {
    st: Mutex<MemStoreState>,
}

impl MemStore {
    pub fn new() -> Self {
        Self::default()
    }
    /// The commit-log blocks created so far.
    pub fn pages(&self) -> Vec<BlockNo> {
        self.st.lock().pages.iter().copied().collect()
    }
    /// The CSN-log blocks created so far.
    pub fn csn_pages(&self) -> Vec<BlockNo> {
        self.st.lock().csn_pages.iter().copied().collect()
    }
    /// The ranges `clog_create_pages` was called with, in order.
    pub fn created_by_calls(&self) -> Vec<(BlockNo, BlockNo)> {
        self.st.lock().created.clone()
    }
    /// The ranges `csn_create_pages` was called with, in order.
    pub fn csn_created_by_calls(&self) -> Vec<(BlockNo, BlockNo)> {
        self.st.lock().csn_created.clone()
    }
    /// Test helper: sets a bit with no record (for xids a test makes up).
    pub fn force(&self, xid: Xid, status: ClogStatus) {
        self.st.lock().clog.insert(xid, status);
    }
    /// Test helper: sets a CSN with no page and no gate.
    pub fn force_csn(&self, xid: Xid, csn: Csn) {
        self.st.lock().csn.insert(xid, csn);
    }
}

impl TxnStore for MemStore {
    fn clog_get(&self, xid: Xid) -> Result<ClogStatus, TxnError> {
        Ok(self.st.lock().clog.get(&xid).copied().unwrap_or(ClogStatus::InProgress))
    }
    fn clog_gate(&self, _xids: &[Xid], _csn: bool) -> Result<ClogGate, TxnError> {
        Ok(ClogGate::default())
    }
    fn clog_set_logged(&self, _gate: &mut ClogGate, set: &[(Xid, ClogStatus)], log: &mut dyn FnMut() -> Lsn) -> Lsn {
        let lsn = log();
        let mut st = self.st.lock();
        for (xid, status) in set {
            let b = clog_block(*xid).expect("an addressable xid");
            assert!(st.pages.contains(&b), "xid {xid} was handed out before its commit-log page {b} was created (spec 8.6)");
            st.clog.insert(*xid, *status);
        }
        lsn
    }
    fn clog_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        let mut st = self.st.lock();
        st.pages.extend(first..=last);
        st.created.push((first, last));
        Ok(())
    }
    fn csn_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        let mut st = self.st.lock();
        st.csn_pages.extend(first..=last);
        st.csn_created.push((first, last));
        Ok(())
    }
    fn csn_get(&self, xid: Xid) -> Result<Csn, TxnError> {
        Ok(self.st.lock().csn.get(&xid).copied().unwrap_or(0))
    }
    fn csn_set(&self, _gate: &ClogGate, xid: Xid, csn: Csn) {
        let mut st = self.st.lock();
        let b = csn_block(xid).expect("an addressable xid");
        assert!(st.csn_pages.contains(&b), "xid {xid} was handed out before its CSN-log page {b} was created (spec 12.2)");
        st.csn.insert(xid, csn);
    }
}

/// What `MemWal` was asked to do, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoggedRecord {
    Commit { xid: Xid, rec: CommitRecord, end: Lsn },
    AppliedAdvance { pos: RaftPos, end: Lsn },
    Abort { xid: Xid, rec: AbortRecord },
    AllocBatch(AllocBatch),
}

#[derive(Default)]
struct MemWalState {
    next_lsn: Lsn,
    durable: Lsn,
    records: Vec<LoggedRecord>,
    fail_flush: bool,
    closed: bool,
    flushes: u64,
}

/// An in-memory log: LSNs grow by 100 per record, `flush` makes everything inserted durable.
pub struct MemWal {
    st: Mutex<MemWalState>,
}

impl Default for MemWal {
    fn default() -> Self {
        Self { st: Mutex::new(MemWalState { next_lsn: 1000, ..Default::default() }) }
    }
}

impl MemWal {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn records(&self) -> Vec<LoggedRecord> {
        self.st.lock().records.clone()
    }
    pub fn durable_lsn(&self) -> Lsn {
        self.st.lock().durable
    }
    /// The position after the last record inserted: what a checkpoint takes as `redo_lsn`.
    pub fn insert_lsn(&self) -> Lsn {
        self.st.lock().next_lsn
    }
    /// How many `flush` calls succeeded.
    pub fn flushes(&self) -> u64 {
        self.st.lock().flushes
    }
    /// The next `flush` calls fail, as a failed write would.
    pub fn set_fail_flush(&self, fail: bool) {
        self.st.lock().fail_flush = fail;
    }
    /// The log no longer accepts records: every gate fails.
    pub fn close(&self) {
        self.st.lock().closed = true;
    }
    /// The log accepts records again (a test that models a recovered or replaced log).
    pub fn reopen(&self) {
        self.st.lock().closed = false;
    }
    fn gate(&self, len: usize) -> Result<WalPermit, TxnError> {
        if self.st.lock().closed {
            return Err(TxnError::Quarantined("log closed".into()));
        }
        Ok(WalPermit::issue(len))
    }
}

impl WalSink for MemWal {
    fn gate_commit(&self, rec: &CommitRecord) -> Result<Staged, TxnError> {
        Ok(Staged { permit: self.gate(rec.encoded_len())?, bytes: rec.encode() })
    }
    fn gate_applied_advance(&self) -> Result<WalPermit, TxnError> {
        self.gate(16)
    }
    fn gate_abort(&self, rec: &AbortRecord) -> Result<Staged, TxnError> {
        Ok(Staged { permit: self.gate(rec.encoded_len())?, bytes: rec.encode() })
    }
    fn insert_commit(&self, staged: Staged, xid: Xid) -> Csn {
        let rec = CommitRecord::decode(&staged.bytes).expect("the gate encoded it");
        let mut st = self.st.lock();
        st.next_lsn += 100;
        let end = st.next_lsn;
        st.records.push(LoggedRecord::Commit { xid, rec, end });
        end
    }
    fn insert_applied_advance(&self, _permit: WalPermit, pos: RaftPos) -> Csn {
        let mut st = self.st.lock();
        st.next_lsn += 100;
        let end = st.next_lsn;
        st.records.push(LoggedRecord::AppliedAdvance { pos, end });
        end
    }
    fn insert_abort(&self, staged: Staged, xid: Xid) -> Lsn {
        let rec = AbortRecord::decode(&staged.bytes).expect("the gate encoded it");
        let mut st = self.st.lock();
        st.next_lsn += 100;
        st.records.push(LoggedRecord::Abort { xid, rec });
        st.next_lsn
    }
    fn log_alloc_batch(&self, batch: AllocBatch) -> Result<(), TxnError> {
        self.gate(24)?;
        let mut st = self.st.lock();
        st.next_lsn += 100;
        st.records.push(LoggedRecord::AllocBatch(batch));
        Ok(())
    }
    fn flush(&self, upto: Lsn) -> Result<(), TxnError> {
        let mut st = self.st.lock();
        if st.fail_flush {
            return Err(TxnError::OutcomeUnknown("flush failed".into()));
        }
        st.durable = st.durable.max(upto);
        st.flushes += 1;
        Ok(())
    }
}
