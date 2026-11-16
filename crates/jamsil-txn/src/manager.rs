//! The transaction manager (spec 12.8, 12.11, 17.4): xid assignment, snapshots, commit and abort, the
//! four positions of the sequencer, and the queue behind `durable_applied_index`.

use crate::alloc::Allocator;
use crate::lock::{LockError, LockManager, LockMode, LockOwner, LockTag, WaitPolicy, WoundReason};
use crate::published::{Barrier, GateRequest, GateVerdict, Published, PublishedState};
use crate::sessions::{Horizon, SessionSlots, SlotId};
use crate::snapshot::{OwnTxn, Snapshot};
use crate::store::{ClogGate, TxnStore, WalSink};
use crate::TxnError;
use jamsil_page::txnrec::{AbortRecord, AllocBatch, ClogStatus, CommitRecord, RaftPos};
use jamsil_page::*;
use jamsil_vfs::env::Clock;
use jamsil_vfs::sync::atomic::{AtomicBool, AtomicU64, STORE_BEFORE_LOAD};
use jamsil_vfs::sync::{Arc, Condvar, Mutex};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XidStatus {
    InProgress,
    Committed,
    Aborted,
}

#[derive(Debug, Clone)]
pub struct TxnOptions {
    /// `storage.synchronous`: true = FULL, false = NORMAL (spec 17.4).
    pub synchronous_full: bool,
    /// The size of the session-slot array: how many snapshots can be held at once.
    pub sessions: usize,
    /// How long a wait for a lock lasts before `55P03` (`storage.write_lock_timeout_ms`, default
    /// 30,000; spec 18.1). It also bounds a held-back `TxPrepare` and a wait for a read gate.
    pub lock_timeout: Duration,
    /// How long a waiter sleeps before it searches for a deadlock (`storage.deadlock_timeout_ms`).
    pub deadlock_timeout: Duration,
    /// How many xids one allocation batch covers (spec 8.6: 4,096). A smaller value is for tests that
    /// want many batch boundaries; production uses the default.
    pub xid_batch: u64,
    /// Debug builds assert invariant 1 of spec 12.12 at the barrier: `applied_index` grows by exactly
    /// one per replicated commit or applied-advance. The sequencer's admission check (spec 12.8 step 1)
    /// guarantees it; a caller that has no sequencer yet turns the check off.
    pub check_continuity: bool,
}

impl Default for TxnOptions {
    fn default() -> Self {
        Self {
            synchronous_full: true,
            sessions: 1024,
            lock_timeout: Duration::from_millis(30_000),
            deadlock_timeout: Duration::from_millis(1000),
            xid_batch: crate::alloc::XID_BATCH,
            check_continuity: true,
        }
    }
}

/// What recovery established (spec 9.3), and so where the manager starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recovered {
    /// The first id of each kind not covered by any allocation batch in the log. Every xid below
    /// `next_xid` that has no commit record is aborted (the crash rule of spec 9.3 step 7).
    pub next_xid: Xid,
    pub next_oid: u32,
    pub next_relfile: RelFileId,
    /// The CSN of the last published commit.
    pub csn: Csn,
    /// The Raft index and term of the last commit or applied-advance record in the valid log.
    pub applied: u64,
    pub applied_term: u64,
    pub schema_version: u64,
}

impl Default for Recovered {
    fn default() -> Self {
        Self { next_xid: FIRST_NORMAL_XID, next_oid: FIRST_USER_OID, next_relfile: FIRST_RELFILE, csn: 0, applied: 0, applied_term: 0, schema_version: 0 }
    }
}

/// What the sequencer says about an entry that arrives (spec 17.2, 12.8 step 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// `index == admitted_index + 1`: accepted into the pipeline.
    Admitted,
    /// `index <= applied_index`: already applied; skip it.
    AlreadyApplied,
    /// Above `applied_index` but not above `admitted_index`: already in flight. A repeated delivery
    /// waits for its publication and then reports it as applied.
    InFlight,
    /// `index > admitted_index + 1`: a gap (`57P03`); nothing is applied.
    Gap,
}

/// The four positions of the sequencer (spec 12.8), read at one moment each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Positions {
    /// The last entry accepted into the pipeline.
    pub admitted_index: u64,
    /// The last entry whose commit or applied-advance record is inserted.
    pub logged_position: u64,
    /// The last entry published by the barrier.
    pub applied_index: u64,
    /// The last entry whose record is synced.
    pub durable_applied_index: u64,
}

/// What `prepare` returns: the `schema_version` read in step 4 of the gate, under the publication
/// lock, for the caller to stamp the change set with (spec 12.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrepareInfo {
    pub schema_version: u64,
}

/// The isolation levels of spec 12.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Isolation {
    /// A new snapshot for each statement; the default.
    ReadCommitted,
    /// One snapshot for the transaction; `40001` on a write conflict with a committed transaction.
    RepeatableRead,
    /// As repeatable read, plus the commit gate.
    Serializable,
}

/// One transaction. It has no xid until its first write.
#[derive(Debug)]
pub struct Txn {
    xid: Option<Xid>,
    slot: Option<SlotId>,
    command: u32,
    isolation: Isolation,
    /// Names the transaction in the Prepared set.
    token: u64,
    /// The identity the lock manager knows the transaction by.
    owner: LockOwner,
    /// The one snapshot of a REPEATABLE READ or SERIALIZABLE transaction.
    fixed: Option<Snapshot>,
    /// The tables read (recorded only at SERIALIZABLE) and written.
    read_tables: BTreeSet<Oid>,
    write_tables: BTreeSet<Oid>,
    /// The transaction contains DDL.
    ddl: bool,
    prepared: bool,
    /// How long a wait for a lock of this transaction lasts: set from each statement (`TxExec`), and
    /// the manager's `storage.write_lock_timeout_ms` when that is zero. `prepare` uses the value of the
    /// last statement (spec 12.11).
    lock_timeout: Duration,
    /// Live sub-xids; the commit record lists them.
    pub sub_xids: Vec<Xid>,
    /// The parent of each sub-xid that is not a child of the top-level transaction (spec 12.9: the
    /// parent-child map lives in memory only).
    parents: HashMap<Xid, Xid>,
    /// Relation files to unlink when this commits.
    pub unlink: Vec<RelFileId>,
    /// Relation files this transaction created; an abort removes them.
    pub created: Vec<RelFileId>,
    /// `Some` if the transaction changed the schema.
    pub schema_version: Option<u64>,
}

impl Txn {
    pub fn xid(&self) -> Option<Xid> {
        self.xid
    }
    /// The next command number: it continues across statements and never goes back.
    pub fn next_command(&mut self) -> u32 {
        self.command += 1;
        self.command
    }
    pub fn command(&self) -> u32 {
        self.command
    }
    pub fn isolation(&self) -> Isolation {
        self.isolation
    }
    pub fn owner(&self) -> &LockOwner {
        &self.owner
    }
    /// The lock timeout of the transaction's last statement.
    pub fn lock_timeout(&self) -> Duration {
        self.lock_timeout
    }
    /// A table this transaction read. Only a SERIALIZABLE transaction records reads (spec 12.6).
    pub fn note_read(&mut self, table: Oid) {
        if self.isolation == Isolation::Serializable {
            self.read_tables.insert(table);
        }
    }
    pub fn note_write(&mut self, table: Oid) {
        self.write_tables.insert(table);
    }
    /// The transaction contains DDL (spec 15.3).
    pub fn mark_ddl(&mut self) {
        self.ddl = true;
    }
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }
    /// The live sub-xids below `sub`, children before grandchildren.
    fn descendants_of(&self, sub: Xid) -> Vec<Xid> {
        let mut out = vec![];
        let mut frontier = vec![sub];
        while let Some(p) = frontier.pop() {
            for s in &self.sub_xids {
                if self.parents.get(s) == Some(&p) && !out.contains(s) {
                    out.push(*s);
                    frontier.push(*s);
                }
            }
        }
        out
    }
    fn own(&self) -> Option<OwnTxn> {
        self.xid.map(|xid| OwnTxn { xid, sub_xids: self.sub_xids.clone(), command: self.command })
    }
}

/// What the commit mutex guards (spec 9.2 step 2: a checkpoint reads these in the same hold in
/// which it takes `redo_lsn`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Logged {
    /// The Raft index of the last commit or applied-advance record inserted, published or not.
    pub logged_position: u64,
    /// Its term.
    pub logged_term: u64,
    /// The schema version of the last commit record that carried one.
    pub logged_schema_version: u64,
}

struct CommitState {
    logged: Logged,
    next_ticket: u64,
}

/// Pairs of (end LSN, Raft position) of the records not yet known durable; `on_durable` pops every
/// pair at or below the durable position and publishes the last as `durable_applied_index` (spec
/// 17.4). Under `FULL` the two indexes are equal whenever `Apply` has replied.
pub struct DurableTracker {
    queue: Mutex<VecDeque<(Lsn, u64)>>,
    durable_applied: AtomicU64,
}

impl DurableTracker {
    pub fn new(durable_applied: u64) -> Self {
        Self { queue: Mutex::new(VecDeque::new()), durable_applied: AtomicU64::new(durable_applied) }
    }
    pub fn push(&self, end: Lsn, pos: RaftPos) {
        self.queue.lock().push_back((end, pos.index));
    }
    /// Called when the log became durable up to `lsn`.
    pub fn on_durable(&self, lsn: Lsn) {
        let mut q = self.queue.lock();
        let mut last = None;
        while q.front().is_some_and(|(end, _)| *end <= lsn) {
            last = q.pop_front().map(|(_, index)| index);
        }
        if let Some(index) = last {
            self.durable_applied.fetch_max(index, STORE_BEFORE_LOAD);
        }
    }
    pub fn durable_applied_index(&self) -> u64 {
        self.durable_applied.load(STORE_BEFORE_LOAD)
    }
}

pub struct TxnManager {
    published: Published,
    slots: SessionSlots,
    store: Arc<dyn TxnStore>,
    wal: Arc<dyn WalSink>,
    clock: Arc<dyn Clock>,
    full: bool,
    check_continuity: bool,
    /// `admitted_index` (spec 12.8): the sequencer's continuity check on arrival.
    admitted: Mutex<u64>,
    commit: Mutex<CommitState>,
    /// The publication queue: commits are published in the order their records were inserted.
    queue: Mutex<VecDeque<u64>>,
    queue_cv: Condvar,
    durable: DurableTracker,
    locks: LockManager,
    lock_timeout: Duration,
    next_token: AtomicU64,
    /// Counts the changes of the Prepared set; DDL waits on it (spec 12.6).
    epoch: AtomicU64,
    quiesce: Mutex<()>,
    quiesce_cv: Condvar,
    /// xids, oids and relfile ids, in logged batches (spec 8.6).
    alloc: Allocator,
    /// Every xid below this that has no commit record is aborted (the crash rule, spec 9.3 step 7),
    /// applied at lookup time, so recovery writes nothing.
    crash_horizon: Xid,
    /// Aborts whose record the log did not accept. The commit-log bit cannot change without a record
    /// (invariant 5 of spec 7.10), so `status` consults this set before the commit log until the
    /// restart, whose crash rule gives the same verdict.
    unlogged_aborts: Mutex<BTreeSet<Xid>>,
    has_unlogged_aborts: AtomicBool,
}

/// What `checkpoint_capture` read in one hold of the commit mutex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capture<R> {
    pub redo: R,
    pub logged: Logged,
    /// `next_xid` and the oldest active xid, as published at that moment.
    pub published: PublishedState,
    /// The three allocation high-water marks (spec 9.2 step 2): each the first id not covered by a
    /// logged batch.
    pub alloc: AllocBatch,
}

/// A commit whose record is in the log: it resolves at step 6 (spec 12.11). Dropping it unresolved
/// resolves it (the record is in the log and the ones behind it wait in the queue), except while the
/// thread is panicking, when it leaves the queue.
#[must_use = "a commit ticket must be waited for"]
pub struct CommitTicket<'m> {
    mgr: &'m TxnManager,
    pending: Option<Pending>,
    csn: Csn,
    queue_ticket: u64,
    pos: Option<RaftPos>,
}

enum Pending {
    /// A transaction with an xid: its commit record is in the log.
    Txn { txn: Txn, xid: Xid, gate: ClogGate },
    /// An applied-advance record, for a transaction that wrote nothing or for a Raft no-op.
    Advance(Option<Txn>),
    /// Nothing was logged: a transaction that wrote nothing and has no Raft position.
    Nothing,
}

/// Why `commit` handed the transaction back: the gate of spec 8.8 refused before anything was
/// logged. The transaction is exactly as it was, still Prepared, with its locks held.
#[derive(Debug)]
pub struct CommitRefused {
    pub txn: Txn,
    pub error: TxnError,
}

impl CommitTicket<'_> {
    /// The commit sequence number: the end LSN of the commit record. Known once step 3 is done.
    pub fn csn(&self) -> Csn {
        self.csn
    }

    /// Steps 4 to 6: flush the log up to the record (under `FULL`), publish in queue order, release.
    pub fn wait(self) -> Result<Csn, TxnError> {
        let mgr = self.mgr;
        mgr.resolve_all(vec![self]).pop().expect("one ticket, one verdict")
    }
}

impl Drop for CommitTicket<'_> {
    fn drop(&mut self) {
        if self.pending.is_none() {
            return;
        }
        if std::thread::panicking() {
            self.mgr.withdraw(self.queue_ticket);
            return;
        }
        // A dropped ticket is a programming error (spec 12.11). A debug build panics, after it has taken
        // the ticket out of the queue so that the commits behind it do not stall; a release build
        // resolves it, so the publication queue cannot stall, and logs an error. A flush that fails
        // here cannot report its unknown outcome to anyone: no correct caller reaches this path.
        if cfg!(debug_assertions) {
            self.mgr.withdraw(self.queue_ticket);
            panic!("a CommitTicket was dropped without being waited for (spec 12.11)");
        }
        eprintln!("jamsil-txn: error: a CommitTicket was dropped without being waited for; resolving it");
        let me = CommitTicket { mgr: self.mgr, pending: self.pending.take(), csn: self.csn, queue_ticket: self.queue_ticket, pos: self.pos };
        let _ = self.mgr.resolve_all(vec![me]);
    }
}

impl TxnManager {
    pub fn new(store: Arc<dyn TxnStore>, wal: Arc<dyn WalSink>, clock: Arc<dyn Clock>, opts: TxnOptions, rec: Recovered) -> Self {
        let clock2 = clock.clone();
        Self {
            published: Published::with_schema_version(rec.next_xid, rec.csn, rec.applied, rec.schema_version),
            slots: SessionSlots::new(opts.sessions),
            store,
            wal,
            clock,
            full: opts.synchronous_full,
            check_continuity: opts.check_continuity,
            admitted: Mutex::new(rec.applied),
            commit: Mutex::new(CommitState { logged: Logged { logged_position: rec.applied, logged_term: rec.applied_term, logged_schema_version: rec.schema_version }, next_ticket: 0 }),
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            durable: DurableTracker::new(rec.applied),
            locks: LockManager::new(clock2, opts.deadlock_timeout),
            lock_timeout: opts.lock_timeout,
            next_token: AtomicU64::new(1),
            epoch: AtomicU64::new(0),
            quiesce: Mutex::new(()),
            quiesce_cv: Condvar::new(),
            alloc: Allocator::with_xid_batch(AllocBatch { next_xid: rec.next_xid, next_oid: rec.next_oid, next_relfile: rec.next_relfile }, opts.xid_batch),
            crash_horizon: rec.next_xid,
            unlogged_aborts: Mutex::new(BTreeSet::new()),
            has_unlogged_aborts: AtomicBool::new(false),
        }
    }

    // ---- the sequencer (spec 12.8 step 1, 17.2) ------------------------------------------------

    /// Step 1 of a commit: the sequencer admits entries in log order, so `index` must equal
    /// `admitted_index + 1`. Answers by the table of spec 17.2. The applier is one thread; a repeated
    /// delivery that finds its entry in flight waits for the publication, which is the caller's job.
    pub fn admit(&self, index: u64) -> Admission {
        let mut admitted = self.admitted.lock();
        if index <= self.published.read().applied_index {
            Admission::AlreadyApplied
        } else if index <= *admitted {
            Admission::InFlight
        } else if index == *admitted + 1 {
            *admitted = index;
            Admission::Admitted
        } else {
            Admission::Gap
        }
    }

    /// If an admitted entry fails before its record is inserted, `admitted_index` falls back to
    /// `logged_position` (spec 12.8): the entries behind it are withdrawn, and the next delivery starts
    /// from the first one that has no record.
    pub fn withdraw_admitted(&self) {
        let logged = self.logged().logged_position;
        *self.admitted.lock() = logged;
    }

    /// The four positions. `admitted_index >= logged_position >= applied_index` always holds, and under
    /// `FULL`, `durable_applied_index` is never below `applied_index` once `Apply` has replied.
    pub fn positions(&self) -> Positions {
        // Read in the order that cannot show a later position ahead of an earlier one: the one that
        // moves last first.
        let applied_index = self.published.read().applied_index;
        let logged_position = self.logged().logged_position;
        let admitted_index = *self.admitted.lock();
        Positions { admitted_index, logged_position, applied_index, durable_applied_index: self.durable.durable_applied_index() }
    }

    // ---- reads ---------------------------------------------------------------------------------

    pub fn published(&self) -> &Published {
        &self.published
    }
    pub fn slots(&self) -> &SessionSlots {
        &self.slots
    }
    pub fn locks(&self) -> &LockManager {
        &self.locks
    }
    pub fn allocator(&self) -> &Allocator {
        &self.alloc
    }
    pub fn logged(&self) -> Logged {
        self.commit.lock().logged
    }
    pub fn durable_applied_index(&self) -> u64 {
        self.durable.durable_applied_index()
    }
    /// Tells the tracker that the log is durable up to `lsn` (the WAL writer calls it after a sync).
    pub fn on_wal_durable(&self, lsn: Lsn) {
        self.durable.on_durable(lsn);
    }

    /// How an xid ended: the active set first, and the commit log only after that, so no thread
    /// reads the commit log for an xid that is still active (invariant 7 of spec 12.12). An xid
    /// leaves the active set only after its commit-log bits are final, or, when the log accepted no
    /// abort record, after it was remembered in `unlogged_aborts`. An xid below the crash horizon
    /// that the commit log does not show as committed is aborted (spec 9.3 step 7).
    pub fn status(&self, xid: Xid) -> Result<XidStatus, TxnError> {
        if self.published.is_active(xid) {
            return Ok(XidStatus::InProgress);
        }
        if self.has_unlogged_aborts.load(STORE_BEFORE_LOAD) && self.unlogged_aborts.lock().contains(&xid) {
            return Ok(XidStatus::Aborted);
        }
        Ok(match self.store.clog_get(xid)? {
            ClogStatus::Committed => XidStatus::Committed,
            ClogStatus::Aborted => XidStatus::Aborted,
            ClogStatus::InProgress if xid >= FIRST_NORMAL_XID && xid < self.crash_horizon => XidStatus::Aborted,
            ClogStatus::InProgress => XidStatus::InProgress,
        })
    }

    /// The CSN of a committed xid, if the CSN log still holds it.
    pub fn commit_csn(&self, xid: Xid) -> Result<Option<Csn>, TxnError> {
        if self.status(xid)? != XidStatus::Committed {
            return Ok(None);
        }
        Ok(Some(self.store.csn_get(xid)?).filter(|c| *c != 0))
    }

    pub fn horizon(&self) -> Horizon {
        self.slots.horizon(&self.published)
    }

    // ---- transactions --------------------------------------------------------------------------

    /// Sets the lock timeout of the transaction from the statement that is about to run (`TxExec.
    /// lock_timeout_ms`, spec 13.4): zero means the manager's default (`storage.write_lock_timeout_ms`).
    pub fn set_lock_timeout(&self, txn: &mut Txn, ms: u64) {
        txn.lock_timeout = if ms == 0 { self.lock_timeout } else { Duration::from_millis(ms) };
    }

    pub fn begin(&self) -> Txn {
        self.begin_with(Isolation::ReadCommitted)
    }

    pub fn begin_with(&self, isolation: Isolation) -> Txn {
        let token = self.next_token.fetch_add(1, STORE_BEFORE_LOAD);
        Txn {
            xid: None,
            slot: None,
            command: 0,
            isolation,
            token,
            owner: LockOwner::new(token),
            fixed: None,
            read_tables: BTreeSet::new(),
            write_tables: BTreeSet::new(),
            ddl: false,
            prepared: false,
            lock_timeout: self.lock_timeout,
            sub_xids: Vec::new(),
            parents: HashMap::new(),
            unlink: Vec::new(),
            created: Vec::new(),
            schema_version: None,
        }
    }

    /// The next xid. The fast path takes the publication lock and advances the counter inside the
    /// current batch; when the batch is used up, the slow path (outside that lock) creates the
    /// commit-log pages of the next one and logs it (spec 8.6).
    fn allocate_xid(&self) -> Result<Xid, TxnError> {
        loop {
            let limit = self.alloc.xid_limit();
            if let Some(x) = self.published.allocate_xid_below(limit) {
                return Ok(x);
            }
            self.alloc.extend_xids(limit, &*self.wal, &*self.store)?;
        }
    }

    /// Assigns the xid at the first write, and takes the exclusive lock on it that holds from
    /// assignment to the end of the transaction (spec 13.4). Fails only when the log no longer
    /// accepts records and the current batch is used up: an xid handed out then would have no
    /// durable trace.
    pub fn assign_xid(&self, txn: &mut Txn) -> Result<Xid, TxnError> {
        if let Some(x) = txn.xid {
            return Ok(x);
        }
        let x = self.allocate_xid()?;
        txn.xid = Some(x);
        txn.owner.set_xid(x);
        self.locks.lock(&txn.owner, LockTag::Xid(x), LockMode::Exclusive, WaitPolicy::NoWait).expect("a new xid has no lock to wait for");
        Ok(x)
    }

    /// Starts a sub-transaction (a savepoint, spec 12.9) below the top-level transaction.
    pub fn assign_sub_xid(&self, txn: &mut Txn) -> Result<Xid, TxnError> {
        self.assign_sub_xid_in(txn, None)
    }

    /// Starts a sub-transaction below `parent`, a live sub-xid of this transaction (a nested
    /// savepoint), or below the top-level transaction when `parent` is `None`. Its first write gets a
    /// sub-xid, which has its own xid lock so that a waiter wakes when the sub-transaction rolls
    /// back. The commit record lists the sub-xids that survive.
    pub fn assign_sub_xid_in(&self, txn: &mut Txn, parent: Option<Xid>) -> Result<Xid, TxnError> {
        assert!(txn.xid.is_some(), "a sub-transaction belongs to a transaction that has an xid");
        if let Some(p) = parent {
            assert!(txn.sub_xids.contains(&p), "the parent savepoint is live");
        }
        let x = self.allocate_xid()?;
        txn.sub_xids.push(x);
        if let Some(p) = parent {
            txn.parents.insert(x, p);
        }
        self.locks.lock(&txn.owner, LockTag::Xid(x), LockMode::Exclusive, WaitPolicy::NoWait).expect("a new xid has no lock to wait for");
        Ok(x)
    }

    /// `ROLLBACK TO` a savepoint: the sub-xid and its children are marked aborted at once, so their
    /// versions are invisible and their stamps ignored, and their xid locks are released, so that
    /// waiters wake and look again. An abort record is inserted for it, unflushed like every abort
    /// record: its header xid is the sub-xid, `sub_xids` lists its children, and `created` the
    /// relation files they created (the caller unlinks them). The record is what covers the change
    /// of the commit-log bits (spec 8.7); the verdict after a crash does not depend on it. The bits
    /// say "aborted" before the sub-xids leave the active set (invariant 2).
    pub fn rollback_sub_xid(&self, txn: &mut Txn, sub: Xid, created: &[RelFileId]) {
        if !txn.sub_xids.contains(&sub) {
            return;
        }
        let children = txn.descendants_of(sub);
        let mut all = vec![sub];
        all.extend(&children);
        self.mark_aborted(sub, &AbortRecord { sub_xids: children, created: created.to_vec() }, &all);
        txn.sub_xids.retain(|s| !all.contains(s));
        txn.parents.retain(|k, v| !all.contains(k) && !all.contains(v));
        txn.created.retain(|r| !created.contains(r));
        self.published.finish_abort_with(&all, None);
        for s in &all {
            self.locks.unlock(&txn.owner, LockTag::Xid(*s), LockMode::Exclusive);
        }
    }

    pub fn alloc_oid(&self) -> Result<Oid, TxnError> {
        self.alloc.alloc_oid(&*self.wal)
    }
    pub fn alloc_relfile(&self) -> Result<RelFileId, TxnError> {
        self.alloc.alloc_relfile(&*self.wal)
    }

    /// Takes a snapshot. The session's slot is first set to the current global horizon, then the
    /// snapshot is taken, then the slot is updated, so a horizon computation can never overtake a
    /// snapshot that is being taken (spec 12.3). Fails only when no slot is free.
    pub fn snapshot(&self, txn: &mut Txn) -> Result<Snapshot, TxnError> {
        if txn.isolation != Isolation::ReadCommitted {
            // One snapshot for the transaction: the slot keeps pinning it until the transaction ends.
            if let Some(f) = &txn.fixed {
                return Ok(Snapshot { own: txn.own(), ..f.clone() });
            }
        }
        let slot = self.slot_of(txn)?;
        let h = self.published.read();
        self.slots.set(slot, h.last_published_csn, h.oldest_active_xid);
        // A suspension point between the two steps, for the scenario that runs a horizon
        // computation exactly here (spec 19.7).
        jamsil_vfs::sched::point("snapshot:slot_pinned");
        let s = self.published.read();
        self.slots.set(slot, s.last_published_csn, s.oldest_active_xid);
        let snap = self.snapshot_from(s, txn.own());
        if txn.isolation != Isolation::ReadCommitted {
            txn.fixed = Some(Snapshot { own: None, ..snap.clone() });
        }
        Ok(snap)
    }

    fn snapshot_from(&self, s: PublishedState, own: Option<OwnTxn>) -> Snapshot {
        Snapshot { csn: s.last_published_csn, xmin: s.oldest_active_xid, xmax: s.next_xid, applied_index: s.applied_index, own }
    }

    fn slot_of(&self, txn: &mut Txn) -> Result<SlotId, TxnError> {
        if let Some(s) = txn.slot {
            return Ok(s);
        }
        let s = self.slots.claim().ok_or(TxnError::TooManySessions)?;
        txn.slot = Some(s);
        Ok(s)
    }

    /// The statement is over: the slot pins nothing until the next snapshot (READ COMMITTED).
    pub fn end_statement(&self, txn: &Txn) {
        if let Some(s) = txn.slot {
            self.slots.clear(s);
        }
    }

    /// A snapshot at an applied index of the leader's history, for a transaction that read on a
    /// follower and now writes (spec 12.6). It first claims a slot set to the current global
    /// horizon, which pins vacuum from that moment, then looks the index up in the ring. It succeeds
    /// only if the entry is at or above the pinned horizon, and then holds the slot at it until the
    /// transaction ends. Otherwise it releases the slot and fails.
    pub fn snapshot_at(&self, txn: &mut Txn, applied_index: u64) -> Result<Snapshot, TxnError> {
        let slot = self.slot_of(txn)?;
        let pinned = self.slots.horizon(&self.published);
        self.slots.set(slot, pinned.csn, pinned.xmin);
        let ok = self.published.ring_lookup(applied_index).filter(|e| e.csn >= pinned.csn && e.xmin >= pinned.xmin);
        match ok {
            Some(e) => {
                self.slots.set(slot, e.csn, e.xmin);
                let next = self.published.read().next_xid;
                Ok(Snapshot { csn: e.csn, xmin: e.xmin, xmax: next, applied_index: e.applied_index, own: txn.own() })
            }
            None => {
                self.slots.release(slot);
                txn.slot = None;
                Err(TxnError::SnapshotTooOld)
            }
        }
    }

    fn release_slot(&self, txn: &mut Txn) {
        if let Some(s) = txn.slot.take() {
            self.slots.release(s);
        }
    }

    // ---- commit and abort (spec 12.8) ----------------------------------------------------------

    /// Steps 1 to 3 of spec 12.8, and a ticket that resolves at step 6. Step 1 (admission in log
    /// order) belongs to the sequencer in the engine facade. `pos` is `None` for a node-local commit.
    /// A transaction that wrote nothing has no xid and no commit record: with a position, its entry is
    /// logged as an applied-advance record; without one, nothing is logged.
    ///
    /// One caller can log several entries before the flush they share: it calls `commit` for each and
    /// then `resolve_all` on the tickets, or `wait` on each in turn.
    ///
    /// If the gate refuses before anything is logged (a log that does not accept records), the
    /// failure is definite and the transaction is handed back, still Prepared, with its locks held:
    /// a local Prepared transaction stays Prepared until its entry is delivered again or `AbortAll`
    /// aborts it (spec 17.2). After the record is inserted, a failed flush leaves the transaction as
    /// it is, in the active set and the Prepared set, with an unknown outcome (spec 5.5).
    // The error is the transaction itself, handed back (spec 12.11): large by design, and only on a
    // path that is rare.
    #[allow(clippy::result_large_err)]
    pub fn commit(&self, mut txn: Txn, pos: Option<RaftPos>) -> Result<CommitTicket<'_>, CommitRefused> {
        let Some(xid) = txn.xid else {
            return match pos {
                Some(p) => match self.log_applied_advance(p) {
                    Ok((csn, queue_ticket)) => Ok(CommitTicket { mgr: self, pending: Some(Pending::Advance(Some(txn))), csn, queue_ticket, pos }),
                    Err(error) => Err(CommitRefused { txn, error }),
                },
                None => {
                    let csn = self.published.read().last_published_csn;
                    self.release_slot(&mut txn);
                    self.locks.unlock_all(&txn.owner);
                    Ok(CommitTicket { mgr: self, pending: Some(Pending::Nothing), csn, queue_ticket: 0, pos: None })
                }
            };
        };
        let rec = CommitRecord {
            raft: pos,
            commit_time: self.clock.now_micros(),
            schema_version: txn.schema_version,
            sub_xids: txn.sub_xids.clone(),
            unlink: txn.unlink.clone(),
        };
        let mut xids = vec![xid];
        xids.extend(&txn.sub_xids);
        // Before the critical section: the gates (spec 8.8), for the record and for the standalone
        // images that setting the bits may need.
        let gates = self.wal.gate_commit(&rec).and_then(|p| self.store.clog_gate(&xids, true).map(|g| (p, g)));
        let (staged, mut gate) = match gates {
            Ok(g) => g,
            Err(error) => return Err(CommitRefused { txn, error }),
        };
        // The pins are held now (spec 12.11): from here to step 6 nothing is pinned, read or allocated
        // under the commit mutex or the publication lock. A named suspension point for the scenario that
        // counts the buffer pool's reads after it (spec 19.7).
        jamsil_vfs::sched::point("commit:gated");
        // Step 3: under a short commit mutex, insert the record, set the bits of X and its sub-xids
        // in the commit log within the same critical section, and queue for publication. The bits
        // must change in the same critical section as the record: otherwise a checkpoint could
        // persist the commit-log page without the bit and start redo after the record.
        let set: Vec<(Xid, ClogStatus)> = xids.iter().map(|x| (*x, ClogStatus::Committed)).collect();
        let (csn, queue_ticket) = {
            let mut c = self.commit.lock();
            let mut staged = Some(staged);
            let csn = self.store.clog_set_logged(&mut gate, &set, &mut || self.wal.insert_commit(staged.take().expect("one record per permit"), xid));
            if let Some(p) = pos {
                c.logged.logged_position = p.index;
                c.logged.logged_term = p.term;
                self.durable.push(csn, p);
            }
            if let Some(v) = txn.schema_version {
                c.logged.logged_schema_version = v;
            }
            let t = c.next_ticket;
            c.next_ticket += 1;
            self.queue.lock().push_back(t);
            (csn, t)
        };
        jamsil_vfs::sched::point("commit:logged");
        Ok(CommitTicket { mgr: self, pending: Some(Pending::Txn { txn, xid, gate }), csn, queue_ticket, pos })
    }

    /// `commit` and `wait` in one call, for tests and for node-local commits. A local commit has no
    /// entry to be delivered again, so when the gate refuses this aborts the transaction.
    pub fn commit_sync(&self, txn: Txn, pos: Option<RaftPos>) -> Result<Csn, TxnError> {
        match self.commit(txn, pos) {
            Ok(ticket) => ticket.wait(),
            Err(CommitRefused { txn, error }) => {
                self.abort(txn);
                Err(error)
            }
        }
    }

    /// An applied-advance record (Raft no-ops, empty transactions): the same steps without a
    /// transaction (spec 8.7).
    pub fn applied_advance(&self, pos: RaftPos) -> Result<CommitTicket<'_>, TxnError> {
        let (csn, queue_ticket) = self.log_applied_advance(pos)?;
        Ok(CommitTicket { mgr: self, pending: Some(Pending::Advance(None)), csn, queue_ticket, pos: Some(pos) })
    }

    pub fn applied_advance_sync(&self, pos: RaftPos) -> Result<Csn, TxnError> {
        self.applied_advance(pos)?.wait()
    }

    fn log_applied_advance(&self, pos: RaftPos) -> Result<(Csn, u64), TxnError> {
        let permit = self.wal.gate_applied_advance()?;
        let mut c = self.commit.lock();
        let csn = self.wal.insert_applied_advance(permit, pos);
        c.logged.logged_position = pos.index;
        c.logged.logged_term = pos.term;
        self.durable.push(csn, pos);
        let t = c.next_ticket;
        c.next_ticket += 1;
        self.queue.lock().push_back(t);
        Ok((csn, t))
    }

    /// Steps 4 to 6 for a group of tickets: one flush covers the group (spec 17.2), then each is
    /// published in queue order. A failed flush gives every ticket the unknown outcome, and the
    /// transactions stay as they are; the ones behind them do not wait for them for ever.
    pub fn resolve_all(&self, mut tickets: Vec<CommitTicket<'_>>) -> Vec<Result<Csn, TxnError>> {
        // Step 4: flush the log up to the last record of the group.
        let upto = tickets.iter().filter(|t| !matches!(t.pending, Some(Pending::Nothing))).map(|t| t.csn).max();
        if let (true, Some(upto)) = (self.full, upto) {
            if let Err(e) = self.wal.flush(upto) {
                for t in &mut tickets {
                    if t.pending.take().is_some_and(|p| !matches!(p, Pending::Nothing)) {
                        self.withdraw(t.queue_ticket);
                    }
                }
                return tickets.iter().map(|_| Err(e.clone())).collect();
            }
            self.durable.on_durable(upto);
        }
        jamsil_vfs::sched::point("commit:flushed");
        tickets.iter_mut().map(|t| self.publish_ticket(t)).collect()
    }

    /// Steps 5 and 6 for one ticket.
    fn publish_ticket(&self, t: &mut CommitTicket<'_>) -> Result<Csn, TxnError> {
        let csn = t.csn;
        match t.pending.take() {
            Some(Pending::Txn { mut txn, xid, gate }) => {
                // Step 5: publish, in queue order. The pages the barrier writes were pinned at the gate.
                let barrier = Barrier { token: txn.token, written: &txn.write_tables, schema_version: txn.schema_version };
                self.publish_in_order(t.queue_ticket, t.pos, |published, store| {
                    published.publish_commit_with(store, &gate, Some(xid), &txn.sub_xids, csn, t.pos, Some(barrier))
                });
                // Step 6: release the locks and the read gates, wake the waiters (the lock manager
                // does) and the transactions that wait for the Prepared set to change.
                self.release_slot(&mut txn);
                self.locks.unlock_all(&txn.owner);
                self.prepared_set_changed();
                drop(gate); // the pins go at step 6
            }
            Some(Pending::Advance(txn)) => {
                let none = ClogGate::default();
                self.publish_in_order(t.queue_ticket, t.pos, |published, store| published.publish_commit(store, &none, None, &[], csn, t.pos));
                if let Some(mut txn) = txn {
                    self.release_slot(&mut txn);
                    self.locks.unlock_all(&txn.owner);
                }
            }
            Some(Pending::Nothing) | None => {}
        }
        Ok(csn)
    }

    /// Waits until every record logged before this one has been published or withdrawn, runs the
    /// barrier, and lets the next in line go (spec 12.8 step 5: "In queue order").
    fn publish_in_order(&self, ticket: u64, pos: Option<RaftPos>, barrier: impl FnOnce(&Published, &dyn TxnStore)) {
        let mut q = self.queue.lock();
        while q.front() != Some(&ticket) {
            self.queue_cv.wait(&mut q);
        }
        drop(q); // the barrier runs without the queue lock; the front stays this ticket
        if let (true, Some(p)) = (self.check_continuity, pos) {
            // Invariant 1: `applied_index` grows by exactly one per replicated commit. This ticket is
            // the only publisher now, so the value read here is the one the barrier will see.
            debug_assert_eq!(p.index, self.published.read().applied_index + 1, "applied_index must grow by exactly one");
        }
        barrier(&self.published, &*self.store);
        let mut q = self.queue.lock();
        let popped = q.pop_front();
        debug_assert_eq!(popped, Some(ticket));
        drop(q);
        self.queue_cv.notify_all();
    }

    /// A commit whose flush failed leaves the queue: its outcome is unknown, and the ones behind it
    /// must not wait for it for ever. The engine quarantines the log (spec 5.5).
    fn withdraw(&self, ticket: u64) {
        self.queue.lock().retain(|t| *t != ticket);
        self.queue_cv.notify_all();
    }

    /// Abort inserts an abort record without flushing, marks the xid aborted, and releases the slot.
    /// The row versions stay where they are for pruning and vacuum. The bits say "aborted" before
    /// the xid leaves the active set (invariant 2). It works while the log is quarantined (`AbortAll`
    /// must): see `mark_aborted`.
    pub fn abort(&self, mut txn: Txn) {
        if let Some(xid) = txn.xid {
            let mut all = vec![xid];
            all.extend(&txn.sub_xids);
            self.mark_aborted(xid, &AbortRecord { sub_xids: txn.sub_xids.clone(), created: txn.created.clone() }, &all);
            self.published.finish_abort_with(&all, Some(txn.token));
        }
        self.release_slot(&mut txn);
        // Locks and read gates go, and anything waiting on them wakes.
        self.locks.unlock_all(&txn.owner);
        self.prepared_set_changed();
    }

    /// Inserts the abort record and sets the bits in one critical section. When the log accepts no
    /// record the bits stay as they are, because a change under an exclusive latch without a record
    /// breaks invariant 5 of spec 7.10 (and the first change after a checkpoint has no image to
    /// repair a torn write), and the xids go into an in-memory set that `status` consults first. The
    /// verdict after a restart does not need the record: the crash rule aborts them.
    fn mark_aborted(&self, header: Xid, rec: &AbortRecord, all: &[Xid]) {
        let gates = self.wal.gate_abort(rec).and_then(|p| self.store.clog_gate(all, false).map(|g| (p, g)));
        match gates {
            Ok((staged, mut gate)) => {
                let set: Vec<(Xid, ClogStatus)> = all.iter().map(|x| (*x, ClogStatus::Aborted)).collect();
                let mut staged = Some(staged);
                // Both passes over the commit-log pages run inside one hold of the commit mutex, as a
                // commit's do: a checkpoint takes its redo point under that mutex, and if it could publish
                // a new one between the passes, the change of the second pass would be the first change after
                // it, with no image (spec 9.2, 12.2).
                let _hold = self.commit.lock();
                self.store.clog_set_logged(&mut gate, &set, &mut || self.wal.insert_abort(staged.take().expect("one record per permit"), header));
            }
            Err(_) => {
                self.unlogged_aborts.lock().extend(all.iter().copied());
                self.has_unlogged_aborts.store(true, STORE_BEFORE_LOAD);
            }
        }
    }

    /// Aborts that were remembered in memory because the log accepted no record (tests, `Status`).
    pub fn unlogged_aborts(&self) -> usize {
        self.unlogged_aborts.lock().len()
    }

    // ---- the commit gate (spec 12.6) -----------------------------------------------------------

    /// `TxPrepare`'s validation. Under the publication lock, the same lock the barrier takes, it never
    /// sees a half-published commit. A SERIALIZABLE transaction fails with `40001` if a table it read
    /// was committed to after its snapshot, or is written by a member of the Prepared set; if it
    /// writes a table that a member has read-gated, it waits for that gate through the lock manager
    /// and starts again; otherwise it registers its own read gates and joins the Prepared set. Below
    /// SERIALIZABLE the read set is empty, so only the waits of step 3 and the DDL step apply. A
    /// transaction that wrote nothing has no xid and never enters the gate (spec 12.6).
    pub fn prepare(&self, txn: &mut Txn) -> Result<PrepareInfo, TxnError> {
        assert!(!txn.prepared, "a transaction prepares once");
        if txn.xid.is_none() {
            txn.prepared = true;
            return Ok(PrepareInfo { schema_version: self.published.schema_version() });
        }
        let snapshot_csn = txn.fixed.as_ref().map_or(0, |s| s.csn);
        // The lock timeout bounds the whole call, not each wait inside it.
        let deadline = self.clock.now_micros() + txn.lock_timeout.as_micros() as u64;
        loop {
            let epoch = self.epoch.load(STORE_BEFORE_LOAD);
            let req = GateRequest { token: txn.token, snapshot_csn, read: &txn.read_tables, write: &txn.write_tables, ddl: txn.ddl };
            let (owner, reads) = (&txn.owner, &txn.read_tables);
            // Registering a gate never blocks, so it can happen inside the publication lock.
            let verdict = self.published.prepare(&req, || {
                for t in reads {
                    self.locks
                        .lock(owner, LockTag::ReadGate(*t), LockMode::Shared, WaitPolicy::NoWait)
                        .expect("registering a read gate never blocks");
                }
            });
            match verdict {
                GateVerdict::Admitted { schema_version } => {
                    txn.prepared = true;
                    return Ok(PrepareInfo { schema_version });
                }
                GateVerdict::Conflict => return Err(TxnError::SerializationFailure),
                GateVerdict::WaitGate(t) => {
                    let left = Duration::from_micros(deadline.saturating_sub(self.clock.now_micros()));
                    if let Err(e) = self.locks.lock(&txn.owner, LockTag::ReadGate(t), LockMode::GateWait, WaitPolicy::Timeout(left)) {
                        self.give_up_gate(txn.token);
                        return Err(TxnError::Lock(e));
                    }
                }
                GateVerdict::WaitQuiesce => {
                    if let Err(e) = self.wait_for_prepared_set_to_change(epoch, &txn.owner, deadline) {
                        self.give_up_gate(txn.token);
                        return Err(e);
                    }
                }
            }
        }
    }

    /// A `TxPrepare` that gave up (timeout, cancellation, wound): if it was a DDL transaction in
    /// line, it leaves the line, and the calls it held back may go on.
    fn give_up_gate(&self, token: u64) {
        self.published.cancel_ddl_wait(token);
        self.prepared_set_changed();
    }

    /// Wakes the transactions that wait for the Prepared set to change (DDL, spec 12.6).
    fn prepared_set_changed(&self) {
        self.epoch.fetch_add(1, STORE_BEFORE_LOAD);
        // Taking the mutex means a waiter between its check and its wait cannot miss the notify.
        let _g = self.quiesce.lock();
        self.quiesce_cv.notify_all();
    }

    /// A held-back `TxPrepare` obeys the lock timeout and cancellation like any other wait (spec
    /// 12.6), and a wound by apply (spec 13.7).
    fn wait_for_prepared_set_to_change(&self, seen: u64, owner: &LockOwner, deadline: u64) -> Result<(), TxnError> {
        let mut g = self.quiesce.lock();
        loop {
            if self.epoch.load(STORE_BEFORE_LOAD) != seen {
                return Ok(());
            }
            if owner.is_wounded() {
                return Err(TxnError::Lock(LockError::Wounded));
            }
            if owner.is_cancelled() {
                return Err(TxnError::Lock(LockError::Cancelled));
            }
            let now = self.clock.now_micros();
            if now >= deadline {
                return Err(TxnError::Lock(LockError::Timeout));
            }
            self.quiesce_cv.wait_for(&mut g, Duration::from_micros(deadline - now));
        }
    }

    /// `CancelQuery` for the transaction of `owner`: ends its lock wait or its held-back `TxPrepare`
    /// with `57014`.
    pub fn cancel(&self, owner: &LockOwner) {
        self.locks.cancel(owner);
        let _g = self.quiesce.lock();
        self.quiesce_cv.notify_all();
    }

    /// Apply priority (spec 13.7): wounds the holders and waiters of the tag, including a held-back
    /// `TxPrepare`, and returns their xids.
    pub fn wound(&self, tag: LockTag) -> Vec<Xid> {
        let xids = self.locks.wound(tag, WoundReason::ApplyPriority);
        let _g = self.quiesce.lock();
        self.quiesce_cv.notify_all();
        xids
    }

    // ---- checkpoint (spec 9.2) -----------------------------------------------------------------

    /// Steps 1 and 2 of a checkpoint, on this side: `take_redo` (the engine takes `redo_lsn` and
    /// publishes the image threshold) runs while the commit mutex is held, and in the same hold this
    /// reads the logged position and the logged schema version. A checkpoint describes the state
    /// implied by all WAL below `redo_lsn`, not the published state: a commit whose record lies below
    /// `redo_lsn` but has not passed the barrier is wholly inside the checkpoint.
    pub fn checkpoint_capture<R>(&self, take_redo: impl FnOnce() -> R) -> Capture<R> {
        let c = self.commit.lock();
        let redo = take_redo();
        let logged = c.logged;
        drop(c);
        // The counters are high-water marks and may only be ahead (spec 9.2).
        Capture { redo, logged, published: self.published.read(), alloc: self.alloc.high_water() }
    }
}
