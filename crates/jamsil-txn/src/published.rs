//! The published state and the three writers (spec 12.3).
//!
//! A snapshot is taken by reading four numbers under a sequence lock: `last_published_csn`,
//! `oldest_active_xid`, `next_xid` and `applied_index`. Three events write them, and every one of
//! them holds the publication lock: xid allocation, which advances `next_xid` and adds to the active
//! set; abort, which removes from it; and the commit barrier. The read side takes no lock and scans
//! no list of running transactions, so its cost does not grow with the number of sessions.

use crate::seqlock::SeqLock;
use crate::store::{ClogGate, TxnStore};
use jamsil_page::txnrec::RaftPos;
use jamsil_page::*;
use jamsil_vfs::sync::Mutex;
use std::collections::{BTreeSet, HashMap, VecDeque};

const F_CSN: usize = 0;
const F_OLDEST: usize = 1;
const F_NEXT: usize = 2;
const F_APPLIED: usize = 3;

/// The ring of `snapshot_at` (spec 12.6): the last 65,536 applied positions.
pub const RING_LEN: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublishedState {
    /// The CSN of the last published commit.
    pub last_published_csn: Csn,
    /// The oldest xid in the active set; `next_xid` while it is empty.
    pub oldest_active_xid: Xid,
    /// The first xid not yet assigned.
    pub next_xid: Xid,
    /// The Raft index of the last published replicated commit or applied-advance.
    pub applied_index: u64,
}

/// One entry of the ring: where a snapshot at an applied index lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingEntry {
    pub applied_index: u64,
    pub csn: Csn,
    pub xmin: Xid,
}

/// A member of the Prepared set (spec 12.6): a transaction that passed its commit gate and has not
/// yet passed the publication barrier. A transaction waiting in the commit queue or for a WAL flush
/// is a member.
struct Prepared {
    token: u64,
    read: BTreeSet<Oid>,
    write: BTreeSet<Oid>,
    ddl: bool,
}

struct Inner {
    active: BTreeSet<Xid>,
    next_xid: Xid,
    last_csn: Csn,
    applied: u64,
    ring: VecDeque<RingEntry>,
    /// The CSN of the last commit that wrote each table (spec 12.6 step 1), updated by the barrier.
    last_commit_csn: HashMap<Oid, Csn>,
    prepared: Vec<Prepared>,
    /// Transactions that contain DDL and wait for the Prepared set to empty, in arrival order. From
    /// the moment the first starts to wait, later calls are held back, or a steady stream of them
    /// would starve it (spec 12.6).
    ddl_waiting: VecDeque<u64>,
    /// The `schema_version` of the last published commit that carried one (spec 12.6 step 4).
    schema_version: u64,
}

/// What a transaction asks of the commit gate of spec 12.6.
pub struct GateRequest<'a> {
    /// Identifies the transaction in the Prepared set.
    pub token: u64,
    /// The CSN of the transaction's snapshot, for step 1.
    pub snapshot_csn: Csn,
    /// The tables read. Empty below SERIALIZABLE.
    pub read: &'a BTreeSet<Oid>,
    /// The tables written.
    pub write: &'a BTreeSet<Oid>,
    /// The transaction contains DDL: it waits until the Prepared set is empty, and holds back later
    /// calls until it ends (spec 15.3).
    pub ddl: bool,
}

/// The answer of the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    /// The transaction joined the Prepared set and registered its read gates. Carries the
    /// `schema_version` read under the publication lock, for the caller to stamp the change set with.
    Admitted { schema_version: u64 },
    /// `40001`.
    Conflict,
    /// A table this transaction writes is read-gated by a member of the Prepared set: wait for the
    /// gate through the lock manager, then start again (step 3).
    WaitGate(Oid),
    /// DDL is in the way, or this is DDL and the Prepared set is not empty: wait, then start again.
    WaitQuiesce,
}

/// What the commit barrier does for the transaction besides publishing its CSN (spec 12.8 step 5):
/// update `last_commit_csn` for the tables it wrote, and remove it from the Prepared set.
pub struct Barrier<'a> {
    pub token: u64,
    pub written: &'a BTreeSet<Oid>,
    /// The schema version this commit installs, if it changed the schema (spec 12.8 step 5).
    pub schema_version: Option<u64>,
}

impl Inner {
    fn oldest(&self) -> Xid {
        self.active.first().copied().unwrap_or(self.next_xid)
    }
}

pub struct Published {
    seq: SeqLock<4>,
    /// The publication lock: every writer holds it.
    inner: Mutex<Inner>,
}

impl Published {
    /// `first_xid` is the first xid to hand out after a start, `csn` and `applied` what recovery
    /// established.
    pub fn new(first_xid: Xid, csn: Csn, applied: u64) -> Self {
        Self::with_schema_version(first_xid, csn, applied, 0)
    }

    pub fn with_schema_version(first_xid: Xid, csn: Csn, applied: u64, schema_version: u64) -> Self {
        Self {
            seq: SeqLock::new([csn, first_xid, first_xid, applied]),
            inner: Mutex::new(Inner {
                active: BTreeSet::new(),
                next_xid: first_xid,
                last_csn: csn,
                applied,
                ring: VecDeque::new(),
                last_commit_csn: HashMap::new(),
                prepared: Vec::new(),
                ddl_waiting: VecDeque::new(),
                schema_version,
            }),
        }
    }

    /// The four numbers as of one instant between two writes. Takes no lock.
    pub fn read(&self) -> PublishedState {
        let [csn, oldest, next, applied] = self.seq.read();
        PublishedState { last_published_csn: csn, oldest_active_xid: oldest, next_xid: next, applied_index: applied }
    }

    /// Stores the four fields inside one write section.
    fn publish(&self, g: &Inner) {
        self.seq.begin_write();
        // A named suspension point inside the section, for scenarios that suspend a writer here
        // (spec 19.7). The reader must not see this state.
        jamsil_vfs::sched::point("published:mid_write");
        self.seq.store(F_CSN, g.last_csn);
        self.seq.store(F_OLDEST, g.oldest());
        self.seq.store(F_NEXT, g.next_xid);
        self.seq.store(F_APPLIED, g.applied);
        self.seq.end_write();
    }

    /// Writer 1: assigns the next xid and adds it to the active set. Unbounded: the manager uses
    /// `allocate_xid_below`, which stays inside a logged batch.
    pub fn allocate_xid(&self) -> Xid {
        self.allocate_xid_below(Xid::MAX).expect("the xid space is not exhausted")
    }

    /// Writer 1, the fast path of a batched allocation (spec 8.6): advances the counter inside the
    /// current batch and does nothing else under the publication lock. `None` when the batch is
    /// used up; the caller then takes the slow path, outside this lock.
    pub fn allocate_xid_below(&self, limit: Xid) -> Option<Xid> {
        let mut g = self.inner.lock();
        let x = g.next_xid;
        if x >= limit {
            return None;
        }
        g.next_xid += 1;
        g.active.insert(x);
        self.publish(&g);
        Some(x)
    }

    /// Writer 2: removes an aborted xid from the active set. Its commit-log bits say "aborted" by
    /// now (invariant 2 of spec 12.12: an xid leaves the active set only after it has aborted or its
    /// CSN is published).
    pub fn finish_abort(&self, xid: Xid) {
        self.finish_abort_with(&[xid], None);
    }

    /// Writer 2 for a transaction and its sub-transactions, which also leaves the Prepared set.
    pub fn finish_abort_with(&self, xids: &[Xid], token: Option<u64>) {
        let mut g = self.inner.lock();
        for xid in xids {
            let was = g.active.remove(xid);
            debug_assert!(was, "xid {xid} was not active");
        }
        if let Some(t) = token {
            g.prepared.retain(|p| p.token != t);
            g.ddl_waiting.retain(|w| *w != t);
        }
        self.publish(&g);
    }

    /// A DDL transaction gave up waiting (its wait timed out or was cancelled): later calls may go on.
    pub fn cancel_ddl_wait(&self, token: u64) {
        self.inner.lock().ddl_waiting.retain(|w| *w != token);
    }

    /// Writer 3, the commit barrier: in one critical section it writes the CSN of the transaction
    /// and of each sub-xid into the CSN log, sets `last_published_csn` and `applied_index`, adds the
    /// position to the ring of `snapshot_at`, and removes the transaction from the active set
    /// (spec 12.8 step 5). `xid` is `None` for an applied-advance, which has no transaction.
    pub fn publish_commit(&self, store: &dyn TxnStore, gate: &ClogGate, xid: Option<Xid>, subs: &[Xid], csn: Csn, pos: Option<RaftPos>) {
        self.publish_commit_with(store, gate, xid, subs, csn, pos, None);
    }

    /// The commit barrier with the gate's bookkeeping: in the same critical section it updates
    /// `last_commit_csn` for the tables the transaction wrote and removes it from the Prepared set, so
    /// no committing writer can slip between steps 1 and 2 of the gate (spec 12.6).
    #[allow(clippy::too_many_arguments)]
    pub fn publish_commit_with(
        &self,
        store: &dyn TxnStore,
        gate: &ClogGate,
        xid: Option<Xid>,
        subs: &[Xid],
        csn: Csn,
        pos: Option<RaftPos>,
        barrier: Option<Barrier<'_>>,
    ) {
        let mut g = self.inner.lock();
        // Invariant 1: CSNs are published in strictly increasing order. (That `applied_index` grows
        // by exactly one per replicated commit is checked by the manager, which knows whether a
        // sequencer admitted the entries in log order: see `TxnOptions::check_continuity`.)
        debug_assert!(csn > g.last_csn, "CSN {csn} published after {}", g.last_csn);
        if let Some(x) = xid {
            // The pages were pinned at the commit's gate, so these writes do no I/O under the lock.
            store.csn_set(gate, x, csn);
            for s in subs {
                store.csn_set(gate, *s, csn);
            }
            let was = g.active.remove(&x);
            debug_assert!(was, "xid {x} was not active");
            for s in subs {
                g.active.remove(s);
            }
        }
        if let Some(b) = barrier {
            for t in b.written {
                g.last_commit_csn.insert(*t, csn);
            }
            g.prepared.retain(|p| p.token != b.token);
            if let Some(v) = b.schema_version {
                g.schema_version = v;
            }
        }
        g.last_csn = csn;
        if let Some(p) = pos {
            g.applied = p.index;
        }
        if let Some(p) = pos {
            let xmin = g.oldest();
            if g.ring.len() == RING_LEN {
                g.ring.pop_front();
            }
            g.ring.push_back(RingEntry { applied_index: p.index, csn, xmin });
        }
        self.publish(&g);
    }

    /// The commit gate (spec 12.6), under the publication lock, so that it never sees a half-published
    /// commit. `on_admit` runs inside the same critical section, after the transaction joined the
    /// Prepared set: it registers the read gates with the lock manager, which never blocks.
    ///
    /// 1. Fail if a table read has a `last_commit_csn` newer than the transaction's snapshot.
    /// 2. Fail if a table read is in the write set of any member of the Prepared set.
    /// 3. If a table written is read-gated by another member, the caller waits for that gate and
    ///    starts again at step 1.
    /// 4. Register the reads as gates, join the Prepared set with the writes.
    pub fn prepare(&self, req: &GateRequest<'_>, on_admit: impl FnOnce()) -> GateVerdict {
        let mut g = self.inner.lock();
        // A transaction that contains DDL waits until the Prepared set is empty, and from the moment
        // it starts to wait it holds back all later calls until it ends. Several DDL transactions go
        // in arrival order: one is admitted only when it is first in line.
        let mine = req.token;
        let others = g.prepared.iter().filter(|p| p.token != mine).count();
        let ddl_in_prepared = g.prepared.iter().any(|p| p.token != mine && p.ddl);
        let first_in_line = g.ddl_waiting.front().map_or(true, |t| *t == mine);
        let held_back = if req.ddl { !first_in_line || ddl_in_prepared || others > 0 } else { !g.ddl_waiting.is_empty() || ddl_in_prepared };
        if held_back {
            if req.ddl && !g.ddl_waiting.contains(&mine) {
                g.ddl_waiting.push_back(mine); // later calls are held back from now on
            }
            return GateVerdict::WaitQuiesce;
        }
        // From here the verdict is final or the caller waits for a read gate; a DDL transaction that
        // fails leaves the line.
        let fail = |g: &mut Inner| g.ddl_waiting.retain(|w| *w != mine);
        // Step 1.
        if req.read.iter().any(|t| g.last_commit_csn.get(t).is_some_and(|c| *c > req.snapshot_csn)) {
            fail(&mut g);
            return GateVerdict::Conflict;
        }
        // Step 2.
        if g.prepared.iter().any(|p| p.token != mine && req.read.iter().any(|t| p.write.contains(t))) {
            fail(&mut g);
            return GateVerdict::Conflict;
        }
        // Step 3. The transaction stays in the line while it waits for the gate.
        for t in req.write {
            if g.prepared.iter().any(|p| p.token != mine && p.read.contains(t)) {
                return GateVerdict::WaitGate(*t);
            }
        }
        // Step 4.
        g.ddl_waiting.retain(|w| *w != mine);
        g.prepared.push(Prepared { token: mine, read: req.read.clone(), write: req.write.clone(), ddl: req.ddl });
        on_admit();
        GateVerdict::Admitted { schema_version: g.schema_version }
    }

    /// Whether some thread holds the publication lock right now. For tests that check that no WAL
    /// insert and no page pin happens under it (spec 8.6, 12.3); only exact when nothing else runs.
    pub fn lock_is_held(&self) -> bool {
        self.inner.try_lock().is_none()
    }

    /// The `schema_version` of the last published commit that carried one.
    pub fn schema_version(&self) -> u64 {
        self.inner.lock().schema_version
    }

    /// Members of the Prepared set (tests and the invariant checks).
    pub fn prepared_count(&self) -> usize {
        self.inner.lock().prepared.len()
    }

    pub fn is_active(&self, xid: Xid) -> bool {
        self.inner.lock().active.contains(&xid)
    }

    /// The ring entry for an applied index, if it is still in the ring.
    pub fn ring_lookup(&self, applied_index: u64) -> Option<RingEntry> {
        let g = self.inner.lock();
        g.ring.iter().rev().find(|e| e.applied_index == applied_index).copied()
    }
}
