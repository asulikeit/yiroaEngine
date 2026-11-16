//! `Apply` of spec 17.2, without the part that needs rows: the sequencer's admission, no-ops, the commit
//! of a local Prepared transaction at its position, remote entries run by an `Executor`, and the batch
//! rules (groups, one flush for each, publication in log order, failure inside a batch).
//!
//! The entry payload is generic: what a remote entry contains (a change set) is not this crate's
//! business. The engine supplies an `Executor` that runs the operations of an entry inside a
//! transaction; `jamsil-heap` and `jamsil-exec` will implement it.

use crate::manager::{Admission, CommitRefused, CommitTicket, Isolation, TxnManager};
use crate::{Txn, TxnError};
use jamsil_page::txnrec::RaftPos;
use jamsil_vfs::env::Clock;
use jamsil_vfs::sync::{thread, Arc, Mutex};
use jamsil_wal::QuarantineKind;
use std::collections::HashMap;
use std::time::Duration;

/// A resource failure is retried with the same entry for this long, then the store is quarantined with a
/// reason that a restart clears (spec 17.2).
pub const RESOURCE_RETRY: Duration = Duration::from_secs(30);

pub enum EntryKind<P> {
    /// A Raft no-op: an applied-advance record.
    Noop,
    /// A transaction. If `tx_id` names a local Prepared transaction it is committed at this position;
    /// otherwise the entry is applied as a remote one.
    Txn { tx_id: u128, payload: P },
}

pub struct Entry<P> {
    pub index: u64,
    pub term: u64,
    pub kind: EntryKind<P>,
}

/// Why running the operations of a remote entry failed, by cause (spec 17.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyError {
    /// A payload checksum or `schema_version` mismatch, a key missing or present against the operation,
    /// an `old_digest` mismatch: quarantine at once with `JDB-3001`.
    Divergence(String),
    /// A page checksum failure: the same.
    Corruption(String),
    /// Disk full, no free buffer, memory budget exhausted: abort the internal transaction, keep
    /// `applied_index`, retry the same entry with backoff for 30 s.
    Resource(String),
    /// The internal transaction was about to wait for an xid that belongs to an unpublished entry of its
    /// own call (the backstop of the batch rules): close the group, publish, and retry the operation.
    WouldWaitOnOwnGroup,
    /// Fatal I/O: the process ends with exit code 75.
    Fatal(String),
}

/// Runs the operations of a remote entry. Implemented by the layers above.
pub trait Executor<P>: Send + Sync {
    /// Steps 2 to 4 of remote apply: finds the rows, changes heap and indexes, writes the outbox events,
    /// inside `txn`. The applier commits at the entry's position afterwards.
    fn execute(&self, txn: &mut Txn, payload: &P) -> Result<(), ApplyError>;
    /// Whether `this` depends on an earlier, unpublished entry of the call (spec 17.2): it touches a row,
    /// identified by table and primary key, that one of them wrote, or either of the two contains DDL.
    fn depends(&self, earlier: &[&P], this: &P) -> bool;
}

/// What a failed entry reports (spec 17.2, "Failure inside a batch").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyFailure {
    /// `57P03`: the entry is above `admitted_index + 1`; nothing was applied.
    Gap,
    /// `57P03`: the entry was admitted and then withdrawn; deliver it again.
    Retry,
    /// `JDB-3001`: the store is quarantined. Only a snapshot install clears a divergence.
    Quarantine(String),
    /// A failed WAL sync or another fatal I/O error: nothing further is published, and the caller ends the
    /// process with exit code 75.
    Fatal(String),
    /// A failure of the transaction layer that has no case of its own.
    Txn(TxnError),
}

/// `applied_index` as published, and the first entry that failed with its error, so the caller knows
/// where to resume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyOutcome {
    pub applied_index: u64,
    pub failed: Option<(u64, ApplyFailure)>,
}

/// What the applier calls to quarantine the store, saying what clears it.
pub type QuarantineFn = Arc<dyn Fn(QuarantineKind, &str) + Send + Sync>;

pub struct Applier {
    mgr: Arc<TxnManager>,
    clock: Arc<dyn Clock>,
    /// Quarantines the store (a divergence, a corruption, retries used up).
    quarantine: QuarantineFn,
    /// Local Prepared transactions by `tx_id` (spec 12.1): in memory only.
    prepared: Mutex<HashMap<u128, Txn>>,
}

/// The entries logged and not yet published: one group of the batch rules.
struct Group<'m, 'p, P> {
    tickets: Vec<CommitTicket<'m>>,
    payloads: Vec<&'p P>,
}

impl<'m, 'p, P> Group<'m, 'p, P> {
    fn new() -> Self {
        Self { tickets: Vec::new(), payloads: Vec::new() }
    }
}

impl Applier {
    pub fn new(mgr: Arc<TxnManager>, clock: Arc<dyn Clock>, quarantine: QuarantineFn) -> Self {
        Self { mgr, clock, quarantine, prepared: Mutex::new(HashMap::new()) }
    }

    /// `TxPrepare` succeeded: the transaction waits for its entry, holding every lock.
    pub fn register_prepared(&self, tx_id: u128, txn: Txn) {
        self.prepared.lock().insert(tx_id, txn);
    }

    pub fn is_prepared(&self, tx_id: u128) -> bool {
        self.prepared.lock().contains_key(&tx_id)
    }

    /// `AbortAll` (a leader change): every local Prepared transaction aborts. Works while the log is
    /// quarantined (spec 12.8).
    pub fn abort_all(&self) {
        let all: Vec<Txn> = self.prepared.lock().drain().map(|(_, t)| t).collect();
        for t in all {
            self.mgr.abort(t);
        }
    }

    /// Handles `entries` in index order and returns the published `applied_index` together with the first
    /// failed index and its error (spec 17.2).
    pub fn apply<P>(&self, entries: &[Entry<P>], exec: &dyn Executor<P>) -> ApplyOutcome {
        let mut group: Group<'_, '_, P> = Group::new();
        let mut failed = None;
        for e in entries {
            if let Err(f) = self.apply_one(e, exec, &mut group) {
                failed = Some((e.index, f));
                break;
            }
        }
        // The group is closed at the last inserted commit record, at the configured durability level, before
        // a failed entry is retried or withdrawn, and at the end of the call.
        if let Err(f) = self.close(&mut group) {
            let at = failed.as_ref().map_or(entries.last().map_or(0, |e| e.index), |(i, _)| *i);
            if failed.is_none() || matches!(f, ApplyFailure::Fatal(_)) {
                failed = Some((at, f));
            }
        }
        if failed.is_some() {
            // An admitted entry that failed before its record was inserted withdraws the ones behind it.
            self.mgr.withdraw_admitted();
        }
        ApplyOutcome { applied_index: self.mgr.positions().applied_index, failed }
    }

    fn apply_one<'a, 'p, P>(&'a self, e: &'p Entry<P>, exec: &dyn Executor<P>, group: &mut Group<'a, 'p, P>) -> Result<(), ApplyFailure> {
        let pos = RaftPos { index: e.index, term: e.term };
        match self.mgr.admit(e.index) {
            Admission::AlreadyApplied => {
                // Already applied; skip it. A local Prepared transaction with the same `tx_id` is aborted.
                if let EntryKind::Txn { tx_id, .. } = &e.kind {
                    if let Some(t) = self.prepared.lock().remove(tx_id) {
                        self.mgr.abort(t);
                    }
                }
                return Ok(());
            }
            Admission::Gap => return Err(ApplyFailure::Gap),
            Admission::InFlight => return self.wait_published(e.index),
            Admission::Admitted => {}
        }
        match &e.kind {
            EntryKind::Noop => {
                let t = self.mgr.applied_advance(pos).map_err(|er| self.txn_failure(er))?;
                group.tickets.push(t);
                Ok(())
            }
            EntryKind::Txn { tx_id, payload } => {
                let local = self.prepared.lock().remove(tx_id);
                if let Some(txn) = local {
                    // A local Prepared transaction: commit it at this position (spec 12.8).
                    return match self.mgr.commit(txn, Some(pos)) {
                        Ok(t) => {
                            group.tickets.push(t);
                            Ok(())
                        }
                        // The gate refused before anything was logged: it stays Prepared and commits when
                        // its entry is delivered again, or aborts on `AbortAll`.
                        Err(CommitRefused { txn, error }) => {
                            self.prepared.lock().insert(*tx_id, txn);
                            Err(self.txn_failure(error))
                        }
                    };
                }
                self.remote(pos, payload, exec, group)
            }
        }
    }

    /// Remote apply (spec 17.2): an internal transaction, committed at the entry's position.
    fn remote<'a, 'p, P>(&'a self, pos: RaftPos, payload: &'p P, exec: &dyn Executor<P>, group: &mut Group<'a, 'p, P>) -> Result<(), ApplyFailure> {
        // A dependent entry first flushes and publishes everything logged so far; it then opens a new group
        // and reads the published state through the ordinary paths.
        if exec.depends(&group.payloads, payload) {
            self.close(group)?;
        }
        let start = self.clock.now_micros();
        let mut backoff = Duration::from_millis(10);
        loop {
            let mut txn = self.mgr.begin_with(Isolation::ReadCommitted);
            match exec.execute(&mut txn, payload) {
                Ok(()) => {
                    if let Err(e) = self.mgr.prepare(&mut txn) {
                        self.mgr.abort(txn);
                        return Err(self.txn_failure(e));
                    }
                    return match self.mgr.commit(txn, Some(pos)) {
                        Ok(t) => {
                            group.tickets.push(t);
                            group.payloads.push(payload);
                            Ok(())
                        }
                        Err(CommitRefused { txn, error }) => {
                            self.mgr.abort(txn);
                            Err(self.txn_failure(error))
                        }
                    };
                }
                Err(err) => {
                    // The internal transaction is aborted, and the failure is handled by its cause.
                    self.mgr.abort(txn);
                    match err {
                        ApplyError::WouldWaitOnOwnGroup => {
                            // The backstop: a dependency that the two rules missed costs a flush, never a
                            // deadlock.
                            self.close(group)?;
                        }
                        ApplyError::Resource(why) => {
                            let waited = Duration::from_micros(self.clock.now_micros() - start);
                            if waited >= RESOURCE_RETRY {
                                let msg = format!("JDB-3001 resource failure not cleared after {} s: {why}", RESOURCE_RETRY.as_secs());
                                // A restart clears it, without a snapshot (spec 17.2).
                                (self.quarantine)(QuarantineKind::Restart, &msg);
                                return Err(ApplyFailure::Quarantine(msg));
                            }
                            // Entries logged so far are published first: nothing waits on them for ever.
                            self.close(group)?;
                            thread::sleep(backoff);
                            backoff = (backoff * 2).min(Duration::from_secs(1));
                        }
                        ApplyError::Divergence(m) | ApplyError::Corruption(m) => {
                            let msg = format!("JDB-3001 {m}");
                            // Divergence and corruption: only a snapshot install clears them.
                            (self.quarantine)(QuarantineKind::Corruption, &msg);
                            return Err(ApplyFailure::Quarantine(msg));
                        }
                        ApplyError::Fatal(m) => return Err(ApplyFailure::Fatal(m)),
                    }
                }
            }
        }
    }

    /// A repeated delivery of an entry that is in flight waits for its publication and then reports it as
    /// applied; if the entry was withdrawn meanwhile, it must be delivered again (`57P03`).
    fn wait_published(&self, index: u64) -> Result<(), ApplyFailure> {
        loop {
            let p = self.mgr.positions();
            if p.applied_index >= index {
                return Ok(());
            }
            if p.admitted_index < index {
                return Err(ApplyFailure::Retry);
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Flushes once for the group and publishes it in order (steps 4 to 6 of spec 12.8).
    fn close(&self, group: &mut Group<'_, '_, impl Sized>) -> Result<(), ApplyFailure> {
        group.payloads.clear();
        if group.tickets.is_empty() {
            return Ok(());
        }
        let tickets = std::mem::take(&mut group.tickets);
        let results = self.mgr.resolve_all(tickets);
        for r in results {
            if let Err(e) = r {
                return Err(match e {
                    // A failed sync is a fatal I/O failure, not a failure of the batch: nothing further is
                    // published, and the process ends with exit code 75.
                    TxnError::OutcomeUnknown(m) => ApplyFailure::Fatal(m),
                    other => self.txn_failure(other),
                });
            }
        }
        Ok(())
    }

    fn txn_failure(&self, e: TxnError) -> ApplyFailure {
        match e {
            TxnError::Quarantined(m) => ApplyFailure::Quarantine(m),
            TxnError::Corrupt(m) => ApplyFailure::Quarantine(m),
            TxnError::OutcomeUnknown(m) => ApplyFailure::Fatal(m),
            other => ApplyFailure::Txn(other),
        }
    }
}
