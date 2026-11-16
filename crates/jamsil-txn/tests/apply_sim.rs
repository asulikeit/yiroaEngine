//! `Apply` of spec 17.2 under the deterministic scheduler: the table of entries, and the batch scenarios
//! of 19.7 ("Batched apply", "Batch on one key", "Batch with DDL", "Failure inside a batch", "Sync failure
//! inside a batch"), with an executor that checks the dependency rules at the moment it runs.

use jamsil_txn::*;
use jamsil_vfs::sched::{ctl, Sim};
use jamsil_vfs::sync::atomic::{AtomicU64, Ordering};
use jamsil_vfs::sync::{thread, Arc, Mutex};
use jamsil_wal::QuarantineKind;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Clone, Debug)]
enum Op {
    Insert(u64),
    Update(u64),
    Delete(u64),
    Ddl(u64),
}

#[derive(Clone, Debug)]
struct Payload {
    index: u64,
    ops: Vec<Op>,
    /// How many times the executor fails with a resource error before it succeeds (u64::MAX: always).
    fail_times: u64,
}

fn payload(index: u64, ops: Vec<Op>) -> Payload {
    Payload { index, ops, fail_times: 0 }
}

fn keys(p: &Payload) -> Vec<u64> {
    p.ops.iter().filter_map(|o| if let Op::Insert(k) | Op::Update(k) | Op::Delete(k) = o { Some(*k) } else { None }).collect()
}

fn ddl(p: &Payload) -> bool {
    p.ops.iter().any(|o| matches!(o, Op::Ddl(_)))
}

/// A model of the rows. The executor checks, when it runs, that every entry it depends on was published:
/// that is what the rules of 17.2 promise.
struct Model {
    mgr: Arc<TxnManager>,
    rows: Mutex<HashMap<u64, u64>>,
    last_writer: Mutex<HashMap<u64, u64>>,
    ddl_at: Mutex<Option<u64>>,
    attempts: Mutex<HashMap<u64, u64>>,
    executed: AtomicU64,
}

impl Executor<Payload> for Model {
    fn execute(&self, txn: &mut Txn, p: &Payload) -> Result<(), ApplyError> {
        let n = {
            let mut a = self.attempts.lock();
            let n = a.entry(p.index).or_insert(0);
            *n += 1;
            *n
        };
        if p.fail_times == u64::MAX || n <= p.fail_times {
            return Err(ApplyError::Resource("no free buffer".into()));
        }
        let applied = self.mgr.positions().applied_index;
        for k in keys(p) {
            if let Some(w) = self.last_writer.lock().get(&k) {
                assert!(applied >= *w, "entry {} runs against key {k} that entry {w} wrote, and {w} is not published (applied {applied})", p.index);
            }
        }
        if let Some(d) = *self.ddl_at.lock() {
            assert!(applied >= d, "entry {} runs before the DDL entry {d} is published", p.index);
        }
        self.mgr.assign_xid(txn).unwrap();
        for op in &p.ops {
            match op {
                Op::Insert(k) => {
                    assert!(self.rows.lock().insert(*k, p.index).is_none(), "key {k} present against an insert");
                }
                Op::Update(k) => assert!(self.rows.lock().insert(*k, p.index).is_some(), "key {k} missing for an update"),
                Op::Delete(k) => assert!(self.rows.lock().remove(k).is_some(), "key {k} missing for a delete"),
                Op::Ddl(v) => {
                    txn.schema_version = Some(*v);
                    *self.ddl_at.lock() = Some(p.index);
                }
            }
            txn.note_write(1);
        }
        for k in keys(p) {
            self.last_writer.lock().insert(k, p.index);
        }
        self.executed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn depends(&self, earlier: &[&Payload], this: &Payload) -> bool {
        earlier.iter().any(|e| ddl(e) || ddl(this) || keys(e).iter().any(|k| keys(this).contains(k)))
    }
}

struct Env {
    mgr: Arc<TxnManager>,
    wal: Arc<MemWal>,
    model: Arc<Model>,
    applier: Arc<Applier>,
    quarantined: Arc<Mutex<Vec<(QuarantineKind, String)>>>,
}

fn env(sim: &Sim, applied: u64) -> Env {
    let (store, wal) = (Arc::new(MemStore::new()), Arc::new(MemWal::new()));
    let mgr = Arc::new(TxnManager::new(
        store,
        wal.clone(),
        sim.clock(),
        TxnOptions { sessions: 64, ..Default::default() },
        Recovered { csn: 1000, applied, applied_term: 1, ..Default::default() },
    ));
    let model = Arc::new(Model {
        mgr: mgr.clone(),
        rows: Mutex::new(HashMap::new()),
        last_writer: Mutex::new(HashMap::new()),
        ddl_at: Mutex::new(None),
        attempts: Mutex::new(HashMap::new()),
        executed: AtomicU64::new(0),
    });
    let quarantined: Arc<Mutex<Vec<(QuarantineKind, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let q = quarantined.clone();
    let applier = Arc::new(Applier::new(mgr.clone(), sim.clock(), Arc::new(move |k: QuarantineKind, m: &str| q.lock().push((k, m.to_string())))));
    Env { mgr, wal, model, applier, quarantined }
}

fn entry(p: Payload) -> Entry<Payload> {
    Entry { index: p.index, term: 1, kind: EntryKind::Txn { tx_id: u128::from(p.index) + 1_000, payload: p } }
}

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

/// 19.7, "Batched apply": three independent entries 101, 102 and 103 in one batch, suspended before the
/// publication. All three are admitted and logged, readers see 100 until publication, one flush covers
/// them, and they are published in order.
#[test]
fn batched_apply_three_independent_entries() {
    let sim = Sim::new(1);
    let e = env(&sim, 100);
    sim.run(|| {
        ctl::hold("commit:flushed");
        let (a, m, done) = (e.applier.clone(), e.model.clone(), Arc::new(Mutex::new(None)));
        let d2 = done.clone();
        let h = spawn("applier", move || {
            let entries: Vec<_> = (101..=103).map(|i| entry(payload(i, vec![Op::Insert(i)]))).collect();
            *d2.lock() = Some(a.apply(&entries, &*m));
        });
        ctl::await_point("commit:flushed");
        let p = e.mgr.positions();
        assert_eq!((p.admitted_index, p.logged_position, p.applied_index), (103, 103, 100), "all admitted and logged; readers still see 100");
        assert_eq!(e.wal.flushes(), 1, "one flush covers them");
        ctl::release("commit:flushed");
        h.join().unwrap();
        let out = done.lock().take().unwrap();
        assert_eq!(out, ApplyOutcome { applied_index: 103, failed: None });
        let csns: Vec<_> = e.wal.records().into_iter().filter_map(|r| if let LoggedRecord::Commit { end, .. } = r { Some(end) } else { None }).collect();
        assert_eq!(csns.len(), 3);
        assert!(csns.windows(2).all(|w| w[0] < w[1]), "published in order");
        assert_eq!(e.wal.flushes(), 1);
    });
}

/// 19.7, "Batch on one key": an insert, an update and a delete of one primary key as entries 101 to 103. The
/// batch runs as three groups without waiting on itself, and the row is absent at the end. The executor
/// asserts that each entry ran after the one it depends on was published.
#[test]
fn batch_on_one_key_runs_as_three_groups() {
    let sim = Sim::new(2);
    let e = env(&sim, 100);
    sim.run(|| {
        let entries = vec![
            entry(payload(101, vec![Op::Insert(7)])),
            entry(payload(102, vec![Op::Update(7)])),
            entry(payload(103, vec![Op::Delete(7)])),
        ];
        let out = e.applier.apply(&entries, &*e.model);
        assert_eq!(out, ApplyOutcome { applied_index: 103, failed: None });
        assert_eq!(e.wal.flushes(), 3, "three groups: a flush for each");
        assert!(e.model.rows.lock().get(&7).is_none(), "the row is absent at the end");
    });
}

/// 19.7, "Batch with DDL": a table creation followed by DML that uses it, in one batch. Two groups: the DDL
/// is published before the DML runs, so no schema mismatch can be reported.
#[test]
fn a_batch_with_ddl_publishes_the_ddl_before_the_dml_runs() {
    let sim = Sim::new(3);
    let e = env(&sim, 100);
    sim.run(|| {
        let entries = vec![entry(payload(101, vec![Op::Ddl(2)])), entry(payload(102, vec![Op::Insert(1)])), entry(payload(103, vec![Op::Insert(2)]))];
        let out = e.applier.apply(&entries, &*e.model);
        assert_eq!(out, ApplyOutcome { applied_index: 103, failed: None });
        assert_eq!(e.wal.flushes(), 2, "the DDL alone, then the two independent inserts");
        assert_eq!(e.mgr.published().schema_version(), 2);
    });
}

/// 17.2: a resource failure is retried with the same entry, and the batch continues if a retry succeeds.
#[test]
fn a_resource_failure_that_clears_is_retried_and_the_batch_continues() {
    let sim = Sim::new(4);
    let e = env(&sim, 100);
    sim.run(|| {
        let mut flaky = payload(102, vec![Op::Insert(2)]);
        flaky.fail_times = 3;
        let entries = vec![entry(payload(101, vec![Op::Insert(1)])), entry(flaky), entry(payload(103, vec![Op::Insert(3)]))];
        let out = e.applier.apply(&entries, &*e.model);
        assert_eq!(out, ApplyOutcome { applied_index: 103, failed: None });
        assert_eq!(*e.model.attempts.lock().get(&102).unwrap(), 4, "three failures and a success");
        assert!(e.quarantined.lock().is_empty());
    });
}

/// 19.7, "Failure inside a batch": the commit record of 101 is logged, then 102 fails by memory exhaustion
/// and does not clear, with repeated deliveries waiting on 101 and on 102. 101 is published and its waiter
/// gets success; 102 is withdrawn after the retries are used up, the store is quarantined, its waiter gets
/// the retry code, and the outcome names 102.
#[test]
fn failure_inside_a_batch_publishes_101_and_names_102() {
    let sim = Sim::new(5);
    let e = env(&sim, 100);
    let clock = sim.clock();
    sim.run(|| {
        let mut bad = payload(102, vec![Op::Insert(2)]);
        bad.fail_times = u64::MAX;
        let entries = vec![entry(payload(101, vec![Op::Insert(1)])), entry(bad.clone()), entry(payload(103, vec![Op::Insert(3)]))];
        let main: Arc<Mutex<Option<ApplyOutcome>>> = Arc::new(Mutex::new(None));
        let (a, m, o2) = (e.applier.clone(), e.model.clone(), main.clone());
        let t0 = clock.now_micros();
        let h = spawn("applier", move || *o2.lock() = Some(a.apply(&entries, &*m)));
        thread::sleep(Duration::from_millis(5));
        // A repeated delivery of 101 and one of 102, while the first call retries 102.
        let waiters: Arc<Mutex<Vec<(u64, ApplyOutcome)>>> = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for i in [101u64, 102] {
            let (a, m, w, bad) = (e.applier.clone(), e.model.clone(), waiters.clone(), bad.clone());
            hs.push(spawn(&format!("repeat{i}"), move || {
                let p = if i == 102 { bad } else { payload(i, vec![Op::Insert(1)]) };
                let out = a.apply(&[entry(p)], &*m);
                w.lock().push((i, out));
            }));
        }
        h.join().unwrap();
        for h in hs {
            h.join().unwrap();
        }
        let out = main.lock().take().unwrap();
        assert_eq!(out.applied_index, 101, "101 is published, 102 and 103 are not");
        let (idx, failure) = out.failed.clone().expect("the outcome names the failed entry");
        assert_eq!(idx, 102);
        assert!(matches!(failure, ApplyFailure::Quarantine(ref m) if m.contains("JDB-3001")), "{failure:?}");
        assert!((30_000_000..31_500_000).contains(&(clock.now_micros() - t0)), "retried with backoff for 30 s");
        assert_eq!(e.quarantined.lock().len(), 1);
        assert_eq!(e.quarantined.lock()[0].0, QuarantineKind::Restart, "retries used up: a restart clears it, without a snapshot (17.2)");
        let p = e.mgr.positions();
        assert_eq!((p.admitted_index, p.logged_position), (101, 101), "admitted_index fell back to logged_position");
        let w = waiters.lock().clone();
        let of = |i: u64| w.iter().find(|(x, _)| *x == i).unwrap().1.clone();
        assert_eq!(of(101).failed, None, "the waiter on 101 gets success");
        assert_eq!(of(102).failed, Some((102, ApplyFailure::Retry)), "the waiter on 102 gets the retry code");
        assert_eq!(e.model.executed.load(Ordering::SeqCst), 1, "only 101 ran");
    });
}

/// 19.7, "Sync failure inside a batch": under FULL the commit records of 101 and 102 are logged and the sync
/// that closes the group fails. Nothing is published: the caller is told the failure is fatal (exit code 75).
#[test]
fn a_failed_sync_publishes_nothing_and_is_fatal() {
    let sim = Sim::new(6);
    let e = env(&sim, 100);
    sim.run(|| {
        e.wal.set_fail_flush(true);
        let entries = vec![entry(payload(101, vec![Op::Insert(1)])), entry(payload(102, vec![Op::Insert(2)]))];
        let out = e.applier.apply(&entries, &*e.model);
        assert_eq!(out.applied_index, 100, "101 is not published before the process exits");
        assert!(matches!(out.failed, Some((_, ApplyFailure::Fatal(_)))), "{out:?}");
        assert_eq!(e.mgr.positions().logged_position, 102, "the records are in the log; recovery decides them (9.9)");
    });
}

/// The table of 17.2: skip, gap, no-op, and a local Prepared transaction committed at its position or
/// aborted when its entry is already applied.
#[test]
fn the_table_of_entries() {
    let sim = Sim::new(7);
    let e = env(&sim, 100);
    sim.run(|| {
        let t = &e.mgr;
        // A local Prepared transaction for entry 101, and one whose entry (99) is already applied.
        let mk = |ddl: bool| {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            tx.note_write(1);
            if ddl {
                tx.mark_ddl();
            }
            t.prepare(&mut tx).unwrap();
            tx
        };
        let (tx101, tx99) = (mk(false), mk(false));
        let x101 = tx101.xid().unwrap();
        let x99 = tx99.xid().unwrap();
        e.applier.register_prepared(1_101, tx101);
        e.applier.register_prepared(1_099, tx99);

        let skip = Entry { index: 99, term: 1, kind: EntryKind::Txn { tx_id: 1_099, payload: payload(99, vec![]) } };
        let local = Entry { index: 101, term: 1, kind: EntryKind::Txn { tx_id: 1_101, payload: payload(101, vec![]) } };
        let noop: Entry<Payload> = Entry { index: 102, term: 1, kind: EntryKind::Noop };
        let gap = entry(payload(105, vec![Op::Insert(5)]));

        let out = e.applier.apply(&[skip, local, noop, gap], &*e.model);
        assert_eq!(out.applied_index, 102);
        assert_eq!(out.failed, Some((105, ApplyFailure::Gap)), "an index above admitted + 1 is a gap (57P03)");
        assert_eq!(t.status(x99).unwrap(), XidStatus::Aborted, "the local Prepared transaction of an applied entry is aborted");
        assert_eq!(t.status(x101).unwrap(), XidStatus::Committed, "the local one for 101 committed at its position");
        assert!(!e.applier.is_prepared(1_101) && !e.applier.is_prepared(1_099));
        let kinds: Vec<&str> = e
            .wal
            .records()
            .iter()
            .filter_map(|r| match r {
                LoggedRecord::Commit { rec, .. } => Some(if rec.raft.map(|p| p.index) == Some(101) { "commit 101" } else { "commit" }),
                LoggedRecord::AppliedAdvance { pos, .. } => Some(if pos.index == 102 { "advance 102" } else { "advance" }),
                LoggedRecord::Abort { .. } => Some("abort"),
                LoggedRecord::AllocBatch(_) => None,
            })
            .collect();
        assert_eq!(kinds, ["abort", "commit 101", "advance 102"], "the skipped entry aborts its local transaction, then 101 commits and 102 is an applied-advance");
    });
}

/// 17.2, point 6: a local Prepared transaction whose entry was withdrawn stays Prepared. It commits when
/// its entry is delivered again, or aborts on `AbortAll`.
#[test]
fn a_local_prepared_transaction_whose_entry_failed_stays_prepared() {
    let sim = Sim::new(8);
    let e = env(&sim, 100);
    sim.run(|| {
        let t = &e.mgr;
        let mut tx = t.begin();
        let x = t.assign_xid(&mut tx).unwrap();
        tx.note_write(1);
        t.prepare(&mut tx).unwrap();
        e.applier.register_prepared(1_101, tx);
        let local = || Entry { index: 101, term: 1, kind: EntryKind::Txn { tx_id: 1_101, payload: payload(101, vec![]) } };

        e.wal.close();
        let out = e.applier.apply(&[local()], &*e.model);
        assert!(matches!(out.failed, Some((101, ApplyFailure::Quarantine(_)))), "{out:?}");
        assert!(e.applier.is_prepared(1_101), "still Prepared");
        assert_eq!(e.mgr.positions().admitted_index, 100, "and the entry was withdrawn");

        e.wal.reopen();
        let out = e.applier.apply(&[local()], &*e.model);
        assert_eq!(out, ApplyOutcome { applied_index: 101, failed: None });
        assert_eq!(t.status(x).unwrap(), XidStatus::Committed, "committed when its entry is delivered again");

        // And AbortAll aborts one that is never delivered.
        let mut tx = t.begin();
        let y = t.assign_xid(&mut tx).unwrap();
        tx.note_write(2);
        t.prepare(&mut tx).unwrap();
        e.applier.register_prepared(2_000, tx);
        e.wal.close();
        e.applier.abort_all();
        assert_eq!(t.status(y).unwrap(), XidStatus::Aborted, "AbortAll works while the log is quarantined");
        assert!(!e.applier.is_prepared(2_000));
    });
}
