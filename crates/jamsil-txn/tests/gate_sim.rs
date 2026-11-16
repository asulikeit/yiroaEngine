//! The commit gate (spec 12.6), sub-transactions (12.9), the snapshots of the isolation levels and the
//! checkpoint capture (9.2), under the deterministic scheduler.

use jamsil_page::txnrec::{AbortRecord, RaftPos};
use jamsil_page::*;
use jamsil_txn::*;
use jamsil_vfs::sched::{ctl, Sim};
use jamsil_vfs::sync::{thread, Arc, Mutex};
use std::time::Duration;

const MS: Duration = Duration::from_millis(1);

type Outcomes = Vec<(u8, Result<Csn, TxnError>)>;

struct Env {
    txn: Arc<TxnManager>,
    wal: Arc<MemWal>,
}

fn env(sim: &Sim) -> Env {
    let (store, wal) = (Arc::new(MemStore::new()), Arc::new(MemWal::new()));
    let txn = Arc::new(TxnManager::new(
        store.clone(),
        wal.clone(),
        sim.clock(),
        TxnOptions { sessions: 64, ..Default::default() },
        Recovered { csn: 1000, ..Default::default() },
    ));
    Env { txn, wal }
}

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

fn barrier(arrived: &Arc<Mutex<usize>>, n: usize) {
    *arrived.lock() += 1;
    while *arrived.lock() < n {
        thread::yield_now();
    }
}

/// Prepare, then commit if the gate lets it; abort if not. The outcome and the CSN.
fn finish(t: &TxnManager, mut tx: Txn) -> Result<Csn, TxnError> {
    match t.prepare(&mut tx) {
        Ok(_) => t.commit_sync(tx, None),
        Err(e) => {
            t.abort(tx);
            Err(e)
        }
    }
}

const A: Oid = 1;
const B: Oid = 2;

/// Write skew: T1 reads A and writes B, T2 reads B and writes A, both from the same state. Each
/// would be correct alone and together they are not serializable. At SERIALIZABLE the gate fails
/// the one that prepares second, in every schedule; below it, reads are not recorded and both commit.
fn skew(seed: u64, level: Isolation) -> (Result<Csn, TxnError>, Result<Csn, TxnError>) {
    let sim = Sim::new(seed);
    let e = env(&sim);
    sim.run(|| {
        let arrived = Arc::new(Mutex::new(0usize));
        let out: Arc<Mutex<Outcomes>> = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for (id, read, write) in [(1u8, A, B), (2, B, A)] {
            let (t, arrived, out) = (e.txn.clone(), arrived.clone(), out.clone());
            hs.push(spawn(&format!("t{id}"), move || {
                let mut tx = t.begin_with(level);
                t.assign_xid(&mut tx).unwrap();
                t.snapshot(&mut tx).unwrap();
                tx.note_read(read);
                tx.note_write(write);
                barrier(&arrived, 2); // both have their snapshot before either prepares
                let r = finish(&t, tx);
                out.lock().push((id, r));
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        let mut v = out.lock().clone();
        v.sort_by_key(|(id, _)| *id);
        (v[0].1.clone(), v[1].1.clone())
    })
}

#[test]
fn write_skew_is_prevented_at_serializable_in_every_schedule() {
    for seed in 0..60 {
        let (a, b) = skew(seed, Isolation::Serializable);
        let failed = [&a, &b].iter().filter(|r| matches!(r, Err(TxnError::SerializationFailure))).count();
        let ok = [&a, &b].iter().filter(|r| r.is_ok()).count();
        assert_eq!((ok, failed), (1, 1), "seed {seed}: exactly one of the two must fail with 40001: {a:?} / {b:?}");
    }
}

/// The control: below SERIALIZABLE reads are not recorded, the gate has nothing to compare, and the
/// skew commits. If this failed, the test above would prove nothing.
#[test]
fn write_skew_commits_below_serializable() {
    for seed in 0..20 {
        let (a, b) = skew(seed, Isolation::RepeatableRead);
        assert!(a.is_ok() && b.is_ok(), "seed {seed}: {a:?} / {b:?}");
    }
}

/// Step 3: a writer of a table that a Prepared transaction has read-gated waits for the gate through
/// the lock manager, and is admitted only after that transaction has committed.
#[test]
fn a_writer_of_a_gated_table_waits_for_the_reader_to_finish() {
    for seed in 0..10 {
        let sim = Sim::new(seed);
        let e = env(&sim);
        let clock = sim.clock();
        sim.run(|| {
            let log: Arc<Mutex<Vec<(&'static str, u64)>>> = Arc::new(Mutex::new(Vec::new()));
            let prepared = Arc::new(Mutex::new(false));
            let (t, l, c, p) = (e.txn.clone(), log.clone(), clock.clone(), prepared.clone());
            let reader = spawn("reader", move || {
                let mut tx = t.begin_with(Isolation::Serializable);
                t.assign_xid(&mut tx).unwrap();
                t.snapshot(&mut tx).unwrap();
                tx.note_read(A);
                tx.note_write(B);
                t.prepare(&mut tx).unwrap();
                *p.lock() = true;
                l.lock().push(("reader prepared", c.now_micros()));
                thread::sleep(200 * MS);
                let csn = t.commit_sync(tx, None).unwrap();
                l.lock().push(("reader committed", csn));
            });
            while !*prepared.lock() {
                thread::yield_now();
            }
            let (t, l, c) = (e.txn.clone(), log.clone(), clock.clone());
            let writer = spawn("writer", move || {
                let mut tx = t.begin(); // READ COMMITTED: the gate applies at every level
                t.assign_xid(&mut tx).unwrap();
                tx.note_write(A);
                let started = c.now_micros();
                t.prepare(&mut tx).unwrap();
                l.lock().push(("writer admitted", c.now_micros() - started));
                let csn = t.commit_sync(tx, None).unwrap();
                l.lock().push(("writer committed", csn));
            });
            reader.join().unwrap();
            writer.join().unwrap();
            let log = log.lock().clone();
            let at = |what: &str| log.iter().find(|(w, _)| *w == what).unwrap().1;
            assert!(at("writer admitted") >= 190_000, "seed {seed}: the writer waited for the gate ({} us)", at("writer admitted"));
            assert!(at("writer committed") > at("reader committed"), "seed {seed}: and commits after the reader");
            assert_eq!(e.txn.published().prepared_count(), 0);
            assert_eq!(e.txn.locks().entries(), 0, "gates and xid locks are all released");
        });
    }
}

/// What the property test records of each transaction that committed.
#[derive(Clone, Debug)]
struct Done {
    read: Vec<Oid>,
    write: Vec<Oid>,
    snapshot_csn: Csn,
    csn: Csn,
}

/// Spec 12.6: "Whenever a SERIALIZABLE transaction T read a table that U writes, U commits after T,
/// and commit order is a valid serial order." Here T read X, U wrote X, and T did not see U (U's CSN is
/// above T's snapshot). Random transactions of all three isolation levels, over three tables, in
/// schedules the seed picks.
#[test]
fn a_serializable_transaction_never_reads_a_table_that_a_later_snapshot_commit_wrote_before_it() {
    let mut committed_total = 0;
    let mut failed_total = 0;
    for seed in 0..40u64 {
        let sim = Sim::new(seed);
        let e = env(&sim);
        let (done, failed) = sim.run(|| {
            let done: Arc<Mutex<Vec<Done>>> = Arc::new(Mutex::new(Vec::new()));
            let failed = Arc::new(Mutex::new(0usize));
            let hs: Vec<_> = (0..5u64)
                .map(|w| {
                    let (t, done, failed) = (e.txn.clone(), done.clone(), failed.clone());
                    spawn(&format!("w{w}"), move || {
                        let mut r = seed.wrapping_mul(31).wrapping_add(w + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        let mut next = move |n: u64| {
                            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            (r >> 33) % n
                        };
                        for _ in 0..3 {
                            let level = [Isolation::Serializable, Isolation::Serializable, Isolation::ReadCommitted][next(3) as usize];
                            let mut tx = t.begin_with(level);
                            t.assign_xid(&mut tx).unwrap();
                            let snap = t.snapshot(&mut tx).unwrap();
                            thread::yield_now();
                            let tables = [1u32, 2, 3];
                            let (mut read, mut write) = (Vec::new(), Vec::new());
                            for tb in tables {
                                if next(2) == 0 {
                                    tx.note_read(tb);
                                    if level == Isolation::Serializable {
                                        read.push(tb);
                                    }
                                }
                            }
                            for tb in tables {
                                if next(3) == 0 {
                                    tx.note_write(tb);
                                    write.push(tb);
                                }
                            }
                            if write.is_empty() {
                                let tb = 1 + next(3) as u32;
                                tx.note_write(tb);
                                write.push(tb);
                            }
                            thread::yield_now();
                            match finish(&t, tx) {
                                Ok(csn) => done.lock().push(Done { read, write, snapshot_csn: snap.csn, csn }),
                                Err(TxnError::SerializationFailure) => *failed.lock() += 1,
                                Err(e) => panic!("seed {seed}: {e}"),
                            }
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let d = done.lock().clone();
            let f = *failed.lock();
            (d, f)
        });
        committed_total += done.len();
        failed_total += failed;
        for t in &done {
            for u in &done {
                if std::ptr::eq(t, u) || !t.read.iter().any(|x| u.write.contains(x)) {
                    continue;
                }
                if u.csn > t.snapshot_csn {
                    assert!(u.csn > t.csn, "seed {seed}: U wrote a table T read, T did not see U, and U committed first: T {t:?} U {u:?}");
                }
            }
        }
        assert_eq!(e.txn.published().prepared_count(), 0, "seed {seed}");
        assert_eq!(e.txn.locks().entries(), 0, "seed {seed}: every gate and xid lock was released");
    }
    assert!(committed_total > 100, "the workload must commit plenty ({committed_total})");
    assert!(failed_total > 0, "and the gate must have refused some ({failed_total})");
}

/// Spec 12.6: a transaction that contains DDL waits until the Prepared set is empty, and holds back
/// later calls until it ends. The second half is what keeps a stream of ordinary commits from
/// starving the DDL for ever.
#[test]
fn ddl_waits_for_the_prepared_set_and_holds_back_the_calls_that_come_after_it() {
    for seed in 0..10 {
        let sim = Sim::new(seed);
        let e = env(&sim);
        let clock = sim.clock();
        sim.run(|| {
            let t0 = clock.now_micros();
            let log: Arc<Mutex<Vec<(&'static str, u64)>>> = Arc::new(Mutex::new(Vec::new()));
            let go = |name: &'static str, start: u64, work: u64, ddl: bool, table: Oid| {
                let (t, l, c) = (e.txn.clone(), log.clone(), clock.clone());
                spawn(name, move || {
                    thread::sleep(Duration::from_millis(start));
                    let mut tx = t.begin();
                    t.assign_xid(&mut tx).unwrap();
                    tx.note_write(table);
                    if ddl {
                        tx.mark_ddl();
                    }
                    t.prepare(&mut tx).unwrap();
                    l.lock().push((name, c.now_micros() - t0)); // admitted
                    thread::sleep(Duration::from_millis(work));
                    t.commit_sync(tx, None).unwrap();
                })
            };
            let hs = [go("early", 0, 300, false, 10), go("ddl", 50, 100, true, 11), go("late", 100, 10, false, 12)];
            for h in hs {
                h.join().unwrap();
            }
            let log = log.lock().clone();
            let at = |n: &str| log.iter().find(|(w, _)| *w == n).unwrap().1;
            assert!(at("ddl") >= 300_000, "seed {seed}: the DDL was admitted at {} us, before the earlier commit ended", at("ddl"));
            assert!(
                at("late") >= at("ddl") + 100_000,
                "seed {seed}: the later call was admitted at {} us, before the DDL (admitted at {} us) ended",
                at("late"),
                at("ddl")
            );
        });
    }
}

/// Spec 12.9: a sub-transaction has a sub-xid and its own xid lock. Rolling back marks the sub-xid
/// aborted and wakes the waiters; the commit record lists the sub-xids that survive, and they take the
/// CSN of their parent.
#[test]
fn sub_transactions_roll_back_individually_and_commit_with_their_parent() {
    let sim = Sim::new(1);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        let x = t.assign_xid(&mut tx).unwrap();
        let (s1, s2) = (t.assign_sub_xid(&mut tx).unwrap(), t.assign_sub_xid(&mut tx).unwrap());
        assert!(x < s1 && s1 < s2);

        // A waiter on the first sub-xid wakes at its rollback, not before.
        let waiter = LockOwner::new(900);
        let (t2, w2) = (e.txn.clone(), waiter.clone());
        let h = spawn("waiter", move || {
            t2.locks().wait_for_xid(&w2, s1, Duration::from_secs(60)).unwrap();
        });
        thread::sleep(50 * MS);
        tx.created.push(77);
        t.rollback_sub_xid(&mut tx, s1, &[77]);
        assert!(tx.created.is_empty(), "the relation file the savepoint created is no longer the transaction's to remove");
        h.join().unwrap();
        assert_eq!(t.status(s1).unwrap(), XidStatus::Aborted);
        assert_eq!(t.status(s2).unwrap(), XidStatus::InProgress);
        assert_eq!(tx.sub_xids, [s2], "only the live sub-xid is left");

        let csn = t.commit_sync(tx, None).unwrap();
        assert_eq!(t.status(x).unwrap(), XidStatus::Committed);
        assert_eq!(t.status(s2).unwrap(), XidStatus::Committed);
        assert_eq!(t.status(s1).unwrap(), XidStatus::Aborted, "the rolled-back one stays aborted");
        assert_eq!(t.commit_csn(s2).unwrap(), Some(csn), "a surviving sub-xid takes its parent's CSN");
        assert_eq!(t.commit_csn(x).unwrap(), Some(csn));
        assert_eq!(t.commit_csn(s1).unwrap(), None);
        let rec = e.wal.records();
        assert!(
            rec.iter().any(|r| matches!(r, LoggedRecord::Commit { rec, .. } if rec.sub_xids == [s2])),
            "the commit record lists the survivor only: {rec:?}"
        );
        // `ROLLBACK TO` inserted an abort record (spec 8.7, answer B3): the header xid is the
        // sub-xid, `sub_xids` its children (none here) and `created` the files they created.
        let aborts: Vec<_> = rec.iter().filter_map(|r| if let LoggedRecord::Abort { xid, rec } = r { Some((*xid, rec.clone())) } else { None }).collect();
        assert_eq!(aborts, [(s1, AbortRecord { sub_xids: vec![], created: vec![77] })], "{rec:?}");
        let p = t.published().read();
        assert_eq!(p.oldest_active_xid, p.next_xid, "no sub-xid is left active");
        assert_eq!(t.locks().entries(), 0);
    });
}

#[test]
fn an_abort_aborts_the_live_sub_transactions_too() {
    let sim = Sim::new(2);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        let x = t.assign_xid(&mut tx).unwrap();
        let s = t.assign_sub_xid(&mut tx).unwrap();
        t.abort(tx);
        assert_eq!((t.status(x).unwrap(), t.status(s).unwrap()), (XidStatus::Aborted, XidStatus::Aborted));
        let p = t.published().read();
        assert_eq!(p.oldest_active_xid, p.next_xid);
    });
}

/// Spec 12.6: a REPEATABLE READ or SERIALIZABLE transaction has one snapshot; READ COMMITTED takes a
/// new one for each statement.
#[test]
fn repeatable_read_keeps_one_snapshot_and_read_committed_takes_a_new_one() {
    let sim = Sim::new(3);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let commit_other = || {
            let mut o = t.begin();
            t.assign_xid(&mut o).unwrap();
            t.commit_sync(o, None).unwrap();
        };
        let mut rr = t.begin_with(Isolation::RepeatableRead);
        let mut rc = t.begin();
        let (a1, b1) = (t.snapshot(&mut rr).unwrap(), t.snapshot(&mut rc).unwrap());
        commit_other();
        let (a2, b2) = (t.snapshot(&mut rr).unwrap(), t.snapshot(&mut rc).unwrap());
        assert_eq!((a2.csn, a2.xmax), (a1.csn, a1.xmax), "repeatable read: the same snapshot");
        assert!(b2.csn > b1.csn, "read committed: a new snapshot for the next statement");
        t.abort(rr);
        t.abort(rc);
        assert_eq!(t.slots().in_use(), 0);
    });
}

/// Spec 9.2: a checkpoint describes the state implied by all WAL below `redo_lsn`, not the published
/// state. A commit whose record is below `redo_lsn` but has not passed the barrier is wholly inside
/// the checkpoint: its Raft position and schema version are captured, though `applied_index` has not
/// moved. Taken at each suspension point of the commit.
#[test]
fn a_checkpoint_describes_the_log_not_the_published_state() {
    for point in ["commit:logged", "commit:flushed"] {
        let sim = Sim::new(4);
        let e = env(&sim);
        sim.run(|| {
            let t = &e.txn;
            for i in 1..=6u64 {
                let mut tx = t.begin();
                t.assign_xid(&mut tx).unwrap();
                t.commit_sync(tx, Some(RaftPos { index: i, term: 1 })).unwrap();
            }
            // Before the commit under test starts, the checkpoint knows 6 and no schema change.
            let before = t.checkpoint_capture(|| e.wal.insert_lsn());
            assert_eq!(before.logged.logged_position, 6);
            assert_eq!(before.logged.logged_schema_version, 0);

            ctl::hold(point);
            let t2 = e.txn.clone();
            let committer = spawn("committer", move || {
                let mut tx = t2.begin();
                t2.assign_xid(&mut tx).unwrap();
                tx.schema_version = Some(5);
                t2.commit_sync(tx, Some(RaftPos { index: 7, term: 1 })).unwrap();
            });
            ctl::await_point(point);
            // The record is in the log. The barrier has not run.
            let cap = t.checkpoint_capture(|| e.wal.insert_lsn());
            let record_end = e.wal.records().iter().rev().find_map(|r| if let LoggedRecord::Commit { end, .. } = r { Some(*end) } else { None }).unwrap();
            assert!(cap.redo >= record_end, "{point}: the record lies below redo_lsn");
            assert_eq!(cap.logged.logged_position, 7, "{point}: so its Raft position is in the checkpoint");
            assert_eq!(cap.logged.logged_schema_version, 5, "{point}: and its schema version");
            assert_eq!(cap.published.applied_index, 6, "{point}: although applied_index has not moved");
            assert!(cap.redo > before.redo);

            ctl::release(point);
            committer.join().unwrap();
            assert_eq!(t.published().read().applied_index, 7);
            assert_eq!(t.logged().logged_position, 7);
        });
    }
}

/// Step 1 on its own: T2 read A, and a transaction that wrote A committed after T2's snapshot and
/// before T2 prepares. No schedule is involved: the order is fixed.
#[test]
fn step_1_fails_a_transaction_whose_read_table_was_committed_to_after_its_snapshot() {
    let sim = Sim::new(1);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut t2 = t.begin_with(Isolation::Serializable);
        t.assign_xid(&mut t2).unwrap();
        t.snapshot(&mut t2).unwrap();
        t2.note_read(A);
        t2.note_write(B);

        let mut t1 = t.begin();
        t.assign_xid(&mut t1).unwrap();
        t1.note_write(A);
        finish(t, t1).unwrap(); // committed after t2's snapshot

        assert_eq!(t.prepare(&mut t2), Err(TxnError::SerializationFailure));
        t.abort(t2);
        assert_eq!(t.published().prepared_count(), 0);
    });
}

/// Step 2 on its own: T1 is in the Prepared set, not committed yet, and writes a table T2 read.
#[test]
fn step_2_fails_a_transaction_whose_read_table_is_written_by_a_prepared_one() {
    let sim = Sim::new(2);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut t1 = t.begin_with(Isolation::Serializable);
        let mut t2 = t.begin_with(Isolation::Serializable);
        for tx in [&mut t1, &mut t2] {
            t.assign_xid(tx).unwrap();
            t.snapshot(tx).unwrap();
        }
        t1.note_write(A);
        t1.note_read(B);
        t2.note_read(A);
        t2.note_write(B);
        t.prepare(&mut t1).unwrap();
        assert_eq!(t.prepare(&mut t2), Err(TxnError::SerializationFailure), "T1 is prepared and writes A, which T2 read");
        t.abort(t2);
        t.commit_sync(t1, None).unwrap();
        assert_eq!(t.published().prepared_count(), 0);
        assert_eq!(t.locks().entries(), 0);
    });
}
