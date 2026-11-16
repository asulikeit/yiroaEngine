//! The answers of round 7 to the commit path, under the deterministic scheduler: the ticket form of
//! `commit` (spec 12.11), the hand-back of a refused transaction (B2), aborts while the log is
//! quarantined (B2), the abort record of `ROLLBACK TO` (B3), DDL in arrival order and held-back
//! `TxPrepare` calls that obey timeouts and cancellation (B1), and the deadlock edge of `GateWait`.

use jamsil_page::txnrec::{AbortRecord, ClogStatus, RaftPos};
use jamsil_page::*;
use jamsil_txn::*;
use jamsil_vfs::sched::Sim;
use jamsil_vfs::sync::{thread, Arc, Mutex};
use std::time::Duration;

const MS: Duration = Duration::from_millis(1);

/// What a spawned `prepare` reports: its name, its verdict and how long it waited (microseconds).
type Prepares = Arc<Mutex<Vec<(&'static str, Result<(), TxnError>, u64)>>>;
type LockOutcomes = Arc<Mutex<Vec<(&'static str, Result<(), LockError>, u64)>>>;
type HeldPrepare = Arc<Mutex<Option<(Result<(), TxnError>, u64, Txn)>>>;
const A: Oid = 1;
const B: Oid = 2;

struct Env {
    txn: Arc<TxnManager>,
    store: Arc<MemStore>,
    wal: Arc<MemWal>,
}

fn env_with(sim: &Sim, opts: TxnOptions) -> Env {
    let (store, wal) = (Arc::new(MemStore::new()), Arc::new(MemWal::new()));
    let txn = Arc::new(TxnManager::new(store.clone(), wal.clone(), sim.clock(), opts, Recovered { csn: 1000, ..Default::default() }));
    Env { txn, store, wal }
}

fn env(sim: &Sim) -> Env {
    env_with(sim, TxnOptions { sessions: 64, ..Default::default() })
}

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

fn pos(i: u64) -> RaftPos {
    RaftPos { index: i, term: 1 }
}

fn writer(t: &TxnManager, table: Oid) -> Txn {
    let mut tx = t.begin();
    t.assign_xid(&mut tx).unwrap();
    tx.note_write(table);
    tx
}

/// B2: if the gate refuses before anything is logged, `commit` hands the transaction back, still
/// Prepared, with its locks held and its xid active. It commits when its entry is delivered again,
/// once the log accepts records.
#[test]
fn a_refused_commit_hands_the_prepared_transaction_back_and_it_commits_when_delivered_again() {
    let sim = Sim::new(1);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = writer(t, A);
        let xid = tx.xid().unwrap();
        t.prepare(&mut tx).unwrap();
        let logged = e.wal.records().len();
        e.wal.close();

        let CommitRefused { txn: mut tx, error } = t.commit(tx, Some(pos(1))).err().expect("the gate refuses");
        assert!(matches!(error, TxnError::Quarantined(_)), "{error}");
        assert!(tx.is_prepared(), "handed back still Prepared");
        assert_eq!(t.published().prepared_count(), 1, "and still a member of the Prepared set");
        assert_eq!(t.status(xid).unwrap(), XidStatus::InProgress, "its xid is still active");
        assert!(tx.owner().holds(LockTag::Xid(xid), LockMode::Exclusive), "and it keeps its locks");
        assert_eq!(e.wal.records().len(), logged, "nothing was logged");
        assert_eq!(t.logged().logged_position, 0, "and logged_position did not move");

        // The same entry, delivered again once the log accepts records.
        e.wal.reopen();
        let ticket = t.commit(tx, Some(pos(1))).expect("accepted now");
        let csn = ticket.wait().unwrap();
        assert_eq!(t.status(xid).unwrap(), XidStatus::Committed);
        assert_eq!(t.commit_csn(xid).unwrap(), Some(csn));
        assert_eq!((t.published().prepared_count(), t.locks().entries()), (0, 0));
        assert_eq!(t.published().read().applied_index, 1);
        tx = t.begin();
        t.abort(tx);
    });
}

/// B2, the other way out: `AbortAll` aborts a handed-back transaction while the log is still
/// quarantined. The abort record cannot be inserted, so the bits stay as they are, and `status`
/// answers from an in-memory set. Nothing is left active, and a restart agrees (the crash rule).
#[test]
fn an_abort_while_the_log_is_quarantined_is_remembered_in_memory() {
    let sim = Sim::new(2);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = writer(t, A);
        let xid = tx.xid().unwrap();
        let sub = t.assign_sub_xid(&mut tx).unwrap();
        t.prepare(&mut tx).unwrap();
        e.wal.close();
        let CommitRefused { txn: tx, .. } = t.commit(tx, Some(pos(1))).err().unwrap();
        let before = e.wal.records().len();

        t.abort(tx);
        assert_eq!(e.wal.records().len(), before, "a quarantined log took no abort record");
        for x in [xid, sub] {
            assert_eq!(e.store.clog_get(x).unwrap(), ClogStatus::InProgress, "a bit cannot change without a record that covers it");
            assert_eq!(t.status(x).unwrap(), XidStatus::Aborted, "status answers from the in-memory set, before the commit log");
            assert_eq!(t.commit_csn(x).unwrap(), None);
        }
        assert_eq!(t.unlogged_aborts(), 2);
        let p = t.published().read();
        assert_eq!(p.oldest_active_xid, p.next_xid, "nothing is left in the active set");
        assert_eq!((t.published().prepared_count(), t.locks().entries()), (0, 0));

        // A rolled-back savepoint is remembered too.
        let mut tx = writer(t, B);
        let s = t.assign_sub_xid(&mut tx).unwrap();
        t.rollback_sub_xid(&mut tx, s, &[]);
        assert_eq!(t.status(s).unwrap(), XidStatus::Aborted);
        t.abort(tx);

        // A manager built after a restart gives the same verdict by the crash rule: every xid below
        // the recovered `next_xid` that is not committed is aborted.
        let next = t.published().read().next_xid;
        let again = TxnManager::new(
            e.store.clone(),
            e.wal.clone(),
            Sim::new(0).clock(),
            TxnOptions::default(),
            Recovered { next_xid: next, csn: 1000, ..Default::default() },
        );
        assert_eq!([again.status(xid).unwrap(), again.status(sub).unwrap(), again.status(s).unwrap()], [XidStatus::Aborted; 3]);
        assert_eq!(again.status(next).unwrap(), XidStatus::InProgress, "an xid not yet handed out is not aborted");
    });
}

/// Spec 12.11: `commit` returns once the record is logged, and the ticket resolves at step 6. A group
/// of entries is logged, flushed once and published in order (spec 17.2). Until the first is
/// published, readers still see the old position.
#[test]
fn a_group_of_tickets_shares_one_flush_and_publishes_in_order() {
    let sim = Sim::new(3);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tickets = Vec::new();
        let mut xids = Vec::new();
        for i in 1..=3u64 {
            let mut tx = writer(t, A + i as Oid);
            t.prepare(&mut tx).unwrap();
            xids.push(tx.xid().unwrap());
            tickets.push(t.commit(tx, Some(pos(i))).ok().unwrap());
        }
        let csns: Vec<_> = tickets.iter().map(|k| k.csn()).collect();
        assert!(csns.windows(2).all(|w| w[0] < w[1]), "CSNs are in log order: {csns:?}");
        assert_eq!(t.logged().logged_position, 3, "all three are logged");
        assert_eq!(t.published().read().applied_index, 0, "and none is published: readers still see 0");
        assert_eq!(e.wal.flushes(), 0, "nothing was flushed yet");
        assert!(xids.iter().all(|x| t.status(*x).unwrap() == XidStatus::InProgress), "the early bits are hidden by the active set");

        let results = t.resolve_all(tickets);
        assert_eq!(results.into_iter().map(Result::unwrap).collect::<Vec<_>>(), csns);
        assert_eq!(e.wal.flushes(), 1, "one flush covers the group");
        assert!(e.wal.durable_lsn() >= csns[2]);
        assert_eq!(t.published().read().applied_index, 3);
        assert_eq!(t.published().read().last_published_csn, csns[2]);
        assert!(xids.iter().all(|x| t.status(*x).unwrap() == XidStatus::Committed));
        assert_eq!((t.published().prepared_count(), t.locks().entries()), (0, 0));
    });
}

/// A dropped ticket is a programming error (spec 12.11). A debug build panics, and takes the ticket out of
/// the queue first, so that the commits behind it do not stall; a release build resolves it in `Drop`.
/// Either way the commits behind it go through.
#[test]
fn a_dropped_ticket_is_an_error_and_does_not_block_the_commits_behind_it() {
    let sim = Sim::new(4);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = writer(t, A);
        let xid = tx.xid().unwrap();
        t.prepare(&mut tx).unwrap();
        let ticket = t.commit(tx, None).ok().unwrap();
        let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(ticket)));
        if cfg!(debug_assertions) {
            assert!(dropped.is_err(), "a debug build panics on a dropped ticket");
            assert_eq!(t.status(xid).unwrap(), XidStatus::InProgress, "and the commit was not published");
        } else {
            assert!(dropped.is_ok());
            assert_eq!(t.status(xid).unwrap(), XidStatus::Committed, "a release build resolves it");
        }
        let tx = writer(t, B);
        t.commit_sync(tx, None).expect("the commit behind the dropped ticket goes through");
    });
}

/// A thread that is already panicking does not panic again when it drops a ticket.
#[test]
fn a_ticket_dropped_while_panicking_leaves_the_queue_quietly() {
    let sim = Sim::new(14);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let tx = writer(t, A);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ticket = t.commit(tx, None).ok().unwrap();
            panic!("the caller fails with a ticket in hand");
        }));
        assert!(r.is_err());
        let tx = writer(t, B);
        t.commit_sync(tx, None).expect("and the commits behind it go through");
    });
}

/// B7: the lock timeout is the transaction's own, set from each statement; zero means the manager's
/// default (spec 12.11, 13.4). `prepare` uses the value of the last statement.
#[test]
fn the_lock_timeout_is_per_transaction_and_zero_means_the_default() {
    let sim = Sim::new(15);
    let e = env_with(&sim, TxnOptions { sessions: 64, lock_timeout: 400 * MS, ..Default::default() });
    let clock = sim.clock();
    sim.run(|| {
        let t = &e.txn;
        let mut p = writer(t, A);
        t.prepare(&mut p).unwrap(); // a DDL transaction waits for it
        for (ms, want) in [(50u64, 50_000u64), (0, 400_000), (600, 600_000)] {
            let mut tx = writer(t, B);
            tx.mark_ddl();
            t.set_lock_timeout(&mut tx, 7_000); // an earlier statement
            t.set_lock_timeout(&mut tx, ms); // the last statement is the one that counts
            let start = clock.now_micros();
            assert_eq!(t.prepare(&mut tx).map(|_| ()), Err(TxnError::Lock(LockError::Timeout)));
            let waited = clock.now_micros() - start;
            assert!((want..want + 30_000).contains(&waited), "set to {ms} ms: waited {waited} us, wanted about {want}");
            t.abort(tx);
        }
        t.abort(p);
    });
}

/// B3: `ROLLBACK TO` inserts an abort record, unflushed. Its header xid is the sub-xid that is rolled
/// back, `sub_xids` lists its children (nested savepoints), `created` the relation files they
/// created. The survivors commit with their parent.
#[test]
fn rollback_to_logs_an_abort_record_for_the_sub_xid_and_its_children() {
    let sim = Sim::new(5);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = writer(t, A);
        let top = tx.xid().unwrap();
        let s1 = t.assign_sub_xid(&mut tx).unwrap();
        let s2 = t.assign_sub_xid_in(&mut tx, Some(s1)).unwrap();
        let s3 = t.assign_sub_xid_in(&mut tx, Some(s2)).unwrap();
        let s4 = t.assign_sub_xid(&mut tx).unwrap();
        tx.created.extend([5, 6, 7]);
        let flushed = e.wal.flushes();

        t.rollback_sub_xid(&mut tx, s1, &[5, 6]);
        let aborts: Vec<_> = e
            .wal
            .records()
            .into_iter()
            .filter_map(|r| if let LoggedRecord::Abort { xid, rec } = r { Some((xid, rec)) } else { None })
            .collect();
        assert_eq!(aborts, [(s1, AbortRecord { sub_xids: vec![s2, s3], created: vec![5, 6] })], "children, grandchildren, and the files they created");
        assert_eq!(e.wal.flushes(), flushed, "unflushed, like every abort record");
        for s in [s1, s2, s3] {
            assert_eq!(t.status(s).unwrap(), XidStatus::Aborted);
            assert_eq!(e.store.clog_get(s).unwrap(), ClogStatus::Aborted, "the record covers the change of the bits");
        }
        assert_eq!(t.status(s4).unwrap(), XidStatus::InProgress, "a sibling survives");
        assert_eq!(tx.sub_xids, [s4]);
        assert_eq!(tx.created, [7], "only the files of the rolled-back branch are gone");
        assert!(!tx.owner().holds(LockTag::Xid(s2), LockMode::Exclusive), "their xid locks are released");
        assert!(tx.owner().holds(LockTag::Xid(s4), LockMode::Exclusive));

        let csn = t.commit_sync(tx, None).unwrap();
        assert_eq!(t.commit_csn(top).unwrap(), Some(csn));
        assert_eq!(t.commit_csn(s4).unwrap(), Some(csn));
        let commit_subs: Vec<_> = e.wal.records().into_iter().filter_map(|r| if let LoggedRecord::Commit { rec, .. } = r { Some(rec.sub_xids) } else { None }).collect();
        assert_eq!(commit_subs, [vec![s4]], "the commit record lists the survivors only");
    });
}

/// B1.3: several DDL transactions go in arrival order, and an ordinary call that arrives after the
/// first DDL started to wait is held back until all of them have ended.
#[test]
fn ddl_transactions_are_admitted_in_arrival_order_and_hold_back_later_calls() {
    for seed in 0..12 {
        let sim = Sim::new(seed);
        let e = env(&sim);
        sim.run(|| {
            let t = &e.txn;
            let mut p = writer(t, A);
            t.prepare(&mut p).unwrap(); // a member of the Prepared set that the DDL transactions wait for
            let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
            let mut handles = Vec::new();
            for (name, delay, ddl) in [("ddl1", 10, true), ("ddl2", 20, true), ("ddl3", 30, true), ("plain", 40, false)] {
                let (t, order) = (e.txn.clone(), order.clone());
                handles.push(spawn(name, move || {
                    thread::sleep(delay * MS);
                    let mut tx = writer(&t, B);
                    if ddl {
                        tx.mark_ddl();
                    }
                    t.prepare(&mut tx).unwrap();
                    order.lock().push(name);
                    thread::sleep(5 * MS);
                    t.commit_sync(tx, None).unwrap();
                }));
            }
            thread::sleep(100 * MS);
            assert!(order.lock().is_empty(), "seed {seed}: nobody is admitted while a member of the Prepared set is there");
            t.commit_sync(p, None).unwrap();
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(*order.lock(), ["ddl1", "ddl2", "ddl3", "plain"], "seed {seed}");
            assert_eq!((t.published().prepared_count(), t.locks().entries()), (0, 0));
        });
    }
}

/// B1.3: a `TxPrepare` that is held back obeys the lock timeout like any other wait, and a DDL
/// transaction whose wait ended leaves the line, so the calls it held back go on.
#[test]
fn a_held_back_prepare_obeys_the_lock_timeout() {
    let sim = Sim::new(7);
    let e = env_with(&sim, TxnOptions { sessions: 64, lock_timeout: 200 * MS, ..Default::default() });
    let clock = sim.clock();
    sim.run(|| {
        let t = &e.txn;
        let mut p = writer(t, A);
        t.prepare(&mut p).unwrap();
        let res: Prepares = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        // ddl1 waits for `p` and times out at 200 ms; ddl2 arrives behind it and times out at 350 ms;
        // `plain` arrives at 100 ms, is held back by the DDL line the whole time, and times out at 300 ms.
        // `late` arrives at 220 ms, after ddl1 left but while ddl2 is still in line.
        for (name, delay, ddl) in [("ddl1", 0, true), ("plain", 100, false), ("ddl2", 150, true)] {
            let (t, res, clock) = (e.txn.clone(), res.clone(), clock.clone());
            handles.push(spawn(name, move || {
                thread::sleep(delay * MS);
                let mut tx = writer(&t, B);
                if ddl {
                    tx.mark_ddl();
                }
                let start = clock.now_micros();
                let r = t.prepare(&mut tx).map(|_| ());
                res.lock().push((name, r, clock.now_micros() - start));
                t.abort(tx);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let res = res.lock().clone();
        for name in ["ddl1", "plain", "ddl2"] {
            let (_, r, waited) = res.iter().find(|(w, ..)| *w == name).unwrap();
            assert_eq!(*r, Err(TxnError::Lock(LockError::Timeout)), "{name}");
            assert!((200_000..260_000).contains(waited), "{name} waited the lock timeout: {waited} us");
        }
        t.abort(p);
        // The line is empty again: an ordinary call goes straight through.
        let mut tx = writer(t, B);
        t.prepare(&mut tx).expect("the DDL transactions left the line, so nothing is held back");
        t.abort(tx);
    });
}

/// B1.3, the other half: when the DDL transaction that held a call back gives up, the call goes on.
#[test]
fn a_call_held_back_by_a_ddl_goes_on_when_the_ddl_gives_up() {
    let sim = Sim::new(10);
    let e = env_with(&sim, TxnOptions { sessions: 64, lock_timeout: 200 * MS, ..Default::default() });
    let clock = sim.clock();
    sim.run(|| {
        let t = &e.txn;
        let mut p = writer(t, A);
        t.prepare(&mut p).unwrap();
        let res: Prepares = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for (name, delay, ddl) in [("ddl", 0, true), ("plain", 10, false)] {
            let (t, res, clock) = (e.txn.clone(), res.clone(), clock.clone());
            handles.push(spawn(name, move || {
                thread::sleep(delay * MS);
                let mut tx = writer(&t, B);
                if ddl {
                    tx.mark_ddl();
                }
                let start = clock.now_micros();
                let r = t.prepare(&mut tx).map(|_| ());
                res.lock().push((name, r, clock.now_micros() - start));
                t.abort(tx);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let res = res.lock().clone();
        let (_, r, waited) = res.iter().find(|(w, ..)| *w == "ddl").unwrap();
        assert_eq!((*r).clone(), Err(TxnError::Lock(LockError::Timeout)));
        assert!((200_000..260_000).contains(waited), "{waited}");
        let (_, r, waited) = res.iter().find(|(w, ..)| *w == "plain").unwrap();
        assert_eq!(*r, Ok(()), "the call goes on once the DDL left the line");
        assert!((185_000..260_000).contains(waited), "after about as long as the DDL waited: {waited} us");
        t.abort(p);
    });
}

/// B1.3: and cancellation ends it, `57014`; so does a wound by apply (`40001`, spec 13.7).
#[test]
fn a_held_back_prepare_obeys_cancellation_and_wounding() {
    let sim = Sim::new(8);
    let e = env(&sim);
    let clock = sim.clock();
    sim.run(|| {
        let t = &e.txn;
        let mut p = writer(t, A);
        t.prepare(&mut p).unwrap();
        for wound in [false, true] {
            let mut ddl = writer(t, B);
            ddl.mark_ddl();
            let (owner, xid) = (ddl.owner().clone(), ddl.xid().unwrap());
            let out: HeldPrepare = Arc::new(Mutex::new(None));
            let (t1, c1, o1) = (e.txn.clone(), clock.clone(), out.clone());
            let h = spawn("ddl", move || {
                let start = c1.now_micros();
                let r = t1.prepare(&mut ddl).map(|_| ());
                *o1.lock() = Some((r, c1.now_micros() - start, ddl));
            });
            thread::sleep(50 * MS);
            if wound {
                assert_eq!(t.wound(LockTag::Xid(xid)), [xid]);
            } else {
                t.cancel(&owner);
            }
            h.join().unwrap();
            let (r, waited, ddl) = out.lock().take().unwrap();
            let want = if wound { LockError::Wounded } else { LockError::Cancelled };
            assert_eq!(r, Err(TxnError::Lock(want)));
            assert!((50_000..80_000).contains(&waited), "it ended at the {want:?}, not at the timeout: {waited} us");
            t.abort(ddl);
        }
        t.abort(p);
        assert_eq!(t.locks().entries(), 0);
    });
}

/// B1.1: "The deadlock detector must see an edge from the waiting writer to every shared holder."
/// A reader holds a read gate and waits for a lock that the writer holds; the writer waits for the
/// gate. The search finds the cycle after the deadlock timeout, long before the lock timeout.
#[test]
fn the_deadlock_detector_sees_the_edge_from_a_gate_waiter_to_the_readers() {
    let sim = Sim::new(9);
    let clock = sim.clock();
    let locks = Arc::new(LockManager::new(sim.clock(), Duration::from_secs(1)));
    sim.run(|| {
        let t0 = clock.now_micros();
        let (reader, writer_) = (LockOwner::new(1), LockOwner::new(2));
        let (gate, other) = (LockTag::ReadGate(A), LockTag::Table(9));
        locks.lock(&reader, gate, LockMode::Shared, WaitPolicy::NoWait).unwrap();
        locks.lock(&writer_, other, LockMode::Exclusive, WaitPolicy::NoWait).unwrap();
        let outcome: LockOutcomes = Arc::new(Mutex::new(Vec::new()));
        let long = WaitPolicy::Timeout(Duration::from_secs(30));
        let (l1, o1, c1, r1) = (locks.clone(), outcome.clone(), clock.clone(), reader.clone());
        let h1 = spawn("reader", move || {
            let r = l1.lock(&r1, other, LockMode::Exclusive, long);
            o1.lock().push(("reader", r, c1.now_micros()));
            l1.unlock_all(&r1);
        });
        let (l2, o2, c2, w2) = (locks.clone(), outcome.clone(), clock.clone(), writer_.clone());
        let h2 = spawn("writer", move || {
            let r = l2.lock(&w2, gate, LockMode::GateWait, long);
            o2.lock().push(("writer", r, c2.now_micros()));
            l2.unlock_all(&w2);
        });
        h1.join().unwrap();
        h2.join().unwrap();
        let out = outcome.lock().clone();
        assert!(out.iter().any(|(_, r, _)| *r == Err(LockError::Deadlock)), "the cycle was found: {out:?}");
        assert!(out.iter().all(|(_, _, at)| *at - t0 < 5_000_000), "within seconds, not at the 30 s timeout: {out:?}");
        assert_eq!(locks.entries(), 0);
    });
}
