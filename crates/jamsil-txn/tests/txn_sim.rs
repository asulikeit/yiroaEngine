//! The transaction manager under the deterministic scheduler (spec 12, 19.2): every interleaving
//! the scheduler picks must keep the invariants of section 12.12.

use jamsil_page::txnrec::{ClogStatus, RaftPos};
use jamsil_page::*;
use jamsil_txn::*;
use jamsil_vfs::sched::{self, ctl, Sim};
use jamsil_vfs::sync::{thread, Arc, Mutex};
use std::panic::{catch_unwind, AssertUnwindSafe};

struct Env {
    txn: Arc<TxnManager>,
    store: Arc<MemStore>,
    wal: Arc<MemWal>,
}

fn env(sim: &Sim, full: bool) -> Env {
    let (store, wal) = (Arc::new(MemStore::new()), Arc::new(MemWal::new()));
    let txn = Arc::new(TxnManager::new(
        store.clone(),
        wal.clone(),
        sim.clock(),
        TxnOptions { synchronous_full: full, sessions: 64, ..Default::default() },
        Recovered { csn: 1000, ..Default::default() },
    ));
    Env { txn, store, wal }
}

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

fn message(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

fn pos(i: u64) -> RaftPos {
    RaftPos { index: i, term: 1 }
}

/// What a snapshot taker saw: the snapshot and, for every xid below its `xmax`, the verdict it had
/// at that instant.
struct Observation {
    snap: Snapshot,
    verdicts: Vec<(Xid, bool)>,
}

/// Three writers commit or abort transactions while two readers take snapshots, in a schedule the
/// seed picks. Afterwards: a snapshot sees a commit exactly if its CSN is at most the snapshot's, and
/// it never changed its verdict about any xid (invariant 3); CSNs are in log order (invariant 1);
/// every xid left the active set after its bits were final (invariants 2 and 7); and each reader's
/// successive snapshots only move forward.
fn interleave(seed: u64) {
    let sim = Sim::new(seed);
    let e = env(&sim, true);
    sim.run(|| {
        let outcomes: Arc<Mutex<Vec<(Xid, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let observed: Arc<Mutex<Vec<Observation>>> = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for w in 0..3u64 {
            let (t, o) = (e.txn.clone(), outcomes.clone());
            hs.push(spawn(&format!("writer{w}"), move || {
                let mut r = seed ^ (w + 1).wrapping_mul(0x9E37_79B9);
                for _ in 0..4 {
                    let mut tx = t.begin();
                    let xid = t.assign_xid(&mut tx).unwrap();
                    r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let commit = (r >> 33) % 3 != 0;
                    if commit {
                        t.commit_sync(tx, None).unwrap();
                    } else {
                        t.abort(tx);
                    }
                    o.lock().push((xid, commit));
                }
            }));
        }
        for rd in 0..2 {
            let (t, s, ob) = (e.txn.clone(), e.store.clone(), observed.clone());
            hs.push(spawn(&format!("reader{rd}"), move || {
                let mut tx = t.begin();
                let mut last: Option<Snapshot> = None;
                for _ in 0..6 {
                    let snap = t.snapshot(&mut tx).unwrap();
                    assert!(snap.xmin <= snap.xmax, "xmin {} above xmax {}", snap.xmin, snap.xmax);
                    for x in 16..snap.xmin {
                        assert_ne!(s.clog_get(x).unwrap(), ClogStatus::InProgress, "xid {x} is below xmin {} but not finished", snap.xmin);
                    }
                    if let Some(l) = &last {
                        assert!(snap.csn >= l.csn && snap.xmin >= l.xmin && snap.xmax >= l.xmax, "a later snapshot went backwards");
                    }
                    let verdicts = (16..snap.xmax).map(|x| (x, xid_committed_before(&snap, x, &*s).unwrap())).collect();
                    ob.lock().push(Observation { snap: snap.clone(), verdicts });
                    last = Some(snap);
                    t.end_statement(&tx);
                    thread::yield_now();
                }
                t.abort(tx);
            }));
        }
        for h in hs {
            h.join().unwrap();
        }

        let outcomes = outcomes.lock().clone();
        // Invariant 1: the commit records are in log order and the CSNs are strictly increasing.
        let ends: Vec<Csn> = e
            .wal
            .records()
            .iter()
            .filter_map(|r| if let LoggedRecord::Commit { end, .. } = r { Some(*end) } else { None })
            .collect();
        assert!(ends.windows(2).all(|w| w[0] < w[1]), "seed {seed}: CSNs must increase: {ends:?}");
        assert_eq!(e.txn.published().read().last_published_csn, *ends.last().unwrap_or(&1000));
        let end_state = e.txn.published().read();
        assert_eq!(end_state.oldest_active_xid, end_state.next_xid, "no xid is left active");
        assert_eq!(end_state.next_xid, 16 + 12);

        for (xid, committed) in &outcomes {
            let want = if *committed { XidStatus::Committed } else { XidStatus::Aborted };
            assert_eq!(e.txn.status(*xid).unwrap(), want, "seed {seed}: xid {xid}");
            assert_eq!(e.txn.commit_csn(*xid).unwrap().is_some(), *committed);
        }
        // Each snapshot sees exactly the commits at or below its CSN, and its verdicts have not changed.
        for ob in observed.lock().iter() {
            for (x, verdict_then) in &ob.verdicts {
                let committed = outcomes.iter().any(|(ox, c)| ox == x && *c);
                let csn = e.store.csn_get(*x).unwrap();
                let want = committed && csn <= ob.snap.csn;
                assert_eq!(*verdict_then, want, "seed {seed}: snapshot {:?} on xid {x} (committed {committed}, csn {csn})", ob.snap);
                assert_eq!(xid_committed_before(&ob.snap, *x, &*e.store).unwrap(), *verdict_then, "seed {seed}: the verdict on xid {x} changed (invariant 3)");
            }
        }
    });
}

#[test]
fn snapshots_stay_consistent_while_transactions_commit_and_abort_in_every_schedule() {
    for seed in 0..60 {
        interleave(seed);
    }
}

/// Spec 12.3: a writer is inside its section, suspended at a fixed point. A reader that starts now
/// must not return what the writer has stored so far, nor the state before: it waits, and sees the
/// finished write.
#[test]
fn a_reader_never_sees_a_writer_that_is_suspended_inside_its_section() {
    for seed in 0..10 {
        let sim = Sim::new(seed);
        let e = env(&sim, true);
        sim.run(|| {
            let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let seen: Arc<Mutex<Option<PublishedState>>> = Arc::new(Mutex::new(None));
            ctl::hold("published:mid_write");
            let t = e.txn.clone();
            let writer = spawn("writer", move || {
                t.published().allocate_xid();
            });
            ctl::await_point("published:mid_write"); // the writer is inside its section
            let (t, l2, s2) = (e.txn.clone(), log.clone(), seen.clone());
            let reader = spawn("reader", move || {
                let s = t.published().read();
                l2.lock().push("read returned".into());
                *s2.lock() = Some(s);
            });
            for _ in 0..40 {
                thread::yield_now(); // the reader would have returned by now if it could
            }
            assert!(log.lock().is_empty(), "seed {seed}: the reader returned while the writer was inside its section");
            log.lock().push("released".into());
            ctl::release("published:mid_write");
            writer.join().unwrap();
            reader.join().unwrap();
            assert_eq!(*log.lock(), ["released", "read returned"]);
            let s = seen.lock().unwrap();
            assert_eq!((s.next_xid, s.oldest_active_xid), (17, 16), "the reader sees the finished write, whole");
        });
    }
}

/// Spec 12.3: "A horizon computation can never overtake a snapshot that is being taken." The slot is
/// set to the global horizon first. Here the world moves on between that step and the snapshot, and
/// the horizon computed meanwhile still holds the old position.
#[test]
fn the_horizon_never_overtakes_a_snapshot_that_is_being_taken() {
    for seed in 0..10 {
        let sim = Sim::new(seed);
        let e = env(&sim, true);
        sim.run(|| {
            let t = e.txn.clone();
            // Some history before.
            for _ in 0..2 {
                let mut tx = t.begin();
                t.assign_xid(&mut tx).unwrap();
                t.commit_sync(tx, None).unwrap();
            }
            let before = t.published().read();
            let taken: Arc<Mutex<Option<Snapshot>>> = Arc::new(Mutex::new(None));
            ctl::hold("snapshot:slot_pinned");
            let (t2, out) = (t.clone(), taken.clone());
            let snapshotter = spawn("snapshotter", move || {
                let mut tx = t2.begin();
                *out.lock() = Some(t2.snapshot(&mut tx).unwrap());
            });
            ctl::await_point("snapshot:slot_pinned"); // slot set to the horizon, snapshot not yet taken

            // The world moves on: three more commits, one of them leaves an xid active.
            for _ in 0..3 {
                let mut tx = t.begin();
                t.assign_xid(&mut tx).unwrap();
                t.commit_sync(tx, None).unwrap();
            }
            let h = t.horizon();
            assert_eq!(h.csn, before.last_published_csn, "seed {seed}: the horizon must still hold the position the slot pinned");
            assert!(h.xmin <= before.oldest_active_xid);

            ctl::release("snapshot:slot_pinned");
            snapshotter.join().unwrap();
            let s = taken.lock().clone().unwrap();
            assert!(s.csn > before.last_published_csn, "the snapshot is newer than the pinned position");
            assert!(t.horizon().csn <= s.csn, "and the horizon stays at or below it while it is held");
        });
    }
}

/// Spec 12.8 step 5: commits are published in the order their records were inserted, whatever order
/// their flushes finish in. Both commits are held after their records are in the log, released
/// together, and race; the publication barrier asserts that CSNs only increase.
#[test]
fn commits_are_published_in_log_order_whatever_the_flush_order() {
    for seed in 0..40 {
        let sim = Sim::new(seed);
        let e = env(&sim, true);
        sim.run(|| {
            ctl::hold("commit:logged");
            let mut hs = Vec::new();
            for i in 0..3 {
                let t = e.txn.clone();
                hs.push(spawn(&format!("committer{i}"), move || {
                    let mut tx = t.begin();
                    t.assign_xid(&mut tx).unwrap();
                    t.commit_sync(tx, None).unwrap();
                }));
            }
            // Wait until all three have logged their records and are held.
            for _ in 0..3 {
                ctl::await_point("commit:logged");
                thread::yield_now();
            }
            let logged = e.wal.records().len();
            assert!(logged >= 1);
            ctl::release("commit:logged");
            for h in hs {
                h.join().unwrap();
            }
            let ends: Vec<Csn> = e
                .wal
                .records()
                .iter()
                .filter_map(|r| if let LoggedRecord::Commit { end, .. } = r { Some(*end) } else { None })
                .collect();
            assert_eq!(ends.len(), 3);
            assert_eq!(e.txn.published().read().last_published_csn, *ends.iter().max().unwrap(), "seed {seed}");
        });
    }
}

#[test]
fn replicated_commits_and_applied_advances_move_applied_index_by_one() {
    let sim = Sim::new(1);
    let e = env(&sim, true);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        t.assign_xid(&mut tx).unwrap();
        t.commit_sync(tx, Some(pos(1))).unwrap();
        t.applied_advance_sync(pos(2)).unwrap();
        let tx = t.begin(); // a transaction that wrote nothing has no xid and no commit record
        let csn = t.commit_sync(tx, Some(pos(3))).unwrap();
        assert_eq!(t.published().read().applied_index, 3);
        assert_eq!(t.published().read().last_published_csn, csn);
        assert_eq!(t.logged().logged_position, 3);
        assert_eq!(t.durable_applied_index(), 3, "under FULL the two indexes are equal after the reply");
        let kinds: Vec<&str> = e
            .wal
            .records()
            .iter()
            .map(|r| match r {
                LoggedRecord::Commit { .. } => "commit",
                LoggedRecord::AppliedAdvance { .. } => "advance",
                LoggedRecord::Abort { .. } => "abort",
                LoggedRecord::AllocBatch(_) => "batch",
            })
            .collect();
        assert_eq!(kinds, ["batch", "commit", "advance", "advance"], "the first xid logs its batch; the empty transaction was logged as an applied-advance");
    });
}

/// Spec 17.4: under NORMAL the reply comes before the sync, and `durable_applied_index` follows the
/// WAL writer's syncs, never ahead of them.
#[test]
fn durable_applied_index_follows_the_syncs_under_normal() {
    let sim = Sim::new(2);
    let e = env(&sim, false);
    sim.run(|| {
        let t = &e.txn;
        let mut ends = Vec::new();
        for i in 1..=3u64 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            ends.push(t.commit_sync(tx, Some(pos(i))).unwrap());
        }
        assert_eq!(t.published().read().applied_index, 3, "NORMAL replies before the sync");
        assert_eq!(t.durable_applied_index(), 0);
        t.on_wal_durable(ends[1]); // the writer synced up to the second commit
        assert_eq!(t.durable_applied_index(), 2);
        t.on_wal_durable(ends[1]);
        assert_eq!(t.durable_applied_index(), 2, "idempotent");
        t.on_wal_durable(ends[2] + 1);
        assert_eq!(t.durable_applied_index(), 3);
    });
}

/// Spec 12.6: `snapshot_at` claims a slot at the current horizon, looks the index up in the ring,
/// and succeeds only if the entry is at or above the pinned horizon.
#[test]
fn snapshot_at_succeeds_only_above_the_pinned_horizon() {
    let sim = Sim::new(3);
    let e = env(&sim, true);
    sim.run(|| {
        let t = &e.txn;
        let mut csns = Vec::new();
        for i in 1..=5u64 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            csns.push(t.commit_sync(tx, Some(pos(i))).unwrap());
        }
        // Nothing holds the horizon back: it is the present, so an older index is too old.
        let mut tx = t.begin();
        assert_eq!(t.snapshot_at(&mut tx, 3), Err(TxnError::SnapshotTooOld));
        assert_eq!(t.snapshot_at(&mut tx, 999), Err(TxnError::SnapshotTooOld), "an index the ring never held");
        assert_eq!(t.slots().in_use(), 0, "a failed attempt releases its slot");
        // The present itself is always available.
        let now = t.snapshot_at(&mut tx, 5).unwrap();
        assert_eq!((now.csn, now.applied_index), (csns[4], 5));
        t.abort(tx);

        // A vacuum that keeps its horizon back (`snapshot_retention_s`) makes older ones succeed.
        let retention = t.slots().claim().unwrap();
        t.slots().set(retention, csns[0], 16);
        let mut tx = t.begin();
        let s = t.snapshot_at(&mut tx, 3).unwrap();
        assert_eq!((s.csn, s.applied_index), (csns[2], 3));
        assert!(t.horizon().csn <= csns[0], "and the slot now holds the horizon at its position");
        t.abort(tx);
        t.slots().release(retention);
    });
}

#[test]
fn a_failed_flush_is_an_unknown_outcome_and_does_not_block_the_commits_behind_it() {
    let sim = Sim::new(4);
    let e = env(&sim, true);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        let xid = t.assign_xid(&mut tx).unwrap();
        e.wal.set_fail_flush(true);
        let err = t.commit_sync(tx, None).unwrap_err();
        assert!(matches!(err, TxnError::OutcomeUnknown(_)), "{err}");
        assert!(e.wal.records().iter().any(|r| matches!(r, LoggedRecord::Commit { .. })), "the record was inserted");
        assert_eq!(t.status(xid).unwrap(), XidStatus::InProgress, "it was not published, so it is still in the active set");

        e.wal.set_fail_flush(false);
        let mut tx = t.begin();
        t.assign_xid(&mut tx).unwrap();
        t.commit_sync(tx, None).expect("the commit behind the failed one must not wait for it for ever");
    });
}

#[test]
fn a_log_that_does_not_accept_records_stops_a_commit_before_anything_is_logged() {
    let sim = Sim::new(5);
    let e = env(&sim, true);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        let xid = t.assign_xid(&mut tx).unwrap();
        e.wal.close();
        let n = e.wal.records().len();
        // `commit_sync` is the wrapper for node-local commits, which have no entry to be delivered
        // again, so it aborts the transaction the gate refused (the ticket form hands it back: see
        // `ticket_sim.rs`). The log accepts no abort record either, so the bit stays as it is and the
        // manager remembers the abort in memory (answer B2).
        let err = t.commit_sync(tx, None).unwrap_err();
        assert!(matches!(err, TxnError::Quarantined(_)), "a definite failure: {err}");
        assert_eq!(e.wal.records().len(), n, "nothing was logged: no commit record, and no abort record on a log that does not accept records");
        assert_eq!(e.store.clog_get(xid).unwrap(), ClogStatus::InProgress, "a bit cannot change without a record covering it");
        assert_eq!(t.status(xid).unwrap(), XidStatus::Aborted, "and status answers from the in-memory set");
        assert_eq!(t.unlogged_aborts(), 1);
        let p = t.published().read();
        assert_eq!(p.oldest_active_xid, p.next_xid, "and nothing is left in the active set");
    });
}

#[test]
fn an_abort_marks_the_xid_and_logs_without_flushing() {
    let sim = Sim::new(6);
    let e = env(&sim, true);
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        let xid = t.assign_xid(&mut tx).unwrap();
        tx.created.push(77);
        t.abort(tx);
        assert_eq!(t.status(xid).unwrap(), XidStatus::Aborted);
        assert_eq!(t.commit_csn(xid).unwrap(), None);
        assert_eq!(e.wal.durable_lsn(), 0, "an abort is not flushed");
        let rec = e.wal.records();
        assert!(rec.iter().any(|r| matches!(r, LoggedRecord::Abort { xid: x, rec } if *x == xid && rec.created == [77])), "{rec:?}");
        assert_eq!(t.published().read().oldest_active_xid, t.published().read().next_xid);
    });
}

/// A status observer polling while another task aborts: the status only ever moves from "in
/// progress" to its final value, never back (invariants 2 and 7).
#[test]
fn status_moves_from_in_progress_to_final_and_never_back() {
    for seed in 0..40 {
        let sim = Sim::new(seed);
        let e = env(&sim, true);
        sim.run(|| {
            let mut tx = e.txn.begin();
            let xid = e.txn.assign_xid(&mut tx).unwrap();
            let (t, seen) = (e.txn.clone(), Arc::new(Mutex::new(Vec::new())));
            let s2 = seen.clone();
            let observer = spawn("observer", move || {
                for _ in 0..12 {
                    s2.lock().push(t.status(xid).unwrap());
                    thread::yield_now();
                }
            });
            if seed % 2 == 0 {
                e.txn.commit_sync(tx, None).unwrap();
            } else {
                e.txn.abort(tx);
            }
            observer.join().unwrap();
            let seen = seen.lock().clone();
            let first_final = seen.iter().position(|s| *s != XidStatus::InProgress).unwrap_or(seen.len());
            assert!(seen[first_final..].iter().all(|s| *s == seen[first_final]), "seed {seed}: {seen:?}");
        });
    }
}

/// A deadlock or lost wake-up inside the commit path would show up here as a failed run; the
/// publication queue is the only place that waits.
#[test]
fn many_committers_never_deadlock() {
    for seed in 0..30 {
        let sim = Sim::new(seed);
        let e = env(&sim, seed % 2 == 0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            sim.run(|| {
                let hs: Vec<_> = (0..5)
                    .map(|i| {
                        let t = e.txn.clone();
                        spawn(&format!("c{i}"), move || {
                            for _ in 0..3 {
                                let mut tx = t.begin();
                                t.assign_xid(&mut tx).unwrap();
                                t.commit_sync(tx, None).unwrap();
                            }
                        })
                    })
                    .collect();
                for h in hs {
                    h.join().unwrap();
                }
            })
        }));
        if let Err(e) = r {
            panic!("seed {seed}: {}", message(e));
        }
    }
    let _ = sched::in_simulation();
}
