//! The lock manager (spec 13) under the deterministic scheduler: the conflict matrix, first-in-first-out
//! queues, deadlock detection, timeouts and cancellation on the virtual clock, wounding, and wake-ups.

use jamsil_page::Tid;
use jamsil_txn::*;
use jamsil_vfs::sched::Sim;
use jamsil_vfs::sync::{thread, Arc, Mutex};
use std::time::Duration;

use LockMode::*;

const SEC: Duration = Duration::from_secs(1);

type Outcomes = Vec<(u64, Result<(), LockError>)>;

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

fn mgr(sim: &Sim) -> Arc<LockManager> {
    Arc::new(LockManager::new(sim.clock(), SEC))
}

fn wait(d: Duration) -> WaitPolicy {
    WaitPolicy::Timeout(d)
}

const TABLE: LockTag = LockTag::Table(100);

/// Every owner holds its first lock before any asks for its second, in every schedule: the cycle
/// always forms.
fn barrier(arrived: &Arc<Mutex<usize>>, n: usize) {
    *arrived.lock() += 1;
    while *arrived.lock() < n {
        thread::yield_now();
    }
}

/// The table of spec 13.2, as the sets of modes each request conflicts with.
fn expected_conflicts(requested: LockMode) -> Vec<LockMode> {
    match requested {
        AccessShare => vec![AccessExclusive],
        RowExclusive => vec![Share, AccessExclusive],
        Share => vec![RowExclusive, ShareUpdateExclusive, AccessExclusive],
        ShareUpdateExclusive => vec![Share, ShareUpdateExclusive, AccessExclusive],
        AccessExclusive => vec![AccessShare, RowExclusive, Share, ShareUpdateExclusive, AccessExclusive],
        _ => unreachable!(),
    }
}

#[test]
fn the_conflict_matrix_is_the_one_of_the_specification() {
    let modes = [AccessShare, RowExclusive, Share, ShareUpdateExclusive, AccessExclusive];
    let sim = Sim::new(1);
    let m = mgr(&sim);
    sim.run(|| {
        for requested in modes {
            for held in modes {
                let (a, b) = (LockOwner::new(1), LockOwner::new(2));
                m.lock(&a, TABLE, held, WaitPolicy::NoWait).unwrap();
                let got = m.lock(&b, TABLE, requested, WaitPolicy::NoWait);
                let conflict = expected_conflicts(requested).contains(&held);
                assert_eq!(got.is_err(), conflict, "{requested:?} requested against {held:?} held");
                assert_eq!(requested.conflicts(held), conflict);
                assert_eq!(held.conflicts(requested), conflict, "the matrix is symmetric");
                m.unlock_all(&a);
                m.unlock_all(&b);
            }
        }
        assert_eq!(m.entries(), 0, "nothing is left in the lock table");
    });
}

/// Spec 13.2: wait queues are first-in, first-out, so `AccessExclusive` cannot starve. A reader that
/// arrives behind a waiting `AccessExclusive` queues behind it, though it would not conflict with
/// the lock that is held.
#[test]
fn a_request_does_not_jump_ahead_of_a_conflicting_waiter() {
    for seed in 0..20 {
        let sim = Sim::new(seed);
        let m = mgr(&sim);
        sim.run(|| {
            let (a, b, c) = (LockOwner::new(1), LockOwner::new(2), LockOwner::new(3));
            let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
            m.lock(&a, TABLE, AccessShare, WaitPolicy::NoWait).unwrap();

            let (m2, b2, o2) = (m.clone(), b.clone(), order.clone());
            let writer = spawn("ddl", move || {
                m2.lock(&b2, TABLE, AccessExclusive, wait(60 * SEC)).unwrap();
                o2.lock().push("ddl");
                m2.unlock_all(&b2);
            });
            thread::sleep(SEC / 10); // the DDL is queued
            let (m3, c3, o3) = (m.clone(), c.clone(), order.clone());
            let reader = spawn("reader", move || {
                m3.lock(&c3, TABLE, AccessShare, wait(60 * SEC)).unwrap();
                o3.lock().push("reader");
                m3.unlock_all(&c3);
            });
            thread::sleep(SEC / 10);
            assert!(order.lock().is_empty(), "seed {seed}: the reader must not have been granted past the waiting DDL");
            m.unlock_all(&a);
            writer.join().unwrap();
            reader.join().unwrap();
            assert_eq!(*order.lock(), ["ddl", "reader"], "seed {seed}: first in, first out");
        });
    }
}

/// An owner that holds a lock takes it again at once, also when a conflicting waiter is queued
/// behind it: waiting would be waiting for a waiter that waits for this owner.
#[test]
fn taking_a_held_lock_again_is_granted_at_once() {
    let sim = Sim::new(2);
    let m = mgr(&sim);
    sim.run(|| {
        let (a, b) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&a, TABLE, AccessShare, WaitPolicy::NoWait).unwrap();
        let (m2, b2) = (m.clone(), b.clone());
        let ddl = spawn("ddl", move || {
            assert_eq!(m2.lock(&b2, TABLE, AccessExclusive, wait(5 * SEC)), Err(LockError::Timeout));
        });
        thread::sleep(SEC / 10);
        m.lock(&a, TABLE, AccessShare, wait(SEC)).expect("granted although a conflicting waiter is queued");
        ddl.join().unwrap();
        m.unlock_all(&a);
    });
}

/// Spec 13.5: a waiter that has slept for `deadlock_timeout` searches the wait-for graph and aborts
/// with `40P01` if it finds a cycle. The first run is the classic one, two owners and two tables
/// taken in opposite orders.
#[test]
fn two_owners_taking_two_tables_in_opposite_orders_deadlock_and_one_is_aborted() {
    let mut aborted_one = 0;
    for seed in 0..40 {
        let sim = Sim::new(seed);
        let m = mgr(&sim);
        let outcomes = sim.run(|| {
            let results: Arc<Mutex<Outcomes>> = Arc::new(Mutex::new(Vec::new()));
            let arrived = Arc::new(Mutex::new(0usize));
            let mut hs = Vec::new();
            for (id, first, second) in [(1u64, 100u32, 101u32), (2, 101, 100)] {
                let (m, r, arrived) = (m.clone(), results.clone(), arrived.clone());
                hs.push(spawn(&format!("owner{id}"), move || {
                    let o = LockOwner::new(id);
                    m.lock(&o, LockTag::Table(first), AccessExclusive, wait(30 * SEC)).unwrap();
                    barrier(&arrived, 2);
                    let got = m.lock(&o, LockTag::Table(second), AccessExclusive, wait(30 * SEC));
                    r.lock().push((id, got));
                    m.unlock_all(&o); // transaction end, whether it committed or was aborted
                }));
            }
            for h in hs {
                h.join().unwrap();
            }
            let v = results.lock().clone();
            v
        });
        let deadlocks = outcomes.iter().filter(|(_, r)| *r == Err(LockError::Deadlock)).count();
        assert!(deadlocks >= 1, "seed {seed}: the cycle must be found: {outcomes:?}");
        assert!(outcomes.iter().all(|(_, r)| r.is_ok() || *r == Err(LockError::Deadlock)), "seed {seed}: {outcomes:?}");
        assert!(sim.now() < Duration::from_secs(10), "seed {seed}: found long before the lock timeout, at {:?}", sim.now());
        if deadlocks == 1 {
            aborted_one += 1;
        }
    }
    assert!(aborted_one > 0, "in some schedule exactly one owner is aborted and the other goes on");
}

#[test]
fn a_cycle_of_three_is_found_too() {
    for seed in 0..20 {
        let sim = Sim::new(seed);
        let m = mgr(&sim);
        let deadlocks = sim.run(|| {
            let n: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
            let arrived = Arc::new(Mutex::new(0usize));
            let hs: Vec<_> = (0..3u32)
                .map(|i| {
                    let (m, n, arrived) = (m.clone(), n.clone(), arrived.clone());
                    spawn(&format!("o{i}"), move || {
                        let o = LockOwner::new(i as u64 + 1);
                        m.lock(&o, LockTag::Table(i), AccessExclusive, wait(30 * SEC)).unwrap();
                        barrier(&arrived, 3);
                        if m.lock(&o, LockTag::Table((i + 1) % 3), AccessExclusive, wait(30 * SEC)) == Err(LockError::Deadlock) {
                            *n.lock() += 1;
                        }
                        m.unlock_all(&o);
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let n = *n.lock();
            n
        });
        assert!(deadlocks >= 1, "seed {seed}: a cycle of three must be found");
    }
}

#[test]
fn a_wait_ends_with_a_timeout_exactly_when_the_lock_timeout_passes() {
    let sim = Sim::new(3);
    let (m, clock) = (mgr(&sim), sim.clock());
    sim.run(|| {
        let (a, b) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&a, TABLE, AccessExclusive, WaitPolicy::NoWait).unwrap();
        let t0 = clock.now_micros();
        assert_eq!(m.lock(&b, TABLE, AccessShare, wait(5 * SEC)), Err(LockError::Timeout), "55P03");
        assert_eq!(clock.now_micros() - t0, 5_000_000, "after exactly the lock timeout, in virtual time");
        assert_eq!(m.lock(&b, TABLE, AccessShare, WaitPolicy::NoWait), Err(LockError::Timeout), "NOWAIT fails at once");
        m.unlock_all(&a);
        assert_eq!(m.entries(), 0, "a timed-out waiter leaves nothing behind");
    });
}

#[test]
fn cancelling_a_waiter_interrupts_its_wait() {
    let sim = Sim::new(4);
    let (m, clock) = (mgr(&sim), sim.clock());
    sim.run(|| {
        let (a, b) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&a, TABLE, AccessExclusive, WaitPolicy::NoWait).unwrap();
        let (m2, b2, c2) = (m.clone(), b.clone(), clock.clone());
        let t0 = clock.now_micros();
        let waiter = spawn("waiter", move || {
            let r = m2.lock(&b2, TABLE, AccessShare, wait(60 * SEC));
            assert_eq!(r, Err(LockError::Cancelled), "57014");
            assert_eq!(c2.now_micros() - t0, 100_000, "at the moment of the cancel, not at the timeout");
        });
        thread::sleep(SEC / 10);
        m.cancel(&b);
        waiter.join().unwrap();
    });
}

/// Spec 13.7: apply must be applied. It wounds the holders and waiters of the tag; their waits end
/// with `Wounded` and the call names their xids so that apply can abort them.
#[test]
fn wounding_ends_the_waits_of_a_tag_and_names_the_xids() {
    let sim = Sim::new(5);
    let m = mgr(&sim);
    sim.run(|| {
        let (holder, waiter) = (LockOwner::new(1), LockOwner::new(2));
        holder.set_xid(40);
        waiter.set_xid(41);
        m.lock(&holder, TABLE, RowExclusive, WaitPolicy::NoWait).unwrap();
        let (m2, w2) = (m.clone(), waiter.clone());
        let h = spawn("waiter", move || {
            assert_eq!(m2.lock(&w2, TABLE, AccessExclusive, wait(60 * SEC)), Err(LockError::Wounded));
        });
        thread::sleep(SEC / 10);
        let mut xids = m.wound(TABLE, WoundReason::ApplyPriority);
        xids.sort();
        assert_eq!(xids, [40, 41]);
        h.join().unwrap();
        assert!(holder.is_wounded());
        assert_eq!(m.lock(&holder, LockTag::Table(5), AccessShare, WaitPolicy::NoWait), Err(LockError::Wounded), "the holder's next lock fails too");
        m.unlock_all(&holder); // the abort releases everything and clears the flag
        m.lock(&holder, LockTag::Table(5), AccessShare, WaitPolicy::NoWait).unwrap();
    });
}

/// No wake-up is lost: many owners queue for one exclusive lock, and they all get it, in the order
/// they queued, in every schedule.
#[test]
fn every_waiter_gets_the_lock_in_queue_order_and_no_wakeup_is_lost() {
    for seed in 0..40 {
        let sim = Sim::new(seed);
        let m = mgr(&sim);
        sim.run(|| {
            let holder = LockOwner::new(100);
            m.lock(&holder, TABLE, AccessExclusive, WaitPolicy::NoWait).unwrap();
            let order: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
            let mut hs = Vec::new();
            for id in 1..=5u64 {
                let (m, o) = (m.clone(), order.clone());
                hs.push(spawn(&format!("w{id}"), move || {
                    let me = LockOwner::new(id);
                    m.lock(&me, TABLE, AccessExclusive, wait(600 * SEC)).unwrap();
                    o.lock().push(id);
                    thread::yield_now();
                    m.unlock_all(&me);
                }));
                thread::sleep(SEC / 100); // each is queued before the next arrives
            }
            m.unlock_all(&holder);
            for h in hs {
                h.join().unwrap();
            }
            assert_eq!(*order.lock(), [1, 2, 3, 4, 5], "seed {seed}");
            assert_eq!(m.entries(), 0);
        });
    }
}

#[test]
fn unlock_since_releases_only_what_was_taken_after_the_mark() {
    let sim = Sim::new(6);
    let m = mgr(&sim);
    sim.run(|| {
        let (a, b) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&a, LockTag::Table(1), RowExclusive, WaitPolicy::NoWait).unwrap();
        let savepoint = a.mark();
        m.lock(&a, LockTag::Table(2), RowExclusive, WaitPolicy::NoWait).unwrap();
        m.lock(&a, LockTag::Row(2, Tid { block: 3, slot: 4 }), Shared, WaitPolicy::NoWait).unwrap();
        m.unlock_since(&a, savepoint); // ROLLBACK TO SAVEPOINT
        assert!(a.holds(LockTag::Table(1), RowExclusive));
        assert!(!a.holds(LockTag::Table(2), RowExclusive));
        m.lock(&b, LockTag::Table(2), AccessExclusive, WaitPolicy::NoWait).expect("table 2 was released");
        assert!(m.lock(&b, LockTag::Table(1), AccessExclusive, WaitPolicy::NoWait).is_err(), "table 1 is still held");
        m.unlock_all(&a);
        m.unlock_all(&b);
        assert_eq!(m.entries(), 0);
    });
}

/// Spec 13.4: a transaction holds an exclusive lock on its own xid, and "wait for X" is a shared
/// request on it. The waiter wakes when the holder ends, and gives up with `55P03` if it never does.
#[test]
fn waiting_for_a_transaction_ends_when_it_ends_or_when_the_timeout_passes() {
    let sim = Sim::new(7);
    let (m, clock) = (mgr(&sim), sim.clock());
    sim.run(|| {
        let (holder, waiter) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&holder, LockTag::Xid(30), Exclusive, WaitPolicy::NoWait).unwrap();
        let t0 = clock.now_micros();
        let (m2, w2, c2) = (m.clone(), waiter.clone(), clock.clone());
        let h = spawn("waiter", move || {
            m2.wait_for_xid(&w2, 30, 60 * SEC).unwrap();
            assert_eq!(c2.now_micros() - t0, 50_000, "woken as soon as the holder ends");
        });
        thread::sleep(SEC / 20);
        m.unlock_all(&holder);
        h.join().unwrap();

        m.lock(&holder, LockTag::Xid(31), Exclusive, WaitPolicy::NoWait).unwrap();
        assert_eq!(m.wait_for_xid(&waiter, 31, 3 * SEC), Err(LockError::Timeout));
        m.unlock_all(&holder);
        assert_eq!(m.entries(), 0, "the shared requests of the waiters left nothing behind");
    });
}
