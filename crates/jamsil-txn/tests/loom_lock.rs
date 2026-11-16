//! Loom model 4 of spec 18.5 and 19.2: lock grant and wake-up. Run with `./loom.sh`; an ordinary
//! `cargo test` compiles this file to nothing.
//!
//! The lock manager's partitions use the facade's mutex and condition variable, which are loom's under
//! `--cfg loom`. Loom does not model time, so a waiter that is never woken does not time out: it
//! blocks, and loom reports the deadlock. Every model below therefore checks, over all interleavings
//! up to the preemption bound, that a waiter is woken when its lock becomes grantable (no lost
//! wake-up), that the grant respects the conflict table, and that the three wake-up paths of the
//! manager (release, cancel, and a gate wait that stands in nobody's way) do not interfere.
#![cfg(loom)]

use jamsil_txn::*;
use jamsil_vfs::env::Clock;
use jamsil_vfs::sync::Arc as StdArc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;
use std::time::Duration;

/// Time does not pass in a model: the deadline and the deadlock search are never reached.
struct FrozenClock;
impl Clock for FrozenClock {
    fn now_micros(&self) -> u64 {
        1_000_000
    }
}

fn manager() -> Arc<LockManager> {
    let clock: StdArc<dyn Clock> = StdArc::new(FrozenClock);
    Arc::new(LockManager::new(clock, Duration::from_secs(1)))
}

const LONG: WaitPolicy = WaitPolicy::Timeout(Duration::from_secs(3600));
const TAG: LockTag = LockTag::Xid(7);

/// A waiter for a lock that a holder releases at any moment: it is woken, whatever the order of
/// "check, enqueue, sleep" against "release, notify".
#[test]
fn a_release_wakes_the_waiter_in_every_interleaving() {
    loom::model(|| {
        let m = manager();
        let (holder, waiter) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&holder, TAG, LockMode::Exclusive, WaitPolicy::NoWait).unwrap();
        let (m2, w2) = (m.clone(), waiter.clone());
        let t = thread::spawn(move || m2.lock(&w2, TAG, LockMode::Shared, LONG));
        m.unlock_all(&holder);
        assert_eq!(t.join().unwrap(), Ok(()));
        assert!(waiter.holds(TAG, LockMode::Shared));
        m.unlock_all(&waiter);
        assert_eq!(m.entries(), 0);
    });
}

/// Two waiters of conflicting modes behind one holder: both are granted in the end, and an exclusive
/// grant never overlaps a shared one.
#[test]
fn granted_waiters_respect_the_conflict_table() {
    loom::model(|| {
        let m = manager();
        let holder = LockOwner::new(1);
        m.lock(&holder, TAG, LockMode::Exclusive, WaitPolicy::NoWait).unwrap();
        let (readers, writers) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let (m1, r1, w1) = (m.clone(), readers.clone(), writers.clone());
        let exclusive = thread::spawn(move || {
            let me = LockOwner::new(2);
            m1.lock(&me, TAG, LockMode::Exclusive, LONG).unwrap();
            assert_eq!((r1.load(Ordering::SeqCst), w1.fetch_add(1, Ordering::SeqCst)), (0, 0), "nobody else is in while an exclusive holder is");
            w1.fetch_sub(1, Ordering::SeqCst);
            m1.unlock_all(&me);
        });
        let (m2, r2, w2) = (m.clone(), readers.clone(), writers.clone());
        let shared = thread::spawn(move || {
            let me = LockOwner::new(3);
            m2.lock(&me, TAG, LockMode::Shared, LONG).unwrap();
            r2.fetch_add(1, Ordering::SeqCst);
            assert_eq!(w2.load(Ordering::SeqCst), 0, "no exclusive holder is in while a shared one is");
            r2.fetch_sub(1, Ordering::SeqCst);
            m2.unlock_all(&me);
        });
        m.unlock_all(&holder);
        exclusive.join().unwrap();
        shared.join().unwrap();
        assert_eq!(m.entries(), 0);
    });
}

/// Spec 13.5: a writer that waits for a read gate (`GateWait`) is woken when the last reader leaves,
/// and it never keeps another reader from registering: registration happens under the publication lock
/// and must not block, in whatever order the writer's wait and the registration fall.
#[test]
fn a_gate_waiter_is_woken_and_never_blocks_a_registering_reader() {
    loom::model(|| {
        let m = manager();
        let gate = LockTag::ReadGate(3);
        let first = LockOwner::new(1);
        m.lock(&first, gate, LockMode::Shared, WaitPolicy::NoWait).unwrap();
        let (m1, w1) = (m.clone(), LockOwner::new(2));
        let writer = thread::spawn(move || m1.lock(&w1, gate, LockMode::GateWait, LONG));
        let (m2, r2) = (m.clone(), LockOwner::new(3));
        let second = thread::spawn(move || {
            // NoWait: if the gate waiter stood in the way this would fail.
            m2.lock(&r2, gate, LockMode::Shared, WaitPolicy::NoWait).expect("registering a read gate never blocks");
            m2.unlock_all(&r2);
        });
        m.unlock_all(&first);
        second.join().unwrap();
        assert_eq!(writer.join().unwrap(), Ok(()), "the writer got through once the readers left");
        assert_eq!(m.entries(), 0, "and a gate wait leaves nothing in the table");
    });
}

/// `CancelQuery` against the wake-up: whether the cancel flag is set before the waiter enqueues, while
/// it checks, or while it sleeps, the waiter ends with `57014`, and the lock it waited for is intact.
#[test]
fn a_cancel_wakes_the_waiter_in_every_interleaving() {
    loom::model(|| {
        let m = manager();
        let (holder, waiter) = (LockOwner::new(1), LockOwner::new(2));
        m.lock(&holder, TAG, LockMode::Exclusive, WaitPolicy::NoWait).unwrap();
        let (m2, w2) = (m.clone(), waiter.clone());
        let t = thread::spawn(move || m2.lock(&w2, TAG, LockMode::Shared, LONG));
        m.cancel(&waiter);
        assert_eq!(t.join().unwrap(), Err(LockError::Cancelled));
        assert!(holder.holds(TAG, LockMode::Exclusive));
        m.unlock_all(&holder);
        assert_eq!(m.entries(), 0);
    });
}
