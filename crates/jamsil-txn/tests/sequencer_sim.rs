//! The sequencer's admission and the four positions (spec 12.8 step 1, 17.2), under the deterministic
//! scheduler: the table of 17.2, a batch of three independent entries, a failure inside a batch, and
//! a repeated delivery that races with the first.

use jamsil_page::txnrec::RaftPos;
use jamsil_page::*;
use jamsil_txn::*;
use jamsil_vfs::sched::Sim;
use jamsil_vfs::sync::{thread, Arc, Mutex};

fn env(sim: &Sim, applied: u64) -> Arc<TxnManager> {
    Arc::new(TxnManager::new(
        Arc::new(MemStore::new()),
        Arc::new(MemWal::new()),
        sim.clock(),
        TxnOptions { sessions: 64, ..Default::default() },
        Recovered { csn: 1000, applied, applied_term: 1, ..Default::default() },
    ))
}

fn pos(i: u64) -> RaftPos {
    RaftPos { index: i, term: 1 }
}

fn entry(t: &TxnManager, table: Oid) -> Txn {
    let mut tx = t.begin();
    t.assign_xid(&mut tx).unwrap();
    tx.note_write(table);
    tx
}

fn ordered(p: Positions) -> bool {
    p.admitted_index >= p.logged_position && p.logged_position >= p.applied_index
}

/// The table of 17.2, and spec 17.2's batch of three independent entries 101 to 103: admitted in turn,
/// logged, readers still see 100, one flush, published in order.
#[test]
fn admission_follows_the_table_and_a_batch_keeps_the_positions_in_order() {
    let sim = Sim::new(1);
    let t = env(&sim, 100);
    sim.run(|| {
        assert_eq!(t.admit(100), Admission::AlreadyApplied, "index <= applied_index: skip");
        assert_eq!(t.admit(99), Admission::AlreadyApplied);
        assert_eq!(t.admit(102), Admission::Gap, "index > admitted_index + 1: 57P03, nothing admitted");
        assert_eq!(t.positions().admitted_index, 100, "a gap admits nothing");
        assert_eq!(t.admit(101), Admission::Admitted);
        assert_eq!(t.admit(101), Admission::InFlight, "a repeated delivery of an entry in flight");
        assert_eq!(t.admit(102), Admission::Admitted);
        assert_eq!(t.admit(103), Admission::Admitted);
        assert_eq!(t.positions(), Positions { admitted_index: 103, logged_position: 100, applied_index: 100, durable_applied_index: 100 });

        // Logged entry by entry, flushed once, published in order.
        let mut tickets = Vec::new();
        for (i, table) in [(101, 1), (102, 2), (103, 3)] {
            tickets.push(t.commit(entry(&t, table), Some(pos(i))).ok().unwrap());
            let p = t.positions();
            assert!(ordered(p), "{p:?}");
            assert_eq!(p.logged_position, i);
        }
        let p = t.positions();
        assert_eq!((p.admitted_index, p.logged_position, p.applied_index), (103, 103, 100), "readers still see 100");
        assert_eq!(t.admit(103), Admission::InFlight, "and 103 is still in flight until it is published");
        for r in t.resolve_all(tickets) {
            r.unwrap();
        }
        let p = t.positions();
        assert_eq!(p, Positions { admitted_index: 103, logged_position: 103, applied_index: 103, durable_applied_index: 103 });
        assert_eq!(t.admit(103), Admission::AlreadyApplied);
        assert_eq!(t.admit(104), Admission::Admitted);
    });
}

/// Spec 17.2, failure inside a batch: 101 has its commit record, 102 fails before its record is
/// inserted. `admitted_index` falls back to `logged_position`; 101 is published; 102 is delivered again
/// and admitted; 103, which was admitted behind it, is withdrawn and must be delivered again too.
#[test]
fn a_failure_inside_a_batch_withdraws_the_entries_behind_it() {
    let sim = Sim::new(2);
    let t = env(&sim, 100);
    sim.run(|| {
        for i in 101..=103 {
            assert_eq!(t.admit(i), Admission::Admitted);
        }
        let ticket = t.commit(entry(&t, 1), Some(pos(101))).ok().unwrap(); // 101 has its record
        // 102 fails before anything is logged.
        t.withdraw_admitted();
        assert_eq!(t.positions(), Positions { admitted_index: 101, logged_position: 101, applied_index: 100, durable_applied_index: 100 });
        ticket.wait().unwrap();
        assert_eq!(t.positions().applied_index, 101, "101 is published, whatever happened to 102");
        assert_eq!(t.admit(103), Admission::Gap, "103 was withdrawn: it must wait for 102");
        assert_eq!(t.admit(102), Admission::Admitted, "the retry of 102");
        assert_eq!(t.admit(103), Admission::Admitted);
        assert!(ordered(t.positions()));
    });
}

/// A repeated delivery that races with the first: exactly one is admitted; the other finds the entry in
/// flight, or already applied if the first has been published by then.
#[test]
fn of_two_concurrent_deliveries_of_one_entry_exactly_one_is_admitted() {
    for seed in 0..30 {
        let sim = Sim::new(seed);
        let t = env(&sim, 0);
        sim.run(|| {
            let seen: Arc<Mutex<Vec<Admission>>> = Arc::new(Mutex::new(Vec::new()));
            let handles: Vec<_> = (0..3)
                .map(|i| {
                    let (t, seen) = (t.clone(), seen.clone());
                    thread::spawn_named(&format!("deliver{i}"), |_| {}, move || {
                        let a = t.admit(1);
                        seen.lock().push(a);
                        if a == Admission::Admitted {
                            t.commit_sync(entry(&t, 1), Some(pos(1))).unwrap();
                        }
                    })
                    .unwrap()
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            let seen = seen.lock().clone();
            assert_eq!(seen.iter().filter(|a| **a == Admission::Admitted).count(), 1, "seed {seed}: {seen:?}");
            assert!(seen.iter().all(|a| matches!(a, Admission::Admitted | Admission::InFlight | Admission::AlreadyApplied)));
            assert_eq!(t.positions().applied_index, 1);
        });
    }
}
