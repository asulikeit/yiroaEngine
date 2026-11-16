//! Allocation in logged batches (spec 8.6, 9.3 "Xid counter"): the layout of what is logged, the order
//! of the records against the first use of an id, the commit-log pages of a batch, and the rule that
//! neither a batch record nor a page creation happens under the publication lock (answer A1).

use jamsil_page::txnrec::{self, clog_block, AllocBatch, RaftPos};
use jamsil_page::*;
use jamsil_txn::*;
use jamsil_vfs::sched::{ctl, Sim};
use jamsil_vfs::sync::{thread, Arc, Mutex};
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// What a task saw when it was handed an xid: its name, the xid, whether its commit-log page existed,
/// and how many batch records were in the log.
type Seen = Arc<Mutex<Vec<(&'static str, Xid, bool, usize)>>>;

struct Env {
    txn: Arc<TxnManager>,
    store: Arc<MemStore>,
    wal: Arc<MemWal>,
}

fn env_from(sim: &Sim, rec: Recovered) -> Env {
    env_batch(sim, rec, XID_BATCH)
}

fn env_batch(sim: &Sim, rec: Recovered, xid_batch: u64) -> Env {
    let (store, wal) = (Arc::new(MemStore::new()), Arc::new(MemWal::new()));
    let opts = TxnOptions { sessions: 64, xid_batch, ..Default::default() };
    let txn = Arc::new(TxnManager::new(store.clone(), wal.clone(), sim.clock(), opts, rec));
    Env { txn, store, wal }
}

fn env(sim: &Sim) -> Env {
    env_from(sim, Recovered { csn: 1000, ..Default::default() })
}

fn batches(wal: &MemWal) -> Vec<AllocBatch> {
    wal.records().into_iter().filter_map(|r| if let LoggedRecord::AllocBatch(b) = r { Some(b) } else { None }).collect()
}

/// The first xid logs a batch of 4,096 before it is handed out, and creates the commit-log page; the
/// next batch is logged exactly when the 4,097th xid is asked for.
#[test]
fn the_first_xid_of_a_batch_logs_it_first() {
    let sim = Sim::new(1);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        assert!(batches(&e.wal).is_empty() && e.store.pages().is_empty(), "nothing is logged before the first allocation");
        let mut tx = t.begin();
        assert_eq!(t.assign_xid(&mut tx).unwrap(), 16);
        assert_eq!(batches(&e.wal), [AllocBatch { next_xid: 16 + 4096, next_oid: FIRST_USER_OID, next_relfile: FIRST_RELFILE }]);
        assert_eq!(e.store.pages(), [0], "the commit-log page exists before the xid does");
        assert!(matches!(e.wal.records()[0], LoggedRecord::AllocBatch(_)), "and the batch is the first record in the log");
        t.abort(tx);

        for _ in 0..4095 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            t.abort(tx);
        }
        assert_eq!(batches(&e.wal).len(), 1, "xids 16 to 4111 are one batch");
        let mut tx = t.begin();
        assert_eq!(t.assign_xid(&mut tx).unwrap(), 16 + 4096);
        assert_eq!(batches(&e.wal).len(), 2, "the next one is logged when the batch is used up");
        assert_eq!(batches(&e.wal)[1].next_xid, 16 + 8192);
        t.abort(tx);
    });
}

/// After a restart the counters begin at the recovered values, and a batch is logged before the first
/// use. A batch that crosses into a commit-log page that does not exist yet creates it first; the page
/// the recovered counter lies in exists already, and is not created again (that would zero it).
#[test]
fn a_restart_begins_at_the_recovered_counter_and_creates_only_the_pages_that_are_new() {
    let sim = Sim::new(2);
    // 32,640 xids fit in a page: the batch from 32,540 to 36,636 crosses into page 1.
    let e = env_from(&sim, Recovered { next_xid: 32_540, csn: 1000, ..Default::default() });
    e.store.clog_create_pages(0, 0).unwrap(); // what the restart finds on disk
    sim.run(|| {
        let t = &e.txn;
        let mut tx = t.begin();
        assert_eq!(t.assign_xid(&mut tx).unwrap(), 32_540, "no id is skipped by a restart");
        assert_eq!(e.store.created_by_calls(), [(0, 0), (1, 1)], "page 0 holds the xids below the recovered counter: it exists, and is not zeroed again");
        assert_eq!(batches(&e.wal), [AllocBatch { next_xid: 32_540 + 4096, next_oid: FIRST_USER_OID, next_relfile: FIRST_RELFILE }]);
        t.abort(tx);
    });
}

/// Oids and relfile ids come in batches of 1,024, and every record carries the current end of all three.
#[test]
fn oids_and_relfile_ids_are_batched_and_every_record_carries_all_three_marks() {
    let sim = Sim::new(3);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let oids: Vec<_> = (0..1025).map(|_| t.alloc_oid().unwrap()).collect();
        assert_eq!(oids[0], FIRST_USER_OID);
        assert!(oids.windows(2).all(|w| w[1] == w[0] + 1));
        let rels: Vec<_> = (0..3).map(|_| t.alloc_relfile().unwrap()).collect();
        assert_eq!(rels, [16, 17, 18]);
        let mut tx = t.begin();
        t.assign_xid(&mut tx).unwrap();
        t.abort(tx);
        assert_eq!(
            batches(&e.wal),
            [
                AllocBatch { next_xid: 16, next_oid: 16_384 + 1024, next_relfile: 16 },
                AllocBatch { next_xid: 16, next_oid: 16_384 + 2048, next_relfile: 16 },
                AllocBatch { next_xid: 16, next_oid: 16_384 + 2048, next_relfile: 16 + 1024 },
                AllocBatch { next_xid: 16 + 4096, next_oid: 16_384 + 2048, next_relfile: 16 + 1024 },
            ],
            "the xid field is the restart value until a batch of xids exists"
        );
        // A checkpoint records the same three marks (spec 9.2 step 2).
        let cap = t.checkpoint_capture(|| e.wal.insert_lsn());
        assert_eq!(cap.alloc, AllocBatch { next_xid: 16 + 4096, next_oid: 16_384 + 2048, next_relfile: 16 + 1024 });
        assert_eq!(cap.alloc, *batches(&e.wal).last().unwrap(), "and it is the last batch in the log");
    });
}

/// An xid is never handed out without a logged batch: when the log accepts no records and the batch is
/// used up, the allocation fails (JDB-3001). Inside the batch it goes on.
#[test]
fn a_quarantined_log_stops_the_allocation_only_when_the_batch_is_used_up() {
    let sim = Sim::new(4);
    let e = env(&sim);
    sim.run(|| {
        let t = &e.txn;
        let mut first = t.begin();
        t.assign_xid(&mut first).unwrap();
        e.wal.close();
        for _ in 0..4095 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).expect("inside the logged batch");
            t.abort(tx);
        }
        let mut tx = t.begin();
        assert!(matches!(t.assign_xid(&mut tx), Err(TxnError::Quarantined(_))));
        assert_eq!(tx.xid(), None, "the transaction has no xid: nothing was handed out");
        assert!(matches!(t.alloc_oid(), Err(TxnError::Quarantined(_))));
        e.wal.reopen();
        assert!(t.assign_xid(&mut tx).is_ok(), "and it goes on when the log accepts records again");
        t.abort(tx);
        t.abort(first);
        assert_eq!(batches(&e.wal).len(), 2);
    });
}

/// The end of the id space is an error, not a wrap-around.
#[test]
fn the_end_of_the_id_space_is_an_error() {
    let sim = Sim::new(5);
    let e = env_from(&sim, Recovered { next_xid: u64::MAX - 10, next_oid: u32::MAX - 5, csn: 1000, ..Default::default() });
    sim.run(|| {
        let mut tx = e.txn.begin();
        assert_eq!(e.txn.assign_xid(&mut tx), Err(TxnError::Exhausted("xid")));
        assert_eq!(e.txn.alloc_oid(), Err(TxnError::Exhausted("oid")));
    });
}

/// A `WalSink` and a `TxnStore` that check, at every insert and every page creation, that the
/// publication lock is free (spec 8.6: "Neither an allocation batch nor a zero-page record is inserted
/// while the publication lock is held"). The test is single-threaded, so the probe is exact.
struct Probe {
    wal: Arc<MemWal>,
    store: Arc<MemStore>,
    txn: OnceLock<Arc<TxnManager>>,
    checks: Mutex<Vec<&'static str>>,
}

impl Probe {
    fn free(&self, what: &'static str) {
        let t = self.txn.get().expect("set before use");
        assert!(!t.published().lock_is_held(), "{what} happened under the publication lock");
        self.checks.lock().push(what);
    }
}

impl WalSink for Probe {
    fn gate_commit(&self, r: &txnrec::CommitRecord) -> Result<Staged, TxnError> {
        self.wal.gate_commit(r)
    }
    fn gate_applied_advance(&self) -> Result<jamsil_page::walrec::WalPermit, TxnError> {
        self.wal.gate_applied_advance()
    }
    fn gate_abort(&self, r: &txnrec::AbortRecord) -> Result<Staged, TxnError> {
        self.wal.gate_abort(r)
    }
    fn insert_commit(&self, s: Staged, x: Xid) -> Csn {
        self.free("a commit record");
        self.wal.insert_commit(s, x)
    }
    fn insert_applied_advance(&self, p: jamsil_page::walrec::WalPermit, pos: RaftPos) -> Csn {
        self.free("an applied-advance record");
        self.wal.insert_applied_advance(p, pos)
    }
    fn insert_abort(&self, s: Staged, x: Xid) -> Lsn {
        self.free("an abort record");
        self.wal.insert_abort(s, x)
    }
    fn log_alloc_batch(&self, b: AllocBatch) -> Result<(), TxnError> {
        self.free("an allocation batch");
        self.wal.log_alloc_batch(b)
    }
    fn flush(&self, upto: Lsn) -> Result<(), TxnError> {
        self.wal.flush(upto)
    }
}

impl TxnStore for Probe {
    fn clog_get(&self, x: Xid) -> Result<txnrec::ClogStatus, TxnError> {
        self.store.clog_get(x)
    }
    fn clog_gate(&self, x: &[Xid], csn: bool) -> Result<ClogGate, TxnError> {
        // The gate pins pages and may read them: never under the publication lock.
        self.free("a gate (pins and permits)");
        self.store.clog_gate(x, csn)
    }
    fn clog_set_logged(&self, g: &mut ClogGate, set: &[(Xid, txnrec::ClogStatus)], log: &mut dyn FnMut() -> Lsn) -> Lsn {
        self.free("a change to the commit log");
        self.store.clog_set_logged(g, set, log)
    }
    fn clog_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        self.free("the creation of a commit-log page");
        self.store.clog_create_pages(first, last)
    }
    fn csn_create_pages(&self, first: BlockNo, last: BlockNo) -> Result<(), TxnError> {
        self.free("the creation of a CSN-log page");
        self.store.csn_create_pages(first, last)
    }
    fn csn_get(&self, x: Xid) -> Result<Csn, TxnError> {
        self.store.csn_get(x)
    }
    fn csn_set(&self, g: &ClogGate, x: Xid, c: Csn) {
        // The barrier runs this inside the publication lock, which is where it belongs: it must do no I/O,
        // and that is the store's rule (the gate pinned the page).
        self.store.csn_set(g, x, c)
    }
}

#[test]
fn nothing_is_logged_or_created_under_the_publication_lock() {
    let sim = Sim::new(6);
    let probe = Arc::new(Probe { wal: Arc::new(MemWal::new()), store: Arc::new(MemStore::new()), txn: OnceLock::new(), checks: Mutex::new(Vec::new()) });
    let t = Arc::new(TxnManager::new(
        probe.clone(),
        probe.clone(),
        sim.clock(),
        TxnOptions { sessions: 64, ..Default::default() },
        Recovered { next_xid: 32_600, csn: 1000, ..Default::default() },
    ));
    probe.txn.set(t.clone()).ok().unwrap();
    probe.store.clog_create_pages(0, 0).unwrap(); // the page of the recovered counter exists already
    probe.store.csn_create_pages(32_600 / 1020, 32_600 / 1020).unwrap(); // and so does the CSN-log page of its first xid
    sim.run(|| {
        // A control: the probe can tell. Inside the gate's critical section the lock is held.
        assert!(!t.published().lock_is_held());
        let (none, mut held) = (std::collections::BTreeSet::new(), false);
        let req = GateRequest { token: 999, snapshot_csn: 0, read: &none, write: &none, ddl: false };
        t.published().prepare(&req, || held = t.published().lock_is_held());
        t.published().finish_abort_with(&[], Some(999));
        assert!(held, "the probe sees the publication lock held while the gate runs");
        assert!(!t.published().lock_is_held());
        for i in 0..300u64 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            let sub = t.assign_sub_xid(&mut tx).unwrap();
            if i % 3 == 0 {
                t.rollback_sub_xid(&mut tx, sub, &[]);
            }
            if i % 2 == 0 {
                t.commit_sync(tx, Some(RaftPos { index: i / 2 + 1, term: 1 })).unwrap();
            } else {
                t.abort(tx);
            }
        }
        for _ in 0..3000 {
            let mut tx = t.begin();
            t.assign_xid(&mut tx).unwrap();
            t.abort(tx);
        }
        t.alloc_oid().unwrap();
        t.alloc_relfile().unwrap();
        let checks = probe.checks.lock().clone();
        for what in [
            "an allocation batch",
            "the creation of a commit-log page",
            "the creation of a CSN-log page",
            "a gate (pins and permits)",
            "a commit record",
            "an abort record",
            "a change to the commit log",
        ] {
            assert!(checks.contains(&what), "the workload exercised {what}");
        }
    });
}

/// While one thread is halfway through making a batch, no other thread gets an xid of it: not before
/// the commit-log pages exist, and not before the batch record is in the log. The allocating thread is
/// held at a named suspension point inside the slow path (spec 19.7) and the other one is given every
/// chance.
#[test]
fn nobody_gets_an_xid_of_a_batch_that_is_half_made() {
    for point in ["alloc:before_pages", "alloc:before_limit"] {
        let sim = Sim::new(31);
        // The recovered counter is the first xid of a page that does not exist yet.
        let e = env_batch(&sim, Recovered { next_xid: 32_640, csn: 1000, ..Default::default() }, 64);
        e.store.clog_create_pages(0, 0).unwrap();
        sim.run(|| {
            ctl::hold(point);
            let got: Seen = Arc::new(Mutex::new(Vec::new()));
            let mut handles = Vec::new();
            for (name, wait_for_hold) in [("first", false), ("second", true)] {
                if wait_for_hold {
                    ctl::await_point(point); // the first thread is inside the slow path now
                }
                let (t, store, wal, got) = (e.txn.clone(), e.store.clone(), e.wal.clone(), got.clone());
                handles.push(
                    thread::spawn_named(
                        name,
                        |_| {},
                        move || {
                            let mut tx = t.begin();
                            let x = t.assign_xid(&mut tx).unwrap();
                            let page = store.pages().contains(&clog_block(x).unwrap());
                            got.lock().push((name, x, page, batches(&wal).len()));
                            t.abort(tx);
                        },
                    )
                    .unwrap(),
                );
            }
            thread::sleep(std::time::Duration::from_millis(50));
            assert!(got.lock().is_empty(), "{point}: nobody has an xid while the batch is half made: {:?}", got.lock().clone());
            ctl::release(point);
            for h in handles {
                h.join().unwrap();
            }
            let got = got.lock().clone();
            assert_eq!(got.len(), 2);
            assert!(got.iter().all(|(_, _, page, batch)| *page && *batch == 1), "{point}: page and batch record exist when the xid is handed out: {got:?}");
            assert_ne!(got[0].1, got[1].1);
        });
    }
}

/// Concurrent allocation under random schedules, with batches of 64 so that there are many
/// boundaries. Each seed restarts exactly at the first xid of a commit-log page that does not exist
/// yet, so the first xid of the first batch needs the page the batch creates: a thread that is handed
/// an xid before the page exists is caught. Every xid is handed out once, its page exists when it is
/// handed out, the batch records follow each other, and the last one covers every xid.
#[test]
fn concurrent_allocation_never_hands_an_xid_out_before_its_page_or_twice() {
    for seed in 0..12u64 {
        let sim = Sim::new(seed);
        let start = 32_640 * (seed + 1);
        let e = env_batch(&sim, Recovered { next_xid: start, csn: 1000, ..Default::default() }, 64);
        e.store.clog_create_pages(0, seed as u32).unwrap(); // the pages below the recovered counter exist already
        sim.run(|| {
            let seen: Arc<Mutex<Vec<Xid>>> = Arc::new(Mutex::new(Vec::new()));
            let mut handles = Vec::new();
            for task in 0..4 {
                let (t, store, seen) = (e.txn.clone(), e.store.clone(), seen.clone());
                handles.push(
                    thread::spawn_named(
                        &format!("alloc{task}"),
                        |_| {},
                        move || {
                            for _ in 0..100 {
                                let mut tx = t.begin();
                                let x = t.assign_xid(&mut tx).unwrap();
                                assert!(store.pages().contains(&clog_block(x).unwrap()), "xid {x} was handed out before its commit-log page");
                                seen.lock().push(x);
                                t.abort(tx);
                            }
                        },
                    )
                    .unwrap(),
                );
            }
            for h in handles {
                h.join().unwrap();
            }
            let seen = seen.lock().clone();
            let unique: BTreeSet<_> = seen.iter().copied().collect();
            assert_eq!(unique.len(), seen.len(), "seed {seed}: no xid twice");
            assert_eq!(seen.len(), 400);
            let b = batches(&e.wal);
            assert!(b.windows(2).all(|w| w[0].next_xid + 64 == w[1].next_xid), "seed {seed}: batches follow each other without a gap: {b:?}");
            assert_eq!(b.last().unwrap().next_xid, start + 7 * 64, "seed {seed}: 400 xids need seven batches of 64");
            assert!(*unique.last().unwrap() < b.last().unwrap().next_xid);
            assert_eq!(e.store.created_by_calls(), [(0, seed as u32), (seed as u32 + 1, seed as u32 + 1)], "the new page once, and no page again");
        });
    }
}
