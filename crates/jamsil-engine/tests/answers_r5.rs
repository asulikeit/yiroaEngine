//! Tests for the decisions recorded in history/Answer-20261115-1900.md and for `jamsil-txn`:
//! the engine under the deterministic scheduler (spec 19.2), and the transaction layer's scenarios.

use jamsil_engine::*;
use jamsil_vfs::env::SeededRng;
use jamsil_vfs::sched::Sim;
use jamsil_vfs::sync::thread;
use jamsil_vfs::{SimVfs, Vfs};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const SEG: u64 = 256 * 1024;

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

/// Engine options for a run under the scheduler: the background threads are on, and the clock is
/// the simulation's.
fn sim_opts(sim: &Sim, full: bool) -> EngineOptions {
    EngineOptions {
        buffer_frames: 64,
        wal_segment_size: Some(SEG),
        synchronous_full: full,
        background_threads: true,
        clock: sim.clock(),
        rng: SeededRng::new(0x5EED),
        fatal_handler: Some(Arc::new(|m: &str| panic!("FATAL {m}"))),
        ..Default::default()
    }
}

/// Spec 8.5 and 17.4: under `NORMAL` a commit is acknowledged before its record is synced, and the
/// WAL writer syncs it within 200 ms. The writer runs as a task of the scheduler and its timer
/// reads the virtual clock, so the bound is exact and costs no real time: this cannot be tested any
/// other way.
#[test]
fn normal_durability_is_bounded_by_200_ms_of_virtual_time() {
    for seed in 0..10u64 {
        let sim = Sim::new(seed);
        let o = sim_opts(&sim, false);
        sim.run(|| {
            let v: Arc<dyn Vfs> = Arc::new(SimVfs::new(seed));
            let e = NativeEngine::open_or_create(v, root(), o).unwrap();
            let pos = AppliedPos { index: 1, term: 1 };
            let end = e.commit(e.alloc_xid().unwrap(), Some(pos), &[]).unwrap();
            assert!(e.wal.durable_lsn() < end, "seed {seed}: NORMAL replies before the sync");
            assert_eq!(e.durable_applied_index(), 0);

            thread::sleep(Duration::from_millis(100));
            assert!(e.wal.durable_lsn() < end, "seed {seed}: the writer's timer has not fired yet");
            thread::sleep(Duration::from_millis(110));
            assert!(e.wal.durable_lsn() >= end, "seed {seed}: within 200 ms the writer has synced the commit");
            assert_eq!(e.durable_applied_index(), 1, "and `durable_applied_index` follows");
            e.halt().unwrap();
        });
    }
}

/// The same engine, run twice under one seed, ends in the same bytes: the disk and the scheduler
/// are driven by one seed (spec 19.2).
#[test]
fn an_engine_run_under_the_scheduler_replays_from_its_seed() {
    fn run(seed: u64) -> (u64, std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>) {
        let sim = Sim::new(seed);
        let vfs = SimVfs::new(seed);
        let (o, v2) = (sim_opts(&sim, true), vfs.clone());
        sim.run(|| {
            let v: Arc<dyn Vfs> = Arc::new(v2);
            let e = NativeEngine::open_or_create(v, root(), o).unwrap();
            for i in 1..=3u64 {
                e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: i, term: 1 }), &[]).unwrap();
            }
            e.halt().unwrap();
        });
        (sim.schedule_hash(), vfs.dump())
    }
    assert_eq!(run(11), run(11), "the same seed replays identically, schedule and bytes");
    assert_ne!(run(11).0, run(12).0, "another seed schedules differently");
}

// ---- Spec 8.7: the strict decoders, at recovery -----------------------------------------------

/// A transaction record that does not decode is corruption, not something to skip (spec 8.7, 18.2).
/// One trailing byte on a commit record is enough.
#[test]
fn a_commit_record_that_does_not_decode_stops_recovery_with_corruption() {
    use jamsil_page::txnrec::CommitRecord;
    use jamsil_page::walrec::{rmgr, xact, RecordBuilder};
    let sim = SimVfs::new(91);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let o = EngineOptions { background_threads: false, wal_segment_size: Some(SEG), fatal_handler: Some(Arc::new(|m: &str| panic!("FATAL {m}"))), ..Default::default() };
    let e = NativeEngine::open_or_create(v.clone(), root(), o.clone()).unwrap();
    let mut bad = CommitRecord { commit_time: 5, ..Default::default() }.encode();
    bad.push(0); // one byte too long
    let rec = RecordBuilder::new(rmgr::XACT, xact::COMMIT).xid(20).main(&bad);
    let permit = e.wal.check_size(&rec).unwrap();
    let end = e.wal.insert(permit, rec).end;
    e.wal.flush(end).unwrap();
    drop(e);
    let err = NativeEngine::open(v, root(), o).err().expect("recovery must refuse the record");
    assert!(matches!(err, EngineError::Corruption(_)), "{err}");
    assert_eq!(err.sqlstate(), Some("XX001"));
}

/// The engine's own commit, abort and applied-advance records are in the layouts of spec 8.7.
#[test]
fn the_engines_transaction_records_are_in_the_layouts_of_the_specification() {
    use jamsil_page::txnrec::*;
    use jamsil_page::walrec::{rmgr, xact};
    let sim = SimVfs::new(92);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let o = EngineOptions { background_threads: false, wal_segment_size: Some(SEG), ..Default::default() };
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    let x = e.alloc_xid().unwrap();
    e.commit(x, Some(AppliedPos { index: 7, term: 2 }), &[1000, 1001]).unwrap();
    e.commit(e.alloc_xid().unwrap(), None, &[]).unwrap(); // node-local
    e.applied_advance(AppliedPos { index: 8, term: 2 }).unwrap();
    e.abort(e.alloc_xid().unwrap());
    e.wal.flush_all().unwrap(); // an abort is not flushed by itself, and the reader sees only flushed log
    let mut rd = e.wal.reader(e.wal.segment_size());
    let mut recs = Vec::new();
    while let Ok(r) = rd.next_record() {
        if r.rmgr == rmgr::XACT {
            recs.push(r);
        }
    }
    let c = CommitView::parse(&recs[0].main).unwrap();
    assert_eq!((recs[0].info, recs[0].xid), (xact::COMMIT, x));
    assert_eq!(c.raft(), Some(RaftPos { index: 7, term: 2 }));
    assert_eq!(c.unlink().collect::<Vec<_>>(), [1000, 1001]);
    assert_eq!(recs[0].main.len(), 44 + 16);
    assert_eq!(CommitView::parse(&recs[1].main).unwrap().raft(), None, "a node-local commit has no Raft position");
    assert_eq!(decode_applied_advance(&recs[2].main).unwrap(), RaftPos { index: 8, term: 2 });
    assert_eq!(AbortRecord::decode(&recs[3].main).unwrap(), AbortRecord::default());
}
