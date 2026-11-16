//! Tests for the decisions recorded in history/Answer-20261115-1400.md: the WAL is synced before
//! redo reads it (A2, spec 9.3 step 4), a panicked thread is reported by `join` and by shutdown
//! (B23), the SQLSTATE of an unknown commit outcome (A6), and the block-size bound (B24).

use jamsil_engine::*;
use jamsil_page::control::{ControlData, DbState, SLOT_LEN};
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::env::{ManualClock, SeededRng};
use jamsil_vfs::{FaultPlan, FaultVfs, SimVfs, Vfs};
use jamsil_wal::WalError;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const REL: RelFileId = 100;
const PAGES: u32 = 24;
const SEG: u64 = 256 * 1024;

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

/// The fatal handler panics, so a test can kill the process at any I/O boundary of recovery
/// without ending the test binary.
fn opts(frames: usize) -> EngineOptions {
    EngineOptions {
        buffer_frames: frames,
        wal_segment_size: Some(SEG),
        wal_consistency_check: true,
        background_threads: false,
        clock: ManualClock::new(1_000_000),
        rng: SeededRng::new(0x5EED),
        fatal_handler: Some(Arc::new(|m: &str| panic!("FATAL {m}"))),
        ..Default::default()
    }
}

fn extend_raw(e: &NativeEngine, rel: RelFileId, n: u32) -> Result<(), EngineError> {
    let first = e.extend_relation(rel, ForkId::Main, n)?;
    e.init_raw_pages(e.alloc_xid().unwrap(), rel, ForkId::Main, first, n)
}

fn cell_off(c: usize) -> u16 {
    (64 + 8 * c) as u16
}

fn put(e: &NativeEngine, x: Xid, page: u32, c: usize, v: u64) {
    e.write_bytes(x, PageTag::new(REL, ForkId::Main, page), cell_off(c), &v.to_le_bytes()).unwrap();
}

fn get(e: &NativeEngine, page: u32, c: usize) -> u64 {
    let p = e.pin(PageTag::new(REL, ForkId::Main, page)).expect("page must verify");
    let g = p.read();
    rd_u64(&g[..], cell_off(c) as usize)
}

// ---- A2: the log is synced before redo reads it ----------------------------------------------

/// A process killed between the write and the sync of a commit: the log's tail is readable by the
/// next start but was never synced. `kill_after` is how many mutating calls the flush gets before
/// the process dies (1: the write, then death before the sync).
fn killed_between_write_and_sync(seed: u64, kill_after: i64) -> SimVfs {
    let sim = SimVfs::new(seed);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    // A pool larger than the relation: no eviction forces the log out before the commit does.
    let e = NativeEngine::open_or_create(v, root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, PAGES).unwrap();
    for p in 0..PAGES {
        let x = e.alloc_xid().unwrap();
        put(&e, x, p, 0, p as u64 + 1);
        e.commit(x, Some(AppliedPos { index: p as u64 + 1, term: 1 }), &[]).unwrap();
    }
    e.checkpoint().unwrap();
    // One transaction over every page; its records sit in the WAL buffer until the commit.
    let x = e.alloc_xid().unwrap();
    for p in 0..PAGES {
        put(&e, x, p, 1, 1000 + p as u64);
    }
    plan.die_after_ops.store(kill_after, Ordering::SeqCst);
    let r = catch_unwind(AssertUnwindSafe(|| e.commit(x, Some(AppliedPos { index: PAGES as u64 + 1, term: 1 }), &[])));
    assert!(r.is_err(), "the process must die inside the commit's flush");
    // A process kill: the engine is dropped and the file system keeps what is in its page cache.
    drop(e);
    sim
}

/// Spec 9.3 step 4. Redo reads a log tail that is not durable, applies it, and a small pool writes
/// pages out during redo. If the process then loses power before recovery ends, no page may carry
/// a `page_lsn` beyond the log that survives, or the next start would call it corrupt (spec 7.3
/// step 5). The sweep kills the second start at every I/O boundary.
#[test]
fn redo_never_runs_ahead_of_the_durable_log() {
    for seed in [7u64, 8] {
        for kill_after in [1i64, 2] {
            let mut deaths_inside_recovery = 0;
            // A start makes about 25 mutating calls, so 60 budgets cover every boundary and a few
            // runs that complete.
            for budget in 0..60i64 {
                let sim = killed_between_write_and_sync(seed, kill_after);
                let plan = FaultPlan::new();
                let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
                plan.die_after_ops.store(budget, Ordering::SeqCst);
                // A small pool: redo of 24 pages must evict, so pages are written out during redo.
                let r = catch_unwind(AssertUnwindSafe(|| NativeEngine::open(v, root(), opts(8)).map(drop)));
                if !matches!(r, Ok(Ok(()))) {
                    deaths_inside_recovery += 1;
                }
                sim.crash(); // power loss, wherever recovery had got to

                let plain: Arc<dyn Vfs> = Arc::new(sim.clone());
                let ctx = format!("seed {seed}, kill_after {kill_after}, budget {budget}");
                let e = NativeEngine::open(plain, root(), opts(64)).unwrap_or_else(|er| panic!("{ctx}: restart failed: {er}"));
                let n = e.verify_fork(REL, ForkId::Main).unwrap_or_else(|er| panic!("{ctx}: a page is not served: {er}"));
                assert_eq!(n, PAGES, "{ctx}");
                for p in 0..PAGES {
                    assert_eq!(get(&e, p, 0), p as u64 + 1, "{ctx}: acknowledged commit lost on page {p}");
                }
            }
            assert!(deaths_inside_recovery >= 15, "the sweep must kill recovery at many boundaries ({deaths_inside_recovery})");
        }
    }
}

/// Spec 9.3 step 2: redo runs whatever the state field says. The field only decides how the start is
/// reported.
#[test]
fn redo_runs_even_when_the_control_file_says_shut_down_cleanly() {
    let sim = SimVfs::new(31);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let e = NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 2).unwrap();
    e.shutdown().unwrap();
    drop(e);
    assert!(!NativeEngine::open(v.clone(), root(), opts(64)).unwrap().recovery.crash_recovery, "a clean shutdown is not a crash recovery");

    let e = NativeEngine::open(v.clone(), root(), opts(64)).unwrap();
    let x = e.alloc_xid().unwrap();
    put(&e, x, 1, 2, 4242);
    e.commit(x, Some(AppliedPos { index: 9, term: 1 }), &[]).unwrap();
    drop(e);
    sim.crash();

    // Make the control file lie: the checkpoint it names predates the commit, but it says "clean".
    let ctl = root().join("gen-000001").join("control");
    let raw = sim.file_bytes(&ctl).unwrap();
    let (mut cd, _) = ControlData::pick(&raw).unwrap();
    assert_ne!(cd.state, DbState::ShutDown);
    cd.state = DbState::ShutDown;
    let enc = cd.encode();
    sim.poke(&ctl, 0, &enc);
    sim.poke(&ctl, SLOT_LEN as u64, &enc);

    let e = NativeEngine::open(v, root(), opts(64)).unwrap();
    assert!(!e.recovery.crash_recovery, "the state field says clean, so the start is reported as clean");
    assert!(e.recovery.records_replayed > 0, "but redo must have run");
    assert_eq!(e.applied().index, 9, "the commit after the checkpoint must be replayed");
    assert_eq!(get(&e, 1, 2), 4242);
}

// ---- B23: a panicked thread is reported -------------------------------------------------------

/// A writer thread that died of a panic must show up at shutdown: the control file must not say
/// "shut down cleanly" on top of a log whose writer failed.
#[test]
fn shutdown_reports_a_writer_thread_that_panicked() {
    let sim = SimVfs::new(41);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    let o = EngineOptions { background_threads: true, ..opts(64) };
    let e = NativeEngine::open_or_create(v.clone(), root(), o).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 1).unwrap();
    e.checkpoint().unwrap();

    // The next flush by the writer fails its sync, and the test's fatal handler panics.
    e.set_fatal_handler(Arc::new(|m: &str| panic!("FATAL {m}")));
    plan.fail_all_syncs.store(true, Ordering::SeqCst);
    put(&e, e.alloc_xid().unwrap(), 0, 0, 1);
    let deadline = Instant::now() + Duration::from_secs(20);
    while !e.wal.is_poisoned() {
        assert!(Instant::now() < deadline, "the writer thread never ran into the failed sync");
        jamsil_vfs::sync::thread::sleep(Duration::from_millis(5));
    }
    plan.fail_all_syncs.store(false, Ordering::SeqCst);

    let err = e.shutdown().expect_err("a writer that panicked is not a clean stop");
    assert!(matches!(err, EngineError::Wal(WalError::WriterPanicked)), "{err}");
    assert_ne!(e.control_snapshot().state, DbState::ShutDown, "no clean shutdown may be declared");
}

#[test]
fn a_writer_thread_that_ends_normally_is_a_clean_stop() {
    let sim = SimVfs::new(42);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let o = EngineOptions { background_threads: true, ..opts(64) };
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    e.shutdown().unwrap();
    assert!(e.halt().is_ok(), "stopping twice is harmless");
}

// ---- A6: SQLSTATE of an unknown outcome -------------------------------------------------------

#[test]
fn an_unknown_commit_outcome_has_sqlstate_08007() {
    let err = EngineError::OutcomeUnknown("flush failed".into());
    assert_eq!(err.sqlstate(), Some("08007"));
    assert_ne!(err.sqlstate(), EngineError::DiskFull.sqlstate(), "the client must not read it as an abort or a disk error");
}

// ---- B24: the block-size bound ----------------------------------------------------------------

/// The size check also refuses a block payload beyond the 16-bit length field, with the same
/// error (54000) as an oversized record, before anything is inserted.
#[test]
fn check_size_refuses_a_block_payload_beyond_the_length_field() {
    let sim = SimVfs::new(43);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    // 1 MiB segments: the record limit (a quarter of a segment) must lie above the field's maximum.
    let e = NativeEngine::open_or_create(v, root(), EngineOptions { wal_segment_size: Some(1 << 20), ..opts(64) }).unwrap();
    let tag = PageTag::new(REL, ForkId::Main, 0);
    let ok = vec![1u8; MAX_BLOCK_DATA];
    assert!(e.wal.check_size(&RecordBuilder::new(rmgr::HEAP, 1).block(BlockRef::new(tag).data(&ok))).is_ok());
    let big = vec![1u8; MAX_BLOCK_DATA + 1];
    let err = e.wal.check_size(&RecordBuilder::new(rmgr::HEAP, 1).block(BlockRef::new(tag).data(&big))).unwrap_err();
    assert!(matches!(err, WalError::RecordTooLarge { len: 65_536, max: 65_535 }), "{err}");
}

// ---- Section C: the length of a fork ----------------------------------------------------------

/// Spec 6.7: the length is the number of whole pages, and a trailing partial page is a torn
/// extension that redo formats again. Process killed after the extend record became durable but
/// before any page was written; the file holds three whole pages and the start of a fourth.
#[test]
fn redo_formats_a_torn_trailing_page_again() {
    let sim = SimVfs::new(51);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let e = NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    e.extend_relation(REL, ForkId::Main, 5).unwrap();
    e.commit(e.alloc_xid().unwrap(), None, &[]).unwrap(); // the log is durable, the pages are not written
    drop(e);

    let file = root().join("gen-000001/base/0000000000000064.main.0");
    sim.poke(&file, 0, &vec![0u8; 3 * PAGE_SIZE + 100]);
    let e = NativeEngine::open(v, root(), opts(64)).unwrap();
    assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap(), 5, "every page, the torn one included, must be served");
}

// ---- A gap in spec 9.3: what a killed process left in the page cache ---------------------------

/// After a process kill the next start sees files whose directory entries no `sync_dir` covered,
/// and its checkpoint moves the redo point past the `CREATE` records that would recreate them.
/// Unless recovery syncs the directories first, a power loss right after that start loses the file
/// with nothing left to replay.
///
/// The test also holds an update that was written out to its file but never synced. That half does
/// not need a sync of the file: the first change to a page after a checkpoint logs a full-page
/// image, redo restores it, and the recovery checkpoint writes and syncs the page. It is kept as an
/// end-to-end check of that argument.
#[test]
fn recovery_syncs_what_a_killed_process_left_in_the_page_cache() {
    let page1_file = root().join("gen-000001/base/0000000000000064.main.0");
    let cell_at = PAGE_SIZE + cell_off(3) as usize;
    for seed in 60..68u64 {
        let sim = SimVfs::new(seed);
        let v: Arc<dyn Vfs> = Arc::new(sim.clone());
        // A small pool, so that churn on another relation pushes a dirty page out to its file.
        let e = NativeEngine::open_or_create(v.clone(), root(), opts(8)).unwrap();
        e.create_relation(REL, ForkId::Main).unwrap();
        extend_raw(&e, REL, 4).unwrap();
        e.checkpoint().unwrap(); // the first relation's file and directory entry are durable from here

        // An update of page 1, committed: the log is durable, the page is only in the pool.
        let x = e.alloc_xid().unwrap();
        put(&e, x, 1, 3, 777);
        e.commit(x, Some(AppliedPos { index: 3, term: 1 }), &[]).unwrap();
        // A second relation, created after the checkpoint: its directory entry is not synced. Its
        // pages push page 1 out of the pool, so the update reaches the file without a sync.
        let rel2 = REL + 1;
        e.create_relation(rel2, ForkId::Main).unwrap();
        for _ in 0..12 {
            let first = e.extend_relation(rel2, ForkId::Main, 1).unwrap();
            e.init_raw_pages(e.alloc_xid().unwrap(), rel2, ForkId::Main, first, 1).unwrap();
        }
        e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 4, term: 1 }), &[]).unwrap();
        let in_file = sim.file_bytes(&page1_file).unwrap();
        assert_eq!(&in_file[cell_at..cell_at + 8], &777u64.to_le_bytes(), "seed {seed}: the update must have been written out");
        drop(e); // process kill: nothing is lost from the page cache

        // The next start completes recovery and its end-of-recovery checkpoint ...
        drop(NativeEngine::open(v.clone(), root(), opts(8)).unwrap());
        // ... and then the power goes.
        sim.crash();

        let ctx = format!("seed {seed}");
        let e = NativeEngine::open(v, root(), opts(8)).unwrap_or_else(|er| panic!("{ctx}: {er}"));
        assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap(), 4, "{ctx}");
        assert_eq!(e.verify_fork(rel2, ForkId::Main).unwrap(), 12, "{ctx}: the second relation's file must still exist");
        assert_eq!(get(&e, 1, 3), 777, "{ctx}: an acknowledged update was lost");
        assert_eq!(e.applied().index, 4, "{ctx}");
    }
}

// ---- Spec 6.1 steps 2 and 3: what creation leaves behind --------------------------------------

/// A created database already has its log (the first segment and two spares at full size) and a
/// shutdown checkpoint record that the control file names, in both slots. The first start is an
/// ordinary recovery of that one record, not a special case.
#[test]
fn create_writes_the_log_and_a_shutdown_checkpoint_before_the_first_open() {
    let sim = SimVfs::new(71);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    NativeEngine::create(v.clone(), root(), &opts(64)).unwrap();
    let gen = root().join("gen-000001");

    let raw = sim.file_bytes(&gen.join("control")).unwrap();
    let (a, b) = (ControlData::decode(&raw[..SLOT_LEN]).unwrap(), ControlData::decode(&raw[SLOT_LEN..2 * SLOT_LEN]).unwrap());
    assert_eq!(a, b, "both slots are written");
    assert_eq!(a.state, DbState::ShutDown);
    assert_ne!(a.checkpoint_lsn, 0, "the control file names a checkpoint record");
    assert_eq!(a.redo_lsn, a.checkpoint_lsn, "a shutdown checkpoint's redo point is its own start");

    let mut segs: Vec<String> = v.list(&gen.join("wal")).unwrap().into_iter().map(|e| e.name).collect();
    segs.sort();
    assert_eq!(segs.len(), 3, "the first segment and two spares: {segs:?}");
    for n in &segs {
        assert_eq!(sim.file_bytes(&gen.join("wal").join(n)).unwrap().len() as u64, SEG, "{n} is created at full size");
    }

    let e = NativeEngine::open(v, root(), opts(64)).unwrap();
    assert!(!e.recovery.crash_recovery, "the first start after creation is not a crash recovery");
    assert_eq!(e.recovery.records_replayed, 1, "only the checkpoint record is replayed");
}

/// A control file that names no checkpoint record is damaged: creation always writes one.
#[test]
fn a_control_file_that_names_no_checkpoint_is_corruption() {
    let sim = SimVfs::new(72);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    NativeEngine::create(v.clone(), root(), &opts(64)).unwrap();
    let ctl = root().join("gen-000001").join("control");
    let raw = sim.file_bytes(&ctl).unwrap();
    let (mut cd, _) = ControlData::pick(&raw).unwrap();
    cd.checkpoint_lsn = 0;
    let enc = cd.encode();
    sim.poke(&ctl, 0, &enc);
    sim.poke(&ctl, SLOT_LEN as u64, &enc);
    let err = NativeEngine::open(v, root(), opts(64)).err().expect("must fail");
    assert!(matches!(err, EngineError::Corruption(_)), "{err}");
}
