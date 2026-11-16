//! Tests for the decisions recorded in history/Answer-20261115-0811.md: creation and open (A2, A3),
//! spare segments (A6), torn extension pages (A9), the hint-bit rule (A11), the background WAL
//! writer (B2), latch rules (B6) and the injectable clock and random source (B10, A2).

use jamsil_engine::*;
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::env::{ManualClock, SeededRng};
use jamsil_vfs::{FaultPlan, FaultVfs, OpenOpts, SimVfs, Vfs};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const REL: RelFileId = 100;
const SEG: u64 = 256 * 1024;

fn opts() -> EngineOptions {
    EngineOptions {
        buffer_frames: 64,
        wal_segment_size: Some(SEG),
        wal_consistency_check: true,
        background_threads: false,
        ..Default::default()
    }
}
fn root() -> &'static Path {
    Path::new("/db/rdb")
}
fn gen1() -> PathBuf {
    root().join("gen-000001")
}
fn sim_vfs(seed: u64) -> (SimVfs, Arc<dyn Vfs>) {
    let sim = SimVfs::new(seed);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    (sim, v)
}
fn extend_raw(e: &NativeEngine, rel: RelFileId, n: u32) -> Result<(), EngineError> {
    let first = e.extend_relation(rel, ForkId::Main, n)?;
    e.init_raw_pages(e.alloc_xid().unwrap(), rel, ForkId::Main, first, n)
}
fn read_byte(e: &NativeEngine, page: u32, off: usize) -> u8 {
    let p = e.pin(PageTag::new(REL, ForkId::Main, page)).expect("page must verify");
    let g = p.read();
    g[off]
}
fn records(e: &NativeEngine) -> Vec<WalRecord> {
    let mut rd = e.wal.reader(e.wal.segment_size());
    let mut out = Vec::new();
    while let Ok(r) = rd.next_record() {
        out.push(r);
    }
    out
}

// ---- A2: create and open ---------------------------------------------------------------------

#[test]
fn open_on_an_empty_location_is_not_initialized_and_changes_nothing() {
    let (_sim, v) = sim_vfs(1);
    assert!(matches!(NativeEngine::open(v.clone(), root(), opts()), Err(EngineError::NotInitialized)));
    assert!(!v.exists(root()), "open must not create the directory");
    v.create_dir_all(root()).unwrap();
    assert!(matches!(NativeEngine::open(v.clone(), root(), opts()), Err(EngineError::NotInitialized)));
    assert!(v.list(root()).unwrap().is_empty(), "open must not write anything");
}

#[test]
fn create_builds_generation_one_and_refuses_a_second_creation() {
    let (_sim, v) = sim_vfs(2);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    let names: Vec<String> = v.list(root()).unwrap().into_iter().map(|e| e.name).collect();
    assert_eq!(names, ["CURRENT", "gen-000001"], "init.tmp and CURRENT.tmp must be gone: {names:?}");
    assert!(matches!(NativeEngine::create(v.clone(), root(), &opts()), Err(EngineError::AlreadyInitialized)));
    let e = NativeEngine::open(v, root(), opts()).unwrap();
    assert_eq!(e.wal_segment_size(), SEG);
}

#[test]
fn system_id_comes_from_the_injected_rng() {
    fn system_id(seed: u64) -> u64 {
        let (_sim, v) = sim_vfs(3);
        let o = EngineOptions { rng: SeededRng::new(seed), ..opts() };
        NativeEngine::open_or_create(v, root(), o).unwrap().control_snapshot().system_id
    }
    assert_eq!(system_id(5), system_id(5), "the same seed must give the same system_id");
    assert_ne!(system_id(5), system_id(6));
}

#[test]
fn every_directory_state_has_one_defined_outcome() {
    // CURRENT absent but a generation present: creation can never leave this, so it is permanent.
    let (_sim, v) = sim_vfs(4);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    v.unlink(&root().join("CURRENT")).unwrap();
    let err = NativeEngine::open(v.clone(), root(), opts()).err().expect("must fail");
    assert!(matches!(err, EngineError::FormatMismatch(_)), "{err}");
    assert_eq!(err.exit_code(), Some(78));

    // CURRENT names a generation that does not exist.
    let (_sim, v) = sim_vfs(5);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    let c = v.open(&root().join("CURRENT"), OpenOpts::create()).unwrap();
    c.write_at(b"gen-000007\n", 0).unwrap();
    let err = NativeEngine::open(v.clone(), root(), opts()).err().expect("must fail");
    assert_eq!(err.exit_code(), Some(78), "{err}");

    // CURRENT written, init.tmp not yet renamed: open finishes the creation.
    let (_sim, v) = sim_vfs(6);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    v.rename(&gen1(), &root().join("init.tmp")).unwrap();
    let e = NativeEngine::open(v.clone(), root(), opts()).unwrap();
    assert!(v.exists(&gen1()) && !v.exists(&root().join("init.tmp")));
    drop(e);
}

#[test]
fn creation_is_crash_safe_at_every_io_boundary() {
    // Creation now builds the log as well (spec 6.1 steps 2 and 3), so it makes a few hundred
    // mutating calls. A death inside the log's own flush is a fatal failure, which the test's
    // handler turns into a panic instead of the process exit of production.
    let o = EngineOptions { fatal_handler: Some(Arc::new(|m: &str| panic!("FATAL {m}"))), ..opts() };
    const BUDGETS: i64 = 400;
    let mut completed = 0;
    for budget in 0..BUDGETS {
        let (sim, _) = sim_vfs(budget as u64);
        let plan = FaultPlan::new();
        let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
        plan.die_after_ops.store(budget, Ordering::SeqCst);
        if matches!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| NativeEngine::create(v.clone(), root(), &o))), Ok(Ok(()))) {
            completed += 1;
        }
        plan.die_after_ops.store(-1, Ordering::SeqCst);
        sim.crash();
        let e = NativeEngine::open_or_create(v.clone(), root(), opts())
            .unwrap_or_else(|er| panic!("budget {budget}: the node cannot start after an interrupted creation: {er}"));
        e.create_relation(REL, ForkId::Main).unwrap();
        extend_raw(&e, REL, 2).unwrap();
        let x = e.alloc_xid().unwrap();
        e.write_bytes(x, PageTag::new(REL, ForkId::Main, 1), 64, &[7]).unwrap();
        e.commit(x, Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
        drop(e);
        sim.crash();
        let e = NativeEngine::open(v, root(), opts()).unwrap_or_else(|er| panic!("budget {budget}: reopen: {er}"));
        assert_eq!(read_byte(&e, 1, 64), 7, "budget {budget}");
    }
    assert!(completed > 0 && completed < BUDGETS, "the sweep must cover both interrupted and complete creations ({completed} completed)");
    assert!(completed > 150, "the sweep range should be mostly complete creations, so it reaches the last I/O boundary ({completed})");
}

// ---- A3: segment size ------------------------------------------------------------------------

#[test]
fn the_control_file_decides_the_segment_size() {
    let (_sim, v) = sim_vfs(7);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    let wrong = EngineOptions { wal_segment_size: Some(2 * SEG), ..opts() };
    let err = NativeEngine::open(v.clone(), root(), wrong).err().expect("a differing option is an error");
    assert_eq!(err.exit_code(), Some(78), "{err}");
    let unset = EngineOptions { wal_segment_size: None, ..opts() };
    assert_eq!(NativeEngine::open(v.clone(), root(), unset).unwrap().wal_segment_size(), SEG);

    for bad in [SEG / 2, 3 * SEG, 2 << 30, 1000] {
        let o = EngineOptions { wal_segment_size: Some(bad), ..opts() };
        let (_s, v2) = sim_vfs(8);
        assert!(matches!(NativeEngine::create(v2, root(), &o), Err(EngineError::FormatMismatch(_))), "size {bad}");
    }
    // The default is 64 MiB.
    let (_s, v3) = sim_vfs(9);
    let o = EngineOptions { wal_segment_size: None, ..opts() };
    assert_eq!(NativeEngine::open_or_create(v3, root(), o).unwrap().wal_segment_size(), 64 << 20);
}

// ---- A6: spare segments ----------------------------------------------------------------------

fn is_spare_shaped(sim: &SimVfs, segno: u64, system_id: u64) -> bool {
    let p = gen1().join("wal").join(format!("{segno:016x}.wal"));
    let Some(b) = sim.file_bytes(&p) else { return false };
    b.len() as u64 == SEG && check_seg_header(&b[..SEG_HEADER_LEN], system_id, segno).is_ok() && b[SEG_HEADER_LEN..].iter().all(|&z| z == 0)
}

#[test]
fn two_spares_are_kept_and_recovery_keeps_them() {
    let (sim, v) = sim_vfs(10);
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    let system_id = e.control_snapshot().system_id;
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 4).unwrap();
    // Fill more than one segment of log.
    let big = vec![5u8; 4000];
    while e.wal.insert_lsn() < SEG + SEG / 2 {
        e.write_bytes(e.alloc_xid().unwrap(), PageTag::new(REL, ForkId::Main, 0), 64, &big).unwrap();
    }
    e.wal.flush_all().unwrap();
    let cur = (e.wal.insert_lsn() - 1) / SEG;
    assert!(cur >= 1);
    for s in [cur + 1, cur + 2] {
        assert!(is_spare_shaped(&sim, s, system_id), "segment {s} must be a spare under its final name");
    }
    // A later segment that is not a spare goes away; the spares stay.
    let junk = gen1().join("wal").join(format!("{:016x}.wal", cur + 5));
    let f = v.open(&junk, OpenOpts::create()).unwrap();
    f.write_at(&encode_seg_header(system_id, cur + 5), 0).unwrap();
    f.write_at(&[0xEE; 64], 4096).unwrap();
    f.sync_data().unwrap();
    v.sync_dir(&gen1().join("wal")).unwrap();
    drop(e);
    sim.crash();
    let e = NativeEngine::open(v, root(), opts()).unwrap();
    let after = (e.wal.insert_lsn() - 1) / SEG;
    for s in [after + 1, after + 2] {
        assert!(is_spare_shaped(&sim, s, system_id), "segment {s} must survive recovery as a spare");
    }
    assert!(!sim.file_bytes(&junk).is_some(), "a later segment holding other bytes is removed");
}

// ---- A9: extension pages ---------------------------------------------------------------------

/// Spec 7.8, as changed by answer A2 of round 4: redo of an extend record formats every page of its
/// range again without reading it, so it neither fails on nor trusts what it finds there (this
/// replaces answer A9 of round 1, which left such a page alone). Everything that is later used lies
/// in the same redo range and rebuilds the page.
#[test]
fn redo_of_an_extension_formats_the_pages_without_reading_them() {
    let (sim, v) = sim_vfs(11);
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    e.extend_relation(REL, ForkId::Main, 4).unwrap();
    let x = e.alloc_xid().unwrap();
    e.init_and_write_bytes(x, PageTag::new(REL, ForkId::Main, 1), 64, &[9, 9, 9, 9]).unwrap();
    e.commit(x, None, &[]).unwrap();
    drop(e);
    sim.crash();
    // The file holds neither valid pages nor zeros at block 1: something that is not a page. Redo
    // must not read it, so it must not fail on it either.
    let path = gen1().join("base").join(format!("{REL:016x}.main.0"));
    let f = v.open(&path, OpenOpts::create()).unwrap();
    f.write_at(&vec![0xAB; PAGE_SIZE], PAGE_SIZE as u64).unwrap();
    f.sync_data().unwrap();
    v.sync_dir(&gen1().join("base")).unwrap();
    let e = NativeEngine::open(v.clone(), root(), opts()).expect("redo must not fail on the stray page");
    assert_eq!(read_byte(&e, 1, 64), 9, "the WILL_INIT record rebuilds the page");
    e.checkpoint().unwrap();
    drop(e);
    let e = NativeEngine::open(v, root(), opts()).unwrap();
    assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap(), 4, "every page verifies from disk");
}

// ---- A11: hint bits --------------------------------------------------------------------------

#[test]
fn a_hint_after_the_redo_point_logs_an_image_and_a_second_one_does_not() {
    let (sim, v) = sim_vfs(12);
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 2).unwrap();
    e.checkpoint().unwrap(); // the page's last change now lies below the redo point
    let images = |e: &NativeEngine| {
        e.wal.flush_all().unwrap(); // the reader sees only what is in the files
        records(e).iter().filter(|r| r.rmgr == rmgr::XLOG && r.info == xlog::PAGE_IMAGE).count()
    };
    let before = images(&e);
    let tag = PageTag::new(REL, ForkId::Main, 1);
    e.set_hint_byte(tag, 100, 0xAB).unwrap();
    assert_eq!(images(&e), before + 1, "the first hint after a checkpoint needs a full-page image");
    e.set_hint_byte(tag, 101, 0xCD).unwrap();
    assert_eq!(images(&e), before + 1, "a page logged since the redo point needs none");
    e.wal.flush_all().unwrap();
    drop(e);
    sim.crash();
    let e = NativeEngine::open(v, root(), opts()).unwrap();
    assert!(e.recovery.restored_images >= 1);
    // The image carried the first hint. The second one may legitimately be lost: a hint is advice.
    assert_eq!(read_byte(&e, 1, 100), 0xAB);
}

// ---- B2: the WAL writer thread ---------------------------------------------------------------

#[test]
fn the_writer_thread_flushes_a_log_nobody_waits_on() {
    let (_sim, v) = sim_vfs(13);
    let o = EngineOptions { synchronous_full: false, background_threads: true, ..opts() };
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 1).unwrap();
    let x = e.alloc_xid().unwrap();
    e.write_bytes(x, PageTag::new(REL, ForkId::Main, 0), 64, &[1]).unwrap();
    let end = e.commit(x, Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
    let t0 = Instant::now();
    while e.wal.durable_lsn() < end {
        assert!(t0.elapsed() < Duration::from_secs(3), "NORMAL commit not flushed by the writer within 3 s");
        jamsil_vfs::sync::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(e.durable_applied_index(), 1);
}

// ---- B6: latch rules -------------------------------------------------------------------------

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "commit flush while holding page latches")]
fn a_commit_flush_while_holding_a_page_latch_is_rejected() {
    let (_sim, v) = sim_vfs(14);
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 1).unwrap();
    let pinned = e.pin(PageTag::new(REL, ForkId::Main, 0)).unwrap();
    let _latch = pinned.write();
    let _ = e.commit(e.alloc_xid().unwrap(), None, &[]);
}

// ---- B10: injectable clock -------------------------------------------------------------------

#[test]
fn the_commit_record_takes_its_timestamp_from_the_injected_clock() {
    let (_sim, v) = sim_vfs(15);
    let clock = ManualClock::new(1_700_000_000_000_000);
    let o = EngineOptions { clock: clock.clone(), ..opts() };
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
    clock.advance(5);
    e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
    let ts: Vec<u64> =
        records(&e).iter().filter(|r| r.rmgr == rmgr::XACT && r.info == xact::COMMIT).map(|r| rd_u64(&r.main, 20)).collect();
    assert_eq!(ts, [1_700_000_000_000_000, 1_700_000_000_000_005]);
}

// ---- round 2 answers -------------------------------------------------------------------------

/// A5: hints racing checkpoints and write-outs, then a crash. Probabilistic: it cannot prove the
/// ordering argument of spec 7.6, but a hint that escaped its image would leave a torn page.
#[test]
fn hints_racing_checkpoints_leave_no_unrepairable_page() {
    for seed in 0..4u64 {
        let (sim, v) = sim_vfs(100 + seed);
        let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
        e.create_relation(REL, ForkId::Main).unwrap();
        extend_raw(&e, REL, 8).unwrap();
        e.commit(e.alloc_xid().unwrap(), None, &[]).unwrap();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|sc| {
            sc.spawn(|| {
                let mut i = 0u64;
                while !stop.load(Ordering::SeqCst) {
                    i += 1;
                    let tag = PageTag::new(REL, ForkId::Main, (i % 8) as u32);
                    e.set_hint_byte(tag, 100 + (i % 50) as u16, (i % 251) as u8).unwrap();
                }
            });
            sc.spawn(|| {
                while !stop.load(Ordering::SeqCst) {
                    e.checkpoint().unwrap();
                    e.pool.bg_write(8);
                }
            });
            jamsil_vfs::sync::thread::sleep(Duration::from_millis(300));
            stop.store(true, Ordering::SeqCst);
        });
        e.wal.flush_all().unwrap();
        drop(e);
        sim.crash();
        let e = NativeEngine::open(v, root(), opts()).unwrap_or_else(|er| panic!("seed {seed}: {er}"));
        assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap(), 8, "seed {seed}");
    }
}

/// A10: a failed spare raises the space-low flag and the log goes on into the spares that exist;
/// the quarantine starts only when a segment is needed and there is none. The commit in flight is
/// "outcome unknown"; later ones fail definitely; a restart clears it.
#[test]
fn a_full_disk_raises_space_low_first_and_quarantines_only_when_spares_run_out() {
    let sim = SimVfs::new(21);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 1).unwrap();
    e.wal.flush_all().unwrap();
    assert!(!e.space_low());
    plan.disk_full.store(true, Ordering::SeqCst);

    let big = vec![3u8; 4000];
    let (mut acked, mut first_error) = (0u64, None);
    let mut space_low_before_error = false;
    for i in 1..=2000u64 {
        let x = e.alloc_xid().unwrap();
        let tag = PageTag::new(REL, ForkId::Main, 0);
        let r = e.write_bytes(x, tag, 64, &big).map(|_| ()).and_then(|_| e.commit(x, Some(AppliedPos { index: i, term: 1 }), &[]).map(|_| ()));
        match r {
            Ok(()) => {
                acked = i;
                space_low_before_error |= e.space_low();
            }
            Err(er) => {
                first_error = Some((i, er));
                break;
            }
        }
    }
    let (failed_at, err) = first_error.expect("the disk is full: the log must run out of spares eventually");
    assert!(space_low_before_error, "the flag must be raised while commits still succeed");
    // Every record carries a full-page image under wal_consistency_check, so the current segment and
    // its two spares hold about 60 commits; the point is that the log kept going into the spares.
    assert!(acked >= 30 && e.wal.insert_lsn() > 3 * SEG, "the log must go on into its spares (acked {acked})");
    assert!(matches!(err, EngineError::OutcomeUnknown(_)), "the commit in flight is outcome unknown, got: {err}");
    assert_eq!(err.sqlstate(), Some("08007"), "a client must never read an unknown outcome as an abort");
    assert!(e.wal.is_quarantined());
    // Later commits never reach the file: a definite failure.
    let x = e.alloc_xid().unwrap();
    let later = e.commit(x, Some(AppliedPos { index: failed_at + 1, term: 1 }), &[]);
    assert!(matches!(later, Err(EngineError::Wal(_))), "later commit: {later:?}");

    // A restart clears this kind of quarantine; nothing acknowledged is lost, and the commit whose
    // outcome was unknown is either there or not.
    drop(e);
    sim.crash();
    plan.disk_full.store(false, Ordering::SeqCst);
    let e = NativeEngine::open(v, root(), opts()).unwrap();
    let got = e.applied().index;
    assert!(got >= acked && got <= failed_at, "applied {got}, acked {acked}, unknown {failed_at}");
    assert!(!e.wal.is_quarantined() && !e.space_low());
    e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: got + 1, term: 1 }), &[]).unwrap();
}

/// B18: a commit record whose size depends on user data is measured before the critical section.
#[test]
fn an_oversized_commit_record_fails_with_54000_before_anything_is_logged() {
    let (_sim, v) = sim_vfs(22);
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    let x = e.alloc_xid().unwrap(); // allocating an xid batch logs a record of its own
    let before = e.wal.insert_lsn();
    let too_many: Vec<RelFileId> = (1000..1000 + 9000).collect(); // 72 KB against a 64 KB limit
    let err = e.commit(x, Some(AppliedPos { index: 1, term: 1 }), &too_many).unwrap_err();
    assert!(matches!(err, EngineError::RecordTooLarge { .. }), "{err}");
    assert_eq!(err.sqlstate(), Some("54000"));
    assert_eq!(e.wal.insert_lsn(), before, "nothing may be inserted for a record that cannot fit");
    assert_eq!(e.applied().index, 0, "the failed commit must not move the applied position");
    let fits: Vec<RelFileId> = (1000..1100).collect();
    e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 1, term: 1 }), &fits).unwrap();
}

/// A1 table, row 3 and B20: a clean open removes `init.tmp/`; `open_or_create` creates only in the
/// `NotInitialized` state and never in the two failure states.
#[test]
fn open_clears_a_stale_init_directory_and_open_or_create_respects_the_failure_states() {
    let (_sim, v) = sim_vfs(23);
    NativeEngine::create(v.clone(), root(), &opts()).unwrap();
    let stale = root().join("init.tmp");
    v.create_dir_all(&stale.join("wal")).unwrap();
    v.open(&stale.join("control"), OpenOpts::create()).unwrap();
    drop(NativeEngine::open(v.clone(), root(), opts()).unwrap());
    assert!(!v.exists(&stale), "init.tmp must be removed when CURRENT names an existing generation");

    // Failure state 1: a generation without CURRENT.
    v.unlink(&root().join("CURRENT")).unwrap();
    let listing = |v: &Arc<dyn Vfs>| -> Vec<String> { v.list(root()).unwrap().into_iter().map(|e| e.name).collect() };
    let before = listing(&v);
    let err = NativeEngine::open_or_create(v.clone(), root(), opts()).err().expect("must fail");
    assert_eq!(err.exit_code(), Some(78), "{err}");
    assert_eq!(listing(&v), before, "nothing may be created in a failure state");

    // Failure state 2: CURRENT names a generation that is missing.
    let c = v.open(&root().join("CURRENT"), OpenOpts::create()).unwrap();
    c.write_at(b"gen-000009\n", 0).unwrap();
    let before = listing(&v);
    let err = NativeEngine::open_or_create(v.clone(), root(), opts()).err().expect("must fail");
    assert_eq!(err.exit_code(), Some(78), "{err}");
    assert_eq!(listing(&v), before);
}

/// A9: a simulated tick runs the writer's own body.
#[test]
fn a_writer_tick_flushes_like_the_writer_thread() {
    let (_sim, v) = sim_vfs(24);
    let o = EngineOptions { synchronous_full: false, ..opts() };
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    let end = e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
    assert!(e.wal.durable_lsn() < end, "NORMAL: the commit has not been synced yet");
    e.wal.writer_tick();
    assert!(e.wal.durable_lsn() >= end);
    assert_eq!(e.durable_applied_index(), 1);
}
