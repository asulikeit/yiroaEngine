//! Tests for the decisions recorded in history/Answer-20261115-1800.md: a full disk on a write into
//! an existing segment quarantines (A6), the `SWITCH` gap is never written (B28), `check_size` is
//! the gate before every critical section (A4), and the two rules of what redo may rely on (A2).

use jamsil_engine::*;
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::env::{ManualClock, SeededRng};
use jamsil_vfs::{FaultPlan, FaultVfs, SimVfs, Vfs};
use jamsil_wal::WalError;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const REL: RelFileId = 100;
const SEG: u64 = 256 * 1024;

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

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

// ---- A6: a full disk on a write into an existing segment --------------------------------------

/// Spec 5.5. On the supported file systems a write into a preallocated segment does not need new
/// space, so `ENOSPC` there means the environment is broken. The log quarantines at once instead of
/// retrying at the next flush and failing every commit in silence. The records of the failed batch
/// stay "outcome unknown", because the failed write may have handed bytes to the file.
#[test]
fn a_full_disk_on_a_write_into_an_existing_segment_quarantines() {
    let sim = SimVfs::new(81);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    let e = NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 1).unwrap();
    let x = e.alloc_xid().unwrap();
    put(&e, x, 0, 0, 5);
    e.commit(x, Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
    assert!(!e.wal.is_quarantined());

    plan.write_enospc.store(true, Ordering::SeqCst);
    let x = e.alloc_xid().unwrap();
    put(&e, x, 0, 0, 6);
    let err = e.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap_err();
    assert!(matches!(err, EngineError::OutcomeUnknown(_)), "the commit in flight is unknown: {err}");
    assert_eq!(err.sqlstate(), Some("08007"));
    assert!(e.wal.is_quarantined(), "and the log is quarantined, not left to retry");
    assert!(e.space_low());

    // Nothing is retried: even with the disk healthy again, later commits fail definitely.
    plan.write_enospc.store(false, Ordering::SeqCst);
    let x = e.alloc_xid().unwrap();
    let later = e.commit(x, Some(AppliedPos { index: 3, term: 1 }), &[]);
    assert!(matches!(later, Err(EngineError::Wal(WalError::Quarantined(_)))), "{later:?}");

    // A restart clears the quarantine and settles the unknown commit by the log.
    drop(e);
    sim.crash();
    let e = NativeEngine::open(v, root(), opts(64)).unwrap();
    assert!(!e.wal.is_quarantined());
    let applied = e.applied().index;
    assert!(applied == 1 || applied == 2, "applied {applied}");
    assert_eq!(get(&e, 0, 0), if applied == 2 { 6 } else { 5 });
    e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: applied + 1, term: 1 }), &[]).unwrap();
}

// ---- A4: check_size is the gate before every critical section ----------------------------------

/// Spec 8.8. `check_size` fails when the log no longer accepts records, so that no critical section
/// starts only to add records that never reach the file. That bounds the overshoot of the WAL buffer
/// after a failed flush (spec 8.4) to the one critical section each thread is already in.
#[test]
fn a_quarantined_log_stops_every_critical_section_before_it_starts() {
    let sim = SimVfs::new(82);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let e = NativeEngine::open_or_create(v, root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 2).unwrap();
    e.commit(e.alloc_xid().unwrap(), None, &[]).unwrap();
    let x = e.alloc_xid().unwrap();
    let tag = PageTag::new(REL, ForkId::Main, 0);

    e.wal.quarantine(jamsil_wal::QuarantineKind::Restart, "test");
    let (lsn, dirty) = (e.wal.insert_lsn(), e.pool.dirty_count());
    let quarantined = |r: Result<(), EngineError>, what: &str| {
        assert!(matches!(r, Err(EngineError::Wal(WalError::Quarantined(_)))), "{what}: {r:?}");
    };
    quarantined(e.write_bytes(x, tag, cell_off(0), &[1]).map(|_| ()), "write_bytes");
    quarantined(e.init_and_write_bytes(x, tag, cell_off(0), &[1]).map(|_| ()), "init_and_write_bytes");
    quarantined(e.create_relation(REL + 1, ForkId::Main), "create_relation");
    quarantined(e.extend_relation(REL, ForkId::Main, 1).map(|_| ()), "extend_relation");
    quarantined(e.applied_advance(AppliedPos { index: 9, term: 1 }).map(|_| ()), "applied_advance");
    quarantined(e.commit(x, None, &[]).map(|_| ()), "commit");
    quarantined(e.checkpoint(), "checkpoint");
    // A hint is optional (spec 7.6): on a log that does not accept records it is skipped, not an error.
    assert!(!e.set_hint_byte(tag, 100, 1).unwrap(), "the hint is skipped");
    assert_eq!(e.wal.insert_lsn(), lsn, "nothing may have been inserted");
    assert_eq!(e.pool.dirty_count(), dirty, "no page may have been dirtied");
    assert_eq!(e.smgr.nblocks(REL, ForkId::Main).unwrap(), 2, "and no fork extended");
}

/// The images are taken inside the critical section, so the size check before it counts a full
/// image for each block the record will reference, whether or not one is attached yet.
#[test]
fn check_size_counts_a_full_image_for_each_block() {
    let sim = SimVfs::new(83);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let e = NativeEngine::open_or_create(v, root(), opts(64)).unwrap(); // 256 KiB segments: a 64 KiB limit
    let main = vec![0u8; 40_000];
    let tags: Vec<PageTag> = (0..4).map(|b| PageTag::new(REL, ForkId::Main, b)).collect();
    let with = |n: usize| tags[..n].iter().fold(RecordBuilder::new(rmgr::HEAP, 1).main(&main), |r, t| r.block(BlockRef::new(*t)));

    let four = with(4);
    assert!(four.encoded_len() < 64 * 1024, "without images the record fits ({})", four.encoded_len());
    assert!(four.worst_case_len() > 64 * 1024, "with an image for each block it cannot ({})", four.worst_case_len());
    assert!(matches!(e.wal.check_size(&four), Err(WalError::RecordTooLarge { .. })));
    assert!(e.wal.check_size(&with(2)).is_ok(), "two images still fit");
}

// ---- A2: what redo may rely on (spec 9.3, 7.8) -------------------------------------------------

/// The hole in the argument that file contents need no sync, in the four steps of the answer.
///
/// 1. The dead process extends a fork and writes some new kind-0 pages to the page cache; no sync
///    covers them.
/// 2. Redo of the extend record finds them present. Under the old rule it left them alone: they
///    were not dirty in the new process, and their file was not in its sync queue.
/// 3. The end-of-recovery checkpoint moves the redo point past the extend record.
/// 4. A power loss drops the unsynced writes, and the next read of such a page reports corruption
///    or finds the fork shorter than it was.
///
/// Rule 2 (7.8): redo formats every page of the range again, without reading it, and leaves it
/// dirty, so the recovery checkpoint writes and syncs it.
#[test]
fn extension_pages_that_a_killed_process_wrote_survive_a_power_loss_after_recovery() {
    let file = root().join("gen-000001/base/0000000000000064.main.0");
    for seed in 100..108u64 {
        let sim = SimVfs::new(seed);
        let v: Arc<dyn Vfs> = Arc::new(sim.clone());
        // 1. A small pool, so that the churn on a second relation pushes every new page of the
        //    first out to its file, where it sits in the page cache.
        let e = NativeEngine::open_or_create(v.clone(), root(), opts(8)).unwrap();
        e.create_relation(REL, ForkId::Main).unwrap();
        extend_raw(&e, REL, 4).unwrap();
        e.checkpoint().unwrap();
        e.extend_relation(REL, ForkId::Main, 6).unwrap();
        let other = REL + 1;
        e.create_relation(other, ForkId::Main).unwrap();
        e.extend_relation(other, ForkId::Main, 12).unwrap();
        e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 1, term: 1 }), &[]).unwrap();
        assert!(
            sim.file_bytes(&file).unwrap().len() >= 10 * PAGE_SIZE,
            "seed {seed}: the new pages must have been written to the file"
        );
        drop(e); // the process is killed; the page cache stays

        // 2. and 3. Recovery, with its end-of-recovery checkpoint.
        drop(NativeEngine::open(v.clone(), root(), opts(8)).unwrap());
        // 4. The power goes.
        sim.crash();

        let ctx = format!("seed {seed}");
        let e = NativeEngine::open(v, root(), opts(8)).unwrap_or_else(|er| panic!("{ctx}: {er}"));
        assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap_or_else(|er| panic!("{ctx}: {er}")), 10, "{ctx}");
        assert_eq!(e.verify_fork(other, ForkId::Main).unwrap_or_else(|er| panic!("{ctx}: {er}")), 12, "{ctx}");
    }
}

/// Rule 1 (9.3): every page that a replayed record names is left dirty, also when the page is
/// already up to date and the record is skipped. The record here has no full-page image (a record
/// type that does not need one), so redo reads the page from the file, finds it up to date, and
/// skips the record; the page was written by a process that was killed before any sync.
#[test]
fn a_page_that_redo_skips_as_up_to_date_is_still_written_and_synced() {
    let page1 = root().join("gen-000001/base/0000000000000064.main.0");
    let cell_at = PAGE_SIZE + cell_off(3) as usize;
    for seed in 120..128u64 {
        let sim = SimVfs::new(seed);
        let v: Arc<dyn Vfs> = Arc::new(sim.clone());
        let e = NativeEngine::open_or_create(v.clone(), root(), opts(8)).unwrap();
        e.create_relation(REL, ForkId::Main).unwrap();
        extend_raw(&e, REL, 4).unwrap();
        e.checkpoint().unwrap(); // the fork is durable from here

        // A change to page 1 with no image: logged by hand, as a record type that needs none.
        let x = e.alloc_xid().unwrap();
        let tag = PageTag::new(REL, ForkId::Main, 1);
        {
            let p = e.pin(tag).unwrap();
            let mut w = p.write();
            w[cell_at - PAGE_SIZE..cell_at - PAGE_SIZE + 8].copy_from_slice(&4242u64.to_le_bytes());
            w.mark_dirty();
            let mut payload = (cell_off(3)).to_le_bytes().to_vec();
            payload.extend_from_slice(&4242u64.to_le_bytes());
            let rec = RecordBuilder::new(rmgr::TEST, testop::SET_BYTES).xid(x).block(BlockRef::new(tag).data(&payload));
            let permit = e.wal.check_size(&rec).unwrap();
            let range = e.wal.insert(permit, rec);
            w.set_lsn(range.end);
        }
        e.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
        // Churn on another relation pushes page 1 out to its file, unsynced. The clock sweep gives a
        // page that was used recently a few rounds before it is evicted, so the churn is long.
        let other = REL + 1;
        e.create_relation(other, ForkId::Main).unwrap();
        e.extend_relation(other, ForkId::Main, 60).unwrap();
        e.commit(e.alloc_xid().unwrap(), Some(AppliedPos { index: 3, term: 1 }), &[]).unwrap();
        let in_file = sim.file_bytes(&page1).unwrap();
        assert_eq!(&in_file[cell_at..cell_at + 8], &4242u64.to_le_bytes(), "seed {seed}: the page must have been written out");
        drop(e); // killed

        // Recovery skips the record (the page in the file is up to date), then the power goes.
        let r = NativeEngine::open(v.clone(), root(), opts(8)).unwrap();
        assert!(r.recovery.records_replayed > 0);
        drop(r);
        sim.crash();

        let ctx = format!("seed {seed}");
        let e = NativeEngine::open(v, root(), opts(8)).unwrap_or_else(|er| panic!("{ctx}: {er}"));
        assert_eq!(get(&e, 1, 3), 4242, "{ctx}: an acknowledged change was lost");
    }
}
