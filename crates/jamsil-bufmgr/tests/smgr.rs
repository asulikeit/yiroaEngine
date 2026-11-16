//! Relation file manager: file geometry (answer A12) and descriptor safety (answer B7).

use jamsil_bufmgr::*;
use jamsil_page::*;
use jamsil_vfs::sync::atomic::Ordering;
use jamsil_vfs::{FaultPlan, FaultVfs, SimVfs, Vfs};
use std::path::Path;
use std::sync::Arc;

fn page() -> Vec<u8> {
    vec![1u8; PAGE_SIZE]
}

/// Writes a page the way extension does: the known length rises first (spec 6.7).
fn put(s: &Smgr, tag: PageTag) {
    s.ensure_len(tag.rel, tag.fork, tag.block + 1).unwrap();
    s.write(tag, &page()).unwrap();
}

#[test]
fn commit_logs_live_in_xact_with_256_page_segments() {
    let sim = SimVfs::new(1);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let s = Smgr::new(v, Path::new("/g")).unwrap();
    for b in [0, 255, 256, 600] {
        put(&s, PageTag::new(CLOG_RELFILE, ForkId::Main, b));
    }
    put(&s, PageTag::new(CSNLOG_RELFILE, ForkId::Main, 3));
    put(&s, PageTag::new(100, ForkId::Main, 2));

    assert_eq!(sim.file_bytes(Path::new("/g/xact/clog.000000")).unwrap().len(), 256 * PAGE_SIZE);
    assert!(sim.file_bytes(Path::new("/g/xact/clog.000001")).is_some(), "block 256 is the first page of segment 1");
    assert!(sim.file_bytes(Path::new("/g/xact/clog.000002")).is_some(), "block 600 lies in segment 2");
    assert!(sim.file_bytes(Path::new("/g/xact/csnlog.000000")).is_some());
    assert!(sim.file_bytes(Path::new("/g/base/0000000000000064.main.0")).is_some());
    assert_eq!(s.geometry(100).pages_per_segment, 131_072);
    assert_eq!(s.geometry(CLOG_RELFILE).pages_per_segment, 256);

    // Old commit-log segments can be unlinked; the page table is untouched by that.
    s.sync_all().unwrap();
    assert_eq!(s.log_segments(CLOG_RELFILE).unwrap(), [0, 1, 2]);
    assert_eq!(s.unlink_log_segments_below(CLOG_RELFILE, 1).unwrap(), 1);
    s.sync_all().unwrap();
    assert_eq!(s.log_segments(CLOG_RELFILE).unwrap(), [1, 2]);
    let mut buf = vec![0u8; PAGE_SIZE];
    assert!(!s.read(PageTag::new(CLOG_RELFILE, ForkId::Main, 5), &mut buf).unwrap(), "segment 0 is gone");
    assert!(s.read(PageTag::new(CLOG_RELFILE, ForkId::Main, 256), &mut buf).unwrap());
    assert!(s.list_relfiles().unwrap().contains(&100) && !s.list_relfiles().unwrap().contains(&CLOG_RELFILE));
}

#[test]
fn a_descriptor_with_unsynced_writes_is_never_closed() {
    let sim = SimVfs::new(2);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let s = Smgr::with_max_open(v, Path::new("/g"), 4).unwrap();
    for rel in 100..110 {
        put(&s, PageTag::new(rel, ForkId::Main, 0));
    }
    assert_eq!(s.open_descriptors(), 10, "the cache grows past its bound while nothing is clean");
    assert_eq!(s.unsynced_files(), 10);

    s.sync_all().unwrap();
    assert_eq!(s.unsynced_files(), 0);
    // Now clean descriptors can be evicted, and only they.
    put(&s, PageTag::new(200, ForkId::Main, 0));
    assert_eq!(s.open_descriptors(), 4);
    assert_eq!(s.unsynced_files(), 1, "the dirty descriptor survived the eviction");

    // A write after the sync makes a descriptor dirty again, and it is kept.
    put(&s, PageTag::new(100, ForkId::Main, 1));
    for rel in 300..320 {
        let mut buf = vec![0u8; PAGE_SIZE];
        let _ = s.read(PageTag::new(rel, ForkId::Main, 0), &mut buf);
    }
    assert!(s.unsynced_files() >= 2);
}

#[test]
fn a_failed_sync_leaves_the_descriptor_open_and_dirty() {
    let sim = SimVfs::new(3);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim, plan.clone(), None));
    let s = Smgr::with_max_open(v, Path::new("/g"), 2).unwrap();
    put(&s, PageTag::new(100, ForkId::Main, 0));
    plan.fail_all_syncs.store(true, Ordering::SeqCst);
    assert!(s.sync_all().is_err());
    plan.fail_all_syncs.store(false, Ordering::SeqCst);
    // The caller treats the failure as fatal; whatever else happens, the file is not closed.
    for rel in 101..110 {
        put(&s, PageTag::new(rel, ForkId::Main, 0));
    }
    s.sync_all().unwrap();
    assert_eq!(s.unsynced_files(), 1, "only the descriptor whose sync failed is still dirty");
}

/// After a restart the files can be longer than anything this process has written. The fork's
/// length must come from the files, never from the highest page written since the restart: a short
/// answer would make the next extension overwrite the tail of the fork (spec 6.7: the length starts
/// from the size of the files, whatever call touches the fork first).
///
/// Seed: none. The defect was found by `hints_racing_checkpoints_leave_no_unrepairable_page`
/// (crates/jamsil-engine/tests/answers.rs), which runs real threads and cannot be replayed from a
/// seed; the failing run was not traced further. This test is deterministic, and the M1 crash
/// workload now reaches the same state by design: it extends after restarts and checks `nblocks`
/// against what was acknowledged (`run_seed` in crates/jamsil-engine/tests/m1.rs).
#[test]
fn a_restarted_smgr_knows_the_length_from_the_files() {
    let sim = SimVfs::new(4);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let rel = 100;
    let first = Smgr::new(v.clone(), Path::new("/g")).unwrap();
    for b in 0..8 {
        put(&first, PageTag::new(rel, ForkId::Main, b));
    }
    first.sync_all().unwrap();
    drop(first);

    // A new process writes only page 1, as redo restoring one image would.
    let again = Smgr::new(v.clone(), Path::new("/g")).unwrap();
    // The first call of the new process is a bare write, as when redo restores one image.
    again.write(PageTag::new(rel, ForkId::Main, 1), &page()).unwrap();
    assert_eq!(again.nblocks(rel, ForkId::Main).unwrap(), 8);

    // And replaying a create record for a fork that already has pages must not reset its length.
    let third = Smgr::new(v, Path::new("/g")).unwrap();
    third.create_fork(rel, ForkId::Main).unwrap();
    assert_eq!(third.nblocks(rel, ForkId::Main).unwrap(), 8);
}

/// Spec 6.7: the length is the number of whole pages. A trailing partial page is a torn write or a
/// torn extension and is not counted; writing it whole counts it.
#[test]
fn a_trailing_partial_page_is_not_counted_in_the_length() {
    let sim = SimVfs::new(5);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let rel = 100;
    let first = Smgr::new(v.clone(), Path::new("/g")).unwrap();
    for b in 0..3 {
        put(&first, PageTag::new(rel, ForkId::Main, b));
    }
    first.sync_all().unwrap();
    drop(first);
    let path = Path::new("/g/base/0000000000000064.main.0");
    assert_eq!(sim.file_bytes(path).unwrap().len(), 3 * PAGE_SIZE);
    sim.poke(path, 3 * PAGE_SIZE as u64, &[7u8; 100]); // the first 100 bytes of a fourth page

    let again = Smgr::new(v, Path::new("/g")).unwrap();
    assert_eq!(again.nblocks(rel, ForkId::Main).unwrap(), 3, "a partial page is not a page");
    let mut buf = vec![0u8; PAGE_SIZE];
    assert!(again.read(PageTag::new(rel, ForkId::Main, 3), &mut buf).unwrap(), "its bytes can still be read");
    assert!(buf[..100].iter().all(|b| *b == 7) && buf[100..].iter().all(|b| *b == 0), "zero-filled past the end");

    put(&again, PageTag::new(rel, ForkId::Main, 3));
    assert_eq!(again.nblocks(rel, ForkId::Main).unwrap(), 4, "written whole, it is page 3");
}

/// The debug assertion of spec 6.7 rule 3: a write at or beyond the known length, other than by
/// extension, is a caller's bug.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "only extension may raise it")]
fn writing_beyond_the_known_length_is_a_debug_assertion() {
    let sim = SimVfs::new(6);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let s = Smgr::new(v, Path::new("/g")).unwrap();
    put(&s, PageTag::new(100, ForkId::Main, 0));
    let _ = s.write(PageTag::new(100, ForkId::Main, 5), &page());
}

/// A release build does not refuse the write, since that would block a legitimate page, but it
/// raises the length and counts the violation, so that it cannot go unnoticed. (Run with
/// `cargo test --release`; a debug build stops at the assertion instead.)
#[test]
#[cfg(not(debug_assertions))]
fn a_release_build_raises_the_length_and_counts_the_violation() {
    let sim = SimVfs::new(7);
    let v: Arc<dyn Vfs> = Arc::new(sim);
    let s = Smgr::new(v, Path::new("/g")).unwrap();
    put(&s, PageTag::new(100, ForkId::Main, 0));
    assert_eq!(s.length_raised_by_write(), 0, "an extension is not a violation");
    s.write(PageTag::new(100, ForkId::Main, 5), &page()).unwrap();
    assert_eq!(s.length_raised_by_write(), 1);
    assert_eq!(s.nblocks(100, ForkId::Main).unwrap(), 6, "the length was raised, not lost");
}

/// Spec 9.3 "What redo may rely on": a file operation of an `SMGR` record is carried out again and
/// queued for sync, also when the fork exists already. An ordinary create of an existing fork queues
/// nothing, because the process that made it knows whether it synced it; a recovering process does
/// not know what the dead one did.
#[test]
fn a_fork_created_again_by_redo_is_queued_for_sync_even_when_it_exists() {
    let sim = SimVfs::new(8);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim, plan.clone(), None));
    let first = Smgr::new(v.clone(), Path::new("/g")).unwrap();
    first.create_fork(100, ForkId::Main).unwrap();
    first.sync_all().unwrap();
    drop(first);

    let again = Smgr::new(v, Path::new("/g")).unwrap();
    again.create_fork(100, ForkId::Main).unwrap();
    plan.fail_all_syncs.store(true, Ordering::SeqCst);
    assert!(again.sync_all().is_ok(), "an ordinary create of an existing fork queues nothing to sync");
    again.recreate_fork(100, ForkId::Main).unwrap();
    assert!(again.sync_all().is_err(), "a recreate during redo queues the file and its directory: a sync was attempted");
}
