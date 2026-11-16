//! Named acceptance scenarios of spec 19.7 that apply to the durable-page layer.

use jamsil_engine::*;
use jamsil_page::*;
use jamsil_vfs::{SimVfs, Vfs};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Extension formats kind-0 pages; the first record that uses a page gives it its purpose.
fn extend_raw(e: &NativeEngine, rel: RelFileId, n: u32) -> Result<(), EngineError> {
    let first = e.extend_relation(rel, ForkId::Main, n)?;
    e.init_raw_pages(e.alloc_xid().unwrap(), rel, ForkId::Main, first, n)
}

const REL: RelFileId = 100;
const SEG: u64 = 256 * 1024;

fn opts() -> EngineOptions {
    EngineOptions { buffer_frames: 64, wal_segment_size: Some(SEG), wal_consistency_check: true, ..Default::default() }
}
fn root() -> &'static Path {
    Path::new("/db/rdb")
}
fn wal_file(lsn: Lsn) -> PathBuf {
    root().join("gen-000001/wal").join(format!("{:016x}.wal", lsn / SEG))
}
fn setup() -> (SimVfs, Arc<dyn Vfs>, Arc<NativeEngine>) {
    let sim = SimVfs::new(7);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 4).unwrap();
    (sim, v, e)
}
fn commit_n(e: &NativeEngine, idx: u64) -> Csn {
    let x = e.alloc_xid().unwrap();
    e.write_bytes(x, PageTag::new(REL, ForkId::Main, 0), 64, &idx.to_le_bytes()).unwrap();
    e.commit(x, Some(AppliedPos { index: idx, term: 1 }), &[]).unwrap()
}

#[test]
fn damaged_log_proven_durable_stops_recovery_and_changes_nothing() {
    let (sim, v, e) = setup();
    let mut ends = Vec::new();
    for i in 1..=5 {
        ends.push(commit_n(&e, i));
    }
    // Commit 2's record is followed by durability markers from the later flushes.
    let victim = ends[1] - 12;
    sim.flip_bit(&wal_file(victim), victim % SEG, 3);
    let before = sim.file_bytes(&wal_file(victim)).unwrap();
    drop(e);
    sim.crash();
    match NativeEngine::open_or_create(v, root(), opts()) {
        Err(EngineError::Corruption(m)) => assert!(m.contains("durability marker") || m.contains("valid record"), "{m}"),
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("corruption was not detected"),
    }
    let after = sim.file_bytes(&wal_file(victim)).unwrap();
    assert_eq!(before, after, "recovery must not modify the log when it reports corruption");
}

#[test]
fn damaged_log_in_last_sync_batch_is_a_torn_tail() {
    let (sim, v, e) = setup();
    commit_n(&e, 1);
    let last = commit_n(&e, 2);
    // The marker after commit 2 was never synced; flipping a byte of commit 2 cannot be told
    // from a torn write (the stated limit of detection).
    let victim = last - 12;
    sim.flip_bit(&wal_file(victim), victim % SEG, 1);
    drop(e);
    sim.crash();
    let e = NativeEngine::open_or_create(v, root(), opts()).expect("must be taken for a torn tail");
    assert_eq!(e.recovery.log_end, LogEnd::TornTail);
    assert_eq!(e.applied().index, 1, "valid prefix of the commit order");
}

#[test]
fn zeroed_page_is_reported_never_served() {
    let (sim, v, e) = setup();
    commit_n(&e, 1);
    e.shutdown().unwrap();
    drop(e);
    let path = root().join(format!("gen-000001/base/{REL:016x}.main.0"));
    sim.poke(&path, 2 * PAGE_SIZE as u64, &vec![0u8; PAGE_SIZE]);
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    let err = e.pin(PageTag::new(REL, ForkId::Main, 2)).err().expect("zero page must not be served");
    assert!(err.to_string().contains("corrupt"), "{err}");
    // Other pages stay readable.
    assert!(e.pin(PageTag::new(REL, ForkId::Main, 1)).is_ok());
}

#[test]
fn bit_flip_in_data_page_is_detected() {
    let (sim, v, e) = setup();
    commit_n(&e, 1);
    e.shutdown().unwrap();
    drop(e);
    let path = root().join(format!("gen-000001/base/{REL:016x}.main.0"));
    sim.flip_bit(&path, PAGE_SIZE as u64 + 500, 6);
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    assert!(e.pin(PageTag::new(REL, ForkId::Main, 1)).is_err());
}

#[test]
fn page_written_to_wrong_offset_is_detected() {
    let (sim, v, e) = setup();
    commit_n(&e, 1);
    e.shutdown().unwrap();
    drop(e);
    let path = root().join(format!("gen-000001/base/{REL:016x}.main.0"));
    let bytes = sim.file_bytes(&path).unwrap();
    // Copy page 1 over page 3: valid page, wrong location. The checksum seed covers the tag.
    let p1 = bytes[PAGE_SIZE..2 * PAGE_SIZE].to_vec();
    sim.poke(&path, 3 * PAGE_SIZE as u64, &p1);
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    assert!(e.pin(PageTag::new(REL, ForkId::Main, 3)).is_err());
}

#[test]
fn reply_lost_commit_is_present_and_whole() {
    let (sim, v, e) = setup();
    commit_n(&e, 1);
    commit_n(&e, 2); // flushed; the reply would have been sent next
    drop(e);
    sim.crash();
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    assert_eq!(e.applied().index, 2);
    let p = e.pin(PageTag::new(REL, ForkId::Main, 0)).unwrap();
    assert_eq!(rd_u64(&p.read()[..], 64), 2);
}

#[test]
fn normal_durability_loses_only_a_suffix() {
    let sim = SimVfs::new(11);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let mut o = opts();
    o.synchronous_full = false;
    let e = NativeEngine::open_or_create(v.clone(), root(), o.clone()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 2).unwrap();
    for i in 1..=10u64 {
        let x = e.alloc_xid().unwrap();
        e.write_bytes(x, PageTag::new(REL, ForkId::Main, 0), 64, &i.to_le_bytes()).unwrap();
        e.commit(x, Some(AppliedPos { index: i, term: 1 }), &[]).unwrap();
        if i == 6 {
            e.wal.flush_all().unwrap();
        }
    }
    assert!(e.durable_applied_index() >= 6 && e.durable_applied_index() <= 10);
    let durable = e.durable_applied_index();
    drop(e);
    sim.crash();
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    let got = e.applied().index;
    assert!(got >= durable, "applied_index {got} fell below durable_applied_index {durable}");
}

#[test]
fn fatal_sync_failure_ends_the_process_policy() {
    use jamsil_vfs::{FaultPlan, FaultVfs};
    use std::sync::atomic::Ordering;
    let sim = SimVfs::new(5);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    let e = NativeEngine::open_or_create(v.clone(), root(), opts()).unwrap();
    e.set_fatal_handler(Arc::new(|m: &str| panic!("exit 75: {m}")));
    e.create_relation(REL, ForkId::Main).unwrap();
    plan.fail_all_syncs.store(true, Ordering::SeqCst);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| commit_n_safe(&e)));
    assert!(r.is_err(), "a failed fdatasync must be fatal and never retried");
}

fn commit_n_safe(e: &NativeEngine) {
    let x = e.alloc_xid().unwrap();
    let _ = e.commit(x, Some(AppliedPos { index: 1, term: 1 }), &[]);
}

#[test]
fn xid_with_a_durable_trace_is_never_reused() {
    let (sim, v, e) = setup();
    let a = e.alloc_xid().unwrap();
    e.write_bytes(a, PageTag::new(REL, ForkId::Main, 1), 64, &1u64.to_le_bytes()).unwrap();
    // Another xid that is handed out but leaves no durable trace may be reused: nothing on disk
    // can name it. The flush below makes `a`'s record durable.
    let _b = e.alloc_xid().unwrap();
    e.wal.flush_all().unwrap();
    drop(e);
    sim.crash();
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    assert!(e.alloc_xid().unwrap() > a);
}

#[test]
fn disk_full_on_extension_fails_cleanly() {
    let sim = SimVfs::new(9);
    let v: Arc<dyn Vfs> = Arc::new(sim.clone());
    let e = NativeEngine::open_or_create(v, root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    // Allow almost nothing beyond what exists.
    let used: u64 = (1..).map(|_| 0u64).take(0).sum::<u64>() + 1;
    let _ = used;
    sim.set_capacity(Some(3 * SEG));
    let r = extend_raw(&e, REL, 2);
    // Either fits or fails with a disk-full style error; it must never leave a corrupt state.
    if let Err(err) = r {
        assert!(!err.to_string().is_empty());
    }
    sim.set_capacity(None);
    extend_raw(&e, REL, 2).unwrap();
    e.verify_fork(REL, ForkId::Main).unwrap();
}
