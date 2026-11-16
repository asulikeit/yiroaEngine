//! M1 gate: a simulated page-update workload with crashes and torn writes loses no acknowledged
//! write and serves no corrupt page (spec section 21).

use jamsil_engine::*;
use jamsil_page::*;
use jamsil_vfs::env::{ManualClock, SeededRng};
use jamsil_vfs::{FaultPlan, FaultVfs, SimVfs, Vfs};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Extension formats kind-0 pages; the first record that uses a page gives it its purpose.
fn extend_raw(e: &NativeEngine, rel: RelFileId, n: u32) -> Result<(), EngineError> {
    let first = e.extend_relation(rel, ForkId::Main, n)?;
    e.init_raw_pages(e.alloc_xid().unwrap(), rel, ForkId::Main, first, n)
}

const REL: RelFileId = 100;
const PAGES: u32 = 24;
const CELLS: usize = 8;

fn opts(frames: usize) -> EngineOptions {
    // Fixed clock and seeded randomness: a run is a pure function of its seed.
    EngineOptions {
        buffer_frames: frames,
        wal_segment_size: Some(256 * 1024),
        wal_consistency_check: true,
        // A small WAL buffer, so that inserters wait for flushes all through the crash workload
        // (B3, spec 8.4): with full-page images on every record it holds only a few records.
        wal_buffer_bytes: 32 * 1024,
        background_threads: false,
        clock: ManualClock::new(1_000_000),
        rng: SeededRng::new(0x5EED),
        // A death inside recovery must not end the test binary (it would exit with code 75).
        fatal_handler: Some(Arc::new(|m: &str| panic!("FATAL {m}"))),
        ..Default::default()
    }
}

fn panic_on_fatal(e: &NativeEngine) {
    e.set_fatal_handler(Arc::new(|m: &str| panic!("FATAL {m}")));
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn cell_off(c: usize) -> u16 {
    (64 + 8 * c) as u16
}

fn read_cell(e: &NativeEngine, page: u32, c: usize) -> u64 {
    let p = e.pin(PageTag::new(REL, ForkId::Main, page)).expect("page must verify");
    let g = p.read();
    rd_u64(&g[..], cell_off(c) as usize)
}

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

#[test]
fn write_commit_restart_roundtrip() {
    let vfs = SimVfs::new(1);
    let v: Arc<dyn Vfs> = Arc::new(vfs.clone());
    let e = NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 4).unwrap();
    let x = e.alloc_xid().unwrap();
    e.write_bytes(x, PageTag::new(REL, ForkId::Main, 2), cell_off(0), &77u64.to_le_bytes()).unwrap();
    e.commit(x, Some(AppliedPos { index: 5, term: 2 }), &[]).unwrap();
    drop(e);
    vfs.crash();
    let e = NativeEngine::open_or_create(v, root(), opts(64)).unwrap();
    assert_eq!(e.applied(), AppliedPos { index: 5, term: 2 });
    assert_eq!(read_cell(&e, 2, 0), 77);
    assert_eq!(e.verify_fork(REL, ForkId::Main).unwrap(), 4);
    assert!(e.alloc_xid().unwrap() > x, "xid must never be reused");
}

#[test]
fn clean_shutdown_and_reopen() {
    let vfs = SimVfs::new(2);
    let v: Arc<dyn Vfs> = Arc::new(vfs.clone());
    let e = NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    extend_raw(&e, REL, 2).unwrap();
    e.write_bytes(e.alloc_xid().unwrap(), PageTag::new(REL, ForkId::Main, 1), cell_off(3), &9u64.to_le_bytes()).unwrap();
    e.applied_advance(AppliedPos { index: 1, term: 1 }).unwrap();
    e.shutdown().unwrap();
    drop(e);
    let e = NativeEngine::open_or_create(v, root(), opts(64)).unwrap();
    assert_eq!(read_cell(&e, 1, 3), 9);
    assert_eq!(e.applied().index, 1);
}

#[test]
fn wrong_cluster_id_is_a_format_error() {
    let vfs = SimVfs::new(3);
    let v: Arc<dyn Vfs> = Arc::new(vfs.clone());
    drop(NativeEngine::open_or_create(v.clone(), root(), opts(64)).unwrap());
    let mut o = opts(64);
    o.cluster_id = b"other".to_vec();
    assert!(matches!(NativeEngine::open_or_create(v, root(), o), Err(EngineError::FormatMismatch(_))));
}

/// One seeded run: random updates, extensions, commits, checkpoints, evictions, and three kinds of
/// ending for every round: power loss, a process kill that leaves the page cache intact, and a
/// power loss followed by a death in the middle of recovery.
///
/// The model keeps, per page and cell, the last acknowledged value and the last attempted one, and
/// for the fork the length an acknowledged extension guarantees and the length attempted. Every
/// restart checks the length of the fork as well as the cells (spec 6.7): the defect of the first
/// round of answers, a fork length that restarted at zero, was invisible to a workload that never
/// asked for the length and never extended after a restart.
fn run_seed(seed: u64, inject_sync_failures: bool) -> SimVfs {
    let sim = SimVfs::new(seed);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim.clone(), plan.clone(), None));
    let mut rng = Rng(seed ^ 0xABCD);
    let mut acked = vec![[0u64; CELLS]; PAGES as usize];
    let mut attempted = vec![[0u64; CELLS]; PAGES as usize];
    let (mut pages_acked, mut pages_attempted) = (0u32, 0u32);
    let mut counter = 0u64;
    let mut applied_idx = 0u64;
    let mut acked_idx = 0u64;
    let mut created = false;

    for round in 0..8 {
        let e = NativeEngine::open_or_create(v.clone(), root(), opts(16)).unwrap_or_else(|er| panic!("seed {seed} round {round}: open failed: {er}"));
        panic_on_fatal(&e);
        if created {
            // The fork keeps every page an acknowledged extension made, and no page that was
            // never attempted. Asked before anything else is done with the engine.
            let n = e.smgr.nblocks(REL, ForkId::Main).unwrap();
            assert!(
                n >= pages_acked && n <= pages_attempted,
                "seed {seed} round {round}: fork length {n}, acknowledged {pages_acked}, attempted {pages_attempted}"
            );
            pages_acked = n;
            pages_attempted = n;
            acked.truncate(n as usize);
            attempted.truncate(n as usize);
            // Oracle: nothing acknowledged is lost, nothing impossible appears, no page is corrupt.
            for p in 0..n {
                for c in 0..CELLS {
                    let got = read_cell(&e, p, c);
                    assert!(
                        got >= acked[p as usize][c] && got <= attempted[p as usize][c],
                        "seed {seed} round {round}: page {p} cell {c}: got {got}, acked {}, attempted {}",
                        acked[p as usize][c],
                        attempted[p as usize][c]
                    );
                    // Whatever survived is now the floor.
                    acked[p as usize][c] = got;
                    attempted[p as usize][c] = got;
                }
            }
            assert!(e.applied().index >= acked_idx, "seed {seed}: applied_index went backwards");
            applied_idx = e.applied().index;
            acked_idx = applied_idx;
        } else {
            e.create_relation(REL, ForkId::Main).unwrap();
            extend_raw(&e, REL, PAGES).unwrap_or(());
            e.commit(e.alloc_xid().unwrap(), None, &[]).unwrap();
            (pages_acked, pages_attempted) = (PAGES, PAGES);
            created = true;
        }
        if inject_sync_failures && round % 2 == 1 {
            plan.fail_sync_after.store(rng.below(12) as i64, Ordering::SeqCst);
        }
        let n_ops = 20 + rng.below(80);
        let result = catch_unwind(AssertUnwindSafe(|| {
            for _ in 0..n_ops {
                let x = e.alloc_xid().unwrap();
                if rng.below(12) == 0 {
                    // Extension: the new pages start where the fork ends, whatever happened before
                    // the last restart. A fork length that was too short would start them inside
                    // the fork and overwrite acknowledged pages.
                    let k = 1 + rng.below(3) as u32;
                    let want = pages_attempted;
                    pages_attempted += k;
                    acked.extend((0..k).map(|_| [0u64; CELLS]));
                    attempted.extend((0..k).map(|_| [0u64; CELLS]));
                    let first = e.extend_relation(REL, ForkId::Main, k).unwrap();
                    assert_eq!(first, want, "seed {seed}: the extension must start at the end of the fork");
                    e.init_raw_pages(x, REL, ForkId::Main, first, k).unwrap();
                    applied_idx += 1;
                    e.commit(x, Some(AppliedPos { index: applied_idx, term: 1 }), &[]).unwrap();
                    pages_acked = pages_attempted;
                    acked_idx = applied_idx;
                    continue;
                }
                let k = 1 + rng.below(3);
                let mut writes = Vec::new();
                for _ in 0..k {
                    counter += 1;
                    let (p, c) = (rng.below(pages_attempted as u64) as u32, rng.below(CELLS as u64) as usize);
                    attempted[p as usize][c] = counter;
                    e.write_bytes(x, PageTag::new(REL, ForkId::Main, p), cell_off(c), &counter.to_le_bytes()).unwrap();
                    writes.push((p, c, counter));
                }
                applied_idx += 1;
                e.commit(x, Some(AppliedPos { index: applied_idx, term: 1 }), &[]).unwrap();
                for (p, c, v) in writes {
                    acked[p as usize][c] = v;
                }
                acked_idx = applied_idx;
                match rng.below(25) {
                    0 => e.checkpoint().unwrap(),
                    1 => {
                        e.pool.bg_write(8);
                    }
                    _ => {}
                }
            }
        }));
        if result.is_err() {
            assert!(inject_sync_failures, "seed {seed}: unexpected panic");
        }
        plan.fail_sync_after.store(-1, Ordering::SeqCst);
        drop(e);
        match rng.below(10) {
            // A process kill: the file system keeps everything in its page cache, so the next
            // start reads a log tail that was never synced (spec 9.3 step 4).
            0 | 1 => {}
            // Power loss, then a death at a random I/O boundary of the next start, then power
            // loss again: recovery must be restartable at every point.
            2 | 3 => {
                sim.crash();
                plan.die_after_ops.store(rng.below(40) as i64, Ordering::SeqCst);
                let _ = catch_unwind(AssertUnwindSafe(|| NativeEngine::open(v.clone(), root(), opts(16)).map(drop)));
                plan.die_after_ops.store(-1, Ordering::SeqCst);
                sim.crash();
            }
            _ => sim.crash(),
        }
    }
    sim
}

#[test]
fn crash_workload_many_seeds() {
    for seed in 0..60 {
        let _ = run_seed(seed, false);
    }
}

#[test]
fn crash_workload_with_sync_failures() {
    for seed in 1000..1040 {
        let _ = run_seed(seed, true);
    }
}

/// M0 gate: the simulator replays a seed identically. Two runs of the same seed, crashes and torn
/// writes included, end in byte-identical file systems; a different seed does not.
#[test]
fn the_same_seed_replays_identically() {
    for seed in [3u64, 17, 1003] {
        let (a, b) = (run_seed(seed, seed > 1000), run_seed(seed, seed > 1000));
        assert_eq!(a.dump(), b.dump(), "seed {seed} did not replay identically");
    }
    assert_ne!(run_seed(3, false).dump(), run_seed(4, false).dump());
}
