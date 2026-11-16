//! Spec 7.6, answer A5: `mark_dirty_hint` sets the frame's dirty flag first, reads the redo point
//! second and decides third. The loom model of the dirty flag against the redo point
//! (`crates/jamsil-engine/tests/loom_dirty_redo.rs`) assumes this order; this test pins it in the
//! production code.

use jamsil_bufmgr::*;
use jamsil_page::page::PageRef;
use jamsil_page::*;
use jamsil_page::walrec::WalPermit;
use jamsil_vfs::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use jamsil_vfs::sync::Mutex;
use jamsil_vfs::{SimVfs, Vfs};
use std::path::Path;
use std::sync::Arc;

/// A WAL that records how many frames were dirty at the moment its redo point was read.
struct SpyWal {
    pool: Mutex<Option<Arc<BufferPool>>>,
    redo: Lsn,
    dirty_when_read: AtomicUsize,
    images: AtomicU64,
    /// The log no longer accepts records: it gives no permit.
    declines: AtomicBool,
}

impl WalFlush for SpyWal {
    fn flush(&self, _upto: Lsn) -> Result<(), BufError> {
        Ok(())
    }
    fn redo_lsn(&self) -> Lsn {
        let pool = self.pool.lock().clone().expect("pool attached");
        self.dirty_when_read.store(pool.dirty_count(), Ordering::SeqCst);
        self.redo
    }
    fn permit_page_image(&self) -> Option<WalPermit> {
        if self.declines.load(Ordering::SeqCst) {
            None
        } else {
            Some(WalPermit::issue(16 * 1024))
        }
    }
    fn insert_page_image(&self, _permit: WalPermit, _tag: PageTag, _page: &[u8]) -> Lsn {
        self.images.fetch_add(1, Ordering::SeqCst);
        self.redo + 50
    }
    fn insert_lsn(&self) -> Lsn {
        u64::MAX
    }
}

fn setup(redo: Lsn) -> (Arc<SpyWal>, Arc<BufferPool>) {
    let v: Arc<dyn Vfs> = Arc::new(SimVfs::new(1));
    let smgr = Arc::new(Smgr::new(v, Path::new("/g")).unwrap());
    let wal = Arc::new(SpyWal {
        pool: Mutex::new(None),
        redo,
        dirty_when_read: AtomicUsize::new(usize::MAX),
        images: AtomicU64::new(0),
        declines: AtomicBool::new(false),
    });
    let pool = BufferPool::new(8, smgr, wal.clone());
    *wal.pool.lock() = Some(pool.clone());
    (wal, pool)
}

#[test]
fn the_flag_is_set_before_the_redo_point_is_read() {
    let (wal, pool) = setup(100);
    let p = pool.pin(PageTag::new(100, ForkId::Main, 0), ReadMode::Init).unwrap();
    assert_eq!(pool.dirty_count(), 0);
    // A fresh page has page_lsn 0, which is at or below the redo point: an image is due.
    assert!(p.try_hint(|page| page[100] = 7), "nothing contends the latch and the log accepts records");
    assert_eq!(wal.dirty_when_read.load(Ordering::SeqCst), 1, "the frame must already be dirty when the redo point is read");
    assert_eq!(wal.images.load(Ordering::SeqCst), 1, "a page not logged since the redo point gets a standalone image");
    assert_eq!(p.read()[100], 7, "the hint is in the page, and so in the image");
    let lsn = PageRef(&p.read()[..]).lsn();
    assert_eq!(lsn, 150, "and its page_lsn moves to the end of the image record");

    // The same page again: its lsn is now above the redo point, so the hint only marks it dirty.
    assert!(p.try_hint(|page| page[101] = 8));
    assert_eq!(wal.images.load(Ordering::SeqCst), 1, "no second image");
}

/// Spec 7.6: a hint is optional. A contended latch or a log that does not accept records skips it:
/// the closure does not run, nothing is dirtied or logged, and the call neither blocks nor fails.
#[test]
fn a_hint_is_skipped_when_the_latch_is_contended_or_the_log_declines() {
    let (wal, pool) = setup(100);
    let p = pool.pin(PageTag::new(100, ForkId::Main, 0), ReadMode::Init).unwrap();

    let held = p.write(); // another writer holds the latch
    let mut ran = false;
    assert!(!p.try_hint(|_| ran = true), "a contended latch skips the hint, without waiting for it");
    assert!(!ran, "and the closure does not run");
    drop(held);

    wal.declines.store(true, Ordering::SeqCst);
    assert!(!p.try_hint(|_| ran = true), "a log that gives no permit skips the hint");
    assert!(!ran);
    assert_eq!(pool.dirty_count(), 0, "nothing was dirtied");
    assert_eq!(wal.images.load(Ordering::SeqCst), 0, "and nothing was logged");

    wal.declines.store(false, Ordering::SeqCst);
    assert!(p.try_hint(|_| ran = true));
    assert!(ran, "once the latch is free and the log accepts records, the hint is set");
}

/// The other half of the argument: the checkpoint's scan runs after its redo point is published.
/// `mark_checkpoint_needed` takes the point and a debug build checks that it is published.
#[test]
fn the_checkpoint_scan_needs_its_redo_point_published() {
    let (_wal, pool) = setup(500); // the published redo point is 500
    assert_eq!(pool.mark_checkpoint_needed(500), 0, "scanning for the point that is published is fine");
    assert_eq!(pool.mark_checkpoint_needed(400), 0, "so is an older one");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "before its redo point 900 was published")]
fn a_scan_for_a_redo_point_that_is_not_published_yet_is_refused() {
    let (_wal, pool) = setup(500);
    pool.mark_checkpoint_needed(900);
}
