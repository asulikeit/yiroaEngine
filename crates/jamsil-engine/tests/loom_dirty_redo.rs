//! Loom model 5 of spec 18.5 and 19.2: the frame's dirty flag against the checkpoint's published
//! redo point. Run with `./loom.sh` (it sets `--cfg loom`); an ordinary `cargo test` compiles this
//! file to nothing.
//!
//! Two threads and two atomics. A page change stores the dirty flag and then loads the redo point
//! (`mark_dirty` before the WAL insert, spec 7.5, and `mark_dirty_hint`, spec 7.6: both decide from
//! the redo point they read after the flag is set). A checkpoint stores its new redo point and then
//! loads the flag (the frame scan of spec 9.2). The page's `page_lsn` lies between the old and the
//! new redo point, so a change that reads the old point logs no image, and then the scan must find
//! the frame dirty and write it before the checkpoint can complete.
//!
//! The four atomic operations are the production functions, `jamsil_bufmgr::dirty_flag` and
//! `jamsil_wal::redo_point`, with their production orderings. What the model assumes rather than
//! checks is the order of the calls in `mark_dirty_hint` and in the checkpoint; the first is tested
//! in `crates/jamsil-bufmgr/tests/hint_order.rs`, and the stress test
//! `hints_racing_checkpoints_leave_no_unrepairable_page` runs the real code path.
#![cfg(loom)]

use jamsil_bufmgr::dirty_flag;
use jamsil_wal::redo_point;
use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;
use loom::thread;

const OLD_REDO: u64 = 100;
const NEW_REDO: u64 = 200;
const PAGE_LSN: u64 = 150;

#[test]
fn a_page_change_and_a_checkpoint_never_both_miss_each_other() {
    loom::model(|| {
        let state = Arc::new(AtomicU64::new(0));
        let point = Arc::new(AtomicU64::new(OLD_REDO));

        let (s, p) = (state.clone(), point.clone());
        let page_change = thread::spawn(move || {
            dirty_flag::set(&s);
            let redo = redo_point::read(&p);
            PAGE_LSN <= redo // true: a full-page image is logged
        });

        redo_point::publish(&point, NEW_REDO);
        let seen_dirty = dirty_flag::is_set(&state);

        let image_logged = page_change.join().unwrap();
        assert!(
            image_logged || seen_dirty,
            "no image was logged and the checkpoint's scan did not see the frame dirty: a torn write \
             of this page would have no record to repair it from"
        );
    });
}

/// The same model with the orderings weakened to what a careless rewrite might choose. Loom must
/// find the outcome in which both sides miss each other, or the model above proves nothing.
#[test]
#[should_panic(expected = "no image was logged")]
fn with_acquire_and_release_both_sides_can_miss_each_other() {
    loom::model(|| {
        let state = Arc::new(AtomicU64::new(0));
        let point = Arc::new(AtomicU64::new(OLD_REDO));

        let (s, p) = (state.clone(), point.clone());
        let page_change = thread::spawn(move || {
            s.fetch_or(1, Ordering::Release);
            let redo = p.load(Ordering::Acquire);
            PAGE_LSN <= redo
        });

        point.store(NEW_REDO, Ordering::Release);
        let seen_dirty = state.load(Ordering::Acquire) & 1 != 0;

        let image_logged = page_change.join().unwrap();
        assert!(image_logged || seen_dirty, "no image was logged and the checkpoint's scan did not see the frame dirty");
    });
}

/// The order is the whole of answer A5. With the production functions but the load before the
/// store, which is how the hint worked before the answer, both sides can miss each other.
#[test]
#[should_panic(expected = "no image was logged")]
fn reading_the_redo_point_before_setting_the_flag_leaves_the_window_open() {
    loom::model(|| {
        let state = Arc::new(AtomicU64::new(0));
        let point = Arc::new(AtomicU64::new(OLD_REDO));

        let (s, p) = (state.clone(), point.clone());
        let page_change = thread::spawn(move || {
            let redo = redo_point::read(&p); // wrong: decide first ...
            dirty_flag::set(&s); // ... mark afterwards
            PAGE_LSN <= redo
        });

        redo_point::publish(&point, NEW_REDO);
        let seen_dirty = dirty_flag::is_set(&state);

        let image_logged = page_change.join().unwrap();
        assert!(image_logged || seen_dirty, "no image was logged and the checkpoint's scan did not see the frame dirty");
    });
}
