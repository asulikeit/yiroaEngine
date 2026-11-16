//! Loom model 3 of spec 18.5 and 19.2: a snapshot reader against the three writers of the published
//! state (xid allocation, abort, the commit barrier), and against each ordered pair of them. Run
//! with `./loom.sh`; an ordinary `cargo test` compiles this file to nothing.
//!
//! The reader is the lock-free `Published::read`. The writers are the production sections, run by
//! one writer thread one after the other: they all hold the publication lock, so two writers are
//! never inside their sections together, and a pair is a second section that follows the first.
//! What the model checks is that a read, started at any point, returns the state before the first
//! section, after it, or after the second, and never a mixture of fields; and that two reads in a
//! row never go backwards.
//!
//! Loom treats `SeqCst` accesses as `AcqRel` (its README) and models `fence(SeqCst)` exactly. The
//! sequence lock carries a `SeqCst` fence under `cfg(loom)` only, which adds nothing to its
//! `SeqCst` accesses, so the model checks the same guarantee; see `seqlock.rs`.
#![cfg(loom)]

use jamsil_page::txnrec::RaftPos;
use jamsil_txn::*;
use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;
use loom::thread;

#[derive(Clone, Copy, Debug)]
enum W {
    Alloc,
    Abort,
    Commit,
}

/// Two xids are active (16 and 17), so that an abort and a commit each have one to finish.
fn setup() -> (Published, MemStore) {
    let p = Published::new(16, 1000, 0);
    assert_eq!((p.allocate_xid(), p.allocate_xid()), (16, 17));
    let store = MemStore::new();
    store.csn_create_pages(0, 0).unwrap(); // the CSN-log page of these xids exists, as in production
    (p, store)
}

fn run(p: &Published, store: &MemStore, w: W) {
    match w {
        W::Alloc => {
            p.allocate_xid();
        }
        W::Abort => p.finish_abort(16),
        W::Commit => p.publish_commit(store, &ClogGate::default(), Some(17), &[], 1100, Some(RaftPos { index: 1, term: 1 })),
    }
}

fn model(first: W, second: W) {
    loom::model(move || {
        // The states a read may return, in order: the same sections run one after the other.
        let (reference, rstore) = setup();
        let s0 = reference.read();
        run(&reference, &rstore, first);
        let s1 = reference.read();
        run(&reference, &rstore, second);
        let s2 = reference.read();
        let allowed = [s0, s1, s2];

        let (p, store) = setup();
        let (p, store) = (Arc::new(p), Arc::new(store));
        let (pw, sw) = (p.clone(), store.clone());
        let writer = thread::spawn(move || {
            run(&pw, &sw, first);
            run(&pw, &sw, second);
        });
        let a = p.read();
        let b = p.read();
        writer.join().unwrap();
        let idx = |s: &PublishedState| allowed.iter().position(|x| x == s);
        let (ia, ib) = (idx(&a), idx(&b));
        assert!(ia.is_some() && ib.is_some(), "a read returned a mixture of fields: {a:?} / {b:?} not in {allowed:?}");
        assert!(ia <= ib, "two reads in a row went backwards: {a:?} then {b:?}");
    });
}

/// One test for each ordered pair of the three writers, so that they run in parallel.
macro_rules! pair {
    ($name:ident, $first:ident, $second:ident) => {
        #[test]
        fn $name() {
            model(W::$first, W::$second);
        }
    };
}

pair!(a_reader_against_allocation_then_allocation, Alloc, Alloc);
pair!(a_reader_against_allocation_then_abort, Alloc, Abort);
pair!(a_reader_against_allocation_then_commit, Alloc, Commit);
pair!(a_reader_against_abort_then_allocation, Abort, Alloc);
pair!(a_reader_against_abort_then_commit, Abort, Commit);
pair!(a_reader_against_commit_then_allocation, Commit, Alloc);
pair!(a_reader_against_commit_then_abort, Commit, Abort);

/// The control: a sequence lock with weaker orderings and no fence, the way a careless rewrite
/// might write it. Loom must find a read that mixes two states.
#[test]
#[should_panic(expected = "torn")]
fn with_acquire_and_release_the_reader_can_see_a_torn_state() {
    loom::model(|| {
        let (seq, a, b) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let (s2, a2, b2) = (seq.clone(), a.clone(), b.clone());
        let writer = thread::spawn(move || {
            s2.store(1, Ordering::Release);
            a2.store(1, Ordering::Relaxed);
            b2.store(1, Ordering::Relaxed);
            s2.store(2, Ordering::Release);
        });
        let s1 = seq.load(Ordering::Acquire);
        let (x, y) = (a.load(Ordering::Relaxed), b.load(Ordering::Relaxed));
        let s3 = seq.load(Ordering::Acquire);
        if s1 == s3 && s1 % 2 == 0 {
            assert_eq!(x, y, "torn: the reader accepted fields from two states");
        }
        writer.join().unwrap();
    });
}
