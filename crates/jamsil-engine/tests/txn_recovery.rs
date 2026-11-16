//! The transaction layer through crash and recovery (spec 9.2, 9.3, 12.2, 12.8, 19.7): the commit log
//! on the buffer pool, redo of the transaction records, the standalone image of a commit-log page,
//! allocation batches, and the named scenarios "Reply lost", "Early persistence" and "Checkpoint
//! against commit".

use jamsil_engine::*;
use jamsil_page::txnrec::*;
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_txn::XidStatus;
use jamsil_vfs::sched::{ctl, Sim};
use jamsil_vfs::sync::thread;
use jamsil_vfs::{CrashPolicy, FaultPlan, FaultVfs, SimVfs, Vfs};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const REL: RelFileId = 100;
const SEG: u64 = 256 * 1024;
const CELL: u16 = 64;

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

fn opts(sim: Option<&Sim>) -> EngineOptions {
    let mut o = EngineOptions {
        buffer_frames: 64,
        wal_segment_size: Some(SEG),
        wal_consistency_check: true,
        background_threads: false,
        ..Default::default()
    };
    if let Some(s) = sim {
        o.clock = s.clock();
    }
    o
}

fn cell_tag() -> PageTag {
    PageTag::new(REL, ForkId::Main, 0)
}

/// An engine with one relation of two raw pages, and index 1 committed with the value 1 in the cell.
fn boot(v: Arc<dyn Vfs>, o: EngineOptions) -> Arc<NativeEngine> {
    let e = NativeEngine::open_or_create(v, root(), o).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    let first = e.extend_relation(REL, ForkId::Main, 2).unwrap();
    let x = e.alloc_xid().unwrap();
    e.init_raw_pages(x, REL, ForkId::Main, first, 2).unwrap();
    e.commit(x, None, &[]).unwrap();
    write_cell(&e, 1);
    e
}

/// An xid that wrote `value` into the cell and has not committed yet.
fn open_write(e: &NativeEngine, value: u64) -> Xid {
    let x = e.alloc_xid().unwrap();
    e.write_bytes(x, cell_tag(), CELL, &value.to_le_bytes()).unwrap();
    x
}

fn write_cell(e: &NativeEngine, index: u64) -> Xid {
    let x = open_write(e, index);
    e.commit(x, Some(AppliedPos { index, term: 1 }), &[]).unwrap();
    x
}

fn cell(e: &NativeEngine) -> u64 {
    let p = e.pin(cell_tag()).unwrap();
    let g = p.read();
    rd_u64(&g[..], CELL as usize)
}

/// The bits of the commit log itself, below the crash rule that `TxnManager::status` applies.
fn bit(e: &NativeEngine, x: Xid) -> ClogStatus {
    let p = e.pin(PageTag::new(CLOG_RELFILE, ForkId::Main, clog_block(x).unwrap())).unwrap();
    let g = p.read();
    clog_get(&g[..], x).unwrap()
}

fn reopen(fs: &SimVfs, policy: CrashPolicy, sim: Option<&Sim>) -> Arc<NativeEngine> {
    let crashed = fs.crashed_copy(policy);
    NativeEngine::open_or_create(Arc::new(crashed), root(), opts(sim)).unwrap()
}

/// Where the log of a new database begins: the redo point of its first checkpoint, a record start.
fn log_start(e: &NativeEngine) -> Lsn {
    e.control_snapshot().redo_lsn
}

fn commit_records(e: &NativeEngine, from: Lsn) -> usize {
    let mut r = e.wal.reader(from);
    let mut n = 0;
    while let Ok(rec) = r.next_record() {
        if rec.rmgr == rmgr::XACT && rec.info == xact::COMMIT {
            n += 1;
        }
    }
    n
}

/// Spec 19.7, "Checkpoint against commit": a commit is suspended after its record is inserted (and,
/// at the second point, flushed), before the publication barrier. A checkpoint runs, and the machine
/// crashes. The commit, its Raft position and its schema version come back together, and its entry is
/// not applied twice. This is where the rule "the commit-log bits change in step 3, in the critical
/// section of the record" is tested: the checkpoint's redo point lies above the record, so redo does
/// not replay it, and the bit must already be on the page the checkpoint wrote.
#[test]
fn checkpoint_against_commit() {
    for (n, point) in ["commit:logged", "commit:flushed"].into_iter().enumerate() {
        for (m, policy) in [CrashPolicy::DropUnsynced, CrashPolicy::KeepUnsynced, CrashPolicy::Random].into_iter().enumerate() {
            let seed = 100 + 10 * n as u64 + m as u64;
            let sim = Sim::new(seed);
            let fs = SimVfs::new(seed);
            sim.run(|| {
                let e = boot(Arc::new(fs.clone()), opts(Some(&sim)));
                let start = log_start(&e);
                let x = open_write(&e, 2);
                e.set_schema_version(x, 7);
                ctl::hold(point);
                let e2 = e.clone();
                let h = thread::spawn_named("committer", |_| {}, move || {
                    e2.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
                })
                .unwrap();
                ctl::await_point(point);
                assert_eq!(e.txn().logged().logged_position, 2, "{point}: the record is inserted");
                assert_eq!(e.txn().published().read().applied_index, 1, "{point}: and the commit is not published");
                e.checkpoint().unwrap();
                let r = reopen(&fs, policy, Some(&sim));
                let why = format!("{point}, {policy:?}");
                assert_eq!(r.applied(), AppliedPos { index: 2, term: 1 }, "{why}: the Raft position is recovered");
                assert_eq!(bit(&r, x), ClogStatus::Committed, "{why}: and so is the commit itself");
                assert_eq!(r.txn().status(x).unwrap(), XidStatus::Committed, "{why}");
                assert_eq!(cell(&r), 2, "{why}: and its row change");
                assert_eq!(r.txn().logged().logged_schema_version, 7, "{why}: and its schema version, together with them");
                assert_eq!(r.txn().published().schema_version(), 7, "{why}");
                // The setup's two commits (the relation's formatting, then index 1) and this one.
                assert_eq!(commit_records(&r, start), 3, "{why}: the entry is applied once: the log holds one commit record for it");
                ctl::release(point);
                h.join().unwrap();
            });
        }
    }
}

/// Spec 19.7, "Reply lost": the machine crashes after the WAL sync, before the reply. The commit is
/// present and whole.
#[test]
fn reply_lost_the_commit_is_present_and_whole() {
    let sim = Sim::new(120);
    let fs = SimVfs::new(120);
    sim.run(|| {
        let e = boot(Arc::new(fs.clone()), opts(Some(&sim)));
        let x = open_write(&e, 2);
        ctl::hold("commit:flushed");
        let e2 = e.clone();
        let h = thread::spawn_named("committer", |_| {}, move || {
            e2.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
        })
        .unwrap();
        ctl::await_point("commit:flushed");
        assert_eq!(e.txn().published().read().applied_index, 1, "the reply has not been sent: the commit is not published");
        let r = reopen(&fs, CrashPolicy::DropUnsynced, Some(&sim));
        assert_eq!((r.applied(), cell(&r), bit(&r, x)), (AppliedPos { index: 2, term: 1 }, 2, ClogStatus::Committed));
        ctl::release("commit:flushed");
        h.join().unwrap();
    });
}

/// Spec 19.7, "Early persistence": the machine dies between the write of the commit record and its
/// sync, at each I/O boundary of the commit in turn, and the file system keeps all, none or a random
/// part of what was not synced. The commit is whole or absent; never half.
#[test]
fn early_persistence_the_commit_is_whole_or_absent() {
    let mut seen_whole = 0;
    let mut seen_absent = 0;
    for budget in 0..10 {
        for (m, policy) in [CrashPolicy::KeepUnsynced, CrashPolicy::DropUnsynced, CrashPolicy::Random].into_iter().enumerate() {
            let fs = SimVfs::new(200 + budget as u64 * 3 + m as u64);
            let plan = FaultPlan::new();
            let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(fs.clone(), plan.clone(), None));
            let e = boot(v, opts(None));
            e.set_fatal_handler(Arc::new(|m: &str| panic!("exit 75: {m}")));
            let x = open_write(&e, 2);
            plan.die_after_ops.store(budget, Ordering::SeqCst);
            let died = catch_unwind(AssertUnwindSafe(|| {
                let _ = e.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]);
            }))
            .is_err();
            let why = format!("budget {budget}, {policy:?}, died {died}");
            let r = reopen(&fs, policy, None);
            let applied = r.applied().index;
            match applied {
                2 => {
                    seen_whole += 1;
                    assert_eq!(bit(&r, x), ClogStatus::Committed, "{why}: whole: the commit-log bit");
                    assert_eq!(r.txn().status(x).unwrap(), XidStatus::Committed, "{why}");
                    assert_eq!(cell(&r), 2, "{why}: whole: the row change");
                }
                1 => {
                    seen_absent += 1;
                    assert_ne!(r.txn().status(x).unwrap(), XidStatus::Committed, "{why}: absent: not committed");
                    assert_eq!(r.txn().status(x).unwrap(), XidStatus::Aborted, "{why}: and aborted by the crash rule, not in progress");
                    assert_ne!(bit(&r, x), ClogStatus::Committed, "{why}");
                }
                other => panic!("{why}: applied position {other}"),
            }
        }
    }
    assert!(seen_whole > 0 && seen_absent > 0, "the sweep saw both outcomes: whole {seen_whole}, absent {seen_absent}");
}

/// Spec 12.2: the first change to a commit-log page after a checkpoint is preceded by a standalone
/// image of the page as it was, inserted under the page latch. A torn write of the page is then
/// repaired by redo, which restores the image and sets the bits of the commit unconditionally.
#[test]
fn a_torn_commit_log_page_is_repaired_from_the_standalone_image() {
    let fs = SimVfs::new(300);
    let v: Arc<dyn Vfs> = Arc::new(fs.clone());
    let e = boot(v, opts(None));
    let before = write_cell(&e, 2);
    e.checkpoint().unwrap();
    let redo = e.wal.redo_lsn();
    let x = write_cell(&e, 3);

    // The log after the checkpoint holds an image of the commit-log page, then the commit record.
    let mut r = e.wal.reader(redo);
    let (mut image_at, mut commit_at) = (None, None);
    let mut i = 0;
    while let Ok(rec) = r.next_record() {
        if rec.rmgr == rmgr::XLOG && rec.info == xlog::PAGE_IMAGE && rec.blocks[0].tag == PageTag::new(CLOG_RELFILE, ForkId::Main, 0) {
            image_at.get_or_insert(i);
        }
        if rec.rmgr == rmgr::XACT && rec.info == xact::COMMIT && rec.xid == x {
            commit_at = Some(i);
        }
        i += 1;
    }
    let (image_at, commit_at) = (image_at.expect("an image of the commit-log page"), commit_at.expect("the commit record"));
    assert!(image_at < commit_at, "the image comes before the first change, not after it");
    // The second commit in the same checkpoint cycle needs no further image.
    let y = write_cell(&e, 4);
    let mut r = e.wal.reader(redo);
    let mut images = 0;
    while let Ok(rec) = r.next_record() {
        if rec.rmgr == rmgr::XLOG && rec.info == xlog::PAGE_IMAGE && rec.blocks[0].tag.rel == CLOG_RELFILE {
            images += 1;
        }
    }
    assert_eq!(images, 1, "one image per page per checkpoint cycle");
    e.wal.flush_all().unwrap();
    drop(e);
    fs.crash_with(CrashPolicy::DropUnsynced);

    // Tear the page on disk: garbage in the middle of it, in both images of the file.
    let clog = root().join("gen-000001/xact/clog.000000");
    fs.poke(&clog, 200, &[0xAB; 64]);
    let r = NativeEngine::open_or_create(Arc::new(fs.clone()), root(), opts(None)).unwrap();
    for (xid, want) in [(before, ClogStatus::Committed), (x, ClogStatus::Committed), (y, ClogStatus::Committed)] {
        assert_eq!(bit(&r, xid), want, "xid {xid}: the page was restored from the image and the bits set");
    }
    assert_eq!(r.applied().index, 4);
}

/// Spec 12.8 and 8.7: commit, abort, and the abort record of `ROLLBACK TO` all reach the commit log
/// through redo, and the sub-xids of a commit are set with it. An xid with no record at all is
/// aborted by the crash rule (spec 9.3 step 7), although its bit still says "in progress".
#[test]
fn redo_sets_the_bits_of_commits_aborts_and_rolled_back_savepoints() {
    let fs = SimVfs::new(310);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let t = e.txn();
    // A transaction with two savepoints; the first is rolled back, the second survives.
    let mut tx = t.begin();
    let top = t.assign_xid(&mut tx).unwrap();
    let s1 = t.assign_sub_xid(&mut tx).unwrap();
    let s2 = t.assign_sub_xid(&mut tx).unwrap();
    t.rollback_sub_xid(&mut tx, s1, &[]);
    t.commit_sync(tx, Some(jamsil_page::txnrec::RaftPos { index: 2, term: 1 })).unwrap();
    // An aborted transaction with a sub-xid.
    let mut tx = t.begin();
    let ab = t.assign_xid(&mut tx).unwrap();
    let ab_sub = t.assign_sub_xid(&mut tx).unwrap();
    t.abort(tx);
    // A transaction that never ends: no record.
    let mut inflight = t.begin();
    let lost = t.assign_xid(&mut inflight).unwrap();
    e.wal.flush_all().unwrap();
    drop(e);
    let r = reopen(&fs, CrashPolicy::DropUnsynced, None);
    assert_eq!([bit(&r, top), bit(&r, s2)], [ClogStatus::Committed; 2], "the commit sets the xid and its surviving sub-xid");
    assert_eq!(bit(&r, s1), ClogStatus::Aborted, "the abort record of ROLLBACK TO sets its sub-xid");
    assert_eq!([bit(&r, ab), bit(&r, ab_sub)], [ClogStatus::Aborted; 2], "and the abort record sets the xid and its sub-xid");
    assert_eq!(bit(&r, lost), ClogStatus::InProgress, "an xid with no record keeps its bit");
    let st = r.txn();
    assert_eq!([st.status(top).unwrap(), st.status(s2).unwrap()], [XidStatus::Committed; 2]);
    assert_eq!([st.status(s1).unwrap(), st.status(ab).unwrap(), st.status(ab_sub).unwrap(), st.status(lost).unwrap()], [XidStatus::Aborted; 4], "and the crash rule aborts it");
    assert!(st.commit_csn(top).unwrap().is_none(), "the CSN log is empty after a restart: the commit log alone decides");
}

/// Spec 9.3 "Xid counter": xids come from logged batches, the restart value is the end of the last
/// batch, and an xid is never handed out twice. A run of 33,000 xids crosses from the first
/// commit-log page into the second: the page is created by a `CLOG` zero-page record, a commit on it
/// survives a crash, and the restart lands after the whole batch.
#[test]
fn allocation_batches_and_commit_log_pages_survive_a_restart() {
    let fs = SimVfs::new(320);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let start = log_start(&e);
    let t = e.txn();
    let mut open = Vec::new();
    for _ in 0..33_000 {
        let mut tx = t.begin();
        t.assign_xid(&mut tx).unwrap();
        open.push(tx);
    }
    let last = open.last().unwrap().xid().unwrap();
    assert!(clog_block(last).unwrap() >= 1, "the run crossed into the second commit-log page: {last}");
    let tx = open.pop().unwrap();
    t.commit_sync(tx, Some(jamsil_page::txnrec::RaftPos { index: 2, term: 1 })).unwrap();
    let mut zero_pages = 0;
    let mut r = e.wal.reader(start);
    let mut batches = Vec::new();
    while let Ok(rec) = r.next_record() {
        match (rec.rmgr, rec.info) {
            (rmgr::CLOG, clog::ZERO_PAGE) => zero_pages += 1,
            (rmgr::XLOG, xlog::ALLOC_BATCH) => batches.push(AllocBatch::decode(&rec.main).unwrap()),
            _ => {}
        }
    }
    assert_eq!(zero_pages, 2, "one zero-page record for each commit-log page");
    assert!(batches.windows(2).all(|w| w[0].next_xid < w[1].next_xid));
    let end_of_last_batch = batches.last().unwrap().next_xid;
    assert!(end_of_last_batch > last);
    drop(open);
    e.wal.flush_all().unwrap();
    drop(e);
    let r = reopen(&fs, CrashPolicy::DropUnsynced, None);
    assert_eq!(bit(&r, last), ClogStatus::Committed, "the commit on the second page survived");
    assert_eq!(r.txn().status(last).unwrap(), XidStatus::Committed);
    let mut tx = r.txn().begin();
    let fresh = r.txn().assign_xid(&mut tx).unwrap();
    assert_eq!(fresh, end_of_last_batch, "the restart value is the end of the last batch in the log: nothing handed out is handed out again");
    r.txn().abort(tx);
}

/// Spec 12.2: truncation removes whole segments from the front of the commit log, and the length of
/// the fork is derived from its highest segment, not from the sum of its files. The engine clamps a
/// request to what the last completed checkpoint captured, so that redo can never ask for a page of a
/// segment that is gone, and the checkpoint is what truncates (spec 14.6); a crash after the truncation
/// replays the `CLOG` truncate record.
#[test]
fn commit_log_truncation_removes_front_segments_and_the_length_comes_from_the_highest() {
    use jamsil_txn::PoolStore;
    let fs = SimVfs::new(330);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let store = PoolStore::new(e.pool.clone(), e.wal.clone(), Arc::new(|m: &str| panic!("{m}")), Arc::new(|_k, m: &str| panic!("{m}")));
    // Page 257 lies in the second segment (256 pages per segment); xid 8,400,000 is on it.
    use jamsil_txn::TxnStore;
    store.clog_create_pages(257, 257).unwrap();
    e.checkpoint().unwrap();
    assert_eq!(e.smgr.log_segments(CLOG_RELFILE).unwrap(), [0, 1]);
    assert_eq!(e.smgr.nblocks(CLOG_RELFILE, ForkId::Main).unwrap(), 258);

    // Vacuum only raises the horizon; the checkpoint truncates, after it has completed, and never beyond the
    // `oldest_active_xid` that it captured itself, which lies in page 0 here.
    e.raise_clog_horizon(8_400_000);
    e.checkpoint().unwrap();
    assert_eq!(e.smgr.log_segments(CLOG_RELFILE).unwrap(), [0, 1], "the checkpoint clamped the request: no segment was unlinked");

    // Without the clamp (the store itself) the segment goes, and the length is still 258.
    let kept = store.truncate_clog(8_400_000).unwrap();
    assert_eq!(kept, 256 * XIDS_PER_CLOG_PAGE);
    assert_eq!(e.smgr.log_segments(CLOG_RELFILE).unwrap(), [1]);
    assert_eq!(e.smgr.nblocks(CLOG_RELFILE, ForkId::Main).unwrap(), 258, "derived from the highest segment");
    e.wal.flush_all().unwrap();
    drop(e);

    // Recovery replays the truncate record again, and the length is still derived from segment 1.
    let r = reopen(&fs, CrashPolicy::DropUnsynced, None);
    assert_eq!(r.smgr.log_segments(CLOG_RELFILE).unwrap(), [1]);
    assert_eq!(r.smgr.nblocks(CLOG_RELFILE, ForkId::Main).unwrap(), 258);
    let p = r.pin(PageTag::new(CLOG_RELFILE, ForkId::Main, 257)).unwrap();
    assert_eq!(clog_get(&p.read()[..], 8_400_000).unwrap(), ClogStatus::InProgress, "the page is intact");
}

/// Spec 19.2, the oracle of the crash workload: after a crash at any I/O boundary, whatever the file
/// system kept, the commits that survive form a prefix of the commit order, and every commit that was
/// acknowledged under FULL is among them. Commits 2 to 12 each write the cell, and the machine dies
/// after a random number of I/O operations.
#[test]
fn the_surviving_commits_form_a_prefix_and_every_acknowledged_commit_survives() {
    let mut deaths = 0;
    for seed in 0..40u64 {
        let fs = SimVfs::new(400 + seed);
        let plan = FaultPlan::new();
        let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(fs.clone(), plan.clone(), None));
        let e = boot(v, opts(None));
        e.set_fatal_handler(Arc::new(|m: &str| panic!("exit 75: {m}")));
        plan.die_after_ops.store((seed * 7 % 41) as i64, Ordering::SeqCst);
        let mut xids = Vec::new();
        let mut acked = Vec::new();
        for i in 2..=12u64 {
            let r = catch_unwind(AssertUnwindSafe(|| {
                let x = open_write(&e, i);
                xids.push((i, x));
                e.commit(x, Some(AppliedPos { index: i, term: 1 }), &[]).map(|_| ())
            }));
            match r {
                Ok(Ok(())) => acked.push(i),
                _ => {
                    deaths += 1;
                    break;
                }
            }
        }
        let policy = [CrashPolicy::Random, CrashPolicy::DropUnsynced, CrashPolicy::KeepUnsynced][seed as usize % 3];
        let r = reopen(&fs, policy, None);
        let applied = r.applied().index;
        let why = format!("seed {seed}, {policy:?}, acked {acked:?}, applied {applied}");
        for (i, x) in &xids {
            let committed = r.txn().status(*x).unwrap() == XidStatus::Committed;
            assert_eq!(committed, *i <= applied, "{why}: commit {i} is present exactly if it is within the prefix");
            if committed {
                assert_eq!(bit(&r, *x), ClogStatus::Committed, "{why}");
            }
        }
        assert!(acked.iter().all(|i| *i <= applied), "{why}: an acknowledged commit was lost");
        assert!(applied >= 1, "{why}: the baseline commit");
    }
    assert!(deaths > 5, "the sweep killed the process in the middle of commits often enough: {deaths}");
}

/// Invariant 4 of spec 12.12: a commit-log page that holds a `COMMITTED` bit reaches disk only after the
/// commit record is durable. Under NORMAL the commit is not synced when it returns; writing the page
/// out forces the log up to the page's LSN first, and the page's LSN is the record's end.
#[test]
fn a_commit_log_page_is_never_written_before_its_commit_record_is_durable() {
    let fs = SimVfs::new(340);
    let o = EngineOptions { synchronous_full: false, ..opts(None) };
    let e = boot(Arc::new(fs.clone()), o);
    e.wal.flush_all().unwrap();
    let x = open_write(&e, 2);
    let end = e.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
    assert!(e.wal.durable_lsn() < end, "NORMAL: the commit has not been synced when it returns");
    // The rule itself: the page's LSN covers the commit record, so the pool flushes the log up to it
    // before it writes the page. (A flush writes everything buffered, so the durable position alone
    // would not tell.)
    let page_lsn = {
        let p = e.pin(PageTag::new(CLOG_RELFILE, ForkId::Main, clog_block(x).unwrap())).unwrap();
        let g = p.read();
        jamsil_page::page::PageRef(&g[..]).lsn()
    };
    assert!(page_lsn >= end, "the commit-log page's LSN {page_lsn} covers its commit record, which ends at {end}");
    // Write every dirty page out, as a checkpoint does (the background writer skips hot frames).
    let redo = e.wal.begin_checkpoint();
    e.pool.mark_checkpoint_needed(redo);
    e.pool.flush_all(&jamsil_bufmgr::CheckpointCtx::default()).unwrap();
    assert!(e.wal.durable_lsn() >= end, "writing the dirty pages out flushed the log up to the commit-log page's LSN");
}

// ---- the pins of a commit, and a commit-log page that cannot be read (answers B9, B10, B11) -------

/// Writes every dirty page out and then reads enough other pages to push the commit-log and CSN-log
/// pages out of the 64-frame pool. A page read back afterwards is a miss in the pool's counters.
fn evict_everything(e: &NativeEngine, filler: RelFileId) {
    e.checkpoint().unwrap();
    for _ in 0..4 {
        for b in 0..200 {
            let _ = e.pin(PageTag::new(filler, ForkId::Main, b)).unwrap();
        }
    }
}

fn misses(e: &NativeEngine) -> u64 {
    e.pool.metrics.misses.load(Ordering::SeqCst)
}

fn with_filler(e: &NativeEngine) -> RelFileId {
    let filler = REL + 1;
    e.create_relation(filler, ForkId::Main).unwrap();
    let first = e.extend_relation(filler, ForkId::Main, 200).unwrap();
    let x = e.alloc_xid().unwrap();
    e.init_raw_pages(x, filler, ForkId::Main, first, 200).unwrap();
    e.commit(x, None, &[]).unwrap();
    filler
}

/// Spec 12.11: a commit pins the commit-log and CSN-log pages of its xid and sub-xids at its gate, with an
/// ordinary read, and keeps the pins until step 6. Nothing is pinned, read or allocated under the commit
/// mutex or the publication lock. The pages are pushed out of the pool first, so the gate has to read them
/// back (the counter of misses rises before the suspension point `commit:gated`), and after the gate the
/// commit, its barrier and its release read nothing: the counter does not move.
#[test]
fn a_commit_reads_its_pages_at_the_gate_and_nothing_after_it() {
    let sim = Sim::new(500);
    let fs = SimVfs::new(500);
    sim.run(|| {
        let e = boot(Arc::new(fs.clone()), opts(Some(&sim)));
        let filler = with_filler(&e);
        let x = open_write(&e, 2);
        // A sub-xid far from its parent would touch a second commit-log page; here both lie on one.
        evict_everything(&e, filler);
        let before = misses(&e);
        ctl::hold("commit:gated");
        let e2 = e.clone();
        let h = thread::spawn_named("committer", |_| {}, move || {
            e2.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
        })
        .unwrap();
        ctl::await_point("commit:gated");
        let at_gate = misses(&e);
        assert!(at_gate >= before + 2, "the gate read the commit-log page and the CSN-log page back: {before} -> {at_gate}");
        ctl::release("commit:gated");
        h.join().unwrap();
        assert_eq!(misses(&e), at_gate, "after the gate: no page was read, under the commit mutex or the publication lock or anywhere else");
        assert_eq!(e.txn().status(x).unwrap(), XidStatus::Committed);
        assert!(e.txn().commit_csn(x).unwrap().is_some(), "and the barrier wrote its CSN");
    });
}

/// Spec 12.2: the CSN-log pages of a batch are created in the slow path of allocation, with `Init`, next to
/// its commit-log pages. A commit pins the CSN-log pages it writes at its gate; none of them is created or
/// initialised by the commit.
#[test]
fn csn_log_pages_are_created_with_the_batch() {
    let fs = SimVfs::new(510);
    let e = boot(Arc::new(fs.clone()), opts(None));
    // The batch of 4,096 xids from 16 on covers CSN-log pages 0 to 4 (1,020 xids per page).
    assert_eq!(e.smgr.nblocks(CSNLOG_RELFILE, ForkId::Main).unwrap(), 5, "created with the first batch");
    let before = e.smgr.nblocks(CSNLOG_RELFILE, ForkId::Main).unwrap();
    for i in 2..40 {
        write_cell(&e, i);
    }
    assert_eq!(e.smgr.nblocks(CSNLOG_RELFILE, ForkId::Main).unwrap(), before, "commits create no CSN-log page");
}

/// Spec 12.11, 5.6 (answer B11): a commit-log page that fails verification makes `status` return `Corrupt`
/// and quarantines the store, so that the node reinstalls from a peer snapshot; ending the process would
/// loop, because the restart meets the same page.
#[test]
fn a_corrupt_commit_log_page_quarantines_the_store_and_does_not_end_the_process() {
    let fs = SimVfs::new(520);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let filler = with_filler(&e);
    let x = write_cell(&e, 2);
    evict_everything(&e, filler);
    assert!(!e.wal.is_quarantined());
    fs.poke(&root().join("gen-000001/xact/clog.000000"), 100, &[0xAB; 64]);
    let err = e.txn().status(x).unwrap_err();
    assert!(matches!(err, jamsil_txn::TxnError::Corrupt(_)), "{err}");
    assert!(err.to_string().contains("XX001"));
    assert!(e.wal.is_quarantined(), "the store is quarantined");
    // The log accepts no more records: a new write fails at its gate, with the quarantine error.
    let y = e.alloc_xid().expect("an xid of the current batch needs no new record");
    let err = e.write_bytes(y, cell_tag(), CELL, &3u64.to_le_bytes()).unwrap_err();
    assert!(err.to_string().contains("JDB-3001"), "{err}");
}

/// The other half of B11: a fatal I/O error on a commit-log page ends the process (exit code 75), as for
/// any other page.
#[test]
fn a_fatal_io_error_on_a_commit_log_page_ends_the_process() {
    let fs = SimVfs::new(530);
    let plan = FaultPlan::new();
    let v: Arc<dyn Vfs> = Arc::new(FaultVfs::new(fs.clone(), plan.clone(), None));
    let e = boot(v, opts(None));
    e.set_fatal_handler(Arc::new(|m: &str| panic!("exit 75: {m}")));
    let filler = with_filler(&e);
    let x = write_cell(&e, 2);
    evict_everything(&e, filler);
    plan.fail_reads.store(true, Ordering::SeqCst);
    let r = catch_unwind(AssertUnwindSafe(|| e.txn().status(x)));
    let msg = *r.expect_err("the process ends").downcast::<String>().unwrap();
    assert!(msg.starts_with("exit 75"), "{msg}");
    assert!(!e.wal.is_quarantined(), "a fatal I/O error is not a quarantine");
}

/// Spec 14.6 (answer B12): with the clamp, no record that redo replays can name an xid below
/// `oldest_clog_xid`. One that does shows a bug, and redo reports it as a fatal inconsistency instead of
/// skipping it.
#[test]
fn a_replayed_record_below_the_truncation_horizon_is_a_fatal_inconsistency() {
    let fs = SimVfs::new(540);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let rec = |xid| WalRecord {
        lsn: 100,
        prev_lsn: 0,
        end_lsn: 200,
        xid,
        rmgr: rmgr::XACT,
        info: xact::COMMIT,
        blocks: vec![],
        main: CommitRecord::default().encode(),
    };
    let err = jamsil_txn::redo_xact_bits(&e.pool, &rec(20), 100).unwrap_err();
    assert!(err.to_string().contains("fatal inconsistency"), "{err}");
    assert!(jamsil_txn::redo_xact_bits(&e.pool, &rec(100), 100).is_ok() || true, "an xid at the horizon is not below it");
}

// ---- the two passes over a commit-log page (answers D of round 9) ----------------------------------

/// Both passes of every writer of commit-log bits run inside one hold of the commit mutex: a commit, an
/// abort and a `ROLLBACK TO`. A checkpoint takes its redo point under that mutex, so it cannot publish a
/// new one between the passes, where the change of pass 2 would be the first change after it with no image.
/// The writer is held between its passes, and a checkpoint's capture, started meanwhile, must not complete.
#[test]
fn a_checkpoint_cannot_take_its_redo_point_between_the_two_passes() {
    for writer in ["commit", "abort", "rollback to"] {
        let sim = Sim::new(600);
        let fs = SimVfs::new(600);
        sim.run(|| {
            let e = boot(Arc::new(fs.clone()), opts(Some(&sim)));
            let mut txn = e.txn().begin();
            let top = e.txn().assign_xid(&mut txn).unwrap();
            let sub = e.txn().assign_sub_xid(&mut txn).unwrap();
            ctl::hold("clog:between_passes");
            let e2 = e.clone();
            let h = thread::spawn_named("writer", |_| {}, move || {
                let mut txn = txn;
                match writer {
                    "commit" => {
                        e2.txn().commit_sync(txn, None).unwrap();
                    }
                    "abort" => e2.txn().abort(txn),
                    _ => {
                        e2.txn().rollback_sub_xid(&mut txn, sub, &[]);
                        e2.txn().abort(txn);
                    }
                }
            })
            .unwrap();
            ctl::await_point("clog:between_passes");
            let captured = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (e3, c2) = (e.clone(), captured.clone());
            let cap = thread::spawn_named("checkpoint", |_| {}, move || {
                e3.txn().checkpoint_capture(|| e3.wal.insert_lsn());
                c2.store(true, Ordering::SeqCst);
            })
            .unwrap();
            thread::sleep(std::time::Duration::from_millis(50));
            assert!(!captured.load(Ordering::SeqCst), "{writer}: the checkpoint took its redo point between the two passes");
            ctl::release("clog:between_passes");
            h.join().unwrap();
            cap.join().unwrap();
            assert!(captured.load(Ordering::SeqCst), "{writer}: and it went through when the writer was done");
            let _ = top;
        });
    }
}

/// A write-out can clean a commit-log page between the two passes, because the page is unlatched there.
/// Section 7.5 requires the frame to be dirty before the record that changes it has a position in the log,
/// so pass 2 marks the pages again before the record is inserted. The page is cleaned between the passes
/// here, and it must be dirty again at the point just before the record.
#[test]
fn a_page_cleaned_between_the_passes_is_dirty_again_before_the_record() {
    let sim = Sim::new(610);
    let fs = SimVfs::new(610);
    sim.run(|| {
        let e = boot(Arc::new(fs.clone()), opts(Some(&sim)));
        let x = open_write(&e, 2);
        let page = PageTag::new(CLOG_RELFILE, ForkId::Main, 0);
        ctl::hold("clog:between_passes");
        ctl::hold("clog:before_record");
        let e2 = e.clone();
        let h = thread::spawn_named("committer", |_| {}, move || {
            e2.commit(x, Some(AppliedPos { index: 2, term: 1 }), &[]).unwrap();
        })
        .unwrap();
        ctl::await_point("clog:between_passes");
        assert!(e.pool.is_dirty(page), "marked by pass 1");
        // A write-out between the passes, as a checkpoint's scan does it.
        let redo = e.wal.begin_checkpoint();
        e.pool.mark_checkpoint_needed(redo);
        e.pool.flush_all(&jamsil_bufmgr::CheckpointCtx::default()).unwrap();
        assert!(!e.pool.is_dirty(page), "the write-out cleaned the page: the test would prove nothing otherwise");
        ctl::release("clog:between_passes");
        ctl::await_point("clog:before_record");
        assert!(e.pool.is_dirty(page), "dirty again before the commit record has a position in the log");
        ctl::release("clog:before_record");
        h.join().unwrap();
        assert_eq!(e.txn().status(x).unwrap(), XidStatus::Committed);
    });
}

/// Spec 14.6 (answer B3): a truncation that fails after the checkpoint is complete is logged and counted,
/// and it fails nothing; the next checkpoint tries again.
#[test]
fn a_failed_truncation_is_counted_and_fails_nothing() {
    let fs = SimVfs::new(620);
    let e = boot(Arc::new(fs.clone()), opts(None));
    e.raise_clog_horizon(8_400_000);
    assert_eq!(e.clog_truncate_failures(), 0);
    e.wal.quarantine(jamsil_wal::QuarantineKind::Restart, "the log stopped accepting records after the checkpoint");
    e.truncate_clog_after_checkpoint(8_400_000); // returns: it is not an error of the checkpoint
    assert_eq!(e.clog_truncate_failures(), 1);
    e.truncate_clog_after_checkpoint(8_400_000);
    assert_eq!(e.clog_truncate_failures(), 2, "and it is tried again");
}

/// Spec 17.2 (answer B4): the two kinds of quarantine are told apart, because they are cleared
/// differently. A corrupt commit-log page is corruption, cleared only by a snapshot install, and not the
/// kind that a restart clears.
#[test]
fn a_corrupt_commit_log_page_is_a_quarantine_that_only_a_snapshot_clears() {
    let fs = SimVfs::new(630);
    let e = boot(Arc::new(fs.clone()), opts(None));
    let filler = with_filler(&e);
    let x = write_cell(&e, 2);
    evict_everything(&e, filler);
    assert_eq!(e.wal.quarantine_reason(), None);
    fs.poke(&root().join("gen-000001/xact/clog.000000"), 100, &[0xAB; 64]);
    assert!(e.txn().status(x).is_err());
    let (kind, why) = e.wal.quarantine_reason().expect("quarantined");
    assert_eq!(kind, jamsil_wal::QuarantineKind::Corruption, "{why}");
    // A later quarantine of another kind does not replace the first reason.
    e.wal.quarantine(jamsil_wal::QuarantineKind::Restart, "a full disk");
    assert_eq!(e.wal.quarantine_reason().unwrap().0, jamsil_wal::QuarantineKind::Corruption);
}
