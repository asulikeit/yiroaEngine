//! B3: the WAL buffer is bounded (spec 8.4, 8.5). An inserter that finds it full waits for a flush,
//! a flush that fails does not hang or fail the insert, and the durability marker that `flush`
//! inserts does not wait for itself.

use jamsil_page::rd_u32;
use jamsil_page::walrec::*;
use jamsil_vfs::sync::thread;
use jamsil_vfs::{Advice, DirEntry, FaultPlan, FaultVfs, IoResult, OpenOpts, SimVfs, Vfs, VfsFile};
use std::sync::atomic::AtomicU64;
use jamsil_wal::*;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

const SEG: u64 = 256 * 1024;

fn wal_on(vfs: Arc<dyn Vfs>, buffer_bytes: u64) -> Arc<Wal> {
    let cfg = WalConfig { segment_size: SEG, system_id: 7, buffer_bytes };
    Wal::open(vfs, Path::new("/wal"), cfg, StartPos::Fresh).unwrap()
}

fn payload(tag: u8, seq: u32, len: usize) -> Vec<u8> {
    let mut payload = vec![tag; len];
    payload[..4].copy_from_slice(&seq.to_le_bytes());
    payload
}

/// Inserts one record whose main payload starts with `seq` and is filled with `tag`, through the
/// gate: `check_size` first, then `insert` on the permit (spec 8.8).
fn ins(wal: &Wal, tag: u8, seq: u32, len: usize) -> LsnRange {
    let p = payload(tag, seq, len);
    let rec = RecordBuilder::new(rmgr::TEST, 9).main(&p);
    let permit = wal.check_size(&rec).expect("the gate is open");
    wal.insert(permit, rec)
}

fn read_all(wal: &Wal) -> Vec<WalRecord> {
    let mut rd = wal.reader(wal.segment_size());
    let mut out = Vec::new();
    while let Ok(r) = rd.next_record() {
        out.push(r);
    }
    out
}

#[test]
fn a_full_buffer_makes_the_inserter_flush_and_the_buffer_stays_bounded() {
    let cap = 16 * 1024u64;
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(1));
    let wal = wal_on(vfs, cap);
    let mut max_seen = 0;
    // 3,000 records of about 1 KiB: three megabytes through a 16 KiB buffer, and across several
    // segments, so the SWITCH padding is accounted for as well.
    for i in 0..3000u32 {
        ins(&wal, 1, i, 900);
        max_seen = max_seen.max(wal.buffered_bytes());
    }
    assert!(max_seen <= cap, "the buffer held {max_seen} bytes, capacity {cap}");
    assert!(wal.insert_lsn() > 10 * SEG / 4, "the test must cross segment boundaries");
    wal.flush_all().unwrap();

    let got: Vec<u32> = read_all(&wal).iter().filter(|r| r.rmgr == rmgr::TEST).map(|r| rd_u32(&r.main, 0)).collect();
    assert_eq!(got, (0..3000).collect::<Vec<_>>(), "every record survives, in order");
}

#[test]
fn a_record_larger_than_the_buffer_is_still_accepted_when_the_buffer_is_empty() {
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(2));
    let wal = wal_on(vfs, 4096);
    ins(&wal, 2, 0, 20_000);
    ins(&wal, 2, 1, 20_000);
    wal.flush_all().unwrap();
    assert_eq!(read_all(&wal).iter().filter(|r| r.rmgr == rmgr::TEST).count(), 2);
}

/// An insert may not fail inside a critical section, and it may not wait forever for a flush that
/// cannot happen: with the disk full and no spare to write into, the log is quarantined, the
/// inserts go on past the capacity, and the failure shows at the callers that flush. The permits
/// were taken before the failure, as each critical section's gate is: a permit taken just before
/// the log is quarantined is honoured (spec 8.4, 8.8). A gate taken afterwards would refuse.
#[test]
fn an_insert_does_not_hang_or_fail_when_the_flush_it_waits_for_fails() {
    let sim = SimVfs::new(3);
    let plan = FaultPlan::new();
    let vfs: Arc<dyn Vfs> = Arc::new(FaultVfs::new(sim, plan.clone(), None));
    let cap = 8 * 1024u64;
    let wal = wal_on(vfs, cap);
    plan.disk_full.store(true, Ordering::SeqCst);

    let (tx, rx) = std::sync::mpsc::channel();
    let w = wal.clone();
    thread::spawn_named("inserter", |m| panic!("{m}"), move || {
        // A permit belongs to its thread, so the thread takes them all while the gate is open.
        let payloads: Vec<Vec<u8>> = (0..2000u32).map(|i| payload(3, i, 900)).collect();
        let permits: Vec<_> = payloads
            .iter()
            .map(|p| w.check_size(&RecordBuilder::new(rmgr::TEST, 9).main(p)).unwrap())
            .collect();
        for (p, permit) in payloads.iter().zip(permits) {
            w.insert(permit, RecordBuilder::new(rmgr::TEST, 9).main(p));
        }
        tx.send(()).unwrap();
    })
    .unwrap();
    rx.recv_timeout(Duration::from_secs(30)).expect("the inserters hung waiting for a flush that cannot succeed");
    assert!(wal.is_quarantined());
    assert!(wal.buffered_bytes() > cap, "after the failure the buffer is allowed to grow");
    assert!(wal.flush_all().is_err(), "and the failure shows where the log is flushed");
    let late = RecordBuilder::new(rmgr::TEST, 9);
    assert!(matches!(wal.check_size(&late), Err(WalError::Quarantined(_))), "a gate taken now refuses");
}

/// Four threads insert through a small buffer while the real writer thread runs. Nothing is lost
/// or reordered within a thread, the chain of records validates, and nothing deadlocks.
#[test]
fn concurrent_inserters_and_the_writer_thread_share_a_small_buffer() {
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(4));
    let wal = wal_on(vfs, 8 * 1024);
    wal.start_writer(Duration::from_millis(5), |m| panic!("{m}"));
    let (n_threads, per_thread) = (4u8, 600u32);
    let (tx, rx) = std::sync::mpsc::channel();
    for t in 0..n_threads {
        let (w, tx) = (wal.clone(), tx.clone());
        thread::spawn_named(&format!("inserter-{t}"), |m| panic!("{m}"), move || {
            for i in 0..per_thread {
                ins(&w, t, i, 500 + (i as usize % 7) * 40);
            }
            tx.send(()).unwrap();
        })
        .unwrap();
    }
    for _ in 0..n_threads {
        rx.recv_timeout(Duration::from_secs(60)).expect("an inserter hung");
    }
    wal.stop().unwrap();
    wal.flush_all().unwrap();

    let mut next = vec![0u32; n_threads as usize];
    for r in read_all(&wal).iter().filter(|r| r.rmgr == rmgr::TEST) {
        let (t, seq) = (r.main[5] as usize, rd_u32(&r.main, 0));
        assert_eq!(seq, next[t], "thread {t}: records must appear in the order it inserted them");
        next[t] += 1;
    }
    assert!(next.iter().all(|n| *n == per_thread), "{next:?}");
}

/// Spec 18.5: a panic in the writer thread quarantines the log. What was inserted so far may have
/// reached the file, so it is an unknown outcome; anything inserted afterwards is a definite
/// failure.
#[test]
fn a_panic_in_the_writer_thread_quarantines_the_log() {
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(5));
    let wal = wal_on(vfs, 1 << 20);
    let before = ins(&wal, 5, 0, 100);
    // A permit taken just before the quarantine is still honoured (spec 8.8).
    let p = payload(5, 1, 100);
    let late = RecordBuilder::new(rmgr::TEST, 9).main(&p);
    let permit = wal.check_size(&late).unwrap();
    wal.quarantine(jamsil_wal::QuarantineKind::Restart, "thread jamsil-wal-writer panicked: boom");
    assert!(wal.is_quarantined());
    assert!(matches!(wal.flush(before.end), Err(WalError::OutcomeUnknown(_))), "bytes handed over before the panic");
    let after = wal.insert(permit, late);
    assert!(matches!(wal.flush(after.end), Err(WalError::Quarantined(_))), "later records never reach the file");
}

// ---- B28: the gap after a SWITCH is never written ----------------------------------------------

/// Counts the bytes that `write_at` is asked to write.
struct CountingVfs {
    inner: Arc<dyn Vfs>,
    written: Arc<AtomicU64>,
}

struct CountingFile {
    inner: Box<dyn VfsFile>,
    written: Arc<AtomicU64>,
}

impl Vfs for CountingVfs {
    fn open(&self, path: &Path, opts: OpenOpts) -> IoResult<Box<dyn VfsFile>> {
        Ok(Box::new(CountingFile { inner: self.inner.open(path, opts)?, written: self.written.clone() }))
    }
    fn rename(&self, from: &Path, to: &Path) -> IoResult<()> {
        self.inner.rename(from, to)
    }
    fn unlink(&self, path: &Path) -> IoResult<()> {
        self.inner.unlink(path)
    }
    fn sync_dir(&self, dir: &Path) -> IoResult<()> {
        self.inner.sync_dir(dir)
    }
    fn list(&self, dir: &Path) -> IoResult<Vec<DirEntry>> {
        self.inner.list(dir)
    }
    fn remove_dir(&self, dir: &Path) -> IoResult<()> {
        self.inner.remove_dir(dir)
    }
    fn free_space(&self, dir: &Path) -> IoResult<u64> {
        self.inner.free_space(dir)
    }
    fn create_dir_all(&self, dir: &Path) -> IoResult<()> {
        self.inner.create_dir_all(dir)
    }
    fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
    }
    fn lock_exclusive(&self, path: &Path) -> IoResult<Box<dyn Send + Sync>> {
        self.inner.lock_exclusive(path)
    }
}

impl VfsFile for CountingFile {
    fn read_at(&self, buf: &mut [u8], off: u64) -> IoResult<usize> {
        self.inner.read_at(buf, off)
    }
    fn write_at(&self, buf: &[u8], off: u64) -> IoResult<()> {
        self.written.fetch_add(buf.len() as u64, Ordering::SeqCst);
        self.inner.write_at(buf, off)
    }
    fn sync_data(&self) -> IoResult<()> {
        self.inner.sync_data()
    }
    fn allocate(&self, off: u64, len: u64) -> IoResult<()> {
        self.inner.allocate(off, len)
    }
    fn reserve(&self, off: u64, len: u64) -> IoResult<()> {
        self.inner.reserve(off, len)
    }
    fn truncate(&self, len: u64) -> IoResult<()> {
        self.inner.truncate(len)
    }
    fn advise(&self, off: u64, len: u64, hint: Advice) {
        self.inner.advise(off, len, hint)
    }
    fn len(&self) -> IoResult<u64> {
        self.inner.len()
    }
}

/// Spec 8.1: the bytes between a `SWITCH` and the end of its segment are never written. A segment
/// is zero-filled when it is created and never reused, so the gap is zero already, and the written
/// position jumps over it. Without this, every segment switch wrote up to a whole segment of zeros,
/// and the inserter that crossed it waited behind the write.
#[test]
fn the_gap_after_a_switch_is_never_written() {
    let sim = SimVfs::new(6);
    let written = Arc::new(AtomicU64::new(0));
    let vfs: Arc<dyn Vfs> = Arc::new(CountingVfs { inner: Arc::new(sim.clone()), written: written.clone() });
    let wal = wal_on(vfs, 1 << 20);

    // Fill the first segment until less than 60 KB is left, then insert a record of about 64 KB,
    // which does not fit: a SWITCH pads the rest of the segment.
    let mut n = 0u32;
    while wal.insert_lsn() % SEG < SEG - 60 * 1024 || wal.insert_lsn() / SEG < 1 {
        ins(&wal, 7, n, 8 * 1024);
        n += 1;
    }
    let before = wal.insert_lsn();
    ins(&wal, 7, n, 64_000);
    n += 1;
    let after = wal.insert_lsn();
    assert!(after / SEG > before / SEG, "the big record must have gone into the next segment");
    let gap = (before / SEG + 1) * SEG - before;
    assert!(gap > 50_000, "the test wants a large gap, got {gap}");
    wal.flush_all().unwrap();

    // Everything between the start of the stream and the insert position, minus the gap, plus the
    // 32-byte headers that the creation of a spare writes: nothing for the gap.
    let stream = wal.insert_lsn() - SEG;
    let w = written.load(Ordering::SeqCst);
    assert!(w + gap - 1000 <= stream, "{w} bytes were written for a stream of {stream} bytes with a gap of {gap}");

    // And the log still reads back whole, across the gap.
    let got: Vec<u32> = read_all(&wal).iter().filter(|r| r.rmgr == rmgr::TEST).map(|r| rd_u32(&r.main, 0)).collect();
    assert_eq!(got, (0..n).collect::<Vec<_>>());
    let seg0 = sim.file_bytes(&jamsil_wal::segment_path(Path::new("/wal"), 1)).unwrap();
    let tail = &seg0[(before - SEG) as usize + 40..];
    assert!(tail.iter().all(|b| *b == 0), "the gap in the file is zero");
}

/// Spec 8.8: the permit carries the worst-case length that was checked, and a debug build asserts
/// that the record inserted stays within it. A permit for a small record cannot be spent on a large
/// one. (That a record cannot be inserted without a permit at all is enforced by the types.)
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "on a permit for")]
fn a_record_larger_than_its_permit_is_refused_in_a_debug_build() {
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(8));
    let wal = wal_on(vfs, 1 << 20);
    let small = payload(1, 0, 100);
    let permit = wal.check_size(&RecordBuilder::new(rmgr::TEST, 9).main(&small)).unwrap();
    let big = payload(1, 1, 5000);
    wal.insert(permit, RecordBuilder::new(rmgr::TEST, 9).main(&big));
}
