//! Allocation of xids, oids and relfile ids in logged batches (spec 8.6, 9.3 "Xid counter").
//!
//! A batch is 4,096 xids, or 1,024 oids or relfile ids. Its record carries three high-water marks,
//! each the first id that the batch does not cover. Nothing is flushed for it: any durable trace of
//! an id lies after its batch record in the log, so a restart that raises each counter to the
//! recorded value never hands an id out twice.
//!
//! The fast path of an xid allocation takes the publication lock and only advances the counter
//! inside the current batch (`Published::allocate_xid_below`). Neither the batch record nor the
//! commit-log pages of a new batch are inserted under that lock: an insert can wait for buffer space
//! and a pin can evict a page. Both belong to the slow path taken when a batch is used up, under the
//! mutex of this allocator. It creates the commit-log pages for the whole new batch, logs the batch,
//! and only then raises the limit that the fast path reads, so no xid of the batch exists before its
//! page does.

use crate::store::{TxnStore, WalSink};
use crate::TxnError;
use jamsil_page::txnrec::{clog_block, csn_block, AllocBatch};
use jamsil_page::*;
use jamsil_vfs::sync::atomic::{AtomicU64, STORE_BEFORE_LOAD};
use jamsil_vfs::sync::Mutex;

pub const XID_BATCH: u64 = 4_096;
pub const ID_BATCH: u64 = 1_024;

struct Slow {
    oid_next: u64,
    oid_end: u64,
    rel_next: u64,
    rel_end: u64,
    /// The commit-log pages `0..clog_pages` have been created.
    clog_pages: u64,
    /// The CSN-log pages below this block have been created. The CSN log is empty after a start, so
    /// the first page is the one that holds the first xid handed out.
    csn_next: u64,
}

pub struct Allocator {
    /// "A mutex of its own": the slow path of every counter.
    slow: Mutex<Slow>,
    /// The size of a batch of xids: `XID_BATCH`, smaller only in tests that want many boundaries.
    xid_batch: u64,
    /// The first xid not covered by a logged batch. The fast path reads it without the mutex; it
    /// only grows, so a stale read is conservative.
    xid_end: AtomicU64,
}

impl Allocator {
    /// `restart` holds the counters recovery established (each the first id not covered by any batch
    /// in the log). Nothing is handed out from them before a new batch is logged.
    pub fn new(restart: AllocBatch) -> Self {
        Self::with_xid_batch(restart, XID_BATCH)
    }

    pub fn with_xid_batch(restart: AllocBatch, xid_batch: u64) -> Self {
        assert!(xid_batch > 0);
        // Every page that holds an xid below the restart value exists: it was created before its xid
        // was handed out. The page of `restart.next_xid` itself, if it is a new one, was never used,
        // so creating it again (a zero page) destroys nothing. On a new database nothing exists yet.
        let clog_pages = if restart.next_xid <= FIRST_NORMAL_XID { 0 } else { u64::from(clog_block(restart.next_xid - 1).unwrap_or(0)) + 1 };
        Self {
            slow: Mutex::new(Slow {
                oid_next: u64::from(restart.next_oid),
                oid_end: u64::from(restart.next_oid),
                rel_next: restart.next_relfile,
                rel_end: restart.next_relfile,
                clog_pages,
                csn_next: u64::from(csn_block(restart.next_xid).unwrap_or(0)),
            }),
            xid_end: AtomicU64::new(restart.next_xid),
            xid_batch,
        }
    }

    /// The first xid not covered by a logged batch.
    pub fn xid_limit(&self) -> Xid {
        self.xid_end.load(STORE_BEFORE_LOAD)
    }

    /// The three high-water marks: each the first id not covered by a logged batch. A checkpoint
    /// records them (spec 9.2 step 2: "Counters ... may only be ahead").
    pub fn high_water(&self) -> AllocBatch {
        let s = self.slow.lock();
        AllocBatch { next_xid: self.xid_end.load(STORE_BEFORE_LOAD), next_oid: s.oid_end as u32, next_relfile: s.rel_end }
    }

    /// The slow path of an xid allocation: the limit the caller saw is used up. Creates the
    /// commit-log pages of the next batch, logs it, and raises the limit. A caller that finds the
    /// limit already raised by another thread has nothing to do.
    pub fn extend_xids(&self, seen_limit: Xid, wal: &dyn WalSink, store: &dyn TxnStore) -> Result<(), TxnError> {
        let mut s = self.slow.lock();
        if self.xid_end.load(STORE_BEFORE_LOAD) != seen_limit {
            return Ok(());
        }
        let new_end = seen_limit.checked_add(self.xid_batch).ok_or(TxnError::Exhausted("xid"))?;
        let last = u64::from(clog_block(new_end - 1).ok_or(TxnError::Exhausted("xid"))?);
        if s.clog_pages <= last {
            // Named suspension points for the scenario that holds the allocating thread halfway
            // (spec 19.7): no other thread may get an xid of the batch while it is half made.
            jamsil_vfs::sched::point("alloc:before_pages");
            store.clog_create_pages(s.clog_pages as BlockNo, last as BlockNo)?;
            s.clog_pages = last + 1;
        }
        wal.log_alloc_batch(AllocBatch { next_xid: new_end, next_oid: s.oid_end as u32, next_relfile: s.rel_end })?;
        // The CSN-log pages of the batch, next to its commit-log pages: created with `Init` here and
        // nowhere else, so two commits can never initialise one page, and unpinned. A commit pins the
        // pages it writes at its gate (spec 12.2).
        let csn_last = u64::from(csn_block(new_end - 1).ok_or(TxnError::Exhausted("xid"))?);
        if s.csn_next <= csn_last {
            store.csn_create_pages(s.csn_next as BlockNo, csn_last as BlockNo)?;
            s.csn_next = csn_last + 1;
        }
        jamsil_vfs::sched::point("alloc:before_limit");
        self.xid_end.store(new_end, STORE_BEFORE_LOAD);
        Ok(())
    }

    pub fn alloc_oid(&self, wal: &dyn WalSink) -> Result<Oid, TxnError> {
        let mut s = self.slow.lock();
        if s.oid_next >= s.oid_end {
            let new_end = s.oid_end + ID_BATCH;
            let end32 = u32::try_from(new_end).map_err(|_| TxnError::Exhausted("oid"))?;
            wal.log_alloc_batch(AllocBatch { next_xid: self.xid_end.load(STORE_BEFORE_LOAD), next_oid: end32, next_relfile: s.rel_end })?;
            s.oid_end = new_end;
        }
        let oid = s.oid_next as Oid;
        s.oid_next += 1;
        Ok(oid)
    }

    pub fn alloc_relfile(&self, wal: &dyn WalSink) -> Result<RelFileId, TxnError> {
        let mut s = self.slow.lock();
        if s.rel_next >= s.rel_end {
            let new_end = s.rel_end.checked_add(ID_BATCH).ok_or(TxnError::Exhausted("relfile id"))?;
            wal.log_alloc_batch(AllocBatch { next_xid: self.xid_end.load(STORE_BEFORE_LOAD), next_oid: s.oid_end as u32, next_relfile: new_end })?;
            s.rel_end = new_end;
        }
        let rel = s.rel_next;
        s.rel_next += 1;
        Ok(rel)
    }
}
