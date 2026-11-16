//! Session slots and the horizon (spec 12.3).
//!
//! Each session owns a slot in a fixed array and stores its snapshot's `csn` and `xmin` there. The
//! slot is first set to the current global horizon, then the snapshot is taken, then the slot is
//! updated: a horizon computation can therefore never overtake a snapshot that is being taken.

use crate::published::Published;
use jamsil_page::*;
use jamsil_vfs::sync::atomic::{AtomicU64, STORE_BEFORE_LOAD};

/// What an empty slot holds: it pins nothing.
pub const NO_SNAPSHOT: u64 = u64::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotId(usize);

pub struct SessionSlots {
    used: Vec<AtomicU64>,
    csn: Vec<AtomicU64>,
    xmin: Vec<AtomicU64>,
}

/// The oldest state any snapshot or the present still needs: pruning and vacuum may remove what is
/// dead to everything at or after it (spec 12.11, 14.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Horizon {
    /// Every xid below this is finished for every snapshot.
    pub xmin: Xid,
    /// The oldest CSN a snapshot can still see at.
    pub csn: Csn,
}

impl SessionSlots {
    pub fn new(n: usize) -> Self {
        let arr = |v: u64| (0..n).map(|_| AtomicU64::new(v)).collect::<Vec<_>>();
        Self { used: arr(0), csn: arr(NO_SNAPSHOT), xmin: arr(NO_SNAPSHOT) }
    }

    /// Takes a free slot, or `None` if all are in use.
    pub fn claim(&self) -> Option<SlotId> {
        (0..self.used.len())
            .find(|i| self.used[*i].compare_exchange(0, 1, STORE_BEFORE_LOAD, STORE_BEFORE_LOAD).is_ok())
            .map(SlotId)
    }

    pub fn set(&self, id: SlotId, csn: Csn, xmin: Xid) {
        self.csn[id.0].store(csn, STORE_BEFORE_LOAD);
        self.xmin[id.0].store(xmin, STORE_BEFORE_LOAD);
    }

    /// The slot pins nothing until it is set again (the end of a READ COMMITTED statement).
    pub fn clear(&self, id: SlotId) {
        self.set(id, NO_SNAPSHOT, NO_SNAPSHOT);
    }

    pub fn release(&self, id: SlotId) {
        self.clear(id);
        self.used[id.0].store(0, STORE_BEFORE_LOAD);
    }

    pub fn in_use(&self) -> usize {
        self.used.iter().filter(|u| u.load(STORE_BEFORE_LOAD) != 0).count()
    }

    /// The horizon. It starts from the published state, which only moves forward (`oldest_active_xid`
    /// never decreases), and takes the minimum with every slot. A slot is set before its snapshot is
    /// taken, so a snapshot that is being taken is already counted. Each of the two numbers of a slot
    /// is a lower bound of the snapshot's own, so reading them one by one cannot overstate the
    /// horizon.
    pub fn horizon(&self, published: &Published) -> Horizon {
        let g = published.read();
        let (mut xmin, mut csn) = (g.oldest_active_xid, g.last_published_csn);
        for i in 0..self.csn.len() {
            xmin = xmin.min(self.xmin[i].load(STORE_BEFORE_LOAD));
            csn = csn.min(self.csn[i].load(STORE_BEFORE_LOAD));
        }
        Horizon { xmin, csn }
    }

    /// The numbers a slot holds; for tests.
    pub fn get(&self, id: SlotId) -> (Csn, Xid) {
        (self.csn[id.0].load(STORE_BEFORE_LOAD), self.xmin[id.0].load(STORE_BEFORE_LOAD))
    }
}
