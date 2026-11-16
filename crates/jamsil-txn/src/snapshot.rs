//! Snapshots (spec 12.3): a snapshot is a single number, the CSN of the last transaction it can see,
//! with the xid bounds that tell which rule of section 12.4 decides each xid.

use jamsil_page::*;

/// The reader's own transaction, which sees its own changes by command number (spec 12.10).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OwnTxn {
    pub xid: Xid,
    /// Live sub-xids: those not rolled back.
    pub sub_xids: Vec<Xid>,
    /// The current command number. A statement sees its own transaction's changes only from earlier
    /// commands.
    pub command: u32,
}

impl OwnTxn {
    pub fn owns(&self, x: Xid) -> bool {
        self.xid == x || self.sub_xids.contains(&x)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Sees every commit with `commit_csn <= csn`.
    pub csn: Csn,
    /// Every xid below this had finished when the snapshot was taken.
    pub xmin: Xid,
    /// The first xid not yet assigned when the snapshot was taken.
    pub xmax: Xid,
    /// The applied index at that instant (`Query.min_applied`, `wait_applied`).
    pub applied_index: u64,
    pub own: Option<OwnTxn>,
}
