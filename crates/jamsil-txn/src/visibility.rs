//! Visibility (spec 12.4), over the tuple-header fields of section 10.1.
//!
//! *committed-before(S, X)* for snapshot S and xid X is decided by the first rule that applies:
//!
//! 1. X is the reader's own xid or one of its live sub-xids: decided by the command number.
//! 2. `X >= S.xmax`: false.
//! 3. `X < S.xmin`: true if the commit log says committed. A hint bit short-cuts the lookup.
//! 4. Otherwise: true if the CSN log holds a CSN for X and it is at most `S.csn`.
//!
//! Hint bits are consulted only inside rule 3. A `COMMITTED` hint says nothing about when the
//! commit happened, so it never decides visibility for an xid at or above `S.xmin`. An `ABORTED`
//! hint is final for every snapshot and may be used anywhere.
//!
//! A tuple is visible to S if its `xmin` is committed-before S, and its `xmax` is zero, lock-only,
//! or not committed-before S.

use crate::snapshot::Snapshot;
use crate::store::TxnStore;
use crate::TxnError;
use jamsil_page::txnrec::ClogStatus;
use jamsil_page::*;
use std::collections::HashMap;

/// The `infomask` bits of section 10.1 (bits 0 to 9, confirmed in round 8) are defined with the tuple, in
/// `jamsil-page`.
pub use jamsil_page::tuple::infomask;

pub use jamsil_page::tuple::TUPLE_HEADER_LEN;

/// The fields of the 32-byte tuple header that visibility needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleHeader {
    pub xmin: Xid,
    pub xmax: Xid,
    pub cid: u32,
    pub infomask: u16,
}

impl TupleHeader {
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() < TUPLE_HEADER_LEN {
            return corrupt("tuple header shorter than 32 bytes");
        }
        Ok(Self { xmin: rd_u64(b, 0), xmax: rd_u64(b, 8), cid: rd_u32(b, 16), infomask: rd_u16(b, 26) })
    }

    pub fn encode(&self) -> [u8; TUPLE_HEADER_LEN] {
        let mut b = [0u8; TUPLE_HEADER_LEN];
        wr_u64(&mut b, 0, self.xmin);
        wr_u64(&mut b, 8, self.xmax);
        wr_u32(&mut b, 16, self.cid);
        wr_u16(&mut b, 26, self.infomask);
        b
    }

    fn has(&self, bit: u16) -> bool {
        self.infomask & bit != 0
    }
}

/// Combo ids (spec 12.10): a tuple created and deleted inside one transaction needs two command
/// numbers and the header has one field, so it stores a combo id that this map resolves to the pair.
/// It lives until the top-level transaction ends, across savepoint rollbacks.
#[derive(Debug, Clone, Default)]
pub struct ComboMap {
    pairs: HashMap<u32, (u32, u32)>,
}

impl ComboMap {
    pub fn insert(&mut self, combo: u32, cmin: u32, cmax: u32) {
        self.pairs.insert(combo, (cmin, cmax));
    }

    /// The creating and the deleting command of a header.
    fn commands(&self, h: &TupleHeader) -> (u32, u32) {
        if h.has(infomask::CID_IS_COMBO) {
            self.pairs.get(&h.cid).copied().unwrap_or((h.cid, h.cid))
        } else {
            (h.cid, h.cid)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hint {
    None,
    Committed,
    Aborted,
}

/// *committed-before(S, X)*. `command` is the command number of the stamp, used by rule 1.
fn committed_before(s: &Snapshot, x: Xid, hint: Hint, command: u32, store: &dyn TxnStore) -> Result<bool, TxnError> {
    if let Some(own) = &s.own {
        if own.owns(x) {
            return Ok(command < own.command); // rule 1
        }
    }
    if hint == Hint::Aborted {
        return Ok(false); // final for every snapshot
    }
    if x >= s.xmax {
        return Ok(false); // rule 2
    }
    if x < s.xmin {
        // rule 3
        return Ok(match hint {
            Hint::Committed => true,
            _ => store.clog_get(x)? == ClogStatus::Committed,
        });
    }
    let csn = store.csn_get(x)?; // rule 4
    Ok(csn != 0 && csn <= s.csn)
}

/// Is the version visible to the snapshot? A commit-log or CSN-log page that fails verification is an
/// error (`XX001`), and it quarantines the store (spec 12.11).
pub fn tuple_visible(s: &Snapshot, h: &TupleHeader, combos: &ComboMap, store: &dyn TxnStore) -> Result<bool, TxnError> {
    let (cmin, cmax) = combos.commands(h);
    let xmin_hint = if h.has(infomask::XMIN_ABORTED) {
        Hint::Aborted
    } else if h.has(infomask::XMIN_COMMITTED) {
        Hint::Committed
    } else {
        Hint::None
    };
    if !committed_before(s, h.xmin, xmin_hint, cmin, store)? {
        return Ok(false);
    }
    if h.xmax == 0 || h.has(infomask::XMAX_LOCK_ONLY) {
        return Ok(true);
    }
    let xmax_hint = if h.has(infomask::XMAX_ABORTED) {
        Hint::Aborted
    } else if h.has(infomask::XMAX_COMMITTED) {
        Hint::Committed
    } else {
        Hint::None
    };
    Ok(!committed_before(s, h.xmax, xmax_hint, cmax, store)?)
}

/// *committed-before(S, X)* for an xid on its own, outside a tuple: sub-xids follow the same four
/// rules with the CSN they share with their parent. A reader's own xids have no command number here
/// and count as not committed-before.
pub fn xid_committed_before(s: &Snapshot, x: Xid, store: &dyn TxnStore) -> Result<bool, TxnError> {
    committed_before(s, x, Hint::None, u32::MAX, store)
}
