//! Visibility (spec 12.4) over synthetic tuple headers: each rule of *committed-before*, the hint
//! bits (a `COMMITTED` hint only inside rule 3, an `ABORTED` hint everywhere), lock-only stamps,
//! the reader's own changes by command number, sub-xids and combo ids.

use jamsil_page::txnrec::ClogStatus;
use jamsil_page::*;
use jamsil_txn::infomask::*;
use jamsil_txn::*;

/// The world these tests read:
///
/// | xid | clog | CSN |
/// | --- | --- | --- |
/// | 20 | committed | 100 |
/// | 21 | aborted | none |
/// | 22 | committed | 200 |
/// | 23 | in progress | none |
/// | 24 | committed | 140 |
fn world() -> MemStore {
    let s = MemStore::new();
    s.force(20, ClogStatus::Committed);
    s.force_csn(20, 100);
    s.force(21, ClogStatus::Aborted);
    s.force(22, ClogStatus::Committed);
    s.force_csn(22, 200);
    s.force(24, ClogStatus::Committed);
    s.force_csn(24, 140);
    s
}

/// Sees commits up to CSN 150; xids below 22 had finished; xids from 25 on are not assigned yet.
fn snap() -> Snapshot {
    Snapshot { csn: 150, xmin: 22, xmax: 25, applied_index: 0, own: None }
}

fn hdr(xmin: Xid, xmax: Xid, infomask: u16) -> TupleHeader {
    TupleHeader { xmin, xmax, cid: 0, infomask }
}

fn visible(s: &Snapshot, h: TupleHeader) -> bool {
    tuple_visible(s, &h, &ComboMap::default(), &world()).unwrap()
}

#[test]
fn each_rule_of_committed_before_decides_its_own_xids() {
    let (s, w) = (snap(), world());
    assert!(xid_committed_before(&s, 20, &w).unwrap(), "rule 3: below xmin, the commit log says committed");
    assert!(!xid_committed_before(&s, 21, &w).unwrap(), "rule 3: below xmin, aborted");
    assert!(!xid_committed_before(&s, 22, &w).unwrap(), "rule 4: committed, but after the snapshot (CSN 200 > 150)");
    assert!(!xid_committed_before(&s, 23, &w).unwrap(), "rule 4: still in progress, no CSN");
    assert!(xid_committed_before(&s, 24, &w).unwrap(), "rule 4: committed at CSN 140 <= 150");
    assert!(!xid_committed_before(&s, 25, &w).unwrap(), "rule 2: not assigned when the snapshot was taken");
    assert!(!xid_committed_before(&s, 99, &w).unwrap(), "rule 2");
}

#[test]
fn a_plain_tuple_is_visible_if_its_creator_committed_before_and_nobody_deleted_it() {
    let s = snap();
    assert!(visible(&s, hdr(20, 0, 0)), "created by a committed xid, never deleted");
    assert!(!visible(&s, hdr(21, 0, 0)), "created by an aborted xid");
    assert!(!visible(&s, hdr(22, 0, 0)), "created by a commit the snapshot cannot see");
    assert!(visible(&s, hdr(24, 0, 0)), "created by a commit the snapshot can see (rule 4)");
}

#[test]
fn a_delete_hides_the_tuple_only_if_the_deleter_committed_before() {
    let s = snap();
    assert!(!visible(&s, hdr(20, 24, 0)), "deleted by a commit the snapshot can see");
    assert!(visible(&s, hdr(20, 22, 0)), "deleted by a commit the snapshot cannot see yet");
    assert!(visible(&s, hdr(20, 23, 0)), "deleted by a transaction still in progress");
    assert!(visible(&s, hdr(20, 21, 0)), "deleted by an aborted transaction: the stamp is ignored");
    assert!(visible(&s, hdr(20, 24, XMAX_LOCK_ONLY)), "a lock-only stamp deletes nothing");
}

#[test]
fn a_committed_hint_never_decides_for_an_xid_at_or_above_xmin() {
    let s = snap();
    // Xid 22 committed at CSN 200, after the snapshot. The hint says only "committed", not when.
    assert!(!visible(&s, hdr(22, 0, XMIN_COMMITTED)), "rule 4 decides, not the hint");
    assert!(visible(&s, hdr(20, 22, XMAX_COMMITTED)), "the deleter's hint does not make the delete visible either");
    // Below xmin the hint short-cuts the commit-log lookup: here the log has no bit at all.
    assert!(visible(&s, hdr(10, 0, XMIN_COMMITTED)), "rule 3 with the hint");
    assert!(!visible(&s, hdr(10, 0, 0)), "and without the hint the commit log has no committed bit for xid 10");
}

#[test]
fn an_aborted_hint_is_final_for_every_snapshot_and_works_anywhere() {
    let s = snap();
    assert!(!visible(&s, hdr(24, 0, XMIN_ABORTED)), "xid 24 committed at CSN 140, but the hint says aborted: final");
    assert!(!visible(&s, hdr(20, 0, XMIN_ABORTED)), "below xmin too");
    assert!(visible(&s, hdr(20, 24, XMAX_ABORTED)), "an aborted deleter's stamp is ignored");
}

#[test]
fn the_reader_sees_its_own_changes_by_command_number() {
    let own = OwnTxn { xid: 30, sub_xids: vec![], command: 3 };
    let s = Snapshot { csn: 150, xmin: 22, xmax: 31, applied_index: 0, own: Some(own) };
    let at = |xmin, xmax, cid, im| tuple_visible(&s, &TupleHeader { xmin, xmax, cid, infomask: im }, &ComboMap::default(), &world()).unwrap();
    assert!(at(30, 0, 2, 0), "created by an earlier command of this transaction");
    assert!(!at(30, 0, 3, 0), "created by the current command: an UPDATE does not process the rows it just wrote");
    assert!(!at(30, 0, 4, 0), "created by a later command");
    assert!(!at(20, 30, 1, 0), "deleted by an earlier command of this transaction");
    assert!(at(20, 30, 3, 0), "deleted by the current command: still visible to it");
}

#[test]
fn a_tuple_created_and_deleted_in_one_transaction_resolves_through_the_combo_map() {
    let mut combos = ComboMap::default();
    combos.insert(900, 1, 3); // created by command 1, deleted by command 3
    let h = TupleHeader { xmin: 30, xmax: 30, cid: 900, infomask: CID_IS_COMBO };
    let seen_at = |command: u32| {
        let s = Snapshot { csn: 150, xmin: 22, xmax: 31, applied_index: 0, own: Some(OwnTxn { xid: 30, sub_xids: vec![], command }) };
        tuple_visible(&s, &h, &combos, &world()).unwrap()
    };
    assert!(!seen_at(1), "command 1 created it, and sees only earlier commands");
    assert!(seen_at(2), "created at 1, not yet deleted");
    assert!(seen_at(3), "deleted by the current command: still visible to it");
    assert!(!seen_at(4), "deleted by command 3");
}

#[test]
fn a_rolled_back_sub_transactions_stamp_is_ignored() {
    let w = world();
    w.force(31, ClogStatus::Aborted); // the sub-xid, rolled back
    let live = OwnTxn { xid: 30, sub_xids: vec![31], command: 5 };
    let gone = OwnTxn { xid: 30, sub_xids: vec![], command: 5 };
    let tuple = TupleHeader { xmin: 20, xmax: 31, cid: 2, infomask: 0 };
    let sees = |own: OwnTxn| tuple_visible(&Snapshot { csn: 150, xmin: 22, xmax: 32, applied_index: 0, own: Some(own) }, &tuple, &ComboMap::default(), &w).unwrap();
    assert!(!sees(live), "while the sub-transaction is live, its delete is the reader's own");
    assert!(sees(gone), "after the rollback the stamp is ignored and the tuple is visible again");
}

#[test]
fn the_tuple_header_round_trips_through_its_32_bytes() {
    let h = TupleHeader { xmin: 0x1122_3344_5566_7788, xmax: 9, cid: 0xABCD, infomask: XMIN_COMMITTED | HAS_NULL };
    let b = h.encode();
    assert_eq!(b.len(), 32);
    assert_eq!(rd_u64(&b, 0), h.xmin);
    assert_eq!(rd_u64(&b, 8), 9);
    assert_eq!(rd_u32(&b, 16), 0xABCD);
    assert_eq!(rd_u16(&b, 26), h.infomask);
    assert_eq!(TupleHeader::decode(&b).unwrap(), h);
    assert!(TupleHeader::decode(&b[..31]).is_err());
}
