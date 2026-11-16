//! Payloads of the `HEAP` resource manager (spec 10.3). Main payloads are packed little-endian; a slot is
//! the 1-based slot number of 6.2. The decoders are strict like those of the transaction records: the
//! length must match exactly, reserved bytes and unknown flag bits are zero, and the encodings are
//! canonical, so `encode(decode(bytes)) == bytes` (the fuzz target `heap_record` asserts it).

use crate::*;

/// Bit 0 of `INSERT`, `MULTI_INSERT`, `DELETE`, `LOCK` and `VISIBLE`'s companions: the page's
/// `ALL_VISIBLE` flag and the visibility-map bit are cleared by the same record.
pub const FLAG_CLEAR_ALL_VISIBLE: u8 = 1;
/// `UPDATE` flags: clear the old page's visibility-map bit, clear the new page's, and "both versions are
/// on one page".
pub const UPD_CLEAR_OLD: u8 = 1;
pub const UPD_CLEAR_NEW: u8 = 2;
pub const UPD_SAME_PAGE: u8 = 4;

fn slot_ok(s: u16) -> Result<u16, Corrupt> {
    if s == 0 {
        return corrupt("slot 0: slots are counted from 1");
    }
    Ok(s)
}

// ---- INSERT and MULTI_INSERT: slot or count, flags, one zero byte ---------------------------------

pub const SMALL_MAIN_LEN: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsertMain {
    pub slot: u16,
    pub clear_all_visible: bool,
}

impl InsertMain {
    pub fn encode(&self) -> [u8; SMALL_MAIN_LEN] {
        let mut b = [0u8; SMALL_MAIN_LEN];
        wr_u16(&mut b, 0, self.slot);
        b[2] = if self.clear_all_visible { FLAG_CLEAR_ALL_VISIBLE } else { 0 };
        b
    }
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() != SMALL_MAIN_LEN {
            return corrupt("INSERT payload is not 4 bytes");
        }
        if b[2] & !FLAG_CLEAR_ALL_VISIBLE != 0 || b[3] != 0 {
            return corrupt("INSERT: unknown flag bits or a non-zero reserved byte");
        }
        Ok(Self { slot: slot_ok(rd_u16(b, 0))?, clear_all_visible: b[2] & FLAG_CLEAR_ALL_VISIBLE != 0 })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiInsertMain {
    pub n: u16,
    pub clear_all_visible: bool,
}

impl MultiInsertMain {
    pub fn encode(&self) -> [u8; SMALL_MAIN_LEN] {
        let mut b = [0u8; SMALL_MAIN_LEN];
        wr_u16(&mut b, 0, self.n);
        b[2] = if self.clear_all_visible { FLAG_CLEAR_ALL_VISIBLE } else { 0 };
        b
    }
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() != SMALL_MAIN_LEN {
            return corrupt("MULTI_INSERT payload is not 4 bytes");
        }
        if b[2] & !FLAG_CLEAR_ALL_VISIBLE != 0 || b[3] != 0 {
            return corrupt("MULTI_INSERT: unknown flag bits or a non-zero reserved byte");
        }
        let n = rd_u16(b, 0);
        if n == 0 {
            return corrupt("MULTI_INSERT of no tuples");
        }
        Ok(Self { n, clear_all_visible: b[2] & FLAG_CLEAR_ALL_VISIBLE != 0 })
    }
}

/// The data of the heap page's block reference of a `MULTI_INSERT`: `n` times (`slot` u16, `len` u16,
/// tuple). Slots are ascending, and every tuple has at least a header's length.
pub fn encode_multi_insert_data(tuples: &[(u16, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for (slot, t) in tuples {
        out.extend_from_slice(&slot.to_le_bytes());
        out.extend_from_slice(&(t.len() as u16).to_le_bytes());
        out.extend_from_slice(t);
    }
    out
}

pub fn decode_multi_insert_data(b: &[u8], n: u16) -> Result<Vec<(u16, &[u8])>, Corrupt> {
    let mut out = Vec::with_capacity((n as usize).min(1024));
    let mut pos = 0usize;
    let mut last = 0u16;
    for _ in 0..n {
        let Some(h) = b.get(pos..pos + 4) else { return corrupt("MULTI_INSERT data ends inside an item header") };
        let (slot, len) = (slot_ok(rd_u16(h, 0))?, rd_u16(h, 2) as usize);
        if slot <= last {
            return corrupt("MULTI_INSERT slots are not ascending");
        }
        if len < crate::tuple::TUPLE_HEADER_LEN {
            return corrupt("MULTI_INSERT tuple shorter than its header");
        }
        let Some(t) = b.get(pos + 4..pos + 4 + len) else { return corrupt("MULTI_INSERT data ends inside a tuple") };
        out.push((slot, t));
        last = slot;
        pos += 4 + len;
    }
    if pos != b.len() {
        return corrupt("MULTI_INSERT data has bytes after its last tuple");
    }
    Ok(out)
}

// ---- DELETE and LOCK: 28 bytes ----------------------------------------------------------------------

pub const DELETE_MAIN_LEN: usize = 28;

/// `DELETE` stamps `xmax` and `cid` on a version, and `LOCK` stamps `xmax` with `XMAX_LOCK_ONLY` (spec
/// 10.3). `infomask` is the whole new value of the tuple's `infomask`, so a delete after the transaction's
/// own `FOR UPDATE` can clear `XMAX_LOCK_ONLY`. `prune_xid` 0 leaves the page's value unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeleteMain {
    pub slot: u16,
    pub clear_all_visible: bool,
    pub xmax: Xid,
    pub cid: u32,
    pub infomask: u16,
    pub prune_xid: Xid,
}

impl DeleteMain {
    pub fn encode(&self) -> [u8; DELETE_MAIN_LEN] {
        let mut b = [0u8; DELETE_MAIN_LEN];
        wr_u16(&mut b, 0, self.slot);
        b[2] = if self.clear_all_visible { FLAG_CLEAR_ALL_VISIBLE } else { 0 };
        wr_u64(&mut b, 4, self.xmax);
        wr_u32(&mut b, 12, self.cid);
        wr_u16(&mut b, 16, self.infomask);
        wr_u64(&mut b, 20, self.prune_xid);
        b
    }
    /// `lock` is true for a `LOCK` record: `XMAX_LOCK_ONLY` must be set and `prune_xid` is zero.
    pub fn decode(b: &[u8], lock: bool) -> Result<Self, Corrupt> {
        if b.len() != DELETE_MAIN_LEN {
            return corrupt("DELETE or LOCK payload is not 28 bytes");
        }
        if b[2] & !FLAG_CLEAR_ALL_VISIBLE != 0 || b[3] != 0 || b[18] != 0 || b[19] != 0 {
            return corrupt("DELETE: unknown flag bits or a non-zero reserved byte");
        }
        let m = Self {
            slot: slot_ok(rd_u16(b, 0))?,
            clear_all_visible: b[2] & FLAG_CLEAR_ALL_VISIBLE != 0,
            xmax: rd_u64(b, 4),
            cid: rd_u32(b, 12),
            infomask: rd_u16(b, 16),
            prune_xid: rd_u64(b, 20),
        };
        if m.infomask & !crate::tuple::infomask::ALL != 0 {
            return corrupt("DELETE: infomask bits 10 to 15 are set");
        }
        if m.xmax == 0 {
            return corrupt("DELETE or LOCK with xmax 0");
        }
        if lock && (m.infomask & crate::tuple::infomask::XMAX_LOCK_ONLY == 0 || m.prune_xid != 0 || m.clear_all_visible) {
            return corrupt("LOCK: XMAX_LOCK_ONLY must be set; prune_xid is zero, and a lock does not clear the visibility bit");
        }
        Ok(m)
    }
}

// ---- UPDATE and HOT_UPDATE: 28 bytes --------------------------------------------------------------

pub const UPDATE_MAIN_LEN: usize = 28;

/// One record for the delete of the old version and the insert of the new one (spec 10.3). The old
/// version gets `xmax`, `cid` and `old_infomask` (the whole new value); the new tuple is the data of its
/// block reference, which is the old page's when `same_page`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateMain {
    pub old_slot: u16,
    pub new_slot: u16,
    pub clear_old: bool,
    pub clear_new: bool,
    pub same_page: bool,
    pub old_infomask: u16,
    pub xmax: Xid,
    pub cid: u32,
    pub prune_xid: Xid,
}

impl UpdateMain {
    pub fn encode(&self) -> [u8; UPDATE_MAIN_LEN] {
        let mut b = [0u8; UPDATE_MAIN_LEN];
        wr_u16(&mut b, 0, self.old_slot);
        wr_u16(&mut b, 2, self.new_slot);
        b[4] = (if self.clear_old { UPD_CLEAR_OLD } else { 0 }) | (if self.clear_new { UPD_CLEAR_NEW } else { 0 }) | (if self.same_page { UPD_SAME_PAGE } else { 0 });
        wr_u16(&mut b, 6, self.old_infomask);
        wr_u64(&mut b, 8, self.xmax);
        wr_u32(&mut b, 16, self.cid);
        wr_u64(&mut b, 20, self.prune_xid);
        b
    }
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() != UPDATE_MAIN_LEN {
            return corrupt("UPDATE payload is not 28 bytes");
        }
        if b[4] & !(UPD_CLEAR_OLD | UPD_CLEAR_NEW | UPD_SAME_PAGE) != 0 || b[5] != 0 {
            return corrupt("UPDATE: unknown flag bits or a non-zero reserved byte");
        }
        let m = Self {
            old_slot: slot_ok(rd_u16(b, 0))?,
            new_slot: slot_ok(rd_u16(b, 2))?,
            clear_old: b[4] & UPD_CLEAR_OLD != 0,
            clear_new: b[4] & UPD_CLEAR_NEW != 0,
            same_page: b[4] & UPD_SAME_PAGE != 0,
            old_infomask: rd_u16(b, 6),
            xmax: rd_u64(b, 8),
            cid: rd_u32(b, 16),
            prune_xid: rd_u64(b, 20),
        };
        if m.old_infomask & !crate::tuple::infomask::ALL != 0 {
            return corrupt("UPDATE: infomask bits 10 to 15 are set");
        }
        if m.xmax == 0 {
            return corrupt("UPDATE with xmax 0");
        }
        if m.same_page && m.old_slot == m.new_slot {
            return corrupt("UPDATE: both versions in one slot");
        }
        if m.same_page && m.clear_old != m.clear_new {
            return corrupt("UPDATE on one page clears its bit for both versions or for neither");
        }
        Ok(m)
    }

    /// How many visibility-map pages the record references: none, or one, or two. When both heap pages
    /// share one map page and both bits are cleared the record carries one reference to it (spec 10.3);
    /// the caller passes whether they share.
    pub fn vm_refs(&self, share_one_map_page: bool) -> usize {
        match (self.clear_old, self.clear_new) {
            (false, false) => 0,
            (true, true) if self.same_page || share_one_map_page => 1,
            (true, true) => 2,
            _ => 1,
        }
    }
}

// ---- PRUNE ------------------------------------------------------------------------------------------

pub const PRUNE_FIXED_LEN: usize = 16;

/// One `HEAP_PRUNE` record for a page: redirects, dead slots (no data, kept for the indexes) and unused
/// slots, then compaction (spec 10.4). `prune_xid` is the page's new value.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneMain {
    pub prune_xid: Xid,
    pub redirects: Vec<(u16, u16)>,
    pub dead: Vec<u16>,
    pub unused: Vec<u16>,
}

impl PruneMain {
    pub fn encoded_len(&self) -> usize {
        PRUNE_FIXED_LEN + 4 * self.redirects.len() + 2 * (self.dead.len() + self.unused.len())
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut b = vec![0u8; PRUNE_FIXED_LEN];
        wr_u16(&mut b, 0, self.redirects.len() as u16);
        wr_u16(&mut b, 2, self.dead.len() as u16);
        wr_u16(&mut b, 4, self.unused.len() as u16);
        wr_u64(&mut b, 8, self.prune_xid);
        for (from, to) in &self.redirects {
            b.extend_from_slice(&from.to_le_bytes());
            b.extend_from_slice(&to.to_le_bytes());
        }
        for s in self.dead.iter().chain(&self.unused) {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b
    }
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() < PRUNE_FIXED_LEN {
            return corrupt("PRUNE payload shorter than its fixed part");
        }
        if b[6] != 0 || b[7] != 0 {
            return corrupt("PRUNE: reserved bytes are not zero");
        }
        let (nr, nd, nu) = (rd_u16(b, 0) as usize, rd_u16(b, 2) as usize, rd_u16(b, 4) as usize);
        if b.len() != PRUNE_FIXED_LEN + 4 * nr + 2 * (nd + nu) {
            return corrupt("PRUNE payload length does not match its counts");
        }
        let mut pos = PRUNE_FIXED_LEN;
        let u16s = |n: usize, pos: &mut usize| -> Result<Vec<u16>, Corrupt> {
            (0..n)
                .map(|_| {
                    let v = rd_u16(b, *pos);
                    *pos += 2;
                    slot_ok(v)
                })
                .collect()
        };
        let mut redirects = Vec::with_capacity(nr);
        for _ in 0..nr {
            let (from, to) = (slot_ok(rd_u16(b, pos))?, slot_ok(rd_u16(b, pos + 2))?);
            if from == to {
                return corrupt("PRUNE: a slot redirects to itself");
            }
            redirects.push((from, to));
            pos += 4;
        }
        let dead = u16s(nd, &mut pos)?;
        let unused = u16s(nu, &mut pos)?;
        Ok(Self { prune_xid: rd_u64(b, 8), redirects, dead, unused })
    }
}

// ---- VISIBLE ----------------------------------------------------------------------------------------

pub const VISIBLE_MAIN_LEN: usize = 8;

/// `VISIBLE`: vacuum sets a page's `ALL_VISIBLE` flag and its visibility-map bit. `flags` is zero in 1.0.
pub fn encode_visible() -> [u8; VISIBLE_MAIN_LEN] {
    [0u8; VISIBLE_MAIN_LEN]
}

pub fn decode_visible(b: &[u8]) -> Result<(), Corrupt> {
    if b.len() != VISIBLE_MAIN_LEN || b.iter().any(|&x| x != 0) {
        return corrupt("VISIBLE payload is not 8 zero bytes");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::infomask;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn slot(&mut self) -> u16 {
            (self.next() % 65_535) as u16 + 1
        }
    }

    #[test]
    fn the_layouts_are_those_of_the_specification() {
        let d = DeleteMain { slot: 0x11, clear_all_visible: true, xmax: 0x22, cid: 0x33, infomask: infomask::XMAX_LOCK_ONLY, prune_xid: 0x44 }.encode();
        assert_eq!(d.len(), 28);
        assert_eq!((rd_u16(&d, 0), d[2], d[3]), (0x11, 1, 0));
        assert_eq!((rd_u64(&d, 4), rd_u32(&d, 12), rd_u16(&d, 16), rd_u16(&d, 18), rd_u64(&d, 20)), (0x22, 0x33, infomask::XMAX_LOCK_ONLY, 0, 0x44));
        let u = UpdateMain { old_slot: 1, new_slot: 2, clear_old: true, clear_new: false, same_page: true, old_infomask: 0x0A, xmax: 0x55, cid: 0x66, prune_xid: 0x77 };
        let b = u.encode();
        assert_eq!(b.len(), 28);
        assert_eq!((rd_u16(&b, 0), rd_u16(&b, 2), b[4], b[5], rd_u16(&b, 6)), (1, 2, 0b101, 0, 0x0A), "old_infomask replaces new_tid_block");
        assert_eq!((rd_u64(&b, 8), rd_u32(&b, 16), rd_u64(&b, 20)), (0x55, 0x66, 0x77));
        let p = PruneMain { prune_xid: 9, redirects: vec![(1, 2)], dead: vec![3, 4], unused: vec![5] }.encode();
        assert_eq!(p.len(), 16 + 4 + 4 + 2);
        assert_eq!((rd_u16(&p, 0), rd_u16(&p, 2), rd_u16(&p, 4), rd_u64(&p, 8)), (1, 2, 1, 9));
        assert_eq!(InsertMain { slot: 7, clear_all_visible: false }.encode(), [7, 0, 0, 0]);
    }

    #[test]
    fn every_record_round_trips() {
        let mut r = Rng(1);
        for _ in 0..2000 {
            let i = InsertMain { slot: r.slot(), clear_all_visible: r.next() % 2 == 0 };
            assert_eq!(InsertMain::decode(&i.encode()).unwrap(), i);
            let m = MultiInsertMain { n: r.slot(), clear_all_visible: r.next() % 2 == 0 };
            assert_eq!(MultiInsertMain::decode(&m.encode()).unwrap(), m);
            let d = DeleteMain { slot: r.slot(), clear_all_visible: r.next() % 2 == 0, xmax: r.next() | 1, cid: r.next() as u32, infomask: (r.next() as u16) & infomask::ALL, prune_xid: r.next() };
            assert_eq!(DeleteMain::decode(&d.encode(), false).unwrap(), d);
            let l = DeleteMain { infomask: d.infomask | infomask::XMAX_LOCK_ONLY, prune_xid: 0, clear_all_visible: false, ..d };
            assert_eq!(DeleteMain::decode(&l.encode(), true).unwrap(), l);
            let (a, b) = (r.slot(), r.slot());
            let same = r.next() % 2 == 0;
            let co = r.next() % 2 == 0;
            let u = UpdateMain {
                old_slot: a,
                new_slot: if a == b { b.wrapping_add(1).max(1) } else { b },
                clear_old: co,
                clear_new: if same { co } else { r.next() % 2 == 0 },
                same_page: same,
                old_infomask: (r.next() as u16) & infomask::ALL,
                xmax: r.next() | 1,
                cid: r.next() as u32,
                prune_xid: r.next(),
            };
            if u.old_slot != u.new_slot {
                assert_eq!(UpdateMain::decode(&u.encode()).unwrap(), u);
            }
            let p = PruneMain {
                prune_xid: r.next(),
                redirects: (0..r.next() % 4).map(|_| (r.slot(), r.slot())).filter(|(a, b)| a != b).collect(),
                dead: (0..r.next() % 5).map(|_| r.slot()).collect(),
                unused: (0..r.next() % 5).map(|_| r.slot()).collect(),
            };
            assert_eq!(PruneMain::decode(&p.encode()).unwrap(), p);
            assert_eq!(p.encode().len(), p.encoded_len());
        }
        decode_visible(&encode_visible()).unwrap();
    }

    #[test]
    fn the_decoders_are_strict() {
        let good = InsertMain { slot: 3, clear_all_visible: true }.encode();
        assert!(InsertMain::decode(&good).is_ok());
        assert!(InsertMain::decode(&good[..3]).is_err());
        assert!(InsertMain::decode(&[0, 0, 0, 0]).is_err(), "slot 0");
        assert!(InsertMain::decode(&[3, 0, 2, 0]).is_err(), "an unknown flag");
        assert!(InsertMain::decode(&[3, 0, 0, 1]).is_err(), "the reserved byte");
        assert!(MultiInsertMain::decode(&[0, 0, 0, 0]).is_err(), "no tuples");

        let d = DeleteMain { slot: 3, clear_all_visible: false, xmax: 5, cid: 1, infomask: 0, prune_xid: 0 };
        let mut b = d.encode();
        assert!(DeleteMain::decode(&b, false).is_ok());
        b[18] = 1;
        assert!(DeleteMain::decode(&b, false).is_err(), "the two zero bytes");
        let mut b = d.encode();
        wr_u16(&mut b, 16, 1 << 12);
        assert!(DeleteMain::decode(&b, false).is_err(), "infomask bit 12");
        assert!(DeleteMain { xmax: 0, ..d }.encode().len() == 28 && DeleteMain::decode(&DeleteMain { xmax: 0, ..d }.encode(), false).is_err(), "xmax 0");
        assert!(DeleteMain::decode(&d.encode(), true).is_err(), "a LOCK without XMAX_LOCK_ONLY");
        let lock = DeleteMain { infomask: infomask::XMAX_LOCK_ONLY, ..d };
        assert!(DeleteMain::decode(&lock.encode(), true).is_ok());
        assert!(DeleteMain::decode(&DeleteMain { prune_xid: 9, ..lock }.encode(), true).is_err(), "a LOCK has prune_xid zero");

        let u = UpdateMain { old_slot: 1, new_slot: 2, clear_old: false, clear_new: false, same_page: false, old_infomask: 0, xmax: 4, cid: 0, prune_xid: 0 };
        let mut b = u.encode();
        assert!(UpdateMain::decode(&b).is_ok());
        b[4] = 8;
        assert!(UpdateMain::decode(&b).is_err(), "an unknown flag");
        let mut b = u.encode();
        b[5] = 1;
        assert!(UpdateMain::decode(&b).is_err());
        assert!(UpdateMain::decode(&UpdateMain { same_page: true, new_slot: 1, ..u }.encode()).is_err(), "one slot twice");
        assert!(UpdateMain::decode(&UpdateMain { same_page: true, clear_old: true, ..u }.encode()).is_err(), "one page, one bit for two versions");

        let p = PruneMain { prune_xid: 1, redirects: vec![(1, 2)], dead: vec![3], unused: vec![4] }.encode();
        assert!(PruneMain::decode(&p).is_ok());
        assert!(PruneMain::decode(&p[..p.len() - 1]).is_err());
        assert!(PruneMain::decode(&[&p[..], &[0][..]].concat()).is_err());
        let mut q = p.clone();
        q[6] = 1;
        assert!(PruneMain::decode(&q).is_err());
        let mut self_redirect = p.clone();
        wr_u16(&mut self_redirect, 18, 1);
        assert!(PruneMain::decode(&self_redirect).is_err(), "a slot redirecting to itself");
        let mut huge = p.clone();
        wr_u16(&mut huge, 0, u16::MAX);
        assert!(PruneMain::decode(&huge).is_err(), "counts that do not fit the length");
        assert!(decode_visible(&[0; 7]).is_err() && decode_visible(&[1, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn multi_insert_data_round_trips_and_is_strict() {
        let t1 = vec![1u8; 40];
        let t2 = vec![2u8; 100];
        let data = encode_multi_insert_data(&[(1, &t1), (5, &t2)]);
        let items = decode_multi_insert_data(&data, 2).unwrap();
        assert_eq!(items, vec![(1u16, &t1[..]), (5u16, &t2[..])]);
        assert!(decode_multi_insert_data(&data, 3).is_err(), "n larger than the data");
        assert!(decode_multi_insert_data(&data, 1).is_err(), "bytes after the last tuple");
        let desc = encode_multi_insert_data(&[(5, &t1), (1, &t2)]);
        assert!(decode_multi_insert_data(&desc, 2).is_err(), "slots must ascend");
        let short = encode_multi_insert_data(&[(1, &t1[..10])]);
        assert!(decode_multi_insert_data(&short, 1).is_err(), "a tuple shorter than its header");
    }

    #[test]
    fn an_update_carries_one_reference_for_a_map_page_both_pages_share() {
        let base = UpdateMain { old_slot: 1, new_slot: 2, clear_old: true, clear_new: true, same_page: false, old_infomask: 0, xmax: 4, cid: 0, prune_xid: 0 };
        assert_eq!(base.vm_refs(false), 2);
        assert_eq!(base.vm_refs(true), 1, "both pages share one map page");
        assert_eq!(UpdateMain { same_page: true, ..base }.vm_refs(false), 1);
        assert_eq!(UpdateMain { clear_new: false, ..base }.vm_refs(false), 1);
        assert_eq!(UpdateMain { clear_new: false, clear_old: false, ..base }.vm_refs(false), 0);
    }
}
