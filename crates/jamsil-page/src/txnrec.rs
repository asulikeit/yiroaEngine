//! Transaction records and the pages of the commit log and the CSN log (spec 8.7, 12.2).
//!
//! The record decoders are strict, as 8.7 says: the length must match the counts exactly, unknown
//! flag bits must be zero, and a field whose flag is clear must be zero. Anything else is
//! `Corrupt`. They never panic and never overflow, whatever the counts say. Every encoding is
//! canonical, so `encode(decode(bytes)) == bytes` for every input that decodes; the fuzz target
//! (`txn_record`) asserts it.

use crate::page::{PageKind, PageMut, PageRef};
use crate::*;

/// The position of a replicated entry in the Raft log (spec 12.11).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct RaftPos {
    pub index: u64,
    pub term: u64,
}

// ---- Commit record (spec 8.7) -----------------------------------------------------------------

pub const COMMIT_FIXED_LEN: usize = 44;
/// Bit 0: `raft_index` and `raft_term` are valid. A commit without it is node-local.
pub const COMMIT_HAS_RAFT: u32 = 1;
/// Bit 1: `schema_version` is valid.
pub const COMMIT_HAS_SCHEMA: u32 = 2;
const COMMIT_KNOWN_FLAGS: u32 = COMMIT_HAS_RAFT | COMMIT_HAS_SCHEMA;

/// Bytes of an array of `n` eight-byte items, or `None` if it does not fit in memory.
fn array_bytes(n: u32) -> Option<usize> {
    (n as usize).checked_mul(8)
}

/// The commit record, owned. `encode` and `CommitView::parse` are inverse.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommitRecord {
    /// `None` for a node-local commit (outbox trimming, statistics): redo does not move the applied
    /// position for it.
    pub raft: Option<RaftPos>,
    /// Microseconds since the Unix epoch, from `Clock`.
    pub commit_time: u64,
    /// `Some` if the transaction changed the schema.
    pub schema_version: Option<u64>,
    pub sub_xids: Vec<Xid>,
    pub unlink: Vec<RelFileId>,
}

impl CommitRecord {
    /// `44 + 8 * (n_sub + n_unlink)` bytes.
    pub fn encoded_len(&self) -> usize {
        COMMIT_FIXED_LEN + 8 * (self.sub_xids.len() + self.unlink.len())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        self.encode_into(&mut out);
        out
    }

    pub fn encode_into(&self, out: &mut Vec<u8>) {
        let mut flags = 0u32;
        if self.raft.is_some() {
            flags |= COMMIT_HAS_RAFT;
        }
        if self.schema_version.is_some() {
            flags |= COMMIT_HAS_SCHEMA;
        }
        let raft = self.raft.unwrap_or_default();
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&raft.index.to_le_bytes());
        out.extend_from_slice(&raft.term.to_le_bytes());
        out.extend_from_slice(&self.commit_time.to_le_bytes());
        out.extend_from_slice(&self.schema_version.unwrap_or(0).to_le_bytes());
        out.extend_from_slice(&(self.sub_xids.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.unlink.len() as u32).to_le_bytes());
        for x in &self.sub_xids {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for r in &self.unlink {
            out.extend_from_slice(&r.to_le_bytes());
        }
    }

    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        let v = CommitView::parse(b)?;
        Ok(Self {
            raft: v.raft(),
            commit_time: v.commit_time(),
            schema_version: v.schema_version(),
            sub_xids: v.sub_xids().collect(),
            unlink: v.unlink().collect(),
        })
    }
}

/// A commit record validated in place: redo reads it without copying the arrays.
#[derive(Debug, Clone, Copy)]
pub struct CommitView<'a> {
    b: &'a [u8],
    n_sub: usize,
    n_unlink: usize,
}

impl<'a> CommitView<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Self, Corrupt> {
        if b.len() < COMMIT_FIXED_LEN {
            return corrupt("commit record shorter than its fixed part");
        }
        let flags = rd_u32(b, 0);
        if flags & !COMMIT_KNOWN_FLAGS != 0 {
            return corrupt("commit record has unknown flag bits");
        }
        if flags & COMMIT_HAS_RAFT == 0 && (rd_u64(b, 4) != 0 || rd_u64(b, 12) != 0) {
            return corrupt("commit record has a Raft position but no HAS_RAFT flag");
        }
        if flags & COMMIT_HAS_SCHEMA == 0 && rd_u64(b, 28) != 0 {
            return corrupt("commit record has a schema version but no HAS_SCHEMA flag");
        }
        let (n_sub, n_unlink) = (rd_u32(b, 36), rd_u32(b, 40));
        // One comparison, before any array is touched: both counts are at fixed offsets.
        let want = array_bytes(n_sub)
            .and_then(|a| array_bytes(n_unlink).and_then(|c| a.checked_add(c)))
            .and_then(|arrays| arrays.checked_add(COMMIT_FIXED_LEN));
        if want != Some(b.len()) {
            return corrupt("commit record length does not match its counts");
        }
        Ok(Self { b, n_sub: n_sub as usize, n_unlink: n_unlink as usize })
    }

    pub fn raft(&self) -> Option<RaftPos> {
        (rd_u32(self.b, 0) & COMMIT_HAS_RAFT != 0).then(|| RaftPos { index: rd_u64(self.b, 4), term: rd_u64(self.b, 12) })
    }
    pub fn commit_time(&self) -> u64 {
        rd_u64(self.b, 20)
    }
    pub fn schema_version(&self) -> Option<u64> {
        (rd_u32(self.b, 0) & COMMIT_HAS_SCHEMA != 0).then(|| rd_u64(self.b, 28))
    }
    pub fn n_sub(&self) -> usize {
        self.n_sub
    }
    pub fn n_unlink(&self) -> usize {
        self.n_unlink
    }
    pub fn sub_xids(&self) -> impl Iterator<Item = Xid> + 'a {
        let (b, n) = (self.b, self.n_sub);
        (0..n).map(move |i| rd_u64(b, COMMIT_FIXED_LEN + 8 * i))
    }
    pub fn unlink(&self) -> impl Iterator<Item = RelFileId> + 'a {
        let (b, start, n) = (self.b, COMMIT_FIXED_LEN + 8 * self.n_sub, self.n_unlink);
        (0..n).map(move |i| rd_u64(b, start + 8 * i))
    }
}

// ---- Abort record (spec 8.7) ------------------------------------------------------------------

pub const ABORT_FIXED_LEN: usize = 8;

/// `8 + 8 * (n_sub + n_created)` bytes: both counts first, then `sub_xids` and `created`, the
/// relation files the transaction created, which an abort removes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbortRecord {
    pub sub_xids: Vec<Xid>,
    pub created: Vec<RelFileId>,
}

impl AbortRecord {
    pub fn encoded_len(&self) -> usize {
        ABORT_FIXED_LEN + 8 * (self.sub_xids.len() + self.created.len())
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&(self.sub_xids.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.created.len() as u32).to_le_bytes());
        for x in &self.sub_xids {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for r in &self.created {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out
    }

    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() < ABORT_FIXED_LEN {
            return corrupt("abort record shorter than its fixed part");
        }
        let (n_sub, n_created) = (rd_u32(b, 0), rd_u32(b, 4));
        let want = array_bytes(n_sub)
            .and_then(|a| array_bytes(n_created).and_then(|c| a.checked_add(c)))
            .and_then(|arrays| arrays.checked_add(ABORT_FIXED_LEN));
        if want != Some(b.len()) {
            return corrupt("abort record length does not match its counts");
        }
        let (n_sub, n_created) = (n_sub as usize, n_created as usize);
        Ok(Self {
            sub_xids: (0..n_sub).map(|i| rd_u64(b, ABORT_FIXED_LEN + 8 * i)).collect(),
            created: (0..n_created).map(|i| rd_u64(b, ABORT_FIXED_LEN + 8 * (n_sub + i))).collect(),
        })
    }
}

// ---- Applied-advance record (spec 8.7) --------------------------------------------------------

pub const APPLIED_ADVANCE_LEN: usize = 16;

/// 16 bytes: `raft_index` u64, `raft_term` u64. Used for Raft no-ops and for entries of
/// transactions that wrote nothing.
pub fn encode_applied_advance(pos: RaftPos) -> [u8; APPLIED_ADVANCE_LEN] {
    let mut out = [0u8; APPLIED_ADVANCE_LEN];
    wr_u64(&mut out, 0, pos.index);
    wr_u64(&mut out, 8, pos.term);
    out
}

pub fn decode_applied_advance(b: &[u8]) -> Result<RaftPos, Corrupt> {
    if b.len() != APPLIED_ADVANCE_LEN {
        return corrupt("applied-advance record is not 16 bytes");
    }
    Ok(RaftPos { index: rd_u64(b, 0), term: rd_u64(b, 8) })
}

// ---- Allocation batch and commit-log truncate (spec 8.6) --------------------------------------

pub const ALLOC_BATCH_LEN: usize = 24;

/// The allocation batch (`XLOG`): three high-water marks, each the first id the batch does not
/// cover. Redo raises each counter to the recorded value if it is lower. Oids are 32-bit, so the
/// upper half of the `next_oid` field is zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocBatch {
    pub next_xid: Xid,
    pub next_oid: u32,
    pub next_relfile: RelFileId,
}

impl AllocBatch {
    pub fn encode(&self) -> [u8; ALLOC_BATCH_LEN] {
        let mut out = [0u8; ALLOC_BATCH_LEN];
        wr_u64(&mut out, 0, self.next_xid);
        wr_u64(&mut out, 8, u64::from(self.next_oid));
        wr_u64(&mut out, 16, self.next_relfile);
        out
    }

    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() != ALLOC_BATCH_LEN {
            return corrupt("allocation batch is not 24 bytes");
        }
        let oid = rd_u64(b, 8);
        let Ok(next_oid) = u32::try_from(oid) else {
            return corrupt("allocation batch: the upper half of next_oid is not zero");
        };
        Ok(Self { next_xid: rd_u64(b, 0), next_oid, next_relfile: rd_u64(b, 16) })
    }

    /// Redo: raises each counter to the recorded value if it is lower.
    pub fn raise(&mut self, other: AllocBatch) {
        self.next_xid = self.next_xid.max(other.next_xid);
        self.next_oid = self.next_oid.max(other.next_oid);
        self.next_relfile = self.next_relfile.max(other.next_relfile);
    }
}

pub const CLOG_TRUNCATE_LEN: usize = 8;

/// The `CLOG` truncate record: 8 bytes, `oldest_xid`. Redo unlinks the segments wholly below it.
pub fn encode_clog_truncate(oldest_xid: Xid) -> [u8; CLOG_TRUNCATE_LEN] {
    oldest_xid.to_le_bytes()
}

pub fn decode_clog_truncate(b: &[u8]) -> Result<Xid, Corrupt> {
    if b.len() != CLOG_TRUNCATE_LEN {
        return corrupt("commit-log truncate record is not 8 bytes");
    }
    Ok(rd_u64(b, 0))
}

// ---- Commit-log pages (spec 12.2) -------------------------------------------------------------

/// 2 bits per xid after the 32-byte header: `(8192 - 32) * 4`.
pub const XIDS_PER_CLOG_PAGE: u64 = 32_640;
/// 8 bytes per entry after the header: `(8192 - 32) / 8`.
pub const CSNS_PER_PAGE: u64 = 1_020;

/// How a transaction ended, as the commit log records it. The value 3 is reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ClogStatus {
    InProgress = 0,
    Committed = 1,
    Aborted = 2,
}

impl ClogStatus {
    fn from_bits(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::InProgress),
            1 => Some(Self::Committed),
            2 => Some(Self::Aborted),
            _ => None,
        }
    }
}

/// Block of the commit log that holds `xid`, or `None` if the xid lies beyond what a 32-bit block
/// number reaches (no real xid does).
pub fn clog_block(xid: Xid) -> Option<BlockNo> {
    u32::try_from(xid / XIDS_PER_CLOG_PAGE).ok()
}

/// Byte and bit shift of `xid` within its page: byte `32 + (X % 32640) / 4`, bit `2 * (X % 4)`.
pub fn clog_slot(xid: Xid) -> (usize, u32) {
    let r = xid % XIDS_PER_CLOG_PAGE;
    (page::HEADER_SIZE + (r / 4) as usize, 2 * (r % 4) as u32)
}

/// Formats an empty commit-log page: every xid in progress.
pub fn init_clog_page(page: &mut [u8]) {
    PageMut(page).init(PageKind::Clog, 0);
}

/// The status of `xid` on the commit-log page that holds it. `Corrupt` if the page is not a
/// commit-log page or the bits hold the reserved value.
pub fn clog_get(page: &[u8], xid: Xid) -> Result<ClogStatus, Corrupt> {
    if PageRef(page).kind() != PageKind::Clog {
        return corrupt("not a commit-log page");
    }
    let (byte, shift) = clog_slot(xid);
    ClogStatus::from_bits((page[byte] >> shift) & 3).ok_or_else(|| Corrupt("commit log holds the reserved value".into()))
}

/// Sets the status of `xid`. The caller holds the page latch exclusively.
pub fn clog_set(page: &mut [u8], xid: Xid, status: ClogStatus) {
    let (byte, shift) = clog_slot(xid);
    page[byte] = (page[byte] & !(3 << shift)) | ((status as u8) << shift);
}

// ---- CSN-log pages (spec 12.2) ----------------------------------------------------------------

/// Block of the CSN log that holds `xid`.
pub fn csn_block(xid: Xid) -> Option<BlockNo> {
    u32::try_from(xid / CSNS_PER_PAGE).ok()
}

/// Byte offset of `xid` within its page: `32 + 8 * (X % 1020)`.
pub fn csn_slot(xid: Xid) -> usize {
    page::HEADER_SIZE + 8 * (xid % CSNS_PER_PAGE) as usize
}

pub fn init_csn_page(page: &mut [u8]) {
    PageMut(page).init(PageKind::CsnLog, 0);
}

/// The commit CSN of `xid`, or 0 if none was written.
pub fn csn_get(page: &[u8], xid: Xid) -> Result<Csn, Corrupt> {
    if PageRef(page).kind() != PageKind::CsnLog {
        return corrupt("not a CSN-log page");
    }
    Ok(rd_u64(page, csn_slot(xid)))
}

pub fn csn_set(page: &mut [u8], xid: Xid, csn: Csn) {
    wr_u64(page, csn_slot(xid), csn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::new_page_buf;

    /// A small deterministic generator, so the property tests need no extra dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_commit(r: &mut Rng) -> CommitRecord {
        CommitRecord {
            raft: (r.below(2) == 0).then(|| RaftPos { index: r.next(), term: r.next() }),
            commit_time: r.next(),
            schema_version: (r.below(2) == 0).then(|| r.next()),
            sub_xids: (0..r.below(6)).map(|_| r.next()).collect(),
            unlink: (0..r.below(6)).map(|_| r.next()).collect(),
        }
    }

    #[test]
    fn a_commit_record_round_trips_and_has_the_specified_size() {
        let mut r = Rng(1);
        for _ in 0..2000 {
            let c = random_commit(&mut r);
            let bytes = c.encode();
            assert_eq!(bytes.len(), 44 + 8 * (c.sub_xids.len() + c.unlink.len()), "44 + 8 * (n_sub + n_unlink)");
            assert_eq!(bytes.len(), c.encoded_len());
            assert_eq!(CommitRecord::decode(&bytes).unwrap(), c);
            let v = CommitView::parse(&bytes).unwrap();
            assert_eq!((v.n_sub(), v.n_unlink()), (c.sub_xids.len(), c.unlink.len()));
            assert_eq!(v.sub_xids().collect::<Vec<_>>(), c.sub_xids);
            assert_eq!(v.unlink().collect::<Vec<_>>(), c.unlink);
        }
    }

    #[test]
    fn the_commit_layout_is_the_one_of_the_specification() {
        let c = CommitRecord {
            raft: Some(RaftPos { index: 0x11, term: 0x22 }),
            commit_time: 0x33,
            schema_version: Some(0x44),
            sub_xids: vec![0x55, 0x66],
            unlink: vec![0x77],
        };
        let b = c.encode();
        assert_eq!(rd_u32(&b, 0), 3, "flags: HAS_RAFT | HAS_SCHEMA");
        assert_eq!([rd_u64(&b, 4), rd_u64(&b, 12), rd_u64(&b, 20), rd_u64(&b, 28)], [0x11, 0x22, 0x33, 0x44]);
        assert_eq!((rd_u32(&b, 36), rd_u32(&b, 40)), (2, 1), "both counts sit in the fixed part");
        assert_eq!([rd_u64(&b, 44), rd_u64(&b, 52), rd_u64(&b, 60)], [0x55, 0x66, 0x77], "sub_xids, then unlink");
        assert_eq!(b.len(), 68);
    }

    #[test]
    fn the_commit_decoder_is_strict() {
        let good = CommitRecord {
            raft: Some(RaftPos { index: 5, term: 1 }),
            commit_time: 9,
            schema_version: None,
            sub_xids: vec![21, 22],
            unlink: vec![100],
        }
        .encode();
        assert!(CommitRecord::decode(&good).is_ok());
        assert!(CommitRecord::decode(&good[..good.len() - 1]).is_err(), "one byte short");
        let mut longer = good.clone();
        longer.push(0);
        assert!(CommitRecord::decode(&longer).is_err(), "one byte too long");
        assert!(CommitRecord::decode(&good[..43]).is_err(), "shorter than the fixed part");

        let mut flags = good.clone();
        flags[0] |= 4;
        assert!(CommitRecord::decode(&flags).is_err(), "an unknown flag bit");

        let mut no_raft = good.clone();
        no_raft[0] &= !1;
        assert!(CommitRecord::decode(&no_raft).is_err(), "a Raft position without HAS_RAFT");
        let mut schema = good.clone();
        wr_u64(&mut schema, 28, 7);
        assert!(CommitRecord::decode(&schema).is_err(), "a schema version without HAS_SCHEMA");

        // Counts that would overflow a length computation must fail, not panic.
        let mut huge = good.clone();
        wr_u32(&mut huge, 36, u32::MAX);
        wr_u32(&mut huge, 40, u32::MAX);
        assert!(CommitRecord::decode(&huge).is_err());
    }

    #[test]
    fn the_allocation_batch_and_the_truncate_record_have_the_specified_layouts() {
        let b = AllocBatch { next_xid: 0x11, next_oid: 0x22, next_relfile: 0x33 }.encode();
        assert_eq!(b.len(), 24);
        assert_eq!([rd_u64(&b, 0), rd_u64(&b, 8), rd_u64(&b, 16)], [0x11, 0x22, 0x33], "xid, oid, relfile");
        assert_eq!(AllocBatch::decode(&b).unwrap(), AllocBatch { next_xid: 0x11, next_oid: 0x22, next_relfile: 0x33 });
        assert!(AllocBatch::decode(&b[..23]).is_err() && AllocBatch::decode(&[&b[..], &[0]].concat()).is_err());
        let mut wide = b;
        wr_u64(&mut wide, 8, 1 << 32);
        assert!(AllocBatch::decode(&wide).is_err(), "the upper half of next_oid must be zero");

        let mut a = AllocBatch { next_xid: 10, next_oid: 50, next_relfile: 7 };
        a.raise(AllocBatch { next_xid: 9, next_oid: 60, next_relfile: 7 });
        assert_eq!(a, AllocBatch { next_xid: 10, next_oid: 60, next_relfile: 7 }, "each counter rises only");

        assert_eq!(decode_clog_truncate(&encode_clog_truncate(77)).unwrap(), 77);
        assert!(decode_clog_truncate(&[0u8; 7]).is_err() && decode_clog_truncate(&[0u8; 9]).is_err());
    }

    #[test]
    fn abort_and_applied_advance_round_trip_and_are_strict() {
        let mut r = Rng(2);
        for _ in 0..1000 {
            let a = AbortRecord {
                sub_xids: (0..r.below(5)).map(|_| r.next()).collect(),
                created: (0..r.below(5)).map(|_| r.next()).collect(),
            };
            let b = a.encode();
            assert_eq!(b.len(), 8 + 8 * (a.sub_xids.len() + a.created.len()));
            assert_eq!(AbortRecord::decode(&b).unwrap(), a);
            assert!(AbortRecord::decode(&b[..b.len() - 1]).is_err());
            let mut longer = b.clone();
            longer.push(1);
            assert!(AbortRecord::decode(&longer).is_err());
        }
        let mut huge = vec![0u8; 8];
        wr_u32(&mut huge, 0, u32::MAX);
        wr_u32(&mut huge, 4, u32::MAX);
        assert!(AbortRecord::decode(&huge).is_err());

        let p = RaftPos { index: 42, term: 7 };
        assert_eq!(decode_applied_advance(&encode_applied_advance(p)).unwrap(), p);
        assert!(decode_applied_advance(&[0u8; 15]).is_err());
        assert!(decode_applied_advance(&[0u8; 17]).is_err());
    }

    #[test]
    fn commit_log_addressing_follows_the_specification() {
        assert_eq!(XIDS_PER_CLOG_PAGE, ((PAGE_SIZE - 32) * 4) as u64);
        assert_eq!(CSNS_PER_PAGE, ((PAGE_SIZE - 32) / 8) as u64);
        for xid in [0u64, 1, 3, 4, 16, 32_639, 32_640, 32_641, 1_000_000, 123_456_789] {
            let block = xid / 32_640;
            let byte = 32 + (xid % 32_640) / 4;
            let bit = 2 * (xid % 4);
            assert_eq!(clog_block(xid), Some(block as u32));
            assert_eq!(clog_slot(xid), (byte as usize, bit as u32), "xid {xid}");
            assert_eq!(csn_block(xid), Some((xid / 1020) as u32));
            assert_eq!(csn_slot(xid), (32 + 8 * (xid % 1020)) as usize);
        }
        assert_eq!(clog_block(u64::MAX), None, "an xid beyond a 32-bit block number is not addressable");
        // The last xid of a page is on the page's last byte.
        assert_eq!(clog_slot(32_639).0, PAGE_SIZE - 1);
        assert_eq!(csn_slot(1_019) + 8, PAGE_SIZE);
    }

    #[test]
    fn every_xid_of_a_commit_log_page_holds_its_own_two_bits() {
        let mut page = new_page_buf();
        init_clog_page(&mut page[..]);
        let mut r = Rng(3);
        let mut model = vec![ClogStatus::InProgress; XIDS_PER_CLOG_PAGE as usize];
        for _ in 0..20_000 {
            let i = r.below(XIDS_PER_CLOG_PAGE);
            let s = [ClogStatus::InProgress, ClogStatus::Committed, ClogStatus::Aborted][r.below(3) as usize];
            clog_set(&mut page[..], i, s);
            model[i as usize] = s;
        }
        for (i, want) in model.iter().enumerate() {
            assert_eq!(clog_get(&page[..], i as u64).unwrap(), *want, "xid {i}");
        }
        // The same slots on a later page read the same: the page number does not enter the slot.
        assert_eq!(clog_get(&page[..], 5 * XIDS_PER_CLOG_PAGE + 17).unwrap(), model[17]);
        // The header is untouched.
        assert_eq!(PageRef(&page[..]).kind(), PageKind::Clog);
    }

    #[test]
    fn a_commit_log_page_detects_the_reserved_value_and_a_page_of_the_wrong_kind() {
        let mut page = new_page_buf();
        init_clog_page(&mut page[..]);
        let (byte, shift) = clog_slot(6);
        page[byte] |= 3 << shift;
        assert!(clog_get(&page[..], 6).is_err(), "the value 3 is reserved");
        assert!(clog_get(&page[..], 5).is_ok(), "its neighbours are unaffected");

        let mut heap = new_page_buf();
        PageMut(&mut heap[..]).init(PageKind::Heap, 0);
        assert!(clog_get(&heap[..], 0).is_err());
        assert!(csn_get(&heap[..], 0).is_err());
    }

    #[test]
    fn the_csn_log_holds_one_csn_per_xid_and_zero_means_none() {
        let mut page = new_page_buf();
        init_csn_page(&mut page[..]);
        assert_eq!(csn_get(&page[..], 77).unwrap(), 0, "a fresh page holds no CSN");
        csn_set(&mut page[..], 77, 123_456_789_012);
        csn_set(&mut page[..], 78, 5);
        assert_eq!(csn_get(&page[..], 77).unwrap(), 123_456_789_012);
        assert_eq!(csn_get(&page[..], 78).unwrap(), 5);
        assert_eq!(csn_get(&page[..], 76).unwrap(), 0);
        assert_eq!(csn_get(&page[..], 77 + CSNS_PER_PAGE).unwrap(), 123_456_789_012, "the slot repeats on the next page");
    }
}
