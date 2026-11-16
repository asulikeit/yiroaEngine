//! WAL record format (spec 8.2, 8.3).

use crate::page::restore_hole;
use crate::*;

pub const REC_HEADER_LEN: usize = 40;
pub const BLOCK_REF_LEN: usize = 20;
pub const MAX_RECORD_LEN: usize = 1024 * 1024;
/// The data length of a block reference is a 16-bit field, so a block carries 65,535 bytes at most.
/// 65,536 would wrap to zero. Payloads that depend on user data go in the main section.
pub const MAX_BLOCK_DATA: usize = u16::MAX as usize;
pub const MAX_BLOCKS: usize = 4;
pub const SEG_HEADER_LEN: usize = 32;
pub const SEG_MAGIC: u64 = 0x4A41_4D57_414C_5347; // "JAMWALSG"

pub const BLK_HAS_IMAGE: u8 = 1;
pub const BLK_WILL_INIT: u8 = 2;
/// Image present only because `wal_consistency_check` asked for it (spec 9.6).
pub const BLK_FORCED_IMAGE: u8 = 4;

/// Resource manager ids (spec 8.6).
pub mod rmgr {
    pub const XLOG: u8 = 1;
    pub const XACT: u8 = 2;
    pub const SMGR: u8 = 3;
    pub const CLOG: u8 = 4;
    pub const HEAP: u8 = 5;
    pub const OVERFLOW: u8 = 6;
    pub const BTREE: u8 = 7;
    pub const SEQ: u8 = 8;
    /// Generic page-update records used by tests and the M1 workload.
    pub const TEST: u8 = 15;
}

/// `XLOG` operation codes.
pub mod xlog {
    pub const CHECKPOINT: u8 = 1;
    pub const END_OF_RECOVERY: u8 = 2;
    pub const SWITCH: u8 = 3;
    pub const PAGE_IMAGE: u8 = 4;
    pub const DURABILITY_MARKER: u8 = 5;
    pub const ALLOC_BATCH: u8 = 6;
}

/// Operation codes of the generic test resource manager (`rmgr::TEST`).
pub mod testop {
    /// Block 0 data: `[offset u16][bytes..]` written at `offset` in the page.
    pub const SET_BYTES: u8 = 1;
    /// Same, but the page is first formatted as `Raw` (`WILL_INIT`).
    pub const INIT_SET_BYTES: u8 = 2;
}

/// `XACT` operation codes.
pub mod xact {
    pub const COMMIT: u8 = 1;
    pub const ABORT: u8 = 2;
    pub const APPLIED_ADVANCE: u8 = 3;
}

/// `HEAP` operation codes (spec 8.6, 10.3). `VISIBLE` is the record that 8.6 calls "mark visible".
pub mod heapop {
    pub const INSERT: u8 = 1;
    pub const MULTI_INSERT: u8 = 2;
    pub const DELETE: u8 = 3;
    pub const LOCK: u8 = 4;
    pub const UPDATE: u8 = 5;
    pub const HOT_UPDATE: u8 = 6;
    pub const PRUNE: u8 = 7;
    pub const VISIBLE: u8 = 8;
}

/// `CLOG` operation codes.
pub mod clog {
    /// One block reference flagged `WILL_INIT`, no payload.
    pub const ZERO_PAGE: u8 = 1;
    /// Payload of 8 bytes: `oldest_xid`.
    pub const TRUNCATE: u8 = 2;
}

/// `SMGR` operation codes.
pub mod smgr {
    pub const CREATE: u8 = 1;
    pub const EXTEND: u8 = 2;
    pub const TRUNCATE: u8 = 3;
    pub const RELATION_BUILT: u8 = 4;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockData {
    pub tag: PageTag,
    pub flags: u8,
    /// Page image minus the hole: `(hole_off, hole_len, bytes)`.
    pub image: Option<(u16, u16, Vec<u8>)>,
    pub data: Vec<u8>,
}

impl BlockData {
    pub fn new(tag: PageTag) -> Self {
        Self { tag, flags: 0, image: None, data: Vec::new() }
    }
    pub fn will_init(&self) -> bool {
        self.flags & BLK_WILL_INIT != 0
    }
    /// Restores the full page image into `out`.
    pub fn restore_image(&self, out: &mut [u8]) -> Result<(), Corrupt> {
        match &self.image {
            Some((ho, hl, bytes)) => restore_hole(*ho, *hl, bytes, out),
            None => corrupt("block has no image"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub prev_lsn: Lsn,
    pub end_lsn: Lsn,
    pub xid: Xid,
    pub rmgr: u8,
    pub info: u8,
    pub blocks: Vec<BlockData>,
    pub main: Vec<u8>,
}

/// One block reference of a record under construction. It borrows its payloads (spec 8.8), so
/// naming a block allocates nothing.
#[derive(Debug, Clone, Copy)]
pub struct BlockRef<'a> {
    pub tag: PageTag,
    /// `BLK_WILL_INIT` and `BLK_FORCED_IMAGE`; `BLK_HAS_IMAGE` follows from `image`.
    pub flags: u8,
    /// The whole page as it is now; assembly strips the unused gap (spec 8.3).
    pub image: Option<&'a [u8]>,
    pub data: &'a [u8],
}

impl<'a> BlockRef<'a> {
    pub fn new(tag: PageTag) -> Self {
        Self { tag, flags: 0, image: None, data: &[] }
    }
    pub fn data(mut self, data: &'a [u8]) -> Self {
        self.data = data;
        self
    }
    /// Attaches a full-page image: `page` is the whole page, `PAGE_SIZE` bytes.
    pub fn image(mut self, page: &'a [u8]) -> Self {
        assert_eq!(page.len(), PAGE_SIZE);
        self.image = Some(page);
        self
    }
    pub fn flags(mut self, flags: u8) -> Self {
        self.flags |= flags;
        self
    }
    /// Bytes the image takes in the record: the page minus its unused gap.
    fn image_len(&self) -> usize {
        self.image.map_or(0, |p| PAGE_SIZE - page::hole_of(p).1 as usize)
    }
}

/// A record under construction. Has no LSN yet. It borrows its payloads and holds at most
/// `MAX_BLOCKS` block references in a fixed array, so building a record allocates nothing: the
/// builder is completed inside a critical section, where a post-image can only be taken after the
/// page has changed and fallible allocation is forbidden (spec 7.5, 8.8).
#[derive(Debug, Clone, Default)]
pub struct RecordBuilder<'a> {
    pub rmgr: u8,
    pub info: u8,
    pub xid: Xid,
    blocks: [Option<BlockRef<'a>>; MAX_BLOCKS],
    n_blocks: usize,
    pub main: &'a [u8],
}

impl<'a> RecordBuilder<'a> {
    pub fn new(rmgr: u8, info: u8) -> Self {
        Self { rmgr, info, ..Default::default() }
    }
    pub fn xid(mut self, xid: Xid) -> Self {
        self.xid = xid;
        self
    }
    pub fn main(mut self, m: &'a [u8]) -> Self {
        self.main = m;
        self
    }
    pub fn block(mut self, b: BlockRef<'a>) -> Self {
        assert!(self.n_blocks < MAX_BLOCKS, "a record references at most {MAX_BLOCKS} blocks");
        self.blocks[self.n_blocks] = Some(b);
        self.n_blocks += 1;
        self
    }
    pub fn blocks(&self) -> impl Iterator<Item = &BlockRef<'a>> {
        self.blocks[..self.n_blocks].iter().flatten()
    }

    /// Size of the assembled record in bytes, padded to 8, with the images that are attached now.
    pub fn encoded_len(&self) -> usize {
        let blocks: usize = self.blocks().map(|b| b.image_len() + b.data.len()).sum();
        padded(REC_HEADER_LEN + self.n_blocks * BLOCK_REF_LEN + blocks + self.main.len())
    }

    /// The largest the record can become: a full image for each block it references, whether or
    /// not one is attached yet. The images are taken later, inside the critical section, so the
    /// size check that runs before it counts them (spec 8.8).
    pub fn worst_case_len(&self) -> usize {
        let blocks: usize = self.blocks().map(|b| PAGE_SIZE + b.data.len()).sum();
        padded(REC_HEADER_LEN + self.n_blocks * BLOCK_REF_LEN + blocks + self.main.len())
    }
}

/// Proof that one record passed the gate before a critical section (spec 8.8). `Wal::check_size`
/// returns it and `Wal::insert` consumes it, so no record can be inserted without having passed
/// the gate, and the compiler enforces that. It is neither `Clone` nor `Send`: it belongs to one
/// record and one thread. It carries the worst-case length that was checked, and a debug build
/// asserts that the record inserted stays within it.
///
/// The type lives here because `jamsil-wal` issues it and `jamsil-bufmgr` passes it through the
/// `WalFlush` trait, and neither crate may import the other (spec 4.1). Only `jamsil-wal` calls
/// [`WalPermit::issue`]; no `unsafe` is involved, so that rule is a convention.
#[derive(Debug)]
pub struct WalPermit {
    worst_case: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl WalPermit {
    /// For `jamsil-wal` only (see the type's documentation).
    pub fn issue(worst_case: usize) -> Self {
        Self { worst_case, _not_send: std::marker::PhantomData }
    }
    pub fn worst_case(&self) -> usize {
        self.worst_case
    }
    /// A debug build checks that the record fits the permit; a release build trusts the gate.
    pub fn assert_covers(&self, rec: &RecordBuilder<'_>) {
        debug_assert!(
            rec.encoded_len() <= self.worst_case,
            "a record of {} bytes was inserted on a permit for {}",
            rec.encoded_len(),
            self.worst_case
        );
    }
}

/// Bytes of a record whose header still needs its LSN fields and final CRC. The buffer is the
/// thread's own scratch buffer (spec 8.4 step 1): it goes back when the record is dropped, so
/// assembling allocates only the first time and when a record is larger than any before.
#[derive(Debug, Clone)]
pub struct SealedPayload {
    pub bytes: Vec<u8>,
    payload_crc: u32,
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// What each thread keeps of its assembly buffer between records (spec 18.3): 64 KiB, which
/// covers a record with four page images. A larger record grows the buffer at the gate, for that
/// record only, and the buffer shrinks back at the next gate.
pub const ASSEMBLY_KEEP: usize = 64 * 1024;

/// Makes the calling thread's assembly buffer able to hold a record of `n` bytes, and no larger
/// than `ASSEMBLY_KEEP` otherwise. `check_size` calls it before a critical section, so that the
/// insert inside does not allocate. Returns the capacity afterwards.
pub fn reserve_assembly_buffer(n: usize) -> usize {
    SCRATCH
        .try_with(|s| {
            let mut v = s.borrow_mut();
            if v.capacity() > ASSEMBLY_KEEP.max(n) {
                // An earlier record grew the buffer past the cap: give the memory back.
                *v = Vec::with_capacity(ASSEMBLY_KEEP.max(n));
            } else if v.capacity() < n {
                let have = v.len();
                v.reserve(n - have);
            }
            v.capacity()
        })
        .unwrap_or(0)
}

impl Drop for SealedPayload {
    fn drop(&mut self) {
        let _ = SCRATCH.try_with(|s| {
            let mut slot = s.borrow_mut();
            if slot.capacity() < self.bytes.capacity() {
                self.bytes.clear();
                *slot = std::mem::take(&mut self.bytes);
            }
        });
    }
}

pub fn padded(n: usize) -> usize {
    (n + 7) & !7
}

/// Assembles everything except `lsn`, `prev_lsn` and the final CRC (spec 8.4 step 1).
pub fn assemble(p: &RecordBuilder<'_>) -> SealedPayload {
    let mut buf = SCRATCH.try_with(|s| std::mem::take(&mut *s.borrow_mut())).unwrap_or_default();
    buf.clear();
    buf.resize(REC_HEADER_LEN, 0);
    for b in p.blocks() {
        assert!(
            b.data.len() <= MAX_BLOCK_DATA,
            "block data is limited to {MAX_BLOCK_DATA} bytes (a 16-bit length); larger payloads go in main"
        );
        let mut r = [0u8; BLOCK_REF_LEN];
        wr_u64(&mut r, 0, b.tag.rel);
        wr_u32(&mut r, 8, b.tag.block);
        r[12] = b.tag.fork as u8;
        let mut flags = b.flags & !BLK_HAS_IMAGE;
        if b.image.is_some() {
            flags |= BLK_HAS_IMAGE;
        }
        r[13] = flags;
        wr_u16(&mut r, 14, b.data.len() as u16);
        if let Some(page) = b.image {
            let (ho, hl) = page::hole_of(page);
            wr_u16(&mut r, 16, ho);
            wr_u16(&mut r, 18, hl);
        }
        buf.extend_from_slice(&r);
    }
    for b in p.blocks() {
        if let Some(page) = b.image {
            let (ho, hl) = page::hole_of(page);
            buf.extend_from_slice(&page[..ho as usize]);
            buf.extend_from_slice(&page[ho as usize + hl as usize..]);
        }
        buf.extend_from_slice(b.data);
    }
    buf.extend_from_slice(p.main);
    let total = padded(buf.len());
    buf.resize(total, 0);
    assert!(total <= MAX_RECORD_LEN, "record too large");
    wr_u32(&mut buf, 0, total as u32);
    buf[32] = p.rmgr;
    buf[33] = p.info;
    buf[34] = p.n_blocks as u8;
    wr_u32(&mut buf, 36, p.main.len() as u32);
    wr_u64(&mut buf, 24, p.xid);
    let payload_crc = crc32c::crc32c(&buf[REC_HEADER_LEN..]);
    SealedPayload { bytes: buf, payload_crc }
}

impl SealedPayload {
    pub fn total_len(&self) -> usize {
        self.bytes.len()
    }
    /// Step 4 of insertion: fix `lsn`/`prev_lsn`, finish the CRC over header bytes 8..40.
    pub fn seal(&mut self, lsn: Lsn, prev_lsn: Lsn) {
        wr_u64(&mut self.bytes, 8, lsn);
        wr_u64(&mut self.bytes, 16, prev_lsn);
        let crc = crc32c::crc32c_append(self.payload_crc, &self.bytes[8..REC_HEADER_LEN]);
        wr_u32(&mut self.bytes, 4, crc);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// Not enough bytes, bad length, wrong lsn/prev, or CRC mismatch: the caller classifies.
    Invalid(String),
}

/// Decodes and validates one record at `buf[0..]` per the four checks of spec 8.2.
pub fn decode(buf: &[u8], expect_lsn: Lsn, expect_prev: Lsn) -> Result<WalRecord, RecordError> {
    let bad = |m: &str| Err(RecordError::Invalid(m.to_string()));
    if buf.len() < REC_HEADER_LEN {
        return bad("short header");
    }
    let total = rd_u32(buf, 0) as usize;
    if !(REC_HEADER_LEN..=MAX_RECORD_LEN).contains(&total) || total % 8 != 0 || total > buf.len() {
        return bad("bad total_len");
    }
    let lsn = rd_u64(buf, 8);
    if lsn != expect_lsn {
        return bad("lsn mismatch");
    }
    let prev = rd_u64(buf, 16);
    if prev != expect_prev {
        return bad("prev_lsn mismatch");
    }
    let mut crc = crc32c::crc32c(&buf[REC_HEADER_LEN..total]);
    crc = crc32c::crc32c_append(crc, &buf[8..REC_HEADER_LEN]);
    if crc != rd_u32(buf, 4) {
        return bad("crc mismatch");
    }
    let n_blocks = buf[34] as usize;
    let main_len = rd_u32(buf, 36) as usize;
    if n_blocks > MAX_BLOCKS || REC_HEADER_LEN + n_blocks * BLOCK_REF_LEN > total {
        return bad("bad block count");
    }
    let mut refs = Vec::new();
    for i in 0..n_blocks {
        let o = REC_HEADER_LEN + i * BLOCK_REF_LEN;
        let Some(fork) = ForkId::from_u8(buf[o + 12]) else { return bad("bad fork") };
        refs.push((
            PageTag::new(rd_u64(buf, o), fork, rd_u32(buf, o + 8)),
            buf[o + 13],
            rd_u16(buf, o + 14) as usize,
            rd_u16(buf, o + 16),
            rd_u16(buf, o + 18),
        ));
    }
    let mut pos = REC_HEADER_LEN + n_blocks * BLOCK_REF_LEN;
    let mut blocks = Vec::new();
    for (tag, flags, data_len, ho, hl) in refs {
        let mut b = BlockData { tag, flags, image: None, data: Vec::new() };
        if flags & BLK_HAS_IMAGE != 0 {
            let ilen = PAGE_SIZE.saturating_sub(hl as usize);
            if pos + ilen > total {
                return bad("image overruns record");
            }
            b.image = Some((ho, hl, buf[pos..pos + ilen].to_vec()));
            pos += ilen;
        }
        if pos + data_len > total {
            return bad("block data overruns record");
        }
        b.data = buf[pos..pos + data_len].to_vec();
        pos += data_len;
        blocks.push(b);
    }
    if pos + main_len > total {
        return bad("main overruns record");
    }
    let Some(end_lsn) = lsn.checked_add(total as u64) else { return bad("lsn overflows") };
    Ok(WalRecord {
        lsn,
        prev_lsn: prev,
        end_lsn,
        xid: rd_u64(buf, 24),
        rmgr: buf[32],
        info: buf[33],
        blocks,
        main: buf[pos..pos + main_len].to_vec(),
    })
}

pub fn encode_seg_header(system_id: u64, segno: u64) -> [u8; SEG_HEADER_LEN] {
    let mut h = [0u8; SEG_HEADER_LEN];
    wr_u64(&mut h, 0, SEG_MAGIC);
    wr_u32(&mut h, 8, 1);
    wr_u64(&mut h, 12, system_id);
    // bytes 20..28: segment number; 28..32: crc of the first 28 bytes
    wr_u64(&mut h, 20, segno);
    let crc = crc32c::crc32c(&h[..28]);
    wr_u32(&mut h, 28, crc);
    h
}

pub fn check_seg_header(h: &[u8], system_id: u64, segno: u64) -> Result<(), Corrupt> {
    if h.len() < SEG_HEADER_LEN || rd_u64(h, 0) != SEG_MAGIC {
        return corrupt("bad segment magic");
    }
    if crc32c::crc32c(&h[..28]) != rd_u32(h, 28) {
        return corrupt("segment header crc");
    }
    if rd_u64(h, 12) != system_id || rd_u64(h, 20) != segno {
        return corrupt("segment header identity");
    }
    Ok(())
}
