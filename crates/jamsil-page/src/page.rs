//! Common 32-byte page header, slotted-page operations and checksums (spec 6.4, 6.5).

use crate::*;

pub const HEADER_SIZE: usize = 32;
pub const SLOT_SIZE: usize = 4;

pub const FLAG_ALL_VISIBLE: u16 = 1;
pub const FLAG_HAS_FREE_SLOTS: u16 = 2;
pub const FLAG_PAGE_FULL: u16 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PageKind {
    Unused = 0,
    Heap = 1,
    BtreeMeta = 2,
    BtreeInternal = 3,
    BtreeLeaf = 4,
    Overflow = 5,
    Fsm = 6,
    Vm = 7,
    Clog = 8,
    CsnLog = 9,
    /// Generic payload page for tests and tooling. Kinds 240 to 255 are reserved for tests.
    Raw = 240,
}

impl PageKind {
    /// Slotted kinds keep a free gap between `lower` and `upper`. Every other kind keeps its data in
    /// the body and sets `lower = upper = 32`, so a hole-stripped image never drops payload
    /// (spec 6.4, 7.3). Kind 0 alone leaves `upper` at the special area, 8,192.
    pub fn has_free_gap(self) -> bool {
        matches!(self, Self::Unused | Self::Heap | Self::BtreeInternal | Self::BtreeLeaf)
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Unused,
            1 => Self::Heap,
            2 => Self::BtreeMeta,
            3 => Self::BtreeInternal,
            4 => Self::BtreeLeaf,
            5 => Self::Overflow,
            6 => Self::Fsm,
            7 => Self::Vm,
            8 => Self::Clog,
            9 => Self::CsnLog,
            240 => Self::Raw,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SlotState {
    Unused = 0,
    Normal = 1,
    Redirect = 2,
    Dead = 3,
}

pub type PageBuf = [u8; PAGE_SIZE];

pub fn new_page_buf() -> Box<PageBuf> {
    Box::new([0u8; PAGE_SIZE])
}

/// CRC32C seed: the checksum of the 13 little-endian bytes of the page's location.
pub fn tag_seed(tag: &PageTag) -> u32 {
    let mut b = [0u8; 13];
    b[..8].copy_from_slice(&tag.rel.to_le_bytes());
    b[8] = tag.fork as u8;
    b[9..].copy_from_slice(&tag.block.to_le_bytes());
    crc32c::crc32c(&b)
}

/// Checksum over the whole page with the checksum field read as zero.
pub fn compute_checksum(page: &[u8], tag: &PageTag) -> u32 {
    debug_assert_eq!(page.len(), PAGE_SIZE);
    let mut c = crc32c::crc32c_append(tag_seed(tag), &page[..8]);
    c = crc32c::crc32c_append(c, &[0u8; 4]);
    crc32c::crc32c_append(c, &page[12..])
}

pub fn set_checksum(page: &mut [u8], tag: &PageTag) {
    let c = compute_checksum(page, tag);
    wr_u32(page, 8, c);
}

pub fn stored_checksum(page: &[u8]) -> u32 {
    rd_u32(page, 8)
}

/// Verifies checksum, kind and header sanity. An all-zero page fails.
pub fn verify(page: &[u8], tag: &PageTag) -> Result<(), Corrupt> {
    if page.len() != PAGE_SIZE {
        return corrupt("bad page length");
    }
    if compute_checksum(page, tag) != stored_checksum(page) {
        return corrupt(format!("checksum mismatch at {tag:?}"));
    }
    if PageKind::from_u8(page[20]).is_none() {
        return corrupt(format!("unknown page kind {} at {tag:?}", page[20]));
    }
    let (lower, upper, special) = (rd_u16(page, 14) as usize, rd_u16(page, 16) as usize, rd_u16(page, 18) as usize);
    if !(HEADER_SIZE <= lower && lower <= upper && upper <= special && special <= PAGE_SIZE) {
        return corrupt(format!("bad page bounds at {tag:?}"));
    }
    Ok(())
}

pub fn is_all_zero(page: &[u8]) -> bool {
    page.iter().all(|&b| b == 0)
}

/// Read-only accessors, usable on `&[u8]` or `&PageBuf`.
pub struct PageRef<'a>(pub &'a [u8]);
/// Mutating accessors.
pub struct PageMut<'a>(pub &'a mut [u8]);

macro_rules! read_api {
    () => {
        pub fn lsn(&self) -> Lsn {
            rd_u64(self.bytes(), 0)
        }
        pub fn flags(&self) -> u16 {
            rd_u16(self.bytes(), 12)
        }
        pub fn lower(&self) -> usize {
            rd_u16(self.bytes(), 14) as usize
        }
        pub fn upper(&self) -> usize {
            rd_u16(self.bytes(), 16) as usize
        }
        pub fn special_start(&self) -> usize {
            rd_u16(self.bytes(), 18) as usize
        }
        pub fn kind(&self) -> PageKind {
            PageKind::from_u8(self.bytes()[20]).unwrap_or(PageKind::Unused)
        }
        pub fn prune_xid(&self) -> Xid {
            rd_u64(self.bytes(), 24)
        }
        /// Number of slots. Garbage in `lower` gives a small, safe answer, never a panic.
        pub fn n_slots(&self) -> usize {
            (self.lower().min(PAGE_SIZE).saturating_sub(HEADER_SIZE)) / SLOT_SIZE
        }
        pub fn free_space(&self) -> usize {
            self.upper().saturating_sub(self.lower())
        }
        pub fn special(&self) -> &[u8] {
            self.bytes().get(self.special_start()..).unwrap_or(&[])
        }
        /// Slot numbers are 1-based.
        pub fn slot(&self, slot: u16) -> (usize, SlotState, usize) {
            let o = HEADER_SIZE + (slot as usize).saturating_sub(1) * SLOT_SIZE;
            if slot == 0 || o + SLOT_SIZE > self.bytes().len() {
                return (0, SlotState::Unused, 0);
            }
            let v = rd_u32(self.bytes(), o);
            let state = match (v >> 15) & 3 {
                0 => SlotState::Unused,
                1 => SlotState::Normal,
                2 => SlotState::Redirect,
                _ => SlotState::Dead,
            };
            ((v & 0x7FFF) as usize, state, (v >> 17) as usize)
        }
        /// Tuple bytes of a Normal slot.
        pub fn item(&self, slot: u16) -> Option<&[u8]> {
            if slot == 0 || slot as usize > self.n_slots() {
                return None;
            }
            let (off, st, len) = self.slot(slot);
            if st != SlotState::Normal {
                return None;
            }
            self.bytes().get(off..off + len)
        }
    };
}

impl<'a> PageRef<'a> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
    read_api!();
}

impl PageMut<'_> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
    read_api!();

    /// Formats an empty page of `kind` with `special_size` bytes of special area.
    pub fn init(&mut self, kind: PageKind, special_size: usize) {
        self.0.fill(0);
        let special = PAGE_SIZE - special_size;
        wr_u16(self.0, 14, HEADER_SIZE as u16);
        wr_u16(self.0, 16, special as u16);
        wr_u16(self.0, 18, special as u16);
        self.0[20] = kind as u8;
        self.0[21] = 1;
        if !kind.has_free_gap() {
            wr_u16(self.0, 16, HEADER_SIZE as u16);
        }
    }
    pub fn set_lsn(&mut self, lsn: Lsn) {
        wr_u64(self.0, 0, lsn);
    }
    pub fn set_flags(&mut self, f: u16) {
        wr_u16(self.0, 12, f);
    }
    pub fn set_prune_xid(&mut self, x: Xid) {
        wr_u64(self.0, 24, x);
    }
    pub fn special_mut(&mut self) -> &mut [u8] {
        let s = self.special_start();
        &mut self.0[s..]
    }

    fn write_slot(&mut self, slot: u16, off: usize, st: SlotState, len: usize) {
        let o = HEADER_SIZE + (slot as usize - 1) * SLOT_SIZE;
        let v = (off as u32 & 0x7FFF) | ((st as u32) << 15) | ((len as u32) << 17);
        wr_u32(self.0, o, v);
    }

    /// Sets the offset of a slot, keeping its state and length (compaction).
    pub fn set_slot_offset(&mut self, slot: u16, off: usize) {
        let (_, st, len) = self.slot(slot);
        self.write_slot(slot, off, st, len);
    }

    /// Sets the length of a slot, keeping its state and offset.
    pub fn set_slot_len(&mut self, slot: u16, len: usize) {
        let (off, st, _) = self.slot(slot);
        self.write_slot(slot, off, st, len);
    }

    pub fn set_slot_state(&mut self, slot: u16, st: SlotState) {
        let (off, _, len) = self.slot(slot);
        self.write_slot(slot, off, st, len);
    }

    /// Turns a slot into a redirect to another slot number.
    pub fn set_redirect(&mut self, slot: u16, target: u16) {
        self.write_slot(slot, target as usize, SlotState::Redirect, 0);
    }

    /// Adds an item in the first unused slot, or a new slot. Returns the slot number,
    /// or `None` if there is not enough room.
    pub fn add_item(&mut self, data: &[u8]) -> Option<u16> {
        let n = self.n_slots();
        let reuse = (1..=n as u16).find(|&s| self.slot(s).1 == SlotState::Unused);
        let need = data.len() + if reuse.is_some() { 0 } else { SLOT_SIZE };
        if self.free_space() < need {
            return None;
        }
        let new_upper = self.upper() - data.len();
        self.0[new_upper..new_upper + data.len()].copy_from_slice(data);
        wr_u16(self.0, 16, new_upper as u16);
        let slot = match reuse {
            Some(s) => s,
            None => {
                wr_u16(self.0, 14, (self.lower() + SLOT_SIZE) as u16);
                n as u16 + 1
            }
        };
        self.write_slot(slot, new_upper, SlotState::Normal, data.len());
        Some(slot)
    }

    /// Puts an item at exactly `slot` (1-based): an unused slot, or the next new one. This is what redo of a
    /// heap record does (spec 10.3: "redo places a tuple at exactly the recorded slot"), and the normal path
    /// calls it too, after it has chosen the slot, so the two cannot differ. A slot that is in use, a slot
    /// beyond the next new one, or too little room is `Corrupt`: the record cannot describe this page.
    pub fn place_item(&mut self, slot: u16, data: &[u8]) -> Result<(), Corrupt> {
        let n = self.n_slots();
        if slot == 0 || slot as usize > n + 1 {
            return corrupt("slot beyond the next new one");
        }
        let fresh = slot as usize == n + 1;
        if !fresh && self.slot(slot).1 != SlotState::Unused {
            return corrupt("the slot a record names is in use");
        }
        if data.len() > 0x7FFF || self.free_space() < data.len() + if fresh { SLOT_SIZE } else { 0 } {
            return corrupt("no room for the tuple a record names");
        }
        let new_upper = self.upper() - data.len();
        self.0[new_upper..new_upper + data.len()].copy_from_slice(data);
        wr_u16(self.0, 16, new_upper as u16);
        if fresh {
            wr_u16(self.0, 14, (self.lower() + SLOT_SIZE) as u16);
        }
        self.write_slot(slot, new_upper, SlotState::Normal, data.len());
        Ok(())
    }

    /// The bytes of a Normal item, for a change in place (stamping `xmax`).
    pub fn item_mut(&mut self, slot: u16) -> Option<&mut [u8]> {
        if slot == 0 || slot as usize > self.n_slots() {
            return None;
        }
        let (off, st, len) = self.slot(slot);
        if st != SlotState::Normal {
            return None;
        }
        self.0.get_mut(off..off + len)
    }

    /// Inserts an item at position `slot` (1-based), shifting later slots up. Used by B-tree
    /// pages where slot order is key order.
    pub fn insert_item_at(&mut self, slot: u16, data: &[u8]) -> bool {
        let n = self.n_slots();
        assert!(slot as usize >= 1 && slot as usize <= n + 1);
        if self.free_space() < data.len() + SLOT_SIZE {
            return false;
        }
        let base = HEADER_SIZE + (slot as usize - 1) * SLOT_SIZE;
        let end = HEADER_SIZE + n * SLOT_SIZE;
        self.0.copy_within(base..end, base + SLOT_SIZE);
        wr_u16(self.0, 14, (self.lower() + SLOT_SIZE) as u16);
        let new_upper = self.upper() - data.len();
        self.0[new_upper..new_upper + data.len()].copy_from_slice(data);
        wr_u16(self.0, 16, new_upper as u16);
        self.write_slot(slot, new_upper, SlotState::Normal, data.len());
        true
    }

    /// Removes slot `slot`, shifting later slots down. Leaves a hole in tuple data until `compact`.
    pub fn remove_slot_shift(&mut self, slot: u16) {
        let n = self.n_slots();
        let base = HEADER_SIZE + (slot as usize - 1) * SLOT_SIZE;
        let end = HEADER_SIZE + n * SLOT_SIZE;
        self.0.copy_within(base + SLOT_SIZE..end, base);
        wr_u16(self.0, 14, (self.lower() - SLOT_SIZE) as u16);
    }

    /// Replaces a Normal item's bytes in place when the new data is not longer.
    pub fn overwrite_item(&mut self, slot: u16, data: &[u8]) -> bool {
        let (off, st, len) = self.slot(slot);
        if st != SlotState::Normal || data.len() > len {
            return false;
        }
        self.0[off..off + data.len()].copy_from_slice(data);
        self.write_slot(slot, off, SlotState::Normal, data.len());
        true
    }

    /// Rewrites tuple data contiguously at the end of the page, dropping data of unused,
    /// dead and redirect slots. Slot numbers stay stable.
    pub fn compact(&mut self) {
        let n = self.n_slots();
        let special = self.special_start();
        let mut items: Vec<(u16, Vec<u8>)> = Vec::new();
        for s in 1..=n as u16 {
            let (off, st, len) = self.slot(s);
            if st == SlotState::Normal {
                items.push((s, self.0[off..off + len].to_vec()));
            }
        }
        let mut upper = special;
        for (s, data) in &items {
            upper -= data.len();
            self.0[upper..upper + data.len()].copy_from_slice(data);
            self.write_slot(*s, upper, SlotState::Normal, data.len());
        }
        for s in 1..=n as u16 {
            let (_, st, _) = self.slot(s);
            if st == SlotState::Dead || st == SlotState::Unused {
                self.write_slot(s, 0, st, 0);
            }
        }
        let lower = self.lower();
        self.0[lower..upper].fill(0);
        wr_u16(self.0, 16, upper as u16);
    }
}

/// Where the unused gap between `lower` and `upper` lies in a page image: `(offset, len)`, or
/// `(0, 0)` if the page has none (spec 8.3). Reads two header fields; allocates nothing.
pub fn hole_of(page: &[u8]) -> (u16, u16) {
    let lower = rd_u16(page, 14) as usize;
    let upper = rd_u16(page, 16) as usize;
    if lower < upper && upper <= PAGE_SIZE {
        (lower as u16, (upper - lower) as u16)
    } else {
        (0, 0)
    }
}

/// A page image with the gap between `lower` and `upper` removed (spec 8.3).
/// Returns `(hole_offset, hole_len, bytes)`.
pub fn strip_hole(page: &[u8]) -> (u16, u16, Vec<u8>) {
    let (ho, hl) = hole_of(page);
    let (lower, upper) = (ho as usize, ho as usize + hl as usize);
    let mut out = Vec::with_capacity(PAGE_SIZE - hl as usize);
    out.extend_from_slice(&page[..lower]);
    out.extend_from_slice(&page[upper..]);
    (ho, hl, out)
}

pub fn restore_hole(hole_off: u16, hole_len: u16, bytes: &[u8], out: &mut [u8]) -> Result<(), Corrupt> {
    let (ho, hl) = (hole_off as usize, hole_len as usize);
    if bytes.len() + hl != PAGE_SIZE || ho > bytes.len() {
        return corrupt("bad page image length");
    }
    out[..ho].copy_from_slice(&bytes[..ho]);
    out[ho..ho + hl].fill(0);
    out[ho + hl..].copy_from_slice(&bytes[ho..]);
    Ok(())
}
