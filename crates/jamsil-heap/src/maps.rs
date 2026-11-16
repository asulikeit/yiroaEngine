//! Positions in the free-space map (fork 1) and the visibility map (fork 2), and in the bitmap pages of
//! the overflow fork (spec 10.5 to 10.7). All of it is arithmetic over the 32-byte page header.

use jamsil_page::page::{PageKind, PageMut, PageRef, HEADER_SIZE};
use jamsil_page::*;

/// One byte per heap page: `8,192 - 32`.
pub const FSM_ENTRIES_PER_PAGE: u32 = 8_160;
/// One bit per heap page: `(8,192 - 32) * 8`.
pub const VM_BITS_PER_PAGE: u32 = 65_280;
/// Two bits per overflow page: `(8,192 - 32) * 4`. Bitmap blocks are 0, 32,641, 65,282, ...
pub const OVERFLOW_PAGES_PER_BITMAP: u32 = 32_640;

/// Free-space map: the entry of heap block `b` is in map block `b / 8160`, at byte `32 + b % 8160`.
pub fn fsm_position(b: BlockNo) -> (BlockNo, usize) {
    (b / FSM_ENTRIES_PER_PAGE, HEADER_SIZE + (b % FSM_ENTRIES_PER_PAGE) as usize)
}

/// The value of an entry: free bytes in units of 32, rounded down, at most 255, so the map never promises
/// more than a page has. Zero means no known space.
pub fn fsm_value(free_bytes: usize) -> u8 {
    (free_bytes / 32).min(255) as u8
}

/// Visibility map: the bit of heap block `b` is in map block `b / 65280`, in byte `32 + (b % 65280) / 8`,
/// at bit `b % 8` from the low bit.
pub fn vm_position(b: BlockNo) -> (BlockNo, usize, u8) {
    let r = b % VM_BITS_PER_PAGE;
    (b / VM_BITS_PER_PAGE, HEADER_SIZE + (r / 8) as usize, (r % 8) as u8)
}

/// Overflow fork: the bitmap block that covers `block`, or `None` if `block` is itself a bitmap block.
pub fn overflow_bitmap_of(block: BlockNo) -> Option<BlockNo> {
    let stride = OVERFLOW_PAGES_PER_BITMAP + 1;
    (block % stride != 0).then(|| block / stride * stride)
}

/// The position of the two bits of overflow page `block`: the byte in its bitmap page and the bit of
/// "allocated"; "first page of a value" is the next bit. `i` counts from 0 for the page right after the
/// bitmap page.
pub fn overflow_bits(block: BlockNo) -> Option<(BlockNo, usize, u8)> {
    let bm = overflow_bitmap_of(block)?;
    let i = block - bm - 1;
    Some((bm, HEADER_SIZE + (i / 4) as usize, 2 * (i % 4) as u8))
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("{0}")]
pub struct MapError(pub String);

/// Formats a map page. Redo does it on first use of a page that the fork's extension left as kind 0
/// (spec 7.8): the same rule in the normal path and in redo.
pub fn init_map_page(page: &mut [u8], kind: PageKind) {
    PageMut(page).init(kind, 0);
}

fn ensure_kind(page: &mut [u8], kind: PageKind) -> Result<(), MapError> {
    let k = PageRef(page).kind();
    if k == kind {
        return Ok(());
    }
    if k == PageKind::Unused {
        init_map_page(page, kind);
        return Ok(());
    }
    Err(MapError(format!("a {kind:?} page was expected, found {k:?}")))
}

/// Sets or clears the visibility-map bit of heap block `b` on its map page.
pub fn vm_set(page: &mut [u8], b: BlockNo, value: bool) -> Result<(), MapError> {
    ensure_kind(page, PageKind::Vm)?;
    let (_, byte, bit) = vm_position(b);
    if value {
        page[byte] |= 1 << bit;
    } else {
        page[byte] &= !(1 << bit);
    }
    Ok(())
}

pub fn vm_get(page: &[u8], b: BlockNo) -> bool {
    let (_, byte, bit) = vm_position(b);
    PageRef(page).kind() == PageKind::Vm && page[byte] >> bit & 1 == 1
}

pub fn fsm_set(page: &mut [u8], b: BlockNo, free_bytes: usize) -> Result<(), MapError> {
    ensure_kind(page, PageKind::Fsm)?;
    page[fsm_position(b).1] = fsm_value(free_bytes);
    Ok(())
}

pub fn fsm_get(page: &[u8], b: BlockNo) -> u8 {
    if PageRef(page).kind() == PageKind::Fsm {
        page[fsm_position(b).1]
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jamsil_page::page::new_page_buf;

    #[test]
    fn the_positions_are_those_of_the_specification() {
        assert_eq!(FSM_ENTRIES_PER_PAGE as usize, PAGE_SIZE - 32);
        assert_eq!(VM_BITS_PER_PAGE as usize, (PAGE_SIZE - 32) * 8);
        assert_eq!(OVERFLOW_PAGES_PER_BITMAP as usize, (PAGE_SIZE - 32) * 4);
        assert_eq!(fsm_position(0), (0, 32));
        assert_eq!(fsm_position(8_159), (0, 32 + 8_159));
        assert_eq!(fsm_position(8_160), (1, 32));
        assert_eq!(vm_position(0), (0, 32, 0));
        assert_eq!(vm_position(9), (0, 33, 1));
        assert_eq!(vm_position(65_279), (0, 32 + 8_159, 7));
        assert_eq!(vm_position(65_280), (1, 32, 0));
        // `b % 8` equals `(b % 65280) % 8`, because 65,280 is a multiple of 8.
        for b in [0u32, 7, 8, 65_279, 65_280, 65_287, 1_000_003, u32::MAX] {
            assert_eq!(vm_position(b).2 as u32, b % 8);
        }
        // Bitmap blocks are 0, 32,641, 65,282; each covers the 32,640 pages after it.
        assert_eq!(overflow_bitmap_of(0), None);
        assert_eq!(overflow_bitmap_of(1), Some(0));
        assert_eq!(overflow_bitmap_of(32_640), Some(0));
        assert_eq!(overflow_bitmap_of(32_641), None);
        assert_eq!(overflow_bitmap_of(32_642), Some(32_641));
        assert_eq!(overflow_bitmap_of(65_282), None);
        assert_eq!(overflow_bits(1), Some((0, 32, 0)));
        assert_eq!(overflow_bits(2), Some((0, 32, 2)));
        assert_eq!(overflow_bits(5), Some((0, 33, 0)));
        assert_eq!(overflow_bits(32_640), Some((0, 32 + 8_159, 6)));
        assert_eq!(overflow_bits(32_642), Some((32_641, 32, 0)));
    }

    #[test]
    fn the_fsm_value_rounds_down_and_never_promises_more_than_a_page_has() {
        assert_eq!([fsm_value(0), fsm_value(31), fsm_value(32), fsm_value(63), fsm_value(8_160), fsm_value(100_000)], [0, 0, 1, 1, 255, 255]);
        for free in 0..=8_160usize {
            assert!(fsm_value(free) as usize * 32 <= free);
        }
    }

    #[test]
    fn map_pages_are_formatted_on_first_use_and_hold_their_bits() {
        let mut buf = new_page_buf(); // a kind-0 page, as extension leaves it
        let page = &mut buf[..];
        assert!(!vm_get(page, 3));
        vm_set(page, 3, true).unwrap();
        vm_set(page, 4000, true).unwrap();
        assert!(vm_get(page, 3) && vm_get(page, 4000) && !vm_get(page, 4));
        vm_set(page, 3, false).unwrap();
        assert!(!vm_get(page, 3) && vm_get(page, 4000));
        assert_eq!(PageRef(page).kind(), PageKind::Vm);
        let mut heap = new_page_buf();
        PageMut(&mut heap[..]).init(PageKind::Heap, 0);
        assert!(vm_set(&mut heap[..], 1, true).is_err(), "a page of another kind is not formatted over");
        let mut f = new_page_buf();
        fsm_set(&mut f[..], 8_161, 4_000).unwrap();
        assert_eq!(fsm_get(&f[..], 8_161), 125);
        assert_eq!(fsm_get(&f[..], 8_162), 0);
    }
}
