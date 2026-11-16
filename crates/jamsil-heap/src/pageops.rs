//! Page-level operations of the heap (spec 10.3, 10.4): the one set of functions that the normal path and
//! redo both call. Each takes the payload of its record, so the page that redo builds from a record is the
//! page the normal path built when it wrote the record. Nothing here allocates: these run inside the
//! critical section of the write protocol (spec 7.5), under the page latch.

use jamsil_page::heaprec::*;
use jamsil_page::page::{PageKind, PageMut, PageRef, SlotState, FLAG_ALL_VISIBLE, FLAG_HAS_FREE_SLOTS, SLOT_SIZE};
use jamsil_page::tuple::{infomask2, TUPLE_HEADER_LEN};
use jamsil_page::*;

/// Formats an empty heap page: kind 1, no special area.
pub fn init_heap_page(page: &mut [u8]) {
    PageMut(page).init(PageKind::Heap, 0);
}

/// A page that the extension of the fork left as kind 0 is formatted by the first record that uses it,
/// which carries `WILL_INIT` (spec 7.8). The normal path calls this before it puts the first tuple on a
/// page and sets the flag on the block reference if it returns `true`; redo formats a page whose block
/// reference carries the flag, with the same function.
pub fn format_if_unused(page: &mut [u8]) -> bool {
    if PageRef(page).kind() == PageKind::Unused {
        init_heap_page(page);
        true
    } else {
        false
    }
}

/// The first unused slot, or the next new one, if a tuple of `len` bytes fits; `None` if it does not. The
/// normal path of an insert calls it to choose the slot that the record will name.
pub fn choose_insert_slot(page: &[u8], len: usize) -> Option<u16> {
    let p = PageRef(page);
    let n = p.n_slots();
    let reuse = (1..=n as u16).find(|&s| p.slot(s).1 == SlotState::Unused);
    let need = len + if reuse.is_some() { 0 } else { SLOT_SIZE };
    (len <= 0x7FFF && p.free_space() >= need).then(|| reuse.unwrap_or(n as u16 + 1))
}

/// `HAS_FREE_SLOTS` says that some slot is unused (spec 6.4). Every operation that can change that calls
/// this last, so that it is a function of the slots and nothing else.
pub fn refresh_free_slots_flag(page: &mut [u8]) {
    let any = {
        let p = PageRef(page);
        (1..=p.n_slots() as u16).any(|s| p.slot(s).1 == SlotState::Unused)
    };
    let mut flags = PageRef(page).flags();
    if any {
        flags |= FLAG_HAS_FREE_SLOTS;
    } else {
        flags &= !FLAG_HAS_FREE_SLOTS;
    }
    PageMut(page).set_flags(flags);
}

fn clear_all_visible(page: &mut [u8]) {
    let f = PageRef(page).flags();
    PageMut(page).set_flags(f & !FLAG_ALL_VISIBLE);
}

fn all_visible(page: &[u8]) -> bool {
    PageRef(page).flags() & FLAG_ALL_VISIBLE != 0
}

/// The value `prune_xid` of a delete or an update record carries: the new `xmax` if it is older than the
/// page's `prune_xid` or the page has none, else 0, which leaves the page's value alone (spec 10.3).
pub fn prune_xid_for(page: &[u8], xmax: Xid) -> Xid {
    let cur = PageRef(page).prune_xid();
    if cur == 0 || xmax < cur {
        xmax
    } else {
        0
    }
}

/// Whether the page's `ALL_VISIBLE` flag is set: the record then clears it and the visibility-map bit.
pub fn page_is_all_visible(page: &[u8]) -> bool {
    all_visible(page)
}

/// The normal path of an insert: chooses the slot, puts the tuple there, and returns the payload of the
/// `INSERT` record. `None` if the tuple does not fit.
pub fn insert(page: &mut [u8], tuple: &[u8]) -> Option<InsertMain> {
    let slot = choose_insert_slot(page, tuple.len())?;
    let m = InsertMain { slot, clear_all_visible: all_visible(page) };
    apply_insert(page, &m, tuple).ok()?;
    Some(m)
}

/// Redo of `INSERT`, and the second half of the normal path: the tuple goes at exactly the recorded slot.
pub fn apply_insert(page: &mut [u8], m: &InsertMain, tuple: &[u8]) -> Result<(), Corrupt> {
    PageMut(page).place_item(m.slot, tuple)?;
    if m.clear_all_visible {
        clear_all_visible(page);
    }
    refresh_free_slots_flag(page);
    Ok(())
}

/// Redo of `MULTI_INSERT`: every tuple at its recorded slot.
pub fn apply_multi_insert(page: &mut [u8], m: &MultiInsertMain, tuples: &[(u16, &[u8])]) -> Result<(), Corrupt> {
    if tuples.len() != m.n as usize {
        return corrupt("MULTI_INSERT carries a different number of tuples than it says");
    }
    for (slot, t) in tuples {
        PageMut(page).place_item(*slot, t)?;
    }
    if m.clear_all_visible {
        clear_all_visible(page);
    }
    refresh_free_slots_flag(page);
    Ok(())
}

fn stamp(page: &mut [u8], slot: u16, xmax: Xid, cid: u32, infomask: u16) -> Result<(), Corrupt> {
    let mut p = PageMut(page);
    let Some(t) = p.item_mut(slot) else { return corrupt("the slot a record names holds no tuple") };
    if t.len() < TUPLE_HEADER_LEN {
        return corrupt("the tuple a record names is shorter than its header");
    }
    wr_u64(t, 8, xmax);
    wr_u32(t, 16, cid);
    wr_u16(t, 26, infomask);
    Ok(())
}

/// The normal path of a delete: stamps `xmax`, `cid` and the whole new `infomask` on the version at `slot`
/// and returns the payload of the `DELETE` record. The caller clears the two `xmax` hint bits in
/// `infomask` (spec 10.3).
pub fn delete(page: &mut [u8], slot: u16, xmax: Xid, cid: u32, infomask: u16) -> Result<DeleteMain, Corrupt> {
    let m = DeleteMain { slot, clear_all_visible: all_visible(page), xmax, cid, infomask, prune_xid: prune_xid_for(page, xmax) };
    apply_delete(page, &m)?;
    Ok(m)
}

/// Redo of `DELETE`, and the second half of the normal path.
pub fn apply_delete(page: &mut [u8], m: &DeleteMain) -> Result<(), Corrupt> {
    stamp(page, m.slot, m.xmax, m.cid, m.infomask)?;
    if m.prune_xid != 0 {
        PageMut(page).set_prune_xid(m.prune_xid);
    }
    if m.clear_all_visible {
        clear_all_visible(page);
    }
    Ok(())
}

/// The normal path of a row lock: `xmax` with `XMAX_LOCK_ONLY` in `infomask`.
pub fn lock(page: &mut [u8], slot: u16, xmax: Xid, cid: u32, infomask: u16) -> Result<DeleteMain, Corrupt> {
    let m = DeleteMain { slot, clear_all_visible: false, xmax, cid, infomask, prune_xid: 0 };
    apply_lock(page, &m)?;
    Ok(m)
}

/// Redo of `LOCK`.
pub fn apply_lock(page: &mut [u8], m: &DeleteMain) -> Result<(), Corrupt> {
    stamp(page, m.slot, m.xmax, m.cid, m.infomask)
}

/// The old page's half of an update: stamps the old version, links it to the new one by `ctid`, and marks
/// it `HOT_UPDATED` for a heap-only update. `new_tid` is the new version's tuple id.
pub fn apply_update_old(page: &mut [u8], m: &UpdateMain, new_tid: Tid, hot: bool) -> Result<(), Corrupt> {
    stamp(page, m.old_slot, m.xmax, m.cid, m.old_infomask)?;
    {
        let mut p = PageMut(page);
        let t = p.item_mut(m.old_slot).expect("stamped just now");
        t[20..26].copy_from_slice(&new_tid.to_bytes());
        if hot {
            let i2 = rd_u16(t, 28) | infomask2::HOT_UPDATED;
            wr_u16(t, 28, i2);
        }
    }
    if m.prune_xid != 0 {
        PageMut(page).set_prune_xid(m.prune_xid);
    }
    if m.clear_old {
        clear_all_visible(page);
    }
    Ok(())
}

/// The new page's half of an update: the new version goes at exactly the recorded slot.
pub fn apply_update_new(page: &mut [u8], m: &UpdateMain, tuple: &[u8]) -> Result<(), Corrupt> {
    PageMut(page).place_item(m.new_slot, tuple)?;
    if m.clear_new {
        clear_all_visible(page);
    }
    refresh_free_slots_flag(page);
    Ok(())
}

/// Compaction (spec 10.4): live tuples are packed downward from `special` in ascending slot order, the
/// data of unused, dead and redirect slots is dropped, and the gap is zeroed. Deterministic, and
/// allocation-free: the page is copied to the stack once. `PageMut::compact` gives the same page.
pub fn compact(page: &mut [u8]) {
    let mut copy = [0u8; PAGE_SIZE];
    copy.copy_from_slice(page);
    let src = PageRef(&copy);
    let n = src.n_slots();
    let mut upper = src.special_start();
    let mut p = PageMut(page);
    for s in 1..=n as u16 {
        let (off, st, len) = src.slot(s);
        if st == SlotState::Normal {
            upper -= len;
            p.0[upper..upper + len].copy_from_slice(&copy[off..off + len]);
            p.set_slot_offset(s, upper);
        } else if st == SlotState::Dead || st == SlotState::Unused {
            p.set_slot_offset(s, 0);
            if st == SlotState::Unused || st == SlotState::Dead {
                p.set_slot_len(s, 0);
            }
        }
    }
    let lower = PageRef(p.0).lower();
    p.0[lower..upper].fill(0);
    wr_u16(p.0, 16, upper as u16);
}

/// Redo of `PRUNE`, and the normal path: redirects, dead slots and unused slots, then compaction, then the
/// page's new `prune_xid`. One function, called by both, so that the compaction is the same.
pub fn apply_prune(page: &mut [u8], m: &PruneMain) -> Result<(), Corrupt> {
    let n = PageRef(page).n_slots();
    let ok = |s: u16| (s as usize) <= n;
    if !m.redirects.iter().all(|(a, b)| ok(*a) && ok(*b)) || !m.dead.iter().chain(&m.unused).all(|s| ok(*s)) {
        return corrupt("PRUNE names a slot beyond the page's slot array");
    }
    {
        let mut p = PageMut(page);
        for (from, to) in &m.redirects {
            p.set_redirect(*from, *to);
        }
        for s in &m.dead {
            p.set_slot_state(*s, SlotState::Dead);
        }
        for s in &m.unused {
            p.set_slot_state(*s, SlotState::Unused);
        }
    }
    compact(page);
    PageMut(page).set_prune_xid(m.prune_xid);
    refresh_free_slots_flag(page);
    Ok(())
}

/// Redo of `VISIBLE`: vacuum sets the page's `ALL_VISIBLE` flag (spec 10.7).
pub fn apply_visible(page: &mut [u8]) {
    let f = PageRef(page).flags();
    PageMut(page).set_flags(f | FLAG_ALL_VISIBLE);
}
