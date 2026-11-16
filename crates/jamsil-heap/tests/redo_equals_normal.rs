//! The redo consistency check of spec 9.6 for the heap, over bytes: a random workload runs through the
//! normal path on one set of pages; every record it writes is encoded, decoded and redone on a shadow set;
//! the two sets must be equal after every record. Pages of the shadow set are compared with the lsn and
//! checksum masked, as the engine's checker does.

use jamsil_heap::*;
use jamsil_page::heaprec::*;
use jamsil_page::page::{new_page_buf, PageBuf, PageKind, PageMut, PageRef, SlotState, FLAG_ALL_VISIBLE};
use jamsil_page::tuple::*;
use jamsil_page::walrec::{heapop, rmgr, BlockData, WalRecord};
use jamsil_page::*;
use std::collections::HashMap;

const REL: RelFileId = 100;

fn heap(b: BlockNo) -> PageTag {
    PageTag::new(REL, ForkId::Main, b)
}
fn vmap(b: BlockNo) -> PageTag {
    PageTag::new(REL, ForkId::Vm, b)
}

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

type Pages = HashMap<PageTag, Box<PageBuf>>;

fn fresh() -> Pages {
    let mut m: Pages = HashMap::new();
    // Page 0 is formatted; page 1 is what the extension of the fork leaves: kind 0, zeros. The first record
    // that uses it carries WILL_INIT.
    let mut p = new_page_buf();
    init_heap_page(&mut p[..]);
    m.insert(heap(0), p);
    m.insert(heap(1), new_page_buf());
    m.insert(vmap(0), new_page_buf()); // kind 0, as the extension of the fork leaves it
    m
}

/// The redo side: finds the page, skips it if it already reflects the record, applies, stamps the LSN.
struct Shadow {
    pages: Pages,
    applied: u64,
}

impl RedoPages for Shadow {
    fn apply_block(&mut self, rec: &WalRecord, i: usize, f: &mut dyn FnMut(&mut [u8; PAGE_SIZE]) -> Result<(), HeapRedoError>) -> Result<(), HeapRedoError> {
        let tag = rec.blocks[i].tag;
        let page = self.pages.get_mut(&tag).ok_or_else(|| HeapRedoError::Pages(format!("no page {tag:?}")))?;
        if PageRef(&page[..]).lsn() >= rec.end_lsn {
            return Ok(()); // already reflects this record
        }
        f(page)?;
        PageMut(&mut page[..]).set_lsn(rec.end_lsn);
        self.applied += 1;
        Ok(())
    }
}

fn masked(p: &PageBuf) -> Vec<u8> {
    let mut v = p.to_vec();
    v[..12].fill(0);
    v
}

fn same(a: &Pages, b: &Pages, what: &str, step: usize) {
    for (tag, p) in a {
        let (x, y) = (masked(p), masked(&b[tag]));
        if x != y {
            let at: Vec<_> = x.iter().zip(&y).enumerate().filter(|(_, (u, v))| u != v).take(6).map(|(i, (u, v))| (i, *u, *v)).collect();
            panic!("step {step}: {what}: page {tag:?} differs between the normal path and redo: (offset, normal, redo) {at:?}");
        }
    }
}

fn record(info: u8, lsn: u64, main: Vec<u8>, blocks: Vec<(PageTag, Vec<u8>)>) -> WalRecord {
    record_with(info, lsn, main, blocks, &[])
}

/// `init` lists the indexes of the block references that carry `WILL_INIT`.
fn record_with(info: u8, lsn: u64, main: Vec<u8>, blocks: Vec<(PageTag, Vec<u8>)>, init: &[usize]) -> WalRecord {
    WalRecord {
        lsn,
        prev_lsn: lsn.saturating_sub(100),
        end_lsn: lsn + 100,
        xid: 20,
        rmgr: rmgr::HEAP,
        info,
        blocks: blocks
            .into_iter()
            .enumerate()
            .map(|(i, (tag, data))| BlockData { tag, flags: if init.contains(&i) { jamsil_page::walrec::BLK_WILL_INIT } else { 0 }, image: None, data })
            .collect(),
        main,
    }
}

fn tuple(r: &mut Rng, xmin: Xid) -> Vec<u8> {
    let len = r.below(180) as usize;
    let payload: Vec<u8> = (0..len).map(|i| (i as u64 ^ r.next()) as u8).collect();
    let hdr = TupleHdr { xmin, xmax: 0, cid: 0, ctid: Tid { block: 0, slot: 1 }, infomask: 0, infomask2: 0, hoff: 0 };
    encode_tuple(&hdr, &[(ColType::Fixed(8), Value::Fixed(&xmin.to_le_bytes())), (ColType::Varlen, Value::Inline(&payload))]).unwrap()
}

fn normal_slots(p: &PageBuf) -> Vec<u16> {
    let pr = PageRef(&p[..]);
    (1..=pr.n_slots() as u16).filter(|&s| pr.slot(s).1 == SlotState::Normal).collect()
}

#[test]
fn redo_builds_the_page_the_normal_path_built() {
    let mut kinds = [0usize; 8];
    for seed in 0..40u64 {
        let mut r = Rng(seed);
        let (mut a, mut shadow) = (fresh(), Shadow { pages: fresh(), applied: 0 });
        let mut lsn = 1_000u64;
        let mut xid = 20u64;
        for step in 0..300 {
            xid += 1;
            lsn += 100;
            let pb = r.below(2) as u32;
            let which = r.below(100);
            let rec = if which < 35 {
                // INSERT
                let t = tuple(&mut r, xid);
                let page = &mut a.get_mut(&heap(pb)).unwrap()[..];
                let will_init = format_if_unused(page);
                let was_av = page_is_all_visible(page);
                let Some(m) = insert(page, &t) else { continue };
                let mut blocks = vec![(heap(pb), t)];
                if m.clear_all_visible {
                    assert!(was_av);
                    vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], pb, false).unwrap();
                    blocks.push((vmap(0), vec![]));
                }
                kinds[0] += 1;
                record_with(heapop::INSERT, lsn, m.encode().to_vec(), blocks, if will_init { &[0] } else { &[] })
            } else if which < 50 {
                // DELETE
                let slots = normal_slots(&a[&heap(pb)]);
                let Some(&s) = slots.get(r.below(slots.len().max(1) as u64) as usize) else { continue };
                let page = &mut a.get_mut(&heap(pb)).unwrap()[..];
                let m = delete(page, s, xid, 1, infomask::XMAX_COMMITTED).unwrap();
                let mut blocks = vec![(heap(pb), vec![])];
                if m.clear_all_visible {
                    vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], pb, false).unwrap();
                    blocks.push((vmap(0), vec![]));
                }
                kinds[1] += 1;
                record(heapop::DELETE, lsn, m.encode().to_vec(), blocks)
            } else if which < 58 {
                // LOCK
                let slots = normal_slots(&a[&heap(pb)]);
                let Some(&s) = slots.get(r.below(slots.len().max(1) as u64) as usize) else { continue };
                let m = lock(&mut a.get_mut(&heap(pb)).unwrap()[..], s, xid, 2, infomask::XMAX_LOCK_ONLY).unwrap();
                kinds[2] += 1;
                record(heapop::LOCK, lsn, m.encode().to_vec(), vec![(heap(pb), vec![])])
            } else if which < 80 {
                // UPDATE, heap-only or not, on one page or two
                let qb = r.below(2) as u32;
                let slots = normal_slots(&a[&heap(pb)]);
                let Some(&old) = slots.get(r.below(slots.len().max(1) as u64) as usize) else { continue };
                let t = tuple(&mut r, xid);
                let same_page = pb == qb;
                // The slot the new version will take is chosen before the old one is stamped (stamping
                // changes no slot), on the page the new version goes to.
                let Some(new_slot) = choose_insert_slot(&a[&heap(qb)][..], t.len()) else { continue };
                if same_page && new_slot == old {
                    continue;
                }
                let (av_old, av_new) = (page_is_all_visible(&a[&heap(pb)][..]), page_is_all_visible(&a[&heap(qb)][..]));
                let m = UpdateMain {
                    old_slot: old,
                    new_slot,
                    clear_old: av_old,
                    clear_new: if same_page { av_old } else { av_new },
                    same_page,
                    old_infomask: infomask::XMAX_COMMITTED & !infomask::XMAX_HINTS,
                    xmax: xid,
                    cid: 3,
                    prune_xid: prune_xid_for(&a[&heap(pb)][..], xid),
                };
                let hot = r.below(2) == 0;
                let new_tid = Tid { block: qb, slot: new_slot };
                apply_update_old(&mut a.get_mut(&heap(pb)).unwrap()[..], &m, new_tid, hot).unwrap();
                apply_update_new(&mut a.get_mut(&heap(qb)).unwrap()[..], &m, &t).unwrap();
                let mut blocks = vec![(heap(pb), vec![])];
                if same_page {
                    blocks[0].1 = t;
                } else {
                    blocks.push((heap(qb), t));
                }
                // Both heap pages share map page 0, so one reference when both bits are cleared.
                let share = !same_page;
                let n_vm = m.vm_refs(share);
                for _ in 0..n_vm {
                    blocks.push((vmap(0), vec![]));
                }
                if m.clear_old {
                    vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], pb, false).unwrap();
                }
                if m.clear_new {
                    vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], qb, false).unwrap();
                }
                kinds[3] += 1;
                record(if hot { heapop::HOT_UPDATE } else { heapop::UPDATE }, lsn, m.encode().to_vec(), blocks)
            } else if which < 90 {
                // PRUNE: some versions die (dead slot, data dropped), some dead slots are reclaimed, and a
                // root redirects to a live member.
                let page = &mut a.get_mut(&heap(pb)).unwrap()[..];
                let (normal, dead): (Vec<u16>, Vec<u16>) = {
                    let p = PageRef(page);
                    let n: Vec<u16> = (1..=p.n_slots() as u16).filter(|&s| p.slot(s).1 == SlotState::Normal).collect();
                    let d: Vec<u16> = (1..=p.n_slots() as u16).filter(|&s| p.slot(s).1 == SlotState::Dead).collect();
                    (n, d)
                };
                let mut m = PruneMain { prune_xid: if r.below(2) == 0 { 0 } else { xid }, ..Default::default() };
                let mut live = normal.clone();
                for s in &normal {
                    match r.below(6) {
                        0 => {
                            m.dead.push(*s);
                            live.retain(|x| x != s);
                        }
                        1 => {
                            m.unused.push(*s);
                            live.retain(|x| x != s);
                        }
                        _ => {}
                    }
                }
                for s in &dead {
                    if r.below(2) == 0 {
                        m.unused.push(*s);
                    }
                }
                if live.len() >= 2 && r.below(2) == 0 {
                    let (from, to) = (live[0], live[1]);
                    m.redirects.push((from, to));
                }
                // A slot is named once.
                m.dead.retain(|s| !m.redirects.iter().any(|(f, _)| f == s));
                m.unused.retain(|s| !m.redirects.iter().any(|(f, _)| f == s));
                apply_prune(page, &m).unwrap();
                kinds[4] += 1;
                record(heapop::PRUNE, lsn, m.encode(), vec![(heap(pb), vec![])])
            } else {
                // VISIBLE: vacuum sets the flag and the map bit
                // Vacuum marks only a page that holds tuples, so never one that no record has formatted yet.
                if page_is_all_visible(&a[&heap(pb)][..]) || PageRef(&a[&heap(pb)][..]).kind() == PageKind::Unused {
                    continue;
                }
                apply_visible(&mut a.get_mut(&heap(pb)).unwrap()[..]);
                vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], pb, true).unwrap();
                kinds[5] += 1;
                record(heapop::VISIBLE, lsn, encode_visible().to_vec(), vec![(heap(pb), vec![]), (vmap(0), vec![])])
            };
            // Invariant 3 of spec 10.9: a set visibility-map bit implies the page's `ALL_VISIBLE` flag.
            for b in 0..2u32 {
                if vm_bit(&a[&vmap(0)][..], b) {
                    assert!(page_is_all_visible(&a[&heap(b)][..]), "seed {seed} step {step}: a set map bit on a page that is not all-visible");
                }
            }
            if matches!(rec.info, heapop::INSERT | heapop::DELETE | heapop::UPDATE | heapop::HOT_UPDATE) {
                assert!(!page_is_all_visible(&a[&heap(pb)][..]), "seed {seed} step {step}: a changed page is still all-visible");
            }
            redo_heap(&mut shadow, &rec).unwrap_or_else(|e| panic!("seed {seed} step {step}: redo of {:?} failed: {e}", rec.info));
            same(&a, &shadow.pages, &format!("seed {seed}, record type {}", rec.info), step);
            // A second delivery of the same record changes nothing: the page already reflects it.
            let before = shadow.applied;
            redo_heap(&mut shadow, &rec).unwrap();
            assert_eq!(shadow.applied, before, "a page that reflects a record is skipped");
        }
        // The flags of every page agree with its slots.
        for b in 0..2 {
            let p = PageRef(&a[&heap(b)][..]);
            let any_unused = (1..=p.n_slots() as u16).any(|s| p.slot(s).1 == SlotState::Unused);
            assert_eq!(p.flags() & jamsil_page::page::FLAG_HAS_FREE_SLOTS != 0, any_unused);
        }
    }
    assert!(kinds.iter().take(6).all(|k| *k > 50), "every record type was exercised: {kinds:?}");
}

/// The compaction of the heap is the compaction of the page code, byte for byte, and allocates nothing.
#[test]
fn the_heaps_compaction_equals_the_page_codes() {
    let mut r = Rng(77);
    for _ in 0..300 {
        let mut a = new_page_buf();
        init_heap_page(&mut a[..]);
        for _ in 0..r.below(40) {
            let t = vec![r.below(250) as u8; 30 + r.below(120) as usize];
            let _ = PageMut(&mut a[..]).add_item(&t);
        }
        let n = PageRef(&a[..]).n_slots() as u16;
        for s in 1..=n {
            match r.below(4) {
                0 => PageMut(&mut a[..]).set_slot_state(s, SlotState::Dead),
                1 => PageMut(&mut a[..]).set_slot_state(s, SlotState::Unused),
                2 if s > 1 => PageMut(&mut a[..]).set_redirect(s, 1),
                _ => {}
            }
        }
        let mut b = a.clone();
        compact(&mut a[..]);
        PageMut(&mut b[..]).compact();
        assert_eq!(a[..], b[..]);
        // Live tuples are packed downward from `special` in ascending slot order.
        let p = PageRef(&a[..]);
        let mut upper = PAGE_SIZE;
        for s in 1..=p.n_slots() as u16 {
            let (off, st, len) = p.slot(s);
            if st == SlotState::Normal {
                upper -= len;
                assert_eq!(off, upper, "slot {s}");
            }
        }
        assert_eq!(p.upper(), upper);
    }
}

#[test]
fn redo_refuses_what_a_record_cannot_describe() {
    let mut shadow = Shadow { pages: fresh(), applied: 0 };
    let t = tuple(&mut Rng(1), 21);
    let ok = InsertMain { slot: 1, clear_all_visible: false };
    let rec = record(heapop::INSERT, 1_000, ok.encode().to_vec(), vec![(heap(0), t.clone())]);
    redo_heap(&mut shadow, &rec).unwrap();
    // The same slot again, on a page that does not reflect it yet: the slot is in use.
    let mut again = record(heapop::INSERT, 2_000, ok.encode().to_vec(), vec![(heap(0), t.clone())]);
    assert!(matches!(redo_heap(&mut shadow, &again), Err(HeapRedoError::Corrupt(_))), "a record that names a slot in use");
    // A slot beyond the next new one.
    again.main = InsertMain { slot: 9, clear_all_visible: false }.encode().to_vec();
    assert!(redo_heap(&mut shadow, &again).is_err());
    // A flag that promises a map page the record does not carry.
    again.main = InsertMain { slot: 2, clear_all_visible: true }.encode().to_vec();
    assert!(redo_heap(&mut shadow, &again).is_err(), "one block reference too few");
    // An unknown record type, a payload of the wrong length, a heap reference on the wrong fork.
    let bad = record(99, 3_000, vec![], vec![(heap(0), vec![])]);
    assert!(redo_heap(&mut shadow, &bad).is_err());
    let short = record(heapop::DELETE, 3_000, vec![0; 27], vec![(heap(0), vec![])]);
    assert!(redo_heap(&mut shadow, &short).is_err());
    let wrong = record(heapop::PRUNE, 3_000, PruneMain::default().encode(), vec![(vmap(0), vec![])]);
    assert!(redo_heap(&mut shadow, &wrong).is_err());
    // A visibility-map reference that names the wrong map page for the heap block.
    let m = InsertMain { slot: 2, clear_all_visible: true };
    let mut off = record(heapop::INSERT, 4_000, m.encode().to_vec(), vec![(heap(0), t), (vmap(5), vec![])]);
    assert!(redo_heap(&mut shadow, &off).is_err());
    off.blocks[1].tag = vmap(0);
    // The heap page is not marked all-visible, but the record says it cleared the flag: the redo is still
    // deterministic, because it only clears.
    redo_heap(&mut shadow, &off).unwrap();
    assert_eq!(PageRef(&shadow.pages[&heap(0)][..]).kind(), PageKind::Heap);
    let _ = FLAG_ALL_VISIBLE;
}

/// Rules that the normal path and redo share by construction, so that comparing them cannot test them:
/// they are asserted directly.
#[test]
fn a_change_clears_all_visible_lowers_prune_xid_and_links_the_hot_chain() {
    let mut r = Rng(9);
    let mut a = fresh();
    // Fill page 0, make it all-visible (vacuum), then change it.
    let t1 = tuple(&mut r, 21);
    let s1 = insert(&mut a.get_mut(&heap(0)).unwrap()[..], &t1).unwrap().slot;
    let s2 = insert(&mut a.get_mut(&heap(0)).unwrap()[..], &tuple(&mut r, 22)).unwrap().slot;
    apply_visible(&mut a.get_mut(&heap(0)).unwrap()[..]);
    vm_set(&mut a.get_mut(&vmap(0)).unwrap()[..], 0, true).unwrap();
    assert!(page_is_all_visible(&a[&heap(0)][..]));

    // An insert on an all-visible page says that it clears the flag and the map bit, and clears the flag.
    let m = insert(&mut a.get_mut(&heap(0)).unwrap()[..], &tuple(&mut r, 23)).unwrap();
    assert!(m.clear_all_visible);
    assert!(!page_is_all_visible(&a[&heap(0)][..]), "a change clears ALL_VISIBLE");
    // On a page that is not all-visible the record does not say so.
    assert!(!insert(&mut a.get_mut(&heap(0)).unwrap()[..], &tuple(&mut r, 24)).unwrap().clear_all_visible);

    // `prune_xid` is lowered, never raised, and 0 in the record leaves it alone (spec 10.3).
    let page = &mut a.get_mut(&heap(0)).unwrap()[..];
    assert_eq!(PageRef(page).prune_xid(), 0);
    let d = delete(page, s1, 60, 1, infomask::XMAX_COMMITTED).unwrap();
    assert_eq!((d.prune_xid, PageRef(page).prune_xid()), (60, 60), "the first delete sets it");
    let d = delete(page, s2, 70, 1, 0).unwrap();
    assert_eq!((d.prune_xid, PageRef(page).prune_xid()), (0, 60), "a newer xmax leaves it at 60");
    let s3 = insert(page, &tuple(&mut r, 25)).unwrap().slot;
    let d = delete(page, s3, 40, 1, 0).unwrap();
    assert_eq!((d.prune_xid, PageRef(page).prune_xid()), (40, 40), "an older xmax lowers it");

    // A lock stamps `xmax` with the lock-only bit and does not touch `prune_xid`, and a delete after it can
    // clear the bit, because `infomask` is the whole new value (answer C2, change 1).
    let s4 = insert(page, &tuple(&mut r, 26)).unwrap().slot;
    lock(page, s4, 80, 1, infomask::XMAX_LOCK_ONLY).unwrap();
    assert_eq!(PageRef(page).prune_xid(), 40);
    let h = TupleHdr::decode(PageRef(page).item(s4).unwrap()).unwrap();
    assert!(h.has(infomask::XMAX_LOCK_ONLY) && h.xmax == 80);
    delete(page, s4, 80, 2, 0).unwrap();
    let h = TupleHdr::decode(PageRef(page).item(s4).unwrap()).unwrap();
    assert!(!h.has(infomask::XMAX_LOCK_ONLY) && h.cid == 2, "the delete cleared XMAX_LOCK_ONLY");

    // A heap-only update links the old version to the new by `ctid` and flags it `HOT_UPDATED`; the other
    // update links it too and does not flag it.
    for hot in [true, false] {
        let mut pg = fresh();
        let old = insert(&mut pg.get_mut(&heap(0)).unwrap()[..], &tuple(&mut r, 21)).unwrap().slot;
        let t = tuple(&mut r, 30);
        let new_slot = choose_insert_slot(&pg[&heap(0)][..], t.len()).unwrap();
        let m = UpdateMain { old_slot: old, new_slot, clear_old: false, clear_new: false, same_page: true, old_infomask: 0, xmax: 30, cid: 1, prune_xid: 30 };
        let new_tid = Tid { block: 0, slot: new_slot };
        let page = &mut pg.get_mut(&heap(0)).unwrap()[..];
        apply_update_old(page, &m, new_tid, hot).unwrap();
        apply_update_new(page, &m, &t).unwrap();
        let oh = TupleHdr::decode(PageRef(page).item(old).unwrap()).unwrap();
        assert_eq!(oh.ctid, new_tid, "the old version points at the new one");
        assert_eq!(oh.infomask2 & infomask2::HOT_UPDATED != 0, hot);
        assert_eq!(oh.xmax, 30);
        assert_eq!(PageRef(page).prune_xid(), 30);
    }
}
