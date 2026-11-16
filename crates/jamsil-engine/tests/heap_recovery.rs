//! The heap through the real write path, a crash and recovery (spec 9.6, 10.3): a random workload of every
//! `HEAP` record through `HeapRel`, with `wal_consistency_check` on, so that every record carries a forced
//! image and recovery compares the page redo builds with the page the operation produced. After the crash
//! every heap page and map page must equal the page before it.

use jamsil_engine::*;
use jamsil_heap::*;
use jamsil_page::heaprec::PruneMain;
use jamsil_page::page::{PageKind, PageRef, SlotState};
use jamsil_page::tuple::*;
use jamsil_page::*;
use jamsil_vfs::{CrashPolicy, SimVfs, Vfs};
use std::path::Path;
use std::sync::Arc;

const REL: RelFileId = 200;
const BLOCKS: u32 = 4;

fn root() -> &'static Path {
    Path::new("/db/rdb")
}

fn opts() -> EngineOptions {
    EngineOptions { buffer_frames: 64, wal_segment_size: Some(256 * 1024), wal_consistency_check: true, background_threads: false, ..Default::default() }
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
        self.next() % n.max(1)
    }
}

fn setup(fs: &SimVfs) -> Arc<NativeEngine> {
    let e = NativeEngine::open_or_create(Arc::new(fs.clone()) as Arc<dyn Vfs>, root(), opts()).unwrap();
    e.create_relation(REL, ForkId::Main).unwrap();
    e.extend_relation(REL, ForkId::Main, BLOCKS).unwrap();
    e.create_relation(REL, ForkId::Vm).unwrap();
    e.extend_relation(REL, ForkId::Vm, 1).unwrap();
    e.checkpoint().unwrap();
    e
}

fn tuple(r: &mut Rng, xid: Xid) -> Vec<u8> {
    let payload: Vec<u8> = (0..r.below(200)).map(|_| r.next() as u8).collect();
    let hdr = TupleHdr { xmin: xid, xmax: 0, cid: 0, ctid: Tid { block: 0, slot: 1 }, infomask: 0, infomask2: 0, hoff: 0 };
    encode_tuple(&hdr, &[(ColType::Fixed(8), Value::Fixed(&xid.to_le_bytes())), (ColType::Varlen, Value::Inline(&payload))]).unwrap()
}

/// Every page of the relation as the pool holds it, with the LSN and the checksum masked.
fn pages(e: &NativeEngine) -> Vec<(PageTag, Vec<u8>)> {
    let mut v = Vec::new();
    let tags = (0..BLOCKS).map(|b| PageTag::new(REL, ForkId::Main, b)).chain([PageTag::new(REL, ForkId::Vm, 0)]);
    for tag in tags {
        let p = e.pin(tag).unwrap();
        let g = p.read();
        let mut bytes = g.to_vec();
        bytes[..12].fill(0);
        v.push((tag, bytes));
    }
    v
}

#[test]
fn a_random_heap_workload_survives_a_crash_page_for_page() {
    let mut records = [0u32; 8];
    for (n, seed) in (0..12u64).enumerate() {
        let fs = SimVfs::new(900 + seed);
        let e = setup(&fs);
        let rel = HeapRel { pool: &e.pool, wal: &e.wal, rel: REL, check_images: true };
        let mut r = Rng(seed);
        let mut xid = 100u64;
        let checkpoint_at = if n % 2 == 0 { Some(150) } else { None };
        for step in 0..400 {
            if Some(step) == checkpoint_at {
                // The first change of every page after this one is preceded by an image.
                e.checkpoint().unwrap();
            }
            xid += 1;
            let b = r.below(BLOCKS as u64) as u32;
            match r.below(100) {
                0..=39 => {
                    if rel.insert(xid, b, &tuple(&mut r, xid)).is_ok() {
                        records[0] += 1;
                    }
                }
                40..=54 => {
                    let live = rel.live_slots(b).unwrap();
                    if let Some(&s) = live.get(r.below(live.len() as u64) as usize) {
                        rel.delete(xid, Tid { block: b, slot: s }, 1).unwrap();
                        records[1] += 1;
                    }
                }
                55..=62 => {
                    let live = rel.live_slots(b).unwrap();
                    if let Some(&s) = live.get(r.below(live.len() as u64) as usize) {
                        rel.lock_row(xid, Tid { block: b, slot: s }, 1).unwrap();
                        records[2] += 1;
                    }
                }
                63..=84 => {
                    let live = rel.live_slots(b).unwrap();
                    if let Some(&s) = live.get(r.below(live.len() as u64) as usize) {
                        let nb = if r.below(2) == 0 { b } else { r.below(BLOCKS as u64) as u32 };
                        let hot = nb == b && r.below(2) == 0;
                        let mut t = tuple(&mut r, xid);
                        if hot {
                            let mut h = TupleHdr::decode(&t).unwrap();
                            h.infomask2 |= infomask2::HEAP_ONLY;
                            t[..32].copy_from_slice(&h.encode());
                        }
                        if rel.update(xid, Tid { block: b, slot: s }, nb, &t, 2, hot).is_ok() {
                            records[3] += 1;
                        }
                    }
                }
                85..=92 => {
                    // Prune: some live versions die, some dead slots are reclaimed.
                    let (live, dead) = {
                        let p = e.pin(PageTag::new(REL, ForkId::Main, b)).unwrap();
                        let g = p.read();
                        let pr = PageRef(&g[..]);
                        let live: Vec<u16> = (1..=pr.n_slots() as u16).filter(|&s| pr.slot(s).1 == SlotState::Normal).collect();
                        let dead: Vec<u16> = (1..=pr.n_slots() as u16).filter(|&s| pr.slot(s).1 == SlotState::Dead).collect();
                        (live, dead)
                    };
                    let mut plan = PruneMain { prune_xid: if r.below(2) == 0 { 0 } else { xid }, ..Default::default() };
                    let mut left = live.clone();
                    for s in &live {
                        match r.below(5) {
                            0 => {
                                plan.dead.push(*s);
                                left.retain(|x| x != s);
                            }
                            1 => {
                                plan.unused.push(*s);
                                left.retain(|x| x != s);
                            }
                            _ => {}
                        }
                    }
                    plan.unused.extend(dead.iter().filter(|_| r.below(2) == 0));
                    if left.len() >= 2 && r.below(2) == 0 {
                        plan.redirects.push((left[0], left[1]));
                        plan.dead.retain(|s| *s != left[0]);
                        plan.unused.retain(|s| *s != left[0]);
                    }
                    rel.prune(xid, b, &plan).unwrap();
                    records[4] += 1;
                }
                _ => {
                    let formatted = {
                        let p = e.pin(PageTag::new(REL, ForkId::Main, b)).unwrap();
                        let g = p.read();
                        PageRef(&g[..]).kind() == PageKind::Heap && !page_is_all_visible(&g[..])
                    };
                    if formatted {
                        rel.mark_visible(xid, b).unwrap();
                        records[5] += 1;
                    }
                }
            }
        }
        // Every third seed ends with a checkpoint that completes after the last change: recovery then has
        // nothing to replay, so every page must already be on disk, which it is only if each change marked
        // its page dirty before its record had a position in the log (spec 7.5).
        let clean_end = n % 3 == 0;
        if clean_end {
            e.checkpoint().unwrap();
        }
        let before = pages(&e);
        e.wal.flush_all().unwrap();
        drop(e);
        fs.crash_with(CrashPolicy::DropUnsynced);
        // Recovery replays every record from the last checkpoint with the redo consistency check on: a
        // record whose redo diverges from the page image of its original operation panics here.
        let r2 = NativeEngine::open_or_create(Arc::new(fs.clone()) as Arc<dyn Vfs>, root(), opts()).unwrap();
        let after = pages(&r2);
        for ((tag, a), (_, b)) in before.iter().zip(&after) {
            assert_eq!(a, b, "seed {seed}: page {tag:?} after recovery differs from the page before the crash");
        }
        if clean_end {
            assert!(r2.recovery.records_replayed < 10, "seed {seed}: nothing was left to replay ({} records)", r2.recovery.records_replayed);
        } else {
            assert!(r2.recovery.records_replayed > 100, "seed {seed}: recovery replayed the workload ({} records)", r2.recovery.records_replayed);
        }
    }
    assert!(records.iter().take(6).all(|n| *n > 100), "every record type was exercised: {records:?}");
}

/// A VM bit is set only with `VISIBLE` and cleared by the first change of its page, in the same record
/// (spec 10.7, invariant 3 of 10.9): after the workload, a set bit implies the page's flag.
#[test]
fn a_set_visibility_map_bit_implies_the_all_visible_flag() {
    let fs = SimVfs::new(950);
    let e = setup(&fs);
    let rel = HeapRel { pool: &e.pool, wal: &e.wal, rel: REL, check_images: true };
    let mut r = Rng(5);
    let t = rel.insert(101, 0, &tuple(&mut r, 101)).unwrap();
    rel.mark_visible(102, 0).unwrap();
    let flag = |b: BlockNo| {
        let p = e.pin(PageTag::new(REL, ForkId::Main, b)).unwrap();
        let g = p.read();
        page_is_all_visible(&g[..])
    };
    let bit = |b: BlockNo| {
        let p = e.pin(PageTag::new(REL, ForkId::Vm, 0)).unwrap();
        let g = p.read();
        vm_bit(&g[..], b)
    };
    assert!(flag(0) && bit(0));
    // A delete on an all-visible page clears the flag and the bit, in its own record.
    rel.delete(103, t, 1).unwrap();
    assert!(!flag(0) && !bit(0), "the first change clears both");
}
