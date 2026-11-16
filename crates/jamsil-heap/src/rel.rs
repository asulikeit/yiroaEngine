//! The write path of the heap over the buffer pool and the WAL (spec 10.3, 7.5): the page operations of
//! `pageops` inside the protocol of every page change.
//!
//! Before the critical section: the gate (a permit for the record, spec 8.8) and the pins. In it, in this
//! order: latch the pages (heap pages in block order, then the visibility-map page: rules 5 and 6 of
//! spec 13.6), change them, mark them dirty, insert the record, set the page LSNs. Whether a block
//! reference carries an image is decided inside the insert, against the redo point read under the
//! insertion lock (spec 8.3): an image if the page has not been logged since it, and, with
//! `wal_consistency_check`, a forced image for the redo checker (spec 9.6).
//!
//! What is not here yet: conflicts with other transactions (spec 12.7), the free-space map, overflow
//! values and the scan: they are the rest of the interface of 10.8. These operations take the version a
//! caller has already resolved.

use crate::maps::vm_position;
use crate::pageops::*;
use jamsil_bufmgr::{BufError, BufferPool, PageWrite, ReadMode};
use jamsil_page::heaprec::*;
use jamsil_page::page::{PageRef, SlotState, SLOT_SIZE};
use jamsil_page::tuple::{infomask, TupleHdr, TUPLE_HEADER_LEN};
use jamsil_page::walrec::{heapop, rmgr, BlockRef, RecordBuilder, BLK_FORCED_IMAGE, BLK_WILL_INIT};
use jamsil_page::*;
use jamsil_vfs::sync::latch::Class;
use jamsil_wal::{Wal, WalError};

#[derive(Debug, thiserror::Error)]
pub enum HeapError {
    #[error("{0}")]
    Wal(#[from] WalError),
    #[error("{0}")]
    Buf(#[from] BufError),
    /// The tuple does not fit the page (the caller asks the free-space map for another, or extends).
    #[error("no room on the page")]
    NoRoom,
    /// The tuple id names no live version.
    #[error("no tuple at {0:?}")]
    NoTuple(Tid),
    /// The visibility-map page for the block does not exist (the fork was not extended).
    #[error("the visibility-map page {0} does not exist")]
    NoVmPage(BlockNo),
    #[error("corrupt: {0}")]
    Corrupt(String),
}

impl From<Corrupt> for HeapError {
    fn from(c: Corrupt) -> Self {
        HeapError::Corrupt(c.0)
    }
}

/// One relation of the heap: its pages are in `pool`, its records go to `wal`.
pub struct HeapRel<'a> {
    pub pool: &'a BufferPool,
    pub wal: &'a Wal,
    pub rel: RelFileId,
    /// `storage.wal_consistency_check` (spec 9.6): every record carries images of its pages.
    pub check_images: bool,
}

type Guard<'g> = PageWrite<'g>;

/// The blocks of one record, in the order of its block references.
struct Blocks<'b> {
    tag: [PageTag; 4],
    data: [&'b [u8]; 4],
    init: [bool; 4],
    n: usize,
}

impl<'b> Blocks<'b> {
    fn new() -> Self {
        let t = PageTag::new(0, ForkId::Main, 0);
        Self { tag: [t; 4], data: [&[]; 4], init: [false; 4], n: 0 }
    }
    fn push(&mut self, tag: PageTag, data: &'b [u8], init: bool) {
        self.tag[self.n] = tag;
        self.data[self.n] = data;
        self.init[self.n] = init;
        self.n += 1;
    }
}

impl HeapRel<'_> {
    fn tag(&self, block: BlockNo) -> PageTag {
        PageTag::new(self.rel, ForkId::Main, block)
    }
    fn vm_tag(&self, heap_block: BlockNo) -> PageTag {
        PageTag::new(self.rel, ForkId::Vm, vm_position(heap_block).0)
    }

    /// The gate: a permit for a record of this shape, before any latch (spec 8.8). The shape counts a full
    /// image for each block, so it does not matter which of them end up carrying one.
    fn gate(&self, op: u8, main_len: usize, blocks: &Blocks<'_>) -> Result<jamsil_page::walrec::WalPermit, HeapError> {
        let zeros = vec![0u8; main_len];
        let mut b = RecordBuilder::new(rmgr::HEAP, op).main(&zeros);
        for i in 0..blocks.n {
            b = b.block(BlockRef::new(blocks.tag[i]).data(blocks.data[i]));
        }
        Ok(self.wal.check_size(&b)?)
    }

    /// Marks the pages dirty, inserts the record, sets the LSNs: the end of the critical section. `main` is
    /// the payload the operation produced; `guards` are the latched pages in the order of `blocks`.
    fn emit(&self, op: u8, xid: Xid, main: &[u8], blocks: &Blocks<'_>, guards: &mut [&mut Guard<'_>], permit: jamsil_page::walrec::WalPermit) -> LsnRange {
        for g in guards.iter_mut() {
            g.mark_dirty();
        }
        let check = self.check_images;
        let range = {
            let pages: [&PageBuf; 4] = {
                // Immutable views of the latched pages for the builder.
                let mut v: [Option<&PageBuf>; 4] = [None; 4];
                for (i, g) in guards.iter().enumerate() {
                    v[i] = Some(&**g);
                }
                [v[0].unwrap_or(&ZERO), v[1].unwrap_or(&ZERO), v[2].unwrap_or(&ZERO), v[3].unwrap_or(&ZERO)]
            };
            self.wal.insert_with(permit, |redo| {
                let mut b = RecordBuilder::new(rmgr::HEAP, op).xid(xid).main(main);
                for (i, page) in pages.iter().copied().enumerate().take(blocks.n) {
                    let mut r = BlockRef::new(blocks.tag[i]).data(blocks.data[i]);
                    if blocks.init[i] {
                        r = r.flags(BLK_WILL_INIT);
                    } else if PageRef(&page[..]).lsn() <= redo {
                        r = r.image(&page[..]);
                    } else if check {
                        r = r.image(&page[..]).flags(BLK_FORCED_IMAGE);
                    }
                    b = b.block(r);
                }
                b
            })
        };
        for g in guards.iter_mut() {
            g.set_lsn(range.end);
        }
        range
    }

    fn pin_vm(&self, heap_block: BlockNo) -> Result<Option<jamsil_bufmgr::PinnedPage<'_>>, HeapError> {
        let vb = vm_position(heap_block).0;
        if self.pool.smgr().nblocks(self.rel, ForkId::Vm).map_err(BufError::Io)? <= vb {
            return Ok(None);
        }
        Ok(Some(self.pool.pin(self.vm_tag(heap_block), ReadMode::Normal)?))
    }

    /// Inserts a tuple on `block` (spec 10.3 insert, steps 3 and 4). The block comes from the free-space
    /// map or an extension. Returns the tuple id.
    pub fn insert(&self, xid: Xid, block: BlockNo, tuple: &[u8]) -> Result<Tid, HeapError> {
        let mut blocks = Blocks::new();
        blocks.push(self.tag(block), tuple, false);
        blocks.push(self.vm_tag(block), &[], false);
        let permit = self.gate(heapop::INSERT, SMALL_MAIN_LEN, &blocks)?;
        let hp = self.pool.pin(self.tag(block), ReadMode::Normal)?;
        let vp = self.pin_vm(block)?;
        let mut hw = hp.write_as(Class::Heap);
        // A tuple that cannot fit an empty page changes nothing and logs nothing.
        if PageRef(&hw[..]).kind() == jamsil_page::page::PageKind::Unused && tuple.len() + SLOT_SIZE > PAGE_SIZE - jamsil_page::page::HEADER_SIZE {
            return Err(HeapError::NoRoom);
        }
        let will_init = format_if_unused(&mut hw[..]);
        let Some(m) = insert(&mut hw[..], tuple) else {
            // Nothing was changed but the formatting of a page no record names: undo it by not logging a
            // page that was unused; a formatted page that is full is left as it was.
            return Err(HeapError::NoRoom);
        };
        let mut vw = None;
        if m.clear_all_visible {
            let Some(vp) = &vp else { return Err(HeapError::NoVmPage(vm_position(block).0)) };
            let mut w = vp.write_as(Class::Aux);
            crate::maps::vm_set(&mut w[..], block, false).map_err(|e| HeapError::Corrupt(e.0))?;
            vw = Some(w);
        }
        let main = m.encode();
        let mut b2 = Blocks::new();
        b2.push(self.tag(block), tuple, will_init);
        if vw.is_some() {
            b2.push(self.vm_tag(block), &[], false);
        }
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(2);
        guards.push(&mut hw);
        if let Some(v) = vw.as_mut() {
            guards.push(v);
        }
        self.emit(heapop::INSERT, xid, &main, &b2, &mut guards, permit);
        Ok(Tid { block, slot: m.slot })
    }

    fn read_tuple_infomask(page: &[u8], slot: u16) -> Result<u16, HeapError> {
        let p = PageRef(page);
        let t = p.item(slot).ok_or(HeapError::NoTuple(Tid { block: 0, slot }))?;
        if t.len() < TUPLE_HEADER_LEN {
            return Err(HeapError::Corrupt("a tuple shorter than its header".into()));
        }
        Ok(TupleHdr::decode(t)?.infomask)
    }

    /// Deletes (stamps `xmax` and `cid` on) the version at `tid`. The two `xmax` hint bits are cleared with
    /// the stamp, and `XMAX_LOCK_ONLY` too: the transaction's own earlier lock is replaced (spec 10.3).
    pub fn delete(&self, xid: Xid, tid: Tid, cid: u32) -> Result<(), HeapError> {
        let mut blocks = Blocks::new();
        blocks.push(self.tag(tid.block), &[], false);
        blocks.push(self.vm_tag(tid.block), &[], false);
        let permit = self.gate(heapop::DELETE, DELETE_MAIN_LEN, &blocks)?;
        let hp = self.pool.pin(self.tag(tid.block), ReadMode::Normal)?;
        let vp = self.pin_vm(tid.block)?;
        let mut hw = hp.write_as(Class::Heap);
        let old = Self::read_tuple_infomask(&hw[..], tid.slot).map_err(|e| match e {
            HeapError::NoTuple(_) => HeapError::NoTuple(tid),
            o => o,
        })?;
        let new_mask = old & !(infomask::XMAX_HINTS | infomask::XMAX_LOCK_ONLY);
        let m = delete(&mut hw[..], tid.slot, xid, cid, new_mask)?;
        let mut vw = None;
        if m.clear_all_visible {
            let Some(vp) = &vp else { return Err(HeapError::NoVmPage(vm_position(tid.block).0)) };
            let mut w = vp.write_as(Class::Aux);
            crate::maps::vm_set(&mut w[..], tid.block, false).map_err(|e| HeapError::Corrupt(e.0))?;
            vw = Some(w);
        }
        let mut b2 = Blocks::new();
        b2.push(self.tag(tid.block), &[], false);
        if vw.is_some() {
            b2.push(self.vm_tag(tid.block), &[], false);
        }
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(2);
        guards.push(&mut hw);
        if let Some(v) = vw.as_mut() {
            guards.push(v);
        }
        self.emit(heapop::DELETE, xid, &m.encode(), &b2, &mut guards, permit);
        Ok(())
    }

    /// `SELECT ... FOR UPDATE`: stamps `xmax` with `XMAX_LOCK_ONLY` (spec 10.3 row lock).
    pub fn lock_row(&self, xid: Xid, tid: Tid, cid: u32) -> Result<(), HeapError> {
        let mut blocks = Blocks::new();
        blocks.push(self.tag(tid.block), &[], false);
        let permit = self.gate(heapop::LOCK, DELETE_MAIN_LEN, &blocks)?;
        let hp = self.pool.pin(self.tag(tid.block), ReadMode::Normal)?;
        let mut hw = hp.write_as(Class::Heap);
        let old = Self::read_tuple_infomask(&hw[..], tid.slot).map_err(|e| match e {
            HeapError::NoTuple(_) => HeapError::NoTuple(tid),
            o => o,
        })?;
        let mask = (old & !infomask::XMAX_HINTS) | infomask::XMAX_LOCK_ONLY;
        let m = lock(&mut hw[..], tid.slot, xid, cid, mask)?;
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(1);
        guards.push(&mut hw);
        self.emit(heapop::LOCK, xid, &m.encode(), &blocks, &mut guards, permit);
        Ok(())
    }

    /// An update: the old version at `old` gets `xmax`, `cid` and a link to the new one, and the new tuple
    /// goes on `new_block` (spec 10.3). `hot` is the heap-only update, which the caller decided: the new
    /// block is the old one and no indexed column changed, and `new_tuple` already carries `HEAP_ONLY`.
    /// The two heap pages are latched in block order (rule 5 of spec 13.6).
    pub fn update(&self, xid: Xid, old: Tid, new_block: BlockNo, new_tuple: &[u8], cid: u32, hot: bool) -> Result<Tid, HeapError> {
        let same_page = old.block == new_block;
        let mut blocks = Blocks::new();
        blocks.push(self.tag(old.block), if same_page { new_tuple } else { &[] }, false);
        if !same_page {
            blocks.push(self.tag(new_block), new_tuple, false);
        }
        blocks.push(self.vm_tag(old.block), &[], false);
        if !same_page && vm_position(old.block).0 != vm_position(new_block).0 {
            blocks.push(self.vm_tag(new_block), &[], false);
        }
        let permit = self.gate(if hot { heapop::HOT_UPDATE } else { heapop::UPDATE }, UPDATE_MAIN_LEN, &blocks)?;
        let (op, np) = (self.pool.pin(self.tag(old.block), ReadMode::Normal)?, if same_page { None } else { Some(self.pool.pin(self.tag(new_block), ReadMode::Normal)?) });
        let (vo, vn) = (self.pin_vm(old.block)?, if same_page { None } else { self.pin_vm(new_block)? });
        // Block order: the lower heap block first.
        let mut ow;
        let mut nw: Option<Guard<'_>>;
        match &np {
            None => {
                ow = op.write_as(Class::Heap);
                nw = None;
            }
            Some(np) if new_block > old.block => {
                ow = op.write_as(Class::Heap);
                nw = Some(np.write_as(Class::Heap));
            }
            Some(np) => {
                nw = Some(np.write_as(Class::Heap));
                ow = op.write_as(Class::Heap);
            }
        }
        let old_mask = Self::read_tuple_infomask(&ow[..], old.slot).map_err(|e| match e {
            HeapError::NoTuple(_) => HeapError::NoTuple(old),
            o => o,
        })? & !(infomask::XMAX_HINTS | infomask::XMAX_LOCK_ONLY);
        let target: &mut Guard<'_> = nw.as_mut().unwrap_or(&mut ow);
        let will_init = PageRef(&target[..]).kind() == jamsil_page::page::PageKind::Unused;
        if will_init && new_tuple.len() + SLOT_SIZE > PAGE_SIZE - jamsil_page::page::HEADER_SIZE {
            return Err(HeapError::NoRoom);
        }
        if will_init {
            format_if_unused(&mut target[..]);
        }
        let Some(new_slot) = choose_insert_slot(&target[..], new_tuple.len()).filter(|s| !same_page || *s != old.slot) else {
            return Err(HeapError::NoRoom);
        };
        let (av_old, av_new) = (page_is_all_visible(&ow[..]), nw.as_ref().is_some_and(|n| page_is_all_visible(&n[..])));
        let m = UpdateMain {
            old_slot: old.slot,
            new_slot,
            clear_old: av_old,
            clear_new: if same_page { av_old } else { av_new },
            same_page,
            old_infomask: old_mask,
            xmax: xid,
            cid,
            prune_xid: prune_xid_for(&ow[..], xid),
        };
        let new_tid = Tid { block: new_block, slot: new_slot };
        apply_update_old(&mut ow[..], &m, new_tid, hot)?;
        if let Some(n) = nw.as_mut() {
            apply_update_new(&mut n[..], &m, new_tuple)?;
        } else {
            apply_update_new(&mut ow[..], &m, new_tuple)?;
        }
        // The visibility-map pages: one reference if both heap pages share a map page.
        let share = !same_page && vm_position(old.block).0 == vm_position(new_block).0;
        let mut vw_old = None;
        let mut vw_new = None;
        if m.clear_old || m.clear_new {
            let Some(vo) = &vo else { return Err(HeapError::NoVmPage(vm_position(old.block).0)) };
            let mut w = vo.write_as(Class::Aux);
            if m.clear_old {
                crate::maps::vm_set(&mut w[..], old.block, false).map_err(|e| HeapError::Corrupt(e.0))?;
            }
            if m.clear_new && (same_page || share) {
                crate::maps::vm_set(&mut w[..], new_block, false).map_err(|e| HeapError::Corrupt(e.0))?;
            }
            vw_old = Some(w);
        }
        if m.clear_new && !same_page && !share {
            let Some(vn) = &vn else { return Err(HeapError::NoVmPage(vm_position(new_block).0)) };
            let mut w = vn.write_as(Class::Aux);
            crate::maps::vm_set(&mut w[..], new_block, false).map_err(|e| HeapError::Corrupt(e.0))?;
            vw_new = Some(w);
        }
        // The order of the references is fixed by the flags (spec 10.3): the old page; the new page unless
        // both are one; then the map pages whose bits are cleared. With a map page that only the new page's
        // bit lives on (the old page's bit was not cleared), the reference is the first one.
        let mut b2 = Blocks::new();
        b2.push(self.tag(old.block), if same_page { new_tuple } else { &[] }, false);
        if !same_page {
            b2.push(self.tag(new_block), new_tuple, will_init);
        } else if will_init {
            // A same-page update cannot start a page: the old version is on it.
        }
        if vw_old.is_some() {
            b2.push(self.vm_tag(old.block), &[], false);
        }
        if vw_new.is_some() {
            b2.push(self.vm_tag(new_block), &[], false);
        }
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(4);
        guards.push(&mut ow);
        if let Some(n) = nw.as_mut() {
            guards.push(n);
        }
        if let Some(v) = vw_old.as_mut() {
            guards.push(v);
        }
        if let Some(v) = vw_new.as_mut() {
            guards.push(v);
        }
        self.emit(if hot { heapop::HOT_UPDATE } else { heapop::UPDATE }, xid, &m.encode(), &b2, &mut guards, permit);
        Ok(new_tid)
    }

    /// Prunes one page (spec 10.4) as the caller planned it: redirects, dead and unused slots, then the
    /// deterministic compaction, in one `PRUNE` record.
    pub fn prune(&self, xid: Xid, block: BlockNo, plan: &PruneMain) -> Result<(), HeapError> {
        let main = plan.encode();
        let mut blocks = Blocks::new();
        blocks.push(self.tag(block), &[], false);
        let permit = self.gate(heapop::PRUNE, main.len(), &blocks)?;
        let hp = self.pool.pin(self.tag(block), ReadMode::Normal)?;
        let mut hw = hp.write_as(Class::Heap);
        apply_prune(&mut hw[..], plan)?;
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(1);
        guards.push(&mut hw);
        self.emit(heapop::PRUNE, xid, &main, &blocks, &mut guards, permit);
        Ok(())
    }

    /// Vacuum marks a page all-visible: its flag and its visibility-map bit, one `VISIBLE` record (spec
    /// 10.7).
    pub fn mark_visible(&self, xid: Xid, block: BlockNo) -> Result<(), HeapError> {
        let mut blocks = Blocks::new();
        blocks.push(self.tag(block), &[], false);
        blocks.push(self.vm_tag(block), &[], false);
        let permit = self.gate(heapop::VISIBLE, VISIBLE_MAIN_LEN, &blocks)?;
        let hp = self.pool.pin(self.tag(block), ReadMode::Normal)?;
        let Some(vp) = self.pin_vm(block)? else { return Err(HeapError::NoVmPage(vm_position(block).0)) };
        let mut hw = hp.write_as(Class::Heap);
        apply_visible(&mut hw[..]);
        let mut vw = vp.write_as(Class::Aux);
        crate::maps::vm_set(&mut vw[..], block, true).map_err(|e| HeapError::Corrupt(e.0))?;
        let mut guards: Vec<&mut Guard<'_>> = Vec::with_capacity(2);
        guards.push(&mut hw);
        guards.push(&mut vw);
        self.emit(heapop::VISIBLE, xid, &encode_visible(), &blocks, &mut guards, permit);
        Ok(())
    }

    /// The live slots of a page (tests and tools).
    pub fn live_slots(&self, block: BlockNo) -> Result<Vec<u16>, HeapError> {
        let hp = self.pool.pin(self.tag(block), ReadMode::Normal)?;
        let r = hp.read_as(Class::Heap);
        let p = PageRef(&r[..]);
        Ok((1..=p.n_slots() as u16).filter(|&s| p.slot(s).1 == SlotState::Normal).collect())
    }
}

static ZERO: PageBuf = [0u8; PAGE_SIZE];
type PageBuf = jamsil_page::page::PageBuf;
type LsnRange = jamsil_wal::LsnRange;
