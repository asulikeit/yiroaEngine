//! Redo of the `HEAP` records (spec 10.3, 8.6).

use crate::maps::{vm_get, vm_position, vm_set, MapError};
use crate::pageops::*;
use jamsil_page::heaprec::*;
use jamsil_page::walrec::{heapop, WalRecord};
use jamsil_page::*;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum HeapRedoError {
    /// The record or the page it names is not what the specification says: recovery fails with
    /// `JDB-3001`.
    #[error("corrupt heap record: {0}")]
    Corrupt(String),
    /// A failure of the layer below (reading a page, I/O).
    #[error("{0}")]
    Pages(String),
}

impl From<Corrupt> for HeapRedoError {
    fn from(c: Corrupt) -> Self {
        HeapRedoError::Corrupt(c.0)
    }
}

impl From<MapError> for HeapRedoError {
    fn from(c: MapError) -> Self {
        HeapRedoError::Corrupt(c.0)
    }
}

/// What redo needs from the layer that owns the pages: the four states of `RedoCtx::block` (spec 8.6) of
/// one block reference. The implementation restores an image, or finds the page up to date, or finds the
/// relation dropped, and calls `f` only for a page that needs redo, under its exclusive latch; afterwards it
/// marks the page dirty and sets its LSN to the record's end.
pub trait RedoPages {
    fn apply_block(
        &mut self,
        rec: &WalRecord,
        i: usize,
        f: &mut dyn FnMut(&mut [u8; PAGE_SIZE]) -> Result<(), HeapRedoError>,
    ) -> Result<(), HeapRedoError>;
}

fn tag(rec: &WalRecord, i: usize) -> Result<PageTag, HeapRedoError> {
    rec.blocks.get(i).map(|b| b.tag).ok_or_else(|| HeapRedoError::Corrupt(format!("the record has no block reference {i}")))
}

fn data(rec: &WalRecord, i: usize) -> Result<&[u8], HeapRedoError> {
    rec.blocks.get(i).map(|b| &b.data[..]).ok_or_else(|| HeapRedoError::Corrupt(format!("the record has no block reference {i}")))
}

fn expect_blocks(rec: &WalRecord, n: usize) -> Result<(), HeapRedoError> {
    if rec.blocks.len() != n {
        return Err(HeapRedoError::Corrupt(format!("the record has {} block references, expected {n}", rec.blocks.len())));
    }
    Ok(())
}

/// Formats the page of a block reference that carries `WILL_INIT` (spec 7.8): redo reads nothing for it, so
/// the page is zeros, and the record that first uses it makes it a heap page.
fn prep(rec: &WalRecord, i: usize, p: &mut [u8; PAGE_SIZE]) {
    if rec.blocks[i].will_init() {
        init_heap_page(&mut p[..]);
    }
}

/// Clears the visibility-map bit of the heap block `heap` on the map page that reference `i` names.
fn clear_vm(pages: &mut dyn RedoPages, rec: &WalRecord, i: usize, heap: BlockNo) -> Result<(), HeapRedoError> {
    let t = tag(rec, i)?;
    if t.fork != ForkId::Vm || t.block != vm_position(heap).0 || t.rel != tag(rec, 0)?.rel {
        return Err(HeapRedoError::Corrupt("the visibility-map reference does not name the page of the heap block".into()));
    }
    pages.apply_block(rec, i, &mut |p| Ok(vm_set(&mut p[..], heap, false)?))
}

/// Redo of one `HEAP` record. `rec.xid`, the LSNs and the block references come from the log; everything
/// else is decoded here, strictly.
pub fn redo_heap(pages: &mut dyn RedoPages, rec: &WalRecord) -> Result<(), HeapRedoError> {
    let heap = tag(rec, 0)?;
    if heap.fork != ForkId::Main {
        return Err(HeapRedoError::Corrupt("the first block reference of a heap record is a heap page".into()));
    }
    match rec.info {
        heapop::INSERT => {
            let m = InsertMain::decode(&rec.main)?;
            expect_blocks(rec, 1 + usize::from(m.clear_all_visible))?;
            let tuple = data(rec, 0)?;
            pages.apply_block(rec, 0, &mut |p| {
                prep(rec, 0, p);
                Ok(apply_insert(&mut p[..], &m, tuple)?)
            })?;
            if m.clear_all_visible {
                clear_vm(pages, rec, 1, heap.block)?;
            }
        }
        heapop::MULTI_INSERT => {
            let m = MultiInsertMain::decode(&rec.main)?;
            expect_blocks(rec, 1 + usize::from(m.clear_all_visible))?;
            let items = decode_multi_insert_data(data(rec, 0)?, m.n)?;
            pages.apply_block(rec, 0, &mut |p| {
                prep(rec, 0, p);
                Ok(apply_multi_insert(&mut p[..], &m, &items)?)
            })?;
            if m.clear_all_visible {
                clear_vm(pages, rec, 1, heap.block)?;
            }
        }
        heapop::DELETE | heapop::LOCK => {
            let lock = rec.info == heapop::LOCK;
            let m = DeleteMain::decode(&rec.main, lock)?;
            expect_blocks(rec, 1 + usize::from(m.clear_all_visible))?;
            pages.apply_block(rec, 0, &mut |p| {
                if lock {
                    apply_lock(&mut p[..], &m)?;
                } else {
                    apply_delete(&mut p[..], &m)?;
                }
                Ok(())
            })?;
            if m.clear_all_visible {
                clear_vm(pages, rec, 1, heap.block)?;
            }
        }
        heapop::UPDATE | heapop::HOT_UPDATE => {
            let m = UpdateMain::decode(&rec.main)?;
            let hot = rec.info == heapop::HOT_UPDATE;
            // The order of the block references is fixed by the flags: the old page; the new page unless both
            // are one page; then the visibility-map pages whose bits are cleared, one if the two heap pages
            // share their map page.
            let new_i = if m.same_page { 0 } else { 1 };
            let new_tag = tag(rec, new_i)?;
            if !m.same_page && (new_tag.rel != heap.rel || new_tag.fork != ForkId::Main) {
                return Err(HeapRedoError::Corrupt("the new page of an update is a heap page of the same relation".into()));
            }
            let first_vm = 1 + usize::from(!m.same_page);
            let share = !m.same_page && vm_position(heap.block).0 == vm_position(new_tag.block).0;
            expect_blocks(rec, first_vm + m.vm_refs(share))?;
            let new_tuple = if m.same_page { data(rec, 0)? } else { data(rec, 1)? };
            let new_tid = Tid { block: new_tag.block, slot: m.new_slot };
            if m.same_page {
                // One page: stamp the old version, then place the new one, in one hold of the latch.
                pages.apply_block(rec, 0, &mut |p| {
                    apply_update_old(&mut p[..], &m, new_tid, hot)?;
                    Ok(apply_update_new(&mut p[..], &m, new_tuple)?)
                })?;
            } else {
                pages.apply_block(rec, 0, &mut |p| Ok(apply_update_old(&mut p[..], &m, new_tid, hot)?))?;
                pages.apply_block(rec, 1, &mut |p| {
                    prep(rec, 1, p);
                    Ok(apply_update_new(&mut p[..], &m, new_tuple)?)
                })?;
            }
            // The visibility-map bits. A page is applied once per block reference: the first visit stamps its
            // LSN, and a second one would be skipped as "already reflects this record". So when the two heap
            // pages share one map page, the one reference clears both bits in one visit.
            if share || m.same_page {
                if m.clear_old || m.clear_new {
                    let t = tag(rec, first_vm)?;
                    if t.fork != ForkId::Vm || t.rel != heap.rel || t.block != vm_position(heap.block).0 {
                        return Err(HeapRedoError::Corrupt("the visibility-map reference does not name the page of the heap block".into()));
                    }
                    let (old_b, new_b) = (heap.block, new_tag.block);
                    pages.apply_block(rec, first_vm, &mut |p| {
                        if m.clear_old {
                            vm_set(&mut p[..], old_b, false)?;
                        }
                        if m.clear_new {
                            vm_set(&mut p[..], new_b, false)?;
                        }
                        Ok(())
                    })?;
                }
            } else {
                let mut next = first_vm;
                if m.clear_old {
                    clear_vm(pages, rec, next, heap.block)?;
                    next += 1;
                }
                if m.clear_new {
                    clear_vm(pages, rec, next, new_tag.block)?;
                }
            }
        }
        heapop::PRUNE => {
            let m = PruneMain::decode(&rec.main)?;
            expect_blocks(rec, 1)?;
            pages.apply_block(rec, 0, &mut |p| Ok(apply_prune(&mut p[..], &m)?))?;
        }
        heapop::VISIBLE => {
            decode_visible(&rec.main)?;
            expect_blocks(rec, 2)?;
            let vm = tag(rec, 1)?;
            if vm.fork != ForkId::Vm || vm.block != vm_position(heap.block).0 || vm.rel != heap.rel {
                return Err(HeapRedoError::Corrupt("the visibility-map reference does not name the page of the heap block".into()));
            }
            pages.apply_block(rec, 0, &mut |p| {
                apply_visible(&mut p[..]);
                Ok(())
            })?;
            pages.apply_block(rec, 1, &mut |p| Ok(vm_set(&mut p[..], heap.block, true)?))?;
        }
        other => return Err(HeapRedoError::Corrupt(format!("unknown heap record type {other}"))),
    }
    Ok(())
}

/// The visibility-map bit of `heap_block` on `page` (tests).
pub fn vm_bit(page: &[u8], heap_block: BlockNo) -> bool {
    vm_get(page, heap_block)
}
