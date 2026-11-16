//! Spec 7.5 and 8.8: nothing inside a critical section may allocate, because fallible allocation
//! is forbidden there. Building a record allocates nothing, and once `check_size` has run, which
//! it does before the critical section, neither does assembling it or inserting it.
//!
//! The test installs a global allocator that counts the allocations of the calling thread. It
//! lives in this crate because spec 18.5 allows `unsafe` only in `jamsil-vfs` and `jamsil-bufmgr`.

use jamsil_page::page::{new_page_buf, PageKind, PageMut};
use jamsil_page::walrec::*;
use jamsil_page::*;
use jamsil_vfs::{SimVfs, Vfs};
use jamsil_wal::{StartPos, Wal, WalConfig};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to `System`, which upholds the contract of `GlobalAlloc`; the
// counter is a thread-local `Cell` with a constant initializer, so counting never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: the caller's contract is passed on unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` come from the caller, who got them from `alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: the caller's contract is passed on unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

#[test]
fn building_assembling_and_inserting_a_record_allocate_nothing_after_the_size_check() {
    let vfs: Arc<dyn Vfs> = Arc::new(SimVfs::new(1));
    let cfg = WalConfig { segment_size: 1 << 20, system_id: 7, buffer_bytes: 1 << 20 };
    let wal = Wal::open(vfs, Path::new("/wal"), cfg, StartPos::Fresh).unwrap();

    // Everything the record borrows exists before the critical section.
    let mut page = new_page_buf();
    PageMut(&mut page[..]).init(PageKind::Heap, 0);
    let data = vec![1u8; 100];
    let main = vec![2u8; 50];

    // Before the critical section: the gate, which also reserves the thread's assembly buffer.
    let tags: Vec<PageTag> = (0..4).map(|b| PageTag::new(100, ForkId::Main, b)).collect();
    let draft = RecordBuilder::new(rmgr::HEAP, 1).main(&main).block(BlockRef::new(tags[0]).data(&data));
    let draft4 = (1..4).fold(draft, |r, i| r.block(BlockRef::new(tags[i]).data(&data)));
    // One permit for each of the two records inserted below. Each is for one record only.
    let (permit1, permit2) = (wal.check_size(&draft4).unwrap(), wal.check_size(&draft4).unwrap());

    // The critical section: complete the builder with images, assemble, insert.
    let before = allocs();
    let mut rec = RecordBuilder::new(rmgr::HEAP, 1).xid(9).main(&main);
    for t in &tags {
        rec = rec.block(BlockRef::new(*t).image(&page[..]).data(&data));
    }
    let built = allocs();
    assert_eq!(built - before, 0, "building a record with four images allocates nothing");

    let r1 = wal.insert(permit1, rec.clone());
    let r2 = wal.insert(permit2, rec);
    assert_eq!(allocs() - built, 0, "assembling and inserting allocate nothing once check_size has run");
    assert!(r2.start >= r1.end);
}

/// The counter works: assembling a record without the gate on a thread that has not reserved its
/// assembly buffer does allocate it, which is exactly what the gate moves out of the critical
/// section.
#[test]
fn without_the_gate_the_first_assembly_on_a_thread_allocates_its_buffer() {
    let main = vec![2u8; 50];
    let rec = RecordBuilder::new(rmgr::HEAP, 1).main(&main);
    let before = allocs();
    let _sealed = assemble(&rec);
    assert!(allocs() - before > 0);
}
