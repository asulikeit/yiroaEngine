//! The sequence lock of the published state (spec 12.3, "Implementation contract").
//!
//! The sequence number and every field are `AtomicU64`. No plain field is read and written
//! concurrently and no `UnsafeCell` is involved, so the re-check is never asked to excuse a data
//! race. A writer holds the publication lock, makes the sequence odd, stores the fields and makes it
//! even again. A reader loads the sequence, the fields and the sequence again, and retries if the
//! first load was odd or the two differ. All loads and stores use `SeqCst`.
//!
//! This type knows nothing about the lock: the writer methods are `unsafe`-free but their caller
//! must hold it. `Published` (module `published`) does, and the loom model of this module's
//! protocol (`tests/loom_published.rs`) runs exactly this code.

use jamsil_vfs::sync::atomic::{AtomicU64, STORE_BEFORE_LOAD};

pub struct SeqLock<const N: usize> {
    seq: AtomicU64,
    fields: [AtomicU64; N],
}

impl<const N: usize> SeqLock<N> {
    pub fn new(init: [u64; N]) -> Self {
        Self { seq: AtomicU64::new(0), fields: init.map(AtomicU64::new) }
    }

    /// Starts a write section: the sequence becomes odd. The caller holds the publication lock, so
    /// there is one writer at a time.
    pub fn begin_write(&self) {
        let s = self.seq.load(STORE_BEFORE_LOAD);
        debug_assert!(s % 2 == 0, "a write section was begun inside another");
        self.seq.store(s + 1, STORE_BEFORE_LOAD);
        // Loom models `SeqCst` accesses as `AcqRel` (its README) and `fence(SeqCst)` exactly; the
        // fence adds nothing to the `SeqCst` accesses, so under loom it lets the model check the same
        // guarantee, and production does not pay for it.
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
    }

    pub fn store(&self, i: usize, v: u64) {
        debug_assert!(self.seq.load(STORE_BEFORE_LOAD) % 2 == 1, "a field was stored outside a write section");
        self.fields[i].store(v, STORE_BEFORE_LOAD);
    }

    /// Ends the write section: the sequence becomes even again.
    pub fn end_write(&self) {
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
        let s = self.seq.load(STORE_BEFORE_LOAD);
        debug_assert!(s % 2 == 1, "no write section to end");
        self.seq.store(s + 1, STORE_BEFORE_LOAD);
    }

    /// Reads all fields as of one instant between two write sections. Takes no lock; retries while
    /// a writer is inside a section.
    pub fn read(&self) -> [u64; N] {
        loop {
            let before = self.seq.load(STORE_BEFORE_LOAD);
            if before % 2 == 0 {
                let mut out = [0u64; N];
                for (o, f) in out.iter_mut().zip(&self.fields) {
                    *o = f.load(STORE_BEFORE_LOAD);
                }
                #[cfg(loom)]
                jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
                if self.seq.load(STORE_BEFORE_LOAD) == before {
                    return out;
                }
            }
            relax();
        }
    }

    /// How many write sections have completed or begun; for tests.
    pub fn sequence(&self) -> u64 {
        self.seq.load(STORE_BEFORE_LOAD)
    }
}

/// Gives a writer that is inside a section the chance to finish. In a simulation this is a
/// scheduling point, so a reader that finds a suspended writer lets it run.
fn relax() {
    #[cfg(loom)]
    loom::thread::yield_now();
    #[cfg(not(loom))]
    jamsil_vfs::sync::thread::yield_now();
}
