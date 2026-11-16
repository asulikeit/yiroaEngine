//! The internal `sync` facade (spec 18.5, milestone M0): the one place engine code gets locks,
//! condition variables, atomics and threads from. Routing everything through here is what lets a
//! controlled scheduler or loom replace the primitives later without touching call sites.
//!
//! The module also holds the debug-build latch-order checker of spec 13.6 rule B, because it needs
//! a per-thread stack that every crate can reach.

pub use std::sync::{Arc, Weak};

/// Atomics are the standard ones. Under `--cfg loom` they are loom's, which offers the same names
/// and lets a model explore every interleaving and every outcome the memory model allows (spec
/// 19.2). Only the atomics change: the mutex wrappers stay as they are, so a loom model that wants
/// to cover a lock needs the wrappers first (B1, B5).
pub mod atomic {
    #[cfg(not(loom))]
    pub use std::sync::atomic::*;

    #[cfg(loom)]
    pub use loom::sync::atomic::*;

    /// The ordering of every operation of the store-before-load protocols (spec 7.5, 7.6, 9.2): a
    /// page change sets the dirty flag and then reads the published redo point, and a checkpoint
    /// publishes the redo point and then scans for the flag. Only `SeqCst` guarantees that one of
    /// the two sees the other. The four operations (`jamsil_bufmgr::dirty_flag`,
    /// `jamsil_wal::redo_point`) all use this constant, and `the_protocol_ordering_is_seq_cst`
    /// pins it, so that nobody weakens one of them while the loom model keeps passing on its fence.
    pub const STORE_BEFORE_LOAD: Ordering = Ordering::SeqCst;
}

// The surface below is fixed on purpose (spec 19.2): a mutex, a reader-writer lock, a condition
// variable, atomics and named threads, with only the calls that a simulated or loom variant can
// also offer. The types are thin newtypes, so nothing of `parking_lot` beyond these calls can reach
// engine code: no guard mapping, no upgradable reads, no fair unlock, no `lock_api` traits.
// `clippy.toml` forbids the underlying types outside this module.

/// Under a running simulation (`crate::sched`) every operation below is a scheduling point and the
/// lock state lives in the scheduler; otherwise each is a thin call into `parking_lot`. The check is
/// one relaxed atomic load when no simulation is running. Even in a simulation the data stays
/// behind a `parking_lot` lock, which is then never contended, because only the task that holds the
/// baton runs: the scheduler decides who may take a lock, and the real lock follows.
#[allow(clippy::disallowed_types)]
// Under `--cfg loom` the mutex and the condition variable are loom's (`loom_prim`), so the ones below,
// and the scheduler hooks they call, are unused there.
#[cfg_attr(loom, allow(dead_code))]
mod prim {
    use crate::sched::{self, Ctx};
    use std::ops::{Deref, DerefMut};
    use std::sync::atomic::{AtomicU64 as IdCell, Ordering as IdOrder};
    use std::time::Duration;

    /// The scheduler's name for a lock or a condition variable, given out on first use in a
    /// simulation.
    fn object_id(cell: &IdCell) -> u64 {
        let v = cell.load(IdOrder::Relaxed);
        if v != 0 {
            return v;
        }
        let n = sched::new_object_id();
        match cell.compare_exchange(0, n, IdOrder::Relaxed, IdOrder::Relaxed) {
            Ok(_) => n,
            Err(cur) => cur,
        }
    }

    pub struct Mutex<T> {
        id: IdCell,
        inner: parking_lot::Mutex<T>,
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    /// `inner` is `None` only while a condition-variable wait has released the lock.
    pub struct MutexGuard<'a, T> {
        inner: Option<parking_lot::MutexGuard<'a, T>>,
        lock: &'a parking_lot::Mutex<T>,
        sim: Option<(Ctx, u64)>,
    }

    impl<T> Mutex<T> {
        pub fn new(v: T) -> Self {
            Self { id: IdCell::new(0), inner: parking_lot::Mutex::new(v) }
        }
        pub fn lock(&self) -> MutexGuard<'_, T> {
            let sim = sched::current().map(|ctx| {
                let id = object_id(&self.id);
                ctx.lock_acquire(id, true);
                (ctx, id)
            });
            MutexGuard { inner: Some(self.inner.lock()), lock: &self.inner, sim }
        }
        pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
            match sched::current() {
                Some(ctx) => {
                    let id = object_id(&self.id);
                    if !ctx.lock_try_acquire(id, true) {
                        return None;
                    }
                    Some(MutexGuard { inner: Some(self.inner.lock()), lock: &self.inner, sim: Some((ctx, id)) })
                }
                None => self.inner.try_lock().map(|g| MutexGuard { inner: Some(g), lock: &self.inner, sim: None }),
            }
        }
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.inner.as_ref().expect("the lock is held")
        }
    }
    impl<T> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            self.inner.as_mut().expect("the lock is held")
        }
    }
    impl<T> Drop for MutexGuard<'_, T> {
        fn drop(&mut self) {
            // The real lock goes first, so that a waiter the scheduler wakes finds it free.
            self.inner.take();
            if let Some((ctx, id)) = &self.sim {
                ctx.lock_release(*id, true);
            }
        }
    }

    pub struct RwLock<T> {
        id: IdCell,
        inner: parking_lot::RwLock<T>,
    }

    impl<T: Default> Default for RwLock<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    pub struct RwLockReadGuard<'a, T> {
        inner: Option<parking_lot::RwLockReadGuard<'a, T>>,
        sim: Option<(Ctx, u64)>,
    }
    pub struct RwLockWriteGuard<'a, T> {
        inner: Option<parking_lot::RwLockWriteGuard<'a, T>>,
        sim: Option<(Ctx, u64)>,
    }

    impl<T> RwLock<T> {
        pub fn new(v: T) -> Self {
            Self { id: IdCell::new(0), inner: parking_lot::RwLock::new(v) }
        }
        pub fn read(&self) -> RwLockReadGuard<'_, T> {
            let sim = sched::current().map(|ctx| {
                let id = object_id(&self.id);
                ctx.lock_acquire(id, false);
                (ctx, id)
            });
            RwLockReadGuard { inner: Some(self.inner.read()), sim }
        }
        pub fn write(&self) -> RwLockWriteGuard<'_, T> {
            let sim = sched::current().map(|ctx| {
                let id = object_id(&self.id);
                ctx.lock_acquire(id, true);
                (ctx, id)
            });
            RwLockWriteGuard { inner: Some(self.inner.write()), sim }
        }
        pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
            match sched::current() {
                Some(ctx) => {
                    let id = object_id(&self.id);
                    if !ctx.lock_try_acquire(id, true) {
                        return None;
                    }
                    Some(RwLockWriteGuard { inner: Some(self.inner.write()), sim: Some((ctx, id)) })
                }
                None => self.inner.try_write().map(|g| RwLockWriteGuard { inner: Some(g), sim: None }),
            }
        }
    }

    impl<T> Deref for RwLockReadGuard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.inner.as_ref().expect("the lock is held")
        }
    }
    impl<T> Drop for RwLockReadGuard<'_, T> {
        fn drop(&mut self) {
            self.inner.take();
            if let Some((ctx, id)) = &self.sim {
                ctx.lock_release(*id, false);
            }
        }
    }
    impl<T> Deref for RwLockWriteGuard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.inner.as_ref().expect("the lock is held")
        }
    }
    impl<T> DerefMut for RwLockWriteGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            self.inner.as_mut().expect("the lock is held")
        }
    }
    impl<T> Drop for RwLockWriteGuard<'_, T> {
        fn drop(&mut self) {
            self.inner.take();
            if let Some((ctx, id)) = &self.sim {
                ctx.lock_release(*id, true);
            }
        }
    }

    pub struct Condvar {
        id: IdCell,
        inner: parking_lot::Condvar,
    }

    impl Default for Condvar {
        fn default() -> Self {
            Self { id: IdCell::new(0), inner: parking_lot::Condvar::new() }
        }
    }

    impl Condvar {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn wait<T>(&self, g: &mut MutexGuard<'_, T>) {
            self.wait_inner(g, None);
        }
        /// Waits at most `d`. Returns true if the wait timed out. A timed wait is a scheduling point
        /// in simulation, where the duration is measured on the simulated clock.
        pub fn wait_for<T>(&self, g: &mut MutexGuard<'_, T>, d: Duration) -> bool {
            self.wait_inner(g, Some(d))
        }
        fn wait_inner<T>(&self, g: &mut MutexGuard<'_, T>, d: Option<Duration>) -> bool {
            match g.sim.clone() {
                Some((ctx, lock_id)) => {
                    // Release the real lock, let the scheduler release its own and park the task,
                    // then take both again.
                    g.inner.take();
                    let timed_out = ctx.cv_wait(object_id(&self.id), lock_id, d);
                    ctx.lock_acquire(lock_id, true);
                    g.inner = Some(g.lock.lock());
                    timed_out
                }
                None => {
                    let inner = g.inner.as_mut().expect("the lock is held");
                    match d {
                        Some(d) => self.inner.wait_for(inner, d).timed_out(),
                        None => {
                            self.inner.wait(inner);
                            false
                        }
                    }
                }
            }
        }
        pub fn notify_one(&self) {
            match sched::current() {
                Some(ctx) => ctx.cv_notify(object_id(&self.id), false),
                None => {
                    self.inner.notify_one();
                }
            }
        }
        pub fn notify_all(&self) {
            match sched::current() {
                Some(ctx) => ctx.cv_notify(object_id(&self.id), true),
                None => {
                    self.inner.notify_all();
                }
            }
        }
    }
}

pub use prim::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(not(loom))]
pub use prim::{Condvar, Mutex, MutexGuard};
#[cfg(loom)]
pub use loom_prim::{Condvar, Mutex, MutexGuard};

/// Under `--cfg loom` the mutex and the condition variable are loom's, so a model can cover a lock
/// and a wake-up (spec 18.5, model 4: lock grant and wake-up). Same calls as the others, nothing
/// more. Loom does not model time: a timed wait is a wait, and a lost wake-up shows up as a
/// deadlock, which is what such a model looks for. The reader-writer lock stays as it is: no model
/// needs it yet (B1, B5).
#[cfg(loom)]
mod loom_prim {
    use std::ops::{Deref, DerefMut};
    use std::time::Duration;

    pub struct Mutex<T> {
        inner: loom::sync::Mutex<T>,
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    /// `inner` is `None` only while a condition-variable wait holds it.
    pub struct MutexGuard<'a, T> {
        inner: Option<loom::sync::MutexGuard<'a, T>>,
    }

    impl<T> Mutex<T> {
        pub fn new(v: T) -> Self {
            Self { inner: loom::sync::Mutex::new(v) }
        }
        pub fn lock(&self) -> MutexGuard<'_, T> {
            MutexGuard { inner: Some(self.inner.lock().expect("loom mutexes do not poison")) }
        }
        pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
            self.inner.try_lock().ok().map(|g| MutexGuard { inner: Some(g) })
        }
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.inner.as_ref().expect("the lock is held")
        }
    }
    impl<T> DerefMut for MutexGuard<'_, T> {
        fn deref_mut(&mut self) -> &mut T {
            self.inner.as_mut().expect("the lock is held")
        }
    }

    #[derive(Default)]
    pub struct Condvar {
        inner: loom::sync::Condvar,
    }

    impl Condvar {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn wait<T>(&self, g: &mut MutexGuard<'_, T>) {
            let guard = g.inner.take().expect("the lock is held");
            g.inner = Some(self.inner.wait(guard).expect("loom mutexes do not poison"));
        }
        /// A wait: loom does not model time, so it never reports a timeout.
        pub fn wait_for<T>(&self, g: &mut MutexGuard<'_, T>, d: Duration) -> bool {
            let guard = g.inner.take().expect("the lock is held");
            let (guard, r) = self.inner.wait_timeout(guard, d).expect("loom mutexes do not poison");
            g.inner = Some(guard);
            r.timed_out()
        }
        pub fn notify_one(&self) {
            self.inner.notify_one();
        }
        pub fn notify_all(&self) {
            self.inner.notify_all();
        }
    }
}

/// Named threads and sleeping. Sleeps and timed waits go through here so that a simulated variant
/// can treat them as scheduling points on the simulated clock (answer A3). In a simulation a spawned
/// thread is a task of the scheduler, and `sleep` takes virtual time.
pub mod thread {
    use crate::sched::{self, SimJoin};
    use std::io;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::time::Duration;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ThreadId(std::thread::ThreadId);

    pub fn current_id() -> ThreadId {
        ThreadId(std::thread::current().id())
    }

    #[allow(clippy::disallowed_methods)]
    pub fn sleep(d: Duration) {
        match sched::current() {
            Some(ctx) => ctx.sleep(d),
            None => std::thread::sleep(d),
        }
    }

    pub fn yield_now() {
        match sched::current() {
            Some(ctx) => ctx.yield_now(),
            None => std::thread::yield_now(),
        }
    }

    /// The thread panicked. The panic itself went to the handler given to [`spawn_named`]; this
    /// only carries the fact, so that a caller can tell a clean stop from a failed one.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ThreadPanicked;

    impl std::fmt::Display for ThreadPanicked {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("thread panicked")
        }
    }

    impl std::error::Error for ThreadPanicked {}

    enum Handle {
        /// The thread's result is whether its body panicked: `spawn_named` catches the panic inside
        /// the thread, so `std`'s own `join` would never see it.
        Real(std::thread::JoinHandle<bool>),
        Sim(SimJoin),
    }

    pub struct JoinHandle(Handle);

    impl JoinHandle {
        pub fn id(&self) -> ThreadId {
            match &self.0 {
                Handle::Real(h) => ThreadId(h.thread().id()),
                Handle::Sim(j) => ThreadId(j.thread_id()),
            }
        }
        /// Waits for the thread. `Err` if its body panicked (`std` and loom return a result here
        /// too). In a simulation the wait is a scheduling point.
        pub fn join(self) -> Result<(), ThreadPanicked> {
            let panicked = match self.0 {
                Handle::Real(h) => h.join().unwrap_or(true),
                Handle::Sim(j) => j.join(sched::current()),
            };
            if panicked {
                Err(ThreadPanicked)
            } else {
                Ok(())
            }
        }
    }

    /// Spawns a named background thread whose panic is handed to `on_panic` instead of being lost
    /// (spec 18.5: a panic quarantines the store). Every engine thread starts here.
    #[allow(clippy::disallowed_methods)]
    pub fn spawn_named<F, P>(name: &str, on_panic: P, f: F) -> io::Result<JoinHandle>
    where
        F: FnOnce() + Send + 'static,
        P: FnOnce(String) + Send + 'static,
    {
        let owned = name.to_string();
        let body = move || match catch_unwind(AssertUnwindSafe(f)) {
            Ok(()) => false,
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                on_panic(format!("thread {owned} panicked: {msg}"));
                true
            }
        };
        if let Some(ctx) = sched::current() {
            return Ok(JoinHandle(Handle::Sim(ctx.spawn(name, Box::new(body))?)));
        }
        let h = std::thread::Builder::new().name(name.to_string()).spawn(body)?;
        Ok(JoinHandle(Handle::Real(h)))
    }
}

/// Debug-build latch-order checking (spec 13.6, rule B). Release builds compile it to nothing.
///
/// Page latches announce their [`Class`] when taken. The checker rejects the reverse of every
/// order that section 13.6 prescribes, and `assert_none_held` implements rule A: no thread waits
/// on the lock manager, a client or a commit flush while it holds a latch.
pub mod latch {
    /// What kind of page a latch protects. `Unordered` is for tests and raw pages: it is counted
    /// for rule A but never order-checked.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Class {
        Unordered,
        /// B-tree data page; `level` 0 is a leaf.
        Btree { level: u8 },
        BtreeMeta,
        Heap,
        /// Visibility-map, overflow and commit-log pages (rule 6).
        Aux,
    }

    /// Identifies a page within its class: (relation file, block).
    pub type Key = (u64, u32);

    #[cfg(debug_assertions)]
    thread_local! {
        static HELD: std::cell::RefCell<Vec<(Class, Key)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Proof that a latch is registered; dropping it unregisters.
    pub struct Ticket {
        #[cfg(debug_assertions)]
        entry: (Class, Key),
    }

    #[cfg(debug_assertions)]
    fn violates(held: (Class, Key), new: (Class, Key)) -> bool {
        use Class::*;
        match (held.0, new.0) {
            (Unordered, _) | (_, Unordered) => false,
            (Btree { level: a }, Btree { level: b }) => b < a,
            (BtreeMeta, Btree { .. }) => true,
            (Heap, Btree { .. } | BtreeMeta) => true,
            (Heap, Heap) => held.1 .0 == new.1 .0 && new.1 .1 <= held.1 .1,
            (Aux, Heap | Btree { .. } | BtreeMeta) => true,
            _ => false,
        }
    }

    /// Registers a latch about to be taken. Panics (debug builds) on an order violation.
    pub fn acquire(class: Class, key: Key) -> Ticket {
        #[cfg(debug_assertions)]
        {
            let entry = (class, key);
            HELD.with(|h| {
                let mut h = h.borrow_mut();
                for &held in h.iter() {
                    assert!(!violates(held, entry), "latch order violation: {entry:?} requested while holding {held:?}");
                }
                h.push(entry);
            });
            Ticket { entry }
        }
        #[cfg(not(debug_assertions))]
        {
            let _ = (class, key);
            Ticket {}
        }
    }

    impl Drop for Ticket {
        fn drop(&mut self) {
            #[cfg(debug_assertions)]
            HELD.with(|h| {
                let mut h = h.borrow_mut();
                if let Some(i) = h.iter().rposition(|e| *e == self.entry) {
                    h.remove(i);
                }
            });
        }
    }

    /// Rule A. Call before any wait on the lock manager, a client or a commit flush.
    pub fn assert_none_held(what: &str) {
        #[cfg(debug_assertions)]
        HELD.with(|h| {
            let h = h.borrow();
            assert!(h.is_empty(), "{what} while holding page latches {:?}", *h);
        });
        #[cfg(not(debug_assertions))]
        let _ = what;
    }

    /// Number of latches the calling thread holds (always 0 in release builds).
    pub fn held_count() -> usize {
        #[cfg(debug_assertions)]
        return HELD.with(|h| h.borrow().len());
        #[cfg(not(debug_assertions))]
        0
    }
}

#[cfg(test)]
mod prim_tests {
    use super::thread;
    use super::*;
    use std::time::Duration;

    #[test]
    fn condvar_times_out_and_wakes() {
        let m = Arc::new(Mutex::new(false));
        let cv = Arc::new(Condvar::new());
        {
            let mut g = m.lock();
            assert!(cv.wait_for(&mut g, Duration::from_millis(5)), "nobody notifies: the wait times out");
        }
        let (m2, cv2) = (m.clone(), cv.clone());
        let h = thread::spawn_named("t-notify", |_| {}, move || {
            *m2.lock() = true;
            cv2.notify_all();
        })
        .unwrap();
        let mut g = m.lock();
        while !*g {
            cv.wait_for(&mut g, Duration::from_millis(50));
        }
        drop(g);
        assert_eq!(h.join(), Ok(()), "a thread that finishes normally joins cleanly");
    }

    #[test]
    fn the_protocol_ordering_is_seq_cst() {
        assert_eq!(atomic::STORE_BEFORE_LOAD, atomic::Ordering::SeqCst);
    }

    #[test]
    fn rwlock_allows_many_readers_and_one_writer() {
        let l = RwLock::new(5u32);
        let (a, b) = (l.read(), l.read());
        assert_eq!((*a, *b), (5, 5));
        assert!(l.try_write().is_none());
        drop((a, b));
        *l.write() += 1;
        assert_eq!(*l.read(), 6);
    }

    #[test]
    fn a_panicking_thread_reports_to_its_handler() {
        let seen = Arc::new(Mutex::new(String::new()));
        let s2 = seen.clone();
        let h = thread::spawn_named("t-panics", move |m| *s2.lock() = m, || panic!("boom")).unwrap();
        assert_eq!(h.join(), Err(thread::ThreadPanicked), "join must report the panic, not swallow it");
        let msg = seen.lock().clone();
        assert!(msg.contains("t-panics") && msg.contains("boom"), "{msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::latch::*;

    #[test]
    fn child_before_parent_is_allowed_and_the_reverse_is_not() {
        let _leaf = acquire(Class::Btree { level: 0 }, (7, 1));
        let _parent = acquire(Class::Btree { level: 1 }, (7, 2));
        assert_eq!(held_count(), if cfg!(debug_assertions) { 2 } else { 0 });
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "latch order violation")]
    fn parent_then_child_panics() {
        let _parent = acquire(Class::Btree { level: 1 }, (7, 2));
        let _leaf = acquire(Class::Btree { level: 0 }, (7, 1));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "latch order violation")]
    fn heap_blocks_must_ascend() {
        let _a = acquire(Class::Heap, (9, 5));
        let _b = acquire(Class::Heap, (9, 3));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "while holding page latches")]
    fn waiting_with_a_latch_panics() {
        let _a = acquire(Class::Heap, (9, 5));
        assert_none_held("commit flush");
    }

    #[test]
    fn released_latches_leave_no_trace() {
        {
            let _a = acquire(Class::Aux, (2, 0));
        }
        assert_none_held("commit flush");
    }
}
