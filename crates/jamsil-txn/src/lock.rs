//! The lock manager (spec 13): table locks in five modes, transaction-id locks, shared row locks and
//! read gates, held until transaction end, with first-in-first-out wait queues, a timeout,
//! cancellation, deadlock detection, and wounding for apply priority.
//!
//! The lock table has 64 partitions by tag hash. Each is a mutex around a hash map of entries, and
//! each entry holds its granted list and its wait queue. A transaction holds an exclusive lock on its
//! own xid from assignment to end; "wait for X" is a shared request on xid X, released as soon as it is
//! granted (13.4).
//!
//! Not built yet: the fast path of 13.2 (a SHOULD), the per-table counter of shared row locks, and the
//! trade of 10,000 shared row locks for one table lock (13.3).

use jamsil_page::*;
use jamsil_vfs::env::Clock;
use jamsil_vfs::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, STORE_BEFORE_LOAD};
use jamsil_vfs::sync::{Arc, Condvar, Mutex};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

const PARTITIONS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockTag {
    Table(Oid),
    Xid(Xid),
    Row(Oid, Tid),
    ReadGate(Oid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    // Table modes (spec 13.2).
    AccessShare,
    RowExclusive,
    Share,
    ShareUpdateExclusive,
    AccessExclusive,
    // The modes of the other tags: a holder of an xid or a row stamp is exclusive; a waiter for it or
    // a shared row locker is shared. A reader that registers a read gate holds it `Shared`.
    Shared,
    Exclusive,
    /// A writer waiting for the holders of a read gate to leave (spec 12.6 step 3). It waits for the
    /// `Shared` holders, but it is never granted anything and never keeps a reader from registering:
    /// registration happens inside the publication lock and must not block (spec 13.3).
    GateWait,
}

impl LockMode {
    /// Whether two modes conflict. The table is the one of spec 13.2.
    pub fn conflicts(self, other: LockMode) -> bool {
        use LockMode::*;
        match (self, other) {
            (Exclusive, _) | (_, Exclusive) => true,
            (GateWait, Shared) | (Shared, GateWait) => true,
            (GateWait, _) | (_, GateWait) => false,
            (Shared, _) | (_, Shared) => false,
            (AccessExclusive, _) | (_, AccessExclusive) => true,
            (AccessShare, _) | (_, AccessShare) => false,
            (RowExclusive, Share) | (Share, RowExclusive) => true,
            (RowExclusive, _) | (_, RowExclusive) => false,
            (Share, ShareUpdateExclusive) | (ShareUpdateExclusive, Share) => true,
            (ShareUpdateExclusive, ShareUpdateExclusive) => true,
            (Share, Share) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockError {
    /// `55P03`: the lock timeout passed, or `NoWait` found the lock taken.
    Timeout,
    /// `40P01`: the waiter is part of a cycle of waits.
    Deadlock,
    /// `40001`: apply needed this lock and wounded its holder.
    Wounded,
    /// `57014`: the session's cancel flag was set.
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitPolicy {
    NoWait,
    Timeout(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WoundReason {
    /// A replicated entry needs the lock (spec 13.7): the holder is aborted with `JDB-2002`.
    ApplyPriority,
}

struct OwnerState {
    id: u64,
    xid: AtomicU64,
    cancel: AtomicBool,
    wounded: AtomicBool,
    /// The partition this owner is waiting in, or `usize::MAX`.
    waiting_in: AtomicUsize,
    held: Mutex<Vec<(LockTag, LockMode)>>,
}

/// The identity of a session or transaction as the lock manager sees it. Clones share one state.
#[derive(Clone)]
pub struct LockOwner(Arc<OwnerState>);

impl LockOwner {
    pub fn new(id: u64) -> Self {
        Self(Arc::new(OwnerState {
            id,
            xid: AtomicU64::new(0),
            cancel: AtomicBool::new(false),
            wounded: AtomicBool::new(false),
            waiting_in: AtomicUsize::new(usize::MAX),
            held: Mutex::new(Vec::new()),
        }))
    }
    pub fn id(&self) -> u64 {
        self.0.id
    }
    /// Records the owner's xid once it has one, so that `wound` can name it.
    pub fn set_xid(&self, xid: Xid) {
        self.0.xid.store(xid, STORE_BEFORE_LOAD);
    }
    pub fn is_wounded(&self) -> bool {
        self.0.wounded.load(STORE_BEFORE_LOAD)
    }
    /// The session's cancel flag is set (`CancelQuery`, `57014`).
    pub fn is_cancelled(&self) -> bool {
        self.0.cancel.load(STORE_BEFORE_LOAD)
    }
    /// A mark of the locks held now, for `unlock_since` (spec 12.9: a lock mark at `SAVEPOINT`).
    pub fn mark(&self) -> LockMark {
        LockMark(self.0.held.lock().len())
    }
    pub fn holds(&self, tag: LockTag, mode: LockMode) -> bool {
        self.0.held.lock().contains(&(tag, mode))
    }
}

impl std::fmt::Debug for LockOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LockOwner({})", self.0.id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockMark(usize);

struct Grant {
    owner: Arc<OwnerState>,
    mode: LockMode,
}

struct Waiter {
    owner: Arc<OwnerState>,
    mode: LockMode,
}

#[derive(Default)]
struct Entry {
    granted: Vec<Grant>,
    queue: VecDeque<Waiter>,
}

struct Partition {
    entries: Mutex<HashMap<LockTag, Entry>>,
    cv: Condvar,
}

pub struct LockManager {
    parts: Vec<Partition>,
    clock: Arc<dyn Clock>,
    deadlock_timeout: Duration,
}

fn hash(tag: &LockTag) -> usize {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut h);
    h.finish() as usize % PARTITIONS
}

impl LockManager {
    /// `deadlock_timeout`: how long a waiter sleeps before it searches for a cycle
    /// (`storage.deadlock_timeout_ms`, default 1,000).
    pub fn new(clock: Arc<dyn Clock>, deadlock_timeout: Duration) -> Self {
        let parts = (0..PARTITIONS).map(|_| Partition { entries: Mutex::new(HashMap::new()), cv: Condvar::new() }).collect();
        Self { parts, clock, deadlock_timeout }
    }

    fn now(&self) -> u64 {
        self.clock.now_micros()
    }

    /// A request is granted now if it conflicts with nothing granted to another owner and with no
    /// waiter already in the queue: a request does not jump ahead of a conflicting waiter, so
    /// `AccessExclusive` cannot starve (spec 13.2).
    fn grantable_now(e: &Entry, owner: &OwnerState, mode: LockMode) -> bool {
        e.granted.iter().all(|g| g.owner.id == owner.id || !g.mode.conflicts(mode))
            && e.queue.iter().all(|w| w.owner.id == owner.id || w.mode == LockMode::GateWait || !w.mode.conflicts(mode))
    }

    /// A queued request is granted when it conflicts with nothing granted and with no waiter ahead
    /// of it.
    fn grantable_waiter(e: &Entry, owner: &OwnerState, mode: LockMode) -> bool {
        if !e.granted.iter().all(|g| g.owner.id == owner.id || !g.mode.conflicts(mode)) {
            return false;
        }
        for w in &e.queue {
            if w.owner.id == owner.id {
                return true;
            }
            if w.mode != LockMode::GateWait && w.mode.conflicts(mode) {
                return false;
            }
        }
        true
    }

    /// Takes a lock, waiting for it if `wait` allows. A wait ends with `Timeout` after the timeout,
    /// with `Cancelled` when the owner's cancel flag is set, with `Wounded` when apply needs the
    /// lock, and with `Deadlock` when the owner's search, started after `deadlock_timeout`, finds it
    /// in a cycle of waits (spec 13.4, 13.5).
    pub fn lock(&self, owner: &LockOwner, tag: LockTag, mode: LockMode, wait: WaitPolicy) -> Result<(), LockError> {
        let st = &owner.0;
        let pi = hash(&tag);
        let part = &self.parts[pi];
        let mut g = part.entries.lock();
        if st.wounded.load(STORE_BEFORE_LOAD) {
            return Err(LockError::Wounded);
        }
        let e = g.entry(tag).or_default();
        // Taking again a lock the owner already holds in the same mode is granted at once. Queueing
        // behind a conflicting waiter would wait for a waiter that is itself waiting for this owner.
        let held_already = e.granted.iter().any(|x| x.owner.id == st.id && x.mode == mode);
        if held_already || Self::grantable_now(e, st, mode) {
            if mode != LockMode::GateWait {
                e.granted.push(Grant { owner: st.clone(), mode });
                st.held.lock().push((tag, mode));
            } else if e.granted.is_empty() && e.queue.is_empty() {
                // A gate wait is granted nothing, so the entry made for it must not stay behind.
                g.remove(&tag);
            }
            return Ok(());
        }
        let WaitPolicy::Timeout(timeout) = wait else {
            return Err(LockError::Timeout);
        };
        e.queue.push_back(Waiter { owner: st.clone(), mode });
        st.waiting_in.store(pi, STORE_BEFORE_LOAD);
        // The waiter stores `waiting_in` and then loads the cancel and wound flags; a canceller stores
        // its flag and then loads `waiting_in` (`cancel`, `wound`). Both sides are `SeqCst`, so one of
        // them sees the other. Loom treats `SeqCst` as `AcqRel` (false alarms), but models
        // `fence(SeqCst)` exactly: under loom only, the fence lets the model check the same guarantee
        // (as `jamsil_bufmgr::dirty_flag` does).
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD);
        let start = self.now();
        let deadline = start + timeout.as_micros() as u64;
        let mut next_search = start + self.deadlock_timeout.as_micros() as u64;
        let result = loop {
            if st.wounded.load(STORE_BEFORE_LOAD) {
                break Err(LockError::Wounded);
            }
            if st.cancel.load(STORE_BEFORE_LOAD) {
                break Err(LockError::Cancelled);
            }
            let e = g.get(&tag).expect("a waiter keeps its entry alive");
            if Self::grantable_waiter(e, st, mode) {
                break Ok(());
            }
            let now = self.now();
            if now >= deadline {
                break Err(LockError::Timeout);
            }
            if now >= next_search {
                // Search without holding the partition: the search takes every partition in turn.
                drop(g);
                let cycle = self.in_a_cycle(st.id);
                g = part.entries.lock();
                next_search = self.now() + self.deadlock_timeout.as_micros() as u64;
                if cycle {
                    break Err(LockError::Deadlock);
                }
                continue;
            }
            let wake = deadline.min(next_search);
            part.cv.wait_for(&mut g, Duration::from_micros(wake - now));
        };
        let e = g.get_mut(&tag).expect("a waiter keeps its entry alive");
        e.queue.retain(|w| w.owner.id != st.id);
        st.waiting_in.store(usize::MAX, STORE_BEFORE_LOAD);
        match result {
            Ok(()) => {
                if mode != LockMode::GateWait {
                    e.granted.push(Grant { owner: st.clone(), mode });
                    st.held.lock().push((tag, mode));
                } else if e.granted.is_empty() && e.queue.is_empty() {
                    g.remove(&tag);
                }
            }
            Err(_) => {
                if e.granted.is_empty() && e.queue.is_empty() {
                    g.remove(&tag);
                }
            }
        }
        // Someone behind this waiter may be grantable now.
        part.cv.notify_all();
        result
    }

    /// Releases the lock `(tag, mode)` held by `owner_id` and wakes the waiters of its partition.
    fn release(&self, owner_id: u64, tag: LockTag, mode: LockMode) {
        let part = &self.parts[hash(&tag)];
        {
            let mut g = part.entries.lock();
            if let Some(e) = g.get_mut(&tag) {
                if let Some(i) = e.granted.iter().position(|x| x.owner.id == owner_id && x.mode == mode) {
                    e.granted.remove(i);
                }
                if e.granted.is_empty() && e.queue.is_empty() {
                    g.remove(&tag);
                }
            }
        }
        part.cv.notify_all();
    }

    /// Releases one lock the owner holds, for example a read gate when it is not needed any more.
    pub fn unlock(&self, owner: &LockOwner, tag: LockTag, mode: LockMode) {
        let at = owner.0.held.lock().iter().rposition(|h| *h == (tag, mode));
        if let Some(i) = at {
            owner.0.held.lock().remove(i);
            self.release(owner.0.id, tag, mode);
        }
    }

    /// Releases every lock of the owner (transaction end).
    pub fn unlock_all(&self, owner: &LockOwner) {
        let held = std::mem::take(&mut *owner.0.held.lock());
        for (tag, mode) in held.into_iter().rev() {
            self.release(owner.0.id, tag, mode);
        }
        owner.0.wounded.store(false, STORE_BEFORE_LOAD);
        owner.0.cancel.store(false, STORE_BEFORE_LOAD);
    }

    /// Releases the locks taken after `mark` and keeps the earlier ones (`ROLLBACK TO SAVEPOINT`).
    pub fn unlock_since(&self, owner: &LockOwner, mark: LockMark) {
        let later: Vec<_> = {
            let mut held = owner.0.held.lock();
            if mark.0 >= held.len() {
                return;
            }
            held.split_off(mark.0)
        };
        for (tag, mode) in later.into_iter().rev() {
            self.release(owner.0.id, tag, mode);
        }
    }

    /// "Wait for X" is a shared request on xid X, released as soon as it is granted (spec 13.4).
    pub fn wait_for_xid(&self, waiter: &LockOwner, xid: Xid, timeout: Duration) -> Result<(), LockError> {
        let tag = LockTag::Xid(xid);
        self.lock(waiter, tag, LockMode::Shared, WaitPolicy::Timeout(timeout))?;
        self.unlock(waiter, tag, LockMode::Shared);
        Ok(())
    }

    /// Sets the owner's cancel flag and wakes it if it is waiting (`CancelQuery`, 57014).
    pub fn cancel(&self, owner: &LockOwner) {
        owner.0.cancel.store(true, STORE_BEFORE_LOAD);
        #[cfg(loom)]
        jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD); // see `lock`
        self.wake(owner);
    }

    fn wake(&self, owner: &LockOwner) {
        let pi = owner.0.waiting_in.load(STORE_BEFORE_LOAD);
        if pi != usize::MAX {
            // Take the partition lock so that the flag is seen by a waiter that is about to sleep.
            let _g = self.parts[pi].entries.lock();
            self.parts[pi].cv.notify_all();
        }
    }

    /// Apply priority (spec 13.7): wounds every holder and waiter of the tag, so that their next
    /// lock call or their current wait fails with `Wounded`, and returns their xids.
    pub fn wound(&self, tag: LockTag, _reason: WoundReason) -> Vec<Xid> {
        let part = &self.parts[hash(&tag)];
        let victims: Vec<Arc<OwnerState>> = {
            let g = part.entries.lock();
            match g.get(&tag) {
                Some(e) => e.granted.iter().map(|x| x.owner.clone()).chain(e.queue.iter().map(|w| w.owner.clone())).collect(),
                None => Vec::new(),
            }
        };
        let mut xids = Vec::new();
        for v in victims {
            v.wounded.store(true, STORE_BEFORE_LOAD);
            #[cfg(loom)]
            jamsil_vfs::sync::atomic::fence(STORE_BEFORE_LOAD); // see `lock`
            let x = v.xid.load(STORE_BEFORE_LOAD);
            if x != 0 && !xids.contains(&x) {
                xids.push(x);
            }
            self.wake(&LockOwner(v));
        }
        xids
    }

    /// Searches the wait-for graph from `start`: does a chain of waits lead back to it? An owner
    /// waits for the other owners that hold a conflicting lock on its tag and for the conflicting
    /// waiters ahead of it. Takes every partition, in order, so the graph is one consistent picture.
    fn in_a_cycle(&self, start: u64) -> bool {
        let guards: Vec<_> = self.parts.iter().map(|p| p.entries.lock()).collect();
        let mut edges: HashMap<u64, Vec<u64>> = HashMap::new();
        for g in &guards {
            for e in g.values() {
                for (i, w) in e.queue.iter().enumerate() {
                    let out = edges.entry(w.owner.id).or_default();
                    for h in &e.granted {
                        if h.owner.id != w.owner.id && h.mode.conflicts(w.mode) {
                            out.push(h.owner.id);
                        }
                    }
                    for ahead in e.queue.iter().take(i) {
                        // A `GateWait` waiter stands in nobody's way (see `grantable_waiter`).
                        if ahead.owner.id != w.owner.id && ahead.mode != LockMode::GateWait && ahead.mode.conflicts(w.mode) {
                            out.push(ahead.owner.id);
                        }
                    }
                }
            }
        }
        // Depth-first from `start`: a cycle through it is a path back to it.
        let mut seen = std::collections::HashSet::new();
        let mut stack: Vec<u64> = edges.get(&start).cloned().unwrap_or_default();
        while let Some(n) = stack.pop() {
            if n == start {
                return true;
            }
            if seen.insert(n) {
                stack.extend(edges.get(&n).cloned().unwrap_or_default());
            }
        }
        false
    }

    /// Number of entries in the lock table (tests).
    pub fn entries(&self) -> usize {
        self.parts.iter().map(|p| p.entries.lock().len()).sum()
    }
}

#[cfg(test)]
mod tests {
    /// The wake-up of a cancel or a wound against a waiter that is about to sleep is a store-before-load
    /// protocol (see `lock`): the canceller stores its flag and loads `waiting_in`, the waiter stores
    /// `waiting_in` and loads the flag. Only `SeqCst` guarantees that one of them sees the other, so every
    /// access to the three words uses `STORE_BEFORE_LOAD`, and nobody can weaken one without this test
    /// noticing while the loom model keeps passing on its fence. (`jamsil_vfs::sync::atomic` pins the
    /// constant itself.)
    #[test]
    fn the_cancel_wound_and_waiting_words_use_the_protocol_ordering() {
        let src = include_str!("lock.rs");
        let code = src.split("#[cfg(test)]").next().unwrap();
        let mut checked = 0;
        for (n, line) in code.lines().enumerate() {
            let touches = [".cancel.", ".wounded.", ".waiting_in."].iter().any(|w| line.contains(w));
            let is_op = [".store(", ".load(", ".swap(", ".fetch_"].iter().any(|o| line.contains(o));
            if touches && is_op && !line.trim_start().starts_with("//") {
                checked += 1;
                assert!(line.contains("STORE_BEFORE_LOAD"), "line {}: {line}", n + 1);
            }
        }
        assert!(checked >= 8, "the test found the accesses it is about: {checked}");
        assert!(code.matches("fence(STORE_BEFORE_LOAD)").count() >= 3, "the loom fences after the three stores");
    }
}
