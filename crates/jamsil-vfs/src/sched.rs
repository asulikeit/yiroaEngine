//! The deterministic scheduler behind the `sync` facade (spec 19.2).
//!
//! A simulated run is a set of tasks. Each task is a real OS thread, but only the task that holds
//! the baton runs: the others are parked. Control passes at scheduling points only, which are the
//! points where engine code touches the facade: acquiring a lock, waiting on or notifying a
//! condition variable, spawning, joining, sleeping and `yield_now`. At each one a seeded generator
//! picks the next task among the runnable ones, so a run is a pure function of its seed.
//!
//! Time is virtual. It advances only when no task can run, and then it jumps to the earliest
//! deadline of a sleeper or a timed wait. `sleep(200 ms)` therefore takes no real time, and the WAL
//! writer's 200 ms bound can be tested (spec 8.5).
//!
//! If no task can run and nothing waits for a deadline, the run fails with a report of every task
//! and what it waits for: that is how a lock-order deadlock or a lost wake-up shows up.
//!
//! Production pays one relaxed atomic load per facade operation, to find out that no simulation is
//! running at all.
// Under `--cfg loom` the facade's mutex and condition variable are loom's, which leaves the
// scheduler's lock and condition-variable support unused.
#![cfg_attr(loom, allow(dead_code))]

#![allow(clippy::disallowed_types)]

use crate::env::Clock;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar as StdCondvar, Mutex as StdMutex, MutexGuard as StdGuard};
use std::time::Duration;

/// Number of simulations running in this process. Zero is the production fast path.
static ACTIVE_SIMS: AtomicUsize = AtomicUsize::new(0);
/// Ids for locks and condition variables, unique in the process.
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn new_object_id() -> u64 {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    static CTX: RefCell<Option<Ctx>> = const { RefCell::new(None) };
}

#[derive(Clone)]
pub(crate) struct Ctx {
    shared: Arc<Shared>,
    task: usize,
}

/// The context of the calling thread if it is a task of a running simulation.
pub(crate) fn current() -> Option<Ctx> {
    if ACTIVE_SIMS.load(Ordering::Relaxed) == 0 {
        return None;
    }
    CTX.with(|c| c.borrow().clone())
}

/// True if the calling thread is a task of a running simulation.
pub fn in_simulation() -> bool {
    current().is_some()
}

struct Parker {
    flag: StdMutex<bool>,
    cv: StdCondvar,
}

impl Parker {
    fn new() -> Arc<Self> {
        Arc::new(Self { flag: StdMutex::new(false), cv: StdCondvar::new() })
    }
    fn park(&self) {
        let mut f = self.flag.lock().unwrap();
        while !*f {
            f = self.cv.wait(f).unwrap();
        }
        *f = false;
    }
    fn unpark(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_one();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Block {
    Lock(u64),
    Cond(u64),
    Sleep,
    Join(usize),
    /// Held at a named suspension point until released.
    Point(String),
    /// A driver waiting for some task to reach a named point.
    Await(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Status {
    Runnable,
    Blocked(Block),
    Finished,
}

struct Task {
    name: String,
    status: Status,
    /// Virtual time at which a blocked task wakes by itself (sleeps and timed waits).
    wake_at: Option<u64>,
    parker: Arc<Parker>,
    panicked: bool,
}

#[derive(Default)]
struct LockState {
    writer: Option<usize>,
    readers: usize,
    waiters: Vec<usize>,
}

struct State {
    rng: u64,
    now_us: u64,
    steps: u64,
    tasks: Vec<Task>,
    locks: HashMap<u64, LockState>,
    /// Waiters of each condition variable, in arrival order.
    cvs: HashMap<u64, Vec<usize>>,
    holds: HashSet<String>,
    trace_hash: u64,
    failure: Option<String>,
}

struct Shared {
    st: StdMutex<State>,
    seed: u64,
}

fn splitmix(rng: &mut u64) -> u64 {
    *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *rng;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Shared {
    fn lock(&self) -> StdGuard<'_, State> {
        // A panic of a task never leaves the state half-updated: transitions are short and total.
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Passes the baton. `me` has already been marked blocked or left runnable by the caller. Picks
    /// the next task at random among the runnable ones, advancing virtual time if there are none,
    /// then parks `me` until it is picked again. Returns once `me` runs again; panics if the run
    /// failed meanwhile.
    fn reschedule<'a>(&'a self, mut st: StdGuard<'a, State>, me: usize) {
        loop {
            if let Some(f) = &st.failure {
                let f = f.clone();
                drop(st);
                panic!("{f}");
            }
            let runnable: Vec<usize> =
                st.tasks.iter().enumerate().filter(|(_, t)| t.status == Status::Runnable).map(|(i, _)| i).collect();
            if !runnable.is_empty() {
                let pick = runnable[(splitmix(&mut st.rng) % runnable.len() as u64) as usize];
                st.steps += 1;
                st.trace_hash = (st.trace_hash ^ (pick as u64 + 1)).wrapping_mul(0x0000_0100_0000_01B3);
                if pick == me {
                    return;
                }
                let parker = st.tasks[pick].parker.clone();
                let mine = st.tasks[me].parker.clone();
                drop(st);
                parker.unpark();
                mine.park();
                st = self.lock();
                if let Some(f) = &st.failure {
                    let f = f.clone();
                    drop(st);
                    panic!("{f}");
                }
                // The task that woke us made us runnable and picked us: we hold the baton.
                return;
            }
            // Nothing can run: jump to the earliest deadline.
            let next = st.tasks.iter().filter(|t| matches!(t.status, Status::Blocked(_))).filter_map(|t| t.wake_at).min();
            match next {
                Some(t) => {
                    st.now_us = st.now_us.max(t);
                    let now = st.now_us;
                    for task in st.tasks.iter_mut() {
                        if matches!(task.status, Status::Blocked(_)) && task.wake_at.is_some_and(|w| w <= now) {
                            task.status = Status::Runnable;
                            task.wake_at = None;
                        }
                    }
                }
                None => {
                    let report = Self::report(&st);
                    st.failure = Some(report.clone());
                    let parkers: Vec<_> = st.tasks.iter().map(|t| t.parker.clone()).collect();
                    drop(st);
                    for p in parkers {
                        p.unpark();
                    }
                    panic!("{report}");
                }
            }
        }
    }

    fn report(st: &State) -> String {
        let mut s = format!("simulation deadlock at virtual time {} us: no task can run and none waits for a deadline\n", st.now_us);
        for (i, t) in st.tasks.iter().enumerate() {
            s.push_str(&format!("  task {i} {:?}: {:?}\n", t.name, t.status));
        }
        s
    }

    fn block(&self, me: usize, why: Block, wake_at: Option<u64>) {
        let mut st = self.lock();
        st.tasks[me].status = Status::Blocked(why);
        st.tasks[me].wake_at = wake_at;
        self.reschedule(st, me);
    }

    fn yield_now(&self, me: usize) {
        let st = self.lock();
        self.reschedule(st, me);
    }

    fn make_runnable(st: &mut State, task: usize) {
        st.tasks[task].status = Status::Runnable;
        st.tasks[task].wake_at = None;
    }

    // ---- locks -----------------------------------------------------------------------------

    /// Takes the simulated lock `id` for `me`, blocking while another task holds it. A scheduling
    /// point comes first, so the order in which tasks reach a lock is explored.
    fn lock_acquire(&self, me: usize, id: u64, write: bool) {
        self.yield_now(me);
        loop {
            let mut st = self.lock();
            let ls = st.locks.entry(id).or_default();
            let free = if write { ls.writer.is_none() && ls.readers == 0 } else { ls.writer.is_none() };
            if free {
                if write {
                    ls.writer = Some(me);
                } else {
                    ls.readers += 1;
                }
                return;
            }
            ls.waiters.push(me);
            st.tasks[me].status = Status::Blocked(Block::Lock(id));
            st.tasks[me].wake_at = None;
            self.reschedule(st, me);
        }
    }

    fn lock_try_acquire(&self, me: usize, id: u64, write: bool) -> bool {
        let mut st = self.lock();
        let ls = st.locks.entry(id).or_default();
        let free = if write { ls.writer.is_none() && ls.readers == 0 } else { ls.writer.is_none() };
        if free {
            if write {
                ls.writer = Some(me);
            } else {
                ls.readers += 1;
            }
        }
        free
    }

    /// Releases the simulated lock and makes its waiters runnable; they retry in an order the
    /// scheduler picks. Never a scheduling point, because it runs from `Drop`.
    fn lock_release(&self, id: u64, write: bool) {
        let mut st = self.lock();
        let waiters = {
            let ls = st.locks.entry(id).or_default();
            if write {
                ls.writer = None;
            } else {
                ls.readers = ls.readers.saturating_sub(1);
            }
            std::mem::take(&mut ls.waiters)
        };
        for w in waiters {
            if st.tasks[w].status == Status::Blocked(Block::Lock(id)) {
                Self::make_runnable(&mut st, w);
            }
        }
    }

    // ---- condition variables ---------------------------------------------------------------

    /// Waits on `cv` after releasing the simulated lock `lock`; returns true if the wait timed out.
    /// The caller re-takes the lock.
    fn cv_wait(&self, me: usize, cv: u64, lock: u64, timeout: Option<Duration>) -> bool {
        {
            let mut st = self.lock();
            let waiters = {
                let ls = st.locks.entry(lock).or_default();
                ls.writer = None;
                std::mem::take(&mut ls.waiters)
            };
            for w in waiters {
                if st.tasks[w].status == Status::Blocked(Block::Lock(lock)) {
                    Self::make_runnable(&mut st, w);
                }
            }
            st.cvs.entry(cv).or_default().push(me);
            st.tasks[me].status = Status::Blocked(Block::Cond(cv));
            st.tasks[me].wake_at = timeout.map(|d| st.now_us + d.as_micros() as u64);
            self.reschedule(st, me);
        }
        // Still on the list: nobody notified us, so the deadline woke us.
        let mut st = self.lock();
        let list = st.cvs.entry(cv).or_default();
        if let Some(pos) = list.iter().position(|t| *t == me) {
            list.remove(pos);
            true
        } else {
            false
        }
    }

    fn cv_notify(&self, me: usize, cv: u64, all: bool) {
        {
            let mut st = self.lock();
            let woken: Vec<usize> = {
                let list = st.cvs.entry(cv).or_default();
                if all {
                    std::mem::take(list)
                } else if list.is_empty() {
                    Vec::new()
                } else {
                    vec![list.remove(0)]
                }
            };
            for w in woken {
                Self::make_runnable(&mut st, w);
            }
        }
        self.yield_now(me);
    }

    // ---- tasks -----------------------------------------------------------------------------

    fn sleep(&self, me: usize, d: Duration) {
        let wake = self.lock().now_us + d.as_micros() as u64;
        self.block(me, Block::Sleep, Some(wake));
    }

    /// A task has run to its end: wake its joiners and hand the baton on.
    fn task_finished(&self, me: usize, panicked: bool) {
        let mut st = self.lock();
        st.tasks[me].status = Status::Finished;
        st.tasks[me].panicked = panicked;
        for i in 0..st.tasks.len() {
            if st.tasks[i].status == Status::Blocked(Block::Join(me)) {
                Self::make_runnable(&mut st, i);
            }
        }
        if st.failure.is_some() {
            return;
        }
        // Hand the baton on without parking: this thread is about to end.
        loop {
            let runnable: Vec<usize> =
                st.tasks.iter().enumerate().filter(|(_, t)| t.status == Status::Runnable).map(|(i, _)| i).collect();
            if !runnable.is_empty() {
                let pick = runnable[(splitmix(&mut st.rng) % runnable.len() as u64) as usize];
                st.steps += 1;
                st.trace_hash = (st.trace_hash ^ (pick as u64 + 1)).wrapping_mul(0x0000_0100_0000_01B3);
                let parker = st.tasks[pick].parker.clone();
                drop(st);
                parker.unpark();
                return;
            }
            let next = st.tasks.iter().filter(|t| matches!(t.status, Status::Blocked(_))).filter_map(|t| t.wake_at).min();
            match next {
                Some(t) => {
                    st.now_us = st.now_us.max(t);
                    let now = st.now_us;
                    for task in st.tasks.iter_mut() {
                        if matches!(task.status, Status::Blocked(_)) && task.wake_at.is_some_and(|w| w <= now) {
                            task.status = Status::Runnable;
                            task.wake_at = None;
                        }
                    }
                }
                None => {
                    // Everything left is blocked for good, or this was the last task.
                    if st.tasks.iter().any(|t| matches!(t.status, Status::Blocked(_))) {
                        let report = Self::report(&st);
                        st.failure = Some(report);
                        let parkers: Vec<_> = st.tasks.iter().map(|t| t.parker.clone()).collect();
                        drop(st);
                        for p in parkers {
                            p.unpark();
                        }
                    }
                    return;
                }
            }
        }
    }
}

// ---- public handle --------------------------------------------------------------------------

/// One simulated run. `run` makes the calling thread the root task; tasks spawned through
/// `jamsil_vfs::sync::thread::spawn_named` inside it are scheduled by this simulation.
pub struct Sim {
    shared: Arc<Shared>,
}

impl Sim {
    pub fn new(seed: u64) -> Self {
        let st = State {
            rng: seed,
            now_us: 0,
            steps: 0,
            tasks: Vec::new(),
            locks: HashMap::new(),
            cvs: HashMap::new(),
            holds: HashSet::new(),
            trace_hash: 0xCBF2_9CE4_8422_2325,
            failure: None,
        };
        Self { shared: Arc::new(Shared { st: StdMutex::new(st), seed }) }
    }

    pub fn seed(&self) -> u64 {
        self.shared.seed
    }

    /// Runs `f` as the root task. Every task it spawns must be joined before it returns: a task
    /// left behind is a bug of the scenario and fails the run. A failure of the run (a deadlock)
    /// panics out of the task that detects it, and so out of `run` if that is the root.
    pub fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        {
            let mut st = self.shared.lock();
            assert!(st.tasks.is_empty(), "a Sim runs once");
            st.tasks.push(Task {
                name: "root".into(),
                status: Status::Runnable,
                wake_at: None,
                parker: Parker::new(),
                panicked: false,
            });
        }
        ACTIVE_SIMS.fetch_add(1, Ordering::SeqCst);
        CTX.with(|c| *c.borrow_mut() = Some(Ctx { shared: self.shared.clone(), task: 0 }));
        struct Done<'a>(&'a Sim);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                CTX.with(|c| *c.borrow_mut() = None);
                ACTIVE_SIMS.fetch_sub(1, Ordering::SeqCst);
                let mut st = self.0.shared.lock();
                if st.failure.is_none() && !std::thread::panicking() {
                    // Nothing to do: the leak check below runs on the normal path.
                }
                st.tasks[0].status = Status::Finished;
            }
        }
        let guard = Done(self);
        let r = f();
        {
            let st = self.shared.lock();
            let leaked: Vec<String> =
                st.tasks.iter().skip(1).filter(|t| t.status != Status::Finished).map(|t| format!("{:?}", t.name)).collect();
            if !leaked.is_empty() && st.failure.is_none() {
                drop(st);
                drop(guard);
                panic!("tasks still alive at the end of the run: {}", leaked.join(", "));
            }
        }
        drop(guard);
        r
    }

    /// The virtual clock of this run, for `EngineOptions::clock` (spec 19.2).
    pub fn clock(&self) -> Arc<dyn Clock> {
        Arc::new(SimClock { shared: self.shared.clone() })
    }

    /// Virtual time since the start of the run.
    pub fn now(&self) -> Duration {
        Duration::from_micros(self.shared.lock().now_us)
    }

    /// Number of scheduling decisions so far.
    pub fn steps(&self) -> u64 {
        self.shared.lock().steps
    }

    /// A hash of the sequence of tasks that ran: equal for equal seeds, and it differs when the
    /// schedule differs.
    pub fn schedule_hash(&self) -> u64 {
        self.shared.lock().trace_hash
    }
}

struct SimClock {
    shared: Arc<Shared>,
}

impl Clock for SimClock {
    fn now_micros(&self) -> u64 {
        // Microseconds since the Unix epoch, as `SystemClock` reports, starting at a fixed date so
        // that runs are comparable: 2026-01-01.
        const EPOCH_2026_US: u64 = 1_767_225_600_000_000;
        EPOCH_2026_US + self.shared.lock().now_us
    }
}

// ---- operations the facade calls -------------------------------------------------------------

impl Ctx {
    pub(crate) fn yield_now(&self) {
        self.shared.yield_now(self.task);
    }
    pub(crate) fn sleep(&self, d: Duration) {
        self.shared.sleep(self.task, d);
    }
    pub(crate) fn lock_acquire(&self, id: u64, write: bool) {
        self.shared.lock_acquire(self.task, id, write);
    }
    pub(crate) fn lock_try_acquire(&self, id: u64, write: bool) -> bool {
        self.shared.lock_try_acquire(self.task, id, write)
    }
    pub(crate) fn lock_release(&self, id: u64, write: bool) {
        self.shared.lock_release(id, write);
    }
    pub(crate) fn cv_wait(&self, cv: u64, lock: u64, timeout: Option<Duration>) -> bool {
        self.shared.cv_wait(self.task, cv, lock, timeout)
    }
    pub(crate) fn cv_notify(&self, cv: u64, all: bool) {
        self.shared.cv_notify(self.task, cv, all);
    }

    /// Starts a task. `body` returns whether the task's own body panicked.
    pub(crate) fn spawn(&self, name: &str, body: Box<dyn FnOnce() -> bool + Send>) -> std::io::Result<SimJoin> {
        let (id, parker) = {
            let mut st = self.shared.lock();
            let parker = Parker::new();
            st.tasks.push(Task {
                name: name.to_string(),
                status: Status::Runnable,
                wake_at: None,
                parker: parker.clone(),
                panicked: false,
            });
            (st.tasks.len() - 1, parker)
        };
        let ctx = Ctx { shared: self.shared.clone(), task: id };
        let handle = std::thread::Builder::new().name(name.to_string()).spawn(move || {
            CTX.with(|c| *c.borrow_mut() = Some(ctx.clone()));
            // Wait for the first turn; a failure of the run before it ends the task unrun.
            parker.park();
            let failed = ctx.shared.lock().failure.is_some();
            let panicked = if failed { true } else { body() };
            CTX.with(|c| *c.borrow_mut() = None);
            ctx.shared.task_finished(id, panicked);
            panicked
        })?;
        // The new task may run before the spawner continues.
        self.yield_now();
        Ok(SimJoin { shared: self.shared.clone(), task: id, handle })
    }

    /// A named suspension point (spec 19.7): the task blocks here while the point is held.
    pub(crate) fn point(&self, name: &str) {
        let mut st = self.shared.lock();
        if !st.holds.contains(name) {
            return;
        }
        st.tasks[self.task].status = Status::Blocked(Block::Point(name.to_string()));
        st.tasks[self.task].wake_at = None;
        for i in 0..st.tasks.len() {
            if st.tasks[i].status == Status::Blocked(Block::Await(name.to_string())) {
                Shared::make_runnable(&mut st, i);
            }
        }
        self.shared.reschedule(st, self.task);
    }

    fn hold(&self, name: &str) {
        self.shared.lock().holds.insert(name.to_string());
    }

    fn await_point(&self, name: &str) {
        loop {
            let mut st = self.shared.lock();
            if st.tasks.iter().any(|t| t.status == Status::Blocked(Block::Point(name.to_string()))) {
                return;
            }
            st.tasks[self.task].status = Status::Blocked(Block::Await(name.to_string()));
            st.tasks[self.task].wake_at = None;
            self.shared.reschedule(st, self.task);
        }
    }

    fn release(&self, name: &str) {
        let mut st = self.shared.lock();
        st.holds.remove(name);
        for i in 0..st.tasks.len() {
            if st.tasks[i].status == Status::Blocked(Block::Point(name.to_string())) {
                Shared::make_runnable(&mut st, i);
            }
        }
        drop(st);
        self.yield_now();
    }
}

/// The handle of a simulated task.
pub(crate) struct SimJoin {
    shared: Arc<Shared>,
    task: usize,
    handle: std::thread::JoinHandle<bool>,
}

impl SimJoin {
    pub(crate) fn thread_id(&self) -> std::thread::ThreadId {
        self.handle.thread().id()
    }

    /// Waits, in simulated terms, until the task is finished. Returns whether it panicked.
    pub(crate) fn join(self, ctx: Option<Ctx>) -> bool {
        if let Some(ctx) = ctx {
            loop {
                let mut st = self.shared.lock();
                if st.tasks[self.task].status == Status::Finished {
                    break;
                }
                st.tasks[ctx.task].status = Status::Blocked(Block::Join(self.task));
                st.tasks[ctx.task].wake_at = None;
                self.shared.reschedule(st, ctx.task);
            }
        }
        // The task is finished or failing: its thread ends without waiting for anyone.
        self.handle.join().unwrap_or(true)
    }
}

// ---- controls for scenarios ---------------------------------------------------------------------

/// Controls of a running simulation, for the named scenarios of spec 19.7: hold a suspension point,
/// wait until a task reaches it, do something (take a checkpoint, crash the file system), and
/// release it. Call them from a task of the simulation.
pub mod ctl {
    use super::*;

    fn ctx() -> Ctx {
        current().expect("simulation controls are used inside Sim::run")
    }

    /// Tasks that reach `point(name)` block there until `release(name)`.
    pub fn hold(name: &str) {
        ctx().hold(name);
    }

    /// Blocks the calling task until some task is held at `name`.
    pub fn await_point(name: &str) {
        ctx().await_point(name);
    }

    /// Lets the tasks held at `name` go on, and stops holding the point.
    pub fn release(name: &str) {
        ctx().release(name);
    }
}

/// A named suspension point in engine code. Costs one atomic load when no simulation is running.
pub fn point(name: &str) {
    if let Some(ctx) = current() {
        ctx.point(name);
    }
}
