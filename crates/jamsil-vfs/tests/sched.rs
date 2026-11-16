//! The deterministic scheduler (spec 19.2): a run is a pure function of its seed, scheduling points
//! explore interleavings, virtual time costs no real time, and a deadlock or a lost wake-up is
//! reported instead of hanging.

use jamsil_vfs::sched::{self, ctl, Sim};
use jamsil_vfs::sync::{thread, Arc, Condvar, Mutex, RwLock};
use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

fn spawn<F: FnOnce() + Send + 'static>(name: &str, f: F) -> thread::JoinHandle {
    thread::spawn_named(name, |_| {}, f).unwrap()
}

fn message(e: Box<dyn std::any::Any + Send>) -> String {
    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
}

/// Three tasks add to a shared counter under a lock.
fn counter_run(seed: u64) -> (u64, u64) {
    let sim = Sim::new(seed);
    let total = sim.run(|| {
        let m = Arc::new(Mutex::new(0u64));
        let hs: Vec<_> = (0..3)
            .map(|i| {
                let m = m.clone();
                spawn(&format!("t{i}"), move || {
                    for _ in 0..5 {
                        *m.lock() += 1;
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let v = *m.lock();
        v
    });
    (total, sim.schedule_hash())
}

#[test]
fn a_run_is_a_pure_function_of_its_seed() {
    for seed in [1u64, 7, 99] {
        assert_eq!(counter_run(seed), counter_run(seed), "seed {seed} must replay identically");
    }
    let hashes: HashSet<u64> = (0..40).map(|s| counter_run(s).1).collect();
    assert!(hashes.len() >= 10, "different seeds must explore different schedules ({} distinct of 40)", hashes.len());
}

#[test]
fn a_lock_gives_mutual_exclusion_in_every_schedule() {
    for seed in 0..100 {
        assert_eq!(counter_run(seed).0, 15, "seed {seed}");
    }
}

/// The control for the test above: with the lock released between the read and the write, some
/// schedule must lose an update. If none did, the scheduler would not be exploring interleavings.
#[test]
fn the_scheduler_finds_a_lost_update() {
    let mut lost = Vec::new();
    for seed in 0..60u64 {
        let sim = Sim::new(seed);
        let total = sim.run(|| {
            let m = Arc::new(Mutex::new(0u64));
            let hs: Vec<_> = (0..3)
                .map(|i| {
                    let m = m.clone();
                    spawn(&format!("t{i}"), move || {
                        for _ in 0..4 {
                            let v = *m.lock();
                            thread::yield_now();
                            *m.lock() = v + 1;
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let v = *m.lock();
            v
        });
        if total < 12 {
            lost.push(seed);
        }
    }
    // With a scheduling point between every read and write, every schedule is expected to lose one.
    assert!(!lost.is_empty(), "no seed lost an update in 60 schedules: the scheduler would not be exploring interleavings");
}

fn abba(seed: u64) -> Result<(), String> {
    let sim = Sim::new(seed);
    catch_unwind(AssertUnwindSafe(|| {
        sim.run(|| {
            let (a, b) = (Arc::new(Mutex::new(())), Arc::new(Mutex::new(())));
            let (a1, b1, a2, b2) = (a.clone(), b.clone(), a, b);
            let t1 = spawn("ab", move || {
                let _x = a1.lock();
                thread::yield_now();
                let _y = b1.lock();
            });
            let t2 = spawn("ba", move || {
                let _x = b2.lock();
                thread::yield_now();
                let _y = a2.lock();
            });
            let _ = t1.join();
            let _ = t2.join();
        })
    }))
    .map_err(message)
}

#[test]
fn a_lock_order_deadlock_is_reported_not_hung() {
    let results: Vec<_> = (0..60u64).map(abba).collect();
    let deadlocked: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
    assert!(!deadlocked.is_empty(), "some schedule must deadlock");
    assert!(results.iter().any(|r| r.is_ok()), "and some must not");
    let report = deadlocked[0];
    assert!(report.contains("deadlock") && report.contains("\"ab\"") && report.contains("\"ba\""), "the report names the tasks: {report}");
}

/// A condition variable waited on without a predicate loses a notification that came first.
fn wakeup(seed: u64, with_predicate: bool) -> Result<(), String> {
    let sim = Sim::new(seed);
    catch_unwind(AssertUnwindSafe(|| {
        sim.run(|| {
            let st = Arc::new((Mutex::new(false), Condvar::new()));
            let s2 = st.clone();
            let producer = spawn("producer", move || {
                *s2.0.lock() = true;
                s2.1.notify_one();
            });
            let consumer_state = st.clone();
            let consumer = spawn("consumer", move || {
                let mut g = consumer_state.0.lock();
                if with_predicate {
                    while !*g {
                        consumer_state.1.wait(&mut g);
                    }
                } else {
                    consumer_state.1.wait(&mut g);
                }
            });
            let _ = producer.join();
            let _ = consumer.join();
        })
    }))
    .map_err(message)
}

#[test]
fn a_lost_wakeup_is_found_and_the_predicate_loop_fixes_it() {
    assert!((0..60).any(|s| wakeup(s, false).is_err()), "waiting without a predicate must lose a wake-up in some schedule");
    for seed in 0..100 {
        assert_eq!(wakeup(seed, true), Ok(()), "seed {seed}: with the loop no schedule hangs");
    }
}

#[test]
fn virtual_time_costs_no_real_time_and_wakes_sleepers_in_deadline_order() {
    let started = Instant::now();
    let sim = Sim::new(3);
    let order = sim.run(|| {
        let order = Arc::new(Mutex::new(Vec::new()));
        let hs: Vec<_> = [30u64, 10, 20]
            .into_iter()
            .map(|secs| {
                let order = order.clone();
                spawn(&format!("s{secs}"), move || {
                    thread::sleep(Duration::from_secs(secs * 120)); // up to an hour
                    order.lock().push(secs);
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let v = order.lock().clone();
        v
    });
    assert_eq!(order, [10, 20, 30]);
    assert!(sim.now() >= Duration::from_secs(3600), "virtual time ran an hour: {:?}", sim.now());
    assert!(started.elapsed() < Duration::from_secs(5), "and took {:?} of real time", started.elapsed());
}

#[test]
fn a_timed_wait_times_out_on_the_virtual_clock_or_is_notified() {
    let sim = Sim::new(4);
    let clock = sim.clock();
    sim.run(|| {
        let st = Arc::new((Mutex::new(()), Condvar::new()));
        let mut g = st.0.lock();
        let t0 = clock.now_micros();
        assert!(st.1.wait_for(&mut g, Duration::from_secs(5)), "nobody notifies: the wait times out");
        assert_eq!(clock.now_micros() - t0, 5_000_000, "after exactly five virtual seconds");
        drop(g);

        let s2 = st.clone();
        let notifier = spawn("notifier", move || {
            thread::sleep(Duration::from_secs(2));
            s2.1.notify_one();
        });
        let mut g = st.0.lock();
        let t1 = clock.now_micros();
        let timed_out = st.1.wait_for(&mut g, Duration::from_secs(60));
        assert!(!timed_out, "notified after two seconds, long before the deadline");
        assert_eq!(clock.now_micros() - t1, 2_000_000);
        drop(g);
        notifier.join().unwrap();
    });
}

#[test]
fn the_simulated_clock_follows_virtual_time() {
    let sim = Sim::new(5);
    let clock = sim.clock();
    let (a, b) = sim.run(|| {
        let a = clock.now_micros();
        thread::sleep(Duration::from_millis(250));
        (a, clock.now_micros())
    });
    assert_eq!(b - a, 250_000);
}

#[test]
fn a_panicking_task_is_reported_by_join() {
    let sim = Sim::new(6);
    sim.run(|| {
        let h = spawn("boom", || panic!("boom"));
        assert_eq!(h.join(), Err(thread::ThreadPanicked));
        let ok = spawn("fine", || {});
        assert_eq!(ok.join(), Ok(()));
    });
}

#[test]
fn a_task_left_behind_fails_the_run() {
    let sim = Sim::new(7);
    let r = catch_unwind(AssertUnwindSafe(|| {
        sim.run(|| {
            let _leaked = spawn("leaked", || thread::sleep(Duration::from_secs(1)));
        })
    }));
    assert!(message(r.unwrap_err()).contains("still alive"));
}

/// Spec 19.7: fixed suspension points. A task reaching a held point stops there; the driver sees it
/// arrive, does something, and lets it go.
#[test]
fn a_held_suspension_point_stops_a_task_until_released() {
    for seed in 0..20 {
        let sim = Sim::new(seed);
        sim.run(|| {
            let log = Arc::new(Mutex::new(Vec::new()));
            ctl::hold("after_insert");
            let l2 = log.clone();
            let h = spawn("committer", move || {
                l2.lock().push("inserted");
                sched::point("after_insert");
                l2.lock().push("flushed");
            });
            ctl::await_point("after_insert");
            assert_eq!(*log.lock(), ["inserted"], "the task is held at the point, before the flush");
            log.lock().push("checkpoint");
            ctl::release("after_insert");
            h.join().unwrap();
            assert_eq!(*log.lock(), ["inserted", "checkpoint", "flushed"], "seed {seed}");
        });
    }
}

#[test]
fn a_reader_writer_lock_admits_readers_together_and_writers_alone() {
    let mut together = false;
    for seed in 0..100u64 {
        let sim = Sim::new(seed);
        let (max_readers, total) = sim.run(|| {
            let rw = Arc::new(RwLock::new(0u64));
            let live = Arc::new(Mutex::new((0usize, 0usize))); // (readers now, most readers at once)
            let mut hs = Vec::new();
            for i in 0..2 {
                let (rw, live) = (rw.clone(), live.clone());
                hs.push(spawn(&format!("r{i}"), move || {
                    let _g = rw.read();
                    {
                        let mut l = live.lock();
                        l.0 += 1;
                        l.1 = l.1.max(l.0);
                    }
                    thread::yield_now();
                    live.lock().0 -= 1;
                }));
            }
            for i in 0..2 {
                let rw = rw.clone();
                hs.push(spawn(&format!("w{i}"), move || {
                    for _ in 0..3 {
                        let mut g = rw.write();
                        let v = *g;
                        thread::yield_now();
                        *g = v + 1; // exclusive: no update is lost
                    }
                }));
            }
            for h in hs {
                h.join().unwrap();
            }
            let most = live.lock().1;
            let total = *rw.read();
            (most, total)
        });
        assert_eq!(total, 6, "seed {seed}: writers are exclusive");
        together |= max_readers >= 2;
    }
    assert!(together, "some schedule must let two readers in at once");
}
