//! Injectable time and randomness (spec 18.5). Production uses the system clock and the operating
//! system's random source; the simulator injects a fixed clock and a seeded generator.

use crate::sync::atomic::{AtomicU64, Ordering};
use crate::sync::Arc;

pub trait Clock: Send + Sync {
    /// Microseconds since the Unix epoch.
    fn now_micros(&self) -> u64;
}

pub trait Rng: Send + Sync {
    fn next_u64(&self) -> u64;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now_micros(&self) -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_micros() as u64).unwrap_or(0)
    }
}

/// A clock that only moves when told to.
pub struct ManualClock(AtomicU64);

impl ManualClock {
    pub fn new(start_micros: u64) -> Arc<Self> {
        Arc::new(Self(AtomicU64::new(start_micros)))
    }
    pub fn advance(&self, micros: u64) {
        self.0.fetch_add(micros, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now_micros(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// The operating system's random source (`/dev/urandom`).
pub struct OsRng;

impl Rng for OsRng {
    fn next_u64(&self) -> u64 {
        use std::io::Read;
        let mut b = [0u8; 8];
        let ok = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)).is_ok();
        if !ok {
            // No random source: fall back to the clock mixed with an address, never to a constant.
            let t = SystemClock.now_micros();
            return t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (&b as *const _ as u64).rotate_left(23);
        }
        u64::from_le_bytes(b)
    }
}

/// Deterministic generator (splitmix64) for simulation.
pub struct SeededRng(AtomicU64);

impl SeededRng {
    pub fn new(seed: u64) -> Arc<Self> {
        Arc::new(Self(AtomicU64::new(seed)))
    }
}

impl Rng for SeededRng {
    fn next_u64(&self) -> u64 {
        let mut z = self.0.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::SeqCst).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_replays_and_os_rng_varies() {
        let (a, b) = (SeededRng::new(7), SeededRng::new(7));
        assert_eq!((0..4).map(|_| a.next_u64()).collect::<Vec<_>>(), (0..4).map(|_| b.next_u64()).collect::<Vec<_>>());
        assert_ne!(OsRng.next_u64(), OsRng.next_u64());
    }
}
