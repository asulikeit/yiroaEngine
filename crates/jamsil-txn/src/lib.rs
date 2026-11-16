//! The transaction layer (spec 12): xids, the commit log and the CSN log, snapshots and visibility,
//! the commit sequencer, and the lock manager (spec 13).
//!
//! The commit log, the CSN log and the WAL are reached through the traits of `store`, so that every
//! interleaving runs under the deterministic scheduler against in-memory fakes (spec 19.2); the
//! engine implements them over the buffer pool and the WAL.
#![forbid(unsafe_code)]

pub mod alloc;
pub mod applier;
pub mod lock;
pub mod manager;
pub mod published;
pub mod real;
pub mod seqlock;
pub mod sessions;
pub mod snapshot;
pub mod store;
pub mod visibility;

pub use alloc::*;
pub use applier::*;
pub use lock::*;
pub use manager::*;
pub use published::*;
pub use real::{redo_alloc_batch, redo_clog_truncate, redo_clog_zero_page, redo_xact_bits, PoolStore, RedoError, WalBridge};
pub use sessions::*;
pub use snapshot::*;
pub use store::*;
pub use visibility::*;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum TxnError {
    /// A record whose size depends on user data is over the limit: SQLSTATE 54000.
    #[error("54000 record of {len} bytes exceeds the limit of {max}")]
    RecordTooLarge { len: usize, max: usize },
    /// The log no longer accepts records (`JDB-3001`): nothing of this commit reached it.
    #[error("JDB-3001 wal quarantined: {0}")]
    Quarantined(String),
    /// The commit record was inserted and its flush failed: it may or may not survive a restart
    /// (SQLSTATE 08007).
    #[error("08007 commit outcome unknown: {0}")]
    OutcomeUnknown(String),
    /// A commit-log or CSN-log page failed verification (`XX001`): the store is quarantined, and the
    /// node reinstalls from a peer snapshot (spec 5.6).
    #[error("XX001 corrupt page: {0}")]
    Corrupt(String),
    /// A page of the commit log could not be read or written for a reason that is not corruption
    /// (no free buffer, a disk that is full).
    #[error("XX000 buffer pool: {0}")]
    Buffer(String),
    /// A counter ran out (`53000`): the xid space, or the 32-bit oid space.
    #[error("53000 {0} space exhausted")]
    Exhausted(&'static str),
    #[error("53300 too many snapshots held at once")]
    TooManySessions,
    /// `snapshot_at` found the index below the horizon: SQLSTATE 40001.
    #[error("40001 snapshot too old")]
    SnapshotTooOld,
    /// `40001`: the commit gate of spec 12.6 refused a SERIALIZABLE transaction.
    #[error("40001 could not serialize access")]
    SerializationFailure,
    /// A wait for a lock ended: `55P03` on timeout, `40P01` for a deadlock, `40001` when wounded
    /// (`JDB-2002` by apply), `57014` when cancelled.
    #[error("lock wait failed: {0:?}")]
    Lock(LockError),
}
