//! Buffer pool and relation file manager (spec section 7).
//!
//! The frame memory is plain safe Rust (`RwLock<[u8; PAGE_SIZE]>` per frame) rather than one
//! aligned mapping, so this crate currently contains no `unsafe` (see Question.md).
#![forbid(unsafe_code)]

mod pool;
mod smgr;

pub use pool::*;
pub use smgr::*;

use jamsil_page::walrec::WalPermit;
use jamsil_page::{Corrupt, Lsn, PageTag};
use jamsil_vfs::IoError;

/// What the buffer pool needs from the WAL, without importing it (spec 4.1): five calls.
pub trait WalFlush: Send + Sync {
    /// Blocks until `durable_lsn >= upto`.
    fn flush(&self, upto: Lsn) -> Result<(), BufError>;
    /// The current redo point: a page whose `page_lsn` is at or below it needs a full-page image
    /// before its next change (spec 8.3).
    fn redo_lsn(&self) -> Lsn;
    /// A permit for one standalone page-image record (spec 8.8), or `None` if the log does not
    /// accept records. Taken before the critical section of a hint (spec 7.6).
    fn permit_page_image(&self) -> Option<WalPermit>;
    /// Inserts a standalone full-page image record for `page` on that permit and returns its end
    /// LSN (spec 7.6).
    fn insert_page_image(&self, permit: WalPermit, tag: PageTag, page: &[u8]) -> Lsn;
    /// The current insert position; a page claiming a later `page_lsn` is corrupt (spec 7.3).
    fn insert_lsn(&self) -> Lsn;
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum BufError {
    #[error("corrupt page: {0}")]
    Corrupt(String),
    #[error("io: {0}")]
    Io(IoError),
    #[error("no free buffer")]
    NoFreeBuffer,
    #[error("wal: {0}")]
    Wal(String),
}

impl From<IoError> for BufError {
    fn from(e: IoError) -> Self {
        BufError::Io(e)
    }
}
impl From<Corrupt> for BufError {
    fn from(e: Corrupt) -> Self {
        BufError::Corrupt(e.0)
    }
}

impl BufError {
    /// SQLSTATE of spec 18.2.
    pub fn sqlstate(&self) -> &'static str {
        match self {
            BufError::Corrupt(_) => "XX001",
            BufError::NoFreeBuffer => "53200",
            BufError::Io(e) if e.kind == jamsil_vfs::IoErrorKind::DiskFull => "53100",
            _ => "XX000",
        }
    }
}
