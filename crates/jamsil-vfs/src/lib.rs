//! File and I/O layer (spec section 5). All file access in the engine goes through [`Vfs`].

pub mod env;
mod fault;
mod posix;
pub mod sched;
mod sim;
pub mod sync;

pub use fault::{FaultPlan, FaultVfs};
pub use posix::PosixVfs;
pub use sim::{CrashPolicy, SimVfs};

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoErrorKind {
    DiskFull,
    Corrupt,
    Transient,
    Fatal,
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind:?}: {msg}")]
pub struct IoError {
    pub kind: IoErrorKind,
    pub msg: String,
}

impl IoError {
    pub fn new(kind: IoErrorKind, msg: impl Into<String>) -> Self {
        Self { kind, msg: msg.into() }
    }
    pub fn not_found(path: &Path) -> Self {
        Self::new(IoErrorKind::Fatal, format!("not found: {}", path.display()))
    }
    pub fn is_not_found(&self) -> bool {
        self.msg.starts_with("not found:")
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        let kind = match e.raw_os_error() {
            Some(libc::ENOSPC) | Some(libc::EDQUOT) => IoErrorKind::DiskFull,
            Some(libc::EAGAIN) | Some(libc::EINTR) => IoErrorKind::Transient,
            _ => IoErrorKind::Fatal,
        };
        if e.kind() == std::io::ErrorKind::NotFound {
            return Self::new(kind, format!("not found: {e}"));
        }
        Self::new(kind, e.to_string())
    }
}

pub type IoResult<T> = Result<T, IoError>;

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenOpts {
    pub create: bool,
    pub write: bool,
}

impl OpenOpts {
    pub fn read_only() -> Self {
        Self { create: false, write: false }
    }
    pub fn read_write() -> Self {
        Self { create: false, write: true }
    }
    pub fn create() -> Self {
        Self { create: true, write: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advice {
    WillNeed,
    Sequential,
    DontNeed,
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
}

pub trait Vfs: Send + Sync + 'static {
    fn open(&self, path: &Path, opts: OpenOpts) -> IoResult<Box<dyn VfsFile>>;
    /// Not durable until `sync_dir`.
    fn rename(&self, from: &Path, to: &Path) -> IoResult<()>;
    /// Not durable until `sync_dir`.
    fn unlink(&self, path: &Path) -> IoResult<()>;
    fn sync_dir(&self, dir: &Path) -> IoResult<()>;
    fn list(&self, dir: &Path) -> IoResult<Vec<DirEntry>>;
    /// Removes an empty directory. Durable after `sync_dir` of its parent.
    fn remove_dir(&self, dir: &Path) -> IoResult<()>;
    fn free_space(&self, dir: &Path) -> IoResult<u64>;
    fn create_dir_all(&self, dir: &Path) -> IoResult<()>;
    fn exists(&self, path: &Path) -> bool;
    /// Takes the exclusive process lock on a file (spec 5.3 rule 6). Held until the guard drops.
    fn lock_exclusive(&self, path: &Path) -> IoResult<Box<dyn Send + Sync>>;
}

pub trait VfsFile: Send + Sync {
    /// Short read only at EOF.
    fn read_at(&self, buf: &mut [u8], off: u64) -> IoResult<usize>;
    /// All bytes or an error.
    fn write_at(&self, buf: &[u8], off: u64) -> IoResult<()>;
    /// `fdatasync`.
    fn sync_data(&self) -> IoResult<()>;
    /// Zero-filled extension of `[off, off+len)`; the file grows to cover it.
    fn allocate(&self, off: u64, len: u64) -> IoResult<()>;
    /// Claims space without changing the file length.
    fn reserve(&self, off: u64, len: u64) -> IoResult<()>;
    fn truncate(&self, len: u64) -> IoResult<()>;
    fn advise(&self, off: u64, len: u64, hint: Advice);
    fn len(&self) -> IoResult<u64>;
    fn is_empty(&self) -> IoResult<bool> {
        Ok(self.len()? == 0)
    }
}

/// Reads exactly `buf.len()` bytes or fails with `Corrupt` on EOF.
pub fn read_exact_at(f: &dyn VfsFile, buf: &mut [u8], off: u64) -> IoResult<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = f.read_at(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(IoError::new(IoErrorKind::Corrupt, format!("short read at {off}")));
        }
        done += n;
    }
    Ok(())
}

pub fn join(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}
