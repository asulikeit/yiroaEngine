//! Fault-injection wrapper (spec 5.2). Wraps another backend.

use crate::*;
use crate::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use crate::sync::Arc;

/// Shared switches; clone the `Arc` to flip them from a test.
#[derive(Default)]
pub struct FaultPlan {
    /// Every `write_at`/`allocate`/`reserve` that grows a file fails with `ENOSPC`.
    pub disk_full: AtomicBool,
    /// The next N `sync_data` calls (counting down) fail; negative means never.
    pub fail_sync_after: AtomicI64,
    pub fail_all_syncs: AtomicBool,
    /// Writes fail with a fatal EIO.
    pub fail_writes: AtomicBool,
    /// Reads fail with a fatal EIO.
    pub fail_reads: AtomicBool,
    /// Every `write_at` fails with `ENOSPC`, also inside a file that is long enough already (what
    /// a copy-on-write or thin-provisioned file system can do to a preallocated segment).
    pub write_enospc: AtomicBool,
    /// Process death: after this many more mutating calls (create, write, sync, rename, unlink,
    /// truncate, allocate) every further mutating call fails. Negative means never. Run a
    /// procedure with every budget in turn, then crash the file system underneath, to test that
    /// the procedure is crash-safe at each I/O boundary.
    pub die_after_ops: AtomicI64,
}

impl FaultPlan {
    pub fn new() -> Arc<Self> {
        let p = Self::default();
        p.fail_sync_after.store(-1, Ordering::SeqCst);
        p.die_after_ops.store(-1, Ordering::SeqCst);
        Arc::new(p)
    }
}

impl FaultPlan {
    fn tick(&self) -> IoResult<()> {
        let n = self.die_after_ops.load(Ordering::SeqCst);
        if n == 0 {
            return Err(IoError::new(IoErrorKind::Fatal, "injected process death"));
        }
        if n > 0 {
            self.die_after_ops.fetch_sub(1, Ordering::SeqCst);
        }
        Ok(())
    }
}

pub struct FaultVfs<V: Vfs> {
    inner: V,
    plan: Arc<FaultPlan>,
    marker_dir: Option<PathBuf>,
}

impl<V: Vfs> FaultVfs<V> {
    /// `marker_dir`: directory checked for the marker files `disk_full` and `sqlite_full`.
    pub fn new(inner: V, plan: Arc<FaultPlan>, marker_dir: Option<PathBuf>) -> Self {
        Self { inner, plan, marker_dir }
    }
}

struct FaultFile<'a> {
    inner: Box<dyn VfsFile>,
    plan: Arc<FaultPlan>,
    full_marker: Option<PathBuf>,
    _p: std::marker::PhantomData<&'a ()>,
}

fn full(plan: &FaultPlan, marker: &Option<PathBuf>, vfs_exists: impl Fn(&Path) -> bool) -> bool {
    plan.disk_full.load(Ordering::SeqCst)
        || marker.as_ref().is_some_and(|d| vfs_exists(&d.join("disk_full")) || vfs_exists(&d.join("sqlite_full")))
}

impl<V: Vfs> Vfs for FaultVfs<V> {
    fn open(&self, path: &Path, opts: OpenOpts) -> IoResult<Box<dyn VfsFile>> {
        if opts.create && !self.inner.exists(path) {
            self.plan.tick()?;
        }
        Ok(Box::new(FaultFile {
            inner: self.inner.open(path, opts)?,
            plan: self.plan.clone(),
            full_marker: self.marker_dir.clone(),
            _p: std::marker::PhantomData,
        }))
    }
    fn rename(&self, from: &Path, to: &Path) -> IoResult<()> {
        self.plan.tick()?;
        self.inner.rename(from, to)
    }
    fn unlink(&self, path: &Path) -> IoResult<()> {
        self.plan.tick()?;
        self.inner.unlink(path)
    }
    fn sync_dir(&self, dir: &Path) -> IoResult<()> {
        self.plan.tick()?;
        self.inner.sync_dir(dir)
    }
    fn list(&self, dir: &Path) -> IoResult<Vec<DirEntry>> {
        self.inner.list(dir)
    }
    fn remove_dir(&self, dir: &Path) -> IoResult<()> {
        self.plan.tick()?;
        self.inner.remove_dir(dir)
    }
    fn free_space(&self, dir: &Path) -> IoResult<u64> {
        self.inner.free_space(dir)
    }
    fn create_dir_all(&self, dir: &Path) -> IoResult<()> {
        self.inner.create_dir_all(dir)
    }
    fn exists(&self, path: &Path) -> bool {
        self.inner.exists(path)
    }
    fn lock_exclusive(&self, path: &Path) -> IoResult<Box<dyn Send + Sync>> {
        self.inner.lock_exclusive(path)
    }
}

impl FaultFile<'_> {
    fn is_full(&self) -> bool {
        full(&self.plan, &self.full_marker, |p| p.exists())
    }
    fn enospc() -> IoError {
        IoError::new(IoErrorKind::DiskFull, "injected ENOSPC")
    }
}

impl VfsFile for FaultFile<'_> {
    fn read_at(&self, buf: &mut [u8], off: u64) -> IoResult<usize> {
        if self.plan.fail_reads.load(Ordering::SeqCst) {
            return Err(IoError::new(IoErrorKind::Fatal, "injected EIO on read"));
        }
        self.inner.read_at(buf, off)
    }
    fn write_at(&self, buf: &[u8], off: u64) -> IoResult<()> {
        self.plan.tick()?;
        if self.plan.fail_writes.load(Ordering::SeqCst) {
            return Err(IoError::new(IoErrorKind::Fatal, "injected EIO"));
        }
        if self.plan.write_enospc.load(Ordering::SeqCst) {
            return Err(Self::enospc());
        }
        if self.is_full() && off + buf.len() as u64 > self.inner.len()? {
            return Err(Self::enospc());
        }
        self.inner.write_at(buf, off)
    }
    fn sync_data(&self) -> IoResult<()> {
        self.plan.tick()?;
        if self.plan.fail_all_syncs.load(Ordering::SeqCst) {
            return Err(IoError::new(IoErrorKind::Fatal, "injected fdatasync failure"));
        }
        let n = self.plan.fail_sync_after.load(Ordering::SeqCst);
        if n == 0 {
            self.plan.fail_sync_after.store(-1, Ordering::SeqCst);
            return Err(IoError::new(IoErrorKind::Fatal, "injected fdatasync failure"));
        } else if n > 0 {
            self.plan.fail_sync_after.fetch_sub(1, Ordering::SeqCst);
        }
        self.inner.sync_data()
    }
    fn allocate(&self, off: u64, len: u64) -> IoResult<()> {
        self.plan.tick()?;
        if self.is_full() {
            return Err(Self::enospc());
        }
        self.inner.allocate(off, len)
    }
    fn reserve(&self, off: u64, len: u64) -> IoResult<()> {
        if self.is_full() {
            return Err(Self::enospc());
        }
        self.inner.reserve(off, len)
    }
    fn truncate(&self, len: u64) -> IoResult<()> {
        self.plan.tick()?;
        self.inner.truncate(len)
    }
    fn advise(&self, off: u64, len: u64, hint: Advice) {
        self.inner.advise(off, len, hint)
    }
    fn len(&self) -> IoResult<u64> {
        self.inner.len()
    }
}
