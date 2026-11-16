use crate::*;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;

pub struct PosixVfs;

struct PosixFile {
    f: File,
}

fn errno_result(rc: i32) -> IoResult<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc).into())
    }
}

impl Vfs for PosixVfs {
    fn open(&self, path: &Path, opts: OpenOpts) -> IoResult<Box<dyn VfsFile>> {
        let f = OpenOptions::new()
            .read(true)
            .write(opts.write)
            .create(opts.create)
            .open(path)?;
        Ok(Box::new(PosixFile { f }))
    }
    fn rename(&self, from: &Path, to: &Path) -> IoResult<()> {
        Ok(fs::rename(from, to)?)
    }
    fn unlink(&self, path: &Path) -> IoResult<()> {
        Ok(fs::remove_file(path)?)
    }
    fn sync_dir(&self, dir: &Path) -> IoResult<()> {
        Ok(File::open(dir)?.sync_all()?)
    }
    fn remove_dir(&self, dir: &Path) -> IoResult<()> {
        Ok(fs::remove_dir(dir)?)
    }
    fn list(&self, dir: &Path) -> IoResult<Vec<DirEntry>> {
        let mut out = Vec::new();
        for e in fs::read_dir(dir)? {
            let e = e?;
            out.push(DirEntry {
                name: e.file_name().to_string_lossy().into_owned(),
                is_dir: e.file_type()?.is_dir(),
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
    fn free_space(&self, dir: &Path) -> IoResult<u64> {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes())
            .map_err(|e| IoError::new(IoErrorKind::Fatal, e.to_string()))?;
        // SAFETY: `statvfs` is a plain C struct of integers, for which all-zero bytes are valid.
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        // SAFETY: `c` is a valid NUL-terminated path and `st` is a properly sized out-parameter.
        let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(st.f_bavail as u64 * st.f_frsize as u64)
    }
    fn create_dir_all(&self, dir: &Path) -> IoResult<()> {
        Ok(fs::create_dir_all(dir)?)
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
    fn lock_exclusive(&self, path: &Path) -> IoResult<Box<dyn Send + Sync>> {
        let f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
        // SAFETY: fd is valid for the lifetime of `f`.
        let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return Err(IoError::new(IoErrorKind::Fatal, "data directory is locked by another process"));
        }
        Ok(Box::new(f))
    }
}

impl VfsFile for PosixFile {
    fn read_at(&self, buf: &mut [u8], off: u64) -> IoResult<usize> {
        loop {
            match self.f.read_at(buf, off) {
                Ok(n) => return Ok(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }
    fn write_at(&self, buf: &[u8], off: u64) -> IoResult<()> {
        // write_all_at loops on EINTR and short writes.
        Ok(self.f.write_all_at(buf, off)?)
    }
    fn sync_data(&self) -> IoResult<()> {
        self.f
            .sync_data()
            .map_err(|e| IoError::new(IoErrorKind::Fatal, format!("fdatasync failed: {e}")))
    }
    fn allocate(&self, off: u64, len: u64) -> IoResult<()> {
        // SAFETY: valid fd; plain syscall.
        let rc = unsafe { libc::posix_fallocate(self.f.as_raw_fd(), off as i64, len as i64) };
        errno_result(rc)
    }
    fn reserve(&self, off: u64, len: u64) -> IoResult<()> {
        // SAFETY: valid fd; FALLOC_FL_KEEP_SIZE keeps the file length.
        let rc = unsafe { libc::fallocate(self.f.as_raw_fd(), libc::FALLOC_FL_KEEP_SIZE, off as i64, len as i64) };
        if rc != 0 {
            let e = std::io::Error::last_os_error();
            // Filesystems without fallocate support: not an error for correctness.
            if matches!(e.raw_os_error(), Some(libc::EOPNOTSUPP) | Some(libc::ENOSYS)) {
                return Ok(());
            }
            return Err(e.into());
        }
        Ok(())
    }
    fn truncate(&self, len: u64) -> IoResult<()> {
        Ok(self.f.set_len(len)?)
    }
    fn advise(&self, off: u64, len: u64, hint: Advice) {
        let a = match hint {
            Advice::WillNeed => libc::POSIX_FADV_WILLNEED,
            Advice::Sequential => libc::POSIX_FADV_SEQUENTIAL,
            Advice::DontNeed => libc::POSIX_FADV_DONTNEED,
        };
        // SAFETY: advisory only; errors are ignored.
        unsafe {
            libc::posix_fadvise(self.f.as_raw_fd(), off as i64, len as i64, a);
        }
    }
    fn len(&self) -> IoResult<u64> {
        Ok(self.f.metadata()?.len())
    }
}
