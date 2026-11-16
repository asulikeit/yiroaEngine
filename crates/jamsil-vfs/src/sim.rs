//! In-memory deterministic backend. Models a disk with a durable and a volatile image per file.

use crate::*;
use crate::sync::{Arc, Mutex};
use std::collections::{BTreeSet, HashMap};

const SECTOR: usize = 512;

#[derive(Clone)]
enum Op {
    Write(u64, Vec<u8>),
    Truncate(u64),
}

/// What a power loss does to the writes that no sync covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashPolicy {
    /// Each unsynced write survives per 512-byte sector with probability 1/2, in order.
    Random,
    /// Every unsynced write survives: the case "`SimVfs` keeping the unsynced commit record" of
    /// spec 19.7 ("Early persistence").
    KeepUnsynced,
    /// No unsynced write survives.
    DropUnsynced,
}

#[derive(Default, Clone)]
struct Inode {
    volatile: Vec<u8>,
    durable: Vec<u8>,
    pending: Vec<Op>,
}

fn apply(img: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Write(off, data) => {
            let end = *off as usize + data.len();
            if img.len() < end {
                img.resize(end, 0);
            }
            img[*off as usize..end].copy_from_slice(data);
        }
        Op::Truncate(len) => img.resize(*len as usize, 0),
    }
}

type InodeRef = Arc<Mutex<Inode>>;

#[derive(Default)]
struct State {
    ns: HashMap<PathBuf, InodeRef>,
    durable_ns: HashMap<PathBuf, InodeRef>,
    dirs: BTreeSet<PathBuf>,
    /// Directory renames not yet made durable by a `sync_dir` of their parent.
    pending_dir_renames: Vec<(PathBuf, PathBuf)>,
    rng: u64,
    capacity: Option<u64>,
}

#[derive(Clone)]
pub struct SimVfs {
    st: Arc<Mutex<State>>,
}

fn next(rng: &mut u64) -> u64 {
    *rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *rng;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn survives(policy: CrashPolicy, rng: &mut u64) -> bool {
    match policy {
        CrashPolicy::Random => next(rng) & 1 == 1,
        CrashPolicy::KeepUnsynced => true,
        CrashPolicy::DropUnsynced => false,
    }
}

impl SimVfs {
    pub fn new(seed: u64) -> Self {
        let st = State { rng: seed, ..Default::default() };
        Self { st: Arc::new(Mutex::new(st)) }
    }

    /// Limits the total volatile bytes; further growth fails with `DiskFull`.
    pub fn set_capacity(&self, bytes: Option<u64>) {
        self.st.lock().capacity = bytes;
    }

    /// Power loss: every synced byte survives; each unsynced write survives per 512-byte
    /// sector with probability 1/2, in order. Unsynced namespace changes are lost.
    pub fn crash(&self) {
        self.crash_with(CrashPolicy::Random);
    }

    /// A copy of the file system as a power loss would leave it, while `self` goes on undisturbed.
    /// For the scenarios that crash at a suspension point while the tasks of the simulation are still
    /// held there (spec 19.7).
    pub fn crashed_copy(&self, policy: CrashPolicy) -> SimVfs {
        let copy = {
            let st = self.st.lock();
            // Inodes are shared between the namespace and the durable namespace; the copy keeps that.
            let mut map: HashMap<*const Mutex<Inode>, InodeRef> = HashMap::new();
            let mut dup = |i: &InodeRef| -> InodeRef {
                map.entry(Arc::as_ptr(i)).or_insert_with(|| Arc::new(Mutex::new(i.lock().clone()))).clone()
            };
            let ns = st.ns.iter().map(|(p, i)| (p.clone(), dup(i))).collect();
            let durable_ns = st.durable_ns.iter().map(|(p, i)| (p.clone(), dup(i))).collect();
            SimVfs {
                st: Arc::new(Mutex::new(State {
                    ns,
                    durable_ns,
                    dirs: st.dirs.clone(),
                    pending_dir_renames: st.pending_dir_renames.clone(),
                    rng: st.rng,
                    capacity: st.capacity,
                })),
            }
        };
        copy.crash_with(policy);
        copy
    }

    pub fn crash_with(&self, policy: CrashPolicy) {
        let mut st = self.st.lock();
        let durable_ns = st.durable_ns.clone();
        let mut rng = st.rng;
        let mut paths: Vec<_> = durable_ns.keys().cloned().collect();
        paths.sort();
        for p in paths {
            let ino = durable_ns[&p].clone();
            let mut g = ino.lock();
            let pending = std::mem::take(&mut g.pending);
            let mut img = g.durable.clone();
            for op in &pending {
                match op {
                    Op::Write(off, data) => {
                        let mut pos = 0usize;
                        while pos < data.len() {
                            let abs = *off as usize + pos;
                            let sector_end = (abs / SECTOR + 1) * SECTOR;
                            let n = (sector_end - abs).min(data.len() - pos);
                            if survives(policy, &mut rng) {
                                apply(&mut img, &Op::Write(abs as u64, data[pos..pos + n].to_vec()));
                            }
                            pos += n;
                        }
                    }
                    Op::Truncate(_) => {
                        if survives(policy, &mut rng) {
                            apply(&mut img, op);
                        }
                    }
                }
            }
            g.durable = img.clone();
            g.volatile = img;
        }
        st.rng = rng;
        st.ns = durable_ns;
        // Directory renames that no sync made durable are undone.
        let undo = std::mem::take(&mut st.pending_dir_renames);
        for (from, to) in undo.into_iter().rev() {
            remap_dirs(&mut st.dirs, &to, &from);
        }
    }

    /// Flips one bit in both images (media corruption after a successful sync).
    pub fn flip_bit(&self, path: &Path, off: u64, bit: u8) {
        let st = self.st.lock();
        if let Some(ino) = st.ns.get(path) {
            let mut g = ino.lock();
            g.volatile[off as usize] ^= 1 << (bit % 8);
            if (off as usize) < g.durable.len() {
                g.durable[off as usize] ^= 1 << (bit % 8);
            }
        }
    }

    /// Overwrites bytes in both images, bypassing the pending queue.
    pub fn poke(&self, path: &Path, off: u64, data: &[u8]) {
        let st = self.st.lock();
        if let Some(ino) = st.ns.get(path) {
            let mut g = ino.lock();
            let op = Op::Write(off, data.to_vec());
            apply(&mut g.volatile, &op);
            apply(&mut g.durable, &op);
        }
    }

    /// Every file with its volatile bytes, for comparing two runs of the same seed.
    pub fn dump(&self) -> std::collections::BTreeMap<PathBuf, Vec<u8>> {
        self.st.lock().ns.iter().map(|(p, i)| (p.clone(), i.lock().volatile.clone())).collect()
    }

    pub fn file_bytes(&self, path: &Path) -> Option<Vec<u8>> {
        self.st.lock().ns.get(path).map(|i| i.lock().volatile.clone())
    }

    fn total_bytes(st: &State) -> u64 {
        st.ns.values().map(|i| i.lock().volatile.len() as u64).sum()
    }
}

struct SimFile {
    ino: InodeRef,
    vfs: SimVfs,
    writable: bool,
}

/// Moves every directory at or below `from` to the same place below `to`.
fn remap_dirs(dirs: &mut BTreeSet<PathBuf>, from: &Path, to: &Path) {
    let moved: Vec<PathBuf> = dirs.iter().filter(|d| d.starts_with(from)).cloned().collect();
    for d in moved {
        dirs.remove(&d);
        dirs.insert(to.join(d.strip_prefix(from).unwrap()));
    }
}

fn remap_ns(ns: &mut HashMap<PathBuf, InodeRef>, from: &Path, to: &Path) {
    let moved: Vec<PathBuf> = ns.keys().filter(|p| p.starts_with(from)).cloned().collect();
    for p in moved {
        let ino = ns.remove(&p).unwrap();
        ns.insert(to.join(p.strip_prefix(from).unwrap()), ino);
    }
}

fn parent(p: &Path) -> PathBuf {
    p.parent().map(|x| x.to_path_buf()).unwrap_or_default()
}

impl Vfs for SimVfs {
    fn open(&self, path: &Path, opts: OpenOpts) -> IoResult<Box<dyn VfsFile>> {
        let mut st = self.st.lock();
        let ino = match st.ns.get(path) {
            Some(i) => i.clone(),
            None if opts.create => {
                let i: InodeRef = Arc::new(Mutex::new(Inode::default()));
                st.ns.insert(path.to_path_buf(), i.clone());
                i
            }
            None => return Err(IoError::not_found(path)),
        };
        Ok(Box::new(SimFile { ino, vfs: self.clone(), writable: opts.write }))
    }
    fn rename(&self, from: &Path, to: &Path) -> IoResult<()> {
        let mut st = self.st.lock();
        if st.dirs.contains(from) {
            let State { ns, dirs, pending_dir_renames, .. } = &mut *st;
            remap_ns(ns, from, to);
            remap_dirs(dirs, from, to);
            pending_dir_renames.push((from.to_path_buf(), to.to_path_buf()));
            return Ok(());
        }
        let ino = st.ns.remove(from).ok_or_else(|| IoError::not_found(from))?;
        st.ns.insert(to.to_path_buf(), ino);
        Ok(())
    }
    fn unlink(&self, path: &Path) -> IoResult<()> {
        let mut st = self.st.lock();
        st.ns.remove(path).map(|_| ()).ok_or_else(|| IoError::not_found(path))
    }
    fn sync_dir(&self, dir: &Path) -> IoResult<()> {
        let mut st = self.st.lock();
        let pending = std::mem::take(&mut st.pending_dir_renames);
        for (from, to) in pending {
            if parent(&from) == dir || parent(&to) == dir {
                remap_ns(&mut st.durable_ns, &from, &to);
            } else {
                st.pending_dir_renames.push((from, to));
            }
        }
        st.durable_ns.retain(|p, _| parent(p) != dir);
        let add: Vec<_> = st.ns.iter().filter(|(p, _)| parent(p) == dir).map(|(p, i)| (p.clone(), i.clone())).collect();
        for (p, i) in add {
            st.durable_ns.insert(p, i);
        }
        Ok(())
    }
    fn remove_dir(&self, dir: &Path) -> IoResult<()> {
        let mut st = self.st.lock();
        if st.ns.keys().any(|p| p.starts_with(dir)) || st.dirs.iter().any(|d| d != dir && d.starts_with(dir)) {
            return Err(IoError::new(IoErrorKind::Fatal, format!("directory not empty: {}", dir.display())));
        }
        if !st.dirs.remove(dir) {
            return Err(IoError::not_found(dir));
        }
        Ok(())
    }
    fn list(&self, dir: &Path) -> IoResult<Vec<DirEntry>> {
        let st = self.st.lock();
        let mut out = Vec::new();
        for p in st.ns.keys() {
            if parent(p) == dir {
                out.push(DirEntry { name: p.file_name().unwrap().to_string_lossy().into_owned(), is_dir: false });
            }
        }
        for d in &st.dirs {
            if parent(d) == dir {
                out.push(DirEntry { name: d.file_name().unwrap().to_string_lossy().into_owned(), is_dir: true });
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
    fn free_space(&self, _dir: &Path) -> IoResult<u64> {
        let st = self.st.lock();
        Ok(match st.capacity {
            Some(c) => c.saturating_sub(Self::total_bytes(&st)),
            None => u64::MAX / 2,
        })
    }
    fn create_dir_all(&self, dir: &Path) -> IoResult<()> {
        let mut st = self.st.lock();
        let mut cur = Some(dir);
        while let Some(d) = cur {
            if d.as_os_str().is_empty() || d == Path::new("/") {
                break;
            }
            st.dirs.insert(d.to_path_buf());
            cur = d.parent();
        }
        Ok(())
    }
    fn exists(&self, path: &Path) -> bool {
        let st = self.st.lock();
        st.ns.contains_key(path) || st.dirs.contains(path)
    }
    fn lock_exclusive(&self, _path: &Path) -> IoResult<Box<dyn Send + Sync>> {
        Ok(Box::new(()))
    }
}

impl SimFile {
    fn check_space(&self, new_len: usize) -> IoResult<()> {
        let st = self.vfs.st.lock();
        if let Some(cap) = st.capacity {
            let cur = SimVfs::total_bytes(&st);
            let mine = self.ino.lock().volatile.len();
            if new_len > mine && cur + (new_len - mine) as u64 > cap {
                return Err(IoError::new(IoErrorKind::DiskFull, "sim: no space"));
            }
        }
        Ok(())
    }
}

impl VfsFile for SimFile {
    fn read_at(&self, buf: &mut [u8], off: u64) -> IoResult<usize> {
        let g = self.ino.lock();
        let off = off as usize;
        if off >= g.volatile.len() {
            return Ok(0);
        }
        let n = buf.len().min(g.volatile.len() - off);
        buf[..n].copy_from_slice(&g.volatile[off..off + n]);
        Ok(n)
    }
    fn write_at(&self, buf: &[u8], off: u64) -> IoResult<()> {
        assert!(self.writable, "write on read-only handle");
        self.check_space(off as usize + buf.len())?;
        let mut g = self.ino.lock();
        let op = Op::Write(off, buf.to_vec());
        apply(&mut g.volatile, &op);
        g.pending.push(op);
        Ok(())
    }
    fn sync_data(&self) -> IoResult<()> {
        let mut g = self.ino.lock();
        let pending = std::mem::take(&mut g.pending);
        for op in &pending {
            apply(&mut g.durable, op);
        }
        Ok(())
    }
    fn allocate(&self, off: u64, len: u64) -> IoResult<()> {
        let end = (off + len) as usize;
        self.check_space(end)?;
        let mut g = self.ino.lock();
        if g.volatile.len() < end {
            let old = g.volatile.len() as u64;
            let op = Op::Write(old, vec![0u8; end - old as usize]);
            apply(&mut g.volatile, &op);
            g.pending.push(op);
        }
        Ok(())
    }
    fn reserve(&self, off: u64, len: u64) -> IoResult<()> {
        let end = (off + len) as usize;
        let st = self.vfs.st.lock();
        if let Some(cap) = st.capacity {
            let mine = self.ino.lock().volatile.len();
            let cur = SimVfs::total_bytes(&st);
            if end > mine && cur + (end - mine) as u64 > cap {
                return Err(IoError::new(IoErrorKind::DiskFull, "sim: no space"));
            }
        }
        Ok(())
    }
    fn truncate(&self, len: u64) -> IoResult<()> {
        let mut g = self.ino.lock();
        let op = Op::Truncate(len);
        apply(&mut g.volatile, &op);
        g.pending.push(op);
        Ok(())
    }
    fn advise(&self, _off: u64, _len: u64, _hint: Advice) {}
    fn len(&self) -> IoResult<u64> {
        Ok(self.ino.lock().volatile.len() as u64)
    }
}
