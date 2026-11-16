//! The same engine on the real filesystem (PosixVfs).

use jamsil_engine::*;
use jamsil_page::*;
use jamsil_vfs::{PosixVfs, Vfs};
use std::sync::Arc;

/// Extension formats kind-0 pages; the first record that uses a page gives it its purpose.
fn extend_raw(e: &NativeEngine, rel: RelFileId, n: u32) -> Result<(), EngineError> {
    let first = e.extend_relation(rel, ForkId::Main, n)?;
    e.init_raw_pages(e.alloc_xid().unwrap(), rel, ForkId::Main, first, n)
}

#[test]
fn real_filesystem_restart() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("posix-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let root = dir.join("rdb");
    let v: Arc<dyn Vfs> = Arc::new(PosixVfs);
    let o = EngineOptions { buffer_frames: 64, wal_segment_size: Some(1 << 20), ..Default::default() };
    {
        let e = NativeEngine::open_or_create(v.clone(), &root, o.clone()).unwrap();
        e.create_relation(100, ForkId::Main).unwrap();
        extend_raw(&e, 100, 2).unwrap();
        let x = e.alloc_xid().unwrap();
        e.write_bytes(x, PageTag::new(100, ForkId::Main, 1), 64, &1234u64.to_le_bytes()).unwrap();
        e.commit(x, Some(AppliedPos { index: 3, term: 1 }), &[]).unwrap();
        // A second process (second open) must be refused while the first holds the lock.
        assert!(NativeEngine::open_or_create(v.clone(), &root, o.clone()).is_err());
    }
    // Dropped without shutdown: recovery path.
    let e = NativeEngine::open_or_create(v.clone(), &root, o.clone()).unwrap();
    assert_eq!(e.applied().index, 3);
    let p = e.pin(PageTag::new(100, ForkId::Main, 1)).unwrap();
    assert_eq!(rd_u64(&p.read()[..], 64), 1234);
    drop(p);
    e.shutdown().unwrap();
    drop(e);
    let e = NativeEngine::open_or_create(v, &root, o).unwrap();
    assert_eq!(e.applied().index, 3);
    let _ = std::fs::remove_dir_all(&dir);
}
