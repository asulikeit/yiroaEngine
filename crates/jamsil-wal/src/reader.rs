use crate::*;

/// Why sequential reading stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadStop {
    /// No segment file holds the next position.
    NoSegment { pos: Lsn },
    /// The record (or segment header) at `pos` failed validation.
    Invalid { pos: Lsn, why: String },
}

impl ReadStop {
    pub fn pos(&self) -> Lsn {
        match self {
            Self::NoSegment { pos } | Self::Invalid { pos, .. } => *pos,
        }
    }
}

pub struct WalReader {
    vfs: Arc<dyn Vfs>,
    dir: PathBuf,
    cfg: WalConfig,
    pos: Lsn,
    prev: Option<Lsn>,
    cur: Option<(u64, Arc<dyn VfsFile>)>,
    header_checked: u64,
    valid_end: Lsn,
}

impl WalReader {
    pub fn new(vfs: Arc<dyn Vfs>, dir: PathBuf, cfg: WalConfig, from: Lsn) -> Self {
        Self { vfs, dir, cfg, pos: from, prev: None, cur: None, header_checked: 0, valid_end: from }
    }

    /// Position of the next record to read (after boundary normalization).
    pub fn pos(&self) -> Lsn {
        self.pos
    }
    /// End of the last valid record (or the starting position if none was read). After a
    /// SWITCH record this is the end of its segment.
    pub fn valid_end(&self) -> Lsn {
        self.valid_end
    }
    /// Start LSN of the last record returned.
    pub fn last_start(&self) -> Option<Lsn> {
        self.prev
    }

    fn file(&mut self, segno: u64) -> Result<Option<Arc<dyn VfsFile>>, ReadStop> {
        if let Some((s, f)) = &self.cur {
            if *s == segno {
                return Ok(Some(f.clone()));
            }
        }
        let path = segment_path(&self.dir, segno);
        if !self.vfs.exists(&path) {
            return Ok(None);
        }
        let f: Arc<dyn VfsFile> = match self.vfs.open(&path, OpenOpts::read_only()) {
            Ok(f) => Arc::from(f),
            Err(e) => return Err(ReadStop::Invalid { pos: self.pos, why: e.to_string() }),
        };
        self.cur = Some((segno, f.clone()));
        Ok(Some(f))
    }

    /// Returns the next valid record, or the reason reading stopped.
    pub fn next_record(&mut self) -> Result<WalRecord, ReadStop> {
        let seg = self.cfg.segment_size;
        loop {
            // Normalize (spec 8.1, answer A8): a position inside a segment header, which includes
            // the boundary itself, moves to the first record slot of that segment.
            if self.pos % seg < SEG_HEADER_LEN as u64 {
                self.pos = self.pos / seg * seg + SEG_HEADER_LEN as u64;
            }
            let segno = self.pos / seg;
            let Some(f) = self.file(segno)? else {
                return Err(ReadStop::NoSegment { pos: self.pos });
            };
            if self.header_checked != segno {
                let mut h = [0u8; SEG_HEADER_LEN];
                let ok = matches!(f.read_at(&mut h, 0), Ok(n) if n == SEG_HEADER_LEN);
                if !ok || check_seg_header(&h, self.cfg.system_id, segno).is_err() {
                    return Err(ReadStop::Invalid { pos: self.pos, why: "bad segment header".into() });
                }
                self.header_checked = segno;
            }
            let seg_off = self.pos - segno * seg;
            let seg_end = (segno + 1) * seg;
            let room = (seg_end - self.pos) as usize;
            if room == 0 {
                self.pos = seg_end;
                continue;
            }
            if room < REC_HEADER_LEN {
                return Err(ReadStop::Invalid { pos: self.pos, why: "tail smaller than a header".into() });
            }
            let mut head = [0u8; REC_HEADER_LEN];
            match f.read_at(&mut head, seg_off) {
                Ok(n) if n == REC_HEADER_LEN => {}
                _ => return Err(ReadStop::Invalid { pos: self.pos, why: "short read".into() }),
            }
            let total = rd_u32(&head, 0) as usize;
            if !(REC_HEADER_LEN..=MAX_RECORD_LEN).contains(&total) || total % 8 != 0 || total > room {
                return Err(ReadStop::Invalid { pos: self.pos, why: "bad total_len".into() });
            }
            let mut buf = vec![0u8; total];
            match f.read_at(&mut buf, seg_off) {
                Ok(n) if n == total => {}
                _ => return Err(ReadStop::Invalid { pos: self.pos, why: "short read".into() }),
            }
            let expect_prev = self.prev.unwrap_or_else(|| rd_u64(&buf, 16));
            match decode(&buf, self.pos, expect_prev) {
                Ok(rec) => {
                    self.prev = Some(rec.lsn);
                    self.pos = rec.end_lsn;
                    if rec.rmgr == rmgr::XLOG && rec.info == xlog::SWITCH {
                        self.pos = seg_end;
                    }
                    self.valid_end = self.pos;
                    return Ok(rec);
                }
                Err(RecordError::Invalid(why)) => return Err(ReadStop::Invalid { pos: self.pos, why }),
            }
        }
    }
}
