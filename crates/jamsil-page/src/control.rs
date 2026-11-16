//! Control file: two 4 KiB slots, alternately written (spec 6.6).

use crate::*;

pub const SLOT_LEN: usize = 4096;
pub const CONTROL_MAGIC: u64 = 0x4A41_4D53_494C_4342; // "JAMSILCB"
pub const FORMAT_VERSION: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DbState {
    ShutDown = 0,
    InProduction = 1,
    InRecovery = 2,
    InstallingSnapshot = 3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlData {
    pub format_version: u32,
    pub page_size: u32,
    pub wal_segment_size: u32,
    pub system_id: u64,
    pub cluster_id: [u8; 64],
    pub generation: u64,
    pub state: DbState,
    pub checkpoint_lsn: Lsn,
    pub redo_lsn: Lsn,
    pub next_xid: Xid,
    pub next_oid: Oid,
    pub next_relfile: RelFileId,
    pub oldest_clog_xid: Xid,
    pub applied_index: u64,
    pub applied_term: u64,
    pub schema_version: u64,
}

impl ControlData {
    pub fn new(system_id: u64, cluster_id: &[u8]) -> Self {
        let mut cid = [0u8; 64];
        let n = cluster_id.len().min(64);
        cid[..n].copy_from_slice(&cluster_id[..n]);
        Self {
            format_version: FORMAT_VERSION,
            page_size: PAGE_SIZE as u32,
            wal_segment_size: WAL_SEGMENT_SIZE as u32,
            system_id,
            cluster_id: cid,
            generation: 0,
            state: DbState::ShutDown,
            checkpoint_lsn: 0,
            redo_lsn: 0,
            next_xid: FIRST_NORMAL_XID,
            next_oid: FIRST_USER_OID,
            next_relfile: FIRST_RELFILE,
            oldest_clog_xid: FIRST_NORMAL_XID,
            applied_index: 0,
            applied_term: 0,
            schema_version: 0,
        }
    }

    pub fn encode(&self) -> [u8; SLOT_LEN] {
        let mut b = [0u8; SLOT_LEN];
        let mut o = 0;
        macro_rules! put {
            ($v:expr, $n:expr) => {{
                b[o..o + $n].copy_from_slice(&$v.to_le_bytes());
                o += $n;
            }};
        }
        put!(CONTROL_MAGIC, 8);
        put!(self.format_version, 4);
        put!(self.page_size, 4);
        put!(self.wal_segment_size, 4);
        put!(self.system_id, 8);
        b[o..o + 64].copy_from_slice(&self.cluster_id);
        o += 64;
        put!(self.generation, 8);
        b[o] = self.state as u8;
        o += 1;
        put!(self.checkpoint_lsn, 8);
        put!(self.redo_lsn, 8);
        put!(self.next_xid, 8);
        put!(self.next_oid, 4);
        put!(self.next_relfile, 8);
        put!(self.oldest_clog_xid, 8);
        put!(self.applied_index, 8);
        put!(self.applied_term, 8);
        put!(self.schema_version, 8);
        debug_assert!(o < SLOT_LEN - 4);
        let crc = crc32c::crc32c(&b[..SLOT_LEN - 4]);
        wr_u32(&mut b, SLOT_LEN - 4, crc);
        b
    }

    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() < SLOT_LEN {
            return corrupt("short control slot");
        }
        if crc32c::crc32c(&b[..SLOT_LEN - 4]) != rd_u32(b, SLOT_LEN - 4) {
            return corrupt("control slot crc");
        }
        if rd_u64(b, 0) != CONTROL_MAGIC {
            return corrupt("control magic");
        }
        let mut o = 8;
        macro_rules! get {
            ($f:ident, $n:expr) => {{
                let v = $f(b, o);
                o += $n;
                v
            }};
        }
        let format_version = get!(rd_u32, 4);
        let page_size = get!(rd_u32, 4);
        let wal_segment_size = get!(rd_u32, 4);
        let system_id = get!(rd_u64, 8);
        let mut cluster_id = [0u8; 64];
        cluster_id.copy_from_slice(&b[o..o + 64]);
        o += 64;
        let generation = get!(rd_u64, 8);
        let state = match b[o] {
            0 => DbState::ShutDown,
            1 => DbState::InProduction,
            2 => DbState::InRecovery,
            3 => DbState::InstallingSnapshot,
            _ => return corrupt("control state"),
        };
        o += 1;
        let checkpoint_lsn = get!(rd_u64, 8);
        let redo_lsn = get!(rd_u64, 8);
        let next_xid = get!(rd_u64, 8);
        let next_oid = get!(rd_u32, 4);
        let next_relfile = get!(rd_u64, 8);
        let oldest_clog_xid = get!(rd_u64, 8);
        let applied_index = get!(rd_u64, 8);
        let applied_term = get!(rd_u64, 8);
        let schema_version = rd_u64(b, o);
        Ok(Self {
            format_version,
            page_size,
            wal_segment_size,
            system_id,
            cluster_id,
            generation,
            state,
            checkpoint_lsn,
            redo_lsn,
            next_xid,
            next_oid,
            next_relfile,
            oldest_clog_xid,
            applied_index,
            applied_term,
            schema_version,
        })
    }

    /// Picks the valid slot with the higher generation. Returns the data and the slot index (0/1).
    pub fn pick(file: &[u8]) -> Result<(Self, usize), Corrupt> {
        let a = file.get(..SLOT_LEN).and_then(|s| Self::decode(s).ok());
        let b = file.get(SLOT_LEN..2 * SLOT_LEN).and_then(|s| Self::decode(s).ok());
        match (a, b) {
            (Some(a), Some(b)) => Ok(if b.generation > a.generation { (b, 1) } else { (a, 0) }),
            (Some(a), None) => Ok((a, 0)),
            (None, Some(b)) => Ok((b, 1)),
            (None, None) => corrupt("no valid control slot"),
        }
    }
}
