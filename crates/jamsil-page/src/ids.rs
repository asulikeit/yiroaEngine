//! Identifiers (spec 6.2).

pub type Oid = u32;
pub type RelFileId = u64;
pub type BlockNo = u32;
pub type Xid = u64;
pub type Lsn = u64;
pub type Csn = u64;

pub const FIRST_USER_OID: Oid = 16_384;
pub const INVALID_XID: Xid = 0;
pub const BOOTSTRAP_XID: Xid = 1;
pub const FIRST_NORMAL_XID: Xid = 16;
/// Relfile ids 1 and 2 are the commit log and the CSN log.
pub const CLOG_RELFILE: RelFileId = 1;
pub const CSNLOG_RELFILE: RelFileId = 2;
pub const FIRST_RELFILE: RelFileId = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ForkId {
    Main = 0,
    Fsm = 1,
    Vm = 2,
    Overflow = 3,
}

impl ForkId {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Main,
            1 => Self::Fsm,
            2 => Self::Vm,
            3 => Self::Overflow,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Fsm => "fsm",
            Self::Vm => "vm",
            Self::Overflow => "ovf",
        }
    }
}

/// Tuple id: block number then a 1-based slot number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tid {
    pub block: BlockNo,
    pub slot: u16,
}

impl Tid {
    pub fn to_bytes(self) -> [u8; 6] {
        let mut b = [0u8; 6];
        b[..4].copy_from_slice(&self.block.to_le_bytes());
        b[4..].copy_from_slice(&self.slot.to_le_bytes());
        b
    }
    pub fn from_bytes(b: &[u8]) -> Self {
        Self { block: u32::from_le_bytes(b[..4].try_into().unwrap()), slot: u16::from_le_bytes([b[4], b[5]]) }
    }
}

/// (RelFileId, ForkId, BlockNo): names one page on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PageTag {
    pub rel: RelFileId,
    pub fork: ForkId,
    pub block: BlockNo,
}

impl PageTag {
    pub fn new(rel: RelFileId, fork: ForkId, block: BlockNo) -> Self {
        Self { rel, fork, block }
    }
}
