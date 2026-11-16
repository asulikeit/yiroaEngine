//! The heap tuple (spec 10.1, 10.2): the 32-byte header and the row encoding.
//!
//! The decoders are strict, as 10.2 says: every violation is `Corrupt`, and nothing panics or overflows
//! whatever the bytes say. The encoding is canonical, so `encode(decode(bytes)) == bytes` for every input
//! that decodes; the fuzz target `heap_tuple` asserts it. Canonical includes four rules that 10.2 does not
//! spell out and that make one byte form of each value: `HAS_NULL` is set exactly when a bit of the bitmap
//! is set, `HAS_VARLEN` exactly when an inline variable-length value is present, `HAS_EXTERNAL` exactly
//! when an external pointer is, and the bytes of a tuple end where its last column ends.

use crate::*;

pub const TUPLE_HEADER_LEN: usize = 32;
/// A table has at most this many columns (spec 10.1); a wider definition fails with `54011`.
pub const MAX_COLUMNS: usize = 1_600;
/// Insert step 1 (spec 10.3): a tuple longer than this, as it would be stored inline, moves its largest
/// variable-length values to the overflow fork.
pub const MAX_INLINE_TUPLE: usize = 4_000;
/// Length of an external pointer (spec 10.2).
pub const EXTERNAL_PTR_LEN: usize = 16;
pub const EXTERNAL_TAG: u8 = 0xC1;

/// The `infomask` bits of spec 10.1, in the order the specification lists them: bits 0 to 9; bits 10 to
/// 15 are zero.
pub mod infomask {
    pub const HAS_NULL: u16 = 1 << 0;
    pub const HAS_VARLEN: u16 = 1 << 1;
    pub const HAS_EXTERNAL: u16 = 1 << 2;
    pub const XMIN_COMMITTED: u16 = 1 << 3;
    pub const XMIN_ABORTED: u16 = 1 << 4;
    pub const XMAX_COMMITTED: u16 = 1 << 5;
    pub const XMAX_ABORTED: u16 = 1 << 6;
    pub const XMAX_LOCK_ONLY: u16 = 1 << 7;
    pub const UPDATED: u16 = 1 << 8;
    pub const CID_IS_COMBO: u16 = 1 << 9;
    pub const ALL: u16 = (1 << 10) - 1;
    /// The two `xmax` hint bits: a writer that stamps a new `xmax` clears them (spec 10.3).
    pub const XMAX_HINTS: u16 = XMAX_COMMITTED | XMAX_ABORTED;
}

/// `infomask2`: bits 0 to 10 are the column count, bit 14 `HOT_UPDATED`, bit 15 `HEAP_ONLY`.
pub mod infomask2 {
    pub const NATTS_MASK: u16 = 0x07FF;
    pub const HOT_UPDATED: u16 = 1 << 14;
    pub const HEAP_ONLY: u16 = 1 << 15;
    pub const ALL: u16 = NATTS_MASK | HOT_UPDATED | HEAP_ONLY;
}

/// The 32-byte tuple header (spec 10.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TupleHdr {
    pub xmin: Xid,
    pub xmax: Xid,
    pub cid: u32,
    /// The next version; the tuple's own tid if it is the latest.
    pub ctid: Tid,
    pub infomask: u16,
    pub infomask2: u16,
    /// Offset of the first column from the tuple start: 32 plus the bitmap.
    pub hoff: u8,
}

/// Bytes of the null bitmap of `natts` columns.
pub fn bitmap_len(natts: usize) -> usize {
    natts.div_ceil(8)
}

impl TupleHdr {
    pub fn natts(&self) -> usize {
        (self.infomask2 & infomask2::NATTS_MASK) as usize
    }
    pub fn has(&self, bit: u16) -> bool {
        self.infomask & bit != 0
    }

    pub fn encode(&self) -> [u8; TUPLE_HEADER_LEN] {
        let mut b = [0u8; TUPLE_HEADER_LEN];
        wr_u64(&mut b, 0, self.xmin);
        wr_u64(&mut b, 8, self.xmax);
        wr_u32(&mut b, 16, self.cid);
        b[20..26].copy_from_slice(&self.ctid.to_bytes());
        wr_u16(&mut b, 26, self.infomask);
        wr_u16(&mut b, 28, self.infomask2);
        b[30] = self.hoff;
        b
    }

    /// Reads and checks the header of `b`, the whole tuple.
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() < TUPLE_HEADER_LEN {
            return corrupt("tuple shorter than its 32-byte header");
        }
        let (infomask, infomask2) = (rd_u16(b, 26), rd_u16(b, 28));
        if infomask & !infomask::ALL != 0 {
            return corrupt("tuple infomask has bits 10 to 15 set");
        }
        if infomask2 & !infomask2::ALL != 0 {
            return corrupt("tuple infomask2 has bits 11 to 13 set");
        }
        if b[31] != 0 {
            return corrupt("tuple header: the reserved byte is not zero");
        }
        // A version cannot be both committed and aborted, whichever side of the stamp it is.
        if infomask & (infomask::XMIN_COMMITTED | infomask::XMIN_ABORTED) == (infomask::XMIN_COMMITTED | infomask::XMIN_ABORTED)
            || infomask & infomask::XMAX_HINTS == infomask::XMAX_HINTS
        {
            return corrupt("tuple carries both a committed and an aborted hint");
        }
        let h = Self {
            xmin: rd_u64(b, 0),
            xmax: rd_u64(b, 8),
            cid: rd_u32(b, 16),
            ctid: Tid::from_bytes(&b[20..26]),
            infomask,
            infomask2,
            hoff: b[30],
        };
        if h.natts() > MAX_COLUMNS {
            return corrupt("tuple has more than 1,600 columns");
        }
        let want = TUPLE_HEADER_LEN + if h.has(infomask::HAS_NULL) { bitmap_len(h.natts()) } else { 0 };
        if h.hoff as usize != want {
            return corrupt("tuple hoff does not match the header and the bitmap");
        }
        // The bitmap is inside the tuple. (A fuzz finding: HAS_NULL with columns and a 32-byte tuple.)
        if h.hoff as usize > b.len() {
            return corrupt("tuple ends inside its null bitmap");
        }
        Ok(h)
    }
}

/// The physical kind of a column, as the heap needs it. What a column means (`char(n)`, `jsonb`) is not the
/// heap's business (spec 10.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColType {
    /// One byte, 0 or 1.
    Bool,
    /// A fixed width of 1, 2, 4, 8 or 16 bytes.
    Fixed(u8),
    /// Variable length: inline, or an external pointer.
    Varlen,
}

/// An external pointer (spec 10.2): 16 bytes, tag `0xC1`, `flags` zero in 1.0, two reserved zero bytes,
/// then `first_block`, `raw_len` and `value_crc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalPtr {
    pub first_block: u32,
    pub raw_len: u32,
    pub value_crc: u32,
}

impl ExternalPtr {
    pub fn encode(&self) -> [u8; EXTERNAL_PTR_LEN] {
        let mut b = [0u8; EXTERNAL_PTR_LEN];
        b[0] = EXTERNAL_TAG;
        wr_u32(&mut b, 4, self.first_block);
        wr_u32(&mut b, 8, self.raw_len);
        wr_u32(&mut b, 12, self.value_crc);
        b
    }
    pub fn decode(b: &[u8]) -> Result<Self, Corrupt> {
        if b.len() != EXTERNAL_PTR_LEN || b[0] != EXTERNAL_TAG {
            return corrupt("not an external pointer");
        }
        if b[1] != 0 {
            return corrupt("external pointer: flags are zero in 1.0");
        }
        if b[2] != 0 || b[3] != 0 {
            return corrupt("external pointer: the reserved bytes are not zero");
        }
        Ok(Self { first_block: rd_u32(b, 4), raw_len: rd_u32(b, 8), value_crc: rd_u32(b, 12) })
    }
}

/// One column value of a tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value<'a> {
    Null,
    /// The bytes of a `Bool` or `Fixed` column.
    Fixed(&'a [u8]),
    /// The payload of an inline variable-length value.
    Inline(&'a [u8]),
    External(ExternalPtr),
}

/// The longest payload a four-byte header can hold: 24 bits.
pub const MAX_INLINE_PAYLOAD: usize = (1 << 24) - 1;

fn varlen_header_len(payload: usize) -> usize {
    if payload <= 127 {
        1
    } else {
        4
    }
}

fn value_len(t: ColType, v: &Value<'_>) -> Result<usize, Corrupt> {
    Ok(match (t, v) {
        (_, Value::Null) => 0,
        (ColType::Bool, Value::Fixed(b)) if b.len() == 1 && b[0] <= 1 => 1,
        (ColType::Fixed(n), Value::Fixed(b)) if b.len() == n as usize && matches!(n, 1 | 2 | 4 | 8 | 16) => b.len(),
        (ColType::Varlen, Value::Inline(p)) if p.len() <= MAX_INLINE_PAYLOAD => varlen_header_len(p.len()) + p.len(),
        (ColType::Varlen, Value::External(_)) => EXTERNAL_PTR_LEN,
        _ => return corrupt("a value does not fit its column"),
    })
}

/// The length of the tuple as it would be stored inline (spec 10.3 insert step 1): header, bitmap and every
/// column. Fails if a value does not fit its column type.
pub fn tuple_len(cols: &[(ColType, Value<'_>)]) -> Result<usize, Corrupt> {
    let nulls = cols.iter().any(|(_, v)| *v == Value::Null);
    let mut n = TUPLE_HEADER_LEN + if nulls { bitmap_len(cols.len()) } else { 0 };
    for (t, v) in cols {
        n += value_len(*t, v)?;
    }
    Ok(n)
}

/// Encodes a tuple. `hdr` supplies `xmin`, `xmax`, `cid`, `ctid` and the status bits of `infomask` and
/// `infomask2`; the column count, `HAS_NULL`, `HAS_VARLEN`, `HAS_EXTERNAL` and `hoff` are computed here,
/// and any value that does not fit its type is an error.
pub fn encode_tuple(hdr: &TupleHdr, cols: &[(ColType, Value<'_>)]) -> Result<Vec<u8>, Corrupt> {
    if cols.len() > MAX_COLUMNS {
        return corrupt("more than 1,600 columns (54011)");
    }
    let total = tuple_len(cols)?;
    let nulls = cols.iter().any(|(_, v)| *v == Value::Null);
    let mut h = *hdr;
    h.infomask &= !(infomask::HAS_NULL | infomask::HAS_VARLEN | infomask::HAS_EXTERNAL);
    if nulls {
        h.infomask |= infomask::HAS_NULL;
    }
    if cols.iter().any(|(_, v)| matches!(v, Value::Inline(_))) {
        h.infomask |= infomask::HAS_VARLEN;
    }
    if cols.iter().any(|(_, v)| matches!(v, Value::External(_))) {
        h.infomask |= infomask::HAS_EXTERNAL;
    }
    h.infomask2 = (h.infomask2 & !infomask2::NATTS_MASK) | cols.len() as u16;
    h.hoff = (TUPLE_HEADER_LEN + if nulls { bitmap_len(cols.len()) } else { 0 }) as u8;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&h.encode());
    if nulls {
        let mut bm = vec![0u8; bitmap_len(cols.len())];
        for (i, (_, v)) in cols.iter().enumerate() {
            if *v == Value::Null {
                bm[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend_from_slice(&bm);
    }
    for (_, v) in cols {
        match v {
            Value::Null => {}
            Value::Fixed(b) => out.extend_from_slice(b),
            Value::Inline(p) if p.len() <= 127 => {
                out.push(p.len() as u8);
                out.extend_from_slice(p);
            }
            Value::Inline(p) => {
                out.extend_from_slice(&(0x80u32 | ((p.len() as u32) << 8)).to_le_bytes());
                out.extend_from_slice(p);
            }
            Value::External(x) => out.extend_from_slice(&x.encode()),
        }
    }
    debug_assert_eq!(out.len(), total);
    Ok(out)
}

/// Decodes a whole tuple against the column types of its table. A tuple may hold fewer columns than the
/// table has now (the missing ones read as the stored default, spec 15.4): the result has one value per
/// column the tuple holds. More columns than `types` is `Corrupt`.
pub fn decode_tuple<'a>(b: &'a [u8], types: &[ColType]) -> Result<(TupleHdr, Vec<Value<'a>>), Corrupt> {
    let h = TupleHdr::decode(b)?;
    let natts = h.natts();
    if natts > types.len() {
        return corrupt("tuple has more columns than its table");
    }
    let has_null = h.has(infomask::HAS_NULL);
    let bitmap = &b[TUPLE_HEADER_LEN..h.hoff as usize];
    if has_null {
        if bitmap.len() != bitmap_len(natts) {
            return corrupt("null bitmap has the wrong length");
        }
        let spare = natts % 8;
        if spare != 0 && bitmap[bitmap.len() - 1] >> spare != 0 {
            return corrupt("unused bits of the null bitmap are not zero");
        }
        if bitmap.iter().all(|&x| x == 0) {
            return corrupt("HAS_NULL is set but no column is null");
        }
    }
    let mut pos = h.hoff as usize;
    let mut out = Vec::with_capacity(natts);
    let (mut any_inline, mut any_external) = (false, false);
    for (i, t) in types.iter().take(natts).enumerate() {
        if has_null && bitmap[i / 8] >> (i % 8) & 1 == 1 {
            out.push(Value::Null);
            continue;
        }
        let rest = &b[pos..];
        let v = match t {
            ColType::Bool => {
                let Some(&x) = rest.first() else { return corrupt("tuple ends inside a column") };
                if x > 1 {
                    return corrupt("a boolean is 0 or 1");
                }
                pos += 1;
                Value::Fixed(&rest[..1])
            }
            ColType::Fixed(n) => {
                if !matches!(n, 1 | 2 | 4 | 8 | 16) {
                    return corrupt("unsupported fixed width");
                }
                let n = *n as usize;
                let Some(x) = rest.get(..n) else { return corrupt("tuple ends inside a column") };
                pos += n;
                Value::Fixed(x)
            }
            ColType::Varlen => {
                let Some(&tag) = rest.first() else { return corrupt("tuple ends inside a column") };
                if tag & 0x80 == 0 {
                    let n = tag as usize;
                    let Some(p) = rest.get(1..1 + n) else { return corrupt("tuple ends inside a value") };
                    pos += 1 + n;
                    any_inline = true;
                    Value::Inline(p)
                } else if tag & 0xC0 == 0x80 {
                    if tag != 0x80 {
                        return corrupt("the low bits of a four-byte header are reserved and zero");
                    }
                    let Some(hd) = rest.get(..4) else { return corrupt("tuple ends inside a header") };
                    let n = (rd_u32(hd, 0) >> 8) as usize;
                    if n <= 127 {
                        return corrupt("a four-byte header for a payload of 127 bytes or fewer is refused");
                    }
                    let Some(p) = rest.get(4..4 + n) else { return corrupt("tuple ends inside a value") };
                    pos += 4 + n;
                    any_inline = true;
                    Value::Inline(p)
                } else if tag == EXTERNAL_TAG {
                    let Some(x) = rest.get(..EXTERNAL_PTR_LEN) else { return corrupt("tuple ends inside an external pointer") };
                    pos += EXTERNAL_PTR_LEN;
                    any_external = true;
                    Value::External(ExternalPtr::decode(x)?)
                } else {
                    return corrupt("a tag byte of the form 11xxxxxx other than 0xC1 is refused");
                }
            }
        };
        out.push(v);
    }
    if pos != b.len() {
        return corrupt("a tuple ends where its last column ends: bytes follow");
    }
    if h.has(infomask::HAS_VARLEN) != any_inline || h.has(infomask::HAS_EXTERNAL) != any_external {
        return corrupt("HAS_VARLEN or HAS_EXTERNAL does not match the columns");
    }
    Ok((h, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn hdr() -> TupleHdr {
        TupleHdr { xmin: 20, xmax: 0, cid: 3, ctid: Tid { block: 7, slot: 2 }, infomask: infomask::XMIN_COMMITTED, infomask2: 0, hoff: 0 }
    }

    #[test]
    fn the_header_has_the_layout_of_the_specification() {
        let h = TupleHdr { xmin: 0x11, xmax: 0x22, cid: 0x33, ctid: Tid { block: 0x44, slot: 0x55 }, infomask: 0x0A, infomask2: 0x4003, hoff: 32 };
        let b = h.encode();
        assert_eq!([rd_u64(&b, 0), rd_u64(&b, 8)], [0x11, 0x22]);
        assert_eq!(rd_u32(&b, 16), 0x33);
        assert_eq!((rd_u32(&b, 20), rd_u16(&b, 24)), (0x44, 0x55), "ctid: block, then a 1-based slot");
        assert_eq!((rd_u16(&b, 26), rd_u16(&b, 28), b[30], b[31]), (0x0A, 0x4003, 32, 0));
        assert_eq!(TupleHdr::decode(&b).unwrap(), h);
    }

    #[test]
    fn the_header_decoder_is_strict() {
        let h = TupleHdr { xmin: 1, xmax: 0, cid: 0, ctid: Tid { block: 0, slot: 1 }, infomask: 0, infomask2: 0, hoff: 32 };
        let good = h.encode();
        assert!(TupleHdr::decode(&good).is_ok());
        assert!(TupleHdr::decode(&good[..31]).is_err());
        let with = |f: &dyn Fn(&mut [u8; 32])| {
            let mut b = good;
            f(&mut b);
            TupleHdr::decode(&b)
        };
        assert!(with(&|b| b[27] |= 0x04).is_err(), "infomask bit 10");
        assert!(with(&|b| b[29] |= 0x08).is_err(), "infomask2 bit 11");
        assert!(with(&|b| b[29] |= 0x20).is_err(), "infomask2 bit 13");
        assert!(with(&|b| b[31] = 1).is_err(), "the reserved byte");
        assert!(with(&|b| b[30] = 33).is_err(), "hoff without a bitmap");
        assert!(with(&|b| wr_u16(b, 26, infomask::XMIN_COMMITTED | infomask::XMIN_ABORTED)).is_err());
        assert!(with(&|b| wr_u16(b, 26, infomask::XMAX_HINTS)).is_err());
        assert!(with(&|b| wr_u16(b, 28, 1601)).is_err(), "more than 1,600 columns");
        // A fuzz finding: the header says there is a bitmap and the tuple has no room for it.
        let mut nullable = good;
        wr_u16(&mut nullable, 26, infomask::HAS_NULL);
        wr_u16(&mut nullable, 28, 1);
        nullable[30] = 33;
        assert!(TupleHdr::decode(&nullable).is_err(), "hoff beyond the end of the tuple");
        let mut with_bitmap = nullable.to_vec();
        with_bitmap.push(0b1);
        assert!(TupleHdr::decode(&with_bitmap).is_ok());
        // Bit 14 and 15 are the HOT bits and are allowed.
        assert!(with(&|b| wr_u16(b, 28, infomask2::HOT_UPDATED | infomask2::HEAP_ONLY)).is_ok());
    }

    fn sample<'a>(payload: &'a [u8], big: &'a [u8]) -> Vec<(ColType, Value<'a>)> {
        vec![
            (ColType::Bool, Value::Fixed(&[1])),
            (ColType::Fixed(8), Value::Null),
            (ColType::Varlen, Value::Inline(payload)),
            (ColType::Fixed(4), Value::Fixed(&[1, 2, 3, 4])),
            (ColType::Varlen, Value::Inline(big)),
            (ColType::Varlen, Value::External(ExternalPtr { first_block: 9, raw_len: 70_000, value_crc: 0xABCD })),
            (ColType::Varlen, Value::Null),
        ]
    }

    #[test]
    fn a_row_round_trips_and_the_headers_are_the_shortest_form() {
        let (small, big) = (vec![7u8; 127], vec![9u8; 128]);
        let cols = sample(&small, &big);
        let types: Vec<_> = cols.iter().map(|(t, _)| *t).collect();
        let bytes = encode_tuple(&hdr(), &cols).unwrap();
        assert_eq!(bytes.len(), tuple_len(&cols).unwrap());
        let (h, vals) = decode_tuple(&bytes, &types).unwrap();
        assert_eq!(h.natts(), 7);
        assert!(h.has(infomask::HAS_NULL | infomask::HAS_VARLEN | infomask::HAS_EXTERNAL));
        assert_eq!(h.hoff as usize, 32 + 1, "seven columns need a one-byte bitmap");
        assert_eq!(vals, cols.iter().map(|(_, v)| *v).collect::<Vec<_>>());
        // Column 1 and column 6 are null: bits 1 and 6, from the low bit, a set bit meaning NULL.
        assert_eq!(bytes[32], 0b0100_0010);
        // 127 bytes use the one-byte header, 128 the four-byte one.
        assert_eq!(bytes[33 + 1], 127);
        let four = 33 + 1 + 1 + 127 + 4;
        assert_eq!(rd_u32(&bytes, four) & 0xFF, 0x80, "the low byte of the four-byte header is exactly 0x80");
        assert_eq!(rd_u32(&bytes, four) >> 8, 128);
    }

    #[test]
    fn a_row_without_nulls_has_no_bitmap_and_a_short_tuple_reads_fewer_columns() {
        let cols = vec![(ColType::Fixed(2), Value::Fixed(&[1, 0])), (ColType::Varlen, Value::Inline(b"ab"))];
        let bytes = encode_tuple(&hdr(), &cols).unwrap();
        assert_eq!(bytes.len(), 32 + 2 + 3);
        let h = TupleHdr::decode(&bytes).unwrap();
        assert!(!h.has(infomask::HAS_NULL) && h.hoff == 32);
        // The table has gained a column since: the tuple still decodes, with the columns it holds.
        let wider = [ColType::Fixed(2), ColType::Varlen, ColType::Bool];
        assert_eq!(decode_tuple(&bytes, &wider).unwrap().1.len(), 2);
        assert!(decode_tuple(&bytes, &wider[..1]).is_err(), "more columns than the table has");
    }

    #[test]
    fn the_row_decoder_is_strict_and_the_encoding_canonical() {
        let types = [ColType::Bool, ColType::Varlen, ColType::Fixed(2)];
        let cols = vec![(ColType::Bool, Value::Fixed(&[0])), (ColType::Varlen, Value::Inline(b"abc")), (ColType::Fixed(2), Value::Fixed(&[5, 6]))];
        let good = encode_tuple(&hdr(), &cols).unwrap();
        assert!(decode_tuple(&good, &types).is_ok());
        let bad = |f: &dyn Fn(&mut Vec<u8>)| {
            let mut b = good.clone();
            f(&mut b);
            decode_tuple(&b, &types).is_err()
        };
        assert!(bad(&|b| b[32] = 2), "a boolean is 0 or 1");
        assert!(bad(&|b| b.push(0)), "bytes after the last column");
        assert!(bad(&|b| {
            b.pop();
        }), "a column cut short");
        // A four-byte header for a payload of 3 bytes: refused, though it decodes to the same value.
        let mut long_form = good[..33].to_vec();
        long_form.extend_from_slice(&(0x80u32 | (3 << 8)).to_le_bytes());
        long_form.extend_from_slice(b"abc");
        long_form.extend_from_slice(&[5, 6]);
        assert!(decode_tuple(&long_form, &types).is_err(), "the encoder uses the shortest header, and the decoder refuses the other");
        // The low bits of the four-byte header are reserved.
        let big = vec![1u8; 200];
        let cols2 = vec![(ColType::Varlen, Value::Inline(&big))];
        let mut b2 = encode_tuple(&hdr(), &cols2).unwrap();
        assert!(decode_tuple(&b2, &[ColType::Varlen]).is_ok());
        b2[32] |= 0x01;
        assert!(decode_tuple(&b2, &[ColType::Varlen]).is_err());
        // Tags of the form 11xxxxxx other than 0xC1, and external flags.
        let ext = vec![(ColType::Varlen, Value::External(ExternalPtr { first_block: 1, raw_len: 2, value_crc: 3 }))];
        let mut b3 = encode_tuple(&hdr(), &ext).unwrap();
        assert!(decode_tuple(&b3, &[ColType::Varlen]).is_ok());
        b3[33] = 1;
        assert!(decode_tuple(&b3, &[ColType::Varlen]).is_err(), "external flags are zero in 1.0");
        b3[33] = 0;
        b3[32] = 0xC0;
        assert!(decode_tuple(&b3, &[ColType::Varlen]).is_err(), "0xC0 is not a tag");
        b3[32] = 0xFF;
        assert!(decode_tuple(&b3, &[ColType::Varlen]).is_err());
        // HAS_NULL without a null, a bitmap with a spare bit set, and flags that do not match the columns.
        let nulls = vec![(ColType::Bool, Value::Null), (ColType::Bool, Value::Fixed(&[1]))];
        let nb = encode_tuple(&hdr(), &nulls).unwrap();
        assert!(decode_tuple(&nb, &[ColType::Bool; 2]).is_ok());
        let mut spare = nb.clone();
        spare[32] |= 0b100;
        assert!(decode_tuple(&spare, &[ColType::Bool; 2]).is_err(), "a spare bit of the bitmap");
        let mut none = nb.clone();
        none[32] = 0;
        assert!(decode_tuple(&none, &[ColType::Bool; 2]).is_err(), "HAS_NULL with no null column");
        // The same rule where nothing else is wrong: one boolean column, a bitmap with no bit set, the value.
        let mut canon = TupleHdr { xmin: 20, xmax: 0, cid: 0, ctid: Tid { block: 0, slot: 1 }, infomask: infomask::HAS_NULL, infomask2: 1, hoff: 33 }.encode().to_vec();
        canon.extend_from_slice(&[0b0, 1]);
        assert!(decode_tuple(&canon, &[ColType::Bool]).is_err(), "HAS_NULL with a bitmap that has no bit set has a shorter, canonical form");
        canon[32] = 0b1;
        canon.pop();
        assert!(decode_tuple(&canon, &[ColType::Bool]).is_ok(), "and with the bit set it is a null column");
        let mut flag = good.clone();
        flag[26] |= infomask::HAS_EXTERNAL as u8;
        assert!(decode_tuple(&flag, &types).is_err());
    }

    #[test]
    fn random_rows_round_trip_byte_for_byte() {
        let mut r = Rng(5);
        let pool: Vec<u8> = (0..70_000u32).map(|i| (i * 7) as u8).collect();
        for _ in 0..1500 {
            let ncols = r.below(40) as usize;
            let mut types = Vec::new();
            let mut spec = Vec::new();
            for _ in 0..ncols {
                let k = r.below(5);
                let (t, v): (ColType, Value) = match k {
                    0 => (ColType::Bool, Value::Fixed(&pool[r.below(1) as usize..][..1])),
                    1 => (ColType::Fixed([1u8, 2, 4, 8, 16][r.below(5) as usize]), Value::Null),
                    2 => (ColType::Varlen, Value::Inline(&pool[..[0usize, 1, 127, 128, 129, 1000, 40_000][r.below(7) as usize]])),
                    3 => (ColType::Varlen, Value::External(ExternalPtr { first_block: r.next() as u32, raw_len: r.next() as u32, value_crc: r.next() as u32 })),
                    _ => (ColType::Varlen, Value::Null),
                };
                types.push(t);
                spec.push((t, v));
            }
            let bytes = encode_tuple(&hdr(), &spec).unwrap();
            let (h, vals) = decode_tuple(&bytes, &types).unwrap();
            assert_eq!(vals, spec.iter().map(|(_, v)| *v).collect::<Vec<_>>());
            let again = encode_tuple(&h, &spec).unwrap();
            assert_eq!(again, bytes, "encode(decode(x)) == x");
            assert_eq!(bytes.len(), tuple_len(&spec).unwrap());
        }
    }

    #[test]
    fn a_value_that_does_not_fit_its_column_is_refused_and_1601_columns_too() {
        assert!(encode_tuple(&hdr(), &[(ColType::Bool, Value::Fixed(&[2, 0]))]).is_err());
        assert!(encode_tuple(&hdr(), &[(ColType::Bool, Value::Fixed(&[2]))]).is_err(), "a boolean is 0 or 1");
        assert!(encode_tuple(&hdr(), &[(ColType::Fixed(3), Value::Fixed(&[0, 0, 0]))]).is_err());
        assert!(encode_tuple(&hdr(), &[(ColType::Fixed(4), Value::Inline(b"x"))]).is_err());
        let many = vec![(ColType::Bool, Value::Fixed(&[0])); 1601];
        assert!(encode_tuple(&hdr(), &many).is_err(), "54011");
        let max = vec![(ColType::Bool, Value::Null); 1600];
        let b = encode_tuple(&hdr(), &max).unwrap();
        assert_eq!(b[30] as usize, 32 + 200, "hoff never exceeds 232 and fits its byte");
        assert!(decode_tuple(&b, &vec![ColType::Bool; 1600]).is_ok());
    }
}
