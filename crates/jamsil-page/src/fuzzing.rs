//! Entry points shared by the `cargo-fuzz` targets in `fuzz/` and the in-tree smoke test
//! (spec 18.5, 19 layer 1): no decoder may panic on arbitrary bytes, and whatever a decoder
//! accepts must survive a round trip.
//!
//! Each function takes arbitrary bytes and either returns or fails an assertion; a panic is a
//! finding.

use crate::control::*;
use crate::page::*;
use crate::txnrec::*;
use crate::walrec::*;
use crate::*;

fn pad<const N: usize>(data: &[u8]) -> Box<[u8; N]> {
    let mut b = Box::new([0u8; N]);
    let n = data.len().min(N);
    b[..n].copy_from_slice(&data[..n]);
    b
}

/// Page accessors, checksum verification and the hole round trip on an arbitrary page.
pub fn page_target(data: &[u8]) {
    let page = pad::<PAGE_SIZE>(data);
    let tag = PageTag::new(rd_u64(&pad::<8>(data)[..], 0) & 0xFFFF, ForkId::Main, 7);
    let _ = verify(&page[..], &tag);
    let _ = is_all_zero(&page[..]);
    let p = PageRef(&page[..]);
    let _ = (p.lsn(), p.flags(), p.lower(), p.upper(), p.special_start(), p.kind(), p.prune_xid(), p.free_space());
    let _ = p.special();
    let n = p.n_slots();
    assert!(n <= PAGE_SIZE / SLOT_SIZE);
    for s in 0..=(n.min(300) as u16 + 1) {
        let _ = p.slot(s);
        let _ = p.item(s);
    }
    let (ho, hl, bytes) = strip_hole(&page[..]);
    let mut out = [0u8; PAGE_SIZE];
    restore_hole(ho, hl, &bytes, &mut out).expect("a stripped page must restore");
    let (ho, hl) = (ho as usize, hl as usize);
    assert_eq!(out[..ho], page[..ho]);
    assert_eq!(out[ho + hl..], page[ho + hl..]);
    assert!(out[ho..ho + hl].iter().all(|&b| b == 0));
}

/// A scripted sequence of slotted-page operations on a page built by `init`: the page must stay
/// self-consistent and every stored item must read back unchanged.
pub fn page_ops_target(data: &[u8]) {
    let mut buf = new_page_buf();
    let kind = match data.first().copied().unwrap_or(0) % 3 {
        0 => PageKind::Heap,
        1 => PageKind::BtreeLeaf,
        _ => PageKind::BtreeInternal,
    };
    let special = (data.get(1).copied().unwrap_or(0) as usize % 4) * 8;
    PageMut(&mut buf[..]).init(kind, special);
    let mut model: Vec<(u16, Vec<u8>)> = Vec::new();
    let mut i = 2;
    // Bounded script: the invariant check after each step is linear in the page's item count.
    let end = data.len().min(2 + 400);
    while i + 1 < end {
        let (op, arg) = (data[i] % 5, data[i + 1]);
        i += 2;
        let mut p = PageMut(&mut buf[..]);
        match op {
            0 | 1 => {
                let len = (arg as usize) * (1 + (op as usize) * 8);
                let item = vec![arg; len];
                if let Some(s) = p.add_item(&item) {
                    model.retain(|(m, _)| *m != s);
                    model.push((s, item));
                }
            }
            2 => {
                if !model.is_empty() {
                    let (s, _) = model.remove(arg as usize % model.len());
                    p.set_slot_state(s, SlotState::Unused);
                }
            }
            3 => {
                p.compact();
            }
            _ => {
                let n = model.len().max(1);
                if let Some((s, item)) = model.get_mut(arg as usize % n) {
                    let shorter = item.len() / 2;
                    if p.overwrite_item(*s, &item[..shorter]) {
                        item.truncate(shorter);
                    }
                }
            }
        }
        let r = PageRef(&buf[..]);
        assert!(r.lower() >= HEADER_SIZE && r.lower() <= r.upper() && r.upper() <= r.special_start());
        for (s, item) in &model {
            assert_eq!(r.item(*s), Some(&item[..]), "slot {s}");
        }
    }
}

/// Raw and CRC-fixed WAL record bytes; a decoded record must re-encode to an equal record.
pub fn wal_record_target(data: &[u8]) {
    let lsn = rd_u64(&pad::<24>(data)[..], 8);
    let prev = rd_u64(&pad::<24>(data)[..], 16);
    let _ = decode(data, lsn, prev);
    // The CRC would reject almost every mutation before the parser sees it: fix up the length and
    // CRC so the fuzzer reaches the parsing of blocks.
    if data.len() >= REC_HEADER_LEN {
        let total = data.len() & !7;
        if total >= REC_HEADER_LEN {
            let mut buf = data[..total].to_vec();
            wr_u32(&mut buf, 0, total as u32);
            let mut crc = crc32c::crc32c(&buf[REC_HEADER_LEN..]);
            crc = crc32c::crc32c_append(crc, &buf[8..REC_HEADER_LEN]);
            wr_u32(&mut buf, 4, crc);
            let (lsn, prev) = (rd_u64(&buf, 8), rd_u64(&buf, 16));
            if let Ok(rec) = decode(&buf, lsn, prev) {
                assert_eq!(rec.end_lsn, lsn + total as u64);
                for b in &rec.blocks {
                    let mut out = [0u8; PAGE_SIZE];
                    let _ = b.restore_image(&mut out);
                }
            }
        }
    }
}

/// A record built from parts decodes to the same parts.
pub fn wal_roundtrip_target(data: &[u8]) {
    let mut it = data.iter().copied();
    let mut next = || it.next().unwrap_or(0);
    let (rm, info) = (next(), next());
    let xid = u64::from(next()) << 8 | u64::from(next());
    // The builder borrows its payloads, so they are made first.
    struct Blk {
        tag: PageTag,
        flags: u8,
        data: Vec<u8>,
        page: Option<Box<PageBuf>>,
    }
    let mut blks = Vec::new();
    for _ in 0..(next() % (MAX_BLOCKS as u8 + 1)) {
        let tag = PageTag::new(u64::from(next()), ForkId::from_u8(next() % 4).unwrap(), u32::from(next()));
        let flags = next() & (BLK_WILL_INIT | BLK_FORCED_IMAGE);
        let data: Vec<u8> = (0..next() as usize).map(|_| next()).collect();
        let page = if next() % 2 == 0 {
            let mut page = new_page_buf();
            PageMut(&mut page[..]).init(PageKind::Heap, 0);
            let _ = PageMut(&mut page[..]).add_item(&data);
            Some(page)
        } else {
            None
        };
        blks.push(Blk { tag, flags, data, page });
    }
    let main: Vec<u8> = (0..next() as usize * 3).map(|_| next()).collect();
    let mut parts = RecordBuilder::new(rm, info).xid(xid).main(&main);
    for b in &blks {
        let mut r = BlockRef::new(b.tag).flags(b.flags).data(&b.data);
        if let Some(p) = &b.page {
            r = r.image(&p[..]);
        }
        parts = parts.block(r);
    }
    let mut sealed = assemble(&parts);
    assert_eq!(sealed.total_len(), parts.encoded_len(), "the size check must agree with the encoder");
    assert!(parts.worst_case_len() >= parts.encoded_len(), "the bound before the critical section is an upper bound");
    sealed.seal(8192 + 32, 0);
    let rec = decode(&sealed.bytes, 8192 + 32, 0).expect("a record we built must decode");
    assert_eq!((rec.rmgr, rec.info, rec.xid, &rec.main[..]), (parts.rmgr, parts.info, parts.xid, parts.main));
    assert_eq!(rec.blocks.len(), parts.blocks().count());
    for (a, b) in rec.blocks.iter().zip(parts.blocks()) {
        let image = b.image.map(page::strip_hole);
        assert_eq!((a.tag, &a.data[..], &a.image), (b.tag, b.data, &image));
    }
}

/// Control-file slots: arbitrary bytes, and bytes with a repaired CRC so the field parser runs.
pub fn control_target(data: &[u8]) {
    let _ = ControlData::decode(data);
    let _ = ControlData::pick(data);
    let mut slot = pad::<SLOT_LEN>(data);
    let crc = crc32c::crc32c(&slot[..SLOT_LEN - 4]);
    wr_u32(&mut slot[..], SLOT_LEN - 4, crc);
    // Real magic as well, otherwise almost every input stops at the first field.
    wr_u64(&mut slot[..], 0, CONTROL_MAGIC);
    let crc = crc32c::crc32c(&slot[..SLOT_LEN - 4]);
    wr_u32(&mut slot[..], SLOT_LEN - 4, crc);
    if let Ok(cd) = ControlData::decode(&slot[..]) {
        assert_eq!(ControlData::decode(&cd.encode()[..]).as_ref(), Ok(&cd), "decode(encode(x)) must equal x");
    }
    let mut two = vec![0u8; 2 * SLOT_LEN];
    two[..SLOT_LEN].copy_from_slice(&slot[..]);
    let _ = ControlData::pick(&two);
}

/// WAL segment headers.
pub fn seg_header_target(data: &[u8]) {
    let id = rd_u64(&pad::<8>(data)[..], 0);
    let segno = rd_u64(&pad::<16>(data)[..], 8);
    let _ = check_seg_header(data, id, segno);
    let h = encode_seg_header(id, segno);
    check_seg_header(&h, id, segno).expect("a header we encoded must check");
    assert!(check_seg_header(&h, id, segno.wrapping_add(1)).is_err());
}

/// Every target, for the in-tree smoke test.
/// A fuzz target: its name and its body.
/// Transaction records (spec 8.7): arbitrary bytes into the three decoders. The decoders are strict
/// and their encodings canonical, so whatever decodes must encode back to the same bytes. A record
/// built from the bytes as parts decodes to itself.
pub fn txn_record_target(data: &[u8]) {
    if let Ok(c) = CommitRecord::decode(data) {
        assert_eq!(c.encode(), data, "a commit record that decodes must re-encode to the same bytes");
        let v = CommitView::parse(data).expect("the view accepts what the decoder accepts");
        assert_eq!((v.n_sub(), v.n_unlink()), (c.sub_xids.len(), c.unlink.len()));
    } else {
        assert!(CommitView::parse(data).is_err(), "the view and the decoder agree");
    }
    if let Ok(a) = AbortRecord::decode(data) {
        assert_eq!(a.encode(), data, "an abort record that decodes must re-encode to the same bytes");
    }
    if let Ok(p) = decode_applied_advance(data) {
        assert_eq!(&encode_applied_advance(p)[..], data);
    }
    if let Ok(b) = AllocBatch::decode(data) {
        assert_eq!(&b.encode()[..], data, "an allocation batch that decodes must re-encode to the same bytes");
    }
    if let Ok(x) = decode_clog_truncate(data) {
        assert_eq!(&encode_clog_truncate(x)[..], data);
    }
    // A record built from the bytes as parts decodes to itself.
    let mut it = data.iter().copied();
    let mut next = || it.next().unwrap_or(0);
    let take_u64 = |next: &mut dyn FnMut() -> u8| (0..8).fold(0u64, |a, _| a << 8 | u64::from(next()));
    let c = CommitRecord {
        raft: (next() % 2 == 0).then(|| RaftPos { index: take_u64(&mut next), term: take_u64(&mut next) }),
        commit_time: take_u64(&mut next),
        schema_version: (next() % 2 == 0).then(|| take_u64(&mut next)),
        sub_xids: (0..next() % 5).map(|_| take_u64(&mut next)).collect(),
        unlink: (0..next() % 5).map(|_| take_u64(&mut next)).collect(),
    };
    assert_eq!(CommitRecord::decode(&c.encode()).as_ref(), Ok(&c));
    let a = AbortRecord { sub_xids: c.sub_xids.clone(), created: c.unlink.clone() };
    assert_eq!(AbortRecord::decode(&a.encode()).as_ref(), Ok(&a));
}

/// The commit-log and CSN-log accessors (spec 12.2) on an arbitrary page and xid: no panic, an
/// error for a page of the wrong kind or the reserved value, and `get(set(x)) == x` on a formatted
/// page without touching any other xid.
pub fn clog_page_target(data: &[u8]) {
    let xid = data.get(..8).map_or(0, |b| rd_u64(b, 0));
    let rest = data.get(8..).unwrap_or(&[]);
    let arbitrary = pad::<PAGE_SIZE>(rest);
    let _ = clog_get(&arbitrary[..], xid);
    let _ = csn_get(&arbitrary[..], xid);
    let (_, _) = (clog_block(xid), csn_block(xid));
    let (byte, shift) = clog_slot(xid);
    assert!(byte < PAGE_SIZE && shift < 8);
    assert!(csn_slot(xid) + 8 <= PAGE_SIZE);

    let mut page = new_page_buf();
    init_clog_page(&mut page[..]);
    let other = xid.wrapping_add(1 + u64::from(data.get(8).copied().unwrap_or(0)));
    let (before_other, before_xid) = (clog_get(&page[..], other).unwrap(), clog_get(&page[..], xid).unwrap());
    assert_eq!((before_other, before_xid), (ClogStatus::InProgress, ClogStatus::InProgress));
    let status = [ClogStatus::InProgress, ClogStatus::Committed, ClogStatus::Aborted][data.get(9).copied().unwrap_or(0) as usize % 3];
    clog_set(&mut page[..], xid, status);
    assert_eq!(clog_get(&page[..], xid).unwrap(), status);
    if other % XIDS_PER_CLOG_PAGE != xid % XIDS_PER_CLOG_PAGE {
        assert_eq!(clog_get(&page[..], other).unwrap(), ClogStatus::InProgress, "another xid's bits are untouched");
    }
    let mut csns = new_page_buf();
    init_csn_page(&mut csns[..]);
    csn_set(&mut csns[..], xid, 0xDEAD_BEEF_0000 + u64::from(status as u8));
    assert_eq!(csn_get(&csns[..], xid).unwrap(), 0xDEAD_BEEF_0000 + u64::from(status as u8));
}


/// Payloads of the `HEAP` records (spec 10.3): arbitrary bytes into every decoder. Whatever decodes must
/// encode back to the same bytes, because the decoders are strict and the encodings canonical.
pub fn heap_record_target(data: &[u8]) {
    use crate::heaprec::*;
    if let Ok(m) = InsertMain::decode(data) {
        assert_eq!(&m.encode()[..], data);
    }
    if let Ok(m) = MultiInsertMain::decode(data) {
        assert_eq!(&m.encode()[..], data);
    }
    for lock in [false, true] {
        if let Ok(m) = DeleteMain::decode(data, lock) {
            assert_eq!(&m.encode()[..], data, "a DELETE or LOCK that decodes must re-encode to the same bytes");
        }
    }
    if let Ok(m) = UpdateMain::decode(data) {
        assert_eq!(&m.encode()[..], data);
        let _ = (m.vm_refs(false), m.vm_refs(true));
    }
    if let Ok(m) = PruneMain::decode(data) {
        assert_eq!(m.encode(), data);
        assert_eq!(m.encoded_len(), data.len());
    }
    let _ = decode_visible(data);
    // The data of a `MULTI_INSERT`: the count comes from the first two bytes, the rest is the data.
    if data.len() >= 2 {
        let n = rd_u16(data, 0);
        if let Ok(items) = decode_multi_insert_data(&data[2..], n) {
            assert_eq!(encode_multi_insert_data(&items), &data[2..]);
        }
    }
}

/// Heap tuples (spec 10.1, 10.2): arbitrary bytes decoded against column types taken from the input. A
/// tuple that decodes encodes back to the same bytes; and a row built from the input decodes to itself.
pub fn heap_tuple_target(data: &[u8]) {
    use crate::tuple::*;
    let Some((&k, rest)) = data.split_first() else { return };
    let ncols = (k % 24) as usize;
    let types: Vec<ColType> = (0..ncols)
        .map(|i| match rest.get(i).copied().unwrap_or(0) % 8 {
            0 => ColType::Bool,
            1 => ColType::Fixed(1),
            2 => ColType::Fixed(2),
            3 => ColType::Fixed(4),
            4 => ColType::Fixed(8),
            5 => ColType::Fixed(16),
            _ => ColType::Varlen,
        })
        .collect();
    let body = rest.get(ncols..).unwrap_or(&[]);
    let _ = TupleHdr::decode(body);
    if let Ok((h, vals)) = decode_tuple(body, &types) {
        let cols: Vec<(ColType, Value<'_>)> = types.iter().copied().zip(vals.iter().copied()).collect();
        assert_eq!(encode_tuple(&h, &cols).as_deref(), Ok(body), "a tuple that decodes must re-encode to the same bytes");
        assert_eq!(tuple_len(&cols), Ok(body.len()));
    }
    // A row built from the bytes decodes to itself.
    let mut pos = 0usize;
    let mut built: Vec<(ColType, Value<'_>)> = Vec::new();
    for t in &types {
        let sel = body.get(pos).copied().unwrap_or(0);
        pos += 1;
        let v = match (t, sel % 4) {
            (_, 0) => Value::Null,
            (ColType::Bool, _) => Value::Fixed(if sel & 4 == 0 { &[0] } else { &[1] }),
            (ColType::Fixed(n), _) => {
                let n = *n as usize;
                let end = (pos + n).min(body.len());
                if end - pos == n { Value::Fixed(&body[pos..end]) } else { Value::Null }
            }
            (ColType::Varlen, 1) => {
                let len = (sel as usize * 3).min(body.len().saturating_sub(pos));
                Value::Inline(&body[pos..pos + len])
            }
            (ColType::Varlen, _) => Value::External(ExternalPtr { first_block: rd_u32(&pad::<4>(&body[pos.min(body.len())..])[..], 0), raw_len: sel as u32, value_crc: 7 }),
        };
        built.push((*t, v));
    }
    let hdr = TupleHdr { xmin: 20, xmax: 0, cid: 0, ctid: Tid { block: 0, slot: 1 }, infomask: 0, infomask2: 0, hoff: 0 };
    let bytes = encode_tuple(&hdr, &built).expect("values built to fit their columns");
    let (_, vals) = decode_tuple(&bytes, &types).expect("what the encoder wrote must decode");
    assert_eq!(vals, built.iter().map(|(_, v)| *v).collect::<Vec<_>>());
}

pub type Target = (&'static str, fn(&[u8]));

pub const TARGETS: &[Target] = &[
    ("page", page_target),
    ("page_ops", page_ops_target),
    ("wal_record", wal_record_target),
    ("wal_roundtrip", wal_roundtrip_target),
    ("control", control_target),
    ("seg_header", seg_header_target),
    ("txn_record", txn_record_target),
    ("clog_page", clog_page_target),
    ("heap_record", heap_record_target),
    ("heap_tuple", heap_tuple_target),
];
