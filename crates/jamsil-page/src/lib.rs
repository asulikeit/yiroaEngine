//! Byte layouts of pages, control file and WAL records (spec section 6, 8.2).
//! No I/O, no threads, no `unsafe`.
#![forbid(unsafe_code)]

pub mod control;
pub mod fuzzing;
pub mod heaprec;
pub mod ids;
pub mod page;
pub mod tuple;
pub mod txnrec;
pub mod walrec;

pub use ids::*;

pub const PAGE_SIZE: usize = 8192;
pub const WAL_SEGMENT_SIZE: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("corrupt: {0}")]
pub struct Corrupt(pub String);

pub fn corrupt<T>(msg: impl Into<String>) -> Result<T, Corrupt> {
    Err(Corrupt(msg.into()))
}

pub fn rd_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
pub fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
pub fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
pub fn wr_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
pub fn wr_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
pub fn wr_u64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use control::*;
    use page::*;
    use walrec::*;

    #[test]
    fn slotted_page_add_remove_compact() {
        let mut buf = new_page_buf();
        let mut p = PageMut(&mut buf[..]);
        p.init(PageKind::Heap, 0);
        let a = p.add_item(b"hello").unwrap();
        let b = p.add_item(b"world!!").unwrap();
        assert_eq!((a, b), (1, 2));
        assert_eq!(p.item(2).unwrap(), b"world!!");
        p.set_slot_state(1, SlotState::Dead);
        p.compact();
        assert_eq!(p.item(2).unwrap(), b"world!!");
        assert_eq!(p.add_item(b"x"), Some(3), "dead slot is not reused until unused");
        p.set_slot_state(1, SlotState::Unused);
        assert_eq!(p.add_item(b"yy"), Some(1));
        while p.add_item(&[7u8; 100]).is_some() {}
        assert!(p.free_space() < 104);
    }

    #[test]
    fn checksum_binds_page_to_location() {
        let tag = PageTag::new(5, ForkId::Main, 9);
        let mut buf = new_page_buf();
        PageMut(&mut buf[..]).init(PageKind::Heap, 0);
        set_checksum(&mut buf[..], &tag);
        verify(&buf[..], &tag).unwrap();
        assert!(verify(&buf[..], &PageTag::new(5, ForkId::Main, 10)).is_err());
        assert!(verify(&[0u8; PAGE_SIZE], &tag).is_err());
        buf[4000] ^= 1;
        assert!(verify(&buf[..], &tag).is_err());
    }

    #[test]
    fn a_valid_checksum_does_not_excuse_bad_header_bounds() {
        let tag = PageTag::new(5, ForkId::Main, 9);
        for (lower, upper, special) in [(20u16, 100u16, 8192u16), (200, 100, 8192), (32, 9000, 8192), (32, 8000, 7000), (32, 100, 9000)] {
            let mut buf = new_page_buf();
            PageMut(&mut buf[..]).init(PageKind::Heap, 0);
            wr_u16(&mut buf[..], 14, lower);
            wr_u16(&mut buf[..], 16, upper);
            wr_u16(&mut buf[..], 18, special);
            set_checksum(&mut buf[..], &tag);
            assert!(verify(&buf[..], &tag).is_err(), "bounds {lower}/{upper}/{special} must be Corrupt");
        }
    }

    #[test]
    fn image_hole_roundtrip() {
        let mut buf = new_page_buf();
        let mut p = PageMut(&mut buf[..]);
        p.init(PageKind::Heap, 24);
        p.add_item(&[1, 2, 3, 4]).unwrap();
        let (ho, hl, bytes) = strip_hole(&buf[..]);
        assert!(hl > 0 && bytes.len() < PAGE_SIZE);
        let mut out = new_page_buf();
        restore_hole(ho, hl, &bytes, &mut out[..]).unwrap();
        assert_eq!(&out[..], &buf[..]);
    }

    #[test]
    fn control_slot_roundtrip_and_torn_slot() {
        let mut c = ControlData::new(42, b"cluster");
        c.generation = 3;
        c.applied_index = 99;
        let enc = c.encode();
        assert_eq!(ControlData::decode(&enc).unwrap(), c);
        let mut older = c.clone();
        older.generation = 2;
        let mut file = vec![0u8; 2 * SLOT_LEN];
        file[..SLOT_LEN].copy_from_slice(&older.encode());
        file[SLOT_LEN..].copy_from_slice(&enc);
        assert_eq!(ControlData::pick(&file).unwrap().1, 1);
        file[SLOT_LEN + 100] ^= 0xFF; // tear the newer slot
        let (picked, slot) = ControlData::pick(&file).unwrap();
        assert_eq!((slot, picked.generation), (0, 2));
    }

    #[test]
    fn wal_record_roundtrip_and_validation() {
        let tag = PageTag::new(7, ForkId::Main, 3);
        let mut page = new_page_buf();
        PageMut(&mut page[..]).init(PageKind::Heap, 0);
        let data = vec![9u8; 13];
        let parts = RecordBuilder::new(rmgr::HEAP, 1)
            .xid(77)
            .block(BlockRef::new(tag).image(&page[..]).data(&data))
            .main(b"main-payload");
        let mut s = assemble(&parts);
        s.seal(1000, 900);
        let rec = decode(&s.bytes, 1000, 900).unwrap();
        assert_eq!(rec.xid, 77);
        assert_eq!(rec.main, b"main-payload");
        assert_eq!(rec.blocks[0].data, vec![9; 13]);
        let mut out = new_page_buf();
        rec.blocks[0].restore_image(&mut out[..]).unwrap();
        assert_eq!(&out[..], &page[..]);
        assert!(decode(&s.bytes, 1008, 900).is_err(), "wrong lsn");
        assert!(decode(&s.bytes, 1000, 901).is_err(), "wrong prev");
        let mut bad = s.bytes.clone();
        let n = bad.len();
        bad[n - 9] ^= 1;
        assert!(decode(&bad, 1000, 900).is_err(), "crc");
    }

    /// The length field of a block reference has 16 bits: 65,535 bytes fit, 65,536 would wrap to 0.
    #[test]
    fn block_data_is_bounded_by_its_16_bit_length_field() {
        let tag = PageTag::new(7, ForkId::Main, 3);
        let fits = vec![0xABu8; MAX_BLOCK_DATA];
        let mut s = assemble(&RecordBuilder::new(rmgr::HEAP, 1).block(BlockRef::new(tag).data(&fits)));
        s.seal(1000, 900);
        let rec = decode(&s.bytes, 1000, 900).unwrap();
        assert_eq!(rec.blocks[0].data.len(), 65_535, "the largest value of the field must survive");

        let too_big = vec![0xABu8; MAX_BLOCK_DATA + 1];
        let r = std::panic::catch_unwind(|| assemble(&RecordBuilder::new(rmgr::HEAP, 1).block(BlockRef::new(tag).data(&too_big))));
        assert!(r.is_err(), "65,536 bytes must be refused, not truncated to 0");
    }

    /// Spec 18.3: a thread keeps 64 KiB of assembly buffer between records, which covers a record with
    /// four page images. A larger record grows the buffer at the gate, for that record only, and the
    /// buffer shrinks back at the next gate. (Each test runs on its own thread, and the buffer is
    /// per thread.)
    #[test]
    fn the_assembly_buffer_keeps_64_kib_between_records() {
        let four_images = 4 * PAGE_SIZE + 4096;
        let kept = reserve_assembly_buffer(four_images);
        assert!(kept >= four_images && kept <= ASSEMBLY_KEEP, "a record with four images needs no more than the cap: {kept}");
        assert_eq!(reserve_assembly_buffer(four_images), kept, "and the second gate does not grow it");

        let huge = vec![7u8; 400 * 1024];
        let grown = reserve_assembly_buffer(huge.len() + 64);
        assert!(grown >= 400 * 1024, "a larger record grows the buffer at the gate");
        drop(assemble(&RecordBuilder::new(rmgr::HEAP, 1).main(&huge)));
        assert!(reserve_assembly_buffer(0) > 0);

        // The next gate is for a small record: the memory goes back.
        let back = reserve_assembly_buffer(1024);
        assert!(back <= ASSEMBLY_KEEP, "shrunk back to at most the cap, got {back}");
    }

    #[test]
    fn decoders_do_not_panic_on_garbage() {
        let mut x = 12345u64;
        for _ in 0..2000 {
            let len = (x % 300) as usize;
            let buf: Vec<u8> = (0..len)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 33) as u8
                })
                .collect();
            let _ = decode(&buf, 0, 0);
            let _ = ControlData::decode(&buf);
            let _ = verify(&buf, &PageTag::new(1, ForkId::Main, 0));
        }
    }
}
