//! A deterministic stand-in for the `cargo-fuzz` run: every fuzz target on a few hundred thousand
//! generated inputs (random bytes, and valid encodings with random mutations). It runs on stable
//! on every commit; `fuzz/` runs the same targets under libFuzzer.

use jamsil_page::control::ControlData;
use jamsil_page::fuzzing::TARGETS;
use jamsil_page::page::*;
use jamsil_page::walrec::*;
use jamsil_page::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    fn mutate(&mut self, b: &mut Vec<u8>) {
        for _ in 0..1 + self.below(6) {
            match self.below(4) {
                0 if !b.is_empty() => {
                    let i = self.below(b.len());
                    b[i] ^= 1 << self.below(8);
                }
                1 if !b.is_empty() => {
                    let i = self.below(b.len());
                    b[i] = self.next() as u8;
                }
                2 => {
                    let n = self.below(b.len() + 1);
                    b.truncate(n);
                }
                _ => {
                    let i = self.below(b.len() + 1);
                    let n = 1 + self.below(16);
                    let extra = self.bytes(n);
                    b.splice(i..i, extra);
                }
            }
        }
    }
}

fn seeds(rng: &mut Rng) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let tag = PageTag::new(9, ForkId::Main, 3);
    let mut page = new_page_buf();
    PageMut(&mut page[..]).init(PageKind::Heap, 0);
    for i in 0..20 {
        let _ = PageMut(&mut page[..]).add_item(&vec![i as u8; 40 + i]);
    }
    set_checksum(&mut page[..], &tag);
    out.push(page.to_vec());
    let data = rng.bytes(33);
    let rec = RecordBuilder::new(rmgr::HEAP, 3)
        .xid(77)
        .main(b"hello world")
        .block(BlockRef::new(tag).image(&page[..]))
        .block(BlockRef::new(PageTag::new(1, ForkId::Fsm, 2)).data(&data));
    let mut sealed = assemble(&rec);
    sealed.seal(8224, 8192);
    out.push(sealed.bytes.clone());
    out.push(ControlData::new(42, b"cluster").encode().to_vec());
    out.push(encode_seg_header(42, 7).to_vec());
    out.push(Vec::new());
    out
}

#[test]
fn no_decoder_panics_and_accepted_input_round_trips() {
    let mut rng = Rng(0xFEED);
    let seeds = seeds(&mut rng);
    for (name, target) in TARGETS {
        for round in 0..6000 {
            let input = match round % 3 {
                0 => {
                    let n = rng.below(if round % 9 == 0 { 9000 } else { 300 });
                    rng.bytes(n)
                }
                _ => {
                    let mut b = seeds[rng.below(seeds.len())].clone();
                    rng.mutate(&mut b);
                    b
                }
            };
            let r = std::panic::catch_unwind(|| target(&input));
            if r.is_err() {
                let shown: Vec<u8> = input.iter().copied().take(64).collect();
                panic!("target {name} panicked on a {}-byte input starting {shown:?}", input.len());
            }
        }
    }
}

/// Inputs libFuzzer found that once panicked a target.
#[test]
fn fuzz_findings_stay_fixed() {
    let findings: &[(&str, &[u8])] = &[
        // heap_tuple, found by libFuzzer in round 10: HAS_NULL with columns and a 32-byte tuple (hoff 33).
        (
            "heap_tuple",
            &[
                0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2a, 0x0, 0x0, 0x0, 0x0, 0xff, 0xef, 0xff, 0xfb, 0x0, 0x0, 0xff, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff,
                0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x1, 0x1, 0x1, 0x0, 0x21, 0x0,
            ],
        ),
        (
        "wal_record",
        &[
            185, 0, 0, 0, 0, 16, 16, 184, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 43, 184, 184, 184, 184,
            58, 43, 0, 0, 184, 1, 0, 0, 0, 0, 0, 0, 16, 0, 0, 184, 184, 1, 0, 0, 0, 0, 0, 0, 16, 16, 184,
        ],
    ),
    ];
    for (name, input) in findings {
        let (_, target) = TARGETS.iter().find(|(n, _)| n == name).unwrap();
        target(input);
    }
}
