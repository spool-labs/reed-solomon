//! THE WIRE-COMPAT GATE — the pass/fail bar for tape-reed-solomon.
//!
//! For a range of (k, m) shapes and shard sizes with seeded random data:
//!   (a) encode-parity: tape_rs parity bytes == reed-solomon-erasure parity bytes.
//!   (b) cross-impl reconstruct: encode with rse, erase up to m shards, rebuild
//!       with tape_rs, assert recovered == original (proves we decode already-
//!       stored parity, which a same-impl round-trip cannot).
//!   (c) round-trip: encode with tape_rs, erase, rebuild with tape_rs.
//!   (d) prepared decode: rebuild the rse-encoded stripe through a
//!       PreparedDecoder and through reconstruct_rows; both must match (b).

use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon as Rse;
use tape_reed_solomon::ReedSolomon as TapeRs;

const PARAMS: &[(usize, usize)] = &[
    (2, 2),
    (4, 4),
    (6, 4),
    (7, 13),
    (10, 7),
    (10, 10),
    (17, 17),
    (20, 10),
    (32, 32),
];
const SIZES: &[usize] = &[32, 1024, 32768];

fn random_shards(k: usize, m: usize, sz: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Vec::with_capacity(k + m);
    for _ in 0..k {
        let mut s = vec![0u8; sz];
        rng.fill_bytes(&mut s);
        v.push(s);
    }
    for _ in 0..m {
        v.push(vec![0u8; sz]);
    }
    v
}

/// Pick `e` distinct indices in `0..total` to erase.
fn pick_erasures(total: usize, e: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..total).collect();
    // Fisher-Yates partial shuffle.
    for i in 0..e {
        let j = i + (rng.next_u32() as usize) % (total - i);
        idx.swap(i, j);
    }
    let mut out = idx[0..e].to_vec();
    out.sort_unstable();
    out
}

fn reconstruct_with_tape(
    rs: &TapeRs,
    full: &[Vec<u8>],
    erasures: &[usize],
) -> Vec<Vec<u8>> {
    let mut shards = full.to_vec();
    // Zero out erased shard contents so a successful rebuild is meaningful.
    for &i in erasures {
        for b in shards[i].iter_mut() {
            *b = 0;
        }
    }
    {
        let mut opt: Vec<(&mut [u8], bool)> =
            shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
        for &i in erasures {
            opt[i].1 = false;
        }
        rs.reconstruct(&mut opt).unwrap();
    }
    shards
}

/// Rebuild through prepare_decode and through reconstruct_rows; both must
/// recover the original stripe.
fn reconstruct_prepared_and_rows(
    rs: &TapeRs,
    full: &[Vec<u8>],
    erasures: &[usize],
) -> bool {
    let total = full.len();
    let sz = full[0].len();
    let mut present = vec![true; total];
    for &i in erasures {
        present[i] = false;
    }

    let decoder = rs.prepare_decode(&present).unwrap();
    let mut shards = full.to_vec();
    for &i in erasures {
        shards[i].fill(0);
    }
    {
        let mut opt: Vec<(&mut [u8], bool)> =
            shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
        for &i in erasures {
            opt[i].1 = false;
        }
        decoder.reconstruct(&mut opt).unwrap();
    }
    let prepared_ok = shards == full;

    let mut rows = Vec::with_capacity(total * sz);
    for shard in full {
        rows.extend_from_slice(shard);
    }
    for &i in erasures {
        rows[i * sz..(i + 1) * sz].fill(0);
    }
    rs.reconstruct_rows(&mut rows, sz, &present).unwrap();
    let mut rows_ok = true;
    for (i, shard) in full.iter().enumerate() {
        if &rows[i * sz..(i + 1) * sz] != shard.as_slice() {
            rows_ok = false;
        }
    }

    prepared_ok && rows_ok
}

#[test]
fn wire_compat_gate() {
    let mut rng = StdRng::seed_from_u64(0xC1A7);
    let mut checked = 0usize;
    let mut passed_a = 0usize;
    let mut passed_b = 0usize;
    let mut passed_c = 0usize;
    let mut passed_d = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for &(k, m) in PARAMS {
        for &sz in SIZES {
            let base = random_shards(k, m, sz, &mut rng);
            let tape = TapeRs::new(k, m).unwrap();
            let rse = Rse::new(k, m).unwrap();

            // (a) encode-parity equality.
            let mut a = base.clone();
            let mut b = base.clone();
            tape.encode(&mut a).unwrap();
            rse.encode(&mut b).unwrap();
            let encode_ok = (k..k + m).all(|i| a[i] == b[i]);
            if encode_ok {
                passed_a += 1;
            } else {
                failures.push(format!("(a) encode-parity k={k} m={m} sz={sz}"));
            }

            // `a` (tape-encoded) and `b` (rse-encoded) are now full valid stripes;
            // they are equal, so use them as the golden originals.
            let original = b.clone();

            // Try several erasure patterns, including the maximal m erasures.
            let mut b_ok = true;
            let mut c_ok = true;
            let mut d_ok = true;
            for &e in &[0usize, 1, m / 2, m] {
                let erasures = pick_erasures(k + m, e, &mut rng);

                // (b) cross-impl: rse-encoded parity -> tape reconstruct.
                let rebuilt = reconstruct_with_tape(&tape, &original, &erasures);
                if rebuilt != original {
                    b_ok = false;
                }

                // (c) round-trip: tape-encoded -> tape reconstruct.
                let rebuilt2 = reconstruct_with_tape(&tape, &a, &erasures);
                if rebuilt2 != a {
                    c_ok = false;
                }

                // (d) prepared decoder + contiguous rows on the rse stripe.
                if !reconstruct_prepared_and_rows(&tape, &original, &erasures) {
                    d_ok = false;
                }
            }
            if b_ok {
                passed_b += 1;
            } else {
                failures.push(format!("(b) cross-impl reconstruct k={k} m={m} sz={sz}"));
            }
            if c_ok {
                passed_c += 1;
            } else {
                failures.push(format!("(c) round-trip k={k} m={m} sz={sz}"));
            }
            if d_ok {
                passed_d += 1;
            } else {
                failures.push(format!("(d) prepared/rows reconstruct k={k} m={m} sz={sz}"));
            }

            checked += 1;
        }
    }

    eprintln!(
        "wire-compat gate: {checked} (k,m)x size combos over {} shapes x {} sizes",
        PARAMS.len(),
        SIZES.len()
    );
    eprintln!("  (a) encode-parity:          {passed_a}/{checked}");
    eprintln!("  (b) cross-impl reconstruct: {passed_b}/{checked}");
    eprintln!("  (c) round-trip:             {passed_c}/{checked}");
    eprintln!("  (d) prepared/rows:          {passed_d}/{checked}");

    assert!(
        failures.is_empty(),
        "wire-compat gate FAILED:\n{}",
        failures.join("\n")
    );
}
