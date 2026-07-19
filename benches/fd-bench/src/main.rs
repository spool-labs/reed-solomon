// SPDX-License-Identifier: Apache-2.0
//! Same-run head-to-head at the production shape (7,13) and (10,10):
//! firedancer vs tape `encode_fused` vs sia vs reed-solomon-erasure(+simd-accel).
//! x86 only: firedancer is a C FFI. Run via run.sh on a GCP c3 (GFNI + AVX-512).
//!
//! Also reports byte-compat: does firedancer produce the SAME parity as
//! reed-solomon-erasure at (7,13)? (The prior gate only tested up to (32,32).)

use fd_reedsol::ReedSolomon as Fd;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon as Rse;
use sia_reed_solomon::ReedSolomon as Sia;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon as Tape;

// The two generated production shapes, then runtime shapes that exercise the
// staged and fused paths. (32,32) is Agave's turbine shred shape; (128,128) and
// (32,96) are Anza's evolution shapes (agave#9495) that exceed firedancer's
// 67-parity cap, so fd reports them unsupported.
const SHAPES: &[(usize, usize)] =
    &[(7, 13), (10, 10), (14, 14), (16, 16), (18, 6), (32, 32), (128, 128), (32, 96)];
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];

fn base(k: usize, m: usize, sz: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = (0..k)
        .map(|_| {
            let mut s = vec![0u8; sz];
            rng.fill_bytes(&mut s);
            s
        })
        .collect();
    for _ in 0..m {
        v.push(vec![0u8; sz]);
    }
    v
}

fn secs<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64()
}

/// firedancer encode into a fresh copy of `b`, or None if the shape/encode is
/// unsupported (so an unsupported (7,13) doesn't crash the whole bench).
fn fd_encode(k: usize, m: usize, b: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
    let mut v = b.to_vec();
    let mut fd = Fd::new(k, m).ok()?;
    fd.encode(&mut v).ok()?;
    Some(v)
}

fn main() {
    println!("=== firedancer head-to-head (encode), x86 ===");
    println!("arch: {}", std::env::consts::ARCH);

    for &(k, m) in SHAPES {
        // Byte-compat: firedancer/tape/sia parity vs reed-solomon-erasure.
        let mut rng = StdRng::seed_from_u64(0xF1AE);
        let b = base(k, m, 4096, &mut rng);
        let mut a = b.clone();
        Rse::new(k, m).unwrap().encode(&mut a).unwrap();
        let par = |x: &[Vec<u8>]| (k..k + m).all(|i| x[i] == a[i]);
        let mut t = b.clone();
        Tape::new(k, m).unwrap().encode(&mut t).unwrap();
        let mut s = b.clone();
        Sia::new(k, m).unwrap().encode(&mut s).unwrap();
        let fd_par = match fd_encode(k, m, &b) {
            Some(x) => {
                if par(&x) {
                    "MATCH"
                } else {
                    "DIFFERS"
                }
            }
            None => "unsupported",
        };
        println!(
            "\n--- shape ({k},{m}): parity vs rse — tape:{} sia:{} firedancer:{} ---",
            if par(&t) { "MATCH" } else { "DIFFERS" },
            if par(&s) { "MATCH" } else { "DIFFERS" },
            fd_par,
        );
        println!(
            "{:<8} {:>10} {:>12} {:>10} {:>12} {:>6}",
            "size", "rse(C)", "tapeFused", "sia", "firedancer", "iters"
        );

        for &sz in SIZES {
            let payload = (k * sz) as f64;
            let iters = (200_000_000usize / (k * sz)).clamp(20, 2000);
            let warmup = (iters / 10).clamp(3, 30);
            let b = base(k, m, sz, &mut rng);
            let mib = |secs: f64| payload * iters as f64 / (1024.0 * 1024.0) / secs;

            let rse = Rse::new(k, m).unwrap();
            let mut s = b.clone();
            let v_rse = mib(secs(|| { rse.encode(&mut s).unwrap(); }, warmup, iters));

            let tape = Tape::new(k, m).unwrap();
            let mut s = b.clone();
            let v_tape = mib(secs(|| { tape.encode_fused(&mut s).unwrap(); }, warmup, iters));

            let sia = Sia::new(k, m).unwrap();
            let mut s = b.clone();
            let v_sia = mib(secs(|| { sia.encode(&mut s).unwrap(); }, warmup, iters));

            let v_fd = match Fd::new(k, m) {
                Ok(mut fd) => {
                    let mut s = b.clone();
                    // prime once (fd_reedsol may lazily build tables)
                    let _ = fd.encode(&mut s);
                    let mut s = b.clone();
                    mib(secs(|| { let _ = fd.encode(&mut s); }, warmup, iters))
                }
                Err(_) => 0.0,
            };

            println!(
                "{:<8} {:>10.0} {:>12.0} {:>10.0} {:>12.0} {:>6}",
                size_label(sz), v_rse, v_tape, v_sia, v_fd, iters
            );
        }
    }
    println!("\npayload MiB/s (data bytes = k*size). tapeFused = ReedSolomon::encode_fused.");
    println!("firedancer 0 = Fd::new failed for that shape (unsupported).");
}

fn size_label(sz: usize) -> &'static str {
    match sz {
        100 => "100 B",
        1_000 => "1 KB",
        10_000 => "10 KB",
        100_000 => "100 KB",
        1_000_000 => "1 MB",
        _ => "?",
    }
}
