// SPDX-License-Identifier: Apache-2.0
//! Head-to-head RS **encode** throughput: tape-reed-solomon vs sia_reed_solomon
//! vs reed-solomon-erasure — the last WITH `simd-accel`, i.e. the C SIMD backend
//! Clay actually ships on native (C NEON on aarch64, C AVX2/etc. on x86_64). So
//! every speedup below is the REAL native delta, not the scalar-rse number.
//!
//!   aarch64 (dev host): NEON.   x86_64 (GCP c3 Sapphire Rapids): AVX2/AVX-512/GFNI.
//!
//! Run here:    cargo run --release
//! Run on GCP:  ./run.sh     (see README.md)

use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon as Rse;
use sia_reed_solomon::ReedSolomon as Sia;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon as Tape;

// (7,13) = tape-internal PRODUCTION shape (ClayParams::DEFAULT n=20,k=7,d=16 -> m=13,
// RS original_count=k=7). (10,10) kept for continuity with earlier runs.
const SHAPES: &[(usize, usize)] = &[(7, 13), (10, 10)];
// per-plane (100 B–10 KB) through full-row / large (100 KB–4 MiB = sia's sweet spot).
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000, 4_194_304];

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

fn size_label(sz: usize) -> String {
    if sz % (1 << 20) == 0 {
        format!("{} MiB", sz >> 20)
    } else if sz >= 1000 {
        format!("{} KB", sz / 1000)
    } else {
        format!("{} B", sz)
    }
}

fn detected_features() -> String {
    let mut f: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("gfni") { f.push("gfni"); }
        if std::is_x86_feature_detected!("avx512f") { f.push("avx512f"); }
        if std::is_x86_feature_detected!("avx512bw") { f.push("avx512bw"); }
        if std::is_x86_feature_detected!("avx2") { f.push("avx2"); }
        if std::is_x86_feature_detected!("ssse3") { f.push("ssse3"); }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") { f.push("neon"); }
        if std::arch::is_aarch64_feature_detected!("sha3") { f.push("sha3"); }
    }
    if f.is_empty() { "scalar".into() } else { f.join(",") }
}

/// Wire-compat sanity: tape, sia and rse must all produce identical parity.
fn gate() {
    let mut rng = StdRng::seed_from_u64(0xC1A7);
    let params = [(2usize, 2usize), (6, 4), (10, 10), (20, 10), (17, 17)];
    let mut ok = true;
    for &(k, m) in &params {
        for &sz in &[64usize, 1024, 65536] {
            let b = base(k, m, sz, &mut rng);
            let (mut a_rse, mut a_tape, mut a_sia) = (b.clone(), b.clone(), b.clone());
            Rse::new(k, m).unwrap().encode(&mut a_rse).unwrap();
            Tape::new(k, m).unwrap().encode(&mut a_tape).unwrap();
            Sia::new(k, m).unwrap().encode(&mut a_sia).unwrap();
            let agree = (k..k + m).all(|i| a_rse[i] == a_tape[i] && a_rse[i] == a_sia[i]);
            ok &= agree;
            if !agree {
                println!("  GATE DIFFER at (k={k}, m={m}, sz={sz})");
            }
        }
    }
    println!(
        "wire-compat gate (tape == sia == reed-solomon-erasure parity): {}",
        if ok { "PASS" } else { "FAIL ***" }
    );
}

fn mib(payload: f64, iters: usize, secs: f64) -> f64 {
    payload * iters as f64 / (1024.0 * 1024.0) / secs
}

fn run_shape(k: usize, m: usize) {
    println!("\n--- shape ({k},{m}): {k} data + {m} parity — payload MiB/s (divide by rse(C) for speedup) ---");
    #[cfg(target_arch = "x86_64")]
    println!(
        "{:<7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11} {:>9} {:>6}",
        "size", "rse(C)", "ssse3", "avx2", "avx512", "gfni1", "gfniFused", "sia", "iters"
    );
    #[cfg(not(target_arch = "x86_64"))]
    println!(
        "{:<7} {:>10} {:>10} {:>10} {:>6}",
        "size", "rse(C)", "tape", "sia", "iters"
    );

    let mut rng = StdRng::seed_from_u64(0x5EED);
    for &sz in SIZES {
        let payload = (k * sz) as f64;
        let iters = (200_000_000usize / (k * sz)).clamp(20, 2000);
        let warmup = (iters / 10).clamp(3, 30);
        let b = base(k, m, sz, &mut rng);
        let m1 = |s: f64| mib(payload, iters, s);

        // Correctness before timing: sia and tape (and, on x86, every forced
        // single-output kernel + the fused path) must match rse byte-for-byte.
        let mut a = b.clone();
        Rse::new(k, m).unwrap().encode(&mut a).unwrap();
        let par = |x: &Vec<Vec<u8>>| (k..k + m).all(|i| x[i] == a[i]);
        {
            let mut s = b.clone();
            Sia::new(k, m).unwrap().encode(&mut s).unwrap();
            let mut t = b.clone();
            Tape::new(k, m).unwrap().encode(&mut t).unwrap();
            assert!(par(&s) && par(&t), "sia/tape parity mismatch ({k},{m}) sz={sz}");
        }

        let rse = Rse::new(k, m).unwrap();
        let mut s = b.clone();
        let v_rse = m1(secs(|| { rse.encode(&mut s).unwrap(); }, warmup, iters));
        let sia = Sia::new(k, m).unwrap();
        let mut s = b.clone();
        let v_sia = m1(secs(|| { sia.encode(&mut s).unwrap(); }, warmup, iters));
        let tape = Tape::new(k, m).unwrap();

        #[cfg(target_arch = "x86_64")]
        {
            // ssse3, avx2, avx512-nibble, gfni-single — each forced, same blocked loop.
            let mut kv = [0f64; 4];
            for kind in 0..4u8 {
                let mut f = b.clone();
                tape.encode_forced(&mut f, kind).unwrap();
                assert!(par(&f), "forced kind={kind} mismatch ({k},{m}) sz={sz}");
                let mut s = b.clone();
                kv[kind as usize] =
                    m1(secs(|| { tape.encode_forced(&mut s, kind).unwrap(); }, warmup, iters));
            }
            let mut fu = b.clone();
            tape.encode_fused(&mut fu).unwrap();
            assert!(par(&fu), "fused mismatch ({k},{m}) sz={sz}");
            let mut s = b.clone();
            let v_fu = m1(secs(|| { tape.encode_fused(&mut s).unwrap(); }, warmup, iters));
            println!(
                "{:<7} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>11.0} {:>9.0} {:>6}",
                size_label(sz), v_rse, kv[0], kv[1], kv[2], kv[3], v_fu, v_sia, iters
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut s = b.clone();
            let v_tape = m1(secs(|| { tape.encode(&mut s).unwrap(); }, warmup, iters));
            println!(
                "{:<7} {:>10.0} {:>10.0} {:>10.0} {:>6}",
                size_label(sz), v_rse, v_tape, v_sia, iters
            );
        }
    }
}

/// Ramp the CPU to a steady clock before measuring the first cell.
fn prime(ms: u64) {
    let rs = Tape::new(10, 10).unwrap();
    let mut s = base(10, 10, 10_000, &mut StdRng::seed_from_u64(1));
    let t = Instant::now();
    while (t.elapsed().as_millis() as u64) < ms {
        for _ in 0..64 {
            rs.encode(&mut s).unwrap();
        }
    }
    std::hint::black_box(&s);
}

fn main() {
    println!("=== tape-reed-solomon head-to-head (encode) ===");
    println!("arch: {}   detected: {}", std::env::consts::ARCH, detected_features());
    println!("baseline: reed-solomon-erasure galois_8 +simd-accel (the C SIMD backend Clay ships)");
    gate();
    prime(400);
    for &(k, m) in SHAPES {
        run_shape(k, m);
    }
    println!(
        "\npayload = data bytes only (k*size); speedup = column / rse(C). x86 columns:\n\
         ssse3/avx2/avx512 = nibble-split single-output kernels; gfni1 = GFNI single-output;\n\
         gfniFused = GFNI multi-output (encode_fused). aarch64: tape = dispatched encode."
    );
}
