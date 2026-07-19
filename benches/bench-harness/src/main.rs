// SPDX-License-Identifier: Apache-2.0
//! Head-to-head RS **encode** throughput: tape-reed-solomon vs sia_reed_solomon
//! vs reed-solomon-erasure, the last WITH `simd-accel`, i.e. the C SIMD backend
//! the production system actually ships on native (C NEON on aarch64, C AVX2/etc. on x86_64). So
//! every speedup below is the REAL native delta, not the scalar-rse number.
//!
//!   aarch64 (dev host): NEON.   x86_64 (GCP c3 Sapphire Rapids): AVX2/AVX-512/GFNI.
//!
//! Run here:    cargo run --release
//! Run on GCP:  ./run.sh     (see README.md)

use additive_fft_reed_solomon::{Gf2p8_11d, Rs as Afft};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon as Rse;
use sia_reed_solomon::ReedSolomon as Sia;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon as Tape;

// (7,13) = the production shape (production profile n=20, k=7, d=16 -> m=13,
// RS original_count=k=7). (10,10) kept for continuity with earlier runs.
const SHAPES: &[(usize, usize)] = &[(7, 13), (10, 10)];
// production-reachable shapes beyond the two production ones, e.g. a profile
// (20,6,19) -> RS(14,14) through shortening; all five carry generated
// programs now.
const EXTRA_SHAPES: &[(usize, usize)] = &[(14, 14), (16, 16), (18, 6), (64, 64)];
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

/// additive-fft-reed-solomon at the benched shape, when representable: its
/// codec is const-generic with power-of-two N and T, so most of our shapes
/// do not exist for it. Different wire (Cantor basis), so throughput only.
fn afft_encode_mib(k: usize, m: usize, sz: usize, warmup: usize, iters: usize) -> Option<f64> {
    match (k, m) {
        (16, 16) => Some(afft_run::<32, 16>(k, sz, warmup, iters)),
        (64, 64) => Some(afft_run::<128, 64>(k, sz, warmup, iters)),
        _ => None,
    }
}

fn afft_run<const N: usize, const T: usize>(
    k: usize,
    sz: usize,
    warmup: usize,
    iters: usize,
) -> f64 {
    let rs = Afft::<N, T>::new();
    let mut rng = StdRng::seed_from_u64(0xAFF7);
    let mut message = vec![Gf2p8_11d::from(0u8); k * sz];
    for symbol in message.iter_mut() {
        *symbol = Gf2p8_11d::from((rng.next_u32() & 0xff) as u8);
    }
    let mut parity = vec![Gf2p8_11d::from(0u8); T * sz];
    let mut workspace = vec![Gf2p8_11d::from(0u8); T * sz];
    let elapsed = secs(
        || rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, sz),
        warmup,
        iters,
    );
    mib((k * sz) as f64, iters, elapsed)
}

fn run_shape(k: usize, m: usize) {
    let route = Tape::new(k, m).unwrap().encode_route(10_000);
    println!("\n--- shape ({k},{m}): {k} data + {m} parity [tape route: {route}] — payload MiB/s (divide by rse(C) for speedup) ---");
    #[cfg(target_arch = "x86_64")]
    println!(
        "{:<7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11} {:>9} {:>9} {:>6}",
        "size", "rse(C)", "ssse3", "avx2", "avx512", "gfni1", "gfniFused", "sia", "afft", "iters"
    );
    #[cfg(not(target_arch = "x86_64"))]
    println!(
        "{:<7} {:>10} {:>10} {:>10} {:>10} {:>6}",
        "size", "rse(C)", "tape", "sia", "afft", "iters"
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
            // ssse3, avx2, avx512-nibble, gfni-single: each forced, same blocked loop.
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
            let afft = afft_encode_mib(k, m, sz, warmup, iters)
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".into());
            println!(
                "{:<7} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>11.0} {:>9.0} {:>9} {:>6}",
                size_label(sz), v_rse, kv[0], kv[1], kv[2], kv[3], v_fu, v_sia, afft, iters
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut s = b.clone();
            let v_tape = m1(secs(|| { tape.encode(&mut s).unwrap(); }, warmup, iters));
            let afft = afft_encode_mib(k, m, sz, warmup, iters)
                .map(|v| format!("{v:.0}"))
                .unwrap_or_else(|| "-".into());
            println!(
                "{:<7} {:>10.0} {:>10.0} {:>10.0} {:>10} {:>6}",
                size_label(sz), v_rse, v_tape, v_sia, afft, iters
            );
        }
    }
}

/// Reconstruct throughput for one erasure pattern: reed-solomon-erasure vs
/// tape's cached-plan reconstruct vs an explicit PreparedDecoder. Payload is
/// k*size, matching the encode tables. sia has no comparable slice-reconstruct
/// entry point, so it sits this table out.
fn run_reconstruct(k: usize, m: usize) {
    let sizes: &[usize] = &[100, 1_000, 10_000, 100_000, 1_000_000];

    // Worst case erases as much data as the code can survive.
    let single: Vec<usize> = vec![0];
    let mut worst: Vec<usize> = (0..m.min(k)).collect();
    worst.extend(k..k + m.saturating_sub(k));
    let patterns: &[(&str, &Vec<usize>)] = &[("1 data", &single), ("max m", &worst)];

    for (label, erasures) in patterns {
        println!(
            "\n--- reconstruct ({k},{m}), erased {} [{label}] — payload MiB/s ---",
            erasures.len()
        );
        println!(
            "{:<7} {:>10} {:>10} {:>10} {:>6}",
            "size", "rse(C)", "tape", "tapePrep", "iters"
        );

        let mut rng = StdRng::seed_from_u64(0xDECD);
        for &sz in sizes {
            let payload = (k * sz) as f64;
            let iters = (200_000_000usize / (k * sz)).clamp(20, 2000);
            let warmup = (iters / 10).clamp(3, 30);

            let mut full = base(k, m, sz, &mut rng);
            let tape = Tape::new(k, m).unwrap();
            let rse = Rse::new(k, m).unwrap();
            tape.encode(&mut full).unwrap();

            let mut present = vec![true; k + m];
            for &e in erasures.iter() {
                present[e] = false;
            }
            let prepared = tape.prepare_decode(&present).unwrap();

            // Correctness first: a zeroed-out erasure set must rebuild exactly.
            for reconstruct in [0u8, 1, 2] {
                let mut shards = full.clone();
                for &e in erasures.iter() {
                    shards[e].fill(0);
                }
                let mut view: Vec<(&mut [u8], bool)> =
                    shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
                for &e in erasures.iter() {
                    view[e].1 = false;
                }
                match reconstruct {
                    0 => rse.reconstruct(&mut view).unwrap(),
                    1 => tape.reconstruct(&mut view).unwrap(),
                    _ => prepared.reconstruct(&mut view).unwrap(),
                }
                assert_eq!(shards, full, "reconstruct mismatch ({k},{m}) sz={sz}");
            }

            // Timing: the erased slots keep valid lengths and just get flagged
            // absent each iteration, so the measured work is reconstruct only.
            let mut shards = full.clone();
            let time_one = |which: u8, shards: &mut Vec<Vec<u8>>| {
                secs(
                    || {
                        let mut view: Vec<(&mut [u8], bool)> = shards
                            .iter_mut()
                            .map(|s| (s.as_mut_slice(), true))
                            .collect();
                        for &e in erasures.iter() {
                            view[e].1 = false;
                        }
                        match which {
                            0 => rse.reconstruct(&mut view).unwrap(),
                            1 => tape.reconstruct(&mut view).unwrap(),
                            _ => prepared.reconstruct(&mut view).unwrap(),
                        }
                    },
                    warmup,
                    iters,
                )
            };
            let v_rse = mib(payload, iters, time_one(0, &mut shards));
            let v_tape = mib(payload, iters, time_one(1, &mut shards));
            let v_prep = mib(payload, iters, time_one(2, &mut shards));

            println!(
                "{:<7} {:>10.0} {:>10.0} {:>10.0} {:>6}",
                size_label(sz), v_rse, v_tape, v_prep, iters
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
    println!("baseline: reed-solomon-erasure galois_8 +simd-accel (the C SIMD backend the production system ships)");
    gate();
    prime(400);
    for &(k, m) in SHAPES {
        run_shape(k, m);
    }
    for &(k, m) in EXTRA_SHAPES {
        run_shape(k, m);
    }
    for &(k, m) in SHAPES {
        run_reconstruct(k, m);
    }
    println!(
        "\npayload = data bytes only (k*size); speedup = column / rse(C). x86 columns:\n\
         ssse3/avx2/avx512 = nibble-split single-output kernels; gfni1 = GFNI single-output;\n\
         gfniFused = GFNI multi-output (encode_fused). aarch64: tape = dispatched encode.\n\
         afft = additive-fft-reed-solomon (Cantor basis, different wire, no parity gate;\n\
         only power-of-two shapes exist for it; scalar LUT kernel off GFNI hosts)."
    );
}
