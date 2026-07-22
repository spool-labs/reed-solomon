// SPDX-License-Identifier: Apache-2.0
//! How GF(2^16) recover cost scales with the number of missing data shards,
//! against reed-solomon-simd's flat-cost FFT decoder.
//!
//! ReedSolomon16 reconstructs only the rows that are missing, so its cost is
//! linear in the loss count; a Leopard-style decoder pays two size-n
//! transforms regardless. This prints both curves at the outer shapes so the
//! decode-tier decision is one look: if realistic loss counts sit left of the
//! crossover, the matrix plans stay; if they sit right, an FFT decode tier is
//! worth building.
//!
//! Both sides are measured data-only (rs-simd restores only originals) with
//! the erasure pattern fixed across iterations, so our plan cache and their
//! internal caches are both warm: steady-state cost, not inversion cost.

use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::ReedSolomonDecoder;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const RUNS: &[(usize, usize, usize)] = &[(86, 170, 1 << 20), (171, 341, 256 << 10)];

fn secs<F: FnMut()>(mut f: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0xDEC0DE);
    for &(k, m, sz) in RUNS {
        let rs = ReedSolomon16::new(k, m).expect("codec should build");
        let mut shards: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut s = vec![0u8; sz];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();
        shards.extend(std::iter::repeat_with(|| vec![0u8; sz]).take(m));
        rs.encode(&mut shards).expect("encode");

        let payload = (k * sz) as f64 / (1024.0 * 1024.0);
        let iters = ((256 << 20) / (k * sz).max(1)).clamp(3, 50);
        println!("({k},{m}) @ {} KiB shards, data-only recover MiB/s:", sz >> 10);
        println!("{:>8} {:>12} {:>12}", "missing", "tape16", "rs-simd");

        for &missing in &[1usize, 2, 4, 8, 16, 32, 64, k.min(m)] {
            if missing > k || missing > m {
                continue;
            }

            // Ours: erase the first `missing` data shards, flag-based view.
            let tape_s = secs(
                || {
                    let mut work = shards.clone();
                    let mut view: Vec<(&mut [u8], bool)> =
                        work.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
                    for e in 0..missing {
                        view[e].1 = false;
                    }
                    rs.reconstruct_data(&mut view).expect("reconstruct");
                    std::hint::black_box(work[0][0]);
                },
                1,
                iters,
            );
            // Remove the clone cost measured on its own.
            let clone_s = secs(
                || {
                    std::hint::black_box(shards.clone());
                },
                1,
                iters,
            );

            let mut decoder =
                ReedSolomonDecoder::new(k, m, sz).expect("rs-simd decoder should build");
            let simd_s = secs(
                || {
                    for i in missing..k {
                        decoder.add_original_shard(i, &shards[i]).expect("add original");
                    }
                    for r in 0..missing {
                        decoder.add_recovery_shard(r, &shards[k + r]).expect("add recovery");
                    }
                    let result = decoder.decode().expect("decode");
                    std::hint::black_box(result.restored_original(0).expect("restored").len());
                },
                1,
                iters,
            );

            println!(
                "{:>8} {:>8.0} MiB/s {:>8.0} MiB/s",
                missing,
                payload / (tape_s - clone_s).max(1e-9),
                payload / simd_s,
            );
        }
        println!();
    }
}
