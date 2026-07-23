// SPDX-License-Identifier: Apache-2.0
//! Head-to-head GF(2^16) RS **encode** throughput: ReedSolomon16 vs
//! reed-solomon-simd (Leopard-style, what the production outer path ships
//! today). Shapes are the outer operating points: data = total/3 per the
//! snapshot denominator, at the realistic shard sizes.
//!
//! Throughput is data payload (k * shard bytes) over wall time, the same
//! definition as the other harnesses, so numbers line up across runs.
//!
//! Run: cargo run --release

use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::ReedSolomonEncoder;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

// (data, parity) at shard bytes. The outer coder always uses 4 MiB chunks
// (tape-internal MAX_CHUNK_BYTES) with k = ceil(n/3) of the active group
// count: (11,20) is today's 31-group network, the rest are the scale points.
// One sub-chunk row watches the cache-resident end.
const RUNS: &[(usize, usize, usize)] = &[
    (11, 20, 4 << 20),
    (43, 85, 4 << 20),
    (86, 170, 4 << 20),
    (171, 341, 4 << 20),
    (86, 170, 64 << 10),
];

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

fn mib(payload: f64, iters: usize, s: f64) -> f64 {
    payload * iters as f64 / (1024.0 * 1024.0) / s
}

fn size_label(sz: usize) -> String {
    if sz % (1 << 20) == 0 {
        format!("{}M", sz >> 20)
    } else {
        format!("{}K", sz >> 10)
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0x16_16);
    println!("gf16 encode: data-payload MiB/s, higher is better\n");
    println!(
        "{:<16} {:>6} {:>12} {:>12} {:>8}",
        "shape @ shard", "route", "tape16", "rs-simd", "ratio"
    );

    for &(k, m, sz) in RUNS {
        let data: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut s = vec![0u8; sz];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; sz]; m];

        // Keep total work per engine near a fixed budget so big and small
        // shard runs both finish quickly with stable numbers.
        let iters = (512 << 20) / (k * sz).max(1);
        let iters = iters.clamp(3, 200);
        let payload = (k * sz) as f64;

        let rs = ReedSolomon16::new(k, m).expect("codec should build");
        let route = rs.encode_route();
        let tape_s = secs(
            || rs.encode_sep(&data, &mut parity).expect("encode should succeed"),
            1,
            iters,
        );

        let mut encoder =
            ReedSolomonEncoder::new(k, m, sz).expect("rs-simd encoder should build");
        let simd_s = secs(
            || {
                for shard in &data {
                    encoder.add_original_shard(shard).expect("add shard");
                }
                let result = encoder.encode().expect("rs-simd encode");
                std::hint::black_box(result.recovery(0).expect("recovery shard").len());
            },
            1,
            iters,
        );

        let tape = mib(payload, iters, tape_s);
        let simd = mib(payload, iters, simd_s);
        println!(
            "({k},{m}) @ {:<5} {:>6} {:>9.0} MiB/s {:>9.0} MiB/s {:>7.2}x",
            size_label(sz),
            route,
            tape,
            simd,
            tape / simd,
        );
    }
}
