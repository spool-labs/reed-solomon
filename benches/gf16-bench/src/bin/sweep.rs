// SPDX-License-Identifier: Apache-2.0
//! FFT/matrix encode crossover sweep: measures both paths per shape so the
//! `profitable()` routing threshold can be checked against reality instead of
//! a multiply-count model.

use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const SHAPES: &[(usize, usize)] = &[
    (7, 13),
    (10, 10),
    (11, 20),
    (16, 16),
    (17, 33),
    (43, 85),
    (32, 32),
    (32, 96),
    (48, 144),
    (64, 64),
    (86, 84),
    (86, 170),
    (100, 50),
    (128, 128),
];
const DEFAULT_SIZE: usize = 64 << 10;

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

fn main() {
    let size: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(DEFAULT_SIZE);
    let mut rng = StdRng::seed_from_u64(0x5EEB);
    println!(
        "{:<10} {:>7} {:>12} {:>12} {:>10}",
        "shape", "route", "fft-or-route", "matrix", "verdict"
    );
    for &(k, m) in SHAPES {
        let rs = ReedSolomon16::new(k, m).expect("codec should build");
        let data: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut s = vec![0u8; size];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; size]; m];
        let iters = ((256 << 20) / (k * size)).clamp(3, 400);
        let payload = (k * size) as f64;

        let routed = secs(
            || rs.encode_sep(&data, &mut parity).expect("encode"),
            2,
            iters,
        );
        let matrix = secs(
            || rs.encode_sep_matrix(&data, &mut parity).expect("encode"),
            2,
            iters,
        );

        let mib = |s: f64| payload * iters as f64 / (1024.0 * 1024.0) / s;
        let route = rs.encode_route();
        let verdict = if route == "fft" && matrix < routed * 0.98 {
            "MISROUTED"
        } else if route == "matrix" && routed < matrix {
            "(matrix==routed)"
        } else {
            "ok"
        };
        println!(
            "({k},{m})     {route:>7} {:>8.0} MiB/s {:>8.0} MiB/s {verdict:>12}",
            mib(routed),
            mib(matrix),
        );
    }
}
