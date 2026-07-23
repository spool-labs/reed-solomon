// SPDX-License-Identifier: Apache-2.0
//! Profiling target: pin one shape and spin encode for the sampler.
//! Usage: profile [k m shard_bytes] (defaults to 86 170 1MiB)

use rand::{rngs::StdRng, RngCore, SeedableRng};
use tape_reed_solomon::ReedSolomon16;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (k, m, sz) = match args.as_slice() {
        [k, m, sz] => (*k, *m, *sz),
        _ => (86, 170, 1 << 20),
    };
    let mut rng = StdRng::seed_from_u64(0x16);
    let data: Vec<Vec<u8>> = (0..k)
        .map(|_| {
            let mut s = vec![0u8; sz];
            rng.fill_bytes(&mut s);
            s
        })
        .collect();
    let mut parity: Vec<Vec<u8>> = vec![vec![0u8; sz]; m];
    let rs = ReedSolomon16::new(k, m).expect("codec should build");
    assert_eq!(rs.encode_route(), "fft");

    let start = std::time::Instant::now();
    let mut iters = 0usize;
    while start.elapsed().as_secs_f64() < 5.0 {
        rs.encode_sep(&data, &mut parity).expect("encode");
        iters += 1;
    }
    let s = start.elapsed().as_secs_f64();
    let mib = (k * sz * iters) as f64 / (1024.0 * 1024.0) / s;
    println!("({k},{m}) @ {sz}: {mib:.0} MiB/s over {iters} iters");
}
