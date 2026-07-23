// What can MAX_CHUNK_BYTES actually be, and what does raising it cost?
//
// tape-internal pins it at 4 MiB, documented as a reed-solomon-simd shard-size
// constraint. Two questions: does rs-simd really refuse larger shards, and does
// ReedSolomon16 throughput hold up as the chunk grows now that encode is
// strip-mined? The working-set column is the number that actually decides it:
// an outer encode holds n * chunk resident, per node, per in-flight segment.
//
// Run: cargo run --release --bin chunksize -- [total_groups]
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::ReedSolomonEncoder;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const SIZES: &[usize] = &[
    256 << 10,
    1 << 20,
    4 << 20, // today
    16 << 20,
    64 << 20,
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

fn label(sz: usize) -> String {
    if sz >= 1 << 20 {
        format!("{}M", sz >> 20)
    } else {
        format!("{}K", sz >> 10)
    }
}

fn main() {
    let groups: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(30);
    let k = groups.div_ceil(3);
    let m = groups - k;
    let mut rng = StdRng::seed_from_u64(0xC401);
    let rs = ReedSolomon16::new(k, m).expect("codec");

    println!("chunk-size sweep at {groups} groups -> k={k} m={m}\n");
    println!(
        "{:>7} {:>10} {:>12} {:>12} {:>9} {:>10} {:>12}",
        "chunk", "workset", "tape16 enc", "rs-simd enc", "ratio", "segment", "decode(1)"
    );

    for &sz in SIZES {
        let data: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut s = vec![0u8; sz];
                rng.fill_bytes(&mut s);
                s
            })
            .collect();
        let mut parity: Vec<Vec<u8>> = vec![vec![0u8; sz]; m];
        let payload = (k * sz) as f64;
        // Keep total moved bytes roughly fixed so big chunks stay quick.
        let iters = ((1usize << 30) / (k * sz)).clamp(3, 40);
        let mib = |s: f64, it: usize| payload * it as f64 / (1024.0 * 1024.0) / s;

        let enc = mib(
            secs(|| { rs.encode_sep(&data, &mut parity).expect("e"); }, 1, iters),
            iters,
        );

        // Does rs-simd accept this shard size at all?
        let simd = match ReedSolomonEncoder::new(k, m, sz) {
            Ok(mut e) => {
                let s = secs(
                    || {
                        for shard in &data {
                            e.add_original_shard(shard).expect("add");
                        }
                        let r = e.encode().expect("encode");
                        std::hint::black_box(r.recovery(0).expect("rec").len());
                    },
                    1,
                    iters,
                );
                Some(mib(s, iters))
            }
            Err(_) => None,
        };

        // Single-shard-loss recovery, the common repair case. The stripe is
        // built once and only the erased row is dropped per iteration: cloning
        // all n shards inside the timed region would measure a 1.9 GiB memcpy
        // at the large chunks, not the decode.
        let mut full: Vec<Vec<u8>> = data.clone();
        full.extend(parity.iter().cloned());
        let mut opt: Vec<Option<Vec<u8>>> = full.into_iter().map(Some).collect();
        let d_iters = iters.clamp(2, 8);
        let dec = mib(
            secs(
                || {
                    opt[0] = None;
                    rs.reconstruct_data(&mut opt).expect("recover");
                    std::hint::black_box(opt[0].as_ref().unwrap()[0]);
                },
                1,
                d_iters,
            ),
            d_iters,
        );

        let workset = (groups * sz) as f64 / (1024.0 * 1024.0);
        let segment = (k * sz) as f64 / (1024.0 * 1024.0);
        match simd {
            Some(s) => println!(
                "{:>7} {:>8.0} MiB {:>7.0} MiB/s {:>7.0} MiB/s {:>8.2}x {:>7.0} MiB {:>7.0} MiB/s",
                label(sz), workset, enc, s, enc / s, segment, dec
            ),
            None => println!(
                "{:>7} {:>8.0} MiB {:>7.0} MiB/s {:>12} {:>9} {:>7.0} MiB {:>7.0} MiB/s",
                label(sz), workset, enc, "REFUSED", "-", segment, dec
            ),
        }
    }
}
