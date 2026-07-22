// Strip-mining probe: is the 4 MiB encode DRAM-bound rather than multiply-bound?
// RS encodes each field-element column independently, so encoding a 4 MiB shard
// as contiguous strips and concatenating must be bit-identical to encoding it
// whole. If strips win, the loss at 4 MiB is cache residency, not arithmetic.
//
// One shape per process: at (171,341) the shard set alone is 2 GiB, and holding
// a reference copy alongside rs-simd's padded internal buffers will not fit.
// Run: cargo run --release --bin strip -- <shape-index>
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::ReedSolomonEncoder;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const SHAPES: &[(usize, usize)] = &[(11, 20), (43, 85), (86, 170), (171, 341)];
const STRIPS: &[usize] = &[16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20];

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
    let idx: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    let (k, m) = SHAPES[idx];
    let sz = 4usize << 20;
    let mut rng = StdRng::seed_from_u64(0x5721);

    let data: Vec<Vec<u8>> = (0..k)
        .map(|_| {
            let mut s = vec![0u8; sz];
            rng.fill_bytes(&mut s);
            s
        })
        .collect();
    let mut parity: Vec<Vec<u8>> = vec![vec![0u8; sz]; m];
    let rs = ReedSolomon16::new(k, m).expect("codec");
    let payload = (k * sz) as f64;
    let iters = 12usize;

    // Bit-identity of strips vs whole is checked only on the small shape, where
    // a reference copy is affordable; the property is shape-independent.
    let want = if idx == 0 {
        rs.encode_sep(&data, &mut parity).expect("encode");
        Some(parity.clone())
    } else {
        None
    };

    let mut cells = Vec::new();
    for &strip in STRIPS {
        let s = secs(
            || {
                let mut off = 0;
                while off < sz {
                    let w = strip.min(sz - off);
                    // Borrow one column strip out of every shard.
                    let din: Vec<&[u8]> = data.iter().map(|d| &d[off..off + w]).collect();
                    let mut dout: Vec<&mut [u8]> =
                        parity.iter_mut().map(|p| &mut p[off..off + w]).collect();
                    rs.encode_sep(&din, &mut dout).expect("strip encode");
                    off += w;
                }
            },
            1,
            iters,
        );
        if let Some(w) = &want {
            assert_eq!(&parity, w, "strip {strip} diverged from whole encode");
        }
        cells.push(payload * iters as f64 / (1024.0 * 1024.0) / s);
    }
    drop(want);

    let mut encoder = ReedSolomonEncoder::new(k, m, sz).expect("rs-simd");
    let simd_s = secs(
        || {
            for shard in &data {
                encoder.add_original_shard(shard).expect("add");
            }
            let r = encoder.encode().expect("encode");
            std::hint::black_box(r.recovery(0).expect("rec").len());
        },
        1,
        iters,
    );
    let simd = payload * iters as f64 / (1024.0 * 1024.0) / simd_s;
    let best = cells.iter().cloned().fold(0.0f64, f64::max);

    if idx == 0 {
        println!(
            "{:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>9}",
            "shape", "16K", "64K", "256K", "1M", "whole", "rs-simd", "best/simd"
        );
    }
    print!("({k},{m})");
    for c in &cells {
        print!("{:>10.0}", c);
    }
    println!("{:>10.0} {:>8.2}x", simd, best / simd);
}
