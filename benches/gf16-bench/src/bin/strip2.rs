// Follow-up to `strip`: find the best strip width, then see what threading buys.
// Strips are independent column ranges, so a thread can own a byte range of every
// shard with no sharing at all. rs-simd is single-threaded, so the threaded column
// is a fair picture of wall-clock at the snapshot operating point, not a fair
// picture of per-core efficiency; both are printed.
//
// Run: cargo run --release --bin strip2 -- <shape-index> [threads]
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::ReedSolomonEncoder;
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const SHAPES: &[(usize, usize)] = &[(11, 20), (43, 85), (86, 170), (171, 341)];
const STRIPS: &[usize] = &[2 << 10, 4 << 10, 8 << 10, 16 << 10, 32 << 10];

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

// Encode one contiguous byte range of every shard, walking it in `strip` steps.
fn encode_range(
    rs: &ReedSolomon16,
    data: &[&[u8]],
    parity: &mut [&mut [u8]],
    strip: usize,
) {
    let len = data[0].len();
    let mut off = 0;
    while off < len {
        let w = strip.min(len - off);
        let din: Vec<&[u8]> = data.iter().map(|d| &d[off..off + w]).collect();
        let mut dout: Vec<&mut [u8]> = parity
            .iter_mut()
            .map(|p| {
                // Reborrow the strip out of the thread's own range.
                let s: &mut [u8] = p;
                &mut s[off..off + w]
            })
            .collect();
        rs.encode_sep(&din, &mut dout).expect("strip encode");
        off += w;
    }
}

fn main() {
    let idx: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    let threads: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(8);
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
    let dref: Vec<&[u8]> = data.iter().map(|d| d.as_slice()).collect();
    let mut parity: Vec<Vec<u8>> = vec![vec![0u8; sz]; m];
    let rs = ReedSolomon16::new(k, m).expect("codec");
    let payload = (k * sz) as f64;
    let iters = 12usize;
    let mib = |s: f64| payload * iters as f64 / (1024.0 * 1024.0) / s;

    let want = if idx == 0 {
        rs.encode_sep(&data, &mut parity).expect("encode");
        Some(parity.clone())
    } else {
        None
    };

    // Single-threaded strip-width sweep.
    let mut cells = Vec::new();
    for &strip in STRIPS {
        let s = secs(
            || {
                let mut pref: Vec<&mut [u8]> =
                    parity.iter_mut().map(|p| p.as_mut_slice()).collect();
                encode_range(&rs, &dref, &mut pref, strip);
            },
            1,
            iters,
        );
        if let Some(w) = &want {
            assert_eq!(&parity, w, "strip {strip} diverged");
        }
        cells.push(mib(s));
    }

    // Threaded: each thread takes a disjoint byte range of every shard.
    let best_strip = STRIPS[cells
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0];
    let span = sz.div_ceil(threads).next_multiple_of(64);
    let thr_s = secs(
        || {
            // Split every parity shard into per-thread ranges, then transpose so
            // each thread holds one range from each shard.
            let mut cols: Vec<Vec<&mut [u8]>> = (0..threads).map(|_| Vec::new()).collect();
            for p in parity.iter_mut() {
                for (t, chunk) in p.chunks_mut(span).enumerate() {
                    cols[t].push(chunk);
                }
            }
            std::thread::scope(|sc| {
                for (t, mut col) in cols.into_iter().enumerate() {
                    let rs = &rs;
                    let data = &data;
                    sc.spawn(move || {
                        if col.is_empty() {
                            return;
                        }
                        let off = t * span;
                        let w = col[0].len();
                        let din: Vec<&[u8]> =
                            data.iter().map(|d| &d[off..off + w]).collect();
                        encode_range(rs, &din, &mut col, best_strip);
                    });
                }
            });
        },
        1,
        iters,
    );
    if let Some(w) = &want {
        assert_eq!(&parity, w, "threaded encode diverged");
    }
    let thr = mib(thr_s);
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
    let simd = mib(simd_s);
    let best1 = cells.iter().cloned().fold(0.0f64, f64::max);

    if idx == 0 {
        println!(
            "{:<11} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8} {:>7} {:>7}",
            "shape", "2K", "4K", "8K", "16K", "32K", "thr", "rs-simd", "1t/sd", "thr/sd"
        );
    }
    print!("({k},{m})");
    for c in &cells {
        print!("{:>8.0}", c);
    }
    println!(
        "{:>9.0} {:>8.0} {:>6.2}x {:>6.2}x",
        thr,
        simd,
        best1 / simd,
        thr / simd
    );
}
