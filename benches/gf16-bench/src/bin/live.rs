// The live tape-internal operating point, end to end against rs-simd.
//
// Shape comes from the chain: k = snapshot_outer_k(total_groups) =
// ceil(total_groups/3), n = total_groups.
//
// Shard size is NOT fixed at MAX_CHUNK_BYTES. `OuterCoder::encode` derives
// chunk_bytes = ceil(segment_len / k) rounded up to 64, and 4 MiB is only the
// ceiling that returns TooMuchData. So the real shard is
// compressed_snapshot_segment / k, and benchmarking at 4 MiB measures the worst
// corner rather than the operating point. Pass the real one when known:
//   cargo run --release --bin live -- <total_groups> <threads> <shard_bytes>
//
// Encode is measured four ways (whole shard, 16 KiB column strips, strips on
// T threads) and decode across the loss counts that actually occur. Strip and
// thread results are asserted bit-identical to the whole-shard encode, since
// RS treats every field-element column independently.
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};
use std::time::Instant;
use tape_reed_solomon::ReedSolomon16;

const CHUNK: usize = 4 << 20; // tape-internal MAX_CHUNK_BYTES
const STRIP: usize = 16 << 10; // best width on M4, from `strip2`

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

fn encode_range(rs: &ReedSolomon16, data: &[&[u8]], parity: &mut [&mut [u8]], strip: usize) {
    let len = data[0].len();
    let mut off = 0;
    while off < len {
        let w = strip.min(len - off);
        let din: Vec<&[u8]> = data.iter().map(|d| &d[off..off + w]).collect();
        let mut dout: Vec<&mut [u8]> = parity
            .iter_mut()
            .map(|p| {
                let s: &mut [u8] = p;
                &mut s[off..off + w]
            })
            .collect();
        rs.encode_sep(&din, &mut dout).expect("strip encode");
        off += w;
    }
}

fn main() {
    let groups: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20);
    let threads: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(8);
    let k = groups.div_ceil(3);
    let m = groups - k;
    // Shard bytes: given, else the MAX_CHUNK_BYTES ceiling. Rounded to 64 the
    // way OuterCoder::encode rounds it.
    let sz = std::env::args()
        .nth(3)
        .and_then(|a| a.parse::<usize>().ok())
        .unwrap_or(CHUNK)
        .max(64)
        .div_ceil(64)
        * 64;
    assert!(sz <= CHUNK, "shard {sz} exceeds MAX_CHUNK_BYTES; OuterCoder would reject it");

    println!("tape-internal live shape: {groups} groups -> k={k} m={m}");
    println!(
        "shard {sz} B ({:.2} MiB), so segment = k*shard = {:.2} MiB of compressed snapshot",
        sz as f64 / (1024.0 * 1024.0),
        (k * sz) as f64 / (1024.0 * 1024.0),
    );
    println!("(k = ceil(groups/3) per snapshot_outer_k; shard = ceil(segment/k), cap 4 MiB)\n");

    let mut rng = StdRng::seed_from_u64(0x11_20);
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
    // Scale with payload: a fixed count is far too few at small shards, where a
    // single pass is under a millisecond and the timer is mostly noise.
    let iters = ((512usize << 20) / (k * sz)).clamp(12, 4000);
    // Each shard size is run in its own process, so the first timed pass would
    // otherwise pay first-touch page faults and a cold plan cache; that showed
    // up as a spurious 0.83x at 128 KiB shards.
    let warmup = (iters / 8).max(3);
    let mib = |s: f64| payload * iters as f64 / (1024.0 * 1024.0) / s;

    rs.encode_sep(&data, &mut parity).expect("encode");
    let want = parity.clone();

    let whole = mib(secs(|| { rs.encode_sep(&data, &mut parity).expect("e"); }, warmup, iters));

    let strip = mib(secs(
        || {
            let mut pref: Vec<&mut [u8]> = parity.iter_mut().map(|p| p.as_mut_slice()).collect();
            encode_range(&rs, &dref, &mut pref, STRIP);
        },
        warmup,
        iters,
    ));
    assert_eq!(parity, want, "strip encode diverged");

    let span = sz.div_ceil(threads).next_multiple_of(64);
    let thr = mib(secs(
        || {
            let mut cols: Vec<Vec<&mut [u8]>> = (0..threads).map(|_| Vec::new()).collect();
            for p in parity.iter_mut() {
                for (t, chunk) in p.chunks_mut(span).enumerate() {
                    cols[t].push(chunk);
                }
            }
            std::thread::scope(|sc| {
                for (t, mut col) in cols.into_iter().enumerate() {
                    let (rs, data) = (&rs, &data);
                    sc.spawn(move || {
                        if col.is_empty() {
                            return;
                        }
                        let off = t * span;
                        let w = col[0].len();
                        let din: Vec<&[u8]> = data.iter().map(|d| &d[off..off + w]).collect();
                        encode_range(rs, &din, &mut col, STRIP);
                    });
                }
            });
        },
        warmup,
        iters,
    ));
    assert_eq!(parity, want, "threaded encode diverged");

    let mut enc = ReedSolomonEncoder::new(k, m, sz).expect("rs-simd enc");
    let simd_e = mib(secs(
        || {
            for shard in &data {
                enc.add_original_shard(shard).expect("add");
            }
            let r = enc.encode().expect("encode");
            std::hint::black_box(r.recovery(0).expect("rec").len());
        },
        warmup,
        iters,
    ));

    println!("ENCODE (data-payload MiB/s)");
    println!("  rs-simd                 {simd_e:>8.0}");
    println!("  tape16 whole shard      {whole:>8.0}   {:.2}x", whole / simd_e);
    println!("  tape16 16K strips       {strip:>8.0}   {:.2}x", strip / simd_e);
    println!("  tape16 strips x{threads:<2}       {thr:>8.0}   {:.2}x", thr / simd_e);

    // Full stripe for decode: data followed by parity.
    let mut full: Vec<Vec<u8>> = data.clone();
    full.extend(parity.iter().cloned());

    println!("\nDECODE (data-only recover, MiB/s)");
    println!("  {:>8} {:>12} {:>12} {:>9}", "missing", "tape16", "rs-simd", "ratio");
    let d_iters = ((256usize << 20) / (k * sz)).clamp(8, 400);
    let d_warmup = (d_iters / 8).max(2);
    for miss in 1..=k.min(m) {
        // Erase the first `miss` data shards; pattern fixed across iterations so
        // both sides run warm, steady-state, not paying inversion setup.
        // Stripe built once per erasure pattern: cloning all n shards inside the
        // timed region measures a memcpy, not the decode, and understated
        // recover by roughly 2x. Only the erased rows are dropped per iteration.
        let mut opt: Vec<Option<Vec<u8>>> = full.iter().cloned().map(Some).collect();
        let t_s = secs(
            || {
                for o in opt.iter_mut().take(miss) {
                    *o = None;
                }
                rs.reconstruct_data(&mut opt).expect("recover");
                std::hint::black_box(opt[0].as_ref().unwrap()[0]);
            },
            d_warmup,
            d_iters,
        );
        let mut dec = ReedSolomonDecoder::new(k, m, sz).expect("rs-simd dec");
        let s_s = secs(
            || {
                for i in miss..k {
                    dec.add_original_shard(i, &full[i]).expect("orig");
                }
                for j in 0..miss {
                    dec.add_recovery_shard(j, &full[k + j]).expect("rec");
                }
                let r = dec.decode().expect("decode");
                std::hint::black_box(r.restored_original(0).expect("restored")[0]);
            },
            d_warmup,
            d_iters,
        );
        let dm = |s: f64| payload * d_iters as f64 / (1024.0 * 1024.0) / s;
        let (a, b) = (dm(t_s), dm(s_s));
        println!("  {miss:>8} {a:>10.0} MiB/s {b:>10.0} MiB/s {:>8.2}x", a / b);
    }
}
