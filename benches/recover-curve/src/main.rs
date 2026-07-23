// SPDX-License-Identifier: Apache-2.0
//! How recover cost scales with the number of missing data shards.
//!
//! `reconstruct_with_plan` rebuilds only the rows that are actually missing, so
//! recover is cheap when a couple of shreds are lost and expensive only in the
//! pathological corner. An FFT/Leopard erasure decoder is the opposite: its cost
//! is flat in the loss count. That makes the whole Tier 2 decision (see
//! AGAVE-RS-PLAN.md section 4) a single question, which this bench answers:
//!
//!   at how many missing shards does a flat-cost decoder stop being a regression?
//!
//! The crossover is reported against a flat decoder costing 2x and 3x a measured
//! encode, which brackets what a Leopard-style decode runs (two size-n additive
//! transforms plus a pointwise pass). Compare the crossover against the real
//! loss-count histogram from Tier 0 before building anything.
//!
//! Also reports the cost of the two `ReconstructShard` shard forms. `(T, bool)`
//! keeps the buffer and flips a flag; `Option<T>` drops the buffer and makes
//! recover reallocate it. Measured, that costs within noise at 1 missing and a
//! few percent at full loss, so the choice is a readability call rather than a
//! performance one. The column is here to keep it that way: if an allocator or
//! shard-plumbing change ever makes the forms diverge, this catches it.

use std::time::Instant;
use tape_reed_solomon::ReedSolomon;

/// (32,32) at 987 B is Agave's turbine shred shape; (7,13) at 10 KB is tape's.
const CASES: &[(usize, usize, usize)] = &[(32, 32, 987), (7, 13, 10_000)];

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

/// One encoded FEC set with deterministic contents.
fn encoded(k: usize, m: usize, len: usize, rs: &ReedSolomon) -> Vec<Vec<u8>> {
    let mut shards: Vec<Vec<u8>> = (0..k + m)
        .map(|i| (0..len).map(|j| ((i * 31 + j * 17) & 0xff) as u8).collect())
        .collect();
    rs.encode(&mut shards).expect("encode");
    shards
}

/// Least-squares fit of `cost = fixed + marginal * missing`.
fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| a * b).sum();
    let sxx: f64 = xs.iter().map(|a| a * a).sum();
    let marginal = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    ((sy - marginal * sx) / n, marginal)
}

fn main() {
    for &(k, m, len) in CASES {
        let rs = ReedSolomon::new(k, m).expect("shape");
        let base = encoded(k, m, len, &rs);

        // Encode on the same host, so the crossover estimate below is not
        // anchored to numbers measured on some other machine.
        let enc_us = {
            let mut s = base.clone();
            secs(|| rs.encode(&mut s).expect("encode"), 200, 5_000) * 1e6
        };

        println!(
            "\n({k},{m}) {len} B shards, route {:?}, encode {enc_us:.3} us",
            rs.encode_route(len)
        );
        println!(" missing   us/recover   vs 1-missing");

        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        let mut first = 0.0f64;
        for missing in 1..=k {
            // (Vec<u8>, bool): the buffer stays put, the flag marks it absent,
            // so nothing allocates inside the timed region.
            let mut shards: Vec<(Vec<u8>, bool)> =
                base.iter().map(|b| (b.clone(), true)).collect();
            for s in shards.iter_mut().take(missing) {
                s.1 = false;
            }
            // Warm the decode-plan cache so we time the apply, not the inversion.
            rs.reconstruct_data(&mut shards).expect("reconstruct");

            let us = secs(
                || {
                    for s in shards.iter_mut().take(missing) {
                        s.1 = false;
                    }
                    rs.reconstruct_data(&mut shards).expect("reconstruct");
                },
                200,
                5_000,
            ) * 1e6;

            if missing == 1 {
                first = us;
            }
            xs.push(missing as f64);
            ys.push(us);
            println!("   {missing:>2}      {us:>7.3}       {:>5.2}x", us / first);
        }

        let (fixed, marginal) = fit(&xs, &ys);
        println!("\nfit: cost ~ {fixed:.3} us fixed + {marginal:.3} us per missing shard");
        println!(
            "     fixed share: {:.0}% at 1 missing, {:.0}% at {} missing",
            fixed / ys[0] * 100.0,
            fixed / ys[(k / 8).max(1).min(k - 1)] * 100.0,
            (k / 8).max(1) + 1
        );

        // A flat-cost decoder only wins right of the crossover. Left of it, it is
        // a straight regression on every recover that actually happens.
        for mult in [2.0f64, 3.0] {
            let flat = mult * enc_us;
            let cross = (flat - fixed) / marginal;
            let verdict = if cross >= k as f64 {
                "never pays at this shape".to_string()
            } else {
                format!("crosses at {cross:.1} of {k} missing")
            };
            println!("flat decode at {mult:.0}x encode = {flat:>6.2} us -> {verdict}");
        }

        // Shard-form overhead, measured at the loss counts turbine actually sees.
        println!("\nshard form, us/recover:");
        println!(" missing   (Vec,bool)   Option<Vec>   penalty");
        for &missing in &[1usize, 4, k] {
            if missing > k {
                continue;
            }
            let mut tup: Vec<(Vec<u8>, bool)> = base.iter().map(|b| (b.clone(), true)).collect();
            for s in tup.iter_mut().take(missing) {
                s.1 = false;
            }
            rs.reconstruct_data(&mut tup).expect("reconstruct");
            let a = secs(
                || {
                    for s in tup.iter_mut().take(missing) {
                        s.1 = false;
                    }
                    rs.reconstruct_data(&mut tup).expect("reconstruct");
                },
                200,
                5_000,
            ) * 1e6;

            let mut opt: Vec<Option<Vec<u8>>> = base.iter().cloned().map(Some).collect();
            for s in opt.iter_mut().take(missing) {
                *s = None;
            }
            rs.reconstruct_data(&mut opt).expect("reconstruct");
            let b = secs(
                || {
                    for s in opt.iter_mut().take(missing) {
                        *s = None;
                    }
                    rs.reconstruct_data(&mut opt).expect("reconstruct");
                },
                200,
                5_000,
            ) * 1e6;

            println!(
                "   {missing:>2}      {a:>8.3}     {b:>9.3}     {:>5.2}x",
                b / a
            );
        }
    }
}
