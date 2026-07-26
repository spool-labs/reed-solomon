use std::time::Instant;
use tape_reed_solomon::ReedSolomon;

/// Shards carved from one buffer, each offset by `skew` bytes past the previous
/// shard end, so equal-sized buffers stop sharing a cache set index.
fn timed(k: usize, m: usize, len: usize, skew: usize) -> f64 {
    let n = k + m;
    let stride = len + skew;
    let mut backing = vec![0u8; stride * n + 64];
    let base = (backing.as_ptr() as usize).next_multiple_of(64) - backing.as_ptr() as usize;

    let mut shards: Vec<&mut [u8]> = Vec::with_capacity(n);
    let mut rest = &mut backing[base..];
    for _ in 0..n {
        let (head, tail) = rest.split_at_mut(stride);
        shards.push(&mut head[..len]);
        rest = tail;
    }
    for (i, s) in shards.iter_mut().enumerate().take(k) {
        s.fill((i % 251) as u8);
    }

    let rs = ReedSolomon::new(k, m).unwrap();
    rs.encode(&mut shards).unwrap();
    let reps = (64 * 1024 * 1024 / (n * len)).clamp(3, 200);
    let start = Instant::now();
    for _ in 0..reps {
        rs.encode(&mut shards).unwrap();
    }
    let secs = start.elapsed().as_secs_f64() / reps as f64;
    (n * len) as f64 / secs / 1e9
}

fn main() {
    println!("{:>8} {:>10} {:>10} {:>10} {:>10} {:>8}", "shape", "len", "skew 0", "skew 64", "skew 320", "gain");
    for &(k, m) in &[(7usize, 13usize), (10, 10), (16, 16), (17, 33), (32, 32)] {
        for len in [16 * 1024usize, 1024 * 1024, 4 * 1024 * 1024] {
            let a = timed(k, m, len, 0);
            let b = timed(k, m, len, 64);
            let c = timed(k, m, len, 320);
            println!(
                "{:>8} {len:>10} {a:>10.2} {b:>10.2} {c:>10.2} {:>7.2}x",
                format!("({k},{m})"),
                c.max(b) / a
            );
        }
    }
}
