// Which encode route each snapshot shape takes, and whether the host can
// actually execute it.
//
// `profitable()` picks fft vs matrix from the shape alone (on x86 the test is
// 9*mult_count < 2*k*m), but the x86 FFT executor needs GFNI. A host without it
// (Zen 3 / Milan) still routes to fft and then falls back to the portable
// arena, while the matrix path there is fused-AVX2 accelerated. So on Milan the
// fft column is the risk column: it marks shapes routed to a SIMD executor the
// host does not have.
//
// Run natively for this host's answer; run the x86_64 build under Rosetta to
// read the no-GFNI x86 routing, which is what Milan does.
use tape_reed_solomon::ReedSolomon16;

fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "host x86_64: gfni={} avx2={} avx512f={}",
            std::is_x86_feature_detected!("gfni"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f"),
        );
        println!("(gfni=false means every fft-routed shape runs the portable arena)\n");
    }
    #[cfg(target_arch = "aarch64")]
    println!("host aarch64: profitable() always routes fft, tower_neon executes it\n");

    println!("{:>7} {:>10} {:>8}", "groups", "shape", "route");
    let mut first_fft = None;
    for groups in [3usize, 6, 12, 20, 30, 31, 40, 50, 60, 75, 100, 128, 150, 200, 256, 300] {
        let k = groups.div_ceil(3);
        let m = groups - k;
        if m == 0 {
            continue;
        }
        let rs = ReedSolomon16::new(k, m).expect("codec");
        let route = rs.encode_route();
        if route == "fft" && first_fft.is_none() {
            first_fft = Some(groups);
        }
        println!("{groups:>7} {:>10} {route:>8}", format!("({k},{m})"));
    }
    match first_fft {
        Some(g) => println!("\nfirst fft-routed fleet size: {g} groups"),
        None => println!("\nno snapshot shape routes to fft on this target"),
    }
}
