// RS backend gate + native perf, to run on an x86 GCP node.
//
// GATE: does firedancer's fd_reedsol produce byte-identical parity to
// reed-solomon-erasure at Clay's actual (k,m) params (not just turbine 32/32)?
// Encode-parity equality is a SUFFICIENT proof of full wire-compat: identical
// parity => identical generator matrix => reconstruct also agrees.
//
// PERF: firedancer encode vs reed-solomon-erasure encode throughput (native).
use rand::{rngs::StdRng, RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon as Rse;
use fd_reedsol::ReedSolomon as Fd; // crate package firedancer-reed-solomon, lib name fd_reedsol
use std::time::Instant;

fn shards(k: usize, m: usize, sz: usize, rng: &mut StdRng) -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Vec::with_capacity(k + m);
    for _ in 0..k {
        let mut s = vec![0u8; sz];
        rng.fill_bytes(&mut s);
        v.push(s);
    }
    for _ in 0..m {
        v.push(vec![0u8; sz]);
    }
    v
}

fn main() {
    let params = [(2, 2), (4, 4), (6, 4), (10, 7), (10, 10), (17, 17), (32, 32)];
    let sizes = [32usize, 1024, 32768];
    let mut rng = StdRng::seed_from_u64(0xC1A7);

    println!("=== GATE: firedancer parity == reed-solomon-erasure parity ===");
    let mut all_ok = true;
    for &(k, m) in &params {
        for &sz in &sizes {
            let mut a = shards(k, m, sz, &mut rng);
            let mut b = a.clone();
            Rse::new(k, m).unwrap().encode(&mut a).unwrap();
            Fd::new(k, m).unwrap().encode(&mut b).unwrap();
            let ok = (k..k + m).all(|i| a[i] == b[i]);
            all_ok &= ok;
            let tag = if (k, m) == (10, 10) { "  <-- Clay (20,10,13)" } else { "" };
            println!("  k={:>2} m={:>2} sz={:>5}: {}{}", k, m, sz, if ok { "MATCH" } else { "DIFFER ***" }, tag);
        }
    }
    println!("\nGATE RESULT: {}", if all_ok { "PASS -> firedancer is wire-compatible, proceed" } else { "FAIL -> firedancer DIFFERS, do NOT swap (compat required)" });

    println!("\n=== PERF (RS shape 10->10, production sizes, payload MiB/s) ===");
    println!("  [per-plane sub_chunk: 100KB stripe->100B, 1MB stripe->1KB]  [full-row: ->10KB / ->100KB]");
    for &sz in &[100usize,1000,10000,100000] {
        let (k,m)=(10usize,10usize);
        let payload=(k*sz) as f64; let base=shards(k,m,sz,&mut rng);
        let bench=|mut run: Box<dyn FnMut()>|{ for _ in 0..30 {run();} let it=1000; let t=Instant::now(); for _ in 0..it {run();} payload*it as f64/1048576.0/t.elapsed().as_secs_f64() };
        let rse=Rse::new(k,m).unwrap(); let mut s1=base.clone();
        let r1=bench(Box::new(move||{rse.encode(&mut s1).unwrap();}));
        let fd=Fd::new(k,m).unwrap(); let mut s2=base.clone();
        let r2=bench(Box::new(move||{fd.encode(&mut s2).unwrap();}));
        println!("  sz={:>7}: rse {:>5.0}  firedancer {:>6.0} MiB/s  ({:.1}x)", sz,r1,r2,r2/r1);
    }
}
