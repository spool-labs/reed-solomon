// SPDX-License-Identifier: Apache-2.0
//! wasm simd128 encode bench. `setup(k,m,sz)` builds the codec + shards once;
//! `run_fused/percoeff/scalar(iters)` do N encodes on that state (JS times them).
//! The returned checksum both prevents dead-code elimination and lets the runner
//! confirm the three paths agree byte-for-byte.

use std::cell::RefCell;
use tape_reed_solomon::ReedSolomon;
use wasm_bindgen::prelude::*;

thread_local! {
    static STATE: RefCell<Option<(ReedSolomon, Vec<Vec<u8>>, usize)>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
pub fn setup(k: usize, m: usize, sz: usize) {
    let rs = ReedSolomon::new(k, m).unwrap();
    let mut shards: Vec<Vec<u8>> = (0..k)
        .map(|s| (0..sz).map(|i| ((i * 31 + s * 17) & 0xff) as u8).collect())
        .collect();
    for _ in 0..m {
        shards.push(vec![0u8; sz]);
    }
    STATE.with(|st| *st.borrow_mut() = Some((rs, shards, k)));
}

fn run(iters: usize, fused: bool) -> u32 {
    STATE.with(|st| {
        let mut b = st.borrow_mut();
        let (rs, shards, k) = b.as_mut().unwrap();
        let k = *k;
        let mut sum = 0u32;
        for _ in 0..iters {
            if fused {
                rs.encode(shards.as_mut_slice()).unwrap();
            } else {
                rs.encode_scalar(shards.as_mut_slice()).unwrap();
            }
            // Fold a parity byte in so the loop can't be optimised away.
            sum = sum.wrapping_add(shards[k][sum as usize % shards[k].len()] as u32);
        }
        sum
    })
}

#[wasm_bindgen]
pub fn run_fused(iters: usize) -> u32 {
    run(iters, true)
}

#[wasm_bindgen]
pub fn run_scalar(iters: usize) -> u32 {
    run(iters, false)
}

/// Byte-exact differential: fused `encode` vs `encode_scalar` over specialised
/// (7,13)/(10,10) and fallback (4,2)/(6,4) shapes and tail/sub-block lengths.
/// Returns the number of mismatched parity bytes (0 = pass).
#[wasm_bindgen]
pub fn verify() -> u32 {
    let mut bad = 0u32;
    for &(k, m) in &[(7usize, 13usize), (10, 10), (4, 2), (6, 4)] {
        let rs = ReedSolomon::new(k, m).unwrap();
        for &len in &[8usize, 15, 16, 17, 31, 32, 100, 255, 256, 1000, 1430, 4096] {
            let base: Vec<Vec<u8>> = (0..k + m)
                .map(|s| {
                    if s < k {
                        (0..len).map(|i| ((i * 31 + s * 17 + 7) & 0xff) as u8).collect()
                    } else {
                        vec![0u8; len]
                    }
                })
                .collect();
            let mut a = base.clone();
            rs.encode(a.as_mut_slice()).unwrap();
            let mut b2 = base.clone();
            rs.encode_scalar(b2.as_mut_slice()).unwrap();
            for i in k..k + m {
                if a[i] != b2[i] {
                    bad += 1;
                }
            }
        }
    }
    bad
}
