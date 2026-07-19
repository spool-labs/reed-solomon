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

/// Reconstruct benchmark on the setup state: encodes once, then per iteration
/// flags the first `erased` shards absent (lengths intact) and rebuilds them
/// through the plan-cached fused path.
#[wasm_bindgen]
pub fn run_reconstruct(iters: usize, erased: usize) -> u32 {
    STATE.with(|st| {
        let mut b = st.borrow_mut();
        let (rs, shards, k) = b.as_mut().unwrap();
        let k = *k;
        rs.encode(shards.as_mut_slice()).unwrap();
        let mut sum = 0u32;
        for _ in 0..iters {
            let mut view: Vec<(&mut [u8], bool)> = shards
                .iter_mut()
                .map(|s| (s.as_mut_slice(), true))
                .collect();
            for slot in view.iter_mut().take(erased) {
                slot.1 = false;
            }
            rs.reconstruct(&mut view).unwrap();
            sum = sum.wrapping_add(shards[k][sum as usize % shards[k].len()] as u32);
        }
        sum
    })
}

/// Byte-exact reconstruct gate: erase up to m shards from an encoded stripe,
/// rebuild, and compare against the original. Returns mismatched bytes.
#[wasm_bindgen]
pub fn verify_reconstruct() -> u32 {
    let mut bad = 0u32;
    for &(k, m) in &[(7usize, 13usize), (10, 10), (14, 14), (16, 16), (6, 4)] {
        let rs = ReedSolomon::new(k, m).unwrap();
        for &len in &[16usize, 100, 1000] {
            let mut original: Vec<Vec<u8>> = (0..k + m)
                .map(|s| {
                    if s < k {
                        (0..len).map(|i| ((i * 31 + s * 17 + 7) & 0xff) as u8).collect()
                    } else {
                        vec![0u8; len]
                    }
                })
                .collect();
            rs.encode(original.as_mut_slice()).unwrap();

            for &erased in &[1usize, m / 2, m] {
                let mut shards = original.clone();
                for shard in shards.iter_mut().take(erased) {
                    shard.fill(0);
                }
                let mut view: Vec<(&mut [u8], bool)> = shards
                    .iter_mut()
                    .map(|s| (s.as_mut_slice(), true))
                    .collect();
                for slot in view.iter_mut().take(erased) {
                    slot.1 = false;
                }
                rs.reconstruct(&mut view).unwrap();
                for (a, b) in shards.iter().zip(original.iter()) {
                    for (x, y) in a.iter().zip(b.iter()) {
                        if x != y {
                            bad += 1;
                        }
                    }
                }
            }
        }
    }
    bad
}

/// Byte-exact differential: routed `encode` vs `encode_scalar` over the
/// generated (7,13)/(10,10), staged (16,16), and fused (4,2)/(6,4)/(14,14)
/// shapes and tail/sub-block lengths.
/// Returns the number of mismatched parity bytes (0 = pass).
#[wasm_bindgen]
pub fn verify() -> u32 {
    let mut bad = 0u32;
    for &(k, m) in &[
        (7usize, 13usize),
        (10, 10),
        (4, 2),
        (6, 4),
        (14, 14),
        (16, 16),
    ] {
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
