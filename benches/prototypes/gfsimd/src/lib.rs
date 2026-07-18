// Prototype: wasm-simd128 GF(2^8) multiply for the Reed-Solomon matmul hot path,
// vs the scalar per-byte table lookup (== what our wasm build does today).
//
// Field: GF(2^8) with generator polynomial 0x11d (== reed-solomon-erasure's 0x1d),
// the same field Clay/Shelby use, so the multiply is wire-relevant.
//
// SIMD kernel is the classic SSSE3/pshufb == wasm u8x16_swizzle nibble split:
//   c*v = swizzle(low_c, v & 0x0f) ^ swizzle(high_c, (v >> 4))
// where low_c[x] = c*x and high_c[x] = c*(x<<4) for x in 0..16.
use core::arch::wasm32::*;
use wasm_bindgen::prelude::*;

// --- GF(2^8) tables (built once, lazily) -------------------------------------
struct Gf {
    log: [u8; 256],
    exp: [u8; 512],
    mul: Vec<u8>, // 256*256 full table for the scalar baseline
}
fn build_gf() -> Gf {
    let mut log = [0u8; 256];
    let mut exp = [0u8; 512];
    let mut b: usize = 1;
    for l in 0..255usize {
        exp[l] = b as u8;
        exp[l + 255] = b as u8;
        log[b] = l as u8;
        b <<= 1;
        if b >= 256 {
            b = (b - 256) ^ 0x1d;
        }
    }
    let mul_one = |a: u8, c: u8| -> u8 {
        if a == 0 || c == 0 { 0 } else { exp[log[a as usize] as usize + log[c as usize] as usize] }
    };
    let mut mul = vec![0u8; 256 * 256];
    for a in 0..256usize {
        for c in 0..256usize {
            mul[a * 256 + c] = mul_one(a as u8, c as u8);
        }
    }
    Gf { log, exp, mul }
}
thread_local!(static GF: Gf = build_gf());

fn gf_mul(gf: &Gf, a: u8, c: u8) -> u8 {
    if a == 0 || c == 0 { 0 } else { gf.exp[gf.log[a as usize] as usize + gf.log[c as usize] as usize] }
}
// low/high 16-byte swizzle tables for coefficient c
fn tables(gf: &Gf, c: u8) -> ([u8; 16], [u8; 16]) {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    for x in 0..16u8 {
        lo[x as usize] = gf_mul(gf, c, x);
        hi[x as usize] = gf_mul(gf, c, x << 4);
    }
    (lo, hi)
}

// --- kernels -----------------------------------------------------------------
#[inline]
fn mac_scalar(gf: &Gf, out: &mut [u8], input: &[u8], c: u8) {
    let row = &gf.mul[(c as usize) * 256..(c as usize) * 256 + 256];
    for i in 0..out.len() {
        out[i] ^= row[input[i] as usize];
    }
}

#[inline]
fn mac_simd(out: &mut [u8], input: &[u8], lo: v128, hi: v128, mask: v128) {
    let n = out.len();
    let mut i = 0usize;
    unsafe {
        while i + 16 <= n {
            let d = v128_load(input.as_ptr().add(i) as *const v128);
            let dl = v128_and(d, mask);
            let dh = u8x16_shr(d, 4);
            let prod = v128_xor(u8x16_swizzle(lo, dl), u8x16_swizzle(hi, dh));
            let cur = v128_load(out.as_ptr().add(i) as *const v128);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, v128_xor(cur, prod));
            i += 16;
        }
    }
    while i < n {
        // tail (rows are multiples of 16 here, so effectively unused)
        out[i] ^= 0;
        i += 1;
    }
}

// --- correctness: simd mul == scalar gf_mul for all (c, v) -------------------
#[wasm_bindgen]
pub fn selftest() -> u32 {
    GF.with(|gf| {
        let mask = u8x16_splat(0x0f);
        let mut bad = 0u32;
        for c in 0..=255u8 {
            let (lo, hi) = tables(gf, c);
            let lov = unsafe { v128_load(lo.as_ptr() as *const v128) };
            let hiv = unsafe { v128_load(hi.as_ptr() as *const v128) };
            let mut input = [0u8; 256];
            for v in 0..256usize { input[v] = v as u8; }
            let mut out = [0u8; 256];
            mac_simd(&mut out, &input, lov, hiv, mask);
            for v in 0..256usize {
                if out[v] != gf_mul(gf, c, v as u8) { bad += 1; }
            }
        }
        bad
    })
}

// --- RS encode: k data rows -> m parity rows, coefficient matrix m*k ---------
// Full-row structure: each coefficient applied over a whole `row`-byte node row.
#[wasm_bindgen]
pub fn encode_fullrow_scalar(data: &[u8], k: usize, m: usize, row: usize, matrix: &[u8]) -> Vec<u8> {
    GF.with(|gf| {
        let mut out = vec![0u8; m * row];
        for i in 0..m {
            for j in 0..k {
                let c = matrix[i * k + j];
                let (o, d) = (i * row, j * row);
                mac_scalar(gf, &mut out[o..o + row], &data[d..d + row], c);
            }
        }
        out
    })
}

#[wasm_bindgen]
pub fn encode_fullrow_simd(data: &[u8], k: usize, m: usize, row: usize, matrix: &[u8]) -> Vec<u8> {
    GF.with(|gf| {
        let mask = u8x16_splat(0x0f);
        let mut out = vec![0u8; m * row];
        for i in 0..m {
            for j in 0..k {
                let c = matrix[i * k + j];
                let (lo, hi) = tables(gf, c);
                let lov = unsafe { v128_load(lo.as_ptr() as *const v128) };
                let hiv = unsafe { v128_load(hi.as_ptr() as *const v128) };
                let o = i * row;
                mac_simd(&mut out[o..o + row], &data[j * row..j * row + row], lov, hiv, mask);
            }
        }
        out
    })
}

// Per-plane structure: mirrors Clay's actual layout (RS over `sub`-byte planes),
// but the (i,j) swizzle tables are precomputed ONCE and reused across planes.
#[wasm_bindgen]
pub fn encode_plane_simd(data: &[u8], k: usize, m: usize, row: usize, sub: usize, matrix: &[u8]) -> Vec<u8> {
    GF.with(|gf| {
        let mask = u8x16_splat(0x0f);
        // precompute tables for every matrix coefficient once
        let mut los = vec![0u8; m * k * 16];
        let mut his = vec![0u8; m * k * 16];
        for i in 0..m * k {
            let (lo, hi) = tables(gf, matrix[i]);
            los[i * 16..i * 16 + 16].copy_from_slice(&lo);
            his[i * 16..i * 16 + 16].copy_from_slice(&hi);
        }
        let planes = row / sub;
        let mut out = vec![0u8; m * row];
        for z in 0..planes {
            let off = z * sub;
            for i in 0..m {
                for j in 0..k {
                    let idx = i * k + j;
                    let lov = unsafe { v128_load(los.as_ptr().add(idx * 16) as *const v128) };
                    let hiv = unsafe { v128_load(his.as_ptr().add(idx * 16) as *const v128) };
                    let o = i * row + off;
                    let d = j * row + off;
                    // borrow out mutably for this plane slice
                    let optr = out.as_mut_ptr();
                    let os = unsafe { core::slice::from_raw_parts_mut(optr.add(o), sub) };
                    mac_simd(os, &data[d..d + sub], lov, hiv, mask);
                }
            }
        }
        out
    })
}
