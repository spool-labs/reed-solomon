//! wasm32 simd128 GF(2^8) slice-multiply kernel.
#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use crate::galois::MUL_TABLE;
use core::arch::wasm32::*;

/// Upper bound on data + parity shards, so input pointers fit a stack array.
const MAX_SHARDS: usize = 256;

/// Build the low/high 16-byte swizzle tables for coefficient `c`
#[inline]
fn tables(c: u8) -> ([u8; 16], [u8; 16]) {
    let row = &MUL_TABLE[c as usize];
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    let mut x = 0usize;
    while x < 16 {
        lo[x] = row[x];
        hi[x] = row[x << 4];
        x += 1;
    }
    (lo, hi)
}

/// `out[i] = coeff * input[i]` over GF(2^8)
#[inline]
pub fn mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    let (lo, hi) = tables(coeff);
    let mask = u8x16_splat(0x0f);
    let lov = unsafe { v128_load(lo.as_ptr() as *const v128) };
    let hiv = unsafe { v128_load(hi.as_ptr() as *const v128) };

    let n = out.len();
    let mut i = 0usize;
    unsafe {
        while i + 16 <= n {
            let d = v128_load(input.as_ptr().add(i) as *const v128);
            let dl = v128_and(d, mask);
            let dh = u8x16_shr(d, 4);
            let prod = v128_xor(u8x16_swizzle(lov, dl), u8x16_swizzle(hiv, dh));
            v128_store(out.as_mut_ptr().add(i) as *mut v128, prod);
            i += 16;
        }
    }
    // Scalar tail for the remaining <16 bytes.
    if i < n {
        let row = &MUL_TABLE[coeff as usize];
        while i < n {
            out[i] = row[input[i] as usize];
            i += 1;
        }
    }
}

/// `out[i] ^= coeff * input[i]` over GF(2^8)
#[inline]
pub fn mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    let (lo, hi) = tables(coeff);
    let mask = u8x16_splat(0x0f);
    let lov = unsafe { v128_load(lo.as_ptr() as *const v128) };
    let hiv = unsafe { v128_load(hi.as_ptr() as *const v128) };

    let n = out.len();
    let mut i = 0usize;
    unsafe {
        while i + 16 <= n {
            let d = v128_load(input.as_ptr().add(i) as *const v128);
            let dl = v128_and(d, mask);
            let dh = u8x16_shr(d, 4);
            let prod = v128_xor(u8x16_swizzle(lov, dl), u8x16_swizzle(hiv, dh));
            let cur = v128_load(out.as_ptr().add(i) as *const v128);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, v128_xor(cur, prod));
            i += 16;
        }
    }
    // Scalar tail for the remaining <16 bytes.
    if i < n {
        let row = &MUL_TABLE[coeff as usize];
        while i < n {
            out[i] ^= row[input[i] as usize];
            i += 1;
        }
    }
}

/// Compute all parity shards `parity[o] = Σ_i gen_rows[o][i] * data[i]`, with a per-coefficient fallback for unspecialised shapes or sub-block shards
pub fn encode_fused<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    tables: &[u8],
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
) {
    let (k, m) = (data.len(), parity.len());
    let len = data.first().map(|s| s.as_ref().len()).unwrap_or(0);
    if len >= 16 {
        // SAFETY: len >= 16; `tables` covers m*k; shard-count/length invariants
        // are the caller's contract (upheld by `ReedSolomon::encode`).
        unsafe {
            match (k, m) {
                (7, 13) => return encode_fused_k::<7, 13, _, _>(tables, data, parity),
                (10, 10) => return encode_fused_k::<10, 10, _, _>(tables, data, parity),
                _ => {}
            }
            // General: tile the m outputs into NOUT <= 6 passes.
            let mut o = 0usize;
            while o < m {
                let n = (m - o).min(6);
                let out = &mut parity[o..o + n];
                match n {
                    6 => tiled_k::<6, _, _>(out, data, tables, o),
                    5 => tiled_k::<5, _, _>(out, data, tables, o),
                    4 => tiled_k::<4, _, _>(out, data, tables, o),
                    3 => tiled_k::<3, _, _>(out, data, tables, o),
                    2 => tiled_k::<2, _, _>(out, data, tables, o),
                    _ => tiled_k::<1, _, _>(out, data, tables, o),
                }
                o += n;
            }
        }
        return;
    }
    // Fallback: per-coefficient (len < 16).
    let k = data.len();
    for (o, out) in parity.iter_mut().enumerate() {
        let out = out.as_mut();
        for i in 0..k {
            let c = gen_rows[o].as_ref()[i];
            if i == 0 {
                mul_slice(out, data[0].as_ref(), c);
            } else {
                mul_slice_xor(out, data[i].as_ref(), c);
            }
        }
    }
}

/// Specialised (const K, M) fused kernel: split each input once, reuse across the M outputs
///
/// # Safety
/// `data.len() == K`, `parity.len() == M`, every shard equal length `>= 16`, `tables` holds at least `M*K*32` bytes.
unsafe fn encode_fused_k<const K: usize, const M: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    tables: &[u8],
    data: &[In],
    parity: &mut [Out],
) {
    const W: usize = 16;
    let len = data[0].as_ref().len();
    let mask = u8x16_splat(0x0f);
    let tp = tables.as_ptr();
    let in_ptr: [*const u8; K] = core::array::from_fn(|i| data[i].as_ref().as_ptr());
    let out_ptr: [*mut u8; M] = core::array::from_fn(|o| parity[o].as_mut().as_mut_ptr());

    let mut pos = 0usize;
    loop {
        let p = if pos + W <= len { pos } else { len - W };

        // Nibble-split every input once; reused by all M outputs.
        let mut dl = [u8x16_splat(0); K];
        let mut dh = [u8x16_splat(0); K];
        for i in 0..K {
            let v = v128_load(in_ptr[i].add(p) as *const v128);
            dl[i] = v128_and(v, mask);
            dh[i] = u8x16_shr(v, 4);
        }

        for o in 0..M {
            let mut acc = u8x16_splat(0);
            for i in 0..K {
                let base = (o * K + i) * 32;
                let lov = v128_load(tp.add(base) as *const v128);
                let hiv = v128_load(tp.add(base + 16) as *const v128);
                acc = v128_xor(
                    acc,
                    v128_xor(u8x16_swizzle(lov, dl[i]), u8x16_swizzle(hiv, dh[i])),
                );
            }
            v128_store(out_ptr[o].add(p) as *mut v128, acc);
        }

        if p + W >= len {
            break;
        }
        pos += W;
    }
}

/// Tiled fused kernel: `NOUT` outputs at a time for a runtime `k`. `base` is the
/// first output-row index into `tables`; one input load serves all `NOUT`
/// outputs. Overlapped final block, so any `len >= 16` is covered.
///
/// # Safety
/// `out.len() == NOUT`, every shard equal length `>= 16`; `tables` holds at least
/// `(base + NOUT) * k * 32` bytes.
unsafe fn tiled_k<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    out: &mut [Out],
    input: &[In],
    tables: &[u8],
    base: usize,
) {
    const W: usize = 16;
    let k = input.len();
    let len = if k > 0 { input[0].as_ref().len() } else { 0 };
    let mask = u8x16_splat(0x0f);
    let tp = tables.as_ptr();
    let mut in_ptr = [core::ptr::null::<u8>(); MAX_SHARDS];
    for i in 0..k {
        in_ptr[i] = input[i].as_ref().as_ptr();
    }
    let out_ptr: [*mut u8; NOUT] = core::array::from_fn(|j| out[j].as_mut().as_mut_ptr());

    let mut pos = 0usize;
    loop {
        let p = if pos + W <= len { pos } else { len - W };
        let mut accs = [u8x16_splat(0); NOUT];
        for i in 0..k {
            let v = v128_load((*in_ptr.get_unchecked(i)).add(p) as *const v128);
            let lo_idx = v128_and(v, mask);
            let hi_idx = u8x16_shr(v, 4);
            for j in 0..NOUT {
                let bt = ((base + j) * k + i) * 32;
                let lov = v128_load(tp.add(bt) as *const v128);
                let hiv = v128_load(tp.add(bt + 16) as *const v128);
                accs[j] = v128_xor(
                    accs[j],
                    v128_xor(u8x16_swizzle(lov, lo_idx), u8x16_swizzle(hiv, hi_idx)),
                );
            }
        }
        for j in 0..NOUT {
            v128_store(out_ptr[j].add(p) as *mut v128, accs[j]);
        }
        if p + W >= len {
            break;
        }
        pos += W;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::scalar;

    // Every coeff 0..=255 over all 256 byte values plus an odd-length tail must match scalar::*.
    #[test]
    fn matches_scalar_all_coeffs() {
        let mut input = [0u8; 256];
        for (v, b) in input.iter_mut().enumerate() {
            *b = v as u8;
        }
        // Odd, non-multiple-of-16 length that still exercises the tail.
        let odd = &input[..251];

        for c in 0..=255u8 {
            // mul_slice, full 256-byte (16-aligned) input.
            let mut got = [0u8; 256];
            let mut want = [0u8; 256];
            mul_slice(&mut got, &input, c);
            scalar::mul_slice(&mut want, &input, c);
            assert_eq!(got, want, "mul_slice mismatch, coeff={c}");

            // mul_slice, odd length (scalar tail path).
            let mut got_odd = vec![0u8; odd.len()];
            let mut want_odd = vec![0u8; odd.len()];
            mul_slice(&mut got_odd, odd, c);
            scalar::mul_slice(&mut want_odd, odd, c);
            assert_eq!(got_odd, want_odd, "mul_slice odd mismatch, coeff={c}");

            // mul_slice_xor accumulates into a pre-seeded buffer.
            let seed: Vec<u8> = (0..256u32).map(|x| (x.wrapping_mul(37) ^ 0xa5) as u8).collect();
            let mut got_x = seed.clone();
            let mut want_x = seed.clone();
            mul_slice_xor(&mut got_x, &input, c);
            scalar::mul_slice_xor(&mut want_x, &input, c);
            assert_eq!(got_x, want_x, "mul_slice_xor mismatch, coeff={c}");

            // mul_slice_xor, odd length (scalar tail path).
            let mut got_xo = seed[..odd.len()].to_vec();
            let mut want_xo = seed[..odd.len()].to_vec();
            mul_slice_xor(&mut got_xo, odd, c);
            scalar::mul_slice_xor(&mut want_xo, odd, c);
            assert_eq!(got_xo, want_xo, "mul_slice_xor odd mismatch, coeff={c}");
        }
    }

    /// Fused kernel must be byte-identical to the per-shard scalar reference across specialised, fallback, and tail shapes
    #[test]
    fn fused_matches_scalar() {
        for &(k, m) in &[(7usize, 13usize), (10, 10), (4, 2)] {
            for &len in &[8usize, 15, 16, 17, 31, 100, 255, 256, 1000, 1430] {
                let matrix: Vec<u8> = (0..m * k).map(|i| ((i * 37 + 1) & 0xff) as u8).collect();
                let gen_rows: Vec<&[u8]> = (0..m).map(|r| &matrix[r * k..(r + 1) * k]).collect();

                // Cached lo/hi tables, exactly as `ReedSolomon::new` builds them.
                let mut tables = Vec::with_capacity(m * k * 32);
                for &c in &matrix {
                    let row = &MUL_TABLE[c as usize];
                    for x in 0..16 {
                        tables.push(row[x]);
                    }
                    for x in 0..16 {
                        tables.push(row[x << 4]);
                    }
                }

                let data: Vec<Vec<u8>> = (0..k)
                    .map(|s| (0..len).map(|i| ((i * 31 + s * 17) & 0xff) as u8).collect())
                    .collect();
                let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

                let mut want = vec![vec![0u8; len]; m];
                for o in 0..m {
                    for i in 0..k {
                        let c = matrix[o * k + i];
                        if i == 0 {
                            scalar::mul_slice(&mut want[o], &data[i], c);
                        } else {
                            scalar::mul_slice_xor(&mut want[o], &data[i], c);
                        }
                    }
                }

                let mut got = vec![vec![0u8; len]; m];
                {
                    let mut outs: Vec<&mut [u8]> =
                        got.iter_mut().map(|v| v.as_mut_slice()).collect();
                    encode_fused(&tables, &gen_rows, &ins, &mut outs);
                }
                assert_eq!(got, want, "fused k={k} m={m} len={len}");
            }
        }
    }
}
