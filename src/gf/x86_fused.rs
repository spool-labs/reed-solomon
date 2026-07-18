// SPDX-License-Identifier: Apache-2.0
//! Fused multi-output GF(2^8) dot product for x86_64: one streaming pass over the
//! `k` inputs produces `NOUT` parity outputs at once. GFNI + AVX-512 uses affine
//! multiply with a 3-way-XOR (`vpternlogd`) fold; AVX2-only uses nibble-split.
//! Byte-identical to the per-shard encode (GF addition is XOR).
#![cfg(target_arch = "x86_64")]

use super::gfni::gfni_matrix;
use crate::galois;
use core::arch::x86_64::*;

/// Upper bound on data + parity shards (`ReedSolomon::new` rejects more), so input
/// pointers fit a stack array and encode needs no per-call heap allocation.
const MAX_SHARDS: usize = 256;

/// The GFNI affine broadcast (`i64`) for "multiply by `c`" — a thin re-export of
/// `super::gfni::gfni_matrix` so callers can precompute and **cache** the
/// per-coefficient matrices once (e.g. `ReedSolomon` builds them at construction),
/// avoiding a `gfni_matrix` rebuild on every encode.
pub fn affine_of(c: u8) -> i64 {
    gfni_matrix(c)
}

/// Affine matrices (and raw coefficients) for a whole `NOUT x k` coefficient
/// matrix, row-major. Held in memory deliberately so `vgf2p8affineqb` can fold
/// the `set1_epi64` broadcast into its `m64bcst` operand.
pub struct AffineMatrices {
    k: usize,
    coeffs: Vec<u8>,
    m: Vec<i64>, // per (j,i): the 8x8 GF(2) bit-matrix for "multiply by coeff"
}

impl AffineMatrices {
    /// `matrix` is row-major `nout x k`: coefficient for (output `j`, input `i`)
    /// at `matrix[j * k + i]`.
    pub fn new(matrix: &[u8], nout: usize, k: usize) -> Self {
        assert_eq!(matrix.len(), nout * k);
        Self {
            k,
            coeffs: matrix.to_vec(),
            m: matrix.iter().map(|&c| gfni_matrix(c)).collect(),
        }
    }

    #[inline(always)]
    unsafe fn at(&self, j: usize, i: usize) -> i64 {
        *self.m.get_unchecked(j * self.k + i)
    }

    /// Raw coefficient for (output `j`, input `i`) — used by the tail path.
    #[inline(always)]
    fn coeff(&self, j: usize, i: usize) -> u8 {
        self.coeffs[j * self.k + i]
    }
}

/// `out[j][b] = Σ_i coeff[base+j][i] * input[i][b]`, all `j` in one pass. `base`
/// is the first output-row index in `mats`, so one `AffineMatrices` covering all
/// m outputs serves every tile.
///
/// # Safety
/// Requires `gfni` + `avx512f`. All shards must be equal length; `out.len() ==
/// NOUT`; `input.len() == mats.k`; `mats` holds at least `base + NOUT` rows.
#[target_feature(enable = "gfni,avx512f")]
pub unsafe fn dot_prod_gfni<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    out: &mut [Out],
    input: &[In],
    mats: &AffineMatrices,
    base: usize,
) {
    let k = input.len();
    let len = if k > 0 { input[0].as_ref().len() } else { 0 };
    debug_assert_eq!(mats.k, k);
    debug_assert_eq!(out.len(), NOUT);

    let mut ins = [core::ptr::null::<u8>(); MAX_SHARDS];
    for i in 0..k {
        ins[i] = input[i].as_ref().as_ptr();
    }
    let out_ptr: [*mut u8; NOUT] = core::array::from_fn(|j| out[j].as_mut().as_mut_ptr());

    // Main loop: 128 bytes (2x64B) per block, all NOUT outputs at once.
    let mut b = 0usize;
    while b + 128 <= len {
        let mut acc = [[_mm512_setzero_si512(); 2]; NOUT];

        // Shard pairs: two affines combined with one ternlog (3-way XOR).
        let mut i = 0usize;
        while i + 2 <= k {
            let p0 = ins.get_unchecked(i).add(b) as *const __m512i;
            let p1 = ins.get_unchecked(i + 1).add(b) as *const __m512i;
            let a = [_mm512_loadu_si512(p0), _mm512_loadu_si512(p0.add(1))];
            let c = [_mm512_loadu_si512(p1), _mm512_loadu_si512(p1.add(1))];

            for j in 0..NOUT {
                let m0 = _mm512_set1_epi64(mats.at(base + j, i));
                let m1 = _mm512_set1_epi64(mats.at(base + j, i + 1));
                for u in 0..2 {
                    acc[j][u] = _mm512_ternarylogic_epi32::<0x96>(
                        acc[j][u],
                        _mm512_gf2p8affine_epi64_epi8::<0>(a[u], m0),
                        _mm512_gf2p8affine_epi64_epi8::<0>(c[u], m1),
                    );
                }
            }
            i += 2;
        }

        // Odd shard out: plain xor-accumulate.
        if i < k {
            let p = ins.get_unchecked(i).add(b) as *const __m512i;
            let a = [_mm512_loadu_si512(p), _mm512_loadu_si512(p.add(1))];
            for j in 0..NOUT {
                let m = _mm512_set1_epi64(mats.at(base + j, i));
                for u in 0..2 {
                    acc[j][u] = _mm512_xor_si512(
                        acc[j][u],
                        _mm512_gf2p8affine_epi64_epi8::<0>(a[u], m),
                    );
                }
            }
        }

        // Each parity block stored exactly once.
        for j in 0..NOUT {
            let d = out_ptr[j].add(b) as *mut __m512i;
            _mm512_storeu_si512(d, acc[j][0]);
            _mm512_storeu_si512(d.add(1), acc[j][1]);
        }
        b += 128;
    }

    // Tail (< 128 bytes): GF addition is XOR, so accumulation order is irrelevant.
    if b < len {
        for j in 0..NOUT {
            for i in 0..k {
                let cf = mats.coeff(base + j, i);
                if i == 0 {
                    super::mul_slice(&mut out[j].as_mut()[b..], &input[0].as_ref()[b..], cf);
                } else {
                    super::mul_slice_xor(&mut out[j].as_mut()[b..], &input[i].as_ref()[b..], cf);
                }
            }
        }
    }
}

// AVX2-only fused path (no GFNI/AVX-512): same one-load-per-input dataflow as the
// GFNI kernel but with `vpshufb` nibble-split, tiled to fit AVX2's 16 YMM registers.

/// Pre-broadcast `vpshufb` nibble tables for a row-major `NOUT x k` coefficient
/// matrix: `lo[x] = c*x`, `hi[x] = c*(x<<4)`, each duplicated into both 128-bit
/// lanes. `ReedSolomon` caches one at construction so encode rebuilds nothing.
#[derive(Debug, Clone)]
pub struct NibbleTables {
    k: usize,
    coeffs: Vec<u8>,
    lo: Vec<u8>,
    hi: Vec<u8>,
}

impl NibbleTables {
    /// `matrix` is row-major `nout x k`: coefficient for (output `j`, input `i`)
    /// at `matrix[j * k + i]`.
    pub fn new(matrix: &[u8], nout: usize, k: usize) -> Self {
        assert_eq!(matrix.len(), nout * k);
        let mut lo = vec![0u8; nout * k * 32];
        let mut hi = vec![0u8; nout * k * 32];
        for (idx, &c) in matrix.iter().enumerate() {
            let base = idx * 32;
            for x in 0..16usize {
                let l = galois::mul(c, x as u8);
                let h = galois::mul(c, (x as u8) << 4);
                lo[base + x] = l;
                lo[base + 16 + x] = l;
                hi[base + x] = h;
                hi[base + 16 + x] = h;
            }
        }
        Self {
            k,
            coeffs: matrix.to_vec(),
            lo,
            hi,
        }
    }

    #[inline(always)]
    unsafe fn lo_ptr(&self, j: usize, i: usize) -> *const u8 {
        self.lo.as_ptr().add((j * self.k + i) * 32)
    }

    #[inline(always)]
    unsafe fn hi_ptr(&self, j: usize, i: usize) -> *const u8 {
        self.hi.as_ptr().add((j * self.k + i) * 32)
    }

    /// Raw coefficient for (output `j`, input `i`) — used by the tail path.
    #[inline(always)]
    fn coeff(&self, j: usize, i: usize) -> u8 {
        self.coeffs[j * self.k + i]
    }
}

/// `out[j][b] = Σ_i coeff[base+j][i] * input[i][b]`, all `NOUT` outputs in one
/// streaming pass. `base` is the first output-row index in `tab`, so one
/// `NibbleTables` covering all m outputs serves every tile.
///
/// # Safety
/// Requires `avx2`. All shards must be equal length; `out.len() == NOUT`;
/// `input.len() == tab.k`; `tab` holds at least `base + NOUT` rows.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_prod_avx2<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    out: &mut [Out],
    input: &[In],
    tab: &NibbleTables,
    base: usize,
) {
    let k = input.len();
    let len = if k > 0 { input[0].as_ref().len() } else { 0 };
    debug_assert_eq!(tab.k, k);
    debug_assert_eq!(out.len(), NOUT);

    let mut ins = [core::ptr::null::<u8>(); MAX_SHARDS];
    for i in 0..k {
        ins[i] = input[i].as_ref().as_ptr();
    }
    let out_ptr: [*mut u8; NOUT] = core::array::from_fn(|j| out[j].as_mut().as_mut_ptr());
    let mask = _mm256_set1_epi8(0x0f);

    // Main loop: 32 bytes per block, all NOUT outputs at once. Each input is
    // loaded once and its two nibble indices reused across the whole tile; only
    // the accumulators stay resident, so NOUT<=6 leaves headroom in 16 YMM.
    let mut b = 0usize;
    while b + 32 <= len {
        let mut acc = [_mm256_setzero_si256(); NOUT];
        for i in 0..k {
            let v = _mm256_loadu_si256(ins.get_unchecked(i).add(b) as *const __m256i);
            let lo_idx = _mm256_and_si256(v, mask);
            let hi_idx = _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask);
            for j in 0..NOUT {
                let lo_t = _mm256_loadu_si256(tab.lo_ptr(base + j, i) as *const __m256i);
                let hi_t = _mm256_loadu_si256(tab.hi_ptr(base + j, i) as *const __m256i);
                let prod = _mm256_xor_si256(
                    _mm256_shuffle_epi8(lo_t, lo_idx),
                    _mm256_shuffle_epi8(hi_t, hi_idx),
                );
                acc[j] = _mm256_xor_si256(acc[j], prod);
            }
        }
        for j in 0..NOUT {
            let d = out_ptr[j].add(b) as *mut __m256i;
            _mm256_storeu_si256(d, acc[j]);
        }
        b += 32;
    }

    // Tail (< 32 bytes): GF addition is XOR, so accumulation order is irrelevant.
    if b < len {
        for j in 0..NOUT {
            for i in 0..k {
                let cf = tab.coeff(base + j, i);
                if i == 0 {
                    super::mul_slice(&mut out[j].as_mut()[b..], &input[0].as_ref()[b..], cf);
                } else {
                    super::mul_slice_xor(&mut out[j].as_mut()[b..], &input[i].as_ref()[b..], cf);
                }
            }
        }
    }
}

/// Fused encode of all `m = parity.len()` outputs, tiling into `NOUT <= 6` passes:
/// fused GFNI on GFNI+AVX-512, fused AVX2 on AVX2-only, else per-shard. All shards
/// share one length; `gen_rows[o]` holds the `k` coefficients for output `o`.
pub fn encode<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
) {
    if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx512f") {
        let m = parity.len();
        let k = data.len();

        // Affine matrices for the whole m x k generator, built once per encode;
        // tiles view into it via a base output-row offset (no per-tile alloc).
        let mut flat = vec![0u8; m * k];
        for o in 0..m {
            let row = gen_rows[o].as_ref();
            for i in 0..k {
                flat[o * k + i] = row[i];
            }
        }
        let mats = AffineMatrices::new(&flat, m, k);

        let mut o = 0usize;
        while o < m {
            let n = (m - o).min(6);
            // SAFETY: gfni + avx512f just checked; shard-length / count invariants
            // are the caller's contract (upheld by `encode_fused`).
            unsafe {
                match n {
                    6 => run_tile::<6, _, _>(o, data, parity, &mats),
                    5 => run_tile::<5, _, _>(o, data, parity, &mats),
                    4 => run_tile::<4, _, _>(o, data, parity, &mats),
                    3 => run_tile::<3, _, _>(o, data, parity, &mats),
                    2 => run_tile::<2, _, _>(o, data, parity, &mats),
                    _ => run_tile::<1, _, _>(o, data, parity, &mats),
                }
            }
            o += n;
        }
    } else if is_x86_feature_detected!("avx2") {
        // AVX2-only: fuse outputs; tables built per call here (the cached-table
        // entry point is `encode_ymm_dispatch`).
        let m = parity.len();
        let k = data.len();

        let mut flat = vec![0u8; m * k];
        for o in 0..m {
            let row = gen_rows[o].as_ref();
            for i in 0..k {
                flat[o * k + i] = row[i];
            }
        }
        let tab = NibbleTables::new(&flat, m, k);
        // SAFETY: avx2 just checked; `tab` covers all m x k; shard-length / count
        // invariants are the caller's contract (upheld by `encode_fused`).
        unsafe { encode_avx2_tiles(&tab, data, parity) };
    } else {
        // No SIMD at all (pre-AVX2 x86): correctness via the per-shard kernels.
        encode_fallback(gen_rows, data, parity);
    }
}

/// Run the fused AVX2 path over all `parity.len()` outputs, tiling into `NOUT<=6`
/// passes against `tab`. Shared by the per-call [`encode`] branch and the
/// cached-table [`encode_ymm_dispatch`] path.
///
/// # Safety
/// Caller guarantees `avx2`, that `tab` covers all `parity.len()` outputs over
/// `data.len()` inputs, and the equal-length shard contract. `dot_prod_avx2`
/// handles any length (< 32 falls through to its scalar tail).
#[inline]
unsafe fn encode_avx2_tiles<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    tab: &NibbleTables,
    data: &[In],
    parity: &mut [Out],
) {
    let m = parity.len();
    let mut o = 0usize;
    while o < m {
        let n = (m - o).min(6);
        match n {
            6 => run_tile_avx2::<6, _, _>(o, data, parity, tab),
            5 => run_tile_avx2::<5, _, _>(o, data, parity, tab),
            4 => run_tile_avx2::<4, _, _>(o, data, parity, tab),
            3 => run_tile_avx2::<3, _, _>(o, data, parity, tab),
            2 => run_tile_avx2::<2, _, _>(o, data, parity, tab),
            _ => run_tile_avx2::<1, _, _>(o, data, parity, tab),
        }
        o += n;
    }
}

/// One fixed-`NOUT` tile: run the fused kernel for output rows `[o, o+NOUT)`
/// (looked up in `mats` via the base offset) into `parity[o .. o+NOUT]`.
///
/// # Safety
/// Caller guarantees `gfni` + `avx512f`, that `mats` holds at least `o + NOUT`
/// rows, and that there are at least `o + NOUT` parity rows.
#[inline]
unsafe fn run_tile<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    o: usize,
    data: &[In],
    parity: &mut [Out],
    mats: &AffineMatrices,
) {
    let out = &mut parity[o..o + NOUT];
    unsafe { dot_prod_gfni::<NOUT, _, _>(out, data, mats, o) };
}

/// One fixed-`NOUT` AVX2 tile.
///
/// # Safety
/// Caller guarantees `avx2` and that `tab` and `parity` cover `o + NOUT` rows.
#[inline]
unsafe fn run_tile_avx2<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    o: usize,
    data: &[In],
    parity: &mut [Out],
    tab: &NibbleTables,
) {
    let out = &mut parity[o..o + NOUT];
    unsafe { dot_prod_avx2::<NOUT, _, _>(out, data, tab, o) };
}

/// Per-shard fallback used when the CPU lacks GFNI / AVX-512. Routes through the
/// dispatch kernels (`super::mul_slice` — SIMD nibble/scalar as available).
fn encode_fallback<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
) {
    let k = data.len();
    for (o, out) in parity.iter_mut().enumerate() {
        let out = out.as_mut();
        for i in 0..k {
            let c = gen_rows[o].as_ref()[i];
            if i == 0 {
                super::mul_slice(out, data[0].as_ref(), c);
            } else {
                super::mul_slice_xor(out, data[i].as_ref(), c);
            }
        }
    }
}

// 256-bit variant for the production shapes: at 256 bits all M outputs fit the
// 32-register YMM file, so each input is loaded once per 32-byte block and reused
// across every output (throughput stays flat past L2). An overlapped final block
// (clamp pos = len-32, recompute — idempotent) removes the scalar tail.

/// `parity[o][..] = Σ_i gen_rows[o][i] * data[i][..]`, all outputs per 32-byte
/// block, register-resident. Overlapped final block (no scalar tail).
///
/// # Safety
/// Requires gfni + avx2 + avx512f + avx512vl. `data.len() == K`, `parity.len() ==
/// M`, every shard equal length `>= 32`; `gen_rows[o]` has `K` coefficients.
#[allow(dead_code)] // unused when a backend is pinned (dispatch bypassed)
#[target_feature(enable = "gfni,avx2,avx512f,avx512vl")]
unsafe fn encode_ymm<const K: usize, const M: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    affine: &[i64],
    data: &[In],
    parity: &mut [Out],
) {
    const W: usize = 32;
    let len = data[0].as_ref().len();

    // Broadcast the cached affine matrices (row-major M x K) into registers.
    let mut mats = [[_mm256_setzero_si256(); K]; M];
    for o in 0..M {
        for i in 0..K {
            mats[o][i] = _mm256_set1_epi64x(affine[o * K + i]);
        }
    }
    let in_ptr: [*const u8; K] = core::array::from_fn(|i| data[i].as_ref().as_ptr());
    let out_ptr: [*mut u8; M] = core::array::from_fn(|o| parity[o].as_mut().as_mut_ptr());

    let mut pos = 0usize;
    loop {
        let p = if pos + W <= len { pos } else { len - W };

        // Load every input block once; held in registers across all M outputs.
        let mut inb = [_mm256_setzero_si256(); K];
        for i in 0..K {
            inb[i] = _mm256_loadu_si256(in_ptr[i].add(p) as *const __m256i);
        }

        for o in 0..M {
            // acc = c0*in0; fold input pairs with ternlog (3-way XOR).
            let mut acc = _mm256_gf2p8affine_epi64_epi8::<0>(inb[0], mats[o][0]);
            let mut i = 1usize;
            while i + 1 < K {
                let a = _mm256_gf2p8affine_epi64_epi8::<0>(inb[i], mats[o][i]);
                let b = _mm256_gf2p8affine_epi64_epi8::<0>(inb[i + 1], mats[o][i + 1]);
                acc = _mm256_ternarylogic_epi32::<0x96>(acc, a, b);
                i += 2;
            }
            if i < K {
                let a = _mm256_gf2p8affine_epi64_epi8::<0>(inb[i], mats[o][i]);
                acc = _mm256_xor_si256(acc, a);
            }
            _mm256_storeu_si256(out_ptr[o].add(p) as *mut __m256i, acc);
        }

        if p + W >= len {
            break;
        }
        pos += W;
    }
}

/// Fused encode entry: cached GFNI/AVX2 kernels with runtime dispatch, or a
/// pinned kernel when a backend feature is set.
#[allow(unused_variables)]
pub fn encode_ymm_dispatch<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    affine: &[i64],
    nibbles: &NibbleTables,
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
) {
    #[cfg(any(feature = "scalar", feature = "ssse3", feature = "avx512"))]
    return encode_fallback(gen_rows, data, parity);
    #[cfg(feature = "avx2")]
    // SAFETY: the `avx2` feature pins a target that supports AVX2.
    return unsafe { encode_avx2_tiles(nibbles, data, parity) };
    #[cfg(feature = "gfni")]
    {
        let (k, m) = (data.len(), parity.len());
        let len = data.first().map(|s| s.as_ref().len()).unwrap_or(0);
        if len >= 32 {
            // SAFETY: the `gfni` feature pins a target with GFNI + AVX-512(VL).
            unsafe {
                match (k, m) {
                    (7, 13) => return encode_ymm::<7, 13, _, _>(affine, data, parity),
                    (10, 10) => return encode_ymm::<10, 10, _, _>(affine, data, parity),
                    _ => {}
                }
            }
        }
        return encode(gen_rows, data, parity);
    }

    #[cfg(not(any(
        feature = "scalar",
        feature = "ssse3",
        feature = "avx2",
        feature = "avx512",
        feature = "gfni",
    )))]
    {
        let (k, m) = (data.len(), parity.len());
        let len = data.first().map(|s| s.as_ref().len()).unwrap_or(0);
        let specialised = is_x86_feature_detected!("gfni")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx2");
        if specialised && len >= 32 {
            // SAFETY: features checked; len >= 32; caller upholds shard invariants.
            unsafe {
                match (k, m) {
                    (7, 13) => return encode_ymm::<7, 13, _, _>(affine, data, parity),
                    (10, 10) => return encode_ymm::<10, 10, _, _>(affine, data, parity),
                    _ => {}
                }
            }
        }
        if is_x86_feature_detected!("avx2")
            && !(is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx512f"))
        {
            debug_assert_eq!(nibbles.k, k);
            // SAFETY: avx2 checked; nibbles cover all m*k; equal-length contract holds.
            unsafe { encode_avx2_tiles(nibbles, data, parity) };
            return;
        }
        encode(gen_rows, data, parity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::scalar;

    fn scalar_reference(
        matrix: &[u8],
        nout: usize,
        k: usize,
        data: &[Vec<u8>],
        len: usize,
    ) -> Vec<Vec<u8>> {
        let mut want = vec![vec![0u8; len]; nout];
        for j in 0..nout {
            for i in 0..k {
                let c = matrix[j * k + i];
                if i == 0 {
                    scalar::mul_slice(&mut want[j], &data[i], c);
                } else {
                    scalar::mul_slice_xor(&mut want[j], &data[i], c);
                }
            }
        }
        want
    }

    fn gen_data(k: usize, len: usize) -> Vec<Vec<u8>> {
        (0..k)
            .map(|s| (0..len).map(|i| ((i * 31 + s * 17) as u8) ^ 0x5a).collect())
            .collect()
    }

    fn gen_matrix(nout: usize, k: usize) -> Vec<u8> {
        (0..nout * k).map(|i| (i as u8).wrapping_mul(53) | 1).collect()
    }

    /// # Safety: caller ensures `gfni` + `avx512f`.
    unsafe fn run_one<const NOUT: usize>(
        matrix: &[u8],
        k: usize,
        data: &[Vec<u8>],
        len: usize,
    ) -> Vec<Vec<u8>> {
        let mats = AffineMatrices::new(matrix, NOUT, k);
        let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let mut got = vec![vec![0u8; len]; NOUT];
        {
            let mut outs: Vec<&mut [u8]> = got.iter_mut().map(|v| v.as_mut_slice()).collect();
            unsafe { dot_prod_gfni::<NOUT, _, _>(&mut outs, &ins, &mats, 0) };
        }
        got
    }

    // Lengths crossing the 128-byte block boundary in every phase plus tails.
    const LENS: &[usize] = &[0, 1, 7, 127, 128, 129, 200, 255, 256, 257, 1000];

    macro_rules! check_nout {
        ($nout:literal) => {{
            for &k in &[1usize, 3, 6, 10, 20] {
                for &len in LENS {
                    let data = gen_data(k, len);
                    let matrix = gen_matrix($nout, k);
                    let want = scalar_reference(&matrix, $nout, k, &data, len);
                    let got = unsafe { run_one::<$nout>(&matrix, k, &data, len) };
                    assert_eq!(got, want, "gfni NOUT={} k={} len={}", $nout, k, len);
                }
            }
        }};
    }

    /// Differential vs scalar; a no-op on hosts without GFNI + AVX-512.
    #[test]
    fn fused_gfni_matches_scalar() {
        if !(is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx512f")) {
            return;
        }

        check_nout!(1);
        check_nout!(2);
        check_nout!(4);
        check_nout!(6);

        // The m > NOUT tiling driver (m=10 -> 6+4, m=7 -> 6+1, m=4 -> 4).
        for &(k, m) in &[(10usize, 10usize), (20, 10), (6, 4), (4, 7)] {
            for &len in &[100usize, 4096, 10000, 32768] {
                let matrix: Vec<u8> = (0..m * k)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(1))
                    .collect();
                let gen_rows: Vec<&[u8]> = (0..m).map(|r| &matrix[r * k..(r + 1) * k]).collect();
                let data = gen_data(k, len);
                let want = scalar_reference(&matrix, m, k, &data, len);

                let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
                let mut got = vec![vec![0u8; len]; m];
                {
                    let mut outs: Vec<&mut [u8]> =
                        got.iter_mut().map(|v| v.as_mut_slice()).collect();
                    encode(&gen_rows, &ins, &mut outs);
                }
                assert_eq!(got, want, "gfni driver k={k} m={m} len={len}");
            }
        }

        // 256-bit register-resident variant at the production shapes, incl.
        // lengths that exercise the overlapped tail (non-multiples of 32).
        if is_x86_feature_detected!("avx512vl") {
            for &(k, m) in &[(7usize, 13usize), (10, 10)] {
                for &len in &[32usize, 33, 63, 100, 1000, 4096, 10000] {
                    let matrix: Vec<u8> = (0..m * k)
                        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(1))
                        .collect();
                    let gen_rows: Vec<&[u8]> =
                        (0..m).map(|r| &matrix[r * k..(r + 1) * k]).collect();
                    let data = gen_data(k, len);
                    let want = scalar_reference(&matrix, m, k, &data, len);
                    let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
                    let mut got = vec![vec![0u8; len]; m];
                    {
                        let affine: Vec<i64> = matrix.iter().map(|&c| affine_of(c)).collect();
                        let nibbles = NibbleTables::new(&matrix, m, k);
                        let mut outs: Vec<&mut [u8]> =
                            got.iter_mut().map(|v| v.as_mut_slice()).collect();
                        encode_ymm_dispatch(&affine, &nibbles, &gen_rows, &ins, &mut outs);
                    }
                    assert_eq!(got, want, "ymm k={k} m={m} len={len}");
                }
            }
        }
    }

    /// # Safety: caller ensures `avx2`.
    unsafe fn run_one_avx2<const NOUT: usize>(
        matrix: &[u8],
        k: usize,
        data: &[Vec<u8>],
        len: usize,
    ) -> Vec<Vec<u8>> {
        let tab = NibbleTables::new(matrix, NOUT, k);
        let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let mut got = vec![vec![0u8; len]; NOUT];
        {
            let mut outs: Vec<&mut [u8]> = got.iter_mut().map(|v| v.as_mut_slice()).collect();
            unsafe { dot_prod_avx2::<NOUT, _, _>(&mut outs, &ins, &tab, 0) };
        }
        got
    }

    /// Differential vs scalar for the fused AVX2 path; a no-op without AVX2.
    #[test]
    fn fused_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // Lengths crossing the 32-byte block boundary in every phase plus tails.
        let lens = [0usize, 1, 7, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000];

        macro_rules! check_nout_avx2 {
            ($nout:literal) => {{
                for &k in &[1usize, 3, 6, 10, 20] {
                    for &len in &lens {
                        let data = gen_data(k, len);
                        let matrix = gen_matrix($nout, k);
                        let want = scalar_reference(&matrix, $nout, k, &data, len);
                        let got = unsafe { run_one_avx2::<$nout>(&matrix, k, &data, len) };
                        assert_eq!(got, want, "avx2 NOUT={} k={} len={}", $nout, k, len);
                    }
                }
            }};
        }

        check_nout_avx2!(1);
        check_nout_avx2!(2);
        check_nout_avx2!(4);
        check_nout_avx2!(6);

        // The m > NOUT tiling driver through `encode` (m=13 -> 6+6+1, m=10 -> 6+4,
        // m=7 -> 6+1). On this AVX2-only-in-test path `encode` routes here.
        for &(k, m) in &[(7usize, 13usize), (10, 10), (20, 10), (6, 4), (4, 7)] {
            for &len in &[32usize, 33, 100, 4096, 10000, 32768] {
                let matrix: Vec<u8> = (0..m * k)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(1))
                    .collect();
                let gen_rows: Vec<&[u8]> = (0..m).map(|r| &matrix[r * k..(r + 1) * k]).collect();
                let data = gen_data(k, len);
                let want = scalar_reference(&matrix, m, k, &data, len);

                let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
                let mut got = vec![vec![0u8; len]; m];
                {
                    let mut outs: Vec<&mut [u8]> =
                        got.iter_mut().map(|v| v.as_mut_slice()).collect();
                    encode(&gen_rows, &ins, &mut outs);
                }
                assert_eq!(got, want, "avx2 driver k={k} m={m} len={len}");
            }
        }
    }
}
