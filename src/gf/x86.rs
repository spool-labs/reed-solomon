//! x86_64 GF(2^8) multiply kernels (SSSE3 / AVX2 / AVX-512 nibble-split, GFNI
//! affine), runtime-dispatched or pinned by feature. Byte-identical to the scalar
//! reference for all 256 coefficients; `XOR` is a const generic.
#![cfg(target_arch = "x86_64")]

use super::gfni::gfni_matrix;
use super::scalar;
use crate::galois;
use core::arch::x86_64::*;

/// 16-byte low/high nibble shuffle tables for coefficient `c`:
///   lo[x] = c * x,  hi[x] = c * (x << 4),  for x in 0..16.
#[inline]
fn nibble_tables(c: u8) -> ([u8; 16], [u8; 16]) {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    let mut x = 0u8;
    while x < 16 {
        lo[x as usize] = galois::mul(c, x);
        hi[x as usize] = galois::mul(c, x << 4);
        x += 1;
    }
    (lo, hi)
}

/// Scalar remainder for lengths not covered by a full vector, straight through
/// the byte-exact reference.
#[inline]
fn tail(out: &mut [u8], input: &[u8], coeff: u8, xor: bool) {
    if out.is_empty() {
        return;
    }
    if xor {
        scalar::mul_slice_xor(out, input, coeff);
    } else {
        scalar::mul_slice(out, input, coeff);
    }
}

// SSSE3 nibble-split (16-byte pshufb)

#[target_feature(enable = "ssse3")]
unsafe fn mul_ssse3<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let (lo, hi) = nibble_tables(coeff);
    let lo_v = _mm_loadu_si128(lo.as_ptr() as *const __m128i);
    let hi_v = _mm_loadu_si128(hi.as_ptr() as *const __m128i);
    let mask = _mm_set1_epi8(0x0f);
    let n = input.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);
        let lo_idx = _mm_and_si128(v, mask);
        let hi_idx = _mm_and_si128(_mm_srli_epi16::<4>(v), mask);
        let mut prod = _mm_xor_si128(
            _mm_shuffle_epi8(lo_v, lo_idx),
            _mm_shuffle_epi8(hi_v, hi_idx),
        );
        if XOR {
            let cur = _mm_loadu_si128(out.as_ptr().add(i) as *const __m128i);
            prod = _mm_xor_si128(prod, cur);
        }
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, prod);
        i += 16;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

// AVX2 nibble-split (32-byte vpshufb)

#[target_feature(enable = "avx2")]
unsafe fn mul_avx2<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let (lo, hi) = nibble_tables(coeff);
    let lo_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo.as_ptr() as *const __m128i));
    let hi_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi.as_ptr() as *const __m128i));
    let mask = _mm256_set1_epi8(0x0f);
    let n = input.len();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);
        let lo_idx = _mm256_and_si256(v, mask);
        let hi_idx = _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask);
        let mut prod = _mm256_xor_si256(
            _mm256_shuffle_epi8(lo_v, lo_idx),
            _mm256_shuffle_epi8(hi_v, hi_idx),
        );
        if XOR {
            let cur = _mm256_loadu_si256(out.as_ptr().add(i) as *const __m256i);
            prod = _mm256_xor_si256(prod, cur);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, prod);
        i += 32;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

// AVX-512BW nibble-split (64-byte vpshufb)

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn mul_avx512<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let (lo, hi) = nibble_tables(coeff);
    let lo_v = _mm512_broadcast_i32x4(_mm_loadu_si128(lo.as_ptr() as *const __m128i));
    let hi_v = _mm512_broadcast_i32x4(_mm_loadu_si128(hi.as_ptr() as *const __m128i));
    let mask = _mm512_set1_epi8(0x0f);
    let n = input.len();
    let mut i = 0usize;
    while i + 64 <= n {
        let v = _mm512_loadu_si512(input.as_ptr().add(i) as *const __m512i);
        let lo_idx = _mm512_and_si512(v, mask);
        let hi_idx = _mm512_and_si512(_mm512_srli_epi16::<4>(v), mask);
        let mut prod = _mm512_xor_si512(
            _mm512_shuffle_epi8(lo_v, lo_idx),
            _mm512_shuffle_epi8(hi_v, hi_idx),
        );
        if XOR {
            let cur = _mm512_loadu_si512(out.as_ptr().add(i) as *const __m512i);
            prod = _mm512_xor_si512(prod, cur);
        }
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, prod);
        i += 64;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

// GFNI affine (in-register 8x8 bit-matrix, no tables)

#[allow(dead_code)] // unused when a wider backend is pinned
#[target_feature(enable = "gfni,sse2")]
unsafe fn mul_gfni128<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let m = _mm_set1_epi64x(gfni_matrix(coeff));
    let n = input.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v = _mm_loadu_si128(input.as_ptr().add(i) as *const __m128i);
        let mut prod = _mm_gf2p8affine_epi64_epi8::<0>(v, m);
        if XOR {
            let cur = _mm_loadu_si128(out.as_ptr().add(i) as *const __m128i);
            prod = _mm_xor_si128(prod, cur);
        }
        _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, prod);
        i += 16;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

// AVX2, not just AVX: the body uses `_mm256_xor_si256` (256-bit integer XOR).
#[allow(dead_code)] // unused when a non-GFNI backend is pinned
#[target_feature(enable = "gfni,avx2")]
unsafe fn mul_gfni256<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let m = _mm256_set1_epi64x(gfni_matrix(coeff));
    let n = input.len();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);
        let mut prod = _mm256_gf2p8affine_epi64_epi8::<0>(v, m);
        if XOR {
            let cur = _mm256_loadu_si256(out.as_ptr().add(i) as *const __m256i);
            prod = _mm256_xor_si256(prod, cur);
        }
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, prod);
        i += 32;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

#[target_feature(enable = "gfni,avx512f")]
unsafe fn mul_gfni512<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let m = _mm512_set1_epi64(gfni_matrix(coeff));
    let n = input.len();
    let mut i = 0usize;
    while i + 64 <= n {
        let v = _mm512_loadu_si512(input.as_ptr().add(i) as *const __m512i);
        let mut prod = _mm512_gf2p8affine_epi64_epi8::<0>(v, m);
        if XOR {
            let cur = _mm512_loadu_si512(out.as_ptr().add(i) as *const __m512i);
            prod = _mm512_xor_si512(prod, cur);
        }
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, prod);
        i += 64;
    }
    tail(&mut out[i..], &input[i..], coeff, XOR);
}

// safe dispatch entry points

#[inline]
fn run<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());

    #[cfg(feature = "ssse3")]
    return unsafe { mul_ssse3::<XOR>(out, input, coeff) };
    #[cfg(feature = "avx2")]
    return unsafe { mul_avx2::<XOR>(out, input, coeff) };
    #[cfg(feature = "avx512")]
    return unsafe { mul_avx512::<XOR>(out, input, coeff) };
    #[cfg(feature = "gfni")]
    return unsafe { mul_gfni256::<XOR>(out, input, coeff) };

    #[cfg(not(any(
        feature = "ssse3",
        feature = "avx2",
        feature = "avx512",
        feature = "gfni",
    )))]
    {
        if is_x86_feature_detected!("gfni") {
            if is_x86_feature_detected!("avx512f") {
                unsafe { mul_gfni512::<XOR>(out, input, coeff) };
                return;
            }
            if is_x86_feature_detected!("avx2") {
                unsafe { mul_gfni256::<XOR>(out, input, coeff) };
                return;
            }
            unsafe { mul_gfni128::<XOR>(out, input, coeff) };
            return;
        }
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            unsafe { mul_avx512::<XOR>(out, input, coeff) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { mul_avx2::<XOR>(out, input, coeff) };
            return;
        }
        if is_x86_feature_detected!("ssse3") {
            unsafe { mul_ssse3::<XOR>(out, input, coeff) };
            return;
        }
        tail(out, input, coeff, XOR);
    }
}

/// `out[i] = coeff * input[i]` over GF(2^8).
#[inline]
pub fn mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    run::<false>(out, input, coeff);
}

/// `out[i] ^= coeff * input[i]` over GF(2^8).
#[inline]
pub fn mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    run::<true>(out, input, coeff);
}

/// Bench hook: run one kernel by index (0 ssse3, 1 avx2, 2 avx512, 3 gfni512),
/// bypassing dispatch; falls back to the scalar tail if the feature is absent.
#[doc(hidden)]
pub fn forced<const XOR: bool>(kind: u8, out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    unsafe {
        match kind {
            0 if is_x86_feature_detected!("ssse3") => mul_ssse3::<XOR>(out, input, coeff),
            1 if is_x86_feature_detected!("avx2") => mul_avx2::<XOR>(out, input, coeff),
            2 if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") => {
                mul_avx512::<XOR>(out, input, coeff)
            }
            3 if is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx512f") => {
                mul_gfni512::<XOR>(out, input, coeff)
            }
            _ => tail(out, input, coeff, XOR),
        }
    }
}

// differential test vs scalar

#[cfg(test)]
mod tests {
    use super::*;

    /// Input cycling through all 256 byte values, `len` bytes long.
    fn make_input(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i & 0xff) as u8).collect()
    }

    /// A distinct nonzero prior buffer, so the xor path can't pass by accident.
    fn make_prior(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 7 + 3) & 0xff) as u8).collect()
    }

    /// Lengths spanning every vector width plus unaligned / odd tails.
    const LENS: &[usize] = &[
        0, 1, 7, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000,
    ];

    /// Compare a kernel pair (assign / xor) against `scalar::*` byte-for-byte,
    /// for every coefficient 0..=255 and every length in `LENS`.
    fn differential<M, X>(kind: &str, mul: M, mul_xor: X)
    where
        M: Fn(&mut [u8], &[u8], u8),
        X: Fn(&mut [u8], &[u8], u8),
    {
        for coeff in 0u16..=255 {
            let coeff = coeff as u8;
            for &len in LENS {
                let input = make_input(len);

                let mut got = vec![0u8; len];
                mul(&mut got, &input, coeff);
                let mut want = vec![0u8; len];
                scalar::mul_slice(&mut want, &input, coeff);
                assert_eq!(got, want, "{kind} mul_slice coeff={coeff} len={len}");

                let mut got_x = make_prior(len);
                mul_xor(&mut got_x, &input, coeff);
                let mut want_x = make_prior(len);
                scalar::mul_slice_xor(&mut want_x, &input, coeff);
                assert_eq!(got_x, want_x, "{kind} mul_slice_xor coeff={coeff} len={len}");
            }
        }
    }

    #[test]
    fn matches_scalar() {
        // The public dispatch path (whatever this host actually selects).
        differential("dispatch", mul_slice, mul_slice_xor);

        // Plus each feature-gated kernel directly, when the CPU supports it, so
        // every path is exercised rather than only the top-preferred one.
        if is_x86_feature_detected!("ssse3") {
            differential(
                "ssse3",
                |o, i, c| unsafe { mul_ssse3::<false>(o, i, c) },
                |o, i, c| unsafe { mul_ssse3::<true>(o, i, c) },
            );
        }
        if is_x86_feature_detected!("avx2") {
            differential(
                "avx2",
                |o, i, c| unsafe { mul_avx2::<false>(o, i, c) },
                |o, i, c| unsafe { mul_avx2::<true>(o, i, c) },
            );
        }
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            differential(
                "avx512",
                |o, i, c| unsafe { mul_avx512::<false>(o, i, c) },
                |o, i, c| unsafe { mul_avx512::<true>(o, i, c) },
            );
        }
        if is_x86_feature_detected!("gfni") {
            differential(
                "gfni128",
                |o, i, c| unsafe { mul_gfni128::<false>(o, i, c) },
                |o, i, c| unsafe { mul_gfni128::<true>(o, i, c) },
            );
            if is_x86_feature_detected!("avx2") {
                differential(
                    "gfni256",
                    |o, i, c| unsafe { mul_gfni256::<false>(o, i, c) },
                    |o, i, c| unsafe { mul_gfni256::<true>(o, i, c) },
                );
            }
            if is_x86_feature_detected!("avx512f") {
                differential(
                    "gfni512",
                    |o, i, c| unsafe { mul_gfni512::<false>(o, i, c) },
                    |o, i, c| unsafe { mul_gfni512::<true>(o, i, c) },
                );
            }
        }
    }
}
