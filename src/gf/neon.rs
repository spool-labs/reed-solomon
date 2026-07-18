//! aarch64 NEON GF(2^8) slice-multiply kernel.

use crate::galois;
use core::arch::aarch64::*;

/// Build the two 16-entry nibble tables for `coeff`
#[inline]
fn build_tables(coeff: u8) -> ([u8; 16], [u8; 16]) {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    let mut x = 0usize;
    while x < 16 {
        lo[x] = galois::mul(coeff, x as u8);
        hi[x] = galois::mul(coeff, (x as u8) << 4);
        x += 1;
    }
    (lo, hi)
}

/// SIMD core. `XOR` selects `out ^= c*input` (true) vs `out = c*input` (false)
///
/// # Safety
/// Caller must ensure the `neon` feature is available on this CPU. `out` and `input` must have equal length (asserted by the public wrappers).
#[target_feature(enable = "neon")]
unsafe fn mul_slice_core<const XOR: bool>(out: &mut [u8], input: &[u8], coeff: u8) {
    let (lo, hi) = build_tables(coeff);
    let lo_v = vld1q_u8(lo.as_ptr());
    let hi_v = vld1q_u8(hi.as_ptr());
    let low_mask = vdupq_n_u8(0x0f);

    let len = out.len();
    let mut i = 0usize;

    while i + 16 <= len {
        let v = vld1q_u8(input.as_ptr().add(i));
        let lo_idx = vandq_u8(v, low_mask);
        let hi_idx = vshrq_n_u8::<4>(v);
        let prod = veorq_u8(vqtbl1q_u8(lo_v, lo_idx), vqtbl1q_u8(hi_v, hi_idx));
        let dst = out.as_mut_ptr().add(i);
        if XOR {
            let cur = vld1q_u8(dst);
            vst1q_u8(dst, veorq_u8(cur, prod));
        } else {
            vst1q_u8(dst, prod);
        }
        i += 16;
    }

    // Scalar tail for the remaining < 16 bytes.
    let row = &galois::MUL_TABLE[coeff as usize];
    while i < len {
        let p = row[input[i] as usize];
        if XOR {
            out[i] ^= p;
        } else {
            out[i] = p;
        }
        i += 1;
    }
}

/// `out[i] = coeff * input[i]` over GF(2^8)
#[inline]
pub fn mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    // SAFETY: dispatched only under `is_aarch64_feature_detected!("neon")`.
    unsafe { mul_slice_core::<false>(out, input, coeff) }
}

/// `out[i] ^= coeff * input[i]` over GF(2^8)
#[inline]
pub fn mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    // SAFETY: dispatched only under `is_aarch64_feature_detected!("neon")`.
    unsafe { mul_slice_core::<true>(out, input, coeff) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf::scalar;

    // All 256 byte values plus an odd tail to exercise the scalar remainder path.
    fn sample_input() -> Vec<u8> {
        let mut v: Vec<u8> = (0..=255u8).collect(); // 256 bytes (16-aligned)
        v.extend_from_slice(&[7, 200, 1, 0, 255, 42, 128]); // +7 -> len 263, odd
        v
    }

    #[test]
    fn neon_matches_scalar_mul_slice_all_coeffs() {
        let input = sample_input();
        for c in 0..=255u8 {
            let mut got = vec![0u8; input.len()];
            let mut want = vec![0u8; input.len()];
            mul_slice(&mut got, &input, c);
            scalar::mul_slice(&mut want, &input, c);
            assert_eq!(got, want, "mul_slice mismatch for coeff {c}");
        }
    }

    #[test]
    fn neon_matches_scalar_mul_slice_xor_all_coeffs() {
        let input = sample_input();
        // Non-trivial pre-existing accumulator so XOR-into is really tested.
        let seed: Vec<u8> = input.iter().map(|b| b.wrapping_mul(3).wrapping_add(1)).collect();
        for c in 0..=255u8 {
            let mut got = seed.clone();
            let mut want = seed.clone();
            mul_slice_xor(&mut got, &input, c);
            scalar::mul_slice_xor(&mut want, &input, c);
            assert_eq!(got, want, "mul_slice_xor mismatch for coeff {c}");
        }
    }

    #[test]
    fn neon_matches_scalar_various_lengths() {
        // Every length from 0..40 crosses the 16-byte boundary in every phase.
        for len in 0..40usize {
            let input: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(97).wrapping_add(13)).collect();
            for c in [0u8, 1, 2, 3, 27, 128, 200, 255] {
                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                mul_slice(&mut got, &input, c);
                scalar::mul_slice(&mut want, &input, c);
                assert_eq!(got, want, "mul_slice len {len} coeff {c}");
            }
        }
    }
}
