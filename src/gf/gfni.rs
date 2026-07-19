//! GFNI affine-matrix derivation for "multiply by `c`" in GF(2^8), used by the
//! x86 `gf2p8affineqb` kernels. Pure scalar arithmetic, verified on any host.

use crate::galois;

/// Affine qwords for every coefficient, built at compile time so kernels can
/// index them with literal constants
///
/// Only the x86 FFT executor consumes it, so it exists exactly where that
/// executor does (plus its own test).
#[cfg(all(target_arch = "x86_64", any(test, fft_enabled)))]
pub(crate) static AFFINE: [i64; 256] = gen_affine_table();

#[cfg(all(target_arch = "x86_64", any(test, fft_enabled)))]
const fn gen_affine_table() -> [i64; 256] {
    let mul = galois::gen_mul_table();
    let mut table = [0i64; 256];
    let mut c = 0usize;
    while c < 256 {
        let mut q: u64 = 0;
        let mut r = 0u32;
        while r < 8 {
            let mut byte = 0u8;
            let mut j = 0u32;
            while j < 8 {
                let prod = mul[c][(1usize << j) & 0xff];
                if (prod >> (7 - r)) & 1 == 1 {
                    byte |= 1u8 << j;
                }
                j += 1;
            }
            q |= (byte as u64) << (8 * r);
            r += 1;
        }
        table[c] = q as i64;
        c += 1;
    }
    table
}

/// 8x8 GF(2) bit-matrix for "multiply by `c`", packed as the qword operand
/// `gf2p8affineqb` expects.
///
/// The instruction computes `out.bit[i] = parity(v AND A.byte[7 - i])` (imm8 = 0);
/// multiply-by-`c` being GF(2)-linear gives `A.byte[r].bit[j] = bit_(7-r)(c * (1<<j))`.
#[inline]
pub(crate) fn gfni_matrix(c: u8) -> i64 {
    let mut q: u64 = 0;
    let mut r = 0u32;
    while r < 8 {
        let mut byte = 0u8;
        let mut j = 0u32;
        while j < 8 {
            let prod = galois::mul(c, 1u8 << j);
            if (prod >> (7 - r)) & 1 == 1 {
                byte |= 1u8 << j;
            }
            j += 1;
        }
        q |= (byte as u64) << (8 * r);
        r += 1;
    }
    q as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar model of `gf2p8affineqb` with imm8 = 0: `dst.bit[i] = parity(v AND A.byte[7 - i])`.
    fn affine_model(matrix: u64, v: u8) -> u8 {
        let a = matrix.to_le_bytes(); // a[r] == A.byte[r]
        let mut out = 0u8;
        for i in 0..8 {
            if (v & a[7 - i]).count_ones() & 1 == 1 {
                out |= 1 << i;
            }
        }
        out
    }

    /// Fixed points: `c == 1` yields the identity matrix, `c == 0` the zero matrix.
    #[test]
    fn identity_and_zero() {
        assert_eq!(gfni_matrix(1) as u64, 0x0102_0408_1020_4080u64);
        assert_eq!(gfni_matrix(0) as u64, 0);
    }

    // the compile-time table agrees with the runtime derivation
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn affine_table_matches() {
        for c in 0..=255u8 {
            assert_eq!(AFFINE[c as usize], gfni_matrix(c), "c={c}");
        }
    }

    /// For every (coeff, byte), the modeled affine transform of `gfni_matrix(c)` equals `c * v`.
    #[test]
    fn matrix_models_field_multiply() {
        for c in 0..=255u8 {
            let m = gfni_matrix(c) as u64;
            for v in 0..=255u8 {
                assert_eq!(affine_model(m, v), galois::mul(c, v), "c={c} v={v}");
            }
        }
    }
}
