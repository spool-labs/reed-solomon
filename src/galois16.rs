//! GF((2^8)^2) tower field arithmetic, built on the GF(2^8) field in `galois`.
//!
//! An element is a pair `(high, low)` of GF(2^8) bytes, standing for
//! `high * y + low` in `GF(2^8)[y] / (y^2 + y + L)`, packed into a `u16` as
//! `(high << 8) | low`. The reduction constant `L` is chosen so the polynomial
//! is irreducible; a monic `y^2 + y + L` is irreducible over GF(2^8) exactly
//! when the trace of `L` is 1, and [`TOWER_L`] is the smallest such byte.
//!
//! Multiplying by a fixed coefficient is a 2x2 matrix of GF(2^8) constant
//! multiplies acting on `(high, low)`, which is what the SIMD kernels ride;
//! [`const_matrix`] produces those four constants. The scalar [`mul`] here is
//! the reference the kernels are differentially tested against.

use crate::galois;

/// Reduction constant for `y^2 + y + L`: the smallest byte with trace 1, so the
/// polynomial is irreducible over GF(2^8). Verified by the `smallest_valid_l`
/// test, which also confirms every smaller byte has trace 0.
pub const TOWER_L: u8 = 32;

/// The high plane byte of a symbol. Shards carry the two planes separately as
/// `[low N | high N]`, so this is also the wire split the kernels index by.
#[inline]
pub(crate) fn high(x: u16) -> u8 {
    (x >> 8) as u8
}

/// The low plane byte of a symbol.
#[inline]
pub(crate) fn low(x: u16) -> u8 {
    (x & 0xff) as u8
}

/// Rebuilds a symbol from its two plane bytes.
#[inline]
pub(crate) fn pack(high: u8, low: u8) -> u16 {
    ((high as u16) << 8) | low as u16
}

/// Field multiplication of two arbitrary elements (the reference product).
///
/// Schoolbook multiply in `GF(2^8)[y]` followed by reduction of `y^2` to
/// `y + L`. Roughly five GF(2^8) multiplies.
#[inline]
pub fn mul(a: u16, b: u16) -> u16 {
    let (a_high, a_low) = (high(a), low(a));
    let (b_high, b_low) = (high(b), low(b));

    // (a_high*y + a_low)(b_high*y + b_low)
    //   = product_2*y^2 + product_1*y + product_0
    let product_2 = galois::mul(a_high, b_high);
    let product_1 = galois::mul(a_high, b_low) ^ galois::mul(a_low, b_high);
    let product_0 = galois::mul(a_low, b_low);

    // Reduce y^2 = y + L: the y^2 term folds into both components.
    let result_high = product_1 ^ product_2;
    let result_low = product_0 ^ galois::mul(product_2, TOWER_L);
    pack(result_high, result_low)
}

/// The four GF(2^8) constants that multiply-by-`coefficient` becomes when it
/// acts on a variable `(high, low)` pair, returned as `[m00, m01, m10, m11]`:
///
/// ```text
/// out_high = m00 * a_high + m01 * a_low
/// out_low  = m10 * a_high + m11 * a_low
/// ```
///
/// This is the 2x2 block the fused SIMD kernels apply per (output, input) pair
/// over the split byte planes. Derived from [`mul`] with the coefficient held
/// constant, and checked against it in `const_matrix_matches_mul`.
// The NEON build applies tower constants through the Karatsuba triple instead,
// so only the portable and x86 lowerings reach for the dense block.
#[cfg(any(test, not(gf16_neon_enabled)))]
#[inline]
pub fn const_matrix(coefficient: u16) -> [u8; 4] {
    let (c_high, c_low) = (high(coefficient), low(coefficient));
    [
        c_high ^ c_low,                 // m00
        c_high,                         // m01
        galois::mul(c_high, TOWER_L),   // m10
        c_low,                          // m11
    ]
}

/// `base` raised to `power`, by square-and-multiply. Only the field-structure
/// test needs it; the coder reaches for [`mul`] and [`inv`] directly.
#[cfg(test)]
pub fn exp(base: u16, mut power: usize) -> u16 {
    let mut result = 1u16; // the ONE element is (0, 1)
    let mut factor = base;
    while power > 0 {
        if power & 1 == 1 {
            result = mul(result, factor);
        }
        factor = mul(factor, factor);
        power >>= 1;
    }
    result
}

/// Multiplicative inverse of a nonzero element.
///
/// Uses the tower structure rather than a full Fermat exponentiation: the
/// GF(2^8)-conjugate of `high*y + low` is `high*y + (high + low)` (the field's
/// nontrivial automorphism sends `y` to its other root `y + 1`), and the product
/// of an element with its conjugate is the norm, an element of GF(2^8). So the
/// inverse is `conjugate * norm^{-1}`, costing a handful of GF(2^8) operations.
/// Only the cold matrix-inversion path calls this. Panics on a zero input.
pub fn inv(a: u16) -> u16 {
    assert!(a != 0, "GF(2^16) inverse of zero");
    let (a_high, a_low) = (high(a), low(a));
    let conjugate = pack(a_high, a_high ^ a_low);

    // norm = a * conjugate lands in GF(2^8), so its high byte is zero.
    let norm = mul(a, conjugate);
    debug_assert_eq!(high(norm), 0, "norm must lie in GF(2^8)");
    let norm_inv = galois::div(1, low(norm));

    // Scale the conjugate by the GF(2^8) scalar norm_inv, component-wise.
    pack(
        galois::mul(high(conjugate), norm_inv),
        galois::mul(low(conjugate), norm_inv),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Trace GF(2^8) -> GF(2): Tr(x) = x + x^2 + x^4 + ... + x^128.
    fn gf8_trace(mut x: u8) -> u8 {
        let mut accumulator = 0u8;
        for _ in 0..8 {
            accumulator ^= x;
            x = galois::mul(x, x);
        }
        accumulator
    }

    // TOWER_L is the smallest byte making y^2 + y + L irreducible (trace 1), and
    // every smaller byte is reducible (trace 0).
    #[test]
    fn smallest_valid_l() {
        for candidate in 0..TOWER_L {
            assert_eq!(gf8_trace(candidate), 0, "L={candidate} should have trace 0");
        }
        assert_eq!(gf8_trace(TOWER_L), 1);
        // Exactly half the bytes qualify.
        let valid = (0u16..256).filter(|&c| gf8_trace(c as u8) == 1).count();
        assert_eq!(valid, 128);
    }

    // The 2x2 constant matrix the kernels apply equals the reference product,
    // for every coefficient. Exhaustive over structured coefficients, sampled
    // over the rest, with a structured plus pseudo-random set of data values.
    #[test]
    fn const_matrix_matches_mul() {
        fn apply(matrix: &[u8; 4], a: u16) -> u16 {
            let (a_high, a_low) = ((a >> 8) as u8, (a & 0xff) as u8);
            let out_high = galois::mul(matrix[0], a_high) ^ galois::mul(matrix[1], a_low);
            let out_low = galois::mul(matrix[2], a_high) ^ galois::mul(matrix[3], a_low);
            ((out_high as u16) << 8) | out_low as u16
        }

        let structured = [0u16, 1, 0x0100, TOWER_L as u16, 0x00ff, 0xff00, 0xffff, 0x1234];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u16
        };

        for coefficient in 0u32..65536 {
            let coefficient = coefficient as u16;
            let matrix = const_matrix(coefficient);
            for &a in &structured {
                assert_eq!(apply(&matrix, a), mul(coefficient, a), "c={coefficient} a={a}");
            }
            for _ in 0..4 {
                let a = next();
                assert_eq!(apply(&matrix, a), mul(coefficient, a), "c={coefficient} a={a}");
            }
        }

        // Fully exhaust the data side for the structured coefficients.
        for &coefficient in &[0u16, 1, 0x0100, TOWER_L as u16, 0xffff] {
            let matrix = const_matrix(coefficient);
            for a in 0u32..65536 {
                let a = a as u16;
                assert_eq!(apply(&matrix, a), mul(coefficient, a), "exhaustive c={coefficient} a={a}");
            }
        }

        // The two structured cases the kernel design leans on.
        assert_eq!(const_matrix(1), [1, 0, 0, 1], "multiply-by-one is the identity block");
        assert_eq!(const_matrix(0x0100), [1, 1, TOWER_L, 0], "multiply-by-y");
    }

    // GF((2^8)^2) is a field of order 65536: a primitive element of full order
    // exists, and every nonzero element has an inverse.
    #[test]
    fn field_structure() {
        // 65535 = 3 * 5 * 17 * 257; full order iff no proper-divisor power is 1.
        let primes = [3usize, 5, 17, 257];
        let has_full_order = |x: u16| x != 0 && primes.iter().all(|&p| exp(x, 65535 / p) != 1);
        let generator = (2u32..65536)
            .map(|c| c as u16)
            .find(|&c| has_full_order(c))
            .expect("a primitive element exists");
        assert_eq!(exp(generator, 65535), 1);

        for a in 1u32..65536 {
            let a = a as u16;
            assert_eq!(mul(a, inv(a)), 1, "a={a} inverse wrong");
        }
    }

    // Multiplication is associative and distributes over addition. Commutativity
    // is structural (the product is symmetric in a and b).
    #[test]
    fn field_axioms() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u16
        };
        for _ in 0..200_000 {
            let (a, b, c) = (next(), next(), next());
            assert_eq!(mul(mul(a, b), c), mul(a, mul(b, c)), "assoc a={a} b={b} c={c}");
            assert_eq!(mul(a, b ^ c), mul(a, b) ^ mul(a, c), "distrib a={a} b={b} c={c}");
            assert_eq!(mul(a, b), mul(b, a), "commut a={a} b={b}");
        }
    }
}
