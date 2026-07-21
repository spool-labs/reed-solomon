//! The coefficient field abstraction shared by the matrix layer.
//!
//! `Matrix` and the systematic-generator construction are identical arithmetic
//! whether the code runs over GF(2^8) or the GF((2^8)^2) tower field; only the
//! element type and its multiply/inverse differ. `FieldElement` captures exactly
//! that difference, implemented directly on `u8` and `u16` so no witness value
//! has to be threaded through every matrix call.

/// One element of a binary extension field, with the operations the matrix
/// layer needs: addition (xor), multiplication, inversion, and the distinct
/// evaluation point used per Vandermonde row.
pub trait FieldElement: Copy + PartialEq + Eq + core::fmt::Debug {
    /// The additive identity, and the empty parity accumulator.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;

    /// Field addition, which is xor in every characteristic-2 field.
    fn gf_add(self, rhs: Self) -> Self;

    /// Field multiplication.
    fn gf_mul(self, rhs: Self) -> Self;

    /// Multiplicative inverse. Only ever called on nonzero elements (the
    /// Gaussian elimination checks for a zero pivot first), so a zero input is
    /// a caller bug and may panic.
    fn gf_inv(self) -> Self;

    /// The distinct field element assigned to Vandermonde row `index`. Rows must
    /// map to distinct elements for the top block to stay invertible; the
    /// natural choice is the field element numbered `index`.
    fn evaluation_point(index: usize) -> Self;
}

impl FieldElement for u8 {
    const ZERO: u8 = 0;
    const ONE: u8 = 1;

    #[inline]
    fn gf_add(self, rhs: u8) -> u8 {
        self ^ rhs
    }

    #[inline]
    fn gf_mul(self, rhs: u8) -> u8 {
        crate::galois::mul(self, rhs)
    }

    #[inline]
    fn gf_inv(self) -> u8 {
        // div(1, x) is the inverse; div panics on a zero divisor, matching the
        // nonzero-only contract above.
        crate::galois::div(1, self)
    }

    #[inline]
    fn evaluation_point(index: usize) -> u8 {
        // Total shard count is capped at 256, so every row index fits a byte and
        // the bytes stay distinct.
        index as u8
    }
}

impl FieldElement for u16 {
    const ZERO: u16 = 0;
    const ONE: u16 = 1;

    #[inline]
    fn gf_add(self, rhs: u16) -> u16 {
        self ^ rhs
    }

    #[inline]
    fn gf_mul(self, rhs: u16) -> u16 {
        crate::galois16::mul(self, rhs)
    }

    #[inline]
    fn gf_inv(self) -> u16 {
        crate::galois16::inv(self)
    }

    #[inline]
    fn evaluation_point(index: usize) -> u16 {
        // Total shard count is capped at 65536, so every row index fits a u16.
        index as u16
    }
}
