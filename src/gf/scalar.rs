//! Scalar GF(2^8) slice-multiply kernel: the reference implementation.

use crate::galois::MUL_TABLE;

/// `out[i] = coeff * input[i]` over GF(2^8).
#[inline]
pub fn mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    let row = &MUL_TABLE[coeff as usize];
    for (o, &i) in out.iter_mut().zip(input.iter()) {
        *o = row[i as usize];
    }
}

/// `out[i] ^= coeff * input[i]` over GF(2^8).
#[inline]
pub fn mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    assert_eq!(input.len(), out.len());
    let row = &MUL_TABLE[coeff as usize];
    for (o, &i) in out.iter_mut().zip(input.iter()) {
        *o ^= row[i as usize];
    }
}
