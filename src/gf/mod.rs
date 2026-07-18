//! GF(2^8) slice-multiply dispatch — routes field arithmetic to the best
//! available SIMD kernel for the target architecture at runtime.

pub mod scalar;

// Affine-matrix derivation for the x86 `gf2p8affineqb` kernels. Pure scalar
// arithmetic, so it is compiled and tested on any host, not just x86_64.
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) mod gfni;

#[cfg(target_arch = "wasm32")]
pub mod wasm128;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod neon_fused;

#[cfg(target_arch = "x86_64")]
pub mod x86;

// Fused multi-output GFNI kernel: one streaming pass over the data per output
// tile (`vgf2p8affineqb` + `vpternlogd`). Used by `ReedSolomon::encode_fused`
// on GFNI + AVX-512 hosts.
#[cfg(target_arch = "x86_64")]
pub mod x86_fused;

/// `out[i] = coeff * input[i]` over GF(2^8). Panics if lengths differ.
#[inline]
pub fn mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(feature = "scalar")]
    scalar::mul_slice(out, input, coeff);
    #[cfg(not(feature = "scalar"))]
    dispatch_mul_slice(out, input, coeff);
}

/// `out[i] ^= coeff * input[i]` over GF(2^8). Panics if lengths differ.
#[inline]
pub fn mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(feature = "scalar")]
    scalar::mul_slice_xor(out, input, coeff);
    #[cfg(not(feature = "scalar"))]
    dispatch_mul_slice_xor(out, input, coeff);
}

// --- aarch64 --------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(feature = "neon")]
    return neon::mul_slice(out, input, coeff);
    #[cfg(not(feature = "neon"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        neon::mul_slice(out, input, coeff);
    } else {
        scalar::mul_slice(out, input, coeff);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(feature = "neon")]
    return neon::mul_slice_xor(out, input, coeff);
    #[cfg(not(feature = "neon"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        neon::mul_slice_xor(out, input, coeff);
    } else {
        scalar::mul_slice_xor(out, input, coeff);
    }
}

// --- x86_64 ---------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    if is_x86_feature_detected!("gfni")
        || is_x86_feature_detected!("avx512f")
        || is_x86_feature_detected!("avx2")
        || is_x86_feature_detected!("ssse3")
    {
        x86::mul_slice(out, input, coeff);
    } else {
        scalar::mul_slice(out, input, coeff);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    if is_x86_feature_detected!("gfni")
        || is_x86_feature_detected!("avx512f")
        || is_x86_feature_detected!("avx2")
        || is_x86_feature_detected!("ssse3")
    {
        x86::mul_slice_xor(out, input, coeff);
    } else {
        scalar::mul_slice_xor(out, input, coeff);
    }
}

// --- wasm32 ---------------------------------------------------------------
#[cfg(target_arch = "wasm32")]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(target_feature = "simd128")]
    {
        wasm128::mul_slice(out, input, coeff);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        scalar::mul_slice(out, input, coeff);
    }
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    #[cfg(target_feature = "simd128")]
    {
        wasm128::mul_slice_xor(out, input, coeff);
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        scalar::mul_slice_xor(out, input, coeff);
    }
}

// --- every other arch -----------------------------------------------------
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    scalar::mul_slice(out, input, coeff);
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    scalar::mul_slice_xor(out, input, coeff);
}
