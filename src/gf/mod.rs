//! GF(2^8) slice-multiply dispatch: routes field arithmetic to the best
//! available SIMD kernel for the target architecture at runtime.

pub mod scalar;
pub mod tables;

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

// Fused Karatsuba kernel for the GF((2^8)^2) tower coder (three GF(2^8)
// products per coefficient instead of the dense four).
#[cfg(gf16_neon_enabled)]
pub mod tower_neon;
#[cfg(gf16_x86_enabled)]
pub(crate) mod tower_x86;

// Staged-program executors for the GF((2^8)^2) FFT encode: op-major arena
// streaming, one per architecture.
#[cfg(gf16_neon_enabled)]
pub(crate) mod fft16_neon;
#[cfg(gf16_x86_enabled)]
pub(crate) mod fft16_x86;

// Compiled FFT encode executors for the production shapes; compiled exactly
// where ReedSolomon::encode can route to them.
#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
pub(crate) mod fft_neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar")))]
pub(crate) mod fft_wasm;
#[cfg(all(
    target_arch = "x86_64",
    not(any(feature = "scalar", feature = "ssse3", feature = "avx2", feature = "avx512"))
))]
pub(crate) mod fft_x86;

// The FFT executor for this build, aliased so encode/encode_route name one
// backend instead of branching per arch. Exactly one alias exists wherever
// build.rs emits `fft_enabled`.
#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
pub(crate) use fft_neon as fft_active;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar")))]
pub(crate) use fft_wasm as fft_active;
#[cfg(all(
    target_arch = "x86_64",
    not(any(feature = "scalar", feature = "ssse3", feature = "avx2", feature = "avx512"))
))]
pub(crate) use fft_x86 as fft_active;

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

// aarch64
#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
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

#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
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

// x86_64
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    // A pinned backend already knows its kernel, so skip runtime detection.
    #[cfg(any(feature = "ssse3", feature = "avx2", feature = "avx512", feature = "gfni"))]
    return x86::mul_slice(out, input, coeff);
    #[cfg(not(any(feature = "ssse3", feature = "avx2", feature = "avx512", feature = "gfni")))]
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

#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    // A pinned backend already knows its kernel, so skip runtime detection.
    #[cfg(any(feature = "ssse3", feature = "avx2", feature = "avx512", feature = "gfni"))]
    return x86::mul_slice_xor(out, input, coeff);
    #[cfg(not(any(feature = "ssse3", feature = "avx2", feature = "avx512", feature = "gfni")))]
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

// wasm32
#[cfg(all(target_arch = "wasm32", not(feature = "scalar")))]
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

#[cfg(all(target_arch = "wasm32", not(feature = "scalar")))]
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

// every other arch
#[cfg(not(any(
    feature = "scalar",
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
#[inline]
fn dispatch_mul_slice(out: &mut [u8], input: &[u8], coeff: u8) {
    scalar::mul_slice(out, input, coeff);
}

#[cfg(not(any(
    feature = "scalar",
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
#[inline]
fn dispatch_mul_slice_xor(out: &mut [u8], input: &[u8], coeff: u8) {
    scalar::mul_slice_xor(out, input, coeff);
}
