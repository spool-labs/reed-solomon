//! Emits the cfg aliases for the compiled encode executors, so the call sites
//! gate on one name instead of repeating the per-arch predicate. Keep these in
//! exact agreement with the backend module gates in `src/gf/mod.rs`.
//!
//! `fft_enabled`, the GF(2^8) FFT executors (`fft_neon` / `fft_wasm` / `fft_x86`):
//!
//!   aarch64                         and not scalar
//!   wasm32   with simd128           and not scalar
//!   x86_64   without an SSE/AVX pin  and not scalar   (gfni does not pin)
//!
//! `gf16_neon_enabled` / `gf16_x86_enabled`, the GF((2^8)^2) tower kernels
//! (`tower_neon` / `fft16_neon` / `fft16_x86`). There is no wasm tower executor,
//! which is why these are siblings of `fft_enabled` rather than uses of it:
//!
//!   aarch64                         and not scalar
//!   x86_64   without an SSE/AVX pin  and not scalar
//!
//! `gf16_simd_enabled` is either of the two, for the parts of the program IR
//! that only a SIMD executor reads.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(fft_enabled)");
    println!("cargo:rustc-check-cfg=cfg(gf16_neon_enabled)");
    println!("cargo:rustc-check-cfg=cfg(gf16_x86_enabled)");
    println!("cargo:rustc-check-cfg=cfg(gf16_simd_enabled)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_simd128 = target_features.split(',').any(|f| f == "simd128");

    let feature = |name: &str| std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some();
    let x86_pinned = feature("SSSE3") || feature("AVX2") || feature("AVX512");
    let scalar = feature("SCALAR");

    let fft_enabled = !scalar
        && match arch.as_str() {
            "aarch64" => true,
            "wasm32" => has_simd128,
            "x86_64" => !x86_pinned,
            _ => false,
        };

    let gf16_neon_enabled = !scalar && arch == "aarch64";
    let gf16_x86_enabled = !scalar && arch == "x86_64" && !x86_pinned;

    for (emit, name) in [
        (fft_enabled, "fft_enabled"),
        (gf16_neon_enabled, "gf16_neon_enabled"),
        (gf16_x86_enabled, "gf16_x86_enabled"),
        (gf16_neon_enabled || gf16_x86_enabled, "gf16_simd_enabled"),
    ] {
        if emit {
            println!("cargo:rustc-cfg={name}");
        }
    }
}
