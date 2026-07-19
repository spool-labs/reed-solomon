//! Emits the `fft_enabled` cfg when this build compiles the FFT encode
//! executors, so the call sites gate on one alias instead of repeating the
//! per-arch predicate. Keep this in exact agreement with the backend module
//! gates in `src/gf/mod.rs` (`fft_neon` / `fft_wasm` / `fft_x86`):
//!
//!   aarch64                         and not scalar
//!   wasm32   with simd128           and not scalar
//!   x86_64   without an SSE/AVX pin  and not scalar   (gfni does not pin)

fn main() {
    println!("cargo:rustc-check-cfg=cfg(fft_enabled)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_simd128 = target_features.split(',').any(|f| f == "simd128");

    let feature = |name: &str| std::env::var_os(format!("CARGO_FEATURE_{name}")).is_some();
    let x86_pinned = feature("SSSE3") || feature("AVX2") || feature("AVX512");

    let fft_enabled = !feature("SCALAR")
        && match arch.as_str() {
            "aarch64" => true,
            "wasm32" => has_simd128,
            "x86_64" => !x86_pinned,
            _ => false,
        };

    if fft_enabled {
        println!("cargo:rustc-cfg=fft_enabled");
    }
}
