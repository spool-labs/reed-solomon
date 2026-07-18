//! Pure-Rust Reed-Solomon erasure coder over GF(2^8) with a runtime-dispatched SIMD kernel layer.
//!
//! # Example
//! ```
//! use tape_reed_solomon::ReedSolomon;
//! let rs = ReedSolomon::new(4, 2).unwrap();
//! let mut shards: Vec<Vec<u8>> = vec![vec![1u8; 8], vec![2; 8], vec![3; 8], vec![4; 8],
//!                                     vec![0; 8], vec![0; 8]];
//! // encode over `Vec<&mut [u8]>`
//! let mut slices: Vec<&mut [u8]> = shards.iter_mut().map(|s| s.as_mut_slice()).collect();
//! rs.encode(&mut slices).unwrap();
//! ```

// The scalar pin overrides every SIMD kernel, so combining it with another
// backend feature can only mean a build mistake.
#[cfg(all(
    feature = "scalar",
    any(
        feature = "ssse3",
        feature = "avx2",
        feature = "avx512",
        feature = "gfni",
        feature = "neon"
    )
))]
compile_error!("feature \"scalar\" cannot be combined with another backend feature");

// The x86 pins each hardwire one kernel, so cargo feature unification across
// two of them would silently run the narrowest. Refuse to build instead.
// Combining an x86 pin with "neon" stays legal for multi-target workspaces.
#[cfg(any(
    all(feature = "ssse3", any(feature = "avx2", feature = "avx512", feature = "gfni")),
    all(feature = "avx2", any(feature = "avx512", feature = "gfni")),
    all(feature = "avx512", feature = "gfni"),
))]
compile_error!("enable at most one x86 backend feature (ssse3, avx2, avx512, gfni)");

pub mod galois;
pub mod gf;
mod errors;
mod matrix;
mod reedsolomon;

pub use crate::errors::Error;
pub use crate::reedsolomon::{PreparedDecoder, ReconstructShard, ReedSolomon};
