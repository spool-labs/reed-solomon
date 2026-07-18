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

pub mod galois;
pub mod gf;
mod errors;
mod matrix;
mod reedsolomon;

pub use crate::errors::Error;
pub use crate::reedsolomon::{ReconstructShard, ReedSolomon};
