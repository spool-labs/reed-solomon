//! Coefficient rows prepared once for the fused multi-output kernels
//!
//! A RowTables owns a coefficient matrix (one row per output, one column per
//! input) together with every per-architecture kernel table derived from it,
//! so hot paths apply the matrix without rebuilding anything per call.

#[cfg(any(
    all(target_arch = "aarch64", not(feature = "scalar")),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
))]
use crate::galois::MUL_TABLE;

#[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
use super::neon_fused;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar")))]
use super::wasm128;
#[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
use super::x86_fused;
#[cfg(any(
    feature = "scalar",
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
use super::{mul_slice, mul_slice_xor};

/// Per-coefficient lo and hi nibble tables, 16 bytes each, built at compile
/// time. Shared by every executor that indexes coefficients by nibble: the
/// GF(2^8) FFT cores and the GF((2^8)^2) tower core.
#[cfg(any(
    all(target_arch = "aarch64", not(feature = "scalar")),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
))]
pub(crate) static NIBBLE_PAIRS: [[u8; 32]; 256] = gen_nibble_pairs();

#[cfg(any(
    all(target_arch = "aarch64", not(feature = "scalar")),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
))]
const fn gen_nibble_pairs() -> [[u8; 32]; 256] {
    let mul = crate::galois::gen_mul_table();
    let mut pairs = [[0u8; 32]; 256];
    let mut c = 0usize;
    while c < 256 {
        let mut x = 0usize;
        while x < 16 {
            pairs[c][x] = mul[c][x];
            pairs[c][16 + x] = mul[c][x << 4];
            x += 1;
        }
        c += 1;
    }
    pairs
}

/// The 32-byte nibble-pair table for one GF(2^8) coefficient: the 16 low-nibble
/// products then the 16 high-nibble products.
///
/// This layout is the contract between every table builder and the NEON, wasm,
/// and tower kernels that index it, so it lives in one place. Hoisting the
/// `MUL_TABLE` row also turns 32 double-indexed lookups into one row lookup.
#[cfg(any(
    all(target_arch = "aarch64", not(feature = "scalar")),
    all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
))]
pub(crate) fn nibble_pair(coefficient: u8) -> [u8; 32] {
    let product_row = &MUL_TABLE[coefficient as usize];
    let mut pair = [0u8; 32];
    pair[..16].copy_from_slice(&product_row[..16]);
    for x in 0..16 {
        pair[16 + x] = product_row[x << 4];
    }
    pair
}

/// Coefficient rows plus the prepared kernel tables needed to apply them
#[derive(Clone, Debug)]
pub struct RowTables {
    input_count: usize,
    rows: Vec<Vec<u8>>,

    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    affine: x86_fused::AffineMatrices,
    #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
    nibbles: x86_fused::NibbleTables,

    #[cfg(any(
        all(target_arch = "aarch64", not(feature = "scalar")),
        all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
    ))]
    nibble_pairs: Vec<u8>,
}

impl RowTables {
    /// Build the kernel tables for the given coefficient rows
    pub fn new(rows: Vec<Vec<u8>>, input_count: usize) -> RowTables {
        for row in &rows {
            debug_assert_eq!(row.len(), input_count);
        }

        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        let (affine, nibbles) = {
            let output_count = rows.len();
            let mut flat = vec![0u8; output_count * input_count];
            for (output, row) in rows.iter().enumerate() {
                flat[output * input_count..(output + 1) * input_count].copy_from_slice(row);
            }
            (
                x86_fused::AffineMatrices::new(&flat, output_count, input_count),
                x86_fused::NibbleTables::new(&flat, output_count, input_count),
            )
        };

        // One 16-byte lo table and one 16-byte hi table per coefficient, laid
        // out exactly as the NEON and wasm kernels index them.
        #[cfg(any(
            all(target_arch = "aarch64", not(feature = "scalar")),
            all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
        ))]
        let nibble_pairs = {
            let mut tables = Vec::with_capacity(rows.len() * input_count * 32);
            for row in &rows {
                for &coefficient in row {
                    tables.extend_from_slice(&nibble_pair(coefficient));
                }
            }
            tables
        };

        RowTables {
            input_count,
            rows,
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            affine,
            #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
            nibbles,
            #[cfg(any(
                all(target_arch = "aarch64", not(feature = "scalar")),
                all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar"))
            ))]
            nibble_pairs,
        }
    }

    /// The raw coefficient rows
    pub fn rows(&self) -> &[Vec<u8>] {
        &self.rows
    }

    /// Multiply the inputs by the prepared rows, overwriting the outputs
    ///
    /// Routes through the fused kernel for this architecture, or the scalar
    /// kernel when the scalar feature pins it. Inputs and outputs must all
    /// share one length and the input count must match the prepared rows.
    pub fn apply<In: AsRef<[u8]>, Out: AsMut<[u8]>>(&self, inputs: &[In], outputs: &mut [Out]) {
        if outputs.is_empty() {
            return;
        }
        debug_assert_eq!(inputs.len(), self.input_count);

        #[cfg(feature = "scalar")]
        return self.apply_fallback(inputs, outputs);

        #[cfg(all(target_arch = "x86_64", not(feature = "scalar")))]
        return x86_fused::encode_with_tables(
            &self.affine,
            &self.nibbles,
            &self.rows,
            inputs,
            outputs,
        );
        #[cfg(all(target_arch = "aarch64", not(feature = "scalar")))]
        return neon_fused::encode_fused(&self.nibble_pairs, &self.rows, inputs, outputs);
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(feature = "scalar")))]
        return wasm128::encode_fused(&self.nibble_pairs, &self.rows, inputs, outputs);

        #[cfg(not(any(
            feature = "scalar",
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        self.apply_fallback(inputs, outputs)
    }

    /// Per-coefficient path through the dispatched slice kernels
    #[cfg(any(
        feature = "scalar",
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))
    ))]
    fn apply_fallback<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
        &self,
        inputs: &[In],
        outputs: &mut [Out],
    ) {
        for (output_index, output) in outputs.iter_mut().enumerate() {
            let output = output.as_mut();
            for (input_index, input) in inputs.iter().enumerate() {
                let coefficient = self.rows[output_index][input_index];
                if input_index == 0 {
                    mul_slice(output, input.as_ref(), coefficient);
                } else {
                    mul_slice_xor(output, input.as_ref(), coefficient);
                }
            }
        }
    }
}
