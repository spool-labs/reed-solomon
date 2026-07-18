//! Reed-Solomon erasure coder over GF(2^8).
//!
//! ```ignore
//! let rs = ReedSolomon::new(data_shards, parity_shards)?;
//! rs.encode(&mut slices)?;      // slices: Vec<&mut [u8]>
//! rs.reconstruct(&mut shards)?; // shards: Vec<(&mut [u8], bool)>
//! ```

use crate::errors::Error;
use crate::gf;
use crate::matrix::Matrix;

/// A container that may hold a shard, used during reconstruction over GF(2^8)
pub trait ReconstructShard {
    /// The size of the shard data; `None` if absent.
    fn len(&self) -> Option<usize>;

    /// Mutable view of the shard data; `None` if uninitialized.
    fn get(&mut self) -> Option<&mut [u8]>;

    /// Mutable view, initializing to `len` if it was absent. On success returns
    /// `Ok(slice)` for an already-present shard, or `Err(Ok(slice))` for one we
    /// just initialized (i.e. a missing shard to be reconstructed). `Err(Err(_))`
    /// signals a hard error (e.g. wrong size).
    fn get_or_initialize(
        &mut self,
        len: usize,
    ) -> Result<&mut [u8], Result<&mut [u8], Error>>;
}

impl<T: AsRef<[u8]> + AsMut<[u8]>> ReconstructShard for (T, bool) {
    fn len(&self) -> Option<usize> {
        if !self.1 {
            None
        } else {
            Some(self.0.as_ref().len())
        }
    }

    fn get(&mut self) -> Option<&mut [u8]> {
        if !self.1 {
            None
        } else {
            Some(self.0.as_mut())
        }
    }

    fn get_or_initialize(
        &mut self,
        len: usize,
    ) -> Result<&mut [u8], Result<&mut [u8], Error>> {
        let present = self.1;
        let x = self.0.as_mut();
        if x.len() == len {
            if present {
                Ok(x)
            } else {
                Err(Ok(x))
            }
        } else {
            Err(Err(Error::IncorrectShardSize))
        }
    }
}

impl<T: AsRef<[u8]> + AsMut<[u8]> + FromIterator<u8>> ReconstructShard for Option<T> {
    fn len(&self) -> Option<usize> {
        self.as_ref().map(|x| x.as_ref().len())
    }

    fn get(&mut self) -> Option<&mut [u8]> {
        self.as_mut().map(|x| x.as_mut())
    }

    fn get_or_initialize(
        &mut self,
        len: usize,
    ) -> Result<&mut [u8], Result<&mut [u8], Error>> {
        let is_some = self.is_some();
        let x = self
            .get_or_insert_with(|| core::iter::repeat_n(0u8, len).collect())
            .as_mut();
        if is_some {
            Ok(x)
        } else {
            Err(Ok(x))
        }
    }
}

/// Reed-Solomon erasure code encoder/decoder over GF(2^8).
#[derive(Debug, Clone)]
pub struct ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,
    matrix: Matrix,
    /// Parity generator rows (bottom `m` rows of the systematic matrix), cached
    /// so the hot paths borrow them instead of rebuilding the row list per call
    parity_rows: Vec<Vec<u8>>,
    /// Cached affine broadcasts for the m parity rows, built once so the fused
    /// x86 encode has no per-call matrix build
    #[cfg(target_arch = "x86_64")]
    parity_affine: Vec<i64>,
    /// Cached nibble tables for the same rows, so the fused AVX2 encode is
    /// allocation-free per call
    #[cfg(target_arch = "x86_64")]
    parity_nibbles: gf::x86_fused::NibbleTables,
    /// Cached nibble lo/hi tables (`lo|hi` per coefficient) so the fused wasm
    /// encode has no per-call table rebuild
    #[cfg(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64"))]
    parity_tables: Vec<u8>,
}

impl PartialEq for ReedSolomon {
    fn eq(&self, other: &Self) -> bool {
        self.data_shard_count == other.data_shard_count
            && self.parity_shard_count == other.parity_shard_count
    }
}

impl ReedSolomon {
    /// Builds the systematic generator matrix as `vandermonde(total, data) * invert(top block)`
    fn build_matrix(data_shards: usize, total_shards: usize) -> Matrix {
        let vandermonde = Matrix::vandermonde(total_shards, data_shards);
        let top = vandermonde.sub_matrix(0, 0, data_shards, data_shards);
        vandermonde.multiply(&top.invert().unwrap())
    }

    /// Creates a new Reed-Solomon coder.
    ///
    /// Returns `Error::TooFewDataShards` if `data_shards == 0`,
    /// `Error::TooFewParityShards` if `parity_shards == 0`,
    /// `Error::TooManyShards` if `data_shards + parity_shards > 256`.
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<ReedSolomon, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > 256 {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;
        let matrix = Self::build_matrix(data_shards, total_shards);

        // Own the parity rows once so the hot paths never rebuild the pointer list.
        let parity_rows: Vec<Vec<u8>> = (data_shards..total_shards)
            .map(|o| matrix.get_row(o).to_vec())
            .collect();

        // Precompute the affine matrices once so the fused x86 encode never rebuilds them.
        #[cfg(target_arch = "x86_64")]
        let parity_affine: Vec<i64> = {
            let mut v = Vec::with_capacity(parity_shards * data_shards);
            for o in data_shards..total_shards {
                let row = matrix.get_row(o);
                for i in 0..data_shards {
                    v.push(gf::x86_fused::affine_of(row[i]));
                }
            }
            v
        };

        // Cached nibble tables for the fused AVX2 encode, built once here.
        #[cfg(target_arch = "x86_64")]
        let parity_nibbles: gf::x86_fused::NibbleTables = {
            let mut flat = vec![0u8; parity_shards * data_shards];
            for (oi, o) in (data_shards..total_shards).enumerate() {
                let row = matrix.get_row(o);
                flat[oi * data_shards..(oi + 1) * data_shards]
                    .copy_from_slice(&row[..data_shards]);
            }
            gf::x86_fused::NibbleTables::new(&flat, parity_shards, data_shards)
        };

        // Cached nibble lo/hi tables for the fused wasm kernel, built once here.
        #[cfg(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64"))]
        let parity_tables: Vec<u8> = {
            let mut t = Vec::with_capacity(parity_shards * data_shards * 32);
            for o in data_shards..total_shards {
                let row = matrix.get_row(o);
                for i in 0..data_shards {
                    let mrow = &crate::galois::MUL_TABLE[row[i] as usize];
                    for x in 0..16 {
                        t.push(mrow[x]); // lo[x] = c*x
                    }
                    for x in 0..16 {
                        t.push(mrow[x << 4]); // hi[x] = c*(x<<4)
                    }
                }
            }
            t
        };

        Ok(ReedSolomon {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: total_shards,
            matrix,
            parity_rows,
            #[cfg(target_arch = "x86_64")]
            parity_affine,
            #[cfg(target_arch = "x86_64")]
            parity_nibbles,
            #[cfg(any(all(target_arch = "wasm32", target_feature = "simd128"), target_arch = "aarch64"))]
            parity_tables,
        })
    }

    pub fn data_shard_count(&self) -> usize {
        self.data_shard_count
    }

    pub fn parity_shard_count(&self) -> usize {
        self.parity_shard_count
    }

    pub fn total_shard_count(&self) -> usize {
        self.total_shard_count
    }

    /// For each output row, accumulate `matrix_rows[out][in] * inputs[in]`.
    /// The first input uses `mul_slice` (overwrite), the rest `mul_slice_xor`.
    fn code_some_slices<Row: AsRef<[u8]>, T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        matrix_rows: &[Row],
        inputs: &[T],
        outputs: &mut [U],
    ) {
        for (i_input, input) in inputs.iter().enumerate() {
            let input = input.as_ref();
            for (i_row, output) in outputs.iter_mut().enumerate() {
                let coeff = matrix_rows[i_row].as_ref()[i_input];
                let output = output.as_mut();
                if i_input == 0 {
                    gf::mul_slice(output, input, coeff);
                } else {
                    gf::mul_slice_xor(output, input, coeff);
                }
            }
        }
    }

    /// Constructs the parity shards, overwriting the parity slots.
    ///
    /// `shards` holds `data_shard_count` data shards followed by
    /// `parity_shard_count` parity shards, all of equal length.
    pub fn encode<T: AsRef<[u8]> + AsMut<[u8]>>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;

        let (input, output) = shards.split_at_mut(self.data_shard_count);
        let parity_rows = &self.parity_rows;

        // x86 routes through the fused path (byte-identical to the per-coefficient
        // dispatch); it falls back to a tiled per-shard path on shapes and CPUs it
        // does not specialise. Other architectures use the per-coefficient path.
        #[cfg(target_arch = "x86_64")]
        gf::x86_fused::encode_ymm_dispatch(
            &self.parity_affine,
            &self.parity_nibbles,
            parity_rows,
            input,
            output,
        );
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        gf::wasm128::encode_fused(&self.parity_tables, parity_rows, input, output);
        #[cfg(target_arch = "aarch64")]
        gf::neon_fused::encode_fused(&self.parity_tables, parity_rows, input, output);
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )))]
        {
            self.code_some_slices(parity_rows, input, output);
        }
        Ok(())
    }

    /// Alias for [`encode`](Self::encode), kept for callers that name the fused path directly
    pub fn encode_fused<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [T],
    ) -> Result<(), Error> {
        self.encode(shards)
    }

    /// Bench-only: encode forcing a specific single-output x86 kernel.
    /// `kind`: 0 ssse3, 1 avx2, 2 avx512-nibble, 3 gfni512-single. Byte-identical to [`encode`]
    #[cfg(target_arch = "x86_64")]
    #[doc(hidden)]
    pub fn encode_forced<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [T],
        kind: u8,
    ) -> Result<(), Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;
        let (input, output) = shards.split_at_mut(self.data_shard_count);
        if input.is_empty() {
            return Ok(());
        }
        let parity_rows = &self.parity_rows;
        let len = input[0].as_ref().len();
        const BLK: usize = 32 * 1024;
        let mut start = 0;
        while start < len {
            let end = (start + BLK).min(len);
            for (i_row, out) in output.iter_mut().enumerate() {
                let row = &parity_rows[i_row];
                let orow = out.as_mut();
                let oblk = &mut orow[start..end];
                for (i_in, inp) in input.iter().enumerate() {
                    let iblk = &inp.as_ref()[start..end];
                    if i_in == 0 {
                        gf::x86::forced::<false>(kind, oblk, iblk, row[i_in]);
                    } else {
                        gf::x86::forced::<true>(kind, oblk, iblk, row[i_in]);
                    }
                }
            }
            start = end;
        }
        Ok(())
    }

    /// Encode through the scalar reference kernel, bypassing SIMD dispatch.
    #[doc(hidden)]
    pub fn encode_scalar<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [T],
    ) -> Result<(), Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;

        let (input, output) = shards.split_at_mut(self.data_shard_count);
        let parity_rows = &self.parity_rows;
        for (i_input, inp) in input.iter().enumerate() {
            let inp = inp.as_ref();
            for (i_row, out) in output.iter_mut().enumerate() {
                let coeff = parity_rows[i_row][i_input];
                let out = out.as_mut();
                if i_input == 0 {
                    gf::scalar::mul_slice(out, inp, coeff);
                } else {
                    gf::scalar::mul_slice_xor(out, inp, coeff);
                }
            }
        }
        Ok(())
    }

    /// Checks whether the parity shards are correct for the given data.
    pub fn verify<T: AsRef<[u8]>>(&self, shards: &[T]) -> Result<bool, Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;

        let len = shards[0].as_ref().len();
        let mut buffer: Vec<Vec<u8>> = (0..self.parity_shard_count)
            .map(|_| vec![0u8; len])
            .collect();

        let data = &shards[0..self.data_shard_count];
        self.code_some_slices(&self.parity_rows, data, &mut buffer);

        let to_check = &shards[self.data_shard_count..];
        let ok = buffer
            .iter()
            .enumerate()
            .all(|(i, computed)| computed.as_slice() == to_check[i].as_ref());
        Ok(ok)
    }

    /// Reconstructs all missing shards (data and parity) in place.
    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, false)
    }

    /// Reconstructs only the missing data shards in place.
    pub fn reconstruct_data<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, true)
    }

    /// Builds the k x k decode matrix from the rows of the generator matrix
    /// that correspond to the shards we still have, then inverts it.
    fn data_decode_matrix(&self, valid_indices: &[usize]) -> Matrix {
        let mut sub_matrix = Matrix::new(self.data_shard_count, self.data_shard_count);
        for (sub_row, &valid_index) in valid_indices.iter().enumerate() {
            for c in 0..self.data_shard_count {
                sub_matrix.set(sub_row, c, self.matrix.get(valid_index, c));
            }
        }
        sub_matrix.invert().unwrap()
    }

    fn reconstruct_internal<T: ReconstructShard>(
        &self,
        shards: &mut [T],
        data_only: bool,
    ) -> Result<(), Error> {
        if shards.len() != self.total_shard_count {
            return Err(if shards.len() < self.total_shard_count {
                Error::TooFewShards
            } else {
                Error::TooManyShards
            });
        }

        let data_shard_count = self.data_shard_count;

        // Are all shards present? If so, nothing to do.
        let mut number_present = 0;
        let mut shard_len = None;
        for shard in shards.iter_mut() {
            if let Some(len) = shard.len() {
                if len == 0 {
                    return Err(Error::EmptyShard);
                }
                number_present += 1;
                if let Some(old_len) = shard_len {
                    if len != old_len {
                        return Err(Error::IncorrectShardSize);
                    }
                }
                shard_len = Some(len);
            }
        }

        if number_present == self.total_shard_count {
            return Ok(());
        }
        if number_present < data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }

        let shard_len = shard_len.expect("at least one shard present");

        let mut sub_shards: Vec<&[u8]> = Vec::with_capacity(data_shard_count);
        let mut missing_data_slices: Vec<&mut [u8]> = Vec::with_capacity(self.parity_shard_count);
        let mut missing_parity_slices: Vec<&mut [u8]> = Vec::with_capacity(self.parity_shard_count);
        let mut valid_indices: Vec<usize> = Vec::with_capacity(data_shard_count);
        let mut invalid_indices: Vec<usize> = Vec::with_capacity(data_shard_count);

        for (matrix_row, shard) in shards.iter_mut().enumerate() {
            let shard_data = if matrix_row >= data_shard_count && data_only {
                shard.get().ok_or(None)
            } else {
                shard.get_or_initialize(shard_len).map_err(Some)
            };

            match shard_data {
                Ok(shard) => {
                    if sub_shards.len() < data_shard_count {
                        sub_shards.push(shard);
                        valid_indices.push(matrix_row);
                    }
                }
                Err(None) => {
                    invalid_indices.push(matrix_row);
                }
                Err(Some(x)) => {
                    let shard = x?;
                    if matrix_row < data_shard_count {
                        missing_data_slices.push(shard);
                    } else {
                        missing_parity_slices.push(shard);
                    }
                    invalid_indices.push(matrix_row);
                }
            }
        }

        let data_decode_matrix = self.data_decode_matrix(&valid_indices);

        // Re-create any missing data shards.
        let mut matrix_rows: Vec<&[u8]> = Vec::with_capacity(self.parity_shard_count);
        for &i_slice in invalid_indices.iter().take_while(|&&i| i < data_shard_count) {
            matrix_rows.push(data_decode_matrix.get_row(i_slice));
        }
        self.code_some_slices(&matrix_rows, &sub_shards, &mut missing_data_slices);

        if data_only {
            return Ok(());
        }

        // Recompute any missing parity shards from the (now complete) data.
        let mut matrix_rows: Vec<&[u8]> = Vec::with_capacity(self.parity_shard_count);
        for &i_slice in invalid_indices.iter().skip_while(|&&i| i < data_shard_count) {
            matrix_rows.push(self.parity_rows[i_slice - data_shard_count].as_slice());
        }

        // Gather all data shards: existing ones from `sub_shards`, freshly
        // reconstructed ones from `missing_data_slices`, in original order.
        let mut all_data_slices: Vec<&[u8]> = Vec::with_capacity(data_shard_count);
        let mut i_old = 0;
        let mut next_maybe_good = 0;
        for (i_new, &i_slice) in invalid_indices
            .iter()
            .take_while(|&&i| i < data_shard_count)
            .enumerate()
        {
            for _ in next_maybe_good..i_slice {
                all_data_slices.push(sub_shards[i_old]);
                i_old += 1;
            }
            all_data_slices.push(&*missing_data_slices[i_new]);
            next_maybe_good = i_slice + 1;
        }
        for _ in next_maybe_good..data_shard_count {
            all_data_slices.push(sub_shards[i_old]);
            i_old += 1;
        }

        self.code_some_slices(&matrix_rows, &all_data_slices, &mut missing_parity_slices);
        Ok(())
    }

    // --- full-row batched encode ---

    /// Encodes parity over full contiguous node rows in one sweep.
    ///
    /// `data` is `data_shard_count` rows of `row_len` bytes laid out contiguously
    /// (row `j` at `data[j*row_len .. (j+1)*row_len]`). Returns
    /// `parity_shard_count * row_len` parity bytes, row `i` first.
    ///
    /// Panics if `data.len() != data_shard_count * row_len`.
    pub fn encode_rows(&self, data: &[u8], row_len: usize) -> Vec<u8> {
        assert_eq!(
            data.len(),
            self.data_shard_count * row_len,
            "data must be data_shard_count * row_len bytes"
        );
        let mut out = vec![0u8; self.parity_shard_count * row_len];
        let parity_rows = &self.parity_rows;
        for j in 0..self.data_shard_count {
            let input = &data[j * row_len..(j + 1) * row_len];
            for i in 0..self.parity_shard_count {
                let coeff = parity_rows[i][j];
                let o = &mut out[i * row_len..(i + 1) * row_len];
                if j == 0 {
                    gf::mul_slice(o, input, coeff);
                } else {
                    gf::mul_slice_xor(o, input, coeff);
                }
            }
        }
        out
    }

    // TODO: full-row reconstruct — one inversion + one sweep over whole rows.
    //   pub fn reconstruct_rows(&self, node_rows: &mut [u8], row_len: usize, erased: &[bool]) -> Result<(), Error>

    // TODO: prepared decoder — invert once, reuse across per-plane calls.
    //   pub fn prepare_decode(&self, erased: &[bool]) -> Result<PreparedDecoder, Error>

    // --- internal checks ------------------------------------------------------

    fn check_piece_count_all(&self, count: usize) -> Result<(), Error> {
        if count < self.total_shard_count {
            Err(Error::TooFewShards)
        } else if count > self.total_shard_count {
            Err(Error::TooManyShards)
        } else {
            Ok(())
        }
    }

    fn check_equal_lengths<T: AsRef<[u8]>>(&self, shards: &[T]) -> Result<(), Error> {
        let len = shards[0].as_ref().len();
        if len == 0 {
            return Err(Error::EmptyShard);
        }
        for s in shards.iter() {
            if s.as_ref().len() != len {
                return Err(Error::IncorrectShardSize);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_rows_matches_per_shard_encode() {
        let (k, m, row_len) = (6usize, 4usize, 293usize);
        let rs = ReedSolomon::new(k, m).unwrap();

        // Deterministic pseudo-random data.
        let mut data = vec![0u8; k * row_len];
        let mut x: u32 = 0x1234_5678;
        for b in data.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 16) as u8;
        }

        // Per-shard encode.
        let mut shards: Vec<Vec<u8>> = (0..k)
            .map(|j| data[j * row_len..(j + 1) * row_len].to_vec())
            .collect();
        shards.extend((0..m).map(|_| vec![0u8; row_len]));
        rs.encode(&mut shards).unwrap();
        let mut expected = Vec::with_capacity(m * row_len);
        for i in 0..m {
            expected.extend_from_slice(&shards[k + i]);
        }

        // Full-row batched encode.
        let got = rs.encode_rows(&data, row_len);
        assert_eq!(got, expected);
    }

    #[test]
    fn encode_fused_matches_encode() {
        // Fused encode must be byte-identical to the per-shard encode at the
        // production shapes and a spread of shard sizes, including tail lengths.
        for &(k, m) in &[(10usize, 10usize), (20, 10), (4, 2), (6, 4), (17, 17)] {
            let rs = ReedSolomon::new(k, m).unwrap();
            for &len in &[1usize, 32, 63, 100, 1024, 10000, 65536] {
                let mut data = vec![0u8; k * len];
                let mut x: u32 = 0x9E37_79B9;
                for b in data.iter_mut() {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    *b = (x >> 16) as u8;
                }
                let mut base: Vec<Vec<u8>> = (0..k)
                    .map(|j| data[j * len..(j + 1) * len].to_vec())
                    .collect();
                base.extend((0..m).map(|_| vec![0u8; len]));

                let mut a = base.clone();
                let mut b = base.clone();
                rs.encode(&mut a).unwrap();
                rs.encode_fused(&mut b).unwrap();
                assert_eq!(a, b, "encode_fused != encode k={k} m={m} len={len}");

                // And the scalar reference agrees too.
                let mut c = base.clone();
                rs.encode_scalar(&mut c).unwrap();
                assert_eq!(a, c, "encode_scalar != encode k={k} m={m} len={len}");
            }
        }
    }

    #[test]
    fn round_trip_reconstruct() {
        let rs = ReedSolomon::new(4, 4).unwrap();
        let len = 100;
        let mut shards: Vec<Vec<u8>> = (0..4).map(|j| vec![j as u8 + 1; len]).collect();
        shards.extend((0..4).map(|_| vec![0u8; len]));
        rs.encode(&mut shards).unwrap();
        let original = shards.clone();

        // Erase 2 data + 1 parity.
        let mut opt: Vec<(&mut [u8], bool)> = shards
            .iter_mut()
            .map(|s| (s.as_mut_slice(), true))
            .collect();
        opt[0].1 = false;
        opt[2].1 = false;
        opt[5].1 = false;

        rs.reconstruct(&mut opt).unwrap();
        for (i, s) in shards.iter().enumerate() {
            assert_eq!(s, &original[i], "shard {i} mismatch");
        }
    }
}