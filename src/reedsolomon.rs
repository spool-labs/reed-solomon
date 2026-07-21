//! Reed-Solomon erasure coder over GF(2^8).
//!
//! ```ignore
//! let rs = ReedSolomon::new(data_shards, parity_shards)?;
//! rs.encode(&mut slices)?;      // slices: Vec<&mut [u8]>
//! rs.reconstruct(&mut shards)?; // shards: Vec<(&mut [u8], bool)>
//! ```

use std::sync::{Arc, RwLock};

use crate::errors::Error;
use crate::gf;
use crate::gf::tables::RowTables;
#[cfg(fft_enabled)]
use crate::fft::{self, StagedProgram};
use crate::matrix::Matrix;

/// Verify recomputes parity in blocks of this size, so scratch stays cache
/// resident and a mismatch is caught without recomputing the whole shard
const VERIFY_BLOCK_SIZE: usize = 32 * 1024;

/// Decode plans kept per codec. Callers typically cycle through a small set of
/// erasure patterns per decode, so a short list scanned linearly is enough
const DECODE_PLAN_CACHE_LIMIT: usize = 64;

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

/// Bitmask of which shard slots are present, one bit per shard index
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PresenceMask([u64; 4]);

impl PresenceMask {
    fn empty() -> PresenceMask {
        PresenceMask([0u64; 4])
    }

    fn set(&mut self, index: usize) {
        self.0[index / 64] |= 1u64 << (index % 64);
    }
}

/// Everything reconstruction needs for one erasure pattern, built once and
/// reused across calls that see the same pattern
#[derive(Debug)]
struct DecodePlan {
    /// The exact presence pattern this plan serves
    presence: PresenceMask,
    /// Erased data shard indexes, ascending
    missing_data: Vec<usize>,
    /// Erased parity shard indexes, ascending
    missing_parity: Vec<usize>,
    /// Inverted decode rows that rebuild the missing data from the survivors
    data_tables: RowTables,
    /// Parity generator rows that rebuild the missing parity from full data
    parity_tables: RowTables,
}

/// Recently used decode plans, most recent first
type PlanCache = Vec<(PresenceMask, Arc<DecodePlan>)>;

/// Reed-Solomon erasure code encoder/decoder over GF(2^8).
#[derive(Debug)]
pub struct ReedSolomon {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,

    matrix: Matrix<u8>,
    parity: RowTables,

    #[cfg(fft_enabled)]
    staged: Option<Arc<StagedProgram>>,

    decode_plans: RwLock<PlanCache>,
}

impl ReedSolomon {
    /// Builds the systematic generator matrix as `vandermonde(total, data) * invert(top block)`
    fn build_matrix(data_shards: usize, total_shards: usize) -> Result<Matrix<u8>, Error> {
        let vandermonde = Matrix::<u8>::vandermonde(total_shards, data_shards);
        let top = vandermonde.sub_matrix(0, 0, data_shards, data_shards);
        Ok(vandermonde.multiply(&top.invert()?))
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
        let matrix = Self::build_matrix(data_shards, total_shards)?;

        // Own the parity rows once so no hot path rebuilds tables per call.
        let mut parity_rows: Vec<Vec<u8>> = Vec::with_capacity(parity_shards);
        for output in data_shards..total_shards {
            parity_rows.push(matrix.get_row(output).to_vec());
        }

        Ok(ReedSolomon {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: total_shards,
            matrix,
            parity: RowTables::new(parity_rows, data_shards),
            #[cfg(fft_enabled)]
            staged: Self::build_staged(data_shards, parity_shards),
            decode_plans: RwLock::new(Vec::new()),
        })
    }

    /// Compile the staged FFT program when it will actually pay off
    ///
    /// The generated straight-line executors own the production shapes, tiny
    /// and non-power-of-two data counts gain nothing over the matrix path,
    /// oversized register files fall back to it, and a shape only routes here
    /// when its real multiply count clearly undercuts the schoolbook product.
    #[cfg(fft_enabled)]
    fn build_staged(data_shards: usize, parity_shards: usize) -> Option<Arc<StagedProgram>> {
        // Route staged only for power-of-two data counts. Those shapes
        // interpolate with a single inverse transform, few big stages, and
        // almost no glue; measured on M4, (16,16) staged beats the fused
        // kernels about 2x while non-power shapes like (14,14) lose to them,
        // because their inter-stage glue serializes on store-to-load
        // forwarding. Generated shapes and tiny counts also never reach the
        // (relatively costly) program build, so gate all three before it.
        let is_generated =
            crate::fft_programs::GENERATED_SHAPES.contains(&(data_shards, parity_shards));
        if is_generated || data_shards < 2 || !data_shards.is_power_of_two() {
            return None;
        }
        let program = fft::build_staged_program(data_shards, parity_shards);
        if program.register_count > fft::MAX_STAGED_REGISTERS {
            return None;
        }

        // Take it only when the real multiply count clearly undercuts the
        // schoolbook product. The margin is wider for the table-lookup
        // multiply (NEON, wasm) than for the single-instruction GFNI affine.
        #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))]
        const MULT_MARGIN: usize = 3;
        #[cfg(target_arch = "x86_64")]
        const MULT_MARGIN: usize = 2;
        if program.mult_count * MULT_MARGIN >= data_shards * parity_shards {
            return None;
        }
        Some(Arc::new(program))
    }

    /// Bench and test introspection: the path this codec's encode takes for
    /// the given shard length on this build
    #[doc(hidden)]
    pub fn encode_route(&self, shard_len: usize) -> &'static str {
        #[cfg(fft_enabled)]
        if gf::fft_active::eligible(shard_len) {
            let generated = crate::fft_programs::GENERATED_SHAPES
                .contains(&(self.data_shard_count, self.parity_shard_count));
            if generated {
                return "fft-generated";
            }
            if self.staged.is_some() {
                return "fft-staged";
            }
        }
        let _ = shard_len;
        "fused-matrix"
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

    /// Constructs the parity shards, overwriting the parity slots.
    ///
    /// `shards` holds `data_shard_count` data shards followed by
    /// `parity_shard_count` parity shards, all of equal length.
    pub fn encode<T: AsRef<[u8]> + AsMut<[u8]>>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;

        let (input, output) = shards.split_at_mut(self.data_shard_count);
        self.encode_into(input, output)
    }

    /// Encode with the data and parity shards in separate slices, matching
    /// `reed-solomon-erasure`'s `encode_sep`: `data` is read-only and each
    /// `parity` shard is overwritten. All shards must share one length.
    ///
    /// Byte-identical to [`encode`](Self::encode) for the same data.
    pub fn encode_sep<T: AsRef<[u8]>, U: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        if data.len() != self.data_shard_count {
            return Err(if data.len() < self.data_shard_count {
                Error::TooFewDataShards
            } else {
                Error::TooManyDataShards
            });
        }
        if parity.len() != self.parity_shard_count {
            return Err(if parity.len() < self.parity_shard_count {
                Error::TooFewParityShards
            } else {
                Error::TooManyParityShards
            });
        }
        let len = data[0].as_ref().len();
        let uneven = data.iter().any(|s| s.as_ref().len() != len)
            || parity.iter().any(|s| s.as_ref().len() != len);
        if uneven {
            return Err(Error::IncorrectShardSize);
        }
        self.encode_into(data, parity)
    }

    /// Shared encode core: parity outputs from the data inputs, routed through
    /// the generated/staged FFT programs or the fused matrix kernels, all
    /// byte-identical. Callers guarantee the shard counts and equal lengths.
    fn encode_into<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
        &self,
        input: &[In],
        output: &mut [Out],
    ) -> Result<(), Error> {
        // The production shapes run their generated FFT programs and any
        // other profitable shape runs its staged program; both need a
        // fraction of the schoolbook multiplies. Remaining shapes and
        // sub-strip lengths take the fused matrix path. All byte-identical.
        #[cfg(fft_enabled)]
        if gf::fft_active::eligible(input[0].as_ref().len()) {
            if gf::fft_active::encode_generated(
                self.data_shard_count,
                self.parity_shard_count,
                input,
                output,
            ) {
                return Ok(());
            }
            if let Some(program) = &self.staged {
                gf::fft_active::encode_staged(program, input, output);
                return Ok(());
            }
        }

        self.parity.apply(input, output);
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
        let parity_rows = self.parity.rows();
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
        let parity_rows = self.parity.rows();
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
    ///
    /// Recomputes parity through the fused kernels in cache-sized blocks and
    /// stops at the first mismatch.
    pub fn verify<T: AsRef<[u8]>>(&self, shards: &[T]) -> Result<bool, Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_equal_lengths(shards)?;

        let shard_len = shards[0].as_ref().len();
        let block_len = shard_len.min(VERIFY_BLOCK_SIZE);
        let mut scratch: Vec<Vec<u8>> = vec![vec![0u8; block_len]; self.parity_shard_count];

        let data = &shards[..self.data_shard_count];
        let parity = &shards[self.data_shard_count..];

        let mut start = 0;
        while start < shard_len {
            let end = (start + block_len).min(shard_len);

            let mut inputs: Vec<&[u8]> = Vec::with_capacity(data.len());
            for shard in data {
                inputs.push(&shard.as_ref()[start..end]);
            }
            let mut outputs: Vec<&mut [u8]> = Vec::with_capacity(scratch.len());
            for block in scratch.iter_mut() {
                outputs.push(&mut block[..end - start]);
            }
            self.parity.apply(&inputs, &mut outputs);

            for (index, computed) in outputs.iter().enumerate() {
                if computed[..] != parity[index].as_ref()[start..end] {
                    return Ok(false);
                }
            }
            start = end;
        }
        Ok(true)
    }

    /// Reconstructs all missing shards (data and parity) in place.
    ///
    /// The decode matrix for the erasure pattern is inverted once and cached,
    /// so repeated calls with the same pattern skip straight to the fused
    /// kernels.
    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, false)
    }

    /// Reconstructs only the missing data shards in place.
    pub fn reconstruct_data<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, true)
    }

    /// Prepare a decoder for one presence pattern, `true` per present shard
    ///
    /// The returned decoder owns everything reconstruction needs, so callers
    /// that decode many stripes with one erasure pattern skip the per-call
    /// pattern lookup entirely.
    pub fn prepare_decode(&self, present: &[bool]) -> Result<PreparedDecoder, Error> {
        if present.len() != self.total_shard_count {
            return Err(Error::InvalidShardFlags);
        }

        let mut presence = PresenceMask::empty();
        let mut number_present = 0;
        for (index, &is_present) in present.iter().enumerate() {
            if is_present {
                presence.set(index);
                number_present += 1;
            }
        }
        if number_present < self.data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }

        Ok(PreparedDecoder {
            data_shard_count: self.data_shard_count,
            total_shard_count: self.total_shard_count,
            plan: self.plan_for(presence)?,
        })
    }

    /// Reconstructs missing rows inside one contiguous buffer
    ///
    /// `rows` holds `total_shard_count` rows of `row_len` bytes back to back
    /// and `present` flags each row, `true` when its contents are valid.
    /// Missing rows are rebuilt in place with one plan lookup and one fused
    /// sweep over the whole rows.
    pub fn reconstruct_rows(
        &self,
        rows: &mut [u8],
        row_len: usize,
        present: &[bool],
    ) -> Result<(), Error> {
        if present.len() != self.total_shard_count {
            return Err(Error::InvalidShardFlags);
        }
        if row_len == 0 {
            return Err(Error::EmptyShard);
        }
        if rows.len() != self.total_shard_count * row_len {
            return Err(Error::IncorrectShardSize);
        }

        let mut shards: Vec<(&mut [u8], bool)> = Vec::with_capacity(self.total_shard_count);
        for (row, &is_present) in rows.chunks_mut(row_len).zip(present) {
            shards.push((row, is_present));
        }
        self.reconstruct(&mut shards)
    }

    /// Builds the k x k decode matrix from the rows of the generator matrix
    /// that correspond to the shards we still have, then inverts it.
    fn data_decode_matrix(&self, valid_indices: &[usize]) -> Result<Matrix<u8>, Error> {
        let mut sub_matrix = Matrix::<u8>::new(self.data_shard_count, self.data_shard_count);
        for (sub_row, &valid_index) in valid_indices.iter().enumerate() {
            for c in 0..self.data_shard_count {
                sub_matrix.set(sub_row, c, self.matrix.get(valid_index, c));
            }
        }
        sub_matrix.invert()
    }

    /// Build the decode plan for one presence pattern
    fn build_plan(&self, presence: PresenceMask) -> Result<DecodePlan, Error> {
        let data_shard_count = self.data_shard_count;

        let mut valid_indices: Vec<usize> = Vec::with_capacity(data_shard_count);
        let mut missing_data: Vec<usize> = Vec::new();
        let mut missing_parity: Vec<usize> = Vec::new();
        for index in 0..self.total_shard_count {
            if presence.0[index / 64] & (1u64 << (index % 64)) != 0 {
                if valid_indices.len() < data_shard_count {
                    valid_indices.push(index);
                }
            } else if index < data_shard_count {
                missing_data.push(index);
            } else {
                missing_parity.push(index);
            }
        }

        let decode_matrix = self.data_decode_matrix(&valid_indices)?;
        let mut data_rows: Vec<Vec<u8>> = Vec::with_capacity(missing_data.len());
        for &index in &missing_data {
            data_rows.push(decode_matrix.get_row(index).to_vec());
        }
        let mut parity_rows: Vec<Vec<u8>> = Vec::with_capacity(missing_parity.len());
        for &index in &missing_parity {
            parity_rows.push(self.parity.rows()[index - data_shard_count].to_vec());
        }

        Ok(DecodePlan {
            presence,
            missing_data,
            missing_parity,
            data_tables: RowTables::new(data_rows, data_shard_count),
            parity_tables: RowTables::new(parity_rows, data_shard_count),
        })
    }

    /// Fetch the cached plan for a presence pattern, building it on first use
    ///
    /// The hit path takes only a shared read lock and does not reorder the
    /// cache, so many threads sharing one codec (Agave's parallel receive path)
    /// reconstruct concurrently instead of serializing on an exclusive lock.
    fn plan_for(&self, presence: PresenceMask) -> Result<Arc<DecodePlan>, Error> {
        {
            let plans = self.decode_plans.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            for (mask, plan) in plans.iter() {
                if *mask == presence {
                    return Ok(plan.clone());
                }
            }
        }

        // Miss: build outside the lock, then take the write lock to insert. A
        // thread that raced us to the same pattern already inserted it, so reuse
        // that entry and drop our redundant build.
        let plan = Arc::new(self.build_plan(presence)?);
        let mut plans = self.decode_plans.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, existing)) = plans.iter().find(|(mask, _)| *mask == presence) {
            return Ok(existing.clone());
        }
        plans.insert(0, (presence, plan.clone()));
        plans.truncate(DECODE_PLAN_CACHE_LIMIT);
        Ok(plan)
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

        let (presence, number_present, shard_len) = scan_shards(shards)?;
        if number_present == self.total_shard_count {
            return Ok(());
        }
        if number_present < self.data_shard_count {
            return Err(Error::TooFewShardsPresent);
        }
        let Some(shard_len) = shard_len else {
            return Err(Error::TooFewShardsPresent);
        };

        let plan = self.plan_for(presence)?;
        reconstruct_with_plan(self.data_shard_count, &plan, shards, shard_len, data_only)
    }

    // full-row batched encode

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
        let mut parity = vec![0u8; self.parity_shard_count * row_len];
        self.apply_rows(data, row_len, &mut parity);
        parity
    }

    /// Encodes parity over full contiguous node rows into a caller buffer
    ///
    /// Same layout as `encode_rows`, without the per-call allocation. `parity`
    /// must hold `parity_shard_count * row_len` bytes.
    pub fn encode_rows_into(
        &self,
        data: &[u8],
        row_len: usize,
        parity: &mut [u8],
    ) -> Result<(), Error> {
        if data.len() != self.data_shard_count * row_len {
            return Err(Error::IncorrectShardSize);
        }
        if parity.len() != self.parity_shard_count * row_len {
            return Err(Error::IncorrectShardSize);
        }
        self.apply_rows(data, row_len, parity);
        Ok(())
    }

    /// Fused parity sweep over contiguous rows, lengths already validated
    fn apply_rows(&self, data: &[u8], row_len: usize, parity: &mut [u8]) {
        if row_len == 0 {
            return;
        }
        let mut inputs: Vec<&[u8]> = Vec::with_capacity(self.data_shard_count);
        for row in data.chunks(row_len) {
            inputs.push(row);
        }
        let mut outputs: Vec<&mut [u8]> = Vec::with_capacity(self.parity_shard_count);
        for row in parity.chunks_mut(row_len) {
            outputs.push(row);
        }
        self.parity.apply(&inputs, &mut outputs);
    }

    // internal checks

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

impl Clone for ReedSolomon {
    fn clone(&self) -> ReedSolomon {
        let plans = self.decode_plans.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        ReedSolomon {
            data_shard_count: self.data_shard_count,
            parity_shard_count: self.parity_shard_count,
            total_shard_count: self.total_shard_count,
            matrix: self.matrix.clone(),
            parity: self.parity.clone(),
            #[cfg(fft_enabled)]
            staged: self.staged.clone(),
            decode_plans: RwLock::new(plans.clone()),
        }
    }
}

impl PartialEq for ReedSolomon {
    fn eq(&self, other: &Self) -> bool {
        self.data_shard_count == other.data_shard_count
            && self.parity_shard_count == other.parity_shard_count
    }
}

/// A decoder prepared for one presence pattern
///
/// Owns the inverted decode matrix and its kernel tables, so reconstructing
/// many stripes with the same erasure pattern pays for the inversion once.
/// The shards handed to it must match the prepared pattern exactly.
#[derive(Clone, Debug)]
pub struct PreparedDecoder {
    data_shard_count: usize,
    total_shard_count: usize,
    plan: Arc<DecodePlan>,
}

impl PreparedDecoder {
    /// Reconstructs all missing shards (data and parity) in place.
    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, false)
    }

    /// Reconstructs only the missing data shards in place.
    pub fn reconstruct_data<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, true)
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

        let (presence, number_present, shard_len) = scan_shards(shards)?;
        if presence != self.plan.presence {
            return Err(Error::InvalidShardFlags);
        }
        if number_present == self.total_shard_count {
            return Ok(());
        }
        let Some(shard_len) = shard_len else {
            return Err(Error::TooFewShardsPresent);
        };

        reconstruct_with_plan(self.data_shard_count, &self.plan, shards, shard_len, data_only)
    }
}

/// One pass over the shards: presence mask, present count, and the shared
/// shard length with empty and mismatched sizes rejected
fn scan_shards<Shard: ReconstructShard>(
    shards: &[Shard],
) -> Result<(PresenceMask, usize, Option<usize>), Error> {
    let mut presence = PresenceMask::empty();
    let mut number_present = 0;
    let mut shard_len: Option<usize> = None;
    for (index, shard) in shards.iter().enumerate() {
        if let Some(len) = shard.len() {
            if len == 0 {
                return Err(Error::EmptyShard);
            }
            match shard_len {
                Some(existing) if existing != len => return Err(Error::IncorrectShardSize),
                _ => shard_len = Some(len),
            }
            presence.set(index);
            number_present += 1;
        }
    }
    Ok((presence, number_present, shard_len))
}

/// Gather the survivor and missing slices, then run the plan's fused tables:
/// first the missing data from the survivors, then the missing parity from
/// the completed data
///
/// Vec gathers measured faster than stack arrays here on both native and
/// wasm; the allocator fast path beats initializing wide pointer arrays.
fn reconstruct_with_plan<Shard: ReconstructShard>(
    data_shard_count: usize,
    plan: &DecodePlan,
    shards: &mut [Shard],
    shard_len: usize,
    data_only: bool,
) -> Result<(), Error> {
    let mut sub_shards: Vec<&[u8]> = Vec::with_capacity(data_shard_count);
    let mut missing_data_slices: Vec<&mut [u8]> = Vec::with_capacity(plan.missing_data.len());
    let mut missing_parity_slices: Vec<&mut [u8]> =
        Vec::with_capacity(plan.missing_parity.len());

    for (index, shard) in shards.iter_mut().enumerate() {
        let shard_data = if index >= data_shard_count && data_only {
            shard.get().ok_or(None)
        } else {
            shard.get_or_initialize(shard_len).map_err(Some)
        };

        match shard_data {
            Ok(shard) => {
                if sub_shards.len() < data_shard_count {
                    sub_shards.push(shard);
                }
            }
            Err(None) => {}
            Err(Some(initialized)) => {
                let shard = initialized?;
                if index < data_shard_count {
                    missing_data_slices.push(shard);
                } else {
                    missing_parity_slices.push(shard);
                }
            }
        }
    }

    if !missing_data_slices.is_empty() {
        plan.data_tables.apply(&sub_shards, &mut missing_data_slices);
    }
    if data_only || missing_parity_slices.is_empty() {
        return Ok(());
    }

    // Stitch the complete data back together in index order: survivors from
    // sub_shards, recovered shards from missing_data_slices. Every present
    // data shard sits at the front of sub_shards because data indexes sort
    // before parity indexes.
    let mut all_data: Vec<&[u8]> = Vec::with_capacity(data_shard_count);
    let mut survivor = 0;
    let mut recovered = 0;
    for index in 0..data_shard_count {
        if recovered < plan.missing_data.len() && plan.missing_data[recovered] == index {
            all_data.push(&*missing_data_slices[recovered]);
            recovered += 1;
        } else {
            all_data.push(sub_shards[survivor]);
            survivor += 1;
        }
    }
    plan.parity_tables.apply(&all_data, &mut missing_parity_slices);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_data(len: usize, mut seed: u32) -> Vec<u8> {
        let mut data = vec![0u8; len];
        for b in data.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (seed >> 16) as u8;
        }
        data
    }

    fn encoded_shards(k: usize, m: usize, len: usize) -> Vec<Vec<u8>> {
        let rs = ReedSolomon::new(k, m).expect("codec should build");
        let data = deterministic_data(k * len, 0x1234_5678);
        let mut shards: Vec<Vec<u8>> = (0..k)
            .map(|j| data[j * len..(j + 1) * len].to_vec())
            .collect();
        shards.extend((0..m).map(|_| vec![0u8; len]));
        rs.encode(&mut shards).expect("encode should succeed");
        shards
    }

    #[test]
    fn encode_rows_matches_per_shard_encode() {
        let (k, m, row_len) = (6usize, 4usize, 293usize);
        let rs = ReedSolomon::new(k, m).unwrap();

        // Deterministic pseudo-random data.
        let data = deterministic_data(k * row_len, 0x1234_5678);

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

        // Full-row batched encode, allocating and into a caller buffer.
        let got = rs.encode_rows(&data, row_len);
        assert_eq!(got, expected);
        let mut buffer = vec![0u8; m * row_len];
        rs.encode_rows_into(&data, row_len, &mut buffer).unwrap();
        assert_eq!(buffer, expected);
    }

    #[test]
    fn encode_fused_matches_encode() {
        // Fused encode must be byte-identical to the per-shard encode at the
        // production shapes and a spread of shard sizes, including tail lengths
        // and both FFT strip boundaries.
        for &(k, m) in &[(7usize, 13usize), (10, 10), (20, 10), (4, 2), (6, 4), (17, 17)] {
            let rs = ReedSolomon::new(k, m).unwrap();
            for &len in &[1usize, 15, 16, 17, 31, 32, 63, 100, 1024, 10000, 65536] {
                let data = deterministic_data(k * len, 0x9E37_79B9);
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

    // encode_sep (separate data/parity slices, rse-compatible) produces the
    // same parity bytes as the combined encode, including Agave's (32,32)/987
    #[test]
    fn encode_sep_matches_encode() {
        for &(k, m) in &[(7usize, 13usize), (10, 10), (32, 32), (4, 2), (18, 6)] {
            let rs = ReedSolomon::new(k, m).unwrap();
            for &len in &[16usize, 100, 987, 1024] {
                let data = deterministic_data(k * len, 0x1234_5678);
                let data_shards: Vec<Vec<u8>> =
                    (0..k).map(|j| data[j * len..(j + 1) * len].to_vec()).collect();

                let mut combined = data_shards.clone();
                combined.extend((0..m).map(|_| vec![0u8; len]));
                rs.encode(&mut combined).unwrap();

                let mut parity = vec![vec![0u8; len]; m];
                rs.encode_sep(&data_shards, &mut parity).unwrap();

                assert_eq!(&combined[k..], &parity[..], "k={k} m={m} len={len}");
            }
        }
    }

    // runtime shapes route through the staged program and match the scalar
    // reference on every strip phase, including shortened runtime shapes
    #[test]
    fn staged_matches_scalar() {
        for &(k, m) in &[
            (12usize, 8usize),
            (14, 14),
            (16, 16),
            (18, 6),
            (24, 8),
            (32, 32),
        ] {
            let rs = ReedSolomon::new(k, m).expect("codec should build");
            for &len in &[15usize, 16, 17, 31, 32, 33, 100, 1000, 4096] {
                let data = deterministic_data(k * len, 0x51A6ED);
                let mut shards: Vec<Vec<u8>> = (0..k)
                    .map(|j| data[j * len..(j + 1) * len].to_vec())
                    .collect();
                shards.extend((0..m).map(|_| vec![0u8; len]));

                let mut expected = shards.clone();
                rs.encode_scalar(&mut expected).expect("scalar encode should succeed");
                rs.encode(&mut shards).expect("encode should succeed");
                assert_eq!(shards, expected, "k={k} m={m} len={len}");
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

    // every erasure pattern reconstructs to the original through the plan cache
    #[test]
    fn reconstruct_patterns() {
        for &(k, m) in &[(4usize, 2usize), (7, 13), (10, 10), (6, 4)] {
            for &len in &[15usize, 100, 1024] {
                let rs = ReedSolomon::new(k, m).expect("codec should build");
                let original = encoded_shards(k, m, len);

                // Walk a spread of patterns, each twice so the second call
                // exercises the cached plan.
                let patterns: Vec<Vec<usize>> = vec![
                    vec![0],
                    vec![k - 1],
                    vec![k],
                    (0..m.min(k)).collect(),
                    (k..k + m).collect(),
                    (0..m).map(|e| (e * 2) % (k + m)).collect(),
                ];
                for erased in &patterns {
                    for _ in 0..2 {
                        let mut shards = original.clone();
                        for &e in erased {
                            shards[e].fill(0);
                        }
                        let mut view: Vec<(&mut [u8], bool)> = shards
                            .iter_mut()
                            .map(|s| (s.as_mut_slice(), true))
                            .collect();
                        for &e in erased {
                            view[e].1 = false;
                        }
                        rs.reconstruct(&mut view).expect("reconstruct should succeed");
                        assert_eq!(shards, original, "k={k} m={m} len={len} erased={erased:?}");
                    }
                }
            }
        }
    }

    // a prepared decoder matches ad hoc reconstruction and rejects other patterns
    #[test]
    fn prepared_decoder() {
        let (k, m, len) = (7usize, 13usize, 200usize);
        let rs = ReedSolomon::new(k, m).expect("codec should build");
        let original = encoded_shards(k, m, len);

        let erased = [1usize, 3, 8, 15];
        let mut present = vec![true; k + m];
        for &e in &erased {
            present[e] = false;
        }
        let decoder = rs.prepare_decode(&present).expect("prepare should succeed");

        // Many stripes, one prepared pattern.
        for _ in 0..3 {
            let mut shards = original.clone();
            for &e in &erased {
                shards[e].fill(0);
            }
            let mut view: Vec<(&mut [u8], bool)> = shards
                .iter_mut()
                .map(|s| (s.as_mut_slice(), true))
                .collect();
            for &e in &erased {
                view[e].1 = false;
            }
            decoder.reconstruct(&mut view).expect("reconstruct should succeed");
            assert_eq!(shards, original);
        }

        // A different pattern must be rejected, not silently miscomputed.
        let mut shards = original.clone();
        shards[0].fill(0);
        let mut view: Vec<(&mut [u8], bool)> = shards
            .iter_mut()
            .map(|s| (s.as_mut_slice(), true))
            .collect();
        view[0].1 = false;
        assert_eq!(
            decoder.reconstruct(&mut view),
            Err(Error::InvalidShardFlags)
        );
    }

    // data only reconstruction fills data and leaves missing parity untouched
    #[test]
    fn prepared_data_only() {
        let (k, m, len) = (6usize, 4usize, 128usize);
        let rs = ReedSolomon::new(k, m).expect("codec should build");
        let original = encoded_shards(k, m, len);

        let erased = [2usize, k + 1];
        let mut present = vec![true; k + m];
        for &e in &erased {
            present[e] = false;
        }
        let decoder = rs.prepare_decode(&present).expect("prepare should succeed");

        let mut shards = original.clone();
        for &e in &erased {
            shards[e].fill(0);
        }
        let mut view: Vec<(&mut [u8], bool)> = shards
            .iter_mut()
            .map(|s| (s.as_mut_slice(), true))
            .collect();
        for &e in &erased {
            view[e].1 = false;
        }
        decoder.reconstruct_data(&mut view).expect("reconstruct should succeed");

        assert_eq!(shards[2], original[2]);
        assert_eq!(shards[k + 1], vec![0u8; len], "missing parity must stay untouched");
    }

    // contiguous rows reconstruct in place through the same plans
    #[test]
    fn reconstruct_rows_roundtrip() {
        let (k, m, row_len) = (10usize, 10usize, 257usize);
        let rs = ReedSolomon::new(k, m).expect("codec should build");
        let original = encoded_shards(k, m, row_len);

        let mut rows = Vec::with_capacity((k + m) * row_len);
        for shard in &original {
            rows.extend_from_slice(shard);
        }
        let pristine = rows.clone();

        let erased = [0usize, 4, 12, 19];
        let mut present = vec![true; k + m];
        for &e in &erased {
            present[e] = false;
            rows[e * row_len..(e + 1) * row_len].fill(0);
        }

        rs.reconstruct_rows(&mut rows, row_len, &present)
            .expect("reconstruct should succeed");
        assert_eq!(rows, pristine);
    }

    // verify accepts valid stripes and pinpoints corruption in any block
    #[test]
    fn verify_blocks() {
        let (k, m) = (4usize, 3usize);
        // Lengths on both sides of the verify block size, including a tail.
        for &len in &[100usize, VERIFY_BLOCK_SIZE, VERIFY_BLOCK_SIZE + 17] {
            let rs = ReedSolomon::new(k, m).expect("codec should build");
            let shards = encoded_shards(k, m, len);
            assert!(rs.verify(&shards).expect("verify should run"), "len={len}");

            // One flipped byte per region must fail, wherever it hides.
            for &position in &[0usize, len / 2, len - 1] {
                let mut corrupted = shards.clone();
                corrupted[k][position] ^= 0x01;
                assert!(
                    !rs.verify(&corrupted).expect("verify should run"),
                    "len={len} position={position}"
                );
            }
        }
    }
}
