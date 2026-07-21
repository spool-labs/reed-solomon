//! Reed-Solomon erasure coder over the GF((2^8)^2) tower field, for outer
//! distribution shapes that need more than 256 shards.
//!
//! # Wire
//! A shard is a byte buffer of even length `2N`, holding `N` field elements in
//! split-plane layout: the `N` low bytes first, then the `N` high bytes. Element
//! `s` of a shard is `(shard[N + s] as high, shard[s] as low)`.
//!
//! # Kernels
//! Multiplying a variable `(high, low)` pair by a fixed coefficient is a 2x2
//! matrix of GF(2^8) constant multiplies (see [`galois16::const_matrix`]). So an
//! `m x k` tower generator, applied over the split planes, is exactly an
//! `2m x 2k` GF(2^8) matrix product: the `2k` input planes map to the `2m` output
//! planes. [`RowTables16`] assembles that expanded matrix once and hands it to
//! the existing [`RowTables`] fused kernels, which already stream, tile, and
//! finish with an overlapped vector tail. No tower-specific SIMD is written.

use std::sync::{Arc, RwLock};

use crate::errors::Error;
use crate::galois16;
use crate::matrix::Matrix;

// aarch64 (no pinned scalar) runs the fused Karatsuba tower kernel; every other
// target runs the dense 2m x 2k expansion through the shared GF(2^8) kernels.
#[cfg(gf16_neon_enabled)]
use crate::gf::tables::nibble_pair;
#[cfg(gf16_neon_enabled)]
use crate::gf::tower_neon;
#[cfg(not(gf16_neon_enabled))]
use crate::gf::tables::RowTables;

/// Verify recomputes parity a symbol block at a time so scratch stays cache
/// resident. Counted in field elements; each element is two shard bytes.
const VERIFY_BLOCK_SYMBOLS: usize = 16 * 1024;

/// Decode plans kept per codec, same rationale as the GF(2^8) coder.
const DECODE_PLAN_CACHE_LIMIT: usize = 64;

use crate::reedsolomon::ReconstructShard;

/// The `m x k` tower generator rows plus the kernel tables that apply them over
/// split planes: fused Karatsuba constants on aarch64, the dense `2m x 2k`
/// GF(2^8) expansion elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct RowTables16 {
    data_count: usize,
    rows: Vec<Vec<u16>>,

    #[cfg(gf16_neon_enabled)]
    kara: KaraTables,
    // Expanded generator over planes, ordered [all low planes, all high planes]
    // on both input and output sides.
    #[cfg(not(gf16_neon_enabled))]
    expanded: RowTables,
}

/// Per-constant nibble tables for the three Karatsuba products `c_lo`,
/// `c_hi + c_lo`, and `c_hi * L`, each `m*k*32` bytes at index `(j*k+i)*32`
/// (16-byte low table then 16-byte high table).
#[cfg(gf16_neon_enabled)]
#[derive(Clone, Debug)]
struct KaraTables {
    c0: Vec<u8>,
    c1: Vec<u8>,
    c2: Vec<u8>,
}

#[cfg(gf16_neon_enabled)]
impl KaraTables {
    fn new(rows: &[Vec<u16>], data_count: usize) -> KaraTables {
        let output_count = rows.len();
        let mut c0 = vec![0u8; output_count * data_count * 32];
        let mut c1 = vec![0u8; output_count * data_count * 32];
        let mut c2 = vec![0u8; output_count * data_count * 32];
        for (j, row) in rows.iter().enumerate() {
            debug_assert_eq!(row.len(), data_count);
            for (i, &coefficient) in row.iter().enumerate() {
                let c_high = galois16::high(coefficient);
                let c_low = galois16::low(coefficient);
                let (k0, k1, k2) =
                    (c_low, c_high ^ c_low, crate::galois::mul(c_high, galois16::TOWER_L));
                let base = (j * data_count + i) * 32;
                c0[base..base + 32].copy_from_slice(&nibble_pair(k0));
                c1[base..base + 32].copy_from_slice(&nibble_pair(k1));
                c2[base..base + 32].copy_from_slice(&nibble_pair(k2));
            }
        }
        KaraTables { c0, c1, c2 }
    }
}

impl RowTables16 {
    /// Build the kernel tables from `m x k` tower coefficient rows.
    pub(crate) fn new(rows: Vec<Vec<u16>>, data_count: usize) -> RowTables16 {
        #[cfg(gf16_neon_enabled)]
        let kara = KaraTables::new(&rows, data_count);

        #[cfg(not(gf16_neon_enabled))]
        let expanded = {
            // Input plane columns: low_i at `i`, high_i at `data_count + i`.
            // Output plane rows: low_j at `j`, high_j at `output_count + j`.
            let output_count = rows.len();
            let mut e: Vec<Vec<u8>> =
                (0..2 * output_count).map(|_| vec![0u8; 2 * data_count]).collect();
            for (j, row) in rows.iter().enumerate() {
                debug_assert_eq!(row.len(), data_count);
                for (i, &coefficient) in row.iter().enumerate() {
                    // [m00, m01, m10, m11] with
                    //   out_high = m00*a_high + m01*a_low
                    //   out_low  = m10*a_high + m11*a_low
                    let [m00, m01, m10, m11] = galois16::const_matrix(coefficient);
                    e[j][i] = m11;
                    e[j][data_count + i] = m10;
                    e[output_count + j][i] = m01;
                    e[output_count + j][data_count + i] = m00;
                }
            }
            RowTables::new(e, 2 * data_count)
        };

        RowTables16 {
            data_count,
            rows,
            #[cfg(gf16_neon_enabled)]
            kara,
            #[cfg(not(gf16_neon_enabled))]
            expanded,
        }
    }

    /// The raw tower coefficient rows.
    pub(crate) fn rows(&self) -> &[Vec<u16>] {
        &self.rows
    }

    /// Apply the generator to whole shards, overwriting the outputs. Every shard
    /// shares one even length; each is split into its low and high planes and
    /// presented as `[low planes, high planes]` to `apply_planes`.
    pub(crate) fn apply<In: AsRef<[u8]>, Out: AsMut<[u8]>>(&self, inputs: &[In], outputs: &mut [Out]) {
        if outputs.is_empty() {
            return;
        }
        debug_assert_eq!(inputs.len(), self.data_count);
        let plane_len = inputs[0].as_ref().len() / 2;

        let mut plane_inputs: Vec<&[u8]> = Vec::with_capacity(2 * self.data_count);
        for shard in inputs {
            plane_inputs.push(&shard.as_ref()[..plane_len]);
        }
        for shard in inputs {
            plane_inputs.push(&shard.as_ref()[plane_len..]);
        }

        let mut low_outputs: Vec<&mut [u8]> = Vec::with_capacity(2 * self.rows.len());
        let mut high_outputs: Vec<&mut [u8]> = Vec::with_capacity(self.rows.len());
        for shard in outputs.iter_mut() {
            let (low, high) = shard.as_mut().split_at_mut(plane_len);
            low_outputs.push(low);
            high_outputs.push(high);
        }
        low_outputs.append(&mut high_outputs);

        self.apply_planes(&plane_inputs, &mut low_outputs);
    }

    /// Apply the generator to plane slices already in `[low planes, high planes]`
    /// order on both sides (whole shards, or symbol sub-ranges from verify).
    /// Routes to the fused Karatsuba kernel on aarch64, the dense expansion
    /// elsewhere, without reshuffling the planes.
    pub(crate) fn apply_planes(&self, plane_inputs: &[&[u8]], plane_outputs: &mut [&mut [u8]]) {
        debug_assert_eq!(plane_inputs.len(), 2 * self.data_count);
        debug_assert_eq!(plane_outputs.len(), 2 * self.rows.len());

        #[cfg(gf16_neon_enabled)]
        {
            let (low_inputs, high_inputs) = plane_inputs.split_at(self.data_count);
            let (low_outputs, high_outputs) = plane_outputs.split_at_mut(self.rows.len());
            tower_neon::encode(
                &self.kara.c0,
                &self.kara.c1,
                &self.kara.c2,
                &self.rows,
                low_inputs,
                high_inputs,
                low_outputs,
                high_outputs,
            );
        }

        #[cfg(not(gf16_neon_enabled))]
        self.expanded.apply(plane_inputs, plane_outputs);
    }
}

/// Bitmask of which shard slots are present, heap sized to the shard count so no
/// 256-shard limit is baked in.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PresenceMask(Box<[u64]>);

impl PresenceMask {
    fn empty(total_shards: usize) -> PresenceMask {
        PresenceMask(vec![0u64; total_shards.div_ceil(64)].into_boxed_slice())
    }

    fn set(&mut self, index: usize) {
        self.0[index / 64] |= 1u64 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        self.0[index / 64] & (1u64 << (index % 64)) != 0
    }
}

/// Everything reconstruction needs for one erasure pattern, built once per
/// pattern and reused.
#[derive(Debug)]
struct DecodePlan {
    presence: PresenceMask,
    missing_data: Vec<usize>,
    missing_parity: Vec<usize>,
    data_tables: RowTables16,
    parity_tables: RowTables16,
}

type PlanCache = Vec<(PresenceMask, Arc<DecodePlan>)>;

/// Reed-Solomon erasure coder over GF((2^8)^2).
#[derive(Debug)]
pub struct ReedSolomon16 {
    data_shard_count: usize,
    parity_shard_count: usize,
    total_shard_count: usize,

    matrix: Matrix<u16>,
    parity: RowTables16,
    // Encode routes through the compiled FFT program when its O(n log k) beats
    // the matrix O(k*m) for this shape; reconstruct always stays on the matrix
    // plans. Compiled once here, then replayed per encode over a strip arena
    // the executor allocates per call.
    staged: Option<Arc<crate::fft16::StagedProgram16>>,

    decode_plans: RwLock<PlanCache>,
}

impl ReedSolomon16 {
    /// Builds the systematic generator as `vandermonde(total, data) * invert(top block)`.
    fn build_matrix(data_shards: usize, total_shards: usize) -> Result<Matrix<u16>, Error> {
        let vandermonde = Matrix::<u16>::vandermonde(total_shards, data_shards);
        let top = vandermonde.sub_matrix(0, 0, data_shards, data_shards);
        Ok(vandermonde.multiply(&top.invert()?))
    }

    /// Creates a new coder.
    ///
    /// Returns `Error::TooFewDataShards` if `data_shards == 0`,
    /// `Error::TooFewParityShards` if `parity_shards == 0`,
    /// `Error::TooManyShards` if `data_shards + parity_shards > 65536`.
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<ReedSolomon16, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > 65536 {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;
        let matrix = Self::build_matrix(data_shards, total_shards)?;

        let mut parity_rows: Vec<Vec<u16>> = Vec::with_capacity(parity_shards);
        for output in data_shards..total_shards {
            parity_rows.push(matrix.get_row(output).to_vec());
        }

        // Compile the staged FFT program and keep it where its measured cost
        // beats the matrix kernels for this shape.
        let staged = if (2..=crate::fft16::MAX_FFT_DATA_SHARDS).contains(&data_shards) {
            let program = crate::fft16::build_staged_program16(data_shards, parity_shards);
            crate::fft16::profitable(&program, data_shards, parity_shards)
                .then(|| Arc::new(program))
        } else {
            None
        };

        Ok(ReedSolomon16 {
            data_shard_count: data_shards,
            parity_shard_count: parity_shards,
            total_shard_count: total_shards,
            matrix,
            parity: RowTables16::new(parity_rows, data_shards),
            staged,
            decode_plans: RwLock::new(Vec::new()),
        })
    }

    /// Bench and test introspection: encode through the matrix kernels
    /// regardless of the route, for calibrating the FFT/matrix crossover.
    #[doc(hidden)]
    pub fn encode_sep_matrix<In: AsRef<[u8]>, Out: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        data: &[In],
        parity: &mut [Out],
    ) -> Result<(), Error> {
        self.check_shards(
            data.iter()
                .map(|s| s.as_ref())
                .chain(parity.iter().map(|s| s.as_ref())),
        )?;
        self.parity.apply(data, parity);
        Ok(())
    }

    /// Bench and test introspection: which encode path this shape takes.
    #[doc(hidden)]
    pub fn encode_route(&self) -> &'static str {
        if self.staged.is_some() {
            "fft"
        } else {
            "matrix"
        }
    }

    /// Shared encode core: the compiled FFT program for shapes where it beats
    /// the matrix kernels, otherwise the fused matrix path. Byte-identical
    /// either way.
    fn encode_into<In: AsRef<[u8]>, Out: AsMut<[u8]>>(&self, input: &[In], output: &mut [Out]) {
        if let Some(program) = &self.staged {
            crate::fft16::encode(program, input, output);
        } else {
            self.parity.apply(input, output);
        }
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
    /// `parity_shard_count` parity shards, all of equal even length.
    pub fn encode<T: AsRef<[u8]> + AsMut<[u8]>>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.check_piece_count_all(shards.len())?;
        self.check_shards(shards.iter().map(|s| s.as_ref()))?;

        let (input, output) = shards.split_at_mut(self.data_shard_count);
        self.encode_into(input, output);
        Ok(())
    }

    /// Encode with data and parity in separate slices; `data` is read-only and
    /// each `parity` shard is overwritten. All shards share one even length.
    pub fn encode_sep<In: AsRef<[u8]>, Out: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        data: &[In],
        parity: &mut [Out],
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
        self.check_shards(
            data.iter()
                .map(|s| s.as_ref())
                .chain(parity.iter().map(|s| s.as_ref())),
        )?;
        self.encode_into(data, parity);
        Ok(())
    }

    /// Checks whether the parity shards are correct for the given data, a symbol
    /// block at a time, stopping at the first mismatch.
    pub fn verify<T: AsRef<[u8]>>(&self, shards: &[T]) -> Result<bool, Error> {
        self.check_piece_count_all(shards.len())?;
        let shard_len = self.check_shards(shards.iter().map(|s| s.as_ref()))?;
        let symbol_count = shard_len / 2;

        let data = &shards[..self.data_shard_count];
        let parity = &shards[self.data_shard_count..];
        let block = symbol_count.min(VERIFY_BLOCK_SYMBOLS);
        // One mini scratch plane pair per parity output.
        let mut scratch: Vec<Vec<u8>> = vec![vec![0u8; 2 * block]; self.parity_shard_count];

        let mut start = 0;
        while start < symbol_count {
            let end = (start + block).min(symbol_count);
            let width = end - start;

            // Input planes for this symbol range, no copy: low_i[start..end] and
            // high_i[start..end] carved straight out of each data shard.
            let mut plane_inputs: Vec<&[u8]> = Vec::with_capacity(2 * self.data_shard_count);
            for shard in data {
                plane_inputs.push(&shard.as_ref()[start..end]);
            }
            for shard in data {
                plane_inputs.push(&shard.as_ref()[symbol_count + start..symbol_count + end]);
            }

            // low_outputs absorbs the highs below, so it is sized for both.
            let mut low_outputs: Vec<&mut [u8]> =
                Vec::with_capacity(2 * self.parity_shard_count);
            let mut high_outputs: Vec<&mut [u8]> = Vec::with_capacity(self.parity_shard_count);
            for block_shard in scratch.iter_mut() {
                let (low, high) = block_shard.split_at_mut(block);
                low_outputs.push(&mut low[..width]);
                high_outputs.push(&mut high[..width]);
            }
            low_outputs.append(&mut high_outputs);
            self.parity.apply_planes(&plane_inputs, &mut low_outputs);

            for (index, block_shard) in scratch.iter().enumerate() {
                let parity_shard = parity[index].as_ref();
                if block_shard[..width] != parity_shard[start..end] {
                    return Ok(false);
                }
                if block_shard[block..block + width]
                    != parity_shard[symbol_count + start..symbol_count + end]
                {
                    return Ok(false);
                }
            }
            start = end;
        }
        Ok(true)
    }

    /// Reconstructs all missing shards (data and parity) in place.
    pub fn reconstruct<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, false)
    }

    /// Reconstructs only the missing data shards in place.
    pub fn reconstruct_data<T: ReconstructShard>(&self, shards: &mut [T]) -> Result<(), Error> {
        self.reconstruct_internal(shards, true)
    }

    /// Prepare a decoder for one presence pattern, `true` per present shard.
    pub fn prepare_decode(&self, present: &[bool]) -> Result<PreparedDecoder16, Error> {
        if present.len() != self.total_shard_count {
            return Err(Error::InvalidShardFlags);
        }

        let mut presence = PresenceMask::empty(self.total_shard_count);
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

        Ok(PreparedDecoder16 {
            data_shard_count: self.data_shard_count,
            total_shard_count: self.total_shard_count,
            plan: self.plan_for(presence)?,
        })
    }

    /// Reconstructs missing rows inside one contiguous buffer of
    /// `total_shard_count` rows of `row_len` bytes each.
    pub fn reconstruct_rows(
        &self,
        rows: &mut [u8],
        row_len: usize,
        present: &[bool],
    ) -> Result<(), Error> {
        if present.len() != self.total_shard_count {
            return Err(Error::InvalidShardFlags);
        }
        if row_len == 0 || !row_len.is_multiple_of(2) {
            return Err(Error::IncorrectShardSize);
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

    fn data_decode_matrix(&self, valid_indices: &[usize]) -> Result<Matrix<u16>, Error> {
        let mut sub_matrix = Matrix::<u16>::new(self.data_shard_count, self.data_shard_count);
        for (sub_row, &valid_index) in valid_indices.iter().enumerate() {
            for c in 0..self.data_shard_count {
                sub_matrix.set(sub_row, c, self.matrix.get(valid_index, c));
            }
        }
        sub_matrix.invert()
    }

    fn build_plan(&self, presence: PresenceMask) -> Result<DecodePlan, Error> {
        let data_shard_count = self.data_shard_count;

        let mut valid_indices: Vec<usize> = Vec::with_capacity(data_shard_count);
        let mut missing_data: Vec<usize> = Vec::new();
        let mut missing_parity: Vec<usize> = Vec::new();
        for index in 0..self.total_shard_count {
            if presence.get(index) {
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
        let mut data_rows: Vec<Vec<u16>> = Vec::with_capacity(missing_data.len());
        for &index in &missing_data {
            data_rows.push(decode_matrix.get_row(index).to_vec());
        }
        let mut parity_rows: Vec<Vec<u16>> = Vec::with_capacity(missing_parity.len());
        for &index in &missing_parity {
            parity_rows.push(self.parity.rows()[index - data_shard_count].to_vec());
        }

        Ok(DecodePlan {
            presence,
            missing_data,
            missing_parity,
            data_tables: RowTables16::new(data_rows, data_shard_count),
            parity_tables: RowTables16::new(parity_rows, data_shard_count),
        })
    }

    /// Fetch the cached plan for a presence pattern, building it on first use.
    /// The hit path takes only a shared read lock so many threads reconstruct
    /// concurrently.
    fn plan_for(&self, presence: PresenceMask) -> Result<Arc<DecodePlan>, Error> {
        {
            let plans = self.decode_plans.read().unwrap_or_else(|poisoned| poisoned.into_inner());
            for (mask, plan) in plans.iter() {
                if *mask == presence {
                    return Ok(plan.clone());
                }
            }
        }

        let plan = Arc::new(self.build_plan(presence.clone())?);
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

        let (presence, number_present, shard_len) = scan_shards(shards, self.total_shard_count)?;
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

    fn check_piece_count_all(&self, count: usize) -> Result<(), Error> {
        if count < self.total_shard_count {
            Err(Error::TooFewShards)
        } else if count > self.total_shard_count {
            Err(Error::TooManyShards)
        } else {
            Ok(())
        }
    }

    /// All shards share one nonzero even length; returns it.
    fn check_shards<'a>(&self, shards: impl Iterator<Item = &'a [u8]>) -> Result<usize, Error> {
        let mut shard_len: Option<usize> = None;
        for shard in shards {
            note_shard_len(shard.len(), &mut shard_len)?;
        }
        shard_len.ok_or(Error::EmptyShard)
    }
}

impl Clone for ReedSolomon16 {
    fn clone(&self) -> ReedSolomon16 {
        let plans = self.decode_plans.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        ReedSolomon16 {
            data_shard_count: self.data_shard_count,
            parity_shard_count: self.parity_shard_count,
            total_shard_count: self.total_shard_count,
            matrix: self.matrix.clone(),
            parity: self.parity.clone(),
            staged: self.staged.clone(),
            decode_plans: RwLock::new(plans.clone()),
        }
    }
}

impl PartialEq for ReedSolomon16 {
    fn eq(&self, other: &Self) -> bool {
        self.data_shard_count == other.data_shard_count
            && self.parity_shard_count == other.parity_shard_count
    }
}

/// A decoder prepared for one presence pattern; pays the inversion once and is
/// reused across stripes with the same erasure pattern.
#[derive(Clone, Debug)]
pub struct PreparedDecoder16 {
    data_shard_count: usize,
    total_shard_count: usize,
    plan: Arc<DecodePlan>,
}

impl PreparedDecoder16 {
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

        let (presence, number_present, shard_len) = scan_shards(shards, self.total_shard_count)?;
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

/// Folds one shard length into the running shared length, rejecting empty, odd,
/// and mismatched sizes. The wire carries two bytes per field element, so an odd
/// length cannot be a whole number of symbols.
fn note_shard_len(len: usize, shard_len: &mut Option<usize>) -> Result<(), Error> {
    if len == 0 {
        return Err(Error::EmptyShard);
    }
    if !len.is_multiple_of(2) {
        return Err(Error::IncorrectShardSize);
    }
    match shard_len {
        Some(existing) if *existing != len => Err(Error::IncorrectShardSize),
        slot => {
            *slot = Some(len);
            Ok(())
        }
    }
}

/// One pass over the shards: presence mask, present count, and the shared even
/// shard length, rejecting empty, odd, and mismatched sizes.
fn scan_shards<Shard: ReconstructShard>(
    shards: &[Shard],
    total_shards: usize,
) -> Result<(PresenceMask, usize, Option<usize>), Error> {
    let mut presence = PresenceMask::empty(total_shards);
    let mut number_present = 0;
    let mut shard_len: Option<usize> = None;
    for (index, shard) in shards.iter().enumerate() {
        if let Some(len) = shard.len() {
            note_shard_len(len, &mut shard_len)?;
            presence.set(index);
            number_present += 1;
        }
    }
    Ok((presence, number_present, shard_len))
}

/// Gather survivors and missing slices, then run the plan's tables: missing data
/// from the survivors, then missing parity from the completed data. Identical in
/// shape to the GF(2^8) path; the plane split lives inside `RowTables16::apply`.
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
        let rs = ReedSolomon16::new(k, m).expect("codec should build");
        let data = deterministic_data(k * len, 0x1234_5678);
        let mut shards: Vec<Vec<u8>> = (0..k)
            .map(|j| data[j * len..(j + 1) * len].to_vec())
            .collect();
        shards.extend((0..m).map(|_| vec![0u8; len]));
        rs.encode(&mut shards).expect("encode should succeed");
        shards
    }

    // The independent oracle: parity computed symbol by symbol through the
    // scalar tower multiply, over the split-plane wire.
    fn symbolwise_parity(rows: &[Vec<u16>], data: &[Vec<u8>], len: usize) -> Vec<Vec<u8>> {
        let symbol_count = len / 2;
        let k = data.len();
        let mut parity = vec![vec![0u8; len]; rows.len()];
        for (j, row) in rows.iter().enumerate() {
            for s in 0..symbol_count {
                let mut acc = 0u16;
                for i in 0..k {
                    let symbol =
                        ((data[i][symbol_count + s] as u16) << 8) | data[i][s] as u16;
                    acc ^= galois16::mul(row[i], symbol);
                }
                parity[j][s] = (acc & 0xff) as u8;
                parity[j][symbol_count + s] = (acc >> 8) as u8;
            }
        }
        parity
    }

    // Fused/host encode agrees with the symbol-wise scalar reference across
    // shapes and even lengths, including sub-plane and boundary sizes.
    #[test]
    fn encode_matches_symbolwise() {
        for &(k, m) in &[(4usize, 2usize), (7, 13), (10, 10), (17, 33)] {
            let rs = ReedSolomon16::new(k, m).unwrap();
            for &len in &[2usize, 30, 32, 34, 62, 64, 66, 200, 2048, 65536] {
                let data = deterministic_data(k * len, 0x9E37_79B9);
                let data_shards: Vec<Vec<u8>> = (0..k)
                    .map(|j| data[j * len..(j + 1) * len].to_vec())
                    .collect();

                let mut shards = data_shards.clone();
                shards.extend((0..m).map(|_| vec![0u8; len]));
                rs.encode(&mut shards).unwrap();

                let want = symbolwise_parity(rs.parity.rows(), &data_shards, len);
                assert_eq!(&shards[k..], &want[..], "k={k} m={m} len={len}");
            }
        }
    }

    // FFT-routed shapes: encode runs through the additive FFT and must stay
    // byte-identical to the symbol-wise scalar oracle (hence the matrix wire),
    // across the executor's strip boundary and its tails.
    #[test]
    fn fft_encode_matches_symbolwise() {
        for &(k, m) in &[(32usize, 96usize), (48, 144), (86, 170), (128, 128)] {
            let rs = ReedSolomon16::new(k, m).unwrap();
            assert_eq!(rs.encode_route(), "fft", "k={k} m={m} should route to fft");
            for &len in &[2usize, 34, 200, 2 * 2048, 2 * 2048 + 34, 20000] {
                let data = deterministic_data(k * len, 0x51A6_ED01);
                let data_shards: Vec<Vec<u8>> =
                    (0..k).map(|j| data[j * len..(j + 1) * len].to_vec()).collect();
                let mut shards = data_shards.clone();
                shards.extend((0..m).map(|_| vec![0u8; len]));
                rs.encode(&mut shards).unwrap();
                let want = symbolwise_parity(rs.parity.rows(), &data_shards, len);
                assert_eq!(&shards[k..], &want[..], "k={k} m={m} len={len}");
            }
        }
    }

    // Frozen wire vector: a fixed shape and input must always produce these exact
    // parity bytes. Locks L, the plane order, and the generator construction.
    #[test]
    fn golden_wire_vector() {
        let rs = ReedSolomon16::new(3, 2).unwrap();
        // 2 symbols per shard: [low_0, low_1, high_0, high_1].
        let mut shards: Vec<Vec<u8>> = vec![
            vec![0x01, 0x02, 0x03, 0x04],
            vec![0x05, 0x06, 0x07, 0x08],
            vec![0x09, 0x0a, 0x0b, 0x0c],
            vec![0; 4],
            vec![0; 4],
        ];
        rs.encode(&mut shards).unwrap();
        assert_eq!(shards[3], [13, 14, 15, 0], "parity row 0 drifted");
        assert_eq!(shards[4], [17, 18, 19, 84], "parity row 1 drifted");
    }

    // Every erasure pattern up to m losses reconstructs, through the plan cache
    // (each pattern run twice), including shapes past the 256-shard wall.
    #[test]
    fn reconstruct_patterns() {
        for &(k, m) in &[(4usize, 2usize), (17, 33), (10, 10), (100, 200)] {
            for &len in &[2usize, 34, 512] {
                let original = encoded_shards(k, m, len);
                let total = k + m;
                let patterns: Vec<Vec<usize>> = vec![
                    vec![0],
                    vec![k - 1],
                    vec![k],
                    (0..m.min(k)).collect(),
                    (k..k + m).collect(),
                    (0..m).map(|e| (e * 3) % total).collect(),
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
                        rs_reconstruct(k, m, &mut view);
                        assert_eq!(shards, original, "k={k} m={m} len={len} erased={erased:?}");
                    }
                }
            }
        }
    }

    fn rs_reconstruct(k: usize, m: usize, view: &mut [(&mut [u8], bool)]) {
        let rs = ReedSolomon16::new(k, m).unwrap();
        rs.reconstruct(view).expect("reconstruct should succeed");
    }

    // A prepared decoder matches ad hoc reconstruction and rejects other patterns.
    #[test]
    fn prepared_decoder() {
        let (k, m, len) = (17usize, 33usize, 200usize);
        let rs = ReedSolomon16::new(k, m).unwrap();
        let original = encoded_shards(k, m, len);

        let erased = [1usize, 3, 20, 40];
        let mut present = vec![true; k + m];
        for &e in &erased {
            present[e] = false;
        }
        let decoder = rs.prepare_decode(&present).unwrap();

        for _ in 0..3 {
            let mut shards = original.clone();
            for &e in &erased {
                shards[e].fill(0);
            }
            let mut view: Vec<(&mut [u8], bool)> =
                shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
            for &e in &erased {
                view[e].1 = false;
            }
            decoder.reconstruct(&mut view).unwrap();
            assert_eq!(shards, original);
        }

        let mut shards = original.clone();
        shards[0].fill(0);
        let mut view: Vec<(&mut [u8], bool)> =
            shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
        view[0].1 = false;
        assert_eq!(decoder.reconstruct(&mut view), Err(Error::InvalidShardFlags));
    }

    // Data-only reconstruction fills data and leaves missing parity untouched.
    #[test]
    fn prepared_data_only() {
        let (k, m, len) = (6usize, 4usize, 128usize);
        let rs = ReedSolomon16::new(k, m).unwrap();
        let original = encoded_shards(k, m, len);

        let erased = [2usize, k + 1];
        let mut present = vec![true; k + m];
        for &e in &erased {
            present[e] = false;
        }
        let decoder = rs.prepare_decode(&present).unwrap();

        let mut shards = original.clone();
        for &e in &erased {
            shards[e].fill(0);
        }
        let mut view: Vec<(&mut [u8], bool)> =
            shards.iter_mut().map(|s| (s.as_mut_slice(), true)).collect();
        for &e in &erased {
            view[e].1 = false;
        }
        decoder.reconstruct_data(&mut view).unwrap();

        assert_eq!(shards[2], original[2]);
        assert_eq!(shards[k + 1], vec![0u8; len], "missing parity must stay untouched");
    }

    // verify accepts valid stripes and pinpoints corruption in either plane and
    // across the symbol-block boundary.
    #[test]
    fn verify_blocks() {
        let (k, m) = (4usize, 3usize);
        for &len in &[200usize, 2 * VERIFY_BLOCK_SYMBOLS, 2 * VERIFY_BLOCK_SYMBOLS + 34] {
            let rs = ReedSolomon16::new(k, m).unwrap();
            let shards = encoded_shards(k, m, len);
            assert!(rs.verify(&shards).unwrap(), "len={len}");

            for &position in &[0usize, len / 2, len - 1] {
                let mut corrupted = shards.clone();
                corrupted[k][position] ^= 0x01;
                assert!(!rs.verify(&corrupted).unwrap(), "len={len} position={position}");
            }
        }
    }

    // Odd shard lengths are rejected; the wire is field elements, two bytes each.
    #[test]
    fn odd_length_rejected() {
        let rs = ReedSolomon16::new(2, 2).unwrap();
        let mut shards: Vec<Vec<u8>> = vec![vec![0u8; 5]; 4];
        assert_eq!(rs.encode(&mut shards), Err(Error::IncorrectShardSize));
    }

    // encode_sep produces the same parity as the combined encode.
    #[test]
    fn encode_sep_matches_encode() {
        for &(k, m) in &[(4usize, 2usize), (17, 33)] {
            let rs = ReedSolomon16::new(k, m).unwrap();
            for &len in &[34usize, 512] {
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
}
