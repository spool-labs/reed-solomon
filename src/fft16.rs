//! GF((2^8)^2) additive (Lin-Chung-Han novel-basis) FFT encode, compiled per
//! shape.
//!
//! Generalizes the staged-program machinery in `fft.rs` to the tower field,
//! using the standard basis {2^0..2^15} so evaluation point `i` is the u16 `i`.
//! The systematic RS code is basis-independent, so an FFT systematic encode is
//! byte-identical to `ReedSolomon16`'s Vandermonde wire (checked in
//! `reedsolomon16::tests`).
//!
//! A shape compiles once, at codec construction, into a [`StagedProgram16`]:
//! interpolation of the data polynomial, then folded aligned chunks covering
//! exactly the parity points, lowered to whole-transform stages plus glue ops
//! over a virtual register file. Executors replay the program strip by strip
//! over one arena allocated per encode call; every tower constant lowers to
//! three GF(2^8) bytes (the Karatsuba products), so the SIMD executors index
//! the same static nibble tables the GF(2^8) kernels use.
//!
//! Iteration order contract, shared by the lowering, the interpreters, and the
//! SIMD executors: a forward FFT walks levels from high to low, an inverse FFT
//! from low to high; within a level, blocks walk in ascending base order with
//! one constant per block, and butterflies in ascending index order.

use std::sync::OnceLock;

use crate::galois16;

/// Largest data-shard count the FFT path serves; beyond this the compiled
/// register file grows past cache and the routing falls back to the matrix
/// kernels.
pub const MAX_FFT_DATA_SHARDS: usize = 1024;

/// Transform levels: enough to cover `MAX_FFT_DATA_SHARDS` (block <= 2^LEVELS).
const LEVELS: usize = MAX_FFT_DATA_SHARDS.trailing_zeros() as usize + 1;

/// Staged register files above this size fall back to the matrix path. The
/// executors hold `register_count` strip pairs in one arena; past this count
/// the arena outgrows the cache tier the op-major streaming leans on.
pub(crate) const MAX_STAGED_REGISTERS16: usize = 2048;

/// WHAT[j][x]: normalized subspace polynomial vanishing on span{2^0..2^{j-1}},
/// scaled so WHAT[j][2^j] == 1. LEVELS x 65536 u16, built once per process.
fn what() -> &'static [Vec<u16>] {
    static WHAT: OnceLock<Vec<Vec<u16>>> = OnceLock::new();
    WHAT.get_or_init(|| {
        // W_0(x) = x ; W_{j+1}(x) = W_j(x) * (W_j(x) + W_j(2^j))
        let mut w = vec![vec![0u16; 65536]; LEVELS];
        for (x, slot) in w[0].iter_mut().enumerate() {
            *slot = x as u16;
        }
        for j in 0..LEVELS - 1 {
            let pivot = w[j][1usize << j];
            let (lower, upper) = w.split_at_mut(j + 1);
            for (dst, &value) in upper[0].iter_mut().zip(lower[j].iter()) {
                *dst = galois16::mul(value, value ^ pivot);
            }
        }
        // Normalize each level by the inverse of its pivot, in place: the
        // recurrence above is finished, so no level is read again.
        for j in 0..LEVELS {
            let inverse = galois16::inv(w[j][1usize << j]);
            for slot in w[j].iter_mut() {
                *slot = galois16::mul(*slot, inverse);
            }
        }
        w
    })
}

/// Whether a compiled program should carry this shape's encode over the
/// fused matrix kernels.
///
/// Measured at 4 MiB shards by `gf16-bench --bin sweep`: on aarch64 the
/// compiled program beat the matrix tiles at every tested shape, down to
/// (7,13), so any program that fits the executors routes to it. On x86 the
/// GFNI affine tiles do one coefficient multiply about 4.5x cheaper than one
/// butterfly, which keeps the smallest shapes on the matrix; the knife-edge
/// shapes on Zen 5 sit exactly where the multiply-count rule predicts,
/// (11,20) at ratio 4.1 to the matrix and (17,33) at 5.0 to the FFT.
pub(crate) fn profitable(
    program: &StagedProgram16,
    data_shards: usize,
    parity_shards: usize,
) -> bool {
    // cfg! rather than #[cfg] keeps the shape arguments live on every target,
    // so the aarch64 arm still folds away without a lint suppression.
    program.register_count <= MAX_STAGED_REGISTERS16
        && (cfg!(target_arch = "aarch64")
            || 9 * program.mult_count < 2 * data_shards * parity_shards)
}

/// One field operation over the virtual register file.
///
/// The builder emits only the two xor forms against an all-zero backdrop;
/// lowering rewrites first writes into the full-write forms so that registers
/// never depend on their starting contents.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FftOp16 {
    /// registers[dst] ^= c * registers[src]
    MulXor { dst: u16, src: u16, c: u16 },
    /// registers[dst] ^= registers[src]
    Xor { dst: u16, src: u16 },
    /// registers[dst] = c * registers[src]
    MulCopy { dst: u16, src: u16, c: u16 },
    /// registers[dst] = registers[src]
    Copy { dst: u16, src: u16 },
}

/// One build step: a whole transform over a contiguous register range, or a
/// single glue op.
#[derive(Clone, Copy, Debug)]
enum Item {
    Transform { inverse: bool, start: u16, size: u16, beta: usize },
    Glue(FftOp16),
}

/// One stage of a compiled encode.
#[derive(Clone, Copy, Debug)]
pub(crate) enum StagedStage16 {
    /// In-place transform of `size` registers starting at `start`, its block
    /// constants at `consts_start` in the constant pool
    Transform { inverse: bool, start: u16, size: u16, consts_start: u32 },
    /// A run of glue ops in the glue pool
    Glue { start: u32, len: u32 },
}

/// A shape compiled for the staged executors: block transforms plus glue,
/// replayed per strip over a flat register file.
///
/// Registers 0..k hold the data rows on entry, `zero_regs` must be cleared
/// before each replay, and everything else is defined before use.
#[derive(Clone, Debug)]
pub(crate) struct StagedProgram16 {
    pub(crate) stages: Vec<StagedStage16>,
    pub(crate) consts: Vec<u16>,
    /// Karatsuba byte triple `[c_lo, c_hi + c_lo, c_hi * L]` per transform
    /// constant, parallel to `consts`; the SIMD executors index the static
    /// GF(2^8) nibble tables by these instead of deriving them per strip. Only
    /// those executors read it, so a scalar or wasm build carries it unused
    /// rather than paying a second lowering path.
    #[cfg_attr(not(gf16_simd_enabled), allow(dead_code))]
    pub(crate) consts_kara: Vec<[u8; 3]>,
    /// Whether the block behind each constant is a pruned dead subtree,
    /// parallel to `consts`; skipped wholesale by every executor.
    pub(crate) consts_skip: Vec<bool>,
    pub(crate) glue: Vec<FftOp16>,
    /// The same triples per glue op (zeroed for the non-multiplying forms).
    #[cfg_attr(not(gf16_simd_enabled), allow(dead_code))]
    pub(crate) glue_kara: Vec<[u8; 3]>,
    pub(crate) zero_regs: Vec<u16>,
    pub(crate) parity_regs: Vec<u16>,
    pub(crate) register_count: usize,
    /// Multiplies actually executed: nonzero-constant butterflies plus glue,
    /// the cost model the base-strategy choice leans on.
    pub(crate) mult_count: usize,
}

/// The Karatsuba byte triple of a tower constant: the three GF(2^8) products
/// the fused kernels take, `[c_lo, c_hi + c_lo, c_hi * L]`.
fn kara_triple(c: u16) -> [u8; 3] {
    let c_hi = (c >> 8) as u8;
    let c_lo = (c & 0xff) as u8;
    [c_lo, c_hi ^ c_lo, crate::galois::mul(c_hi, galois16::TOWER_L)]
}

struct Builder {
    items: Vec<Item>,
    next_register: u16,
}

impl Builder {
    fn alloc(&mut self, count: usize) -> Vec<u16> {
        let start = self.next_register;
        self.next_register += count as u16;
        (start..self.next_register).collect()
    }

    fn push_transform(&mut self, inverse: bool, regs: &[u16], beta: usize) {
        for pair in regs.windows(2) {
            debug_assert_eq!(pair[1], pair[0] + 1, "transform range must be contiguous");
        }
        self.items.push(Item::Transform {
            inverse,
            start: regs[0],
            size: regs.len() as u16,
            beta,
        });
    }

    /// Coefficients to values on the coset beta + {0..N-1}, in place
    fn fft(&mut self, regs: &[u16], beta: usize) {
        if regs.len() > 1 {
            self.push_transform(false, regs, beta);
        }
    }

    /// Values on the coset beta + {0..N-1} to coefficients, in place
    fn ifft(&mut self, regs: &[u16], beta: usize) {
        if regs.len() > 1 {
            self.push_transform(true, regs, beta);
        }
    }

    /// Interpolate the unique polynomial of degree below k through the values
    /// in regs[0..k] at points beta + {0..k-1}; regs holds its coefficients on
    /// return, zero above k. Registers beyond k must be zero on entry.
    fn interpolate(&mut self, regs: &[u16], beta: usize, k: usize) {
        let n = regs.len();
        if k == n {
            self.ifft(regs, beta);
            return;
        }
        let half = n / 2;
        if k <= half {
            self.interpolate(&regs[..half], beta, k);
            return;
        }

        // Lower half is fully known: its transform gives L = A + c*B, where
        // the polynomial splits as P = A + What*B across the two half cosets
        // and c is What at this coset's offset.
        self.ifft(&regs[..half], beta);
        let level = n.trailing_zeros() as usize - 1;
        let c = what()[level][beta];

        // The upper half sees Q = L + B, so peeling the values of L off the
        // known upper values leaves the values of B, whose degree is below
        // k - half. Interpolate it recursively, then fix A = L + c*B.
        let scratch = self.alloc(half);
        for j in 0..half {
            self.items.push(Item::Glue(FftOp16::Xor { dst: scratch[j], src: regs[j] }));
        }
        self.fft(&scratch, beta ^ half);
        for i in 0..(k - half) {
            self.items.push(Item::Glue(FftOp16::Xor { dst: regs[half + i], src: scratch[i] }));
        }
        self.interpolate(&regs[half..], beta ^ half, k - half);
        for j in 0..(k - half) {
            if c == 1 {
                self.items.push(Item::Glue(FftOp16::Xor { dst: regs[j], src: regs[half + j] }));
            } else if c != 0 {
                self.items.push(Item::Glue(FftOp16::MulXor {
                    dst: regs[j],
                    src: regs[half + j],
                    c,
                }));
            }
        }
    }

    /// Forward transform whose registers above `defined` are known zero: the
    /// first level peels off as explicit butterfly ops, which lowering
    /// normalizes into copies (dropping the multiplies against zero), then the
    /// two halves transform as regular blocks. Kills both the dead first-level
    /// work and the per-strip zeroing of the tail.
    fn fft_with_zero_tail(&mut self, regs: &[u16], beta: usize, defined: usize) {
        let size = regs.len();
        if defined >= size || size == 1 {
            self.fft(regs, beta);
            return;
        }
        let half = size / 2;
        debug_assert!(defined > half, "transform sizing keeps the defined prefix past half");
        let level = size.trailing_zeros() as usize - 1;
        let c = what()[level][beta];
        for i in 0..half {
            if c == 1 {
                self.items.push(Item::Glue(FftOp16::Xor { dst: regs[i], src: regs[half + i] }));
            } else if c != 0 {
                self.items.push(Item::Glue(FftOp16::MulXor {
                    dst: regs[i],
                    src: regs[half + i],
                    c,
                }));
            }
            self.items.push(Item::Glue(FftOp16::Xor { dst: regs[half + i], src: regs[i] }));
        }
        self.fft(&regs[..half], beta);
        self.fft(&regs[half..], beta ^ half);
    }

    /// Fold the base coefficients onto the size-aligned coset at `position` and
    /// transform the chunk in place, yielding the code values at points
    /// `position..position + size`. The higher basis polynomials are constant
    /// on a small coset, so folded coefficient j sums every coefficient
    /// congruent to j below the chunk size, scaled by that constant.
    fn folded_chunk(&mut self, base: &[u16], k: usize, position: usize, size: usize) -> Vec<u16> {
        let transform = base.len();
        let transform_levels = transform.trailing_zeros() as usize;
        let chunk = self.alloc(size);
        let shift = size.trailing_zeros() as usize;

        for (j, &chunk_reg) in chunk.iter().enumerate() {
            for h in 0..(transform / size) {
                let index = h * size + j;
                if index >= k {
                    continue;
                }
                let mut factor = 1u16;
                for (level, what_row) in
                    what().iter().enumerate().take(transform_levels).skip(shift)
                {
                    if index & (1 << level) != 0 {
                        factor = galois16::mul(factor, what_row[position]);
                    }
                }
                if factor == 0 {
                    continue;
                }
                if factor == 1 {
                    self.items
                        .push(Item::Glue(FftOp16::Xor { dst: chunk_reg, src: base[index] }));
                } else {
                    self.items.push(Item::Glue(FftOp16::MulXor {
                        dst: chunk_reg,
                        src: base[index],
                        c: factor,
                    }));
                }
            }
        }
        // A full-size chunk copies the base coefficients, so its tail above k
        // is never written; smaller chunks fold every register.
        self.fft_with_zero_tail(&chunk, position, k.min(size));
        chunk
    }
}

/// How the parity points inside the base transform get their values.
#[derive(Clone, Copy, PartialEq)]
enum BaseStrategy {
    /// Transform the base registers in place, last; parity points fall out of
    /// the full-size transform. Cheap when most of its outputs are live.
    InPlaceFft,
    /// Fold onto the smallest aligned coset window covering `[k, transform)`
    /// and transform only that. Cheap when the live tail is small.
    Window,
}

/// Compile a staged encode for a shape.
///
/// Trailing parity cosets are covered by folded chunks; parity inside the base
/// transform comes from whichever of the two base strategies costs fewer
/// multiplies for this shape, decided by compiling both.
pub(crate) fn build_staged_program16(k: usize, m: usize) -> StagedProgram16 {
    let n = k + m;
    let transform = k.next_power_of_two();
    let in_base = n.min(transform).saturating_sub(k);

    if in_base == 0 {
        return build_with_base_strategy(k, m, BaseStrategy::Window);
    }
    let window = build_with_base_strategy(k, m, BaseStrategy::Window);
    let in_place = build_with_base_strategy(k, m, BaseStrategy::InPlaceFft);
    if window.mult_count < in_place.mult_count {
        window
    } else {
        in_place
    }
}

fn build_with_base_strategy(k: usize, m: usize, strategy: BaseStrategy) -> StagedProgram16 {
    let n = k + m;
    let transform = k.next_power_of_two();

    let mut builder = Builder {
        items: Vec::new(),
        next_register: 0,
    };
    let base = builder.alloc(transform);
    builder.interpolate(&base, 0, k);

    let mut parity_regs: Vec<u16> = vec![0; m];

    // Parity points inside the base transform, via the chosen strategy. The
    // window is the smallest aligned coset covering [k, min(n, transform)); its
    // folds read the base coefficients, so it runs before any in-place fft.
    let base_end = n.min(transform);
    if base_end > k && strategy == BaseStrategy::Window {
        let mut size = 1usize;
        while (k & !(size - 1)) + size < base_end {
            size *= 2;
        }
        let position = k & !(size - 1);
        let chunk = builder.folded_chunk(&base, k, position, size);
        for (offset, &reg) in chunk.iter().enumerate() {
            let point = position + offset;
            if point >= k && point < n {
                parity_regs[point - k] = reg;
            }
        }
    }

    // Trailing chunks: fold the base coefficients onto each aligned coset,
    // then transform the chunk in place.
    let mut position = transform;
    while position < n {
        let alignment = 1usize << position.trailing_zeros().min(30);
        let needed = n - position;
        let size = alignment.min(needed.next_power_of_two()).min(transform);
        let chunk = builder.folded_chunk(&base, k, position, size);
        for (offset, &reg) in chunk.iter().enumerate() {
            let point = position + offset;
            if point >= k && point < n {
                parity_regs[point - k] = reg;
            }
        }
        position += size;
    }

    // The base transform last: it turns the coefficient registers back into
    // values, handing us the parity points inside the base coset for free.
    // Coefficients above k are structurally zero, so the tail peels.
    if base_end > k && strategy == BaseStrategy::InPlaceFft {
        builder.fft_with_zero_tail(&base, 0, k);
        parity_regs[..base_end - k].copy_from_slice(&base[k..base_end]);
    }

    lower_staged(&builder, &parity_regs, k)
}

/// Lower built items to the staged form, with dead code eliminated.
///
/// Three passes over the item stream:
/// 1. backward register-level liveness from the parity registers: glue that
///    feeds no parity drops, and every transform learns which of its outputs
///    are actually read,
/// 2. forward zero-normalization of the surviving glue (first writes become
///    full writes, reads of still-zero registers drop) plus the runtime zero
///    list for transforms that read zero-state registers,
/// 3. serialization: each transform becomes a block stage whose constants
///    carry a skip flag for sub-blocks with no live output, so dead subtrees
///    of partially-used transforms cost nothing at run time while live blocks
///    keep their hoisted tables.
fn lower_staged(builder: &Builder, parity_regs: &[u16], k: usize) -> StagedProgram16 {
    let register_count = builder.next_register as usize;

    // Pass 1: backward liveness. A transform reads and writes its whole
    // range, so its entry marks the range live and records the live outputs;
    // accumulating glue keeps its destination live, full writes never occur
    // here (the builder emits only xor forms).
    let mut live = vec![false; register_count];
    for &reg in parity_regs {
        live[reg as usize] = true;
    }
    let mut keep_glue = vec![false; builder.items.len()];
    let mut live_out: Vec<Vec<bool>> = vec![Vec::new(); builder.items.len()];
    for (item_index, item) in builder.items.iter().enumerate().rev() {
        match *item {
            Item::Glue(op) => match op {
                FftOp16::MulXor { dst, src, .. } | FftOp16::Xor { dst, src } => {
                    if live[dst as usize] {
                        keep_glue[item_index] = true;
                        live[src as usize] = true;
                    }
                }
                FftOp16::MulCopy { .. } | FftOp16::Copy { .. } => {
                    unreachable!("builder emits only xor forms")
                }
            },
            Item::Transform { start, size, .. } => {
                let range = start as usize..(start + size) as usize;
                live_out[item_index] = live[range.clone()].to_vec();
                live[range].fill(true);
            }
        }
    }

    // Pass 2 and 3 interleaved in item order: normalize surviving glue
    // against the zero state, serialize transforms with per-block skips.
    let mut is_zero = vec![true; register_count];
    is_zero[..k].fill(false);
    let mut needs_zero = vec![false; register_count];

    let mut stages: Vec<StagedStage16> = Vec::new();
    let mut consts: Vec<u16> = Vec::new();
    let mut consts_kara: Vec<[u8; 3]> = Vec::new();
    let mut consts_skip: Vec<bool> = Vec::new();
    let mut glue: Vec<FftOp16> = Vec::new();
    let mut glue_kara: Vec<[u8; 3]> = Vec::new();
    let mut mult_count = 0usize;
    let mut open_glue_start: Option<u32> = None;

    for (item_index, item) in builder.items.iter().enumerate() {
        match *item {
            Item::Glue(op) => {
                if !keep_glue[item_index] {
                    continue;
                }
                let normalized = match op {
                    FftOp16::MulXor { dst, src, c } => {
                        if is_zero[src as usize] {
                            continue;
                        }
                        if is_zero[dst as usize] {
                            is_zero[dst as usize] = false;
                            FftOp16::MulCopy { dst, src, c }
                        } else {
                            FftOp16::MulXor { dst, src, c }
                        }
                    }
                    FftOp16::Xor { dst, src } => {
                        if is_zero[src as usize] {
                            continue;
                        }
                        if is_zero[dst as usize] {
                            is_zero[dst as usize] = false;
                            FftOp16::Copy { dst, src }
                        } else {
                            FftOp16::Xor { dst, src }
                        }
                    }
                    FftOp16::MulCopy { .. } | FftOp16::Copy { .. } => {
                        unreachable!("builder emits only xor forms")
                    }
                };
                let c = match normalized {
                    FftOp16::MulXor { c, .. } | FftOp16::MulCopy { c, .. } => {
                        mult_count += 1;
                        c
                    }
                    _ => 0,
                };
                if open_glue_start.is_none() {
                    open_glue_start = Some(glue.len() as u32);
                }
                glue.push(normalized);
                glue_kara.push(kara_triple(c));
            }
            Item::Transform { inverse, start, size, beta } => {
                if let Some(glue_start) = open_glue_start.take() {
                    stages.push(StagedStage16::Glue {
                        start: glue_start,
                        len: glue.len() as u32 - glue_start,
                    });
                }

                // A transform reads its whole range: registers still in their
                // initial state must really be zero at runtime.
                for reg in start..start + size {
                    if is_zero[reg as usize] {
                        needs_zero[reg as usize] = true;
                    }
                    is_zero[reg as usize] = false;
                }

                // Any live output in a block's span keeps the block; a fully
                // dead span (a pruned subtree of a partially-used transform)
                // is skipped wholesale at run time.
                let alive = &live_out[item_index];
                let consts_start = consts.len() as u32;
                let size = size as usize;
                let levels = size.trailing_zeros() as usize;
                for step in 0..levels {
                    let level = if inverse { step } else { levels - 1 - step };
                    let stride = 1usize << level;
                    let mut block = 0usize;
                    while block < size {
                        let c = what()[level][beta ^ block];
                        let skip = !alive[block..block + 2 * stride].iter().any(|&l| l);
                        consts.push(c);
                        consts_kara.push(kara_triple(c));
                        consts_skip.push(skip);
                        if c != 0 && !skip {
                            mult_count += stride;
                        }
                        block += 2 * stride;
                    }
                }
                stages.push(StagedStage16::Transform {
                    inverse,
                    start,
                    size: size as u16,
                    consts_start,
                });
            }
        }
    }
    if let Some(glue_start) = open_glue_start.take() {
        stages.push(StagedStage16::Glue {
            start: glue_start,
            len: glue.len() as u32 - glue_start,
        });
    }

    let mut zero_regs: Vec<u16> = Vec::new();
    for (reg, &needed) in needs_zero.iter().enumerate() {
        if needed {
            zero_regs.push(reg as u16);
        }
    }

    StagedProgram16 {
        stages,
        consts,
        consts_kara,
        consts_skip,
        glue,
        glue_kara,
        zero_regs,
        parity_regs: parity_regs.to_vec(),
        register_count,
        mult_count,
    }
}

/// Replay a staged program over one u16 register per slot.
///
/// The scalar reference the SIMD executors are differentially tested against,
/// and the encode path for shards too short for a vector strip.
pub(crate) fn run_staged_symbols(program: &StagedProgram16, registers: &mut [u16]) {
    debug_assert_eq!(registers.len(), program.register_count);
    for &reg in &program.zero_regs {
        registers[reg as usize] = 0;
    }
    for stage in &program.stages {
        match *stage {
            StagedStage16::Glue { start, len } => {
                for op in &program.glue[start as usize..(start + len) as usize] {
                    match *op {
                        FftOp16::MulXor { dst, src, c } => {
                            registers[dst as usize] ^= galois16::mul(c, registers[src as usize]);
                        }
                        FftOp16::Xor { dst, src } => {
                            registers[dst as usize] ^= registers[src as usize];
                        }
                        FftOp16::MulCopy { dst, src, c } => {
                            registers[dst as usize] = galois16::mul(c, registers[src as usize]);
                        }
                        FftOp16::Copy { dst, src } => {
                            registers[dst as usize] = registers[src as usize];
                        }
                    }
                }
            }
            StagedStage16::Transform { inverse, start, size, consts_start } => {
                let size = size as usize;
                let start = start as usize;
                let levels = size.trailing_zeros() as usize;
                let mut const_index = consts_start as usize;
                for step in 0..levels {
                    let level = if inverse { step } else { levels - 1 - step };
                    let stride = 1usize << level;
                    let mut block = 0usize;
                    while block < size {
                        let c = program.consts[const_index];
                        let skip = program.consts_skip[const_index];
                        const_index += 1;
                        if skip {
                            block += 2 * stride;
                            continue;
                        }
                        for i in block..block + stride {
                            let lo = start + i;
                            let hi = lo + stride;
                            if inverse {
                                registers[hi] ^= registers[lo];
                                if c != 0 {
                                    registers[lo] ^= galois16::mul(c, registers[hi]);
                                }
                            } else {
                                if c != 0 {
                                    registers[lo] ^= galois16::mul(c, registers[hi]);
                                }
                                registers[hi] ^= registers[lo];
                            }
                        }
                        block += 2 * stride;
                    }
                }
            }
        }
    }
}

/// Encode by symbol-at-a-time program replay, for shards below a vector strip.
fn encode_symbolwise<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    let plane = data[0].as_ref().len() / 2;
    let mut registers = vec![0u16; program.register_count];
    for s in 0..plane {
        for (i, shard) in data.iter().enumerate() {
            let shard = shard.as_ref();
            registers[i] = galois16::pack(shard[plane + s], shard[s]);
        }
        run_staged_symbols(program, &mut registers);
        for (p, &reg) in program.parity_regs.iter().enumerate() {
            let value = registers[reg as usize];
            let shard = parity[p].as_mut();
            shard[s] = galois16::low(value);
            shard[plane + s] = galois16::high(value);
        }
    }
}

/// Portable strip executor: registers are `2 * STRIP`-byte plane pairs in one
/// flat arena, ops run through the dispatched GF(2^8) slice kernels as the
/// dense 2x2 block per tower constant.
#[cfg(not(gf16_neon_enabled))]
mod portable {
    use super::{FftOp16, StagedProgram16, StagedStage16};
    use crate::galois16;

    /// Symbols per strip; the arena is `register_count * 2 * STRIP` bytes,
    /// sized so the slice kernels amortize their per-call overhead.
    const STRIP: usize = 1024;

    /// `dst ^= c * src` over a register pair, as four GF(2^8) slice multiplies.
    ///
    /// # Safety
    /// `dst` and `src` are distinct in-bounds register indices.
    #[inline]
    unsafe fn mul_xor(arena: *mut u8, dst: usize, src: usize, c: u16, s: usize) {
        let m = galois16::const_matrix(c); // [m00, m01, m10, m11]
        let dst_lo = core::slice::from_raw_parts_mut(arena.add(dst * 2 * STRIP), s);
        let dst_hi = core::slice::from_raw_parts_mut(arena.add(dst * 2 * STRIP + STRIP), s);
        let src_lo = core::slice::from_raw_parts(arena.add(src * 2 * STRIP), s);
        let src_hi = core::slice::from_raw_parts(arena.add(src * 2 * STRIP + STRIP), s);
        crate::gf::mul_slice_xor(dst_hi, src_hi, m[0]);
        crate::gf::mul_slice_xor(dst_hi, src_lo, m[1]);
        crate::gf::mul_slice_xor(dst_lo, src_hi, m[2]);
        crate::gf::mul_slice_xor(dst_lo, src_lo, m[3]);
    }

    /// `dst ^= src` over a register pair.
    ///
    /// # Safety
    /// `dst` and `src` are distinct in-bounds register indices.
    #[inline]
    unsafe fn xor(arena: *mut u8, dst: usize, src: usize, s: usize) {
        let d = arena.add(dst * 2 * STRIP);
        let sr = arena.add(src * 2 * STRIP) as *const u8;
        for x in 0..s {
            *d.add(x) ^= *sr.add(x);
            *d.add(STRIP + x) ^= *sr.add(STRIP + x);
        }
    }

    /// Clear a register pair.
    ///
    /// # Safety
    /// `reg` is an in-bounds register index.
    #[inline]
    unsafe fn clear(arena: *mut u8, reg: usize, s: usize) {
        core::ptr::write_bytes(arena.add(reg * 2 * STRIP), 0, s);
        core::ptr::write_bytes(arena.add(reg * 2 * STRIP + STRIP), 0, s);
    }

    pub(super) fn encode<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
        program: &StagedProgram16,
        data: &[In],
        parity: &mut [Out],
    ) {
        let plane = data[0].as_ref().len() / 2;
        let mut arena = vec![0u8; program.register_count * 2 * STRIP];
        let regs = arena.as_mut_ptr();

        let mut offset = 0usize;
        while offset < plane {
            let s = STRIP.min(plane - offset);
            for (i, shard) in data.iter().enumerate() {
                let shard = shard.as_ref();
                // SAFETY: register i is in bounds; s <= STRIP.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        shard.as_ptr().add(offset),
                        regs.add(i * 2 * STRIP),
                        s,
                    );
                    core::ptr::copy_nonoverlapping(
                        shard.as_ptr().add(plane + offset),
                        regs.add(i * 2 * STRIP + STRIP),
                        s,
                    );
                }
            }
            // SAFETY: all register indices come from the compiled program,
            // which stays inside register_count; dst != src by construction.
            unsafe {
                for &reg in &program.zero_regs {
                    clear(regs, reg as usize, s);
                }
                for stage in &program.stages {
                    match *stage {
                        StagedStage16::Glue { start, len } => {
                            for op in &program.glue[start as usize..(start + len) as usize] {
                                match *op {
                                    FftOp16::MulXor { dst, src, c } => {
                                        mul_xor(regs, dst as usize, src as usize, c, s);
                                    }
                                    FftOp16::Xor { dst, src } => {
                                        xor(regs, dst as usize, src as usize, s);
                                    }
                                    FftOp16::MulCopy { dst, src, c } => {
                                        clear(regs, dst as usize, s);
                                        mul_xor(regs, dst as usize, src as usize, c, s);
                                    }
                                    FftOp16::Copy { dst, src } => {
                                        clear(regs, dst as usize, s);
                                        xor(regs, dst as usize, src as usize, s);
                                    }
                                }
                            }
                        }
                        StagedStage16::Transform { inverse, start, size, consts_start } => {
                            let size = size as usize;
                            let start = start as usize;
                            let levels = size.trailing_zeros() as usize;
                            let mut const_index = consts_start as usize;
                            for step in 0..levels {
                                let level = if inverse { step } else { levels - 1 - step };
                                let stride = 1usize << level;
                                let mut block = 0usize;
                                while block < size {
                                    let c = program.consts[const_index];
                                    let skip = program.consts_skip[const_index];
                                    const_index += 1;
                                    if skip {
                                        block += 2 * stride;
                                        continue;
                                    }
                                    for i in block..block + stride {
                                        let lo = start + i;
                                        let hi = lo + stride;
                                        if inverse {
                                            xor(regs, hi, lo, s);
                                            if c != 0 {
                                                mul_xor(regs, lo, hi, c, s);
                                            }
                                        } else {
                                            if c != 0 {
                                                mul_xor(regs, lo, hi, c, s);
                                            }
                                            xor(regs, hi, lo, s);
                                        }
                                    }
                                    block += 2 * stride;
                                }
                            }
                        }
                    }
                }
            }
            for (p, &reg) in program.parity_regs.iter().enumerate() {
                let shard = parity[p].as_mut();
                // SAFETY: parity register in bounds; s <= STRIP.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        regs.add(reg as usize * 2 * STRIP) as *const u8,
                        shard.as_mut_ptr().add(offset),
                        s,
                    );
                    core::ptr::copy_nonoverlapping(
                        regs.add(reg as usize * 2 * STRIP + STRIP) as *const u8,
                        shard.as_mut_ptr().add(plane + offset),
                        s,
                    );
                }
            }
            offset += s;
        }
    }
}

/// FFT systematic encode through a compiled program: `parity` overwritten from
/// `data`. Shards are `2N` bytes each in plane-pair layout `[low N | high N]`,
/// all one length; byte-identical to the matrix encode.
pub(crate) fn encode<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    if data.is_empty() || parity.is_empty() {
        return;
    }
    let plane = data[0].as_ref().len() / 2;
    if plane == 0 {
        return;
    }

    #[cfg(gf16_neon_enabled)]
    {
        if plane >= 16 {
            crate::gf::fft16_neon::encode_staged16(program, data, parity);
        } else {
            encode_symbolwise(program, data, parity);
        }
    }

    #[cfg(not(gf16_neon_enabled))]
    {
        // The GFNI executor when the host has it, otherwise the portable
        // strip arena; symbol replay only for sub-strip shards.
        #[cfg(gf16_x86_enabled)]
        if plane >= 16 && crate::gf::fft16_x86::available() {
            crate::gf::fft16_x86::encode_staged16(program, data, parity);
            return;
        }
        if plane >= 64 {
            portable::encode(program, data, parity);
        } else {
            encode_symbolwise(program, data, parity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The independent oracle: one symbol column through the scalar tower
    // multiply against the systematic generator rows, built the same way
    // ReedSolomon16 builds them.
    fn oracle(k: usize, m: usize, symbols: &[u16]) -> Vec<u16> {
        let vandermonde = crate::matrix::Matrix::<u16>::vandermonde(k + m, k);
        let top = vandermonde.sub_matrix(0, 0, k, k);
        let generator = vandermonde.multiply(&top.invert().expect("top block invertible"));
        (k..k + m)
            .map(|r| {
                generator
                    .get_row(r)
                    .iter()
                    .zip(symbols.iter())
                    .fold(0u16, |acc, (&c, &s)| acc ^ galois16::mul(c, s))
            })
            .collect()
    }

    // The compiled program matches the generator matrix on every parity point,
    // across shapes that exercise every base strategy and chunk pattern:
    // window smaller than base, power-of-two k, multi-chunk tails, k past the
    // half boundary, tiny shapes.
    #[test]
    fn staged_matches_matrix() {
        let shapes = [
            (2usize, 2usize),
            (3, 5),
            (5, 3),
            (7, 13),
            (16, 16),
            (17, 33),
            (32, 96),
            (48, 144),
            (63, 65),
            (86, 170),
            (100, 50),
            (128, 128),
        ];
        let mut seed = 0xC0FF_EE16_5EEDu64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u16
        };
        for &(k, m) in &shapes {
            let program = build_staged_program16(k, m);
            for _ in 0..8 {
                let symbols: Vec<u16> = (0..k).map(|_| next()).collect();
                // Deliberately stale so the zero list is really exercised.
                let mut registers = vec![0xA5A5u16; program.register_count];
                registers[..k].copy_from_slice(&symbols);
                run_staged_symbols(&program, &mut registers);
                let got: Vec<u16> = program
                    .parity_regs
                    .iter()
                    .map(|&reg| registers[reg as usize])
                    .collect();
                assert_eq!(got, oracle(k, m, &symbols), "k={k} m={m}");
            }
        }
    }

    // Manual introspection: dump program composition per shape.
    // cargo test --release dump_stats -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_stats() {
        for &(k, m) in &[(7usize, 13usize), (10, 10), (11, 20), (16, 16), (17, 33), (86, 170), (171, 341), (128, 128)] {
            let p = build_staged_program16(k, m);
            println!(
                "({k},{m}): regs={} mults={} glue={} zero={} stages:",
                p.register_count,
                p.mult_count,
                p.glue.len(),
                p.zero_regs.len()
            );
            let mut glue_mul = 0usize;
            let mut glue_other = 0usize;
            for op in &p.glue {
                match op {
                    FftOp16::MulXor { .. } | FftOp16::MulCopy { .. } => glue_mul += 1,
                    _ => glue_other += 1,
                }
            }
            println!("  glue: {glue_mul} mul, {glue_other} xor/copy");
            println!("  zero_regs: {:?}", p.zero_regs);
            for stage in &p.stages {
                match *stage {
                    StagedStage16::Transform { inverse, start, size, consts_start } => {
                        let levels = (size as usize).trailing_zeros() as usize;
                        let range = consts_start as usize..consts_start as usize + (size as usize - 1);
                        let zeros = p.consts[range].iter().filter(|&&c| c == 0).count();
                        println!(
                            "  {} size={size} start={start} zero-consts={zeros} mults~{}",
                            if inverse { "ifft" } else { " fft" },
                            (size as usize / 2) * levels
                        );
                    }
                    StagedStage16::Glue { len, .. } => println!("  glue run len={len}"),
                }
            }
        }
    }

    // The window strategy keeps the multiply count well under the naive
    // full-transform-per-coset cost at the flagship outer shape.
    #[test]
    fn window_beats_full_transforms() {
        let program = build_staged_program16(86, 170);
        // Naive recursion: interpolate (~490) plus two full 128-point
        // transforms (~900) landed near 1400.
        assert!(program.mult_count < 1100, "mult_count regressed: {}", program.mult_count);
        assert!(program.register_count <= MAX_STAGED_REGISTERS16);
    }
}
