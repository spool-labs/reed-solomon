//! Lin-Chung-Han novel-basis FFT machinery for Reed-Solomon encoding
//!
//! Implements the transform from "Novel Polynomial Basis With Fast Fourier
//! Transform and Its Application to Reed-Solomon Erasure Codes" (Lin, Chung,
//! Han, 2016) over this crate's GF(2^8), using the standard basis 1, 2, 4,
//! .., 128 so the evaluation point with index i is exactly the byte i. That
//! makes interpolation through points 0..k-1 and evaluation at points k..n-1
//! produce byte-identical parity to the generator matrix code.
//!
//! The builder expresses an encode as transform items (whole FFT/IFFT blocks
//! over contiguous register ranges) plus glue ops. Two lowerings share it:
//! the flat path expands everything to single ops and feeds the generated
//! straight-line executors for the production shapes, and the staged path
//! keeps the blocks whole so runtime-shaped codecs can execute them through
//! const-generic block kernels.
//!
//! Iteration order contract, shared by the flat expansion, the constant
//! serialization, the block kernels, and the interpreters: a forward FFT
//! walks levels from high to low, an inverse FFT from low to high; within a
//! level, blocks walk in ascending base order with one constant per block,
//! and butterflies in ascending index order.

use crate::galois;

/// Number of transform levels; 2^8 covers every point of GF(2^8)
const LEVELS: usize = 8;

/// Normalized subspace polynomials, one 256-entry table per level
///
/// WHAT[j][x] is the subspace polynomial vanishing on the span of 1..2^j-1,
/// scaled so WHAT[j][2^j] = 1. Butterfly constants are lookups into this.
static WHAT: [[u8; 256]; LEVELS] = gen_what_tables();

const fn gen_what_tables() -> [[u8; 256]; LEVELS] {
    let mul = galois::gen_mul_table();

    // W_0(x) = x; W_{j+1}(x) = W_j(x) * (W_j(x) + W_j(2^j)), the classic
    // subspace polynomial doubling, using that W_j is GF(2)-linear.
    let mut w = [[0u8; 256]; LEVELS];
    let mut x = 0usize;
    while x < 256 {
        w[0][x] = x as u8;
        x += 1;
    }
    let mut j = 0usize;
    while j + 1 < LEVELS {
        let pivot = w[j][1usize << j];
        let mut x = 0usize;
        while x < 256 {
            let value = w[j][x];
            w[j + 1][x] = mul[value as usize][(value ^ pivot) as usize];
            x += 1;
        }
        j += 1;
    }

    // Normalize each level by the inverse of its pivot value.
    let mut what = [[0u8; 256]; LEVELS];
    let mut j = 0usize;
    while j < LEVELS {
        let pivot = w[j][1usize << j];
        let mut inverse = 0u8;
        let mut candidate = 1usize;
        while candidate < 256 {
            if mul[pivot as usize][candidate] == 1 {
                inverse = candidate as u8;
                break;
            }
            candidate += 1;
        }
        let mut x = 0usize;
        while x < 256 {
            what[j][x] = mul[w[j][x] as usize][inverse as usize];
            x += 1;
        }
        j += 1;
    }
    what
}

/// One field operation over the virtual register file
///
/// The builder emits only the two xor forms against an all-zero backdrop;
/// normalization rewrites first writes into the full-write forms so that
/// registers never depend on their starting contents.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FftOp {
    /// registers[dst] ^= c * registers[src]
    MulXor { dst: u16, src: u16, c: u8 },
    /// registers[dst] ^= registers[src]
    Xor { dst: u16, src: u16 },
    /// registers[dst] = c * registers[src]
    MulCopy { dst: u16, src: u16, c: u8 },
    /// registers[dst] = registers[src]
    Copy { dst: u16, src: u16 },
}

/// One build step: a whole transform over a contiguous register range, or a
/// single glue op
#[derive(Clone, Copy, Debug)]
enum Item {
    Transform { inverse: bool, start: u16, size: u16, beta: usize },
    Glue(FftOp),
}

/// A compiled encode: replay the ops, then read the parity registers
///
/// Registers 0..k hold the data rows on entry; every other register is
/// defined before use. parity_regs[p] names the register holding parity p.
#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct FftProgram {
    pub(crate) ops: Vec<FftOp>,
    pub(crate) register_count: usize,
    pub(crate) parity_regs: Vec<u16>,
    pub(crate) mult_count: usize,
}

/// One stage of a staged encode
#[derive(Clone, Copy, Debug)]
pub(crate) enum StagedStage {
    /// In-place transform of `size` registers starting at `start`, its block
    /// constants at `consts_start` in the constant pool
    Transform { inverse: bool, start: u16, size: u16, consts_start: u32 },
    /// A run of glue ops in the glue pool
    Glue { start: u32, len: u32 },
}

/// Staged register files above this size fall back to the fused matrix path
pub(crate) const MAX_STAGED_REGISTERS: usize = 160;

/// A shape compiled for the staged executor: block transforms plus glue,
/// replayed per strip over a stack register array
///
/// Registers 0..k hold the data rows on entry, `zero_regs` must be cleared
/// before each replay, and everything else is defined before use.
#[derive(Clone, Debug)]
pub(crate) struct StagedProgram {
    pub(crate) stages: Vec<StagedStage>,
    pub(crate) consts: Vec<u8>,
    pub(crate) glue: Vec<FftOp>,
    pub(crate) zero_regs: Vec<u16>,
    pub(crate) parity_regs: Vec<u16>,
    pub(crate) register_count: usize,
    /// Multiplies actually executed: nonzero-constant butterflies plus glue
    pub(crate) mult_count: usize,
    /// Butterfly xors, paid by every butterfly regardless of its constant
    #[allow(dead_code)] // read by the op-count tests and kept for routing tuning
    pub(crate) xor_count: usize,
    /// Vectors crossing a stage boundary: each transform loads and stores
    /// its whole range once
    #[allow(dead_code)] // read by the op-count tests and kept for routing tuning
    pub(crate) boundary_count: usize,
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
        let c = WHAT[level][beta];

        // The upper half sees Q = L + B, so peeling the values of L off the
        // known upper values leaves the values of B, whose degree is below
        // k - half. Interpolate it recursively, then fix A = L + c*B.
        let scratch = self.alloc(half);
        for j in 0..half {
            self.items.push(Item::Glue(FftOp::Xor { dst: scratch[j], src: regs[j] }));
        }
        self.fft(&scratch, beta ^ half);
        for i in 0..(k - half) {
            self.items.push(Item::Glue(FftOp::Xor { dst: regs[half + i], src: scratch[i] }));
        }
        self.interpolate(&regs[half..], beta ^ half, k - half);
        for j in 0..(k - half) {
            if c == 1 {
                self.items.push(Item::Glue(FftOp::Xor { dst: regs[j], src: regs[half + j] }));
            } else if c != 0 {
                self.items.push(Item::Glue(FftOp::MulXor {
                    dst: regs[j],
                    src: regs[half + j],
                    c,
                }));
            }
        }
    }
}

/// Expand transform items to single butterfly ops in the contract order
#[cfg(test)]
fn flatten_items(items: &[Item]) -> Vec<FftOp> {
    let mut ops: Vec<FftOp> = Vec::new();
    for item in items {
        match *item {
            Item::Glue(op) => ops.push(op),
            Item::Transform { inverse, start, size, beta } => {
                let n = size as usize;
                let levels = n.trailing_zeros() as usize;
                let level_order: Vec<usize> = if inverse {
                    (0..levels).collect()
                } else {
                    (0..levels).rev().collect()
                };
                for level in level_order {
                    let stride = 1usize << level;
                    let mut base = 0usize;
                    while base < n {
                        let c = WHAT[level][beta ^ base];
                        for i in base..base + stride {
                            let lo = start + i as u16;
                            let hi = start + (i + stride) as u16;
                            if inverse {
                                ops.push(FftOp::Xor { dst: hi, src: lo });
                                if c == 1 {
                                    ops.push(FftOp::Xor { dst: lo, src: hi });
                                } else if c != 0 {
                                    ops.push(FftOp::MulXor { dst: lo, src: hi, c });
                                }
                            } else {
                                if c == 1 {
                                    ops.push(FftOp::Xor { dst: lo, src: hi });
                                } else if c != 0 {
                                    ops.push(FftOp::MulXor { dst: lo, src: hi, c });
                                }
                                ops.push(FftOp::Xor { dst: hi, src: lo });
                            }
                        }
                        base += 2 * stride;
                    }
                }
            }
        }
    }
    ops
}

/// Build the shared front of an encode: interpolation of the data polynomial
/// over the base transform, returning the base registers
fn build_interpolation(builder: &mut Builder, k: usize) -> Vec<u16> {
    let transform = k.next_power_of_two();
    let base = builder.alloc(transform);
    builder.interpolate(&base, 0, k);
    base
}

/// Compile the flat encode program for the generated executors
///
/// The program interpolates the data polynomial and evaluates it at the
/// parity points, so its parity bytes equal the generator matrix encode
/// byte for byte.
#[cfg(test)]
pub(crate) fn build_encode_program(k: usize, m: usize) -> FftProgram {
    let n = k + m;
    let transform = k.next_power_of_two();

    let mut builder = Builder {
        items: Vec::new(),
        next_register: 0,
    };
    let base = build_interpolation(&mut builder, k);

    // Extra size-aligned cosets covering parity points beyond the base
    // transform get a copy of the coefficients and a forward transform each.
    // Dead-code elimination below trims the outputs nobody asked for.
    let mut parity_regs: Vec<u16> = vec![0; m];
    let mut block_base = transform;
    while block_base < n {
        let block = builder.alloc(transform);
        for j in 0..k.min(transform) {
            builder.items.push(Item::Glue(FftOp::Xor { dst: block[j], src: base[j] }));
        }
        builder.fft(&block, block_base);
        for (point, &reg) in (block_base..block_base + transform).zip(block.iter()) {
            if point >= k && point < n {
                parity_regs[point - k] = reg;
            }
        }
        block_base += transform;
    }

    // The base transform last: it turns the coefficient registers back into
    // values, handing us the parity points inside the base coset for free.
    builder.fft(&base, 0);
    let in_base = n.min(transform).saturating_sub(k);
    parity_regs[..in_base].copy_from_slice(&base[k..k + in_base]);

    let flat = flatten_items(&builder.items);
    let normalized = normalize_zero_reads(&flat, builder.next_register as usize, k);

    // Backward liveness pass from the parity registers. Full-write ops kill
    // their destination, so everything upstream of an overwritten value drops.
    let mut needed = vec![false; builder.next_register as usize];
    for &reg in &parity_regs {
        needed[reg as usize] = true;
    }
    let mut kept: Vec<FftOp> = Vec::with_capacity(normalized.len());
    for &op in normalized.iter().rev() {
        match op {
            FftOp::MulXor { dst, src, .. } | FftOp::Xor { dst, src } => {
                if needed[dst as usize] {
                    needed[src as usize] = true;
                    kept.push(op);
                }
            }
            FftOp::MulCopy { dst, src, .. } | FftOp::Copy { dst, src } => {
                if needed[dst as usize] {
                    needed[dst as usize] = false;
                    needed[src as usize] = true;
                    kept.push(op);
                }
            }
        }
    }
    kept.reverse();

    let mut mult_count = 0usize;
    for op in &kept {
        if matches!(op, FftOp::MulXor { .. } | FftOp::MulCopy { .. }) {
            mult_count += 1;
        }
    }

    let (ops, parity_regs, register_count) =
        remap_registers(&kept, &parity_regs, builder.next_register as usize, k);

    FftProgram {
        ops,
        register_count,
        parity_regs,
        mult_count,
    }
}

/// Compile a staged encode for a runtime shape
///
/// Trailing parity cosets are covered by folding the coefficients down to
/// each aligned chunk (the higher basis polynomials are constant on a small
/// coset) and running a chunk-sized forward transform, so no whole-transform
/// output ever goes to waste.
pub(crate) fn build_staged_program(k: usize, m: usize) -> StagedProgram {
    let n = k + m;
    let transform = k.next_power_of_two();
    let transform_levels = transform.trailing_zeros() as usize;

    let mut builder = Builder {
        items: Vec::new(),
        next_register: 0,
    };
    let base = build_interpolation(&mut builder, k);

    let mut parity_regs: Vec<u16> = vec![0; m];

    // Trailing chunks: fold the base coefficients onto each aligned coset,
    // then transform the chunk in place.
    let mut position = transform;
    while position < n {
        let alignment = 1usize << position.trailing_zeros().min(30);
        let needed = n - position;
        let size = alignment.min(needed.next_power_of_two()).min(transform);
        let chunk = builder.alloc(size);
        let shift = size.trailing_zeros() as usize;

        for (j, &chunk_reg) in chunk.iter().enumerate() {
            // Folded coefficient j sums every coefficient congruent to j
            // below the chunk size, scaled by the constant value the high
            // basis polynomials take on this coset.
            for h in 0..(transform / size) {
                let index = h * size + j;
                if index >= k {
                    continue;
                }
                let mut factor = 1u8;
                for (level, what_row) in
                    WHAT.iter().enumerate().take(transform_levels).skip(shift)
                {
                    if index & (1 << level) != 0 {
                        factor = galois::mul(factor, what_row[position]);
                    }
                }
                if factor == 0 {
                    continue;
                }
                if factor == 1 {
                    builder
                        .items
                        .push(Item::Glue(FftOp::Xor { dst: chunk_reg, src: base[index] }));
                } else {
                    builder.items.push(Item::Glue(FftOp::MulXor {
                        dst: chunk_reg,
                        src: base[index],
                        c: factor,
                    }));
                }
            }
        }
        builder.fft(&chunk, position);

        for (offset, &reg) in chunk.iter().enumerate() {
            let point = position + offset;
            if point >= k && point < n {
                parity_regs[point - k] = reg;
            }
        }
        position += size;
    }

    // The base transform last, handing us the parity points inside the base
    // coset; the folds above read the coefficients, so order matters.
    builder.fft(&base, 0);
    let in_base = n.min(transform).saturating_sub(k);
    parity_regs[..in_base].copy_from_slice(&base[k..k + in_base]);

    lower_staged(&builder, &parity_regs, k)
}

/// Lower built items to the staged form: normalize glue against the zero
/// state, record which registers must be cleared per replay, serialize the
/// per-block constants, and count the multiplies that actually execute
fn lower_staged(builder: &Builder, parity_regs: &[u16], k: usize) -> StagedProgram {
    let register_count = builder.next_register as usize;
    let mut is_zero = vec![true; register_count];
    is_zero[..k].fill(false);
    let mut needs_zero = vec![false; register_count];

    let mut stages: Vec<StagedStage> = Vec::new();
    let mut consts: Vec<u8> = Vec::new();
    let mut glue: Vec<FftOp> = Vec::new();
    let mut mult_count = 0usize;
    let mut xor_count = 0usize;
    let mut boundary_count = 0usize;
    let mut open_glue_start: Option<u32> = None;

    for item in &builder.items {
        match *item {
            Item::Glue(op) => {
                let normalized = match op {
                    FftOp::MulXor { dst, src, c } => {
                        if is_zero[src as usize] {
                            continue;
                        }
                        if is_zero[dst as usize] {
                            is_zero[dst as usize] = false;
                            FftOp::MulCopy { dst, src, c }
                        } else {
                            FftOp::MulXor { dst, src, c }
                        }
                    }
                    FftOp::Xor { dst, src } => {
                        if is_zero[src as usize] {
                            continue;
                        }
                        if is_zero[dst as usize] {
                            is_zero[dst as usize] = false;
                            FftOp::Copy { dst, src }
                        } else {
                            FftOp::Xor { dst, src }
                        }
                    }
                    FftOp::MulCopy { .. } | FftOp::Copy { .. } => {
                        unreachable!("builder emits only xor forms")
                    }
                };
                if matches!(normalized, FftOp::MulXor { .. } | FftOp::MulCopy { .. }) {
                    mult_count += 1;
                }
                if open_glue_start.is_none() {
                    open_glue_start = Some(glue.len() as u32);
                }
                glue.push(normalized);
            }
            Item::Transform { inverse, start, size, beta } => {
                if let Some(glue_start) = open_glue_start.take() {
                    stages.push(StagedStage::Glue {
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

                let consts_start = consts.len() as u32;
                let n = size as usize;
                let levels = n.trailing_zeros() as usize;
                boundary_count += 2 * n;
                let level_order: Vec<usize> = if inverse {
                    (0..levels).collect()
                } else {
                    (0..levels).rev().collect()
                };
                for level in level_order {
                    let stride = 1usize << level;
                    let mut base = 0usize;
                    while base < n {
                        let c = WHAT[level][beta ^ base];
                        consts.push(c);
                        if c != 0 {
                            mult_count += stride;
                        }
                        xor_count += stride;
                        base += 2 * stride;
                    }
                }
                stages.push(StagedStage::Transform { inverse, start, size, consts_start });
            }
        }
    }
    if let Some(glue_start) = open_glue_start.take() {
        stages.push(StagedStage::Glue {
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

    StagedProgram {
        stages,
        consts,
        glue,
        zero_regs,
        parity_regs: parity_regs.to_vec(),
        register_count,
        mult_count,
        xor_count,
        boundary_count,
    }
}

/// Track which registers still hold their initial zero and rewrite the ops so
/// no kept op ever reads or accumulates into an undefined register
#[cfg(test)]
fn normalize_zero_reads(ops: &[FftOp], register_count: usize, k: usize) -> Vec<FftOp> {
    let mut is_zero = vec![true; register_count];
    is_zero[..k].fill(false);

    let mut out: Vec<FftOp> = Vec::with_capacity(ops.len());
    for &op in ops {
        match op {
            FftOp::MulXor { dst, src, c } => {
                if is_zero[src as usize] {
                    continue;
                }
                if is_zero[dst as usize] {
                    out.push(FftOp::MulCopy { dst, src, c });
                    is_zero[dst as usize] = false;
                } else {
                    out.push(FftOp::MulXor { dst, src, c });
                }
            }
            FftOp::Xor { dst, src } => {
                if is_zero[src as usize] {
                    continue;
                }
                if is_zero[dst as usize] {
                    out.push(FftOp::Copy { dst, src });
                    is_zero[dst as usize] = false;
                } else {
                    out.push(FftOp::Xor { dst, src });
                }
            }
            FftOp::MulCopy { .. } | FftOp::Copy { .. } => {
                unreachable!("builder emits only xor forms");
            }
        }
    }
    out
}

/// Renumber registers with reuse so the executor file stays small
///
/// Data registers keep ids 0..k. Every other register is assigned from a free
/// pool when first written and returned to it after its last read; every
/// register is fully defined before use after normalization, so reuse never
/// observes stale contents. Parity registers stay live to the end.
#[cfg(test)]
fn remap_registers(
    ops: &[FftOp],
    parity_regs: &[u16],
    register_count: usize,
    k: usize,
) -> (Vec<FftOp>, Vec<u16>, usize) {
    let mut last_use = vec![0usize; register_count];
    for (position, op) in ops.iter().enumerate() {
        let (dst, src) = op_registers(*op);
        last_use[dst as usize] = position;
        last_use[src as usize] = position;
    }
    for &reg in parity_regs {
        last_use[reg as usize] = ops.len();
    }

    let mut mapping: Vec<Option<u16>> = vec![None; register_count];
    for (data_reg, slot) in mapping.iter_mut().enumerate().take(k) {
        *slot = Some(data_reg as u16);
    }
    let mut next_fresh = k as u16;
    let mut free: Vec<u16> = Vec::new();
    let mut expires: Vec<Vec<u16>> = vec![Vec::new(); ops.len() + 1];
    for (virtual_reg, &position) in last_use.iter().enumerate() {
        if mapping[virtual_reg].is_none() {
            expires[position].push(virtual_reg as u16);
        }
    }
    // Data registers whose last use is before the end free their slot too.
    for data_reg in 0..k {
        if last_use[data_reg] < ops.len() {
            expires[last_use[data_reg]].push(data_reg as u16);
        }
    }

    let mut assign = |mapping: &mut Vec<Option<u16>>, free: &mut Vec<u16>, virtual_reg: u16| {
        if mapping[virtual_reg as usize].is_none() {
            let id = match free.pop() {
                Some(id) => id,
                None => {
                    let id = next_fresh;
                    next_fresh += 1;
                    id
                }
            };
            mapping[virtual_reg as usize] = Some(id);
        }
    };

    let mut remapped: Vec<FftOp> = Vec::with_capacity(ops.len());
    for (position, op) in ops.iter().enumerate() {
        let (dst, src) = op_registers(*op);
        assign(&mut mapping, &mut free, src);
        assign(&mut mapping, &mut free, dst);
        let new_dst = mapping[dst as usize].expect("assigned");
        let new_src = mapping[src as usize].expect("assigned");
        let out = match *op {
            FftOp::MulXor { c, .. } => FftOp::MulXor { dst: new_dst, src: new_src, c },
            FftOp::Xor { .. } => FftOp::Xor { dst: new_dst, src: new_src },
            FftOp::MulCopy { c, .. } => FftOp::MulCopy { dst: new_dst, src: new_src, c },
            FftOp::Copy { .. } => FftOp::Copy { dst: new_dst, src: new_src },
        };
        remapped.push(out);
        for &virtual_reg in &expires[position] {
            if let Some(id) = mapping[virtual_reg as usize] {
                free.push(id);
            }
        }
    }

    let parity_out: Vec<u16> = parity_regs
        .iter()
        .map(|&reg| mapping[reg as usize].expect("parity register assigned"))
        .collect();
    (remapped, parity_out, next_fresh as usize)
}

/// The destination and source register of any op
#[cfg(test)]
fn op_registers(op: FftOp) -> (u16, u16) {
    match op {
        FftOp::MulXor { dst, src, .. }
        | FftOp::Xor { dst, src }
        | FftOp::MulCopy { dst, src, .. }
        | FftOp::Copy { dst, src } => (dst, src),
    }
}

/// Replay a flat program over single bytes: data[0..k] in, parities returned
#[cfg(test)]
pub(crate) fn run_program_bytes(program: &FftProgram, data: &[u8]) -> Vec<u8> {
    let mut registers = vec![0u8; program.register_count];
    registers[..data.len()].copy_from_slice(data);
    run_ops_bytes(&program.ops, &mut registers);
    program
        .parity_regs
        .iter()
        .map(|&reg| registers[reg as usize])
        .collect()
}

#[cfg(test)]
fn run_ops_bytes(ops: &[FftOp], registers: &mut [u8]) {
    for op in ops {
        match *op {
            FftOp::MulXor { dst, src, c } => {
                registers[dst as usize] ^= galois::mul(c, registers[src as usize]);
            }
            FftOp::Xor { dst, src } => {
                registers[dst as usize] ^= registers[src as usize];
            }
            FftOp::MulCopy { dst, src, c } => {
                registers[dst as usize] = galois::mul(c, registers[src as usize]);
            }
            FftOp::Copy { dst, src } => {
                registers[dst as usize] = registers[src as usize];
            }
        }
    }
}

/// Replay a staged program over single bytes, mirroring the SIMD runners:
/// stale register file, explicit zeroing, block transforms, glue
#[cfg(test)]
pub(crate) fn run_staged_bytes(program: &StagedProgram, data: &[u8]) -> Vec<u8> {
    // Deliberately stale so the zero list is really exercised.
    let mut registers = vec![0xA5u8; program.register_count];
    registers[..data.len()].copy_from_slice(data);
    for &reg in &program.zero_regs {
        registers[reg as usize] = 0;
    }

    for stage in &program.stages {
        match *stage {
            StagedStage::Glue { start, len } => {
                let ops = &program.glue[start as usize..(start + len) as usize];
                run_ops_bytes(ops, &mut registers);
            }
            StagedStage::Transform { inverse, start, size, consts_start } => {
                let n = size as usize;
                let levels = n.trailing_zeros() as usize;
                let mut const_index = consts_start as usize;
                let level_order: Vec<usize> = if inverse {
                    (0..levels).collect()
                } else {
                    (0..levels).rev().collect()
                };
                for level in level_order {
                    let stride = 1usize << level;
                    let mut base = 0usize;
                    while base < n {
                        let c = program.consts[const_index];
                        const_index += 1;
                        for i in base..base + stride {
                            let lo = (start as usize) + i;
                            let hi = lo + stride;
                            if inverse {
                                registers[hi] ^= registers[lo];
                                registers[lo] ^= galois::mul(c, registers[hi]);
                            } else {
                                registers[lo] ^= galois::mul(c, registers[hi]);
                                registers[hi] ^= registers[lo];
                            }
                        }
                        base += 2 * stride;
                    }
                }
            }
        }
    }

    program
        .parity_regs
        .iter()
        .map(|&reg| registers[reg as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // regenerates src/fft_programs.rs; run after changing the builder:
    //   cargo test dump_programs -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_programs() {
        println!("//! Generated executor programs for the production shapes");
        println!("//!");
        println!("//! Produced by the dump_programs test in fft.rs; regenerate with");
        println!("//!   cargo test dump_programs -- --ignored --nocapture");
        println!("//! whenever the program builder changes. The byte-identity tests");
        println!("//! catch a stale copy.");
        let shapes = [
            (7usize, 13usize, "7_13"),
            (10, 10, "10_10"),
            (14, 14, "14_14"),
            (16, 16, "16_16"),
            (18, 6, "18_6"),
            (32, 32, "32_32"), // Agave shred FEC shape, 987-byte shards
        ];
        println!();
        println!("/// Every shape with a generated program; routing and the staged");
        println!("/// builder consult this single list");
        println!("pub(crate) const GENERATED_SHAPES: &[(usize, usize)] = &[");
        for (k, m, _) in shapes {
            println!("    ({k}, {m}),");
        }
        println!("];");
        for (k, m, name) in shapes {
            let program = build_encode_program(k, m);
            println!();
            println!(
                "// shape ({k},{m}): {} registers, {} ops, {} multiplies",
                program.register_count,
                program.ops.len(),
                program.mult_count
            );
            println!(
                "pub(crate) const FFT_REGS_{}: usize = {};",
                name.to_uppercase(),
                program.register_count
            );
            println!("macro_rules! fft_program_{name} {{");
            println!("    ($engine:ty, $r:ident, $in:ident, $out:ident, $off:ident) => {{");
            println!("        crate::macros::fft_run!($engine, $r, $in, $out, $off;");
            for shard in 0..k {
                println!("            ld {shard} {shard};");
            }
            for op in &program.ops {
                match *op {
                    FftOp::MulXor { dst, src, c } => println!("            mx {dst} {src} {c};"),
                    FftOp::Xor { dst, src } => println!("            xo {dst} {src};"),
                    FftOp::MulCopy { dst, src, c } => println!("            mc {dst} {src} {c};"),
                    FftOp::Copy { dst, src } => println!("            cp {dst} {src};"),
                }
            }
            for (parity, &reg) in program.parity_regs.iter().enumerate() {
                println!("            st {parity} {reg};");
            }
            println!("        );");
            println!("    }};");
            println!("}}");
            println!("pub(crate) use fft_program_{name};");
        }
    }

    // the normalized subspace polynomials vanish on their span and hit 1 at
    // their pivot
    #[test]
    fn table_shape() {
        for level in 0..LEVELS {
            for (point, &value) in WHAT[level].iter().enumerate().take(1usize << level) {
                assert_eq!(value, 0, "level {level} point {point}");
            }
            assert_eq!(WHAT[level][1usize << level], 1, "pivot level {level}");
        }
        // Linearity of the underlying map survives normalization.
        for level in 0..LEVELS {
            for a in 0..256usize {
                let b = 0x5a;
                assert_eq!(
                    WHAT[level][a ^ b],
                    WHAT[level][a] ^ WHAT[level][b],
                    "linearity level {level} a {a}"
                );
            }
        }
    }

    // basis polynomial with index i evaluated at x
    fn basis_eval(i: usize, x: usize) -> u8 {
        let mut product = 1u8;
        for level in 0..LEVELS {
            if i & (1 << level) != 0 {
                product = galois::mul(product, WHAT[level][x]);
            }
        }
        product
    }

    fn slow_eval(coeffs: &[u8], x: usize) -> u8 {
        let mut value = 0u8;
        for (i, &a) in coeffs.iter().enumerate() {
            value ^= galois::mul(a, basis_eval(i, x));
        }
        value
    }

    fn pseudo(seed: &mut u32) -> u8 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed >> 16) as u8
    }

    fn transform_only_program(n: usize, beta: usize, inverse_after: bool) -> FftProgram {
        let mut builder = Builder { items: Vec::new(), next_register: 0 };
        let regs = builder.alloc(n);
        builder.fft(&regs, beta);
        if inverse_after {
            builder.ifft(&regs, beta);
        }
        FftProgram {
            ops: flatten_items(&builder.items),
            register_count: n,
            parity_regs: regs.clone(),
            mult_count: 0,
        }
    }

    // the compiled fft matches direct basis evaluation on every coset
    #[test]
    fn fft_matches_slow() {
        let mut seed = 0xC0FFEE;
        for &n in &[2usize, 4, 8, 16, 32] {
            for &beta in &[0usize, n, 2 * n, 96, 128, 224] {
                let coeffs: Vec<u8> = (0..n).map(|_| pseudo(&mut seed)).collect();
                let program = transform_only_program(n, beta, false);
                let values = run_program_bytes(&program, &coeffs);
                for (i, &value) in values.iter().enumerate() {
                    assert_eq!(
                        value,
                        slow_eval(&coeffs, beta ^ i),
                        "n {n} beta {beta} i {i}"
                    );
                }
            }
        }
    }

    // the inverse transform undoes the forward transform
    #[test]
    fn ifft_inverts() {
        let mut seed = 0xBEEF;
        for &n in &[2usize, 4, 8, 16, 32] {
            for &beta in &[0usize, n, 64, 160] {
                let coeffs: Vec<u8> = (0..n).map(|_| pseudo(&mut seed)).collect();
                let program = transform_only_program(n, beta, true);
                let round_trip = run_program_bytes(&program, &coeffs);
                assert_eq!(round_trip, coeffs, "n {n} beta {beta}");
            }
        }
    }

    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (1, 4),
        (2, 2),
        (3, 5),
        (4, 2),
        (5, 3),
        (6, 4),
        (7, 13),
        (9, 3),
        (10, 10),
        (10, 7),
        (11, 5),
        (13, 7),
        (14, 14),
        (16, 4),
        (17, 17),
        (18, 6),
        (20, 10),
        (24, 8),
        (32, 32),
        (37, 21),
    ];

    // compiled programs reproduce the generator matrix parity byte for byte
    #[test]
    fn program_matches_matrix() {
        let mut seed = 0x7A9E;
        for &(k, m) in SHAPES {
            let rs = crate::ReedSolomon::new(k, m).expect("codec should build");
            let flat = build_encode_program(k, m);
            let staged = build_staged_program(k, m);

            let len = 64usize;
            let mut shards: Vec<Vec<u8>> = (0..k)
                .map(|_| (0..len).map(|_| pseudo(&mut seed)).collect())
                .collect();
            shards.extend((0..m).map(|_| vec![0u8; len]));
            rs.encode_scalar(&mut shards).expect("encode should succeed");

            #[allow(clippy::needless_range_loop)] // byte position indexes k shards at once
            for byte in 0..len {
                let column: Vec<u8> = (0..k).map(|shard| shards[shard][byte]).collect();
                let flat_parity = run_program_bytes(&flat, &column);
                let staged_parity = run_staged_bytes(&staged, &column);
                for p in 0..m {
                    assert_eq!(
                        flat_parity[p], shards[k + p][byte],
                        "flat k {k} m {m} parity {p} byte {byte}"
                    );
                    assert_eq!(
                        staged_parity[p], shards[k + p][byte],
                        "staged k {k} m {m} parity {p} byte {byte}"
                    );
                }
            }
        }
    }

    // top of field exploration: compiled cost vs schoolbook across rates
    #[test]
    #[ignore]
    fn top_of_field_counts() {
        for &(k, m) in &[
            (32usize, 32usize),
            (64, 64),
            (128, 128),
            (200, 55),
            (200, 20),
            (230, 25),
            (250, 5),
            (100, 156),
        ] {
            let staged = build_staged_program(k, m);
            eprintln!(
                "({k},{m}): staged {} mults / {} xors / {} boundary / {} glue / {} regs / {} stages (schoolbook {})",
                staged.mult_count,
                staged.xor_count,
                staged.boundary_count,
                staged.glue.len(),
                staged.register_count,
                staged.stages.len(),
                k * m
            );
        }
    }

    // the compiled op counts undercut the schoolbook multiply count at the
    // production shapes, for both lowerings
    #[test]
    fn program_op_counts() {
        for &(k, m) in &[(7usize, 13usize), (10, 10), (14, 14), (16, 16), (18, 6)] {
            let flat = build_encode_program(k, m);
            let staged = build_staged_program(k, m);
            assert!(
                flat.mult_count < k * m,
                "flat k {k} m {m}: {} mults vs schoolbook {}",
                flat.mult_count,
                k * m
            );
            assert!(
                staged.mult_count < k * m,
                "staged k {k} m {m}: {} mults vs schoolbook {}",
                staged.mult_count,
                k * m
            );
            eprintln!(
                "({k},{m}): flat {} mults / {} regs, staged {} mults / {} xors / {} boundary / {} glue / {} regs / {} stages (schoolbook {})",
                flat.mult_count,
                flat.register_count,
                staged.mult_count,
                staged.xor_count,
                staged.boundary_count,
                staged.glue.len(),
                staged.register_count,
                staged.stages.len(),
                k * m
            );
        }
    }
}
