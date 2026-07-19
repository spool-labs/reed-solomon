//! NEON executor for the compiled FFT encode programs
//!
//! Runs the generated straight-line programs from fft_programs.rs over
//! 16-byte strips with every program register in a NEON register. The last
//! strip overlaps the previous one instead of taking a scalar tail; parity
//! bytes are pure functions of data bytes, so recomputing them is idempotent.
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use crate::galois;
use crate::fft::{FftOp, StagedProgram, StagedStage, MAX_STAGED_REGISTERS};
use crate::fft_programs::{
    fft_program_10_10, fft_program_14_14, fft_program_16_16, fft_program_18_6,
    fft_program_32_32, fft_program_7_13, FFT_REGS_10_10, FFT_REGS_14_14, FFT_REGS_16_16,
    FFT_REGS_18_6, FFT_REGS_32_32, FFT_REGS_7_13,
};

/// Strip width in bytes, one NEON vector
const STRIP: usize = 16;

/// Per-coefficient lo and hi nibble tables, 16 bytes each, compile time
static NIBBLE_PAIRS: [[u8; 32]; 256] = gen_nibble_pairs();

const fn gen_nibble_pairs() -> [[u8; 32]; 256] {
    let mul = galois::gen_mul_table();
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

/// The engine the expanded programs call into, one method per op form
struct Engine;

impl Engine {
    #[inline(always)]
    unsafe fn load(ptr: *const u8, off: usize) -> uint8x16_t {
        vld1q_u8(ptr.add(off))
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u8, off: usize, value: uint8x16_t) {
        vst1q_u8(ptr.add(off), value)
    }

    #[inline(always)]
    unsafe fn mul_of(value: uint8x16_t, c: u8) -> uint8x16_t {
        let pair = NIBBLE_PAIRS[c as usize].as_ptr();
        let lo = vqtbl1q_u8(vld1q_u8(pair), vandq_u8(value, vdupq_n_u8(0x0f)));
        let hi = vqtbl1q_u8(vld1q_u8(pair.add(16)), vshrq_n_u8::<4>(value));
        veorq_u8(lo, hi)
    }

    #[inline(always)]
    unsafe fn mul_xor(dst: uint8x16_t, src: uint8x16_t, c: u8) -> uint8x16_t {
        veorq_u8(dst, Self::mul_of(src, c))
    }

    #[inline(always)]
    unsafe fn xor(dst: uint8x16_t, src: uint8x16_t) -> uint8x16_t {
        veorq_u8(dst, src)
    }
}

/// Engine for cores compiled with the sha3 feature: every accumulating
/// multiply folds its nibble combine and the accumulate into one veor3q,
/// which removes one instruction at every such site
struct Sha3Engine;

impl Sha3Engine {
    #[inline(always)]
    unsafe fn load(ptr: *const u8, off: usize) -> uint8x16_t {
        Engine::load(ptr, off)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u8, off: usize, value: uint8x16_t) {
        Engine::store(ptr, off, value)
    }

    /// A bare multiply has no accumulate to fold, so it stays two-way
    #[allow(dead_code)] // the current generated programs carry no mc ops
    #[inline(always)]
    unsafe fn mul_of(value: uint8x16_t, c: u8) -> uint8x16_t {
        Engine::mul_of(value, c)
    }

    #[inline(always)]
    unsafe fn mul_xor(dst: uint8x16_t, src: uint8x16_t, c: u8) -> uint8x16_t {
        let pair = NIBBLE_PAIRS[c as usize].as_ptr();
        let lo = vqtbl1q_u8(vld1q_u8(pair), vandq_u8(src, vdupq_n_u8(0x0f)));
        let hi = vqtbl1q_u8(vld1q_u8(pair.add(16)), vshrq_n_u8::<4>(src));
        veor3q_u8(dst, lo, hi)
    }

    #[inline(always)]
    unsafe fn xor(dst: uint8x16_t, src: uint8x16_t) -> uint8x16_t {
        Engine::xor(dst, src)
    }
}

/// The one method the two engines disagree on, for the generic staged path
trait MulXorEngine {
    /// registers combine as dst xor (c times src)
    unsafe fn mul_xor(dst: uint8x16_t, src: uint8x16_t, c: u8) -> uint8x16_t;
}

impl MulXorEngine for Engine {
    #[inline(always)]
    unsafe fn mul_xor(dst: uint8x16_t, src: uint8x16_t, c: u8) -> uint8x16_t {
        Engine::mul_xor(dst, src, c)
    }
}

impl MulXorEngine for Sha3Engine {
    #[inline(always)]
    unsafe fn mul_xor(dst: uint8x16_t, src: uint8x16_t, c: u8) -> uint8x16_t {
        Sha3Engine::mul_xor(dst, src, c)
    }
}

macro_rules! fft_strip_loop {
    ($program:ident, $engine:ty, $data:ident, $parity:ident, $regs:ident, $k:literal, $m:literal) => {{
        let len = $data[0].as_ref().len();
        let input: [*const u8; $k] = core::array::from_fn(|i| $data[i].as_ref().as_ptr());
        let output: [*mut u8; $m] =
            core::array::from_fn(|o| $parity[o].as_mut().as_mut_ptr());

        let mut pos = 0usize;
        loop {
            let off = if pos + STRIP <= len { pos } else { len - STRIP };
            let mut r = [vdupq_n_u8(0); $regs];
            $program!($engine, r, input, output, off);
            if off + STRIP >= len {
                break;
            }
            pos += STRIP;
        }
    }};
}

macro_rules! fft_executor {
    ($name:ident, $core:ident, $core_sha3:ident, $program:ident, $regs:ident, $k:literal, $m:literal) => {
        /// Encode one production shape through its compiled program
        ///
        /// Caller guarantees the shard counts match the shape and all shards
        /// share one length of at least the strip width.
        pub(crate) fn $name<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            data: &[In],
            parity: &mut [Out],
        ) {
            debug_assert_eq!(data.len(), $k);
            debug_assert_eq!(parity.len(), $m);
            debug_assert!(data[0].as_ref().len() >= STRIP);
            // SAFETY: NEON is aarch64 baseline and the sha3 core only runs
            // when detection says so; counts and lengths are the caller's
            // contract, checked above in debug builds.
            if std::arch::is_aarch64_feature_detected!("sha3") {
                unsafe { $core_sha3(data, parity) }
            } else {
                unsafe { $core(data, parity) }
            }
        }

        #[target_feature(enable = "neon")]
        unsafe fn $core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(data: &[In], parity: &mut [Out]) {
            fft_strip_loop!($program, Engine, data, parity, $regs, $k, $m)
        }

        #[target_feature(enable = "neon,sha3")]
        unsafe fn $core_sha3<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            data: &[In],
            parity: &mut [Out],
        ) {
            fft_strip_loop!($program, Sha3Engine, data, parity, $regs, $k, $m)
        }
    };
}

fft_executor!(encode_7_13, core_7_13, core_7_13_sha3, fft_program_7_13, FFT_REGS_7_13, 7, 13);
fft_executor!(encode_10_10, core_10_10, core_10_10_sha3, fft_program_10_10, FFT_REGS_10_10, 10, 10);
fft_executor!(encode_14_14, core_14_14, core_14_14_sha3, fft_program_14_14, FFT_REGS_14_14, 14, 14);
fft_executor!(encode_16_16, core_16_16, core_16_16_sha3, fft_program_16_16, FFT_REGS_16_16, 16, 16);
fft_executor!(encode_18_6, core_18_6, core_18_6_sha3, fft_program_18_6, FFT_REGS_18_6, 18, 6);
fft_executor!(encode_32_32, core_32_32, core_32_32_sha3, fft_program_32_32, FFT_REGS_32_32, 32, 32);
/// Shard length that can ride the FFT path: at least one full strip
#[inline]
pub(crate) fn eligible(len: usize) -> bool {
    len >= STRIP
}

/// Route a shape to its generated executor, reporting whether one exists
///
/// The shape list is the generated GENERATED_SHAPES; adding a shape there
/// and registering it here is the whole cost of generated-tier coverage.
pub(crate) fn encode_generated<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    k: usize,
    m: usize,
    data: &[In],
    parity: &mut [Out],
) -> bool {
    match (k, m) {
        (7, 13) => encode_7_13(data, parity),
        (10, 10) => encode_10_10(data, parity),
        (14, 14) => encode_14_14(data, parity),
        (16, 16) => encode_16_16(data, parity),
        (18, 6) => encode_18_6(data, parity),
        (32, 32) => encode_32_32(data, parity),
        _ => return false,
    }
    true
}


/// Encode a runtime shape through its staged program
///
/// Caller guarantees the shard counts match the program's shape and all
/// shards share one length of at least the strip width.
pub(crate) fn encode_staged<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    debug_assert_eq!(parity.len(), program.parity_regs.len());
    debug_assert!(program.register_count <= MAX_STAGED_REGISTERS);
    debug_assert!(data[0].as_ref().len() >= STRIP);
    // SAFETY: NEON is aarch64 baseline and the sha3 core only runs when
    // detection says so; counts and lengths are the caller's contract,
    // checked above in debug builds.
    if std::arch::is_aarch64_feature_detected!("sha3") {
        unsafe { staged_core_sha3(program, data, parity) }
    } else {
        unsafe { staged_core(program, data, parity) }
    }
}

#[target_feature(enable = "neon")]
unsafe fn staged_core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<Engine, In, Out>(program, data, parity)
}

#[target_feature(enable = "neon,sha3")]
unsafe fn staged_core_sha3<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<Sha3Engine, In, Out>(program, data, parity)
}

#[inline(always)]
unsafe fn staged_core_impl<E: MulXorEngine, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    let len = data[0].as_ref().len();

    // One register file for the whole call; the compiled program defines
    // every register before use, so strips only re-zero the listed ones.
    let mut file = [vdupq_n_u8(0); MAX_STAGED_REGISTERS];
    let regs = file.as_mut_ptr();

    let mut pos = 0usize;
    loop {
        let off = if pos + STRIP <= len { pos } else { len - STRIP };

        for &reg in &program.zero_regs {
            *regs.add(reg as usize) = vdupq_n_u8(0);
        }
        for (i, shard) in data.iter().enumerate() {
            *regs.add(i) = vld1q_u8(shard.as_ref().as_ptr().add(off));
        }

        for stage in &program.stages {
            match *stage {
                StagedStage::Transform { inverse, start, size, consts_start } => {
                    let base = regs.add(start as usize);
                    let consts = program.consts.as_ptr().add(consts_start as usize);
                    run_transform::<E>(base, size as usize, inverse, consts);
                }
                StagedStage::Glue { start, len } => {
                    let ops = &program.glue[start as usize..(start + len) as usize];
                    for op in ops {
                        match *op {
                            FftOp::MulXor { dst, src, c } => {
                                let value = E::mul_xor(
                                    *regs.add(dst as usize),
                                    *regs.add(src as usize),
                                    c,
                                );
                                *regs.add(dst as usize) = value;
                            }
                            FftOp::Xor { dst, src } => {
                                let value =
                                    Engine::xor(*regs.add(dst as usize), *regs.add(src as usize));
                                *regs.add(dst as usize) = value;
                            }
                            FftOp::MulCopy { dst, src, c } => {
                                *regs.add(dst as usize) =
                                    Engine::mul_of(*regs.add(src as usize), c);
                            }
                            FftOp::Copy { dst, src } => {
                                *regs.add(dst as usize) = *regs.add(src as usize);
                            }
                        }
                    }
                }
            }
        }

        for (p, &reg) in program.parity_regs.iter().enumerate() {
            vst1q_u8(parity[p].as_mut().as_mut_ptr().add(off), *regs.add(reg as usize));
        }

        if off + STRIP >= len {
            break;
        }
        pos += STRIP;
    }
}

/// Dispatch one transform to its unrolled block kernel
#[inline(always)]
unsafe fn run_transform<E: MulXorEngine>(
    base: *mut uint8x16_t,
    size: usize,
    inverse: bool,
    consts: *const u8,
) {
    match (size, inverse) {
        (2, false) => fft_block::<2, E>(base, consts),
        (2, true) => ifft_block::<2, E>(base, consts),
        (4, false) => fft_block::<4, E>(base, consts),
        (4, true) => ifft_block::<4, E>(base, consts),
        (8, false) => fft_block::<8, E>(base, consts),
        (8, true) => ifft_block::<8, E>(base, consts),
        (16, false) => fft_block::<16, E>(base, consts),
        (16, true) => ifft_block::<16, E>(base, consts),
        (32, false) => fft_block::<32, E>(base, consts),
        (32, true) => ifft_block::<32, E>(base, consts),
        (64, false) => fft_block::<64, E>(base, consts),
        (64, true) => ifft_block::<64, E>(base, consts),
        (128, false) => fft_block::<128, E>(base, consts),
        (128, true) => ifft_block::<128, E>(base, consts),
        (256, false) => fft_block::<256, E>(base, consts),
        (256, true) => ifft_block::<256, E>(base, consts),
        _ => unreachable!("transform sizes are powers of two up to 256"),
    }
}

/// Forward transform block: levels high to low, one constant per block,
/// nonzero constants pay the multiply, zero blocks fold with xors alone
///
/// The range is copied into a local array whose address never escapes, so
/// the unrolled butterflies run register-resident; only the copies in and
/// out touch the stage register file.
#[inline(always)]
unsafe fn fft_block<const N: usize, E: MulXorEngine>(regs: *mut uint8x16_t, mut consts: *const u8) {
    let mut local = [vdupq_n_u8(0); N];
    for (i, slot) in local.iter_mut().enumerate() {
        *slot = *regs.add(i);
    }

    let levels = N.trailing_zeros() as usize;
    let mut level = levels;
    while level > 0 {
        level -= 1;
        let stride = 1usize << level;
        let mut base = 0usize;
        while base < N {
            let c = *consts;
            consts = consts.add(1);
            let mut i = base;
            if c != 0 {
                while i < base + stride {
                    local[i] = E::mul_xor(local[i], local[i + stride], c);
                    local[i + stride] = Engine::xor(local[i + stride], local[i]);
                    i += 1;
                }
            } else {
                while i < base + stride {
                    local[i + stride] = Engine::xor(local[i + stride], local[i]);
                    i += 1;
                }
            }
            base += 2 * stride;
        }
    }

    for (i, value) in local.iter().enumerate() {
        *regs.add(i) = *value;
    }
}

/// Inverse transform block: levels low to high, butterfly order reversed
#[inline(always)]
unsafe fn ifft_block<const N: usize, E: MulXorEngine>(regs: *mut uint8x16_t, mut consts: *const u8) {
    let mut local = [vdupq_n_u8(0); N];
    for (i, slot) in local.iter_mut().enumerate() {
        *slot = *regs.add(i);
    }

    let levels = N.trailing_zeros() as usize;
    for level in 0..levels {
        let stride = 1usize << level;
        let mut base = 0usize;
        while base < N {
            let c = *consts;
            consts = consts.add(1);
            let mut i = base;
            if c != 0 {
                while i < base + stride {
                    local[i + stride] = Engine::xor(local[i + stride], local[i]);
                    local[i] = E::mul_xor(local[i], local[i + stride], c);
                    i += 1;
                }
            } else {
                while i < base + stride {
                    local[i + stride] = Engine::xor(local[i + stride], local[i]);
                    i += 1;
                }
            }
            base += 2 * stride;
        }
    }

    for (i, value) in local.iter().enumerate() {
        *regs.add(i) = *value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // both compiled shapes match the scalar matrix encode on every strip
    // phase, including the overlapped tail
    #[test]
    fn matches_scalar() {
        for &(k, m) in &[(7usize, 13usize), (10, 10)] {
            let rs = crate::ReedSolomon::new(k, m).expect("codec should build");
            for &len in &[16usize, 17, 31, 32, 33, 100, 1000, 4096] {
                let data: Vec<Vec<u8>> = (0..k)
                    .map(|s| (0..len).map(|i| ((i * 31 + s * 17) as u8) ^ 0x5a).collect())
                    .collect();

                let mut shards = data.clone();
                shards.extend((0..m).map(|_| vec![0u8; len]));
                rs.encode_scalar(&mut shards).expect("scalar encode should succeed");
                let want = &shards[k..];

                let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
                let mut got = vec![vec![0u8; len]; m];
                {
                    let mut outs: Vec<&mut [u8]> =
                        got.iter_mut().map(|v| v.as_mut_slice()).collect();
                    if k == 7 {
                        encode_7_13(&ins, &mut outs);
                    } else {
                        encode_10_10(&ins, &mut outs);
                    }
                }
                assert_eq!(got, want, "k={k} m={m} len={len}");
            }
        }
    }
}
