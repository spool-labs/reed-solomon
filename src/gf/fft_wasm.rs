//! wasm simd128 executor for the compiled FFT encode programs
//!
//! Runs the generated straight-line programs from fft_programs.rs over
//! 16-byte strips, and staged programs for runtime shapes, mirroring the
//! NEON executor with swizzle-based multiplies. The module only compiles
//! when the simd128 target feature is on, so no runtime detection exists.
//! The last strip overlaps the previous one instead of taking a scalar
//! tail; parity bytes are pure functions of data bytes, so recomputing
//! them is idempotent.
#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use core::arch::wasm32::*;

use crate::galois;
use crate::fft::{FftOp, StagedProgram, StagedStage, MAX_STAGED_REGISTERS};
use crate::fft_programs::{
    fft_program_10_10, fft_program_14_14, fft_program_16_16, fft_program_18_6,
    fft_program_32_32, fft_program_7_13, FFT_REGS_10_10, FFT_REGS_14_14, FFT_REGS_16_16,
    FFT_REGS_18_6, FFT_REGS_32_32, FFT_REGS_7_13,
};

/// Strip width in bytes, one v128 vector
const STRIP: usize = 16;

use crate::gf::tables::NIBBLE_PAIRS;

/// The engine the expanded programs call into, one method per op form
struct Engine;

impl Engine {
    #[inline(always)]
    unsafe fn load(ptr: *const u8, off: usize) -> v128 {
        v128_load(ptr.add(off) as *const v128)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u8, off: usize, value: v128) {
        v128_store(ptr.add(off) as *mut v128, value)
    }

    #[inline(always)]
    unsafe fn mul_of(value: v128, c: u8) -> v128 {
        let pair = NIBBLE_PAIRS[c as usize].as_ptr();
        let lo = u8x16_swizzle(v128_load(pair as *const v128), v128_and(value, u8x16_splat(0x0f)));
        let hi = u8x16_swizzle(v128_load(pair.add(16) as *const v128), u8x16_shr(value, 4));
        v128_xor(lo, hi)
    }

    #[inline(always)]
    unsafe fn mul_xor(dst: v128, src: v128, c: u8) -> v128 {
        v128_xor(dst, Self::mul_of(src, c))
    }

    #[inline(always)]
    unsafe fn xor(dst: v128, src: v128) -> v128 {
        v128_xor(dst, src)
    }
}

macro_rules! fft_executor {
    ($name:ident, $core:ident, $program:ident, $regs:ident, $k:literal, $m:literal) => {
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
            // SAFETY: simd128 is a compile-time target feature here; counts
            // and lengths are the caller's contract, checked in debug builds.
            unsafe { $core(data, parity) }
        }

        unsafe fn $core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(data: &[In], parity: &mut [Out]) {
            let len = data[0].as_ref().len();
            let input: [*const u8; $k] = core::array::from_fn(|i| data[i].as_ref().as_ptr());
            let output: [*mut u8; $m] =
                core::array::from_fn(|o| parity[o].as_mut().as_mut_ptr());

            let mut pos = 0usize;
            loop {
                let off = if pos + STRIP <= len { pos } else { len - STRIP };
                let mut r = [u8x16_splat(0); $regs];
                $program!(Engine, r, input, output, off);
                if off + STRIP >= len {
                    break;
                }
                pos += STRIP;
            }
        }
    };
}

fft_executor!(encode_7_13, core_7_13, fft_program_7_13, FFT_REGS_7_13, 7, 13);
fft_executor!(encode_10_10, core_10_10, fft_program_10_10, FFT_REGS_10_10, 10, 10);
fft_executor!(encode_14_14, core_14_14, fft_program_14_14, FFT_REGS_14_14, 14, 14);
fft_executor!(encode_16_16, core_16_16, fft_program_16_16, FFT_REGS_16_16, 16, 16);
fft_executor!(encode_18_6, core_18_6, fft_program_18_6, FFT_REGS_18_6, 18, 6);
fft_executor!(encode_32_32, core_32_32, fft_program_32_32, FFT_REGS_32_32, 32, 32);
/// Shard length that can ride the FFT path: at least one full strip
#[inline]
/// simd128 has one SIMD tier, so both the generated and staged executors are
/// available on the same condition: a full strip fits.
pub(crate) fn generated_eligible(len: usize) -> bool {
    len >= STRIP
}

pub(crate) fn staged_eligible(len: usize) -> bool {
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
    // SAFETY: simd128 is a compile-time target feature here; counts and
    // lengths are the caller's contract, checked above in debug builds.
    unsafe { staged_core(program, data, parity) }
}

unsafe fn staged_core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    let len = data[0].as_ref().len();

    // One register file for the whole call; the compiled program defines
    // every register before use, so strips only re-zero the listed ones.
    let mut file = [u8x16_splat(0); MAX_STAGED_REGISTERS];
    let regs = file.as_mut_ptr();

    let mut pos = 0usize;
    loop {
        let off = if pos + STRIP <= len { pos } else { len - STRIP };

        for &reg in &program.zero_regs {
            *regs.add(reg as usize) = u8x16_splat(0);
        }
        for (i, shard) in data.iter().enumerate() {
            *regs.add(i) = v128_load(shard.as_ref().as_ptr().add(off) as *const v128);
        }

        for stage in &program.stages {
            match *stage {
                StagedStage::Transform { inverse, start, size, consts_start } => {
                    let base = regs.add(start as usize);
                    let consts = program.consts.as_ptr().add(consts_start as usize);
                    run_transform(base, size as usize, inverse, consts);
                }
                StagedStage::Glue { start, len } => {
                    let ops = &program.glue[start as usize..(start + len) as usize];
                    for op in ops {
                        match *op {
                            FftOp::MulXor { dst, src, c } => {
                                let value = Engine::mul_xor(
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
            v128_store(
                parity[p].as_mut().as_mut_ptr().add(off) as *mut v128,
                *regs.add(reg as usize),
            );
        }

        if off + STRIP >= len {
            break;
        }
        pos += STRIP;
    }
}

/// Dispatch one transform to its unrolled block kernel
#[inline(always)]
unsafe fn run_transform(base: *mut v128, size: usize, inverse: bool, consts: *const u8) {
    match (size, inverse) {
        (2, false) => fft_block::<2>(base, consts),
        (2, true) => ifft_block::<2>(base, consts),
        (4, false) => fft_block::<4>(base, consts),
        (4, true) => ifft_block::<4>(base, consts),
        (8, false) => fft_block::<8>(base, consts),
        (8, true) => ifft_block::<8>(base, consts),
        (16, false) => fft_block::<16>(base, consts),
        (16, true) => ifft_block::<16>(base, consts),
        (32, false) => fft_block::<32>(base, consts),
        (32, true) => ifft_block::<32>(base, consts),
        (64, false) => fft_block::<64>(base, consts),
        (64, true) => ifft_block::<64>(base, consts),
        (128, false) => fft_block::<128>(base, consts),
        (128, true) => ifft_block::<128>(base, consts),
        (256, false) => fft_block::<256>(base, consts),
        (256, true) => ifft_block::<256>(base, consts),
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
unsafe fn fft_block<const N: usize>(regs: *mut v128, mut consts: *const u8) {
    let mut local = [u8x16_splat(0); N];
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
                    local[i] = Engine::mul_xor(local[i], local[i + stride], c);
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
unsafe fn ifft_block<const N: usize>(regs: *mut v128, mut consts: *const u8) {
    let mut local = [u8x16_splat(0); N];
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
                    local[i] = Engine::mul_xor(local[i], local[i + stride], c);
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
