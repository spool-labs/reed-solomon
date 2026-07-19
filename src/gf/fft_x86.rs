//! GFNI executor for the compiled FFT encode programs
//!
//! Runs the generated straight-line programs from fft_programs.rs over
//! 32-byte strips with every program register in a ymm register, multiplying
//! through gf2p8affineqb against the compile-time affine table. Uses only
//! VEX GFNI plus AVX2, so it covers every GFNI host including ones without
//! AVX-512. The last strip overlaps the previous one instead of taking a
//! scalar tail; parity bytes are pure functions of data bytes, so
//! recomputing them is idempotent.
#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

use crate::gf::gfni::AFFINE;
use crate::fft::{FftOp, StagedProgram, StagedStage, MAX_STAGED_REGISTERS};
use crate::fft_programs::{
    fft_program_10_10, fft_program_14_14, fft_program_16_16, fft_program_18_6, fft_program_7_13,
    FFT_REGS_10_10, FFT_REGS_14_14, FFT_REGS_16_16, FFT_REGS_18_6, FFT_REGS_7_13,
};

/// Strip width in bytes, one ymm vector
const STRIP: usize = 32;

/// Strip width of the zmm cores, one 64-byte vector
const ZMM_STRIP: usize = 64;

/// The engine the expanded programs call into, one method per op form
struct Engine;

impl Engine {
    #[inline(always)]
    unsafe fn load(ptr: *const u8, off: usize) -> __m256i {
        _mm256_loadu_si256(ptr.add(off) as *const __m256i)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u8, off: usize, value: __m256i) {
        _mm256_storeu_si256(ptr.add(off) as *mut __m256i, value)
    }

    #[inline(always)]
    unsafe fn mul_of(value: __m256i, c: u8) -> __m256i {
        let matrix = _mm256_set1_epi64x(AFFINE[c as usize]);
        _mm256_gf2p8affine_epi64_epi8::<0>(value, matrix)
    }

    #[inline(always)]
    unsafe fn mul_xor(dst: __m256i, src: __m256i, c: u8) -> __m256i {
        _mm256_xor_si256(dst, Self::mul_of(src, c))
    }

    #[inline(always)]
    unsafe fn xor(dst: __m256i, src: __m256i) -> __m256i {
        _mm256_xor_si256(dst, src)
    }

    #[inline(always)]
    unsafe fn zero() -> __m256i {
        _mm256_setzero_si256()
    }
}

/// zmm engine for AVX-512 GFNI hosts: same program, 64-byte strips, so the
/// per-strip count of loads, stores, and xors halves while the multiply
/// count per byte stays the same
struct ZmmEngine;

impl ZmmEngine {
    #[inline(always)]
    unsafe fn load(ptr: *const u8, off: usize) -> __m512i {
        _mm512_loadu_si512(ptr.add(off) as *const __m512i)
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut u8, off: usize, value: __m512i) {
        _mm512_storeu_si512(ptr.add(off) as *mut __m512i, value)
    }

    #[allow(dead_code)] // the current generated programs carry no mc ops
    #[inline(always)]
    unsafe fn mul_of(value: __m512i, c: u8) -> __m512i {
        let matrix = _mm512_set1_epi64(AFFINE[c as usize]);
        _mm512_gf2p8affine_epi64_epi8::<0>(value, matrix)
    }

    #[inline(always)]
    unsafe fn mul_xor(dst: __m512i, src: __m512i, c: u8) -> __m512i {
        let matrix = _mm512_set1_epi64(AFFINE[c as usize]);
        _mm512_xor_si512(dst, _mm512_gf2p8affine_epi64_epi8::<0>(src, matrix))
    }

    #[inline(always)]
    unsafe fn xor(dst: __m512i, src: __m512i) -> __m512i {
        _mm512_xor_si512(dst, src)
    }

    #[inline(always)]
    unsafe fn zero() -> __m512i {
        _mm512_setzero_si512()
    }
}

macro_rules! fft_strip_loop {
    ($program:ident, $engine:ty, $strip:expr, $data:ident, $parity:ident, $regs:ident, $k:literal, $m:literal) => {{
        let len = $data[0].as_ref().len();
        let input: [*const u8; $k] = core::array::from_fn(|i| $data[i].as_ref().as_ptr());
        let output: [*mut u8; $m] =
            core::array::from_fn(|o| $parity[o].as_mut().as_mut_ptr());

        let mut pos = 0usize;
        loop {
            let off = if pos + $strip <= len { pos } else { len - $strip };
            let mut r = [<$engine>::zero(); $regs];
            $program!($engine, r, input, output, off);
            if off + $strip >= len {
                break;
            }
            pos += $strip;
        }
    }};
}

macro_rules! fft_executor {
    ($name:ident, $core:ident, $core_zmm:ident, $program:ident, $regs:ident, $k:literal, $m:literal) => {
        /// Encode one production shape through its compiled program
        ///
        /// Caller guarantees gfni and avx2 are available, the shard counts
        /// match the shape, and all shards share one length of at least the
        /// strip width. Hosts with AVX-512 GFNI and room for a 64-byte strip
        /// run the zmm core.
        pub(crate) fn $name<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            data: &[In],
            parity: &mut [Out],
        ) {
            debug_assert_eq!(data.len(), $k);
            debug_assert_eq!(parity.len(), $m);
            debug_assert!(data[0].as_ref().len() >= STRIP);
            // SAFETY: the caller checked gfni and avx2 (pinned or detected),
            // the zmm core additionally checks avx512f; counts and lengths
            // are the caller's contract, checked in debug builds.
            if data[0].as_ref().len() >= ZMM_STRIP
                && (cfg!(feature = "gfni") || is_x86_feature_detected!("avx512f"))
            {
                unsafe { $core_zmm(data, parity) }
            } else {
                unsafe { $core(data, parity) }
            }
        }

        #[target_feature(enable = "gfni,avx2")]
        unsafe fn $core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(data: &[In], parity: &mut [Out]) {
            fft_strip_loop!($program, Engine, STRIP, data, parity, $regs, $k, $m)
        }

        #[target_feature(enable = "gfni,avx512f")]
        unsafe fn $core_zmm<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            data: &[In],
            parity: &mut [Out],
        ) {
            fft_strip_loop!($program, ZmmEngine, ZMM_STRIP, data, parity, $regs, $k, $m)
        }
    };
}

fft_executor!(encode_7_13, core_7_13, core_zmm_7_13, fft_program_7_13, FFT_REGS_7_13, 7, 13);
fft_executor!(encode_10_10, core_10_10, core_zmm_10_10, fft_program_10_10, FFT_REGS_10_10, 10, 10);
fft_executor!(encode_14_14, core_14_14, core_zmm_14_14, fft_program_14_14, FFT_REGS_14_14, 14, 14);
fft_executor!(encode_16_16, core_16_16, core_zmm_16_16, fft_program_16_16, FFT_REGS_16_16, 16, 16);
fft_executor!(encode_18_6, core_18_6, core_zmm_18_6, fft_program_18_6, FFT_REGS_18_6, 18, 6);
/// Shard length and host that can ride the FFT path: GFNI plus AVX2 present
/// (pinned or detected) and at least one full strip
#[inline]
pub(crate) fn eligible(len: usize) -> bool {
    let has_gfni = cfg!(feature = "gfni")
        || (is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx2"));
    has_gfni && len >= STRIP
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
        _ => return false,
    }
    true
}


/// Encode a runtime shape through its staged program
///
/// Caller guarantees gfni and avx2 are available, the shard counts match the
/// program's shape, and all shards share one length of at least the strip
/// width.
pub(crate) fn encode_staged<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    debug_assert_eq!(parity.len(), program.parity_regs.len());
    debug_assert!(program.register_count <= MAX_STAGED_REGISTERS);
    debug_assert!(data[0].as_ref().len() >= STRIP);
    // SAFETY: the caller checked gfni and avx2 (pinned or detected); counts
    // and lengths are its contract, checked above in debug builds.
    unsafe { staged_core(program, data, parity) }
}

#[target_feature(enable = "gfni,avx2")]
unsafe fn staged_core<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram,
    data: &[In],
    parity: &mut [Out],
) {
    let len = data[0].as_ref().len();

    // One register file for the whole call; the compiled program defines
    // every register before use, so strips only re-zero the listed ones.
    let mut file = [_mm256_setzero_si256(); MAX_STAGED_REGISTERS];
    let regs = file.as_mut_ptr();

    let mut pos = 0usize;
    loop {
        let off = if pos + STRIP <= len { pos } else { len - STRIP };

        for &reg in &program.zero_regs {
            *regs.add(reg as usize) = _mm256_setzero_si256();
        }
        for (i, shard) in data.iter().enumerate() {
            *regs.add(i) = _mm256_loadu_si256(shard.as_ref().as_ptr().add(off) as *const __m256i);
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
            _mm256_storeu_si256(
                parity[p].as_mut().as_mut_ptr().add(off) as *mut __m256i,
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
unsafe fn run_transform(base: *mut __m256i, size: usize, inverse: bool, consts: *const u8) {
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
unsafe fn fft_block<const N: usize>(regs: *mut __m256i, mut consts: *const u8) {
    let mut local = [_mm256_setzero_si256(); N];
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
unsafe fn ifft_block<const N: usize>(regs: *mut __m256i, mut consts: *const u8) {
    let mut local = [_mm256_setzero_si256(); N];
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

#[cfg(test)]
mod tests {
    use super::*;

    // both compiled shapes match the scalar matrix encode on every strip
    // phase, including the overlapped tail; a no-op without GFNI
    #[test]
    fn matches_scalar() {
        if !(is_x86_feature_detected!("gfni") && is_x86_feature_detected!("avx2")) {
            return;
        }
        for &(k, m) in &[(7usize, 13usize), (10, 10)] {
            let rs = crate::ReedSolomon::new(k, m).expect("codec should build");
            for &len in &[32usize, 33, 63, 64, 65, 100, 1000, 4096] {
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
