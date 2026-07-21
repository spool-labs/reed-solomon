//! GFNI executor for the compiled GF((2^8)^2) staged FFT encode programs
//!
//! The x86 mirror of `fft16_neon`: op-major over cache-blocked strips,
//! registers as `STRIP`-symbol plane pairs in one flat arena, every butterfly
//! or glue op a single fused streaming pass with its constants held in
//! vectors. Multiplies run through `gf2p8affineqb` against the compile-time
//! [`AFFINE`] table; every constant the 16-bit programs carry is a GF(2^8)
//! byte (the Karatsuba triple), so one affine matrix per byte covers all of
//! it. Constants in the GF(2^8) subfield act per plane, one affine each; full
//! tower constants apply the dense 2x2 block, four affines per pair.
//!
//! Cores exist for ymm (VEX GFNI plus AVX2, covering every GFNI host) and zmm
//! (AVX-512), picked at run time.
#![cfg(gf16_x86_enabled)]

use core::arch::x86_64::*;

use crate::fft16::{FftOp16, StagedProgram16, StagedStage16, MAX_STAGED_REGISTERS16};
use crate::gf::gfni::AFFINE;

/// Symbols per strip register. Wider than the NEON executor's 256 because the
/// zmm core moves 64 bytes per load and wants the longer run to amortize it.
const STRIP: usize = 512;

/// The vector operations one core needs, implemented per register width.
trait Vec8: Copy {
    const WIDTH: usize;
    unsafe fn load(p: *const u8) -> Self;
    unsafe fn store(p: *mut u8, v: Self);
    unsafe fn xor(a: Self, b: Self) -> Self;
    /// Broadcast the affine matrix of multiply-by-`c`.
    unsafe fn matrix(c: u8) -> Self;
    /// `matrix * v` per byte through gf2p8affineqb.
    unsafe fn affine(v: Self, m: Self) -> Self;
}

impl Vec8 for __m256i {
    const WIDTH: usize = 32;

    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        _mm256_loadu_si256(p as *const __m256i)
    }

    #[inline(always)]
    unsafe fn store(p: *mut u8, v: Self) {
        _mm256_storeu_si256(p as *mut __m256i, v)
    }

    #[inline(always)]
    unsafe fn xor(a: Self, b: Self) -> Self {
        _mm256_xor_si256(a, b)
    }

    #[inline(always)]
    unsafe fn matrix(c: u8) -> Self {
        _mm256_set1_epi64x(AFFINE[c as usize])
    }

    #[inline(always)]
    unsafe fn affine(v: Self, m: Self) -> Self {
        _mm256_gf2p8affine_epi64_epi8::<0>(v, m)
    }
}

impl Vec8 for __m512i {
    const WIDTH: usize = 64;

    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        _mm512_loadu_si512(p as *const __m512i)
    }

    #[inline(always)]
    unsafe fn store(p: *mut u8, v: Self) {
        _mm512_storeu_si512(p as *mut __m512i, v)
    }

    #[inline(always)]
    unsafe fn xor(a: Self, b: Self) -> Self {
        _mm512_xor_si512(a, b)
    }

    #[inline(always)]
    unsafe fn matrix(c: u8) -> Self {
        _mm512_set1_epi64(AFFINE[c as usize])
    }

    #[inline(always)]
    unsafe fn affine(v: Self, m: Self) -> Self {
        _mm512_gf2p8affine_epi64_epi8::<0>(v, m)
    }
}

/// One register pair inside the arena.
#[derive(Clone, Copy)]
struct Reg {
    lo: *mut u8,
    hi: *mut u8,
}

#[inline(always)]
unsafe fn reg(arena: *mut u8, index: usize) -> Reg {
    let lo = arena.add(index * 2 * STRIP);
    Reg { lo, hi: lo.add(STRIP) }
}

/// Fused butterfly over `s` bytes per plane: forward is `A ^= c*B; B ^= A`,
/// inverse is `B ^= A; A ^= c*B`. The caller guarantees `c > 1`.
#[inline(always)]
unsafe fn butterfly<V: Vec8, const INV: bool>(a: Reg, b: Reg, c: u16, kara: &[u8; 3], s: usize) {
    if c < 256 {
        // Subfield constant: the tower multiply acts per plane.
        let m = V::matrix(kara[0]);
        let mut x = 0usize;
        while x < s {
            let mut b_lo = V::load(b.lo.add(x));
            let mut b_hi = V::load(b.hi.add(x));
            if INV {
                b_lo = V::xor(b_lo, V::load(a.lo.add(x)));
                b_hi = V::xor(b_hi, V::load(a.hi.add(x)));
            }
            let a_lo = V::xor(V::load(a.lo.add(x)), V::affine(b_lo, m));
            let a_hi = V::xor(V::load(a.hi.add(x)), V::affine(b_hi, m));
            if !INV {
                b_lo = V::xor(b_lo, a_lo);
                b_hi = V::xor(b_hi, a_hi);
            }
            V::store(a.lo.add(x), a_lo);
            V::store(a.hi.add(x), a_hi);
            V::store(b.lo.add(x), b_lo);
            V::store(b.hi.add(x), b_hi);
            x += V::WIDTH;
        }
    } else {
        // Dense 2x2 from the stored triple: m00 = k1, m01 = k1 + k0,
        // m10 = k2, m11 = k0.
        let m00 = V::matrix(kara[1]);
        let m01 = V::matrix(kara[1] ^ kara[0]);
        let m10 = V::matrix(kara[2]);
        let m11 = V::matrix(kara[0]);
        let mut x = 0usize;
        while x < s {
            let mut b_lo = V::load(b.lo.add(x));
            let mut b_hi = V::load(b.hi.add(x));
            if INV {
                b_lo = V::xor(b_lo, V::load(a.lo.add(x)));
                b_hi = V::xor(b_hi, V::load(a.hi.add(x)));
            }
            let a_lo = V::xor(
                V::xor(V::load(a.lo.add(x)), V::affine(b_hi, m10)),
                V::affine(b_lo, m11),
            );
            let a_hi = V::xor(
                V::xor(V::load(a.hi.add(x)), V::affine(b_hi, m00)),
                V::affine(b_lo, m01),
            );
            if !INV {
                b_lo = V::xor(b_lo, a_lo);
                b_hi = V::xor(b_hi, a_hi);
            }
            V::store(a.lo.add(x), a_lo);
            V::store(a.hi.add(x), a_hi);
            V::store(b.lo.add(x), b_lo);
            V::store(b.hi.add(x), b_hi);
            x += V::WIDTH;
        }
    }
}

/// `B ^= A` over both planes: the zero-constant butterfly, same both ways.
#[inline(always)]
unsafe fn fold<V: Vec8>(a: Reg, b: Reg, s: usize) {
    let mut x = 0usize;
    while x < s {
        V::store(b.lo.add(x), V::xor(V::load(b.lo.add(x)), V::load(a.lo.add(x))));
        V::store(b.hi.add(x), V::xor(V::load(b.hi.add(x)), V::load(a.hi.add(x))));
        x += V::WIDTH;
    }
}

/// The identity-constant butterfly: `A ^= B; B ^= A` (or reversed).
#[inline(always)]
unsafe fn butterfly_one<V: Vec8, const INV: bool>(a: Reg, b: Reg, s: usize) {
    let mut x = 0usize;
    while x < s {
        let mut a_lo = V::load(a.lo.add(x));
        let mut a_hi = V::load(a.hi.add(x));
        let mut b_lo = V::load(b.lo.add(x));
        let mut b_hi = V::load(b.hi.add(x));
        if INV {
            b_lo = V::xor(b_lo, a_lo);
            b_hi = V::xor(b_hi, a_hi);
            a_lo = V::xor(a_lo, b_lo);
            a_hi = V::xor(a_hi, b_hi);
        } else {
            a_lo = V::xor(a_lo, b_lo);
            a_hi = V::xor(a_hi, b_hi);
            b_lo = V::xor(b_lo, a_lo);
            b_hi = V::xor(b_hi, a_hi);
        }
        V::store(a.lo.add(x), a_lo);
        V::store(a.hi.add(x), a_hi);
        V::store(b.lo.add(x), b_lo);
        V::store(b.hi.add(x), b_hi);
        x += V::WIDTH;
    }
}

/// `dst (^)= c * src` over `s` bytes per plane; `ACC` accumulates, otherwise
/// overwrites. The caller guarantees `c > 1`.
#[inline(always)]
unsafe fn mul_into<V: Vec8, const ACC: bool>(d: Reg, src: Reg, c: u16, kara: &[u8; 3], s: usize) {
    if c < 256 {
        let m = V::matrix(kara[0]);
        let mut x = 0usize;
        while x < s {
            let p_lo = V::affine(V::load(src.lo.add(x)), m);
            let p_hi = V::affine(V::load(src.hi.add(x)), m);
            let out_lo = if ACC { V::xor(V::load(d.lo.add(x)), p_lo) } else { p_lo };
            let out_hi = if ACC { V::xor(V::load(d.hi.add(x)), p_hi) } else { p_hi };
            V::store(d.lo.add(x), out_lo);
            V::store(d.hi.add(x), out_hi);
            x += V::WIDTH;
        }
    } else {
        let m00 = V::matrix(kara[1]);
        let m01 = V::matrix(kara[1] ^ kara[0]);
        let m10 = V::matrix(kara[2]);
        let m11 = V::matrix(kara[0]);
        let mut x = 0usize;
        while x < s {
            let s_lo = V::load(src.lo.add(x));
            let s_hi = V::load(src.hi.add(x));
            let p_lo = V::xor(V::affine(s_hi, m10), V::affine(s_lo, m11));
            let p_hi = V::xor(V::affine(s_hi, m00), V::affine(s_lo, m01));
            let out_lo = if ACC { V::xor(V::load(d.lo.add(x)), p_lo) } else { p_lo };
            let out_hi = if ACC { V::xor(V::load(d.hi.add(x)), p_hi) } else { p_hi };
            V::store(d.lo.add(x), out_lo);
            V::store(d.hi.add(x), out_hi);
            x += V::WIDTH;
        }
    }
}

/// `dst = src` over both planes.
#[inline(always)]
unsafe fn copy_into(d: Reg, src: Reg, s: usize) {
    core::ptr::copy_nonoverlapping(src.lo, d.lo, s);
    core::ptr::copy_nonoverlapping(src.hi, d.hi, s);
}

/// Whether this host can run the GFNI executor at all.
#[inline]
pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("gfni") && std::is_x86_feature_detected!("avx2")
}

/// Encode a shape through its staged program.
///
/// Shards are plane pairs `[low N | high N]` of one even length with
/// `N >= 16`; counts match the program's shape; the caller checked
/// [`available`].
pub(crate) fn encode_staged16<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    debug_assert_eq!(parity.len(), program.parity_regs.len());
    debug_assert!(program.register_count <= MAX_STAGED_REGISTERS16);
    debug_assert!(data[0].as_ref().len() / 2 >= 16);
    debug_assert!(available());
    // SAFETY: features checked by the caller through `available` and here for
    // the zmm tier; counts and lengths are the caller's contract, checked
    // above in debug builds.
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
        unsafe { staged_core_zmm(program, data, parity) }
    } else {
        unsafe { staged_core_ymm(program, data, parity) }
    }
}

#[target_feature(enable = "gfni,avx2")]
unsafe fn staged_core_ymm<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<__m256i, In, Out>(program, data, parity)
}

#[target_feature(enable = "gfni,avx512f,avx512bw")]
unsafe fn staged_core_zmm<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<__m512i, In, Out>(program, data, parity)
}

#[inline(always)]
unsafe fn staged_core_impl<V: Vec8, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    let plane = data[0].as_ref().len() / 2;

    // One flat arena for the whole call, zeroed once; the kernels stream over
    // `ceil(s/WIDTH)*WIDTH` bytes, so short tails run on the arena's slack
    // and the copies below carry only the live `s` bytes.
    let mut arena = vec![0u8; program.register_count * 2 * STRIP];
    let regs = arena.as_mut_ptr();

    let mut offset = 0usize;
    while offset < plane {
        let s = STRIP.min(plane - offset);
        let vs = s.next_multiple_of(V::WIDTH).min(STRIP);

        for &r in &program.zero_regs {
            let r = reg(regs, r as usize);
            core::ptr::write_bytes(r.lo, 0, vs);
            core::ptr::write_bytes(r.hi, 0, vs);
        }
        for (i, shard) in data.iter().enumerate() {
            let shard = shard.as_ref().as_ptr();
            let r = reg(regs, i);
            core::ptr::copy_nonoverlapping(shard.add(offset), r.lo, s);
            core::ptr::copy_nonoverlapping(shard.add(plane + offset), r.hi, s);
        }

        for stage in &program.stages {
            match *stage {
                StagedStage16::Transform { inverse, start, size, consts_start } => {
                    if inverse {
                        transform::<V, true>(program, regs, start as usize, size as usize, consts_start, vs);
                    } else {
                        transform::<V, false>(program, regs, start as usize, size as usize, consts_start, vs);
                    }
                }
                StagedStage16::Glue { start, len } => {
                    for index in start as usize..(start + len) as usize {
                        let kara = &program.glue_kara[index];
                        match program.glue[index] {
                            FftOp16::MulXor { dst, src, c } => {
                                let (d, src) = (reg(regs, dst as usize), reg(regs, src as usize));
                                if c == 1 {
                                    fold::<V>(src, d, vs);
                                } else {
                                    mul_into::<V, true>(d, src, c, kara, vs);
                                }
                            }
                            FftOp16::Xor { dst, src } => {
                                fold::<V>(reg(regs, src as usize), reg(regs, dst as usize), vs);
                            }
                            FftOp16::MulCopy { dst, src, c } => {
                                let (d, src) = (reg(regs, dst as usize), reg(regs, src as usize));
                                if c == 1 {
                                    copy_into(d, src, vs);
                                } else {
                                    mul_into::<V, false>(d, src, c, kara, vs);
                                }
                            }
                            FftOp16::Copy { dst, src } => {
                                copy_into(reg(regs, dst as usize), reg(regs, src as usize), vs);
                            }
                        }
                    }
                }
            }
        }

        for (p, &r) in program.parity_regs.iter().enumerate() {
            let shard = parity[p].as_mut().as_mut_ptr();
            let r = reg(regs, r as usize);
            core::ptr::copy_nonoverlapping(r.lo as *const u8, shard.add(offset), s);
            core::ptr::copy_nonoverlapping(r.hi as *const u8, shard.add(plane + offset), s);
        }
        offset += s;
    }
}

/// One in-place transform: levels high to low forward, low to high inverse,
/// one constant per block streamed butterfly by butterfly with its matrices
/// held in vectors. Pruned dead subtrees skip; zero constants fold with xors
/// alone; the identity constant skips the multiply too.
///
/// `inline(always)` is load-bearing: the caller's target features must reach
/// the intrinsics here, or they compile as outlined per-multiply calls.
#[inline(always)]
unsafe fn transform<V: Vec8, const INV: bool>(
    program: &StagedProgram16,
    regs: *mut u8,
    start: usize,
    size: usize,
    consts_start: u32,
    s: usize,
) {
    let levels = size.trailing_zeros() as usize;
    let mut const_index = consts_start as usize;
    for step in 0..levels {
        let level = if INV { step } else { levels - 1 - step };
        let stride = 1usize << level;
        let mut block = 0usize;
        while block < size {
            let c = program.consts[const_index];
            let kara = &program.consts_kara[const_index];
            let skip = program.consts_skip[const_index];
            const_index += 1;
            if !skip {
                for i in block..block + stride {
                    let a = reg(regs, start + i);
                    let b = reg(regs, start + i + stride);
                    if c == 0 {
                        fold::<V>(a, b, s);
                    } else if c == 1 {
                        butterfly_one::<V, INV>(a, b, s);
                    } else {
                        butterfly::<V, INV>(a, b, c, kara, s);
                    }
                }
            }
            block += 2 * stride;
        }
    }
}
