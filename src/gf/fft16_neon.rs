//! NEON executor for the compiled GF((2^8)^2) staged FFT encode programs
//!
//! Executes op-major over cache-blocked strips: registers are `STRIP`-symbol
//! plane pairs in one flat L2-resident arena, and every butterfly or glue op
//! is a single fused streaming pass over its two registers with the constant's
//! tables held in vectors for the whole run. That gives each op sequential,
//! hardware-prefetchable access with fully independent lanes, so the SIMD
//! pipes stay busy; walking the program costs once per strip, amortized over
//! kilobytes instead of a vector width.
//!
//! Multiply forms, cheapest first:
//! - constants in the GF(2^8) subfield (every twiddle of a transform whose
//!   coset offsets stay below 256 lands there) act per plane: four `vqtbl1q`
//!   per 32 symbol-bytes,
//! - full tower constants apply the dense 2x2 block in the butterflies (eight
//!   lookups, no serial sum) and the three-product Karatsuba form in glue
//!   multiplies (six lookups; the accumulate hides its sum chain).
//!
//! Every constant reaches the kernels as GF(2^8) bytes precomputed at program
//! build, so all table access stays inside one static 8 KB nibble table.
#![cfg(all(target_arch = "aarch64", not(feature = "scalar")))]

use core::arch::aarch64::*;

use crate::fft16::{FftOp16, StagedProgram16, StagedStage16, MAX_STAGED_REGISTERS16};

/// Symbols per strip register. The arena is `register_count * 2 * STRIP`
/// bytes, sized for L2 at the shapes the staged path serves.
const STRIP: usize = 256;

use crate::gf::tables::NIBBLE_PAIRS;

/// The nibble table pair of one GF(2^8) constant, loaded into registers.
#[derive(Clone, Copy)]
struct Tables {
    lo: uint8x16_t,
    hi: uint8x16_t,
}

#[inline(always)]
unsafe fn tables(c: u8) -> Tables {
    let pair = NIBBLE_PAIRS[c as usize].as_ptr();
    Tables { lo: vld1q_u8(pair), hi: vld1q_u8(pair.add(16)) }
}

/// Three-way xor: one `veor3q` on cores with the sha3 feature, two `veorq`
/// elsewhere. The one method the two engines disagree on.
trait XorEngine {
    unsafe fn eor3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t;
}

struct Engine;
impl XorEngine for Engine {
    #[inline(always)]
    unsafe fn eor3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        veorq_u8(veorq_u8(a, b), c)
    }
}

struct Sha3Engine;
impl XorEngine for Sha3Engine {
    #[inline(always)]
    unsafe fn eor3(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
        veor3q_u8(a, b, c)
    }
}

/// `c * value` for one plane vector as `eor3(acc, table_lo, table_hi)`.
#[inline(always)]
unsafe fn mul_acc<E: XorEngine>(
    acc: uint8x16_t,
    t: Tables,
    value: uint8x16_t,
    mask: uint8x16_t,
) -> uint8x16_t {
    E::eor3(
        acc,
        vqtbl1q_u8(t.lo, vandq_u8(value, mask)),
        vqtbl1q_u8(t.hi, vshrq_n_u8::<4>(value)),
    )
}

/// `c * value` for one plane vector, no accumulator.
#[inline(always)]
unsafe fn mul_of(t: Tables, value: uint8x16_t, mask: uint8x16_t) -> uint8x16_t {
    veorq_u8(
        vqtbl1q_u8(t.lo, vandq_u8(value, mask)),
        vqtbl1q_u8(t.hi, vshrq_n_u8::<4>(value)),
    )
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
/// inverse is `B ^= A; A ^= c*B`. `c` picks the subfield or Karatsuba form;
/// the caller guarantees `c > 1`.
#[inline(always)]
unsafe fn butterfly<E: XorEngine, const INV: bool>(a: Reg, b: Reg, c: u16, kara: &[u8; 3], s: usize) {
    let mask = vdupq_n_u8(0x0f);
    if c < 256 {
        // Subfield constant: the tower multiply acts per plane.
        let t = tables(kara[0]);
        let mut x = 0usize;
        while x < s {
            let mut b_lo = vld1q_u8(b.lo.add(x));
            let mut b_hi = vld1q_u8(b.hi.add(x));
            if INV {
                b_lo = veorq_u8(b_lo, vld1q_u8(a.lo.add(x)));
                b_hi = veorq_u8(b_hi, vld1q_u8(a.hi.add(x)));
            }
            let a_lo = mul_acc::<E>(vld1q_u8(a.lo.add(x)), t, b_lo, mask);
            let a_hi = mul_acc::<E>(vld1q_u8(a.hi.add(x)), t, b_hi, mask);
            if !INV {
                b_lo = veorq_u8(b_lo, a_lo);
                b_hi = veorq_u8(b_hi, a_hi);
            }
            vst1q_u8(a.lo.add(x), a_lo);
            vst1q_u8(a.hi.add(x), a_hi);
            vst1q_u8(b.lo.add(x), b_lo);
            vst1q_u8(b.hi.add(x), b_hi);
            x += 16;
        }
    } else {
        // Dense 2x2 form: same lookup count as Karatsuba here (the shared t0
        // gives Karatsuba nothing once the sum's extract cost is paid) and a
        // shallower dependency chain. The four dense constants come straight
        // from the stored triple: m00 = k1, m01 = k1+k0, m10 = k2, m11 = k0.
        let m00 = tables(kara[1]);
        let m01 = tables(kara[1] ^ kara[0]);
        let m10 = tables(kara[2]);
        let m11 = tables(kara[0]);
        let mut x = 0usize;
        while x < s {
            let mut b_lo = vld1q_u8(b.lo.add(x));
            let mut b_hi = vld1q_u8(b.hi.add(x));
            if INV {
                b_lo = veorq_u8(b_lo, vld1q_u8(a.lo.add(x)));
                b_hi = veorq_u8(b_hi, vld1q_u8(a.hi.add(x)));
            }
            let a_lo = mul_acc::<E>(
                mul_acc::<E>(vld1q_u8(a.lo.add(x)), m10, b_hi, mask),
                m11,
                b_lo,
                mask,
            );
            let a_hi = mul_acc::<E>(
                mul_acc::<E>(vld1q_u8(a.hi.add(x)), m00, b_hi, mask),
                m01,
                b_lo,
                mask,
            );
            if !INV {
                b_lo = veorq_u8(b_lo, a_lo);
                b_hi = veorq_u8(b_hi, a_hi);
            }
            vst1q_u8(a.lo.add(x), a_lo);
            vst1q_u8(a.hi.add(x), a_hi);
            vst1q_u8(b.lo.add(x), b_lo);
            vst1q_u8(b.hi.add(x), b_hi);
            x += 16;
        }
    }
}

/// `B ^= A` over both planes: the zero-constant butterfly, same both ways.
#[inline(always)]
unsafe fn fold(a: Reg, b: Reg, s: usize) {
    let mut x = 0usize;
    while x < s {
        vst1q_u8(b.lo.add(x), veorq_u8(vld1q_u8(b.lo.add(x)), vld1q_u8(a.lo.add(x))));
        vst1q_u8(b.hi.add(x), veorq_u8(vld1q_u8(b.hi.add(x)), vld1q_u8(a.hi.add(x))));
        x += 16;
    }
}

/// The identity-constant butterfly: `A ^= B; B ^= A` (or reversed).
#[inline(always)]
unsafe fn butterfly_one<const INV: bool>(a: Reg, b: Reg, s: usize) {
    let mut x = 0usize;
    while x < s {
        let mut a_lo = vld1q_u8(a.lo.add(x));
        let mut a_hi = vld1q_u8(a.hi.add(x));
        let mut b_lo = vld1q_u8(b.lo.add(x));
        let mut b_hi = vld1q_u8(b.hi.add(x));
        if INV {
            b_lo = veorq_u8(b_lo, a_lo);
            b_hi = veorq_u8(b_hi, a_hi);
            a_lo = veorq_u8(a_lo, b_lo);
            a_hi = veorq_u8(a_hi, b_hi);
        } else {
            a_lo = veorq_u8(a_lo, b_lo);
            a_hi = veorq_u8(a_hi, b_hi);
            b_lo = veorq_u8(b_lo, a_lo);
            b_hi = veorq_u8(b_hi, a_hi);
        }
        vst1q_u8(a.lo.add(x), a_lo);
        vst1q_u8(a.hi.add(x), a_hi);
        vst1q_u8(b.lo.add(x), b_lo);
        vst1q_u8(b.hi.add(x), b_hi);
        x += 16;
    }
}

/// `dst (^)= c * src` over `s` bytes per plane; `ACC` accumulates, otherwise
/// overwrites. The caller guarantees `c > 1`.
#[inline(always)]
unsafe fn mul_into<E: XorEngine, const ACC: bool>(
    d: Reg,
    src: Reg,
    c: u16,
    kara: &[u8; 3],
    s: usize,
) {
    let mask = vdupq_n_u8(0x0f);
    if c < 256 {
        let t = tables(kara[0]);
        let mut x = 0usize;
        while x < s {
            let s_lo = vld1q_u8(src.lo.add(x));
            let s_hi = vld1q_u8(src.hi.add(x));
            let acc_lo = if ACC { vld1q_u8(d.lo.add(x)) } else { vdupq_n_u8(0) };
            let acc_hi = if ACC { vld1q_u8(d.hi.add(x)) } else { vdupq_n_u8(0) };
            vst1q_u8(d.lo.add(x), mul_acc::<E>(acc_lo, t, s_lo, mask));
            vst1q_u8(d.hi.add(x), mul_acc::<E>(acc_hi, t, s_hi, mask));
            x += 16;
        }
    } else {
        let t0 = tables(kara[0]);
        let t1 = tables(kara[1]);
        let t2 = tables(kara[2]);
        let mut x = 0usize;
        while x < s {
            let s_lo = vld1q_u8(src.lo.add(x));
            let s_hi = vld1q_u8(src.hi.add(x));
            let sum = veorq_u8(s_lo, s_hi);
            let acc_lo = if ACC { vld1q_u8(d.lo.add(x)) } else { vdupq_n_u8(0) };
            let acc_hi = if ACC { vld1q_u8(d.hi.add(x)) } else { vdupq_n_u8(0) };
            let p0 = mul_of(t0, s_lo, mask);
            vst1q_u8(d.lo.add(x), mul_acc::<E>(veorq_u8(acc_lo, p0), t2, s_hi, mask));
            vst1q_u8(d.hi.add(x), mul_acc::<E>(veorq_u8(acc_hi, p0), t1, sum, mask));
            x += 16;
        }
    }
}

/// `dst = src` over both planes.
#[inline(always)]
unsafe fn copy_into(d: Reg, src: Reg, s: usize) {
    core::ptr::copy_nonoverlapping(src.lo, d.lo, s);
    core::ptr::copy_nonoverlapping(src.hi, d.hi, s);
}

/// Encode a shape through its staged program.
///
/// Shards are plane pairs `[low N | high N]` of one even length with
/// `N >= 16`; counts match the program's shape.
pub(crate) fn encode_staged16<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    debug_assert_eq!(parity.len(), program.parity_regs.len());
    debug_assert!(program.register_count <= MAX_STAGED_REGISTERS16);
    debug_assert!(data[0].as_ref().len() / 2 >= 16);
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
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<Engine, In, Out>(program, data, parity)
}

#[target_feature(enable = "neon,sha3")]
unsafe fn staged_core_sha3<In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    staged_core_impl::<Sha3Engine, In, Out>(program, data, parity)
}

#[inline(always)]
unsafe fn staged_core_impl<E: XorEngine, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    program: &StagedProgram16,
    data: &[In],
    parity: &mut [Out],
) {
    let plane = data[0].as_ref().len() / 2;

    // One flat arena for the whole call, zeroed once; the kernels stream over
    // `ceil(s/16)*16` bytes, so short tails run on the arena's slack and the
    // copies below carry only the live `s` bytes to and from the shards.
    let mut arena = vec![0u8; program.register_count * 2 * STRIP];
    let regs = arena.as_mut_ptr();

    let mut offset = 0usize;
    while offset < plane {
        let s = STRIP.min(plane - offset);
        let vs = s.next_multiple_of(16).min(STRIP);

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
                        transform::<E, true>(program, regs, start as usize, size as usize, consts_start, vs);
                    } else {
                        transform::<E, false>(program, regs, start as usize, size as usize, consts_start, vs);
                    }
                }
                StagedStage16::Glue { start, len } => {
                    for index in start as usize..(start + len) as usize {
                        let kara = &program.glue_kara[index];
                        match program.glue[index] {
                            FftOp16::MulXor { dst, src, c } => {
                                let (d, src) = (reg(regs, dst as usize), reg(regs, src as usize));
                                if c == 1 {
                                    fold(src, d, vs);
                                } else {
                                    mul_into::<E, true>(d, src, c, kara, vs);
                                }
                            }
                            FftOp16::Xor { dst, src } => {
                                fold(reg(regs, src as usize), reg(regs, dst as usize), vs);
                            }
                            FftOp16::MulCopy { dst, src, c } => {
                                let (d, src) = (reg(regs, dst as usize), reg(regs, src as usize));
                                if c == 1 {
                                    copy_into(d, src, vs);
                                } else {
                                    mul_into::<E, false>(d, src, c, kara, vs);
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
/// one constant per block streamed butterfly by butterfly with its tables
/// held in vectors. Pruned dead subtrees skip; zero constants fold with xors
/// alone; the identity constant skips the multiply too.
///
/// `inline(always)` is load-bearing: the caller's target features must reach
/// the intrinsics here, or they compile as outlined per-multiply calls.
#[inline(always)]
unsafe fn transform<E: XorEngine, const INV: bool>(
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
                        fold(a, b, s);
                    } else if c == 1 {
                        butterfly_one::<INV>(a, b, s);
                    } else {
                        butterfly::<E, INV>(a, b, c, kara, s);
                    }
                }
            }
            block += 2 * stride;
        }
    }
}
