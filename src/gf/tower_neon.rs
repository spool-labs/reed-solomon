//! aarch64 NEON fused Karatsuba encode for the GF((2^8)^2) tower field.
//!
//! One streaming pass over the split planes. Per input pair (lo, hi) the sum
//! `lo ^ hi` is formed once in-register; per (output, input) the three GF(2^8)
//! products `t0 = c_lo*lo`, `t1 = (c_hi+c_lo)*sum`, `t2 = (c_hi*L)*hi` are
//! computed (six `vqtbl1q` lookups) with `t0` shared between the two output
//! planes:
//!   out_lo += t0 + t2
//!   out_hi += t0 + t1
//! That is six lookups per tower coefficient where the dense 2x2 block needs
//! eight, so encode and reconstruct run about a third faster. Byte-identical to
//! the dense path (checked in `reedsolomon16`), and the fold uses `veor3q_u8`
//! (SHA3 three-way xor) when the CPU has it.
#![cfg(all(target_arch = "aarch64", not(feature = "scalar")))]

use core::arch::aarch64::*;

use crate::galois16;

/// One fixed-`NOUT` tile: outputs `[base, base+NOUT)` over all `k` inputs.
///
/// # Safety
/// Requires the enabled feature. Every plane is equal length `>= 16`; the three
/// tables each hold at least `(base + NOUT) * k * 32` bytes.
macro_rules! kara_tile {
    ($name:ident, $feat:literal, $acc:ident $x:ident $y:ident => $fold:expr) => {
        #[target_feature(enable = $feat)]
        #[allow(clippy::too_many_arguments)] // one streaming kernel, not an API
        unsafe fn $name<const NOUT: usize>(
            out_lo: &mut [&mut [u8]],
            out_hi: &mut [&mut [u8]],
            lo_in: &[&[u8]],
            hi_in: &[&[u8]],
            c0: *const u8,
            c1: *const u8,
            c2: *const u8,
            k: usize,
            base: usize,
        ) {
            const W: usize = 16;
            let n = lo_in[0].len();
            let mask = vdupq_n_u8(0x0f);
            let lop: [*mut u8; NOUT] = core::array::from_fn(|j| out_lo[j].as_mut_ptr());
            let hop: [*mut u8; NOUT] = core::array::from_fn(|j| out_hi[j].as_mut_ptr());

            let mut pos = 0usize;
            loop {
                let p = if pos + W <= n { pos } else { n - W };
                let mut acc_lo = [vdupq_n_u8(0); NOUT];
                let mut acc_hi = [vdupq_n_u8(0); NOUT];
                for i in 0..k {
                    let lo_v = vld1q_u8(lo_in.get_unchecked(i).as_ptr().add(p));
                    let hi_v = vld1q_u8(hi_in.get_unchecked(i).as_ptr().add(p));
                    let sum_v = veorq_u8(lo_v, hi_v);
                    let lo_l = vandq_u8(lo_v, mask);
                    let lo_h = vshrq_n_u8::<4>(lo_v);
                    let hi_l = vandq_u8(hi_v, mask);
                    let hi_h = vshrq_n_u8::<4>(hi_v);
                    let su_l = vandq_u8(sum_v, mask);
                    let su_h = vshrq_n_u8::<4>(sum_v);
                    for j in 0..NOUT {
                        let bt = ((base + j) * k + i) * 32;
                        let t0 = veorq_u8(
                            vqtbl1q_u8(vld1q_u8(c0.add(bt)), lo_l),
                            vqtbl1q_u8(vld1q_u8(c0.add(bt + 16)), lo_h),
                        );
                        let t2 = veorq_u8(
                            vqtbl1q_u8(vld1q_u8(c2.add(bt)), hi_l),
                            vqtbl1q_u8(vld1q_u8(c2.add(bt + 16)), hi_h),
                        );
                        let t1 = veorq_u8(
                            vqtbl1q_u8(vld1q_u8(c1.add(bt)), su_l),
                            vqtbl1q_u8(vld1q_u8(c1.add(bt + 16)), su_h),
                        );
                        acc_lo[j] = {
                            let ($acc, $x, $y) = (acc_lo[j], t0, t2);
                            $fold
                        };
                        acc_hi[j] = {
                            let ($acc, $x, $y) = (acc_hi[j], t0, t1);
                            $fold
                        };
                    }
                }
                for j in 0..NOUT {
                    vst1q_u8(lop[j].add(p), acc_lo[j]);
                    vst1q_u8(hop[j].add(p), acc_hi[j]);
                }
                if p + W >= n {
                    break;
                }
                pos += W;
            }
        }
    };
}

kara_tile!(kara_neon, "neon", acc x y => veorq_u8(acc, veorq_u8(x, y)));
kara_tile!(kara_sha3, "neon,sha3", acc x y => veor3q_u8(acc, x, y));

/// Fused Karatsuba encode over split planes.
///
/// `lo_in`/`hi_in` are the `k` input planes; `out_lo`/`out_hi` the `m` output
/// planes, all equal length. `c0`/`c1`/`c2` are the per-constant nibble tables
/// (`c_lo`, `c_hi+c_lo`, `c_hi*L`) laid out as `[(j*k+i)*32]` with a 16-byte lo
/// table then a 16-byte hi table. `rows` supplies the raw coefficients for the
/// short-plane scalar path.
#[allow(clippy::too_many_arguments)] // split-plane tables and buffers, not an API surface
pub fn encode(
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
    rows: &[Vec<u16>],
    lo_in: &[&[u8]],
    hi_in: &[&[u8]],
    out_lo: &mut [&mut [u8]],
    hi_out: &mut [&mut [u8]],
) {
    let k = lo_in.len();
    let m = out_lo.len();
    if m == 0 {
        return;
    }
    let n = if k > 0 { lo_in[0].len() } else { 0 };

    // Too short for a vector block: the scalar tower multiply is simplest here.
    if n < 16 {
        for j in 0..m {
            for x in 0..n {
                let mut acc = 0u16;
                for i in 0..k {
                    acc ^= galois16::mul(rows[j][i], galois16::pack(hi_in[i][x], lo_in[i][x]));
                }
                out_lo[j][x] = galois16::low(acc);
                hi_out[j][x] = galois16::high(acc);
            }
        }
        return;
    }

    let sha3 = std::arch::is_aarch64_feature_detected!("sha3");
    let (p0, p1, p2) = (c0.as_ptr(), c1.as_ptr(), c2.as_ptr());
    // SAFETY: n >= 16; tables cover all m outputs; feature checked; planes equal
    // length upheld by the caller.
    unsafe {
        let mut o = 0usize;
        while o < m {
            let t = (m - o).min(6);
            let ol = &mut out_lo[o..o + t];
            let oh = &mut hi_out[o..o + t];
            // The tile width is a const generic, so it has to be enumerated.
            // Choosing the engine first keeps that enumeration to one copy.
            macro_rules! tile {
                ($kernel:ident) => {
                    match t {
                        6 => $kernel::<6>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                        5 => $kernel::<5>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                        4 => $kernel::<4>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                        3 => $kernel::<3>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                        2 => $kernel::<2>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                        _ => $kernel::<1>(ol, oh, lo_in, hi_in, p0, p1, p2, k, o),
                    }
                };
            }
            if sha3 {
                tile!(kara_sha3)
            } else {
                tile!(kara_neon)
            }
            o += t;
        }
    }
}
