//! x86_64 fused Karatsuba encode for the GF((2^8)^2) tower field.
//!
//! The x86 twin of [`super::tower_neon`], and it exists for the same reason:
//! applying a tower constant through the dense 2x2 block costs four GF(2^8)
//! products (eight nibble lookups), while the Karatsuba triple costs three
//! (six). Until this kernel, only aarch64 took the cheaper route and every x86
//! host ran the dense `2m x 2k` expansion, paying 4/3 the multiplies.
//!
//! Per input pair the sum `lo ^ hi` is formed once in-register; per (output,
//! input) the three products `t0 = c_lo*lo`, `t1 = (c_hi+c_lo)*sum`,
//! `t2 = (c_hi*L)*hi` are computed with `t0` shared between the output planes:
//!   out_lo += t0 + t2
//!   out_hi += t0 + t1
//!
//! Nibble tables come from `nibble_pair`, so this rides `pshufb`/`vpshufb` and
//! needs no GFNI: it lands on any SSSE3 host and widens on AVX2. Byte-identical
//! to the dense path, which `tower_x86_matches_expanded` checks directly.
#![cfg(gf16_x86_enabled)]

use core::arch::x86_64::*;

use crate::galois16;

/// One fixed-`NOUT` tile: outputs `[base, base+NOUT)` over all `k` inputs.
///
/// The vector width, the load/store/xor/shuffle set, and the nibble-index split
/// are all parameters, so the SSSE3 and AVX2 kernels are one body. x86 has no
/// 8-bit shift, so the high nibble is `srli_epi16::<4>` masked to 0x0f.
///
/// # Safety
/// Requires the enabled feature. Every plane is equal length `>= W`; the three
/// tables each hold at least `(base + NOUT) * k * 32` bytes.
macro_rules! kara_tile {
    (
        $name:ident, $feat:literal, $vec:ty, $width:expr,
        load = $load:ident, store = $store:ident, xor = $xor:ident,
        and = $and:ident, srli = $srli:ident, splat = $splat:ident,
        shuffle = $shuffle:ident, table = $table:ident
    ) => {
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
            const W: usize = $width;
            let n = lo_in[0].len();
            let mask = $splat(0x0f);
            let lop: [*mut u8; NOUT] = core::array::from_fn(|j| out_lo[j].as_mut_ptr());
            let hop: [*mut u8; NOUT] = core::array::from_fn(|j| out_hi[j].as_mut_ptr());

            let mut pos = 0usize;
            loop {
                // Final block backs up to end at `n`, so a plane that is not a
                // whole number of vectors redoes a few bytes instead of
                // dropping to scalar. Writes are pure stores of a full
                // accumulator, so the overlap is idempotent.
                let p = if pos + W <= n { pos } else { n - W };
                let mut acc_lo = [$splat(0); NOUT];
                let mut acc_hi = [$splat(0); NOUT];
                for i in 0..k {
                    let lo_v = $load(lo_in.get_unchecked(i).as_ptr().add(p) as *const $vec);
                    let hi_v = $load(hi_in.get_unchecked(i).as_ptr().add(p) as *const $vec);
                    let sum_v = $xor(lo_v, hi_v);
                    let lo_l = $and(lo_v, mask);
                    let lo_h = $and($srli::<4>(lo_v), mask);
                    let hi_l = $and(hi_v, mask);
                    let hi_h = $and($srli::<4>(hi_v), mask);
                    let su_l = $and(sum_v, mask);
                    let su_h = $and($srli::<4>(sum_v), mask);
                    for j in 0..NOUT {
                        let bt = ((base + j) * k + i) * 32;
                        let t0 = $xor(
                            $shuffle($table(c0.add(bt)), lo_l),
                            $shuffle($table(c0.add(bt + 16)), lo_h),
                        );
                        let t2 = $xor(
                            $shuffle($table(c2.add(bt)), hi_l),
                            $shuffle($table(c2.add(bt + 16)), hi_h),
                        );
                        let t1 = $xor(
                            $shuffle($table(c1.add(bt)), su_l),
                            $shuffle($table(c1.add(bt + 16)), su_h),
                        );
                        acc_lo[j] = $xor(acc_lo[j], $xor(t0, t2));
                        acc_hi[j] = $xor(acc_hi[j], $xor(t0, t1));
                    }
                }
                for j in 0..NOUT {
                    $store(lop[j].add(p) as *mut $vec, acc_lo[j]);
                    $store(hop[j].add(p) as *mut $vec, acc_hi[j]);
                }
                if p + W >= n {
                    break;
                }
                pos += W;
            }
        }
    };
}

// A 16-byte table read as one lane, and the same table broadcast to both lanes
// of a ymm. `vpshufb` indexes within each 128-bit lane, so the AVX2 kernel needs
// the table present in both; the compiler folds this to one `vbroadcasti128`.
#[inline(always)]
unsafe fn table_sse(p: *const u8) -> __m128i {
    _mm_loadu_si128(p as *const __m128i)
}

#[inline(always)]
unsafe fn table_avx(p: *const u8) -> __m256i {
    _mm256_broadcastsi128_si256(_mm_loadu_si128(p as *const __m128i))
}

kara_tile!(
    kara_ssse3, "ssse3", __m128i, 16,
    load = _mm_loadu_si128, store = _mm_storeu_si128, xor = _mm_xor_si128,
    and = _mm_and_si128, srli = _mm_srli_epi16, splat = _mm_set1_epi8,
    shuffle = _mm_shuffle_epi8, table = table_sse
);
kara_tile!(
    kara_avx2, "avx2", __m256i, 32,
    load = _mm256_loadu_si256, store = _mm256_storeu_si256, xor = _mm256_xor_si256,
    and = _mm256_and_si256, srli = _mm256_srli_epi16, splat = _mm256_set1_epi8,
    shuffle = _mm256_shuffle_epi8, table = table_avx
);

/// Whether this host can run the Karatsuba tower kernel at all: it rides
/// `pshufb`, so SSSE3 is the floor. Hosts without it fall back to the dense
/// expansion.
pub(crate) fn available() -> bool {
    std::is_x86_feature_detected!("ssse3")
}

/// Fused Karatsuba encode over split planes.
///
/// `lo_in`/`hi_in` are the `k` input planes; `out_lo`/`out_hi` the `m` output
/// planes, all equal length. `c0`/`c1`/`c2` are the per-constant nibble tables
/// (`c_lo`, `c_hi+c_lo`, `c_hi*L`) laid out as `[(j*k+i)*32]` with a 16-byte lo
/// table then a 16-byte hi table. `rows` supplies the raw coefficients for the
/// short-plane scalar path.
#[allow(clippy::too_many_arguments)] // split-plane tables and buffers, not an API surface
pub(crate) fn encode(
    c0: &[u8],
    c1: &[u8],
    c2: &[u8],
    rows: &[Vec<u16>],
    lo_in: &[&[u8]],
    hi_in: &[&[u8]],
    out_lo: &mut [&mut [u8]],
    out_hi: &mut [&mut [u8]],
) {
    let k = lo_in.len();
    let m = out_lo.len();
    if m == 0 {
        return;
    }
    let n = if k > 0 { lo_in[0].len() } else { 0 };

    let avx2 = std::is_x86_feature_detected!("avx2");
    let width = if avx2 { 32 } else { 16 };

    // Too short for a vector block: the scalar tower multiply is simplest here.
    if n < width {
        for j in 0..m {
            for x in 0..n {
                let mut acc = 0u16;
                for i in 0..k {
                    acc ^= galois16::mul(rows[j][i], galois16::pack(hi_in[i][x], lo_in[i][x]));
                }
                out_lo[j][x] = galois16::low(acc);
                out_hi[j][x] = galois16::high(acc);
            }
        }
        return;
    }

    let (p0, p1, p2) = (c0.as_ptr(), c1.as_ptr(), c2.as_ptr());
    // SAFETY: n >= width; tables cover all m outputs; features checked above;
    // planes equal length upheld by the caller.
    unsafe {
        let mut o = 0usize;
        while o < m {
            // Six live accumulators plus the nibble temporaries fit the 16
            // architectural ymm registers; the same tile depth as the NEON side.
            let t = (m - o).min(6);
            let ol = &mut out_lo[o..o + t];
            let oh = &mut out_hi[o..o + t];
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
            if avx2 {
                tile!(kara_avx2)
            } else {
                tile!(kara_ssse3)
            }
            o += t;
        }
    }
}
