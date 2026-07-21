//! aarch64 NEON fused multi-output GF(2^8) encode: one streaming pass over the
//! `k` inputs produces the parity outputs, nibble-split with `vqtbl1q_u8`. The
//! two production shapes hold all outputs register-resident; any other shape is
//! fused by tiling the outputs into `NOUT <= 6` passes. Byte-identical to the
//! per-coefficient path (GF addition is XOR). The accumulate fold uses
//! `veor3q_u8` (SHA3 3-way XOR) when the CPU supports it.
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use super::scalar;

/// Encode shards shorter than one vector block by padding into one
///
/// The per-coefficient scalar route pays m * k table passes, an order of
/// magnitude worse at 15 bytes. Parity is a pure function of the data, so the
/// zero padding only ever produces padding and is dropped with the tail.
#[inline(never)]
fn encode_short<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    tables: &[u8],
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
    len: usize,
) {
    /// One NEON vector
    const BLOCK: usize = 16;
    /// Widest shape the scratch covers, twice the widest generated shape;
    /// wider ones keep the scalar route
    const MAX_SHARDS: usize = 64;

    let (k, m) = (data.len(), parity.len());
    if len == 0 {
        return;
    }

    if k > MAX_SHARDS || m > MAX_SHARDS {
        for (o, out) in parity.iter_mut().enumerate() {
            let out = out.as_mut();
            for (i, shard) in data.iter().enumerate() {
                let c = gen_rows[o].as_ref()[i];
                if i == 0 {
                    scalar::mul_slice(out, shard.as_ref(), c);
                } else {
                    scalar::mul_slice_xor(out, shard.as_ref(), c);
                }
            }
        }
        return;
    }

    let mut in_buf = [[0u8; BLOCK]; MAX_SHARDS];
    let mut out_buf = [[0u8; BLOCK]; MAX_SHARDS];
    for (i, shard) in data.iter().enumerate() {
        in_buf[i][..len].copy_from_slice(shard.as_ref());
    }

    // len is BLOCK here, so this cannot re-enter the short path
    encode_fused(tables, gen_rows, &in_buf[..k], &mut out_buf[..m]);

    for (o, out) in parity.iter_mut().enumerate() {
        out.as_mut().copy_from_slice(&out_buf[o][..len]);
    }
}

/// Fused encode of all parity outputs. `tables` holds, per (output, input)
/// pair, a 16-byte lo table followed by a 16-byte hi table. Works for any
/// shape; shards shorter than one vector block are padded into one.
pub fn encode_fused<Rows: AsRef<[u8]>, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
    tables: &[u8],
    gen_rows: &[Rows],
    data: &[In],
    parity: &mut [Out],
) {
    let (k, m) = (data.len(), parity.len());
    let len = data.first().map(|s| s.as_ref().len()).unwrap_or(0);

    // The kernels below do unchecked table and shard reads.
    assert!(tables.len() >= m * k * 32, "nibble tables do not cover the matrix");
    for shard in data {
        assert_eq!(shard.as_ref().len(), len, "input shards must share one length");
    }
    for shard in parity.iter_mut() {
        assert_eq!(shard.as_mut().len(), len, "parity shards must share one length");
    }

    if len < 16 {
        encode_short(tables, gen_rows, data, parity, len);
        return;
    }

    let sha3 = std::arch::is_aarch64_feature_detected!("sha3");
    // SAFETY: len >= 16; `tables` covers m*k; feature checked; shard invariants
    // upheld by `ReedSolomon::encode`.
    unsafe {
        match (k, m) {
            (7, 13) if sha3 => return resident_sha3::<7, 13, _, _>(tables, data, parity),
            (7, 13) => return resident_neon::<7, 13, _, _>(tables, data, parity),
            (10, 10) if sha3 => return resident_sha3::<10, 10, _, _>(tables, data, parity),
            (10, 10) => return resident_neon::<10, 10, _, _>(tables, data, parity),
            _ => {}
        }
        // General: tile the m outputs into NOUT <= 6 passes.
        let mut o = 0usize;
        while o < m {
            let n = (m - o).min(6);
            let out = &mut parity[o..o + n];
            match (n, sha3) {
                (6, true) => tiled_sha3::<6, _, _>(out, data, tables, o),
                (6, false) => tiled_neon::<6, _, _>(out, data, tables, o),
                (5, true) => tiled_sha3::<5, _, _>(out, data, tables, o),
                (5, false) => tiled_neon::<5, _, _>(out, data, tables, o),
                (4, true) => tiled_sha3::<4, _, _>(out, data, tables, o),
                (4, false) => tiled_neon::<4, _, _>(out, data, tables, o),
                (3, true) => tiled_sha3::<3, _, _>(out, data, tables, o),
                (3, false) => tiled_neon::<3, _, _>(out, data, tables, o),
                (2, true) => tiled_sha3::<2, _, _>(out, data, tables, o),
                (2, false) => tiled_neon::<2, _, _>(out, data, tables, o),
                (_, true) => tiled_sha3::<1, _, _>(out, data, tables, o),
                (_, false) => tiled_neon::<1, _, _>(out, data, tables, o),
            }
            o += n;
        }
    }
}

// Register-resident kernel for a fixed `K x M`: nibble-split each input once,
// hold the `K` split pairs across all `M` outputs (inputs read once).
//
// # Safety
// Requires the enabled feature. `data.len() == K`, `parity.len() == M`, every
// shard equal length `>= 16`; `tables` holds at least `M*K*32` bytes.
macro_rules! resident_kernel {
    ($name:ident, $feat:literal, $acc:ident $lo:ident $hi:ident => $fold:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name<const K: usize, const M: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            tables: &[u8],
            data: &[In],
            parity: &mut [Out],
        ) {
            const W: usize = 16;
            let len = data[0].as_ref().len();
            let low_mask = vdupq_n_u8(0x0f);
            let tp = tables.as_ptr();
            let in_ptr: [*const u8; K] = core::array::from_fn(|i| data[i].as_ref().as_ptr());
            let out_ptr: [*mut u8; M] = core::array::from_fn(|o| parity[o].as_mut().as_mut_ptr());

            let mut pos = 0usize;
            loop {
                let p = if pos + W <= len { pos } else { len - W };
                let mut dl = [vdupq_n_u8(0); K];
                let mut dh = [vdupq_n_u8(0); K];
                for i in 0..K {
                    let v = vld1q_u8(in_ptr[i].add(p));
                    dl[i] = vandq_u8(v, low_mask);
                    dh[i] = vshrq_n_u8::<4>(v);
                }
                for o in 0..M {
                    let mut accv = vdupq_n_u8(0);
                    for i in 0..K {
                        let base = (o * K + i) * 32;
                        let $lo = vqtbl1q_u8(vld1q_u8(tp.add(base)), dl[i]);
                        let $hi = vqtbl1q_u8(vld1q_u8(tp.add(base + 16)), dh[i]);
                        let $acc = accv;
                        accv = $fold;
                    }
                    vst1q_u8(out_ptr[o].add(p), accv);
                }
                if p + W >= len {
                    break;
                }
                pos += W;
            }
        }
    };
}

// Tiled kernel: `NOUT` outputs at a time for a runtime `k`, `base` is the first
// output-row index into `tables`. One input load serves all `NOUT` outputs.
//
// # Safety
// Requires the enabled feature. `out.len() == NOUT`, every shard equal length
// `>= 16`; `tables` holds at least `(base + NOUT) * k * 32` bytes.
macro_rules! tiled_kernel {
    ($name:ident, $feat:literal, $acc:ident $lo:ident $hi:ident => $fold:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $name<const NOUT: usize, In: AsRef<[u8]>, Out: AsMut<[u8]>>(
            out: &mut [Out],
            input: &[In],
            tables: &[u8],
            base: usize,
        ) {
            const W: usize = 16;
            let k = input.len();
            let len = if k > 0 { input[0].as_ref().len() } else { 0 };
            let low_mask = vdupq_n_u8(0x0f);
            let tp = tables.as_ptr();
            let out_ptr: [*mut u8; NOUT] = core::array::from_fn(|j| out[j].as_mut().as_mut_ptr());

            let mut pos = 0usize;
            loop {
                let p = if pos + W <= len { pos } else { len - W };
                let mut accs = [vdupq_n_u8(0); NOUT];
                for i in 0..k {
                    let v = vld1q_u8(input.get_unchecked(i).as_ref().as_ptr().add(p));
                    let lo_idx = vandq_u8(v, low_mask);
                    let hi_idx = vshrq_n_u8::<4>(v);
                    for j in 0..NOUT {
                        let bt = ((base + j) * k + i) * 32;
                        let $lo = vqtbl1q_u8(vld1q_u8(tp.add(bt)), lo_idx);
                        let $hi = vqtbl1q_u8(vld1q_u8(tp.add(bt + 16)), hi_idx);
                        let $acc = accs[j];
                        accs[j] = $fold;
                    }
                }
                for j in 0..NOUT {
                    vst1q_u8(out_ptr[j].add(p), accs[j]);
                }
                if p + W >= len {
                    break;
                }
                pos += W;
            }
        }
    };
}

resident_kernel!(resident_neon, "neon", acc lo hi => veorq_u8(acc, veorq_u8(lo, hi)));
resident_kernel!(resident_sha3, "neon,sha3", acc lo hi => veor3q_u8(acc, lo, hi));
tiled_kernel!(tiled_neon, "neon", acc lo hi => veorq_u8(acc, veorq_u8(lo, hi)));
tiled_kernel!(tiled_sha3, "neon,sha3", acc lo hi => veor3q_u8(acc, lo, hi));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::galois::MUL_TABLE;
    use crate::gf::scalar;

    fn tables(matrix: &[u8], m: usize, k: usize) -> Vec<u8> {
        let mut t = vec![0u8; m * k * 32];
        for (idx, &c) in matrix.iter().enumerate() {
            let row = &MUL_TABLE[c as usize];
            let base = idx * 32;
            for x in 0..16 {
                t[base + x] = row[x];
                t[base + 16 + x] = row[x << 4];
            }
        }
        t
    }

    fn scalar_ref(matrix: &[u8], m: usize, k: usize, data: &[Vec<u8>], len: usize) -> Vec<Vec<u8>> {
        let mut want = vec![vec![0u8; len]; m];
        for o in 0..m {
            for i in 0..k {
                let c = matrix[o * k + i];
                if i == 0 {
                    scalar::mul_slice(&mut want[o], &data[i], c);
                } else {
                    scalar::mul_slice_xor(&mut want[o], &data[i], c);
                }
            }
        }
        want
    }

    /// Differential vs scalar across the resident shapes, tiled shapes (m not a
    /// multiple of 6, m > 6), the len<16 fallback, and boundary lengths.
    #[test]
    fn fused_matches_scalar() {
        let shapes = [(7, 13), (10, 10), (6, 4), (20, 10), (3, 17), (1, 1), (32, 32)];
        for &(k, m) in &shapes {
            for &len in &[0usize, 1, 15, 16, 17, 31, 100, 251, 1000] {
                let matrix: Vec<u8> = (0..m * k)
                    .map(|i| (i as u8).wrapping_mul(37).wrapping_add(1))
                    .collect();
                let gen_rows: Vec<&[u8]> = (0..m).map(|r| &matrix[r * k..(r + 1) * k]).collect();
                let data: Vec<Vec<u8>> = (0..k)
                    .map(|s| (0..len).map(|i| ((i * 31 + s * 17) as u8) ^ 0x5a).collect())
                    .collect();
                let want = scalar_ref(&matrix, m, k, &data, len);
                let tab = tables(&matrix, m, k);
                let ins: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
                let mut got = vec![vec![0u8; len]; m];
                {
                    let mut outs: Vec<&mut [u8]> =
                        got.iter_mut().map(|v| v.as_mut_slice()).collect();
                    encode_fused(&tab, &gen_rows, &ins, &mut outs);
                }
                assert_eq!(got, want, "k={k} m={m} len={len}");
            }
        }
    }
}
