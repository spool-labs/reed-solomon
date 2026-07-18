# tape-reed-solomon — encode benchmarks

Two benchmarks answer two different questions:

- **Correctness / wire-compat** lives in the crate (`tests/parity.rs`) and diffs the
  pure-Rust **scalar** `reed-solomon-erasure` — no C toolchain, runs anywhere.
- **Performance** lives in `bench-harness/` (a standalone package, *not* a crate
  dependency). It benchmarks against rse **+`simd-accel`** — the C SIMD backend Clay
  actually ships on native — and against **`sia_reed_solomon`**.
  Reproduce: `cd bench-harness && cargo run --release`. x86 GFNI/AVX-512: `bench-harness/run.sh` on a GCP c3.

Speedups are vs the **C-accelerated** rse (the real native delta, not the misleading
scalar-rse number).

## wasm simd128 — Node 24 (local), production shapes

The fused wasm kernel — cached lo/hi nibble tables built once at `new`, all m
outputs per 16-byte block with the `v&0f`/`v>>4` split reused across outputs, and
an overlapped tail (no scalar remainder) — is the **default wasm `encode`**. Same
levers as the x86/NEON work, ported to simd128; fusion is a clear win here (unlike
the M4's wide-cache NEON). Reproduce locally (no GCP): `wasm-bench/run.mjs` under
Node 24. Differential `verify()` (fused == scalar, specialised + fallback shapes,
tail/sub-block lengths) runs first and gates the bench.

MiB/s, single-thread:

| size | (7,13) scalar → fused | (10,10) scalar → fused |
|---|---|---|
| 256 B   | 253 → 4100 (16×) | 324 → 5313 (16×) |
| 512 B   | 252 → 4283 (17×) | 326 → 5485 (17×) |
| 1 KB    | 254 → 4493 (18×) | 323 → 5672 (18×) |
| 1.43 KB | 253 → 3905 (15×) | 315 → 5618 (18×) |
| 10 KB   | 251 → 4579 (18×) | 326 → 5810 (18×) |

**~12–18× over scalar** across the PR-70 per-plane range (100 B–1.43 KB). The
biggest jump vs the old per-coefficient wasm kernel is at the small end, where the
cached tables + overlapped tail removed the per-call rebuild — exactly the x86
small-shard fix. (100 B is noisy run-to-run at these iteration counts; ≥256 B is stable.)

## Methodology note — single-thread, and why

sia's default features are `["parallel", "simd"]` (`parallel = rayon`), and it also
cache-blocks the length dimension into 32 KiB blocks. The harness pins **sia to
single-threaded** (`default-features = false, features = ["simd"]`) because **Clay
parallelises across stripes upstream** (tape-internal; `OPTIMIZATION-STATUS.md`:
"threading inside the crate would just fight it"). sia's rayon threads a *single*
encode across all cores, which would oversubscribe against Clay's stripe-level
parallelism — so the operating point that matters for Clay is one encode, single-
threaded. tape and reed-solomon-erasure are single-threaded already.

## Head-to-head, M4 Max, aarch64 NEON, single-thread, MiB/s (speedup vs rse +simd-accel)

Re-run 2026-07-18 on the current code: the fused NEON kernel (cached lo/hi
tables on the struct, all outputs per 16-byte block, sha3 3-way XOR fold,
overlapped tail) is now the default aarch64 `encode`. An earlier version of
this section, measured before that landed, showed tape at roughly half these
numbers under 100 KB and concluded "no deficit anywhere"; both are corrected
below. Wire-compat gate: `tape == sia == reed-solomon-erasure` parity: **PASS**.

### Shape (7,13), production
| size | rse (C NEON) | tape (fused) | sia |
|---|---|---|---|
| 100 B  |  909 | 4610 (5.1x) | 1496 (1.6x) |
| 1 KB   | 1515 | 5451 (3.6x) | 3479 (2.3x) |
| 10 KB  | 1542 | 5459 (3.5x) | 3601 (2.3x) |
| 100 KB | 1445 | 2499 (1.7x) | 3430 (2.4x) |
| 1 MB   | 1486 | 2542 (1.7x) | 3231 (2.2x) |
| 4 MiB  | 1524 | 2306 (1.5x) | 3264 (2.1x) |

### Shape (10,10)
| size | rse (C NEON) | tape (fused) | sia |
|---|---|---|---|
| 100 B  | 1349 | 6438 (4.8x) | 2015 (1.5x) |
| 1 KB   | 1951 | 7222 (3.7x) | 4539 (2.3x) |
| 10 KB  | 1953 | 7142 (3.7x) | 4632 (2.4x) |
| 100 KB | 1993 | 5374 (2.7x) | 4597 (2.3x) |
| 1 MB   | 1985 | 5327 (2.7x) | 4290 (2.2x) |
| 4 MiB  | 1940 | 3024 (1.6x) | 4231 (2.2x) |

**Reading.** At the sizes Clay actually encodes per plane (100 B to 10 KB),
fused NEON is 3.5-5x over the C backend and 1.5-3x over sia; the old
per-coefficient tape numbers (1851-4573 at (10,10)) are history. Past 100 KB at
(7,13) the picture inverts: the single-pass fused kernel walks k+m concurrent
streams (20 for both shapes) and falls off a cliff once the working set leaves
cache, while sia's 32 KiB length-blocking stays flat, leaving tape at 0.7x sia
at 100 KB to 4 MiB. (10,10) holds up to 1 MB and drops at 4 MiB. A
length-blocked or lower-fusion variant for large shards is the known fix if
those sizes ever matter; the production per-plane range is unaffected.

## Reconstruct, M4 Max, single-thread, MiB/s (payload = k * size)

New table 2026-07-18: reconstruct now inverts the decode matrix once per
erasure pattern (cached on the codec, or held by an explicit PreparedDecoder)
and rebuilds through the same fused kernels as encode. sia exposes no
comparable slice-reconstruct entry point, so the comparison is against rse.

### Erased: 1 data shard (repair)
| shape | size | rse(C) | tape | tapePrepared |
|---|---|---|---|---|
| (7,13)  | 10 KB | 19478 | 33964 (1.7x) | 32102 (1.6x) |
| (7,13)  | 1 MB  | 19858 | 35363 (1.8x) | 35391 (1.8x) |
| (10,10) | 10 KB | 19304 | 39287 (2.0x) | 38495 (2.0x) |
| (10,10) | 1 MB  | 20069 | 31902 (1.6x) | 32431 (1.6x) |

### Erased: the maximal m shards (worst case)
| shape | size | rse(C) | tape | tapePrepared |
|---|---|---|---|---|
| (7,13)  | 1 KB   | 1446 | 4314 (3.0x) | 4359 (3.0x) |
| (7,13)  | 10 KB  | 1529 | 4731 (3.1x) | 4716 (3.1x) |
| (7,13)  | 1 MB   | 1548 | 3813 (2.5x) | 4044 (2.6x) |
| (10,10) | 1 KB   | 1878 | 6385 (3.4x) | 6452 (3.4x) |
| (10,10) | 10 KB  | 1969 | 7118 (3.6x) | 7024 (3.6x) |
| (10,10) | 1 MB   | 2019 | 5439 (2.7x) | 5281 (2.6x) |

**Reading.** Worst-case reconstruct tracks the fused encode numbers, which is
the point: the decode rows go through the same kernels now. The cached-plan
path (`reconstruct`, what Clay calls per layer) matches the explicit
`PreparedDecoder`, so the plan cache is doing its job; prepared decoding
exists for callers that want to skip the per-call pattern lookup and the
codec's plan mutex entirely.

## The threading confound (corrects an earlier version of this doc)

An earlier run left sia on its **default** features, i.e. rayon across all 14 cores,
and reported sia at 5.3× / 12.9× / 17.7× at 100 KB / 1 MB / 4 MiB — "5–8× faster than
tape". That was **single-thread tape vs 14-thread sia**. The entire large-shard gap was
multi-threading; it disappears above. sia's 32 KiB cache-blocking gives **no** single-
thread advantage on the M4 (blocked sia ≈ unblocked tape at 4 MiB), so tape is not
memory-bound on this hardware. (An earlier "(20,20) proves compute-bound" claim was also
wrong — (20,20) halves under both models and distinguishes neither.)

## Build-vs-vendor

**Build is validated on performance.** At Clay's operating point (one encode at
a time, parallelism upstream, per-plane sizes of 100 B to 10 KB) tape beats
both the C backend and sia outright, pure-Rust and tape-owned. The one place
sia still leads on the M4 is 100 KB and up at (7,13), which Clay does not hit;
see the reading above. tape should **not** add rayon (it would fight Clay's
stripe parallelism); this is a case where tape correctly differs from sia.

## Levers

- **Cache-blocking the fused kernel for large shards: open.** The fused
  single-pass kernel reads each input and writes each output exactly once, so
  its traffic is already minimal; what kills it past cache on the M4 is 20
  concurrent streams. A length-blocked or lower-fusion tile for 100 KB+ shards
  is the candidate fix, only worth it if Clay ever encodes at those sizes.
  (`verify` and the bench-only `encode_forced` already block at 32 KiB.)
- **Multi-threading: no.** Parallelism lives upstream in Clay.
- **Fused NEON: kept, default, MEASURED.** An earlier note here said "dropped,
  a wash"; that predated the wasm-style rework (cached tables on the struct,
  overlapped tail, sha3 fold). The current kernel is 3.5-5x over C and
  1.5-3x over sia at production sizes, and is what `encode` runs on aarch64.
- **Fused GFNI (x86_ops.md): kept, MEASURED.** 3.3-6.3x over C on Intel `c3`
  plus AMD Zen 4/5 (see below); `encode` routes there on GFNI hosts.
- **Fused AVX2 (Milan / AVX2-only): added, MEASURED.** 1.5-3.2x over C on
  Zen 3; the fused kernel covers hosts with no GFNI/AVX-512 (`dot_prod_avx2`).
- **Decode path: landed, MEASURED on M4.** Reconstruct inverts once per
  erasure pattern (plan cache plus `PreparedDecoder`), rebuilds through the
  fused kernels, and `verify`/`encode_rows` ride the same tables. 2.5-3.6x
  over rse at worst-case erasures. x86 numbers pending the next GCP run.

## x86 results — GCP c3 (Sapphire Rapids: GFNI + AVX-512), single-thread

Ran 2026-07-18 on an ephemeral `c3-standard-4` (created + deleted; flags: gfni avx512f
avx512bw avx2 ssse3). **Correctness: all x86 kernels EXECUTED for the first time** —
SSSE3/AVX2/AVX-512/GFNI + the fused GFNI path passed the differential-vs-scalar tests
(all 256 coeffs) + the wire-compat parity gate: 10 passed / 0 failed, gate PASS.

MiB/s (speedup vs rse +simd-accel = C AVX2), single-thread. `tape` = `encode` (per-coeff
GFNI dispatch); `tape-gfniFused` = `encode_fused` (`gf::x86_fused`, ternlog multi-output).

### Shape (7,13) — production
| size | rse(C) | tape | tape-gfniFused | sia |
|---|---|---|---|---|
| 100 B  |  699 |  173 (0.25×) |  109 (0.16×) |  979 (1.40×) |
| 1 KB   | 1907 | 1503 (0.79×) |  965 (0.51×) | 3277 (1.72×) |
| 10 KB  | 1924 | 2765 (1.44×) | 6267 (3.26×) | 2588 (1.35×) |
| 100 KB | 1931 | 2327 (1.20×) | 8777 (4.54×) | 2464 (1.28×) |
| 1 MB   | 1363 | 1310 (0.96×) | 5605 (4.11×) | 2299 (1.69×) |
| 4 MiB  |  619 |  765 (1.24×) | 2837 (4.58×) |  919 (1.48×) |

### Shape (10,10)
| size | rse(C) | tape | tape-gfniFused | sia |
|---|---|---|---|---|
| 10 KB  | 2473 | 3553 (1.44×) |  9726 (3.93×) | 3350 (1.35×) |
| 100 KB | 2560 | 3112 (1.22×) | 13615 (5.32×) | 3243 (1.27×) |
| 1 MB   | 1633 | 1675 (1.03×) |  9447 (5.79×) | 3043 (1.86×) |
| 4 MiB  |  882 | 1043 (1.18×) |  3984 (4.52×) | 1164 (1.32×) |

**Reading — x86 is the mirror image of NEON:**
- **The fused GFNI kernel is a multiple, not a wash:** 3.3–5.8× over the C backend and
  **2–4× over single-thread sia** at ≥ 10 KB. This is the payoff of keeping the GFNI-fused
  kernel (and confirms dropping the NEON one — measured, fusing is a wash on NEON, a
  multiple on GFNI). At production (7,13) @ 10 KB: fused **6267 MiB/s (3.26×)**.
- **Plain `tape` `encode` is mediocre on x86** (weak small, modest large; loses to sia most
  sizes) — the per-coefficient GFNI dispatch has too much overhead. **On x86 the default
  `encode` should route to the fused GFNI path** (2–4× faster, no downside ≥ 10 KB).
- **Small shards (100 B–1 KB): both tape paths lose to C and sia** (0.16–0.79×) — per-call
  Rust overhead (`parity_rows()` allocates every encode, closures, bounds checks). Fixable;
  only bites the 100 KB-stripe config (100 B per-plane).
- **sia single-thread on x86 is only 1.3–1.9×** — its earlier 5–18× was rayon.

**Build-vs-vendor, x86:** validated, and tape's GFNI-fused kernel is faster than sia at the
production size and above. Both follow-ups have since landed: x86 `encode` routes through the
fused GFNI path, and every shape now runs on tables cached at construction (the per-call
alloc behind the 100 B-1 KB losses is gone, with the sub-vector tail on the scalar kernel).
The numbers above predate those fixes; re-measure on the next GCP run.

## AMD Zen 3 / Zen 4 / Zen 5 (GCP), single-thread — 2026-07-18

Ran on ephemeral GCP nodes (each created + deleted). This adds the **AMD** leg — Intel `c3` above is the only prior
x86 data — and, on Zen 3, the **first-ever execution of the fused AVX2 kernel**.

| µarch | machine | CPU (Google SKU) | features | fused path exercised |
|-------|---------|------------------|----------|----------------------|
| Zen 3 | `t2d-standard-4` | EPYC 7B13 (Milan) | avx2 ssse3        | **fused AVX2** (`dot_prod_avx2`; `encode_fused` routes here with no GFNI) |
| Zen 4 | `c3d-standard-4` | EPYC 9B14 (Genoa) | +gfni +avx512     | fused GFNI (`encode_ymm`/`dot_prod_gfni`) |
| Zen 5 | `c4d-standard-4` | EPYC 9B45 (Turin) | +gfni +avx512     | fused GFNI |

**Correctness.** All three: `cargo test --release` = 11 passed / 0 failed (incl. the
`fused_avx2_matches_scalar` differential — which actually executes on any AVX2 box — and,
on Zen 4/5, `fused_gfni_matches_scalar`) + wire-compat parity gate PASS. Zen 3 closes the
"AVX2-only unrun" gap: the fused AVX2 encode is now byte-for-byte verified vs scalar on real
Milan silicon.

### Zen 3 (Milan, no GFNI) — the fused AVX2 kernel

`fusedAVX2` = the `gfniFused` column (on an AVX2-only host `encode_fused` dispatches to the
fused AVX2 tiles). MiB/s, single-thread:

Shape (10,10):
| size | rse(C) | single-out avx2 | sia | **fusedAVX2** | vs rse | vs sia |
|---|---|---|---|---|---|---|
| 100 KB | 3268 | 3172 | 3341 | **5010** | 1.53× | 1.50× |
| 1 MB   | 2841 | 2807 | 2918 | **4715** | 1.66× | 1.62× |
| 4 MiB  | 1238 | 1877 | 1886 | **3959** | 3.20× | 2.10× |

Shape (7,13):
| size | rse(C) | sia | **fusedAVX2** | vs rse | vs sia |
|---|---|---|---|---|---|
| 100 KB | 2619 | 2609 | **3732** | 1.43× | 1.43× |
| 1 MB   | 1763 | 2397 | **3532** | 2.00× | 1.47× |
| 4 MiB  | 1041 | 1515 | **3168** | 3.04× | 2.09× |

Output-fusion holds throughput ~flat past L2 (3.2–4.0 GB/s at 4 MiB) while the per-coefficient
paths (rse, sia, single-output avx2) go memory-bound and roughly halve. Small shards (100 B):
fusedAVX2 loses to sia (672 vs 1007 at (7,13)) — the `parity_rows()` per-call alloc.

### Zen 4 / Zen 5 (Genoa/Turin, GFNI) — fused GFNI

Shape (10,10), `fusedGFNI` MiB/s (vs rse C):
| size | Zen4 rse | Zen4 fusedGFNI | Zen5 rse | Zen5 fusedGFNI |
|---|---|---|---|---|
| 100 KB | 3652 | **14923 (4.09×)** | 5901 | **19385 (3.28×)** |
| 1 MB   | 3629 | **14860 (4.10×)** | 4864 | **19072 (3.92×)** |
| 4 MiB  | 1565 | **9914 (6.34×)**  | 2749 | **17252 (6.28×)** |

3.3–6.3× over the C backend on both. Turin (Zen 5) is the absolute-throughput leader (DDR5 +
native 512-bit datapath); its *single-output* AVX-512 columns also hold up far past L2 (7619 @
100 KB, 3983 @ 4 MiB) where Zen 4's double-pumped 512-bit collapses (4283 → 1211).

### firedancer head-to-head (`fd-bench`, x86 C FFI)

`fd-bench` clones the firedancer fork and builds its C `fd_reedsol`. **Byte-compat: firedancer
parity == rse at (7,13) and (10,10)** (MATCH — extends the prior gate). On Zen 3 firedancer
uses its AVX2 backend; on Zen 4/5 its GFNI backend. MiB/s, `tapeFused` = `encode_fused`:

| µarch | shape | size | rse(C) | tapeFused | sia | firedancer |
|---|---|---|---|---|---|---|
| Zen 3 | (7,13)  | 10 KB  | 2411 | 3452 | 2739 | **3666** |
| Zen 3 | (7,13)  | 100 KB | 2598 | 3339 | 2670 | **3683** |
| Zen 3 | (10,10) | 100 KB | 3373 | **5292** | 3312 | **5292** |
| Zen 3 | (10,10) | 1 MB   | 2661 | 4587 | 2874 | **5203** |
| Zen 4 | (7,13)  | 10 KB  | 2763 | **11541** | 4183 | 10598 |
| Zen 4 | (7,13)  | 100 KB | 2728 | **11109** | 3568 | 10073 |
| Zen 4 | (10,10) | 10 KB  | 4641 | 16093 | 7871 | **16833** |
| Zen 4 | (10,10) | 1 MB   | 3606 | 14753 | 4538 | **14811** |
| Zen 5 | (7,13)  | 100 KB | 4455 | 15292 | 5251 | **15991** |
| Zen 5 | (10,10) | 10 KB  | 6300 | 21293 | 10271 | **25025** |
| Zen 5 | (10,10) | 100 KB | 5961 | 19966 | 6708 | **22839** |

**Reading — firedancer:**
- **tape's fused kernels are in firedancer's league** — both 3–6× over the C rse baseline, both
  far above sia, and both byte-compatible with rse.
- **Zen 4, (7,13): tape BEATS firedancer** at 1 KB–100 KB (11541 vs 10598 @ 10 KB) — the
  register-resident `encode_ymm::<7,13>` specialization pays off.
- **(10,10), and Zen 5 broadly: firedancer edges ahead** — ~5% (Zen 4) to ~14–18% (Zen 5) at
  (10,10). Its arithmetic + zero per-call allocation.
- **Small shards (100 B): firedancer dominates tape 2–4×** (fd 2495 vs tape 628 @ (7,13) Zen 3) —
  the `parity_rows()` / per-call setup overhead again, the same gap seen vs sia.

**Verdict.** The fused AVX2 kernel makes **Milan a first-class target** (1.5-3.2x over the C
backend, beats sia), and on GFNI hosts tape **trades blows with firedancer**, winning the
specialized (7,13) and staying within ~15% at (10,10), while remaining pure-Rust and
rse-wire-compatible. The one consistent deficit was 100 B shards (per-call alloc); that
alloc is gone now that every shape encodes from construction-time tables, pending
re-measurement on GCP.

## x86 / AVX benches — how to reproduce (GCP)

`bench-harness/run.sh` runs the single-thread head-to-head (tape's dispatched + fused kernels
vs sia vs rse +simd-accel) and `fd-bench/run.sh` adds the firedancer C-FFI comparison. Machine
types by µarch: `c3-standard-4` (Intel Sapphire Rapids, GFNI+AVX-512), `c3d-standard-4` (Zen 4
Genoa, GFNI), `c4d-standard-4` (Zen 5 Turin, GFNI), `t2d`/`c2d-standard-4` (Zen 3 Milan,
AVX2-only — the box that exercises the fused AVX2 path). Runtime dispatch means running *on* a
box exercises that box's kernels with no build flags. Use an ephemeral-node workflow
(label `purpose=rs-gate-ephemeral`, always delete). Note:
the project's global 12-vCPU quota fits ~one 4-vCPU node at a time — delete each before the next.
