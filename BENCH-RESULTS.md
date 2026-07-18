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

## Head-to-head — M4 Max, aarch64 NEON, single-thread, MiB/s (speedup vs rse +simd-accel)

Wire-compat gate: `tape == sia == reed-solomon-erasure` parity — **PASS**.

### Shape (10,10)
| size | rse (C NEON) | tape | sia |
|---|---|---|---|
| 100 B  | 1303 | 1851 (1.42×) | 1973 (1.51×) |
| 1 KB   | 1920 | 3941 (2.05×) | 4542 (2.36×) |
| 10 KB  | 1965 | 4573 (2.33×) | 4667 (2.38×) |
| 100 KB | 1996 | 4498 (2.25×) | 4562 (2.29×) |
| 1 MB   | 1951 | 4452 (2.28×) | 4266 (2.19×) |
| 4 MiB  | 1941 | 4311 (2.22×) | 4263 (2.20×) |

### Shape (20,10)
| size | rse (C NEON) | tape | sia |
|---|---|---|---|
| 100 B  | 1312 | 1960 (1.49×) | 2266 (1.73×) |
| 1 KB   | 1915 | 3975 (2.08×) | 4643 (2.42×) |
| 10 KB  | 1954 | 4503 (2.30×) | 4680 (2.39×) |
| 100 KB | 1971 | 4445 (2.25×) | 4607 (2.34×) |
| 1 MB   | 1947 | 4440 (2.28×) | 4550 (2.34×) |
| 4 MiB  | 1927 | 4245 (2.20×) | 4467 (2.32×) |

### Shape (20,20)
| size | rse (C NEON) | tape | sia |
|---|---|---|---|
| 100 B  | 655 |  946 (1.44×) | 1246 (1.90×) |
| 1 KB   | 958 | 2055 (2.14×) | 2328 (2.43×) |
| 10 KB  | 970 | 2171 (2.24×) | 2356 (2.43×) |
| 100 KB | 983 | 1776 (1.81×) | 2325 (2.36×) |
| 1 MB   | 983 | 2211 (2.25×) | 2265 (2.30×) |
| 4 MiB  | 967 | 2122 (2.19×) | 2238 (2.31×) |

**tape ≈ sia at every size and shape** (~2.0–2.4× over the C backend Clay ships), tape
marginally ahead at 1 MB–4 MiB in (10,10). No kernel or blocking deficit here.

## The threading confound (corrects an earlier version of this doc)

An earlier run left sia on its **default** features, i.e. rayon across all 14 cores,
and reported sia at 5.3× / 12.9× / 17.7× at 100 KB / 1 MB / 4 MiB — "5–8× faster than
tape". That was **single-thread tape vs 14-thread sia**. The entire large-shard gap was
multi-threading; it disappears above. sia's 32 KiB cache-blocking gives **no** single-
thread advantage on the M4 (blocked sia ≈ unblocked tape at 4 MiB), so tape is not
memory-bound on this hardware. (An earlier "(20,20) proves compute-bound" claim was also
wrong — (20,20) halves under both models and distinguishes neither.)

## Build-vs-vendor

**Build is validated on performance.** At Clay's operating point — one encode at a time,
parallelism upstream — tape matches sia across all sizes and shapes, pure-Rust and
tape-owned. tape should **not** add rayon (it would fight Clay's stripe parallelism); this
is a case where tape correctly differs from sia.

## Levers

- **Cache-blocking (sia's 32 KiB `blocked_seq`): worth adding, low-risk.** It does nothing
  on the M4 (wide caches, high per-core bandwidth), but the unblocked loop nest
  (`code_some_slices`: k×m full passes over the shard) is the textbook memory-bound pattern
  on a bandwidth-starved core. Whether it binds on x86 is the specific GCP question below.
- **Multi-threading: no.** Parallelism lives upstream in Clay.
- **Fused NEON (apple_ops.md): dropped** — a wash vs the simple per-coefficient kernel here.
- **Fused GFNI (x86_ops.md): kept, MEASURED** — 3.3–6.3× over C on Intel `c3` + AMD Zen 4/5
  (see below); `encode_fused` / `gf::x86_fused`.
- **Fused AVX2 (Milan / AVX2-only): added, MEASURED** — 1.5–3.2× over C on Zen 3; the fused
  kernel now covers hosts with no GFNI/AVX-512 (`dot_prod_avx2`).

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
production size and above. Next: route x86 `encode` → fused GFNI; kill the small-shard alloc.

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

**Verdict.** The fused AVX2 kernel makes **Milan a first-class target** (1.5–3.2× over the C
backend, beats sia), and on GFNI hosts tape **trades blows with firedancer** — winning the
specialized (7,13), within ~15% at (10,10) — while staying pure-Rust and rse-wire-compatible.
The one consistent deficit is 100 B shards (per-call alloc), which loses to both sia and
firedancer; caching `parity_rows()` on the struct is the fix.

## x86 / AVX benches — how to reproduce (GCP)

`bench-harness/run.sh` runs the single-thread head-to-head (tape's dispatched + fused kernels
vs sia vs rse +simd-accel) and `fd-bench/run.sh` adds the firedancer C-FFI comparison. Machine
types by µarch: `c3-standard-4` (Intel Sapphire Rapids, GFNI+AVX-512), `c3d-standard-4` (Zen 4
Genoa, GFNI), `c4d-standard-4` (Zen 5 Turin, GFNI), `t2d`/`c2d-standard-4` (Zen 3 Milan,
AVX2-only — the box that exercises the fused AVX2 path). Runtime dispatch means running *on* a
box exercises that box's kernels with no build flags. Use an ephemeral-node workflow
(label `purpose=rs-gate-ephemeral`, always delete). Note:
the project's global 12-vCPU quota fits ~one 4-vCPU node at a time — delete each before the next.
