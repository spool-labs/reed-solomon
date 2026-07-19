# tape-reed-solomon: encode benchmarks

Two benchmarks answer two different questions:

- **Correctness / wire-compat** lives in the crate (`tests/parity.rs`) and diffs the
  pure-Rust **scalar** `reed-solomon-erasure`, so it needs no C toolchain and runs anywhere.
- **Performance** lives in `bench-harness/` (a standalone package, *not* a crate
  dependency). It benchmarks against rse **+`simd-accel`** (the C SIMD backend that
  ships in production on native) and against **`sia_reed_solomon`**.

Speedups are vs the **C-accelerated** rse (the real native delta, not the misleading
scalar-rse number). This doc leads with the current state; the run-by-run record,
including corrections to earlier claims, is the **History** log at the bottom.

## Current state (2026-07-19)

**Encode routing.** Every shape picks one of three byte-identical routes, all
interpolating through points 0..k-1 and evaluating at k..n-1 (exactly what the
generator matrix defines, so parity never depends on the route). `encode_route(len)`
reports the choice:

- **Generated FFT programs** for the listed `GENERATED_SHAPES` ((7,13), (10,10),
  (14,14), (16,16), (18,6)), compiled straight-line and register-resident on NEON,
  GFNI (64-byte zmm strips where AVX-512 is present, else 32-byte ymm), and wasm
  simd128. A fraction of the schoolbook multiplies (29 vs 91 at (7,13), 34 vs 100
  at (10,10)). Adding a shape is one generator line plus its per-backend registration.
- **Staged FFT** for any other power-of-two data count, compiled at construction into
  block transforms plus glue over a stack register file.
- **Fused matrix kernels** for every remaining shape and every sub-strip length,
  finishing with an overlapped 32-byte vector tail so only shards shorter than one
  window hit scalar code.

**x86 flagship result (Zen 5 Turin, GFNI + AVX-512, zmm tier).** Same-run
head-to-head vs firedancer, MiB/s single-thread, every cell (2026-07-19 final run;
byte-clean on ymm and zmm cores, parity matched rse/sia/firedancer on all shapes):

| shape | 100 B | 1 KB | 10 KB | 100 KB | 1 MB |
|---|---|---|---|---|---|
| (7,13) tape | 22275 | 44719 | 45837 | 30248 | 13272 |
| (7,13) fd | 8474 | 15435 | 17089 | 15248 | 7348 |
| (10,10) tape | 26100 | 79741 | 55259 | 45324 | 39811 |
| (10,10) fd | 12020 | 22178 | 22397 | 22073 | 21191 |
| (14,14) tape | 24720 | 53607 | 47055 | 34423 | 36799 |
| (14,14) fd | 15072 | 28407 | 26116 | 26238 | 25890 |
| (16,16) tape | 33343 | 70166 | 58603 | 34811 | 27325 |
| (16,16) fd | 15744 | 31131 | 33788 | 27858 | 26772 |
| (18,6) tape | 73127 | 114447 | 93750 | 61036 | 51317 |
| (18,6) fd | 17072 | 30213 | 32973 | 27852 | 28830 |

tape leads all 25 cells, from 1.02x at the DRAM-bound (16,16) 1 MB corner to 4.3x
at (18,6) 100 B, typical 1.7x to 3.6x. (18,6) reaches 114 GB/s payload at 1 KB
because eighteen data bytes ride every six-output column. Milan-class hosts (AVX2,
no GFNI) do not run FFT; they take the fused AVX2 path (History → *AMD Zen 3*).

**NEON (M4 Max).** FFT plus the sha3 three-way-xor fold, 10 KB, MiB/s: (7,13)
**10433**, (10,10) **13947**, (14,14) **9735**, staged (16,16) **5588**, 1.5x to
2.1x over the pre-FFT fused kernel, ahead of both rse (C NEON) and sia everywhere.
(Per-run tables: History → *sha3 tier*, *staged executor*.)

**wasm (Node 24).** FFT-routed simd128, 12x to 18x over scalar across the per-plane
range: (7,13) **4866**, (10,10) **7087**, (16,16) **6343** MiB/s. Fusion is a clear
win here, unlike the M4's wide-cache NEON. (History → *wasm consumes FFT*.)

**Reconstruct.** Deliberately stays on cached matrix plans (see *Levers*), not FFT.
Inverts once per erasure pattern (cached on the codec, or held by an explicit
`PreparedDecoder`) and rebuilds through the same fused kernels as encode. sia exposes
no comparable slice-reconstruct entry point, so the comparison is against rse.

### M4 Max, single-thread, MiB/s (payload = k * size)

Erased 1 data shard (repair):
| shape | size | rse(C) | tape | tapePrepared |
|---|---|---|---|---|
| (7,13)  | 10 KB | 19478 | 33964 (1.7x) | 32102 (1.6x) |
| (7,13)  | 1 MB  | 19858 | 35363 (1.8x) | 35391 (1.8x) |
| (10,10) | 10 KB | 19304 | 39287 (2.0x) | 38495 (2.0x) |
| (10,10) | 1 MB  | 20069 | 31902 (1.6x) | 32431 (1.6x) |

Erased the maximal m shards (worst case):
| shape | size | rse(C) | tape | tapePrepared |
|---|---|---|---|---|
| (7,13)  | 1 KB   | 1446 | 4314 (3.0x) | 4359 (3.0x) |
| (7,13)  | 10 KB  | 1529 | 4731 (3.1x) | 4716 (3.1x) |
| (7,13)  | 1 MB   | 1548 | 3813 (2.5x) | 4044 (2.6x) |
| (10,10) | 1 KB   | 1878 | 6385 (3.4x) | 6452 (3.4x) |
| (10,10) | 10 KB  | 1969 | 7118 (3.6x) | 7024 (3.6x) |
| (10,10) | 1 MB   | 2019 | 5439 (2.7x) | 5281 (2.6x) |

Worst-case reconstruct tracks the fused encode numbers, which is the point: the
decode rows go through the same kernels. The cached-plan path (`reconstruct`, what
production calls per layer) matches the explicit `PreparedDecoder`, so the plan cache is
doing its job; prepared decoding skips the per-call pattern lookup and the codec's
plan mutex. On x86 (Zen 5) worst-case reconstruct rides the same overlapped 32-byte
tail: (7,13) all-13-erased reaches 2352 MiB/s at 100 B and 13431 at 1 KB. wasm
worst-case does 3682 at (7,13) 1 KB and 5930 at (10,10) 10 KB.

## Methodology: single-thread, and why

because **the production system parallelises across stripes upstream**, so threading inside the crate would just fight it.

## Reproduce

- **Native / NEON, local:** `cd bench-harness && cargo run --release`.
- **wasm simd128, local:** build with
  `RUSTFLAGS="-C target-feature=+simd128" wasm-pack build wasm-bench --target nodejs --release`,
  then `node wasm-bench/run.mjs` under Node 24. The differential `verify()` (fused ==
  scalar across generated, staged, and fused shapes and tail lengths) and the
  reconstruct gate run first and gate the bench.
- **x86 GFNI / AVX-512, GCP:** `bench-harness/run.sh` runs the single-thread
  head-to-head (tape's dispatched + fused + FFT kernels vs sia vs rse +simd-accel),
  and `fd-bench/run.sh` adds the firedancer C-FFI comparison. Machine types by µarch:
  `c3-standard-4` (Intel Sapphire Rapids, GFNI+AVX-512), `c3d-standard-4` (Zen 4
  Genoa, GFNI), `c4d-standard-4` (Zen 5 Turin, GFNI), `t2d`/`c2d-standard-4` (Zen 3
  Milan, AVX2-only, exercises the fused AVX2 path). Runtime dispatch means running
  *on* a box exercises that box's kernels with no build flags. Use an ephemeral-node
  workflow (label `purpose=rs-gate-ephemeral`, always delete); the project's global
  12-vCPU quota fits ~one 4-vCPU node at a time. When other nodes hold the quota, the
  -2 size of the same family gives identical single-thread ratios (verified on c4d:
  85% of firedancer at (10,10) 10 KB on both -4 and -2).

## Levers & decisions

- **Generated-tier coverage: current lever.** Five shapes carry generated programs;
  the staged tier is the floor for any other power-of-two shape, and fused matrix
  covers the rest. On x86 this closed the one big gap firedancer held: its
  full-coverage codegen used to lead our runtime shapes 2-3x; the five-shape generated
  tier plus the overlapped tail now put tape ahead everywhere (flagship table above).
  A zmm staged runner, or generating still more shapes, is the remaining extension.
- **Decode does not consume FFT: deliberate.** Typical repair (one or two erasures) is
  already near the information floor on matrix plans: every missing shard must combine
  all k survivors, and a plan applies exactly that one dot row per missing shard
  through the fused kernels. The FFT decoding formulation for arbitrary survivor sets
  (the formal-derivative method firedancer uses for recover) costs about three
  transforms plus pointwise work regardless of erasure count, which loses to the plans
  at the production n of about 20 for every pattern up to the worst case. The one
  pattern FFT accelerates (all data present, parity missing) is re-encoding, which
  production already routes to encode. If far larger shapes ever matter, the staged stage vocabulary
  extends to the derivative method by adding a pointwise stage kind.
- **Large-shard cache-blocking: superseded.** The pre-FFT fused kernel fell off a
  cliff past cache (20 concurrent streams); the "length-blocked kernel for 100 KB+"
  lever is superseded by FFT, which leads at every measured size.
- **Milan / AVX2 (no GFNI): fused AVX2, not FFT.** The FFT executors need GFNI; hosts
  without it take the fused AVX2 tiles (History → *AMD Zen 3*, the only AVX2/Milan run).

## History (chronological log)

Superseded numeric tables are dropped in favour of a pointer to where the current
figures live; the reasoning and the corrections are kept.

### M4 Max fused NEON, pre-FFT (2026-07-18)

First measurement of the fused NEON kernel (cached lo/hi tables on the struct, all
outputs per 16-byte block, sha3 3-way XOR fold, overlapped tail) as the default
aarch64 `encode`: 3.5-5x over the C backend and 1.5-3x over sia at production sizes
(100 B to 10 KB). An earlier version of this section, measured before that landed,
showed tape at roughly half those numbers and concluded "no deficit anywhere"; both
were wrong. Past 100 KB at (7,13) the single-pass fused kernel walked 20 concurrent
streams and fell off a cliff once the working set left cache. Superseded by the FFT
encode numbers (Current state); the large-shard lever it motivated is closed.

### The threading confound (corrects an earlier version of this doc)

An earlier run left sia on its **default** features, i.e. rayon across all 14 cores,
and reported sia at 5.3x / 12.9x / 17.7x at 100 KB / 1 MB / 4 MiB, i.e. "5-8x faster
than tape". That was **single-thread tape vs 14-thread sia**. The entire large-shard gap
was multi-threading. sia's 32 KiB cache-blocking gives **no** single-thread advantage
on the M4 (blocked sia ≈ unblocked tape at 4 MiB), so tape is not memory-bound on this
hardware. (An earlier "(20,20) proves compute-bound" claim was also wrong: (20,20)
halves under both models and distinguishes neither.)

### x86 Intel c3 Sapphire Rapids, pre-FFT (2026-07-18)

First-ever execution of the x86 kernels: SSSE3/AVX2/AVX-512/GFNI plus the fused GFNI
path passed the differential-vs-scalar tests (all 256 coeffs) and the wire-compat
parity gate (10 passed / 0 failed). Findings, since acted on: the fused GFNI kernel is
a multiple, not a wash (3.3-5.8x over C, 2-4x over single-thread sia at ≥ 10 KB),
while plain per-coefficient `encode` was mediocre, so x86 `encode` routes to the
fused (now FFT) path. The only Intel run; superseded on throughput by the GFNI/FFT
numbers above.

### AMD Zen 3 / Zen 4 / Zen 5, pre-FFT (2026-07-18)

Ephemeral GCP nodes. Correctness on all three: `cargo test --release` 11 passed / 0
failed (incl. `fused_avx2_matches_scalar` on real Milan silicon, and
`fused_gfni_matches_scalar` on Zen 4/5) plus wire-compat gate PASS. The Zen 4/5 fused
GFNI numbers (3.3-6.3x over C) and the pre-FFT firedancer head-to-head (tape roughly
even at (7,13), ~15% behind at (10,10), 2-4x behind at 100 B on per-call alloc) are
all superseded by the FFT runs. **Zen 3 (Milan, AVX2, no GFNI) is not**: it is the
only AVX2 data and the path Milan-class hosts still take:

Shape (10,10), fused AVX2 MiB/s (`dot_prod_avx2`, where `encode_fused` routes with no GFNI):
| size | rse(C) | single-out avx2 | sia | **fusedAVX2** | vs rse | vs sia |
|---|---|---|---|---|---|---|
| 100 KB | 3268 | 3172 | 3341 | **5010** | 1.53x | 1.50x |
| 1 MB   | 2841 | 2807 | 2918 | **4715** | 1.66x | 1.62x |
| 4 MiB  | 1238 | 1877 | 1886 | **3959** | 3.20x | 2.10x |

Shape (7,13): 100 KB **3732** (1.43x rse), 1 MB **3532** (2.00x), 4 MiB **3168**
(3.04x). Output-fusion holds throughput ~flat past L2 (3.2-4.0 GB/s at 4 MiB) while
the per-coefficient paths go memory-bound and halve. Small shards (100 B) lost to sia
on the `parity_rows()` per-call alloc, since removed.

### Zen 5, decode + caching re-run (2026-07-18, later same day)

First execution of the reworked x86 kernels (cached GFNI tiles for every shape,
assert-guarded entry, scalar tails) on real Turin silicon; 16 unit tests plus the
extended wire gate (prepared-decoder and contiguous-rows legs) passed. The headline
was 100 B encode: construction-time caching plus the scalar tail lifted it from
109-173 MiB/s (0.16-0.25x rse) to 6.6-7x over rse, roughly a 70x jump, with 10 KB+
unchanged. First x86 reconstruct numbers landed here too, tracking fused encode; the
one honest gap was (7,13) all-13-erased under ~1 KB, whose decode shapes (7,7)/(6,7)
missed the specializations and hit the 128-byte-floor scalar tail, the motivation for
the overlapped tail below. Encode numbers superseded by FFT.

### FFT encode: the firedancer gap closed (2026-07-18, third run)

The load-bearing "why FFT" result. The schoolbook matrix encode measured at ~90% of
its own arithmetic ceiling, so the residual firedancer gap at (10,10) was algorithmic,
not implementation. Encoding at the production shapes now runs a compiled
Lin-Chung-Han FFT program (interpolate through 0..k-1, evaluate at the parity points,
the same code the matrix defines, so parity is byte-identical), needing 29 multiplies
at (7,13) vs schoolbook's 91 and 34 vs 100 at (10,10), from generated straight-line
kernels (NEON and VEX GFNI plus AVX2). Full suite green including the GFNI executor
differential and the wire gate. The first ymm tables put tape ahead of firedancer at
every size (1.2-2.6x) where prior runs had it at 0.85x-even; those tables are
superseded by the zmm tier and the five-shape final run above.

### Arbitrary shapes: the staged FFT executor (2026-07-18, fourth pass)

Production profiles are runtime data: shortening turns legal profiles into shapes like
(14,14) (from a profile 20,6,19) or (18,6) (from 20,14,19). The builder compiles ANY
shape at construction into a staged program: whole FFT/IFFT block stages over a stack
register file, short glue runs between stages, trailing parity cosets folded down so no
transform output is wasted. Byte-checked against the matrix encode across 21 shapes.
The routing rule (now in *Levers*): staged wins when the data count is a power of two
(interpolation collapses to a single inverse transform, few big stages, almost no
glue) and the multiply saving is decisive; non-power counts pay recursion glue that
serializes on store-to-load forwarding, measured slower than fused, so they stay fused.
M4, 10 KB, route as printed by the bench:

| shape | route | rse(C) | tape | sia |
|---|---|---|---|---|
| (7,13)  | fft-generated | 1520 | **9348**  | 3712 |
| (10,10) | fft-generated | 2020 | **12401** | 4846 |
| (14,14) | fused-matrix  | 1421 | **4800**  | 3396 |
| (16,16) | fft-staged    | 1244 | **5201**  | 3018 |
| (18,6)  | fused-matrix  | 3343 | **11615** | 7999 |

Every shape rides its best measured path. ((14,14)/(18,6) later gained generated
programs, and the sha3 fold lifted the fft rows; see below.)

### wasm consumes FFT too (2026-07-18, same day)

The wasm simd128 executor is a mechanical twin of the NEON one (swizzle is the
table-lookup equivalent), consuming the same generated programs and staged builder
with the same power-of-two routing rule; the differential gates routed encode against
scalar before timing. Node 24 on the M4, routed `encode`, MiB/s:

| shape | route | before | now |
|---|---|---|---|
| (7,13)  | fft-generated | 4493 (1 KB) | 4866 |
| (10,10) | fft-generated | 5672 (1 KB) | 6912 |
| (10,10) | fft-generated | 5810 (10 KB) | 7087 |
| (16,16) | fft-staged    | 3342 (10 KB, fused) | 4206 |

Gains 6-24% on generated shapes and 1.2x for staged (16,16), smaller than native
because V8's register allocation spills the program file harder than LLVM; the same
routing rule still picks the winner per shape. Decode on wasm runs the same
plan-cached fused path as native (worst-case all-m-erased 3682 MiB/s at (7,13) 1 KB,
5930 at (10,10) 10 KB), gated by an erase-up-to-m reconstruct check.

### Small-shard reconstruct overhead hunt (2026-07-18, fifth pass)

The (7,13) worst-case reconstruct at 100 B was overhead-bound (700 payload bytes vs
~0.4us per-call). Two A/B runs: replacing the four gather Vecs with stack arrays LOST
~20% on both native and wasm (the allocator fast path beats initializing wide pointer
arrays), so **the Vec gather stays**, recorded by a code comment. The real cost was the
256-slot pointer array each tiled kernel initialized per call, paid twice per
reconstruct; the kernels now index their input slices directly. (7,13) all-13-erased
100 B: native 1627 → 2304 MiB/s (2.8x rse), wasm 873 → 1006.

### The sha3 tier and the multiply-by-one harvest (2026-07-19)

An instruction audit showed the (7,13) strip loop at 6.4 sustained IPC on the 8-wide
M4: issue-bound, so only removing instructions moves it. Two removals. Every
accumulating multiply in the generated programs ends in a nibble combine feeding one
accumulate, so a sha3 engine tier folds both into one veor3q at every such site (29/29
at (7,13), 34/34 at (10,10)); executors compile a plain and a sha3 core and dispatch
on detection. And the builder emits multiply-by-one as a plain xor, dropping staged
(16,16) from 82 to 66 real multiplies. M4, 10 KB, MiB/s: (7,13) 9348 → **10433**,
(10,10) 12401 → **13947**, staged (16,16) 5201 → **5588** (matching the 11.4% and 14%
predicted from site counts). wasm has no three-way xor and GFNI's affine already
produces one value per multiply, so neither gets the fold; the x86 analogue is width
(the zmm tier below), not folding.

### Zen 5 with the zmm tier and the runtime shapes (2026-07-19)

The generated FFT executors gained a zmm core (64-byte strips, same programs,
dispatched when AVX-512 GFNI is present and the shard covers a strip), halving the
per-strip loads/stores/xors while multiplies per byte stay fixed; unlike the earlier
zmm schoolbook experiment it carries no tables, so that negative result does not
transfer. First execution on Turin, byte-clean across the 64-byte boundary. The zmm
strips added 38-85% over the ymm tier at cache-resident sizes ((10,10) 1 KB nearly
doubled), putting the production shapes 2.5-3.8x over firedancer (superseded by the
final table above). This run exposed the runtime-shape gap (firedancer's
full-coverage codegen led (14,14)/(16,16) 2-3x) and the ugliest number, sub-128-byte
fused tails collapsing to 87 MiB/s at (14,14) 100 B, both fixed next.

### The overlapped vector tail, validated (2026-07-19, second run)

The tiled dot products now finish with overlapped 32-byte vector windows instead of a
scalar tail, so only shards shorter than one window see scalar code (overlapped bytes
recompute to identical values because outputs are pure functions of inputs). Gated
byte-clean on Turin. The 100 B collapse was erased (45x at (14,14), 18x at (18,6)),
1 KB rows moved 7-9x, and (18,6) beat firedancer at every size except 100 B; small-
shard reconstruct rode the same tiles ((7,13) worst-case 100 B 296 → 2352, 1 KB
2701 → 13431). This left the generated-coverage gap as the last x86 lever, closed by
the five-shape final run (Current state).
