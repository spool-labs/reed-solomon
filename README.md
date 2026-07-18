# tape-reed-solomon

Pure-Rust Reed-Solomon erasure coding over **GF(2^8)** (primitive polynomial
`0x11d`), with SIMD-accelerated field arithmetic.

## API

```rust
let rs = ReedSolomon::new(data_shards, parity_shards)?;
rs.encode(&mut shards)?;        // shards: data shards followed by parity shards
rs.reconstruct(&mut shards)?;   // fills in missing shards in place
```

`new`, `encode`, `verify`, `reconstruct`, `reconstruct_data`, and `encode_rows`
(a batched encode over whole contiguous rows in a single pass).

## Backends

Field math routes through `gf::mul_slice` / `gf::mul_slice_xor`. The scalar kernel
is the reference; every SIMD kernel builds its tables from `galois::mul`, so all
kernels are byte-identical to scalar by construction, each with a scalar tail for
sub-vector remainders.

| arch    | backend                                   | kernel              |
|---------|-------------------------------------------|---------------------|
| x86_64  | GFNI > AVX-512BW > AVX2 > SSSE3 > scalar   | `src/gf/x86.rs`     |
| aarch64 | NEON                                      | `src/gf/neon.rs`    |
| wasm32  | simd128 (under `+simd128`)                | `src/gf/wasm128.rs` |
| other   | scalar                                    | `src/gf/scalar.rs`  |

## Cargo features

The backend can be pinned at build time instead of detected at runtime — useful
for a homogeneous fleet, a single-kernel benchmark, or a smaller binary. Set **at
most one**; a pinned kernel the target CPU lacks faults (illegal instruction) at
runtime, so pin only what the target supports.

| feature  | backend                          |
|----------|----------------------------------|
| `scalar` | portable, no SIMD (any target)   |
| `ssse3`  | x86_64 SSSE3 (128-bit)           |
| `avx2`   | x86_64 AVX2 (256-bit)            |
| `avx512` | x86_64 AVX-512BW (512-bit)       |
| `gfni`   | x86_64 GFNI + AVX-512            |
| `neon`   | aarch64 NEON                     |

The default is **`gfni`** (the full x86 build). For runtime CPU dispatch with
graceful fallback, build with `--no-default-features`. For a non-GFNI x86 target,
build with `--no-default-features --features avx2` (or `ssse3`). Because the
default already pins one backend, any override must start from
`--no-default-features`.

On wasm32 the backend is chosen by the `+simd128` target-feature:

```sh
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

## Testing

```sh
cargo test    # unit tests, per-backend differential-vs-scalar, wire-compat gate
```

`tests/parity.rs` verifies byte-identical output against an independent reference
implementation across 8 shard shapes x 3 sizes, plus cross-implementation
reconstruct and round-trip. On x86 hosts `cargo test` exercises whichever of
GFNI/AVX-512/AVX2/SSSE3 the CPU supports (add `--no-default-features` on a
non-GFNI x86 host); on aarch64, NEON.

Benchmarks and cross-implementation harnesses live under `benches/`.

## License

Apache-2.0; see `LICENSE` and `THIRD-PARTY-NOTICES.md`.
