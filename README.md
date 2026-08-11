# tape-reed-solomon

[![Crates.io](https://img.shields.io/crates/v/tape-reed-solomon.svg)](https://crates.io/crates/tape-reed-solomon)
[![Documentation](https://docs.rs/tape-reed-solomon/badge.svg)](https://docs.rs/tape-reed-solomon)
[![License](https://img.shields.io/crates/l/tape-reed-solomon.svg)](LICENSE)

Fast Reed-Solomon erasure coding in pure Rust.

Given a set of data shards, `tape-reed-solomon` generates parity shards that
can recover data lost in storage or transit. With `k` data shards and `m`
parity shards, any `m` missing shards can be reconstructed.

The crate is designed for storage, archival, and distributed systems that need
high-throughput encoding and recovery. It uses SIMD where available, selects the
best implementation at runtime, and falls back to portable Rust everywhere
else.

## Installation

```toml
[dependencies]
tape-reed-solomon = "0.2"
```

## Example

```rust
use tape_reed_solomon::ReedSolomon;

// Four data shards and two parity shards.
let rs = ReedSolomon::new(4, 2)?;
let mut shards = vec![
    vec![1u8; 1024],
    vec![2u8; 1024],
    vec![3u8; 1024],
    vec![4u8; 1024],
    vec![0u8; 1024],
    vec![0u8; 1024],
];

rs.encode(&mut shards)?;

// Lose one shard, then reconstruct it from the survivors.
let expected = shards[1].clone();
let mut damaged: Vec<Option<Vec<u8>>> = shards.into_iter().map(Some).collect();
damaged[1] = None;

rs.reconstruct(&mut damaged)?;
assert_eq!(damaged[1].as_deref(), Some(expected.as_slice()));
# Ok::<(), tape_reed_solomon::Error>(())
```

The code is systematic: encoding leaves the original data shards unchanged and
writes parity into the remaining shards.

## Which coder should I use?

Use `ReedSolomon` for almost every workload. It supports up to 256 total shards
and accepts shards of any byte length.

`ReedSolomon16` supports up to 65,536 total shards for unusually wide storage or
distribution schemes. It uses a wider field and requires even-length shards.

Both provide encoding, verification, reconstruction, and reusable decoding for
repeated erasure patterns. See the [API documentation](https://docs.rs/tape-reed-solomon)
for the complete interface.

## Performance and correctness

Hardware acceleration is selected automatically on supported native CPUs, with
portable and WebAssembly implementations available for other targets.

See [BENCHMARKS.md](BENCHMARKS.md) for measurements, methodology, and
comparisons with other Reed-Solomon implementations.

Output and reconstruction are tested against the independent
`reed-solomon-erasure` crate, and accelerated implementations are checked
against the portable implementation.

```sh
cargo test
```

## License

Apache-2.0. See [LICENSE](LICENSE) and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
