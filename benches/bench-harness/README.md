# bench-harness — tape vs sia vs reed-solomon-erasure (head-to-head)

Benchmark-only. **Not** part of the `tape-reed-solomon` crate: it path-deps the
crate and adds the two things the shipping crate must never depend on —
`reed-solomon-erasure` **with `simd-accel`** (the C SIMD backend Clay actually
ships) and `sia_reed_solomon` (a storage competitor, reference only). Its
`[workspace]` table keeps it standalone, so `cargo build`/`cargo test` in the
crate never touch it.

The baseline is the C-accelerated rse — so every speedup is the **real native
delta**, unlike the crate's wire-compat gate which (correctly) diffs against the
pure-Rust scalar rse.

## Run on this host (aarch64 → NEON)

```
cd bench-harness
cargo run --release
```

Answers the open question from `../BENCH-RESULTS.md`: at the full-row sizes
(100 KB–4 MiB) the plan's I5 restructure targets, does sia's large-shard kernel
actually pull ahead of tape's, or was that a scalar-baseline artifact?

## Run on x86 (GFNI + AVX-512 → the AVX benches)

Cannot run on Apple Silicon. On a GCP `c3-standard-4` (Sapphire Rapids) or
`c3d-standard-4` (Zen4):

```
# from your workstation, per TAPE-RS-PLAN.md (ephemeral node, always delete):
gcloud compute scp --recurse /Users/k/solana/tape-public/tape-reed-solomon rs-test-tmp:~ --zone=<z>
gcloud compute ssh rs-test-tmp --zone=<z> --command="cd tape-reed-solomon/bench-harness && bash run.sh"
gcloud compute instances delete rs-test-tmp --zone=<z> --quiet
```

`run.sh` installs the toolchain, builds with `-C target-cpu=native`, and runs.
On x86 the table gains a `tape-gfniFused` column (the `encode_fused` GFNI +
`vpternlogd` kernel); on aarch64 that column is hidden (the fused NEON path was
dropped as a wash — see `../BENCH-RESULTS.md`).
