#!/usr/bin/env bash
# AVX/GFNI head-to-head — run on an x86_64 Linux node with GFNI + AVX-512.
# GCP: c3-standard-4 (Intel Sapphire Rapids) or c3d-standard-4 (AMD Zen4); both
# have GFNI. Measures tape (GFNI/AVX-512 kernels + the fused GFNI path) vs sia vs
# reed-solomon-erasure(+simd-accel, C AVX2), then you DELETE the node.
#
# Copy the WHOLE tape-reed-solomon crate to the node first (this harness path-deps
# `..`), e.g.:
#   gcloud compute scp --recurse tape-reed-solomon rs-test-tmp:~  --zone=<z>
#   gcloud compute ssh rs-test-tmp --zone=<z> \
#     --command="cd tape-reed-solomon/bench-harness && bash run.sh"
# Provision/delete per TAPE-RS-PLAN.md (labels=purpose=rs-gate-ephemeral; always delete).
set -euo pipefail
cd "$(dirname "$0")"

command -v cc >/dev/null || { echo "installing build tools"; sudo apt-get update -y && sudo apt-get install -y build-essential clang; }
command -v cargo >/dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source "$HOME/.cargo/env"; }

echo "arch: $(uname -m)  cpu flags: $(grep -m1 -o -E 'gfni|avx512f|avx512bw|avx2|ssse3' /proc/cpuinfo | sort -u | tr '\n' ' ')"

# 1. CORRECTNESS — the x86 SIMD kernels (GFNI / AVX-512 / AVX2 / SSSE3) EXECUTE here
#    for the first time. cargo test runs the per-kernel differential-vs-scalar tests
#    (all 256 coeffs, incl. the fused GFNI path) + the wire-compat parity gate on
#    real x86. This is the step the aarch64 dev host could only typecheck.
echo "=== [1/2] crate correctness on real x86 (GFNI/AVX-512 differential + parity gate) ==="
( cd "$(dirname "$0")/../.." && RUSTFLAGS="-C target-cpu=native" cargo test --release )

# 2. PERFORMANCE — the single-thread head-to-head. target-cpu=native lets rse's
#    simd-accel C and tape's runtime dispatch pick the best kernels (GFNI + AVX-512
#    on Sapphire Rapids). Correctness above is unaffected by the flag.
echo "=== [2/2] head-to-head bench (tape GFNI/AVX-512 + fused vs sia vs rse+simd-accel) ==="
RUSTFLAGS="-C target-cpu=native" cargo run --release
