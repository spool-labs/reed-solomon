#!/usr/bin/env bash
# Run on an x86_64 Linux node (GCP: a GFNI-capable machine is ideal, e.g. c3/c3d
# = Sapphire Rapids, or n2 Ice Lake). Verifies firedancer<->reed-solomon-erasure
# byte-compat and benches native throughput, then you can delete the node.
set -euo pipefail
cd "$(dirname "$0")"

command -v cc >/dev/null || { echo "installing build tools"; sudo apt-get update -y && sudo apt-get install -y build-essential git clang; }
command -v cargo >/dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source "$HOME/.cargo/env"; }

# firedancer-reed-solomon next to this crate. Shallow, and only init the firedancer
# submodule (NOT its nested submodules) — the reedsol build needs just src/ballet+util.
if [ ! -d ../firedancer-reed-solomon ]; then
  git clone --depth 1 https://github.com/leafaar/firedancer-reed-solomon ../firedancer-reed-solomon
  ( cd ../firedancer-reed-solomon && git submodule update --init --depth 1 firedancer )
fi

# target-cpu=native lets firedancer's build.rs pick the fastest ARITH_IMPL
# (GFNI+AVX512 on Sapphire Rapids). The compat GATE result is independent of this;
# only the PERF numbers change.
echo "arch: $(uname -m)  cpu flags: $(grep -m1 -o -E 'gfni|avx512f|avx2' /proc/cpuinfo | sort -u | tr '\n' ' ')"
RUSTFLAGS="-C target-cpu=native" cargo run --release
