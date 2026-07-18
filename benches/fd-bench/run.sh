#!/usr/bin/env bash
# Same-run firedancer head-to-head — run on an x86_64 GFNI + AVX-512 node (GCP c3).
# Clones the firedancer fork next to the crate, builds the C via FFI, benches, and
# you delete the node. (firedancer is x86-only C, which is why this is separate
# from bench-harness and cannot run on the aarch64 dev host.)
set -euo pipefail
cd "$(dirname "$0")"

command -v cc >/dev/null || { echo "installing build tools"; sudo apt-get update -y && sudo apt-get install -y build-essential clang git; }
command -v cargo >/dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; source "$HOME/.cargo/env"; }
source "$HOME/.cargo/env" 2>/dev/null || true

# firedancer-reed-solomon fork next to the crate (../ = tape-reed-solomon). Shallow;
# only the firedancer submodule (not its nested submodules) — reedsol needs just
# src/ballet + util. Same recipe as reference/prototypes/rs-compat-check/run.sh.
if [ ! -d ../firedancer-reed-solomon ]; then
  git clone --depth 1 https://github.com/crypt0miester/firedancer-reed-solomon ../firedancer-reed-solomon
  ( cd ../firedancer-reed-solomon && git submodule update --init --depth 1 firedancer )
fi

echo "arch: $(uname -m)  cpu flags: $(grep -m1 -o -E 'gfni|avx512f|avx512bw|avx2' /proc/cpuinfo | sort -u | tr '\n' ' ')"
RUSTFLAGS="-C target-cpu=native" cargo run --release
