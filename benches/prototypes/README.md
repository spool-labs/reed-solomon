# Preserved prototypes

These are the **byte-verified origin prototypes** for `tape-reed-solomon`,
preserved here (source only — no `target/` or `pkg/` build directories) so the
feasibility proofs referenced in `PLAN.md` live durably alongside the crate.
They are tape's own clean-room assets, kept for provenance; they are not built as
part of the crate.

## `gfsimd/`

The original clean-room **wasm128 GF(2^8) slice-multiply** prototype. The crate's
`src/gf/wasm128.rs` kernel derives from this prototype's nibble-split swizzle
(the technique is public — Plank et al. / Intel ISA-L). Measured 3.2x per-plane /
5.45x full-row over scalar in wasm. `bench.mjs` is its Node benchmark harness.

## `rs-compat-check/`

The x86 wire-compat **gate + bench harness** (`run.sh` installs Rust +
build-essential, builds, runs). This is the harness pointed at an ephemeral c3
Sapphire Rapids GCP node to runtime-verify the GFNI/AVX-512 path — see `PLAN.md`
and the repo `README.md` for the ephemeral-node workflow. Not run from here.
