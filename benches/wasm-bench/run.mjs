// wasm simd128 encode bench + differential (Node 24). Build first:
//   RUSTFLAGS="-C target-feature=+simd128" wasm-pack build wasm-bench --target nodejs --release
// then: node wasm-bench/run.mjs
import { setup, run_fused, run_scalar, run_reconstruct, verify, verify_reconstruct } from './pkg/wasm_bench.js';

// Differential first: fused `encode` must equal `encode_scalar` byte-for-byte.
const bad = verify();
console.log(`differential (fused encode == scalar, specialised + fallback shapes, tail lengths): ${bad === 0 ? 'PASS' : `FAIL (${bad} mismatched bytes)`}`);
if (bad !== 0) process.exit(1);
const badR = verify_reconstruct();
console.log(`reconstruct gate (erase up to m, rebuild == original, five shapes): ${badR === 0 ? 'PASS' : `FAIL (${badR} mismatched bytes)`}`);
if (badR !== 0) process.exit(1);

const SHAPES = [[7, 13], [10, 10], [16, 16], [14, 14], [18, 6]];
// the production per-plane range (~100 B – 1.4 KB at the 1 MB stripe cap), plus a large one.
const SIZES = [100, 256, 512, 1024, 1430, 10000];

function measure(fn, k, m, sz) {
  setup(k, m, sz);
  const payload = k * sz;
  const iters = Math.max(200, Math.min(5000, Math.floor(80_000_000 / payload)));
  fn(Math.min(iters, 100)); // warmup
  const t0 = performance.now();
  fn(iters);
  const t1 = performance.now();
  return (payload * iters) / (1024 * 1024) / ((t1 - t0) / 1000);
}

const pad = (s, n) => String(s).padStart(n);

for (const [k, m] of SHAPES) {
  console.log(`\n--- (${k},${m}) encode, wasm simd128, MiB/s (Node 24) ---`);
  console.log(`${pad('size', 6)} ${pad('scalar', 8)} ${pad('fused', 8)}  ${pad('fused/scalar', 12)}`);
  for (const sz of SIZES) {
    const sc = measure(run_scalar, k, m, sz);
    const fu = measure(run_fused, k, m, sz);
    console.log(`${pad(sz, 6)} ${pad(sc.toFixed(0), 8)} ${pad(fu.toFixed(0), 8)}  ${pad((fu / sc).toFixed(1) + 'x', 12)}`);
  }
}
console.log('\nfused = encode (cached tables + multi-output + overlapped tail), the default wasm path.');

for (const [k, m] of SHAPES) {
  console.log(`\n--- (${k},${m}) reconstruct, worst case ${m} erased, MiB/s (Node 24) ---`);
  console.log(`${pad('size', 6)} ${pad('reconstruct', 12)}`);
  for (const sz of SIZES) {
    const v = measure((iters) => run_reconstruct(iters, m), k, m, sz);
    console.log(`${pad(sz, 6)} ${pad(v.toFixed(0), 12)}`);
  }
}
