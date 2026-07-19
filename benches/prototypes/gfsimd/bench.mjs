// Verify the wasm-simd128 GF multiply is correct, then race it against the scalar
// table kernel over the exact production (20,10,13) RS working set (k=10 -> m=10, node
// rows of 32768B; and the per-plane 32B structure the production system currently uses).
import { createRequire } from 'node:module';
import { randomFillSync } from 'node:crypto';
const require = createRequire(import.meta.url);
const g = require('./pkg/gfsimd.js');

const k = 10, m = 10, row = 32768, sub = 32;
const payload = k * row; // 320 KiB
const WARM = 40, ITERS = 400;

console.log('selftest (simd mul == gf_mul over all 256x256):', g.selftest() === 0 ? 'PASS ✓' : `FAIL (${g.selftest()} mismatches) ✗`);

const data = new Uint8Array(payload); randomFillSync(data);
const matrix = new Uint8Array(m * k); randomFillSync(matrix);
for (let i = 0; i < matrix.length; i++) if (matrix[i] === 0) matrix[i] = 1; // nonzero coeffs

const rS = g.encode_fullrow_scalar(data, k, m, row, matrix);
const rF = g.encode_fullrow_simd(data, k, m, row, matrix);
const rP = g.encode_plane_simd(data, k, m, row, sub, matrix);
const eq = (a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)) === 0;
console.log('simd fullrow == scalar :', eq(rS, rF) ? 'MATCH ✓' : 'DIFFER ✗');
console.log('simd plane   == scalar :', eq(rS, rP) ? 'MATCH ✓' : 'DIFFER ✗');

function bench(fn) {
  for (let i = 0; i < WARM; i++) fn();
  const s = process.hrtime.bigint();
  for (let i = 0; i < ITERS; i++) fn();
  const secs = Number(process.hrtime.bigint() - s) / 1e9;
  return { ms: secs / ITERS * 1000, mibps: (payload * ITERS) / 1048576 / secs };
}
const p = (name, r, base) => console.log(`  ${name.padEnd(26)} ${r.ms.toFixed(3).padStart(8)} ms   ${r.mibps.toFixed(0).padStart(5)} MiB/s   ${base ? (r.mibps / base).toFixed(2) + 'x vs scalar' : ''}`);

console.log(`\n=== RS encode k=${k}->m=${m}, payload=${payload / 1024}KiB, ${ITERS} iters (M4 Max, wasm) ===`);
const base = bench(() => g.encode_fullrow_scalar(data, k, m, row, matrix));
p('scalar (per-byte table)', base, null);
p('simd128 full-row (32KB)', bench(() => g.encode_fullrow_simd(data, k, m, row, matrix)), base.mibps);
p('simd128 per-plane (32B)', bench(() => g.encode_plane_simd(data, k, m, row, sub, matrix)), base.mibps);

console.log(`\ncontext (payload MiB/s, from earlier runs):`);
console.log(`  native production encode: scalar RS 193  ->  C-SIMD RS 371 (1.92x)\n`);
