#!/usr/bin/env node
// One chart: tape's speedup over firedancer, one line per shape, across shard
// size, from the x86 Zen 5 Turin flagship numbers (BENCH-RESULTS.md).
//
// Dependency-free. Emits charts/speedup.svg (renders directly in a GitHub
// README via <img>) plus charts/index.html for local preview. Regenerate:
//   node charts/gen.mjs
//
// Colors validated with the dataviz skill on the dark navy surface #11151d:
// the four hues clear the normal-vision floor (19.3) and >= 3:1 contrast; CVD
// sits in the 6-8 band, covered by direct labels + the legend. Five series is
// one past the safe four, so the fifth line reuses a hue as a dashed line
// (color + style), never a fifth confusable hue.

import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = dirname(fileURLToPath(import.meta.url));

const C = {
  blue: '#3987e5',
  green: '#008300',
  magenta: '#d55181',
  yellow: '#c98500',
  panel: '#11151d',
  ink: '#eef2f8',
  sub: '#aab4c4',
  muted: '#6b7789',
  baseline: '#2a3242',
};
const FONT = 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace';

// x86 Zen 5 Turin, MiB/s single-thread
const SIZE_LABELS = ['100 B', '1 KB', '10 KB', '100 KB', '1 MB'];
const XV = [100, 1000, 10000, 100000, 1000000];
// Two dashed pairings keep 6 shapes on 4 CVD-safe hues: blue = (7,13) solid /
// (10,10) dashed; yellow = (18,6) solid / (32,32) dashed. (32,32) is Agave's
// shred FEC shape and firedancer's own hand-tuned shape, so it is the lowest
// line, yet tape still leads at every size. One c4d Zen 5 Turin run.
const SERIES = [
  { name: '(7,13)',  color: C.blue,    dash: false, tape: [29419, 46959, 44169, 32131, 12790], fd: [8113, 15010, 17194, 15728, 7210] },
  { name: '(10,10)', color: C.blue,    dash: true,  tape: [32775, 76241, 56210, 43946, 42848], fd: [11484, 21146, 22088, 22315, 21587] },
  { name: '(14,14)', color: C.green,   dash: false, tape: [32987, 53120, 46959, 35577, 34225], fd: [12788, 28521, 27480, 26287, 26535] },
  { name: '(16,16)', color: C.magenta, dash: false, tape: [39853, 57585, 56039, 36047, 28537], fd: [13504, 31667, 34451, 29496, 26989] },
  { name: '(18,6)',  color: C.yellow,  dash: false, tape: [70816, 115281, 94649, 65136, 51166], fd: [17173, 30458, 33807, 28612, 28437] },
  { name: '(32,32)', color: C.yellow,  dash: true,  tape: [22963, 31585, 32404, 23767, 14665], fd: [12303, 23713, 21179, 20152, 13123] },
];
for (const s of SERIES) s.spd = s.tape.map((t, i) => t / s.fd[i]);

const log10 = (x) => Math.log(x) / Math.LN10;
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
function txt(x, y, s, o = {}) {
  const { fill = C.muted, size = 12.5, weight = 400, anchor = 'start' } = o;
  return `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="${fill}" font-family='${FONT}' font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
}
function niceAxis(maxVal) {
  const raw = maxVal / 5;
  const mag = Math.pow(10, Math.floor(log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const max = Math.ceil(maxVal / step) * step;
  const ticks = [];
  for (let t = 0; t <= max + step * 1e-6; t += step) ticks.push(t);
  return { max, ticks };
}

function chart() {
  const W = 920, H = 420;
  const PL = 60, PR = 92, PT = 52, PB = 84;
  const x0 = PL, x1 = W - PR, y0 = H - PB, y1 = PT;
  const xAt = (v) => x0 + ((log10(v) - log10(100)) / (log10(1e6) - log10(100))) * (x1 - x0);
  const ax = niceAxis(Math.max(...SERIES.flatMap((s) => s.spd)));
  const yAt = (v) => y0 - (v / ax.max) * (y0 - y1);

  let s = '';
  s += `<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="14" fill="${C.panel}" stroke="#ffffff" stroke-opacity="0.08"/>`;
  s += txt(PL, 28, 'Encode throughput vs firedancer', { fill: C.ink, size: 17, weight: 600 });
  s += txt(x1, 28, 'speedup = tape ÷ firedancer · x86 Zen 5 · single-thread', { fill: C.muted, size: 12, anchor: 'end' });

  for (const t of ax.ticks) {
    const y = yAt(t);
    s += `<line x1="${x0}" y1="${y.toFixed(1)}" x2="${x1}" y2="${y.toFixed(1)}" stroke="#ffffff" stroke-opacity="${t === 0 ? 0 : 0.06}"/>`;
    s += txt(x0 - 12, y + 4, `${t}x`, { anchor: 'end', size: 12 });
  }
  // parity reference at 1.0x
  const py = yAt(1);
  s += `<line x1="${x0}" y1="${py.toFixed(1)}" x2="${x1}" y2="${py.toFixed(1)}" stroke="${C.muted}" stroke-width="1.25" stroke-dasharray="4 4"/>`;
  s += txt(x0 + 6, py - 8, 'parity (1.0x)', { size: 11, fill: C.muted });
  s += `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y0}" stroke="${C.baseline}"/>`;
  XV.forEach((v, i) => {
    const x = xAt(v);
    s += `<line x1="${x.toFixed(1)}" y1="${y0}" x2="${x.toFixed(1)}" y2="${y0 + 5}" stroke="${C.baseline}"/>`;
    s += txt(x, y0 + 22, SIZE_LABELS[i], { anchor: 'middle', size: 12 });
  });

  // Agave shred-size marker (987 B, essentially at the 1 KB tick)
  const ax987 = xAt(987);
  s += `<line x1="${ax987.toFixed(1)}" y1="${(y1 - 4).toFixed(1)}" x2="${ax987.toFixed(1)}" y2="${y0}" stroke="${C.sub}" stroke-width="1" stroke-dasharray="3 4" stroke-opacity="0.5"/>`;
  s += txt(ax987, y1 - 8, '987 B · Agave shred', { anchor: 'middle', size: 11, fill: C.sub });

  // series
  for (const ser of SERIES) {
    const pts = ser.spd.map((v, i) => [xAt(XV[i]), yAt(v)]);
    const dash = ser.dash ? ' stroke-dasharray="7 5"' : '';
    s += `<polyline fill="none" stroke="${ser.color}" stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round"${dash} points="${pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')}"/>`;
    for (const [px, py2] of pts) s += `<circle cx="${px.toFixed(1)}" cy="${py2.toFixed(1)}" r="3.6" fill="${ser.color}" stroke="${C.panel}" stroke-width="1.5"/>`;
  }

  // right-end direct labels, de-collided
  const ends = SERIES.map((ser) => ({ y: yAt(ser.spd[4]), name: ser.name, color: ser.color }))
    .sort((a, b) => a.y - b.y);
  const MIN = 15;
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < MIN) ends[i].y = ends[i - 1].y + MIN;
  for (const e of ends) s += txt(x1 + 10, e.y + 4, e.name, { fill: e.color, size: 12.5, weight: 600 });

  // peak callout on the top series at its max
  const top = SERIES.reduce((a, b) => (Math.max(...b.spd) > Math.max(...a.spd) ? b : a));
  const pk = top.spd.indexOf(Math.max(...top.spd));
  s += txt(xAt(XV[pk]), yAt(top.spd[pk]) - 11, `${top.spd[pk].toFixed(1)}x`, { fill: C.ink, size: 12.5, weight: 600, anchor: pk === 0 ? 'start' : 'middle' });

  // legend (solid vs dashed shown)
  const ly = H - 30;
  let lx = PL;
  for (const ser of SERIES) {
    const dash = ser.dash ? ' stroke-dasharray="6 4"' : '';
    s += `<line x1="${lx}" y1="${ly}" x2="${lx + 24}" y2="${ly}" stroke="${ser.color}" stroke-width="2.5" stroke-linecap="round"${dash}/>`;
    s += txt(lx + 32, ly + 4, ser.name, { fill: C.sub, size: 12.5 });
    lx += 32 + ser.name.length * 8.5 + 26;
  }
  s += txt(PL, ly + 26, 'dashed shares a hue: blue (7,13)/(10,10), yellow (18,6)/(32,32). (32,32) is Agave shred', { fill: C.muted, size: 11 });

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img">${s}</svg>`;
}

const svg = chart();
writeFileSync(join(OUT, 'speedup.svg'), svg);
writeFileSync(join(OUT, 'index.html'), `<!doctype html><meta charset="utf-8"><title>bench chart</title><style>body{margin:0;background:#0a0d13;padding:28px;display:flex;justify-content:center}svg{max-width:100%;height:auto}</style>${svg}`);
console.log('wrote charts/speedup.svg + index.html');
