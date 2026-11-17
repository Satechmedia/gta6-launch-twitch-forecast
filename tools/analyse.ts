// Exploratory analysis of the comparables in a case file: how does each release-day hour relate to hour 1?
//   node tools/analyse.ts <path/to/case.in>
// Prints, per comparable: hour 1's level, pre-launch features, and log ratios ln(v_h / v_1) at selected hours;
// then the cross-launch distribution of the log ratio per hour, overall and split by group.

import { readFileSync } from 'node:fs';

const HOUR_MS = 3600_000;
const path = process.argv[2];
if (!path) throw new Error('usage: node tools/analyse.ts <case.in>');
const c = JSON.parse(readFileSync(path, 'utf8'));

interface Row { name: string; tier: string; rolling: boolean; v1: number; pre: (number | null)[]; day: (number | null)[]; ch1: number | null; chDay: (number | null)[] }

const tierOf = (unlock: string): string => (/Tier: ([a-z-]+)/.exec(unlock)?.[1] ?? 'unknown');
const isRolling = (unlock: string): boolean => /midnight local|rolling|12 AM local/i.test(unlock);

const rows: Row[] = [];
for (const k of c.comparables) {
  const t0 = Date.parse(k.day_start);
  const v = new Map<number, number>();
  const ch = new Map<number, number>();
  for (const p of k.hourly) {
    const o = Math.round((Date.parse(p.from) - t0) / HOUR_MS);
    if (typeof p.viewers === 'number') v.set(o, p.viewers);
    if (typeof p.channels === 'number') ch.set(o, p.channels);
  }
  const v1 = v.get(0);
  const tier = tierOf(k.unlock);
  if (v1 === undefined || tier === 'target-history') { console.log(`skip ${k.name}: hour 1 ${v1}, tier ${tier}`); continue; }
  const pre = Array.from({ length: 48 }, (_, i) => v.get(-48 + i) ?? null);
  const day = Array.from({ length: 24 }, (_, i) => v.get(i) ?? null);
  const chDay = Array.from({ length: 24 }, (_, i) => ch.get(i) ?? null);
  rows.push({ name: k.name, tier, rolling: isRolling(k.unlock), v1, pre, day, ch1: ch.get(0) ?? null, chDay });
}

const lr = (a: number | null, b: number): string => (a === null || a <= 0 ? '   . ' : (Math.log(a / b) >= 0 ? '+' : '') + Math.log(a / b).toFixed(2));
const pad = (s: string, n: number): string => s.slice(0, n).padEnd(n);

console.log('\nPer launch: ln(v_h / v_1). pre24 = mean of the 24 h before the day; preMax = max of those; last3 = hours -3..-1; d24 = same hour the day before');
console.log(pad('name', 30), 'tier roll      v1  pre24 preMax  last3 slope |   h2    h4    h6    h9   h12   h15   h18   h21   h24 | peakH');
const SHOW = [2, 4, 6, 9, 12, 15, 18, 21, 24];
for (const r of rows.sort((a, b) => b.v1 - a.v1)) {
  const p24 = r.pre.slice(24).filter((x): x is number => x !== null);
  const meanPre = p24.length ? p24.reduce((s, x) => s + x, 0) / p24.length : null;
  const maxPre = p24.length ? Math.max(...p24) : null;
  const l3 = r.pre.slice(45).filter((x): x is number => x !== null);
  const last3 = l3.length ? l3.reduce((s, x) => s + x, 0) / l3.length : null;
  const prev = r.pre[47];
  let peakH = 1; let peak = r.v1;
  r.day.forEach((x, i) => { if (x !== null && x > peak) { peak = x; peakH = i + 1; } });
  console.log(pad(r.name, 30), pad(r.tier, 4), r.rolling ? 'Y' : 'n', String(r.v1).padStart(8), lr(meanPre, r.v1), lr(maxPre, r.v1), lr(last3, r.v1), lr(prev, r.v1), '|',
    SHOW.map((h) => lr(r.day[h - 1], r.v1)).join(' '), '|', peakH);
}

const quant = (xs: number[], p: number): number => {
  const s = [...xs].sort((a, b) => a - b);
  if (!s.length) return NaN;
  const i = p * (s.length - 1); const lo = Math.floor(i); const hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
};
const dist = (label: string, sel: (r: Row) => boolean): void => {
  const g = rows.filter(sel);
  console.log(`\n${label} (n=${g.length}): hour, n, q05 q25 q50 q75 q95 of ln(v_h/v_1), sd`);
  for (let h = 2; h <= 24; h++) {
    const xs = g.map((r) => r.day[h - 1]).map((x, i) => (x !== null && x > 0 ? Math.log(x / g[i].v1) : null)).filter((x): x is number => x !== null);
    const m = xs.reduce((s, x) => s + x, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
    console.log(String(h).padStart(3), String(xs.length).padStart(3), [0.05, 0.25, 0.5, 0.75, 0.95].map((p) => quant(xs, p).toFixed(2).padStart(6)).join(' '), sd.toFixed(2).padStart(6));
  }
};
dist('All launches', () => true);
dist('Rolling midnight-local', (r) => r.rolling);
dist('Not rolling / unverified', (r) => !r.rolling);
dist('Hour 1 already big (v1 >= 20k)', (r) => r.v1 >= 20000);
dist('Hour 1 big and falling into the day (last3 > v1)', (r) => {
  const l3 = r.pre.slice(45).filter((x): x is number => x !== null);
  return r.v1 >= 20000 && l3.length > 0 && l3.reduce((s, x) => s + x, 0) / l3.length > r.v1;
});

// How much does the pre-day explain? Correlate ln(v_h/v_1) with ln(v_{h-24}/v_{-1}) style features.
const corr = (xs: number[], ys: number[]): number => {
  const n = xs.length; const mx = xs.reduce((s, x) => s + x, 0) / n; const my = ys.reduce((s, x) => s + x, 0) / n;
  let sxy = 0; let sxx = 0; let syy = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
};
console.log('\nCorrelation across launches of ln(v_h/v_1) with: A = ln(v_{h-24}/v_{1-24}) (yesterday\'s same move), B = ln(preMax24/v_1), C = ln(v_{-1}/v_1) (momentum into hour 1)');
for (let h = 2; h <= 24; h += 2) {
  const A: number[][] = [[], []]; const B: number[][] = [[], []]; const C: number[][] = [[], []];
  for (const r of rows) {
    const y = r.day[h - 1]; if (y === null || y <= 0) continue;
    const ly = Math.log(y / r.v1);
    const a1 = r.pre[24 + h - 1]; const a0 = r.pre[24];
    if (a1 !== null && a0 !== null && a1 > 0 && a0 > 0) { A[0].push(Math.log(a1 / a0)); A[1].push(ly); }
    const p24 = r.pre.slice(24).filter((x): x is number => x !== null);
    if (p24.length) { B[0].push(Math.log(Math.max(...p24) / r.v1)); B[1].push(ly); }
    const m = r.pre[47]; if (m !== null && m > 0) { C[0].push(Math.log(m / r.v1)); C[1].push(ly); }
  }
  console.log(String(h).padStart(3), `A n=${A[0].length} r=${corr(A[0], A[1]).toFixed(2)}`, `B n=${B[0].length} r=${corr(B[0], B[1]).toFixed(2)}`, `C n=${C[0].length} r=${corr(C[0], C[1]).toFixed(2)}`);
}

// The target category's own recent history, as the case gives it.
console.log(`\nCase ${c.id}: ${c.game.name}, hour 1 = ${c.known[0].viewers}; last 30 history_hourly rows:`);
for (const p of c.history_hourly.slice(-30)) console.log(' ', p.from, p.viewers, p.channels);
