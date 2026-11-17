// Local scorer and leave-one-out backtest.
//   node tools/score.ts <kit input dir> [name=value ...]
//
// 1. Runs main.ts's forecast on every practice case and checks each output's shape. The practice answers are not in
//    the kit, so no practice loss can be computed here.
// 2. Leave-one-out backtest on the comparables: each past launch becomes a pseudo-case (history up to the end of its
//    release day's hour 1, answer = hours 2..24), forecast from the other launches only, scored with the rules' exact
//    pinball loss on the log scale, next to the persistence baseline's loss on the same pseudo-cases.
// name=value arguments override model parameters (e.g. spreadInflation=1.2) to compare variants.

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

process.env.FORECAST_LIBRARY_ONLY = '1';
const model = await import('../main.ts');
const { LEVELS, DEFAULT_PARAMS, forecastTarget, forecastCaseText, baselineQuantiles, internals } = model;

const inputDir = process.argv[2];
if (!inputDir) throw new Error('usage: node tools/score.ts <input dir> [param=value ...]');
const params = { ...DEFAULT_PARAMS } as Record<string, number>;
let verbose = false;
for (const arg of process.argv.slice(3)) {
  if (arg === 'verbose') { verbose = true; continue; }
  const [k, v] = arg.split('=');
  if (!(k in params) || !Number.isFinite(Number(v))) throw new Error(`unknown parameter ${arg}`);
  params[k] = Number(v);
}

/** The rules' pinball loss of one item: 2 x max(tau x d, (tau - 1) x d), d = ln(y) - ln(q). */
const pinball = (y: number, q: number, tau: number): number => {
  const d = Math.log(Math.max(1, y)) - Math.log(q);
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

interface Tally { loss: number; items: number; absMedian: number; covered: number; hours: number }
const newTally = (): Tally => ({ loss: 0, items: 0, absMedian: 0, covered: 0, hours: 0 });
const add = (t: Tally, y: number, q: number[]): void => {
  LEVELS.forEach((tau: number, j: number) => { t.loss += pinball(y, q[j], tau); t.items += 1; });
  t.absMedian += Math.abs(Math.log(Math.max(1, y)) - Math.log(q[2]));
  t.covered += q[0] <= y && y <= q[4] ? 1 : 0;
  t.hours += 1;
};
const show = (t: Tally): string => `loss ${(t.loss / t.items).toFixed(4)}  typical miss ${(100 * (Math.exp(t.absMedian / t.hours) - 1)).toFixed(1).padStart(6)}%  cover90 ${(t.covered / t.hours).toFixed(3)}  (${t.hours} hours)`;

// The worked example of the rules, as a self-test of the loss.
{
  const t = newTally();
  add(t, 1_000_000, [600000, 800000, 900000, 1100000, 1500000]);
  add(t, 700_000, [600000, 800000, 900000, 1100000, 1500000]);
  const got = (t.loss / t.items).toFixed(6);
  if (got !== '0.112545') throw new Error(`scorer self-test failed: ${got}`);
}

// --- 1. practice cases: run and validate -------------------------------------------------------------------------------
const files = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
const rawCases = files.map((f) => JSON.parse(readFileSync(join(inputDir, f), 'utf8')));
console.log('Practice cases (answers are not in the kit: shape check and a look at the forecast only)');
for (let i = 0; i < files.length; i++) {
  const { rows } = forecastCaseText(JSON.stringify(rawCases[i]), files[i].slice(0, -3));
  const p1 = rawCases[i].known[0].viewers;
  const ok = rows.length === 23 && rows.every((r: { hour: number; q: number[] }, k: number) => r.hour === k + 2 && r.q.length === 5 && r.q.every((x, j) => Number.isFinite(x) && x > 0 && x <= 1e9 && (j === 0 || x >= r.q[j - 1])));
  const pick = [2, 4, 6, 9, 12, 15, 18, 21, 24].map((h) => { const q = rows[h - 2].q; return `h${h}:${(q[2] / p1).toFixed(2)}[${(q[0] / p1).toFixed(2)}-${(q[4] / p1).toFixed(2)}]`; });
  console.log(` ${files[i]} ${String(rawCases[i].game.name).padEnd(24)} hour1 ${String(p1).padStart(7)} valid=${ok}  median/hour1 [q05-q95]: ${pick.join(' ')}`);
}

// --- 2. leave-one-out on the comparables --------------------------------------------------------------------------------
// The pool is the union of the comparables over the case files (they are the same 38 in each practice case).
const pool = new Map<string, ReturnType<typeof internals.parseComparable>>();
for (const c of rawCases) for (const k of c.comparables) { const parsed = internals.parseComparable(k); if (parsed) pool.set(parsed.name, parsed); }
const launches = [...pool.values()].filter((c): c is NonNullable<typeof c> => c !== null);

const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
const groups: Record<string, { model: Tally; base: Tally }> = {};
const tallyFor = (g: string) => (groups[g] ??= { model: newTally(), base: newTally() });
const perHour = HOURS.map(() => ({ model: newTally(), base: newTally() }));

console.log(`\nLeave-one-out over ${launches.length} past launches (pseudo-case = history to the end of hour 1; answer = hours 2..24)`);
for (const held of launches) {
  const hour1 = held.viewers.get(0);
  if (hour1 === undefined || hour1 < 1) continue;
  const history = new Map([...held.viewers].filter(([offset]) => offset <= 0));
  const target = { name: held.name, dayStartMs: held.dayStartMs, viewers: history, hour1, traits: held.traits, endOfHourLog: null, topStreamShare: null };
  const others = launches.filter((c) => c !== held);
  const rows = forecastTarget(target, others, HOURS, params);
  const mine = newTally(); const base = newTally();
  const inGroups = ['all'];
  if (hour1 >= 20000) inGroups.push('hour1>=20k');
  if (hour1 >= 20000 && held.quality >= 1) inGroups.push('hour1>=20k clean');
  if (hour1 >= 20000 && held.traits.consoleOnly) inGroups.push('hour1>=20k console-only');
  if (hour1 >= 20000 && held.traits.rolling) inGroups.push('hour1>=20k rolling');
  if (hour1 >= 100000) inGroups.push('hour1>=100k');
  for (const row of rows) {
    const y = held.viewers.get(row.hour - 1);
    if (y === undefined) continue;
    const b = baselineQuantiles(hour1, row.hour);
    add(mine, y, row.q); add(base, y, b);
    if (hour1 >= 20000) { add(perHour[row.hour - 2].model, y, row.q); add(perHour[row.hour - 2].base, y, b); }
    for (const g of inGroups) { add(tallyFor(g).model, y, row.q); add(tallyFor(g).base, y, b); }
  }
  if (verbose && mine.items) console.log(` ${held.name.slice(0, 32).padEnd(32)} ${held.traits.consoleOnly ? 'console' : 'pc     '} ${held.traits.rolling ? 'roll' : '    '} hour1 ${String(Math.round(hour1)).padStart(7)}  model ${(mine.loss / mine.items).toFixed(3)} cover ${(mine.covered / mine.hours).toFixed(2)}  baseline ${(base.loss / base.items).toFixed(3)}`);
}
for (const [g, t] of Object.entries(groups)) {
  console.log(`\n[${g}]\n  model    ${show(t.model)}\n  baseline ${show(t.base)}`);
}
if (verbose) {
  console.log('\nPer hour, launches with hour1 >= 20k: model loss / cover, baseline loss / cover');
  HOURS.forEach((h, i) => console.log(`  h${String(h).padStart(2)}  ${(perHour[i].model.loss / perHour[i].model.items).toFixed(3)} ${(perHour[i].model.covered / perHour[i].model.hours).toFixed(2)}   ${(perHour[i].base.loss / perHour[i].base.items).toFixed(3)} ${(perHour[i].base.covered / perHour[i].base.hours).toFixed(2)}`));
}
