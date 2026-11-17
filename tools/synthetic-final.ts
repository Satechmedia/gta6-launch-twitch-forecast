// Writes a synthetic case shaped like the final's h00, to exercise the code paths the practice cases never reach:
// minute reads in `history`, `streams`, `categories_hourly`, a winter release day and a very large hour 1.
//   node tools/synthetic-final.ts <a practice case.in> <output dir> [hour-1 viewers] [end-of-hour drift, ln units]
// The viewer numbers are invented by a fixed formula (no randomness): this tests plumbing, it is not a forecast input.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [source, outDir, hour1Arg, driftArg] = process.argv.slice(2);
if (!source || !outDir) throw new Error('usage: node tools/synthetic-final.ts <case.in> <out dir> [hour1] [drift]');
const hour1 = Number(hour1Arg ?? 1_500_000);
const drift = Number(driftArg ?? 0.1);
const practice = JSON.parse(readFileSync(source, 'utf8'));

const MIN = 60_000; const HOUR = 60 * MIN;
const dayStart = Date.parse('2026-11-19T00:00:00Z');
const recorderStart = Date.parse('2026-11-17T06:00:00Z');
const nzUnlock = Date.parse('2026-11-18T11:00:00Z');
const iso = (ms: number): string => new Date(ms).toISOString().replace('.000Z', 'Z');

/** An invented pre-launch level: a quiet hype plateau, then a climb from the New Zealand unlock to hour 1. */
const level = (ms: number): number => {
  const wobble = 1 + 0.03 * Math.sin(ms / (7 * MIN));
  if (ms < nzUnlock) return 60_000 * (1 + 0.4 * Math.sin((ms - recorderStart) / (24 * HOUR) * 2 * Math.PI)) * wobble;
  const progress = (ms - nzUnlock) / (dayStart - nzUnlock);
  const base = Math.exp(Math.log(90_000) + (Math.log(hour1) - Math.log(90_000)) * Math.min(1, progress) ** 0.6);
  // Inside hour 1 the level drifts linearly in log by `drift` from start to end, with the hour's mean at hour1.
  const inHour = ms >= dayStart ? Math.exp(drift * ((ms - dayStart) / HOUR - 0.5)) : 1;
  return base * inHour * wobble;
};

const history: Array<{ at: string; viewers: number; channels: number }> = [];
for (let t = recorderStart; t < dayStart + HOUR; t += MIN) {
  if ((t / MIN) % 97 === 0) continue; // a few missed reads
  history.push({ at: iso(t), viewers: Math.round(level(t)), channels: Math.round(level(t) / 90) });
}
const hourly: Array<{ from: string; viewers: number; channels: number; reads: number }> = [];
for (let t = recorderStart; t < dayStart + HOUR; t += HOUR) {
  const reads = history.filter((r) => Date.parse(r.at) >= t && Date.parse(r.at) < t + HOUR);
  hourly.push({ from: iso(t), viewers: Math.round(reads.reduce((s, r) => s + r.viewers, 0) / reads.length), channels: Math.round(reads.reduce((s, r) => s + r.channels, 0) / reads.length), reads: reads.length });
}
const first = hourly[hourly.length - 1];

const synthetic = {
  schema: practice.schema, id: 'h00',
  game: { name: 'Grand Theft Auto VI', twitch_id: '1867279146' },
  day_start: iso(dayStart), checkpoint: iso(dayStart + HOUR), levels: practice.levels,
  known: [{ hour: 1, from: iso(dayStart), until: iso(dayStart + HOUR), viewers: first.viewers, channels: first.channels, reads: first.reads }],
  forecast: Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, from: iso(dayStart + (i + 1) * HOUR), until: iso(dayStart + (i + 2) * HOUR) })),
  history, history_hourly: hourly,
  streams: Array.from({ length: 20 }, (_, i) => ({ login: `stream${i}`, viewers: Math.round(first.viewers * 0.12 / (i + 1)), language: i % 3 === 0 ? 'es' : 'en', started_at: iso(dayStart - (i + 1) * HOUR) })),
  categories_hourly: hourly.slice(-24).map((h) => ({ from: h.from, top: [{ name: 'Grand Theft Auto VI', viewers: h.viewers, channels: h.channels }, { name: 'Just Chatting', viewers: 400_000, channels: 5000 }] })),
  comparables: practice.comparables,
  notes: 'synthetic test case',
};
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'h00.in'), JSON.stringify(synthetic));
console.log(`wrote ${join(outDir, 'h00.in')}: hour 1 = ${first.viewers} from ${first.reads} minute reads, ${history.length} history points`);
