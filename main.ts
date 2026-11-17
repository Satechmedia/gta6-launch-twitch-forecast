// Launch-day Twitch viewer forecaster for the make.fun bounty gta6-twitch-ccu-forecast-01 (Node 22, standard library only).
//
// For each case it forecasts, for hours 2..24 of the release day, five quantiles of the category's concurrent viewers:
//
//   1. Every comparable launch is turned into a curve of log ratios ln(v_h / v_1) to its own hour 1, aligned on the
//      local clock (US daylight-saving shifts the midnight unlocks by one UTC hour between summer and winter launches).
//   2. Each comparable gets a similarity weight: platform (console-only or not), rolling midnight unlock, early-access
//      contamination, the size of hour 1, the momentum going into hour 1 and how high the day before ran.
//   3. For each hour the weighted log ratios form a Gaussian mixture (one bump per comparable, plus a small persistence
//      bump that takes over when no comparable is similar); the forecast quantiles are that mixture's quantiles,
//      widened by a calibration factor fitted by leave-one-out backtests, and anchored on hour 1's print.
//   4. In the final the minute reads inside hour 1 say where the hour ended relative to its mean; that surprise moves
//      the near hours. The top streams' concentration, when present, widens the spread slightly.
//   5. Every row is validated (finite, > 0, <= 1e9, non-decreasing); anything wrong falls back to the persistence
//      baseline, so a valid file is always written.
//
// Deterministic: no randomness, no clock, no network. Run locally with
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// ---------------------------------------------------------------------------------------------------------------------
// Constants and types
// ---------------------------------------------------------------------------------------------------------------------

const HOUR_MS = 3_600_000;
/** The quantile levels the contest scores, in row order. */
export const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels (the house baseline's constants). */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const BASELINE_SIGMA = 0.15;
const MAX_VIEWERS = 1e9;

/** Every tunable number of the model, in one place; tools/backtest.ts overrides them to compare variants. */
export interface ModelParams {
  /** Kernel width on ln(hour-1 viewers): launches of a similar size count more. */
  scaleWidth: number;
  /** The target's size is capped, for the size kernel only, at this quantile of the comparables' sizes: a target
   *  bigger than every past launch is matched to the big ones as a group, not to the single biggest. */
  scaleCapQuantile: number;
  /** Kernel width on the momentum into hour 1, ln(v_1 / v_0). */
  momentumWidth: number;
  /** Kernel width on ln(max of the previous 24 h / v_1). */
  preMaxWidth: number;
  /** Weight multiplier of a comparable whose platform class (console-only vs with PC) differs from the target's. */
  platformMismatch: number;
  /** Weight multiplier of a comparable whose unlock style (rolling midnight vs not) differs from the target's. */
  rollingMismatch: number;
  /** Weight multiplier of an early-access-contaminated or weak comparable. */
  contaminated: number;
  /** Align comparables on the local clock (1) or on UTC (0). */
  dstAlign: number;
  /** Share of a weighted regression slope on ln(hour-1 viewers) applied to each curve (0 = none). */
  scaleSlopeShare: number;
  /** Largest size difference (ln units) the slope correction is applied over. */
  scaleSlopeClamp: number;
  /** Mixture bandwidth as a share of the weighted standard deviation of the hour's log ratios. */
  bandwidthShare: number;
  /** Smallest bandwidth: bandwidthFloor * sqrt(h - 1), capped at bandwidthFloorCap. */
  bandwidthFloor: number;
  bandwidthFloorCap: number;
  /** Weight of the persistence bump (ln ratio 0, baseline spread) relative to one perfectly similar comparable. */
  persistenceWeight: number;
  /** Factor applied to every quantile's distance from the median (calibration). */
  spreadInflation: number;
  /** Share of the end-of-hour-1 surprise carried into hour 2, and its decay per hour, and its long-run share. */
  surpriseShare: number;
  surpriseDecayHours: number;
  surpriseFloor: number;
  /** Spread widening per ln unit by which the target's hour 1 exceeds the size cap (beyond the data's support), and
   *  the most ln units it is applied over. */
  extrapolationSpread: number;
  extrapolationClamp: number;
  /** Extra log spread per unit of squared top-stream share (one streamer holding the category). */
  concentrationSpread: number;
}

export const DEFAULT_PARAMS: ModelParams = {
  scaleWidth: 1.6,
  scaleCapQuantile: 0.85,
  momentumWidth: 0.3,
  preMaxWidth: 0.9,
  platformMismatch: 0.3,
  rollingMismatch: 0.7,
  contaminated: 0.5,
  dstAlign: 1,
  scaleSlopeShare: 0.5,
  scaleSlopeClamp: 1.5,
  bandwidthShare: 0.38,
  bandwidthFloor: 0.1,
  bandwidthFloorCap: 0.22,
  persistenceWeight: 0.25,
  spreadInflation: 1.0,
  surpriseShare: 0.75,
  surpriseDecayHours: 4,
  surpriseFloor: 0.3,
  extrapolationSpread: 0.1,
  extrapolationClamp: 2,
  concentrationSpread: 0.6,
};

/** What kind of launch a series is: used to match comparables to the target. */
export interface LaunchTraits {
  consoleOnly: boolean;
  rolling: boolean;
}

/** An hourly viewer series around a release day; offset 0 is the day's first hour (hour 1), -1 the hour before. */
export interface Series {
  name: string;
  dayStartMs: number;
  viewers: Map<number, number>;
}

export interface Comparable extends Series {
  traits: LaunchTraits;
  /** 1 for a clean comparable, lower for early-access-contaminated or weak ones. */
  quality: number;
}

/** The target of one forecast: its hourly history up to and including hour 1, and optional extras of the final. */
export interface Target extends Series {
  traits: LaunchTraits;
  hour1: number;
  /** ln(level at the end of hour 1 / hour 1's mean) from minute reads, or null when the history is hourly. */
  endOfHourLog: number | null;
  /** Share of the category's viewers held by its largest stream at the checkpoint, or null. */
  topStreamShare: number | null;
}

export interface ForecastRow { hour: number; q: number[] }

// ---------------------------------------------------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------------------------------------------------

const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const gaussKernel = (d: number, width: number): number => Math.exp(-0.5 * (d / width) ** 2);
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Standard normal CDF (Abramowitz-Stegun 7.1.26 error function, |error| < 1.5e-7). */
function normalCdf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erf = 1 - poly * Math.exp(-(x * x) / 2);
  return x >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

interface Bump { mean: number; sd: number; weight: number }

/** The quantile at level p of a weighted mixture of normal bumps, by bisection. */
function mixtureQuantile(bumps: Bump[], p: number): number {
  const total = bumps.reduce((s, b) => s + b.weight, 0);
  let lo = Infinity; let hi = -Infinity;
  for (const b of bumps) { lo = Math.min(lo, b.mean - 8 * b.sd); hi = Math.max(hi, b.mean + 8 * b.sd); }
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    let cdf = 0;
    for (const b of bumps) cdf += b.weight * normalCdf((mid - b.mean) / b.sd);
    if (cdf / total < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ---------------------------------------------------------------------------------------------------------------------
// Calendar: where US midnight falls in UTC (daylight saving), to align launches on the local clock
// ---------------------------------------------------------------------------------------------------------------------

/** The day of month of the n-th Sunday of a UTC month. */
function nthSunday(year: number, month: number, n: number): number {
  const firstWeekday = new Date(Date.UTC(year, month, 1)).getUTCDay();
  return 1 + ((7 - firstWeekday) % 7) + 7 * (n - 1);
}

/** The UTC hour of US-east midnight on the given day: 4 under daylight time (2nd Sunday of March to 1st of November), else 5. */
export function usEastMidnightUtcHour(dayStartMs: number): number {
  const d = new Date(dayStartMs);
  const year = d.getUTCFullYear();
  const dstStart = Date.UTC(year, 2, nthSunday(year, 2, 2));
  const dstEnd = Date.UTC(year, 10, nthSunday(year, 10, 1));
  return dayStartMs >= dstStart && dayStartMs < dstEnd ? 4 : 5;
}

// ---------------------------------------------------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------------------------------------------------

/** Builds an offset -> viewers map from rows with a start time; several rows in one hour are averaged. */
function hourlyMap(rows: unknown, timeKey: string, dayStartMs: number): Map<number, number> {
  const sums = new Map<number, { sum: number; n: number }>();
  if (Array.isArray(rows)) {
    for (const r of rows) {
      if (r === null || typeof r !== 'object') continue;
      const at = Date.parse(String((r as Record<string, unknown>)[timeKey]));
      const v = (r as Record<string, unknown>).viewers;
      if (!Number.isFinite(at) || !isNum(v) || v < 0) continue;
      const offset = Math.floor((at - dayStartMs) / HOUR_MS);
      const cell = sums.get(offset) ?? { sum: 0, n: 0 };
      cell.sum += v; cell.n += 1;
      sums.set(offset, cell);
    }
  }
  const out = new Map<number, number>();
  for (const [offset, cell] of sums) out.set(offset, cell.sum / cell.n);
  return out;
}

/** Console-only when the platform list, parentheticals aside, names neither PC nor a streaming service. */
function isConsoleOnly(platforms: string): boolean {
  return !/\bPC\b|Stadia|Steam/i.test(platforms.replace(/\([^)]*\)/g, ''));
}

/** A rolling unlock: midnight local time, country by country. */
function isRollingUnlock(unlock: string): boolean {
  return /midnight local|rolling|12 AM local/i.test(unlock);
}

function parseComparable(raw: unknown): Comparable | null {
  if (raw === null || typeof raw !== 'object') return null;
  const k = raw as Record<string, unknown>;
  const dayStartMs = Date.parse(String(k.day_start));
  if (!Number.isFinite(dayStartMs)) return null;
  const unlock = String(k.unlock ?? '');
  const tier = /Tier:\s*([a-z-]+)/i.exec(unlock)?.[1]?.toLowerCase() ?? 'comparable';
  if (tier === 'target-history' || /not a launch/i.test(unlock)) return null;
  const viewers = hourlyMap(k.hourly, 'from', dayStartMs);
  const quality = tier === 'comparable' ? 1 : 0;
  return {
    name: String(k.name ?? ''), dayStartMs, viewers, quality,
    traits: { consoleOnly: isConsoleOnly(String(k.platforms ?? '')), rolling: isRollingUnlock(unlock) },
  };
}

/** ln(level at the end of hour 1 / hour 1's print), from the minute reads of hour 1 when there are enough of them. */
function endOfHourSurprise(history: unknown, dayStartMs: number, hour1: number): number | null {
  if (!Array.isArray(history)) return null;
  const reads: Array<{ at: number; v: number }> = [];
  for (const r of history) {
    if (r === null || typeof r !== 'object') continue;
    const at = Date.parse(String((r as Record<string, unknown>).at));
    const v = (r as Record<string, unknown>).viewers;
    if (Number.isFinite(at) && isNum(v) && v > 0 && at >= dayStartMs && at < dayStartMs + HOUR_MS) reads.push({ at, v });
  }
  if (reads.length < 20) return null;
  reads.sort((a, b) => a.at - b.at);
  // Least-squares line through ln(viewers) over the hour's second half, read off at the hour's end: robust to one odd read.
  const tail = reads.filter((r) => r.at >= dayStartMs + HOUR_MS / 2);
  if (tail.length < 8) return null;
  const xs = tail.map((r) => (r.at - dayStartMs) / HOUR_MS);
  const ys = tail.map((r) => Math.log(r.v));
  const mx = xs.reduce((s, x) => s + x, 0) / xs.length;
  const my = ys.reduce((s, y) => s + y, 0) / ys.length;
  let sxy = 0; let sxx = 0;
  for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxx > 0 ? sxy / sxx : 0;
  const endLevel = my + clamp(slope, -1.5, 1.5) * (1 - mx);
  return clamp(endLevel - Math.log(hour1), -0.6, 0.6);
}

/** The largest stream's share of the category at the checkpoint, when the case lists its top streams. */
function topStreamShare(streams: unknown, hour1: number): number | null {
  if (!Array.isArray(streams) || streams.length === 0) return null;
  let top = 0;
  for (const s of streams) {
    const v = s !== null && typeof s === 'object' ? (s as Record<string, unknown>).viewers : null;
    if (isNum(v) && v > top) top = v;
  }
  return top > 0 ? clamp(top / hour1, 0, 1) : null;
}

interface ParsedCase { id: string; hours: number[]; target: Target; comparables: Comparable[] }

function parseCase(raw: Record<string, unknown>, fallbackId: string): ParsedCase {
  const known = Array.isArray(raw.known) ? (raw.known as Array<Record<string, unknown>>) : [];
  const first = known.find((k) => k && k.hour === 1);
  if (!first || !isNum(first.viewers)) throw new Error('no hour-1 print');
  const hour1 = Math.max(1, first.viewers);
  const dayStartMs = Date.parse(String(raw.day_start));
  if (!Number.isFinite(dayStartMs)) throw new Error('no day_start');
  let viewers = hourlyMap(raw.history_hourly, 'from', dayStartMs);
  if (viewers.size === 0) viewers = hourlyMap(raw.history, 'at', dayStartMs);
  viewers.set(0, hour1);
  for (const offset of [...viewers.keys()]) if (offset > 0) viewers.delete(offset);
  const forecast = Array.isArray(raw.forecast) ? (raw.forecast as Array<Record<string, unknown>>) : [];
  let hours = forecast.map((f) => f?.hour).filter((h): h is number => isNum(h) && Number.isInteger(h) && h >= 2 && h <= 24);
  if (hours.length !== 23 || new Set(hours).size !== 23) hours = Array.from({ length: 23 }, (_, i) => i + 2);
  const game = (raw.game ?? {}) as Record<string, unknown>;
  const target: Target = {
    name: String(game.name ?? ''), dayStartMs, viewers, hour1,
    // Every case of this contest is a console launch unlocking at midnight local time (the rules say so).
    traits: { consoleOnly: true, rolling: true },
    endOfHourLog: endOfHourSurprise(raw.history, dayStartMs, hour1),
    topStreamShare: topStreamShare(raw.streams, hour1),
  };
  const comparables = (Array.isArray(raw.comparables) ? raw.comparables : []).map(parseComparable).filter((c): c is Comparable => c !== null);
  return { id: String(raw.id ?? fallbackId), hours, target, comparables };
}

// ---------------------------------------------------------------------------------------------------------------------
// Features, similarity weights and log-ratio curves
// ---------------------------------------------------------------------------------------------------------------------

/** A series' viewers at an hour offset: the point itself, or a log-linear fill between points at most 3 hours apart. */
function viewersAt(s: Series, offset: number): number | null {
  const exact = s.viewers.get(offset);
  if (exact !== undefined) return exact > 0 ? exact : null;
  for (let gap = 1; gap <= 2; gap++) {
    for (let left = 1; left <= gap; left++) {
      const a = s.viewers.get(offset - left); const b = s.viewers.get(offset + (gap + 1 - left));
      if (a !== undefined && b !== undefined && a > 0 && b > 0) {
        const share = left / (gap + 1);
        return Math.exp(Math.log(a) * (1 - share) + Math.log(b) * share);
      }
    }
  }
  return null;
}

interface Features { logLevel: number; momentum: number | null; preMax: number | null }

/** What is known at the end of the anchor hour: its size, the move into it, and how high the 24 h before it ran. */
function featuresAt(s: Series, anchor: number): Features | null {
  const level = viewersAt(s, anchor);
  if (level === null) return null;
  const before = viewersAt(s, anchor - 1);
  let max = 0;
  for (let o = anchor - 24; o < anchor; o++) { const v = s.viewers.get(o); if (v !== undefined && v > max) max = v; }
  return {
    logLevel: Math.log(level),
    momentum: before === null ? null : clamp(Math.log(level / before), -1.5, 1.5),
    preMax: max > 0 ? clamp(Math.log(max / level), -2, 3) : null,
  };
}

function similarityWeight(target: Target, tf: Features, kernelLevel: number, c: Comparable, cf: Features, p: ModelParams): number {
  let w = c.quality >= 1 ? 1 : p.contaminated;
  if (c.traits.consoleOnly !== target.traits.consoleOnly) w *= p.platformMismatch;
  if (c.traits.rolling !== target.traits.rolling) w *= p.rollingMismatch;
  w *= gaussKernel(cf.logLevel - kernelLevel, p.scaleWidth);
  // A feature missing on either side is neither evidence for nor against: a neutral half weight.
  w *= tf.momentum !== null && cf.momentum !== null ? gaussKernel(cf.momentum - tf.momentum, p.momentumWidth) : 0.5;
  w *= tf.preMax !== null && cf.preMax !== null ? gaussKernel(cf.preMax - tf.preMax, p.preMaxWidth) : 0.5;
  return w;
}

interface WeightedCurve { name: string; weight: number; logLevel: number; logRatio: Array<number | null> }
/** The weighted curves and how far (ln units) the target's hour 1 lies above the sizes the comparables support. */
interface Evidence { curves: WeightedCurve[]; excess: number; momentum: number | null }

/** Each usable comparable's weight and its curve ln(v_h / v_anchor) for hours 2..24 (index h), on the target's clock. */
function buildCurves(target: Target, comparables: Comparable[], p: ModelParams): Evidence {
  const tf = featuresAt(target, 0);
  if (tf === null) return { curves: [], excess: 0, momentum: null };
  const targetMidnight = usEastMidnightUtcHour(target.dayStartMs);
  // A comparable in daylight time has its local midnights one UTC hour earlier than a winter target: read it shifted.
  const shiftOf = (c: Comparable): number => (p.dstAlign ? targetMidnight - usEastMidnightUtcHour(c.dayStartMs) : 0);
  const levels = comparables.map((c) => featuresAt(c, -shiftOf(c))?.logLevel).filter(isNum).sort((a, b) => a - b);
  const cap = levels.length > 0 ? levels[Math.min(levels.length - 1, Math.floor(p.scaleCapQuantile * levels.length))] : tf.logLevel;
  const kernelLevel = Math.min(tf.logLevel, cap);
  const curves: WeightedCurve[] = [];
  for (const c of comparables) {
    const shift = shiftOf(c);
    const anchor = -shift;
    const cf = featuresAt(c, anchor);
    if (cf === null) continue;
    const weight = similarityWeight(target, tf, kernelLevel, c, cf, p);
    if (!(weight > 1e-6)) continue;
    const base = Math.exp(cf.logLevel);
    const logRatio: Array<number | null> = new Array(25).fill(null);
    for (let h = 2; h <= 24; h++) {
      const v = viewersAt(c, h - 1 - shift);
      logRatio[h] = v === null ? null : Math.log(v / base);
    }
    curves.push({ name: c.name, weight, logLevel: cf.logLevel, logRatio });
  }
  return { curves, excess: tf.logLevel - kernelLevel, momentum: tf.momentum };
}

/** Weighted least-squares slope of the hour's log ratios on ln(hour-1 size): bigger launches ramp less. */
function scaleSlope(points: Array<{ x: number; y: number; w: number }>): number {
  const sw = points.reduce((s, q) => s + q.w, 0);
  if (points.length < 6 || !(sw > 0)) return 0;
  const mx = points.reduce((s, q) => s + q.w * q.x, 0) / sw;
  const my = points.reduce((s, q) => s + q.w * q.y, 0) / sw;
  let sxy = 0; let sxx = 0;
  for (const q of points) { sxy += q.w * (q.x - mx) * (q.y - my); sxx += q.w * (q.x - mx) ** 2; }
  // A ridge term keeps the slope near 0 when the sizes barely differ.
  return clamp(sxy / (sxx + 0.5 * sw), -0.5, 0.5);
}

// ---------------------------------------------------------------------------------------------------------------------
// Quantile construction
// ---------------------------------------------------------------------------------------------------------------------

/** The persistence baseline's five quantiles for hour h. */
export function baselineQuantiles(hour1: number, h: number): number[] {
  return Z.map((z) => round2(hour1 * Math.exp(z * BASELINE_SIGMA * Math.sqrt(h - 1))));
}

/** How much of the end-of-hour-1 surprise hour h inherits. */
function surpriseCarry(h: number, p: ModelParams): number {
  return p.surpriseFloor + (p.surpriseShare - p.surpriseFloor) * Math.exp(-(h - 2) / p.surpriseDecayHours);
}

/** Five log-ratio quantiles (relative to hour 1's print) for hour h. */
function hourLogQuantiles(target: Target, evidence: Evidence, h: number, p: ModelParams): number[] {
  const { curves } = evidence;
  const targetLogLevel = Math.log(target.hour1);
  const points = curves.filter((c) => c.logRatio[h] !== null).map((c) => ({ x: c.logLevel, y: c.logRatio[h] as number, w: c.weight }));
  const slope = p.scaleSlopeShare > 0 ? p.scaleSlopeShare * scaleSlope(points) : 0;
  const adjusted = points.map((q) => ({ y: q.y + slope * clamp(targetLogLevel - q.x, -p.scaleSlopeClamp, p.scaleSlopeClamp), w: q.w }));

  const sw = adjusted.reduce((s, q) => s + q.w, 0);
  const persistenceSd = BASELINE_SIGMA * Math.sqrt(h - 1);
  const bumps: Bump[] = [{ mean: 0, sd: persistenceSd, weight: p.persistenceWeight }];
  if (sw > 0) {
    const mean = adjusted.reduce((s, q) => s + q.w * q.y, 0) / sw;
    const sd = Math.sqrt(adjusted.reduce((s, q) => s + q.w * (q.y - mean) ** 2, 0) / sw);
    const floor = Math.min(p.bandwidthFloorCap, p.bandwidthFloor * Math.sqrt(h - 1));
    const bandwidth = Math.max(floor, p.bandwidthShare * sd);
    for (const q of adjusted) bumps.push({ mean: q.y, sd: bandwidth, weight: q.w });
  }
  const raw = LEVELS.map((level) => mixtureQuantile(bumps, level));

  // Calibration: stretch around the median, more so for a target bigger than anything in the comparables; one
  // dominant streamer makes the category jumpier.
  const stretch = p.spreadInflation * (1 + p.extrapolationSpread * clamp(evidence.excess, 0, p.extrapolationClamp));
  const extra = target.topStreamShare !== null ? p.concentrationSpread * target.topStreamShare ** 2 : 0;
  const median = raw[2];
  // Where hour 1 ended relative to its mean, net of the half-hour of trend the hourly momentum already implies
  // (the comparables are matched on that momentum, so their curves carry it).
  const surprise = target.endOfHourLog !== null ? target.endOfHourLog - 0.5 * clamp(evidence.momentum ?? 0, -0.4, 0.4) : 0;
  const shift = surpriseCarry(h, p) * clamp(surprise, -0.5, 0.5);
  return raw.map((q, j) => median + shift + (q - median) * stretch + Z[j] * extra);
}

/** The forecast rows of one target from its comparables. Pure: the same inputs give the same rows. */
export function forecastTarget(target: Target, comparables: Comparable[], hours: number[], params: ModelParams = DEFAULT_PARAMS): ForecastRow[] {
  const evidence = buildCurves(target, comparables, params);
  return hours.map((hour) => {
    const logQ = hourLogQuantiles(target, evidence, hour, params);
    return { hour, q: logQ.map((x) => round2(clamp(target.hour1 * Math.exp(x), 0.01, MAX_VIEWERS))) };
  });
}

// ---------------------------------------------------------------------------------------------------------------------
// Validation and output
// ---------------------------------------------------------------------------------------------------------------------

/** True when a row is five finite numbers in (0, 1e9], non-decreasing. */
function isValidRow(row: ForecastRow): boolean {
  if (!Array.isArray(row.q) || row.q.length !== LEVELS.length) return false;
  for (let j = 0; j < row.q.length; j++) {
    const x = row.q[j];
    if (!isNum(x) || x <= 0 || x > MAX_VIEWERS) return false;
    if (j > 0 && x < row.q[j - 1]) return false;
  }
  return true;
}

/** Rows for exactly hours 2..24, each valid; a missing or invalid row is replaced by the persistence baseline's. */
function sanitiseRows(rows: ForecastRow[], hour1: number): ForecastRow[] {
  const byHour = new Map<number, ForecastRow>();
  for (const row of rows) if (Number.isInteger(row.hour) && row.hour >= 2 && row.hour <= 24 && isValidRow(row)) byHour.set(row.hour, row);
  const out: ForecastRow[] = [];
  for (let h = 2; h <= 24; h++) out.push(byHour.get(h) ?? { hour: h, q: baselineQuantiles(hour1, h) });
  return out;
}

/** Parses a case file's text and forecasts it; any failure degrades to the persistence baseline, never to a crash. */
export function forecastCaseText(text: string, fallbackId: string): { id: string; rows: ForecastRow[] } {
  let hour1 = 1;
  try {
    const raw = JSON.parse(text) as Record<string, unknown>;
    const known = Array.isArray(raw.known) ? (raw.known as Array<Record<string, unknown>>) : [];
    const first = known.find((k) => k && k.hour === 1);
    if (first && isNum(first.viewers)) hour1 = Math.max(1, first.viewers);
    const parsed = parseCase(raw, fallbackId);
    return { id: fallbackId, rows: sanitiseRows(forecastTarget(parsed.target, parsed.comparables, parsed.hours), parsed.target.hour1) };
  } catch {
    return { id: fallbackId, rows: sanitiseRows([], hour1) };
  }
}

/** Internal steps exposed for tools/score.ts (pseudo-cases for the leave-one-out backtest) and for inspection. */
export const internals = { parseComparable, parseCase, featuresAt, buildCurves, viewersAt };

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  mkdirSync(outputDir, { recursive: true });
  const inputs = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    let text = '';
    try { text = readFileSync(join(inputDir, file), 'utf8'); } catch { /* an unreadable file still gets a baseline row set */ }
    const { rows } = forecastCaseText(text, id);
    writeFileSync(join(outputDir, `${id}.out`), `${JSON.stringify({ hours: rows })}\n`);
  }
}

// tools/ import this file as a library with FORECAST_LIBRARY_ONLY=1; the grader runs it as a program.
if (process.env.FORECAST_LIBRARY_ONLY !== '1') main();
