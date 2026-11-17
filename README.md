# gta6-launch-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Satechmedia/gta6-launch-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Satechmedia/gta6-launch-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour: given everything recorded up to the end of the release day's first hour, it writes five quantiles (5%, 25%, 50%, 75%, 95%) of the category's concurrent viewers for each of hours 2 to 24. The forecast is a similarity-weighted ensemble of past big launches' hour-by-hour curves, anchored on hour 1's count, in a single dependency-free Node 22 program.

## How it works

- **Curves, not levels.** Every comparable launch becomes a curve of log ratios `ln(viewers at hour h / viewers at hour 1)`. Curves are aligned on the local clock: US daylight saving moves the midnight unlocks by one UTC hour between summer and winter launches.
- **Similarity weights.** A comparable counts more when it looks like the target at the end of hour 1: console-only or not, rolling midnight unlock or a single global one, early-access contamination, the size of hour 1, the momentum going into hour 1, and how high the previous 24 hours ran. A target bigger than every past launch is matched to the big launches as a group, and its spread is widened for the extrapolation.
- **Quantiles from a mixture.** For each hour the weighted log ratios form a Gaussian mixture (one bump per comparable, shifted by a shrunk regression on launch size, plus a small persistence bump that takes over when nothing is similar). The five quantiles are read off that mixture, so the spread and its skew come from what past launches actually did.
- **Final-only signals, used defensively.** Minute reads inside hour 1 show where the hour ended relative to its mean; that surprise, net of the hourly trend, moves the near hours and fades. A category held up by one dominant stream gets a slightly wider spread. Both are skipped when the data is absent.
- **Always a valid file.** Every row is checked (finite, positive, at most 1e9, non-decreasing); anything wrong, down to an unreadable case, falls back to the persistence baseline. No randomness, no clock, no network.

`tools/analyse.ts` is the exploratory analysis of the comparables, `tools/score.ts` runs the forecaster on the practice cases and a leave-one-out backtest over the past launches with the contest's exact pinball loss, and `tools/synthetic-final.ts` builds a final-shaped test case. Leave-one-out, launches with at least 20,000 viewers in hour 1: loss 0.254 against the persistence baseline's 0.321, 90% coverage 0.92; console-only launches: 0.174 against 0.283.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=y3d8gmga&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
