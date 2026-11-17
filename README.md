[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fauzinfau4/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fauzinfau4/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecaster

Entry for **gta6-twitch-ccu-forecast-01** ("Forecast GTA VI's launch-day Twitch viewers hour by hour", make.fun). A Python 3.13 standard-library-only program that reads one case file (`/input/<id>.in`) and forecasts Twitch concurrent viewers in the "Grand Theft Auto VI" category for hours 2–24 of release day 2026-11-19 (UTC), as five quantiles per hour (5%, 25%, 50%, 75%, 95%).

## Method

For each comparable past launch, take the log-ratio of each release-day hour to its own hour-1 print, `L_h = ln(v_h / v_1)`. Weight each comparable by a Gaussian kernel in log hour-1 size (bandwidth 0.7, widened adaptively when the effective sample is small): tiny hour-1 launches can grow 100–2000x by evening while huge ones grow only ~1–3x, and mixing the regimes would blow up the upper tail. Per forecast hour, pool the weighted log-ratios of hours h−1..h+1, take weighted quantiles, and blend 50/50 in log space with a normal-IQR approximation (sigma from weighted IQR/1.349, floored 0.15, capped 1.5) to steady the noisy 5%/95% tails. The GTA VI "comparable" is excluded (its unlock field says "Not a launch"), as is any curve missing a release-day hour. Fallback when no comparable is usable: flat median at the hour-1 print with a widening spread.

## Validation

- Practice cases p00–p03: all 4 outputs valid (23 rows, hours 2–24, ordered quantiles, files ~1.5 KB).
- Runtime ~0.05 s per case (limit 8 s); two runs byte-identical.
- Leave-one-out backtest over the 37 usable comparables (real code path): **loss 0.4288 vs house baseline 0.9296** (−53.9%); on big launches (hour-1 print ≥ 10k, the regime of GTA VI): **0.2637 vs baseline 0.3508** (−25%), typical miss 58.0% vs 81.3%, 90% coverage 0.894 vs 0.766. Beats the baseline on 33/37 launches. For reference the house baseline's stated practice loss is ~0.387.

## Files

- `main.py` — the forecaster (entrypoint)
- `backtest.py` — leave-one-out backtest harness (dev tool)
- `LICENSE` — MIT

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=7tgjyx9m&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
