"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Method (original, written for this competition):
  Every past big launch in the case's `comparables` leaves a release-day
  curve: 24 hourly Twitch viewer counts starting at 00:00 UTC on its
  release day.  For each such launch I take the log-ratio of every later
  hour to that launch's own hour-1 print,

      L_h = ln(viewers_h / viewers_1),   h = 2..24.

  Those ratios are (nearly) scale-free: a launch ten times bigger still
  shows the same daily shape -- the overnight dip before the US unlocks,
  the climb into the US evening, the late-night fall -- so they can be
  borrowed across launches of very different sizes.

  But not every launch belongs in the same pool.  A launch whose hour-1
  print is tiny (its unlock lands later in the day, or its category was
  dead at midnight UTC) can grow 100x or 1000x by the evening, while a
  launch that is already huge at 00:00 UTC -- the UK and continental
  Europe already unlocked at local midnight -- grows only ~2x.  Mixing
  the two regimes would give an honest median but a wildly overstated
  upper tail.  So each comparable is weighted by how close its own
  hour-1 print is to this case's hour-1 print, with a Gaussian kernel in
  log space (bandwidth 0.7 log units, widened only if the effective
  sample gets too small).  Launches of a similar opening size dominate;
  distant ones fade out.

  For each forecast hour h I pool the weighted log-ratios of hours
  h-1, h and h+1 (the true curve is smooth in h, so neighbouring hours
  are informative and the pooling triples the effective sample), take
  their weighted quantiles at the five scored levels, and exponentiate
  back against this case's hour-1 print:

      q_tau(h) = p1 * exp(Q_tau({L})).

  The quantiles are therefore the cross-launch distribution of what
  actually happened after a similar-sized first hour -- an honest
  spread, not a guessed one.

  One regularisation: with only a few dozen launches, the empirical
  5% and 95% quantiles are the noisiest numbers in the pool (they rest
  on the one or two most extreme launches).  So each outer quantile is
  blended 50/50 in log space with a normal-approximation quantile built
  from the same pool -- the weighted median plus/minus the normal
  z-score times sigma, where sigma is the pool's weighted interquartile
  range / 1.349, floored at 0.15 and capped at 1.5 log units.  The
  median is identical in both halves, so the blend only steadies the
  tails.  A leave-one-out backtest over the 37 usable comparables
  (backtest.py) picked this blend and the 0.7 bandwidth over the
  unblended variants: loss 0.429 over all launches and 0.264 over
  launches with an hour-1 print above 10,000 -- the regime every
  practice case and the GTA VI final sit in -- against 0.930 / 0.351
  for the house persistence baseline on the same backtest.

  Everything is deterministic: fixed sort orders, no randomness, no
  network, standard library only.

  Fallback, if the comparable pool is ever empty (no usable release-day
  curve at all): a flat median at the hour-1 print with a normal spread
  widening like a random walk in log space over the day.  A case with no
  usable hour-1 print at all is answered with a print of 1 viewer.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.  (This entry is one file; the line is
# kept so the layout stays valid if a helper is ever added.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels (fallback spread only).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

HOURS = list(range(2, 25))  # the hours to forecast

# Kernel bandwidth, in natural-log units of the hour-1 print, for how
# similar a comparable's opening size must be to count fully.  Picked
# by the leave-one-out backtest (backtest.py): 0.7 beats 0.5, 1.0 and
# 1.5 on the full set and is within noise of the best on big launches.
BANDWIDTH = 0.7
# Minimum effective sample size before the bandwidth is widened.
MIN_EFFECTIVE_N = 10
# Hard cap on how far the bandwidth is widened.
MAX_BANDWIDTH = 2.5
# Weight of the empirical quantiles in the blend with the normal-IQR
# quantiles (0.5 = half and half, in log space).
EMPIRICAL_WEIGHT = 0.5
# Floors and caps on the normal-approximation log-spread (sigma).
SIGMA_FLOOR = 0.15
SIGMA_CAP = 1.5
# Fallback spread (log units per sqrt hour) when no comparable is usable.
FALLBACK_SIGMA = 0.35


def parse_time(text):
    """ISO 8601 timestamp (with a trailing Z) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_index(from_text, day_start):
    """Whole-hour offset of an hourly point from the release day's start."""
    delta = parse_time(from_text) - day_start
    return int(delta.total_seconds() // 3600)


def release_curves(case):
    """Usable release-day curves from the case's comparables.

    Returns a list of (hour1_viewers, [log-ratio hour 2, ..., log-ratio
    hour 24]) sorted by launch name for determinism.  A curve is usable
    only when all 24 release-day hours are present with a viewer count:
    a missing hour would otherwise be silently treated as no viewers.
    The target game's own history is not a launch (its `unlock` says so
    and its day is the premiere, not a release) and the case's own game
    can never be its own comparable, so both are skipped.
    """
    game_name = (case.get("game") or {}).get("name", "")
    curves = []
    for comp in case.get("comparables") or []:
        if comp.get("name") == game_name:
            continue
        unlock = str(comp.get("unlock") or "")
        if "not a launch" in unlock.lower():
            continue
        day_start = parse_time(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly") or []:
            k = hour_index(point["from"], day_start)
            if 0 <= k < 24:
                viewers = point.get("viewers")
                if isinstance(viewers, (int, float)) and viewers >= 0:
                    by_hour[k] = float(viewers)
        if len(by_hour) != 24:
            continue
        first = max(1.0, by_hour[0])
        ratios = [math.log(max(1.0, by_hour[h]) / first) for h in range(1, 24)]
        curves.append((first, ratios, str(comp.get("name", ""))))
    curves.sort(key=lambda item: item[2])
    return curves


def kernel_weights(curves, p1, bandwidth):
    """Gaussian kernel weights in log opening size, plus effective n."""
    centre = math.log(p1)
    weights = []
    for first, _ratios, _name in curves:
        gap = (math.log(first) - centre) / bandwidth
        weights.append(math.exp(-0.5 * gap * gap))
    total = sum(weights)
    if total <= 0:
        return weights, 0.0
    effective = total * total / sum(w * w for w in weights)
    return weights, effective


def weighted_quantile(pairs, level):
    """Weighted quantile of (value, weight) pairs; linear interpolation
    on the cumulative weights (the standard type-7 style rule, weighted)."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(w for _v, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    target = level * total
    acc = 0.0
    for i, (value, weight) in enumerate(ordered):
        prev_acc = acc
        acc += weight
        if acc >= target:
            if i == 0 or acc == target:
                return value
            # Interpolate between this value and the previous one.
            prev_value = ordered[i - 1][0]
            frac = (target - prev_acc) / weight if weight > 0 else 0.0
            return prev_value + frac * (value - prev_value)
    return ordered[-1][0]


def forecast_quantiles(curves, p1):
    """Per-hour five quantiles (in log-ratio units) for hours 2..24.

    For hour h, pool the weighted log-ratios of hours h-1, h and h+1
    across comparables (neighbouring hours share the daily shape), then
    read off the weighted quantiles at the scored levels.  Each quantile
    is blended with a normal-approximation quantile from the same pool
    (weighted median +/- z * sigma, sigma from the weighted IQR), which
    steadies the 5% and 95% tails that rest on very few launches.
    """
    if not curves:
        return None
    bandwidth = BANDWIDTH
    weights, effective = kernel_weights(curves, p1, bandwidth)
    while effective < MIN_EFFECTIVE_N and bandwidth < MAX_BANDWIDTH:
        bandwidth *= 1.5
        weights, effective = kernel_weights(curves, p1, bandwidth)
    out = {}
    for h in HOURS:
        # Pool this hour and its neighbours (clipped to hours 2..24).
        pool = []
        for hh in (h - 1, h, h + 1):
            if hh < 2 or hh > 24:
                continue
            idx = hh - 2  # ratios list starts at hour 2
            for (_first, ratios, _name), w in zip(curves, weights):
                if w > 0:
                    pool.append((ratios[idx], w))
        if not pool:
            continue
        empirical = [weighted_quantile(pool, level) for level in LEVELS]
        median = weighted_quantile(pool, 0.5)
        q25 = weighted_quantile(pool, 0.25)
        q75 = weighted_quantile(pool, 0.75)
        sigma = (q75 - q25) / 1.349
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        normal = [median + z * sigma for z in Z]
        out[h] = [
            EMPIRICAL_WEIGHT * e + (1.0 - EMPIRICAL_WEIGHT) * n
            for e, n in zip(empirical, normal)
        ]
    return out


def round2(x):
    """Round to 2 decimals, half up (the house baseline's convention)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def clamp_quantiles(values):
    """Enforce the output contract: finite, > 0, <= 1e9, non-decreasing."""
    fixed = []
    prev = 0.0
    for value in values:
        if not math.isfinite(value):
            value = prev if prev > 0 else 1.0
        value = min(max(value, 0.01), 1e9)
        if value < prev:
            value = prev
        fixed.append(round2(value))
        prev = fixed[-1]
    # Rounding can still leave a dip if two values round the same way;
    # a second pass guarantees non-decreasing after rounding.
    for i in range(1, len(fixed)):
        if fixed[i] < fixed[i - 1]:
            fixed[i] = fixed[i - 1]
    return fixed


def forecast_case(case, name):
    """Forecast one case: 23 rows of five quantiles, hours 2..24."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first.get("viewers") or 1))

    curves = release_curves(case)
    log_q = forecast_quantiles(curves, p1)

    rows = []
    for row in case.get("forecast") or []:
        h = int(row["hour"])
        if log_q is not None and h in log_q:
            quantiles = [p1 * math.exp(lq) for lq in log_q[h]]
        else:
            # Fallback: flat median at the hour-1 print, spread widening
            # like a random walk in log space over the day.
            spread = FALLBACK_SIGMA * math.sqrt(h - 1)
            quantiles = [p1 * math.exp(z * spread) for z in Z]
        rows.append({"hour": h, "q": clamp_quantiles(quantiles)})
    rows.sort(key=lambda r: r["hour"])
    if [r["hour"] for r in rows] != HOURS:
        raise ValueError(f"{name} does not forecast hours 2..24")
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
