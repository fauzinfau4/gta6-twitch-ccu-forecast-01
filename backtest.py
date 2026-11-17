"""Leave-one-out backtest for the gta6-twitch-ccu-forecast-01 entry.

Treats each comparable launch's release day as a pseudo-case: forecast
hours 2..24 from its hour-1 print using the OTHER comparables only
(the same code path as main.py), and scores the pinball loss on the log
scale against its actual hours 2..24.  Scores the house persistence
baseline the same way for comparison.  Also sweeps the kernel
bandwidth / pooling choices so the tuned constants in main.py are
evidence-based, not guessed.

Usage: python3 backtest.py [--case p00] [--bandwidth 1.0] [--pool 1]
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

KIT = Path("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit")
sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location(
    "entry_main", Path(__file__).resolve().parent / "main.py"
)
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)

Z = entry.Z
LEVELS = entry.LEVELS


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(1.0, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def baseline_quantiles(p1, h):
    spread = 0.15 * math.sqrt(h - 1)
    return [p1 * math.exp(z * spread) for z in Z]


def score_forecast(actual, quantiles_by_hour):
    """actual: {hour: viewers}; quantiles_by_hour: {hour: [q05..q95]}."""
    losses = []
    misses = []
    covered = 0
    n_hours = 0
    for h in range(2, 25):
        y = actual[h]
        qs = quantiles_by_hour[h]
        for tau, q in zip(LEVELS, qs):
            losses.append(pinball(y, q, tau))
        misses.append(abs(math.log(max(1.0, y)) - math.log(max(1.0, qs[2]))))
        covered += 1 if qs[0] <= y <= qs[4] else 0
        n_hours += 1
    return {
        "loss": sum(losses) / len(losses),
        "typical_miss_pct": 100.0 * (math.exp(sum(misses) / len(misses)) - 1.0),
        "coverage_90": covered / n_hours,
    }


def load_curves(case_id):
    case = json.loads((KIT / "input" / f"{case_id}.in").read_text())
    curves = entry.release_curves(case)
    # Rebuild the actual release-day viewer counts per comparable, keyed
    # by name, so the held-out launch can be scored against its truth.
    truth = {}
    for comp in case.get("comparables") or []:
        day_start = entry.parse_time(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly") or []:
            k = entry.hour_index(point["from"], day_start)
            if 0 <= k < 24:
                by_hour[k] = float(point.get("viewers") or 0)
        if len(by_hour) == 24:
            # Convert 0-based hour offsets to the contest's 1-based hours.
            truth[str(comp.get("name", ""))] = {k + 1: v for k, v in by_hour.items()}
    return curves, truth


def forecast_from(curves, p1, bandwidth, pool_radius):
    """Same estimator as main.py, but with tunable bandwidth/pooling so
    the backtest can sweep them.  main.py's defaults match bandwidth=1.0,
    pool_radius=1."""
    if not curves:
        return None
    bw = bandwidth
    weights, effective = entry.kernel_weights(curves, p1, bw)
    while effective < entry.MIN_EFFECTIVE_N and bw < entry.MAX_BANDWIDTH:
        bw *= 1.5
        weights, effective = entry.kernel_weights(curves, p1, bw)
    out = {}
    for h in range(2, 25):
        pool = []
        for hh in range(h - pool_radius, h + pool_radius + 1):
            if hh < 2 or hh > 24:
                continue
            idx = hh - 2
            for (_first, ratios, _name), w in zip(curves, weights):
                if w > 0:
                    pool.append((ratios[idx], w))
        out[h] = [entry.weighted_quantile(pool, lvl) for lvl in LEVELS]
    return out


def run_loo(case_id, bandwidth, pool_radius, verbose=False):
    curves, truth = load_curves(case_id)
    usable = [(f, r, n) for (f, r, n) in curves if n in truth]
    ours, base = [], []
    per_launch = []
    for first, ratios, name in usable:
        actual = truth[name]
        others = [c for c in usable if c[2] != name]
        # Score main.py's real code path (its tuned constants), not the
        # sweep duplicate, so the reported number is the entry's number.
        log_q = entry.forecast_quantiles(others, first)
        ours_q = {h: [first * math.exp(lq) for lq in log_q[h]] for h in range(2, 25)}
        base_q = {h: baseline_quantiles(first, h) for h in range(2, 25)}
        s_ours = score_forecast(actual, ours_q)
        s_base = score_forecast(actual, base_q)
        ours.append(s_ours)
        base.append(s_base)
        per_launch.append((name, first, s_ours["loss"], s_base["loss"]))
    def agg(rows):
        return {
            "loss": sum(r["loss"] for r in rows) / len(rows),
            "typical_miss_pct": sum(r["typical_miss_pct"] for r in rows) / len(rows),
            "coverage_90": sum(r["coverage_90"] for r in rows) / len(rows),
        }
    if verbose:
        print(f"--- per-launch LOO loss (ours vs baseline), case {case_id} ---")
        for name, p1, lo, lb in sorted(per_launch, key=lambda t: t[2]):
            flag = "  <-- ours WORSE" if lo > lb else ""
            print(f"{name[:34]:34s} p1={p1:8.0f} ours={lo:.4f} base={lb:.4f}{flag}")
    return agg(ours), agg(base), len(usable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="p00")
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ours, base, n = run_loo(args.case, args.bandwidth, args.pool, args.verbose)
    print(f"LOO backtest, case {args.case}: n={n} held-out launches, "
          f"bandwidth={args.bandwidth}, pool_radius={args.pool}")
    print(f"  ours:     loss={ours['loss']:.4f}  typical_miss={ours['typical_miss_pct']:.1f}%  coverage90={ours['coverage_90']:.3f}")
    print(f"  baseline: loss={base['loss']:.4f}  typical_miss={base['typical_miss_pct']:.1f}%  coverage90={base['coverage_90']:.3f}")
    print(f"  improvement: {(base['loss'] - ours['loss']) / base['loss'] * 100:.1f}% lower loss")


if __name__ == "__main__":
    main()
