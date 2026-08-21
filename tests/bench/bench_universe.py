"""Universe-scale benchmark: what actually costs at 2,000 tickers.

Everything here is measured on a small universe and extrapolated with the
measured per-unit cost, because the full 2,000-ticker shapes take minutes to
hours -- the extrapolation is stated explicitly so it can be checked.
"""

import gc
import time

import numpy as np
import pandas as pd

from standard_quant_tools import _sqt_core as c
from standard_quant_tools.analysis.cointegration import cointegration_test
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.modeling.portfolio_eval import (
    transform_predictions_to_weights,
)


def best_of(fn, reps=5):
    fn()
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    finally:
        gc.enable()
    return best


def fmt(sec):
    if sec < 1:
        return f"{sec*1e3:8.2f} ms"
    if sec < 90:
        return f"{sec:8.2f} s "
    if sec < 5400:
        return f"{sec/60:8.2f} min"
    return f"{sec/3600:8.2f} hr"


rng = np.random.default_rng(0)

print("=" * 78)
print("1. PAIRWISE COINTEGRATION SCAN  -  the O(N^2) universe problem")
print("=" * 78)
for n_bars in (500, 1000, 2000):
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_bars)))
    raw = best_of(lambda: c.engle_granger(a, b), 7)
    sa = pd.Series(a, index=pd.date_range("2015-01-01", periods=n_bars, freq="B"))
    sb = pd.Series(b, index=sa.index)
    wrapped = best_of(lambda: cointegration_test(sa, sb), 5)
    print(f"\n  n_bars={n_bars}")
    print(f"    raw binding engle_granger      {fmt(raw)}   per pair")
    print(
        f"    via cointegration_test wrapper {fmt(wrapped)}   per pair"
        f"   (+{(wrapped/raw - 1)*100:.0f}% Python overhead)"
    )
    for universe in (500, 2000):
        pairs = universe * (universe - 1) // 2
        print(
            f"      universe={universe:5d} -> {pairs:>10,} pairs"
            f" = {fmt(pairs*wrapped)} serial,"
            f" {fmt(pairs*wrapped/16)} on 16 cores"
        )

print()
print("=" * 78)
print("2. PORTFOLIO SIMULATION  -  per-bar Python loop over a dense universe")
print("=" * 78)
for n_tickers in (50, 200, 500):
    n_bars = 504
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    tickers = [f"T{i:04d}" for i in range(n_tickers)]
    price_data = {}
    for t in tickers:
        cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_bars)))
        price_data[t] = pd.DataFrame(
            {
                "Open": cl * 0.999,
                "High": cl * 1.01,
                "Low": cl * 0.99,
                "Close": cl,
                "Volume": np.full(n_bars, 1e7),
            },
            index=idx,
        )
    reb = idx[::5]
    w = rng.normal(0, 1, (len(reb), n_tickers))
    w = w / np.abs(w).sum(axis=1, keepdims=True)
    tw = pd.DataFrame(w, index=reb, columns=tickers)
    try:
        t = best_of(lambda: run_portfolio_simulation(price_data, tw), 3)
        print(
            f"  n_tickers={n_tickers:4d}  n_bars={n_bars}  rebalances={len(reb)}"
            f"   {fmt(t)}   ({t/n_bars*1e6:6.1f} us/bar)"
        )
    except Exception as exc:
        print(
            f"  n_tickers={n_tickers:4d}  SKIPPED: {type(exc).__name__}: {str(exc)[:80]}"
        )

print()
print("=" * 78)
print("3. PREDICTIONS -> WEIGHTS  -  per-date Python loop over the panel")
print("=" * 78)
from standard_quant_tools.modeling.specs import PredictionTransformSpec  # noqa: E402

for n_entities in (100, 500, 2000):
    n_dates = 504
    idx = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    cols = [f"T{i:04d}" for i in range(n_entities)]
    scores = pd.DataFrame(
        rng.normal(0, 1, (n_dates, n_entities)), index=idx, columns=cols
    )
    rets = pd.DataFrame(
        rng.normal(0, 0.01, (n_dates, n_entities)), index=idx, columns=cols
    )
    try:
        spec = PredictionTransformSpec(
            method="top_bottom_quantile",
            long_quantile=0.2,
            short_quantile=0.2,
            gross_exposure=1.0,
            net_exposure=0.0,
        )
        t = best_of(lambda: transform_predictions_to_weights(scores, spec, rets), 3)
        print(
            f"  n_entities={n_entities:5d}  n_dates={n_dates}   {fmt(t)}"
            f"   ({t/n_dates*1e6:6.1f} us/date)"
        )
    except Exception as exc:
        print(
            f"  n_entities={n_entities:5d}  SKIPPED: {type(exc).__name__}: {str(exc)[:80]}"
        )

print()
print("=" * 78)
print("4. MONTE CARLO  -  scaling to large simulation counts")
print("=" * 78)
vals = rng.normal(0.0005, 0.012, 2000)
for sims in (10_000, 100_000, 1_000_000):
    horizon = 252
    full_bytes = sims * horizon * 8
    t_term = best_of(
        lambda: c.simulate_forward_paths_terminal(vals, horizon, sims, 20, 10000.0, 42),
        3,
    )
    line = (
        f"  sims={sims:>9,}  horizon={horizon}  terminal-only {fmt(t_term)}"
        f"   full matrix = {full_bytes/1e9:6.2f} GB"
    )
    if full_bytes < 2e9:
        t_full = best_of(
            lambda: c.simulate_forward_paths(vals, horizon, sims, 20, 10000.0, 42), 3
        )
        line += f"   full {fmt(t_full)}"
    print(line)
