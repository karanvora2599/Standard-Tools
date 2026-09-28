"""
Every parallel kernel answers the same doubles whatever the thread count.

Each OpenMP loop in the extension splits independent work -- a parameter
combination, a ticker, a pair, a contract, a date, a simulated path, a run
of a sort -- across threads, so how many threads ran it must not be
observable in a single bit of the output. Two things make that worth a test
of its own:

  - the count cannot be varied inside one process. OpenMP reads
    OMP_NUM_THREADS once, when its thread pool starts, and SQT_NUM_THREADS
    is cached on first use, so setting either with monkeypatch and calling
    again compares a build with itself. Each count here is a fresh
    interpreter;
  - which runtime schedules the work is a build choice (vcomp, or LLVM's
    libomp under MSVC with SQT_OPENMP_LLVM), and a scheduling difference that
    leaked into the numbers would show up here as a difference between
    counts on that runtime.

SQT_OMP_MIN_WORK=0 sends every region with more than one task parallel, so
inputs small enough for a unit test still take the split. One thread is the
serial path, the reference the others are held to.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

THREAD_COUNTS = (1, 2, 4, 8)

_KERNELS = r"""
import hashlib, json
import numpy as np
from standard_quant_tools import _sqt_core as c

rng = np.random.default_rng(20260928)
out = {}

def put(name, value):
    if isinstance(value, dict):
        for key in sorted(value):
            put(f"{name}.{key}", value[key])
        return
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    out[name] = f"{arr.dtype}{arr.shape}:{digest}"

def prices(n, seed):
    r = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, n)))

# backtest.cpp: one summary per signal row, one per indicator pair
p = prices(400, 1)
signals = rng.choice([-1.0, 0.0, 1.0], size=(37, 400))
put("batch_run_strategy", c.batch_run_strategy(p, signals))
ind = np.cumsum(rng.normal(0, 1, (8, 400)), axis=1)
pairs = np.array([[a, b] for a in range(8) for b in range(8) if a != b], np.int32)
put("batch_backtest_crossover", c.batch_backtest_crossover(p, ind, pairs))

# hurst.cpp: one DFA per window
put("rolling_hurst", c.rolling_hurst(rng.normal(0, 0.01, 700), 120, 1, "dfa", 10))

# cointegration.cpp: one Engle-Granger per pair
panel = np.vstack([prices(260, 10 + i) for i in range(8)])
eg_pairs = np.array([[a, b] for a in range(8) for b in range(a + 1, 8)], np.int32)
put("batch_engle_granger", c.batch_engle_granger(panel, eg_pairs))

# indicators.cpp: one ticker per task
close = np.vstack([prices(300, 30 + i) for i in range(12)])
put("technical_indicators_panel", c.technical_indicators_panel(
    close * 1.01, close * 0.99, close, compute_rsi=True, compute_adx=True,
    compute_atr=True, compute_bollinger=True, compute_stochastic=True))

# monte_carlo.cpp: one seeded path per task
returns = rng.normal(0, 0.01, 500)
put("simulate_forward_paths", c.simulate_forward_paths(returns, 30, 600, 10, 1e4, 7))
put("simulate_forward_paths_terminal",
    c.simulate_forward_paths_terminal(returns, 30, 600, 10, 1e4, 7))

# panel_stats.cpp: per column, per date, per permutation, per entity, and the
# pooled rank, whose sort is split into per-thread runs above 50,000 rows
values = rng.normal(0, 1, (600, 7))
stats = c.fit_preprocess_stats(values, 0.01, 0.99)
put("fit_preprocess_stats", stats)
put("apply_preprocess_stats", c.apply_preprocess_stats(
    values, stats["lo"], stats["hi"], stats["mean"], stats["std"]))
n_dates, n_entities = 60, 50
codes = np.repeat(np.arange(n_dates, dtype=np.int64), n_entities)
cs = np.round(rng.normal(0, 1, (codes.size, 3)), 1)  # rounded: ties to rank
put("standardize_by_date", c.standardize_by_date(cs, codes, n_dates, 3.0))
put("rank_by_date", c.rank_by_date(cs, codes, n_dates))
y_true, y_pred = cs[:, 0].copy(), cs[:, 1].copy()
for spearman in (True, False):
    put(f"cross_sectional_correlation.{spearman}",
        c.cross_sectional_correlation(y_true, y_pred, codes, n_dates, spearman))
big_true = np.round(rng.normal(0, 1, 60_000), 2)
big_pred = np.round(big_true + rng.normal(0, 1, 60_000), 2)
put("cross_sectional_correlation.pooled", c.cross_sectional_correlation(
    big_true, big_pred, np.zeros(60_000, np.int64), 1, True))
put("permutation_null_ic",
    c.permutation_null_ic(y_true, y_pred, codes, n_dates, 40, 11, True))
dates = np.tile(np.arange(n_dates, dtype=np.int64), n_entities) * 86_400_000_000_000
entity = np.repeat(np.arange(n_entities, dtype=np.int64), n_dates)
put("label_uniqueness", c.label_uniqueness(
    dates, dates + 5 * 86_400_000_000_000, entity, n_entities))

# options.cpp: one contract per task, and a spot x contract grid
n = 476
strike = 230.0 * rng.uniform(0.7, 1.35, n)
expiry = rng.uniform(7, 730, n) / 365.0
vol = rng.uniform(0.12, 0.6, n)
is_call = (np.arange(n) % 2).astype(np.uint8)
spot = np.full(n, 230.0)
rate, q = np.full(n, 0.043), np.full(n, 0.005)
greeks = c.black_scholes_greeks_batch(spot, strike, expiry, vol, rate, q, is_call)
price = greeks["price"] * np.exp(rng.normal(0.0, 0.02, n))
put("implied_volatility_batch",
    c.implied_volatility_batch(price, spot, strike, expiry, rate, q, is_call))
spots = 230.0 * np.linspace(0.7, 1.3, 61)
put("black_scholes_greeks_batch.grid", c.black_scholes_greeks_batch(
    spots, strike, expiry, vol, rate, q, is_call, grid=True))

print(json.dumps(out))
"""


def _digests(threads: int) -> dict:
    env = {
        **os.environ,
        "SQT_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        "SQT_OMP_MIN_WORK": "0",
    }
    env.pop("SQT_DISABLE_NATIVE", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_KERNELS)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@requires_cpp
class TestThreadCountIsNotObservable:
    def test_every_parallel_kernel_is_bit_identical_at_1_2_4_and_8_threads(self):
        """One fresh interpreter per count; the serial run is the reference.
        A kernel that differs names itself and the count it differed at."""
        reference = _digests(THREAD_COUNTS[0])
        assert len(reference) >= 25, sorted(reference)
        differing = {}
        for threads in THREAD_COUNTS[1:]:
            got = _digests(threads)
            assert set(got) == set(reference)
            for name, digest in got.items():
                if digest != reference[name]:
                    differing.setdefault(name, []).append(threads)
        runtime = _cpp.__build_info__.get("openmp_runtime")
        assert not differing, (
            "these kernels' output depends on the thread count: "
            f"{differing} (OpenMP runtime: {runtime})"
        )
