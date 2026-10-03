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

Since the CHANGELOG entry of 2026-10-02 the count is also decided at run
time: a region goes parallel on its estimated serial time rather than a unit
count, and the pooled rank's sort divides the threads among the calls
running at once from several Python threads. Neither may move a bit either,
so the same set of kernels is also run from concurrent callers, and under
each setting of the work threshold.
"""

from __future__ import annotations

import functools
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
CALLERS = 4

# argv[1]: "serial" runs every kernel once and prints {name: digest};
# "concurrent" runs the set from argv[2] Python threads at once and prints
# {"runs": [{name: digest}, ...], "errors": [...]} -- one dict per pass.
_KERNELS = r"""
import hashlib, json, sys, threading, traceback
import numpy as np
from standard_quant_tools import _sqt_core as c


def put(out, name, value):
    if isinstance(value, dict):
        for key in sorted(value):
            put(out, f"{name}.{key}", value[key])
        return
    arr = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    out[name] = f"{arr.dtype}{arr.shape}:{digest}"


def prices(n, seed):
    r = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, n)))


def jobs():
    rng = np.random.default_rng(20260928)
    todo = []
    add = lambda name, fn: todo.append((name, fn))

    # backtest.cpp: one summary per signal row, one per indicator pair
    p = prices(400, 1)
    signals = rng.choice([-1.0, 0.0, 1.0], size=(37, 400))
    add("batch_run_strategy", lambda: c.batch_run_strategy(p, signals))
    ind = np.cumsum(rng.normal(0, 1, (8, 400)), axis=1)
    pairs = np.array([[a, b] for a in range(8) for b in range(8) if a != b], np.int32)
    add("batch_backtest_crossover", lambda: c.batch_backtest_crossover(p, ind, pairs))

    # hurst.cpp: one DFA per window
    x = rng.normal(0, 0.01, 700)
    add("rolling_hurst", lambda: c.rolling_hurst(x, 120, 1, "dfa", 10))

    # cointegration.cpp: one Engle-Granger per pair
    panel = np.vstack([prices(260, 10 + i) for i in range(8)])
    eg_pairs = np.array([[a, b] for a in range(8) for b in range(a + 1, 8)], np.int32)
    add("batch_engle_granger", lambda: c.batch_engle_granger(panel, eg_pairs))

    # indicators.cpp: one ticker per task
    close = np.vstack([prices(300, 30 + i) for i in range(12)])
    add("technical_indicators_panel", lambda: c.technical_indicators_panel(
        close * 1.01, close * 0.99, close, compute_rsi=True, compute_adx=True,
        compute_atr=True, compute_bollinger=True, compute_stochastic=True))

    # monte_carlo.cpp: one seeded path per task
    returns = rng.normal(0, 0.01, 500)
    add("simulate_forward_paths",
        lambda: c.simulate_forward_paths(returns, 30, 600, 10, 1e4, 7))
    add("simulate_forward_paths_terminal",
        lambda: c.simulate_forward_paths_terminal(returns, 30, 600, 10, 1e4, 7))

    # panel_stats.cpp: per column, per date, per permutation, per entity, and
    # the pooled rank, whose sort is split into per-thread runs above 50,000
    # rows
    values = rng.normal(0, 1, (600, 7))
    values[rng.random(values.shape) < 0.03] = np.nan  # gaps, kept by apply
    stats = c.fit_preprocess_stats(values, 0.01, 0.99)
    add("fit_preprocess_stats", lambda: c.fit_preprocess_stats(values, 0.01, 0.99))
    add("apply_preprocess_stats", lambda: c.apply_preprocess_stats(
        values, stats["lo"], stats["hi"], stats["mean"], stats["std"]))
    n_dates, n_entities = 60, 50
    codes = np.repeat(np.arange(n_dates, dtype=np.int64), n_entities)
    cs = np.round(rng.normal(0, 1, (codes.size, 3)), 1)  # rounded: ties to rank
    add("standardize_by_date", lambda: c.standardize_by_date(cs, codes, n_dates, 3.0))
    add("rank_by_date", lambda: c.rank_by_date(cs, codes, n_dates))
    y_true, y_pred = cs[:, 0].copy(), cs[:, 1].copy()
    for spearman in (True, False):
        add(f"cross_sectional_correlation.{spearman}",
            lambda s=spearman: c.cross_sectional_correlation(
                y_true, y_pred, codes, n_dates, s))
    big_true = np.round(rng.normal(0, 1, 60_000), 2)
    big_pred = np.round(big_true + rng.normal(0, 1, 60_000), 2)
    zeros = np.zeros(60_000, np.int64)
    add("cross_sectional_correlation.pooled", lambda: c.cross_sectional_correlation(
        big_true, big_pred, zeros, 1, True))
    add("permutation_null_ic",
        lambda: c.permutation_null_ic(y_true, y_pred, codes, n_dates, 40, 11, True))
    add("permutation_null_ic.many",
        lambda: c.permutation_null_ic(y_true, y_pred, codes, n_dates, 997, 3, False))
    dates = np.tile(np.arange(n_dates, dtype=np.int64), n_entities) * 86_400_000_000_000
    entity = np.repeat(np.arange(n_entities, dtype=np.int64), n_dates)
    add("label_uniqueness", lambda: c.label_uniqueness(
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
    add("implied_volatility_batch",
        lambda: c.implied_volatility_batch(price, spot, strike, expiry, rate, q, is_call))
    spots = 230.0 * np.linspace(0.7, 1.3, 61)
    add("black_scholes_greeks_batch.grid", lambda: c.black_scholes_greeks_batch(
        spots, strike, expiry, vol, rate, q, is_call, grid=True))
    # A selection (gamma and the price_finite flag) on a spot axis several
    # blocks long, so a contract's row is split across tasks.
    wide = 230.0 * np.linspace(0.5, 1.5, 300)
    add("black_scholes_greeks_batch.gamma_grid", lambda: c.black_scholes_greeks_batch(
        wide, strike, expiry, vol, rate, q, is_call, True, (1 << 2) | (1 << 12)))

    # correlation.cpp: blocks of complete columns, then pairs with a gap.
    corr_values = rng.normal(0, 0.01, (300, 45))
    gapped = corr_values.copy()
    gapped[rng.random(gapped.shape) < 0.05] = np.nan
    add("pearson_correlation", lambda: c.pearson_correlation(corr_values, 1))
    add("pearson_correlation.gapped", lambda: c.pearson_correlation(gapped, 1))
    return todo


def run(todo):
    out = {}
    for name, fn in todo:
        put(out, name, fn())
    return out


todo = jobs()
if sys.argv[1] == "serial":
    print(json.dumps(run(todo)))
    sys.exit(0)

callers = int(sys.argv[2])
barrier = threading.Barrier(callers)
runs, errors = [], []
lock = threading.Lock()


def record(out):
    with lock:
        runs.append(out)


def in_step(k):
    # The same kernel from every caller at once: a barrier before each call.
    try:
        out = {}
        for name, fn in todo:
            barrier.wait(timeout=120)
            put(out, name, fn())
        record(out)
    except BaseException:
        errors.append(traceback.format_exc())
        barrier.abort()


def rotated(k):
    # Every caller through the whole set, each starting at a different
    # kernel, so different kernels' regions overlap.
    try:
        record(run(todo[k:] + todo[:k]))
    except BaseException:
        errors.append(traceback.format_exc())


for target in (in_step, rotated):
    threads = [threading.Thread(target=target, args=(k,)) for k in range(callers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
print(json.dumps({"runs": runs, "errors": errors}))
"""

_POLICY_VARIABLES = ("SQT_NUM_THREADS", "OMP_NUM_THREADS", "SQT_OMP_MIN_WORK")


def _run(settings: dict, *argv: str) -> Any:
    env = {k: v for k, v in os.environ.items() if k not in _POLICY_VARIABLES}
    env.update(settings)
    env.pop("SQT_DISABLE_NATIVE", None)
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_KERNELS), *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _digests(threads: int) -> dict:
    return _run(
        {
            "SQT_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "SQT_OMP_MIN_WORK": "0",
        },
        "serial",
    )


@functools.lru_cache(maxsize=1)
def _reference() -> dict:
    return _digests(THREAD_COUNTS[0])


def _differences(reference: dict, got: dict) -> list:
    assert set(got) == set(reference), sorted(set(got) ^ set(reference))
    return sorted(name for name, digest in got.items() if digest != reference[name])


@requires_cpp
class TestThreadCountIsNotObservable:
    def test_every_parallel_kernel_is_bit_identical_at_1_2_4_and_8_threads(self):
        """One fresh interpreter per count; the serial run is the reference.
        A kernel that differs names itself and the count it differed at."""
        reference = _reference()
        assert len(reference) >= 25, sorted(reference)
        differing = {}
        for threads in THREAD_COUNTS[1:]:
            for name in _differences(reference, _digests(threads)):
                differing.setdefault(name, []).append(threads)
        runtime = _cpp.__build_info__.get("openmp_runtime")
        assert not differing, (
            "these kernels' output depends on the thread count: "
            f"{differing} (OpenMP runtime: {runtime})"
        )


@requires_cpp
class TestTheWorkThresholdIsNotObservable:
    """Which calls go parallel is decided on estimated serial time, with
    SQT_OMP_MIN_WORK as the old unit rule when it is set (the CHANGELOG
    entry of 2026-10-02). Whichever way each call is decided, it answers
    the serial reference."""

    @pytest.mark.parametrize(
        "min_work",
        [None, "50000", "0", "not-a-number", "-5", ""],
        ids=["time-rule", "old-default", "always", "unparsable", "negative", "empty"],
    )
    def test_every_setting_answers_the_serial_reference(self, min_work):
        settings = {} if min_work is None else {"SQT_OMP_MIN_WORK": min_work}
        got = _run(settings, "serial")
        assert not _differences(_reference(), got)


@requires_cpp
class TestConcurrentCallersAreNotObservable:
    """Calls run at once from several Python threads compete for one
    runtime, and the pooled rank's sort splits into as many runs as its
    share of the threads, which depends on what else is running (the
    CHANGELOG entry of 2026-10-02). Every pass, from every caller, must
    still answer the serial reference."""

    @pytest.mark.parametrize(
        "settings",
        [
            {"SQT_OMP_MIN_WORK": "0"},
            {},
            {"SQT_OMP_MIN_WORK": "0", "SQT_NUM_THREADS": "8"},
        ],
        ids=["every-region-parallel", "default-policy", "capped-at-8"],
    )
    def test_concurrent_callers_answer_the_serial_reference(self, settings):
        reference = _reference()
        got = _run(settings, "concurrent", str(CALLERS))
        assert not got["errors"], got["errors"][0]
        assert len(got["runs"]) == 2 * CALLERS
        differing = {}
        for k, run in enumerate(got["runs"]):
            for name in _differences(reference, run):
                differing.setdefault(name, []).append(k)
        assert not differing, (
            "these kernels' output depends on what else was running: "
            f"{differing} (passes 0-{CALLERS - 1} in step, the rest rotated)"
        )
