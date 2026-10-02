"""Build-variant benchmark: the same kernels, whichever extension is built.

Compares builds of `_sqt_core` rather than kernels against their fallbacks --
the OpenMP runtime (vcomp against LLVM's libomp under SQT_OPENMP_LLVM), a
profile-guided build against a plain one. Each kernel is timed on the raw
binding, warm, REPS times in this process, and the median is reported with
the fastest and slowest run beside it. A build comparison needs more than one
process as well: run this several times per build and compare the medians of
the per-process medians, which is what `--json` is for.

    python tests/bench/bench_build.py                      # OpenMP default
    SQT_NUM_THREADS=1 python tests/bench/bench_build.py    # serial kernels
    python tests/bench/bench_build.py --json out.json      # machine-readable

The header prints the build facts (`_sqt_core.__build_info__`), so a result
file says which build produced it.

TWO REGIMES, because the runtimes differ in both and a library meets both:

- WARM: each kernel is called back to back for BENCH_WARMUP seconds
  (default 0.25) before its REPS timed calls. A single warm-up call was not
  enough in either direction: a serial kernel timed right after a parallel
  one ran up to 1.6x slower on vcomp, whose idle workers keep spinning for
  about 100 ms after a region, and a short kernel timed right after a pause
  ran up to 3x slower while the core left its idle clock.
- COLD (`--cold N`, parallel kernels only): N single calls, each after
  BENCH_IDLE seconds (default 0.3) without work -- a kernel called between
  stretches of Python, after the runtime's workers have gone to sleep.
"""

import argparse
import gc
import json
import os
import statistics
import time

import numpy as np

import standard_quant_tools
from standard_quant_tools import _sqt_core as c

REPS = int(os.environ.get("BENCH_REPS", "15"))
WARMUP = float(os.environ.get("BENCH_WARMUP", "0.25"))
IDLE = float(os.environ.get("BENCH_IDLE", "0.3"))


def cold(fn, calls):
    runs = []
    for _ in range(calls):
        time.sleep(IDLE)
        t0 = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - t0) * 1e3)
    return runs


def timed(fn, reps=REPS):
    fn()
    until = time.perf_counter() + WARMUP
    while time.perf_counter() < until:
        fn()
    runs = []
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            runs.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return runs


def prices(n, seed):
    r = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, n)))


def ohlc(n, seed):
    cl = prices(n, seed)
    return cl * 1.005, cl * 0.995, cl


def cases():
    rng = np.random.default_rng(7)
    out = []

    def case(name, parallel, fn):
        out.append((name, parallel, fn))

    # rolling regression: serial, the AVX2+FMA reduction on a capable CPU
    y, x = prices(200_000, 2), prices(200_000, 3)
    case("rolling_beta n=200k w=60", False, lambda: c.rolling_beta(y, x, 60))
    case("rolling_beta n=200k w=252", False, lambda: c.rolling_beta(y, x, 252))
    yf, f = rng.normal(0, 1, 5_000), rng.normal(0, 1, (5_000, 3))
    case(
        "rolling_factor_loadings n=5k w=252 k=3",
        False,
        lambda: c.rolling_factor_loadings(yf, f, 252),
    )

    # backtest grids: parallel across combinations
    p2k = prices(2_000, 1)
    s2k = rng.choice([-1.0, 0.0, 1.0], (2_000, 2_000))
    case(
        "batch_run_strategy 2000 bars x 2000",
        True,
        lambda: c.batch_run_strategy(p2k, s2k),
    )
    p500 = prices(500, 4)
    s500 = rng.choice([-1.0, 0.0, 1.0], (5_000, 500))
    case(
        "batch_run_strategy 500 bars x 5000",
        True,
        lambda: c.batch_run_strategy(p500, s500),
    )
    ind = np.cumsum(rng.normal(0, 1, (50, 2_000)), axis=1)
    pairs = np.array([[a, b] for a in range(50) for b in range(50) if a != b], np.int32)
    case(
        "batch_backtest_crossover 2000 x 2450",
        True,
        lambda: c.batch_backtest_crossover(p2k, ind, pairs),
    )

    # the indicator panel: parallel across tickers
    close = np.vstack([prices(1_000, 100 + i) for i in range(500)])
    high, low = close * 1.01, close * 0.99
    case(
        "technical_indicators_panel 500 x 1000",
        True,
        lambda: c.technical_indicators_panel(
            high,
            low,
            close,
            compute_rsi=True,
            compute_adx=True,
            compute_atr=True,
            compute_bollinger=True,
            compute_stochastic=True,
        ),
    )

    # portfolio simulation: the serial bar loop on dense matrices
    n_bars, n_tickers = 2_000, 1_000
    pc = np.ascontiguousarray(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, (n_bars, n_tickers)), axis=0))
    )
    rebal = np.arange(0, n_bars - 1, 21, dtype=np.int64)
    w = rng.normal(0, 1, (rebal.size, n_tickers))
    w = np.ascontiguousarray(w / np.abs(w).sum(axis=1, keepdims=True))
    gaps = np.ones(n_bars)

    def portfolio():
        res = c.run_portfolio_simulation(
            pc, pc, w, rebal, gaps, 1e6, 0.001, 0.001, 0.0005, 1.0, 1.0,
            0.0, 0.0, 0, None, None, 0, 0.0, 0.0, False, 0.0, 0.0,
        )  # fmt: skip
        assert int(res["status"]) == 0, res["status"]

    case("run_portfolio_simulation 1000 x 2000", False, portfolio)

    # option chains: parallel across contracts / grid points
    n = 476
    strike = 230.0 * np.tile(np.linspace(0.7, 1.35, 14), 34)
    expiry = np.repeat(np.linspace(7, 730, 34), 14) / 365.0
    vol = 0.24 - 0.35 * np.log(strike / 230.0)
    is_call = (np.arange(n) % 2).astype(np.uint8)
    spot, rate, q = np.full(n, 230.0), np.full(n, 0.043), np.full(n, 0.005)
    fair = c.black_scholes_greeks_batch(spot, strike, expiry, vol, rate, q, is_call)
    quote = fair["price"] * np.exp(rng.normal(0.0, 0.02, n))
    case(
        "implied_volatility_batch 476",
        True,
        lambda: c.implied_volatility_batch(
            quote, spot, strike, expiry, rate, q, is_call
        ),
    )
    spots = 230.0 * np.linspace(0.7, 1.3, 61)
    case(
        "black_scholes_greeks_batch 61 x 476",
        True,
        lambda: c.black_scholes_greeks_batch(
            spots, strike, expiry, vol, rate, q, is_call, grid=True
        ),
    )

    # other parallel kernels
    a = rng.normal(0, 0.01, 5_000)
    case(
        "rolling_hurst n=5000 w=252",
        True,
        lambda: c.rolling_hurst(a, 252, 1, "dfa", 10),
    )
    r = rng.normal(0, 0.01, 2_000)
    case(
        "simulate_forward_paths 20k x 252",
        True,
        lambda: c.simulate_forward_paths(r, 252, 20_000, 20, 1e4, 42),
    )

    # branch-heavy serial code: single backtests, state machines, indicators
    p5k = prices(5_000, 5)
    sig5k = rng.choice([-1.0, 0.0, 1.0], 5_000)
    case("run_strategy n=5000", False, lambda: c.run_strategy(p5k, sig5k))
    big = prices(200_000, 6)
    # a 20-bar entry channel and a 10-bar exit channel, known at the prior bar
    windows = np.lib.stride_tricks.sliding_window_view
    entry = np.full(big.size, np.nan)
    exit_ = np.full(big.size, np.nan)
    entry[20:] = windows(big, 20).max(axis=1)[:-1]
    exit_[10:] = windows(big, 10).min(axis=1)[:-1]
    case(
        "donchian_state_machine n=200k",
        False,
        lambda: c.donchian_state_machine(big, entry, exit_),
    )
    vwap = big * (1.0 + rng.normal(0, 0.01, big.size))
    case(
        "vwap_reversion_state_machine n=200k",
        False,
        lambda: c.vwap_reversion_state_machine(big, vwap, 0.01),
    )
    h, lo_, cl = ohlc(200_000, 8)
    case(
        "technical_indicators n=200k (all 5)",
        False,
        lambda: c.technical_indicators(
            h,
            lo_,
            cl,
            compute_rsi=True,
            compute_adx=True,
            compute_atr=True,
            compute_bollinger=True,
            compute_stochastic=True,
        ),
    )
    case(
        "parabolic_sar n=200k",
        False,
        lambda: c.parabolic_sar(h, lo_, 0.02, 0.02, 0.2),
    )
    # the CUSUM scan of an AR(1) null: serial, 200 paths of 2,105 steps.
    # Drawn last so the inputs of every case above are unchanged.
    zc = rng.normal(0, 1, (200, 2_105))
    case("cusum_peaks 200 x 2105", False, lambda: c.cusum_peaks(zc, 631, 0.5))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write the per-kernel runs here")
    parser.add_argument(
        "--cold",
        type=int,
        default=0,
        metavar="N",
        help="also time N calls of each parallel kernel, each after an idle pause",
    )
    args = parser.parse_args()

    status = standard_quant_tools.native_build_status()
    if not status.used:
        raise SystemExit(
            f"the extension is not in use ({status.verdict}); nothing to measure"
        )
    facts = dict(c.__build_info__)
    threads = os.environ.get("SQT_NUM_THREADS", "(unset)")
    print(
        f"# build: openmp={facts.get('openmp')} runtime={facts.get('openmp_runtime')} "
        f"pgo={facts.get('pgo')} native_arch={facts.get('native_arch')} "
        f"digest={facts['source_digest'][:12]}"
    )
    print(f"# SQT_NUM_THREADS={threads}  reps={REPS}")
    results = {}
    for name, parallel, fn in cases():
        runs = timed(fn)
        results[name] = {"parallel": parallel, "runs_ms": runs}
        line = (
            f"{name:<42} {statistics.median(runs):10.3f} ms"
            f"   [{min(runs):.3f} .. {max(runs):.3f}]"
        )
        if args.cold and parallel:
            cold_runs = cold(fn, args.cold)
            results[name]["cold_ms"] = cold_runs
            line += f"   cold {statistics.median(cold_runs):.3f} ms"
        print(line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {"facts": facts, "threads": threads, "reps": REPS, "results": results},
                fh,
                indent=1,
            )


if __name__ == "__main__":
    main()
