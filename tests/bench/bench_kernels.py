"""Fresh performance baseline for every _sqt_core kernel.

Measures the RAW BINDING (not the Python wrapper), min-of-N, at several
problem sizes, and reports the empirical scaling exponent between the two
largest sizes so an O(n) kernel is visibly distinguishable from an O(n*w) one.

Run with SQT_NUM_THREADS=1 for the serial picture, unset for the parallel one.
"""

import gc
import os
import sys
import time

import numpy as np

from standard_quant_tools import _sqt_core as c

REPS = int(os.environ.get("BENCH_REPS", "7"))
RNG = np.random.default_rng(7)


def bench(fn, reps=REPS):
    fn()  # warm
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    finally:
        gc.enable()
    return best * 1e3  # ms


def prices(n, seed=1):
    r = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(r.normal(0, 0.01, n)))


def ohlc(n, seed=1):
    cl = prices(n, seed)
    return cl * 1.005, cl * 0.995, cl


CASES = []


def case(group, name, sizes, make):
    CASES.append((group, name, sizes, make))


# ── indicators: expected O(n) ────────────────────────────────────────────────
SIZES = [2_000, 20_000, 200_000]
case(
    "indicator",
    "rsi(14)",
    SIZES,
    lambda n: (lambda p=prices(n): lambda: c.rsi(p, 14))(),
)
case(
    "indicator",
    "wilder_atr(14)",
    SIZES,
    lambda n: (lambda hlc=ohlc(n): lambda: c.wilder_atr(*hlc, 14))(),
)
case(
    "indicator",
    "adx(14)",
    SIZES,
    lambda n: (lambda hlc=ohlc(n): lambda: c.adx(*hlc, 14))(),
)
case(
    "indicator",
    "parabolic_sar",
    SIZES,
    lambda n: (
        lambda hlc=ohlc(n): lambda: c.parabolic_sar(hlc[0], hlc[1], 0.02, 0.02, 0.2)
    )(),
)
case(
    "indicator",
    "bollinger(20)",
    SIZES,
    lambda n: (lambda p=prices(n): lambda: c.bollinger_bands(p, 20, 2.0))(),
)
case(
    "indicator",
    "stochastic(14,3)",
    SIZES,
    lambda n: (lambda hlc=ohlc(n): lambda: c.stochastic_oscillator(*hlc, 14, 3))(),
)
case(
    "indicator",
    "technical_indicators(all 5)",
    SIZES,
    lambda n: (
        lambda hlc=ohlc(n): lambda: c.technical_indicators(
            *hlc,
            compute_rsi=True,
            compute_adx=True,
            compute_atr=True,
            compute_bollinger=True,
            compute_stochastic=True,
        )
    )(),
)

# ── regression ───────────────────────────────────────────────────────────────
case(
    "regression",
    "rolling_beta(w=60)",
    SIZES,
    lambda n: (
        lambda y=prices(n, 2), x=prices(n, 3): lambda: c.rolling_beta(y, x, 60)
    )(),
)
case(
    "regression",
    "rolling_beta(w=252)",
    SIZES,
    lambda n: (
        lambda y=prices(n, 2), x=prices(n, 3): lambda: c.rolling_beta(y, x, 252)
    )(),
)
for k in (3, 10):
    case(
        "regression",
        f"rolling_factor_loadings(w=60,k={k})",
        [2_000, 5_000, 20_000],
        (
            lambda k: lambda n: (
                lambda y=RNG.normal(0, 1, n), f=RNG.normal(
                    0, 1, (n, k)
                ): lambda: c.rolling_factor_loadings(y, f, 60)
            )()
        )(k),
    )
case(
    "regression",
    "rolling_factor_loadings(w=252,k=3)",
    [2_000, 5_000, 20_000],
    lambda n: (
        lambda y=RNG.normal(0, 1, n), f=RNG.normal(
            0, 1, (n, 3)
        ): lambda: c.rolling_factor_loadings(y, f, 252)
    )(),
)

# ── cointegration ────────────────────────────────────────────────────────────
case(
    "cointegration",
    "engle_granger",
    [500, 2_000, 8_000],
    lambda n: (
        lambda y0=prices(n, 4), y1=prices(n, 5): lambda: c.engle_granger(y0, y1)
    )(),
)
case(
    "cointegration",
    "ols2",
    SIZES,
    lambda n: (lambda y=prices(n, 4), x=prices(n, 5): lambda: c.ols2(y, x))(),
)
case(
    "cointegration",
    "kalman_2state",
    SIZES,
    lambda n: (
        lambda y=prices(n, 4), x=prices(n, 5): lambda: c.kalman_filter_2state(
            y, x, 1e-4, 1e-3
        )
    )(),
)

# ── hurst ────────────────────────────────────────────────────────────────────
case(
    "hurst",
    "hurst_dfa",
    [1_000, 5_000, 20_000],
    lambda n: (lambda a=RNG.normal(0, 0.01, n): lambda: c.hurst_dfa(a))(),
)
case(
    "hurst",
    "rolling_hurst(w=200,step=1)",
    [2_000, 5_000, 20_000],
    lambda n: (
        lambda a=RNG.normal(0, 0.01, n): lambda: c.rolling_hurst(a, 200, 1, "dfa", 10)
    )(),
)

# ── backtest ─────────────────────────────────────────────────────────────────
case(
    "backtest",
    "run_strategy",
    SIZES,
    lambda n: (
        lambda p=prices(n), s=RNG.choice([-1.0, 0.0, 1.0], n): lambda: c.run_strategy(
            p, s
        )
    )(),
)
for combos in (500, 5_000):
    case(
        "backtest",
        f"batch_run_strategy(x{combos})",
        [500, 2_000],
        (
            lambda combos: lambda n: (
                lambda p=prices(n), s=RNG.choice([-1.0, 0.0, 1.0], (combos, n)).astype(
                    np.float64
                ): lambda: c.batch_run_strategy(p, s)
            )()
        )(combos),
    )
case(
    "backtest",
    "batch_backtest_crossover(x2450)",
    [500, 2_000],
    lambda n: (
        lambda p=prices(n), ind=np.cumsum(
            RNG.normal(0, 1, (50, n)), axis=1
        ), pi=np.array(
            [[a, b] for a in range(50) for b in range(50) if a != b], dtype=np.int32
        ): lambda: c.batch_backtest_crossover(
            p, ind, pi
        )
    )(),
)

# ── monte carlo / garch ──────────────────────────────────────────────────────
case(
    "montecarlo",
    "simulate_forward_paths(h=252)",
    [1_000, 10_000, 50_000],
    lambda sims: (
        lambda v=RNG.normal(0, 0.01, 2000): lambda: c.simulate_forward_paths(
            v, 252, sims, 20, 10000.0, 42
        )
    )(),
)
case(
    "montecarlo",
    "simulate_forward_paths_terminal(h=252)",
    [1_000, 10_000, 50_000],
    lambda sims: (
        lambda v=RNG.normal(0, 0.01, 2000): lambda: c.simulate_forward_paths_terminal(
            v, 252, sims, 20, 10000.0, 42
        )
    )(),
)
case(
    "garch",
    "garch11_neg_loglik_grad",
    SIZES,
    lambda n: (
        lambda r=np.abs(RNG.normal(0, 0.01, n)) ** 2: lambda: c.garch11_neg_loglik_grad(
            r, 1e-6, 0.1, 0.85, True
        )
    )(),
)


def main():
    threads = os.environ.get("SQT_NUM_THREADS", "(unset)")
    print(f"# baseline  SQT_NUM_THREADS={threads}  reps={REPS}")
    print(
        f"{'group':<13} {'kernel':<38} "
        + " ".join(f"{s:>11}" for s in ("size1", "size2", "size3"))
        + "   scaling"
    )
    print("-" * 108)
    results = {}
    for group, name, sizes, make in CASES:
        row, times = [], []
        for s in sizes:
            fn = make(s)
            ms = bench(fn)
            times.append(ms)
            row.append(f"{ms:>9.3f}ms")
        # empirical exponent between the last two sizes
        if len(sizes) >= 2 and times[-2] > 0:
            expo = np.log(times[-1] / times[-2]) / np.log(sizes[-1] / sizes[-2])
        else:
            expo = float("nan")
        print(
            f"{group:<13} {name:<38} "
            + " ".join(f"{v:>11}" for v in row)
            + f"   n^{expo:.2f}"
        )
        results[name] = (sizes, times)
        sys.stdout.flush()
    return results


if __name__ == "__main__":
    main()
