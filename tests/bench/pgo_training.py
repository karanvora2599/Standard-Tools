"""Training workload for a profile-guided build of `_sqt_core`.

Step 2 of the PGO workflow in Documentation/30_build_guide.md: run this
against the instrumented extension (SQT_PGO_GENERATE=ON) and the process
writes its branch and call counts beside the extension when it exits; the
SQT_PGO_USE configure folds them into the profile.

It calls the raw bindings of every kernel family -- indicators, single and
batched backtests, the signal state machines, the portfolio bar loop,
rolling regression, Hurst, cointegration, Monte Carlo, GARCH, the Kalman
filters, the CUSUM scan, the panel statistics, the option chains, the
correlation matrix, the binomial lattice, the regime EM step and the two
order-event passes -- over a spread of sizes and parameters, with inputs
drawn from their own seeds rather than from the benchmark's. A profile only
knows the paths it was shown, so a kernel this script leaves out is
optimized as if it were cold.

    python tests/bench/pgo_training.py     # ~5 s instrumented; PGO_ROUNDS=5

It runs single-threaded unless SQT_NUM_THREADS is set: the instrumented
binary's counters are shared by every thread, and on 16 threads the
contention made a 2,000 x 2,000 batch_run_strategy grid 3.5x slower than on
one (0.96 s against 0.27 s) without changing which paths it takes. It
refuses to run against anything but an instrumented build, because training
the fallback trains nothing.
"""

import os
import sys

os.environ.setdefault("SQT_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import standard_quant_tools  # noqa: E402

ROUNDS = int(os.environ.get("PGO_ROUNDS", "5"))


def prices(rng, n, vol=0.01):
    return 100.0 * np.exp(np.cumsum(rng.normal(0, vol, n)))


def ohlc(rng, n):
    cl = prices(rng, n)
    spread = np.abs(rng.normal(0, 0.004, n))
    return cl * (1 + spread), cl * (1 - spread), cl


def rolling(fn, a, w):
    view = np.lib.stride_tricks.sliding_window_view(a, w)
    out = np.full(a.shape, np.nan)
    out[w - 1 :] = fn(view, axis=1)
    return out


def signals(rng, n, kind):
    if kind == "sparse":
        # a position held between occasional flips, as a trend rule trades
        s = np.zeros(n)
        flips = rng.choice(n, size=max(1, n // 40), replace=False)
        s[flips] = rng.choice([-1.0, 0.0, 1.0], flips.size)
        last = np.where(np.isin(np.arange(n), flips), np.arange(n), 0)
        return s[np.maximum.accumulate(last)]
    if kind == "long_only":
        return rng.choice([0.0, 1.0], n, p=[0.6, 0.4])
    return rng.choice([-1.0, 0.0, 1.0], n)


def indicators(c, rng):
    for n in (500, 5_000, 50_000):
        h, lo, cl = ohlc(rng, n)
        for period in (7, 14, 28):
            c.rsi(cl, period)
            c.adx(h, lo, cl, period)
            c.wilder_atr(h, lo, cl, period)
            c.stochastic_oscillator(h, lo, cl, period, 3)
        c.bollinger_bands(cl, 20, 2.0)
        c.parabolic_sar(h, lo, 0.02, 0.02, 0.2)
        c.technical_indicators(
            h,
            lo,
            cl,
            compute_rsi=True,
            compute_adx=True,
            compute_atr=True,
            compute_bollinger=True,
            compute_stochastic=True,
        )
    cl = np.vstack([prices(rng, 1_000) for _ in range(100)])
    c.technical_indicators_panel(
        cl * 1.01,
        cl * 0.99,
        cl,
        compute_rsi=True,
        compute_adx=True,
        compute_atr=True,
        compute_bollinger=True,
        compute_stochastic=True,
    )


def backtests(c, rng):
    for n in (250, 2_000, 10_000):
        p = prices(rng, n)
        for kind in ("dense", "sparse", "long_only"):
            s = signals(rng, n, kind)
            c.run_strategy(p, s)
            c.run_strategy(p, s, 1e5, 0.0005, 0.0002, 252.0, p * 1.001, 0.02)
    p = prices(rng, 1_000)
    c.batch_run_strategy(p, rng.choice([-1.0, 0.0, 1.0], (400, 1_000)))
    ind = np.cumsum(rng.normal(0, 1, (20, 1_000)), axis=1)
    pairs = np.array([[a, b] for a in range(20) for b in range(20) if a != b], np.int32)
    c.batch_backtest_crossover(p, ind, pairs)


def state_machines(c, rng):
    for n in (1_000, 20_000, 100_000):
        cl = prices(rng, n)
        for w in (20, 55):
            hi = rolling(np.max, cl, w)
            lo = rolling(np.min, cl, w // 2)
            entry = np.concatenate(([np.nan], hi[:-1]))
            exit_ = np.concatenate(([np.nan], lo[:-1]))
            c.donchian_state_machine(cl, entry, exit_)
        vwap = rolling(np.mean, cl, 20)
        for threshold in (0.005, 0.02):
            c.vwap_reversion_state_machine(cl, vwap, threshold)


def portfolio(c, rng):
    for n_bars, n_tickers, every in ((500, 50, 5), (2_000, 300, 21)):
        close = 100.0 * np.exp(
            np.cumsum(rng.normal(0, 0.015, (n_bars, n_tickers)), axis=0)
        )
        close = np.ascontiguousarray(close)
        rebal = np.arange(0, n_bars - 1, every, dtype=np.int64)
        w = rng.normal(0, 1, (rebal.size, n_tickers))
        w = np.ascontiguousarray(w / np.abs(w).sum(axis=1, keepdims=True))
        gaps = np.ones(n_bars)
        volume = np.full((n_bars, n_tickers), 5e7)
        vol = np.full((n_bars, n_tickers), 0.02)
        configs = (
            (0, None, None, 0, 0.0, 0.0, False, 0.0, 0.0),
            (1, None, None, 1, 0.005, 1.0, False, 0.0, 0.0),
            (0, volume, vol, 0, 0.0, 0.0, True, 0.1, 0.05),
        )
        for fill, dv, vl, model, per_share, min_c, impact, coef, adv in configs:
            c.run_portfolio_simulation(
                close, close, w, rebal, gaps, 1e6, 0.001, 0.001, 0.0005, 1.0,
                1.0, 25.0, 0.05, fill, dv, vl, model, per_share, min_c, impact,
                coef, adv,
            )  # fmt: skip


def regression_and_series(c, rng):
    for n in (1_000, 20_000):
        y, x = prices(rng, n), prices(rng, n)
        for w in (20, 60, 252):
            c.rolling_beta(y, x, w)
        c.ols2(y, x)
        c.kalman_filter_1state(y, x, 1e-4, 1e-3)
        c.kalman_filter_2state(y, x, 1e-4, 1e-3)
    for k in (1, 3, 5):
        c.rolling_factor_loadings(
            rng.normal(0, 1, 2_000), rng.normal(0, 1, (2_000, k)), 60
        )
    for n in (500, 2_000):
        a = rng.normal(0, 0.01, n)
        c.hurst_dfa(a)
        c.hurst_rs(a)
    c.rolling_hurst(rng.normal(0, 0.01, 1_500), 200, 1, "dfa", 10)
    c.rolling_hurst(rng.normal(0, 0.01, 1_500), 200, 5, "rs", 10)
    for n in (250, 2_000):
        c.engle_granger(prices(rng, n), prices(rng, n))
    panel = np.vstack([prices(rng, 500) for _ in range(12)])
    pairs = np.array([[a, b] for a in range(12) for b in range(a + 1, 12)], np.int32)
    c.batch_engle_granger(panel, pairs)
    r = rng.normal(0, 0.01, 1_000)
    c.simulate_forward_paths(r, 60, 2_000, 10, 1e4, 1)
    c.simulate_forward_paths_terminal(r, 252, 2_000, 20, 1e4, 2)
    sq = r**2
    for params in ((1e-6, 0.05, 0.9), (1e-5, 0.2, 0.7)):
        c.garch11_variance_recursion(sq, *params)
        c.garch11_neg_loglik(sq, *params, True)
        c.garch11_neg_loglik_grad(sq, *params, True)
    # the CUSUM scan of an AR(1) null, at the shapes liquidity_events and
    # the basis detector run it: 200 paths, reference window 30%
    for n in (120, 2_105):
        c.cusum_peaks(rng.normal(0, 1, (200, n)), int(n * 0.3), 0.5)
    c.cusum_peaks(rng.normal(0, 1, (37, 500)), 0, 0.0)


def panel_stats(c, rng):
    n_dates, n_entities = 250, 200
    codes = np.repeat(np.arange(n_dates, dtype=np.int64), n_entities)
    values = np.round(rng.normal(0, 1, (codes.size, 4)), 2)
    stats = c.fit_preprocess_stats(values, 0.01, 0.99)
    c.apply_preprocess_stats(
        values, stats["lo"], stats["hi"], stats["mean"], stats["std"]
    )
    c.standardize_by_date(values, codes, n_dates, 3.0)
    c.rank_by_date(values, codes, n_dates)
    y, yhat = values[:, 0].copy(), values[:, 1].copy()
    for spearman in (True, False):
        c.cross_sectional_correlation(y, yhat, codes, n_dates, spearman)
    c.cross_sectional_correlation(y, yhat, np.zeros(y.size, np.int64), 1, True)
    c.permutation_null_ic(y, yhat, codes, n_dates, 20, 5, True)
    dates = np.tile(np.arange(n_dates, dtype=np.int64), n_entities) * 86_400_000_000_000
    entity = np.repeat(np.arange(n_entities, dtype=np.int64), n_dates)
    c.label_uniqueness(dates, dates + 5 * 86_400_000_000_000, entity, n_entities)


def options(c, rng):
    for n in (50, 476, 5_000):
        strike = 230.0 * rng.uniform(0.6, 1.5, n)
        expiry = rng.uniform(2, 900, n) / 365.0
        vol = rng.uniform(0.08, 0.9, n)
        is_call = rng.integers(0, 2, n).astype(np.uint8)
        spot, rate, q = np.full(n, 230.0), np.full(n, 0.043), np.full(n, 0.005)
        fair = c.black_scholes_greeks_batch(spot, strike, expiry, vol, rate, q, is_call)
        quote = fair["price"] * np.exp(rng.normal(0.0, 0.03, n))
        c.implied_volatility_batch(quote, spot, strike, expiry, rate, q, is_call)
        spots = 230.0 * np.linspace(0.7, 1.3, 31)
        c.black_scholes_greeks_batch(
            spots, strike, expiry, vol, rate, q, is_call, grid=True
        )


def correlation(c, rng):
    # Both paths of pandas' correlation: complete panels in either order
    # (the shared-recursion path) and one with gaps (the per-pair loop).
    for n_rows, n_cols in ((60, 8), (500, 40), (2_106, 235)):
        returns = rng.normal(0, 0.012, (n_rows, n_cols))
        c.pearson_correlation(returns, 1)
        c.pearson_correlation(np.asfortranarray(returns), 1)
    gapped = rng.normal(0, 0.012, (500, 40))
    gapped[rng.random(gapped.shape) < 0.05] = np.nan
    c.pearson_correlation(gapped, 1)


def lattice_regimes_and_order_events(c, rng):
    # The binomial lattice, European and American, calls and puts, at the
    # tool's default 200 steps and up to 2,000.
    for steps in (10, 200, 2_000):
        dt = rng.uniform(0.1, 2.0) / steps
        up = np.exp(rng.uniform(0.1, 0.6) * np.sqrt(dt))
        powers = np.arange(steps + 1, dtype=float)
        u, d = np.power(up, powers), np.power(1.0 / up, powers)
        growth, disc = np.exp(0.01 * dt), np.exp(-0.04 * dt)
        p = (growth - 1.0 / up) / (up - 1.0 / up)
        for sign in (1.0, -1.0):
            for american in (False, True):
                c.binomial_lattice(
                    u, d, 100.0, rng.uniform(70, 130), sign, p, disc, american
                )
    # The regime step, 2 to 5 regimes, as detect_regimes drives it.
    for n, k in ((250, 2), (2_000, 3), (5_000, 4), (2_000, 5)):
        x = rng.normal(0.0003, 0.012, n)
        means = np.quantile(x, np.linspace(0.1, 0.9, k))
        variances, weights = np.full(k, x.var(ddof=1)), np.full(k, 1.0 / k)
        exponents = np.stack(
            [-0.5 * (x - m) ** 2 / v for m, v in zip(means, variances)]
        )
        for _ in range(20):
            _, counts, means, variances, exponents, _ = c.regime_em_step(
                x, np.exp(exponents), means, variances, weights
            )
            weights = counts / n
    # The order-event passes over a coded stream: adds, cancels, fills,
    # trades, a clear, and the snapshot that opens it.
    n = 200_000
    actions = rng.choice([0, 0, 0, 1, 1, 2, 4], size=n).astype(np.int64)
    actions[n // 2] = 3
    orders = np.minimum(np.cumsum(actions == 0), n - 1).astype(np.int64)
    terminate = (actions == 1) | (actions == 2)
    orders[terminate] = np.maximum(
        orders[terminate] - rng.integers(0, 50, terminate.sum()), 0
    )
    sides = rng.integers(0, 2, n).astype(np.int64)
    price = 100.0 + rng.integers(-8, 9, n) * 0.01
    size = rng.integers(1, 9, n) * 100.0
    snapshot = np.zeros(n, dtype=bool)
    snapshot[:100] = True
    stamps = np.cumsum(rng.integers(1, 10_000_000, n)).astype(np.int64)
    c.order_queue_ahead(orders, n, actions, sides, price, size, snapshot)
    c.order_lifetimes(orders, n, actions, snapshot, stamps)


def main():
    status = standard_quant_tools.native_build_status()
    if not status.used:
        sys.exit(f"the extension is not in use ({status.verdict}): nothing to train")
    from standard_quant_tools import _sqt_core as c

    pgo = dict(c.__build_info__).get("pgo")
    if pgo != "generate":
        sys.exit(
            f"this extension is not instrumented (pgo={pgo!r}); build with SQT_PGO_GENERATE=ON"
        )
    for round_ in range(ROUNDS):
        rng = np.random.default_rng(1_000 + round_)
        for family in (
            indicators,
            backtests,
            state_machines,
            portfolio,
            regression_and_series,
            panel_stats,
            options,
            # Last, so every family above draws the inputs it always has.
            correlation,
            lattice_regimes_and_order_events,
        ):
            family(c, rng)
        print(f"round {round_ + 1}/{ROUNDS} done", flush=True)


if __name__ == "__main__":
    main()
