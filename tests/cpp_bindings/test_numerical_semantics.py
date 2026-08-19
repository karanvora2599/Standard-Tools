"""
Regression tests for the numerical-semantics pass.

Three defects motivated these, all of which produced wrong numbers silently
rather than raising, and none of which the pre-existing suite could see:

  1. Trade statistics priced every event at the previous CLOSE even when the
     caller asked to fill at the open, so the native summary and the Python
     trade log described different trades.
  2. Singularity tolerances carried a max(scale, 1.0) floor, which turns a
     relative test into an absolute one below unit scale -- the same analysis
     in different units gave different answers.
  3. ADF lag selection scored each candidate lag on its own sample, with an
     information criterion built from the unbiased variance, against a
     candidate set built by a different rule than statsmodels', and the
     Engle-Granger step-2 regression carried an intercept statsmodels omits.

The suite pins BEHAVIOUR (parity, invariance, agreement with a trusted
reference), not implementation text -- an earlier test in this directory
asserted a specific tolerance literal appeared in the .cpp source, which could
not fail for the reason it claimed to test and broke the moment the underlying
cancellation problem was actually fixed.
"""

import numpy as np
import pandas as pd
import pytest

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core",
    reason="native extension not built",
)

from standard_quant_tools.backtest.engine import (  # noqa: E402
    _compute_trade_stats,
    run_strategy,
)

FILL_MODES = ("close", "next_open", "hl2_exploratory")


def _ohlcv(rng, n):
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.015))
    open_ = close * (1 + rng.standard_normal(n) * 0.004)
    high = np.maximum(close, open_) * (1 + np.abs(rng.standard_normal(n)) * 0.003)
    low = np.minimum(close, open_) * (1 - np.abs(rng.standard_normal(n)) * 0.003)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.full(n, 1e6),
        },
        index=pd.date_range("2022-01-03", periods=n, freq="B"),
    )


class TestFillAwareTradeStatistics:
    """
    The equity curve used ref_prices[i]; trade accounting still used
    prices[i-1]. Under fill_price="next_open"/"hl2_exploratory" the summary
    therefore described a trade that never happened, and run_strategy_summary
    -- which is what the parameter grid and walk-forward rank on -- had the
    same defect.
    """

    def test_hand_verified_next_open_round_trip(self):
        """Entry fills at Open[1]=105, exit at Open[3]=125: 19.047619%.
        Booked at the previous closes (100 -> 120) it read 20%."""
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 105.0, 115.0, 125.0],
                "High": [101.0, 111.0, 121.0, 131.0],
                "Low": [99.0, 104.0, 114.0, 124.0],
                "Close": [100.0, 110.0, 120.0, 130.0],
                "Volume": [1e6] * 4,
            },
            index=idx,
        )
        sig = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
        r = run_strategy(
            df, sig, fill_price="next_open",
            commission_pct=0.0, slippage_pct=0.0, include_trade_log=True,
        )
        assert r["avg_trade_return_pct"] == pytest.approx((125 - 105) / 105 * 100, abs=1e-4)
        assert r["trade_log"]["entry_price"].iloc[0] == pytest.approx(105.0)
        assert r["trade_log"]["exit_price"].iloc[0] == pytest.approx(125.0)

    def test_a_losing_fill_is_not_reported_as_a_winner(self):
        """The sign, not just the magnitude, was wrong: a 104 -> 99 lot
        (-4.81%) was booked as 100 -> 102 (+2.00%), so one result dict
        carried win_rate=1.0 and profit_factor=inf beside a losing trade."""
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 104.0, 100.0, 99.0],
                "High": [105.0, 106.0, 103.0, 104.0],
                "Low": [98.0, 98.0, 99.0, 98.0],
                "Close": [100.0, 101.0, 102.0, 103.0],
                "Volume": [1e6] * 4,
            },
            index=idx,
        )
        sig = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
        r = run_strategy(
            df, sig, fill_price="next_open",
            commission_pct=0.0, slippage_pct=0.0, include_trade_log=True,
        )
        assert r["avg_trade_return_pct"] < 0
        assert r["win_rate"] == 0.0
        assert r["trade_log"]["return_pct"].iloc[0] < 0

    @pytest.mark.parametrize("fill_price", FILL_MODES)
    @pytest.mark.parametrize(
        "style",
        ["long_flat", "reversals", "resize_and_reduce", "leveraged_flips"],
    )
    def test_summary_stats_match_the_trade_log(self, fill_price, style):
        """Native summary vs the Python trade log, over every fill mode and
        every position-event shape: plain entries/exits, sign reversals,
        same-sign resizes, partial reductions, and leveraged flips through
        zero -- with and without transaction costs."""
        rng = np.random.default_rng(abs(hash((fill_price, style))) % (2**32))
        mismatches = []
        for trial in range(25):
            n = int(rng.integers(40, 140))
            df = _ohlcv(rng, n)
            idx = df.index
            if style == "long_flat":
                raw = rng.choice([0.0, 1.0], size=n)
            elif style == "reversals":
                raw = rng.choice([-1.0, 1.0], size=n)
            elif style == "resize_and_reduce":
                raw = rng.choice([0.0, 0.5, 1.0, 2.5, -1.0], size=n)
            else:
                raw = np.round(rng.uniform(-3, 3, size=n), 2)
            sig = pd.Series(raw, index=idx)
            cost_c, cost_s = (0.0, 0.0) if trial % 2 else (0.001, 0.0005)

            r = run_strategy(
                df, sig, fill_price=fill_price,
                commission_pct=cost_c, slippage_pct=cost_s,
                include_trade_log=True,
            )
            log = r["trade_log"]
            if len(log) == 0:
                continue
            py = _compute_trade_stats(log)
            if r["num_trades"] != len(log):
                mismatches.append((trial, "num_trades", r["num_trades"], len(log)))
            # 5e-3 not 0: _build_trade_log rounds return_pct to 4 decimals for
            # display before these stats are recomputed from it.
            if abs(r["avg_trade_return_pct"] - py["avg_trade_return_pct"]) > 5e-3:
                mismatches.append(
                    (trial, "avg", r["avg_trade_return_pct"], py["avg_trade_return_pct"])
                )
        assert not mismatches, mismatches

    def test_the_grid_kernel_agrees_with_the_single_backtest(self):
        """run_strategy_summary (batch path) carried the same defect
        independently, so the grid ranked strategies on the wrong number."""
        rng = np.random.default_rng(4)
        n = 120
        df = _ohlcv(rng, n)
        close = df["Close"].to_numpy()
        ref = df["Open"].to_numpy()
        signals = rng.choice([-1.0, 0.0, 1.0], size=n)

        single = _cpp.run_strategy(close, signals, 10_000.0, 0.001, 0.0005, 252.0, ref)
        batch = _cpp.batch_run_strategy(
            close, signals.reshape(1, -1), 10_000.0, 0.001, 0.0005, 252.0, ref
        )
        # batch rows are positional; num_trades / win_rate / avg live after the
        # equity-derived metrics, so compare against the dict by value.
        row = batch[0]
        assert single["num_trades"] in row
        assert any(
            v == pytest.approx(single["avg_trade_return_pct"], abs=1e-9)
            for v in row
            if isinstance(v, float)
        )


class TestScaleInvariance:
    """
    numerics::is_negligible_pivot floored its relative threshold at
    max(|scale|, 1.0), which makes the test ABSOLUTE for data below unit
    magnitude -- the smaller and therefore safer the data, the more aggressive
    the rejection. rolling_beta on x = [1e-8, 2e-8, ...], y = 2x returned NaN
    for a beta of exactly 2.
    """

    SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12)

    def test_rolling_beta_is_scale_invariant(self):
        rng = np.random.default_rng(3)
        x0 = rng.standard_normal(300)
        noise = rng.standard_normal(300) * 0.1
        base = None
        for s in self.SCALES:
            x = x0 * s
            beta = _cpp.rolling_beta(2.0 * x + noise * s, x, 60)[-1]
            assert np.isfinite(beta), f"NaN at scale {s:g}"
            if base is None:
                base = beta
            else:
                assert beta == pytest.approx(base, rel=1e-9), f"drift at scale {s:g}"

    def test_exactly_collinear_small_scale_data_recovers_the_slope(self):
        """The audit's minimal case: y = 2x at 1e-8 magnitude."""
        x = np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8])
        beta = _cpp.rolling_beta(2.0 * x, x, 5)
        assert beta[-1] == pytest.approx(2.0, rel=1e-9)

    def test_rolling_factor_loadings_are_scale_invariant(self):
        """Broke at 1e-6 -- 100x more reachable than the rolling_beta case --
        because the Cholesky pivot test compared every factor column against
        the intercept column's diagonal (the window length)."""
        rng = np.random.default_rng(3)
        f0 = rng.standard_normal((300, 3))
        noise = rng.standard_normal(300) * 0.1
        true_beta = np.array([1.5, -0.5, 0.8])
        base = None
        for s in self.SCALES:
            factors = f0 * s
            y = factors @ true_beta + noise * s
            row = _cpp.rolling_factor_loadings(y, factors, 60)[-1]
            assert np.all(np.isfinite(row)), f"NaN at scale {s:g}"
            # Only the LOADINGS are scale-invariant. y and the factors are
            # scaled together here, so each slope is a ratio of two quantities
            # that both carry s and the s cancels -- but the intercept carries
            # the units of y and must scale linearly with it. Asserting
            # invariance on row[0] would be asserting the regression is wrong.
            loadings = row[1:]
            if base is None:
                base = loadings
            else:
                assert loadings == pytest.approx(base, rel=1e-9), f"drift at scale {s:g}"
            # Loose: a 60-bar window with noise recovers the true loadings to
            # a couple of percent. The tight assertion is the invariance one
            # above; this only catches a result that is stable but wrong.
            assert loadings == pytest.approx(true_beta, rel=0.05), f"wrong at scale {s:g}"

    def test_ols2_is_scale_invariant(self):
        rng = np.random.default_rng(8)
        x0 = rng.standard_normal(200)
        noise = rng.standard_normal(200) * 0.05
        base = None
        for s in self.SCALES:
            x = x0 * s
            slope = _cpp.ols2(3.0 * x + noise * s, x)["slope"]
            assert np.isfinite(slope), f"NaN at scale {s:g}"
            if base is None:
                base = slope
            else:
                assert slope == pytest.approx(base, rel=1e-9), f"drift at scale {s:g}"

    def test_a_wide_column_scale_spread_is_not_mistaken_for_collinearity(self):
        """The mirror-image failure. A design of [intercept, factors*1e13] is
        perfectly well conditioned, but its intercept column's R diagonal is
        1e13 smaller than the factors', so a rank test that ratios against the
        largest diagonal rejects it. Column equilibration is what stops the
        rank decision from depending on the caller's units."""
        rng = np.random.default_rng(5)
        f0 = rng.standard_normal((200, 3))
        for s in (1e10, 1e13, 1e15):
            factors = f0 * s
            y = factors @ np.array([1.5, -0.5, 0.8]) + rng.standard_normal(200) * s * 1e-3
            row = _cpp.rolling_factor_loadings(y, factors, 60)[-1]
            assert np.all(np.isfinite(row)), f"NaN at factor scale {s:g}"
            assert row[1] == pytest.approx(1.5, rel=1e-3)


class TestRankDeficiencyPolicy:
    """
    One policy, both backends: a rank-deficient window yields NaN. The C++
    path returned NaN while the NumPy fallback returned its minimum-norm
    solution, so the same call gave blanks or numbers depending only on
    whether the extension had been compiled.
    """

    def test_duplicate_factors_yield_nan_in_both_backends(self):
        import standard_quant_tools.analysis.multi_factor as mf

        rng = np.random.default_rng(3)
        n = 200
        f = rng.standard_normal((n, 2))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        frame = pd.DataFrame(
            np.column_stack([f[:, 0], f[:, 0], f[:, 1]]),
            index=idx,
            columns=["f1", "f1_dup", "f2"],
        )
        y = pd.Series(f[:, 0] * 2.0 + rng.standard_normal(n) * 0.1, index=idx)

        had_cpp = mf.HAS_CPP
        try:
            mf.HAS_CPP = True
            native = mf.rolling_factor_loadings(y, frame, window=60).iloc[-1]
            mf.HAS_CPP = False
            numpy_ = mf.rolling_factor_loadings(y, frame, window=60).iloc[-1]
        finally:
            mf.HAS_CPP = had_cpp

        assert native.isna().all()
        assert numpy_.isna().all()


class TestAdfMatchesStatsmodels:
    """
    Exact differential tests against statsmodels, per the audit: selected lag
    AND statistic, not a broad cointegrated/not-cointegrated assertion.
    """

    @staticmethod
    def _series(rng, n):
        e = rng.standard_normal(n)
        a1, a2 = rng.uniform(-0.5, 0.5), rng.uniform(-0.35, 0.35)
        u = np.zeros(n)
        for t in range(2, n):
            u[t] = a1 * u[t - 1] + a2 * u[t - 2] + e[t]
        return np.cumsum(u) if rng.random() < 0.5 else u * 3.0

    def test_engle_granger_matches_statsmodels_coint(self):
        coint = pytest.importorskip("statsmodels.tsa.stattools").coint
        rng = np.random.default_rng(20260819)
        for _ in range(40):
            n = int(rng.integers(80, 300))
            u = self._series(rng, n)
            b = np.cumsum(rng.standard_normal(n)) * 0.5 + 100
            a = 2.0 * b + u * rng.uniform(0.5, 3.0)

            native = _cpp.engle_granger(a, b)
            expected_stat, expected_p, _ = coint(a, b, trend="c", autolag="aic")

            assert native["adf_statistic"] == pytest.approx(expected_stat, abs=1e-7)
            # p-values come from the MacKinnon (2010) response surface, which is
            # an approximation of the same distribution statsmodels interpolates.
            assert native["p_value"] == pytest.approx(expected_p, abs=5e-3)

    def test_lag_selection_uses_a_common_sample(self):
        """Scoring each lag on its own sample compares criteria built from
        different response vectors. Measured over 200 series, the shipped rule
        disagreed with statsmodels on 33% of them; a common hold-back sample
        plus the MLE variance reached exact agreement on all 200."""
        coint = pytest.importorskip("statsmodels.tsa.stattools").coint
        rng = np.random.default_rng(4242)
        disagreements = 0
        trials = 30
        for _ in range(trials):
            n = int(rng.integers(120, 280))
            u = self._series(rng, n)
            b = np.cumsum(rng.standard_normal(n)) * 0.5 + 100
            a = 2.0 * b + u * rng.uniform(0.5, 3.0)
            native = _cpp.engle_granger(a, b)
            expected_stat, _, _ = coint(a, b, trend="c", autolag="aic")
            if abs(native["adf_statistic"] - expected_stat) > 1e-7:
                disagreements += 1
        assert disagreements == 0, f"{disagreements}/{trials} disagreed"

    def test_bic_also_matches(self):
        coint = pytest.importorskip("statsmodels.tsa.stattools").coint
        rng = np.random.default_rng(77)
        for _ in range(20):
            n = int(rng.integers(100, 260))
            u = self._series(rng, n)
            b = np.cumsum(rng.standard_normal(n)) * 0.5 + 100
            a = 1.3 * b + u * rng.uniform(0.5, 2.0)
            native = _cpp.engle_granger(a, b, -1, False)  # use_aic=False -> BIC
            expected_stat, _, _ = coint(a, b, trend="c", autolag="bic")
            assert native["adf_statistic"] == pytest.approx(expected_stat, abs=1e-7)


class TestScalarConfigValidation:
    """
    Scalar configuration parameters are validated at the binding; input DATA and
    indicator window/period arguments deliberately are NOT (see the note in
    bindings.cpp -- degenerate data yields NaN so one bad bar cannot reject a
    whole panel, which this project already fixed once).
    """

    PRICES = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    SIGNALS = np.array([1.0, 1.0, 0.0, 0.0, 1.0])

    @pytest.mark.parametrize(
        "kwargs, why",
        [
            (dict(initial_capital=0.0), "zero capital gave total_return=nan beside sharpe=9.99"),
            (dict(initial_capital=-100.0), "negative capital reported a +1.7% return"),
            (dict(periods_per_year=-1.0), "negative annualization gave NaN volatility"),
            (dict(commission_pct=-0.1), "negative commission made the strategy profitable"),
            (dict(slippage_pct=-0.1), "negative slippage, same"),
        ],
    )
    def test_nonsense_backtest_config_is_rejected(self, kwargs, why):
        args = dict(
            initial_capital=10_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            periods_per_year=252.0,
        )
        args.update(kwargs)
        with pytest.raises(ValueError):
            _cpp.run_strategy(
                self.PRICES,
                self.SIGNALS,
                args["initial_capital"],
                args["commission_pct"],
                args["slippage_pct"],
                args["periods_per_year"],
            ), why

    @pytest.mark.parametrize("bad", [dict(n_simulations=0), dict(horizon_days=0),
                                     dict(block_size=0), dict(initial_capital=-1.0)])
    def test_nonsense_simulation_config_is_rejected(self, bad):
        args = dict(horizon_days=10, n_simulations=5, block_size=5, initial_capital=10_000.0)
        args.update(bad)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(
                np.full(50, 0.01),
                args["horizon_days"],
                args["n_simulations"],
                args["block_size"],
                args["initial_capital"],
                1,
            )

    def test_valid_config_still_runs(self):
        out = _cpp.run_strategy(self.PRICES, self.SIGNALS, 10_000.0, 0.0, 0.0, 252.0)
        # executed = [0, 1, 1, 0, 0] -> 1.01 * (102/101) = 1.02 exactly.
        assert out["total_return"] == pytest.approx(0.02, abs=1e-12)

    def test_degenerate_data_still_yields_nan_rather_than_raising(self):
        """The contract these validators must not break."""
        assert np.all(np.isnan(_cpp.rsi(self.PRICES, -5)))
        assert np.all(np.isnan(_cpp.rolling_beta(self.PRICES, self.PRICES, 0)))
        nan_in = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        assert np.all(np.isnan(_cpp.rolling_beta(nan_in, self.PRICES, 3)))
