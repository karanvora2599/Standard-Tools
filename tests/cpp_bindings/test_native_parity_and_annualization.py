"""
Regression tests for the native-layer audit: annualization, Calmar parity,
wiped-out semantics, the ADF cancellation/failure split, and the walk-forward
execution-model mismatch.

The theme is backend parity. Every number here was measured against the
pre-fix build, so a regression fails with the size of the divergence rather
than a bare mismatch.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.metrics.risk_metrics import calmar_ratio

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)


class TestNativeCalmarMatchesPython:
    """
    Python was corrected to `(len(series) - 1) / periods_per_year` because N
    level observations span N-1 return intervals. The kernel still divided by
    n, so the two backends disagreed about the same backtest:

        n =  21   native -4.398278   python -4.582008   (4.01%)
        n =  63   native  4.136000   python  4.211419   (1.79%)
        n = 252   native  7.095421   python  7.131805   (0.51%)

    Negligible on long histories, material on exactly the short windows a
    walk-forward fold uses.
    """

    @pytest.mark.parametrize("n", [21, 63, 252])
    def test_calmar_agrees_across_backends(self, n):
        rng = np.random.default_rng(3)
        close = 100 * np.cumprod(1 + rng.normal(0.0015, 0.012, n))
        result = _cpp.run_strategy(close, np.ones(n), 10_000.0, 0.0, 0.0)
        native = result["calmar_ratio"]
        python = calmar_ratio(pd.Series(result["equity_curve"]))
        if np.isinf(native) or np.isinf(python):
            pytest.skip("no drawdown in this series; Calmar is infinite")
        assert native == pytest.approx(python, rel=1e-9)

    def test_a_wiped_out_strategy_reports_total_loss_on_both(self):
        """
        The native branch was skipped entirely for a non-positive final
        equity, leaving calmar_ratio at its 0.0 default — so a wiped-out
        backtest scored 0.0 (which reads as *neutral*) against Python's -1.0.
        """
        close = np.array([100.0, 80.0, 40.0, 5.0, 1.0])
        result = _cpp.run_strategy(close, np.full(5, 5.0), 10_000.0, 0.0, 0.0)
        assert result["final_equity"] <= 0.0
        assert result["calmar_ratio"] == pytest.approx(-1.0)
        assert calmar_ratio(pd.Series(result["equity_curve"])) == pytest.approx(-1.0)


class TestNativeAnnualizationIsAParameter:
    """
    `constexpr double kPPY = 252.0` was correct only for daily equity bars.
    The data and modeling layers now support 1h/5m/1m intervals and 24/7
    markets, so an hourly backtest reported a "Sharpe" annualized as though
    its bars were trading days.
    """

    def _series(self, n=500):
        return 100 * np.cumprod(1 + np.random.default_rng(1).normal(0.0001, 0.003, n))

    def test_volatility_scales_with_the_supplied_factor(self):
        close = self._series()
        signals = np.ones(len(close))
        daily = _cpp.run_strategy(close, signals, 10_000.0, 0.0, 0.0)
        hourly = _cpp.run_strategy(
            close, signals, 10_000.0, 0.0, 0.0, periods_per_year=252 * 6.5
        )
        ratio = hourly["annualized_volatility"] / daily["annualized_volatility"]
        assert ratio == pytest.approx(np.sqrt((252 * 6.5) / 252), rel=1e-9)

    def test_the_default_is_still_daily(self):
        """Existing callers must be unaffected."""
        close = self._series()
        signals = np.ones(len(close))
        assert _cpp.run_strategy(close, signals, 10_000.0, 0.0, 0.0)[
            "annualized_volatility"
        ] == pytest.approx(
            _cpp.run_strategy(
                close, signals, 10_000.0, 0.0, 0.0, periods_per_year=252.0
            )["annualized_volatility"]
        )

    def test_the_batch_kernel_honours_it_too(self):
        """A parameter that reached only the single-call path would leave the
        grid annualizing differently from the backtest it is meant to
        reproduce."""
        close = self._series()
        signals = np.ones(len(close))
        single = _cpp.run_strategy(
            close, signals, 10_000.0, 0.0, 0.0, periods_per_year=252 * 6.5
        )
        batch = _cpp.batch_run_strategy(
            close,
            signals.reshape(1, -1),
            10_000.0,
            0.0,
            0.0,
            periods_per_year=252 * 6.5,
        )
        assert batch[0][2] == pytest.approx(single["annualized_volatility"])


class TestAdfCancellationVersusFailure:
    """
    RSS is mathematically non-negative, and `yty - bXty` is a difference of
    two large nearly-equal quantities — the classic cancellation setup. Every
    negative RSS was treated as a perfect fit, so a numerical breakdown
    (materially negative RSS from an ill-conditioned solve) produced
    `adf_statistic = -inf`: the STRONGEST POSSIBLE evidence of cointegration,
    silently.
    """

    def test_a_genuine_perfect_fit_still_reports_maximal_stationarity(self):
        """The legitimate case that the -inf branch exists for: a perfectly
        collinear pair has a constant spread, and statsmodels converges on
        -inf / 0.0 here too."""
        x = np.cumsum(np.random.default_rng(0).normal(0, 1, 200)) + 100
        result = _cpp.engle_granger(2.0 * x + 5.0, x)
        assert result["adf_statistic"] == -np.inf
        assert result["p_value"] == 0.0

    def test_ordinary_pairs_are_unaffected(self):
        rng = np.random.default_rng(5)
        x = np.cumsum(rng.normal(0, 1, 400)) + 100
        cointegrated = _cpp.engle_granger(x + rng.normal(0, 0.5, 400), x)
        independent = _cpp.engle_granger(np.cumsum(rng.normal(0, 1, 400)) + 100, x)
        assert cointegrated["p_value"] < 0.05
        assert independent["p_value"] > 0.5
        assert np.isfinite(cointegrated["adf_statistic"])

    def test_the_tolerance_is_relative_not_absolute(self):
        """RSS carries the units of y-squared, so an absolute threshold would
        classify the same data differently merely rescaled."""
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "src"
            / "standard_quant_tools"
            / "_cpp"
            / "src"
            / "cointegration.cpp"
        ).read_text(encoding="utf-8")
        assert "rss_tol = 1e-8 * (yty" in source


class TestWalkForwardExecutionModelIsConsistent:
    """
    `backtest_grid` defaults to fill_price="close" while the out-of-sample leg
    honoured the caller's mode, so a walk-forward selected parameters under
    same-close execution and scored them under next-open execution.

    Measured across 25 random series with a realistic overnight gap, the
    WINNING parameter pair differed between the two fill modes on 7 of them —
    so the out-of-sample number was not a test of the parameters chosen.
    """

    def test_both_walk_forward_grids_pass_fill_price(self):
        import inspect
        import re

        from standard_quant_tools.agent import tools as T

        for name in (
            "run_walk_forward_backtest",
            "run_regime_adaptive_walkforward_backtest",
        ):
            source = inspect.getsource(getattr(T, name))
            assert re.search(
                r"backtest_grid\((?:[^()]|\([^()]*\))*fill_price=input_data\.fill_price",
                source,
                re.S,
            ), f"{name} optimizes under a different execution model than it evaluates"

    def test_fill_mode_really_can_change_the_winner(self):
        """Pins the mechanism, so the fix is never mistaken for a no-op."""
        from standard_quant_tools.backtest.engine import backtest_grid

        n = 400
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        rng = np.random.default_rng(0)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.013, n)), index=idx)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0.0, 0.005, n))
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(close, open_) * 1.006,
                "Low": np.minimum(close, open_) * 0.994,
                "Close": close,
                "Volume": 1e6,
            },
            index=idx,
        )
        grid = {"fast_period": [5, 10, 15, 20], "slow_period": [30, 50, 80]}
        picks = {}
        for mode in ("close", "next_open"):
            best = backtest_grid(
                df,
                "sma_crossover",
                grid,
                n_workers=1,
                fill_price=mode,
                sort_by="sharpe_ratio",
            ).iloc[0]
            picks[mode] = (int(best["fast_period"]), int(best["slow_period"]))
        assert (
            picks["close"] != picks["next_open"]
        ), "this seed no longer demonstrates the divergence; pick another"
