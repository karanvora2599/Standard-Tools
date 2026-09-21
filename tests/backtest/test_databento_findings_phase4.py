"""
Phase 4 of the Databento live fix plan (CHANGELOG, 2026-09-20): the backtest engine and
the portfolio surface.

The live findings (CHANGELOG, 2026-09-20; D6 and "Also
in the backtest and portfolio surface") measured each of these on real
prices. The tests here reproduce each defect's shape offline and pin the
fix:

  D6      a bar that moves like a split is screened and warned about
  turnover run_strategy returns the turnover and cost it computed
  callable a custom strategy's signal is range-checked in the grid
  ADV     max_adv_participation sizes a trade down instead of refusing
  PSD     an indefinite covariance is repaired and the repair is named
  rows    estimate_covariance says how many rows a short history removed
  NaN     build_portfolio refuses or fills under a named policy
  HRP     the weights do not depend on column order
  adv     plan_rebalance names an entity missing from adv
  source  the screener takes a data source
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import BacktestCompactInput
from standard_quant_tools.agent.runtimes.backtest.tools import run_backtest_compact
from standard_quant_tools.backtest.engine import (
    SPLIT_SCREEN_THRESHOLD,
    backtest_grid,
    run_strategy,
)
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import (
    hierarchical_risk_parity,
    max_diversification,
    risk_parity,
)
from standard_quant_tools.portfolio.covariance import estimate_covariance
from standard_quant_tools.portfolio.portfolio import build_portfolio
from standard_quant_tools.portfolio.rebalance import plan_rebalance
from standard_quant_tools.screener.screener import screen_stocks


def _bars(closes, start="2023-01-02") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    index = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": np.full(len(closes), 1_000_000.0),
        },
        index=index,
    )


def _with_split(n: int = 120, at: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    closes = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    closes[at:] = closes[at:] / 2.0  # an unadjusted 2:1 split
    return _bars(closes)


# ── D6 ───────────────────────────────────────────────────────────────────


class TestTheSplitScreen:
    """LRCX's 10:1 split reported buy-and-hold at -62.40% against +276.04%
    true, and a short held through it printed a fictitious +93%; nothing
    under backtest/ read the provider's adjusted flag."""

    def test_a_split_sized_bar_is_named_with_its_date(self):
        frame = _with_split()
        signals = pd.Series(1.0, index=frame.index)
        result = run_strategy(frame, signals)
        (warning,) = [w for w in result["warnings"] if w.startswith("SPLIT SCREEN")]
        assert str(frame.index[60].date()) in warning
        move = float(frame["Close"].pct_change().iloc[60])
        assert f"{move:+.1%}" in warning and move < -0.45
        assert "not known" in warning

    def test_the_provider_flag_phrases_the_warning(self):
        frame = _with_split()
        signals = pd.Series(1.0, index=frame.index)
        frame.attrs["adjusted"] = False
        (warning,) = [
            w for w in run_strategy(frame, signals)["warnings"] if "SPLIT" in w
        ]
        assert "adjusted=False" in warning and "real bar" in warning
        (warning,) = [
            w
            for w in run_strategy(frame, signals, adjusted=True)["warnings"]
            if "SPLIT" in w
        ]
        assert "adjusted=True" in warning

    def test_an_ordinary_series_is_not_screened(self):
        rng = np.random.default_rng(7)
        frame = _bars(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 200)))
        result = run_strategy(frame, pd.Series(1.0, index=frame.index))
        assert not any("SPLIT" in w for w in result["warnings"])
        assert SPLIT_SCREEN_THRESHOLD == 0.35

    def test_the_warning_reaches_the_compact_tool(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        frame = _with_split(n=300, at=150)
        frame.attrs["adjusted"] = False
        provider = MagicMock()
        provider.get_ohlcv.return_value = frame
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        result = run_backtest_compact(
            BacktestCompactInput(
                symbol="LRCX",
                start_date="2023-01-02",
                end_date="2024-02-01",
                strategy_type="sma_crossover",
                parameters={"fast_period": 10, "slow_period": 30},
            )
        )
        assert any("SPLIT SCREEN" in w for w in result.warnings)
        assert any("adjusted=False" in w for w in result.warnings)


class TestTurnoverAndCostAreReturned:
    def test_a_round_trip_is_two_units_of_turnover(self):
        frame = _bars(np.linspace(100, 110, 20))
        signals = pd.Series(0.0, index=frame.index)
        signals.iloc[5:10] = 1.0
        result = run_strategy(frame, signals, commission_pct=0.001, slippage_pct=0.0005)
        assert result["turnover"] == pytest.approx(2.0)
        assert result["realized_cost_pct"] == pytest.approx(2.0 * 0.0015)

    def test_the_compact_tool_carries_them(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        rng = np.random.default_rng(8)
        frame = _bars(100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, 300)))
        provider = MagicMock()
        provider.get_ohlcv.return_value = frame
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        result = run_backtest_compact(
            BacktestCompactInput(
                symbol="AAA",
                start_date="2023-01-02",
                end_date="2024-02-01",
                strategy_type="sma_crossover",
                parameters={"fast_period": 5, "slow_period": 20},
                commission_pct=0.001,
                slippage_pct=0.0005,
            )
        )
        assert result.costs.turnover is not None and result.costs.turnover > 0
        assert result.costs.realized_cost_pct == pytest.approx(
            result.costs.turnover * 0.0015, abs=1e-6
        )


class TestACustomCallableIsRangeChecked:
    """A signal of 2.0 silently ran a levered book through every grid
    combination."""

    def test_a_levered_signal_is_refused_by_name(self):
        frame = _bars(np.linspace(100, 120, 60))

        def levered(price_data, k):
            return pd.Series(float(k), index=price_data.index)

        with pytest.raises(ValidationError, match=r"outside \[-1, 1\]"):
            backtest_grid(frame, strategy=levered, param_grid={"k": [2.0]}, n_workers=1)

    def test_a_unit_signal_runs(self):
        frame = _bars(np.linspace(100, 120, 60))

        def unit(price_data, k):
            return pd.Series(float(k), index=price_data.index)

        grid = backtest_grid(
            frame, strategy=unit, param_grid={"k": [1.0, -1.0]}, n_workers=1
        )
        assert len(grid) == 2


# ── the ADV cap ──────────────────────────────────────────────────────────


class TestTheAdvCapIsACap:
    def test_an_uncapped_rebalance_records_nothing_capped(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        price_data = {"AAPL": _bars([100, 101, 102, 103, 104], "2023-01-02")}
        targets = pd.DataFrame({"AAPL": [0.5]}, index=[dates[0]])
        result = run_portfolio_simulation(
            price_data, targets, max_adv_participation=0.5
        )
        first = result["rebalance_log"].iloc[0]
        assert first["n_capped"] == 0 and first["capped_notional"] == 0.0
        assert not any("sized down" in w for w in result["warnings"])

    def test_the_book_holds_what_the_cap_allowed(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        frame = _bars([100, 100, 100, 100, 100], "2023-01-02")
        frame["Volume"] = 100.0  # $10,000 of daily volume at $100
        targets = pd.DataFrame({"AAPL": [1.0]}, index=[dates[0]])
        result = run_portfolio_simulation(
            {"AAPL": frame},
            targets,
            initial_capital=10_000.0,
            max_adv_participation=0.1,
        )
        first = result["rebalance_log"].iloc[0]
        # 10% of $10,000 ADV is $1,000 of a $10,000 order.
        assert first["n_capped"] == 1
        assert first["capped_notional"] == pytest.approx(9_000.0, rel=1e-6)
        assert first["capped"] == ["AAPL"]
        assert float(result["leverage_curve"].max()) == pytest.approx(0.1, rel=1e-2)


# ── PSD ──────────────────────────────────────────────────────────────────


def _indefinite():
    names = ["A", "B", "C"]
    matrix = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]) * 0.04
    assert np.linalg.eigvalsh(matrix).min() < 0
    return pd.DataFrame(matrix, index=names, columns=names)


class TestAnIndefiniteCovarianceIsRepairedAndNamed:
    """A ragged panel's pairwise covariance had a smallest eigenvalue of
    -2.77e-03 and max_diversification returned a NEGATIVE weighted average
    volatility with zero warnings."""

    def test_max_diversification_repairs_and_warns(self):
        result = max_diversification(_indefinite())
        assert any("not positive semi-definite" in w for w in result["warnings"])
        assert result["weighted_average_volatility"] > 0
        assert np.isfinite(result["diversification_ratio"])

    def test_risk_parity_repairs_and_warns(self):
        result = risk_parity(_indefinite())
        assert any("not positive semi-definite" in w for w in result["warnings"])
        assert sum(result["weights"].values()) == pytest.approx(1.0)

    def test_a_proper_covariance_is_left_alone(self):
        names = ["A", "B"]
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=names, columns=names)
        result = risk_parity(cov)
        assert not any("semi-definite" in w for w in result["warnings"])


class TestCovarianceSaysWhatItDropped:
    """One short history silently removed 400 of 512 rows with warnings: []
    and moved risk-parity weights by 12.4% of NAV."""

    def _returns(self, n=300):
        rng = np.random.default_rng(9)
        return pd.DataFrame(
            {k: rng.normal(0, 0.01, n) for k in ("A", "B", "C")},
            index=pd.bdate_range("2023-01-02", periods=n),
        )

    def test_dropped_rows_are_counted_and_the_short_asset_named(self):
        returns = self._returns()
        returns.iloc[:100, returns.columns.get_loc("C")] = np.nan
        result = estimate_covariance(returns, method="sample")
        assert result["n_rows_dropped"] == 100
        assert result["n_observations"] == 200
        (warning,) = [w for w in result["warnings"] if "were dropped" in w]
        assert "100 of 300" in warning and "'C'" in warning

    def test_a_complete_panel_drops_nothing(self):
        result = estimate_covariance(self._returns(), method="sample")
        assert result["n_rows_dropped"] == 0
        assert not any("were dropped" in w for w in result["warnings"])


class TestBuildPortfolioMissingPolicy:
    """A NaN passed through and three downstream calculations treated it
    three ways; 20 missing days added +290 bp of CAGR."""

    def _returns(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame(
            {k: rng.normal(0, 0.01, 100) for k in ("A", "B")},
            index=pd.bdate_range("2023-01-02", periods=100),
        )
        frame.iloc[10:30, 0] = np.nan
        return frame

    def test_refused_by_default_with_the_count(self):
        with pytest.raises(ValidationError, match="20 missing"):
            build_portfolio(self._returns(), [0.5, 0.5])

    def test_zero_keeps_every_date(self):
        series = build_portfolio(self._returns(), [0.5, 0.5], missing="zero")
        assert len(series) == 100 and np.isfinite(series.to_numpy()).all()

    def test_drop_removes_the_holed_dates(self):
        series = build_portfolio(self._returns(), [0.5, 0.5], missing="drop")
        assert len(series) == 80

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(ValidationError, match="missing="):
            build_portfolio(self._returns(), [0.5, 0.5], missing="ignore")


class TestHrpDoesNotDependOnColumnOrder:
    """Forty permutations of one universe moved single weights by up to
    8.7 pp."""

    def test_every_permutation_gives_the_same_weights(self):
        rng = np.random.default_rng(11)
        n = 400
        f1, f2 = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
        data = {}
        for i in range(4):
            data[f"g1_{i}"] = 0.9 * f1 + rng.normal(0, 0.004, n)
            data[f"g2_{i}"] = 0.9 * f2 + rng.normal(0, 0.004, n)
        frame = pd.DataFrame(data)
        reference = hierarchical_risk_parity(frame)["weights"]
        for seed in range(5):
            order = list(np.random.default_rng(seed).permutation(frame.columns))
            weights = hierarchical_risk_parity(frame[order])["weights"]
            for name, value in reference.items():
                assert weights[name] == pytest.approx(value, abs=1e-12)


class TestPlanRebalanceNamesAMissingAdv:
    def test_a_name_missing_from_adv_is_refused_by_name(self):
        with pytest.raises(ValidationError, match=r"\['DDD'\]"):
            plan_rebalance(
                {"AAA": 1.0},
                {"AAA": 0.0, "DDD": 1.0},
                portfolio_value=100e6,
                adv={"AAA": 50e6},
                max_days=3,
            )

    def test_without_adv_the_plan_is_unpriced_as_before(self):
        result = plan_rebalance(
            {"AAA": 1.0}, {"AAA": 0.0, "DDD": 1.0}, portfolio_value=100e6, max_days=3
        )
        assert result["total_cost_bps"] is None


class TestTheScreenerTakesASource:
    def test_the_source_reaches_the_factory(self, monkeypatch, sample_ohlcv):
        seen = []
        provider = MagicMock()
        provider.get_ohlcv_async = AsyncMock(return_value=sample_ohlcv)
        provider.get_ohlcv.return_value = sample_ohlcv

        def get_provider(source="yfinance", *args, **kwargs):
            seen.append(source)
            return provider

        monkeypatch.setattr(DataFactory, "get_provider", get_provider)
        result = screen_stocks(
            ["AAA", "BBB"], filters={}, n_workers=1, source="databento"
        )
        assert seen == ["databento"]
        assert len(result) == 2
