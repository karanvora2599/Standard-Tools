"""
Answers that looked right at the tool boundary.

Five of them, each one line or one field away from the library that already
had the right number:

  - a grid sorted by `annualized_volatility` ranked descending and returned
    its MOST volatile combination, then reported that metric as absent from
    every row it returned;
  - `run_screener` accepted a column name it could not sort by and returned
    input order with nothing said;
  - a score panel whose conversion put more than `max_position_pct` on one
    name had the whole simulation refused, over a weight the caller never
    stated;
  - `max_drawdown_pct` meant a fraction in the stress test and a percentage
    in the futures engine, both reachable, neither saying which;
  - `run_pca_analysis` reported loadings from one decomposition and
    contributions from a second one with different defaults, and
    `get_volatility_estimators` / `optimize_hierarchical_risk_parity`
    annualized at 252 whatever bars they were handed.

See the CHANGELOG entry of 2026-09-22.
"""

from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    BacktestOptInput,
    OptimizationRun,
    PCAInput,
    PortfolioSimulationInput,
    ScreenerInput,
    SignalType,
    VolatilityEstimatorsInput,
)
from standard_quant_tools.agent.runtimes.backtest.futures_tools import (
    FuturesBacktestInput,
    run_futures_backtest,
)
from standard_quant_tools.agent.runtimes.backtest.tools import (
    _LOWER_IS_BETTER,
    _cap_constructed_weights,
    run_backtest_optimization,
    run_portfolio_simulation,
)
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    HRPInput,
    optimize_hierarchical_risk_parity,
)
from standard_quant_tools.agent.runtimes.research.tools import (
    get_volatility_estimators,
    run_pca_analysis,
    run_screener,
)
from standard_quant_tools.analysis.pca import factor_contributions, pca_returns
from standard_quant_tools.backtest.stress_test import replay_stress_scenario
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError

START = "2020-01-01"
END = "2022-09-01"


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    spread = close.abs() * 0.01 + 0.05
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": pd.Series(2_000_000.0, index=close.index),
        }
    )


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """700 bars with enough structure that a parameter grid separates."""
    rng = np.random.default_rng(99)
    n = 700
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.013, n))
    dates = pd.date_range(START, periods=n, freq="B")
    return _ohlcv(pd.Series(close, index=dates))


@pytest.fixture
def one_symbol(trending_ohlcv, monkeypatch):
    provider = MagicMock()
    provider.get_ohlcv.return_value = trending_ohlcv
    provider.get_ohlcv_async = AsyncMock(return_value=trending_ohlcv)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


# ── the grid's sort direction ────────────────────────────────────────────


GRID = {"fast_period": [5, 10, 15, 20], "slow_period": [40, 60]}


def _optimize(sort_by: str):
    return run_backtest_optimization(
        BacktestOptInput(
            symbol="AAPL",
            strategy="sma_crossover",
            start_date=START,
            end_date=END,
            param_grid=GRID,
            sort_by=sort_by,
            top_n=20,
        )
    )


class TestTheGridSortsTowardTheBetterNumber:
    def test_sorting_by_volatility_returns_the_quietest_combination(self, one_symbol):
        result = _optimize("annualized_volatility")
        volatilities = [row.annualized_volatility for row in result.top_results]
        assert len(volatilities) == 8
        assert None not in volatilities
        assert volatilities[0] == min(volatilities)
        assert volatilities == sorted(volatilities)

    def test_sorting_by_sharpe_still_returns_the_highest(self, one_symbol):
        result = _optimize("sharpe_ratio")
        sharpes = [row.sharpe_ratio for row in result.top_results]
        assert sharpes[0] == max(sharpes)
        assert sharpes == sorted(sharpes, reverse=True)

    def test_the_two_orderings_disagree_on_this_grid(self, one_symbol):
        """Otherwise the direction would be untestable here."""
        by_vol = _optimize("annualized_volatility").best_params
        by_sharpe = _optimize("sharpe_ratio").best_params
        assert by_vol != by_sharpe

    def test_drawdown_is_signed_so_descending_is_already_right_for_it(self, one_symbol):
        result = _optimize("max_drawdown")
        drawdowns = [row.max_drawdown for row in result.top_results]
        assert all(value <= 0.0 for value in drawdowns)
        assert drawdowns[0] == max(drawdowns)
        assert "max_drawdown" not in _LOWER_IS_BETTER

    def test_trade_count_sorts_descending(self, one_symbol):
        result = _optimize("num_trades")
        counts = [row.num_trades for row in result.top_results]
        assert counts[0] == max(counts)


class TestEverySortableMetricIsOnTheRow:
    def test_the_literal_is_a_subset_of_the_row_fields(self):
        """`top_results[0][sort_by]` returned None after sorting by it,
        for four of the ten choices."""
        choices = set(get_args(BacktestOptInput.model_fields["sort_by"].annotation))
        assert choices <= set(OptimizationRun.model_fields)

    @pytest.mark.parametrize(
        "metric",
        ["annualized_volatility", "profit_factor", "win_rate", "avg_trade_return_pct"],
    )
    def test_the_four_that_were_missing_now_carry_a_number(self, one_symbol, metric):
        result = _optimize(metric)
        assert getattr(result.top_results[0], metric) is not None


# ── the screener's sort column ───────────────────────────────────────────


PASSES_EVERYTHING = {"pe_ratio_max": 100.0}
PASSES_NOTHING = {"pe_ratio_max": 1.0}


class TestTheScreenerSaysWhenItCannotSort:
    def test_a_column_this_screen_did_not_produce_is_refused_by_name(
        self, patched_factory
    ):
        with pytest.raises(ValidationError) as excinfo:
            run_screener(
                ScreenerInput(
                    tickers=["AAPL", "MSFT"],
                    filters=PASSES_EVERYTHING,
                    sort_by="pe_ratio",
                )
            )
        message = str(excinfo.value)
        assert "sort_by='pe_ratio'" in message
        assert "forward_pe" in message
        assert "sort_by=None" in message

    def test_a_real_column_sorts(self, patched_factory):
        result = run_screener(
            ScreenerInput(
                tickers=["AAPL", "MSFT"],
                filters=PASSES_EVERYTHING,
                sort_by="price_to_book",
            )
        )
        assert result.num_passed == 2
        assert result.warnings == []

    def test_no_sort_is_input_order_and_says_nothing(self, patched_factory):
        result = run_screener(
            ScreenerInput(
                tickers=["MSFT", "AAPL", "GOOGL"],
                filters=PASSES_EVERYTHING,
                sort_by=None,
            )
        )
        assert result.tickers_passed == ["MSFT", "AAPL", "GOOGL"]
        assert result.warnings == []

    def test_a_screen_where_nothing_passes_says_it_sorted_nothing(
        self, patched_factory
    ):
        result = run_screener(
            ScreenerInput(
                tickers=["AAPL", "MSFT"],
                filters=PASSES_NOTHING,
                sort_by="forward_pe",
            )
        )
        assert result.num_passed == 0
        assert len(result.warnings) == 1
        assert "sorted nothing" in result.warnings[0]

    def test_an_empty_screen_without_a_sort_says_nothing(self, patched_factory):
        result = run_screener(
            ScreenerInput(tickers=["AAPL"], filters=PASSES_NOTHING, sort_by=None)
        )
        assert result.num_passed == 0
        assert result.warnings == []


# ── the position cap on the score path ───────────────────────────────────


CAP = 0.5


class TestTheCapRedistributesRatherThanTruncating:
    def test_a_planted_row_is_clipped_and_keeps_its_gross(self):
        frame = pd.DataFrame(
            {"AAPL": [0.9], "MSFT": [0.1]},
            index=pd.to_datetime(["2023-02-15"]),
        )
        capped, summary = _cap_constructed_weights(frame, CAP)
        assert summary["tickers"] == ["AAPL"]
        assert summary["n_dates"] == 1
        assert float(capped.abs().to_numpy().max()) <= CAP + 1e-12
        assert float(capped.abs().sum(axis=1).iloc[0]) == pytest.approx(1.0)
        assert summary["clipped_weight"] == pytest.approx(0.4)
        assert summary["shortfall"] == pytest.approx(0.0, abs=1e-12)

    def test_a_row_already_under_the_cap_is_returned_untouched(self):
        frame = pd.DataFrame(
            {"AAPL": [0.4], "MSFT": [-0.4], "GOOGL": [0.2]},
            index=pd.to_datetime(["2023-02-15"]),
        )
        capped, summary = _cap_constructed_weights(frame, CAP)
        assert summary["n_dates"] == 0
        pd.testing.assert_frame_equal(capped, frame.astype(float))

    def test_each_rebalance_date_keeps_its_own_label_and_its_own_gross(self):
        """Three dates, two of which need capping and one of which does not:
        the clipped rows must land back on their own dates."""
        dates = pd.to_datetime(["2023-01-03", "2023-02-01", "2023-03-01"])
        frame = pd.DataFrame(
            {
                "AAPL": [0.2, 0.9, -0.8],
                "MSFT": [-0.2, 0.1, 0.2],
                "GOOGL": [0.6, 0.0, 0.0],
            },
            index=dates,
        )
        capped, summary = _cap_constructed_weights(frame, CAP)
        assert list(capped.index) == list(dates)
        assert summary["tickers"] == ["AAPL", "GOOGL"]
        assert summary["n_dates"] == 3
        assert float(capped.abs().to_numpy().max()) <= CAP + 1e-12
        assert capped.abs().sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
        # The untouched name absorbs its share and nothing changes sign.
        assert capped.loc[dates[2], "AAPL"] == pytest.approx(-CAP)
        assert capped.loc[dates[2], "MSFT"] == pytest.approx(CAP)

    def test_the_sign_of_a_short_survives_the_clip(self):
        frame = pd.DataFrame(
            {"AAPL": [-0.9], "MSFT": [0.1]},
            index=pd.to_datetime(["2023-02-15"]),
        )
        capped, _ = _cap_constructed_weights(frame, CAP)
        assert capped.iloc[0]["AAPL"] == pytest.approx(-CAP)

    def test_a_cap_too_tight_for_the_gross_reports_the_shortfall(self):
        frame = pd.DataFrame(
            {"AAPL": [1.0], "MSFT": [0.0]},
            index=pd.to_datetime(["2023-02-15"]),
        )
        capped, summary = _cap_constructed_weights(frame, CAP)
        assert float(capped.abs().sum(axis=1).iloc[0]) == pytest.approx(CAP)
        assert summary["shortfall"] == pytest.approx(0.5)


@pytest.fixture
def two_volatilities(monkeypatch):
    """
    Two tickers whose trailing volatilities stand 9:1 apart, so an equal
    pair of scores converts under `vol_scaled` to exactly 0.9 / 0.1 -- a
    weight the caller never stated, above a 0.5 cap.
    """
    n = 80
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    alternating = np.array([1.0 if i % 2 else -1.0 for i in range(n)])
    frames = {}
    for ticker, size in (("AAPL", 0.001), ("MSFT", 0.009)):
        close = 100.0 * np.cumprod(1.0 + size * alternating)
        frames[ticker] = _ohlcv(pd.Series(close, index=dates))
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: frames[symbol]
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: frames[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return dates


def _panel(dates, value_a, value_b):
    rebalance = str(dates[30].date())
    return {"AAPL": {rebalance: value_a}, "MSFT": {rebalance: value_b}}


def _simulation_input(dates, signal_type, values, **overrides):
    payload = dict(
        tickers=["AAPL", "MSFT"],
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        target_weights=values,
        signal_type=signal_type,
        max_position_pct=CAP,
        initial_capital=100_000.0,
    )
    payload.update(overrides)
    return PortfolioSimulationInput(**payload)


class TestAScoreIsCappedAndAStatedWeightIsRefused:
    def test_the_run_completes_and_names_the_ticker(self, two_volatilities):
        result = run_portfolio_simulation(
            _simulation_input(
                two_volatilities,
                SignalType.SCORE,
                _panel(two_volatilities, 1.0, 1.0),
                construction_method="vol_scaled",
                gross_leverage=1.0,
            )
        )
        capped = [w for w in result.warnings if "position cap" in w]
        assert len(capped) == 1
        assert "AAPL" in capped[0]
        assert "max_position_pct=0.5" in capped[0]
        assert result.n_rebalances == 1

    def test_the_stated_weight_is_still_refused(self, two_volatilities):
        """Under signal_type='target_weight' the 0.9 is the caller's own
        statement, so it is an error rather than an over-large output."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError) as excinfo:
            _simulation_input(
                two_volatilities,
                SignalType.TARGET_WEIGHT,
                _panel(two_volatilities, 0.9, 0.1),
            )
        message = str(excinfo.value)
        assert "target_weight" in message
        assert "0.5" in message

    def test_a_score_panel_under_the_cap_produces_no_warning(self, two_volatilities):
        result = run_portfolio_simulation(
            _simulation_input(
                two_volatilities,
                SignalType.SCORE,
                _panel(two_volatilities, 1.0, 1.0),
                construction_method="rank_weighted",
                gross_leverage=1.0,
            )
        )
        assert not any("position cap" in w for w in result.warnings)


# ── one drawdown convention ──────────────────────────────────────────────


class TestTheTwoDrawdownFieldsAgree:
    @staticmethod
    def _futures(prices):
        return run_futures_backtest(
            FuturesBacktestInput(
                prices=prices,
                target_contracts={"2024-01-02": 1.0},
                multiplier=1.0,
                initial_capital=100_000.0,
            )
        )

    def test_a_planted_twenty_percent_fall_reads_the_same_on_both_doors(self):
        result = self._futures(
            {
                "2024-01-02": 100_000.0,
                "2024-01-03": 80_000.0,
                "2024-01-04": 80_000.0,
            }
        )
        stress = replay_stress_scenario(
            pd.DataFrame(
                {"AAPL": [-0.20, 0.0]},
                index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
            ),
            [1.0],
        )
        assert result.max_drawdown == pytest.approx(-0.20)
        assert result.max_drawdown == pytest.approx(stress["max_drawdown_pct"])
        assert result.max_drawdown_pct == pytest.approx(-20.0)

    def test_a_flat_path_is_zero_on_both(self):
        result = self._futures(
            {
                "2024-01-02": 100.0,
                "2024-01-03": 100.0,
                "2024-01-04": 100.0,
            }
        )
        stress = replay_stress_scenario(
            pd.DataFrame(
                {"AAPL": [0.0, 0.0]},
                index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
            ),
            [1.0],
        )
        assert result.max_drawdown == 0.0
        assert result.max_drawdown_pct == 0.0
        assert stress["max_drawdown_pct"] == 0.0

    def test_the_deprecated_field_says_which_unit_it_is(self):
        from standard_quant_tools.agent.runtimes.backtest.futures_tools import (
            FuturesBacktestResult,
        )

        description = FuturesBacktestResult.model_fields["max_drawdown_pct"].description
        assert "PERCENTAGE" in description
        assert "DEPRECATED" in description


# ── one decomposition per PCA answer ─────────────────────────────────────


@pytest.fixture
def one_dominant_asset(monkeypatch):
    """Three tickers, one of them several times more volatile than the
    others, so standardizing changes the answer."""
    rng = np.random.default_rng(5)
    n = 400
    dates = pd.date_range(START, periods=n, freq="B")
    common = rng.normal(0.0, 0.01, n)
    frames = {}
    scales = {"AAPL": 5.0, "MSFT": 1.0, "GOOGL": 0.8}
    for ticker, scale in scales.items():
        returns = scale * common + rng.normal(0.0, 0.004, n)
        close = 100.0 * np.cumprod(1.0 + returns)
        frames[ticker] = _ohlcv(pd.Series(close, index=dates))
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: frames[symbol]
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: frames[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return frames


def _returns_frame(frames):
    return pd.DataFrame(
        {t: f["Close"].pct_change(fill_method=None) for t, f in frames.items()}
    ).dropna()


class TestBothHalvesOfThePcaAnswerDescribeOneDecomposition:
    TICKERS = ["AAPL", "MSFT", "GOOGL"]

    def _run(self, standardize):
        return run_pca_analysis(
            PCAInput(
                tickers=self.TICKERS,
                start_date=START,
                end_date=END,
                n_components=2,
                standardize=standardize,
            )
        )

    def test_unstandardized_contributions_differ_from_standardized_ones(
        self, one_dominant_asset
    ):
        raw = self._run(False).factor_contributions
        scaled = self._run(True).factor_contributions
        assert raw != scaled

    def test_each_ticker_sum_is_its_r_squared_on_the_same_factor_returns(
        self, one_dominant_asset
    ):
        contributions = self._run(False).factor_contributions
        returns = _returns_frame(one_dominant_asset)[self.TICKERS]
        factors = pca_returns(returns, n_components=2, standardize=False)[
            "factor_returns"
        ].to_numpy()
        design = np.column_stack([np.ones(len(factors)), factors])
        for ticker, per_component in contributions.items():
            y = returns[ticker].to_numpy()
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = float(np.sum((y - design @ beta) ** 2))
            total = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - residual / total
            assert sum(per_component.values()) == pytest.approx(r_squared, abs=1e-3)

    def test_the_default_run_is_unchanged(self, one_dominant_asset):
        """At standardize=True the handler's decomposition and the one
        `factor_contributions` used to build for itself are the same one, so
        passing it must reproduce the previous numbers exactly."""
        returns = _returns_frame(one_dominant_asset)[self.TICKERS]
        before = factor_contributions(returns, n_components=2)
        after = factor_contributions(
            returns,
            n_components=2,
            pca_result=pca_returns(returns, n_components=2, standardize=True),
        )
        pd.testing.assert_frame_equal(before, after)


# ── the annualization is the caller's ────────────────────────────────────


WEEKLY = 52
DAILY = 252
WEEKLY_SCALE = np.sqrt(WEEKLY / DAILY)

_ESTIMATORS = [
    "close_to_close_annualized",
    "parkinson_annualized",
    "garman_klass_annualized",
    "yang_zhang_annualized",
]


class TestTheVolatilityEstimatorsTakeTheBarFrequency:
    @staticmethod
    def _run(**overrides):
        payload = dict(symbol="AAPL", start_date=START, end_date=END, period=20)
        payload.update(overrides)
        return get_volatility_estimators(VolatilityEstimatorsInput(**payload))

    def test_weekly_bars_scale_every_estimator_by_the_root_of_the_ratio(
        self, one_symbol
    ):
        daily = self._run()
        weekly = self._run(periods_per_year=WEEKLY)
        for field in _ESTIMATORS:
            assert getattr(weekly, field) == pytest.approx(
                getattr(daily, field) * WEEKLY_SCALE, rel=1e-4
            )

    def test_the_ratio_between_two_estimators_is_unchanged_by_it(self, one_symbol):
        daily = self._run()
        weekly = self._run(periods_per_year=WEEKLY)
        assert weekly.yang_zhang_vs_close_to_close_ratio == pytest.approx(
            daily.yang_zhang_vs_close_to_close_ratio, rel=1e-3
        )

    def test_the_convention_is_echoed(self, one_symbol):
        assert self._run().periods_per_year == DAILY
        assert self._run(periods_per_year=WEEKLY).periods_per_year == WEEKLY

    def test_the_default_reproduces_todays_numbers(self, one_symbol):
        """252 was hardcoded, so the default has to be the identical run."""
        explicit = self._run(periods_per_year=DAILY)
        default = self._run()
        for field in _ESTIMATORS:
            assert getattr(default, field) == getattr(explicit, field)

    def test_a_zero_frequency_is_refused(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            VolatilityEstimatorsInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                periods_per_year=0,
            )


class TestHierarchicalRiskParityTakesTheBarFrequency:
    @staticmethod
    def _returns():
        rng = np.random.default_rng(3)
        common = rng.normal(0.0, 0.01, 260)
        return {
            name: (scale * common + rng.normal(0.0, 0.005, 260)).tolist()
            for name, scale in (("AAPL", 1.0), ("MSFT", 0.8), ("GOOGL", 1.4))
        }

    def test_the_volatility_scales_and_the_weights_do_not(self):
        returns = self._returns()
        daily = optimize_hierarchical_risk_parity(HRPInput(returns=returns))
        weekly = optimize_hierarchical_risk_parity(
            HRPInput(returns=returns, periods_per_year=WEEKLY)
        )
        assert weekly.portfolio_volatility == pytest.approx(
            daily.portfolio_volatility * WEEKLY_SCALE, rel=1e-9
        )
        assert weekly.weights == daily.weights

    def test_the_risk_contributions_move_with_it(self):
        returns = self._returns()
        daily = optimize_hierarchical_risk_parity(HRPInput(returns=returns))
        weekly = optimize_hierarchical_risk_parity(
            HRPInput(returns=returns, periods_per_year=WEEKLY)
        )
        for name, value in daily.risk_contributions.items():
            assert weekly.risk_contributions[name] == pytest.approx(
                value * WEEKLY_SCALE, rel=1e-9
            )

    def test_the_convention_is_echoed(self):
        result = optimize_hierarchical_risk_parity(
            HRPInput(returns=self._returns(), periods_per_year=WEEKLY)
        )
        assert result.periods_per_year == WEEKLY

    def test_the_default_reproduces_todays_numbers(self):
        returns = self._returns()
        default = optimize_hierarchical_risk_parity(HRPInput(returns=returns))
        explicit = optimize_hierarchical_risk_parity(
            HRPInput(returns=returns, periods_per_year=DAILY)
        )
        assert default.portfolio_volatility == explicit.portfolio_volatility
        assert default.risk_contributions == explicit.risk_contributions
