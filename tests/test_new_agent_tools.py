"""Tests for the 5 new agentic tools (mocked data provider)."""

import pandas as pd
import numpy as np
import pytest

from standard_quant_tools.agent.models import (
    RegimeAdaptiveInput,
    PairScannerInput,
    WalkForwardInput,
    RiskAttributionInput,
    PositionSizerInput,
)
from standard_quant_tools.agent.tools import (
    get_agent_tools,
    run_regime_adaptive_backtest,
    scan_pairs,
    run_walk_forward_backtest,
    get_portfolio_risk_attribution,
    get_position_size,
)

START, END = "2022-01-01", "2024-01-01"


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def long_ohlcv() -> pd.DataFrame:
    """700-bar OHLCV — enough for walk-forward (252 train + 63 test × 2 windows)."""
    np.random.seed(99)
    n = 700
    returns = np.random.normal(0.0003, 0.013, n)
    close = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    spread = np.random.uniform(0.2, 1.2, n)
    return pd.DataFrame(
        {
            "Open": pd.Series(close * 0.999, index=dates),
            "High": pd.Series(close + spread, index=dates),
            "Low": pd.Series(close - spread, index=dates),
            "Close": pd.Series(close, index=dates),
            "Volume": pd.Series(np.random.randint(1_000_000, 5_000_000, n).astype(float), index=dates),
        }
    )


@pytest.fixture
def patched_long(long_ohlcv, monkeypatch):
    """Patch DataFactory to return the long OHLCV for every symbol."""
    from unittest.mock import AsyncMock, MagicMock
    from standard_quant_tools.data.factory import DataFactory

    provider = MagicMock()
    provider.get_ohlcv.return_value = long_ohlcv
    provider.get_ohlcv_async = AsyncMock(return_value=long_ohlcv)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


# ── Tool registry ──────────────────────────────────────────────────────────────

class TestToolRegistry:
    def test_now_has_seventeen_tools(self):
        assert len(get_agent_tools()) == 17

    def test_new_tool_names_present(self):
        names = {t["function"]["name"] for t in get_agent_tools()}
        assert "run_regime_adaptive_backtest" in names
        assert "scan_pairs" in names
        assert "run_walk_forward_backtest" in names
        assert "get_portfolio_risk_attribution" in names
        assert "get_position_size" in names

    def test_all_new_tools_have_valid_schema(self):
        new_tools = [
            t for t in get_agent_tools()
            if t["function"]["name"] in {
                "run_regime_adaptive_backtest", "scan_pairs",
                "run_walk_forward_backtest", "get_portfolio_risk_attribution",
                "get_position_size",
            }
        ]
        for tool in new_tools:
            schema = tool["function"]["parameters"]
            assert schema.get("type") == "object"
            assert "properties" in schema


# ── Feature 1: Regime-Adaptive Strategy Selector ──────────────────────────────

class TestRegimeAdaptiveBacktest:
    def test_returns_result(self, patched_long):
        inp = RegimeAdaptiveInput(
            symbol="AAPL", start_date=START, end_date=END,
            n_workers=1,
        )
        result = run_regime_adaptive_backtest(inp)
        assert result.symbol == "AAPL"

    def test_regime_is_valid(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.regime in ("trending", "mean_reverting", "random_walk")

    def test_selected_strategy_is_valid(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.selected_strategy in (
            "sma_crossover", "rsi_mean_reversion", "macd_crossover", "bollinger_reversion"
        )

    def test_hurst_bounded(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert 0.0 <= result.hurst <= 1.0 or result.hurst == 0.0  # 0.0 on NaN fallback

    def test_backtest_fields_populated(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert isinstance(result.backtest.sharpe_ratio, float)
        assert result.backtest.final_equity > 0
        assert 0.0 <= result.backtest.win_rate <= 1.0

    def test_grid_combinations_positive(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.grid_combinations > 0

    def test_best_parameters_nonempty(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert len(result.best_parameters) > 0

    def test_custom_param_grid_respected(self, patched_long):
        inp = RegimeAdaptiveInput(
            symbol="AAPL", start_date=START, end_date=END,
            sma_param_grid={"fast_period": [5], "slow_period": [20]},
        )
        result = run_regime_adaptive_backtest(inp)
        if result.selected_strategy == "sma_crossover":
            assert result.grid_combinations == 1


# ── Feature 2: Pair Scanner ────────────────────────────────────────────────────

class TestScanPairs:
    def test_returns_result(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START, end_date=END,
        )
        result = scan_pairs(inp)
        assert result.n_pairs_tested >= 0

    def test_result_structure(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START, end_date=END,
        )
        result = scan_pairs(inp)
        assert isinstance(result.n_pairs_tested, int)
        assert isinstance(result.n_pairs_cointegrated, int)
        assert isinstance(result.pairs, list)

    def test_pairs_respect_max_pairs(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL", "TSLA"],
            start_date=START, end_date=END,
            max_pairs=2,
        )
        result = scan_pairs(inp)
        assert result.n_pairs_returned <= 2
        assert len(result.pairs) <= 2

    def test_pair_fields_populated(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START, end_date=END,
        )
        result = scan_pairs(inp)
        for pair in result.pairs:
            assert pair.symbol_a != pair.symbol_b
            assert 0.0 <= pair.p_value <= 1.0
            assert pair.half_life_days > 0
            assert pair.signal in ("long_a_short_b", "short_a_long_b", "neutral")

    def test_n_pairs_tested_equals_combinations(self, patched_long):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        inp = PairScannerInput(tickers=tickers, start_date=START, end_date=END)
        result = scan_pairs(inp)
        # C(3,2) = 3 combinations
        assert result.n_pairs_tested == 3

    def test_pairs_sorted_by_half_life(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL", "TSLA"],
            start_date=START, end_date=END,
        )
        result = scan_pairs(inp)
        hls = [p.half_life_days for p in result.pairs]
        assert hls == sorted(hls)

    def test_single_ticker_returns_zero_pairs(self, patched_long):
        inp = PairScannerInput(tickers=["AAPL"], start_date=START, end_date=END)
        result = scan_pairs(inp)
        assert result.n_pairs_tested == 0
        assert len(result.pairs) == 0


# ── Feature 3: Walk-Forward Backtest ──────────────────────────────────────────

class TestWalkForwardBacktest:
    def test_returns_result(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252, test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert result.symbol == "AAPL"
        assert result.strategy == "sma_crossover"

    def test_windows_populated(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252, test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert result.n_windows >= 1
        assert len(result.windows) == result.n_windows

    def test_window_dates_sequential(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252, test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        for i, win in enumerate(result.windows):
            assert win.window_index == i
            assert win.train_start <= win.train_end
            assert win.test_start <= win.test_end
            assert win.train_end < win.test_start

    def test_aggregate_metrics_are_floats(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252, test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert isinstance(result.avg_oos_sharpe, float)
        assert isinstance(result.avg_oos_return, float)
        assert isinstance(result.avg_oos_max_drawdown, float)
        assert 0.0 <= result.pct_windows_profitable <= 1.0

    def test_param_stability_has_correct_keys(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252, test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert "fast_period" in result.param_stability
        assert "slow_period" in result.param_stability
        for key, info in result.param_stability.items():
            assert "most_common" in info
            assert "frequency" in info
            assert 0.0 <= info["frequency"] <= 1.0

    def test_invalid_strategy_raises(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="nonexistent_strategy",
            param_grid={"fast_period": [5]},
            train_bars=252, test_bars=63,
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_walk_forward_backtest(inp)

    def test_insufficient_data_raises(self, patched_long, long_ohlcv, monkeypatch):
        from unittest.mock import MagicMock
        from standard_quant_tools.data.factory import DataFactory

        tiny_df = long_ohlcv.iloc[:50]
        prov = MagicMock()
        prov.get_ohlcv.return_value = tiny_df
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: prov)

        inp = WalkForwardInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5], "slow_period": [30]},
            train_bars=252, test_bars=63,
        )
        with pytest.raises(ValueError, match="Not enough data"):
            run_walk_forward_backtest(inp)


# ── Feature 4: Portfolio Risk Attribution ─────────────────────────────────────

class TestPortfolioRiskAttribution:
    def test_returns_result(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_portfolio_metrics_are_floats(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        for field in ("annualized_return", "annualized_volatility", "sharpe_ratio",
                      "sortino_ratio", "max_drawdown", "var_95", "cvar_95", "information_ratio"):
            assert isinstance(getattr(result, field), float), f"{field} not float"

    def test_risk_contributions_sum_to_one(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        total = sum(result.asset_risk_contributions.values())
        assert abs(total - 1.0) < 1e-4, f"MCR sum {total} != 1.0"

    def test_risk_contributions_have_correct_keys(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert set(result.asset_risk_contributions.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_pca_keys_correct(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START, end_date=END,
            n_components=2,
        )
        result = get_portfolio_risk_attribution(inp)
        assert set(result.pca_variance_explained.keys()) == {"PC1", "PC2"}
        assert set(result.portfolio_pc_exposures.keys()) == {"PC1", "PC2"}

    def test_factor_fields_none_without_input(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.factor_loadings is None
        assert result.factor_r_squared is None
        assert result.factor_alpha is None

    def test_factor_fields_populated_with_input(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START, end_date=END,
            factor_tickers=["SPY"],
            factor_names=["mkt"],
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.factor_loadings is not None
        assert "mkt" in result.factor_loadings
        assert result.factor_r_squared is not None
        assert result.factor_alpha is not None

    def test_max_drawdown_non_positive(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START, end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.max_drawdown <= 0.0


# ── Feature 5: Position Sizer ──────────────────────────────────────────────────

class TestPositionSizer:
    def test_returns_result(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.symbol == "AAPL"

    def test_last_close_positive(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.last_close > 0

    def test_atr_positive(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.atr > 0

    def test_fixed_risk_sizing_math(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
            atr_multiplier=2.0,
        )
        result = get_position_size(inp)
        expected_dollar_risk = 100_000.0 * 0.01
        expected_stop = result.atr * 2.0
        expected_shares = int(expected_dollar_risk / expected_stop)
        assert result.shares_fixed_risk == expected_shares

    def test_max_loss_bounded_by_risk_pct(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )
        result = get_position_size(inp)
        assert result.max_loss_fixed_risk <= 100_000.0 * 0.01 + 1  # rounding tolerance

    def test_without_kelly_inputs_recommendation_is_fixed_risk(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.recommended_sizing == "fixed_risk"
        assert result.kelly_fraction is None

    def test_with_kelly_inputs_populated(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
            win_rate=0.55, avg_win_pct=0.05, avg_loss_pct=0.03,
        )
        result = get_position_size(inp)
        assert result.kelly_fraction is not None
        assert result.kelly_fraction >= 0.0
        assert result.shares_half_kelly is not None
        assert result.recommended_sizing in ("half_kelly", "fixed_risk")

    def test_negative_edge_falls_back_to_fixed_risk(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=100_000.0,
            win_rate=0.30, avg_win_pct=0.02, avg_loss_pct=0.05,
        )
        result = get_position_size(inp)
        assert result.kelly_fraction == 0.0
        assert result.recommended_sizing == "fixed_risk"

    def test_recommended_shares_is_int(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL", start_date=START, end_date=END,
            account_equity=50_000.0,
            win_rate=0.6, avg_win_pct=0.06, avg_loss_pct=0.03,
        )
        result = get_position_size(inp)
        assert isinstance(result.recommended_shares, int)
        assert result.recommended_shares >= 0
