"""
Tests for run_signal_panel_backtest: bring-your-own signal matrix across a
ticker universe, combined into portfolio-level metrics.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest import run_signal_panel_backtest, run_strategy
from standard_quant_tools.portfolio import build_portfolio, portfolio_metrics
from standard_quant_tools.error import ValidationError


def _make_ohlcv(seed: int, n: int = 150) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * (1 + pd.Series(rng.normal(0, 0.012, n), index=dates)).cumprod()
    return pd.DataFrame({
        "Open": close, "High": close * 1.001, "Low": close * 0.999,
        "Close": close, "Volume": 1_000_000.0,
    })


@pytest.fixture(scope="module")
def universe():
    tickers = ["AAA", "BBB", "CCC"]
    price_data = {t: _make_ohlcv(seed=i) for i, t in enumerate(tickers)}
    signal_panel = pd.DataFrame({
        t: (price_data[t]["Close"].pct_change(5) > 0).astype(int)
        for t in tickers
    })
    return tickers, price_data, signal_panel


class TestSignalPanelBacktest:
    def test_output_structure(self, universe):
        tickers, price_data, signal_panel = universe
        result = run_signal_panel_backtest(price_data, signal_panel)
        assert set(result.keys()) == {
            "tickers", "per_ticker", "portfolio_returns", "portfolio_metrics"
        }
        assert result["tickers"] == tickers
        assert set(result["per_ticker"]) == set(tickers)
        assert isinstance(result["portfolio_returns"], pd.Series)
        assert isinstance(result["portfolio_metrics"], dict)

    def test_per_ticker_matches_direct_run_strategy(self, universe):
        tickers, price_data, signal_panel = universe
        result = run_signal_panel_backtest(price_data, signal_panel)
        for t in tickers:
            manual = run_strategy(price_data[t], signal_panel[t])
            panel_r = result["per_ticker"][t]
            assert panel_r["sharpe_ratio"] == pytest.approx(manual["sharpe_ratio"], abs=1e-9)
            assert panel_r["total_return"] == pytest.approx(manual["total_return"], abs=1e-9)

    def test_portfolio_metrics_match_manual_reconstruction(self, universe):
        tickers, price_data, signal_panel = universe
        weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
        result = run_signal_panel_backtest(price_data, signal_panel, weights=weights)

        manual_returns = pd.DataFrame({
            t: run_strategy(price_data[t], signal_panel[t])["equity_curve"].pct_change().fillna(0.0)
            for t in tickers
        }).dropna(how="any")
        w = [0.5, 0.3, 0.2]
        manual_metrics = portfolio_metrics(manual_returns, w)
        manual_port_returns = build_portfolio(manual_returns, w)

        assert result["portfolio_metrics"]["sharpe_ratio"] == pytest.approx(
            manual_metrics["sharpe_ratio"], abs=1e-9
        )
        pd.testing.assert_series_equal(
            result["portfolio_returns"].sort_index(),
            manual_port_returns.sort_index(),
            check_names=False,
        )

    def test_default_weights_are_equal(self, universe):
        tickers, price_data, signal_panel = universe
        result = run_signal_panel_backtest(price_data, signal_panel)
        assert result["portfolio_metrics"]["weights"] == pytest.approx(
            [1 / 3, 1 / 3, 1 / 3]
        )

    def test_weights_list_order_matches_columns(self, universe):
        tickers, price_data, signal_panel = universe
        result = run_signal_panel_backtest(
            price_data, signal_panel, weights=[0.5, 0.3, 0.2]
        )
        assert result["portfolio_metrics"]["weights"] == [0.5, 0.3, 0.2]

    def test_missing_ticker_raises_validation_error(self, universe):
        tickers, price_data, signal_panel = universe
        partial_price_data = {tickers[0]: price_data[tickers[0]]}
        with pytest.raises(ValidationError, match="BBB"):
            run_signal_panel_backtest(partial_price_data, signal_panel)

    def test_include_trade_log_propagates_per_ticker(self, universe):
        tickers, price_data, signal_panel = universe
        result = run_signal_panel_backtest(
            price_data, signal_panel, include_trade_log=True
        )
        for t in tickers:
            assert "trade_log" in result["per_ticker"][t]
