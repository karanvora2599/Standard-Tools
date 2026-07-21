"""
Tests for the bring-your-own-signal agent tools: run_custom_signal_backtest
and run_signal_panel_backtest. Both accept a signal computed entirely outside
this library and only backtest/combine it.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from standard_quant_tools.agent.models import (
    CustomSignalBacktestInput, SignalPanelBacktestInput,
)
from standard_quant_tools.agent.tools import (
    get_agent_tools, dispatch,
    run_custom_signal_backtest, run_signal_panel_backtest,
)

START, END = "2023-01-01", "2024-01-01"


def _toy_signal(df: pd.DataFrame) -> dict:
    """Deterministic {date: value} signal built from the mocked OHLCV."""
    sig = (df["Close"].pct_change(5) > 0).astype(int)
    return {str(d.date()): float(v) for d, v in sig.items()}


class TestToolRegistration:
    def test_both_tools_registered(self):
        names = {t["function"]["name"] for t in get_agent_tools()}
        assert "run_custom_signal_backtest" in names
        assert "run_signal_panel_backtest" in names


class TestCustomSignalBacktest:
    def test_returns_backtest_result(self, patched_factory):
        df = patched_factory.get_ohlcv("AAPL", START, END)
        inp = CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals=_toy_signal(df),
        )
        result = run_custom_signal_backtest(inp)
        assert result.total_return is not None
        assert result.num_trades >= 0

    def test_dispatch_returns_plain_dict(self, patched_factory):
        df = patched_factory.get_ohlcv("AAPL", START, END)
        result = dispatch("run_custom_signal_backtest", {
            "symbol": "AAPL", "start_date": START, "end_date": END,
            "signals": _toy_signal(df),
        })
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result

    def test_all_flat_signal_produces_no_trades(self, patched_factory):
        df = patched_factory.get_ohlcv("AAPL", START, END)
        flat_signal = {str(d.date()): 0.0 for d in df.index}
        inp = CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END, signals=flat_signal,
        )
        result = run_custom_signal_backtest(inp)
        assert result.num_trades == 0
        assert result.total_return == 0.0


class TestSignalPanelBacktestTool:
    def test_returns_result_with_all_tickers(self, patched_factory):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        df = patched_factory.get_ohlcv("AAPL", START, END)
        panel = {t: _toy_signal(df) for t in tickers}

        inp = SignalPanelBacktestInput(
            tickers=tickers, start_date=START, end_date=END,
            signal_panel=panel,
        )
        result = run_signal_panel_backtest(inp)
        assert result.tickers == tickers
        assert set(result.per_ticker.keys()) == set(tickers)
        assert "sharpe_ratio" in result.portfolio_metrics

    def test_dispatch_returns_plain_dict(self, patched_factory):
        tickers = ["AAPL", "MSFT"]
        df = patched_factory.get_ohlcv("AAPL", START, END)
        panel = {t: _toy_signal(df) for t in tickers}

        result = dispatch("run_signal_panel_backtest", {
            "tickers": tickers, "start_date": START, "end_date": END,
            "signal_panel": panel,
        })
        assert isinstance(result, dict)
        assert set(result["per_ticker"].keys()) == set(tickers)

    def test_custom_weights_applied(self, patched_factory):
        tickers = ["AAPL", "MSFT"]
        df = patched_factory.get_ohlcv("AAPL", START, END)
        panel = {t: _toy_signal(df) for t in tickers}

        inp = SignalPanelBacktestInput(
            tickers=tickers, start_date=START, end_date=END,
            signal_panel=panel, weights={"AAPL": 0.7, "MSFT": 0.3},
        )
        result = run_signal_panel_backtest(inp)
        assert result.portfolio_metrics["weights"] == pytest.approx([0.7, 0.3])

    def test_missing_ticker_in_panel_raises_validation_error(self):
        with pytest.raises(ValidationError, match="MSFT"):
            SignalPanelBacktestInput(
                tickers=["AAPL", "MSFT"], start_date=START, end_date=END,
                signal_panel={"AAPL": {"2023-01-03": 1.0}},
            )

    def test_weights_not_summing_to_one_raises_validation_error(self):
        with pytest.raises(ValidationError, match="sum to 1"):
            SignalPanelBacktestInput(
                tickers=["AAPL", "MSFT"], start_date=START, end_date=END,
                signal_panel={"AAPL": {"2023-01-03": 1.0}, "MSFT": {"2023-01-03": 0.0}},
                weights={"AAPL": 0.5, "MSFT": 0.6},
            )

    def test_weights_keys_must_match_tickers(self):
        with pytest.raises(ValidationError, match="weights keys"):
            SignalPanelBacktestInput(
                tickers=["AAPL", "MSFT"], start_date=START, end_date=END,
                signal_panel={"AAPL": {"2023-01-03": 1.0}, "MSFT": {"2023-01-03": 0.0}},
                weights={"AAPL": 0.5, "GOOGL": 0.5},
            )
