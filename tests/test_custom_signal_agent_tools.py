"""
Tests for the bring-your-own-signal agent tools: run_custom_signal_backtest
and run_signal_panel_backtest. Both accept a signal computed entirely outside
this library and only backtest/combine it.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from standard_quant_tools.agent.models import (
    CustomSignalBacktestInput, SignalPanelBacktestInput, SignalType,
)
from standard_quant_tools.agent.tools import (
    get_agent_tools, dispatch,
    run_custom_signal_backtest, run_signal_panel_backtest,
)
from standard_quant_tools.data.factory import DataFactory

START, END = "2023-01-01", "2024-01-01"


def _toy_signal(df: pd.DataFrame) -> dict:
    """Deterministic {date: value} signal built from the mocked OHLCV."""
    sig = (df["Close"].pct_change(5) > 0).astype(int)
    return {str(d.date()): float(v) for d, v in sig.items()}


@pytest.fixture
def gapped_factory(sample_close, monkeypatch: pytest.MonkeyPatch):
    """
    Like patched_factory, but Open has a genuine intraday gap from the same
    bar's Close (Open = 0.999 * Close), unlike patched_factory's Open =
    Close.shift(1), which makes fill_price="next_open" collapse to "close"
    by construction (no gap to price differently) — see test_fill_price.py.
    """
    from unittest.mock import MagicMock
    close = sample_close
    spread = pd.Series(0.5, index=close.index)
    df = pd.DataFrame({
        "Open": close * 0.999,
        "High": close + spread,
        "Low": close - spread,
        "Close": close,
        "Volume": pd.Series(1_000_000.0, index=close.index),
    })
    provider = MagicMock()
    provider.get_ohlcv.return_value = df
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


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

    def test_next_open_fill_price_differs_from_default(self, gapped_factory):
        df = gapped_factory.get_ohlcv("AAPL", START, END)
        signal = _toy_signal(df)
        base = run_custom_signal_backtest(
            CustomSignalBacktestInput(symbol="AAPL", start_date=START, end_date=END, signals=signal)
        )
        next_open = run_custom_signal_backtest(
            CustomSignalBacktestInput(
                symbol="AAPL", start_date=START, end_date=END, signals=signal, fill_price="next_open",
            )
        )
        assert base.final_equity != pytest.approx(next_open.final_equity, abs=1e-6)


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

    def test_next_open_fill_price_differs_from_default(self, gapped_factory):
        tickers = ["AAPL", "MSFT"]
        df = gapped_factory.get_ohlcv("AAPL", START, END)
        panel = {t: _toy_signal(df) for t in tickers}

        base = run_signal_panel_backtest(
            SignalPanelBacktestInput(tickers=tickers, start_date=START, end_date=END, signal_panel=panel)
        )
        next_open = run_signal_panel_backtest(
            SignalPanelBacktestInput(
                tickers=tickers, start_date=START, end_date=END, signal_panel=panel, fill_price="next_open",
            )
        )
        assert base.portfolio_metrics["total_return"] != pytest.approx(
            next_open.portfolio_metrics["total_return"], abs=1e-9
        )


class TestSignalTypeValidation:
    """
    SignalType is opt-in: the default (SCORE) is unrestricted and must be
    byte-for-byte the permissive behavior every existing caller already
    gets — these tests exist specifically to lock that backward-compat
    guarantee in, alongside the new opt-in DIRECTION/TARGET_WEIGHT checks.
    """

    def test_default_signal_type_is_score_and_unrestricted(self):
        # 2.5 is out of range for direction/target_weight but must be
        # accepted silently under the default — this is today's exact
        # pre-existing permissive behavior, unchanged.
        inp = CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals={"2023-01-03": 2.5, "2023-01-04": -3.7},
        )
        assert inp.signal_type == SignalType.SCORE

    def test_score_accepts_anything(self):
        CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals={"2023-01-03": 100.0}, signal_type=SignalType.SCORE,
        )

    def test_direction_rejects_out_of_range_value(self):
        with pytest.raises(ValidationError, match="direction"):
            CustomSignalBacktestInput(
                symbol="AAPL", start_date=START, end_date=END,
                signals={"2023-01-03": 0.5}, signal_type=SignalType.DIRECTION,
            )

    def test_direction_accepts_exact_values(self):
        CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals={"2023-01-03": 1.0, "2023-01-04": -1.0, "2023-01-05": 0.0},
            signal_type=SignalType.DIRECTION,
        )

    def test_target_weight_rejects_over_bound(self):
        with pytest.raises(ValidationError, match="target_weight"):
            CustomSignalBacktestInput(
                symbol="AAPL", start_date=START, end_date=END,
                signals={"2023-01-03": 1.5}, signal_type=SignalType.TARGET_WEIGHT,
                max_abs_weight=1.0,
            )

    def test_target_weight_accepts_within_bound(self):
        CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals={"2023-01-03": 0.8}, signal_type=SignalType.TARGET_WEIGHT,
            max_abs_weight=1.0,
        )

    def test_target_weight_respects_custom_max_abs_weight(self):
        CustomSignalBacktestInput(
            symbol="AAPL", start_date=START, end_date=END,
            signals={"2023-01-03": 1.8}, signal_type=SignalType.TARGET_WEIGHT,
            max_abs_weight=2.0,
        )
        with pytest.raises(ValidationError, match="target_weight"):
            CustomSignalBacktestInput(
                symbol="AAPL", start_date=START, end_date=END,
                signals={"2023-01-03": 1.8}, signal_type=SignalType.TARGET_WEIGHT,
                max_abs_weight=1.0,
            )

    def test_signal_panel_default_is_score_and_unrestricted(self):
        SignalPanelBacktestInput(
            tickers=["AAPL"], start_date=START, end_date=END,
            signal_panel={"AAPL": {"2023-01-03": 5.0}},
        )

    def test_signal_panel_direction_rejects_out_of_range_and_names_ticker(self):
        with pytest.raises(ValidationError, match="MSFT"):
            SignalPanelBacktestInput(
                tickers=["AAPL", "MSFT"], start_date=START, end_date=END,
                signal_panel={
                    "AAPL": {"2023-01-03": 1.0},
                    "MSFT": {"2023-01-03": 0.5},
                },
                signal_type=SignalType.DIRECTION,
            )

    def test_signal_panel_target_weight_rejects_over_bound(self):
        with pytest.raises(ValidationError, match="target_weight"):
            SignalPanelBacktestInput(
                tickers=["AAPL"], start_date=START, end_date=END,
                signal_panel={"AAPL": {"2023-01-03": 3.0}},
                signal_type=SignalType.TARGET_WEIGHT, max_abs_weight=1.0,
            )
