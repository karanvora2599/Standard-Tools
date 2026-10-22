"""
A stress replay that leaves a ticker out says how much of the book it left.

`run_stress_test` replays only the tickers with data for the window and
scales their weights up to stand in for the rest. That is the right choice --
booking a missing ticker at 0% would understate any crash -- but it replays a
different portfolio from the one asked about, and before the CHANGELOG entry
of 2026-10-22 the result said so only by listing the ticker's name. A book
40% in a ticker with no data reported the other 60%'s loss as its own.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import StressTestInput
from standard_quant_tools.agent.runtimes.portfolio.tools import run_stress_test
from standard_quant_tools.data.factory import DataFactory

DAYS = pd.date_range("2020-02-18", periods=6, freq="B")


def _frame(returns, index=DAYS):
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns)])
    return pd.DataFrame({"Close": close}, index=index[: len(close)])


@pytest.fixture
def market(monkeypatch):
    """OLD has the whole window, NEW lists two days in, GONE has nothing."""
    frames = {
        "OLD": _frame([-0.10, 0.02, 0.01, -0.03, 0.0]),
        "NEW": _frame([0.01, -0.02, 0.0], index=DAYS[2:]),
    }

    def _by_symbol(symbol, *args, **kwargs):
        if symbol not in frames:
            raise KeyError(f"no data for {symbol}")
        return frames[symbol].copy()

    provider = MagicMock()
    provider.get_ohlcv.side_effect = _by_symbol
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)


def _replay(tickers, weights):
    return run_stress_test(
        StressTestInput(
            tickers=tickers,
            weights=weights,
            scenario="custom",
            custom_start_date="2020-02-18",
            custom_end_date="2020-02-25",
        )
    )


class TestCoverage:
    def test_a_missing_ticker_is_named_with_its_weight(self, market):
        """GONE held 40%; the replay is the other 60%, scaled to 100%."""
        result = _replay(["OLD", "GONE"], [0.6, 0.4])
        assert result.tickers_missing_data == ["GONE"]
        assert result.weight_coverage == pytest.approx(0.6)
        assert result.portfolio_return_pct == pytest.approx(
            0.90 * 1.02 * 1.01 * 0.97 - 1.0, abs=1e-6
        )
        assert any(
            "GONE (40.0%)" in w and "60.0%" in w for w in result.warnings
        ), result.warnings

    def test_a_full_book_says_nothing(self, market):
        result = _replay(["OLD"], [1.0])
        assert result.weight_coverage == pytest.approx(1.0)
        assert result.warnings == []

    def test_a_short_leg_counts_by_its_size(self, market):
        """Gross weight: a 50% short with no data is half of a 100/-50
        book's 150% gross, not a negative share of it."""
        result = _replay(["OLD", "GONE"], [1.0, -0.5])
        assert result.weight_coverage == pytest.approx(1.0 / 1.5)

    def test_days_one_ticker_lacks_are_reported(self, market):
        """NEW listed two days into the window, so the replay starts there
        and the -10% first day is not in it."""
        result = _replay(["OLD", "NEW"], [0.5, 0.5])
        assert result.weight_coverage == pytest.approx(1.0)
        assert result.n_trading_days == 3
        assert any(
            "2 of 5 days" in w and "NEW" in w for w in result.warnings
        ), result.warnings
