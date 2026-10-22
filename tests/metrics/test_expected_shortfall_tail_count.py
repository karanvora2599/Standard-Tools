"""
An expected shortfall travels with the number of returns it is the mean of.

`cvar` averages the returns at or below the empirical VaR quantile. Its
estimator is kept, and on a short sample that is very few points: a year of
daily returns at 99% leaves three, so `get_extended_risk_metrics.cvar_99`
was the mean of the three worst days, reported to six decimals beside
nothing that said so. Since the CHANGELOG entry of 2026-10-22 every result
that reports an expected shortfall reports its tail count, and warns below
CVAR_MIN_TAIL_OBSERVATIONS.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.analysis.inference import bootstrap_statistic
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.metrics import (
    CVAR_MIN_TAIL_OBSERVATIONS,
    cvar,
    cvar_tail_observations,
)

# 252 distinct daily returns: a seeded shuffle of an even grid, so every
# quantile and every tail is known exactly.
YEAR = np.random.default_rng(7).permutation(np.linspace(-0.05, 0.05, 252))
WORST_THREE = np.sort(YEAR)[:3]


def _ohlcv(returns):
    close = 100.0 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(returns)])
    index = pd.date_range("2022-01-03", periods=len(close), freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.full(len(close), 1e6),
        },
        index=index,
    )


@pytest.fixture
def one_year(monkeypatch):
    frames = {
        "ONE": _ohlcv(YEAR),
        # Distinct from ONE by a wiggle too small to reorder the grid much
        # and too irregular to tie, so a 50/50 book has 252 distinct days.
        "TWO": _ohlcv(YEAR + 1e-4 * np.sin(np.arange(len(YEAR)))),
        "SPY": _ohlcv(0.0005 + 0.004 * np.sin(np.arange(len(YEAR)))),
    }

    def _by_symbol(symbol, *args, **kwargs):
        return frames[str(symbol)].copy()

    provider = MagicMock()
    provider.get_ohlcv.side_effect = _by_symbol
    provider.get_ohlcv_async = AsyncMock(side_effect=_by_symbol)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)


class TestTheCount:
    @pytest.mark.parametrize(
        "confidence, expected", [(0.99, 3), (0.95, 13), (0.90, 26)]
    )
    def test_a_year_of_daily_returns(self, confidence, expected):
        """The 1% quantile of 252 points sits between the third and fourth
        worst, so three returns are at or below it."""
        assert cvar_tail_observations(pd.Series(YEAR), confidence) == expected

    def test_it_counts_the_points_cvar_averages(self):
        """The estimator is unchanged: the mean loss of exactly those
        points."""
        assert cvar(pd.Series(YEAR), 0.99) == pytest.approx(
            -WORST_THREE.mean(), abs=1e-15
        )

    def test_the_floor(self):
        assert CVAR_MIN_TAIL_OBSERVATIONS == 10


class TestEveryDoorReportsIt:
    def test_extended_risk_metrics_at_99(self, one_year):
        """The door the count was written for: three points, and a warning
        that says three."""
        out = dispatch(
            "get_extended_risk_metrics",
            {
                "symbol": "ONE",
                "benchmark": "SPY",
                "start_date": "2022-01-01",
                "end_date": "2023-12-31",
            },
        )
        assert out["cvar_99"] == pytest.approx(-WORST_THREE.mean(), abs=1e-6)
        assert out["cvar_99_tail_observations"] == 3
        assert any("worst 3 of 252" in w for w in out["warnings"])

    def test_analyze_stock_risk_at_95_is_above_the_floor(self, one_year):
        out = dispatch(
            "analyze_stock_risk", {"symbol": "ONE", "benchmark": "SPY", "period": "1y"}
        )
        assert out["cvar_95_tail_observations"] == 13
        assert not any("cvar_95" in w for w in out["warnings"])

    def test_the_portfolio_doors(self, one_year):
        book = {
            "tickers": ["ONE", "TWO"],
            "weights": [0.5, 0.5],
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "benchmark": "SPY",
        }
        analysis = dispatch("get_portfolio_analysis", book)
        attribution = dispatch(
            "get_portfolio_risk_attribution", {**book, "n_components": 1}
        )
        for out in (analysis, attribution):
            assert out["cvar_95_tail_observations"] == 13

    def test_series_metrics_warns_on_a_short_series(self):
        """Thirty returns at 95%: the quantile falls between the second and
        third worst, so the shortfall is the mean of two."""
        out = dispatch(
            "calculate_series_metrics",
            {"series": {"values": list(YEAR[:30])}, "metrics": ["cvar"]},
        )
        assert out["cvar_tail_observations"] == 2
        assert any("worst 2 of 30" in w for w in out["warnings"])

    def test_series_metrics_reports_no_count_unasked(self):
        out = dispatch(
            "calculate_series_metrics",
            {"series": {"values": list(YEAR)}, "metrics": ["sharpe_ratio"]},
        )
        assert out["cvar_tail_observations"] is None

    def test_the_bootstrap_of_a_shortfall_warns_on_a_short_series(self):
        out = bootstrap_statistic(list(YEAR[:40]), statistic="cvar_95", n_bootstrap=200)
        assert any("worst 2 of 40" in w for w in out["warnings"])
