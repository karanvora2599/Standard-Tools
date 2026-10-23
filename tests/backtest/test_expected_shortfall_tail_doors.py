"""
The two backtest doors that reported an expected shortfall without its tail
count now report it.

`cvar_95` is the mean of the returns at or below the 5% quantile, and on a
short run that is a handful of days: the five other doors that report one
say how many and warn below ten (see the CHANGELOG entry on growth rates and
tail counts). `run_portfolio_simulation` and `run_backtest_compact` said
nothing, so a shortfall over four days read like one over a tail.

Each run below is planted so the count is known before it runs: one asset
held whole at zero cost, so the book's returns are the asset's, with four
losing days below every other return. The 5% quantile of n returns sits at
position 0.05 x (n - 1), so 61 returns put it at the fourth lowest (the
tail is the four) and 401 distinct returns at the 21st lowest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import BacktestCompactInput
from standard_quant_tools.agent.runtimes.backtest.tools import run_backtest_compact
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.data.factory import DataFactory

#: The four losing days and the bars they fall on.
LOSSES = {12: -0.05, 25: -0.04, 38: -0.035, 51: -0.03}


def _bars(returns: np.ndarray) -> pd.DataFrame:
    close = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + returns)))
    index = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(len(close), 1_000_000.0),
        },
        index=index,
    )


def _four_losses(n_returns: int, gain: float) -> np.ndarray:
    returns = np.full(n_returns, gain)
    for bar, loss in LOSSES.items():
        returns[bar] = loss
    return returns


@pytest.fixture
def portfolio_tool(monkeypatch):
    """dispatch('run_portfolio_simulation', ...) over one frame we hand it,
    held whole from the first bar at zero cost."""
    frames = {}
    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.backtest.tools.fetch_ohlcv_panel_sync",
        lambda tickers, *_a, **_k: {t: frames[t] for t in tickers},
    )

    def run(frame: pd.DataFrame):
        frames.clear()
        frames["AAA"] = frame
        first = str(frame.index[0].date())
        return dispatch(
            "run_portfolio_simulation",
            {
                "tickers": ["AAA"],
                "start_date": first,
                "end_date": str(frame.index[-1].date()),
                "target_weights": {"AAA": {first: 1.0}},
                "commission_pct": 0.0,
                "slippage_pct": 0.0,
                "fill_price": "close",
            },
        )

    return run


class TestPortfolioSimulation:
    def test_four_losing_days_are_a_tail_of_four_and_warned(self, portfolio_tool):
        """60 bar returns and the opening one (zero at zero cost): 61
        returns, of which the four losses are the tail."""
        out = portfolio_tool(_bars(_four_losses(60, 0.002)))
        assert out["cvar_95_tail_observations"] == 4
        # The shortfall really is the mean of those four losses.
        assert out["cvar_95"] == pytest.approx(
            -np.mean(list(LOSSES.values())), abs=1e-6
        )
        (warning,) = [w for w in out["warnings"] if w.startswith("cvar_95 ")]
        assert "the worst 4 of 61 returns at 95% confidence" in warning

    def test_a_long_run_has_a_tail_of_its_quantile_and_no_warning(self, portfolio_tool):
        """400 distinct daily returns and the opening zero: the 5% quantile
        of 401 sits just above the 21st lowest, so the tail is 21 and the
        estimate is not warned about."""
        returns = np.random.default_rng(7).normal(0.0004, 0.01, 400)
        out = portfolio_tool(_bars(returns))
        assert out["cvar_95_tail_observations"] == 21
        assert not [w for w in out["warnings"] if w.startswith("cvar_95 ")]


class _Stub:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        return self.frame.copy()


class TestCompactBacktest:
    def test_four_losing_days_are_a_tail_of_four_and_warned(self, monkeypatch):
        """A rising price the crossover is long through from its 30th bar,
        with the four losses after that: 80 strategy returns, zero before
        the position, +0.5% after it but for the four. The tail is the
        four."""
        frame = _bars(_four_losses(79, 0.005))
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: _Stub(frame))
        out = run_backtest_compact(
            BacktestCompactInput(
                symbol="AAA",
                start_date=str(frame.index[0].date()),
                end_date=str(frame.index[-1].date()),
                strategy_type="sma_crossover",
                parameters={"fast_period": 5, "slow_period": 10},
                commission_pct=0.0,
                slippage_pct=0.0,
            )
        )
        assert out.risk.cvar_95_tail_observations == 4
        assert out.risk.cvar_95 == pytest.approx(
            -np.mean(list(LOSSES.values())), abs=1e-6
        )
        (warning,) = [w for w in out.warnings if w.startswith("cvar_95 ")]
        assert "the worst 4 of 80 returns at 95% confidence" in warning
