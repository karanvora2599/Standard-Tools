"""
The risk-free rate, and the honest boundary around it.

Seventeen tools report a Sharpe ratio. Until now exactly one took a rate,
so the other sixteen silently measured total return per unit of risk and
called it Sharpe. At a 4-5% short rate that is not a rounding difference --
for a low-volatility strategy it is most of the number.

Three of them compute the ratio at the Python tool layer and now take a
real rate. The rest get theirs from run_strategy, which hard-codes zero in
the Python path AND in the C++ kernel. The tests below pin both halves:
that the rate works where it is offered, and that it is NOT offered where
it would be accepted and ignored -- an argument a tool silently discards is
worse than one it never advertised.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    AnalysisInput,
    PortfolioInput,
    RiskAttributionInput,
)
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.tools import _TOOL_DISPATCH


@pytest.fixture
def steady_prices():
    """A gently rising series: positive Sharpe at rf=0, so raising the rate
    has somewhere to move it."""
    n = 400
    rng = np.random.default_rng(5)
    # Drift plus real downside: a monotonic series has no negative returns,
    # so its Sortino is infinite and the test could not see the rate move it.
    close = 100.0 * np.exp(np.linspace(0, 0.30, n) + rng.normal(0, 0.006, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


@pytest.fixture
def stub(monkeypatch, steady_prices):
    class _Stub:
        def get_ohlcv(self, *a, **k):
            return steady_prices

    for runtime in ("research", "portfolio"):
        monkeypatch.setattr(
            f"standard_quant_tools.agent.runtimes.{runtime}.tools."
            "DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
            raising=False,
        )
    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.research.tools.fetch_returns_sync",
        lambda tickers, start, end: pd.DataFrame(
            {t: steady_prices["Close"].pct_change().dropna() for t in tickers}
        ),
        raising=False,
    )
    return steady_prices


class TestTheRateIsHonoured:
    def test_a_higher_rate_lowers_the_single_asset_sharpe(self, stub):
        research = resolve("research")
        base = {
            "symbol": "TEST",
            "benchmark": "TEST",
            "period": "1y",
        }
        at_zero = research.dispatch("analyze_stock_risk", base)
        at_five = research.dispatch(
            "analyze_stock_risk", {**base, "risk_free_rate": 0.05}
        )
        assert at_five["sharpe_ratio"] < at_zero["sharpe_ratio"]
        assert at_five["sortino_ratio"] < at_zero["sortino_ratio"]

    def test_the_default_reproduces_the_old_behaviour(self, stub):
        """Zero is the historical assumption, so an existing caller that
        never passes a rate must get exactly what it got before."""
        research = resolve("research")
        base = {"symbol": "TEST", "benchmark": "TEST", "period": "1y"}
        implicit = research.dispatch("analyze_stock_risk", base)
        explicit = research.dispatch(
            "analyze_stock_risk", {**base, "risk_free_rate": 0.0}
        )
        assert implicit["sharpe_ratio"] == explicit["sharpe_ratio"]

    def test_the_three_tool_layer_inputs_all_carry_it(self):
        for model in (AnalysisInput, PortfolioInput, RiskAttributionInput):
            assert "risk_free_rate" in model.model_fields, model.__name__
            assert model.model_fields["risk_free_rate"].default == 0.0

    def test_a_negative_or_absurd_rate_is_rejected(self):
        for bad in (-0.01, 1.5):
            with pytest.raises(Exception):
                AnalysisInput(symbol="X", risk_free_rate=bad)


class TestTheBoundaryIsHonest:
    def test_engine_backed_tools_do_not_advertise_a_rate_they_ignore(self):
        """run_strategy computes Sharpe with the rate fixed at zero, in
        both the Python path and the C++ kernel. Offering a field on these
        inputs would mean accepting an argument and discarding it, which
        reads to a caller as support."""
        engine_backed = {
            "run_sma_backtest",
            "run_rsi_backtest",
            "run_macd_backtest",
            "run_bollinger_backtest",
            "run_buy_and_hold",
            "run_custom_signal_backtest",
            "run_walk_forward_backtest",
            "run_backtest_optimization",
        }
        for name in engine_backed:
            _fn, model = _TOOL_DISPATCH[name]
            assert "risk_free_rate" not in model.model_fields, (
                f"{name} advertises a risk_free_rate the engine cannot "
                "honour — either thread it through run_strategy and the "
                "C++ kernel, or do not offer it."
            )

    def test_the_engine_really_does_fix_the_rate_at_zero(self):
        """If this ever stops being true, the field above should be added
        rather than this test relaxed."""
        import inspect

        from standard_quant_tools.backtest import engine

        source = inspect.getsource(engine.run_strategy)
        assert "risk_free_rate" not in source
