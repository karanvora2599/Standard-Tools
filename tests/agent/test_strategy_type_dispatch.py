"""`strategy_type` selects the strategy. For a long time it selected nothing.

`BacktestInput.strategy_type` is a **required** field whose enum names every
strategy, and `run_sma_backtest`, `run_rsi_backtest`, `run_macd_backtest` and
`run_bollinger_backtest` each ignored it — every one hardcoded its own registry
lookup and ran that, whatever was asked for.

Measured on SPY over 2026-01-02 to 2026-09-02, asking one tool for four
different strategies returned the identical number four times:

    run_sma_backtest    every strategy_type ->  +4.54%,  3 trades
    run_rsi_backtest    every strategy_type -> +10.32%,  1 trade
    run_macd_backtest   every strategy_type ->  +8.11%,  6 trades

The consequence lands hardest where the mistake is most likely. Asking
`run_sma_backtest` for `buy_and_hold` is how a baseline gets fetched, and it
returned an SMA crossover: +4.54% against a true +12.05% on the same window.
That is the difference between an active strategy appearing to beat the market
and appearing to lose to it, and every comparison downstream inherits it.

`BacktestResult` carries no strategy name, which makes this worse rather than
better — nothing in the payload contradicts the caller, so filing the answer
under the requested name is correct bookkeeping of an incorrect number.

These tests use a synthetic price series, so they assert *dispatch* rather than
returns — the property that broke, and one that needs no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import BacktestInput
from standard_quant_tools.agent.runtimes.backtest.tools import (
    run_bollinger_backtest,
    run_macd_backtest,
    run_rsi_backtest,
    run_sma_backtest,
)
from standard_quant_tools.backtest.strategies import (
    BASELINE_REGISTRY,
    RUNNABLE,
    STRATEGY_REGISTRY,
)
from standard_quant_tools.error import ValidationError

TOOLS = {
    "run_sma_backtest": run_sma_backtest,
    "run_rsi_backtest": run_rsi_backtest,
    "run_macd_backtest": run_macd_backtest,
    "run_bollinger_backtest": run_bollinger_backtest,
}

#: Every name a caller may give as `strategy_type`: the active strategies plus
#: the baselines. `custom_signal` is excluded deliberately — it supplies its own
#: signal series and has its own tool.
DISPATCHABLE = sorted(RUNNABLE)


@pytest.fixture
def prices() -> pd.DataFrame:
    """A trending series with enough shape that the strategies disagree.

    Deliberately not random: a flat or purely noisy series can make two
    different strategies produce identical signals by accident, and a test that
    passes for that reason would not have caught the bug it exists for.
    """
    n = 320
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = np.linspace(100.0, 160.0, n)
    wave = 8.0 * np.sin(np.linspace(0, 9 * np.pi, n))
    close = trend + wave
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.2,
            "Low": close - 1.2,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch, prices):
    """Serve the fixture instead of a provider. These tests are about dispatch."""
    from standard_quant_tools.agent.runtimes.backtest import tools as mod

    class _Provider:
        def get_ohlcv(self, symbol, start, end, interval="1d"):
            return prices

    monkeypatch.setattr(mod.DataFactory, "get_provider", staticmethod(lambda *a, **k: _Provider()))


def _input(strategy_type: str) -> BacktestInput:
    return BacktestInput(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2025-03-01",
        strategy_type=strategy_type,
        parameters={},
        initial_capital=100_000.0,
        commission_pct=0.0,
        slippage_pct=0.0,
        fill_price="next_open",
    )


def _run(tool, strategy_type: str):
    return TOOLS[tool](_input(strategy_type))


# --- the property that broke ---------------------------------------------------------


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_one_tool_runs_whichever_strategy_is_named(tool):
    """The regression, stated directly: the same tool, asked for different
    strategies, must not return the same backtest four times."""
    results = {name: _run(tool, name) for name in DISPATCHABLE}
    signatures = {
        (round(r.total_return, 10), r.num_trades) for r in results.values()
    }
    assert len(signatures) > 1, (
        f"{tool} returned an identical backtest for every strategy_type in "
        f"{DISPATCHABLE} — strategy_type is being ignored"
    )


@pytest.mark.parametrize("strategy", DISPATCHABLE)
def test_every_tool_agrees_on_a_given_strategy(strategy):
    """The other half of the same property. Which of the four tools is called
    must not change the answer once `strategy_type` names the strategy — the
    tool names are aliases, and an alias that changes the result is not one."""
    outcomes = {
        tool: (round(_run(tool, strategy).total_return, 10), _run(tool, strategy).num_trades)
        for tool in TOOLS
    }
    assert len(set(outcomes.values())) == 1, (
        f"the four tools disagree on {strategy}: {outcomes}"
    )


def test_the_result_carries_no_strategy_label_so_the_caller_owns_the_name():
    """`BacktestResult` has no `strategy_type` field, and that is worth pinning.

    It means the tool never contradicted itself in writing — the mislabelling
    happened entirely in the caller's head, which is *worse*, because there is
    nothing in the payload to catch it. Asking for `buy_and_hold` and filing the
    answer under that name was correct bookkeeping of an incorrect number.

    If a label is ever added, it must be the strategy that ran.
    """
    fields = set(type(_run("run_sma_backtest", "sma_crossover")).model_fields)
    assert "strategy_type" not in fields, (
        "a label has been added — it must now be asserted to match the strategy "
        "that actually ran, not the one that was requested"
    )


# --- the baseline ----------------------------------------------------------------------


def test_the_baseline_is_runnable_but_not_selectable():
    """The distinction is load-bearing.

    `STRATEGY_REGISTRY` is read by the grid, walk-forward, the optimiser and the
    regime-adaptive sweep as *the set of strategies a search may choose
    between*. Putting the baseline in there lets a selector "choose" to hold,
    which is not a finding — it is the null hypothesis wearing a strategy's
    name. It also breaks the contract that every registry name is accepted by
    `WalkForwardInput.strategy`, and hands a parameter sweep something with no
    parameters to sweep.

    Adding it to the registry did exactly that: thirteen tests, and they were
    right to break.
    """
    assert "buy_and_hold" in BASELINE_REGISTRY
    assert "buy_and_hold" in RUNNABLE
    assert "buy_and_hold" not in STRATEGY_REGISTRY
    assert set(RUNNABLE) == set(STRATEGY_REGISTRY) | set(BASELINE_REGISTRY)


def test_buy_and_hold_holds(prices):
    """One trade, in from the first bar. Any indicator strategy is flat during
    its own lookback; a baseline that shared that gap would understate the very
    thing it exists to measure."""
    signals = BASELINE_REGISTRY["buy_and_hold"](prices)
    assert len(signals) == len(prices)
    assert (signals == 1.0).all()

    result = _run("run_sma_backtest", "buy_and_hold")
    assert result.num_trades == 1

    # And it tracks the underlying, which is the whole point of a baseline.
    underlying = prices["Close"].iloc[-1] / prices["Open"].iloc[1] - 1
    assert result.total_return == pytest.approx(underlying, abs=0.02)


def test_buy_and_hold_takes_no_parameters(prices):
    """A parameter passed to it is a caller who believes there is a knob, and
    there is not. `list_strategies` reports it as a synthetic label precisely
    because it has no parameter contract, so it validates its own emptiness
    rather than borrowing the strategy parameter machinery."""
    from standard_quant_tools.backtest.strategy_params import STRATEGY_PARAM_SCHEMA

    assert "buy_and_hold" not in STRATEGY_PARAM_SCHEMA
    with pytest.raises(ValidationError, match="takes no parameters"):
        BASELINE_REGISTRY["buy_and_hold"](prices, period=20)


def test_buy_and_hold_beats_a_late_entering_strategy_on_a_rising_series(prices):
    """The specific comparison the bug corrupted. On a series that rises, a
    baseline in from bar one must not lose to a strategy that spends its warmup
    in cash — and while it was mislabelled, it appeared to."""
    baseline = _run("run_sma_backtest", "buy_and_hold").total_return
    crossover = _run("run_sma_backtest", "sma_crossover").total_return
    assert baseline > crossover


# --- what is refused --------------------------------------------------------------------


def test_custom_signal_is_refused_with_a_pointer():
    """It supplies its own signal series, so there is nothing for these tools to
    compute. Refusing with the name of the right tool beats running something
    else under its label."""
    with pytest.raises(ValidationError, match="run_custom_signal_backtest"):
        _run("run_sma_backtest", "custom_signal")


def test_an_unknown_strategy_is_refused_at_the_model_boundary():
    """`strategy_type` is a `Literal`, so pydantic rejects an unknown name
    before the tool is reached — and the error names every valid one."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="sma_crossover"):
        _input("no_such_strategy")


def test_the_dispatcher_still_guards_a_name_the_model_would_have_allowed():
    """Defence for a direct call, and for the day the enum and the registry
    drift apart — the enum is hand-maintained and the registry is not."""
    from standard_quant_tools.agent.runtimes.backtest.tools import _dispatch_backtest

    payload = _input("sma_crossover")
    object.__setattr__(payload, "strategy_type", "no_such_strategy")
    with pytest.raises(ValidationError, match="Unknown strategy"):
        _dispatch_backtest(payload, "sma_crossover")


def test_the_enum_and_the_registry_have_not_drifted():
    """Every dispatchable name the schema advertises must exist in the registry.
    `custom_signal` is the one deliberate exception, and it is refused by name."""
    import typing

    advertised = set(typing.get_args(BacktestInput.model_fields["strategy_type"].annotation))
    missing = advertised - set(RUNNABLE) - {"custom_signal"}
    assert not missing, f"the schema offers strategies the registry cannot run: {missing}"
