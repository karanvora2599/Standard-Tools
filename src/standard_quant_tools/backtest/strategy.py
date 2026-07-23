"""
Formal type for the signal-generating callable every STRATEGY_REGISTRY
entry (backtest/strategies.py) and every backtest_grid(strategy=<callable>)
custom function (backtest/engine.py) already satisfies structurally. This
module changes no runtime behavior — it names an existing contract so other
code (and users writing a custom `strategy=` callable) can annotate
against it, closing the "standardized Strategy interface" gap without
introducing a parallel engine.

Explicitly deferred: the event-driven Strategy Protocol from the original
design doc (`initialize`/`on_data`/`on_fill`, `OrderIntent`, `Fill`,
`StrategyContext`). This library's engine is signal-array-based end to end
(run_strategy, run_portfolio_simulation) — there is no execution loop that
would call those methods. Building that Protocol with nothing to verify it
against would be a hollow type, and a second, unused event-driven engine is
out of scope for "finishing" existing work — a documented gap, not a
silently dropped one.
"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class VectorizedStrategy(Protocol):
    """
    A callable (price_data, **params) -> pd.Series of {-1, 0, 1} signals.
    Every function in STRATEGY_REGISTRY (backtest/strategies.py) satisfies
    this structurally, as does any custom `strategy=` callable passed to
    backtest_grid (backtest/engine.py) or referenced by
    run_backtest_optimization.

    Known limitation of @runtime_checkable Protocols with only a
    `__call__` member: isinstance() can only confirm the object is
    callable, not that its signature actually matches — Python has no
    runtime mechanism to check parameter names/types against a Protocol.
    Use this for static type-checking (mypy/pyright) where it's fully
    enforced; treat an isinstance() check as a callability check only.
    """

    def __call__(self, price_data: pd.DataFrame, **params: Any) -> pd.Series: ...
