"""
Tests for backtest/strategy.py's VectorizedStrategy Protocol.

Note: because the Protocol only declares __call__, isinstance() can only
confirm callability, not that the signature actually matches — a known
limitation of @runtime_checkable Protocols with a single dunder member
(Python has no runtime mechanism to check parameter names/types). These
tests demonstrate every STRATEGY_REGISTRY entry and a custom callable
passing that callability check, and rely on static type-checking (not
these tests) for real signature enforcement.
"""

import pandas as pd

from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.backtest.strategy import VectorizedStrategy


class TestVectorizedStrategyProtocol:
    def test_every_registered_strategy_satisfies_protocol(self):
        for name, fn in STRATEGY_REGISTRY.items():
            assert isinstance(
                fn, VectorizedStrategy
            ), f"{name} does not satisfy VectorizedStrategy"

    def test_custom_callable_satisfies_protocol(self):
        def my_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
            return (price_data["Close"] > threshold).astype(float)

        assert isinstance(my_signal, VectorizedStrategy)

    def test_non_callable_does_not_satisfy_protocol(self):
        assert not isinstance("not a strategy", VectorizedStrategy)
        assert not isinstance(42, VectorizedStrategy)
