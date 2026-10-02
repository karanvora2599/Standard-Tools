"""
At the default level, an indicator does not build the arguments of a debug
line nobody will see.

Python evaluates a call's arguments before the call, so
`logger.debug("... %s", result.dropna().iloc[-1])` ran the dropna on every
call whatever the level, and logger.debug only then discarded it. For RSI
and ADX on 2,115 bars that was most of the call (see the CHANGELOG entry
of 2026-10-01). These six now build those arguments only under
`logger.isEnabledFor(logging.DEBUG)`, as `stochastic_oscillator` and
`bollinger_bands` already did.

Each test replaces the module's `logger.debug` with a recorder. At WARNING
the summary line is never requested, so its arguments were never built; at
DEBUG it is requested with the values it always carried, and the result is
the same at both levels.
"""

import logging
from typing import Any, Callable, List, Tuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators import momentum, trend, volatility, volume


def _bars(n: int = 300):
    rng = np.random.default_rng(5)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)), index=index)
    high = close + rng.uniform(0.1, 1.2, n)
    low = close - rng.uniform(0.1, 1.2, n)
    vol = pd.Series(rng.integers(100_000, 1_000_000, n).astype(float), index=index)
    return high, low, close, vol


H, L, C, V = _bars()

# (module, call, the summary line's format prefix, the values it carries)
CALLS: List[Tuple[Any, Callable[[], Any], str, Callable[[Any], tuple]]] = [
    (
        momentum,
        lambda: momentum.rsi(C, 14),
        "[rsi] last=",
        lambda r: (
            float(r.dropna().iloc[-1]),
            float(r.dropna().min()),
            float(r.dropna().max()),
        ),
    ),
    (
        trend,
        lambda: trend.adx(H, L, C, 14),
        "[adx] last ",
        lambda r: (
            float(r.dropna()["DI_Plus"].iloc[-1]),
            float(r.dropna()["DI_Minus"].iloc[-1]),
            float(r.dropna()["ADX"].iloc[-1]),
            "strong" if float(r.dropna()["ADX"].iloc[-1]) > 25 else "weak",
        ),
    ),
    (
        trend,
        lambda: trend.macd(C),
        "[macd] last ",
        lambda r: tuple(
            float(r.dropna()[k].iloc[-1]) for k in ("MACD", "Signal", "Histogram")
        ),
    ),
    (
        volatility,
        lambda: volatility.atr(H, L, C, 14),
        "[atr] last=",
        lambda r: (float(r.dropna().iloc[-1]),),
    ),
    (
        volume,
        lambda: volume.vwap(H, L, C, V, period=20),
        "[vwap] last=",
        lambda r: (float(r.dropna().iloc[-1]),),
    ),
    (
        volume,
        lambda: volume.obv(C, V),
        "[obv] final=",
        lambda r: (
            float(r.iloc[-1]),
            "up" if float(r.iloc[-1]) > float(r.iloc[0]) else "down",
        ),
    ),
]
IDS = ["rsi", "adx", "macd", "atr", "vwap", "obv"]


def _record(module: Any, level: int, caplog, monkeypatch) -> List[tuple]:
    """Set the module logger's level (restored by caplog at teardown) and
    record every logger.debug call it is asked to make."""
    requested: List[tuple] = []
    caplog.set_level(level, logger=module.logger.name)
    monkeypatch.setattr(
        module.logger, "debug", lambda msg, *args, **kw: requested.append((msg, args))
    )
    return requested


@pytest.mark.parametrize("module,call,prefix,expected", CALLS, ids=IDS)
class TestTheSummaryLine:
    def test_is_not_built_at_warning(
        self, module, call, prefix, expected, caplog, monkeypatch
    ):
        requested = _record(module, logging.WARNING, caplog, monkeypatch)
        call()
        assert not [msg for msg, _ in requested if msg.startswith(prefix)]

    def test_is_built_at_debug_with_the_same_values(
        self, module, call, prefix, expected, caplog, monkeypatch
    ):
        requested = _record(module, logging.DEBUG, caplog, monkeypatch)
        result = call()
        lines = [args for msg, args in requested if msg.startswith(prefix)]
        assert lines == [expected(result)]

    def test_the_result_does_not_depend_on_the_level(
        self, module, call, prefix, expected, caplog, monkeypatch
    ):
        _record(module, logging.WARNING, caplog, monkeypatch)
        quiet = call()
        _record(module, logging.DEBUG, caplog, monkeypatch)
        loud = call()
        if isinstance(quiet, pd.DataFrame):
            pd.testing.assert_frame_equal(quiet, loud)
        else:
            pd.testing.assert_series_equal(quiet, loud)


def test_a_series_with_no_valid_value_logs_nothing_at_debug(caplog, monkeypatch):
    """The emptiness check still sits inside the guard: an all-NaN result
    (period longer than the data) requests no summary line at DEBUG."""
    requested = _record(momentum, logging.DEBUG, caplog, monkeypatch)
    momentum.rsi(C.iloc[:10], 14)
    assert not [msg for msg, _ in requested if msg.startswith("[rsi] last=")]
