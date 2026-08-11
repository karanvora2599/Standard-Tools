"""
Fixtures specific to the modeling test suite. tests/conftest.py's shared
`mock_provider` returns identical OHLCV for every symbol requested, which
is fine for single-symbol tool tests but degenerates a multi-entity
panel/PCA test (every entity would be perfectly correlated) — so this
package gets its own per-symbol randomized provider fixture.
"""

import hashlib
from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.metadata import DataSetMetadata

_N = (
    500  # bars per symbol; >= the largest Phase-1 feature lookback (252) with headroom.
)


def make_ohlcv(symbol: str, n: int = _N) -> pd.DataFrame:
    """Deterministic per-symbol synthetic OHLCV (seeded from the symbol
    name), so different tickers get genuinely different, reproducible
    price paths across test runs.

    Seeded from SHA-256 of the symbol, not the builtin `hash()`. Python
    salts string hashing per interpreter process (PYTHONHASHSEED), so
    `hash("AAPL")` differs between runs — the old seeding was reproducible
    only WITHIN one process, which is exactly where a flaky
    numerically-sensitive test would be hardest to reproduce afterwards.
    """
    seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def mock_metadata(
    symbol: str = "AAA",
    interval: str = "1d",
    *,
    survivorship_free: bool = False,
    point_in_time: bool = False,
) -> DataSetMetadata:
    """False/False by default — what every provider this package ships
    actually reports. Tests that want the warnings absent must opt in."""
    return DataSetMetadata(
        provider="mock",
        adjusted=True,
        survivorship_free=survivorship_free,
        point_in_time=point_in_time,
        frequency=interval,
        timezone="America/New_York",
    )


def make_provider_mock(fetch: Callable[[str], pd.DataFrame], **metadata_kwargs):
    """
    Build a provider mock whose sync and async fetch paths are driven by
    the SAME function.

    Wiring only `get_ohlcv` is not enough now that build_dataset fetches the
    universe concurrently, and getting it wrong is quiet rather than loud:
    an unspecced MagicMock happily returns a non-awaitable `get_ohlcv_async`
    attribute, `await` on it raises TypeError, and that TypeError is
    collected as a per-symbol fetch failure — so a test asserting only that
    the error names a symbol PASSES while exercising nothing it meant to.
    Two tests in this suite did exactly that. Routing both paths through one
    function makes the two agree by construction.
    """
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, start, end, interval="1d": fetch(
        symbol
    )
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, start, end, interval="1d": fetch(symbol)
    )
    provider.get_metadata.side_effect = lambda symbol, interval="1d": mock_metadata(
        symbol, interval, **metadata_kwargs
    )
    return provider


@pytest.fixture
def multi_symbol_provider() -> MagicMock:
    provider = MagicMock()
    # interval is accepted (with a default) rather than required: the
    # modeling builder now passes it positionally on every call, while
    # other callers in the wider suite still fetch with three arguments.
    provider.get_ohlcv.side_effect = lambda symbol, start, end, interval="1d": (
        make_ohlcv(symbol)
    )
    # build_dataset fetches the universe through the async path
    # (dataset/fetch.py), as do some existing agent.tools functions
    # (e.g. run_signal_panel_backtest via portfolio.fetch_ohlcv_panel_async)
    # -- without this, asyncio.gather chokes on a plain (non-awaitable)
    # MagicMock return value.
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, start, end, interval="1d": make_ohlcv(symbol)
    )
    # A real DataSetMetadata, not the auto-created MagicMock attribute an
    # unspecced mock would otherwise return: dataset/coverage.py reads
    # point_in_time/survivorship_free off it, and a MagicMock's truthy
    # attributes would silently suppress exactly the warnings under test.
    # False/False mirrors what every provider this package ships reports.
    provider.get_metadata.side_effect = lambda symbol, interval="1d": DataSetMetadata(
        provider="mock",
        adjusted=True,
        survivorship_free=False,
        point_in_time=False,
        frequency=interval,
        timezone="America/New_York",
    )
    return provider


@pytest.fixture
def patched_multi_factory(
    multi_symbol_provider: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Patches DataFactory.get_provider to the per-symbol randomized
    provider above, for the duration of one test."""
    monkeypatch.setattr(
        DataFactory, "get_provider", lambda *a, **kw: multi_symbol_provider
    )
    return multi_symbol_provider


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """
    Every modeling test writes artifacts to a per-test temp directory,
    never the real SQT_RUNS_DIR.

    Audit is REDIRECTED to a temp directory rather than disabled. It used
    to be switched off here (`SQT_AUDIT_ENABLED=0`) on the grounds that it
    was irrelevant to these tests and would pollute the real log — but the
    second concern is solved by redirection alone, and disabling it meant
    the claimed modeling->audit integration was never exercised end to end
    by the modeling suite at all. Tests that want to inspect the log can
    read SQT_AUDIT_DIR; tests that don't care are unaffected.
    """
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
