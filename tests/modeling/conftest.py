"""
Fixtures specific to the modeling test suite. tests/conftest.py's shared
`mock_provider` returns identical OHLCV for every symbol requested, which
is fine for single-symbol tool tests but degenerates a multi-entity
panel/PCA test (every entity would be perfectly correlated) — so this
package gets its own per-symbol randomized provider fixture.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.factory import DataFactory

_N = 500  # bars per symbol; >= the largest Phase-1 feature lookback (252) with headroom.


def make_ohlcv(symbol: str, n: int = _N) -> pd.DataFrame:
    """Deterministic per-symbol synthetic OHLCV (seeded from the symbol
    name), so different tickers get genuinely different, reproducible
    price paths across test runs."""
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=dates)


@pytest.fixture
def multi_symbol_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, start, end: make_ohlcv(symbol)
    return provider


@pytest.fixture
def patched_multi_factory(
    multi_symbol_provider: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Patches DataFactory.get_provider to the per-symbol randomized
    provider above, for the duration of one test."""
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: multi_symbol_provider)
    return multi_symbol_provider


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Every modeling test writes artifacts to a per-test temp directory,
    never the real SQT_RUNS_DIR, with audit writes disabled (irrelevant
    to what these tests check, and would otherwise pollute the real
    audit log)."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
