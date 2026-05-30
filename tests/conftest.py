"""
Shared fixtures for the Standard Quant Tools test suite.

Data fixtures are session-scoped (computed once per pytest run).
Provider/factory fixtures are function-scoped (fresh mock per test).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.base import FinancialRatios, TickerInfo
from standard_quant_tools.data.factory import DataFactory


# ── Synthetic market data ─────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def sample_close() -> pd.Series:
    """500-bar synthetic close price with trend + noise."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.012, 500)
    prices = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range('2022-01-01', periods=500, freq='B')
    return pd.Series(prices, index=dates, name='Close')


@pytest.fixture(scope='session')
def sample_ohlcv(sample_close: pd.Series) -> pd.DataFrame:
    """Full OHLCV DataFrame derived from the synthetic close series."""
    np.random.seed(0)
    n = len(sample_close)
    close = sample_close.values
    spread = np.random.uniform(0.3, 1.5, n)
    high = pd.Series(close + spread, index=sample_close.index)
    low = pd.Series(close - spread, index=sample_close.index)
    open_ = sample_close.shift(1).fillna(sample_close.iloc[0])
    volume = pd.Series(
        np.random.randint(500_000, 5_000_000, n).astype(float),
        index=sample_close.index,
    )
    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': sample_close,
        'Volume': volume,
    })


@pytest.fixture(scope='session')
def sample_returns(sample_close: pd.Series) -> pd.Series:
    return sample_close.pct_change().dropna()


@pytest.fixture(scope='session')
def sample_equity(sample_close: pd.Series) -> pd.Series:
    """Equity curve starting at $10,000."""
    returns = sample_close.pct_change().fillna(0)
    return 10_000.0 * (1 + returns).cumprod()


@pytest.fixture(scope='session')
def benchmark_returns(sample_returns: pd.Series) -> pd.Series:
    """Benchmark returns: correlated with sample but with independent noise."""
    np.random.seed(7)
    noise = pd.Series(np.random.normal(0, 0.006, len(sample_returns)), index=sample_returns.index)
    return (0.7 * sample_returns + noise).rename('Benchmark')


# ── Mock data provider ────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider(sample_ohlcv: pd.DataFrame) -> MagicMock:
    """DataProvider mock that returns synthetic data without network calls."""
    provider = MagicMock()
    provider.get_ohlcv.return_value = sample_ohlcv
    provider.get_ohlcv_async = AsyncMock(return_value=sample_ohlcv)
    provider.get_ticker_info.return_value = TickerInfo(
        symbol='AAPL',
        name='Apple Inc.',
        sector='Technology',
        industry='Consumer Electronics',
        full_time_employees=161_000,
        city='Cupertino',
        country='United States',
        website='https://www.apple.com',
    )
    provider.get_financial_ratios.return_value = FinancialRatios(
        forward_pe=28.5,
        trailing_pe=30.2,
        price_to_book=45.0,
        debt_to_equity=150.0,
        return_on_equity=0.45,
        profit_margins=0.25,
        dividend_yield=0.005,
        market_cap=2_800_000_000_000,
    )
    return provider


@pytest.fixture
def patched_factory(mock_provider: MagicMock, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patches DataFactory.get_provider to return mock_provider for the test."""
    monkeypatch.setattr(DataFactory, 'get_provider', lambda *a, **kw: mock_provider)
    return mock_provider
