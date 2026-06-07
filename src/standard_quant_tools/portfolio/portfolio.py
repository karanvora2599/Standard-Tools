import asyncio
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    var_historical, cvar, information_ratio
)
from standard_quant_tools.metrics.return_metrics import cagr, annualized_volatility
from standard_quant_tools.error import ValidationError


def build_portfolio(
    returns_df: pd.DataFrame,
    weights: Union[List[float], np.ndarray],
) -> pd.Series:
    """
    Compute daily weighted portfolio returns.

    Args:
        returns_df: DataFrame where each column is a ticker's daily returns.
        weights: Portfolio weights (must sum to 1.0). Order matches returns_df columns.

    Returns:
        pd.Series of daily portfolio returns.
    """
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != returns_df.shape[1]:
        raise ValidationError(
            f"weights length ({len(w)}) must match number of tickers ({returns_df.shape[1]})"
        )
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValidationError(f"weights must sum to 1.0, got {w.sum():.4f}")

    # Matrix multiply: (n_days, n_assets) @ (n_assets,) → (n_days,)
    return pd.Series(returns_df.values @ w, index=returns_df.index, name='Portfolio')


def portfolio_metrics(
    returns_df: pd.DataFrame,
    weights: Union[List[float], np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive portfolio-level performance metrics.

    Uses NumPy matrix operations for covariance, which is ~50x faster
    than iterative per-pair calculations.

    Returns:
        Dict with return, risk, and ratio metrics plus correlation matrix.
    """
    w = np.asarray(weights, dtype=np.float64)
    port_returns = build_portfolio(returns_df, w)
    equity_curve = (1 + port_returns).cumprod()

    # Covariance (annualized): single O(n·k²) BLAS call via numpy
    cov_matrix = returns_df.cov().to_numpy(dtype=np.float64) * periods_per_year
    port_vol = float(np.sqrt(w @ cov_matrix @ w))

    annual_ret = float(cagr(equity_curve, periods_per_year))
    excess = annual_ret - risk_free_rate
    sr = excess / port_vol if port_vol != 0 else 0.0

    metrics: Dict[str, Any] = {
        'annualized_return': round(annual_ret, 6),
        'annualized_volatility': round(port_vol, 6),
        'sharpe_ratio': round(sr, 4),
        'sortino_ratio': round(sortino_ratio(port_returns, risk_free_rate, periods_per_year), 4),
        'max_drawdown': round(max_drawdown(equity_curve), 6),
        'calmar_ratio': round(calmar_ratio(equity_curve, periods_per_year), 4),
        'var_95': round(var_historical(port_returns, 0.95), 6),
        'cvar_95': round(cvar(port_returns, 0.95), 6),
        'total_return': round(float(equity_curve.iloc[-1] - 1), 6),
        'tickers': list(returns_df.columns),
        'weights': w.tolist(),
    }

    if benchmark_returns is not None:
        metrics['information_ratio'] = round(
            information_ratio(port_returns, benchmark_returns, periods_per_year), 4
        )

    return metrics


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix across tickers.
    Uses pandas (backed by NumPy BLAS) — O(n·k²) time.

    Returns:
        DataFrame with tickers as both index and columns.
    """
    return returns_df.corr()


async def fetch_returns_async(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
) -> pd.DataFrame:
    """
    Fetch OHLCV for multiple tickers concurrently and return a returns DataFrame.
    One network round-trip per ticker, fully async.
    """
    from standard_quant_tools.data.factory import DataFactory
    provider = DataFactory.get_provider()

    tasks = [
        provider.get_ohlcv_async(ticker, start_date, end_date, interval)
        for ticker in tickers
    ]
    dfs = await asyncio.gather(*tasks)

    # Build aligned close-price matrix then compute returns
    close_prices = pd.DataFrame(
        {ticker: df['Close'] for ticker, df in zip(tickers, dfs)}
    )
    return close_prices.pct_change().dropna()


def fetch_returns_sync(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = '1d',
) -> pd.DataFrame:
    """Synchronous wrapper around fetch_returns_async."""
    return asyncio.run(fetch_returns_async(tickers, start_date, end_date, interval))
