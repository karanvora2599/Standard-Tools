import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import cagr
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio,
    cvar,
    information_ratio,
    max_drawdown,
    sortino_ratio,
    var_historical,
)


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
    logger.debug(
        "[portfolio] build  assets=%d  weights=%s  weight_sum=%.4f",
        returns_df.shape[1],
        [round(float(x), 4) for x in w],
        float(w.sum()),
    )
    if returns_df.empty:
        # Everything downstream assumes at least one row; portfolio_metrics
        # reaches for equity_curve.iloc[-1] and failed far from this cause.
        raise ValidationError(
            "returns_df is empty — there are no observations to weight. This is "
            "commonly the result of an inner join over tickers with no shared "
            "dates rather than a genuinely empty request."
        )
    if len(w) != returns_df.shape[1]:
        raise ValidationError(
            f"weights length ({len(w)}) must match number of tickers ({returns_df.shape[1]})"
        )
    # Finiteness before the sum check. A NaN weight was caught only
    # incidentally (the sum became NaN, so the message blamed the sum), and an
    # infinite one would have been reported the same misleading way. Naming
    # the real problem matters more here than usual, because the caller's next
    # move is to go looking at their weights, not their arithmetic.
    if not np.all(np.isfinite(w)):
        raise ValidationError(
            f"weights contains {int(np.sum(~np.isfinite(w)))} non-finite value(s); "
            "every weight must be a finite number"
        )
    values = returns_df.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values[~np.isnan(values)])):
        n_inf = int(np.sum(np.isinf(values)))
        raise ValidationError(
            f"returns_df contains {n_inf} infinite value(s). An infinity passes "
            "straight through the weighted matrix multiply below and poisons "
            "every portfolio-level metric derived from it."
        )
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValidationError(f"weights must sum to 1.0, got {w.sum():.4f}")

    # Matrix multiply: (n_days, n_assets) @ (n_assets,) → (n_days,)
    return pd.Series(returns_df.values @ w, index=returns_df.index, name="Portfolio")


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
        "annualized_return": round(annual_ret, 6),
        "annualized_volatility": round(port_vol, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(
            sortino_ratio(port_returns, risk_free_rate, periods_per_year), 4
        ),
        "max_drawdown": round(max_drawdown(equity_curve), 6),
        "calmar_ratio": round(calmar_ratio(equity_curve, periods_per_year), 4),
        "var_95": round(var_historical(port_returns, 0.95), 6),
        "cvar_95": round(cvar(port_returns, 0.95), 6),
        "total_return": round(float(equity_curve.iloc[-1] - 1), 6),
        "tickers": list(returns_df.columns),
        "weights": w.tolist(),
    }

    if benchmark_returns is not None:
        metrics["information_ratio"] = round(
            information_ratio(port_returns, benchmark_returns, periods_per_year), 4
        )

    logger.debug(
        "[portfolio] metrics  return=%.2f%%  vol=%.2f%%  sharpe=%.3f  sortino=%.3f  maxdd=%.2f%%",
        metrics["annualized_return"] * 100,
        metrics["annualized_volatility"] * 100,
        metrics["sharpe_ratio"],
        metrics["sortino_ratio"],
        metrics["max_drawdown"] * 100,
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
    interval: str = "1d",
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

    # PER TICKER FIRST, then assemble. Building the matrix first and
    # calling pct_change() on it is not the same computation: the union
    # index puts a NaN wherever one name lacks a bar, and DataFrame
    # .pct_change(fill_method=None) still defaults to fill_method="pad", so that missing
    # price is FORWARD-FILLED and the name is credited with a 0.00% return
    # on a day it did not trade. Measured on a two-name panel with one
    # halted day:
    #
    #     per-ticker   01-02 A 0.010000  B 0.040000
    #                  01-04 A 0.009804  B 0.057692
    #     assemble-first
    #                  01-03 A 0.009901  B 0.000000   <-- fabricated
    #
    # A fabricated zero biases covariance, correlation and PCA toward
    # understating a halted name's volatility and its correlation to
    # everything else -- and this panel feeds the portfolio optimizer, the
    # research correlation and PCA tools, and fetch_returns_panel. It also
    # emitted a FutureWarning, so pandas dropping the pad would have
    # changed every one of those results silently.
    #
    # Same three steps as modeling.dataset.alignment.build_returns_panel,
    # which got this right; not imported because portfolio/ sits below
    # modeling/ and the dependency would run the wrong way.
    returns = {
        ticker: df["Close"].pct_change(fill_method=None)
        for ticker, df in zip(tickers, dfs)
    }
    return pd.DataFrame(returns).dropna(how="any")


def fetch_returns_sync(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Synchronous wrapper around fetch_returns_async."""
    return asyncio.run(fetch_returns_async(tickers, start_date, end_date, interval))


async def fetch_ohlcv_panel_async(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch full OHLCV for multiple tickers concurrently. One network
    round-trip per ticker, fully async — same concurrency pattern as
    fetch_returns_async, but returns the complete per-ticker DataFrame
    (Open/High/Low/Close/Volume) instead of collapsing to a single Close-
    based returns column. For callers that only need Close-derived returns
    (correlation, optimization, Monte Carlo), fetch_returns_async/
    fetch_returns_sync above is the right, cheaper choice; this exists for
    callers that also need Volume/OHLC (e.g. a portfolio simulation's ADV/
    volatility-based transaction cost model), which a returns-only frame
    can't supply.
    """
    from standard_quant_tools.data.factory import DataFactory

    provider = DataFactory.get_provider()

    tasks = [
        provider.get_ohlcv_async(ticker, start_date, end_date, interval)
        for ticker in tickers
    ]
    dfs = await asyncio.gather(*tasks)
    return dict(zip(tickers, dfs))


def fetch_ohlcv_panel_sync(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper around fetch_ohlcv_panel_async."""
    return asyncio.run(fetch_ohlcv_panel_async(tickers, start_date, end_date, interval))
