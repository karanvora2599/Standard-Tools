"""
Multi-ticker signal-panel backtest.

Bring your own signal matrix (one column per ticker, values in {-1, 0, 1})
computed however you like — this module does not generate or assume any
particular alpha model. It backtests each ticker's column with the existing
`run_strategy` engine (full C++ speed where available) and combines the
realized per-ticker returns into portfolio-level metrics via the existing
`portfolio` module. No new backtest or metric math is introduced here.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.portfolio import build_portfolio, portfolio_metrics

logger = logging.getLogger(__name__)


def run_signal_panel_backtest(
    price_data: Dict[str, pd.DataFrame],
    signal_panel: pd.DataFrame,
    weights: Optional[Union[List[float], Dict[str, float]]] = None,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    benchmark_returns: Optional[pd.Series] = None,
    include_trade_log: bool = False,
    fill_price: str = "close",
) -> Dict[str, Any]:
    """
    Backtest a pre-computed signal panel across a ticker universe.

    Args:
        price_data:   Dict mapping ticker -> OHLCV DataFrame (must contain 'Close'
                      and cover signal_panel's date range for that ticker).
        signal_panel: DataFrame indexed by date, one column per ticker, values
                      in {-1, 0, 1} (short/flat/long). Column set defines the
                      universe for this backtest.
        weights:      Portfolio weights, either a list matching signal_panel's
                      column order or a {ticker: weight} dict. Must sum to 1.0.
                      Defaults to equal weight across signal_panel's columns.
        initial_capital, commission_pct, slippage_pct: passed through to
                      run_strategy for every ticker.
        benchmark_returns: optional benchmark return series for portfolio-level
                      Information Ratio (passed through to portfolio_metrics).
        include_trade_log: passed through to run_strategy per ticker.
        fill_price:   "close" (default) or "next_open" — passed through to
                      run_strategy for every ticker; see its docstring.

    Returns:
        {
          "tickers": [...],
          "per_ticker": {ticker: run_strategy(...) result dict, ...},
          "portfolio_returns": pd.Series (daily weighted portfolio returns),
          "portfolio_metrics": portfolio_metrics(...) result dict,
        }

    Note: per-ticker equity curves are aligned to their common date range
    (inner join) before being combined into the portfolio — a ticker whose
    price_data doesn't cover the full signal_panel range will shrink the
    portfolio's effective date range accordingly.
    """
    tickers = list(signal_panel.columns)
    missing = [t for t in tickers if t not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

    logger.debug("[signal_panel] tickers=%d  bars=%d", len(tickers), len(signal_panel))

    per_ticker_results: Dict[str, Any] = {}
    returns_cols: Dict[str, pd.Series] = {}

    for ticker in tickers:
        result = run_strategy(
            price_data[ticker],
            signal_panel[ticker],
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            include_trade_log=include_trade_log,
            fill_price=fill_price,
        )
        per_ticker_results[ticker] = result
        # Realized per-bar strategy return, recovered from the equity curve —
        # mathematically identical to the internal strategy_returns series,
        # without needing run_strategy to expose it separately.
        returns_cols[ticker] = result["equity_curve"].pct_change().fillna(0.0)

    returns_df = pd.DataFrame(returns_cols).dropna(how="any")

    # The docstring has always required weights to cover every ticker and sum
    # to 1.0, but nothing enforced it: a dict missing a ticker raised a bare
    # KeyError naming only the ticker, a wrong-length list silently
    # misaligned weights against columns, and weights summing to anything
    # other than 1.0 produced a scaled portfolio that still looked valid.
    if weights is None:
        w: List[float] = [1.0 / len(tickers)] * len(tickers)
    elif isinstance(weights, dict):
        missing_w = [t for t in tickers if t not in weights]
        if missing_w:
            raise ValidationError(f"weights is missing entries for: {missing_w}")
        extra_w = [t for t in weights if t not in tickers]
        if extra_w:
            raise ValidationError(
                f"weights has entries for tickers not in signal_panel: {extra_w}"
            )
        w = [float(weights[t]) for t in tickers]
    else:
        w = [float(x) for x in weights]
        if len(w) != len(tickers):
            raise ValidationError(
                f"weights length ({len(w)}) must match the number of tickers "
                f"({len(tickers)}) in signal_panel"
            )
    total_w = sum(w)
    if abs(total_w - 1.0) > 1e-6:
        raise ValidationError(f"weights must sum to 1.0, got {total_w:.6f}")

    metrics = portfolio_metrics(returns_df, w, benchmark_returns=benchmark_returns)
    portfolio_returns = build_portfolio(returns_df, w)

    logger.debug(
        "[signal_panel] portfolio  sharpe=%.3f  return=%.2f%%  maxdd=%.2f%%",
        metrics["sharpe_ratio"],
        metrics["annualized_return"] * 100,
        metrics["max_drawdown"] * 100,
    )

    return {
        "tickers": tickers,
        "per_ticker": per_ticker_results,
        "portfolio_returns": portfolio_returns,
        "portfolio_metrics": metrics,
    }
