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


_CALENDAR_POLICIES = frozenset({"hold", "flat", "error"})


def _align_signal_to_calendar(
    signal: pd.Series,
    price_index: pd.Index,
    policy: str,
    ticker: str,
) -> pd.Series:
    """
    Reindex one ticker's signal onto its own full price calendar.

    Without this, run_strategy's price/signal intersection silently DELETES
    the trading days a sparse signal does not mention, compressing a month of
    price movement into one "bar" — see run_signal_panel_backtest's own
    docstring for the measured 32x volatility distortion.

    "hold" carries the last signal forward, which is what a rebalance
    schedule means: a monthly signal is a position held through the month,
    not a position that exists on one day. Dates BEFORE the first signal are
    filled flat rather than back-filled, since no view had been expressed
    yet and back-filling would be look-ahead.
    """
    signal = signal.sort_index()
    on_calendar = signal.reindex(price_index)
    if not on_calendar.isna().any():
        return on_calendar.astype(float)

    if policy == "error":
        n_missing = int(on_calendar.isna().sum())
        raise ValidationError(
            f"{ticker}: signal covers {len(signal.dropna())} of "
            f"{len(price_index)} trading dates, leaving {n_missing} bars "
            "without a signal. With signal_calendar_policy='error' this is "
            "refused rather than guessed at. Use 'hold' to carry the last "
            "signal forward (a rebalance schedule) or 'flat' to be out of "
            "the market between signal dates."
        )
    if policy == "hold":
        # ffill only: leading NaNs stay NaN here and become 0.0 below, so a
        # position is never held before the model expressed a view.
        on_calendar = on_calendar.ffill()
    return on_calendar.fillna(0.0).astype(float)


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
    signal_calendar_policy: str = "hold",
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
        signal_calendar_policy: what to do when a ticker's signal series is
                      sparser than its price series — "hold" (default: carry
                      the last signal forward, the natural reading of a
                      rebalance schedule), "flat" (0.0 between signal dates,
                      i.e. in the market only on dates that have a signal),
                      or "error" (refuse). See the calendar note below.

    Calendar preservation:
        run_strategy intersects price dates with signal dates and then takes
        pct_change() over WHAT REMAINS, so a signal series sparser than the
        price series does not read as "hold" — the intervening trading days
        disappear from the price axis entirely and the bars either side
        become adjacent. A monthly signal against daily prices turns
        Jan 31 -> Feb 28 into a single "bar" carrying a month of price
        movement.

        Measured on a 120-bar daily series driven by the same exposure, once
        with a daily signal and once with the same signal sampled monthly:
        annualized volatility 0.0241 against 0.7735 — a 32x distortion of
        risk, from identical prices. Total return can still look right,
        which is what made it easy to miss; per-bar volatility, Sharpe and
        drawdown are all wrong.

        Every ticker's signal is therefore reindexed onto that ticker's own
        full price calendar before the backtest runs. The agent wrapper
        already did this; the library function that it and every direct
        caller sit on did not.

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
    if not tickers:
        raise ValidationError(
            "signal_panel has no columns — there is no universe to backtest. "
            "(The default equal weighting would otherwise divide by zero.)"
        )
    if signal_calendar_policy not in _CALENDAR_POLICIES:
        raise ValidationError(
            f"signal_calendar_policy must be one of {sorted(_CALENDAR_POLICIES)}, "
            f"got {signal_calendar_policy!r}"
        )

    logger.debug("[signal_panel] tickers=%d  bars=%d", len(tickers), len(signal_panel))

    per_ticker_results: Dict[str, Any] = {}
    returns_cols: Dict[str, pd.Series] = {}

    for ticker in tickers:
        aligned_signal = _align_signal_to_calendar(
            signal_panel[ticker],
            price_data[ticker].index,
            signal_calendar_policy,
            ticker,
        )
        result = run_strategy(
            price_data[ticker],
            aligned_signal,
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
        returns_cols[ticker] = result["equity_curve"].pct_change(fill_method=None).fillna(0.0)

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
