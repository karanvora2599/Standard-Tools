"""
Stock Screener — filter a universe of tickers by fundamental and technical criteria.

Design:
  - Async-first: all data fetching is concurrent via asyncio.gather.
  - Filters are expressed as a plain dict with intuitive key names.
  - Returns a ranked pd.DataFrame so agents get clean structured output.

Supported filter keys
─────────────────────
Fundamental:
  pe_ratio_max        float   Forward P/E upper bound
  pb_ratio_max        float   Price-to-Book upper bound
  debt_equity_max     float   Debt-to-Equity upper bound
  roe_min             float   Return on Equity lower bound (as decimal, e.g. 0.15)
  profit_margin_min   float   Profit margin lower bound (as decimal)
  div_yield_min       float   Dividend yield lower bound (as decimal)
  market_cap_min      int     Market-cap lower bound (USD)

Technical (computed over start_date → end_date):
  rsi_max             float   RSI(14) upper bound — screen for oversold
  rsi_min             float   RSI(14) lower bound — screen for overbought
  price_above_sma     int     Close must be above SMA(N)
  price_below_sma     int     Close must be below SMA(N)
  beta_max            float   Beta vs SPY upper bound
  beta_min            float   Beta vs SPY lower bound
"""

import asyncio
import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators.momentum import rsi as calc_rsi
from standard_quant_tools.indicators.trend import sma as calc_sma
from standard_quant_tools.analysis.regression import calculate_beta


async def _fetch_ticker_data(
    provider,
    ticker: str,
    start_date: str,
    end_date: str,
    filters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Fetch all required data for one ticker and evaluate filters.
    Returns a result dict if the ticker passes, None if it fails any filter.
    """
    row: Dict[str, Any] = {'ticker': ticker}

    # ── Fundamental data ──────────────────────────────────────────────────
    needs_fundamentals = any(k in filters for k in (
        'pe_ratio_max', 'pb_ratio_max', 'debt_equity_max',
        'roe_min', 'profit_margin_min', 'div_yield_min', 'market_cap_min'
    ))
    needs_ohlcv = any(k in filters for k in (
        'rsi_max', 'rsi_min', 'price_above_sma', 'price_below_sma',
        'beta_max', 'beta_min'
    ))

    try:
        if needs_fundamentals:
            ratios = await asyncio.get_event_loop().run_in_executor(
                None, provider.get_financial_ratios, ticker
            )
            row.update({
                'forward_pe': ratios.forward_pe,
                'price_to_book': ratios.price_to_book,
                'debt_to_equity': ratios.debt_to_equity,
                'return_on_equity': ratios.return_on_equity,
                'profit_margins': ratios.profit_margins,
                'dividend_yield': ratios.dividend_yield,
                'market_cap': ratios.market_cap,
            })

            if 'pe_ratio_max' in filters and (
                ratios.forward_pe is None or ratios.forward_pe > filters['pe_ratio_max']
            ):
                return None
            if 'pb_ratio_max' in filters and (
                ratios.price_to_book is None or ratios.price_to_book > filters['pb_ratio_max']
            ):
                return None
            if 'debt_equity_max' in filters and (
                ratios.debt_to_equity is None or ratios.debt_to_equity > filters['debt_equity_max']
            ):
                return None
            if 'roe_min' in filters and (
                ratios.return_on_equity is None or ratios.return_on_equity < filters['roe_min']
            ):
                return None
            if 'profit_margin_min' in filters and (
                ratios.profit_margins is None or ratios.profit_margins < filters['profit_margin_min']
            ):
                return None
            if 'div_yield_min' in filters and (
                ratios.dividend_yield is None or ratios.dividend_yield < filters['div_yield_min']
            ):
                return None
            if 'market_cap_min' in filters and (
                ratios.market_cap is None or ratios.market_cap < filters['market_cap_min']
            ):
                return None

        if needs_ohlcv:
            df = await provider.get_ohlcv_async(ticker, start_date, end_date)
            close = df['Close']
            last_close = float(close.iloc[-1])
            row['last_close'] = round(last_close, 2)

            if 'rsi_max' in filters or 'rsi_min' in filters:
                rsi_vals = calc_rsi(close, 14)
                last_rsi = float(rsi_vals.dropna().iloc[-1])
                row['rsi_14'] = round(last_rsi, 2)
                if 'rsi_max' in filters and last_rsi > filters['rsi_max']:
                    return None
                if 'rsi_min' in filters and last_rsi < filters['rsi_min']:
                    return None

            if 'price_above_sma' in filters:
                n = int(filters['price_above_sma'])
                sma_vals = calc_sma(close, n)
                if last_close <= float(sma_vals.dropna().iloc[-1]):
                    return None
                row[f'sma_{n}'] = round(float(sma_vals.dropna().iloc[-1]), 2)

            if 'price_below_sma' in filters:
                n = int(filters['price_below_sma'])
                sma_vals = calc_sma(close, n)
                if last_close >= float(sma_vals.dropna().iloc[-1]):
                    return None
                row[f'sma_{n}'] = round(float(sma_vals.dropna().iloc[-1]), 2)

            if 'beta_max' in filters or 'beta_min' in filters:
                spy_df = await provider.get_ohlcv_async('SPY', start_date, end_date)
                asset_ret = close.pct_change().dropna()
                spy_ret = spy_df['Close'].pct_change().dropna()
                stats = calculate_beta(asset_ret, spy_ret)
                beta = stats['beta']
                row['beta'] = round(beta, 4)
                if 'beta_max' in filters and beta > filters['beta_max']:
                    return None
                if 'beta_min' in filters and beta < filters['beta_min']:
                    return None

    except Exception:
        return None

    return row


async def screen_stocks_async(
    tickers: List[str],
    filters: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Async screener: evaluates all tickers concurrently and returns a DataFrame
    of those passing all filters.

    Args:
        tickers:    List of ticker symbols to screen.
        filters:    Dict of filter criteria (see module docstring).
        start_date: Historical start for technical indicators (default: 1 year ago).
        end_date:   Historical end (default: today).
        sort_by:    Optional column to sort results by.
        ascending:  Sort direction.

    Returns:
        pd.DataFrame with one row per passing ticker, sorted if requested.
    """
    end: str = end_date or datetime.date.today().isoformat()
    start: str = start_date or (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

    provider = DataFactory.get_provider()

    tasks = [
        _fetch_ticker_data(provider, ticker, start, end, filters)
        for ticker in tickers
    ]
    results = await asyncio.gather(*tasks)

    passing = [r for r in results if r is not None]
    if not passing:
        return pd.DataFrame()

    df = pd.DataFrame(passing).set_index('ticker')

    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    return df


def screen_stocks(
    tickers: List[str],
    filters: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """Synchronous wrapper around screen_stocks_async."""
    return asyncio.run(
        screen_stocks_async(tickers, filters, start_date, end_date, sort_by, ascending)
    )
