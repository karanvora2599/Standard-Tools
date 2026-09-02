"""
Stock Screener — filter a universe of tickers by fundamental and technical criteria.

Design:
  - Async-first: all data fetching is concurrent via asyncio.gather.
  - For large universes (> 20 tickers), a ProcessPoolExecutor splits the ticker
    list into batches so multiple CPU cores drive independent event loops in
    parallel, bypassing the GIL for the full pipeline (fetch + compute).
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
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import pandas as pd

from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import rsi as calc_rsi
from standard_quant_tools.indicators.trend import sma as calc_sma
from standard_quant_tools.validation import last_finite

_VALID_FILTER_KEYS = frozenset(
    (
        "pe_ratio_max",
        "pb_ratio_max",
        "debt_equity_max",
        "roe_min",
        "profit_margin_min",
        "div_yield_min",
        "market_cap_min",
        "rsi_max",
        "rsi_min",
        "price_above_sma",
        "price_below_sma",
        "beta_max",
        "beta_min",
    )
)


# Filters whose value is a window LENGTH in bars, not a threshold: these
# must be positive integers, and are passed straight to an indicator.
_WINDOW_FILTERS = frozenset(("price_above_sma", "price_below_sma"))

# Default minimum overlap with the benchmark before a beta filter will act on
# an estimate. A beta from a handful of bars is noise, and this module does
# not merely REPORT beta -- it makes an include/exclude decision from it. See
# _fetch_ticker_data's beta branch for why a too-short overlap is an error
# rather than a value.
#
# 20 is a judgment call, not a mathematical bound, so it is a DEFAULT rather
# than a fixed rule: callers screening weekly bars, or deliberately hunting
# recent listings, have a legitimate reason to lower it. Override per call via
# the min_beta_obs argument on screen_stocks / screen_stocks_async.
DEFAULT_MIN_BETA_OBS = 20

# The one value that is NOT a matter of taste. calculate_beta returns its
# all-zero sentinel below two overlapping points, and that sentinel is
# indistinguishable from a real beta of 0.0 -- which is the bug this floor
# exists to close. Allowing a floor under 2 would reopen it, so the override
# is bounded rather than free.
_ABSOLUTE_MIN_BETA_OBS = 2


# Retained: this was the name the constant shipped under. Kept as an alias so
# an existing import keeps resolving rather than failing at import time.
_MIN_BETA_OBS = DEFAULT_MIN_BETA_OBS


def _validate_min_beta_obs(min_beta_obs: int) -> int:
    """Validate the beta-overlap floor with the same up-front, once-per-call
    discipline the filter values get -- a bad floor should not surface as a
    per-ticker error repeated across the universe."""
    if isinstance(min_beta_obs, bool) or not isinstance(min_beta_obs, int):
        raise ValidationError(
            f"min_beta_obs must be an int, got {type(min_beta_obs).__name__} "
            f"({min_beta_obs!r})"
        )
    if min_beta_obs < _ABSOLUTE_MIN_BETA_OBS:
        raise ValidationError(
            f"min_beta_obs must be >= {_ABSOLUTE_MIN_BETA_OBS}, got {min_beta_obs}. "
            "Below two overlapping observations calculate_beta returns an "
            "all-zero sentinel that is indistinguishable from a real beta of "
            "0.0, so a lower floor would let an unestimable ticker pass a "
            "beta_max screen — the exact failure this floor exists to prevent."
        )
    return min_beta_obs


def _validate_filter_keys(filters: Dict[str, Any]) -> None:
    unknown = set(filters) - _VALID_FILTER_KEYS
    if unknown:
        raise ValidationError(
            f"Unknown filter key(s): {sorted(unknown)}. Valid keys: "
            f"{sorted(_VALID_FILTER_KEYS)}"
        )


def _validate_filter_values(filters: Dict[str, Any]) -> None:
    """
    Validate the filter VALUES, not just their names.

    Only the keys used to be checked, and every consequence of a bad value
    surfaced per ticker rather than once:

      - NaN was the dangerous one. NaN fails every comparison, so
        `last_rsi > filters["rsi_max"]` is False for every ticker and an
        "oversold" screen silently became a no-op that admitted RSI 100.
        A filter that rejects nothing looks exactly like a filter nothing
        failed.
      - A wrong type or an out-of-range window (rsi_max="fifty",
        price_above_sma=-5) raised inside the per-ticker try/except, so one
        malformed filter came back as N identical "error" entries across the
        universe with no statement that the FILTER was the problem.

    Checked once, up front, for both the async and process-pool entry points.
    """
    for key, value in filters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                f"filter {key!r} must be a number, got {type(value).__name__} "
                f"({value!r}). A non-numeric bound cannot be compared against "
                "the computed value and would fail once per ticker instead of "
                "once here."
            )
        if not math.isfinite(float(value)):
            raise ValidationError(
                f"filter {key!r} must be finite, got {value!r}. NaN compares "
                "False against everything, so it does not restrict the screen "
                "— it silently disables it, and every ticker passes. An "
                "infinite bound is accepted by no comparison that is "
                "meaningful here either."
            )
        if key in _WINDOW_FILTERS:
            if float(value) != int(value) or int(value) < 1:
                raise ValidationError(
                    f"filter {key!r} is a window length in bars and must be a "
                    f"positive whole number, got {value!r}"
                )


# ── Per-ticker async evaluation ───────────────────────────────────────────────


async def _fetch_ticker_data(
    provider,
    ticker: str,
    start_date: str,
    end_date: str,
    filters: Dict[str, Any],
    spy_df: Optional[pd.DataFrame] = None,
    min_beta_obs: int = DEFAULT_MIN_BETA_OBS,
) -> Tuple[str, str, Any]:
    """
    Fetch all required data for one ticker and evaluate filters.

    Returns a 3-tuple (status, ticker, payload):
      ("passed", ticker, row_dict) — passed every filter.
      ("failed_filter", ticker, filter_key) — failed a specific filter
          condition (genuine rejection: the ticker's own data didn't meet
          the requested bound).
      ("error", ticker, error_message) — a data-fetch/compute exception
          (network error, missing data, bad response, etc.).

    A ticker that raised an exception must never be indistinguishable from
    one that simply failed a filter — both used to collapse to `None`,
    which meant a screener run silently returning zero results couldn't
    tell you whether every ticker was genuinely rejected or every ticker's
    data fetch was broken.
    """
    row: Dict[str, Any] = {"ticker": ticker}

    needs_fundamentals = any(
        k in filters
        for k in (
            "pe_ratio_max",
            "pb_ratio_max",
            "debt_equity_max",
            "roe_min",
            "profit_margin_min",
            "div_yield_min",
            "market_cap_min",
        )
    )
    needs_ohlcv = any(
        k in filters
        for k in (
            "rsi_max",
            "rsi_min",
            "price_above_sma",
            "price_below_sma",
            "beta_max",
            "beta_min",
        )
    )

    try:
        if needs_fundamentals:
            ratios = await asyncio.get_event_loop().run_in_executor(
                None, provider.get_financial_ratios, ticker
            )
            row.update(
                {
                    "forward_pe": ratios.forward_pe,
                    "price_to_book": ratios.price_to_book,
                    "debt_to_equity": ratios.debt_to_equity,
                    "return_on_equity": ratios.return_on_equity,
                    "profit_margins": ratios.profit_margins,
                    "dividend_yield": ratios.dividend_yield,
                    "market_cap": ratios.market_cap,
                }
            )

            if "pe_ratio_max" in filters and (
                ratios.forward_pe is None or ratios.forward_pe > filters["pe_ratio_max"]
            ):
                return ("failed_filter", ticker, "pe_ratio_max")
            if "pb_ratio_max" in filters and (
                ratios.price_to_book is None
                or ratios.price_to_book > filters["pb_ratio_max"]
            ):
                return ("failed_filter", ticker, "pb_ratio_max")
            if "debt_equity_max" in filters and (
                ratios.debt_to_equity is None
                or ratios.debt_to_equity > filters["debt_equity_max"]
            ):
                return ("failed_filter", ticker, "debt_equity_max")
            if "roe_min" in filters and (
                ratios.return_on_equity is None
                or ratios.return_on_equity < filters["roe_min"]
            ):
                return ("failed_filter", ticker, "roe_min")
            if "profit_margin_min" in filters and (
                ratios.profit_margins is None
                or ratios.profit_margins < filters["profit_margin_min"]
            ):
                return ("failed_filter", ticker, "profit_margin_min")
            if "div_yield_min" in filters and (
                ratios.dividend_yield is None
                or ratios.dividend_yield < filters["div_yield_min"]
            ):
                return ("failed_filter", ticker, "div_yield_min")
            if "market_cap_min" in filters and (
                ratios.market_cap is None
                or ratios.market_cap < filters["market_cap_min"]
            ):
                return ("failed_filter", ticker, "market_cap_min")

        if needs_ohlcv:
            df = await provider.get_ohlcv_async(ticker, start_date, end_date)
            close = df["Close"]
            last_close = float(close.iloc[-1])
            row["last_close"] = round(last_close, 2)

            if "rsi_max" in filters or "rsi_min" in filters:
                rsi_vals = calc_rsi(close, 14)
                last_rsi = last_finite(rsi_vals, "rsi_vals")
                row["rsi_14"] = round(last_rsi, 2)
                if "rsi_max" in filters and last_rsi > filters["rsi_max"]:
                    return ("failed_filter", ticker, "rsi_max")
                if "rsi_min" in filters and last_rsi < filters["rsi_min"]:
                    return ("failed_filter", ticker, "rsi_min")

            if "price_above_sma" in filters:
                n = int(filters["price_above_sma"])
                sma_vals = calc_sma(close, n)
                if last_close <= last_finite(sma_vals, "sma_vals"):
                    return ("failed_filter", ticker, "price_above_sma")
                row[f"sma_{n}"] = round(last_finite(sma_vals, "sma_vals"), 2)

            if "price_below_sma" in filters:
                n = int(filters["price_below_sma"])
                sma_vals = calc_sma(close, n)
                if last_close >= last_finite(sma_vals, "sma_vals"):
                    return ("failed_filter", ticker, "price_below_sma")
                row[f"sma_{n}"] = round(last_finite(sma_vals, "sma_vals"), 2)

            if "beta_max" in filters or "beta_min" in filters:
                _spy = (
                    spy_df
                    if spy_df is not None
                    else await provider.get_ohlcv_async("SPY", start_date, end_date)
                )
                asset_ret = close.pct_change(fill_method=None).dropna()
                spy_ret = _spy["Close"].pct_change(fill_method=None).dropna()
                # calculate_beta returns {"alpha": 0, "beta": 0, "r_squared": 0}
                # when fewer than 2 points overlap -- a sentinel, but an
                # indistinguishable one, because 0.0 is also a perfectly
                # legitimate beta. This module then FILTERED on it: a ticker
                # whose history did not overlap the benchmark at all reported
                # beta 0.0 and PASSED beta_max=0.5, so "beta could not be
                # estimated" was silently read as "this is a very low-beta
                # stock" -- backwards for the defensive screen that bound
                # exists to express.
                #
                # The overlap is checked here rather than inferred from the
                # returned value, since the sentinel and a real answer are the
                # same number.
                overlap = len(asset_ret.index.intersection(spy_ret.index))
                if overlap < min_beta_obs:
                    return (
                        "error",
                        ticker,
                        f"beta not estimable: only {overlap} bar(s) overlap the "
                        f"benchmark over {start_date}..{end_date} (need "
                        f"{min_beta_obs}). Reported as an error rather than a "
                        "beta of 0.0, which would pass a beta_max screen.",
                    )
                stats = calculate_beta(asset_ret, spy_ret)
                beta = stats["beta"]
                row["beta"] = round(beta, 4)
                if "beta_max" in filters and beta > filters["beta_max"]:
                    return ("failed_filter", ticker, "beta_max")
                if "beta_min" in filters and beta < filters["beta_min"]:
                    return ("failed_filter", ticker, "beta_min")

    except Exception as exc:
        return ("error", ticker, str(exc))

    return ("passed", ticker, row)


# ── Async screener (single process) ──────────────────────────────────────────


async def screen_stocks_async(
    tickers: List[str],
    filters: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    min_beta_obs: int = DEFAULT_MIN_BETA_OBS,
) -> pd.DataFrame:
    """
    Async screener: evaluates all tickers concurrently via asyncio.gather.

    Args:
        tickers:    List of ticker symbols to screen.
        filters:    Dict of filter criteria (see module docstring).
        start_date: Historical start for technical indicators (default: 1 year ago).
        end_date:   Historical end (default: today).
        sort_by:    Optional column to sort results by.
        ascending:  Sort direction.
        min_beta_obs: Minimum bars a ticker must share with the benchmark
            before a beta_max/beta_min filter will act on its estimate;
            below it the ticker is reported as an error rather than given a
            beta of 0.0 that would pass a beta_max screen. Defaults to
            DEFAULT_MIN_BETA_OBS (20), a judgment call rather than a
            mathematical bound — lower it for weekly bars or a deliberate
            recent-listing screen. Must be >= 2, below which calculate_beta
            returns a sentinel indistinguishable from a real beta.

    Returns:
        pd.DataFrame with one row per passing ticker, sorted if requested.
        df.attrs["failed_filters"] maps ticker -> the specific filter key it
        failed (genuine rejection). df.attrs["failed_tickers"] maps ticker
        -> error message for a data-fetch/compute exception — kept
        separate from failed_filters so a broken data fetch is never
        indistinguishable from a ticker that simply didn't meet the bar.

    Raises:
        ValidationError: filters contains an unrecognized key, a filter value
            is non-numeric/non-finite/an invalid window, or min_beta_obs is
            below 2.
    """
    _validate_filter_keys(filters)
    _validate_filter_values(filters)
    min_beta_obs = _validate_min_beta_obs(min_beta_obs)
    end: str = end_date or datetime.date.today().isoformat()
    start: str = (
        start_date or (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    )
    logger.debug(
        "[screener] tickers=%d  filters=%s  %s → %s",
        len(tickers),
        list(filters.keys()),
        start,
        end,
    )

    provider = DataFactory.get_provider()

    # Pre-fetch SPY once for the whole batch when beta filters are active
    spy_df: Optional[pd.DataFrame] = None
    if "beta_max" in filters or "beta_min" in filters:
        try:
            spy_df = await provider.get_ohlcv_async("SPY", start, end)
        except Exception:
            spy_df = None

    tasks = [
        _fetch_ticker_data(provider, ticker, start, end, filters, spy_df, min_beta_obs)
        for ticker in tickers
    ]
    raw = await asyncio.gather(*tasks)

    passing = [payload for status, _, payload in raw if status == "passed"]
    failed_filters = {
        t: reason for status, t, reason in raw if status == "failed_filter"
    }
    failed_tickers = {t: reason for status, t, reason in raw if status == "error"}
    logger.debug(
        "[screener] passed=%d  failed_filter=%d  error=%d / %d (%.0f%% passed)",
        len(passing),
        len(failed_filters),
        len(failed_tickers),
        len(tickers),
        100 * len(passing) / len(tickers) if tickers else 0,
    )
    if not passing:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(passing).set_index("ticker")
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

    df.attrs["failed_filters"] = failed_filters
    df.attrs["failed_tickers"] = failed_tickers
    return df


# ── Module-level batch worker (picklable for ProcessPoolExecutor) ─────────────


def _screen_batch(args: tuple) -> pd.DataFrame:
    """
    Worker: screen a batch of tickers in a child process.
    Each worker runs its own asyncio event loop so there is no shared state.

    Every tunable the parent resolved has to travel in this tuple. A
    parameter left out of it does not fail — it silently reverts to its
    default inside the child, so the same call would screen differently at
    n_workers=1 than at n_workers=8. Unpacked strictly (no defaults) so
    adding a parameter without threading it here is an immediate TypeError
    rather than a quiet divergence between the two paths.
    """
    tickers, filters, start_date, end_date, min_beta_obs = args
    return asyncio.run(
        screen_stocks_async(
            tickers, filters, start_date, end_date, min_beta_obs=min_beta_obs
        )
    )


# ── Public sync entry point ───────────────────────────────────────────────────


def screen_stocks(
    tickers: List[str],
    filters: Dict[str, Any],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    n_workers: Optional[int] = None,
    min_beta_obs: int = DEFAULT_MIN_BETA_OBS,
) -> pd.DataFrame:
    """
    Screen a universe of tickers against fundamental and technical filters.

    For universes with more than 20 tickers, the ticker list is split across
    multiple worker processes (one asyncio event loop per process) to bypass
    the GIL and saturate available CPU cores.  For small universes the
    overhead of spawning processes outweighs the benefit, so a single
    asyncio.gather call is used instead.

    Args:
        tickers:   Universe of tickers to screen.
        filters:   Filter criteria dict (see module docstring).
        start_date, end_date: Date range for technical filters.
        sort_by:   Column to rank results by.
        ascending: Sort direction.
        n_workers: Override process count. Pass 1 to force single-process mode.
                   Defaults to cpu_count for large universes, 1 for small ones.
        min_beta_obs: Minimum bars a ticker must share with the benchmark
                   before a beta filter acts on its estimate (default
                   DEFAULT_MIN_BETA_OBS = 20, minimum 2). Applied identically
                   in single- and multi-process mode.

    Returns:
        pd.DataFrame with one row per passing ticker, sorted if requested.
        df.attrs carries "failed_filters" (ticker -> filter key it failed),
        "failed_tickers" (ticker -> error message for a data-fetch/compute
        exception), and "failed_batches" (one message per worker-process
        batch that raised before returning any per-ticker result, naming
        the tickers it carried — n_workers > 1 only). Every ticker in a
        crashed batch also appears individually in "failed_tickers", so no
        symbol can go missing from all three without a trace. A ticker's
        exception is never indistinguishable from a genuine filter
        rejection.

    Raises:
        ValidationError: filters contains an unrecognized key.

    Example (large universe)::

        result = screen_stocks(
            tickers=sp500_list,   # 500 tickers
            filters={"pe_ratio_max": 25, "rsi_max": 50},
            sort_by="rsi_14",
            n_workers=8,
        )
    """
    _validate_filter_keys(filters)
    _validate_filter_values(filters)
    min_beta_obs = _validate_min_beta_obs(min_beta_obs)
    end: str = end_date or datetime.date.today().isoformat()
    start: str = (
        start_date or (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    )

    n = len(tickers)

    # Determine effective worker count
    if n_workers is None:
        # Single process is faster for small universes (no spawn overhead)
        n_workers = 1 if n <= 20 else min(os.cpu_count() or 4, max(n // 10, 2))
    logger.debug(
        "[screener:screen_stocks] universe=%d  workers=%d  filters=%s",
        n,
        n_workers,
        list(filters.keys()),
    )

    if n_workers <= 1:
        result = asyncio.run(
            screen_stocks_async(
                tickers, filters, start, end, sort_by, ascending, min_beta_obs
            )
        )
        result.attrs.setdefault("failed_batches", [])
        return result

    # Split tickers into roughly equal batches across workers
    batch_size = (n + n_workers - 1) // n_workers
    batches = [tickers[i : i + batch_size] for i in range(0, n, batch_size)]

    batch_results: List[pd.DataFrame] = []
    failed_batches: List[str] = []
    failed_tickers_from_batches: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # The batch each future came from is kept alongside it. Recording
        # only the exception string said THAT a batch died but not WHICH
        # tickers went with it, so those tickers were absent from the
        # results, absent from failed_filters and absent from
        # failed_tickers -- indistinguishable from never having been asked
        # for. The screener's whole contract is that a rejection, an error
        # and a pass are told apart.
        futures = {
            executor.submit(
                _screen_batch, (batch, filters, start, end, min_beta_obs)
            ): batch
            for batch in batches
        }
        for future, batch in futures.items():
            try:
                batch_results.append(future.result())
            except Exception as exc:
                failed_batches.append(
                    f"batch of {len(batch)} ticker(s) {batch} failed: {exc}"
                )
                # Also recorded per ticker, so a caller checking one symbol
                # does not have to parse the batch list to find out what
                # happened to it.
                for t in batch:
                    failed_tickers_from_batches[t] = f"worker batch failed: {exc}"

    failed_filters: Dict[str, str] = {}
    failed_tickers: Dict[str, str] = dict(failed_tickers_from_batches)
    for df in batch_results:
        failed_filters.update(df.attrs.get("failed_filters", {}))
        failed_tickers.update(df.attrs.get("failed_tickers", {}))

    non_empty = [df for df in batch_results if not df.empty]
    if not non_empty:
        combined = pd.DataFrame()
    else:
        combined = pd.concat(non_empty)
        if sort_by and sort_by in combined.columns:
            combined = combined.sort_values(sort_by, ascending=ascending)

    combined.attrs["failed_filters"] = failed_filters
    combined.attrs["failed_tickers"] = failed_tickers
    combined.attrs["failed_batches"] = failed_batches
    return combined
