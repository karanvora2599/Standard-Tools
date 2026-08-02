import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from standard_quant_tools.validation import require_finite_array

_VALID_FILL_PRICES = ("close", "next_open", "hl2_exploratory")

# Column order for _cpp_core.batch_run_strategy's flat (num_tests, 11) array
# return -- MUST stay in sync with bindings.cpp's batch_run_strategy binding,
# which writes exactly these 11 columns in exactly this order.
_BATCH_METRIC_COLUMNS = [
    "final_equity",
    "total_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "num_trades",
    "avg_trade_return_pct",
]

# ── Optional C++ fast path ────────────────────────────────────────────────────
from typing import Any as _Any

_cpp_core: _Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_trade_log(
    ref_prices: pd.Series,
    close_prices: pd.Series,
    executed: pd.Series,
    cost_per_unit: float = 0.0,
) -> pd.DataFrame:
    """
    Build a per-trade log reconciled with the equity curve's own P&L.

    entry_price/exit_price use ref_prices — the same reference price series
    run_strategy's return calculation uses: Close[i-1] under
    fill_price="close" (since executed[i] = signals[i-1], a position that
    "appears" in `executed` at event date i actually earns its first
    return over Close[i-1] -> Close[i], so i-1's close is its true economic
    entry/exit point — the review's finding), or Open[i] / (High[i]+Low[i])/2
    directly under "next_open"/"hl2_exploratory", where the two-leg
    decomposition already prices entries/exits at that bar's own reference
    price (no shift needed there).

    return_pct is reduced by 2 * abs(position_size) * cost_per_unit (entry +
    exit — matching the two separate pos_diff-triggered cost deductions
    run_strategy applies to the equity curve for a completed round trip,
    each of magnitude abs(pdiff) * cost_per_unit), or 1 * abs(position_size)
    * cost_per_unit for a position still open at the final bar (a
    synthesized mark-to-market "exit", not a real exit event — the equity
    curve never deducted a second cost for it either). Scaling by
    abs(position_size) matters as much as scaling raw_pnl by it: cost_per_unit
    is a cost *per unit of notional exposure traded*, so a 5x-leveraged
    trade must pay 5x the cost a 1x trade pays, not the same flat amount —
    previously it did not, silently under-costing every leveraged (non-±1)
    SCORE-style position. This does not attempt to reconcile exactly with
    the equity curve for a same-sign *resize* (e.g. 1.0->2.5, a single
    pos_diff event treated here as closing a 1.0-sized trade and opening a
    fresh 2.5-sized one, each independently costed at 2x their own size
    rather than the smaller abs(pdiff)=1.5 the equity curve actually
    charges for that one event) — a known, documented approximation.

    position_size is the actual executed signal value held during the
    trade (e.g. 2.5 for a SCORE signal sized at 2.5x leverage), not just
    its sign — run_strategy's own return calculation multiplies the raw
    price return by this same value (strategy_returns[i] = executed[i] *
    returns[i]), so return_pct must scale with it too, not silently treat
    every trade as if it were exactly 1x/-1x. direction ("long"/"short") is
    still reported as a readable label derived from position_size's sign.

    Vectorized detection of position changes; only iterates over trade
    events (orders-of-magnitude fewer than bars).
    """
    pos_diff = executed.diff()
    pos_diff.iloc[0] = executed.iloc[0]

    trade_event_idx = pos_diff[pos_diff != 0].index
    if len(trade_event_idx) == 0:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "direction",
                "entry_price",
                "exit_price",
                "position_size",
                "return_pct",
            ]
        )

    records: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] = {}

    for date in trade_event_idx:
        ref_price = ref_prices[date]
        new_pos = executed[date]

        if open_trade:
            position_size = open_trade["position_size"]
            entry_price = open_trade["entry_price"]
            raw_pnl = (ref_price - entry_price) / entry_price * position_size
            net_pnl = raw_pnl - 2 * abs(position_size) * cost_per_unit
            records.append(
                {
                    "entry_date": open_trade["entry_date"],
                    "exit_date": date,
                    "direction": "long" if position_size > 0 else "short",
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(ref_price, 4),
                    "position_size": round(position_size, 4),
                    "return_pct": round(net_pnl * 100, 4),
                }
            )
            open_trade = {}

        if new_pos != 0:
            open_trade = {
                "entry_date": date,
                "entry_price": ref_price,
                "position_size": float(new_pos),
            }

    # Close any position still open at the last bar (e.g. buy-and-hold, trend strategies
    # that never exit). Mark the exit date and price as the final bar in the series --
    # equity is always marked to Close regardless of fill_price, so this synthesized
    # "exit" uses Close too, not ref_prices.
    if open_trade:
        last_date = close_prices.index[-1]
        last_price = close_prices.iloc[-1]
        position_size = open_trade["position_size"]
        entry_price = open_trade["entry_price"]
        raw_pnl = (last_price - entry_price) / entry_price * position_size
        net_pnl = (
            raw_pnl - abs(position_size) * cost_per_unit
        )  # entry cost only -- no real exit event occurred
        records.append(
            {
                "entry_date": open_trade["entry_date"],
                "exit_date": last_date,
                "direction": "long" if position_size > 0 else "short",
                "entry_price": round(entry_price, 4),
                "exit_price": round(float(last_price), 4),
                "position_size": round(position_size, 4),
                "return_pct": round(net_pnl * 100, 4),
            }
        )

    return pd.DataFrame(records)


def _compute_trade_stats(trade_log: pd.DataFrame) -> Dict[str, float]:
    if trade_log.empty:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "avg_trade_return_pct": 0.0,
        }

    num_trades = len(trade_log)
    winners = trade_log[trade_log["return_pct"] > 0]
    losers = trade_log[trade_log["return_pct"] <= 0]

    win_rate = len(winners) / num_trades
    gross_profit = float(winners["return_pct"].to_numpy(dtype=float).sum())
    gross_loss = float(np.abs(losers["return_pct"].to_numpy(dtype=float)).sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    return {
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "num_trades": num_trades,
        "avg_trade_return_pct": round(float(trade_log["return_pct"].mean()), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────


def run_strategy(
    price_data: pd.DataFrame,
    signal_series: pd.Series,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    include_trade_log: bool = False,
    fill_price: str = "close",
) -> Dict[str, Any]:
    """
    Vectorized backtesting engine with transaction costs.

    Args:
        price_data: DataFrame with 'Close' column (and 'Open' if fill_price="next_open").
        signal_series: Series of 1 (long), 0 (flat), -1 (short).
        initial_capital: Starting capital.
        commission_pct: Commission per unit of position changed (default 0.1%).
        slippage_pct: Slippage per unit of position changed (default 0.05%).
        include_trade_log: If True, build and return per-trade log.
        fill_price: "close" (default) — a signal known at bar t-1's close is
            assumed filled at that same close, earning bar t's full
            close-to-close return. "next_open" — decomposes each bar into
            an overnight leg (prior close -> this bar's open, priced at
            yesterday's position) and an intraday leg (this bar's open ->
            close, priced at today's position), so an entry only earns its
            own open-to-close move, an exit still bears the overnight gap
            it was held through before selling at the open, and a held
            position sums the two legs instead of compounding them (a
            second-order, daily-bar-negligible approximation).
            "hl2_exploratory" — identical two-leg decomposition, but using
            that bar's own (High + Low) / 2 ("HL2") as the reference fill
            price instead of Open. This is NOT a bid/ask midpoint quote —
            it requires knowing the bar's High and Low, which are only
            determined once the bar has already completed, so pricing a
            fill at a bar's own HL2 is look-ahead the same way fill_price=
            "close" is (see the warning below); the name says "exploratory"
            deliberately, so it's never mistaken for a real, tradable
            execution price. entry_price/exit_price in the trade log use
            this same fill-mode-aware reference price, not always Close —
            see _build_trade_log.

    Returns:
        Dict with performance metrics, equity curve, and optionally
        trade_log. When fill_price is "close" or "hl2_exploratory",
        result["warnings"] includes a look-ahead-bias caveat (see below).

    Raises:
        ValidationError: fill_price is not one of "close", "next_open", "hl2_exploratory".
    """
    if fill_price not in _VALID_FILL_PRICES:
        raise ValidationError(
            f"fill_price must be one of {_VALID_FILL_PRICES}, got {fill_price!r}"
        )
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValidationError(
            f"initial_capital must be positive and finite, got {initial_capital!r} "
            "— a zero/negative/non-finite value silently produces inf/nan in "
            "total_return and calmar_ratio instead of a meaningful result."
        )
    for name, value in (
        ("commission_pct", commission_pct),
        ("slippage_pct", slippage_pct),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValidationError(
                f"{name} must be non-negative and finite, got {value!r}"
            )

    # Fast path: skip the intersection + two .loc[] calls entirely when the
    # indices are already identical (the common case for a signal derived
    # directly from price_data) -- .equals() is a cheap array comparison,
    # intersection+loc is real allocation work neither index needs here.
    if price_data.index.equals(signal_series.index):
        idx = price_data.index
    else:
        idx = price_data.index.intersection(signal_series.index)
    prices = price_data.loc[idx, "Close"]
    signals = signal_series.loc[idx]

    n_bars = len(prices)
    logger.debug(
        "[run_strategy] bars=%d  capital=%.0f  commission=%.4f  slippage=%.4f  fill_price=%s",
        n_bars,
        initial_capital,
        commission_pct,
        slippage_pct,
        fill_price,
    )

    # `returns`/`executed` are NOT computed here anymore -- the C++ path
    # below needs neither (it recomputes both internally from raw prices/
    # signals), and building them unconditionally was pure waste whenever
    # the C++ kernel actually ran. Each is now computed only where it's
    # actually used: `executed` lazily inside the C++ branch (only if
    # include_trade_log requests a Python-side trade log) or unconditionally
    # at the top of the Python fallback branch below (where both are
    # genuinely needed for the return/cost calculation itself).

    warnings: List[str] = []
    if fill_price == "close":
        warnings.append(
            "fill_price='close': a signal known at bar t-1's close is assumed filled "
            "at that same close. If signal_series was derived from that bar's own "
            "Close (e.g. a same-day indicator/score), this is a look-ahead bias — the "
            "trade could not actually have been placed at that price in real time. "
            "Use fill_price='next_open' for a lookahead-free simulation."
        )
    elif fill_price == "hl2_exploratory":
        warnings.append(
            "fill_price='hl2_exploratory': fills at a bar's own (High + Low) / 2 — "
            "not a real bid/ask midpoint quote, and not knowable until that bar has "
            "already completed (High/Low are only determined in retrospect), so this "
            "is look-ahead the same way fill_price='close' is. Intended for "
            "exploratory analysis only; use fill_price='next_open' for a "
            "lookahead-free simulation."
        )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    # Pass raw signals — C++ applies the one-bar lag internally (executed[i] = signals[i-1]).
    # Do NOT pass `executed` here: it is already shifted, which would cause a 2-bar lag.
    # The compiled kernel only knows Close prices, so it's scoped to the default
    # fill_price="close" — "next_open" always routes to the Python path below.
    if fill_price == "close" and HAS_CPP and _cpp_core is not None:
        logger.debug("[run_strategy] using C++ kernel")
        prices_arr = prices.to_numpy(dtype=np.float64)
        signals_arr = signals.to_numpy(dtype=np.float64)
        require_finite_array(prices_arr, "prices", "run_strategy")
        require_finite_array(signals_arr, "signals", "run_strategy")
        r = _cpp_core.run_strategy(
            prices_arr, signals_arr, initial_capital, commission_pct, slippage_pct
        )
        equity_curve = pd.Series(r["equity_curve"], index=idx)
        # win_rate/profit_factor/num_trades/avg_trade_return_pct: read
        # straight from the native result. backtest.cpp's own trade-log
        # logic uses the identical convention _build_trade_log does
        # (entry_price = prices[i-1], entry_size = signal magnitude, cost
        # scaled by position size) and this session's own CI verification
        # work (TestNativeTradeStatsCorrectness, run against a real
        # compiled _sqt_core on live CI, not just locally) already
        # confirmed native and Python trade stats agree exactly -- so
        # rebuilding the full Python trade log here just to recompute
        # numbers the C++ kernel already returned was pure redundant work,
        # not a correctness requirement. The Python trade log itself is
        # still built below, but only when include_trade_log actually asks
        # for the DataFrame, not for its stats.
        result: Dict[str, Any] = {
            "final_equity": round(float(r["final_equity"]), 2),
            "total_return": round(float(r["total_return"]), 6),
            "annualized_volatility": round(float(r["annualized_volatility"]), 6),
            "sharpe_ratio": round(float(r["sharpe_ratio"]), 4),
            "sortino_ratio": round(float(r["sortino_ratio"]), 4),
            "max_drawdown": round(float(r["max_drawdown"]), 6),
            "calmar_ratio": round(float(r["calmar_ratio"]), 4),
            "num_trades": int(r["num_trades"]),
            "win_rate": round(float(r["win_rate"]), 4),
            "profit_factor": round(float(r["profit_factor"]), 4),
            "avg_trade_return_pct": round(float(r["avg_trade_return_pct"]), 4),
            "equity_curve": equity_curve,
            "warnings": warnings,
        }
        if include_trade_log:
            executed = signals.shift(1).fillna(0.0)
            result["trade_log"] = _build_trade_log(
                prices.shift(1),
                prices,
                executed,
                commission_pct + slippage_pct,
            )
        logger.debug(
            "[run_strategy] C++  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
            result["total_return"] * 100,
            result["sharpe_ratio"],
            result["num_trades"],
            result["max_drawdown"] * 100,
        )
        return result

    # ── Python fallback ───────────────────────────────────────────────────────
    logger.debug("[run_strategy] using Python fallback  fill_price=%s", fill_price)
    returns = prices.pct_change().fillna(0.0)
    executed = signals.shift(1).fillna(0.0)
    cost_per_unit = commission_pct + slippage_pct
    pos_diff = executed.diff().fillna(executed.iloc[0])
    transaction_costs = pos_diff.abs() * cost_per_unit

    if fill_price in ("next_open", "hl2_exploratory"):
        # Two-leg decomposition, correct for entries, continuations, exits,
        # and same-bar flips alike:
        #   overnight leg (Close[t-1] -> ref_price[t]) priced at YESTERDAY's
        #     position (executed.shift(1)) — captures the gap a position
        #     still held overnight is exposed to, including on an exit bar
        #     (sold at today's reference price, so still exposed to the
        #     overnight gap but not today's remaining move).
        #   intraday leg (ref_price[t] -> Close[t]) priced at TODAY's
        #     position (executed) — captures a same-day entry's move from
        #     the reference price to the close, and a held-through day's
        #     remaining move.
        # For an unchanged position this sums two simple returns instead of
        # compounding them (their product is the only difference from pure
        # close-to-close — negligible for daily bars, standard in overnight
        # vs. intraday P&L attribution). "next_open" uses that bar's Open as
        # the reference price; "hl2_exploratory" uses (High + Low) / 2 —
        # NOT a real bid/ask midpoint, and only knowable after the bar has
        # already completed (see the look-ahead warning above).
        if fill_price == "next_open":
            ref_prices = price_data.loc[idx, "Open"]
        else:
            ref_prices = (
                price_data.loc[idx, "High"] + price_data.loc[idx, "Low"]
            ) / 2.0
        overnight_leg = ((ref_prices - prices.shift(1)) / prices.shift(1)).fillna(0.0)
        intraday_leg = (prices - ref_prices) / ref_prices
        executed_prev = executed.shift(1).fillna(0.0)
        gross_returns = executed_prev * overnight_leg + executed * intraday_leg
        strategy_returns = gross_returns - transaction_costs
    else:
        strategy_returns = executed * returns - transaction_costs
        # executed[i] = signals[i-1], so a position "appearing" in `executed`
        # at bar i actually earns its first return over Close[i-1] -> Close[i]
        # — Close[i-1] is its true economic entry/exit reference, not Close[i]
        # (only used for the trade log below; the return calc above is
        # already correct as-is).
        ref_prices = prices.shift(1)
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    total_ret = cumulative_return(equity_curve)
    annual_vol = annualized_volatility(strategy_returns)
    sr = sharpe_ratio(strategy_returns)
    srt = sortino_ratio(strategy_returns)
    mdd = max_drawdown(equity_curve)
    cal = calmar_ratio(equity_curve)
    final_eq = (
        float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital
    )

    result = {
        "final_equity": round(final_eq, 2),
        "total_return": round(total_ret, 6),
        "annualized_volatility": round(annual_vol, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(srt, 4),
        "max_drawdown": round(mdd, 6),
        "calmar_ratio": round(cal, 4),
        "equity_curve": equity_curve,
        "warnings": warnings,
    }

    trade_log = _build_trade_log(ref_prices, prices, executed, cost_per_unit)
    result.update(_compute_trade_stats(trade_log))

    if include_trade_log:
        result["trade_log"] = trade_log

    logger.debug(
        "[run_strategy] Python  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
        result["total_return"] * 100,
        result["sharpe_ratio"],
        result["num_trades"],
        result["max_drawdown"] * 100,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Grid search — module-level worker (must be picklable for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────


def _run_grid_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for backtest_grid. Must live at module level to be
    picklable by ProcessPoolExecutor on Windows (spawn start method).

    Only reached via ProcessPoolExecutor when `strategy` was a registry
    name (see backtest_grid) — a raw user callable is never sent through
    this path, since arbitrary callables (lambdas, closures) are frequently
    unpicklable across the spawn boundary.
    """
    df = job["price_data"]
    signal_fn = STRATEGY_REGISTRY[job["strategy"]]
    signals = signal_fn(df, **job["params"])

    result = run_strategy(
        df,
        signals,
        initial_capital=job["initial_capital"],
        commission_pct=job["commission_pct"],
        slippage_pct=job["slippage_pct"],
        fill_price=job.get("fill_price", "close"),
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.update(job["params"])
    return result


def _run_signal_fn_job(
    price_data: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    params: Dict[str, Any],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    fill_price: str = "close",
) -> Dict[str, Any]:
    """
    Sequential-only counterpart to _run_grid_job for a user-supplied signal
    callable. Always runs in the calling process (never via
    ProcessPoolExecutor), so signal_fn need not be picklable.
    """
    signals = signal_fn(price_data, **params)
    result = run_strategy(
        price_data,
        signals,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        fill_price=fill_price,
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.update(params)
    return result


def backtest_grid(
    price_data: pd.DataFrame,
    strategy: Union[str, Callable[..., pd.Series]],
    param_grid: Dict[str, List],
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    sort_by: str = "sharpe_ratio",
    ascending: bool = False,
    n_workers: Optional[int] = None,
    fill_price: str = "close",
) -> pd.DataFrame:
    """
    Run a backtest across every parameter combination in param_grid in parallel.

    Args:
        price_data:     OHLCV DataFrame (from provider.get_ohlcv).
        strategy:       Either one of the built-in registry names
                        ('sma_crossover', 'rsi_mean_reversion', 'macd_crossover',
                        'bollinger_reversion', 'donchian_breakout',
                        'momentum_timeseries', 'vwap_reversion', 'adx_trend'
                        — see backtest.strategies.STRATEGY_REGISTRY), or your
                        own signal-generating callable with signature
                        `(price_data: pd.DataFrame, **params)
                        -> pd.Series` (values in {-1, 0, 1}). A custom callable
                        still gets the full C++ batch-kernel speedup when
                        `_sqt_core` is built — only the metric computation runs
                        in C++, so it has no idea whether the signal came from
                        a built-in strategy or your own model.
        param_grid:     Dict mapping parameter name → list of values.
                        e.g. {'fast_period': [5, 10, 20], 'slow_period': [30, 50]}
        initial_capital: Starting capital for every backtest.
        commission_pct: Commission per trade side (fraction).
        slippage_pct:   Slippage per trade side (fraction).
        sort_by:        Output column to rank results by (default: 'sharpe_ratio').
        ascending:      Sort direction (default: False = best first).
        n_workers:      Worker processes. Defaults to os.cpu_count().
                        Pass 1 to run sequentially (no subprocess overhead).
                        Ignored (forced to 1) for a custom callable when the
                        C++ extension is not built — arbitrary callables
                        (lambdas, closures) are frequently unpicklable across
                        the ProcessPoolExecutor spawn boundary.
        fill_price:     "close" (default), "next_open", or "hl2_exploratory" — see
                        run_strategy. Forces the Python path for the latter
                        two (the C++ batch kernel only knows Close prices)
                        regardless of n_workers/HAS_CPP.

    Returns:
        pd.DataFrame with one row per parameter combination, sorted by sort_by.
        Columns include all metric keys plus the parameter names.

    Example (built-in strategy)::

        df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")
        results = backtest_grid(
            df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
        )
        print(results[["fast_period", "slow_period", "sharpe_ratio", "total_return"]].head())

    Example (your own signal, still grid-searched and C++-accelerated)::

        def my_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
            # any proprietary alpha logic — the grid searcher doesn't care
            edge = my_model.score(price_data)
            return (edge > threshold).astype(int)

        results = backtest_grid(
            df,
            strategy=my_signal,
            param_grid={"threshold": [0.1, 0.2, 0.3]},
        )
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValidationError(
            f"initial_capital must be positive and finite, got {initial_capital!r}"
        )
    for name, value in (
        ("commission_pct", commission_pct),
        ("slippage_pct", slippage_pct),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValidationError(
                f"{name} must be non-negative and finite, got {value!r}"
            )

    is_custom = callable(strategy)
    if is_custom:
        signal_fn: Callable[..., pd.Series] = strategy  # type: ignore[assignment]
        strategy_label = getattr(strategy, "__name__", "custom_strategy")
    else:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        signal_fn = STRATEGY_REGISTRY[strategy]
        strategy_label = strategy

    # Build all parameter combinations
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    t0 = time.perf_counter()

    # ── C++ batch path ────────────────────────────────────────────────────────
    # Generate all signal arrays in Python, then ship the entire batch to C++
    # in a single call — no subprocess overhead, no per-combo boundary crossing.
    # Scoped to fill_price="close" — the compiled kernel only knows Close prices.
    if fill_price == "close" and HAS_CPP and _cpp_core is not None:
        # Checked before the try/except below -- that except catches
        # Exception broadly (to fall back to the Python grid loop on any
        # C++ failure), which would otherwise silently swallow a
        # ValidationError and mask bad input behind a confusing fallback
        # instead of rejecting it.
        prices_arr = price_data["Close"].to_numpy(dtype=np.float64)
        require_finite_array(prices_arr, "prices", "batch_run_strategy")
        try:
            # Build (num_combos × n_bars) signal matrix
            sig_rows = []
            for combo in combos:
                params = dict(zip(keys, combo))
                sig_rows.append(
                    signal_fn(price_data, **params).to_numpy(dtype=np.float64)
                )
            signals_mat = np.ascontiguousarray(
                np.vstack(sig_rows), dtype=np.float64
            )  # shape: (num_combos, n_bars)
            require_finite_array(signals_mat, "signals", "batch_run_strategy")

            logger.debug(
                "[backtest_grid] strategy=%s  combos=%d  path=C++  sort_by=%s",
                strategy_label,
                len(combos),
                sort_by,
            )

            # win_rate/profit_factor/num_trades/avg_trade_return_pct here come
            # straight from the native kernel's own trade-log logic, same as
            # run_strategy's single-call C++ path above -- unlike that path,
            # nothing overwrites them per-combo here (rebuilding a Python-side
            # trade log for every parameter combination in the grid would
            # defeat the point of the batch C++ path's speed). This used to
            # be a real gap (native entry price one bar off, no commission/
            # slippage in trade returns), but backtest.cpp's run_strategy
            # (which batch_run_strategy calls per test) now uses the same
            # fill-aware, cost-aware accounting as _build_trade_log directly,
            # so these native stats should already agree with the Python
            # recomputation without an override -- see
            # tests/test_backtest.py's TestNativeTradeStatsCorrectness for
            # the gated equivalence check.
            # A flat (num_tests, 11) NumPy array instead of a Python list of
            # dicts -- for a large grid (thousands of combos), building that
            # many Python dict objects just to immediately feed them into
            # pd.DataFrame(rows) was itself real, avoidable overhead. Column
            # order is a fixed contract with bindings.cpp -- see
            # _BATCH_METRIC_COLUMNS above.
            metrics_arr = _cpp_core.batch_run_strategy(
                prices_arr,
                signals_mat,
                initial_capital,
                commission_pct,
                slippage_pct,
            )
            metrics_df = pd.DataFrame(metrics_arr, columns=_BATCH_METRIC_COLUMNS)
            metrics_df["num_trades"] = metrics_df["num_trades"].astype(int)
            metrics_df["final_equity"] = metrics_df["final_equity"].round(2)
            metrics_df["total_return"] = metrics_df["total_return"].round(6)
            metrics_df["annualized_volatility"] = metrics_df[
                "annualized_volatility"
            ].round(6)
            metrics_df["sharpe_ratio"] = metrics_df["sharpe_ratio"].round(4)
            metrics_df["sortino_ratio"] = metrics_df["sortino_ratio"].round(4)
            metrics_df["max_drawdown"] = metrics_df["max_drawdown"].round(6)
            metrics_df["calmar_ratio"] = metrics_df["calmar_ratio"].round(4)
            metrics_df["win_rate"] = metrics_df["win_rate"].round(4)
            metrics_df["profit_factor"] = metrics_df["profit_factor"].round(4)
            metrics_df["avg_trade_return_pct"] = metrics_df[
                "avg_trade_return_pct"
            ].round(4)

            params_df = pd.DataFrame(combos, columns=keys)
            df_out = pd.concat([metrics_df, params_df.reset_index(drop=True)], axis=1)
            if sort_by in df_out.columns:
                df_out = df_out.sort_values(sort_by, ascending=ascending).reset_index(
                    drop=True
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if not df_out.empty and sort_by in df_out.columns:
                best = df_out.iloc[0]
                best_params = {k: best[k] for k in keys if k in best}
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  best %s=%.4f  params=%s",
                    elapsed_ms,
                    sort_by,
                    best[sort_by],
                    best_params,
                )
            else:
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  %d results",
                    elapsed_ms,
                    len(df_out),
                )
            return df_out

        except ValidationError:
            # Bad input (e.g. NaN/Inf in a generated signal array) is a
            # real problem to surface, not something to silently retry
            # via the Python fallback below.
            raise
        except Exception as exc:
            logger.warning(
                "[backtest_grid] C++ batch path failed (%s) — falling back to Python",
                exc,
            )

    # ── Python fallback ───────────────────────────────────────────────────────
    if is_custom:
        # A raw callable may not be picklable (lambda, closure, notebook-defined
        # function) — always run sequentially in this process rather than risk
        # an opaque PicklingError from a ProcessPoolExecutor worker. The C++
        # batch path above (when built) already handles custom callables at
        # full speed with no subprocessing involved, so this only gives up
        # parallelism in the one uncommon case: no C++ extension AND >1 workers
        # requested AND a custom strategy.
        if n_workers is not None and n_workers != 1:
            logger.debug(
                "[backtest_grid] custom strategy without C++ extension — "
                "forcing sequential execution (n_workers=%s ignored)",
                n_workers,
            )
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=1 (forced)  path=Python  sort_by=%s",
            strategy_label,
            len(combos),
            sort_by,
        )
        results = [
            _run_signal_fn_job(
                price_data,
                signal_fn,
                dict(zip(keys, combo)),
                initial_capital,
                commission_pct,
                slippage_pct,
                fill_price=fill_price,
            )
            for combo in combos
        ]
    else:
        jobs = [
            {
                "price_data": price_data,
                "strategy": strategy,
                "params": dict(zip(keys, combo)),
                "initial_capital": initial_capital,
                "commission_pct": commission_pct,
                "slippage_pct": slippage_pct,
                "fill_price": fill_price,
            }
            for combo in combos
        ]
        workers = n_workers if n_workers is not None else (os.cpu_count() or 4)
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=%d  path=Python  sort_by=%s",
            strategy_label,
            len(jobs),
            workers,
            sort_by,
        )

        if workers == 1 or len(jobs) == 1:
            results = [_run_grid_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run_grid_job, jobs))

    df_out = pd.DataFrame(results)
    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not df_out.empty and sort_by in df_out.columns:
        best = df_out.iloc[0]
        best_params = {k: best[k] for k in keys if k in best}
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  best %s=%.4f  params=%s",
            elapsed_ms,
            sort_by,
            best[sort_by],
            best_params,
        )
    else:
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  %d results", elapsed_ms, len(df_out)
        )

    return df_out
