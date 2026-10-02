"""
Extended backtest diagnostics: drawdown episodes, trade expectancy/MAE-MFE,
and exposure statistics.

Sharpe and total return alone don't tell a quant whether a backtest is
trustworthy. These functions are computed entirely from data a backtest
already produces (`equity_curve`, `trade_log` from `backtest/engine.py`, and
the OHLCV frame each tool already fetched) — no engine changes required.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import drawdown_series

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Drawdown episodes
# ──────────────────────────────────────────────────────────────────


def drawdown_periods(equity_curve: pd.Series) -> pd.DataFrame:
    """
    One row per drawdown episode: the peak before the decline, the trough,
    and the recovery point (None if still underwater at the end of the
    series). Reuses the existing `drawdown_series` — no new drawdown math.

    Columns: start, trough, end, depth, duration_bars, recovery_bars.
    `duration_bars` spans peak -> recovery (or peak -> last bar if
    unrecovered); `recovery_bars` spans trough -> recovery (None if
    unrecovered).

    `start` IS THE PEAK BAR, not the first bar below it. That matters
    because `analysis.diagnostics.drawdown_profile` answers the same
    question with the other convention -- its episodes start on the first
    bar underwater -- so the two report starts exactly one bar apart on the
    same equity curve, agreeing on trough and depth. Both are reachable
    from the agent surface, under result models that both happen to be
    called `DrawdownEpisode`.

    Neither convention is wrong and neither is changed here, because each
    is the one its own duration field is measured from: this counts peak ->
    recovery, and that one counts underwater -> recovery. The difference is
    pinned by test_drawdown_start_conventions so it stays deliberate.
    """
    if equity_curve.empty:
        return pd.DataFrame(
            columns=[
                "start",
                "trough",
                "end",
                "depth",
                "duration_bars",
                "recovery_bars",
            ]
        )

    dd = drawdown_series(equity_curve)
    idx = equity_curve.index
    n = len(equity_curve)

    records: List[Dict[str, Any]] = []
    in_dd = False
    peak_i = 0
    trough_i: Optional[int] = None
    trough_val = 0.0

    # Read the drawdown buffer once. This state machine is sequential and
    # cannot be vectorized, but it was calling `dd.iloc[i]` up to three
    # times per bar -- profiled at 67% of the function -- to read one
    # number it could hold in a local.
    dd_values = dd.to_numpy(dtype=float)

    for i in range(n):
        current = dd_values[i]
        if current < 0:
            if not in_dd:
                in_dd = True
                trough_i = i
                trough_val = float(current)
            elif float(current) < trough_val:
                trough_i = i
                trough_val = float(current)
        else:
            if in_dd:
                assert trough_i is not None
                records.append(
                    {
                        "start": idx[peak_i],
                        "trough": idx[trough_i],
                        "end": idx[i],
                        "depth": round(trough_val, 6),
                        "duration_bars": i - peak_i,
                        "recovery_bars": i - trough_i,
                    }
                )
                in_dd = False
            peak_i = i

    if in_dd:
        assert trough_i is not None
        records.append(
            {
                "start": idx[peak_i],
                "trough": idx[trough_i],
                "end": None,
                "depth": round(trough_val, 6),
                "duration_bars": (n - 1) - peak_i,
                "recovery_bars": None,
            }
        )

    return pd.DataFrame(
        records,
        columns=["start", "trough", "end", "depth", "duration_bars", "recovery_bars"],
    )


def top_n_drawdowns(equity_curve: pd.Series, n: int = 5) -> pd.DataFrame:
    """The n deepest drawdown episodes, worst first (depth is negative)."""
    periods = drawdown_periods(equity_curve)
    if periods.empty:
        return periods
    return periods.sort_values("depth", ascending=True).head(n).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
# Trade diagnostics
# ──────────────────────────────────────────────────────────────────


def trade_expectancy(trade_log: pd.DataFrame) -> Dict[str, Any]:
    """
    Expectancy, average winner/loser, payoff ratio, and consecutive-streak
    stats from the `return_pct` column `_build_trade_log`
    (backtest/engine.py) already produces (values are in percent, e.g. 2.5
    means +2.5%).
    """
    if trade_log.empty:
        return {
            "expectancy_pct": 0.0,
            "avg_winner_pct": 0.0,
            "avg_loser_pct": 0.0,
            "payoff_ratio": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    returns = trade_log["return_pct"].to_numpy(dtype=float)
    # Checked BEFORE the three-way split below, and this is a hole the split
    # itself opened. Moving from `~is_win` to explicit `> 0` / `< 0` tests
    # made NaN satisfy NEITHER, so a NaN trade return was silently bucketed
    # with the breakevens -- counted in the denominator of win_rate, excluded
    # from both averages, and treated as a streak-breaker. The previous
    # two-way split had at least counted it as a loss. Neither is right: an
    # unmeasurable trade is not a flat trade.
    if not np.all(np.isfinite(returns)):
        n_bad = int(np.sum(~np.isfinite(returns)))
        raise ValidationError(
            f"trade_log['return_pct'] contains {n_bad} non-finite value(s). A "
            "NaN trade return is neither a win, a loss, nor a breakeven — it "
            "is a trade whose outcome is unknown, and bucketing it with the "
            "flat trades would understate both the win rate and the loss "
            "streak."
        )
    # A trade that returns exactly 0.0 is neither a win nor a loss. It used
    # to fall into `losses` (via ~is_win), which dragged avg_loser toward
    # zero, inflated the loss count, and — worst — extended
    # max_consecutive_losses through what were actually flat trades. On a
    # win/breakeven/loss triple it reported 2 consecutive losses.
    #
    # Breakevens are excluded from BOTH sides rather than reassigned, since
    # counting them as wins would flatter the win rate just as wrongly.
    is_win = returns > 0
    is_loss = returns < 0
    wins = returns[is_win]
    losses = returns[is_loss]

    win_rate = len(wins) / len(returns)
    avg_winner = float(wins.mean()) if len(wins) else 0.0
    avg_loser = float(losses.mean()) if len(losses) else 0.0  # <= 0

    # THREE STATES HERE TOO. `(1 - win_rate)` is the not-won rate, which
    # includes breakevens, so every flat trade was priced at `avg_loser`.
    # On [10.0, 0.0, -5.0] that returned exactly -0.0 -- a profitable set
    # of trades reported as breakeven, with the headline's sign flipped --
    # where the mean is +1.667. A breakeven trade contributes its own
    # return, which is zero, so it belongs in the denominator and not in
    # either average. The streak loop below already got this right.
    loss_rate = len(losses) / len(returns)
    expectancy = win_rate * avg_winner + loss_rate * avg_loser
    payoff_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float("inf")

    # Three states, not two. Iterating `is_win` alone made every non-win a
    # loss, so a breakeven trade extended a LOSING STREAK — the one statistic
    # here a reader is most likely to treat as a risk signal. A flat trade
    # breaks both streaks instead of continuing either.
    max_w = cur_w = max_l = cur_l = 0
    for r in returns:
        if r > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        elif r < 0:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)
        else:
            cur_w = cur_l = 0

    return {
        "expectancy_pct": round(expectancy, 4),
        "avg_winner_pct": round(avg_winner, 4),
        "avg_loser_pct": round(avg_loser, 4),
        "payoff_ratio": (
            round(payoff_ratio, 4) if np.isfinite(payoff_ratio) else float("inf")
        ),
        "max_consecutive_wins": int(max_w),
        "max_consecutive_losses": int(max_l),
    }


def trade_excursions(trade_log: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add `mae_pct` (maximum adverse excursion) and `mfe_pct` (maximum
    favorable excursion) columns to a copy of `trade_log`, in percent,
    relative to each trade's `entry_price`.

    Recovered without any engine change: re-walks `price_data`'s existing
    High/Low columns between each trade's own entry_date/exit_date (already
    present in `trade_log`) — the engine never needed to track this itself.

    Note what `entry_price` means for a lot that changed size during its
    life: it is the WEIGHTED-AVERAGE COST BASIS across the whole lot, not
    the price of its opening leg, so these excursions are measured from a
    computed basis rather than from a level that necessarily ever traded.
    That is the intended reference — it is the same basis `return_pct` is
    measured against, so MAE/MFE and the realized return stay on one scale —
    but it means `mae_pct` is NOT directly comparable to a stop-loss placed
    at a fill price. A lot opened at 100 and doubled at 110 carries a basis
    of 105, and a dip to 99 reads as −5.7% here, not −1.0%.
    """
    if trade_log.empty:
        result = trade_log.copy()
        result["mae_pct"] = pd.Series(dtype=float)
        result["mfe_pct"] = pd.Series(dtype=float)
        return result

    windows = _window_bounds(trade_log, price_data)
    if windows is not None:
        fast_mae, fast_mfe = _excursions_from_bounds(trade_log, price_data, *windows)
        result = trade_log.copy()
        result["mae_pct"] = fast_mae
        result["mfe_pct"] = fast_mfe
        return result

    # The general path, for anything the bounds above cannot locate exactly
    # (an unsorted price index, string dates, a date dtype other than the
    # index's own). Same arithmetic, one label slice per trade.
    mae_list: List[float] = []
    mfe_list: List[float] = []
    for _, row in trade_log.iterrows():
        window = price_data.loc[row["entry_date"] : row["exit_date"]]
        entry_price = float(row["entry_price"])
        # NaN, not 0.0. An empty price window or an unusable entry price means
        # the excursion could not be measured — and 0.0 is the single most
        # flattering answer available, reading as "this trade never moved
        # against me at all". A trade whose prices are missing is not a
        # risk-free trade. Averages built on these (avg_mae_pct) would
        # otherwise be pulled toward zero by exactly the trades with no data.
        if window.empty or not np.isfinite(entry_price) or entry_price <= 0:
            mae_list.append(float("nan"))
            mfe_list.append(float("nan"))
            continue

        is_long = row["direction"] == "long"
        high, low = float(window["High"].max()), float(window["Low"].min())
        if is_long:
            mfe = (high - entry_price) / entry_price
            mae = (low - entry_price) / entry_price
        else:
            mfe = (entry_price - low) / entry_price
            mae = (entry_price - high) / entry_price
        mfe_list.append(round(mfe * 100, 4))
        mae_list.append(round(mae * 100, 4))

    result = trade_log.copy()
    result["mae_pct"] = mae_list
    result["mfe_pct"] = mfe_list
    return result


def _same_dates(column: Any, index: pd.Index) -> bool:
    """True when `column` holds dates of exactly `index`'s dtype (unit and
    timezone included), so a position lookup means what a label lookup does."""
    return (
        isinstance(column, pd.Series)
        and isinstance(index, pd.DatetimeIndex)
        and column.dtype == index.dtype
    )


def _window_bounds(
    trade_log: pd.DataFrame, price_data: pd.DataFrame
) -> Optional["tuple[np.ndarray, np.ndarray]"]:
    """
    Each trade's `price_data.loc[entry_date : exit_date]` as positions
    `[start, stop)`, found for every trade at once -- or None when that
    equivalence is not guaranteed and the per-trade slice must be used.

    On a sorted DatetimeIndex a label slice IS a binary search: pandas
    resolves the left label with `searchsorted(side="left")` and the right
    one with `side="right"`, whether or not either date is present and
    however many times it repeats. One vectorized search per side gives the
    same rows that 805 separate `.loc` slices did, at a fraction of the
    cost. Anything else (an unsorted index, string or mismatched-dtype
    dates, a missing date) keeps the slice, because there `.loc` follows
    other rules.
    """
    index = price_data.index
    if not trade_log.columns.is_unique or not {
        "entry_date",
        "exit_date",
        "entry_price",
        "direction",
    }.issubset(trade_log.columns):
        return None
    entries = trade_log["entry_date"]
    exits = trade_log["exit_date"]
    if not (_same_dates(entries, index) and _same_dates(exits, index)):
        return None
    if not index.is_monotonic_increasing or entries.isna().any() or exits.isna().any():
        return None
    for name in ("High", "Low"):
        # A plain numeric column reduces to the same number either way; an
        # object or extension column keeps pandas' own max/min rules.
        if name in price_data.columns:
            column = price_data[name]
            if not (
                isinstance(column, pd.Series)
                and isinstance(column.dtype, np.dtype)
                and column.dtype.kind in "fiu"
            ):
                return None
    starts = index.searchsorted(entries.array, side="left")
    stops = index.searchsorted(exits.array, side="right")
    return np.asarray(starts, dtype=np.intp), np.asarray(stops, dtype=np.intp)


def _column_extremes(
    price_data: pd.DataFrame, starts: np.ndarray, stops: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """
    max(High) and min(Low) over each `[start, stop)`, skipping NaN the way
    `Series.max()` / `Series.min()` do (all-NaN gives NaN).

    `fmax`/`fmin` ignore a NaN operand, and a maximum is exact -- no
    rounding happens in choosing one value from a set -- so this is the
    same number pandas returns, in one `reduceat` per column instead of two
    reductions per trade. Entries for empty windows are meaningless and
    never read.
    """
    high = price_data["High"].to_numpy(dtype=float)
    low = price_data["Low"].to_numpy(dtype=float)
    # reduceat needs every boundary to be a valid position, and a stop may
    # be one past the last bar; a trailing NaN is a no-op under fmax/fmin.
    high = np.append(high, np.nan)
    low = np.append(low, np.nan)
    bounds = np.empty(2 * len(starts), dtype=np.intp)
    bounds[0::2] = starts
    bounds[1::2] = stops
    return (
        np.fmax.reduceat(high, bounds)[0::2],
        np.fmin.reduceat(low, bounds)[0::2],
    )


def _excursions_from_bounds(
    trade_log: pd.DataFrame,
    price_data: pd.DataFrame,
    starts: np.ndarray,
    stops: np.ndarray,
) -> "tuple[List[float], List[float]]":
    """The per-trade loop of `trade_excursions`, reading located windows."""
    entry_prices = [float(p) for p in trade_log["entry_price"]]
    nonempty = (stops > starts) & (price_data.shape[1] > 0)
    measurable = [
        bool(ok) and np.isfinite(p) and p > 0 for ok, p in zip(nonempty, entry_prices)
    ]
    # Only read High/Low when some trade will use them, as the slice did.
    if any(measurable):
        highs, lows = _column_extremes(price_data, starts, stops)
    else:
        highs = lows = np.full(len(starts), np.nan)

    mae_list: List[float] = []
    mfe_list: List[float] = []
    for ok, entry_price, direction, high, low in zip(
        measurable, entry_prices, trade_log["direction"], highs.tolist(), lows.tolist()
    ):
        if not ok:
            mae_list.append(float("nan"))
            mfe_list.append(float("nan"))
            continue
        if direction == "long":
            mfe = (high - entry_price) / entry_price
            mae = (low - entry_price) / entry_price
        else:
            mfe = (entry_price - low) / entry_price
            mae = (entry_price - high) / entry_price
        mfe_list.append(round(mfe * 100, 4))
        mae_list.append(round(mae * 100, 4))
    return mae_list, mfe_list


# ──────────────────────────────────────────────────────────────────
# Exposure diagnostics
# ──────────────────────────────────────────────────────────────────


def exposure_stats(
    executed_signal: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Exposure profile from the already-shifted (one-bar-lag) executed
    position series — same convention `run_strategy` uses internally
    (`signals.shift(1)`). `avg_holding_period_bars` is derived from
    `trade_log`'s own entry/exit dates located against `executed_signal`'s
    index, when a trade log is supplied.
    """
    values = executed_signal.to_numpy(dtype=float)
    # A NaN position satisfies `!= 0`, so a missing position counted as time
    # IN the market and then made avg_gross/net_exposure NaN. Measured on a
    # 3-bar series with one NaN: time_in_market 0.6667 with both exposure
    # averages NaN. Missing exposure is not exposure.
    if len(values) and not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValidationError(
            f"executed_signal contains {n_bad} non-finite value(s). A NaN "
            "position satisfies `!= 0`, so it would be counted as time in the "
            "market while making every exposure average NaN."
        )
    if len(values) == 0:
        return {
            "time_in_market": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_net_exposure": 0.0,
            "pct_long": 0.0,
            "pct_short": 0.0,
            "avg_holding_period_bars": None,
        }

    avg_holding_period_bars: Optional[float] = None
    if trade_log is not None and not trade_log.empty:
        holding_bars = _holding_bars(executed_signal.index, trade_log)
        if len(holding_bars):
            avg_holding_period_bars = round(float(np.mean(holding_bars)), 2)

    return {
        "time_in_market": round(float((values != 0).mean()), 4),
        "avg_gross_exposure": round(float(np.abs(values).mean()), 4),
        "avg_net_exposure": round(float(values.mean()), 4),
        "pct_long": round(float((values > 0).mean()), 4),
        "pct_short": round(float((values < 0).mean()), 4),
        "avg_holding_period_bars": avg_holding_period_bars,
    }


def _holding_bars(idx: pd.Index, trade_log: pd.DataFrame) -> Any:
    """
    Bars held by each trade whose entry and exit dates are both on `idx`,
    in trade-log order; a trade with either date missing is skipped.

    On a unique DatetimeIndex, with dates of the index's own dtype, one
    `get_indexer` per column finds every position at once, and it agrees
    with `get_loc` key for key: the same position where the date is
    present, and -1 exactly where `get_loc` raises KeyError. Anything else
    -- a repeated timestamp above all, where `get_loc` answers with a slice
    and the trade is excluded with a warning -- is located one trade at a
    time.
    """
    if (
        idx.is_unique
        and "entry_date" in trade_log.columns
        and "exit_date" in trade_log.columns
        and _same_dates(trade_log["entry_date"], idx)
        and _same_dates(trade_log["exit_date"], idx)
    ):
        entries = idx.get_indexer(pd.Index(trade_log["entry_date"]))
        exits = idx.get_indexer(pd.Index(trade_log["exit_date"]))
        found = (entries >= 0) & (exits >= 0)
        return exits[found] - entries[found]

    holding_bars: List[int] = []
    for _, row in trade_log.iterrows():
        try:
            entry_pos = idx.get_loc(row["entry_date"])
            exit_pos = idx.get_loc(row["exit_date"])
        except KeyError:
            continue
        # On a non-unique index get_loc returns a slice or boolean mask
        # instead of an int, and int() on either raises TypeError — which
        # the KeyError-only guard above did not catch, turning a duplicate
        # timestamp into a crash instead of a skipped trade.
        if not isinstance(entry_pos, int) or not isinstance(exit_pos, int):
            logger.warning(
                "[exposure_stats] ambiguous index position for trade "
                "%s -> %s (duplicate timestamps?) — excluded from "
                "avg_holding_period_bars",
                row["entry_date"],
                row["exit_date"],
            )
            continue
        holding_bars.append(exit_pos - entry_pos)
    return holding_bars
