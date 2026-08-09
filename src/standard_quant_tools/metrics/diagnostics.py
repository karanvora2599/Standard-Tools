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

    for i in range(n):
        if dd.iloc[i] < 0:
            if not in_dd:
                in_dd = True
                trough_i = i
                trough_val = float(dd.iloc[i])
            elif float(dd.iloc[i]) < trough_val:
                trough_i = i
                trough_val = float(dd.iloc[i])
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
    is_win = returns > 0
    wins = returns[is_win]
    losses = returns[~is_win]

    win_rate = len(wins) / len(returns)
    avg_winner = float(wins.mean()) if len(wins) else 0.0
    avg_loser = float(losses.mean()) if len(losses) else 0.0  # <= 0
    expectancy = win_rate * avg_winner + (1 - win_rate) * avg_loser
    payoff_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float("inf")

    max_w = cur_w = max_l = cur_l = 0
    for won in is_win:
        if won:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

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
    relative to each trade's entry price.

    Recovered without any engine change: re-walks `price_data`'s existing
    High/Low columns between each trade's own entry_date/exit_date (already
    present in `trade_log`) — the engine never needed to track this itself.
    """
    if trade_log.empty:
        result = trade_log.copy()
        result["mae_pct"] = pd.Series(dtype=float)
        result["mfe_pct"] = pd.Series(dtype=float)
        return result

    mae_list: List[float] = []
    mfe_list: List[float] = []
    for _, row in trade_log.iterrows():
        window = price_data.loc[row["entry_date"] : row["exit_date"]]
        entry_price = float(row["entry_price"])
        if window.empty or entry_price == 0:
            mae_list.append(0.0)
            mfe_list.append(0.0)
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
        idx = executed_signal.index
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
        if holding_bars:
            avg_holding_period_bars = round(float(np.mean(holding_bars)), 2)

    return {
        "time_in_market": round(float((values != 0).mean()), 4),
        "avg_gross_exposure": round(float(np.abs(values).mean()), 4),
        "avg_net_exposure": round(float(values.mean()), 4),
        "pct_long": round(float((values > 0).mean()), 4),
        "pct_short": round(float((values < 0).mean()), 4),
        "avg_holding_period_bars": avg_holding_period_bars,
    }
