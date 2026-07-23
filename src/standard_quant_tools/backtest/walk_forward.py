"""
Walk-forward out-of-sample stitching utilities.

run_walk_forward_backtest() (agent/tools.py) computes per-window OOS stats
independently, then averages them across windows. That misrepresents
compounding: e.g. windows of +20% and -20% average to 0% but compound to
roughly -4%. These helpers instead stitch the per-window OOS return series
into one chronological series and compute metrics from a single equity
curve, reusing the existing metrics functions rather than introducing new
metric math.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from standard_quant_tools.metrics.return_metrics import cumulative_return
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
)

logger = logging.getLogger(__name__)


def stitch_oos_returns(window_returns: List[pd.Series]) -> pd.Series:
    """
    Concatenate per-window out-of-sample return series into one
    chronological series. Walk-forward test windows never overlap (each
    window's cursor advances by test_bars), so this is a plain
    concat + sort, not a merge.
    """
    if not window_returns:
        return pd.Series(dtype=float)
    return pd.concat(window_returns).sort_index()


def compute_stitched_metrics(
    oos_returns: pd.Series, initial_capital: float = 10_000.0,
) -> Dict[str, float]:
    """
    Build one equity curve from the stitched OOS returns and compute
    metrics off it — the economically correct alternative to averaging
    each window's independently-computed metrics.
    """
    if oos_returns.empty:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }
    equity_curve = initial_capital * (1 + oos_returns).cumprod()
    return {
        "total_return": float(cumulative_return(equity_curve)),
        "sharpe_ratio": float(sharpe_ratio(oos_returns)),
        "sortino_ratio": float(sortino_ratio(oos_returns)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "calmar_ratio": float(calmar_ratio(equity_curve)),
    }


def longest_losing_streak(window_returns: List[float]) -> int:
    """Longest run of consecutive windows with a negative OOS return."""
    longest = 0
    current = 0
    for r in window_returns:
        if r < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def parameter_turnover(window_params: List[Dict[str, Any]]) -> float:
    """
    Fraction of consecutive window pairs whose best_params differ.
    0.0 if there are fewer than two windows (no transition to measure).
    """
    if len(window_params) < 2:
        return 0.0
    changes = sum(
        1 for prev, curr in zip(window_params, window_params[1:]) if prev != curr
    )
    return round(changes / (len(window_params) - 1), 4)
