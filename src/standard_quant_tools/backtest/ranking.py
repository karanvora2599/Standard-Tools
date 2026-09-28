"""
Which rows of a ranked comparison are allowed to win it.

Every door that ranks backtests -- `backtest_grid` on both of its paths,
`compare_strategies`, the regime-adaptive walk-forward's choice between
strategies, the strategy matrix and the optimization tool -- orders rows by
one metric and reads the first one as the answer. Two kinds of row must not
be able to come first:

- a row whose metric is not a finite number. A ratio over an empty
  denominator is +inf ("never lost"), and pandas puts +inf at the top of a
  descending sort; NaN has no place in an order at all.
- a row that never traded. It has no drawdown, no volatility and no loss,
  so it wins every "smaller is better" metric and ties or wins the ratios.
  A do-nothing parameter set (a slow average longer than the window) won
  `backtest_grid` under Calmar and Sortino, and the genuine +50% strategy in
  the same grid ranked last.

Such rows keep their values and are listed after every ranked row, in their
original order, and every door reports how many there were. This is the
rule `robustness.parameter_sensitivity` already applied to its own input.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_float_array(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(
        dtype=np.float64
    )


def rank_order(
    values: Sequence[Any],
    num_trades: Optional[Sequence[Any]] = None,
    ascending: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Positions of `values` best first, and how many rows could not be ranked.

    A row is ranked only if its value is finite and, when `num_trades` is
    given, it traded at least once. Ranked rows come first, ordered by value
    (`ascending` for a metric where smaller is better), ties kept in their
    original order; the rest follow, also in their original order. A
    non-numeric value counts as not finite.
    """
    key = _as_float_array(values)
    rankable = np.isfinite(key)
    if num_trades is not None:
        trades = _as_float_array(num_trades)
        rankable &= trades > 0
    positions = np.arange(key.size)
    ranked = positions[rankable]
    # Stable, so equal values keep the order the caller produced them in --
    # pandas' default sort is not, which made a tie's winner arbitrary.
    order = np.argsort(key[rankable] if ascending else -key[rankable], kind="stable")
    ordered = np.concatenate([ranked[order], positions[~rankable]])
    return ordered, int((~rankable).sum())


def rank_rows(df: pd.DataFrame, sort_by: str, ascending: bool) -> pd.DataFrame:
    """
    `df` ordered best first by `sort_by` under the rule above.

    The count of unranked rows is stamped on `attrs["n_unrankable"]`. A
    frame without the column is returned in its own order with a count of 0.
    """
    if sort_by not in df.columns:
        out = df.reset_index(drop=True)
        out.attrs["n_unrankable"] = 0
        return out
    order, n_unrankable = rank_order(
        df[sort_by],
        df["num_trades"] if "num_trades" in df.columns else None,
        ascending,
    )
    out = df.iloc[order].reset_index(drop=True)
    out.attrs = dict(df.attrs)
    out.attrs["n_unrankable"] = n_unrankable
    return out


def unrankable_note(n_unrankable: int, n_total: int, sort_by: str, noun: str) -> str:
    """The sentence every ranked door reports its excluded rows with."""
    return (
        f"{n_unrankable} of {n_total} {noun}(s) have no rankable {sort_by} -- "
        "they never traded, or the metric is undefined or infinite (a ratio "
        "over an empty denominator) -- and are listed after every ranked one "
        "rather than competing for first place."
    )
