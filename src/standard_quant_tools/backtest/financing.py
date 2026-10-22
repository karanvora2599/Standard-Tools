"""
What cash earns and what borrowing costs, on one declared day count.

Every engine here subtracted a risk-free rate in its Sharpe while the cash
it held earned nothing and the leverage and shorts it ran were financed for
free. A book half in cash at a 4% rate read a Sharpe of 0.57 against 0.84
for the same asset fully invested, and a dollar-neutral book, whose cash is
its whole equity plus its short proceeds, went from 0.04 at a zero rate to
-0.70 at 5%. See the CHANGELOG entry of 2026-10-22.

The rule all three engines now share: positive cash accrues at a cash rate,
negative cash at a debit rate, shorts pay a borrow fee, and the reported
Sharpe and Sortino measure the excess over the rate cash ACTUALLY accrued
on each bar -- the same rate, the same day count, from the same first bar.
So a book that holds only cash has an excess return of exactly zero, and a
constant k-times position financed at f returns k*R - (k - 1)*f on every
bar.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.costs import DEFAULT_DAY_COUNT, day_count_basis
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    has_no_dispersion,
    sharpe_ratio,
    sortino_ratio,
)
from standard_quant_tools.numeric_contract import require_finite_scalar
from standard_quant_tools.validation import validate_series

__all__ = [
    "SHORT_PROCEEDS",
    "calendar_day_gaps",
    "accrued_rate_per_bar",
    "ratios_over_accrued_rate",
    "resolve_debit_rate",
    "financing_summary",
    "negative_cash_note",
]

#: Where a short sale's proceeds sit. `net_cash` (the default, and what the
#: shared-cash engine always did) puts them in the account's cash, where
#: they earn the cash rate with everything else. `segregated` holds them as
#: collateral with the lender: they earn only the short rebate, and the
#: account's interest-bearing cash is its balance less the short book.
SHORT_PROCEEDS = ("net_cash", "segregated")


def calendar_day_gaps(index: pd.Index) -> np.ndarray:
    """
    Calendar days from each bar to the one before it, as float64, with 0.0
    on the first bar: an account opened on bar 0 has carried nothing into
    it, so nothing accrues there.

    The same whole-day count the shared-cash engine's kernel has always been
    handed, so the two engines and the benchmark below count one Friday to
    Monday as three days alike.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(
            "financing accrues on calendar days, so it needs a DatetimeIndex; "
            f"got {type(index).__name__}. Pass dated bars, or set every rate "
            "(cash, debit, borrow, rebate and the risk-free rate) to 0."
        )
    gaps = np.zeros(len(index), dtype=np.float64)
    if len(index) > 1:
        deltas = np.diff(index.to_numpy()).astype("timedelta64[D]")
        gaps[1:] = deltas.astype(np.float64)
    return gaps


def accrued_rate_per_bar(
    index: pd.Index, annual_rate: float, day_count: str = DEFAULT_DAY_COUNT
) -> pd.Series:
    """
    What one unit of cash earned on each bar at `annual_rate`:
    annual_rate * (calendar days since the previous bar / basis), and 0.0
    on the first bar.

    The expression is the engine's own credit, `cash * rate * (days /
    basis)`, divided through by the cash, so a book holding nothing but cash
    has an excess return over this series of zero to rounding.
    """
    basis = day_count_basis(day_count)
    gaps = calendar_day_gaps(index)
    return pd.Series(annual_rate * (gaps / basis), index=index, dtype=np.float64)


@validate_series()
def _ratios(
    returns: pd.Series, benchmark: np.ndarray, periods_per_year: int
) -> Tuple[float, float]:
    excess = returns - benchmark
    std = returns.std()
    if has_no_dispersion(returns, std):
        sharpe = float("nan")
    else:
        sharpe = float((excess.mean() / std) * np.sqrt(periods_per_year))
    downside_sq = np.minimum(excess.to_numpy(dtype=np.float64), 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean())) * np.sqrt(periods_per_year)
    mean_excess = float(excess.mean())
    if np.isnan(downside_dev):
        sortino = float("nan")
    elif downside_dev == 0:
        sortino = np.inf if mean_excess > 0 else float("nan")
    else:
        sortino = float((mean_excess * periods_per_year) / downside_dev)
    return sharpe, sortino


def ratios_over_accrued_rate(
    returns: pd.Series,
    risk_free_rate: float,
    day_count: str = DEFAULT_DAY_COUNT,
    periods_per_year: int = 252,
    benchmark: Optional[pd.Series] = None,
) -> Tuple[float, float]:
    """
    Sharpe and Sortino of `returns` over the risk-free rate as cash accrued
    it, bar by bar, on the declared day count.

    The formulas are `metrics.risk_metrics.sharpe_ratio` and
    `sortino_ratio` exactly -- mean excess over the standard deviation of the
    returns, and over the root-mean-square of the negative excesses -- with
    the per-bar benchmark in place of `rate / periods_per_year`. That swap
    is the point: a rate accrued over calendar days is what the engine
    credits, so a book that sits in cash scores an excess of zero rather
    than the gap between calendar-day and per-bar accrual, and the first
    bar, over which nothing accrued, is not charged a period of the rate.

    At a zero rate this IS `sharpe_ratio(returns, 0.0)` and
    `sortino_ratio(returns, 0.0)`, called as such, so every result reported
    before the rate entered the cash is reproduced bit for bit.

    `benchmark`, when given, is the per-bar rate itself (aligned to
    `returns`); otherwise it is built from the index of `returns`.
    """
    rate = require_finite_scalar(
        risk_free_rate, "risk_free_rate", "ratios_over_accrued_rate"
    )
    if rate == 0.0:
        return (
            sharpe_ratio(returns, 0.0, periods_per_year),
            sortino_ratio(returns, 0.0, periods_per_year),
        )
    if benchmark is None:
        benchmark = accrued_rate_per_bar(returns.index, rate, day_count)
    bench = benchmark.reindex(returns.index).to_numpy(dtype=np.float64)
    if not np.isfinite(bench).all():
        raise ValidationError(
            "the per-bar risk-free benchmark does not cover every return bar; "
            "it must be built on the same index as the returns."
        )
    return _ratios(returns, bench, periods_per_year)


def resolve_debit_rate(
    margin_interest_rate: Optional[float],
    risk_free_rate: float,
    margin_spread_bps: float = 0.0,
) -> float:
    """
    The annual rate charged on negative cash: the caller's own rate when
    given, otherwise the risk-free rate plus `margin_spread_bps`, floored at
    zero -- a broker does not pay an account to borrow.
    """
    if margin_interest_rate is not None:
        return float(margin_interest_rate)
    if not math.isfinite(margin_spread_bps) or margin_spread_bps < 0:
        raise ValidationError(
            f"margin_spread_bps must be finite and >= 0, got {margin_spread_bps!r}"
        )
    return max(float(risk_free_rate) + float(margin_spread_bps) / 10_000.0, 0.0)


def financing_summary(
    *,
    day_count: str,
    cash_rate: float,
    margin_interest_rate: float,
    borrow_fee_bps: float,
    short_proceeds: str,
    short_rebate_rate: float,
    totals: Dict[str, float],
) -> Dict[str, Any]:
    """The rates a run was financed at, echoed beside what they came to."""
    credited = float(totals.get("interest_credited", 0.0))
    charged = float(totals.get("interest_charged", 0.0))
    borrow = float(totals.get("borrow_charged", 0.0))
    rebate = float(totals.get("rebate_credited", 0.0))
    return {
        "day_count": day_count,
        "cash_rate": float(cash_rate),
        "margin_interest_rate": float(margin_interest_rate),
        "borrow_fee_bps": float(borrow_fee_bps),
        "short_proceeds": short_proceeds,
        "short_rebate_rate": float(short_rebate_rate),
        "interest_credited": credited,
        "interest_charged": charged,
        "borrow_charged": borrow,
        "rebate_credited": rebate,
        "net_financing": credited + rebate - charged - borrow,
        "debit_balance_years": float(totals.get("debit_balance_years", 0.0)),
    }


def negative_cash_note(
    cash_curve: pd.Series,
    financing: Dict[str, Any],
) -> Optional[str]:
    """
    The warning for a book whose cash went negative, with the number in it.

    It used to read only "cash went negative at one or more bars -- implied
    margin borrowing", which said neither how much was borrowed nor what it
    was charged, so a 2x book financed at zero and one financed at 8% read
    the same. None when cash never went negative.
    """
    negative = cash_curve < 0
    n_negative = int(negative.sum())
    if not n_negative:
        return None
    lowest = float(cash_curve.min())
    rate = float(financing["margin_interest_rate"])
    day_count = financing["day_count"]
    charged = float(financing["interest_charged"])
    balance_years = float(financing["debit_balance_years"])
    head = (
        f"cash went negative on {n_negative} of {len(cash_curve)} bar(s), to a "
        f"low of {lowest:,.2f} — implied margin borrowing"
    )
    if rate > 0.0:
        return (
            f"{head}, charged at margin_interest_rate={rate:g} ({day_count}): "
            f"{charged:,.2f} of interest over the run."
        )
    per_point = balance_years * 0.01
    return (
        f"{head}, financed for nothing: margin_interest_rate=0. The debit "
        f"came to {balance_years:,.2f} currency-years ({day_count}), so each "
        f"1% of margin rate would have cost {per_point:,.2f}; set "
        "margin_interest_rate to what the broker charges, or read the return "
        "as an upper bound."
    )
