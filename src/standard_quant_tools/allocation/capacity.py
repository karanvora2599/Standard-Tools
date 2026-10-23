"""
How much capital a strategy can take: its net return and Sharpe ratio as
assets under management grow and its own trades start moving prices.

THE COST. Every change in the weight panel is a trade. At AUM A, a change
of |dw| in instrument i costs, as a fraction of AUM,

    |dw| (s_i / 2 + k sigma_i sqrt(A |dw| / ADV_i))

-- half the quoted spread s_i, plus square-root market impact: k times the
instrument's daily volatility times the square root of the trade's share
of its average daily volume. The cost of a period is the sum over its
trades, and net return is gross return minus it, period by period.

THE IMPACT COEFFICIENT. k defaults to 1: the square-root law's prefactor is
measured of order one across markets and decades (Toth et al. 2011,
"Anomalous price impact and the critical nature of liquidity"; Bouchaud,
Bonart, Donier & Gould 2018, "Trades, Quotes and Prices", ch. 12), and it
is the default of the library's own `backtest.costs.sqrt_impact_bps`. It
is a model, not a measurement of these instruments; pass a calibrated one.

BREAK-EVEN IN CLOSED FORM. The mean cost is a + b sqrt(A) -- a the mean
spread cost, b the mean of sum_i k sigma_i |dw|^1.5 / sqrt(ADV_i) -- so
the mean net return is zero at

    A* = ((mean gross - a) / b)^2.

For one instrument trading |dw| = tau every period that is (ADV / tau)
((alpha / tau - s/2) / (k sigma))^2. The AUM at half the gross Sharpe
ratio has no closed form, because the cost's own variation moves the
volatility; it is solved on sqrt(A) between zero and the break-even.

Which instruments bind is AUM-free in this model: every impact cost scales
with sqrt(A), so each instrument's share of the impact is the same at any
size. Their participation is reported at the break-even AUM; above about
20% of a day's volume the square-root law is outside the range it was
measured on, and the result says so.

WHAT IS ASSUMED. The weights are targets as fractions of AUM; the trade
into the weights dated d is made, and charged, on the first return date on
or after d; the first weight row is bought from cash; the drift of the
positions between weight dates is not traded. The ADV, volatility and
spread are the caller's, one figure per instrument for the whole sample.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError

__all__ = [
    "DEFAULT_IMPACT_COEFFICIENT",
    "PARTICIPATION_LIMIT",
    "capacity_curve",
]

_WHO = "capacity_curve"

#: k in k sigma sqrt(participation); see the module docstring.
DEFAULT_IMPACT_COEFFICIENT = 1.0

#: A day's participation above which the square-root law is extrapolated.
PARTICIPATION_LIMIT = 0.2

#: Points in the default AUM grid, from a thousandth of the break-even to
#: ten times it.
_GRID_POINTS = 25

#: Instruments listed as binding.
_BINDING_SHOWN = 10


def _instrument_map(
    values: Mapping[str, float], field: str, names: List[str]
) -> np.ndarray:
    missing = [n for n in names if n not in values]
    extra = [n for n in values if n not in set(names)]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"lacks {missing[:5]}, which the weights hold")
        if extra:
            parts.append(f"names {extra[:5]}, which the weights do not hold")
        raise ValidationError(
            f"{_WHO}: {field} must name exactly the weights' instruments; it "
            f"{' and '.join(parts)}. Give one figure per instrument."
        )
    return np.array([float(values[n]) for n in names], dtype=float)


def _trades(returns: pd.Series, weights: pd.DataFrame) -> Dict[str, Any]:
    """
    The weight changes as trades on the return dates: |dw| per weight row,
    the return date each is charged on, and notes on rows that fall
    outside the returns.
    """
    notes: List[str] = []
    frame = weights.sort_index().fillna(0.0)
    dates = returns.index
    early = frame.index < dates[0]
    if early.sum() > 1:
        notes.append(
            f"{_WHO}: {int(early.sum()) - 1} weight row(s) before the first "
            f"return ({dates[0].date()}) were dropped; the last of them is the "
            "position held when the returns start, bought from cash then."
        )
        keep = ~early
        keep[np.flatnonzero(early)[-1]] = True
        frame = frame[keep]
    late = frame.index > dates[-1]
    if late.any():
        notes.append(
            f"{_WHO}: {int(late.sum())} weight row(s) after the last return "
            f"({dates[-1].date()}) have no return to charge their trade to and "
            "were dropped."
        )
        frame = frame[~late]
    if frame.empty:
        raise ValidationError(
            f"{_WHO}: no weight row falls on or before the returns' last date "
            f"({dates[-1].date()}); the weights and the returns do not overlap."
        )
    held = frame.to_numpy(dtype=float)
    change = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1]))))
    where = dates.searchsorted(frame.index, side="left")
    return {"change": change, "where": where, "notes": notes, "rows": len(frame)}


def _sharpe(net: np.ndarray, rate: float, periods: int) -> float:
    sd = float(np.std(net, ddof=1))
    if not sd > 0.0:
        return math.nan
    return float(np.mean(net) - rate) / sd * math.sqrt(periods)


def capacity_curve(
    returns: pd.Series,
    weights: pd.DataFrame,
    *,
    adv: Mapping[str, float],
    daily_volatility: Mapping[str, float],
    spread_bps: Mapping[str, float],
    periods_per_year: int,
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
    aum: Optional[Sequence[float]] = None,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Net return and Sharpe ratio against AUM, the AUM at half the gross
    Sharpe, the break-even AUM, and the instruments whose impact binds.

    `returns` is the strategy's gross per-period return stream, dated;
    `weights` a date-by-instrument frame of target weights (fractions of
    AUM); `adv` is average daily volume in currency, `daily_volatility` a
    decimal and `spread_bps` the full quoted spread, all keyed by the
    weights' instruments. `aum` defaults to a grid around the break-even.

    Raises:
        ValidationError: a per-instrument map that names other instruments
        than the weights, a non-positive ADV, weights and returns that do
        not overlap, or an AUM that is not positive.
    """
    names = [str(c) for c in weights.columns]
    if not names:
        raise ValidationError(f"{_WHO}: the weights name no instrument.")
    volume = _instrument_map(adv, "adv", names)
    vol = _instrument_map(daily_volatility, "daily_volatility", names)
    spread = _instrument_map(spread_bps, "spread_bps", names) / 10_000.0
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        bad = [names[i] for i in np.flatnonzero(~(volume > 0.0))]
        raise ValidationError(
            f"{_WHO}: adv must be a positive volume in currency for every "
            f"instrument; {bad[:5]} are not. An instrument with no volume has "
            "no capacity to estimate -- drop it or supply its volume."
        )
    if np.any(vol < 0.0) or np.any(spread < 0.0):
        raise ValidationError(f"{_WHO}: daily_volatility and spread_bps must be >= 0.")
    k = float(impact_coefficient)
    periods = int(periods_per_year)
    rate = float(risk_free_rate) / periods
    gross = returns.to_numpy(dtype=float)
    n = gross.shape[0]
    if n < 3:
        raise ValidationError(f"{_WHO}: {n} returns; a Sharpe ratio needs at least 3.")

    trades = _trades(returns, weights)
    change = trades["change"]
    where = trades["where"]
    # Products over every weight row: on one BLAS thread, so a wide panel's
    # costs are the same bits whatever the caller's thread count.
    with single_threaded_blas():
        spread_row = change @ (spread / 2.0)
        impact_row = (change**1.5) @ (k * vol / np.sqrt(volume))
    spread_t = np.bincount(where, weights=spread_row, minlength=n)
    impact_t = np.bincount(where, weights=impact_row, minlength=n)

    mean_gross = float(gross.mean())
    a = float(spread_t.mean())
    b = float(impact_t.mean())
    gross_sharpe = _sharpe(gross, rate, periods)
    warnings: List[str] = list(trades["notes"])

    breakeven = math.nan
    if mean_gross <= a:
        breakeven = 0.0
        warnings.append(
            f"{_WHO}: the spread alone costs {a * periods:.4g} a year against "
            f"a gross mean of {mean_gross * periods:.4g}: no AUM breaks even, "
            "so the break-even AUM is 0."
        )
    elif b > 0.0:
        breakeven = ((mean_gross - a) / b) ** 2
    else:
        warnings.append(
            f"{_WHO}: no impact grows with AUM -- nothing is ever traded, or "
            "every traded instrument has zero volatility -- so no AUM breaks "
            "even and the curve is flat."
        )

    def sharpe_at(root_aum: float) -> float:
        return _sharpe(gross - spread_t - impact_t * root_aum, rate, periods)

    half_aum = math.nan
    if gross_sharpe > 0.0:
        target = gross_sharpe / 2.0
        if not sharpe_at(0.0) > target:
            half_aum = 0.0
        elif b > 0.0:
            high = math.sqrt(breakeven) if breakeven > 0.0 else 1.0
            for _ in range(200):
                if sharpe_at(high) < target:
                    break
                high *= 2.0
            grid = np.linspace(0.0, high, 65)
            values = np.array([sharpe_at(u) for u in grid])
            cross = int(np.argmax(values < target))
            if values[cross] < target:
                root = brentq(
                    lambda u: sharpe_at(u) - target,
                    float(grid[cross - 1]),
                    float(grid[cross]),
                    xtol=1e-12,
                    rtol=1e-14,
                    maxiter=500,
                )
                half_aum = root * root

    if aum is None:
        centre = breakeven if breakeven > 0.0 else half_aum
        if not (centre > 0.0 and math.isfinite(centre)):
            centre = 1e8
        grid_aum = list(np.geomspace(centre / 1000.0, centre * 10.0, _GRID_POINTS))
    else:
        grid_aum = [float(x) for x in aum]
        bad = [x for x in grid_aum if not (math.isfinite(x) and x > 0.0)]
        if bad:
            raise ValidationError(
                f"{_WHO}: aum must be positive amounts of currency; got {bad[:5]}."
            )
        grid_aum = sorted(grid_aum)

    curve: List[Dict[str, float]] = []
    for size in grid_aum:
        root = math.sqrt(size)
        cost = spread_t + impact_t * root
        net = gross - cost
        curve.append(
            {
                "aum": size,
                "net_return": float(net.mean()) * periods,
                "net_sharpe": _sharpe(net, rate, periods),
                "cost_per_year": float(cost.mean()) * periods,
                "impact_cost_per_year": b * root * periods,
                "max_participation": float(np.max(change * size / volume)),
            }
        )

    reference = breakeven if breakeven > 0.0 else grid_aum[-1]
    impact_by = (change**1.5).sum(axis=0) * k * vol / np.sqrt(volume)
    spread_by = change.sum(axis=0) * spread / 2.0
    total_impact = float(impact_by.sum())
    cost_by = spread_by + impact_by * math.sqrt(reference)
    total_cost = float(cost_by.sum())
    participation = (change * reference / volume).max(axis=0)
    years = n / periods
    binding: List[Dict[str, Any]] = []
    for i in np.argsort(-impact_by, kind="stable")[:_BINDING_SHOWN]:
        binding.append(
            {
                "instrument": names[i],
                "impact_share": (
                    float(impact_by[i]) / total_impact if total_impact > 0 else math.nan
                ),
                "cost_share": (
                    float(cost_by[i]) / total_cost if total_cost > 0 else math.nan
                ),
                "turnover_per_year": float(change[:, i].sum()) / years,
                "max_participation": float(participation[i]),
            }
        )
    over = [names[i] for i in np.flatnonzero(participation > PARTICIPATION_LIMIT)]
    if over:
        warnings.append(
            f"{_WHO}: at {reference:,.0f} of AUM {over[:5]} trade more than "
            f"{PARTICIPATION_LIMIT:.0%} of a day's volume in one trade, beyond "
            "the range the square-root law was measured on; the curve there "
            "is an extrapolation."
        )

    return {
        "n_periods": n,
        "n_instruments": len(names),
        "n_weight_rows": trades["rows"],
        "periods_per_year": periods,
        "impact_coefficient": k,
        "gross_return": mean_gross * periods,
        "gross_sharpe": gross_sharpe,
        "spread_cost_per_year": a * periods,
        "turnover_per_year": float(change.sum()) / years,
        "breakeven_aum": breakeven,
        "half_sharpe_aum": half_aum,
        "reference_aum": reference,
        "curve": curve,
        "binding": binding,
        "warnings": warnings,
    }
