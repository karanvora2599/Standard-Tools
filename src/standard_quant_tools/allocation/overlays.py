"""
Risk overlays on a return stream: how much of the stream to hold each
period, decided from what was known the period before, and what that does
to its volatility, drawdown and floor.

Each overlay sets an exposure e_t -- the stream's weight in the account,
a multiple of equity -- at the start of period t from information through
t - 1 only, and the account earns

    e_t r_t + (1 - e_t) c - cost |e_t - e_{t-1}|

with c the per-period cash rate (cash_rate / periods_per_year), earned on
the part not in the stream and paid on the part above 1, and `cost` the
caller's bps per unit change in exposure. The first exposure is bought
from cash. Exposure is rebalanced to its target every period; the drift of
the position between rebalances is not traded, so turnover is
target-to-target.

VOL_TARGET. e_t = min(max_leverage, target / sigma_t), sigma_t the EWMA
volatility (RiskMetrics 1996: sigma^2_t = lambda sigma^2_{t-1} + (1 -
lambda) r^2_{t-1}, about a zero mean) of the returns through t - 1. The
estimate is seeded with the mean square of the first `warmup` returns, so
the overlaid stream starts after them. On i.i.d. returns the realized
volatility sits a few percent above the target: the noise in sigma_t
raises E[1 / sigma_t^2] above 1 / sigma^2 (about 3% at lambda = 0.94).

CPPI (Black & Jones 1987; Black & Perold 1992). Wealth V starts at 1 with
a floor F = floor x V that accrues at the cash rate. e_t = min(max_leverage,
m max(V - F, 0) / V): m times the cushion. A one-period loss r worse than
-1/m takes the cushion C to C (1 + m r) < 0 -- the floor is breached by
exactly that much -- and from then the exposure is zero and wealth and
floor both grow at the cash rate: the overlay is CASH-LOCKED for the rest
of the sample, which is what a gap does to CPPI and why it is reported.

TIPP (Estep & Kritzman 1988) is CPPI whose floor also ratchets up with the
peak: F_t = max(F_{t-1} (1 + c), floor x V_t), so gains are partly locked
in and the floor never falls.

The overlay reports and publishes; it does not trade, and nothing here
rebalances anything outside the numbers it returns.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import cagr, equity_from_returns
from standard_quant_tools.metrics.risk_metrics import max_drawdown

__all__ = ["METHODS", "apply_overlay"]

_WHO = "apply_overlay"

METHODS = ("vol_target", "cppi", "tipp")


def _vol_target(
    r: np.ndarray, target: float, lam: float, warmup: int, cap: float
) -> np.ndarray:
    """The exposure for each period after `warmup`, from returns before it."""
    variance = float(np.mean(r[:warmup] ** 2))
    out = np.empty(r.shape[0] - warmup)
    for j, t in enumerate(range(warmup, r.shape[0])):
        sigma = math.sqrt(variance)
        out[j] = cap if sigma == 0.0 else min(cap, target / sigma)
        variance = lam * variance + (1.0 - lam) * r[t] * r[t]
    return out


def _floor_overlay(
    r: np.ndarray,
    *,
    tipp: bool,
    multiplier: float,
    floor: float,
    cap: float,
    cash: float,
    cost: float,
) -> Dict[str, np.ndarray]:
    """CPPI or TIPP over every period: exposure, account return, wealth and
    floor, each decided from the wealth and floor at the period's start."""
    n = r.shape[0]
    exposure = np.empty(n)
    account = np.empty(n)
    wealth = np.empty(n)
    floors = np.empty(n)
    value, level, previous = 1.0, floor, 0.0
    for t in range(n):
        cushion = value - level
        # The cushion's share of wealth first: with no floor it is exactly
        # 1, so the exposure is exactly the multiplier.
        e = (
            0.0
            if cushion <= 0.0 or value <= 0.0
            else min(cap, multiplier * (cushion / value))
        )
        ret = e * r[t] + (1.0 - e) * cash - cost * abs(e - previous)
        value = value * (1.0 + ret)
        level = level * (1.0 + cash)
        if tipp:
            level = max(level, floor * value)
        exposure[t], account[t], wealth[t], floors[t] = e, ret, value, level
        previous = e
    return {"exposure": exposure, "account": account, "wealth": wealth, "floor": floors}


def _stats(returns: pd.Series, periods: int) -> Dict[str, float]:
    equity = equity_from_returns(returns)
    return {
        "volatility": float(returns.std(ddof=1)) * math.sqrt(periods),
        "annualized_return": float(cagr(equity, periods)),
        "max_drawdown": max(float(max_drawdown(equity)), -1.0),
    }


def apply_overlay(
    returns: pd.Series,
    *,
    method: str,
    periods_per_year: int,
    target_volatility: float = 0.10,
    ewma_lambda: float = 0.94,
    warmup: int = 20,
    multiplier: float = 3.0,
    floor: float = 0.8,
    max_leverage: float = 1.0,
    cash_rate: float = 0.0,
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    The overlaid stream, its exposure path, and what the overlay changed.

    `returns` is one dated per-period return series. `target_volatility`
    and `cash_rate` are annual decimals; `cost_bps` is charged per unit
    change in exposure; `floor` is a fraction of starting wealth (CPPI) or
    of peak wealth (TIPP). See the module docstring for each method.

    Returns a dict with `overlaid` and `exposure` (Series on the overlay's
    dates) and the summary figures: None where a figure does not apply to
    the method (a target for CPPI, a floor for vol_target), NaN where it
    applies and is undefined (the worst breach of a floor never breached).

    Raises:
        ValidationError: an unknown method, no more returns than the
        warm-up, or a parameter outside its range.
    """
    if method not in METHODS:
        raise ValidationError(
            f"{_WHO}: method={method!r}; expected one of {list(METHODS)}."
        )
    periods = int(periods_per_year)
    r = returns.to_numpy(dtype=float)
    n = r.shape[0]
    if not np.all(np.isfinite(r)):
        raise ValidationError(f"{_WHO}: the returns are not all finite.")
    cash = float(cash_rate) / periods
    cost = float(cost_bps) / 10_000.0
    cap = float(max_leverage)
    warnings: List[str] = []
    floors: Optional[np.ndarray] = None
    wealth: Optional[np.ndarray] = None

    if method == "vol_target":
        if n < warmup + 2:
            raise ValidationError(
                f"{_WHO}: {n} returns, and vol_target needs more than "
                f"warmup={warmup} + 1: the first {warmup} only seed the "
                "volatility estimate. Pass a longer stream or a shorter warmup."
            )
        start = int(warmup)
        exposure = _vol_target(
            r,
            float(target_volatility) / math.sqrt(periods),
            float(ewma_lambda),
            start,
            cap,
        )
        held = r[start:]
        previous = np.concatenate([[0.0], exposure[:-1]])
        account = (
            exposure * held
            + (1.0 - exposure) * cash
            - cost * np.abs(exposure - previous)
        )
    else:
        if n < 2:
            raise ValidationError(f"{_WHO}: {method} needs at least two returns.")
        start = 0
        path = _floor_overlay(
            r,
            tipp=method == "tipp",
            multiplier=float(multiplier),
            floor=float(floor),
            cap=cap,
            cash=cash,
            cost=cost,
        )
        exposure, account = path["exposure"], path["account"]
        wealth, floors = path["wealth"], path["floor"]
        held = r
        previous = np.concatenate([[0.0], exposure[:-1]])

    dates = returns.index[start:]
    overlaid = pd.Series(account, index=dates, name=str(returns.name or "stream"))
    exposure_series = pd.Series(exposure, index=dates, name="exposure")
    before = _stats(pd.Series(held, index=dates), periods)
    after = _stats(overlaid, periods)
    if np.any(account <= -1.0):
        first = dates[int(np.argmax(account <= -1.0))]
        warnings.append(
            f"{_WHO}: the overlaid account lost everything on "
            f"{first.date()} (a period's return at or below -100% at "
            f"exposure {float(exposure[int(np.argmax(account <= -1.0))]):.3g}); "
            "every figure after it is of a wiped-out account. Lower "
            "max_leverage."
        )
    changes = np.abs(exposure - previous)
    at_cap = float(np.mean(exposure >= cap - 1e-12))

    out: Dict[str, Any] = {
        "method": method,
        "overlaid": overlaid,
        "exposure": exposure_series,
        "n_periods": int(len(dates)),
        "warmup_periods": start,
        "first_date": dates[0],
        "last_date": dates[-1],
        "periods_per_year": periods,
        "target_volatility": (
            float(target_volatility) if method == "vol_target" else None
        ),
        "realized_volatility": after["volatility"],
        "stream_volatility": before["volatility"],
        "realized_over_target": (
            after["volatility"] / float(target_volatility)
            if method == "vol_target"
            else None
        ),
        "annualized_return_before": before["annualized_return"],
        "annualized_return_after": after["annualized_return"],
        "max_drawdown_before": before["max_drawdown"],
        "max_drawdown_after": after["max_drawdown"],
        "exposure_mean": float(exposure.mean()),
        "exposure_min": float(exposure.min()),
        "exposure_max": float(exposure.max()),
        "exposure_final": float(exposure[-1]),
        "share_at_max_leverage": at_cap,
        "turnover": float(changes.sum()),
        "turnover_per_year": float(changes.mean()) * periods,
        "cost_total": float((cost * changes).sum()),
        "cost_per_year": float((cost * changes).mean()) * periods,
        "floor_breaches": None,
        "worst_breach": None,
        "periods_below_floor": None,
        "cash_locked": None,
        "cash_locked_from": None,
        "final_floor": None,
        "warnings": warnings,
    }
    if floors is not None and wealth is not None:
        above = np.concatenate([[1.0 >= float(floor)], wealth[:-1] >= floors[:-1]])
        below = wealth < floors
        crossings = int(np.sum(above & below))
        out["floor_breaches"] = crossings
        out["periods_below_floor"] = int(below.sum())
        out["worst_breach"] = (
            float(np.min((wealth[below] - floors[below]) / floors[below]))
            if below.any()
            else math.nan
        )
        out["final_floor"] = float(floors[-1])
        # Locked: the cushion was gone at a decision and stayed gone, so the
        # exposure is zero from then to the end of the sample.
        cushion_gone = np.concatenate(
            [[1.0 <= float(floor)], wealth[:-1] <= floors[:-1]]
        )
        locked = bool(cushion_gone[-1])
        out["cash_locked"] = locked
        if locked:
            stuck = np.flatnonzero(~cushion_gone)
            first = int(stuck[-1]) + 1 if stuck.size else 0
            out["cash_locked_from"] = dates[first]
            warnings.append(
                f"{_WHO}: the cushion was gone from {dates[first].date()} and "
                "the exposure was zero from then to the end: the overlay is "
                "cash-locked, and wealth and floor both grow at the cash rate "
                "from there, so it cannot recover."
            )
        if crossings:
            warnings.append(
                f"{_WHO}: wealth fell below the floor {crossings} time(s), at "
                f"worst {out['worst_breach']:.4g} of it. A one-period loss "
                f"worse than -1/multiplier = {-1.0 / float(multiplier):.4g} "
                "breaches the floor whatever the cushion."
            )
    return out
