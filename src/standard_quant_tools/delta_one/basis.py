"""
Spot against a quoted future: what the difference is, and whether it is big.

TWO QUESTIONS THAT LOOK LIKE ONE. "What is the basis" is a valuation
question with an arithmetic answer, and `cash_futures_basis` answers it.
"Is this basis wide" is a completely different question -- it needs the
name's own history, because ES at 38 bps over fair means nothing until you
know it has spent a year between -5 and +25. `basis_history` answers that
one. Keeping them apart matters because the first is available from a
single quote and the second needs a series, and a tool that demanded a
series to answer the first would refuse most of the questions asked of it.

WHY THE DECOMPOSITION IS THE POINT. A single "the future is 9 points rich"
is nearly useless, because the interesting question is always which term
disagrees. `analysis.derivatives.implied_forward_price` already breaks
carry into financing, dividend and borrow and this module carries that
split through to the answer, then adds the number a trader actually acts
on: the financing rate that would make the quoted price fair. If that
comes back at 5.4% against a 4.1% SOFR, the future is not rich -- funding
is expensive, and the trade is a funding trade rather than an arbitrage.

WHAT A WIDE BASIS USUALLY IS. In order of frequency: a stale or
non-simultaneous spot print, a wrong dividend assumption, a borrow move,
a genuine funding dislocation, and only then something worth trading. The
warnings say so, because a tool that reports 9 points rich without that
ordering trains a reader to reach for the last explanation first.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.cointegration import half_life, spread_zscore
from standard_quant_tools.analysis.derivatives import _positive
from standard_quant_tools.error import ValidationError

from .carry import forward_price, observed_carry_rate, solve_carry

__all__ = ["basis_history", "cash_futures_basis"]

#: Default richness tolerance, in annualized basis points of the carry
#: spread. 25 bps matches `check_put_call_parity`'s tolerance and is set
#: where it is for the same reason: below it, the difference is
#: indistinguishable from a non-simultaneous spot print, and calling that
#: "rich" would flag every quote on a moving screen.
DEFAULT_TOLERANCE_BPS = 25.0


def cash_futures_basis(
    *,
    spot: float,
    future_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    borrow_rate: float = 0.0,
    tolerance_bps: float = DEFAULT_TOLERANCE_BPS,
) -> Dict[str, Any]:
    """
    A quoted future against its carry-fair value, with the cause attributed.

    Returns the basis three ways because they answer different questions:
    in POINTS of the underlying (what the screen shows), as an annualized
    RATE (what is comparable across expiries), and as the implied FINANCING
    (what a funding desk would quote). A single number in points cannot be
    compared between a March and a December contract; the annualized spread
    can.
    """
    s = _positive(spot, "spot")
    f = _positive(future_price, "future_price")
    t = _positive(time_to_expiry, "time_to_expiry")
    tolerance = float(tolerance_bps)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValidationError(
            f"tolerance_bps must be finite and non-negative, got {tolerance_bps!r}."
        )

    fair = forward_price(
        spot=s,
        time_to_expiry=t,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        borrow_rate=borrow_rate,
    )
    fair_future = float(fair["forward"])
    fair_carry = float(fair["net_carry_rate"])
    market_carry = observed_carry_rate(spot=s, forward=f, time_to_expiry=t)

    carry_spread = market_carry - fair_carry
    spread_bps = carry_spread * 10_000.0

    if abs(spread_bps) <= tolerance:
        classification = "fair"
    elif spread_bps > 0:
        classification = "future_rich"
    else:
        classification = "future_cheap"

    implied = solve_carry(
        spot=s,
        forward=f,
        time_to_expiry=t,
        solve_for="financing_rate",
        dividend_yield=dividend_yield,
        borrow_rate=borrow_rate,
    )

    warnings: List[str] = []
    if classification != "fair":
        warnings.append(
            f"The future is {abs(spread_bps):.0f} bps annualized "
            f"{'rich' if spread_bps > 0 else 'cheap'} to carry, which is "
            f"{abs(f - fair_future):.2f} points. Before treating that as "
            "edge, rule out the likelier causes in order: a spot print that "
            "is not simultaneous with the future, a wrong dividend "
            "assumption, and a borrow that has moved. Genuine dislocation "
            "is last on that list, not first."
        )
    if t < 7.0 / 365.0:
        warnings.append(
            f"Time to expiry is {t * 365:.1f} days. Annualizing a basis over "
            "a period this short multiplies both the signal and every "
            "pricing error by roughly "
            f"{1 / t:.0f}x, so the annualized figure is unstable even when "
            "the points basis is not."
        )
    if dividend_yield == 0.0:
        warnings.append(
            "dividend_yield was zero. On an index or a paying single name "
            "that pushes the entire dividend into the basis and the future "
            "will read cheap by roughly the yield."
        )

    warnings.extend(fair["warnings"])

    return {
        "spot": float(s),
        "future": float(f),
        "fair_future": fair_future,
        "time_to_expiry": float(t),
        "observed_basis_points": float(f - s),
        "fair_basis_points": float(fair_future - s),
        "basis_spread_points": float(f - fair_future),
        "observed_carry_rate": float(market_carry),
        "fair_carry_rate": float(fair_carry),
        "carry_spread_rate": float(carry_spread),
        "annualized_basis_spread_bps": float(spread_bps),
        "classification": classification,
        "tolerance_bps": tolerance,
        "implied_financing_rate": float(implied["solved_rate"]),
        "implied_financing_bps": float(implied["solved_rate_bps"]),
        "components": {
            "financing_points": float(fair["components"]["financing"]),
            "dividend_points": float(fair["components"]["dividend"]),
            "borrow_points": float(fair["components"]["borrow"]),
        },
        "warnings": warnings,
    }


def basis_history(
    *,
    spot: Sequence[float],
    futures: Sequence[float],
    window: Optional[int] = None,
    time_to_expiry: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    A basis series turned into the statistics that make it tradeable.

    THE BASIS IS MEASURED IN BPS OF SPOT, not in points. Points are not
    comparable through time on anything that has moved -- 20 points on an
    index at 3,000 and at 6,000 are different trades -- and a z-score of a
    points series on a trending underlying mostly measures the trend.

    The z-score is the number to read, and `window` decides whether it is
    honest. With a window it is rolling and point-in-time. Without one it
    is computed against the full sample, which uses the end of the series
    to judge the middle of it: fine for describing history, look-ahead in a
    backtest. Defaults to full-sample because the common use is description
    and the warning says which one you got.
    """
    s = _series(spot, "spot")
    f = _series(futures, "futures")
    if len(s) != len(f):
        raise ValidationError(
            f"spot has {len(s)} observations and futures has {len(f)}. They "
            "must be aligned and the same length -- this function cannot "
            "tell which end of the shorter one is missing."
        )
    if (s <= 0).any():
        raise ValidationError(
            "spot contains a non-positive price, so the basis in bps of spot "
            "is undefined there."
        )
    if len(s) < 3:
        raise ValidationError(
            f"only {len(s)} observations; a basis distribution needs more "
            "than that to mean anything. Twenty is a reasonable floor."
        )

    basis_points = f - s
    basis_bps = (f / s - 1.0) * 10_000.0

    annualized_bps: Optional[pd.Series] = None
    if time_to_expiry is not None:
        t = _series(time_to_expiry, "time_to_expiry")
        if len(t) != len(s):
            raise ValidationError(
                f"time_to_expiry has {len(t)} observations against "
                f"{len(s)} prices; they must be aligned."
            )
        if (t <= 0).any():
            raise ValidationError(
                "time_to_expiry contains a non-positive value. A contract at "
                "or past expiry has no annualized basis."
            )
        annualized_bps = np.log(f / s) / t * 10_000.0

    target = annualized_bps if annualized_bps is not None else basis_bps
    zscores = spread_zscore(target, window=window)
    current = float(target.iloc[-1])
    finite = target.dropna()

    warnings: List[str] = []
    if window is None:
        warnings.append(
            "The z-score is FULL-SAMPLE: its mean and standard deviation "
            "include observations after each point, so it describes history "
            "but cannot be used as a point-in-time signal. Pass `window` for "
            "a rolling z-score that only looks backward."
        )
    else:
        warnings.append(
            f"Rolling {window}-observation z-score, so the first {window - 1} "
            "values are undefined rather than zero."
        )
    if annualized_bps is None:
        warnings.append(
            "Basis is in bps of spot, NOT annualized -- no time_to_expiry "
            "was given. A series that spans a roll therefore steps when the "
            "contract changes, because a near contract and a far one carry "
            "different amounts of time, not different amounts of richness."
        )

    hl = half_life(target.dropna())
    if not math.isfinite(hl):
        warnings.append(
            "Half-life is undefined: the fitted AR(1) coefficient is not "
            "negative, so this basis series shows no mean reversion over the "
            "window given. Trading it as a spread assumes reversion that the "
            "data does not show."
        )

    return {
        "n_observations": int(len(s)),
        "current_basis_bps": current,
        "current_basis_points": float(basis_points.iloc[-1]),
        "mean_bps": float(finite.mean()),
        "std_bps": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
        "min_bps": float(finite.min()),
        "max_bps": float(finite.max()),
        "zscore": float(zscores.iloc[-1]) if len(zscores.dropna()) else float("nan"),
        "percentile": float((finite < current).mean() * 100.0),
        "half_life_observations": float(hl),
        "annualized": annualized_bps is not None,
        "window": window,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _series(values: Sequence[float], name: str) -> pd.Series:
    """A clean float Series, refusing what cannot be arithmetic."""
    if values is None:
        raise ValidationError(f"{name} is required and was not given")
    try:
        series = pd.Series(list(values), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a sequence of numbers; {exc}") from None
    if series.empty:
        raise ValidationError(f"{name} is empty.")
    if not np.isfinite(series.to_numpy()).all():
        raise ValidationError(
            f"{name} contains a non-finite value. A NaN in one leg would "
            "propagate silently into the basis and then into its mean."
        )
    return series
