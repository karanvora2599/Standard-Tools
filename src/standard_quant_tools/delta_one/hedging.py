"""
From a beta to a number of contracts, and whether the hedge worked.

THE TRANSLATION IS THE MISSING PIECE. This library has had the hedge
mathematics for a long time -- `calculate_beta`, `rolling_beta`,
`multi_factor_regression`, a Ledoit-Wolf covariance, marginal risk
contributions. What it has never had is the step that turns any of that
into something you can send to a broker:

    portfolio beta -> dollar beta -> contract notional -> multiplier
      -> contract count -> ROUNDING -> residual exposure

Every stage is arithmetic and none of it was written down, so it was being
done by hand or by a model reasoning about multipliers in its head. That is
exactly the kind of deterministic chain a library should own.

ROUNDING IS THE PART THAT MATTERS. A hedge of -903.2 contracts is not
available; -903 is. The 0.2 contracts left over is $62,000 of unhedged beta
at a 6200 index, and it is the number that decides whether the hedge is
finished or whether a second instrument is needed. Reporting only the
rounded count hides it, so everything here reports the exact figure, the
rounded figure and the residual, and never just one of them.

WHY EFFECTIVENESS IS A SEPARATE FUNCTION. `futures_hedge` says what to
trade under an assumption -- that the beta you handed it is the beta that
will hold. `hedge_effectiveness` is the test of that assumption against
history, and it is a different question with a different input (two return
series rather than two prices). A hedge ratio that was right on average and
unstable throughout is a hedge that did not work, and only the second
function can see it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import max_drawdown

__all__ = ["HEDGE_OBJECTIVES", "futures_hedge", "hedge_effectiveness", "tracking_error"]

#: What "hedged" is being taken to mean. The two are genuinely different
#: trades and the difference is the hedge instrument's own beta: matching
#: notional and matching risk coincide only when that beta is exactly 1.
HEDGE_OBJECTIVES: Dict[str, str] = {
    "beta_neutral": (
        "Zero net beta to the hedge instrument. Sizes on dollar beta, so a "
        "portfolio with beta 1.12 sells 12% more notional than it holds. "
        "This is what 'hedged' normally means."
    ),
    "dollar_neutral": (
        "Equal and opposite NOTIONAL, ignoring beta. Sells exactly the "
        "portfolio's market value. Correct when the portfolio is the index, "
        "and wrong by the beta gap otherwise -- a 1.3-beta book left "
        "dollar-neutral keeps 30% of its market exposure."
    ),
}

TRADING_DAYS = 252


def tracking_error(
    returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualized standard deviation of active return.

    THIS DID NOT EXIST as a function anywhere in the library. It was a local
    variable inside `information_ratio`, computed and then thrown away with
    the ratio -- so the denominator of the most-quoted active-risk statistic
    could not be inspected on its own. It is the whole answer to "how
    closely does this track", and a replication basket is judged on nothing
    else.

    Uses the SAMPLE standard deviation (ddof=1), matching `information_ratio`
    so the two cannot disagree about the same portfolio.
    """
    a = _aligned_returns(returns, benchmark_returns)
    if len(a) < 2:
        raise ValidationError(
            f"tracking error needs at least two overlapping observations, "
            f"got {len(a)}. With one there is no dispersion to measure."
        )
    active = a["portfolio"] - a["benchmark"]
    return float(active.std(ddof=1) * math.sqrt(periods_per_year))


def futures_hedge(
    *,
    portfolio_value: float,
    portfolio_beta: float,
    future_price: float,
    multiplier: float,
    future_beta: float = 1.0,
    objective: str = "beta_neutral",
    existing_contracts: float = 0.0,
) -> Dict[str, Any]:
    """
    The futures position that neutralizes a portfolio's market exposure.

        contracts = -(V * beta_p) / (F * multiplier * beta_f)

    `future_beta` is the hedge instrument's beta to the thing the portfolio
    was regressed against, and it defaults to 1.0 because that is true when
    the two are the same index. It is NOT true when hedging an S&P book
    with NQ, and leaving it at 1 there under-hedges by the ratio -- which
    is why it is an argument rather than a constant.

    `existing_contracts` makes this incremental. Passing what is already
    held returns the trade to do rather than the target, and the two are
    different numbers whenever a hedge is being adjusted rather than put on.
    """
    if objective not in HEDGE_OBJECTIVES:
        raise ValidationError(
            f"objective={objective!r} is not one of {sorted(HEDGE_OBJECTIVES)}."
        )
    v = _finite(portfolio_value, "portfolio_value")
    beta_p = _finite(portfolio_beta, "portfolio_beta")
    f = _finite(future_price, "future_price")
    m = _finite(multiplier, "multiplier")
    beta_f = _finite(future_beta, "future_beta")
    held = _finite(existing_contracts, "existing_contracts")

    if f <= 0 or m <= 0:
        raise ValidationError(
            f"future_price ({f!r}) and multiplier ({m!r}) must both be "
            "positive; one contract's notional is their product."
        )
    if objective == "beta_neutral" and beta_f == 0:
        raise ValidationError(
            "future_beta is zero, so this instrument has no exposure to the "
            "thing being hedged and no quantity of it neutralizes anything. "
            "The hedge is undefined rather than infinite."
        )

    contract_notional = f * m
    dollar_beta = v * beta_p

    if objective == "beta_neutral":
        exposure_per_contract = contract_notional * beta_f
        exact = -dollar_beta / exposure_per_contract
    else:
        exposure_per_contract = contract_notional
        exact = -v / contract_notional

    rounded = float(round(exact))
    trade_exact = exact - held
    trade_rounded = float(round(trade_exact))

    residual_dollar_beta = dollar_beta + rounded * contract_notional * beta_f
    post_beta = residual_dollar_beta / v if v else float("nan")

    warnings: List[str] = []
    if abs(exact) < 1.0:
        warnings.append(
            f"The exact hedge is {exact:.3f} contracts, so rounding to "
            f"{rounded:.0f} changes the hedge by "
            f"{abs(rounded - exact) / abs(exact) * 100 if exact else float('nan'):.0f}%. "
            "One contract is too coarse for this position -- a micro "
            "contract or a different instrument is the fix, not rounding."
        )
    else:
        residual_pct = (
            abs(residual_dollar_beta) / abs(dollar_beta) * 100.0 if dollar_beta else 0.0
        )
        if residual_pct > 1.0:
            warnings.append(
                f"Rounding leaves {residual_dollar_beta:,.0f} of dollar beta "
                f"unhedged, {residual_pct:.1f}% of the original exposure. "
                "That is the residual to decide about, not a rounding error."
            )
    if objective == "dollar_neutral" and abs(beta_p - 1.0) > 0.05:
        warnings.append(
            f"Dollar-neutral on a portfolio with beta {beta_p:.2f} leaves "
            f"{abs(beta_p - 1.0) * 100:.0f}% of market exposure standing. "
            "That may be intended, but 'hedged' usually means beta_neutral."
        )
    if beta_f != 1.0:
        warnings.append(
            f"Sized against a hedge instrument with beta {beta_f:.3f} to the "
            "portfolio's benchmark. If that beta is itself estimated, its "
            "error passes straight into the contract count."
        )
    warnings.append(
        "This is a POINT-IN-TIME hedge under the beta you supplied. Beta "
        "drifts, and the hedge decays with it -- hedge_effectiveness on the "
        "realized series is what says whether that drift mattered."
    )

    return {
        "objective": objective,
        "objective_meaning": HEDGE_OBJECTIVES[objective],
        "portfolio_value": v,
        "portfolio_beta": beta_p,
        "dollar_beta": float(dollar_beta),
        "future_price": f,
        "multiplier": m,
        "future_beta": beta_f,
        "contract_notional": float(contract_notional),
        "exposure_per_contract": float(exposure_per_contract),
        "contracts_exact": float(exact),
        "contracts_rounded": rounded,
        "existing_contracts": held,
        "trade_contracts_exact": float(trade_exact),
        "trade_contracts_rounded": trade_rounded,
        "hedge_notional": float(abs(rounded) * contract_notional),
        "pre_hedge_dollar_beta": float(dollar_beta),
        "post_hedge_dollar_beta": float(residual_dollar_beta),
        "pre_hedge_beta": beta_p,
        "post_hedge_beta": float(post_beta),
        "residual_dollar_beta": float(residual_dollar_beta),
        "warnings": warnings,
    }


def hedge_effectiveness(
    *,
    portfolio_returns: Sequence[float],
    hedge_returns: Sequence[float],
    hedge_ratio: float,
    window: Optional[int] = 60,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    Whether a hedge ratio actually removed risk, measured on realized returns.

    `hedge_ratio` is the units of the hedge instrument held per unit of
    portfolio, SIGNED -- a short hedge is negative. The hedged series is
    `portfolio + hedge_ratio * hedge`, so passing a positive ratio for what
    was meant to be a short hedge doubles the exposure instead of removing
    it, and the volatility comparison below will say so loudly.

    The rolling hedge ratio is the diagnostic most worth reading. A hedge
    whose ratio averaged 1.0 while ranging from 0.4 to 1.7 was never a
    hedge; it was two different positions that happened to average out, and
    the volatility reduction it shows in-sample will not repeat.
    """
    frame = _aligned_returns(portfolio_returns, hedge_returns)
    if len(frame) < 3:
        raise ValidationError(
            f"only {len(frame)} overlapping observations; hedge effectiveness "
            "cannot be judged from that. Sixty is a reasonable floor."
        )
    ratio = _finite(hedge_ratio, "hedge_ratio")

    unhedged = frame["portfolio"]
    hedge = frame["benchmark"]
    hedged = unhedged + ratio * hedge

    vol_before = float(unhedged.std(ddof=1) * math.sqrt(periods_per_year))
    vol_after = float(hedged.std(ddof=1) * math.sqrt(periods_per_year))

    beta_before = calculate_beta(unhedged, hedge)
    beta_after = calculate_beta(hedged, hedge)

    dd_before = float(max_drawdown((1.0 + unhedged).cumprod()))
    dd_after = float(max_drawdown((1.0 + hedged).cumprod()))

    rolling = None
    ratio_stability: Dict[str, Any] = {}
    if window and len(frame) > window:
        roll = rolling_beta(unhedged, hedge, window=window)["Rolling_Beta"].dropna()
        if len(roll):
            rolling = roll
            ratio_stability = {
                "mean": float(roll.mean()),
                "std": float(roll.std(ddof=1)) if len(roll) > 1 else float("nan"),
                "min": float(roll.min()),
                "max": float(roll.max()),
                "range": float(roll.max() - roll.min()),
                "sign_flips": int((np.sign(roll).diff().fillna(0) != 0).sum()),
            }

    te = float((unhedged - (-ratio) * hedge).std(ddof=1) * math.sqrt(periods_per_year))
    correlation = float(unhedged.corr(hedge))
    vol_reduction = (
        (1.0 - vol_after / vol_before) * 100.0 if vol_before else float("nan")
    )

    warnings: List[str] = []
    if vol_after >= vol_before:
        warnings.append(
            f"The hedge INCREASED volatility, from {vol_before:.2%} to "
            f"{vol_after:.2%}. The commonest cause is a sign error: "
            f"hedge_ratio is {ratio:+.4f}, and a short hedge must be "
            "negative. The second commonest is an instrument that does not "
            f"track -- correlation here is {correlation:.2f}."
        )
    if abs(correlation) < 0.7:
        warnings.append(
            f"Correlation between the two legs is {correlation:.2f}. Below "
            "about 0.7 the hedge instrument is not really tracking the "
            "portfolio, and the residual is a new position rather than a "
            "leftover."
        )
    if ratio_stability and ratio_stability.get("range", 0) > 0.5:
        warnings.append(
            f"The rolling hedge ratio ranged from "
            f"{ratio_stability['min']:.2f} to {ratio_stability['max']:.2f}. "
            "A single static ratio over a range that wide is an average of "
            "two regimes rather than a description of either, and the "
            "in-sample volatility reduction will not repeat out of sample."
        )
    if ratio_stability.get("sign_flips"):
        warnings.append(
            f"The rolling beta changed sign {ratio_stability['sign_flips']} "
            "time(s). A hedge instrument whose relationship inverts is not a "
            "hedge over the whole window, whatever the full-sample number says."
        )
    warnings.append(
        "Measured IN SAMPLE on the ratio supplied. A ratio fitted on this "
        "same window will always look effective here; the rolling stability "
        "above is what indicates whether it would have held."
    )

    return {
        "n_observations": int(len(frame)),
        "hedge_ratio": ratio,
        "volatility_before": vol_before,
        "volatility_after": vol_after,
        "volatility_reduction_pct": float(vol_reduction),
        "beta_before": float(beta_before["beta"]),
        "beta_after": float(beta_after["beta"]),
        "r_squared_before": float(beta_before["r_squared"]),
        "correlation": correlation,
        "tracking_error": te,
        "max_drawdown_before": dd_before,
        "max_drawdown_after": dd_after,
        "drawdown_reduction_pct": (
            float((1.0 - abs(dd_after) / abs(dd_before)) * 100.0)
            if dd_before
            else float("nan")
        ),
        "rolling_hedge_ratio": ratio_stability or None,
        "window": window if rolling is not None else None,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return out


def _aligned_returns(portfolio: Any, benchmark: Any) -> pd.DataFrame:
    """
    Two return series joined on their shared index, non-finite rows dropped.

    An INNER join, deliberately. A date one leg observed and the other did
    not is a missing comparison, not a zero one, and filling it would put a
    real portfolio return against a fabricated zero hedge return on exactly
    the days a hedge is judged by.
    """
    p = _as_series(portfolio, "portfolio_returns")
    b = _as_series(benchmark, "benchmark_returns")
    frame = pd.concat({"portfolio": p, "benchmark": b}, axis=1, join="inner")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValidationError(
            "the two return series share no usable observations. Check that "
            "they cover the same dates and are returns rather than prices."
        )
    return frame


def _as_series(values: Any, name: str) -> pd.Series:
    if values is None:
        raise ValidationError(f"{name} is required and was not given")
    if isinstance(values, pd.Series):
        return values.astype("float64")
    try:
        return pd.Series(list(values), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a sequence of numbers; {exc}") from None
