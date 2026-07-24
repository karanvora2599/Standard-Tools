"""
Liquidity and capacity diagnostics: how much of a ticker's own trading
volume a trade would consume, how long a position would take to unwind,
sector concentration, and an overall capacity estimate for a target-weight
portfolio. All pure functions — no engine changes required to use them
standalone; run_portfolio_simulation additionally wires adv_participation
into a max_adv_participation validation check (backtest/portfolio_engine.py).
"""

import logging
from typing import Any, Dict, List

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def adv_participation(notional: float, avg_dollar_volume: float) -> float:
    """
    Fraction of average dollar volume a trade's notional represents.
    Returns 0.0 when avg_dollar_volume <= 0 (no volume baseline available)
    rather than dividing by zero — the conservative "can't estimate, assume
    no constraint" fallback used elsewhere in this module.
    """
    if avg_dollar_volume <= 0:
        return 0.0
    return abs(notional) / avg_dollar_volume


def days_to_liquidate(
    shares: float, avg_daily_volume: float, max_participation: float
) -> float:
    """
    Estimated trading days to fully unwind a position without exceeding
    max_participation of the ticker's own average daily (share) volume.

    Raises:
        ValidationError: avg_daily_volume <= 0, or max_participation <= 0
        (both would make the estimate either undefined or infinite in a
        way that's more useful to surface as an error than silently return
        inf for).
    """
    if avg_daily_volume <= 0:
        raise ValidationError(f"avg_daily_volume must be > 0, got {avg_daily_volume}")
    if max_participation <= 0:
        raise ValidationError(f"max_participation must be > 0, got {max_participation}")
    tradeable_per_day = avg_daily_volume * max_participation
    return abs(shares) / tradeable_per_day


def sector_exposure(
    weights: Dict[str, float], sectors: Dict[str, str]
) -> Dict[str, float]:
    """
    Aggregate portfolio weight by sector. Tickers missing from `sectors`
    (or whose sector is the "Unknown" yfinance falls back to — see
    data/base.py's TickerInfo) are bucketed into "Unknown" rather than
    being silently dropped from the totals.
    """
    totals: Dict[str, float] = {}
    for ticker, weight in weights.items():
        sector = sectors.get(ticker, "Unknown")
        totals[sector] = totals.get(sector, 0.0) + weight
    return totals


def capacity_report(
    tickers: List[str],
    avg_dollar_volumes: Dict[str, float],
    target_weights: Dict[str, float],
    max_participation: float,
) -> Dict[str, Any]:
    """
    For each ticker, the maximum account size deployable at
    max_participation of its own average dollar volume, given its target
    weight — a name with a large target weight and thin volume caps the
    whole portfolio's capacity, not just its own position.

    Args:
        tickers: Universe.
        avg_dollar_volumes: {ticker: average dollar volume (Close * Volume,
            typically a rolling mean)}.
        target_weights: {ticker: target fraction of account equity}. A
            ticker with weight 0 imposes no capacity constraint (contributes
            float('inf') to that ticker's own max, excluded from the binding
            calculation).
        max_participation: Max fraction of avg_dollar_volume a single
            position may represent.

    Returns:
        Dict with per_ticker ({ticker: max_account_size}), binding_ticker
        (the name imposing the tightest constraint, None if every weight is
        0), max_account_size (the overall capacity — the smallest
        per-ticker max, or float('inf') if no ticker has a nonzero weight).

    Raises:
        ValidationError: any ticker missing from avg_dollar_volumes or
        target_weights, or max_participation <= 0.
    """
    if max_participation <= 0:
        raise ValidationError(f"max_participation must be > 0, got {max_participation}")
    missing_adv = [t for t in tickers if t not in avg_dollar_volumes]
    if missing_adv:
        raise ValidationError(
            f"avg_dollar_volumes is missing entries for: {missing_adv}"
        )
    missing_w = [t for t in tickers if t not in target_weights]
    if missing_w:
        raise ValidationError(f"target_weights is missing entries for: {missing_w}")

    per_ticker: Dict[str, float] = {}
    for t in tickers:
        weight = abs(target_weights[t])
        if weight <= 0:
            per_ticker[t] = float("inf")
            continue
        # max position dollar size = max_participation * avg_dollar_volume;
        # max account size = max position size / weight.
        per_ticker[t] = (max_participation * avg_dollar_volumes[t]) / weight

    finite = {t: v for t, v in per_ticker.items() if v != float("inf")}
    if not finite:
        binding_ticker = None
        max_account_size = float("inf")
    else:
        binding_ticker = min(finite, key=lambda t: finite[t])
        max_account_size = finite[binding_ticker]

    result = {
        "per_ticker": per_ticker,
        "binding_ticker": binding_ticker,
        "max_account_size": max_account_size,
    }
    logger.debug(
        "[constraints] capacity_report  binding=%s  max_account_size=%.0f",
        binding_ticker,
        max_account_size if max_account_size != float("inf") else -1.0,
    )
    return result
