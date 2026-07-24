"""
Pluggable transaction-cost model building blocks: commission, spread,
market impact, short-borrow fees, and margin interest — each a small pure
function so run_portfolio_simulation (or any caller) can compose exactly
the subset it needs. Today's existing flat commission_pct/slippage_pct
behavior is percentage_commission + fixed_bps_spread, kept as the default
everywhere for backward compatibility.
"""

import logging
import math

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def percentage_commission(notional: float, rate: float) -> float:
    """Commission as a fraction of trade notional — today's existing model."""
    return abs(notional) * rate


def per_share_commission(
    shares: float, rate_per_share: float, minimum: float = 0.0
) -> float:
    """
    Commission as a flat rate per share traded, with an optional minimum
    floor (e.g. many brokers charge max($0.005/share, $1.00) per order).
    """
    return max(abs(shares) * rate_per_share, minimum)


def fixed_bps_spread(notional: float, bps: float) -> float:
    """Spread cost as a fixed number of basis points of trade notional."""
    return abs(notional) * (bps / 10_000.0)


def pct_of_range_spread(
    notional: float, high: float, low: float, close: float, pct: float
) -> float:
    """
    Spread cost as a fraction of the bar's own (High - Low) range, scaled
    to notional terms via the bar's Close. No real bid/ask data exists in
    this library's OHLCV frames, so this is a documented proxy — the same
    spirit as run_strategy's fill_price="hl2_exploratory" mode.

    Raises:
        ValidationError: close <= 0 (can't scale a range into a fraction).
    """
    if close <= 0:
        raise ValidationError(f"close must be > 0, got {close}")
    range_frac = (high - low) / close
    return abs(notional) * range_frac * pct


def sqrt_impact_bps(
    participation: float, volatility: float, coefficient: float = 1.0
) -> float:
    """
    Square-root market impact model: impact_bps = coefficient * volatility
    * sqrt(participation), the standard form used in practitioner impact
    models (e.g. Almgren et al.). `participation` = order notional /
    average dollar volume (caller supplies avg_dollar_volume, typically a
    rolling mean of Close * Volume — no new data dependency, Volume is
    already present in every OHLCV frame). `volatility` is a per-bar
    (not annualized) return volatility.

    Raises:
        ValidationError: participation < 0 (can't take sqrt of a negative
        number — a negative participation is a caller error, not a valid
        trade size).
    """
    if participation < 0:
        raise ValidationError(f"participation must be >= 0, got {participation}")
    return coefficient * volatility * math.sqrt(participation) * 10_000.0


def impact_cost(
    notional: float,
    avg_dollar_volume: float,
    volatility: float,
    coefficient: float = 1.0,
) -> float:
    """
    Dollar impact cost for a trade, combining sqrt_impact_bps with the
    trade's own notional. avg_dollar_volume <= 0 (e.g. no volume data for
    an illiquid/synthetic ticker) returns 0.0 rather than dividing by zero
    — no impact estimate is possible without a volume baseline, and zero
    is the conservative "assume no impact model" fallback other cost
    functions in this module default to.
    """
    if avg_dollar_volume <= 0:
        return 0.0
    participation = abs(notional) / avg_dollar_volume
    bps = sqrt_impact_bps(participation, volatility, coefficient)
    return abs(notional) * (bps / 10_000.0)


def short_borrow_cost(notional: float, annual_bps: float, days: float = 1.0) -> float:
    """Daily-accrued borrow fee on a short position's notional."""
    return abs(notional) * (annual_bps / 10_000.0) * (days / 365.0)


def margin_interest(cash: float, annual_rate: float, days: float = 1.0) -> float:
    """
    Daily-accrued interest on negative cash (implied margin borrowing).
    Returns 0.0 when cash >= 0 — there's nothing borrowed to accrue
    interest on.
    """
    if cash >= 0:
        return 0.0
    return abs(cash) * annual_rate * (days / 365.0)
