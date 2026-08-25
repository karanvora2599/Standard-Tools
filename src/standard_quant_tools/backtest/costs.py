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


def _cost_rate(name: str, value: float, allow_negative: bool = False) -> float:
    """
    Validate one cost parameter.

    Every function in this module is a bare arithmetic expression, so a
    negative rate does not fail — it returns a NEGATIVE COST, which downstream
    is indistinguishable from a rebate. Measured before this guard existed:

        percentage_commission(1e6, rate=-0.001)  -> -1000.0
        fixed_bps_spread(1e6, bps=-10)           -> -1000.0
        short_borrow_cost(1e6, annual_bps=-500)  -> -4109.59

    A backtest charging negative commission earns money by trading, which
    flatters every strategy that turns over more. If a genuine rebate is
    intended it should be modelled explicitly rather than arriving as a
    sign error, so these are rejected.

    NaN is checked before the sign, because `value < 0` is False for NaN —
    the comparison-shaped guard would have passed it straight through.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__} ({value!r})"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            f"{name} must be finite, got {value!r}. NaN compares False against "
            "every bound, so it would pass a sign check and silently produce a "
            "NaN cost."
        )
    if not allow_negative and number < 0:
        raise ValidationError(
            f"{name} must be >= 0, got {number}. A negative cost is a credit: "
            "the backtest would be paid to trade, flattering exactly the "
            "strategies that turn over most. Model a genuine rebate explicitly "
            "rather than as a negative cost."
        )
    return number


def percentage_commission(notional: float, rate: float) -> float:
    """Commission as a fraction of trade notional — today's existing model."""
    _cost_rate("rate", rate)
    _cost_rate("notional", notional, allow_negative=True)
    return abs(notional) * rate


def per_share_commission(
    shares: float, rate_per_share: float, minimum: float = 0.0
) -> float:
    """
    Commission as a flat rate per share traded, with an optional minimum
    floor (e.g. many brokers charge max($0.005/share, $1.00) per order).

    A zero-share trade costs 0.0, not `minimum` — the floor is a per-ORDER
    minimum, and no order is placed when nothing is traded. (run_portfolio_
    simulation already skips zero-size trades before reaching here, but this
    function is public and documented as composable, so it must not invent a
    commission for a trade that never happened.)
    """
    _cost_rate("rate_per_share", rate_per_share)
    _cost_rate("minimum", minimum)
    _cost_rate("shares", shares, allow_negative=True)
    if shares == 0:
        return 0.0
    return max(abs(shares) * rate_per_share, minimum)


def fixed_bps_spread(notional: float, bps: float) -> float:
    """Spread cost as a fixed number of basis points of trade notional."""
    _cost_rate("bps", bps)
    _cost_rate("notional", notional, allow_negative=True)
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
    _cost_rate("pct", pct)
    _cost_rate("high", high)
    _cost_rate("low", low)
    _cost_rate("close", close)
    if high < low:
        raise ValidationError(
            f"high ({high}) is below low ({low}) — the bar's range is inverted, "
            "so the spread proxy derived from it would be negative."
        )
    if close <= 0:
        raise ValidationError(f"close must be > 0, got {close}")
    range_frac = (high - low) / close
    return abs(notional) * range_frac * pct


def sqrt_impact_bps(
    participation: float, volatility: float, coefficient: float = 1.0
) -> float:
    """
    Square-root market impact model, returned in BASIS POINTS:
    impact_bps = coefficient * volatility * sqrt(participation) * 10_000
    — the standard form used in practitioner impact models (e.g. Almgren et
    al.), with the trailing 1e4 converting the fractional impact into the bps
    unit this function's name promises (impact_cost below divides it back
    out). `participation` = order notional /
    average dollar volume (caller supplies avg_dollar_volume, typically a
    rolling mean of Close * Volume — no new data dependency, Volume is
    already present in every OHLCV frame). `volatility` is a per-bar
    (not annualized) return volatility.

    Raises:
        ValidationError: participation < 0 (can't take sqrt of a negative
        number — a negative participation is a caller error, not a valid
        trade size).
    """
    _cost_rate("participation", participation)
    _cost_rate("volatility", volatility)
    _cost_rate("coefficient", coefficient)
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
    trade's own notional.

    Returns NaN — "not estimable" — when no usable volume baseline exists
    (non-positive or non-finite avg_dollar_volume). It used to return 0.0
    and describe that as conservative; it is the opposite:

        impact_cost(1e9, adv=0,   vol=0.30) -> 0.0            (looked free)
        impact_cost(1e9, adv=1e7, vol=0.30) -> 3,000,000,000  (honest)

    So the ticker with NO liquidity data was scored as the cheapest in the
    universe to trade, and a capacity report would happily route size into
    exactly the names it knew least about. NaN cannot be mistaken for a
    measurement; 0.0 can.
    """
    if not math.isfinite(avg_dollar_volume) or avg_dollar_volume <= 0:
        return float("nan")
    participation = abs(notional) / avg_dollar_volume
    bps = sqrt_impact_bps(participation, volatility, coefficient)
    return abs(notional) * (bps / 10_000.0)


def short_borrow_cost(notional: float, annual_bps: float, days: float = 1.0) -> float:
    """Daily-accrued borrow fee on a short position's notional."""
    _cost_rate("annual_bps", annual_bps)
    _cost_rate("days", days)
    return abs(notional) * (annual_bps / 10_000.0) * (days / 365.0)


def margin_interest(cash: float, annual_rate: float, days: float = 1.0) -> float:
    """
    Daily-accrued interest on negative cash (implied margin borrowing).
    Returns 0.0 when cash >= 0 — there's nothing borrowed to accrue
    interest on.
    """
    _cost_rate("annual_rate", annual_rate)
    _cost_rate("days", days)
    if cash >= 0:
        return 0.0
    return abs(cash) * annual_rate * (days / 365.0)


def directional_commission(
    notional: float, buy_rate: float, sell_rate: float, is_buy: bool
) -> float:
    """
    Commission that differs by side.

    `percentage_commission` charges the same rate whichever way the trade
    goes, which is not what US equity execution actually costs. The SEC
    Section 31 fee and the FINRA TAF are levied on SALES only, so a
    round trip is not symmetric and a strategy whose turnover is
    sell-heavy is undercharged by a single blended rate. The same asymmetry
    shows up wherever a venue prices the two sides differently.

    Both rates must be >= 0: this is a commission on either side, and a
    negative one is the credit `_cost_rate` exists to reject. For a genuine
    rebate use `maker_taker_cost`, which takes it deliberately.

    Args:
        notional:  Trade notional. Signed or unsigned -- the magnitude is
                   used, and `is_buy` carries the direction, because
                   inferring side from the sign of a notional is how a
                   short sale gets charged as a buy.
        buy_rate:  Fraction of notional charged when buying.
        sell_rate: Fraction of notional charged when selling.
        is_buy:    True for a buy, False for a sell.
    """
    _cost_rate("notional", notional, allow_negative=True)
    _cost_rate("buy_rate", buy_rate)
    _cost_rate("sell_rate", sell_rate)
    return abs(notional) * (buy_rate if is_buy else sell_rate)


def maker_taker_cost(
    notional: float, taker_rate: float, maker_rate: float, is_maker: bool
) -> float:
    """
    Maker/taker pricing, where the maker rate MAY be a rebate.

    This is the one function in this module that accepts a negative rate,
    and it is separate from the others precisely so that a negative cost
    can only arrive on purpose. `_cost_rate` rejects negatives everywhere
    else because a credit flatters exactly the strategies that turn over
    most; a maker rebate is a real credit, so it is modelled here rather
    than smuggled in as a sign error somewhere that reads it as a cost.

    Two things this does not do, deliberately:

    * It does not decide whether a fill was passive. Nothing in an OHLCV
      bar knows that. The caller asserts `is_maker`, and asserting it for
      every fill is a claim about queue behaviour this library cannot
      check -- see the note in Documentation/04_backtesting.md.
    * It does not bound the rebate. A venue schedule is a fact, not a
      parameter to tune, and clamping it would hide a mis-entered one.

    Args:
        notional:    Trade notional; the magnitude is used.
        taker_rate:  Fraction charged when removing liquidity. Must be >= 0.
        maker_rate:  Fraction charged when adding liquidity. NEGATIVE means
                     a rebate -- the venue pays you.
        is_maker:    Which side of the schedule this fill lands on.
    """
    _cost_rate("notional", notional, allow_negative=True)
    _cost_rate("taker_rate", taker_rate)
    _cost_rate("maker_rate", maker_rate, allow_negative=True)
    rate = maker_rate if is_maker else taker_rate
    if is_maker and maker_rate < 0:
        logger.debug(
            "[maker_taker_cost] maker rebate applied: %.6f of notional. This "
            "is a credit -- verify it against the venue's published schedule.",
            -maker_rate,
        )
    return abs(notional) * rate
