"""
Sizing one position: how many shares a risk budget buys behind an ATR stop,
and what half-Kelly says when the strategy's win statistics are known.

This is the arithmetic behind `get_position_size`, kept here so a direct
caller and the tool size a position -- and refuse one -- the same way. The
tool fetches the bars and computes the ATR; everything after that is this
function.

    fixed risk   shares = (account_equity x risk_per_trade_pct)
                          / (atr_multiplier x ATR)
    Kelly        a trade returns +W with probability p and -L with
                 probability q = 1 - p, where W = avg_win_pct and
                 L = avg_loss_pct are returns ON THE POSITION. Holding a
                 fraction f of equity in it grows equity by
                     g(f) = p ln(1 + f W) + q ln(1 - f L)
                 per trade, which is greatest at
                     f* = p / L - q / W          (kelly_notional_fraction)
    half Kelly   shares = account_equity x min(f* / 2, max_leverage)
                          / last_close

WHAT f* IS, AND WHAT IT USED TO BE. f* is a NOTIONAL: the position's value
as a multiple of equity. Until the CHANGELOG entry of 2026-10-22 this
module computed (b p - q) / b with b = W / L, which is f* x L -- the
fraction of equity LOST on an average losing trade at full Kelly -- and
then bought that fraction of equity as a position. At p = 0.55, W = 6%,
L = 3% that bought 16.2% of equity where half-Kelly is 542%: a third of a
hundredth of the size. The number is still reported, as
`kelly_risk_fraction`, under a name that says what it is.

WHY A CAP. f* for an ordinary edge is a multiple of equity -- 10.8 in the
example above -- because a trade that loses 3% of the position risks only
3% of it. Half of that is still leverage most accounts cannot carry, so
the Kelly position is capped at `max_leverage` times equity (1, no
leverage, by default), and a warning names the cap whenever it binds.

NO EDGE IS ZERO SHARES. When f* <= 0 the statistics say the trade loses
money in expectation (p W - q L <= 0), and Kelly's answer is to take none.
That is reported as zero shares and a warning. The fixed-risk figures are
still computed -- they say what the risk budget would buy behind the stop --
but they are not offered as the recommendation for a trade with no edge.

HOW SURE f* IS. With `n_trades`, the win rate's 95% Wilson interval is
mapped through f*, which rises with p; the interval holds W and L fixed,
so the true range is wider.

WHAT IS REFUSED, AND WHY BY NAME. The inputs the tool's schema checks are
checked again here: equity, multiplier and leverage cap strictly positive,
a risk fraction in (0, 1], a win rate in [0, 1], a non-negative average win
and a positive average loss, a trade count of at least one -- and Kelly's
three statistics together or not at all, because one of them alone used to
fall back to fixed-risk sizing without a word. Past those, three quantities
can leave the float range from finite inputs, and each used to escape as
something that named no input: a stop distance (an infinite stop sized
every position at zero shares and reported the worst-case loss as
0 x inf = NaN), a share count (`int()` of an infinite quotient raised a
bare OverflowError), and a position value (a count near the float ceiling
times the price came back as an infinite position). See the CHANGELOG
entry of 2026-09-28.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_scalar

__all__ = ["kelly_log_growth", "kelly_notional_fraction", "size_position"]

_WHO = "size_position"

#: Two-sided 95%: the win rate's Wilson interval is drawn at this z.
_Z95 = NormalDist().inv_cdf(0.975)


def _positive(value: Any, name: str) -> float:
    number = require_finite_scalar(value, name, _WHO)
    if number <= 0:
        raise ValidationError(f"{_WHO}: {name} must be positive, got {number!r}")
    return number


def _whole_shares(amount: float, per_share: float, what: str) -> int:
    """
    Whole shares that `amount` buys at `per_share`, never negative.

    `int()` of a non-finite quotient is an OverflowError or a ValueError
    that names neither the position nor the input behind it, so a quotient
    beyond the float range is refused here in the sizer's own terms.
    """
    count = amount / per_share
    if not math.isfinite(count):
        raise ValidationError(
            f"{what} is not a finite number of shares ({amount:g} / "
            f"{per_share:g}). account_equity is in dollars and "
            "risk_per_trade_pct a fraction of it; check both are on that "
            "scale."
        )
    return max(int(count), 0)


def _position_value(shares: int, last_close: float, what: str) -> float:
    """What `shares` are worth at `last_close`, refused past the float
    range: a count near the ceiling times a price is an infinite position,
    not a recommendation."""
    value = shares * last_close
    if not math.isfinite(value):
        raise ValidationError(
            f"{what} is worth more than the float range ({shares:.3g} shares "
            f"at {last_close:g}). account_equity is in dollars; check it is "
            "on that scale."
        )
    return value


def kelly_notional_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    The growth-optimal position as a multiple of equity, f* = p/L - q/W.

    It solves g'(f) = 0 for g(f) = p ln(1 + f W) + q ln(1 - f L). Three
    inputs have no interior optimum and return what the optimum is instead:
    a win rate of 1 with a positive win (no trade loses, so growth rises
    without bound: +inf), an average win of 0 with some losses (no trade
    gains: -inf), and both (every trade returns zero: NaN).
    """
    loss_probability = 1.0 - win_rate
    if loss_probability == 0.0:
        return math.inf if avg_win > 0 else math.nan
    if avg_win == 0.0:
        return -math.inf
    return win_rate / avg_loss - loss_probability / avg_win


def kelly_log_growth(
    fraction: float, win_rate: float, avg_win: float, avg_loss: float
) -> float:
    """
    Expected log growth of equity per trade, p ln(1 + f W) + q ln(1 - f L),
    at a position of `fraction` times equity. -inf when a losing trade
    would take the whole account (f L >= 1) and losses can happen.
    """
    loss_probability = 1.0 - win_rate
    growth = win_rate * math.log1p(fraction * avg_win) if win_rate > 0 else 0.0
    if loss_probability > 0:
        if fraction * avg_loss >= 1.0:
            return -math.inf
        growth += loss_probability * math.log1p(-fraction * avg_loss)
    return growth


def _wilson_interval(successes_share: float, n: int) -> Tuple[float, float]:
    """The 95% Wilson score interval for a proportion seen over `n` trials."""
    z2 = _Z95 * _Z95
    denominator = 1.0 + z2 / n
    centre = (successes_share + z2 / (2.0 * n)) / denominator
    half_width = (
        _Z95
        * math.sqrt(successes_share * (1.0 - successes_share) / n + z2 / (4.0 * n * n))
        / denominator
    )
    return max(centre - half_width, 0.0), min(centre + half_width, 1.0)


def _kelly(
    account_equity: float,
    last_close: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_leverage: float,
    n_trades: Optional[int],
) -> Dict[str, Any]:
    """Every Kelly figure, and the warnings that explain the ones that
    are not plain numbers."""
    warnings: List[str] = []
    f_star = kelly_notional_fraction(win_rate, avg_win, avg_loss)
    edge = win_rate * avg_win - (1.0 - win_rate) * avg_loss
    stats = (
        f"win_rate={win_rate:g}, avg_win_pct={avg_win:g} and "
        f"avg_loss_pct={avg_loss:g}"
    )

    if math.isnan(f_star):
        position_fraction = 0.0
        warnings.append(
            f"{stats}: every trade returns zero, so no position size changes "
            "growth. kelly_notional_fraction, kelly_risk_fraction, "
            "half_kelly_notional_fraction and the growth at full and half "
            "Kelly are null, and Kelly takes no position -- 0 shares."
        )
    elif f_star == math.inf:
        position_fraction = max_leverage
        warnings.append(
            f"{stats}: no trade in the sample lost, so growth rises without "
            "bound as the position grows -- kelly_notional_fraction, "
            "kelly_risk_fraction, half_kelly_notional_fraction and the "
            "growth at full and half Kelly are null. A sample with no losing "
            "trade is not evidence that a trade cannot lose. The position is "
            f"sized at the max_leverage cap, {max_leverage:g}x account_equity."
        )
    elif f_star <= 0:
        position_fraction = 0.0
        shown = (
            "unbounded below, because no trade gains (kelly_notional_fraction, "
            "kelly_risk_fraction and half_kelly_notional_fraction are null)"
            if f_star == -math.inf
            else f"{f_star:.4g}x account_equity"
        )
        warnings.append(
            f"{stats} give no edge: the expected return per trade is "
            f"p x W - q x L = {edge:+.4g}, the growth-optimal position is "
            f"{shown}, and Kelly takes none -- 0 shares. The fixed-risk "
            "figures say what the risk budget would buy behind the stop; they "
            "are not a recommendation to take a trade without an edge. The "
            "growth at full and half Kelly is null: there is no Kelly-sized "
            "position to measure it at."
        )
    else:
        position_fraction = min(f_star / 2.0, max_leverage)

    capped = f_star / 2.0 > max_leverage if f_star > 0 else False
    kelly_equity = account_equity * position_fraction
    shares = (
        _whole_shares(kelly_equity, last_close, "the half-Kelly position")
        if last_close > 0
        else 0
    )
    value = _position_value(shares, last_close, "the half-Kelly position")
    if position_fraction > 0 and shares == 0:
        warnings.append(
            f"The half-Kelly position, {position_fraction:.4g}x "
            f"account_equity = {kelly_equity:,.2f}, buys less than one share "
            f"at {last_close:g}, so Kelly sizes 0 shares."
        )
    if capped and math.isfinite(f_star):
        warnings.append(
            f"Half-Kelly is {f_star / 2.0:.4g}x account_equity "
            f"(kelly_notional_fraction {f_star:.4g} / 2); max_leverage="
            f"{max_leverage:g} caps the position at {max_leverage:g}x "
            f"account_equity, so it is sized at the cap -- {shares} shares. "
            "Raise max_leverage only for an account that can carry that much "
            "leverage, and only as far as the statistics behind f* deserve."
        )

    held_fraction = value / account_equity
    bounded = math.isfinite(f_star)
    finite_kelly = bounded and f_star > 0
    result: Dict[str, Any] = {
        # A non-finite f* makes every fraction built from it non-finite:
        # null, with the warning above saying why.
        "kelly_notional_fraction": f_star if bounded else None,
        "kelly_risk_fraction": f_star * avg_loss if bounded else None,
        "half_kelly_notional_fraction": f_star / 2.0 if bounded else None,
        "kelly_position_fraction": position_fraction,
        "kelly_capped_by_max_leverage": bool(capped),
        "shares_half_kelly": shares,
        "position_value_half_kelly": value,
        "portfolio_pct_half_kelly": held_fraction,
        "log_growth_per_trade_full_kelly": (
            kelly_log_growth(f_star, win_rate, avg_win, avg_loss)
            if finite_kelly
            else None
        ),
        "log_growth_per_trade_half_kelly": (
            kelly_log_growth(f_star / 2.0, win_rate, avg_win, avg_loss)
            if finite_kelly
            else None
        ),
        "log_growth_per_trade_recommended": kelly_log_growth(
            held_fraction, win_rate, avg_win, avg_loss
        ),
        "win_rate_interval_95": None,
        "kelly_notional_fraction_interval_95": None,
    }

    if n_trades is not None:
        low, high = _wilson_interval(win_rate, n_trades)
        f_low = kelly_notional_fraction(low, avg_win, avg_loss)
        f_high = kelly_notional_fraction(high, avg_win, avg_loss)
        result["win_rate_interval_95"] = [low, high]
        result["kelly_notional_fraction_interval_95"] = [
            f_low if math.isfinite(f_low) else None,
            f_high if math.isfinite(f_high) else None,
        ]
        for end, rate, f_end in (("lower", low, f_low), ("upper", high, f_high)):
            if not math.isfinite(f_end):
                warnings.append(
                    f"kelly_notional_fraction_interval_95's {end} end is null: "
                    f"at a win rate of {rate:.3f}, {_why_unbounded(f_end)}."
                )
        if f_star > 0 and not f_low > 0:
            warnings.append(
                f"Over n_trades={n_trades} trades the win rate's 95% interval "
                f"is [{low:.3f}, {high:.3f}], and at its low end Kelly takes no "
                "position: the edge itself is not established at 95%. The "
                "interval holds avg_win_pct and avg_loss_pct fixed, so the "
                "true range is wider still."
            )
    result["warnings"] = warnings
    return result


def _why_unbounded(value: float) -> str:
    """Why f* has no finite value, for a warning beside the null."""
    if value == math.inf:
        return "no trade loses, so the growth-optimal position is unbounded"
    if value == -math.inf:
        return "no trade gains (avg_win_pct is 0), so it is unbounded below"
    return "every trade returns zero, so no position size changes growth"


def size_position(
    account_equity: float,
    last_close: float,
    last_atr: float,
    *,
    risk_per_trade_pct: float = 0.01,
    atr_multiplier: float = 2.0,
    win_rate: Optional[float] = None,
    avg_win_pct: Optional[float] = None,
    avg_loss_pct: Optional[float] = None,
    max_leverage: float = 1.0,
    n_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fixed-risk shares behind an ATR stop, and half-Kelly shares when all
    three win statistics are given.

    `recommended_sizing` is `half_kelly` whenever the three statistics are
    given -- including when Kelly says zero shares, because the statistics
    say the trade has no edge -- and `fixed_risk` when they are not. Values
    come back unrounded -- rounding is presentation, and the tool does it.
    A Kelly figure with no finite value is None, and `warnings` says why.
    """
    account_equity = _positive(account_equity, "account_equity")
    risk_per_trade_pct = require_finite_scalar(
        risk_per_trade_pct, "risk_per_trade_pct", _WHO, maximum=1.0
    )
    if risk_per_trade_pct <= 0:
        raise ValidationError(
            f"{_WHO}: risk_per_trade_pct must be in (0, 1], got {risk_per_trade_pct}"
        )
    atr_multiplier = _positive(atr_multiplier, "atr_multiplier")
    max_leverage = _positive(max_leverage, "max_leverage")
    last_close = require_finite_scalar(last_close, "last_close", _WHO)
    last_atr = require_finite_scalar(last_atr, "last_atr", _WHO, minimum=0.0)
    # Each Kelly input is checked whenever it is given, as the tool's schema
    # checks it, not only when all three arrive and Kelly is computed.
    if win_rate is not None:
        win_rate = require_finite_scalar(
            win_rate, "win_rate", _WHO, minimum=0.0, maximum=1.0
        )
    if avg_win_pct is not None:
        avg_win_pct = require_finite_scalar(
            avg_win_pct, "avg_win_pct", _WHO, minimum=0.0
        )
    if avg_loss_pct is not None:
        avg_loss_pct = _positive(avg_loss_pct, "avg_loss_pct")
    if n_trades is not None:
        if isinstance(n_trades, bool) or int(n_trades) != n_trades or n_trades < 1:
            raise ValidationError(
                f"{_WHO}: n_trades must be a whole number of trades, at least "
                f"1, got {n_trades!r}"
            )
        n_trades = int(n_trades)

    kelly_inputs = {
        "win_rate": win_rate,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
    }
    given = [name for name, value in kelly_inputs.items() if value is not None]
    if given and len(given) < 3:
        missing = [name for name in kelly_inputs if name not in given]
        raise ValidationError(
            f"{_WHO}: {', '.join(given)} given without {', '.join(missing)}. "
            "Kelly needs all three -- the win rate and the average winning and "
            "losing trade returns -- so pass the missing ones, or none of the "
            "three for fixed-risk sizing alone."
        )
    has_kelly_inputs = len(given) == 3
    if n_trades is not None and not has_kelly_inputs:
        raise ValidationError(
            f"{_WHO}: n_trades={n_trades} was given without win_rate, "
            "avg_win_pct and avg_loss_pct. It sets the confidence interval on "
            "Kelly's fraction, so it needs the three statistics Kelly reads."
        )

    stop_distance = last_atr * atr_multiplier
    if not math.isfinite(stop_distance):
        raise ValidationError(
            f"atr_multiplier={atr_multiplier:g} times the last ATR "
            f"({last_atr:g}) is not a finite stop distance. A multiplier is "
            "a few ATRs -- 1 to 5 is the usual range."
        )
    dollar_risk = account_equity * risk_per_trade_pct
    shares_fr = (
        _whole_shares(dollar_risk, stop_distance, "the fixed-risk position")
        if stop_distance > 0
        else 0
    )
    value_fr = _position_value(shares_fr, last_close, "the fixed-risk position")

    kelly: Dict[str, Any] = {
        "kelly_notional_fraction": None,
        "kelly_risk_fraction": None,
        "half_kelly_notional_fraction": None,
        "kelly_position_fraction": None,
        "kelly_capped_by_max_leverage": None,
        "shares_half_kelly": None,
        "position_value_half_kelly": None,
        "portfolio_pct_half_kelly": None,
        "log_growth_per_trade_full_kelly": None,
        "log_growth_per_trade_half_kelly": None,
        "log_growth_per_trade_recommended": None,
        "win_rate_interval_95": None,
        "kelly_notional_fraction_interval_95": None,
        "warnings": [],
    }
    if has_kelly_inputs:
        assert win_rate is not None and avg_win_pct is not None
        assert avg_loss_pct is not None
        payoff = avg_win_pct / avg_loss_pct
        if not math.isfinite(payoff):
            raise ValidationError(
                f"avg_win_pct={avg_win_pct:g} over avg_loss_pct="
                f"{avg_loss_pct:g} is beyond the float range, so Kelly has no "
                "finite payoff ratio. Both are decimal returns (0.05 = 5%)."
            )
        kelly = _kelly(
            account_equity,
            last_close,
            win_rate,
            avg_win_pct,
            avg_loss_pct,
            max_leverage,
            n_trades,
        )

    recommended_shares = kelly["shares_half_kelly"] if has_kelly_inputs else shares_fr
    warnings = list(kelly.pop("warnings"))
    return {
        "stop_distance": stop_distance,
        "dollar_risk": dollar_risk,
        "shares_fixed_risk": shares_fr,
        "position_value_fixed_risk": value_fr,
        "portfolio_pct_fixed_risk": value_fr / account_equity,
        "max_loss_fixed_risk": shares_fr * stop_distance,
        "max_leverage": max_leverage,
        **kelly,
        "recommended_sizing": "half_kelly" if has_kelly_inputs else "fixed_risk",
        "recommended_shares": recommended_shares,
        "recommended_position_value": recommended_shares * last_close,
        "warnings": warnings,
    }
