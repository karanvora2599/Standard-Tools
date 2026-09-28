"""
Sizing one position: how many shares a risk budget buys behind an ATR stop,
and what half-Kelly says when the strategy's win statistics are known.

This is the arithmetic behind `get_position_size`, kept here so a direct
caller and the tool size a position -- and refuse one -- the same way. The
tool fetches the bars and computes the ATR; everything after that is this
function.

    fixed risk   shares = (account_equity x risk_per_trade_pct)
                          / (atr_multiplier x ATR)
    half Kelly   f = (b p - q) / b, with b = avg_win / avg_loss,
                 p = win_rate, q = 1 - p; shares = account_equity x f / 2
                 / last_close

WHAT IS REFUSED, AND WHY BY NAME. The inputs the tool's schema checks are
checked again here: equity and multiplier strictly positive, a risk fraction
in (0, 1], a win rate in [0, 1], a non-negative average win and a positive
average loss. Past those, three quantities can leave the float range from
finite inputs, and each used to escape as something that named no input: a
stop distance (an infinite stop sized every position at zero shares and
reported the worst-case loss as 0 x inf = NaN), a share count (`int()` of an
infinite quotient raised a bare OverflowError), and a position value (a
count near the float ceiling times the price came back as an infinite
position). See the CHANGELOG entry of 2026-09-28.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_scalar

__all__ = ["size_position"]

_WHO = "size_position"


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
) -> Dict[str, Any]:
    """
    Fixed-risk shares behind an ATR stop, and half-Kelly shares when all
    three win statistics are given.

    `recommended_sizing` is `half_kelly` only when Kelly has a positive
    edge and buys at least one share; otherwise it is `fixed_risk`. Values
    come back unrounded -- rounding is presentation, and the tool does it.
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

    kelly_fraction: Optional[float] = None
    shares_hk: Optional[int] = None
    value_hk: Optional[float] = None
    has_kelly_inputs = (
        win_rate is not None and avg_win_pct is not None and avg_loss_pct is not None
    )
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
        raw_kelly = (
            (payoff * win_rate - (1.0 - win_rate)) / payoff if payoff > 0 else 0.0
        )
        kelly_fraction = round(max(raw_kelly, 0.0), 4)
        half_kelly_equity = account_equity * kelly_fraction * 0.5
        shares_hk = (
            _whole_shares(half_kelly_equity, last_close, "the half-Kelly position")
            if last_close > 0
            else 0
        )
        value_hk = _position_value(shares_hk, last_close, "the half-Kelly position")

    use_kelly = bool(
        has_kelly_inputs
        and kelly_fraction
        and kelly_fraction > 0
        and (shares_hk or 0) > 0
    )
    recommended_shares = (shares_hk or 0) if use_kelly else shares_fr
    return {
        "stop_distance": stop_distance,
        "dollar_risk": dollar_risk,
        "shares_fixed_risk": shares_fr,
        "position_value_fixed_risk": value_fr,
        "portfolio_pct_fixed_risk": value_fr / account_equity,
        "max_loss_fixed_risk": shares_fr * stop_distance,
        "kelly_fraction": kelly_fraction,
        "shares_half_kelly": shares_hk,
        "position_value_half_kelly": value_hk,
        "portfolio_pct_half_kelly": (
            value_hk / account_equity if value_hk is not None else None
        ),
        "recommended_sizing": "half_kelly" if use_kelly else "fixed_risk",
        "recommended_shares": recommended_shares,
        "recommended_position_value": recommended_shares * last_close,
    }
