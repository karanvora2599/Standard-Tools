"""
What a set of holdings costs to finance over a horizon: the account's
financing terms, `backtest.financing.FinancingSpec`, as a calculator.

THE ACCOUNT. `equity` is the account's own capital. Every line is a funded
holding -- bought or sold short for cash at its base notional -- so the
account's cash is equity minus the net of the lines (long minus short).
Under `short_proceeds="net_cash"` a short's proceeds are in that cash;
under `"segregated"` the lender holds them, they earn the rebate, and the
cash that accrues interest is equity minus the longs. Positive cash earns
the benchmark less the credit spread, negative cash pays the benchmark plus
the debit spread, every short pays its borrow on its market value, and
segregated proceeds earn the benchmark less the rebate spread. A future or
a swap is not funded this way -- its financing is in its price -- so it is
left out of the lines.

SIMPLE INTEREST ON TODAY'S BALANCES. The balances are held at today's
prices and FX rates for the whole horizon, and interest accrues on the
balance, not on interest accrued within it: an amount is balance x
sum(rate x days) / basis over the periods of the horizon. So $10m long on
$5m of equity at 5.30% + 50 bp for 30 days ACT/360 pays 5m x 5.80% x 30 /
360 = 24,166.67.

THE HORIZON is a number of calendar days, under a scalar benchmark, or a
start and an end date, which a dated benchmark needs. A rate is in force
from its date to the next one the benchmark gives -- the spec's rule that
a balance accrues at the rate of the day it was carried from, so Friday's
fixing covers the weekend. Under `benchmark_fill="refuse"` the start date
needs a fixing of its own; under `"previous"` an earlier one carries
forward. A gap between fixings longer than a long weekend is reported.

ONE CURRENCY OF CASH. Lines in other currencies are valued in the base
currency at the caller's rates, and all the cash is the base currency's,
financed over the one benchmark. An account that holds cash in several
currencies at their own rates is several calls.

LEVERAGE IS ONE EXPRESSION. With no spreads, a book k times its equity long
in an asset returning R over the horizon returns k R - (k - 1) f on its
equity, f = benchmark x days / basis, for every k: the debit on k > 1 and
the credit on k < 1 are the same balance x rate.

REFUSED, by argument and value: a short with no borrow rate -- its own
`borrow_fee_bps`, the spec's `borrow_bps[instrument]` or
`default_borrow_bps` -- as the spec refuses it; a short at a negative
price, which has no market value to charge a borrow on; equity that is
not finite and positive; a horizon given twice, or not at all, or in days
against a dated benchmark; an end date that is not after the start.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.financing import (
    FinancingSpec,
    as_financing_spec,
    calendar_day_gaps,
)
from standard_quant_tools.delta_one.daycount import to_date
from standard_quant_tools.error import ValidationError

from .currency import currency_code
from .positions import (
    PositionValue,
    _attribute,
    check_borrow_fee_bps,
    position_values,
)

__all__ = [
    "MAX_EQUITY",
    "MAX_HORIZON_DAYS",
    "FinancingCost",
    "ShortBorrow",
    "financing_cost",
]

#: The largest equity this accepts, as the positions bound their notionals.
MAX_EQUITY = 1e18
#: A hundred years of calendar days.
MAX_HORIZON_DAYS = 36_600
#: The longest run of days without a fixing that is not reported: Friday
#: to Tuesday across a holiday Monday.
_QUIET_GAP_DAYS = 4

#: The day a horizon given in days is counted from. A scalar benchmark is
#: the same rate on every day, so the date is a label the result omits.
_ORIGIN = pd.Timestamp("2000-01-03")


@dataclass(frozen=True)
class ShortBorrow:
    """One short line's borrow over the horizon."""

    instrument: str
    currency: str
    #: The short's market value in the base currency, a positive amount.
    market_value: float
    #: Annual borrow fee in basis points.
    borrow_bps: float
    #: Where the fee came from: the line's own `borrow_fee_bps`, the spec's
    #: `borrow_bps` for the instrument, or its `default_borrow_bps`.
    source: str
    cost: float


@dataclass(frozen=True)
class FinancingCost:
    """The financing of a set of holdings over a horizon, in the base
    currency. Costs are positive; `credit_interest` and `rebate` are income
    (negative when the rate they accrue at is). `net_cost` is debit
    interest plus borrow less credit interest and rebate."""

    base_currency: str
    day_count: str
    basis: float
    #: ISO dates of the horizon; None when it was given in days.
    start_date: Optional[str]
    end_date: Optional[str]
    days: int
    #: days / basis.
    year_fraction: float
    #: `FinancingSpec.benchmark_report` over the horizon.
    benchmark: Dict[str, Any]
    equity: float
    long: float
    short: float
    net: float
    gross: float
    #: equity - net: the account's cash, short proceeds included.
    cash: float
    #: The cash that accrues interest: `cash`, less the segregated proceeds.
    interest_bearing_cash: float
    #: The short proceeds the lender holds; 0 under net_cash.
    segregated_proceeds: float
    #: The mean annual rate each side accrued at, weighted by days.
    credit_rate: float
    debit_rate: float
    rebate_rate: float
    debit_interest: float
    credit_interest: float
    borrow_cost: float
    rebate: float
    net_cost: float
    #: What the idle cash and the segregated proceeds earn below the
    #: benchmark: positive cash x credit spread, plus proceeds x rebate
    #: spread, over the horizon. The interest a cash balance forgoes by
    #: sitting with the broker rather than at the benchmark.
    cash_drag: float
    #: net_cost / gross / year_fraction: the annual rate the holdings'
    #: gross exposure is financed at. None when the gross is zero.
    effective_rate_on_gross: Optional[float]
    #: net_cost / equity / year_fraction.
    effective_rate_on_equity: float
    borrows: Tuple[ShortBorrow, ...]
    #: Instruments the spec gives a borrow rate that no short line holds.
    unused_borrow_bps: Tuple[str, ...]
    notes: Tuple[str, ...]


def _equity(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"equity={value!r} is not a number. Give a number.")
    equity = float(value)
    if not math.isfinite(equity) or equity <= 0 or equity > MAX_EQUITY:
        raise ValidationError(
            f"equity={value!r} must be a finite number above 0 and at most "
            f"{MAX_EQUITY:g}: it is the account's own capital in the base "
            "currency, the part of the holdings the broker does not fund. "
            "An account with no capital of its own is a deficit, not a "
            "financing question."
        )
    return equity


def _horizon_index(
    spec: FinancingSpec,
    days: Optional[int],
    start: Any,
    end: Any,
) -> Tuple[pd.DatetimeIndex, Optional[str], Optional[str]]:
    """The accrual dates of the horizon: its start, every fixing strictly
    inside it, and its end. Each period accrues at the rate in force on its
    first date."""
    dated_horizon = start is not None or end is not None
    if days is not None and dated_horizon:
        raise ValidationError(
            f"the horizon is given twice: days={days!r} and start/end dates "
            f"({start!r}, {end!r}). Give the number of days, or the two dates."
        )
    if days is None and not dated_horizon:
        raise ValidationError(
            "no horizon was given. Give days (calendar days, under a scalar "
            "benchmark_rate) or start and end dates."
        )
    if days is not None:
        if isinstance(days, bool) or not isinstance(days, numbers.Integral):
            raise ValidationError(
                f"days={days!r} is not a whole number of calendar days."
            )
        if not 1 <= int(days) <= MAX_HORIZON_DAYS:
            raise ValidationError(
                f"days={days!r} must be between 1 and {MAX_HORIZON_DAYS:,} "
                "calendar days."
            )
        if spec.is_dated:
            raise ValidationError(
                f"days={days!r} with a dated benchmark_rate: a dated rate is "
                "read on dates, so give start and end dates instead of a "
                "number of days."
            )
        index = pd.DatetimeIndex([_ORIGIN, _ORIGIN + pd.Timedelta(days=int(days))])
        return index, None, None
    if start is None or end is None:
        raise ValidationError(
            f"start={start!r} and end={end!r}: a dated horizon needs both "
            "dates ('YYYY-MM-DD')."
        )
    first = pd.Timestamp(to_date(start, "start"))
    last = pd.Timestamp(to_date(end, "end"))
    if last <= first:
        raise ValidationError(
            f"end={str(last.date())!r} is not after start="
            f"{str(first.date())!r}. The horizon runs from start to end and "
            "needs at least one day."
        )
    if (last - first).days > MAX_HORIZON_DAYS:
        raise ValidationError(
            f"start {first.date()} to end {last.date()} is more than "
            f"{MAX_HORIZON_DAYS:,} calendar days."
        )
    fixings = spec.benchmark_dates()
    inside = fixings[(fixings > first) & (fixings < last)]
    index = pd.DatetimeIndex([first, *inside, last])
    return index, str(first.date()), str(last.date())


def _borrow_rate(
    position: Any, index: int, instrument: str, spec: FinancingSpec
) -> Tuple[Optional[float], str]:
    own = check_borrow_fee_bps(
        _attribute(position, "borrow_fee_bps", index, None),
        f"positions[{index}].borrow_fee_bps",
    )
    if own is not None:
        return own, "position"
    if instrument in spec.borrow_bps:
        return float(spec.borrow_bps[instrument]), "borrow_bps"
    if spec.default_borrow_bps is not None:
        return float(spec.default_borrow_bps), "default_borrow_bps"
    return None, ""


def _refuse_unborrowed(missing: List[Tuple[int, str]]) -> None:
    named = ", ".join(f"positions[{i}] {name!r}" for i, name in missing[:5])
    more = len(missing) - 5
    first = missing[0][1]
    raise ValidationError(
        f"financing has no borrow rate for {named}"
        + (f" and {more} more" if more > 0 else "")
        + ", held short. A short is borrowed stock, and with a financing spec "
        "in use nothing borrows for free by default: give the position its "
        f"own borrow_fee_bps, set borrow_bps[{first!r}] (annual basis points), "
        "or set default_borrow_bps, which every unlisted instrument pays."
    )


def _gap_notes(index: pd.DatetimeIndex, gaps: np.ndarray) -> List[str]:
    """A line for each run of more than _QUIET_GAP_DAYS without a fixing,
    so a missing fortnight of rates does not pass for a long weekend."""
    notes = []
    for i in range(1, len(index) - 1):
        if gaps[i] > _QUIET_GAP_DAYS:
            notes.append(
                f"benchmark_rate gives no fixing from {index[i - 1].date()} to "
                f"{index[i].date()} ({int(gaps[i])} calendar days): the "
                f"{index[i - 1].date()} rate is in force for all of them."
            )
    if gaps[-1] > _QUIET_GAP_DAYS:
        notes.append(
            f"the rate in force on {index[-2].date()} is the last the horizon "
            f"reads, {int(gaps[-1])} calendar days before its end on "
            f"{index[-1].date()}: it is in force to the end."
        )
    return notes[:5]


def financing_cost(
    positions: Sequence[Any],
    *,
    equity: float,
    financing: Any,
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]] = None,
    days: Optional[int] = None,
    start: Any = None,
    end: Any = None,
) -> FinancingCost:
    """
    The debit and credit interest, short borrow, rebate, net cost, cash drag
    and effective rates of `positions` financed under `financing` over a
    horizon, in `base_currency`. See the module docstring for the account
    and the arithmetic.

    `positions` are what `position_values` reads (each may carry its own
    `borrow_fee_bps`, which overrides the spec's); `financing` is a
    `FinancingSpec`, its input model or a mapping of its fields; the
    horizon is `days` (scalar benchmark) or `start` and `end`.
    """
    spec = as_financing_spec(financing)
    if spec is None:
        raise ValidationError(
            "financing is required: the benchmark rate, its day count and the "
            "spreads are what the holdings are financed at."
        )
    positions = list(positions)
    values: List[PositionValue] = position_values(positions, base_currency, fx_rates)
    base = currency_code(base_currency, argument="base_currency")
    capital = _equity(equity)
    index, start_date, end_date = _horizon_index(spec, days, start, end)

    borrows_in: List[Tuple[PositionValue, float, str]] = []
    missing: List[Tuple[int, str]] = []
    for i, (position, value) in enumerate(zip(positions, values)):
        if value.side != "short":
            continue
        if value.price < 0:
            raise ValidationError(
                f"positions[{i}] {value.instrument!r} is short at a negative "
                f"price ({value.price!r}). A borrow fee is charged on the "
                "market value of the stock borrowed, and a negative price has "
                "none. Leave the line out, or give the borrow's value as a "
                "positive price."
            )
        rate, source = _borrow_rate(position, i, value.instrument, spec)
        if rate is None:
            missing.append((i, value.instrument))
        else:
            borrows_in.append((value, rate, source))
    if missing:
        _refuse_unborrowed(missing)

    # The spec's own schedule: the credit, debit and rebate rate each period
    # accrues at, the rate in force on the period's first date.
    schedule = spec.schedule(index, assets=())
    gaps = calendar_day_gaps(index)
    total_days = int(gaps.sum())
    basis = spec.basis
    tau = total_days / basis

    def rate_days(rates: np.ndarray) -> float:
        """sum(rate x days) over the periods, each at the rate in force on
        its first date."""
        return math.fsum(float(r) * float(g) for r, g in zip(rates[1:], gaps[1:]))

    credit_rd = rate_days(schedule.credit)
    debit_rd = rate_days(schedule.debit)
    rebate_rd = rate_days(schedule.rebate) if spec.segregated else 0.0
    bench_rd = rate_days(schedule.benchmark)
    # What one unit of balance accrues over the horizon on each side.
    credit_f, debit_f = credit_rd / basis, debit_rd / basis
    rebate_f, bench_f = rebate_rd / basis, bench_rd / basis

    long = math.fsum(v.base_notional for v in values if v.side == "long")
    short = 0.0 - math.fsum(v.base_notional for v in values if v.side == "short")
    net = math.fsum(v.base_notional for v in values)
    gross = math.fsum(abs(v.base_notional) for v in values)
    cash = capital - net
    proceeds = short if spec.segregated else 0.0
    bearing = cash - proceeds

    debit = (0.0 - bearing) * debit_f if bearing < 0 else 0.0
    credit = bearing * credit_f if bearing > 0 else 0.0
    rebate = proceeds * rebate_f
    borrows = tuple(
        ShortBorrow(
            instrument=value.instrument,
            currency=value.currency,
            market_value=0.0 - value.base_notional,
            borrow_bps=rate,
            source=source,
            cost=(0.0 - value.base_notional) * (rate / 10_000.0) * tau,
        )
        for value, rate, source in borrows_in
    )
    borrow = math.fsum(b.cost for b in borrows)
    net_cost = math.fsum([debit, borrow, -credit, -rebate])
    drag = math.fsum(
        [max(bearing, 0.0) * (bench_f - credit_f), proceeds * (bench_f - rebate_f)]
        if spec.segregated
        else [max(bearing, 0.0) * (bench_f - credit_f)]
    )

    held_short = {v.instrument for v in values if v.side == "short"}
    unused = tuple(sorted(a for a in spec.borrow_bps if a not in held_short))
    notes: List[str] = []
    if spec.is_dated:
        notes.extend(_gap_notes(index, gaps))
    held = sorted({v.currency for v in values} - {base})
    if held:
        notes.append(
            f"lines in {', '.join(held)} are valued in {base} at fx_rates, and "
            "all the cash is financed in it over the one benchmark; cash held "
            "in another currency at its own rate is a separate call."
        )
    if unused:
        notes.append(
            f"borrow_bps names {', '.join(repr(u) for u in unused[:5])}, which "
            "no short line holds (instruments match exactly, case included)."
        )

    return FinancingCost(
        base_currency=base,
        day_count=spec.day_count,
        basis=basis,
        start_date=start_date,
        end_date=end_date,
        days=total_days,
        year_fraction=tau,
        benchmark=spec.benchmark_report(index),
        equity=capital,
        long=long,
        short=short,
        net=net,
        gross=gross,
        cash=cash,
        interest_bearing_cash=bearing,
        segregated_proceeds=proceeds,
        credit_rate=credit_rd / total_days,
        debit_rate=debit_rd / total_days,
        rebate_rate=rebate_rd / total_days,
        debit_interest=debit,
        credit_interest=credit,
        borrow_cost=borrow,
        rebate=rebate,
        net_cost=net_cost,
        cash_drag=drag,
        effective_rate_on_gross=net_cost / gross / tau if gross > 0 else None,
        effective_rate_on_equity=net_cost / capital / tau,
        borrows=borrows,
        unused_borrow_bps=unused,
        notes=tuple(notes),
    )
