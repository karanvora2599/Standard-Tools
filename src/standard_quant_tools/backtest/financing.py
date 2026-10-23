"""
What cash earns and what borrowing costs, on one declared day count.

Every engine here subtracted a risk-free rate in its Sharpe while the cash
it held earned nothing and the leverage and shorts it ran were financed for
free. A book half in cash at a 4% rate read a Sharpe of 0.57 against 0.84
for the same asset fully invested, and a dollar-neutral book, whose cash is
its whole equity plus its short proceeds, went from 0.04 at a zero rate to
-0.70 at 5%. See the CHANGELOG entry of 2026-10-22.

The rule all three engines now share: positive cash accrues at a cash rate,
negative cash at a debit rate, shorts pay a borrow fee, and the reported
Sharpe and Sortino measure the excess over the rate cash ACTUALLY accrued
on each bar -- the same rate, the same day count, from the same first bar.
So a book that holds only cash has an excess return of exactly zero, and a
constant k-times position financed at f returns k*R - (k - 1)*f on every
bar.

THE FULL TERMS. The engines first took one scalar per term: a cash rate, a
margin rate, one borrow fee for every short. A broker states them as one
benchmark (a policy or overnight rate, which moves) with a spread it keeps
on credit balances and one it charges on debits, a borrow rate per name,
and a rebate on proceeds it holds. `FinancingSpec` is those terms. Each
engine resolves it -- or the scalars, which are the same thing with a
constant benchmark and no spreads -- into one `FinancingSchedule`: the
annual rate each bar's balance accrues at, per side, and the borrow rate
per asset. The Python loop and the native kernel read the same arrays.

A balance carried from one bar to the next accrues at the rate in force
on the bar it was carried FROM: an overnight rate published for a date is
the rate on money lent from that date to the next business day, so
Friday's rate covers the weekend. The first bar's rate is therefore the
first one used, and the last bar's is never needed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_quant_tools.backtest.costs import (
    DAY_COUNT_BASIS,
    DEFAULT_DAY_COUNT,
    _cost_rate,
    day_count_basis,
    require_annual_rate,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    has_no_dispersion,
    sharpe_ratio,
    sortino_ratio,
)
from standard_quant_tools.numeric_contract import require_finite_scalar
from standard_quant_tools.validation import validate_series

__all__ = [
    "SHORT_PROCEEDS",
    "BENCHMARK_FILLS",
    "FinancingSpec",
    "FinancingSpecInput",
    "FinancingSchedule",
    "as_financing_spec",
    "scalar_schedule",
    "refuse_scalars_with_spec",
    "require_financing_day_count",
    "calendar_day_gaps",
    "accrued_rate_per_bar",
    "ratios_over_accrued_rate",
    "ratios_over_benchmark",
    "resolve_debit_rate",
    "financing_summary",
    "negative_cash_note",
]

#: Where a short sale's proceeds sit. `net_cash` (the default, and what the
#: shared-cash engine always did) puts them in the account's cash, where
#: they earn the cash rate with everything else. `segregated` holds them as
#: collateral with the lender: they earn only the short rebate, and the
#: account's interest-bearing cash is its balance less the short book.
SHORT_PROCEEDS = ("net_cash", "segregated")

#: What a dated benchmark does on a bar date it has no rate for. `refuse`
#: (the default) names the first such date; `previous` carries the last
#: known rate forward, and still refuses a bar before the first rate.
BENCHMARK_FILLS = ("refuse", "previous")

# 30/360 in any of its spellings: 30/360, 30E/360, 30U/360, "30/360 US".
_THIRTY_360 = re.compile(r"^\s*30\s*[A-Z]*\s*/\s*360", re.IGNORECASE)


def require_financing_day_count(day_count: Any, name: str = "day_count") -> float:
    """
    The days in the year a financing accrual divides by, for ACT/360 or
    ACT/365F, and a refusal for anything else -- 30/360 with its reason.

    30/360 counts every month as 30 days. It is the convention of a coupon
    paid between scheduled dates; a cash balance, a margin loan and a stock
    borrow accrue on the calendar days they are outstanding, which is what
    every engine here hands its accruals.
    """
    if isinstance(day_count, str) and _THIRTY_360.match(day_count):
        raise ValidationError(
            f"{name}={day_count!r} is refused for financing. 30/360 counts "
            "every month as 30 days, the convention of a bond coupon paid "
            "between scheduled dates; a cash balance, a margin loan and a "
            "stock borrow accrue on the calendar days they are outstanding, "
            "so under 30/360 a balance held from the 30th to the 31st would "
            "accrue nothing and one held from 28 February to 1 March would "
            "accrue three days for one. Use 'ACT/360' (how USD margin, repo "
            "and money-market rates are quoted) or 'ACT/365F'."
        )
    try:
        return DAY_COUNT_BASIS[day_count]
    except (KeyError, TypeError):
        raise ValidationError(
            f"{name} must be one of {sorted(DAY_COUNT_BASIS)}, got "
            f"{day_count!r}. ACT/360 is how USD margin and money-market rates "
            "are usually quoted."
        ) from None


def _bar_dates(index: pd.Index, what: str = "the bars") -> pd.DatetimeIndex:
    """Each bar's calendar date, naive: a daily rate applies to the whole
    day, so an intraday or time-zone-aware bar is read on its local date."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(
            f"a dated benchmark is read by date, so {what} need a "
            f"DatetimeIndex; got {type(index).__name__}."
        )
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _benchmark_series(value: Any) -> pd.Series:
    """A dated benchmark as a float Series on naive, sorted, unique dates,
    with every rate a finite decimal at most 1 in magnitude."""
    if isinstance(value, pd.Series):
        series = value.copy()
    else:
        series = pd.Series(dict(value), dtype=object)
    if series.empty:
        raise ValidationError(
            "benchmark_rate is an empty dated series. Give a rate for each "
            "date, or one scalar rate for the whole run."
        )
    keys = list(series.index)
    try:
        if all(isinstance(k, str) for k in keys):
            stamps = pd.DatetimeIndex(pd.to_datetime(keys, format="ISO8601"))
        else:
            stamps = pd.DatetimeIndex(pd.to_datetime(keys))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "benchmark_rate's keys must be dates ('YYYY-MM-DD'); "
            f"{exc.__class__.__name__}: {exc}"
        ) from None
    if stamps.isna().any():
        bad = [k for k, gap in zip(series.index, stamps.isna()) if gap]
        raise ValidationError(f"benchmark_rate has keys that are not dates: {bad[:5]}")
    dates = _bar_dates(stamps, "benchmark_rate's keys")
    if dates.has_duplicates:
        repeated = sorted({str(d.date()) for d in dates[dates.duplicated()]})
        raise ValidationError(
            f"benchmark_rate names the same date more than once: {repeated[:5]}. "
            "One rate per date; keep the one you mean."
        )
    rates = []
    for key, raw in zip(series.index, series.to_numpy()):
        rates.append(require_annual_rate(raw, f"benchmark_rate[{str(key)!r}]"))
    return pd.Series(rates, index=dates, dtype=np.float64).sort_index()


def _spread_bps(value: Any, name: str) -> float:
    """A spread or borrow rate in basis points: finite and >= 0."""
    return _cost_rate(name, value)


@dataclass(frozen=True, eq=False)
class FinancingSpec:
    """
    How an account is financed, in the terms a broker states them.

    Args:
        benchmark_rate: the annual rate the account's terms are quoted over,
            as a decimal (0.053 is 5.30%; negative allowed, at most 1 in
            magnitude). A scalar, or a dated series -- a pd.Series on dates,
            or a mapping of ISO date to rate -- read on each bar's date.
        day_count: "ACT/360" or "ACT/365F". Required: a rate quoted ACT/360
            and accrued ACT/365F is 1.4% short of what is charged. 30/360 is
            refused with its reason (see require_financing_day_count).
        benchmark_fill: for a dated benchmark, what a bar date with no rate
            does. "refuse" (default) names the first such date; "previous"
            carries the last known rate forward and refuses a bar before the
            first rate.
        credit_spread_bps: positive cash earns benchmark - this.
        debit_spread_bps: negative cash pays benchmark + this. Neither side
            is floored: a benchmark below the credit spread charges a credit
            balance, as a negative policy rate does.
        borrow_bps: annual borrow fee per asset, in basis points.
        default_borrow_bps: the fee for an asset not in `borrow_bps`. A
            short in an asset with neither is refused by name: with the full
            terms in use, nothing borrows for free by default.
        short_proceeds: "net_cash" (the proceeds are cash and earn the
            credit rate) or "segregated" (the lender holds them; they earn
            benchmark - rebate_spread_bps, and interest accrues on cash less
            the short book).
        rebate_spread_bps: the lender's spread under the benchmark on
            segregated proceeds. Refused under "net_cash", where the
            proceeds earn the credit rate already.

    Every rate a bar's balance accrues at is rate * balance * (calendar days
    since the previous bar / basis), at the benchmark in force on the
    previous bar -- see the module docstring.
    """

    benchmark_rate: Union[float, Mapping[Any, float], pd.Series]
    day_count: str
    benchmark_fill: str = "refuse"
    credit_spread_bps: float = 0.0
    debit_spread_bps: float = 0.0
    borrow_bps: Mapping[str, float] = field(default_factory=dict)
    default_borrow_bps: Optional[float] = None
    short_proceeds: str = "net_cash"
    rebate_spread_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "_basis", require_financing_day_count(self.day_count))
        if self.benchmark_fill not in BENCHMARK_FILLS:
            raise ValidationError(
                f"benchmark_fill must be one of {BENCHMARK_FILLS}, got "
                f"{self.benchmark_fill!r}"
            )
        if isinstance(self.benchmark_rate, (pd.Series, Mapping)):
            object.__setattr__(self, "_series", _benchmark_series(self.benchmark_rate))
            object.__setattr__(self, "_rate", None)
        else:
            object.__setattr__(self, "_series", None)
            object.__setattr__(
                self,
                "_rate",
                require_annual_rate(self.benchmark_rate, "benchmark_rate"),
            )
        for name in ("credit_spread_bps", "debit_spread_bps", "rebate_spread_bps"):
            object.__setattr__(self, name, _spread_bps(getattr(self, name), name))
        if not isinstance(self.borrow_bps, Mapping):
            raise ValidationError(
                "borrow_bps must map each asset to its annual borrow fee in "
                f"basis points, got {type(self.borrow_bps).__name__}"
            )
        borrow: Dict[str, float] = {}
        for asset, bps in self.borrow_bps.items():
            if not isinstance(asset, str):
                raise ValidationError(
                    f"borrow_bps is keyed by asset name, got the key {asset!r}"
                )
            borrow[asset] = _spread_bps(bps, f"borrow_bps[{asset!r}]")
        object.__setattr__(self, "borrow_bps", borrow)
        if self.default_borrow_bps is not None:
            object.__setattr__(
                self,
                "default_borrow_bps",
                _spread_bps(self.default_borrow_bps, "default_borrow_bps"),
            )
        if self.short_proceeds not in SHORT_PROCEEDS:
            raise ValidationError(
                f"short_proceeds must be one of {SHORT_PROCEEDS}, got "
                f"{self.short_proceeds!r}"
            )
        if self.rebate_spread_bps != 0.0 and self.short_proceeds != "segregated":
            raise ValidationError(
                f"rebate_spread_bps={self.rebate_spread_bps:g} with short_proceeds="
                f"{self.short_proceeds!r}: net-cash proceeds are cash and earn "
                "the credit rate, so there is no rebate to take a spread from. "
                "Set short_proceeds='segregated' for proceeds the lender holds, "
                "or drop the spread."
            )

    # ── reading it ───────────────────────────────────────────────────────

    @property
    def basis(self) -> float:
        """The days in the year every accrual divides by."""
        return float(self._basis)  # type: ignore[attr-defined]

    @property
    def is_dated(self) -> bool:
        """Whether the benchmark is a dated series rather than one rate."""
        return self._series is not None  # type: ignore[attr-defined]

    @property
    def segregated(self) -> bool:
        return self.short_proceeds == "segregated"

    def borrow_rate_for(self, asset: Optional[str]) -> Optional[float]:
        """The asset's annual borrow fee in bps, its own or the default;
        None when it has neither."""
        if asset is not None and asset in self.borrow_bps:
            return float(self.borrow_bps[asset])
        return self.default_borrow_bps

    def benchmark_on(self, dates: pd.Index) -> np.ndarray:
        """The annual benchmark in force on each date, under the fill rule."""
        if not self.is_dated:
            return np.full(len(dates), self._rate, dtype=np.float64)  # type: ignore[attr-defined]
        bar_dates = _bar_dates(dates)
        series: pd.Series = self._series  # type: ignore[attr-defined]
        values = series.to_numpy(dtype=np.float64)
        if self.benchmark_fill == "refuse":
            pos = series.index.get_indexer(bar_dates)
            gaps = np.flatnonzero(pos < 0)
            if gaps.size:
                first, last = bar_dates[gaps[0]], bar_dates[gaps[-1]]
                raise ValidationError(
                    f"financing.benchmark_rate has no rate for the bar dated "
                    f"{first.date()} ({gaps.size} bar date(s) lack one, the "
                    f"last {last.date()}). Under benchmark_fill='refuse' "
                    "every bar a balance is carried from -- all but the last "
                    "-- needs its own rate. Supply the missing dates, or set "
                    "benchmark_fill='previous' to carry the last rate forward."
                )
            return values[pos]
        pos = series.index.searchsorted(bar_dates, side="right") - 1
        before = np.flatnonzero(pos < 0)
        if before.size:
            raise ValidationError(
                f"financing.benchmark_rate starts on {series.index[0].date()}, "
                f"after the bar dated {bar_dates[before[0]].date()}. "
                "benchmark_fill='previous' carries a known rate forward, and "
                "no rate is known before the first one: start the benchmark "
                f"on or before {bar_dates[0].date()}."
            )
        return values[pos]

    def applied_benchmark(self, index: pd.Index) -> np.ndarray:
        """
        The benchmark each bar's accrual is charged at: the rate in force on
        the bar before it. Element 0 is never accrued (nothing is carried
        into the opening bar) and holds the first rate.

        A scalar gives that rate on every bar, and a dated series that is
        constant gives the same array, so the two are the same bits.
        """
        n = len(index)
        if not self.is_dated:
            return np.full(n, self._rate, dtype=np.float64)  # type: ignore[attr-defined]
        if n < 2:
            return np.zeros(n, dtype=np.float64)
        carried = self.benchmark_on(index[:-1])
        out = np.empty(n, dtype=np.float64)
        out[1:] = carried
        out[0] = carried[0]
        return out

    def benchmark_report(self, index: pd.Index) -> Dict[str, Any]:
        """What the benchmark was over a run on `index`: the scalar, or the
        first, last and mean rate used, the mean weighted by the calendar
        days each rate was in force."""
        if not self.is_dated:
            rate = float(self._rate)  # type: ignore[attr-defined]
            return {
                "kind": "scalar",
                "rate": rate,
                "first": rate,
                "last": rate,
                "mean": rate,
                "first_date": None,
                "last_date": None,
                "fill": None,
                "n_bars_filled": None,
            }
        report: Dict[str, Any] = {
            "kind": "dated",
            "rate": None,
            "first": None,
            "last": None,
            "mean": None,
            "first_date": None,
            "last_date": None,
            "fill": self.benchmark_fill,
            "n_bars_filled": 0,
        }
        if len(index) < 2:
            return report
        carried = self.benchmark_on(index[:-1])
        dates = _bar_dates(index[:-1])
        gaps = calendar_day_gaps(index)[1:]
        total = float(gaps.sum())
        mean = (
            float((carried * gaps).sum() / total)
            if total > 0
            else float(carried.mean())
        )
        series: pd.Series = self._series  # type: ignore[attr-defined]
        report.update(
            {
                "first": float(carried[0]),
                "last": float(carried[-1]),
                "mean": mean,
                "first_date": str(dates[0].date()),
                "last_date": str(dates[-1].date()),
                "n_bars_filled": int((series.index.get_indexer(dates) < 0).sum()),
            }
        )
        return report

    def schedule(
        self,
        index: pd.Index,
        assets: Sequence[str],
        shorted: Sequence[str] = (),
    ) -> "FinancingSchedule":
        """
        The per-bar rates and per-asset borrow a run on `index` over
        `assets` accrues at. `shorted` are the assets the run holds short at
        some point; one with no borrow rate is refused by name.
        """
        missing = [a for a in shorted if self.borrow_rate_for(a) is None]
        if missing:
            raise ValidationError(
                f"financing has no borrow rate for {missing}, held short in "
                "this run. A short is borrowed stock, and with a financing "
                "spec in use nothing borrows for free by default: set "
                f"borrow_bps[{missing[0]!r}] (annual basis points) or "
                "default_borrow_bps, which every unlisted asset pays."
            )
        applied = self.applied_benchmark(index)
        n = len(index)
        borrow = np.array(
            [
                rate if (rate := self.borrow_rate_for(a)) is not None else 0.0
                for a in assets
            ],
            dtype=np.float64,
        )
        return FinancingSchedule(
            day_count=self.day_count,
            basis=self.basis,
            credit=applied - self.credit_spread_bps / 10_000.0,
            debit=applied + self.debit_spread_bps / 10_000.0,
            rebate=(
                applied - self.rebate_spread_bps / 10_000.0
                if self.segregated
                else np.zeros(n, dtype=np.float64)
            ),
            borrow_bps=borrow,
            segregated=self.segregated,
            benchmark=applied,
            spec=self,
        )


@dataclass(frozen=True, eq=False)
class FinancingSchedule:
    """
    The rates one run accrues at, resolved: per bar, the annual rate
    credited on positive free cash, charged on negative free cash and
    earned on segregated proceeds, each the rate that bar's accrual uses;
    per asset, the annual borrow fee in basis points. Built from a
    `FinancingSpec` or from the scalar shorthand (`scalar_schedule`), and
    read identically by the Python loop and the native kernel.
    """

    day_count: str
    basis: float
    credit: np.ndarray
    debit: np.ndarray
    rebate: np.ndarray
    borrow_bps: np.ndarray
    segregated: bool
    #: The benchmark each bar accrued at (a spec only); the per-bar rate a
    #: Sharpe measured over this run subtracts.
    benchmark: Optional[np.ndarray] = None
    spec: Optional[FinancingSpec] = None

    @property
    def any_borrow(self) -> bool:
        return bool((self.borrow_bps > 0.0).any())

    @property
    def active(self) -> bool:
        """Whether any balance accrues anything. With every rate at zero and
        proceeds in cash the financing block does not run, which is what
        keeps a zero-rate run bit for bit what it was."""
        return bool(
            self.any_borrow
            or (self.debit != 0.0).any()
            or (self.credit != 0.0).any()
            or self.segregated
            or (self.rebate != 0.0).any()
        )

    @property
    def uniform_borrow(self) -> Optional[float]:
        """The one borrow rate every asset pays, or None when they differ.
        One rate is charged on the short book's summed notional, the
        arithmetic the engines have always used; differing rates are charged
        asset by asset."""
        if self.borrow_bps.size and bool((self.borrow_bps == self.borrow_bps[0]).all()):
            return float(self.borrow_bps[0])
        return None

    def benchmark_accrual(self, index: pd.Index) -> Optional[pd.Series]:
        """What one unit of cash earned at the benchmark on each bar --
        rate * days / basis, 0.0 on bar 0 -- or None without a spec."""
        if self.benchmark is None:
            return None
        gaps = calendar_day_gaps(index)
        return pd.Series(
            self.benchmark * (gaps / self.basis), index=index, dtype=np.float64
        )


def scalar_schedule(
    n_bars: int,
    n_assets: int,
    *,
    cash_rate: float,
    margin_interest_rate: float,
    borrow_fee_bps: float,
    short_proceeds: str,
    short_rebate_rate: float,
    day_count: str,
) -> FinancingSchedule:
    """The scalar shorthand as a schedule: each rate on every bar, one
    borrow fee for every asset. The engines' arithmetic on it is the
    arithmetic they ran on the scalars, bit for bit."""
    return FinancingSchedule(
        day_count=day_count,
        basis=day_count_basis(day_count),
        credit=np.full(n_bars, float(cash_rate), dtype=np.float64),
        debit=np.full(n_bars, float(margin_interest_rate), dtype=np.float64),
        rebate=np.full(n_bars, float(short_rebate_rate), dtype=np.float64),
        borrow_bps=np.full(n_assets, float(borrow_fee_bps), dtype=np.float64),
        segregated=short_proceeds == "segregated",
    )


def refuse_scalars_with_spec(
    given: Mapping[str, Any], defaults: Mapping[str, Any], where: str
) -> None:
    """
    Refuse a financing spec passed together with any scalar financing term
    set away from its default. The scalars are a shorthand for a spec, so
    both at once would state the account's terms twice; picking one
    silently would discard the other.
    """
    set_away = [
        f"{name}={value!r}"
        for name, value in given.items()
        if value is not None and value != defaults.get(name)
    ]
    if set_away:
        raise ValidationError(
            f"{where}: financing (the full terms) was passed together with "
            f"{', '.join(set_away)}. The scalar rates are a shorthand for the "
            "same terms, so the two together would state the account's "
            "financing twice. Put everything in financing -- benchmark_rate "
            "(which is also the rate the Sharpe measures against), the credit "
            "and debit spreads, borrow_bps, short_proceeds and day_count -- "
            "or drop financing and keep the scalars."
        )


def as_financing_spec(value: Any, where: str = "financing") -> Optional[FinancingSpec]:
    """A FinancingSpec from what a caller passed: None, a FinancingSpec, its
    input model, or a plain mapping of the same fields."""
    if value is None or isinstance(value, FinancingSpec):
        return value
    if isinstance(value, FinancingSpecInput):
        return value.to_spec()
    if isinstance(value, Mapping):
        return FinancingSpecInput.model_validate(dict(value)).to_spec()
    raise ValidationError(
        f"{where} must be a FinancingSpec (or a mapping of its fields), got "
        f"{type(value).__name__}"
    )


def calendar_day_gaps(index: pd.Index) -> np.ndarray:
    """
    Calendar days from each bar to the one before it, as float64, with 0.0
    on the first bar: an account opened on bar 0 has carried nothing into
    it, so nothing accrues there.

    The same whole-day count the shared-cash engine's kernel has always been
    handed, so the two engines and the benchmark below count one Friday to
    Monday as three days alike.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidationError(
            "financing accrues on calendar days, so it needs a DatetimeIndex; "
            f"got {type(index).__name__}. Pass dated bars, or set every rate "
            "(cash, debit, borrow, rebate and the risk-free rate) to 0."
        )
    gaps = np.zeros(len(index), dtype=np.float64)
    if len(index) > 1:
        deltas = np.diff(index.to_numpy()).astype("timedelta64[D]")
        gaps[1:] = deltas.astype(np.float64)
    return gaps


def accrued_rate_per_bar(
    index: pd.Index, annual_rate: float, day_count: str = DEFAULT_DAY_COUNT
) -> pd.Series:
    """
    What one unit of cash earned on each bar at `annual_rate`:
    annual_rate * (calendar days since the previous bar / basis), and 0.0
    on the first bar.

    The expression is the engine's own credit, `cash * rate * (days /
    basis)`, divided through by the cash, so a book holding nothing but cash
    has an excess return over this series of zero to rounding.
    """
    basis = day_count_basis(day_count)
    gaps = calendar_day_gaps(index)
    return pd.Series(annual_rate * (gaps / basis), index=index, dtype=np.float64)


@validate_series()
def _ratios(
    returns: pd.Series, benchmark: np.ndarray, periods_per_year: int
) -> Tuple[float, float]:
    excess = returns - benchmark
    std = returns.std()
    if has_no_dispersion(returns, std):
        sharpe = float("nan")
    else:
        sharpe = float((excess.mean() / std) * np.sqrt(periods_per_year))
    downside_sq = np.minimum(excess.to_numpy(dtype=np.float64), 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean())) * np.sqrt(periods_per_year)
    mean_excess = float(excess.mean())
    if np.isnan(downside_dev):
        sortino = float("nan")
    elif downside_dev == 0:
        sortino = np.inf if mean_excess > 0 else float("nan")
    else:
        sortino = float((mean_excess * periods_per_year) / downside_dev)
    return sharpe, sortino


def ratios_over_accrued_rate(
    returns: pd.Series,
    risk_free_rate: float,
    day_count: str = DEFAULT_DAY_COUNT,
    periods_per_year: int = 252,
    benchmark: Optional[pd.Series] = None,
) -> Tuple[float, float]:
    """
    Sharpe and Sortino of `returns` over the risk-free rate as cash accrued
    it, bar by bar, on the declared day count.

    The formulas are `metrics.risk_metrics.sharpe_ratio` and
    `sortino_ratio` exactly -- mean excess over the standard deviation of the
    returns, and over the root-mean-square of the negative excesses -- with
    the per-bar benchmark in place of `rate / periods_per_year`. That swap
    is the point: a rate accrued over calendar days is what the engine
    credits, so a book that sits in cash scores an excess of zero rather
    than the gap between calendar-day and per-bar accrual, and the first
    bar, over which nothing accrued, is not charged a period of the rate.

    At a zero rate this IS `sharpe_ratio(returns, 0.0)` and
    `sortino_ratio(returns, 0.0)`, called as such, so every result reported
    before the rate entered the cash is reproduced bit for bit.

    `benchmark`, when given, is the per-bar rate itself (aligned to
    `returns`); otherwise it is built from the index of `returns`.
    """
    rate = require_finite_scalar(
        risk_free_rate, "risk_free_rate", "ratios_over_accrued_rate"
    )
    if rate == 0.0:
        return (
            sharpe_ratio(returns, 0.0, periods_per_year),
            sortino_ratio(returns, 0.0, periods_per_year),
        )
    if benchmark is None:
        benchmark = accrued_rate_per_bar(returns.index, rate, day_count)
    bench = benchmark.reindex(returns.index).to_numpy(dtype=np.float64)
    if not np.isfinite(bench).all():
        raise ValidationError(
            "the per-bar risk-free benchmark does not cover every return bar; "
            "it must be built on the same index as the returns."
        )
    return _ratios(returns, bench, periods_per_year)


def ratios_over_benchmark(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> Tuple[float, float]:
    """
    Sharpe and Sortino of `returns` over a per-bar benchmark accrual -- a
    financing spec's benchmark as the run accrued it (see
    `FinancingSchedule.benchmark_accrual`), dated or not.

    The same formulas as `ratios_over_accrued_rate`. A benchmark that is
    zero on every bar is `sharpe_ratio(returns, 0.0)` and
    `sortino_ratio(returns, 0.0)` themselves, so a spec at a zero benchmark
    reports what a run with no rate always reported.
    """
    bench = benchmark.reindex(returns.index).to_numpy(dtype=np.float64)
    if not np.isfinite(bench).all():
        raise ValidationError(
            "the per-bar benchmark does not cover every return bar; it must "
            "be built on the same index as the returns."
        )
    if not bench.any():
        return (
            sharpe_ratio(returns, 0.0, periods_per_year),
            sortino_ratio(returns, 0.0, periods_per_year),
        )
    return _ratios(returns, bench, periods_per_year)


def resolve_debit_rate(
    margin_interest_rate: Optional[float],
    risk_free_rate: float,
    margin_spread_bps: float = 0.0,
) -> float:
    """
    The annual rate charged on negative cash: the caller's own rate when
    given, otherwise the risk-free rate plus `margin_spread_bps`, floored at
    zero -- a broker does not pay an account to borrow.
    """
    if margin_interest_rate is not None:
        return float(margin_interest_rate)
    if not math.isfinite(margin_spread_bps) or margin_spread_bps < 0:
        raise ValidationError(
            f"margin_spread_bps must be finite and >= 0, got {margin_spread_bps!r}"
        )
    return max(float(risk_free_rate) + float(margin_spread_bps) / 10_000.0, 0.0)


def _optional_float(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def financing_summary(
    *,
    day_count: str,
    cash_rate: Optional[float],
    margin_interest_rate: Optional[float],
    borrow_fee_bps: Optional[float],
    short_proceeds: str,
    short_rebate_rate: Optional[float],
    totals: Dict[str, float],
    source: str = "scalars",
    benchmark: Optional[Dict[str, Any]] = None,
    credit_spread_bps: Optional[float] = None,
    debit_spread_bps: Optional[float] = None,
    rebate_spread_bps: Optional[float] = None,
    borrow_bps_by_asset: Optional[Dict[str, float]] = None,
    borrow_charged_by_asset: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    The rates a run was financed at, echoed beside what they came to.

    `source` is "scalars" (the shorthand) or "spec" (a FinancingSpec). Under
    a spec the four scalar rates are the constant rates it resolved to, or
    None where they moved with a dated benchmark (the credit, debit and
    rebate rates) or differed by asset (the borrow fee); `benchmark`, the
    spreads and the per-asset borrow then say what they were.
    """
    credited = float(totals.get("interest_credited", 0.0))
    charged = float(totals.get("interest_charged", 0.0))
    borrow = float(totals.get("borrow_charged", 0.0))
    rebate = float(totals.get("rebate_credited", 0.0))
    return {
        "day_count": day_count,
        "cash_rate": _optional_float(cash_rate),
        "margin_interest_rate": _optional_float(margin_interest_rate),
        "borrow_fee_bps": _optional_float(borrow_fee_bps),
        "short_proceeds": short_proceeds,
        "short_rebate_rate": _optional_float(short_rebate_rate),
        "interest_credited": credited,
        "interest_charged": charged,
        "borrow_charged": borrow,
        "rebate_credited": rebate,
        "net_financing": credited + rebate - charged - borrow,
        "debit_balance_years": float(totals.get("debit_balance_years", 0.0)),
        "source": source,
        "benchmark": benchmark,
        "credit_spread_bps": _optional_float(credit_spread_bps),
        "debit_spread_bps": _optional_float(debit_spread_bps),
        "rebate_spread_bps": _optional_float(rebate_spread_bps),
        "borrow_bps_by_asset": dict(borrow_bps_by_asset or {}),
        "borrow_charged_by_asset": dict(borrow_charged_by_asset or {}),
    }


def schedule_summary(
    schedule: FinancingSchedule,
    index: pd.Index,
    assets: Sequence[str],
    shorted: Sequence[str],
    totals: Dict[str, float],
    borrow_by_asset: np.ndarray,
    scalars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """`financing_summary` for one run's schedule: the scalars it was given,
    or the spec's terms and what each resolved to over `index`."""
    position = {a: i for i, a in enumerate(assets)}
    shorted_rates = {a: float(schedule.borrow_bps[position[a]]) for a in shorted}
    charged_by_asset = {a: float(borrow_by_asset[position[a]]) for a in shorted}
    spec = schedule.spec
    if spec is None:
        assert scalars is not None
        return financing_summary(
            day_count=schedule.day_count,
            totals=totals,
            borrow_bps_by_asset=shorted_rates,
            borrow_charged_by_asset=charged_by_asset,
            **scalars,
        )
    constant = not spec.is_dated
    benchmark = spec.benchmark_report(index)
    distinct = set(shorted_rates.values())
    if len(distinct) == 1:
        borrow_fee: Optional[float] = distinct.pop()
    elif not distinct:
        borrow_fee = spec.default_borrow_bps
    else:
        borrow_fee = None
    rate = benchmark["rate"]
    return financing_summary(
        day_count=schedule.day_count,
        cash_rate=rate - spec.credit_spread_bps / 10_000.0 if constant else None,
        margin_interest_rate=(
            rate + spec.debit_spread_bps / 10_000.0 if constant else None
        ),
        borrow_fee_bps=borrow_fee,
        short_proceeds=spec.short_proceeds,
        short_rebate_rate=(
            (rate - spec.rebate_spread_bps / 10_000.0 if constant else None)
            if spec.segregated
            else 0.0
        ),
        totals=totals,
        source="spec",
        benchmark=benchmark,
        credit_spread_bps=spec.credit_spread_bps,
        debit_spread_bps=spec.debit_spread_bps,
        rebate_spread_bps=spec.rebate_spread_bps,
        borrow_bps_by_asset=shorted_rates,
        borrow_charged_by_asset=charged_by_asset,
    )


def negative_cash_note(
    cash_curve: pd.Series,
    financing: Dict[str, Any],
) -> Optional[str]:
    """
    The warning for a book whose cash went negative, with the number in it.

    It used to read only "cash went negative at one or more bars -- implied
    margin borrowing", which said neither how much was borrowed nor what it
    was charged, so a 2x book financed at zero and one financed at 8% read
    the same. None when cash never went negative.
    """
    negative = cash_curve < 0
    n_negative = int(negative.sum())
    if not n_negative:
        return None
    lowest = float(cash_curve.min())
    day_count = financing["day_count"]
    charged = float(financing["interest_charged"])
    balance_years = float(financing["debit_balance_years"])
    head = (
        f"cash went negative on {n_negative} of {len(cash_curve)} bar(s), to a "
        f"low of {lowest:,.2f} — implied margin borrowing"
    )
    if financing.get("source") == "spec":
        bench = financing.get("benchmark") or {}
        terms = (
            f"the benchmark ({bench.get('kind')}) plus debit_spread_bps="
            f"{financing.get('debit_spread_bps'):g}"
        )
        return (
            f"{head}, charged at {terms} ({day_count}): {charged:,.2f} of "
            "interest over the run."
        )
    rate = float(financing["margin_interest_rate"])
    if rate > 0.0:
        return (
            f"{head}, charged at margin_interest_rate={rate:g} ({day_count}): "
            f"{charged:,.2f} of interest over the run."
        )
    per_point = balance_years * 0.01
    return (
        f"{head}, financed for nothing: margin_interest_rate=0. The debit "
        f"came to {balance_years:,.2f} currency-years ({day_count}), so each "
        f"1% of margin rate would have cost {per_point:,.2f}; set "
        "margin_interest_rate to what the broker charges, or read the return "
        "as an upper bound."
    )


# ── The input model ──────────────────────────────────────────────────────
#
# The same terms as a validated input, for the agent tools and for
# modeling's PortfolioSimSpec. It lives here, beside the dataclass, rather
# than in the agent package because PortfolioSimSpec is a library model the
# agent package itself imports: modeling cannot import the agent package
# while it loads. agent/runtimes/_financing.py is where the tools take it.

_MAX_SPREAD_BPS = 10_000.0
_MAX_BORROW_BPS = 100_000.0


def _finite_bounded(value: float, name: str, low: float, high: float) -> float:
    if not math.isfinite(value) or value < low or value > high:
        raise ValueError(
            f"{name}={value!r} must be a finite number in [{low:g}, {high:g}]"
        )
    return value


def _compact_schema(schema: Dict[str, Any]) -> None:
    """Drop the model description and the per-field titles -- each a
    restatement of a name already in the schema -- from the input model's
    JSON schema, which every tool that takes it carries in full."""
    schema.pop("description", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


class FinancingSpecInput(BaseModel):
    """The account's financing terms. See backtest.financing.FinancingSpec.

    The descriptions are terse on purpose: this model is inlined into the
    schema of every tool that takes it, so each word is paid once a tool.
    """

    # The docstring above is for readers of the code; the schema carries the
    # field descriptions and nothing a caller does not read.
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_schema)

    benchmark_rate: Union[float, Dict[str, float]] = Field(
        ...,
        description="Annual decimal (0.053 = 5.30%), or {ISO date: rate}; a "
        "balance accrues at its previous bar's rate. Also the Sharpe's rate.",
    )
    benchmark_fill: Literal["refuse", "previous"] = Field(
        "refuse",
        description="A bar date with no rate: 'refuse', or carry the "
        "'previous' rate.",
    )
    day_count: Literal["ACT/360", "ACT/365F"] = Field(
        ...,
        description="Required; accruals are rate x balance x days / basis.",
    )
    credit_spread_bps: float = Field(
        0.0,
        ge=0.0,
        le=_MAX_SPREAD_BPS,
        description="Credit earns benchmark - this.",
    )
    debit_spread_bps: float = Field(
        0.0,
        ge=0.0,
        le=_MAX_SPREAD_BPS,
        description="Debit pays benchmark + this.",
    )
    borrow_bps: Dict[str, float] = Field(
        default_factory=dict,
        description="{asset: annual borrow bps}.",
    )
    default_borrow_bps: Optional[float] = Field(
        None,
        ge=0.0,
        le=_MAX_BORROW_BPS,
        description="Borrow for unlisted assets; a short with no rate is refused.",
    )
    short_proceeds: Literal["net_cash", "segregated"] = Field(
        "net_cash",
        description="'segregated': held by the lender at benchmark - "
        "rebate_spread_bps.",
    )
    rebate_spread_bps: float = Field(
        0.0,
        ge=0.0,
        le=_MAX_SPREAD_BPS,
        description="Rebate is benchmark - this.",
    )

    @field_validator("day_count", mode="before")
    @classmethod
    def _day_count_with_reason(cls, value: Any) -> Any:
        if isinstance(value, str) and _THIRTY_360.match(value):
            # The library's own refusal, with the reason, rather than the
            # bare literal error a 30/360 would otherwise get.
            try:
                require_financing_day_count(value, "financing.day_count")
            except ValidationError as exc:
                raise ValueError(str(exc)) from None
        return value

    @field_validator("benchmark_rate")
    @classmethod
    def _rates_are_decimals(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if not value:
                raise ValueError(
                    "benchmark_rate is an empty dated series: give a rate per "
                    "date, or one number for the whole run."
                )
            for key, rate in value.items():
                _finite_bounded(float(rate), f"benchmark_rate[{key!r}]", -1.0, 1.0)
            return value
        return _finite_bounded(float(value), "benchmark_rate", -1.0, 1.0)

    @field_validator("borrow_bps")
    @classmethod
    def _borrow_is_bounded(cls, value: Dict[str, float]) -> Dict[str, float]:
        for asset, bps in value.items():
            _finite_bounded(float(bps), f"borrow_bps[{asset!r}]", 0.0, _MAX_BORROW_BPS)
        return value

    @model_validator(mode="after")
    def _rebate_needs_segregated_proceeds(self) -> "FinancingSpecInput":
        if self.rebate_spread_bps != 0.0 and self.short_proceeds != "segregated":
            raise ValueError(
                f"rebate_spread_bps={self.rebate_spread_bps:g} needs "
                "short_proceeds='segregated': net-cash proceeds are cash and "
                "earn the credit rate, so there is no rebate to take a spread from."
            )
        return self

    def to_spec(self, benchmark: Optional[Mapping[Any, float]] = None) -> FinancingSpec:
        """
        The library spec. `benchmark` is the dated benchmark with its keys
        already parsed (the agent tools parse them with parse_date_keys);
        without it the keys are read here as ISO dates.
        """
        rate: Any = self.benchmark_rate
        if isinstance(rate, dict):
            rate = pd.Series(dict(benchmark if benchmark is not None else rate))
        return FinancingSpec(
            benchmark_rate=rate,
            day_count=self.day_count,
            benchmark_fill=self.benchmark_fill,
            credit_spread_bps=self.credit_spread_bps,
            debit_spread_bps=self.debit_spread_bps,
            borrow_bps=dict(self.borrow_bps),
            default_borrow_bps=self.default_borrow_bps,
            short_proceeds=self.short_proceeds,
            rebate_spread_bps=self.rebate_spread_bps,
        )
