"""
The futures account reads its date-keyed inputs in bulk, with the same results.

`run_futures_simulation` and `run_futures_backtest` spent as long converting
their inputs as running the bar loop (the CHANGELOG entry of 2026-10-04):
the date index inferred from the keys one at a time, `pd.to_datetime`
boxing every stamp of an index that already was one, a Timestamp built per
contract-map key and hashed again on every bar, and on the tool's side a
`pd.isna` and a dict probe per key of every map and a Timestamp per bar of
the returned equity curve. Those steps now read arrays.

Nothing they produce may change. The code as it stood is kept below
(`_before`, `_before_series`, `_before_parse_date_keys`, and the tool's two
per-bar comprehensions), and every output of the two -- every curve with its
index, dtype and name, every record, total and warning, every refusal's type
and text, and every Python warning raised on the way -- must be EXACTLY
equal: floats to the bit, Timestamps to their unit, zone and fold. The inputs
cover the shapes the engine reads: daily, gapped and intraday bars; naive,
UTC, zoneinfo, pytz and fixed-offset zones, across both 2024 daylight-saving
changes; keys as Timestamps, ISO strings, dates, datetimes and datetime64,
mixed and unsorted; contract maps that cover every bar, some bars, other
dates, another zone or none; and the error paths.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import math
import warnings
from typing import Any, Dict, Iterator, List, Mapping, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes._shared import _named, parse_date_keys
from standard_quant_tools.agent.runtimes.backtest import futures_tools
from standard_quant_tools.backtest import futures_hedge_backtest
from standard_quant_tools.backtest.futures_engine import (
    _ISO_DAYS_FROM,
    _float_series,
    _iso_days,
    _match_days,
    _non_negative,
    _positive,
    _timestamp_index,
    run_futures_simulation,
)
from standard_quant_tools.error import ValidationError

try:
    import zoneinfo

    _NEW_YORK: Any = zoneinfo.ZoneInfo("America/New_York")
except Exception:  # no time zone database on this machine
    _NEW_YORK = None

try:
    import pytz

    _PYTZ_NEW_YORK: Any = pytz.timezone("US/Eastern")
except Exception:
    _PYTZ_NEW_YORK = None

_KOLKATA = dt.timezone(dt.timedelta(hours=5, minutes=30))


# ── the code as it stood ─────────────────────────────────────────────────


def _before_series(mapping: Mapping[Any, float], name: str) -> pd.Series:
    if not mapping:
        raise ValidationError(f"{name} is empty.")
    try:
        series = pd.Series(dict(mapping), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must map dates to numbers; {exc}") from None
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _before(
    *,
    prices: Mapping[Any, float],
    target_contracts: Mapping[Any, float],
    multiplier: float,
    initial_capital: float = 1_000_000.0,
    initial_margin: float = 0.0,
    maintenance_margin: Optional[float] = None,
    commission_per_contract: float = 0.0,
    slippage_points: float = 0.0,
    collateral_rate: float = 0.0,
    contract_map: Optional[Mapping[Any, str]] = None,
    allow_fractional: bool = False,
    roll_day_prior_prices: Optional[Mapping[Any, float]] = None,
) -> Dict[str, Any]:
    m = _positive(multiplier, "multiplier")
    capital = _positive(initial_capital, "initial_capital")
    im = _non_negative(initial_margin, "initial_margin")
    mm = (
        im
        if maintenance_margin is None
        else _non_negative(maintenance_margin, "maintenance_margin")
    )
    if mm > im:
        raise ValidationError(
            f"maintenance_margin ({mm}) exceeds initial_margin ({im}). A "
            "position would be in call the moment it opened."
        )
    commission = _non_negative(commission_per_contract, "commission_per_contract")
    slippage = _non_negative(slippage_points, "slippage_points")

    price_series = _before_series(prices, "prices")
    if (price_series <= 0).any():
        raise ValidationError("prices contains a non-positive value.")
    targets = _before_series(target_contracts, "target_contracts").reindex(
        price_series.index
    )
    targets = targets.ffill().fillna(0.0)

    rolls_by_date: Dict[Any, str] = {}
    if contract_map is not None:
        if not contract_map:
            raise ValidationError(
                "contract_map is empty. Pass None to model no roll at all, "
                "which says so in the warnings, rather than an empty map "
                "that looks like one and is not."
            )
        rolls_by_date = {
            pd.Timestamp(key): str(value) for key, value in contract_map.items()
        }

    dates = price_series.index
    n = len(dates)
    if n < 2:
        raise ValidationError(
            f"{n} price observation(s); a simulation needs at least two bars "
            "for variation margin to be defined."
        )

    prior_by_date: Dict[Any, float] = {}
    if roll_day_prior_prices:
        prior_by_date = {
            pd.Timestamp(key): float(value)
            for key, value in roll_day_prior_prices.items()
        }
    margin_limited: List[Dict[str, Any]] = []

    contracts = 0.0
    cash = capital
    margin_posted = 0.0
    warnings: List[str] = []

    equity_curve = np.empty(n)
    cash_curve = np.empty(n)
    margin_curve = np.empty(n)
    position_curve = np.empty(n)
    exposure_curve = np.empty(n)

    total_commission = 0.0
    total_slippage = 0.0
    total_interest = 0.0
    total_variation = 0.0
    margin_calls: List[Dict[str, Any]] = []
    rolls: List[Dict[str, Any]] = []

    price_values = price_series.to_numpy(dtype="float64")
    target_values = targets.to_numpy(dtype="float64")
    day_counts = [max(days, 0) for days in (dates[1:] - dates[:-1]).days.tolist()]

    previous_price = float(price_values[0])
    previous_contract = rolls_by_date.get(dates[0])

    for i, date in enumerate(dates):
        price = float(price_values[i])

        current_contract = rolls_by_date.get(date)
        rolled = (
            i > 0
            and contract_map is not None
            and current_contract is not None
            and previous_contract is not None
            and current_contract != previous_contract
            and contracts != 0.0
        )

        roll_day_booked = False
        if i > 0 and contracts != 0.0 and not rolled:
            variation = (price - previous_price) * contracts * m
            cash += variation
            total_variation += variation
        elif rolled and date in prior_by_date:
            variation = (prior_by_date[date] - previous_price) * contracts * m
            cash += variation
            total_variation += variation
            roll_day_booked = True

        if i > 0 and collateral_rate and cash > 0:
            days = day_counts[i - 1]
            interest = cash * collateral_rate * days / 365.0
            cash += interest
            total_interest += interest

        if rolled:
            legs = 2.0 * abs(contracts)
            cost = legs * commission + legs * slippage * m
            cash -= cost
            total_commission += legs * commission
            total_slippage += legs * slippage * m
            rolls.append(
                {
                    "date": str(date),
                    "from": previous_contract,
                    "to": current_contract,
                    "contracts": float(contracts),
                    "cost": float(cost),
                    "spread_points": float(
                        price
                        - (prior_by_date[date] if roll_day_booked else previous_price)
                    ),
                    "variation_margin_skipped": not roll_day_booked,
                }
            )

        target = float(target_values[i])
        if not allow_fractional:
            target = float(round(target))
        if im > 0 and target != 0.0:
            same_side = contracts * target > 0
            base = abs(contracts) if same_side else 0.0
            equity_before = cash + margin_posted
            headroom = (
                math.floor((equity_before - base * im) / im)
                if equity_before > base * im
                else 0
            )
            affordable = base + max(headroom, 0)
            if abs(target) > affordable:
                limited = math.copysign(max(affordable, 0.0), target)
                if not allow_fractional:
                    limited = float(round(limited))
                margin_limited.append(
                    {
                        "date": str(date),
                        "requested": float(target),
                        "filled": float(limited),
                        "equity": float(cash + margin_posted),
                    }
                )
                target = limited
        delta = target - contracts
        if abs(delta) > 1e-9:
            cost = abs(delta) * commission + abs(delta) * slippage * m
            cash -= cost
            total_commission += abs(delta) * commission
            total_slippage += abs(delta) * slippage * m
            contracts = target
            required = abs(contracts) * im
            cash -= required - margin_posted
            margin_posted = required

        equity = cash + margin_posted
        exposure = abs(contracts) * price * m

        required_maintenance = abs(contracts) * mm
        if mm > 0 and contracts != 0.0 and equity < required_maintenance:
            affordable = math.floor(equity / mm) if mm > 0 else 0.0
            affordable = max(affordable, 0.0)
            reduced_to = math.copysign(min(abs(contracts), affordable), contracts)
            closed = contracts - reduced_to
            if abs(closed) > 1e-9:
                cost = abs(closed) * commission + abs(closed) * slippage * m
                cash -= cost
                total_commission += abs(closed) * commission
                total_slippage += abs(closed) * slippage * m
                contracts = reduced_to
                required = abs(contracts) * im
                cash -= required - margin_posted
                margin_posted = required
                equity = cash + margin_posted
                exposure = abs(contracts) * price * m
                margin_calls.append(
                    {
                        "date": str(date),
                        "equity": float(equity),
                        "required": float(required_maintenance),
                        "contracts_closed": float(closed),
                    }
                )

        if equity <= 0:
            warnings.append(
                f"The account went to zero equity on {date}. Everything from "
                "that bar on is a simulation of an account that no longer "
                "exists; the curves are truncated there."
            )
            equity_curve[i:] = equity
            cash_curve[i:] = cash
            margin_curve[i:] = margin_posted
            position_curve[i:] = 0.0
            exposure_curve[i:] = 0.0
            break

        equity_curve[i] = equity
        cash_curve[i] = cash
        margin_curve[i] = margin_posted
        position_curve[i] = contracts
        exposure_curve[i] = exposure
        previous_price = price
        if current_contract is not None:
            previous_contract = current_contract

    equity = pd.Series(equity_curve, index=dates, name="equity")
    exposure = pd.Series(exposure_curve, index=dates, name="exposure")
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = (exposure / equity).replace([np.inf, -np.inf], np.nan)

    if im == 0.0:
        warnings.append(
            "initial_margin is zero, so this models an account that posts "
            "nothing to hold a position. That is a useful idealization and "
            "it is not a futures account -- leverage below is unbounded by "
            "construction and no margin call can ever fire."
        )
    if margin_calls:
        warnings.append(
            f"{len(margin_calls)} margin call(s) forced the position down. A "
            "backtest that financed those calls instead would be testing a "
            "strategy nobody could have run."
        )
    if margin_limited:
        warnings.append(
            f"{len(margin_limited)} fill(s) were sized down to what the account "
            "could post initial margin for; the requested and filled sizes "
            "are in `margin_limited_fills`. The target asked for more than "
            "the account could carry on those bars."
        )
    if rolls and any(r["variation_margin_skipped"] for r in rolls):
        skipped = sum(1 for r in rolls if r["variation_margin_skipped"])
        warnings.append(
            f"On {skipped} roll day(s) the variation margin was not booked: a "
            "single price series holds the NEW contract's close on a roll "
            "day and the old contract's move is unknown. Pass "
            "roll_day_prior_prices (the old contract's close on each roll "
            "day) to book it; on a live ES year the omission was $7,025 per "
            "contract."
        )
    if contract_map is None:
        warnings.append(
            "No contract_map, so NO ROLL was modelled and these prices are "
            "assumed to be one contract throughout. Over any horizon longer "
            "than a single expiry that omits the largest recurring cost of "
            "holding a future."
        )
    warnings.append(
        "Equity is cash plus posted margin. The contracts contribute no "
        "market value, because their profit has already been credited to "
        "cash as variation margin -- counting both would double it. "
        "`leverage` is therefore ECONOMIC EXPOSURE over equity, not the "
        "gross-market-value ratio the cash engine reports, and the two are "
        "not comparable."
    )

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return {
        "equity_curve": equity,
        "cash_curve": pd.Series(cash_curve, index=dates, name="cash"),
        "margin_curve": pd.Series(margin_curve, index=dates, name="margin"),
        "position_curve": pd.Series(position_curve, index=dates, name="contracts"),
        "exposure_curve": exposure,
        "leverage_curve": leverage,
        "initial_capital": capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / capital - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "max_leverage": float(np.nanmax(leverage.to_numpy())),
        "peak_exposure": float(exposure.max()),
        "total_variation_margin": float(total_variation),
        "total_commission": float(total_commission),
        "total_slippage": float(total_slippage),
        "total_collateral_interest": float(total_interest),
        "n_margin_calls": len(margin_calls),
        "margin_calls": margin_calls,
        "n_rolls": len(rolls),
        "rolls": rolls,
        "margin_limited_fills": margin_limited,
        "warnings": warnings,
    }


def _before_parse_date_keys(
    mapping: Mapping[Any, Any],
    field: str,
    tool: str,
    *,
    finite: bool = False,
) -> Dict[pd.Timestamp, Any]:
    keys = list(mapping)
    if not keys:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            parsed = pd.to_datetime(
                pd.Index([str(k) for k in keys], dtype=object),
                format="ISO8601",
                errors="coerce",
            )
        except (TypeError, ValueError):
            parsed = None
    if parsed is None or not isinstance(parsed, pd.DatetimeIndex):
        raise ValidationError(
            f"{tool}: the keys of {field} mix time-zone-aware and naive dates, "
            "so they cannot share one index. Write every key the same way, "
            "e.g. all as 'YYYY-MM-DD'."
        )
    bad = [key for key, stamp in zip(keys, parsed) if pd.isna(stamp)]
    if bad:
        raise ValidationError(
            f"{tool}: {len(bad)} key(s) of {field} are not ISO dates: "
            f"{_named(bad)}. {field} maps an ISO date ('YYYY-MM-DD') to its "
            "value; a map keyed by tickers or labels is a different input."
        )
    seen: Dict[pd.Timestamp, Any] = {}
    repeated: List[str] = []
    for key, stamp in zip(keys, parsed):
        if stamp in seen:
            repeated.append(f"{seen[stamp]!r} and {key!r}")
        else:
            seen[stamp] = key
    if repeated:
        raise ValidationError(
            f"{tool}: keys of {field} name the same date twice: "
            f"{_named(repeated)}. Only one value per date can be used; keep "
            "the one you mean."
        )
    out: Dict[pd.Timestamp, Any] = {}
    non_finite: List[Any] = []
    for key, stamp in zip(keys, parsed):
        value = mapping[key]
        if finite:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                non_finite.append(key)
                continue
            value = number
        out[stamp] = value
    if non_finite:
        raise ValidationError(
            f"{tool}: {field} has a missing or non-finite value on "
            f"{len(non_finite)} date(s): {_named(non_finite)}. Every value "
            "here enters the running account, where one NaN or infinity makes "
            "every later figure non-finite. Drop those dates or supply the "
            "value."
        )
    return out


def _before_by_day(curve: pd.Series) -> Dict[str, float]:
    return {str(k.date()): v for k, v in curve.items()}


def _before_non_finite_bars(curve: pd.Series) -> int:
    return int((~curve.apply(futures_tools._is_finite_number)).sum())


@contextlib.contextmanager
def _tools_as_before() -> Iterator[None]:
    """The two futures tools wired to the code as it stood."""
    with (
        mock.patch.object(futures_tools, "parse_date_keys", _before_parse_date_keys),
        mock.patch.object(futures_tools, "run_futures_simulation", _before),
        mock.patch.object(futures_tools, "_by_day", _before_by_day),
        mock.patch.object(futures_tools, "_non_finite_bars", _before_non_finite_bars),
        mock.patch.object(futures_hedge_backtest, "run_futures_simulation", _before),
    ):
        yield


# ── exact comparison ─────────────────────────────────────────────────────


def _outcome(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """What one call did -- its result, or its exception's type and text --
    and the Python warnings it raised, in order."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result: Any = ("returned", fn(*args, **kwargs))
        except Exception as exc:  # the refusal is part of what is compared
            result = ("raised", type(exc), str(exc))
    return result, [(w.category, str(w.message)) for w in caught]


def _same_stamp(a: Any, b: Any, where: str) -> None:
    assert type(b) is type(a), where
    if a is pd.NaT:
        assert b is pd.NaT, where
        return
    assert a == b, (where, a, b)
    assert a.unit == b.unit, (where, a.unit, b.unit)
    assert type(a.tzinfo) is type(b.tzinfo), where
    assert str(a.tzinfo) == str(b.tzinfo), where
    assert a.fold == b.fold, where
    assert str(a) == str(b), where
    assert hash(a) == hash(b), where


def _same_index(a: pd.Index, b: pd.Index, where: str) -> None:
    assert type(b) is type(a), (where, type(a), type(b))
    assert a.dtype == b.dtype, (where, a.dtype, b.dtype)
    assert a.name == b.name, where
    assert len(a) == len(b), where
    if isinstance(a, pd.DatetimeIndex):
        assert a.freq == b.freq, where
        assert type(a.tz) is type(b.tz), where
        assert str(a.tz) == str(b.tz), where
        assert a.asi8.tobytes() == b.asi8.tobytes(), where
    else:
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}.index[{i}]")


def _same(a: Any, b: Any, where: str = "result") -> None:
    """Equal to the bit: floats by their bytes (so -0.0 and NaN count),
    Series and indexes with their dtype, name, frequency and zone,
    Timestamps with their unit, zone and fold, containers in order."""
    if isinstance(a, pd.Series):
        assert type(b) is pd.Series, where
        assert a.name == b.name, where
        _same_index(a.index, b.index, where)
        assert a.dtype == b.dtype, where
        assert a.to_numpy().tobytes() == b.to_numpy().tobytes(), where
    elif isinstance(a, pd.Index):
        _same_index(a, b, where)
    elif isinstance(a, (pd.Timestamp, type(pd.NaT))):
        _same_stamp(a, b, where)
    elif isinstance(a, float):
        assert type(b) is float, (where, type(b))
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), (where, a, b)
    elif isinstance(a, dict):
        assert type(b) is type(a), where
        assert len(a) == len(b), (where, len(a), len(b))
        for (ka, va), (kb, vb) in zip(a.items(), b.items()):
            _same(ka, kb, f"{where} key {ka!r}")
            _same(va, vb, f"{where}[{ka!r}]")
    elif isinstance(a, (list, tuple)):
        assert type(b) is type(a), (where, type(a), type(b))
        assert len(a) == len(b), (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{where}[{i}]")
    else:
        assert type(b) is type(a), (where, type(a), type(b))
        assert a == b, (where, a, b)


def _both_engines(**kwargs: Any) -> Any:
    old = _outcome(_before, **kwargs)
    new = _outcome(run_futures_simulation, **kwargs)
    _same(old, new)
    return new[0]


# ── inputs ───────────────────────────────────────────────────────────────


def _bars(kind: str, n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    """One shape of bar index. The zoned intraday shapes run through the
    November 2024 change, whose repeated 1 am hour gives stamps with
    fold=1, and the daily zoned shape through the March change."""
    if kind == "daily":
        return pd.bdate_range("2019-01-02", periods=n)
    if kind == "gaps":
        offsets = np.cumsum(rng.integers(1, 7, n))
        return pd.Timestamp("2019-01-02") + pd.to_timedelta(offsets, unit="D")
    if kind == "intraday":
        return pd.date_range("2024-03-07 09:00", periods=n, freq="97min")
    if kind == "utc":
        return pd.date_range("2024-11-01 14:30", periods=n, freq="45min", tz="UTC")
    if kind == "new_york":
        return pd.date_range("2024-11-01 18:00", periods=n, freq="30min", tz=_NEW_YORK)
    if kind == "new_york_daily":
        return pd.bdate_range("2024-01-02", periods=n, tz=_NEW_YORK)
    if kind == "pytz":
        start = pd.Timestamp("2024-11-01 18:00").tz_localize(_PYTZ_NEW_YORK)
        return pd.date_range(start, periods=n, freq="30min")
    if kind == "offset":
        return pd.date_range("2024-01-02 09:15", periods=n, freq="6h", tz=_KOLKATA)
    raise AssertionError(kind)


_KINDS = [
    "daily",
    "gaps",
    "intraday",
    "utc",
    "new_york",
    "new_york_daily",
    "pytz",
    "offset",
]


def _skip_without_zone(kind: str) -> None:
    if kind.startswith("new_york") and _NEW_YORK is None:
        pytest.skip("no time zone database for zoneinfo on this machine")
    if kind == "pytz" and _PYTZ_NEW_YORK is None:
        pytest.skip("pytz is not installed")


def _spell(stamps: List[pd.Timestamp], spelling: str) -> List[Any]:
    """The same dates as one kind of mapping key."""
    if spelling == "timestamps":
        return list(stamps)
    if spelling == "iso_days":
        return [s.strftime("%Y-%m-%d") for s in stamps]
    if spelling == "iso":
        return [s.isoformat() for s in stamps]
    if spelling == "mixed":
        return [s if i % 2 else s.isoformat() for i, s in enumerate(stamps)]
    if spelling == "dates":
        return [s.date() for s in stamps]
    if spelling == "pydatetime":
        return [s.to_pydatetime() for s in stamps]
    if spelling == "datetime64":
        return [s.to_datetime64() for s in stamps]
    if spelling == "utc_timestamps":
        return [s.tz_convert("UTC") for s in stamps]
    if spelling == "naive_days":
        return [s.strftime("%Y-%m-%d") for s in stamps]
    raise AssertionError(spelling)


def _account(rng: np.random.Generator) -> Dict[str, Any]:
    terms: Dict[str, Any] = dict(
        multiplier=float(rng.choice([5.0, 50.0])),
        initial_capital=float(rng.choice([60_000.0, 250_000.0, 5_000_000.0])),
        commission_per_contract=float(rng.choice([0.0, 2.25])),
        slippage_points=float(rng.choice([0.0, 0.25])),
        collateral_rate=float(rng.choice([0.0, 0.0425])),
        allow_fractional=bool(rng.random() < 0.3),
    )
    if rng.random() < 0.7:
        im = float(rng.choice([6_000.0, 12_000.0, 40_000.0]))
        terms["initial_margin"] = im
        terms["maintenance_margin"] = im * float(rng.choice([0.8, 1.0]))
    return terms


def _case(
    seed: int,
    kind: str,
    price_keys: str,
    contract_keys: Optional[str],
    coverage: str = "every_bar",
    n: int = 160,
) -> Dict[str, Any]:
    """An account over `n` bars of `kind`, its maps keyed as named.

    `coverage` is which dates the contract map names: every bar, every bar
    in shuffled order, or a few bars plus dates that are not bars."""
    rng = np.random.default_rng(seed)
    bars = list(_bars(kind, n, rng))
    price = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    targets = np.round(rng.normal(0.0, 6.0, n), 1)
    keep = rng.random(n) < 0.3
    keep[0] = True
    keys = _spell(bars, price_keys)
    kwargs: Dict[str, Any] = dict(
        prices={k: float(p) for k, p in zip(keys, price)},
        target_contracts={k: float(t) for k, t, x in zip(keys, targets, keep) if x},
        **_account(rng),
    )
    if contract_keys is None:
        return kwargs
    width = int(rng.integers(8, 40))
    codes = [f"C{i // width}" for i in range(n)]
    named = list(range(n))
    if coverage == "shuffled":
        rng.shuffle(named)
    elif coverage == "sparse":
        named = [i for i in range(n) if i == 0 or i % width == 0 or i % 7 == 3]
    ckeys = _spell(bars, contract_keys)
    cmap: Dict[Any, Any] = {ckeys[i]: codes[i] for i in named}
    if coverage == "sparse":
        # Dates that are not bars, which no bar can find.
        for extra in _spell([bars[0] - pd.Timedelta(days=400)], contract_keys):
            cmap[extra] = "OLD"
    kwargs["contract_map"] = cmap
    if rng.random() < 0.6:
        kwargs["roll_day_prior_prices"] = {
            ckeys[i]: float(price[i] * (1 + rng.normal(0, 0.003)))
            for i in range(1, n)
            if i % width == 0
        }
    return kwargs


_ENGINE_CASES = []
for _kind in _KINDS:
    _daily = _kind in ("daily", "gaps", "new_york_daily")
    _aware = _kind not in ("daily", "gaps", "intraday")
    _price_spellings = ["timestamps", "iso"] + (
        ["iso_days"] if not _aware and _daily else []
    )
    _contract_spellings: List[Optional[str]] = [None, "timestamps", "iso", "mixed"]
    if _daily:
        _contract_spellings += ["iso_days", "dates", "pydatetime"]
    if not _aware:
        _contract_spellings += ["datetime64"]
    if _aware:
        _contract_spellings += ["utc_timestamps"]
    if _daily and _aware:
        _contract_spellings += ["naive_days"]
    for _prices in _price_spellings:
        for _contracts in _contract_spellings:
            for _coverage in ("every_bar", "shuffled", "sparse"):
                if _contracts is None and _coverage != "every_bar":
                    continue
                _ENGINE_CASES.append((_kind, _prices, _contracts, _coverage))


@pytest.mark.parametrize(
    "kind, price_keys, contract_keys, coverage",
    _ENGINE_CASES,
    ids=["-".join(str(p) for p in case) for case in _ENGINE_CASES],
)
def test_every_shape_of_input_gives_the_same_account(
    kind, price_keys, contract_keys, coverage
):
    _skip_without_zone(kind)
    seed = _ENGINE_CASES.index((kind, price_keys, contract_keys, coverage))
    _both_engines(**_case(seed, kind, price_keys, contract_keys, coverage))


@pytest.mark.parametrize("seed", range(40))
def test_random_accounts_give_the_same_account(seed):
    """Long daily runs, where the rolls, margin calls and sized-down fills
    all fire, keyed as the tool keys them and as a JSON payload does."""
    spelling = "timestamps" if seed % 2 else "iso_days"
    out = _both_engines(**_case(seed, "daily", spelling, spelling, n=700))
    assert out[0] == "returned"


@pytest.mark.parametrize("n", [_ISO_DAYS_FROM - 1, _ISO_DAYS_FROM])
@pytest.mark.parametrize("coverage", ["every_bar", "shuffled"])
def test_either_side_of_the_bulk_size_gives_the_same_account(n, coverage):
    """A map of ISO days one key short of the bulk path, and one at it."""
    _both_engines(**_case(n, "daily", "iso_days", "iso_days", coverage, n=n))


def test_the_random_accounts_reach_every_branch():
    """The comparison above is only as good as what it exercises."""
    outs = []
    for seed in range(40):
        spelling = "timestamps" if seed % 2 else "iso_days"
        outs.append(
            run_futures_simulation(**_case(seed, "daily", spelling, spelling, n=700))
        )
    assert any(o["n_margin_calls"] for o in outs), "no margin call fired"
    assert any(o["margin_limited_fills"] for o in outs), "no fill was sized down"
    assert any(o["n_rolls"] for o in outs), "no roll fired"
    assert any(
        any(not r["variation_margin_skipped"] for r in o["rolls"]) for o in outs
    ), "no roll booked the old contract's move"
    assert any(o["total_collateral_interest"] > 0 for o in outs), "no interest"


class TestOddKeys:
    """Keys the bulk paths do not read, which must still give what they gave."""

    def _base(self, keys: List[Any], **extra: Any) -> Dict[str, Any]:
        prices = {k: 100.0 + i for i, k in enumerate(keys)}
        return dict(
            prices=prices,
            target_contracts={keys[0]: 2.0},
            multiplier=10.0,
            initial_capital=100_000.0,
            collateral_rate=0.03,
            **extra,
        )

    def test_two_spellings_of_one_date_in_the_prices(self):
        """A repeated stamp: the index goes through `pd.to_datetime`."""
        keys = ["2024-01-02", "2024-01-02T00:00", "2024-01-03", "2024-01-04"]
        _both_engines(**self._base(keys))

    def test_a_key_that_parses_to_no_date(self):
        keys = ["2024-01-02", "NaT", "2024-01-03", "2024-01-04"]
        _both_engines(**self._base(keys, contract_map={"2024-01-02": "A"}))

    def test_timestamps_of_two_units(self):
        keys = [
            pd.Timestamp("2024-01-02").as_unit("s"),
            pd.Timestamp("2024-01-03").as_unit("ns"),
            pd.Timestamp("2024-01-04").as_unit("us"),
        ]
        cmap = {keys[0]: "A", keys[1]: "A", keys[2].as_unit("ms"): "B"}
        _both_engines(**self._base(keys, contract_map=cmap))

    def test_timestamps_outside_the_nanosecond_range(self):
        """pandas 2 cannot hold them in a nanosecond index and refuses the
        map; pandas 3 holds them in seconds. Either way, as before."""
        keys = [
            pd.Timestamp("1500-01-02").as_unit("s"),
            pd.Timestamp("1500-01-03").as_unit("s"),
            pd.Timestamp("1500-01-05").as_unit("s"),
        ]
        cmap = {"1500-01-02": "A", "1500-01-03": "A", "1500-01-05": "B"}
        _both_engines(**self._base(keys, contract_map=cmap))
        _both_engines(**self._base(keys, contract_map=dict(zip(keys, "AAB"))))

    def test_contract_keys_at_the_ends_of_the_calendar(self):
        """Year 0 hashes as its raw integer, which differs between units;
        years 1 and 9999 are the last that hash on their fields."""
        keys = ["2024-01-02", "2024-01-03", "2024-01-04"]
        for ends in (["0001-01-01", "9999-12-31"], ["0000-01-01", "9999-12-31"]):
            cmap = {ends[0]: "Z", "2024-01-02": "A", "2024-01-03": "B", ends[1]: "Y"}
            _both_engines(**self._base(keys, contract_map=cmap))

    def test_timestamps_in_two_zones(self):
        keys = [
            pd.Timestamp("2024-01-02", tz="UTC"),
            pd.Timestamp("2024-01-03", tz=_KOLKATA),
            pd.Timestamp("2024-01-04", tz="UTC"),
        ]
        _both_engines(**self._base(keys, contract_map=dict(zip(keys, "AAB"))))

    def test_naive_and_aware_timestamps_together(self):
        keys = [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-04"),
        ]
        _both_engines(**self._base(keys, contract_map=dict(zip(keys, "AAB"))))

    def test_a_roll_in_the_repeated_hour_keyed_in_another_zone(self):
        """The contract changes on the second 1 am of 3 November 2024, and
        the map names the bars by their UTC instants: the dict does not find
        the repeated hour's bars, so the roll lands an hour later, as
        before."""
        if _NEW_YORK is None:
            pytest.skip("no time zone database for zoneinfo on this machine")
        bars = list(
            pd.date_range("2024-11-02 22:00", periods=16, freq="30min", tz=_NEW_YORK)
        )
        change = next(i for i, b in enumerate(bars) if b.fold)
        for keys in ([b.tz_convert("UTC") for b in bars], bars):
            cmap = {k: ("A" if i < change else "B") for i, k in enumerate(keys)}
            out = _both_engines(
                **self._base(bars, contract_map=cmap, initial_margin=1_000.0)
            )
            assert out[0] == "returned" and out[1]["n_rolls"] == 1

    def test_contract_codes_that_are_not_strings(self):
        keys = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        cmap = {"2024-01-02": 1, "2024-01-03": 1, "2024-01-04": 2.5, "2024-01-05": None}
        _both_engines(**self._base(keys, contract_map=cmap))

    def test_a_timestamp_subclass_as_a_key(self):
        class Stamp(pd.Timestamp):
            pass

        keys = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        cmap = {Stamp("2024-01-02"): "A", Stamp("2024-01-03"): "B"}
        _both_engines(**self._base(keys, contract_map=cmap))

    @pytest.mark.parametrize(
        "bad",
        ["2024-02-30", "2024-13-01", "not-a-date", "2024/01/03", "2024-1-03", ""],
    )
    def test_a_contract_key_that_is_not_a_date(self, bad):
        keys = ["2024-01-02", "2024-01-03", "2024-01-04"]
        cmap = {"2024-01-02": "A", bad: "B", "2024-01-04": "B"}
        _both_engines(**self._base(keys, contract_map=cmap))
        _both_engines(
            **self._base(keys, contract_map={"2024-01-02": "A"}),
        )
        _both_engines(
            **self._base(
                keys,
                contract_map={"2024-01-02": "A", "2024-01-03": "B"},
                roll_day_prior_prices={bad: 100.0},
            )
        )


class TestRefusals:
    """Every refusal keeps its type and its words."""

    _PRICES = {"2024-01-02": 100.0, "2024-01-03": 101.0, "2024-01-04": 99.0}

    @pytest.mark.parametrize(
        "change",
        [
            {"prices": {}},
            {"prices": {"2024-01-02": 100.0}},
            {"prices": {"2024-01-02": 100.0, "2024-01-03": "x"}},
            {"prices": {"2024-01-02": 100.0, "2024-01-03": -1.0}},
            {"prices": {"2024-01-02": 100.0, "2024-01-03": None}},
            {"prices": {"2024-01-02": 100.0, "garbage": 101.0}},
            {
                "prices": {
                    pd.Timestamp("2024-01-02"): 100.0,
                    pd.Timestamp("2024-01-03"): "x",
                }
            },
            {"target_contracts": {}},
            {"target_contracts": {"2024-01-02": "two"}},
            {"contract_map": {}},
            {"multiplier": 0.0},
            {"initial_margin": 100.0, "maintenance_margin": 200.0},
            {"commission_per_contract": -1.0},
            {"collateral_rate": math.nan},
        ],
    )
    def test_the_same_refusal(self, change):
        kwargs = dict(
            prices=self._PRICES,
            target_contracts={"2024-01-02": 1.0},
            multiplier=10.0,
        )
        kwargs.update(change)
        _both_engines(**kwargs)


# ── the tools ────────────────────────────────────────────────────────────


def _tool_payload(seed: int, kind: str, n: int = 200) -> Dict[str, Any]:
    """A JSON-shaped payload: ISO-string keys, as an agent sends them."""
    rng = np.random.default_rng(seed)
    bars = list(_bars(kind, n, rng))
    spell = "iso_days" if kind in ("daily", "gaps") else "iso"
    keys = _spell(bars, spell)
    price = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    targets = np.round(rng.normal(0.0, 6.0, n), 1)
    keep = rng.random(n) < 0.3
    keep[0] = True
    width = int(rng.integers(8, 40))
    payload: Dict[str, Any] = dict(
        prices={k: float(p) for k, p in zip(keys, price)},
        target_contracts={k: float(t) for k, t, x in zip(keys, targets, keep) if x},
        contract_map={k: f"C{i // width}" for i, k in enumerate(keys)},
        **_account(rng),
    )
    if seed % 3 == 0:
        payload["contract_map"] = None
    elif seed % 3 == 1:
        payload["roll_day_prior_prices"] = {
            keys[i]: float(price[i] * 0.999) for i in range(1, n) if i % width == 0
        }
    return payload


def _both_tools(name: str, payload: Dict[str, Any]) -> Any:
    tool, model = futures_tools.FUTURES_TOOL_DISPATCH[name]
    with _tools_as_before():
        old = _outcome(lambda: tool(model(**payload)).model_dump())
    new = _outcome(lambda: tool(model(**payload)).model_dump())
    _same(old, new)
    return new[0]


_TOOL_KINDS = ["daily", "gaps", "intraday", "utc", "offset"]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("kind", _TOOL_KINDS)
def test_the_futures_backtest_tool_gives_the_same_result(kind, seed):
    _skip_without_zone(kind)
    out = _both_tools("run_futures_backtest", _tool_payload(seed, kind))
    assert out[0] == "returned"


def test_an_equity_curve_past_floating_point_is_counted_the_same():
    """The bars whose equity overflowed are counted for the warning."""
    payload = dict(
        prices={"2024-01-02": 1e300, "2024-01-03": 1e308, "2024-01-04": 1e308},
        target_contracts={"2024-01-02": 1e6},
        multiplier=1e6,
        initial_capital=1e15,
        allow_fractional=True,
    )
    out = _both_tools("run_futures_backtest", payload)
    assert any("bar(s)" in w for w in out[1]["warnings"])


@pytest.mark.parametrize(
    "change",
    [
        {"prices": {"2024-01-02": 1.0, "AAPL": 2.0}},
        {"prices": {"2024-01-02": 1.0, "2024-01-02T00:00:00": 2.0}},
        {"prices": {"2024-01-02": 1.0, "2024-01-03": math.nan}},
        {"prices": {"2024-01-02": 1.0, "2024-01-03T00:00:00+00:00": 2.0}},
        {"contract_map": {"2024-01-02": "A", "2024-01-02T00:00": "B"}},
        {"contract_map": {"2024-01-02": "A", "not a date": "B"}},
        {"roll_day_prior_prices": {"2024-01-03": math.inf}},
    ],
)
def test_the_futures_backtest_tool_refuses_the_same_way(change):
    payload = dict(
        prices={"2024-01-02": 100.0, "2024-01-03": 101.0},
        target_contracts={"2024-01-02": 1.0},
        multiplier=10.0,
    )
    payload.update(change)
    out = _both_tools("run_futures_backtest", payload)
    assert out[0] == "raised"


@pytest.mark.parametrize("seed", range(4))
def test_the_hedge_tool_gives_the_same_result(seed):
    """It hands the engine its contract map with the payload's own keys."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2023-01-02", periods=300)
    keys = [d.strftime("%Y-%m-%d") for d in days]
    future = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
    book = 2e6 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 300)))
    payload = dict(
        portfolio_values=dict(zip(keys, book.tolist())),
        future_prices=dict(zip(keys, future.tolist())),
        multiplier=50.0,
        portfolio_beta=1.1,
        rehedge=["daily", "weekly", "monthly", "drift"][seed],
        commission_per_contract=2.0,
        slippage_points=0.25,
        contract_map={k: f"ES{i // 63}" for i, k in enumerate(keys)},
    )
    out = _both_tools("run_futures_hedge_backtest", payload)
    assert out[0] == "returned"
    assert out[1]["n_rolls"] > 0


# ── parse_date_keys ──────────────────────────────────────────────────────


_PARSE_CASES: List[Any] = [
    {},
    {"2024-01-03": 2.0, "2024-01-02T00:00:00": 1.0},
    {"2024-01-02": 1.0, "2024-01-02T00:00:00": 2.0, "2024-01-02 00:00": 3.0},
    {"2024-01-02": 1.0, "AAPL": 2.0, "": 3.0, "2019-13-45": 4.0},
    {"AAPL": 1.0, "MSFT": 2.0},
    {"2024-01-02": 1.0, "2024-01-03": math.nan, "2024-01-04": math.inf},
    {"2024-01-02": 1.0, "2024-01-03": None, "2024-01-04": "x", "2024-01-05": "2.5"},
    {"2024-01-02T09:30:00Z": 1.0, "2024-01-02T10:30:00+00:00": 2.0},
    {"2024-01-02T09:30:00+05:30": 1.0, "2024-01-02T10:30:00+05:30": 2.0},
    {"2024-01-02T09:30:00+05:30": 1.0, "2024-01-02T10:30:00-04:00": 2.0},
    {"2024-01-02": 1.0, "2024-01-02T10:30:00+00:00": 2.0},
    {"2024-11-03T01:30:00-04:00": 1.0, "2024-11-03T01:30:00-05:00": 2.0},
    {"2024-11-03T05:30:00Z": 1.0, "2024-11-03T01:30:00-04:00": 2.0},
    {"2024-01-02T09:30:00.123456789": 1.0, "2024-01-02T09:30:00.123456": 2.0},
    {pd.Timestamp("2024-01-02"): 1.0, dt.date(2024, 1, 3): 2.0, 20240104: 3.0},
    {f"2024-{m:02d}-{d:02d}": float(m * d) for m in range(1, 13) for d in range(1, 29)},
]


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("mapping", _PARSE_CASES, ids=range(len(_PARSE_CASES)))
def test_parse_date_keys_gives_the_same_keys_or_the_same_refusal(mapping, finite):
    old = _outcome(_before_parse_date_keys, mapping, "prices", "a_tool", finite=finite)
    new = _outcome(parse_date_keys, mapping, "prices", "a_tool", finite=finite)
    _same(old, new)


# ── the bulk paths on their own ──────────────────────────────────────────


def _per_bar(lookup: Dict[Any, Any], dates: pd.DatetimeIndex) -> List[Any]:
    return [lookup.get(date) for date in dates]


class TestTheContractDays:
    """`_iso_days` reads only exact ISO days, and `_match_days` gives each
    bar what a dict of those days, built by `pd.Timestamp`, gives it."""

    def _check(self, spelled: List[str], dates: pd.DatetimeIndex) -> Any:
        days = _iso_days(spelled)
        assert days is not None
        codes = [f"K{i}" for i in range(len(spelled))]
        want = _per_bar({pd.Timestamp(s): c for s, c in zip(spelled, codes)}, dates)
        # The dict of the parsed days, used when the match declines.
        assert _per_bar(dict(zip(list(days), codes)), dates) == want
        got = _match_days(days, codes, dates)
        if got is not None:
            assert got == want
        return got

    def test_daily_bars_are_matched(self):
        dates = pd.bdate_range("2024-01-02", periods=300)
        spelled = [d.strftime("%Y-%m-%d") for d in dates]
        got = self._check(spelled, dates)
        assert got is not None and None not in got
        # Every seventh bar, plus dates that are not bars.
        extra = ["2023-06-01", "2031-12-31"]
        got = self._check(spelled[::7] + extra, dates)
        assert got is not None and got.count(None) == len(dates) - len(spelled[::7])

    def test_intraday_bars_find_only_their_midnights(self):
        dates = pd.date_range("2024-01-02", periods=200, freq="60min")
        got = self._check(["2024-01-02", "2024-01-03", "2024-01-05"], dates)
        assert got is not None
        assert [i for i, code in enumerate(got) if code] == [0, 24, 72]

    def test_bars_in_other_units_are_matched_without_loss(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="D").as_unit("s")
        assert self._check(["2024-01-03"], dates) is not None
        shifted = (dates + pd.Timedelta(1, "ns")).as_unit("ns")
        got = self._check(["2024-01-03"], shifted)
        assert got is not None and set(got) == {None}

    def test_a_missing_bar_date_finds_nothing(self):
        dates = pd.DatetimeIndex(["2024-01-02", pd.NaT, "2024-01-04"])
        got = self._check(["2024-01-02", "2024-01-04"], dates)
        assert got == ["K0", None, "K1"]

    def test_zoned_bars_are_left_to_the_dict(self):
        """A naive day never equals a zoned bar; the dict says so itself."""
        dates = pd.bdate_range("2024-01-02", periods=10, tz="UTC")
        assert self._check(["2024-01-02", "2024-01-03"], dates) is None

    def test_far_years_are_matched_where_pandas_can_hold_them(self):
        try:
            dates = pd.DatetimeIndex(
                [
                    pd.Timestamp("1500-01-02").as_unit("s"),
                    pd.Timestamp("9999-12-31").as_unit("s"),
                ]
            )
        except Exception:
            pytest.skip("this pandas cannot hold these years in an index")
        assert (
            self._check(["1500-01-02", "9999-12-31", "0001-01-01"], dates) is not None
        )

    @pytest.mark.parametrize(
        "keys",
        [
            ["2024-01-02", "2024-1-02"],
            ["2024-01-02", "2024-01-02T00:00"],
            ["2024/01/02"],
            ["２０２４-01-02"],
            ["2024-02-30"],
            ["2023-02-29"],
            ["2024-13-01"],
            ["0000-01-01"],
            ["2024-01-02", pd.Timestamp("2024-01-03")],
            [20240102],
        ],
        ids=range(10),
    )
    def test_anything_else_is_built_one_key_at_a_time(self, keys):
        assert _iso_days(keys) is None

    def test_the_ends_of_the_calendar_are_read_in_bulk(self):
        days = _iso_days(["0001-01-01", "9999-12-31", "2024-02-29"])
        assert days is not None and str(days.dtype) == "datetime64[s]"


class TestTheIndexFromTimestamps:
    """`_float_series` builds the index pandas would infer, or lets pandas."""

    @pytest.mark.parametrize(
        "keys",
        [
            list(pd.bdate_range("2024-01-02", periods=60)),
            list(pd.date_range("2024-01-02", periods=60, freq="h", tz="UTC")),
            list(pd.date_range("2024-01-02", periods=60, freq="h", tz=_KOLKATA)),
            [pd.Timestamp("2024-01-02").as_unit("s"), pd.Timestamp("2024-01-03")],
            [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")],
            [pd.Timestamp("1500-01-02").as_unit("s"), pd.Timestamp("2024-01-02")],
            [
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz=_KOLKATA),
            ],
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03", tz="UTC")],
            [pd.Timestamp("2024-01-02"), "2024-01-03"],
        ],
        ids=range(9),
    )
    def test_the_same_series(self, keys):
        data = {k: float(i) for i, k in enumerate(keys)}
        old = _outcome(lambda: pd.Series(data, dtype="float64"))
        new = _outcome(lambda: _float_series(data))
        _same(old, new)

    def test_the_tool_shape_is_built_directly(self):
        keys = list(pd.bdate_range("2024-01-02", periods=60))
        assert _timestamp_index(keys) is not None
        assert _timestamp_index(keys + ["2024-05-01"]) is None
