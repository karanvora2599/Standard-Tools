"""
The futures account reads its prices, targets and day counts from arrays.

`run_futures_simulation` used to read `series.iloc[i]` and compute
`(dates[i] - dates[i - 1]).days` on every bar, boxing a scalar or two
Timestamps each time, and that boxing was most of the loop's cost. It now
reads the same float64 values from ndarrays and the same whole-day counts
from one precomputed list (see the CHANGELOG entry of 2026-10-01).

The pre-change loop is kept below as `_reference`, and every output of the
two -- every curve, every record, every total and every warning -- is
required to be EXACTLY equal: no tolerance, because nothing about the
arithmetic changed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.futures_engine import (
    _non_negative,
    _positive,
    _series,
    run_futures_simulation,
)
from standard_quant_tools.error import ValidationError


def _reference(
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
    """`run_futures_simulation` as it stood before the arrays, comments cut."""
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

    price_series = _series(prices, "prices")
    if (price_series <= 0).any():
        raise ValidationError("prices contains a non-positive value.")
    targets = _series(target_contracts, "target_contracts").reindex(price_series.index)
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

    previous_price = float(price_series.iloc[0])
    previous_contract = rolls_by_date.get(dates[0])

    for i, date in enumerate(dates):
        price = float(price_series.iloc[i])
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
            days = max((dates[i] - dates[i - 1]).days, 0)
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

        target = float(targets.iloc[i])
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


# ── exact comparison ─────────────────────────────────────────────────────


def _same_float(a: float, b: float) -> bool:
    """Bit-for-bit, with NaN equal to NaN and the sign of zero kept."""
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def _assert_identical(old: Dict[str, Any], new: Dict[str, Any]) -> None:
    assert list(old) == list(new)
    for key, want in old.items():
        got = new[key]
        if isinstance(want, pd.Series):
            assert isinstance(got, pd.Series), key
            assert want.name == got.name, key
            assert want.index.equals(got.index), key
            assert want.dtype == got.dtype, key
            assert want.to_numpy().tobytes() == got.to_numpy().tobytes(), key
        elif isinstance(want, float):
            assert type(got) is float, key
            assert _same_float(want, got), (key, want, got)
        else:
            assert type(got) is type(want), key
            assert got == want, key


def _both(**kwargs: Any) -> Dict[str, Any]:
    old = _reference(**kwargs)
    new = run_futures_simulation(**kwargs)
    _assert_identical(old, new)
    return new


# ── planted answers ──────────────────────────────────────────────────────


class TestPlantedDayCounts:
    """Interest accrues on the calendar days between bars, so a weekend
    earns three days and a same-day pair of bars earns none."""

    def test_a_weekend_earns_three_days_of_interest(self):
        prices = {"2024-01-05": 100.0, "2024-01-08": 100.0}  # Friday, Monday
        out = _both(
            prices=prices,
            target_contracts={"2024-01-05": 0.0},
            multiplier=1.0,
            initial_capital=365_000.0,
            collateral_rate=0.10,
        )
        # 365,000 x 10% x 3 / 365 = 300 exactly.
        assert out["total_collateral_interest"] == 300.0
        assert out["final_equity"] == 365_300.0

    def test_no_rate_is_no_interest(self):
        out = _both(
            prices={"2024-01-05": 100.0, "2024-01-08": 100.0},
            target_contracts={"2024-01-05": 0.0},
            multiplier=1.0,
            initial_capital=365_000.0,
            collateral_rate=0.0,
        )
        assert out["total_collateral_interest"] == 0.0
        assert out["final_equity"] == 365_000.0

    def test_two_bars_on_one_day_earn_nothing(self):
        out = _both(
            prices={"2024-01-05 09:30": 100.0, "2024-01-05 15:30": 100.0},
            target_contracts={"2024-01-05 09:30": 0.0},
            multiplier=1.0,
            initial_capital=365_000.0,
            collateral_rate=0.10,
        )
        assert out["total_collateral_interest"] == 0.0

    def test_a_long_gap_earns_every_day_of_it(self):
        out = _both(
            prices={"2024-01-01": 100.0, "2024-01-31": 100.0},
            target_contracts={"2024-01-01": 0.0},
            multiplier=1.0,
            initial_capital=365_000.0,
            collateral_rate=0.10,
        )
        assert out["total_collateral_interest"] == 3_000.0

    def test_targets_are_read_on_their_own_bar(self):
        """The array read is positional: the target on bar i sizes bar i."""
        dates = pd.bdate_range("2024-01-01", periods=4)
        out = _both(
            prices=dict(zip(dates, [100.0, 101.0, 103.0, 102.0])),
            target_contracts={dates[1]: 2.0, dates[3]: 0.0},
            multiplier=10.0,
            initial_capital=1_000.0,
        )
        assert out["position_curve"].tolist() == [0.0, 2.0, 2.0, 0.0]
        # Held from 101 to 102 (via 103): 2 x 10 x (102 - 101) = 20.
        assert out["total_variation_margin"] == 20.0


# ── random and edge inputs against the reference ─────────────────────────


def _scenario(seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 400))
    kind = seed % 4
    if kind == 0:
        dates = pd.bdate_range("2019-01-01", periods=n)
    elif kind == 1:
        # Ragged calendar gaps of 1 to 6 days.
        offsets = np.cumsum(rng.integers(1, 7, n))
        dates = pd.Timestamp("2019-01-01") + pd.to_timedelta(offsets, unit="D")
    elif kind == 2:
        # Intraday bars, several per day, tz-aware.
        dates = pd.date_range("2019-01-01 14:30", periods=n, freq="97min", tz="UTC")
    else:
        dates = pd.date_range("2019-01-01", periods=n, freq="D")
    vol = float(rng.choice([0.002, 0.01, 0.05]))
    price = 4000.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    targets = np.round(rng.normal(0.0, 6.0, n), 1)
    keep = rng.random(n) < 0.3
    keep[0] = True
    target_map = {d: float(t) for d, t, k in zip(dates, targets, keep) if k}
    kwargs: Dict[str, Any] = dict(
        prices={d: float(p) for d, p in zip(dates, price)},
        target_contracts=target_map,
        multiplier=float(rng.choice([5.0, 50.0])),
        initial_capital=float(rng.choice([50_000.0, 250_000.0, 5_000_000.0])),
        commission_per_contract=float(rng.choice([0.0, 2.25])),
        slippage_points=float(rng.choice([0.0, 0.25])),
        collateral_rate=float(rng.choice([0.0, 0.0425])),
        allow_fractional=bool(rng.random() < 0.3),
    )
    if rng.random() < 0.7:
        im = float(rng.choice([6_000.0, 12_000.0, 40_000.0]))
        kwargs["initial_margin"] = im
        kwargs["maintenance_margin"] = im * float(rng.choice([0.8, 1.0]))
    if rng.random() < 0.6:
        width = int(rng.integers(5, 60))
        kwargs["contract_map"] = {d: f"C{i // width}" for i, d in enumerate(dates)}
        if rng.random() < 0.5:
            kwargs["roll_day_prior_prices"] = {
                d: float(p * (1 + rng.normal(0, 0.003)))
                for i, (d, p) in enumerate(zip(dates, price))
                if i and i % width == 0
            }
    return kwargs


@pytest.mark.parametrize("seed", range(60))
def test_random_accounts_match_the_reference_exactly(seed):
    _both(**_scenario(seed))


def test_the_scenarios_reach_every_branch():
    """The comparison above is only as good as what it exercises."""
    outs = [run_futures_simulation(**_scenario(seed)) for seed in range(60)]
    assert any(o["n_margin_calls"] for o in outs), "no margin call fired"
    assert any(o["margin_limited_fills"] for o in outs), "no fill was sized down"
    assert any(o["n_rolls"] for o in outs), "no roll fired"
    assert any(
        any(not r["variation_margin_skipped"] for r in o["rolls"]) for o in outs
    ), "no roll booked the old contract's move"
    assert any(o["total_collateral_interest"] > 0 for o in outs), "no interest"


def test_an_account_wiped_out_matches_the_reference():
    dates = pd.bdate_range("2024-01-01", periods=6)
    _both(
        prices=dict(zip(dates, [100.0, 100.0, 40.0, 10.0, 5.0, 5.0])),
        target_contracts={dates[0]: 10.0},
        multiplier=100.0,
        initial_capital=10_000.0,
        collateral_rate=0.05,
    )


def test_two_bars_and_one_row_inputs_match_the_reference():
    _both(
        prices={"2024-01-02": 10.0, "2024-01-03": 11.0},
        target_contracts={"2024-01-02": 1.0},
        multiplier=1.0,
    )
    for impl in (_reference, run_futures_simulation):
        with pytest.raises(ValidationError, match="at least two bars"):
            impl(
                prices={"2024-01-02": 10.0},
                target_contracts={"2024-01-02": 1.0},
                multiplier=1.0,
            )


def test_unsorted_keys_match_the_reference():
    """`_series` sorts the dates, so the day counts are taken in order."""
    _both(
        prices={"2024-01-10": 12.0, "2024-01-02": 10.0, "2024-01-05": 11.0},
        target_contracts={"2024-01-02": 3.0},
        multiplier=1.0,
        collateral_rate=0.03,
    )
