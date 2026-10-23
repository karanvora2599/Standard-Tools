"""
`analyze_margin_runway` and `finance.margin.margin_runway`: a stress path
walked day by day against the margin requirements.

The planted path is the one a desk draws on a whiteboard: initial 10k,
maintenance 8k, equity 12k, losing 1k a day. Equity is 7k on day 5, the
first day below 8k, so the call comes on day 5 for the 3k that restores
the initial 10k. The buffer is exact -- the cash that keeps the lowest day
at maintenance, and one unit less calls -- and a call reads no day after
its own, so changing the path later moves nothing reported before.
Revalued paths are held to a brute force: the margin of each day's prices
from `margin_requirement`, walked by a plain loop.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Any, Dict, List, Optional

import pytest

from standard_quant_tools.agent.runtimes.finance import analyze_margin_runway
from standard_quant_tools.agent.runtimes.finance.margin_models import (
    MarginRunwayInput,
)
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance.margin import margin_requirement, margin_runway


def _line(instrument: str, quantity: float, price: float, **extra: Any) -> Dict:
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": "USD",
        **extra,
    }


def _planted(**kwargs: Any):
    args = dict(
        equity_changes=[-1_000.0] * 10,
        initial_margin=10_000.0,
        maintenance_margin=8_000.0,
    )
    args.update(kwargs)
    return margin_runway(12_000.0, **args)


class TestThePlantedPath:
    def test_the_first_call_is_on_day_five_for_three_thousand(self):
        """Equity 7k on day 5 is the first day below 8k, and restoring the
        initial 10k asks for 3k."""
        r = _planted()
        assert r.first_call_day == 5
        assert r.first_call_amount == 3_000.0
        assert r.calls[0].equity == 7_000.0 and r.calls[0].target == 10_000.0
        assert (r.initial_today, r.maintenance_today) == (10_000.0, 8_000.0)

    def test_restoring_to_maintenance_asks_only_for_the_shortfall(self):
        """The same call restoring only maintenance asks for the 1k below 8k."""
        r = _planted(restore_to="maintenance")
        assert (r.first_call_day, r.first_call_amount) == (5, 1_000.0)

    def test_equity_at_maintenance_exactly_is_not_a_call(self):
        """Day 4 has equity 8,000, equal to maintenance: the rule is equity
        BELOW maintenance, so the call waits for day 5."""
        r = _planted(equity_changes=[-1_000.0] * 4)
        assert r.first_call_day is None and r.tightest_cushion == 0.0

    def test_a_path_that_never_breaches_has_no_call_and_says_why(self):
        """No day below maintenance: no call, a null day and amount with the
        reason, and no buffer needed."""
        result = dispatch(
            "analyze_margin_runway",
            {
                "equity": 12_000,
                "equity_changes": [-500, 200, -700],
                "initial_margin": 10_000,
                "maintenance_margin": 8_000,
            },
        )
        assert result["first_call_day"] is None
        assert result["first_call_amount"] is None
        assert result["calls"] == [] and result["buffer"] == 0.0
        assert any(
            w.startswith("first_call_day is null: the equity stays at or above")
            for w in result["warnings"]
        )

    def test_the_buffer_is_exact(self):
        """The lowest day is day 10 at 2,000 against 8,000: 6,000 of cash
        today avoids every call, and 5,999 calls on day 10."""
        r = _planted()
        assert (r.buffer, r.tightest_day) == (6_000.0, 10)
        enough = margin_runway(
            12_000.0 + r.buffer,
            equity_changes=[-1_000.0] * 10,
            initial_margin=10_000.0,
            maintenance_margin=8_000.0,
        )
        assert enough.first_call_day is None
        short = margin_runway(
            12_000.0 + r.buffer - 1.0,
            equity_changes=[-1_000.0] * 10,
            initial_margin=10_000.0,
            maintenance_margin=8_000.0,
        )
        assert short.first_call_day == 10

    def test_the_buffer_is_the_least_float_that_clears_every_day(self):
        """On changes no binary float holds exactly, the buffer still leaves
        every day at or above maintenance, and the next float below it
        leaves one day under -- each day's equity taken as the exact sum of
        the path so far, rounded once."""
        rng = random.Random(5)
        checked = 0
        for _ in range(50):
            changes = [round(rng.uniform(-900, 600), 2) for _ in range(40)]
            r = margin_runway(
                5_000.3,
                equity_changes=changes,
                initial_margin=4_000.1,
                maintenance_margin=3_000.7,
            )
            if r.buffer == 0.0:
                continue
            total, path = Fraction(5_000.3), [5_000.3]
            for change in changes:
                total += Fraction(change)
                path.append(float(total))
            assert all(e + r.buffer >= 3_000.7 for e in path)
            below = math.nextafter(r.buffer, 0.0)
            assert any(e + below < 3_000.7 for e in path)
            checked += 1
        assert checked > 10


class TestCallsAndDeleveraging:
    def test_an_unmet_call_deleverages_when_its_cure_period_ends(self):
        """A day-5 call with two days to cure and no recovery starts
        deleveraging on day 7."""
        r = _planted(cure_days=2)
        assert r.deleveraging_day == 7
        assert r.calls[0].outcome == "deleveraged" and r.calls[0].deadline == 7

    def test_no_cure_period_deleverages_on_the_call_day(self):
        """With no time to meet a call, deleveraging starts the day it is made."""
        assert _planted(cure_days=0).deleveraging_day == 5

    def test_a_call_the_market_cures_is_followed_by_the_next_one(self):
        """Down to 7k on day 5, back to 11k on day 6 -- over the 10k
        target, so the call is cured -- then down to 7.5k on day 8: a
        second call, for 2.5k."""
        r = margin_runway(
            12_000.0,
            equity_changes=[-1_000.0] * 5 + [4_000.0, -2_000.0, -1_500.0],
            initial_margin=10_000.0,
            maintenance_margin=8_000.0,
            cure_days=3,
        )
        assert [(c.day, c.outcome, c.resolved_day) for c in r.calls] == [
            (5, "cured", 6),
            (8, "outstanding", None),
        ]
        assert r.calls[1].amount == 2_500.0
        assert r.deleveraging_day is None

    def test_a_call_outstanding_when_the_path_ends_is_named(self):
        """The default cure period is FINRA 4210(f)(6)'s 15 days, longer
        than the 10-day path: the call is outstanding and deleveraging is
        null with the reason."""
        result = dispatch(
            "analyze_margin_runway",
            {
                "equity": 12_000,
                "equity_changes": [-1_000] * 10,
                "initial_margin": 10_000,
                "maintenance_margin": 8_000,
            },
        )
        assert result["cure_days"] == 15
        assert result["calls"][0]["outcome"] == "outstanding"
        assert result["deleveraging_day"] is None
        assert any("unmet when the path ends" in w for w in result["warnings"])

    def test_an_account_below_maintenance_today_is_called_on_day_zero(self):
        """An account already below maintenance is in call today, before any
        step of the path."""
        r = margin_runway(
            7_000.0,
            equity_changes=[0.0],
            initial_margin=10_000.0,
            maintenance_margin=8_000.0,
        )
        assert (r.first_call_day, r.first_call_amount) == (0, 3_000.0)

    def test_equity_below_zero_is_the_first_deficit_day(self):
        """Losing 5k a day from 12k, equity is -3k on day 3: the first day the
        account owes the broker."""
        r = _planted(equity_changes=[-5_000.0] * 4)
        assert r.first_deficit_day == 3 and r.lowest_equity == -8_000.0

    def test_nothing_after_a_day_moves_what_is_reported_on_or_before_it(self):
        """The first call reads no later day: rewriting the path after the
        call leaves its day, its amount and its equity unchanged."""
        rng = random.Random(9)
        base = [-1_000.0] * 5 + [rng.uniform(-2_000, 2_000) for _ in range(20)]
        first = _planted(equity_changes=base)
        for _ in range(20):
            changed = base[:5] + [rng.uniform(-5_000, 5_000) for _ in range(20)]
            again = _planted(equity_changes=changed)
            assert (again.first_call_day, again.first_call_amount) == (
                first.first_call_day,
                first.first_call_amount,
            )
            assert again.calls[0].equity == first.calls[0].equity


def _brute_force(
    equity: float, positions: List[Dict], returns: Dict[str, List[float]]
) -> Dict[str, Optional[float]]:
    """Each day's prices compounded line by line, each day's requirement
    from margin_requirement on those prices, and a plain loop for the first
    day below maintenance."""
    days = len(next(iter(returns.values())))
    prices = {p["instrument"]: p["price"] for p in positions}
    first_day = amount = None
    deficits = []
    for t in range(days + 1):
        if t:
            for name in prices:
                prices[name] *= 1.0 + returns[name][t - 1]
        today = [dict(p, price=prices[p["instrument"]]) for p in positions]
        req = margin_requirement(today)
        value_change = sum(
            p["quantity"] * (prices[p["instrument"]] - p["price"]) for p in positions
        )
        e = equity + value_change
        deficits.append(req.maintenance - e)
        if first_day is None and e < req.maintenance:
            first_day, amount = t, req.initial - e
    return {"day": first_day, "amount": amount, "buffer": max(0.0, max(deficits))}


class TestRevaluedPaths:
    def test_a_falling_long_is_called_where_the_closed_form_says(self):
        """$100k long on $50k equity, falling 10% a day: equity is 50k -
        100k(1 - 0.9^t) against 25% x 100k x 0.9^t, first below on day 4
        (0.9^t < 2/3), for the initial 50% x 65,610 less equity 15,610 =
        17,195."""
        r = margin_runway(
            50_000.0, positions=[_line("A", 1_000, 100.0)], returns={"A": [-0.1] * 10}
        )
        t = math.ceil(math.log(2 / 3) / math.log(0.9))
        assert r.first_call_day == t == 4
        assert r.first_call_amount == pytest.approx(50_000 * (1 - 0.9**4), rel=1e-12)
        assert r.source == "revalued"

    def test_a_short_that_rallies_through_five_dollars_changes_tier(self):
        """A short at $4 needs its whole value; at $5 and over it needs $5 a
        share. The revalued path reads each day's tier, as the brute force
        on each day's prices does."""
        positions = [_line("S", -1_000, 4.0), _line("L", 2_000, 10.0)]
        returns = {"S": [0.08] * 12, "L": [-0.01] * 12}
        r = margin_runway(9_000.0, positions=positions, returns=returns)
        oracle = _brute_force(9_000.0, positions, returns)
        assert r.first_call_day == oracle["day"]
        assert r.first_call_amount == pytest.approx(oracle["amount"], rel=1e-9)
        assert r.buffer == pytest.approx(oracle["buffer"], rel=1e-9)

    def test_random_books_match_the_brute_force(self):
        """Long and short stocks and leveraged ETFs on random daily returns:
        the first call day, its amount and the buffer agree with the brute
        force on every book."""
        rng = random.Random(21)
        for trial in range(25):
            positions = [
                _line(
                    f"N{i}",
                    rng.choice([-1, 1]) * rng.randint(10, 2_000),
                    rng.uniform(2.0, 60.0),
                    **(
                        {"kind": "etf", "leverage": rng.choice([2, 3])}
                        if rng.random() < 0.3
                        else {}
                    ),
                )
                for i in range(rng.randint(1, 6))
            ]
            returns = {
                p["instrument"]: [rng.gauss(0.0, 0.04) for _ in range(30)]
                for p in positions
            }
            equity = rng.uniform(5_000, 80_000)
            r = margin_runway(equity, positions=positions, returns=returns)
            oracle = _brute_force(equity, positions, returns)
            assert r.first_call_day == oracle["day"], trial
            if oracle["amount"] is not None:
                assert r.first_call_amount == pytest.approx(
                    oracle["amount"], rel=1e-9, abs=1e-6
                )
            assert r.buffer == pytest.approx(oracle["buffer"], rel=1e-9, abs=1e-6)

    def test_positions_held_fixed_use_todays_requirement(self):
        """$100k long and $20k short: maintenance 25k + 6k = 31k, initial
        the Reg T 50k + 10k = 60k. Losing 10k a day from 60k, equity is 30k
        on day 3, the first below 31k, and the call restores 60k."""
        positions = [_line("A", 1_000, 100.0), _line("S", -1_000, 20.0)]
        today = margin_requirement(positions)
        r = margin_runway(60_000.0, positions=positions, equity_changes=[-10_000.0] * 3)
        assert (r.initial_today, r.maintenance_today) == (
            today.initial,
            today.maintenance,
        )
        assert r.source == "positions"
        assert (r.first_call_day, r.first_call_amount) == (3, 30_000.0)


class TestRefusals:
    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({}, "no stress path was given"),
            ({"equity_changes": [1.0]}, "both needed"),
            (
                {
                    "equity_changes": [1.0],
                    "initial_margin": 1.0,
                    "maintenance_margin": 2.0,
                },
                "is above initial_margin",
            ),
            (
                {
                    "equity_changes": [1.0],
                    "positions": [_line("A", 1, 1.0)],
                    "initial_margin": 1.0,
                },
                "twice",
            ),
            ({"returns": {"A": [0.1]}}, "none were given"),
            (
                {
                    "positions": [_line("A", 1, 1.0)],
                    "returns": {"A": [0.1]},
                    "equity_changes": [1.0],
                },
                "two stress paths",
            ),
            (
                {"positions": [_line("A", 1, 1.0)], "returns": {"B": [0.1]}},
                "must name the same instruments",
            ),
            (
                {
                    "positions": [_line("A", 1, 1.0), _line("B", 1, 1.0)],
                    "returns": {"A": [0.1], "B": [0.1, 0.2]},
                },
                "different lengths",
            ),
            (
                {"positions": [_line("A", 1, 1.0)], "returns": {"A": [-1.5]}},
                "below -1",
            ),
            (
                {
                    "positions": [
                        _line(
                            "C",
                            -1,
                            2.0,
                            multiplier=100,
                            kind="call",
                            strike=110,
                            underlying_price=100,
                        )
                    ],
                    "returns": {"C": [0.1]},
                },
                "is an option",
            ),
            (
                {"equity_changes": [], "initial_margin": 1, "maintenance_margin": 1},
                "empty",
            ),
            (
                {
                    "equity_changes": [1.0],
                    "initial_margin": 1,
                    "maintenance_margin": 1,
                    "restore_to": "half",
                },
                "restore_to=",
            ),
        ],
    )
    def test_a_path_or_a_requirement_stated_wrongly_is_refused(self, kwargs, needle):
        """Each malformed path or requirement is refused by name rather than
        walked on a guess."""
        with pytest.raises(ValidationError) as caught:
            margin_runway(1_000.0, **kwargs)
        assert needle in str(caught.value)

    def test_the_tool_passes_holdings_through(self):
        """Through the tool, holdings and returns reach the revalued walk and
        call on the closed-form day."""
        result = analyze_margin_runway(
            MarginRunwayInput(
                equity=50_000,
                holdings={
                    "positions": [_line("A", 1_000, 100.0)],
                    "base_currency": "USD",
                },
                returns={"A": [-0.1] * 10},
            )
        )
        assert result.first_call_day == 4
        assert result.source == "revalued"
