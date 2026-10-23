"""
`estimate_financing_cost` and `finance.financing_cost`: what a set of
positions costs to finance over a horizon under the account's terms.

Every number is planted: the arithmetic is balance x rate x days / basis
on balances held at today's prices, so each test computes its expected
value from the terms by hand and holds the tool to it. The terms are the
`FinancingSpec` the engines read, so the refusals are the spec's -- a short
with no borrow rate is refused, not borrowed for free.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import pytest

from standard_quant_tools.agent.runtimes.finance import estimate_financing_cost
from standard_quant_tools.agent.runtimes.finance.margin_models import (
    FinancingCostInput,
)
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance.financing_cost import financing_cost


def _line(instrument: str, quantity: float, price: float, **extra: Any) -> Dict:
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": extra.pop("currency", "USD"),
        **extra,
    }


def _terms(**extra: Any) -> Dict[str, Any]:
    return {"benchmark_rate": 0.053, "day_count": "ACT/360", **extra}


def _cost(positions: List[Dict], equity: float, terms: Dict, **horizon: Any):
    return financing_cost(
        positions,
        equity=equity,
        financing=terms,
        base_currency="USD",
        **({"days": 30} if not horizon else horizon),
    )


class TestPlantedAnswers:
    def test_a_levered_long_pays_the_debit_rate_on_what_it_borrows(self):
        """$10m long on $5m of equity borrows $5m; at 5.30% + 50 bp for 30
        days ACT/360 that is 5m x 0.058 x 30 / 360 = 24,166.67, and nothing
        else is charged or credited."""
        r = _cost([_line("A", 100_000, 100.0)], 5e6, _terms(debit_spread_bps=50))
        assert r.debit_interest == pytest.approx(5e6 * 0.058 * 30 / 360, rel=1e-12)
        assert round(r.debit_interest, 2) == 24_166.67
        assert r.net_cost == r.debit_interest
        assert r.credit_interest == r.borrow_cost == r.rebate == 0.0
        assert r.cash == r.interest_bearing_cash == -5e6
        assert r.debit_rate == pytest.approx(0.058, rel=1e-15)
        # The cost as an annual rate: on the $10m gross and on the $5m equity.
        assert r.effective_rate_on_gross == pytest.approx(0.029, rel=1e-12)
        assert r.effective_rate_on_equity == pytest.approx(0.058, rel=1e-12)

    def test_the_same_horizon_by_dates_is_the_same_cost(self):
        """30 calendar days given as two dates accrue exactly what 30 days
        given as a number do under a scalar benchmark."""
        by_days = _cost([_line("A", 100_000, 100.0)], 5e6, _terms(debit_spread_bps=50))
        by_dates = _cost(
            [_line("A", 100_000, 100.0)],
            5e6,
            _terms(debit_spread_bps=50),
            start="2026-03-02",
            end="2026-04-01",
        )
        assert by_dates.days == 30
        assert by_dates.debit_interest == by_days.debit_interest
        assert (by_dates.start_date, by_dates.end_date) == ("2026-03-02", "2026-04-01")
        assert (by_days.start_date, by_days.end_date) == (None, None)

    @pytest.mark.parametrize("k", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    def test_a_book_at_leverage_k_returns_k_r_less_k_minus_one_f(self, k):
        """With no spreads, k times the equity long in an asset returning R
        over the horizon returns kR - (k - 1) f on the equity, f = benchmark
        x days / basis: the debit above 1x and the credit below it are one
        expression, to 1e-12."""
        rng = random.Random(7)
        equity = 1_000_000.0
        price = 50.0
        for _ in range(20):
            R = rng.uniform(-0.2, 0.2)
            rate = rng.uniform(-0.01, 0.08)
            days = rng.randint(1, 400)
            positions = [] if k == 0 else [_line("A", k * equity / price, price)]
            if not positions:
                # A book with no lines holds only cash; one zero-priced line
                # stands in, since a position list is never empty.
                positions = [_line("Z", 1.0, 0.0)]
            r = financing_cost(
                positions,
                equity=equity,
                financing={"benchmark_rate": rate, "day_count": "ACT/365F"},
                base_currency="USD",
                days=days,
            )
            f = rate * days / 365.0
            pnl = k * equity * R - r.net_cost
            assert pnl / equity == pytest.approx(k * R - (k - 1) * f, abs=1e-12)

    def test_a_dated_benchmark_charges_each_day_at_the_rate_in_force(self):
        """Fixings 5% on Friday 2 January and 6% on Monday 5 January, a
        horizon to Wednesday 7 January: Friday's rate covers Friday to
        Monday (3 days) and Monday's the last 2, so $950k borrowed pays
        950k x (0.05 x 3 + 0.06 x 2) / 360 = 712.50. A fixing after the
        horizon is never read."""
        r = _cost(
            [_line("A", 200_000, 10.0)],
            1.05e6,
            _terms(
                benchmark_rate={
                    "2026-01-02": 0.05,
                    "2026-01-05": 0.06,
                    "2026-01-20": 0.20,
                }
            ),
            start="2026-01-02",
            end="2026-01-07",
        )
        assert r.debit_interest == pytest.approx(950_000 * 0.27 / 360, rel=1e-12)
        assert r.debit_rate == pytest.approx(0.27 / 5, rel=1e-12)
        assert r.benchmark["kind"] == "dated"
        assert (r.benchmark["first"], r.benchmark["last"]) == (0.05, 0.06)

    def test_a_long_gap_between_fixings_is_reported(self):
        """A rate given on the 2nd and next on the 16th is in force for all
        fourteen days, and the result says so rather than leaving a missing
        fortnight of fixings to look like a quiet period."""
        r = _cost(
            [_line("A", 200_000, 10.0)],
            1e6,
            _terms(benchmark_rate={"2026-01-02": 0.05, "2026-01-16": 0.06}),
            start="2026-01-02",
            end="2026-01-20",
        )
        assert any("2026-01-02 to 2026-01-16 (14 calendar days)" in n for n in r.notes)

    def test_idle_cash_earns_the_credit_rate_and_the_spread_is_the_drag(self):
        """$400k of a $1m account invested leaves $600k idle; at 5.30% less
        25 bp for 90 days ACT/360 it earns 600k x 0.0505 x 90 / 360, and the
        25 bp the broker keeps is the cash drag: 600k x 0.0025 x 90 / 360."""
        r = _cost(
            [_line("A", 4_000, 100.0)],
            1e6,
            _terms(credit_spread_bps=25),
            days=90,
        )
        assert r.credit_interest == pytest.approx(600_000 * 0.0505 * 0.25, rel=1e-12)
        assert r.cash_drag == pytest.approx(600_000 * 0.0025 * 0.25, rel=1e-9)
        assert r.net_cost == pytest.approx(-r.credit_interest, rel=1e-15)
        assert r.debit_interest == 0.0

    def test_each_short_pays_its_own_borrow_from_where_the_terms_put_it(self):
        """A line's own borrow_fee_bps overrides the terms' borrow_bps for
        its instrument, which overrides default_borrow_bps; each short pays
        market value x bps x days / basis and the result names the source."""
        positions = [
            _line("A", -1_000, 50.0, borrow_fee_bps=500),
            _line("B", -2_000, 25.0),
            _line("C", -100, 10.0),
            _line("L", 10_000, 10.0),
        ]
        r = _cost(
            positions,
            1e6,
            _terms(borrow_bps={"A": 30, "B": 300}, default_borrow_bps=40),
            days=360,
        )
        rows = {b.instrument: b for b in r.borrows}
        assert {k: (v.borrow_bps, v.source) for k, v in rows.items()} == {
            "A": (500.0, "position"),
            "B": (300.0, "borrow_bps"),
            "C": (40.0, "default_borrow_bps"),
        }
        assert rows["A"].cost == pytest.approx(50_000 * 0.05, rel=1e-12)
        assert rows["B"].cost == pytest.approx(50_000 * 0.03, rel=1e-12)
        assert rows["C"].cost == pytest.approx(1_000 * 0.004, rel=1e-12)
        assert r.borrow_cost == pytest.approx(2_500 + 1_500 + 4, rel=1e-12)

    def test_segregated_proceeds_earn_the_rebate_and_not_the_credit_rate(self):
        """A $200k short with segregated proceeds: the lender holds the
        $200k and pays benchmark - 40 bp on it; the account's own $1m of
        cash earns the credit rate. Under net_cash the $1.2m all earns the
        credit rate and there is no rebate."""
        positions = [_line("S", -10_000, 20.0)]
        terms = _terms(default_borrow_bps=25)
        seg = _cost(
            positions,
            1e6,
            {**terms, "short_proceeds": "segregated", "rebate_spread_bps": 40},
            days=360,
        )
        assert seg.segregated_proceeds == 200_000.0
        assert seg.interest_bearing_cash == 1_000_000.0
        assert seg.rebate == pytest.approx(200_000 * 0.049, rel=1e-12)
        assert seg.credit_interest == pytest.approx(1_000_000 * 0.053, rel=1e-12)
        assert seg.cash_drag == pytest.approx(200_000 * 0.004, rel=1e-9)
        net = _cost(positions, 1e6, terms, days=360)
        assert net.rebate == 0.0 and net.segregated_proceeds == 0.0
        assert net.credit_interest == pytest.approx(1_200_000 * 0.053, rel=1e-12)

    def test_lines_in_other_currencies_are_financed_in_the_base(self):
        """A EUR line at 1.10 USD is $1.1m of exposure, financed in dollars
        over the one benchmark, and the result says that a cash balance in
        euros at its own rate is a separate calculation."""
        r = financing_cost(
            [_line("E", 10_000, 100.0, currency="EUR")],
            equity=600_000,
            financing=_terms(),
            base_currency="USD",
            fx_rates={"EUR": 1.10},
            days=360,
        )
        assert r.long == pytest.approx(1_100_000, rel=1e-15)
        assert r.debit_interest == pytest.approx(500_000 * 0.053, rel=1e-12)
        assert any("EUR" in note for note in r.notes)

    def test_reordering_the_positions_moves_no_number(self):
        """Every total is a correctly rounded sum, so the same lines in any
        order give the same bits."""
        rng = random.Random(3)
        positions = [
            _line(
                f"N{i}", rng.choice([-1, 1]) * rng.uniform(1, 1e5), rng.uniform(1, 500)
            )
            for i in range(60)
        ]
        terms = _terms(default_borrow_bps=75, debit_spread_bps=30)
        first = _cost(positions, 2e6, terms)
        rng.shuffle(positions)
        again = _cost(positions, 2e6, terms)
        assert first.net_cost == again.net_cost
        assert first.borrow_cost == again.borrow_cost
        assert first.gross == again.gross


class TestRefusals:
    def test_a_short_with_no_borrow_rate_is_refused_by_name(self):
        """No borrow_fee_bps on the line, no borrow_bps for it and no
        default: the short would borrow for free, so it is refused, naming
        the line and the three ways to give it a rate."""
        with pytest.raises(ValidationError) as caught:
            _cost([_line("L", 100, 10.0), _line("XYZ", -100, 10.0)], 1e5, _terms())
        message = str(caught.value)
        assert "positions[1] 'XYZ'" in message
        for fix in ("borrow_fee_bps", "borrow_bps['XYZ']", "default_borrow_bps"):
            assert fix in message

    @pytest.mark.parametrize(
        "horizon, needle",
        [
            ({}, "no horizon was given"),
            ({"days": 30, "start": "2026-01-02", "end": "2026-02-01"}, "twice"),
            ({"start": "2026-01-02"}, "needs both dates"),
            ({"start": "2026-02-01", "end": "2026-02-01"}, "is not after"),
            ({"days": 0}, "between 1 and"),
        ],
    )
    def test_a_horizon_must_be_given_once_and_run_forward(self, horizon, needle):
        """A horizon given twice, half given, empty or zero days is refused
        rather than one reading picked: the cost scales with it."""
        with pytest.raises(ValidationError, match=needle):
            financing_cost(
                [_line("A", 1, 1.0)],
                equity=1.0,
                financing=_terms(),
                base_currency="USD",
                **horizon,
            )

    def test_a_dated_benchmark_needs_dates_not_days(self):
        """A dated rate is read on dates, so a number of days against it has no
        rate to read and is refused."""
        with pytest.raises(ValidationError, match="give start and end dates"):
            _cost(
                [_line("A", 1, 1.0)],
                1.0,
                _terms(benchmark_rate={"2026-01-02": 0.05}),
                days=10,
            )

    def test_a_start_without_a_fixing_is_refused_unless_the_rate_carries(self):
        """Under benchmark_fill='refuse' the start date needs a fixing of
        its own, as every bar a balance is carried from does in the
        engines; 'previous' carries the last one forward."""
        rates = {"2026-01-02": 0.05}
        with pytest.raises(
            ValidationError, match="no rate for the bar dated 2026-01-05"
        ):
            _cost(
                [_line("A", 2, 1.0)],
                1.0,
                _terms(benchmark_rate=rates),
                start="2026-01-05",
                end="2026-01-06",
            )
        r = _cost(
            [_line("A", 2, 1.0)],
            1.0,
            _terms(benchmark_rate=rates, benchmark_fill="previous"),
            start="2026-01-05",
            end="2026-01-06",
        )
        assert r.debit_interest == pytest.approx(1.0 * 0.05 / 360, rel=1e-12)

    def test_a_short_at_a_negative_price_is_refused(self):
        """A borrow is charged on the value of the stock borrowed, which a
        negative price does not have."""
        with pytest.raises(ValidationError, match="negative price"):
            _cost([_line("S", -10, -5.0)], 100.0, _terms(default_borrow_bps=10))

    @pytest.mark.parametrize("equity", [0.0, -1.0, math.nan, math.inf])
    def test_equity_must_be_finite_and_positive(self, equity):
        """The account's own capital is what the broker does not fund; none, a
        deficit or a non-finite one has no financing to price."""
        with pytest.raises(ValidationError, match="equity="):
            _cost([_line("A", 1, 1.0)], equity, _terms())


class TestTheTool:
    def _args(self, **extra: Any) -> Dict[str, Any]:
        return {
            "positions": [_line("A", 100_000, 100.0)],
            "base_currency": "USD",
            "equity": 5e6,
            "financing": _terms(debit_spread_bps=50),
            "days": 30,
            **extra,
        }

    def test_dispatch_reports_the_planted_cost(self):
        """Through the tool surface, the planted levered long reports its
        24,166.67 and a scalar benchmark."""
        result = dispatch("estimate_financing_cost", self._args())
        assert round(result["net_cost"], 2) == 24_166.67
        assert result["benchmark"]["kind"] == "scalar"
        assert result["borrows"] == []

    def test_a_zero_gross_book_has_no_rate_on_gross_and_says_why(self):
        """Every price zero: there is no gross exposure, so the rate on it
        is null with the reason, while the rate on equity is a number."""
        result = estimate_financing_cost(
            FinancingCostInput(**self._args(positions=[_line("Z", 10, 0.0)]))
        )
        assert result.effective_rate_on_gross is None
        assert result.effective_rate_on_equity is not None
        assert any("effective_rate_on_gross is null" in w for w in result.warnings)

    def test_a_date_that_is_not_iso_is_refused_by_field(self):
        """A date written another way is refused naming the field, not read as
        a guess."""
        with pytest.raises(ValidationError, match="start_date="):
            estimate_financing_cost(
                FinancingCostInput(
                    **self._args(
                        days=None, start_date="02/01/2026", end_date="2026-03-01"
                    )
                )
            )

    def test_the_same_input_gives_the_same_bits(self):
        """Nothing is fetched or drawn, so a repeat call is the same answer to
        the bit."""
        args = self._args(
            positions=[_line("A", 1234.5, 98.76), _line("B", -77, 12.3)],
            financing=_terms(default_borrow_bps=123, credit_spread_bps=10),
        )
        assert dispatch("estimate_financing_cost", args) == dispatch(
            "estimate_financing_cost", args
        )
