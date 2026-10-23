"""
NPV, every real IRR, MIRR, duration, convexity and DV01 of a stream of cash
flows, against answers worked by hand or published.

THE PLANTED ANSWERS.

    annuity      5 x 100 at 8%, first a period out   100 (1 - 1.08^-5) / 0.08
                                                     = 399.27100
    par bond     10 years, 5% semiannual, at 5%      price 100
                 Macaulay (1+j)/j (1 - (1+j)^-20)/2  = 7.989446 years (j = 2.5%)
                 modified Macaulay / 1.025           = 7.794581
                 DV01 modified x 100 x 1e-4          = 0.0779458
    zero         7 years at 6% annual                Macaulay 7,
                                                     convexity 7 x 8 / 1.06^2
                                                     = 49.83980
    XIRR         Excel's documented example          0.373362535, XNPV(9%)
                                                     2086.65 (ACT/365F, annual)
    MIRR         Excel's documented example          12.6094% at 10% / 12%
    Lorie-Savage -1,600 / +10,000 / -10,000          IRRs 25% and 400%, so no
                                                     IRR is reported
    no root      every flow positive                 no IRR

Beyond them: the roots are checked against numpy's companion-matrix roots
on random periodic flows and a brute-force grid on dated ones, Norstrom's
condition against the roots above 0%, the durations against finite
differences of the NPV, and the compounding and day-count conversions
against their identities.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import (
    analyze_cash_flows,
    dated_cash_flows,
    periodic_cash_flows,
    rate_roots,
)
from standard_quant_tools.finance.cash_flows import (
    MAX_EXACT_SIGN_CHANGES,
    cumulative_sign_changes,
    log_rate,
)

#: Excel's XIRR / XNPV documentation example.
XIRR_FLOWS = {
    "2008-01-01": -10000.0,
    "2008-03-01": 2750.0,
    "2008-10-30": 4250.0,
    "2009-02-15": 3250.0,
    "2009-04-01": 2750.0,
}
#: Excel's MIRR documentation example: a loan of 120,000 and five years
#: of income, financed at 10% and reinvested at 12%.
MIRR_FLOWS = [-120000.0, 39000.0, 30000.0, 21000.0, 37000.0, 46000.0]
#: Lorie and Savage's pump: an IRR of 25% and one of 400%.
PUMP = [-1600.0, 10000.0, -10000.0]


@pytest.fixture(scope="module")
def finance():
    return resolve("finance")


def _par_bond():
    return [0.0] + [2.5] * 19 + [102.5]


def _numpy_roots(flows):
    """The positive real roots of sum c_k v^k by the companion matrix, as
    continuously compounded rates -ln v: an oracle that shares nothing
    with the library's Rolle isolation."""
    roots = np.roots(np.asarray(flows, dtype=float)[::-1])
    real = [v.real for v in roots if abs(v.imag) <= 1e-9 * abs(v) and v.real > 0.0]
    return sorted(-math.log(v) for v in real)


class TestPlantedAnswers:
    def test_an_annuity_is_worth_399_27100(self, finance):
        """Five payments of 100 at 8%, the first one period out, through the
        tool: the closed form C (1 - (1 + r)^-n) / r to 1e-12."""
        out = finance.dispatch(
            "analyze_cash_flows", {"flows": [0.0] + [100.0] * 5, "rate": 0.08}
        )
        closed = 100.0 * (1.0 - 1.08**-5) / 0.08
        assert out["npv"] == pytest.approx(closed, rel=1e-12)
        assert round(out["npv"], 5) == 399.27100

    def test_a_par_bond_has_the_closed_form_duration_and_dv01(self, finance):
        """A 10-year 5% semiannual bond at a 5% yield is priced at par, with
        Macaulay duration (1+j)/j (1 - (1+j)^-N) / 2 years, modified duration
        Macaulay / (1 + y/2), and DV01 modified x price x 1e-4."""
        out = finance.dispatch(
            "analyze_cash_flows",
            {"flows": _par_bond(), "periods_per_year": 2, "rate": 0.05},
        )
        j, n = 0.025, 20
        macaulay = (1 + j) / j * (1 - (1 + j) ** -n) / 2
        assert out["npv"] == pytest.approx(100.0, rel=1e-13)
        assert out["macaulay_duration"] == pytest.approx(macaulay, rel=1e-12)
        assert round(out["macaulay_duration"], 6) == 7.989446
        assert round(out["modified_duration"], 6) == 7.794581
        assert round(out["dv01"], 7) == 0.0779458

    def test_the_dv01_is_the_central_difference_of_the_price(self, finance):
        """DV01 against the price moved 1 bp each way and differenced: the
        analytic number is the derivative, so the two agree to the
        convexity term, about 1e-8 here."""
        price = {}
        for y in (0.0499, 0.0501):
            price[y] = finance.dispatch(
                "analyze_cash_flows",
                {"flows": _par_bond(), "periods_per_year": 2, "rate": y},
            )["npv"]
        out = finance.dispatch(
            "analyze_cash_flows",
            {"flows": _par_bond(), "periods_per_year": 2, "rate": 0.05},
        )
        assert out["dv01"] == pytest.approx(
            (price[0.0499] - price[0.0501]) / 2, rel=1e-6
        )

    def test_a_seven_year_zero_has_convexity_49_83980(self, finance):
        """A zero's Macaulay duration is its maturity, and its convexity
        T (T + 1) / (1 + y)^2 under annual compounding."""
        out = finance.dispatch(
            "analyze_cash_flows", {"flows": [0.0] * 7 + [100.0], "rate": 0.06}
        )
        assert out["macaulay_duration"] == pytest.approx(7.0, rel=1e-14)
        assert out["convexity"] == pytest.approx(7 * 8 / 1.06**2, rel=1e-12)
        assert round(out["convexity"], 5) == 49.83980

    def test_the_documented_xirr_and_xnpv(self, finance):
        """Excel's documented example: XIRR 0.373362535 and XNPV at 9% of
        2086.65, with time in days / 365 from the first date and annual
        compounding -- ACT/365F."""
        out = finance.dispatch(
            "analyze_cash_flows",
            {"dated_flows": XIRR_FLOWS, "day_count": "ACT/365F", "rate": 0.09},
        )
        assert out["irr_status"] == "unique"
        assert out["irr"] == pytest.approx(0.373362535, abs=2e-9)
        assert round(out["npv"], 2) == 2086.65
        assert out["norstrom_holds"] is True
        assert out["compounding"] == "annual"
        assert out["first_date"] == "2008-01-01"

    def test_the_documented_mirr_is_12_6094_percent(self, finance):
        """Excel's MIRR example: the negative flows discounted at 10%, the
        positive compounded to year 5 at 12%, ((FV / -PV)^(1/5) - 1)."""
        out = finance.dispatch(
            "analyze_cash_flows",
            {"flows": MIRR_FLOWS, "finance_rate": 0.10, "reinvest_rate": 0.12},
        )
        fv = sum(c * 1.12 ** (5 - t) for t, c in enumerate(MIRR_FLOWS) if c > 0)
        pv = sum(c / 1.10**t for t, c in enumerate(MIRR_FLOWS) if c < 0)
        assert out["mirr"] == pytest.approx((fv / -pv) ** 0.2 - 1, rel=1e-12)
        assert round(out["mirr"], 6) == 0.126094

    def test_the_lorie_savage_pump_has_two_irrs_and_no_irr(self, finance):
        """Two rates zero the NPV, 25% and 400%; neither is the IRR, so it
        is null with both listed and the reason given."""
        out = finance.dispatch("analyze_cash_flows", {"flows": PUMP})
        assert out["irr"] is None
        assert out["irr_status"] == "multiple"
        assert out["irr_roots"] == pytest.approx([0.25, 4.0], rel=1e-12)
        assert out["sign_changes"] == 2
        assert out["norstrom_holds"] is False
        assert out["root_search"] == "exact"
        assert any(w.startswith("irr is null") for w in out["warnings"])

    def test_an_all_positive_stream_has_no_irr(self, finance):
        """No sign change, so by Descartes no root: the IRR is null and no
        search was needed."""
        out = finance.dispatch("analyze_cash_flows", {"flows": [1.0, 2.0, 3.0]})
        assert out["irr"] is None
        assert out["irr_status"] == "none"
        assert out["irr_roots"] == []
        assert out["root_search"] == "none"

    def test_a_stream_whose_roots_are_complex_has_no_irr(self, finance):
        """100 - 300 v + 250 v^2 has a negative discriminant: two sign
        changes and no real root."""
        out = finance.dispatch("analyze_cash_flows", {"flows": [100.0, -300.0, 250.0]})
        assert out["irr"] is None
        assert out["irr_status"] == "none"
        assert out["sign_changes"] == 2
        assert out["irr_roots"] == []

    def test_a_double_root_is_one_irr(self):
        """(1 - v)^2 touches zero at v = 1 without crossing: the critical
        point is the root, found once, at 0%."""
        out = analyze_cash_flows(periodic_cash_flows([1.0, -2.0, 1.0]))
        assert out["irr_status"] == "unique"
        assert out["irr"] == pytest.approx(0.0, abs=1e-12)


class TestEveryRootIsFound:
    def test_random_periodic_flows_match_the_companion_matrix(self):
        """300 random streams of 2 to 12 flows: the same count of real
        roots as numpy's companion-matrix eigenvalues, at the same rates."""
        rng = np.random.default_rng(20261023)
        for _ in range(300):
            flows = rng.normal(size=int(rng.integers(2, 13)))
            found = rate_roots(np.arange(flows.size, dtype=float), flows)
            assert found.method in ("exact", "none")
            oracle = _numpy_roots(flows)
            assert len(found.log_rates) == len(oracle), flows
            assert np.allclose(found.log_rates, oracle, rtol=1e-7, atol=1e-9)

    def test_dated_flows_match_a_brute_force_grid(self):
        """Irregular dates have no polynomial to factor, so the oracle is
        brute force: the NPV on 400,001 rates and every sign change between
        neighbours, on streams with a known root count."""
        rng = np.random.default_rng(7)
        start = dt.date(2020, 1, 1)
        for _ in range(40):
            n = int(rng.integers(3, 7))
            days = np.sort(rng.choice(np.arange(1, 1500), n - 1, replace=False))
            amounts = rng.normal(size=n)
            flows = {start.isoformat(): float(amounts[0])}
            for d, a in zip(days, amounts[1:]):
                flows[(start + dt.timedelta(days=int(d))).isoformat()] = float(a)
            schedule = dated_cash_flows(flows, day_count="ACT/365F")
            t = np.asarray(schedule.times)
            c = np.asarray(schedule.amounts)
            found = rate_roots(t, c)
            u = np.linspace(-3.0, 3.0, 400_001)
            npv = (c * np.exp(-np.outer(u, t))).sum(axis=1)
            crossings = u[:-1][np.sign(npv[:-1]) != np.sign(npv[1:])]
            inside = [r for r in found.log_rates if -3.0 < r < 3.0]
            assert len(inside) == len(crossings), flows
            assert np.allclose(inside, crossings, atol=2e-5)

    def test_past_twenty_sign_changes_the_search_is_a_grid_and_says_so(self, finance):
        """(v - 0.9) times 1 - v + v^2 - ... + v^24 alternates in sign 25
        times and has one positive root, v = 0.9. The grid finds it -- an
        IRR of 1/0.9 - 1 -- and does not certify it unique."""
        alternating = np.array([(-1.0) ** k for k in range(25)])
        flows = np.convolve(alternating, [-0.9, 1.0])
        out = finance.dispatch("analyze_cash_flows", {"flows": list(flows)})
        assert out["sign_changes"] > MAX_EXACT_SIGN_CHANGES
        assert out["root_search"] == "grid"
        assert out["irr_status"] == "uncertified"
        assert out["irr"] is None
        assert out["irr_roots"] == pytest.approx([1 / 0.9 - 1], rel=1e-10)


class TestNorstrom:
    def test_one_cumulative_sign_change_means_one_irr_above_zero(self):
        """Norstrom's sufficient condition, tested on 500 random streams
        that satisfy it: exactly one root above 0%, whatever else there is."""
        rng = np.random.default_rng(11)
        checked = 0
        while checked < 500:
            flows = rng.normal(size=int(rng.integers(3, 10)))
            _changes, holds = cumulative_sign_changes(list(flows))
            if not holds:
                continue
            checked += 1
            roots = rate_roots(np.arange(flows.size, dtype=float), flows)
            assert sum(1 for u in roots.log_rates if u > 0.0) == 1, flows

    def test_the_pump_fails_it(self):
        """-1,600 / 8,400 / -1,600 changes sign twice."""
        assert cumulative_sign_changes(PUMP) == (2, False)


class TestConventions:
    def test_the_irr_converts_between_compoundings(self, finance):
        """One stream, three quotes of one rate: (1 + y/2)^2 = 1 + y_annual
        = exp(y_continuous)."""
        flows = [-100.0, 3.0, 3.0, 3.0, 103.0]
        irr = {
            c: finance.dispatch(
                "analyze_cash_flows",
                {"flows": flows, "periods_per_year": 2, "compounding": c},
            )["irr"]
            for c in ("periodic", "annual", "continuous")
        }
        assert irr["periodic"] == pytest.approx(0.06, rel=1e-12)
        assert (1 + irr["periodic"] / 2) ** 2 == pytest.approx(
            1 + irr["annual"], rel=1e-13
        )
        assert math.exp(irr["continuous"]) == pytest.approx(
            1 + irr["annual"], rel=1e-13
        )

    def test_the_day_count_rescales_the_xirr(self, finance):
        """Under ACT/360 every time is 365/360 of its ACT/365F value, so
        ln(1 + r) scales by 360/365: the same flows, a different quote."""
        irr = {
            d: finance.dispatch(
                "analyze_cash_flows", {"dated_flows": XIRR_FLOWS, "day_count": d}
            )["irr"]
            for d in ("ACT/365F", "ACT/360")
        }
        assert math.log1p(irr["ACT/360"]) == pytest.approx(
            math.log1p(irr["ACT/365F"]) * 360 / 365, rel=1e-12
        )

    def test_the_npv_at_the_irr_is_zero(self, finance):
        """The IRR is a root of the NPV the tool also reports."""
        irr = finance.dispatch(
            "analyze_cash_flows", {"dated_flows": XIRR_FLOWS, "day_count": "ACT/365F"}
        )["irr"]
        npv = finance.dispatch(
            "analyze_cash_flows",
            {"dated_flows": XIRR_FLOWS, "day_count": "ACT/365F", "rate": irr},
        )["npv"]
        assert abs(npv) < 1e-9

    @pytest.mark.parametrize("compounding", ["periodic", "annual", "continuous"])
    def test_duration_and_convexity_are_the_derivatives(self, compounding):
        """-(1/P) dP/dy and (1/P) d2P/dy2 by central differences of the NPV,
        on dated flows under each compounding."""
        flows = {
            "2024-01-15": 4.0,
            "2024-09-02": 4.0,
            "2025-06-30": 4.0,
            "2026-03-31": 104.0,
        }
        y, h = 0.045, 1e-4

        def npv(rate):
            schedule = dated_cash_flows(
                flows, day_count="ACT/ACT", compounding=compounding, periods_per_year=2
            )
            return analyze_cash_flows(schedule, rate=rate)

        mid = npv(y)
        up, down = npv(y + h)["npv"], npv(y - h)["npv"]
        price = mid["npv"]
        assert mid["modified_duration"] == pytest.approx(
            -(up - down) / (2 * h * price), rel=1e-6
        )
        assert mid["convexity"] == pytest.approx(
            (up - 2 * price + down) / (h * h * price), rel=1e-4
        )

    def test_the_order_of_dated_keys_changes_no_bit(self, finance):
        """The flows are sorted by date and summed exactly, so a reversed
        mapping gives the same bits."""
        forward = finance.dispatch(
            "analyze_cash_flows",
            {"dated_flows": XIRR_FLOWS, "day_count": "ACT/365F", "rate": 0.09},
        )
        backward = finance.dispatch(
            "analyze_cash_flows",
            {
                "dated_flows": dict(reversed(list(XIRR_FLOWS.items()))),
                "day_count": "ACT/365F",
                "rate": 0.09,
            },
        )
        assert forward == backward

    def test_a_flow_at_time_zero_is_not_discounted(self, finance):
        """Flow 0 of a list is now: the NPV of [-100, 110] at 10% is 0,
        not Excel NPV's -9.09 (which discounts the first flow a period)."""
        out = finance.dispatch(
            "analyze_cash_flows", {"flows": [-100.0, 110.0], "rate": 0.10}
        )
        assert out["npv"] == pytest.approx(0.0, abs=1e-12)

    def test_without_a_rate_nothing_is_discounted_and_nothing_is_said(self, finance):
        """npv and the durations were not asked for: null, with no line
        each, while the IRR is still reported."""
        out = finance.dispatch("analyze_cash_flows", {"flows": [-100.0, 60.0, 60.0]})
        assert out["npv"] is None and out["dv01"] is None
        assert out["irr_status"] == "unique"
        assert not any("npv" in w or "dv01" in w for w in out["warnings"])

    def test_a_zero_present_value_has_no_duration_and_says_why(self, finance):
        """With the price among the flows the PV at the yield is zero, and
        a duration divides by it: null, and the reason names the price."""
        out = finance.dispatch(
            "analyze_cash_flows", {"flows": [-100.0, 110.0], "rate": 0.10}
        )
        assert out["macaulay_duration"] is None
        assert any("price is not one of its flows" in w for w in out["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "arguments, named",
        [
            ({}, "exactly one of flows / dated_flows"),
            (
                {"flows": [1.0], "dated_flows": {"2024-01-02": 1.0}},
                "exactly one of flows / dated_flows",
            ),
            ({"dated_flows": {"2024-01-02": -1.0, "2025-01-02": 2.0}}, "day_count"),
            ({"flows": [-1.0, 2.0], "day_count": "ACT/360"}, "day_count"),
            ({"flows": [-1.0, 2.0], "finance_rate": 0.1}, "reinvest_rate"),
            ({"flows": [-1.0, float("nan")]}, "not finite"),
            ({"flows": [0.0, 0.0]}, "every amount"),
            ({"flows": [-1e300, 1.0]}, "unit mistake"),
            ({"flows": [-1.0, 2.0], "rate": 53.0}, "rate"),
            (
                {
                    "dated_flows": {"2024-01-02": -1.0, "not-a-date": 2.0},
                    "day_count": "ACT/365F",
                },
                "not-a-date",
            ),
            (
                {
                    "dated_flows": {"2024-01-02": -1.0, "2024-01-02T10:00": 2.0},
                    "day_count": "ACT/365F",
                },
                "2024-01-02",
            ),
        ],
    )
    def test_a_bad_input_is_refused_by_name(self, finance, arguments, named):
        with pytest.raises((ValidationError, pydantic.ValidationError)) as exc:
            finance.dispatch("analyze_cash_flows", arguments)
        assert named in str(exc.value)

    def test_a_rate_at_minus_one_hundred_percent_has_no_discount_factor(self):
        with pytest.raises(ValidationError, match="-100% per compounding period"):
            log_rate(-1.0, "annual")
        with pytest.raises(ValidationError, match="no discount factor"):
            log_rate(-2.0, "periodic", 2)
        assert log_rate(-0.5, "continuous") == -0.5
