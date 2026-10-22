"""
`analyze_dividend_points` compares a present value with a present value.

What a quoted future implies, S - F exp(-rT), is the present value of the
dividends in the window. The forecast it was compared with was their
nominal sum, so a future priced EXACTLY off a forecast dividend reported a
gap -- 0.74 index points on a 60-point dividend 91 days out at 5% -- and
called it "a position" (see the CHANGELOG entry of 2026-10-22). Each
forecast dividend is now discounted to as_of at the financing rate, from
its pay date when one is given and its ex-date otherwise, and both totals
are reported.
"""

import math

import pytest

from standard_quant_tools.agent.runtimes.delta_one.models import DividendPointsInput
from standard_quant_tools.agent.runtimes.delta_one.tools import (
    analyze_dividend_points,
)
from standard_quant_tools.delta_one.dividends import dividend_points
from standard_quant_tools.error import ValidationError

SPOT, RATE = 6000.0, 0.05
AS_OF, EXPIRY = "2026-01-01", "2026-07-02"  # 182 days
T = 182 / 365.0
EX_DATE = "2026-04-02"  # 91 days in
T_EX = 91 / 365.0
POINTS = 60.0


def _constituent(**extra):
    return {
        "symbol": "AAA",
        "shares": POINTS,
        "dividend_per_share": 1.0,
        "ex_date": EX_DATE,
        **extra,
    }


def _priced_off(t_paid: float) -> float:
    """The exact discrete-dividend forward for one 60-point dividend whose
    cash arrives at `t_paid` years."""
    return (SPOT - POINTS * math.exp(-RATE * t_paid)) * math.exp(RATE * T)


def _run(constituents, future_price, **kw):
    return dividend_points(
        constituents,
        divisor=1.0,
        as_of=AS_OF,
        expiry=EXPIRY,
        spot=SPOT,
        future_price=future_price,
        financing_rate=RATE,
        time_to_expiry=T,
        **kw,
    )


class TestAFutureAtFairValueShowsNoGap:
    def test_the_planted_future_reports_zero_and_no_position(self):
        """Priced exactly off the forecast: the gap is zero, where it was
        0.743 points of pure discounting reported as a position."""
        out = _run([_constituent()], _priced_off(T_EX))
        assert out["forecast_minus_implied"] == pytest.approx(0.0, abs=1e-9)
        assert not any("It is a position" in w for w in out["warnings"])

    def test_both_totals_are_reported(self):
        out = _run([_constituent()], _priced_off(T_EX))
        assert out["total_index_points"] == pytest.approx(POINTS)
        assert out["forecast_pv_points"] == pytest.approx(
            POINTS * math.exp(-RATE * T_EX), rel=1e-12
        )
        assert out["implied_index_points"] == pytest.approx(
            out["forecast_pv_points"], abs=1e-9
        )
        (row,) = out["points_by_constituent"]
        assert row["discounted_to"] == "ex_date"
        assert row["discount_years"] == pytest.approx(T_EX)

    def test_a_real_disagreement_is_still_a_position(self):
        """One more point of dividend than the future prices is a gap of one
        point in present value, and still reported as one."""
        out = _run([_constituent(shares=POINTS + 1.0)], _priced_off(T_EX))
        assert out["forecast_minus_implied"] == pytest.approx(
            math.exp(-RATE * T_EX), rel=1e-9
        )
        assert any("It is a position" in w for w in out["warnings"])


class TestThePayDateIsWhenTheCashArrives:
    def test_a_pay_date_is_discounted_to_and_matches_a_future_priced_on_it(self):
        pay_date, t_paid = "2026-04-23", 112 / 365.0  # three weeks after ex
        out = _run([_constituent(pay_date=pay_date)], _priced_off(t_paid))
        assert out["forecast_minus_implied"] == pytest.approx(0.0, abs=1e-9)
        (row,) = out["points_by_constituent"]
        assert row["discounted_to"] == "pay_date"
        assert out["n_discounted_to_ex_date"] == 0
        assert not any("discounted to their ex-date" in w for w in out["warnings"])

    def test_discounting_to_the_ex_date_says_so(self):
        out = _run([_constituent()], _priced_off(T_EX))
        assert out["n_discounted_to_ex_date"] == 1
        assert any("discounted to their ex-date" in w for w in out["warnings"])

    def test_a_pay_date_before_the_ex_date_is_refused(self):
        with pytest.raises(ValidationError, match="pay_date 2026-03-01 precedes"):
            _run([_constituent(pay_date="2026-03-01")], _priced_off(T_EX))


class TestTheClocksAgree:
    def test_a_time_to_expiry_off_the_dates_is_said(self):
        out = dividend_points(
            [_constituent()],
            divisor=1.0,
            as_of=AS_OF,
            expiry=EXPIRY,
            spot=SPOT,
            future_price=_priced_off(T_EX),
            financing_rate=RATE,
            time_to_expiry=0.25,
        )
        assert any("the two clocks disagree" in w for w in out["warnings"])

    def test_without_a_rate_nothing_is_discounted_and_that_is_said(self):
        out = dividend_points([_constituent()], divisor=1.0, as_of=AS_OF, expiry=EXPIRY)
        assert out["forecast_pv_points"] is None
        assert out["total_index_points"] == pytest.approx(POINTS)
        assert any("not discounted" in w for w in out["warnings"])


class TestTheToolDoor:
    def test_the_tool_reports_the_present_value_and_a_zero_gap(self):
        out = analyze_dividend_points(
            DividendPointsInput(
                constituents=[_constituent()],
                divisor=1.0,
                as_of=AS_OF,
                expiry=EXPIRY,
                spot=SPOT,
                future_price=_priced_off(T_EX),
                financing_rate=RATE,
                time_to_expiry=T,
            )
        )
        assert out.forecast_pv_points == pytest.approx(
            POINTS * math.exp(-RATE * T_EX), rel=1e-12
        )
        assert out.total_index_points == pytest.approx(POINTS)
        assert out.forecast_minus_implied == pytest.approx(0.0, abs=1e-9)
        assert out.day_count == "ACT/365F"
        assert not any("It is a position" in w for w in out.warnings)
