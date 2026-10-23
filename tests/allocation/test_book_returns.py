"""
A portfolio's return with deposits and withdrawals: the true time-weighted
return, linked Modified Dietz and the money-weighted return (GIPS 2020).

THE DEPOSIT AT THE TOP, by hand. 100 grows 20% in a year to 120; the owner
adds 120, and the 240 falls 10% in the next year to 216.

    time-weighted    1.2 x 0.9 - 1                              = +8.000%
    money-weighted   100 (1 + r)^2 + 120 (1 + r) = 216, r       = -1.2549% a year

The manager earned 8% over the two years; the owner, who put most of the
money in at the top, lost.

MODIFIED DIETZ, by hand. 100 at the start of a 30-day month, 115 at its
end, and 10 deposited on day 15, weighted (30 - 15) / 30 = 0.5:

    (115 - 100 - 10) / (100 + 0.5 x 10)                         = 4.7619%

With no valuation on the deposit's date there is no true TWR, and the
month is under a year, so nothing is annualized.
"""

from __future__ import annotations

import math

import pydantic
import pytest
from scipy.optimize import brentq

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.allocation import measure_book_returns
from standard_quant_tools.error import ValidationError

DEPOSIT_AT_THE_TOP = {
    "valuations": {"2021-01-01": 100.0, "2022-01-01": 120.0, "2023-01-01": 216.0},
    "flows": {"2022-01-01": 120.0},
}
DIETZ_MONTH = {
    "valuations": {"2021-01-01": 100.0, "2021-01-31": 115.0},
    "flows": {"2021-01-16": 10.0},
}


@pytest.fixture(scope="module")
def allocation():
    return resolve("allocation")


class TestThePlantedBooks:
    def test_the_deposit_at_the_top_is_plus_8_and_minus_1_2549(self, allocation):
        """The two answers of the module docstring: +8.000% time-weighted,
        -1.2549% a year money-weighted (the quadratic's root), and Modified
        Dietz equal to the true TWR because the flow has a valuation."""
        out = allocation.dispatch("measure_book_returns", DEPOSIT_AT_THE_TOP)
        assert out["time_weighted_return"] == pytest.approx(0.08, rel=1e-13)
        assert round(out["time_weighted_return"], 5) == 0.08000
        growth = (-120 + math.sqrt(120**2 + 4 * 100 * 216)) / 200
        assert out["money_weighted_annualized"] == pytest.approx(growth - 1, rel=1e-12)
        assert round(out["money_weighted_annualized"], 6) == -0.012549
        assert out["money_weighted_return"] == pytest.approx(growth**2 - 1, rel=1e-12)
        assert out["modified_dietz_return"] == pytest.approx(0.08, rel=1e-13)
        assert out["time_weighted_annualized"] == pytest.approx(
            math.sqrt(1.08) - 1, rel=1e-13
        )
        assert out["timing_gap"] == pytest.approx(growth**2 - 1 - 0.08, rel=1e-12)
        assert out["years"] == 2.0 and out["mwr_status"] == "unique"

    def test_modified_dietz_is_4_7619_and_there_is_no_true_twr(self, allocation):
        """5 / 105, the deposit's date named as missing a valuation, and no
        annualized figure for a 30-day period."""
        out = allocation.dispatch("measure_book_returns", DIETZ_MONTH)
        assert out["modified_dietz_return"] == pytest.approx(5 / 105, rel=1e-13)
        assert round(out["modified_dietz_return"], 6) == 0.047619
        assert out["time_weighted_return"] is None
        assert out["missing_valuation_dates"] == ["2021-01-16"]
        assert any("missing_valuation_dates" in w for w in out["warnings"])
        for field in (
            "time_weighted_annualized",
            "modified_dietz_annualized",
            "money_weighted_annualized",
        ):
            assert out[field] is None
        assert any("under one" in w for w in out["warnings"])

    def test_the_month_s_mwr_is_the_day_weighted_irr(self, allocation):
        """The owner's cash -- -100 on day 0, -10 on day 15, +115 on day 30
        -- solved for the month's return by Brent's method here, outside
        the library."""
        out = allocation.dispatch("measure_book_returns", DIETZ_MONTH)

        def npv(period_return):
            g = 1.0 + period_return
            return -100.0 - 10.0 * g ** (-15 / 30) + 115.0 / g

        assert out["money_weighted_return"] == pytest.approx(
            brentq(npv, -0.5, 1.0, xtol=1e-15), rel=1e-10
        )

    def test_with_no_flows_the_three_agree(self, allocation):
        """Nothing moved in or out, so every method is V_end / V_0 - 1."""
        out = allocation.dispatch(
            "measure_book_returns",
            {
                "valuations": {
                    "2020-01-02": 50.0,
                    "2020-06-30": 47.0,
                    "2021-03-31": 58.0,
                }
            },
        )
        expected = 58.0 / 50.0 - 1
        for field in (
            "time_weighted_return",
            "modified_dietz_return",
            "money_weighted_return",
        ):
            assert out[field] == pytest.approx(expected, rel=1e-12), field
        assert out["n_flows"] == 0 and out["large_flows"] == []


class TestConventions:
    def test_a_valuation_that_includes_its_flow_gives_the_same_returns(
        self, allocation
    ):
        """The deposit-at-the-top book written the other way: the 2022
        valuation already holds the deposit, so it reads 240 and the flow
        is inside the first year. Same TWR, same MWR."""
        before = allocation.dispatch("measure_book_returns", DEPOSIT_AT_THE_TOP)
        after = allocation.dispatch(
            "measure_book_returns",
            {
                "valuations": {
                    "2021-01-01": 100.0,
                    "2022-01-01": 240.0,
                    "2023-01-01": 216.0,
                },
                "flows": {"2022-01-01": 120.0},
                "flows_in_valuation": True,
            },
        )
        for field in ("time_weighted_return", "money_weighted_annualized"):
            assert after[field] == pytest.approx(before[field], rel=1e-13), field

    def test_the_day_count_moves_the_annual_rate_not_the_period_return(
        self, allocation
    ):
        """Over the same days, ACT/360 counts more years than ACT/365F: the
        MWR over the period is the same, its annual rate is not."""
        book = {
            "valuations": {"2020-01-02": 100.0, "2021-06-30": 112.0},
            "flows": {"2020-09-01": 5.0},
        }
        a = allocation.dispatch(
            "measure_book_returns", {**book, "day_count": "ACT/365F"}
        )
        b = allocation.dispatch(
            "measure_book_returns", {**book, "day_count": "ACT/360"}
        )
        assert a["money_weighted_return"] == pytest.approx(
            b["money_weighted_return"], rel=1e-12
        )
        assert math.log1p(b["money_weighted_annualized"]) == pytest.approx(
            math.log1p(a["money_weighted_annualized"]) * 360 / 365, rel=1e-12
        )

    def test_a_pump_shaped_book_has_two_mwrs_and_no_mwr(self, allocation):
        """1,600 in, 10,000 out a year later, 10,000 back in the year after
        and nothing left: the owner's cash is Lorie and Savage's pump, with
        annual roots 25% and 400%, so the MWR is null with both listed."""
        out = allocation.dispatch(
            "measure_book_returns",
            {
                "valuations": {
                    "2021-01-01": 1600.0,
                    "2022-01-01": 10000.0,
                    "2023-01-01": 0.0,
                    "2023-07-01": 0.0,
                },
                "flows": {"2022-01-01": -10000.0, "2023-01-01": 10000.0},
            },
        )
        assert out["money_weighted_return"] is None
        assert out["mwr_status"] == "multiple"
        years = out["years"]
        assert out["mwr_roots"] == pytest.approx(
            [1.25**years - 1, 5.0**years - 1], rel=1e-10
        )

    def test_changing_a_later_valuation_changes_no_earlier_period(self, allocation):
        """Each sub-period's returns use only the valuations and flows up to
        its end, so moving the last value leaves every earlier row alone."""
        book = {
            "valuations": {
                "2020-01-02": 100.0,
                "2020-04-01": 104.0,
                "2020-07-01": 98.0,
                "2020-10-01": 110.0,
            },
            "flows": {"2020-04-01": 10.0},
        }
        base = allocation.dispatch("measure_book_returns", book)
        moved = allocation.dispatch(
            "measure_book_returns",
            {**book, "valuations": {**book["valuations"], "2020-10-01": 80.0}},
        )
        assert moved["periods"][:2] == base["periods"][:2]
        assert moved["periods"][2] != base["periods"][2]


class TestLargeFlows:
    def test_a_flow_above_ten_percent_is_flagged(self, allocation):
        book = {
            "valuations": {
                "2020-01-02": 100.0,
                "2020-02-03": 101.0,
                "2020-03-02": 120.0,
            },
            "flows": {"2020-02-03": 15.0, "2020-02-17": 5.0},
        }
        out = allocation.dispatch("measure_book_returns", book)
        assert [row["date"] for row in out["large_flows"]] == ["2020-02-03"]
        assert out["large_flows"][0]["share_of_value"] == pytest.approx(15 / 101)
        assert out["large_flows"][0]["valued"] is True
        relaxed = allocation.dispatch(
            "measure_book_returns", {**book, "large_flow_share": 0.2}
        )
        assert relaxed["large_flows"] == []

    def test_an_unvalued_large_flow_says_the_dietz_figure_is_approximate(
        self, allocation
    ):
        out = allocation.dispatch(
            "measure_book_returns",
            {
                "valuations": {"2020-01-02": 100.0, "2020-03-02": 140.0},
                "flows": {"2020-02-03": 30.0},
            },
        )
        assert out["large_flows"][0]["valued"] is False
        assert any("approximation" in w for w in out["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "arguments, named",
        [
            ({"valuations": {"2020-01-02": 100.0}}, "valuations"),
            (
                {"valuations": {"2020-01-02": 100.0, "2020-02-03": -1.0}},
                "negative",
            ),
            (
                {
                    "valuations": {"2020-01-02": 100.0, "2020-02-03": 101.0},
                    "flows": {"2020-02-03": 5.0},
                },
                "2020-02-03",
            ),
            (
                {
                    "valuations": {"2020-01-02": 100.0, "2020-02-03": 101.0},
                    "flows": {"2019-12-31": 5.0},
                },
                "2019-12-31",
            ),
            (
                {
                    "valuations": {"2020-01-02": 100.0, "2020-02-03": 101.0},
                    "flows": {"2020-01-02": 5.0},
                    "flows_in_valuation": True,
                },
                "2020-01-02",
            ),
            (
                {"valuations": {"2020-01-02": 100.0, "AAPL": 101.0}},
                "AAPL",
            ),
            (
                {
                    "valuations": {"2020-01-02": 100.0, "2020-02-03": 101.0},
                    "large_flow_share": 0.0,
                },
                "large_flow_share",
            ),
        ],
    )
    def test_a_bad_input_is_refused_by_name(self, allocation, arguments, named):
        with pytest.raises((ValidationError, pydantic.ValidationError)) as exc:
            allocation.dispatch("measure_book_returns", arguments)
        assert named in str(exc.value)

    def test_the_library_refuses_two_keys_on_one_date(self):
        with pytest.raises(ValidationError, match="2020-01-02"):
            measure_book_returns(
                {"2020-01-02": 100.0, "2020-01-02T12:00": 100.5, "2020-03-02": 101.0}
            )
