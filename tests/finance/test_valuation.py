"""
A discounted cash flow value, its reverse, and its sensitivity grid.

THE PLANTED FORECAST. Free cash flow 10 in year 1, growing at 3% a year,
discounted at 9%. A forecast of N years and a Gordon terminal value at the
same 3% add up to the perpetuity FCF_1 / (r - g) = 10 / 0.06 = 166.667
whatever N is, and the terminal value's share of it is ((1 + g) /
(1 + r))^N: the part of the perpetuity after year N.

THE REVERSE. The value at 9% is the target; solving for the discount rate
that gives it recovers 9% to 1e-10, inside the bracket it reports.
"""

from __future__ import annotations

import math

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import dcf_valuation

FCF1, RATE, GROWTH = 10.0, 0.09, 0.03


def _growing(n_years):
    return [FCF1 * (1 + GROWTH) ** t for t in range(n_years)]


@pytest.fixture(scope="module")
def finance():
    return resolve("finance")


class TestGordon:
    @pytest.mark.parametrize("n_years", [1, 7, 30])
    def test_constant_growth_is_the_perpetuity(self, finance, n_years):
        """FCF_1 / (r - g) = 166.667, and the terminal share the closed
        form ((1 + g) / (1 + r))^N."""
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": _growing(n_years),
                "discount_rate": RATE,
                "terminal_growth": GROWTH,
            },
        )
        assert out["value"] == pytest.approx(FCF1 / (RATE - GROWTH), rel=1e-13)
        assert round(out["value"], 3) == 166.667
        assert out["terminal_share"] == pytest.approx(
            ((1 + GROWTH) / (1 + RATE)) ** n_years, rel=1e-12
        )
        assert out["pv_forecast"] + out["pv_terminal"] == pytest.approx(out["value"])

    def test_the_implied_exit_multiple_is_terminal_value_over_the_last_flow(
        self, finance
    ):
        """(1 + g) / (r - g) = 17.1667 for a Gordon value at 9% and 3%."""
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": _growing(5),
                "discount_rate": RATE,
                "terminal_growth": GROWTH,
            },
        )
        assert out["implied_exit_multiple"] == pytest.approx(
            (1 + GROWTH) / (RATE - GROWTH), rel=1e-13
        )
        assert out["exit_multiple"] is None and out["implied_terminal_growth"] is None

    @pytest.mark.parametrize("growth", [0.09, 0.12])
    def test_growth_at_or_above_the_rate_is_refused(self, finance, growth):
        with pytest.raises(pydantic.ValidationError) as exc:
            finance.dispatch(
                "value_cash_flows",
                {
                    "cash_flows": [10.0],
                    "discount_rate": RATE,
                    "terminal_growth": growth,
                },
            )
        assert "terminal_growth" in str(exc.value) and "discount_rate" in str(exc.value)


class TestExitMultiple:
    def test_the_value_is_the_forecast_plus_the_discounted_multiple(self, finance):
        """Five flows and 12x a year-5 EBITDA of 30, at 10%, by hand; and
        the growth that multiple implies, put back into Gordon, gives the
        same terminal value."""
        flows = [5.0, 6.0, 7.0, 8.0, 9.0]
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": flows,
                "discount_rate": 0.10,
                "terminal_method": "exit_multiple",
                "exit_multiple": 12.0,
                "terminal_metric": 30.0,
            },
        )
        by_hand = sum(c / 1.1 ** (t + 1) for t, c in enumerate(flows))
        by_hand += 12.0 * 30.0 / 1.1**5
        assert out["value"] == pytest.approx(by_hand, rel=1e-13)
        g = out["implied_terminal_growth"]
        assert 9.0 * (1 + g) / (0.10 - g) == pytest.approx(360.0, rel=1e-12)
        assert out["terminal_growth"] is None and out["implied_exit_multiple"] is None


class TestReverse:
    def test_the_planted_rate_is_recovered_to_1e_10(self, finance):
        flows = _growing(7)
        target = finance.dispatch(
            "value_cash_flows",
            {"cash_flows": flows, "discount_rate": RATE, "terminal_growth": GROWTH},
        )["value"]
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": flows,
                "terminal_growth": GROWTH,
                "target_value": target,
                "solve_for": "discount_rate",
            },
        )
        assert abs(out["implied"] - RATE) < 1e-10
        low, high = out["implied_bracket"]
        assert low <= out["implied"] <= high
        assert out["search_range"][0] > GROWTH and out["search_range"][1] == 10.0
        assert out["discount_rate"] == out["implied"]
        assert out["value"] == pytest.approx(target, rel=1e-12)

    def test_the_planted_growth_is_recovered(self, finance):
        flows = _growing(7)
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": flows,
                "discount_rate": RATE,
                "target_value": FCF1 / (RATE - GROWTH),
                "solve_for": "terminal_growth",
            },
        )
        assert abs(out["implied"] - GROWTH) < 1e-10
        assert out["search_range"][1] < RATE

    def test_the_planted_multiple_is_recovered_in_closed_form(self, finance):
        flows = [5.0, 6.0, 7.0]
        common = {
            "cash_flows": flows,
            "discount_rate": 0.08,
            "terminal_method": "exit_multiple",
        }
        target = finance.dispatch(
            "value_cash_flows", {**common, "exit_multiple": 14.0}
        )["value"]
        out = finance.dispatch(
            "value_cash_flows",
            {**common, "target_value": target, "solve_for": "exit_multiple"},
        )
        assert out["implied"] == pytest.approx(14.0, rel=1e-13)
        assert out["implied_bracket"] is None

    def test_a_target_reached_twice_has_no_implied_rate(self, finance):
        """-100 now and 300 at year 5: the value falls below -40 and climbs
        back towards 0, so a target of -20 is reached at two rates (about
        43% and 400%). Both are listed and neither is chosen."""
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": [-100.0, 0.0, 0.0, 0.0, 0.0],
                "terminal_method": "exit_multiple",
                "exit_multiple": 10.0,
                "terminal_metric": 30.0,
                "target_value": -20.0,
                "solve_for": "discount_rate",
            },
        )
        assert out["implied"] is None
        assert len(out["implied_roots"]) == 2
        for r in out["implied_roots"]:
            x = 1 / (1 + r)
            assert -100 * x + 300 * x**5 == pytest.approx(-20.0, abs=1e-9)
        assert out["value"] is None

    def test_an_unreachable_target_is_null_with_the_range(self, finance):
        """Positive flows and a Gordon value never reach a negative target."""
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": [10.0, 10.0],
                "terminal_growth": 0.02,
                "target_value": -50.0,
                "solve_for": "discount_rate",
            },
        )
        assert out["implied"] is None and out["implied_roots"] == []
        assert any("outside what" in w for w in out["warnings"])


class TestSensitivity:
    def test_the_centre_is_the_value_and_the_grid_is_monotone(self, finance):
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": _growing(5),
                "discount_rate": RATE,
                "terminal_growth": GROWTH,
            },
        )
        grid = out["sensitivity"]
        assert grid["terminal_axis"] == "terminal_growth"
        values = grid["values"]
        assert values[2][2] == pytest.approx(out["value"], rel=1e-14)
        for row in values:
            assert all(a < b for a, b in zip(row, row[1:]))  # more growth, more
        for column in zip(*values):
            assert all(a > b for a, b in zip(column, column[1:]))  # higher rate, less

    def test_a_cell_with_growth_at_the_rate_is_null_with_the_reason(self, finance):
        out = finance.dispatch(
            "value_cash_flows",
            {
                "cash_flows": [10.0],
                "discount_rate": 0.05,
                "terminal_growth": 0.02,
                "sensitivity_rates": [0.02, 0.05],
                "sensitivity_terminal": [0.01, 0.03],
            },
        )
        values = out["sensitivity"]["values"]
        assert values[0] == [
            pytest.approx(10.0 * 1.01 / 0.01 / 1.02 + 10.0 / 1.02),
            None,
        ]
        assert all(v is not None for v in values[1])
        assert any("not below its rate" in w for w in out["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "change, named",
        [
            ({"terminal_growth": None}, "terminal_growth"),
            ({"discount_rate": None}, "discount_rate"),
            ({"exit_multiple": 10.0}, "exit_multiple"),
            ({"target_value": 100.0}, "solve_for"),
            ({"target_value": 100.0, "solve_for": "discount_rate"}, "discount_rate"),
            ({"target_value": 100.0, "solve_for": "exit_multiple"}, "exit_multiple"),
            ({"cash_flows": []}, "cash_flows"),
            ({"cash_flows": [1.0] * 201}, "cash_flows"),
            ({"cash_flows": [float("inf")]}, "cash_flows"),
        ],
    )
    def test_through_the_tool(self, finance, change, named):
        arguments = {
            "cash_flows": [10.0, 11.0],
            "discount_rate": 0.09,
            "terminal_growth": 0.02,
            **change,
        }
        arguments = {k: v for k, v in arguments.items() if v is not None}
        with pytest.raises((ValidationError, pydantic.ValidationError)) as exc:
            finance.dispatch("value_cash_flows", arguments)
        assert named in str(exc.value)

    def test_the_other_methods_growth_is_refused_in_the_library(self):
        with pytest.raises(ValidationError, match="belongs to"):
            dcf_valuation(
                [10.0],
                discount_rate=0.1,
                terminal_method="exit_multiple",
                exit_multiple=8.0,
                terminal_growth=0.02,
            )
        assert math.isfinite(
            dcf_valuation(
                [10.0],
                discount_rate=0.1,
                terminal_method="exit_multiple",
                exit_multiple=8.0,
            )["value"]
        )
