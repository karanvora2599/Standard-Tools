"""
Brinson-Fachler attribution, held to a hand-worked period and to the
linking identity.

One period: a portfolio 60/40 in two segments returning 5% and 2% against
a 50/50 benchmark returning 4% and 2% is ahead by 0.008, and by hand that
is allocation 0.002, selection 0.005 and interaction 0.001 -- each segment's
too. Over many periods, Carino's and Menchero's coefficients each make the
linked effects sum to the cumulative active return, prod(1 + R_p) -
prod(1 + R_b), to 1e-12 on random books; each coefficient is recomputed
here from its published formula. Weights that do not sum to one are
refused with their sum.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.allocation.attribution import brinson_fachler
from standard_quant_tools.error import ValidationError

PLANTED = {
    "portfolio_weights": {"2024-03-31": {"A": 0.6, "B": 0.4}},
    "benchmark_weights": {"2024-03-31": {"A": 0.5, "B": 0.5}},
    "portfolio_returns": {"2024-03-31": {"A": 0.05, "B": 0.02}},
    "benchmark_returns": {"2024-03-31": {"A": 0.04, "B": 0.02}},
}


def _call(arguments):
    return resolve("allocation").dispatch("attribute_returns_brinson", arguments)


def _random_book(periods, segments, seed):
    rng = np.random.default_rng(seed)
    out = {k: {} for k in PLANTED}
    for t in range(periods):
        label = f"2023-{1 + t // 28:02d}-{1 + t % 28:02d}"
        wp = rng.dirichlet(np.ones(segments))
        wb = rng.dirichlet(np.ones(segments))
        names = [f"S{i}" for i in range(segments)]
        out["portfolio_weights"][label] = dict(zip(names, map(float, wp)))
        out["benchmark_weights"][label] = dict(zip(names, map(float, wb)))
        out["portfolio_returns"][label] = dict(
            zip(names, map(float, rng.normal(0.01, 0.05, segments)))
        )
        out["benchmark_returns"][label] = dict(
            zip(names, map(float, rng.normal(0.008, 0.04, segments)))
        )
    return out


class TestOnePeriodByHand:
    def test_the_planted_total_is_eight_basis_points_split_by_hand(self):
        out = _call(PLANTED)
        assert out["active_return"] == pytest.approx(0.008, abs=1e-15)
        assert out["portfolio_return"] == pytest.approx(0.038, abs=1e-15)
        assert out["benchmark_return"] == pytest.approx(0.030, abs=1e-15)
        assert out["allocation"] == pytest.approx(0.002, abs=1e-15)
        assert out["selection"] == pytest.approx(0.005, abs=1e-15)
        assert out["interaction"] == pytest.approx(0.001, abs=1e-15)
        a, b = out["by_segment"]["A"], out["by_segment"]["B"]
        # A: (0.6-0.5)(0.04-0.03), 0.5(0.05-0.04), (0.1)(0.01)
        assert (a["allocation"], a["selection"], a["interaction"]) == pytest.approx(
            (0.001, 0.005, 0.001), abs=1e-15
        )
        # B: (0.4-0.5)(0.02-0.03), 0.5(0), (-0.1)(0)
        assert (b["allocation"], b["selection"], b["interaction"]) == pytest.approx(
            (0.001, 0.0, 0.0), abs=1e-15
        )
        assert b["interaction"] == 0.0 and math.copysign(1, b["interaction"]) == 1
        assert out["by_period"][0]["linking_coefficient"] == pytest.approx(1.0)

    def test_the_same_book_on_both_sides_has_no_effect(self):
        same = {
            "portfolio_weights": PLANTED["benchmark_weights"],
            "benchmark_weights": PLANTED["benchmark_weights"],
            "portfolio_returns": PLANTED["benchmark_returns"],
            "benchmark_returns": PLANTED["benchmark_returns"],
        }
        out = _call(same)
        assert out["active_return"] == 0.0
        assert out["allocation"] == out["selection"] == out["interaction"] == 0.0


class TestLinking:
    @pytest.mark.parametrize("linking", ["carino", "menchero"])
    def test_linked_effects_sum_to_the_cumulative_active_return(self, linking):
        """Twelve periods of five segments: the linked allocation, selection
        and interaction add up to prod(1 + R_p) - prod(1 + R_b) to 1e-12,
        and so does every segment's sum with every other's."""
        book = _random_book(12, 5, seed=1)
        out = _call({**book, "linking": linking})
        rp = [r["portfolio_return"] for r in out["by_period"]]
        rb = [r["benchmark_return"] for r in out["by_period"]]
        cumulative = math.prod(1 + x for x in rp) - math.prod(1 + x for x in rb)
        assert out["active_return"] == pytest.approx(cumulative, abs=1e-15)
        assert abs(out["sum_of_effects"] - cumulative) < 1e-12
        total = sum(e["total"] for e in out["by_segment"].values())
        assert abs(total - cumulative) < 1e-12
        # The unlinked single-period effects do NOT add up to it.
        assert (
            abs(sum(r["active_return"] for r in out["by_period"]) - cumulative) > 1e-5
        )

    def test_carino_coefficients_are_the_log_ratios(self):
        book = _random_book(3, 4, seed=2)
        out = _call({**book, "linking": "carino"})
        rp = [r["portfolio_return"] for r in out["by_period"]]
        rb = [r["benchmark_return"] for r in out["by_period"]]
        RP = math.prod(1 + x for x in rp) - 1
        RB = math.prod(1 + x for x in rb) - 1
        K = (math.log(1 + RP) - math.log(1 + RB)) / (RP - RB)
        for row, p, b in zip(out["by_period"], rp, rb):
            k = (math.log(1 + p) - math.log(1 + b)) / (p - b)
            assert row["linking_coefficient"] == pytest.approx(k / K, rel=1e-10)

    def test_menchero_coefficients_are_m_plus_the_correction(self):
        book = _random_book(4, 3, seed=3)
        out = _call({**book, "linking": "menchero"})
        rp = [r["portfolio_return"] for r in out["by_period"]]
        rb = [r["benchmark_return"] for r in out["by_period"]]
        T = len(rp)
        RP = math.prod(1 + x for x in rp) - 1
        RB = math.prod(1 + x for x in rb) - 1
        M = ((RP - RB) / T) / ((1 + RP) ** (1 / T) - (1 + RB) ** (1 / T))
        a = [p - b for p, b in zip(rp, rb)]
        scale = ((RP - RB) - M * sum(a)) / sum(x * x for x in a)
        for row, x in zip(out["by_period"], a):
            assert row["linking_coefficient"] == pytest.approx(M + scale * x, rel=1e-12)

    def test_the_periods_are_linked_in_date_order_whatever_the_key_order(self):
        book = _random_book(6, 3, seed=4)
        labels = list(book["benchmark_weights"])
        random.Random(5).shuffle(labels)
        shuffled = {k: {p: v[p] for p in labels} for k, v in book.items()}
        one, two = _call(book), _call(shuffled)
        assert [r["period"] for r in two["by_period"]] == sorted(labels)
        assert one["by_segment"] == two["by_segment"]


class TestRefusals:
    def test_weights_that_do_not_sum_to_one_are_refused_with_the_sum(self):
        bad = {**PLANTED, "portfolio_weights": {"2024-03-31": {"A": 0.6, "B": 0.37}}}
        with pytest.raises(ValidationError, match="sum to 0.97"):
            _call(bad)

    def test_a_held_segment_needs_a_benchmark_return(self):
        bad = {**PLANTED, "benchmark_returns": {"2024-03-31": {"A": 0.04}}}
        with pytest.raises(ValidationError, match=r"benchmark_returns lacks \['B'\]"):
            _call(bad)

    def test_a_segment_the_portfolio_does_not_hold_needs_no_portfolio_return(self):
        book = {
            "portfolio_weights": {"2024-03-31": {"A": 1.0}},
            "benchmark_weights": {"2024-03-31": {"A": 0.5, "B": 0.5}},
            "portfolio_returns": {"2024-03-31": {"A": 0.05}},
            "benchmark_returns": {"2024-03-31": {"A": 0.04, "B": 0.02}},
        }
        out = _call(book)
        assert out["by_segment"]["B"]["selection"] == 0.0
        assert out["by_segment"]["B"]["allocation"] == pytest.approx(-0.5 * -0.01)
        assert out["active_return"] == pytest.approx(0.05 - 0.03, abs=1e-15)
        assert any("held none of" in w for w in out["warnings"])

    def test_periods_must_match_across_the_maps(self):
        bad = {
            **PLANTED,
            "benchmark_returns": {
                "2024-03-31": {"A": 0.04, "B": 0.02},
                "2024-06-30": {"A": 0.0, "B": 0.0},
            },
        }
        with pytest.raises(ValidationError, match="same periods"):
            _call(bad)

    def test_a_period_that_is_not_a_date_is_refused(self):
        bad = {k: {"Q1": v["2024-03-31"]} for k, v in PLANTED.items()}
        with pytest.raises(ValidationError, match="not ISO dates"):
            _call(bad)

    @pytest.mark.parametrize(
        "field,value,needle",
        [
            ("portfolio_returns", {"2024-03-31": {"A": float("nan")}}, "not finite"),
            ("linking", "geometric", "linking"),
            ("benchmark_weights", {}, "empty"),
        ],
    )
    def test_the_input_model_refuses_by_name(self, field, value, needle):
        with pytest.raises(PydanticValidationError, match=needle):
            _call({**PLANTED, field: value})

    def test_the_library_refuses_an_unknown_linking(self):
        with pytest.raises(ValidationError, match="linking"):
            brinson_fachler(["p"], {}, {}, {}, {}, linking="geometric")
