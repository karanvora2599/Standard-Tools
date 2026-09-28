"""
A ratio channel refuses a leg too small to be a price, and the monitor
applies the price ceiling itself.

The monitor's tool door bounds every tick by MAGNITUDE, from above only,
because `absolute_points` takes a leg below zero. That left the ratio
channels open underneath: a reference of 1e-300 under a primary at the
ceiling made `relative_bps` infinite, and the warm-up's mean and variance
went to inf and NaN -- a monitor answering with no baseline instead of a
refusal. On `annualized_bps` a primary of 5e-324 underflowed the ratio to
0.0 and `log` raised a bare math domain error, and a time to expiry of
1e-310 years divided a finite log-ratio into infinity. A direct caller of
the library did not even meet the ceiling.

The ratio channels now refuse a leg below 1e-8 -- the floor the library
already puts on a spot or strike -- and `annualized_bps` a time to expiry
below its 1e-8-year minimum; every channel refuses a leg beyond the price
ceiling. `absolute_points` still takes negative legs and legs below the
floor. See the CHANGELOG entry of 2026-09-28.
"""

from __future__ import annotations

import math

import pytest

from standard_quant_tools.analysis.derivatives import MAX_PRICE
from standard_quant_tools.delta_one.streaming import (
    new_spread_monitor,
    update_spread_monitor,
)
from standard_quant_tools.error import ValidationError

N = 12


def _feed(channel, primary, reference, time_to_expiry=None):
    state = new_spread_monitor(channel=channel, warmup=10)
    return update_spread_monitor(
        state, primary=primary, reference=reference, time_to_expiry=time_to_expiry
    )


def _drifting(level: float):
    return [level + 0.01 * i for i in range(N)]


class TestRatioChannelsRefuseALegTooSmallToBeAPrice:
    def test_a_tiny_reference_under_a_ceiling_primary_is_refused(self):
        """It used to overflow to an infinite spread and a NaN baseline."""
        with pytest.raises(ValidationError, match=r"reference=1e-300, below the 1e-08"):
            _feed("relative_bps", [MAX_PRICE] * N, [1e-300] * N)

    def test_one_tiny_reference_late_in_a_batch_is_refused_at_its_tick(self):
        with pytest.raises(ValidationError, match="observation 11 has reference"):
            _feed("relative_bps", _drifting(100.0), [100.0] * (N - 1) + [1e-300])

    def test_the_annualized_channel_refuses_a_tiny_reference(self):
        with pytest.raises(ValidationError, match="reference=1e-300"):
            _feed("annualized_bps", [100.0] * N, [1e-300] * N, [0.5] * N)

    def test_a_subnormal_primary_is_refused_not_a_math_domain_error(self):
        with pytest.raises(ValidationError, match="primary=5e-324"):
            _feed("annualized_bps", [5e-324] * N, [MAX_PRICE] * N, [0.5] * N)

    def test_a_time_to_expiry_below_the_minimum_is_refused(self):
        """1e-310 years divided a finite log-ratio into infinity."""
        with pytest.raises(ValidationError, match="time_to_expiry=1e-310 years"):
            _feed("annualized_bps", _drifting(101.0), [100.0] * N, [1e-310] * N)

    def test_legs_at_the_floor_and_the_ceiling_give_a_finite_baseline(self):
        """The edge null case: both bounds are inclusive, and the widest
        ratio they allow stays inside the float range."""
        primary = [MAX_PRICE * (1 - 1e-3 * i) for i in range(N)]
        result = _feed("relative_bps", primary, [1e-8] * N)
        assert math.isfinite(result["baseline_mean"])
        assert math.isfinite(result["baseline_std"])

    def test_an_ordinary_basis_is_monitored(self):
        """The null case."""
        result = _feed("relative_bps", _drifting(100.1), [100.0] * N)
        assert result["baseline_mean"] == pytest.approx(14.5, rel=1e-2)
        assert not result["triggered"]


class TestEveryChannelAppliesThePriceCeiling:
    @pytest.mark.parametrize(
        "channel, tte",
        [
            ("relative_bps", None),
            ("annualized_bps", [0.5] * N),
            ("absolute_points", None),
        ],
    )
    def test_a_leg_beyond_the_ceiling_is_refused(self, channel, tte):
        """Prices scaled by 1e300 overflowed the warm-up variance to
        infinity, on any channel; only the tool door refused them."""
        with pytest.raises(ValidationError, match=r"primary has 12 price\(s\) beyond"):
            _feed(channel, [1e300 * (i + 1) for i in range(N)], [1.0] * N, tte)

    def test_a_negative_leg_beyond_the_ceiling_is_refused_on_magnitude(self):
        with pytest.raises(ValidationError, match="reference has 1 price"):
            _feed("absolute_points", _drifting(1.0), [0.0] * (N - 1) + [-1.5e12])


class TestAbsolutePointsStillTakesAnyLegAPriceCanBe:
    def test_negative_legs_are_a_spread_in_points(self):
        """The null case: a roll spread on a contract that settled below
        zero is a difference, and no ratio is taken."""
        result = _feed("absolute_points", _drifting(-37.6), [-40.0] * N)
        assert result["current_value"] == pytest.approx(2.51)

    def test_a_leg_below_the_ratio_floor_is_fine_in_points(self):
        result = _feed("absolute_points", _drifting(1.0), [1e-300] * N)
        assert math.isfinite(result["baseline_mean"])
