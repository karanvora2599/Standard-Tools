"""
The live-feed layer: depth books, and a monitor that survives being paused.

Two identities carry most of the weight here.

  * A monitor fed a hundred ticks in one call and a hundred calls of one
    tick must reach the SAME state. That is what makes the call frequency a
    deployment decision rather than a modelling one, and it is the property
    that breaks first if any accumulator is recomputed instead of carried.
  * The microprice equals the midpoint exactly on a balanced book, and
    leans toward the THIN side otherwise. The direction reads backwards and
    is the whole point: the heavy side is the side that absorbs.

The degenerate-baseline case is tested explicitly because it is where the
first implementation was wrong: the naive `sum_sq/n - mean**2` variance
returned 1.4e-06 on seventy identical ticks rather than 0, slipped past a
`std <= 0` guard, and produced a CUSUM statistic of 1.7 billion.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.order_book import (
    book_metrics,
    depth_profile,
    microprice,
)
from standard_quant_tools.delta_one.streaming import (
    CHANNELS,
    new_spread_monitor,
    reset_spread_monitor,
    update_spread_monitor,
)
from standard_quant_tools.error import ValidationError


def _book(n=400, levels=5, seed=4):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.02, n))
    columns = {}
    for i in range(levels):
        columns[f"bid_price_{i}"] = mid - 0.01 * (i + 1)
        columns[f"ask_price_{i}"] = mid + 0.01 * (i + 1)
        # Thinning with depth, which is what a real book does.
        columns[f"bid_size_{i}"] = rng.integers(100, 900, n) / (i + 1)
        columns[f"ask_size_{i}"] = rng.integers(100, 900, n) / (i + 1)
    return pd.DataFrame(columns)


class TestMicroprice:
    def test_a_balanced_book_is_exactly_the_midpoint(self):
        assert microprice(99.0, 500, 101.0, 500) == 100.0
        assert microprice(10.0, 7, 10.5, 7) == pytest.approx(10.25)

    def test_it_leans_toward_the_thin_side(self):
        """The heavy side absorbs, so price sits nearer the thin one."""
        heavy_bid = microprice(99.0, 5000, 101.0, 100)
        heavy_ask = microprice(99.0, 100, 101.0, 5000)
        assert heavy_bid > 100.0
        assert heavy_ask < 100.0
        # Symmetric: mirroring the book mirrors the answer about the mid.
        assert heavy_bid - 100.0 == pytest.approx(100.0 - heavy_ask)

    def test_an_empty_book_has_no_size_weighted_price(self):
        assert np.isnan(microprice(99.0, 0, 101.0, 0))


class TestBookMetrics:
    def test_it_reads_every_complete_level(self):
        out = book_metrics(_book(levels=5))
        assert out["levels_available"] == 5
        assert out["levels_read"] == 5

    def test_a_one_level_book_has_no_slope(self):
        """Top-of-book cannot answer a question about depth behind it."""
        out = book_metrics(_book(), levels=1)
        assert out["depth_slope"] is None
        assert any("one level" in w for w in out["warnings"])

    def test_a_quote_panel_is_refused_by_name(self):
        panel = pd.DataFrame({"bid_price": [99.0], "ask_price": [101.0]})
        with pytest.raises(ValidationError, match="get_order_book"):
            book_metrics(panel)

    def test_the_microprice_lean_is_a_position_in_the_spread(self):
        out = book_metrics(_book())
        assert 0.0 <= out["mean_microprice_lean"] <= 1.0

    def test_a_crossed_book_is_excluded_rather_than_averaged(self):
        book = _book(n=50)
        # Cross ten snapshots by putting the offer under the bid.
        book.loc[:9, "ask_price_0"] = book.loc[:9, "bid_price_0"] - 0.05
        out = book_metrics(book)
        assert out["n_crossed"] == 10
        assert out["mean_spread"] > 0
        assert any("crossed" in w for w in out["warnings"])

    def test_depth_thins_with_distance(self):
        rows = depth_profile(_book())["profile"]
        sizes = [r["mean_bid_size"] for r in rows]
        assert sizes == sorted(sizes, reverse=True)
        distances = [r["mean_bid_distance_bps"] for r in rows]
        assert distances == sorted(distances)


class TestSpreadMonitor:
    def _feed(self, n=200, shift=45.0, at=None, seed=7):
        """A spread that steps partway through, returned as (primary, reference).

        `at` defaults to 60% of the window rather than a fixed index, so a
        short feed does not ask for a negative-length second segment.
        """
        at = int(n * 0.6) if at is None else at
        rng = np.random.default_rng(seed)
        bps = np.concatenate([rng.normal(30, 4, at), rng.normal(30 + shift, 4, n - at)])
        reference = 6000 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, n)))
        return reference * (1 + bps / 10_000), reference

    def test_a_batch_and_a_stream_reach_the_same_state(self):
        """The property that makes call frequency a deployment choice."""
        primary, reference = self._feed()
        batched = update_spread_monitor(
            new_spread_monitor(warmup=60), primary=primary, reference=reference
        )
        state = new_spread_monitor(warmup=60)
        for i in range(len(primary)):
            streamed = update_spread_monitor(
                state, primary=[primary[i]], reference=[reference[i]]
            )
            state = streamed["state"]
        assert batched["state"]["first_crossing_at"] == state["first_crossing_at"]
        assert batched["peak_statistic"] == pytest.approx(streamed["peak_statistic"])
        assert batched["baseline_mean"] == pytest.approx(state["baseline_mean"])
        assert batched["baseline_std"] == pytest.approx(state["baseline_std"])

    def test_nothing_triggers_during_warm_up(self):
        primary, reference = self._feed(n=40, at=20)
        out = update_spread_monitor(
            new_spread_monitor(warmup=60), primary=primary, reference=reference
        )
        assert out["warming_up"] is True
        assert out["triggered"] is False
        assert out["baseline_mean"] is None
        assert any("warming up" in w for w in out["warnings"])

    def test_a_sustained_shift_triggers_and_a_stable_feed_does_not(self):
        shifted, reference = self._feed(shift=45.0)
        flat, reference2 = self._feed(shift=0.0)
        assert update_spread_monitor(
            new_spread_monitor(warmup=60), primary=shifted, reference=reference
        )["triggered"]
        assert not update_spread_monitor(
            new_spread_monitor(warmup=60), primary=flat, reference=reference2
        )["triggered"]

    def test_the_alert_fires_once_and_not_again(self):
        primary, reference = self._feed()
        first = update_spread_monitor(
            new_spread_monitor(warmup=60), primary=primary, reference=reference
        )
        assert first["alert"] is not None
        again = update_spread_monitor(
            first["state"], primary=primary[-10:], reference=reference[-10:]
        )
        assert again["triggered"] is True
        assert again["alert"] is None
        assert any("NOT re-alerting" in w for w in again["warnings"])

    def test_reset_can_keep_or_relearn_the_baseline(self):
        primary, reference = self._feed()
        out = update_spread_monitor(
            new_spread_monitor(warmup=60), primary=primary, reference=reference
        )
        kept = reset_spread_monitor(out["state"], keep_baseline=True)
        assert kept["triggered"] is False
        assert kept["baseline_mean"] == pytest.approx(out["baseline_mean"])
        relearn = reset_spread_monitor(out["state"], keep_baseline=False)
        assert relearn["baseline_mean"] is None
        assert relearn["n"] == 0

    def test_a_frozen_warm_up_cannot_produce_an_enormous_statistic(self):
        """The bug the naive variance caused: 1.7 billion from 70 flat ticks."""
        out = update_spread_monitor(
            new_spread_monitor(warmup=60),
            primary=[100.3] * 70,
            reference=[100.0] * 70,
        )
        assert out["baseline_std"] == 0.0
        assert out["degenerate_baseline"] is True
        moved = update_spread_monitor(
            out["state"], primary=[100.9] * 40, reference=[100.0] * 40
        )
        assert moved["triggered"] is False
        assert moved["statistic"] == 0.0

    def test_a_state_from_another_version_is_refused_rather_than_resumed(self):
        state = new_spread_monitor(warmup=10)
        state["version"] = 0
        with pytest.raises(ValidationError, match="version"):
            update_spread_monitor(state, primary=[100.1], reference=[100.0])

    def test_mismatched_legs_are_refused(self):
        with pytest.raises(ValidationError, match="both legs"):
            update_spread_monitor(
                new_spread_monitor(warmup=10),
                primary=[100.0, 101.0],
                reference=[100.3],
            )


class TestTheThreeChannels:
    """Five roadmap monitors, three formulas, one tool.

    What is pinned here is that each channel computes what it says and that
    the sign convention is the same across all of them: positive means the
    PRIMARY leg is dear to the reference.
    """

    def _run(self, channel, primary, reference, **kwargs):
        state = new_spread_monitor(channel=channel, warmup=10)
        return update_spread_monitor(
            state, primary=primary, reference=reference, **kwargs
        )

    def test_every_channel_is_documented_and_has_a_stated_use(self):
        from standard_quant_tools.delta_one.streaming import CHANNEL_USES

        assert set(CHANNELS) == set(CHANNEL_USES)
        for name, description in CHANNELS.items():
            assert len(description) > 60, name
            assert len(CHANNEL_USES[name]) > 40, name

    def test_relative_bps_is_a_ratio_in_basis_points(self):
        out = self._run("relative_bps", [101.0] * 12, [100.0] * 12)
        assert out["current_value"] == pytest.approx(100.0)

    def test_absolute_points_is_a_difference(self):
        out = self._run("absolute_points", [6072.0] * 12, [6041.0] * 12)
        assert out["current_value"] == pytest.approx(31.0)

    def test_annualized_bps_is_a_rate(self):
        out = self._run(
            "annualized_bps",
            [101.0] * 12,
            [100.0] * 12,
            time_to_expiry=[0.5] * 12,
        )
        expected = math.log(1.01) / 0.5 * 10_000.0
        assert out["current_value"] == pytest.approx(expected)

    def test_the_sign_convention_is_the_same_on_every_channel(self):
        """Positive always means the PRIMARY leg is dear."""
        for channel, extra in (
            ("relative_bps", {}),
            ("absolute_points", {}),
            ("annualized_bps", {"time_to_expiry": [0.5] * 12}),
        ):
            dear = self._run(channel, [101.0] * 12, [100.0] * 12, **extra)
            cheap = self._run(channel, [99.0] * 12, [100.0] * 12, **extra)
            assert dear["current_value"] > 0, channel
            assert cheap["current_value"] < 0, channel

    def test_an_etf_premium_is_the_relative_channel(self):
        """primary=ETF price, reference=NAV. 40 bps premium."""
        out = self._run("relative_bps", [100.40] * 12, [100.0] * 12)
        assert out["current_value"] == pytest.approx(40.0)

    def test_a_roll_spread_is_the_points_channel(self):
        """A calendar spread in bps of a 6000 index would round to nothing."""
        out = self._run("absolute_points", [6072.0] * 12, [6041.0] * 12)
        assert out["current_value"] == pytest.approx(31.0)
        as_bps = self._run("relative_bps", [6072.0] * 12, [6041.0] * 12)
        assert as_bps["current_value"] == pytest.approx(51.31, abs=0.02)

    def test_absolute_points_accepts_a_negative_reference(self):
        """A difference needs no positive denominator; a ratio does."""
        out = self._run("absolute_points", [1.0] * 12, [-2.0] * 12)
        assert out["current_value"] == pytest.approx(3.0)
        with pytest.raises(ValidationError, match="divides by it"):
            self._run("relative_bps", [1.0] * 12, [-2.0] * 12)

    def test_annualized_insists_on_a_time_to_expiry(self):
        with pytest.raises(ValidationError, match="time_to_expiry"):
            self._run("annualized_bps", [101.0] * 12, [100.0] * 12)

    def test_the_other_channels_refuse_one(self):
        for channel in ("relative_bps", "absolute_points"):
            with pytest.raises(ValidationError, match="does not use it"):
                self._run(
                    channel, [101.0] * 12, [100.0] * 12, time_to_expiry=[0.5] * 12
                )

    def test_an_unknown_channel_names_the_three(self):
        with pytest.raises(ValidationError, match="three formulas"):
            new_spread_monitor(channel="index_arb")

    def test_the_channel_travels_in_the_state(self):
        out = self._run("absolute_points", [6072.0] * 12, [6041.0] * 12)
        assert out["channel"] == "absolute_points"
        assert out["state"]["channel"] == "absolute_points"
        resumed = update_spread_monitor(
            out["state"], primary=[6080.0], reference=[6041.0]
        )
        assert resumed["current_value"] == pytest.approx(39.0)
