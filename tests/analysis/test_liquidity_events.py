"""
Multi-channel change detection: which part of the market moved.

WHAT THESE TESTS ARE REALLY FOR. Two bugs were found by running the detector
on data with no shock in it, and both would have shipped as confident
alarms rather than as errors:

1. **A CUSUM over a price LEVEL fires on drift alone.** Eight of eight pure
   random walks with no shock triggered `mid_price`, because a level is not
   stationary and the statistic accumulates the walk itself. The channel is
   now the mid RETURN, and `mid_price` survives as an explicitly refused
   channel so the trap is visible rather than merely absent.

2. **A near-constant reference window produces an unbounded statistic.** A
   frozen spread moving from 1.00 bps to 1.02 bps produced a CUSUM peak of
   286,431 and a severity of "very high". The denominator was near zero.
   That is now labelled, and the level shift is reported in the channel's
   own units so a reader can see 1.00 -> 1.02 without decoding an
   accumulated statistic.

Neither failure crashes, returns nothing, or looks wrong. Both return a
large number with obscure units, which is the shape of bug a detector nobody
watches closely will keep forever. So the tests below are built around data
whose answer is known: a null with no shock, and a shock in a channel that
is NOT price.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.liquidity_events import (
    CHANNELS,
    DEFAULT_THRESHOLD,
    available_channels,
    cusum,
    declared_channels,
    detect_liquidity_events,
)
from standard_quant_tools.error import ValidationError


def _market(seed=0, *, shock=False, n=1500, live_spread=True):
    """
    Trades and quotes with a KNOWN answer.

    When `shock` is set, the spread widens and flow turns one-sided from 60%
    through — while the mid keeps the same random walk it always had. That
    is the case the whole tool exists for: liquidity deteriorates and price
    has not moved yet.
    """
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-03-01 09:30", periods=n, freq="1s")
    on = (np.arange(n) >= int(n * 0.6)) if shock else np.zeros(n, bool)
    mid = 100 + np.cumsum(rng.normal(0, 0.004, n))
    base = np.where(on, 0.06, 0.01)
    half = base * (rng.lognormal(0, 0.25, n) if live_spread else 1.0)
    quotes = pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price": mid - half,
            "ask_price": mid + half,
            "bid_size": 500.0,
            "ask_size": 500.0,
        }
    )
    side = np.where(on, rng.choice([1, 1, 1, -1], n), rng.choice([1, -1], n))
    trades = pd.DataFrame(
        {
            "timestamp": stamps,
            "price": mid + side * half * 0.9,
            "size": rng.integers(80, 300, n).astype(float),
        }
    )
    return trades, quotes


class TestTheChannelSetIsData:
    def test_adding_a_channel_is_a_table_entry(self):
        """One tool over a declared set, not one tool per channel -- the
        same rule that keeps STRATEGY_REGISTRY from becoming twelve backtest
        tools."""
        assert len(declared_channels()) > len(available_channels())
        for name in declared_channels():
            assert CHANNELS[name].description.strip()
            assert CHANNELS[name].requires

    def test_an_unavailable_channel_is_declared_not_omitted(self):
        """An agent asking for `ofi` should learn the channel exists and
        what it needs, rather than that the name was never heard of."""
        assert "ofi" in declared_channels()
        assert "ofi" not in available_channels()
        assert "order book" in CHANNELS["ofi"].why_unavailable()

    def test_an_unknown_channel_is_refused_with_a_suggestion(self):
        with pytest.raises(ValidationError) as exc:
            detect_liquidity_events(channels=["spred"])
        assert "spread" in str(exc.value)


class TestAPriceLevelIsNotStationary:
    """Bug 1, pinned. A CUSUM on a level fires on drift alone."""

    def test_mid_return_does_not_fire_on_a_pure_random_walk(self):
        fired = 0
        for seed in range(8):
            _trades, quotes = _market(seed)
            report = detect_liquidity_events(
                channels=["mid_return"], quotes=quotes, freq="30s"
            )
            fired += report["results"][0]["triggered"]
        assert fired <= 1, (
            f"mid_return triggered on {fired}/8 random walks with no shock. "
            "That is what a CUSUM over a price LEVEL does; the channel must "
            "be the return."
        )

    def test_mid_price_is_refused_and_says_why(self):
        """Kept in the table so the trap is visible. 'Unknown channel' would
        be the wrong answer to a reasonable-sounding question."""
        report = detect_liquidity_events(
            channels=["mid_price"], quotes=_market()[1], freq="30s"
        )
        assert report["channels_run"] == []
        reason = report["unavailable"][0]["reason"]
        assert "drift alone" in reason
        assert "mid_return" in reason


class TestADegenerateBaselineIsLabelled:
    """Bug 2, pinned. A near-constant reference window makes the statistic a
    ratio to nearly zero."""

    def test_a_frozen_channel_is_flagged(self):
        _trades, quotes = _market(0, live_spread=False)
        result = detect_liquidity_events(
            channels=["spread"], quotes=quotes, freq="30s"
        )["results"][0]
        assert result["degenerate_baseline"]
        assert any("nearly constant" in n for n in result["notes"])

    def test_a_real_channel_is_not_flagged(self):
        _trades, quotes = _market(0, live_spread=True)
        result = detect_liquidity_events(
            channels=["spread"], quotes=quotes, freq="30s"
        )["results"][0]
        assert not result["degenerate_baseline"]

    def test_the_detection_is_labelled_not_suppressed(self):
        """A genuinely calm period before a genuine shock is the case this
        tool is for. Refusing to report it would be worse than labelling
        it."""
        _trades, quotes = _market(0, shock=True, live_spread=False)
        result = detect_liquidity_events(
            channels=["spread"], quotes=quotes, freq="30s"
        )["results"][0]
        assert result["triggered"]
        assert result["degenerate_baseline"]

    def test_the_shift_is_reported_in_the_channels_own_units(self):
        """The number a reader can check. A peak statistic is accumulated
        and unbounded; a before-and-after mean is not."""
        _trades, quotes = _market(0, shock=True)
        result = detect_liquidity_events(
            channels=["spread"], quotes=quotes, freq="30s"
        )["results"][0]
        assert result["shift"] > 0
        assert result["mean_after_reference"] > result["baseline_mean"]

    def test_the_report_says_which_channels_to_distrust(self):
        _trades, quotes = _market(0, live_spread=False)
        report = detect_liquidity_events(channels=["spread"], quotes=quotes, freq="30s")
        assert any("denominator near zero" in w for w in report["warnings"])


class TestItFindsTheRightChannel:
    def test_liquidity_channels_fire_while_price_does_not(self):
        """The plan's own example, and the reason the tool exists: the book
        deteriorates before the mid moves."""
        trades, quotes = _market(0, shock=True)
        report = detect_liquidity_events(
            channels=["mid_return", "spread", "effective_spread"],
            trades=trades,
            quotes=quotes,
            freq="30s",
        )
        by_name = {r["channel"]: r for r in report["results"]}
        assert by_name["spread"]["triggered"]
        assert by_name["effective_spread"]["triggered"]
        assert not by_name["mid_return"]["triggered"]

    def test_that_sequence_is_explained_rather_than_flagged_as_odd(self):
        trades, quotes = _market(0, shock=True)
        report = detect_liquidity_events(
            channels=["mid_return", "spread"],
            trades=trades,
            quotes=quotes,
            freq="30s",
        )
        assert any("ordinary sequence" in w for w in report["warnings"])

    def test_the_worst_channel_is_the_one_with_the_largest_statistic(self):
        trades, quotes = _market(0, shock=True)
        report = detect_liquidity_events(
            channels=["mid_return", "spread", "realized_vol"],
            trades=trades,
            quotes=quotes,
            freq="30s",
        )
        peaks = [r["peak_statistic"] for r in report["results"]]
        assert peaks == sorted(peaks, reverse=True)
        assert report["worst_channel"] == report["results"][0]["channel"]

    def test_a_missing_frame_makes_the_channel_unavailable_not_absent(self):
        """Dropping it would let a caller ask for signed volume, get a clean
        report with no row, and conclude the flow was balanced."""
        _trades, quotes = _market(0)
        report = detect_liquidity_events(
            channels=["spread", "signed_volume"], quotes=quotes, freq="30s"
        )
        assert [u["channel"] for u in report["unavailable"]] == ["signed_volume"]
        assert any("could not be run" in w for w in report["warnings"])


class TestTheCusumItself:
    def test_the_baseline_excludes_the_shock(self):
        """A shock inside its own baseline inflates the denominator it is
        measured against and hides itself."""
        series = pd.Series([1.0] * 50 + [50.0] * 50)
        result = cusum(series + np.random.default_rng(0).normal(0, 0.1, 100))
        assert result["triggered"]
        assert result["baseline_mean"] < 2.0, (
            "the baseline absorbed the shock, which is the failure that "
            "makes a detector find nothing"
        )

    def test_a_crossing_inside_the_reference_window_does_not_count(self):
        """The reference window defines normal; a trigger inside it would be
        the detector firing on the data that taught it what normal is."""
        rng = np.random.default_rng(0)
        series = pd.Series(rng.normal(0, 1, 200))
        result = cusum(series, reference_fraction=0.5)
        if result["triggered"]:
            crossing = pd.Timestamp if False else int(result["first_crossing"])
            assert crossing >= result["n_reference"]

    def test_a_constant_series_reports_no_scale_rather_than_no_event(self):
        result = cusum(pd.Series([3.0] * 60))
        assert not result["triggered"]
        assert "no scale" in result["reason"]
        assert "not evidence that nothing happened" in result["reason"]

    def test_too_short_a_series_is_refused(self):
        with pytest.raises(ValidationError, match="at least 10"):
            cusum(pd.Series([1.0, 2.0, 3.0]))

    def test_severity_climbs_with_the_statistic(self):
        rng = np.random.default_rng(0)
        seen = []
        for size in (3.0, 30.0):
            series = pd.Series(
                np.concatenate([rng.normal(0, 1, 80), rng.normal(size, 1, 80)])
            )
            seen.append(cusum(series)["severity"])
        assert seen[0] in {"low", "moderate", "high", "very high"}
        assert seen[-1] == "very high"

    def test_the_default_threshold_is_calibrated_against_pure_noise(self):
        """
        The bug this pins: the textbook threshold of 5.0 alarmed on iid noise
        36% of the time at n=120 and 82% at n=1000 -- WORSE with more data,
        which is the tell. CUSUM's design figure is an average run length,
        and "did anything happen anywhere in this window" is a different
        question: any fixed threshold alarms eventually.

        Calibrated to ~5% over the whole window, and checked at three
        lengths because a threshold that only holds at one is the same bug
        again.
        """
        for n, trials in ((120, 120), (400, 120), (1000, 80)):
            fired = sum(
                cusum(pd.Series(np.random.default_rng(seed).normal(0, 1, n)))[
                    "triggered"
                ]
                for seed in range(trials)
            )
            rate = fired / trials
            assert rate <= 0.15, (
                f"{rate:.0%} of pure-noise series of length {n} triggered at "
                f"the default threshold of {DEFAULT_THRESHOLD}. A detector "
                "that alarms on nothing is worse than no detector."
            )

    def test_a_real_shock_still_clears_the_higher_threshold(self):
        """Raising the threshold must not have bought quiet by going deaf."""
        rng = np.random.default_rng(0)
        series = pd.Series(
            np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 100)])
        )
        assert cusum(series)["triggered"]

    def test_direction_distinguishes_a_rise_from_a_fall(self):
        rng = np.random.default_rng(0)
        up = pd.Series(np.concatenate([rng.normal(0, 1, 60), rng.normal(8, 1, 60)]))
        down = pd.Series(np.concatenate([rng.normal(0, 1, 60), rng.normal(-8, 1, 60)]))
        assert cusum(up)["direction"] == "up"
        assert cusum(down)["direction"] == "down"


class TestTheL2Contract:
    """Phase 3 step 1: declared before any implementation, so the analysis
    that consumes a book can be written and tested before a source exists."""

    def test_no_provider_serves_an_order_book_and_says_so_by_name(self):
        from standard_quant_tools.data.factory import DataFactory

        provider = DataFactory.get_provider("yfinance")
        with pytest.raises(NotImplementedError) as exc:
            provider.get_order_book("AAPL", "2024-01-01", "2024-01-02")
        message = str(exc.value)
        assert "YFinanceProvider" in message
        assert "describe_data_capabilities" in message

    def test_the_refusal_warns_against_substituting_top_of_book(self):
        """Two levels of L1 masquerading as depth would report every book as
        perfectly balanced, which reads as a calm market rather than as
        missing data."""
        from standard_quant_tools.data.factory import DataFactory

        with pytest.raises(NotImplementedError) as exc:
            DataFactory.get_provider("yfinance").get_order_book(
                "AAPL", "2024-01-01", "2024-01-02"
            )
        assert "perfectly balanced" in str(exc.value)

    def test_the_column_layout_is_declared(self):
        from standard_quant_tools.data.base import DataProvider

        assert "bid_price_{level}" in DataProvider.ORDER_BOOK_COLUMNS
        assert "ask_size_{level}" in DataProvider.ORDER_BOOK_COLUMNS

    def test_every_l2_channel_names_the_order_book_as_its_requirement(self):
        for name, channel in CHANNELS.items():
            if not channel.available and name != "mid_price":
                assert "orderbook" in channel.requires, name
