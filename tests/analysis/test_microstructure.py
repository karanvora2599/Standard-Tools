"""
Tick-level microstructure estimators.

The fixtures here are built so the right answer is known by arithmetic, not
by recording whatever the implementation happened to produce. A book quoted
99.95 / 100.05 has a midpoint of exactly 100.00 and a spread of exactly 10
basis points; a buy filled at the ask therefore pays exactly 10 bps
effective, and a trade at the midpoint pays exactly zero. Every assertion
below traces back to one of those.

Two properties get the most attention because getting them wrong produces
plausible numbers rather than obvious failures:

  - The prevailing quote must be strictly BEFORE the trade. A quote stamped
    at the same instant already reflects the trade, and using it biases
    every spread toward zero -- a bug that makes execution look free.
  - Unclassifiable trades must be dropped, not defaulted to a side. A
    coin-flip sign puts noise into every size-weighted mean while leaving
    the output looking complete.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.liquidity_events import _signed_volume
from standard_quant_tools.analysis.microstructure import (
    effective_spread,
    intraday_volume_profile,
    microstructure_summary,
    quoted_spread,
    sign_trades,
    signs_positional,
    trade_size_profile,
)
from standard_quant_tools.analysis.microstructure_estimators import kyle_lambda
from standard_quant_tools.error import ValidationError

BASE = pd.Timestamp("2024-03-01 14:30:00")


def _quotes(rows):
    """rows: (offset_seconds, bid, ask, bid_size, ask_size)."""
    index = [BASE + pd.Timedelta(seconds=s) for s, *_ in rows]
    return pd.DataFrame(
        {
            "bid_price": [r[1] for r in rows],
            "ask_price": [r[2] for r in rows],
            "bid_size": [r[3] for r in rows],
            "ask_size": [r[4] for r in rows],
        },
        index=pd.DatetimeIndex(index),
    )


def _trades(rows):
    """rows: (offset_seconds, price, size)."""
    index = [BASE + pd.Timedelta(seconds=s) for s, *_ in rows]
    return pd.DataFrame(
        {"price": [r[1] for r in rows], "size": [r[2] for r in rows]},
        index=pd.DatetimeIndex(index),
    )


def _planted_tape(n=1200, seed=3, tie_share=0.4):
    """
    A tape in time order whose every side is known, with repeated stamps.

    The mid moves only between distinct timestamps, every print is one cent
    either side of it, and a quote carrying that mid is stamped a
    millisecond before each distinct timestamp. Lee-Ready therefore signs
    each print by its own price against the mid, whatever order a group of
    prints sharing a timestamp arrives in, and `side` is the answer row for
    row. `tie_share` of the prints repeat their predecessor's timestamp;
    live tapes run from a fifth to over half.
    """
    rng = np.random.default_rng(seed)
    step = np.where(rng.random(n) < tie_share, 0, rng.integers(5, 400, n))
    step[0] = 0
    stamps = BASE + pd.to_timedelta(np.cumsum(step), unit="ms")
    new = np.r_[True, step[1:] > 0]
    mid = 100.0 + np.cumsum(np.where(new, rng.normal(0, 0.01, n), 0.0))
    side = rng.choice([-1.0, 1.0], n)
    trades = pd.DataFrame(
        {"price": mid + 0.01 * side, "size": rng.integers(1, 300, n).astype(float)},
        index=pd.DatetimeIndex(stamps),
    )
    quotes = pd.DataFrame(
        {"bid_price": mid[new] - 0.01, "ask_price": mid[new] + 0.01},
        index=pd.DatetimeIndex(stamps[new] - pd.Timedelta(milliseconds=1)),
    )
    return trades, quotes, side


def _halves_swapped(frame):
    """Row order for the frame's second half followed by its first -- two
    pulls concatenated the wrong way round -- cut between two distinct
    timestamps so that no group of rows sharing one is split."""
    stamps = frame.index
    cut = next(
        i for i in range(len(frame) // 2, len(frame)) if stamps[i] != stamps[i - 1]
    )
    return np.r_[np.arange(cut, len(frame)), np.arange(cut)]


def _shuffled_keeping_ties(frame, seed):
    """
    A random row order for a frame in time order, in which every group of
    rows sharing a timestamp still arrives in its original relative order.

    Putting it back in time order with ties kept in arrival order gives the
    original frame exactly, so every such shuffle has one right answer.
    """
    rng = np.random.default_rng(seed)
    n = len(frame)
    group = pd.factorize(frame.index)[0]
    shuffled = rng.permutation(n)
    slots = np.lexsort((np.arange(n), group[shuffled]))
    members = shuffled[np.lexsort((shuffled, group[shuffled]))]
    order = np.empty(n, dtype=int)
    order[slots] = members
    return order


@pytest.fixture
def book():
    """A steady 99.95 / 100.05 book: mid 100.00, spread exactly 10 bps."""
    return _quotes([(s, 99.95, 100.05, 500, 500) for s in (0, 10, 20, 30, 40, 50, 60)])


class TestQuotedSpread:
    def test_spread_and_mid_are_exact(self, book):
        result = quoted_spread(book)
        assert (result["mid"] == 100.0).all()
        assert result["spread_bps"].to_numpy() == pytest.approx(10.0)

    def test_a_crossed_quote_is_dropped_not_averaged_in(self):
        """A negative spread is a feed artifact — a stale side, a crossed
        cross-venue book. Averaging one in pulls every summary toward zero
        and makes the venue look cheaper than it is."""
        quotes = _quotes(
            [
                (0, 99.95, 100.05, 100, 100),
                (10, 100.10, 99.90, 100, 100),  # crossed
                (20, 99.95, 100.05, 100, 100),
            ]
        )
        result = quoted_spread(quotes)
        assert len(result) == 2
        assert result["spread_bps"].to_numpy() == pytest.approx(10.0)

    def test_imbalance_signs_toward_the_heavier_side(self):
        quotes = _quotes([(0, 99.95, 100.05, 900, 100)])
        assert quoted_spread(quotes)["imbalance"].iloc[0] == pytest.approx(0.8)

    def test_a_positional_index_is_refused(self):
        frame = pd.DataFrame({"bid_price": [1.0], "ask_price": [1.1]})
        with pytest.raises(ValidationError) as exc:
            quoted_spread(frame)
        assert "timestamp" in str(exc.value)


class TestSignTrades:
    def test_above_mid_is_a_buy_and_below_is_a_sell(self, book):
        trades = _trades([(5, 100.05, 100), (15, 99.95, 100)])
        signs = sign_trades(trades, book)
        assert list(signs) == [1, -1]

    def test_a_midpoint_trade_falls_back_to_the_tick_test(self, book):
        """At the midpoint the quote comparison says nothing, so the rule
        is the tick test: higher than the last different price is a buy."""
        trades = _trades([(5, 99.95, 100), (15, 100.00, 100)])
        signs = sign_trades(trades, book)
        assert signs.iloc[-1] == 1

    def test_the_prevailing_quote_is_strictly_earlier(self):
        """The bias that matters. If a quote stamped at the same instant as
        the trade were used, this buy would be measured against a book that
        already moved for it, and the spread would collapse."""
        quotes = _quotes([(0, 99.95, 100.05, 100, 100), (10, 100.04, 100.06, 100, 100)])
        trades = _trades([(10, 100.05, 100)])
        # Against the EARLIER book (mid 100.00) this is a buy above the mid.
        # Against the same-instant book (mid 100.05) it would be at the mid
        # and get no quote signal at all.
        assert sign_trades(trades, quotes).iloc[0] == 1

    def test_a_run_of_identical_prices_keeps_the_move_that_started_it(self):
        """The tick test compares against the last DIFFERENT price. Using
        the last price would leave every repeat unclassified."""
        trades = _trades([(0, 100.00, 10), (1, 100.05, 10), (2, 100.05, 10)])
        signs = sign_trades(trades)
        assert list(signs) == [1, 1]

    def test_unclassifiable_trades_are_dropped_not_defaulted(self):
        """The first trade has no prior price and sits at the midpoint;
        there is no information to sign it with, and inventing one would
        put a coin flip into every average."""
        trades = _trades([(5, 100.00, 100), (15, 100.05, 100)])
        quotes = _quotes([(0, 99.95, 100.05, 100, 100)])
        signs = sign_trades(trades, quotes)
        assert len(signs) == 1
        assert signs.index[0] == BASE + pd.Timedelta(seconds=15)


class TestPositionalSignsFollowTheCallersRows:
    """
    `signs_positional` is lined up against the caller's own rows by every
    consumer, so its array has to be in those rows' order -- whatever order
    they arrived in. Vendor tapes are not always in time order, and two
    pulls concatenated the wrong way round is an ordinary caller pattern.
    The signs are still COMPUTED in time order: the tick rule means the
    previous print in time, not the previous row.
    """

    def test_two_pulls_concatenated_out_of_order_are_signed_row_for_row(self):
        trades, quotes, side = _planted_tape()
        order = _halves_swapped(trades)
        signs = signs_positional(trades.iloc[order], quotes)
        np.testing.assert_array_equal(signs, side[order])

    def test_the_tick_rule_still_reads_the_previous_print_in_time(self):
        trades, _, _ = _planted_tape()
        order = _halves_swapped(trades)
        np.testing.assert_array_equal(
            signs_positional(trades.iloc[order]), signs_positional(trades)[order]
        )

    def test_a_tape_in_time_order_gets_its_planted_sides(self):
        """Null case: nothing to reorder, and the answer is the truth."""
        trades, quotes, side = _planted_tape()
        np.testing.assert_array_equal(signs_positional(trades, quotes), side)
        labelled = sign_trades(trades)
        tick = signs_positional(trades)
        decided = np.isfinite(tick) & (tick != 0)
        np.testing.assert_array_equal(tick[decided], labelled.to_numpy())

    def test_signed_volume_does_not_depend_on_the_row_order(self):
        trades, quotes, _ = _planted_tape()
        swapped = trades.iloc[_halves_swapped(trades)]
        pd.testing.assert_series_equal(
            _signed_volume(swapped, quotes, freq="10s"),
            _signed_volume(trades, quotes, freq="10s"),
        )

    def test_kyle_lambda_from_a_tape_does_not_depend_on_the_row_order(self):
        trades, quotes, _ = _planted_tape(n=3000)
        swapped = trades.iloc[_halves_swapped(trades)]
        in_order = kyle_lambda(trades=trades, quotes=quotes, freq="5s")
        out_of_order = kyle_lambda(trades=swapped, quotes=quotes, freq="5s")
        assert out_of_order["kyle_lambda"] == in_order["kyle_lambda"]
        assert out_of_order["r_squared"] == in_order["r_squared"]


class TestTiesKeepTheirArrivalOrder:
    """
    Rows sharing a timestamp are put in time order WITHOUT being reordered
    among themselves. Live tapes repeat timestamps on a fifth to over half
    of their prints, the tick rule reads the previous print, and on the
    quote side the last update at a timestamp is the book that stood. An
    unstable sort permuted every tie group as soon as one row was out of
    place, so the same tape gave a different answer per arrival order.
    """

    def test_every_order_that_keeps_ties_in_arrival_order_gives_one_answer(self):
        trades, _, _ = _planted_tape(n=3000)
        expected_signs = signs_positional(trades)
        expected_fraction = microstructure_summary(trades)["buy_volume_fraction"]

        unique = np.flatnonzero(~trades.index.duplicated(keep=False))
        one_late_print = np.arange(len(trades))
        one_late_print[[unique[100], unique[101]]] = [unique[101], unique[100]]
        orders = [one_late_print] + [
            _shuffled_keeping_ties(trades, seed) for seed in range(6)
        ]
        for order in orders:
            shuffled = trades.iloc[order]
            summary = microstructure_summary(shuffled)
            assert summary["buy_volume_fraction"] == expected_fraction
            np.testing.assert_array_equal(
                signs_positional(shuffled), expected_signs[order]
            )

    def test_the_last_quote_update_at_a_timestamp_is_the_one_that_prevails(self):
        """Each timestamp carries a stale quote and then the live one, and a
        trade a millisecond later is measured against the live mid. Against
        the stale mid, five cents low, every sell reads as a buy."""
        rng = np.random.default_rng(2)
        n = 400
        stamps = np.repeat(BASE + pd.to_timedelta(np.arange(n) * 100, unit="ms"), 2)
        live = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
        mids = np.empty(2 * n)
        mids[0::2] = live - 0.05
        mids[1::2] = live
        quotes = pd.DataFrame(
            {"bid_price": mids - 0.01, "ask_price": mids + 0.01},
            index=pd.DatetimeIndex(stamps),
        )
        side = rng.choice([-1.0, 1.0], n)
        trades = pd.DataFrame(
            {"price": live + 0.01 * side, "size": 100.0},
            index=pd.DatetimeIndex(stamps[1::2] + pd.Timedelta(milliseconds=1)),
        )
        np.testing.assert_array_equal(signs_positional(trades, quotes), side)
        for seed in range(4):
            shuffled = quotes.iloc[_shuffled_keeping_ties(quotes, seed)]
            np.testing.assert_array_equal(signs_positional(trades, shuffled), side)


class TestEffectiveSpread:
    def test_a_fill_at_the_ask_pays_the_full_quoted_spread(self, book):
        trades = _trades([(5, 100.05, 100)])
        result = effective_spread(trades, book)
        assert result["effective_spread_bps"].iloc[0] == pytest.approx(10.0)

    def test_a_fill_at_the_mid_pays_nothing(self, book):
        trades = _trades([(5, 99.95, 100), (15, 100.00, 100)])
        result = effective_spread(trades, book)
        assert result["effective_spread_bps"].iloc[-1] == pytest.approx(0.0)

    def test_a_sell_at_the_bid_pays_the_same_as_a_buy_at_the_ask(self, book):
        """The sign convention has to make both sides positive, or the
        average of a balanced tape is zero and the venue looks free."""
        trades = _trades([(5, 100.05, 100), (15, 99.95, 100)])
        result = effective_spread(trades, book)
        assert (result["effective_spread_bps"] > 0).all()
        assert result["effective_spread_bps"].iloc[0] == pytest.approx(
            result["effective_spread_bps"].iloc[1]
        )

    def test_sweeping_through_the_book_costs_more_than_the_quote(self, book):
        trades = _trades([(5, 100.15, 100)])
        result = effective_spread(trades, book)
        assert result["effective_spread_bps"].iloc[0] == pytest.approx(30.0)

    def test_the_realized_split_adds_up(self):
        """effective = realized + impact, by construction. A buy at 100.05
        against a mid of 100.00 that drifts to 100.02 keeps 6 bps for the
        provider and moved the market 4."""
        quotes = _quotes(
            [
                (0, 99.95, 100.05, 100, 100),
                (65, 99.97, 100.07, 100, 100),  # mid 100.02, one horizon later
            ]
        )
        trades = _trades([(5, 100.05, 100)])
        result = effective_spread(trades, quotes, pd.Timedelta(seconds=60))
        row = result.iloc[0]
        assert row["effective_spread_bps"] == pytest.approx(10.0)
        assert row["realized_spread_bps"] == pytest.approx(6.0, abs=1e-6)
        assert row["price_impact_bps"] == pytest.approx(4.0, abs=1e-6)

    def test_a_non_positive_horizon_is_refused(self, book):
        trades = _trades([(5, 100.05, 100)])
        with pytest.raises(ValidationError) as exc:
            effective_spread(trades, book, pd.Timedelta(seconds=0))
        assert "positive" in str(exc.value)

    def test_no_usable_quotes_is_an_error_not_a_zero(self):
        trades = _trades([(5, 100.05, 100)])
        crossed = _quotes([(0, 100.10, 99.90, 100, 100)])
        with pytest.raises(ValidationError) as exc:
            effective_spread(trades, crossed)
        assert "no usable quotes" in str(exc.value)


class TestSummary:
    def test_size_weighting_differs_from_count_weighting(self, book):
        """The whole reason both are reported. One large expensive print
        among many small cheap ones is what a position actually pays."""
        trades = _trades([(5, 100.05, 1), (15, 100.05, 1), (25, 100.15, 10_000)])
        summary = microstructure_summary(trades, book)
        # Two 10 bps odd lots and one 30 bps block: by count the tape looks
        # like ~17 bps, by size it cost ~30. The position pays the latter.
        assert summary["effective_spread_bps_mean"] == pytest.approx(16.67, abs=0.1)
        assert summary["effective_spread_bps_size_weighted"] == pytest.approx(
            30.0, abs=0.1
        )

    def test_buy_volume_fraction_reads_directional_flow(self, book):
        trades = _trades([(5, 100.05, 900), (15, 99.95, 100)])
        summary = microstructure_summary(trades, book)
        assert summary["buy_volume_fraction"] == pytest.approx(0.9)

    def test_vwap_is_size_weighted(self, book):
        trades = _trades([(5, 100.00, 1), (15, 110.00, 9)])
        assert microstructure_summary(trades, book)["vwap"] == pytest.approx(109.0)

    def test_without_quotes_it_says_what_it_could_not_measure(self):
        trades = _trades([(0, 100.00, 10), (1, 100.05, 10)])
        summary = microstructure_summary(trades, None)
        assert "quoted_spread_bps_mean" not in summary
        assert any("less accurate than Lee-Ready" in n for n in summary["notes"])


class TestProfiles:
    def test_size_buckets_are_quantiles_of_this_symbol(self):
        sizes = [1, 2, 5, 10, 100, 1_000, 5_000, 10_000]
        trades = _trades([(i, 100.0, s) for i, s in enumerate(sizes)])
        profile = trade_size_profile(trades, buckets=4)
        assert len(profile["buckets"]) == 4
        assert sum(b["volume_fraction"] for b in profile["buckets"]) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_block_heavy_volume_shows_in_the_largest_bucket(self):
        trades = _trades([(i, 100.0, 1) for i in range(90)] + [(90, 100.0, 1_000_000)])
        profile = trade_size_profile(trades, buckets=5)
        assert profile["largest_bucket_volume_fraction"] > 0.99

    def test_a_single_size_reports_one_bucket_honestly(self):
        trades = _trades([(i, 100.0, 100) for i in range(10)])
        profile = trade_size_profile(trades, buckets=5)
        assert len(profile["buckets"]) == 1
        assert profile["largest_bucket_volume_fraction"] == 1.0

    def test_intraday_profile_finds_the_peak_bucket(self):
        trades = _trades([(0, 100.0, 10), (60, 100.0, 10), (3_600, 100.0, 1_000)])
        profile = intraday_volume_profile(trades, freq="30min")
        assert profile["peak_volume_fraction"] > 0.9
        assert profile["peak_time"] == "15:30:00"

    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(3)
        trades = _trades(
            [
                (int(s), 100.0, float(v))
                for s, v in zip(range(0, 7200, 30), rng.integers(1, 500, 240))
            ]
        )
        profile = intraday_volume_profile(trades, freq="15min")
        assert sum(b["volume_fraction"] for b in profile["buckets"]) == pytest.approx(
            1.0, abs=1e-5
        )


class TestEveryTradeProfileBucketIsReported:
    """
    The tape's time-of-day profile returned only the buckets some print fell
    in, labelled by the clock: a tape from the open and the close alone had
    no midday, so the trough the U is measured against was simply missing.
    """

    @staticmethod
    def _open_and_close(tz=None):
        day = "2024-03-04"
        stamps = [f"{day} 09:31", f"{day} 09:45", f"{day} 15:40", f"{day} 15:55"]
        index = pd.DatetimeIndex(stamps)
        if tz is not None:
            index = index.tz_localize(tz)
        return pd.DataFrame({"price": 100.0, "size": [400, 300, 200, 500]}, index=index)

    def test_the_empty_middle_is_returned_and_is_the_trough(self):
        profile = intraday_volume_profile(self._open_and_close(), freq="30min")
        assert profile["n_buckets"] == 13
        assert [b["bucket"] for b in profile["buckets"]] == list(range(13))
        assert profile["buckets"][0]["time"] == "09:30:00"
        assert profile["buckets"][-1]["time"] == "15:30:00"
        assert profile["n_empty_buckets"] == 11
        middle = profile["buckets"][1:-1]
        assert all(b["volume_fraction"] == 0.0 and b["n_trades"] == 0 for b in middle)
        # The trough is a bucket id -- the first of the empty ones -- and
        # not a position in a shortened list that pointed at the close.
        assert profile["trough_bucket"] == 1
        assert profile["trough_volume_fraction"] == 0.0
        assert profile["peak_bucket"] == 0
        assert any("11 of 13 buckets" in w for w in profile["warnings"])

    def test_a_zoned_tape_is_bucketed_over_the_session(self):
        profile = intraday_volume_profile(
            self._open_and_close("America/New_York"), freq="1h"
        )
        assert profile["bucket_span"] == ["09:30:00", "16:00:00"]
        assert profile["n_buckets"] == 7
        assert profile["extended_hours_share"] == 0.0

    def test_a_naive_tape_outside_the_session_is_extended_and_said(self):
        trades = self._open_and_close()
        trades.index = trades.index.append(pd.DatetimeIndex(["2024-03-04 17:10"]))[1:]
        profile = intraday_volume_profile(trades, freq="30min")
        assert profile["bucket_span"][1] == "17:30:00"
        assert any("outside" in w for w in profile["warnings"])

    def test_null_a_full_day_has_no_empty_bucket(self):
        index = pd.date_range("2024-03-04 09:30", "2024-03-04 15:59", freq="5min")
        trades = pd.DataFrame({"price": 100.0, "size": 100.0}, index=index)
        profile = intraday_volume_profile(trades, freq="30min")
        assert profile["n_empty_buckets"] == 0
        assert not [w for w in profile["warnings"] if "hold no trades" in w]

    def test_null_a_duration_that_is_not_one_is_refused(self):
        with pytest.raises(ValidationError, match="freq"):
            intraday_volume_profile(self._open_and_close(), freq="soon")
