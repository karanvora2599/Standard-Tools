"""
The microstructure tools against tapes and books shaped like real ones.

WHAT THESE ARE FOR. Every case below is one the surface used to answer
wrongly or not at all, and each has a planted answer rather than a shape
check:

    a tape with repeated timestamps     raised from inside pandas
    an inline book with no clock        reported null rates and said so;
                                        one WITH a clock could not be
                                        expressed at all
    the realized/impact split           computed, published, and left out
                                        of the summary
    an order lifetime                   a mean and a median over a 47x
                                        skew, and no tail
    mbp-10 order counts                 carried in the file, projected
                                        away before anything read them
    a London tape                       measured against New York's
                                        session, or refused for looking
                                        unusual
    the liquidation cost coefficient    pinned at 0.1 across a 1,000x range

See the CHANGELOG entry of 2026-09-22.

THE TAPES ARE BUILT SO THE ANSWER IS ARITHMETIC. Quotes are a steady
99.98 / 100.02 around a mid of exactly 100.00, so a trade at the ask is
exactly 4 bps effective; volume in the session profile is planted bucket by
bucket so `open_share` is a number the test chose rather than one it
observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.analysis import liquidity_events, microstructure
from standard_quant_tools.error import ValidationError

BASE = pd.Timestamp("2026-03-02 14:30:00")


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """A private artifact store, so a published ref belongs to one test."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _quotes(n_seconds: int = 400) -> pd.DataFrame:
    """A steady two-sided quote: mid 100.00, spread 4 bps, one per second."""
    index = pd.DatetimeIndex(
        [BASE + pd.Timedelta(seconds=s - 1) for s in range(n_seconds)]
    )
    return pd.DataFrame(
        {
            "bid_price": 99.98,
            "ask_price": 100.02,
            "bid_size": 500.0,
            "ask_size": 500.0,
        },
        index=index,
    )


def _tape(offsets) -> pd.DataFrame:
    """Alternating trades at the ask and the bid, one per offset."""
    offsets = list(offsets)
    return pd.DataFrame(
        {
            "price": [100.02 if i % 2 else 99.98 for i in range(len(offsets))],
            "size": 100.0,
        },
        index=pd.DatetimeIndex([BASE + pd.Timedelta(seconds=s) for s in offsets]),
    )


def _repeating_tape(n: int = 50, repeat_every: int = 5) -> pd.DataFrame:
    """A tape where one print in `repeat_every` shares its predecessor's
    timestamp -- 20% at the default, against 23.6% measured on a live
    session of a large-cap name."""
    offsets, seconds = [], 0
    for index in range(n):
        if index and index % repeat_every == 1:
            offsets.append(offsets[-1])
        else:
            seconds += 1
            offsets.append(seconds)
    return _tape(offsets)


def _increasing_tape(n: int = 50) -> pd.DataFrame:
    """The same tape with every stamp distinct, one second apart."""
    return _tape(range(1, n + 1))


def _publish_tape_and_quotes(trades: pd.DataFrame, run: str = "run") -> tuple:
    handoff.publish(trades, "tick_tape", run, "tape")
    handoff.publish(_quotes(), "quote_panel", run, "quotes")
    return f"sqt://tick_tape/{run}/tape", f"sqt://quote_panel/{run}/quotes"


class TestARepeatedTimestampIsJustAnotherRow:
    """The tape shape every real feed has: two prints in one nanosecond."""

    def test_every_print_on_a_repeating_tape_is_accounted_for(self, runs_dir):
        trades = _repeating_tape()
        repeated = int(trades.index.duplicated().sum())
        assert repeated == 10, "the fixture should repeat a fifth of its stamps"

        tape_ref, quote_ref = _publish_tape_and_quotes(trades)
        result = dispatch(
            "classify_trade_direction",
            {
                "tick_tape_ref": tape_ref,
                "quote_panel_ref": quote_ref,
                "run_id": "run",
                "name": "signed",
            },
        )

        assert result["n_trades"] == len(trades)
        assert (
            result["n_buys"] + result["n_sells"] + result["n_unclassified"]
            == result["n_trades"]
        ), "a print is bought, sold, or unclassified; there is no fourth state"
        assert result["method"] == "lee_ready"

    def test_the_signed_tape_it_publishes_has_one_row_per_print(self, runs_dir):
        """The failure this replaces dropped rows by label and fanned others
        out, so the published artifact was a different tape from the one
        handed in."""
        trades = _repeating_tape()
        tape_ref, quote_ref = _publish_tape_and_quotes(trades)
        result = dispatch(
            "classify_trade_direction",
            {
                "tick_tape_ref": tape_ref,
                "quote_panel_ref": quote_ref,
                "run_id": "run",
                "name": "signed",
            },
        )

        signed = handoff.resolve(result["ref"], expect="tick_tape")
        assert len(signed) == len(trades)
        assert list(signed.index) == list(trades.index)
        assert "sign" in signed.columns
        assert signed["price"].tolist() == trades["price"].tolist()

    def test_a_strictly_increasing_tape_matches_the_labelled_signs(self, runs_dir):
        """Where the labelled classification CAN be aligned, the positional
        one gives the same answer -- the fix is about which rows survive,
        not about which side they are given."""
        trades = _increasing_tape()
        assert not trades.index.duplicated().any()

        labelled = microstructure.sign_trades(trades, _quotes())
        tape_ref, quote_ref = _publish_tape_and_quotes(trades)
        result = dispatch(
            "classify_trade_direction",
            {
                "tick_tape_ref": tape_ref,
                "quote_panel_ref": quote_ref,
                "run_id": "run",
                "name": "signed",
            },
        )

        assert result["n_buys"] == int((labelled > 0).sum())
        assert result["n_sells"] == int((labelled < 0).sum())
        assert result["n_unclassified"] == len(trades) - len(labelled)


class TestTheInlineBookCanCarryAClock:
    """`timestamp` is the first column of the book contract, and the inline
    path could not express it -- so three of the result's fields were
    unreachable without registering a file."""

    @staticmethod
    def _snapshots(with_timestamp: bool):
        rows = []
        for index in range(12):
            row = {
                "bid_price_0": 99.98 + 0.01 * (index % 2),
                "ask_price_0": 100.02 + 0.01 * (index % 2),
                "bid_size_0": 500.0 + index,
                "ask_size_0": 400.0 - index,
                "bid_price_1": 99.97,
                "ask_price_1": 100.03,
                "bid_size_1": 800.0,
                "ask_size_1": 700.0,
            }
            if with_timestamp:
                row["timestamp"] = f"2026-03-02T09:30:{index:02d}.000Z"
            rows.append(row)
        return rows

    def test_iso_timestamps_give_finite_per_second_rates(self):
        result = dispatch(
            "get_order_book_metrics",
            {"snapshots": self._snapshots(True), "include_dynamics": True},
        )
        # Eleven seconds of wall clock across twelve snapshots.
        assert result["updates_per_second"] == pytest.approx(12 / 11)
        for field in ("ofi_per_second", "mid_changes_per_second"):
            assert result[field] is not None
            assert np.isfinite(result[field])

    def test_without_a_timestamp_the_rates_are_null_and_say_why(self):
        result = dispatch(
            "get_order_book_metrics",
            {"snapshots": self._snapshots(False), "include_dynamics": True},
        )
        for field in (
            "ofi_per_second",
            "updates_per_second",
            "mid_changes_per_second",
        ):
            assert result[field] is None, f"{field} should be null, not zero"
        assert any(
            "no usable timestamp span" in note for note in result["warnings"]
        ), result["warnings"]
        # The window totals still exist: only the RATES need a clock.
        assert result["ofi"] is not None
        assert result["mid_changes"] is not None


class TestTheSpreadSeriesReportsBothHalves:
    """The split is computed, published in the ref's columns, and used to be
    left out of the summary the caller reads first."""

    @staticmethod
    def _tape():
        return _tape(range(2, 120, 2))

    def test_the_two_means_are_the_published_columns_means(self, runs_dir):
        tape_ref, quote_ref = _publish_tape_and_quotes(self._tape())
        result = dispatch(
            "get_effective_spread_series",
            {
                "tick_tape_ref": tape_ref,
                "quote_panel_ref": quote_ref,
                "realized_horizon_seconds": 10,
                "run_id": "run",
                "name": "split",
            },
        )

        series = handoff.resolve(result["ref"], expect="tick_tape")
        assert result["realized_mean_bps"] == pytest.approx(
            float(series["realized_spread_bps"].dropna().mean())
        )
        assert result["impact_mean_bps"] == pytest.approx(
            float(series["price_impact_bps"].dropna().mean())
        )
        # They are the two halves of the number already reported.
        assert result["realized_mean_bps"] + result["impact_mean_bps"] == (
            pytest.approx(result["mean_bps"])
        )

    def test_without_a_horizon_both_are_null_not_zero(self, runs_dir):
        tape_ref, quote_ref = _publish_tape_and_quotes(self._tape())
        result = dispatch(
            "get_effective_spread_series",
            {
                "tick_tape_ref": tape_ref,
                "quote_panel_ref": quote_ref,
                "run_id": "run",
                "name": "unsplit",
            },
        )
        assert result["mean_bps"] is not None
        assert result["realized_mean_bps"] is None
        assert result["impact_mean_bps"] is None
        assert any("NOT" in note and "split" in note for note in result["warnings"])


class TestTheLifetimeTailIsReported:
    """A mean and a median over a distribution this skewed describe two
    different populations and neither describes the tail."""

    @staticmethod
    def _events(pairs):
        """`pairs` is (count, lifetime_seconds); one add and one cancel each.

        Timestamps are formatted with microseconds throughout: a frame
        mixing '09:30:00' and '09:30:00.010000' is parsed under one inferred
        format and the odd ones out become NaT.
        """
        rows, clock, order = [], 0.0, 0

        def _stamp(seconds: float) -> str:
            return (BASE + pd.Timedelta(seconds=seconds)).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )

        for count, lifetime in pairs:
            for _ in range(count):
                for offset, action in ((0.0, "A"), (lifetime, "C")):
                    rows.append(
                        {
                            "timestamp": _stamp(clock + offset),
                            "order_id": f"o{order}",
                            "action": action,
                            "side": "B",
                            "price": 100.0,
                            "size": 100.0,
                        }
                    )
                order += 1
                clock += lifetime + 0.5
        return rows

    def test_a_planted_tail_reaches_the_result(self):
        # Ninety-five orders at 10 ms and five at 30 s. Everything up to and
        # including p90 sits on the fast population; only p99 is inside the
        # slow one, which is the point of reporting it.
        result = dispatch(
            "get_order_event_metrics", {"events": self._events([(95, 0.01), (5, 30.0)])}
        )
        cancelled = result["cancelled"]
        assert cancelled["n"] == 100
        assert cancelled["median_seconds"] == pytest.approx(0.01)
        assert cancelled["p75_seconds"] == pytest.approx(0.01)
        assert cancelled["p90_seconds"] == pytest.approx(0.01)
        assert cancelled["p99_seconds"] == pytest.approx(30.0, rel=1e-3)

    def test_two_orders_of_magnitude_put_the_quartile_on_the_median(self):
        """Sixty orders at 10 ms and forty at 1 s: the median and the lower
        quartile are the same 10 ms, the mean is forty times either, and the
        upper quartile is the only one of the four that sees the slow
        population."""
        result = dispatch(
            "get_order_event_metrics",
            {"events": self._events([(60, 0.01), (40, 1.0)])},
        )
        cancelled = result["cancelled"]
        assert cancelled["n"] == 100
        assert cancelled["median_seconds"] == pytest.approx(0.01)
        assert cancelled["p25_seconds"] == pytest.approx(cancelled["median_seconds"])
        assert cancelled["p75_seconds"] == pytest.approx(1.0)
        assert cancelled["mean_seconds"] == pytest.approx(0.406, rel=1e-3)
        assert cancelled["mean_seconds"] > 40 * cancelled["median_seconds"]


def _book_frame(levels: int = 2, rows: int = 20, counts: bool = True):
    mid = 100.0 + np.arange(rows) * 0.001
    data = {"timestamp": pd.date_range("2026-03-02 09:30", periods=rows, freq="100ms")}
    for level in range(levels):
        offset = 0.01 * level
        data[f"bid_price_{level}"] = np.round(mid - 0.02 - offset, 4)
        data[f"ask_price_{level}"] = np.round(mid + 0.02 + offset, 4)
        data[f"bid_size_{level}"] = np.full(rows, 500.0 + 100 * level)
        data[f"ask_size_{level}"] = np.full(rows, 400.0 + 100 * level)
        if counts:
            data[f"bid_count_{level}"] = np.full(rows, 5.0 + level)
            data[f"ask_count_{level}"] = np.full(rows, 3.0 + level)
    return pd.DataFrame(data)


class TestOrderCountsAreReadable:
    """Size says how much is in front of you; the count says how many queue
    positions that is, and an mbp-10 export carries it."""

    @staticmethod
    def _inline(frame: pd.DataFrame):
        rows = frame.copy()
        rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        return rows.to_dict("records")

    def test_the_flag_reports_them_at_the_touch_and_per_level(self):
        result = dispatch(
            "get_order_book_metrics",
            {
                "snapshots": self._inline(_book_frame()),
                "include_profile": True,
                "include_order_counts": True,
            },
        )
        assert result["mean_bid_count"] == pytest.approx(5.0)
        assert result["mean_ask_count"] == pytest.approx(3.0)
        assert [level["mean_bid_count"] for level in result["profile"]] == [5.0, 6.0]
        assert [level["mean_ask_count"] for level in result["profile"]] == [3.0, 4.0]

    def test_without_the_flag_the_counts_are_absent(self):
        result = dispatch(
            "get_order_book_metrics",
            {"snapshots": self._inline(_book_frame()), "include_profile": True},
        )
        assert result["mean_bid_count"] is None
        assert result["mean_ask_count"] is None
        assert all(level["mean_bid_count"] is None for level in result["profile"])
        # Everything else is unchanged by the flag.
        assert result["mean_touch_size"] == pytest.approx(900.0)

    def test_a_feed_without_the_columns_returns_null_rather_than_refusing(self):
        result = dispatch(
            "get_order_book_metrics",
            {
                "snapshots": self._inline(_book_frame(counts=False)),
                "include_profile": True,
                "include_order_counts": True,
            },
        )
        assert result["mean_bid_count"] is None
        assert result["n_snapshots"] == 20

    def test_a_registered_book_widens_its_projection_for_them(self, runs_dir, tmp_path):
        """The reference path reads a PROJECTION of the file, and the counts
        were outside it -- so a column that was in the export could not be
        seen from a ref however the caller asked."""
        path = tmp_path / "book.parquet"
        _book_frame(rows=200).to_parquet(path, index=False)
        ref, _ = handoff.publish_external(str(path), "order_book_panel", "run", "book")

        with_counts = dispatch(
            "get_order_book_metrics",
            {"ref": ref, "include_profile": True, "include_order_counts": True},
        )
        assert with_counts["mean_bid_count"] == pytest.approx(5.0)
        assert with_counts["profile"][1]["mean_ask_count"] == pytest.approx(4.0)

        without = dispatch(
            "get_order_book_metrics", {"ref": ref, "include_profile": True}
        )
        assert without["mean_bid_count"] is None
        assert without["mean_spread_bps"] == pytest.approx(
            with_counts["mean_spread_bps"]
        )


#: Volume per bucket of a planted London day. Open-heavy, close-heavy,
#: trough in the middle: the shape `u_shaped` exists to recognize.
LONDON_SHARES = [
    0.255,
    0.09,
    0.06,
    0.05,
    0.04,
    0.035,
    0.03,
    0.035,
    0.04,
    0.05,
    0.06,
    0.09,
    0.165,
]


def _london_day(n_buckets: int = 13):
    """Five-minute bars over an LSE session, with a planted bucket profile.

    The volume in each bucket is set so `open_share` is exactly the first
    entry of LONDON_SHARES, which is what makes the assertion a planted
    answer rather than a restatement of the computation.
    """
    stamps = pd.date_range("2026-03-02 08:00", "2026-03-02 16:25", freq="5min")
    minutes = stamps.hour * 60 + stamps.minute
    low, high = int(minutes.min()), int(minutes.max())
    buckets = np.minimum(
        ((minutes - low) / (high - low) * n_buckets).astype(int), n_buckets - 1
    )
    per_bucket = np.bincount(buckets, minlength=n_buckets)
    volume = [LONDON_SHARES[b] * 1000.0 / per_bucket[b] for b in buckets]
    return [s.strftime("%Y-%m-%dT%H:%M:%S") for s in stamps], volume


class TestTheSessionBelongsToTheVenue:
    """A profile bucketed over somebody else's session measures the wrong
    end of the day, and the caller had no way to say whose session it was."""

    def test_a_london_tape_under_its_own_session_recovers_the_shape(self):
        timestamps, volume = _london_day()
        result = dispatch(
            "get_intraday_volume_profile",
            {
                "volume": volume,
                "timestamps": timestamps,
                "index_timezone": "Europe/London",
                "exchange_timezone": "Europe/London",
                "session_start": "08:00",
                "session_end": "16:30",
            },
        )
        assert result["open_share"] == pytest.approx(0.255, abs=0.005)
        assert result["close_share"] == pytest.approx(0.165, abs=0.005)
        assert result["u_shaped"] is True
        assert result["extended_hours_share"] == pytest.approx(0.0)
        assert result["session"] == ["08:00", "16:30"]
        assert result["n_bars"] == len(volume)

    def test_the_new_york_default_refuses_and_names_the_session(self):
        timestamps, volume = _london_day()
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "get_intraday_volume_profile",
                {
                    "volume": volume,
                    "timestamps": timestamps,
                    "index_timezone": "Europe/London",
                },
            )
        message = str(exc.value)
        assert "09:30-16:00" in message
        assert "America/New_York" in message
        assert "exchange_timezone" in message

    def test_an_unknown_zone_is_refused_by_name(self):
        timestamps, volume = _london_day()
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch(
                "get_intraday_volume_profile",
                {
                    "volume": volume,
                    "timestamps": timestamps,
                    "exchange_timezone": "Nowhere/City",
                },
            )
        assert "Nowhere/City" in str(exc.value)
        assert "IANA" in str(exc.value)

    def test_a_session_time_that_is_not_hh_mm_is_refused(self):
        timestamps, volume = _london_day()
        with pytest.raises(pydantic.ValidationError) as exc:
            dispatch(
                "get_intraday_volume_profile",
                {
                    "volume": volume,
                    "timestamps": timestamps,
                    "exchange_timezone": "Europe/London",
                    "index_timezone": "Europe/London",
                    "session_start": "9:3",
                    "session_end": "16:30",
                },
            )
        assert "9:3" in str(exc.value)
        assert "09:30" in str(exc.value)


class TestTheLiquidationCostCoefficientIsTheCallers:
    """One pinned model parameter was setting the scale of a reported cost."""

    BOOK = {
        "positions": {"AAPL": 5_000_000.0, "THIN": 2_000_000.0},
        "volatilities": {"AAPL": 0.32, "THIN": 0.55},
        "daily_volumes": {"AAPL": 40_000_000.0, "THIN": 3_000_000.0},
    }

    def _run(self, **extra):
        return dispatch("get_liquidity_adjusted_var", {**self.BOOK, **extra})

    def test_the_coefficient_moves_the_cost_a_thousandfold(self):
        low = self._run(impact_coefficient=0.01)
        high = self._run(impact_coefficient=10.0)
        assert high["expected_liquidation_cost"] == pytest.approx(
            1000.0 * low["expected_liquidation_cost"], rel=1e-9
        )

    def test_the_risk_number_itself_does_not_move(self):
        """It is a COST, not a risk: it is reported beside the quantile
        rather than inside it, and nothing about the holding period
        changes."""
        low = self._run(impact_coefficient=0.01)
        high = self._run(impact_coefficient=10.0)
        for field in ("naive_var", "liquidity_adjusted_var"):
            assert low[field] == pytest.approx(high[field])

    def test_the_default_is_the_librarys_own_and_changes_nothing(self):
        assert self._run()["expected_liquidation_cost"] == pytest.approx(
            self._run(impact_coefficient=0.1)["expected_liquidation_cost"]
        )

    def test_an_out_of_range_coefficient_is_refused(self):
        with pytest.raises(pydantic.ValidationError):
            self._run(impact_coefficient=-1.0)


class TestOneChannelList:
    """There were two ways to ask which channels exist, and the shorter one
    answered that a declared channel had never been heard of."""

    def test_the_computable_subset_is_no_longer_its_own_function(self):
        assert not hasattr(liquidity_events, "available_channels")
        assert "available_channels" not in liquidity_events.__all__

    def test_the_depth_channels_are_declared_and_say_what_they_need(self):
        declared = liquidity_events.declared_channels()
        for name in ("ofi", "book_imbalance", "l5_imbalance", "depth_slope"):
            assert name in declared, f"{name} should be named, not omitted"
            channel = liquidity_events.CHANNELS[name]
            assert not channel.available
            assert "order book" in channel.why_unavailable()

    def test_the_channels_that_run_on_trades_and_quotes_are_available(self):
        for name in ("spread", "signed_volume", "mid_return"):
            assert liquidity_events.CHANNELS[name].available
