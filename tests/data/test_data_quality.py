"""Tests for data/quality.py — missing-bar/stale-price/price-jump detection on synthetic OHLCV."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import DataQualityReportInput
from standard_quant_tools.agent.runtimes.research.tools import (
    get_data_quality_report,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.data.quality import (
    SAMPLE_FEEDS,
    detect_duplicate_timestamps,
    detect_missing_bars,
    detect_ohlc_inconsistencies,
    detect_out_of_order_timestamps,
    detect_price_jumps,
    detect_sample_feed,
    detect_stale_prices,
    detect_volume_anomalies,
)


def _ohlcv(closes, dates):
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000.0] * len(closes),
        },
        index=dates,
    )


class TestDetectMissingBars:
    def test_no_gaps_in_dense_business_day_series(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        df = _ohlcv([100.0] * 10, dates)
        assert detect_missing_bars(df) == []

    def test_planted_gap_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        # Drop the 5th business day (index 4) to create a gap.
        gapped = dates.delete(4)
        df = _ohlcv([100.0] * 9, gapped)
        gaps = detect_missing_bars(df)
        assert len(gaps) == 1
        assert gaps[0]["date"] == str(dates[4].date())

    def test_fewer_than_two_rows_returns_empty(self):
        dates = pd.bdate_range("2023-01-02", periods=1)
        df = _ohlcv([100.0], dates)
        assert detect_missing_bars(df) == []

    def test_the_span_is_the_earliest_to_the_latest_bar_not_the_first_to_last_row(
        self,
    ):
        """On an index out of order, the first row is not the earliest bar:
        the span between the first and last rows ran backwards and was
        empty, so a real gap went unreported."""
        dates = pd.bdate_range("2023-01-02", periods=10)
        gapped = dates.delete(4)
        shuffled = gapped[::-1]  # the latest bar first
        df = _ohlcv([100.0] * len(shuffled), shuffled)
        assert [g["date"] for g in detect_missing_bars(df)] == [str(dates[4].date())]


class TestDetectStalePrices:
    def test_no_stale_run_below_threshold(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = _ohlcv([100.0, 101.0, 100.0, 102.0, 101.0], dates)
        assert detect_stale_prices(df, n=3) == []

    def test_planted_stale_run_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=6)
        df = _ohlcv([100.0, 105.0, 105.0, 105.0, 105.0, 110.0], dates)
        runs = detect_stale_prices(df, n=3)
        assert len(runs) == 1
        assert runs[0]["run_length"] == 4
        assert runs[0]["price"] == pytest.approx(105.0)
        assert runs[0]["start"] == str(dates[1].date())
        assert runs[0]["end"] == str(dates[4].date())

    def test_run_shorter_than_n_not_flagged(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = _ohlcv([100.0, 105.0, 105.0, 110.0, 115.0], dates)
        assert detect_stale_prices(df, n=3) == []

    def test_empty_dataframe_returns_empty(self):
        df = _ohlcv([], pd.DatetimeIndex([]))
        assert detect_stale_prices(df) == []


class TestDetectPriceJumps:
    def test_no_jump_below_threshold(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        df = _ohlcv([100.0, 105.0, 108.0], dates)
        assert detect_price_jumps(df, threshold=0.15) == []

    def test_planted_jump_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        df = _ohlcv([100.0, 100.0, 50.0], dates)  # -50% single-bar move
        jumps = detect_price_jumps(df, threshold=0.15)
        assert len(jumps) == 1
        assert jumps[0]["date"] == str(dates[2].date())
        assert jumps[0]["pct_change"] == pytest.approx(-0.5)

    def test_fewer_than_two_rows_returns_empty(self):
        dates = pd.bdate_range("2023-01-02", periods=1)
        df = _ohlcv([100.0], dates)
        assert detect_price_jumps(df) == []


def _tape(rows: int = 60) -> pd.DataFrame:
    """A consolidated-looking daily frame with realistic volume noise."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2024-01-02", periods=rows)
    close = 100.0 + np.cumsum(rng.normal(0, 1, rows))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(30_000_000, 40_000_000, rows).astype(float),
        },
        index=index,
    )
    if rows > 40:
        frame.iloc[40, frame.columns.get_loc("Volume")] = 0.0  # a halted session
    return frame


class TestASampleFeedIsAQuestionOfProvenance:
    """
    The volume check compares each bar with the frame's own trailing median,
    so it cannot see a feed that is thin on every bar. That is pinned here
    as the limitation it is; the answer comes from the dataset the provider
    stamped on the frame.
    """

    def test_the_volume_check_is_blind_to_scale(self):
        tape = _tape()
        sample = tape.assign(Volume=tape["Volume"] * 0.036)
        on_tape = [(a["date"], a["kind"]) for a in detect_volume_anomalies(tape)]
        on_sample = [(a["date"], a["kind"]) for a in detect_volume_anomalies(sample)]
        assert on_tape == on_sample == [(str(tape.index[40].date()), "zero")]

    def test_a_frame_stamped_with_the_sample_feed_is_flagged(self):
        frame = _tape()
        frame.attrs.update({"dataset": "EQUS.MINI", "provider": "databento"})
        found = detect_sample_feed(frame)
        assert found == {
            "dataset": "EQUS.MINI",
            "provider": "databento",
            "note": SAMPLE_FEEDS["EQUS.MINI"],
        }

    @pytest.mark.parametrize(
        "attrs",
        [
            {"dataset": "EQUS.SUMMARY", "provider": "databento"},
            {"dataset": "XNAS.ITCH"},
            {},
            {"dataset": None},
        ],
        ids=["summary-feed", "venue-feed", "no-stamp", "null-stamp"],
    )
    def test_anything_else_is_not_known_to_be_a_sample(self, attrs):
        frame = _tape()
        frame.attrs.update(attrs)
        assert detect_sample_feed(frame) is None


class TestIndexIntegrity:
    def test_a_repeated_label_is_reported_with_every_position(self):
        index = pd.bdate_range("2024-01-02", periods=6)
        repeated = index.insert(3, index[2])
        frame = _ohlcv([100.0] * 7, repeated)
        assert detect_duplicate_timestamps(frame) == [
            {"timestamp": str(index[2].date()), "count": 2, "positions": [2, 3]}
        ]

    def test_swapped_labels_are_reported_where_time_runs_backwards(self):
        index = list(pd.bdate_range("2024-01-02", periods=6))
        index[2], index[3] = index[3], index[2]
        frame = _ohlcv([100.0] * 6, pd.DatetimeIndex(index))
        assert detect_out_of_order_timestamps(frame) == [
            {
                "position": 3,
                "timestamp": str(index[3].date()),
                "previous": str(index[2].date()),
            }
        ]
        assert detect_duplicate_timestamps(frame) == []

    def test_intraday_labels_keep_their_time_of_day(self):
        index = pd.DatetimeIndex(
            ["2024-01-02 14:30", "2024-01-02 14:31", "2024-01-02 14:31"]
        )
        frame = _ohlcv([100.0] * 3, index)
        (entry,) = detect_duplicate_timestamps(frame)
        assert entry["timestamp"] == "2024-01-02T14:31:00"

    def test_a_clean_frame_has_neither(self):
        frame = _ohlcv([100.0] * 30, pd.bdate_range("2024-01-02", periods=30))
        assert detect_duplicate_timestamps(frame) == []
        assert detect_out_of_order_timestamps(frame) == []


def _bars(open_, high, low, close):
    index = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": [1_000.0] * len(close),
        },
        index=index,
    )


class TestOhlcConsistency:
    def test_low_above_high_is_reported_once_at_its_row(self):
        frame = _bars(
            [100.0, 100.0, 100.0],
            [101.0, 101.0, 101.0],
            [99.0, 151.5, 99.0],
            [100.0] * 3,
        )
        found = detect_ohlc_inconsistencies(frame)
        assert [(f["position"], f["kind"]) for f in found] == [(1, "low_above_high")]
        assert found[0]["low"] == 151.5 and found[0]["high"] == 101.0
        assert found[0]["date"] == str(frame.index[1].date())

    def test_an_open_or_close_outside_the_range_is_reported(self):
        frame = _bars(
            [100.0, 103.0, 100.0],
            [101.0, 101.0, 101.0],
            [99.0, 99.0, 99.0],
            [100.0, 100.0, 98.0],
        )
        found = detect_ohlc_inconsistencies(frame)
        assert [(f["position"], f["kind"]) for f in found] == [
            (1, "open_outside_range"),
            (2, "close_outside_range"),
        ]

    def test_a_clean_frame_gives_nothing(self):
        assert detect_ohlc_inconsistencies(_tape()) == []

    def test_a_bar_where_all_four_prices_are_equal_is_not_flagged(self):
        flat = _bars([100.0] * 4, [100.0] * 4, [100.0] * 4, [100.0] * 4)
        assert detect_ohlc_inconsistencies(flat) == []

    def test_floating_point_noise_in_a_scaled_price_is_not_flagged(self):
        frame = _bars([100.0], [100.0], [99.0], [100.0 * (1 + 1e-13)])
        assert detect_ohlc_inconsistencies(frame) == []

    def test_a_frame_without_high_and_low_gives_nothing(self):
        frame = pd.DataFrame(
            {"Close": [1.0, 2.0]}, index=pd.bdate_range("2024-01-02", periods=2)
        )
        assert detect_ohlc_inconsistencies(frame) == []


# ── The report carries every check ────────────────────────────────────────────


def _report(frame: pd.DataFrame):
    provider = MagicMock()
    provider.get_ohlcv.return_value = frame
    provider.get_metadata.return_value = DataSetMetadata(
        provider="databento",
        adjusted=False,
        survivorship_free=True,
        point_in_time=False,
        frequency="1d",
        timezone="UTC",
    )
    with patch.object(DataFactory, "get_provider", return_value=provider):
        return get_data_quality_report(
            DataQualityReportInput(
                symbol="AAPL", start_date="2024-01-02", end_date="2024-03-29"
            )
        )


class TestTheReportCarriesTheNewChecks:
    def test_a_sample_feed_frame_is_named_in_the_report(self):
        frame = _tape()
        frame.attrs.update({"dataset": "EQUS.MINI", "provider": "databento"})
        result = _report(frame)
        assert result.served_dataset == "EQUS.MINI"
        assert result.sample_feed is True
        assert result.sample_feed_note == SAMPLE_FEEDS["EQUS.MINI"]

    def test_the_summary_feed_is_served_and_not_a_sample(self):
        frame = _tape()
        frame.attrs.update({"dataset": "EQUS.SUMMARY", "provider": "databento"})
        result = _report(frame)
        assert result.served_dataset == "EQUS.SUMMARY"
        assert result.sample_feed is False
        assert result.sample_feed_note is None

    def test_integrity_findings_reach_the_report(self):
        frame = _tape(10)
        index = list(frame.index)
        index[5], index[6] = index[6], index[5]
        index[8] = index[7]
        frame.index = pd.DatetimeIndex(index)
        frame.iloc[2, frame.columns.get_loc("Low")] = frame["High"].iloc[2] * 1.5
        result = _report(frame)
        assert [d.positions for d in result.duplicate_timestamps] == [[7, 8]]
        assert [o.position for o in result.out_of_order_timestamps] == [6]
        assert [(o.position, o.kind) for o in result.ohlc_inconsistencies] == [
            (2, "low_above_high")
        ]

    def test_a_clean_frame_reports_none_of_them(self):
        result = _report(_tape())
        assert result.served_dataset is None
        assert result.sample_feed is False
        assert result.duplicate_timestamps == []
        assert result.out_of_order_timestamps == []
        assert result.ohlc_inconsistencies == []
