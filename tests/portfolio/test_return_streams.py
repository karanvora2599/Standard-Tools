"""
Dated return streams meet on the dates they share, and say what that cost.

Two return series of the same length look aligned and need not be: one a
week later than the other, one with another market's holidays. Read row by
row they pair each day with some other day. `align_return_streams` keeps the
dates every stream has, reports per stream what it supplied and what it lost
-- before the common window, after it, inside it -- and refuses what cannot
be placed in time or cannot be returns. `combine_return_streams` publishes
the aligned panel as a `returns_panel` reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.streams import (
    MAX_PLAUSIBLE_TYPICAL_RETURN,
    align_return_streams,
    alignment_notes,
    combine_return_streams,
    refuse_if_not_returns,
    streams_from_panel,
)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _two_calendars():
    """
    Two 10-return streams of the same length on different calendars.

    `us` trades 2024-01-02..2024-01-15 on business days (10 days). `eu` has
    the same count but starts two business days later and skips 2024-01-08:
    2024-01-04..2024-01-18 less one day. Read by position the two are
    "aligned"; by date they share exactly 7 days.
    """
    us_dates = pd.bdate_range("2024-01-02", periods=10)
    eu_dates = pd.bdate_range("2024-01-04", periods=11).drop(pd.Timestamp("2024-01-08"))
    us = pd.Series(np.arange(1, 11) / 1000.0, index=us_dates)
    eu = pd.Series(-np.arange(1, 11) / 1000.0, index=eu_dates)
    return us, eu


class TestTwoCalendarsMeetOnTheirTrueIntersection:
    def test_equal_lengths_align_on_the_dates_both_have(self):
        """
        Planted: `us` covers Jan 2-15, `eu` Jan 4-18 without Jan 8. The
        shared dates are Jan 4, 5, 9, 10, 11, 12, 15 -- seven -- and each
        value on them is the stream's own, not its neighbour by position.
        """
        us, eu = _two_calendars()
        assert len(us) == len(eu) == 10
        frame, report = align_return_streams({"us": us, "eu": eu}, min_overlap=5)
        shared = pd.to_datetime(
            [
                "2024-01-04",
                "2024-01-05",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-15",
            ]
        )
        assert list(frame.index) == list(shared)
        assert list(frame.columns) == ["us", "eu"]
        assert frame["us"].tolist() == us.loc[shared].tolist()
        assert frame["eu"].tolist() == eu.loc[shared].tolist()
        assert report["n_dates"] == 7
        assert report["n_dates_any"] == 13
        assert (report["first_date"], report["last_date"]) == (
            "2024-01-04",
            "2024-01-15",
        )

    def test_the_report_names_what_each_stream_lost_and_where(self):
        """
        `us` loses Jan 2 and 3 before the window (it starts first) and Jan 8
        inside it (eu lacks it); `eu` loses Jan 16, 17 and 18 after the
        window (it ends last). The three drop counts sum to what was
        supplied.
        """
        us, eu = _two_calendars()
        _frame, report = align_return_streams({"us": us, "eu": eu}, min_overlap=5)
        assert report["window_start"] == "2024-01-04"
        assert report["window_end"] == "2024-01-15"
        assert report["start_set_by"] == ["eu"]
        assert report["end_set_by"] == ["us"]
        assert report["by_stream"]["us"] == {
            "n_supplied": 10,
            "first_date": "2024-01-02",
            "last_date": "2024-01-15",
            "n_kept": 7,
            "n_dropped_before": 2,
            "n_dropped_after": 0,
            "n_dropped_inside": 1,
        }
        assert report["by_stream"]["eu"] == {
            "n_supplied": 10,
            "first_date": "2024-01-04",
            "last_date": "2024-01-18",
            "n_kept": 7,
            "n_dropped_before": 0,
            "n_dropped_after": 3,
            "n_dropped_inside": 0,
        }
        (note,) = alignment_notes(report, "probe")
        assert "{'us': 3, 'eu': 3}" in note

    def test_read_by_position_the_same_values_pair_other_days(self):
        """Why the dates matter: by position, row 0 pairs Jan 2 with Jan 4.
        Aligned by date no row pairs two different days."""
        us, eu = _two_calendars()
        frame, _report = align_return_streams({"us": us, "eu": eu}, min_overlap=1)
        by_position = pd.DataFrame({"us": us.to_numpy(), "eu": eu.to_numpy()})
        assert not np.allclose(by_position["us"].to_numpy()[:7], frame["us"].to_numpy())

    def test_an_unsorted_stream_is_read_in_date_order(self):
        us, eu = _two_calendars()
        frame, _ = align_return_streams({"us": us.iloc[::-1], "eu": eu}, min_overlap=1)
        assert frame.index.is_monotonic_increasing
        assert frame["us"].tolist() == us.loc[frame.index].tolist()


class TestWhatIsRefused:
    def test_levels_passed_as_returns_are_refused(self):
        """An equity curve near 100 is a level series: its typical |value| is
        a 10,000% move a period. Refused by the one rule, with the remedy."""
        dates = pd.bdate_range("2024-01-02", periods=30)
        curve = pd.Series(100.0 + np.arange(30), index=dates)
        returns = pd.Series(np.full(30, 0.001), index=dates)
        with pytest.raises(ValidationError) as refusal:
            align_return_streams({"fund": curve, "r": returns}, min_overlap=5)
        message = str(refusal.value)
        assert "stream 'fund'" in message
        assert "level series" in message
        assert "pct_change" in message

    def test_a_short_overlap_is_refused_naming_it_and_the_limiting_stream(self):
        """
        Planted: `a` and `b` share 60 dates; `late` starts on the 56th, so
        all three share only 5. Leaving `late` out would share 60, so it is
        the stream named, with its range.
        """
        dates = pd.bdate_range("2024-01-01", periods=60)
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(0, 0.01, 60), index=dates)
        b = pd.Series(rng.normal(0, 0.01, 60), index=dates)
        late = pd.Series(
            rng.normal(0, 0.01, 30), index=pd.bdate_range(dates[55], periods=30)
        )
        with pytest.raises(ValidationError) as refusal:
            align_return_streams({"a": a, "b": b, "late": late}, min_overlap=20)
        message = str(refusal.value)
        assert "share 5 date(s), fewer than min_overlap=20" in message
        assert "['late'] limit(s) it most: without it 60 dates" in message
        assert "its start set by ['late']" in message
        assert str(dates[55].date()) in message

    def test_streams_with_no_common_window_say_so(self):
        first = pd.Series([0.01] * 5, index=pd.bdate_range("2024-01-01", periods=5))
        second = pd.Series([0.01] * 5, index=pd.bdate_range("2024-03-01", periods=5))
        with pytest.raises(ValidationError, match="cover no common window"):
            align_return_streams({"x": first, "y": second}, min_overlap=1)

    def test_one_short_stream_is_named(self):
        one = pd.Series([0.01] * 3, index=pd.bdate_range("2024-01-01", periods=3))
        with pytest.raises(ValidationError, match="stream 'one' has 3 dated"):
            align_return_streams({"one": one}, min_overlap=20)

    def test_a_repeated_date_is_refused(self):
        index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-03"])
        with pytest.raises(ValidationError, match="names the same date twice"):
            align_return_streams(
                {"s": pd.Series([0.01, 0.02, 0.03], index=index)}, min_overlap=1
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_missing_or_non_finite_value_is_refused(self, bad):
        index = pd.bdate_range("2024-01-02", periods=3)
        with pytest.raises(ValidationError, match="missing or non-finite value"):
            align_return_streams(
                {"s": pd.Series([0.01, bad, 0.03], index=index)}, min_overlap=1
            )

    @pytest.mark.parametrize(
        "undated",
        [
            [0.01, 0.02, 0.03],
            np.array([0.01, 0.02, 0.03]),
            pd.Series([0.01, 0.02, 0.03]),
        ],
    )
    def test_undated_values_are_not_a_stream(self, undated):
        """A list, an array and a positionally indexed Series all carry
        values without dates, so their alignment cannot be checked."""
        dated = pd.Series([0.01] * 3, index=pd.bdate_range("2024-01-02", periods=3))
        with pytest.raises(ValidationError, match="Undated values are not a stream"):
            align_return_streams({"dated": dated, "undated": undated}, min_overlap=1)

    def test_streams_in_different_time_zones_are_refused(self):
        index = pd.bdate_range("2024-01-02", periods=3)
        with pytest.raises(ValidationError, match="different time zones"):
            align_return_streams(
                {
                    "naive": pd.Series([0.01] * 3, index=index),
                    "utc": pd.Series([0.01] * 3, index=index.tz_localize("UTC")),
                },
                min_overlap=1,
            )

    @pytest.mark.parametrize("bad", [0, -3, 2.5, True])
    def test_min_overlap_is_a_positive_whole_number(self, bad):
        dated = pd.Series([0.01] * 3, index=pd.bdate_range("2024-01-02", periods=3))
        with pytest.raises(ValidationError, match="min_overlap"):
            align_return_streams({"s": dated}, min_overlap=bad)

    def test_no_streams_and_an_empty_stream_are_refused(self):
        with pytest.raises(ValidationError, match="no streams"):
            align_return_streams({}, min_overlap=1)
        with pytest.raises(ValidationError, match="no observations"):
            align_return_streams(
                {"e": pd.Series([], index=pd.DatetimeIndex([]), dtype=float)},
                min_overlap=1,
            )


class TestTheRuleThatRefusesLevels:
    def test_it_is_one_rule_reachable_from_its_old_home(self):
        """The rule moved into the library; the data runtime's names still
        import and are the same objects."""
        from standard_quant_tools.agent.runtimes.data import models

        assert models._refuse_if_not_returns is refuse_if_not_returns
        assert models._MAX_PLAUSIBLE_TYPICAL_RETURN == MAX_PLAUSIBLE_TYPICAL_RETURN

    def test_an_all_positive_money_market_series_is_not_refused(self):
        refuse_if_not_returns(pd.Series([0.0002] * 50), "probe")


class TestAPanelIsReadAsStreams:
    def test_a_blank_cell_is_a_date_that_column_lacks(self):
        """Planted: B is blank on the first two of five dates. B's stream
        is three dates long and aligning keeps those three."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        panel = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05], "B": [np.nan, np.nan, 0.1, 0.2, 0.3]},
            index=dates,
        )
        streams = streams_from_panel(panel, what="probe")
        assert len(streams["A"]) == 5 and len(streams["B"]) == 3
        frame, report = align_return_streams(streams, min_overlap=1)
        assert list(frame.index) == list(dates[2:])
        assert report["by_stream"]["A"]["n_dropped_before"] == 2

    def test_iso_labels_are_dates_and_positions_are_not(self):
        panel = pd.DataFrame({"A": [0.01, 0.02]}, index=["2024-01-02", "2024-01-03"])
        (series,) = streams_from_panel(panel, what="probe").values()
        assert isinstance(series.index, pd.DatetimeIndex)
        with pytest.raises(ValidationError, match="Undated values are not a stream"):
            streams_from_panel(pd.DataFrame({"A": [0.01, 0.02]}), what="probe")
        # Positions held as objects are still positions, not instants from 1970.
        with pytest.raises(ValidationError, match="Undated values are not a stream"):
            streams_from_panel(
                pd.DataFrame({"A": [0.01, 0.02]}, index=pd.Index([0, 1], dtype=object)),
                what="probe",
            )

    def test_a_column_that_is_not_there_is_named(self):
        panel = pd.DataFrame(
            {"A": [0.01, 0.02]}, index=pd.bdate_range("2024-01-02", periods=2)
        )
        with pytest.raises(ValidationError, match=r"no column \['Z'\]"):
            streams_from_panel(panel, what="probe", columns=["Z"])


class TestCombineReturnStreams:
    def test_the_published_panel_resolves_to_the_aligned_frame_exactly(self, runs_dir):
        """Planted: the two-calendar streams. The reference resolves to the
        seven shared dates and both streams' own values, bit for bit."""
        us, eu = _two_calendars()
        out = combine_return_streams(
            {"us": us, "eu": eu}, run_id="combine", min_overlap=5
        )
        expected, report = align_return_streams({"us": us, "eu": eu}, min_overlap=5)
        assert out["returns_ref"] == "sqt://returns_panel/combine/streams"
        assert out["columns"] == ["us", "eu"]
        assert out["alignment"] == report
        resolved = handoff.resolve(out["returns_ref"], expect="returns_panel")
        pd.testing.assert_frame_equal(resolved, expected, check_freq=False)
        assert out["warnings"] and "dropped" in out["warnings"][0]

    def test_publishing_twice_at_one_run_id_is_refused(self, runs_dir):
        us, eu = _two_calendars()
        combine_return_streams({"us": us, "eu": eu}, run_id="twice", min_overlap=5)
        with pytest.raises(ValidationError, match="already published"):
            combine_return_streams({"us": us, "eu": eu}, run_id="twice", min_overlap=5)

    def test_stream_specs_resolve_beside_series(self, runs_dir):
        """A spec of dated values and a column of a published panel resolve
        to the same dated series as the Series they stand for."""
        us, eu = _two_calendars()
        panel_ref = handoff.publish(
            pd.DataFrame({"eu": eu, "other": eu * 2}),
            kind="returns_panel",
            run_id="src",
            name="panel",
        )
        out = combine_return_streams(
            {
                "us": {"values": {str(d.date()): float(v) for d, v in us.items()}},
                "eu": {"ref": panel_ref, "column": "eu"},
            },
            run_id="specs",
            min_overlap=5,
        )
        expected, _report = align_return_streams({"us": us, "eu": eu}, min_overlap=5)
        resolved = handoff.resolve(out["returns_ref"], expect="returns_panel")
        pd.testing.assert_frame_equal(resolved, expected, check_freq=False)

    def test_undated_values_in_a_spec_are_refused_with_the_reason(self, runs_dir):
        us, eu = _two_calendars()
        with pytest.raises(ValueError, match="undated values are not a stream"):
            combine_return_streams(
                {"us": {"values": us.tolist()}, "eu": {"values": eu.tolist()}},
                run_id="undated",
                min_overlap=5,
            )
