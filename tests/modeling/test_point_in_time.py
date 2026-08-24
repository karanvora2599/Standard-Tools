"""
The point-in-time join.

Every check here is a leakage check wearing different clothes. The whole
value of the module is that a feature at t sees only what was AVAILABLE at
t — not what was true at t, and not the final restated version of it — so
the tests are built around records whose event time, first availability and
revision are deliberately far apart, and they assert what each panel row is
allowed to have seen.

The worked example throughout is an earnings figure:

    period_end   2026-06-30   the quarter it describes
    reported_at  2026-07-29   when anyone could act on it
    revised_at   2026-08-14   when the number changed

A join on period_end reads it a month early. A join that takes the latest
value reads the revision two weeks early. Both are ordinary-looking joins.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.point_in_time import (
    asof_join,
    coverage_report,
    validate_pit_frame,
)


def _panel(dates, entities=("AAA", "BBB")):
    rows = [(pd.Timestamp(d), e) for d in dates for e in entities]
    return pd.DataFrame(rows, columns=["date", "entity"])


EARNINGS = pd.DataFrame(
    [
        # entity, event_time (quarter end), available_time, eps
        ("AAA", "2026-06-30", "2026-07-29", 1.50),  # first report
        ("AAA", "2026-06-30", "2026-08-14", 1.42),  # restated two weeks later
        ("AAA", "2026-09-30", "2026-10-28", 1.61),
        ("BBB", "2026-06-30", "2026-07-31", 0.90),
    ],
    columns=["entity", "event_time", "available_time", "eps"],
)


class TestValidation:
    def test_missing_columns_are_named(self):
        frame = pd.DataFrame({"entity": ["AAA"], "event_time": ["2026-06-30"]})
        with pytest.raises(ValidationError, match="available_time"):
            validate_pit_frame(frame)

    def test_availability_before_the_event_is_rejected(self):
        """
        The substantive check. A record available before the period it
        describes has ended is the two columns swapped, and left alone it
        makes every model built on the data look prescient.
        """
        frame = pd.DataFrame(
            {
                "entity": ["AAA"],
                "event_time": ["2026-07-29"],
                "available_time": ["2026-06-30"],
            }
        )
        with pytest.raises(ValidationError, match="available BEFORE"):
            validate_pit_frame(frame)

    def test_unparseable_timestamps_are_rejected(self):
        frame = pd.DataFrame(
            {
                "entity": ["AAA"],
                "event_time": ["2026-06-30"],
                "available_time": ["not a date"],
            }
        )
        with pytest.raises(ValidationError, match="unparseable or missing"):
            validate_pit_frame(frame)

    def test_a_global_series_needs_no_entity(self):
        frame = pd.DataFrame(
            {"event_time": ["2026-06-30"], "available_time": ["2026-07-15"]}
        )
        assert len(validate_pit_frame(frame, require_entity=False)) == 1

    def test_timestamps_come_back_as_datetimes(self):
        out = validate_pit_frame(EARNINGS)
        assert pd.api.types.is_datetime64_any_dtype(out["available_time"])
        assert pd.api.types.is_datetime64_any_dtype(out["event_time"])


class TestAsofJoin:
    def test_a_record_is_invisible_before_it_was_published(self):
        """The headline. The quarter ended 2026-06-30; nobody had the number
        until 2026-07-29."""
        panel = _panel(["2026-07-01", "2026-07-15", "2026-07-28", "2026-07-29"])
        out = asof_join(panel, EARNINGS, fields=["eps"])
        aaa = out[out["entity"] == "AAA"].set_index("date")["eps"]
        assert pd.isna(aaa["2026-07-01"])
        assert pd.isna(aaa["2026-07-15"])
        assert pd.isna(aaa["2026-07-28"])
        assert aaa["2026-07-29"] == 1.50

    def test_a_revision_is_invisible_before_it_was_made(self):
        """
        The subtler half. Joining on availability but taking the LATEST value
        would show 1.42 from the first report onward. Reproducing a decision
        means seeing the number as it was, mistake included.
        """
        panel = _panel(["2026-08-01", "2026-08-13", "2026-08-14", "2026-09-01"])
        out = asof_join(panel, EARNINGS, fields=["eps"])
        aaa = out[out["entity"] == "AAA"].set_index("date")["eps"]
        assert aaa["2026-08-01"] == 1.50, "saw the revision before it was made"
        assert aaa["2026-08-13"] == 1.50, "saw the revision before it was made"
        assert aaa["2026-08-14"] == 1.42
        assert aaa["2026-09-01"] == 1.42

    def test_availability_on_the_bar_itself_counts(self):
        panel = _panel(["2026-07-29"])
        out = asof_join(panel, EARNINGS, fields=["eps"])
        assert out[out["entity"] == "AAA"]["eps"].iloc[0] == 1.50

    def test_entities_do_not_see_each_others_records(self):
        panel = _panel(["2026-07-30"])
        out = asof_join(panel, EARNINGS, fields=["eps"]).set_index("entity")["eps"]
        assert out["AAA"] == 1.50
        assert pd.isna(out["BBB"]), "BBB's figure was not out until 2026-07-31"

    def test_the_next_quarter_supersedes_the_previous_one(self):
        panel = _panel(["2026-10-27", "2026-10-28"])
        out = asof_join(panel, EARNINGS, fields=["eps"])
        aaa = out[out["entity"] == "AAA"].set_index("date")["eps"]
        assert aaa["2026-10-27"] == 1.42
        assert aaa["2026-10-28"] == 1.61

    def test_an_unsorted_panel_still_joins_correctly(self):
        """
        merge_asof needs both sides sorted and does NOT raise when they are
        not — it produces wrong matches. So the join sorts rather than
        trusting, and this is the test that would catch removing that.
        """
        panel = _panel(["2026-09-01", "2026-07-01", "2026-08-14", "2026-07-29"])
        shuffled = panel.iloc[np.random.default_rng(0).permutation(len(panel))]
        out = asof_join(shuffled.reset_index(drop=True), EARNINGS, fields=["eps"])
        aaa = out[out["entity"] == "AAA"].set_index("date")["eps"]
        assert pd.isna(aaa["2026-07-01"])
        assert aaa["2026-07-29"] == 1.50
        assert aaa["2026-08-14"] == 1.42
        assert aaa["2026-09-01"] == 1.42

    def test_row_order_and_count_are_preserved(self):
        panel = _panel(["2026-07-01", "2026-08-14"])
        out = asof_join(panel, EARNINGS, fields=["eps"])
        assert len(out) == len(panel)
        pd.testing.assert_frame_equal(
            out[["date", "entity"]], panel[["date", "entity"]]
        )

    def test_prefix_namespaces_the_joined_columns(self):
        panel = _panel(["2026-08-14"])
        out = asof_join(panel, EARNINGS, fields=["eps"], prefix="fundamental.")
        assert "fundamental.eps" in out.columns and "eps" not in out.columns

    def test_a_colliding_column_is_refused(self):
        panel = _panel(["2026-08-14"]).assign(eps=0.0)
        with pytest.raises(ValidationError, match="would overwrite"):
            asof_join(panel, EARNINGS, fields=["eps"])

    def test_unknown_field_is_refused(self):
        with pytest.raises(ValidationError, match="no column"):
            asof_join(_panel(["2026-08-14"]), EARNINGS, fields=["revenue"])


class TestGlobalSeries:
    RELEASES = pd.DataFrame(
        [
            # the June CPI print describes June and lands in mid-July
            ("2026-06-30", "2026-07-14", 3.1),
            ("2026-07-31", "2026-08-12", 2.9),
        ],
        columns=["event_time", "available_time", "cpi"],
    )

    def test_a_release_reaches_every_entity_from_its_release_date(self):
        panel = _panel(["2026-07-13", "2026-07-14"])
        out = asof_join(panel, self.RELEASES, fields=["cpi"], by_entity=False)
        by_date = out.set_index(["date", "entity"])["cpi"]
        assert pd.isna(by_date[(pd.Timestamp("2026-07-13"), "AAA")])
        assert by_date[(pd.Timestamp("2026-07-14"), "AAA")] == 3.1
        assert by_date[(pd.Timestamp("2026-07-14"), "BBB")] == 3.1

    def test_the_month_it_describes_is_not_the_join_key(self):
        """A join on event_time would have shown the June figure from
        2026-06-30 — two weeks before it existed."""
        out = asof_join(
            _panel(["2026-07-01"]), self.RELEASES, fields=["cpi"], by_entity=False
        )
        assert out["cpi"].isna().all()


class TestStaleness:
    def test_a_series_that_stops_updating_goes_missing_rather_than_flat(self):
        """
        Without a bound, a feed that dies keeps supplying its last value
        forever and the model learns from a number that stopped being a
        measurement years earlier.
        """
        panel = _panel(["2026-08-14", "2026-12-01"])
        unbounded = asof_join(panel, EARNINGS, fields=["eps"])
        bounded = asof_join(
            panel, EARNINGS, fields=["eps"], max_staleness=pd.Timedelta(days=30)
        )
        aaa_unbounded = unbounded[unbounded["entity"] == "AAA"].set_index("date")["eps"]
        aaa_bounded = bounded[bounded["entity"] == "AAA"].set_index("date")["eps"]
        assert aaa_unbounded["2026-12-01"] == 1.61
        assert pd.isna(aaa_bounded["2026-12-01"])
        # The fresh row is unaffected either way.
        assert aaa_bounded["2026-08-14"] == 1.42

    def test_a_nonpositive_bound_is_refused(self):
        with pytest.raises(ValidationError, match="must be positive"):
            asof_join(
                _panel(["2026-08-14"]),
                EARNINGS,
                fields=["eps"],
                max_staleness=pd.Timedelta(0),
            )


class TestCoverageReport:
    def test_reports_partial_availability(self):
        panel = _panel(["2026-07-01", "2026-08-14"])
        joined = asof_join(panel, EARNINGS, fields=["eps"])
        warnings = coverage_report(panel, joined, ["eps"])
        assert any("not yet available" in w for w in warnings)

    def test_reports_a_field_that_was_never_available(self):
        panel = _panel(["2020-01-01"])
        joined = asof_join(panel, EARNINGS, fields=["eps"])
        warnings = coverage_report(panel, joined, ["eps"])
        assert any(w.startswith("WARNING") for w in warnings)

    def test_says_nothing_when_everything_resolved(self):
        panel = _panel(["2026-11-01"], entities=("AAA",))
        joined = asof_join(panel, EARNINGS, fields=["eps"])
        assert coverage_report(panel, joined, ["eps"]) == []
