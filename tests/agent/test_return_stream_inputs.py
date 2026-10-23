"""
`ReturnStream`: one dated return series from exactly one place.

It is `DataSource` with the difference that matters for several series:
inline values carry their DATES. Two equal-length lists from different
calendars look aligned and are not, so an undated list is refused with that
reason, and a stream is a reference column, {ISO date: return}, or a symbol
over an explicit window.
"""

from __future__ import annotations

import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.streams import (
    ReturnStream,
    resolve_return_streams,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.streams import align_return_streams


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _dated(dates, values):
    return {str(d.date()): float(v) for d, v in zip(dates, values)}


class TestTheInputModel:
    def test_two_calendars_as_dated_values_align_on_their_shared_dates(self):
        """
        Planted: two ten-return streams, the second starting two business
        days later. As dated values they resolve and align on the 8 dates
        they share, each losing 2.
        """
        first = pd.bdate_range("2024-01-02", periods=10)
        second = pd.bdate_range("2024-01-04", periods=10)
        streams = {
            "a": ReturnStream(values=_dated(first, [0.001 * i for i in range(10)])),
            "b": ReturnStream(values=_dated(second, [-0.001 * i for i in range(10)])),
        }
        series = resolve_return_streams(streams, what="probe")
        frame, report = align_return_streams(series, min_overlap=5)
        assert report["n_dates"] == 8
        assert report["by_stream"]["a"]["n_dropped_before"] == 2
        assert report["by_stream"]["b"]["n_dropped_after"] == 2
        assert frame["a"].tolist() == [0.001 * i for i in range(2, 10)]

    def test_the_same_values_as_undated_lists_are_refused(self):
        """The case equal lengths hide: the values above without their
        dates cannot be checked against each other, so they are not a
        stream."""
        with pytest.raises(pydantic.ValidationError) as refusal:
            ReturnStream(values=[0.001 * i for i in range(10)])
        message = str(refusal.value)
        assert "undated values are not a stream" in message
        assert "{ISO date: return}" in message

    @pytest.mark.parametrize(
        "fields, says",
        [
            ({}, "exactly one of ref / values / symbol; got none"),
            (
                {"ref": "sqt://returns_panel/r/x", "values": {"2024-01-02": 0.01}},
                "exactly one of ref / values / symbol",
            ),
            ({"values": {"2024-01-02": 0.01}, "column": "A"}, "column picks"),
            (
                {"values": {"2024-01-02": 0.01}, "start_date": "2024-01-02"},
                "only apply to a `symbol`",
            ),
            ({"symbol": "AAPL", "start_date": "2024-01-02"}, "needs end_date"),
            ({"values": {"2024-01-02": float("nan")}}, "not finite"),
            ({"values": {"2024-01-02": float("inf")}}, "not finite"),
            ({"values": {"Jan 2": 0.01}}, "not ISO dates"),
            ({"values": {}}, "at least 1"),
            ({"ref": "sqt://returns_panel/r/x", "extra": 1}, "Extra inputs"),
        ],
    )
    def test_each_rule_is_refused_by_name(self, fields, says):
        with pytest.raises(pydantic.ValidationError, match=says):
            ReturnStream(**fields)

    def test_a_symbol_with_its_window_is_accepted(self):
        stream = ReturnStream(
            symbol="AAPL", start_date="2024-01-02", end_date="2024-06-28"
        )
        assert stream.source is None


class TestResolving:
    def test_a_reference_column_resolves_to_its_dated_series(self, runs_dir):
        dates = pd.bdate_range("2024-01-02", periods=5)
        panel = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03, 0.04, 0.05], "B": [None, 0.1, 0.2, 0.3, 0.4]},
            index=dates,
        )
        ref = handoff.publish(panel, kind="returns_panel", run_id="s", name="p")
        out = resolve_return_streams(
            {"b": ReturnStream(ref=ref, column="B")}, what="probe"
        )
        assert out["b"].index.tolist() == list(dates[1:])
        assert out["b"].tolist() == [0.1, 0.2, 0.3, 0.4]

    def test_a_panel_of_several_columns_needs_column(self, runs_dir):
        dates = pd.bdate_range("2024-01-02", periods=3)
        ref = handoff.publish(
            pd.DataFrame({"A": [0.01] * 3, "B": [0.02] * 3}, index=dates),
            kind="returns_panel",
            run_id="s2",
            name="p",
        )
        with pytest.raises(ValidationError, match="name the one this stream is"):
            resolve_return_streams({"x": ReturnStream(ref=ref)}, what="probe")

    def test_a_reference_of_another_kind_is_refused_naming_both(self, runs_dir):
        dates = pd.bdate_range("2024-01-02", periods=3)
        ref = handoff.publish(
            pd.Series([100.0, 101.0, 102.0], index=dates, name="equity"),
            kind="equity_curve",
            run_id="s3",
            name="curve",
        )
        with pytest.raises(ValidationError) as refusal:
            resolve_return_streams({"x": ReturnStream(ref=ref)}, what="probe")
        message = str(refusal.value)
        assert "'returns_panel'" in message and "'equity_curve'" in message
        assert "convert_reference" in message

    def test_a_symbol_is_fetched_over_its_own_window_from_its_provider(
        self, monkeypatch
    ):
        """DataSource's rule, kept: the window and the provider are the
        caller's, and the result is the dated close-to-close returns."""
        from standard_quant_tools.data.factory import DataFactory

        seen = {}
        dates = pd.bdate_range("2024-01-02", periods=4)

        class Provider:
            def get_ohlcv(self, symbol, start, end):
                seen.update(symbol=symbol, start=start, end=end)
                return pd.DataFrame(
                    {"Close": [100.0, 101.0, 99.99, 100.9899]}, index=dates
                )

        monkeypatch.setattr(
            DataFactory, "get_provider", staticmethod(lambda *a, **k: Provider())
        )
        out = resolve_return_streams(
            {
                "s": ReturnStream(
                    symbol="AAA", start_date="2024-01-02", end_date="2024-01-05"
                )
            },
            what="probe",
        )
        assert seen == {"symbol": "AAA", "start": "2024-01-02", "end": "2024-01-05"}
        assert out["s"].index.tolist() == list(dates[1:])
        assert out["s"].tolist() == pytest.approx([0.01, -0.01, 0.01], abs=1e-12)
