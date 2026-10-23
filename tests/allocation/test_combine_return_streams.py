"""
`combine_return_streams`: several return streams on the dates they all
share, published as one `returns_panel`.

THE PLANTED STREAMS. Three daily streams on three calendars: `us` on 12
business days from 2024-01-02; `eu` from 2024-01-04 on 12 business days
without 2024-01-08 and 2024-01-15 (so 10); `fund` a column of a published
panel from 2024-01-03 to 2024-01-17, without 2024-01-10. The window every
stream covers runs 2024-01-04 (eu's start) to 2024-01-17 (where us and
fund end), and inside it eu lacks two dates and fund one more, so the
streams share 7 dates.
Each stream's values are distinct multiples of 1e-4, so a value moved to
another date or another stream cannot pass for the right one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.error import ValidationError


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _call(tool, arguments):
    return resolve("allocation").dispatch(tool, arguments)


def _planted():
    us = pd.Series(
        1e-4 * np.arange(1, 13), index=pd.bdate_range("2024-01-02", periods=12)
    )
    eu_dates = pd.bdate_range("2024-01-04", periods=12).drop(
        [pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")]
    )
    eu = pd.Series(-1e-4 * np.arange(1, 11) - 0.002, index=eu_dates)
    fund_dates = pd.bdate_range("2024-01-03", "2024-01-17").drop(
        pd.Timestamp("2024-01-10")
    )
    fund = pd.Series(1e-4 * np.arange(1, len(fund_dates) + 1) + 0.01, index=fund_dates)
    return us, eu, fund


def _values(series):
    return {str(d.date()): float(v) for d, v in series.items()}


def _streams(runs_dir):
    us, eu, fund = _planted()
    panel = handoff.publish(
        pd.DataFrame({"fund": fund, "other": fund * 3}),
        kind="returns_panel",
        run_id="src",
        name="funds",
    )
    return {
        "us": {"values": _values(us)},
        "eu": {"values": _values(eu)},
        "fund": {"ref": panel, "column": "fund"},
    }


class TestThePlantedCurvesComeBackExactly:
    def test_the_reference_holds_the_shared_dates_and_each_streams_own_values(
        self, runs_dir
    ):
        """The published panel is the seven shared dates and, on each, the
        value each stream itself carried there -- the same bits."""
        us, eu, fund = _planted()
        out = _call(
            "combine_return_streams",
            {"streams": _streams(runs_dir), "run_id": "combined", "min_overlap": 5},
        )
        shared = us.index.intersection(eu.index).intersection(fund.index)
        assert len(shared) == 7
        assert out["returns_ref"] == "sqt://returns_panel/combined/streams"
        assert out["columns"] == ["us", "eu", "fund"]
        assert out["n_dates"] == 7
        assert (out["first_date"], out["last_date"]) == ("2024-01-04", "2024-01-17")
        panel = handoff.resolve(out["returns_ref"], expect="returns_panel")
        assert list(panel.index) == list(shared)
        for name, series in (("us", us), ("eu", eu), ("fund", fund)):
            assert (
                panel[name].to_numpy().tobytes() == series[shared].to_numpy().tobytes()
            )

    def test_the_report_says_what_each_stream_lost_and_where(self, runs_dir):
        """us loses 2 before the window (01-02, 01-03) and 3 inside it
        (01-08 and 01-15, which eu lacks, and 01-10, which fund lacks); eu
        loses 1 inside (01-10) and 2 after (01-18, 01-19); fund loses 1
        before (01-03) and 2 inside (01-08, 01-15)."""
        out = _call(
            "combine_return_streams",
            {"streams": _streams(runs_dir), "run_id": "report", "min_overlap": 5},
        )
        report = out["alignment"]
        assert report["window_start"] == "2024-01-04"
        assert report["window_end"] == "2024-01-17"
        assert report["start_set_by"] == ["eu"]
        assert report["end_set_by"] == ["us", "fund"]
        rows = report["by_stream"]
        assert rows["us"] == {
            "n_supplied": 12,
            "first_date": "2024-01-02",
            "last_date": "2024-01-17",
            "n_kept": 7,
            "n_dropped_before": 2,
            "n_dropped_after": 0,
            "n_dropped_inside": 3,
        }
        assert (
            rows["eu"]["n_dropped_before"],
            rows["eu"]["n_dropped_after"],
            rows["eu"]["n_dropped_inside"],
        ) == (0, 2, 1)
        assert (
            rows["fund"]["n_dropped_before"],
            rows["fund"]["n_dropped_after"],
            rows["fund"]["n_dropped_inside"],
        ) == (1, 0, 2)
        assert out["warnings"] and "Nothing was filled" in out["warnings"][0]

    def test_columns_of_a_panel_meet_on_the_dates_they_share(self, runs_dir):
        """returns_ref with columns: a blank cell is a date that column
        lacks, so the two columns meet on the rows where both have a value."""
        us, eu, _fund = _planted()
        ref = handoff.publish(
            pd.DataFrame({"us": us, "eu": eu, "unused": us * 2}),
            kind="returns_panel",
            run_id="src2",
            name="wide",
        )
        out = _call(
            "combine_return_streams",
            {
                "returns_ref": ref,
                "columns": ["eu", "us"],
                "run_id": "picked",
                "min_overlap": 5,
            },
        )
        shared = us.index.intersection(eu.index)
        panel = handoff.resolve(out["returns_ref"], expect="returns_panel")
        assert out["columns"] == ["eu", "us"]
        assert list(panel.index) == list(shared)
        assert panel["eu"].to_numpy().tobytes() == eu[shared].to_numpy().tobytes()


class TestRefusals:
    def test_an_undated_list_is_not_a_stream(self, runs_dir):
        with pytest.raises(pydantic.ValidationError, match="undated"):
            _call(
                "combine_return_streams",
                {
                    "streams": {
                        "a": {"values": [0.01, 0.02]},
                        "b": {"values": [0.0, 0.01]},
                    },
                    "run_id": "undated",
                },
            )

    def test_a_short_overlap_names_the_stream_that_limits_it(self, runs_dir):
        """Seven shared dates against min_overlap=8: refused, naming eu,
        without which 10 dates would be shared."""
        with pytest.raises(
            ValidationError, match=r"share 7 date.*\['eu'\] limit.*without it 10"
        ):
            _call(
                "combine_return_streams",
                {"streams": _streams(runs_dir), "run_id": "short", "min_overlap": 8},
            )

    def test_levels_passed_as_returns_are_refused(self, runs_dir):
        us, eu, _fund = _planted()
        levels = 100.0 * (1.0 + us).cumprod()
        with pytest.raises(ValidationError, match="level series"):
            _call(
                "combine_return_streams",
                {
                    "streams": {
                        "us": {"values": _values(levels)},
                        "eu": {"values": _values(eu)},
                    },
                    "run_id": "levels",
                    "min_overlap": 2,
                },
            )

    def test_a_symbol_needs_both_ends_of_its_window(self, runs_dir):
        """This is the one door that fetches a stream by symbol, and only
        over a window the caller names: a symbol without its end date is
        refused before anything is fetched."""
        with pytest.raises(pydantic.ValidationError, match="end_date"):
            _call(
                "combine_return_streams",
                {
                    "streams": {"a": {"symbol": "AAPL", "start_date": "2024-01-02"}},
                    "run_id": "fetch",
                },
            )

    @pytest.mark.parametrize(
        "extra",
        [
            {"returns_ref": "sqt://returns_panel/x/y"},
            {"columns": ["us"]},
        ],
    )
    def test_streams_take_neither_a_reference_nor_columns_beside_them(
        self, runs_dir, extra
    ):
        with pytest.raises(pydantic.ValidationError):
            _call(
                "combine_return_streams",
                {"streams": _streams(runs_dir), "run_id": "both", **extra},
            )

    def test_a_used_run_id_is_refused(self, runs_dir):
        arguments = {"streams": _streams(runs_dir), "run_id": "twice", "min_overlap": 5}
        _call("combine_return_streams", arguments)
        with pytest.raises(ValidationError, match="already published"):
            _call("combine_return_streams", arguments)


class TestWhoFetches:
    def test_only_the_combining_tool_can_fetch(self):
        """combine_return_streams takes the full stream type, symbol form
        included, so it is advertised as reading market data. The
        allocating tools take dated values or a reference only, so they are
        not, and stay inside the surface's determinism layer. Each tool that
        publishes at a run_id is advertised as not idempotent."""
        from standard_quant_tools.mcp.catalog import build_catalog

        catalog = build_catalog()
        assert catalog["combine_return_streams"].reads_market_data is True
        for name in ("allocate_across_sleeves", "evaluate_allocation_rule"):
            assert catalog[name].reads_market_data is False, name
            assert "symbol" not in str(catalog[name].input_schema), name
        assert catalog["combine_return_streams"].idempotent is False
        assert catalog["evaluate_allocation_rule"].idempotent is False
        assert catalog["allocate_across_sleeves"].idempotent is True
