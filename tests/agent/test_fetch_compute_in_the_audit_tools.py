"""
The fetch/compute split and the instruction-set path, as the provenance
tools report them.

`explain_decision` shows one call's split -- per data source and summed --
and the path its kernels took. `describe_audit_log`, asked for its per-day
breakdown, reports per tool the median and p95 of `fetch_ms` and
`compute_ms`, so "slow kernel or slow vendor" is one call rather than an
inference from `duration_ms`. Records without the split -- written before
it existed, or failed calls -- are counted as not split rather than
skewing the statistics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from pydantic import BaseModel

from standard_quant_tools import _native_build as nb
from standard_quant_tools import audit
from standard_quant_tools.agent.runtimes.meta import tools as meta_tools
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.dispatch import _run_and_record

PAST_DAY = "2020-01-01"


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return directory


class _Probe(BaseModel):
    payload: Dict[str, Any] = {}


def _a_call_that_fetches(directory: Path) -> str:
    def tool(model):
        audit.record_data_access(
            "AAPL", "2024-01-01", "2024-06-01", "1d", "disk_cache", "abc"
        )
        return {"beta": 1.1}

    _run_and_record("probe_tool", tool, _Probe())
    day = audit._iter_day_files(directory)[-1]
    return json.loads(day.read_text(encoding="utf-8").splitlines()[-1])["request_id"]


class TestExplainShowsTheSplit:
    def test_per_source_and_summed(self, isolated_audit_log: Path):
        rid = _a_call_that_fetches(isolated_audit_log)
        result = dispatch("explain_decision", {"request_id": rid})
        (source,) = result["data_sources"]
        assert source["fetch_ms"] is not None
        assert result["fetch_ms"] == source["fetch_ms"]
        assert result["compute_ms"] == pytest.approx(
            result["duration_ms"] - result["fetch_ms"], abs=0.002
        )

    def test_and_the_path_and_the_rounded_hash(self, isolated_audit_log: Path):
        rid = _a_call_that_fetches(isolated_audit_log)
        result = dispatch("explain_decision", {"request_id": rid})
        assert result["native_isa"] == nb.native_isa()
        assert result["output_hash_rounded"]

    def test_a_record_from_before_the_fields_reports_none(
        self, isolated_audit_log: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Null case: an older record has no such keys, and the answer is
        None rather than a guess -- not a zero fetch time."""
        rid = _a_call_that_fetches(isolated_audit_log)
        real = meta_tools._find_audit_record

        def without_fields(*args, **kwargs):
            record = dict(real(*args, **kwargs))
            for name in ("fetch_ms", "compute_ms", "native_isa", "output_hash_rounded"):
                record.pop(name, None)
            record["data_sources"] = [
                {k: v for k, v in s.items() if k != "fetch_ms"}
                for s in record["data_sources"]
            ]
            return record

        monkeypatch.setattr(meta_tools, "_find_audit_record", without_fields)
        result = dispatch("explain_decision", {"request_id": rid})
        assert result["fetch_ms"] is None and result["compute_ms"] is None
        assert result["native_isa"] is None
        assert result["output_hash_rounded"] is None
        assert result["data_sources"][0]["fetch_ms"] is None


def _line(
    tool: str, fetch: Any = None, compute: Any = None, status: str = "ok", i: int = 0
) -> Dict[str, Any]:
    line: Dict[str, Any] = {
        "request_id": f"{tool}-{i}",
        "timestamp_utc": f"{PAST_DAY}T00:00:{i:02d}+00:00",
        "tool_name": tool,
        "status": status,
        "duration_ms": 1.0,
        "prev_record_hash": "0" * 16,
        "record_hash": "f" * 16,
    }
    if fetch is not None:
        line["fetch_ms"] = fetch
        line["compute_ms"] = compute
    return line


def _plant(directory: Path, lines: List[Dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{PAST_DAY}.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )


def _timings(result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["tool_name"]: row for row in result["tool_timings"]}


class TestDescribeReportsTheSplitPerTool:
    def test_median_and_p95_over_the_split_records(self, isolated_audit_log: Path):
        fetches = [100.0, 200.0, 300.0, 400.0]
        computes = [10.0, 20.0, 30.0, 40.0]
        lines = [
            _line("get_technical_analysis", f, c, i=i)
            for i, (f, c) in enumerate(zip(fetches, computes))
        ]
        lines += [
            _line("get_technical_analysis", i=10),  # before the split
            _line("get_technical_analysis", i=11),
            # A failed call carries a fetch_ms but no compute_ms.
            {
                **_line("get_technical_analysis", status="error", i=12),
                "fetch_ms": 50.0,
                "compute_ms": None,
            },
            _line("list_strategies", 0.0, 5.0, i=13),
        ]
        _plant(isolated_audit_log, lines)

        result = dispatch("describe_audit_log", {"include_days": True})
        rows = _timings(result)
        tech = rows["get_technical_analysis"]
        assert (tech["split"], tech["not_split"]) == (4, 3)
        assert tech["fetch_ms_median"] == 250.0
        assert tech["fetch_ms_p95"] == pytest.approx(np.percentile(fetches, 95))
        assert tech["compute_ms_median"] == 25.0
        assert tech["compute_ms_p95"] == pytest.approx(np.percentile(computes, 95))
        listed = rows["list_strategies"]
        assert (listed["split"], listed["not_split"]) == (1, 0)
        assert listed["fetch_ms_median"] == 0.0
        assert listed["compute_ms_p95"] == 5.0
        assert [row["tool_name"] for row in result["tool_timings"]] == sorted(rows)
        assert any("3 record(s)" in note for note in result["notes"])

    def test_a_trail_from_before_the_split_is_all_not_split(
        self, isolated_audit_log: Path
    ):
        """Null case: no statistics are invented for records that never
        carried the split."""
        _plant(
            isolated_audit_log,
            [_line("run_hurst_analysis", i=i) for i in range(3)],
        )
        result = dispatch("describe_audit_log", {"include_days": True})
        (row,) = result["tool_timings"]
        assert (row["split"], row["not_split"]) == (0, 3)
        assert row["fetch_ms_median"] is None and row["compute_ms_p95"] is None

    def test_the_summary_alone_parses_no_record(self, isolated_audit_log: Path):
        _plant(isolated_audit_log, [_line("list_strategies", 1.0, 2.0)])
        result = dispatch("describe_audit_log", {})
        assert result["tool_timings"] == []
        assert any("fetching data and computing" in n for n in result["notes"])

    def test_real_calls_feed_it(self, isolated_audit_log: Path):
        """The writer and the reader agree on the field names."""
        _a_call_that_fetches(isolated_audit_log)
        _a_call_that_fetches(isolated_audit_log)
        result = dispatch("describe_audit_log", {"include_days": True})
        row = _timings(result)["probe_tool"]
        assert row["split"] == 2 and row["not_split"] == 0
        assert row["fetch_ms_median"] is not None
