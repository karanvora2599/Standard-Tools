"""
A data source's content hash does not depend on the pandas version, and a
replay compares an earlier record's hash in the form it was taken.

Every provider hashed the frame it served with `audit.hash_dataframe`, which
covers each column's `str(dtype)`: `object` under pandas 2, `str` under
pandas 3. A tick frame carries text columns (`action`, `side`, `symbol`),
so the same trades hashed differently under the two, and a replay under the
other pandas reported the data as changed with the note "the provider
likely revised historical values" -- a revision that never happened (the
CHANGELOG entry of 2026-10-04).

New entries record `audit.canonical_frame_hash` with `content_hash_version`
2. An entry written before carries no version key and holds the earlier
hash; it is never rewritten (a record is hashed as it was stored), and a
replay checks it against the replayed frame's `hash_dataframe` as read and
under the other pandas's dtype names and datetime resolutions, and says it
cannot tell a revision from a pandas difference when none reproduces it.

The literals below were computed under pandas 2.3.3 and pandas 3.0.5 and are
asserted under whichever one runs the suite: CI runs Python 3.10, which
resolves pandas 2, and Python 3.11 and 3.12, which resolve pandas 3.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from standard_quant_tools import audit
from standard_quant_tools.agent import tools as tools_module
from standard_quant_tools.audit import hashing
from standard_quant_tools.audit.context import _data_sources_var
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.audit.hashing import (
    canonical_frame_hash,
    hash_dataframe,
    hash_payload,
)
from standard_quant_tools.audit.legacy_hash import legacy_hash_variant
from standard_quant_tools.audit.writer import AuditWriter

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3

#: `canonical_frame_hash` of `_ticks()`, the same under both.
TICKS_CANONICAL = "0d38327fa4f9890a"
#: `hash_dataframe` of `_ticks()` as each pandas computes it: the values are
#: hashed alike, and the schema says `object` under pandas 2, `str` under 3.
TICKS_LEGACY_PANDAS_2 = "a83dde63c3afe64c"
TICKS_LEGACY_PANDAS_3 = "94fa99c4c000c20c"
OWN_LEGACY = TICKS_LEGACY_PANDAS_3 if PANDAS_3 else TICKS_LEGACY_PANDAS_2
OTHER_LEGACY = TICKS_LEGACY_PANDAS_2 if PANDAS_3 else TICKS_LEGACY_PANDAS_3

#: A frame shaped like `score_model`'s predictions: `entity` is `object` and
#: `date` `datetime64[ns]` under pandas 2, `str` and `datetime64[us]` under
#: pandas 3.
PREDICTIONS_CANONICAL = "97b8fc10a6bf9a32"
PREDICTIONS_LEGACY_PANDAS_2 = "f5083b380f52fafa"
PREDICTIONS_LEGACY_PANDAS_3 = "64b305a9662e2489"

KEY = ("AAPL", "2024-01-02", "2024-01-03", "trades")
REVISED = "provider likely revised"
UNDECIDED = "cannot tell a revision by the provider from a difference"


def _ticks() -> pd.DataFrame:
    stamps = pd.DatetimeIndex(
        [
            "2024-01-02 14:30:00.000000001",
            "2024-01-02 14:30:00.5",
            "2024-01-02 14:30:01",
        ],
        tz="UTC",
    ).as_unit("ns")
    return pd.DataFrame(
        {
            "price": [187.15, 187.16, np.nan],
            "size": [100.0, 5.0, 20.0],
            "action": ["A", "C", "T"],
            "side": ["B", "A", "N"],
            "order_id": np.array([11, 12, 13], dtype="uint64"),
            "symbol": ["AAPL", "AAPL", "AAPL"],
        },
        index=pd.Index(stamps, name="timestamp"),
    )


def _edited() -> pd.DataFrame:
    frame = _ticks()
    frame.iloc[0, frame.columns.get_loc("price")] = 187.14
    return frame


def _bars(unit: str) -> pd.DataFrame:
    """Daily bars whose index is at `unit`: pandas 3 builds `[ms]` from a
    millisecond epoch where pandas 2 builds `[ns]`."""
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]).as_unit(unit)
    return pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "Close": [1.5, 2.5, 3.5], "Volume": [1e6, 2e6, 3e6]},
        index=index,
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": ["AAA", "BBB"],
            "date": pd.to_datetime(["2023-12-29", "2023-12-29"]).to_numpy(),
            "prediction": [0.25, -0.5],
        }
    )


class _Input(BaseModel):
    symbol: str = "AAPL"


class _Output(BaseModel):
    rows: int = 0


@pytest.fixture(autouse=True)
def _audit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return tmp_path / "audit"


@pytest.fixture
def served(monkeypatch) -> Dict[str, Any]:
    """`ticks_probe`, a tool that reports one fetched frame the way a
    provider does; the returned dict holds the frame it serves."""
    state: Dict[str, Any] = {"frame": _ticks(), "report": None}

    def tool(_model: _Input) -> _Output:
        if state["report"] is not None:
            state["report"]()
        else:
            audit.record_frame_access(
                *KEY, source="databento:XNAS.ITCH", frame=state["frame"]
            )
        return _Output(rows=len(state["frame"]))

    monkeypatch.setitem(tools_module._TOOL_DISPATCH, "ticks_probe", (tool, _Input))
    return state


@pytest.fixture
def open_record():
    token = _data_sources_var.set([])
    try:
        yield lambda: list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token)


def _record(
    content_hash: str, version: Optional[int] = None, rows: int = 3
) -> Dict[str, Any]:
    """A decision record of `ticks_probe` with one data source. `rows` is
    the output it recorded; anything but 3 makes the replayed output differ."""
    source: Dict[str, Any] = {
        "symbol": KEY[0],
        "start": KEY[1],
        "end": KEY[2],
        "interval": KEY[3],
        "source": "databento:XNAS.ITCH",
        "content_hash": content_hash,
        "fetch_ms": 1.0,
    }
    if version is not None:
        source["content_hash_version"] = version
    return {
        "request_id": "r1",
        "tool_name": "ticks_probe",
        "input": {"symbol": "AAPL"},
        "data_sources": [source],
        "output_hash": hash_payload(_Output(rows=rows).model_dump()),
        "status": "ok",
    }


def _notes(result: "audit.ReplayResult") -> str:
    return "\n".join(result.notes)


class TestTheHashDoesNotDependOnPandas:
    def test_the_pinned_values_hold_under_this_pandas(self):
        """The earlier hash of a tick frame is not the same under the two
        pandas; the one recorded now is."""
        assert canonical_frame_hash(_ticks()) == TICKS_CANONICAL
        assert hash_dataframe(_ticks()) == OWN_LEGACY
        assert TICKS_LEGACY_PANDAS_2 != TICKS_LEGACY_PANDAS_3

    def test_a_predictions_frame_too(self):
        """What `score_model` names its predictions file after."""
        legacy = (
            PREDICTIONS_LEGACY_PANDAS_3 if PANDAS_3 else PREDICTIONS_LEGACY_PANDAS_2
        )
        assert canonical_frame_hash(_predictions()) == PREDICTIONS_CANONICAL
        assert hash_dataframe(_predictions()) == legacy

    @pytest.mark.parametrize("dtype", [object, "string"])
    def test_text_storage_and_index_resolution_do_not_change_it(self, dtype):
        frame = _ticks()
        for column in ("action", "side", "symbol"):
            frame[column] = frame[column].astype(dtype)
        assert canonical_frame_hash(frame) == TICKS_CANONICAL
        bars = canonical_frame_hash(_bars("ns"))
        assert canonical_frame_hash(_bars("ms")) == bars


class TestANewEntryRecordsVersion2:
    def test_a_reported_frame_records_the_hash_and_its_version(self, open_record):
        audit.record_frame_access(*KEY, source="databento:XNAS.ITCH", frame=_ticks())
        (entry,) = open_record()
        assert entry["content_hash"] == TICKS_CANONICAL
        assert entry["content_hash_version"] == audit.DATA_SOURCE_HASH_VERSION == 2
        assert not {k for k in entry if k.startswith("legacy")}

    def test_outside_a_record_nothing_is_hashed(self, monkeypatch):
        def refuse(_frame):
            raise AssertionError("hashed with no record open")

        monkeypatch.setattr(hashing, "canonical_frame_hash", refuse)
        audit.record_frame_access(*KEY, source="databento:XNAS.ITCH", frame=_ticks())

    def test_a_digest_given_by_the_caller_records_a_version_only_when_told(
        self, open_record
    ):
        """`record_data_access` keeps its meaning for a caller that hashes
        its own frames: no version key unless the caller names one."""
        audit.record_data_access(*KEY, source="mine", content_hash="abc")
        audit.record_data_access(
            *KEY, source="mine", content_hash="def", content_hash_version=2
        )
        first, second = open_record()
        assert "content_hash_version" not in first
        assert second["content_hash_version"] == 2

    def test_a_day_with_both_forms_verifies(self, _audit_dir):
        """An entry written before the version key and one written now sit
        in one chain, and the day verifies: records are hashed as stored."""
        earlier = audit.DecisionRecord(
            request_id="r0",
            timestamp_utc="2026-10-03T12:00:00+00:00",
            tool_name="ticks_probe",
            input={"symbol": "AAPL"},
            data_sources=_record(OTHER_LEGACY)["data_sources"],
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        AuditWriter().write(earlier)

        def tool(_model):
            audit.record_frame_access(
                *KEY, source="databento:XNAS.ITCH", frame=_ticks()
            )
            return _Output(rows=3)

        _run_and_record("ticks_probe", tool, _Input())
        (day,) = audit._iter_day_files(_audit_dir)
        sources = [
            json.loads(line)["data_sources"][0]
            for line in day.read_text(encoding="utf-8").splitlines()
        ]
        assert "content_hash_version" not in sources[0]
        assert sources[1]["content_hash"] == TICKS_CANONICAL
        assert sources[1]["content_hash_version"] == 2
        assert audit.verify_audit_trail_integrity(_audit_dir) == []


class TestReplayComparesLikeWithLike:
    @pytest.mark.parametrize("recorded", ["own", "other"])
    def test_an_earlier_hash_from_either_pandas_reproduces(self, served, recorded):
        """The record a pandas-2 run wrote and the one a pandas-3 run wrote
        both reproduce here: one as read, the other with the text columns
        spelled the way the other pandas spells them."""
        stored = OWN_LEGACY if recorded == "own" else OTHER_LEGACY
        result = audit.verify_replay(_record(stored))

        (match,) = result.data_source_matches
        assert match["match"] is True and match["hash_version"] == 1
        assert match["old_hash"] == stored and match["new_hash"] == OWN_LEGACY
        assert result.output_match is True
        assert REVISED not in _notes(result)
        if recorded == "own":
            assert match["reproduced_with"] is None and result.notes == []
        else:
            spelled = "object" if PANDAS_3 else "str"
            assert match["reproduced_with"] == f"text spelled {spelled}"
            assert "reproduces it with text spelled" in _notes(result)

    def test_a_datetime_index_at_another_resolution_reproduces(self, served):
        """An index pandas 2 built at `[ns]` and pandas 3 at `[ms]`: the
        values are the same instants, and the variant says how."""
        served["frame"] = _bars("ms")
        result = audit.verify_replay(_record(hash_dataframe(_bars("ns"))))
        (match,) = result.data_source_matches
        assert match["match"] is True
        assert match["reproduced_with"] == "the [ms] index at [ns]"

    def test_a_current_hash_is_compared_without_the_earlier_one(
        self, served, monkeypatch
    ):
        def refuse(_frame):
            raise AssertionError("the earlier hash was taken for a version-2 record")

        monkeypatch.setattr(hashing, "hash_dataframe", refuse)
        result = audit.verify_replay(_record(TICKS_CANONICAL, version=2))
        (match,) = result.data_source_matches
        assert match["match"] is True and match["hash_version"] == 2
        assert match["new_hash"] == TICKS_CANONICAL
        assert result.notes == []

    def test_a_changed_value_under_the_current_hash_is_a_revision(self, served):
        """Null case: the hash that does not depend on pandas missed, so
        the data changed, and the note says so as it always has."""
        stale = canonical_frame_hash(_edited())
        result = audit.verify_replay(_record(stale, version=2, rows=4))
        (match,) = result.data_source_matches
        assert match["match"] is False and match["hash_version"] == 2
        assert REVISED in _notes(result)
        assert UNDECIDED not in _notes(result)

    def test_an_earlier_hash_that_nothing_reproduces_is_undecided(self, served):
        """A version-1 miss under every representation tried: a revised
        value and a pandas outside the variants leave the same miss, and
        the record does not say which pandas wrote it."""
        result = audit.verify_replay(_record(hash_dataframe(_edited()), rows=4))
        (match,) = result.data_source_matches
        assert match["match"] is False and match["hash_version"] == 1
        assert match["new_hash"] == OWN_LEGACY
        notes = _notes(result)
        assert UNDECIDED in notes
        assert f"this replay runs pandas {pd.__version__}" in notes
        assert "does not say which pandas hashed it" in notes
        assert REVISED not in notes

    def test_an_unversioned_digest_is_not_compared_with_a_versioned_one(self, served):
        """A provider outside this library reporting its own digest,
        replayed against a record that holds the current form."""
        served["report"] = lambda: audit.record_data_access(
            *KEY, source="mine", content_hash="their-own"
        )
        result = audit.verify_replay(_record(TICKS_CANONICAL, version=2))
        (match,) = result.data_source_matches
        assert match["match"] is None and match["hash_version"] == 2
        assert "were not compared" in _notes(result)

    def test_two_unversioned_digests_are_compared_as_given(self, served):
        """Null case: a provider outside this library, both times. Nothing
        is re-hashed, and the comparison is the one replay always made."""
        served["report"] = lambda: audit.record_data_access(
            *KEY, source="mine", content_hash="their-own"
        )
        result = audit.verify_replay(_record("their-own"))
        (match,) = result.data_source_matches
        assert match["match"] is True and match["new_hash"] == "their-own"

    def test_the_earlier_hash_never_reaches_a_written_record(self, served, _audit_dir):
        """A call the replayed tool dispatches writes its own record, from
        its own list: the replay's comparison keys stay on the replay."""

        def inner(_model):
            audit.record_frame_access(
                *KEY, source="databento:XNAS.ITCH", frame=_ticks()
            )
            return _Output(rows=3)

        def report():
            _run_and_record("inner_probe", inner, _Input())
            audit.record_frame_access(
                *KEY, source="databento:XNAS.ITCH", frame=_ticks()
            )

        served["report"] = report
        result = audit.verify_replay(_record(OTHER_LEGACY))
        assert result.data_source_matches[0]["match"] is True

        (day,) = audit._iter_day_files(_audit_dir)
        (line,) = day.read_text(encoding="utf-8").splitlines()
        (written,) = json.loads(line)["data_sources"]
        assert set(written) == {
            "symbol",
            "start",
            "end",
            "interval",
            "source",
            "content_hash",
            "content_hash_version",
            "fetch_ms",
        }


class TestTheVariantSearch:
    def test_an_index_resolution_alone(self):
        assert (
            legacy_hash_variant(_bars("ms"), hash_dataframe(_bars("ns")))
            == "the [ms] index at [ns]"
        )

    def test_a_changed_value_has_no_variant(self):
        """Null case: SHA-256 makes a false match from a changed value
        impossible."""
        edited = _bars("ns")
        edited.iloc[0, 0] = 1.25
        assert legacy_hash_variant(edited, hash_dataframe(_bars("ns"))) is None
        assert legacy_hash_variant(_edited(), OWN_LEGACY) is None

    def test_the_dataset_check_uses_the_same_search(self):
        from standard_quant_tools.modeling.dataset import integrity

        assert integrity.legacy_hash_variant is legacy_hash_variant


class TestTheCurrentHashIsExported:
    def test_beside_hash_dataframe(self):
        """`standard_quant_tools.audit` names both forms, so a caller that
        hashes its own frames for `record_data_access` can record version
        2 without reaching into a submodule."""
        assert audit.canonical_frame_hash is hashing.canonical_frame_hash
        assert audit.hash_dataframe is hashing.hash_dataframe
        assert {"canonical_frame_hash", "hash_dataframe"} <= set(audit.__all__)
        assert audit.canonical_frame_hash(_ticks()) == TICKS_CANONICAL


#: The note `replay_decision` adds to a `data_changed` verdict.
POINT_IN_TIME = "does not guarantee point-in-time values"

KEY_2 = ("MSFT", "2024-01-02", "2024-01-03", "trades")


def _write(record: Dict[str, Any]) -> None:
    """Writes `record` into the audit trail as the writer does, so the
    replay finds it by request id."""
    AuditWriter().write(
        audit.DecisionRecord(
            timestamp_utc="2026-10-03T12:00:00+00:00",
            cpp_available=False,
            duration_ms=1.0,
            **record,
        )
    )


def _source(key, content_hash: str, version: Optional[int] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = dict(
        zip(("symbol", "start", "end", "interval"), key),
        source="databento:XNAS.ITCH",
        content_hash=content_hash,
    )
    if version is not None:
        entry["content_hash_version"] = version
    return entry


class TestAnUndecidedSourceIsNotADataChange:
    """A replay whose only differing inputs are version-1 hashes nothing
    reproduces gets its own verdict. It was `data_changed`, and the note
    "the normal consequence of a provider that does not guarantee
    point-in-time values" followed the replay's own note that it cannot
    tell a revision from a pandas difference (the CHANGELOG entry of
    2026-10-04)."""

    def test_replay_decision_says_data_undecided(self, served):
        _write(_record(hash_dataframe(_edited()), rows=4))
        result = tools_module.dispatch("replay_decision", {"request_id": "r1"})

        assert result["verdict"] == "data_undecided"
        (match,) = result["data_source_matches"]
        assert match["matches"] is False and match["undecided"] is True
        notes = "\n".join(result["notes"])
        assert UNDECIDED in notes
        assert "cannot say whether the data, the code or the pandas version" in notes
        assert POINT_IN_TIME not in notes and REVISED not in notes

    def test_a_decided_change_beside_it_is_still_data_changed(self, served):
        """Null case: a version-2 hash missed too, which no pandas explains,
        so the verdict and its note are the ones a revision always got."""
        served["report"] = lambda: [
            audit.record_frame_access(
                *key, source="databento:XNAS.ITCH", frame=_ticks()
            )
            for key in (KEY, KEY_2)
        ]
        record = _record(hash_dataframe(_edited()), rows=4)
        record["data_sources"] = [
            _source(KEY, hash_dataframe(_edited())),
            _source(KEY_2, canonical_frame_hash(_edited()), version=2),
        ]
        _write(record)
        result = tools_module.dispatch("replay_decision", {"request_id": "r1"})

        assert result["verdict"] == "data_changed"
        undecided = {m["symbol"]: m["undecided"] for m in result["data_source_matches"]}
        assert undecided == {"AAPL": True, "MSFT": False}
        assert any(POINT_IN_TIME in note for note in result["notes"])

    def test_an_output_that_reproduces_is_reproduced(self, served):
        """Null case: the verdict is about the output first, as before."""
        _write(_record(hash_dataframe(_edited()), rows=3))
        result = tools_module.dispatch("replay_decision", {"request_id": "r1"})
        assert result["verdict"] == "reproduced"
        assert result["data_source_matches"][0]["undecided"] is True

    def test_a_respelled_hash_is_not_undecided(self, served):
        """Null case: a version-1 hash the other pandas's spelling
        reproduces is the recorded data, and the output decides."""
        _write(_record(OTHER_LEGACY, rows=4))
        result = tools_module.dispatch("replay_decision", {"request_id": "r1"})
        assert result["verdict"] == "code_changed"
        (match,) = result["data_source_matches"]
        assert match["matches"] is True and match["undecided"] is False

    def test_sqt_replay_exits_1_and_marks_the_source(self, served):
        """The output did not reproduce, which is what exit code 1 says;
        the report marks the source the replay could not decide."""
        from standard_quant_tools import cli

        _write(_record(hash_dataframe(_edited()), rows=4))
        report, exit_code = cli._replay("r1")

        assert exit_code == 1
        assert "match=False  undecided" in report
        assert UNDECIDED in report
