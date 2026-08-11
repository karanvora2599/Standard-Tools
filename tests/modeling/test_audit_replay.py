"""
Regression tests for modeling audit + replay.

`modeling_dispatch` faithfully wrote decision records via
`audit._run_and_record`, but `verify_replay` hardcoded
`agent.tools._TOOL_DISPATCH` — so a `run_model_experiment` record could not
be replayed at all. And modeling mints a fresh UUID-based dataset_id/model_id
on every run, embedding it in artifact paths, so even after resolution a
byte-identical re-run would never match literally: every modeling replay
would report a false mismatch, which is worse than no support at all because
it looks like evidence of drift.

The modeling test fixture also disabled audit globally, so the integration
these tests cover was previously never exercised end to end.
"""

import json
import os
from pathlib import Path

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit.replay import (
    _has_volatile_identifiers,
    _resolve_tool,
    normalize_identifiers,
)
from standard_quant_tools.modeling.agent import modeling_dispatch


def _dataset_args() -> dict:
    return {
        "spec": {
            "universe": ["AAA", "BBB"],
            "start": "2022-01-01",
            "end": "2023-12-31",
            "features": [{"id": "technical.rsi"}, {"id": "market.momentum"}],
            "target": {"type": "forward_return", "horizon": 5},
        }
    }


def _read_records() -> list:
    directory = Path(os.environ["SQT_AUDIT_DIR"])
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        if not audit._DAY_FILE_RE.match(path.name):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


class TestToolResolution:
    def test_modeling_tools_resolve(self):
        """The hardcoded agent-only registry made every one of these
        fail with 'Unknown tool'."""
        for name in (
            "list_features",
            "build_model_dataset",
            "run_model_experiment",
            "score_model",
            "inspect_model",
        ):
            _, _, surface = _resolve_tool(name)
            assert surface == "modeling"

    def test_agent_tools_still_resolve(self):
        _, _, surface = _resolve_tool("analyze_stock_risk")
        assert surface == "agent"

    def test_unknown_tool_names_both_registries(self):
        with pytest.raises(ValueError, match="modeling tool registry"):
            _resolve_tool("no_such_tool")


class TestIdentifierNormalization:
    def test_detects_run_specific_ids(self):
        assert _has_volatile_identifiers({"model_id": "mdl_a1b2c3d4e5f6"})
        assert _has_volatile_identifiers({"uri": "/runs/ds_0123456789ab/panel.parquet"})
        assert not _has_volatile_identifiers({"n_folds": 4, "name": "momentum"})

    def test_same_substance_different_ids_normalizes_equal(self):
        a = {
            "model_id": "mdl_a1b2c3d4e5f6",
            "uri": "/runs/mdl_a1b2c3d4e5f6/oos_predictions.parquet",
            "oos_metrics": {"r2": 0.1},
        }
        b = {
            "model_id": "mdl_999999999999",
            "uri": "/runs/mdl_999999999999/oos_predictions.parquet",
            "oos_metrics": {"r2": 0.1},
        }
        assert normalize_identifiers(a) == normalize_identifiers(b)

    def test_real_drift_still_differs(self):
        """Normalization must not be so broad it hides genuine change."""
        a = {"model_id": "mdl_a1b2c3d4e5f6", "oos_metrics": {"r2": 0.1}}
        b = {"model_id": "mdl_999999999999", "oos_metrics": {"r2": 0.9}}
        assert normalize_identifiers(a) != normalize_identifiers(b)

    def test_only_the_id_pattern_is_rewritten(self):
        """A string that merely CONTAINS 'mdl' must be left alone."""
        payload = {"note": "model_dl_pipeline", "feature": "mdl_not_an_id"}
        assert normalize_identifiers(payload) == payload

    def test_nested_containers_walked(self):
        payload = {"folds": [{"uri": "/runs/ds_0123456789ab/p.parquet"}]}
        assert normalize_identifiers(payload)["folds"][0]["uri"] == (
            "/runs/<run_id>/p.parquet"
        )


class TestModelingAuditIntegration:
    def test_dispatch_writes_a_decision_record(self, patched_multi_factory):
        """
        The modeling fixture used to disable audit entirely, so this
        integration was asserted nowhere.
        """
        modeling_dispatch("build_model_dataset", _dataset_args())
        records = _read_records()
        assert len(records) == 1
        assert records[0]["tool_name"] == "build_model_dataset"
        assert records[0]["status"] == "ok"

    def test_record_carries_both_hashes(self, patched_multi_factory):
        modeling_dispatch("build_model_dataset", _dataset_args())
        record = _read_records()[0]
        assert record["output_hash"]
        assert record["output_hash_normalized"]
        # The dataset_id makes the literal output differ from the
        # normalized one, which is exactly why both are stored.
        assert record["output_hash"] != record["output_hash_normalized"]

    def test_failed_call_still_recorded(self, patched_multi_factory):
        args = _dataset_args()
        args["spec"]["features"] = [{"id": "does.not.exist"}]
        with pytest.raises(Exception):
            modeling_dispatch("build_model_dataset", args)
        record = _read_records()[0]
        assert record["status"] == "error"
        assert record["output_hash"] is None


class TestModelingReplay:
    def test_replay_reproduces_a_dataset_build(self, patched_multi_factory):
        """
        End to end: dispatch through the audit trail, then replay the
        stored record. The dataset_id differs between the two runs by
        construction, so this only passes because comparison is semantic.
        """
        modeling_dispatch("build_model_dataset", _dataset_args())
        record = _read_records()[0]

        result = audit.verify_replay(record)
        assert result.tool_name == "build_model_dataset"
        assert result.output_match is True
        assert any("normalized" in note for note in result.notes)

    def test_replay_detects_a_changed_spec(self, patched_multi_factory):
        """Semantic comparison must still catch a real difference — here a
        different feature set, replayed against the original record."""
        modeling_dispatch("build_model_dataset", _dataset_args())
        record = _read_records()[0]

        tampered = dict(record)
        tampered["input"] = {
            "spec": {
                **_dataset_args()["spec"],
                "features": [{"id": "technical.rsi"}],
            }
        }
        result = audit.verify_replay(tampered)
        assert result.output_match is False

    def test_legacy_record_without_normalized_hash_is_not_comparable(
        self, patched_multi_factory
    ):
        """
        A record written before normalized hashing cannot be told apart
        from a genuine mismatch, so replay reports None (unknown) rather
        than False (drift) — reporting drift would be a false accusation.
        """
        modeling_dispatch("build_model_dataset", _dataset_args())
        record = dict(_read_records()[0])
        record.pop("output_hash_normalized", None)

        result = audit.verify_replay(record)
        assert result.output_match is None
        assert any("predates normalized" in note for note in result.notes)
