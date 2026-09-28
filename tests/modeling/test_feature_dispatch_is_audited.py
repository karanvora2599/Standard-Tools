"""
`feature_dispatch` takes the audited path every other dispatcher takes.

A direct Python call ran the feature-lab tool and returned its dump: no
decision record was written, so `last_request_id()` named nothing and
`explain_decision` had nothing to explain; a non-finite scalar argument
reached the analysis instead of being refused by name; and a NaN statistic
left the dispatcher as NaN. The MCP route already went through the
runtime's audited `dispatch`; the direct door now does too.
"""

import json
import math
from pathlib import Path

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit.dispatch import last_request_id
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent.feature_models import FeatureRedundancyInput
from standard_quant_tools.modeling.agent.feature_tools import (
    feature_dispatch,
    get_feature_redundancy,
)
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec


@pytest.fixture
def audit_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.delenv("SQT_AUDIT_ENABLED", raising=False)
    return directory


def _records(directory: Path):
    return [
        json.loads(line)
        for day in audit._iter_day_files(directory)
        for line in day.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def dataset(patched_multi_factory):
    spec = DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[
            FeatureSpec(id="technical.rsi"),
            FeatureSpec(id="risk.rolling_beta"),
            FeatureSpec(id="risk.realized_volatility"),
        ],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    return build_model_dataset(BuildModelDatasetInput(spec=spec)).dataset_id


def _ablation_with(half_life_days: float) -> dict:
    return {
        "dataset_id": "never-read",
        "spec": {
            "task": "regression",
            "estimator": {"type": "ridge"},
            "validation": {
                "method": "walk_forward",
                "train_window": 150,
                "test_window": 60,
            },
            "weighting": {"method": "time_decay", "half_life_days": half_life_days},
        },
    }


class TestADirectCallIsRecorded:
    def test_it_writes_a_decision_record_that_names_it(self, dataset, audit_dir):
        feature_dispatch("get_feature_redundancy", {"dataset_id": dataset})

        request_id = last_request_id()
        assert request_id is not None
        [record] = [r for r in _records(audit_dir) if r["request_id"] == request_id]
        assert record["tool_name"] == "get_feature_redundancy"
        assert record["input"]["dataset_id"] == dataset
        assert record["status"] == "ok" and record["output_hash"]
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_the_answer_is_the_tools_own(self, dataset, audit_dir):
        """Null case: the path changed, the result did not -- apart from a
        non-finite number arriving as null, which strict JSON needs."""
        payload = feature_dispatch("get_feature_redundancy", {"dataset_id": dataset})
        direct = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset)
        ).model_dump()

        def _same(a, b):
            if isinstance(a, dict):
                return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return len(a) == len(b) and all(map(_same, a, b))
            if a is None and isinstance(b, float):
                return not math.isfinite(b)
            return a == b

        assert _same(payload, direct)
        json.dumps(payload, allow_nan=False)

    def test_with_recording_off_the_result_still_arrives(
        self, dataset, audit_dir, monkeypatch
    ):
        """Null case: SQT_AUDIT_ENABLED=0 is honoured here as everywhere."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        payload = feature_dispatch("get_feature_redundancy", {"dataset_id": dataset})
        assert payload["n_features"] == 3
        assert _records(audit_dir) == [] and last_request_id() is None


class TestItRefusesAtTheDoor:
    def test_a_non_finite_scalar_is_refused_by_name_before_anything_runs(
        self, audit_dir
    ):
        """The dataset id names nothing: the refusal comes first, as at
        every other dispatcher, and writes no record."""
        with pytest.raises(ValidationError, match="half_life_days=inf must be finite"):
            feature_dispatch("run_feature_ablation", _ablation_with(float("inf")))
        assert _records(audit_dir) == [] and last_request_id() is None

    def test_a_finite_scalar_reaches_the_tool(self, audit_dir):
        """Null case: the same call with a finite value gets past the
        policy and is refused by the tool itself, which is recorded."""
        with pytest.raises(Exception) as refused:
            feature_dispatch("run_feature_ablation", _ablation_with(30.0))
        assert "must be finite" not in str(refused.value)
        [record] = _records(audit_dir)
        assert record["tool_name"] == "run_feature_ablation"
        assert record["status"] == "error"

    def test_an_unknown_tool_is_refused_and_forgets_the_last_id(
        self, dataset, audit_dir
    ):
        feature_dispatch("get_feature_redundancy", {"dataset_id": dataset})
        assert last_request_id() is not None
        with pytest.raises(ValidationError, match="unknown feature_lab tool"):
            feature_dispatch("get_feature_everything", {"dataset_id": dataset})
        assert last_request_id() is None
        assert len(_records(audit_dir)) == 1
