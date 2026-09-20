"""
Lifecycle stages and monitoring: where a model is on the way from fitted
to trusted, and whether the world it scores still looks like the world it
was fitted on.

A promotion is a decision, recorded in an append-only log beside the
manifest; the manifest itself is never touched, which the test checks by
hashing its bytes across a promotion. Monitoring is checked both ways:
identical inputs read as stable, inputs shifted by five standard
deviations read as severe, and a realized IC that matches the validation
reads as stable while one with the opposite sign reads as severe.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    InspectModelInput,
    ListModelsInput,
    MonitorModelInput,
    PromoteModelInput,
)
from standard_quant_tools.modeling.agent.tools import (
    MODELING_TOOL_DISPATCH,
    inspect_model,
    list_models,
    monitor_model,
    promote_model,
)
from standard_quant_tools.modeling.monitoring import (
    PSI_SEVERE,
    drift_report,
    prediction_drift,
    realized_ic,
)
from standard_quant_tools.modeling.registry.lifecycle import (
    STAGES,
    current_stage,
    promote,
    promotions,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_monitoring_reference,
)
from standard_quant_tools.modeling.scoring import score_model

from .test_scoring import _dataset_spec, _train_a_model_with_spec

UNIVERSE = ["AAA", "BBB", "CCC"]


class TestLifecycle:
    def test_a_model_starts_as_a_candidate_and_climbs_one_stage_at_a_time(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_lifecycle")
        assert current_stage(model_id) == "candidate"
        assert promotions(model_id) == []
        with pytest.raises(ValidationError, match="skips"):
            promote(model_id, "production", "it looked good in the notebook")
        record = promote(
            model_id,
            "validated",
            "rank IC held on every fold",
            actor="reviewer",
            evidence=["sqt://runs/ds_lifecycle/panel"],
        )
        assert record.from_stage == "candidate" and record.to_stage == "validated"
        assert current_stage(model_id) == "validated"
        promote(model_id, "staging", "paper trading for two weeks")
        promote(model_id, "production", "paper results matched the validation")
        with pytest.raises(ValidationError, match="already at"):
            promote(model_id, "production", "promoting it again by mistake")
        # A demotion is a decision too, and is recorded like one.
        promote(model_id, "staging", "two features drifted; rolling back")
        promote(model_id, "archived", "replaced by a retrained model")
        with pytest.raises(ValidationError, match="terminal"):
            promote(model_id, "validated", "trying to revive it after all")
        history = promotions(model_id)
        assert [p.to_stage for p in history] == [
            "validated",
            "staging",
            "production",
            "staging",
            "archived",
        ]
        assert history[0].actor == "reviewer" and history[0].evidence == [
            "sqt://runs/ds_lifecycle/panel"
        ]
        # Append-only JSON lines, one per decision, every field present.
        log = _artifacts.run_dir(model_id) / "promotions.jsonl"
        lines = [json.loads(line) for line in log.read_text().splitlines() if line]
        assert len(lines) == 5
        assert set(lines[0]) == {
            "from_stage",
            "to_stage",
            "reason",
            "actor",
            "timestamp_utc",
            "evidence",
        }

    def test_the_manifest_is_untouched_by_a_promotion(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_immutable")
        path = _artifacts.run_dir(model_id) / "manifest.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        promote(model_id, "validated", "the evidence was read and accepted")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert "stage" not in json.loads(path.read_text(encoding="utf-8"))

    def test_bad_promotions_are_refused_by_name(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_refusals")
        with pytest.raises(ValidationError, match="not a lifecycle stage"):
            promote(model_id, "deployed", "a stage that does not exist")
        with pytest.raises(ValidationError, match="reason"):
            promote(model_id, "validated", "ok")
        with pytest.raises(ValidationError, match="no registered model"):
            promote("mdl_nope", "validated", "a model that does not exist")
        with pytest.raises(PydanticValidationError):
            PromoteModelInput(
                model_id=model_id, to_stage="deployed", reason="long enough"
            )
        assert STAGES == ("candidate", "validated", "staging", "production", "archived")

    def test_the_tools_record_filter_and_show_the_stage(self, patched_multi_factory):
        a = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_tools_a")
        b = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_tools_b")
        result = promote_model(
            PromoteModelInput(
                model_id=a, to_stage="validated", reason="folds agree on the sign"
            )
        )
        assert (result.from_stage, result.to_stage) == ("candidate", "validated")
        assert result.history[-1]["reason"] == "folds agree on the sign"
        validated = list_models(ListModelsInput(stage="validated"))
        assert [m.model_id for m in validated.models] == [a]
        candidates = list_models(ListModelsInput(stage="candidate"))
        assert b in {m.model_id for m in candidates.models}
        assert a not in {m.model_id for m in candidates.models}
        everything = list_models(ListModelsInput())
        assert {m.model_id: m.stage for m in everything.models}.items() >= {
            a: "validated",
            b: "candidate",
        }.items()
        summary = inspect_model(InspectModelInput(model_id=a, view="summary")).data
        assert summary["stage"] == "validated"
        assert summary["promotions"][0]["to_stage"] == "validated"


class TestMonitoring:
    def test_registration_keeps_a_reference_and_scoring_keeps_the_features(
        self, patched_multi_factory
    ):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_monitor")
        manifest = load_manifest(model_id)
        assert manifest.monitoring["feature_reference_rows"] > 0
        assert manifest.monitoring["prediction_reference_rows"] > 0
        assert {
            "feature_profile.json",
            "feature_reference",
            "prediction_reference",
        } <= set(manifest.content_hashes)
        profile, features, predictions = load_monitoring_reference(model_id)
        assert set(profile["features"]) == set(manifest.feature_ids)
        assert all(
            len(entry["quantile_edges"]) == 11 for entry in profile["features"].values()
        )
        assert list(features.columns) == ["date", "entity", *manifest.feature_ids]
        assert list(predictions.columns) == ["date", "entity", "prediction"]

        scored = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert scored["features_uri"]
        current = _artifacts.load_artifact(scored["features_uri"])
        assert list(current.columns) == ["entity", "date", *manifest.feature_ids]
        assert len(current) == scored["n_entities"]

        report = monitor_model(
            MonitorModelInput(
                model_id=model_id, predictions_uri=scored["predictions_uri"]
            )
        )
        assert report.features_uri == scored["features_uri"]
        assert report.stage == "candidate"
        assert [r.feature for r in report.feature_drift] == manifest.feature_ids
        for row in report.feature_drift:
            assert np.isfinite(row.psi) and row.psi >= 0.0
            assert 0.0 <= row.ks <= 1.0
            assert row.status in {"stable", "moderate", "severe"}
        assert report.prediction_drift["n_current"] == scored["n_entities"]
        assert report.prediction_drift["n_reference"] > 0
        assert report.thresholds["psi_severe"] == PSI_SEVERE
        assert report.overall_status in {"stable", "moderate", "severe"}
        assert report.realized_ic is None and report.warnings == []

    def test_planted_drift_is_severe_and_no_drift_is_stable(self):
        rng = np.random.default_rng(0)
        reference = pd.DataFrame(
            {"a": rng.normal(size=4000), "b": rng.normal(size=4000)}
        )
        same = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        rows = {r["feature"]: r for r in drift_report(reference, same, ["a", "b"])}
        assert rows["a"]["status"] == "stable" and rows["a"]["psi"] < 0.1
        shifted = same.assign(a=same["a"] + 5.0)
        rows = {r["feature"]: r for r in drift_report(reference, shifted, ["a", "b"])}
        assert rows["a"]["status"] == "severe"
        assert rows["a"]["psi"] > PSI_SEVERE and rows["a"]["ks"] > 0.5
        assert rows["b"]["status"] == "stable"
        # A feature the current frame lacks is unknown, not silently skipped.
        rows = drift_report(reference, same[["a"]], ["a", "b"])
        assert rows[1]["status"] == "unknown"
        drift = prediction_drift(reference["a"].to_numpy(), shifted["a"].to_numpy())
        assert (
            drift["status"] == "severe"
            and drift["mean_current"] > drift["mean_reference"]
        )

    def test_realized_ic_sits_where_the_validation_said_or_does_not(self):
        rng = np.random.default_rng(1)
        predictions = pd.DataFrame(
            {
                "entity": [f"E{i:02d}" for i in range(30)],
                "date": pd.Timestamp("2024-01-05"),
                "prediction": rng.normal(size=30),
            }
        )
        agreeing = predictions[["entity"]].assign(
            realized=predictions["prediction"] + rng.normal(scale=0.2, size=30)
        )
        good = realized_ic(
            predictions, agreeing, validation_ic_mean=0.05, validation_ic_std=0.10
        )
        assert good["realized_ic"] > 0.8 and good["status"] == "stable"
        assert good["z_versus_validation"] > 2.0 and good["n_matched"] == 30
        opposite = predictions[["entity"]].assign(realized=-predictions["prediction"])
        bad = realized_ic(
            predictions, opposite, validation_ic_mean=0.05, validation_ic_std=0.10
        )
        assert bad["realized_ic"] < -0.9 and bad["status"] == "severe"
        unknown = realized_ic(
            predictions, agreeing, validation_ic_mean=None, validation_ic_std=None
        )
        assert unknown["status"] == "unknown" and unknown["z_versus_validation"] is None
        with pytest.raises(ValidationError, match="entity"):
            realized_ic(
                predictions,
                pd.DataFrame({"realized": [1.0]}),
                validation_ic_mean=0,
                validation_ic_std=1,
            )
        with pytest.raises(ValidationError, match="at least 3"):
            realized_ic(
                predictions, agreeing.head(2), validation_ic_mean=0, validation_ic_std=1
            )

    def test_outcomes_reach_the_tool(self, patched_multi_factory, tmp_path):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_outcomes")
        scored = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        predictions = _artifacts.load_artifact(scored["predictions_uri"])
        outcomes = predictions[["entity"]].assign(realized=predictions["prediction"])
        # Outcomes are an artifact like everything else the tools read: a
        # path outside SQT_RUNS_DIR is refused by the artifact layer.
        outcomes_uri = _artifacts.save_artifact(
            outcomes, run_id=model_id, name="outcomes"
        )
        with pytest.raises(ValidationError, match="escapes"):
            monitor_model(
                MonitorModelInput(
                    model_id=model_id,
                    predictions_uri=scored["predictions_uri"],
                    outcomes_ref=str(tmp_path / "elsewhere.parquet"),
                )
            )
        report = monitor_model(
            MonitorModelInput(
                model_id=model_id,
                predictions_uri=scored["predictions_uri"],
                outcomes_ref=outcomes_uri,
            )
        )
        assert report.realized_ic is not None
        assert report.realized_ic["realized_ic"] == pytest.approx(1.0)
        assert report.realized_ic["n_matched"] == len(predictions)

    def test_a_model_without_references_is_refused_by_name(
        self, patched_multi_factory, monkeypatch
    ):
        from standard_quant_tools.modeling.agent import tools as tools_module

        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_legacy_mon")
        scored = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        monkeypatch.setattr(
            tools_module, "load_monitoring_reference", lambda _id: ({}, None, None)
        )
        with pytest.raises(ValidationError, match="Retrain"):
            monitor_model(
                MonitorModelInput(
                    model_id=model_id, predictions_uri=scored["predictions_uri"]
                )
            )

    def test_the_surface_grew_by_exactly_two(self):
        assert len(MODELING_TOOL_DISPATCH) == 22
        assert {"promote_model", "monitor_model"} <= set(MODELING_TOOL_DISPATCH)
