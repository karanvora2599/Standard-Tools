"""
Regression tests for the P1 provenance/immutability findings.

The registered-model directory was a collection of individually atomic
files, not a verified package: every artifact below the manifest was plain
JSON or a joblib blob that anything with write access could edit, and no
loader ever checked. These tests tamper with each artifact and assert the
tampering is detected.

Scope note, deliberately: this is INTEGRITY, not authenticity. The manifest
is the root of trust and cannot contain its own digest, so an attacker able
to rewrite both an artifact AND manifest.json is still out of scope —
closing that needs manifest signing, the way audit/signing.py does for
decision records.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.hashing import hash_dataframe
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.registry.feature_provenance import (
    feature_implementation_hash,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_dataset_spec,
    load_manifest,
    load_model,
    load_model_spec,
    load_preprocessing_stats,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
    )


def _train(dataset_id: str = "ds_prov_test") -> str:
    spec = _spec()
    built = build_dataset(spec)
    panel_uri = _artifacts.save_artifact(
        built["panel"], run_id=dataset_id, name="panel"
    )
    directory = Path(panel_uri).parent
    _artifacts.save_json(directory, "dataset_spec", spec.model_dump())

    model_spec = ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )
    dataset = {
        "panel": built["panel"],
        "feature_ids": built["feature_ids"],
        "target_id": built["target_id"],
        "data_hash": built["data_hash"],
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }
    return run_experiment(dataset, model_spec, dataset_id=dataset_id)["model_id"]


@pytest.fixture
def trained_model(patched_multi_factory) -> str:
    return _train()


class TestModelPackageIsSelfContained:
    def test_dataset_spec_is_bundled_with_the_model(self, trained_model):
        """
        score_model used to re-read SQT_RUNS_DIR/<dataset_id>/dataset_spec.json
        on every call, so archiving or deleting the dataset made an
        otherwise-valid model unscoreable.
        """
        directory = _artifacts.run_dir(trained_model)
        assert (directory / "dataset_spec.json").exists()

    def test_scoring_spec_survives_dataset_directory_deletion(self, trained_model):
        import shutil

        shutil.rmtree(_artifacts.run_dir("ds_prov_test"))
        spec = load_dataset_spec(trained_model)
        assert [f["id"] for f in spec["features"]] == [
            "technical.rsi",
            "market.momentum",
        ]


class TestContentHashesRecorded:
    def test_every_artifact_has_a_recorded_digest(self, trained_model):
        manifest = load_manifest(trained_model)
        assert set(manifest.content_hashes) >= {
            "model.joblib",
            "model_spec.json",
            "preprocessing_stats.json",
            "dataset_spec.json",
            "oos_predictions",
        }
        assert all(len(h) == 16 for h in manifest.content_hashes.values())

    def test_dataset_spec_hash_is_persisted(self, trained_model):
        """build_dataset computed spec_hash and then discarded it."""
        assert load_manifest(trained_model).dataset_spec_hash is not None

    def test_feature_implementation_hashes_recorded(self, trained_model):
        manifest = load_manifest(trained_model)
        assert set(manifest.feature_implementation_hashes) == {
            "technical.rsi",
            "market.momentum",
        }
        assert all(v != "" for v in manifest.feature_implementation_hashes.values())

    def test_feature_hash_tracks_the_implementation_not_the_id(self):
        """Two different features must not share an implementation hash."""
        a = feature_implementation_hash("technical.rsi")
        b = feature_implementation_hash("market.momentum")
        assert a != b
        assert feature_implementation_hash("does.not.exist") == "unavailable"


class TestTamperDetection:
    """Each test edits one artifact and asserts the next load rejects it."""

    def test_edited_dataset_spec_is_rejected(self, trained_model):
        """
        The concrete scenario: change an RSI period from 14 to 100 after
        training and every later score_model call silently feeds the
        registered estimator a differently-defined feature.
        """
        path = _artifacts.run_dir(trained_model) / "dataset_spec.json"
        spec = json.loads(path.read_text())
        spec["features"][0]["params"] = {"period": 100}
        path.write_text(json.dumps(spec, indent=2))

        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            load_dataset_spec(trained_model)

    def test_edited_preprocessing_stats_is_rejected(self, trained_model):
        path = _artifacts.run_dir(trained_model) / "preprocessing_stats.json"
        stats = json.loads(path.read_text())
        first = next(iter(stats))
        stats[first]["mean"] = 999.0
        path.write_text(json.dumps(stats, indent=2))

        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            load_preprocessing_stats(trained_model)

    def test_swapped_model_binary_is_rejected_before_deserialization(
        self, trained_model
    ):
        """
        joblib.load executes code from the file, so a swapped binary is an
        arbitrary-code-execution vector. The digest must be checked BEFORE
        deserializing, not after.
        """
        path = _artifacts.run_dir(trained_model) / "model.joblib"
        path.write_bytes(b"not a real joblib payload")

        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            load_model(trained_model)

    def test_edited_model_spec_is_rejected(self, trained_model):
        path = _artifacts.run_dir(trained_model) / "model_spec.json"
        spec = json.loads(path.read_text())
        spec["random_seed"] = 999
        path.write_text(json.dumps(spec, indent=2))

        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            load_model_spec(trained_model)

    def test_untampered_artifacts_all_load(self, trained_model):
        """The guard must not reject a clean package."""
        assert load_model(trained_model) is not None
        assert load_model_spec(trained_model) is not None
        assert load_preprocessing_stats(trained_model)
        assert load_dataset_spec(trained_model)


class TestDatasetHashing:
    def test_dataset_hash_covers_column_names(self, patched_multi_factory):
        """
        Modeling hashed the panel with pd.util.hash_pandas_object, a
        per-row digest blind to column labels — the exact collision the
        audit package was already fixed for.
        """
        a = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})
        b = pd.DataFrame({"other_x": [1.0, 2.0], "other_y": [3.0, 4.0]})
        assert hash_dataframe(a) != hash_dataframe(b)

        legacy_a = pd.util.hash_pandas_object(a, index=True).to_numpy().tobytes()
        legacy_b = pd.util.hash_pandas_object(b, index=True).to_numpy().tobytes()
        assert legacy_a == legacy_b, "demonstrates the collision being fixed"

    def test_builder_uses_the_column_aware_hash(self, patched_multi_factory):
        built = build_dataset(_spec())
        assert built["data_hash"] == hash_dataframe(built["panel"])


class TestTransactionalCommit:
    def test_manifest_is_written_last(self, trained_model):
        """
        manifest.json is the commit point: every loader keys off it, so a
        crash mid-registration leaves a directory that is simply not a
        model rather than a half-written one that looks loadable.
        """
        directory = _artifacts.run_dir(trained_model)
        manifest_mtime = (directory / "manifest.json").stat().st_mtime_ns
        for name in ("model.joblib", "model_spec.json", "preprocessing_stats.json"):
            assert (directory / name).stat().st_mtime_ns <= manifest_mtime

    def test_model_without_manifest_is_not_loadable(self, trained_model):
        (_artifacts.run_dir(trained_model) / "manifest.json").unlink()
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest(trained_model)
