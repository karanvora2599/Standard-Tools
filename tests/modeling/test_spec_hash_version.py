"""
The dataset spec hash is versioned, and version 2 survives an additive field.

`dataset_spec_hash` covered every field of `DatasetSpec`, so adding one
with a default changed the hash of every persisted dataset and made
`run_model_experiment` refuse them -- 15_modeling.md's "One upgrade note"
records it happening for `horizons`. The plan adds several dataset-level
fields (a missing-data policy, a calendar, point-in-time features), and
each would have been another mass invalidation.

Version 2 hashes the spec with default-valued fields excluded, so a field
nobody set does not enter the identity of a dataset built before it
existed. The metadata records which version produced the stored hash, and
the verifier recomputes with that version: a dataset recorded under
version 1 keeps verifying under version 1.
"""

import hashlib
import json

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    run_model_experiment,
)
from standard_quant_tools.modeling.dataset.builder import (
    SPEC_HASH_VERSION,
    dataset_spec_hash,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _spec_kwargs():
    return dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


class WiderSpec(DatasetSpec):
    """A DatasetSpec with one more defaulted field: what the next release
    looks like to a dataset persisted under this one."""

    later_field: int = 0


class TestTheHashSurvivesAnAdditiveField:
    def test_version_2_is_unchanged_by_a_field_nobody_set(self):
        kwargs = _spec_kwargs()
        today = dataset_spec_hash(DatasetSpec(**kwargs), version=2)
        tomorrow = dataset_spec_hash(WiderSpec(**kwargs), version=2)
        assert today == tomorrow

    def test_version_1_was_not(self):
        """The failure being removed, kept so the property above cannot pass
        for a reason that has nothing to do with excluding defaults."""
        kwargs = _spec_kwargs()
        today = dataset_spec_hash(DatasetSpec(**kwargs), version=1)
        tomorrow = dataset_spec_hash(WiderSpec(**kwargs), version=1)
        assert today != tomorrow

    def test_version_2_still_sees_a_field_that_was_set(self):
        kwargs = _spec_kwargs()
        assert dataset_spec_hash(
            WiderSpec(**kwargs, later_field=1), version=2
        ) != dataset_spec_hash(DatasetSpec(**kwargs), version=2)

    def test_an_explicit_default_is_the_same_spec(self):
        """provider='yfinance' typed out and provider omitted describe one
        dataset, and under version 2 they are one hash."""
        kwargs = _spec_kwargs()
        assert dataset_spec_hash(DatasetSpec(**kwargs, provider="yfinance")) == (
            dataset_spec_hash(DatasetSpec(**kwargs))
        )

    def test_a_changed_feature_parameter_still_changes_it(self):
        kwargs = _spec_kwargs()
        changed = dict(kwargs, features=[FeatureSpec(id="technical.rsi", params={"period": 21})])
        assert dataset_spec_hash(DatasetSpec(**kwargs)) != dataset_spec_hash(
            DatasetSpec(**changed)
        )

    def test_version_1_is_the_full_dump(self):
        spec = DatasetSpec(**_spec_kwargs())
        expected = hashlib.sha256(spec.model_dump_json().encode()).hexdigest()
        assert dataset_spec_hash(spec, version=1) == expected

    def test_the_default_is_the_current_version(self):
        spec = DatasetSpec(**_spec_kwargs())
        assert SPEC_HASH_VERSION == 2
        assert dataset_spec_hash(spec) == dataset_spec_hash(spec, version=2)

    def test_an_unknown_version_is_refused(self):
        with pytest.raises(ValidationError, match="version"):
            dataset_spec_hash(DatasetSpec(**_spec_kwargs()), version=3)


class TestTheVersionTravelsWithTheDataset:
    @pytest.fixture
    def dataset_id(self, patched_multi_factory):
        return build_model_dataset(
            BuildModelDatasetInput(spec=DatasetSpec(**_spec_kwargs()))
        ).dataset_id

    def _meta_path(self, dataset_id):
        return _artifacts.run_dir(dataset_id) / "dataset_meta.json"

    def test_a_built_dataset_records_version_2(self, dataset_id):
        meta = _artifacts.load_json(str(self._meta_path(dataset_id)))
        assert meta["spec_hash_version"] == 2
        assert meta["spec_hash"] == dataset_spec_hash(DatasetSpec(**_spec_kwargs()), version=2)

    def test_the_experiment_verifies_and_the_manifest_records_it(self, dataset_id):
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_model_spec())
        )
        manifest = load_manifest(result.model_id)
        assert manifest.dataset_spec_hash_version == 2
        assert manifest.dataset_spec_hash == dataset_spec_hash(
            DatasetSpec(**_spec_kwargs()), version=2
        )

    def test_a_dataset_recorded_under_version_1_still_verifies(self, dataset_id):
        """The shape every dataset persisted before this release has: a
        version-1 hash and no version key."""
        path = self._meta_path(dataset_id)
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["spec_hash"] = dataset_spec_hash(DatasetSpec(**_spec_kwargs()), version=1)
        del meta["spec_hash_version"]
        path.write_text(json.dumps(meta), encoding="utf-8")
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_model_spec())
        )
        assert load_manifest(result.model_id).dataset_spec_hash_version == 1

    def test_an_edited_spec_is_still_refused(self, dataset_id):
        spec_path = _artifacts.run_dir(dataset_id) / "dataset_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["features"][0]["params"] = {"period": 100}
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with pytest.raises(ValidationError, match="no longer matches"):
            run_model_experiment(
                RunModelExperimentInput(dataset_id=dataset_id, spec=_model_spec())
            )
