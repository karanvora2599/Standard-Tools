"""
Phase 1B of `Development/modeling_capability_gaps_plan.md`: the capability
report, the adapters, the engine's calibration caveat, and the three
deletions.

WHAT THESE TESTS PIN. Not that the report is well-formed -- that is pinned
elsewhere -- but that each of the eight rows of the 1B table actually
closed, and that the detectors it added do not fire on the case they are
supposed to leave alone. Every class names its row.

The calibration pair is the expensive one and is deliberately the smallest
run that can show the effect: one entity, 260 business days, two features,
`logistic` (which exposes `coef_`, so uncalibrated importances are finite
by construction), three walk-forward folds, `register=False` so nothing
reaches disk.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling import artifacts as modeling_artifacts
from standard_quant_tools.modeling import capabilities as capabilities_module
from standard_quant_tools.modeling import limits
from standard_quant_tools.modeling.adapters import (
    ClassificationAdapter,
    RegressionAdapter,
    available_tasks,
)
from standard_quant_tools.modeling.analysis import feature_ablation
from standard_quant_tools.modeling.capabilities import (
    estimator_capabilities,
    modeling_capabilities,
)
from standard_quant_tools.modeling.engine import (
    _calibration_importance_warning,
    run_experiment,
)
from standard_quant_tools.modeling.estimators import boosting
from standard_quant_tools.modeling.estimators import survival as survival_estimators
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.specs import (
    TARGET_KINDS,
    EstimatorSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.targets.registry import list_targets


@pytest.fixture(scope="module")
def report():
    return modeling_capabilities()


# ── 1B.1 ────────────────────────────────────────────────────────────────


class TestTargetDetailCarriesTheWholeDefinition:
    """1B.1 (G6): `targets.detail` is built from `list_targets()` and
    reports `censored`, `cross_sectional`, `requires`, `param_schema` and
    `default_params` beside the four keys it already had."""

    def test_the_detail_covers_exactly_the_registry(self, report):
        detail = report["targets"]["detail"]
        assert set(detail) == {definition.id for definition in list_targets()}
        assert set(detail) == set(TARGET_KINDS)

    def test_the_four_original_keys_are_unchanged(self, report):
        """`tests/modeling/test_agent_tools.py` reads these; a wider
        detail must not be a different one."""
        for name, entry in report["targets"]["detail"].items():
            assert {"buildable", "tasks", "continuous", "description"} <= set(entry)
            assert entry["description"], name
            assert entry["tasks"], name
            assert entry["buildable"] is TARGET_KINDS[name].buildable

    def test_the_censored_label_says_so(self, report):
        """`time_to_fill` is the only censored label, and
        `register_external_panel` refuses it without an event column."""
        detail = report["targets"]["detail"]
        assert detail["time_to_fill"]["censored"] is True
        assert detail["time_to_fill"]["tasks"] == ["survival"]

    def test_an_uncensored_label_is_not_reported_as_censored(self, report):
        """The null case: the flag distinguishes, it does not decorate."""
        detail = report["targets"]["detail"]
        assert detail["forward_return"]["censored"] is False
        assert [
            name
            for name, entry in detail.items()
            if entry["censored"] and "survival" not in entry["tasks"]
        ] == []

    def test_the_cross_sectional_labels_say_so(self, report):
        detail = report["targets"]["detail"]
        assert detail["forward_return_rank"]["cross_sectional"] is True
        assert detail["forward_return_market_neutral"]["cross_sectional"] is True

    def test_a_per_entity_label_is_not_reported_as_cross_sectional(self, report):
        """The null case. A forward return is a function of one name's own
        prices, so a one-name universe is not degenerate for it."""
        assert report["targets"]["detail"]["forward_return"]["cross_sectional"] is False

    def test_every_label_names_the_columns_it_reads(self, report):
        for name, entry in report["targets"]["detail"].items():
            assert entry["requires"], name

    def test_the_parameter_bounds_come_across(self, report):
        for name, entry in report["targets"]["detail"].items():
            assert isinstance(entry["param_schema"], list), name
            assert entry["param_schema"] == sorted(entry["param_schema"]), name
            assert isinstance(entry["default_params"], dict), name
            # A default that names a parameter the schema does not bound
            # would be a default no spec could restate.
            assert set(entry["default_params"]) <= set(entry["param_schema"]), name

    def test_the_note_explains_both_new_flags(self, report):
        note = report["targets"]["note"]
        assert "event_column" in note
        assert "register_external_panel" in note
        assert "cross_sectional" in note
        assert "one-name universe" in note


# ── 1B.2 ────────────────────────────────────────────────────────────────


class TestTasksReportWhatCanActuallyBeFitted:
    """1B.2 (D-1): `capabilities.tasks` is the `targets` shape now --
    `fitted` / `no_estimator_installed` / `all` / `note` -- read off
    ESTIMATOR_REGISTRY rather than the static adapter table."""

    def test_all_still_names_the_adapter_table(self, report):
        assert report["tasks"]["all"] == available_tasks()

    def test_the_three_lists_partition_cleanly(self, report):
        tasks = report["tasks"]
        assert sorted(tasks["fitted"] + tasks["no_estimator_installed"]) == sorted(
            tasks["all"]
        )
        assert not set(tasks["fitted"]) & set(tasks["no_estimator_installed"])

    def test_a_task_with_an_estimator_is_fitted(self, report):
        """The null case for the detector below: regression has a dozen
        registered estimators and none of them is optional."""
        assert "regression" in report["tasks"]["fitted"]
        assert "regression" not in report["tasks"]["no_estimator_installed"]

    def test_a_task_with_no_estimator_moves(self, monkeypatch):
        """Planted: ranking's estimators all come from lightgbm/xgboost,
        so on a machine without either the registry holds none -- which is
        the environment D-1 was found in. Filtering them out reproduces it
        here, where both libraries happen to be installed."""
        filtered = {
            key: value
            for key, value in ESTIMATOR_REGISTRY.items()
            if key[0] != "ranking"
        }
        monkeypatch.setattr(capabilities_module, "ESTIMATOR_REGISTRY", filtered)
        tasks = modeling_capabilities()["tasks"]
        assert "ranking" in tasks["no_estimator_installed"]
        assert "ranking" not in tasks["fitted"]
        # Still ADVERTISED as a task -- the adapter and the spec validation
        # exist; what is absent is anything to fit.
        assert "ranking" in tasks["all"]

    def test_the_note_names_the_refusal_and_the_cause(self, report):
        note = report["tasks"]["note"]
        assert "validate_model_spec" in note
        assert "optional_dependencies" in note


# ── 1B.3 ────────────────────────────────────────────────────────────────


def _classification_dataset(n: int = 260, seed: int = 0) -> dict:
    """A hand-built single-entity panel with a real binary signal, so
    every walk-forward fold carries both classes and a fitted `logistic`
    has coefficients to report. Not fetched through `build_dataset`: the
    point of this pair is the calibration wrapper, not the builder."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    probability = 1.0 / (1.0 + np.exp(-(1.5 * f0 - 0.8 * f1)))
    target = (rng.random(n) < probability).astype(float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": ["X"] * n,
            "f0": f0,
            "f1": f1,
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["f0", "f1"],
        "target_id": "forward_direction:1",
        "data_hash": "h",
    }


def _classification_spec(calibration: str) -> ModelSpec:
    return ModelSpec(
        task="classification",
        estimator=EstimatorSpec(type="logistic", calibration=calibration),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=0),
        random_seed=0,
    )


@pytest.fixture(scope="module")
def calibration_runs():
    """The same rows and the same seed through both specs, so the only
    difference between the two results is the calibration wrapper."""
    return {
        method: run_experiment(
            _classification_dataset(),
            _classification_spec(method),
            dataset_id="ds_test",
            register=False,
        )
        for method in ("none", "isotonic")
    }


class TestCalibrationSaysWhatItCostsTheImportances:
    """1B.3 (D-2): a calibrated run warns that
    `feature_importance_summary` is NaN by construction, and the report
    grows a `calibration` section saying the importance flags describe the
    uncalibrated estimator."""

    def test_the_uncalibrated_run_reports_finite_importances(self, calibration_runs):
        """The null case. Same spec, same rows, calibration='none'."""
        summary = calibration_runs["none"]["feature_importance_summary"]
        assert set(summary) == {"f0", "f1"}
        for feature, entry in summary.items():
            assert math.isfinite(entry["mean"]), feature
            assert math.isfinite(entry["signed_mean"]), feature

    def test_the_uncalibrated_run_carries_no_such_warning(self, calibration_runs):
        warnings = calibration_runs["none"]["warnings"]
        assert not [w for w in warnings if "CalibratedClassifierCV" in w]

    def test_the_calibrated_run_reports_nan_for_every_feature(self, calibration_runs):
        summary = calibration_runs["isotonic"]["feature_importance_summary"]
        assert set(summary) == {"f0", "f1"}
        for feature, entry in summary.items():
            assert all(math.isnan(value) for value in entry.values()), feature

    def test_the_calibrated_run_says_why(self, calibration_runs):
        named = [
            w
            for w in calibration_runs["isotonic"]["warnings"]
            if "CalibratedClassifierCV" in w
        ]
        assert len(named) == 1, calibration_runs["isotonic"]["warnings"]
        message = named[0]
        assert "feature_importance_summary" in message
        assert "NaN" in message
        # The remedy, and the reason the capability report is not lying.
        assert "calibration='none'" in message
        assert "uncalibrated" in message

    def test_the_calibrated_run_still_produced_metrics(self, calibration_runs):
        """The warning is a caveat, not a failure: the run is otherwise
        exactly the run it was before."""
        assert calibration_runs["isotonic"]["oos_metrics"]["accuracy"] > 0.0
        assert calibration_runs["isotonic"]["n_folds"] == (
            calibration_runs["none"]["n_folds"]
        )

    def test_an_estimator_with_nothing_to_lose_is_not_warned_about(self):
        """The second null case, without a fit: HistGradientBoosting
        exposes neither `coef_` nor `feature_importances_`, so its
        importances are NaN calibrated or not and a warning here would be
        noise on every such run."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        spec = _classification_spec("isotonic")
        assert (
            _calibration_importance_warning(spec, HistGradientBoostingClassifier) == []
        )
        assert _calibration_importance_warning(spec, LogisticRegression) != []
        assert (
            _calibration_importance_warning(
                _classification_spec("none"), LogisticRegression
            )
            == []
        )

    def test_the_report_has_a_calibration_section(self, report):
        section = report["calibration"]
        assert section["methods"] == ["none", "isotonic", "sigmoid"]
        assert section["calibration_folds"]["minimum"] == 2
        assert section["calibration_folds"]["maximum"] == 10
        assert section["calibration_folds"]["default"] == 3

    def test_the_section_says_the_flags_describe_the_uncalibrated_estimator(
        self, report
    ):
        note = report["calibration"]["note"]
        assert "CalibratedClassifierCV" in note
        assert "feature_importance_summary" in note
        assert "UNCALIBRATED" in note

    def test_the_per_estimator_flags_are_unchanged(self, report):
        """Kept as they are ON PURPOSE: they describe the class, which is
        what `fold_feature_importance` finds when nothing wraps it, and
        `generate_modeling_reference.py` prints them."""
        entries = {(e["task"], e["name"]): e for e in report["estimators"]}
        assert entries[("classification", "logistic")]["exposes_coefficients"] is True
        assert (
            entries[("classification", "random_forest")]["exposes_feature_importance"]
            is True
        )


# ── 1B.4 ────────────────────────────────────────────────────────────────


class TestTheTargetSchemaSeparatesBuildableFromExternal:
    """1B.4 (D-9): the JSON-schema enum still lists every registered id --
    an external label is legal on `ExternalTarget.target_type` -- and the
    same hook now writes which of them `build_model_dataset` can actually
    derive."""

    @staticmethod
    def _schema():
        return TargetSpec.model_json_schema()["properties"]["type"]

    def test_the_enum_is_still_the_whole_registry(self):
        assert self._schema()["enum"] == sorted(TARGET_KINDS)

    def test_the_buildable_subset_is_in_the_schema(self):
        schema = self._schema()
        assert schema["x-buildable"] == sorted(
            name for name, kind in TARGET_KINDS.items() if kind.buildable
        )
        assert set(schema["x-buildable"]) < set(schema["enum"])

    def test_an_external_label_is_offered_but_not_as_buildable(self):
        """The null case for the split: `future_mid_return` stays legal at
        the boundary and is no longer advertised as derivable."""
        schema = self._schema()
        assert "future_mid_return" in schema["enum"]
        assert "future_mid_return" not in schema["x-buildable"]
        assert "forward_return" in schema["x-buildable"]

    def test_the_description_names_the_way_in_for_the_others(self):
        description = self._schema()["description"]
        assert "register_external_panel" in description
        assert "forward_return" in description
        assert "future_mid_return" in description

    def test_the_external_panel_input_gets_the_same_hook(self):
        from standard_quant_tools.modeling.agent.models import ExternalTarget

        schema = ExternalTarget.model_json_schema()["properties"]["target_type"]
        assert schema["enum"] == sorted(TARGET_KINDS)
        assert "time_to_fill" in schema["enum"]
        assert "time_to_fill" not in schema["x-buildable"]


# ── 1B.5 ────────────────────────────────────────────────────────────────


class TestThePartialFitFlagIsGone:
    """1B.5: `supports_partial_fit` described sklearn rather than this
    runtime and nothing read it."""

    def test_no_estimator_entry_carries_it(self):
        assert [
            entry["name"]
            for entry in estimator_capabilities()
            if "supports_partial_fit" in entry
        ] == []

    def test_no_adapter_emits_it(self):
        from sklearn.linear_model import LogisticRegression, SGDRegressor

        classification = ClassificationAdapter().capabilities(LogisticRegression)
        # SGD is one of the four that reported True.
        regression = RegressionAdapter().capabilities(SGDRegressor)
        assert "supports_partial_fit" not in classification
        assert "supports_partial_fit" not in regression

    def test_the_flags_that_do_describe_this_runtime_survive(self):
        """The null case: a narrower dict, not a broken one."""
        from sklearn.linear_model import LogisticRegression

        capabilities = ClassificationAdapter().capabilities(LogisticRegression)
        assert {
            "task",
            "input_kind",
            "needs_groups",
            "score_has_scale",
            "supports_sample_weight",
            "supports_probability",
            "accepts_missing",
            "exposes_coefficients",
            "exposes_feature_importance",
        } == set(capabilities)


# ── 1B.6 ────────────────────────────────────────────────────────────────


class TestOneDefaultMaxFitsExists:
    """1B.6: `feature_ablation.DEFAULT_MAX_FITS = 200` collided by name
    with the live `limits.DEFAULT_MAX_FITS = 500` and was read by nothing.
    Deleted without re-pointing the ablation input, which would have
    raised its cap two and a half times in silence."""

    def test_the_ablation_module_no_longer_declares_one(self):
        assert not hasattr(feature_ablation, "DEFAULT_MAX_FITS")
        assert "DEFAULT_MAX_FITS" not in feature_ablation.__all__

    def test_the_live_one_is_untouched(self):
        assert limits.DEFAULT_MAX_FITS == 500

    def test_the_ablation_cap_did_not_move(self):
        """The null case, and the trap the deletion could have sprung: the
        input's own 200 stands on its own."""
        from standard_quant_tools.modeling.agent.feature_models import (
            FeatureAblationInput,
        )

        assert FeatureAblationInput.model_fields["max_fits"].default == 200


# ── 1B.7 ────────────────────────────────────────────────────────────────


class TestSurvivalPublishesOneXgboostFlag:
    """1B.7: `HAS_XGBOOST_SURVIVAL` duplicated `boosting.HAS_XGBOOST` --
    the flag the capability report publishes -- while the call that
    produced it did the registration that actually matters."""

    def test_the_duplicate_name_is_gone(self):
        assert not hasattr(survival_estimators, "HAS_XGBOOST_SURVIVAL")
        assert "HAS_XGBOOST_SURVIVAL" not in survival_estimators.__all__

    def test_the_registration_still_happened(self):
        """The whole point of keeping the call: the name went, the side
        effect stayed."""
        if not boosting.HAS_XGBOOST:  # pragma: no cover - environment
            pytest.skip("xgboost is not installed")
        assert ("survival", "xgboost_cox") in ESTIMATOR_REGISTRY
        assert ("survival", "xgboost_aft") in ESTIMATOR_REGISTRY

    def test_the_published_flag_agrees_with_the_registry(self, report):
        """The null case: absent xgboost, both the flag and the pair go."""
        reported = report["optional_dependencies"]["xgboost"]
        assert reported is boosting.HAS_XGBOOST
        assert (("survival", "xgboost_cox") in ESTIMATOR_REGISTRY) is reported

    def test_the_survival_baseline_is_registered_either_way(self):
        assert ("survival", "cox_ph") in ESTIMATOR_REGISTRY


# ── 1B.8 ────────────────────────────────────────────────────────────────


class TestLocalStoreIsGone:
    """1B.8: `modeling.artifacts.local_store` had zero callers in `src/`
    and `tests/` and was documented nowhere, despite an earlier survey
    keeping it as "the documented entry point"."""

    def test_the_function_and_its_export_are_gone(self):
        assert not hasattr(modeling_artifacts, "local_store")
        assert "local_store" not in modeling_artifacts.__all__

    def test_the_path_based_helpers_are_untouched(self):
        """The null case: the module's actual surface still stands."""
        for name in ("run_dir", "save_json", "load_json", "hash_file", "verify_file"):
            assert name in modeling_artifacts.__all__
            assert hasattr(modeling_artifacts, name)

    def test_the_callers_that_want_a_local_store_build_it_themselves(self):
        """Where the capability actually lives, which is why deleting the
        wrapper removes nothing."""
        from standard_quant_tools.artifact_store import LocalArtifactStore
        from standard_quant_tools.modeling.registry import package

        assert package.LocalArtifactStore is LocalArtifactStore
