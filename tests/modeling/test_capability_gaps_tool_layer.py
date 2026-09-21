"""
The capability-gaps fixes of 2026-09-21 (CHANGELOG): the modeling
tool layer says what is true and returns what it already computed.

Nine rows, nine planted answers. Two tool descriptions advertised a
capability that did not exist (1A.1, 1A.2); one input was missing so every
external prediction was scored against an oracle (1A.3); five numbers were
computed, bound to a name and thrown away before the tool boundary (1A.4
turnover, 1A.5 the record set a feature reads, 1A.6 which NAME lost the
rows, 1A.7 the profile drift is measured against, 1A.8 "this is rsi at lag
3"); and one result carried caveats in a field nothing reads as a caveat
(1A.9).

Every detector here has its null case beside it: a signal whose ordering
never moves reports zero turnover and one entity reports none; a feature
computed from bars alone reports no frame_kind; a spec with no lags
produces an empty label map. A detector that fires on everything has found
nothing.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import ConvertReferenceInput
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.meta.tools import convert_reference
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.dataset_tools import (
    ExplainRowLossInput,
    explain_dataset_row_loss,
)
from standard_quant_tools.modeling.agent.models import (
    AnalyzeModelErrorsInput,
    BuildEnsembleInput,
    BuildEnsembleResult,
    BuildModelDatasetInput,
    InspectModelInput,
    ListFeaturesInput,
    MonitorModelInput,
    RunModelExperimentInput,
    RunModelExperimentResult,
    ScorePredictionsInput,
    ScorePredictionsResult,
)
from standard_quant_tools.modeling.agent.tools import (
    _MODELING_TOOL_DEFS,
    analyze_model_errors,
    build_model_dataset,
    build_model_ensemble,
    inspect_model,
    list_features,
    monitor_model,
    run_model_experiment,
    score_predictions,
)
from standard_quant_tools.modeling.features.base import (
    FeatureDefinition,
    FeatureScope,
    TemporalSupport,
)
from standard_quant_tools.modeling.features.registry import (
    FEATURE_REGISTRY,
    register_feature,
)
from standard_quant_tools.modeling.monitoring import PROFILE_BINS
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    MissingDataSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.search import rank_turnover

from .conftest import make_ohlcv, make_provider_mock

UNIVERSE = ["AAA", "BBB", "CCC"]


# ── helpers ─────────────────────────────────────────────────────────────


def _spec(features=None, **overrides) -> DatasetSpec:
    fields = dict(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=features
        or [FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    fields.update(overrides)
    return DatasetSpec(**fields)


def _ridge(seed: int = 1) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=seed,
    )


def _panel(n_entities: int = 12, n_dates: int = 40, seed: int = 0) -> pd.DataFrame:
    """A prediction frame whose ORDERING is planted: entity E0i always
    ranks i-th, so the turnover of this frame is exactly zero."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2023-01-02", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    order = np.tile(np.arange(n_entities, dtype=float), n_dates)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "prediction": order,
            # A target whose mean is 5.0 and which the ordering predicts
            # weakly: the level is what makes an honest baseline at
            # train_mean=0.0 score below zero.
            "target": order + 5.0 + rng.normal(scale=3.0, size=order.size),
        }
    )


def _publish(frame: pd.DataFrame, name: str) -> str:
    return handoff.publish(frame, kind="predictions", run_id="scored_frames", name=name)


# ── 1A.1 ────────────────────────────────────────────────────────────────


def _ensemble_description() -> str:
    return next(
        description
        for name, description, _input in _MODELING_TOOL_DEFS
        if name == "build_model_ensemble"
    )


class TestTheEnsembleRefSaysWhatItCarries:
    """
    Pins 1A.1 (D-12). `build_model_ensemble`'s description said the
    published reference is one "that score_predictions and the backtest
    bridge read like any other". The backtest half is true; the scoring
    half never was -- the ref carries date, entity and prediction, and
    score_predictions refuses a frame with no `target`. The SENTENCE is
    made honest here, not the behaviour: the refusal below is the same
    refusal as before, and it is now the one the description predicts.
    """

    def test_the_description_no_longer_claims_scoring_reads_it(self):
        description = _ensemble_description()
        assert "reads like any other" in description  # the backtest half stands
        assert "score_predictions and the backtest bridge read like any other" not in (
            description
        )
        assert "NO realized outcome" in description
        assert "refuses" in description

    def test_the_ref_field_carries_the_same_caveat(self):
        description = BuildEnsembleResult.model_fields["ref"].description
        assert "NO realized outcome" in description
        assert "no 'target' column" in description

    def test_the_published_ref_still_refuses_to_be_scored(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec())
        ).dataset_id
        model_ids = [
            run_model_experiment(
                RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge(seed))
            ).model_id
            for seed in (1, 2)
        ]
        result = build_model_ensemble(
            BuildEnsembleInput(
                model_ids=model_ids, run_id="ensemble_caveat", name="combined"
            )
        )
        frame = handoff.resolve(result.ref, expect="predictions")
        assert "target" not in frame.columns
        assert any("no realized outcome" in w for w in result.warnings)
        with pytest.raises(ValidationError, match="no 'target' column"):
            score_predictions(
                ScorePredictionsInput(predictions_ref=result.ref, task="regression")
            )


# ── 1A.2 ────────────────────────────────────────────────────────────────


class TestConvertReferenceTaskDescription:
    """
    Pins 1A.2 (D-13). The field said `task` was "Required unless the
    reference carries a model_id" -- a condition `convert.py` never
    checks, in a module where the string `model_id` does not appear. The
    real rule is by destination kind, and both halves of it are planted
    below.
    """

    def test_the_description_names_the_kinds_and_not_a_model_id(self):
        description = ConvertReferenceInput.model_fields["task"].description
        assert "model_id" not in description
        assert "signal_panel" in description and "score_panel" in description
        assert "REQUIRED" in description and "OPTIONAL" in description

    def test_a_signal_panel_refuses_without_a_task(self):
        ref = _publish(_panel(), "convert_signal")
        with pytest.raises(ValidationError, match="needs `task`"):
            convert_reference(
                ConvertReferenceInput(
                    ref=ref,
                    to_kind="signal_panel",
                    run_id="convert_task",
                    name="signal",
                )
            )

    def test_a_score_panel_passes_the_predictions_through_without_one(self):
        frame = _panel()
        ref = _publish(frame, "convert_score")
        result = convert_reference(
            ConvertReferenceInput(
                ref=ref, to_kind="score_panel", run_id="convert_task", name="scores"
            )
        )
        assert result.kind == "score_panel"
        assert result.entities == frame["entity"].nunique()
        assert any("passed through unchanged" in note for note in result.notes)


# ── 1A.3 ────────────────────────────────────────────────────────────────


class TestTheBaselineIsNotAnOracle:
    """
    Pins 1A.3. `score_predictions` called `baseline_regression_metrics`
    with one argument and `regression_metrics` without `train_y`, so every
    prediction ever scored through this tool was compared against the
    SCORED set's own mean -- a constant nobody could have known in
    advance, whose R2 is 0.0 by construction. `train_mean` is the one
    number that fixes it, and the same frame is scored both ways here.
    """

    def test_the_same_frame_scored_both_ways(self):
        ref = _publish(_panel(), "baseline")
        oracle = score_predictions(
            ScorePredictionsInput(predictions_ref=ref, task="regression")
        )
        # The planted target has mean ~5.0 + the ordering; a forecaster who
        # had seen only training data where the outcome averaged zero would
        # have predicted 0.0, and that constant is far worse than the
        # oracle's.
        honest = score_predictions(
            ScorePredictionsInput(
                predictions_ref=ref, task="regression", train_mean=0.0
            )
        )
        assert oracle.baseline["baseline_is_oracle"] == 1.0
        assert honest.baseline["baseline_is_oracle"] == 0.0
        assert oracle.baseline["baseline_r2"] == 0.0
        assert honest.baseline["baseline_r2"] < 0
        assert honest.baseline["baseline_mae"] > oracle.baseline["baseline_mae"]

    def test_both_metric_calls_get_it_not_just_the_baseline_block(self):
        """`regression_metrics` embeds the baseline too, and it was the
        call that silently kept the oracle when only the other was fixed."""
        ref = _publish(_panel(), "baseline_metrics")
        honest = score_predictions(
            ScorePredictionsInput(
                predictions_ref=ref, task="regression", train_mean=0.0
            )
        )
        assert honest.metrics["baseline_is_oracle"] == 0.0
        assert honest.metrics["baseline_r2"] == honest.baseline["baseline_r2"]

    def test_the_oracle_run_says_so_in_both_notes_and_warnings(self):
        ref = _publish(_panel(), "baseline_notes")
        oracle = score_predictions(
            ScorePredictionsInput(predictions_ref=ref, task="regression")
        )
        honest = score_predictions(
            ScorePredictionsInput(
                predictions_ref=ref, task="regression", train_mean=0.0
            )
        )
        assert any("SCORED set's own mean" in note for note in oracle.notes)
        assert any("baseline_is_oracle=1.0" in w for w in oracle.warnings)
        assert not any("baseline_is_oracle=1.0" in w for w in honest.warnings)


# ── 1A.4 ────────────────────────────────────────────────────────────────


class TestPredictionTurnover:
    """
    Pins 1A.4. `rank_turnover` was live inside the hyperparameter search
    and absent from its own module's `__all__`; no tool reported the
    turnover of a SIGNAL, which is the bridge between an IC and a
    net-of-cost P&L. Three planted orderings: one that never moves, one
    reversed on every date, and one with no cross-section at all.
    """

    def test_it_is_exported(self):
        from standard_quant_tools.modeling.validation import search

        assert "rank_turnover" in search.__all__

    def test_a_constant_ordering_never_turns_over(self):
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=_publish(_panel(), "turnover_flat"), task="regression"
            )
        )
        assert result.prediction_turnover == 0.0

    def test_an_ordering_reversed_every_date_turns_over_half(self):
        n_entities, n_dates = 12, 40
        frame = _panel(n_entities, n_dates)
        base = np.arange(n_entities, dtype=float)
        frame["prediction"] = np.concatenate(
            [base if k % 2 == 0 else base[::-1] for k in range(n_dates)]
        )
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=_publish(frame, "turnover_reversed"),
                task="regression",
            )
        )
        # The planted answer, computed by hand: reversing the percentile
        # ranks of an even cross-section moves each name by |2i - n - 1|/n,
        # which averages to exactly one half.
        assert result.prediction_turnover == pytest.approx(0.5)
        assert result.prediction_turnover == pytest.approx(
            rank_turnover(
                frame["prediction"].to_numpy(),
                frame["date"].to_numpy(),
                frame["entity"].to_numpy(),
            )
        )
        assert any("prediction_turnover" in w for w in result.warnings)

    def test_one_entity_has_no_turnover_to_report(self):
        frame = _panel()
        frame = frame[frame["entity"] == "E00"].copy()
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=_publish(frame, "turnover_single"), task="regression"
            )
        )
        # None, not 0.0: an ordering of one cannot change, and zero would
        # read as "costless" rather than "undefined".
        assert result.prediction_turnover is None

    def test_a_frame_without_an_entity_column_is_not_asked(self):
        frame = _panel(n_entities=1).drop(columns=["entity"])
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=_publish(frame, "turnover_no_entity"),
                task="regression",
            )
        )
        assert result.prediction_turnover is None


# ── 1A.5 ────────────────────────────────────────────────────────────────


class TestTheCatalogNamesTheRecordSet:
    """
    Pins 1A.5. The registry has carried `frame_kind` and `fields` since
    point-in-time features existed and `list_features` dropped both, so
    the one thing an agent could not learn from the catalog was that
    `fundamental.*` reads a record set a bars-only provider does not
    serve. Exactly three features declare them, and the null case is every
    other entry.
    """

    def test_the_fundamentals_name_their_frame_and_fields(self):
        catalog = list_features(ListFeaturesInput()).features
        fundamentals = [e for e in catalog if e.id.startswith("fundamental.")]
        assert len(fundamentals) == 3
        for entry in fundamentals:
            assert entry.frame_kind == "fundamentals"
            assert entry.fields
            assert all(isinstance(f, str) and f for f in entry.fields)
            assert "max_staleness_days" in entry.default_params

    def test_every_other_entry_declares_neither(self):
        catalog = list_features(ListFeaturesInput()).features
        others = [e for e in catalog if not e.id.startswith("fundamental.")]
        assert others
        assert all(e.frame_kind is None and e.fields == [] for e in others)

    def test_the_declaring_entries_are_exactly_the_point_in_time_ones(self):
        catalog = list_features(ListFeaturesInput()).features
        assert {e.id for e in catalog if e.frame_kind} == {
            e.id for e in catalog if e.scope == "point_in_time"
        }


# ── 1A.6 ────────────────────────────────────────────────────────────────


def _short_gap(ohlcv, context, **params):
    """Close, blanked over a long block for the SHORT-history entity only.

    The feature function is handed one entity's OHLCV and no name, so the
    frame's LENGTH is the identity: the provider below gives CCC 260 bars
    and everyone else 500. The planted answer is that CCC alone loses
    about 234 rows while AAA and BBB lose their fourteen bars of RSI
    warm-up each.
    """
    series = ohlcv["Close"].copy()
    if len(series) < 400:
        series.iloc[20:240] = np.nan
    return series


@pytest.fixture
def _planted_short_history(monkeypatch):
    register_feature(
        FeatureDefinition(
            id="test.short_gap",
            description="planted: blank for the short-history entity",
            fn=_short_gap,
            temporal_support=TemporalSupport.PIT_SAFE,
            scope=FeatureScope.ENTITY,
            lookback=0,
        ),
        overwrite=True,
    )

    def _fetch(symbol):
        return make_ohlcv(symbol, n=260 if symbol == "CCC" else 500)

    monkeypatch.setattr(
        DataFactory, "get_provider", lambda *a, **kw: make_provider_mock(_fetch)
    )
    yield
    FEATURE_REGISTRY.pop("test.short_gap", None)


class TestWhichNameLostTheRows:
    """
    Pins 1A.6. `attribute_drops` computes `per_entity_rows_dropped` and
    `explain_dataset_row_loss` read five of its six keys. "Which NAME lost
    the rows" is a different question from "which feature", with a
    different remedy: no feature can be dropped to recover a short
    history.
    """

    def _result(self):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(
                spec=_spec(
                    features=[
                        FeatureSpec(id="technical.rsi"),
                        FeatureSpec(id="test.short_gap"),
                    ],
                    missing=MissingDataSpec(policy="keep"),
                )
            )
        ).dataset_id
        return explain_dataset_row_loss(ExplainRowLossInput(dataset_id=dataset_id))

    def test_the_per_entity_counts_sum_to_the_rows_lost(self, _planted_short_history):
        result = self._result()
        assert result.rows_lost > 0
        assert sum(result.per_entity_rows_dropped.values()) == result.rows_lost

    def test_it_names_the_short_history_entity(self, _planted_short_history):
        result = self._result()
        per_entity = result.per_entity_rows_dropped
        assert set(per_entity) == set(UNIVERSE)
        worst = max(per_entity, key=per_entity.__getitem__)
        assert worst == "CCC"
        assert per_entity["CCC"] > per_entity["AAA"] + per_entity["BBB"]

    def test_one_entity_holding_over_half_the_loss_is_a_warning(
        self, _planted_short_history
    ):
        result = self._result()
        assert result.per_entity_rows_dropped["CCC"] > result.rows_lost / 2
        assert any("'CCC' alone accounts for" in w for w in result.warnings)

    def test_an_even_universe_does_not_trip_the_warning(self, patched_multi_factory):
        """The null case: three entities with the same history lose the
        same warm-up each, so no name accounts for more than half."""
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec(missing=MissingDataSpec(policy="keep")))
        ).dataset_id
        result = explain_dataset_row_loss(ExplainRowLossInput(dataset_id=dataset_id))
        assert sum(result.per_entity_rows_dropped.values()) == result.rows_lost
        assert max(result.per_entity_rows_dropped.values()) <= result.rows_lost / 2
        assert not any("alone accounts for" in w for w in result.warnings)


# ── 1A.7 ────────────────────────────────────────────────────────────────


class TestMonitorReturnsItsReference:
    """
    Pins 1A.7. `monitor_model` loaded the training profile, bound it to a
    name and never used it again, so every PSI it reported was a distance
    from something the caller could not see.
    """

    def test_the_profile_comes_back_with_the_drift_it_explains(
        self, patched_multi_factory
    ):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec())
        ).dataset_id
        model_id = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        ).model_id
        manifest = load_manifest(model_id)
        scored = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        report = monitor_model(
            MonitorModelInput(
                model_id=model_id, predictions_uri=scored["predictions_uri"]
            )
        )
        profile = report.training_profile
        assert profile["bins"] == PROFILE_BINS
        assert set(profile["features"]) == set(manifest.feature_ids)
        for entry in profile["features"].values():
            assert len(entry["quantile_edges"]) == PROFILE_BINS + 1
            assert entry["quantile_edges"] == sorted(entry["quantile_edges"])
            assert entry["n"] > 0
            assert 0.0 <= entry["missing_rate"] <= 1.0
        # The reference is the thing the drift rows were read against, so
        # they describe the same features.
        assert {r.feature for r in report.feature_drift} == set(profile["features"])


# ── 1A.8 ────────────────────────────────────────────────────────────────


class TestLagColumnsAreLabelled:
    """
    Pins 1A.8. `parse_lag_column` had no caller anywhere outside its own
    module, while its docstring claimed it was "what lets
    analyze_model_errors and the importance summary report 'this is rsi at
    lag 3'". A spec with three lags produced importance rows of opaque
    strings and a refusal that listed them raw.
    """

    @pytest.fixture
    def lagged_model_id(self, patched_multi_factory) -> str:
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(
                spec=_spec(features=[FeatureSpec(id="technical.rsi", lags=[1, 2, 3])])
            )
        ).dataset_id
        return run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        ).model_id

    def test_every_lagged_column_is_named_with_its_depth(self, lagged_model_id):
        data = inspect_model(
            InspectModelInput(model_id=lagged_model_id, view="feature_importance")
        ).data
        labels = data["feature_labels"]
        assert labels == {
            f"technical.rsi__lag{lag}": {"feature": "technical.rsi", "lag": lag}
            for lag in (1, 2, 3)
        }
        # The unlagged base column is in the importance summary and NOT in
        # the labels: the map says which columns ARE lagged.
        assert "technical.rsi" in data["feature_importance_summary"]
        assert "technical.rsi" not in labels
        assert set(labels) <= set(data["feature_importance_summary"])

    def test_the_error_refusal_uses_the_same_label(self, lagged_model_id):
        with pytest.raises(ValidationError, match=r"technical\.rsi at lag 3"):
            analyze_model_errors(
                AnalyzeModelErrorsInput(model_id=lagged_model_id, feature="nope")
            )

    def test_a_spec_with_no_lags_labels_nothing(self, patched_multi_factory):
        """The null case: an empty map, not a map of Nones."""
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec())
        ).dataset_id
        model_id = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        ).model_id
        data = inspect_model(
            InspectModelInput(model_id=model_id, view="feature_importance")
        ).data
        assert data["feature_importance_summary"]
        assert data["feature_labels"] == {}


# ── 1A.9 ────────────────────────────────────────────────────────────────


class TestScorePredictionsFollowsTheConvention:
    """
    Pins 1A.9. Every other result on this surface carries `warnings`;
    `ScorePredictionsResult` carried only `notes`, so the two caveats that
    say a headline number is not what it looks like -- an oracle baseline,
    a turnover that has to be paid -- arrived in a field read as
    commentary. `notes` is kept; existing callers read it.
    """

    def test_the_result_has_both_fields(self):
        fields = ScorePredictionsResult.model_fields
        assert "warnings" in fields and "notes" in fields
        assert fields["warnings"].annotation == fields["notes"].annotation

    def test_a_clean_run_warns_about_nothing(self):
        """The null case: an honest baseline and an ordering that never
        moves leaves the warnings empty while the notes still explain."""
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=_publish(_panel(), "convention_clean"),
                task="regression",
                train_mean=0.0,
            )
        )
        assert result.warnings == []
        assert result.notes


class TestTheExperimentResultKeepsTheEnginesWarnings:
    """
    Pins 1A.9's sibling: `engine.run_experiment` returns a `warnings` key
    from both of its return paths -- the calibration caveat of D-2 travels
    there -- and `RunModelExperimentResult` had no such field and no
    `extra="forbid"`, so pydantic dropped it between the engine and the
    agent without a word. The field is what carries a run's own caveats,
    as distinct from the dataset's, which travel on the manifest as
    `dataset_warnings`.
    """

    def test_the_field_exists_and_a_plain_run_fills_it_with_nothing(
        self, patched_multi_factory
    ):
        assert "warnings" in RunModelExperimentResult.model_fields
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec())
        ).dataset_id
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_ridge())
        )
        # An uncalibrated ridge has nothing to caveat, so the null case is
        # an empty list -- the point is that it ARRIVES, not that it is
        # full. The engine's key is the same object either way.
        assert result.warnings == []
        assert isinstance(result.warnings, list)
