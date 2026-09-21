"""
Three things the CHANGELOG entry of 2026-09-21 changed: the portfolio
simulator reached by reference as well as by model id, ONE
predictions -> score_panel reshape behind both of its callers, and the
conformal band a scored model emits reported beside the point prediction.

What each group plants:

The simulator on a reference. The same predictions reached two ways --
through a registered model and through a published reference -- must
produce the same weights. That is the whole claim: `evaluate_model_
portfolio` keeps its manifest head and both doors open onto one
simulation, so an ensemble is tradeable rather than only describable. The
refusals the manifest used to supply (cpcv, an interval-less frame asked
to size by uncertainty, a price window nobody named) are planted beside
it, because a door that opens onto a wrong answer is worse than one that
is shut.

One reshape. `meta/convert.py` had a second implementation that checked
three column names and let a duplicate (entity, date) pair overwrite
itself -- a smaller but perfectly valid-looking panel. Both entry points
now agree cell for cell, including on the caller-settable classification
threshold, and the frames the old one accepted silently are refused with
the simulator's messages.

The interval. `summary_stats` describes the point prediction; the band
beside it was returned by no view. The planted case is a conformal model
whose radius is known from its own manifest: mean width must be exactly
twice it. The null case is a point-only model, which gets an empty dict
rather than a dict of zeros -- an absent interval is not a zero-width one.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import ConvertReferenceInput
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.meta.tools import convert_reference
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent import (
    BuildEnsembleInput,
    BuildModelDatasetInput,
    RunModelExperimentInput,
    build_model_dataset,
    build_model_ensemble,
    run_model_experiment,
)
from standard_quant_tools.modeling.agent.portfolio_models import (
    EvaluatePredictionsPortfolioInput,
)
from standard_quant_tools.modeling.agent.portfolio_tools import (
    evaluate_predictions_portfolio,
)
from standard_quant_tools.modeling.portfolio_eval import (
    evaluate_model_portfolio,
    predictions_to_score_panel,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.scoring import _interval_statistics, score_model
from standard_quant_tools.modeling.specs import (
    ConformalSpec,
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PortfolioSimSpec,
    PredictionTransformSpec,
    TargetSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


# ── Fixtures ────────────────────────────────────────────────────────────


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[
            FeatureSpec(id="market.momentum", params={"lookback": 20}, alias="mom_20"),
            FeatureSpec(id="technical.rsi", params={"period": 14}),
        ],
        target=TargetSpec(type="forward_return", horizon=5),
    )


def _ridge(seed: int, alpha: float = 1.0) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": alpha}),
        validation=ValidationSpec(
            train_window=150, test_window=40, embargo=5, min_folds=2
        ),
        random_seed=seed,
    )


@pytest.fixture
def trained(patched_multi_factory):
    """A dataset and two models on it, through the agent tools -- only that
    path persists the dataset_spec.json the five price fields come from."""
    dataset = build_model_dataset(BuildModelDatasetInput(spec=_dataset()))
    model_ids = [
        run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset.dataset_id, spec=_ridge(seed, alpha)
            )
        ).model_id
        for seed, alpha in ((1, 1.0), (2, 0.25))
    ]
    return {"dataset_id": dataset.dataset_id, "model_ids": model_ids}


def _publish_oos(model_id: str, name: str, producer: str = "test_publisher") -> str:
    """The model's registered OOS predictions, republished as a reference."""
    frame = _artifacts.load_artifact(str(load_manifest(model_id).oos_predictions_uri))
    return handoff.publish(
        frame, "predictions", "refeval", name, producer=producer, overwrite=True
    )


def _probabilities(n_dates: int = 6, n_entities: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        [
            {"date": date, "entity": entity, "prediction": float(rng.uniform(0.2, 0.8))}
            for date in pd.bdate_range("2024-01-02", periods=n_dates)
            for entity in [f"E{i:02d}" for i in range(n_entities)]
        ]
    )


def _convert(ref: str, name: str, **kwargs):
    return convert_reference(
        ConvertReferenceInput(
            ref=ref, to_kind="score_panel", run_id="refeval_conv", name=name, **kwargs
        )
    )


# ── The simulator on a reference ────────────────────────────────────────


class TestTheSimulatorTakesAReference:
    def test_an_ensemble_ref_and_a_dataset_id_are_enough_to_trade_it(self, trained):
        """`build_model_ensemble` publishes a reference that, before this,
        could be correlated and described and never traded: passing it to
        the simulator failed at model-id validation."""
        ensemble = build_model_ensemble(
            BuildEnsembleInput(
                model_ids=trained["model_ids"], run_id="refeval_ens", name="combined"
            )
        )
        result = evaluate_predictions_portfolio(
            EvaluatePredictionsPortfolioInput(
                predictions_ref=ensemble.ref,
                task="regression",
                dataset_id=trained["dataset_id"],
                transform=PredictionTransformSpec(
                    method="cross_sectional_rank",
                    max_position_weight=0.5,
                    rebalance_frequency="weekly",
                ),
                run_id="refeval_eval",
            )
        )
        assert result.source_ref == ensemble.ref
        assert np.isfinite(result.metrics["sharpe_ratio"])
        assert result.coverage["n_entities"] == len(UNIVERSE)
        assert result.coverage["n_rebalance_dates"] >= 2

        weights = _artifacts.load_artifact(result.target_weights_uri)
        assert not weights.empty
        assert set(weights.columns) <= set(UNIVERSE)
        assert np.allclose(weights.abs().sum(axis=1), 1.0)

    def test_both_doors_open_onto_the_same_weights(self, trained):
        """The claim of the refactor. One arithmetic, two entry points: a
        registered model and a reference carrying the same predictions
        must produce the same target weights, hash for hash."""
        model_id = trained["model_ids"][0]
        by_model = evaluate_model_portfolio(model_id)
        by_reference = evaluate_predictions_portfolio(
            EvaluatePredictionsPortfolioInput(
                predictions_ref=_publish_oos(model_id, "oos_same"),
                task="regression",
                dataset_id=trained["dataset_id"],
                run_id="refeval_same",
            )
        )
        assert (
            by_reference.provenance["target_weights_hash"]
            == by_model["provenance"]["target_weights_hash"]
        )
        pd.testing.assert_frame_equal(
            _artifacts.load_artifact(by_reference.target_weights_uri),
            _artifacts.load_artifact(by_model["target_weights_uri"]),
        )
        assert by_reference.metrics["sharpe_ratio"] == pytest.approx(
            by_model["metrics"]["sharpe_ratio"], rel=1e-12
        )

    def test_a_cpcv_shaped_reference_is_refused_by_name(self, trained):
        """A `path` column means one prediction per (date, entity, PATH):
        alternative histories, not a sequence. The model path refuses cpcv
        off the manifest; a bare frame has only its shape to give it
        away."""
        frame = _artifacts.load_artifact(
            str(load_manifest(trained["model_ids"][0]).oos_predictions_uri)
        )
        doubled = pd.concat([frame.assign(path=0), frame.assign(path=1)])
        ref = handoff.publish(
            doubled, "predictions", "refeval", "cpcv_shaped", producer="test"
        )
        with pytest.raises(ValidationError, match="combinatorial purged"):
            evaluate_predictions_portfolio(
                EvaluatePredictionsPortfolioInput(
                    predictions_ref=ref,
                    task="regression",
                    dataset_id=trained["dataset_id"],
                    run_id="refeval_cpcv",
                )
            )

    def test_uncertainty_scaling_without_an_interval_reproduces_the_refusal(
        self, trained
    ):
        """`scale_by_uncertainty` reads the FRAME's lower/upper columns,
        not manifest.distribution -- so the refusal a point-only model
        gets is the one a point-only reference gets, word for word."""
        ref = _publish_oos(trained["model_ids"][0], "oos_point")
        with pytest.raises(ValidationError, match="ModelSpec.intervals"):
            evaluate_predictions_portfolio(
                EvaluatePredictionsPortfolioInput(
                    predictions_ref=ref,
                    task="regression",
                    dataset_id=trained["dataset_id"],
                    transform=PredictionTransformSpec(method="uncertainty_scaled"),
                    run_id="refeval_unc",
                )
            )

    def test_provenance_names_the_reference_and_its_producer(self, trained):
        """There is no registered digest behind a reference, so what the
        provenance CAN say is which reference was read and who published
        it -- and it has to say that much, or the track record names
        nothing at all."""
        ref = _publish_oos(
            trained["model_ids"][0], "oos_prov", producer="refeval_publisher"
        )
        result = evaluate_predictions_portfolio(
            EvaluatePredictionsPortfolioInput(
                predictions_ref=ref,
                task="regression",
                dataset_id=trained["dataset_id"],
                run_id="refeval_prov",
            )
        )
        provenance = result.provenance
        assert provenance["source_ref"] == ref
        assert provenance["producer"] == "refeval_publisher"
        assert provenance["source_content_hash"]
        assert provenance["dataset_id"] == trained["dataset_id"]
        assert provenance["task"] == "regression"
        assert provenance["target_weights_hash"]
        # The claim it must NOT make: nothing here was checked against a
        # hash recorded at registration time.
        assert "oos_predictions_hash" not in provenance
        assert not any("no producer" in w for w in result.warnings)

    def test_a_reference_without_a_producer_says_so(self, trained):
        ref = handoff.publish(
            _artifacts.load_artifact(
                str(load_manifest(trained["model_ids"][0]).oos_predictions_uri)
            ),
            "predictions",
            "refeval",
            "oos_anonymous",
        )
        result = evaluate_predictions_portfolio(
            EvaluatePredictionsPortfolioInput(
                predictions_ref=ref,
                task="regression",
                dataset_id=trained["dataset_id"],
                run_id="refeval_anon",
            )
        )
        assert result.provenance["producer"] is None
        assert any("records no producer" in w for w in result.warnings)

    def test_no_dataset_and_no_price_fields_is_refused_by_name(self, trained):
        """The wrong provider or window simulates perfectly cleanly against
        the wrong prices, so neither is a default this tool can pick."""
        ref = _publish_oos(trained["model_ids"][0], "oos_nofields")
        with pytest.raises(ValidationError) as excinfo:
            evaluate_predictions_portfolio(
                EvaluatePredictionsPortfolioInput(
                    predictions_ref=ref, task="regression", run_id="refeval_nofields"
                )
            )
        message = str(excinfo.value)
        assert "dataset_id" in message
        for field in ("interval", "provider", "start_date", "end_date"):
            assert field in message

    def test_the_five_fields_given_explicitly_are_enough(self, trained):
        """A caller with no dataset_id -- an externally computed alpha --
        names the price window itself and gets the same simulation."""
        ref = _publish_oos(trained["model_ids"][0], "oos_explicit")
        result = evaluate_predictions_portfolio(
            EvaluatePredictionsPortfolioInput(
                predictions_ref=ref,
                task="regression",
                interval="1d",
                provider="mock",
                start_date="2022-01-01",
                end_date="2023-12-31",
                run_id="refeval_explicit",
            )
        )
        assert np.isfinite(result.metrics["sharpe_ratio"])
        assert result.provenance["interval"] == "1d"
        assert result.provenance["provider"] == "mock"

    def test_a_raw_artifact_path_is_not_a_reference(self, trained):
        uri = str(load_manifest(trained["model_ids"][0]).oos_predictions_uri)
        with pytest.raises(ValidationError, match="raw artifact path"):
            evaluate_predictions_portfolio(
                EvaluatePredictionsPortfolioInput(
                    predictions_ref=uri,
                    task="regression",
                    dataset_id=trained["dataset_id"],
                    run_id="refeval_path",
                )
            )


# ── One predictions -> score_panel ──────────────────────────────────────


class TestOneScorePanelImplementation:
    def test_the_threshold_agrees_cell_for_cell(self):
        """`convert_reference` recentred by a caller-settable threshold and
        the simulator's reshape by a hard-coded 0.5. One function now, and
        the parameter reaches it."""
        frame = _probabilities()
        ref = handoff.publish(frame, "predictions", "refeval", "proba")
        converted = _convert(ref, "sp_06", task="classification", proba_threshold=0.6)
        panel = handoff.resolve(converted.ref, expect="score_panel")
        expected = predictions_to_score_panel(
            frame, "classification", proba_threshold=0.6
        )
        assert set(panel) == set(expected.columns)
        for entity, per_date in panel.items():
            for date, value in per_date.items():
                assert value == pytest.approx(
                    float(expected.loc[pd.Timestamp(date), entity])
                )
        assert any("0.6" in note for note in converted.notes)

    def test_a_duplicate_pair_is_refused_instead_of_collapsing(self):
        """It used to overwrite itself in a dict -- a panel that is smaller
        than the frame and looks entirely valid."""
        frame = _probabilities(n_dates=3, n_entities=2)
        ref = handoff.publish(
            pd.concat([frame, frame.head(1)]), "predictions", "refeval", "dupes"
        )
        with pytest.raises(ValidationError, match="duplicate"):
            _convert(ref, "sp_dupes", task="classification")

    def test_a_non_finite_prediction_is_refused(self):
        frame = _probabilities(n_dates=3, n_entities=2)
        frame.loc[0, "prediction"] = np.nan
        ref = handoff.publish(frame, "predictions", "refeval", "nonfinite")
        with pytest.raises(ValidationError, match="non-finite"):
            _convert(ref, "sp_nonfinite")

    def test_an_empty_frame_is_refused(self):
        """Two layers deep: the handoff store will not publish an empty
        frame at all, and the reshape behind the conversion refuses one on
        its own -- so a frame that lost its rows between publication and
        conversion cannot become a panel of nothing either."""
        empty = _probabilities().iloc[:0]
        with pytest.raises(ValidationError, match="cannot save an empty artifact"):
            handoff.publish(empty, "predictions", "refeval", "empty")
        with pytest.raises(ValidationError, match="no rows"):
            predictions_to_score_panel(empty, "classification")

    def test_the_notes_are_the_ones_the_fleet_already_asserts(self):
        frame = _probabilities()
        ref = handoff.publish(frame, "predictions", "refeval", "notes")
        classified = _convert(ref, "sp_notes_c", task="classification")
        assert any("recentred" in note for note in classified.notes)
        untasked = _convert(ref, "sp_notes_n")
        assert any(
            "task" in note and "classification" in note for note in untasked.notes
        )
        values = [
            value
            for per_date in handoff.resolve(untasked.ref, expect="score_panel").values()
            for value in per_date.values()
        ]
        assert min(values) == pytest.approx(float(frame["prediction"].min()))

    def test_the_default_threshold_reproduces_every_planted_panel(self):
        """The numbers `tests/modeling/test_portfolio_eval.py` and
        `tests/modeling/test_bridge.py` pin, through both entry points."""
        long_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "entity": ["AAA", "BBB"],
                "prediction": [0.7, 0.3],
            }
        )
        panel = predictions_to_score_panel(long_df, "classification")
        assert panel.loc[pd.Timestamp("2024-01-01"), "AAA"] == pytest.approx(0.2)
        assert panel.loc[pd.Timestamp("2024-01-01"), "BBB"] == pytest.approx(-0.2)

        ref = handoff.publish(long_df, "predictions", "refeval", "default")
        converted = handoff.resolve(
            _convert(ref, "sp_default", task="classification").ref, expect="score_panel"
        )
        assert converted["AAA"]["2024-01-01 00:00:00"] == pytest.approx(0.2)
        assert converted["BBB"]["2024-01-01 00:00:00"] == pytest.approx(-0.2)

        # A ranker's score is already centred on the cross-section; only a
        # probability gets the shift.
        signed = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "entity": ["AAA", "BBB"],
                "prediction": [0.4, -0.3],
            }
        )
        ranked = predictions_to_score_panel(signed, task="ranking")
        assert ranked.equals(predictions_to_score_panel(signed, task="regression"))
        assert ranked.loc[pd.Timestamp("2024-01-02"), "AAA"] == 0.4

    def test_a_task_of_none_passes_through_and_a_bad_one_is_refused(self):
        frame = _probabilities(n_dates=2, n_entities=2)
        passthrough = predictions_to_score_panel(frame, None)
        assert passthrough.to_numpy().min() == pytest.approx(
            float(frame["prediction"].min())
        )
        with pytest.raises(ValidationError, match="task must be one of"):
            predictions_to_score_panel(frame, "banana")


# ── The interval a scored model emits ───────────────────────────────────


class TestTheScoredIntervalIsVisible:
    def test_the_mean_width_is_twice_the_conformal_radius(self, patched_multi_factory):
        """The planted case: the radius is on the model's own manifest, and
        lower/upper are written as prediction -/+ it, so the mean width is
        exactly 2r and nothing else."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_interval",
            model_spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                intervals=ConformalSpec(alpha=0.1),
                random_seed=1,
            ),
        )
        result = score_model(
            model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        radius = float(load_manifest(model_id).distribution["conformal"]["radius"])
        stats = result["interval_stats"]
        assert stats["interval_mean_width"] == pytest.approx(2 * radius)
        assert stats["interval_median_width"] == pytest.approx(2 * radius)
        assert stats["interval_min_width"] == pytest.approx(2 * radius)
        assert stats["interval_max_width"] == pytest.approx(2 * radius)
        assert stats["n_intervals"] == result["n_entities"]
        assert all(np.isfinite(value) for value in stats.values())

        # The ratio, and the warning that must fire exactly with it.
        spread = result["summary_stats"]["max"] - result["summary_stats"]["min"]
        ratio = stats["interval_width_over_prediction_spread"]
        assert ratio == pytest.approx(stats["interval_mean_width"] / spread)
        fired = any(
            "wider than the entire cross-section" in w for w in result["warnings"]
        )
        assert fired == (ratio > 1.0)

    def test_a_point_only_model_gets_an_empty_dict_and_its_old_summary(
        self, patched_multi_factory
    ):
        """An absent interval is not a zero-width one, and summary_stats is
        byte-identical to what it was before this field existed."""
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_point_only")
        result = score_model(model_id, as_of="2023-12-29", universe=["AAA", "BBB"])
        assert result["interval_stats"] == {}

        frame = _artifacts.load_artifact(result["predictions_uri"])
        assert list(frame.columns) == ["entity", "date", "prediction"]
        assert result["summary_stats"] == {
            "mean": float(frame["prediction"].mean()),
            "std": float(frame["prediction"].std()),
            "min": float(frame["prediction"].min()),
            "max": float(frame["prediction"].max()),
        }
        assert not any(
            "wider than the entire cross-section" in w for w in result["warnings"]
        )

    def test_the_implausible_width_warning_fires_only_when_it_should(self):
        """The case this was written for: a mean width of 0.169 against a
        prediction range of 0.0024 -- a band roughly seventy times the
        entire cross-section's spread."""
        implausible = pd.DataFrame(
            {
                "prediction": [0.0034, 0.0046, 0.0058],
                "lower": [0.0034 - 0.0845, 0.0046 - 0.0845, 0.0058 - 0.0845],
                "upper": [0.0034 + 0.0845, 0.0046 + 0.0845, 0.0058 + 0.0845],
            }
        )
        stats, warnings = _interval_statistics(implausible)
        assert stats["interval_mean_width"] == pytest.approx(0.169)
        assert stats["interval_width_over_prediction_spread"] > 1.0
        assert len(warnings) == 1
        assert "wider than the entire cross-section" in warnings[0]
        assert "0.169" in warnings[0]

        tight = pd.DataFrame(
            {
                "prediction": [-0.02, 0.0, 0.02],
                "lower": [-0.025, -0.005, 0.015],
                "upper": [-0.015, 0.005, 0.025],
            }
        )
        stats, warnings = _interval_statistics(tight)
        assert stats["interval_mean_width"] == pytest.approx(0.01)
        assert stats["interval_width_over_prediction_spread"] == pytest.approx(0.25)
        assert warnings == []

    def test_a_degenerate_cross_section_omits_the_ratio_rather_than_dividing(self):
        """Every name predicted the same number: the spread is zero, and
        'the ratio is undefined' is a different claim from 'the ratio is
        enormous'. No key, no infinity."""
        flat = pd.DataFrame(
            {"prediction": [0.01, 0.01], "lower": [0.0, 0.0], "upper": [0.02, 0.02]}
        )
        stats, warnings = _interval_statistics(flat)
        assert "interval_width_over_prediction_spread" not in stats
        assert stats["n_intervals"] == 2
        assert warnings == []

    def test_non_finite_widths_are_dropped_not_emitted(self):
        holed = pd.DataFrame(
            {
                "prediction": [0.01, 0.02, 0.03],
                "lower": [0.0, np.nan, 0.02],
                "upper": [0.02, 0.04, 0.04],
            }
        )
        stats, _ = _interval_statistics(holed)
        assert stats["n_intervals"] == 2
        assert all(np.isfinite(value) for value in stats.values())

        assert _interval_statistics(
            pd.DataFrame({"prediction": [0.01], "lower": [np.nan], "upper": [np.nan]})
        ) == ({}, [])
        assert _interval_statistics(pd.DataFrame({"prediction": [0.01]})) == ({}, [])
