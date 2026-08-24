"""
Learning-to-rank estimators.

These exist to close a mismatch: the pipeline judges a cross-sectional model
on rank IC — did it order the names correctly today — while every other
estimator in the registry optimizes squared error or log loss.

Most of the risk is not in the objective, it is in the three things that have
to be true before either library will train correctly, only one of which
raises when you get it wrong:

  1. the LABEL must be integer relevance grades (this one raises)
  2. the ROWS must be ordered by query group (this one does NOT — both
     libraries assume it and never check)
  3. the GROUP counts must match that ordering (also silent)

So most of this file is about 2 and 3.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators import boosting
from standard_quant_tools.modeling.estimators.registry import (
    ESTIMATOR_REGISTRY,
    validate_params,
)
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    RankingSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.ranking import (
    group_sizes,
    ndcg_at_k,
    ranking_metrics,
    relevance_grades,
)

RANKERS = [name for _, name in ESTIMATOR_REGISTRY if _ == "ranking"] or []
requires_ranker = pytest.mark.skipif(
    not RANKERS, reason="neither lightgbm nor xgboost is installed"
)


class TestRelevanceGrades:
    def test_grades_are_integers_in_range(self):
        rng = np.random.default_rng(0)
        dates = np.repeat(pd.date_range("2020-01-01", periods=10), 20)
        grades = relevance_grades(rng.normal(0, 1, 200), dates, n_grades=8)
        assert grades.dtype.kind == "i"
        assert grades.min() >= 0 and grades.max() <= 7

    def test_grading_is_monotone_within_each_date(self):
        """The grade must never contradict the target's own ordering inside a
        date — that ordering is the entire thing the ranker is learning."""
        rng = np.random.default_rng(1)
        dates = np.repeat(pd.date_range("2020-01-01", periods=6), 25)
        target = rng.normal(0, 0.02, 150)
        grades = relevance_grades(target, dates, n_grades=5)
        frame = pd.DataFrame({"date": dates, "t": target, "g": grades})
        for _, group in frame.groupby("date"):
            ordered = group.sort_values("t")["g"].to_numpy()
            assert np.all(np.diff(ordered) >= 0)

    def test_grading_is_per_date_not_pooled(self):
        """
        A grade pooled across dates would be asking the ranker to rank
        today's names against last year's, which is not what a query group
        is. Two dates with wildly different target levels must both span the
        full grade range.
        """
        dates = np.repeat(pd.date_range("2020-01-01", periods=2), 16)
        target = np.r_[np.linspace(-1.0, -0.9, 16), np.linspace(5.0, 6.0, 16)]
        grades = relevance_grades(target, dates, n_grades=4)
        for half in (grades[:16], grades[16:]):
            assert set(half.tolist()) == {0, 1, 2, 3}

    def test_a_thin_cross_section_does_not_fake_more_levels(self):
        """Three entities cannot fill eight buckets. Grading them across
        three levels is honest; spreading them to 0 and 7 would invent an
        ordering gap that is not in the data."""
        dates = np.repeat(pd.date_range("2020-01-01", periods=4), 3)
        grades = relevance_grades(np.arange(12, dtype=float), dates, n_grades=8)
        assert set(grades.tolist()) == {0, 1, 2}

    def test_ties_do_not_invent_an_ordering(self):
        """Equal targets must not be split into different grades wherever the
        bucket boundary allows them to share one."""
        dates = np.repeat(pd.date_range("2020-01-01", periods=1), 8)
        target = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        grades = relevance_grades(target, dates, n_grades=2)
        assert len(set(grades[:4].tolist())) == 1
        assert len(set(grades[4:].tolist())) == 1
        assert grades[0] < grades[4]

    def test_rejects_too_few_grades(self):
        with pytest.raises(ValidationError, match="n_grades"):
            relevance_grades(np.arange(4.0), np.repeat("d", 4), n_grades=1)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationError, match="same length"):
            relevance_grades(np.arange(4.0), np.repeat("d", 3))

    def test_empty_input(self):
        assert relevance_grades(np.array([]), np.array([])).size == 0


class TestGroupSizes:
    def test_counts_consecutive_runs(self):
        assert list(group_sizes(np.array(["a", "a", "b", "b", "b"]))) == [2, 3]

    def test_rejects_unsorted_rows(self):
        """
        The important one. Both libraries take `group` as consecutive counts
        and never verify the rows are actually ordered that way — unsorted
        input trains a model on the wrong groupings and reports nothing.
        """
        with pytest.raises(ValidationError, match="not sorted by date"):
            group_sizes(np.array(["a", "b", "a", "b"]))

    def test_sums_to_the_row_count(self):
        dates = np.repeat(pd.date_range("2020-01-01", periods=7), 11)
        assert group_sizes(dates).sum() == len(dates)

    def test_empty(self):
        assert group_sizes(np.array([])).size == 0


class TestNDCG:
    def test_perfect_ranking_scores_one(self):
        dates = np.repeat("d", 10)
        grades = np.arange(10)
        assert ndcg_at_k(grades.astype(float), grades, dates, 5) == pytest.approx(1.0)

    def test_reversed_ranking_scores_near_zero(self):
        dates = np.repeat("d", 10)
        grades = np.arange(10)
        assert ndcg_at_k(-grades.astype(float), grades, dates, 5) < 0.1

    def test_a_date_with_no_ordering_to_find_is_skipped(self):
        """
        Every name at grade 0 means no ordering beats any other. Scoring that
        0.0 would be indistinguishable from ranking it backwards, so it is
        skipped — and a panel of nothing but such dates is NaN, not zero.
        """
        dates = np.repeat("d", 10)
        assert np.isnan(ndcg_at_k(np.arange(10.0), np.zeros(10), dates, 5))

    def test_rejects_a_nonsense_cutoff(self):
        with pytest.raises(ValidationError, match="k must be"):
            ndcg_at_k(np.arange(4.0), np.arange(4), np.repeat("d", 4), 0)


class TestRankingMetrics:
    def test_reports_ic_and_ndcg_but_not_r2(self):
        """
        A ranker's score is an ordering on an arbitrary scale — LambdaRank is
        invariant to any monotone transform of it — so R2 and MAE against a
        return would be measuring a scale the quantity does not have.
        Reporting them would invite exactly the comparison they cannot
        support.
        """
        rng = np.random.default_rng(2)
        dates = np.repeat(pd.date_range("2020-01-01", periods=20), 15)
        y = rng.normal(0, 0.02, 300)
        scores = 0.4 * y + rng.normal(0, 0.02, 300)
        metrics = ranking_metrics(y, scores, dates)
        assert "cs_rank_ic_mean" in metrics and "ndcg_at_5" in metrics
        assert "r2" not in metrics and "mae" not in metrics

    def test_is_invariant_to_a_monotone_rescale_of_the_score(self):
        """The property that makes r2 meaningless is the same one that makes
        the reported metrics trustworthy — so it gets asserted."""
        rng = np.random.default_rng(3)
        dates = np.repeat(pd.date_range("2020-01-01", periods=20), 15)
        y = rng.normal(0, 0.02, 300)
        scores = 0.4 * y + rng.normal(0, 0.02, 300)
        base = ranking_metrics(y, scores, dates)
        rescaled = ranking_metrics(y, scores * 1000.0 + 7.0, dates)
        assert rescaled["cs_rank_ic_mean"] == pytest.approx(
            base["cs_rank_ic_mean"], abs=1e-12
        )
        assert rescaled["ndcg_at_5"] == pytest.approx(base["ndcg_at_5"], abs=1e-12)


class TestRankingSpec:
    def test_defaults(self):
        spec = RankingSpec()
        assert spec.n_grades == 8 and spec.ndcg_at == [5, 10]

    def test_n_grades_is_capped_at_lightgbms_own_limit(self):
        """
        31, not a round number, and not a preference. LightGBM's default
        label_gain table holds 31 entries (2^i - 1 for i in 0..30); a 32nd
        grade fails at fit time with 'Label 31 is not less than the number of
        label mappings'. Found by sweeping n_grades, which is exactly the
        kind of thing that would otherwise surface as a crash several folds
        into a user's run.
        """
        assert RankingSpec(n_grades=31).n_grades == 31
        with pytest.raises(ValueError):
            RankingSpec(n_grades=32)

    def test_rejects_an_empty_cutoff_list(self):
        with pytest.raises(ValueError, match="at least one cut-off"):
            RankingSpec(ndcg_at=[])

    def test_rejects_a_nonpositive_cutoff(self):
        with pytest.raises(ValueError, match=">= 1"):
            RankingSpec(ndcg_at=[5, 0])


@requires_ranker
class TestRankingEstimators:
    def test_registered_under_their_own_task(self):
        """
        Not as regression estimators, and the difference is not cosmetic: a
        ranker needs query groups at fit time, needs its target graded first,
        and emits a score no regression metric can be computed against.
        Filing them under 'regression' would let a caller reach them through
        a path that supplies none of that.
        """
        assert RANKERS
        for name in RANKERS:
            assert ("regression", name) not in ESTIMATOR_REGISTRY
            assert ("classification", name) not in ESTIMATOR_REGISTRY

    def test_param_ceilings_still_apply(self):
        name = RANKERS[0]
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_params("ranking", name, {"n_estimators": 10_000_000})
        with pytest.raises(ValidationError, match="does not accept"):
            validate_params("ranking", name, {"not_a_param": 1})


def _ranking_dataset(n_dates=320, n_entities=20, seed=0):
    """A panel whose cross-sectional ORDERING is learnable."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        f1 = rng.normal(0, 1, n_entities)
        f2 = rng.normal(0, 1, n_entities)
        latent = 0.8 * f1 - 0.4 * f2
        target = latent * 0.01 + rng.normal(0, 0.01, n_entities)
        for entity in range(n_entities):
            rows.append(
                (date, f"E{entity:03d}", f1[entity], f2[entity], target[entity])
            )
    panel = pd.DataFrame(rows, columns=["date", "entity", "f1", "f2", "target"])
    from standard_quant_tools.audit.hashing import hash_dataframe

    return {
        "panel": panel,
        "feature_ids": ["f1", "f2"],
        "target_id": "forward_return:5",
        "data_hash": hash_dataframe(panel),
        "spec_hash": "test",
        "entities": sorted(panel["entity"].unique()),
        "warnings": [],
        "drop_attribution": {},
    }


def _ranking_spec(estimator, **kwargs):
    return ModelSpec(
        task="ranking",
        estimator=EstimatorSpec(type=estimator, params={"n_estimators": 30}),
        validation=ValidationSpec(train_window=150, test_window=75, embargo=2),
        random_seed=1,
        **kwargs,
    )


@requires_ranker
class TestRankingEndToEnd:
    @pytest.mark.parametrize("name", RANKERS)
    def test_runs_and_learns_the_ordering(self, name):
        from standard_quant_tools.modeling.engine import run_experiment

        result = run_experiment(_ranking_dataset(), _ranking_spec(name), "ds")
        assert result["n_folds"] > 0
        metrics = result["oos_metrics"]
        # The ordering is genuinely in the data, so a working ranker finds it.
        assert metrics["cs_rank_ic_mean"] > 0.05, metrics["cs_rank_ic_mean"]
        assert 0.0 <= metrics["ndcg_at_5"] <= 1.0

    @pytest.mark.parametrize("name", RANKERS)
    def test_reports_no_regression_metrics(self, name):
        from standard_quant_tools.modeling.engine import run_experiment

        result = run_experiment(_ranking_dataset(), _ranking_spec(name), "ds")
        assert "r2" not in result["oos_metrics"]
        assert "mae" not in result["oos_metrics"]

    def test_a_discrete_target_is_rejected_for_ranking(self):
        from standard_quant_tools.modeling.engine import run_experiment

        dataset = _ranking_dataset()
        dataset["target_id"] = "forward_direction:5"
        with pytest.raises(ValidationError, match="task='ranking' expects one of"):
            run_experiment(dataset, _ranking_spec(RANKERS[0]), "ds")

    def test_custom_grade_count_is_honoured(self):
        from standard_quant_tools.modeling.engine import run_experiment

        result = run_experiment(
            _ranking_dataset(),
            _ranking_spec(RANKERS[0], ranking=RankingSpec(n_grades=4, ndcg_at=[3])),
            "ds",
        )
        assert "ndcg_at_3" in result["oos_metrics"]
        assert "ndcg_at_5" not in result["oos_metrics"]

    def test_row_order_does_not_change_the_result(self):
        """
        The silent failure mode this whole design guards against. Both
        libraries assume rows arrive grouped by date and never check, so a
        shuffled panel must still produce the same model — the engine sorts
        before fitting rather than trusting the caller.
        """
        from standard_quant_tools.modeling.engine import run_experiment

        dataset = _ranking_dataset()
        ordered = run_experiment(dataset, _ranking_spec(RANKERS[0]), "ds")

        shuffled = dict(dataset)
        rng = np.random.default_rng(9)
        panel = dataset["panel"]
        shuffled["panel"] = panel.iloc[rng.permutation(len(panel))].reset_index(
            drop=True
        )
        from standard_quant_tools.audit.hashing import hash_dataframe

        shuffled["data_hash"] = hash_dataframe(shuffled["panel"])
        result = run_experiment(shuffled, _ranking_spec(RANKERS[0]), "ds")

        assert result["oos_metrics"]["cs_rank_ic_mean"] == pytest.approx(
            ordered["oos_metrics"]["cs_rank_ic_mean"], abs=1e-9
        )

    def test_the_registered_model_is_refit_the_same_way_it_was_validated(self):
        """
        The final estimator is refit on the whole panel and is what actually
        scores. If it were fitted without grading and grouping, the deployed
        model would differ from the validated one — the quietest way to make
        a validation number describe something else.
        """
        from standard_quant_tools.modeling.engine import run_experiment
        from standard_quant_tools.modeling.registry.model_registry import load_model

        result = run_experiment(_ranking_dataset(), _ranking_spec(RANKERS[0]), "ds")
        estimator = load_model(result["model_id"])
        scores = estimator.predict(np.array([[1.0, 0.0], [-1.0, 0.0]]))
        assert scores.shape == (2,)
        # Higher f1 is a better name in this data, so it must score higher.
        assert scores[0] > scores[1]


class TestManifestSurvivesNaNRoundTrip:
    """
    A pre-existing bug this work surfaced, pinned separately because it has
    nothing to do with ranking.

    JSON has no NaN, so a NaN metric is written as `null`. ModelManifest
    declared those fields as plain `float` and therefore rejected the file it
    had itself written. Since summarize_importance correctly reports
    signed_mean / signed_std / sign_consistency as NaN for any estimator with
    no coefficient sign, EVERY tree-based model was unloadable —
    random_forest, gradient boosting, LightGBM, XGBoost — and so unscoreable,
    since score_model goes through load_model. Linear models were unaffected,
    which is why it survived: they have a sign, so none of their fields were
    ever NaN.
    """

    def test_a_manifest_with_null_metrics_loads(self):
        from standard_quant_tools.modeling.registry.manifests import ModelManifest

        manifest = ModelManifest(
            model_id="m",
            version=1,
            task="regression",
            estimator_type="random_forest",
            estimator_params={},
            feature_ids=["f1"],
            target_id="forward_return:5",
            dataset_id="d",
            dataset_hash="h",
            validation_method="walk_forward",
            # exactly what json.load gives back for a NaN it wrote
            oos_metrics={"r2": 0.1, "auc": None},
            feature_importance_summary={
                "f1": {"mean": 1.0, "signed_mean": None, "sign_consistency": None}
            },
            n_folds=2,
            oos_predictions_uri="u",
            random_seed=1,
            created_at_utc="2026-01-01T00:00:00Z",
        )
        assert np.isnan(manifest.oos_metrics["auc"])
        assert np.isnan(manifest.feature_importance_summary["f1"]["signed_mean"])
        assert manifest.oos_metrics["r2"] == 0.1

    @pytest.mark.parametrize(
        "estimator,params",
        [("ridge", {}), ("random_forest", {"n_estimators": 10})],
    )
    def test_a_registered_model_can_be_loaded_back(self, estimator, params):
        """The end-to-end version: register, then load. random_forest is the
        case that used to fail; ridge is the control that never did."""
        from standard_quant_tools.modeling.engine import run_experiment
        from standard_quant_tools.modeling.registry.model_registry import load_model

        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type=estimator, params=params),
            validation=ValidationSpec(train_window=150, test_window=75, embargo=2),
            random_seed=1,
        )
        result = run_experiment(_ranking_dataset(), spec, "ds")
        assert load_model(result["model_id"]) is not None
