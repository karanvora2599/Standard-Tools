"""
Ablation: what each feature is worth to a fitted model.

Every other feature tool scores a feature ON ITS OWN. That is cheap and
usually right, and it cannot answer the question that decides whether a
feature stays in a model: how much worse the model gets without it. A strong
feature that duplicates another contributes nothing marginal; a mediocre
feature that is the sole source of some information can be the one holding
the model up.

THE TWO THINGS MOST LIKELY TO BE WRONG, and what pins them:

- **The sign.** `contribution` means "how much the model loses without this
  feature" for every metric, which means the subtraction has to reverse for
  metrics where lower is better. Getting that backwards ranks the most
  important feature last, and the result still looks like a plausible
  table. Tested directly on both metric directions.
- **The cost.** This refits per feature, so a 40-feature panel at 8 folds is
  328 fits. The refusal has to fire BEFORE any fitting, or the guard is
  decoration. Tested by asserting the refusal is instant and names the
  number.
"""

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent.feature_models import FeatureAblationInput
from standard_quant_tools.modeling.agent.feature_tools import run_feature_ablation
from standard_quant_tools.modeling.analysis.feature_ablation import (
    _lower_is_better,
    ablation_contributions,
    estimate_ablation_fits,
    summarize_ablation,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
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


def _model_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=60, embargo=5),
        random_seed=11,
    )


@pytest.fixture
def dataset(patched_multi_factory):
    return build_model_dataset(BuildModelDatasetInput(spec=_dataset_spec())).dataset_id


class TestTheFitEstimate:
    def test_it_counts_the_baseline_too(self):
        """(features + 1) x folds. Forgetting the baseline understates the
        cost by a fold, which is the difference between a guard that fires
        and one that fires late."""
        assert estimate_ablation_fits(n_features=40, n_folds=8) == 328
        assert estimate_ablation_fits(n_features=1, n_folds=1) == 2

    def test_it_refuses_nonsense(self):
        with pytest.raises(ValidationError):
            estimate_ablation_fits(n_features=0, n_folds=4)
        with pytest.raises(ValidationError):
            estimate_ablation_fits(n_features=4, n_folds=0)


class TestTheSign:
    """`contribution` is always 'how much the model loses without this
    feature'. For a lower-is-better metric that reverses the subtraction,
    and getting it wrong produces a plausible-looking table ranked exactly
    backwards."""

    def test_higher_is_better_metric(self):
        rows = ablation_contributions(
            baseline=0.60, without={"useful": 0.40, "useless": 0.61}, metric="r2"
        )
        by_name = {r["feature"]: r for r in rows}
        assert by_name["useful"]["contribution"] == pytest.approx(0.20)
        assert by_name["useless"]["contribution"] == pytest.approx(-0.01)
        assert rows[0]["feature"] == "useful"

    def test_lower_is_better_metric(self):
        """Removing a useful feature RAISES the error, so the contribution
        must come out positive."""
        rows = ablation_contributions(
            baseline=0.10, without={"useful": 0.30, "useless": 0.09}, metric="mae"
        )
        by_name = {r["feature"]: r for r in rows}
        assert by_name["useful"]["contribution"] == pytest.approx(0.20)
        assert by_name["useless"]["contribution"] == pytest.approx(-0.01)
        assert rows[0]["feature"] == "useful"

    @pytest.mark.parametrize(
        "metric,lower",
        [
            ("mae", True),
            ("mse", True),
            ("rmse", True),
            ("oos_mae", True),
            ("log_loss", True),
            ("r2", False),
            ("ic_mean", False),
            ("accuracy", False),
            ("mean_absolute_error", False),
        ],
    )
    def test_metric_direction_is_recognised(self, metric, lower):
        assert _lower_is_better(metric) is lower

    def test_ranking_is_stable_for_ties(self):
        rows = ablation_contributions(
            baseline=1.0, without={"b": 0.5, "a": 0.5}, metric="r2"
        )
        assert [r["feature"] for r in rows] == ["a", "b"]
        assert [r["rank"] for r in rows] == [1, 2]


class TestTheSummary:
    def test_a_negative_contribution_is_surfaced(self):
        rows = ablation_contributions(
            baseline=0.5, without={"harmful": 0.7, "good": 0.2}, metric="r2"
        )
        summary = summarize_ablation(rows, "r2")
        assert summary["n_negative_contributions"] == 1
        assert any("WORSE" in w for w in summary["warnings"])
        assert any("confirm on a second period" in w for w in summary["warnings"])

    def test_no_positive_contribution_says_so(self):
        rows = ablation_contributions(
            baseline=0.5, without={"a": 0.5, "b": 0.5}, metric="r2"
        )
        summary = summarize_ablation(rows, "r2")
        assert any(
            "no feature made a positive contribution" in w for w in summary["warnings"]
        )

    def test_best_and_worst_bracket_the_ranking(self):
        rows = ablation_contributions(
            baseline=1.0, without={"a": 0.1, "b": 0.9, "c": 0.5}, metric="r2"
        )
        summary = summarize_ablation(rows, "r2")
        assert summary["best_feature"] == "a"
        assert summary["worst_feature"] == "b"


class TestTheGuard:
    def test_an_oversized_run_is_refused_before_fitting(self, dataset):
        """The guard is the whole reason this tool is safe to expose. If it
        fired after the fits, it would be documentation."""
        import time

        started = time.monotonic()
        with pytest.raises(ValidationError) as exc:
            run_feature_ablation(
                FeatureAblationInput(dataset_id=dataset, spec=_model_spec(), max_fits=1)
            )
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, (
            f"the refusal took {elapsed:.1f}s, which means fitting started "
            "before the budget was checked"
        )
        message = str(exc.value)
        assert "fits" in message
        assert "max_fits=" in message, "the refusal must name the way past it"

    def test_the_refusal_names_the_number_that_would_work(self, dataset):
        with pytest.raises(ValidationError) as exc:
            run_feature_ablation(
                FeatureAblationInput(dataset_id=dataset, spec=_model_spec(), max_fits=1)
            )
        message = str(exc.value)
        # The suggested max_fits has to be the actual requirement, or a
        # caller following the advice hits the same refusal again.
        import re

        suggested = int(re.search(r"max_fits=(\d+) to accept", message).group(1))
        result = run_feature_ablation(
            FeatureAblationInput(
                dataset_id=dataset, spec=_model_spec(), max_fits=suggested
            )
        )
        assert result.n_fits == suggested

    def test_one_feature_is_refused(self, dataset):
        """Removing the only feature leaves nothing to fit, so there is no
        comparison to make."""
        from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

        _panel, meta, _dir = _load_dataset_panel(dataset)
        with pytest.raises(ValidationError, match="at least two features"):
            run_feature_ablation(
                FeatureAblationInput(
                    dataset_id=dataset,
                    spec=_model_spec(),
                    features=[meta["feature_ids"][0]],
                    max_fits=1000,
                )
            )

    def test_a_mistyped_feature_is_refused_with_a_suggestion(self, dataset):
        from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

        _panel, meta, _dir = _load_dataset_panel(dataset)
        real = meta["feature_ids"][0]
        with pytest.raises(ValidationError) as exc:
            run_feature_ablation(
                FeatureAblationInput(
                    dataset_id=dataset,
                    spec=_model_spec(),
                    features=[real.replace(".", "_")],
                    max_fits=1000,
                )
            )
        assert real in str(exc.value)


class TestTheRun:
    def test_it_ranks_every_feature(self, dataset):
        result = run_feature_ablation(
            FeatureAblationInput(dataset_id=dataset, spec=_model_spec(), max_fits=1000)
        )
        assert result.n_features == len(result.contributions)
        assert [c.rank for c in result.contributions] == list(
            range(1, len(result.contributions) + 1)
        )
        assert result.n_fits == (result.n_features + 1) * result.n_folds

    def test_it_names_the_metric_it_ranked_on(self, dataset):
        """An ablation ranked on a metric the caller did not choose and
        cannot see is a table of numbers with no stated meaning."""
        result = run_feature_ablation(
            FeatureAblationInput(dataset_id=dataset, spec=_model_spec(), max_fits=1000)
        )
        assert result.metric
        assert isinstance(result.lower_is_better, bool)
        assert result.baseline_metric is not None

    def test_it_registers_no_models(self, dataset):
        """41 candidate models in the registry to answer one question is not
        a trade worth making."""
        from standard_quant_tools.modeling.agent.models import ListModelsInput
        from standard_quant_tools.modeling.agent.tools import list_models

        before = len(list_models(ListModelsInput()).models)
        run_feature_ablation(
            FeatureAblationInput(dataset_id=dataset, spec=_model_spec(), max_fits=1000)
        )
        after = len(list_models(ListModelsInput()).models)
        assert after == before, (
            f"ablation registered {after - before} models; refits during a "
            "comparison are candidates nobody asked for"
        )

    def test_an_unknown_metric_is_refused_with_the_options(self, dataset):
        with pytest.raises(ValidationError) as exc:
            run_feature_ablation(
                FeatureAblationInput(
                    dataset_id=dataset,
                    spec=_model_spec(),
                    metric="not_a_metric",
                    max_fits=1000,
                )
            )
        assert "Available:" in str(exc.value)


class TestRegisterFalseChangesNothingButPersistence:
    """The claim ablation rests on: an unregistered fit reports the same
    numbers a registered one would. If they diverged, ablation would be
    ranking features on a cheaper approximation while describing itself as
    the real thing."""

    def test_metrics_are_identical_either_way(self, dataset):
        from standard_quant_tools.modeling.agent.tools import _load_dataset_panel
        from standard_quant_tools.modeling.engine import run_experiment

        panel, meta, _dir = _load_dataset_panel(dataset)
        payload = {
            "panel": panel,
            "feature_ids": meta["feature_ids"],
            "target_id": meta["target_id"],
            "data_hash": meta["data_hash"],
            "spec_hash": meta.get("spec_hash"),
            "warnings": meta.get("warnings", []),
        }
        spec = _model_spec()
        registered = run_experiment(payload, spec, dataset_id=dataset)
        plain = run_experiment(payload, spec, dataset_id=dataset, register=False)

        assert plain["oos_metrics"] == registered["oos_metrics"]
        assert plain["n_folds"] == registered["n_folds"]
        assert plain["model_id"] is None
        assert plain["oos_predictions_uri"] is None
        assert registered["model_id"] is not None
