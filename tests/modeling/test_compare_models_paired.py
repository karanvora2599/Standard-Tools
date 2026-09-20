"""
`compare_models(method='paired')` end to end: two registered models on the
same label, compared on the rows both predicted.

The exact oracle is a model compared against its own twin -- the same
spec registered twice -- whose per-date difference is identically zero:
mean zero, an interval that contains zero, a hit rate of zero, and the
verdict 'indistinguishable'. The structural checks then run on a genuinely
different candidate, and the refusals on a candidate with a different task
or label.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    CompareModelsInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    compare_models,
    run_model_experiment,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _dataset(target: TargetSpec, features=None) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=features
        or [
            FeatureSpec(id="technical.rsi"),
            FeatureSpec(id="market.momentum"),
            FeatureSpec(id="risk.realized_volatility"),
        ],
        target=target,
        benchmark="SPY",
    )


def _model(task="regression", estimator="ridge", seed=1, **params) -> ModelSpec:
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=seed,
    )


@pytest.fixture
def regression_dataset(patched_multi_factory) -> str:
    return build_model_dataset(
        BuildModelDatasetInput(spec=_dataset(TargetSpec(horizon=5)))
    ).dataset_id


def _train(dataset_id: str, spec: ModelSpec) -> str:
    return run_model_experiment(
        RunModelExperimentInput(dataset_id=dataset_id, spec=spec)
    ).model_id


class TestATwinIsIndistinguishable:
    def test_the_difference_is_exactly_zero(self, regression_dataset):
        a = _train(regression_dataset, _model(alpha=1.0))
        twin = _train(regression_dataset, _model(alpha=1.0))
        result = compare_models(
            CompareModelsInput(model_ids=[a, twin], method="paired", n_bootstrap=200)
        )
        assert result.method == "paired"
        assert result.reference_model_id == a
        assert len(result.pairs) == 1
        pair = result.pairs[0]
        assert pair.model_id == twin and pair.reference_model_id == a
        assert pair.mean_difference == 0.0
        assert pair.ci_lower <= 0.0 <= pair.ci_upper
        assert pair.hit_rate == 0.0
        assert pair.verdict == "indistinguishable"
        assert pair.mean_candidate == pair.mean_reference
        assert pair.p_value_holm >= pair.p_value
        assert any("Holm" in note for note in result.notes)


class TestADifferentCandidate:
    def test_the_comparison_is_populated_and_consistent(self, regression_dataset):
        a = _train(regression_dataset, _model(alpha=1.0))
        b = _train(regression_dataset, _model(estimator="huber"))
        result = compare_models(
            CompareModelsInput(
                model_ids=[a, b], method="paired", reference_model_id=a, n_bootstrap=300
            )
        )
        pair = result.pairs[0]
        assert pair.n_dates > 30 and pair.n_rows > pair.n_dates
        assert pair.ci_lower <= pair.mean_difference <= pair.ci_upper
        assert 0.0 <= pair.p_value <= 1.0
        assert pair.verdict in {"candidate_better", "reference_better", "indistinguishable"}
        assert pair.diebold_mariano is not None
        assert pair.diebold_mariano["loss"] == "squared_error"
        assert pair.diebold_mariano["lag"] == 4  # a five-bar label overlaps four
        # The headline ranking is still reported beside it.
        assert {c.model_id for c in result.comparisons} == {a, b}

    def test_holm_adjusts_across_several_candidates(self, regression_dataset):
        a = _train(regression_dataset, _model(alpha=1.0))
        b = _train(regression_dataset, _model(estimator="huber"))
        c = _train(regression_dataset, _model(alpha=100.0))
        result = compare_models(
            CompareModelsInput(
                model_ids=[a, b, c], method="paired", reference_model_id=a, n_bootstrap=200
            )
        )
        assert [p.model_id for p in result.pairs] == [b, c]
        for pair in result.pairs:
            assert pair.p_value_holm >= pair.p_value
        assert max(p.p_value_holm for p in result.pairs) <= 1.0

    def test_the_default_reference_is_the_first_model(self, regression_dataset):
        a = _train(regression_dataset, _model(alpha=1.0))
        b = _train(regression_dataset, _model(estimator="huber"))
        result = compare_models(
            CompareModelsInput(model_ids=[b, a], method="paired", n_bootstrap=100)
        )
        assert result.reference_model_id == b
        assert result.pairs[0].model_id == a


class TestRefusals:
    def test_a_reference_outside_the_candidates(self):
        with pytest.raises(PydanticValidationError, match="not in model_ids"):
            CompareModelsInput(
                model_ids=["mdl_a", "mdl_b"], method="paired", reference_model_id="mdl_c"
            )

    def test_a_different_label_is_refused(self, patched_multi_factory):
        five = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset(TargetSpec(horizon=5)))
        ).dataset_id
        ten = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset(TargetSpec(horizon=10)))
        ).dataset_id
        a = _train(five, _model(alpha=1.0))
        b = _train(ten, _model(alpha=1.0))
        with pytest.raises(ValidationError, match="two labels are two questions"):
            compare_models(CompareModelsInput(model_ids=[a, b], method="paired"))

    def test_a_different_task_is_refused(self, patched_multi_factory):
        regression = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset(TargetSpec(horizon=5)))
        ).dataset_id
        classification = build_model_dataset(
            BuildModelDatasetInput(
                spec=_dataset(TargetSpec(type="forward_direction", horizon=5))
            )
        ).dataset_id
        a = _train(regression, _model(alpha=1.0))
        b = _train(classification, _model(task="classification", estimator="logistic"))
        with pytest.raises(ValidationError, match="not the same quantity"):
            compare_models(CompareModelsInput(model_ids=[a, b], method="paired"))

    def test_the_headline_method_is_unchanged(self, regression_dataset):
        a = _train(regression_dataset, _model(alpha=1.0))
        b = _train(regression_dataset, _model(estimator="huber"))
        result = compare_models(CompareModelsInput(model_ids=[a, b]))
        assert result.method == "headline"
        assert result.pairs == [] and result.reference_model_id is None
        assert result.best_by_task["regression"] in {a, b}
