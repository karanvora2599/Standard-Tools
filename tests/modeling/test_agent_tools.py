"""Tests for the 5-tool modeling agent surface: full pipeline through
list_features -> build_model_dataset -> run_model_experiment ->
score_model -> inspect_model, modeling_dispatch routing, and the
architectural guarantee that these tools never leak into the existing
46-tool get_agent_tools()/TOOL_CATEGORY registry."""

import pytest

from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    InspectModelInput,
    ListFeaturesInput,
    RunModelExperimentInput,
    ScoreModelInput,
    build_model_dataset,
    get_modeling_tools,
    inspect_model,
    list_features,
    modeling_dispatch,
    run_model_experiment,
    score_model,
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
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="risk.rolling_beta")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _model_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=11,
    )


class TestListFeatures:
    def test_returns_full_catalog_by_default(self):
        result = list_features(ListFeaturesInput(category=None))
        assert len(result.features) >= 21

    def test_filters_by_category(self):
        result = list_features(ListFeaturesInput(category="risk"))
        assert {f.id for f in result.features} == {
            "risk.realized_volatility",
            "risk.rolling_beta",
            "risk.atr_pct",
            "risk.bollinger_pct_b",
            "risk.parkinson_volatility",
            "risk.garman_klass_volatility",
            "risk.rolling_drawdown",
        }


class TestFullPipeline:
    def test_build_run_score_inspect(self, patched_multi_factory):
        ds_result = build_model_dataset(BuildModelDatasetInput(spec=_dataset_spec()))
        assert ds_result.rows > 0
        assert ds_result.dataset_id.startswith("ds_")

        exp_result = run_model_experiment(
            RunModelExperimentInput(dataset_id=ds_result.dataset_id, spec=_model_spec())
        )
        assert exp_result.model_id.startswith("mdl_")
        assert exp_result.n_folds >= 1

        score_result = score_model(
            ScoreModelInput(
                model_id=exp_result.model_id,
                as_of="2023-12-29",
                universe=["AAA", "BBB", "CCC"],
            )
        )
        assert score_result.n_entities == 3
        assert score_result.predictions_uri

        for view in ("summary", "feature_importance", "validation", "lineage"):
            inspect_result = inspect_model(
                InspectModelInput(model_id=exp_result.model_id, view=view)
            )
            assert inspect_result.view == view
            assert inspect_result.data

    def test_pipeline_via_modeling_dispatch(self, patched_multi_factory):
        """The same pipeline, routed through modeling_dispatch (as an LLM
        tool call would be) instead of calling the tool functions directly."""
        ds_dict = modeling_dispatch(
            "build_model_dataset", {"spec": _dataset_spec().model_dump()}
        )
        exp_dict = modeling_dispatch(
            "run_model_experiment",
            {"dataset_id": ds_dict["dataset_id"], "spec": _model_spec().model_dump()},
        )
        score_dict = modeling_dispatch(
            "score_model",
            {
                "model_id": exp_dict["model_id"],
                "as_of": "2023-12-29",
                "universe": ["AAA", "BBB", "CCC"],
            },
        )
        assert score_dict["n_entities"] == 3


class TestModelingDispatch:
    def test_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown modeling tool"):
            modeling_dispatch("not_a_real_tool", {})

    def test_get_modeling_tools_returns_exactly_five(self):
        tools = get_modeling_tools()
        assert len(tools) == 5
        assert {t["function"]["name"] for t in tools} == {
            "list_features",
            "build_model_dataset",
            "run_model_experiment",
            "score_model",
            "inspect_model",
        }

    def test_tool_shape_matches_existing_agent_surface_envelope(self):
        """Same {"type": "function", "function": {...}} envelope as
        agent.tools.get_agent_tools(), so the same LLM client code can
        consume either registry."""
        tool = get_modeling_tools()[0]
        assert tool["type"] == "function"
        assert {"name", "description", "parameters"} <= set(tool["function"].keys())


class TestArchitecturalSeparationFromExistingAgentSurface:
    """The core architectural rule this whole runtime exists to enforce:
    modeling tools are a separate registry, never merged into the
    existing 46-tool get_agent_tools()/TOOL_CATEGORY surface."""

    def test_modeling_tool_names_absent_from_existing_tool_category(self):
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        modeling_names = {t["function"]["name"] for t in get_modeling_tools()}
        assert not modeling_names & set(TOOL_CATEGORY.keys())

    def test_modeling_tool_names_absent_from_get_agent_tools(self):
        from standard_quant_tools.agent.tools import get_agent_tools

        existing_names = {t["function"]["name"] for t in get_agent_tools()}
        modeling_names = {t["function"]["name"] for t in get_modeling_tools()}
        assert not modeling_names & existing_names
