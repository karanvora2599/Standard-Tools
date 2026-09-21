"""
Phase 2 of Development/databento_live_fix_plan.md: the deployed model is
the validated model.

The live findings (Development/databento_live_findings.md, D14-D18, D20)
measured each of these on real prices. The tests here reproduce each
defect's shape offline and pin the fix:

  D14  the refit deploys the parameters a search chose, never the base
  D15  a cross-sectional model is pinned to its universe, waivable by name
  D16  the lineage names the feed each entity's bars came from
  D17  a universe-scope feature's value does not depend on the frame start
  D18  a feature absent from a fold's training rows is refused by name,
       and the per-fold missing rate travels with the report
  D20  the delisting diagnostic names the entity that binds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDRegressor

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.adapters import RegressionAdapter
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    InspectModelInput,
    RunModelExperimentInput,
    ScoreModelInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    inspect_model,
    run_model_experiment,
)
from standard_quant_tools.modeling.agent.tools import score_model as score_model_tool
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.dataset.alignment import build_returns_panel
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.coverage import intersection_warnings
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.features.schedule import bar_ordinals, refit_mask
from standard_quant_tools.modeling.features.transforms import apply_preprocessing
from standard_quant_tools.modeling.plan import plan_experiment
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    SearchSpec,
    StepSpec,
    TargetSpec,
    ValidationSpec,
)

from .conftest import make_ohlcv

GRID = [0.001, 100.0, 10000.0]


def _dataset_spec(**overrides) -> DatasetSpec:
    defaults = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


def _model_spec(
    *, search=None, normalization="pooled", **estimator_params
) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(
            type="ridge", params={"alpha": 1.0, **estimator_params}
        ),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        preprocessing=PreprocessingSpec(normalization=normalization),
        search=search,
        random_seed=1,
    )


def _ridge_grid() -> SearchSpec:
    return SearchSpec(param_grid={"alpha": GRID}, inner_splits=2)


# ── D14 ──────────────────────────────────────────────────────────────────


class TestTheDeployedParametersAreTheSearchedOnes:
    """
    Each fold reassigned `fold_params` from its inner search, and the refit
    instantiated from the spec's base values: a ridge searched over
    {0.001, 100, 10000} was deployed at alpha=1.0, which no fold had scored.
    """

    def test_a_ridge_grid_deploys_a_grid_value_never_the_base(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _model_spec(search=_ridge_grid()), "ds_d14")
        manifest = load_manifest(result["model_id"])
        alpha = manifest.estimator_params["alpha"]
        assert alpha in GRID and alpha != 1.0
        # The artifact that scores carries the same value the manifest records.
        assert load_model(result["model_id"]).alpha == alpha

    def test_the_choice_is_the_full_panel_search_and_the_report_says_so(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _model_spec(search=_ridge_grid()), "ds_d14b")
        report = result["validation_report"]
        manifest = load_manifest(result["model_id"])
        assert manifest.deployed_params_source == "full_panel_search"
        assert report["deployed_params_source"] == "full_panel_search"
        final = report["final_search"]
        assert final["searched"] and final["n_inner_folds"] == 2
        assert final["best_params"]["alpha"] == manifest.estimator_params["alpha"]
        assert report["deployed_params"] == manifest.estimator_params
        # The per-fold selections remain beside it, so a reader can see
        # whether the deployed choice agrees with what the folds validated.
        assert all(r["searched"] for r in report["hyperparameter_search"])

    def test_the_final_search_is_planned_and_counted(self, patched_multi_factory):
        dataset = build_dataset(_dataset_spec())
        spec = _model_spec(search=_ridge_grid())
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        plan = plan_experiment(spec, dates, panel=dataset["panel"])
        assert plan.n_inner_final == 2
        assert plan.n_fits_final_search == len(GRID) * 2
        assert plan.n_fits_refit == 1 + len(GRID) * 2
        assert plan.n_fits == plan.n_fits_folds + plan.n_fits_refit
        result = run_experiment(dataset, spec, "ds_d14c", register=False)
        fits = result["validation_report"]["fits"]
        assert fits["final_search"] == len(GRID) * 2
        assert fits["planned"] == plan.n_fits

    def test_the_final_search_is_refused_over_budget_like_any_other_fit(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        spec = _model_spec(search=_ridge_grid())
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        just_short = plan_experiment(spec, dates, panel=dataset["panel"]).n_fits - 1
        tight = spec.model_copy(
            update={"budget": spec.budget.model_copy(update={"max_fits": just_short})}
        )
        with pytest.raises(ValidationError, match="full panel"):
            run_experiment(dataset, tight, "ds_d14d", register=False)

    def test_without_a_search_the_spec_is_deployed_and_says_so(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _model_spec(alpha=2.5), "ds_d14e")
        manifest = load_manifest(result["model_id"])
        assert manifest.estimator_params == {"alpha": 2.5}
        assert manifest.deployed_params_source == "spec"
        assert result["validation_report"]["final_search"] is None

    def test_the_summary_view_shows_the_deployed_parameters(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _model_spec(search=_ridge_grid()), "ds_d14f")
        view = inspect_model(
            InspectModelInput(model_id=result["model_id"], view="summary")
        )
        assert view.data["estimator_params"]["alpha"] in GRID
        assert view.data["deployed_params_source"] == "full_panel_search"


# ── D15 ──────────────────────────────────────────────────────────────────


class TestACrossSectionalModelIsPinnedToItsUniverse:
    """
    `cross_sectional_standardize` fits nothing: it standardizes within the
    rows of the call. Narrowing eight trained names to three moved one
    row's score by 544% and inverted a forest's ranking, and nothing warned.
    """

    def _train(self, normalization: str, dataset_id: str) -> str:
        # What build_model_dataset + run_model_experiment do: the spec is
        # persisted beside the panel and bundled into the model, which is
        # what scoring rebuilds features from.
        spec = _dataset_spec()
        built = build_dataset(spec)
        directory = _artifacts.run_dir(dataset_id)
        _artifacts.save_artifact(built["panel"], run_id=dataset_id, name="panel")
        _artifacts.save_json(directory, "dataset_spec", spec.model_dump())
        dataset = {
            "panel": built["panel"],
            "feature_ids": built["feature_ids"],
            "target_id": built["target_id"],
            "data_hash": built["data_hash"],
            "spec_hash": built["spec_hash"],
            "dataset_spec": spec.model_dump(),
        }
        result = run_experiment(
            dataset, _model_spec(normalization=normalization), dataset_id
        )
        return result["model_id"]

    def test_a_subset_is_refused_by_name(self, patched_multi_factory):
        model_id = self._train("cross_sectional", "ds_d15a")
        with pytest.raises(ValidationError, match="cross_sectional_standardize"):
            score_model(model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB"])

    def test_the_training_universe_in_any_order_scores_without_a_warning(
        self, patched_multi_factory
    ):
        model_id = self._train("cross_sectional", "ds_d15b")
        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["CCC", "AAA", "BBB"]
        )
        assert result["n_entities"] == 3 and result["warnings"] == []

    def test_allow_scores_and_says_what_was_refit(self, patched_multi_factory):
        model_id = self._train("cross_sectional", "ds_d15c")
        result = score_model(
            model_id=model_id,
            as_of="2023-12-29",
            universe=["AAA", "BBB"],
            universe_policy="allow",
        )
        assert result["n_entities"] == 2
        (warning,) = result["warnings"]
        assert "refit on the scoring cross-section of 2 entities" in warning
        assert "median 3" in warning

    def test_a_pooled_model_is_not_pinned(self, patched_multi_factory):
        model_id = self._train("pooled", "ds_d15d")
        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB"]
        )
        assert result["n_entities"] == 2 and result["warnings"] == []

    def test_the_manifest_records_the_training_width(self, patched_multi_factory):
        model_id = self._train("cross_sectional", "ds_d15e")
        width = load_manifest(model_id).training_cross_section
        assert width["min"] == 3 and width["max"] == 3 and width["median"] == 3.0
        assert width["n_dates"] > 100

    def test_the_tool_carries_the_policy_and_the_warning(self, patched_multi_factory):
        model_id = self._train("cross_sectional", "ds_d15f")
        with pytest.raises(ValidationError, match="universe_policy='allow'"):
            score_model_tool(
                ScoreModelInput(model_id=model_id, as_of="2023-12-29", universe=["AAA"])
            )
        result = score_model_tool(
            ScoreModelInput(
                model_id=model_id,
                as_of="2023-12-29",
                universe=["AAA", "CCC"],
                universe_policy="allow",
            )
        )
        assert len(result.warnings) == 1

    def test_an_unknown_policy_is_refused(self, patched_multi_factory):
        model_id = self._train("pooled", "ds_d15g")
        with pytest.raises(ValidationError, match="universe_policy"):
            score_model(
                model_id=model_id,
                as_of="2023-12-29",
                universe=["AAA"],
                universe_policy="maybe",
            )


# ── D16 ──────────────────────────────────────────────────────────────────


class TestTheLineageNamesTheFeed:
    """Two models built from one spec on two feeds carried identical
    recorded identity and differed by 22% on the headline metric."""

    def test_the_dataset_meta_and_the_manifest_carry_data_sources(
        self, patched_multi_factory
    ):
        spec = _dataset_spec()
        built = build_model_dataset(BuildModelDatasetInput(spec=spec))
        assert set(built.data_sources) == {"AAA", "BBB", "CCC"}
        assert all(v.startswith(spec.provider) for v in built.data_sources.values())
        meta = _artifacts.load_json(
            str(_artifacts.run_dir(built.dataset_id) / "dataset_meta.json")
        )
        assert meta["data_sources"] == built.data_sources
        run = run_model_experiment(
            RunModelExperimentInput(dataset_id=built.dataset_id, spec=_model_spec())
        )
        manifest = load_manifest(run.model_id)
        assert manifest.data_sources == built.data_sources
        lineage = inspect_model(
            InspectModelInput(model_id=run.model_id, view="lineage")
        )
        assert lineage.data["data_sources"] == built.data_sources

    def test_a_provider_that_names_its_dataset_is_recorded_per_entity(
        self, patched_multi_factory
    ):
        provider = patched_multi_factory

        def tagged(symbol, *args, **kwargs):
            frame = make_ohlcv(symbol)
            frame.attrs["dataset"] = "EQUS.SUMMARY" if symbol != "CCC" else "EQUS.MINI"
            return frame

        # The builder fetches through the async path; both agree.
        provider.get_ohlcv.side_effect = tagged
        provider.get_ohlcv_async.side_effect = tagged
        spec = _dataset_spec()
        built = build_dataset(spec)
        assert built["data_sources"] == {
            "AAA": f"{spec.provider}:EQUS.SUMMARY",
            "BBB": f"{spec.provider}:EQUS.SUMMARY",
            "CCC": f"{spec.provider}:EQUS.MINI",
        }


# ── D17 ──────────────────────────────────────────────────────────────────


UNIVERSE_FEATURES = (
    "factors.pca_loading",
    "factors.pca_factor_return",
    "network.avg_correlation",
    "network.mst_degree",
)


class TestAUniverseScopeValueDoesNotDependOnTheFrameStart:
    """
    The refit grid was anchored on the frame's first bar, so recomputing
    after dropping k leading bars was bit-identical only when k was a
    multiple of refit_every; at k=1 zero of 2,080 values matched.
    """

    @pytest.fixture(scope="class")
    def returns_panel(self) -> pd.DataFrame:
        closes = {
            s: make_ohlcv(s, n=400)["Close"] for s in ("AAA", "BBB", "CCC", "DDD")
        }
        return build_returns_panel(closes)

    @pytest.mark.parametrize("feature_id", UNIVERSE_FEATURES)
    @pytest.mark.parametrize("k", [1, 3, 7, 10, 13])
    def test_dropping_leading_bars_leaves_every_computable_value_identical(
        self, returns_panel, feature_id, k
    ):
        fn = get_feature(feature_id).fn
        full = fn(returns_panel, None, window=100, refit_every=10)
        part = fn(returns_panel.iloc[k:], None, window=100, refit_every=10)
        valid = part.dropna(how="all")
        assert len(valid) > 250
        np.testing.assert_array_equal(
            valid.to_numpy(), full.loc[valid.index].to_numpy()
        )

    @pytest.mark.parametrize("feature_id", UNIVERSE_FEATURES)
    def test_truncating_trailing_bars_leaves_every_earlier_value_identical(
        self, returns_panel, feature_id
    ):
        fn = get_feature(feature_id).fn
        full = fn(returns_panel, None, window=100, refit_every=10)
        cut = 250
        part = fn(returns_panel.iloc[: cut + 1], None, window=100, refit_every=10)
        assert np.array_equal(
            part.to_numpy(), full.iloc[: cut + 1].to_numpy(), equal_nan=True
        )

    def test_the_grid_is_a_function_of_the_date(self):
        daily = pd.bdate_range("2024-01-02", periods=30)
        ordinals = bar_ordinals(daily)
        assert np.all(np.diff(ordinals) == 1)
        # The same date has the same number in a frame that starts later.
        assert bar_ordinals(daily[7:])[0] == ordinals[7]
        # A grid bar is one whose number is a multiple of refit_every, once
        # a full window precedes it.
        mask = refit_mask(daily, window=5, refit_every=3)
        assert not mask[:4].any()
        assert np.array_equal(
            np.flatnonzero(mask),
            np.flatnonzero((ordinals % 3 == 0) & (np.arange(30) >= 4)),
        )

    def test_intraday_bars_count_their_own_spacing(self):
        minutes = pd.date_range("2024-01-02 09:30", periods=100, freq="min")
        ordinals = bar_ordinals(minutes)
        assert np.all(np.diff(ordinals) == 1)
        assert bar_ordinals(minutes[40:])[0] == ordinals[40]

    def test_a_positional_index_counts_positions(self):
        assert list(bar_ordinals(pd.RangeIndex(5))) == [0, 1, 2, 3, 4]


# ── D18 ──────────────────────────────────────────────────────────────────


def _holed_dataset(n_missing_dates: int, n: int = 200) -> dict:
    """Two entities, two features; `f_late` is NaN for the first
    `n_missing_dates` dates of every entity."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    rows = []
    for entity in ("A", "B"):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "entity": entity,
                    "f_early": rng.normal(),
                    "f_late": np.nan if i < n_missing_dates else rng.normal(),
                    "target": rng.normal() * 0.01,
                }
            )
    return {
        "panel": pd.DataFrame(rows),
        "feature_ids": ["f_early", "f_late"],
        "target_id": "forward_return:5",
        "data_hash": "holed",
    }


def _imputing_spec(estimator: str = "ridge", **params) -> ModelSpec:
    steps = [StepSpec(type="impute"), StepSpec(type="zscore")]
    if estimator == "hist_gradient_boosting":
        steps = []
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(train_window=60, test_window=20, embargo=0),
        preprocessing=PreprocessingSpec(steps=steps) if steps else PreprocessingSpec(),
        random_seed=1,
    )


class TestAFeatureAbsentFromAFoldIsRefused:
    """
    Five of eight live folds trained on a feature that was 100% NaN in
    their training rows -- imputed at 0.0, a constant -- while the run
    reported full fold coverage and averaged its importance in.
    """

    def test_an_all_missing_training_column_is_refused_with_the_fold(self):
        with pytest.raises(ValidationError, match=r"\['f_late'\].*fold 0"):
            run_experiment(
                _holed_dataset(80), _imputing_spec(alpha=1.0), "ds", register=False
            )

    def test_hist_gradient_boosting_is_refused_by_name_not_by_numpy(self):
        with pytest.raises(ValidationError, match="f_late"):
            run_experiment(
                _holed_dataset(80),
                _imputing_spec("hist_gradient_boosting", max_iter=10),
                "ds",
                register=False,
            )

    def test_a_partly_missing_column_trains_and_the_rate_is_recorded_per_fold(
        self,
    ):
        result = run_experiment(
            _holed_dataset(30), _imputing_spec(alpha=1.0), "ds", register=False
        )
        report = result["validation_report"]
        rates = report["missing_rate_by_fold"]
        assert rates["f_early"] == [0.0] * result["n_folds"]
        assert rates["f_late"][0] == pytest.approx(0.5)
        assert rates["f_late"][-1] == 0.0
        assert report["folds"][0]["missing_rate_train"]["f_late"] == pytest.approx(0.5)

    def test_a_panel_without_holes_records_none(self):
        result = run_experiment(
            _holed_dataset(0), _imputing_spec(alpha=1.0), "ds", register=False
        )
        assert result["validation_report"]["missing_rate_by_fold"] is None
        assert result["validation_report"]["folds"][0]["missing_rate_train"] is None


# ── D20 ──────────────────────────────────────────────────────────────────


class TestTheDelistingDiagnosticNamesTheBindingEntity:
    """All nine live names started on the same date and one was delisted;
    the warning named AAPL, dropping which recovered nothing."""

    def _panel(self, frames):
        return build_returns_panel({s: f["Close"] for s, f in frames.items()})

    def test_the_entity_that_ends_early_is_named(self):
        frames = {s: make_ohlcv(s, n=400) for s in ("AAA", "BBB", "CCC")}
        frames["DDD"] = make_ohlcv("DDD", n=400).iloc[:100]
        (message,) = intersection_warnings(frames, self._panel(frames), True)
        assert "The binding symbol is DDD" in message
        assert "recovers 300 of the 300 missing date(s)" in message
        assert "AAA" not in message

    def test_the_entity_that_starts_late_is_still_named(self):
        frames = {s: make_ohlcv(s, n=400) for s in ("AAA", "BBB")}
        frames["CCC"] = frames["AAA"].iloc[300:].copy()
        (message,) = intersection_warnings(frames, self._panel(frames), True)
        assert "The binding symbol is CCC" in message and "25%" in message

    def test_two_short_histories_are_reported_as_jointly_binding(self):
        frames = {s: make_ohlcv(s, n=400) for s in ("AAA", "BBB")}
        frames["CCC"] = frames["AAA"].iloc[:100].copy()
        frames["DDD"] = frames["BBB"].iloc[:100].copy()
        (message,) = intersection_warnings(frames, self._panel(frames), True)
        assert "No single symbol is binding" in message


# ── Also ─────────────────────────────────────────────────────────────────


class TestTheSmallerItems:
    def test_sgd_reports_its_coefficients(self):
        assert RegressionAdapter().capabilities(SGDRegressor)["exposes_coefficients"]
        entries = {
            (e["task"], e["name"]): e for e in modeling_capabilities()["estimators"]
        }
        assert entries[("regression", "sgd")]["exposes_coefficients"] is True

    def test_the_legacy_stats_marker_is_refused_not_applied(self):
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="marker"):
            apply_preprocessing(frame, {"legacy": False, "note": "see the state"})
        # And the real form still applies.
        out = apply_preprocessing(
            frame, {"x": {"lo": 0.0, "hi": 3.0, "mean": 1.5, "std": 0.5}}
        )
        assert list(out["x"]) == [-1.0, 1.0]
