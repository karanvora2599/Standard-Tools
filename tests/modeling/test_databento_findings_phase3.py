"""
Phase 3 of Development/databento_live_fix_plan.md: selection and inference.

The live findings (Development/databento_live_findings.md, D4, D10, D19
and the CPCV and paired items) measured each of these on real prices. The
tests here reproduce each defect's shape offline and pin the fix:

  D4    select_features reads a selection window and reports the holdout IC
  D10   the permutation null keeps each feature's serial correlation
  D19   check_leakage runs the empirical screen when it has a panel
  CPCV  the paired comparison refuses a cpcv model; a fold record names
        its test blocks
  purge the report says when the purge could not run, and an external
        panel's label end is derived from its horizon
  ties  a paired comparison counts ties instead of scoring them as losses
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import SelectFeaturesInput
from standard_quant_tools.modeling.agent.feature_tools import select_features
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    CheckLeakageInput,
    CompareModelsInput,
    RegisterExternalPanelInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_panel,
    build_model_dataset,
    check_leakage,
    compare_models,
    register_external_panel,
    run_model_experiment,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as select_features_on,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    permutation_test_ic,
)
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.features.base import (
    FeatureDefinition,
    TemporalSupport,
)
from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.comparison import paired_comparison


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


def _walk_forward(**params) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0, **params}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


def _cpcv(n_splits: int = 4, n_test_splits: int = 2) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(
            method="cpcv", n_splits=n_splits, n_test_splits=n_test_splits, embargo=0
        ),
        random_seed=1,
    )


# ── D4 ───────────────────────────────────────────────────────────────────


def _noise_panel(n_features: int = 60, n_dates: int = 300, n_entities: int = 20):
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    rows = []
    for date in dates:
        for j in range(n_entities):
            row = {"date": date, "entity": f"E{j}", "target": rng.normal()}
            for k in range(n_features):
                row[f"noise_{k}"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows), [f"noise_{k}" for k in range(n_features)]


class TestSelectionDoesNotReadTheHoldout:
    """
    Sixty pure-noise columns, the top five by full-panel IC: the same
    walk-forward on those five scored +0.045 against +0.002 for five chosen
    blind -- 70% of the real model's headline, manufactured from noise.
    """

    def test_the_selection_window_and_the_holdout_are_reported(self):
        panel, features = _noise_panel()
        result = select_features_on(panel, features, max_features=5)
        assert result["selection_window"]["n_dates"] == 210
        assert result["holdout_window"]["n_dates"] == 90
        assert result["selection_window"]["end"] < result["holdout_window"]["start"]
        assert set(result["holdout_ic"]) == set(result["selected"])
        assert len(result["selected"]) == 5
        assert any("holdout_ic" in w for w in result["warnings"])

    def test_the_selected_noise_is_optimistic_in_sample_and_not_out(self):
        panel, features = _noise_panel()
        result = select_features_on(panel, features, max_features=5)
        chosen = result["selected"]
        in_sample = np.mean([abs(result["selection_ic"][f]) for f in chosen])
        # Signed by the selection's own sign, so a feature that flipped
        # counts against the selection rather than for it.
        aligned = np.mean(
            [
                np.sign(result["selection_ic"][f]) * result["holdout_ic"][f]
                for f in chosen
            ]
        )
        assert in_sample > 0.02
        assert aligned < in_sample / 2
        assert abs(aligned) < 0.02

    def test_selection_end_names_the_cutoff(self):
        panel, features = _noise_panel(n_features=5, n_dates=100)
        result = select_features_on(panel, features, selection_end="2022-03-31")
        assert result["selection_window"]["end"] == "2022-03-31"
        assert result["holdout_window"]["start"] > "2022-03-31"
        with pytest.raises(ValidationError, match="selection_end"):
            select_features_on(panel, features, selection_end="2030-01-01")

    def test_no_holdout_is_allowed_and_warned(self):
        panel, features = _noise_panel(n_features=5, n_dates=100)
        result = select_features_on(panel, features, holdout_fraction=0.0)
        assert result["holdout_window"] is None and result["holdout_ic"] == {}
        assert any("WHOLE panel" in w for w in result["warnings"])

    def test_the_tool_carries_the_windows(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        result = select_features(SelectFeaturesInput(dataset_id=dataset_id))
        assert result.holdout_window is not None
        assert result.selection_window["n_dates"] > result.holdout_window["n_dates"]
        assert set(result.holdout_ic) == set(result.selected)
        assert result.warnings
        explicit = select_features(
            SelectFeaturesInput(dataset_id=dataset_id, selection_end="2023-06-30")
        )
        assert explicit.selection_window["end"] == "2023-06-30"


# ── D10 ──────────────────────────────────────────────────────────────────


def _autocorrelated_panel(seed: int, n_dates: int = 200, n_entities: int = 20):
    """A feature with AR(1) phi=0.99 per entity, independent of a target
    that is an overlapping five-bar forward sum: the null is TRUE and both
    sides are autocorrelated, which is the live regime."""
    rng = np.random.default_rng(seed)
    horizon, phi = 5, 0.99
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    frames = []
    for j in range(n_entities):
        eps = rng.normal(size=n_dates + horizon)
        feature = np.empty(n_dates + horizon)
        feature[0] = eps[0]
        for t in range(1, n_dates + horizon):
            feature[t] = phi * feature[t - 1] + np.sqrt(1 - phi**2) * eps[t]
        returns = rng.normal(size=n_dates + horizon)
        cumulative = np.concatenate([[0.0], np.cumsum(returns)])
        target = (
            cumulative[horizon + 1 : n_dates + horizon + 1]
            - cumulative[1 : n_dates + 1]
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "entity": f"E{j}",
                    "feature": feature[:n_dates],
                    "target": target,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestTheNullKeepsTheSerialCorrelation:
    """
    The within-date shuffle rejected a true null 27-35% of the time on
    autocorrelated features against an overlapping label; every live
    feature sat in that regime.
    """

    def test_circular_shift_is_calibrated_where_within_date_is_not(self):
        seeds = range(20)
        circular = sum(
            permutation_test_ic(
                _autocorrelated_panel(seed),
                "feature",
                n_permutations=50,
                random_seed=seed,
            )["significant_at_05"]
            for seed in seeds
        )
        within = sum(
            permutation_test_ic(
                _autocorrelated_panel(seed),
                "feature",
                n_permutations=50,
                random_seed=seed,
                null="within_date",
            )["significant_at_05"]
            for seed in seeds
        )
        assert circular <= 4, f"{circular}/20 true nulls rejected under circular_shift"
        assert within >= 3, f"{within}/20: the within-date null should over-reject here"
        assert within > circular

    def test_the_default_is_still_calibrated_on_iid_noise(self):
        rng = np.random.default_rng(0)
        significant = 0
        for seed in range(10):
            dates = pd.bdate_range("2022-01-03", periods=120)
            rows = [
                {
                    "date": d,
                    "entity": f"E{j}",
                    "noise": rng.normal(),
                    "target": rng.normal(),
                }
                for d in dates
                for j in range(15)
            ]
            result = permutation_test_ic(
                pd.DataFrame(rows), "noise", n_permutations=100, random_seed=seed
            )
            significant += int(result["significant_at_05"])
        assert significant <= 3

    def test_the_result_names_the_null_and_the_regime(self):
        result = permutation_test_ic(
            _autocorrelated_panel(1), "feature", n_permutations=30, random_seed=1
        )
        assert result["null"] == "circular_shift"
        assert np.isfinite(result["ic_autocorrelation_lag1"])
        assert result["ic_autocorrelation_lag1"] > 0.3
        with pytest.raises(ValidationError, match="null="):
            permutation_test_ic(
                _autocorrelated_panel(1), "feature", n_permutations=30, null="global"
            )

    def test_a_real_signal_is_still_found(self):
        panel = _autocorrelated_panel(2)
        panel["real"] = panel["target"] * 0.5 + np.random.default_rng(2).normal(
            size=len(panel)
        )
        result = permutation_test_ic(panel, "real", n_permutations=100, random_seed=2)
        assert result["significant_at_05"]


# ── D19 ──────────────────────────────────────────────────────────────────


def _target_copy(ohlcv, context, horizon: int = 5):
    return ohlcv["Close"].pct_change(horizon).shift(-horizon)


LEAK = FeatureDefinition(
    id="leak.target_copy",
    description="The five-bar forward return, declared safe.",
    fn=_target_copy,
    default_params={"horizon": 5},
    temporal_support=TemporalSupport.PIT_SAFE,
    requires=["Close"],
    lookback=5,
)


class TestTheLeakageCheckScreensThePanel:
    """A feature that IS the target passed as safe=True on its declaration
    alone; the empirical screen on the same panel flags it at IC 1.000."""

    def test_a_copy_of_the_target_is_a_finding_with_a_dataset(
        self, patched_multi_factory, monkeypatch
    ):
        monkeypatch.setitem(FEATURE_REGISTRY, LEAK.id, LEAK)
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(
                spec=_dataset_spec(
                    features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id=LEAK.id)]
                )
            )
        ).dataset_id
        result = check_leakage(
            CheckLeakageInput(
                feature_ids=["technical.rsi", LEAK.id], dataset_id=dataset_id
            )
        )
        assert result.scope == "declared_and_empirical"
        assert not result.safe
        (finding,) = result.findings
        assert finding.feature_id == LEAK.id
        assert finding.temporal_support == "empirical"
        assert result.screen[LEAK.id]["flagged"]
        assert result.screen[LEAK.id]["ic_at_zero"] > 0.9
        assert not result.screen["technical.rsi"]["flagged"]

    def test_without_a_dataset_safe_says_what_it_rests_on(self, monkeypatch):
        monkeypatch.setitem(FEATURE_REGISTRY, LEAK.id, LEAK)
        result = check_leakage(CheckLeakageInput(feature_ids=[LEAK.id]))
        assert result.safe
        assert result.scope == "declared_temporal_support_only"
        assert result.screen == {}
        assert any("DECLARED" in note for note in result.notes)

    def test_honest_features_pass_the_screen(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        result = check_leakage(CheckLeakageInput(dataset_id=dataset_id))
        assert result.safe and result.scope == "declared_and_empirical"
        assert set(result.screen) == {"technical.rsi", "market.momentum"}


# ── CPCV ─────────────────────────────────────────────────────────────────


class TestCpcvIsRefusedByThePairedComparison:
    """The join on (date, entity) was a 25x cartesian product for a cpcv
    model, and produced a 'significant' p=0.013 from nothing."""

    def test_a_cpcv_candidate_or_reference_is_refused(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        walk = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_walk_forward())
        ).model_id
        combinatorial = run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_cpcv())
        ).model_id
        with pytest.raises(ValidationError, match="cpcv"):
            compare_models(
                CompareModelsInput(
                    model_ids=[walk, combinatorial], method="paired", n_bootstrap=100
                )
            )
        with pytest.raises(ValidationError, match="cpcv"):
            compare_models(
                CompareModelsInput(
                    model_ids=[combinatorial, walk], method="paired", n_bootstrap=100
                )
            )
        # The headline ranking still works: it reads the manifests only.
        ranked = compare_models(CompareModelsInput(model_ids=[walk, combinatorial]))
        assert len(ranked.comparisons) == 2


class TestAFoldRecordNamesItsTestBlocks:
    """One start..end span for a cpcv fold read as a window containing
    1,912 rows against n_test_rows 1,304: the training dates between the
    blocks were inside it."""

    def test_blocks_are_contiguous_runs_and_account_for_every_test_row(
        self, patched_multi_factory
    ):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _cpcv(), "ds_blocks", register=False)
        panel = dataset["panel"]
        dates = pd.to_datetime(panel["date"])
        multi_block = 0
        for record in result["validation_report"]["folds"]:
            blocks = record["test_blocks"]
            assert blocks and blocks[0]["start"] == record["test_start"]
            assert blocks[-1]["end"] == record["test_end"]
            in_blocks = sum(
                int(((dates >= b["start"]) & (dates <= b["end"])).sum()) for b in blocks
            )
            assert in_blocks == record["n_test_rows"]
            multi_block += len(blocks) > 1
        assert multi_block > 0

    def test_a_walk_forward_fold_has_one_block(self, patched_multi_factory):
        dataset = build_dataset(_dataset_spec())
        result = run_experiment(dataset, _walk_forward(), "ds_one", register=False)
        for record in result["validation_report"]["folds"]:
            assert len(record["test_blocks"]) == 1


# ── the purge report ─────────────────────────────────────────────────────


def _unlabelled_dataset(n: int = 200) -> dict:
    rng = np.random.default_rng(5)
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    rows = [
        {"date": d, "entity": e, "f": rng.normal(), "target": rng.normal() * 0.01}
        for e in ("A", "B")
        for d in dates
    ]
    return {
        "panel": pd.DataFrame(rows),
        "feature_ids": ["f"],
        "target_id": "forward_return:5",
        "data_hash": "unlabelled",
    }


class TestThePurgeReportSaysWhetherItRan:
    """Without a label_end_date column the purge was a no-op and wrote 0,
    the same value a clean run gives -- on a panel with 280 rows whose
    label reached the test window."""

    def test_no_label_end_is_not_applicable_not_zero(self):
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=60, test_window=20, embargo=0),
            search=SearchSpec(param_grid={"alpha": [0.1, 1.0]}, inner_splits=2),
            random_seed=1,
        )
        result = run_experiment(_unlabelled_dataset(), spec, "ds", register=False)
        report = result["validation_report"]
        assert report["purge"] == "not_applicable"
        assert report["n_train_rows_purged_overlap"] is None
        assert result["n_train_rows_purged_overlap"] is None
        assert all(
            r["purge"] == "not_applicable" for r in report["hyperparameter_search"]
        )

    def test_a_built_dataset_purges_on_its_label_end(self, patched_multi_factory):
        dataset = build_dataset(_dataset_spec())
        # Embargo 0, so the five-bar labels of the last training rows reach
        # into the test window and there is something to purge.
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=0),
            random_seed=1,
        )
        result = run_experiment(dataset, spec, "ds", register=False)
        report = result["validation_report"]
        assert report["purge"] == "label_end"
        assert isinstance(report["n_train_rows_purged_overlap"], int)
        assert report["n_train_rows_purged_overlap"] > 0


def _external_panel(path, n_dates: int = 200, entities=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(6)
    index = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    frames = []
    for entity in entities:
        alpha = rng.normal(0, 1, n_dates)
        frames.append(
            pd.DataFrame(
                {
                    "date": index,
                    "entity": entity,
                    "alpha": alpha,
                    "noise": rng.normal(0, 1, n_dates),
                    "target": 0.004 * alpha + rng.normal(0, 0.002, n_dates),
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return str(path)


class TestAnExternalPanelGetsItsLabelEndFromTheHorizon:
    """Two docstrings said the horizon was purging; the purge read only a
    column that registration never wrote."""

    def test_the_label_end_is_horizon_rows_ahead_per_entity(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        path = _external_panel(tmp_path / "panel.parquet")
        registered = register_external_panel(
            RegisterExternalPanelInput(path=path, horizon=5)
        )
        panel, _meta, _dir = _load_dataset_panel(registered.dataset_id)
        assert "label_end_date" in panel.columns
        one = panel[panel["entity"] == "AAA"].sort_values("date")
        expected = one["date"].shift(-5)
        pd.testing.assert_series_equal(
            one["label_end_date"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )
        assert one["label_end_date"].isna().sum() == 5

    def test_and_the_engine_purges_on_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        path = _external_panel(tmp_path / "panel.parquet")
        registered = register_external_panel(
            RegisterExternalPanelInput(path=path, horizon=5)
        )
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=60, test_window=20, embargo=0),
            random_seed=1,
        )
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=registered.dataset_id, spec=spec)
        )
        assert result.validation_report["purge"] == "label_end"
        assert result.n_train_rows_purged_overlap > 0

    def test_a_declared_label_end_column_is_kept_as_given(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        frame = pd.read_parquet(_external_panel(tmp_path / "raw.parquet"))
        frame["ends"] = frame["date"] + pd.Timedelta(days=3)
        path = tmp_path / "panel.parquet"
        frame.to_parquet(path, index=False)
        registered = register_external_panel(
            RegisterExternalPanelInput(
                path=str(path), horizon=5, label_end_column="ends"
            )
        )
        panel, _meta, _dir = _load_dataset_panel(registered.dataset_id)
        assert (panel["label_end_date"] - panel["date"]).eq(pd.Timedelta(days=3)).all()


# ── ties ─────────────────────────────────────────────────────────────────


class TestTiesAreTies:
    def test_identical_models_tie_on_every_date(self):
        rng = np.random.default_rng(7)
        rows = [
            {
                "date": d,
                "entity": f"E{j}",
                "prediction": rng.normal(),
                "target": rng.normal(),
            }
            for d in pd.bdate_range("2022-01-03", periods=40)
            for j in range(6)
        ]
        frame = pd.DataFrame(rows)
        result = paired_comparison(frame, frame, task="regression", n_bootstrap=100)
        assert result["n_ties"] == result["n_dates"]
        assert result["n_a_better"] == result["n_b_better"] == 0
        assert np.isnan(result["hit_rate"])
        assert result["verdict"] == "indistinguishable"

    def test_a_decided_comparison_excludes_ties_from_the_rate(self):
        rng = np.random.default_rng(8)
        dates = pd.bdate_range("2022-01-03", periods=40)
        rows = [
            {
                "date": d,
                "entity": f"E{j}",
                "prediction": rng.normal(),
                "target": rng.normal(),
            }
            for d in dates
            for j in range(6)
        ]
        a = pd.DataFrame(rows)
        b = a.copy()
        # The candidate is the truth on the last 20 dates and identical on the first 20.
        late = b["date"] >= dates[20]
        b.loc[late, "prediction"] = b.loc[late, "target"]
        result = paired_comparison(a, b, task="regression", n_bootstrap=100)
        assert result["n_ties"] == 20
        assert result["n_b_better"] == 20
        assert result["hit_rate"] == 1.0
