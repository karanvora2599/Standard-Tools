"""
Tests for modeling.portfolio_eval: the OOS-predictions -> target-weights
-> shared-cash-simulation path.

Split into (a) unit tests over the weight construction, which is pure
arithmetic and where the exposure/cap invariants either hold exactly or
do not, and (b) an end-to-end test running the real
build_dataset -> run_experiment -> evaluate_model_portfolio chain against
the synthetic multi-symbol provider.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    EvaluateModelPortfolioInput,
    RunModelExperimentInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent import (
    evaluate_model_portfolio as evaluate_model_portfolio_tool,
)
from standard_quant_tools.modeling.agent import (
    modeling_dispatch,
    run_model_experiment,
)
from standard_quant_tools.modeling.portfolio_eval import (
    apply_exposure_targets,
    evaluate_model_portfolio,
    predictions_to_score_panel,
    select_rebalance_dates,
    transform_predictions_to_weights,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PortfolioSimSpec,
    PredictionTransformSpec,
    TargetSpec,
    ValidationSpec,
)

# ── Helpers ─────────────────────────────────────────────────────────────


def _score_panel(n_dates: int = 10, n_entities: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    entities = [f"S{i:02d}" for i in range(n_entities)]
    return pd.DataFrame(
        rng.normal(0, 0.02, (n_dates, n_entities)), index=dates, columns=entities
    )


def _long_predictions(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.stack()
        .rename("prediction")
        .rename_axis(index=["date", "entity"])
        .reset_index()
    )


# ── Exposure targeting ──────────────────────────────────────────────────


class TestExposureTargets:
    def test_gross_and_net_are_both_hit_exactly(self):
        """The whole reason for the two-book split: a single rescale can
        control gross OR net, never both."""
        scores = np.array([3.0, 1.0, -1.0, -2.0, 0.5])
        weights, diag = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=1.0
        )
        assert np.isclose(np.abs(weights).sum(), 1.0)
        assert np.isclose(weights.sum(), 0.0)
        assert diag["n_long"] == 3
        assert diag["n_short"] == 2

    @pytest.mark.parametrize("net", [-0.4, 0.0, 0.3, 0.6])
    def test_arbitrary_net_target_is_hit(self, net):
        scores = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
        weights, _ = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=net, max_position_weight=1.0
        )
        assert np.isclose(np.abs(weights).sum(), 1.0)
        assert np.isclose(weights.sum(), net)

    def test_long_only_when_net_equals_gross(self):
        scores = np.array([2.0, 1.0, -1.0, -3.0])
        weights, _ = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=1.0, max_position_weight=1.0
        )
        assert (weights >= 0).all()
        assert np.isclose(weights.sum(), 1.0)
        # The bearish names get exactly zero, not a small short.
        assert weights[2] == 0.0 and weights[3] == 0.0

    def test_nan_entities_get_exactly_zero_weight(self):
        scores = np.array([2.0, np.nan, -1.0, np.nan])
        weights, diag = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=1.0
        )
        assert weights[1] == 0.0 and weights[3] == 0.0
        assert diag["n_long"] == 1 and diag["n_short"] == 1

    def test_all_zero_scores_produce_no_position(self):
        weights, diag = apply_exposure_targets(
            np.zeros(5), gross_exposure=1.0, net_exposure=0.0, max_position_weight=1.0
        )
        assert (weights == 0.0).all()
        assert diag["realized_gross"] == 0.0


class TestPositionCap:
    def test_cap_is_never_exceeded(self):
        scores = np.array([100.0, 1.0, 1.0, 1.0, -100.0, -1.0, -1.0, -1.0])
        weights, _ = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=0.2
        )
        assert np.abs(weights).max() <= 0.2 + 1e-9

    def test_capping_redistributes_rather_than_shrinking_gross(self):
        """A cap that merely truncated would silently deliver less gross
        exposure than requested — the portfolio would sit in cash without
        anything saying so."""
        scores = np.array([100.0, 1.0, 1.0, 1.0, -100.0, -1.0, -1.0, -1.0])
        weights, _ = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=0.2
        )
        assert np.isclose(np.abs(weights).sum(), 1.0)
        assert np.isclose(weights.sum(), 0.0)

    def test_cascading_cap_terminates_and_still_hits_gross(self):
        """Redistribution can push a previously-uncapped name over the cap,
        which must then be capped in turn."""
        scores = np.array([50.0, 40.0, 30.0, 1.0, 1.0, -50.0, -40.0, -30.0, -1.0, -1.0])
        weights, _ = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=0.15
        )
        assert np.abs(weights).max() <= 0.15 + 1e-9
        assert np.isclose(np.abs(weights).sum(), 1.0)

    def test_infeasible_book_reports_shortfall_instead_of_breaching_cap(self):
        """Two long names cannot hold 0.5 gross at a 0.1 cap. The honest
        outcome is a smaller book, reported — not a breached risk limit."""
        scores = np.array([2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        weights, diag = apply_exposure_targets(
            scores, gross_exposure=1.0, net_exposure=0.0, max_position_weight=0.1
        )
        assert np.abs(weights).max() <= 0.1 + 1e-9
        assert diag["realized_long_gross"] == pytest.approx(0.2)
        assert diag["realized_gross"] < 1.0


# ── Score panel ─────────────────────────────────────────────────────────


class TestScorePanel:
    def test_long_to_wide_roundtrip(self):
        panel = _score_panel(n_dates=4, n_entities=3)
        rebuilt = predictions_to_score_panel(_long_predictions(panel), "regression")
        # check_freq=False: the source panel's index carries a BusinessDay
        # freq that a pivot cannot infer. Irrelevant to the values.
        pd.testing.assert_frame_equal(
            rebuilt, panel, check_names=False, check_freq=False
        )

    def test_classification_probabilities_are_recentred(self):
        """A raw probability is in [0, 1], so sign() is +1 for every name —
        a 'long everything' portfolio dressed up as a signal."""
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

    def test_missing_pairs_stay_nan(self):
        long_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "entity": ["AAA", "BBB", "AAA"],
                "prediction": [0.01, 0.02, 0.03],
            }
        )
        panel = predictions_to_score_panel(long_df, "regression")
        assert np.isnan(panel.loc[pd.Timestamp("2024-01-02"), "BBB"])

    def test_bad_task_rejected(self):
        with pytest.raises(ValidationError, match="regression"):
            predictions_to_score_panel(_long_predictions(_score_panel(2, 2)), "banana")


# ── Rebalance schedule ──────────────────────────────────────────────────


class TestRebalanceSchedule:
    def test_daily_keeps_every_date(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        assert len(select_rebalance_dates(dates, "daily")) == 30

    def test_weekly_takes_first_date_of_each_week(self):
        dates = pd.date_range("2024-01-01", periods=15, freq="B")  # 3 weeks
        picked = select_rebalance_dates(dates, "weekly")
        assert len(picked) == 3
        # Mondays: first business day of each ISO week in the range.
        assert list(picked) == [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-01-15"),
        ]

    def test_monthly_takes_first_date_of_each_month(self):
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        picked = select_rebalance_dates(dates, "monthly")
        assert (picked == picked.to_series().groupby(picked.to_period("M")).min()).all()

    def test_schedule_never_depends_on_a_later_date(self):
        """First-of-period, not last: truncating the calendar must not
        change any rebalance date that was already selected."""
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        full = select_rebalance_dates(dates, "weekly")
        truncated = select_rebalance_dates(dates[:25], "weekly")
        assert list(truncated) == list(full[: len(truncated)])

    def test_unknown_frequency_rejected(self):
        with pytest.raises(ValidationError, match="daily/weekly/monthly"):
            select_rebalance_dates(pd.date_range("2024-01-01", periods=3), "hourly")


# ── Transform ───────────────────────────────────────────────────────────


class TestTransform:
    @pytest.mark.parametrize(
        "method",
        [
            "sign",
            "cross_sectional_rank",
            "cross_sectional_zscore",
            "top_bottom_quantile",
        ],
    )
    def test_every_method_hits_the_exposure_targets(self, method):
        panel = _score_panel(n_dates=8, n_entities=20, seed=1)
        spec = PredictionTransformSpec(
            method=method, gross_exposure=1.0, net_exposure=0.0, max_position_weight=0.5
        )
        weights, diag = transform_predictions_to_weights(panel, spec)
        assert np.allclose(weights.abs().sum(axis=1), 1.0)
        assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-9)
        assert diag["n_dates_below_target_gross"] == 0

    def test_rank_method_orders_weights_by_prediction(self):
        panel = pd.DataFrame(
            [[0.05, 0.01, -0.01, -0.05]],
            index=pd.to_datetime(["2024-01-01"]),
            columns=["A", "B", "C", "D"],
        )
        spec = PredictionTransformSpec(
            method="cross_sectional_rank", max_position_weight=1.0
        )
        weights, _ = transform_predictions_to_weights(panel, spec)
        row = weights.iloc[0]
        assert row["A"] > row["B"] > 0 > row["C"] > row["D"]

    def test_magnitude_survives_unlike_the_signal_panel_bridge(self):
        """The gap this module exists to close: the bridge maps both 0.05
        and 0.01 to +1.0, so a strong and a marginal prediction get the
        same position."""
        panel = pd.DataFrame(
            [[0.05, 0.01, -0.01, -0.05]],
            index=pd.to_datetime(["2024-01-01"]),
            columns=["A", "B", "C", "D"],
        )
        spec = PredictionTransformSpec(
            method="cross_sectional_zscore", max_position_weight=1.0
        )
        weights, _ = transform_predictions_to_weights(panel, spec)
        assert weights.iloc[0]["A"] > weights.iloc[0]["B"] * 2

    def test_top_bottom_quantile_holds_only_the_tails(self):
        panel = _score_panel(n_dates=3, n_entities=10, seed=2)
        spec = PredictionTransformSpec(
            method="top_bottom_quantile",
            long_quantile=0.2,
            short_quantile=0.2,
            max_position_weight=1.0,
        )
        weights, diag = transform_predictions_to_weights(panel, spec)
        assert diag["mean_n_long"] == 2.0
        assert diag["mean_n_short"] == 2.0
        assert (weights != 0).sum(axis=1).eq(4).all()

    def test_sparse_cross_sections_are_weighted_per_date(self):
        """A date where an entity has no prediction must weight the names
        that ARE present, not drop the date and shorten the track record."""
        panel = _score_panel(n_dates=6, n_entities=5, seed=3)
        panel.iloc[2, 0] = np.nan
        panel.iloc[4, 1] = np.nan
        spec = PredictionTransformSpec(max_position_weight=1.0)
        weights, diag = transform_predictions_to_weights(panel, spec)
        assert weights.iloc[2, 0] == 0.0
        assert weights.iloc[4, 1] == 0.0
        assert np.allclose(weights.abs().sum(axis=1), 1.0)
        assert diag["n_dates"] == 6
        assert diag["min_names_per_date"] == 4

    def test_single_name_cross_section_is_left_flat(self):
        """One name is not a cross-section — allocating 100% to it would
        be acting on a comparison that was never made."""
        panel = pd.DataFrame(
            [[0.05, np.nan, np.nan]],
            index=pd.to_datetime(["2024-01-01"]),
            columns=["A", "B", "C"],
        )
        weights, diag = transform_predictions_to_weights(
            panel, PredictionTransformSpec()
        )
        assert (weights.iloc[0] == 0.0).all()
        assert diag["n_dates_with_no_position"] == 1

    def test_position_cap_is_respected_across_the_panel(self):
        panel = _score_panel(n_dates=5, n_entities=40, seed=4)
        spec = PredictionTransformSpec(max_position_weight=0.05)
        weights, diag = transform_predictions_to_weights(panel, spec)
        assert weights.abs().to_numpy().max() <= 0.05 + 1e-9
        assert diag["max_abs_weight"] <= 0.05 + 1e-9

    def test_each_date_is_transformed_independently(self):
        """Point-in-time by construction: appending future dates must not
        change any earlier date's weights."""
        panel = _score_panel(n_dates=10, n_entities=8, seed=5)
        spec = PredictionTransformSpec(max_position_weight=1.0)
        full, _ = transform_predictions_to_weights(panel, spec)
        partial, _ = transform_predictions_to_weights(panel.iloc[:5], spec)
        pd.testing.assert_frame_equal(full.iloc[:5], partial)

    def test_volatility_scaling_shrinks_the_high_vol_name(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        panel = pd.DataFrame(
            [[0.02, 0.02, -0.02, -0.02]] * 60,
            index=dates,
            columns=["CALM", "WILD", "C", "D"],
        )
        rng = np.random.default_rng(9)
        returns = pd.DataFrame(
            {
                "CALM": rng.normal(0, 0.001, 60),
                "WILD": rng.normal(0, 0.05, 60),
                "C": rng.normal(0, 0.01, 60),
                "D": rng.normal(0, 0.01, 60),
            },
            index=dates,
        )
        spec = PredictionTransformSpec(
            method="cross_sectional_zscore",
            volatility_scale=True,
            volatility_lookback=20,
            max_position_weight=1.0,
        )
        weights, _ = transform_predictions_to_weights(panel, spec, returns)
        last = weights.iloc[-1]
        assert last["CALM"] > last["WILD"]

    def test_empty_panel_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            transform_predictions_to_weights(pd.DataFrame(), PredictionTransformSpec())


# ── Spec validation ─────────────────────────────────────────────────────


class TestSpecValidation:
    def test_net_beyond_gross_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            PredictionTransformSpec(gross_exposure=1.0, net_exposure=1.5)

    def test_overlapping_quantiles_rejected(self):
        with pytest.raises(ValueError, match="overlap"):
            PredictionTransformSpec(
                method="top_bottom_quantile", long_quantile=0.7, short_quantile=0.6
            )

    def test_no_shorts_with_a_short_book_target_rejected(self):
        """short_quantile=0 selects nothing to short, so a dollar-neutral
        target is unsatisfiable — better rejected than silently half-sized."""
        with pytest.raises(ValueError, match="long-only"):
            PredictionTransformSpec(
                method="top_bottom_quantile",
                short_quantile=0.0,
                gross_exposure=1.0,
                net_exposure=0.0,
            )

    def test_long_only_quantile_accepted(self):
        spec = PredictionTransformSpec(
            method="top_bottom_quantile",
            short_quantile=0.0,
            gross_exposure=1.0,
            net_exposure=1.0,
        )
        assert spec.short_quantile == 0.0


# ── End-to-end ──────────────────────────────────────────────────────────


UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def registered_model(patched_multi_factory):
    """Trained through the AGENT TOOLS, not build_dataset/run_experiment
    directly: only the tool path persists the dataset_spec.json that
    evaluate_model_portfolio reads back (for the provider and interval it
    must fetch prices with). Going through the tools is also the path a
    real caller takes, so the fixture exercises the same wiring."""
    dataset = build_model_dataset(
        BuildModelDatasetInput(
            spec=DatasetSpec(
                universe=UNIVERSE,
                start="2022-01-01",
                end="2023-12-31",
                features=[
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                    FeatureSpec(id="technical.rsi", params={"period": 14}),
                ],
                target=TargetSpec(type="forward_return", horizon=5),
            )
        )
    )
    experiment = run_model_experiment(
        RunModelExperimentInput(
            dataset_id=dataset.dataset_id,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(
                    train_window=150, test_window=40, embargo=5, min_folds=2
                ),
                random_seed=1,
            ),
        )
    )
    return experiment.model_id


class TestEndToEnd:
    def test_evaluation_returns_economic_metrics(self, registered_model):
        result = evaluate_model_portfolio(
            registered_model,
            PredictionTransformSpec(
                method="cross_sectional_rank",
                max_position_weight=0.5,
                rebalance_frequency="weekly",
            ),
        )
        assert result["model_id"] == registered_model
        for key in (
            "sharpe_ratio",
            "max_drawdown",
            "cagr",
            "mean_turnover_pct",
            "mean_gross_exposure",
            "estimated_cost_drag_pct",
        ):
            assert key in result["metrics"]
        assert result["coverage"]["n_entities"] == len(UNIVERSE)
        assert result["coverage"]["n_rebalance_dates"] >= 2
        assert np.isfinite(result["metrics"]["sharpe_ratio"])

    def test_weights_artifact_is_persisted_and_reloadable(self, registered_model):
        result = evaluate_model_portfolio(registered_model)
        weights = _artifacts.load_artifact(result["target_weights_uri"])
        assert not weights.empty
        assert set(weights.columns) <= set(UNIVERSE)
        assert weights.abs().to_numpy().max() <= 0.05 + 1e-9

    def test_provenance_records_every_hash(self, registered_model):
        result = evaluate_model_portfolio(registered_model)
        provenance = result["provenance"]
        assert provenance["oos_predictions_hash"]
        assert provenance["target_weights_hash"]
        assert provenance["equity_curve_hash"]
        assert provenance["transform_spec"]["method"] == "cross_sectional_rank"
        assert provenance["portfolio_spec"]["fill_price"] == "next_open"

    def test_same_inputs_reproduce_the_same_weights(self, registered_model):
        first = evaluate_model_portfolio(registered_model)
        second = evaluate_model_portfolio(registered_model)
        assert (
            first["provenance"]["target_weights_hash"]
            == second["provenance"]["target_weights_hash"]
        )
        assert first["metrics"]["sharpe_ratio"] == second["metrics"]["sharpe_ratio"]

    def test_different_transform_writes_a_different_artifact(self, registered_model):
        rank = evaluate_model_portfolio(
            registered_model, PredictionTransformSpec(method="cross_sectional_rank")
        )
        quantile = evaluate_model_portfolio(
            registered_model, PredictionTransformSpec(method="top_bottom_quantile")
        )
        assert rank["target_weights_uri"] != quantile["target_weights_uri"]

    def test_close_fill_is_flagged_as_lookahead(self, registered_model):
        result = evaluate_model_portfolio(
            registered_model, portfolio=PortfolioSimSpec(fill_price="close")
        )
        assert any("look-ahead" in w for w in result["warnings"])

    def test_dataset_warnings_travel_onto_the_evaluation(self, registered_model):
        """A survivors-only universe changes how a Sharpe should be read,
        and this is where someone reads it."""
        result = evaluate_model_portfolio(registered_model)
        assert any("survivorship" in w.lower() for w in result["warnings"])

    def test_gross_above_leverage_limit_rejected_before_simulating(
        self, registered_model
    ):
        with pytest.raises(ValidationError, match="max_gross_leverage"):
            evaluate_model_portfolio(
                registered_model,
                PredictionTransformSpec(gross_exposure=2.0),
                PortfolioSimSpec(max_gross_leverage=1.0),
            )

    def test_position_cap_above_simulator_limit_rejected(self, registered_model):
        with pytest.raises(ValidationError, match="max_position_pct"):
            evaluate_model_portfolio(
                registered_model,
                PredictionTransformSpec(max_position_weight=0.9),
                PortfolioSimSpec(max_position_pct=0.5),
            )

    def test_tampered_predictions_artifact_is_rejected(self, registered_model):
        """Structural validation passes on an edited file that kept its
        shape, so without the digest check a rewritten prediction column
        would produce a clean and entirely fictional equity curve."""
        from standard_quant_tools.modeling.registry.model_registry import load_manifest

        uri = load_manifest(registered_model).oos_predictions_uri
        tampered = _artifacts.load_artifact(uri)
        tampered["prediction"] = tampered["prediction"] * -1.0
        tampered.to_parquet(uri)
        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            evaluate_model_portfolio(registered_model)


class TestAgentSurface:
    def test_tool_wrapper_returns_the_result_model(self, registered_model):
        result = evaluate_model_portfolio_tool(
            EvaluateModelPortfolioInput(model_id=registered_model)
        )
        assert result.model_id == registered_model
        assert "sharpe_ratio" in result.metrics
        assert result.target_weights_uri

    def test_dispatch_returns_json_safe_dict(self, registered_model):
        import json

        payload = modeling_dispatch(
            "evaluate_model_portfolio",
            {
                "model_id": registered_model,
                "transform": {"method": "cross_sectional_rank"},
            },
        )
        # allow_nan=False is the point: a Calmar of inf or a NaN metric must
        # already have been sanitized, not left for a strict parser to reject.
        json.dumps(payload, allow_nan=False)
        assert payload["model_id"] == registered_model
