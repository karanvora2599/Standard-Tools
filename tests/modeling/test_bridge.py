"""Tests for modeling.bridge.oos_predictions_to_signal_panel: pivot +
sign-conversion logic against hand-built prediction data, plus a full
integration test running the real
run_model_experiment -> bridge -> run_signal_panel_backtest chain."""

from pathlib import Path

import pandas as pd
import pytest

from standard_quant_tools.agent.models import SignalPanelBacktestInput, SignalType
from standard_quant_tools.agent.tools import run_signal_panel_backtest
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _save_predictions(rows) -> str:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return _artifacts.save_artifact(
        df, run_id="mdl_bridge_test", name="oos_predictions"
    )


class TestRegressionSignConversion:
    def test_positive_prediction_is_long(self):
        uri = _save_predictions(
            [{"date": "2024-01-01", "entity": "AAA", "prediction": 0.02}]
        )
        panel = oos_predictions_to_signal_panel(uri, task="regression")
        assert panel == {"AAA": {"2024-01-01": 1.0}}

    def test_negative_prediction_is_short(self):
        uri = _save_predictions(
            [{"date": "2024-01-01", "entity": "AAA", "prediction": -0.015}]
        )
        panel = oos_predictions_to_signal_panel(uri, task="regression")
        assert panel == {"AAA": {"2024-01-01": -1.0}}

    def test_deadband_flattens_small_predictions(self):
        uri = _save_predictions(
            [
                {"date": "2024-01-01", "entity": "AAA", "prediction": 0.001},
                {"date": "2024-01-01", "entity": "BBB", "prediction": 0.02},
            ]
        )
        panel = oos_predictions_to_signal_panel(uri, task="regression", deadband=0.005)
        assert panel["AAA"]["2024-01-01"] == 0.0
        assert panel["BBB"]["2024-01-01"] == 1.0

    def test_negative_deadband_rejected(self):
        uri = _save_predictions(
            [{"date": "2024-01-01", "entity": "AAA", "prediction": 0.0}]
        )
        with pytest.raises(ValidationError, match="deadband"):
            oos_predictions_to_signal_panel(uri, task="regression", deadband=-1.0)

    def test_multi_entity_multi_date_pivots_correctly(self):
        uri = _save_predictions(
            [
                {"date": "2024-01-01", "entity": "AAA", "prediction": 0.01},
                {"date": "2024-01-02", "entity": "AAA", "prediction": -0.01},
                {"date": "2024-01-01", "entity": "BBB", "prediction": -0.02},
            ]
        )
        panel = oos_predictions_to_signal_panel(uri, task="regression")
        assert panel == {
            "AAA": {"2024-01-01": 1.0, "2024-01-02": -1.0},
            "BBB": {"2024-01-01": -1.0},
        }


class TestClassificationThreshold:
    def test_long_only_default(self):
        uri = _save_predictions(
            [
                {"date": "2024-01-01", "entity": "AAA", "prediction": 0.8},
                {"date": "2024-01-01", "entity": "BBB", "prediction": 0.2},
            ]
        )
        panel = oos_predictions_to_signal_panel(uri, task="classification")
        assert panel["AAA"]["2024-01-01"] == 1.0
        assert panel["BBB"]["2024-01-01"] == 0.0  # not -1.0 -- long_only default

    def test_symmetric_mode(self):
        uri = _save_predictions(
            [
                {"date": "2024-01-01", "entity": "AAA", "prediction": 0.8},
                {"date": "2024-01-01", "entity": "BBB", "prediction": 0.15},
                {"date": "2024-01-01", "entity": "CCC", "prediction": 0.5},
            ]
        )
        panel = oos_predictions_to_signal_panel(
            uri, task="classification", proba_threshold=0.7, long_only=False
        )
        assert panel["AAA"]["2024-01-01"] == 1.0
        assert panel["BBB"]["2024-01-01"] == -1.0
        assert panel["CCC"]["2024-01-01"] == 0.0

    def test_proba_threshold_out_of_range_rejected(self):
        uri = _save_predictions(
            [{"date": "2024-01-01", "entity": "AAA", "prediction": 0.5}]
        )
        with pytest.raises(ValidationError, match="proba_threshold"):
            oos_predictions_to_signal_panel(
                uri, task="classification", proba_threshold=1.5
            )

    def test_symmetric_mode_below_midpoint_threshold_rejected(self):
        uri = _save_predictions(
            [{"date": "2024-01-01", "entity": "AAA", "prediction": 0.5}]
        )
        with pytest.raises(ValidationError, match="0.5"):
            oos_predictions_to_signal_panel(
                uri, task="classification", proba_threshold=0.3, long_only=False
            )


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


class TestFullBridgeIntegration:
    def test_run_model_experiment_to_signal_panel_backtest(self, patched_multi_factory):
        """The real chain this bridge exists for: train a model, turn its
        walk-forward OOS predictions into a signal panel, and actually
        backtest it via the existing (46-tool registry) backtest tool --
        proving the two registries connect through artifacts, not a new
        tool call, and that the whole thing runs end to end without error."""
        built = build_dataset(_dataset_spec())
        model_spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            random_seed=1,
        )
        dataset = {
            "panel": built["panel"],
            "feature_ids": built["feature_ids"],
            "target_id": built["target_id"],
            "data_hash": built["data_hash"],
        }
        exp_result = run_experiment(dataset, model_spec, dataset_id="ds_bridge_test")
        assert exp_result["oos_predictions_uri"]
        assert Path(exp_result["oos_predictions_uri"]).exists()

        # model_id, not (uri, task): the manifest resolves both together so
        # they cannot disagree. Passing task by hand allows regression
        # predictions to be thresholded as classification probabilities,
        # which yields a nonsensical but valid-looking panel.
        signal_panel = oos_predictions_to_signal_panel(model_id=exp_result["model_id"])
        assert set(signal_panel.keys()) <= {"AAA", "BBB", "CCC"}
        all_values = {v for dates in signal_panel.values() for v in dates.values()}
        assert all_values <= {-1.0, 0.0, 1.0}

        backtest_start = built["panel"]["date"].min().strftime("%Y-%m-%d")
        backtest_end = built["panel"]["date"].max().strftime("%Y-%m-%d")
        result = run_signal_panel_backtest(
            SignalPanelBacktestInput(
                tickers=list(signal_panel.keys()),
                start_date=backtest_start,
                end_date=backtest_end,
                signal_panel=signal_panel,
                signal_type=SignalType.DIRECTION,
                # next_open, not the "close" default. Modeling features are
                # computed from bar t's own OHLC, so a signal dated t is not
                # knowable until t's close has printed -- filling it at that
                # same close is the look-ahead run_strategy's own fill_price
                # warning describes.
                fill_price="next_open",
            )
        )
        assert set(result.per_ticker.keys()) == set(signal_panel.keys())
        assert "sharpe_ratio" in result.portfolio_metrics
