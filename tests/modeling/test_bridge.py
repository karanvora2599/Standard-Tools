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
        """
        Every entity is densified onto the panel's shared calendar, flat
        (0.0) where it has no prediction of its own.

        This assertion previously expected BBB to simply be ABSENT on
        2024-01-02. That looked like a faithful pivot, but
        run_signal_panel_backtest runs run_strategy per ticker against that
        ticker's own signal series, and run_strategy intersects prices down
        to the signal index before taking pct_change — so BBB's price axis
        would have been compressed relative to AAA's, silently, for a
        reason unrelated to BBB's prices.

        0.0 is the honest fill: the model expressed no view for BBB that
        day, and DIRECTION's 0.0 means exactly "flat".
        """
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
            "BBB": {"2024-01-01": -1.0, "2024-01-02": 0.0},
        }
        # Both entities must span the same calendar, or the per-ticker
        # backtests are not run over the same date axis.
        assert set(panel["AAA"]) == set(panel["BBB"])


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


class TestCalendarContinuity:
    """
    `run_strategy` intersects price data down to the supplied signal index
    and then takes `.pct_change()` over what REMAINS. A span with no
    predictions therefore does not read as "flat" — it disappears from the
    price axis, and the bars either side become adjacent.

    Measured directly below: with February absent from a 90-day series, the
    Jan->Mar boundary bar carries ~26x a normal daily return, i.e. a month
    of price movement compressed into a single bar. Total return can still
    look right while volatility, Sharpe and drawdown are all distorted,
    which is why this was easy to miss.
    """

    def test_run_strategy_really_does_compress_a_gap(self):
        """The underlying behavior this guard exists for — pinned so the
        guard isn't mistaken for excess caution."""
        import numpy as np

        from standard_quant_tools.backtest.engine import run_strategy

        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        df = pd.DataFrame({"Close": np.linspace(100, 190, 90)}, index=idx)
        kept = idx[(idx < "2024-02-01") | (idx >= "2024-03-01")]

        prices = df.loc[df.index.intersection(kept), "Close"]
        returns = prices.pct_change()
        boundary = float(returns.loc["2024-03-01"])
        normal = float(returns.loc["2024-01-15"])
        assert boundary > 10 * normal, (
            "a gap must compress adjacent bars — if this stops being true the "
            "continuity guard can be revisited"
        )
        # And the compressed run really does see fewer bars.
        assert len(run_strategy(df, pd.Series(1.0, index=kept))["equity_curve"]) < 90

    def test_discontinuous_artifact_is_rejected(self):
        """A month-long hole is far beyond any holiday cluster."""
        rows = [
            {"date": d.strftime("%Y-%m-%d"), "entity": "AAA", "prediction": 0.01}
            for d in pd.bdate_range("2024-01-01", "2024-01-31")
        ] + [
            {"date": d.strftime("%Y-%m-%d"), "entity": "AAA", "prediction": 0.01}
            for d in pd.bdate_range("2024-03-01", "2024-03-31")
        ]
        uri = _save_predictions(rows)
        with pytest.raises(ValidationError, match="discontinuous"):
            oos_predictions_to_signal_panel(uri, task="regression")

    def test_ordinary_weekends_and_holidays_are_not_rejected(self):
        """The guard must not fire on a normal business-day calendar —
        weekends and a public holiday are gaps too, just small ones."""
        rows = [
            {"date": d.strftime("%Y-%m-%d"), "entity": "AAA", "prediction": 0.01}
            for d in pd.bdate_range("2024-01-01", "2024-03-31")
        ]
        panel = oos_predictions_to_signal_panel(
            _save_predictions(rows), task="regression"
        )
        assert len(panel["AAA"]) == len(pd.bdate_range("2024-01-01", "2024-03-31"))

    def test_entity_level_gap_is_filled_flat_not_dropped(self):
        """
        Repairable, unlike a skipped fold: the date exists in the artifact,
        just not for this entity. Leaving the hole would compress that one
        symbol's price axis while its peers kept the full calendar.
        """
        rows = []
        for d in pd.bdate_range("2024-01-01", "2024-01-31"):
            rows.append(
                {"date": d.strftime("%Y-%m-%d"), "entity": "AAA", "prediction": 0.01}
            )
            # BBB is missing every Wednesday.
            if d.weekday() != 2:
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "entity": "BBB",
                        "prediction": 0.02,
                    }
                )
        panel = oos_predictions_to_signal_panel(
            _save_predictions(rows), task="regression"
        )

        assert set(panel["AAA"]) == set(
            panel["BBB"]
        ), "entities span different calendars"
        wednesdays = [
            d.strftime("%Y-%m-%d")
            for d in pd.bdate_range("2024-01-01", "2024-01-31")
            if d.weekday() == 2
        ]
        assert wednesdays, "fixture must actually contain the missing days"
        for day in wednesdays:
            assert (
                panel["BBB"][day] == 0.0
            ), "a missing prediction must be flat, not absent"
