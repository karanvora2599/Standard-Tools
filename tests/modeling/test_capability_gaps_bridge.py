"""
The verified branch of the model -> backtest bridge is the one the tool
surface exposes, and this library's own predictions can be scored against
their outcomes. See the CHANGELOG entry of 2026-09-21.

`modeling/bridge.py` has always had two branches. The `model_id` one reads
the task from the manifest, refuses a combinatorial-CV model by name, and
verifies the predictions file against the digest recorded at registration.
The `oos_predictions_uri` one checks that `task` is a task. Until now only
the second was reachable from a tool -- through the published COPY of the
predictions, which no manifest covers -- so the agent's only route from a
model to a backtest was the one the bridge's own docstring calls
"explicitly unverified".

Planted here:

  * `backtest_model_signal` publishes a DIRECTION panel whose values are
    in {-1, 0, 1} and which `run_signal_panel_backtest` prices to a finite
    Sharpe;
  * a shape-preserving sign flip of the registered predictions makes it
    raise "has changed since it was registered", and `convert_reference`
    accepts the same tampering -- the contrast is the point, so both
    halves are asserted;
  * cpcv is refused with the walk-forward remedy; `task=` is not
    expressible; a venue-qualified universe points at
    `evaluate_model_portfolio`;
  * `attach_model_outcomes` closes the loop that fails today with "the
    predictions frame has no 'target' column", for a model, for an
    ensemble reference, and for a scored universe;
  * a two-horizon dataset with no label named is REFUSED rather than
    silently scored against the primary;
  * `score_model` publishes a reference, `handoff.resolve` agrees with
    `load_artifact`, and re-scoring the same universe twice does not
    collide.

`tests/modeling/test_bridge.py`'s library-level URI branch stays untouched:
it is deliberately unverified at that level, and this file is about which
branch the TOOL surface exposes.
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import (
    ConvertReferenceInput,
    SignalPanelBacktestInput,
    SignalType,
)
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.meta.tools import convert_reference
from standard_quant_tools.agent.tools import run_signal_panel_backtest
from standard_quant_tools.backtest.artifacts import load_artifact
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import (
    AttachModelOutcomesInput,
    BacktestModelSignalInput,
    BuildEnsembleInput,
    BuildModelDatasetInput,
    RunModelExperimentInput,
    ScoreModelInput,
    ScorePredictionsInput,
)
from standard_quant_tools.modeling.agent.tools import (
    attach_model_outcomes,
    backtest_model_signal,
    build_model_dataset,
    build_model_ensemble,
    run_model_experiment,
    score_model,
    score_predictions,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    resolve_model_artifact,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

from .test_scoring import _train_a_model_with_spec

UNIVERSE = ["AAA", "BBB", "CCC"]
QUALIFIED = ["AAA@XNYS", "BBB@XNYS", "CCC@XNYS"]


def _spec(**overrides) -> DatasetSpec:
    defaults = dict(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


def _model_spec(estimator: str = "ridge", **params) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(
            type=estimator, params=params or ({"alpha": 1.0} if estimator else {})
        ),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


def _cpcv_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(method="cpcv", n_splits=6, n_test_splits=2),
        random_seed=1,
    )


def _dataset(spec: "DatasetSpec | None" = None) -> str:
    return build_model_dataset(BuildModelDatasetInput(spec=spec or _spec())).dataset_id


def _trained(dataset_id: str, spec: "ModelSpec | None" = None):
    return run_model_experiment(
        RunModelExperimentInput(dataset_id=dataset_id, spec=spec or _model_spec())
    )


class TestTheVerifiedRouteToABacktest:
    """The branch built to make a wrong task and a tampered artifact
    impossible was the branch no tool could reach."""

    def test_the_published_panel_is_direction_valid_and_backtests(
        self, patched_multi_factory
    ):
        model = _trained(_dataset())
        result = backtest_model_signal(
            BacktestModelSignalInput(
                model_id=model.model_id, run_id="bridge_signal", name="panel"
            )
        )

        assert result.signal_panel_ref.startswith("sqt://signal_panel/")
        assert result.task == "regression"
        # Verified, not merely recorded: the manifest's digest is what the
        # bridge checked the file against before reading it.
        assert result.oos_predictions_hash
        assert (
            result.oos_predictions_hash
            == load_manifest(model.model_id).content_hashes["oos_predictions"]
        )
        assert any("next_open" in warning for warning in result.warnings)

        panel = handoff.resolve(result.signal_panel_ref, expect="signal_panel")
        values = [value for series in panel.values() for value in series.values()]
        assert set(values) <= {-1.0, 0.0, 1.0}
        assert sorted(panel) == result.entities
        # Every entity carries the whole calendar -- a hole would vanish
        # from the price axis rather than reading as "no position".
        assert {len(series) for series in panel.values()} == {result.n_dates}
        assert result.n_long + result.n_flat + result.n_short == len(values)

        backtested = run_signal_panel_backtest(
            SignalPanelBacktestInput(
                tickers=result.entities,
                start_date=result.first_date,
                end_date=result.last_date,
                signal_panel_ref=result.signal_panel_ref,
                signal_type=SignalType.DIRECTION,
                fill_price="next_open",
            )
        )
        assert set(backtested.per_ticker) == set(result.entities)
        assert math.isfinite(backtested.portfolio_metrics["sharpe_ratio"])

    def test_a_sign_flipped_artifact_is_refused_here_and_accepted_by_convert(
        self, patched_multi_factory, tmp_path
    ):
        """
        Both halves. `run_model_experiment` publishes a COPY of the
        predictions beside the artifact the manifest hashed, and the
        agent's existing route reads that copy -- so the same
        shape-preserving edit is a refusal on one path and silence on the
        other. Both are asserted, because the contrast is the finding.
        """
        model = _trained(_dataset())
        registered = resolve_model_artifact(
            model.model_id, load_manifest(model.model_id).oos_predictions_uri
        )
        published = tmp_path / "runs" / model.model_id / "oos_predictions_ref.parquet"
        assert published.exists(), "run_model_experiment publishes a copy"

        for path in (registered, published):
            frame = pd.read_parquet(path)
            # Same columns, same dtypes, same (entity, date) pairs, all
            # finite: every structural check still passes.
            frame["prediction"] = frame["prediction"] * -1.0
            frame.to_parquet(path)

        with pytest.raises(
            ValidationError, match="has changed since it was registered"
        ):
            backtest_model_signal(
                BacktestModelSignalInput(
                    model_id=model.model_id, run_id="bridge_tamper", name="panel"
                )
            )

        converted = convert_reference(
            ConvertReferenceInput(
                ref=model.oos_predictions_ref,
                to_kind="signal_panel",
                task="regression",
                run_id="bridge_tamper",
                name="converted",
            )
        )
        assert converted.ref.startswith("sqt://signal_panel/"), (
            "the unverified branch accepts the tampered copy without "
            "complaint -- which is why the verified one had to be reachable"
        )

    def test_a_cpcv_model_is_refused_with_the_walk_forward_remedy(
        self, patched_multi_factory
    ):
        model = _trained(_dataset(), _cpcv_spec())
        with pytest.raises(ValidationError, match="walk_forward") as excinfo:
            backtest_model_signal(
                BacktestModelSignalInput(
                    model_id=model.model_id, run_id="bridge_cpcv", name="panel"
                )
            )
        message = str(excinfo.value)
        assert "cpcv" in message and model.model_id in message

    def test_task_is_not_expressible(self):
        """No `task` field is the point: the manifest is the only source,
        so a mismatch cannot be spelled rather than merely discouraged."""
        assert "task" not in BacktestModelSignalInput.model_json_schema()["properties"]
        with pytest.raises(PydanticValidationError, match="task"):
            BacktestModelSignalInput(
                model_id="mdl_whatever",
                run_id="r",
                name="n",
                task="classification",
            )

    def test_venue_qualified_entities_point_at_the_portfolio_evaluator(
        self, patched_multi_factory
    ):
        """The backtest runtime addresses prices by bare symbol, so a panel
        keyed by AAA@XNYS would fetch nothing or the wrong series."""
        model_id = _train_a_model_with_spec(
            _spec(universe=QUALIFIED), dataset_id="ds_venue_keys"
        )
        with pytest.raises(ValidationError, match="evaluate_model_portfolio"):
            backtest_model_signal(
                BacktestModelSignalInput(
                    model_id=model_id, run_id="bridge_venue", name="panel"
                )
            )


class TestAttachingTheOutcomes:
    """Neither this library's experiment reference nor its ensemble
    reference carried a realized outcome, so `score_predictions`
    refused both -- the library could build an ensemble and backtest it
    and could not produce one statistical number for it."""

    def test_the_loop_that_fails_today_closes(self, patched_multi_factory):
        model = _trained(_dataset())

        with pytest.raises(ValidationError, match="no 'target' column"):
            score_predictions(
                ScorePredictionsInput(
                    predictions_ref=model.oos_predictions_ref, task="regression"
                )
            )

        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                model_id=model.model_id, run_id="outcomes_model", name="scoreable"
            )
        )
        assert attached.ref.startswith("sqt://predictions/")
        assert attached.task == "regression"
        assert attached.target_id == "forward_return:5"
        assert attached.horizon == 5
        assert attached.n_rows > 0

        scored = score_predictions(
            ScorePredictionsInput(
                predictions_ref=attached.ref,
                task="regression",
                horizon=attached.horizon,
            )
        )
        assert math.isfinite(scored.cross_sectional_ic["ic_mean"])
        assert scored.n_observations == attached.n_rows

    def test_the_published_columns_are_exactly_the_four(self, patched_multi_factory):
        """`prediction` and `target` and what identifies the pair, and
        nothing else: `lower`/`upper` and a fold id are not outcomes, and a
        scorer that finds them has to decide what they mean."""
        model = _trained(_dataset())
        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                model_id=model.model_id, run_id="outcomes_columns", name="scoreable"
            )
        )
        assert attached.columns == ["date", "entity", "prediction", "target"]
        frame = handoff.resolve(attached.ref, expect="predictions")
        assert list(frame.columns) == ["date", "entity", "prediction", "target"]
        assert len(frame) == attached.n_rows
        assert handoff.describe(attached.ref)["producer"] == (
            "modeling.attach_model_outcomes"
        )

    def test_an_ensemble_reference_scores_against_its_members(
        self, patched_multi_factory
    ):
        """The diversification number `ensemble.py` exists to produce, and
        which was three lines away and unreachable."""
        dataset_id = _dataset()
        ridge = _trained(dataset_id, _model_spec("ridge", alpha=1.0))
        forest = _trained(
            dataset_id, _model_spec("random_forest", n_estimators=20, max_depth=3)
        )
        ensemble = build_model_ensemble(
            BuildEnsembleInput(
                model_ids=[ridge.model_id, forest.model_id],
                run_id="outcomes_ensemble",
                name="combo",
            )
        )

        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                predictions_ref=ensemble.ref,
                dataset_id=dataset_id,
                run_id="outcomes_ensemble",
                name="combo_scoreable",
            )
        )
        # A reference carries no manifest, so the task the LABEL admits is
        # all there is -- a forward return is scoreable as regression or
        # as ranking, and which is claimed is the caller's decision.
        assert attached.task is None
        assert any("scoreable as" in warning for warning in attached.warnings)
        assert attached.horizon == 5

        def _icir(ref: str) -> float:
            return score_predictions(
                ScorePredictionsInput(predictions_ref=ref, task="regression", horizon=5)
            ).cross_sectional_ic["ic_icir"]

        members = [
            _icir(
                attach_model_outcomes(
                    AttachModelOutcomesInput(
                        model_id=model_id,
                        run_id="outcomes_ensemble",
                        name=f"member_{index}",
                    )
                ).ref
            )
            for index, model_id in enumerate((ridge.model_id, forest.model_id))
        ]
        combined = _icir(attached.ref)
        assert all(math.isfinite(value) for value in members)
        assert math.isfinite(combined)
        # Comparable, not necessarily better: the same metric on the same
        # label. A combination scored against the WRONG label would not
        # land anywhere near its own members.
        assert min(members) - 0.5 <= combined <= max(members) + 0.5

    def test_a_two_horizon_dataset_is_refused_rather_than_guessed(
        self, patched_multi_factory
    ):
        dataset_id = _dataset(_spec(target=TargetSpec(horizons=[5, 10])))
        model = _trained(dataset_id)

        # The MODEL path is never ambiguous -- the manifest names the label
        # the model was fit on.
        from_model = attach_model_outcomes(
            AttachModelOutcomesInput(
                model_id=model.model_id,
                run_id="outcomes_multi_horizon",
                name="from_model",
            )
        )
        assert from_model.target_id == "forward_return:5" and from_model.horizon == 5

        # The REFERENCE path is, and the panel's plain `target` column
        # holds the primary, so guessing would have looked like an answer.
        with pytest.raises(ValidationError, match="ambiguous") as excinfo:
            attach_model_outcomes(
                AttachModelOutcomesInput(
                    predictions_ref=model.oos_predictions_ref,
                    dataset_id=dataset_id,
                    run_id="outcomes_multi_horizon",
                    name="guessed",
                )
            )
        assert "'h5'" in str(excinfo.value) and "'h10'" in str(excinfo.value)

        named = attach_model_outcomes(
            AttachModelOutcomesInput(
                predictions_ref=model.oos_predictions_ref,
                dataset_id=dataset_id,
                target="h10",
                run_id="outcomes_multi_horizon",
                name="named",
            )
        )
        assert named.target_id == "forward_return:10" and named.horizon == 10

        with pytest.raises(ValidationError, match="no label named"):
            attach_model_outcomes(
                AttachModelOutcomesInput(
                    predictions_ref=model.oos_predictions_ref,
                    dataset_id=dataset_id,
                    target="h30",
                    run_id="outcomes_multi_horizon",
                    name="absent",
                )
            )

    def test_the_reported_horizon_is_the_datasets(self, patched_multi_factory):
        """It feeds `ScorePredictionsInput.horizon`, which an agent
        currently has to remember from dataset-build time."""
        spec = _spec(target=TargetSpec(horizon=7))
        model = _trained(_dataset(spec))
        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                model_id=model.model_id, run_id="outcomes_horizon", name="scoreable"
            )
        )
        assert attached.horizon == spec.target.horizon == 7

    def test_a_cpcv_model_is_refused_by_name(self, patched_multi_factory):
        """Its frame carries one prediction per path per row, so the join
        would attach one outcome to each and multiply the sample."""
        model = _trained(_dataset(), _cpcv_spec())
        with pytest.raises(ValidationError, match="walk_forward") as excinfo:
            attach_model_outcomes(
                AttachModelOutcomesInput(
                    model_id=model.model_id, run_id="bridge_cpcv", name="scoreable"
                )
            )
        assert "attach_model_outcomes" in str(excinfo.value)

    def test_the_source_is_exactly_one_and_a_reference_needs_its_dataset(self):
        with pytest.raises(PydanticValidationError, match="exactly one"):
            AttachModelOutcomesInput(run_id="r", name="n")
        with pytest.raises(PydanticValidationError, match="exactly one"):
            AttachModelOutcomesInput(
                model_id="mdl_x",
                predictions_ref="sqt://predictions/a/b",
                run_id="r",
                name="n",
            )
        with pytest.raises(PydanticValidationError, match="dataset_id is required"):
            AttachModelOutcomesInput(
                predictions_ref="sqt://predictions/a/b", run_id="r", name="n"
            )
        with pytest.raises(PydanticValidationError, match="read from the model"):
            AttachModelOutcomesInput(
                model_id="mdl_x", dataset_id="ds_x", run_id="r", name="n"
            )


class TestScoreModelPublishesAReference:
    """It returned a filesystem path, and `handoff.resolve` refuses a raw
    path under `expect=` -- so a live scoring run could be monitored for
    drift and could never be scored against outcomes or traded."""

    def test_the_reference_resolves_to_the_same_frame_as_the_path(
        self, patched_multi_factory
    ):
        model = _trained(_dataset())
        result = score_model(
            ScoreModelInput(
                model_id=model.model_id, as_of="2023-12-29", universe=UNIVERSE
            )
        )
        assert result.predictions_ref
        assert result.predictions_ref.startswith(
            f"sqt://predictions/{model.model_id}/scored_"
        )
        resolved = handoff.resolve(result.predictions_ref, expect="predictions")
        pd.testing.assert_frame_equal(
            resolved.reset_index(drop=True),
            load_artifact(result.predictions_uri).reset_index(drop=True),
        )
        # The dead end this closes: the path alone cannot be type-checked.
        with pytest.raises(ValidationError, match="raw artifact path"):
            handoff.resolve(result.predictions_uri, expect="predictions")

    def test_rescoring_the_same_universe_and_date_does_not_collide(
        self, patched_multi_factory
    ):
        """The artifact is content-addressed and the publish is therefore
        idempotent; without overwrite=True the second call would hit the
        'a value is already published' refusal with its own bytes."""
        model = _trained(_dataset())
        first = score_model(
            ScoreModelInput(
                model_id=model.model_id, as_of="2023-12-29", universe=UNIVERSE
            )
        )
        second = score_model(
            ScoreModelInput(
                model_id=model.model_id, as_of="2023-12-29", universe=UNIVERSE
            )
        )
        assert first.predictions_uri == second.predictions_uri
        assert first.predictions_ref == second.predictions_ref

    def test_a_scored_universe_can_now_be_scored_against_outcomes(self, monkeypatch):
        """
        The workflow the missing reference made impossible: score forward,
        wait for the label to resolve, then measure what the live
        predictions were worth.

        The model is fitted on a SHORT history so that `as_of` can be
        after its training cutoff -- score_model refuses anything at or
        before it, because the refit consumed prices through that date --
        and the outcomes come from a dataset built once the fuller history
        exists. That is not test scaffolding; it is the only shape this
        question has, since a genuinely forward prediction has no realized
        outcome at the moment it is made.
        """
        from standard_quant_tools.data.factory import DataFactory

        from .conftest import make_ohlcv, make_provider_mock

        calendar = make_ohlcv("AAA").index

        def _history_through(last):
            provider = make_provider_mock(lambda symbol: make_ohlcv(symbol).loc[:last])
            monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        _history_through(calendar[299])
        model = _trained(_dataset())

        _history_through(calendar[379])
        as_of = calendar[379].strftime("%Y-%m-%d")
        scored = score_model(
            ScoreModelInput(model_id=model.model_id, as_of=as_of, universe=UNIVERSE)
        )
        assert scored.effective_score_date == as_of

        _history_through(calendar[-1])
        outcomes_id = _dataset(_spec())
        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                predictions_ref=scored.predictions_ref,
                dataset_id=outcomes_id,
                run_id="forward_scoring",
                name="forward_scoreable",
            )
        )
        assert attached.columns == ["date", "entity", "prediction", "target"]
        assert attached.n_rows == scored.n_entities
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=attached.ref,
                task="regression",
                horizon=attached.horizon,
            )
        )
        assert result.n_observations == attached.n_rows
        assert result.n_dates == 1
