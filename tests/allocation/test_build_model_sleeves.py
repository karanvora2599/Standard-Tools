"""
Registered models as sleeves: `build_model_sleeves`.

Every number here is checked against something the tool does not compute:
each panel column against the evaluation `evaluate_model_portfolio` makes of
the same model, re-read from its own equity curve with the capital put back
in place of the opening bar; the compounded stream against that evaluation's
`cumulative_return`; the correlations against numpy's; the trial count
against the models this file registered; and the deflated Sharpe ratio at
the reported count against Bailey and Lopez de Prado's formula written out
with the standard library's normal distribution.

The models are real ones -- datasets built and ridge regressions trained
through the modeling tools on the per-symbol synthetic provider the modeling
suite uses -- registered once for the module in a runs directory of its own,
so nothing else in the session can add to the registry this file counts.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.agent.runtimes.allocation.model_sleeve_tools import (
    BuildModelSleevesInput,
)
from standard_quant_tools.agent.runtimes.allocation.model_sleeve_tools import (
    build_model_sleeves as build_model_sleeves_tool,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import model_sleeves
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    RunModelExperimentInput,
    build_model_dataset,
    run_model_experiment,
)
from standard_quant_tools.modeling.model_sleeves import (
    build_model_sleeves,
    opening_capital_returns,
    registered_trials,
)
from standard_quant_tools.modeling.portfolio_eval import evaluate_model_portfolio
from standard_quant_tools.modeling.registry.model_registry import load_manifest
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
from tests.modeling.conftest import make_ohlcv, make_provider_mock

UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
EULER_GAMMA = 0.5772156649015329


def _dataset(horizon: int, lookback: int) -> str:
    return build_model_dataset(
        BuildModelDatasetInput(
            spec=DatasetSpec(
                universe=UNIVERSE,
                start="2022-01-01",
                end="2023-12-31",
                features=[
                    FeatureSpec(
                        id="market.momentum",
                        params={"lookback": lookback},
                        alias=f"mom_{lookback}",
                    ),
                    FeatureSpec(id="technical.rsi", params={"period": 14}),
                ],
                target=TargetSpec(type="forward_return", horizon=horizon),
            )
        )
    ).dataset_id


def _ridge(dataset_id: str, alpha: float, train_window: int, seed: int) -> str:
    return run_model_experiment(
        RunModelExperimentInput(
            dataset_id=dataset_id,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": alpha}),
                validation=ValidationSpec(
                    train_window=train_window, test_window=40, embargo=5, min_folds=2
                ),
                random_seed=seed,
            ),
        )
    ).model_id


def _cpcv(dataset_id: str) -> str:
    return run_model_experiment(
        RunModelExperimentInput(
            dataset_id=dataset_id,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(method="cpcv", n_splits=6, n_test_splits=2),
                random_seed=1,
            ),
        )
    ).model_id


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """
    Two datasets and five registered models, in a runs directory of the
    module's own, priced by the synthetic provider for the whole module (the
    evaluations fetch prices when the tests run, not when this registers).

    Dataset A (5-bar target) holds `m1` and `m2`, which are passed, and
    `m3`, which is registered and never passed; `m2` trains on a longer
    window, so it first trades later. Dataset B (10-bar target) holds `m4`
    and a combinatorial-CV model.
    """
    patch = pytest.MonkeyPatch()
    patch.setenv("SQT_RUNS_DIR", str(tmp_path_factory.mktemp("sleeve_runs")))
    patch.setattr(
        DataFactory, "get_provider", lambda *a, **kw: make_provider_mock(make_ohlcv)
    )
    try:
        a = _dataset(horizon=5, lookback=20)
        b = _dataset(horizon=10, lookback=10)
        ids = {
            "a": a,
            "b": b,
            "m1": _ridge(a, 1.0, 150, seed=1),
            "m2": _ridge(a, 10.0, 200, seed=2),
            "m3": _ridge(a, 100.0, 150, seed=3),
            "m4": _ridge(b, 1.0, 150, seed=4),
            "cpcv": _cpcv(b),
        }
        yield ids
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def default_build(registry):
    """m1 and m2 with every default: idle bars dropped."""
    return build_model_sleeves(
        [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
        run_id="sleeves_default",
    )


@pytest.fixture(scope="module")
def idle_kept_build(registry):
    return build_model_sleeves(
        [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
        run_id="sleeves_idle",
        keep_idle=True,
    )


def _panel(out) -> pd.DataFrame:
    return handoff.resolve(out["returns_ref"], expect="returns_panel")


def _from_capital(model_id: str, portfolio: PortfolioSimSpec = None):
    """
    The oracle: the evaluation run again on its own, its curve read back,
    its first point replaced by the capital, and returns taken by pandas.
    """
    portfolio = portfolio or PortfolioSimSpec()
    evaluation = evaluate_model_portfolio(model_id, portfolio=portfolio)
    curve = handoff.resolve(evaluation["equity_curve_ref"], expect="equity_curve")
    levels = curve.dropna().astype(float).copy()
    levels.iloc[0] = portfolio.initial_capital
    return evaluation, curve, (levels / levels.shift(1) - 1.0).iloc[1:]


class TestTheStreamIsTheEvaluations:
    def test_a_planted_opening_cost_is_in_the_first_return(self):
        """
        A book that paid 10 bps to open: capital 100, the opening bar marked
        at 99.9 after the cost, then +2% and -1%. From the capital the first
        return is 101.898 / 100 - 1 = 1.898%, carrying the cost; read from
        the opening bar, as a plain pct_change does, it would be 2% and the
        stream 10 bps richer than the account.
        """
        equity = pd.Series(
            [99.9, 99.9 * 1.02, 99.9 * 1.02 * 0.99],
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        returns = opening_capital_returns(equity, 100.0)
        assert list(returns.index) == list(equity.index[1:])
        assert returns.iloc[0] == pytest.approx(99.9 * 1.02 / 100.0 - 1.0, abs=1e-15)
        assert returns.iloc[1] == pytest.approx(-0.01, abs=1e-15)
        compounded = float(np.prod(1.0 + returns.to_numpy())) - 1.0
        assert compounded == pytest.approx(equity.iloc[-1] / 100.0 - 1.0, abs=1e-15)

    def test_changing_the_curve_after_a_date_changes_no_return_up_to_it(self):
        """A return is read from its bar and the one before it, so a curve
        rewritten from bar 30 on leaves every return through bar 29 as it
        was, the first one -- read from the capital -- included."""
        rng = np.random.default_rng(5)
        index = pd.bdate_range("2024-01-02", periods=60)
        curve = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 60)), index=index)
        changed = curve.copy()
        changed.iloc[30:] *= 1.0 + rng.normal(0, 0.05, 30)
        before = opening_capital_returns(curve, 100.0)
        after = opening_capital_returns(changed, 100.0)
        cut = index[29]
        np.testing.assert_array_equal(before[:cut].to_numpy(), after[:cut].to_numpy())
        assert not np.array_equal(before[index[30] :], after[index[30] :])

    def test_each_column_is_its_evaluations_return_from_the_capital(
        self, registry, idle_kept_build
    ):
        """With the idle bars kept, the panel is the two evaluations' curves
        bar for bar: every value equals the oracle's exactly."""
        panel = _panel(idle_kept_build)
        for key in ("m1", "m2"):
            model_id = registry[key]
            _evaluation, curve, expected = _from_capital(model_id)
            assert len(panel) == len(curve) - 1
            np.testing.assert_array_equal(
                panel[model_id].to_numpy(),
                expected.reindex(panel.index).to_numpy(),
            )

    def test_the_full_stream_compounds_to_the_cumulative_return(
        self, registry, idle_kept_build
    ):
        panel = _panel(idle_kept_build)
        for key in ("m1", "m2"):
            model_id = registry[key]
            evaluation = evaluate_model_portfolio(model_id)
            compounded = float(np.prod(1.0 + panel[model_id].to_numpy())) - 1.0
            assert abs(compounded - evaluation["metrics"]["cumulative_return"]) < 1e-12

    def test_the_dropped_idle_returns_are_the_accounts_cash(
        self, registry, default_build
    ):
        """
        At the default zero rate the account's bars before its first trade
        earn exactly nothing, so dropping them leaves a stream that still
        compounds to the evaluation's total. Under the next-open fill the
        first live return is the bar after the first rebalance date: the
        weights dated there fill at the next open.
        """
        panel = _panel(default_build)
        for key in ("m1", "m2"):
            model_id = registry[key]
            row = default_build["sleeves"][model_id]
            evaluation, curve, expected = _from_capital(model_id)
            first = pd.Timestamp(row["first_rebalance_date"])
            position = curve.index.get_loc(first)
            idle = expected.iloc[:position]
            assert row["n_idle_returns"] == position == len(idle) > 50
            assert (idle.to_numpy() == 0.0).all()
            assert row["idle_return"] == 0.0
            live = expected.iloc[position:]
            assert live.index[0] == curve.index[position + 1]
            assert row["n_returns"] == len(live)
            compounded = float(np.prod(1.0 + live.to_numpy())) - 1.0
            assert abs(compounded - evaluation["metrics"]["cumulative_return"]) < 1e-12
            np.testing.assert_array_equal(
                panel[model_id].to_numpy(), live.reindex(panel.index).to_numpy()
            )

    def test_with_a_cash_rate_the_idle_growth_closes_the_identity(self, registry):
        """At 3% the idle bars accrue, and (1 + live)(1 + idle) - 1 is the
        evaluation's cumulative return."""
        portfolio = PortfolioSimSpec(risk_free_rate=0.03)
        out = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m3"]}],
            run_id="sleeves_cash",
            portfolio=portfolio,
        )
        panel = _panel(out)
        for key in ("m1", "m3"):
            model_id = registry[key]
            row = out["sleeves"][model_id]
            evaluation, _curve, expected = _from_capital(model_id, portfolio)
            assert row["idle_return"] > 0.0
            # The idle bars accrue the cash rate, about 3% a year on them.
            years = row["n_idle_returns"] / 252.0
            assert row["idle_return"] == pytest.approx(0.03 * years, rel=0.1)
            live = expected.iloc[row["n_idle_returns"] :]
            total = (1.0 + float(np.prod(1.0 + live.to_numpy())) - 1.0) * (
                1.0 + row["idle_return"]
            ) - 1.0
            assert abs(total - evaluation["metrics"]["cumulative_return"]) < 1e-12
            assert set(panel.index) <= set(live.index)

    def test_a_close_fill_goes_live_on_the_rebalance_bar(self, registry):
        portfolio = PortfolioSimSpec(fill_price="close")
        out = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m3"]}],
            run_id="sleeves_close",
            portfolio=portfolio,
        )
        _evaluation, curve, _expected = _from_capital(registry["m1"], portfolio)
        row = out["sleeves"][registry["m1"]]
        first = pd.Timestamp(row["first_rebalance_date"])
        assert row["n_idle_returns"] == curve.index.get_loc(first) - 1
        assert any("look-ahead" in w for w in out["warnings"])


class TestTheSleeveNumbersAreTheEvaluations:
    def test_every_number_is_the_evaluations_own(self, registry, default_build):
        fields = {
            "sharpe": "sharpe_ratio",
            "annual_return": "cagr",
            "volatility": "annualized_volatility",
            "max_drawdown": "max_drawdown",
            "total_return": "cumulative_return",
            "annual_turnover": "annualized_turnover",
            "n_rebalances": "n_rebalances",
        }
        for key in ("m1", "m2"):
            model_id = registry[key]
            row = default_build["sleeves"][model_id]
            evaluation = evaluate_model_portfolio(model_id)
            for ours, theirs in fields.items():
                assert row[ours] == evaluation["metrics"][theirs], ours
            assert row["equity_curve_ref"] == evaluation["equity_curve_ref"]
            assert row["target_weights_ref"] == evaluation["target_weights_ref"]
            manifest = load_manifest(model_id)
            assert row["estimator_type"] == manifest.estimator_type == "ridge"
            assert row["dataset_id"] == manifest.dataset_id == registry["a"]
            assert row["target_horizon"] == 5

    def test_a_models_own_transform_replaces_the_shared_one(self, registry):
        monthly = PredictionTransformSpec(rebalance_frequency="monthly")
        out = build_model_sleeves(
            [
                {"model_id": registry["m1"], "transform": monthly},
                {"model_id": registry["m3"]},
            ],
            run_id="sleeves_override",
        )
        own = evaluate_model_portfolio(registry["m1"], transform=monthly)
        shared = evaluate_model_portfolio(registry["m3"])
        assert (
            out["sleeves"][registry["m1"]]["sharpe"] == own["metrics"]["sharpe_ratio"]
        )
        assert (
            out["sleeves"][registry["m1"]]["n_rebalances"]
            == own["metrics"]["n_rebalances"]
            < shared["metrics"]["n_rebalances"]
        )
        assert out["sleeves"][registry["m3"]]["sharpe"] == (
            shared["metrics"]["sharpe_ratio"]
        )


class TestTheCalendar:
    def test_the_alignment_drops_the_dates_not_shared_and_reports_them(
        self, registry, default_build
    ):
        """m2 trains on 200 bars against m1's 150, so it trades later: the
        panel starts at m2's first live bar, and m1's earlier live returns
        are reported as dropped before the shared window, not filled."""
        panel = _panel(default_build)
        m1, m2 = registry["m1"], registry["m2"]
        _e1, _c1, s1 = _from_capital(m1)
        _e2, _c2, s2 = _from_capital(m2)
        live1 = s1.iloc[default_build["sleeves"][m1]["n_idle_returns"] :]
        live2 = s2.iloc[default_build["sleeves"][m2]["n_idle_returns"] :]
        assert live1.index[0] < live2.index[0]
        assert panel.index[0] == live2.index[0]
        report = default_build["alignment"]
        early = int((live1.index < live2.index[0]).sum())
        assert early > 0
        assert report["by_stream"][m1]["n_dropped_before"] == early
        assert report["by_stream"][m2]["n_dropped_before"] == 0
        assert report["start_set_by"] == [m2]
        assert report["n_dates"] == len(panel) == len(live2)
        assert not panel.isna().any().any()
        assert any("dropped" in w for w in default_build["warnings"])

    def test_the_correlation_is_numpys(self, registry, default_build):
        panel = _panel(default_build)
        expected = np.corrcoef(panel.to_numpy().T)
        names = list(panel.columns)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                assert default_build["correlation"][a][b] == pytest.approx(
                    expected[i, j], abs=1e-12
                )

    def test_different_target_horizons_are_said_to_be(self, registry):
        out = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m4"]}],
            run_id="sleeves_horizons",
        )
        (note,) = [w for w in out["warnings"] if "target horizons" in w]
        assert "5" in note and "10" in note


class TestTheTrialCount:
    def test_it_counts_the_registered_model_not_passed_and_not_another_dataset(
        self, registry, default_build
    ):
        """m3 shares dataset A and was never passed: it is a model tried. m4
        and the cpcv model are on dataset B, which these models were not
        trained on."""
        assert default_build["trials_by_dataset"] == {registry["a"]: 3}
        assert default_build["n_trials_lower_bound"] == 3

    def test_two_datasets_are_both_counted(self, registry):
        out = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m4"]}],
            run_id="sleeves_two_datasets",
        )
        assert out["trials_by_dataset"] == {registry["a"]: 3, registry["b"]: 2}
        assert out["n_trials_lower_bound"] == 5

    def test_the_count_is_never_below_the_models_passed(self, registry, monkeypatch):
        assert registered_trials(["ds_nothing_here"]) == {"ds_nothing_here": 0}
        monkeypatch.setattr(
            model_sleeves,
            "registered_trials",
            lambda ids: {d: 0 for d in ids},
        )
        out = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
            run_id="sleeves_floor",
        )
        assert out["n_trials_lower_bound"] == 2


class TestRefusals:
    def test_one_model_is_refused_at_the_boundary_and_in_the_library(self, registry):
        with pytest.raises(pydantic.ValidationError, match="at least 2"):
            BuildModelSleevesInput(models=[{"model_id": registry["m1"]}], run_id="r")
        with pytest.raises(ValidationError, match="at least 2"):
            build_model_sleeves([{"model_id": registry["m1"]}], run_id="r")

    def test_more_than_twenty_is_refused(self):
        models = [{"model_id": f"mdl_{i}"} for i in range(21)]
        with pytest.raises(pydantic.ValidationError, match="more than the 20"):
            BuildModelSleevesInput(models=models, run_id="r")

    def test_a_model_named_twice_is_refused(self, registry):
        twice = [
            {"model_id": registry["m1"]},
            {"model_id": registry["m1"], "name": "x"},
        ]
        with pytest.raises(pydantic.ValidationError, match="more than once"):
            BuildModelSleevesInput(models=twice, run_id="r")
        with pytest.raises(ValidationError, match="more than once"):
            build_model_sleeves(twice, run_id="r")

    def test_a_sleeve_name_given_twice_is_refused(self, registry):
        clash = [
            {"model_id": registry["m1"]},
            {"model_id": registry["m2"], "name": registry["m1"]},
        ]
        with pytest.raises(ValidationError, match="distinct name"):
            build_model_sleeves(clash, run_id="r")

    def test_a_cpcv_model_is_refused(self, registry):
        with pytest.raises(ValidationError, match="walk_forward"):
            build_model_sleeves(
                [{"model_id": registry["m1"]}, {"model_id": registry["cpcv"]}],
                run_id="r",
            )

    def test_an_unknown_model_is_refused(self, registry):
        with pytest.raises(ValidationError, match="no registered model"):
            build_model_sleeves(
                [{"model_id": registry["m1"]}, {"model_id": "mdl_nothing"}],
                run_id="r",
            )

    def test_too_little_overlap_is_refused(self, registry):
        with pytest.raises(ValidationError, match="min_overlap"):
            build_model_sleeves(
                [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
                run_id="r",
                min_overlap=100_000,
            )

    def test_bars_of_different_periods_are_refused(self, registry, monkeypatch):
        """A weekly model beside a daily one: one row would hold a week's
        return and a day's, and the panel has one periods_per_year (52
        against 252). The interval is planted on m2's spec as read, since
        the synthetic provider serves daily bars only."""
        real = model_sleeves.load_dataset_spec

        def weekly_for_m2(model_id):
            spec = dict(real(model_id))
            if model_id == registry["m2"]:
                spec.update(interval="1wk")
            return spec

        monkeypatch.setattr(model_sleeves, "load_dataset_spec", weekly_for_m2)
        with pytest.raises(ValidationError, match="not the same period"):
            build_model_sleeves(
                [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
                run_id="r",
            )

    def test_a_models_transform_past_the_leverage_names_the_model(self, registry):
        with pytest.raises(ValidationError, match=r"models\[1\]"):
            build_model_sleeves(
                [
                    {"model_id": registry["m1"]},
                    {
                        "model_id": registry["m2"],
                        "transform": PredictionTransformSpec(gross_exposure=2.0),
                    },
                ],
                run_id="r",
            )


class TestIdempotence:
    def test_the_same_call_republishes_the_same_reference(
        self, registry, default_build
    ):
        again = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
            run_id="sleeves_default",
        )
        assert again["returns_ref"] == default_build["returns_ref"]
        pd.testing.assert_frame_equal(_panel(again), _panel(default_build))

    def test_other_models_under_the_same_run_id_publish_beside_it(
        self, registry, default_build
    ):
        other = build_model_sleeves(
            [{"model_id": registry["m1"]}, {"model_id": registry["m3"]}],
            run_id="sleeves_default",
        )
        assert other["returns_ref"] != default_build["returns_ref"]
        assert list(_panel(default_build).columns) == [registry["m1"], registry["m2"]]


class TestTheSurface:
    def test_it_is_advertised_as_evaluate_model_portfolio_is(self):
        """It fetches through the models' datasets as that tool does and
        names no ticker, so its MCP hints are that tool's: not open-world
        by schema, and not idempotent, because it writes artifacts."""
        from standard_quant_tools.mcp.catalog import build_catalog

        catalog = build_catalog()
        ours, theirs = (
            catalog["build_model_sleeves"],
            catalog["evaluate_model_portfolio"],
        )
        assert ours.runtime == "allocation"
        assert ours.reads_market_data == theirs.reads_market_data
        assert ours.idempotent == theirs.idempotent is False

    def test_a_models_transform_is_the_modeling_spec(self):
        model = BuildModelSleevesInput(
            models=[
                {"model_id": "mdl_a", "transform": {"method": "sign"}},
                {"model_id": "mdl_b"},
            ],
            run_id="r",
        )
        assert isinstance(model.models[0].transform, PredictionTransformSpec)
        with pytest.raises(pydantic.ValidationError):
            BuildModelSleevesInput(
                models=[
                    {"model_id": "mdl_a", "transform": {"not_a_field": 1}},
                    {"model_id": "mdl_b"},
                ],
                run_id="r",
            )

    def test_the_tool_answers_through_the_runtime(self, registry):
        payload = resolve("allocation").dispatch(
            "build_model_sleeves",
            {
                "models": [
                    {"model_id": registry["m1"], "name": "first"},
                    {"model_id": registry["m2"], "name": "second"},
                ],
                "run_id": "sleeves_dispatch",
            },
        )
        assert payload["columns"] == ["first", "second"]
        assert payload["n_trials_lower_bound"] == 3
        assert payload["periods_per_year"] == 252
        assert payload["returns_ref"].startswith(
            "sqt://returns_panel/sleeves_dispatch/"
        )
        assert payload["sleeves"]["first"]["model_id"] == registry["m1"]
        typed = build_model_sleeves_tool(
            BuildModelSleevesInput(
                models=[{"model_id": registry["m1"]}, {"model_id": registry["m2"]}],
                run_id="sleeves_typed",
            )
        )
        assert typed.alignment.n_dates == typed.n_dates


class TestTheDeflatedSharpeAtTheReportedCount:
    def test_the_count_lowers_the_deflated_sharpe_by_what_it_implies(
        self, registry, default_build
    ):
        """
        The panel walked forward by evaluate_allocation_rule at n_trials=1
        and at the reported count. The deflated Sharpe probability falls,
        and by exactly the expected maximum of that many null Sharpe ratios
        (Bailey and Lopez de Prado 2014), measured in the rule's own Sharpe
        standard error -- here computed from the rule's published stream
        with the standard library's normal, not the library's.
        """
        count = default_build["n_trials_lower_bound"]
        base = {
            "returns_ref": default_build["returns_ref"],
            "rule": {"method": "inverse_volatility", "covariance_method": "sample"},
            "estimation_window": 60,
            "rebalance_every": 21,
            "n_bootstrap": 0,
            "periods_per_year": default_build["periods_per_year"],
        }
        runtime = resolve("allocation")
        one = runtime.dispatch(
            "evaluate_allocation_rule", {**base, "n_trials": 1, "run_id": "dsr_one"}
        )
        many = runtime.dispatch(
            "evaluate_allocation_rule",
            {**base, "n_trials": count, "run_id": "dsr_many"},
        )
        assert many["n_trials"] == count == 3
        p_one = one["deflated_sharpe_probability"]
        p_many = many["deflated_sharpe_probability"]
        assert p_many < p_one

        stream = handoff.resolve(one["oos_returns_ref"], expect="returns_panel")["rule"]
        values = stream.to_numpy(dtype=float)
        n = values.size
        sharpe = values.mean() / values.std(ddof=1)
        centred = (values - values.mean()) / values.std(ddof=1)
        skew = float((centred**3).mean())
        kurtosis = float((centred**4).mean())
        error = math.sqrt(
            (1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2) / (n - 1)
        )
        normal = NormalDist()
        expected_max = math.sqrt(1.0 / n) * (
            (1.0 - EULER_GAMMA) * normal.inv_cdf(1.0 - 1.0 / count)
            + EULER_GAMMA * normal.inv_cdf(1.0 - 1.0 / (count * math.e))
        )
        assert p_one == pytest.approx(normal.cdf(sharpe / error), abs=1e-9)
        assert p_many == pytest.approx(
            normal.cdf((sharpe - expected_max) / error), abs=1e-9
        )
        assert normal.inv_cdf(p_one) - normal.inv_cdf(p_many) == pytest.approx(
            expected_max / error, abs=1e-6
        )
        assert many["deflation_threshold"] - one["deflation_threshold"] == (
            pytest.approx(expected_max * math.sqrt(252), abs=1e-9)
        )
