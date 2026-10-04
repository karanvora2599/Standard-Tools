"""
A dataset's column names are resolved through the dataset.

A dataset's columns are its features' OUTPUT names -- the alias where one was
given, `mom_126` for market.momentum at 126 bars -- while the catalog knows
only `market.momentum`. Three tools read one vocabulary where the caller held
the other:

    estimate_feature_warmup  took spec dicts only, so pricing a built
                             dataset's own features meant re-typing them
    check_leakage            checked `mom_126` against the catalog and
                             reported it "not in the feature registry",
                             `safe: False`, for a feature that is in it; a
                             catalog id skipped the screen; and the default
                             with a dataset checked the whole registry
    (no tool)                described one dataset's definition at all;
                             inspect_dataset now does, from JSON only

The dataset here is built offline with aliases, the shape the live session
used. See the CHANGELOG entry of 2026-10-04.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.discovery_models import (
    EstimateFeatureWarmupInput,
)
from standard_quant_tools.modeling.agent.discovery_tools import (
    estimate_feature_warmup,
)
from standard_quant_tools.modeling.agent.dispatch import modeling_dispatch
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    CheckLeakageInput,
    InspectDatasetInput,
    RegisterExternalPanelInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    check_leakage,
    inspect_dataset,
    register_external_panel,
    run_model_experiment,
)
from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

ALIASED = [
    FeatureSpec(id="technical.rsi", params={"period": 14}, alias="rsi_14"),
    FeatureSpec(
        id="market.momentum", params={"lookback": 20}, alias="mom_20", lags=[1]
    ),
    FeatureSpec(id="market.momentum", params={"lookback": 60}, alias="mom_60"),
    FeatureSpec(id="technical.macd_histogram", alias="macdh"),
    FeatureSpec(id="risk.realized_volatility"),
]
NAMES = ["rsi_14", "mom_20", "mom_60", "macdh", "risk.realized_volatility"]


def _spec(**overrides) -> DatasetSpec:
    defaults = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=ALIASED,
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


@pytest.fixture
def aliased(patched_multi_factory) -> str:
    return build_model_dataset(BuildModelDatasetInput(spec=_spec())).dataset_id


def _external(tmp_path) -> str:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-02", periods=160)
    frame = pd.DataFrame(
        [
            {
                "date": d,
                "entity": e,
                "alpha": float(rng.normal()),
                "noise": float(rng.normal()),
                "target": float(rng.normal(0, 0.01)),
            }
            for d in dates
            for e in ("AAA", "BBB", "CCC", "DDD")
        ]
    )
    path = tmp_path / "external.parquet"
    frame.to_parquet(path, index=False)
    return register_external_panel(
        RegisterExternalPanelInput(path=str(path), horizon=5)
    ).dataset_id


def _warmup(**kwargs):
    return estimate_feature_warmup(EstimateFeatureWarmupInput(**kwargs))


def _digest(result) -> str:
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


# ── estimate_feature_warmup ──────────────────────────────────────────────

#: The live dataset's eight FeatureSpecs, written out.
LIVE = [
    {"id": "technical.rsi", "params": {"period": 14}, "alias": "rsi_14", "lags": []},
    {
        "id": "market.momentum",
        "params": {"lookback": 20},
        "alias": "mom_20",
        "lags": [],
    },
    {
        "id": "market.momentum",
        "params": {"lookback": 126},
        "alias": "mom_126",
        "lags": [],
    },
    {
        "id": "risk.realized_volatility",
        "params": {"period": 20},
        "alias": "rvol_20",
        "lags": [],
    },
    {
        "id": "risk.rolling_beta",
        "params": {"window": 60},
        "alias": "beta_60",
        "lags": [],
    },
    {
        "id": "volume.volume_surprise",
        "params": {"period": 20},
        "alias": "volsurp_20",
        "lags": [],
    },
    {
        "id": "risk.bollinger_pct_b",
        "params": {"period": 20},
        "alias": "pctb_20",
        "lags": [],
    },
    {"id": "technical.macd_histogram", "params": {}, "alias": "macdh", "lags": []},
]

#: Spec-dict calls and the SHA-256 of their whole JSON result, recorded
#: before the tool took names and ids. Every case is interval arithmetic
#: with no calendar library involved, so the digests hold on every
#: interpreter.
SPEC_DICT_PINS = [
    (
        {"features": LIVE},
        "0a846712d5709b093672a6cd95356b71fedc02b64221528dcb1fab6ac1c02081",
    ),
    (
        {
            "features": [
                {"id": "technical.adx", "lags": [1, 5]},
                {"id": "statistical.hurst", "params": {"window": 300}},
                {"id": "fundamental.net_margin"},
                {"id": "technical.rsi"},
                {"id": "technical.rsi", "params": {"period": 30}},
            ],
            "interval": "1h",
        },
        "aa9434bf7b3aee8fba98fa236c6184858bed54311a68873e98cec7536969a049",
    ),
    (
        {
            "features": [{"id": "market.momentum", "params": {"lookback": 52}}],
            "interval": "1wk",
        },
        "1c4c2badc26e452891d1d7fcaaf76245c6f3db15cfc57b31c015078e98e13517",
    ),
]


class TestTheSpecDictCallIsUnchanged:
    @pytest.mark.parametrize("arguments, digest", SPEC_DICT_PINS)
    def test_bit_for_bit(self, arguments, digest):
        """The call every existing caller makes returns exactly what it
        returned before ids and names were accepted."""
        assert (
            _digest(modeling_dispatch("estimate_feature_warmup", arguments)) == digest
        )

    def test_the_live_numbers(self):
        """The live dataset's features: 126 bars binding on mom_126, 188 to
        converge binding on the MACD histogram, 272.5 calendar days."""
        out = _warmup(features=LIVE)
        assert (out.bars_required, out.binding_feature) == (126, "mom_126")
        assert (out.bars_required_converged, out.converged_binding_feature) == (
            188,
            "macdh",
        )
        assert round(out.calendar_days_converged, 1) == 272.5


class TestADatasetOrModelIsPricedFromItsRecord:
    def test_a_dataset_id_prices_exactly_its_recorded_specs(self, aliased):
        """The same answer as passing the dataset's own spec dicts by hand."""
        by_id = _warmup(dataset_id=aliased).model_dump()
        by_hand = _warmup(features=[f.model_dump() for f in ALIASED]).model_dump()
        assert by_id == by_hand
        assert set(by_id["per_feature"]) == set(NAMES)

    def test_names_select_columns_at_their_built_parameters(self, aliased):
        out = _warmup(dataset_id=aliased, features=["mom_60", "rsi_14"])
        assert set(out.per_feature) == {"mom_60", "rsi_14"}
        assert out.per_feature["mom_60"].resolved == 60
        assert out.binding_feature == "mom_60"

    def test_a_spec_beside_an_id_is_priced_on_the_recorded_interval(self, aliased):
        out = _warmup(
            dataset_id=aliased,
            features=["rsi_14", {"id": "market.momentum", "params": {"lookback": 90}}],
        )
        assert out.binding_feature == "market.momentum"
        assert out.bars_required == 90

    def test_a_model_id_prices_the_spec_the_model_carries(self, aliased):
        model_id = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=aliased,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=1,
                ),
            )
        ).model_id
        assert (
            _warmup(model_id=model_id).model_dump()
            == _warmup(dataset_id=aliased).model_dump()
        )

    def test_an_unknown_name_is_refused_with_the_columns(self, aliased):
        with pytest.raises(ValidationError, match="Its columns are") as caught:
            _warmup(dataset_id=aliased, features=["mom_126"])
        for name in NAMES:
            assert name in str(caught.value)

    def test_a_catalog_id_names_the_columns_built_from_it(self, aliased):
        with pytest.raises(ValidationError, match="mom_20") as caught:
            _warmup(dataset_id=aliased, features=["market.momentum"])
        assert "mom_60" in str(caught.value)

    def test_a_contradicting_interval_is_refused_naming_both(self, aliased):
        with pytest.raises(ValidationError, match="'1h'") as caught:
            _warmup(dataset_id=aliased, interval="1h")
        assert "'1d'" in str(caught.value)

    def test_the_recorded_interval_passed_explicitly_is_accepted(self, aliased):
        assert (
            _warmup(dataset_id=aliased, interval="1d").model_dump()
            == _warmup(dataset_id=aliased).model_dump()
        )

    def test_a_calendar_is_used_when_none_was_recorded(self, aliased):
        pytest.importorskip("exchange_calendars")
        plain = _warmup(dataset_id=aliased)
        venue = _warmup(dataset_id=aliased, calendar="XNYS")
        assert venue.bars_required == plain.bars_required
        assert venue.calendar_days_estimate != plain.calendar_days_estimate

    def test_a_contradicting_calendar_is_refused(self, patched_multi_factory):
        pytest.importorskip("exchange_calendars")
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec(calendar="XNYS"))
        ).dataset_id
        with pytest.raises(ValidationError, match="XLON") as caught:
            _warmup(dataset_id=dataset_id, calendar="XLON")
        assert "XNYS" in str(caught.value)

    def test_an_external_panel_is_refused_by_name(self, tmp_path):
        dataset_id = _external(tmp_path)
        with pytest.raises(ValidationError, match="external panel"):
            _warmup(dataset_id=dataset_id)

    def test_an_edited_spec_is_refused(self, aliased):
        from standard_quant_tools.modeling import artifacts

        path = artifacts.run_dir(aliased) / "dataset_spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["features"][0]["params"]["period"] = 100
        path.write_text(json.dumps(spec), encoding="utf-8")
        with pytest.raises(ValidationError, match="no longer matches"):
            _warmup(dataset_id=aliased)


class TestNamesWithoutAnId:
    def test_a_bare_catalog_id_is_that_feature_at_its_defaults(self):
        assert (
            _warmup(features=["technical.rsi", "market.momentum"]).model_dump()
            == _warmup(
                features=[{"id": "technical.rsi"}, {"id": "market.momentum"}]
            ).model_dump()
        )

    def test_a_column_name_without_an_id_says_how_to_resolve_it(self):
        with pytest.raises(ValidationError, match="dataset_id or model_id"):
            _warmup(features=["mom_126"])

    @pytest.mark.parametrize(
        "arguments",
        [{}, {"dataset_id": "ds_a", "model_id": "mdl_b"}, {"features": []}],
    )
    def test_nothing_or_two_sources_are_refused_at_the_schema(self, arguments):
        with pytest.raises(pydantic.ValidationError):
            EstimateFeatureWarmupInput(**arguments)


# ── check_leakage ────────────────────────────────────────────────────────


class TestCheckLeakageReadsTheDatasetsNames:
    def test_an_alias_is_safe_and_checked_under_its_catalog_id(self, aliased):
        """`mom_20` was reported "not in the feature registry", safe False."""
        result = check_leakage(
            CheckLeakageInput(dataset_id=aliased, feature_ids=["mom_20"])
        )
        assert result.safe is True
        assert result.findings == []
        assert set(result.screen) == {"mom_20"}
        assert any("mom_20 -> market.momentum" in note for note in result.notes)

    def test_a_catalog_id_is_screened_through_every_column_built_from_it(self, aliased):
        """It used to skip the screen: no column carries the catalog id."""
        result = check_leakage(
            CheckLeakageInput(dataset_id=aliased, feature_ids=["market.momentum"])
        )
        assert result.scope == "declared_and_empirical"
        assert set(result.screen) == {"mom_20", "mom_20__lag1", "mom_60"}

    def test_the_default_checks_the_datasets_own_features(self, aliased):
        result = check_leakage(CheckLeakageInput(dataset_id=aliased))
        assert result.n_features_checked == len(NAMES) < len(FEATURE_REGISTRY)
        assert result.safe and result.scope == "declared_and_empirical"
        assert set(result.screen) == set(NAMES) | {"mom_20__lag1"}

    def test_a_current_only_feature_outside_the_dataset_no_longer_fails_it(
        self, aliased, monkeypatch
    ):
        """The default used to run the declared check over the whole
        registry, so a CURRENT_ONLY feature this dataset never touched made
        it unsafe."""
        from standard_quant_tools.modeling.features.base import TemporalSupport

        monkeypatch.setitem(
            FEATURE_REGISTRY,
            "test.current_only",
            FEATURE_REGISTRY["technical.rsi"].model_copy(
                update={
                    "id": "test.current_only",
                    "temporal_support": TemporalSupport.CURRENT_ONLY,
                }
            ),
        )
        assert check_leakage(CheckLeakageInput(dataset_id=aliased)).safe
        assert not check_leakage(CheckLeakageInput()).safe

    def test_an_unknown_name_names_the_datasets_columns(self, aliased):
        result = check_leakage(
            CheckLeakageInput(dataset_id=aliased, feature_ids=["rsi_14", "nope"])
        )
        assert not result.safe
        (finding,) = result.findings
        assert finding.feature_id == "nope"
        assert "not a column of dataset" in finding.problem
        assert "rsi_14" in finding.problem

    def test_a_catalog_feature_the_dataset_lacks_is_declared_only(self, aliased):
        result = check_leakage(
            CheckLeakageInput(
                dataset_id=aliased, feature_ids=["rsi_14", "risk.atr_pct"]
            )
        )
        assert result.safe
        assert set(result.screen) == {"rsi_14"}
        assert any("risk.atr_pct" in note for note in result.notes)

    def test_an_external_panel_is_screened_only(self, tmp_path):
        dataset_id = _external(tmp_path)
        result = check_leakage(CheckLeakageInput(dataset_id=dataset_id))
        assert result.scope == "empirical_only"
        assert result.n_features_checked == 2
        assert set(result.screen) == {"alpha", "noise"}
        assert not any("feature registry" in f.problem for f in result.findings)
        assert any("declared check does not apply" in n for n in result.notes)

    def test_without_a_dataset_nothing_changes(self):
        result = check_leakage(CheckLeakageInput())
        assert result.n_features_checked == len(FEATURE_REGISTRY)
        assert result.scope == "declared_temporal_support_only"
        named = check_leakage(CheckLeakageInput(feature_ids=["mom_20"]))
        assert not named.safe
        assert "not in the feature registry" in named.findings[0].problem


# ── inspect_dataset ──────────────────────────────────────────────────────


class TestInspectDataset:
    def test_it_reports_each_columns_spec_and_the_build(self, aliased):
        out = inspect_dataset(InspectDatasetInput(dataset_id=aliased))
        assert out.storage == "built"
        assert [c.name for c in out.columns] == NAMES
        mom_20 = out.columns[1]
        assert (mom_20.id, mom_20.params, mom_20.alias, mom_20.lags) == (
            "market.momentum",
            {"lookback": 20},
            "mom_20",
            [1],
        )
        assert mom_20.lag_columns == ["mom_20__lag1"]
        assert out.columns[4].id == out.columns[4].name == "risk.realized_volatility"
        assert out.horizon == 5 and out.target["horizon"] == 5
        assert out.universe == ["AAA", "BBB", "CCC"]
        assert sorted(out.entities) == ["AAA", "BBB", "CCC"]
        assert out.rows > 0 and out.n_dates > 0
        assert out.start == "2022-01-01" and out.start_date > out.start
        assert out.interval == "1d" and out.benchmark == "SPY"
        assert out.missing["policy"] == "drop"
        assert "per_feature" in out.drop_attribution
        assert out.warnings
        assert out.spec_hash and out.data_hash

    def test_it_never_reads_the_panel(self, aliased):
        """The panel is not read or hashed: deleting it changes nothing."""
        from standard_quant_tools.modeling import artifacts

        before = inspect_dataset(InspectDatasetInput(dataset_id=aliased))
        os.remove(artifacts.run_dir(aliased) / "panel.parquet")
        assert inspect_dataset(InspectDatasetInput(dataset_id=aliased)) == before

    def test_an_edited_spec_is_refused(self, aliased):
        from standard_quant_tools.modeling import artifacts

        path = artifacts.run_dir(aliased) / "dataset_spec.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["features"][1]["params"]["lookback"] = 21
        path.write_text(json.dumps(spec), encoding="utf-8")
        with pytest.raises(ValidationError, match="no longer matches"):
            inspect_dataset(InspectDatasetInput(dataset_id=aliased))

    def test_an_external_panel(self, tmp_path):
        dataset_id = _external(tmp_path)
        out = inspect_dataset(InspectDatasetInput(dataset_id=dataset_id))
        assert out.storage == "external"
        assert [(c.name, c.id) for c in out.columns] == [
            ("alpha", "alpha"),
            ("noise", "noise"),
        ]
        assert out.source["path"].endswith("external.parquet")
        assert any("external panel" in note for note in out.notes)

    def test_an_unknown_dataset_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="ds_000000000000"):
            inspect_dataset(InspectDatasetInput(dataset_id="ds_000000000000"))

    def test_it_is_a_modeling_tool(self, aliased):
        out = modeling_dispatch("inspect_dataset", {"dataset_id": aliased})
        assert out["dataset_id"] == aliased
        json.dumps(out, allow_nan=False)
