"""
`idempotentHint` says what a repeated call does, for every tool.

MCP's hint asks whether calling a tool again with the same arguments has any
further effect. The catalog derived it from one shape only -- a result field
ending in `_uri`, a file saved per call -- and so advertised 32 tools as
idempotent that publish a reference at the caller's `run_id`. Those are the
opposite of idempotent: `handoff.publish` never overwrites by default, so the
second identical call is refused ("a value is already published at
run_id=..."). A client that trusted the hint and retried got a refusal. See
the CHANGELOG entry of 2026-10-22.

The two tools that register a dataset fell through both shapes: they take no
`run_id` and save no `*_uri`, but each call mints a dataset id and writes a
dataset under it, so a repeat succeeds with a second dataset. A result
naming a dataset or model id the input does not take is the third shape.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.tools import _TOOL_DISPATCH
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.mcp.catalog import ALL_CATEGORIES, build_catalog
from standard_quant_tools.mcp.config import ServerConfig
from standard_quant_tools.mcp.server import build_server
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent import (
    MODELING_TOOL_DISPATCH,
    BuildModelDatasetInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent.feature_tools import FEATURE_TOOL_DISPATCH
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import register_external_panel
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec
from tests.modeling.conftest import make_ohlcv, make_provider_mock

#: The tools whose hint is false because each takes a `run_id` and answers
#: with the reference it published there.
PUBLISH_AT_RUN_ID = frozenset(
    {
        "analyze_basis_history",
        "attach_model_outcomes",
        "backtest_model_signal",
        "build_continuous_futures_series",
        "build_data_bundle",
        "build_model_ensemble",
        "calculate_series_metrics",
        "classify_trade_direction",
        "compute_indicator_panel",
        "construct_weights_from_scores",
        "convert_reference",
        "detect_regimes",
        "estimate_covariance",
        "fetch_ohlcv",
        "fetch_ohlcv_panel",
        "fetch_order_book",
        "fetch_order_events",
        "fetch_quote_panel",
        "fetch_returns_panel",
        "fetch_tick_tape",
        "get_amihud_illiquidity",
        "get_effective_spread_series",
        "get_quoted_spread_series",
        "get_sharpe_stability",
        "register_external_dataset",
        "run_futures_backtest",
        "run_garch_volatility_forecast",
        "run_kalman_hedge_ratio",
        "run_pair_trade_backtest",
        "run_pca_analysis",
        "run_portfolio_simulation",
        "run_signal_panel_backtest",
        "run_walk_forward_backtest",
    }
)

#: Tools that take a `run_id` to NARROW what they read, not to publish.
RUN_ID_AS_A_FILTER = frozenset({"list_artifacts"})

#: The tools whose hint changed from true to false because each registers
#: a dataset under an id it mints: the same arguments twice succeed with two
#: datasets in the store. They take no `run_id` and save no `*_uri`, so
#: neither earlier shape saw them.
MINT_A_NEW_DATASET = frozenset({"build_model_dataset", "register_external_panel"})


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def _dispatch_tables():
    """Every runtime's dispatch table, as the tools are actually registered."""
    for table in (_TOOL_DISPATCH, MODELING_TOOL_DISPATCH, FEATURE_TOOL_DISPATCH):
        for name, entry in table.items():
            fn, model = entry if isinstance(entry, tuple) else (entry, None)
            if model is None:
                model = typing.get_type_hints(fn)["input_data"]
            yield name, model


class TestTheHintFollowsWhatACallWrites:
    def test_the_33_publishers_are_no_longer_advertised_idempotent(self, catalog):
        """The pinned list: every one of them was `true` before, but for
        estimate_covariance, which took no run_id until it learned to publish
        its matrix as a `covariance` reference."""
        assert len(PUBLISH_AT_RUN_ID) == 33
        wrong = sorted(n for n in PUBLISH_AT_RUN_ID if catalog[n].idempotent)
        assert not wrong, f"advertised idempotent but refuse a repeat: {wrong}"

    def test_the_server_sends_false_for_every_one_of_them(self):
        """The hint a client reads, not only the catalog's property."""
        _server, handlers = build_server(
            ServerConfig(
                categories=ALL_CATEGORIES,
                enable_long_running=True,
                tool_detail="full",
            )
        )
        sent = {t.name: t.annotations.idempotent_hint for t in handlers.tools}
        assert {n: sent[n] for n in PUBLISH_AT_RUN_ID} == {
            n: False for n in PUBLISH_AT_RUN_ID
        }

    def test_every_tool_taking_a_run_id_is_held_to_the_rule(self, catalog):
        """
        Read off the dispatch tables rather than the catalog: every
        registered tool whose input declares `run_id` publishes there --
        and so is not idempotent -- unless it is a known filter. A new tool
        taking a `run_id` whose result carries no reference fails here,
        where an author can decide which it is, instead of being
        advertised as idempotent by default.
        """
        takes_run_id = {
            name for name, model in _dispatch_tables() if "run_id" in model.model_fields
        }
        assert RUN_ID_AS_A_FILTER <= takes_run_id
        misread = sorted(
            name
            for name in takes_run_id
            if catalog[name].idempotent != (name in RUN_ID_AS_A_FILTER)
        )
        assert not misread, (
            f"{misread}: a run_id-taking tool whose hint disagrees with what "
            "it does. A publisher must advertise idempotentHint=false; a "
            "tool that only filters by run_id belongs in RUN_ID_AS_A_FILTER."
        )

    def test_a_reference_the_caller_passes_in_is_not_a_write(self, catalog):
        """The other half of the rule: describing or reading a reference
        the caller already holds writes nothing."""
        for name in (
            "describe_reference",
            "read_reference",
            "describe_data_bundle",
            "validate_external_dataset",
            "list_artifacts",
        ):
            assert catalog[name].idempotent is True, name

    def test_files_saved_per_call_are_still_not_idempotent(self, catalog):
        for name in ("run_backtest_compact", "score_model", "join_point_in_time"):
            assert catalog[name].idempotent is False, name


def _minted_id_alone(entry) -> bool:
    """Whether a tool writes ONLY by the minted-id shape: a `*dataset_id`
    or `*model_id` in its result that its input does not take, with no
    `*_uri` and no `run_id` publication beside it."""
    fields = set((entry.output_schema or {}).get("properties", {}))
    inputs = set(entry.input_schema.get("properties", {}))
    minted = any(
        f.endswith(("dataset_id", "model_id")) and f not in inputs for f in fields
    )
    by_uri = any(f.endswith("_uri") for f in fields)
    by_run_id = "run_id" in inputs and any(
        f in ("ref", "refs") or f.endswith(("_ref", "_refs")) for f in fields
    )
    return minted and not by_uri and not by_run_id


class TestAMintedDatasetIsAWrite:
    def test_the_two_registrars_are_no_longer_advertised_idempotent(self, catalog):
        """Both were `true`: neither takes a `run_id` nor saves a `*_uri`."""
        wrong = sorted(n for n in MINT_A_NEW_DATASET if catalog[n].idempotent)
        assert not wrong, f"advertised idempotent but add a dataset: {wrong}"

    def test_the_minted_id_changes_no_other_tool(self, catalog):
        """The new shape decides these two and nothing else: every other
        tool answering with a dataset or model id it was not given --
        run_model_experiment, join_point_in_time -- already saved a file."""
        alone = {n for n, entry in catalog.items() if _minted_id_alone(entry)}
        assert alone == MINT_A_NEW_DATASET

    def test_an_id_the_caller_passes_in_is_read_not_minted(self, catalog):
        """`inspect_dataset` and `inspect_model` take the id they answer
        with: reading a dataset writes nothing."""
        for name in ("inspect_dataset", "inspect_model"):
            assert catalog[name].idempotent is True, name

    def test_the_server_sends_false_for_both(self):
        _server, handlers = build_server(
            ServerConfig(
                categories=ALL_CATEGORIES,
                enable_long_running=True,
                tool_detail="full",
            )
        )
        sent = {t.name: t.annotations.idempotent_hint for t in handlers.tools}
        assert {n: sent[n] for n in MINT_A_NEW_DATASET} == {
            n: False for n in MINT_A_NEW_DATASET
        }


class TestARepeatIsActuallyRefused:
    """The hint is a claim about behaviour, so the behaviour is checked."""

    @pytest.fixture
    def runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        return tmp_path

    def test_the_same_publishing_call_twice_is_refused(self, runs_dir, catalog):
        returns = np.random.default_rng(3).normal(0.0004, 0.01, 300).tolist()
        args = {
            "returns": returns,
            "window": 63,
            "run_id": "repeat_check",
            "name": "rolling",
        }
        research = resolve("research")
        first = research.dispatch("get_sharpe_stability", dict(args))
        assert first["rolling_sharpe_ref"] == (
            "sqt://analytic_series/repeat_check/rolling"
        )
        with pytest.raises(ValidationError, match="already published"):
            research.dispatch("get_sharpe_stability", dict(args))
        assert catalog["get_sharpe_stability"].idempotent is False


class TestARepeatRegistersAnotherDataset:
    """The two registrars' hint is a claim about behaviour too: the same
    arguments twice are not refused and do not answer with the same
    dataset -- each call writes a dataset of its own to the store."""

    @pytest.fixture
    def runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        return tmp_path

    @staticmethod
    def _stored(dataset_id: str) -> bool:
        return (_artifacts.run_dir(dataset_id) / "dataset_meta.json").is_file()

    def test_register_external_panel_twice_is_two_datasets(self, runs_dir, catalog):
        dates = pd.bdate_range("2024-01-02", periods=60)
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": dates,
                        "entity": entity,
                        "alpha": np.linspace(0.0, 1.0, 60) * (i + 1),
                        "target": np.linspace(0.0, 0.01, 60),
                    }
                )
                for i, entity in enumerate(("AAA", "BBB"))
            ],
            ignore_index=True,
        )
        path = runs_dir / "panel.parquet"
        frame.to_parquet(path, index=False)
        args = RegisterExternalPanelInput(path=str(path), horizon=5)
        first = register_external_panel(args)
        second = register_external_panel(args.model_copy())
        assert first.dataset_id != second.dataset_id
        assert self._stored(first.dataset_id) and self._stored(second.dataset_id)
        assert catalog["register_external_panel"].idempotent is False

    def test_build_model_dataset_twice_is_two_datasets(
        self, runs_dir, catalog, monkeypatch
    ):
        provider = make_provider_mock(make_ohlcv)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
        args = BuildModelDatasetInput(
            spec=DatasetSpec(
                universe=["AAA", "BBB"],
                start="2022-01-01",
                end="2023-12-31",
                features=[FeatureSpec(id="market.momentum", params={"lookback": 20})],
                target=TargetSpec(type="forward_return", horizon=5),
            )
        )
        first = build_model_dataset(args)
        second = build_model_dataset(args.model_copy())
        assert first.dataset_id != second.dataset_id
        assert self._stored(first.dataset_id) and self._stored(second.dataset_id)
        assert catalog["build_model_dataset"].idempotent is False
