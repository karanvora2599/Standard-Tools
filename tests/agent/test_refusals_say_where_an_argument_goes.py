"""
A refused argument names where it belongs.

Every input model forbids an argument it does not take, and pydantic's text
for that was "Extra inputs are not permitted" -- for a missing one, "Field
required". In one live session seven first calls were refused that way, five
of them to a tool whose sibling takes exactly what was sent, and nothing in
the refusal pointed there. The dispatchers now build every input through
`build_input`, which keeps the exception and its structure and rewrites two
messages:

    an unknown argument   what the tool takes, the nearest argument name,
                          and for an id the tools that take it -- in the
                          same runtime when any do, else the runtimes
    a missing argument    the all-feature sibling of a single-feature tool,
                          and inspect_model for score_model

Everything that branches on the error -- its class, title, count, each
error's `type` and `loc` -- is unchanged, which the first test holds for
every tool on the surface. See the CHANGELOG entry of 2026-10-04.
"""

from __future__ import annotations

import re

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import _inputs, all_runtimes, owner_of
from standard_quant_tools.agent.runtimes._inputs import (
    EXTRA_HINTS,
    MISSING_HINTS,
    build_input,
)


def _every_tool():
    return [
        (runtime, tool, model)
        for runtime, rt in sorted(all_runtimes().items())
        for tool, (_fn, model) in sorted(rt.dispatch_table.items())
    ]


TOOLS = _every_tool()
_PROBE_KEYS = ("zz_not_an_argument", "dataset_id", "model_id", "symbol")


def _refusal(fn):
    with pytest.raises(pydantic.ValidationError) as caught:
        fn()
    return caught.value


def _shape(exc: pydantic.ValidationError):
    return (
        type(exc),
        exc.title,
        exc.error_count(),
        [(e["type"], e["loc"]) for e in exc.errors()],
    )


def _dispatch(tool, arguments):
    return all_runtimes()[owner_of(tool)].dispatch(tool, arguments)


def _message(exc, field):
    (error,) = [e for e in exc.errors() if e["loc"] == (field,)]
    return error["msg"]


class TestTheStructureIsUnchanged:
    @pytest.mark.parametrize(
        "runtime, tool, model", TOOLS, ids=[t for _r, t, _m in TOOLS]
    )
    def test_an_unknown_argument_keeps_every_type_and_loc(self, runtime, tool, model):
        """For every tool: the same exception class, title and errors, in
        the same order, as constructing the model directly."""
        for key in _PROBE_KEYS:
            if key in model.model_fields:
                continue
            arguments = {key: "x"}
            direct = _refusal(lambda: model(**arguments))
            built = _refusal(lambda: build_input(tool, model, arguments))
            assert _shape(built) == _shape(direct)
            assert key in _message(built, key)

    @pytest.mark.parametrize("tool, field", sorted(MISSING_HINTS))
    def test_a_missing_argument_keeps_every_type_and_loc(self, tool, field):
        model = all_runtimes()[owner_of(tool)].dispatch_table[tool][1]
        direct = _refusal(lambda: model())
        built = _refusal(lambda: build_input(tool, model, {}))
        assert _shape(built) == _shape(direct)
        assert _message(built, field).startswith("Field required")

    def test_an_error_with_nothing_to_add_is_the_original_object(self):
        model = all_runtimes()["data"].dispatch_table["get_dataset_metadata"][1]
        original = _refusal(lambda: model(symbol="AAPL", interval=5))
        assert _inputs.explain_refusal("get_dataset_metadata", model, original) is (
            original
        )

    def test_an_accepted_call_is_built_exactly_as_before(self):
        model = all_runtimes()["feature_lab"].dispatch_table["get_feature_drift"][1]
        arguments = {"dataset_id": "ds_a", "feature": "mom_20", "method": "pearson"}
        assert build_input("get_feature_drift", model, arguments) == model(**arguments)

    def test_a_fault_composing_a_hint_leaves_the_original_refusal(self, monkeypatch):
        def broken(_model):
            raise RuntimeError("hint composition failed")

        monkeypatch.setattr(_inputs, "_what_it_takes", broken)
        model = all_runtimes()["data"].dispatch_table["get_dataset_metadata"][1]
        direct = _refusal(lambda: model(dataset_id="ds_a"))
        built = _refusal(
            lambda: build_input("get_dataset_metadata", model, {"dataset_id": "ds_a"})
        )
        assert str(built) == str(direct)


class TestTheRefusalSaysWhereToGo:
    def test_a_dataset_id_sent_to_the_provider_metadata_tool(self):
        exc = _refusal(
            lambda: _dispatch("get_dataset_metadata", {"dataset_id": "ds_a"})
        )
        message = _message(exc, "dataset_id")
        assert "It takes symbol (optional: source, interval)." in message
        assert "inspect_dataset(dataset_id)" in message

    def test_a_model_id_sent_to_pbo_names_compare_models(self):
        exc = _refusal(
            lambda: _dispatch("estimate_backtest_overfitting", {"model_id": "mdl_a"})
        )
        message = _message(exc, "model_id")
        assert "two or more configurations" in message
        assert "compare_models" in message

    def test_an_id_no_tool_in_the_runtime_takes_names_the_runtimes_that_do(self):
        exc = _refusal(
            lambda: _dispatch(
                "run_walk_forward_backtest",
                {
                    "symbol": "AAPL",
                    "start_date": "2020-01-01",
                    "end_date": "2021-01-01",
                    "strategy": "sma_crossover",
                    "param_grid": {"fast": [5]},
                    "dataset_id": "ds_a",
                },
            )
        )
        assert _message(exc, "dataset_id").endswith(
            "No backtest tool takes dataset_id; the feature_lab and modeling "
            "runtimes have tools that do."
        )

    def test_an_id_other_tools_in_the_runtime_take_names_them(self):
        exc = _refusal(lambda: _dispatch("list_features", {"dataset_id": "ds_a"}))
        message = _message(exc, "dataset_id")
        assert "In the modeling runtime, dataset_id is taken by" in message
        assert re.search(r"and \d+ more\.$", message)

    @pytest.mark.parametrize(
        "tool, arguments, sent, meant",
        [
            (
                "score_model",
                {"model_id": "mdl_a", "as_of_date": "2026-09-30", "universe": ["A"]},
                "as_of_date",
                "as_of",
            ),
            ("fetch_ohlcv", {"symbols": ["AAPL"]}, "symbols", "symbol"),
        ],
    )
    def test_a_near_miss_names_the_argument_meant(self, tool, arguments, sent, meant):
        exc = _refusal(lambda: _dispatch(tool, arguments))
        assert f"Did you mean {meant}?" in _message(exc, sent)

    @pytest.mark.parametrize(
        "tool, sibling",
        [
            ("get_feature_drift", "screen_feature_stability"),
            ("get_feature_regime_stability", "screen_feature_stability"),
            ("run_feature_permutation_test", "screen_feature_significance"),
            ("get_feature_ic_decay", "analyze_features"),
            ("profile_feature", "analyze_features"),
        ],
    )
    def test_a_single_feature_tool_without_a_feature_names_its_screen(
        self, tool, sibling
    ):
        exc = _refusal(lambda: _dispatch(tool, {"dataset_id": "ds_a"}))
        assert sibling in _message(exc, "feature")

    def test_score_model_without_a_date_names_inspect_model_once(self):
        exc = _refusal(lambda: _dispatch("score_model", {"model_id": "mdl_a"}))
        assert "inspect_model(model_id)" in _message(exc, "as_of")
        assert _message(exc, "universe") == "Field required"

    def test_the_whole_tool_list_is_said_once(self):
        exc = _refusal(
            lambda: _dispatch("get_dataset_metadata", {"symbol": "A", "a": 1, "b": 2})
        )
        assert "It takes" in _message(exc, "a")
        assert "It takes" not in _message(exc, "b")


class TestEveryDispatcherCarriesIt:
    @pytest.mark.parametrize(
        "call",
        [
            "runtime",
            "facade",
            "modeling",
            "feature_lab",
            "validate_tool_call",
            "mcp",
        ],
    )
    def test_the_hint_reaches_the_caller(self, call):
        if call == "runtime":
            exc = _refusal(lambda: _dispatch("inspect_model", {"dataset_id": "ds_a"}))
            text = str(exc)
        elif call == "facade":
            from standard_quant_tools.agent.tools import dispatch

            text = str(
                _refusal(lambda: dispatch("list_strategies", {"dataset_id": "x"}))
            )
        elif call == "modeling":
            from standard_quant_tools.modeling.agent.dispatch import modeling_dispatch

            text = str(
                _refusal(
                    lambda: modeling_dispatch("inspect_model", {"dataset_id": "x"})
                )
            )
        elif call == "feature_lab":
            from standard_quant_tools.modeling.agent.feature_tools import (
                feature_dispatch,
            )

            text = str(
                _refusal(
                    lambda: feature_dispatch("get_feature_drift", {"dataset_id": "x"})
                )
            )
        elif call == "validate_tool_call":
            from standard_quant_tools.agent.runtimes.meta.tools import (
                ValidateToolCallInput,
                validate_tool_call,
            )

            result = validate_tool_call(
                ValidateToolCallInput(
                    tool_name="get_dataset_metadata", arguments={"dataset_id": "x"}
                )
            )
            (unknown,) = [p for p in result.problems if p.kind == "unknown"]
            text = unknown.problem
        else:
            import anyio
            import mcp.types as types

            from standard_quant_tools.mcp.config import ServerConfig
            from standard_quant_tools.mcp.server import build_server

            _server, handlers = build_server(
                ServerConfig(categories=("data",), runtimes=("data",))
            )

            async def go():
                return await handlers.call_tool(
                    None,
                    types.CallToolRequestParams(
                        name="get_dataset_metadata", arguments={"dataset_id": "x"}
                    ),
                )

            result = anyio.run(go)
            assert result.is_error is True
            text = result.content[0].text
        assert any(
            name in text
            for name in (
                "inspect_dataset",
                "screen_feature_stability",
                "No meta tool takes dataset_id",
            )
        ), text


class TestTheCuratedHintsAreReal:
    """A hint that names a renamed tool sends the caller to a wall, which is
    worse than the generic text it replaced."""

    @staticmethod
    def _vocabulary():
        tools, arguments = set(), set()
        for runtime in all_runtimes().values():
            for tool, (_fn, model) in runtime.dispatch_table.items():
                tools.add(tool)
                arguments.update(model.model_fields)
        return tools, arguments

    @pytest.mark.parametrize(
        "key, text",
        sorted({**EXTRA_HINTS, **MISSING_HINTS}.items()),
        ids=[f"{t}.{a}" for t, a in sorted({**EXTRA_HINTS, **MISSING_HINTS})],
    )
    def test_every_name_in_a_hint_is_a_tool_or_an_argument(self, key, text):
        tools, arguments = self._vocabulary()
        names = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text))
        assert names, text
        assert not names - tools - arguments, names - tools - arguments
        assert names & tools, f"{key} names no tool"

    @pytest.mark.parametrize("tool, argument", sorted(EXTRA_HINTS))
    def test_an_unknown_argument_hint_is_for_an_argument_the_tool_lacks(
        self, tool, argument
    ):
        model = all_runtimes()[owner_of(tool)].dispatch_table[tool][1]
        assert argument not in model.model_fields

    @pytest.mark.parametrize("tool, argument", sorted(MISSING_HINTS))
    def test_a_missing_argument_hint_is_for_a_required_argument(self, tool, argument):
        model = all_runtimes()[owner_of(tool)].dispatch_table[tool][1]
        assert model.model_fields[argument].is_required()
