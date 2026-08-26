"""
Whole-surface invariants: properties that must hold for all 157 tools.

Every other test file in this suite tests a tool, or a module. These test
the SURFACE — the properties that stop being true when someone adds a tool
and forgets a step, which is a failure mode no per-tool test can catch
because the tool they forgot to register is the tool nobody wrote a test
for.

The registration path for a new tool touches six places: a library function,
an input model, a result model, the runtime's `TOOL_DEFS` and
`TOOL_DISPATCH`, the facade's imports, and `agent.__all__`. Miss any one and
the tool half-exists — advertised but unroutable, or routable but invisible,
or present but with no output schema. Each of those is a distinct failure
and each has its own test below.

WHY A PARTITION AND NOT A COVER. A tool belonging to two runtimes would
dissolve the boundary at exactly the point it matters most: an agent scoped
to `research` could reach a `backtest` tool, and the scoping would become
advisory. `test_the_runtimes_partition_the_surface` is the one that would
catch a convenient duplication.
"""

from __future__ import annotations

import json

import pytest

MINIMUM_RUNTIME_SIZE = 8
PER_RUNTIME_CEILING = 73_728


@pytest.fixture(scope="module")
def runtimes():
    from standard_quant_tools.agent.runtimes import all_runtimes

    return all_runtimes()


@pytest.fixture(scope="module")
def catalog():
    from standard_quant_tools.mcp.catalog import build_catalog

    return build_catalog()


class TestTheSurfaceIsWellFormed:
    def test_the_runtimes_partition_the_surface(self, runtimes):
        """
        Every tool belongs to exactly ONE runtime.

        A tool in two runtimes would make the boundary advisory rather than
        hard, which is the one property the whole runtime design exists to
        provide. Duplicating a convenient tool into a second runtime is the
        obvious shortcut and this is what stops it.
        """
        owners: dict[str, str] = {}
        collisions = []
        for name, runtime in runtimes.items():
            for tool in runtime.dispatch_table:
                if tool in owners:
                    collisions.append(f"{tool}: {owners[tool]} and {name}")
                owners[tool] = name
        assert not collisions, f"tools in more than one runtime: {collisions}"

    def test_advertised_equals_dispatchable(self, runtimes):
        """
        A tool advertised without being dispatchable is a promise the server
        breaks on the first call. One dispatchable without being advertised
        is reachable only by guessing its name, which is the behaviour the
        scoping exists to prevent.
        """
        mismatches = []
        for name, runtime in runtimes.items():
            advertised = {d[0] for d in runtime.tool_defs}
            dispatchable = set(runtime.dispatch_table)
            if advertised != dispatchable:
                mismatches.append(f"{name}: {advertised ^ dispatchable}")
        assert not mismatches, mismatches

    def test_every_runtime_clears_the_floor(self, runtimes):
        """
        A runtime holding two tools is overhead rather than isolation. The
        floor applies on BOTH sides of a split, which is why `derivatives`
        and `microstructure` waited until they held twelve.
        """
        small = {
            n: len(rt) for n, rt in runtimes.items() if len(rt) < MINIMUM_RUNTIME_SIZE
        }
        assert not small, f"runtimes below the floor of {MINIMUM_RUNTIME_SIZE}: {small}"

    def test_every_tool_resolves_to_a_category_its_runtime_owns(self, runtimes):
        """
        The category taxonomy and the runtime boundary answer different
        questions and are allowed to differ — but a tool whose category its
        own runtime does not own is unreachable through `--categories`.

        Resolved the way the RUNTIME resolves it, not by reading
        `TOOL_CATEGORY` directly. `modeling` and `feature_lab` have no entry
        there on purpose: each is one surface rather than a taxonomy, so its
        tools answer with the runtime's own name. Checking the raw dict
        instead reports all 25 of them as uncategorised, which is what the
        first version of this test did.
        """
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        wrong = []
        for name, runtime in runtimes.items():
            for tool in runtime.dispatch_table:
                category = TOOL_CATEGORY.get(tool, name)
                if category not in runtime.categories:
                    wrong.append(f"{tool}: category {category!r} not in {name}")
        assert not wrong, wrong

    def test_no_category_is_owned_by_two_runtimes(self):
        """`--categories X` must resolve to one runtime, or scoping by
        category and scoping by runtime disagree."""
        from standard_quant_tools.agent.runtimes import RUNTIME_CATEGORIES

        seen: dict[str, str] = {}
        collisions = []
        for runtime, categories in RUNTIME_CATEGORIES.items():
            for category in categories:
                if category in seen:
                    collisions.append(f"{category}: {seen[category]} and {runtime}")
                seen[category] = runtime
        assert not collisions, collisions

    def test_every_tool_listed_by_get_tools_is_dispatchable(self, runtimes):
        """
        `get_tools()` filters by category and `dispatch()` does not. A tool
        that lists but cannot route is a promise broken on the first call.
        """
        for name, runtime in runtimes.items():
            listed = {t["function"]["name"] for t in runtime.get_tools()}
            assert listed <= set(runtime.dispatch_table), (
                f"{name} lists tools it cannot dispatch: "
                f"{listed - set(runtime.dispatch_table)}"
            )

    def test_every_tool_is_exported_from_the_facade(self):
        """
        The facade re-exports every tool by name, and a missing export means
        `from standard_quant_tools.agent import x` fails for a tool the
        registry says exists.
        """
        import standard_quant_tools.agent as package
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        missing = sorted(n for n in _TOOL_DISPATCH if n not in package.__all__)
        assert not missing, f"absent from agent.__all__: {missing}"

    def test_the_facade_dispatch_matches_the_runtime_union(self, runtimes):
        """The facade is the union of the analysis runtimes and only that.
        Growing to cover modeling would undo the separation without anybody
        deciding to."""
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        union = {t for rt in runtimes.values() for t in rt.dispatch_table}
        assert set(_TOOL_DISPATCH) <= union, (
            f"facade dispatches tools no runtime owns: "
            f"{set(_TOOL_DISPATCH) - union}"
        )


class TestEverySchemaIsUsable:
    def test_every_input_schema_is_strict_json(self, runtimes):
        """
        `allow_nan=False` is the point. A schema carrying NaN as a default
        serializes to a document a strict parser rejects, and the failure
        surfaces at the transport layer where it is hard to attribute.
        """
        broken = []
        for runtime in runtimes.values():
            for name, _description, model in runtime.tool_defs:
                try:
                    json.dumps(model.model_json_schema(), allow_nan=False)
                except (ValueError, TypeError) as exc:
                    broken.append(f"{name}: {exc}")
        assert not broken, broken

    def test_every_tool_declares_an_output_schema(self, catalog):
        """
        Without a usable return annotation the MCP server silently stops
        declaring structured output, and a client then receives JSON it has
        no schema for. Losing an annotation is a one-character change with
        no other symptom.
        """
        missing = sorted(n for n, e in catalog.items() if e.output_schema is None)
        assert not missing, f"no usable return annotation: {missing}"

    def test_every_input_model_forbids_unknown_arguments(self, runtimes):
        """
        Pydantic's default DROPS an unknown field. A hallucinated or
        misspelled argument then runs on defaults while the caller believes
        it configured something — the tool succeeds and answers a different
        question.
        """
        permissive = []
        for runtime in runtimes.values():
            for name, _description, model in runtime.tool_defs:
                if model.model_config.get("extra") != "forbid":
                    permissive.append(name)
        assert not permissive, (
            "input models that silently drop unknown arguments: " f"{permissive}"
        )

    def test_mutable_defaults_are_isolated_between_instances(self, runtimes):
        """
        Nine input models carry a list default — `indicators`, `parameters`,
        `ndcg_cutoffs`. In a plain dataclass that is the classic shared-state
        bug: one call appends and every later call sees it.

        Pydantic v2 deep-copies defaults per instance, so it is not a bug
        HERE — verified rather than assumed, because the first version of
        this test simply banned mutable defaults and failed on nine models
        that were fine. What is worth pinning is the isolation itself, which
        would break if a model were ever moved off Pydantic or given a
        shared default through some other route.
        """
        leaked = []
        for runtime in runtimes.values():
            for name, _description, model in runtime.tool_defs:
                mutable = [
                    f
                    for f, info in model.model_fields.items()
                    if isinstance(info.default, (list, dict, set))
                ]
                if not mutable:
                    continue
                first = model.model_construct()
                second = model.model_construct()
                for field in mutable:
                    if getattr(first, field) is getattr(second, field):
                        leaked.append(f"{name}.{field}")
        assert not leaked, (
            "mutable defaults SHARED between instances — one call's "
            f"mutation is visible to every later call: {leaked}"
        )


class TestScopingIsEnforced:
    @pytest.mark.parametrize(
        "runtime,foreign",
        [
            ("research", "run_sma_backtest"),
            ("backtest", "optimize_risk_parity"),
            ("portfolio", "get_option_greeks"),
            ("derivatives", "estimate_roll_spread"),
            ("microstructure", "get_bootstrap_interval"),
            ("meta", "run_monte_carlo_trade_paths"),
            ("feature_lab", "check_put_call_parity"),
        ],
    )
    def test_a_foreign_tool_is_refused_by_name(self, runtime, foreign):
        """
        The refusal must NAME the owning runtime. A bare "unknown tool"
        cannot be told apart from a hallucination, so a model that receives
        one guesses again rather than widening its scope.
        """
        from standard_quant_tools.agent.runtimes import resolve

        with pytest.raises(ValueError) as caught:
            resolve(runtime).dispatch(foreign, {})
        message = str(caught.value)
        assert foreign in message
        assert (
            "belongs to the" in message
        ), f"the refusal did not name the owner: {message[:160]}"

    def test_a_hallucinated_name_is_told_it_does_not_exist(self):
        """
        The other case, and it needs a different answer. Telling a model to
        widen its scope for a tool that exists nowhere sends it looking for
        a flag that cannot help.
        """
        from standard_quant_tools.agent.runtimes import resolve

        with pytest.raises(ValueError) as caught:
            resolve("research").dispatch("run_sma_backtestt", {})
        assert "does not exist" in str(caught.value)

    def test_a_moved_tool_says_where_it_went(self):
        """
        A split is a breaking change. Naming the new home turns it into an
        instruction rather than a dead end.
        """
        from standard_quant_tools.agent.runtimes import MOVED_FROM, resolve

        assert MOVED_FROM, "no moved tools recorded"
        for tool, previous in MOVED_FROM.items():
            with pytest.raises(ValueError) as caught:
                resolve(previous).dispatch(tool, {})
            assert "used to be in" in str(
                caught.value
            ), f"{tool} moved from {previous} but the refusal does not say so"


class TestTheContextBudgetHolds:
    def test_every_runtime_fits_the_ceiling_at_the_default_detail(self, runtimes):
        """
        This is the number a client actually pays. The ceiling has been
        argued up once already; the answer to exceeding it is `auto`
        thinning, not a larger constant.
        """
        from standard_quant_tools.mcp.catalog import categories_for_runtimes
        from standard_quant_tools.mcp.config import ServerConfig
        from standard_quant_tools.mcp.server import build_server

        over = {}
        for name in runtimes:
            config = ServerConfig(
                categories=categories_for_runtimes([name]),
                runtimes=(name,),
                enable_long_running=True,
            )
            _server, handlers = build_server(config)
            cost = handlers.context_bytes()
            if cost >= PER_RUNTIME_CEILING:
                over[name] = cost
        assert (
            not over
        ), f"over the {PER_RUNTIME_CEILING:,}-byte per-runtime ceiling: {over}"

    def test_thinning_keeps_every_schema_reachable(self, runtimes):
        """
        `auto` is only safe because `describe_tool` is injected whenever
        anything is thinned. Without that a thinned schema would be gone
        rather than one call away.
        """
        from standard_quant_tools.mcp.catalog import categories_for_runtimes
        from standard_quant_tools.mcp.config import ServerConfig
        from standard_quant_tools.mcp.server import build_server

        for name in runtimes:
            config = ServerConfig(
                categories=categories_for_runtimes([name]),
                runtimes=(name,),
                enable_long_running=True,
            )
            _server, handlers = build_server(config)
            if handlers.thinned:
                assert "describe_tool" in {t.name for t in handlers.tools}, (
                    f"{name} thinned {len(handlers.thinned)} tools without "
                    "injecting describe_tool"
                )

    def test_the_heaviest_runtime_is_a_fraction_of_the_whole(self, catalog):
        """The claim that makes scoping worth doing at all."""
        from standard_quant_tools.mcp.catalog import runtime_costs

        costs = runtime_costs(catalog)
        heaviest = max(size for _count, size in costs.values())
        total = sum(size for _count, size in costs.values())
        assert heaviest < total * 0.45, (
            "one runtime is now most of the surface, so scoping to it saves "
            "little and the boundary needs revisiting"
        )


class TestNoMutationEscapedIntoTheSource:
    """
    A guard added after a mutation was COMMITTED.

    `Development/mutation_testing.py` edits source files in place and
    restores them in a `finally`. It was running in the background when a
    `git add -A` swept the working tree, so `if False:` landed in a commit
    in place of the overflow bound in `analysis/options.py` -- and the file
    then looked clean to git, because the mutation WAS the committed state.

    The harness already refused to start on a dirty tree, which protects
    the developer's work from the harness. Nothing protected a commit from
    the harness. This does: the constant conditions it substitutes cannot
    appear in the source at all.

    It is a useful check independently. `if False:` in committed code is
    either dead code or a disabled guard, and neither should survive review.
    """

    #: What the mutation catalogue substitutes, plus the general smell.
    FORBIDDEN = ("if False:", "if True:", "if 0:", "if 1:")

    def test_no_constant_condition_appears_in_the_source(self):
        import re
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parent.parent.parent / "src"
        offenders = []
        for path in sorted(root.rglob("*.py")):
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for forbidden in self.FORBIDDEN:
                    if re.search(rf"(?<![\w.]){re.escape(forbidden)}", stripped):
                        offenders.append(f"{path.name}:{number}  {stripped[:60]}")
        assert not offenders, (
            "constant conditions in the source. Either dead code, a disabled "
            "guard, or a mutation from Development/mutation_testing.py that "
            "escaped into a commit -- run it with --restore. "
            + "; ".join(offenders)
        )

    def test_the_mutation_catalogue_anchors_all_still_match(self):
        """
        A mutation whose anchor has drifted tests NOTHING, and reports as
        SKIPPED rather than survived -- which is easy to read past. This
        fails instead, in the ordinary test run, so the catalogue cannot
        quietly stop covering the code it names.
        """
        import importlib.util
        from pathlib import Path as _Path

        harness = (
            _Path(__file__).resolve().parent.parent.parent
            / "Development"
            / "mutation_testing.py"
        )
        import sys

        spec = importlib.util.spec_from_file_location("mutation_testing", harness)
        assert spec and spec.loader, f"could not load {harness}"
        module = importlib.util.module_from_spec(spec)
        # REGISTERED BEFORE EXECUTION. The harness uses `@dataclass` under
        # `from __future__ import annotations`, so dataclasses resolves the
        # field annotations by looking the module up in `sys.modules` -- and
        # raises `AttributeError: 'NoneType' object has no attribute
        # '__dict__'` when it is not there yet.
        sys.modules["mutation_testing"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("mutation_testing", None)

        drifted = []
        for mutation in module.MUTATIONS:
            text = mutation.path.read_text(encoding="utf-8")
            found = text.count(mutation.old)
            if found != 1:
                drifted.append(f"{mutation.name} (anchor matched {found})")
        assert not drifted, (
            "mutation anchors no longer match the source, so those mutations "
            f"test nothing: {drifted}"
        )
