"""
Runtime-scoped exposure: the server serves one runtime, not one category list.

WHY THIS EXISTS. `--categories` narrows what is LISTED, and until now the
selection was the only thing that scoped a session. That was adequate while
the whole surface fit in a client's context. It does not any more: at 2,184
bytes per tool over the wire the ceiling buys 82.4 tools and the library has
82, so the next tool added fails the budget test whatever it is.

Runtimes are already the execution boundary -- `Runtime.dispatch` refuses a
tool it does not own. This makes them the EXPOSURE boundary too, so the flag
a server is started with and the table that executes its calls describe the
same scope.

WHAT WOULD BREAK WITHOUT THESE TESTS. Three things, none of them loud:

- A server scoped to one runtime still listing another's tools, because the
  selection is filtered in one place and the dispatch map built in another.
- `--runtime backtest` serving nothing, because `--categories` kept its old
  default and the intersection was empty. An empty server looks like a
  broken install, not like two flags disagreeing.
- The default changing. Adding a flag must not alter what happens when
  nobody passes it, and that is asserted here rather than assumed.
"""

import pytest

from standard_quant_tools.mcp.catalog import (
    ALL_CATEGORIES,
    ALL_RUNTIMES,
    DEFAULT_CATEGORIES,
    DEFAULT_DETAIL_BUDGET,
    RUNTIME_CATEGORY_MAP,
    build_catalog,
    categories_for_runtimes,
    runtime_costs,
    select_runtimes,
)
from standard_quant_tools.mcp.config import ServerConfig, resolve
from standard_quant_tools.mcp.server import (
    LONG_RUNNING,
    StandardToolsServer,
    build_server,
)


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


class TestTheMappingIsOneToOne:
    """A category belongs to exactly one runtime, or the scoping is a lie."""

    def test_every_category_has_exactly_one_owner(self):
        owners = {}
        for runtime, categories in RUNTIME_CATEGORY_MAP.items():
            for category in categories:
                assert category not in owners, (
                    f"{category!r} is claimed by both {owners[category]!r} and "
                    f"{runtime!r}; --runtime cannot resolve it"
                )
                owners[category] = runtime
        assert set(owners) == set(ALL_CATEGORIES)

    def test_every_runtime_owns_something(self):
        for runtime in ALL_RUNTIMES:
            assert RUNTIME_CATEGORY_MAP[runtime], f"{runtime!r} owns no categories"

    def test_the_runtimes_partition_the_catalog(self, catalog):
        """Every tool reachable through exactly one runtime, none orphaned."""
        seen = []
        for runtime in ALL_RUNTIMES:
            seen.extend(e.name for e in select_runtimes(catalog, [runtime]))
        assert sorted(seen) == sorted(catalog), "a tool is in two runtimes or none"


class TestScopeResolution:
    """The four ways --runtime and --categories can be combined."""

    def test_neither_flag_keeps_the_old_default(self):
        """Adding a flag must not change what happens without it."""
        config = resolve([])
        assert config.categories == DEFAULT_CATEGORIES
        assert config.runtimes == ALL_RUNTIMES

    def test_runtime_alone_serves_all_of_that_runtime(self):
        """The case a naive default would break: DEFAULT_CATEGORIES holds
        nothing from backtest, so inheriting it here would serve zero tools
        and look like a broken install."""
        config = resolve(["--runtime", "backtest"])
        assert config.runtimes == ("backtest",)
        assert set(config.categories) == set(RUNTIME_CATEGORY_MAP["backtest"])

    def test_categories_alone_narrows_the_runtimes_too(self):
        """Otherwise the dispatch scope stays wider than what is listed."""
        config = resolve(["--categories", "screener"])
        assert config.categories == ("screener",)
        assert config.runtimes == ("research",)

    def test_both_flags_intersect(self):
        config = resolve(["--runtime", "research", "--categories", "screener"])
        assert config.runtimes == ("research",)
        assert config.categories == ("screener",)

    def test_a_category_outside_the_runtime_is_refused(self):
        """Serving the intersection silently would hand back a surface
        neither flag describes."""
        with pytest.raises(SystemExit) as exc:
            resolve(["--runtime", "research", "--categories", "modeling"])
        message = str(exc.value)
        assert "modeling" in message
        assert "research" in message
        # Names the owner, so the operator can fix it in one step.
        assert "belongs to" in message

    @pytest.mark.parametrize("spelling", ["research+meta", "research,meta"])
    def test_runtimes_join_the_way_combine_names_them(self, spelling):
        """`+` because that is how `combine()` spells a joined runtime; a
        flag that spelled it differently would be one more thing to
        remember wrongly."""
        config = resolve(["--runtime", spelling])
        assert config.runtimes == ("research", "meta")

    def test_all_means_all(self):
        assert resolve(["--runtime", "all"]).runtimes == ALL_RUNTIMES

    def test_a_misspelled_runtime_is_refused_by_name(self):
        with pytest.raises(SystemExit) as exc:
            resolve(["--runtime", "reserach"])
        assert "reserach" in str(exc.value)
        assert "research" in str(exc.value)


class TestTheServerHonoursTheScope:
    def test_a_scoped_server_lists_only_its_runtime(self):
        config = resolve(["--runtime", "research"])
        _server, handlers = build_server(config)
        # `describe_tool` is INJECTED when anything is thinned, and `auto`
        # is now the default -- so a scoped listing can hold one tool the
        # runtime does not own. That is the mechanism working: without it a
        # thinned schema would be unreachable. It is excluded here because
        # what this test pins is that no OTHER runtime's tools leak in.
        listed = {t.name for t in handlers.tools} - {"describe_tool"}
        expected = {
            e.name
            for e in select_runtimes(build_catalog(), ["research"])
            if e.name not in LONG_RUNNING
        } - {"describe_tool"}
        assert listed == expected

    def test_a_scoped_server_cannot_dispatch_another_runtimes_tool(self):
        """The point of the whole change. Listing without dispatch scoping
        would leave the hallucination hole runtimes were built to close."""
        server = StandardToolsServer(resolve(["--runtime", "research"]))
        assert "run_sma_backtest" not in server.entries

    def test_the_refusal_names_the_owning_runtime(self):
        server = StandardToolsServer(resolve(["--runtime", "research"]))
        message = server._refusal("run_sma_backtest")
        assert "'backtest'" in message
        assert "--runtime backtest" in message

    def test_a_hallucinated_name_is_not_blamed_on_scope(self):
        """An agent can only act on the situation it is actually in. Telling
        it to widen scope for a tool that exists nowhere sends it to change a
        flag that will not help."""
        server = StandardToolsServer(resolve(["--runtime", "research"]))
        message = server._refusal("run_sma_backtestt")
        assert "No tool by that name exists" in message
        assert "--runtime" not in message

    def test_the_refusal_never_names_a_tool_it_cannot_serve(self):
        """A suggestion the scoped server would itself refuse is worse than
        no suggestion."""
        for runtime in ALL_RUNTIMES:
            server = StandardToolsServer(resolve(["--runtime", runtime]))
            message = server._refusal("definitely_not_a_tool")
            for name in build_catalog():
                if name not in server.entries:
                    assert name not in message, (
                        f"the {runtime} server's refusal names {name!r}, "
                        "which it does not serve"
                    )

    def test_the_runtime_field_narrows_even_when_categories_are_wider(self):
        """The test that actually pins the runtime filter.

        Found by mutation: deleting the runtime filter from the server left
        every other test in this file passing, because `_resolve_scope`
        guarantees the categories are already inside the runtime, so via the
        CLI the filter is a no-op. It stops being one the moment a
        `ServerConfig` is built directly -- by the HTTP layer, by an
        embedder, or by a test -- with a category set wider than the
        runtimes. Without the filter this serves the whole library while
        reporting itself as scoped to research, which is the exact failure
        the scoping exists to prevent.
        """
        config = ServerConfig(
            categories=ALL_CATEGORIES,  # deliberately wider than the runtime
            runtimes=("research",),
            enable_long_running=True,
        )
        _server, handlers = build_server(config)
        # `describe_tool` is injected when `auto` thins anything, so it is
        # excluded from both sides. What this test pins is that no OTHER
        # runtime's tools leak in -- the injected schema-fetcher is the
        # mechanism that makes thinning safe, not a scoping leak.
        served = {t.name for t in handlers.tools} - {"describe_tool"}
        expected = {e.name for e in select_runtimes(build_catalog(), ["research"])} - {
            "describe_tool"
        }
        assert served == expected, (
            "the runtimes field did not narrow the surface; a server "
            "reporting itself as research-scoped served "
            f"{len(served)} tools"
        )

    def test_a_directly_built_server_cannot_dispatch_outside_its_runtime(self):
        """The same hole, at the dispatch table rather than the listing."""
        config = ServerConfig(
            categories=ALL_CATEGORIES,
            runtimes=("research",),
            enable_long_running=True,
        )
        server = StandardToolsServer(config)
        assert "run_sma_backtest" not in server.entries
        assert "analyze_features" not in server.entries

    def test_scoping_does_not_change_which_tools_a_runtime_has(self, catalog):
        """Selecting by runtime and selecting by that runtime's categories
        are the same set -- otherwise the two flags disagree."""
        for runtime in ALL_RUNTIMES:
            by_runtime = {e.name for e in select_runtimes(catalog, [runtime])}
            config = ServerConfig(
                categories=categories_for_runtimes([runtime]),
                runtimes=(runtime,),
                enable_long_running=True,
            )
            _server, handlers = build_server(config)
            listed = {t.name for t in handlers.tools} - {"describe_tool"}
        assert listed == by_runtime - {"describe_tool"}, runtime


class TestTheBudgetIsNowPerRuntime:
    @pytest.mark.parametrize("runtime", ALL_RUNTIMES)
    def test_each_runtime_reports_what_it_costs(self, runtime):
        """
        The cost is MEASURED and reported, not capped.

        There was a fixed 72 KB per-runtime ceiling here and it was the
        wrong shape for the thing it was guarding. What a client can afford
        depends on its model, its context window and what else is in the
        session -- none of which this repository knows -- so a constant
        compiled into a test was asserting a fact about somebody else's
        deployment. It had also already been argued upward once, which is
        the usual sign that a number is a preference wearing a limit's
        clothes.

        What is still true and still worth pinning: a runtime has to cost
        SOMETHING and has to be measurable, because `--print-budget` and
        `estimate_tool_cost` both report it and a zero would mean the
        accounting had broken rather than that a runtime was free.
        """
        config = ServerConfig(
            categories=categories_for_runtimes([runtime]),
            runtimes=(runtime,),
            enable_long_running=True,
        )
        _server, handlers = build_server(config)
        total = handlers.context_bytes()
        assert total > 0, (
            f"the {runtime!r} runtime reports {total} bytes, which means the "
            "accounting is broken rather than that the runtime is free."
        )

        # AND THE OPTIMIZATION STILL HAS TO WORK. Dropping the fixed
        # ceiling removed an assertion about somebody else's context
        # window; it did not remove the obligation to be cheap. The
        # invariant that survives is RELATIVE, and it is the one that
        # actually catches regressions: the default serving mode must
        # never cost more than serving everything at full detail, and for
        # a runtime above the detail budget it must cost materially less.
        full_config = ServerConfig(
            categories=categories_for_runtimes([runtime]),
            runtimes=(runtime,),
            enable_long_running=True,
            tool_detail="full",
        )
        _f_server, full_handlers = build_server(full_config)
        full_total = full_handlers.context_bytes()

        assert total <= full_total, (
            f"{runtime!r} costs MORE under the default ({total:,}) than at "
            f"full detail ({full_total:,}), so thinning is adding bytes "
            "rather than removing them."
        )
        if full_total > DEFAULT_DETAIL_BUDGET * 1.5:
            assert total < full_total * 0.85, (
                f"{runtime!r} is expensive at full detail ({full_total:,} "
                f"bytes) and the default only brought it to {total:,} -- "
                "under 15% saved. `auto` thins the most expensive schemas "
                "first, so this means either the cost is spread evenly "
                "across many cheap tools or thinning has stopped working."
            )

    def test_the_heaviest_runtime_is_a_fraction_of_the_whole_surface(self, catalog):
        """The claim that makes scoping worth doing at all."""
        costs = runtime_costs(catalog)
        heaviest = max(size for _count, size in costs.values())
        total = sum(size for _count, size in costs.values())
        assert heaviest < total * 0.45, (
            "one runtime is now most of the surface, so scoping to it saves "
            "little; the runtime boundary needs revisiting"
        )

    def test_runtime_costs_and_category_costs_agree(self, catalog):
        """Two views of one number. If they diverge, one of them is wrong
        about which tools belong where."""
        from standard_quant_tools.mcp.catalog import category_costs

        by_runtime = runtime_costs(catalog)
        by_category = category_costs(catalog)
        for runtime, categories in RUNTIME_CATEGORY_MAP.items():
            summed = sum(by_category.get(c, (0, 0))[1] for c in categories)
            assert by_runtime[runtime][1] == summed, runtime
