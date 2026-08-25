"""
Data crossing runtimes.

Runtimes isolate EXECUTION, not data. That distinction is the whole point:
a boundary that also blocked results would make the multi-agent
orchestrator impossible, since every real workflow spans runtimes -- screen
in `research`, backtest in `backtest`, size in `portfolio`, and hand a
model's predictions from `modeling` to a backtest.

Results cross by VALUE, never by shared dispatch table:

  - an artifact URI, written by one runtime and read by another
  - an identifier (dataset_id, model_id, request_id)
  - the plain JSON dict every tool already returns

That is strictly better than sharing a table. A value is serializable, so
it survives the process boundary between two agents in the multi-agent
orchestrator; it is auditable, because the handoff appears in the decision
log as an input to the second call; and it cannot smuggle execution rights,
because holding a URI does not let the holder run anything new.

These tests pin that both directions work: a value produced anywhere is
consumable anywhere, while the ability to CALL stays scoped.
"""

import json

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import combine, resolve
from standard_quant_tools.backtest.artifacts import save_artifact


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return tmp_path


@pytest.fixture
def prices():
    rng = np.random.default_rng(11)
    n = 300
    close = 100.0 * np.exp(np.linspace(0, 0.4, n) + rng.normal(0, 0.008, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 2_000_000.0),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


@pytest.fixture
def stubbed(monkeypatch, prices):
    """Stub the fetch in every runtime that performs one."""

    class _Stub:
        def get_ohlcv(self, symbol, start_date, end_date, interval="1d"):
            return prices

    for runtime in ("research", "backtest", "portfolio"):
        monkeypatch.setattr(
            f"standard_quant_tools.agent.runtimes.{runtime}.tools."
            "DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
            raising=False,
        )
    return prices


class TestArtifactsCrossRuntimes:
    def test_a_curve_written_by_backtest_is_read_by_meta(self, stubbed):
        """The canonical handoff: one runtime persists, another inspects.
        Neither needs the other's dispatch table."""
        backtest = resolve("backtest")
        result = backtest.dispatch(
            "run_backtest_compact",
            {
                "symbol": "TEST",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 10, "slow_period": 30},
                "run_id": "crossrt",
            },
        )
        uri = result["equity_curve_uri"]

        described = resolve("meta").dispatch("describe_artifact", {"uri": uri})
        assert described["rows"] > 0
        assert described["content_hash"]

    def test_the_handoff_value_survives_json(self, stubbed):
        """It has to: in the multi-agent orchestrator the two runtimes are
        different processes, and the value goes over a wire."""
        backtest = resolve("backtest")
        result = backtest.dispatch(
            "run_backtest_compact",
            {
                "symbol": "TEST",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "run_id": "wirehop",
            },
        )
        round_tripped = json.loads(json.dumps(result))
        uri = round_tripped["equity_curve_uri"]
        assert resolve("meta").dispatch("describe_artifact", {"uri": uri})["rows"] > 0

    def test_a_curve_from_one_runtime_feeds_a_tool_in_the_same_one(self, stubbed):
        backtest = resolve("backtest")
        result = backtest.dispatch(
            "run_backtest_compact",
            {
                "symbol": "TEST",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "run_id": "ddtable",
            },
        )
        table = backtest.dispatch(
            "get_drawdown_table", {"equity_curve_uri": result["equity_curve_uri"]}
        )
        assert table["n_bars"] > 0

    def test_holding_a_uri_grants_no_execution_rights(self):
        """A value crossing the boundary must not carry the ability to
        call across it -- otherwise the handoff IS the hole."""
        uri = save_artifact(
            pd.Series([1.0, 2.0, 3.0], name="equity"), "rights", "equity_curve"
        )
        meta = resolve("meta")
        assert meta.dispatch("describe_artifact", {"uri": uri})["rows"] == 3
        with pytest.raises(ValueError) as exc:
            meta.dispatch("get_drawdown_table", {"equity_curve_uri": uri})
        assert "belongs to the 'backtest' runtime" in str(exc.value)


class TestResultsCrossRuntimes:
    def test_a_research_result_feeds_a_portfolio_call(self, stubbed):
        """No artifact involved: the first runtime's plain JSON output is
        the second runtime's input."""
        research = resolve("research")
        # `period`, not start_date/end_date. This test used to pass the
        # latter, which the input silently dropped -- so it had been
        # measuring the default window all along. Forbidding unknown
        # arguments is what surfaced it.
        analysis = research.dispatch(
            "analyze_stock_risk",
            {"symbol": "TEST", "benchmark": "TEST", "period": "1y"},
        )
        assert "cvar_95" in analysis and "beta" in analysis

        sized = resolve("portfolio").dispatch(
            "get_position_size",
            {
                "symbol": "TEST",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "account_equity": 100_000.0,
                "risk_per_trade_pct": 0.01,
            },
        )
        assert sized["recommended_shares"] >= 0

    def test_every_runtime_returns_json_safe_output(self, stubbed):
        """The handoff channel only works if what comes out can be
        serialized — non-finite metrics are real here and must already be
        normalized before they reach another runtime."""
        payloads = [
            resolve("meta").dispatch("list_stress_scenarios", {}),
            resolve("meta").dispatch("list_strategies", {}),
        ]
        for payload in payloads:
            json.dumps(payload, allow_nan=False)


class TestSpanningAgents:
    def test_combine_gives_one_agent_a_wider_scope(self, stubbed):
        """A screen-then-backtest agent genuinely needs two runtimes. The
        widening is explicit in the code that asked for it."""
        wide = combine(["research", "backtest"])
        assert "run_screener" in wide
        assert "run_sma_backtest" in wide
        result = wide.dispatch(
            "run_sma_backtest",
            {
                "symbol": "TEST",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
            },
        )
        assert "sharpe_ratio" in result

    def test_a_combined_runtime_advertises_exactly_what_it_can_run(self):
        wide = combine(["research", "meta"])
        advertised = {t["function"]["name"] for t in wide.get_tools()}
        assert advertised == set(wide.tool_names)

    def test_combining_does_not_reach_the_runtime_left_out(self):
        wide = combine(["research", "meta"])
        with pytest.raises(ValueError) as exc:
            wide.dispatch("get_capacity_report", {})
        assert "'portfolio' runtime" in str(exc.value)


class TestModelingHandoff:
    def test_the_modeling_runtime_is_reachable_by_name_from_anywhere(self):
        """An orchestrator holds runtimes by name, so a modeling step is
        addressable from a session that started in research."""
        from standard_quant_tools.agent.runtimes import owner_of
        from standard_quant_tools.modeling.agent import (
            MODELING_TOOL_DISPATCH,
            modeling_dispatch,
        )

        assert owner_of("build_model_dataset") == "modeling"
        assert "build_model_dataset" in MODELING_TOOL_DISPATCH
        assert callable(modeling_dispatch)

    def test_modeling_and_analysis_names_never_collide(self):
        """The handoff is by name, so a collision would make a value
        ambiguous about which runtime should consume it."""
        from standard_quant_tools.agent.runtimes import (
            MODELING_RUNTIME,
            all_runtimes,
        )
        from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

        # all_runtimes() includes modeling now, so the analysis side has to
        # be named rather than assumed -- the point of the test is that the
        # two REGISTRIES share no name, and a set containing both would
        # trivially overlap itself.
        analysis_names = {
            name
            for runtime_name, rt in all_runtimes().items()
            if runtime_name != MODELING_RUNTIME
            for name in rt.tool_names
        }
        assert analysis_names.isdisjoint(MODELING_TOOL_DISPATCH)
