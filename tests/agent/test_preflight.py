"""
describe_tool, validate_tool_call, and the silent-argument hole they close.

The hole first. None of the 75 tool inputs forbade unknown arguments, so
this succeeded and ran at the default:

    BacktestInput(..., comission_pct=0.05)   # note the typo

The library already argues this exact case one layer down --
strategy_params.py exists because a hallucinated strategy parameter "was
silently swallowed and the strategy ran on its defaults, the caller
believing it had configured something it had not." The same hole was open
at the tool boundary above it, which is where a MODEL chooses the names.

Turning it on found two wrong tests in this repo immediately, both passing
arguments the tool never had. That is the failure mode: not a crash, a
quietly different run.
"""

import pytest

from standard_quant_tools.agent.models import BacktestInput
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.tools import _TOOL_DISPATCH
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH


@pytest.fixture
def meta():
    return resolve("meta")


class TestUnknownArgumentsAreRejected:
    def test_every_tool_input_forbids_extras(self):
        every = {**_TOOL_DISPATCH, **MODELING_TOOL_DISPATCH}
        permissive = [
            name
            for name, (_fn, model) in every.items()
            if model.model_config.get("extra") != "forbid"
        ]
        assert not permissive, (
            "these tools would silently ignore an argument they do not "
            f"take, and run on defaults instead: {permissive}"
        )

    def test_a_typo_is_an_error_not_a_default(self):
        """The concrete case: a misspelled commission ran the backtest at
        0.001 while the caller believed it had set 0.05."""
        with pytest.raises(Exception) as exc:
            BacktestInput(
                symbol="AAPL",
                start_date="2022-01-01",
                end_date="2023-01-01",
                strategy_type="sma_crossover",
                comission_pct=0.05,
            )
        assert "comission_pct" in str(exc.value)

    def test_correct_arguments_still_work(self):
        model = BacktestInput(
            symbol="AAPL",
            start_date="2022-01-01",
            end_date="2023-01-01",
            strategy_type="sma_crossover",
            commission_pct=0.05,
        )
        assert model.commission_pct == 0.05


class TestDescribeTool:
    def test_reports_the_contract_without_calling_anything(self, meta):
        result = meta.dispatch(
            "describe_tool", {"tool_name": "run_sma_backtest", "include_schema": False}
        )
        assert result["runtime"] == "backtest"
        assert set(result["required_arguments"]) >= {"symbol", "start_date", "end_date"}
        assert "sharpe_ratio" in result["result_fields"]
        assert result["reads_market_data"] is True

    def test_describing_is_not_calling_so_scope_does_not_apply(self, meta):
        """meta cannot RUN a backtest tool. It can still say what one
        takes — which is the answer to 'why was that refused'."""
        assert "run_sma_backtest" not in meta
        described = meta.dispatch("describe_tool", {"tool_name": "run_sma_backtest"})
        assert described["runtime"] == "backtest"

    def test_it_covers_the_modeling_runtime_too(self, meta):
        result = meta.dispatch("describe_tool", {"tool_name": "run_model_experiment"})
        assert result["runtime"] == "modeling"

    def test_omitting_the_schema_omits_the_bulk(self, meta):
        with_schema = meta.dispatch(
            "describe_tool", {"tool_name": "run_screener", "include_schema": True}
        )
        without = meta.dispatch(
            "describe_tool", {"tool_name": "run_screener", "include_schema": False}
        )
        assert with_schema["input_schema"] is not None
        assert without["input_schema"] is None

    def test_a_near_miss_gets_a_suggestion(self, meta):
        with pytest.raises(ValidationError) as exc:
            meta.dispatch("describe_tool", {"tool_name": "run_sma_backtestt"})
        assert "run_sma_backtest" in str(exc.value)


class TestValidateToolCall:
    def _check(self, meta, arguments, tool="run_sma_backtest"):
        return meta.dispatch(
            "validate_tool_call", {"tool_name": tool, "arguments": arguments}
        )

    def test_a_good_call_validates_and_shows_the_defaults(self, meta):
        result = self._check(
            meta,
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
            },
        )
        assert result["valid"] is True
        # The point of returning it: commission was never written down.
        assert result["normalized_arguments"]["commission_pct"] == 0.001

    def test_a_missing_argument_is_reported_as_missing(self, meta):
        result = self._check(meta, {"symbol": "AAPL"})
        kinds = {p["kind"] for p in result["problems"]}
        assert result["valid"] is False
        assert kinds == {"missing"}

    def test_an_unknown_argument_is_reported_as_unknown(self, meta):
        """The shape a hallucinated argument takes."""
        result = self._check(
            meta,
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "leverage": 3.0,
            },
        )
        assert result["valid"] is False
        assert any(
            p["kind"] == "unknown" and "leverage" in p["field"]
            for p in result["problems"]
        )

    def test_the_strategy_parameter_contract_is_checked_too(self, meta):
        """`parameters` is an open dict in the JSON schema, so a bad window
        passes schema validation and fails only after the data has been
        fetched. That round trip is what this saves."""
        result = self._check(
            meta,
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "parameters": {"lookback": -20},
            },
        )
        assert result["checked_strategy_parameters"] is True
        assert result["valid"] is False
        assert "unknown parameter" in result["problems"][0]["problem"]

    def test_a_cross_parameter_violation_is_caught(self, meta):
        result = self._check(
            meta,
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 30, "slow_period": 10},
            },
        )
        assert result["valid"] is False
        assert "must be <" in result["problems"][0]["problem"]

    def test_nothing_is_fetched_or_run(self, meta, monkeypatch):
        """A validator that executed would defeat its own purpose."""
        called = []
        monkeypatch.setattr(
            "standard_quant_tools.data.yfinance_provider.YFinanceProvider.get_ohlcv",
            lambda self, *a, **k: called.append(1),
        )
        self._check(
            meta,
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
            },
        )
        assert not called

    def test_it_validates_for_runtimes_it_cannot_execute(self, meta):
        result = self._check(
            meta,
            {"dataset_id": "ds_nope", "spec": {}},
            tool="run_model_experiment",
        )
        assert result["tool_name"] == "run_model_experiment"
        assert result["valid"] is False

    def test_an_unknown_tool_suggests_a_near_match(self, meta):
        with pytest.raises(ValidationError) as exc:
            meta.dispatch(
                "validate_tool_call",
                {"tool_name": "run_screner", "arguments": {}},
            )
        assert "run_screener" in str(exc.value)
