"""
The three discovery tools: list_strategies, list_stress_scenarios and
describe_data_capabilities.

These tools exist to stop a caller guessing, so the tests that matter are
the DRIFT tests -- each one asserts the tool reports the library's own data
structure rather than a copy of it. A test that merely checked "returns
eight strategies" would pass forever while the bounds it reports went
stale, which is the exact failure the tools were added to prevent.

Nothing here touches the network. That is itself part of the contract:
describe_data_capabilities answers "could I fetch ticks" without fetching
anything, so it stays callable when the thing it describes is unavailable.
"""

import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.backtest.strategy_params import (
    _MAX_WINDOW_BARS,
    _RELATIONS,
    STRATEGY_PARAM_SCHEMA,
)
from standard_quant_tools.backtest.stress_test import _SCENARIOS, scenario_dates
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.error import ValidationError


class TestListStrategies:
    def test_reports_every_registered_strategy(self):
        result = dispatch("list_strategies", {})
        names = {s["name"] for s in result["strategies"]}
        assert names == set(STRATEGY_REGISTRY)
        assert names == set(STRATEGY_PARAM_SCHEMA)

    def test_parameters_match_the_schema_exactly(self):
        """The drift guard. Every name, kind, default and bound comes from
        STRATEGY_PARAM_SCHEMA, so a schema change that this tool did not
        pick up fails here rather than misinforming a caller."""
        result = dispatch("list_strategies", {})
        for descriptor in result["strategies"]:
            schema = STRATEGY_PARAM_SCHEMA[descriptor["name"]]
            reported = {p["name"]: p for p in descriptor["parameters"]}
            assert set(reported) == set(schema)
            for name, spec in schema.items():
                assert reported[name]["kind"] == spec.kind
                assert reported[name]["default"] == spec.default
                if spec.kind == "number":
                    assert reported[name]["minimum"] == spec.minimum
                    assert reported[name]["maximum"] == spec.maximum

    def test_window_parameters_report_the_look_ahead_floor(self):
        """A window's real floor is 1, and the reason is not stylistic:
        pandas reads a negative period as a FORWARD window, so a caller who
        believes 0 or -20 is merely unusual would be writing look-ahead."""
        result = dispatch("list_strategies", {})
        windows = [
            p
            for s in result["strategies"]
            for p in s["parameters"]
            if p["kind"] == "window"
        ]
        assert windows, "no window parameters found — the fixture is wrong"
        for param in windows:
            assert param["minimum"] == 1.0
            assert param["maximum"] == float(_MAX_WINDOW_BARS)

    def test_relations_match_the_declared_ones(self):
        result = dispatch("list_strategies", {})
        for descriptor in result["strategies"]:
            expected = _RELATIONS.get(descriptor["name"], ())
            reported = descriptor["relations"]
            assert len(reported) == len(expected)
            for relation, (left, right, why) in zip(reported, expected):
                assert relation["left"] == left
                assert relation["right"] == right
                assert relation["requirement"] == f"{left} < {right}"
                assert relation["why"] == why

    def test_filtering_returns_exactly_one(self):
        result = dispatch("list_strategies", {"strategy_type": "sma_crossover"})
        assert [s["name"] for s in result["strategies"]] == ["sma_crossover"]

    def test_unknown_strategy_is_rejected_with_the_available_set(self):
        with pytest.raises(ValidationError) as exc:
            dispatch("list_strategies", {"strategy_type": "no_such_strategy"})
        assert "sma_crossover" in str(exc.value)

    def test_synthetic_labels_are_reported_separately(self):
        """buy_and_hold and custom_signal are accepted strategy_type values
        that take no parameters. Listing them among the eight would imply
        they have a parameter contract; omitting them entirely would leave
        a caller unable to explain why BacktestInput accepts them."""
        result = dispatch("list_strategies", {})
        assert set(result["synthetic_labels"]) == {"buy_and_hold", "custom_signal"}
        assert not set(result["synthetic_labels"]) & set(STRATEGY_PARAM_SCHEMA)


class TestListStressScenarios:
    def test_reports_every_scenario_the_stress_test_accepts(self):
        result = dispatch("list_stress_scenarios", {})
        assert {s["name"] for s in result["scenarios"]} == set(_SCENARIOS)

    def test_every_reported_window_is_the_real_one(self):
        for scenario in dispatch("list_stress_scenarios", {})["scenarios"]:
            start, end = scenario_dates(scenario["name"])
            assert (scenario["start"], scenario["end"]) == (start, end)
            assert scenario["calendar_days"] > 0


class TestDescribeDataCapabilities:
    def test_yfinance_declares_no_tick_feed(self):
        result = dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert result["available"] is True
        assert result["trades"] is False
        assert result["quotes"] is False
        assert any("not a substitute" in note for note in result["notes"])

    def test_capability_is_probed_by_override_not_by_calling(self):
        """DataProvider.get_trades raises by design, so calling it to find
        out is the failure this tool replaces. The probe must therefore
        agree with the class, and disagree with the base."""
        result = dispatch("describe_data_capabilities", {"source": "polygon"})
        from standard_quant_tools.data.polygon_provider import PolygonProvider

        assert PolygonProvider.get_trades is not DataProvider.get_trades
        assert result["trades"] is True
        assert result["quotes"] is True

    def test_an_unconfigured_provider_still_describes_its_class(self, monkeypatch):
        """A missing API key is a configuration state, not an error: the
        useful answer to 'can I get ticks' is 'yes, once you set a key',
        which requires reporting the class's capability alongside the
        reason it could not be constructed."""
        monkeypatch.delenv("SQT_POLYGON_API_KEY", raising=False)
        monkeypatch.setattr(
            "standard_quant_tools.data.polygon_provider._resolve_polygon_api_key",
            lambda key=None: (_ for _ in ()).throw(RuntimeError("no key configured")),
        )
        result = dispatch("describe_data_capabilities", {"source": "polygon"})
        assert result["available"] is False
        assert "no key configured" in result["unavailable_reason"]
        assert result["trades"] is True
        assert result["guarantees"] == {}

    def test_supported_intervals_come_from_the_provider(self):
        from standard_quant_tools.data.yfinance_provider import YFinanceProvider

        declared = YFinanceProvider.SUPPORTED_INTERVALS
        assert declared is not None, "the provider stopped declaring its intervals"
        result = dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert set(result["supported_intervals"]) == set(declared)

    def test_guarantees_are_the_providers_own_self_report(self):
        result = dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert result["guarantees"] == {
            "adjusted": True,
            "survivorship_free": False,
            "point_in_time": False,
        }

    def test_unknown_source_is_a_caller_error(self):
        with pytest.raises(ValidationError):
            dispatch("describe_data_capabilities", {"source": "not_a_provider"})

    def test_describing_a_provider_fetches_no_market_data(self, monkeypatch):
        def _explode(*args, **kwargs):
            raise AssertionError("describe_data_capabilities fetched market data")

        monkeypatch.setattr(
            "standard_quant_tools.data.yfinance_provider.YFinanceProvider.get_ohlcv",
            _explode,
        )
        dispatch("describe_data_capabilities", {"source": "yfinance"})
