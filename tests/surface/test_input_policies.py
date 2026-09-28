"""
Two input rules the whole surface shares, pinned where they are shared.

A SEED IS A WHOLE NUMBER IN [0, 2**32 - 1]. Every sampler on the surface
takes one, and none bounded it: numpy refused -1 with "expected
non-negative integer" from inside the resampler, and the native Monte
Carlo failed to cast it with an error naming no argument. Nine tools
raised that way. One type, `agent.models.Seed`, now carries the range and
refuses a boolean, which int coercion would otherwise have run as seed 1.

A DATE-KEYED MAP IS KEYED BY DATES. `parse_date_keys` reads every key of
one as ISO 8601 and refuses the ones that are not by name, instead of
letting pandas' own parse error surface from the middle of a computation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import _build
from standard_quant_tools.agent.runtimes._shared import parse_date_keys, parse_iso_date
from standard_quant_tools.error import ValidationError

_SEED_NAMES = ("seed", "random_seed")


def _seed_fields():
    """(tool, input model, field) for every seed a tool input takes."""
    found = []
    for runtime in _build().values():
        for tool, (_fn, model) in sorted(runtime.dispatch_table.items()):
            for name in model.model_fields:
                if name in _SEED_NAMES:
                    found.append((tool, model, name))
    return found


SEED_FIELDS = _seed_fields()
_IDS = [f"{tool}.{name}" for tool, _model, name in SEED_FIELDS]


def _seed_errors(model, name, value):
    """The schema's complaints about `name` alone. The other required
    fields are left out on purpose, so their errors are filtered away."""
    try:
        model.model_validate({name: value})
    except PydanticValidationError as exc:
        return [e for e in exc.errors() if e["loc"] == (name,)]
    return []


class TestEverySeedIsTheSharedType:
    def test_the_scan_finds_every_sampler_that_raised(self):
        tools = {tool for tool, _model, _name in SEED_FIELDS}
        assert {
            "compare_against_random",
            "get_bootstrap_interval",
            "get_robustness_diagnostics",
            "run_monte_carlo_simulation",
            "run_monte_carlo_trade_paths",
            "run_reality_check",
            "run_terminal_monte_carlo",
            "simulate_delta_hedge",
            "compare_signals",
        } <= tools

    @pytest.mark.parametrize("tool,model,name", SEED_FIELDS, ids=_IDS)
    def test_the_schema_states_the_range(self, tool, model, name):
        spec = model.model_json_schema()["properties"][name]
        branches = spec.get("anyOf", [spec])
        integer = next(b for b in branches if b.get("type") == "integer")
        assert integer["minimum"] == 0
        assert integer["maximum"] == 2**32 - 1

    @pytest.mark.parametrize("tool,model,name", SEED_FIELDS, ids=_IDS)
    @pytest.mark.parametrize("bad", [-1, 2**32, True, False])
    def test_a_seed_outside_the_range_is_refused(self, tool, model, name, bad):
        assert _seed_errors(model, name, bad), f"{tool}.{name} accepted {bad!r}"

    @pytest.mark.parametrize("tool,model,name", SEED_FIELDS, ids=_IDS)
    @pytest.mark.parametrize("good", [0, 42, 2**32 - 1])
    def test_a_seed_inside_the_range_is_accepted(self, tool, model, name, good):
        assert _seed_errors(model, name, good) == []


def _returns(n: int, seed: int = 3) -> list:
    return np.random.default_rng(seed).normal(0.001, 0.01, n).tolist()


#: The samplers whose input is all inline, so the refusal and the null case
#: both run through dispatch without a fetch.
_INLINE_SAMPLERS = {
    "compare_against_random": lambda: {"trade_returns": _returns(40)},
    "run_monte_carlo_trade_paths": lambda: {"trade_returns": _returns(40)},
    "run_reality_check": lambda: {
        "strategy_returns": _returns(80, 1),
        "benchmark_returns": {"alt": _returns(80, 2)},
    },
    "get_bootstrap_interval": lambda: {"values": _returns(80)},
    "run_terminal_monte_carlo": lambda: {
        "returns": {"values": _returns(120)},
        "horizon_days": 5,
        "n_simulations": 200,
    },
    "simulate_delta_hedge": lambda: {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 0.25,
        "implied_vol": 0.2,
        "realized_vol": 0.25,
        "n_paths": 20,
    },
}


def _dispatch(tool, arguments):
    from standard_quant_tools.agent.tools import dispatch

    return dispatch(tool, arguments)


class TestANegativeSeedIsRefusedBeforeTheSamplerRuns:
    @pytest.mark.parametrize("tool", sorted(_INLINE_SAMPLERS))
    def test_minus_one_is_a_schema_refusal_naming_the_seed(self, tool):
        with pytest.raises(PydanticValidationError) as exc:
            _dispatch(tool, {**_INLINE_SAMPLERS[tool](), "seed": -1})
        assert any(e["loc"] == ("seed",) for e in exc.value.errors())

    @pytest.mark.parametrize("tool", sorted(_INLINE_SAMPLERS))
    def test_the_same_seed_repeats_the_answer(self, tool):
        """The null case: a legal seed still runs, and still reproduces."""
        first = _dispatch(tool, {**_INLINE_SAMPLERS[tool](), "seed": 7})
        second = _dispatch(tool, {**_INLINE_SAMPLERS[tool](), "seed": 7})
        assert first == second


class TestTheTerminalMonteCarloCapitalIsBounded:
    @pytest.mark.parametrize("capital", [math.inf, 1e308])
    def test_a_capital_no_account_holds_is_refused_by_the_schema(self, capital):
        """+inf used to reach the native kernel, which refused it with an
        untyped ValueError; 1e308 ran."""
        arguments = {**_INLINE_SAMPLERS["run_terminal_monte_carlo"](), "seed": 1}
        with pytest.raises(PydanticValidationError, match="initial_capital"):
            _dispatch(
                "run_terminal_monte_carlo", {**arguments, "initial_capital": capital}
            )


class TestDateKeysAreParsedByName:
    def test_iso_keys_come_back_as_timestamps_in_order(self):
        parsed = parse_date_keys(
            {"2024-01-03": 2.0, "2024-01-02T00:00:00": 1.0}, "prices", "a_tool"
        )
        assert list(parsed) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")]
        assert list(parsed.values()) == [2.0, 1.0]

    @pytest.mark.parametrize("bad", ["not-a-date", "2019-13-45", "AAPL", ""])
    def test_a_key_that_is_not_a_date_is_named(self, bad):
        with pytest.raises(ValidationError) as exc:
            parse_date_keys({"2024-01-02": 1.0, bad: 2.0}, "prices", "a_tool")
        message = str(exc.value)
        assert "a_tool" in message
        assert repr(bad) in message
        assert "not ISO dates" in message

    def test_a_map_keyed_by_tickers_is_refused(self):
        with pytest.raises(ValidationError, match="2 key"):
            parse_date_keys({"AAPL": 1.0, "MSFT": 2.0}, "signals", "a_tool")

    def test_two_spellings_of_one_date_are_refused(self):
        """Parsed, they are the same key, and one value would be dropped
        without a word."""
        with pytest.raises(ValidationError, match="same date twice"):
            parse_date_keys(
                {"2024-01-02": 1.0, "2024-01-02T00:00:00": 2.0}, "prices", "a_tool"
            )

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None])
    def test_finite_refuses_a_missing_value_by_date(self, bad):
        with pytest.raises(ValidationError, match="2024-01-03"):
            parse_date_keys(
                {"2024-01-02": 1.0, "2024-01-03": bad}, "prices", "a_tool", finite=True
            )

    def test_without_finite_a_gap_is_left_alone(self):
        parsed = parse_date_keys({"2024-01-02": math.nan}, "signals", "a_tool")
        assert math.isnan(parsed[pd.Timestamp("2024-01-02")])

    def test_an_empty_map_is_empty(self):
        assert parse_date_keys({}, "prices", "a_tool") == {}

    def test_one_date_parses_or_is_refused_by_field(self):
        assert parse_iso_date("2024-03-15", "expiry", "a_tool") == pd.Timestamp(
            "2024-03-15"
        )
        with pytest.raises(ValidationError, match="contracts\\[0\\].expiry"):
            parse_iso_date("not-a-date", "contracts[0].expiry", "a_tool")
