"""
What `EvaluateModelPortfolioResult.equity_curve_ref` says takes it, held to
the tools' own input models.

The description used to say the reference "is what every reference-taking
tool accepts: get_deflated_sharpe_ratio, the PBO procedure, the reality
check, a re-simulation under different costs". None of those took a
reference: the three validation tools read returns inline, as lists, and no
tool re-simulates from an equity curve. An agent that believed the field
handed the reference to them and was refused. Since the CHANGELOG entry of
2026-10-22 the description names the route that exists -- convert_reference
to `returns_panel`, then a tool whose series is a `DataSource` -- and says
the validation tools take returns inline. These tests fail the day either
half stops being true, so the description changes with the tools.
"""

from __future__ import annotations

import typing
from typing import Any, Dict

import pytest

from standard_quant_tools.agent.runtimes import all_runtimes
from standard_quant_tools.agent.runtimes.data.models import DataSource
from standard_quant_tools.agent.runtimes.meta.convert import CONVERSIONS
from standard_quant_tools.modeling.agent.models import EvaluateModelPortfolioResult

DESCRIPTION = EvaluateModelPortfolioResult.model_fields["equity_curve_ref"].description

#: The tools the description says read returns inline and take no reference.
INLINE_ONLY = (
    "get_deflated_sharpe_ratio",
    "estimate_backtest_overfitting",
    "run_reality_check",
)

#: The tools it says take the converted reference as their series.
TAKE_A_RETURNS_REFERENCE = ("calculate_series_metrics", "run_terminal_monte_carlo")


def _input_models() -> Dict[str, Any]:
    return {
        name: model
        for runtime in all_runtimes().values()
        for name, _description, model in runtime.tool_defs
    }


def _mentions(annotation: Any, target: type) -> bool:
    if annotation is target:
        return True
    return any(_mentions(arg, target) for arg in typing.get_args(annotation))


def _takes_a_reference(model: Any) -> bool:
    """A field that is a `DataSource`, or one named as a reference."""
    return any(
        _mentions(info.annotation, DataSource)
        or name == "ref"
        or name.endswith("_ref")
        or name.endswith("_refs")
        for name, info in model.model_fields.items()
    )


class TestTheDescriptionIsTrue:
    def test_it_names_the_inline_only_tools_as_such(self):
        assert "do NOT take a reference" in DESCRIPTION
        tail = DESCRIPTION.split("do NOT take a reference")[0]
        for name in INLINE_ONLY:
            assert name in tail, f"{name} is not named among the inline-only tools"

    @pytest.mark.parametrize("name", INLINE_ONLY)
    def test_the_inline_only_tools_take_no_reference(self, name):
        model = _input_models()[name]
        assert not _takes_a_reference(model), (
            f"{name} now takes a reference; equity_curve_ref's description "
            "says it does not and must change with it"
        )

    @pytest.mark.parametrize("name", TAKE_A_RETURNS_REFERENCE)
    def test_the_tools_it_names_do_take_a_returns_reference(self, name):
        assert name in DESCRIPTION
        assert _takes_a_reference(_input_models()[name])

    def test_the_conversion_it_names_exists(self):
        assert "convert_reference" in DESCRIPTION
        assert ("equity_curve", "returns_panel") in CONVERSIONS
        assert "convert_reference" in _input_models()
