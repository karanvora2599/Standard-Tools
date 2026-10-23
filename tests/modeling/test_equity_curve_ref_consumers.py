"""
What `EvaluateModelPortfolioResult.equity_curve_ref` says takes it, held to
the tools' own input models.

The description once said the reference "is what every reference-taking
tool accepts: get_deflated_sharpe_ratio, the PBO procedure, the reality
check, a re-simulation under different costs". None of those took a
reference then, and none takes an equity curve now: it is levels, and the
route is convert_reference to `returns_panel`. Since the CHANGELOG entry of
2026-10-22 the description names that route, the two tools whose series is
a `DataSource`, and the three validation tools that take the converted
panel as `returns_ref`. These tests fail the day any half of it stops being
true, so the description changes with the tools.
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

#: The tools it says take the converted reference as `returns_ref`.
TAKE_RETURNS_REF = (
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
    def test_it_says_the_curve_is_levels_and_names_the_conversion(self):
        assert "not returns" in DESCRIPTION
        assert "convert_reference" in DESCRIPTION
        assert ("equity_curve", "returns_panel") in CONVERSIONS
        assert "convert_reference" in _input_models()

    @pytest.mark.parametrize("name", TAKE_RETURNS_REF)
    def test_the_validation_tools_it_names_take_returns_ref(self, name):
        """Named in the description, and the field is on the model."""
        assert name in DESCRIPTION
        assert "returns_ref" in DESCRIPTION
        assert "returns_ref" in _input_models()[name].model_fields

    @pytest.mark.parametrize("name", TAKE_A_RETURNS_REFERENCE)
    def test_the_tools_it_names_do_take_a_returns_reference(self, name):
        assert name in DESCRIPTION
        assert _takes_a_reference(_input_models()[name])

    @pytest.mark.parametrize("name", TAKE_RETURNS_REF)
    def test_an_equity_curve_itself_is_refused_with_the_conversion(
        self, name, tmp_path, monkeypatch
    ):
        """The other half: the curve as it comes back is refused by each
        validation tool, naming both kinds and convert_reference."""
        import pandas as pd

        from standard_quant_tools.agent.runtimes import handoff, resolve
        from standard_quant_tools.error import ValidationError

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        dates = pd.bdate_range("2024-01-02", periods=60)
        curve = handoff.publish(
            pd.Series(range(100, 160), index=dates, dtype=float, name="equity"),
            kind="equity_curve",
            run_id="curve",
            name="eq",
        )
        arguments = {"returns_ref": curve}
        if name == "get_deflated_sharpe_ratio":
            arguments["n_trials"] = 2
        with pytest.raises(ValidationError) as refusal:
            resolve("backtest").dispatch(name, arguments)
        message = str(refusal.value)
        assert "'equity_curve'" in message and "'returns_panel'" in message
        assert "convert_reference" in message
