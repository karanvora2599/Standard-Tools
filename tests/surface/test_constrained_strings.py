"""
A string field that names its choices should not accept anything else.

Six fields on the surface branched on an exact string and fell through to a
default for everything else, so the argument was accepted, ignored, and
answered with a different computation:

    cross_sectional_ic(method='spearman')   mean IC = 1.0000
    cross_sectional_ic(method='SPEARMAN')   mean IC = 0.6457

That is a rank correlation silently becoming a linear one, from a
capitalisation, in a headline metric. `on='Returns'` did the same to
`detect_change_points` and `run_stationarity_tests`, switching the channel
from returns to prices.

TWO WAYS TO BE CORRECT HERE, and the surface uses both. A `Literal` puts the
choices in the schema an agent reads, which is better. Validating in the body
against a module constant is also fine -- `get_option_pricing.model`,
`estimate_covariance.method` and `run_portfolio_simulation.commission_model`
all do that and refuse cleanly. What is not fine is the third option, which
is what these six did: branch on equality and let everything else take the
else-arm.

So this file pins the fixed fields and, separately, holds the line on the
scan that finds candidates -- a NEW bare string whose description names its
choices has to be either constrained or deliberately exempted here.
"""

from __future__ import annotations

import re

import pytest

from standard_quant_tools.mcp.catalog import build_catalog

#: (tool, field) -> the choices the schema must declare.
#:
#: Each of these branched on `== "<default>"` somewhere and gave the other
#: arm to every other string, including a differently-cased spelling of the
#: value the caller meant.
MUST_BE_LITERAL = {
    ("detect_change_points", "on"): {"returns", "price"},
    ("run_stationarity_tests", "on"): {"price", "returns"},
    ("construct_weights_from_scores", "method"): {
        "rank",
        "top_bottom",
        "zscore",
        "vol_scaled",
    },
    ("get_feature_drift", "method"): {"spearman", "pearson"},
    ("get_feature_ic_decay", "method"): {"spearman", "pearson"},
    ("get_feature_regime_stability", "method"): {"spearman", "pearson"},
    ("run_feature_permutation_test", "method"): {"spearman", "pearson"},
}

#: Bare strings whose description names alternatives, and why each is left
#: alone. Verified by calling the tool with a bogus value, not assumed.
#:
#: The first two are the reason this cannot be a blanket rule: their
#: descriptions give EXAMPLES of a pandas offset alias, not a closed set,
#: and turning them into a Literal would refuse valid input.
DELIBERATELY_UNCONSTRAINED = {
    ("detect_liquidity_events", "freq"): "any pandas offset alias",
    ("register_external_panel", "interval"): "any pandas offset alias",
    ("analyze_stock_risk", "period"): "refuses: 'not a recognized window'",
    ("describe_data_capabilities", "source"): "refuses: unknown data provider",
    ("describe_temporal_contract", "source"): "refuses: unknown data provider",
    ("describe_temporal_contract", "frame_kind"): "refuses via TemporalContract",
    ("estimate_covariance", "method"): "refuses against covariance.METHODS",
    ("get_option_pricing", "model"): "refuses: 'unknown pricing model'",
    ("run_portfolio_simulation", "commission_model"): "refuses via _COMMISSION_CODES",
}

_QUOTED = re.compile(r"'([a-z0-9_]{2,30})'")


def _catalog():
    return build_catalog()


def _input_model(tool: str):
    """The input model for any of the 208, from the runtimes rather than the
    facade -- `agent.tools._TOOL_DISPATCH` is the 179-tool analysis union and
    deliberately excludes modeling's 20 and feature_lab's 9, four of which
    are under test here."""
    from standard_quant_tools.agent.runtimes import _build

    for runtime in _build().values():
        entry = runtime.dispatch_table.get(tool)
        if entry is not None:
            return entry[1]
    raise AssertionError(f"{tool!r} is in no runtime")


def _string_fields_naming_their_choices():
    """Bare string fields whose own description names the default and at
    least one alternative -- the shape that hid every one of these."""
    found = set()
    for name, entry in _catalog().items():
        properties = (entry.input_schema or {}).get("properties", {}) or {}
        for field, spec in properties.items():
            if spec.get("type") != "string" or spec.get("enum"):
                continue
            default = spec.get("default")
            if not isinstance(default, str) or not default:
                continue
            choices = set(_QUOTED.findall(spec.get("description") or ""))
            if default in choices and len(choices) >= 2:
                found.add((name, field))
    return found


class TestTheFixedFieldsDeclareTheirChoices:
    @pytest.mark.parametrize(
        "tool,field,expected",
        [(t, f, v) for (t, f), v in sorted(MUST_BE_LITERAL.items())],
    )
    def test_the_schema_names_them(self, tool, field, expected):
        entry = _catalog()[tool]
        spec = (entry.input_schema or {}).get("properties", {})[field]
        # pydantic renders a Literal as an enum, directly or behind a $ref
        # for a shared one.
        enum = spec.get("enum")
        if enum is None:
            for branch in spec.get("anyOf", []) or []:
                enum = enum or branch.get("enum")
        assert enum is not None, f"{tool}.{field} is still an open string"
        assert set(enum) == expected

    @pytest.mark.parametrize("tool,field", sorted(k for k in MUST_BE_LITERAL))
    def test_a_differently_cased_spelling_is_refused(self, tool, field):
        """The failure that made these silent rather than merely
        undiscoverable: the caller meant the value they typed."""
        model = _input_model(tool)
        default = model.model_fields[field].default
        with pytest.raises(Exception):
            model.model_validate({field: str(default).upper()})


class TestNoNewOpenChoiceAppears:
    def test_every_candidate_is_constrained_or_listed(self):
        """The guard. A new bare string whose description names its choices
        has to be made a Literal, or exempted HERE with the reason -- which
        is the step that was missing when these six were written."""
        candidates = _string_fields_naming_their_choices()
        unaccounted = candidates - set(DELIBERATELY_UNCONSTRAINED)
        assert not unaccounted, (
            "these string fields name their choices in prose but accept "
            f"anything: {sorted(unaccounted)}. Make each a Literal, or add "
            "it to DELIBERATELY_UNCONSTRAINED with the reason."
        )

    def test_the_exemptions_are_all_real(self):
        """An exemption for a field that no longer exists is a stale note
        that would hide the next one."""
        catalog = _catalog()
        stale = [
            (tool, field)
            for tool, field in DELIBERATELY_UNCONSTRAINED
            if tool not in catalog
            or field not in (catalog[tool].input_schema or {}).get("properties", {})
        ]
        assert not stale, f"exemptions for fields that are gone: {stale}"

    def test_the_fixed_fields_are_no_longer_candidates(self):
        """They cannot be, since they are enums now -- this fails loudly if
        one is reverted to a bare string."""
        candidates = _string_fields_naming_their_choices()
        regressed = candidates & set(MUST_BE_LITERAL)
        assert not regressed, f"back to an open string: {sorted(regressed)}"


class TestTheLibraryRefusesToo:
    """The schema is the outer wall. A direct caller of the library gets the
    same answer, which is what makes the fix durable rather than positional."""

    def test_an_unknown_ic_method_is_refused_not_defaulted(self):
        import numpy as np
        import pandas as pd

        from standard_quant_tools.error import ValidationError
        from standard_quant_tools.modeling.validation.metrics import (
            cross_sectional_ic,
        )

        rng = np.random.default_rng(0)
        dates = np.repeat(pd.bdate_range("2024-01-02", periods=30).to_numpy(), 10)
        x = pd.Series(rng.normal(size=len(dates)))
        y = pd.Series(np.exp(3 * x))
        for bad in ("SPEARMAN", "Spearman", "kendall", ""):
            with pytest.raises(ValidationError, match="not implemented"):
                cross_sectional_ic(y, x, pd.Series(dates), method=bad)

    def test_the_two_it_does_implement_genuinely_differ(self):
        """Otherwise the silent fallback would have been harmless, and the
        whole point is that it was not."""
        import numpy as np
        import pandas as pd

        from standard_quant_tools.modeling.validation.metrics import (
            cross_sectional_ic,
        )

        rng = np.random.default_rng(0)
        dates = np.repeat(pd.bdate_range("2024-01-02", periods=60).to_numpy(), 20)
        x = pd.Series(rng.normal(size=len(dates)))
        y = pd.Series(np.exp(3 * x))  # monotone, strongly non-linear
        spearman = float(
            np.nanmean(cross_sectional_ic(y, x, pd.Series(dates), method="spearman"))
        )
        pearson = float(
            np.nanmean(cross_sectional_ic(y, x, pd.Series(dates), method="pearson"))
        )
        assert spearman == pytest.approx(1.0, abs=1e-9)
        assert pearson < 0.75
