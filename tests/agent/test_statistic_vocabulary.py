"""
One name per statistic across two tools.

`calculate_series_metrics` answers with `sharpe_ratio`, which is what
`metrics/risk_metrics.py` calls the function and what every result model in
the library spells it. `get_bootstrap_interval` took `sharpe`. Putting an
interval on a Sharpe is the next thing a caller does -- this module's own
docstring says the interval is "the number a decision should be made on" --
and it failed on the spelling, with pydantic's permitted-values list and
nothing to say that one of those values was the number already in hand.

The second half of this file matters more than the first: three names in the
metrics vocabulary look like they should alias onto this one and are NOT the
same statistic. Aliasing those would turn a refusal into a wrong answer with
nothing in the result saying so.
"""

import numpy as np
import pytest
from pydantic import ValidationError as PydanticError

from standard_quant_tools.agent.runtimes.research import inference_tools as it
from standard_quant_tools.agent.runtimes.research.reference_tools import METRIC_NAMES


@pytest.fixture
def returns() -> list:
    rng = np.random.default_rng(11)
    return list(rng.normal(0.0004, 0.011, 400))


def _interval(values, statistic):
    return it.get_bootstrap_interval(
        it.BootstrapInput(
            values=values, statistic=statistic, n_bootstrap=300, seed=3
        )
    )


class TestTheLibrarysSpellingIsAccepted:
    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [("sharpe_ratio", "sharpe"), ("sortino_ratio", "sortino")],
    )
    def test_an_alias_gives_the_same_interval_as_the_canonical_name(
        self, returns, alias, canonical
    ):
        aliased = _interval(returns, alias)
        canon = _interval(returns, canonical)
        assert aliased.point_estimate == canon.point_estimate
        assert (aliased.lower, aliased.upper) == (canon.lower, canon.upper)

    @pytest.mark.parametrize("alias", ["sharpe_ratio", "sortino_ratio"])
    def test_the_aliases_are_exactly_the_names_the_other_tool_answers_with(
        self, alias
    ):
        """The reason these two and not others.

        If `calculate_series_metrics` ever stops using one of these names, the
        alias stops describing anything -- so it is pinned to that tool's
        vocabulary rather than to a guess about what a caller might type.
        """
        assert alias in METRIC_NAMES

    def test_whitespace_does_not_defeat_it(self, returns):
        assert _interval(returns, " sharpe_ratio ").point_estimate == pytest.approx(
            _interval(returns, "sharpe").point_estimate
        )


class TestANearMissIsRefusedAndNamed:
    """Not every name in the other vocabulary has a counterpart here."""

    @pytest.mark.parametrize(
        ("name", "says"),
        [
            ("var_historical", "fixes the level at 95%"),
            ("var_parametric", "HISTORICAL"),
            ("cvar", "cvar_95"),
            ("annualized_volatility", "PERIODIC"),
            ("cumulative_return", "one number for the whole path"),
            ("cagr", "per period"),
        ],
    )
    def test_it_says_which_one_is_here_and_why_it_differs(self, name, says):
        with pytest.raises(PydanticError) as caught:
            it.BootstrapInput(values=[0.0] * 40, statistic=name)
        assert says in str(caught.value)

    @pytest.mark.parametrize(
        "name", ["var_historical", "var_parametric", "cvar", "annualized_volatility"]
    )
    def test_a_near_miss_is_refused_rather_than_quietly_answered(self, name):
        """The property the refusals exist to hold.

        `var_95` fixes the level where `var_historical` takes one. A caller who
        asked for the second and silently got the first would have an interval
        on a quantity they did not ask for.
        """
        with pytest.raises(PydanticError):
            it.BootstrapInput(values=[0.0] * 40, statistic=name)

    def test_an_unknown_name_still_gets_the_plain_list(self):
        """The near-miss table must not swallow the default refusal.

        A name nobody can map should still produce pydantic's enumeration,
        which is the most useful answer when there is nothing to suggest.
        """
        with pytest.raises(PydanticError) as caught:
            it.BootstrapInput(values=[0.0] * 40, statistic="not_a_statistic")
        message = str(caught.value)
        assert "'sharpe'" in message and "'cvar_95'" in message
        assert "not the same quantity" not in message


class TestTheTwoVocabulariesAreAccountedFor:
    def test_every_shared_statistic_is_reachable_by_one_name(self):
        """The audit this change came out of.

        Of the names in both vocabularies: `max_drawdown` already matched,
        `sharpe_ratio`/`sortino_ratio` are now aliased, and the var/cvar pair
        is deliberately not. Anything else appearing in both under different
        spellings is a new split and should fail here.
        """
        here = {
            "mean", "median", "std", "sharpe", "sortino", "skew", "kurtosis",
            "max_drawdown", "win_rate", "var_95", "cvar_95",
        }
        aliased = set(it._STATISTIC_ALIASES)
        explained = set(it._STATISTIC_NEAR_MISSES)
        unaccounted = set(METRIC_NAMES) - here - aliased - explained
        # What remains has no counterpart here at all, which is a different
        # thing from a spelling split: an information ratio or an EVT tail is
        # not a statistic this bootstrap resamples.
        assert unaccounted == {
            "calmar_ratio",
            "drawdown_series",
            "evt_tail_risk",
            "information_ratio",
            "treynor_ratio",
        }

    def test_no_alias_points_at_a_name_that_is_not_here(self):
        here = {
            "mean", "median", "std", "sharpe", "sortino", "skew", "kurtosis",
            "max_drawdown", "win_rate", "var_95", "cvar_95",
        }
        assert set(it._STATISTIC_ALIASES.values()) <= here
