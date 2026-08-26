"""
Portfolio construction, checked against cases with an analytic answer.

RISK PARITY HAS A CLOSED FORM IN TWO SPECIAL CASES and both are tested. When
the correlation matrix is the identity, the equal-risk portfolio is exactly
inverse-volatility -- so a solver that does not reproduce inverse-vol on a
diagonal covariance matrix is wrong, whatever it does elsewhere. And the
defining property itself is checkable at any input: the risk contributions
must be equal to solver precision, which is a much stronger test than
"the weights look plausible".

HRP IS TESTED BY CONSTRUCTION. Eight assets built as two groups of four
around two independent factors: the clustering must recover the groups, and
capital must split roughly half to each. That is an answer known in advance
from how the data was made, not read off the output.

Every "does it warn" test has a matching null case. A tool that flags
concentration on every portfolio is not measuring concentration.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import (
    concentration_analysis,
    factor_exposure_budget,
    hierarchical_risk_parity,
    liquidity_adjusted_var,
    risk_parity,
)

VOLS = np.array([0.10, 0.20, 0.40, 0.15])
CORR = np.array(
    [
        [1.0, 0.3, 0.2, 0.5],
        [0.3, 1.0, 0.4, 0.3],
        [0.2, 0.4, 1.0, 0.1],
        [0.5, 0.3, 0.1, 1.0],
    ]
)
NAMES = list("ABCD")


def _cov(vols=VOLS, corr=CORR, names=NAMES):
    return pd.DataFrame(np.outer(vols, vols) * corr, index=names, columns=names)


class TestRiskParity:
    def test_the_risk_contributions_are_actually_equal(self):
        """The defining property, checked to solver precision. Far stronger
        than asserting the weights look reasonable."""
        result = risk_parity(_cov())
        shares = list(result["risk_shares"].values())
        assert all(s == pytest.approx(0.25, abs=1e-8) for s in shares)
        assert result["converged"]

    def test_a_diagonal_covariance_gives_exactly_inverse_volatility(self):
        """
        The one case with a closed form: with no correlation, equal risk
        contribution IS inverse-volatility weighting. A solver that misses
        this is wrong everywhere.
        """
        result = risk_parity(pd.DataFrame(np.diag(VOLS**2), index=NAMES, columns=NAMES))
        expected = (1 / VOLS) / (1 / VOLS).sum()
        for name, want in zip(NAMES, expected):
            assert result["weights"][name] == pytest.approx(want, rel=1e-8)

    def test_the_volatile_asset_gets_the_smaller_weight(self):
        """Risk parity is not equal weight, and this is the difference."""
        result = risk_parity(_cov())
        assert result["weights"]["A"] > result["weights"]["C"] * 3

    def test_an_explicit_risk_budget_is_honoured(self):
        """Most real mandates are stated as risk budgets rather than as
        equal contributions."""
        result = risk_parity(_cov(), budget=[0.5, 0.3, 0.1, 0.1])
        shares = result["risk_shares"]
        assert shares["A"] == pytest.approx(0.5, abs=1e-8)
        assert shares["B"] == pytest.approx(0.3, abs=1e-8)
        assert shares["C"] == pytest.approx(0.1, abs=1e-8)

    def test_a_budget_is_normalised_rather_than_required_to_sum_to_one(self):
        proportional = risk_parity(_cov(), budget=[5, 3, 1, 1])
        normalised = risk_parity(_cov(), budget=[0.5, 0.3, 0.1, 0.1])
        for name in NAMES:
            assert proportional["weights"][name] == pytest.approx(
                normalised["weights"][name], rel=1e-8
            )

    def test_the_weights_sum_to_one(self):
        assert sum(risk_parity(_cov())["weights"].values()) == pytest.approx(1.0)

    def test_the_contributions_sum_to_the_portfolio_volatility(self):
        """This is what makes 'contribution' a genuine decomposition rather
        than an allocation of blame."""
        result = risk_parity(_cov())
        assert sum(result["risk_contributions"].values()) == pytest.approx(
            result["portfolio_volatility"], rel=1e-9
        )

    def test_it_says_it_ignores_expected_returns(self):
        result = risk_parity(_cov())
        assert any("NO expected returns" in w for w in result["warnings"])

    def test_it_declares_the_implicit_sharpe_assumption(self):
        """Risk parity implicitly bets that Sharpe ratios are similar; where
        they are not it over-weights the low-Sharpe assets."""
        result = risk_parity(_cov())
        assert any("Sharpe ratios are similar" in w for w in result["warnings"])

    def test_a_non_symmetric_matrix_is_refused_as_a_construction_bug(self):
        bad = _cov()
        bad.iloc[0, 1] = 0.99
        with pytest.raises(ValidationError, match="not symmetric"):
            risk_parity(bad)

    def test_a_zero_variance_asset_is_refused(self):
        bad = _cov()
        bad.iloc[2, :] = 0.0
        bad.iloc[:, 2] = 0.0
        with pytest.raises(ValidationError, match="non-positive"):
            risk_parity(bad)

    def test_a_zero_risk_budget_is_refused_as_an_exclusion(self):
        with pytest.raises(ValidationError, match="exclusion rather than a budget"):
            risk_parity(_cov(), budget=[0.5, 0.5, 0.0, 0.0])

    def test_a_budget_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValidationError, match="entries for"):
            risk_parity(_cov(), budget=[0.5, 0.5])

    def test_a_single_asset_is_not_a_portfolio(self):
        with pytest.raises(ValidationError, match="at least two"):
            risk_parity(pd.DataFrame([[0.04]], index=["A"], columns=["A"]))

    def test_failure_to_converge_is_reported_rather_than_hidden(self):
        result = risk_parity(_cov(), max_iterations=1)
        if not result["converged"]:
            assert any("DID NOT CONVERGE" in w for w in result["warnings"])


class TestHierarchicalRiskParity:
    @staticmethod
    def _two_groups(n=400, seed=0):
        """Eight assets, two independent factors, four assets each. The
        grouping is known before the clustering runs."""
        rng = np.random.default_rng(seed)
        f1, f2 = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
        data = {}
        for i in range(4):
            data[f"g1_{i}"] = 0.9 * f1 + rng.normal(0, 0.004, n)
        for i in range(4):
            data[f"g2_{i}"] = 0.9 * f2 + rng.normal(0, 0.004, n)
        return pd.DataFrame(data)

    def test_the_clustering_recovers_the_planted_groups(self):
        result = hierarchical_risk_parity(self._two_groups())
        order = result["cluster_order"]
        groups = [name[:2] for name in order]
        # Every member of a group must sit contiguously in the ordering.
        assert groups == sorted(groups) or groups == sorted(groups, reverse=True)

    def test_capital_splits_evenly_between_two_symmetric_groups(self):
        result = hierarchical_risk_parity(self._two_groups())
        group_one = sum(w for k, w in result["weights"].items() if k.startswith("g1"))
        assert group_one == pytest.approx(0.5, abs=0.08)

    def test_the_weights_sum_to_one(self):
        result = hierarchical_risk_parity(self._two_groups())
        assert sum(result["weights"].values()) == pytest.approx(1.0)

    def test_all_weights_are_non_negative(self):
        """HRP is long-only by construction; a negative weight would mean
        the bisection logic is broken."""
        result = hierarchical_risk_parity(self._two_groups())
        assert all(w >= 0 for w in result["weights"].values())

    def test_it_declares_that_it_optimizes_nothing(self):
        """The honest trade: HRP buys stability by giving up the claim to
        be optimal."""
        result = hierarchical_risk_parity(self._two_groups())
        assert any("NO optimality property" in w for w in result["warnings"])

    def test_it_declares_the_single_linkage_chaining_risk(self):
        result = hierarchical_risk_parity(self._two_groups())
        assert any("chaining" in w for w in result["warnings"])

    def test_a_rank_deficient_sample_is_refused(self):
        rng = np.random.default_rng(1)
        with pytest.raises(ValidationError, match="rank-deficient"):
            hierarchical_risk_parity(pd.DataFrame(rng.normal(0, 0.01, (5, 10))))

    def test_a_thin_sample_is_flagged(self):
        result = hierarchical_risk_parity(self._two_groups(n=30))
        assert any("thin for a correlation matrix" in w for w in result["warnings"])


class TestConcentration:
    def test_a_hundred_equal_positions_have_an_effective_n_of_a_hundred(self):
        result = concentration_analysis({f"s{i}": 0.01 for i in range(100)})
        assert result["effective_n"] == pytest.approx(100.0)
        assert result["herfindahl"] == pytest.approx(0.01)

    def test_one_dominant_position_collapses_the_effective_n(self):
        """
        The number this tool exists to produce: 51 positions with the
        concentration of about 4.
        """
        weights = {f"s{i}": 0.01 for i in range(50)}
        weights["big"] = 0.5
        result = concentration_analysis(weights)
        assert result["n_positions"] == 51
        assert result["effective_n"] < 5
        assert any("effective N" in w for w in result["warnings"])

    def test_an_evenly_spread_portfolio_is_not_flagged(self):
        """The null case: a detector that flags every portfolio measures
        nothing."""
        result = concentration_analysis({f"s{i}": 0.05 for i in range(20)})
        assert not any("effective N of" in w for w in result["warnings"])

    def test_a_long_short_book_is_measured_on_gross_weights(self):
        """
        Net weights make the denominator near-zero and the shares
        meaningless. A market-neutral book has a real effective N.
        """
        result = concentration_analysis({"a": 0.5, "b": 0.5, "c": -0.5, "d": -0.5})
        assert result["is_long_short"]
        assert result["gross_exposure"] == pytest.approx(2.0)
        assert result["net_exposure"] == pytest.approx(0.0)
        assert result["effective_n"] == pytest.approx(4.0)
        assert any("GROSS weights" in w for w in result["warnings"])

    def test_a_large_single_position_is_flagged(self):
        result = concentration_analysis({"big": 0.6, "a": 0.2, "b": 0.2})
        assert any("largest single position" in w for w in result["warnings"])

    def test_it_says_weight_concentration_is_not_risk_concentration(self):
        result = concentration_analysis({f"s{i}": 0.1 for i in range(10)})
        assert any("not concentration by risk" in w for w in result["warnings"])

    def test_the_effective_n_never_exceeds_the_position_count(self):
        for k in (2, 5, 25):
            rng = np.random.default_rng(k)
            raw = np.abs(rng.normal(1, 0.4, k))
            weights = {f"s{i}": float(v / raw.sum()) for i, v in enumerate(raw)}
            assert concentration_analysis(weights)["effective_n"] <= k + 1e-9

    def test_an_all_zero_portfolio_is_refused(self):
        with pytest.raises(ValidationError, match="every weight is zero"):
            concentration_analysis({"a": 0.0, "b": 0.0})

    def test_no_weights_at_all_is_refused(self):
        with pytest.raises(ValidationError, match="no weights"):
            concentration_analysis({})


class TestFactorExposureBudget:
    LOADINGS = pd.DataFrame(
        {"mkt": [1.1, 1.0, 0.9], "growth": [0.8, 0.7, -0.6]},
        index=["AAPL", "MSFT", "XOM"],
    )
    FACTOR_COV = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]], index=["mkt", "growth"], columns=["mkt", "growth"]
    )

    def test_the_exposure_is_the_weighted_sum_of_loadings(self):
        result = factor_exposure_budget(
            {"AAPL": 0.3, "MSFT": 0.3, "XOM": 0.4}, self.LOADINGS
        )
        assert result["exposures"]["mkt"] == pytest.approx(
            0.3 * 1.1 + 0.3 * 1.0 + 0.4 * 0.9
        )
        assert result["exposures"]["growth"] == pytest.approx(
            0.3 * 0.8 + 0.3 * 0.7 + 0.4 * -0.6
        )

    def test_variance_shares_sum_to_one_when_a_covariance_is_given(self):
        result = factor_exposure_budget(
            {"AAPL": 0.3, "MSFT": 0.3, "XOM": 0.4},
            self.LOADINGS,
            factor_covariance=self.FACTOR_COV,
        )
        assert sum(result["factor_variance_shares"].values()) == pytest.approx(1.0)

    def test_a_dominant_factor_is_flagged_however_many_names_are_held(self):
        """
        'I hold 40 names so I am diversified' is the failure this exists
        for.
        """
        loadings = pd.DataFrame(
            {"mkt": np.full(40, 1.0)}, index=[f"n{i}" for i in range(40)]
        )
        result = factor_exposure_budget(
            {f"n{i}": 0.025 for i in range(40)},
            loadings,
            factor_covariance=pd.DataFrame([[0.04]], index=["mkt"], columns=["mkt"]),
        )
        assert any("it is one bet" in w for w in result["warnings"])

    def test_without_a_covariance_it_says_exposure_is_not_risk(self):
        result = factor_exposure_budget({"AAPL": 1.0}, self.LOADINGS)
        assert result["factor_variance_shares"] is None
        assert any("not risk" in w for w in result["warnings"])

    def test_unmapped_positions_are_named_rather_than_dropped_silently(self):
        result = factor_exposure_budget({"AAPL": 0.5, "UNKNOWN": 0.5}, self.LOADINGS)
        assert result["n_unmapped"] == 1
        assert "UNKNOWN" in result["unmapped"]
        assert any("invisible to this decomposition" in w for w in result["warnings"])

    def test_gross_and_net_exposure_differ_for_a_long_short_book(self):
        result = factor_exposure_budget({"AAPL": 0.5, "XOM": -0.5}, self.LOADINGS)
        assert result["gross_exposure"] == pytest.approx(1.0)
        assert result["net_exposure"] == pytest.approx(0.0)

    def test_no_overlapping_assets_is_refused_with_both_name_lists(self):
        with pytest.raises(ValidationError, match="no asset"):
            factor_exposure_budget({"NOPE": 1.0}, self.LOADINGS)

    def test_a_mismatched_factor_covariance_is_refused(self):
        with pytest.raises(ValidationError, match="but there are"):
            factor_exposure_budget(
                {"AAPL": 1.0},
                self.LOADINGS,
                factor_covariance=pd.DataFrame(
                    [[0.04]], index=["mkt"], columns=["mkt"]
                ),
            )


class TestLiquidityAdjustedVar:
    POSITIONS = {"LIQ": 1e6, "ILLIQ": 1e6}
    VOLS = {"LIQ": 0.25, "ILLIQ": 0.45}
    VOLUMES = {"LIQ": 5e8, "ILLIQ": 2e6}

    def test_the_illiquid_position_is_adjusted_and_the_liquid_one_is_not(self):
        """
        Same position size, same confidence -- the only difference is how
        long it takes to get out, and that is the whole point.
        """
        result = liquidity_adjusted_var(self.POSITIONS, self.VOLS, self.VOLUMES)
        rows = {r["asset"]: r for r in result["by_position"]}
        assert rows["LIQ"]["adjustment_multiple"] == pytest.approx(1.0, abs=0.01)
        assert rows["ILLIQ"]["adjustment_multiple"] > 1.5

    def test_the_adjustment_is_the_square_root_of_the_liquidation_horizon(self):
        result = liquidity_adjusted_var(self.POSITIONS, self.VOLS, self.VOLUMES)
        row = next(r for r in result["by_position"] if r["asset"] == "ILLIQ")
        assert row["adjustment_multiple"] == pytest.approx(
            math.sqrt(max(row["liquidation_days"], 1.0)), rel=1e-9
        )

    def test_a_lower_participation_rate_lengthens_the_horizon(self):
        patient = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, participation_rate=0.05
        )
        aggressive = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, participation_rate=0.30
        )
        assert patient["liquidity_adjusted_var"] > aggressive["liquidity_adjusted_var"]

    def test_correlation_of_one_adds_the_risks_linearly(self):
        independent = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, correlation=0.0
        )
        crisis = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, correlation=1.0
        )
        assert crisis["liquidity_adjusted_var"] > independent["liquidity_adjusted_var"]
        total = sum(r["liquidity_adjusted_var"] for r in crisis["by_position"])
        assert crisis["liquidity_adjusted_var"] == pytest.approx(total, rel=1e-6)

    def test_the_cost_is_reported_separately_from_the_quantile(self):
        """Cost is an expectation and VaR is a quantile; adding them gives a
        number that is neither."""
        result = liquidity_adjusted_var(self.POSITIONS, self.VOLS, self.VOLUMES)
        assert result["expected_liquidation_cost"] > 0
        assert result["expected_liquidation_cost"] != result["liquidity_adjusted_var"]
        assert any("reported separately" in w for w in result["warnings"])

    def test_slow_positions_are_named(self):
        result = liquidity_adjusted_var(
            {"SLOW": 1e7}, {"SLOW": 0.4}, {"SLOW": 1e5}, participation_rate=0.10
        )
        assert any("over 5 days to liquidate" in w for w in result["warnings"])

    def test_a_low_correlation_assumption_is_flagged_as_optimistic(self):
        result = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, correlation=0.1
        )
        assert any("correlations go to 1 in a crisis" in w for w in result["warnings"])

    def test_a_higher_confidence_gives_a_larger_var(self):
        low = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, confidence=0.90
        )
        high = liquidity_adjusted_var(
            self.POSITIONS, self.VOLS, self.VOLUMES, confidence=0.99
        )
        assert high["liquidity_adjusted_var"] > low["liquidity_adjusted_var"]

    def test_a_missing_volume_is_refused_because_it_is_the_question(self):
        with pytest.raises(ValidationError, match="entire question"):
            liquidity_adjusted_var({"A": 1e6}, {"A": 0.3}, {})

    def test_a_missing_volatility_is_refused(self):
        with pytest.raises(ValidationError, match="no usable volatility"):
            liquidity_adjusted_var({"A": 1e6}, {}, {"A": 1e7})

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
    def test_a_confidence_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(ValidationError, match="confidence"):
            liquidity_adjusted_var(
                self.POSITIONS, self.VOLS, self.VOLUMES, confidence=bad
            )
