"""
Tests for portfolio/optimize.py: mean_variance_optimize (Markowitz),
risk_parity_weights, black_litterman, and build_bl_views. Pure math on
synthetic returns/covariance matrices, no network/data-provider dependency.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.optimize import (
    HAS_SCIPY,
    black_litterman,
    build_bl_views,
    mean_variance_optimize,
    risk_parity_weights,
)


@pytest.fixture(scope="module")
def uncorrelated_returns() -> pd.DataFrame:
    """3 uncorrelated assets, distinct means/vols -- deliberately simple so
    the closed-form unconstrained solution has no surprises."""
    np.random.seed(0)
    n = 1000
    return pd.DataFrame(
        {
            "A": np.random.normal(0.0004, 0.010, n),
            "B": np.random.normal(0.0006, 0.020, n),
            "C": np.random.normal(0.0003, 0.005, n),
        }
    )


@pytest.fixture(scope="module")
def correlated_returns() -> pd.DataFrame:
    """3 assets sharing a common factor -- a more realistic, non-diagonal
    covariance structure for exercising the constrained (scipy) path."""
    np.random.seed(1)
    n = 1000
    factor = np.random.normal(0.0002, 0.012, n)
    return pd.DataFrame(
        {
            "A": factor + np.random.normal(0.0001, 0.004, n),
            "B": factor * 0.8 + np.random.normal(0.0002, 0.005, n),
            "C": factor * 1.3 + np.random.normal(0.0, 0.006, n),
        }
    )


# ── mean_variance_optimize: validation ──────────────────────────────────────


class TestMeanVarianceValidation:
    def test_unknown_objective_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="objective"):
            mean_variance_optimize(uncorrelated_returns, objective="banana")

    def test_single_asset_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="at least 2 assets"):
            mean_variance_optimize(uncorrelated_returns[["A"]])

    def test_too_few_observations_raises(self):
        df = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValidationError, match="observations"):
            mean_variance_optimize(df)

    def test_target_return_missing_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="target_return"):
            mean_variance_optimize(uncorrelated_returns, objective="target_return")

    def test_target_volatility_missing_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="target_volatility"):
            mean_variance_optimize(uncorrelated_returns, objective="target_volatility")

    def test_target_volatility_non_positive_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="target_volatility"):
            mean_variance_optimize(
                uncorrelated_returns,
                objective="target_volatility",
                target_volatility=0.0,
            )

    def test_infeasible_max_weight_long_only_raises(self, uncorrelated_returns):
        # 3 assets, long-only: max_weight must be >= 1/3
        with pytest.raises(ValidationError, match="infeasible"):
            mean_variance_optimize(
                uncorrelated_returns, max_weight=0.1, allow_short=False
            )

    def test_non_positive_max_weight_raises(self, uncorrelated_returns):
        with pytest.raises(ValidationError, match="max_weight"):
            mean_variance_optimize(
                uncorrelated_returns, max_weight=0.0, allow_short=True
            )

    def test_singular_covariance_raises(self):
        # Two perfectly identical columns -> singular covariance matrix.
        df = pd.DataFrame(
            {"A": np.linspace(0.01, 0.02, 50), "B": np.linspace(0.01, 0.02, 50)}
        )
        with pytest.raises(ValidationError):
            mean_variance_optimize(df, objective="min_volatility", allow_short=True)

    def test_target_volatility_below_global_min_raises(self, uncorrelated_returns):
        min_res = mean_variance_optimize(
            uncorrelated_returns, objective="min_volatility", allow_short=True
        )
        with pytest.raises(ValidationError, match="minimum-variance"):
            mean_variance_optimize(
                uncorrelated_returns,
                objective="target_volatility",
                target_volatility=min_res["expected_volatility"] * 0.5,
                allow_short=True,
            )


# ── mean_variance_optimize: unconstrained closed-form correctness ──────────


class TestMeanVarianceUnconstrained:
    def test_weights_sum_to_one(self, uncorrelated_returns):
        for objective, kwargs in [
            ("min_volatility", {}),
            ("max_sharpe", {}),
            ("target_return", {"target_return": 0.08}),
        ]:
            res = mean_variance_optimize(
                uncorrelated_returns, objective=objective, allow_short=True, **kwargs
            )
            assert sum(res["weights"].values()) == pytest.approx(1.0, abs=1e-8)

    def test_min_volatility_is_the_global_minimum(self, uncorrelated_returns):
        """No feasible unconstrained (sum=1) weight vector should beat the
        reported min-variance volatility -- checked against random candidates."""
        res = mean_variance_optimize(
            uncorrelated_returns, objective="min_volatility", allow_short=True
        )
        cov = uncorrelated_returns.cov().to_numpy() * 252
        min_vol = res["expected_volatility"]

        rng = np.random.default_rng(42)
        for _ in range(200):
            w = rng.normal(size=3)
            w = w / w.sum()  # random unconstrained sum=1 candidate
            candidate_vol = float(np.sqrt(w @ cov @ w))
            assert candidate_vol >= min_vol - 1e-9

    def test_target_return_achieves_target(self, uncorrelated_returns):
        res = mean_variance_optimize(
            uncorrelated_returns,
            objective="target_return",
            target_return=0.10,
            allow_short=True,
        )
        assert res["expected_return"] == pytest.approx(0.10, abs=1e-6)

    def test_target_volatility_achieves_target(self, uncorrelated_returns):
        min_res = mean_variance_optimize(
            uncorrelated_returns, objective="min_volatility", allow_short=True
        )
        target_vol = min_res["expected_volatility"] * 1.5
        res = mean_variance_optimize(
            uncorrelated_returns,
            objective="target_volatility",
            target_volatility=target_vol,
            allow_short=True,
        )
        assert res["expected_volatility"] == pytest.approx(target_vol, abs=1e-6)

    def test_target_volatility_is_on_the_efficient_upper_branch(
        self, uncorrelated_returns
    ):
        """At a given achievable volatility, the efficient-frontier point
        should have a HIGHER return than the inefficient lower-branch point
        at the same volatility -- confirms _solve_unconstrained picks the
        right root of the quadratic."""
        min_res = mean_variance_optimize(
            uncorrelated_returns, objective="min_volatility", allow_short=True
        )
        target_vol = min_res["expected_volatility"] * 2.0
        res = mean_variance_optimize(
            uncorrelated_returns,
            objective="target_volatility",
            target_volatility=target_vol,
            allow_short=True,
        )
        # The efficient point's return must be >= the min-variance point's
        # return (moving further out on the frontier's upper branch never
        # decreases expected return).
        assert res["expected_return"] >= min_res["expected_return"]

    def test_converged_is_always_true_for_closed_form(self, uncorrelated_returns):
        res = mean_variance_optimize(
            uncorrelated_returns, objective="max_sharpe", allow_short=True
        )
        assert res["converged"] is True

    def test_tickers_and_weight_keys_match_columns(self, uncorrelated_returns):
        res = mean_variance_optimize(
            uncorrelated_returns, objective="min_volatility", allow_short=True
        )
        assert res["tickers"] == ["A", "B", "C"]
        assert set(res["weights"]) == {"A", "B", "C"}


# ── mean_variance_optimize: constrained (scipy) path ────────────────────────


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestMeanVarianceConstrained:
    def test_long_only_weights_are_non_negative_and_sum_to_one(
        self, correlated_returns
    ):
        res = mean_variance_optimize(
            correlated_returns, objective="min_volatility", allow_short=False
        )
        assert res["converged"] is True
        for w in res["weights"].values():
            assert w >= -1e-8
        assert sum(res["weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_long_only_min_variance_is_never_better_than_unconstrained(
        self, correlated_returns
    ):
        """Long-only is a strict subset of the unconstrained feasible set,
        so its optimal volatility can never be LOWER than the unconstrained
        global minimum."""
        unconstrained = mean_variance_optimize(
            correlated_returns, objective="min_volatility", allow_short=True
        )
        constrained = mean_variance_optimize(
            correlated_returns, objective="min_volatility", allow_short=False
        )
        assert (
            constrained["expected_volatility"]
            >= unconstrained["expected_volatility"] - 1e-9
        )

    def test_max_weight_bound_is_respected(self, correlated_returns):
        res = mean_variance_optimize(
            correlated_returns,
            objective="max_sharpe",
            allow_short=False,
            max_weight=0.5,
        )
        assert res["converged"] is True
        for w in res["weights"].values():
            assert w <= 0.5 + 1e-6

    def test_allow_short_with_max_weight_bounds_both_directions(
        self, correlated_returns
    ):
        res = mean_variance_optimize(
            correlated_returns,
            objective="min_volatility",
            allow_short=True,
            max_weight=0.6,
        )
        assert res["converged"] is True
        for w in res["weights"].values():
            assert -0.6 - 1e-6 <= w <= 0.6 + 1e-6

    def test_target_return_constrained_achieves_target(self, correlated_returns):
        res = mean_variance_optimize(
            correlated_returns,
            objective="target_return",
            target_return=0.05,
            allow_short=False,
        )
        if res["converged"]:
            assert res["expected_return"] == pytest.approx(0.05, abs=1e-4)


class TestMeanVarianceScipyRequired:
    def test_constrained_without_scipy_raises_clear_error(
        self, monkeypatch, uncorrelated_returns
    ):
        import standard_quant_tools.portfolio.optimize as optimize_mod

        monkeypatch.setattr(optimize_mod, "HAS_SCIPY", False)
        monkeypatch.setattr(optimize_mod, "_scipy_minimize", None)
        with pytest.raises(ValidationError, match="requires scipy"):
            mean_variance_optimize(
                uncorrelated_returns, objective="min_volatility", allow_short=False
            )


# ── risk_parity_weights ──────────────────────────────────────────────────────


class TestRiskParityWeights:
    def test_diagonal_covariance_matches_inverse_volatility(self):
        variances = np.array([0.01, 0.04, 0.0025])  # vols: 0.1, 0.2, 0.05
        cov = np.diag(variances)
        result = risk_parity_weights(cov)
        assert result["converged"] is True

        inv_vol = 1.0 / np.sqrt(variances)
        expected = inv_vol / inv_vol.sum()
        np.testing.assert_allclose(result["weights"], expected, atol=1e-4)

    def test_risk_contributions_are_equal_by_default(self):
        cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.002], [0.0, 0.002, 0.0225]])
        result = risk_parity_weights(cov)
        contribs = result["risk_contributions"]
        assert contribs == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)

    def test_custom_risk_budget_is_respected(self):
        cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.002], [0.0, 0.002, 0.0225]])
        budget = np.array([0.6, 0.2, 0.2])
        result = risk_parity_weights(cov, risk_budget=budget)
        np.testing.assert_allclose(result["risk_contributions"], budget, atol=1e-3)

    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.006, 0.0], [0.006, 0.01, 0.002], [0.0, 0.002, 0.0225]])
        result = risk_parity_weights(cov)
        assert result["weights"].sum() == pytest.approx(1.0, abs=1e-8)

    def test_non_square_cov_raises(self):
        with pytest.raises(ValidationError, match="square"):
            risk_parity_weights(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_mismatched_budget_length_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="length"):
            risk_parity_weights(cov, risk_budget=np.array([0.5, 0.5]))

    def test_budget_not_summing_to_one_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="sum to 1"):
            risk_parity_weights(cov, risk_budget=np.array([0.5, 0.5, 0.5]))

    def test_non_positive_budget_entry_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="> 0"):
            risk_parity_weights(cov, risk_budget=np.array([0.5, 0.5, 0.0]))


# ── black_litterman ──────────────────────────────────────────────────────────


class TestBlackLitterman:
    @pytest.fixture
    def simple_cov_and_weights(self):
        cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.0625]])
        market_weights = np.array([0.4, 0.35, 0.25])
        return cov, market_weights

    def test_view_matching_equilibrium_leaves_posterior_unchanged(
        self, simple_cov_and_weights
    ):
        cov, market_weights = simple_cov_and_weights
        pi = 2.5 * cov @ market_weights
        P = np.array([[1.0, 0.0, 0.0]])
        Q = np.array([pi[0]])
        result = black_litterman(cov, market_weights, P, Q, risk_aversion=2.5, tau=0.05)
        np.testing.assert_allclose(result["posterior_returns"], pi, atol=1e-8)

    def test_strong_view_pulls_posterior_toward_the_view(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        pi = 2.5 * cov @ market_weights
        P = np.array([[1.0, 0.0, 0.0]])
        bullish_q = pi[0] + 0.20  # 20pp above equilibrium
        Q = np.array([bullish_q])
        # Very tight uncertainty (tiny omega) -> posterior should snap close to the view.
        omega = np.array([[1e-8]])
        result = black_litterman(
            cov, market_weights, P, Q, risk_aversion=2.5, tau=0.05, omega=omega
        )
        assert result["posterior_returns"][0] == pytest.approx(bullish_q, abs=1e-3)

    def test_implied_weights_sum_to_one(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        P = np.array([[1.0, -1.0, 0.0]])
        Q = np.array([0.03])
        result = black_litterman(cov, market_weights, P, Q)
        assert result["implied_weights"].sum() == pytest.approx(1.0, abs=1e-8)

    def test_shape_mismatch_market_weights_raises(self, simple_cov_and_weights):
        cov, _ = simple_cov_and_weights
        with pytest.raises(ValidationError, match="market_weights"):
            black_litterman(
                cov, np.array([0.5, 0.5]), np.array([[1.0, 0.0, 0.0]]), np.array([0.05])
            )

    def test_p_column_mismatch_raises(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        with pytest.raises(ValidationError, match="P must have"):
            black_litterman(
                cov, market_weights, np.array([[1.0, 0.0]]), np.array([0.05])
            )

    def test_q_length_mismatch_raises(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        with pytest.raises(ValidationError, match="Q length"):
            black_litterman(
                cov,
                market_weights,
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                np.array([0.05]),
            )

    def test_non_positive_tau_raises(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        with pytest.raises(ValidationError, match="tau"):
            black_litterman(
                cov,
                market_weights,
                np.array([[1.0, 0.0, 0.0]]),
                np.array([0.05]),
                tau=0.0,
            )

    def test_non_positive_risk_aversion_raises(self, simple_cov_and_weights):
        cov, market_weights = simple_cov_and_weights
        with pytest.raises(ValidationError, match="risk_aversion"):
            black_litterman(
                cov,
                market_weights,
                np.array([[1.0, 0.0, 0.0]]),
                np.array([0.05]),
                risk_aversion=-1.0,
            )


# ── build_bl_views ────────────────────────────────────────────────────────


class TestBuildBlViews:
    def test_builds_correct_shapes(self):
        cov = np.eye(3) * 0.04
        views = [
            {"assets": {"A": 1.0}, "view_return": 0.10},
            {"assets": {"B": 1.0, "C": -1.0}, "view_return": 0.02, "confidence": 0.5},
        ]
        P, Q, omega = build_bl_views(["A", "B", "C"], views, cov, tau=0.05)
        assert P.shape == (2, 3)
        assert Q.shape == (2,)
        assert omega.shape == (2, 2)
        np.testing.assert_array_equal(P[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(P[1], [0.0, 1.0, -1.0])
        np.testing.assert_array_equal(Q, [0.10, 0.02])

    def test_lower_confidence_widens_omega(self):
        cov = np.eye(3) * 0.04
        views_full_conf = [
            {"assets": {"A": 1.0}, "view_return": 0.10, "confidence": 1.0}
        ]
        views_low_conf = [
            {"assets": {"A": 1.0}, "view_return": 0.10, "confidence": 0.1}
        ]
        _, _, omega_full = build_bl_views(
            ["A", "B", "C"], views_full_conf, cov, tau=0.05
        )
        _, _, omega_low = build_bl_views(["A", "B", "C"], views_low_conf, cov, tau=0.05)
        assert omega_low[0, 0] > omega_full[0, 0]

    def test_empty_views_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="non-empty"):
            build_bl_views(["A", "B", "C"], [], cov)

    def test_unknown_ticker_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="unknown tickers"):
            build_bl_views(
                ["A", "B", "C"], [{"assets": {"ZZZ": 1.0}, "view_return": 0.1}], cov
            )

    def test_confidence_out_of_range_raises(self):
        cov = np.eye(3) * 0.04
        with pytest.raises(ValidationError, match="confidence"):
            build_bl_views(
                ["A", "B", "C"],
                [{"assets": {"A": 1.0}, "view_return": 0.1, "confidence": 1.5}],
                cov,
            )
