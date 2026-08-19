"""
Regression tests for Pass 3 of the full-codebase audit: a solver reporting
success is not a guarantee the answer is valid.

The theme is one sentence: `result.success` is the solver's opinion of its own
run, not a statement that the returned vector satisfies the constraints it was
given. Everything here either verifies the result independently or rejects an
input that would make the solve meaningless before it starts.
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

GOOD_COV = np.array([[0.04, 0.01], [0.01, 0.05]])
NAN = float("nan")
INF = float("inf")


def _returns(n=400, k=4, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0005, 0.012, (n, k)), columns=list("ABCDEFGH")[:k])


class TestIllConditionedCovariance:
    """
    The rank check added earlier catches an exactly-degenerate covariance. It
    does not catch two assets that are merely almost identical — the far more
    common real case (a share class pair, an ETF and its largest holding).

    Measured on three assets where two differ by 1e-9 of noise:

        rank              3 / 3          (passes the rank check)
        condition number  3.827e+14
        max |weight|      197838.4       converged=True, no warning
    """

    def _near_collinear(self):
        rng = np.random.default_rng(7)
        base = rng.normal(0, 0.01, 500)
        return pd.DataFrame(
            {
                "A": base,
                "B": base + rng.normal(0, 1e-9, 500),
                "C": rng.normal(0, 0.01, 500),
            }
        )

    def test_the_matrix_really_is_full_rank(self):
        """Otherwise this would be caught by the existing rank check and the
        conditioning warning would be untested."""
        cov = self._near_collinear().cov().to_numpy() * 252
        assert np.linalg.matrix_rank(cov) == 3
        assert np.linalg.cond(cov) > 1e10

    def test_ill_conditioning_is_reported(self):
        result = mean_variance_optimize(
            self._near_collinear(), "min_volatility", allow_short=True, max_weight=None
        )
        assert any("ill-conditioned" in w for w in result["warnings"])

    def test_well_conditioned_input_carries_no_such_warning(self):
        result = mean_variance_optimize(
            _returns(), "min_volatility", allow_short=True, max_weight=None
        )
        assert not any("ill-conditioned" in w for w in result["warnings"])


class TestSolutionIsVerifiedIndependently:
    @pytest.mark.skipif(not HAS_SCIPY, reason="constrained path requires scipy")
    def test_infeasible_target_return_is_reported_not_silently_missed(self):
        """
        A long-only target no portfolio can reach returned weights that look
        entirely well-formed: sum(w)=1.0000 with an achieved return of 0.2443
        against the 99.0 requested — the constraint missed by two orders of
        magnitude.
        """
        result = mean_variance_optimize(
            _returns(), "target_return", target_return=99.0, allow_short=False
        )
        assert result["converged"] is False
        assert any("target_return" in w for w in result["warnings"])

    @pytest.mark.skipif(not HAS_SCIPY, reason="constrained path requires scipy")
    def test_a_feasible_target_is_actually_met(self):
        result = mean_variance_optimize(
            _returns(), "target_return", target_return=0.10, allow_short=False
        )
        assert result["converged"] is True
        assert result["expected_return"] == pytest.approx(0.10, abs=1e-6)
        assert result["warnings"] == [] or all(
            "target_return" not in w for w in result["warnings"]
        )

    def test_weights_that_do_not_sum_to_one_are_flagged(self):
        """
        Found by the verification itself: the ill-conditioned case above
        returns weights summing to 0.997433, which the closed-form path
        previously reported as converged=True.
        """
        rng = np.random.default_rng(7)
        base = rng.normal(0, 0.01, 500)
        data = pd.DataFrame(
            {
                "A": base,
                "B": base + rng.normal(0, 1e-9, 500),
                "C": rng.normal(0, 0.01, 500),
            }
        )
        result = mean_variance_optimize(
            data, "min_volatility", allow_short=True, max_weight=None
        )
        total = sum(result["weights"].values())
        if abs(total - 1.0) > 1e-6:
            assert result["converged"] is False
            assert any("sum to" in w for w in result["warnings"])

    def test_a_clean_problem_still_reports_convergence(self):
        result = mean_variance_optimize(
            _returns(), "min_volatility", allow_short=True, max_weight=None
        )
        assert result["converged"] is True
        assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-9)


class TestRiskParityContract:
    @pytest.mark.parametrize(
        "cov",
        [
            np.array([[0.04, NAN], [NAN, 0.05]]),
            np.array([[0.04, INF], [INF, 0.05]]),
        ],
    )
    def test_non_finite_covariance_rejected(self, cov):
        """
        A NaN covariance does not trip the `port_var <= 0` degeneracy guard —
        NaN satisfies no comparison — so it flowed through every iteration and
        emerged as {nan, nan} weights with no error raised.
        """
        with pytest.raises(ValidationError, match="non-finite"):
            risk_parity_weights(cov)

    def test_asymmetric_covariance_rejected(self):
        """Silently accepted before, and quietly used as though it were a
        covariance."""
        with pytest.raises(ValidationError, match="not symmetric"):
            risk_parity_weights(np.array([[0.04, 0.01], [0.02, 0.05]]))

    @pytest.mark.parametrize("bad", [0, -5, 2.5])
    def test_invalid_max_iterations_rejected(self, bad):
        """Zero or negative skipped the loop entirely and returned the
        equal-weight starting vector as unconverged — indistinguishable from a
        genuine convergence failure."""
        with pytest.raises(ValidationError, match="max_iterations"):
            risk_parity_weights(GOOD_COV, max_iterations=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, NAN])
    def test_invalid_tolerance_rejected(self, bad):
        with pytest.raises(ValidationError, match="tol"):
            risk_parity_weights(GOOD_COV, tol=bad)

    def test_non_finite_risk_budget_rejected_as_such(self):
        """It used to reach the sum-to-1.0 check and be reported as a sum
        problem, which points the caller at the wrong thing."""
        with pytest.raises(ValidationError, match="non-finite"):
            risk_parity_weights(GOOD_COV, risk_budget=np.array([NAN, 1.0]))

    def test_a_valid_problem_still_solves(self):
        result = risk_parity_weights(GOOD_COV)
        assert result["converged"]
        assert sum(result["weights"]) == pytest.approx(1.0)


class TestBlackLittermanContract:
    """
    Every matrix and vector here is inverted or multiplied into the posterior,
    so one non-finite entry anywhere made the whole result NaN with no error.
    The scalar guards were comparisons (`tau <= 0`), which NaN passes.
    """

    MW = np.array([0.5, 0.5])
    P = np.array([[1.0, -1.0]])
    Q = np.array([0.05])

    def test_nan_tau_rejected(self):
        with pytest.raises(ValidationError, match="tau"):
            black_litterman(GOOD_COV, self.MW, self.P, self.Q, tau=NAN)

    def test_nan_risk_aversion_rejected(self):
        with pytest.raises(ValidationError, match="risk_aversion"):
            black_litterman(GOOD_COV, self.MW, self.P, self.Q, risk_aversion=NAN)

    def test_nan_view_returns_rejected(self):
        with pytest.raises(ValidationError, match="Q"):
            black_litterman(GOOD_COV, self.MW, self.P, np.array([NAN]))

    def test_nan_market_weights_rejected(self):
        with pytest.raises(ValidationError, match="market_weights"):
            black_litterman(GOOD_COV, np.array([NAN, 0.5]), self.P, self.Q)

    def test_nan_covariance_rejected(self):
        with pytest.raises(ValidationError, match="non-finite"):
            black_litterman(
                np.array([[0.04, NAN], [NAN, 0.05]]), self.MW, self.P, self.Q
            )

    def test_a_valid_problem_still_produces_weights(self):
        result = black_litterman(GOOD_COV, self.MW, self.P, self.Q)
        assert np.all(np.isfinite(result["implied_weights"]))
        assert result["implied_weights"].sum() == pytest.approx(1.0)


class TestBuildBLViewsContract:
    def test_nan_view_return_rejected(self):
        with pytest.raises(ValidationError, match="view_return"):
            build_bl_views(
                ["A", "B"], [{"assets": {"A": 1.0}, "view_return": NAN}], GOOD_COV
            )

    def test_nan_confidence_rejected_before_the_range_check(self):
        """NaN satisfies neither `<= 0` nor `> 1`, so it passed both halves of
        the guard and then divided the omega diagonal by NaN — making every
        posterior weight NaN."""
        with pytest.raises(ValidationError, match="confidence"):
            build_bl_views(
                ["A", "B"],
                [{"assets": {"A": 1.0}, "view_return": 0.1, "confidence": NAN}],
                GOOD_COV,
            )

    def test_infinite_pick_coefficient_rejected(self):
        with pytest.raises(ValidationError, match="pick coefficient"):
            build_bl_views(
                ["A", "B"], [{"assets": {"A": INF}, "view_return": 0.1}], GOOD_COV
            )

    def test_duplicate_tickers_rejected(self):
        """
        The ticker->column map keeps the LAST index, so a view on a repeated
        name silently attached to the wrong slot: on tickers=["A", "A"] a view
        on "A" produced the row [0, 1].
        """
        with pytest.raises(ValidationError, match="duplicates"):
            build_bl_views(
                ["A", "A"], [{"assets": {"A": 1.0}, "view_return": 0.1}], GOOD_COV
            )

    def test_a_well_formed_view_still_builds(self):
        P, Q, omega = build_bl_views(
            ["A", "B"],
            [{"assets": {"A": 1.0, "B": -1.0}, "view_return": 0.05}],
            GOOD_COV,
        )
        assert P.tolist() == [[1.0, -1.0]]
        assert np.all(np.isfinite(omega))


class TestBuildPortfolioHygiene:
    def test_empty_returns_rejected_at_the_boundary(self):
        """portfolio_metrics reaches for equity_curve.iloc[-1], so this used
        to fail far from its cause."""
        from standard_quant_tools.portfolio.portfolio import build_portfolio

        with pytest.raises(ValidationError, match="empty"):
            build_portfolio(_returns().iloc[:0], [0.25] * 4)

    def test_non_finite_weight_named_as_such(self):
        """Caught only incidentally before — the sum became NaN, so the error
        blamed the sum and pointed the caller at the wrong thing."""
        from standard_quant_tools.portfolio.portfolio import build_portfolio

        with pytest.raises(ValidationError, match="non-finite"):
            build_portfolio(_returns(), [NAN, 0.3, 0.3, 0.4])

    def test_infinite_return_rejected(self):
        from standard_quant_tools.portfolio.portfolio import build_portfolio

        data = _returns()
        data.iloc[5, 0] = INF
        with pytest.raises(ValidationError, match="infinite"):
            build_portfolio(data, [0.25] * 4)

    def test_a_valid_portfolio_still_builds(self):
        from standard_quant_tools.portfolio.portfolio import build_portfolio

        series = build_portfolio(_returns(), [0.25] * 4)
        assert len(series) == 400
        assert np.all(np.isfinite(series.to_numpy()))
