"""
Regression tests for the portfolio/screener/agent-tools audit.

Every case here pins the REASON a finding was wrong, not just the new
behavior, so a later change that reintroduces the cause fails with an
explanation rather than a bare assertion mismatch.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.optimize import (
    HAS_SCIPY,
    build_bl_views,
    mean_variance_optimize,
)


class TestTangencySignDegeneracy:
    """
    The closed-form max_sharpe normalizes w = Sigma^-1 (mu - rf*1) by its own
    sum, which equals B - rf*A. The resulting portfolio's excess return is
    the quadratic form (mu-rf)' Sigma^-1 (mu-rf) divided by that sum. The
    numerator is a quadratic form in a positive-definite Sigma, so it is
    ALWAYS positive -- the sign of the excess return is entirely the
    denominator's. A negative denominator therefore flips the answer onto the
    inefficient branch, and an objective named max_sharpe returned the
    MINIMUM-Sharpe portfolio with converged=True.

    Only abs(denom) < 1e-14 was guarded: that catches the un-normalizable
    case and misses the inverted one.
    """

    def _rets(self):
        np.random.seed(11)
        cov = np.array([[0.04, 0.01], [0.01, 0.05]]) / 252.0
        draws = np.random.multivariate_normal([0.10 / 252, 0.08 / 252], cov, 3000)
        return pd.DataFrame(draws, columns=["A", "B"])

    def test_low_risk_free_rate_still_solves_with_positive_sharpe(self):
        r = mean_variance_optimize(
            self._rets(),
            "max_sharpe",
            risk_free_rate=0.0,
            allow_short=True,
            max_weight=None,
        )
        assert r["sharpe_ratio"] > 0
        assert r["converged"]

    def test_risk_free_above_min_variance_return_is_rejected(self):
        """
        Previously produced a NEGATIVE Sharpe reported as converged. The
        supremum of Sharpe over the fully-invested set is genuinely not
        attained in this regime, so it is reported rather than approximated.
        """
        with pytest.raises(ValidationError, match="no maximum-Sharpe portfolio"):
            mean_variance_optimize(
                self._rets(),
                "max_sharpe",
                risk_free_rate=0.90,
                allow_short=True,
                max_weight=None,
            )

    def test_error_names_the_threshold_the_rate_must_be_below(self):
        with pytest.raises(ValidationError, match="minimum-variance"):
            mean_variance_optimize(
                self._rets(),
                "max_sharpe",
                risk_free_rate=0.90,
                allow_short=True,
                max_weight=None,
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="constrained path requires scipy")
    def test_bounded_request_is_refused_when_no_portfolio_beats_the_rate(self):
        """
        This test used to assert the opposite -- that the bounded request
        "still has a solution" because bounds make the feasible set compact.
        A maximum does exist, and it was the pathology: at rf=0.90 no
        portfolio inside max_weight=5 earns more than 0.18, every Sharpe is
        negative, and maximizing a negative Sharpe rewards volatility. It
        returned {A: -4, B: 5} at 5x leverage, Sharpe -1.06, converged=True.
        The bounded path now refuses the regime the closed form refuses,
        naming the highest return the bounds allow.
        """
        with pytest.raises(ValidationError, match="highest attainable"):
            mean_variance_optimize(
                self._rets(),
                "max_sharpe",
                risk_free_rate=0.90,
                allow_short=True,
                max_weight=5.0,
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="constrained path requires scipy")
    def test_bounded_request_below_the_attainable_return_still_solves(self):
        """The null case: a rate the bounded book can beat is answered, with
        a positive Sharpe."""
        r = mean_variance_optimize(
            self._rets(),
            "max_sharpe",
            risk_free_rate=0.0,
            allow_short=True,
            max_weight=5.0,
        )
        assert r["converged"]
        assert r["sharpe_ratio"] > 0


def _exact_moments(means, vols, n=2000, seed=5):
    """Uncorrelated daily returns whose sample means and volatilities are
    EXACTLY the annual figures given, so which asset dominates which is
    known before the optimizer runs."""
    rng = np.random.default_rng(seed)
    data = {}
    for name, mean, vol in zip("ABC", means, vols):
        z = rng.normal(size=n)
        z = (z - z.mean()) / z.std(ddof=1)
        data[name] = mean / 252 + z * vol / np.sqrt(252)
    return pd.DataFrame(data)


def _dominated_book():
    return _exact_moments([0.04, 0.03, 0.05], [0.10, 0.30, 0.20])


@pytest.mark.skipif(not HAS_SCIPY, reason="constrained path requires scipy")
class TestLongOnlyMaxSharpeAboveEveryReturn:
    """
    A 4%/10%-vol, B 3%/30% and C 5%/20%, long-only, rf=10%. B is dominated
    by both others, and it came back as the answer -- 100% B, return 3%,
    vol 30%, Sharpe -0.23, converged=True and no warning -- because with
    every excess return negative the "maximum" Sharpe is the one that
    divides the shortfall by the most volatility.
    """

    def test_a_rate_above_every_return_is_refused(self):
        with pytest.raises(ValidationError, match="positive excess return"):
            mean_variance_optimize(_dominated_book(), "max_sharpe", risk_free_rate=0.10)

    def test_a_rate_above_every_return_is_refused_under_a_cap_too(self):
        with pytest.raises(ValidationError, match="positive excess return"):
            mean_variance_optimize(
                _dominated_book(), "max_sharpe", risk_free_rate=0.10, max_weight=0.5
            )

    def test_at_a_zero_rate_the_answer_is_the_tangency_portfolio(self):
        """
        The null case: at rf=0 every asset beats the rate and the long-only
        answer is the tangency portfolio -- which holds all three, about
        (4, 1/3, 5/4) / 5.58 for uncorrelated assets (mu / sigma^2). The
        unconstrained closed form has no negative weight here, so the two
        paths must agree.
        """
        book = _dominated_book()
        r = mean_variance_optimize(book, "max_sharpe", risk_free_rate=0.0)
        closed = mean_variance_optimize(
            book, "max_sharpe", risk_free_rate=0.0, allow_short=True
        )
        assert r["converged"]
        assert r["sharpe_ratio"] > 0
        for name in "ABC":
            assert r["weights"][name] == pytest.approx(
                closed["weights"][name], abs=1e-4
            )
        assert r["weights"]["B"] == pytest.approx(0.06, abs=0.02)

    def test_a_rate_only_one_asset_beats_is_answered_with_a_positive_sharpe(self):
        """Only C (5%) beats 4.5%. Equal weights start in the negative
        region; the answer must still be the positive-excess portfolio."""
        r = mean_variance_optimize(
            _dominated_book(), "max_sharpe", risk_free_rate=0.045
        )
        assert r["converged"]
        assert r["expected_return"] > 0.045
        assert r["sharpe_ratio"] > 0


class TestCovarianceEstimability:
    """
    A sample covariance built from n_obs observations of n_assets assets has
    rank at most n_obs - 1, so with n_obs <= n_assets it is singular by
    construction. The two solver paths disagreed about that: the closed form
    raised (np.linalg.inv fails), while SLSQP inverts nothing, found a
    direction in the covariance's NULL SPACE, and reported it as a
    zero-variance portfolio with converged=True.
    """

    def _rank_deficient(self):
        np.random.seed(1)
        return pd.DataFrame(
            np.random.normal(0.001, 0.02, (5, 6)),
            columns=[f"A{i}" for i in range(6)],
        )

    def test_the_null_space_portfolio_was_not_actually_riskless(self):
        """
        Records what the old behavior produced so the guard is not later
        mistaken for excess caution: in-sample w'Sigma w was ~1e-14 while the
        same weights carried ~23% annualized volatility out of sample.
        """
        cov = self._rank_deficient().cov().to_numpy() * 252
        assert np.linalg.matrix_rank(cov) < cov.shape[0]
        assert np.linalg.eigvalsh(cov)[0] < 1e-12, "the null space SLSQP exploited"

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(allow_short=True, max_weight=None), id="closed_form"),
            pytest.param(
                dict(allow_short=False, max_weight=None), id="scipy_long_only"
            ),
            pytest.param(
                dict(allow_short=True, max_weight=2.0), id="scipy_short_capped"
            ),
        ],
    )
    def test_every_path_rejects_a_singular_by_construction_covariance(self, kwargs):
        if not HAS_SCIPY and not kwargs["allow_short"]:
            pytest.skip("constrained path requires scipy")
        with pytest.raises(ValidationError, match="observations for"):
            mean_variance_optimize(self._rank_deficient(), "min_volatility", **kwargs)

    def test_enough_observations_still_solves(self):
        np.random.seed(1)
        data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (300, 6)),
            columns=[f"A{i}" for i in range(6)],
        )
        r = mean_variance_optimize(
            data, "min_volatility", allow_short=True, max_weight=None
        )
        assert r["converged"]
        assert r["expected_volatility"] > 0.01, "a real portfolio has real risk"

    def test_perfectly_collinear_assets_rejected_with_ample_observations(self):
        """Enough rows, but two identical columns -- still rank-deficient, and
        still a free zero-variance direction for an optimizer to find."""
        np.random.seed(2)
        a = np.random.normal(0.001, 0.02, 500)
        data = pd.DataFrame(
            {"A": a, "B": np.random.normal(0.001, 0.02, 500), "A_CLONE": a}
        )
        with pytest.raises(ValidationError, match="rank-deficient"):
            mean_variance_optimize(
                data, "min_volatility", allow_short=True, max_weight=None
            )


class TestOptimizerInputHygiene:
    def _rets(self, n=600):
        np.random.seed(5)
        return pd.DataFrame(
            {
                "A": np.random.normal(0.0008, 0.012, n),
                "B": np.random.normal(0.0006, 0.014, n),
            }
        )

    def test_infinite_return_rejected_rather_than_returning_nan_weights(self):
        """
        dropna() removes NaN but not inf, so an infinity reached the solver
        and came back as nan weights with converged=True -- a success flag on
        a result containing no numbers.
        """
        bad = self._rets()
        bad.iloc[5, 0] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            mean_variance_optimize(
                bad, "min_volatility", allow_short=True, max_weight=None
            )

    @pytest.mark.skipif(not HAS_SCIPY, reason="requires scipy")
    def test_max_weight_infeasibility_checked_when_shorting_is_allowed(self):
        """
        Shorting lowers the per-asset FLOOR, not the cap, so the weight sum is
        still bounded above by n * max_weight and summing to 1 is still
        infeasible. Checking this only for long-only let the shorting case
        through, returning weights that summed to 0.6.
        """
        with pytest.raises(ValidationError, match="infeasible"):
            mean_variance_optimize(
                self._rets(), "min_volatility", allow_short=True, max_weight=0.3
            )

    def test_small_sample_is_warned_about_not_rejected(self):
        """15 observations for 2 assets is invertible and nearly meaningless.
        A warning, since a short window is a legitimate thing to ask for."""
        np.random.seed(5)
        short = pd.DataFrame(np.random.normal(0.001, 0.02, (15, 2)), columns=["A", "B"])
        r = mean_variance_optimize(
            short, "min_volatility", allow_short=True, max_weight=None
        )
        assert r["converged"]
        assert any("observations" in w for w in r["warnings"])

    def test_ample_sample_carries_no_warning(self):
        r = mean_variance_optimize(
            self._rets(), "min_volatility", allow_short=True, max_weight=None
        )
        assert r["warnings"] == []


class TestBuildBLViewsErrors:
    """
    These view dicts are agent-reachable through run_portfolio_optimization,
    so a raw KeyError naming only the missing key is what an LLM would have
    to self-correct from.
    """

    def _cov(self):
        np.random.seed(6)
        d = pd.DataFrame(
            {
                "A": np.random.normal(0, 0.01, 300),
                "B": np.random.normal(0, 0.01, 300),
            }
        )
        return d.cov().to_numpy() * 252

    @pytest.mark.parametrize(
        "view,missing",
        [
            ({"assets": {"A": 1.0}}, "view_return"),
            ({"view_return": 0.1}, "assets"),
        ],
    )
    def test_missing_key_raises_validation_error_naming_it(self, view, missing):
        with pytest.raises(ValidationError, match=missing):
            build_bl_views(["A", "B"], [view], self._cov())

    def test_empty_assets_rejected(self):
        with pytest.raises(ValidationError, match="non-empty dict"):
            build_bl_views(
                ["A", "B"], [{"assets": {}, "view_return": 0.1}], self._cov()
            )

    def test_well_formed_view_still_builds(self):
        P, Q, omega = build_bl_views(
            ["A", "B"],
            [{"assets": {"A": 1.0, "B": -1.0}, "view_return": 0.05}],
            self._cov(),
        )
        assert P.shape == (1, 2)
        assert Q.shape == (1,)
        assert omega.shape == (1, 1)
