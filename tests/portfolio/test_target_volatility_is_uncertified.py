"""
A bounded target_volatility answer is never reported as certified.

`mean_variance_optimize(objective="target_volatility")` with bounds
(long-only, or a `max_weight`) maximizes a linear return subject to
w'Sw = sigma^2, a quadratic EQUALITY. That set is not convex: the
fully-invested portfolios inside the bounds at exactly one volatility can
hold several local maxima of expected return, one on the inefficient
branch, and SLSQP stops at whichever it reaches. Over 60 random long-only
problems of six assets, 6 came back up to 16 percentage points of return
below the frontier at the requested volatility, with converged=True, no
warning and a null `certified`. Until the problem is solved exactly, every
bounded solve now reports certified=False with a warning naming the
problem. See the CHANGELOG entry of 2026-10-22.

THE PLANTED PROBLEM. Two assets: A at 25% volatility earning 2%, B at 20%
earning 10%, correlated -0.5. A fully-invested portfolio is (t, 1 - t) with
t the weight in A, and its variance is

    0.0625 t^2 + 0.04 (1 - t)^2 - 0.05 t (1 - t) = 0.1525 t^2 - 0.13 t + 0.04.

At a 13% target (variance 0.0169) the roots are t = 0.6 exactly, returning
0.6 * 2% + 0.4 * 10% = 5.20%, and t = 0.0231 / (0.1525 * 0.6) = 0.25246,
returning 7.98%. The minimum-variance mix is t = 0.13 / 0.305 = 0.4262, so
equal weights (t = 0.5) sit on the inefficient side, and SLSQP -- Newton's
method on the variance along the one line of fully-invested portfolios --
lands on t = 0.6. The closed form (allow_short=True, max_weight=None) takes
the efficient root. The returns are built with exactly these sample
moments, so the answer does not depend on a draw.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    run_portfolio_optimization,
)
from standard_quant_tools.portfolio import optimize as opt
from standard_quant_tools.portfolio.optimize import (
    annualized_mean_cov,
    mean_variance_optimize,
)

pytestmark = pytest.mark.skipif(not opt.HAS_SCIPY, reason="needs scipy")

MU = np.array([0.02, 0.10])
COV = np.array([[0.0625, -0.025], [-0.025, 0.04]])
TARGET = 0.13

#: The two fully-invested portfolios at 13% volatility, from the quadratic.
DOMINATED_T = 0.6
EFFICIENT_T = 0.0231 / (0.1525 * DOMINATED_T)
DOMINATED_RETURN = DOMINATED_T * 0.02 + (1 - DOMINATED_T) * 0.10  # 0.052
EFFICIENT_RETURN = EFFICIENT_T * 0.02 + (1 - EFFICIENT_T) * 0.10  # 0.07980


def _exact_returns(mu, cov, n=500, seed=0, periods_per_year=252, columns=None):
    """Daily returns whose sample mean and covariance, annualized, are
    exactly `mu` and `cov`: a draw whitened to identity sample moments and
    then coloured by the target's Cholesky factor."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    z = np.random.default_rng(seed).normal(size=(n, len(mu)))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z, rowvar=False))).T
    x = z @ np.linalg.cholesky(cov / periods_per_year).T + mu / periods_per_year
    return pd.DataFrame(x, columns=columns or ["A", "B"])


@pytest.fixture(scope="module")
def frame():
    returns = _exact_returns(MU, COV)
    mu, cov = annualized_mean_cov(returns, 252)
    np.testing.assert_allclose(mu, MU, rtol=1e-12)
    np.testing.assert_allclose(cov, COV, rtol=1e-10)
    return returns


def _warned(result):
    return [w for w in result["warnings"] if "not convex" in w]


class TestTheNonConvexSolveSaysSo:
    def test_the_planted_roots_are_the_two_portfolios_at_the_target(self):
        """The oracle, checked before it is used: both mixes are at 13%."""
        for t in (DOMINATED_T, EFFICIENT_T):
            w = np.array([t, 1 - t])
            assert float(np.sqrt(w @ COV @ w)) == pytest.approx(TARGET, abs=1e-12)
        assert DOMINATED_RETURN == pytest.approx(0.052, abs=1e-15)
        assert EFFICIENT_RETURN == pytest.approx(0.0798033, abs=1e-7)

    def test_the_dominated_root_comes_back_uncertified_and_named(self, frame):
        """KNOWN-DOMINATED. SLSQP returns t = 0.6, earning 5.20% where 7.98%
        is attainable at the same volatility inside the same bounds. This
        pins the dominated answer as it stands -- the exact solve of this
        problem is a later change, and when it lands this test flips to the
        efficient root (expected_return 0.0798, certified True). Until then
        the answer must not read as certified: it used to come back with
        converged=True, certified null and no warning."""
        result = mean_variance_optimize(
            frame, "target_volatility", target_volatility=TARGET
        )
        assert result["weights"]["A"] == pytest.approx(DOMINATED_T, abs=1e-6)
        assert result["expected_volatility"] == pytest.approx(TARGET, abs=1e-9)
        assert result["expected_return"] == pytest.approx(DOMINATED_RETURN, abs=1e-6)
        assert result["expected_return"] < EFFICIENT_RETURN - 0.02
        assert result["solver"]["certified"] is False
        (warning,) = _warned(result)
        assert "target_volatility=0.130000" in warning
        assert "objective='target_return'" in warning

    def test_the_closed_form_takes_the_efficient_root(self, frame):
        """The same problem without bounds: the frontier algebra picks the
        upper branch, so the bound in the call above -- which does not bind
        at either root -- is the only thing that changed the answer."""
        result = mean_variance_optimize(
            frame, "target_volatility", target_volatility=TARGET, allow_short=True
        )
        assert result["weights"]["A"] == pytest.approx(EFFICIENT_T, abs=1e-9)
        assert result["expected_return"] == pytest.approx(EFFICIENT_RETURN, abs=1e-9)
        assert result["solver"]["certified"] is None
        assert not _warned(result)

    @pytest.mark.parametrize(
        "bounds",
        [
            dict(),
            dict(max_weight=0.9),
            dict(allow_short=True, max_weight=1.5),
        ],
        ids=["long_only", "capped", "shorting_capped"],
    )
    def test_every_bounded_solve_is_uncertified(self, bounds):
        """Not only the dominated ones: whether a bounded answer is dominated
        cannot be told from the weights without solving the frontier, so the
        flag and the warning go on every one -- here on a five-asset factor
        universe where nothing is known to be wrong."""
        rng = np.random.default_rng(9)
        market = rng.normal(0.0004, 0.01, 600)
        frame = pd.DataFrame(
            {
                f"S{i}": 0.8 * market + rng.normal(0.0002 * (i + 1), 0.008, 600)
                for i in range(5)
            }
        )
        result = mean_variance_optimize(
            frame, "target_volatility", target_volatility=0.16, **bounds
        )
        assert result["solver"]["method"] == "SLSQP"
        assert result["solver"]["certified"] is False
        assert len(_warned(result)) == 1


class TestTheToolCarriesIt:
    def test_run_portfolio_optimization_reports_it(self, frame):
        def fake(tickers, start, end, interval="1d"):
            return frame.rename(columns=dict(zip(["A", "B"], tickers)))

        with patch(
            "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync",
            fake,
        ):
            result = run_portfolio_optimization(
                PortfolioOptimizationInput(
                    tickers=["LOW", "HIGH"],
                    start_date="2023-01-02",
                    end_date="2024-12-31",
                    method="target_volatility",
                    target_volatility=TARGET,
                )
            )
        assert result.solver.certified is False
        assert result.expected_return == pytest.approx(DOMINATED_RETURN, abs=1e-5)
        assert any("not convex" in w for w in result.warnings)
