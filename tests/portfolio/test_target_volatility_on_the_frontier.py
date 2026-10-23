"""
A bounded target_volatility answer is the frontier portfolio at that
volatility, certified.

`mean_variance_optimize(objective="target_volatility")` with bounds
(long-only, or a `max_weight`) maximizes a linear return subject to
w'Sw = sigma^2, a quadratic EQUALITY. That set is not convex, and SLSQP,
which solved it, stopped at whichever local maximum it reached first: over
60 random long-only problems of six assets, 6 came back up to 16 percentage
points of return below the frontier at the requested volatility, with
converged=True. The CHANGELOG entry of 2026-10-22 marked every bounded
answer uncertified and pinned the planted case below as dominated. It is now
solved on the frontier: the critical line algorithm traces the corner
portfolios inside the bounds, and the answer is the mix of the two corners
whose volatilities bracket the target, certified by the frontier's KKT
conditions.

THE PLANTED PROBLEM. Two assets: A at 25% volatility earning 2%, B at 20%
earning 10%, correlated -0.5. A fully-invested portfolio is (t, 1 - t) with
t the weight in A, and its variance is

    0.0625 t^2 + 0.04 (1 - t)^2 - 0.05 t (1 - t) = 0.1525 t^2 - 0.13 t + 0.04.

At a 13% target (variance 0.0169) the roots are t = 0.6 exactly, returning
0.6 * 2% + 0.4 * 10% = 5.20%, and t = 0.0231 / (0.1525 * 0.6) = 0.25246,
returning 7.98%. The minimum-variance mix is t = 0.13 / 0.305 = 0.4262, so
equal weights (t = 0.5) sit on the inefficient side, and SLSQP -- Newton's
method on the variance along the one line of fully-invested portfolios --
landed on t = 0.6. OLD AGAINST NEW: 5.20% before, 7.98% now, 2.78 percentage
points more at the same volatility inside the same bounds. The returns are
built with exactly these sample moments, so nothing depends on a draw.

THE PROBE'S SIXTY PROBLEMS are the ones that measured the defect, replayed
from the same random stream. Each answer is checked against an independent
reference: the maximum return at the target volatility by bisection on the
target return, each step an exact minimum-variance solve at that return.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    run_portfolio_optimization,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import _active_set
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

#: The minimum-variance mix and its volatility, from the same quadratic.
MIN_VARIANCE_T = 0.13 / 0.305
MIN_VOLATILITY = math.sqrt(0.04 - 0.13**2 / (4 * 0.1525))


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


def _the_solve_that_used_to_run(mu, cov, target, bounds):
    """SLSQP as the bounded target_volatility path ran it: from equal
    weights, on the two equalities, with the settings it always had."""
    from scipy.optimize import minimize

    n = len(mu)
    result = minimize(
        lambda w: -float(w @ mu),
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: float(w @ cov @ w) - target**2},
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return result.x


@pytest.fixture(scope="module")
def frame():
    returns = _exact_returns(MU, COV)
    mu, cov = annualized_mean_cov(returns, 252)
    np.testing.assert_allclose(mu, MU, rtol=1e-12)
    np.testing.assert_allclose(cov, COV, rtol=1e-10)
    return returns


class TestThePlantedProblem:
    def test_the_planted_roots_are_the_two_portfolios_at_the_target(self):
        """The oracle, checked before it is used: both mixes are at 13%."""
        for t in (DOMINATED_T, EFFICIENT_T):
            w = np.array([t, 1 - t])
            assert float(np.sqrt(w @ COV @ w)) == pytest.approx(TARGET, abs=1e-12)
        assert DOMINATED_RETURN == pytest.approx(0.052, abs=1e-15)
        assert EFFICIENT_RETURN == pytest.approx(0.0798033, abs=1e-7)

    def test_the_solve_that_used_to_run_stops_on_the_dominated_root(self):
        """What the bounded path returned before: t = 0.6, 5.20%."""
        w = _the_solve_that_used_to_run(MU, COV, TARGET, (0.0, 1.0))
        assert w[0] == pytest.approx(DOMINATED_T, abs=1e-6)
        assert float(w @ MU) == pytest.approx(DOMINATED_RETURN, abs=1e-6)

    def test_the_bounded_solve_returns_the_efficient_root(self, frame):
        """OLD AGAINST NEW. The same request returned 5.20% (t = 0.6) and
        now returns 7.98% (t = 0.25246) at the same 13% volatility: 2.78
        percentage points that were left on the table with
        converged=True."""
        result = mean_variance_optimize(
            frame, "target_volatility", target_volatility=TARGET
        )
        assert result["weights"]["A"] == pytest.approx(EFFICIENT_T, abs=1e-9)
        assert result["expected_volatility"] == pytest.approx(TARGET, abs=1e-10)
        assert result["expected_return"] == pytest.approx(EFFICIENT_RETURN, abs=1e-9)
        gained = result["expected_return"] - DOMINATED_RETURN
        assert gained == pytest.approx(0.0278033, abs=1e-7)
        assert result["converged"] is True
        solver = result["solver"]
        assert solver["method"] == "critical_line"
        assert solver["certified"] is True
        assert solver["certificate"]["stationarity"] <= 1e-12
        assert not any("not convex" in w for w in result["warnings"])

    def test_the_closed_form_takes_the_same_root(self, frame):
        """The same problem without bounds: the frontier algebra's upper
        branch. No bound binds at t = 0.25246, so the bounded answer is the
        same portfolio."""
        bounded = mean_variance_optimize(
            frame, "target_volatility", target_volatility=TARGET
        )
        closed = mean_variance_optimize(
            frame, "target_volatility", target_volatility=TARGET, allow_short=True
        )
        assert closed["solver"]["method"] == "closed_form"
        assert closed["weights"]["A"] == pytest.approx(EFFICIENT_T, abs=1e-9)
        assert bounded["weights"]["A"] == pytest.approx(
            closed["weights"]["A"], abs=1e-12
        )

    def test_a_target_below_the_minimum_variance_volatility_is_refused(self, frame):
        """No portfolio inside the bounds is that quiet. The refusal names
        the attainable range, from the minimum-variance mix t = 0.4262 to
        the highest-return portfolio (all B, 20%)."""
        with pytest.raises(ValidationError) as caught:
            mean_variance_optimize(frame, "target_volatility", target_volatility=0.10)
        text = str(caught.value)
        assert f"[{MIN_VOLATILITY:.6f}, 0.200000]" in text
        assert "objective='min_volatility'" in text

    def test_a_target_above_the_highest_return_portfolio_is_refused(self, frame):
        """Capped at 70%, the highest-return portfolio is (0.3, 0.7), at
        volatility sqrt(0.014725) = 12.13%. A 13% target is attainable
        inside the bounds only by holding MORE of A -- every such portfolio
        earns less than (0.3, 0.7) does at less risk -- so it is refused
        with the range rather than answered with a dominated portfolio."""
        top = math.sqrt(0.0625 * 0.09 + 0.04 * 0.49 - 0.05 * 0.21)
        assert top == pytest.approx(math.sqrt(0.014725), abs=1e-15)
        with pytest.raises(ValidationError) as caught:
            mean_variance_optimize(
                frame, "target_volatility", target_volatility=TARGET, max_weight=0.7
            )
        text = str(caught.value)
        assert f"[{MIN_VOLATILITY:.6f}, {top:.6f}]" in text
        assert "earns 0.076000" in text  # 0.3 * 2% + 0.7 * 10%

    def test_the_ends_of_the_range_are_answered(self, frame):
        """At exactly the minimum volatility the answer is the
        minimum-variance mix, and at the top the highest-return
        portfolio."""
        low = mean_variance_optimize(
            frame, "target_volatility", target_volatility=MIN_VOLATILITY
        )
        assert low["weights"]["A"] == pytest.approx(MIN_VARIANCE_T, abs=1e-9)
        high = mean_variance_optimize(frame, "target_volatility", target_volatility=0.2)
        assert high["weights"]["B"] == pytest.approx(1.0, abs=1e-12)
        assert high["solver"]["certified"] is True


# ── the sixty problems that measured the defect ─────────────────────────


def _probe_problems():
    """The probe's stream: two earlier loops' draws consumed first, then
    sixty long-only problems of six assets, each with a target halfway
    between the minimum and the top volatility of its frontier."""
    rng = np.random.default_rng(1)
    for _ in range(20000):
        n = int(rng.integers(3, 12))
        rng.normal(scale=0.6, size=(n, 2))
        rng.normal(size=(252, 2))
        rng.normal(size=(252, n))
        rng.uniform(0.05, 0.6, size=n)
    for _ in range(20000):
        n = int(rng.integers(3, 8))
        rng.normal(size=252)
        rng.uniform(-1.2, 1.5, size=n)
        rng.normal(size=(252, n))
        rng.uniform(0.2, 1.5, size=n)
        rng.uniform(0.03, 0.6, size=n)
    problems = []
    for _ in range(60):
        n = 6
        X = rng.normal(size=(400, n)) @ rng.normal(scale=0.5, size=(n, n)) * 0.01
        X += rng.normal(scale=0.0005, size=n)
        problems.append(pd.DataFrame(X, columns=[f"a{i}" for i in range(n)]))
    return problems


def _minimum_variance_at(mu, cov, target_return):
    """The exact long-only minimum-variance portfolio at a return."""
    n = len(mu)
    return _active_set.solve(
        2.0 * cov,
        np.zeros(n),
        np.vstack([np.ones(n), mu]),
        np.array([1.0, target_return]),
        0.0,
        np.inf,
    ).x


def _reference_return(mu, cov, target, low, high):
    """The maximum return at volatility `target`, by bisection on the
    return between the minimum-variance and the highest attainable: the
    minimum volatility at a return rises along the efficient branch."""
    for _ in range(80):
        middle = 0.5 * (low + high)
        w = _minimum_variance_at(mu, cov, middle)
        if float(np.sqrt(w @ cov @ w)) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@pytest.fixture(scope="module")
def sixty():
    rows = []
    for frame in _probe_problems():
        mu, cov = annualized_mean_cov(frame, 252)
        least = mean_variance_optimize(frame, "min_volatility")
        top = int(np.argmax(mu))
        top_volatility = float(np.sqrt(cov[top, top]))
        target = least["expected_volatility"] + 0.5 * (
            top_volatility - least["expected_volatility"]
        )
        new = mean_variance_optimize(
            frame, "target_volatility", target_volatility=target
        )
        old = _the_solve_that_used_to_run(mu, cov, target, (0.0, 1.0))
        reference = _reference_return(
            mu, cov, target, least["expected_return"], float(mu[top])
        )
        rows.append(
            dict(
                target=target,
                new=new,
                old_return=float(old @ mu),
                reference=reference,
            )
        )
    return rows


class TestTheProbesSixtyProblems:
    def test_no_answer_is_dominated(self, sixty):
        """Each return equals the bisection reference's to 1e-8, where the
        solve that used to run fell short by up to 16 percentage points."""
        for row in sixty:
            assert row["new"]["expected_return"] == pytest.approx(
                row["reference"], abs=1e-8
            )

    def test_each_target_is_hit_to_1e_10(self, sixty):
        for row in sixty:
            assert abs(row["new"]["expected_volatility"] - row["target"]) <= 1e-10

    def test_each_answer_is_certified(self, sixty):
        for row in sixty:
            solver = row["new"]["solver"]
            assert solver["method"] == "critical_line"
            assert solver["certified"] is True
            assert row["new"]["converged"] is True

    def test_the_solve_that_used_to_run_was_dominated_on_some_of_them(self, sixty):
        """The defect the new answers fix, measured on the same problems:
        the old answer met the volatility and earned less -- on 6 of the
        60, by up to 16.0 percentage points, under scipy 1.17 and 1.18 --
        so the assertions above fail on the old code. Pinned loosely here
        (some, and one by more than a point), since the count is SLSQP's
        path and not this library's. The new answer is never below it."""
        shortfalls = [row["reference"] - row["old_return"] for row in sixty]
        assert sum(gap > 1e-6 for gap in shortfalls) >= 1
        assert max(shortfalls) > 0.01
        for row in sixty:
            assert row["new"]["expected_return"] >= row["old_return"] - 1e-9


class TestTheToolCarriesIt:
    def test_run_portfolio_optimization_reports_the_frontier_portfolio(self, frame):
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
        assert result.converged is True
        assert result.solver.method == "critical_line"
        assert result.solver.certified is True
        assert result.expected_return == pytest.approx(EFFICIENT_RETURN, abs=1e-6)
        assert result.weights["LOW"] == pytest.approx(EFFICIENT_T, abs=1e-6)
        assert not any("not convex" in w for w in result.warnings)
