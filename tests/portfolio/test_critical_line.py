"""
The bounded efficient frontier, exactly: Markowitz's critical line algorithm.

`portfolio.critical_line` traces the corner portfolios of the frontier under
a budget and per-asset bounds -- where an asset reaches or leaves a bound --
from the highest-return portfolio to the minimum-variance one, each with its
risk tolerance lam, return and volatility. Between adjacent corners the
frontier is the mix of the two. Every claim is checked against something the
algorithm did not compute:

- THE CLOSED FORM. Where no bound binds, the bounded frontier is the
  unbounded one, whose weights and slope are the Merton constants'.
- A CERTIFICATE PER CORNER, from this file's own arithmetic: the corner's
  S w - lam mu is level across its free assets (the budget's multiplier)
  and on the right side of that level at every bound, which makes it the
  minimum-variance portfolio at its own return under the bounds. Each corner
  is also the exact active-set solution of that minimum-variance problem.
- A BRUTE FORCE for up to five assets: every assignment of each asset to
  its lower bound, its upper bound or free, the equality-constrained
  minimum on each, and the least variance among the feasible ones. The
  frontier's variance at every return, and its maximum return at every
  volatility (by bisection on the brute force), agree to 1e-8.
- DEGENERATE CASES, by name: tied expected returns at the highest-return
  end, a start with no asset strictly inside its bounds, two assets that
  enter at the same lam, every return tied, and a covariance that is
  singular on the free assets (refused).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import _active_set
from standard_quant_tools.portfolio.critical_line import MAX_ASSETS, critical_line
from standard_quant_tools.portfolio.optimize import (
    HAS_SCIPY,
    annualized_mean_cov,
    frontier_stats,
    frontier_weights,
    mean_variance_optimize,
)


def _problem(seed, n):
    """A sample covariance of n correlated assets and expected returns of
    both signs, annualized."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, n)) @ rng.normal(scale=0.5, size=(n, n)) * 0.01
    cov = np.cov(X, rowvar=False) * 252
    mu = rng.normal(0.08, 0.05, n)
    return mu, cov


def _bounds(kind, n, seed):
    rng = np.random.default_rng(seed + 1000)
    if kind == "long_only":
        return np.zeros(n), np.full(n, np.inf)
    if kind == "capped":
        cap = max(1.0 / n + 0.05, float(rng.uniform(0.2, 0.6)))
        return np.zeros(n), np.full(n, cap)
    cap = max(1.0 / n + 0.01, float(rng.uniform(0.3, 1.5)))
    return np.full(n, -cap), np.full(n, cap)


CASES = [
    (seed, n, kind)
    for seed, n in [(1, 3), (2, 5), (3, 8), (4, 12)]
    for kind in ("long_only", "capped", "shorting")
]
CASE_IDS = [f"{kind}-{n}-{seed}" for seed, n, kind in CASES]


def _kkt_violation(w, lam, mu, cov, lo, hi):
    """This file's own check of the programme 1/2 w'Sw - lam mu'w under the
    budget and the bounds: the spread of S w - lam mu over the free assets,
    and how far a bound asset's entry sits on the wrong side of that level,
    both relative to the terms the entries are made of."""
    g = cov @ w - lam * mu
    scale = float(np.max(np.abs(cov) @ np.abs(w) + abs(lam) * np.abs(mu)))
    width = 1e-9 * max(1.0, float(np.max(np.abs(w))))
    at_lo = w <= lo + width
    at_hi = (w >= hi - width) & ~at_lo
    free = ~(at_lo | at_hi)
    if free.any():
        level = float(np.mean(g[free]))
        spread = float(np.max(np.abs(g[free] - level)))
    else:  # a vertex: any level between the bound entries will do
        level = 0.5 * (
            float(np.min(g[at_lo], initial=np.inf))
            + float(np.max(g[at_hi], initial=-np.inf))
        )
        spread = 0.0
    wrong = max(
        float(np.max(level - g[at_lo], initial=0.0)),
        float(np.max(g[at_hi] - level, initial=0.0)),
    )
    return max(spread, wrong) / scale


def _parametric_qp(mu, cov, lam, lo, hi):
    """The exact active-set solution of minimize 1/2 w'Sw - lam mu'w under
    the budget and the bounds, solved on its own: from a cold start, and
    from equal weights by the primal method when that cycles."""
    n = len(mu)
    ones = np.ones((1, n))
    try:
        return _active_set.solve(cov, -lam * mu, ones, [1.0], lo, hi).x
    except _active_set.ActiveSetFailure:
        return _active_set.solve_primal(
            cov, -lam * mu, ones, [1.0], lo, hi, np.full(n, 1.0 / n)
        ).x


def _brute_force_variance(mu, cov, r, lo, hi):
    """The least variance at return r over every face of the box: each
    asset at its lower bound, its upper bound or free; on each face the
    equality-constrained minimum (a KKT solve), kept when it is inside the
    bounds. The optimum lies on the face whose relative interior holds it,
    where it is that face's minimum, so the least feasible one is it."""
    n = len(mu)
    best = np.inf
    for status in itertools.product((-1, 0, 1), repeat=n):
        status = np.array(status)
        if np.any((status == 1) & ~np.isfinite(hi)):
            continue
        fixed = status != 0
        free = np.flatnonzero(~fixed)
        w = np.where(status == -1, lo, np.where(status == 1, hi, 0.0))
        if free.size == 0:
            if abs(w.sum() - 1) > 1e-12 or abs(mu @ w - r) > 1e-12:
                continue
            best = min(best, float(w @ cov @ w))
            continue
        A = np.vstack([np.ones(free.size), mu[free]])
        b = np.array([1.0 - w[fixed].sum(), r - mu[fixed] @ w[fixed]])
        k = free.size
        K = np.zeros((k + 2, k + 2))
        K[:k, :k] = cov[np.ix_(free, free)]
        K[:k, k:] = A.T
        K[k:, :k] = A
        rhs = np.concatenate([-cov[np.ix_(free, np.flatnonzero(fixed))] @ w[fixed], b])
        x = np.linalg.lstsq(K, rhs, rcond=None)[0]
        w[free] = x[:k]
        if abs(w.sum() - 1) > 1e-10 or abs(mu @ w - r) > 1e-10:
            continue
        if np.any(w < lo - 1e-12) or np.any(w > hi + 1e-12):
            continue
        best = min(best, float(w @ cov @ w))
    return best


# ── where no bound binds, the closed form ───────────────────────────────


class TestSlackBoundsGiveTheClosedForm:
    def test_points_and_slopes_match_the_merton_constants(self):
        """Bounds of +/-5 on five assets bind only near the top of the
        curve. Every frontier point whose closed-form weights sit inside
        them is the closed form's to 1e-12, and its risk tolerance is half
        the closed form's slope of variance against return, (A r - B) / D."""
        mu, cov = _problem(21, 5)
        frontier = critical_line(mu, cov, -5.0, 5.0)
        sigma_inv, ones, A, B, C, D = frontier_stats(mu, cov)
        low, high = frontier.return_range()
        checked = 0
        for r in np.linspace(low, high, 41):
            exact = frontier_weights(sigma_inv, ones, mu, A, B, C, D, float(r))
            if np.max(np.abs(exact)) >= 5.0:
                continue
            point = frontier.at_return(float(r))
            np.testing.assert_allclose(point.weights, exact, rtol=0, atol=1e-12)
            assert point.lam == pytest.approx((A * r - B) / D, rel=1e-10, abs=1e-12)
            checked += 1
        assert checked >= 10

    def test_the_minimum_variance_end_is_the_global_minimum(self):
        """lam = 0: the bounded minimum-variance portfolio, here inside the
        bounds, is Sigma^-1 1 / 1'Sigma^-1 1."""
        mu, cov = _problem(22, 6)
        frontier = critical_line(mu, cov, -5.0, 5.0)
        inverse = np.linalg.inv(cov)
        gmv = inverse.sum(axis=1) / inverse.sum()
        end = frontier.min_variance
        assert end.lam == 0.0
        np.testing.assert_allclose(end.weights, gmv, rtol=0, atol=1e-13)


# ── every corner certifies ──────────────────────────────────────────────


@pytest.mark.parametrize("seed,n,kind", CASES, ids=CASE_IDS)
class TestEveryCornerCertifies:
    def test_the_library_certificate_passes(self, seed, n, kind):
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        assert frontier.certified, frontier.worst_residual
        for corner in frontier.corners:
            assert corner.certificate.worst <= _active_set.CERTIFICATE_TOLERANCE

    def test_this_files_kkt_check_passes(self, seed, n, kind):
        """S w - lam mu level on the free assets and on the right side of
        the level at every bound: each corner minimizes variance at its
        return under the bounds."""
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        lams = [c.lam for c in frontier.corners]
        assert lams == sorted(lams, reverse=True)
        assert lams[-1] >= 0.0
        for corner in frontier.corners:
            # Each corner is stepped from the one before, so the budget's
            # residual accumulates: 1.5e-13 on twelve assets on one BLAS.
            assert abs(corner.weights.sum() - 1.0) <= 1e-12
            assert np.all(corner.weights >= lo) and np.all(corner.weights <= hi)
            assert _kkt_violation(corner.weights, corner.lam, mu, cov, lo, hi) <= 1e-12

    def test_each_corner_is_the_exact_solution_at_its_lam(self, seed, n, kind):
        """The quadratic programme at the corner's lam, solved by the
        active-set method with no help from the trace, gives the corner."""
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        for corner in frontier.corners:
            qp = _parametric_qp(mu, cov, corner.lam, lo, hi)
            np.testing.assert_allclose(corner.weights, qp, rtol=0, atol=1e-9)

    def test_returns_and_volatilities_fall_along_the_corners(self, seed, n, kind):
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        corners = critical_line(mu, cov, lo, hi).corners
        for high, low in zip(corners, corners[1:]):
            assert low.expected_return <= high.expected_return + 1e-15
            assert low.variance <= high.variance + 1e-15

    def test_each_corner_names_what_changed(self, seed, n, kind):
        corners = critical_line(*_problem(seed, n), *_bounds(kind, n, seed)).corners
        for corner in corners[:-1]:
            assert corner.events
            for event in corner.events:
                assert event.kind in {
                    "leaves_lower",
                    "leaves_upper",
                    "reaches_lower",
                    "reaches_upper",
                }


# ── a brute force over every face ───────────────────────────────────────


BRUTE = [
    (31, 3, "long_only"),
    (32, 4, "capped"),
    (33, 4, "shorting"),
    (34, 5, "capped"),
]


@pytest.mark.parametrize("seed,n,kind", BRUTE, ids=[f"{k}-{n}" for _, n, k in BRUTE])
class TestABruteForceAgrees:
    def test_the_variance_at_every_return(self, seed, n, kind):
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        low, high = frontier.return_range()
        for r in np.linspace(low, high, 13):
            point = frontier.at_return(float(r))
            brute = _brute_force_variance(mu, cov, float(r), lo, hi)
            assert point.variance == pytest.approx(brute, rel=1e-8, abs=1e-12)

    def test_the_highest_return_at_every_volatility(self, seed, n, kind):
        """The brute force's frontier inverted by bisection on the return:
        the highest return whose least variance is at most the target."""
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        r_low, r_high = frontier.return_range()
        v_low, v_high = frontier.volatility_range()
        for target in np.linspace(v_low, v_high, 5)[1:-1]:
            a, b = r_low, r_high
            for _ in range(45):
                middle = 0.5 * (a + b)
                if _brute_force_variance(mu, cov, middle, lo, hi) < target**2:
                    a = middle
                else:
                    b = middle
            point = frontier.at_volatility(float(target))
            assert point.volatility == pytest.approx(target, abs=1e-12)
            assert point.expected_return == pytest.approx(0.5 * (a + b), abs=1e-8)

    def test_a_target_a_few_units_from_an_end_is_that_end(self, seed, n, kind):
        """A volatility written as an end corner's own comes back through
        its square a few rounding units from that corner's variance. At the
        minimum-variance corner the variance is flat in the weights, so
        solving for that sliver moved the weights by 1e-8 on one BLAS and not
        another; within a few units the answer is the corner itself."""
        frontier = critical_line(*_problem(seed, n), *_bounds(kind, n, seed))
        eps = float(np.finfo(float).eps)
        bottom, top = frontier.corners[-1], frontier.corners[0]
        for units in (0.0, 4.0):
            near_bottom = float(np.sqrt(bottom.variance * (1.0 + units * eps)))
            assert np.array_equal(
                frontier.at_volatility(near_bottom).weights, bottom.weights
            )
            near_top = float(np.sqrt(top.variance * (1.0 - units * eps)))
            assert np.array_equal(frontier.at_volatility(near_top).weights, top.weights)

    def test_the_ends_are_the_extremes(self, seed, n, kind):
        """The top corner earns the highest return any portfolio inside the
        bounds can (a linear programme), and the last has the least
        variance of any."""
        mu, cov = _problem(seed, n)
        lo, hi = _bounds(kind, n, seed)
        frontier = critical_line(mu, cov, lo, hi)
        w = lo.copy()
        left = 1.0 - lo.sum()
        for i in np.argsort(-mu, kind="stable"):
            add = min(hi[i] - lo[i], left)
            w[i] += add
            left -= add
        assert frontier.top.expected_return == pytest.approx(float(mu @ w), abs=1e-14)
        rs = np.linspace(*frontier.return_range(), 9)
        least = min(_brute_force_variance(mu, cov, float(r), lo, hi) for r in rs)
        assert frontier.min_variance.variance <= least + 1e-14


# ── the maximum-Sharpe portfolio on the bounded frontier ────────────────


@pytest.mark.skipif(not HAS_SCIPY, reason="the exact max_sharpe path needs scipy")
@pytest.mark.parametrize("max_weight", [None, 0.35])
def test_the_tangency_is_the_certified_max_sharpe(max_weight):
    """The frontier's maximum Sharpe ratio, exact piece by piece, is the
    portfolio the exact max_sharpe solve certifies inside the same bounds."""
    rng = np.random.default_rng(41)
    market = rng.normal(0.0004, 0.01, 800)
    frame = {
        f"S{i}": 0.7 * market + rng.normal(0.0002 * (i + 1), 0.009 + 0.001 * i, 800)
        for i in range(5)
    }
    import pandas as pd

    frame = pd.DataFrame(frame)
    mu, cov = annualized_mean_cov(frame, 252)
    upper = np.inf if max_weight is None else max_weight
    tangency = critical_line(mu, cov, 0.0, upper).max_sharpe(0.01)
    solved = mean_variance_optimize(
        frame, "max_sharpe", risk_free_rate=0.01, max_weight=max_weight
    )
    assert solved["solver"]["certified"] is True
    weights = np.array(list(solved["weights"].values()))
    np.testing.assert_allclose(tangency.weights, weights, rtol=0, atol=1e-9)


def test_no_tangency_when_nothing_earns_the_rate():
    mu, cov = _problem(51, 4)
    frontier = critical_line(mu, cov, 0.0, np.inf)
    assert frontier.max_sharpe(frontier.top.expected_return + 0.01) is None


# ── degenerate cases, handled or refused by name ────────────────────────


VOLS = np.array([0.2, 0.25, 0.22, 0.15])
CORR = np.array(
    [[1, 0.3, 0.2, 0.1], [0.3, 1, 0.4, 0.2], [0.2, 0.4, 1, 0.3], [0.1, 0.2, 0.3, 1]]
)
SIGMA = CORR * np.outer(VOLS, VOLS)


class TestDegenerateCases:
    def test_tied_returns_at_the_top_start_at_their_least_variance_split(self):
        """Capped at 50%, A fills first and B and C tie at 8% for the other
        half: every split earns the same, and the frontier's limit as lam
        grows is the split of least variance -- the minimum of the
        portfolio's variance over w_B + w_C = 0.5, found here in closed
        form (the split is interior)."""
        mu = np.array([0.10, 0.08, 0.08, 0.05])
        frontier = critical_line(mu, SIGMA, 0.0, 0.5, names="ABCD")
        top = frontier.top.weights
        assert top[0] == 0.5 and top[3] == 0.0
        # var((0.5, t, 0.5 - t, 0)) is least where (S w)_B = (S w)_C.
        S = SIGMA
        t = (
            0.5
            * ((S[2, 0] - S[1, 0]) + (S[2, 2] - S[1, 2]))
            / (S[1, 1] + S[2, 2] - 2 * S[1, 2])
        )
        assert top[1] == pytest.approx(t, abs=1e-14)
        assert top[2] == pytest.approx(0.5 - t, abs=1e-14)
        assert frontier.certified
        lo, hi = np.zeros(4), np.full(4, 0.5)
        for corner in frontier.corners:
            brute = _brute_force_variance(mu, S, corner.expected_return, lo, hi)
            assert corner.variance == pytest.approx(brute, rel=1e-10)

    def test_a_start_with_no_free_asset_moves_a_pair(self):
        """Capped at 50% with distinct returns, the highest-return portfolio
        is (0.5, 0.5, 0, 0): no asset strictly inside its bounds, so the
        budget's multiplier is not pinned there. The first corner frees B
        from its cap and C from zero together."""
        mu = np.array([0.10, 0.09, 0.08, 0.05])
        frontier = critical_line(mu, SIGMA, 0.0, 0.5, names="ABCD")
        first = frontier.corners[0]
        np.testing.assert_array_equal(first.weights, [0.5, 0.5, 0.0, 0.0])
        assert [e.describe(frontier.names) for e in first.events] == [
            "B leaves its upper bound",
            "C leaves its lower bound",
        ]
        assert frontier.certified

    def test_two_assets_entering_at_one_lam_enter_together(self):
        """B and C are mirror images (same return, volatility and
        correlation with A), so they leave zero at the same lam."""
        mu = np.array([0.10, 0.05, 0.05])
        vols = np.array([0.2, 0.15, 0.15])
        corr = np.array([[1, 0.2, 0.2], [0.2, 1, 0.3], [0.2, 0.3, 1]])
        cov = corr * np.outer(vols, vols)
        frontier = critical_line(mu, cov, 0.0, np.inf, names="ABC")
        first = frontier.corners[0]
        assert {e.describe(frontier.names) for e in first.events} == {
            "B leaves its lower bound",
            "C leaves its lower bound",
        }
        end = frontier.min_variance.weights
        assert end[1] == pytest.approx(end[2], abs=1e-15)
        assert frontier.certified

    def test_every_return_tied_is_one_portfolio(self):
        """Every portfolio earns the same: the frontier is the
        minimum-variance portfolio alone."""
        mu = np.full(4, 0.07)
        frontier = critical_line(mu, SIGMA, 0.0, np.inf)
        assert len(frontier.corners) == 1
        low, high = frontier.return_range()
        assert low == high == pytest.approx(0.07, abs=1e-15)
        inverse = np.linalg.inv(SIGMA)
        gmv = inverse.sum(axis=1) / inverse.sum()
        assert np.all(gmv > 0)
        np.testing.assert_allclose(frontier.top.weights, gmv, rtol=0, atol=1e-14)

    def test_a_covariance_singular_on_the_free_assets_is_refused(self):
        """B has no variance. The frontier starts all in A and B leaves its
        floor at lam = 0.04 / 0.08 -- where S restricted to {A, B} is
        singular, and each piece is a solve on it."""
        cov = np.array([[0.04, 0.0], [0.0, 0.0]])
        mu = np.array([0.10, 0.02])
        with pytest.raises(
            ValidationError, match=r"\['A', 'B'\] is singular .* at lam=0\.5,"
        ):
            critical_line(mu, cov, 0.0, np.inf, names="AB")

    def test_a_zero_variance_direction_at_the_end_is_not_a_corner(self):
        """C is the average of A and B and earns more than their average, so
        S is singular on {A, B, C} but no frontier portfolio above lam = 0
        holds all three: the arbitrage between them pushes A or B to zero.
        An event the arithmetic puts within rounding of lam = 0 is the
        minimum-variance end, not a corner on that singular set."""
        M = np.array([[0.2, 0.0], [0.0, 0.3], [0.1, 0.15]])
        cov = M @ M.T
        mu = np.array([0.06, 0.10, 0.09])
        frontier = critical_line(mu, cov, 0.0, np.inf, names="ABC")
        assert frontier.certified
        assert all(np.sum(c.weights > 0) <= 2 for c in frontier.corners)

    def test_an_infinite_floor_is_refused(self):
        mu, cov = _problem(61, 3)
        with pytest.raises(ValidationError, match="lower bound must be finite"):
            critical_line(mu, cov, -np.inf, 1.0)

    def test_bounds_that_cannot_meet_the_budget_are_refused(self):
        mu, cov = _problem(62, 3)
        with pytest.raises(ValidationError, match="sum to between 0 and 0.9"):
            critical_line(mu, cov, 0.0, 0.3)

    def test_too_many_assets_are_refused_by_name(self):
        n = MAX_ASSETS + 1
        with pytest.raises(ValidationError, match=f"{n} assets"):
            critical_line(np.zeros(n), np.eye(n), 0.0, np.inf)


# ── points between corners ──────────────────────────────────────────────


class TestPointsBetweenCorners:
    def test_a_target_outside_the_frontier_is_refused_with_its_range(self):
        mu, cov = _problem(71, 4)
        frontier = critical_line(mu, cov, 0.0, np.inf)
        low, high = frontier.volatility_range()
        with pytest.raises(ValidationError, match=f"{low:.6f}, {high:.6f}"):
            frontier.at_volatility(high * 1.01)
        r_low, r_high = frontier.return_range()
        with pytest.raises(ValidationError, match=f"{r_low:.6f}, {r_high:.6f}"):
            frontier.at_return(r_low - 0.01)

    def test_a_point_between_corners_is_their_mix(self):
        mu, cov = _problem(72, 6)
        frontier = critical_line(mu, cov, 0.0, 0.4)
        corners = frontier.corners
        k = len(corners) // 2
        high, low = corners[k - 1], corners[k]
        r = 0.3 * high.expected_return + 0.7 * low.expected_return
        point = frontier.at_return(r)
        np.testing.assert_allclose(
            point.weights, 0.3 * high.weights + 0.7 * low.weights, atol=1e-15
        )
        assert point.lam == pytest.approx(0.3 * high.lam + 0.7 * low.lam, rel=1e-14)
        assert point.certificate.certified

    def test_the_same_input_gives_the_same_bits(self):
        mu, cov = _problem(73, 10)
        one = critical_line(mu, cov, 0.0, 0.25)
        two = critical_line(mu.copy(), cov.copy(), 0.0, 0.25)
        assert len(one.corners) == len(two.corners)
        for a, b in zip(one.corners, two.corners):
            assert a.lam == b.lam
            assert a.weights.tobytes() == b.weights.tobytes()
        point = one.at_volatility(math.sqrt(one.corners[1].variance) * 0.99)
        again = two.at_volatility(math.sqrt(two.corners[1].variance) * 0.99)
        assert point.weights.tobytes() == again.weights.tobytes()
