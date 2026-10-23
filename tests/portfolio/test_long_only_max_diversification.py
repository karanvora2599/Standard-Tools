"""
The long-only (and capped) maximum-diversification portfolio, exactly.

`max_diversification` maximizes the diversification ratio
DR(w) = w'sigma / sqrt(w'Sw) over fully-invested weights. By default the
weights are otherwise free and the maximum is the closed form
w ~ Sigma^-1 sigma -- refused when 1'Sigma^-1 sigma is not positive, since
rescaling would flip it into the minimum (see the CHANGELOG entry of
2026-10-22). With `long_only=True` (and optionally `max_weight`) the weights
are also non-negative (and capped), the maximum always exists, and it is
solved exactly and certified against the ratio's KKT conditions.

WHAT IS CHECKED, AND AGAINST WHAT.
- Closed forms: two assets hold inverse-volatility weights at a ratio of
  sqrt(2 / (1 + rho)); n assets at one correlation hold inverse-volatility
  weights at sqrt(n / (1 + (n - 1) rho)); a binding cap on two assets
  leaves the remainder in the other.
- THE PLANTED SIGN-FLIP CASE, re-derived: three assets at volatilities
  3.17, 4.59 and 0.80 whose unconstrained holding is net short. Long-only,
  C is dropped and A and B hold inverse-volatility weights at a ratio of
  sqrt(2 / (1 - 0.474)) = 1.94994; C's correlation with that portfolio is
  above 1 / DR, which is the condition for leaving it out.
- THE CORE PROPERTY (Choueifaty & Coignard 2008): DR is homogeneous of
  degree zero, so the budget's multiplier is zero and the KKT conditions
  say every held asset's correlation with the portfolio is exactly 1 / DR,
  every asset left at zero has at least that, and every asset at its cap
  at most that. Checked with this file's own arithmetic on random problems.
- A BRUTE FORCE for up to six assets: every assignment of each asset to
  zero, its cap or free; on each face the ratio's maximum over the face's
  affine hull (homogenised: minimize y'Sy with sigma'y = 1, a KKT solve),
  kept when it is inside the bounds; the best ratio among them.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    MaxDiversificationInput,
    optimize_max_diversification,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import max_diversification


def _frame(cov, names=None):
    names = names or [chr(ord("A") + i) for i in range(len(cov))]
    return pd.DataFrame(np.asarray(cov, dtype=float), index=names, columns=names)


def _ratio(w, vols, cov):
    return float(w @ vols) / math.sqrt(float(w @ cov @ w))


def _weights(result):
    return np.array(list(result["weights"].values()))


def _random_covariance(seed, n):
    """A random correlation matrix (some of whose unconstrained holdings
    are net short) and volatilities spread over an order of magnitude."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    C = A @ A.T + 0.05 * np.eye(n)
    d = np.sqrt(np.diag(C))
    vols = np.exp(rng.normal(scale=1.0, size=n))
    return (C / np.outer(d, d)) * np.outer(vols, vols)


def _core_property_violation(w, cov):
    """Long-only, uncapped: how far the held assets' correlations with the
    portfolio are from 1/DR, and how far below it any asset left at zero
    sits. (The budget's multiplier is zero because DR does not change when
    the weights are scaled.)"""
    vols = np.sqrt(np.diag(cov))
    sigma_p = math.sqrt(float(w @ cov @ w))
    rho = (cov @ w) / (vols * sigma_p)
    target = sigma_p / float(w @ vols)  # 1 / DR
    held = w > 0
    return max(
        float(np.max(np.abs(rho[held] - target), initial=0.0)),
        float(np.max(target - rho[~held], initial=0.0)),
    )


def _kkt_violation(w, cov, upper):
    """The ratio's KKT conditions under a cap, from this file's arithmetic:
    dDR/dw level across the held assets, at most that level at zero, at
    least it at the cap, relative to the size of its terms. Scaling the
    weights leaves DR alone, so w'dDR/dw = 0 and the level is minus the
    cap times the capped assets' multipliers: below zero once a cap binds,
    which is why the correlations above are no longer level there."""
    vols = np.sqrt(np.diag(cov))
    sigma_p = math.sqrt(float(w @ cov @ w))
    dr = float(w @ vols) / sigma_p
    gradient = vols / sigma_p - dr * (cov @ w) / sigma_p**2
    scale = float(np.max(vols / sigma_p + dr * (np.abs(cov) @ w) / sigma_p**2))
    at_zero = w <= 0.0
    at_cap = w >= upper
    held = ~(at_zero | at_cap)
    level = float(np.mean(gradient[held])) if held.any() else 0.0
    return (
        max(
            float(np.max(np.abs(gradient[held] - level), initial=0.0)),
            float(np.max(gradient[at_zero] - level, initial=0.0)),
            float(np.max(level - gradient[at_cap], initial=0.0)),
            abs(float(w @ gradient)),
        )
        / scale
    )


def _brute_force(cov, cap):
    """The best ratio over every face of the box, with its weights."""
    n = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    upper = np.inf if cap is None else cap
    best = (-np.inf, None)
    for status in itertools.product((-1, 0, 1), repeat=n):
        status = np.array(status)
        if cap is None and np.any(status == 1):
            continue
        free = np.flatnonzero(status == 0)
        capped = np.flatnonzero(status == 1)
        if free.size == 0:
            w = np.where(status == 1, cap, 0.0)
            if abs(w.sum() - 1.0) > 1e-12:
                continue
        else:
            # y = P x with x = (y_free, s): y at the cap is cap * s, at
            # zero is zero, and s = 1'y is what the weights are scaled by.
            k = free.size
            P = np.zeros((n, k + 1))
            P[free, np.arange(k)] = 1.0
            P[capped, k] = cap if cap is not None else 0.0
            ones = np.zeros(k + 1)
            ones[k] = 1.0
            C = np.vstack([vols @ P, np.ones(n) @ P - ones])
            K = np.zeros((k + 3, k + 3))
            K[: k + 1, : k + 1] = 2.0 * P.T @ cov @ P
            K[: k + 1, k + 1 :] = C.T
            K[k + 1 :, : k + 1] = C
            rhs = np.zeros(k + 3)
            rhs[k + 1] = 1.0
            x = np.linalg.lstsq(K, rhs, rcond=None)[0]
            if not x[k] > 0 or np.max(np.abs(C @ x[: k + 1] - [1.0, 0.0])) > 1e-10:
                continue
            w = (P @ x[: k + 1]) / x[k]
        if np.any(w < -1e-12) or np.any(w > upper + 1e-12):
            continue
        ratio = _ratio(w, vols, cov)
        if ratio > best[0]:
            best = (ratio, w)
    return best


# ── closed forms ────────────────────────────────────────────────────────


class TestClosedForms:
    def test_two_assets_hold_inverse_volatility_weights(self):
        vols, rho = np.array([0.1, 0.3]), 0.2
        corr = np.array([[1.0, rho], [rho, 1.0]])
        result = max_diversification(
            _frame(corr * np.outer(vols, vols)), long_only=True
        )
        np.testing.assert_allclose(_weights(result), [0.75, 0.25], atol=1e-14)
        assert result["diversification_ratio"] == pytest.approx(
            math.sqrt(2 / (1 + rho)), rel=1e-14
        )
        assert result["certified"] is True

    def test_one_correlation_holds_inverse_volatility_weights(self):
        n, rho = 5, 0.3
        vols = np.array([0.1, 0.15, 0.2, 0.3, 0.45])
        corr = np.full((n, n), rho) + (1 - rho) * np.eye(n)
        result = max_diversification(
            _frame(corr * np.outer(vols, vols)), long_only=True
        )
        expected = (1 / vols) / (1 / vols).sum()
        np.testing.assert_allclose(_weights(result), expected, atol=1e-13)
        assert result["diversification_ratio"] == pytest.approx(
            math.sqrt(n / (1 + (n - 1) * rho)), rel=1e-13
        )

    def test_a_positive_holding_is_the_closed_form_either_way(self):
        """When Sigma^-1 sigma is already positive the long-only maximum is
        the unconstrained one."""
        vols = np.array([0.2, 0.25, 0.22, 0.15])
        corr = np.array(
            [
                [1, 0.3, 0.2, 0.1],
                [0.3, 1, 0.4, 0.2],
                [0.2, 0.4, 1, 0.3],
                [0.1, 0.2, 0.3, 1],
            ]
        )
        cov = corr * np.outer(vols, vols)
        assert np.all(np.linalg.solve(cov, vols) > 0)
        loose = max_diversification(_frame(cov))
        tight = max_diversification(_frame(cov), long_only=True)
        np.testing.assert_allclose(_weights(tight), _weights(loose), atol=1e-14)

    def test_a_binding_cap_on_two_assets_leaves_the_rest_in_the_other(self):
        """Uncapped the two hold (0.75, 0.25); capped at 0.6 the only
        portfolio with the first at its cap is (0.6, 0.4)."""
        cov = np.array([[0.01, 0.006], [0.006, 0.09]])
        result = max_diversification(_frame(cov), long_only=True, max_weight=0.6)
        np.testing.assert_allclose(_weights(result), [0.6, 0.4], atol=1e-15)
        expected = 0.18 / math.sqrt(0.36 * 0.01 + 0.16 * 0.09 + 2 * 0.24 * 0.006)
        assert result["diversification_ratio"] == pytest.approx(expected, rel=1e-14)
        assert result["certified"] is True


# ── the planted case whose unconstrained holding is net short ───────────


VOLS = np.array([3.1731706, 4.58591887, 0.80118955])
CORR = np.array([[1.0, -0.474, 0.792], [-0.474, 1.0, 0.144], [0.792, 0.144, 1.0]])
SIGMA = CORR * np.outer(VOLS, VOLS)


class TestThePlantedSignFlipCase:
    def test_the_long_only_maximum_rederived(self):
        """Dropping C, A and B hold inverse-volatility weights (two assets)
        at sqrt(2 / (1 + rho_AB)); C's correlation with that portfolio is
        above 1 / DR, so adding any of it lowers the ratio: this is the
        long-only maximum, and the library returns it."""
        w = np.array([VOLS[1], VOLS[0], 0.0]) / (VOLS[0] + VOLS[1])
        dr = math.sqrt(2 / (1 - 0.474))
        assert dr == pytest.approx(1.9499439, abs=1e-7)
        assert _ratio(w, VOLS, SIGMA) == pytest.approx(dr, rel=1e-14)
        sigma_p = math.sqrt(float(w @ SIGMA @ w))
        rho_c = float((SIGMA @ w)[2]) / (VOLS[2] * sigma_p)
        assert rho_c > 1 / dr + 0.1

        result = max_diversification(_frame(SIGMA), long_only=True)
        np.testing.assert_allclose(_weights(result), w, atol=1e-14)
        assert result["diversification_ratio"] == pytest.approx(dr, rel=1e-14)
        assert result["certified"] is True
        assert result["kkt_residual"] <= 1e-12

    def test_the_unconstrained_request_is_still_refused_and_names_the_fix(self):
        with pytest.raises(ValidationError, match="Pass long_only=True") as caught:
            max_diversification(_frame(SIGMA))
        assert "its minimum (-5.3225)" in str(caught.value)

    def test_the_tool_answers_it_long_only(self):
        result = optimize_max_diversification(
            MaxDiversificationInput(
                assets=["A", "B", "C"], covariance=SIGMA.tolist(), long_only=True
            )
        )
        assert result.diversification_ratio == pytest.approx(1.9499439, abs=1e-7)
        assert result.weights["C"] == 0.0
        assert result.long_only is True
        assert result.certified is True
        assert result.method in ("active_set", "critical_line")


# ── the core property and the brute force ───────────────────────────────


RANDOM = [
    (seed, n, cap)
    for seed, n in [(11, 3), (12, 4), (13, 5), (14, 6), (15, 6), (16, 5)]
    for cap in (None, 0.4)
]


@pytest.mark.parametrize(
    "seed,n,cap",
    RANDOM,
    ids=[f"{n}-{seed}-{'cap' if c else 'long'}" for seed, n, c in RANDOM],
)
class TestRandomProblems:
    def test_the_kkt_conditions_hold(self, seed, n, cap):
        """And without a cap, the core property: every held asset's
        correlation with the portfolio is 1 / DR."""
        cov = _random_covariance(seed, n)
        cap = None if cap is not None and cap * n < 1 else cap
        result = max_diversification(_frame(cov), long_only=True, max_weight=cap)
        w = _weights(result)
        upper = np.inf if cap is None else cap
        assert np.all(w >= 0) and np.all(w <= upper)
        assert abs(w.sum() - 1) <= 1e-14
        assert _kkt_violation(w, cov, upper) <= 1e-12
        if cap is None:
            assert _core_property_violation(w, cov) <= 1e-12
        assert result["certified"] is True

    def test_the_brute_force_agrees(self, seed, n, cap):
        cov = _random_covariance(seed, n)
        cap = None if cap is not None and cap * n < 1 else cap
        result = max_diversification(_frame(cov), long_only=True, max_weight=cap)
        best, w = _brute_force(cov, cap)
        assert result["diversification_ratio"] == pytest.approx(best, rel=1e-10)
        np.testing.assert_allclose(_weights(result), w, atol=1e-8)


def test_a_net_short_holding_is_among_the_random_problems():
    """The brute force is not only checking easy cases: some of these
    holdings are net short or short something, where the unconstrained
    tool refuses or shorts."""
    short = 0
    for seed, n, _ in RANDOM:
        cov = _random_covariance(seed, n)
        direction = np.linalg.solve(cov, np.sqrt(np.diag(cov)))
        short += int(np.any(direction < 0))
    assert short >= 3


# ── what the tool takes and says ────────────────────────────────────────


class TestTheTool:
    def test_a_cap_without_long_only_is_refused(self):
        with pytest.raises(pydantic.ValidationError, match="long_only=True"):
            MaxDiversificationInput(
                assets=["A", "B", "C"], covariance=SIGMA.tolist(), max_weight=0.5
            )

    def test_an_infeasible_cap_is_refused_by_name(self):
        with pytest.raises(ValidationError, match=r"max_weight=0\.3 is infeasible"):
            optimize_max_diversification(
                MaxDiversificationInput(
                    assets=["A", "B", "C"],
                    covariance=SIGMA.tolist(),
                    long_only=True,
                    max_weight=0.3,
                )
            )

    def test_the_default_is_the_closed_form_unchanged(self):
        corr = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
        cov = corr * 0.04
        default = optimize_max_diversification(
            MaxDiversificationInput(assets=["A", "B", "C"], covariance=cov.tolist())
        )
        assert default.method == "closed_form"
        assert default.certified is None
        assert default.kkt_residual is None
        assert default.long_only is False
        assert not any("kkt_residual" in w for w in default.warnings)
        direct = max_diversification(_frame(cov))
        assert default.weights == direct["weights"]

    def test_it_dispatches_through_the_runtime(self):
        payload = resolve("portfolio").dispatch(
            "optimize_max_diversification",
            {
                "assets": ["A", "B", "C"],
                "covariance": SIGMA.tolist(),
                "long_only": True,
                "max_weight": 0.55,
            },
        )
        assert payload["certified"] is True
        assert max(payload["weights"].values()) <= 0.55 + 1e-15
        best, _ = _brute_force(SIGMA, 0.55)
        assert payload["diversification_ratio"] == pytest.approx(best, rel=1e-10)
