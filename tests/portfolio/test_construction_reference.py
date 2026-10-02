"""
`risk_parity`, `_quasi_diagonal_order` and `estimate_covariance` against the
implementations they replaced (see the CHANGELOG entry of 2026-10-01).

Each reference is the pre-change code, verbatim. Two of the three changes
are exact and compared exactly. `risk_parity` is not bit-identical and
cannot be: it now updates the marginal-risk vector by one column per
coordinate step instead of recomputing the whole product, which rounds
differently in the last bits. It is compared at rtol=1e-12 on the weights,
with the same iteration count and convergence flag.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import construction, covariance
from standard_quant_tools.portfolio.construction import (
    _covariance_frame,
    _portfolio_volatility,
    _risk_contributions,
)
from standard_quant_tools.portfolio.covariance import (
    METHODS,
    _ewma_covariance,
    _shrink_to_identity,
    _warnings,
)

# ── the implementations before the change, verbatim ──────────────────────


def _reference_risk_parity(
    covariance: Any,
    *,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    budget: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    frame = _covariance_frame(covariance, "risk_parity")
    psd_notes = list(frame.attrs.get("warnings", []))
    matrix = frame.to_numpy()
    n = matrix.shape[0]
    if n < 2:
        raise ValidationError("risk_parity: needs at least two assets.")

    if budget is None:
        targets = np.full(n, 1.0 / n)
    else:
        targets = np.asarray([float(b) for b in budget], dtype=float)
        if targets.size != n:
            raise ValidationError(
                f"risk_parity: budget has {targets.size} entries for {n} assets."
            )
        if (targets <= 0).any():
            raise ValidationError(
                "risk_parity: every risk budget must be positive. A zero "
                "budget means 'do not hold this asset', which is an "
                "exclusion rather than a budget -- drop it from the "
                "covariance matrix instead."
            )
        targets = targets / targets.sum()

    weights = 1.0 / np.sqrt(np.diag(matrix))
    weights = weights / weights.sum()

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        previous = weights.copy()
        volatility = _portfolio_volatility(weights, matrix)
        if volatility <= 0:
            break
        marginal = matrix @ weights
        for i in range(n):
            others = marginal[i] - matrix[i, i] * weights[i]
            discriminant = others**2 + 4.0 * matrix[i, i] * targets[i] * volatility**2
            weights[i] = (-others + math.sqrt(max(discriminant, 0.0))) / (
                2.0 * matrix[i, i]
            )
            marginal = matrix @ weights
        weights = weights / weights.sum()
        if np.max(np.abs(weights - previous)) < tolerance:
            converged = True
            break

    contributions = _risk_contributions(weights, matrix)
    total = contributions.sum()
    shares = contributions / total if total > 0 else contributions
    error = float(np.max(np.abs(shares - targets)))

    warnings: List[str] = list(psd_notes)
    if not converged:
        warnings.append(
            f"DID NOT CONVERGE in {max_iterations} iterations (largest "
            f"risk-share error {error:.2e}). The weights returned are the "
            "last iterate, and they are not a risk parity portfolio. Using "
            "them as one is worse than not having them -- usually the "
            "covariance matrix is near-singular."
        )
    if error > 1e-4:
        warnings.append(
            f"The largest deviation from the target risk share is "
            f"{error:.2e}, which is above what a converged solution should "
            "show."
        )
    warnings.append(
        "Risk parity uses NO expected returns, which is the point: the "
        "standard error on a mean return from two years of daily data is "
        "about the size of the estimate itself, and mean-variance is "
        "maximally sensitive to exactly that input.",
    )
    warnings.append(
        "Equal RISK contribution is not equal weight and not equal return "
        "expectation. A risk parity portfolio is implicitly betting that "
        "Sharpe ratios are similar across assets; where they are not, it "
        "over-weights the low-Sharpe ones."
    )

    return {
        "n_assets": int(n),
        "assets": [str(c) for c in frame.columns],
        "weights": {str(c): float(w) for c, w in zip(frame.columns, weights)},
        "risk_contributions": {
            str(c): float(rc) for c, rc in zip(frame.columns, contributions)
        },
        "risk_shares": {str(c): float(s) for c, s in zip(frame.columns, shares)},
        "target_shares": {str(c): float(t) for c, t in zip(frame.columns, targets)},
        "portfolio_volatility": _portfolio_volatility(weights, matrix),
        "converged": converged,
        "iterations": iterations,
        "max_share_error": error,
        "warnings": warnings,
    }


def _reference_quasi_diagonal_order(distance: np.ndarray) -> List[int]:
    n = distance.shape[0]
    members = {i: [i] for i in range(n)}
    active: List[int] = list(range(n))

    working = np.array(distance, dtype=float, copy=True)
    np.fill_diagonal(working, np.inf)

    while len(active) > 1:
        block = working[np.ix_(active, active)]
        first = int(np.argmin(block))
        a_i, b_i = divmod(first, len(active))
        if a_i > b_i:  # take the upper-triangle representative of the pair
            a_i, b_i = b_i, a_i
        a, b = active[a_i], active[b_i]

        members[a] = members[a] + members[b]
        merged = np.minimum(working[a, :], working[b, :])
        working[a, :] = merged
        working[:, a] = merged
        working[a, a] = np.inf
        active.remove(b)

    clusters = {active[0]: members[active[0]]}
    return list(clusters.values())[0]


def _reference_estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: str = "ledoit_wolf",
    halflife: Optional[float] = 60.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValidationError(
            f"unknown covariance method {method!r}; expected one of {list(METHODS)}"
        )
    kept_columns = returns.dropna(how="all", axis=1)
    frame = kept_columns.dropna()
    n_rows_dropped = int(len(kept_columns) - len(frame))
    shortest = (
        str(kept_columns.notna().sum().idxmin()) if kept_columns.shape[1] else None
    )
    if frame.shape[0] < 2:
        raise ValidationError(
            f"covariance needs at least 2 complete observations, got "
            f"{frame.shape[0]}. Every asset must have a return on every date "
            "used, and rows with any missing value were dropped."
        )
    if frame.shape[1] < 2:
        raise ValidationError("covariance needs at least 2 assets with usable history")

    n_obs, n_assets = frame.shape
    values = frame.to_numpy(dtype=float)
    shrinkage: Optional[float] = None
    effective = float(n_obs)

    if method == "sample":
        cov = np.cov(values, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(values)
        cov = estimator.covariance_
        shrinkage = float(estimator.shrinkage_)
    else:
        cov, effective = _ewma_covariance(
            values, 60.0 if halflife is None else halflife
        )
        if method == "ewma_shrunk":
            cov, shrinkage = _shrink_to_identity(cov, n_obs, n_assets)

    annual = cov * periods_per_year
    eigenvalues = np.linalg.eigvalsh(annual)
    smallest = float(eigenvalues.min())
    condition = float(eigenvalues.max() / smallest) if smallest > 0 else float("inf")

    return {
        "method": method,
        "matrix": {
            row: {col: float(annual[i, j]) for j, col in enumerate(frame.columns)}
            for i, row in enumerate(frame.columns)
        },
        "assets": list(frame.columns),
        "n_observations": int(n_obs),
        "n_assets": int(n_assets),
        "effective_observations": effective,
        "observations_per_parameter": float(
            effective * n_assets / (n_assets * (n_assets + 1) / 2)
        ),
        "shrinkage_intensity": shrinkage,
        "condition_number": condition,
        "smallest_eigenvalue": smallest,
        "annualized": True,
        "n_rows_dropped": n_rows_dropped,
        "warnings": _warnings(
            method,
            n_obs,
            n_assets,
            condition,
            smallest,
            shrinkage,
            n_rows_dropped=n_rows_dropped,
            n_rows_total=int(len(kept_columns)),
            shortest=shortest,
            effective=effective,
        ),
    }


# ── inputs ───────────────────────────────────────────────────────────────


def _factor_returns(n_assets: int, n_bars: int, seed: int) -> pd.DataFrame:
    """A factor model, so the correlations have structure to cluster on."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 1.0, (n_assets, 4))
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5
    factors = rng.normal(0.0, 0.008, (n_bars, 4))
    noise = rng.normal(0.0, 1.0, (n_bars, n_assets)) * rng.uniform(
        0.005, 0.025, n_assets
    )
    return pd.DataFrame(
        factors @ loadings.T + noise,
        index=pd.bdate_range("2019-01-01", periods=n_bars),
        columns=[f"A{i:03d}" for i in range(n_assets)],
    )


def _annual_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * 252


# ── risk_parity ──────────────────────────────────────────────────────────


def _assert_risk_parity_matches(cov, **kwargs):
    try:
        expected = _reference_risk_parity(cov, **kwargs)
    except Exception as error:  # noqa: BLE001 -- the type is what is compared
        with pytest.raises(type(error)):
            construction.risk_parity(cov, **kwargs)
        return None
    actual = construction.risk_parity(cov, **kwargs)
    assert actual["iterations"] == expected["iterations"]
    assert actual["converged"] == expected["converged"]
    assert actual["assets"] == expected["assets"]
    assert actual["n_assets"] == expected["n_assets"]
    assert actual["target_shares"] == expected["target_shares"]
    names = expected["assets"]
    # 12 significant digits, not bit-identical: the marginal-risk vector is
    # updated one column at a time within a sweep instead of recomputed, so
    # its last bits differ. It is rebuilt exactly at the start of every
    # sweep, which keeps the difference at rounding level (measured 1e-14).
    for key in ("weights", "risk_contributions", "risk_shares"):
        np.testing.assert_allclose(
            [actual[key][a] for a in names],
            [expected[key][a] for a in names],
            rtol=1e-12,
            atol=0.0,
        )
    np.testing.assert_allclose(
        actual["portfolio_volatility"], expected["portfolio_volatility"], rtol=1e-12
    )
    # A difference of shares that are equal to ~1e-11: compared absolutely.
    assert abs(actual["max_share_error"] - expected["max_share_error"]) < 1e-12
    if actual["converged"]:
        assert actual["warnings"] == expected["warnings"]
    else:
        assert len(actual["warnings"]) == len(expected["warnings"])
    return actual


class TestRiskParityMatchesTheFullProduct:
    @pytest.mark.parametrize("n_assets,seed", [(5, 0), (30, 1), (120, 2), (235, 3)])
    def test_factor_model_universe(self, n_assets, seed):
        cov = _annual_cov(_factor_returns(n_assets, 2100, seed))
        out = _assert_risk_parity_matches(cov)
        assert out["converged"]

    def test_custom_budget_and_negative_correlation(self):
        cov = pd.DataFrame(
            [[0.04, -0.018, 0.002], [-0.018, 0.09, -0.01], [0.002, -0.01, 0.0225]],
            index=list("ABC"),
            columns=list("ABC"),
        )
        _assert_risk_parity_matches(cov, budget=[0.5, 0.3, 0.2])
        _assert_risk_parity_matches(cov, budget=[3, 2, 1])

    def test_near_singular_covariance(self):
        returns = _factor_returns(8, 300, 4)
        returns["A007"] = returns["A006"] + 1e-7 * np.random.default_rng(5).normal(
            size=len(returns)
        )
        _assert_risk_parity_matches(_annual_cov(returns))

    def test_singular_covariance_is_repaired_the_same_way(self):
        # Fewer observations than assets: rank-deficient, PSD-repaired.
        _assert_risk_parity_matches(_annual_cov(_factor_returns(12, 8, 6)))

    def test_iteration_limit_reached(self):
        cov = _annual_cov(_factor_returns(40, 500, 7))
        out = _assert_risk_parity_matches(cov, max_iterations=2)
        assert not out["converged"] and out["iterations"] == 2

    def test_one_asset_is_refused_by_both(self):
        _assert_risk_parity_matches(pd.DataFrame([[0.04]], index=["A"], columns=["A"]))

    def test_planted_two_assets(self):
        # Two assets: equal risk at w_i proportional to 1 / sigma_i, for any
        # correlation.
        sigma = np.array([0.1, 0.3])
        rho = 0.4
        cov = pd.DataFrame(
            np.outer(sigma, sigma) * np.array([[1, rho], [rho, 1]]),
            index=["A", "B"],
            columns=["A", "B"],
        )
        out = _assert_risk_parity_matches(cov)
        assert out["weights"]["A"] == pytest.approx(0.75, rel=1e-12)
        assert out["weights"]["B"] == pytest.approx(0.25, rel=1e-12)

    def test_null_case_identity(self):
        names = list("ABCDEF")
        out = _assert_risk_parity_matches(
            pd.DataFrame(np.eye(6), index=names, columns=names)
        )
        assert out["iterations"] == 1
        assert all(
            w == pytest.approx(1 / 6, rel=1e-15) for w in out["weights"].values()
        )


# ── _quasi_diagonal_order ────────────────────────────────────────────────


def _distance(returns: pd.DataFrame) -> np.ndarray:
    correlation = returns.corr().to_numpy()
    return np.sqrt(np.clip(0.5 * (1.0 - correlation), 0.0, None))


class TestQuasiDiagonalOrderIsUnchanged:
    @pytest.mark.parametrize("n_assets,seed", [(3, 0), (17, 1), (64, 2), (235, 3)])
    def test_factor_model_universe(self, n_assets, seed):
        distance = _distance(_factor_returns(n_assets, 600, seed))
        expected = _reference_quasi_diagonal_order(distance)
        assert construction._quasi_diagonal_order(distance) == expected
        assert sorted(expected) == list(range(n_assets))

    @pytest.mark.parametrize("seed", range(6))
    def test_ties_break_the_same_way(self, seed):
        # Distances drawn from three values, so most merges are ties.
        rng = np.random.default_rng(seed)
        n = 25
        values = rng.choice([0.25, 0.5, 0.75], size=(n, n))
        distance = np.triu(values, 1) + np.triu(values, 1).T
        assert construction._quasi_diagonal_order(
            distance
        ) == _reference_quasi_diagonal_order(distance)
        everything_equal = np.full((n, n), 0.5)
        assert construction._quasi_diagonal_order(
            everything_equal
        ) == _reference_quasi_diagonal_order(everything_equal)

    def test_one_and_two_assets(self):
        assert construction._quasi_diagonal_order(np.zeros((1, 1))) == [0]
        pair = np.array([[0.0, 0.3], [0.3, 0.0]])
        assert construction._quasi_diagonal_order(pair) == [0, 1]

    def test_nan_inf_and_asymmetric_input(self):
        rng = np.random.default_rng(8)
        distance = rng.uniform(0, 1, (12, 12))  # not symmetric
        assert construction._quasi_diagonal_order(
            distance
        ) == _reference_quasi_diagonal_order(distance)
        with_nan = distance.copy()
        with_nan[3, 7] = np.nan
        assert construction._quasi_diagonal_order(
            with_nan
        ) == _reference_quasi_diagonal_order(with_nan)
        # Every pair at +inf: the degenerate path that merges a cluster
        # with itself must still be followed step for step.
        unreachable = np.full((5, 5), np.inf)
        assert construction._quasi_diagonal_order(
            unreachable
        ) == _reference_quasi_diagonal_order(unreachable)
        partly = rng.uniform(0, 1, (6, 6))
        partly[:, 4:] = np.inf
        partly[4:, :] = np.inf
        assert construction._quasi_diagonal_order(
            partly
        ) == _reference_quasi_diagonal_order(partly)

    def test_planted_two_blocks(self):
        # {0, 2, 4} close together, {1, 3} close together, far apart.
        d = np.full((5, 5), 0.9)
        for group in ([0, 2, 4], [1, 3]):
            for i in group:
                for j in group:
                    d[i, j] = 0.1
        np.fill_diagonal(d, 0.0)
        d[0, 2] = d[2, 0] = 0.05
        order = construction._quasi_diagonal_order(d)
        assert order == _reference_quasi_diagonal_order(d)
        assert order == [0, 2, 4, 1, 3]

    def test_hierarchical_risk_parity_is_identical(self, monkeypatch):
        returns = _factor_returns(60, 400, 9)
        actual = construction.hierarchical_risk_parity(returns)
        monkeypatch.setattr(
            construction, "_quasi_diagonal_order", _reference_quasi_diagonal_order
        )
        assert construction.hierarchical_risk_parity(returns) == actual


# ── estimate_covariance ──────────────────────────────────────────────────


class TestCovarianceMatrixIsUnchanged:
    @pytest.mark.parametrize("method", METHODS)
    def test_every_method(self, method):
        returns = _factor_returns(40, 300, 10)
        expected = _reference_estimate_covariance(returns, method=method)
        actual = covariance.estimate_covariance(returns, method=method)
        assert actual == expected
        assert list(actual["matrix"]) == list(expected["matrix"])
        for row in actual["matrix"]:
            assert list(actual["matrix"][row]) == list(expected["matrix"][row])

    def test_ragged_history_and_non_string_names(self):
        returns = _factor_returns(6, 200, 11)
        returns.iloc[:40, 2] = np.nan
        returns.columns = [10, 20, 30, 40, 50, 60]
        expected = _reference_estimate_covariance(returns, method="sample")
        actual = covariance.estimate_covariance(returns, method="sample")
        assert actual == expected
        assert all(type(k) is int for k in actual["matrix"])

    def test_planted_two_assets(self):
        a = np.array([0.01, -0.02, 0.03, 0.0, -0.01])
        returns = pd.DataFrame({"X": a, "Y": 2 * a})
        out = covariance.estimate_covariance(
            returns, method="sample", periods_per_year=1
        )
        assert out == _reference_estimate_covariance(
            returns, method="sample", periods_per_year=1
        )
        var = float(np.var(a, ddof=1))
        assert out["matrix"]["X"]["X"] == pytest.approx(var, rel=1e-15)
        assert out["matrix"]["X"]["Y"] == pytest.approx(2 * var, rel=1e-15)
        assert out["matrix"]["Y"]["Y"] == pytest.approx(4 * var, rel=1e-15)

    def test_refusals_are_unchanged(self):
        one = pd.DataFrame({"X": [0.01, 0.02, 0.03]})
        for frame in (one, pd.DataFrame({"X": [0.01], "Y": [0.02]})):
            with pytest.raises(ValidationError):
                _reference_estimate_covariance(frame)
            with pytest.raises(ValidationError):
                covariance.estimate_covariance(frame)
