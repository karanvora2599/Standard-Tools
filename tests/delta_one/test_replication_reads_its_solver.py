"""
The replication basket is solved exactly and says whether the answer
certified.

`optimize_replication_basket` ran SLSQP, and from 2026-10-22 read its exit
status. Each solve is now `optimize_portfolio`'s min_tracking_error on the
candidates and the benchmark joined, the benchmark held at weight zero: the
active-set method ends with one linear solve on its final active set, and
`converged` is the KKT certificate computed from the weights. A certified
basket is never worse than SLSQP's on the objective, and SLSQP's own
answer sat up to about 1e-6 away from it in the weights.

THE PLANTED FAILURE. The optimizer's certificate is wrapped to report a
stationarity residual over tolerance; the weights are the real ones, so the
constraint checks still pass and only the certificate says the answer is
not a verified minimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.delta_one import replication
from standard_quant_tools.delta_one.replication import optimize_replication_basket
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import general_optimizer
from standard_quant_tools.portfolio.covariance import estimate_covariance

pytestmark = pytest.mark.skipif(not replication.HAS_SCIPY, reason="needs scipy")


def _universe(n=8, n_obs=300, seed=4):
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.0004, 0.010, n_obs)
    frame = pd.DataFrame(
        {
            f"S{i}": rng.uniform(0.6, 1.4) * factor + rng.normal(0, 0.01, n_obs)
            for i in range(n)
        }
    )
    bench = pd.Series(
        frame.to_numpy() @ np.full(n, 1.0 / n) + rng.normal(0, 1e-3, n_obs)
    )
    return frame, bench


def _fail_the_certificate(monkeypatch):
    real = general_optimizer._certify

    def failing(*args, **kwargs):
        certificate, duals = real(*args, **kwargs)
        return (
            general_optimizer.Certificate(
                certificate.stationarity + 1.0,
                certificate.dual_feasibility,
                certificate.primal_feasibility,
                certificate.complementary_slackness,
            ),
            duals,
        )

    monkeypatch.setattr(general_optimizer, "_certify", failing)


class TestTheCertificateIsRead:
    def test_a_certified_solve_is_converged(self):
        frame, bench = _universe()
        out = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        assert out["converged"] is True
        assert not any("did not certify" in w for w in out["warnings"])

    def test_an_answer_that_does_not_certify_is_reported(self, monkeypatch):
        frame, bench = _universe()
        _fail_the_certificate(monkeypatch)
        out = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        assert out["converged"] is False
        (warning,) = [w for w in out["warnings"] if "did not certify" in w]
        assert "stationarity" in warning
        # Feasible all the same: the certificate is the only signal.
        assert out["net_weight"] == pytest.approx(1.0, abs=1e-12)

    def test_the_tool_carries_the_verdict(self, monkeypatch):
        frame, bench = _universe()
        _fail_the_certificate(monkeypatch)
        out = resolve("delta_one").dispatch(
            "optimize_replication_basket",
            {
                "returns": {c: frame[c].tolist() for c in frame.columns},
                "benchmark_returns": bench.tolist(),
            },
        )
        assert out["converged"] is False
        assert any("did not certify" in w for w in out["warnings"])


class TestTheBasketIsTheExactMinimum:
    @staticmethod
    def _objective_terms(frame, bench):
        joined = frame.copy()
        joined["__b__"] = bench
        matrix = estimate_covariance(joined, method="ledoit_wolf")["matrix"]
        names = list(frame.columns)
        S = np.array([[matrix[a][b] for b in names] for a in names])
        c = np.array([matrix[a]["__b__"] for a in names])
        return names, S, c, float(matrix["__b__"]["__b__"])

    @pytest.mark.parametrize("seed,cap", [(4, None), (7, 0.2), (9, 0.15)])
    def test_slsqp_never_does_better_and_the_kkt_conditions_hold(self, seed, cap):
        """SLSQP with the settings this basket used to have, on the same
        covariance, never reaches a lower tracking variance than the exact
        basket, and lands within 1e-5 of its weights. The basket's own KKT
        conditions are checked here: on the held names the gradient of
        w'Sw - 2w'c is one common value (the budget's multiplier), except
        where a name sits at its cap or at zero."""
        from scipy.optimize import minimize

        frame, bench = _universe(n=10, seed=seed)
        out = optimize_replication_basket(
            returns=frame, benchmark_returns=bench, max_weight=cap
        )
        assert out["converged"] is True
        names, S, c, vb = self._objective_terms(frame, bench)
        w = np.array([out["weights"].get(a, 0.0) for a in names])
        upper = 1.0 if cap is None else cap
        res = minimize(
            lambda x: float(x @ S @ x - 2 * x @ c),
            np.full(len(names), 1 / len(names)),
            jac=lambda x: 2 * (S @ x - c),
            method="SLSQP",
            bounds=[(0.0, upper)] * len(names),
            constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        exact = float(w @ S @ w - 2 * w @ c)
        assert exact <= res.fun + 1e-15
        assert np.max(np.abs(res.x - w)) < 1e-5
        gradient = 2 * (S @ w - c)
        # relative to the terms that cancel, not to their small difference
        scale = float(np.max(2 * (np.abs(S) @ np.abs(w) + np.abs(c))))
        free = (w > 1e-12) & (w < upper - 1e-12)
        assert free.sum() >= 1
        level = gradient[free].mean()
        assert np.max(np.abs(gradient[free] - level)) < 1e-13 * scale
        assert np.all(gradient[w <= 1e-12] >= level - 1e-13 * scale)
        assert np.all(gradient[w >= upper - 1e-12] <= level + 1e-13 * scale)
        assert out["predicted_tracking_error"] == pytest.approx(
            np.sqrt(exact + vb), rel=1e-12
        )


class TestASingularCovarianceIsRefused:
    def test_fewer_observations_than_candidates_under_the_sample_estimate(self):
        """Twenty observations of thirty candidates give a sample covariance
        of rank at most nineteen: some combination of candidates has no
        variance, and the minimum of tracking error is not unique. SLSQP
        returned one of the minima with a warning that the weights were
        unusable; the exact solve refuses, naming the fix. Shrunk, the same
        data still builds a basket, with that warning."""
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            rng.normal(0, 0.01, (20, 30)), columns=[f"S{i}" for i in range(30)]
        )
        bench = pd.Series(frame.to_numpy() @ np.full(30, 1 / 30))
        with pytest.raises(ValidationError) as exc:
            optimize_replication_basket(
                returns=frame, benchmark_returns=bench, covariance_method="sample"
            )
        assert "not positive definite" in str(exc.value)
        assert "ledoit_wolf" in str(exc.value)
        shrunk = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        assert shrunk["converged"] is True
        assert any("unusable" in w for w in shrunk["warnings"])
