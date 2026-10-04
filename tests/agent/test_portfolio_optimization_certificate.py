"""
The portfolio optimization tool carries the optimizer's certificate, and
says what happened when a solve did not converge.

See the CHANGELOG entry of 2026-10-02. The convex mean-variance methods are
solved exactly and certified against their KKT conditions; the report names
the method and carries the residuals. On a non-converged run the tool used
to add "constraints may be infeasible", a guess. SLSQP's commonest stop,
status 8, is a line search that stalled; measured at 235 assets, it
stalled within 2e-9 of the optimum with sum-to-1 met to 2e-11.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import PortfolioOptimizationInput
from standard_quant_tools.agent.tools import run_portfolio_optimization
from standard_quant_tools.portfolio import optimize as opt

pytestmark = pytest.mark.skipif(not opt.HAS_SCIPY, reason="needs scipy")

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _returns(tickers, n=600, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    market = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame(
        {
            t: 0.8 * market + rng.normal(0.0002 * (i + 1), 0.008 + 0.002 * i, n)
            for i, t in enumerate(tickers)
        },
        index=idx,
    )


def _run(**kwargs):
    inp = PortfolioOptimizationInput(
        tickers=TICKERS, start_date="2023-01-02", end_date="2025-06-30", **kwargs
    )

    def fake(req_tickers, start, end, interval="1d"):
        return _returns(req_tickers)

    with patch(
        "standard_quant_tools.agent.runtimes.portfolio.tools.fetch_returns_sync", fake
    ):
        return run_portfolio_optimization(inp)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="min_volatility"),
        dict(method="min_volatility", max_weight=0.3),
        dict(method="max_sharpe"),
        dict(
            method="target_return", target_return=0.12, allow_short=True, max_weight=0.5
        ),
    ],
)
def test_the_tool_reports_the_exact_method_and_its_certificate(kwargs):
    result = _run(**kwargs)
    assert result.converged is True
    solver = result.solver
    assert solver.method == "active_set"
    assert solver.status == 0
    assert solver.certified is True
    assert solver.fallback is None
    certificate = solver.certificate
    assert certificate["tolerance"] == 1e-12
    for key in (
        "stationarity",
        "dual_infeasibility",
        "equality_residual",
        "bound_violation",
    ):
        assert 0.0 <= certificate[key] <= certificate["tolerance"], key
    # The payload stays valid JSON with the new fields in it.
    payload = json.loads(result.model_dump_json())
    assert payload["solver"]["certificate"]["tolerance"] == 1e-12


def test_methods_outside_the_exact_solve_report_no_certification():
    result = _run(method="target_volatility", target_volatility=0.2)
    assert result.solver.method == "SLSQP"
    assert result.solver.certified is None
    assert result.solver.certificate is None


def _stall(monkeypatch):
    real = opt._solve_constrained

    def stalled(*args, **kwargs):
        w, _, report = real(*args, **kwargs)
        report = dict(report)
        report["status"] = 8
        report["message"] = "Positive directional derivative for linesearch"
        return w, False, report

    monkeypatch.setattr(opt, "_solve_constrained", stalled)


def test_a_stalled_solver_is_described_not_called_infeasible(monkeypatch):
    _stall(monkeypatch)
    result = _run(method="target_volatility", target_volatility=0.2)
    assert result.converged is False
    text = " ".join(result.warnings)
    assert "constraints may be infeasible" not in text
    assert "optimizer did not converge: SLSQP ended with status 8" in text
    assert "sum-to-1 residual" in text
    assert "target_volatility residual" in text


def test_a_stalled_capped_max_sharpe_comes_back_certified(monkeypatch):
    """The same stop on capped max_sharpe is solved exactly on SLSQP's
    active set, so the tool reports a certified optimum and no warning."""
    _stall(monkeypatch)
    result = _run(method="max_sharpe", max_weight=0.4)
    assert result.converged is True
    assert result.solver.method == "active_set"
    assert result.solver.certified is True
    assert not any("did not converge" in w for w in result.warnings)


def _wide_returns(tickers, T=2000, seed=12):
    """Three-factor daily returns for a wide universe, where the SVD behind
    a condition number is large enough to split across BLAS threads."""
    rng = np.random.default_rng(seed)
    n = len(tickers)
    loadings = rng.normal(1.0, 0.4, (n, 3)) * np.array([1.0, 0.5, 0.3])
    factors = rng.normal(0.0, 0.008, (T, 3))
    drift = rng.normal(0.0004, 0.0015, n)
    idio = rng.uniform(0.006, 0.03, n)
    data = factors @ loadings.T + drift + rng.normal(0.0, 1.0, (T, n)) * idio
    index = pd.bdate_range("2017-01-02", periods=T)
    return pd.DataFrame(data, columns=list(tickers), index=index)


def test_risk_parity_reports_one_condition_number_at_any_blas_thread_count():
    """risk_parity does not go through mean_variance_optimize and took its
    condition number from a bare np.linalg.cond, which on this 235-asset
    covariance differs in the last bits between one and four BLAS threads.
    It now comes from the same one-thread computation.

    The covariance before it runs on one thread too. Its product kept the
    caller's threads until the CHANGELOG entry of 2026-10-04, and on the CI
    runners' OpenBLAS it gave different last bits at one and four threads,
    so this test failed there. The whole result is now the same at caller
    limits of one, two and four, and its condition number is the one-thread
    condition number of the one-thread covariance, computed here from
    pandas and numpy directly. Each call's number is also the one-thread
    condition number of the covariance that call's caller setting
    estimates."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    tickers = [f"T{i:03d}" for i in range(235)]
    inp = PortfolioOptimizationInput(
        tickers=tickers,
        start_date="2017-01-02",
        end_date="2024-12-31",
        method="risk_parity",
    )

    def fake(req_tickers, start, end, interval="1d"):
        return _wide_returns(req_tickers)

    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        one_thread_cov = _wide_returns(tickers).cov().to_numpy(dtype=float) * 252
        one_thread = float(np.linalg.cond(one_thread_cov))
    results, per_matrix = [], []
    for threads in (1, 2, 4):
        with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
            with patch(
                "standard_quant_tools.agent.runtimes.portfolio.tools."
                "fetch_returns_sync",
                fake,
            ):
                results.append(run_portfolio_optimization(inp))
            _, cov = opt.annualized_mean_cov(_wide_returns(tickers), 252)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            per_matrix.append(float(np.linalg.cond(cov)))
    reported = [result.condition_number for result in results]
    assert reported == per_matrix
    assert reported == [one_thread] * 3
    # Every field, floats by their shortest round-trip repr: equal text is
    # equal bits.
    dumped = [result.model_dump_json() for result in results]
    assert dumped[1] == dumped[0]
    assert dumped[2] == dumped[0]
