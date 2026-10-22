"""
The replication basket reads its solver's exit status.

`optimize_replication_basket` ran SLSQP and kept `result.x` without looking
at `result.success`, so a run that hit the iteration limit or stalled in its
line search returned its last iterate as the basket, with nothing in the
result to say so. It now reports `converged` and, when the solver did not
confirm a minimum, a warning carrying the solver's status and message. See
the CHANGELOG entry of 2026-10-22.

THE PLANTED STOP. The minimizer is wrapped so the real answer comes back
marked the way SciPy marks an iteration-limit stop (status 9); the weights
are the real ones, so the constraint checks still pass and only the
solver's own verdict says the run did not finish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.delta_one import replication
from standard_quant_tools.delta_one.replication import optimize_replication_basket

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


def _stop_at_the_iteration_limit(monkeypatch):
    real = replication._scipy_minimize

    def stopped(*args, **kwargs):
        result = real(*args, **kwargs)
        result.success = False
        result.status = 9
        result.message = "Iteration limit reached"
        return result

    monkeypatch.setattr(replication, "_scipy_minimize", stopped)


class TestTheSolverVerdictIsRead:
    def test_a_finished_solve_is_converged(self):
        frame, bench = _universe()
        out = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        assert out["converged"] is True
        assert not any("did not confirm" in w for w in out["warnings"])

    def test_an_iteration_limit_stop_is_reported(self, monkeypatch):
        frame, bench = _universe()
        _stop_at_the_iteration_limit(monkeypatch)
        out = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        assert out["converged"] is False
        (warning,) = [w for w in out["warnings"] if "did not confirm" in w]
        assert "status 9" in warning
        assert "Iteration limit reached" in warning
        # Feasible all the same: the solver's verdict is the only signal.
        assert out["net_weight"] == pytest.approx(1.0, abs=1e-6)

    def test_the_tool_carries_the_verdict(self, monkeypatch):
        frame, bench = _universe()
        _stop_at_the_iteration_limit(monkeypatch)
        out = resolve("delta_one").dispatch(
            "optimize_replication_basket",
            {
                "returns": {c: frame[c].tolist() for c in frame.columns},
                "benchmark_returns": bench.tolist(),
            },
        )
        assert out["converged"] is False
        assert any("status 9" in w for w in out["warnings"])
