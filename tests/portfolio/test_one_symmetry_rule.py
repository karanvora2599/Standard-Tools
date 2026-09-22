"""
One covariance rule, enforced in one place.

`portfolio/construction.py` carried its own squareness, NaN and symmetry
checks, worded differently from the numeric contract's and applied at a
looser tolerance. Two doors onto one rule, agreeing by inspection rather
than by construction -- and they did not quite agree: the local copy
accepted an asymmetry ten times larger than the contract's, and its NaN
check let an infinity straight through to the eigenvalue code.

The construction module now calls the contract. Two behaviours change and
both are pinned here: the symmetry tolerance tightens from rtol 1e-8 to
1e-9, and an infinite entry is refused as non-finite. See the CHANGELOG
entry of 2026-09-22.
"""

import numpy as np
import pytest

from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    RiskParityInput,
    optimize_risk_parity,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import (
    max_diversification,
    risk_parity,
)

ASSETS = ["AAPL", "MSFT", "GOOGL"]

#: Unit variances so the tolerances read directly: at rtol 1e-9 against an
#: off-diagonal of 0.5 the symmetry threshold is 5e-10, and at the old
#: rtol 1e-8 it was 5e-9.
BASE = [
    [1.0, 0.5, 0.3],
    [0.5, 1.0, 0.4],
    [0.3, 0.4, 1.0],
]

#: Between the two thresholds: the old check accepted it, the contract does
#: not.
ASYMMETRY_BETWEEN_THE_TOLERANCES = 3e-9


def _matrix(**kwargs):
    return [row[:] for row in BASE]


def _asymmetric(delta: float):
    matrix = _matrix()
    matrix[0][1] = matrix[0][1] + delta
    return matrix


class TestTheContractIsTheOneThatAnswers:
    def test_an_asymmetry_between_the_two_tolerances_is_refused(self):
        with pytest.raises(ValidationError, match=r"largest \|A - A'\|"):
            optimize_risk_parity(
                RiskParityInput(
                    assets=ASSETS,
                    covariance=_asymmetric(ASYMMETRY_BETWEEN_THE_TOLERANCES),
                )
            )

    def test_the_refusal_names_the_caller_and_the_size_of_the_asymmetry(self):
        """The old local message said only that the matrix was not
        symmetric. The contract's says by how much, which is the number
        that decides whether this is rounding or a construction bug."""
        with pytest.raises(ValidationError) as excinfo:
            optimize_risk_parity(
                RiskParityInput(
                    assets=ASSETS,
                    covariance=_asymmetric(ASYMMETRY_BETWEEN_THE_TOLERANCES),
                )
            )
        message = str(excinfo.value)
        assert message.startswith("risk_parity: ")
        assert "not symmetric" in message
        assert "3.000e-09" in message

    def test_the_old_local_tolerance_would_have_let_it_through(self):
        """The delta is inside rtol 1e-8 and outside rtol 1e-9 -- which is
        the entire behaviour change, stated as arithmetic rather than as a
        claim about the code."""
        matrix = np.array(_asymmetric(ASYMMETRY_BETWEEN_THE_TOLERANCES))
        assert np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-12)
        assert not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12)

    def test_a_matrix_symmetric_to_machine_precision_still_passes(self):
        """The tolerance exists because a real covariance estimate is
        symmetric only to rounding. Tightening it must not start refusing
        those."""
        result = optimize_risk_parity(
            RiskParityInput(assets=ASSETS, covariance=_asymmetric(1e-16))
        )
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_an_exactly_symmetric_matrix_passes(self):
        result = optimize_risk_parity(
            RiskParityInput(assets=ASSETS, covariance=_matrix())
        )
        assert set(result.weights) == set(ASSETS)


class TestTheGapTheLocalCopyLeft:
    def test_an_infinite_entry_is_refused_as_non_finite(self):
        """The local check tested `isna()`, which is False for inf, so an
        infinity reached the eigenvalue decomposition."""
        matrix = _matrix()
        matrix[2][2] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            risk_parity(_frame(matrix))

    def test_a_nan_entry_is_still_refused(self):
        matrix = _matrix()
        matrix[1][2] = float("nan")
        matrix[2][1] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            risk_parity(_frame(matrix))


class TestTheChecksTheContractDoesNotOwn:
    def test_a_zero_variance_asset_is_still_refused(self):
        """The contract has no rule about the diagonal -- a zero-variance
        asset is a portfolio-construction problem, so that check stays
        local."""
        matrix = _matrix()
        matrix[2] = [0.0, 0.0, 0.0]
        for row in matrix:
            row[2] = 0.0
        with pytest.raises(ValidationError, match="non-positive"):
            risk_parity(_frame(matrix))

    def test_the_psd_repair_still_runs(self):
        """A symmetric matrix with a negative eigenvalue is repaired and
        says so, rather than being refused."""
        matrix = [
            [1.0, 0.99, 0.99],
            [0.99, 1.0, 0.99],
            [0.99, 0.99, 0.10],
        ]
        result = max_diversification(_frame(matrix))
        assert any("positive semi-definite" in w for w in result["warnings"])

    def test_every_construction_door_refuses_the_same_asymmetry(self):
        """The point of calling the contract: one rule, not one rule per
        entry point."""
        matrix = _frame(_asymmetric(ASYMMETRY_BETWEEN_THE_TOLERANCES))
        for solver in (risk_parity, max_diversification):
            with pytest.raises(ValidationError, match=r"largest \|A - A'\|"):
                solver(matrix)


def _frame(matrix):
    import pandas as pd

    return pd.DataFrame(matrix, index=ASSETS, columns=ASSETS)
