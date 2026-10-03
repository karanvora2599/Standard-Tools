"""
`_repair_psd` asks a Cholesky factorization before an eigendecomposition
(see the CHANGELOG entry of 2026-10-02), and its outputs are the
eigenvalue-only implementation's, bit for bit.

The argument: a Cholesky factorization that completes is exact for a
perturbed matrix A + E with ||E|| <= n(n+1)u ||A||, and A + E = R'R is
positive semi-definite -- so up to `_CHOLESKY_PROOF_MAX_ASSETS` a completed
factorization proves the eigenvalue test would have passed, and the frame is
returned untouched as it always was. When it fails, the eigenvalue path runs
unchanged. Checked here on 1,500 matrices built to cover both sides of the
repair threshold -- the decision never disagrees, and the outputs (the
matrix, the warning, the identity of the returned frame) match the old path
exactly -- and on the public functions that call it.

Both implementations run on one BLAS thread here, so the comparison is of
the Cholesky change alone and not of the thread count's last bits.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.portfolio import construction
from standard_quant_tools.portfolio.construction import (
    _CHOLESKY_PROOF_MAX_ASSETS,
    _PSD_TOLERANCE,
    _repair_psd,
)

# ── the implementation before the change, verbatim ──────────────────────


def _reference_repair_psd(
    frame: pd.DataFrame, who: str
) -> "tuple[pd.DataFrame, List[str]]":
    matrix = frame.to_numpy()
    eigenvalues, vectors = np.linalg.eigh(matrix)
    largest = float(eigenvalues.max())
    smallest = float(eigenvalues.min())
    if smallest >= -_PSD_TOLERANCE * max(largest, 1e-300):
        return frame, []
    floor = _PSD_TOLERANCE * largest
    repaired = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    repaired = (repaired + repaired.T) / 2.0
    fixed = pd.DataFrame(repaired, index=frame.index, columns=frame.columns)
    return fixed, [
        f"{who}: the covariance matrix is not positive semi-definite (smallest "
        f"eigenvalue {smallest:.3e} against a largest of {largest:.3e}), which a "
        "pairwise estimate over a ragged panel produces. It was projected onto "
        "the nearest PSD matrix by flooring the eigenvalues before use; the "
        "weights below are for the repaired matrix. Estimate the covariance on "
        "complete rows (estimate_covariance) to avoid the repair."
    ]


def _reference_on_one_thread(frame, who):
    with single_threaded_blas():
        return _reference_repair_psd(frame, who)


# ── matrices on both sides of the threshold ──────────────────────────────

#: Smallest eigenvalue as a multiple of -_PSD_TOLERANCE * largest: inside
#: the tolerance (no repair), on it, and beyond it (repair), plus barely
#: positive ones a factorization may or may not complete on.
PLANTED = (0.01, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 100.0, 1e4, -0.01, -1.0, -100.0)


def _planted(rng, n, multiple):
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    values = rng.uniform(0.1, 1.0, n)
    values[0] = 1.0
    values[-1] = -multiple * _PSD_TOLERANCE
    matrix = (q * values) @ q.T
    return (matrix + matrix.T) / 2.0


def _matrix(rng, trial):
    n = int(rng.choice([3, 10, 40, 120, 235]))
    kind = trial % 5
    if kind == 0:  # a well-sampled covariance
        rows = int(rng.integers(n + 1, 8 * n + 10))
        return np.cov(rng.normal(0, 0.01, (rows, n)), rowvar=False)
    if kind == 1:  # rank-deficient: no more observations than assets
        rows = int(rng.integers(2, n + 1))
        return np.cov(rng.normal(0, 0.01, (rows, n)), rowvar=False)
    if kind == 2:  # pairwise over a ragged panel, usually indefinite
        return _pairwise_covariance(rng, max(3 * n, 60), n)
    return _planted(rng, n, float(rng.choice(PLANTED)))


def _pairwise_covariance(rng, rows, n):
    """Each pair's covariance over the rows where both are present -- what
    `DataFrame.cov` computes on a ragged panel, in matrix form."""
    values = rng.normal(0, 0.01, (rows, n))
    present = (rng.random(values.shape) >= 0.3).astype(float)
    filled = values * present
    pairs = present.T @ present
    sums = filled.T @ present  # [i, j]: the sum of i over rows with both
    matrix = (filled.T @ filled - sums * sums.T / pairs) / (pairs - 1.0)
    return (matrix + matrix.T) / 2.0


def _frame(matrix):
    names = [f"A{i:03d}" for i in range(matrix.shape[0])]
    return pd.DataFrame(matrix, index=names, columns=names)


def _assert_same(actual, expected, frame):
    (got, got_notes), (want, want_notes) = actual, expected
    assert got_notes == want_notes
    if want is frame:
        assert got is frame
    else:
        assert got is not frame
        assert list(got.index) == list(want.index)
        assert list(got.columns) == list(want.columns)
        assert got.to_numpy().tobytes() == want.to_numpy().tobytes()


class TestTheDecisionAndTheOutputs:
    def test_1500_matrices_across_the_threshold(self):
        rng = np.random.default_rng(11)
        counts = {"proved": 0, "eigh_ok": 0, "repaired": 0, "disagreed": 0}
        planted_repairs = planted_passes = 0
        with single_threaded_blas():  # the whole loop, for its speed
            for trial in range(1500):
                frame = _frame(_matrix(rng, trial))
                try:
                    np.linalg.cholesky(frame.to_numpy())
                    proved = True
                except np.linalg.LinAlgError:
                    proved = False
                expected = _reference_repair_psd(frame, "test")
                repaired = bool(expected[1])
                if proved and repaired:
                    counts["disagreed"] += 1
                counts[
                    "proved" if proved else "repaired" if repaired else "eigh_ok"
                ] += 1
                if trial % 5 >= 3:
                    planted_repairs += repaired
                    planted_passes += not repaired
                _assert_same(_repair_psd(frame, "test"), expected, frame)
        # A completed factorization never met a matrix the eigenvalues
        # would have repaired.
        assert counts["disagreed"] == 0
        # Every branch was exercised, the planted ones on both sides.
        assert min(counts["proved"], counts["eigh_ok"], counts["repaired"]) > 50
        assert planted_repairs > 50 and planted_passes > 50

    @pytest.mark.parametrize("multiple", PLANTED)
    @pytest.mark.parametrize("n", [2, 50, 235])
    def test_planted_at_the_threshold(self, n, multiple):
        rng = np.random.default_rng(int(n * 1000 + abs(multiple) * 7))
        frame = _frame(_planted(rng, n, multiple))
        _assert_same(
            _repair_psd(frame, "test"), _reference_on_one_thread(frame, "test"), frame
        )

    def test_the_public_functions_are_unchanged(self, monkeypatch):
        """risk parity, maximum diversification and marginal risk on a
        repaired and on an untouched matrix: identical results with the
        eigenvalue-only repair swapped back in."""
        rng = np.random.default_rng(5)
        values = rng.normal(0, 0.01, (300, 60))
        values[rng.random(values.shape) < 0.35] = np.nan
        ragged = pd.DataFrame(values, columns=[f"A{i:02d}" for i in range(60)])
        inputs = [ragged.cov(min_periods=30), ragged.dropna(how="all").fillna(0).cov()]
        calls = [
            lambda c: construction.risk_parity(c, max_iterations=300),
            construction.max_diversification,
            lambda c: construction.marginal_risk_contribution(
                {name: 1.0 / 60 for name in c.columns}, c
            ),
        ]
        actual = [call(c) for c in inputs for call in calls]
        monkeypatch.setattr(construction, "_repair_psd", _reference_on_one_thread)
        expected = [call(c) for c in inputs for call in calls]
        assert actual == expected
        assert any(r["warnings"] and "not positive" in r["warnings"][0] for r in actual)


class TestTheProofsReach:
    def test_the_size_limit_is_the_bound(self):
        """n(n+1)u within half the tolerance up to the limit, and not past it."""
        u = 2.0**-53
        limit = _CHOLESKY_PROOF_MAX_ASSETS
        assert limit * (limit + 1) * u <= _PSD_TOLERANCE / 2
        assert (limit + 1) * (limit + 2) * u > _PSD_TOLERANCE / 2

    @pytest.mark.parametrize("n, asked", [(670, True), (671, False)])
    def test_above_the_limit_the_eigenvalues_decide_alone(self, monkeypatch, n, asked):
        rng = np.random.default_rng(n)
        a = rng.normal(size=(n + 5, n))
        frame = _frame(a.T @ a / (n + 5))
        calls = []
        real = np.linalg.cholesky

        def spy(matrix):
            calls.append(matrix.shape)
            return real(matrix)

        monkeypatch.setattr(np.linalg, "cholesky", spy)
        result = _repair_psd(frame, "test")
        monkeypatch.undo()
        assert bool(calls) is asked
        _assert_same(result, _reference_on_one_thread(frame, "test"), frame)

    def test_a_known_repair(self):
        """[[1, 2], [2, 1]] has eigenvalues 3 and -1. The repair floors -1
        to 3e-10, giving 1.5 +/- 1.5e-10 in each cell, and says so."""
        frame = _frame(np.array([[1.0, 2.0], [2.0, 1.0]]))
        repaired, notes = _repair_psd(frame, "who")
        expected = np.array(
            [[1.5 + 1.5e-10, 1.5 - 1.5e-10], [1.5 - 1.5e-10, 1.5 + 1.5e-10]]
        )
        np.testing.assert_allclose(repaired.to_numpy(), expected, rtol=1e-12)
        assert notes and notes[0].startswith("who: ")
        assert (
            "smallest eigenvalue -1.000e+00 against a largest of 3.000e+00" in notes[0]
        )

    def test_the_null_cases(self):
        """The identity and a singular PSD matrix: untouched, no note. The
        singular one fails the factorization and is passed by the
        eigenvalues, as it always was."""
        identity = _frame(np.eye(4))
        assert _repair_psd(identity, "test") == (identity, [])
        ones = _frame(np.ones((3, 3)))
        result, notes = _repair_psd(ones, "test")
        assert result is ones and notes == []
