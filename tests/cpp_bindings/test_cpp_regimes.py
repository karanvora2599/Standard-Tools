"""
`regime_em_step` is one numpy EM step of `detect_regimes`, bit for bit.

The fit runs 100 EM iterations on every series it was measured on, each
about 8k + 10 numpy calls: 23-60 ms on 2,000-5,000 daily returns with 2-4
regimes, 87-95% of the regime tool's call from the disk cache. The kernel
does each step but the exponentials, which stay one `np.exp` per regime
as the numpy loop calls them (numpy's float64 exp is its own routine on
some machines and the C library's on others). The loop stays in the module
as the fallback (`_em_python`) and is the reference here, together with
one step written out in numpy below. See the CHANGELOG entry of 2026-10-04.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import stationarity

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = hasattr(_cpp, "regime_em_step")
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="regime_em_step not built")


def _numpy_step(values, means, variances, weights):
    """One iteration of the loop in `_em_python`, as written there."""
    n, k = len(values), len(means)
    responsibility = np.zeros((n, k))
    for j in range(k):
        responsibility[:, j] = weights[j] * (
            np.exp(-0.5 * (values - means[j]) ** 2 / variances[j])
            / math.sqrt(2 * math.pi * variances[j])
        )
    totals = responsibility.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1e-300
    responsibility /= totals
    counts = responsibility.sum(axis=0)
    new_means = (responsibility * values[:, None]).sum(axis=0) / np.maximum(
        counts, 1e-12
    )
    new_vars = (responsibility * (values[:, None] - new_means) ** 2).sum(
        axis=0
    ) / np.maximum(counts, 1e-12)
    new_vars = np.maximum(new_vars, 1e-12)
    converged = np.allclose(new_means, means, atol=1e-10)
    next_exponents = np.stack(
        [-0.5 * (values - new_means[j]) ** 2 / new_vars[j] for j in range(k)]
    )
    return responsibility, counts, new_means, new_vars, next_exponents, converged


def _kernel_step(values, means, variances, weights):
    exponents = np.stack(
        [-0.5 * (values - means[j]) ** 2 / variances[j] for j in range(len(means))]
    )
    exponentials = np.empty_like(exponents)
    for j in range(len(means)):
        np.exp(exponents[j], out=exponentials[j])
    return _cpp.regime_em_step(values, exponentials, means, variances, weights)


def _bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes(), np.flatnonzero(a.ravel() != b.ravel())[:5]


def _returns(n, seed):
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 250) % 2 == 0, 0.008, 0.02)
    return pd.Series(
        rng.normal(0.0003, vol), index=pd.bdate_range("2006-01-02", periods=n)
    )


class TestOneStep:
    @pytest.mark.parametrize("k", [2, 3, 4, 5, 7])
    @pytest.mark.parametrize("seed", range(3))
    def test_the_step_is_numpys(self, k, seed):
        rng = np.random.default_rng(seed * 10 + k)
        values = rng.normal(0.0, 0.015, 3_001)
        means = np.sort(rng.normal(0.0, 0.01, k))
        variances = rng.uniform(1e-5, 4e-4, k)
        weights = rng.dirichlet(np.ones(k))
        got = _kernel_step(values, means, variances, weights)
        want = _numpy_step(values, means, variances, weights)
        _bits(got[0].T, want[0])
        for g, w in zip(got[1:5], want[1:5]):
            _bits(g, w)
        assert got[5] is want[5] is False

    def test_underflow_an_empty_regime_and_the_floors(self):
        """Observations far from every regime underflow to zero density
        (divided by 1e-300), a regime of weight 0 counts nothing, and
        coincident points floor the variance at 1e-12."""
        values = np.array([0.0, 0.0, 0.0, 1e3, -1e3, 0.0])
        means = np.array([0.0, 0.5])
        variances = np.array([1e-6, 1e-6])
        weights = np.array([1.0, 0.0])
        got = _kernel_step(values, means, variances, weights)
        want = _numpy_step(values, means, variances, weights)
        _bits(got[0].T, want[0])
        for g, w in zip(got[1:5], want[1:5]):
            _bits(g, w)
        assert got[3][0] == 1e-12

    def test_the_convergence_test_is_allclose(self):
        """The old means enter only the convergence test, so they are moved
        around the new ones, from equal to 1e-3 away, and the kernel's
        answer is held to np.allclose(new, old, atol=1e-10) at each."""
        values = _returns(500, 3).to_numpy()
        variances, weights = np.full(3, 1e-4), np.full(3, 1 / 3)
        means = np.array([-0.01, 0.0, 0.01])
        exponentials = np.exp(
            np.stack([-0.5 * (values - m) ** 2 / v for m, v in zip(means, variances)])
        )
        new = _cpp.regime_em_step(values, exponentials, means, variances, weights)[2]
        answers = set()
        for scale in (0.0, 1e-13, 5e-11, 1e-10, 2e-10, 1e-9, 1e-7, 1e-5, 1e-3):
            for direction in (1.0, -1.0):
                old = new + direction * scale * (1.0 + np.abs(new))
                got = _cpp.regime_em_step(
                    values, exponentials, old, variances, weights
                )[5]
                assert got is np.allclose(new, old, atol=1e-10), scale
                answers.add(got)
        assert answers == {True, False}


class TestTheWholeFit:
    @pytest.mark.parametrize("n", [20, 251, 2_000, 5_000])
    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_detect_regimes_is_the_same_on_both_paths(self, n, k):
        series = _returns(n, n + k)
        native = stationarity.detect_regimes(series, n_regimes=k)
        try:
            stationarity.HAS_CPP = False
            python = stationarity.detect_regimes(series, n_regimes=k)
        finally:
            stationarity.HAS_CPP = True
        assert native["labels"] == python["labels"]
        for got, want in zip(native["regimes"], python["regimes"]):
            assert got.keys() == want.keys()
            for key in want:
                assert np.float64(got[key]).tobytes() == np.float64(want[key]).tobytes()
        for key in ("persistence", "n_switches", "current_regime", "warnings"):
            assert native[key] == python[key]

    def test_a_fit_that_converges_stops_where_numpy_stops(self):
        """Two clusters far apart converge well inside 100 iterations; the
        kernel's convergence test must end the loop on the same step."""
        rng = np.random.default_rng(5)
        values = np.concatenate([rng.normal(-5, 0.1, 300), rng.normal(5, 0.4, 200)])
        q = np.quantile(values, np.linspace(0.1, 0.9, 2))
        start = (q.copy(), np.full(2, float(values.var(ddof=1))), np.full(2, 0.5))
        steps = {}
        for name, fit in (
            ("native", stationarity._em_native),
            ("numpy", stationarity._em_python),
        ):
            for limit in range(1, 100):
                *_, means, variances, weights = (None, *fit(values, *start, limit))
                if limit > 1 and np.array_equal(means, previous):
                    steps[name] = limit - 1
                    break
                previous = means
        assert steps["native"] == steps["numpy"] < 99
        native = stationarity._em_native(values, *start, 100)
        python = stationarity._em_python(values, *start, 100)
        for g, w in zip(native, python):
            _bits(g, w)

    def test_no_iterations(self):
        values = _returns(50, 1).to_numpy()
        start = (np.array([-0.01, 0.01]), np.full(2, 1e-4), np.full(2, 0.5))
        for g, w in zip(
            stationarity._em_native(values, *start, 0),
            stationarity._em_python(values, *start, 0),
        ):
            _bits(g, w)


class TestTheBinding:
    def test_shapes_are_checked(self):
        values, e = np.zeros(10), np.ones((2, 10))
        ok = (np.zeros(2), np.ones(2), np.full(2, 0.5))
        with pytest.raises(ValueError, match="columns"):
            _cpp.regime_em_step(values, np.ones((2, 9)), *ok)
        with pytest.raises(ValueError, match="one value"):
            _cpp.regime_em_step(values, e, np.zeros(3), np.ones(2), np.full(2, 0.5))
        for k in (1, 8):
            with pytest.raises(ValueError, match="2 to 7"):
                _cpp.regime_em_step(
                    values, np.ones((k, 10)), np.zeros(k), np.ones(k), np.ones(k)
                )
        with pytest.raises(ValueError, match="2-D"):
            _cpp.regime_em_step(values, np.ones(10), *ok)

    def test_concurrent_calls_keep_their_own_series(self):
        """The step runs with the GIL released."""
        series = [_returns(2_000, 60 + i).to_numpy() for i in range(8)]
        start = (np.array([-0.01, 0.0, 0.01]), np.full(3, 1e-4), np.full(3, 1 / 3))
        want = [stationarity._em_native(s, *start, 40) for s in series]
        got: list = [None] * 8

        def work(i):
            for _ in range(3):
                got[i] = stationarity._em_native(series[i], *start, 40)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        for g, w in zip(got, want):
            for a, b in zip(g, w):
                _bits(a, b)

    def test_the_docstring_names_the_contract(self):
        doc = _cpp.regime_em_step.__doc__
        assert "bit for bit" in doc and "ValueError" in doc
