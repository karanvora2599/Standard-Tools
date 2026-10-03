"""
`f_sf_array` is `f_sf` over arrays, to the bit (see the CHANGELOG entry of
2026-10-02).

The continued fraction runs as array passes, each element stopping at the
iteration its scalar loop would break on, with the lgamma terms computed
once per distinct pair of degrees of freedom and `math.log` / `math.exp`
per element. Inputs `f_sf` treats specially -- non-positive, NaN, infinite,
or degrees of freedom outside the array form's bounds -- are `f_sf` itself.

Every comparison here is on the raw bits of the double, against both the
current scalar and the scalar as it was before it shared its front factor
with the array form, which is kept below verbatim.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from standard_quant_tools import _special
from standard_quant_tools._special import f_sf, f_sf_array

# ── the scalar as it was, verbatim but for the names ────────────────────


def _reference_betacf(a: float, b: float, x: float, iterations: int = 300) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _reference_betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _reference_betacf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + b * math.log(1.0 - x)
            + a * math.log(x)
        )
        * _reference_betacf(b, a, 1.0 - x)
        / b
    )


def _reference_f_sf(statistic: float, d1: float, d2: float) -> float:
    if statistic <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    x = d2 / (d2 + d1 * statistic)
    return float(max(0.0, min(1.0, _reference_betainc(d2 / 2.0, d1 / 2.0, x))))


# ── helpers ─────────────────────────────────────────────────────────────


def _bits(values) -> list:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.uint64).tolist()


def _scalar_outcome(fn, statistic, d1, d2):
    try:
        return fn(statistic, d1, d2)
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        return (type(error).__name__, str(error))


def _assert_array_is_the_scalar(statistic, d1, d2):
    """Element for element, against the scalar now and the scalar before.
    Where the scalar raises on an element, the array form must raise the
    same error -- the first element's, as a loop over the scalar would."""
    statistic, d1, d2 = (
        np.asarray(v, dtype=np.float64)
        for v in np.broadcast_arrays(
            np.asarray(statistic, dtype=np.float64),
            np.asarray(d1, dtype=np.float64),
            np.asarray(d2, dtype=np.float64),
        )
    )
    triples = list(
        zip(statistic.ravel().tolist(), d1.ravel().tolist(), d2.ravel().tolist())
    )
    now = [_scalar_outcome(f_sf, *t) for t in triples]
    before = [_scalar_outcome(_reference_f_sf, *t) for t in triples]
    assert [o if isinstance(o, tuple) else _bits([o]) for o in now] == [
        o if isinstance(o, tuple) else _bits([o]) for o in before
    ]
    errors = [o for o in now if isinstance(o, tuple)]
    if errors:
        with pytest.raises((ValueError, OverflowError, ZeroDivisionError)) as raised:
            f_sf_array(statistic, d1, d2)
        assert (type(raised.value).__name__, str(raised.value)) == errors[0]
        return None
    got = f_sf_array(statistic, d1, d2)
    assert got.shape == statistic.shape
    assert got.dtype == np.float64
    assert _bits(got.ravel()) == _bits(now)
    return got


def _lead_lag_shaped(rng, n):
    """Squared t-statistics with d1 = 1 and the degrees of freedom of a
    lagged correlation: the inputs `lead_lag_matrix` sends."""
    rho = rng.uniform(-0.6, 0.6, n)
    dof = rng.choice(np.array([28, 47, 48, 98, 247, 498, 2026]), n)
    t = rho * np.sqrt(dof / np.maximum(1 - rho * rho, 1e-12))
    return t * t, np.ones(n), dof.astype(float)


# ── the tests ───────────────────────────────────────────────────────────


class TestTheArrayFormIsTheScalarToTheBit:
    """Every test twice: as shipped, where small batches and the last few
    iterating elements are handed to the scalar, and with both hand-offs
    off, so every element of even a four-element batch goes through the
    array arithmetic."""

    @pytest.fixture(autouse=True, params=["as_shipped", "array_path_forced"])
    def path(self, request, monkeypatch):
        if request.param == "array_path_forced":
            monkeypatch.setattr(_special, "_F_SF_ARRAY_MIN_BATCH", 0)
            monkeypatch.setattr(_special, "_BETACF_ARRAY_TAIL", 0)
        return request.param

    @pytest.mark.parametrize("seed", range(4))
    def test_lead_lag_shaped_inputs(self, seed):
        statistic, d1, d2 = _lead_lag_shaped(np.random.default_rng(seed), 20_000)
        _assert_array_is_the_scalar(statistic, d1, d2)

    @pytest.mark.parametrize("seed", range(4))
    def test_random_degrees_of_freedom_and_statistics(self, seed):
        rng = np.random.default_rng(100 + seed)
        n = 20_000
        statistic = 10.0 ** rng.uniform(-6, 4, n)
        d1 = np.where(
            rng.random(n) < 0.5, rng.integers(1, 60, n), rng.uniform(0.05, 80, n)
        )
        d2 = np.where(
            rng.random(n) < 0.5, rng.integers(1, 5000, n), 10.0 ** rng.uniform(-1, 6, n)
        )
        _assert_array_is_the_scalar(statistic, d1, d2)

    def test_both_sides_of_the_symmetry_point(self):
        """The continued fraction is taken on whichever side of
        (a + 1) / (a + b + 2) converges; a grid across it, for small and
        large degrees of freedom, puts elements on both sides at once."""
        d1, d2 = np.meshgrid([1.0, 2.0, 5.0, 30.0], [1.0, 3.0, 10.0, 200.0, 1e5])
        statistic = np.geomspace(1e-4, 1e4, 97)[:, None, None]
        _assert_array_is_the_scalar(statistic, d1[None], d2[None])

    def test_elements_that_run_out_of_iterations(self):
        """Large, nearly equal parameters near the mean converge slowly; the
        scalar stops at 300 iterations without converging and so must the
        array form, keeping whatever h it had. Enough of them that, as
        shipped, the array loop itself reaches the cap."""
        d = np.geomspace(2e6, 1e9, 60)
        statistic = np.array([0.999, 1.0, 1.001, 1.0001])
        _assert_array_is_the_scalar(statistic[:, None], d[None, :], d[None, :])

    @pytest.mark.parametrize(
        "value",
        [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan, 5e-324, 1e-300, 1e300],
    )
    def test_edge_statistics(self, value):
        """NaN is 1.0, as Python's min and max make it; zero and negative are
        1.0; infinity is 0.0."""
        d1 = np.array([1.0, 3.0, 1.0, 10.0])
        d2 = np.array([1.0, 50.0, 1e6, 0.5])
        got = _assert_array_is_the_scalar(np.full(4, value), d1, d2)
        if math.isnan(value):
            assert got.tolist() == [1.0] * 4

    @pytest.mark.parametrize(
        "value",
        [
            0.0,
            -0.0,
            -3.0,
            math.inf,
            math.nan,
            2.0**-40,
            2.0**-41,
            2.0**40,
            2.0**40 + 2.0**-12 * 2.0**40,
            1e-300,
            1e15,
            1e100,
            1e300,
        ],
    )
    def test_edge_degrees_of_freedom(self, value):
        """Zero, negative, NaN and infinite degrees of freedom, and huge and
        tiny ones on both sides of the array form's bounds, in either slot."""
        statistic = np.array([1e-3, 0.5, 1.0, 4.0, 1e3, 1e300])
        other = np.array([1.0, 2.0, 7.0, 40.0, 1e3, 1e6])
        _assert_array_is_the_scalar(statistic, value, other)
        _assert_array_is_the_scalar(statistic, other, value)
        _assert_array_is_the_scalar(statistic, value, value)

    def test_huge_statistic_with_huge_degrees_of_freedom(self):
        """x = d2 / (d2 + d1 f) lands mid-interval only when f is as large
        as d2 / d1, so huge degrees of freedom need huge statistics."""
        d = np.array([1e8, 1e12, 1e15, 1e100, 1e300])
        _assert_array_is_the_scalar(
            d[:, None] * np.array([0.5, 1.0, 2.0]), 1.0, d[:, None]
        )
        _assert_array_is_the_scalar(
            d[:, None] * np.array([0.5, 1.0, 2.0]), d[:, None], d[:, None]
        )

    def test_shapes(self):
        assert f_sf_array([], 1, 50).shape == (0,)
        assert f_sf_array(np.zeros((0, 3)), 1, 50).shape == (0, 3)
        scalar = f_sf_array(2.0, 3, 40)
        assert scalar.shape == () and scalar.dtype == np.float64
        assert _bits([float(scalar)]) == _bits([f_sf(2.0, 3, 40)])
        grid = f_sf_array(np.ones((2, 3)), [[1.0], [2.0]], [10.0, 20.0, 30.0])
        assert grid.shape == (2, 3)
        _assert_array_is_the_scalar(np.ones((2, 3)), [[1.0], [2.0]], [10.0, 20.0, 30.0])

    def test_integer_arguments_are_taken_as_doubles(self):
        """`lead_lag_matrix` sends integer degrees of freedom; the scalar
        received them as Python ints and converted at each operation."""
        statistic = np.array([0.3, 2.0, 9.0])
        d2 = np.array([28, 498, 2026])
        got = f_sf_array(statistic, 1, d2)
        expected = [f_sf(s, 1, int(d)) for s, d in zip(statistic.tolist(), d2.tolist())]
        assert _bits(got) == _bits(expected)

    def test_an_input_the_scalar_raises_on_raises_the_same(self):
        """d2 = 5e-324 halves to exactly zero, and with a statistic as small
        x = d2 / (d2 + f) is 0.5, so the scalar reaches `math.lgamma(0.0)`
        and raises. Both forms raise the same error."""
        statistic = np.array([1.0, 5e-324, 3.0])
        d2 = np.array([10.0, 5e-324, 20.0])
        scalar = [_scalar_outcome(f_sf, s, 1.0, d) for s, d in zip(statistic, d2)]
        assert scalar[1][0] == "ValueError"
        with pytest.raises(ValueError, match=scalar[1][1]):
            f_sf_array(statistic, 1.0, d2)

    def test_no_numpy_warnings(self):
        """The scalar never warns; the array form's overflows, NaNs and
        infinities must not either."""
        rng = np.random.default_rng(9)
        statistic = np.concatenate(
            [10.0 ** rng.uniform(-300, 300, 500), [math.nan, math.inf]]
        )
        d = np.concatenate([10.0 ** rng.uniform(-12, 12, 500), [1.0, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = f_sf_array(statistic, d[::-1], d)
        assert _bits(got) == _bits(
            [f_sf(*t) for t in zip(statistic.tolist(), d[::-1].tolist(), d.tolist())]
        )

    def test_known_values(self):
        """Planted answers: an F(1, d) tail is a two-sided t tail, and
        F(2, d2) has the closed form (1 + 2f/d2)^(-d2/2)."""
        assert f_sf_array(0.0, 1, 30).item() == 1.0
        # t = 2.042 is about the two-sided 5% point at 30 degrees of freedom.
        assert f_sf_array(2.0423**2, 1, 30).item() == pytest.approx(0.05, abs=2e-4)
        f = np.array([0.1, 1.0, 3.0, 10.0])
        for d2 in (4.0, 17.0, 120.0):
            closed = (1.0 + 2.0 * f / d2) ** (-d2 / 2.0)
            assert np.allclose(f_sf_array(f, 2.0, d2), closed, rtol=1e-12, atol=1e-15)

    def test_the_lgamma_terms_are_computed_once_per_pair(self, monkeypatch):
        """Ten thousand p-values at three degrees of freedom are three
        lgamma triples, not ten thousand."""
        seen = []
        ratio = _special._lgamma_ratio

        def counted(a, b):
            seen.append((a, b))
            return ratio(a, b)

        monkeypatch.setattr(_special, "_lgamma_ratio", counted)
        rng = np.random.default_rng(1)
        d2 = rng.choice(np.array([40.0, 60.0, 80.0]), 10_000)
        f_sf_array(rng.uniform(0.01, 3.0, 10_000), 1.0, d2)
        assert sorted(set(seen)) == sorted(seen)
        assert len(seen) == 3


class TestTheHandOffsToTheScalar:
    """Where numpy's per-call overhead outweighs the loop, the array form
    hands elements to the scalar. Those hand-offs are what make a small
    lead-lag search no slower than before, so they are pinned too."""

    def test_a_small_batch_never_enters_the_array_loop(self, monkeypatch):
        entered = []
        loop = _special._betacf_array
        monkeypatch.setattr(
            _special, "_betacf_array", lambda *a: entered.append(1) or loop(*a)
        )
        statistic, d1, d2 = _lead_lag_shaped(np.random.default_rng(0), 100)
        _assert_array_is_the_scalar(statistic, d1, d2)
        assert entered == []
        statistic, d1, d2 = _lead_lag_shaped(np.random.default_rng(0), 2000)
        _assert_array_is_the_scalar(statistic, d1, d2)
        assert entered

    def test_only_the_tail_is_finished_by_the_scalar(self, monkeypatch):
        """At most `_BETACF_ARRAY_TAIL` elements per array pass reach
        `betacf`; the rest converge in the array loop."""
        calls = []
        scalar = _special.betacf

        def counted(a, b, x, iterations=300):
            calls.append((a, b, x))
            return scalar(a, b, x, iterations)

        monkeypatch.setattr(_special, "betacf", counted)
        statistic, d1, d2 = _lead_lag_shaped(np.random.default_rng(1), 5000)
        got = f_sf_array(statistic, d1, d2)
        assert 0 < len(calls) <= 2 * _special._BETACF_ARRAY_TAIL
        monkeypatch.setattr(_special, "betacf", scalar)
        assert _bits(got) == _bits(
            [f_sf(*t) for t in zip(statistic.tolist(), d1.tolist(), d2.tolist())]
        )


class TestTheScalarIsUnchanged:
    """`betainc` now builds its front factor from `_lgamma_ratio`, shared
    with the array form. The same terms are added in the same order, so
    the scalar returns what it returned."""

    @pytest.mark.parametrize("seed", range(3))
    def test_random(self, seed):
        rng = np.random.default_rng(200 + seed)
        for _ in range(3000):
            statistic = float(10.0 ** rng.uniform(-5, 4))
            d1 = float(rng.uniform(0.05, 100))
            d2 = float(rng.choice([rng.integers(1, 3000), rng.uniform(0.05, 1e6)]))
            assert _bits([f_sf(statistic, d1, d2)]) == _bits(
                [_reference_f_sf(statistic, d1, d2)]
            )
            a, b, x = (
                float(rng.uniform(0.05, 500)),
                float(rng.uniform(0.05, 500)),
                float(rng.random()),
            )
            assert _bits([_special.betainc(a, b, x)]) == _bits(
                [_reference_betainc(a, b, x)]
            )
            assert _bits([_special.betacf(a, b, x)]) == _bits(
                [_reference_betacf(a, b, x)]
            )
