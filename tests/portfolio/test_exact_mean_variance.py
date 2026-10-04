"""
The convex mean-variance problems are solved exactly, and the answer says so.

min_volatility, target_return and long-only max_sharpe without a binding cap
are strictly convex quadratic programmes; max_sharpe becomes one under
y = w / k. SLSQP stopped 1e-6 to 1.6e-5 away from their optima on
min_volatility, and sometimes reported status 8 at the optimum to within 2e-9.
They are now solved by an active-set method that ends with one linear solve
on the final active set, and each answer carries a KKT certificate computed
from the weights alone. Capped and shorting max_sharpe still run SLSQP and
are then solved exactly on its active set. See the CHANGELOG entry of
2026-10-02.

WHAT IS PINNED, AND HOW IT IS KNOWN.
- Planted problems with closed-form optima: interior min-variance
  (Sigma^-1 1 normalized), interior tangency, a tangency portfolio held at
  a binding cap, a two-asset target (the two equalities fix the weights), a
  two-asset corner, a weakly active bound, and capped and long-only
  problems on a diagonal covariance, which are solved by water-filling.
- Random problems from 5 to 235 assets, live-like factor data and pure
  noise, long-only, capped and shorting. Each is certified by the library,
  and the KKT conditions are checked again in this file with its own
  arithmetic.
- The answer is never worse than SLSQP's on the objective, beyond 1e-12
  relative, once SLSQP's own constraint residual is charged at the shadow
  prices (weak duality). SLSQP stays callable here as the reference, with
  the settings it always had.
- The fallback chain is checked on real problems where the fast method
  fails: one where it cycles and one where it reaches a singular KKT system.
  Stages that real data does not reach are forced.
- Same input, same bits, at any BLAS thread count.
- An infeasible target is refused as before, and a non-converged SLSQP run
  says what happened rather than guessing infeasibility.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.portfolio import _active_set
from standard_quant_tools.portfolio import optimize as opt
from standard_quant_tools.portfolio.optimize import (
    HAS_SCIPY,
    annualized_mean_cov,
    mean_variance_optimize,
)

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="constrained path needs scipy")

TOL = _active_set.CERTIFICATE_TOLERANCE


# ── data ─────────────────────────────────────────────────────────────────


def _factor_frame(seed: int, n: int, T: int) -> pd.DataFrame:
    """Live-like daily returns: three factors, uneven loadings, drifts of
    both signs and idiosyncratic volatility from 0.6% to 3% a day."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(1.0, 0.4, (n, 3)) * np.array([1.0, 0.5, 0.3])
    factors = rng.normal(0.0, 0.008, (T, 3))
    drift = rng.normal(0.0004, 0.0015, n)
    idio = rng.uniform(0.006, 0.03, n)
    data = factors @ loadings.T + drift + rng.normal(0.0, 1.0, (T, n)) * idio
    return pd.DataFrame(data, columns=[f"A{i:02d}" for i in range(n)])


def _noise_frame(seed: int, n: int, T: int) -> pd.DataFrame:
    """Independent N(0, 1.2%) daily returns with no drift: the shape of the
    benchmark input that made SLSQP take seven seconds at 235 assets."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 0.012, (T, n)), columns=[f"N{i:03d}" for i in range(n)]
    )


def _planted_frame(mu: np.ndarray, cov: np.ndarray, T: int = 600) -> pd.DataFrame:
    """Daily returns whose sample annualized mean and covariance equal `mu`
    and `cov` to rounding: draws whitened to an identity sample covariance,
    then coloured."""
    n = len(mu)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(T, n))
    z -= z.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T
    daily = z @ np.linalg.cholesky(cov / 252.0).T + mu / 252.0
    return pd.DataFrame(daily, columns=[f"P{i}" for i in range(n)])


def _moments(frame: pd.DataFrame):
    """(mu, cov) exactly as `mean_variance_optimize` computes them: from
    `dropna()`'s copy, which pandas lays out column-major, so its mean and
    covariance can differ in the last bit from the original frame's."""
    return annualized_mean_cov(frame.dropna(), 252)


def _box(n: int, allow_short: bool, max_weight):
    """The bounds as the problem states them, with a long-only cap of 1 or
    more dropped (nonnegative weights summing to 1 cannot exceed it)."""
    cap = 1.0 if max_weight is None else max_weight
    lower = -cap if allow_short else 0.0
    upper = np.inf if (not allow_short and cap >= 1.0) else cap
    return np.full(n, lower), np.full(n, upper)


# ── an independent KKT check, with its own arithmetic ────────────────────


def _kkt_minvar(w, cov, lower, upper, A, b):
    """Residuals of min w'Sw s.t. A w = b, lower <= w <= upper. A weight
    is at a bound only if it EQUALS it: the exact solve puts fixed weights
    on their bounds, not near them."""
    g = 2.0 * cov @ w
    at_lo, at_hi = w == lower, w == upper
    free = ~(at_lo | at_hi)
    lam = np.linalg.lstsq(A[:, free].T, g[free], rcond=None)[0]
    r = g - A.T @ lam
    scale = np.max(2.0 * np.abs(cov) @ np.abs(w) + np.abs(A.T) @ np.abs(lam))
    stationarity = np.max(np.abs(r[free])) / scale
    dual = max(np.max(-r[at_lo], initial=0.0), np.max(r[at_hi], initial=0.0)) / scale
    return stationarity, dual


def _kkt_sharpe(w, mu, cov, rf, lower, upper):
    """Residuals of max (mu'w - rf)/sqrt(w'Sw) s.t. 1'w = 1 and the box,
    as the minimization of -Sharpe; again, at a bound means equal to it."""
    s = cov @ w
    var = w @ s
    vol = np.sqrt(var)
    excess = w @ mu - rf
    g = -mu / vol + excess * s / (vol * var)
    at_lo, at_hi = w == lower, w == upper
    free = ~(at_lo | at_hi)
    lam = np.mean(g[free])
    r = g - lam
    scale = np.max(np.abs(mu) / vol + excess * np.abs(cov) @ np.abs(w) / (vol * var))
    dual = max(np.max(-r[at_lo], initial=0.0), np.max(r[at_hi], initial=0.0))
    return np.max(np.abs(r[free])) / scale, dual / scale


def _bits(w) -> np.ndarray:
    return np.asarray(w, dtype=np.float64).view(np.uint64)


def _weights(result) -> np.ndarray:
    return np.array(list(result["weights"].values()))


def _result_bits(value):
    """A whole result as a comparable structure, every float by its bits."""
    if isinstance(value, dict):
        return {k: _result_bits(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_bits(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, float):
        return np.float64(value).tobytes()
    return value


# ── planted optima ───────────────────────────────────────────────────────


class TestPlantedOptima:
    def test_interior_minimum_variance_is_the_normalized_inverse_row_sum(self):
        """With Sigma^-1 1 positive the long-only bound never binds, so the
        answer is the textbook w = Sigma^-1 1 / 1'Sigma^-1 1."""
        vols = np.array([0.15, 0.18, 0.22, 0.25, 0.30])
        corr = np.full((5, 5), 0.2) + 0.8 * np.eye(5)
        frame = _planted_frame(np.full(5, 0.08), np.outer(vols, vols) * corr)
        _, cov = _moments(frame)
        raw = np.linalg.solve(cov, np.ones(5))
        assert np.all(raw > 0), "planted to be interior"
        result = mean_variance_optimize(frame, "min_volatility")
        assert result["solver"]["method"] == "active_set"
        assert result["solver"]["certified"] is True
        np.testing.assert_allclose(
            _weights(result), raw / raw.sum(), rtol=0, atol=1e-14
        )

    def test_interior_tangency_is_the_normalized_inverse_excess(self):
        vols = np.array([0.15, 0.18, 0.22, 0.25])
        corr = np.full((4, 4), 0.3) + 0.7 * np.eye(4)
        frame = _planted_frame(
            np.array([0.08, 0.09, 0.11, 0.12]), np.outer(vols, vols) * corr
        )
        mu, cov = _moments(frame)
        raw = np.linalg.solve(cov, mu - 0.02)
        assert np.all(raw > 0), "planted to be interior"
        result = mean_variance_optimize(frame, "max_sharpe", risk_free_rate=0.02)
        assert result["solver"]["certified"] is True
        np.testing.assert_allclose(
            _weights(result), raw / raw.sum(), rtol=0, atol=1e-14
        )

    def test_two_assets_and_a_target_have_one_feasible_portfolio(self):
        """Two equalities on two weights leave nothing to optimize:
        w1 = (t - mu2) / (mu1 - mu2)."""
        frame = _planted_frame(
            np.array([0.12, 0.05]), np.array([[0.04, 0.006], [0.006, 0.01]])
        )
        mu, _ = _moments(frame)
        result = mean_variance_optimize(frame, "target_return", target_return=0.10)
        w1 = (0.10 - mu[1]) / (mu[0] - mu[1])
        np.testing.assert_allclose(_weights(result), [w1, 1.0 - w1], rtol=0, atol=1e-15)
        assert result["converged"] is True

    def test_a_dominated_second_asset_gets_exactly_nothing(self):
        """sigma_12 > sigma_1^2: adding any of asset 2 raises the variance,
        so the optimum is the corner (1, 0) -- exactly, not 1 - 3e-6."""
        frame = _planted_frame(
            np.array([0.06, 0.08]), np.array([[0.01, 0.018], [0.018, 0.09]])
        )
        result = mean_variance_optimize(frame, "min_volatility")
        assert result["weights"] == {"P0": 1.0, "P1": 0.0}
        assert result["solver"]["certified"] is True

    def test_a_weakly_active_bound_does_not_cycle(self):
        """Sigma_12 == Sigma_11 puts the optimum at (1, 0) with a ZERO
        multiplier on w2 >= 0: the bound is active but not binding, which
        is where an active-set method can flip an index between fixed and
        free forever. Exact matrices, so the tie is exact."""
        Q = 2.0 * np.array([[0.04, 0.04], [0.04, 0.09]])
        solution = _active_set.solve(
            Q, np.zeros(2), np.ones((1, 2)), np.array([1.0]), 0.0, np.inf
        )
        np.testing.assert_array_equal(solution.x, [1.0, 0.0])

    def test_capped_minimum_variance_on_a_diagonal_covariance_water_fills(self):
        """Uncorrelated assets: w_i proportional to 1/v_i, with the cap
        binding on the lowest-variance names and the rest re-filled."""
        v = np.array([0.01, 0.02, 0.04, 0.08, 0.16])
        solution = _active_set.solve(
            2.0 * np.diag(v), np.zeros(5), np.ones((1, 5)), np.array([1.0]), 0.0, 0.3
        )
        rest = 1.0 / v[2:]
        expected = np.concatenate([[0.3, 0.3], 0.4 * rest / rest.sum()])
        np.testing.assert_allclose(solution.x, expected, rtol=0, atol=1e-15)
        assert (solution.n_lower, solution.n_upper) == (0, 2)

    def test_a_binding_cap_on_the_tangency_portfolio_holds_it_at_the_cap(self):
        """Uncorrelated, mu/v = (4, 1): the tangency portfolio is (0.8, 0.2).
        Sharpe is unimodal along the only feasible line, so a 0.6 cap gives
        (0.6, 0.4). This path runs SLSQP and then solves exactly on its
        active set."""
        frame = _planted_frame(np.array([0.16, 0.09]), np.diag([0.04, 0.09]))
        result = mean_variance_optimize(frame, "max_sharpe", max_weight=0.6)
        solver = result["solver"]
        assert solver["method"] == "active_set"
        assert solver["certified"] is True
        assert "from SLSQP's answer" in solver["message"]
        assert result["weights"]["P0"] == 0.6
        assert result["weights"]["P1"] == pytest.approx(0.4, abs=1e-15)

    def test_long_only_tangency_on_a_diagonal_covariance_drops_negative_excess(self):
        """y_i proportional to max(mu_i - rf, 0) / v_i."""
        mu = np.array([0.10, 0.02, 0.07, 0.01, 0.12])
        v = np.array([0.04, 0.02, 0.03, 0.05, 0.09])
        programme = opt._convex_programme(
            "max_sharpe", mu, np.diag(v), 0.03, None, False, None
        )
        w = programme.weights(programme.solve().x)
        raw = np.maximum(mu - 0.03, 0.0) / v
        np.testing.assert_allclose(w, raw / raw.sum(), rtol=0, atol=1e-15)
        assert np.all(w[[1, 3]] == 0.0)


# ── certificates on random problems ──────────────────────────────────────

_RANDOM = [
    (seed, n, kind, style, objective)
    for seed, n in enumerate((5, 12, 30, 60, 126, 235))
    for kind in ("factor", "noise")
    for style in ("long", "cap", "short")
    for objective in ("min_volatility", "target_return", "max_sharpe")
    # capped and shorting max_sharpe run SLSQP first: seconds at 126+ names
    if not (objective == "max_sharpe" and style != "long" and n > 60)
]


def _random_problem(seed, n, kind, style):
    T = max(10 * n, 400)
    frame = (_factor_frame if kind == "factor" else _noise_frame)(100 + seed, n, T)
    rng = np.random.default_rng(seed)
    allow_short = style == "short"
    max_weight = None
    if style == "cap":
        max_weight = float(rng.uniform(1.5 / n, min(1.0, 6.0 / n)))
    elif style == "short":
        max_weight = float(rng.uniform(1.5 / n, 0.5))
    return frame, allow_short, max_weight, rng


@pytest.mark.parametrize("seed,n,kind,style,objective", _RANDOM)
def test_random_problems_are_certified_and_check_out_independently(
    seed, n, kind, style, objective
):
    frame, allow_short, max_weight, rng = _random_problem(seed, n, kind, style)
    mu, cov = _moments(frame)
    kwargs = dict(allow_short=allow_short, max_weight=max_weight)
    if objective == "target_return":
        low, high = opt._attainable_return_range(mu, allow_short, max_weight)
        kwargs["target_return"] = low + float(rng.uniform(0.2, 0.9)) * (high - low)
    result = mean_variance_optimize(frame, objective, **kwargs)
    solver = result["solver"]
    assert solver["method"] == "active_set", solver["message"]
    assert solver["certified"] is True
    assert result["converged"] is True
    certificate = solver["certificate"]
    for key in ("stationarity", "dual_infeasibility", "equality_residual"):
        assert certificate[key] <= TOL, (key, certificate)
    assert certificate["bound_violation"] == 0.0

    w = _weights(result)
    lower, upper = _box(n, allow_short, max_weight)
    assert np.all(w >= lower) and np.all(w <= upper), "bounds hold exactly"
    assert abs(w.sum() - 1.0) <= 1e-14
    if objective == "max_sharpe":
        stationarity, dual = _kkt_sharpe(w, mu, cov, 0.0, lower, upper)
    else:
        A, b = np.ones((1, n)), np.array([1.0])
        if objective == "target_return":
            A = np.vstack([A, mu[None, :]])
            b = np.array([1.0, kwargs["target_return"]])
            assert abs(w @ mu - b[1]) <= 1e-12 * max(abs(b[1]), np.abs(mu) @ np.abs(w))
        stationarity, dual = _kkt_minvar(w, cov, lower, upper, A, b)
    assert stationarity <= TOL and dual <= TOL, (stationarity, dual)


# ── never worse than SLSQP ───────────────────────────────────────────────

_COMPARE = [
    (seed, n, kind, style, objective)
    for seed, n in enumerate((5, 8, 12, 20, 30, 45))
    for kind in ("factor", "noise")
    for style, objective in (
        ("long", "min_volatility"),
        ("cap", "min_volatility"),
        ("short", "min_volatility"),
        ("long", "max_sharpe"),
        ("cap", "max_sharpe"),
        ("short", "max_sharpe"),
        ("cap", "target_return"),
        ("short", "target_return"),
    )
]


def test_the_exact_answer_is_never_worse_than_slsqp():
    """
    The comparison table: for every problem, the old SLSQP answer (the same
    `_solve_slsqp` call, settings unchanged) against the new one.

    The comparison charges SLSQP's equality residual at the new answer's
    multipliers. By convexity, for any w inside the box,
    f(w*) <= f(w) - lam'(A w - b), so an answer that misses a target by
    2e-11 cannot pass for a better one. -Sharpe is not convex, but the same
    charge holds to first order, and the second-order term is about
    |dw|^2 ~ 1e-16 here. The charge matters: with five weights at a cap, an
    SLSQP answer summing to 1 - 9.6e-12 is a portfolio with every cap
    relaxed by that much, and its Sharpe ratio is 5e-12 higher for it. The
    new answer must not lose by more than 1e-12 relative. It may lose by
    rounding: two values agreeing to 1e-15 can land either way.
    """
    rows = []
    for seed, n, kind, style, objective in _COMPARE:
        frame, allow_short, max_weight, rng = _random_problem(seed, n, kind, style)
        mu, cov = _moments(frame)
        target = None
        if objective == "target_return":
            low, high = opt._attainable_return_range(mu, allow_short, max_weight)
            target = low + float(rng.uniform(0.2, 0.9)) * (high - low)
        best_w = (
            opt._max_attainable_return(mu, allow_short, max_weight)[1]
            if objective == "max_sharpe"
            else None
        )
        old = opt._solve_slsqp(
            mu, cov, objective, 0.0, target, None, allow_short, max_weight, best_w
        ).weights
        new_result = mean_variance_optimize(
            frame,
            objective,
            target_return=target,
            allow_short=allow_short,
            max_weight=max_weight,
        )
        new = _weights(new_result)
        lower, upper = _box(n, allow_short, max_weight)
        assert np.all(old >= lower) and np.all(old <= upper), "SLSQP clips to bounds"
        if objective == "max_sharpe":
            f_old = -(old @ mu) / np.sqrt(old @ cov @ old)
            f_new = -(new @ mu) / np.sqrt(new @ cov @ new)
        else:
            f_old, f_new = old @ cov @ old, new @ cov @ new
        A, b = np.ones((1, n)), np.array([1.0])
        if target is not None:
            A, b = np.vstack([A, mu[None, :]]), np.array([1.0, target])
        charge = float(np.array(new_result["solver"]["multipliers"]) @ (A @ old - b))
        gap = (f_old - charge - f_new) / abs(f_new)
        rows.append(
            f"{n:3d} {kind:6s} {objective:14s} {style:5s} f_old={f_old:.15e} "
            f"f_new={f_new:.15e} gap={gap:+.1e} max|dw|={np.max(np.abs(old - new)):.1e}"
        )
        assert gap >= -TOL, "\n".join(rows)
        assert np.max(np.abs(old - new)) < 1e-4, "\n".join(rows)
    assert len(rows) == len(_COMPARE)


# ── the fallback chain ───────────────────────────────────────────────────


def _cycling_case():
    """Eight assets, cap 0.25, a target 95% of the way up the attainable
    range: the primal-dual method's cold start returns on pass 6 to the guess
    of pass 2."""
    frame = _factor_frame(70, 8, 600)
    mu, cov = _moments(frame)
    low, high = opt._attainable_return_range(mu, False, 0.25)
    target = low + 0.95 * (high - low)
    programme = opt._convex_programme(
        "target_return", mu, cov, 0.0, target, False, 0.25
    )
    return frame, mu, cov, target, programme


def _singular_case():
    """Six assets, cap 0.4, a target 95% up the range: the cold start
    guesses a single free weight for two equality rows."""
    frame = _factor_frame(2, 6, 600)
    mu, cov = _moments(frame)
    low, high = opt._attainable_return_range(mu, False, 0.4)
    target = low + 0.95 * (high - low)
    programme = opt._convex_programme("target_return", mu, cov, 0.0, target, False, 0.4)
    return frame, mu, cov, target, programme


class TestFallbackChain:
    def test_the_cycling_case_cycles_from_a_cold_start(self):
        *_, programme = _cycling_case()
        with pytest.raises(_active_set.ActiveSetFailure, match="cycled"):
            programme.solve()

    def test_the_cycling_case_reaches_the_fallback_and_is_still_certified(self):
        frame, mu, cov, target, programme = _cycling_case()
        result = mean_variance_optimize(
            frame, "target_return", target_return=target, max_weight=0.25
        )
        solver = result["solver"]
        assert solver["method"] == "active_set"
        assert solver["certified"] is True
        assert "cycled" in solver["fallback"]
        assert "primal active-set method" in solver["message"]
        assert result["warnings"] == []
        # The answer is a function of the final active set: SLSQP's answer
        # polished by the fast method lands on the same bits.
        slsqp = opt._solve_slsqp(
            mu, cov, "target_return", 0.0, target, None, False, 0.25, None
        )
        polished = programme.weights(programme.solve(start=slsqp.weights).x)
        np.testing.assert_array_equal(_bits(_weights(result)), _bits(polished))

    def test_a_singular_guess_reaches_the_fallback_and_is_certified(self):
        frame, _, _, target, programme = _singular_case()
        with pytest.raises(_active_set.ActiveSetFailure, match="singular"):
            programme.solve()
        result = mean_variance_optimize(
            frame, "target_return", target_return=target, max_weight=0.4
        )
        assert result["solver"]["certified"] is True
        assert "singular" in result["solver"]["fallback"]

    def test_when_the_primal_method_fails_too_slsqp_supplies_the_start(
        self, monkeypatch
    ):
        frame, *_, target, programme = _cycling_case()
        expected = mean_variance_optimize(
            frame, "target_return", target_return=target, max_weight=0.25
        )

        def fail(*args, **kwargs):
            raise _active_set.ActiveSetFailure("forced", 1)

        monkeypatch.setattr(_active_set, "solve_primal", fail)
        result = mean_variance_optimize(
            frame, "target_return", target_return=target, max_weight=0.25
        )
        solver = result["solver"]
        assert solver["method"] == "active_set"
        assert solver["certified"] is True
        assert "from SLSQP's answer" in solver["message"]
        assert "primal method stopped: forced" in solver["fallback"]
        assert solver["n_function_evals"] >= 1, "SLSQP ran"
        np.testing.assert_array_equal(
            _bits(_weights(result)), _bits(_weights(expected))
        )

    def test_when_every_exact_stage_fails_slsqp_answers_flagged(
        self, monkeypatch, caplog
    ):
        frame, mu, cov, target, _ = _cycling_case()

        def fail(*args, **kwargs):
            raise _active_set.ActiveSetFailure("forced", 1)

        monkeypatch.setattr(_active_set, "solve", fail)
        monkeypatch.setattr(_active_set, "solve_primal", fail)
        with caplog.at_level(logging.WARNING, logger=opt.__name__):
            result = mean_variance_optimize(
                frame, "target_return", target_return=target, max_weight=0.25
            )
        solver = result["solver"]
        assert solver["method"] == "SLSQP"
        assert solver["certified"] is False
        assert solver["certificate"]["stationarity"] > TOL
        # The weights are exactly what SLSQP has always returned here.
        reference = opt._solve_slsqp(
            mu, cov, "target_return", 0.0, target, None, False, 0.25, None
        )
        np.testing.assert_array_equal(_bits(_weights(result)), _bits(reference.weights))
        assert result["converged"] is reference.converged
        (warning,) = [w for w in result["warnings"] if "certified optimum" in w]
        assert "KKT stationarity residual" in warning
        assert "sum-to-1 residual" in warning
        assert "target_return residual" in warning
        assert any("not certified optimal" in r.message for r in caplog.records)


# ── infeasible targets, as before ────────────────────────────────────────


class TestInfeasibleTargetsAreRefusedAsBefore:
    @pytest.mark.parametrize("side", ["above", "below"])
    def test_a_target_outside_the_attainable_range(self, side):
        frame = _factor_frame(5, 6, 600)
        mu, cov = _moments(frame)
        low, high = opt._attainable_return_range(mu, False, None)
        target = high + 0.05 if side == "above" else low - 0.05
        result = mean_variance_optimize(frame, "target_return", target_return=target)
        assert result["converged"] is False
        solver = result["solver"]
        assert solver["method"] == "SLSQP"
        assert solver["certified"] is None, "no optimum exists to certify"
        assert any(
            f"is outside [{low:.6f}, {high:.6f}]" in w for w in result["warnings"]
        )
        assert any("do not satisfy" in w for w in result["warnings"])
        reference = opt._solve_slsqp(
            mu, cov, "target_return", 0.0, target, None, False, None, None
        )
        np.testing.assert_array_equal(_bits(_weights(result)), _bits(reference.weights))


# ── determinism ──────────────────────────────────────────────────────────


class TestDeterminism:
    _CASES = [
        ("min_volatility", {}),
        ("min_volatility", {"max_weight": 0.03}),
        ("max_sharpe", {}),
        (
            "target_return",
            {"target_return": 0.30, "allow_short": True, "max_weight": 0.1},
        ),
    ]

    @pytest.mark.parametrize("objective,kwargs", _CASES)
    def test_same_input_same_bits(self, objective, kwargs):
        frame = _factor_frame(11, 126, 1500)
        first = mean_variance_optimize(frame, objective, **kwargs)
        second = mean_variance_optimize(frame, objective, **kwargs)
        assert first["solver"]["certified"] is True
        np.testing.assert_array_equal(_bits(_weights(first)), _bits(_weights(second)))

    @pytest.mark.parametrize("objective,kwargs", _CASES)
    def test_the_bits_do_not_depend_on_the_blas_thread_count(self, objective, kwargs):
        threadpoolctl = pytest.importorskip("threadpoolctl")
        frame = _factor_frame(12, 235, 2000)
        mu, cov = _moments(frame)
        programme = opt._convex_programme(
            objective,
            mu,
            cov,
            0.0,
            kwargs.get("target_return"),
            kwargs.get("allow_short", False),
            kwargs.get("max_weight"),
        )
        answers = []
        for threads in (1, 4):
            with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
                result = opt._solve_exactly(
                    programme,
                    mu,
                    cov,
                    objective,
                    0.0,
                    kwargs.get("target_return"),
                    kwargs.get("allow_short", False),
                    kwargs.get("max_weight"),
                    None,
                )
            assert result[2]["certified"] is True
            answers.append(result[0])
        np.testing.assert_array_equal(_bits(answers[0]), _bits(answers[1]))

    @pytest.mark.parametrize("kind", ["noise", "factor"])
    def test_the_condition_number_does_not_depend_on_the_blas_thread_count(self, kind):
        """At 235 assets a bare np.linalg.cond of these covariances differs
        in the last bits between one and four BLAS threads: the factor
        frame's is 781.0238456977911 on one and 781.023845697788 on four
        under numpy 2.0's OpenBLAS, and the noise frame's differs under
        OpenBLAS 0.3.31. The SVD runs on one thread whatever the caller set.

        So does the covariance it is the condition number of. Its product
        kept the caller's threads until the CHANGELOG entry of 2026-10-04,
        and on the CI runners' OpenBLAS it gave different last bits at one
        and four threads, so this test failed there. The reported number is
        now one number at caller limits of one, two and four: the one-thread
        condition number of the one-thread covariance, computed here from
        pandas and numpy directly. Each call's number is also the
        one-thread condition number of the covariance that call's caller
        setting estimates."""
        threadpoolctl = pytest.importorskip("threadpoolctl")
        frame = (_noise_frame if kind == "noise" else _factor_frame)(12, 235, 2000)
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            one_thread_cov = frame.dropna().cov().to_numpy(dtype=float) * 252
            one_thread = float(np.linalg.cond(one_thread_cov))
        reported, per_matrix = [], []
        for threads in (1, 2, 4):
            with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
                result = mean_variance_optimize(frame, "min_volatility")
                _, cov = _moments(frame)
            with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
                per_matrix.append(float(np.linalg.cond(cov)))
            reported.append(result["condition_number"])
        assert reported == per_matrix
        assert reported == [one_thread] * 3

    @pytest.mark.parametrize(
        "objective,kwargs",
        [
            ("min_volatility", {}),
            ("max_sharpe", {}),
            ("min_volatility", {"allow_short": True}),
            ("max_sharpe", {"allow_short": True}),
        ],
    )
    def test_the_whole_answer_does_not_depend_on_the_blas_thread_count(
        self, objective, kwargs
    ):
        """Every number the optimizer returns -- the weights, the expected
        return and volatility, the Sharpe ratio, the solver's report and
        certificate, the condition number -- is the same bits at caller
        limits of one, two and four BLAS threads (see the CHANGELOG entry
        of 2026-10-04). Two steps kept the caller's threads before: the
        covariance product, whose last bits followed the thread count on
        the CI runners' OpenBLAS, and the closed form's inverse (the
        shorting cases here), whose last bits followed it under OpenBLAS
        0.3.27 and 0.3.31 on a 16-thread Windows machine as well."""
        threadpoolctl = pytest.importorskip("threadpoolctl")
        frame = _factor_frame(12, 235, 2000)
        answers = []
        for threads in (1, 2, 4):
            with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
                answers.append(
                    _result_bits(mean_variance_optimize(frame, objective, **kwargs))
                )
        assert answers[1] == answers[0]
        assert answers[2] == answers[0]


# ── what a non-converged run says ────────────────────────────────────────


def _stalled(monkeypatch):
    """SLSQP's real answer, reported the way a status-8 stop reports it."""
    real = opt._solve_constrained

    def stalled(*args, **kwargs):
        w, _, report = real(*args, **kwargs)
        report = dict(report)
        report["status"] = 8
        report["message"] = "Positive directional derivative for linesearch"
        return w, False, report

    monkeypatch.setattr(opt, "_solve_constrained", stalled)


class TestNonConvergenceSaysWhatHappened:
    def test_a_status_8_stop_names_the_solver_status_and_residuals(
        self, monkeypatch, caplog
    ):
        """target_volatility stays with SLSQP alone. A stalled run used to
        log that the weights "may violate the sum-to-1 constraint" and put
        nothing in `warnings`."""
        frame = _factor_frame(3, 8, 600)
        _stalled(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=opt.__name__):
            result = mean_variance_optimize(
                frame, "target_volatility", target_volatility=0.25
            )
        assert result["converged"] is False
        assert result["solver"]["certified"] is None
        (warning,) = [w for w in result["warnings"] if "status 8" in w]
        assert "SLSQP ended with status 8" in warning
        assert "sum-to-1 residual" in warning
        assert "largest bound violation" in warning
        assert "target_volatility residual" in warning
        assert "does not by itself mean the constraints are infeasible" in warning
        messages = " ".join(r.message for r in caplog.records)
        assert "status 8" in messages
        assert "may violate" not in messages

    def test_a_stalled_capped_max_sharpe_is_polished_to_the_certified_optimum(
        self, monkeypatch
    ):
        """Status 8 is SLSQP failing to prove optimality, not failing to be
        near it: solved exactly on its active set, the answer certifies."""
        frame = _factor_frame(3, 8, 600)
        _stalled(monkeypatch)
        result = mean_variance_optimize(frame, "max_sharpe", max_weight=0.3)
        assert result["converged"] is True
        assert result["solver"]["method"] == "active_set"
        assert result["solver"]["certified"] is True
        assert "SLSQP ended with status 8" in result["solver"]["message"]
        assert result["warnings"] == []

    def test_when_the_polish_fails_the_residuals_are_in_the_warning(self, monkeypatch):
        """Twelve assets, five weights free: SLSQP's own answer is 1.7e-8
        from stationary, so without the polish it does not certify."""
        frame = _factor_frame(3, 12, 600)
        _stalled(monkeypatch)

        def fail(*args, **kwargs):
            raise _active_set.ActiveSetFailure("forced", 1)

        monkeypatch.setattr(_active_set, "solve_homogeneous", fail)
        result = mean_variance_optimize(frame, "max_sharpe", max_weight=0.3)
        assert result["converged"] is False
        assert result["solver"]["method"] == "SLSQP"
        assert result["solver"]["certified"] is False
        (warning,) = [w for w in result["warnings"] if "status 8" in w]
        assert "KKT stationarity residual" in warning
        assert "sum-to-1 residual" in warning

    def test_a_covered_objective_is_not_affected_by_slsqp_stalling(self, monkeypatch):
        """SLSQP no longer runs for these unless the exact stages fail."""
        frame = _factor_frame(3, 8, 600)
        _stalled(monkeypatch)
        result = mean_variance_optimize(frame, "max_sharpe")
        assert result["converged"] is True
        assert result["solver"]["method"] == "active_set"
        assert result["solver"]["n_function_evals"] is None
        assert result["warnings"] == []
