"""
`optimize_rebalance` against answers known without it.

Nothing here is a recorded output. Each answer is checked against one of:

- THE CONSTRAINED OPTIMIZER: with no trading cost and no turnover cap the
  programme is optimize_portfolio's max_utility, so both must give the same
  weights and the same shadow prices;
- a CLOSED FORM: on a diagonal covariance with a box and no binding budget
  every weight is its own one-dimensional problem, whose answer is the
  soft-threshold of the frictionless weight around the current one, clipped
  to the box;
- a PLANTED NO-TRADE REGION: forecasts built so that every current weight
  sits inside its band, where the answer is the current weights to the bit;
- a BRUTE FORCE: on small random problems, every combination of states --
  each weight held at one of its breakpoints or free on one of its pieces,
  each row and cap bound or idle -- solved as an equality-constrained QP,
  and the best feasible answer kept; that is the optimum whatever method
  found it;
- FINITE DIFFERENCES: a shadow price is the derivative of the optimal
  utility in its bound, so moving the bound and solving again must move
  the utility by price times step.

Plus determinism, warm starts that change the passes and not the answer,
and the refusals.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import general_optimizer as go
from standard_quant_tools.portfolio.general_optimizer import optimize_portfolio
from standard_quant_tools.portfolio.rebalance_optimizer import (
    REBALANCE_STATES,
    optimize_rebalance,
)

# ── data ─────────────────────────────────────────────────────────────────


def _universe(n: int, seed: int, factors: int = 3):
    """A sample covariance of factor-driven returns (annualized), expected
    returns of both signs, loadings and names."""
    rng = np.random.default_rng(seed)
    obs = max(4 * n, 120)
    loadings = rng.normal(1.0, 0.5, (n, factors))
    returns = rng.normal(0, 0.01, (obs, factors)) @ loadings.T + rng.normal(
        0, 0.015, (obs, n)
    )
    cov = np.cov(returns, rowvar=False) * 252
    names = [f"A{i:02d}" for i in range(n)]
    mu = rng.normal(0.06, 0.05, n)
    return names, cov, mu, loadings, rng


def _frame(names, cov):
    return pd.DataFrame(cov, index=names, columns=names)


def _w(result, names):
    return np.array([result["weights"][a] for a in names])


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / max(
        1.0, float(np.max(np.abs(b)))
    )


def _book(rng, n, cap):
    """A long-only book summing to one with no weight above `cap`."""
    w = rng.dirichlet(np.ones(n))
    for _ in range(60):
        w = np.minimum(w, cap)
        w = w / w.sum()
    w = np.minimum(w, cap)
    return w / w.sum()


def _prices(result):
    return {
        (b["kind"], b["name"]): b["shadow_price"] for b in result["binding_constraints"]
    }


def _fd(fn, x, h=1e-6):
    return (fn(x + h) - fn(x - h)) / (2 * h)


def _utility(w, mu, S, gamma, w0, kappa, eta):
    d = w - w0
    return float(mu @ w - 0.5 * gamma * w @ S @ w - kappa @ np.abs(d) - eta @ (d * d))


# ── zero costs: the constrained optimizer ────────────────────────────────


def _optimizer_problem(seed):
    names, cov, mu, loadings, rng = _universe(int(8 + seed % 5 * 6), seed=300 + seed)
    n = len(names)
    kwargs = dict(
        expected_returns=mu,
        risk_aversion=float(rng.choice([1.5, 4.0, 12.0])),
    )
    if seed % 3 == 0:
        kwargs.update(lower_bounds=0.0, upper_bounds=3.0 / n)
    elif seed % 3 == 1:
        kwargs.update(lower_bounds=-0.1, upper_bounds=0.35)
    else:
        kwargs.update(lower_bounds=0.0, upper_bounds=None, budget_type="at_most")
    if seed % 2 == 0:
        kwargs.update(
            groups={"g1": names[: n // 3], "g2": names[n // 3 : 2 * n // 3]},
            group_max={"g1": 0.3},
            group_min={"g2": 0.2},
        )
    if seed % 4 == 1:
        kwargs.update(
            factor_loadings=pd.DataFrame(loadings[:, :1], index=names, columns=["f"]),
            exposure_min={"f": 0.95},
            exposure_max={"f": 1.05},
        )
    return names, cov, kwargs


@pytest.mark.parametrize("seed", range(10))
def test_without_costs_it_is_the_constrained_optimizer(seed):
    """With kappa = eta = 0 and no turnover cap the objective is
    mu'w - (gamma/2) w'Sw, optimize_portfolio's max_utility: under the same
    bounds, budget, groups and exposures both give the same weights and
    every binding constraint the same shadow price, to 1e-12 relative --
    whatever the current weights, which then cost nothing to leave."""
    names, cov, kwargs = _optimizer_problem(seed)
    try:
        reference = optimize_portfolio(
            _frame(names, cov), objective="max_utility", **kwargs
        )
    except ValidationError:
        pytest.skip("this draw's limits are infeasible")
    current = np.random.default_rng(seed).dirichlet(np.ones(len(names)))
    out = optimize_rebalance(_frame(names, cov), current_weights=current, **kwargs)
    assert reference["certified"] and out["certified"], out["warnings"]
    assert _rel(_w(out, names), _w(reference, names)) < 1e-12
    assert out["utility"] == pytest.approx(
        reference["objective_value"], rel=1e-12, abs=1e-15
    )
    ours = _prices(out)
    theirs = {
        (b["constraint"], b["name"]): b["shadow_price"]
        for b in reference["binding_constraints"]
    }
    assert set(ours) == set(theirs)
    scale = max(1.0, max(abs(v) for v in theirs.values()))
    for key, price in theirs.items():
        assert abs(ours[key] - price) <= 1e-12 * scale, key


def test_a_volatility_cap_without_costs_is_the_optimizers_cap():
    """The volatility cap is optimize_portfolio's search on the same
    programme: with no costs both find the same weights, the same raised
    risk aversion and the same price for the cap."""
    names, cov, mu, _, _ = _universe(12, seed=21)
    free = optimize_portfolio(
        _frame(names, cov),
        objective="max_utility",
        expected_returns=mu,
        upper_bounds=0.4,
    )
    cap = 0.9 * free["volatility"]
    reference = optimize_portfolio(
        _frame(names, cov),
        objective="max_utility",
        expected_returns=mu,
        upper_bounds=0.4,
        max_volatility=cap,
    )
    out = optimize_rebalance(
        _frame(names, cov), expected_returns=mu, upper_bounds=0.4, max_volatility=cap
    )
    assert out["certified"] and reference["certified"]
    assert _rel(_w(out, names), _w(reference, names)) < 1e-10
    assert out["effective_risk_aversion"] == pytest.approx(
        reference["effective_risk_aversion"], rel=1e-9
    )
    assert _prices(out)[("max_volatility", None)] == pytest.approx(
        {
            (b["constraint"], b["name"]): b["shadow_price"]
            for b in reference["binding_constraints"]
        }[("max_volatility", None)],
        rel=1e-8,
    )


def test_a_fixed_weight_is_held_and_priced_as_the_optimizer_prices_it():
    """An asset whose bounds are equal is fixed: without costs its price is
    optimize_portfolio's, and with costs it is still held at its bound,
    its forced trade counted in the turnover."""
    names, cov, mu, _, _ = _universe(9, seed=81)
    lower = {a: 0.0 for a in names}
    upper = {a: 0.3 for a in names}
    lower["A03"] = upper["A03"] = 0.07
    kwargs = dict(expected_returns=mu, lower_bounds=lower, upper_bounds=upper)
    reference = optimize_portfolio(
        _frame(names, cov), objective="max_utility", **kwargs
    )
    out = optimize_rebalance(_frame(names, cov), **kwargs)
    assert out["certified"]
    assert _rel(_w(out, names), _w(reference, names)) < 1e-12
    theirs = {
        (b["constraint"], b["name"]): b["shadow_price"]
        for b in reference["binding_constraints"]
    }
    assert _prices(out)[("fixed_weight", "A03")] == pytest.approx(
        theirs[("fixed_weight", "A03")], rel=1e-12, abs=1e-15
    )
    w0 = np.full(9, 1 / 9)
    costly = optimize_rebalance(
        _frame(names, cov),
        current_weights=w0,
        linear_cost=0.002,
        max_turnover=0.3,
        **kwargs,
    )
    assert costly["certified"]
    assert costly["weights"]["A03"] == 0.07
    assert costly["active_set"]["A03"] == "fixed"
    assert abs(costly["turnover"] - 0.3) <= 1e-12


# ── the closed form: soft-thresholding ───────────────────────────────────


def _soft_threshold(mu, s, gamma, w0, kappa, eta, lo, hi):
    """Each weight's own problem, maximize mu w - (gamma/2) s w^2
    - kappa |w - w0| - eta (w - w0)^2 on [lo, hi]: the stationary point of
    the side it moves to, or w0 when neither side's is past it; a convex
    problem in one variable, so the box clips it."""
    curve = gamma * s + 2.0 * eta
    up = (mu - kappa + 2.0 * eta * w0) / curve
    down = (mu + kappa + 2.0 * eta * w0) / curve
    w = np.where(up > w0, up, np.where(down < w0, down, w0))
    return np.clip(w, lo, hi)


@pytest.mark.parametrize("eta", [0.0, 0.03])
def test_a_diagonal_covariance_soft_thresholds_each_weight(eta):
    """On a diagonal covariance with box bounds and a budget too large to
    bind, the weights do not interact: each is the soft-threshold of its
    frictionless weight around its current weight, clipped to the box --
    and a weight inside its band is the current weight to the bit."""
    rng = np.random.default_rng(5)
    n = 12
    s = rng.uniform(0.02, 0.09, n)
    mu = rng.normal(0.03, 0.04, n)
    w0 = rng.uniform(-0.2, 0.5, n)
    kappa = rng.uniform(0.0, 0.03, n)
    gamma = 3.0
    lo, hi = -0.3, 0.6
    names = [f"d{i}" for i in range(n)]
    out = optimize_rebalance(
        _frame(names, np.diag(s)),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=gamma,
        linear_cost=kappa,
        quadratic_cost=eta,
        lower_bounds=lo,
        upper_bounds=hi,
        budget=100.0,
        budget_type="at_most",
    )
    assert out["certified"], out["warnings"]
    expected = _soft_threshold(mu, s, gamma, w0, kappa, eta, lo, hi)
    assert _rel(_w(out, names), expected) < 1e-13
    held = np.flatnonzero(expected == w0)
    assert held.size >= 2  # the draw plants some weights in their bands
    for i in held:
        assert out["weights"][names[i]] == w0[i]
        assert names[i] in out["no_trade"]


# ── the no-trade region ──────────────────────────────────────────────────


def _planted_no_trade(seed, n=15, kappa=0.01):
    """Forecasts mu = gamma S w0 + lambda 1 + delta with |delta_i| < kappa:
    at w0, with the budget's multiplier -lambda, every weight's gradient
    lies strictly inside its band [-kappa, kappa], so trading anything
    costs more than it earns."""
    names, cov, _, _, rng = _universe(n, seed=seed)
    w0 = _book(rng, n, 0.2)
    gamma = 4.0
    delta = rng.uniform(-0.9, 0.9, n) * kappa
    mu = gamma * cov @ w0 + 0.05 + delta
    return names, cov, mu, w0, gamma


@pytest.mark.parametrize("seed", range(4))
def test_a_planted_no_trade_region_returns_the_current_weights_exactly(seed):
    """Every current weight inside its band: the answer is w0 itself --
    compared with ==, not to a tolerance -- the turnover is 0.0 and every
    asset is in no_trade, certified."""
    names, cov, mu, w0, gamma = _planted_no_trade(30 + seed)
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=dict(zip(names, w0)),
        risk_aversion=gamma,
        linear_cost=0.01,
        upper_bounds=0.5,
    )
    assert out["certified"], out["warnings"]
    assert np.array_equal(_w(out, names), w0)
    assert out["turnover"] == 0.0
    assert out["linear_cost"] == 0.0
    assert sorted(out["no_trade"]) == sorted(names)
    assert set(out["active_set"].values()) == {"hold"}


def test_a_turnover_cap_that_does_not_bind_leaves_the_no_trade_region_exact():
    """The same planted book with a turnover cap present (so every weight
    has a kink at its current weight on the cap's account too) and a quadratic
    cost: still w0 to the bit, and the cap is not reported as binding."""
    names, cov, mu, w0, gamma = _planted_no_trade(41)
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=gamma,
        linear_cost=0.01,
        quadratic_cost=0.05,
        max_turnover=0.2,
    )
    assert out["certified"]
    assert np.array_equal(_w(out, names), w0)
    assert ("max_turnover", None) not in _prices(out)


def test_only_the_weights_outside_their_bands_trade():
    """Plant half the book inside its bands and push the other half's
    forecasts far outside: the first half comes back as w0 exactly, the
    second trades."""
    names, cov, mu, w0, gamma = _planted_no_trade(52, n=12)
    mu = mu.copy()
    mu[:6] += np.array([0.08, -0.08, 0.1, -0.1, 0.12, -0.12])
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=gamma,
        linear_cost=0.01,
    )
    assert out["certified"]
    traded = [a for a in names if a not in out["no_trade"]]
    assert set(names[:6]) <= set(traded)
    for a, v in zip(names, w0):
        if a in out["no_trade"]:
            assert out["weights"][a] == v


# ── the brute force ──────────────────────────────────────────────────────


def _brute_force(mu, S, gamma, w0, kappa, eta, lo, hi, rows, turnover, gross):
    """
    The optimum by enumeration, sharing no code with the solver. Every
    weight is held at one of its finite breakpoints (its bounds, its
    current weight where it is kinked, zero under a gross cap) or free on
    one of the pieces between them, where the costs and caps are linear;
    every inequality row and cap is bound or idle. Each combination is an
    equality-constrained QP solved by one linear system. The true optimum's
    own combination reproduces it exactly, and every other feasible answer
    is a feasible point, so the best feasible utility over all of them is
    the optimum. Returns (utility, weights), or (-inf, None) when nothing
    is feasible.
    """
    n = mu.size
    Hm = gamma * S + 2.0 * np.diag(eta)
    cvec = -mu - 2.0 * eta * w0
    states = []
    for i in range(n):
        points = {lo[i], hi[i]}
        if kappa[i] > 0 or turnover is not None:
            points.add(w0[i])
        if gross is not None:
            points.add(0.0)
        points = sorted(p for p in points if lo[i] <= p <= hi[i])
        mine = [("held", p) for p in points if math.isfinite(p)]
        edges = [lo[i]] + [p for p in points if lo[i] < p < hi[i]] + [hi[i]]
        mine += [("free", a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]
        states.append(mine)
    row_states = []
    for coef, low, high in rows:
        if low == high:
            row_states.append([low])
        else:
            row_states.append([None] + [v for v in (low, high) if math.isfinite(v)])
    cap_states = [
        [False, True] if turnover is not None else [False],
        [False, True] if gross is not None else [False],
    ]
    best = (-math.inf, None)
    for pattern in itertools.product(*states):
        held = np.array([s[0] == "held" for s in pattern])
        F = np.flatnonzero(~held)
        x_held = np.array([s[1] if s[0] == "held" else 0.0 for s in pattern])
        slope = np.zeros(n)
        trade_sign = np.sign(x_held - w0)
        weight_sign = np.sign(x_held)
        for i, s in enumerate(pattern):
            if s[0] == "free":
                a, b = s[1], s[2]
                if math.isfinite(a) and math.isfinite(b):
                    mid = 0.5 * (a + b)
                elif math.isfinite(a):
                    mid = a + 1.0
                else:
                    mid = b - 1.0
                trade_sign[i] = 1.0 if mid > w0[i] else -1.0
                weight_sign[i] = 1.0 if mid > 0 else -1.0
                slope[i] = kappa[i] * trade_sign[i]
        for chosen in itertools.product(*row_states):
            for t_on in cap_states[0]:
                for g_on in cap_states[1]:
                    A = [coef for (coef, _, _), v in zip(rows, chosen) if v is not None]
                    b = [v for v in chosen if v is not None]
                    if t_on:
                        A.append(trade_sign)
                        b.append(turnover + trade_sign @ w0)
                    if g_on:
                        A.append(weight_sign)
                        b.append(gross)
                    A = np.array(A, dtype=float).reshape(-1, n)
                    b = np.array(b, dtype=float)
                    k, m = F.size, A.shape[0]
                    w = x_held.copy()
                    if k:
                        K = np.zeros((k + m, k + m))
                        K[:k, :k] = Hm[np.ix_(F, F)]
                        K[:k, k:] = A[:, F].T
                        K[k:, :k] = A[:, F]
                        H_fixed = np.flatnonzero(held)
                        rhs = np.concatenate(
                            [
                                -(
                                    cvec[F]
                                    + slope[F]
                                    + Hm[np.ix_(F, H_fixed)] @ x_held[H_fixed]
                                ),
                                b - A[:, H_fixed] @ x_held[H_fixed],
                            ]
                        )
                        if np.linalg.matrix_rank(K) < k + m:
                            continue
                        w[F] = np.linalg.solve(K, rhs)[:k]
                    elif m and np.max(np.abs(A @ w - b)) > 1e-12:
                        continue
                    tol = 1e-10
                    if any(
                        s[0] == "free" and not (s[1] - tol <= w[i] <= s[2] + tol)
                        for i, s in enumerate(pattern)
                    ):
                        continue
                    if any(
                        coef @ w < low - tol or coef @ w > high + tol
                        for coef, low, high in rows
                    ):
                        continue
                    if turnover is not None and np.sum(np.abs(w - w0)) > turnover + tol:
                        continue
                    if gross is not None and np.sum(np.abs(w)) > gross + tol:
                        continue
                    value = _utility(w, mu, S, gamma, w0, kappa, eta)
                    if value > best[0]:
                        best = (value, w)
    return best


def _small_problem(seed, n):
    rng = np.random.default_rng(1000 + seed)
    A = rng.normal(size=(n, n))
    S = (A @ A.T) / n * 0.04 + np.diag(rng.uniform(0.01, 0.05, n))
    mu = rng.normal(0.05, 0.08, n)
    gamma = float(rng.uniform(1, 6))
    long_short = seed % 2 == 0
    if long_short:
        lo, hi = np.full(n, -0.5), np.full(n, 0.8)
        w0 = rng.uniform(-0.3, 0.6, n)
    else:
        lo, hi = np.zeros(n), np.full(n, float(rng.choice([0.6, 1.0, 0.45])))
        w0 = np.minimum(rng.dirichlet(np.ones(n)), hi)
    kappa = rng.uniform(0, 0.03, n) * (rng.random(n) < 0.8)
    eta = rng.uniform(0, 0.05, n) * (seed % 3 == 0)
    budget_eq = seed % 4 != 3
    turnover = float(rng.uniform(0.05, 0.8)) if seed % 5 in (1, 2, 4) else None
    gross = float(rng.uniform(1.0, 1.6)) if long_short and seed % 3 != 1 else None
    names = [f"a{i}" for i in range(n)]
    rows = [(np.ones(n), 1.0, 1.0) if budget_eq else (np.ones(n), -math.inf, 1.0)]
    kwargs = dict(
        expected_returns=mu,
        risk_aversion=gamma,
        current_weights=w0,
        linear_cost=kappa,
        quadratic_cost=eta,
        lower_bounds=lo,
        upper_bounds=hi,
        budget_type="equal" if budget_eq else "at_most",
        max_turnover=turnover,
        max_gross=gross,
    )
    if seed % 3 == 2:
        coef = np.zeros(n)
        coef[:2] = 1.0
        rows.append((coef, -math.inf, 0.5))
        kwargs.update(groups={"g": names[:2]}, group_max={"g": 0.5})
    return names, S, mu, gamma, w0, kappa, eta, lo, hi, rows, turnover, gross, kwargs


@pytest.mark.parametrize(
    "n, seed",
    [(3, s) for s in range(16)] + [(4, s) for s in range(6)] + [(5, 1), (5, 3), (5, 7)],
)
def test_small_problems_match_a_brute_force_over_every_state(n, seed):
    """On random problems of three to five assets -- long-only and
    long-short, linear and quadratic costs, budgets equal and at most, a
    group cap, turnover and gross caps -- the weights agree with the brute
    force over every combination of states to 1e-9, certified; and when the
    brute force finds no feasible point, the problem is refused."""
    names, S, mu, gamma, w0, kappa, eta, lo, hi, rows, turnover, gross, kwargs = (
        _small_problem(seed, n)
    )
    best, weights = _brute_force(
        mu, S, gamma, w0, kappa, eta, lo, hi, rows, turnover, gross
    )
    if weights is None:
        with pytest.raises(ValidationError):
            optimize_rebalance(_frame(names, S), **kwargs)
        return
    out = optimize_rebalance(_frame(names, S), **kwargs)
    assert out["certified"], out["warnings"]
    assert np.max(np.abs(_w(out, names) - weights)) < 1e-9
    assert out["utility"] == pytest.approx(best, abs=1e-12)


# ── the caps bind exactly, and their prices are derivatives ──────────────


def _held_book(seed=11, n=14, cap=0.2):
    names, cov, mu, _, rng = _universe(n, seed=seed)
    w0 = _book(rng, n, cap)
    base = dict(
        expected_returns=mu,
        risk_aversion=4.0,
        current_weights=w0,
        linear_cost=0.002,
        quadratic_cost=0.01,
        upper_bounds=cap,
        groups={"g": names[:5]},
        group_max={"g": 0.35},
    )
    return names, cov, base


def test_a_binding_turnover_cap_holds_with_equality_and_is_priced():
    """The turnover cap binds: sum |w - w0| equals it to 1e-12, and its
    shadow price -- what one more unit of turnover is worth, net of what it
    costs to trade -- is the central difference of the optimal utility in
    the cap; so are the budget's and the group's."""
    names, cov, base = _held_book()
    frame = _frame(names, cov)

    def at(**over):
        out = optimize_rebalance(frame, **{**base, "max_turnover": 0.25, **over})
        assert out["certified"]
        return out

    out = at()
    assert abs(out["turnover"] - 0.25) <= 1e-12 * 0.25
    prices = _prices(out)
    assert prices[("max_turnover", None)] > 0.0
    assert prices[("max_turnover", None)] == pytest.approx(
        _fd(lambda t: at(max_turnover=t)["utility"], 0.25), rel=1e-6, abs=1e-9
    )
    assert prices[("budget", None)] == pytest.approx(
        _fd(lambda b: at(budget=b)["utility"], 1.0), rel=1e-6, abs=1e-9
    )
    assert prices[("group_max", "g")] == pytest.approx(
        _fd(lambda g: at(group_max={"g": g})["utility"], 0.35), rel=1e-6, abs=1e-9
    )


def test_a_binding_gross_cap_holds_with_equality_and_is_priced():
    """A long-short book under a gross cap: sum |w| equals the cap to
    1e-12 and its price is the derivative of the optimal utility in it."""
    names, cov, base = _held_book()
    long_short = dict(base, lower_bounds=-0.3, upper_bounds=0.4)
    frame = _frame(names, cov)

    def at(cap):
        out = optimize_rebalance(frame, **long_short, max_gross=cap)
        assert out["certified"]
        return out

    out = at(1.4)
    assert abs(out["gross"] - 1.4) <= 1e-12 * 1.4
    price = _prices(out)[("max_gross", None)]
    assert price > 0.0
    assert price == pytest.approx(
        _fd(lambda g: at(g)["utility"], 1.4), rel=1e-6, abs=1e-9
    )


def test_a_binding_volatility_cap_holds_with_equality_and_is_priced():
    """The volatility cap binds: the volatility equals it to 1e-12, the
    raised risk aversion is gamma + price / cap (the price is 2 nu cap with
    nu the multiplier per unit of variance), and the price is the
    derivative of the optimal utility in the cap."""
    names, cov, base = _held_book()
    frame = _frame(names, cov)
    free = optimize_rebalance(frame, **base)
    # the lowest volatility inside the bounds and the group cap: the floor
    floor = optimize_portfolio(
        frame,
        upper_bounds=base["upper_bounds"],
        groups=base["groups"],
        group_max=base["group_max"],
    )["volatility"]
    cap = 0.5 * (free["volatility"] + floor)

    def at(v):
        out = optimize_rebalance(frame, **base, max_volatility=v)
        assert out["certified"]
        return out

    out = at(cap)
    assert out["volatility"] < free["volatility"]
    assert abs(out["volatility"] - cap) <= 1e-12 * cap
    price = _prices(out)[("max_volatility", None)]
    assert out["effective_risk_aversion"] == pytest.approx(4.0 + price / cap, rel=1e-12)
    assert price == pytest.approx(
        _fd(lambda v: at(v)["utility"], cap), rel=1e-6, abs=1e-9
    )


def test_volatility_and_turnover_caps_bind_together():
    """Both caps at once: each holds with equality and each price is the
    derivative of the optimal utility in its own bound."""
    names, cov, base = _held_book()
    frame = _frame(names, cov)

    def at(v=0.21, t=0.7):
        out = optimize_rebalance(frame, **base, max_volatility=v, max_turnover=t)
        assert out["certified"]
        return out

    out = at()
    prices = _prices(out)
    assert abs(out["volatility"] - 0.21) <= 1e-12 * 0.21
    assert abs(out["turnover"] - 0.7) <= 1e-12 * 0.7
    assert prices[("max_volatility", None)] == pytest.approx(
        _fd(lambda v: at(v=v)["utility"], 0.21), rel=1e-6, abs=1e-9
    )
    assert prices[("max_turnover", None)] == pytest.approx(
        _fd(lambda t: at(t=t)["utility"], 0.7), rel=1e-6, abs=1e-9
    )


def test_the_utility_is_the_objective_at_the_weights():
    """utility, linear_cost, quadratic_cost, turnover, gross and net are
    the stated quantities recomputed here from the weights."""
    names, cov, base = _held_book()
    out = optimize_rebalance(_frame(names, cov), **base, max_turnover=0.25)
    w = _w(out, names)
    w0 = base["current_weights"]
    mu = base["expected_returns"]
    d = w - w0
    assert out["turnover"] == pytest.approx(np.sum(np.abs(d)), rel=1e-14)
    assert out["linear_cost"] == pytest.approx(0.002 * np.sum(np.abs(d)), rel=1e-12)
    assert out["quadratic_cost"] == pytest.approx(0.01 * np.sum(d * d), rel=1e-12)
    assert out["utility"] == pytest.approx(
        _utility(
            w, mu, cov, 4.0, w0, np.full(len(names), 0.002), np.full(len(names), 0.01)
        ),
        rel=1e-12,
    )
    assert out["gross"] == pytest.approx(np.sum(np.abs(w)), rel=1e-14)
    assert out["net"] == pytest.approx(1.0, abs=1e-13)
    for a, v, v0 in zip(names, w, w0):
        assert out["trades"][a] == v - v0


# ── a book that breaks its limits ────────────────────────────────────────


def test_a_book_over_its_caps_pays_the_least_turnover_that_fixes_it():
    """Two weights above a new cap of 0.1 by 0.03 and 0.02: any answer
    must sell 0.05 of them and buy 0.05 elsewhere, so required_turnover is
    0.1 exactly -- reported, warned about, and paid."""
    names, cov, mu, _, _ = _universe(10, seed=61)
    w0 = np.full(10, 0.1)
    w0[0], w0[1] = 0.13, 0.12
    w0[2:] = (1.0 - 0.25) / 8
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        linear_cost=0.001,
        upper_bounds=0.1,
    )
    assert out["certified"]
    assert out["required_turnover"] == pytest.approx(0.1, abs=1e-12)
    assert out["turnover"] >= 0.1 - 1e-12
    assert any("break the constraints" in m for m in out["warnings"])


# ── a book that starts on its binding gross cap ──────────────────────────

#: Five names, dollar-neutral, bounds of 0.3 either way, 15 bps to trade, a
#: gross cap of 1 and a book held exactly on it: the state a sequence of
#: rebalances is in on every date the cap binds, since each answer is the
#: next date's book.
_FIVE_S = np.array(
    [
        [
            3.6288434729318873e-04,
            2.3233416156655411e-04,
            1.7417476873626356e-04,
            1.7041521318484452e-04,
            2.1807329121934580e-05,
        ],
        [
            2.3233416156655411e-04,
            1.2841770519261009e-03,
            1.0778770482161710e-04,
            2.2689697434267138e-04,
            -2.7024897554874124e-04,
        ],
        [
            1.7417476873626356e-04,
            1.0778770482161710e-04,
            3.1321608057799090e-04,
            -1.6091829585984359e-05,
            4.7808300363207207e-05,
        ],
        [
            1.7041521318484452e-04,
            2.2689697434267138e-04,
            -1.6091829585984359e-05,
            1.1719049392109749e-03,
            -3.2008054996403702e-04,
        ],
        [
            2.1807329121934580e-05,
            -2.7024897554874124e-04,
            4.7808300363207207e-05,
            -3.2008054996403702e-04,
            6.8187333156496661e-04,
        ],
    ]
)
_FIVE_ALPHA = np.array(
    [
        0.0043896931488302815,
        0.009795403716599739,
        -0.005441134150640868,
        0.012313517525412897,
        0.01621804461688138,
    ]
)
_FIVE_HELD = np.array([0.25, 0.25, -1 / 6, -1 / 6, -1 / 6])


def _on_the_gross_cap(**over):
    kwargs = dict(
        assets=list("ABCDE"),
        expected_returns=_FIVE_ALPHA,
        current_weights=_FIVE_HELD,
        risk_aversion=2.0,
        linear_cost=0.0015,
        lower_bounds=-0.3,
        upper_bounds=0.3,
        budget=0.0,
        max_gross=1.0,
    )
    kwargs.update(over)
    return optimize_rebalance(_FIVE_S, **kwargs)


def test_a_book_held_on_its_binding_gross_cap_is_solved_exactly():
    """The five-name book on its gross cap with a cost: a step the budget
    fixes to the last bit once moved the cap by rounding, the cap bound on
    one free weight -- parallel to the budget there -- and the next system
    was singular, so no certified answer came back. It certifies, matches
    the brute force over every state, and a warm start from its own answer
    takes one pass."""
    out = _on_the_gross_cap()
    assert out["certified"], out["warnings"]
    n = 5
    best, weights = _brute_force(
        _FIVE_ALPHA,
        _FIVE_S,
        2.0,
        _FIVE_HELD,
        np.full(n, 0.0015),
        np.zeros(n),
        np.full(n, -0.3),
        np.full(n, 0.3),
        [(np.ones(n), 0.0, 0.0)],
        None,
        1.0,
    )
    assert np.max(np.abs(_w(out, list("ABCDE")) - weights)) < 1e-12
    assert out["utility"] == pytest.approx(best, abs=1e-15)
    assert abs(out["gross"] - 1.0) <= 1e-12
    again = _on_the_gross_cap(start=out)
    assert again["certified"] and again["passes"] == 1
    assert again["weights"] == out["weights"]


def test_a_dollar_neutral_book_on_its_gross_cap_certifies_on_every_date():
    """Fifty names, dollar-neutral, a gross cap of 1 and 15 bps, each date
    starting from the last date's answer -- on the cap whenever it bound.
    Every date certifies, cold and warm, and the two agree."""
    rng = np.random.default_rng(3)
    n = 50
    L = rng.normal(0.0, 1.0, (n, 3))
    S = L @ np.diag([0.0004, 0.0002, 0.0001]) @ L.T + np.diag(
        rng.uniform(0.0002, 0.0012, n)
    )
    names = [f"N{i:02d}" for i in range(n)]
    frame = _frame(names, S)
    held = np.zeros(n)
    previous = None
    on_the_cap = 0
    for _ in range(12):
        kwargs = dict(
            expected_returns=rng.normal(0.0, 0.01, n),
            current_weights=held,
            risk_aversion=2.0,
            linear_cost=0.0015,
            lower_bounds=-0.3,
            upper_bounds=0.3,
            budget=0.0,
            max_gross=1.0,
        )
        on_the_cap += abs(np.sum(np.abs(held)) - 1.0) <= 1e-12
        cold = optimize_rebalance(frame, **kwargs)
        warm = optimize_rebalance(frame, **kwargs, start=previous)
        assert cold["certified"], cold["warnings"]
        assert warm["certified"], warm["warnings"]
        assert np.max(np.abs(_w(cold, names) - _w(warm, names))) < 1e-12
        previous = warm
        held = _w(warm, names)
    assert on_the_cap >= 8


@pytest.mark.parametrize("seed", range(3))
def test_a_small_book_rebalanced_on_its_gross_cap_matches_the_brute_force(seed):
    """Four names rebalanced date after date, each answer the next book:
    every date matches the brute force over every state, gross cap and all."""
    rng = np.random.default_rng(500 + seed)
    n = 4
    A = rng.normal(size=(n, n))
    S = (A @ A.T) / n * 0.001 + np.diag(rng.uniform(0.0002, 0.001, n))
    names = list("abcd")
    held = np.zeros(n)
    kappa = np.full(n, 0.0015)
    for _ in range(4):
        alpha = rng.normal(0.0, 0.01, n)
        out = optimize_rebalance(
            _frame(names, S),
            expected_returns=alpha,
            current_weights=held,
            risk_aversion=2.0,
            linear_cost=kappa,
            lower_bounds=-0.4,
            upper_bounds=0.4,
            budget=0.0,
            max_gross=1.0,
        )
        best, weights = _brute_force(
            alpha,
            S,
            2.0,
            held,
            kappa,
            np.zeros(n),
            np.full(n, -0.4),
            np.full(n, 0.4),
            [(np.ones(n), 0.0, 0.0)],
            None,
            1.0,
        )
        assert out["certified"], out["warnings"]
        assert np.max(np.abs(_w(out, names) - weights)) < 1e-9
        held = _w(out, names)


def test_a_no_trade_answer_on_a_binding_gross_cap_is_the_current_weights():
    """Forecasts planted so that the held book is optimal only because the
    gross cap binds: each name's forecast leans psi toward its own sign, so
    without the cap every name would grow, and with it -- the cap at the
    book's own gross, psi its price -- every gradient is inside its band.
    The answer is the held book to the bit; without the cap it trades."""
    names, cov, _, _, rng = _universe(10, seed=91)
    held = rng.uniform(0.05, 0.2, 10) * np.where(np.arange(10) % 2 == 0, 1.0, -1.0)
    held = held - held.mean()  # dollar-neutral
    gamma, kappa, psi = 3.0, 0.002, 0.01
    delta = rng.uniform(-0.8, 0.8, 10) * kappa
    mu = gamma * cov @ held + 0.01 + psi * np.sign(held) + delta
    common = dict(
        expected_returns=mu,
        current_weights=held,
        risk_aversion=gamma,
        linear_cost=kappa,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        budget=0.0,
    )
    out = optimize_rebalance(
        _frame(names, cov), **common, max_gross=float(np.sum(np.abs(held)))
    )
    assert out["certified"], out["warnings"]
    assert np.array_equal(_w(out, names), held)
    assert out["turnover"] == 0.0 and sorted(out["no_trade"]) == sorted(names)
    free = optimize_rebalance(_frame(names, cov), **common)
    assert free["turnover"] > 0.0 and free["gross"] > np.sum(np.abs(held))


# ── a held book on its limits to rounding ────────────────────────────────


def _drifted_book(miss, n=12, seed=97):
    """A long-only book whose first name is at its lower bound 0 and whose
    weights sum to one plus `miss` -- what prices make of a fully invested
    book: on the budget to rounding, not exactly."""
    names, cov, _, _, rng = _universe(n, seed=seed)
    w0 = _book(rng, n, 0.25)
    w0[0] = 0.0
    w0 = w0 / w0.sum()
    w0[3] += miss
    return names, cov, w0


@pytest.mark.parametrize("cap", [None, 0.1])
@pytest.mark.parametrize("miss", [1e-13, -1e-13])
def test_a_book_off_its_budget_by_rounding_trades_nothing_as_held(miss, cap):
    """Forecasts planted so that holding is optimal, from a book that
    misses its budget by 1e-13 either way: the weight freed to carry the
    budget once sat at its lower bound and could not give up the 1e-13, so
    the next system was singular. It is the book on its budget: certified,
    every weight the weight held to the bit, nothing traded, and the
    budget's residual -- the book's own -- reported, not hidden in a name."""
    names, cov, w0 = _drifted_book(miss)
    gamma, kappa = 4.0, 0.0015
    delta = np.random.default_rng(5).uniform(-0.8, 0.8, len(names)) * kappa
    mu = gamma * cov @ w0 + 0.05 + delta
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=gamma,
        linear_cost=kappa,
        upper_bounds=0.3,
        max_turnover=cap,
    )
    assert out["certified"], out["warnings"]
    assert np.array_equal(_w(out, names), w0)
    assert out["turnover"] == 0.0 and out["required_turnover"] == 0.0
    assert any("meet the budget only to" in m for m in out["warnings"])


@pytest.mark.parametrize("cap", [None, 0.1])
@pytest.mark.parametrize("miss", [1e-13, -1e-13])
def test_a_book_off_its_budget_by_rounding_trades_as_the_book_on_it(miss, cap):
    """The same drifted book with forecasts that make some names trade: it
    certifies, the traded names take the budget's residual so the weights
    sum to it to rounding of the solve, every name left alone is its
    weight held to the bit, and the answer is the on-budget book's to
    rounding."""
    names, cov, w0 = _drifted_book(miss)
    mu = np.random.default_rng(6).normal(0.06, 0.05, len(names))
    common = dict(
        expected_returns=mu,
        risk_aversion=4.0,
        linear_cost=0.0015,
        upper_bounds=0.3,
        max_turnover=cap,
    )
    out = optimize_rebalance(_frame(names, cov), current_weights=w0, **common)
    on_budget = optimize_rebalance(
        _frame(names, cov), current_weights=w0 / w0.sum(), **common
    )
    assert out["certified"], out["warnings"]
    w = _w(out, names)
    assert len(out["no_trade"]) < len(names)
    assert abs(w.sum() - 1.0) < 1e-14
    for a, v, v0 in zip(names, w, w0):
        if a in out["no_trade"]:
            assert v == v0
    assert np.max(np.abs(w - _w(on_budget, names))) < 1e-12


def test_a_book_that_really_misses_its_budget_is_solved_from_the_nearest_point():
    """Off the budget by 1e-3 is a book that breaks it, not rounding: the
    least trade that restores it is 1e-3, reported as required_turnover
    and paid, and the answer certifies."""
    names, cov, w0 = _drifted_book(-1e-3)
    mu = np.random.default_rng(6).normal(0.06, 0.05, len(names))
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=4.0,
        linear_cost=0.0015,
        upper_bounds=0.3,
        max_turnover=0.1,
    )
    assert out["certified"], out["warnings"]
    assert out["required_turnover"] == pytest.approx(1e-3, rel=1e-9)
    assert out["turnover"] >= 1e-3 - 1e-12
    assert abs(sum(out["weights"].values()) - 1.0) < 1e-14
    assert any("break the constraints" in m for m in out["warnings"])


def test_a_weight_an_ulp_past_its_cap_is_the_weight_at_its_cap():
    """A capped name drifted an ulp past its cap is on the cap to
    rounding: the solve starts from the book, the clip is the only trade
    it is charged (a rounding-size required_turnover), and nothing is
    reported as breaking the limits."""
    names, cov, w0 = _drifted_book(0.0)
    w0 = np.minimum(w0, 0.2)
    w0[1] = 0.2
    w0[2] = 1.0 - (w0.sum() - w0[2])
    w0[1] = np.nextafter(0.2, 1.0)
    gamma, kappa = 4.0, 0.0015
    mu = gamma * cov @ w0 + 0.05
    out = optimize_rebalance(
        _frame(names, cov),
        expected_returns=mu,
        current_weights=w0,
        risk_aversion=gamma,
        linear_cost=kappa,
        upper_bounds=0.2,
    )
    assert out["certified"], out["warnings"]
    assert out["weights"][names[1]] == 0.2
    assert out["required_turnover"] <= 1e-15
    assert not any("break the constraints" in m for m in out["warnings"])


# ── warm starts and determinism ──────────────────────────────────────────


def test_a_warm_start_changes_the_passes_not_the_answer():
    """Started from the answer to the same problem the solve takes fewer
    passes and returns the same weights; started from the previous date's
    answer of a drifted book it returns the cold answer to 1e-12."""
    names, cov, base = _held_book(n=40, cap=0.08)
    base = dict(base, max_turnover=0.2)
    frame = _frame(names, cov)
    cold = optimize_rebalance(frame, **base)
    again = optimize_rebalance(frame, **base, start=cold)
    assert again["passes"] < cold["passes"]
    assert np.array_equal(_w(again, names), _w(cold, names))
    states_only = optimize_rebalance(frame, **base, start=cold["active_set"])
    assert np.max(np.abs(_w(states_only, names) - _w(cold, names))) < 1e-12

    rng = np.random.default_rng(3)
    drifted = _w(cold, names) * (1.0 + rng.normal(0.0, 0.01, len(names)))
    later = dict(
        base,
        current_weights=drifted / drifted.sum(),
        expected_returns=base["expected_returns"] + rng.normal(0, 0.005, len(names)),
    )
    cold_later = optimize_rebalance(frame, **later)
    warm_later = optimize_rebalance(frame, **later, start=cold)
    assert cold_later["certified"] and warm_later["certified"]
    assert np.max(np.abs(_w(warm_later, names) - _w(cold_later, names))) < 1e-12
    assert set(cold["active_set"].values()) <= set(REBALANCE_STATES)


def test_the_same_problem_gives_the_same_bits():
    """Deterministic: two solves of one problem agree in every bit of every
    number, cold and warm."""
    names, cov, base = _held_book()
    base = dict(base, max_turnover=0.7, max_volatility=0.21)
    a = optimize_rebalance(_frame(names, cov), **base)
    b = optimize_rebalance(_frame(names, cov), **base)
    assert a == b
    c = optimize_rebalance(_frame(names, cov), **base, start=a)
    d = optimize_rebalance(_frame(names, cov), **base, start=a)
    assert c == d


# ── the certificate's kink check ─────────────────────────────────────────


@pytest.mark.parametrize("kappa, certified", [(3.0, True), (1.0, False)])
def test_a_weight_held_at_its_kink_certifies_only_inside_its_band(kappa, certified):
    """minimize x^2/2 - 2x + kappa |x| with x held at its kink 0: the
    gradient there is -2, so holding is optimal exactly when 2 <= kappa.
    The certificate's interval check says so: kappa = 3 certifies, kappa =
    1 fails on dual feasibility by the gap 2 - 1 over the terms' size,
    |c| + kappa = 3."""
    prog = go._Programme(
        np.eye(1), np.array([-2.0]), [], np.array([-10.0]), np.array([10.0]), ["x"]
    )
    kinks = go._Kinks(
        held=np.array([True]),
        point=np.array([0.0]),
        width=np.array([kappa]),
        row_width=np.zeros((1, 0)),
    )
    x = np.zeros(1)
    certificate, _ = go._certify(
        prog, x, np.array([-2.0]), np.array([2.0]), kinks=kinks
    )
    assert certificate.certified is certified
    if not certified:
        assert certificate.dual_feasibility == pytest.approx(1.0 / 3.0)


# ── refusals ─────────────────────────────────────────────────────────────


def _small(**over):
    names, cov, mu, _, _ = _universe(6, seed=71)
    kwargs = dict(expected_returns=mu, current_weights=np.full(6, 1 / 6))
    kwargs.update(over)
    return names, cov, kwargs


@pytest.mark.parametrize(
    "over, words",
    [
        (dict(linear_cost=-0.001), "linear_cost is negative"),
        (
            dict(quadratic_cost={"A00": -1.0, **{f"A0{i}": 0.0 for i in range(1, 6)}}),
            "quadratic_cost is negative",
        ),
        (dict(linear_cost={"A00": 0.001}), "a cost left out is not zero"),
        (
            dict(lower_bounds=0.5, upper_bounds=0.2),
            "lower bound is above the upper bound",
        ),
        (dict(max_turnover=0.0), "max_turnover=0 must be positive"),
        (dict(max_gross=-1.0), "max_gross=-1 must be positive"),
        (dict(risk_aversion=0.0), "risk_aversion=0 must be positive"),
        (dict(expected_returns=None), "expected_returns is required"),
        (dict(budget_type="most"), "budget_type='most'"),
        (dict(start={"A00": "parked"}), "start has state(s) ['parked']"),
    ],
)
def test_bad_input_is_refused_by_name(over, words):
    """Each refusal names the argument, the value and the fix."""
    names, cov, kwargs = _small(**over)
    with pytest.raises(ValidationError) as caught:
        optimize_rebalance(_frame(names, cov), **kwargs)
    assert words in str(caught.value)
    assert str(caught.value).startswith("optimize_rebalance:")


def test_a_turnover_cap_too_small_to_reach_the_limits_is_named():
    """A book 0.05 over a new cap needs a turnover of 0.1; a turnover cap
    of 0.05 cannot reach the limits, and the refusal names the turnover
    cap among the constraints that conflict."""
    names, cov, mu, _, _ = _universe(10, seed=61)
    w0 = np.full(10, 0.1)
    w0[0], w0[1] = 0.13, 0.12
    w0[2:] = (1.0 - 0.25) / 8
    with pytest.raises(ValidationError) as caught:
        optimize_rebalance(
            _frame(names, cov),
            expected_returns=mu,
            current_weights=w0,
            upper_bounds=0.1,
            max_turnover=0.05,
        )
    message = str(caught.value)
    assert "no portfolio satisfies these constraints together" in message
    assert "the turnover cap" in message


def test_a_gross_cap_the_budget_cannot_meet_is_named():
    """Weights that must sum to 1 have a gross weight of at least 1: a
    gross cap of 0.8 is refused naming the budget and the gross cap."""
    names, cov, kwargs = _small(lower_bounds=-1.0, max_gross=0.8)
    with pytest.raises(ValidationError) as caught:
        optimize_rebalance(_frame(names, cov), **kwargs)
    message = str(caught.value)
    assert "the gross cap" in message and "the budget" in message


def test_a_volatility_cap_below_the_floor_is_refused_with_the_floor():
    """A volatility cap below the lowest volatility the other constraints
    allow is refused naming that floor."""
    names, cov, kwargs = _small(max_volatility=0.01)
    with pytest.raises(ValidationError) as caught:
        optimize_rebalance(_frame(names, cov), **kwargs)
    assert "max_volatility=0.01 cannot be met" in str(caught.value)


def test_a_universe_larger_than_the_solver_is_sized_for():
    n = go.MAX_ASSETS + 1
    names = [f"x{i}" for i in range(n)]
    with pytest.raises(ValidationError, match="sized for at most"):
        optimize_rebalance(
            _frame(names, np.eye(n) * 0.04), expected_returns=np.zeros(n)
        )
