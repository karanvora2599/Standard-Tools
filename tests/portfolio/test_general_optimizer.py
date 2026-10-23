"""
`optimize_portfolio` against answers known without it.

Nothing here is a recorded output. Each answer is checked against one of:

- a CLOSED FORM: equality-constrained problems solved by one linear system
  (min variance under a budget and an exposure equality; max utility and
  max active utility under the budget alone, dollar-neutral included), the
  tangency portfolio, and a factor-neutral minimum variance on S = BFB' + D
  inverted through Woodbury rather than directly;
- WATER-FILLING: on a diagonal covariance every asset of a box-constrained
  minimum-variance portfolio sits at a bound or at one common level, found
  here by bisection -- and, with a binding group cap, the group fills to its
  cap at its own level;
- an INDEPENDENT KKT CHECK and WEAK DUALITY: the shadow prices the result
  reports are turned back into Lagrange multipliers, the Lagrangian's
  minimum over all weights is computed in closed form, and it must not
  exceed the primal objective and must meet it to 1e-10; scipy's SLSQP from
  many starts must never find a better portfolio;
- FINITE DIFFERENCES: a shadow price is the derivative of the optimum with
  respect to its bound, so moving the bound and solving again must move the
  objective by price times step.

Plus the refusals: an infeasible set of limits names the ones in conflict,
a cap below the lowest attainable risk names that floor, and a problem
larger than the solver is sized for is refused with its count.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import general_optimizer as go
from standard_quant_tools.portfolio.general_optimizer import optimize_portfolio

TOL = go.TOLERANCE


# ── data ─────────────────────────────────────────────────────────────────


def _universe(n: int, seed: int, factors: int = 3, obs: int = None):
    """A sample covariance of factor-driven returns (annualized), expected
    returns of both signs, and names."""
    rng = np.random.default_rng(seed)
    obs = obs or max(4 * n, 120)
    loadings = rng.normal(1.0, 0.5, (n, factors))
    returns = rng.normal(0, 0.01, (obs, factors)) @ loadings.T + rng.normal(
        0, 0.015, (obs, n)
    )
    cov = np.cov(returns, rowvar=False) * 252
    names = [f"A{i:02d}" for i in range(n)]
    mu = rng.normal(0.06, 0.05, n)
    return names, cov, mu, loadings


def _frame(names, cov):
    return pd.DataFrame(cov, index=names, columns=names)


def _w(result, names):
    return np.array([result["weights"][a] for a in names])


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / max(
        1.0, float(np.max(np.abs(b)))
    )


# ── closed forms ─────────────────────────────────────────────────────────


class TestEqualityOnlyClosedForms:
    def test_min_variance_under_a_budget_and_an_exposure_equality(self):
        """With no bound binding the problem is min w'Sw subject to A w = b,
        whose answer is S^-1 A'(A S^-1 A')^-1 b -- one linear system. Its
        optimal variance is b'(A S^-1 A')^-1 b, so the shadow prices of the
        budget and the exposure are 2 (A S^-1 A')^-1 b."""
        names, cov, _, loadings = _universe(12, seed=1)
        beta = loadings[:, 0]
        A = np.vstack([np.ones(12), beta])
        b = np.array([1.0, 0.8])
        Si = np.linalg.inv(cov)
        M = A @ Si @ A.T
        expected = Si @ A.T @ np.linalg.solve(M, b)
        prices = 2.0 * np.linalg.solve(M, b)

        out = optimize_portfolio(
            _frame(names, cov),
            lower_bounds=-math.inf,
            factor_loadings=pd.DataFrame({"beta": beta}, index=names),
            exposure_min={"beta": 0.8},
            exposure_max={"beta": 0.8},
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-11
        assert out["objective_value"] == pytest.approx(
            float(b @ np.linalg.solve(M, b)), rel=1e-11
        )
        found = {
            b_["constraint"]: b_["shadow_price"] for b_ in out["binding_constraints"]
        }
        assert found["budget"] == pytest.approx(prices[0], rel=1e-9)
        assert found["exposure_equal"] == pytest.approx(prices[1], rel=1e-9)
        assert out["exposures"]["beta"] == pytest.approx(0.8, abs=1e-13)

    def test_max_utility_under_the_budget_alone(self):
        """mu'w - (lambda/2) w'Sw subject to 1'w = 1 has the answer
        S^-1 (mu - eta 1) / lambda, eta set so the weights sum to one."""
        names, cov, mu, _ = _universe(10, seed=2)
        lam = 3.5
        Si = np.linalg.inv(cov)
        one = np.ones(10)
        eta = (one @ Si @ mu - lam) / (one @ Si @ one)
        expected = Si @ (mu - eta) / lam
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=lam,
            lower_bounds=-math.inf,
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-11
        utility = mu @ expected - 0.5 * lam * expected @ cov @ expected
        assert out["objective_value"] == pytest.approx(utility, rel=1e-11)
        # d U* / d B is the budget's multiplier, eta.
        (budget,) = [
            b for b in out["binding_constraints"] if b["constraint"] == "budget"
        ]
        assert budget["shadow_price"] == pytest.approx(eta, rel=1e-9)

    def test_dollar_neutral_max_utility(self):
        """A budget of 0: S^-1 (mu - eta 1) / lambda with 1'w = 0."""
        names, cov, mu, _ = _universe(9, seed=3)
        lam = 2.0
        Si = np.linalg.inv(cov)
        one = np.ones(9)
        eta = (one @ Si @ mu) / (one @ Si @ one)
        expected = Si @ (mu - eta) / lam
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=lam,
            budget=0.0,
            lower_bounds=-math.inf,
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-11
        assert abs(out["net_weight"]) < 1e-14

    def test_max_active_utility_is_the_benchmark_plus_an_active_bet(self):
        """alpha'(w - b) - (lambda/2)(w - b)'S(w - b) under the budget alone:
        w = b + S^-1 (alpha - eta 1) / lambda, eta making the active weights
        sum to zero (the benchmark already sums to the budget)."""
        names, cov, alpha, _ = _universe(11, seed=4)
        bench = np.random.default_rng(40).dirichlet(np.ones(11))
        lam = 6.0
        Si = np.linalg.inv(cov)
        one = np.ones(11)
        eta = (one @ Si @ alpha) / (one @ Si @ one)
        expected = bench + Si @ (alpha - eta) / lam
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=alpha,
            benchmark_weights=dict(zip(names, bench)),
            objective="max_active_utility",
            risk_aversion=lam,
            lower_bounds=-math.inf,
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-11
        active = expected - bench
        assert out["tracking_error"] == pytest.approx(
            math.sqrt(active @ cov @ active), rel=1e-11
        )
        for a, v in zip(names, active):
            assert out["active_weights"][a] == pytest.approx(v, abs=1e-12)

    def test_tracking_the_benchmark_itself_costs_nothing(self):
        """When the benchmark is feasible it is the minimum-tracking-error
        portfolio: tracking error zero, weights equal to it."""
        names, cov, _, _ = _universe(8, seed=5)
        bench = np.random.default_rng(50).dirichlet(np.ones(8))
        out = optimize_portfolio(
            _frame(names, cov),
            benchmark_weights=dict(zip(names, bench)),
            objective="min_tracking_error",
        )
        assert out["certified"]
        assert _rel(_w(out, names), bench) < 1e-13
        assert out["tracking_error"] < 1e-8

    def test_max_sharpe_is_the_tangency_portfolio_when_unconstrained(self):
        """With no bound binding, the maximum-Sharpe fully invested portfolio
        is S^-1 (mu - rf) normalized to sum to one, whenever that sum is
        positive."""
        names, cov, mu, _ = _universe(10, seed=6)
        mu = np.abs(mu) + 0.04
        rf = 0.01
        raw = np.linalg.solve(cov, mu - rf)
        assert raw.sum() > 0
        expected = raw / raw.sum()
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_sharpe",
            risk_free_rate=rf,
            lower_bounds=-math.inf,
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-10
        sharpe = math.sqrt((mu - rf) @ raw)
        assert out["objective_value"] == pytest.approx(sharpe, rel=1e-11)
        assert out["sharpe_ratio"] == pytest.approx(sharpe, rel=1e-11)

    def test_no_tangency_on_the_efficient_branch_is_refused(self):
        """When 1'S^-1(mu - rf) < 0 the normalized S^-1(mu - rf) is the
        MINIMUM-Sharpe portfolio on the inefficient branch, and the maximum
        is approached only by levering a long-short book without limit. The
        homogenised optimum then has no positive scale, and that is refused
        rather than normalized."""
        names, cov, mu, _ = _universe(10, seed=6)
        assert np.linalg.solve(cov, mu - 0.01).sum() < 0
        with pytest.raises(ValidationError, match="without limit"):
            optimize_portfolio(
                _frame(names, cov),
                expected_returns=mu,
                objective="max_sharpe",
                risk_free_rate=0.01,
                lower_bounds=-math.inf,
            )


class TestBudgetAndFixedWeights:
    def test_an_at_most_budget_that_does_not_bind_is_mertons_portfolio(self):
        """With the budget an upper limit the uninvested rest earns nothing,
        and when the utility optimum S^-1 mu / lambda sums to less than the
        budget no limit binds: it is the answer, and the budget is not
        among the binding constraints."""
        names, cov, mu, _ = _universe(8, seed=29)
        mu = np.abs(mu) * 0.3
        lam = 10.0
        expected = np.linalg.solve(cov, mu) / lam
        assert expected.sum() < 1.0
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=lam,
            lower_bounds=-math.inf,
            budget_type="at_most",
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-11
        assert out["binding_constraints"] == []

    def test_a_fixed_weight_is_held_and_priced(self):
        """Equal bounds hold an asset's weight exactly, and its price is the
        derivative of the optimum with respect to that weight."""
        names, cov, mu, _ = _universe(9, seed=30)
        base = dict(expected_returns=mu, objective="max_utility", risk_aversion=3.0)

        def at(value):
            return optimize_portfolio(
                _frame(names, cov),
                **base,
                lower_bounds={names[0]: value},
                upper_bounds={names[0]: value},
            )

        out = at(0.1)
        assert out["certified"]
        assert out["weights"][names[0]] == 0.1
        (price,) = [
            b["shadow_price"]
            for b in out["binding_constraints"]
            if b["constraint"] == "fixed_weight"
        ]
        h = 1e-6
        slope = (at(0.1 + h)["objective_value"] - at(0.1 - h)["objective_value"]) / (
            2 * h
        )
        assert price == pytest.approx(slope, rel=1e-6, abs=1e-9)


class TestFactorNeutrality:
    def test_a_factor_neutral_minimum_variance_matches_woodbury(self):
        """S = BFB' + D. Its inverse by Woodbury, D^-1 - D^-1 B (F^-1 +
        B'D^-1 B)^-1 B'D^-1, never forms S; the factor-neutral minimum
        variance S^-1 A'(A S^-1 A')^-1 (1, 0) computed with it must equal
        the optimizer's answer on the dense S."""
        rng = np.random.default_rng(7)
        n, k = 30, 3
        B = rng.normal(1.0, 0.5, (n, k))
        Fr = rng.normal(size=(200, k)) * 0.15
        F = np.cov(Fr, rowvar=False)
        D = rng.uniform(0.01, 0.09, n)
        cov = B @ F @ B.T + np.diag(D)
        Di = np.diag(1.0 / D)
        inner = np.linalg.inv(np.linalg.inv(F) + B.T @ Di @ B)
        Si = Di - Di @ B @ inner @ B.T @ Di
        A = np.vstack([np.ones(n), B[:, 0]])
        target = np.array([1.0, 0.0])
        expected = Si @ A.T @ np.linalg.solve(A @ Si @ A.T, target)
        names = [f"S{i}" for i in range(n)]
        out = optimize_portfolio(
            _frame(names, cov),
            lower_bounds=-math.inf,
            factor_loadings=pd.DataFrame(B, index=names, columns=["mkt", "f2", "f3"]),
            exposure_min={"mkt": 0.0},
            exposure_max={"mkt": 0.0},
        )
        assert out["certified"]
        assert _rel(_w(out, names), expected) < 1e-9
        assert abs(out["exposures"]["mkt"]) < 1e-13
        # The other two factors are reported, not constrained.
        assert set(out["exposures"]) == {"mkt", "f2", "f3"}


def _water_fill(var, lower, upper, total):
    """argmin sum var_i w_i^2 s.t. sum w = total, lower <= w <= upper, by
    bisection on the common level nu: w_i = clip(nu / var_i, lower, upper)."""
    lo, hi = -1e6, 1e6
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if np.clip(mid / var, lower, upper).sum() < total:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) / var, lower, upper)


class TestWaterFilling:
    def test_every_asset_is_at_a_bound_or_at_the_common_level(self):
        """On a diagonal covariance the KKT conditions say 2 s_i w_i = nu for
        a free asset: one level shared by every asset not at a bound."""
        rng = np.random.default_rng(8)
        var = rng.uniform(0.01, 0.2, 15)
        lower = rng.choice([0.0, 0.02], 15)
        upper = rng.choice([0.08, 0.12, 0.3], 15)
        names = [f"D{i}" for i in range(15)]
        expected = _water_fill(var, lower, upper, 1.0)
        out = optimize_portfolio(
            _frame(names, np.diag(var)),
            lower_bounds=dict(zip(names, lower)),
            upper_bounds=dict(zip(names, upper)),
        )
        w = _w(out, names)
        assert out["certified"]
        assert np.max(np.abs(w - expected)) < 1e-12
        free = (w > lower + 1e-12) & (w < upper - 1e-12)
        levels = var[free] * w[free]
        assert free.sum() >= 2
        assert np.ptp(levels) < 1e-13
        at = ~free
        assert np.all(
            np.isclose(w[at], lower[at], atol=0) | np.isclose(w[at], upper[at], atol=0)
        )

    def test_a_binding_group_cap_fills_each_side_at_its_own_level(self):
        """Nested: with the first group capped below what it would hold, the
        group fills to exactly its cap at one level and the rest fill to the
        remainder at another."""
        rng = np.random.default_rng(9)
        var = rng.uniform(0.01, 0.2, 12)
        var[:6] *= 0.2  # the group's assets are the quiet ones
        names = [f"G{i}" for i in range(12)]
        upper = np.full(12, 0.25)
        cap = 0.35
        inside = _water_fill(var[:6], np.zeros(6), upper[:6], cap)
        outside = _water_fill(var[6:], np.zeros(6), upper[6:], 1.0 - cap)
        out = optimize_portfolio(
            _frame(names, np.diag(var)),
            upper_bounds=0.25,
            groups={"quiet": names[:6]},
            group_max={"quiet": cap},
        )
        w = _w(out, names)
        assert out["certified"]
        assert np.max(np.abs(w - np.concatenate([inside, outside]))) < 1e-12
        assert out["group_totals"]["quiet"] == pytest.approx(cap, abs=1e-14)
        (group,) = [
            b for b in out["binding_constraints"] if b["constraint"] == "group_max"
        ]
        # The two levels differ by the group's multiplier: raising the cap
        # moves weight to the quieter side and lowers the variance.
        assert group["shadow_price"] < 0


# ── independent KKT check, weak duality, an outside oracle ───────────────


def _terms(objective, cov, mu, bench, lam):
    """(Q, c, const, sense) of the minimized F = 1/2 w'Qw + c'w + const,
    built here from the definitions, independently of the module."""
    n = cov.shape[0]
    if objective == "min_variance":
        return 2 * cov, np.zeros(n), 0.0, 1.0
    if objective == "min_tracking_error":
        return 2 * cov, -2 * cov @ bench, float(bench @ cov @ bench), 1.0
    if objective == "max_utility":
        return lam * cov, -mu, 0.0, -1.0
    sb = cov @ bench
    return lam * cov, -mu - lam * sb, float(mu @ bench + 0.5 * lam * bench @ sb), -1.0


def _constraint_rows(names, spec, result):
    """Each binding constraint as (row a, bound, multiplier mu, kind), with
    mu = -dF/dbound recovered from the reported shadow price, so that the
    Lagrangian is F + sum mu (a'w - bound)."""
    n = len(names)
    index = {a: i for i, a in enumerate(names)}
    sense = spec["sense"]
    out = []
    for b in result["binding_constraints"]:
        a = np.zeros(n)
        kind = b["constraint"]
        if kind == "budget":
            a[:] = 1.0
        elif kind.startswith("group"):
            for member in spec["groups"][b["name"]]:
                a[index[member]] = 1.0
        elif kind.startswith("exposure"):
            a = spec["loadings"][:, spec["factors"].index(b["name"])].copy()
        else:
            a[index[b["name"]]] = 1.0
        mu = -sense * b["shadow_price"]
        out.append((a, b["bound"], mu, kind))
    return out


def _dual_value(Q, c, const, rows):
    """min over all w of F(w) + sum mu (a'w - bound), in closed form."""
    q = c.copy()
    shift = const
    for a, bound, mu, _ in rows:
        q += mu * a
        shift -= mu * bound
    return float(-0.5 * q @ np.linalg.solve(Q, q) + shift)


def _random_problem(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 51))
    names, cov, mu, loadings = _universe(n, seed=1000 + seed)
    objective = [
        "min_variance",
        "max_utility",
        "min_tracking_error",
        "max_active_utility",
    ][seed % 4]
    bench = rng.dirichlet(np.ones(n))
    cap = float(rng.choice([2.5 / n, 4.0 / n, 0.4]))
    floor = float(rng.choice([0.0, -0.05]))
    groups = {"g1": names[: n // 3], "g2": names[n // 3 : 2 * n // 3]}
    spec = {
        "groups": groups,
        "loadings": loadings[:, :2],
        "factors": ["f1", "f2"],
    }
    kwargs = dict(
        expected_returns=mu,
        benchmark_weights=dict(zip(names, bench)),
        objective=objective,
        lower_bounds=floor,
        upper_bounds=cap,
        groups=groups,
        group_max={"g1": 0.3},
        group_min={"g2": 0.25},
        factor_loadings=pd.DataFrame(
            loadings[:, :2], index=names, columns=["f1", "f2"]
        ),
        exposure_min={"f1": 0.9},
        exposure_max={"f1": 1.1, "f2": 1.2},
    )
    if objective in ("max_utility", "max_active_utility"):
        kwargs["risk_aversion"] = float(rng.choice([1.0, 4.0, 20.0]))
    return names, cov, mu, bench, kwargs, spec


@pytest.mark.parametrize("seed", range(24))
def test_random_constrained_problems_satisfy_kkt_and_weak_duality(seed):
    """On random problems from 5 to 50 assets with bounds, groups, factor
    limits and a budget: the weights satisfy every constraint, the reported
    shadow prices are multipliers of the right sign, the Lagrangian's
    stationarity holds in this file's own arithmetic, and the dual value
    they give is at most the primal objective and within 1e-10 of it."""
    names, cov, mu, bench, kwargs, spec = _random_problem(seed)
    try:
        out = optimize_portfolio(_frame(names, cov), **kwargs)
    except ValidationError as exc:  # an infeasible draw is refused by name
        assert "cannot all hold" in str(exc)
        pytest.skip("this draw's limits are infeasible")
    assert out["certified"], out["warnings"]
    w = _w(out, names)
    lam = kwargs.get("risk_aversion", 1.0)
    Q, c, const, sense = _terms(kwargs["objective"], cov, mu, bench, lam)
    spec["sense"] = sense
    F = float(0.5 * w @ Q @ w + c @ w + const)
    assert sense * out["objective_value"] == pytest.approx(F, rel=1e-12, abs=1e-15)

    # primal feasibility, recomputed here
    assert abs(w.sum() - 1.0) < 1e-12
    assert w.min() >= kwargs["lower_bounds"] - 1e-13
    assert w.max() <= kwargs["upper_bounds"] + 1e-13
    g1 = sum(out["weights"][a] for a in spec["groups"]["g1"])
    g2 = sum(out["weights"][a] for a in spec["groups"]["g2"])
    assert g1 <= 0.3 + 1e-12 and g2 >= 0.25 - 1e-12
    f1, f2 = spec["loadings"].T @ w
    assert 0.9 - 1e-12 <= f1 <= 1.1 + 1e-12 and f2 <= 1.2 + 1e-12

    rows = _constraint_rows(names, spec, out)
    # dual feasibility: an upper limit's multiplier >= 0, a lower one's <= 0
    for _, _, mu_j, kind in rows:
        if kind in ("upper_bound", "group_max", "exposure_max"):
            assert mu_j >= -1e-12
        if kind in ("lower_bound", "group_min", "exposure_min"):
            assert mu_j <= 1e-12
    # stationarity: grad F + sum mu a = 0
    gradient = Q @ w + c + sum(mu_j * a for a, _, mu_j, _ in rows)
    scale = np.max(np.abs(Q) @ np.abs(w) + np.abs(c))
    assert np.max(np.abs(gradient)) / scale < 1e-10
    # weak duality, and no gap
    dual = _dual_value(Q, c, const, rows)
    assert F >= dual - 1e-10 * max(1.0, abs(F))
    assert F - dual <= 1e-10 * max(1.0, abs(F))


@pytest.mark.parametrize("seed", range(8))
def test_slsqp_from_many_starts_never_finds_a_better_portfolio(seed):
    """The problems are convex, so every local minimum SLSQP can stop at is
    the global one: from twelve random starts, the best SLSQP objective is
    never below the certified answer's beyond its own 1e-9 precision, and
    its weights agree to its precision."""
    from scipy.optimize import minimize

    names, cov, mu, bench, kwargs, spec = _random_problem(100 + seed)
    try:
        out = optimize_portfolio(_frame(names, cov), **kwargs)
    except ValidationError:
        pytest.skip("this draw's limits are infeasible")
    lam = kwargs.get("risk_aversion", 1.0)
    Q, c, const, _ = _terms(kwargs["objective"], cov, mu, bench, lam)
    n = len(names)
    G = np.zeros((2, n))
    index = {a: i for i, a in enumerate(names)}
    for a in spec["groups"]["g1"]:
        G[0, index[a]] = 1.0
    for a in spec["groups"]["g2"]:
        G[1, index[a]] = 1.0
    L = spec["loadings"]
    cons = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
        {"type": "ineq", "fun": lambda w: 0.3 - G[0] @ w, "jac": lambda w: -G[0]},
        {"type": "ineq", "fun": lambda w: G[1] @ w - 0.25, "jac": lambda w: G[1]},
        {"type": "ineq", "fun": lambda w: L[:, 0] @ w - 0.9, "jac": lambda w: L[:, 0]},
        {"type": "ineq", "fun": lambda w: 1.1 - L[:, 0] @ w, "jac": lambda w: -L[:, 0]},
        {"type": "ineq", "fun": lambda w: 1.2 - L[:, 1] @ w, "jac": lambda w: -L[:, 1]},
    ]
    bounds = [(kwargs["lower_bounds"], kwargs["upper_bounds"])] * n
    F = lambda w: float(0.5 * w @ Q @ w + c @ w + const)  # noqa: E731
    ours = F(_w(out, names))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(12):
        start = rng.dirichlet(np.ones(n))
        res = minimize(
            F,
            start,
            jac=lambda w: Q @ w + c,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-15},
        )
        feasible = all(
            (
                (con["fun"](res.x) >= -1e-9)
                if con["type"] == "ineq"
                else abs(con["fun"](res.x)) < 1e-9
            )
            for con in cons
        )
        if res.success and feasible:
            best = res if best is None or res.fun < best.fun else best
    if best is None:
        pytest.skip("SLSQP did not finish feasibly from any start")
    assert ours <= best.fun + 1e-9 * max(1.0, abs(ours))
    assert np.max(np.abs(best.x - _w(out, names))) < 1e-4


def test_cvxpy_agrees_when_it_is_installed():
    """An outside modelling layer, when present: cvxpy's answer to the same
    constrained problems matches to its solver's precision."""
    cp = pytest.importorskip("cvxpy")
    for seed in range(4):
        names, cov, mu, bench, kwargs, spec = _random_problem(200 + seed)
        try:
            out = optimize_portfolio(_frame(names, cov), **kwargs)
        except ValidationError:
            continue
        lam = kwargs.get("risk_aversion", 1.0)
        Q, c, const, _ = _terms(kwargs["objective"], cov, mu, bench, lam)
        n = len(names)
        w = cp.Variable(n)
        index = {a: i for i, a in enumerate(names)}
        g1 = [index[a] for a in spec["groups"]["g1"]]
        g2 = [index[a] for a in spec["groups"]["g2"]]
        L = spec["loadings"]
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(Q)) + c @ w),
            [
                cp.sum(w) == 1,
                w >= kwargs["lower_bounds"],
                w <= kwargs["upper_bounds"],
                cp.sum(w[g1]) <= 0.3,
                cp.sum(w[g2]) >= 0.25,
                L[:, 0] @ w >= 0.9,
                L[:, 0] @ w <= 1.1,
                L[:, 1] @ w <= 1.2,
            ],
        )
        problem.solve()
        assert np.max(np.abs(w.value - _w(out, names))) < 1e-5


# ── shadow prices are derivatives ────────────────────────────────────────


def _fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2 * h)


class TestShadowPricesAreDerivatives:
    def test_bounds_groups_and_budget_against_central_differences(self):
        """Each binding constraint's shadow price is d objective / d bound:
        moving that bound by +-h and solving again moves the objective by
        the price times h, to second order."""
        names, cov, mu, _ = _universe(14, seed=11)
        base = dict(
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=4.0,
            upper_bounds=0.2,
            groups={"g": names[:5]},
            group_max={"g": 0.3},
        )
        out = optimize_portfolio(_frame(names, cov), **base)
        assert out["certified"]
        prices = {
            (b["constraint"], b["name"]): b["shadow_price"]
            for b in out["binding_constraints"]
        }
        h = 1e-6

        def at_group(cap):
            return optimize_portfolio(
                _frame(names, cov), **{**base, "group_max": {"g": cap}}
            )["objective_value"]

        def at_budget(budget):
            return optimize_portfolio(_frame(names, cov), **{**base, "budget": budget})[
                "objective_value"
            ]

        assert prices[("group_max", "g")] == pytest.approx(
            _fd(at_group, 0.3, h), rel=1e-6, abs=1e-9
        )
        assert prices[("budget", None)] == pytest.approx(
            _fd(at_budget, 1.0, h), rel=1e-6, abs=1e-9
        )
        capped = [k for k in prices if k[0] == "upper_bound"]
        assert capped
        name = capped[0][1]

        def at_cap(cap):
            uppers = {a: 0.2 for a in names}
            uppers[name] = cap
            return optimize_portfolio(
                _frame(names, cov), **{**base, "upper_bounds": uppers}
            )["objective_value"]

        assert prices[capped[0]] == pytest.approx(
            _fd(at_cap, 0.2, h), rel=1e-6, abs=1e-9
        )

    def test_max_sharpe_prices_are_sharpe_per_unit(self):
        """For max_sharpe the prices come from the homogenised programme's
        multipliers, scaled into Sharpe-ratio units; they must still be the
        derivatives of the maximum Sharpe ratio. (At a vertex where the caps
        alone fix the weights the derivative has two sides; see below. This
        problem is not such a vertex.)"""
        names, cov, mu, _ = _universe(10, seed=12)
        base = dict(
            expected_returns=mu,
            objective="max_sharpe",
            upper_bounds=0.3,
            groups={"g": names[:4]},
            group_max={"g": 0.2},
        )
        out = optimize_portfolio(_frame(names, cov), **base)
        assert out["certified"]
        prices = {
            (b["constraint"], b["name"]): b["shadow_price"]
            for b in out["binding_constraints"]
        }
        h = 1e-6

        def at_group(cap):
            return optimize_portfolio(
                _frame(names, cov), **{**base, "group_max": {"g": cap}}
            )["objective_value"]

        assert prices[("group_max", "g")] == pytest.approx(
            _fd(at_group, 0.2, h), rel=1e-5, abs=1e-8
        )
        assert prices[("budget", None)] == pytest.approx(
            _fd(
                lambda b: optimize_portfolio(
                    _frame(names, cov), **{**base, "budget": b}
                )["objective_value"],
                1.0,
                h,
            ),
            rel=1e-5,
            abs=1e-8,
        )
        capped = [k for k in prices if k[0] == "upper_bound"]
        assert capped
        name = capped[0][1]

        def at_cap(cap):
            uppers = {a: 0.3 for a in names}
            uppers[name] = cap
            return optimize_portfolio(
                _frame(names, cov), **{**base, "upper_bounds": uppers}
            )["objective_value"]

        assert prices[capped[0]] == pytest.approx(
            _fd(at_cap, 0.3, h), rel=1e-5, abs=1e-8
        )
        assert out["objective_value"] == pytest.approx(out["sharpe_ratio"], rel=1e-12)

    def test_at_a_vertex_the_price_lies_between_the_two_one_sided_rates(self):
        """Four caps of 0.25 holding the whole budget fix the weights, so
        lowering a cap and raising it change the Sharpe ratio at different
        rates, and more than one set of multipliers satisfies the KKT
        conditions. The price reported is one of them: it lies between the
        rate for raising the cap and the rate for lowering it, and quoting it
        for a move in either direction is right only to that bracket."""
        names, cov, mu, _ = _universe(10, seed=12)
        base = dict(expected_returns=mu, objective="max_sharpe", upper_bounds=0.25)
        out = optimize_portfolio(_frame(names, cov), **base)
        assert out["certified"]
        held = [a for a in names if out["weights"][a] == 0.25]
        assert len(held) == 4
        name = held[0]
        (price,) = [
            b["shadow_price"]
            for b in out["binding_constraints"]
            if b["constraint"] == "upper_bound" and b["name"] == name
        ]

        def at_cap(cap):
            uppers = {a: 0.25 for a in names}
            uppers[name] = cap
            return optimize_portfolio(
                _frame(names, cov), **{**base, "upper_bounds": uppers}
            )["objective_value"]

        s0 = out["objective_value"]
        h = 1e-7
        right = (at_cap(0.25 + h) - s0) / h
        left = (s0 - at_cap(0.25 - h)) / h
        assert abs(left - right) > 1e-3
        assert min(left, right) - 1e-6 <= price <= max(left, right) + 1e-6


# ── caps ─────────────────────────────────────────────────────────────────


class TestCaps:
    def test_a_binding_volatility_cap_is_met_and_is_a_higher_risk_aversion(self):
        """The cap holds to 1e-9 when it binds; its multiplier is the rise in
        risk aversion that gives the same answer without the cap; and its
        shadow price is the derivative of the utility with respect to it."""
        names, cov, mu, _ = _universe(12, seed=13)
        free = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=1.0,
            upper_bounds=0.4,
        )
        cap = 0.85 * free["volatility"]
        args = dict(
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=1.0,
            upper_bounds=0.4,
        )
        out = optimize_portfolio(_frame(names, cov), max_volatility=cap, **args)
        assert out["certified"]
        assert abs(out["volatility"] - cap) <= 1e-9 * cap
        lam = out["effective_risk_aversion"]
        assert lam > 1.0
        same = optimize_portfolio(_frame(names, cov), **{**args, "risk_aversion": lam})
        assert np.max(np.abs(_w(same, names) - _w(out, names))) < 1e-9
        (price,) = [
            b["shadow_price"]
            for b in out["binding_constraints"]
            if b["constraint"] == "max_volatility"
        ]
        h = 1e-6
        fn = lambda c: optimize_portfolio(_frame(names, cov), max_volatility=c, **args)[
            "objective_value"
        ]  # noqa: E731
        assert price == pytest.approx(_fd(fn, cap, h), rel=1e-5)
        assert price > 0

    def test_a_cap_that_does_not_bind_is_reported_as_such(self):
        names, cov, mu, _ = _universe(8, seed=14)
        free = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=2.0,
        )
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            objective="max_utility",
            risk_aversion=2.0,
            max_volatility=1.2 * free["volatility"],
        )
        assert out["certified"]
        assert _w(out, names).tobytes() == _w(free, names).tobytes()
        assert any("does not bind" in w for w in out["warnings"])
        assert not any(
            b["constraint"] == "max_volatility" for b in out["binding_constraints"]
        )
        assert out["effective_risk_aversion"] == 2.0

    def test_a_tracking_error_budget_binds_on_active_utility(self):
        names, cov, mu, _ = _universe(15, seed=15)
        bench = dict(zip(names, np.full(15, 1 / 15)))
        out = optimize_portfolio(
            _frame(names, cov),
            expected_returns=mu,
            benchmark_weights=bench,
            objective="max_active_utility",
            risk_aversion=0.5,
            max_tracking_error=0.02,
        )
        assert out["certified"]
        assert abs(out["tracking_error"] - 0.02) <= 1e-9 * 0.02
        assert out["effective_risk_aversion"] > 0.5

    def test_two_caps_bind_together(self):
        names, cov, mu, _ = _universe(12, seed=16)
        bench = dict(zip(names, np.full(12, 1 / 12)))
        args = dict(
            expected_returns=mu,
            benchmark_weights=bench,
            objective="max_utility",
            risk_aversion=0.5,
            upper_bounds=0.4,
        )
        free = optimize_portfolio(_frame(names, cov), **args)
        out = optimize_portfolio(
            _frame(names, cov),
            max_volatility=0.97 * free["volatility"],
            max_tracking_error=0.9 * free["tracking_error"],
            **args,
        )
        assert out["certified"]
        caps = {b["constraint"] for b in out["binding_constraints"]}
        if {"max_volatility", "max_tracking_error"} <= caps:
            assert abs(out["volatility"] - 0.97 * free["volatility"]) <= 1e-9
            assert abs(out["tracking_error"] - 0.9 * free["tracking_error"]) <= 1e-9
        assert out["volatility"] <= 0.97 * free["volatility"] * (1 + 1e-12)
        assert out["tracking_error"] <= 0.9 * free["tracking_error"] * (1 + 1e-12)

    def test_a_cap_below_the_lowest_attainable_risk_is_refused_with_the_floor(self):
        names, cov, mu, _ = _universe(8, seed=17)
        floor = optimize_portfolio(_frame(names, cov))["volatility"]
        with pytest.raises(ValidationError) as exc:
            optimize_portfolio(
                _frame(names, cov),
                expected_returns=mu,
                objective="max_utility",
                max_volatility=0.9 * floor,
            )
        assert "max_volatility" in str(exc.value)
        assert f"{floor:.6g}" in str(exc.value)


# ── refusals ─────────────────────────────────────────────────────────────


class TestInfeasibleLimitsAreRefusedByName:
    def test_caps_that_cannot_hold_the_budget(self):
        names, cov, _, _ = _universe(6, seed=18)
        with pytest.raises(ValidationError) as exc:
            optimize_portfolio(_frame(names, cov), upper_bounds=0.1)
        message = str(exc.value)
        assert "cannot all hold" in message
        assert "the budget (weights sum to 1)" in message
        assert "A00 <= 0.1" in message
        assert "0.4" in message  # the least total violation, 1 - 6 x 0.1

    def test_group_minimums_that_overfill_the_budget(self):
        names, cov, _, _ = _universe(9, seed=19)
        with pytest.raises(ValidationError) as exc:
            optimize_portfolio(
                _frame(names, cov),
                groups={"a": names[:3], "b": names[3:6]},
                group_min={"a": 0.6, "b": 0.6},
            )
        message = str(exc.value)
        assert "group 'a' at least 0.6" in message
        assert "group 'b' at least 0.6" in message
        assert "the budget" in message

    def test_a_neutrality_long_only_cannot_reach(self):
        names, cov, _, _ = _universe(6, seed=20)
        loadings = pd.DataFrame({"beta": np.linspace(0.5, 1.5, 6)}, index=names)
        with pytest.raises(ValidationError) as exc:
            optimize_portfolio(
                _frame(names, cov),
                factor_loadings=loadings,
                exposure_min={"beta": 0.0},
                exposure_max={"beta": 0.0},
            )
        assert "exposure to 'beta' equal to 0" in str(exc.value)


class TestOtherRefusals:
    def test_a_universe_larger_than_the_solver_is_sized_for(self):
        n = go.MAX_ASSETS + 1
        names = [f"X{i}" for i in range(n)]
        with pytest.raises(ValidationError) as exc:
            optimize_portfolio(np.eye(n) * 0.04, assets=names)
        assert f"{n} assets" in str(exc.value)
        assert str(go.MAX_ASSETS) in str(exc.value)

    def test_a_singular_covariance(self):
        names = ["A", "B", "C"]
        X = np.random.default_rng(0).normal(size=(50, 2))
        X = np.column_stack([X, X[:, 0]])
        with pytest.raises(ValidationError, match="not positive definite"):
            optimize_portfolio(np.cov(X, rowvar=False), assets=names)

    def test_max_sharpe_needs_a_positive_full_budget(self):
        names, cov, mu, _ = _universe(6, seed=21)
        for kwargs in ({"budget": 0.0}, {"budget_type": "at_most"}):
            with pytest.raises(ValidationError, match="positive budget held"):
                optimize_portfolio(
                    _frame(names, cov),
                    expected_returns=mu,
                    objective="max_sharpe",
                    **kwargs,
                )

    def test_max_sharpe_with_no_positive_excess_return(self):
        names, cov, _, _ = _universe(6, seed=22)
        with pytest.raises(
            ValidationError, match="no portfolio inside the constraints has a positive"
        ):
            optimize_portfolio(
                _frame(names, cov),
                expected_returns=np.full(6, 0.01),
                risk_free_rate=0.05,
                objective="max_sharpe",
            )

    def test_objectives_name_the_input_they_need(self):
        names, cov, mu, _ = _universe(5, seed=23)
        with pytest.raises(ValidationError, match="needs expected_returns"):
            optimize_portfolio(_frame(names, cov), objective="max_utility")
        with pytest.raises(ValidationError, match="needs benchmark_weights"):
            optimize_portfolio(_frame(names, cov), objective="min_tracking_error")
        with pytest.raises(ValidationError, match="needs benchmark_weights"):
            optimize_portfolio(_frame(names, cov), max_tracking_error=0.05)

    def test_an_equality_that_restates_the_budget(self):
        """Every asset loads 1 on 'market', so holding that exposure at 1
        repeats the budget: the rows are dependent, the systems singular,
        and the price undetermined. Refused by name; at 0.9 the same rows
        are inconsistent, and refused as infeasible instead."""
        names, cov, _, _ = _universe(6, seed=28)
        market = pd.DataFrame({"market": np.ones(6)}, index=names)
        with pytest.raises(ValidationError, match="is implied by the budget"):
            optimize_portfolio(
                _frame(names, cov),
                factor_loadings=market,
                exposure_min={"market": 1.0},
                exposure_max={"market": 1.0},
            )
        with pytest.raises(ValidationError, match="cannot all hold"):
            optimize_portfolio(
                _frame(names, cov),
                factor_loadings=market,
                exposure_min={"market": 0.9},
                exposure_max={"market": 0.9},
            )

    def test_a_missing_forecast_is_not_zero(self):
        names, cov, mu, _ = _universe(5, seed=24)
        with pytest.raises(ValidationError, match="no value for"):
            optimize_portfolio(
                _frame(names, cov),
                expected_returns={names[0]: 0.1},
                objective="max_utility",
            )


# ── the certificate is computed, not trusted ─────────────────────────────


class TestTheCertificate:
    def test_a_perturbed_answer_does_not_certify(self):
        """The same certificate, given the optimum moved along the budget's
        null space by 1e-6, reports stationarity far over tolerance: it is
        computed from the weights, so it notices."""
        names, cov, mu, _ = _universe(8, seed=25)
        out = optimize_portfolio(_frame(names, cov), lower_bounds=-1.0)
        w = _w(out, names)
        prog = go._Programme(
            2 * cov,
            np.zeros(8),
            [go._Row("budget", None, np.ones(8), 1.0, 1.0)],
            np.full(8, -1.0),
            np.full(8, np.inf),
            names,
        )
        good, _ = go._certify(prog, w, 2 * cov @ w, 2 * np.abs(cov) @ np.abs(w))
        assert good.certified
        nudged = w.copy()
        nudged[0] += 1e-6
        nudged[1] -= 1e-6
        bad, _ = go._certify(
            prog, nudged, 2 * cov @ nudged, 2 * np.abs(cov) @ np.abs(nudged)
        )
        assert not bad.certified
        assert bad.stationarity > 1e-8

    def test_an_answer_that_fails_its_certificate_is_reported_uncertified(
        self, monkeypatch
    ):
        """Planted: every stage's answer is judged against a tolerance of
        zero. The result keeps the weights, says certified False, and names
        the failed conditions -- a solver's success is never the evidence."""
        names, cov, mu, _ = _universe(8, seed=26)
        real = go._certify

        def strict(*args, **kwargs):
            certificate, duals = real(*args, **kwargs)
            return (
                go.Certificate(
                    certificate.stationarity + 1.0,
                    certificate.dual_feasibility,
                    certificate.primal_feasibility,
                    certificate.complementary_slackness,
                ),
                duals,
            )

        monkeypatch.setattr(go, "_certify", strict)
        out = optimize_portfolio(_frame(names, cov))
        assert out["certified"] is False
        assert out["weights"]
        assert any(
            "NOT a certified optimum" in w or "does not certify" in w
            for w in out["warnings"]
        )
        assert out["solver"].startswith("NOT certified")


# ── determinism ──────────────────────────────────────────────────────────


def test_the_same_problem_gives_the_same_bits():
    """Every answer is the KKT solve on its final active set, and the path
    to that set is fixed by the inputs: two calls give the same bits, the
    capped search included. (The same at one, two and four BLAS threads is
    held in tests/test_blas_limit.py.)"""
    names, cov, mu, loadings = _universe(40, seed=27)
    args = dict(
        expected_returns=mu,
        objective="max_utility",
        risk_aversion=1.0,
        upper_bounds=0.08,
        groups={"g": names[:10]},
        group_max={"g": 0.2},
        factor_loadings=pd.DataFrame(loadings, index=names, columns=["f0", "f1", "f2"]),
        exposure_min={"f0": 1.0},
    )
    free = optimize_portfolio(_frame(names, cov), **args)
    args["max_volatility"] = 0.95 * free["volatility"]
    first = optimize_portfolio(_frame(names, cov), **args)
    second = optimize_portfolio(_frame(names, cov), **args)
    assert first["certified"]
    assert _w(first, names).tobytes() == _w(second, names).tobytes()
    assert first["binding_constraints"] == second["binding_constraints"]
    assert first["certificate"] == second["certificate"]
