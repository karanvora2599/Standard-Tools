"""
One constrained optimizer over expected returns and a covariance the caller
supplies, solved exactly and certified rather than trusted.

`mean_variance_optimize` estimates its own moments from fetched prices and
takes one box and a fully-invested budget. A mandate is rarely stated that
way: it names its own return forecasts and risk model, a benchmark, sector
limits, a factor it must be neutral to, a tracking-error budget. This takes
all of those, as numbers, and nothing else -- no tickers and no fetch -- so
the same input always gives the same answer.

THE OBJECTIVES, each a function of the weights w:

    min_variance          w'Sw
    max_utility           mu'w - (lambda/2) w'Sw
    max_sharpe            (mu - rf)'w / sqrt(w'Sw)
    min_tracking_error    (w - b)'S(w - b)
    max_active_utility    mu'(w - b) - (lambda/2)(w - b)'S(w - b)

under per-asset bounds, a budget (sum of weights equal to, or at most, a
value; 0 is dollar-neutral), group bounds (sums over named subsets) and
factor-exposure bounds (B'w inside [lo, hi]; lo = hi is neutrality), with
caps on volatility and tracking error.

ALL BUT THE RATIO AND THE CAPS ARE ONE QUADRATIC PROGRAMME, solved by the
exact active-set methods of `_active_set`. The fast primal-dual method
(`solve`) takes equality rows and a box, so each linear inequality enters
it as a SLACK ROW: a group's sum g'w becomes a variable s with g'w - s = 0,
and the group's bounds become s's box. On a fixed active set every row is
then an equality, and the answer is one dense linear solve, whose KKT
residuals come out at 1e-17 to 1e-14 in every test here. The method
guesses whole active sets at once, and with several slack rows it often
guesses one whose KKT system is singular -- measured, on every random
sector-and-factor problem tried -- so the primal
method follows: `solve_rows`, which moves one constraint at a time from a
feasible point and holds an inequality row in its KKT system only while it
binds. With slack rows every row is in every system, binding or not: the
homogenised max Sharpe with a cap on each of 500 assets took 26 s that way
and 1.3 s this way. Its ratio test ignores a move of rounding size
(`STEP_TOLERANCE`): sector groups that cover every asset fix the last
group's sum once the others are at their bounds, and fixing it on its
rounding-sized move made the next system singular in 6 of 6 trials at 500
assets.

THE FEASIBLE POINT AND THE INFEASIBILITY PROOF COME FROM ONE LINEAR
PROGRAMME (`scipy.optimize.linprog`, HiGHS): the least total violation of
the rows, with every bound kept. Zero means feasible, and its answer starts
the primal method. Anything else is refused, naming the constraints whose
dual prices are nonzero in that programme -- a Farkas certificate: the
weighted sum of exactly those constraints is a contradiction, so they are
the ones that cannot hold together.

MAX SHARPE IS HOMOGENISED. With y = w / tau and (mu - rf)'y = 1, the ratio
is 1 / sqrt(y'Sy), and every constraint becomes homogeneous in (y, tau):
the budget 1'y = B tau, a bound l <= w_i as l tau <= y_i, a group's limit
as g'y <= h tau. That is a convex programme in (y, tau) with tau >= 0, the
same solver applies, and w = y / tau. It needs a positive budget held in
full -- a dollar-neutral or partly invested book has the same Sharpe ratio
at every scale, so the ratio cannot choose one -- and some portfolio with a
positive excess return, which the same linear programme checks first. When
the optimum has tau = 0 the ratio is approached only by levering a
long-short book without limit, and no fully invested portfolio attains it;
that is refused rather than normalised onto the wrong branch.

A CAP IS A MULTIPLIER FOUND BY A SEARCH. The volatility cap w'Sw <= s^2 is
quadratic, so it is not a slack row. Its KKT conditions say the answer
solves the programme with nu w'Sw added to the objective for the cap's
multiplier nu >= 0, and that the cap holds with equality when nu > 0. For
max_utility that is the same programme at risk aversion lambda + 2 nu, so
the search is the one over the utility's lambda that hits the cap; for the
other objectives it is the same search over nu. The cap's value is
monotone in nu, so the root is bracketed on (1 - theta) f + theta v,
theta in [0, 1] -- theta = 1 is the lowest the capped quantity can be
inside the other constraints, which is computed first and refuses a cap
below it by name -- and found by Brent's method to the last bits. Two caps
nest the same search.

THE CERTIFICATE IS COMPUTED FROM THE ANSWER, NOT READ FROM THE SOLVER. From
the weights and the constraint data alone: the multipliers that best
explain the objective's gradient on the free coordinates, then
stationarity, the multipliers' signs, every constraint's violation, and
complementary slackness (each multiplier times its constraint's slack),
each against its tolerance. `certified` is True only when all four hold,
and a result that is not certified says which failed.

SHADOW PRICES are those multipliers in the objective's own units: each
binding constraint's `shadow_price` is the derivative of the reported
objective value with respect to that constraint's bound (variance per unit
for min_variance, utility per unit for the utility objectives, Sharpe ratio
per unit for max_sharpe; per unit of volatility for a cap). At a degenerate
vertex -- limits that alone fix the weights, such as four caps of 0.25
holding the whole budget -- the multipliers are not unique, and a price is
one valid value: it lies between the rates for raising and for lowering its
bound.

SIZED FOR UP TO `MAX_ASSETS` ASSETS. Every pass is a dense solve; past a
few hundred assets the primal method's one bound per pass costs seconds,
and a larger problem is refused by name rather than left to run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_covariance
from standard_quant_tools.portfolio import _active_set

__all__ = [
    "MAX_ASSETS",
    "MAX_CONSTRAINT_ROWS",
    "OBJECTIVES",
    "TOLERANCE",
    "optimize_portfolio",
]

#: The largest universe solved. Every pass of the active-set methods is a
#: dense solve of the free variables and binding rows; at 500 assets with
#: sector and factor rows the primal method takes about a second.
MAX_ASSETS = 500

#: The most group and exposure rows taken, for the same reason.
MAX_CONSTRAINT_ROWS = 500

OBJECTIVES = (
    "min_variance",
    "max_utility",
    "max_sharpe",
    "min_tracking_error",
    "max_active_utility",
)

#: Each residual of a certified answer is at most this, relative to the
#: size of the terms it is a residual of (see `_active_set`).
TOLERANCE = _active_set.CERTIFICATE_TOLERANCE

#: A move toward the face minimizer smaller than this, relative to the
#: iterate, is rounding and does not block the primal method's step.
STEP_TOLERANCE = 64.0 * float(np.finfo(float).eps)

#: A coordinate or row within this of its bound, relative to its scale, is
#: treated as at the bound when no solver says otherwise.
_AT_BOUND = _active_set.AT_BOUND_TOLERANCE

#: The least total violation the feasibility programme may report and still
#: call the constraints consistent, relative to their scale. HiGHS holds
#: its own feasibility to 1e-7, so a genuinely feasible problem reports
#: zero or rounding; an infeasible one by at least the gap it cannot close.
_LP_FEASIBLE = 1e-9

_WHO = "optimize_portfolio"
_SHOWN = 8


# ── the programme ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Row:
    """One linear constraint lower <= coef'x <= upper, with its name.

    `kind` is what the caller asked for -- budget, group, exposure -- or,
    in the homogenised Sharpe programme, the per-asset bound a row stands
    for. lower == upper is an equality."""

    kind: str
    name: Optional[str]
    coef: np.ndarray
    lower: float
    upper: float

    @property
    def equality(self) -> bool:
        return self.lower == self.upper


@dataclass
class _Programme:
    """minimize 1/2 x'Qx + c'x subject to the rows and lower <= x <= upper."""

    Q: np.ndarray
    c: np.ndarray
    rows: List[_Row]
    lower: np.ndarray
    upper: np.ndarray
    #: What each variable is: an asset's name, or "tau".
    names: List[str]


class _Reduced:
    """
    A programme as `_active_set` takes it, in two forms.

    Variables fixed by equal bounds are substituted out; equality rows stay
    rows. For the primal-dual method (`_active_set.solve`, which takes only
    equality rows and a box) every inequality row is given a slack variable
    whose box is the row's bounds: the AUGMENTED form. For the primal
    method (`_active_set.solve_rows`) the inequality rows stay rows and
    enter its KKT system only while they bind: the NATIVE form.

    The structure depends only on the constraints, so one instance serves
    every objective a cap search tries; `system(Q, c)` gives each one's
    reduced objective.
    """

    def __init__(self, prog: _Programme) -> None:
        n = prog.lower.size
        self.n = n
        self.m = len(prog.rows)
        pinned = prog.lower == prog.upper
        self.pinned = np.flatnonzero(pinned)
        self.free = np.flatnonzero(~pinned)
        self.x_pinned = prog.lower[self.pinned].copy()
        self.eq_rows = [j for j, r in enumerate(prog.rows) if r.equality]
        self.ineq_rows = [j for j, r in enumerate(prog.rows) if not r.equality]
        k = self.free.size
        self.k = k
        m_eq = len(self.eq_rows)
        m_in = len(self.ineq_rows)

        def shift(j: int) -> float:
            return float(prog.rows[j].coef[self.pinned] @ self.x_pinned)

        # native form
        self.A = np.zeros((m_eq, k))
        self.b = np.zeros(m_eq)
        for i, j in enumerate(self.eq_rows):
            self.A[i] = prog.rows[j].coef[self.free]
            self.b[i] = prog.rows[j].lower - shift(j)
        self.G = np.zeros((m_in, k))
        self.g_lo = np.zeros(m_in)
        self.g_hi = np.zeros(m_in)
        for i, j in enumerate(self.ineq_rows):
            self.G[i] = prog.rows[j].coef[self.free]
            self.g_lo[i] = prog.rows[j].lower - shift(j)
            self.g_hi[i] = prog.rows[j].upper - shift(j)
        self.lo = prog.lower[self.free].copy()
        self.hi = prog.upper[self.free].copy()

        # augmented form: x then one slack per inequality row
        self.N = k + m_in
        self.A_aug = np.zeros((m_eq + m_in, self.N))
        self.A_aug[:m_eq, :k] = self.A
        self.A_aug[m_eq:, :k] = self.G
        self.A_aug[m_eq:, k:] = -np.eye(m_in)
        self.b_aug = np.concatenate([self.b, np.zeros(m_in)])
        self.lo_aug = np.concatenate([self.lo, self.g_lo])
        self.hi_aug = np.concatenate([self.hi, self.g_hi])

    def system(self, Q: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The objective on the free variables."""
        Q_f = Q[np.ix_(self.free, self.free)]
        c_f = c[self.free].copy()
        if self.pinned.size:
            c_f += Q[np.ix_(self.free, self.pinned)] @ self.x_pinned
        return Q_f, c_f

    def augment(
        self, Q_f: np.ndarray, c_f: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The same objective with a zero for each slack."""
        Q_aug = np.zeros((self.N, self.N))
        Q_aug[: self.k, : self.k] = Q_f
        c_aug = np.zeros(self.N)
        c_aug[: self.k] = c_f
        return Q_aug, c_aug

    def reduce(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x[self.free], self.lo, self.hi)

    def lift(self, x_f: np.ndarray) -> np.ndarray:
        """A point of the free variables as a point of the augmented form."""
        slacks = np.clip(self.G @ x_f, self.g_lo, self.g_hi)
        return np.concatenate([x_f, slacks])

    def expand(self, x_f: np.ndarray) -> np.ndarray:
        """The original variables from the free ones."""
        x = np.empty(self.n)
        x[self.free] = x_f
        x[self.pinned] = self.x_pinned
        return x

    def state_from_augmented(self, solution: _active_set.ActiveSetSolution) -> "_State":
        k = self.k
        lower_set = solution.lower_set
        upper_set = solution.upper_set
        if lower_set is None or upper_set is None:
            lower_set = solution.x <= self.lo_aug
            upper_set = (solution.x >= self.hi_aug) & ~lower_set
        return _State(
            x=solution.x[:k].copy(),
            at_lo=lower_set[:k].copy(),
            at_hi=upper_set[:k].copy(),
            row_lo=lower_set[k:].copy(),
            row_hi=upper_set[k:].copy(),
            passes=solution.passes,
        )

    def state_from_rows(self, solution: _active_set.ActiveSetSolution) -> "_State":
        assert solution.lower_set is not None and solution.upper_set is not None
        assert solution.row_lower_set is not None and solution.row_upper_set is not None
        return _State(
            x=solution.x.copy(),
            at_lo=solution.lower_set.copy(),
            at_hi=solution.upper_set.copy(),
            row_lo=solution.row_lower_set.copy(),
            row_hi=solution.row_upper_set.copy(),
            passes=solution.passes,
        )

    def active(
        self, state: "_State"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """A final active set in the original variables and rows: (variable
        at lower, at upper, row at lower, row at upper). Equality rows and
        pinned variables are always active and are not flagged."""
        at_lo = np.zeros(self.n, dtype=bool)
        at_hi = np.zeros(self.n, dtype=bool)
        at_lo[self.free] = state.at_lo
        at_hi[self.free] = state.at_hi
        row_lo = np.zeros(self.m, dtype=bool)
        row_hi = np.zeros(self.m, dtype=bool)
        for i, j in enumerate(self.ineq_rows):
            row_lo[j] = bool(state.row_lo[i])
            row_hi[j] = bool(state.row_hi[i])
        return at_lo, at_hi, row_lo, row_hi


@dataclass
class _State:
    """A solve's final point and active set on the free variables and the
    inequality rows, in the reduced programme's own indexing."""

    x: np.ndarray
    at_lo: np.ndarray
    at_hi: np.ndarray
    row_lo: np.ndarray
    row_hi: np.ndarray
    passes: int = 0


# ── the certificate ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Cap:
    """A quadratic cap (x - centre)'S(x - centre) <= limit^2, with its
    multiplier nu in the units of the objective per unit of variance."""

    kind: str
    S: np.ndarray
    centre: np.ndarray
    limit: float
    nu: float

    def value(self, x: np.ndarray) -> float:
        d = x - self.centre
        return float(d @ self.S @ d)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (self.S @ (x - self.centre))

    def gradient_scale(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.abs(self.S) @ np.abs(x - self.centre))


@dataclass(frozen=True)
class Certificate:
    """The KKT residuals of an answer, each relative to the size of the
    terms it is a residual of, and whether all of them are within
    `tolerance`."""

    stationarity: float
    dual_feasibility: float
    primal_feasibility: float
    complementary_slackness: float
    tolerance: float = TOLERANCE

    @property
    def worst(self) -> float:
        return max(
            self.stationarity,
            self.dual_feasibility,
            self.primal_feasibility,
            self.complementary_slackness,
        )

    @property
    def certified(self) -> bool:
        return bool(math.isfinite(self.worst) and self.worst <= self.tolerance)

    def failed(self) -> List[str]:
        """The conditions over tolerance, with their residuals."""
        return [
            f"{name} {value:.1e}"
            for name, value in self.as_dict().items()
            if name != "tolerance"
            and not (math.isfinite(value) and value <= self.tolerance)
        ]

    def as_dict(self) -> Dict[str, float]:
        return {
            "stationarity": float(self.stationarity),
            "dual_feasibility": float(self.dual_feasibility),
            "primal_feasibility": float(self.primal_feasibility),
            "complementary_slackness": float(self.complementary_slackness),
            "tolerance": float(self.tolerance),
        }


@dataclass
class _Duals:
    """Multipliers in the convention grad f + sum m_j a_j + caps = z, where
    z is each variable's bound multiplier: z >= 0 at a lower bound, <= 0 at
    an upper one, zero when free. A row's m is >= 0 at its upper bound and
    <= 0 at its lower one. Any sign is allowed on an equality row and on a
    variable fixed by equal bounds."""

    rows: np.ndarray
    box: np.ndarray
    at_lower: np.ndarray
    at_upper: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray


def _certify(
    prog: _Programme,
    x: np.ndarray,
    gradient: np.ndarray,
    gradient_scale: np.ndarray,
    active: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    caps: Sequence[_Cap] = (),
) -> Tuple[Certificate, _Duals]:
    """
    The KKT residuals of `x` for `prog` with the caps' multipliers given.

    `gradient` is the objective's gradient at x and `gradient_scale` the
    componentwise size of the terms it sums. `active` is a solver's final
    active set -- (variable at lower, at upper, row at lower, row at upper)
    -- and without it a coordinate or row within `_AT_BOUND` of a bound is
    taken as at it. Either way the row multipliers are the least-squares
    fit of the gradient on the free coordinates, computed here, and every
    residual is computed from `x` and the problem's own data. A wrong
    classification shows as a residual: a variable called fixed that is
    not at its bound has a multiplier times a nonzero slack.
    """
    n = x.size
    rows = prog.rows
    m = len(rows)
    lo, hi = prog.lower, prog.upper
    scale_x = max(1.0, float(np.max(np.abs(x))))
    pinned = lo == hi
    values = np.array([float(r.coef @ x) for r in rows]) if m else np.zeros(0)
    sizes = (
        np.array([float(np.abs(r.coef) @ np.abs(x)) for r in rows])
        if m
        else np.zeros(0)
    )
    if active is None:
        tol = _AT_BOUND * scale_x
        at_lo = (x <= lo + tol) & ~pinned
        at_hi = (x >= hi - tol) & ~at_lo & ~pinned
        row_lo = np.zeros(m, dtype=bool)
        row_hi = np.zeros(m, dtype=bool)
        for j, r in enumerate(rows):
            if r.equality:
                continue
            band = _AT_BOUND * max(1.0, sizes[j])
            row_lo[j] = values[j] <= r.lower + band
            row_hi[j] = (values[j] >= r.upper - band) and not row_lo[j]
    else:
        at_lo, at_hi, row_lo, row_hi = (np.asarray(a, dtype=bool) for a in active)
        at_lo = at_lo & ~pinned
        at_hi = at_hi & ~pinned & ~at_lo
    fixed = at_lo | at_hi | pinned
    free = ~fixed
    equal = np.array([r.equality for r in rows], dtype=bool)
    engaged = equal | row_lo | row_hi

    g = gradient.astype(float).copy()
    g_scale = gradient_scale.astype(float).copy()
    for cap in caps:
        if cap.nu != 0.0:
            g += cap.nu * cap.gradient(x)
            g_scale += abs(cap.nu) * cap.gradient_scale(x)

    multipliers = np.zeros(m)
    idx = np.flatnonzero(engaged)
    if idx.size and free.any():
        C = np.array([rows[j].coef for j in idx])
        fit = np.linalg.lstsq(C[:, free].T, -g[free], rcond=None)[0]
        multipliers[idx] = fit
    elif idx.size:
        # A vertex: every variable is at a bound, so nothing determines the
        # row multipliers. Zero is a valid choice when the bound
        # multipliers then carry the right signs, and the dual check below
        # says whether they do.
        multipliers[idx] = 0.0
    total = g.copy()
    for j in idx:
        total += multipliers[j] * rows[j].coef
    scale = g_scale.copy()
    for j in idx:
        scale += abs(multipliers[j]) * np.abs(rows[j].coef)
    scale_g = max(float(np.max(scale)) if n else 0.0, np.finfo(float).tiny)

    z = np.where(fixed, total, 0.0)
    stationarity = float(np.max(np.abs(total[free]), initial=0.0)) / scale_g

    dual = max(
        float(np.max(-z[at_lo], initial=0.0)),
        float(np.max(z[at_hi], initial=0.0)),
    )
    for j in range(m):
        if equal[j]:
            continue
        width = float(np.max(np.abs(rows[j].coef))) or 1.0
        if row_hi[j]:
            dual = max(dual, -multipliers[j] * width)
        elif row_lo[j]:
            dual = max(dual, multipliers[j] * width)
    for cap in caps:
        dual = max(dual, -cap.nu * float(np.max(cap.gradient_scale(x)) or 0.0))
    dual_feasibility = dual / scale_g

    primal = (
        max(
            float(np.max(lo - x, initial=0.0)),
            float(np.max(x - hi, initial=0.0)),
        )
        / scale_x
    )
    for j, r in enumerate(rows):
        excess = max(r.lower - values[j], values[j] - r.upper, 0.0)
        if excess > 0.0:
            bound = r.lower if values[j] < r.lower else r.upper
            denominator = max(sizes[j], abs(bound), np.finfo(float).tiny)
            primal = max(primal, excess / denominator)
    for cap in caps:
        over = cap.value(x) - cap.limit**2
        if over > 0.0:
            primal = max(primal, over / cap.limit**2)

    comp = 0.0
    with np.errstate(invalid="ignore"):
        gap_lo = np.where(at_lo, np.abs(z) * np.abs(x - lo), 0.0)
        gap_hi = np.where(at_hi, np.abs(z) * np.abs(hi - x), 0.0)
    comp = max(float(np.max(gap_lo, initial=0.0)), float(np.max(gap_hi, initial=0.0)))
    for j, r in enumerate(rows):
        if equal[j] or multipliers[j] == 0.0:
            continue
        bound = r.upper if row_hi[j] else r.lower
        comp = max(comp, abs(multipliers[j]) * abs(values[j] - bound))
    for cap in caps:
        if cap.nu != 0.0:
            comp = max(comp, cap.nu * abs(cap.limit**2 - cap.value(x)))
    complementarity = comp / (scale_g * scale_x)

    certificate = Certificate(
        stationarity=stationarity,
        dual_feasibility=dual_feasibility,
        primal_feasibility=primal,
        complementary_slackness=complementarity,
    )
    return certificate, _Duals(
        rows=multipliers,
        box=z,
        at_lower=at_lo,
        at_upper=at_hi,
        row_lower=row_lo,
        row_upper=row_hi,
    )


# ── solving one programme ────────────────────────────────────────────────


@dataclass
class _Solve:
    """One programme's answer: the point, its certificate and multipliers,
    how it was reached, and its final active set (to start the next
    programme of a search from)."""

    x: np.ndarray
    certificate: Certificate
    duals: _Duals
    method: str
    passes: int
    state: _State
    notes: List[str] = field(default_factory=list)


class _Unsolved(Exception):
    """Every exact stage stopped. `reason` says how each did, and
    `best` is the last uncertified answer, if any stage produced one."""

    def __init__(self, reason: str, best: Optional[_Solve] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.best = best


def _solve_programme(
    prog: _Programme,
    red: _Reduced,
    feasible: Optional[np.ndarray],
    warm: Optional[_Solve] = None,
    caps: Sequence[_Cap] = (),
) -> _Solve:
    """
    The certified optimum of `prog`, or `_Unsolved`.

    The stages, each accepted only when its answer certifies:
    1. the primal-dual method on the slack-row form, from `warm`'s active
       set when there is one (a nearby programme of a search) and from a
       cold start otherwise -- a handful of passes on a box however far
       the answer moved, but with inequality rows it often guesses a set
       whose KKT system is singular, and stops within a few passes;
    2. from `warm`, which is feasible here since only the objective
       differs: the primal method started on that answer's active set;
    3. the primal method from `feasible` (the feasibility programme's
       point), one constraint per pass.
    Each ends with the KKT solve on its final active set.
    """
    Q_f, c_f = red.system(prog.Q, prog.c)
    Stage = Tuple[str, Callable[[], _State]]
    stages: List[Stage] = []

    def rows_from(x_f: np.ndarray, state: Optional[_State]) -> _State:
        solution = _active_set.solve_rows(
            Q_f,
            c_f,
            red.A,
            red.b,
            red.G,
            red.g_lo,
            red.g_hi,
            red.lo,
            red.hi,
            x_f,
            working=None if state is None else (state.at_lo, state.at_hi),
            row_working=None if state is None else (state.row_lo, state.row_hi),
            step_tolerance=STEP_TOLERANCE,
        )
        return red.state_from_rows(solution)

    def primal_dual(start: Optional[np.ndarray]) -> _State:
        Q_aug, c_aug = red.augment(Q_f, c_f)
        solution = _active_set.solve(
            Q_aug,
            c_aug,
            red.A_aug,
            red.b_aug,
            red.lo_aug,
            red.hi_aug,
            start=start,
        )
        return red.state_from_augmented(solution)

    if warm is not None:
        previous = warm.state
        stages.append(
            (
                "the primal-dual active-set method from the previous solve's "
                "active set",
                lambda: primal_dual(red.lift(previous.x)),
            )
        )
        stages.append(
            (
                "the primal active-set method from the previous solve's active set",
                lambda: rows_from(previous.x, previous),
            )
        )
    else:
        stages.append(
            (
                "the primal-dual active-set method from a cold start",
                lambda: primal_dual(None),
            )
        )
    if feasible is not None:
        point = red.reduce(feasible)
        stages.append(
            (
                "the primal active-set method from the feasibility programme's point",
                lambda: rows_from(point, None),
            )
        )
    stopped: List[str] = []
    best: Optional[_Solve] = None
    passes = 0
    for how, run in stages:
        try:
            state = run()
        except (_active_set.ActiveSetFailure, np.linalg.LinAlgError) as exc:
            reason = getattr(exc, "reason", str(exc))
            passes += int(getattr(exc, "passes", 0))
            stopped.append(f"{how} stopped: {reason}")
            continue
        passes += state.passes
        x = red.expand(state.x)
        gradient = prog.Q @ x + prog.c
        scale = np.abs(prog.Q) @ np.abs(x) + np.abs(prog.c)
        certificate, duals = _certify(
            prog, x, gradient, scale, active=red.active(state), caps=caps
        )
        result = _Solve(x, certificate, duals, how, passes, state, list(stopped))
        if certificate.certified:
            return result
        best = result
        stopped.append(
            f"{how} ended on an answer that does not certify "
            f"({', '.join(certificate.failed())})"
        )
    raise _Unsolved("; ".join(stopped), best)


# ── the feasibility programme ────────────────────────────────────────────


def _describe_row(row: _Row, side: str) -> str:
    if row.kind == "budget":
        if row.equality:
            return f"the budget (weights sum to {row.lower:g})"
        return f"the budget (weights sum to at most {row.upper:g})"
    what = f"group {row.name!r}" if row.kind == "group" else f"exposure to {row.name!r}"
    if row.equality:
        return f"{what} equal to {row.lower:g}"
    if side == "lower":
        return f"{what} at least {row.lower:g}"
    return f"{what} at most {row.upper:g}"


def _feasible_point(
    names: Sequence[str],
    rows: Sequence[_Row],
    lower: np.ndarray,
    upper: np.ndarray,
    objective: Optional[np.ndarray] = None,
    objective_cap: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    A point satisfying every row and bound, from the least total violation
    of the rows with the bounds kept; or a refusal naming the constraints
    that cannot hold together.

    With `objective`, a second programme then maximizes objective'x over
    the feasible set with objective'x <= `objective_cap` (so it cannot be
    unbounded) and returns its value too: max_sharpe's check that some
    portfolio has a positive excess return.
    """
    from scipy.optimize import linprog

    n = len(names)
    m = len(rows)
    eq = [j for j, r in enumerate(rows) if r.equality]
    upper_rows = [
        j for j, r in enumerate(rows) if not r.equality and math.isfinite(r.upper)
    ]
    lower_rows = [
        j for j, r in enumerate(rows) if not r.equality and math.isfinite(r.lower)
    ]
    # variables: x (n), then per equality row two elastics (+, -), then one
    # per inequality side
    n_el = 2 * len(eq) + len(upper_rows) + len(lower_rows)
    cost = np.concatenate([np.zeros(n), np.ones(n_el)])
    A_eq = np.zeros((len(eq), n + n_el))
    b_eq = np.zeros(len(eq))
    col = n
    for i, j in enumerate(eq):
        A_eq[i, :n] = rows[j].coef
        A_eq[i, col] = 1.0
        A_eq[i, col + 1] = -1.0
        b_eq[i] = rows[j].lower
        col += 2
    A_ub = np.zeros((len(upper_rows) + len(lower_rows), n + n_el))
    b_ub = np.zeros(len(upper_rows) + len(lower_rows))
    for i, j in enumerate(upper_rows):
        A_ub[i, :n] = rows[j].coef
        A_ub[i, col] = -1.0
        b_ub[i] = rows[j].upper
        col += 1
    for i, j in enumerate(lower_rows):
        k = len(upper_rows) + i
        A_ub[k, :n] = -rows[j].coef
        A_ub[k, col] = -1.0
        b_ub[k] = -rows[j].lower
        col += 1
    bounds = [
        (
            None if not math.isfinite(lo) else float(lo),
            None if not math.isfinite(hi) else float(hi),
        )
        for lo, hi in zip(lower, upper)
    ] + [(0.0, None)] * n_el
    result = linprog(
        cost,
        A_ub=A_ub if A_ub.shape[0] else None,
        b_ub=b_ub if A_ub.shape[0] else None,
        A_eq=A_eq if A_eq.shape[0] else None,
        b_eq=b_eq if A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    if result.status != 0:
        raise ValidationError(
            f"{_WHO}: the feasibility programme did not finish ({result.message}). "
            "Check the bounds and limits for values far outside the scale of "
            "portfolio weights."
        )
    scale = 1.0 + max(
        [abs(r.lower) for r in rows if math.isfinite(r.lower)]
        + [abs(r.upper) for r in rows if math.isfinite(r.upper)]
        + [0.0]
    )
    if result.fun > _LP_FEASIBLE * scale:
        raise ValidationError(
            _infeasible_message(
                names, rows, lower, upper, result, eq, upper_rows, lower_rows
            )
        )
    x = np.clip(np.asarray(result.x[:n], dtype=float), lower, upper)
    if objective is None:
        return x, None

    A_ub2 = [r.coef for r in rows if not r.equality and math.isfinite(r.upper)]
    b_ub2 = [r.upper for r in rows if not r.equality and math.isfinite(r.upper)]
    A_ub2 += [-r.coef for r in rows if not r.equality and math.isfinite(r.lower)]
    b_ub2 += [-r.lower for r in rows if not r.equality and math.isfinite(r.lower)]
    A_ub2.append(objective)
    b_ub2.append(float(objective_cap if objective_cap is not None else 1.0))
    A_eq2 = [r.coef for r in rows if r.equality]
    b_eq2 = [r.lower for r in rows if r.equality]
    best = linprog(
        -objective,
        A_ub=np.array(A_ub2),
        b_ub=np.array(b_ub2),
        A_eq=np.array(A_eq2) if A_eq2 else None,
        b_eq=np.array(b_eq2) if b_eq2 else None,
        bounds=bounds[:n],
        method="highs",
    )
    if best.status != 0:
        raise ValidationError(
            f"{_WHO}: the programme for the highest excess return did not "
            f"finish ({best.message})."
        )
    return np.clip(np.asarray(best.x, dtype=float), lower, upper), float(-best.fun)


def _infeasible_message(
    names: Sequence[str],
    rows: Sequence[_Row],
    lower: np.ndarray,
    upper: np.ndarray,
    result: Any,
    eq: Sequence[int],
    upper_rows: Sequence[int],
    lower_rows: Sequence[int],
) -> str:
    """The refusal: the constraints with a nonzero price in the least-
    violation programme, which together are a proof that no portfolio
    satisfies them."""
    small = 1e-9
    involved: List[str] = []
    eq_marg = getattr(getattr(result, "eqlin", None), "marginals", None)
    ub_marg = getattr(getattr(result, "ineqlin", None), "marginals", None)
    lo_marg = getattr(getattr(result, "lower", None), "marginals", None)
    hi_marg = getattr(getattr(result, "upper", None), "marginals", None)
    if eq_marg is not None:
        for i, j in enumerate(eq):
            if abs(eq_marg[i]) > small:
                involved.append(_describe_row(rows[j], "equal"))
    if ub_marg is not None:
        for i, j in enumerate(upper_rows):
            if abs(ub_marg[i]) > small:
                involved.append(_describe_row(rows[j], "upper"))
        for i, j in enumerate(lower_rows):
            if abs(ub_marg[len(upper_rows) + i]) > small:
                involved.append(_describe_row(rows[j], "lower"))
    n = len(names)
    bound_lo = [
        f"{names[i]} >= {lower[i]:g}"
        for i in range(n)
        if lo_marg is not None and abs(lo_marg[i]) > small
    ]
    bound_hi = [
        f"{names[i]} <= {upper[i]:g}"
        for i in range(n)
        if hi_marg is not None and abs(hi_marg[i]) > small
    ]
    if bound_lo:
        involved.append(
            f"the lower bound(s) {', '.join(bound_lo[:_SHOWN])}"
            + (f" and {len(bound_lo) - _SHOWN} more" if len(bound_lo) > _SHOWN else "")
        )
    if bound_hi:
        involved.append(
            f"the upper bound(s) {', '.join(bound_hi[:_SHOWN])}"
            + (f" and {len(bound_hi) - _SHOWN} more" if len(bound_hi) > _SHOWN else "")
        )
    conflict = "; ".join(involved) if involved else "the constraints as a whole"
    return (
        f"{_WHO}: no portfolio satisfies these constraints together. They "
        f"cannot all hold: {conflict}. The closest any portfolio inside the "
        f"bounds comes is a total violation of {float(result.fun):.6g} (the "
        "least-violation programme's optimum; those are the constraints its "
        "dual prices it on). Loosen one of them -- or check the units: "
        "weights are fractions of the budget."
    )


# ── inputs ───────────────────────────────────────────────────────────────


def _named_covariance(
    covariance: Any, assets: Optional[Sequence[str]]
) -> Tuple[List[str], np.ndarray]:
    if isinstance(covariance, pd.DataFrame):
        names = [str(c) for c in covariance.columns]
        index = [str(i) for i in covariance.index]
        if index != names:
            raise ValidationError(
                f"{_WHO}: the covariance's rows and columns name different "
                f"assets or orders ({index[:_SHOWN]} against {names[:_SHOWN]}). "
                "A covariance is square over one list of assets."
            )
        if assets is not None:
            order = [str(a) for a in assets]
            if sorted(order) != sorted(names):
                raise ValidationError(
                    f"{_WHO}: assets {order[:_SHOWN]} must name exactly the "
                    f"covariance's assets {names[:_SHOWN]}."
                )
            covariance = covariance.loc[order, order]
            names = order
        matrix = covariance.to_numpy(dtype=float)
    else:
        if assets is None:
            raise ValidationError(
                f"{_WHO}: an unnamed covariance needs `assets` to name its "
                "rows; every weight in the answer is keyed by name."
            )
        names = [str(a) for a in assets]
        matrix = np.asarray(covariance, dtype=float)
    if len(set(names)) != len(names):
        repeated = sorted({a for a in names if names.count(a) > 1})
        raise ValidationError(
            f"{_WHO}: the assets repeat {repeated[:_SHOWN]}; each is one row "
            "and column of the covariance."
        )
    if matrix.ndim != 2 or matrix.shape != (len(names), len(names)):
        raise ValidationError(
            f"{_WHO}: the covariance has shape {matrix.shape} for "
            f"{len(names)} assets; it must be square over them."
        )
    return names, matrix


def _vector(
    values: Any,
    names: Sequence[str],
    field_name: str,
    missing: Optional[float],
) -> np.ndarray:
    """A per-asset vector from a scalar, a name-keyed map or a sequence in
    asset order. In a map, an unknown name is refused and a missing one
    takes `missing` -- or is refused when `missing` is None."""
    n = len(names)
    if isinstance(values, (int, float, np.floating, np.integer)) and not isinstance(
        values, bool
    ):
        out = np.full(n, float(values))
    elif isinstance(values, (Mapping, pd.Series)):
        mapping = dict(values.items())
        unknown = [str(k) for k in mapping if str(k) not in set(names)]
        if unknown:
            raise ValidationError(
                f"{_WHO}: {field_name} names {unknown[:_SHOWN]}, which are not "
                "assets of the covariance. A value for an asset that is not "
                "in the problem is a typo or a different universe."
            )
        lookup = {str(k): v for k, v in mapping.items()}
        absent = [a for a in names if a not in lookup]
        if absent and missing is None:
            raise ValidationError(
                f"{_WHO}: {field_name} has no value for {absent[:_SHOWN]}"
                + (f" and {len(absent) - _SHOWN} more" if len(absent) > _SHOWN else "")
                + ". Give every asset one; a missing forecast is not zero."
            )
        out = np.array([float(lookup.get(a, missing)) for a in names])
    else:
        out = np.asarray(values, dtype=float).reshape(-1)
        if out.size != n:
            raise ValidationError(
                f"{_WHO}: {field_name} has {out.size} values for {n} assets."
            )
    return out


def _finite(
    values: np.ndarray, names: Sequence[str], field_name: str, allow_inf: bool = False
) -> None:
    bad = [
        names[i]
        for i, v in enumerate(values)
        if math.isnan(v) or (not allow_inf and not math.isfinite(v))
    ]
    if bad:
        raise ValidationError(
            f"{_WHO}: {field_name} is not finite at {bad[:_SHOWN]}; every "
            "entry must be a finite number."
        )


def _scalar(
    value: Any, field_name: str, *, positive: bool = False, allow_zero: bool = True
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.floating, np.integer)
    ):
        raise ValidationError(f"{_WHO}: {field_name} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{_WHO}: {field_name} must be finite, got {value!r}.")
    if positive and (number < 0 or (number == 0 and not allow_zero)):
        raise ValidationError(
            f"{_WHO}: {field_name}={number:g} must be "
            f"{'positive' if not allow_zero else 'non-negative'}."
        )
    return number


def _loadings_matrix(
    factor_loadings: Any, names: Sequence[str]
) -> Tuple[List[str], np.ndarray]:
    """Assets x factors, every asset with a loading on every factor."""
    if isinstance(factor_loadings, pd.DataFrame):
        frame = factor_loadings.copy()
        frame.index = [str(i) for i in frame.index]
        frame.columns = [str(c) for c in frame.columns]
    else:
        frame = pd.DataFrame(
            {
                str(a): {str(f): v for f, v in dict(row).items()}
                for a, row in dict(factor_loadings).items()
            }
        ).T
    unknown = [a for a in frame.index if a not in set(names)]
    if unknown:
        raise ValidationError(
            f"{_WHO}: factor_loadings names {unknown[:_SHOWN]}, which are not "
            "assets of the covariance."
        )
    missing_assets = [a for a in names if a not in set(frame.index)]
    if missing_assets:
        raise ValidationError(
            f"{_WHO}: factor_loadings has no row for {missing_assets[:_SHOWN]}"
            + (
                f" and {len(missing_assets) - _SHOWN} more"
                if len(missing_assets) > _SHOWN
                else ""
            )
            + ". Every asset needs a loading on every factor (0.0 where it "
            "has none): an exposure summed over some assets is not the "
            "portfolio's."
        )
    frame = frame.loc[list(names)]
    gaps = frame.isna()
    if gaps.to_numpy().any():
        pairs = [f"{a}/{f}" for a in gaps.index for f in gaps.columns if gaps.at[a, f]]
        raise ValidationError(
            f"{_WHO}: {len(pairs)} asset/factor loading(s) are missing: "
            f"{pairs[:_SHOWN]}. Give every asset a loading on every factor "
            "(0.0 where it genuinely has none) -- absent is not zero."
        )
    values = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValidationError(f"{_WHO}: factor_loadings contains a non-finite loading.")
    return [str(c) for c in frame.columns], values


def _limit_map(
    values: Optional[Mapping[str, float]],
    known: Sequence[str],
    field_name: str,
    what: str,
) -> Dict[str, float]:
    if not values:
        return {}
    unknown = [str(k) for k in values if str(k) not in set(known)]
    if unknown:
        raise ValidationError(
            f"{_WHO}: {field_name} names {unknown[:_SHOWN]}, which are not "
            f"{what}. Known: {list(known)[:_SHOWN]}."
        )
    out = {}
    for k, v in values.items():
        out[str(k)] = _scalar(v, f"{field_name}[{k!r}]")
    return out


def _require_positive_definite(
    cov: np.ndarray, names: Sequence[str], free: np.ndarray
) -> Tuple[float, List[str]]:
    """The covariance over the assets that can move must be positive
    definite; its condition number, and a warning above 1e10."""
    sub = cov[np.ix_(free, free)]
    with single_threaded_blas():
        eigenvalues = np.linalg.eigvalsh(sub) if sub.size else np.array([1.0])
    largest = float(eigenvalues.max())
    smallest = float(eigenvalues.min())
    floor = largest * max(sub.shape[0], 1) * float(np.finfo(float).eps)
    if not largest > 0.0 or smallest <= floor:
        raise ValidationError(
            f"{_WHO}: the covariance of the {sub.shape[0]} assets free to move "
            f"is not positive definite (smallest eigenvalue {smallest:.3e}, "
            f"largest {largest:.3e}). Some combination of them has no "
            "variance, so the optimizer would find a riskless-looking "
            "portfolio in that direction and the answer is not unique. "
            "Shrink the estimate (estimate_covariance with "
            "method='ledoit_wolf'), drop a duplicated or collinear asset, or "
            "use more observations than assets."
        )
    condition = largest / smallest
    warnings: List[str] = []
    if condition > 1e10:
        warnings.append(
            f"The covariance is ill-conditioned (condition number "
            f"{condition:.3e}, above 1e10): two or more assets are nearly "
            "collinear, and the weights resolve a difference close to noise. "
            "The answer is still the exact optimum of the matrix given."
        )
    return condition, warnings


def _require_independent_equalities(rows: Sequence[_Row], free: np.ndarray) -> None:
    """
    Refuse an equality limit the others already imply.

    An exposure held at 1 on a factor every asset loads 1 on restates the
    budget; a group held at a value whose members are all fixed restates
    their bounds. Either makes every KKT system singular, and its price is
    not determined -- any split between it and the row it repeats is as
    good -- so it is named rather than solved around.
    """
    equalities = [r for r in rows if r.equality]
    if len(equalities) < 1:
        return
    kept: List[np.ndarray] = []
    with single_threaded_blas():
        for row in equalities:
            trial = np.array(kept + [row.coef[free]])
            if np.linalg.matrix_rank(trial) < trial.shape[0]:
                earlier = [
                    _describe_row(r, "equal") for r in equalities[: len(kept)]
                ] or ["the assets' fixed weights"]
                raise ValidationError(
                    f"{_WHO}: {_describe_row(row, 'equal')} is implied by "
                    f"{'; '.join(earlier)} on the assets free to move: the "
                    "rows are linearly dependent, so it adds no constraint "
                    "and its shadow price is not determined. Drop it, or "
                    "state it as a range (min below max)."
                )
            kept.append(row.coef[free])


# ── the objectives ───────────────────────────────────────────────────────


def _objective_terms(
    objective: str,
    cov: np.ndarray,
    mu: Optional[np.ndarray],
    bench: Optional[np.ndarray],
    risk_aversion: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """(Q, c, constant) with the minimized quantity F(w) = 1/2 w'Qw + c'w
    + constant: the variance, the tracking variance, or minus the utility."""
    n = cov.shape[0]
    if objective == "min_variance":
        return 2.0 * cov, np.zeros(n), 0.0
    if objective == "min_tracking_error":
        assert bench is not None
        sb = cov @ bench
        return 2.0 * cov, -2.0 * sb, float(bench @ sb)
    if objective == "max_utility":
        assert mu is not None
        return risk_aversion * cov, -mu, 0.0
    assert objective == "max_active_utility" and mu is not None and bench is not None
    sb = cov @ bench
    return (
        risk_aversion * cov,
        -mu - risk_aversion * sb,
        float(mu @ bench) + 0.5 * risk_aversion * float(bench @ sb),
    )


def _drop_redundant_bounds(
    lower: np.ndarray, upper: np.ndarray, budget: float, budget_type: str
) -> np.ndarray:
    """
    Upper bounds the budget and the other lower bounds already imply.

    With w_j >= l_j for every other asset and the weights summing to at
    most B, w_i <= B - sum_{j != i} l_j whatever the cap says. A cap at or
    above that never binds, and kept, it makes a portfolio held in one
    asset a vertex with every weight at a bound, where nothing determines
    the budget's multiplier and the KKT system on the free weights is
    empty. A long-only cap of 1 under a budget of 1 is the common case.
    """
    out = upper.copy()
    if not np.all(np.isfinite(lower)):
        return out
    total = float(np.sum(lower))
    implied = budget - (total - lower)
    with np.errstate(invalid="ignore"):
        redundant = np.isfinite(out) & (out >= implied) & (out > lower)
    out[redundant] = np.inf
    return out


# ── the cap search ───────────────────────────────────────────────────────


@dataclass
class _CapSpec:
    kind: str  # "max_volatility" | "max_tracking_error"
    S: np.ndarray
    centre: np.ndarray
    limit: float


def _brent(func: Callable[[float], float], a: float, b: float) -> float:
    """A root of a continuous non-increasing function bracketed by
    f(a) > 0 >= f(b), to the last bits: Brent's method (scipy's brentq)
    with its tolerances at the floor. `func` caches its solves, so the two
    ends brentq evaluates again cost nothing."""
    from scipy.optimize import brentq

    try:
        root = brentq(
            func,
            a,
            b,
            xtol=4.0 * float(np.finfo(float).tiny),
            rtol=4.0 * float(np.finfo(float).eps),
            maxiter=400,
        )
    except RuntimeError as exc:
        raise _Unsolved(f"the cap's multiplier search did not converge: {exc}")
    return float(root)


class _Search:
    """
    The programme with caps, solved by searching each binding cap's
    multiplier.

    For a mixing weight theta in [0, 1], the objective is
    (1 - theta) f + theta v, with v the capped quadratic: theta = 0 is the
    uncapped problem and theta = 1 the lowest v inside the linear
    constraints. Its minimizer moves continuously and v at it does not rise
    as theta does, so the theta at which v equals the cap's square is
    bracketed and Brent's method finds it. The cap's multiplier, in the
    units of f, is nu = theta / (1 - theta).
    """

    def __init__(
        self,
        base: _Programme,
        red: _Reduced,
        feasible: Optional[np.ndarray],
    ) -> None:
        self.base = base
        self.red = red
        self.feasible = feasible
        self.solves = 0
        self.last: Optional[_Solve] = None

    def solve(
        self, weights: Sequence[Tuple[_CapSpec, float]], f_weight: float
    ) -> _Solve:
        """The programme f_weight * f + sum w_k v_k (each v a cap's
        quadratic), certified."""
        Q = f_weight * self.base.Q
        c = f_weight * self.base.c
        for spec, w in weights:
            if w:
                Q = Q + w * 2.0 * spec.S
                c = c - w * 2.0 * (spec.S @ spec.centre)
        prog = _Programme(
            Q, c, self.base.rows, self.base.lower, self.base.upper, self.base.names
        )
        self.solves += 1
        result = _solve_programme(prog, self.red, self.feasible, warm=self.last)
        self.last = result
        return result


def _value(spec: _CapSpec, w: np.ndarray) -> float:
    d = w - spec.centre
    return float(d @ spec.S @ d)


# ── the entry point ──────────────────────────────────────────────────────


def optimize_portfolio(
    covariance: Any,
    *,
    assets: Optional[Sequence[str]] = None,
    expected_returns: Any = None,
    objective: str = "min_variance",
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
    benchmark_weights: Any = None,
    lower_bounds: Any = 0.0,
    upper_bounds: Any = None,
    budget: float = 1.0,
    budget_type: str = "equal",
    groups: Optional[Mapping[str, Sequence[str]]] = None,
    group_min: Optional[Mapping[str, float]] = None,
    group_max: Optional[Mapping[str, float]] = None,
    factor_loadings: Any = None,
    exposure_min: Optional[Mapping[str, float]] = None,
    exposure_max: Optional[Mapping[str, float]] = None,
    max_volatility: Optional[float] = None,
    max_tracking_error: Optional[float] = None,
) -> Dict[str, Any]:
    """
    The exact optimum of one of `OBJECTIVES` under linear constraints and
    volatility or tracking-error caps, with its KKT certificate and the
    shadow price of every binding constraint.

    `covariance` is a named DataFrame, or a matrix with `assets` naming its
    rows; `expected_returns` and `benchmark_weights` are name-keyed maps or
    sequences in asset order, in the covariance's units (both annual, as
    estimate_covariance returns it). A benchmark weight left out is zero.
    `lower_bounds` and `upper_bounds` are a scalar, a map (an asset left
    out keeps 0 below and no cap above) or a sequence; equal bounds fix an
    asset. `budget_type` is "equal" or "at_most"; under "at_most" the
    uninvested remainder earns nothing in the objective, so give expected
    returns in excess of cash. `groups` maps a group to its member assets,
    with `group_min` / `group_max` limits on its summed weight;
    `factor_loadings` (assets x factors, or asset -> {factor: loading})
    with `exposure_min` / `exposure_max` limits on B'w -- min = max holds it
    there, 0 is neutrality. `max_volatility` and `max_tracking_error` are
    caps in the covariance's units of volatility.

    Returns a dict: the weights and active weights, expected return,
    volatility, tracking error and Sharpe ratio, group totals and
    exposures, `binding_constraints` (each with its `shadow_price`, the
    derivative of `objective_value` with respect to its bound),
    `certified` and its `certificate`, and how it was solved. A refusal
    names the argument, the value and the fix; an infeasible set of
    constraints is refused naming the ones that conflict.
    """
    with single_threaded_blas():
        return _optimize(
            covariance,
            assets=assets,
            expected_returns=expected_returns,
            objective=objective,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
            benchmark_weights=benchmark_weights,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget=budget,
            budget_type=budget_type,
            groups=groups,
            group_min=group_min,
            group_max=group_max,
            factor_loadings=factor_loadings,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            max_volatility=max_volatility,
            max_tracking_error=max_tracking_error,
        )


def _optimize(
    covariance: Any,
    *,
    assets: Optional[Sequence[str]],
    expected_returns: Any,
    objective: str,
    risk_aversion: float,
    risk_free_rate: float,
    benchmark_weights: Any,
    lower_bounds: Any,
    upper_bounds: Any,
    budget: float,
    budget_type: str,
    groups: Optional[Mapping[str, Sequence[str]]],
    group_min: Optional[Mapping[str, float]],
    group_max: Optional[Mapping[str, float]],
    factor_loadings: Any,
    exposure_min: Optional[Mapping[str, float]],
    exposure_max: Optional[Mapping[str, float]],
    max_volatility: Optional[float],
    max_tracking_error: Optional[float],
) -> Dict[str, Any]:
    if objective not in OBJECTIVES:
        raise ValidationError(
            f"{_WHO}: objective={objective!r} is not one of {list(OBJECTIVES)}."
        )
    names, cov = _named_covariance(covariance, assets)
    n = len(names)
    if n > MAX_ASSETS:
        raise ValidationError(
            f"{_WHO}: {n} assets, and this optimizer is sized for at most "
            f"{MAX_ASSETS}. Each pass of its exact method is a dense linear "
            "solve over the free assets and binding constraints, and past a "
            "few hundred assets a solve takes seconds to minutes. Optimize a "
            f"universe of at most {MAX_ASSETS} (pre-select, or optimize "
            "sleeves and combine them)."
        )
    if n < 2:
        raise ValidationError(f"{_WHO}: {n} asset; an allocation needs at least two.")
    cov = require_finite_covariance(cov, "covariance", _WHO)
    cov = (cov + cov.T) / 2.0

    mu = None
    if expected_returns is not None:
        mu = _vector(expected_returns, names, "expected_returns", None)
        _finite(mu, names, "expected_returns")
    bench = None
    if benchmark_weights is not None:
        bench = _vector(benchmark_weights, names, "benchmark_weights", 0.0)
        _finite(bench, names, "benchmark_weights")
    lower = _vector(
        0.0 if lower_bounds is None else lower_bounds, names, "lower_bounds", 0.0
    )
    upper = _vector(
        math.inf if upper_bounds is None else upper_bounds,
        names,
        "upper_bounds",
        math.inf,
    )
    _finite(lower, names, "lower_bounds", allow_inf=True)
    _finite(upper, names, "upper_bounds", allow_inf=True)
    if np.any(lower == math.inf) or np.any(upper == -math.inf):
        raise ValidationError(
            f"{_WHO}: a lower bound of +inf or an upper bound of -inf admits no weight."
        )
    crossed = [names[i] for i in range(n) if lower[i] > upper[i]]
    if crossed:
        raise ValidationError(
            f"{_WHO}: the lower bound is above the upper bound for "
            f"{crossed[:_SHOWN]}; no weight lies between them."
        )
    risk_aversion = _scalar(risk_aversion, "risk_aversion")
    if objective in ("max_utility", "max_active_utility") and not risk_aversion > 0:
        raise ValidationError(
            f"{_WHO}: risk_aversion={risk_aversion:g} must be positive for "
            f"{objective}; at zero the objective is linear and its optimum a "
            "corner (or unbounded), which is not this programme. For the "
            "highest return under a volatility cap, give a small risk "
            "aversion with max_volatility: the cap's search raises it to "
            "whatever the cap needs."
        )
    risk_free_rate = _scalar(risk_free_rate, "risk_free_rate")
    budget = _scalar(budget, "budget")
    if budget_type not in ("equal", "at_most"):
        raise ValidationError(
            f"{_WHO}: budget_type={budget_type!r} must be 'equal' (the weights "
            "sum to the budget) or 'at_most' (they sum to at most it)."
        )
    if objective in ("max_utility", "max_sharpe", "max_active_utility") and mu is None:
        raise ValidationError(
            f"{_WHO}: objective={objective!r} needs expected_returns, one per "
            "asset in the covariance's units."
        )
    if objective in ("min_tracking_error", "max_active_utility") and bench is None:
        raise ValidationError(
            f"{_WHO}: objective={objective!r} needs benchmark_weights, the "
            "benchmark's weight in each asset (an asset left out is zero)."
        )
    caps: List[_CapSpec] = []
    if max_volatility is not None:
        limit = _scalar(max_volatility, "max_volatility")
        if not limit > 0:
            raise ValidationError(f"{_WHO}: max_volatility={limit:g} must be positive.")
        caps.append(_CapSpec("max_volatility", cov, np.zeros(n), limit))
    if max_tracking_error is not None:
        limit = _scalar(max_tracking_error, "max_tracking_error")
        if not limit > 0:
            raise ValidationError(
                f"{_WHO}: max_tracking_error={limit:g} must be positive."
            )
        if bench is None:
            raise ValidationError(
                f"{_WHO}: max_tracking_error needs benchmark_weights: tracking "
                "error is measured against them."
            )
        caps.append(_CapSpec("max_tracking_error", cov, bench, limit))
    if objective == "max_sharpe":
        if caps:
            raise ValidationError(
                f"{_WHO}: objective='max_sharpe' takes no max_volatility or "
                "max_tracking_error. Under a volatility cap that binds, the "
                "highest Sharpe ratio is the highest return at the cap: use "
                "objective='max_utility' with max_volatility, whose search "
                "finds it exactly; for a tracking-error budget use "
                "'max_active_utility' with max_tracking_error."
            )
        if budget_type != "equal" or not budget > 0:
            raise ValidationError(
                f"{_WHO}: objective='max_sharpe' needs a positive budget held "
                f"in full (budget_type='equal', budget > 0); got "
                f"budget_type={budget_type!r}, budget={budget:g}. A Sharpe "
                "ratio does not change when a book is scaled, so a "
                "dollar-neutral or partly invested budget leaves the size of "
                "the book undetermined. Use max_utility, whose risk aversion "
                "sets it, or a volatility cap."
            )

    # Rows: the budget, the groups, the exposures.
    rows: List[_Row] = []
    ones = np.ones(n)
    if budget_type == "equal":
        rows.append(_Row("budget", None, ones, budget, budget))
    else:
        rows.append(_Row("budget", None, ones, -math.inf, budget))
    group_names: List[str] = []
    group_rows: Dict[str, np.ndarray] = {}
    if groups:
        position = {a: i for i, a in enumerate(names)}
        for gname, members in groups.items():
            gname = str(gname)
            members = [str(a) for a in members]
            if not members:
                raise ValidationError(f"{_WHO}: group {gname!r} has no members.")
            unknown = [a for a in members if a not in position]
            if unknown:
                raise ValidationError(
                    f"{_WHO}: group {gname!r} names {unknown[:_SHOWN]}, which "
                    "are not assets of the covariance."
                )
            repeated = sorted({a for a in members if members.count(a) > 1})
            if repeated:
                raise ValidationError(
                    f"{_WHO}: group {gname!r} lists {repeated[:_SHOWN]} more "
                    "than once; a member counts once in the group's sum."
                )
            coef = np.zeros(n)
            coef[[position[a] for a in members]] = 1.0
            group_names.append(gname)
            group_rows[gname] = coef
    gmin = _limit_map(group_min, group_names, "group_min", "groups")
    gmax = _limit_map(group_max, group_names, "group_max", "groups")
    for gname in group_names:
        lo_g = gmin.get(gname, -math.inf)
        hi_g = gmax.get(gname, math.inf)
        if lo_g > hi_g:
            raise ValidationError(
                f"{_WHO}: group {gname!r} has group_min {lo_g:g} above group_max {hi_g:g}."
            )
        if math.isfinite(lo_g) or math.isfinite(hi_g):
            rows.append(_Row("group", gname, group_rows[gname], lo_g, hi_g))
    factor_names: List[str] = []
    loadings = np.zeros((n, 0))
    if factor_loadings is not None:
        factor_names, loadings = _loadings_matrix(factor_loadings, names)
    emin = _limit_map(
        exposure_min, factor_names, "exposure_min", "factors of factor_loadings"
    )
    emax = _limit_map(
        exposure_max, factor_names, "exposure_max", "factors of factor_loadings"
    )
    if (emin or emax) and factor_loadings is None:
        raise ValidationError(f"{_WHO}: exposure limits need factor_loadings.")
    for k, fname in enumerate(factor_names):
        lo_f = emin.get(fname, -math.inf)
        hi_f = emax.get(fname, math.inf)
        if lo_f > hi_f:
            raise ValidationError(
                f"{_WHO}: factor {fname!r} has exposure_min {lo_f:g} above exposure_max {hi_f:g}."
            )
        if math.isfinite(lo_f) or math.isfinite(hi_f):
            rows.append(_Row("exposure", fname, loadings[:, k].copy(), lo_f, hi_f))
    if len(rows) - 1 > MAX_CONSTRAINT_ROWS:
        raise ValidationError(
            f"{_WHO}: {len(rows) - 1} group and exposure limits; at most "
            f"{MAX_CONSTRAINT_ROWS} are taken."
        )

    upper_used = _drop_redundant_bounds(lower, upper, budget, budget_type)
    free_assets = np.flatnonzero(lower != upper)
    if free_assets.size == 0:
        raise ValidationError(
            f"{_WHO}: every asset's lower bound equals its upper bound, so "
            "there is nothing to optimize."
        )
    condition, warnings = _require_positive_definite(cov, names, free_assets)

    feasible, _ = _feasible_point(names, rows, lower, upper_used)
    _require_independent_equalities(rows, free_assets)

    if objective == "max_sharpe":
        assert mu is not None
        outcome = _max_sharpe(
            names, cov, mu, risk_free_rate, budget, rows, lower, upper_used
        )
    else:
        Q, c, constant = _objective_terms(objective, cov, mu, bench, risk_aversion)
        base = _Programme(Q, c, rows, lower, upper_used, list(names))
        outcome = _with_caps(objective, base, constant, caps, feasible, risk_aversion)

    return _report(
        objective=objective,
        names=names,
        cov=cov,
        mu=mu,
        bench=bench,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
        group_rows=group_rows,
        factor_names=factor_names,
        loadings=loadings,
        outcome=outcome,
        warnings=warnings,
        condition=condition,
    )


# ── outcomes ─────────────────────────────────────────────────────────────


@dataclass
class _Outcome:
    """What a solve produced, in the weights."""

    weights: Optional[np.ndarray]
    #: The minimized quantity's value (variance, tracking variance, minus
    #: the utility, minus the Sharpe ratio).
    minimized: Optional[float]
    certificate: Optional[Certificate]
    #: (kind, name, bound, d minimized / d bound) for each binding constraint.
    binding: List[Tuple[str, Optional[str], Optional[float], Optional[float]]]
    method: str
    passes: int
    notes: List[str]
    cap_multipliers: Dict[str, Optional[float]] = field(default_factory=dict)
    effective_risk_aversion: Optional[float] = None
    unsolved: Optional[str] = None


def _binding_from(
    prog: _Programme,
    duals: _Duals,
    x: np.ndarray,
    upper_given: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> List[Tuple[str, Optional[str], Optional[float], Optional[float]]]:
    """Every binding linear constraint of a weight-space programme with
    d(minimized)/d(bound): a bound multiplier for a bound, minus the row
    multiplier for a row (see `_Duals`), times `scale`."""
    out: List[Tuple[str, Optional[str], Optional[float], Optional[float]]] = []
    for j, row in enumerate(prog.rows):
        if row.equality:
            kind = f"{row.kind}_equal" if row.kind != "budget" else "budget"
            out.append((kind, row.name, row.lower, -duals.rows[j] * scale))
        elif duals.row_upper[j]:
            kind = "budget" if row.kind == "budget" else f"{row.kind}_max"
            out.append((kind, row.name, row.upper, -duals.rows[j] * scale))
        elif duals.row_lower[j]:
            kind = f"{row.kind}_min"
            out.append((kind, row.name, row.lower, -duals.rows[j] * scale))
    pinned = prog.lower == prog.upper
    for i, name in enumerate(prog.names):
        if pinned[i]:
            g = float(duals.box[i])
            out.append(("fixed_weight", name, float(prog.lower[i]), g * scale))
        elif duals.at_lower[i]:
            out.append(
                ("lower_bound", name, float(prog.lower[i]), float(duals.box[i]) * scale)
            )
        elif duals.at_upper[i]:
            out.append(
                ("upper_bound", name, float(prog.upper[i]), float(duals.box[i]) * scale)
            )
    return out


def _with_caps(
    objective: str,
    base: _Programme,
    constant: float,
    caps: List[_CapSpec],
    feasible: np.ndarray,
    risk_aversion: float,
) -> _Outcome:
    """The quadratic objectives, with any caps found by their multipliers."""
    red = _Reduced(base)
    search = _Search(base, red, feasible)
    notes: List[str] = []

    def minimized(w: np.ndarray) -> float:
        return float(0.5 * w @ base.Q @ w + base.c @ w + constant)

    def unsolved(failure: _Unsolved) -> _Outcome:
        if failure.best is not None:
            best = failure.best
            return _Outcome(
                weights=best.x,
                minimized=minimized(best.x),
                certificate=best.certificate,
                binding=_binding_from(base, best.duals, best.x),
                method=best.method,
                passes=best.passes,
                notes=notes,
                unsolved=failure.reason,
            )
        return _Outcome(None, None, None, [], "", 0, notes, unsolved=failure.reason)

    # Each cap's own floor: the lowest its quantity can be inside the
    # linear constraints. A cap below it cannot hold.
    floors: Dict[str, _Solve] = {}
    for spec in caps:
        try:
            floor = search.solve([(spec, 1.0)], 0.0)
        except _Unsolved as failure:
            return unsolved(failure)
        lowest = _value(spec, floor.x)
        if lowest > spec.limit**2 * (1.0 + TOLERANCE):
            what = "volatility" if spec.kind == "max_volatility" else "tracking error"
            raise ValidationError(
                f"{_WHO}: {spec.kind}={spec.limit:g} cannot be met: the lowest "
                f"{what} any portfolio inside the other constraints reaches is "
                f"{math.sqrt(lowest):.6g}. Raise the cap to at least that, or "
                "loosen the constraints that hold the "
                f"{what} up."
            )
        floors[spec.kind] = floor

    try:
        plain = search.solve([], 1.0)
    except _Unsolved as failure:
        return unsolved(failure)
    binding_caps = [spec for spec in caps if _value(spec, plain.x) > spec.limit**2]
    for spec in caps:
        if spec not in binding_caps:
            what = "volatility" if spec.kind == "max_volatility" else "tracking error"
            notes.append(
                f"{spec.kind}={spec.limit:g} does not bind: the optimum without "
                f"it has {what} {math.sqrt(_value(spec, plain.x)):.6g}."
            )

    final = plain
    nus: Dict[str, float] = {spec.kind: 0.0 for spec in caps}
    try:
        if len(binding_caps) == 1:
            final, nu = _one_cap(search, binding_caps[0], [], 1.0)
            nus[binding_caps[0].kind] = nu
        elif len(binding_caps) == 2:
            first, second = binding_caps
            # Each alone may already satisfy the other.
            alone_1, nu_1 = _one_cap(search, first, [], 1.0)
            if _value(second, alone_1.x) <= second.limit**2:
                final, nus[first.kind] = alone_1, nu_1
                notes.append(f"{second.kind} does not bind once {first.kind} does.")
            else:
                alone_2, nu_2 = _one_cap(search, second, [], 1.0)
                if _value(first, alone_2.x) <= first.limit**2:
                    final, nus[second.kind] = alone_2, nu_2
                    notes.append(f"{first.kind} does not bind once {second.kind} does.")
                else:
                    final, nu_a, nu_b = _two_caps(search, first, second)
                    nus[first.kind], nus[second.kind] = nu_a, nu_b
    except _Unsolved as failure:
        return unsolved(failure)

    w = final.x
    cap_objs = [
        _Cap(spec.kind, spec.S, spec.centre, spec.limit, nus[spec.kind])
        for spec in caps
    ]
    gradient = base.Q @ w + base.c
    scale = np.abs(base.Q) @ np.abs(w) + np.abs(base.c)
    certificate, duals = _certify(
        base, w, gradient, scale, active=red.active(final.state), caps=cap_objs
    )
    binding = _binding_from(base, duals, w)
    multipliers: Dict[str, Optional[float]] = {}
    for spec in caps:
        nu = nus[spec.kind]
        if nu > 0.0 and math.isfinite(nu):
            # d F / d limit = -nu * d(limit^2)/d limit
            binding.append((spec.kind, None, spec.limit, -2.0 * nu * spec.limit))
            multipliers[spec.kind] = nu
        elif math.isinf(nu):
            binding.append((spec.kind, None, spec.limit, None))
            multipliers[spec.kind] = None
        else:
            multipliers[spec.kind] = 0.0
    effective = None
    if objective in ("max_utility", "max_active_utility"):
        same_centre = {
            "max_utility": "max_volatility",
            "max_active_utility": "max_tracking_error",
        }[objective]
        others = [s for s in caps if s.kind != same_centre and nus[s.kind] > 0.0]
        if not others and math.isfinite(nus.get(same_centre, 0.0)):
            effective = risk_aversion + 2.0 * nus.get(same_centre, 0.0)
    method = final.method
    if binding_caps:
        method += f"; the cap's multiplier by Brent's method in {search.solves} solves"
    return _Outcome(
        weights=w,
        minimized=minimized(w),
        certificate=certificate,
        binding=binding,
        method=method,
        passes=final.passes,
        notes=notes + final.notes,
        cap_multipliers=multipliers,
        effective_risk_aversion=effective,
    )


def _one_cap(
    search: _Search,
    spec: _CapSpec,
    fixed: Sequence[Tuple[_CapSpec, float]],
    f_weight: float,
) -> Tuple[_Solve, float]:
    """
    The answer under `spec`, and its multiplier per unit of f (0 when the
    cap does not bind, infinite when it equals the lowest value attainable).

    Searches theta in [0, 1] on f_weight (1 - theta) f + theta v + the
    `fixed` caps' terms scaled by (1 - theta); the cap is met at theta = 1,
    which the caller has checked.
    """
    target = spec.limit**2
    solved: Dict[float, _Solve] = {}

    def at(theta: float) -> _Solve:
        if theta not in solved:
            weights = [(s, (1.0 - theta) * w) for s, w in fixed] + [(spec, theta)]
            solved[theta] = search.solve(weights, (1.0 - theta) * f_weight)
        return solved[theta]

    def excess(theta: float) -> float:
        return _value(spec, at(theta).x) - target

    uncapped = at(0.0)
    low = _value(spec, uncapped.x) - target
    if low <= 0.0:
        return uncapped, 0.0
    high_solve = at(1.0)
    high = _value(spec, high_solve.x) - target
    if high >= 0.0:
        # The cap equals the lowest attainable value: one portfolio meets
        # it, and its multiplier is unbounded.
        return high_solve, math.inf
    theta = _brent(excess, 0.0, 1.0)
    final = at(theta)
    if _value(spec, final.x) > target and theta < 1.0:
        # Brent returns the bracket end nearest the root; take the side on
        # which the cap holds, one ulp further along.
        theta = float(np.nextafter(theta, 1.0))
        final = at(theta)
    nu = theta / (1.0 - theta) if theta < 1.0 else math.inf
    return final, nu


def _two_caps(
    search: _Search, first: _CapSpec, second: _CapSpec
) -> Tuple[_Solve, float, float]:
    """Both caps binding: the outer search on the second cap's weight, the
    inner on the first's at each outer step."""
    target = second.limit**2
    inner: Dict[float, Tuple[_Solve, float]] = {}

    def solve_at(theta2: float) -> Tuple[_Solve, float]:
        if theta2 not in inner:
            inner[theta2] = _one_cap(search, first, [(second, theta2)], 1.0 - theta2)
        return inner[theta2]

    def excess(theta2: float) -> float:
        return _value(second, solve_at(theta2)[0].x) - target

    high = excess(1.0)
    if high > 0.0:
        raise ValidationError(
            f"{_WHO}: {first.kind}={first.limit:g} and {second.kind}="
            f"{second.limit:g} cannot both hold: with the first met, the "
            f"lowest the second reaches is {math.sqrt(target + high):.6g}. "
            "Loosen one of them."
        )
    theta2 = _brent(excess, 0.0, 1.0)
    final, nu1_scaled = solve_at(theta2)
    if _value(second, final.x) > target and theta2 < 1.0:
        theta2 = float(np.nextafter(theta2, 1.0))
        final, nu1_scaled = solve_at(theta2)
    if theta2 >= 1.0:
        return final, math.inf, math.inf
    # Inner objective: (1 - t1)[(1 - t2) f + t2 v2] + t1 v1, so per unit of f
    # nu2 = t2 / (1 - t2) and nu1 = (inner nu) / (1 - t2).
    nu2 = theta2 / (1.0 - theta2)
    nu1 = nu1_scaled / (1.0 - theta2)
    return final, nu1, nu2


# ── max Sharpe ───────────────────────────────────────────────────────────


def _max_sharpe(
    names: Sequence[str],
    cov: np.ndarray,
    mu: np.ndarray,
    risk_free_rate: float,
    budget: float,
    rows: Sequence[_Row],
    lower: np.ndarray,
    upper: np.ndarray,
) -> _Outcome:
    """
    The homogenised programme in (y, tau): minimize y'Sy subject to
    (mu - rf)'y = 1, 1'y = B tau, each w-row as a homogeneous row in
    (y, tau), each nonzero finite bound as a row l tau <= y_i (or
    y_i <= h tau), a zero bound as y_i's own bound, and tau >= 0; then
    w = y / tau.
    """
    n = len(names)
    excess = mu - risk_free_rate
    w_best, top = _feasible_point(
        names, rows, lower, upper, objective=excess, objective_cap=1.0
    )
    assert top is not None
    if not top > 0.0:
        raise ValidationError(
            f"{_WHO}: no portfolio inside the constraints has a positive "
            f"excess return over risk_free_rate={risk_free_rate:g}: the "
            f"highest is {top:.6g}. The Sharpe ratio is then maximized by "
            "the least negative return over the most risk, which is not a "
            "portfolio anyone wants; lower risk_free_rate, revisit "
            "expected_returns, or use min_variance."
        )

    size = n + 1
    rows_y: List[_Row] = []

    def lift(coef: np.ndarray, tau: float) -> np.ndarray:
        out = np.zeros(size)
        out[:n] = coef
        out[n] = tau
        return out

    rows_y.append(_Row("normalisation", None, lift(excess, 0.0), 1.0, 1.0))
    origin: List[Tuple[str, Optional[str], float, str]] = [
        ("normalisation", None, 1.0, "")
    ]
    for row in rows:
        if row.kind == "budget":
            rows_y.append(_Row("budget", None, lift(row.coef, -budget), 0.0, 0.0))
            origin.append(("budget", None, budget, "equal"))
            continue
        if row.equality:
            rows_y.append(
                _Row(row.kind, row.name, lift(row.coef, -row.lower), 0.0, 0.0)
            )
            origin.append((f"{row.kind}_equal", row.name, row.lower, "equal"))
            continue
        if math.isfinite(row.upper):
            rows_y.append(
                _Row(row.kind, row.name, lift(row.coef, -row.upper), -math.inf, 0.0)
            )
            origin.append((f"{row.kind}_max", row.name, row.upper, "upper"))
        if math.isfinite(row.lower):
            rows_y.append(
                _Row(row.kind, row.name, lift(row.coef, -row.lower), 0.0, math.inf)
            )
            origin.append((f"{row.kind}_min", row.name, row.lower, "lower"))
    lower_y = np.full(size, -math.inf)
    upper_y = np.full(size, math.inf)
    lower_y[n] = 0.0
    box_origin: Dict[int, Tuple[str, float]] = {}
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        if lower[i] == upper[i]:
            if lower[i] == 0.0:
                lower_y[i] = upper_y[i] = 0.0
                box_origin[i] = ("fixed_weight", 0.0)
            else:
                rows_y.append(
                    _Row("fixed_weight", names[i], lift(e, -lower[i]), 0.0, 0.0)
                )
                origin.append(("fixed_weight", names[i], float(lower[i]), "equal"))
            continue
        if lower[i] == 0.0:
            lower_y[i] = 0.0
        elif math.isfinite(lower[i]):
            rows_y.append(
                _Row("lower_bound", names[i], lift(e, -lower[i]), 0.0, math.inf)
            )
            origin.append(("lower_bound", names[i], float(lower[i]), "lower"))
        if upper[i] == 0.0:
            upper_y[i] = 0.0
        elif math.isfinite(upper[i]):
            rows_y.append(
                _Row("upper_bound", names[i], lift(e, -upper[i]), -math.inf, 0.0)
            )
            origin.append(("upper_bound", names[i], float(upper[i]), "upper"))
    Q = np.zeros((size, size))
    Q[:n, :n] = 2.0 * cov
    prog = _Programme(
        Q, np.zeros(size), rows_y, lower_y, upper_y, list(names) + ["tau"]
    )
    red = _Reduced(prog)
    scale = float(excess @ w_best)
    start = np.concatenate([w_best / scale, [1.0 / scale]])
    try:
        solved = _solve_programme(prog, red, start)
        certificate, duals, x, method, passes, notes = (
            solved.certificate,
            solved.duals,
            solved.x,
            solved.method,
            solved.passes,
            solved.notes,
        )
        unsolved = None
    except _Unsolved as failure:
        if failure.best is None:
            return _Outcome(None, None, None, [], "", 0, [], unsolved=failure.reason)
        best_solve = failure.best
        certificate, duals, x, method, passes, notes = (
            best_solve.certificate,
            best_solve.duals,
            best_solve.x,
            best_solve.method,
            best_solve.passes,
            best_solve.notes,
        )
        unsolved = failure.reason
    y, tau = x[:n], float(x[n])
    if not tau > TOLERANCE * max(1.0, float(np.max(np.abs(y)))):
        raise ValidationError(
            f"{_WHO}: the maximum Sharpe ratio inside these constraints is "
            "approached only by levering a long-short book without limit "
            "(the homogenised optimum has no positive scale), so no portfolio "
            f"with budget {budget:g} attains it. Bound the weights on both "
            "sides, or use max_utility, whose risk aversion sets the size."
        )
    w = y / tau
    variance_y = float(y @ cov @ y)
    sharpe = 1.0 / math.sqrt(variance_y)
    # d S / d bound = -1/2 S^3 d(y'Sy)/d bound, and d(y'Sy)/d bound is
    # -m tau for a row whose tau coefficient is minus the bound, and
    # z tau for a zero bound held as y_i's own bound (see the module).
    factor = 0.5 * sharpe**3 * tau
    position = {a: i for i, a in enumerate(names)}
    binding: List[Tuple[str, Optional[str], Optional[float], Optional[float]]] = []
    for j, row in enumerate(rows_y):
        kind, name, bound, side = origin[j]
        if kind == "normalisation":
            continue
        active = (
            side == "equal"
            or (side == "upper" and duals.row_upper[j])
            or (side == "lower" and duals.row_lower[j])
        )
        if active:
            # minimized is -S, so d(-S)/d bound = -factor * m.
            binding.append((kind, name, bound, -factor * float(duals.rows[j])))
            if kind in ("lower_bound", "upper_bound", "fixed_weight"):
                # y_i = bound * tau holds to rounding, so y_i / tau is the
                # bound to an ulp; the bound itself is the exact weight.
                w[position[str(name)]] = bound
    for i in range(n):
        if i in box_origin:
            binding.append(
                ("fixed_weight", names[i], 0.0, factor * float(duals.box[i]))
            )
        elif duals.at_lower[i] and lower[i] == 0.0:
            binding.append(("lower_bound", names[i], 0.0, factor * float(duals.box[i])))
        elif duals.at_upper[i] and upper[i] == 0.0:
            binding.append(("upper_bound", names[i], 0.0, factor * float(duals.box[i])))
    return _Outcome(
        weights=w,
        minimized=-sharpe,
        certificate=certificate,
        binding=binding,
        method=method + " on the homogenised programme in (y, tau)",
        passes=passes,
        notes=notes,
        unsolved=unsolved,
    )


# ── the report ───────────────────────────────────────────────────────────


#: The numbers a solve that reached no answer cannot give.
_UNMEASURED = (
    "objective_value",
    "expected_return",
    "volatility",
    "tracking_error",
    "sharpe_ratio",
    "net_weight",
    "gross_weight",
)


def _report(
    *,
    objective: str,
    names: Sequence[str],
    cov: np.ndarray,
    mu: Optional[np.ndarray],
    bench: Optional[np.ndarray],
    risk_aversion: float,
    risk_free_rate: float,
    group_rows: Mapping[str, np.ndarray],
    factor_names: Sequence[str],
    loadings: np.ndarray,
    outcome: _Outcome,
    warnings: List[str],
    condition: float,
) -> Dict[str, Any]:
    """The result dict. `risk_aversion` and `effective_risk_aversion` are
    present only for the utility objectives, the only ones that have one."""
    maximize = objective in ("max_utility", "max_sharpe", "max_active_utility")
    sense = -1.0 if maximize else 1.0
    w = outcome.weights
    certified = bool(
        w is not None
        and outcome.certificate is not None
        and outcome.certificate.certified
        and outcome.unsolved is None
    )
    warnings = list(warnings) + list(outcome.notes)
    if outcome.unsolved is not None:
        warnings.append(
            f"The exact solve did not reach a certified optimum: "
            f"{outcome.unsolved}. "
            + (
                "The weights below are its last answer, NOT a certified "
                "optimum; the certificate says which condition failed."
                if w is not None
                else "No weights are returned."
            )
        )
    elif outcome.certificate is not None and not outcome.certificate.certified:
        warnings.append(
            "The answer does not certify: "
            + ", ".join(outcome.certificate.failed())
            + f" over the tolerance {TOLERANCE:.0e}. It is not a verified optimum."
        )
    utility = objective in ("max_utility", "max_active_utility")
    result: Dict[str, Any] = {
        "objective": objective,
        "assets": list(names),
        "n_assets": len(names),
        "risk_free_rate": risk_free_rate,
        "condition_number": condition,
        "certified": certified,
        "certificate": (
            None if outcome.certificate is None else outcome.certificate.as_dict()
        ),
    }
    if utility:
        result["risk_aversion"] = risk_aversion
        result["effective_risk_aversion"] = outcome.effective_risk_aversion
    if w is None:
        warnings.append(
            f"{', '.join(_UNMEASURED)} are null: the exact solve reached no "
            "answer, so there are no weights to measure."
        )
        result.update({key: None for key in _UNMEASURED})
        result.update(
            {
                "weights": {},
                "active_weights": None,
                "group_totals": {},
                "exposures": {},
                "binding_constraints": [],
                "solver": f"NOT solved: {outcome.unsolved}",
                "warnings": warnings,
            }
        )
        if utility:
            result["effective_risk_aversion"] = None
        return result

    variance = float(w @ cov @ w)
    volatility = math.sqrt(max(variance, 0.0))
    te = None
    active = None
    if bench is not None:
        d = w - bench
        te = math.sqrt(max(float(d @ cov @ d), 0.0))
        active = {a: float(v) for a, v in zip(names, d)}
    sharpe = None
    if mu is not None and volatility > 0.0:
        sharpe = float((mu - risk_free_rate) @ w) / volatility
    binding = []
    for kind, name, bound, derivative in outcome.binding:
        binding.append(
            {
                "constraint": kind,
                "name": name,
                "bound": None if bound is None else float(bound),
                # + 0.0 so a zero price reads 0.0, not -0.0
                "shadow_price": (
                    None if derivative is None else float(sense * derivative) + 0.0
                ),
            }
        )
    binding.sort(
        key=lambda b: (
            -(abs(b["shadow_price"]) if b["shadow_price"] is not None else math.inf),
            b["constraint"],
            str(b["name"]),
        )
    )
    result.update(
        {
            "objective_value": (
                None
                if outcome.minimized is None
                else float(sense * outcome.minimized) + 0.0
            ),
            "weights": {a: float(v) for a, v in zip(names, w)},
            "active_weights": active,
            "expected_return": float(mu @ w) if mu is not None else None,
            "volatility": volatility,
            "tracking_error": te,
            "sharpe_ratio": sharpe,
            "net_weight": float(np.sum(w)),
            "gross_weight": float(np.sum(np.abs(w))),
            "group_totals": {g: float(coef @ w) for g, coef in group_rows.items()},
            "exposures": {
                f: float(loadings[:, k] @ w) for k, f in enumerate(factor_names)
            },
            "binding_constraints": binding,
            "solver": (
                f"solved exactly by {outcome.method}, the final solve in "
                f"{outcome.passes} pass(es)"
                if certified
                else f"NOT certified: {outcome.method}"
                + (f"; {outcome.unsolved}" if outcome.unsolved else "")
            ),
            "warnings": warnings,
        }
    )
    return result
