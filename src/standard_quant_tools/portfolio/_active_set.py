"""
Exact solution of a strictly convex quadratic programme over a box, and the
KKT certificate that says whether a vector is that programme's optimum.

    minimize  1/2 x'Qx + c'x   subject to   A x = b,   lo <= x <= hi

Q is symmetric positive definite and A has a handful of rows (one or two in
this package). `min_volatility` and `target_return` are this programme in
the weights. Long-only `max_sharpe` is this programme after the standard
homogenisation: minimize y'Sy subject to (mu - rf)'y = 1 and y >= 0, then
w = y / 1'y.

THE METHOD IS A PRIMAL-DUAL ACTIVE-SET ONE (Bergounioux, Ito & Kunisch
1999; Hintermueller, Ito & Kunisch 2003). Each pass guesses which variables
sit at their lower bound, which at their upper bound, and which are free. It
fixes the first two groups and solves the equality-constrained KKT system on
the free variables exactly, with one dense linear solve. The answer then
corrects the guess: a free variable outside its bounds is fixed at the bound
it crossed, and a fixed one whose multiplier has the wrong sign is freed.

When a pass leaves the guess unchanged, the KKT conditions hold: the bounds
and the multiplier signs exactly, and stationarity to the rounding of one
solve. For a convex programme those conditions are sufficient, so the answer
is the optimum rather than an iterate within a tolerance of it. It is a
function of the final active set alone, so the same set reached from any
starting point gives the same bits.

TIES ARE BROKEN ONE WAY. A fixed variable whose multiplier is exactly zero
stays fixed, and a free variable exactly at its bound stays free. Both
satisfy the KKT conditions as they stand, so neither has a reason to move.

WHAT IT DOES NOT PROMISE. Without an M-matrix structure the method is not
globally convergent, and it can revisit a guess. Every guess is remembered,
and a repeat ends the run with `ActiveSetFailure`. The exception is a cycle
in which some guess is already optimal to the certificate tolerance: a
weakly active bound, whose multiplier is zero up to rounding, can flip
between fixed and free forever. The best guess in such a cycle is accepted.
A singular KKT system (a free set too small for the equality rows) and the
pass limit also end the run. The caller owns the fallback.

TWO MORE SOLVERS SHARE THE FINAL STEP. `solve_primal` is the primal
active-set method: one bound per pass from a feasible point, so it is slower
but neither cycles on a strictly convex objective nor reaches a singular
system. `solve_homogeneous` is `solve` for max_sharpe under a cap or a short
floor, whose bounds scale with 1'y. Each ends with the KKT solve on its final
active set.

`kkt_certificate` is independent of all of this. It takes a vector and the
objective's gradient there, estimates the multipliers from the vector alone,
and reports the KKT residuals of the programme. It is used on this module's
answers and on any other solver's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: A certified answer's KKT residuals are each at most this, relative to the
#: size of the terms they are residuals of. One dense solve leaves about
#: n * 1e-16, so a solve at a few hundred assets clears it by two orders of
#: magnitude. An iterate a nonlinear solver stopped at misses it by four to
#: eight.
CERTIFICATE_TOLERANCE = 1e-12

#: How close to a bound a starting point's coordinate must be to start fixed
#: at that bound, relative to the starting point's largest coordinate.
START_TOLERANCE = 1e-9

#: How close to a bound a coordinate must be for the certificate to treat it
#: as at that bound, on the same relative scale.
AT_BOUND_TOLERANCE = 1e-9


class ActiveSetFailure(RuntimeError):
    """The method stopped without a fixed point. `reason` says how."""

    def __init__(self, reason: str, passes: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.passes = passes


@dataclass(frozen=True)
class ActiveSetSolution:
    """The optimum, and how it was reached."""

    x: np.ndarray
    passes: int
    n_lower: int
    n_upper: int
    #: True when the run ended in a cycle whose best guess was optimal to
    #: the certificate tolerance (a weakly active bound), not at a fixed point.
    degenerate: bool


@dataclass(frozen=True)
class Certificate:
    """KKT residuals of a vector, and whether they certify it as optimal."""

    stationarity: float
    dual_infeasibility: float
    equality_residual: float
    bound_violation: float
    #: Equality multipliers estimated from the vector, in the convention
    #: grad f = A' multipliers + (bound multipliers), which is SLSQP's.
    multipliers: np.ndarray
    n_free: int

    @property
    def worst(self) -> float:
        return max(
            self.stationarity,
            self.dual_infeasibility,
            self.equality_residual,
            self.bound_violation,
        )

    @property
    def certified(self) -> bool:
        return bool(np.isfinite(self.worst) and self.worst <= CERTIFICATE_TOLERANCE)

    def as_dict(self) -> Dict[str, float]:
        return {
            "stationarity": float(self.stationarity),
            "dual_infeasibility": float(self.dual_infeasibility),
            "equality_residual": float(self.equality_residual),
            "bound_violation": float(self.bound_violation),
            "tolerance": CERTIFICATE_TOLERANCE,
        }


def _as_bounds(n: int, lo: Any, hi: Any) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
    upper = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    return lower, upper


def _kkt_solve(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    at_lo: np.ndarray,
    at_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """x with the fixed variables at their bounds and the free ones solving
    the equality-constrained KKT system; and the equality multipliers, in
    the convention Qx + c + A'lam = 0 on the free set."""
    n = Q.shape[0]
    m = A.shape[0]
    x = np.zeros(n)
    x[at_lo] = lo[at_lo]
    x[at_hi] = hi[at_hi]
    fixed = at_lo | at_hi
    free = np.flatnonzero(~fixed)
    pinned = np.flatnonzero(fixed)
    k = free.size
    if k == 0:
        raise np.linalg.LinAlgError("no free variable")
    a_free = A[:, free]
    K = np.zeros((k + m, k + m))
    K[:k, :k] = Q[np.ix_(free, free)]
    K[:k, k:] = a_free.T
    K[k:, :k] = a_free
    rhs = np.empty(k + m)
    rhs[:k] = -c[free]
    rhs[k:] = b
    if pinned.size:
        x_pinned = x[pinned]
        rhs[:k] -= Q[np.ix_(free, pinned)] @ x_pinned
        rhs[k:] -= A[:, pinned] @ x_pinned
    solution = np.linalg.solve(K, rhs)
    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError("non-finite solution")
    x[free] = solution[:k]
    return x, solution[k:]


def _violation(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x: np.ndarray,
    lam: np.ndarray,
    at_lo: np.ndarray,
    at_hi: np.ndarray,
) -> float:
    """How far one solved guess is from optimal: the larger of its free
    variables' bound violation and its fixed variables' multiplier-sign
    violation, each relative to the scale the certificate uses."""
    free = ~(at_lo | at_hi)
    gradient = Q @ x + c + A.T @ lam
    terms = np.abs(Q) @ np.abs(x) + np.abs(c) + np.abs(A.T) @ np.abs(lam)
    scale_g = max(float(np.max(terms)), np.finfo(float).tiny)
    scale_x = max(1.0, float(np.max(np.abs(x))))
    primal = max(
        float(np.max(lo[free] - x[free], initial=0.0)),
        float(np.max(x[free] - hi[free], initial=0.0)),
    )
    dual = max(
        float(np.max(-gradient[at_lo], initial=0.0)),
        float(np.max(gradient[at_hi], initial=0.0)),
    )
    return max(primal / scale_x, dual / scale_g)


def solve(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: Any,
    hi: Any,
    start: Optional[np.ndarray] = None,
    max_passes: Optional[int] = None,
) -> ActiveSetSolution:
    """
    Minimize 1/2 x'Qx + c'x subject to A x = b and lo <= x <= hi exactly.

    `start` seeds the first guess: a coordinate within `START_TOLERANCE` of
    a bound starts fixed there. With no start every variable starts free,
    so the first pass is the equality-constrained optimum and the bounds it
    breaks become the second guess. Infinite bounds are allowed.

    Raises `ActiveSetFailure` on a cycle, a singular KKT system or the pass
    limit (default max(100, 2n)).
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    c = np.asarray(c, dtype=float).reshape(n)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    lo, hi = _as_bounds(n, lo, hi)
    width = hi - lo
    limit = max(100, 2 * n) if max_passes is None else int(max_passes)

    if start is None:
        at_lo = np.zeros(n, dtype=bool)
        at_hi = np.zeros(n, dtype=bool)
    else:
        x0 = np.asarray(start, dtype=float).reshape(n)
        tol = START_TOLERANCE * max(1.0, float(np.max(np.abs(x0))))
        at_lo = x0 <= lo + tol
        at_hi = (x0 >= hi - tol) & ~at_lo

    seen: Dict[bytes, int] = {}
    trace: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for passes in range(1, limit + 1):
        seen[at_lo.tobytes() + at_hi.tobytes()] = passes - 1
        try:
            x, lam = _kkt_solve(Q, c, A, b, lo, hi, at_lo, at_hi)
        except np.linalg.LinAlgError:
            n_free = int(n - at_lo.sum() - at_hi.sum())
            raise ActiveSetFailure(
                f"the KKT system on pass {passes} was singular: {n_free} free "
                f"variable(s) could not satisfy the {A.shape[0]} equality "
                "constraint(s) with the rest fixed at their bounds",
                passes,
            ) from None
        trace.append((x, lam, at_lo, at_hi))
        multiplier = Q @ x + c + A.T @ lam
        free = ~(at_lo | at_hi)
        with np.errstate(invalid="ignore"):
            leave_lo = at_lo & (multiplier < 0)
            leave_hi = at_hi & (multiplier > 0)
            new_lo = (
                (at_lo & ~leave_lo)
                | (free & (x < lo))
                | (leave_hi & (multiplier > width))
            )
            new_hi = (
                (at_hi & ~leave_hi)
                | (free & (x > hi))
                | (leave_lo & (multiplier < -width))
            )
        new_hi &= ~new_lo
        if np.array_equal(new_lo, at_lo) and np.array_equal(new_hi, at_hi):
            return ActiveSetSolution(
                x=x,
                passes=passes,
                n_lower=int(at_lo.sum()),
                n_upper=int(at_hi.sum()),
                degenerate=False,
            )
        key = new_lo.tobytes() + new_hi.tobytes()
        if key in seen:
            cycle = trace[seen[key] :]
            scores = [_violation(Q, c, A, lo, hi, *state) for state in cycle]
            best = int(np.argmin(scores))
            if scores[best] <= CERTIFICATE_TOLERANCE:
                x, _, best_lo, best_hi = cycle[best]
                return ActiveSetSolution(
                    x=np.clip(x, lo, hi),
                    passes=passes,
                    n_lower=int(best_lo.sum()),
                    n_upper=int(best_hi.sum()),
                    degenerate=True,
                )
            raise ActiveSetFailure(
                f"the active set cycled: pass {passes} returned to the guess "
                f"of pass {seen[key] + 1}, and no guess in the cycle was "
                f"optimal (best KKT violation {scores[best]:.1e})",
                passes,
            )
        at_lo, at_hi = new_lo, new_hi
    raise ActiveSetFailure(f"no fixed point within {limit} passes", limit)


def solve_primal(
    Q: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: Any,
    hi: Any,
    feasible: np.ndarray,
    max_passes: Optional[int] = None,
) -> ActiveSetSolution:
    """
    The same programme by the primal active-set method, from a feasible point.

    The slower, sure method behind `solve`. It keeps a working set of
    variables fixed at a bound and an iterate that satisfies every
    constraint. Each pass solves the KKT system on the working set, which
    gives the minimizer of the face. If that minimizer is inside the box, the
    iterate moves to it and the multipliers are checked: all of the right
    sign is the optimum, and otherwise the most wrongly signed variable is
    freed. If it is outside, the iterate moves toward it as far as the box
    allows and the variable that stopped it is fixed. (Nocedal & Wright,
    Numerical Optimization, Algorithm 16.3, stepping to the face minimizer
    rather than along a step from it, so a full step needs no near-zero
    test.)

    The objective never rises, and a strictly convex one falls on every
    move that is not blocked at once, so no working set recurs except
    through a run of zero-length moves. A working set seen twice after a
    release is therefore a degenerate cycle and ends the run.

    A variable is fixed only when a move toward the face minimizer is
    blocked by it, and that move lies in the null space of the free
    equality rows with a nonzero entry in that variable. So the free rows
    keep their full rank, and the KKT system never becomes singular the way
    `solve`'s guesses can.

    `feasible` must satisfy A x = b and the bounds (to rounding); the
    answer is the KKT solve on the final working set, the same bits `solve`
    gives for that set. Raises `ActiveSetFailure` on a cycle or the pass
    limit (default 10n + 100).
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    c = np.asarray(c, dtype=float).reshape(n)
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    lo, hi = _as_bounds(n, lo, hi)
    limit = 10 * n + 100 if max_passes is None else int(max_passes)
    eps = np.finfo(float).eps

    x = np.clip(np.asarray(feasible, dtype=float).reshape(n), lo, hi)
    at_lo = np.zeros(n, dtype=bool)
    at_hi = np.zeros(n, dtype=bool)
    released: set = set()
    for passes in range(1, limit + 1):
        try:
            z, lam = _kkt_solve(Q, c, A, b, lo, hi, at_lo, at_hi)
        except np.linalg.LinAlgError:
            raise ActiveSetFailure(
                f"the primal method's KKT system on pass {passes} was singular",
                passes,
            ) from None
        free = ~(at_lo | at_hi)
        step = np.where(free, z - x, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            to_lo = np.where(free & (step < 0), (lo - x) / step, np.inf)
            to_hi = np.where(free & (step > 0), (hi - x) / step, np.inf)
        i_lo = int(np.argmin(to_lo))
        i_hi = int(np.argmin(to_hi))
        if min(to_lo[i_lo], to_hi[i_hi]) >= 1.0:
            x = z
            multiplier = Q @ x + c + A.T @ lam
            terms = np.abs(Q) @ np.abs(x) + np.abs(c) + np.abs(A.T) @ np.abs(lam)
            slack = 64.0 * eps * terms
            wrong = np.where(at_lo, -multiplier, 0.0) + np.where(at_hi, multiplier, 0.0)
            wrong = np.where(wrong > slack, wrong / np.maximum(terms, 1e-300), 0.0)
            if not np.any(wrong > 0.0):
                return ActiveSetSolution(
                    x=x,
                    passes=passes,
                    n_lower=int(at_lo.sum()),
                    n_upper=int(at_hi.sum()),
                    degenerate=False,
                )
            release = int(np.argmax(wrong))
            at_lo[release] = False
            at_hi[release] = False
            key = at_lo.tobytes() + at_hi.tobytes()
            if key in released:
                raise ActiveSetFailure(
                    f"the primal method cycled through zero-length moves on "
                    f"pass {passes}",
                    passes,
                )
            released.add(key)
            continue
        if to_lo[i_lo] <= to_hi[i_hi]:
            alpha, block, at_bound = float(to_lo[i_lo]), i_lo, at_lo
            value = lo[i_lo]
        else:
            alpha, block, at_bound = float(to_hi[i_hi]), i_hi, at_hi
            value = hi[i_hi]
        x = np.where(free, x + max(alpha, 0.0) * step, x)
        x[block] = value
        at_bound[block] = True
        x = np.clip(x, lo, hi)
    raise ActiveSetFailure(
        f"the primal method found no optimum within {limit} passes", limit
    )


def solve_homogeneous(
    S: np.ndarray,
    excess: np.ndarray,
    lo: Any,
    hi: Any,
    start: Optional[np.ndarray] = None,
    max_passes: Optional[int] = None,
) -> ActiveSetSolution:
    """
    The maximum-Sharpe weights inside a box, by the primal-dual active-set
    method on the homogenised programme

        minimize y'Sy  subject to  excess'y = 1,  s = 1'y,
                                   lo_i s <= y_i <= hi_i s,

    and w = y / s. With a cap or a short floor the bounds scale with s, so
    they are general linear constraints in (y, s) rather than a box, and
    `solve` does not apply. On a fixed active set they are still equalities:
    a weight at a bound is y_i = c_i s, and the free weights and s come from
    one dense KKT solve. The update and the cycle guard are `solve`'s, with
    no cross jump between bounds and no degenerate acceptance; the caller
    certifies the answer in the weights.

    `start` (weights) seeds the guess. The returned `x` is the weights: the
    free ones are y_i / s, and those at a bound are the bound itself.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    excess = np.asarray(excess, dtype=float).reshape(n)
    lo, hi = _as_bounds(n, lo, hi)
    limit = max(100, 2 * n) if max_passes is None else int(max_passes)
    if start is None:
        at_lo = np.zeros(n, dtype=bool)
        at_hi = np.zeros(n, dtype=bool)
    else:
        w0 = np.asarray(start, dtype=float).reshape(n)
        tol = START_TOLERANCE * max(1.0, float(np.max(np.abs(w0))))
        at_lo = w0 <= lo + tol
        at_hi = (w0 >= hi - tol) & ~at_lo

    seen: Dict[bytes, int] = {}
    for passes in range(1, limit + 1):
        seen[at_lo.tobytes() + at_hi.tobytes()] = passes
        fixed = at_lo | at_hi
        free = np.flatnonzero(~fixed)
        pinned = np.flatnonzero(fixed)
        bound = np.where(at_lo, lo, hi)[pinned]
        k = free.size
        K = np.zeros((k + 3, k + 3))
        K[:k, :k] = 2.0 * S[np.ix_(free, free)]
        coupling = 2.0 * S[np.ix_(free, pinned)] @ bound
        K[:k, k] = coupling
        K[k, :k] = coupling
        K[k, k] = 2.0 * bound @ S[np.ix_(pinned, pinned)] @ bound
        C = np.zeros((2, k + 1))
        C[0, :k] = excess[free]
        C[0, k] = excess[pinned] @ bound
        C[1, :k] = 1.0
        C[1, k] = bound.sum() - 1.0
        K[: k + 1, k + 1 :] = C.T
        K[k + 1 :, : k + 1] = C
        rhs = np.zeros(k + 3)
        rhs[k + 1] = 1.0
        try:
            solution = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            raise ActiveSetFailure(
                f"the homogenised KKT system on pass {passes} was singular",
                passes,
            ) from None
        if not np.all(np.isfinite(solution)) or not solution[k] > 0.0:
            raise ActiveSetFailure(
                f"the homogenised solve on pass {passes} gave no positive scale",
                passes,
            )
        s = float(solution[k])
        y = np.zeros(n)
        y[free] = solution[:k]
        y[pinned] = bound * s
        multiplier = 2.0 * S @ y + solution[k + 1] * excess + solution[k + 2]
        is_free = ~fixed
        new_lo = (at_lo & ~(multiplier < 0)) | (is_free & (y < lo * s))
        new_hi = (at_hi & ~(multiplier > 0)) | (is_free & (y > hi * s))
        new_hi &= ~new_lo
        if np.array_equal(new_lo, at_lo) and np.array_equal(new_hi, at_hi):
            w = y / s
            w[at_lo] = lo[at_lo]
            w[at_hi] = hi[at_hi]
            return ActiveSetSolution(
                x=w,
                passes=passes,
                n_lower=int(at_lo.sum()),
                n_upper=int(at_hi.sum()),
                degenerate=False,
            )
        key = new_lo.tobytes() + new_hi.tobytes()
        if key in seen:
            raise ActiveSetFailure(
                f"the active set cycled: pass {passes} returned to the guess "
                f"of pass {seen[key]}",
                passes,
            )
        at_lo, at_hi = new_lo, new_hi
    raise ActiveSetFailure(f"no fixed point within {limit} passes", limit)


def _multipliers(
    gradient: np.ndarray,
    A: np.ndarray,
    free: np.ndarray,
    at_lo: np.ndarray,
    at_hi: np.ndarray,
) -> np.ndarray:
    """The equality multipliers that best explain `gradient`: least squares
    on the free set when it determines them, else (one row of one sign, a
    vertex) the middle of the interval the bound multipliers' signs allow."""
    m = A.shape[0]
    if free.any() and np.linalg.matrix_rank(A[:, free]) == m:
        return np.linalg.lstsq(A[:, free].T, gradient[free], rcond=None)[0]
    row = A[0]
    if m == 1 and np.all(row > 0):
        ratio = gradient / row
        # at a lower bound the multiplier must leave gradient - row*lam >= 0
        upper = float(np.min(ratio[at_lo], initial=np.inf))
        lower = float(np.max(ratio[at_hi], initial=-np.inf))
        if np.isfinite(upper) and np.isfinite(lower):
            return np.array([(upper + lower) / 2.0])
        if np.isfinite(upper):
            return np.array([upper])
        if np.isfinite(lower):
            return np.array([lower])
    return np.linalg.lstsq(A.T, gradient, rcond=None)[0]


def kkt_certificate(
    x: np.ndarray,
    gradient: np.ndarray,
    gradient_scale: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    lo: Any,
    hi: Any,
) -> Certificate:
    """
    KKT residuals of `x` for: minimize f subject to A x = b, lo <= x <= hi.

    `gradient` is grad f(x). `gradient_scale` is the componentwise size of
    the terms it is a sum of (for 2Sx, 2|S||x|), so stationarity is measured
    against the numbers that cancelled rather than against their difference,
    which is near zero on the free set by design. The multipliers are
    estimated from `x` alone, so a certificate does not take any solver's
    word for them. A coordinate within `AT_BOUND_TOLERANCE` of a bound is
    treated as at it.

    stationarity        largest |grad f - A'lam| over the free coordinates
    dual_infeasibility  largest wrong-signed bound multiplier
    equality_residual   largest |A_j x - b_j| / max(|b_j|, |A_j||x|)
    bound_violation     largest distance outside the box / max(1, max|x|)

    The first two are relative to max(gradient_scale + |A'||lam|). For a
    convex programme all four at zero is optimality, so all four at most
    `CERTIFICATE_TOLERANCE` certifies the answer to that tolerance.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    lo, hi = _as_bounds(n, lo, hi)
    scale_x = max(1.0, float(np.max(np.abs(x))))
    tol = AT_BOUND_TOLERANCE * scale_x
    at_lo = x <= lo + tol
    at_hi = (x >= hi - tol) & ~at_lo
    free = ~(at_lo | at_hi)

    lam = _multipliers(gradient, A, free, at_lo, at_hi)
    residual = gradient - A.T @ lam
    scale = float(np.max(gradient_scale + np.abs(A.T) @ np.abs(lam)))
    scale = max(scale, np.finfo(float).tiny)
    stationarity = float(np.max(np.abs(residual[free]), initial=0.0)) / scale
    dual = max(
        float(np.max(-residual[at_lo], initial=0.0)),
        float(np.max(residual[at_hi], initial=0.0)),
    )
    equality = np.abs(A @ x - b) / np.maximum(
        np.maximum(np.abs(b), np.abs(A) @ np.abs(x)), np.finfo(float).tiny
    )
    bound = max(
        float(np.max(lo - x, initial=0.0)),
        float(np.max(x - hi, initial=0.0)),
    )
    return Certificate(
        stationarity=stationarity,
        dual_infeasibility=dual / scale,
        equality_residual=float(np.max(equality)),
        bound_violation=bound / scale_x,
        multipliers=lam,
        n_free=int(free.sum()),
    )
