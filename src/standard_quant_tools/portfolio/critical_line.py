"""
The efficient frontier under a budget and per-asset bounds, exactly: Markowitz's
critical line algorithm.

    minimize  1/2 w'Sw - lam * mu'w   subject to   1'w = budget,  lo <= w <= hi

for every lam >= 0 at once. lam is the risk tolerance; at a fixed set of
assets held at a bound the solution is LINEAR in lam, so as lam falls from
infinity (the highest-return portfolio) to zero (the minimum-variance one)
the frontier is a chain of straight pieces in the weights. A piece ends where
an asset reaches a bound or leaves one: a CORNER portfolio. Between two
adjacent corners every frontier portfolio is a mix of the two, so the corners
are the whole frontier -- each point on it is one convex combination away,
exact to the rounding of that combination.

REFERENCES. Markowitz, "The optimization of a quadratic function subject to
linear constraints", Naval Research Logistics Quarterly 3 (1956); Niedermayer
& Niedermayer, "Applying Markowitz's critical line algorithm" (2006, in
Handbook of Portfolio Construction, 2010); Bailey & Lopez de Prado, "An
open-source implementation of the critical-line algorithm for portfolio
optimization", Algorithms 6 (2013).

ONE PIECE. With the free assets F and the others fixed at their bounds, the
KKT conditions on F are S_FF w_F + S_FB w_B = lam mu_F + gamma 1 and the
budget, so w_F = alpha + lam beta with

    beta  = S_FF^-1 (d - kappa 1),   d = mu_F - mu_f0,
    kappa = 1'S_FF^-1 d / 1'S_FF^-1 1,

alpha the minimum-variance mix of F given the fixed weights. d is the free
returns less ONE of them (`mu_f0`), which the budget's multiplier absorbs:
when the free returns are tied d is zero exactly, so beta is zero exactly
and a tied piece cannot move by rounding. A fixed asset j's multiplier is
g_j = (Sw)_j - lam mu_j - gamma, also linear in lam; it must stay >= 0 at a
lower bound and <= 0 at an upper one. The piece ends at the largest lam
below the current one where a free weight reaches a bound or a fixed
multiplier changes sign.

THE START is the highest-return portfolio: the bounds filled from the
highest expected return down (a linear programme with a greedy solution).
When expected returns TIE at the margin the linear programme has many
solutions, and the frontier's limit as lam grows is the one of least
variance among them -- a small quadratic programme over the tied assets,
solved exactly by `_active_set`. When no asset is strictly inside its bounds
there, the budget's multiplier is not pinned; the first two assets to move
are the pair at which the interval it may take closes.

Each corner's weights are stepped from the previous corner's along the
piece, w + (lam' - lam) beta, rather than evaluated as alpha + lam beta:
at a large lam those two terms can each dwarf the weights they cancel to.

DEGENERATE CORNERS. Events that coincide are taken one at a time at the
same lam (a zero-length step), each checked against the multipliers on the
new set of free assets. A sequence of such steps that returns to a set of
free assets it has already had is refused by name, as is a covariance that
is singular on the free assets (a riskless asset, or two collinear ones),
since each piece is a solve on it. An event within rounding of lam = 0 is
the minimum-variance end, not a corner.

EVERY CORNER IS CERTIFIED. Each carries the KKT residuals of the programme
above at its own lam (`_active_set.kkt_certificate`, with the budget's
multiplier estimated from the weights). A corner whose residuals pass is the
minimum-variance portfolio at its expected return under the bounds, and with
lam >= 0 the maximum-return one at its volatility.

The linear algebra runs on one BLAS thread (see `_blas`), so the same input
gives the same bits on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import _active_set

#: The dense solves here are sized for up to this many assets. Each corner
#: is a dense solve on the free assets, and a frontier has a few corners per
#: asset.
MAX_ASSETS = 500

#: A weight or a multiplier within this many rounding units (relative to the
#: terms it is a sum of) of the value that ends a piece is treated as having
#: reached it, and the event is taken at the current lam.
_SLACK = 64.0 * float(np.finfo(float).eps)

_EVENT_TEXT = {
    "leaves_lower": "leaves its lower bound",
    "leaves_upper": "leaves its upper bound",
    "reaches_lower": "reaches its lower bound",
    "reaches_upper": "reaches its upper bound",
}


@dataclass(frozen=True)
class Event:
    """One asset changing status at a corner."""

    asset: int
    kind: str

    def describe(self, names: Sequence[str]) -> str:
        return f"{names[self.asset]} {_EVENT_TEXT[self.kind]}"


@dataclass(frozen=True)
class Corner:
    """One corner portfolio: optimal for the programme at `lam`."""

    lam: float
    weights: np.ndarray
    expected_return: float
    variance: float
    #: What changed here, as lam fell through it. Empty for the
    #: minimum-variance end when no asset changes there.
    events: Tuple[Event, ...]
    #: KKT residuals of the programme at `lam`, from the weights.
    certificate: _active_set.Certificate

    @property
    def volatility(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass(frozen=True)
class FrontierPortfolio:
    """A frontier portfolio between two adjacent corners (or at one)."""

    weights: np.ndarray
    lam: float
    expected_return: float
    variance: float
    #: Index of the corner on the higher-return side of the piece, and the
    #: share of it in the mix (1.0 is the corner itself).
    corner: int
    share: float
    certificate: _active_set.Certificate

    @property
    def volatility(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass(frozen=True)
class Frontier:
    """The corners, highest return first, minimum variance last."""

    corners: Tuple[Corner, ...]
    mu: np.ndarray
    cov: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    budget: float
    names: Tuple[str, ...]

    @property
    def top(self) -> Corner:
        return self.corners[0]

    @property
    def min_variance(self) -> Corner:
        return self.corners[-1]

    @property
    def certified(self) -> bool:
        return all(c.certificate.certified for c in self.corners)

    @property
    def worst_residual(self) -> float:
        return max(c.certificate.worst for c in self.corners)

    def return_range(self) -> Tuple[float, float]:
        """(minimum-variance return, highest return): the returns of the
        efficient portfolios inside the bounds."""
        return self.min_variance.expected_return, self.top.expected_return

    def volatility_range(self) -> Tuple[float, float]:
        """(minimum volatility, the highest-return portfolio's volatility)."""
        return self.min_variance.volatility, self.top.volatility

    # ── points between corners ───────────────────────────────────────────

    def _mix(self, k: int, share: float) -> FrontierPortfolio:
        """share * corner k + (1 - share) * corner k + 1, priced and
        certified at the interpolated lam (the piece is linear in lam)."""
        high = self.corners[k]
        if k + 1 >= len(self.corners) or share >= 1.0:
            weights, lam, share = high.weights, high.lam, 1.0
        else:
            low = self.corners[k + 1]
            if share <= 0.0:
                weights, lam = low.weights, low.lam
            else:
                weights = low.weights + share * (high.weights - low.weights)
                weights = np.clip(weights, self.lower, self.upper)
                lam = low.lam + share * (high.lam - low.lam)
        with single_threaded_blas():
            variance = float(weights @ self.cov @ weights)
            certificate = certify(
                weights, lam, self.mu, self.cov, self.lower, self.upper, self.budget
            )
        return FrontierPortfolio(
            weights=weights,
            lam=float(lam),
            expected_return=float(self.mu @ weights),
            variance=variance,
            corner=k,
            share=float(share),
            certificate=certificate,
        )

    def at_return(self, target: float) -> FrontierPortfolio:
        """The minimum-variance portfolio at expected return `target`,
        which must lie in `return_range()`."""
        low, high = self.return_range()
        if not low <= target <= high:
            raise ValidationError(
                f"expected return {target!r} is outside [{low:.6f}, {high:.6f}], "
                "the returns of the efficient portfolios inside these bounds."
            )
        corners = self.corners
        for k in range(len(corners) - 1):
            r_high = corners[k].expected_return
            r_low = corners[k + 1].expected_return
            if target >= r_low:
                if r_high <= r_low:
                    return self._mix(k, 1.0)
                return self._mix(k, (target - r_low) / (r_high - r_low))
        return self._mix(len(corners) - 1, 1.0)

    def at_volatility(self, target: float) -> FrontierPortfolio:
        """The maximum-return portfolio at volatility `target`, which must
        lie in `volatility_range()`. On the piece holding it the variance
        is a quadratic in the share of the riskier corner, increasing on
        [0, 1]; its root there is taken in the cancellation-free form."""
        low, high = self.volatility_range()
        if not low <= target <= high:
            raise ValidationError(
                f"volatility {target!r} is outside [{low:.6f}, {high:.6f}], the "
                "volatilities of the efficient portfolios inside these bounds."
            )
        goal = target * target
        corners = self.corners
        for k in range(len(corners) - 1):
            high_corner, low_corner = corners[k], corners[k + 1]
            if goal >= low_corner.variance or k == len(corners) - 2:
                if goal >= high_corner.variance:
                    return self._mix(k, 1.0)
                if goal <= low_corner.variance:
                    return self._mix(k, 0.0)
                with single_threaded_blas():
                    step = high_corner.weights - low_corner.weights
                    a = float(low_corner.weights @ self.cov @ low_corner.weights)
                    b = float(low_corner.weights @ self.cov @ step)
                    c = float(step @ self.cov @ step)
                excess = goal - a
                if not c > 0.0:
                    return self._mix(k, 1.0)
                root = math.sqrt(max(b * b + c * excess, 0.0))
                denominator = b + root
                share = excess / denominator if denominator > 0.0 else 0.0
                return self._mix(k, min(max(share, 0.0), 1.0))
        return self._mix(len(corners) - 1, 1.0)

    def max_sharpe(self, risk_free_rate: float) -> Optional[FrontierPortfolio]:
        """The maximum-Sharpe portfolio inside the bounds, or None when no
        frontier portfolio earns more than `risk_free_rate`. On each piece
        the Sharpe ratio is (a1 + b1 t) / sqrt(a + 2bt + ct^2) in the share
        t of the riskier corner, whose one stationary point is the root of a
        LINEAR equation, so the maximum over the frontier is exact."""
        if not self.top.expected_return - risk_free_rate > 0.0:
            return None
        best: Optional[Tuple[float, int, float]] = None

        def consider(value: float, k: int, share: float) -> None:
            nonlocal best
            if best is None or value > best[0]:
                best = (value, k, share)

        corners = self.corners
        for k, corner in enumerate(corners):
            if corner.variance > 0.0:
                consider(
                    (corner.expected_return - risk_free_rate) / corner.volatility,
                    k,
                    1.0,
                )
        with single_threaded_blas():
            for k in range(len(corners) - 1):
                high, low = corners[k], corners[k + 1]
                step = high.weights - low.weights
                a1 = low.expected_return - risk_free_rate
                b1 = float(self.mu @ step)
                a = float(low.weights @ self.cov @ low.weights)
                b = float(low.weights @ self.cov @ step)
                c = float(step @ self.cov @ step)
                denominator = b1 * b - a1 * c
                if denominator == 0.0:
                    continue
                share = (a1 * b - b1 * a) / denominator
                if 0.0 < share < 1.0:
                    variance = a + 2.0 * b * share + c * share * share
                    if variance > 0.0:
                        consider((a1 + b1 * share) / math.sqrt(variance), k, share)
        if best is None:
            return None
        _, k, share = best
        return self._mix(k, share)


def certify(
    weights: np.ndarray,
    lam: float,
    mu: np.ndarray,
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: float,
) -> _active_set.Certificate:
    """
    KKT residuals of `weights` for minimize 1/2 w'Sw - lam mu'w subject to
    1'w = budget and the bounds.

    `lam` is the witness: any lam >= 0 at which the residuals vanish proves
    the weights are the minimum-variance portfolio at their own expected
    return (on the set of those portfolios the linear term is constant) and,
    lam being non-negative, the maximum-return one at their own volatility.
    The budget's multiplier and the bound multipliers are estimated from the
    weights alone.
    """
    gradient = cov @ weights - lam * mu
    scale = np.abs(cov) @ np.abs(weights) + abs(lam) * np.abs(mu)
    return _active_set.kkt_certificate(
        weights,
        gradient,
        scale,
        np.ones((1, weights.size)),
        np.array([float(budget)]),
        lower,
        upper,
    )


def _refuse(message: str) -> None:
    raise ValidationError(f"critical line: {message}")


def _validated(
    mu: Any, cov: Any, lower: Any, upper: Any, budget: float, names: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Tuple[str, ...]]:
    mu = np.asarray(mu, dtype=float).reshape(-1)
    n = mu.size
    cov = np.asarray(cov, dtype=float)
    if n < 1 or cov.shape != (n, n):
        _refuse(
            f"the covariance must be {n} x {n} to match the {n} expected returns, "
            f"got shape {cov.shape}."
        )
    if n > MAX_ASSETS:
        _refuse(
            f"{n} assets is more than the {MAX_ASSETS} these dense solves are "
            "sized for. Reduce the universe (or group it) first."
        )
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
        _refuse("the expected returns and the covariance must be finite.")
    if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-12):
        _refuse("the covariance is not symmetric.")
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,)).copy()
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (n,)).copy()
    if not np.all(np.isfinite(lo)):
        _refuse(
            "every lower bound must be finite: without a floor on some weight "
            "the highest-return end of the frontier is unbounded."
        )
    if np.any(np.isnan(hi)) or np.any(hi < lo):
        bad = [i for i in range(n) if not hi[i] >= lo[i]]
        _refuse(f"upper bounds below their lower bounds at assets {bad[:10]}.")
    budget = float(budget)
    if not math.isfinite(budget):
        _refuse(f"budget must be finite, got {budget!r}.")
    floor, ceiling = float(lo.sum()), float(hi.sum())
    if not floor <= budget <= ceiling:
        _refuse(
            f"no portfolio inside the bounds sums to the budget {budget:g}: the "
            f"weights can sum to between {floor:g} and {ceiling:g}."
        )
    labels = (
        tuple(str(x) for x in names)
        if names is not None
        else tuple(f"asset {i}" for i in range(n))
    )
    if len(labels) != n:
        _refuse(f"{len(labels)} names for {n} assets.")
    return mu, cov, lo, hi, budget, labels


class _Tracer:
    """The state of one trace: the weights and each asset's status
    (0 free, -1 at its lower bound, +1 at its upper bound)."""

    def __init__(self, mu, cov, lo, hi, budget, names) -> None:
        self.mu, self.cov, self.lo, self.hi = mu, cov, lo, hi
        self.budget, self.names = budget, names
        self.n = mu.size
        self.status = np.zeros(self.n, dtype=np.int8)
        self.w = lo.copy()

    # ── the highest-return end ───────────────────────────────────────────

    def start(self) -> None:
        """The highest-return portfolio, and least variance among them when
        expected returns tie at the margin."""
        mu, lo, hi, n = self.mu, self.lo, self.hi, self.n
        order = np.argsort(-mu, kind="stable")
        w = lo.copy()
        status = np.full(n, -1, dtype=np.int8)
        remaining = self.budget - float(lo.sum())
        marginal = int(order[-1])
        for i in order:
            room = hi[i] - lo[i]
            if remaining <= room:
                marginal = int(i)
                break
            w[i] = hi[i]
            status[i] = 1
            remaining -= room
        status[marginal] = 0
        w[marginal] = 0.0
        w[marginal] = self.budget - float(w.sum())
        tied = np.flatnonzero(mu == mu[marginal])
        if tied.size > 1:
            self._settle_tie(w, status, tied)
        else:
            self._pin(w, status, marginal)
        self.w, self.status = w, status

    def _pin(self, w: np.ndarray, status: np.ndarray, i: int) -> None:
        """A weight computed as the budget's remainder, set exactly to the
        bound it sits on (or past, by rounding), or left free."""
        if w[i] <= self.lo[i]:
            w[i], status[i] = self.lo[i], -1
        elif w[i] >= self.hi[i]:
            w[i], status[i] = self.hi[i], 1
        else:
            status[i] = 0

    def _settle_tie(self, w: np.ndarray, status: np.ndarray, tied: np.ndarray) -> None:
        """Least variance among the highest-return portfolios: the tied
        assets share what the others leave of the budget."""
        rest = np.setdiff1d(np.arange(self.n), tied)
        for j in rest:
            status[j] = 1 if self.mu[j] > self.mu[tied[0]] else -1
            w[j] = self.hi[j] if status[j] == 1 else self.lo[j]
        share = self.budget - float(w[rest].sum())
        Q = self.cov[np.ix_(tied, tied)]
        c = self.cov[np.ix_(tied, rest)] @ w[rest]
        lo, hi = self.lo[tied], self.hi[tied]
        feasible = lo.copy()
        left = share - float(lo.sum())
        for idx in range(tied.size):
            add = min(hi[idx] - lo[idx], left)
            feasible[idx] += add
            left -= add
        try:
            try:
                solution = _active_set.solve(
                    Q, c, np.ones((1, tied.size)), [share], lo, hi
                )
            except _active_set.ActiveSetFailure:
                solution = _active_set.solve_primal(
                    Q, c, np.ones((1, tied.size)), [share], lo, hi, feasible
                )
        except (_active_set.ActiveSetFailure, np.linalg.LinAlgError) as failure:
            names = [self.names[i] for i in tied]
            _refuse(
                f"expected returns tie at {self.mu[tied[0]]!r} among {names} at "
                "the highest-return end, so the frontier starts at the "
                "least-variance split among them, and that split could not be "
                f"solved ({failure}). Break the tie or drop a tied asset."
            )
        x = solution.x
        for idx, i in enumerate(tied):
            w[i] = x[idx]
            if x[idx] == lo[idx]:
                status[i] = -1
            elif x[idx] == hi[idx]:
                status[i] = 1
            else:
                status[i] = 0

    # ── one piece ────────────────────────────────────────────────────────

    def piece(self, lam: float) -> Any:
        """On the current set of free assets: alpha and beta of the free
        weights (w_F = alpha + lam beta), p and q of the fixed assets'
        multipliers (g = p + lam q), and the budget multiplier's terms.
        `lam` only names the corner in a refusal."""
        free = np.flatnonzero(self.status == 0)
        fixed = np.flatnonzero(self.status != 0)
        mu, cov, w = self.mu, self.cov, self.w
        base = mu[free[0]]
        d = mu[free] - base
        S_FF = cov[np.ix_(free, free)]
        try:
            # A factorization that completes can still carry a pivot that is
            # rounding: a singular matrix often factors in floating point.
            pivots = np.diag(np.linalg.cholesky(S_FF)) ** 2 / np.diag(S_FF)
            if not float(np.min(pivots)) > _SLACK:
                raise np.linalg.LinAlgError("a rounding-size pivot")
            rhs = np.column_stack(
                [np.ones(free.size), d, cov[np.ix_(free, fixed)] @ w[fixed]]
            )
            solved = np.linalg.solve(S_FF, rhs)
        except np.linalg.LinAlgError:
            solved = None
        if solved is None or not np.all(np.isfinite(solved)):
            names = [self.names[i] for i in free]
            _refuse(
                f"the covariance restricted to the {free.size} free assets "
                f"{names[:10]} is singular (not positive definite) at "
                f"lam={lam:.6g}, and each piece of the frontier is a solve on "
                "it: one of them has no variance, two are collinear (a "
                "duplicated asset, or one that tracks another), or the matrix "
                "is not a covariance. Drop the riskless or redundant asset, or "
                "repair the matrix."
            )
        a, m, c = solved[:, 0], solved[:, 1], solved[:, 2]
        one_a = float(a.sum())
        kappa = float(m.sum()) / one_a
        remainder = self.budget - float(w[fixed].sum())
        gamma0 = (remainder + float(c.sum())) / one_a
        alpha = gamma0 * a - c
        beta = m - kappa * a
        S_BF = cov[np.ix_(fixed, free)]
        S_BB_w = cov[np.ix_(fixed, fixed)] @ w[fixed]
        p = S_BF @ alpha + S_BB_w - gamma0
        q = S_BF @ beta - (mu[fixed] - base) + kappa
        return free, fixed, alpha, beta, p, q, gamma0, kappa, base

    def weights_at(self, lam: float, lam_cur: float, free, alpha, beta) -> np.ndarray:
        """The weights at `lam` on the current piece, from the weights at
        `lam_cur`. Stepping from the current corner rather than evaluating
        alpha + lam * beta keeps the rounding to the size of the step: at a
        large lam the two terms of that sum can each be far larger than the
        weights they cancel to, and a budget missed by 3.5e-12 was measured
        that way on a four-asset capped problem."""
        w = self.w.copy()
        if np.isfinite(lam_cur):
            w[free] = w[free] + (lam - lam_cur) * beta
        elif np.any(beta != 0.0):
            w[free] = alpha + lam * beta
        return w


def _next_event(
    tracer: _Tracer, lam_cur: float, just_changed: Set[int]
) -> Tuple[Optional[float], List[Event], Any]:
    """The largest lam below `lam_cur` (or at it, for an event already due
    there to rounding) at which the current piece ends, and what happens.

    From a finite lam each candidate is measured from the current corner:
    a free weight's distance to its bound over its rate, a fixed
    multiplier's value over its rate. Both are on the scale of the weights
    and the multipliers themselves, not of alpha and p, which grow with the
    cancellation described in `_Tracer.weights_at`."""
    free, fixed, alpha, beta, p, q, *_ = tracer.piece(lam_cur)
    lo, hi, mu, cov, w = tracer.lo, tracer.hi, tracer.mu, tracer.cov, tracer.w
    finite = np.isfinite(lam_cur)
    candidates: List[Tuple[float, Event]] = []

    # The scale of the terms a free weight or a fixed multiplier is a sum
    # of, so "already there to rounding" means the same at any size.
    if finite:
        residual = cov @ w - lam_cur * mu
        gamma = float(np.mean(residual[free]))
        multiplier = residual - gamma
        terms = np.abs(cov) @ np.abs(w) + abs(lam_cur) * np.abs(mu) + abs(gamma)
        width = max(1.0, float(np.max(np.abs(w))))

    for idx, i in enumerate(free):
        b = float(beta[idx])
        if b > 0.0:
            kind, bound = "reaches_lower", lo[i]
        elif b < 0.0 and np.isfinite(hi[i]):
            kind, bound = "reaches_upper", hi[i]
        else:
            continue
        if not finite:
            candidates.append(((bound - float(alpha[idx])) / b, Event(int(i), kind)))
            continue
        lam = lam_cur + (bound - float(w[i])) / b
        if lam >= lam_cur:
            if i in just_changed or abs(float(w[i]) - bound) > _SLACK * width:
                continue
            lam = lam_cur
        candidates.append((lam, Event(int(i), kind)))

    for idx, j in enumerate(fixed):
        qj = float(q[idx])
        at_lower = tracer.status[j] == -1
        if at_lower and qj > 0.0:
            kind = "leaves_lower"
        elif not at_lower and qj < 0.0:
            kind = "leaves_upper"
        else:
            continue
        if not finite:
            candidates.append((-float(p[idx]) / qj, Event(int(j), kind)))
            continue
        g = float(multiplier[j])
        lam = lam_cur - g / qj
        if lam >= lam_cur:
            if j in just_changed or abs(g) > _SLACK * float(terms[j]):
                continue
            lam = lam_cur
        candidates.append((lam, Event(int(j), kind)))

    # An event within rounding of lam = 0 is an event at the minimum-variance
    # end, where the portfolio is the same either side of it.
    floor = _SLACK * lam_cur if finite else 0.0
    candidates = [c for c in candidates if c[0] > floor]
    if not candidates:
        return None, [], (free, alpha, beta)
    lam_next = max(c[0] for c in candidates)
    # One event per step; a coincident one is taken by the next step at the
    # same lam, against the multipliers of the new free set.
    first = next(c[1] for c in candidates if c[0] == lam_next)
    return lam_next, [first], (free, alpha, beta)


def _pair_event(tracer: _Tracer, lam_cur: float) -> Tuple[Optional[float], List[Event]]:
    """With no asset free the budget's multiplier lies in an interval, which
    closes as lam falls at the first pair (i at its upper bound, j at its
    lower) with (Sw)_i - lam mu_i = (Sw)_j - lam mu_j. Both then move."""
    s = tracer.cov @ tracer.w
    mu, status = tracer.mu, tracer.status
    floor = _SLACK * lam_cur if np.isfinite(lam_cur) else 0.0
    best: Optional[Tuple[float, int, int]] = None
    for i in np.flatnonzero(status == 1):
        for j in np.flatnonzero(status == -1):
            spread = mu[i] - mu[j]
            if not spread > 0.0:
                continue
            lam = (s[i] - s[j]) / spread
            if not lam < lam_cur or not lam > floor:
                continue
            if best is None or lam > best[0]:
                best = (float(lam), int(i), int(j))
    if best is None:
        return None, []
    lam, i, j = best
    return lam, [Event(i, "leaves_upper"), Event(j, "leaves_lower")]


def critical_line(
    mu: Any,
    cov: Any,
    lower: Any,
    upper: Any,
    budget: float = 1.0,
    names: Optional[Sequence[str]] = None,
) -> Frontier:
    """
    The corner portfolios of the efficient frontier under a budget and
    per-asset bounds, highest return first and minimum variance last.

    Args:
        mu: Expected returns, one per asset.
        cov: Their covariance (symmetric; positive definite on every set of
            assets that is free together).
        lower: Per-asset lower bounds (finite), or one for all.
        upper: Per-asset upper bounds (may be inf), or one for all.
        budget: What the weights sum to.
        names: Asset names for the refusals and event descriptions.

    Raises:
        ValidationError: shapes, non-finite or asymmetric inputs, an
            infinite lower bound, bounds that cannot meet the budget, more
            than `MAX_ASSETS` assets, a covariance singular on the free
            assets, a tie at the highest-return end that cannot be settled,
            or a degenerate corner that cycles.
    """
    mu, cov, lo, hi, budget, labels = _validated(mu, cov, lower, upper, budget, names)
    with single_threaded_blas():
        return _trace(mu, cov, lo, hi, budget, labels)


def _trace(mu, cov, lo, hi, budget, labels) -> Frontier:
    tracer = _Tracer(mu, cov, lo, hi, budget, labels)
    tracer.start()
    n = mu.size
    raw: List[Tuple[float, np.ndarray, Tuple[Event, ...]]] = []
    lam_cur = np.inf
    just_changed: Set[int] = set()
    seen_here: Set[bytes] = set()
    limit = 20 * n + 100
    for _ in range(limit):
        if np.any(tracer.status == 0):
            lam_next, events, (free, alpha, beta) = _next_event(
                tracer, lam_cur, just_changed
            )
            if lam_next is None:
                w_end = tracer.weights_at(0.0, lam_cur, free, alpha, beta)
                raw.append((0.0, w_end, ()))
                break
            w = tracer.weights_at(lam_next, lam_cur, free, alpha, beta)
        else:
            lam_next, events = _pair_event(tracer, lam_cur)
            if lam_next is None:
                raw.append((0.0, tracer.w.copy(), ()))
                break
            w = tracer.w.copy()
        for event in events:
            i = event.asset
            if event.kind == "reaches_lower":
                w[i], tracer.status[i] = lo[i], -1
            elif event.kind == "reaches_upper":
                w[i], tracer.status[i] = hi[i], 1
            else:
                tracer.status[i] = 0
        tracer.w = w
        changed = {e.asset for e in events}
        if raw and lam_next == raw[-1][0]:
            key = tracer.status.tobytes()
            if key in seen_here:
                names = [labels[e.asset] for e in events]
                _refuse(
                    f"the corner at lam={lam_next:.6g} is degenerate: taking its "
                    f"coincident events one at a time ({names}) returned to a "
                    "set of free assets already tried there, so which assets "
                    "move next is not determined. It happens when expected "
                    "returns or covariances tie exactly; perturb or drop one of "
                    "the tied assets."
                )
            seen_here.add(key)
            last_lam, _, last_events = raw[-1]
            raw[-1] = (last_lam, w.copy(), last_events + tuple(events))
            just_changed |= changed
        else:
            seen_here = {tracer.status.tobytes()}
            raw.append((float(lam_next), w.copy(), tuple(events)))
            just_changed = changed
        lam_cur = lam_next
    else:
        _refuse(f"no end to the frontier within {limit} corners.")

    # The minimum-variance end repeats the last corner when the last piece
    # does not move (one free asset, or tied free returns).
    if len(raw) >= 2:
        _, w_end, _ = raw[-1]
        _, w_prev, _ = raw[-2]
        if np.max(np.abs(w_end - w_prev)) <= 8.0 * float(np.finfo(float).eps) * max(
            1.0, float(np.max(np.abs(w_prev)))
        ):
            raw.pop()

    corners = tuple(
        Corner(
            lam=lam,
            weights=w,
            expected_return=float(mu @ w),
            variance=float(w @ cov @ w),
            events=events,
            certificate=certify(w, lam, mu, cov, lo, hi, budget),
        )
        for lam, w, events in raw
    )
    return Frontier(
        corners=corners,
        mu=mu,
        cov=cov,
        lower=lo,
        upper=hi,
        budget=budget,
        names=labels,
    )


__all__ = [
    "MAX_ASSETS",
    "Corner",
    "Event",
    "Frontier",
    "FrontierPortfolio",
    "certify",
    "critical_line",
]
