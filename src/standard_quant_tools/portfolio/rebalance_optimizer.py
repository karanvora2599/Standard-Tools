"""
The trade to make from the weights held, with what trading costs counted:
the exact optimum of a mean-variance utility net of linear and quadratic
trading costs, under the constrained optimizer's limits and caps on
turnover, gross weight and volatility.

`optimize_portfolio` answers "which portfolio do I want". Asked again a week
later with forecasts that moved a little, it answers with a portfolio that
moved a little in every name, and the book that follows it trades every
name every week. Costs are what make holding still worth something, so
this solves the question a rebalance actually asks, from the current
weights w0:

    maximize  mu'w - (gamma/2) w'Sw - sum_i kappa_i |w_i - w0_i|
                                    - sum_i eta_i (w_i - w0_i)^2

under per-asset bounds, a budget (equal to, or at most, a value), group
and factor-exposure limits -- all read exactly as optimize_portfolio reads
them, through its own helpers -- and three caps: turnover
sum |w_i - w0_i| <= T, gross weight sum |w_i| <= G and volatility
sqrt(w'Sw) <= sigma. kappa is a cost per unit of weight traded (0.0015 is
15 bps: spread and commission); eta prices the square of the trade
(impact, which grows with size).

WHY NOT SPLIT THE TRADES. The textbook route writes w - w0 = u - v with
u, v >= 0 and |w - w0| as u + v, which makes the programme a smooth QP in
twice the variables. Its Hessian is singular: moving u_i and v_i up
together leaves the weights, and so the variance, where they are, a
direction of zero curvature that only kappa_i prices. An active-set method
that frees both halves of a pair at once then meets a singular KKT system,
and one that frees one variable at a time avoids it only while every
multiplier it reads is right-signed -- a transiently wrong-signed turnover
row frees the pair, and releasing that row next leaves the system
singular. The same trap comes back for the gross cap (w = p - q) and for
every asset whose kappa is zero. So the method here never splits.

THE METHOD WORKS ON THE WEIGHTS, PIECE BY PIECE. Each weight's line is cut
at its BREAKPOINTS -- its two bounds, its current weight w0_i when trading
it costs something or a turnover cap counts it, and zero when a gross cap
counts it -- and between two breakpoints the costs and the caps' terms are
linear, with known signs. Every weight is in one of two states: HELD at a
breakpoint, or FREE on one piece between two. On a fixed set of states
(a FACE) the programme is a strictly convex QP with linear rows -- the
turnover cap is sum_i s_i (w_i - w0_i) <= T with s_i the trade's sign on
the piece, the gross cap likewise with the weight's sign -- and its
minimizer is one dense KKT solve on the free weights. H = gamma S +
2 diag(eta) is positive definite, so no face has the split's
zero-curvature direction. For a weight held at a kink the optimality
check is two-sided: the smooth gradient plus the rows' multipliers must
lie within the kink's subgradient interval -- [-(kappa_i + rho),
kappa_i + rho] at the current weight, rho the turnover cap's multiplier
-- and a weight outside it is freed onto the piece on the side it points
to. The no-trade band is that interval.

TWO STAGES FIND THE FACE, as in optimize_portfolio. First the primal-dual
active-set method (`_active_set.solve`'s, Hintermueller, Ito & Kunisch
2003): guess a face, solve it, correct every state at once, until a pass
changes nothing. It runs on the programme WITHOUT the turnover and gross
caps: with one binding it was measured to oscillate -- the free set
swinging between most of the book and a handful of names as the cap's
multiplier widened and narrowed every band at once -- and to stop on a
singular system; without them it takes a dozen or two passes, and an
answer that satisfies the caps is the optimum with them (it is optimal
over a larger set). When the caps bind, or the first stage stops, the
primal active-set method of `_active_set.solve_rows` (Nocedal & Wright,
Algorithm 16.3) takes over, with breakpoints in the place of bounds: from
a feasible point, step toward the face's minimizer; a step that would
carry a weight past a breakpoint of its piece, or an idle row or cap past
its limit, stops there and holds that weight (or binds that row); a step
that arrives checks the multipliers and releases the most wrongly signed.
A row joins the working set only on a step that moves it, so the working
rows keep full rank on the free weights. It changes one state per pass --
two passes for each name that trades from its current weight to a bound
-- so its cost is the number of names traded, and a warm start (below)
is what makes a sequence of rebalances cheap.

A WEIGHT IN ITS NO-TRADE REGION COMES BACK AS w0, BIT FOR BIT. Held at its
current-weight breakpoint it is set to that breakpoint's value, which is
w0_i itself. The budget row needs at least one free weight in every KKT
system, so when the answer is to trade nothing at all one weight stays
free at its breakpoint, and its computed value is the budget less the
others -- w0_i to rounding, not exactly. A free weight the working rows
alone determine (no direction of the face moves it) and that lies within
the certificate's tolerance of a breakpoint of its piece is set to the
breakpoint at the end; the certificate is computed for the weights as
returned, and the snap is undone if it would not certify.

A BOOK ON ITS LIMITS TO ROUNDING IS THAT BOOK. A held book is the last
answer drifted by prices: it meets its budget within 1e-13 and its caps
and bounds within an ulp. A book that meets every limit within the
certificate's tolerance starts the solve as it is (a weight an ulp past a
bound is clipped onto it). The first solve absorbs the budget's residual,
in a weight freed on the side, and with the room, to take it -- a weight
at its lower bound cannot give up 1e-13, and freed there it was held again
at once, leaving the budget no free weight and the next system singular.
When something trades, the traded weights take the residual and the
budget holds exactly. When nothing trades, every weight comes back as
held, to the bit, and the budget keeps the book's own residual, reported
in `warnings`: moving one name by 1e-13 to hide it would be a trade nobody
asked for, and would break that name's no-trade guarantee. A book that
misses a limit by more than rounding breaks it: it is solved from the
nearest feasible point, with `required_turnover`.

THE CAPS. Turnover and gross caps are ROWS: on a face they are linear,
they enter the KKT system only while they bind, and their multipliers come
out of the same solve -- one solve per pass however many caps bind, where
a multiplier search per cap would multiply the solves (three nested
bracketed searches for three caps). The volatility cap is quadratic, so it
is what it is in optimize_portfolio: a multiplier nu found by Brent's
method on (1 - theta) f + theta w'Sw, theta in [0, 1] -- the programme at
risk aversion gamma + 2 nu -- with the lowest volatility attainable inside
the other constraints (theta = 1) computed first and a cap below it
refused by name. The search is optimize_portfolio's own (`_one_cap`),
each solve started from the previous one's answer and states.

FEASIBILITY. When the current weights satisfy every constraint, to
rounding (above), they are the starting point. Otherwise a linear programme finds the feasible point
nearest them in turnover -- `required_turnover`, the least any answer must
trade -- and the method starts there; when there is none,
optimize_portfolio's least-violation programme, with the turnover and
gross caps added as rows, refuses naming the constraints its dual prices
(a Farkas certificate, the turnover cap among them when the trade it
allows cannot reach the other limits).

THE CERTIFICATE IS COMPUTED FROM THE ANSWER, by optimize_portfolio's
`_certify` with held kinks: the row multipliers that best explain the
gradient on the free weights (each free weight's cost and cap terms at its
piece's signs), then stationarity, the multipliers' signs -- for a weight
held at a kink, |z_i| within the kink's interval -- every constraint's
violation, and complementary slackness, each against 1e-12 relative.

SHADOW PRICES are in the utility's units: each binding constraint's
`shadow_price` is d utility / d bound -- a bound's, a group's, the
budget's, the turnover cap's (what one more unit of turnover is worth), the
gross cap's, and the volatility cap's per unit of volatility (2 nu sigma).
The trading costs are part of the utility, so a price is net of them. At
a degenerate point (a cap at its limit while the names that move all
trade the same way, so the budget already fixes it) the price is one
valid value between the two one-sided rates.

WARM STARTS. `active_set` in the result names each asset's final state
("hold", "buy", "sell", "lower", "upper", "zero", "fixed"). Passing the
previous answer back as `start=` on the next, nearby problem -- the next
rebalance date -- makes those states and its binding rows and caps the
first guess. With no cap binding, the first stage starts from it; with
one, its KKT point is made feasible by holding each weight that left its
piece and binding each row it broke (a few rounds at most) and the primal
method starts there, from the cold start if that fails. Either way the
answer is the optimum: a start changes the passes, never the answer
beyond rounding.

SIZED FOR UP TO `MAX_ASSETS` ASSETS, as optimize_portfolio is: every pass is
a dense solve over the free weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_covariance
from standard_quant_tools.portfolio.general_optimizer import (
    MAX_ASSETS,
    STEP_TOLERANCE,
    TOLERANCE,
    Certificate,
    _AbsoluteRow,
    _binding_from,
    _Cap,
    _cap_floor_refusal,
    _CapSpec,
    _certify,
    _constraint_rows,
    _drop_redundant_bounds,
    _feasible_point,
    _finite,
    _Kinks,
    _least_absolute_point,
    _named_covariance,
    _one_cap,
    _parse_bounds,
    _Programme,
    _require_independent_equalities,
    _require_positive_definite,
    _Row,
    _scalar,
    _Unsolved,
    _value,
    _vector,
)

__all__ = ["REBALANCE_STATES", "optimize_rebalance"]

_WHO = "optimize_rebalance"
_SHOWN = 8

#: The states an asset can end in, as `active_set` reports them and `start`
#: reads them: held at its current weight, trading up or down on a piece,
#: at its lower or upper bound, held at zero by the gross cap's kink, or
#: fixed by equal bounds.
REBALANCE_STATES = ("hold", "buy", "sell", "lower", "upper", "zero", "fixed")

#: A starting point's coordinate within this of a breakpoint, relative to
#: the point's scale, starts held there (a linear programme's vertex is
#: exact to about this).
_START_SNAP = 1e-9

#: The most rounds of holding and binding that turn a warm start's guess
#: into a feasible start before the cold start is taken instead.
_CRASH_ROUNDS = 25

# ── the pieces ───────────────────────────────────────────────────────────


@dataclass
class _Pieces:
    """
    Each weight's breakpoints, in order: its lower bound, the kinks inside
    its box (the current weight, zero), its upper bound; padded with NaN to
    four. On each piece between consecutive breakpoints, the sign of the
    trade (w - w0) and of the weight. A weight whose bounds are equal has
    one breakpoint and no piece.
    """

    w0: np.ndarray
    points: np.ndarray
    count: np.ndarray
    trade_sign: np.ndarray
    weight_sign: np.ndarray

    @property
    def n(self) -> int:
        return int(self.w0.size)


def _build_pieces(
    lower: np.ndarray,
    upper: np.ndarray,
    w0: np.ndarray,
    trade_kink: np.ndarray,
    zero_kink: bool,
) -> _Pieces:
    n = w0.size
    points = np.full((n, 4), np.nan)
    count = np.zeros(n, dtype=np.intp)
    trade_sign = np.zeros((n, 3))
    weight_sign = np.zeros((n, 3))
    for i in range(n):
        lo, hi = float(lower[i]), float(upper[i])
        if lo == hi:
            pts = [lo]
        else:
            inner: List[float] = []
            # w0 first, so a kink at zero that is also the current weight
            # keeps the current weight's bits
            if trade_kink[i] and lo < w0[i] < hi:
                inner.append(float(w0[i]))
            if zero_kink and lo < 0.0 < hi and not any(v == 0.0 for v in inner):
                inner.append(0.0)
            pts = [lo] + sorted(inner) + [hi]
        count[i] = len(pts)
        points[i, : len(pts)] = pts
        for s in range(len(pts) - 1):
            a, b = pts[s], pts[s + 1]
            if math.isfinite(a) and math.isfinite(b):
                mid = 0.5 * (a + b)
            elif math.isfinite(a):
                mid = a + 1.0 + abs(a)
            elif math.isfinite(b):
                mid = b - 1.0 - abs(b)
            else:
                mid = 0.0
            trade_sign[i, s] = 1.0 if mid > w0[i] else -1.0
            weight_sign[i, s] = 1.0 if mid > 0.0 else -1.0
    return _Pieces(w0, points, count, trade_sign, weight_sign)


@dataclass
class _Face:
    """
    The states: `held` weights at breakpoint `pos`, the others free on
    piece `pos`; the linear inequality rows bound at their lower or upper
    limit; the absolute caps binding or not.
    """

    held: np.ndarray
    pos: np.ndarray
    row_lo: np.ndarray
    row_hi: np.ndarray
    cap_on: np.ndarray

    def copy(self) -> "_Face":
        return _Face(
            self.held.copy(),
            self.pos.copy(),
            self.row_lo.copy(),
            self.row_hi.copy(),
            self.cap_on.copy(),
        )

    def key(self) -> bytes:
        return (
            self.held.tobytes()
            + self.pos.tobytes()
            + self.row_lo.tobytes()
            + self.row_hi.tobytes()
            + self.cap_on.tobytes()
        )


class _Problem:
    """The constraints: pieces, linear rows (equalities always working,
    inequalities as rows), and the absolute caps."""

    def __init__(
        self, pieces: _Pieces, rows: Sequence[_Row], absolute: Sequence[_AbsoluteRow]
    ) -> None:
        n = pieces.n
        self.n = n
        self.pieces = pieces
        self.rows = list(rows)
        eq = [r for r in rows if r.equality]
        ineq = [r for r in rows if not r.equality]
        self.eq_index = [j for j, r in enumerate(rows) if r.equality]
        self.ineq_index = [j for j, r in enumerate(rows) if not r.equality]
        self.A_eq = np.array([r.coef for r in eq], dtype=float).reshape(-1, n)
        self.b_eq = np.array([r.lower for r in eq], dtype=float)
        self.G = np.array([r.coef for r in ineq], dtype=float).reshape(-1, n)
        self.g_lo = np.array([r.lower for r in ineq], dtype=float)
        self.g_hi = np.array([r.upper for r in ineq], dtype=float)
        self.absG = np.abs(self.G)
        self.absolute = list(absolute)
        #: Each cap's sign on each piece: the trade's for turnover, the
        #: weight's for gross.
        self.cap_sign = [
            pieces.trade_sign if cap.kind == "max_turnover" else pieces.weight_sign
            for cap in absolute
        ]
        self.index = np.arange(n)

    def relaxed(self) -> "_Problem":
        """The same programme without its absolute caps (same pieces)."""
        if not self.absolute:
            return self
        return _Problem(self.pieces, self.rows, [])

    def empty_face(self) -> _Face:
        return _Face(
            np.ones(self.n, dtype=bool),
            np.zeros(self.n, dtype=np.intp),
            np.zeros(self.G.shape[0], dtype=bool),
            np.zeros(self.G.shape[0], dtype=bool),
            np.zeros(len(self.absolute), dtype=bool),
        )

    def signs(self, face: _Face) -> List[np.ndarray]:
        """The signs every term takes at each weight: on its piece when
        free, at its breakpoint when held (zero at that term's own kink).
        Item 0 is the trade's; then one per absolute cap."""
        P = self.pieces
        piece = np.minimum(face.pos, 2)
        point = P.points[self.index, face.pos]
        free = ~face.held
        trade = np.where(free, P.trade_sign[self.index, piece], np.sign(point - P.w0))
        out = [trade]
        for cap, sign in zip(self.absolute, self.cap_sign):
            out.append(
                np.where(free, sign[self.index, piece], np.sign(point - cap.centre))
            )
        return out

    def working(
        self, face: _Face, signs: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The working rows as A x = b: the equalities, the bound inequality
        rows, the binding caps (each with its pieces' signs as
        coefficients, so sum s_i (x_i - c_i) = limit). Also the indices of
        the inequality rows and caps it holds, in order."""
        bound = np.flatnonzero(face.row_lo | face.row_hi)
        caps = np.flatnonzero(face.cap_on)
        parts_A = [self.A_eq, self.G[bound]]
        parts_b = [
            self.b_eq,
            np.where(face.row_lo[bound], self.g_lo[bound], self.g_hi[bound]),
        ]
        for k in caps:
            coef = signs[1 + k]
            cap = self.absolute[k]
            parts_A.append(coef.reshape(1, -1))
            parts_b.append(np.array([cap.limit + float(coef @ cap.centre)]))
        A = np.vstack(parts_A)
        b = np.concatenate(parts_b)
        return A, b, bound, caps


# ── one solve ────────────────────────────────────────────────────────────


@dataclass
class _Solved:
    """One programme's answer: the weights, the final states, passes."""

    x: np.ndarray
    face: _Face
    passes: int


@dataclass
class _KKT:
    """A face's KKT solve, on its free weights `F` (each on the piece
    [`lo`, `hi`]): their values `z`, and the working rows' multipliers in
    the convention grad + sum lam a = 0 -- equalities, bound inequality
    rows (`bound`), caps (one per cap, zero where it is off) -- with
    `coef`, each cap's sign on the free weights' pieces."""

    F: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    z: np.ndarray
    lam_eq: np.ndarray
    lam_in: np.ndarray
    cap_mult: np.ndarray
    bound: np.ndarray
    coef: List[np.ndarray]


def _face_kkt(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    x: np.ndarray,
    face: _Face,
) -> _KKT:
    """
    The minimizer of the programme on `face`, held weights at the
    breakpoints `x` holds them at: one dense solve of

        [ H_FF  C' ] [ z   ]   [ -(c_F + kappa_F s_F + H_F,held x_held) ]
        [ C     0  ] [ lam ] = [ the rows' targets less the held part   ]

    over the free weights F, with C the working rows on F -- equalities,
    bound inequality rows, binding caps (sum s_i (x_i - c_i) = limit with
    s the pieces' signs; a held weight counts |x_i - c_i|). Raises
    `np.linalg.LinAlgError` when the system is singular or its solution
    not finite.
    """
    P = prob.pieces
    held = face.held
    pos = face.pos
    F = np.flatnonzero(~held)
    k = F.size
    piece = np.minimum(pos[F], 2)
    lo_F = P.points[F, piece]
    hi_F = P.points[F, piece + 1]
    x_h = x.copy()
    x_h[F] = 0.0
    bound = np.flatnonzero(face.row_lo | face.row_hi)
    coef = [sign[F, piece] for sign in prob.cap_sign]
    parts_C = [prob.A_eq[:, F]]
    parts_r = [prob.b_eq - prob.A_eq @ x_h]
    if bound.size:
        G_b = prob.G[bound]
        parts_C.append(G_b[:, F])
        target = np.where(face.row_lo[bound], prob.g_lo[bound], prob.g_hi[bound])
        parts_r.append(target - G_b @ x_h)
    on = np.flatnonzero(face.cap_on)
    for j in on:
        cap = prob.absolute[j]
        held_part = float(np.sum(np.abs(x - cap.centre)[held]))
        parts_C.append(coef[j].reshape(1, -1))
        parts_r.append(
            np.array([cap.limit - held_part + float(coef[j] @ cap.centre[F])])
        )
    C = np.vstack(parts_C)
    r = np.concatenate(parts_r)
    m_w = C.shape[0]
    m_eq = prob.A_eq.shape[0]
    cap_mult = np.zeros(len(prob.absolute))
    if k == 0:
        if m_w:
            raise np.linalg.LinAlgError("a working row with no free weight")
        empty = np.zeros(0)
        return _KKT(F, lo_F, hi_F, empty, empty, empty, cap_mult, bound, coef)
    q = c[F] + kappa[F] * P.trade_sign[F, piece] + H[F] @ x_h
    K = np.zeros((k + m_w, k + m_w))
    K[:k, :k] = H[np.ix_(F, F)]
    K[:k, k:] = C.T
    K[k:, :k] = C
    sol = np.linalg.solve(K, np.concatenate([-q, r]))
    if not np.all(np.isfinite(sol)):
        raise np.linalg.LinAlgError("a KKT solution that is not finite")
    lam = sol[k:]
    cap_mult[on] = lam[m_eq + bound.size :]
    return _KKT(
        F,
        lo_F,
        hi_F,
        sol[:k],
        lam[:m_eq],
        lam[m_eq : m_eq + bound.size],
        cap_mult,
        bound,
        coef,
    )


def _wrong_signs(
    prob: _Problem,
    H: np.ndarray,
    absH: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    x: np.ndarray,
    face: _Face,
    kkt: _KKT,
) -> List[np.ndarray]:
    """
    How wrongly signed each multiplier of `face` is at `x` (the face's
    minimizer), relative to the size of the terms it sums, 0 where it is
    right: a held weight's on the side it would move to -- [0] right, [1]
    left -- then the bound inequality rows' [2] and the binding caps' [3].

    A held weight's multiplier on a side is the smooth gradient plus the
    rows' multipliers plus that side's piece terms: kappa_i and each
    binding cap's multiplier times the piece's sign. Moving right must not
    lower the objective (>= 0) and moving left must not (<= 0); at a kink
    the two sides differ by 2 (kappa_i + rho), which is the no-trade band.
    """
    P = prob.pieces
    n = prob.n
    eps = float(np.finfo(float).eps)
    bound = kkt.bound
    terms_all = absH @ np.abs(x) + np.abs(c) + kappa
    grad_all = H @ x + c
    if prob.A_eq.shape[0]:
        grad_all = grad_all + kkt.lam_eq @ prob.A_eq
        terms_all = terms_all + np.abs(kkt.lam_eq) @ np.abs(prob.A_eq)
    if bound.size:
        grad_all = grad_all + kkt.lam_in @ prob.G[bound]
        terms_all = terms_all + np.abs(kkt.lam_in) @ prob.absG[bound]
    terms_all = terms_all + float(np.sum(np.abs(kkt.cap_mult)))
    terms_all = np.maximum(terms_all, np.finfo(float).tiny)
    Hd = np.flatnonzero(face.held & (P.count > 1))
    b = face.pos[Hd]
    grad = grad_all[Hd]
    terms = terms_all[Hd]
    r_piece = np.minimum(b, 2)
    l_piece = np.maximum(b - 1, 0)
    g_right = grad + kappa[Hd] * P.trade_sign[Hd, r_piece]
    g_left = grad + kappa[Hd] * P.trade_sign[Hd, l_piece]
    for j, sign in enumerate(prob.cap_sign):
        if kkt.cap_mult[j] != 0.0:
            g_right = g_right + kkt.cap_mult[j] * sign[Hd, r_piece]
            g_left = g_left + kkt.cap_mult[j] * sign[Hd, l_piece]
    slack = 64.0 * eps * terms
    wrong_r = np.zeros(n)
    wrong_l = np.zeros(n)
    wrong_r[Hd] = np.where(
        (b < P.count[Hd] - 1) & (g_right < -slack), -g_right / terms, 0.0
    )
    wrong_l[Hd] = np.where((b > 0) & (g_left > slack), g_left / terms, 0.0)
    scale = float(np.max(terms_all))
    wrong_rows = np.zeros(prob.G.shape[0])
    if bound.size:
        width = np.max(prob.absG[bound], axis=1)
        signed = np.where(face.row_lo[bound], kkt.lam_in, -kkt.lam_in) * width
        wrong_rows[bound] = np.where(signed > 64.0 * eps * scale, signed / scale, 0.0)
    wrong_caps = np.where(
        face.cap_on & (kkt.cap_mult < -64.0 * eps * scale),
        -kkt.cap_mult / scale,
        0.0,
    )
    return [wrong_r, wrong_l, wrong_rows, wrong_caps]


def _stopped(reason: str, passes: int) -> _Unsolved:
    """A stage's stop, carrying the passes it spent."""
    failure = _Unsolved(reason)
    failure.passes = passes  # type: ignore[attr-defined]
    return failure


def _pdas(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    face: _Face,
    max_passes: Optional[int] = None,
) -> _Solved:
    """
    The primal-dual active-set method on the pieces (`_active_set.solve`'s
    method, Hintermueller, Ito & Kunisch 2003): solve the KKT system of a
    guessed face, then correct every state at once -- a free weight that
    left its piece is held at the breakpoint it crossed, a held weight
    whose multiplier points off its breakpoint is freed onto that side's
    piece, a row or cap the answer breaks binds and one whose multiplier
    has the wrong sign is released -- until a pass changes nothing: then
    every primal and dual condition holds, and the answer is the optimum.

    Many states change per pass, so it takes a handful of passes where
    the primal method takes one per state. It is not globally convergent:
    a guess seen before, a singular system or the pass limit (default
    max(100, 2n)) raises `_Unsolved`, and the caller falls back to the
    primal method. A guess whose equality rows lack a free weight has
    weights freed for them first (`_support`), and a bound row or cap that
    the others already fix on the free weights is released
    (`_independent`).
    """
    P = prob.pieces
    n = prob.n
    absH = np.abs(H)
    limit = max(100, 2 * n) if max_passes is None else int(max_passes)
    supported = _support(prob, face)
    if supported is None:
        raise _stopped("the primal-dual method's first guess had no free weight", 0)
    face = _independent(prob, supported)
    seen = {face.key()}
    for passes in range(1, limit + 1):
        z = np.where(face.held, P.points[prob.index, face.pos], 0.0)
        try:
            kkt = _face_kkt(prob, H, c, kappa, z, face)
        except np.linalg.LinAlgError:
            raise _stopped(
                f"the primal-dual method's KKT system on pass {passes} was singular",
                passes,
            ) from None
        F = kkt.F
        z[F] = kkt.z
        tiny = STEP_TOLERANCE * max(1.0, float(np.max(np.abs(z))))
        below = np.zeros(n, dtype=bool)
        above = np.zeros(n, dtype=bool)
        below[F] = kkt.z < kkt.lo - tiny
        above[F] = kkt.z > kkt.hi + tiny
        idle = ~(face.row_lo | face.row_hi)
        row_low = np.zeros(prob.G.shape[0], dtype=bool)
        row_high = np.zeros(prob.G.shape[0], dtype=bool)
        if prob.G.shape[0]:
            value = prob.G @ z
            band = STEP_TOLERANCE * np.maximum(1.0, prob.absG @ np.abs(z))
            row_low = idle & (value < prob.g_lo - band)
            row_high = idle & (value > prob.g_hi + band)
        cap_over = np.zeros(len(prob.absolute), dtype=bool)
        for j, cap in enumerate(prob.absolute):
            if not face.cap_on[j]:
                over = cap.value(z) - cap.limit
                cap_over[j] = over > STEP_TOLERANCE * max(1.0, float(np.sum(np.abs(z))))
        wrong = _wrong_signs(prob, H, absH, c, kappa, z, face, kkt)
        changed = (
            below.any()
            or above.any()
            or row_low.any()
            or row_high.any()
            or cap_over.any()
            or any(np.any(w > 0.0) for w in wrong)
        )
        if not changed:
            z[F] = np.clip(kkt.z, kkt.lo, kkt.hi)
            return _Solved(x=z, face=face, passes=passes)
        new = face.copy()
        new.held = face.held | below | above
        new.pos = np.where(above, face.pos + 1, face.pos)
        go_right = wrong[0] > 0.0
        go_left = wrong[1] > 0.0
        new.held = new.held & ~(go_right | go_left)
        new.pos = np.where(go_left, new.pos - 1, new.pos)
        release_rows = wrong[2] > 0.0
        new.row_lo = (face.row_lo & ~release_rows) | row_low
        new.row_hi = (face.row_hi & ~release_rows) | row_high
        new.cap_on = (face.cap_on & ~(wrong[3] > 0.0)) | cap_over
        next_face = _support(prob, new)
        if next_face is None:
            raise _stopped(
                f"the primal-dual method's guess on pass {passes} left an "
                "equality row no free weight",
                passes,
            )
        next_face = _independent(prob, next_face)
        key = next_face.key()
        if key in seen:
            raise _stopped(
                f"the primal-dual method cycled: pass {passes} returned to an "
                "earlier guess",
                passes,
            )
        seen.add(key)
        face = next_face
    raise _stopped(
        f"the primal-dual method found no fixed point within {limit} passes", limit
    )


def _primal(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    x: np.ndarray,
    face: _Face,
    max_passes: Optional[int] = None,
) -> _Solved:
    """
    The primal active-set method on the pieces (see the module), from a
    feasible `x` whose held weights sit at their breakpoints and whose
    working rows have full rank on the free weights. Raises `_Unsolved`
    on a singular system, a cycle or the pass limit.

    The release rule is the most wrongly signed multiplier, relative to
    the size of the terms it sums; if a set of states repeats -- a cycle
    of zero-length steps at a degenerate point, such as a book that trades
    nothing, where every weight is held and the budget's multiplier is any
    value in an interval -- it switches to the lowest-indexed wrong sign
    (Bland's rule), and a repeat after that ends the run.
    """
    P = prob.pieces
    n = prob.n
    absH = np.abs(H)
    caps = prob.absolute
    limit = (
        20 * (n + prob.G.shape[0] + len(caps)) + 100
        if max_passes is None
        else int(max_passes)
    )
    x = x.copy()
    face = face.copy()
    held = face.held
    pos = face.pos
    seen: set = set()
    bland = False
    for passes in range(1, limit + 1):
        try:
            kkt = _face_kkt(prob, H, c, kappa, x, face)
        except np.linalg.LinAlgError:
            # A row or cap the others fix on the free weights holds its
            # value on this face whether it is in the system or not:
            # dropped, the face and the point are the same, and the
            # system is not singular. Nothing to drop is a real failure.
            # An equality row left with no free weight (a weight that had
            # to carry a start's rounding residual held at a bound) gets one
            # that can carry it; the point is unchanged, the face larger.
            fewer = _independent(prob, face)
            fewer = _support(prob, fewer, x) if fewer is not None else None
            if fewer is None or fewer.key() == face.key():
                raise _Unsolved(
                    f"the primal method's KKT system on pass {passes} was singular",
                    _Solved(x=x, face=face, passes=passes),
                ) from None
            face = fewer
            held = face.held
            pos = face.pos
            continue
        F = kkt.F
        k = F.size
        x_F = x[F]
        step = kkt.z - x_F
        tiny = STEP_TOLERANCE * max(1.0, float(np.max(np.abs(x))))
        alpha = np.inf
        which, block = -1, -1
        if k:
            with np.errstate(divide="ignore", invalid="ignore"):
                to_lo = np.where(step < -tiny, (kkt.lo - x_F) / step, np.inf)
                to_hi = np.where(step > tiny, (kkt.hi - x_F) / step, np.inf)
            i_lo = int(np.argmin(to_lo))
            i_hi = int(np.argmin(to_hi))
            if to_lo[i_lo] <= to_hi[i_hi]:
                alpha, which, block = float(to_lo[i_lo]), 0, i_lo
            else:
                alpha, which, block = float(to_hi[i_hi]), 1, i_hi
            # A row or cap blocks only on a move larger than both rounding
            # of the step (a row the working set already fixes) and
            # rounding of the row's own value: a step the rows determine
            # to the last bit -- one free weight under the budget -- moves
            # every row by rounding, and a cap at its limit would otherwise
            # bind on it, parallel to the budget on that weight, and make
            # the next system singular.
            blocking: List[Tuple[float, int, int]] = []
            idle = np.flatnonzero(~(face.row_lo | face.row_hi))
            if idle.size:
                G_i = prob.G[idle]
                value = G_i @ x
                move = G_i[:, F] @ step
                size = STEP_TOLERANCE * np.maximum(
                    np.abs(G_i[:, F]) @ np.abs(step),
                    np.maximum(1.0, prob.absG[idle] @ np.abs(x)),
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    to_rl = np.where(
                        move < -size, (prob.g_lo[idle] - value) / move, np.inf
                    )
                    to_rh = np.where(
                        move > size, (prob.g_hi[idle] - value) / move, np.inf
                    )
                for side, ratios_side in ((2, to_rl), (3, to_rh)):
                    for r in np.flatnonzero(ratios_side < alpha):
                        blocking.append((float(ratios_side[r]), side, int(idle[r])))
            for j, cap in enumerate(caps):
                if face.cap_on[j]:
                    continue
                cap_move = float(kkt.coef[j] @ step)
                floor = STEP_TOLERANCE * max(
                    1.0, float(np.sum(np.abs(step))), cap.limit
                )
                if cap_move > floor:
                    t = (cap.limit - cap.value(x)) / cap_move
                    if t < alpha:
                        blocking.append((t, 4, j))
            # The nearest row or cap that is independent of the working
            # rows on the free weights: one that is not cannot move on
            # this face, whatever rounding says.
            for t, side, item in sorted(blocking):
                if t < alpha and _adds_a_row(prob, face, kkt, side, item):
                    alpha, which, block = t, side, item
                    break
        alpha = max(alpha, 0.0)
        if alpha < 1.0:
            x[F] = x_F + alpha * step
            if which == 0:
                i = int(F[block])
                held[i] = True
                x[i] = P.points[i, pos[i]]
            elif which == 1:
                i = int(F[block])
                held[i] = True
                pos[i] = pos[i] + 1
                x[i] = P.points[i, pos[i]]
            elif which == 2:
                face.row_lo[block] = True
            elif which == 3:
                face.row_hi[block] = True
            else:
                face.cap_on[block] = True
            # rounding must not carry a free weight off its piece
            still = ~held[F]
            x[F[still]] = np.clip(x[F[still]], kkt.lo[still], kkt.hi[still])
            continue

        x[F] = kkt.z
        candidates = _wrong_signs(prob, H, absH, c, kappa, x, face, kkt)
        if not any(np.any(w > 0.0) for w in candidates):
            # a free weight is on its piece: one that rounding carried a
            # step-tolerance past a breakpoint is put back on it
            x[F] = np.clip(x[F], kkt.lo, kkt.hi)
            return _Solved(x=x, face=face, passes=passes)
        if bland:
            which, item = _lowest_wrong(candidates)
        else:
            best = [float(np.max(w, initial=0.0)) for w in candidates]
            which = int(np.argmax(best))
            item = int(np.argmax(candidates[which]))
        if which == 0:
            held[item] = False
        elif which == 1:
            held[item] = False
            pos[item] = pos[item] - 1
        elif which == 2:
            face.row_lo[item] = False
            face.row_hi[item] = False
        else:
            face.cap_on[item] = False
        key = face.key()
        if key in seen:
            if bland:
                raise _Unsolved(
                    f"the primal method cycled through zero-length moves on pass "
                    f"{passes}, under Bland's rule too",
                    _Solved(x=x, face=face, passes=passes),
                )
            bland = True
        seen.add(key)
    raise _Unsolved(
        f"the primal method found no optimum within {limit} passes",
        _Solved(x=x, face=face, passes=limit),
    )


def _adds_a_row(prob: _Problem, face: _Face, kkt: _KKT, which: int, item: int) -> bool:
    """Whether binding inequality row `item` (`which` 2 or 3) or cap
    `item` (4) raises the rank of the working rows on the free weights."""
    F = kkt.F
    rows = [prob.A_eq[:, F]]
    bound = np.flatnonzero(face.row_lo | face.row_hi)
    if bound.size:
        rows.append(prob.G[bound][:, F])
    for j in np.flatnonzero(face.cap_on):
        rows.append(kkt.coef[j].reshape(1, -1))
    working = np.vstack(rows)
    if which in (2, 3):
        candidate = prob.G[item, F].reshape(1, -1)
    else:
        candidate = kkt.coef[item].reshape(1, -1)
    before = int(np.linalg.matrix_rank(working)) if working.size else 0
    return int(np.linalg.matrix_rank(np.vstack([working, candidate]))) > before


def _crash(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    face: _Face,
) -> Optional[_Solved]:
    """
    A guessed face made a feasible start for the primal method: solve its
    KKT system, hold every free weight that left its piece at the
    breakpoint it crossed, bind every row and cap the answer breaks, and
    solve again, until the answer is feasible -- or None after
    `_CRASH_ROUNDS` rounds or on a singular system. Each round only adds
    constraints, so it moves toward feasibility; the primal method then
    releases whatever it held too many of.
    """
    P = prob.pieces
    idx = prob.index
    face = face.copy()
    for _ in range(_CRASH_ROUNDS):
        supported = _support(prob, face)
        if supported is None:
            return None
        face = _independent(prob, supported)
        z = np.where(face.held, P.points[idx, face.pos], 0.0)
        try:
            kkt = _face_kkt(prob, H, c, kappa, z, face)
        except np.linalg.LinAlgError:
            return None
        F = kkt.F
        z[F] = kkt.z
        tiny = STEP_TOLERANCE * max(1.0, float(np.max(np.abs(z))))
        below = np.zeros(prob.n, dtype=bool)
        above = np.zeros(prob.n, dtype=bool)
        below[F] = kkt.z < kkt.lo - tiny
        above[F] = kkt.z > kkt.hi + tiny
        idle = ~(face.row_lo | face.row_hi)
        row_low = np.zeros(prob.G.shape[0], dtype=bool)
        row_high = np.zeros(prob.G.shape[0], dtype=bool)
        if prob.G.shape[0]:
            value = prob.G @ z
            band = STEP_TOLERANCE * np.maximum(1.0, prob.absG @ np.abs(z))
            row_low = idle & (value < prob.g_lo - band)
            row_high = idle & (value > prob.g_hi + band)
        cap_over = np.array(
            [
                (not face.cap_on[j])
                and cap.value(z) > cap.limit + STEP_TOLERANCE * max(1.0, cap.limit)
                for j, cap in enumerate(prob.absolute)
            ],
            dtype=bool,
        )
        if (
            not (below.any() or above.any() or row_low.any() or row_high.any())
            and not cap_over.any()
        ):
            z[F] = np.clip(kkt.z, kkt.lo, kkt.hi)
            if any(
                cap.value(z) > cap.limit * (1.0 + TOLERANCE) for cap in prob.absolute
            ):
                return None
            return _Solved(x=z, face=face, passes=0)
        face.held = face.held | below | above
        face.pos = np.where(above, face.pos + 1, face.pos)
        face.row_lo = face.row_lo | row_low
        face.row_hi = face.row_hi | row_high
        face.cap_on = face.cap_on | cap_over
    return None


def _solve(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    guess: _Face,
    feasible: _Solved,
) -> Tuple[_Solved, str, int]:
    """
    One programme by whichever stage finishes, which it was, and the
    passes it took. Either answer is the KKT solve on its final face, and
    the certificate is the judge.

    Without a cap binding in `guess`: the primal-dual method on the
    programme WITHOUT the turnover and gross caps, from `guess`. With one
    binding it is measured to oscillate -- the free set swinging between
    most of the book and a handful of names as the cap's multiplier widens
    and narrows every no-trade band at once -- and to stop on a singular
    system. Without them it converges in a dozen or two passes where the
    primal method spends two per traded name, and when its answer
    satisfies the caps it is the optimum with them, since it is optimal
    over a larger set. When it does not, a cap binds, and the primal
    method takes over from `feasible`.

    With a cap binding in `guess` (a warm start, or the previous answer of
    a cap search): the primal method, from `feasible` when that is the
    guess's own answer, and otherwise from the guess made feasible
    (`_crash`), falling back to `feasible`.
    """
    reason = "a cap binds in the first guess"
    spent = 0
    starts = [feasible]
    if not guess.cap_on.any():
        relaxed = prob.relaxed()
        start = relaxed.empty_face()
        start.held = guess.held.copy()
        start.pos = guess.pos.copy()
        start.row_lo = guess.row_lo.copy()
        start.row_hi = guess.row_hi.copy()
        try:
            result = _pdas(relaxed, H, c, kappa, start)
            spent = result.passes
            if all(cap.value(result.x) <= cap.limit for cap in prob.absolute):
                face = result.face
                face.cap_on = np.zeros(len(prob.absolute), dtype=bool)
                return _Solved(result.x, face, spent), "primal-dual", spent
            reason = "the answer without the caps breaks one"
        except _Unsolved as failure:
            reason = failure.reason
            spent = int(getattr(failure, "passes", 0))
    elif guess.key() != feasible.face.key():
        crashed = _crash(prob, H, c, kappa, guess)
        if crashed is not None:
            starts = [crashed, feasible]
    for k, begin in enumerate(starts):
        try:
            result = _primal(prob, H, c, kappa, begin.x, begin.face)
            return result, "primal", spent + result.passes
        except _Unsolved as failure:
            best = failure.best
            spent += best.passes if isinstance(best, _Solved) else 0
            reason = f"{reason}; then {failure.reason}"
            if k == len(starts) - 1:
                stopped = _Unsolved(reason, best)
                stopped.passes = spent  # type: ignore[attr-defined]
                raise stopped from None
    raise AssertionError("no start to solve from")


def _lowest_wrong(candidates: Sequence[np.ndarray]) -> Tuple[int, int]:
    """Bland's rule: the wrongly signed multiplier of the lowest index,
    weights before rows before caps."""
    weights = np.flatnonzero((candidates[0] > 0.0) | (candidates[1] > 0.0))
    if weights.size:
        i = int(weights[0])
        return (0, i) if candidates[0][i] > 0.0 else (1, i)
    for which in (2, 3):
        hit = np.flatnonzero(candidates[which] > 0.0)
        if hit.size:
            return which, int(hit[0])
    raise AssertionError("no wrong sign to release")


# ── starts ───────────────────────────────────────────────────────────────


def _face_at(prob: _Problem, x: np.ndarray, snap: float) -> Tuple[np.ndarray, _Face]:
    """The states of a feasible point: a weight at (within `snap` of,
    relative to the point's scale) a breakpoint is held there, set to it
    exactly; the others are free on the piece that holds them. Rows idle,
    caps off."""
    P = prob.pieces
    x = x.copy()
    face = prob.empty_face()
    tol = snap * max(1.0, float(np.max(np.abs(x))))
    columns = np.arange(4)
    real = (columns < P.count[:, None]) & np.isfinite(P.points)
    with np.errstate(invalid="ignore"):
        distance = np.where(real, np.abs(P.points - x[:, None]), np.inf)
    nearest = np.argmin(distance, axis=1)
    near = distance[prob.index, nearest] <= tol
    face.held = near.copy()
    with np.errstate(invalid="ignore"):
        below = np.sum((columns < P.count[:, None]) & (P.points < x[:, None]), axis=1)
    piece = np.clip(below - 1, 0, np.maximum(P.count - 2, 0))
    face.pos = np.where(near, nearest, piece).astype(np.intp)
    x[near] = P.points[prob.index[near], nearest[near]]
    return x, face


def _independent(prob: _Problem, face: _Face) -> _Face:
    """
    The guess with every binding cap and bound inequality row released whose
    coefficients on the free weights combine those of the rows before it
    (equalities, then caps, then inequality rows): its value on the face is
    already fixed by them, it adds no constraint there, and kept it makes
    the KKT system singular. The commonest case is a turnover cap when
    every free weight trades the same way -- buys at their caps, sells
    free -- where on the free weights it is minus the budget row.
    """
    if not (face.cap_on.any() or face.row_lo.any() or face.row_hi.any()):
        return face
    free = ~face.held
    signs = prob.signs(face)
    kept: List[np.ndarray] = [row[free] for row in prob.A_eq]

    def adds(coef: np.ndarray) -> bool:
        if not kept:
            return bool(np.any(coef != 0.0))
        before = np.linalg.matrix_rank(np.array(kept))
        return int(np.linalg.matrix_rank(np.array(kept + [coef]))) > int(before)

    out = face.copy()
    for k in np.flatnonzero(face.cap_on):
        coef = signs[1 + k][free]
        if adds(coef):
            kept.append(coef)
        else:
            out.cap_on[k] = False
    for j in np.flatnonzero(face.row_lo | face.row_hi):
        coef = prob.G[j, free]
        if adds(coef):
            kept.append(coef)
        else:
            out.row_lo[j] = False
            out.row_hi[j] = False
    return out


def _support(
    prob: _Problem, face: _Face, x: Optional[np.ndarray] = None
) -> Optional[_Face]:
    """
    Free held weights until the equality rows have full rank on the free
    weights -- a KKT system needs it -- or None when no choice does.

    A freed weight starts at its breakpoint, and the face's minimizer moves
    it only as the rows make it: by the rows' RESIDUAL at the starting
    point, among other things. A book held to rounding of its budget
    (prices drifted a fully invested book, and its weights sum to one only
    to 1e-13) leaves a residual of that size, and the weight freed must be
    able to absorb it: one at its lower bound cannot give up 1e-13, and
    freed there it is held again at once, leaving the budget no free weight
    and the next system singular. So each weight is freed on the side its
    share of the residual points to, and when one row lacks support the
    weight chosen is one with room on that side. `x` is the point (held
    weights at their breakpoints, free ones at their values); without it
    the free weights count as zero, which is exact for a row with no free
    weight in it.
    """
    A = prob.A_eq
    m = A.shape[0]
    if m == 0:
        return face
    P = prob.pieces
    idx = prob.index
    face = face.copy()
    free = ~face.held
    values = np.where(face.held, P.points[idx, face.pos], 0.0 if x is None else x)
    residual = prob.b_eq - A @ values

    def room(i: int, step: float) -> Optional[int]:
        """The piece a held weight i is freed on to move by `step`, or None
        when no piece next to its breakpoint takes the move."""
        b = int(face.pos[i])
        up = b < P.count[i] - 1
        down = b > 0
        if step > 0.0 and up and P.points[i, b + 1] - P.points[i, b] >= step:
            return b
        if step < 0.0 and down and P.points[i, b] - P.points[i, b - 1] >= -step:
            return b - 1
        if step == 0.0:
            return b if up else (b - 1 if down else None)
        return None

    if m == 1:
        # the common case, a budget alone: one free weight in it will do
        row = A[0]
        if np.any(row[free] != 0.0):
            return face
        candidates = np.flatnonzero(face.held & (P.count > 1) & (row != 0.0))
        if not candidates.size:
            return None
        for i in candidates:
            piece = room(int(i), float(residual[0] / row[i]))
            if piece is not None:
                face.held[i] = False
                face.pos[i] = piece
                return face
        # no weight can take the whole residual alone: the first one, on
        # the side it points to where there is one
        i = int(candidates[0])
        b = int(face.pos[i])
        face.held[i] = False
        face.pos[i] = b if b < P.count[i] - 1 else b - 1
        return face

    def rank(mask: np.ndarray) -> int:
        cols = A[:, mask]
        return int(np.linalg.matrix_rank(cols)) if cols.size else 0

    current = rank(free)
    if current == m:
        return face
    # two-sided weights (held at a kink inside their box) first: they can
    # take a residual of either sign
    held = np.flatnonzero(face.held & (P.count > 1) & np.any(A != 0.0, axis=0))
    interior = (face.pos[held] > 0) & (face.pos[held] < P.count[held] - 1)
    chosen: List[int] = []
    for i in list(held[interior]) + list(held[~interior]):
        trial = free.copy()
        trial[i] = True
        r = rank(trial)
        if r > current:
            chosen.append(int(i))
            free = trial
            current = r
            if current == m:
                break
    if current < m:
        return None
    # each chosen weight on the side the least-squares absorption of the
    # residual by the free weights moves it
    support = np.flatnonzero(free)
    share = np.linalg.lstsq(A[:, support], residual, rcond=None)[0]
    step = dict(zip(support.tolist(), share.tolist()))
    for i in chosen:
        piece = room(i, step[i])
        if piece is None:
            b = int(face.pos[i])
            piece = b if b < P.count[i] - 1 else b - 1
        face.held[i] = False
        face.pos[i] = piece
    return face


def _read_start(
    start: Any,
) -> Tuple[Optional[Dict[str, str]], List[Tuple[str, Optional[str]]], Dict[str, float]]:
    """A warm start as (asset -> state, the (kind, name) of each binding
    row or cap, asset -> weight): from a previous answer, or from its
    `active_set` alone (then with no rows, caps or weights)."""
    if start is None:
        return None, [], {}
    if not isinstance(start, Mapping):
        raise ValidationError(
            f"{_WHO}: start must be a previous answer of optimize_rebalance "
            f"(or its active_set map), got {type(start).__name__}."
        )
    binding: List[Tuple[str, Optional[str]]] = []
    weights: Dict[str, float] = {}
    states: Any = start
    if "active_set" in start and isinstance(start.get("active_set"), Mapping):
        states = start["active_set"]
        for b in start.get("binding_constraints") or []:
            if isinstance(b, Mapping) and "kind" in b:
                name = b.get("name")
                binding.append((str(b["kind"]), None if name is None else str(name)))
        for a, v in dict(start.get("weights") or {}).items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                weights[str(a)] = float(v)
    bad = sorted({str(v) for v in states.values() if v not in REBALANCE_STATES})
    if bad:
        raise ValidationError(
            f"{_WHO}: start has state(s) {bad[:_SHOWN]}; a start is a previous "
            f"answer (or its active_set), whose states are "
            f"{list(REBALANCE_STATES)}."
        )
    return {str(k): str(v) for k, v in states.items()}, binding, weights


def _face_from_states(
    prob: _Problem,
    names: Sequence[str],
    states: Mapping[str, str],
    binding: Sequence[Tuple[str, Optional[str]]] = (),
    weights: Optional[Mapping[str, float]] = None,
) -> _Face:
    """A first guess from a previous answer, by name: each asset's state
    and the rows and caps that bound. An asset it does not name, or whose
    state does not exist here (no current-weight kink to hold at, no bound
    on that side), starts held at its current weight or the nearer bound;
    a row or cap it names that this problem lacks is ignored. A trading
    weight can have two pieces on its side under a gross cap (above and
    below zero): the one holding its previous weight is taken, else the
    one next to its current weight."""
    weights = weights or {}
    P = prob.pieces
    face = prob.empty_face()
    bound = set(binding)
    for j, cap in enumerate(prob.absolute):
        face.cap_on[j] = (cap.kind, None) in bound
    for k, j in enumerate(prob.ineq_index):
        row = prob.rows[j]
        face.row_lo[k] = (f"{row.kind}_min", row.name) in bound
        face.row_hi[k] = (
            ("budget", None) in bound if row.kind == "budget" else False
        ) or (f"{row.kind}_max", row.name) in bound
        face.row_hi[k] = face.row_hi[k] and not face.row_lo[k]
    for i, name in enumerate(names):
        pts = P.points[i, : P.count[i]]
        w0 = P.w0[i]
        state = states.get(name)
        choice: Optional[Tuple[bool, int]] = None
        if P.count[i] == 1:
            choice = (True, 0)
        elif state == "hold":
            hit = np.flatnonzero(pts == w0)
            if hit.size:
                choice = (True, int(hit[0]))
        elif state == "lower" and math.isfinite(pts[0]):
            choice = (True, 0)
        elif state == "upper" and math.isfinite(pts[-1]):
            choice = (True, P.count[i] - 1)
        elif state == "zero":
            hit = np.flatnonzero((pts == 0.0) & (np.arange(pts.size) > 0))
            hit = hit[hit < P.count[i] - 1]
            if hit.size:
                choice = (True, int(hit[0]))
        elif state in ("buy", "sell"):
            sign = 1.0 if state == "buy" else -1.0
            side = [s for s in range(P.count[i] - 1) if P.trade_sign[i, s] == sign]
            if side:
                before = weights.get(name)
                holding = [
                    s
                    for s in side
                    if before is not None and pts[s] <= before <= pts[s + 1]
                ]
                if holding:
                    choice = (False, holding[0] if sign > 0 else holding[-1])
                else:
                    choice = (False, side[0] if sign > 0 else side[-1])
        if choice is None:
            target = min(max(w0, pts[0]), pts[-1])
            hit = np.flatnonzero(pts == target)
            if hit.size:
                choice = (True, int(hit[0]))
            else:
                s = int(np.searchsorted(pts, target)) - 1
                choice = (False, min(max(s, 0), P.count[i] - 2))
        face.held[i], face.pos[i] = choice
    return face


# ── the volatility cap ───────────────────────────────────────────────────


class _Search:
    """
    `general_optimizer._one_cap`'s view of this programme: the objective
    f_weight * f + sum w_k v_k, each v a cap's quadratic. Each solve takes
    the previous answer's states as its guess and the previous answer as
    its feasible fallback start (only the objective changes, so it stays
    feasible); the first takes `guess`, the caller's start states or the
    starting point's own.
    """

    def __init__(
        self,
        prob: _Problem,
        H: np.ndarray,
        c: np.ndarray,
        kappa: np.ndarray,
        start: _Solved,
        guess: _Face,
    ) -> None:
        self.prob = prob
        self.H = H
        self.c = c
        self.kappa = kappa
        self.last = start
        self.guess = guess
        self.solves = 0
        self.passes = 0
        self.stages: List[str] = []

    def solve(
        self, weights: Sequence[Tuple[_CapSpec, float]], f_weight: float
    ) -> _Solved:
        H = f_weight * self.H
        c = f_weight * self.c
        kappa = f_weight * self.kappa
        for spec, w in weights:
            if w:
                H = H + w * 2.0 * spec.S
                c = c - w * 2.0 * (spec.S @ spec.centre)
        try:
            result, stage, spent = _solve(self.prob, H, c, kappa, self.guess, self.last)
        except _Unsolved as failure:
            self.passes += int(getattr(failure, "passes", 0))
            raise
        self.solves += 1
        self.passes += spent
        self.stages.append(stage)
        self.last = result
        self.guess = result.face
        return result


# ── the certificate ──────────────────────────────────────────────────────


def _determined(prob: _Problem, face: _Face) -> np.ndarray:
    """The free weights the working rows alone fix: no direction of the
    face moves them (their row of the projection onto the rows' span is a
    unit vector)."""
    out = np.zeros(prob.n, dtype=bool)
    free = np.flatnonzero(~face.held)
    signs = prob.signs(face)
    A, _b, _bound, _caps = prob.working(face, signs)
    if A.shape[0] == 0 or free.size == 0:
        return out
    M = A[:, free].T
    U, s, _vt = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return out
    r = int(np.sum(s > s[0] * max(M.shape) * np.finfo(float).eps))
    leverage = np.sum(U[:, :r] ** 2, axis=1)
    out[free] = leverage >= 1.0 - 1e-9
    return out


def _snap(prob: _Problem, x: np.ndarray, face: _Face) -> np.ndarray:
    """The answer with every free weight the rows determine, and that lies
    within the certificate's tolerance of a breakpoint of its piece, set to
    that breakpoint (see the module: a book that trades nothing)."""
    P = prob.pieces
    out = x.copy()
    tol = TOLERANCE * max(1.0, float(np.max(np.abs(x))))
    for i in np.flatnonzero(_determined(prob, face)):
        s = int(face.pos[i])
        for p in (P.points[i, s], P.points[i, s + 1]):
            if math.isfinite(p) and abs(x[i] - p) <= tol:
                out[i] = p
                break
    return out


def _certificate(
    prob: _Problem,
    names: Sequence[str],
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    x: np.ndarray,
    face: _Face,
    caps: Sequence[_Cap],
) -> Tuple[Certificate, Any, _Programme]:
    """
    `_certify` on the programme at `x`, from its own data: each free weight
    on its piece, with the cost's and the caps' signs there; a weight at a
    bound with the signs of the piece inside it; a weight held at a kink
    with that kink's terms in its interval (`_Kinks`). Each binding cap is
    the linear row sum s_i (x_i - c_i) <= limit it is at x.
    """
    P = prob.pieces
    n = prob.n
    idx = prob.index
    held = face.held & (P.count > 1)
    at_lo = held & (face.pos == 0)
    at_hi = held & (face.pos == P.count - 1) & ~at_lo
    kink = held & ~at_lo & ~at_hi
    point = P.points[idx, np.minimum(face.pos, 3)]
    inside = np.where(at_lo, 0, np.where(at_hi, P.count - 2, np.minimum(face.pos, 2)))
    inside = np.maximum(inside, 0)
    free = ~face.held

    def sign_of(pieces_sign: np.ndarray, centre: np.ndarray) -> np.ndarray:
        on_piece = pieces_sign[idx, inside]
        return np.where(free | at_lo | at_hi, on_piece, np.sign(point - centre))

    pinned = P.count == 1
    trade = np.where(pinned, 0.0, sign_of(P.trade_sign, P.w0))
    gradient = H @ x + c + kappa * trade
    scale = np.abs(H) @ np.abs(x) + np.abs(c) + kappa
    rows = list(prob.rows)
    m_lin = len(rows)
    row_lo = np.zeros(m_lin + len(prob.absolute), dtype=bool)
    row_hi = np.zeros(m_lin + len(prob.absolute), dtype=bool)
    for k, j in enumerate(prob.ineq_index):
        row_lo[j] = bool(face.row_lo[k])
        row_hi[j] = bool(face.row_hi[k])
    row_width = np.zeros((n, m_lin + len(prob.absolute)))
    for k, (cap, sign) in enumerate(zip(prob.absolute, prob.cap_sign)):
        # a weight fixed by equal bounds counts |x - c| in the cap like any
        # held weight: its sign at its point
        coef = np.where(pinned, np.sign(point - cap.centre), sign_of(sign, cap.centre))
        rows.append(
            _Row(cap.kind, None, coef, -math.inf, cap.limit + float(coef @ cap.centre))
        )
        row_hi[m_lin + k] = bool(face.cap_on[k])
        row_width[:, m_lin + k] = np.where(kink & (point == cap.centre), 1.0, 0.0)
    lower = P.points[idx, 0]
    upper = P.points[idx, P.count - 1]
    prog = _Programme(H, c, rows, lower, upper, list(names))
    kinks = _Kinks(
        held=kink,
        point=point,
        width=np.where(kink & (point == P.w0), kappa, 0.0),
        row_width=row_width,
    )
    certificate, duals = _certify(
        prog,
        x,
        gradient,
        scale,
        active=(at_lo, at_hi, row_lo, row_hi),
        caps=caps,
        kinks=kinks,
    )
    return certificate, duals, prog


# ── the entry point ──────────────────────────────────────────────────────


def optimize_rebalance(
    covariance: Any,
    *,
    assets: Optional[Sequence[str]] = None,
    expected_returns: Any = None,
    current_weights: Any = None,
    risk_aversion: float = 1.0,
    linear_cost: Any = 0.0,
    quadratic_cost: Any = 0.0,
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
    max_turnover: Optional[float] = None,
    max_gross: Optional[float] = None,
    max_volatility: Optional[float] = None,
    start: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The exact cost-aware rebalance from `current_weights`: the weights that
    maximize mu'w - (gamma/2) w'Sw - sum kappa_i |w_i - w0_i|
    - sum eta_i (w_i - w0_i)^2 under the constraints, with their KKT
    certificate and the shadow price of every binding constraint.

    `covariance`, `assets`, `expected_returns` (required), the bounds, the
    budget, groups and exposures are optimize_portfolio's, read by the
    same code. `current_weights` is a name-keyed map or a sequence in asset
    order (an asset left out holds 0; None is all cash). `linear_cost`
    (kappa, per unit of weight traded: 0.0015 is 15 bps) and
    `quadratic_cost` (eta) are a scalar or one per asset, >= 0; a map must
    name every asset. `max_turnover` caps sum |w - w0|, `max_gross` caps
    sum |w|, `max_volatility` caps sqrt(w'Sw) in the covariance's units.
    `start` is a previous answer (or its `active_set` alone), a warm start
    for a nearby problem such as the next rebalance date: it changes how
    many passes the solve takes, not the answer (see the module).

    Returns a dict: `weights`, `trades` (w - w0), `no_trade` (the assets
    whose weight equals the current weight exactly), `turnover`, `gross`,
    `net`, `expected_return`, `volatility`, `linear_cost`,
    `quadratic_cost`, `utility` (the objective, net of both costs),
    `required_turnover` (the least turnover that reaches the constraints;
    0 when the current weights satisfy them), `certified` and
    `certificate`, `binding_constraints` (each with `kind`, `name`,
    `bound` and `shadow_price`, d utility / d bound), `effective_risk_aversion`
    (gamma raised by a binding volatility cap), `active_set`, `method`,
    `passes` (every solve's, summed) and `warnings`. A refusal names the
    argument, the value and the fix; an infeasible set of constraints is
    refused naming the ones that conflict.
    """
    with single_threaded_blas():
        return _rebalance(
            covariance,
            assets=assets,
            expected_returns=expected_returns,
            current_weights=current_weights,
            risk_aversion=risk_aversion,
            linear_cost=linear_cost,
            quadratic_cost=quadratic_cost,
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
            max_turnover=max_turnover,
            max_gross=max_gross,
            max_volatility=max_volatility,
            start=start,
        )


def _cost_vector(value: Any, names: Sequence[str], field_name: str) -> np.ndarray:
    out = _vector(
        value,
        names,
        field_name,
        None,
        _WHO,
        absent_hint="a cost left out is not zero -- give 0.0 for an asset "
        "that trades free, or one number for every asset",
    )
    _finite(out, names, field_name, who=_WHO)
    negative = [names[i] for i in range(len(names)) if out[i] < 0.0]
    if negative:
        raise ValidationError(
            f"{_WHO}: {field_name} is negative for {negative[:_SHOWN]}"
            + (f" and {len(negative) - _SHOWN} more" if len(negative) > _SHOWN else "")
            + f" (e.g. {float(np.min(out)):g}). A cost below zero pays the "
            "book to trade, so the optimum would churn without limit; give "
            "each asset a cost >= 0 (0.0015 is 15 bps per unit traded)."
        )
    return out


def _positive_cap(
    value: Optional[float], field_name: str, meaning: str
) -> Optional[float]:
    if value is None:
        return None
    limit = _scalar(value, field_name, who=_WHO)
    if not limit > 0:
        raise ValidationError(
            f"{_WHO}: {field_name}={limit:g} must be positive: it caps {meaning}. "
            "Leave it out for no cap."
        )
    return limit


def _feasible_start(
    w: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rows: Sequence[_Row],
    absolute: Sequence[_AbsoluteRow],
) -> Optional[np.ndarray]:
    """
    `w` as a starting point, clipped into its box, when it meets every
    bound, row and cap TO ROUNDING -- within the certificate's tolerance of
    the constraint's own scale, the precision the answer is certified to
    -- or None when it breaks one by more (a book that breaks its limits).

    A held book is the last answer drifted by prices, and its weights meet
    the budget, the caps and the bounds only to rounding: a fully invested
    book sums to one within 1e-13, a capped name sits at its cap plus an
    ulp. Such a book is the book on its limits, and is solved as that book
    -- not sent to the linear programme, whose vertex would start the
    solve from somewhere else and report a "required" trade of 1e-16. The
    clip moves a weight past its bound back onto it, a trade of rounding
    size; an equality row's residual is absorbed by the first solve (see
    `_support`).
    """
    scale = np.maximum(1.0, np.abs(w))
    if np.any(lower - w > TOLERANCE * scale) or np.any(w - upper > TOLERANCE * scale):
        return None
    x = np.clip(w, lower, upper)
    for r in rows:
        v = float(r.coef @ x)
        size = max(
            1.0,
            float(np.abs(r.coef) @ np.abs(x)),
            abs(r.lower) if math.isfinite(r.lower) else 0.0,
            abs(r.upper) if math.isfinite(r.upper) else 0.0,
        )
        if v < r.lower - TOLERANCE * size or v > r.upper + TOLERANCE * size:
            return None
    for cap in absolute:
        if cap.value(x) > cap.limit + TOLERANCE * max(1.0, cap.limit):
            return None
    return x


def _rebalance(
    covariance: Any,
    *,
    assets: Optional[Sequence[str]],
    expected_returns: Any,
    current_weights: Any,
    risk_aversion: float,
    linear_cost: Any,
    quadratic_cost: Any,
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
    max_turnover: Optional[float],
    max_gross: Optional[float],
    max_volatility: Optional[float],
    start: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    names, cov = _named_covariance(covariance, assets, _WHO)
    n = len(names)
    if n > MAX_ASSETS:
        raise ValidationError(
            f"{_WHO}: {n} assets, and this optimizer is sized for at most "
            f"{MAX_ASSETS}. Each pass of its exact method is a dense linear "
            "solve over the free assets and binding constraints. Rebalance "
            f"a universe of at most {MAX_ASSETS} (pre-select, or rebalance "
            "sleeves separately)."
        )
    if n < 2:
        raise ValidationError(f"{_WHO}: {n} asset; a rebalance needs at least two.")
    cov = require_finite_covariance(cov, "covariance", _WHO)
    cov = (cov + cov.T) / 2.0
    if expected_returns is None:
        raise ValidationError(
            f"{_WHO}: expected_returns is required, one per asset in the "
            "covariance's units: without forecasts no trade is worth its cost."
        )
    mu = _vector(expected_returns, names, "expected_returns", None, _WHO)
    _finite(mu, names, "expected_returns", who=_WHO)
    if current_weights is None:
        w0 = np.zeros(n)
    else:
        w0 = _vector(current_weights, names, "current_weights", 0.0, _WHO)
        _finite(w0, names, "current_weights", who=_WHO)
    w0 = w0 + 0.0  # a held -0.0 reads as 0.0
    gamma = _scalar(risk_aversion, "risk_aversion", who=_WHO)
    if not gamma > 0:
        raise ValidationError(
            f"{_WHO}: risk_aversion={gamma:g} must be positive; at zero the "
            "objective is linear and its optimum a corner (or unbounded). For "
            "the highest net return under a volatility cap, give a small risk "
            "aversion with max_volatility: the cap's search raises it."
        )
    kappa = _cost_vector(linear_cost, names, "linear_cost")
    eta = _cost_vector(quadratic_cost, names, "quadratic_cost")
    lower, upper = _parse_bounds(names, lower_bounds, upper_bounds, _WHO)
    budget = _scalar(budget, "budget", who=_WHO)
    if budget_type not in ("equal", "at_most"):
        raise ValidationError(
            f"{_WHO}: budget_type={budget_type!r} must be 'equal' (the weights "
            "sum to the budget) or 'at_most' (they sum to at most it)."
        )
    turnover_cap = _positive_cap(
        max_turnover, "max_turnover", "sum |w - w0|, the weight traded"
    )
    gross_cap = _positive_cap(max_gross, "max_gross", "sum |w|, the gross weight")
    vol_cap = _positive_cap(
        max_volatility, "max_volatility", "sqrt(w'Sw), the covariance's units"
    )
    start_states, start_binding, start_weights = _read_start(start)
    rows, group_rows, factor_names, loadings = _constraint_rows(
        names,
        budget=budget,
        budget_type=budget_type,
        groups=groups,
        group_min=group_min,
        group_max=group_max,
        factor_loadings=factor_loadings,
        exposure_min=exposure_min,
        exposure_max=exposure_max,
        who=_WHO,
    )
    upper_used = _drop_redundant_bounds(lower, upper, budget, budget_type)
    free_assets = np.flatnonzero(lower != upper)
    if free_assets.size == 0:
        raise ValidationError(
            f"{_WHO}: every asset's lower bound equals its upper bound, so "
            "there is nothing to rebalance."
        )
    condition, warnings = _require_positive_definite(cov, names, free_assets, _WHO)
    absolute: List[_AbsoluteRow] = []
    if turnover_cap is not None:
        absolute.append(_AbsoluteRow("max_turnover", w0.copy(), turnover_cap))
    if gross_cap is not None:
        absolute.append(_AbsoluteRow("max_gross", np.zeros(n), gross_cap))

    on_limits = _feasible_start(w0, lower, upper_used, rows, absolute)
    if on_limits is not None:
        x0 = on_limits
        # zero, or the rounding-size clip of a weight an ulp past its bound
        required = float(np.sum(np.abs(on_limits - w0)))
        origin = "the current weights"
    else:
        least = _least_absolute_point(rows, lower, upper_used, absolute, w0, who=_WHO)
        if least is None:
            # The least-violation programme names the conflict (and
            # raises); should it find the set feasible within its own
            # tolerance where the exact programme did not, refuse plainly.
            _feasible_point(names, rows, lower, upper_used, absolute=absolute, who=_WHO)
            raise ValidationError(
                f"{_WHO}: no portfolio satisfies these constraints together "
                "to the precision of the feasibility programme. Loosen the "
                "tightest bound, limit or cap."
            )
        x0, required = least
        required = max(required, 0.0)
        origin = "the feasible point nearest the current weights"
        warnings.append(
            f"The current weights break the constraints; reaching them takes "
            f"a turnover of at least {required:.6g}, which every answer here "
            "pays."
        )
    _require_independent_equalities(rows, free_assets, _WHO)

    pieces = _build_pieces(
        lower,
        upper_used,
        w0,
        trade_kink=(kappa > 0.0) | (turnover_cap is not None),
        zero_kink=gross_cap is not None,
    )
    prob = _Problem(pieces, rows, absolute)
    H = gamma * cov + np.diag(2.0 * eta)
    c = -mu - 2.0 * eta * w0

    # The feasible start. The current weights are exact, and a weight held
    # at its current weight must start there to the bit. A linear
    # programme's vertex is exact to about `_START_SNAP`; snapped, it must
    # still meet the limits to rounding, or it starts unsnapped.
    from_book = on_limits is not None
    x_start, face = _face_at(prob, x0, 0.0 if from_book else _START_SNAP)
    if not from_book and (
        _feasible_start(x_start, lower, upper_used, rows, absolute) is None
    ):
        x_start, face = _face_at(prob, x0, 0.0)
    supported = _support(prob, face, x_start)
    if supported is None:
        raise ValidationError(
            f"{_WHO}: the equality limits cannot all be met by the assets "
            "free to move."
        )
    begin = _Solved(x=x_start, face=supported, passes=0)
    # The first guess: the caller's start states, or the start's own.
    guess = supported
    if start_states is not None:
        guess = _face_from_states(
            prob, names, start_states, start_binding, start_weights
        )
        origin = f"the start's states (feasible fallback: {origin})"

    outcome = _solve_with_cap(prob, H, c, kappa, cov, vol_cap, begin, guess)
    return _report(
        names=names,
        cov=cov,
        mu=mu,
        w0=w0,
        gamma=gamma,
        kappa=kappa,
        eta=eta,
        H=H,
        c=c,
        prob=prob,
        vol_cap=vol_cap,
        outcome=outcome,
        origin=origin,
        required=required,
        condition=condition,
        warnings=warnings,
        group_rows=group_rows,
        factor_names=factor_names,
        loadings=loadings,
    )


@dataclass
class _Outcome:
    solved: Optional[_Solved]
    nu: float
    solves: int
    passes: int
    notes: List[str]
    unsolved: Optional[str] = None
    stages: List[str] = field(default_factory=list)


def _solve_with_cap(
    prob: _Problem,
    H: np.ndarray,
    c: np.ndarray,
    kappa: np.ndarray,
    cov: np.ndarray,
    vol_cap: Optional[float],
    begin: _Solved,
    guess: _Face,
) -> _Outcome:
    notes: List[str] = []
    search = _Search(prob, H, c, kappa, begin, guess)
    try:
        if vol_cap is None:
            final = search.solve([], 1.0)
            return _Outcome(
                final, 0.0, search.solves, search.passes, notes, None, search.stages
            )
        spec = _CapSpec("max_volatility", cov, np.zeros(prob.n), vol_cap)
        floor = search.solve([(spec, 1.0)], 0.0)
        lowest = _value(spec, floor.x)
        if lowest > vol_cap**2 * (1.0 + TOLERANCE):
            raise _cap_floor_refusal(spec, lowest, _WHO)
        plain = search.solve([], 1.0)
        if _value(spec, plain.x) <= vol_cap**2:
            notes.append(
                f"max_volatility={vol_cap:g} does not bind: the optimum without "
                f"it has volatility {math.sqrt(_value(spec, plain.x)):.6g}."
            )
            return _Outcome(
                plain, 0.0, search.solves, search.passes, notes, None, search.stages
            )
        final, nu = _one_cap(search, spec, [], 1.0)
        return _Outcome(
            final, nu, search.solves, search.passes, notes, None, search.stages
        )
    except _Unsolved as failure:
        last = failure.best if isinstance(failure.best, _Solved) else None
        return _Outcome(
            last,
            0.0,
            search.solves,
            search.passes,
            notes,
            unsolved=failure.reason,
            stages=search.stages,
        )


def _state_names(prob: _Problem, x: np.ndarray, face: _Face) -> List[str]:
    P = prob.pieces
    out = []
    for i in range(prob.n):
        if P.count[i] == 1:
            out.append("fixed")
        elif x[i] == P.w0[i]:
            out.append("hold")
        elif face.held[i] and face.pos[i] == 0:
            out.append("lower")
        elif face.held[i] and face.pos[i] == P.count[i] - 1:
            out.append("upper")
        elif face.held[i] and x[i] == 0.0:
            out.append("zero")
        else:
            out.append("buy" if x[i] > P.w0[i] else "sell")
    return out


def _report(
    *,
    names: Sequence[str],
    cov: np.ndarray,
    mu: np.ndarray,
    w0: np.ndarray,
    gamma: float,
    kappa: np.ndarray,
    eta: np.ndarray,
    H: np.ndarray,
    c: np.ndarray,
    prob: _Problem,
    vol_cap: Optional[float],
    outcome: _Outcome,
    origin: str,
    required: float,
    condition: float,
    warnings: List[str],
    group_rows: Mapping[str, np.ndarray],
    factor_names: Sequence[str],
    loadings: np.ndarray,
) -> Dict[str, Any]:
    warnings = list(warnings) + list(outcome.notes)
    result: Dict[str, Any] = {
        "assets": list(names),
        "n_assets": len(names),
        "risk_aversion": gamma,
        "required_turnover": float(required),
        "condition_number": condition,
    }
    solved = outcome.solved
    if solved is None:
        warnings.append(
            f"The exact solve did not reach an answer: {outcome.unsolved}. No "
            "weights are returned."
        )
        result.update(
            {
                "weights": {},
                "trades": {},
                "no_trade": [],
                "turnover": None,
                "gross": None,
                "net": None,
                "expected_return": None,
                "volatility": None,
                "linear_cost": None,
                "quadratic_cost": None,
                "utility": None,
                "certified": False,
                "certificate": None,
                "binding_constraints": [],
                "effective_risk_aversion": None,
                "group_totals": {},
                "exposures": {},
                "active_set": {},
                "method": f"NOT solved: {outcome.unsolved}",
                "passes": outcome.passes,
                "warnings": warnings,
            }
        )
        return result

    nu = outcome.nu
    caps: List[_Cap] = []
    if vol_cap is not None and nu > 0.0 and math.isfinite(nu):
        caps.append(_Cap("max_volatility", cov, np.zeros(prob.n), vol_cap, nu))
    face = solved.face
    raw = solved.x
    snapped = _snap(prob, raw, face)
    certificate, duals, prog = _certificate(
        prob, names, H, c, kappa, snapped, face, caps
    )
    w = snapped
    if not certificate.certified and not np.array_equal(snapped, raw):
        before, duals_raw, prog_raw = _certificate(
            prob, names, H, c, kappa, raw, face, caps
        )
        if before.certified:
            certificate, duals, prog, w = before, duals_raw, prog_raw, raw
    # An equality row the answer misses by more than one solve's rounding
    # misses it by the held book's own residual: every weight that did not
    # trade is the weight held, to the bit, and those weights carry the
    # book's rounding (a drifted book sums to its budget within 1e-13).
    # Moving one name by that much to hide it would be a trade nobody
    # asked for and would break that name's no-trade guarantee; it is
    # reported instead.
    for j in prob.eq_index:
        row = prob.rows[j]
        miss = float(row.coef @ w) - row.lower
        size = max(1.0, float(np.abs(row.coef) @ np.abs(w)))
        if abs(miss) > 64.0 * float(np.finfo(float).eps) * size:
            what = "the budget" if row.kind == "budget" else f"{row.kind} {row.name!r}"
            warnings.append(
                f"The weights meet {what} only to {miss:+.1e}: the current "
                "weights given miss it by about that much (rounding in the "
                "book held), and the names held at them keep it bit for bit. "
                "Scale the book onto its budget for an exact sum."
            )
    certified = bool(certificate.certified and outcome.unsolved is None)
    if outcome.unsolved is not None:
        warnings.append(
            f"The exact solve did not reach a certified optimum: "
            f"{outcome.unsolved}. The weights below are its last feasible "
            "point, NOT a certified optimum."
        )
    elif not certificate.certified:
        warnings.append(
            "The answer does not certify: "
            + ", ".join(certificate.failed())
            + f" over the tolerance {TOLERANCE:.0e}. It is not a verified optimum."
        )

    m_lin = len(prob.rows)
    linear_prog = _Programme(
        prog.Q, prog.c, prog.rows[:m_lin], prog.lower, prog.upper, prog.names
    )
    binding_raw = _binding_from(linear_prog, duals, w)
    for k, cap in enumerate(prob.absolute):
        if face.cap_on[k]:
            binding_raw.append(
                (cap.kind, None, cap.limit, -float(duals.rows[m_lin + k]))
            )
    effective: Optional[float] = gamma
    if vol_cap is not None and nu > 0.0:
        if math.isfinite(nu):
            binding_raw.append(("max_volatility", None, vol_cap, -2.0 * nu * vol_cap))
            effective = gamma + 2.0 * nu
        else:
            binding_raw.append(("max_volatility", None, vol_cap, None))
            effective = None
    binding = [
        {
            "kind": kind,
            "name": name,
            "bound": None if bound is None else float(bound),
            # the utility is minus the minimized quantity; + 0.0 so a zero
            # price reads 0.0, not -0.0
            "shadow_price": None if d is None else float(-d) + 0.0,
        }
        for kind, name, bound, d in binding_raw
    ]
    binding.sort(
        key=lambda b: (
            -(abs(b["shadow_price"]) if b["shadow_price"] is not None else math.inf),
            b["kind"],
            str(b["name"]),
        )
    )

    trades = w - w0
    traded = np.abs(trades)
    variance = float(w @ cov @ w)
    linear = float(kappa @ traded)
    quadratic = float(eta @ (trades * trades))
    expected = float(mu @ w)
    utility = expected - 0.5 * gamma * variance - linear - quadratic
    states = _state_names(prob, w, face)
    final_stage = outcome.stages[-1] if outcome.stages else "primal"
    method = (
        f"the {final_stage} active-set method on each weight's pieces (held "
        "at the current weight, at zero or at a bound, or trading), started "
        f"from {origin}"
    )
    if caps or (vol_cap is not None and nu > 0.0):
        method += (
            f"; the volatility cap's multiplier by Brent's method in "
            f"{outcome.solves} solves"
        )
    result.update(
        {
            "weights": {a: float(v) for a, v in zip(names, w)},
            "trades": {a: float(v) + 0.0 for a, v in zip(names, trades)},
            "no_trade": [a for a, v, v0 in zip(names, w, w0) if v == v0],
            "turnover": float(np.sum(traded)),
            "gross": float(np.sum(np.abs(w))),
            "net": float(np.sum(w)),
            "expected_return": expected,
            "volatility": math.sqrt(max(variance, 0.0)),
            "linear_cost": linear,
            "quadratic_cost": quadratic,
            "utility": utility,
            "certified": certified,
            "certificate": certificate.as_dict(),
            "binding_constraints": binding,
            "effective_risk_aversion": effective,
            "group_totals": {g: float(coef @ w) for g, coef in group_rows.items()},
            "exposures": {
                f: float(loadings[:, k] @ w) for k, f in enumerate(factor_names)
            },
            "active_set": dict(zip(names, states)),
            "method": method,
            "passes": outcome.passes,
            "warnings": warnings,
        }
    )
    return result
