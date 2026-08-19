"""
Portfolio optimization: produces weights, rather than just scoring weights
someone else already picked (portfolio.py's portfolio_metrics) or converting
an existing alpha score into weights (backtest/sizing.py). Three families:

- `mean_variance_optimize` — classic Markowitz mean-variance on four
  objectives (max_sharpe, min_volatility, target_return, target_volatility).
  The unconstrained case (allow_short=True, max_weight=None) is solved in
  closed form via the standard two-fund efficient-frontier parametrization
  (Merton 1972) — numpy only, no solver dependency. Long-only and/or
  bounded-weight requests fall back to scipy.optimize (SLSQP), matching
  this codebase's existing "scipy optional, clear error if needed and
  missing" convention (see metrics.risk_metrics.var_parametric).
- `risk_parity_weights` — equalizes each asset's fractional contribution to
  total portfolio variance (or a custom risk budget) via a damped
  multiplicative fixed-point iteration. This is a documented heuristic, not
  a globally-convergence-proven algorithm (unlike the closed-form
  mean-variance path) — `converged` is reported honestly rather than
  assumed, and callers should check it.
- `black_litterman` — combines a market-equilibrium prior with explicit
  investor views into posterior expected returns/covariance (He & Litterman
  1999). `build_bl_views` is a convenience that turns a plain-dict view list
  into the (P, Q, Omega) matrices this function expects.

Deliberately Pydantic-free (unlike agent/tools.py) so this module is usable
directly with plain numpy/pandas, the same design choice backtest/sizing.py
and analysis/*.py already make.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_OBJECTIVES = frozenset(
    {"max_sharpe", "min_volatility", "target_return", "target_volatility"}
)

try:
    from scipy.optimize import minimize as _scipy_minimize

    HAS_SCIPY = True
except ImportError:
    _scipy_minimize = None
    HAS_SCIPY = False


def _mean_cov(
    returns_df: pd.DataFrame, periods_per_year: int
) -> Tuple[np.ndarray, np.ndarray]:
    mu = returns_df.mean().to_numpy(dtype=float) * periods_per_year
    cov = returns_df.cov().to_numpy(dtype=float) * periods_per_year
    return mu, cov


def _require_finite_scalar(name: str, value: Any) -> float:
    """
    Reject a non-finite or non-numeric scalar before it can reach a
    comparison.

    Order matters. Every domain guard in this module is written as a
    comparison (`<= 0`, `< min_var`), and NaN makes all of them False — so a
    NaN was never rejected by the check that existed to reject bad values;
    it simply flowed through into the covariance algebra and out the other
    side as NaN weights.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name} must be a number, got {type(value).__name__} ({value!r})"
        )
    number = float(value)
    if not np.isfinite(number):
        raise ValidationError(
            f"{name} must be finite, got {value!r}. NaN compares False against "
            "every bound in this module, so it would pass each guard and "
            "produce NaN weights reported as a converged solution."
        )
    return number


def _check_covariance_estimable(n_obs: int, n_assets: int, cov: np.ndarray) -> None:
    """
    Reject a sample covariance that cannot support an optimization, BEFORE
    either solver runs.

    A sample covariance built from `n_obs` observations of `n_assets` assets
    has rank at most `n_obs - 1`, so with n_obs <= n_assets it is singular by
    construction. The closed-form path noticed (np.linalg.inv raises), but
    the scipy path inverts nothing and therefore did not: SLSQP simply found
    a direction in the covariance's NULL SPACE and reported it as a
    zero-variance portfolio. Measured on 5 observations of 6 assets:
    expected_volatility 1.19e-07, in-sample w'Sigma w = 1.4e-14, and
    converged=True -- for a portfolio whose actual out-of-sample volatility
    was 23.1%. A risk-free-looking answer that is purely an artifact of not
    having enough data to estimate risk at all.

    Checked once here, for every objective and both solver paths, so the two
    cannot disagree about whether an input is solvable.
    """
    if n_obs <= n_assets:
        raise ValidationError(
            f"{n_obs} observations for {n_assets} assets cannot estimate a "
            f"covariance matrix: its rank is at most {n_obs - 1}, so it is "
            f"singular by construction and an optimizer can find a "
            f"zero-variance portfolio in its null space that carries real "
            f"risk out of sample. Need strictly more observations than "
            f"assets (>{n_assets}); prefer many more — see the small-sample "
            "warning this function returns."
        )
    rank = int(np.linalg.matrix_rank(cov))
    if rank < n_assets:
        raise ValidationError(
            f"covariance matrix is rank-deficient (rank {rank} < {n_assets} "
            "assets) even though there are enough observations — two or more "
            "assets are perfectly collinear over this window (e.g. a "
            "duplicated ticker, or a share class that tracks another "
            "exactly). Drop the redundant asset(s); an optimizer would "
            "otherwise report a zero-variance portfolio built from the "
            "collinear direction."
        )


def _small_sample_warnings(n_obs: int, n_assets: int) -> List[str]:
    """
    A covariance can be technically invertible and still be worthless.

    Same generating process, min_volatility, 5 assets: 6 observations report
    an annualized volatility of 0.0039 where 250 observations report 0.1376
    -- a ~22x understatement, with converged=True and nothing to distinguish
    the two. The usual rule of thumb is that a sample covariance needs on the
    order of 10 observations per asset before its off-diagonal terms mean
    much; below that the optimizer is fitting estimation noise, and
    mean-variance is famously good at loading up on exactly that noise.

    A warning, not an error: short windows are a legitimate thing to ask
    for, and the caller may know something the estimator does not.
    """
    warnings: List[str] = []
    if n_obs < 10 * n_assets:
        warnings.append(
            f"only {n_obs} observations for {n_assets} assets "
            f"({n_obs / n_assets:.1f} per asset). A sample covariance needs "
            "roughly 10 observations per asset before its off-diagonal terms "
            "are meaningful; below that the optimizer largely fits estimation "
            "noise and will concentrate in whichever asset pair happens to "
            "look most diversifying. Treat these weights as indicative."
        )
    return warnings


def _require_scipy(context: str) -> None:
    if not HAS_SCIPY:
        raise ValidationError(
            f"{context} requires scipy, which is not installed. Install scipy, "
            "or use allow_short=True with max_weight=None, which has a "
            "closed-form solution needing only numpy."
        )


def _frontier_stats(
    mu: np.ndarray, cov: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Merton (1972) two-fund efficient-frontier constants for an unconstrained
    (sum(w)=1, no bounds) portfolio: A = 1'Sigma^-1 1, B = 1'Sigma^-1 mu,
    C = mu'Sigma^-1 mu, D = A*C - B^2. Any point on the frontier is fully
    determined by these plus a target return — see _frontier_weights.
    """
    try:
        sigma_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        raise ValidationError(
            f"covariance matrix is singular/near-singular — cannot solve the "
            f"unconstrained efficient frontier: {e}"
        ) from e
    ones = np.ones(len(mu))
    A = float(ones @ sigma_inv @ ones)
    B = float(ones @ sigma_inv @ mu)
    C = float(mu @ sigma_inv @ mu)
    D = A * C - B * B
    if A <= 0 or abs(D) < 1e-14:
        raise ValidationError(
            "degenerate mean/covariance inputs (A<=0 or D~0) — cannot solve "
            "the unconstrained efficient frontier"
        )
    return sigma_inv, ones, A, B, C, D


def _frontier_weights(
    sigma_inv: np.ndarray,
    ones: np.ndarray,
    mu: np.ndarray,
    A: float,
    B: float,
    C: float,
    D: float,
    target_return: float,
) -> np.ndarray:
    """w(r) = Sigma^-1[(C - B*r)*1 + (A*r - B)*mu] / D — sums to 1 for any r
    (algebraically: 1'w(r) = [(C-Br)A + (Ar-B)B]/D = (AC-B^2)/D = 1)."""
    lam = (C - B * target_return) / D
    gam = (A * target_return - B) / D
    return sigma_inv @ (lam * ones + gam * mu)


def _solve_unconstrained(
    mu: np.ndarray,
    cov: np.ndarray,
    objective: str,
    risk_free_rate: float,
    target_return: Optional[float],
    target_volatility: Optional[float],
) -> np.ndarray:
    sigma_inv, ones, A, B, C, D = _frontier_stats(mu, cov)

    if objective == "min_volatility":
        return _frontier_weights(sigma_inv, ones, mu, A, B, C, D, B / A)

    if objective == "max_sharpe":
        # Tangency portfolio: w ∝ Sigma^-1(mu - rf*1), normalized to sum 1.
        raw = sigma_inv @ (mu - risk_free_rate * ones)
        denom = float(ones @ raw)  # == B - rf*A
        if abs(denom) < 1e-14:
            raise ValidationError(
                "tangency portfolio is degenerate (sum of raw weights ~ 0) — "
                "try a different risk_free_rate"
            )
        # The sign matters, and only checking the magnitude was the bug.
        #
        # Normalizing by 1'w gives a portfolio whose excess return is
        # (mu-rf)'Sigma^-1(mu-rf) / denom. The numerator is a quadratic form
        # in a positive-definite Sigma, so it is ALWAYS positive -- meaning
        # the sign of the excess return is entirely the sign of denom. With
        # denom < 0 the normalization flips the tangency solution onto the
        # INEFFICIENT branch of the frontier, and an objective named
        # max_sharpe returned the minimum-Sharpe portfolio with
        # converged=True. Measured on mu=[0.10,0.08],
        # Sigma=[[.04,.01],[.01,.05]], rf=0.20: Sharpe -0.66.
        #
        # denom = B - rf*A, so this is exactly the condition rf >= B/A: a
        # risk-free rate at or above the global minimum-variance portfolio's
        # own expected return. There is then no tangency portfolio on the
        # efficient branch at all -- the supremum of Sharpe over
        # {w : 1'w = 1} is not attained -- so this is reported rather than
        # approximated.
        if denom < 0:
            min_var_return = B / A
            raise ValidationError(
                f"no maximum-Sharpe portfolio exists for risk_free_rate="
                f"{risk_free_rate:.6f}: it is at or above the global "
                f"minimum-variance portfolio's expected return "
                f"({min_var_return:.6f}), so every fully-invested portfolio on "
                "the efficient branch has negative excess return and the "
                "Sharpe supremum is not attained. Use a risk_free_rate below "
                f"{min_var_return:.6f}, or objective='min_volatility'. "
                "(Bounded requests — allow_short=False and/or max_weight set "
                "— do have a solution here, since bounds make the feasible "
                "set compact; this restriction is specific to the "
                "unconstrained closed form.)"
            )
        return raw / denom

    if objective == "target_return":
        assert target_return is not None  # validated by caller
        return _frontier_weights(sigma_inv, ones, mu, A, B, C, D, target_return)

    if objective == "target_volatility":
        assert target_volatility is not None  # validated by caller
        min_var = 1.0 / A
        if target_volatility**2 < min_var - 1e-12:
            raise ValidationError(
                f"target_volatility={target_volatility:.6f} is below the global "
                f"minimum-variance portfolio's volatility ({np.sqrt(min_var):.6f}) "
                "— no feasible unconstrained portfolio at this risk level"
            )
        discriminant = max(D * (A * target_volatility**2 - 1.0), 0.0)
        r = (B + np.sqrt(discriminant)) / A  # upper (efficient) branch
        return _frontier_weights(sigma_inv, ones, mu, A, B, C, D, r)

    raise AssertionError(
        f"unreachable objective {objective!r}"
    )  # _OBJECTIVES pre-validated


def _solve_constrained(
    mu: np.ndarray,
    cov: np.ndarray,
    objective: str,
    risk_free_rate: float,
    target_return: Optional[float],
    target_volatility: Optional[float],
    allow_short: bool,
    max_weight: Optional[float],
) -> Tuple[np.ndarray, bool]:
    n = len(mu)
    bound = max_weight if max_weight is not None else 1.0
    bounds = [(-bound, bound)] * n if allow_short else [(0.0, bound)] * n
    x0 = np.full(n, 1.0 / n)
    constraints: List[Dict[str, Any]] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    if objective == "min_volatility":

        def obj_fn(w: np.ndarray) -> float:
            return float(w @ cov @ w)

    elif objective == "max_sharpe":

        def obj_fn(w: np.ndarray) -> float:
            vol = float(np.sqrt(w @ cov @ w))
            if vol < 1e-12:
                return 0.0
            return -(float(w @ mu) - risk_free_rate) / vol

    elif objective == "target_return":
        assert target_return is not None

        def obj_fn(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        constraints.append(
            {"type": "eq", "fun": lambda w: float(w @ mu) - target_return}
        )

    elif objective == "target_volatility":
        assert target_volatility is not None

        def obj_fn(w: np.ndarray) -> float:
            return -float(w @ mu)  # maximize return

        constraints.append(
            {"type": "eq", "fun": lambda w: float(w @ cov @ w) - target_volatility**2}
        )
    else:
        raise AssertionError(
            f"unreachable objective {objective!r}"
        )  # _OBJECTIVES pre-validated

    # _solve_constrained is only ever called after _require_scipy() has
    # confirmed _scipy_minimize is not None.
    result = _scipy_minimize(  # type: ignore[misc]
        obj_fn,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return result.x, bool(result.success)


def mean_variance_optimize(
    returns_df: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    allow_short: bool = False,
    max_weight: Optional[float] = None,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Markowitz mean-variance optimization over one of four objectives.

    Args:
        returns_df: Per-period returns, one column per asset.
        objective: "max_sharpe" (tangency portfolio), "min_volatility"
            (global minimum-variance portfolio), "target_return" (min
            variance for a given annualized return), or "target_volatility"
            (max return for a given annualized volatility).
        risk_free_rate: Annualized rate, used by "max_sharpe" only.
        target_return: Required (annualized) for objective="target_return".
        target_volatility: Required (annualized) for objective="target_volatility".
        allow_short: If False (default), weights are constrained to [0, max_weight].
            If True, weights may be negative.
        max_weight: Per-asset weight cap. None (default) means uncapped in
            the allow_short=True case (closed-form, numpy only) or capped at
            1.0 in the allow_short=False case. Setting this (with either
            allow_short value) requires scipy.
        periods_per_year: Annualization factor for returns_df's own frequency.

    Returns:
        Dict with tickers, weights (dict ticker->float), expected_return,
        expected_volatility, sharpe_ratio, objective, converged (bool —
        always True for the closed-form path; reflects the solver's own
        success flag for the scipy path), and warnings (list of str —
        currently the small-sample caveat; empty when the window is long
        enough relative to the asset count).

    Raises:
        ValidationError: unknown objective, fewer than 2 assets, fewer than
            2 observations, non-finite (inf) returns, observations not
            exceeding the asset count (the covariance would be singular by
            construction), a rank-deficient covariance from collinear
            assets, a required target_return/target_volatility missing, an
            infeasible max_weight for the given asset count, a
            singular/degenerate covariance matrix, a risk_free_rate at or
            above the minimum-variance return for the unconstrained
            max_sharpe objective, or (constrained path only) scipy not
            installed.
    """
    if objective not in _OBJECTIVES:
        raise ValidationError(
            f"objective must be one of {sorted(_OBJECTIVES)}, got {objective!r}"
        )
    # Every scalar is checked for finiteness BEFORE any comparison, because
    # NaN satisfies no comparison at all: `if target_volatility <= 0` is
    # False for NaN, so a NaN sailed past its own guard, poisoned mu/cov and
    # came back as {ticker: nan} weights reported with converged=True — a
    # success flag on a result containing no numbers. Verified for
    # risk_free_rate, target_return and target_volatility alike.
    _require_finite_scalar("risk_free_rate", risk_free_rate)
    if target_return is not None:
        _require_finite_scalar("target_return", target_return)
    if target_volatility is not None:
        _require_finite_scalar("target_volatility", target_volatility)
    if max_weight is not None:
        _require_finite_scalar("max_weight", max_weight)
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year < 1
    ):
        raise ValidationError(
            f"periods_per_year must be a positive integer, got "
            f"{periods_per_year!r}. It is the annualization factor for this "
            "return frequency (252 daily, 52 weekly, 12 monthly); zero or "
            "negative makes every annualized quantity meaningless rather "
            "than merely scaled."
        )
    if returns_df.shape[1] < 2:
        raise ValidationError(f"need at least 2 assets, got {returns_df.shape[1]}")
    data = returns_df.dropna()
    if data.shape[0] < 2:
        raise ValidationError(
            "need at least 2 observations after dropping rows with NaN"
        )
    # dropna() removes NaN but NOT +/-inf, so an infinite return propagated
    # straight through mean()/cov() into the solver and came back as
    # {ticker: nan} weights with converged=True -- a success flag on a
    # result containing no numbers.
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        bad_cols = [
            str(c)
            for c, ok in zip(data.columns, np.isfinite(values).all(axis=0))
            if not ok
        ]
        raise ValidationError(
            f"returns contain {n_bad} non-finite value(s) (inf/-inf) in "
            f"column(s) {bad_cols}. dropna() removes NaN but not infinities, "
            "so these would reach the optimizer and produce NaN weights "
            "reported as a converged solution. A common cause is a zero or "
            "negative price in the source series feeding pct_change."
        )
    if objective == "target_return" and target_return is None:
        raise ValidationError("objective='target_return' requires target_return")
    if objective == "target_volatility":
        if target_volatility is None:
            raise ValidationError(
                "objective='target_volatility' requires target_volatility"
            )
        if target_volatility <= 0:
            raise ValidationError(
                f"target_volatility must be > 0, got {target_volatility}"
            )

    n = data.shape[1]
    if max_weight is not None:
        if max_weight <= 0:
            raise ValidationError(f"max_weight must be > 0, got {max_weight}")
        # The upper bound is max_weight per asset whether or not shorting is
        # allowed, so sum(w) <= n * max_weight either way and the constraint
        # sum(w) == 1 is infeasible when n * max_weight < 1. Restricting this
        # check to the long-only case let allow_short=True through: with
        # n=2, max_weight=0.3 SLSQP returned weights summing to 0.6 --
        # violating the one constraint the whole problem is defined by --
        # flagged only by converged=False, while the identical infeasibility
        # raised a clear error on the long-only path.
        if max_weight * n < 1.0 - 1e-9:
            side = "long-only" if not allow_short else "shorting-allowed"
            raise ValidationError(
                f"max_weight={max_weight} is infeasible for {n} {side} assets "
                f"summing to 1.0 (need max_weight >= {1.0 / n:.4f}). Shorting "
                "does not help: it lowers the per-asset floor, not the cap, so "
                "the weights still cannot reach a sum of 1."
            )

    _check_covariance_estimable(data.shape[0], n, _mean_cov(data, 1)[1])
    warnings = _small_sample_warnings(data.shape[0], n)

    mu, cov = _mean_cov(data, periods_per_year)
    tickers = list(data.columns)

    logger.debug(
        "[portfolio_optimize] objective=%s  assets=%d  allow_short=%s  max_weight=%s",
        objective,
        n,
        allow_short,
        max_weight,
    )

    if allow_short and max_weight is None:
        w = _solve_unconstrained(
            mu, cov, objective, risk_free_rate, target_return, target_volatility
        )
        converged = True
    else:
        _require_scipy(
            "constrained mean-variance optimization (allow_short=False and/or max_weight set)"
        )
        w, converged = _solve_constrained(
            mu,
            cov,
            objective,
            risk_free_rate,
            target_return,
            target_volatility,
            allow_short,
            max_weight,
        )

    if not converged:
        # The solver's iterate is still returned (callers who only want a
        # starting point can use it), but it is NOT guaranteed to satisfy
        # sum(w)==1 or the bounds — say so loudly rather than leaving that
        # buried in a boolean the caller may not read.
        logger.warning(
            "[portfolio_optimize] SLSQP did not converge for objective=%s — "
            "returned weights may violate the sum-to-1 constraint (actual "
            "sum: %.6f) and/or the weight bounds. Check result['converged'].",
            objective,
            float(np.sum(w)),
        )

    exp_ret = float(w @ mu)
    exp_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (exp_ret - risk_free_rate) / exp_vol if exp_vol > 1e-12 else 0.0

    return {
        "tickers": tickers,
        "weights": {t: float(wi) for t, wi in zip(tickers, w)},
        "expected_return": exp_ret,
        "expected_volatility": exp_vol,
        "sharpe_ratio": sharpe,
        "objective": objective,
        "converged": converged,
        "warnings": warnings,
    }


def risk_parity_weights(
    cov_matrix: np.ndarray,
    risk_budget: Optional[np.ndarray] = None,
    max_iterations: int = 1000,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Equal (or custom-budgeted) risk contribution portfolio via a damped
    multiplicative fixed-point iteration: at each step, scale each weight by
    sqrt(target_risk_contribution / current_risk_contribution), then
    renormalize to sum 1. At a fixed point this exactly satisfies
    w_i * (Sigma@w)_i == budget_i * (w'Sigma w) for every i — the standard
    risk-parity condition — regardless of the path taken to get there.

    This is a documented heuristic, not a globally-convergence-proven
    algorithm (unlike mean_variance_optimize's closed-form path) — it
    converges reliably in practice for well-conditioned covariance matrices
    (verified here: a diagonal covariance converges to the closed-form
    inverse-volatility weights), but `converged` reflects whether the
    iteration actually reached `tol` within `max_iterations`, not an assumption.

    Args:
        cov_matrix: (n, n) covariance matrix (annualized or not — risk
            contributions are scale-invariant fractions either way).
        risk_budget: (n,) target fractional risk contribution per asset,
            must be positive and sum to 1.0. None (default) means equal
            risk contribution (1/n each).
        max_iterations: Iteration cap.
        tol: Convergence tolerance on the max per-asset weight change
            between iterations.

    Returns:
        Dict with weights (np.ndarray), risk_contributions (np.ndarray,
        fractional, sums to 1), converged (bool), iterations_used (int).

    Raises:
        ValidationError: cov_matrix isn't square, risk_budget's length
            doesn't match, risk_budget isn't positive/doesn't sum to 1, or
            portfolio variance is non-positive at any iteration (cov_matrix
            not positive definite).
    """
    n = cov_matrix.shape[0]
    if cov_matrix.shape != (n, n):
        raise ValidationError(
            f"cov_matrix must be square, got shape {cov_matrix.shape}"
        )
    budget = (
        np.full(n, 1.0 / n)
        if risk_budget is None
        else np.asarray(risk_budget, dtype=float)
    )
    if len(budget) != n:
        raise ValidationError(
            f"risk_budget length ({len(budget)}) must match cov_matrix size ({n})"
        )
    if np.any(budget <= 0):
        raise ValidationError("risk_budget entries must all be > 0")
    if not np.isclose(budget.sum(), 1.0, atol=1e-6):
        raise ValidationError(f"risk_budget must sum to 1.0, got {budget.sum():.6f}")

    w = np.full(n, 1.0 / n)
    converged = False
    iterations_used = 0
    for i in range(max_iterations):
        iterations_used = i + 1
        port_var = float(w @ cov_matrix @ w)
        if port_var <= 0:
            raise ValidationError(
                "portfolio variance is non-positive — cov_matrix is not positive definite"
            )
        marginal = cov_matrix @ w
        risk_contrib = w * marginal
        target = budget * port_var
        ratio = np.sqrt(np.clip(target / np.clip(risk_contrib, 1e-16, None), 1e-8, 1e8))
        w_new = np.clip(w * ratio, 0.0, None)
        total = w_new.sum()
        if total <= 0:
            raise ValidationError("risk parity iteration collapsed to all-zero weights")
        w_new = w_new / total
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            converged = True
            break
        w = w_new

    port_var = float(w @ cov_matrix @ w)
    marginal = cov_matrix @ w
    risk_contrib_pct = (w * marginal) / port_var if port_var > 0 else np.zeros(n)

    logger.debug(
        "[risk_parity] assets=%d  converged=%s  iterations=%d",
        n,
        converged,
        iterations_used,
    )

    return {
        "weights": w,
        "risk_contributions": risk_contrib_pct,
        "converged": converged,
        "iterations_used": iterations_used,
    }


def black_litterman(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    omega: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Black-Litterman posterior expected returns/covariance (He & Litterman
    1999): blends the market-implied equilibrium prior
    (pi = risk_aversion * cov_matrix @ market_weights) with k explicit
    investor views (P: (k,n) pick matrix, Q: (k,) view returns).

    Args:
        cov_matrix: (n, n) covariance matrix (annualized).
        market_weights: (n,) prior portfolio weights (market-cap or equal
            weight), need not sum to exactly 1 but should be a sensible prior.
        P: (k, n) pick matrix — row i encodes view i as a linear combination
            of assets (e.g. [1,0,0] for an absolute view on asset 0, or
            [1,-1,0] for a relative view "asset 0 will outperform asset 1").
        Q: (k,) annualized expected return implied by each view.
        risk_aversion: Market risk-aversion coefficient (delta); higher
            means the equilibrium prior is more risk-averse. 2.5 is a
            commonly used default.
        tau: Scalar controlling confidence in the prior itself (smaller tau
            = more confident in equilibrium, less swayed by views). 0.05 is
            a commonly used default.
        omega: (k, k) view-uncertainty covariance. None (default) uses the
            standard He-Litterman diagonal default,
            omega_kk = tau * (P @ cov_matrix @ P.T)_kk — see build_bl_views
            for a confidence-scaled variant.

    Returns:
        Dict with implied_equilibrium_returns (n,), posterior_returns (n,),
        posterior_cov (n,n), implied_weights (n, sums to 1 — the
        unconstrained optimal holding (risk_aversion*posterior_cov)^-1 @
        posterior_returns, renormalized to sum 1 since the raw optimal
        holding need not).

    Raises:
        ValidationError: shape mismatches, non-positive tau/risk_aversion,
            or a singular cov_matrix/omega.
    """
    n = cov_matrix.shape[0]
    if cov_matrix.shape != (n, n):
        raise ValidationError(
            f"cov_matrix must be square, got shape {cov_matrix.shape}"
        )
    market_weights = np.asarray(market_weights, dtype=float)
    if len(market_weights) != n:
        raise ValidationError(
            f"market_weights length ({len(market_weights)}) must match cov_matrix size ({n})"
        )
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.asarray(Q, dtype=float).reshape(-1)
    k = P.shape[0]
    if P.shape[1] != n:
        raise ValidationError(
            f"P must have {n} columns (one per asset), got {P.shape[1]}"
        )
    if len(Q) != k:
        raise ValidationError(f"Q length ({len(Q)}) must match P's row count ({k})")
    if tau <= 0:
        raise ValidationError(f"tau must be > 0, got {tau}")
    if risk_aversion <= 0:
        raise ValidationError(f"risk_aversion must be > 0, got {risk_aversion}")

    try:
        tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
    except np.linalg.LinAlgError as e:
        raise ValidationError(f"cov_matrix is singular: {e}") from e

    pi = risk_aversion * cov_matrix @ market_weights

    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov_matrix @ P.T))
    else:
        omega = np.asarray(omega, dtype=float)
        if omega.shape != (k, k):
            raise ValidationError(f"omega must be shape ({k}, {k}), got {omega.shape}")

    try:
        omega_inv = np.linalg.inv(omega)
    except np.linalg.LinAlgError as e:
        raise ValidationError(f"omega (view uncertainty) is singular: {e}") from e

    middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    posterior_returns = middle @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    posterior_cov = cov_matrix + middle

    try:
        posterior_cov_inv = np.linalg.inv(posterior_cov)
    except np.linalg.LinAlgError as e:
        raise ValidationError(f"posterior_cov is singular: {e}") from e
    implied_weights_raw = (posterior_cov_inv @ posterior_returns) / risk_aversion
    total = implied_weights_raw.sum()
    if abs(total) < 1e-14:
        raise ValidationError(
            "implied Black-Litterman weights sum to ~0 — cannot renormalize to 1.0"
        )
    implied_weights = implied_weights_raw / total

    logger.debug("[black_litterman] assets=%d  views=%d  tau=%.4f", n, k, tau)

    return {
        "implied_equilibrium_returns": pi,
        "posterior_returns": posterior_returns,
        "posterior_cov": posterior_cov,
        "implied_weights": implied_weights,
    }


def build_bl_views(
    tickers: List[str],
    views: List[Dict[str, Any]],
    cov_matrix: np.ndarray,
    tau: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a plain-dict view list into the (P, Q, omega) matrices
    black_litterman() expects.

    Args:
        tickers: Asset order matching cov_matrix's rows/columns.
        views: Each dict: {"assets": {ticker: pick_coefficient, ...},
            "view_return": float, "confidence": float in (0, 1], default 1.0}.
            E.g. {"assets": {"AAPL": 1.0}, "view_return": 0.15} is an
            absolute view; {"assets": {"AAPL": 1.0, "MSFT": -1.0},
            "view_return": 0.05} is a relative view ("AAPL will outperform
            MSFT by 5%/yr").
        cov_matrix: (n, n) covariance matrix, same order as tickers.
        tau: Same tau that will be passed to black_litterman — needed here
            because the default per-view uncertainty is
            tau * (P @ cov_matrix @ P.T)_kk.

    Returns:
        (P, Q, omega) — pass directly as black_litterman(..., P=P, Q=Q, omega=omega).

    Confidence handling is a documented simplification of Idzorek's (2005)
    full confidence-scaling method, not a reimplementation of it:
    confidence=1.0 (default) uses the standard He-Litterman diagonal
    omega_kk; a lower confidence widens that view's omega_kk proportionally
    (omega_kk = default_kk / confidence), so confidence -> 0 makes the view
    contribute almost nothing to the posterior.

    Raises:
        ValidationError: views is empty, a view is missing a required key,
            a view references an unknown ticker, or a confidence is outside
            (0, 1].
    """
    if not views:
        raise ValidationError("views must be non-empty")
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    k = len(views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    confidences = np.ones(k)
    for i, view in enumerate(views):
        # Raw KeyError here named the missing key and nothing else -- not
        # which view, not what the key is for, and not as the ValidationError
        # every other boundary in this package raises. These view dicts are
        # agent-reachable through run_portfolio_optimization, so the error is
        # what an LLM has to self-correct from.
        for required in ("assets", "view_return"):
            if required not in view:
                raise ValidationError(
                    f"view {i} is missing required key {required!r}. Each view "
                    'must be {"assets": {ticker: pick_coefficient, ...}, '
                    '"view_return": float, "confidence": optional float in '
                    f"(0, 1]}}. Got keys: {sorted(view)}"
                )
        assets = view["assets"]
        if not isinstance(assets, dict) or not assets:
            raise ValidationError(
                f"view {i}'s 'assets' must be a non-empty dict mapping ticker "
                f"to pick coefficient, got {assets!r}"
            )
        unknown = [t for t in assets if t not in ticker_idx]
        if unknown:
            raise ValidationError(f"view references unknown tickers: {unknown}")
        for t, coef in assets.items():
            P[i, ticker_idx[t]] = coef
        Q[i] = view["view_return"]
        confidence = view.get("confidence", 1.0)
        if confidence <= 0 or confidence > 1:
            raise ValidationError(f"confidence must be in (0, 1], got {confidence}")
        confidences[i] = confidence

    default_omega_diag = np.diag(tau * P @ cov_matrix @ P.T).copy()
    default_omega_diag = np.where(default_omega_diag <= 0, 1e-8, default_omega_diag)
    omega = np.diag(default_omega_diag / confidences)
    return P, Q, omega
