"""
The smallest liquid basket that tracks a benchmark.

A DIFFERENT PROBLEM FROM PORTFOLIO OPTIMIZATION, and the difference is the
objective rather than the constraints. `portfolio/optimize.py` minimizes a
portfolio's OWN variance, `w'Sw`. Replication minimizes the variance of the
DIFFERENCE from a benchmark:

    min  Var(Rw - Rb)  =  w'Sw - 2 w'c + var_b

where `c` is the vector of covariances between each candidate and the
benchmark. Nothing in this library computed that, and the two answers are
not close: the minimum-variance portfolio of a universe is typically a
defensive corner of it, while the minimum-tracking-error portfolio is
whatever most resembles the index, defensive or not.

THE COVARIANCE IS BORROWED, NOT REBUILT. `portfolio/covariance.py` already
estimates one with Ledoit-Wolf shrinkage, exponential weighting and
conditioning diagnostics, and it returns an ANNUALIZED matrix so this does
not annualize anything itself. That matters more here than in most places:
a replication basket is fitted on more candidates than a sample covariance
can support, which is precisely the regime shrinkage exists for. The
benchmark is appended as one more column before estimating, so the shrinkage
applies to `S` and `c` together -- shrinking the candidates' covariance
while leaving their covariance with the benchmark raw would tilt the
objective toward whichever names had the noisiest benchmark relationship.

EACH SOLVE IS EXACT AND CERTIFIED. The fit is `optimize_portfolio`'s
min_tracking_error: the benchmark rides along as one more asset with weight
held at zero and benchmark weight one, so (w - b)'S(w - b) over the joined
covariance is exactly w'Sw - 2 w'c + var_b. The active-set method ends with
one linear solve on the final active set, and the answer's KKT conditions
are checked from the weights. SLSQP did this before and stopped about 1e-6
short of the optimum in the weights on this module's test inputs;
`converged` now means the answer certified.

CARDINALITY IS THE HARD PART AND THE ANSWER IS A HEURISTIC. "At most forty
names" is an integer constraint. The quadratic programme is continuous and
cannot express it, and this library has no mixed-integer solver -- so
`max_names` is enforced by solving, keeping the largest positions, and
re-solving on that support. That is standard practice and it is NOT a
global optimum: a different subset of the same size may track better, and
finding the best one is combinatorial. The result says so rather than
presenting a heuristic as a solution, because a tracking error 4 bps worse
than optimal matters far less than a caller believing it is optimal.

IN-SAMPLE TRACKING ERROR IS NOT A FORECAST. The optimizer fits the same
window it is scored on, so the realized figure is the best this basket will
ever look. It is reported next to the covariance-predicted one because the
gap between them is informative: predicted well below realized means the
fit is riding sample noise it will not see again.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import general_optimizer
from standard_quant_tools.portfolio.covariance import estimate_covariance

from .hedging import TRADING_DAYS, tracking_error

__all__ = ["HAS_SCIPY", "optimize_replication_basket"]

HAS_SCIPY = False
try:
    import scipy.optimize  # noqa: F401 -- the feasibility programme's HiGHS

    HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without scipy
    pass

#: The column the benchmark is carried under while it rides along with the
#: candidates through the covariance estimator. Chosen to be something no
#: real ticker can collide with.
_BENCH = "__benchmark__"


def optimize_replication_basket(
    *,
    returns: Any,
    benchmark_returns: Any,
    max_names: Optional[int] = None,
    long_only: bool = True,
    max_weight: Optional[float] = None,
    weight_caps: Optional[Dict[str, float]] = None,
    covariance_method: str = "ledoit_wolf",
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    Weights that minimize tracking error against a benchmark.

    `returns` is a DataFrame of candidate return series, `benchmark_returns`
    the series to track; both are periodic returns as decimals, aligned on
    their index. `weight_caps` is a per-name ceiling -- the natural place to
    put an ADV-derived limit, so a name that cannot be traded in size cannot
    be selected in size.

    Weights sum to one. Shorting is off by default because a replication
    basket that shorts is not a tracking basket, it is a long-short position
    with an index benchmark, and the two are managed differently.
    """
    frame, bench = _aligned(returns, benchmark_returns)
    names = list(frame.columns)
    n_assets = len(names)
    n_obs = len(frame)

    if n_assets < 2:
        raise ValidationError(
            f"replication needs at least two candidates, got {n_assets}."
        )
    if max_names is not None:
        max_names = int(max_names)
        if max_names < 1:
            raise ValidationError(f"max_names={max_names} must be at least 1.")
        if max_names >= n_assets:
            max_names = None  # not a constraint; do not report it as one

    # One estimate over candidates AND benchmark together, then partitioned.
    joined = frame.copy()
    joined[_BENCH] = bench
    estimate = estimate_covariance(
        joined, method=covariance_method, periods_per_year=periods_per_year
    )
    # Indexed BY NAME, not by position: `estimate_covariance` drops any
    # column that is entirely missing, so its `assets` order is not
    # guaranteed to be the order the candidates went in.
    matrix = estimate["matrix"]
    surviving = set(estimate["assets"])
    dropped = [name for name in names if name not in surviving]
    if dropped:
        raise ValidationError(
            f"the covariance estimator dropped {dropped}, so those "
            "candidates cannot be sized. They carry no usable observations "
            "over this window."
        )
    cov = np.array([[float(matrix[a][b]) for b in names] for a in names], dtype=float)
    cross = np.array([float(matrix[a][_BENCH]) for a in names], dtype=float)
    bench_var = float(matrix[_BENCH][_BENCH])

    warnings: List[str] = list(estimate.get("warnings", []))
    if n_obs <= n_assets:
        warnings.append(
            f"{n_obs} observations for {n_assets} candidates. Even shrunk, "
            "there are directions in which the fitted basket has zero "
            "apparent tracking error and real tracking error. Treat these "
            "weights as unusable rather than merely uncertain."
        )

    lower = 0.0 if long_only else -abs(max_weight or 1.0)
    caps = np.full(n_assets, max_weight if max_weight is not None else 1.0, dtype=float)
    if weight_caps:
        unknown = set(weight_caps) - set(names)
        if unknown:
            raise ValidationError(
                f"weight_caps names {sorted(unknown)}, which are not "
                "candidates. A cap on a name that cannot be held is a typo."
            )
        for symbol, cap in weight_caps.items():
            position = names.index(symbol)
            caps[position] = min(caps[position], float(cap))
    if caps.sum() < 1.0:
        raise ValidationError(
            f"the weight caps sum to {caps.sum():.4f}, so no fully-invested "
            "basket satisfies them. Either raise the caps or accept less "
            "than full investment, which this does not model."
        )

    support = np.arange(n_assets)
    weights, stop = _solve(cov, cross, bench_var, support, lower, caps, names)

    thresholded = False
    if max_names is not None:
        # Iterative hard thresholding. Solve, keep the largest positions,
        # re-solve on that support. Two passes rather than one: the re-solve
        # redistributes weight, and a name that was marginal in the full
        # problem is sometimes clearly worth keeping once the rest are gone.
        for _ in range(2):
            active = np.flatnonzero(np.abs(weights) > 1e-8)
            if len(active) <= max_names:
                break
            keep = active[np.argsort(-np.abs(weights[active]))[:max_names]]
            support = np.sort(keep)
            weights, stop = _solve(cov, cross, bench_var, support, lower, caps, names)
            thresholded = True

    held = np.flatnonzero(np.abs(weights) > 1e-8)
    selected = [names[i] for i in held]

    # Predicted from the covariance, realized from the actual series. The
    # covariance is already annualized, so this takes a square root and
    # stops; `tracking_error` annualizes its own input, so it does not.
    # On one BLAS thread, as `hedge_effectiveness` reads its correlation:
    # the replicated series is a product with a term per row, and from
    # pandas 3.1 the correlation is np.dot products too, both of which
    # OpenBLAS splits across threads above 10,000 terms.
    with single_threaded_blas():
        predicted_var = float(
            weights @ cov @ weights - 2.0 * weights @ cross + bench_var
        )
        replicated = pd.Series(frame.to_numpy(dtype=float) @ weights, index=frame.index)
        realized_te = tracking_error(
            replicated, bench, periods_per_year=periods_per_year
        )
        correlation = float(replicated.corr(bench))
        beta = float(cross @ weights / bench_var) if bench_var > 0 else float("nan")
    predicted_te = math.sqrt(max(predicted_var, 0.0))

    # Verified from the returned vector rather than trusted from the solver:
    # the constraints it was given, recomputed, beside its certificate.
    converged = stop is None
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        converged = False
        warnings.append(
            f"The weights sum to {weights.sum():.6f} rather than 1, so the "
            "fully-invested constraint was not met. The basket below is not "
            "the portfolio that was asked for."
        )
    if long_only and float(weights.min()) < -1e-9:
        converged = False
        warnings.append(
            f"A weight came back at {weights.min():.6g} despite long_only, "
            "so the bound was not respected."
        )
    # And the certificate: an answer whose KKT conditions do not hold, or a
    # solve that stopped, is reported, never returned as the minimum.
    if stop is not None:
        warnings.append(
            f"The exact solve did not certify a minimum of tracking error: "
            f"{stop}. The weights are not a verified optimum; converged is "
            "False. The constraint checks above say whether they are at "
            "least feasible."
        )

    if thresholded:
        warnings.append(
            f"max_names={max_names} was enforced by thresholding, not by "
            "solving the integer problem -- this library has no "
            "mixed-integer solver. A different subset of the same size may "
            "track better; finding the best one is combinatorial. Treat "
            "these as a good basket of that size, not the best one."
        )
    warnings.append(
        f"Tracking error is IN SAMPLE: the weights were fitted on the same "
        f"{n_obs} observations they are scored on, so {realized_te:.4%} is "
        f"the best this basket will ever look. Predicted {predicted_te:.4%} "
        "from the covariance -- when the predicted figure is much the "
        "smaller, the fit is riding sample noise."
    )
    if long_only:
        warnings.append(
            "Long-only. A basket allowed to short usually shows lower "
            "tracking error and is not a replication basket -- it is a "
            "long-short position benchmarked to an index."
        )
    if not selected:
        warnings.append(
            "Every weight came back at zero: no basket was solved (the "
            "warning above says why), so there is nothing to hold."
        )

    return {
        "n_candidates": n_assets,
        "n_observations": n_obs,
        "n_selected": len(selected),
        "max_names": max_names,
        "long_only": bool(long_only),
        "covariance_method": covariance_method,
        "weights": {names[i]: float(weights[i]) for i in held},
        "selected_names": selected,
        "predicted_tracking_error": predicted_te,
        "realized_tracking_error": realized_te,
        "correlation": correlation,
        "beta": beta,
        "gross_weight": float(np.abs(weights).sum()),
        "net_weight": float(weights.sum()),
        "largest_weight": float(np.max(np.abs(weights))) if len(held) else 0.0,
        "periods_per_year": int(periods_per_year),
        "converged": converged,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _solve(
    cov: np.ndarray,
    cross: np.ndarray,
    bench_var: float,
    support: np.ndarray,
    lower: float,
    caps: np.ndarray,
    names: List[str],
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Minimize `w'Sw - 2 w'c + var_b` over `support`, fully invested and
    inside caps, exactly: `optimize_portfolio`'s min_tracking_error on the
    candidates and the benchmark joined, the benchmark held at weight zero
    and given benchmark weight one, every candidate off the support held at
    zero. Held weights are substituted out before the solve, so only the
    support's own covariance has to be positive definite.

    Returns the weights and, when the answer did not certify, why (the
    optimizer's account of each stage); None when it did. The optimizer
    runs its linear algebra on one BLAS thread, as SLSQP's solve did here.
    """
    if not HAS_SCIPY:
        raise ValidationError(
            "optimize_replication_basket requires scipy, which is not "
            "installed: its feasibility programme is scipy's HiGHS. There is "
            "no closed form here: long-only and the per-name caps are "
            "inequality constraints, and dropping them would answer a "
            "different question rather than the same one approximately."
        )
    n = len(names)
    joined = np.empty((n + 1, n + 1))
    joined[:n, :n] = cov
    joined[:n, n] = cross
    joined[n, :n] = cross
    joined[n, n] = bench_var
    lower_bounds = np.zeros(n + 1)
    upper_bounds = np.zeros(n + 1)
    lower_bounds[support] = lower
    upper_bounds[support] = caps[support]
    benchmark = np.zeros(n + 1)
    benchmark[n] = 1.0
    try:
        out = general_optimizer.optimize_portfolio(
            joined,
            assets=list(names) + [_BENCH],
            objective="min_tracking_error",
            benchmark_weights=benchmark,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            budget=1.0,
            budget_type="equal",
        )
    except ValidationError as exc:
        raise ValidationError(
            f"optimize_replication_basket: the tracking-error fit on "
            f"{len(support)} candidate(s) was refused -- {exc}"
        ) from exc
    weights = np.zeros(n, dtype=float)
    if out["weights"]:
        weights = np.array([out["weights"][name] for name in names], dtype=float)
        weights[np.setdiff1d(np.arange(n), support)] = 0.0
    if out["certified"]:
        return weights, None
    certificate = out.get("certificate") or {}
    failed = [
        f"{key} {value:.1e}"
        for key, value in certificate.items()
        if key != "tolerance" and not value <= certificate.get("tolerance", 0.0)
    ]
    reason = f"{out.get('solver')} on {len(support)} candidate(s)" + (
        f"; over tolerance: {', '.join(failed)}" if failed else ""
    )
    return weights, reason


def _aligned(returns: Any, benchmark: Any):
    """Candidates and benchmark on their shared index, non-finite dropped.

    An INNER join. A date one side observed and the other did not is a
    missing comparison, not a zero one, and filling it would fit the basket
    against a fabricated benchmark return on exactly the dates that are
    already the hardest to track.
    """
    frame = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(returns)
    series = benchmark if isinstance(benchmark, pd.Series) else pd.Series(benchmark)
    frame = frame.astype(float)
    series = series.astype(float)
    if _BENCH in frame.columns:
        raise ValidationError(
            f"a candidate is named {_BENCH!r}, which this function uses "
            "internally to carry the benchmark. Rename it."
        )
    if len(frame) != len(series) and not frame.index.equals(series.index):
        raise ValidationError(
            f"the candidates have {len(frame)} observations and the benchmark "
            f"{len(series)}, on indexes that do not align. This cannot tell "
            "which end of the shorter one is missing."
        )
    joined = frame.join(series.rename(_BENCH), how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        raise ValidationError("candidates and benchmark share no usable observations.")
    if len(joined) < 3:
        raise ValidationError(
            f"only {len(joined)} usable observations; a covariance needs far "
            "more than that to mean anything."
        )
    return joined.drop(columns=_BENCH), joined[_BENCH]
