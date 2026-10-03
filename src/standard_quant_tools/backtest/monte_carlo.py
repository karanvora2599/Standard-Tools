import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from standard_quant_tools._resampling import block_indices
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

logger = logging.getLogger(__name__)

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


# The per-day equity bands, and the terminal percentiles beside the median.
_BAND_PERCENTILES = (5.0, 50.0, 95.0)
_TERMINAL_PERCENTILES = (5.0, 95.0)


def _may_differ_by_tie_order(values: np.ndarray) -> np.ndarray:
    """Which percentiles a consolidated call may have returned in different
    bits from one call per percentile: those that came out zero or NaN.

    `np.percentile` with several percentiles partitions once at every
    position any of them needs, and one call per percentile partitions at
    only its own. Both put the same VALUE at each position -- an order
    statistic does not depend on how the partition got there -- but values
    that compare equal can land in a different order, and two such values
    can differ in their bits: -0.0 and +0.0, or two NaNs. The linear
    interpolation between two order statistics is then the same double
    unless the result is zero (a signed zero) or NaN, so those are the only
    results the consolidated call is not trusted for. The kernel produces
    -0.0 honestly: a -1 return takes an equity path to zero and a later
    return below -1 flips its sign."""
    with np.errstate(invalid="ignore"):
        return (values == 0.0) | np.isnan(values)


def _band_percentiles(paths: np.ndarray) -> np.ndarray:
    """`np.percentile(paths, q, axis=0)` for q in 5, 50 and 95, row for row
    the doubles the three separate calls return.

    One call partitions each day once for all three percentiles, and it
    partitions a transposed copy, so each day's values are contiguous --
    partitioning along the strided axis of the (paths x days) matrix is what
    made the three calls about 95% of this function. The copy is the one
    the separate calls each made for themselves, made once and partitioned
    in place. A band holding a zero or a NaN is recomputed as its own call
    (see `_may_differ_by_tie_order`), so the result is the separate calls'
    to the bit, sign of zero included.

    The copy is `np.array(..., copy=True)`, not `np.ascontiguousarray`: at a
    one-day horizon the transpose of a (paths x 1) matrix is already
    contiguous, `ascontiguousarray` returns it uncopied, and the in-place
    partition would then reorder the caller's matrix -- which the fallback
    reads."""
    days_by_path = np.array(paths.T, dtype=np.float64, order="C", copy=True)
    bands = np.percentile(days_by_path, _BAND_PERCENTILES, axis=1, overwrite_input=True)
    for row, q in enumerate(_BAND_PERCENTILES):
        if _may_differ_by_tie_order(bands[row]).any():
            bands[row] = np.percentile(paths, q, axis=0)
    return bands


def _terminal_statistics(
    terminal: np.ndarray, initial_capital: float
) -> Dict[str, float]:
    """The terminal-distribution statistics both simulators report.

    The 5th and 95th percentiles of the terminal equity are one
    `np.percentile` call, with the same zero-or-NaN fallback as the bands,
    and the 5th percentile of the terminal return is computed once: it was
    computed twice, for the VaR and again for the CVaR threshold, and both
    uses read the same double."""
    terminal_returns = terminal / initial_capital - 1.0

    terminal_median = float(np.median(terminal))
    tails = np.percentile(terminal, _TERMINAL_PERCENTILES)
    for slot, q in enumerate(_TERMINAL_PERCENTILES):
        if _may_differ_by_tie_order(tails[slot]):
            tails[slot] = np.percentile(terminal, q)
    terminal_p5 = float(tails[0])
    terminal_p95 = float(tails[1])
    prob_loss = float(np.mean(terminal < initial_capital))

    # VaR/CVaR of the simulated terminal-return distribution (positive
    # loss-magnitude convention, matching metrics.risk_metrics.var_historical).
    return_p5 = np.percentile(terminal_returns, 5.0)
    var_95 = float(-return_p5)
    tail = terminal_returns[terminal_returns <= return_p5]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    return {
        "terminal_median": terminal_median,
        "terminal_p5": terminal_p5,
        "terminal_p95": terminal_p95,
        "prob_loss": prob_loss,
        "terminal_var_95": var_95,
        "terminal_cvar_95": cvar_95,
    }


def simulate_forward_paths(
    returns: pd.Series,
    horizon_days: int,
    n_simulations: int = 1000,
    block_size: int = 20,
    initial_capital: float = 10_000.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Monte Carlo forward simulation via moving-block bootstrap of a
    historical return series — projects `n_simulations` possible future
    equity paths over `horizon_days` bars, each built from resampled
    blocks of the ACTUAL historical returns (preserving their real
    distribution shape, fat tails, and short-range autocorrelation, unlike
    a parametric normal-distribution assumption).

    This mirrors the block-resampling approach used by
    backtest.robustness.block_bootstrap_ci (overlapping blocks drawn with
    replacement, concatenated to the target length) — deliberately
    reimplemented here rather than imported, to avoid coupling this new,
    unrelated feature to that function's own evolution and to keep each
    module independently testable.

    Args:
        returns: Historical daily (or per-bar) return series to resample
            from — e.g. a portfolio's realized daily returns.
        horizon_days: Number of forward bars to simulate per path.
        n_simulations: Number of independent simulated paths.
        block_size: Length of each resampled block, in bars.
        initial_capital: Starting capital for every simulated path.
        seed: RNG seed for reproducibility. Reproducibility is only
            guaranteed WITHIN one backend: if the compiled `_sqt_core`
            extension is present, the same seed produces different
            concrete numbers than the pure-Python fallback would (the C++
            path uses its own RNG, not a reimplementation of numpy's
            PCG64 bit stream) — repeat calls on the same machine/build are
            still bit-identical for a given seed.

    Returns:
        Dict with terminal-distribution stats (terminal_median, terminal_p5,
        terminal_p95, prob_loss, terminal_var_95, terminal_cvar_95) and
        per-day percentile equity-curve bands (equity_band_p5,
        equity_band_p50, equity_band_p95 — each a length-horizon_days list).

    Raises:
        ValidationError: empty returns, non-positive horizon_days/
            n_simulations/initial_capital, or block_size not in
            (0, len(returns)].
    """
    n = len(returns)
    if n == 0:
        raise ValidationError("returns is empty")
    if horizon_days <= 0:
        raise ValidationError(f"horizon_days must be > 0, got {horizon_days}")
    if n_simulations <= 0:
        raise ValidationError(f"n_simulations must be > 0, got {n_simulations}")
    if initial_capital <= 0:
        raise ValidationError(f"initial_capital must be > 0, got {initial_capital}")
    if block_size <= 0 or block_size > n:
        raise ValidationError(f"block_size must be in (0, {n}], got {block_size}")

    values = returns.to_numpy(dtype=float)
    # simulate_forward_paths_into validates initial_capital's finiteness
    # but never checked `values` itself -- a single NaN/Inf in the
    # historical returns being resampled from poisons `equity` permanently
    # for every path/bar downstream of when it's sampled
    # (equity *= (1.0 + values[start+k])), with no explicit check anywhere
    # in the native kernel, header, or binding.
    require_finite_array(values, "returns", "simulate_forward_paths")

    if HAS_CPP and _cpp_core is not None:
        paths = _cpp_core.simulate_forward_paths(
            values, horizon_days, n_simulations, block_size, initial_capital, seed
        )
    else:
        rng = np.random.default_rng(seed)

        # (n_simulations, horizon_days) matrix of simulated equity paths
        paths = np.empty((n_simulations, horizon_days), dtype=float)
        for i in range(n_simulations):
            # Resamples n historical observations into a horizon_days path,
            # so the target length is passed explicitly -- see `_resampling`.
            resampled = values[block_indices(n, block_size, rng, horizon_days)]
            paths[i, :] = initial_capital * np.cumprod(1.0 + resampled)

    stats = _terminal_statistics(paths[:, -1], initial_capital)
    bands = _band_percentiles(paths)

    logger.debug(
        "[monte_carlo] horizon=%d  n_sim=%d  block_size=%d  terminal_median=%.2f  prob_loss=%.4f",
        horizon_days,
        n_simulations,
        block_size,
        stats["terminal_median"],
        stats["prob_loss"],
    )

    return {
        **stats,
        "equity_band_p5": bands[0].tolist(),
        "equity_band_p50": bands[1].tolist(),
        "equity_band_p95": bands[2].tolist(),
    }


def simulate_forward_paths_terminal(
    returns: pd.Series,
    horizon_days: int,
    n_simulations: int = 1000,
    block_size: int = 20,
    initial_capital: float = 10_000.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Memory-bounded variant of simulate_forward_paths(): identical moving-
    block bootstrap, but never materializes the full (n_simulations x
    horizon_days) path matrix -- only each path's terminal equity. For a
    large n_simulations x horizon_days (e.g. 1,000,000 x 252 would be a
    ~2GB full path matrix), this avoids that allocation entirely.

    Trade-off: no per-day equity_band_p5/p50/p95 in the result (those
    require the full per-day matrix, which this variant never builds) --
    only the terminal-distribution stats. Use simulate_forward_paths()
    instead if the per-day bands are needed.

    Args, Raises: same as simulate_forward_paths().

    Returns:
        Dict with terminal-distribution stats only (terminal_median,
        terminal_p5, terminal_p95, prob_loss, terminal_var_95,
        terminal_cvar_95) -- no equity_band_* keys.
    """
    n = len(returns)
    if n == 0:
        raise ValidationError("returns is empty")
    if horizon_days <= 0:
        raise ValidationError(f"horizon_days must be > 0, got {horizon_days}")
    if n_simulations <= 0:
        raise ValidationError(f"n_simulations must be > 0, got {n_simulations}")
    if initial_capital <= 0:
        raise ValidationError(f"initial_capital must be > 0, got {initial_capital}")
    if block_size <= 0 or block_size > n:
        raise ValidationError(f"block_size must be in (0, {n}], got {block_size}")

    values = returns.to_numpy(dtype=float)
    require_finite_array(values, "returns", "simulate_forward_paths_terminal")

    if HAS_CPP and _cpp_core is not None:
        terminal = _cpp_core.simulate_forward_paths_terminal(
            values, horizon_days, n_simulations, block_size, initial_capital, seed
        )
    else:
        rng = np.random.default_rng(seed)

        terminal = np.empty(n_simulations, dtype=float)
        for i in range(n_simulations):
            # Resamples n historical observations into a horizon_days path,
            # so the target length is passed explicitly -- see `_resampling`.
            resampled = values[block_indices(n, block_size, rng, horizon_days)]
            terminal[i] = initial_capital * np.prod(1.0 + resampled)

    stats = _terminal_statistics(terminal, initial_capital)

    logger.debug(
        "[monte_carlo_terminal] horizon=%d  n_sim=%d  block_size=%d  "
        "terminal_median=%.2f  prob_loss=%.4f",
        horizon_days,
        n_simulations,
        block_size,
        stats["terminal_median"],
        stats["prob_loss"],
    )

    return stats
