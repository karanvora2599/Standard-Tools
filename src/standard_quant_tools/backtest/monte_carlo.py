import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

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

    if HAS_CPP and _cpp_core is not None:
        paths = _cpp_core.simulate_forward_paths(
            values, horizon_days, n_simulations, block_size, initial_capital, seed
        )
    else:
        rng = np.random.default_rng(seed)
        n_blocks = math.ceil(horizon_days / block_size)
        max_start = n - block_size

        # (n_simulations, horizon_days) matrix of simulated equity paths
        paths = np.empty((n_simulations, horizon_days), dtype=float)
        for i in range(n_simulations):
            starts = rng.integers(0, max_start + 1, size=n_blocks)
            resampled = np.concatenate([values[s : s + block_size] for s in starts])[
                :horizon_days
            ]
            paths[i, :] = initial_capital * np.cumprod(1.0 + resampled)

    terminal = paths[:, -1]
    terminal_returns = terminal / initial_capital - 1.0

    terminal_median = float(np.median(terminal))
    terminal_p5 = float(np.percentile(terminal, 5.0))
    terminal_p95 = float(np.percentile(terminal, 95.0))
    prob_loss = float(np.mean(terminal < initial_capital))

    # VaR/CVaR of the simulated terminal-return distribution (positive
    # loss-magnitude convention, matching metrics.risk_metrics.var_historical).
    var_95 = float(-np.percentile(terminal_returns, 5.0))
    tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5.0)]
    cvar_95 = float(-tail.mean()) if len(tail) > 0 else var_95

    equity_band_p5 = np.percentile(paths, 5.0, axis=0).tolist()
    equity_band_p50 = np.percentile(paths, 50.0, axis=0).tolist()
    equity_band_p95 = np.percentile(paths, 95.0, axis=0).tolist()

    logger.debug(
        "[monte_carlo] horizon=%d  n_sim=%d  block_size=%d  terminal_median=%.2f  prob_loss=%.4f",
        horizon_days,
        n_simulations,
        block_size,
        terminal_median,
        prob_loss,
    )

    return {
        "terminal_median": terminal_median,
        "terminal_p5": terminal_p5,
        "terminal_p95": terminal_p95,
        "prob_loss": prob_loss,
        "terminal_var_95": var_95,
        "terminal_cvar_95": cvar_95,
        "equity_band_p5": equity_band_p5,
        "equity_band_p50": equity_band_p50,
        "equity_band_p95": equity_band_p95,
    }
