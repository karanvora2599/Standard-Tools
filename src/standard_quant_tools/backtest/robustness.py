"""
Robustness diagnostics: is a backtest result — or the best row of a grid
search — actually trustworthy, or a fluke of one sample path / one lucky
parameter combination among many tried?

Three independent checks, each usable on its own:
- block_bootstrap_ci: resample a return series to get a confidence interval
  around a point-estimate metric (e.g. Sharpe), instead of trusting the
  single-sample value.
- parameter_sensitivity: from an existing backtest_grid/run_backtest_
  optimization result, how much better is the best row than the pack —
  a large best-vs-median gap on a small grid is a red flag for overfitting.
- deflated_sharpe_ratio: corrects the observed best Sharpe ratio for the
  fact that it was selected as the maximum of n_trials attempts (Bailey &
  Lopez de Prado, "The Deflated Sharpe Ratio", 2014) — the more parameter
  combinations searched, the higher the bar a genuine Sharpe ratio must
  clear before it's distinguishable from noise.

None of this replaces out-of-sample validation (walk_forward.py) — it
quantifies confidence in a same-sample estimate, which is a different and
complementary question ("how sure am I this number is real" vs. "would it
have held up on unseen data").
"""

import logging
import math
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from standard_quant_tools._special import (
    norm_cdf,
    norm_ppf,
)
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_EULER_MASCHERONI = 0.5772156649015329


# See `_special`: this had 7 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_cdf = norm_cdf

# See `_special`: this had 2 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_ppf = norm_ppf


def block_bootstrap_ci(
    returns: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    n_iterations: int = 1000,
    block_size: int = 20,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Block bootstrap confidence interval for a metric computed from a return
    series. Overlapping blocks of `block_size` consecutive returns are
    resampled with replacement (preserving short-range autocorrelation that
    an i.i.d. resample would destroy), concatenated to the original length,
    and `metric_fn` is recomputed on each resample.

    Args:
        returns: Daily (or per-bar) return series — e.g. a strategy's
            realized returns, not the equity curve.
        metric_fn: Callable taking a pd.Series of returns and returning a
            float (e.g. standard_quant_tools.metrics.risk_metrics.sharpe_ratio).
        n_iterations: Number of bootstrap resamples.
        block_size: Length of each resampled block, in bars.
        confidence: Two-sided confidence level for the reported interval.
        seed: RNG seed for reproducibility — recorded by the audit trail
            when called through the get_robustness_diagnostics agent tool.

    Returns:
        Dict with point_estimate (metric on the original series), ci_lower,
        ci_upper, confidence, n_iterations, block_size.

    Raises:
        ValidationError: empty returns, or block_size not in (0, len(returns)].
    """
    n = len(returns)
    if n == 0:
        raise ValidationError("returns is empty")
    if block_size <= 0 or block_size > n:
        raise ValidationError(f"block_size must be in (0, {n}], got {block_size}")
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence}")
    if n_iterations <= 0:
        # np.percentile over an empty boot_metrics array raises an opaque
        # numpy error instead of naming the actual problem.
        raise ValidationError(f"n_iterations must be > 0, got {n_iterations}")

    rng = np.random.default_rng(seed)
    values = returns.to_numpy(dtype=float)
    n_blocks = math.ceil(n / block_size)
    max_start = n - block_size

    point_estimate = float(metric_fn(returns))
    boot_metrics = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        resampled = np.concatenate([values[s : s + block_size] for s in starts])[:n]
        boot_metrics[i] = metric_fn(pd.Series(resampled))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_metrics, 100.0 * alpha / 2.0))
    upper = float(np.percentile(boot_metrics, 100.0 * (1.0 - alpha / 2.0)))

    logger.debug(
        "[robustness] block_bootstrap_ci  n=%d  block_size=%d  iterations=%d  point=%.4f  ci=[%.4f, %.4f]",
        n,
        block_size,
        n_iterations,
        point_estimate,
        lower,
        upper,
    )

    return {
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "n_iterations": n_iterations,
        "block_size": block_size,
    }


def parameter_sensitivity(
    grid_df: pd.DataFrame, metric_col: str = "sharpe_ratio"
) -> Dict[str, Any]:
    """
    Cheap overfitting proxy from an existing backtest_grid /
    run_backtest_optimization result: how much better is the best trial
    than the pack? A large best-vs-median gap on a grid with few trials is
    a red flag that the top row is a fluke rather than a genuine edge.

    Args:
        grid_df: A backtest_grid result (or any DataFrame with one row per
            parameter combination and a numeric metric_col).
        metric_col: Column to rank by (higher is better).

    Returns:
        Dict with n_trials, best, median, best_minus_median,
        best_minus_rank2 (gap to the second-best trial), and
        best_minus_top5_mean (gap to the mean of ranks 2-5, or 0.0 when
        fewer than 2 trials exist). Trials whose metric_col is NaN/Inf are
        excluded from the ranking (and from n_trials) with a warning — they
        have no comparable value to rank.

    Raises:
        ValidationError: empty grid_df, metric_col not present, or no finite
            values in metric_col.
    """
    if grid_df.empty:
        raise ValidationError("grid_df is empty")
    if metric_col not in grid_df.columns:
        raise ValidationError(
            f"metric_col {metric_col!r} not found in grid_df columns: {list(grid_df.columns)}"
        )

    raw_values = grid_df[metric_col].to_numpy(dtype=float)
    # np.sort places NaN LAST, so [::-1] puts it FIRST -- a single NaN metric
    # (a grid row whose returns had zero variance is the common source) would
    # otherwise become `best`, making every reported gap NaN. A trial that
    # produced no comparable metric is excluded from the ranking rather than
    # allowed to win it.
    values = raw_values[np.isfinite(raw_values)]
    n_dropped = len(raw_values) - len(values)
    if len(values) == 0:
        raise ValidationError(
            f"grid_df[{metric_col!r}] has no finite values "
            f"({len(raw_values)} row(s), all NaN/Inf) — nothing to rank."
        )
    if n_dropped:
        logger.warning(
            "[robustness] parameter_sensitivity: excluded %d trial(s) with a "
            "non-finite %s from the ranking",
            n_dropped,
            metric_col,
        )
    sorted_desc = np.sort(values)[::-1]
    n = len(sorted_desc)
    best = float(sorted_desc[0])
    median = float(np.median(values))

    best_minus_rank2 = float(best - sorted_desc[1]) if n > 1 else 0.0
    if n > 1:
        top_pack = sorted_desc[1 : min(5, n)]
        best_minus_top5_mean = float(best - top_pack.mean())
    else:
        best_minus_top5_mean = 0.0

    result = {
        "n_trials": n,
        "best": best,
        "median": median,
        "best_minus_median": round(best - median, 6),
        "best_minus_rank2": round(best_minus_rank2, 6),
        "best_minus_top5_mean": round(best_minus_top5_mean, 6),
    }
    logger.debug("[robustness] parameter_sensitivity  %s", result)
    return result


def deflated_sharpe_ratio(
    observed_sharpe: float,
    sharpe_trials_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): the probability
    that the true Sharpe ratio exceeds zero, after correcting for
    `observed_sharpe` having been selected as the best of `n_trials`
    independent attempts (e.g. a parameter grid search) rather than a
    single pre-registered test.

    Args:
        observed_sharpe: The best trial's (non-annualized, per-period)
            Sharpe ratio.
        sharpe_trials_std: Standard deviation of the Sharpe ratios actually
            observed across all n_trials (e.g. grid_df["sharpe_ratio"].std()
            from the same grid search) — used as the estimate of the
            cross-trial Sharpe-ratio variance the expected-maximum formula
            needs. This is a measured quantity from the actual search, not
            an assumed theoretical one.
        n_trials: Number of independent trials searched (e.g. grid
            combinations). n_trials <= 1 skips the multiple-testing
            correction entirely (SR0 = 0) — no selection bias to correct
            for with only one trial.
        n_obs: Number of return observations the Sharpe ratio was estimated
            from (bars in the backtest).
        skew, kurtosis: Return-distribution moments used in the Sharpe
            ratio's own standard-error formula (defaults = normal
            distribution: skew=0, kurtosis=3, i.e. no correction beyond the
            basic Sharpe standard error).

    Returns:
        Dict with expected_max_sharpe (SR0 — the bar observed_sharpe must
        clear), deflated_sharpe_ratio (the DSR / probabilistic value in
        [0, 1]), and the echoed inputs n_trials, n_obs.

    Raises:
        ValidationError: n_obs < 2, or the Sharpe standard-error
        denominator is non-positive (degenerate skew/kurtosis input).
    """
    if n_obs < 2:
        raise ValidationError(f"n_obs must be >= 2, got {n_obs}")

    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = sharpe_trials_std * (
            (1.0 - _EULER_MASCHERONI) * _norm_ppf(1.0 - 1.0 / n_trials)
            + _EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        )

    denom_sq = (
        1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2
    )
    if denom_sq <= 0:
        raise ValidationError(
            "degenerate skew/kurtosis input: Sharpe standard-error denominator "
            f"is non-positive ({denom_sq:.6f}) for observed_sharpe={observed_sharpe}, "
            f"skew={skew}, kurtosis={kurtosis}"
        )
    denom = math.sqrt(denom_sq)

    z = (observed_sharpe - sr0) * math.sqrt(n_obs - 1) / denom
    dsr = _norm_cdf(z)

    logger.debug(
        "[robustness] deflated_sharpe_ratio  observed=%.4f  sr0=%.4f  n_trials=%d  n_obs=%d  dsr=%.4f",
        observed_sharpe,
        sr0,
        n_trials,
        n_obs,
        dsr,
    )

    return {
        "expected_max_sharpe": round(sr0, 6),
        "deflated_sharpe_ratio": round(dsr, 6),
        "n_trials": n_trials,
        "n_obs": n_obs,
    }
