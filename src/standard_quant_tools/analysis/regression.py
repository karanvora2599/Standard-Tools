import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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


def calculate_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate static Alpha and Beta using OLS over the two series' shared
    dates.

    Returns NaN for all three statistics when fewer than two observations
    overlap — "not estimable", which is a different statement from a
    measured value.

    This used to return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}, and a
    beta of exactly 0.0 is also a perfectly legitimate measurement (a
    market-neutral asset), so nothing downstream could tell the two apart.
    The consequences were real and pointed the wrong way:

      - The screener filtered on it, so a ticker whose history did not
        overlap the benchmark PASSED beta_max=0.5 — "could not be estimated"
        read as "very low beta", exactly backwards for a defensive screen.
        (It now checks the overlap itself; this removes the trap rather than
        relying on every caller to remember it.)
      - treynor_ratio saw beta == 0 and returned 0.0, turning "no overlapping
        benchmark data" into a plausible-looking risk-adjusted return.

    NaN propagates instead of masquerading, and callers that must branch on
    it can test np.isfinite rather than comparing against a magic number.

    The same rule covers the two degenerate designs, and it is applied here,
    once, so the answer does not depend on which backend computed it:

      - A constant benchmark (or any benchmark the intercept column spans)
        leaves the slope unidentified. alpha, beta and R-squared are all
        NaN. The NumPy path used to return lstsq's minimum-norm solution
        there -- a beta of 2e-6 and an alpha of 7e-4 on a constant
        benchmark, one arbitrary member of an infinite solution set --
        while the native path returned NaN.
      - A constant asset has no variance to explain, so R-squared is 0/0
        and NaN; alpha and beta are still estimable (beta is 0).
    """
    nan = float("nan")
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index].to_numpy(dtype=np.float64)
    x = benchmark_returns.loc[common_index].to_numpy(dtype=np.float64)
    require_finite_array(y, "asset_returns", "calculate_beta")
    require_finite_array(x, "benchmark_returns", "calculate_beta")
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "numpy"
    logger.debug("[beta] n_obs=%d  path=%s", len(y), path)

    if len(y) < 2:
        logger.debug("[beta] not estimable: %d overlapping observation(s)", len(y))
        return {"alpha": nan, "beta": nan, "r_squared": nan}

    if HAS_CPP and _cpp_core is not None:
        # ols2 refuses a singular design itself (a relative test on its
        # normal-equations determinant) and returns NaN for all three.
        r = _cpp_core.ols2(y, x)
        alpha, beta = float(r["intercept"]), float(r["slope"])
        r_squared = float(r["r_squared"])
    else:
        X = np.column_stack([np.ones(len(x)), x])
        beta_hat, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        if rank < 2:
            # The same rank policy rolling_factor_loadings applies per
            # window: a design that is not full rank has no unique slope,
            # and lstsq has already computed the rank, so the test is free.
            logger.debug("[beta] not estimable: design rank %d < 2", rank)
            return {"alpha": nan, "beta": nan, "r_squared": nan}
        alpha, beta = float(beta_hat[0]), float(beta_hat[1])
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum((y - (alpha + beta * x)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan

    # A constant asset has no variance to explain: R-squared is NaN, whatever
    # the backend's own arithmetic produced. Tested on the values, not on
    # ss_tot: thirty copies of 0.01 sum to a mean that differs from 0.01 in
    # the last bit, and the ratio of two rounding residues read as an
    # R-squared of -3.2.
    if bool(np.all(y == y[0])):
        r_squared = nan
    result = {"alpha": alpha, "beta": beta, "r_squared": r_squared}

    logger.debug(
        "[beta] alpha=%.6f  beta=%.4f  R²=%.4f",
        result["alpha"],
        result["beta"],
        result["r_squared"],
    )
    return result


def rolling_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60
) -> pd.DataFrame:
    """
    Calculate rolling OLS Beta of asset vs benchmark over a sliding window.

    Uses C++ incremental O(1)-per-step sum updates when available (10-40× faster
    than two sequential pandas rolling operations). Falls back to an exact
    per-window NumPy computation otherwise.

    Both backends give every window its own beta, to rounding, including the
    windows after a large print has left: a window whose benchmark is flat
    (every value identical) has no beta and is NaN.
    """
    if window <= 1:
        raise ValidationError(
            f"window must be > 1 (a 1-bar window has no variance to regress "
            f"against), got {window}"
        )
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "numpy"
    logger.debug("[rolling_beta] window=%d  bars=%d  path=%s", window, len(y), path)

    # Checked once, unconditionally, BEFORE the C++ try/except below --
    # that except catches Exception broadly (to fall back to pandas on any
    # C++ failure), which would otherwise silently swallow a
    # ValidationError raised inside the try block and mask bad input
    # behind a confusing fallback instead of rejecting it.
    require_finite_array(y.to_numpy(dtype=np.float64), "asset_returns", "rolling_beta")
    require_finite_array(
        x.to_numpy(dtype=np.float64), "benchmark_returns", "rolling_beta"
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        try:
            y_arr = y.to_numpy(dtype=np.float64)
            x_arr = x.to_numpy(dtype=np.float64)
            betas = _cpp_core.rolling_beta(y_arr, x_arr, window)
            return pd.DataFrame({"Rolling_Beta": betas}, index=common_index)
        except Exception as exc:
            logger.warning("[rolling_beta] C++ failed (%s) — using NumPy", exc)

    # ── NumPy fallback ────────────────────────────────────────────────────────
    betas = _rolling_beta_exact(
        y.to_numpy(dtype=np.float64), x.to_numpy(dtype=np.float64), window
    )
    return pd.DataFrame({"Rolling_Beta": betas}, index=common_index)


#: Upper bound on the elements of one (windows x window) block the fallback
#: centres at a time: 8 MB per temporary, whatever the series length.
_BLOCK_ELEMENTS = 1 << 20


def _rolling_beta_exact(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Every window's OLS slope, each computed from that window alone.

    WHY NOT pandas' rolling cov / var. Those are online algorithms: a value
    is added to running sums when it enters and subtracted when it leaves.
    A large print takes the sums' low-order digits with it when it leaves,
    so every later window inherits the damage. Measured under pandas 2.x
    with one 1e8 print among 0.01-scale returns, the windows after it had
    left were wrong by a median factor of 1 and at worst 3.4e3 -- the
    native kernel had the same defect and rebuilds its sums when that
    happens, so the two backends now agree window by window.

    Here each window is centred on its own means (two passes) before the
    cross-products are formed, so no window carries anything from another.
    That is O(n * window) rather than O(n), which is the price of being the
    reference the fast path is checked against; blocks keep the memory
    bounded.

    A window whose benchmark is flat (maximum equal to minimum) has no
    variance to regress against and is NaN, as in the native kernel. The
    test is on the values, not on the centred sum of squares: centring a
    run of identical values on a mean that differs from them in the last
    bit leaves a sum of squares of order 1e-36, and a slope divided by it.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return out
    x_windows = sliding_window_view(x, window)
    y_windows = sliding_window_view(y, window)
    rows = max(1, _BLOCK_ELEMENTS // window)
    for start in range(0, x_windows.shape[0], rows):
        xs = x_windows[start : start + rows]
        ys = y_windows[start : start + rows]
        xc = xs - xs.mean(axis=1, keepdims=True)
        yc = ys - ys.mean(axis=1, keepdims=True)
        sxx = np.einsum("ij,ij->i", xc, xc)
        sxy = np.einsum("ij,ij->i", xc, yc)
        usable = (xs.max(axis=1) > xs.min(axis=1)) & (sxx > 0)
        beta = np.full(len(xs), np.nan)
        beta[usable] = sxy[usable] / sxx[usable]
        out[window - 1 + start : window - 1 + start + len(xs)] = beta
    return out
