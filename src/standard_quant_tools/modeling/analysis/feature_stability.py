"""
Is this feature real, and does it stay real?

`feature_report.py` answers "what does this feature look like, over the
whole sample". Three questions it does not answer, and each of them has
talked somebody out of a strategy:

- **Drift.** A feature computed on 2015-2019 data and deployed in 2024 may
  no longer be the same measurement. The IC over the full sample averages
  across that, and an average across a break describes neither side of it.
- **Stability.** A mean IC of 0.04 is one thing if it is 0.04 every year and
  another if it is 0.30 in one year and -0.05 in the rest. The second is
  not a weaker version of the first; it is a different claim about the world.
- **Significance.** An IC of 0.03 on 60 dates and 20 entities is a number
  you can get from noise. The only honest way to know is to ask how often
  noise produces it, which is what the permutation test does.

NO SCIPY. It is not a declared dependency of this package, and
`feature_report.py` already treats "needs no scipy" as a reason to prefer
one implementation over another. The two statistics here that would usually
come from it -- PSI and the two-sample KS -- are a few lines of numpy each,
and writing them out keeps the dependency surface where it is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.validation.metrics import (
    check_ic_method,
)
from standard_quant_tools.modeling.features.transforms import (
    HAS_CPP,
    _cpp_core,
)

from ..validation.metrics import cross_sectional_ic

logger = logging.getLogger(__name__)

#: PSI convention. These are the thresholds the credit-risk literature
#: settled on and they are conventions rather than tests -- there is no null
#: distribution behind them. Reported so an agent does not have to invent
#: its own cutoff, labelled so it does not mistake them for a p-value.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

#: Bins for the PSI histogram. Ten is the usual choice; the statistic is
#: mildly sensitive to it, which is why the bin count is reported.
_PSI_BINS = 10

#: Guards a PSI term against a bucket that is empty in one window. Without
#: it a single empty bucket sends the statistic to infinity, which says
#: "infinitely drifted" when it means "no observations here".
_PSI_FLOOR = 1e-6


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _require(panel: pd.DataFrame, feature: str) -> None:
    missing = [c for c in ("date", "entity", "target") if c not in panel.columns]
    if missing:
        raise ValidationError(f"panel is missing required columns: {missing}")
    if feature not in panel.columns:
        raise ValidationError(f"panel has no feature {feature!r}")


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = _PSI_BINS
) -> float:
    """
    How far `current` has moved from `reference`, in PSI.

    Bin edges come from the REFERENCE window's quantiles, not from the
    pooled sample. Pooling would let the current window move the edges it is
    being measured against, which mutes exactly the drift the statistic is
    for.
    """
    reference, current = _finite(reference), _finite(current)
    if reference.size == 0 or current.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant reference window has no distribution to drift from.
        return 0.0 if np.allclose(current, reference[0]) else float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size
    ref_pct = np.clip(ref_pct, _PSI_FLOOR, None)
    cur_pct = np.clip(cur_pct, _PSI_FLOOR, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two
    empirical CDFs.

    Written out rather than imported because scipy is not a dependency here.
    The implementation is the textbook one -- evaluate both CDFs at every
    observed value and take the maximum absolute difference.
    """
    reference, current = _finite(reference), _finite(current)
    if reference.size == 0 or current.size == 0:
        return float("nan")
    reference = np.sort(reference)
    current = np.sort(current)
    grid = np.concatenate([reference, current])
    cdf_ref = np.searchsorted(reference, grid, side="right") / reference.size
    cdf_cur = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def feature_drift(
    panel: pd.DataFrame,
    feature: str,
    *,
    split_date: Optional[str] = None,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    How one feature's distribution -- and its IC -- differ either side of a
    date.

    Both halves matter and they fail differently. A feature can drift in
    distribution while keeping its IC (a rescaling), or hold its
    distribution while losing its IC (the relationship decayed). The first
    is a preprocessing problem; the second means the edge is gone. Reporting
    only one of them invites fixing the wrong thing.

    `split_date` defaults to the median date, which splits the panel into
    equal halves by TIME rather than by row count -- an entity that joins
    the universe late should not drag the boundary.
    """
    check_ic_method(method, what="feature_drift")
    _require(panel, feature)
    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no observations")

    dates = pd.to_datetime(frame["date"])
    boundary = pd.Timestamp(split_date) if split_date else dates.median()
    before = frame[dates < boundary]
    after = frame[dates >= boundary]

    if before.empty or after.empty:
        raise ValidationError(
            f"split at {boundary.date()} leaves one side empty "
            f"({len(before)} before, {len(after)} after). Pick a split_date "
            "inside the panel's range."
        )

    ref = before[feature].to_numpy(dtype=float)
    cur = after[feature].to_numpy(dtype=float)
    psi = population_stability_index(ref, cur)

    def _ic(part: pd.DataFrame) -> float:
        usable = part.dropna(subset=["target"])
        if usable.empty:
            return float("nan")
        series = cross_sectional_ic(
            usable["target"].to_numpy(dtype=float),
            usable[feature].to_numpy(dtype=float),
            usable["date"].to_numpy(),
            method=method,
        )
        return float(series.mean()) if len(series) else float("nan")

    ic_before, ic_after = _ic(before), _ic(after)
    verdict = (
        "significant"
        if psi >= PSI_SIGNIFICANT
        else "moderate" if psi >= PSI_MODERATE else "stable"
    )

    ic_flipped = (
        np.isfinite(ic_before)
        and np.isfinite(ic_after)
        and np.sign(ic_before) != np.sign(ic_after)
        and abs(ic_before) > 0.01
        and abs(ic_after) > 0.01
    )

    return {
        "feature": feature,
        "split_date": str(boundary.date()),
        "n_before": int(len(before)),
        "n_after": int(len(after)),
        "psi": psi,
        "psi_bins": _PSI_BINS,
        "psi_verdict": verdict,
        "ks_statistic": ks_statistic(ref, cur),
        "mean_before": float(np.nanmean(ref)) if ref.size else float("nan"),
        "mean_after": float(np.nanmean(cur)) if cur.size else float("nan"),
        "std_before": float(np.nanstd(ref)) if ref.size else float("nan"),
        "std_after": float(np.nanstd(cur)) if cur.size else float("nan"),
        "ic_before": ic_before,
        "ic_after": ic_after,
        "ic_flipped": bool(ic_flipped),
    }


def feature_stability(
    panel: pd.DataFrame,
    feature: str,
    *,
    n_blocks: int = 4,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    The feature's IC computed inside each of `n_blocks` contiguous time
    blocks.

    Contiguous, never shuffled. A feature's whole problem is usually that it
    worked in one regime, and randomly interleaved folds would average that
    away -- which is the failure this function exists to expose rather than
    to reproduce.

    `sign_consistency` is the number to read first: the fraction of blocks
    whose IC has the same sign as the full-sample IC. A mean IC of 0.04 at
    0.5 sign consistency is a coin flip with a good average.
    """
    check_ic_method(method, what="regime_stability")
    _require(panel, feature)
    if n_blocks < 2:
        raise ValidationError("n_blocks must be at least 2")

    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature, "target"]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no usable observations")

    dates = np.sort(pd.to_datetime(frame["date"]).unique())
    if len(dates) < n_blocks:
        raise ValidationError(
            f"{len(dates)} dates cannot be split into {n_blocks} blocks"
        )
    chunks = np.array_split(dates, n_blocks)

    blocks: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        part = frame[pd.to_datetime(frame["date"]).isin(chunk)]
        series = (
            cross_sectional_ic(
                part["target"].to_numpy(dtype=float),
                part[feature].to_numpy(dtype=float),
                part["date"].to_numpy(),
                method=method,
            )
            if not part.empty
            else pd.Series(dtype=float)
        )
        blocks.append(
            {
                "block": index,
                "start": str(pd.Timestamp(chunk[0]).date()),
                "end": str(pd.Timestamp(chunk[-1]).date()),
                "n_dates": int(len(chunk)),
                "ic_mean": float(series.mean()) if len(series) else float("nan"),
            }
        )

    ics = np.array([b["ic_mean"] for b in blocks], dtype=float)
    usable = _finite(ics)
    overall = cross_sectional_ic(
        frame["target"].to_numpy(dtype=float),
        frame[feature].to_numpy(dtype=float),
        frame["date"].to_numpy(),
        method=method,
    )
    overall_ic = float(overall.mean()) if len(overall) else float("nan")
    consistency = (
        float(np.mean(np.sign(usable) == np.sign(overall_ic)))
        if usable.size and np.isfinite(overall_ic)
        else float("nan")
    )

    return {
        "feature": feature,
        "n_blocks": n_blocks,
        "blocks": blocks,
        "ic_overall": overall_ic,
        "ic_block_mean": float(np.mean(usable)) if usable.size else float("nan"),
        "ic_block_std": float(np.std(usable)) if usable.size else float("nan"),
        "ic_block_min": float(np.min(usable)) if usable.size else float("nan"),
        "ic_block_max": float(np.max(usable)) if usable.size else float("nan"),
        "sign_consistency": consistency,
        "worst_block": (
            int(
                blocks[int(np.argmin(np.where(np.isfinite(ics), ics, np.inf)))]["block"]
            )
            if usable.size
            else None
        ),
    }


def _null_distribution(
    target: np.ndarray,
    values: np.ndarray,
    dates: np.ndarray,
    n_permutations: int,
    method: str,
    random_seed: int,
) -> np.ndarray:
    """
    One mean IC per draw, from the kernel when it is present.

    The whole loop goes across the boundary rather than the correlation
    alone. Called from Python it is `n_permutations` shuffles, the same
    number of correlation calls, and the same number of round trips through
    pandas -- and that last part was about a third of the cost, for objects
    nobody looks at. Fusing also lets the ranking happen ONCE: shuffling
    values inside a date permutes their ranks, so for spearman the ranks can
    be shuffled directly instead of recomputed on every draw.

    Two numpy attempts at the same idea measured SLOWER than the loop they
    replaced (0.14x for a global lexsort, 0.6x for rank-once in numpy),
    because the existing kernel counting-sorts in O(n) and numpy has to
    sort. That is why this is C++ and not a rewrite.

    REPRODUCIBILITY IS WITHIN A BACKEND. The kernel uses its own generator,
    not a reimplementation of numpy's PCG64 bit stream, so the same
    `random_seed` gives different DRAWS with and without the extension --
    the contract `simulate_forward_paths` already states. The null they are
    drawn from is the same: measured against the analytic standard deviation
    of the mean IC under the null, the kernel lands at 1.001x and the Python
    path at 1.006x, and a permutation p-value is a property of that null
    rather than of any particular draw.
    """
    if HAS_CPP and hasattr(_cpp_core, "permutation_null_ic"):
        codes, uniques = pd.factorize(dates, sort=True)
        return np.asarray(
            _cpp_core.permutation_null_ic(
                np.ascontiguousarray(target, dtype=np.float64),
                np.ascontiguousarray(values, dtype=np.float64),
                np.ascontiguousarray(codes.astype(np.int64)),
                int(len(uniques)),
                int(n_permutations),
                int(random_seed) & 0xFFFFFFFFFFFFFFFF,
                method == "spearman",
            ),
            dtype=float,
        )

    # Row positions per date, computed once: the shuffle is the inner loop.
    groups = [np.flatnonzero(dates == d) for d in pd.unique(dates)]
    rng = np.random.default_rng(random_seed)
    null = np.empty(n_permutations, dtype=float)
    shuffled = values.copy()
    for i in range(n_permutations):
        for positions in groups:
            shuffled[positions] = rng.permutation(values[positions])
        series = cross_sectional_ic(target, shuffled, dates, method=method)
        null[i] = float(series.mean()) if len(series) else np.nan
    return null


def permutation_test_ic(
    panel: pd.DataFrame,
    feature: str,
    *,
    n_permutations: int = 200,
    method: str = "spearman",
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    How often noise produces an IC this large.

    The feature is shuffled WITHIN each date, which states the null exactly:
    "this feature carries no cross-sectional information within a date". It
    preserves the entities per date, the feature's marginal distribution and
    the target's cross-sectional shape, so nothing but the link is destroyed.

    Measured, a global shuffle produces a null within 2% of this one -- and
    on reflection that is expected rather than surprising, since the IC is
    computed within each date and averaged, so a global shuffle also
    delivers a random assignment inside each date. Within-date is kept
    because it is the null as stated and holds by construction on panel
    shapes that have not been measured, not because the alternative was
    found to be dramatically wrong.

    Returns a two-sided empirical p-value with the +1 correction in both
    numerator and denominator, so a p of exactly 0 is never reported --
    200 permutations cannot distinguish "p < 0.005" from "p = 0", and
    printing 0.0 claims a precision the sample size does not have.
    """
    check_ic_method(method, what="permutation_test")
    _require(panel, feature)
    if n_permutations < 1:
        raise ValidationError("n_permutations must be at least 1")

    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature, "target"]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no usable observations")

    dates = frame["date"].to_numpy()
    target = frame["target"].to_numpy(dtype=float)
    values = frame[feature].to_numpy(dtype=float)

    observed_series = cross_sectional_ic(target, values, dates, method=method)
    observed = float(observed_series.mean()) if len(observed_series) else float("nan")
    if not np.isfinite(observed):
        raise ValidationError(
            f"feature {feature!r} has no computable IC, so there is nothing "
            "to test for significance"
        )

    null = _null_distribution(
        target, values, dates, n_permutations, method, random_seed
    )

    usable = _finite(null)
    at_least_as_extreme = int(np.sum(np.abs(usable) >= abs(observed)))
    p_value = (at_least_as_extreme + 1) / (usable.size + 1)

    return {
        "feature": feature,
        "observed_ic": observed,
        "n_permutations": int(n_permutations),
        "n_usable_permutations": int(usable.size),
        "null_mean": float(np.mean(usable)) if usable.size else float("nan"),
        "null_std": float(np.std(usable)) if usable.size else float("nan"),
        "null_p95_abs": (
            float(np.quantile(np.abs(usable), 0.95)) if usable.size else float("nan")
        ),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "random_seed": int(random_seed),
    }


__all__ = [
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
    "feature_drift",
    "feature_stability",
    "ks_statistic",
    "permutation_test_ic",
    "population_stability_index",
]
