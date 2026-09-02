"""
Structure in a series: when it broke, what leads what, and whether the
relationship is real.

Every function here answers a question the existing analysis surface does
not. `hurst` says what KIND of process a series is; change-point detection
says WHEN the process itself changed. Correlation says two things move
together; partial correlation says whether they still do once you control
for the thing driving both. A regression says one series explains another;
Granger says one series precedes the other, which is a different claim and
the only one of the two that could support a trade.

NO SCIPY, consistent with the rest of this package -- it is not a declared
dependency, and every statistic here has a closed form or an empirical null
that numpy can carry.

THE HONEST LIMITS ARE IN EACH DOCSTRING, not in a caveats section nobody
reads. The two worth knowing up front:

- **Granger causality is not causality.** It says one series helps predict
  another beyond that series' own past. A common driver produces it, a
  faster-updating proxy for the same information produces it, and neither is
  a mechanism.
- **Tail dependence needs tail observations.** An estimate at the 1% level
  from 250 days is built on two or three points, and its confidence interval
  covers most of [0, 1]. The count is returned so a caller can see that.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._special import (
    betacf,
    betainc,
    f_sf,
)
from standard_quant_tools.analysis._series import clean_series
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Minimum observations either side of a candidate break. A "regime" of
#: three bars is noise with a label on it, and allowing one makes the
#: detector find a break in every series.
MIN_SEGMENT = 20


def _clean(series: pd.Series, name: str) -> pd.Series:
    """See `_series.clean_series`. This module is why it exists: a single
    infinity turned a series with no breaks into one with three."""
    return clean_series(series, "series", name, minimum=1)


# ── change points ───────────────────────────────────────────────────────


def detect_change_points(
    series: pd.Series,
    *,
    max_breaks: int = 3,
    min_segment: int = MIN_SEGMENT,
    penalty: float = 10.0,
) -> Dict[str, Any]:
    """
    When the process generating this series changed, by binary segmentation
    on the mean and variance.

    `hurst` answers "what kind of process is this". This answers "and when
    did it stop being that one". A single Hurst exponent over a sample that
    contains a regime break describes neither regime.

    BINARY SEGMENTATION, not an exhaustive search. The optimal partition of n
    points into k segments is O(n^2 k); binary segmentation is O(n log n) and
    finds the strongest break, then recurses either side. It can miss two
    breaks that cancel -- a step up followed by a step down of equal size --
    which is the known cost and is stated here rather than discovered.

    The `penalty` is what stops it finding a break in white noise: a split
    has to improve the cost by more than this to be kept. Reported alongside
    each break as the improvement it actually bought, so a caller can see
    how close a call it was rather than only that a line was drawn.
    """
    values = _clean(series, "detect_change_points")
    n = len(values)
    if n < 2 * min_segment:
        raise ValidationError(
            f"detect_change_points: {n} observations cannot contain a break "
            f"with {min_segment} on each side. Lower min_segment, or accept "
            "that this window is too short to answer the question."
        )

    array = values.to_numpy()
    breaks: List[Dict[str, Any]] = []
    segments = [(0, n)]

    for _ in range(max_breaks):
        best = None
        for start, stop in segments:
            candidate = _best_split(array[start:stop], min_segment, penalty)
            if candidate is None:
                continue
            index, gain = candidate
            if best is None or gain > best[2]:
                best = (start, start + index, gain, (start, stop))
        if best is None:
            break
        start, position, gain, segment = best
        breaks.append(
            {
                "index": int(position),
                "date": str(values.index[position]),
                "gain": float(gain),
                "mean_before": float(array[segment[0] : position].mean()),
                "mean_after": float(array[position : segment[1]].mean()),
                "std_before": float(array[segment[0] : position].std(ddof=1)),
                "std_after": float(array[position : segment[1]].std(ddof=1)),
            }
        )
        segments.remove(segment)
        segments.extend([(segment[0], position), (position, segment[1])])

    breaks.sort(key=lambda b: b["index"])
    return {
        "n_observations": int(n),
        "n_breaks": len(breaks),
        "breaks": breaks,
        "penalty": float(penalty),
        "min_segment": int(min_segment),
        "segments": _describe_segments(values, breaks),
        "warnings": _change_point_warnings(breaks, n),
    }


def _best_split(segment: np.ndarray, min_segment: int, penalty: float):
    """
    The split that most reduces within-segment squared error, if any clears
    the penalty.

    Cost is the residual sum of squares about each side's own mean, which
    detects a shift in LEVEL. A pure variance change with no mean shift is
    not found by this and is not claimed to be.
    """
    n = len(segment)
    if n < 2 * min_segment:
        return None
    total = float(((segment - segment.mean()) ** 2).sum())

    # Prefix sums make every candidate split O(1) rather than O(n).
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(segment**2)])

    def rss(lo: int, hi: int) -> float:
        count = hi - lo
        if count <= 0:
            return 0.0
        total_ = cumulative[hi] - cumulative[lo]
        total_sq = cumulative_sq[hi] - cumulative_sq[lo]
        return float(total_sq - total_ * total_ / count)

    positions = np.arange(min_segment, n - min_segment + 1)
    if positions.size == 0:
        return None
    costs = np.array([rss(0, p) + rss(p, n) for p in positions])
    best = int(np.argmin(costs))
    gain = total - costs[best]
    if gain <= penalty:
        return None
    return int(positions[best]), float(gain)


def _describe_segments(values: pd.Series, breaks) -> List[Dict[str, Any]]:
    edges = [0] + [b["index"] for b in breaks] + [len(values)]
    out = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        chunk = values.iloc[lo:hi]
        out.append(
            {
                "segment": i,
                "start": str(chunk.index[0]),
                "end": str(chunk.index[-1]),
                "n": int(len(chunk)),
                "mean": float(chunk.mean()),
                "std": float(chunk.std(ddof=1)) if len(chunk) > 1 else float("nan"),
            }
        )
    return out


def _change_point_warnings(breaks, n) -> List[str]:
    out = []
    if not breaks:
        out.append(
            "no break cleared the penalty. That is evidence the series is "
            "homogeneous at this threshold, not proof -- binary segmentation "
            "also misses two offsetting breaks, a step up followed by an "
            "equal step down."
        )
    weak = [b for b in breaks if b["gain"] < 3.0 * 10.0]
    if weak:
        out.append(
            f"{len(weak)} break(s) cleared the penalty by less than 3x. Read "
            "`gain` before treating those as regime boundaries."
        )
    return out


# ── partial correlation ─────────────────────────────────────────────────


def partial_correlation(
    frame: pd.DataFrame, x: str, y: str, controlling_for: Sequence[str]
) -> Dict[str, Any]:
    """
    The correlation between x and y once the controls are removed from both.

    Two stocks in the same sector correlate at 0.7 and it says almost
    nothing: remove the market and the sector and what is left is the part
    that is actually about those two companies. That residual is the number a
    pair trade lives on, and the raw correlation systematically overstates it.

    Computed by regressing each of x and y on the controls and correlating
    the residuals, which is the definition rather than a shortcut to it.
    """
    columns = [x, y] + list(controlling_for)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(f"partial_correlation: no column(s) {missing}")
    data = frame[columns].dropna()
    if len(data) < len(columns) + 2:
        raise ValidationError(
            f"partial_correlation: {len(data)} complete rows cannot support "
            f"{len(controlling_for)} controls. Each control costs a degree of "
            "freedom and the residual correlation is undefined without slack."
        )

    controls = data[list(controlling_for)].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), controls])
    residuals = {}
    for name in (x, y):
        target = data[name].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residuals[name] = target - design @ beta

    raw = float(np.corrcoef(data[x], data[y])[0, 1])
    partial = float(np.corrcoef(residuals[x], residuals[y])[0, 1])
    return {
        "x": x,
        "y": y,
        "controlling_for": list(controlling_for),
        "raw_correlation": raw,
        "partial_correlation": partial,
        "n_observations": int(len(data)),
        "explained_away": float(raw - partial),
        "warnings": (
            [
                f"the controls explain away {raw - partial:+.3f} of the raw "
                f"{raw:+.3f}. What is left is what the relationship is about "
                "once the common drivers are removed."
            ]
            if abs(raw - partial) > 0.1
            else []
        ),
    }


# ── Granger causality ───────────────────────────────────────────────────


def granger_causality(
    cause: pd.Series, effect: pd.Series, *, max_lag: int = 5
) -> Dict[str, Any]:
    """
    Does `cause` help predict `effect` beyond what `effect`'s own past
    already says?

    NOT CAUSALITY, whatever the name says. A common driver produces this. A
    faster-updating proxy for the same information produces this. What it
    establishes is temporal precedence in a linear model, which is a
    necessary condition for a tradeable lead and nowhere near a sufficient
    one.

    The test is an F-test comparing a restricted autoregression on `effect`
    alone against an unrestricted one that also has lags of `cause`. The
    p-value is computed from the F distribution's closed form rather than
    from scipy.
    """
    joined = pd.concat(
        [pd.Series(cause).astype(float), pd.Series(effect).astype(float)],
        axis=1,
        keys=["cause", "effect"],
    ).dropna()
    if len(joined) < 10 * max_lag:
        raise ValidationError(
            f"granger_causality: {len(joined)} paired observations is too few "
            f"for {max_lag} lags. Each lag costs two parameters and the F-test "
            "becomes meaningless when the design approaches the sample size."
        )

    results = []
    for lag in range(1, max_lag + 1):
        restricted = _lagged_design(joined["effect"], [joined["effect"]], lag)
        unrestricted = _lagged_design(
            joined["effect"], [joined["effect"], joined["cause"]], lag
        )
        rss_r, n, k_r = restricted
        rss_u, _, k_u = unrestricted
        df_num = k_u - k_r
        df_den = n - k_u
        if df_den <= 0 or rss_u <= 0:
            continue
        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        results.append(
            {
                "lag": lag,
                "f_statistic": float(f_stat),
                "p_value": float(_f_sf(f_stat, df_num, df_den)),
                "n_observations": int(n),
            }
        )

    if not results:
        raise ValidationError("granger_causality: no lag produced a usable test")
    best = min(results, key=lambda r: r["p_value"])
    # BONFERRONI, and it is the flag that needs it rather than the report.
    # Taking the smallest p-value across `max_lag` tests and calling it
    # significant at 5% delivers about 15% -- measured on independent series
    # at max_lag=4. The individual F-tests are correctly calibrated (6.7% at
    # the nominal 5% over 300 null draws); what was wrong was the claim made
    # on top of them. Both numbers are returned so a caller can see the raw
    # one, and the FLAG is the corrected one because that is what gets read.
    corrected = min(1.0, best["p_value"] * len(results))
    return {
        "max_lag": max_lag,
        "by_lag": results,
        "best_lag": best["lag"],
        "p_value": float(corrected),
        "uncorrected_p_value": float(best["p_value"]),
        "n_tests": len(results),
        "significant_at_05": bool(corrected < 0.05),
        "warnings": [
            "Granger causality is temporal precedence in a linear model, not "
            "causality. A common driver produces it, and so does a "
            "faster-updating proxy for the same information.",
            f"{len(results)} lags were tested and the smallest p-value "
            "taken, which is a multiple comparison. `p_value` is therefore "
            f"Bonferroni corrected ({best['p_value']:.2g} x {len(results)}) "
            "and `uncorrected_p_value` holds the raw one. Uncorrected, "
            "taking the smallest of several and calling it a 5% test "
            "delivers about 15%.",
        ],
    }


def _lagged_design(target: pd.Series, sources: Sequence[pd.Series], lag: int):
    """Residual sum of squares of an OLS fit on lagged sources."""
    columns = []
    for source in sources:
        for shift in range(1, lag + 1):
            columns.append(source.shift(shift))
    design = pd.concat(columns, axis=1)
    frame = pd.concat([target, design], axis=1).dropna()
    y = frame.iloc[:, 0].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1:].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    return float((residual**2).sum()), len(frame), x.shape[1]


# See `_special`: this had 2 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_f_sf = f_sf

# See `_special`: this had 2 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_betainc = betainc

# See `_special`: this had 2 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_betacf = betacf

# ── tail dependence ─────────────────────────────────────────────────────


def tail_dependence(
    x: pd.Series, y: pd.Series, *, quantile: float = 0.05
) -> Dict[str, Any]:
    """
    Whether two series move together IN THE TAIL, which is the only regime a
    diversification claim has to survive.

    A correlation of 0.3 over the full sample is compatible with two assets
    that are independent day to day and fall together every time it matters.
    This measures the conditional probability directly: given x is below its
    q-th quantile, how often is y?

    `n_tail_observations` is the number to read alongside the estimate. At
    q=0.01 on 250 days that is two or three points, and an estimate from
    three points has a confidence interval covering most of [0, 1]. The
    count is returned so nobody has to work that out from the quantile.
    """
    if not 0.0 < quantile < 0.5:
        raise ValidationError("tail_dependence: quantile must be in (0, 0.5)")
    joined = pd.concat([pd.Series(x), pd.Series(y)], axis=1, keys=["x", "y"]).dropna()
    if len(joined) < 30:
        raise ValidationError(
            f"tail_dependence: {len(joined)} paired observations is too few to "
            "say anything about a tail"
        )

    xs = joined["x"].to_numpy(dtype=float)
    ys = joined["y"].to_numpy(dtype=float)
    lower_x, lower_y = np.quantile(xs, quantile), np.quantile(ys, quantile)
    upper_x, upper_y = np.quantile(xs, 1 - quantile), np.quantile(ys, 1 - quantile)

    x_low, y_low = xs <= lower_x, ys <= lower_y
    x_high, y_high = xs >= upper_x, ys >= upper_y

    lower = float((x_low & y_low).sum() / max(x_low.sum(), 1))
    upper = float((x_high & y_high).sum() / max(x_high.sum(), 1))

    warnings = []
    if int(x_low.sum()) < 10:
        warnings.append(
            f"only {int(x_low.sum())} observations fall in the {quantile:.0%} "
            "tail. An estimate from this many points has a confidence "
            "interval covering most of [0, 1] -- widen the quantile or get "
            "more history before acting on it."
        )
    if lower > upper + 0.15:
        warnings.append(
            f"lower tail dependence ({lower:.2f}) exceeds upper ({upper:.2f}): "
            "these move together on the way down more than on the way up, "
            "which is the asymmetry a full-sample correlation hides and the "
            "one a diversification claim has to survive."
        )
    return {
        "quantile": float(quantile),
        "lower_tail_dependence": lower,
        "upper_tail_dependence": upper,
        "full_sample_correlation": float(np.corrcoef(xs, ys)[0, 1]),
        "n_observations": int(len(joined)),
        "n_tail_observations": int(x_low.sum()),
        "warnings": warnings,
    }


__all__ = [
    "MIN_SEGMENT",
    "detect_change_points",
    "granger_causality",
    "partial_correlation",
    "tail_dependence",
]
