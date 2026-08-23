"""
What a feature is worth, before any model is fitted.

The existing diagnostics answer "which columns did this estimator lean on",
read off `coef_` / `feature_importances_`. That is a statement about one fit,
in units that differ per estimator, and it cannot distinguish a feature that
carries real information from one the model happened to latch onto. It also
arrives too late to be useful: by then the features are already chosen.

This module answers the earlier question — *is this a good feature* — with
four things an agent can act on:

  DISTRIBUTION   how well populated and how well behaved the column is, and
                 how fast it turns over (a feature that reshuffles the
                 cross-section every day is expensive to trade even when it
                 predicts)
  PREDICTIVE     cross-sectional IC and ICIR, plus the decile spread and
                 monotonicity that say whether the relationship is usable or
                 just statistically present
  REDUNDANCY     which features are restatements of one another, so an agent
                 does not put four expressions of the same latent variable
                 into one model and read the split importances as four
                 separate findings
  LEAKAGE        whether the feature's information is actually available when
                 it claims to be

None of it needs a fitted model, so it runs on a dataset alone.

ONE HORIZON, FOR NOW. Everything here is measured against the panel's own
`target` column, because that is the only target a built dataset carries. The
more useful question — *at what horizon* is this feature predictive — needs
multi-horizon targets in the dataset first; the shape of this module is
deliberately per-(feature, target) so that becomes a loop rather than a
rewrite.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from ..validation.metrics import cross_sectional_ic, summarize_cross_sectional_ic

logger = logging.getLogger(__name__)

# |z| beyond this counts as an outlier for the coverage report. Deliberately
# wide: at 4 sigma a normal column contributes 0.006% of its rows, so anything
# materially above that is a real tail rather than a threshold artefact.
_OUTLIER_SIGMA = 4.0

# Deciles by default. Fewer buckets hide the shape; more of them put too few
# entities in each bucket to mean anything on a small universe.
_DEFAULT_QUANTILES = 10

# A feature must clear this |IC| at shift 0 before the leakage screen will
# flag it, and the margin is deliberately enormous next to what a real signal
# looks like. A rank IC of 0.02-0.03 is a respectable financial feature; 0.05
# is a good one. A feature that is actually reading its own answer produces
# something like 0.99. Setting the floor at 0.05 therefore separates the two
# cases by more than an order of magnitude, and the whole band where "leak"
# and "genuinely skilful" would be hard to tell apart sits below it.
#
# Set low (0.01 was tried) the screen fires on features whose IC is a few
# thousandths — noise, where the shape of the curve means nothing.
_LEAKAGE_MIN_ABS_IC = 0.05

# Above this self-correlation across the shift window, advancing the feature
# cannot reveal much because the feature barely moved. A flat curve is then
# explained by persistence, not by leakage, and the screen abstains.
_LEAKAGE_MAX_PERSISTENCE = 0.95


def _require_columns(panel: pd.DataFrame, feature_ids: Sequence[str]) -> None:
    missing = [c for c in ("date", "entity", "target") if c not in panel.columns]
    if missing:
        raise ValidationError(
            f"feature report: panel is missing required column(s) {missing}. "
            "Pass the frame returned by build_model_dataset under 'panel'."
        )
    unknown = [f for f in feature_ids if f not in panel.columns]
    if unknown:
        raise ValidationError(
            f"feature report: panel has no column(s) {unknown}. "
            f"Available: {sorted(c for c in panel.columns if c not in ('date', 'entity', 'target'))}"
        )


def _safe(value: Any) -> float:
    """A float that survives JSON, with non-finite mapped to NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


# ── Distribution ──────────────────────────────────────────────────────────


def feature_distribution_stats(
    panel: pd.DataFrame, feature_ids: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """
    How well populated and how well behaved each feature is.

    `autocorrelation` is measured WITHIN each entity and averaged, not over
    the stacked column: a pooled autocorrelation over a long panel mostly
    measures the fact that consecutive rows belong to different entities.

    `turnover` is the mean absolute change in an entity's within-date rank
    between consecutive dates, normalized so 0 means the ordering never moves
    and 1 means it is redrawn at random each day. It belongs in a feature
    report because it is the part of a feature's cost that IC cannot see: two
    features with the same IC and very different turnover are not equally
    useful, and the fast one may not survive its own trading costs.
    """
    out: Dict[str, Dict[str, float]] = {}
    n_rows = len(panel)
    grouped = panel.groupby("entity", sort=False)

    for feature in feature_ids:
        column = panel[feature]
        present = column.notna()
        n_present = int(present.sum())
        values = column[present]

        if n_present == 0:
            out[feature] = {
                "coverage": 0.0,
                "n_missing": float(n_rows),
                "mean": float("nan"),
                "std": float("nan"),
                "skew": float("nan"),
                "kurtosis": float("nan"),
                "outlier_rate": float("nan"),
                "autocorrelation": float("nan"),
                "turnover": float("nan"),
            }
            continue

        std = float(values.std())
        if std > 0 and np.isfinite(std):
            z = (values - float(values.mean())) / std
            outlier_rate = float((z.abs() > _OUTLIER_SIGMA).mean())
        else:
            # A constant column has no dispersion, so nothing can be an
            # outlier in it. 0.0 is the answer, not "undefined".
            outlier_rate = 0.0

        autocorr = grouped[feature].apply(lambda s: s.autocorr(lag=1))
        out[feature] = {
            "coverage": _safe(n_present / n_rows) if n_rows else 0.0,
            "n_missing": float(n_rows - n_present),
            "mean": _safe(values.mean()),
            "std": _safe(std),
            "skew": _safe(values.skew()),
            "kurtosis": _safe(values.kurt()),
            "outlier_rate": _safe(outlier_rate),
            "autocorrelation": _safe(autocorr.mean()),
            "turnover": _safe(_rank_turnover(panel, feature)),
        }
    return out


def _rank_turnover(panel: pd.DataFrame, feature: str) -> float:
    """
    Mean absolute change in within-date rank, on a 0-1 scale.

    Ranks are normalized to [0, 1] inside each date first, so the number does
    not depend on how many entities happen to be present that day — otherwise
    a universe that grows over time would look like it turned over more.
    """
    frame = panel[["date", "entity", feature]].dropna()
    if frame.empty:
        return float("nan")
    grouped = frame.groupby("date")[feature]
    counts = grouped.transform("count")
    if (counts <= 1).all():
        return float("nan")
    ranks = grouped.rank(method="average")
    # (rank - 1) / (n - 1) maps to [0, 1]; a single-entity date has no
    # ordering to change and drops out.
    normalized = ((ranks - 1.0) / (counts - 1.0)).where(counts > 1)
    frame = frame.assign(_rank=normalized).dropna(subset=["_rank"])
    if frame.empty:
        return float("nan")
    frame = frame.sort_values(["entity", "date"])
    delta = frame.groupby("entity", sort=False)["_rank"].diff().abs()
    return float(delta.mean()) if delta.notna().any() else float("nan")


# ── Predictive ────────────────────────────────────────────────────────────


def feature_predictive_stats(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    n_quantiles: int = _DEFAULT_QUANTILES,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-sectional IC, ICIR, and the decile shape behind them.

    The IC comes from the same `cross_sectional_ic` the engine reports models
    on — deliberately, so a feature's standalone number and a model's number
    are the same quantity, computed by the same (native-accelerated) code.

    IC alone does not say whether a relationship is usable. A feature can have
    a respectable rank IC while all of it lives in one tail, which is a very
    different proposition from a relationship that holds across the whole
    cross-section. So two more numbers come with it:

      quantile_spread  mean target in the top bucket minus the bottom one,
                       in target units — what a long-short on this feature
                       alone would have captured per period, before costs
      monotonicity     rank correlation between bucket index and bucket mean,
                       which is 1.0 for a cleanly ordered relationship and
                       near 0 for one that is real but not monotone

    A feature with a good IC, a good spread and poor monotonicity is telling
    you it works at the extremes and not in the middle. That is worth knowing
    before it goes into a linear model.
    """
    dates = panel["date"].to_numpy()
    target = panel["target"].to_numpy(dtype=float)
    out: Dict[str, Dict[str, float]] = {}

    for feature in feature_ids:
        values = panel[feature].to_numpy(dtype=float)
        stats: Dict[str, float] = {}
        # Note the argument order: cross_sectional_ic(y_true, y_pred, ...) is
        # symmetric in the correlation, so passing the feature as the
        # "prediction" is exactly the right reading — this asks how well the
        # feature alone would have ranked the cross-section.
        for method, prefix in (("pearson", "ic"), ("spearman", "rank_ic")):
            series = cross_sectional_ic(target, values, dates, method)
            summary = summarize_cross_sectional_ic(series, prefix)
            stats.update({k: _safe(v) for k, v in summary.items()})
        stats.update(_quantile_shape(panel, feature, n_quantiles))
        out[feature] = stats
    return out


def _quantile_shape(
    panel: pd.DataFrame, feature: str, n_quantiles: int
) -> Dict[str, float]:
    """Bucket each date's cross-section by the feature, then look at the
    average target per bucket across dates."""
    frame = panel[["date", feature, "target"]].dropna()
    if frame.empty:
        return {
            "quantile_spread": float("nan"),
            "monotonicity": float("nan"),
            "n_quantiles": float(n_quantiles),
        }

    def _bucket(group: pd.Series) -> pd.Series:
        # Rank-then-cut rather than qcut on raw values: a feature with heavy
        # ties (a discretized or clipped column) makes qcut raise or produce
        # unequal buckets, and the rank is what the bucketing means anyway.
        count = group.notna().sum()
        if count < n_quantiles:
            return pd.Series(np.nan, index=group.index)
        ranks = group.rank(method="first")
        return np.ceil(ranks * n_quantiles / count).clip(1, n_quantiles)

    buckets = frame.groupby("date")[feature].transform(_bucket)
    frame = frame.assign(_bucket=buckets).dropna(subset=["_bucket"])
    if frame.empty:
        return {
            "quantile_spread": float("nan"),
            "monotonicity": float("nan"),
            "n_quantiles": float(n_quantiles),
        }

    # Mean target per bucket per date, then averaged over dates — not a
    # single pooled mean per bucket. Pooling would weight dates by how many
    # entities they happened to carry.
    per_date = frame.groupby(["date", "_bucket"])["target"].mean()
    bucket_means = per_date.groupby("_bucket").mean().sort_index()
    if len(bucket_means) < 2:
        return {
            "quantile_spread": float("nan"),
            "monotonicity": float("nan"),
            "n_quantiles": float(n_quantiles),
        }

    spread = float(bucket_means.iloc[-1] - bucket_means.iloc[0])
    monotonicity = float(
        pd.Series(bucket_means.index.astype(float)).corr(
            pd.Series(bucket_means.to_numpy()), method="spearman"
        )
    )
    return {
        "quantile_spread": _safe(spread),
        "monotonicity": _safe(monotonicity),
        "n_quantiles": float(len(bucket_means)),
    }


# ── Redundancy ────────────────────────────────────────────────────────────


def redundancy_report(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Which features are restatements of one another.

    An agent that puts RSI, the stochastic oscillator, 20-day momentum and
    MACD into one model has not supplied four pieces of evidence; it has
    supplied roughly one, four times. Every importance-style diagnostic then
    splits that one signal across four columns and reports each as modest,
    which is the opposite of the truth.

    Three views, because they fail differently:

      correlation       pairwise, easy to read, blind to the case where no
                        single pair is alarming but the set is jointly
                        near-degenerate
      vif               each feature against ALL the others at once, which
                        catches exactly that case
      condition_number  of the correlation matrix — one number for whether
                        the design is degenerate at all

    Clusters are formed by transitive closure over |correlation| above the
    threshold, which is deliberately the crude choice: it needs no scipy, it
    is easy to explain, and at a threshold this high the "chaining" that would
    make single-linkage clustering misleading is not a practical concern.
    """
    if len(feature_ids) < 2:
        return {
            "correlation": {},
            "spearman_correlation": {},
            "vif": {},
            "condition_number": float("nan"),
            "clusters": [[f] for f in feature_ids],
        }

    frame = panel[list(feature_ids)].dropna()
    if len(frame) < 2:
        return {
            "correlation": {},
            "spearman_correlation": {},
            "vif": {},
            "condition_number": float("nan"),
            "clusters": [[f] for f in feature_ids],
        }

    pearson = frame.corr(method="pearson")
    spearman = frame.corr(method="spearman")

    # VIF from the inverse correlation matrix: its diagonal IS 1/(1-R2_i),
    # which is the definition, and it costs one inversion instead of one
    # regression per feature.
    vif: Dict[str, float] = {}
    condition_number = float("nan")
    matrix = pearson.to_numpy(dtype=float)
    if np.all(np.isfinite(matrix)):
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            smallest = float(np.min(eigenvalues))
            largest = float(np.max(eigenvalues))
            if smallest > 0:
                condition_number = largest / smallest
            else:
                # Exactly singular: at least one feature is a linear
                # combination of the others. Infinity is the honest answer.
                condition_number = float("inf")
            inverse = np.linalg.pinv(matrix)
            for i, feature in enumerate(feature_ids):
                vif[feature] = _safe(inverse[i, i])
        except np.linalg.LinAlgError:  # pragma: no cover - pinv rarely fails
            logger.debug("[modeling] VIF unavailable: correlation matrix is degenerate")

    return {
        "correlation": _frame_to_nested(pearson),
        "spearman_correlation": _frame_to_nested(spearman),
        "vif": vif,
        "condition_number": condition_number,
        "clusters": _correlation_clusters(pearson, cluster_threshold),
    }


def _frame_to_nested(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        str(row): {str(col): _safe(frame.loc[row, col]) for col in frame.columns}
        for row in frame.index
    }


def _correlation_clusters(
    correlation: pd.DataFrame, threshold: float
) -> List[List[str]]:
    """Transitive closure over |corr| >= threshold, via union-find."""
    names = list(correlation.columns)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                union(left, right)

    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    # Largest first, so the thing an agent should look at is on top.
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ── Leakage ───────────────────────────────────────────────────────────────


def lead_lag_ic_curve(
    panel: pd.DataFrame,
    feature: str,
    *,
    max_shift: int = 5,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    IC of the feature against the SAME target, with the feature shifted in
    time — a causality screen.

    THE SIGN CONVENTION, because it is the whole test. A positive shift
    DELAYS the feature (row t is given the value from t-k, so it knows less);
    a negative shift ADVANCES it (row t is given the value from t+k, so it
    knows more). Shifting happens within each entity, never across the
    stacked panel.

    THREE SHAPES, AND ONLY ONE OF THEM IS A LEAK. Measured on real features,
    the curve comes in three distinct forms, and an early version of this
    screen flagged two of them wrongly because it only knew about the first:

      RAMP   a path-dependent feature (momentum, RSI, MACD). `target[t]`
             spans bars t..t+horizon, so the feature evaluated at t+k has
             already observed part of the answer, and advancing it improves
             IC a great deal. Measured: RSI went from -0.003 at shift 0 to
             +0.680 at shift -5. This is the causal signature.

      FLAT   a slow-moving STATE feature (realized volatility, ADX). Its
             predictive content is about the regime, not about the price
             path, so advancing it reveals almost nothing — and separately,
             a highly autocorrelated feature has barely changed over the
             shift window at all. Measured: realized volatility ran
             +0.005 -> +0.014 across the whole range. **This is innocent**,
             and it is the case the first version of this screen got wrong.

      TENT   a leak. IC peaks sharply AT shift 0 and falls away on BOTH
             sides, because the feature at t contains the answer and any
             displacement in either direction destroys the alignment.
             Measured on a planted leak: -0.001 / +0.762 / **+0.986** /
             +0.762 / -0.001.

    So the test is not "did advancing help" — that confuses RAMP with FLAT.
    It is: **is shift 0 a strict peak, and is that peak enormous?** Both
    conditions, because either alone produces false positives:

      peak    IC(0) must exceed both neighbours on its own sign. A FLAT curve
              has no peak; a RAMP has its maximum at the far left.
      floor   |IC(0)| must clear `_LEAKAGE_MIN_ABS_IC`. A curve made of noise
              has peaks everywhere and they mean nothing.

    `persistence` — the feature's self-correlation across the shift window —
    is reported alongside, and the screen abstains above
    `_LEAKAGE_MAX_PERSISTENCE`: a feature that barely moved cannot be
    expected to say anything new when advanced, so its flat curve carries no
    information either way.

    WHAT THIS DOES NOT CATCH. A leak smaller than the floor. A leak in a
    feature so persistent the screen abstains. And any leak that is constant
    across the whole sample, since this test works by displacement in time
    and a uniformly shifted feature is just a different feature. It is a
    SCREEN, not a proof: it tells an agent where to look.
    """
    if max_shift < 1:
        raise ValidationError(f"max_shift must be >= 1, got {max_shift}")

    frame = panel[["date", "entity", feature, "target"]].sort_values(["entity", "date"])
    grouped = frame.groupby("entity", sort=False)[feature]
    target = frame["target"].to_numpy(dtype=float)
    dates = frame["date"].to_numpy()

    curve: Dict[int, float] = {}
    for shift in range(-max_shift, max_shift + 1):
        shifted = grouped.shift(shift).to_numpy(dtype=float)
        series = cross_sectional_ic(target, shifted, dates, method)
        curve[shift] = _safe(series.mean()) if len(series) else float("nan")

    baseline = curve.get(0, float("nan"))
    persistence = _feature_persistence(frame, feature, max_shift)

    # Signed so that "larger" always means "further in the direction the
    # feature actually predicts" — otherwise a negative-IC feature would be
    # judged by whether its IC got less negative, which is backwards.
    sign = 1.0 if baseline >= 0 else -1.0
    left, right = curve.get(-1, float("nan")), curve.get(1, float("nan"))
    peak_ratio = float("nan")
    if np.isfinite(left) and np.isfinite(right) and baseline:
        neighbour = max(sign * left, sign * right)
        peak_ratio = _safe(sign * baseline / neighbour) if neighbour else float("inf")

    flagged = False
    if not np.isfinite(baseline) or abs(baseline) < _LEAKAGE_MIN_ABS_IC:
        reason = (
            f"|IC| at shift 0 is {abs(baseline):.4f}, below the "
            f"{_LEAKAGE_MIN_ABS_IC} floor. A feature reading its own answer scores "
            "an order of magnitude above this, so the screen cannot separate a "
            "small leak from ordinary skill here and declines to judge"
        )
    elif np.isfinite(persistence) and abs(persistence) > _LEAKAGE_MAX_PERSISTENCE:
        reason = (
            f"the feature is {persistence:.3f} self-correlated across +/-{max_shift} "
            "bars, so it barely moved over the window. Advancing it cannot reveal "
            "much, and a flat curve here says nothing either way"
        )
    elif not (np.isfinite(left) and np.isfinite(right)):
        reason = "IC is undefined at one of the neighbouring shifts"
    elif sign * baseline > sign * left and sign * baseline > sign * right:
        flagged = True
        reason = (
            f"IC peaks sharply AT shift 0 ({baseline:+.4f}) and falls away on both "
            f"sides ({left:+.4f} at -1, {right:+.4f} at +1) — the signature of a "
            "value at t that already contains the answer, since displacing it in "
            "either direction destroys the alignment. An honest feature's IC rises "
            "as it is advanced, because advancing lets it see more of the target "
            "window"
        )
    else:
        best_shift = min(curve, key=lambda k: -sign * curve[k])
        reason = (
            f"IC at shift 0 ({baseline:+.4f}) is not a peak; the curve is best at "
            f"shift {best_shift:+d} ({curve[best_shift]:+.4f}), which is the causal "
            "pattern rather than a leak"
        )

    return {
        "curve": {str(k): v for k, v in sorted(curve.items())},
        "ic_at_zero": baseline,
        "peak_ratio": peak_ratio,
        "persistence": persistence,
        "flagged": flagged,
        "reason": reason,
    }


def _feature_persistence(frame: pd.DataFrame, feature: str, shift: int) -> float:
    """
    How much the feature retains of itself across the shift window.

    Near 1.0 means the lead-lag scan has no power on this feature: it is
    comparing the value against a near-copy of itself, so a flat curve is
    explained by the feature not moving rather than by it knowing too much.
    """
    lagged = frame.groupby("entity", sort=False)[feature].shift(shift)
    both = frame[feature].notna() & lagged.notna()
    if int(both.sum()) < 3:
        return float("nan")
    left = frame[feature][both].to_numpy(dtype=float)
    right = lagged[both].to_numpy(dtype=float)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return _safe(np.corrcoef(left, right)[0, 1])


# ── The report ────────────────────────────────────────────────────────────


def build_feature_report(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    n_quantiles: int = _DEFAULT_QUANTILES,
    cluster_threshold: float = 0.9,
    leakage_max_shift: int = 5,
    include_leakage: bool = True,
) -> Dict[str, Any]:
    """
    Everything above, for one dataset, as a JSON-safe dict.

    `include_leakage` is separable because the lead-lag scan costs
    `2 * leakage_max_shift + 1` cross-sectional IC passes per feature, which
    is the expensive part of the report by a wide margin. It is on by default
    anyway: an agent that can compose features is exactly the caller most
    likely to produce one that cheats, and a screen nobody runs catches
    nothing.
    """
    feature_ids = list(feature_ids)
    _require_columns(panel, feature_ids)
    if not feature_ids:
        raise ValidationError("feature report: no features to analyze")

    report: Dict[str, Any] = {
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "n_entities": int(panel["entity"].nunique()),
        "n_features": len(feature_ids),
    }

    distribution = feature_distribution_stats(panel, feature_ids)
    predictive = feature_predictive_stats(panel, feature_ids, n_quantiles=n_quantiles)
    report["features"] = {
        feature: {**distribution[feature], **predictive[feature]}
        for feature in feature_ids
    }
    report["redundancy"] = redundancy_report(
        panel, feature_ids, cluster_threshold=cluster_threshold
    )

    if include_leakage:
        leakage = {
            feature: lead_lag_ic_curve(panel, feature, max_shift=leakage_max_shift)
            for feature in feature_ids
        }
        report["leakage"] = leakage
        report["leakage_flagged"] = sorted(
            f for f, v in leakage.items() if v["flagged"]
        )

    report["warnings"] = _report_warnings(report)
    return report


def _report_warnings(report: Dict[str, Any]) -> List[str]:
    """
    The findings worth surfacing without being asked.

    An agent reading a nested dict will not reliably notice that two features
    are 0.98 correlated. It will notice a sentence saying so.
    """
    warnings: List[str] = []
    features = report.get("features", {})

    thin = [f for f, s in features.items() if s.get("coverage", 1.0) < 0.5]
    if thin:
        warnings.append(
            f"NOTE: {len(thin)} feature(s) are populated on under half the panel "
            f"({', '.join(sorted(thin)[:5])}{'...' if len(thin) > 5 else ''}). "
            "Alignment drops any row where ANY requested feature is missing, so a "
            "thin feature costs rows for every other feature too."
        )

    clusters = [
        c for c in report.get("redundancy", {}).get("clusters", []) if len(c) > 1
    ]
    if clusters:
        biggest = max(clusters, key=len)
        warnings.append(
            f"NOTE: {len(clusters)} group(s) of features are near-duplicates of one "
            f"another, the largest being {biggest}. Importance-style diagnostics "
            "split one signal across such a group and report each member as modest, "
            "which reads as several weak findings rather than one strong one."
        )

    condition = report.get("redundancy", {}).get("condition_number", float("nan"))
    if np.isfinite(condition) and condition > 1000:
        warnings.append(
            f"NOTE: the feature correlation matrix has condition number {condition:,.0f}. "
            "Above ~1000 a linear model's coefficients are not individually "
            "interpretable — they trade off against each other almost freely."
        )

    flagged = report.get("leakage_flagged", [])
    if flagged:
        warnings.append(
            f"WARNING: {len(flagged)} feature(s) show IC peaking sharply AT shift 0 "
            f"and falling away on both sides ({', '.join(flagged)}) — the signature of "
            "a value at t that already contains the answer. An honest feature's IC "
            "instead RISES as it is advanced, because advancing lets it see more of "
            "the target window. This is a screen, not a proof: check how each is "
            "computed before trusting or discarding it."
        )

    negative = [
        f
        for f, s in features.items()
        if np.isfinite(s.get("rank_ic_mean", np.nan))
        and abs(s.get("rank_ic_mean", 0.0)) < 0.005
    ]
    if len(negative) == len(features) and features:
        warnings.append(
            "NOTE: no feature reaches |rank IC| of 0.005 on this target. Either the "
            "target horizon is wrong for these features, or there is nothing here to "
            "model — worth resolving before fitting anything."
        )
    return warnings
