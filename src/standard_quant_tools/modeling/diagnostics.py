"""
Not "is this model good" but "how, and where, is it wrong".

WHY AN AGGREGATE METRIC IS NOT ENOUGH. `score_predictions` reports an R2, an
IC and a baseline, and those answer whether the model beat predicting the
mean. They cannot answer the question anybody actually asks next, which is
whether the model is broadly mediocre or excellent-except-in-the-conditions
you trade. Those two have the same headline number and completely different
consequences, and only a breakdown separates them.

WHAT IS BROKEN DOWN, and why each:

  BY ENTITY. A model carried by three names and useless on the rest is a
  concentration risk disguised as an edge. Reported with the row count
  beside it, because a bias measured on nine observations is not a bias.

  BY PERIOD. A model whose skill lives in one quarter learned that quarter.

  BY PREDICTION DECILE. Where in its OWN range the model is wrong. A model
  accurate in the middle and wrong at the extremes is exactly backwards for
  trading, because the extremes are the positions you take.

  BY FEATURE DECILE. The conditional version: does it fail when the spread
  is wide, when volatility is high, when the book is thin. This is the
  breakdown that turns "the model is mediocre" into "the model fails in
  thin books around the open", which is actionable and the other is not.

CALIBRATION IS A SEPARATE QUESTION FROM ACCURACY. A regression can rank
perfectly and be systematically too confident: predictions spread twice as
wide as the outcomes they predict. Regressing the ACTUAL on the PREDICTED
answers it -- slope 1 and intercept 0 is calibrated, slope below 1 means
the predictions are too spread out. That is invisible in an R2 or an IC and
it changes every position size computed from the prediction.

RESIDUAL AUTOCORRELATION IS EXPECTED HERE, NOT A DEFECT. A 20-bar forward
return sampled every bar overlaps its neighbour in 19 of 20 bars, so
consecutive residuals are correlated by construction. It is reported with
that stated, because the number is genuinely useful for judging how many
INDEPENDENT observations there were and genuinely misleading if read as
model misspecification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

#: A bucket thinner than this is reported with its count and excluded from
#: "worst bucket" claims. A bias measured on a handful of rows is noise
#: wearing a decimal point.
MIN_BUCKET_ROWS = 30

#: How many quantile buckets a numeric breakdown uses.
N_BUCKETS = 10


def _finite(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")


def residual_summary(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    """Shape of the errors, before asking where they fall."""
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 2:
        raise ValidationError(
            "fewer than two rows have both a prediction and an outcome, so "
            "no residual statistic is defined."
        )
    residual = actual[mask] - predicted[mask]
    n = int(residual.size)
    std = float(residual.std(ddof=1)) if n > 1 else float("nan")
    centred = residual - residual.mean()
    skew = kurt = None
    if n > 3 and std > 0:
        skew = float((centred**3).mean() / std**3)
        kurt = float((centred**4).mean() / std**4 - 3.0)
    return {
        "n": n,
        # A non-zero mean residual is BIAS: the model is systematically
        # high or low, which no amount of rank skill corrects.
        "mean_error": float(residual.mean()),
        "mean_absolute_error": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt((residual**2).mean())),
        "std_error": std,
        "skew": skew,
        # Excess kurtosis. A fat residual tail means the model is usually
        # close and occasionally very wrong, which sizing from its average
        # error will not survive.
        "excess_kurtosis": kurt,
        "p05_error": float(np.quantile(residual, 0.05)),
        "p95_error": float(np.quantile(residual, 0.95)),
        "worst_error": float(residual[np.argmax(np.abs(residual))]),
    }


def heteroskedasticity(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """
    Correlation between |error| and the prediction's magnitude.

    Positive means the model is least reliable exactly where it is most
    confident, which is the direction that costs money: the large
    predictions are the ones sized on.
    """
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if mask.sum() < 3:
        return None
    error = np.abs(actual[mask] - predicted[mask])
    magnitude = np.abs(predicted[mask])
    if error.std() == 0 or magnitude.std() == 0:
        return None
    return float(np.corrcoef(error, magnitude)[0, 1])


def residual_autocorrelation(joined: pd.DataFrame) -> Optional[float]:
    """
    Lag-1 residual autocorrelation, computed WITHIN each entity.

    Stacking the panel first would measure the row ordering rather than the
    series: consecutive rows in a long panel are different entities on the
    same date, and their residuals have no reason to relate.
    """
    values: List[float] = []
    for _entity, group in joined.groupby("entity", sort=False):
        residual = group.sort_values("date")["_residual"].to_numpy(dtype="float64")
        residual = residual[np.isfinite(residual)]
        if residual.size < 3 or residual.std() == 0:
            continue
        values.append(float(np.corrcoef(residual[:-1], residual[1:])[0, 1]))
    return float(np.mean(values)) if values else None


def calibration(actual: np.ndarray, predicted: np.ndarray, task: str) -> Dict[str, Any]:
    """
    Whether the prediction's SCALE is right, separately from its ordering.

    For a continuous score, the actual is regressed on the predicted. Slope
    1, intercept 0 is calibrated; slope below 1 means the predictions are
    spread wider than the outcomes, which is over-confidence and is exactly
    what a position size computed from the prediction will over-trade.

    For a probability, the Brier score and expected calibration error say
    whether a stated 0.7 happens seventy percent of the time. A model can
    rank perfectly and still be badly calibrated, and a threshold applied to
    a mis-calibrated probability is applied at the wrong place.
    """
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]
    if a.size < 3:
        return {"n": int(a.size)}

    if task == "classification":
        brier = float(((p - a) ** 2).mean())
        bins = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1]), 0, 9)
        rows = []
        ece = 0.0
        for b in range(10):
            here = bins == b
            if not here.any():
                continue
            share = float(here.mean())
            confidence = float(p[here].mean())
            observed = float(a[here].mean())
            ece += share * abs(confidence - observed)
            rows.append(
                {
                    "bin": b,
                    "n": int(here.sum()),
                    "mean_predicted": confidence,
                    "observed_rate": observed,
                }
            )
        return {
            "n": int(a.size),
            "brier_score": brier,
            # Weighted average gap between stated confidence and observed
            # frequency. Zero is perfect; a threshold on a model with a
            # large ECE fires at the wrong place.
            "expected_calibration_error": float(ece),
            "reliability": rows,
        }

    if p.std() == 0:
        return {
            "n": int(a.size),
            "slope": None,
            "intercept": None,
            "note": "every prediction is identical, so no slope is defined.",
        }
    slope, intercept = np.polyfit(p, a, 1)
    return {
        "n": int(a.size),
        "slope": float(slope),
        "intercept": float(intercept),
        # The prediction's spread against the outcome's. Above 1 means the
        # model is saying more than the data supports.
        "dispersion_ratio": float(p.std() / a.std()) if a.std() > 0 else None,
    }


def _bucket_report(
    joined: pd.DataFrame, key: pd.Series, label: str
) -> List[Dict[str, Any]]:
    """Error statistics per bucket, with the count that qualifies them."""
    frame = joined.assign(_bucket=key)
    rows: List[Dict[str, Any]] = []
    for name, group in frame.groupby("_bucket", sort=True, observed=True):
        residual = group["_residual"].to_numpy(dtype="float64")
        residual = residual[np.isfinite(residual)]
        if residual.size == 0:
            continue
        rows.append(
            {
                label: str(name),
                "n": int(residual.size),
                "mean_error": float(residual.mean()),
                "mean_absolute_error": float(np.abs(residual).mean()),
                "rmse": float(np.sqrt((residual**2).mean())),
                # Below the floor a "bias" is a handful of rows, so the
                # flag travels with the row rather than being inferred.
                "thin": bool(residual.size < MIN_BUCKET_ROWS),
            }
        )
    return rows


def error_attribution(
    joined: pd.DataFrame,
    *,
    feature: Optional[str] = None,
    period: str = "M",
) -> Dict[str, Any]:
    """Where the errors are, by entity, by period, and by magnitude."""
    out: Dict[str, Any] = {
        "by_entity": _bucket_report(joined, joined["entity"], "entity"),
        "by_period": _bucket_report(
            joined,
            pd.to_datetime(joined["date"]).dt.to_period(period).astype(str),
            "period",
        ),
    }
    predicted = _finite(joined["_predicted"])
    if np.isfinite(predicted).sum() >= N_BUCKETS:
        try:
            deciles = pd.qcut(
                pd.Series(predicted, index=joined.index),
                N_BUCKETS,
                labels=False,
                duplicates="drop",
            )
            out["by_prediction_decile"] = _bucket_report(
                joined, deciles.astype("Int64").astype(str), "decile"
            )
        except ValueError:
            out["by_prediction_decile"] = []
    if feature is not None:
        if feature not in joined.columns:
            raise ValidationError(
                f"feature={feature!r} is not a column of this model's dataset "
                "panel, so errors cannot be broken down by it. The panel "
                f"carries: {[c for c in joined.columns if not c.startswith('_')][:15]}"
            )
        values = pd.to_numeric(joined[feature], errors="coerce")
        # `duplicates="drop"` does NOT raise on a feature with few distinct
        # values -- it silently returns fewer buckets, and the result would
        # be labelled "deciles" while being halves. The count is checked
        # rather than the exception, because the exception never comes.
        buckets = pd.qcut(values, N_BUCKETS, labels=False, duplicates="drop")
        produced = int(buckets.nunique(dropna=True))
        if produced < 3:
            raise ValidationError(
                f"{feature!r} splits into {produced} bucket(s), not "
                f"{N_BUCKETS}: it takes too few distinct values "
                f"({int(values.nunique(dropna=True))}) to have deciles. A "
                "flag or a category is not a gradient -- break errors down "
                "by a continuous feature, or group by the flag directly."
            )
        out["by_feature_decile"] = _bucket_report(
            joined, buckets.astype("Int64").astype(str), "decile"
        )
        out["feature"] = feature
        if produced < N_BUCKETS:
            out["feature_note"] = (
                f"NOTE: {feature!r} produced {produced} buckets rather than "
                f"{N_BUCKETS} -- repeated values collapse quantile edges, so "
                "these are not tenths of the sample."
            )
    return out


def worst_buckets(report: Dict[str, Any], *, limit: int = 3) -> List[str]:
    """
    The headline a breakdown exists to produce, as sentences.

    Thin buckets are excluded rather than ranked: the worst bucket of a
    breakdown is almost always the emptiest one, and reporting that as a
    finding is how a diagnostic becomes noise.
    """
    lines: List[str] = []
    for key, label in (
        ("by_entity", "entity"),
        ("by_period", "period"),
        ("by_prediction_decile", "prediction decile"),
        ("by_feature_decile", "feature decile"),
    ):
        rows = [r for r in report.get(key, []) if not r["thin"]]
        if len(rows) < 2:
            continue
        rows.sort(key=lambda r: -r["rmse"])
        worst, best = rows[0], rows[-1]
        if best["rmse"] <= 0:
            continue
        ratio = worst["rmse"] / best["rmse"]
        if ratio < 1.5:
            continue
        name = worst.get(label.split()[0], worst.get("decile", "?"))
        lines.append(
            f"{label} {name}: RMSE {worst['rmse']:.6g} on {worst['n']:,} rows, "
            f"{ratio:.1f}x the best bucket. The model is not uniformly "
            "mediocre -- it is worse here."
        )
    return lines[:limit]


__all__ = [
    "MIN_BUCKET_ROWS",
    "N_BUCKETS",
    "calibration",
    "error_attribution",
    "heteroskedasticity",
    "residual_autocorrelation",
    "residual_summary",
    "worst_buckets",
]
