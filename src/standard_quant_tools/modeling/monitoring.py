"""
Monitoring a deployed model: has the world it scores moved away from the
world it was fitted on, and is it still right.

THREE QUESTIONS, THREE ANSWERS. Feature drift asks whether the inputs the
model is being handed today look like the inputs it was trained on, per
feature, with the population stability index and the two-sample
Kolmogorov-Smirnov statistic that `analysis/feature_stability.py` already
computes. Prediction drift asks the same of the outputs against the
out-of-sample predictions the validation produced. Realized IC asks, once
the outcomes for a scored date exist, whether the model's cross-sectional
rank IC on that date sits where the validation said it would.

WHAT REGISTRATION KEEPS SO THE QUESTIONS CAN BE ASKED. A model registered
here persists a seeded sample of the raw feature rows it was trained on
and a sample of its out-of-sample predictions, plus a small profile
(quantile edges, missing rate) per feature. The samples are what PSI and
KS are computed against -- the reference window's own values, so the
current window cannot move the edges it is measured by -- and they are
hashed into the manifest like every other artifact.

THRESHOLDS ARE REPORTED WITH THE NUMBERS, NEVER ALONE. A PSI of 0.1 is the
conventional "watch" line and 0.25 the conventional "act" line; they are
conventions, and the report says so beside every status it assigns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .analysis.feature_stability import ks_statistic, population_stability_index

#: Rows kept as the monitoring reference. Enough for a ten-bin PSI on
#: every feature; small enough to sit beside the model.
REFERENCE_ROWS = 5_000
#: Quantile edges kept per feature: deciles.
PROFILE_BINS = 10
#: The conventional PSI lines: below the first, stable; above the second,
#: severe; between, moderate. Reported with every status.
PSI_MODERATE = 0.10
PSI_SEVERE = 0.25
#: The KS line above which a distribution has clearly moved.
KS_FLAG = 0.20

THRESHOLDS = {
    "psi_moderate": PSI_MODERATE,
    "psi_severe": PSI_SEVERE,
    "ks_flag": KS_FLAG,
    "note": (
        "PSI below 0.10 is conventionally stable, above 0.25 conventionally "
        "severe; KS above 0.20 marks a clearly moved distribution. These are "
        "conventions, not calibrated to this model."
    ),
}


def _status(psi: float, ks: float) -> str:
    if not np.isfinite(psi) and not np.isfinite(ks):
        return "unknown"
    if (np.isfinite(psi) and psi >= PSI_SEVERE) or (np.isfinite(ks) and ks >= KS_FLAG):
        return "severe"
    if np.isfinite(psi) and psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def reference_sample(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    rows: int = REFERENCE_ROWS,
    seed: int = 0,
) -> pd.DataFrame:
    """A seeded sample of `columns` from `frame`, at most `rows` of it."""
    subset = frame[list(columns)]
    if len(subset) <= rows:
        return subset.reset_index(drop=True)
    picks = np.random.default_rng(seed).choice(len(subset), size=rows, replace=False)
    return subset.iloc[np.sort(picks)].reset_index(drop=True)


def feature_profile(frame: pd.DataFrame, feature_ids: Sequence[str]) -> Dict[str, Any]:
    """Per-feature quantile edges, missing rate and moments on the training
    panel: the summary a reader can inspect without the reference sample."""
    profile: Dict[str, Any] = {"bins": PROFILE_BINS, "features": {}}
    for feature in feature_ids:
        values = frame[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        entry: Dict[str, Any] = {
            "n": int(values.size),
            "missing_rate": (
                float(1.0 - finite.size / values.size) if values.size else 1.0
            ),
        }
        if finite.size:
            entry["quantile_edges"] = [
                float(v)
                for v in np.quantile(finite, np.linspace(0.0, 1.0, PROFILE_BINS + 1))
            ]
            entry["mean"] = float(finite.mean())
            entry["std"] = float(finite.std())
        profile["features"][str(feature)] = entry
    return profile


def _missing_rate(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.mean(~np.isfinite(values))) if values.size else float("nan")


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """PSI, KS and missing rates per feature, current against reference."""
    rows: List[Dict[str, Any]] = []
    for feature in feature_ids:
        if feature not in reference.columns or feature not in current.columns:
            rows.append(
                {
                    "feature": str(feature),
                    "psi": float("nan"),
                    "ks": float("nan"),
                    "missing_rate_reference": float("nan"),
                    "missing_rate_current": float("nan"),
                    "status": "unknown",
                }
            )
            continue
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur)
        ks = ks_statistic(ref, cur)
        rows.append(
            {
                "feature": str(feature),
                "psi": float(psi),
                "ks": float(ks),
                "missing_rate_reference": _missing_rate(ref),
                "missing_rate_current": _missing_rate(cur),
                "status": _status(psi, ks),
            }
        )
    return rows


def prediction_drift(reference: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
    """The same two statistics on the predictions, plus the moments."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_finite = ref[np.isfinite(ref)]
    cur_finite = cur[np.isfinite(cur)]
    psi = population_stability_index(ref, cur)
    ks = ks_statistic(ref, cur)
    return {
        "psi": float(psi),
        "ks": float(ks),
        "mean_reference": float(ref_finite.mean()) if ref_finite.size else float("nan"),
        "mean_current": float(cur_finite.mean()) if cur_finite.size else float("nan"),
        "std_reference": (
            float(ref_finite.std()) if ref_finite.size > 1 else float("nan")
        ),
        "std_current": float(cur_finite.std()) if cur_finite.size > 1 else float("nan"),
        "n_reference": int(ref_finite.size),
        "n_current": int(cur_finite.size),
        "status": _status(psi, ks),
    }


def realized_ic(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    validation_ic_mean: Optional[float],
    validation_ic_std: Optional[float],
) -> Dict[str, Any]:
    """
    The cross-sectional rank IC of the scored predictions against realized
    outcomes, beside where the validation said it would sit.

    `outcomes` carries `entity` and `realized`, and `date` when the
    predictions span more than one; joined on what both carry.
    """
    if "entity" not in outcomes.columns or "realized" not in outcomes.columns:
        raise ValidationError(
            "outcomes need `entity` and `realized` columns (and `date` when the "
            "predictions span several dates)."
        )
    keys = ["entity"] + (["date"] if "date" in outcomes.columns else [])
    joined = predictions.merge(outcomes, on=keys, how="inner")
    joined = joined[np.isfinite(joined["realized"].to_numpy(dtype=float))]
    if len(joined) < 3:
        raise ValidationError(
            f"realized IC needs at least 3 scored entities with an outcome; "
            f"{len(joined)} matched. Check the entity names agree."
        )
    per_date = []
    for _date, group in joined.groupby("date", sort=True):
        if len(group) < 3:
            continue
        ic = group["prediction"].rank().corr(group["realized"].rank())
        if np.isfinite(ic):
            per_date.append(float(ic))
    if not per_date:
        raise ValidationError(
            "no scored date carries three or more matched outcomes, so no "
            "cross-sectional IC can be computed."
        )
    ic_mean = float(np.mean(per_date))
    z = None
    if validation_ic_mean is not None and validation_ic_std:
        z = float((ic_mean - validation_ic_mean) / validation_ic_std)
    return {
        "realized_ic": ic_mean,
        "n_dates": len(per_date),
        "n_matched": int(len(joined)),
        "validation_ic_mean": validation_ic_mean,
        "validation_ic_std": validation_ic_std,
        # How many validation-fold standard deviations the realized IC sits
        # from the validation mean; one date is one draw, so a negative
        # value inside two is noise and outside it is worth a look.
        "z_versus_validation": z,
        "status": (
            "unknown"
            if z is None
            else ("severe" if z <= -2.0 else "moderate" if z <= -1.0 else "stable")
        ),
    }


__all__ = [
    "KS_FLAG",
    "PROFILE_BINS",
    "PSI_MODERATE",
    "PSI_SEVERE",
    "REFERENCE_ROWS",
    "THRESHOLDS",
    "drift_report",
    "feature_profile",
    "prediction_drift",
    "realized_ic",
    "reference_sample",
]
