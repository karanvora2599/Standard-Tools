"""
Is model B actually better than model A, or did it draw a kinder sample?

`compare_models` ranks registered models by a headline out-of-sample
metric. That answers which number is larger and nothing about whether the
difference is larger than the noise in one OOS sample -- and on a few
hundred dates of daily IC, a difference of 0.006 is routinely inside that
noise. Sorting on it selects the model that got the friendlier draw, which
is the quiet way to turn an out-of-sample sample into a tuning set.

THE COMPARISON IS PAIRED, ON THE INTERSECTION. Both models' predictions are
joined on (date, entity) and the same realized outcome, so each date
contributes one IC for A and one for B measured on identical rows. The
object of interest is the per-date DIFFERENCE series: its mean is the
improvement, and its dispersion -- which is far smaller than either
model's own, because the two share every good and bad day -- is what an
interval has to be built on. Comparing two unpaired intervals would find
nothing significant about two models that differ on every single day.

THE INTERVAL IS BLOCK-BOOTSTRAPPED. Daily ICs under an overlapping label
are serially correlated, and an IID resample destroys exactly that,
narrowing the interval by a factor the repository has already measured
(`analysis.inference.bootstrap_statistic`: 2.24x on AR(1) at phi 0.8).
Blocks of n^(1/3) consecutive dates are resampled with replacement, the
rule that function uses, and its block indexer is reused rather than
rewritten.

THE LOSS TEST IS DIEBOLD-MARIANO, where a loss exists. For a regression
the per-date mean squared error differential is tested against a
Newey-West long-run variance at the label horizon, with the
Harvey-Leybourne-Newbold small-sample correction; for a classifier the
Brier differential; for a ranker there is no loss with units, so there is
no DM and the IC difference carries the comparison alone.

MANY CANDIDATES AGAINST ONE REFERENCE need a multiple-testing correction,
and the p-values come back Holm-adjusted. What Holm does not do is control
for the candidates having been SELECTED on this same sample -- that is what
SPA-style tests exist for, and the report says so rather than pretending
the adjustment covers it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.inference import _block_indices
from standard_quant_tools.error import ValidationError

from .metrics import cross_sectional_ic

#: The per-date correlations a comparison can be built on.
COMPARISON_METRICS = ("cs_rank_ic", "cs_ic")

#: The columns a prediction frame must carry to be compared.
REQUIRED_COLUMNS = ("date", "entity", "prediction", "target")


def _moving_block_means(
    values: np.ndarray, n_bootstrap: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap means of `values` under a moving-block resample."""
    n = values.size
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draws[i] = float(values[_block_indices(n, block_size, rng)].mean())
    return draws


def compare_ic_series(
    ic_a: pd.Series,
    ic_b: pd.Series,
    *,
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    The statistics of the per-date difference `ic_b - ic_a`.

    The two series are aligned on their common dates -- which, when they
    came from `paired_comparison`, is every date, because both were
    computed on the same joined rows. Returns the mean difference, a
    block-bootstrap percentile interval on it, a two-sided bootstrap
    p-value (the share of resampled means on the far side of zero, doubled),
    and the fraction of dates on which B beat A.
    """
    if not 0 < confidence < 1:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence!r}")
    joined = pd.concat(
        [ic_a.rename("a"), ic_b.rename("b")], axis=1, join="inner"
    ).dropna()
    n = int(len(joined))
    if n < 10:
        raise ValidationError(
            f"compare_ic_series: only {n} date(s) carry an IC for both models; "
            "a difference measured on fewer than ten dates has no interval "
            "worth reporting."
        )
    diff = (joined["b"] - joined["a"]).to_numpy(dtype=np.float64)
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(int(block_size), n // 2))
    n_bootstrap = max(100, int(n_bootstrap))

    observed = float(diff.mean())
    wins = int(np.sum(diff > 0.0))
    losses = int(np.sum(diff < 0.0))
    rng = np.random.default_rng(int(seed))
    draws = _moving_block_means(diff, n_bootstrap, block_size, rng)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(draws, alpha * 100))
    upper = float(np.percentile(draws, (1 - alpha) * 100))
    # Percentile-bootstrap p-value: how often the resampled mean lands on
    # the far side of zero from the observed one, doubled for two sides
    # and floored at one resample so it is never reported as exactly zero.
    far_side = float(np.mean(draws <= 0.0) if observed > 0 else np.mean(draws >= 0.0))
    p_value = float(min(1.0, 2.0 * max(far_side, 1.0 / n_bootstrap)))

    return {
        "n_dates": n,
        "mean_a": float(joined["a"].mean()),
        "mean_b": float(joined["b"].mean()),
        "mean_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
        "p_value": p_value,
        # Decided dates only: two identical models tie on every date, and
        # a share of ALL dates read that as 'b lost every day'.
        "hit_rate": (float(wins / (wins + losses)) if wins + losses else float("nan")),
        "n_b_better": wins,
        "n_a_better": losses,
        "n_ties": int(n - wins - losses),
        "block_size": int(block_size),
        "n_bootstrap": int(n_bootstrap),
        "verdict": _verdict(observed, lower, upper),
    }


def _verdict(mean: float, lower: float, upper: float) -> str:
    if lower <= 0.0 <= upper:
        return "indistinguishable"
    return "b_better" if mean > 0.0 else "a_better"


def newey_west_variance(values: np.ndarray, lag: int) -> float:
    """
    Long-run variance of the MEAN of `values`, Bartlett kernel to `lag`.

    gamma_0 + 2 * sum_{k=1..lag} (1 - k/(lag+1)) * gamma_k, divided by n.
    `lag=0` is the ordinary variance of the mean. The lag to use for an
    overlapping h-bar label is h-1: two rows h-1 bars apart still share a
    bar, and beyond that they do not.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    centered = x - x.mean()
    variance = float(np.dot(centered, centered) / n)
    for k in range(1, min(int(lag), n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        autocov = float(np.dot(centered[k:], centered[:-k]) / n)
        variance += 2.0 * weight * autocov
    return variance / n


def diebold_mariano(
    loss_a: pd.Series, loss_b: pd.Series, *, lag: int = 0
) -> Dict[str, Any]:
    """
    Diebold-Mariano on the per-date loss differential `loss_a - loss_b`.

    A POSITIVE statistic means B's loss is smaller. The variance is
    Newey-West at `lag`, and the statistic carries the Harvey-Leybourne-
    Newbold correction for the small samples this is used on; the p-value
    is two-sided normal. NaN when the differential has no variance, which
    two identical models produce and which is not evidence of anything.
    """
    from scipy.stats import norm

    joined = pd.concat(
        [loss_a.rename("a"), loss_b.rename("b")], axis=1, join="inner"
    ).dropna()
    n = int(len(joined))
    if n < 10:
        raise ValidationError(
            f"diebold_mariano: only {n} date(s) carry a loss for both models."
        )
    differential = (joined["a"] - joined["b"]).to_numpy(dtype=np.float64)
    mean = float(differential.mean())
    variance = newey_west_variance(differential, int(lag))
    if not math.isfinite(variance) or variance <= 0.0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_differential": mean,
            "lag": int(lag),
            "n_dates": n,
        }
    h = int(lag) + 1
    correction = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    statistic = float(mean / math.sqrt(variance) * correction)
    return {
        "statistic": statistic,
        "p_value": float(2.0 * norm.sf(abs(statistic))),
        "mean_differential": mean,
        "lag": int(lag),
        "n_dates": n,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjustment, monotone and capped at one."""
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return [float(v) for v in adjusted]


def _check_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"paired_comparison: {label} is missing column(s) {missing}; a "
            f"comparable frame carries {list(REQUIRED_COLUMNS)}."
        )
    out = frame[list(REQUIRED_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["entity"] = out["entity"].astype(str)
    return out


def _per_date_loss(joined: pd.DataFrame, side: str, task: str) -> Optional[pd.Series]:
    """Per-date mean loss for one model, or None for a task with no loss
    that has units."""
    if task == "regression":
        loss = (joined["target"] - joined[f"prediction_{side}"]) ** 2
    elif task == "classification":
        loss = (joined["target"] - joined[f"prediction_{side}"]) ** 2  # Brier
    else:
        return None
    return loss.groupby(joined["date"]).mean()


def paired_comparison(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    task: str,
    metric: str = "cs_rank_ic",
    horizon: Optional[int] = None,
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compare two models' predictions on the rows both predicted.

    Each frame carries (date, entity, prediction, target). They are joined
    on (date, entity); the realized outcome must agree on the intersection,
    which refuses the comparison of two models fitted against different
    labels under the same name. The per-date `metric` is computed for each
    on the joined rows, the difference series is bootstrapped, and where
    the task has a loss with units the Diebold-Mariano test is reported
    beside it.
    """
    if metric not in COMPARISON_METRICS:
        raise ValidationError(
            f"paired_comparison: metric={metric!r}; expected one of "
            f"{list(COMPARISON_METRICS)}."
        )
    a = _check_frame(frame_a, "frame_a")
    b = _check_frame(frame_b, "frame_b")
    joined = a.merge(b, on=["date", "entity"], suffixes=("_a", "_b"), how="inner")
    if joined.empty:
        raise ValidationError(
            "paired_comparison: the two models share no (date, entity) row. "
            "Models validated over different windows cannot be compared "
            "paired; check their test spans."
        )
    disagreement = (joined["target_a"] - joined["target_b"]).abs()
    if not np.allclose(
        joined["target_a"], joined["target_b"], atol=1e-12, equal_nan=True
    ):
        raise ValidationError(
            "paired_comparison: the realized outcomes disagree on "
            f"{int((disagreement > 1e-12).sum())} shared row(s). The two models "
            "were fitted against different labels, and a comparison on rows "
            "whose 'truth' differs would be arithmetic on two questions."
        )
    joined = joined.rename(columns={"target_a": "target"}).drop(columns=["target_b"])
    joined = joined.dropna(subset=["prediction_a", "prediction_b", "target"])

    method = "spearman" if metric == "cs_rank_ic" else "pearson"
    dates = joined["date"].to_numpy()
    truth = joined["target"].to_numpy(dtype=np.float64)
    ic_a = cross_sectional_ic(
        truth, joined["prediction_a"].to_numpy(dtype=np.float64), dates, method
    )
    ic_b = cross_sectional_ic(
        truth, joined["prediction_b"].to_numpy(dtype=np.float64), dates, method
    )
    difference = compare_ic_series(
        ic_a,
        ic_b,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence=confidence,
        seed=seed,
    )

    dm: Optional[Dict[str, Any]] = None
    loss_a = _per_date_loss(joined, "a", task)
    loss_b = _per_date_loss(joined, "b", task)
    if loss_a is not None and loss_b is not None:
        lag = max(int(horizon) - 1, 0) if horizon else 0
        dm = diebold_mariano(loss_a, loss_b, lag=lag)
        dm["loss"] = "squared_error" if task == "regression" else "brier"

    warnings: List[str] = []
    if difference["n_dates"] < 60:
        warnings.append(
            f"NOTE: the comparison rests on {difference['n_dates']} dates. An "
            "interval this short is wide by construction, and 'indistinguishable' "
            "on it means 'not enough dates to tell', not 'the same'."
        )
    if difference["verdict"] == "indistinguishable":
        warnings.append(
            f"The {confidence:.0%} interval on the IC difference "
            f"[{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}] "
            "contains zero: this sample does not separate the two models, "
            "whichever headline number is larger."
        )
    return {
        "metric": metric,
        "task": task,
        "n_rows": int(len(joined)),
        "n_entities": int(joined["entity"].nunique()),
        **difference,
        "diebold_mariano": dm,
        "warnings": warnings,
    }


__all__ = [
    "COMPARISON_METRICS",
    "REQUIRED_COLUMNS",
    "compare_ic_series",
    "diebold_mariano",
    "holm_adjust",
    "newey_west_variance",
    "paired_comparison",
]
