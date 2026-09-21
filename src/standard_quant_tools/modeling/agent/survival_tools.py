"""
The survival curve, which a survival model computed and never returned.

A survival model trains end to end here -- concordance, integrated Brier
-- and until now the only number that came out of a registered one was
`predict`: a risk score that ORDERS the cross-section. That answers "who
fills first". It cannot answer "how likely is THIS order to still be
resting in thirty seconds", because a risk has no units and no time
axis. The curve S(t | x) does, it is already what the integrated Brier
score was computed from inside the fold loop, and the fitted baseline
that produces it is persisted with every registered Cox-family model.
This tool reads it back: the same gates as `score_model`, the same
feature matrix, and then `predict_survival_function` instead of
`predict`.

Scoring CALLER-SUPPLIED survival matrices -- handing this library an
(n_rows x n_times) probability matrix of someone else's and asking for
an IPCW Brier score -- is deliberately not built: those four functions
are live inside `validation.survival.survival_metrics` on every survival
fit, and nothing outside this tool can produce a matrix in the shape they
take, so the entry point would have no caller.

See the CHANGELOG entry of 2026-09-21.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from standard_quant_tools.error import ValidationError

from ..scoring import survival_curves as _survival_curves
from .survival_models import (
    EntitySurvival,
    PredictSurvivalCurveInput,
    PredictSurvivalCurveResult,
)

logger = logging.getLogger(__name__)

#: The tool description, here so the module that registers the tool does
#: not have to restate what this door is for.
PREDICT_SURVIVAL_CURVE_DESCRIPTION = (
    "Read a registered SURVIVAL model's curve S(t | x) for each entity as "
    "of a date: how likely each one is to still be waiting at t, plus the "
    "median time until its event. This is the question score_model cannot "
    "answer. score_model returns the survival model's risk score, which "
    "ranks the cross-section -- who fills first -- and has no units; this "
    "returns a probability at each time on a grid, which is what a "
    "deadline is read against and what a desk sizes on. Pass `times` to "
    "ask about specific horizons, or leave it unset for a grid of "
    "quantiles of the model's own baseline event times. median_survival "
    "is the first grid time where the curve falls to 0.5 or below, and is "
    "null when it never crosses inside the grid -- never the grid's last "
    "point. REFUSES: a model whose task is not survival (nothing but a "
    "duration-and-event label estimates a baseline hazard); an estimator "
    "that produces a risk and no survival function, with no fallback to "
    "the risk, because a ranking read as a probability is a number on the "
    "wrong scale; a `times` grid that is not strictly increasing and "
    "non-negative; a matrix over max_matrix_cells when include_matrix is "
    "set; and every gate score_model enforces (training-information "
    "cutoff, feature-implementation drift, universe pins, staleness). "
    "READ THE LEVEL CAREFULLY: under proportional hazards the ORDERING "
    "between these curves is what the model learned, while the LEVEL of "
    "any one of them belongs to the baseline hazard estimated on the "
    "training durations -- if the base rate has moved since training the "
    "ranking can still be right while every probability is off."
)


def predict_survival_curve(
    input_data: PredictSurvivalCurveInput,
) -> PredictSurvivalCurveResult:
    """Survival probabilities over time for each entity, from a registered
    survival model."""
    result: Dict[str, Any] = _survival_curves(
        input_data.model_id,
        input_data.as_of,
        input_data.universe,
        lookback_days=input_data.lookback_days,
        max_staleness_days=input_data.max_staleness_days,
        universe_policy=input_data.universe_policy,
        times=input_data.times,
        n_times=input_data.n_times,
    )

    rows: List[Dict[str, Any]] = result["per_entity"]
    n_entities = len(rows)
    n_times = len(result["times"])
    cells = n_entities * n_times
    # The budget is on what CROSSES THE WIRE, so it is only checked when
    # the matrix was asked for. The curves are computed either way -- the
    # median is read off them -- and refusing the whole call because a
    # universe is wide would take away the medians too.
    if input_data.include_matrix and cells > input_data.max_matrix_cells:
        raise ValidationError(
            f"predict_survival_curve: include_matrix would return "
            f"{n_entities} entities x {n_times} times = {cells} survival "
            f"probabilities, over max_matrix_cells={input_data.max_matrix_cells}. "
            "Lower n_times (the median only ever lands on a grid point, so a "
            "coarser grid costs resolution and not reach), pass `times` with "
            "just the horizons the decision turns on, score a narrower "
            "universe, or raise max_matrix_cells if you really want the whole "
            "matrix. Without include_matrix the same call returns each "
            "entity's risk and median_survival at any width."
        )

    per_entity = [
        EntitySurvival(
            entity=row["entity"],
            risk=row["risk"],
            median_survival=row["median_survival"],
            survival_at_times=(
                row["survival_at_times"] if input_data.include_matrix else None
            ),
        )
        for row in rows
    ]

    logger.debug(
        "[predict_survival_curve] model=%s  as_of=%s  entities=%d  times=%d  "
        "knots=%d",
        input_data.model_id,
        input_data.as_of,
        n_entities,
        n_times,
        result["n_baseline_knots"],
    )
    return PredictSurvivalCurveResult(
        model_id=result["model_id"],
        as_of=result["as_of"],
        effective_score_date=result["effective_score_date"],
        times=result["times"],
        per_entity=per_entity,
        survival_mean_curve=result["survival_mean_curve"],
        n_baseline_knots=result["n_baseline_knots"],
        n_entities=result["n_entities"],
        missing_entities=result["missing_entities"],
        stale_entities=result["stale_entities"],
        warnings=result["warnings"],
    )


__all__ = [
    "PREDICT_SURVIVAL_CURVE_DESCRIPTION",
    "EntitySurvival",
    "PredictSurvivalCurveInput",
    "PredictSurvivalCurveResult",
    "predict_survival_curve",
]
