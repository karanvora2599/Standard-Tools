"""
The input and result for reading a survival model's curve rather than
its risk.

WHY THESE LIVE BESIDE `survival_tools.py`. The module pair is how a tool
that needs none of `tools.py`'s private helpers stays out of that file's
seam (`dataset_tools.py` and `portfolio_models.py` are the precedents).
Everything a curve needs is `scoring.survival_curves` and the loaded
estimator, so nothing here reaches into the shared tool module.

WHY THE INPUT FIELDS ARE COPIED FROM `ScoreModelInput` RATHER THAN
IMPORTED. The two inputs identify the same thing -- a registered model,
a date, a universe -- and run the same gates, but they are asking for
different objects and will not stay identical: this one carries a time
grid, a matrix switch and a cell budget, none of which a point score has
any use for. A shared base class would make every future field on either
side a decision about both.

See the CHANGELOG entry of 2026-09-21 for why the curve became reachable.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..specs import _parse_date
from .models import _NO_PROTECTED_NAMESPACES, Stat

# ── predict_survival_curve ─────────────────────────────────────────────


class PredictSurvivalCurveInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str = Field(..., description="An id returned by run_model_experiment.")
    as_of: str = Field(..., description="Date YYYY-MM-DD to score as of.")
    universe: List[str] = Field(..., min_length=1)
    lookback_days: int = Field(
        400,
        gt=0,
        description="Calendar days of history fetched before as_of — widen for models "
        "using features with unusually large lookback windows.",
    )
    max_staleness_days: Optional[int] = Field(
        None,
        gt=0,
        description="Reject the call if the newest available observation "
        "(effective_score_date) is more than this many calendar days before "
        "as_of. Enforcing a single cross-section date makes every returned "
        "prediction internally consistent, but says nothing about how OLD that "
        "shared date is — a universe whose data stopped six months ago still "
        "produces a perfectly uniform, entirely stale cross-section. Set this "
        "to state how far behind as_of a prediction is still decision-useful. "
        "None (default) does not check; the gap is reported either way, "
        "so it is never invisible.",
    )
    universe_policy: Literal["strict", "allow"] = Field(
        "strict",
        description="What to do when the model standardizes within the scoring "
        "cross-section (a cross_sectional preprocessing step) and `universe` is "
        "not the training universe. 'strict' (default) refuses: every row's score "
        "depends on which other entities are in the call, so a subset is a "
        "different transform, not a smaller sample. 'allow' scores anyway and "
        "returns a warning saying the transform was refit on this cross-section "
        "and how its width compares with the training one. A model with "
        "universe-scope features is refused either way.",
    )
    times: Optional[List[float]] = Field(
        None,
        description="The exact durations to read every curve at, strictly "
        "increasing and non-negative, in the units the model's target counted "
        "(seconds, bars, days — whatever the training label measured). This is "
        "where a deadline goes: ask for [10, 30, 60] and the answer is the "
        "probability of still resting at ten, thirty and sixty. Left unset, "
        "the grid is n_times quantiles of the model's own baseline event "
        "times, which covers the range the training data actually observed.",
    )
    n_times: int = Field(
        32,
        ge=2,
        le=256,
        description="How many points the DEFAULT grid has, ignored when `times` "
        "is given. The points are quantiles of the baseline event times, so "
        "raising this buys resolution on median_survival (which can only ever "
        "be one of the grid points) rather than reach.",
    )
    include_matrix: bool = Field(
        False,
        description="Return the full per-entity curve (survival_at_times) as "
        "well as the median and the risk. Off by default because the matrix is "
        "n_entities x n_times numbers and the median is usually the decision; "
        "turn it on to plot a curve or to read a specific deadline off it.",
    )
    max_matrix_cells: int = Field(
        50_000,
        ge=1,
        description="Refuse rather than return a matrix larger than this many "
        "cells (n_entities x n_times). Only checked when include_matrix is "
        "true. A 500-name universe on a 256-point grid is 128,000 numbers that "
        "no reader reads; the refusal names the product and the ways down.",
    )

    @field_validator("as_of")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        _parse_date(v, "as_of")
        return v

    @field_validator("universe")
    @classmethod
    def _no_duplicate_symbols(cls, v: List[str]) -> List[str]:
        dupes = sorted({s for s in v if v.count(s) > 1})
        if dupes:
            raise ValueError(f"universe contains duplicate symbols: {dupes}")
        return v

    @field_validator("times")
    @classmethod
    def _increasing_and_non_negative(
        cls, v: Optional[List[float]]
    ) -> Optional[List[float]]:
        # Refused at the schema rather than inside the estimator, because
        # S(t) is a non-increasing step function of t: a grid out of order
        # returns columns that appear to RISE, which reads as a model
        # claiming survival improves with time rather than as a caller
        # who typed a list backwards.
        if v is None:
            return v
        if len(v) < 2:
            raise ValueError("times needs at least two points to be a curve")
        if any(t < 0 for t in v):
            raise ValueError(
                "times must be non-negative durations; a negative time has no "
                "survival probability"
            )
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError(
                "times must be strictly increasing — S(t) only ever falls, so "
                "an out-of-order grid would return a curve that appears to rise"
            )
        return v


class EntitySurvival(BaseModel):
    """One name's answer: where it ranks, and how long it is likely to wait."""

    entity: str
    risk: Stat = Field(
        ...,
        description="The model's risk score for this entity — higher means the "
        "event sooner. Identical to what score_model returns for the same "
        "call, and it is an ORDERING with no units, not a probability.",
    )
    median_survival: Stat = Field(
        ...,
        description="The first grid time at which this entity's survival "
        "probability is 0.5 or below. None when the curve never crosses inside "
        "the grid, which means the grid did not reach far enough — never the "
        "grid's last point, because 'we stopped looking here' and 'it happened "
        "here' are opposite claims.",
    )
    survival_at_times: Optional[List[Stat]] = Field(
        None,
        description="S(t) at each entry of `times`, in the same order. Present "
        "only when include_matrix was requested.",
    )


class PredictSurvivalCurveResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    as_of: str
    effective_score_date: str = Field(
        "",
        description="The single observation date every curve was actually "
        "computed from. Distinct from as_of, which is only the date REQUESTED: "
        "the most recent bar available at or before as_of can be earlier (a "
        "market holiday, a provider whose window excluded as_of).",
    )
    times: List[float] = Field(
        default_factory=list,
        description="The durations every curve was read at, ascending. Either "
        "the grid that was asked for, or quantiles of the model's baseline "
        "event times.",
    )
    per_entity: List[EntitySurvival] = Field(
        default_factory=list,
        description="One row per entity that had a scoreable observation on "
        "effective_score_date.",
    )
    survival_mean_curve: List[Stat] = Field(
        default_factory=list,
        description="The average curve across the entities in THIS call — a "
        "description of this universe on this date, not of the model's "
        "baseline. It moves when the universe does.",
    )
    n_baseline_knots: int = Field(
        0,
        description="Distinct event times in the fitted baseline hazard the "
        "curves are scaled from. Zero for an estimator that carries a fitted "
        "distribution instead of an empirical baseline, which the warnings say.",
    )
    n_entities: int = 0
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Requested entities with no scoreable row at all — a "
        "different condition, with a different fix, from a stale one.",
    )
    stale_entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Requested entities whose most recent row predates "
        "effective_score_date, mapped to that older date. Excluded from the "
        "curves so every returned probability shares one observation date.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Conditions that change how these curves should be read: a "
        "grid that stops before the last baseline knot (so a None median means "
        "'not by then', not 'never') or runs past it (a flat extrapolation), "
        "the proportional-hazards caveat that the ordering is the model's "
        "claim while the level is the baseline's, and every staleness or "
        "cross-section warning score_model raises on the same call.",
    )


__all__ = [
    "EntitySurvival",
    "PredictSurvivalCurveInput",
    "PredictSurvivalCurveResult",
]
