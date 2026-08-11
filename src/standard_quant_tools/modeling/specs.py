"""
Declarative specs the modeling runtime executes — the ModelSpec-not-exec()
contract: an LLM (or any caller) builds one of these Pydantic objects and
hands it to `dataset.builder.build_dataset` / `engine.run_experiment`,
never arbitrary Python. Every field here is validated once, at the
boundary, the same discipline `agent/models.py` uses for the analysis
tool surface.
"""

import math
from typing import Dict, List, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


def _parse_date(value: str, field_name: str) -> pd.Timestamp:
    """Shared by DatasetSpec's start/end cross-check and
    modeling.agent.models.ScoreModelInput.as_of — raises the same
    ValueError shape pydantic validators elsewhere in this codebase use
    (e.g. PortfolioInput._check_weights), not a raw pandas parse error."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name}={value!r} is not a valid date: {exc}") from None


class FeatureSpec(BaseModel):
    """One requested feature: a `features.registry.FEATURE_REGISTRY` id
    plus caller-supplied overrides for that feature's `default_params`."""

    id: str = Field(..., description="Feature id, e.g. 'technical.rsi'.")
    params: Dict[str, object] = Field(
        default_factory=dict,
        description="Overrides merged onto the feature's default_params.",
    )


class TargetSpec(BaseModel):
    type: Literal["forward_return", "forward_direction"] = Field(
        "forward_return",
        description=(
            "'forward_return' (default) — continuous forward return, for "
            "task='regression'. 'forward_direction' — 1.0 when that forward "
            "return exceeds `threshold`, else 0.0, for task='classification'."
        ),
    )
    horizon: int = Field(
        ..., gt=0, description="Bars ahead the target return is measured over."
    )
    threshold: float = Field(
        0.0,
        description=(
            "forward_direction only: the forward return a bar must EXCEED to "
            "be labelled 1.0. Default 0.0 = plain up/down. A positive value "
            "(e.g. 0.02) asks for a move of at least that size, which also "
            "makes the classes deliberately imbalanced — check the resulting "
            "class balance before reading accuracy."
        ),
    )

    @model_validator(mode="after")
    def _threshold_only_for_direction(self) -> "TargetSpec":
        if self.type == "forward_return" and self.threshold != 0.0:
            raise ValueError(
                "threshold applies to type='forward_direction' only; "
                "'forward_return' is the raw continuous return."
            )
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")
        return self


class DatasetSpec(BaseModel):
    universe: List[str] = Field(..., min_length=1, description="Ticker symbols.")
    start: str = Field(..., description="Start date YYYY-MM-DD.")
    end: str = Field(..., description="End date YYYY-MM-DD.")
    features: List[FeatureSpec] = Field(..., min_length=1)
    target: TargetSpec
    benchmark: str = Field(
        "SPY",
        min_length=1,
        description="Benchmark symbol — only consumed by features that need one "
        "(e.g. risk.rolling_beta).",
    )

    @field_validator("universe")
    @classmethod
    def _no_duplicate_symbols(cls, v: List[str]) -> List[str]:
        dupes = sorted({s for s in v if v.count(s) > 1})
        if dupes:
            raise ValueError(f"universe contains duplicate symbols: {dupes}")
        return v

    @field_validator("features")
    @classmethod
    def _no_duplicate_feature_ids(cls, v: List["FeatureSpec"]) -> List["FeatureSpec"]:
        ids = [f.id for f in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(
                f"features contains duplicate ids: {dupes} — each feature id may be "
                "requested at most once per dataset (the same underlying feature at two "
                "different parameter settings isn't supported yet, since the output panel "
                "keys one column per feature id)"
            )
        return v

    @model_validator(mode="after")
    def _start_before_end(self) -> "DatasetSpec":
        start_ts = _parse_date(self.start, "start")
        end_ts = _parse_date(self.end, "end")
        if start_ts >= end_ts:
            raise ValueError(
                f"start ({self.start!r}) must be before end ({self.end!r})"
            )
        return self


class EstimatorSpec(BaseModel):
    type: str = Field(
        ...,
        description="Estimator name — must be in estimators.registry.ESTIMATOR_REGISTRY.",
    )
    params: Dict[str, object] = Field(default_factory=dict)


class ValidationSpec(BaseModel):
    method: Literal["walk_forward"] = "walk_forward"
    train_window: int = Field(..., gt=0, description="Bars per training fold.")
    test_window: int = Field(..., gt=0, description="Bars per test fold.")
    embargo: int = Field(
        0,
        ge=0,
        description="Bars excluded between train and test folds to prevent "
        "lookback leakage across the boundary. Note this does NOT need to "
        "cover the target horizon: training rows whose forward-return label "
        "would resolve inside the test window are purged separately, using "
        "each row's own label end date.",
    )
    min_folds: int = Field(
        2,
        ge=1,
        description="Minimum walk-forward folds that must actually COMPLETE "
        "before a model is registered. One surviving fold is a single "
        "train/test split, not walk-forward validation — it cannot show "
        "whether performance holds across time, which is the entire reason "
        "for validating this way. Lower to 1 only for a deliberately short "
        "exploratory run.",
    )


class ModelSpec(BaseModel):
    task: Literal["regression", "classification"]
    estimator: EstimatorSpec
    validation: ValidationSpec
    random_seed: int = Field(
        42, description="Seed passed to the estimator's constructor."
    )
