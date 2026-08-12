"""
Declarative specs the modeling runtime executes — the ModelSpec-not-exec()
contract: an LLM (or any caller) builds one of these Pydantic objects and
hands it to `dataset.builder.build_dataset` / `engine.run_experiment`,
never arbitrary Python. Every field here is validated once, at the
boundary, the same discipline `agent/models.py` uses for the analysis
tool surface.
"""

import math
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from .features.base import RESERVED_PANEL_COLUMNS


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
    alias: Optional[str] = Field(
        None,
        description=(
            "Column name for this feature in the output panel. Defaults to "
            "`id`. Supply one to request the SAME feature at more than one "
            "parameter setting — e.g. market.momentum at lookback 20 and 252 "
            "as 'mom_20' and 'mom_252', a completely standard multi-horizon "
            "model spec that was previously impossible because the panel "
            "keyed one column per feature id."
        ),
    )

    @field_validator("alias")
    @classmethod
    def _alias_is_a_usable_column_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.strip():
            raise ValueError("alias must be a non-empty string")
        # These are reserved by the long panel's own schema; an alias
        # colliding with one would overwrite the column rather than add to it.
        if v in RESERVED_PANEL_COLUMNS:
            raise ValueError(
                f"alias={v!r} is reserved by the panel schema "
                "(date/entity/target/label_end_date)"
            )
        return v

    @property
    def output_name(self) -> str:
        """The panel column this feature produces. `id` when no alias is
        given, so every existing spec keeps its current column name."""
        return self.alias or self.id


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
    # max_length alongside min_length: universe fetching creates a task per
    # symbol, and while a semaphore bounds how many run at once it does not
    # bound how many are created. One valid-looking tool call could
    # therefore request an unbounded workload -- the same
    # agent-triggerable resource-exhaustion path the estimator registry's
    # parameter ceilings close. 1000 is far above any realistic modeling
    # universe and is a budget, not a modeling opinion.
    universe: List[str] = Field(
        ..., min_length=1, max_length=1000, description="Ticker symbols."
    )
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
    provider: Literal["yfinance", "polygon", "bloomberg"] = Field(
        "yfinance",
        description=(
            "Data provider for this dataset. Previously hardcoded to the "
            "DataFactory default, so a model could not be built on anything "
            "else and its lineage never recorded which source it came from. "
            "Credentials are deliberately NOT part of this spec — it is "
            "persisted to disk, hashed into the model's lineage and written "
            "into decision records, so an api_key field here would leak the "
            "key into all three. Providers read their own credentials from "
            "the environment (e.g. SQT_POLYGON_API_KEY)."
        ),
    )
    interval: str = Field(
        "1d",
        min_length=1,
        description=(
            "Bar interval passed to the provider, e.g. '1d' (default), '1h', "
            "'1wk'. The VALUE is validated by the selected provider, which "
            "owns the authoritative list — they differ (BloombergProvider "
            "rejects intraday outright). Note that `target.horizon` and every "
            "feature's lookback count BARS of this interval, and that the "
            "built-in features' default parameters and annualization "
            "constants are calibrated for daily bars: window=252 means one "
            "year at '1d' and about six weeks at '1h'. build_model_dataset "
            "warns when this is not '1d' rather than silently reinterpreting "
            "those defaults."
        ),
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
    def _no_duplicate_output_names(cls, v: List["FeatureSpec"]) -> List["FeatureSpec"]:
        """
        Uniqueness is enforced on the OUTPUT COLUMN, not the feature id.

        Keying on the id meant momentum(20) + momentum(252) — an ordinary
        multi-horizon spec — was rejected outright. What actually cannot
        collide is the panel column name, so that is what is checked; an
        `alias` distinguishes repeated uses of one feature.
        """
        names = [f.output_name for f in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if not dupes:
            return v

        # Distinguish the two causes: a genuine alias collision is a
        # different mistake from repeating a feature without aliasing it.
        repeated_ids = sorted(
            {f.id for f in v if [s.id for s in v].count(f.id) > 1 and f.alias is None}
        )
        if repeated_ids:
            raise ValueError(
                f"features would produce duplicate panel column(s): {dupes}. "
                f"Feature id(s) {repeated_ids} are requested more than once without an "
                "alias — give each use a distinct `alias` (e.g. "
                "FeatureSpec(id='market.momentum', params={'lookback': 20}, "
                "alias='mom_20') alongside alias='mom_252')."
            )
        raise ValueError(
            f"features would produce duplicate panel column(s): {dupes} — two aliases "
            "(or an alias and another feature's id) resolve to the same column name."
        )

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
        42,
        ge=0,
        le=2**32 - 1,
        description="Seed passed to the estimator's constructor. Bounded to "
        "numpy/sklearn's accepted RandomState range [0, 2**32-1]: an arbitrary "
        "Python int outside it (negative, or wider than 32 bits) is rejected "
        "deep inside sklearn rather than at this boundary, where the message "
        "can say which field was wrong.",
    )
