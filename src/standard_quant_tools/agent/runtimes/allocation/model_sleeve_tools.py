"""
Registered models as sleeves, in one call: `build_model_sleeves`.

`modeling.model_sleeves.build_model_sleeves` as a tool. Each model is
evaluated as `evaluate_model_portfolio` evaluates it, its stream is read
from the capital the account opened with, the streams meet on the dates
they share, and the panel is published as a `returns_panel` the other
sleeve tools take as `returns_ref`. The library module's docstring has the
rules: why the idle bars before a model's first trade are dropped, what the
trial count counts and what it cannot see, and what is refused.

THIS TOOL FETCHES, as `evaluate_model_portfolio` does: a registered model
prices its book through its dataset's provider, over its dataset's window.
It names no ticker -- the universe is the model's -- so it is advertised as
that tool is, and the same model, settings and prices give the same panel
under the same reference.

THE SPECS ARE THE MODELING ONES, not copies: `transform` is a
`PredictionTransformSpec` and `portfolio` a `PortfolioSimSpec`, with
`evaluate_model_portfolio`'s defaults. A model's own `transform` is the same
spec; its advertised schema points at the shared field instead of restating
it, because the runtime's listing pays for every nested field once a use.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls
from standard_quant_tools.agent.runtimes.streams import StreamAlignment
from standard_quant_tools.backtest.financing import _compact_schema
from standard_quant_tools.modeling.specs import (
    PortfolioSimSpec,
    PredictionTransformSpec,
)

from .sleeve_models import Stat
from .sleeve_tools import MIN_SHARED_DATES

#: The tool every refusal here names.
_TOOL = "build_model_sleeves"

#: One model's transform: validated as the full spec, advertised as a pointer
#: to the shared `transform` field, whose schema is already in the listing.
ModelTransform = Annotated[
    Optional[PredictionTransformSpec],
    WithJsonSchema(
        {
            "type": "object",
            "description": "This model's transform in place of the shared one: "
            "the fields of transform.",
        }
    ),
]


class ModelSleeve(BaseModel):
    """One registered model, as a sleeve of the panel."""

    model_config = ConfigDict(
        extra="forbid", protected_namespaces=(), json_schema_extra=_compact_schema
    )

    model_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="A model_id from run_model_experiment.",
    )
    name: Optional[str] = Field(
        None, min_length=1, max_length=64, description="Its column; default model_id."
    )
    transform: ModelTransform = None


class BuildModelSleevesInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_schema)

    models: List[ModelSleeve] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="The models, one sleeve each: model_id, and optionally "
        "name and transform.",
    )
    transform: PredictionTransformSpec = Field(
        default_factory=PredictionTransformSpec,
        description="Shared: how predictions become weights; "
        "evaluate_model_portfolio's default.",
    )
    portfolio: PortfolioSimSpec = Field(
        default_factory=PortfolioSimSpec,
        description="Shared simulation terms: capital, costs, fills, leverage.",
    )
    min_overlap: int = Field(
        MIN_SHARED_DATES,
        ge=1,
        le=1_000_000,
        description="Fewest dates every stream must share.",
    )
    keep_idle: bool = Field(
        False,
        description="True keeps each account's cash bars before its first trade.",
    )
    run_id: str = Field(..., description="Where the aligned returns_panel goes.")

    @field_validator("models", mode="before")
    @classmethod
    def _how_many(cls, v: Any) -> Any:
        # Before the list's own bounds, so the refusal says why two and why
        # twenty rather than only that the list is too short or too long.
        if isinstance(v, (list, tuple)):
            from standard_quant_tools.modeling.model_sleeves import check_model_count

            check_model_count(len(v), _TOOL)
        return v

    @model_validator(mode="after")
    def _nothing_twice(self) -> "BuildModelSleevesInput":
        from standard_quant_tools.modeling.model_sleeves import (
            SleeveRequest,
            check_sleeve_requests,
        )

        check_sleeve_requests(
            [SleeveRequest(model_id=m.model_id, name=m.name) for m in self.models],
            _TOOL,
        )
        return self


# ── results ─────────────────────────────────────────────────────────────

#: Why an evaluation's number can be null, as `evaluate_model_portfolio`'s
#: own metrics say it.
_UNDEFINED = "the evaluation's own figure is undefined (see its warnings)"


class ModelSleeveSummary(ExplainsNulls):
    """One sleeve: its model, the evaluation's numbers, and its stream."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    null_reasons = {"*": _UNDEFINED}

    model_id: str
    estimator_type: str = ""
    dataset_id: str = ""
    task: str = ""
    target_horizon: Optional[int] = Field(
        None, description="Bars ahead its label measured; null if not recorded."
    )
    sharpe: Stat = Field(
        None, description="The evaluation's out-of-sample Sharpe, after costs."
    )
    annual_return: Stat = Field(None, description="The evaluation's CAGR.")
    volatility: Stat = Field(None, description="Annualized.")
    max_drawdown: Stat = Field(None, description="Negative: -0.2 is 20% off a peak.")
    total_return: Stat = Field(
        None,
        description="The evaluation's cumulative return: its stream compounds "
        "to it (times 1 + idle_return when the idle bars are dropped).",
    )
    annual_turnover: Stat = Field(None, description="The evaluation's annualized.")
    n_rebalances: int = 0
    first_rebalance_date: str = ""
    n_idle_returns: int = Field(
        0, description="Bars before its first trade: the account in cash."
    )
    idle_return: Stat = Field(
        None, description="Those bars compounded: the cash rate's, 0 at a zero rate."
    )
    n_returns: int = Field(0, description="Returns in its stream before alignment.")
    equity_curve_ref: str = ""
    target_weights_ref: str = ""


class ModelSleevesResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "correlation": "a stream has no dispersion over the shared dates",
        "*": _UNDEFINED,
    }

    returns_ref: str = Field(
        ..., description="The aligned returns_panel: one column per sleeve."
    )
    columns: List[str] = Field(default_factory=list)
    n_dates: int = Field(0, description="Dates every stream has: the panel's rows.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    periods_per_year: int = Field(
        252,
        description="The bars a year every evaluation annualized with; pass it to "
        "the allocation tools.",
    )
    keep_idle: bool = False
    alignment: StreamAlignment = Field(
        default_factory=StreamAlignment,
        description="What each stream supplied and what the common calendar "
        "cost it: before the window every stream covers, after it, and "
        "inside it on a date another stream lacks.",
    )
    sleeves: Dict[str, ModelSleeveSummary] = Field(default_factory=dict)
    correlation: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict, description="Of the aligned streams, sleeve by sleeve."
    )
    trials_by_dataset: Dict[str, int] = Field(
        default_factory=dict,
        description="Models registered on each dataset these models were trained on.",
    )
    n_trials_lower_bound: int = Field(
        2,
        description="Pass as evaluate_allocation_rule's n_trials, at least: every "
        "model registered on these models' datasets, never fewer than the models "
        "passed. A LOWER BOUND on the configurations behind this panel -- a "
        "transform tried or a model deleted leaves no trace in the registry -- "
        "and allocation rules tried on the panel are trials on top of it. Each "
        "model's own hyperparameter search ran inside its walk-forward folds, so "
        "its out-of-sample record already pays for it and it is not counted "
        "again.",
    )
    warnings: List[str] = Field(default_factory=list)


def build_model_sleeves(input_data: BuildModelSleevesInput) -> ModelSleevesResult:
    from standard_quant_tools.modeling import model_sleeves as model_sleeves_lib

    out = model_sleeves_lib.build_model_sleeves(
        input_data.models,
        run_id=input_data.run_id,
        transform=input_data.transform,
        portfolio=input_data.portfolio,
        min_overlap=input_data.min_overlap,
        keep_idle=input_data.keep_idle,
        what=_TOOL,
    )
    report = out["alignment"]
    return ModelSleevesResult(
        returns_ref=out["returns_ref"],
        columns=out["columns"],
        n_dates=report["n_dates"],
        first_date=report["first_date"],
        last_date=report["last_date"],
        periods_per_year=out["periods_per_year"],
        keep_idle=out["keep_idle"],
        alignment=StreamAlignment(**report),
        sleeves={
            name: ModelSleeveSummary(**row) for name, row in out["sleeves"].items()
        },
        correlation=out["correlation"],
        trials_by_dataset=out["trials_by_dataset"],
        n_trials_lower_bound=out["n_trials_lower_bound"],
        warnings=out["warnings"],
    )


TOOL_DEFS = [
    (
        "build_model_sleeves",
        "Evaluates two to twenty registered models as portfolios and publishes "
        "their out-of-sample return streams on one calendar as a returns_panel, "
        "with the registry's trial count. Each runs through the evaluation "
        "evaluate_model_portfolio makes (a shared transform and portfolio, or "
        "the model's own transform); its stream is read from the opening "
        "capital, the cash bars before its first trade dropped unless "
        "keep_idle. Reports the alignment, each sleeve's Sharpe, return, "
        "volatility, drawdown, turnover, rebalances, estimator, dataset and "
        "equity_curve_ref, the streams' correlations, and n_trials_lower_bound "
        "for evaluate_allocation_rule. Refused by name: fewer than two models, "
        "a model or sleeve named twice, a CPCV model, different bar intervals, "
        "too little overlap.",
        BuildModelSleevesInput,
    ),
]

TOOL_DISPATCH = {
    "build_model_sleeves": (build_model_sleeves, BuildModelSleevesInput),
}

__all__ = [
    "BuildModelSleevesInput",
    "ModelSleeve",
    "ModelSleeveSummary",
    "ModelSleevesResult",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "build_model_sleeves",
]
