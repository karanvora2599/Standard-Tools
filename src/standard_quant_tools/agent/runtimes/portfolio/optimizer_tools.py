"""
`optimize_portfolio`: the exact constrained optimizer as a tool.

The library function (`portfolio.general_optimizer`) takes numbers only --
expected returns, a covariance, benchmark weights, limits -- so this tool
takes no tickers and fetches nothing: the same arguments give the same
answer, inside the determinism layer. The covariance arrives in any of the
three forms the construction tools read (rows with `assets`, the map
estimate_covariance returns, or its `covariance_ref`).

THE INPUT IS FLAT ON PURPOSE. Every limit is a plain map -- group -> min,
factor -> max, asset -> bound -- rather than a list of nested constraint
objects. A nested model is inlined into this tool's schema wherever it is
used, and the `portfolio` runtime's schema budget is shared by twenty
tools; the flat form costs a fraction of the bytes and reads the same.

THE RESULT IS CERTIFIED, NOT TRUSTED. `certified` is True only when the
KKT residuals computed from the weights are all within tolerance, and a
null number carries its reason.
"""

from __future__ import annotations

import math
from typing import Annotated, Dict, List, Literal, Optional

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.models import _distinct
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
)
from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import general_optimizer as lib

from .inline_inputs import (
    COVARIANCE_REF_DESCRIPTION,
    NamedCovariance,
    NamedCovarianceInput,
    resolved_covariance,
)

__all__ = [
    "OPTIMIZER_TOOL_DEFS",
    "OPTIMIZER_TOOL_DISPATCH",
    "OptimizePortfolioInput",
    "OptimizePortfolioResult",
    "optimize_portfolio",
]

Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

_UTILITY = ("max_utility", "max_active_utility")


def _finite_entries(values: Optional[Dict[str, float]], field: str):
    """A {name: number} input with every number finite, or a refusal
    naming the entries that are not."""
    if values is None:
        return values
    bad = [name for name, value in values.items() if not math.isfinite(value)]
    if bad:
        raise ValueError(
            f"{field} is not finite at {bad[:10]}; every entry must be a finite "
            "number."
        )
    return values


class OptimizePortfolioInput(NamedCovarianceInput):
    model_config = ConfigDict(extra="forbid")

    assets: Optional[List[str]] = Field(
        None,
        min_length=2,
        max_length=5000,
        description="An inline covariance's row names; optional with covariance_ref.",
    )
    covariance: Optional[NamedCovariance] = Field(
        None,
        description="Rows parallel to `assets`, or estimate_covariance's `matrix` map.",
    )
    covariance_ref: Optional[str] = Field(None, description=COVARIANCE_REF_DESCRIPTION)
    objective: Literal[
        "min_variance",
        "max_utility",
        "max_sharpe",
        "min_tracking_error",
        "max_active_utility",
    ] = Field(
        "min_variance",
        description="max_utility: mu'w - (risk_aversion/2) w'Sw. "
        "max_active_utility: the same on w - benchmark.",
    )
    expected_returns: Optional[Dict[str, float]] = Field(
        None,
        description="Asset -> expected return, in the covariance's units; "
        "every asset. Needed by the max_ objectives.",
    )
    benchmark_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Asset -> benchmark weight (absent = 0). For the tracking "
        "objectives and cap.",
    )
    risk_aversion: Optional[float] = Field(
        None,
        gt=0,
        le=1e6,
        description="lambda, for the two utility objectives only; 1 if omitted.",
    )
    risk_free_rate: float = Field(
        0.0,
        ge=-10,
        le=10,
        description="Same units as expected_returns; for max_sharpe and sharpe_ratio.",
    )
    min_weight: float = Field(
        0.0, ge=-100, le=100, description="Every asset's lower bound; 0 is long-only."
    )
    max_weight: Optional[float] = Field(
        None, ge=-100, le=100, description="Every asset's upper bound; none by default."
    )
    lower_bounds: Optional[Dict[str, float]] = Field(
        None, description="Asset -> lower bound, overriding min_weight."
    )
    upper_bounds: Optional[Dict[str, float]] = Field(
        None,
        description="Asset -> upper bound, overriding max_weight. Equal "
        "bounds fix a weight.",
    )
    budget: float = Field(
        1.0,
        ge=-100,
        le=100,
        description="Sum of weights: 1 invested, 0 dollar-neutral.",
    )
    budget_type: Literal["equal", "at_most"] = Field(
        "equal", description="'at_most' leaves cash, which earns nothing here."
    )
    groups: Optional[Dict[str, List[str]]] = Field(
        None, max_length=500, description="Group -> member assets (sectors, countries)."
    )
    group_min: Optional[Dict[str, float]] = Field(
        None, description="Group -> least total weight."
    )
    group_max: Optional[Dict[str, float]] = Field(
        None, description="Group -> most total weight."
    )
    factor_loadings: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        max_length=5000,
        description="Asset -> {factor: loading}; every asset, every factor.",
    )
    exposure_min: Optional[Dict[str, float]] = Field(
        None, description="Factor -> least B'w. Equal min and max hold it (0 neutral)."
    )
    exposure_max: Optional[Dict[str, float]] = Field(
        None, description="Factor -> most B'w."
    )
    max_volatility: Optional[float] = Field(
        None, gt=0, le=10, description="Volatility cap, the covariance's units."
    )
    max_tracking_error: Optional[float] = Field(
        None, gt=0, le=10, description="Tracking-error cap against benchmark_weights."
    )

    @field_validator("assets")
    @classmethod
    def _assets_once(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        return v if v is None else _distinct(v, "assets")

    @field_validator(
        "expected_returns",
        "benchmark_weights",
        "group_min",
        "group_max",
        "exposure_min",
        "exposure_max",
    )
    @classmethod
    def _finite_maps(cls, v: Optional[Dict[str, float]], info):
        return _finite_entries(v, info.field_name)

    @field_validator("lower_bounds", "upper_bounds")
    @classmethod
    def _finite_bounds(cls, v: Optional[Dict[str, float]], info):
        # An infinite bound is no bound, which leaving the asset out says.
        return _finite_entries(v, info.field_name)

    @field_validator("factor_loadings")
    @classmethod
    def _finite_loadings(
        cls, v: Optional[Dict[str, Dict[str, float]]]
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if v is None:
            return v
        for asset, row in v.items():
            _finite_entries(row, f"factor_loadings[{asset!r}]")
        return v

    @model_validator(mode="after")
    def _read_by_the_objective(self) -> "OptimizePortfolioInput":
        # A field the chosen objective does not read is refused, not
        # ignored: a caller who set risk_aversion for min_variance believes
        # it shaped the answer.
        if self.risk_aversion is not None and self.objective not in _UTILITY:
            raise ValueError(
                f"risk_aversion is read only by max_utility and "
                f"max_active_utility, not objective={self.objective!r}. Drop "
                "it, or choose a utility objective."
            )
        if (self.group_min or self.group_max) and not self.groups:
            raise ValueError(
                "group_min / group_max name groups, and `groups` (group -> "
                "member assets) is not given."
            )
        if (self.exposure_min or self.exposure_max) and not self.factor_loadings:
            raise ValueError(
                "exposure_min / exposure_max need factor_loadings "
                "(asset -> {factor: loading})."
            )
        return self


# ── results ─────────────────────────────────────────────────────────────


#: Why a number of an answer is null when the solve reached no answer. The
#: library names those fields itself in one line in that case.
_NO_ANSWER = "the exact solve reached no answer, so there are no weights to measure"


class BindingConstraint(ExplainsNulls):
    """One constraint at its bound. It has no warnings of its own: a null
    here is reported by the result that holds the row."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "shadow_price": (
            "the cap equals the lowest value any portfolio inside the other "
            "constraints reaches, so one portfolio meets it and its price is "
            "unbounded"
        ),
    }

    constraint: str = Field(
        ...,
        description="budget, lower_bound, upper_bound, fixed_weight, "
        "group_min/max/equal, exposure_min/max/equal, max_volatility or "
        "max_tracking_error.",
    )
    name: Optional[str] = Field(None, description="The asset, group or factor.")
    bound: Stat = None
    shadow_price: Stat = Field(
        None,
        description="d objective_value / d bound: what moving this bound up by "
        "one unit does to the objective, to first order.",
    )


class OptimizerCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stationarity: Stat = None
    dual_feasibility: Stat = None
    primal_feasibility: Stat = None
    complementary_slackness: Stat = None
    tolerance: float = lib.TOLERANCE


class OptimizePortfolioResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "objective_value": _NO_ANSWER,
        "volatility": _NO_ANSWER,
        "net_weight": _NO_ANSWER,
        "gross_weight": _NO_ANSWER,
        "expected_return": "no expected_returns were given",
        "tracking_error": "no benchmark_weights were given",
        "sharpe_ratio": "no expected_returns were given, or the volatility is zero",
        "effective_risk_aversion": (
            "no single risk aversion gives this answer: a cap on a different "
            "quantity binds, or the cap equals the lowest value attainable"
        ),
    }

    objective: str = ""
    objective_value: Stat = Field(
        None,
        description="The objective at the weights: variance (min_variance), "
        "squared tracking error (min_tracking_error), utility, or the Sharpe "
        "ratio. Shadow prices are in its units.",
    )
    n_assets: int = 0
    assets: List[str] = Field(default_factory=list)
    weights: Dict[str, Stat] = Field(default_factory=dict)
    active_weights: Optional[Dict[str, Stat]] = Field(
        None, description="weights - benchmark_weights; null without a benchmark."
    )
    expected_return: Stat = None
    volatility: Stat = None
    tracking_error: Stat = None
    sharpe_ratio: Stat = Field(
        None, description="(expected_returns - risk_free_rate)'w / volatility."
    )
    net_weight: Stat = None
    gross_weight: Stat = None
    group_totals: Dict[str, Stat] = Field(default_factory=dict)
    exposures: Dict[str, Stat] = Field(
        default_factory=dict, description="Factor -> B'w, every factor given."
    )
    binding_constraints: List[BindingConstraint] = Field(
        default_factory=list,
        description="Every constraint at its bound, largest |shadow_price| first.",
    )
    risk_aversion: Stat = Field(
        None, description="As given; null for an objective that takes none."
    )
    effective_risk_aversion: Stat = Field(
        None,
        description="The risk aversion at which the uncapped problem gives this "
        "answer: risk_aversion, raised by a binding cap on the utility's own "
        "risk. Null for an objective that takes none.",
    )
    risk_free_rate: Stat = None
    condition_number: Stat = None
    certified: bool = Field(
        False,
        description="True only when every KKT residual, computed from the "
        "weights, is within the certificate's tolerance.",
    )
    certificate: Optional[OptimizerCertificate] = None
    solver: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


# ── the tool ────────────────────────────────────────────────────────────


def optimize_portfolio(input_data: OptimizePortfolioInput) -> OptimizePortfolioResult:
    who = "optimize_portfolio"
    if input_data.covariance_ref is not None:
        covariance = resolved_covariance(
            input_data.covariance_ref, input_data.assets, who
        )
    else:
        assets = list(input_data.assets or [])
        matrix = input_data.covariance
        if len(matrix) != len(assets) or any(len(row) != len(assets) for row in matrix):
            raise ValidationError(
                f"{who}: the covariance is not square over the {len(assets)} "
                "assets: give one row per asset and one entry per asset in "
                "each row."
            )
        covariance = pd.DataFrame(matrix, index=assets, columns=assets)
    names = [str(a) for a in covariance.columns]
    lower: Dict[str, float] = {a: float(input_data.min_weight) for a in names}
    upper: Dict[str, float] = {
        a: math.inf if input_data.max_weight is None else float(input_data.max_weight)
        for a in names
    }
    for field_name, target in (("lower_bounds", lower), ("upper_bounds", upper)):
        overrides = getattr(input_data, field_name) or {}
        unknown = [a for a in overrides if a not in target]
        if unknown:
            raise ValidationError(
                f"{who}: {field_name} names {unknown[:8]}, which are not assets "
                "of the covariance."
            )
        target.update({a: float(v) for a, v in overrides.items()})
    result = lib.optimize_portfolio(
        covariance,
        expected_returns=input_data.expected_returns,
        objective=input_data.objective,
        risk_aversion=(
            1.0 if input_data.risk_aversion is None else input_data.risk_aversion
        ),
        risk_free_rate=input_data.risk_free_rate,
        benchmark_weights=input_data.benchmark_weights,
        lower_bounds=lower,
        upper_bounds=upper,
        budget=input_data.budget,
        budget_type=input_data.budget_type,
        groups=input_data.groups,
        group_min=input_data.group_min,
        group_max=input_data.group_max,
        factor_loadings=input_data.factor_loadings,
        exposure_min=input_data.exposure_min,
        exposure_max=input_data.exposure_max,
        max_volatility=input_data.max_volatility,
        max_tracking_error=input_data.max_tracking_error,
    )
    return OptimizePortfolioResult(**result)


OPTIMIZER_TOOL_DEFS = [
    (
        "optimize_portfolio",
        "Exact constrained portfolio optimization on expected returns and a "
        "covariance you supply: min variance, max utility, max Sharpe, min "
        "tracking error or max active utility. Takes per-asset bounds, a "
        "budget (=, <=, 0 for dollar-neutral), group and factor-exposure "
        "limits (min = max for neutrality) and volatility or tracking-error "
        "caps. Returns the weights and their risk, every binding constraint "
        "with its shadow price, and a KKT certificate computed from the "
        "weights: certified is never assumed, and an infeasible set of "
        "limits is refused naming the ones that conflict. No tickers: pass "
        "a covariance or estimate_covariance's covariance_ref.",
        OptimizePortfolioInput,
    ),
]

OPTIMIZER_TOOL_DISPATCH = {
    "optimize_portfolio": (optimize_portfolio, OptimizePortfolioInput),
}
