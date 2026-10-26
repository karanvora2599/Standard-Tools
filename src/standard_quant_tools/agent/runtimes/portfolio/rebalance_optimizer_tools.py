"""
`optimize_rebalance`: the cost-aware trade from the weights held, as a tool.

`optimize_portfolio` returns the portfolio to hold; asked again after the
forecasts moved a little, it returns a portfolio that moved a little in
every name, and a book that follows it trades everything every time. This
tool answers what a rebalance asks -- what to trade from here -- with the
trading costs in the objective: a cost per unit traded and one on the
square of the trade, and caps on turnover, gross weight and volatility.

It takes optimize_portfolio's inputs in optimize_portfolio's shape (the
covariance inline or as `covariance_ref`, the bounds as one number with
per-asset overrides, groups and exposures as flat maps) plus the weights
held and the costs. Numbers only, no tickers: the same arguments give the
same answer, inside the determinism layer.

THE RESULT SAYS WHAT THE COSTS HELD BACK. Beside the trade it reports the
frictionless target -- the same problem with no costs and no turnover cap,
which is optimize_portfolio's answer -- the turnover that target would
take, what trading to it would cost, and how far the cost-aware book stays
from it. A name inside its no-trade band is reported in `no_trade`, its
weight exactly the weight held.

The library function is `portfolio.rebalance_optimizer.optimize_rebalance`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, field_validator, model_validator

from standard_quant_tools.agent.models import _distinct
from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import rebalance_optimizer as lib

from .inline_inputs import (
    COVARIANCE_REF_DESCRIPTION,
    NamedCovariance,
    NamedCovarianceInput,
    resolved_covariance,
)
from .optimizer_tools import OptimizerCertificate, Stat, _finite_entries

__all__ = [
    "OptimizeRebalanceInput",
    "OptimizeRebalanceResult",
    "REBALANCE_TOOL_DEFS",
    "REBALANCE_TOOL_DISPATCH",
    "optimize_rebalance",
]

#: A cost per unit of weight traded above this (100%) is not a cost.
_MAX_LINEAR_COST = 1.0

#: The quadratic cost's ceiling: eta = 100 makes a trade of 0.01 of the
#: book cost 1% of it, already past any market's impact.
_MAX_QUADRATIC_COST = 100.0

CostInput = Union[float, Dict[str, float]]


def _cost(value: CostInput, field: str, most: float) -> CostInput:
    """A cost is one number or asset -> number, each finite, >= 0 and at
    most `most`."""
    entries = value if isinstance(value, dict) else {"": value}
    _finite_entries(entries, field)
    bad = {k: v for k, v in entries.items() if not 0.0 <= v <= most}
    if bad:
        where = "" if "" in bad else f" at {sorted(bad)[:10]}"
        raise ValueError(
            f"{field} must be between 0 and {most:g}{where} (a decimal per "
            "unit of weight traded: 0.0015 is 15 bps); got "
            f"{list(bad.values())[:3]}."
        )
    return value


class OptimizeRebalanceInput(NamedCovarianceInput):
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
    expected_returns: Dict[str, float] = Field(
        ...,
        description="Asset -> expected return, the covariance's units; every asset.",
    )
    current_weights: Optional[Dict[str, float]] = Field(
        None, description="Asset -> weight held (absent = 0); none is all cash."
    )
    risk_aversion: float = Field(
        1.0, gt=0, le=1e6, description="gamma in mu'w - (gamma/2) w'Sw."
    )
    linear_cost: CostInput = Field(
        0.0,
        description="Cost per unit of weight traded, decimal (0.0015 = 15 bps): "
        "one number or asset -> cost.",
    )
    quadratic_cost: CostInput = Field(
        0.0, description="eta in eta (w - w0)^2: one number or asset -> eta."
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
        None, description="Asset -> upper bound, overriding max_weight."
    )
    budget: float = Field(1.0, ge=-100, le=100, description="Sum of weights.")
    budget_type: Literal["equal", "at_most"] = "equal"
    groups: Optional[Dict[str, List[str]]] = Field(
        None, max_length=500, description="Group -> member assets."
    )
    group_min: Optional[Dict[str, float]] = None
    group_max: Optional[Dict[str, float]] = None
    factor_loadings: Optional[Dict[str, Dict[str, float]]] = Field(
        None, max_length=5000, description="Asset -> {factor: loading}; every asset."
    )
    exposure_min: Optional[Dict[str, float]] = None
    exposure_max: Optional[Dict[str, float]] = None
    max_turnover: Optional[float] = Field(
        None, gt=0, le=100, description="Cap on sum |w - w0|."
    )
    max_gross: Optional[float] = Field(
        None, gt=0, le=100, description="Cap on sum |w|."
    )
    max_volatility: Optional[float] = Field(
        None, gt=0, le=10, description="Volatility cap, the covariance's units."
    )

    @field_validator("assets")
    @classmethod
    def _assets_once(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        return v if v is None else _distinct(v, "assets")

    @field_validator(
        "expected_returns",
        "current_weights",
        "lower_bounds",
        "upper_bounds",
        "group_min",
        "group_max",
        "exposure_min",
        "exposure_max",
    )
    @classmethod
    def _finite_maps(cls, v: Optional[Dict[str, float]], info):
        return _finite_entries(v, info.field_name)

    @field_validator("linear_cost")
    @classmethod
    def _linear(cls, v: CostInput) -> CostInput:
        return _cost(v, "linear_cost", _MAX_LINEAR_COST)

    @field_validator("quadratic_cost")
    @classmethod
    def _quadratic(cls, v: CostInput) -> CostInput:
        return _cost(v, "quadratic_cost", _MAX_QUADRATIC_COST)

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
    def _limits_need_their_rows(self) -> "OptimizeRebalanceInput":
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


#: Why a number is null when the exact solve reached no answer.
_NO_ANSWER = "the exact solve reached no answer, so there are no weights to measure"
_NO_TARGET = (
    "the frictionless target (no costs, no turnover cap) was not certified, "
    "so nothing is measured against it"
)


class RebalanceBinding(ExplainsNulls):
    """One constraint at its bound. A null here is reported by the result
    that holds the row."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "shadow_price": (
            "the volatility cap equals the lowest volatility the other "
            "constraints allow, so its price is unbounded"
        ),
    }

    kind: str = Field(
        ...,
        description="budget, lower_bound, upper_bound, fixed_weight, group_*, "
        "exposure_*, max_turnover, max_gross or max_volatility.",
    )
    name: Optional[str] = None
    bound: Stat = None
    shadow_price: Stat = Field(
        None,
        description="d utility / d bound, net of trading costs: what loosening "
        "it by one unit is worth.",
    )


class OptimizeRebalanceResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "turnover": _NO_ANSWER,
        "gross": _NO_ANSWER,
        "net": _NO_ANSWER,
        "expected_return": _NO_ANSWER,
        "volatility": _NO_ANSWER,
        "linear_cost": _NO_ANSWER,
        "quadratic_cost": _NO_ANSWER,
        "utility": _NO_ANSWER,
        "effective_risk_aversion": (
            "the volatility cap equals the lowest volatility attainable, so no "
            "finite risk aversion gives this answer"
        ),
        "target_weights": _NO_TARGET,
        "target_turnover": _NO_TARGET,
        "target_trading_cost": _NO_TARGET,
        "target_utility": _NO_TARGET,
        "distance_to_target": _NO_TARGET,
        "tracking_to_target": _NO_TARGET,
    }

    n_assets: int = 0
    assets: List[str] = Field(default_factory=list)
    weights: Dict[str, Stat] = Field(default_factory=dict)
    trades: Dict[str, Stat] = Field(
        default_factory=dict, description="weight - current weight."
    )
    no_trade: List[str] = Field(
        default_factory=list,
        description="Assets whose weight is exactly the weight held.",
    )
    turnover: Stat = Field(None, description="sum |trades|.")
    gross: Stat = None
    net: Stat = None
    expected_return: Stat = None
    volatility: Stat = None
    linear_cost: Stat = Field(None, description="sum kappa |trade|.")
    quadratic_cost: Stat = Field(None, description="sum eta trade^2.")
    utility: Stat = Field(
        None,
        description="mu'w - (gamma/2) w'Sw - linear_cost - quadratic_cost.",
    )
    required_turnover: Stat = Field(
        None,
        description="The least turnover any answer pays to reach the "
        "constraints; 0 when the weights held satisfy them.",
    )
    risk_aversion: Stat = None
    effective_risk_aversion: Stat = Field(
        None, description="risk_aversion raised by a binding volatility cap."
    )
    binding_constraints: List[RebalanceBinding] = Field(
        default_factory=list,
        description="Every constraint at its bound, largest |shadow_price| first.",
    )
    group_totals: Dict[str, Stat] = Field(default_factory=dict)
    exposures: Dict[str, Stat] = Field(default_factory=dict)
    target_weights: Optional[Dict[str, Stat]] = Field(
        None,
        description="The frictionless target: the same problem with no costs "
        "and no turnover cap.",
    )
    target_turnover: Stat = Field(
        None, description="The turnover trading to the target would take."
    )
    target_trading_cost: Stat = Field(
        None, description="What trading to the target would cost."
    )
    target_utility: Stat = Field(
        None,
        description="The target's utility net of that cost; above `utility` "
        "only when the turnover cap forbids the target.",
    )
    distance_to_target: Stat = Field(
        None, description="sum |weights - target_weights|: what the costs held back."
    )
    tracking_to_target: Stat = Field(
        None,
        description="sqrt((w - target)' S (w - target)): that difference as "
        "volatility.",
    )
    condition_number: Stat = None
    certified: bool = Field(
        False,
        description="True only when every KKT residual, computed from the "
        "weights, is within the certificate's tolerance.",
    )
    certificate: Optional[OptimizerCertificate] = None
    method: Optional[str] = None
    passes: int = 0
    warnings: List[str] = Field(default_factory=list)


# ── the tool ────────────────────────────────────────────────────────────


def _bounds(input_data: OptimizeRebalanceInput, names: List[str], who: str):
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
    return lower, upper


def optimize_rebalance(
    input_data: OptimizeRebalanceInput,
) -> OptimizeRebalanceResult:
    who = "optimize_rebalance"
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
    lower, upper = _bounds(input_data, names, who)
    common = dict(
        expected_returns=input_data.expected_returns,
        current_weights=input_data.current_weights,
        risk_aversion=input_data.risk_aversion,
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
        max_gross=input_data.max_gross,
        max_volatility=input_data.max_volatility,
    )
    result = lib.optimize_rebalance(
        covariance,
        linear_cost=input_data.linear_cost,
        quadratic_cost=input_data.quadratic_cost,
        max_turnover=input_data.max_turnover,
        **common,
    )
    result.pop("active_set", None)
    warnings = list(result.pop("warnings", []))
    out = dict(result)
    if result["weights"]:
        _against_target(out, covariance, common, input_data, warnings)
    out["warnings"] = warnings
    return OptimizeRebalanceResult(**out)


def _against_target(
    out: Dict,
    covariance: pd.DataFrame,
    common: Dict,
    input_data: OptimizeRebalanceInput,
    warnings: List[str],
) -> None:
    """The frictionless target and how far the answer stays from it."""
    names = list(out["assets"])
    target = lib.optimize_rebalance(covariance, **common)
    if not target["certified"] or not target["weights"]:
        warnings.append(
            "The frictionless target did not certify ("
            + "; ".join(target["warnings"][-1:])
            + "), so the target fields are null."
        )
        out["target_weights"] = None
        return
    w = np.array([out["weights"][a] for a in names])
    t = np.array([target["weights"][a] for a in names])
    held = input_data.current_weights or {}
    w0 = np.array([float(held.get(a, 0.0)) for a in names])
    kappa = _per_asset(input_data.linear_cost, names)
    eta = _per_asset(input_data.quadratic_cost, names)
    to_target = t - w0
    cost = float(kappa @ np.abs(to_target) + eta @ (to_target * to_target))
    gap = w - t
    cov = covariance.loc[names, names].to_numpy(dtype=float)
    out["target_weights"] = dict(zip(names, (float(v) for v in t)))
    out["target_turnover"] = float(np.sum(np.abs(to_target)))
    out["target_trading_cost"] = cost
    out["target_utility"] = float(target["utility"]) - cost
    out["distance_to_target"] = float(np.sum(np.abs(gap)))
    out["tracking_to_target"] = math.sqrt(max(float(gap @ cov @ gap), 0.0))


def _per_asset(value: CostInput, names: List[str]) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([float(value[a]) for a in names])
    return np.full(len(names), float(value))


REBALANCE_TOOL_DEFS = [
    (
        "optimize_rebalance",
        "The trade to make from the weights you hold: exact mean-variance "
        "weights net of linear and quadratic trading costs, under turnover, "
        "gross and volatility caps. Names inside their no-trade band keep "
        "their weight exactly (listed in no_trade). Takes optimize_portfolio's "
        "bounds, budget, group and factor limits; returns the trades, every "
        "binding constraint's shadow price net of costs (the turnover cap's "
        "is what one more unit of turnover is worth), a KKT certificate "
        "computed from the weights, and the frictionless target beside it -- "
        "its turnover, cost and distance -- to show what the costs held back. "
        "No tickers: pass a covariance or covariance_ref.",
        OptimizeRebalanceInput,
    ),
]

REBALANCE_TOOL_DISPATCH = {
    "optimize_rebalance": (optimize_rebalance, OptimizeRebalanceInput),
}
