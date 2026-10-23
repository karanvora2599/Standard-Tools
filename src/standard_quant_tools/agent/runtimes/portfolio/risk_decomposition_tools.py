"""
Two risk decompositions of a portfolio already held: under a factor model,
and in its tail.

`get_marginal_risk_contribution` splits a portfolio's volatility by asset
from a covariance matrix. These answer the two questions it cannot:

- `decompose_factor_risk` -- how much of the risk is the factors and how
  much the names' own, factor by factor and name by name, and the same for
  the tracking error against a benchmark. The model is Sigma = B F B' + D,
  and D (the specific variances) is required: a factor model without it
  reads every book as a pure factor bet.
- `decompose_tail_risk` -- which positions the expected shortfall and the
  VaR come from, on a scenario matrix rather than on a covariance, so a
  fat-tailed or non-linear book is decomposed in the scenarios it actually
  loses in.

BOTH TAKE INLINE DATA OR A REFERENCE, NEVER TICKERS: a model's loadings and
variances, a scenario matrix, or a published `returns_panel`. Nothing is
fetched, so the same input gives the same bits. The library functions are
`portfolio.factor_risk.decompose_factor_risk` and
`portfolio.tail_risk.decompose_tail_risk`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls
from standard_quant_tools.agent.runtimes.streams import (
    StreamAlignment,
    aligned_returns_ref,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import factor_risk as factor_lib
from standard_quant_tools.portfolio import tail_risk as tail_lib

from .construction_tools import Stat, _finite_map, _Result

#: The most assets a decomposition takes in one call, and factors. Held
#: in the validators rather than advertised in the schema: the bound is a
#: guard against a runaway payload, not something a caller plans around,
#: and the portfolio runtime's advertised schemas have no bytes to spare.
MAX_ASSETS = 5000
MAX_FACTORS = 200

# ── inputs ──────────────────────────────────────────────────────────────


def _sized(values: Optional[Dict], field: str, most: int) -> Optional[Dict]:
    """A map with at least one entry and at most `most`, or a refusal."""
    if values is None:
        return values
    if not values:
        raise ValueError(f"{field} is empty; name at least one entry.")
    if len(values) > most:
        raise ValueError(
            f"{field} has {len(values)} entries, more than the {most} one call "
            "takes. Decompose the book in parts, or aggregate it first."
        )
    return values


def _finite_rows(values: Dict[str, Dict[str, float]], field: str) -> Dict:
    for name, row in values.items():
        _finite_map(row, f"{field}[{name!r}]")
    return values


class FactorRiskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(
        ..., description="Asset -> weight (fraction of the book)."
    )
    factor_loadings: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Asset -> {factor: loading} on every factor (0.0 if none).",
    )
    factor_covariance: Dict[str, Dict[str, float]] = Field(
        ..., description="Factor -> {factor: covariance}. Annual in, annual out."
    )
    specific_variance: Dict[str, float] = Field(
        ...,
        description="Asset -> specific return variance, for every asset of "
        "factor_loadings.",
    )
    benchmark_weights: Optional[Dict[str, float]] = Field(
        None, description="Asset -> benchmark weight, for active risk."
    )

    @field_validator("weights", "specific_variance", "benchmark_weights")
    @classmethod
    def _finite(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict]:
        v = _sized(v, info.field_name, MAX_ASSETS)
        return v if v is None else _finite_map(v, info.field_name)

    @field_validator("factor_loadings", "factor_covariance")
    @classmethod
    def _finite_nested(cls, v: Dict[str, Dict[str, float]], info) -> Dict:
        most = MAX_FACTORS if info.field_name == "factor_covariance" else MAX_ASSETS
        return _finite_rows(_sized(v, info.field_name, most), info.field_name)


class TailRiskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(
        ...,
        description="Asset -> weight: book fraction on returns, quantity on P&L.",
    )
    scenarios: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Asset -> its value in each scenario (scenario i at index "
        "i); the weights' assets.",
    )
    returns_ref: Optional[str] = Field(
        None,
        description="A returns_panel reference in place of scenarios: each "
        "shared date is one.",
    )
    confidence: float = Field(0.95, ge=0.5, lt=1)
    bandwidth: Optional[float] = Field(
        None,
        gt=0,
        description="Component VaR's kernel bandwidth, P&L units; default "
        "Silverman's.",
    )

    @field_validator("weights")
    @classmethod
    def _finite_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        return _finite_map(_sized(v, "weights", MAX_ASSETS), "weights")

    @field_validator("scenarios")
    @classmethod
    def _finite_scenarios(
        cls, v: Optional[Dict[str, List[float]]]
    ) -> Optional[Dict[str, List[float]]]:
        if v is None:
            return v
        bad = {
            name: [i for i, x in enumerate(values) if not math.isfinite(x)][:5]
            for name, values in v.items()
            if not all(math.isfinite(x) for x in values)
        }
        if bad:
            raise ValueError(
                f"scenarios are not finite at {dict(list(bad.items())[:5])} "
                "(asset: scenario positions). A scenario needs a finite value "
                "for every asset; drop that scenario from every list."
            )
        return v

    @field_validator("bandwidth")
    @classmethod
    def _finite_bandwidth(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not math.isfinite(v):
            raise ValueError(
                f"bandwidth must be a finite positive number, got {v!r}; an "
                "infinite one weights every scenario alike, which is the mean "
                "P&L rather than the VaR. Omit it for Silverman's rule."
            )
        return v

    @model_validator(mode="after")
    def _one_source(self) -> "TailRiskInput":
        given = [
            n for n in ("scenarios", "returns_ref") if getattr(self, n) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "decompose_tail_risk needs exactly one of scenarios / "
                f"returns_ref; got {given or 'neither'}. `scenarios` is the "
                "matrix inline, returns_ref a published returns_panel."
            )
        return self


# ── results ─────────────────────────────────────────────────────────────

#: Why a share, contribution or marginal of a zero volatility is null.
_ZERO_RISK = (
    "the volatility (or tracking error) it divides by is zero, so there is no "
    "risk to apportion"
)


class FactorRiskFactor(ExplainsNulls):
    """One factor's row. It has no warnings of its own: a null here is
    reported by the result that holds the row."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "correlation": _ZERO_RISK,
        "contribution": _ZERO_RISK,
        "share": _ZERO_RISK,
    }

    factor: str = ""
    exposure: Stat = Field(None, description="x_k = (B'w)_k.")
    factor_volatility: Stat = Field(None, description="sqrt(F_kk).")
    correlation: Stat = Field(
        None,
        description="The factor's correlation with the portfolio (or the "
        "active portfolio), (F x)_k / (sigma_k sigma).",
    )
    contribution: Stat = Field(
        None,
        description="x_k (F x)_k / sigma = exposure x factor_volatility x "
        "correlation: the factor's Euler contribution to the volatility.",
    )
    share: Stat = Field(None, description="contribution / volatility.")


class FactorRiskAsset(ExplainsNulls):
    """One asset's row, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        name: _ZERO_RISK
        for name in (
            "marginal",
            "contribution",
            "share",
            "factor_contribution",
            "specific_contribution",
        )
    }

    asset: str = ""
    weight: Stat = Field(
        None,
        description="The weight decomposed: the portfolio's under `portfolio`, "
        "portfolio minus benchmark under `active`.",
    )
    marginal: Stat = Field(
        None,
        description="d sigma / d w_i = (Sigma w)_i / sigma. Negative: adding "
        "to it reduces the risk.",
    )
    contribution: Stat = Field(
        None, description="weight x marginal; these sum to the volatility."
    )
    share: Stat = None
    factor_contribution: Stat = Field(
        None, description="The part of contribution that comes through the factors."
    )
    specific_contribution: Stat = Field(
        None, description="w_i^2 d_i / sigma: the asset's own risk."
    )


class FactorRiskSide(ExplainsNulls):
    """The decomposition of one weight vector, reported by its result."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "factor_share": _ZERO_RISK,
        "specific_share": _ZERO_RISK,
        "specific_contribution": _ZERO_RISK,
        "sum_of_contributions": _ZERO_RISK,
    }

    volatility: Stat = Field(
        None,
        description="sqrt(factor_variance + specific_variance), in the inputs' units.",
    )
    variance: Stat = None
    factor_variance: Stat = Field(None, description="x'Fx with x = B'w.")
    specific_variance: Stat = Field(None, description="sum(w_i^2 d_i).")
    factor_volatility: Stat = None
    specific_volatility: Stat = None
    factor_share: Stat = Field(
        None, description="factor_variance / variance: the share of the TOTAL."
    )
    specific_share: Stat = None
    factor_specific_covariance: Stat = Field(
        None,
        description="0 by construction: the model takes specific returns as "
        "uncorrelated with the factors and with each other, which is what "
        "makes variance = factor + specific. Stated, not measured.",
    )
    exposures: Dict[str, Stat] = Field(default_factory=dict)
    by_factor: List[FactorRiskFactor] = Field(default_factory=list)
    specific_contribution: Stat = Field(
        None,
        description="specific_variance / volatility: with the factors' "
        "contributions it sums to the volatility.",
    )
    by_asset: List[FactorRiskAsset] = Field(default_factory=list)
    sum_of_contributions: Stat = Field(
        None, description="Of by_asset: the volatility, to rounding."
    )


class FactorRiskResult(_Result):
    null_reasons = {
        "tracking_error": "no benchmark_weights were given, so there is no "
        "active portfolio",
        "benchmark_volatility": "no benchmark_weights were given",
    }

    n_assets: int = Field(
        0, description="Rows: assets named in weights or the benchmark."
    )
    n_model_assets: int = Field(0, description="Assets in factor_loadings.")
    n_factors: int = 0
    volatility: Stat = Field(
        None,
        description="The portfolio's total volatility, in the units of the "
        "inputs: annual if factor_covariance and specific_variance are.",
    )
    tracking_error: Stat = Field(
        None, description="The active portfolio's volatility, same units."
    )
    benchmark_volatility: Stat = None
    portfolio: FactorRiskSide = Field(default_factory=FactorRiskSide)
    active: Optional[FactorRiskSide] = Field(
        None,
        description="The same decomposition of the active weights (portfolio "
        "minus benchmark); null without benchmark_weights.",
    )
    benchmark_exposures: Optional[Dict[str, Stat]] = None


class TailRiskAsset(ExplainsNulls):
    """One asset's row, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "es_share": "the expected shortfall it is a share of is zero",
        "var_share": "the kernel-smoothed VaR it is a share of is zero",
    }

    asset: str = ""
    weight: Stat = None
    marginal_es: Stat = Field(
        None,
        description="d ES / d w_i: minus the asset's mean in the portfolio's tail.",
    )
    component_es: Stat = Field(
        None,
        description="weight x marginal_es; these sum to es exactly (to "
        "rounding). Negative: a hedge in the tail.",
    )
    es_share: Stat = None
    marginal_var: Stat = Field(
        None,
        description="d VaR / d w_i: minus the kernel-weighted mean of the asset "
        "around the VaR.",
    )
    component_var: Stat = Field(
        None, description="weight x marginal_var; these sum to var_kernel."
    )
    var_share: Stat = Field(None, description="component_var / var_kernel.")


class TailRiskResult(_Result):
    n_scenarios: int = 0
    n_assets: int = 0
    confidence: Stat = None
    var: Stat = Field(
        None,
        description="Value at risk, a LOSS and positive: minus the empirical "
        "(1 - confidence) quantile of the portfolio P&L, as var_historical "
        "gives it. In the scenarios' units.",
    )
    es: Stat = Field(
        None,
        description="Expected shortfall, a loss and positive: minus the mean of "
        "the scenarios at or below that quantile, as cvar gives it.",
    )
    var_kernel: Stat = Field(
        None,
        description="The kernel-smoothed VaR the component VaRs sum to; it "
        "differs from var by the smoothing's bias.",
    )
    n_tail_observations: int = Field(
        0, description="How many scenarios es (and every component_es) averages."
    )
    bandwidth: Stat = Field(
        None, description="The kernel bandwidth used, in P&L units."
    )
    bandwidth_rule: Literal["silverman", "given"] = "silverman"
    kernel_effective_scenarios: Stat = Field(
        None,
        description="(sum K)^2 / sum K^2: about how many scenarios component VaR "
        "rests on.",
    )
    sum_of_es_components: Stat = None
    sum_of_var_components: Stat = None
    by_asset: List[TailRiskAsset] = Field(default_factory=list)
    n_hedges: int = Field(0, description="Assets with a negative component_es.")
    streams: Optional[StreamAlignment] = Field(
        None,
        description="With returns_ref: the dates the weighted columns share "
        "(the scenarios) and what each column lost to them.",
    )


# ── tools ───────────────────────────────────────────────────────────────


def decompose_factor_risk(input_data: FactorRiskInput) -> FactorRiskResult:
    return FactorRiskResult(
        **factor_lib.decompose_factor_risk(
            input_data.weights,
            input_data.factor_loadings,
            input_data.factor_covariance,
            input_data.specific_variance,
            benchmark_weights=input_data.benchmark_weights,
        )
    )


def _scenario_frame(
    scenarios: Dict[str, List[float]], names: List[str], who: str
) -> pd.DataFrame:
    """The inline scenarios as a frame in the weights' order, or a refusal
    naming the assets the two disagree on or the lists of other lengths."""
    known, named = set(scenarios), set(names)
    absent = [a for a in names if a not in known]
    unweighted = [a for a in scenarios if a not in named]
    if absent or unweighted:
        parts = []
        if absent:
            parts.append(f"weights name {absent[:5]}, which have no scenarios")
        if unweighted:
            parts.append(f"scenarios has {unweighted[:5]}, which have no weight")
        raise ValidationError(
            f"{who}: weights and scenarios must name the same assets; "
            f"{' and '.join(parts)}. Give every asset a weight (0.0 to hold "
            "none of it) and a list of scenarios."
        )
    lengths = {name: len(values) for name, values in scenarios.items()}
    if len(set(lengths.values())) > 1:
        raise ValidationError(
            f"{who}: the scenario lists have different lengths ({lengths}). "
            "Scenario i is position i of every list, so every asset needs one "
            "value per scenario."
        )
    return pd.DataFrame({name: scenarios[name] for name in names}, dtype="float64")


def decompose_tail_risk(input_data: TailRiskInput) -> TailRiskResult:
    who = "decompose_tail_risk"
    names = list(input_data.weights)
    streams = None
    notes: List[str] = []
    if input_data.returns_ref is not None:
        frame, streams, notes = aligned_returns_ref(
            input_data.returns_ref,
            what=who,
            min_overlap=tail_lib.MIN_SCENARIOS,
            columns=names,
        )
    else:
        assert input_data.scenarios is not None  # one of the two, by the model
        frame = _scenario_frame(input_data.scenarios, names, who)
    result = tail_lib.decompose_tail_risk(
        frame,
        input_data.weights,
        confidence=input_data.confidence,
        bandwidth=input_data.bandwidth,
    )
    result["warnings"] = notes + list(result.get("warnings", []))
    return TailRiskResult(**result, streams=streams)


RISK_DECOMPOSITION_TOOL_DEFS = [
    (
        "decompose_factor_risk",
        "Splits a portfolio's risk under a factor model (B F B' + D) into "
        "factor and specific parts, by factor and by asset, with tracking "
        "error against a benchmark split the same way. Euler contributions "
        "sum to the volatility; a factor's is exposure x volatility x "
        "correlation. Units are the inputs'.",
        FactorRiskInput,
    ),
    (
        "decompose_tail_risk",
        "Which positions a portfolio's expected shortfall and VaR come from: "
        "Euler component ES and kernel-smoothed component VaR from a scenario "
        "matrix or a returns_ref. Component ES sums to ES exactly, component "
        "VaR to the kernel VaR. Losses are positive, as cvar gives them; the "
        "tail count is always reported.",
        TailRiskInput,
    ),
]

RISK_DECOMPOSITION_TOOL_DISPATCH = {
    "decompose_factor_risk": (decompose_factor_risk, FactorRiskInput),
    "decompose_tail_risk": (decompose_tail_risk, TailRiskInput),
}

__all__ = [
    "FactorRiskInput",
    "FactorRiskResult",
    "RISK_DECOMPOSITION_TOOL_DEFS",
    "RISK_DECOMPOSITION_TOOL_DISPATCH",
    "TailRiskInput",
    "TailRiskResult",
    "decompose_factor_risk",
    "decompose_tail_risk",
]
