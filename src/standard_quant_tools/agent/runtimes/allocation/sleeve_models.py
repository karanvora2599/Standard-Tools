"""
Inputs and results of the three sleeve tools: putting return streams on one
calendar, allocating capital across them, and walking an allocation rule
forward.

THE ALLOCATING TOOLS TAKE STREAMS, NEVER TICKERS: `returns` as
{stream: {ISO date: return}}, or one published `returns_panel` as
`returns_ref` (with `columns`), the field names every allocation tool
shares. They take no `ReturnStream`, whose symbol form would make them
fetchers. `combine_return_streams` is the one door that does take it --
a reference, dated values, or a symbol with both ends of its window -- and
publishes what it aligned as the panel the others read. Every field
description is short on purpose: the runtime's advertised schemas share
one budget, and the rules behind each field are in the library modules
`allocation.sleeves` and `allocation.walk_forward`.

`AllocationRule` is the one nested type: `allocate_across_sleeves` takes
its fields flat, `evaluate_allocation_rule` takes one as `rule`, so a rule
is spelled the same way in both.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.models import Seed
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.agent.runtimes.streams import (
    ReturnStreams,
    StreamAlignment,
    columns_only_with_ref,
    exactly_one_source,
    finite_dated_streams,
)
from standard_quant_tools.allocation.sleeves import SleeveRule
from standard_quant_tools.backtest.financing import _compact_schema

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

#: Every input model's config: no extra arguments, and the schema without
#: the model descriptions and field titles, which restate names already in
#: it -- the runtime's advertised schemas share one budget.
_INPUT = ConfigDict(extra="forbid", json_schema_extra=_compact_schema)

#: The most sleeves one call takes. Held in the validator rather than the
#: schema: a guard against a runaway payload, not something to plan around.
MAX_SLEEVES = 200

Method = Literal[
    "equal",
    "inverse_volatility",
    "risk_parity",
    "hrp",
    "min_variance",
    "max_sharpe",
    "shrunk_sharpe",
]

CovarianceMethod = Literal["sample", "ledoit_wolf", "ewma", "ewma_shrunk"]

#: estimate_covariance's own default half-life, the rule's too.
_DEFAULT_HALFLIFE = SleeveRule.halflife


def _finite_positive(values: Optional[Dict[str, float]], field: str) -> Optional[Dict]:
    if values is None:
        return values
    if not values:
        raise ValueError(f"{field} is empty; name at least one sleeve or omit it.")
    bad = {k: v for k, v in values.items() if not (math.isfinite(v) and v > 0)}
    if bad:
        raise ValueError(
            f"{field} must be finite and positive for every sleeve, got "
            f"{dict(list(bad.items())[:5])}."
        )
    return values


# ── inputs ──────────────────────────────────────────────────────────────


class SleeveReturns(BaseModel):
    """The sleeves' return streams: dated values inline, or one published
    panel. No symbol form: these tools fetch nothing."""

    model_config = _INPUT

    #: The tool a refusal names.
    tool: ClassVar[str] = ""

    returns: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="{stream: {ISO date: return}}: a sleeve per stream, "
        "aligned on the dates they share.",
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of returns."
    )
    columns: Optional[List[str]] = Field(
        None, min_length=1, description="returns_ref's streams; default all."
    )

    @field_validator("returns")
    @classmethod
    def _finite_returns(cls, v: Optional[Dict]) -> Optional[Dict]:
        return finite_dated_streams(v, "returns", most=MAX_SLEEVES)

    @model_validator(mode="after")
    def _a_source(self) -> "SleeveReturns":
        exactly_one_source(self, "returns", "returns_ref", self.tool)
        columns_only_with_ref(self, "columns", "returns_ref", most=MAX_SLEEVES)
        return self


class AllocationRule(BaseModel):
    """An allocation rule: its method, covariance and constraints."""

    model_config = _INPUT

    method: Method = Field(
        "risk_parity",
        description="shrunk_sharpe: the tangency on James-Stein-shrunk Sharpes.",
    )
    covariance_method: CovarianceMethod = "ledoit_wolf"
    halflife: float = Field(
        _DEFAULT_HALFLIFE, gt=0, le=1_000_000, description="ewma's, periods."
    )
    risk_budgets: Optional[Dict[str, float]] = Field(
        None, description="risk_parity: sleeve -> risk share."
    )
    long_only: bool = Field(True, description="False: the optimizers may short.")
    max_weight: Optional[float] = Field(None, gt=0, le=1)
    target_volatility: Optional[float] = Field(
        None, gt=0, le=10, description="Annual; scales the gross."
    )
    max_leverage: float = Field(1.0, gt=0, le=10, description="Its gross cap.")
    capital: Optional[float] = Field(None, gt=0, le=1e15)
    capacity: Optional[Dict[str, float]] = Field(
        None, description="Sleeve -> most capital it takes."
    )
    capacity_excess: Literal["cash", "redistribute"] = "cash"

    @field_validator("risk_budgets", "capacity")
    @classmethod
    def _finite_maps(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict]:
        return _finite_positive(v, info.field_name)

    @model_validator(mode="after")
    def _a_rule(self) -> "AllocationRule":
        # A changed half-life, not a given one: a caller that sends the
        # defaults back is sending nothing.
        if (
            self.halflife != _DEFAULT_HALFLIFE
            and not self.covariance_method.startswith("ewma")
        ):
            raise ValueError(
                f"halflife is read by the ewma covariances only, and "
                f"covariance_method={self.covariance_method!r}. Drop halflife, "
                "or use ewma or ewma_shrunk."
            )
        # The library's own rules, refused here by name rather than at the
        # first fit.
        self.to_rule()
        return self

    def to_rule(self) -> SleeveRule:
        return SleeveRule(
            **{name: getattr(self, name) for name in AllocationRule.model_fields}
        )


class CombineStreamsInput(BaseModel):
    model_config = _INPUT

    streams: Optional[ReturnStreams] = Field(
        None,
        description="Name -> stream: a returns_panel ref (and column), "
        "{ISO date: return} values, or a symbol with both window ends.",
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of streams."
    )
    columns: Optional[List[str]] = Field(
        None, min_length=1, description="returns_ref's streams; default all."
    )
    min_overlap: int = Field(
        20, ge=1, le=1_000_000, description="Fewest dates every stream must share."
    )
    run_id: str = Field(..., description="Where the aligned returns_panel goes.")

    @model_validator(mode="after")
    def _a_source(self) -> "CombineStreamsInput":
        exactly_one_source(self, "streams", "returns_ref", "combine_return_streams")
        columns_only_with_ref(self, "columns", "returns_ref", most=MAX_SLEEVES)
        if self.streams is not None and len(self.streams) > MAX_SLEEVES:
            raise ValueError(
                f"{len(self.streams)} streams, more than the {MAX_SLEEVES} one "
                "call takes."
            )
        return self


class _Conventions(BaseModel):
    cash_rate: float = Field(
        0.0, ge=-10, le=10, description="Decimal a year; Sharpes are over it."
    )
    periods_per_year: int = Field(252, ge=1, le=31_536_000)
    block_size: Optional[int] = Field(None, ge=1, le=100_000)
    random_seed: Seed = 0


class AllocateSleevesInput(SleeveReturns, AllocationRule, _Conventions):
    model_config = _INPUT
    tool: ClassVar[str] = "allocate_across_sleeves"

    n_bootstrap: int = Field(500, ge=0, le=5000, description="0: no weight bands.")


class EvaluateAllocationRuleInput(SleeveReturns, _Conventions):
    model_config = _INPUT
    tool: ClassVar[str] = "evaluate_allocation_rule"

    rule: AllocationRule
    benchmark_rule: Method = "equal"
    n_trials: int = Field(
        ..., ge=1, le=1_000_000, description="Rules and settings tried, this one too."
    )
    estimation_window: int = Field(252, ge=2, le=1_000_000)
    rebalance_every: int = Field(21, ge=1, le=1_000_000)
    cost_bps: float = Field(0.0, ge=0, le=10_000, description="Per unit of turnover.")
    n_bootstrap: int = Field(1000, ge=0, le=20_000)
    run_id: Optional[str] = Field(None, description="Publishes the streams, weights.")


# ── results ─────────────────────────────────────────────────────────────


class _Result(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")

    warnings: List[str] = Field(default_factory=list)


class CombineStreamsResult(_Result):
    returns_ref: str = Field(
        ..., description="The aligned returns_panel: one column per stream."
    )
    columns: List[str] = Field(default_factory=list)
    n_dates: int = Field(0, description="Dates every stream has: the panel's rows.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    alignment: StreamAlignment = Field(
        default_factory=StreamAlignment,
        description="What each stream supplied and what the common calendar "
        "cost it: before the window every stream covers, after it, and "
        "inside it on a date another stream lacks.",
    )


class BindingCap(BaseModel):
    """A cap that binds: the sleeve, which cap, and its limit as a weight."""

    model_config = ConfigDict(extra="forbid")

    sleeve: str
    cap: Literal["max_weight", "capacity"]
    limit: float = Field(..., description="The cap as a share of the capital.")


#: Why a per-sleeve shrinkage number can be null.
_NO_SHARPE = "the sleeve's Sharpe ratio is undefined here"


class ShrinkageReport(ExplainsNulls):
    """What the James-Stein shrinkage did, reported by its result."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {"*": _NO_SHARPE}

    shrinkage: Stat = Field(
        None,
        description="c in [0, 1]: 1 pools every Sharpe ratio to pooled_sharpe, "
        "0 keeps them as estimated.",
    )
    pooled_sharpe: Stat = Field(
        None, description="The precision-weighted mean Sharpe ratio, annual."
    )
    dispersion_statistic: Stat = Field(
        None,
        description="Q = d'V^-1 d, d the Sharpe ratios less the pooled one; "
        "c = min(1, (k - 3) / Q).",
    )
    sharpe: Dict[str, Stat] = Field(default_factory=dict)
    standard_error: Dict[str, Stat] = Field(
        default_factory=dict, description="Lo/Mertens, annual."
    )
    shrunk_sharpe: Dict[str, Stat] = Field(default_factory=dict)


class SleeveAllocationResult(_Result):
    null_reasons = {
        "cash_amount": "no capital was given",
        "target_volatility_scale": "no target_volatility was given",
        "stability_l1": "there are no bootstrap bands: n_bootstrap is 0, or "
        "fewer than half of the resamples could be fitted",
        "in_sample_sharpe": "the allocation's return has no dispersion",
        "effective_n_risk": "the allocation has no volatility to apportion",
        "risk_contributions": "the allocation has no volatility to apportion",
        "risk_shares": "the allocation has no volatility to apportion",
        "effective_n": "the allocation holds nothing",
    }

    method: str = ""
    covariance_method: str = ""
    n_sleeves: int = 0
    n_observations: int = Field(0, description="Shared dates the rule was fitted on.")
    periods_per_year: int = 252
    weights: Dict[str, Stat] = Field(
        default_factory=dict, description="Share of the capital per sleeve."
    )
    cash_weight: Stat = Field(
        None, description="1 - sum(weights); negative is borrowing at cash_rate."
    )
    gross_weight: Stat = None
    net_weight: Stat = None
    capital_by_sleeve: Optional[Dict[str, Stat]] = Field(
        None, description="Currency per sleeve; null without capital."
    )
    cash_amount: Stat = None
    risk_contributions: Dict[str, Stat] = Field(
        default_factory=dict,
        description="w_i (Sigma w)_i / sigma: they sum to ex_ante_volatility.",
    )
    risk_shares: Dict[str, Stat] = Field(
        default_factory=dict, description="risk_contributions / ex_ante_volatility."
    )
    ex_ante_volatility: Stat = Field(
        None, description="sqrt(w'Sigma w), annual, on covariance_method."
    )
    effective_n: Stat = Field(None, description="(sum |w|)^2 / sum w^2.")
    effective_n_risk: Stat = Field(None, description="1 / sum risk_shares^2.")
    target_volatility_scale: Stat = Field(
        None, description="What target_volatility multiplied the weights by."
    )
    binding_caps: List[BindingCap] = Field(default_factory=list)
    in_sample_sharpe: Stat = Field(
        None,
        description="IN-SAMPLE: the allocation's Sharpe ratio over cash_rate on "
        "the returns it was fitted on. Not a forecast; evaluate_allocation_rule "
        "tests the rule.",
    )
    weight_p5: Optional[Dict[str, Stat]] = Field(
        None, description="5th percentile of each bootstrap weight."
    )
    weight_p95: Optional[Dict[str, Stat]] = Field(
        None, description="95th percentile of each bootstrap weight."
    )
    stability_l1: Stat = Field(
        None,
        description="Mean sum |w_boot - w| over the resamples: 0 is a rule the "
        "sample does not move, 2 one that moves all of its capital.",
    )
    n_bootstrap: int = 0
    n_bootstrap_failed: int = Field(
        0, description="Resamples on which the rule could not be fitted."
    )
    block_size: int = 0
    bootstrap_scale: Stat = Field(
        None, description="sqrt(n / (n - block)): what the draws are widened by."
    )
    random_seed: int = 0
    certified: Optional[bool] = Field(
        None, description="The optimizer's KKT certificate; null for the others."
    )
    shrinkage: Optional[ShrinkageReport] = Field(
        None, description="shrunk_sharpe only."
    )
    streams: StreamAlignment = Field(default_factory=StreamAlignment)


#: Why an out-of-sample stream's number can be null.
_FLAT_STREAM = "the stream has no dispersion over the out-of-sample periods"


class OutOfSampleStream(ExplainsNulls):
    """One stream's out-of-sample numbers, reported by the result."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {"sharpe": _FLAT_STREAM}

    sharpe: Stat = Field(None, description="Annual, over cash_rate.")
    annual_return: Stat = Field(None, description="Compound, a year.")
    total_return: Stat = None
    volatility: Stat = Field(None, description="Annual.")
    max_drawdown: Stat = Field(None, description="Negative: -0.2 is 20% off a peak.")
    total_turnover: Stat = Field(
        None, description="Sum of |w - w_drifted| over rebalances, the entry included."
    )
    annual_turnover: Stat = None
    total_cost: Stat = Field(None, description="Sum of each rebalance's cost.")
    n_rebalances: int = 0
    n_unfitted: int = Field(
        0, description="Rebalances the rule could not fit; nothing traded."
    )


class AllocationRuleEvaluation(_Result):
    null_reasons = {
        "sharpe_difference": "a stream has no dispersion, so its Sharpe ratio is "
        "undefined",
        "sharpe_difference_low": "there is no interval: n_bootstrap is 0, or a "
        "Sharpe ratio is undefined on more than half of the resamples",
        "sharpe_difference_high": "there is no interval: n_bootstrap is 0, or a "
        "Sharpe ratio is undefined on more than half of the resamples",
        "difference_p_value": "there is no interval to read it from",
        "look_ahead_gap": "a Sharpe ratio it compares is undefined, or the rule "
        "cannot be fitted on the full sample",
        "deflated_sharpe_probability": "the rule's stream has no deflated Sharpe "
        "ratio (see the warnings)",
        "deflation_threshold": "the rule's stream has no deflated Sharpe ratio",
    }

    method: str = ""
    benchmark_rule: str = ""
    n_sleeves: int = 0
    estimation_window: int = 0
    rebalance_every: int = 0
    n_out_of_sample: int = Field(0, description="Periods scored, each after its fit.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    rule: OutOfSampleStream = Field(default_factory=OutOfSampleStream)
    benchmark: OutOfSampleStream = Field(default_factory=OutOfSampleStream)
    sharpe_difference: Stat = Field(
        None,
        description="Rule minus benchmark: the difference of the Sharpe ratios, "
        "not of the mean returns.",
    )
    sharpe_difference_low: Stat = None
    sharpe_difference_high: Stat = None
    difference_confidence: float = 0.95
    difference_p_value: Stat = Field(
        None, description="Two-sided, from the same bootstrap draws."
    )
    full_sample_fit: Optional[OutOfSampleStream] = Field(
        None,
        description="LOOK-AHEAD: the rule fitted once on every period and held, "
        "scored on the same out-of-sample periods; null when the rule cannot be "
        "fitted on the full sample.",
    )
    look_ahead_gap: Stat = Field(
        None,
        description="full_sample_fit.sharpe - rule.sharpe: how much of an "
        "in-sample result is the fit seeing its own test.",
    )
    full_sample_weights: Optional[Dict[str, Stat]] = None
    deflated_sharpe_probability: Stat = Field(
        None, description="P(true Sharpe > 0) given n_trials, of rule.sharpe."
    )
    deflation_threshold: Stat = Field(
        None, description="The Sharpe ratio the best of n_trials reaches by luck."
    )
    significant_at_95: Optional[bool] = None
    n_trials: int = 1
    final_weights: Dict[str, Stat] = Field(
        default_factory=dict, description="The rule's last chosen weights."
    )
    unfitted_dates: List[str] = Field(default_factory=list)
    oos_returns_ref: Optional[str] = Field(
        None,
        description="returns_panel, columns 'rule' and 'benchmark'; null "
        "without run_id. apply_risk_overlay overlays it.",
    )
    weights_ref: Optional[str] = Field(
        None,
        description="weight_panel of the weights chosen at each rebalance; null "
        "without run_id.",
    )
    n_bootstrap: int = 0
    block_size: int = 0
    bootstrap_scale: Stat = None
    streams: StreamAlignment = Field(default_factory=StreamAlignment)


__all__ = [
    "AllocateSleevesInput",
    "AllocationRule",
    "AllocationRuleEvaluation",
    "BindingCap",
    "CombineStreamsInput",
    "CombineStreamsResult",
    "EvaluateAllocationRuleInput",
    "OutOfSampleStream",
    "ShrinkageReport",
    "SleeveAllocationResult",
    "SleeveReturns",
]
