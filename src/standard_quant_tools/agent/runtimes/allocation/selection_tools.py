"""
The rule-selection tool: choosing among allocation rules as you go, and
testing the choosing out of sample.

- `select_allocation_rule` -- `allocation.selection.select_allocation_rule`:
  every candidate rule walked forward; at each rebalance a selector picks
  by the candidates' own out-of-sample record before that date, with a
  switch margin and an optional top-k blend, and trades to it at a cost.
  Scored against equal weight with the paired Sharpe difference, beside
  the HINDSIGHT best candidate, the probability of backtest overfitting
  over the candidates and the deflated Sharpe ratios at the caller's
  trial count.

It takes the sleeves as the other allocating tools do -- `returns` or a
`returns_ref` panel -- and fetches nothing. A candidate is an
`AllocationRule`, spelled as `evaluate_allocation_rule`'s `rule` is, plus
the name it is reported by.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.streams import StreamAlignment
from standard_quant_tools.allocation import selection as selection_lib
from standard_quant_tools.allocation.sleeves import SleeveRule

from .sleeve_models import (
    _DEFAULT_HALFLIFE,
    _INPUT,
    AllocationRule,
    CovarianceMethod,
    Method,
    OutOfSampleStream,
    RegimeInput,
    SleeveReturns,
    Stat,
    _compact_nested,
    _Conventions,
    _finite_positive,
    _Result,
)
from .sleeve_tools import MIN_SHARED_DATES, _date, _sleeve_frame

# ── inputs ──────────────────────────────────────────────────────────────


class CandidateRule(BaseModel):
    """
    A candidate rule and the name its stream and choices are reported by.

    `AllocationRule`'s fields but the three that are about one book's
    capital -- `capital`, `capacity`, `capacity_excess`. The selector's book
    follows each candidate's weights, which a candidate fits at its own
    walk's capital, not the selector's: a capacity in currency would cap
    them at the wrong amount. Cap a candidate through max_weight instead.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_nested)

    name: Optional[str] = Field(
        None, min_length=1, max_length=64, description="Default: its method, numbered."
    )
    method: Method = "risk_parity"
    covariance_method: CovarianceMethod = "ledoit_wolf"
    halflife: float = Field(
        _DEFAULT_HALFLIFE, gt=0, le=1_000_000, description="ewma's, periods."
    )
    risk_budgets: Optional[Dict[str, float]] = Field(
        None, description="risk_parity: sleeve -> risk share."
    )
    long_only: bool = True
    max_weight: Optional[float] = Field(None, gt=0, le=1)
    target_volatility: Optional[float] = Field(None, gt=0, le=10)
    max_leverage: float = Field(1.0, gt=0, le=10)
    regime: Optional[RegimeInput] = None

    @field_validator("risk_budgets")
    @classmethod
    def _finite_budgets(cls, v: Optional[Dict[str, float]]) -> Optional[Dict]:
        return _finite_positive(v, "risk_budgets")

    @model_validator(mode="after")
    def _a_rule(self) -> "CandidateRule":
        # AllocationRule's own checks, by building one.
        AllocationRule(**self.rule_fields())
        return self

    def rule_fields(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in CandidateRule.model_fields
            if name != "name"
        }

    def to_rule(self) -> SleeveRule:
        return AllocationRule(**self.rule_fields()).to_rule()


class SelectAllocationRuleInput(SleeveReturns, _Conventions):
    model_config = _INPUT
    tool: ClassVar[str] = "select_allocation_rule"

    candidates: List[CandidateRule] = Field(
        ...,
        min_length=2,
        max_length=selection_lib.MAX_CANDIDATES,
        description="The rules to choose among.",
    )
    n_trials: int = Field(
        ..., ge=1, le=1_000_000, description="Selections tried, this one too."
    )
    estimation_window: int = Field(252, ge=2, le=1_000_000)
    rebalance_every: int = Field(21, ge=1, le=1_000_000)
    cost_bps: float = Field(0.0, ge=0, le=10_000, description="Per unit of turnover.")
    selection_window: int = Field(
        selection_lib.DEFAULT_SELECTION_WINDOW,
        ge=2,
        le=1_000_000,
        description="Periods of each rule's record scored.",
    )
    switch_margin: float = Field(
        selection_lib.DEFAULT_SWITCH_MARGIN,
        ge=0,
        le=10,
        description="Sharpe a challenger must win by.",
    )
    combine: Literal["best", "top_k"] = selection_lib.DEFAULT_COMBINE
    top_k: Optional[int] = Field(None, ge=1, le=11, description="Default: half.")
    n_bootstrap: int = Field(1000, ge=0, le=20_000)
    run_id: Optional[str] = Field(None, description="Publishes the streams, weights.")

    @model_validator(mode="after")
    def _a_selection(self) -> "SelectAllocationRuleInput":
        names = self.names()
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"candidate names {repeated[:3]} are repeated; each candidate's "
                "stream and choices are reported by name. Give each its own, "
                "or leave names out: a name left out is the rule's method, "
                "numbered when the method repeats (risk_parity, risk_parity_2)."
            )
        reserved = [n for n in names if n in selection_lib.RESERVED_NAMES]
        if reserved:
            raise ValueError(
                f"candidate name {reserved[0]!r} is reserved for the "
                f"{reserved[0]}'s own published stream. Rename the candidate."
            )
        if self.combine == "best" and self.top_k is not None:
            raise ValueError(
                f"top_k={self.top_k} is read by combine='top_k' only, and "
                "combine='best'. Drop top_k, or use combine='top_k'."
            )
        if self.top_k is not None and self.top_k >= len(self.candidates):
            raise ValueError(
                f"top_k={self.top_k} with {len(self.candidates)} candidates "
                "blends them all at every rebalance, which selects nothing. "
                f"Use a top_k below {len(self.candidates)}, or leave it out "
                "for the better half."
            )
        return self

    def names(self) -> List[str]:
        """Each candidate's name: its own, or its method, numbered from the
        second rule of that method on."""
        out: List[str] = []
        seen: Dict[str, int] = {}
        for c in self.candidates:
            if c.name is not None:
                out.append(c.name)
                continue
            seen[c.method] = seen.get(c.method, 0) + 1
            count = seen[c.method]
            out.append(c.method if count == 1 else f"{c.method}_{count}")
        return out


# ── result ──────────────────────────────────────────────────────────────

#: Why a deflated number can be null.
_NO_DSR = "that stream has no deflated Sharpe ratio (see the warnings)"


class RuleSelectionResult(_Result):
    null_reasons = {
        "sharpe_difference": "a stream has no dispersion, so its Sharpe ratio is "
        "undefined",
        "sharpe_difference_low": "there is no interval: n_bootstrap is 0, or a "
        "Sharpe ratio is undefined on more than half of the resamples",
        "sharpe_difference_high": "there is no interval: n_bootstrap is 0, or a "
        "Sharpe ratio is undefined on more than half of the resamples",
        "difference_p_value": "there is no interval to read it from",
        "hindsight_gap": "no candidate, or not the selector, has a defined Sharpe "
        "ratio over the scored periods",
        "pbo": "the candidates' streams are too short or too flat for the "
        "combinatorial split (see the warnings)",
        "pbo_median_correlation": "there is no PBO",
        "deflated_sharpe_probability": _NO_DSR,
        "deflation_threshold": _NO_DSR,
        "hindsight_deflated_probability": _NO_DSR,
        "hindsight_deflation_threshold": _NO_DSR,
    }

    candidates_tried: List[str] = Field(
        default_factory=list, description="The candidates' names, in order."
    )
    n_sleeves: int = 0
    estimation_window: int = 0
    rebalance_every: int = 0
    selection_window: int = 0
    switch_margin: float = 0.0
    combine: str = "best"
    top_k: Optional[int] = Field(None, description="With combine='top_k' only.")
    n_out_of_sample: int = Field(
        0, description="Periods scored: from the first rebalance that chose."
    )
    n_warm_up: int = Field(
        0,
        description="Periods the selector held the first candidate before any "
        "had selection_window periods of record; not scored.",
    )
    first_scored_date: Optional[str] = None
    last_date: Optional[str] = None
    selector: OutOfSampleStream = Field(
        default_factory=OutOfSampleStream,
        description="The selector's own book, net of its switching costs.",
    )
    benchmark: OutOfSampleStream = Field(
        default_factory=OutOfSampleStream,
        description="Equal weight walked forward the same way.",
    )
    candidates: Dict[str, OutOfSampleStream] = Field(
        default_factory=dict, description="Each candidate's own walk, scored alike."
    )
    times_chosen: Dict[str, int] = Field(
        default_factory=dict,
        description="Scored rebalances at which each candidate was held.",
    )
    n_selections: int = 0
    n_switches: int = Field(
        0, description="Scored rebalances that changed what was held."
    )
    last_choice: List[str] = Field(default_factory=list)
    sharpe_difference: Stat = Field(
        None, description="Selector minus equal weight, in Sharpe ratio."
    )
    sharpe_difference_low: Stat = None
    sharpe_difference_high: Stat = None
    difference_confidence: float = 0.95
    difference_p_value: Stat = Field(
        None, description="Two-sided, from the same bootstrap draws."
    )
    hindsight_best: Optional[str] = Field(
        None,
        description="HINDSIGHT: the candidate with the best Sharpe ratio over the "
        "scored periods, which no one could have known to hold.",
    )
    hindsight_gap: Stat = Field(
        None,
        description="HINDSIGHT: hindsight_best's Sharpe minus the selector's -- "
        "what a backtest that picked the winner afterwards adds.",
    )
    pbo: Stat = Field(
        None,
        description="Probability of backtest overfitting over the candidates: "
        "near 0.5 or above, choosing by record is no better than at random.",
    )
    pbo_median_correlation: Stat = Field(
        None, description="Above about 0.95 the candidates are one rule."
    )
    deflated_sharpe_probability: Stat = Field(
        None, description="P(true Sharpe > 0) of the selector, given n_trials."
    )
    deflation_threshold: Stat = None
    significant_at_95: Optional[bool] = None
    hindsight_deflated_probability: Stat = Field(
        None,
        description="hindsight_best's, at n_trials x the number of candidates.",
    )
    hindsight_deflation_threshold: Stat = None
    n_trials: int = 1
    oos_returns_ref: Optional[str] = Field(
        None,
        description="returns_panel: 'selector', 'benchmark' and each candidate, "
        "warm-up included; null without run_id.",
    )
    weights_ref: Optional[str] = Field(
        None,
        description="weight_panel of what the selector traded to at each "
        "rebalance; null without run_id.",
    )
    n_bootstrap: int = 0
    block_size: int = 0
    bootstrap_scale: Stat = None
    streams: StreamAlignment = Field(default_factory=StreamAlignment)


# ── the tool ────────────────────────────────────────────────────────────


def select_allocation_rule(
    input_data: SelectAllocationRuleInput,
) -> RuleSelectionResult:
    who = "select_allocation_rule"
    frame, alignment, notes = _sleeve_frame(input_data, who, MIN_SHARED_DATES)
    names = input_data.names()
    out = selection_lib.select_allocation_rule(
        frame,
        [(name, c.to_rule()) for name, c in zip(names, input_data.candidates)],
        n_trials=input_data.n_trials,
        estimation_window=input_data.estimation_window,
        rebalance_every=input_data.rebalance_every,
        cost_bps=input_data.cost_bps,
        cash_rate=input_data.cash_rate,
        periods_per_year=input_data.periods_per_year,
        selection_window=input_data.selection_window,
        switch_margin=input_data.switch_margin,
        combine=input_data.combine,
        top_k=input_data.top_k,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        random_seed=input_data.random_seed,
        what=who,
    )
    warnings = notes + out["warnings"]
    returns_ref = weights_ref = None
    if input_data.run_id is not None:
        streams = out["streams"].copy()
        streams.index.name = "date"
        returns_ref = handoff.publish(
            streams,
            kind="returns_panel",
            run_id=input_data.run_id,
            name="selection_returns",
            producer=who,
        )
        weights_ref = handoff.publish(
            out["targets"],
            kind="weight_panel",
            run_id=input_data.run_id,
            name="selection_weights",
            producer=who,
        )
    difference = out["sharpe_difference"]
    deflated: Dict[str, Any] = out["deflated"]
    hindsight: Dict[str, Any] = out["hindsight_deflated"]
    pbo: Optional[Dict[str, Any]] = out["pbo"]
    nan = float("nan")
    dates = out["dates"]
    return RuleSelectionResult(
        candidates_tried=out["names"],
        n_sleeves=int(out["targets"].shape[1]),
        estimation_window=input_data.estimation_window,
        rebalance_every=input_data.rebalance_every,
        selection_window=out["selection_window"],
        switch_margin=out["switch_margin"],
        combine=out["combine"],
        top_k=out["top_k"],
        n_out_of_sample=out["n_out_of_sample"],
        n_warm_up=out["n_warm_up"],
        first_scored_date=_date(dates[0]),
        last_date=_date(dates[-1]),
        selector=OutOfSampleStream(**out["selector"]),
        benchmark=OutOfSampleStream(**out["benchmark"]),
        candidates={
            name: OutOfSampleStream(**summary)
            for name, summary in out["candidates"].items()
        },
        times_chosen=out["times_chosen"],
        n_selections=out["n_selections"],
        n_switches=out["n_switches"],
        last_choice=out["last_choice"],
        sharpe_difference=difference["observed"],
        sharpe_difference_low=difference["lower"],
        sharpe_difference_high=difference["upper"],
        difference_confidence=out["confidence"],
        difference_p_value=difference["p_value"],
        hindsight_best=out["hindsight_best"],
        hindsight_gap=out["hindsight_gap"],
        pbo=nan if pbo is None else pbo["pbo"],
        pbo_median_correlation=(
            nan if pbo is None else pbo["median_configuration_correlation"]
        ),
        deflated_sharpe_probability=deflated.get("deflated_sharpe_probability", nan),
        deflation_threshold=deflated.get("deflation_threshold", nan),
        significant_at_95=deflated.get("significant_at_95"),
        hindsight_deflated_probability=hindsight.get(
            "deflated_sharpe_probability", nan
        ),
        hindsight_deflation_threshold=hindsight.get("deflation_threshold", nan),
        n_trials=out["n_trials"],
        oos_returns_ref=returns_ref,
        weights_ref=weights_ref,
        n_bootstrap=out["n_bootstrap"],
        block_size=out["block_size"],
        bootstrap_scale=difference["scale"],
        streams=alignment,
        warnings=warnings,
    )


TOOL_DEFS = [
    (
        "select_allocation_rule",
        "Chooses among 2-12 allocation rules as it goes, at each rebalance by "
        "their own out-of-sample record before that date, and tests that "
        "choosing against equal weight out of sample. Holds an equal blend of "
        "the better half by trailing Sharpe (or the best one), switching only "
        "past switch_margin and paying for it. Reports the Sharpe difference "
        "with its interval, how often each rule was chosen, the HINDSIGHT "
        "best and its gap, PBO and deflated Sharpes at n_trials.",
        SelectAllocationRuleInput,
    ),
]

TOOL_DISPATCH = {
    "select_allocation_rule": (select_allocation_rule, SelectAllocationRuleInput),
}

__all__ = [
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "select_allocation_rule",
]
