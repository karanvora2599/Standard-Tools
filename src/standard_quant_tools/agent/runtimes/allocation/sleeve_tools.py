"""
The sleeve tools: return streams on one calendar, capital across them, and
the walk-forward test of the rule that split it.

- `combine_return_streams` -- `portfolio.streams.combine_return_streams` as
  a tool: the streams on the dates they all share, published as one
  `returns_panel`, with what each stream lost.
- `allocate_across_sleeves` -- `allocation.sleeves.allocate_across_sleeves`:
  weights, risk, caps and block-bootstrap weight bands.
- `evaluate_allocation_rule` -- `allocation.walk_forward.
  evaluate_allocation_rule`: the rule out of sample against a benchmark
  rule and against itself fitted on the full sample, with the deflated
  Sharpe ratio at the caller's trial count. It takes no overlay: a
  volatility target or CPPI on the published out-of-sample stream is
  `apply_risk_overlay`, and since those overlays are causal on the stream
  the two orders are the same thing.

The two allocating tools take `returns` ({stream: {ISO date: return}}) or
a `returns_ref` panel and fetch nothing, so the same arguments give the
same bits. `combine_return_streams` is the one door that may fetch: a
stream there can also be a symbol over a window the caller names, and what
it publishes is the panel the others read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.streams import (
    StreamAlignment,
    aligned_dated_streams,
    aligned_returns_ref,
    returns_ref_streams,
)
from standard_quant_tools.allocation import sleeves as sleeves_lib
from standard_quant_tools.allocation import walk_forward as walk_forward_lib
from standard_quant_tools.portfolio import streams as streams_lib

from .sleeve_models import (
    AllocateSleevesInput,
    AllocationRuleEvaluation,
    BindingCap,
    CombineStreamsInput,
    CombineStreamsResult,
    EvaluateAllocationRuleInput,
    OutOfSampleStream,
    RegimeReport,
    RegimeWalkSummary,
    ShrinkageReport,
    SleeveAllocationResult,
    SleeveReturns,
)

#: The fewest shared dates an allocation is fitted on, the walk-forward's
#: out-of-sample minimum: fewer, and a covariance of the sleeves is noise.
MIN_SHARED_DATES = walk_forward_lib.MIN_OUT_OF_SAMPLE


def _sleeve_frame(
    data: SleeveReturns, who: str, min_overlap: int
) -> Tuple[pd.DataFrame, StreamAlignment, List[str]]:
    """The sleeves on the dates they share, the report, and what it cost."""
    if data.returns_ref is not None:
        return aligned_returns_ref(
            data.returns_ref,
            what=who,
            min_overlap=min_overlap,
            columns=data.columns,
        )
    assert data.returns is not None  # one of the two, by the model
    return aligned_dated_streams(data.returns, what=who, min_overlap=min_overlap)


def combine_return_streams(input_data: CombineStreamsInput) -> CombineStreamsResult:
    who = "combine_return_streams"
    source: Dict[str, Any]
    if input_data.returns_ref is not None:
        source = returns_ref_streams(
            input_data.returns_ref, what=who, columns=input_data.columns
        )
    else:
        assert input_data.streams is not None
        source = dict(input_data.streams)
    out = streams_lib.combine_return_streams(
        source, run_id=input_data.run_id, min_overlap=input_data.min_overlap, what=who
    )
    report = out["alignment"]
    return CombineStreamsResult(
        returns_ref=out["returns_ref"],
        columns=out["columns"],
        n_dates=report["n_dates"],
        first_date=report["first_date"],
        last_date=report["last_date"],
        alignment=StreamAlignment(**report),
        warnings=out["warnings"],
    )


def allocate_across_sleeves(input_data: AllocateSleevesInput) -> SleeveAllocationResult:
    who = "allocate_across_sleeves"
    frame, alignment, notes = _sleeve_frame(input_data, who, MIN_SHARED_DATES)
    out = sleeves_lib.allocate_across_sleeves(
        frame,
        input_data.to_rule(),
        cash_rate=input_data.cash_rate,
        periods_per_year=input_data.periods_per_year,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        random_seed=input_data.random_seed,
        what=who,
    )
    shrinkage = out.pop("shrinkage")
    binding = out.pop("binding_caps")
    regime = out.pop("regime")
    out["warnings"] = notes + out["warnings"]
    return SleeveAllocationResult(
        **out,
        binding_caps=[BindingCap(**b) for b in binding],
        shrinkage=None if shrinkage is None else ShrinkageReport(**shrinkage),
        regime=None if regime is None else RegimeReport(**regime),
        streams=alignment,
    )


def _date(stamp: Any) -> str:
    stamp = pd.Timestamp(stamp)
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


def evaluate_allocation_rule(
    input_data: EvaluateAllocationRuleInput,
) -> AllocationRuleEvaluation:
    who = "evaluate_allocation_rule"
    # Aligned at the shared minimum; the walk refuses a window that leaves
    # too little out of sample, naming the window.
    frame, alignment, notes = _sleeve_frame(input_data, who, MIN_SHARED_DATES)
    out = walk_forward_lib.evaluate_allocation_rule(
        frame,
        input_data.rule.to_rule(),
        n_trials=input_data.n_trials,
        benchmark_rule=input_data.benchmark_rule,
        estimation_window=input_data.estimation_window,
        rebalance_every=input_data.rebalance_every,
        cost_bps=input_data.cost_bps,
        cash_rate=input_data.cash_rate,
        periods_per_year=input_data.periods_per_year,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        random_seed=input_data.random_seed,
        what=who,
    )
    warnings = notes + out["warnings"]
    returns_ref = weights_ref = None
    if input_data.run_id is not None:
        streams = pd.DataFrame(
            {"rule": out["rule_returns"], "benchmark": out["benchmark_returns"]},
            index=out["dates"],
        )
        streams.index.name = "date"
        returns_ref = handoff.publish(
            streams,
            kind="returns_panel",
            run_id=input_data.run_id,
            name="oos_returns",
            producer=who,
        )
        if len(out["targets"]):
            weights_ref = handoff.publish(
                out["targets"],
                kind="weight_panel",
                run_id=input_data.run_id,
                name="oos_weights",
                producer=who,
            )
        else:
            warnings.append(
                f"{who}: the rule chose no weights at any rebalance, so no "
                "weight_panel was published."
            )
    difference = out["sharpe_difference"]
    deflated = out["deflated"]
    targets = out["targets"]
    dates = out["dates"]
    return AllocationRuleEvaluation(
        method=input_data.rule.method,
        benchmark_rule=input_data.benchmark_rule,
        n_sleeves=int(frame.shape[1]),
        estimation_window=input_data.estimation_window,
        rebalance_every=input_data.rebalance_every,
        n_out_of_sample=out["n_out_of_sample"],
        first_date=_date(dates[0]),
        last_date=_date(dates[-1]),
        rule=OutOfSampleStream(**out["rule"]),
        benchmark=OutOfSampleStream(**out["benchmark"]),
        sharpe_difference=difference["observed"],
        sharpe_difference_low=difference["lower"],
        sharpe_difference_high=difference["upper"],
        difference_confidence=out["confidence"],
        difference_p_value=difference["p_value"],
        full_sample_fit=(
            None
            if out["full_sample"] is None
            else OutOfSampleStream(**out["full_sample"])
        ),
        look_ahead_gap=out["look_ahead_gap"],
        full_sample_weights=out["full_sample_weights"],
        deflated_sharpe_probability=deflated.get(
            "deflated_sharpe_probability", float("nan")
        ),
        deflation_threshold=deflated.get("deflation_threshold", float("nan")),
        significant_at_95=deflated.get("significant_at_95"),
        n_trials=out["n_trials"],
        final_weights=(
            {str(c): float(v) for c, v in targets.iloc[-1].items()}
            if len(targets)
            else {}
        ),
        unfitted_dates=[d for d, _why in out["unfitted"]],
        oos_returns_ref=returns_ref,
        weights_ref=weights_ref,
        n_bootstrap=out["n_bootstrap"],
        block_size=out["block_size"],
        bootstrap_scale=difference["scale"],
        regime=None if out["regime"] is None else RegimeWalkSummary(**out["regime"]),
        streams=alignment,
        warnings=warnings,
    )


TOOL_DEFS = [
    (
        "combine_return_streams",
        "Puts several return streams on the dates they all share and publishes "
        "the aligned panel as one returns_panel reference, with what each stream "
        "lost to the common calendar. Each stream is a returns_panel column, "
        "{ISO date: return} values, or a symbol with both ends of its window. "
        "Refused by name: undated lists, fewer shared dates than min_overlap "
        "(naming the stream that limits it), a repeated date, a non-finite "
        "value, and levels passed as returns.",
        CombineStreamsInput,
    ),
    (
        "allocate_across_sleeves",
        "Splits capital across strategies, sleeves or models by their return "
        "streams: equal, inverse volatility, risk parity, HRP, min variance, max "
        "Sharpe or shrunk Sharpe. Long-only by default, with a max weight, "
        "per-sleeve capacity in currency (the excess to cash or to the others) "
        "and a target volatility refused above max_leverage. Reports risk "
        "contributions, ex-ante volatility, effective N, the binding caps, "
        "block-bootstrap 5th/95th percentile weight bands and their stability. "
        "Its Sharpe ratio is IN-SAMPLE.",
        AllocateSleevesInput,
    ),
    (
        "evaluate_allocation_rule",
        "Walks an allocation rule forward over sleeve return streams -- fit on "
        "a trailing window, drift, rebalance, pay the cost -- and scores it out "
        "of sample against a benchmark rule. Reports each side's Sharpe, return, "
        "volatility and drawdown, the Sharpe difference with a block-bootstrap "
        "interval, the look-ahead gap to the rule fitted on the full sample, "
        "the deflated Sharpe at n_trials, turnover and cost; run_id publishes "
        "the streams and weights. No overlay: apply_risk_overlay on that "
        "stream is the same, being causal.",
        EvaluateAllocationRuleInput,
    ),
]

TOOL_DISPATCH = {
    "combine_return_streams": (combine_return_streams, CombineStreamsInput),
    "allocate_across_sleeves": (allocate_across_sleeves, AllocateSleevesInput),
    "evaluate_allocation_rule": (
        evaluate_allocation_rule,
        EvaluateAllocationRuleInput,
    ),
}

__all__ = [
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "allocate_across_sleeves",
    "combine_return_streams",
    "evaluate_allocation_rule",
]
