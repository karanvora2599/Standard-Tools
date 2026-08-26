"""
What the equity curve does not say.

A backtest reports one path, in one order, and most of what is worth knowing
is about the paths that did not happen: how much of the outcome was sequence
rather than edge, whether the losses arrive in runs a real account could not
sit through, and whether the entry decisions beat a coin.

INPUTS ARE TRADE RETURNS, not an equity curve. A trade is the natural unit
of a strategy's decision -- one entry, one exit, one bet -- and the returns
inside it are not independent draws but the anatomy of a single decision.
Reshuffling trades asks "what if the same edge had dealt these bets in a
different order"; reshuffling daily returns asks something closer to "what
if this were a different strategy".
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.backtesting import trade_analysis as lib

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class MonteCarloTradesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_returns: List[float] = Field(
        ...,
        min_length=20,
        description="One return per TRADE, in the order they happened.",
    )
    n_paths: int = Field(2000, ge=100, le=50000)
    seed: int = Field(0)
    starting_equity: float = Field(1.0, gt=0)


class TradeClusteringInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_returns: List[float] = Field(
        ...,
        min_length=20,
        description="One return per trade, IN ORDER. The order is the data "
        "here -- this is the one tool where shuffling the input destroys the "
        "question.",
    )


class CompareRandomInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_returns: List[float] = Field(..., min_length=20)
    n_simulations: int = Field(2000, ge=100, le=50000)
    seed: int = Field(0)


class ExposureAttributionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(
        ..., min_length=30, description="The ASSET's returns, not the strategy's."
    )
    exposure: List[float] = Field(
        ...,
        min_length=30,
        description="Position size per period, parallel to returns. 1.0 is "
        "fully long, 0.0 flat, negative short.",
    )
    periods_per_year: int = Field(252, ge=1)


class MonteCarloTradesResult(_Result):
    n_trades: int = 0
    n_paths: int = 0
    observed_max_drawdown: Stat = None
    observed_final_equity: Stat = None
    observed_drawdown_percentile: Stat = Field(
        None,
        description="Where the BACKTESTED drawdown sits among the orderings "
        "the same trades could have produced.",
    )
    median_max_drawdown: Stat = None
    p05_max_drawdown: Stat = None
    p95_max_drawdown: Stat = Field(
        None, description="The number to size on, not the backtested one."
    )
    worst_max_drawdown: Stat = None
    mean_fraction_underwater: Stat = None
    final_equity: Stat = Field(
        None,
        description="Identical across paths by construction -- a reshuffle "
        "holds the same trades, so only the PATH differs.",
    )


class TradeClusteringResult(_Result):
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: Stat = None
    n_runs: int = 0
    expected_runs: Stat = None
    runs_z_score: Stat = Field(
        None,
        description="Negative means CLUSTERING (streaks), positive means "
        "alternation. Below -1.96 or above +1.96 is significant.",
    )
    p_value: Stat = None
    clustered: bool = False
    alternating: bool = False
    longest_losing_streak: int = 0
    longest_winning_streak: int = 0


class CompareRandomResult(_Result):
    n_trades: int = 0
    n_simulations: int = 0
    win_rate: Stat = None
    observed_per_trade_sharpe: Stat = None
    random_median: Stat = None
    random_p95: Stat = None
    p_value: Stat = None
    beats_random_at_05: bool = False


class ExposureAttributionResult(_Result):
    n_observations: int = 0
    mean_exposure: Stat = None
    fraction_invested: Stat = None
    fraction_long: Stat = None
    total_mean_return: Stat = None
    passive_contribution: Stat = Field(
        None, description="Average exposure times average return: beta."
    )
    timing_contribution: Stat = Field(
        None,
        description="Covariance of exposure with the return: the only part "
        "that is skill. Usually far smaller than expected.",
    )
    timing_share: Stat = None
    annualized_total: Stat = None
    annualized_passive: Stat = None
    annualized_timing: Stat = None
    exposure_return_correlation: Stat = None


def run_monte_carlo_trade_paths(
    input_data: MonteCarloTradesInput,
) -> MonteCarloTradesResult:
    return MonteCarloTradesResult(
        **lib.monte_carlo_trade_paths(
            input_data.trade_returns,
            n_paths=input_data.n_paths,
            seed=input_data.seed,
            starting_equity=input_data.starting_equity,
        )
    )


def analyze_trade_clustering(
    input_data: TradeClusteringInput,
) -> TradeClusteringResult:
    return TradeClusteringResult(
        **lib.analyze_trade_clustering(input_data.trade_returns)
    )


def compare_against_random(input_data: CompareRandomInput) -> CompareRandomResult:
    return CompareRandomResult(
        **lib.compare_against_random(
            input_data.trade_returns,
            n_simulations=input_data.n_simulations,
            seed=input_data.seed,
        )
    )


def get_exposure_attribution(
    input_data: ExposureAttributionInput,
) -> ExposureAttributionResult:
    return ExposureAttributionResult(
        **lib.exposure_attribution(
            input_data.returns,
            input_data.exposure,
            periods_per_year=input_data.periods_per_year,
        )
    )


TRADE_TOOL_DEFS = [
    (
        "run_monte_carlo_trade_paths",
        "The distribution of outcomes the same edge could have produced, by "
        "RESHUFFLING the trades. A strategy with a 20% backtested drawdown "
        "does not have a 20% drawdown -- it has a distribution of them, and "
        "the backtested one is a single draw that routinely sits near the "
        "middle while the 95th percentile is half again as deep. Sizing so "
        "the backtested drawdown is survivable is sizing on the median "
        "outcome, and half of all realizations are worse. Reshuffles rather "
        "than resamples with replacement, so every path holds the same trades "
        "and ends at the same total return -- which isolates SEQUENCE risk "
        "from edge uncertainty instead of mixing the two.",
        MonteCarloTradesInput,
    ),
    (
        "analyze_trade_clustering",
        "Whether wins and losses arrive in RUNS, measured in the original "
        "order. A 55% win rate is survivable if the losses are scattered and "
        "unholdable if they arrive eleven in a row, and the win rate is "
        "identical in both cases. A runs test gives the z-score: negative is "
        "clustering, positive is alternation (rarer, and usually a strategy "
        "reacting to its own last outcome, which is worth checking for a "
        "state-carrying bug). Read it alongside "
        "run_monte_carlo_trade_paths, whose reshuffling DESTROYS exactly the "
        "clustering measured here -- so its drawdown distribution is "
        "optimistic by this much.",
        TradeClusteringInput,
    ),
    (
        "compare_against_random",
        "Whether the strategy beats a coin that traded the same instrument "
        "the same number of times. Comparing against zero or against "
        "buy-and-hold is the usual test and neither is the right one after a "
        "search: a strategy can beat zero purely by holding a rising asset "
        "with no timing skill at all. The null keeps the trade MAGNITUDES and "
        "the win rate and randomizes the signs, so it tests whether the "
        "sequencing and sizing add anything -- not whether the win rate "
        "does. Measured as CONSERVATIVE: it fired on 1 of 150 skill-free "
        "strategies against a nominal 5%, so a non-rejection is weaker "
        "evidence than it looks.",
        CompareRandomInput,
    ),
    (
        "get_exposure_attribution",
        "How much of the return came from being RIGHT versus from being "
        "INVESTED. A strategy's return is exposure times the market's move, "
        "and splitting it shows whether the P&L came from timing -- holding "
        "more before up moves -- or simply from average exposure to an asset "
        "that rose, which is beta and available for nothing. The timing term "
        "is the covariance between exposure and the subsequent return; it is "
        "usually far smaller than expected and often NEGATIVE in strategies "
        "that look profitable. Also reports time in market, because a Sharpe "
        "computed on a 20%-invested strategy ignores what the idle capital "
        "earns and is not comparable with a fully-invested one's.",
        ExposureAttributionInput,
    ),
]

TRADE_TOOL_DISPATCH = {
    "run_monte_carlo_trade_paths": (
        run_monte_carlo_trade_paths,
        MonteCarloTradesInput,
    ),
    "analyze_trade_clustering": (analyze_trade_clustering, TradeClusteringInput),
    "compare_against_random": (compare_against_random, CompareRandomInput),
    "get_exposure_attribution": (
        get_exposure_attribution,
        ExposureAttributionInput,
    ),
}


class BreakEvenCostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_returns: List[float] = Field(
        ...,
        min_length=20,
        description="One return per trade, NET of whatever cost the backtest "
        "already charged.",
    )
    current_cost_bps: float = Field(
        0.0,
        ge=0,
        description="The per-trade cost the backtest assumed, in basis "
        "points. Supplying it is what turns the break-even into a headroom "
        "multiple.",
    )


class CostSensitivity(BaseModel):
    model_config = ConfigDict(extra="allow")

    cost_bps: Stat = None
    mean_return: Stat = None
    per_trade_sharpe: Stat = None
    profitable: bool = False


class BreakEvenCostResult(_Result):
    n_trades: int = 0
    current_cost_bps: Stat = None
    mean_return_net: Stat = None
    mean_return_gross: Stat = None
    break_even_cost_bps: Stat = Field(
        None, description="The per-trade cost at which the edge disappears."
    )
    headroom_multiple: Stat = Field(
        None,
        description="Break-even over the assumed cost. Under about 2x, the "
        "backtest is a statement about the cost assumption.",
    )
    sensitivity: List[CostSensitivity] = Field(default_factory=list)


def estimate_break_even_cost(
    input_data: BreakEvenCostInput,
) -> BreakEvenCostResult:
    return BreakEvenCostResult(
        **lib.break_even_cost(
            input_data.trade_returns,
            current_cost_bps=input_data.current_cost_bps,
        )
    )


TRADE_TOOL_DEFS.append(
    (
        "estimate_break_even_cost",
        "The per-trade cost at which this edge disappears -- the number every "
        "backtest should report and almost none do. What decides whether a "
        "result survives contact with a real broker is not whether it is "
        "profitable at the assumed cost but how far above it the break-even "
        "sits. A strategy breaking even at 8bp when you modelled 5bp has 1.6x "
        "of headroom, and one bad fill or a widening spread eats it; one "
        "breaking even at 80bp is robust to both. Under about 2x, the "
        "backtest is a statement about the cost ASSUMPTION rather than about "
        "the strategy. Models a FLAT charge and not impact, so a strategy "
        "with headroom here can still fail on capacity.",
        BreakEvenCostInput,
    )
)

TRADE_TOOL_DISPATCH["estimate_break_even_cost"] = (
    estimate_break_even_cost,
    BreakEvenCostInput,
)


__all__ = [
    "estimate_break_even_cost",
    "TRADE_TOOL_DEFS",
    "TRADE_TOOL_DISPATCH",
    "analyze_trade_clustering",
    "compare_against_random",
    "get_exposure_attribution",
    "run_monte_carlo_trade_paths",
]
