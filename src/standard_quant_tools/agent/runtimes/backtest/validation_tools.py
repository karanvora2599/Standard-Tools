"""
How much of a backtest is real: the multiple-testing tools.

Every tool here answers a question the backtest itself cannot. A backtest
reports what a strategy did on a sample. It cannot report how many other
strategies you tried before keeping this one, and that number changes what
the result means more than almost anything inside the backtest does.

INPUTS ARE INLINE. Returns come in as a list of numbers, a trial grid as a
map of configuration name to its return series. Nothing here re-runs a
backtest -- these consume results the caller already has, from this
library's backtest tools or from anywhere else. That is deliberate: the
trials you need to declare include the ones you ran last week and the ones
you ran in a notebook, and no tool can enumerate those for you.

THE MODELS FORBID EXTRA FIELDS. A dropped `n_trials` would silently deflate
for one trial and report a strategy as significant that is not.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.backtesting import overfitting as lib
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(
        default_factory=list,
        description="The conditions under which the numbers above are wrong.",
    )


# ── inputs ──────────────────────────────────────────────────────────────


class DeflatedSharpeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(
        ..., min_length=20, description="Periodic returns as decimals."
    )
    n_trials: int = Field(
        ...,
        ge=1,
        description="How many configurations you tried before keeping this "
        "one. COUNT THE DISCARDED ONES -- the parameter adjusted before the "
        "file was saved is a trial too, and the uncounted trials are usually "
        "the larger number.",
    )
    trial_sharpes: Optional[List[float]] = Field(
        None,
        description="The Sharpe of every variant tried. Their VARIANCE sets "
        "the deflation threshold and is far more informative than the count: "
        "100 near-identical parameter settings deflate much less than 100 "
        "genuinely different ideas.",
    )
    benchmark_sharpe: float = Field(
        0.0, description="A Sharpe the strategy must also beat."
    )
    periods_per_year: int = Field(
        252, ge=1, description="252 for daily, 52 weekly, 12 monthly."
    )


class PBOInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_returns: Dict[str, List[float]] = Field(
        ...,
        description="Configuration name -> its return series. Every series "
        "must be the same length and aligned in time.",
    )
    n_splits: int = Field(
        8,
        ge=4,
        le=16,
        description="Chunks the period is cut into. Must be EVEN -- the "
        "method forms every equal split of the chunks into two halves, so "
        "the count grows as C(n, n/2) and 16 is already 12,870 splits.",
    )


class PurgedCVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_observations: int = Field(
        ..., ge=50, description="Length of the series to be split."
    )
    n_splits: int = Field(6, ge=2, le=20, description="Groups to cut it into.")
    n_test_splits: int = Field(
        2,
        ge=1,
        description="Groups held out per path. Fewer than n_splits. The "
        "number of paths is C(n_splits, n_test_splits).",
    )
    embargo_pct: float = Field(
        0.01,
        ge=0,
        le=0.5,
        description="Fraction of the sample dropped AFTER each test block. "
        "Purging removes the label overlap; the embargo removes the feature "
        "overlap that serial correlation creates.",
    )
    label_horizon: int = Field(
        1,
        ge=1,
        description="How many observations forward the label looks. A 5-day "
        "forward return is 5. This is what purging needs to know.",
    )


class RealityCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_returns: List[float] = Field(..., min_length=30)
    benchmark_returns: Dict[str, List[float]] = Field(
        ...,
        description="The ALTERNATIVES you considered, name -> return series. "
        "The test asks whether this strategy beats the best of these by more "
        "than luck would produce.",
    )
    n_bootstrap: int = Field(1000, ge=100, le=20000)
    block_size: int = Field(
        20,
        ge=1,
        description="Consecutive days per bootstrap block. Blocks preserve "
        "serial correlation; resampling single days makes the null too "
        "narrow and the p-value too small.",
    )
    seed: int = Field(0)


class RegimeStratifiedInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(..., min_length=30)
    regimes: List[str] = Field(
        ...,
        description="A regime label per observation, same length as returns. "
        "Volatility buckets, a bull/bear flag, VIX terciles, calendar years "
        "-- the tool does not care what they mean.",
    )
    periods_per_year: int = Field(252, ge=1)


class ParameterDecayInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter_values: List[float] = Field(..., min_length=5)
    performance: List[float] = Field(
        ..., min_length=5, description="One score per parameter value."
    )
    metric_name: str = Field("sharpe", description="What `performance` is.")


# ── results ─────────────────────────────────────────────────────────────


class DeflatedSharpeResult(_Result):
    n_observations: int = 0
    observed_sharpe: Stat = None
    n_trials: int = 1
    expected_max_sharpe_from_luck: Stat = Field(
        None,
        description="The Sharpe this many trials would produce with no edge "
        "at all. 20 no-edge strategies on two years of daily data give 1.34.",
    )
    deflation_threshold: Stat = None
    deflated_sharpe_probability: Stat = Field(
        None, description="Probability the edge is real. The bar is 0.95."
    )
    significant_at_95: bool = False
    skewness: Stat = None
    kurtosis: Stat = None
    sharpe_standard_error: Stat = None


class PBOResult(_Result):
    n_configurations: int = 0
    n_observations: int = 0
    n_splits: int = 0
    n_combinations: int = 0
    pbo: Stat = Field(
        None,
        description="How often the in-sample winner lands in the bottom half "
        "out of sample. 0.5 means selection has NO skill. It is a property "
        "of the procedure, not of the strategy.",
    )
    median_logit: Stat = None
    median_out_of_sample_rank: Stat = None
    median_configuration_correlation: Stat = Field(
        None,
        description="Above ~0.95 these are one strategy with a parameter "
        "nudged, and the PBO is close to meaningless.",
    )


class CVPath(BaseModel):
    model_config = ConfigDict(extra="allow")

    test_groups: List[int] = Field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_purged: int = 0
    train_ranges: List[List[int]] = Field(
        default_factory=list,
        description="Half-open [start, end) ranges rather than every index. "
        "Purging removes contiguous blocks, so the training set is a union "
        "of a handful of ranges -- and 500 observations across 15 paths is "
        "37 KB as indices against under 2 KB as ranges, with the same "
        "information. Expand with range(start, end).",
    )
    test_ranges: List[List[int]] = Field(default_factory=list)


class PurgedCVResult(_Result):
    n_observations: int = 0
    n_splits: int = 0
    n_test_splits: int = 0
    n_paths: int = 0
    embargo_observations: int = 0
    label_horizon: int = 1
    mean_train_size: Stat = None
    mean_purged: Stat = None
    paths: List[CVPath] = Field(default_factory=list)


class RealityCheckResult(_Result):
    n_observations: int = 0
    n_benchmarks: int = 0
    n_bootstrap: int = 0
    block_size: int = 0
    observed_outperformance: Stat = None
    p_value: Stat = None
    significant_at_05: bool = False
    bootstrap_p95: Stat = None


class RegimeRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    regime: str = ""
    n_observations: int = 0
    share_of_sample: Stat = None
    total_return: Stat = None
    share_of_pnl: Stat = None
    mean_return: Stat = None
    annualized_return: Stat = None
    volatility: Stat = None
    sharpe: Stat = None
    win_rate: Stat = None
    worst: Stat = None


class RegimeStratifiedResult(_Result):
    n_observations: int = 0
    n_regimes: int = 0
    overall_sharpe: Stat = None
    pnl_concentration: Stat = Field(
        None,
        description="Share of total P&L from the single best regime. Above "
        "0.7 the strategy is a bet on that regime recurring.",
    )
    n_profitable_regimes: int = 0
    by_regime: List[RegimeRow] = Field(default_factory=list)


class SurfacePoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    parameter: Stat = None
    performance: Stat = None


class ParameterDecayResult(_Result):
    n_points: int = 0
    metric: str = ""
    best_parameter: Stat = None
    best_performance: Stat = None
    neighbour_mean: Stat = None
    spike_ratio: Stat = Field(
        None,
        description="Neighbours' performance as a fraction of the peak. Near "
        "1 is a broad optimum -- a real effect. Below 0.5 is a spike, which "
        "in a noisy objective is almost always the one setting that fitted "
        "this sample.",
    )
    plateau_fraction: Stat = None
    performance_dispersion: Stat = None
    best_at_grid_edge: bool = False
    surface: List[SurfacePoint] = Field(default_factory=list)


# ── tools ───────────────────────────────────────────────────────────────


def _aligned_frame(mapping: Dict[str, List[float]], who: str) -> pd.DataFrame:
    """
    Equal-length series into a frame, refusing a ragged one by name.

    Pandas would pad the short series with NaN and carry on, producing a
    result computed over a shorter effective sample than the caller thinks --
    which is exactly the kind of quiet degradation these tools exist to
    catch elsewhere.
    """
    if not mapping:
        raise ValidationError(f"{who}: no series given.")
    lengths = {name: len(values) for name, values in mapping.items()}
    if len(set(lengths.values())) > 1:
        raise ValidationError(
            f"{who}: the series have different lengths ({lengths}). They must "
            "be aligned in time -- padding the short ones would compute the "
            "result over a shorter sample than you think."
        )
    return pd.DataFrame(mapping)


def get_deflated_sharpe_ratio(
    input_data: DeflatedSharpeInput,
) -> DeflatedSharpeResult:
    return DeflatedSharpeResult(
        **lib.deflated_sharpe_ratio(
            pd.Series(input_data.returns),
            n_trials=input_data.n_trials,
            trial_sharpes=input_data.trial_sharpes,
            benchmark_sharpe=input_data.benchmark_sharpe,
            periods_per_year=input_data.periods_per_year,
        )
    )


def estimate_backtest_overfitting(input_data: PBOInput) -> PBOResult:
    return PBOResult(
        **lib.probability_of_backtest_overfitting(
            _aligned_frame(input_data.trial_returns, "estimate_backtest_overfitting"),
            n_splits=input_data.n_splits,
        )
    )


def _as_ranges(indices: List[int]) -> List[List[int]]:
    """
    Contiguous runs as half-open [start, end) pairs.

    Purging and the embargo both remove CONTIGUOUS blocks, so a training set
    is a union of a few runs however long the series is. Returning every
    index instead cost 37 KB for a 500-observation, 15-path split and would
    have grown linearly -- 360 KB at 5,000 observations, for a payload an
    agent then has to read back through a context window. The ranges carry
    the same information and are recovered with range(start, end).
    """
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: List[List[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous + 1])
        start = previous = value
    ranges.append([start, previous + 1])
    return ranges


def build_purged_cv_splits(input_data: PurgedCVInput) -> PurgedCVResult:
    result = lib.combinatorial_purged_cv(
        input_data.n_observations,
        n_splits=input_data.n_splits,
        n_test_splits=input_data.n_test_splits,
        embargo_pct=input_data.embargo_pct,
        label_horizon=input_data.label_horizon,
    )
    result["paths"] = [
        {
            "test_groups": path["test_groups"],
            "n_train": path["n_train"],
            "n_test": path["n_test"],
            "n_purged": path["n_purged"],
            "train_ranges": _as_ranges(path["train_index"]),
            "test_ranges": _as_ranges(path["test_index"]),
        }
        for path in result["paths"]
    ]
    return PurgedCVResult(**result)


def run_reality_check(input_data: RealityCheckInput) -> RealityCheckResult:
    return RealityCheckResult(
        **lib.reality_check(
            pd.Series(input_data.strategy_returns),
            _aligned_frame(input_data.benchmark_returns, "run_reality_check"),
            n_bootstrap=input_data.n_bootstrap,
            block_size=input_data.block_size,
            seed=input_data.seed,
        )
    )


def get_regime_stratified_performance(
    input_data: RegimeStratifiedInput,
) -> RegimeStratifiedResult:
    if len(input_data.regimes) != len(input_data.returns):
        raise ValidationError(
            f"get_regime_stratified_performance: {len(input_data.returns)} "
            f"returns against {len(input_data.regimes)} regime labels. They "
            "must be parallel -- one label per observation."
        )
    return RegimeStratifiedResult(
        **lib.regime_stratified_performance(
            pd.Series(input_data.returns),
            pd.Series(input_data.regimes),
            periods_per_year=input_data.periods_per_year,
        )
    )


def analyze_parameter_decay(input_data: ParameterDecayInput) -> ParameterDecayResult:
    return ParameterDecayResult(
        **lib.parameter_decay(
            input_data.parameter_values,
            input_data.performance,
            metric_name=input_data.metric_name,
        )
    )


VALIDATION_TOOL_DEFS = [
    (
        "get_deflated_sharpe_ratio",
        "The probability a Sharpe ratio is real GIVEN how many strategies "
        "were tried. Twenty strategies with no edge whatsoever, on two years "
        "of daily data, produce a best annualized Sharpe of 1.34 -- measured, "
        "not argued. Reporting that without saying 'and I tried 19 others' is "
        "false while every individual number is true. Skew and kurtosis widen "
        "the Sharpe's sampling distribution, so a short-volatility payoff is "
        "penalised here, correctly. Pass trial_sharpes for the accurate "
        "version: their VARIANCE sets the threshold, and 100 near-identical "
        "settings deflate far less than 100 different ideas.",
        DeflatedSharpeInput,
    ),
    (
        "estimate_backtest_overfitting",
        "PBO: how often the configuration that wins in-sample loses "
        "out-of-sample, across every equal split of the period. It measures "
        "your SELECTION PROCEDURE, not the strategy -- 0.5 means picking the "
        "in-sample best is no better than picking at random, and above 0.5 "
        "means the grid is being fitted to noise. Reports the median pairwise "
        "correlation between configurations, because a hundred settings "
        "correlated at 0.99 are one strategy and the PBO on them is "
        "meaningless.",
        PBOInput,
    ),
    (
        "build_purged_cv_splits",
        "Train/test index sets that do not leak, for a label that looks "
        "forward. A label built from a 5-day forward return at time t is a "
        "function of prices through t+5, so plain k-fold puts the test "
        "period's answer inside the training labels at every fold boundary -- "
        "which is why a model shows 0.6 AUC in cross-validation and 0.5 in "
        "production. Purging drops the overlapping training observations and "
        "the embargo drops the stretch after each test block. Combinatorial "
        "rather than sequential, so out-of-sample performance gets a "
        "DISTRIBUTION instead of one number with no error bar.",
        PurgedCVInput,
    ),
    (
        "run_reality_check",
        "White's Reality Check: is this strategy better than the best of the "
        "alternatives, or the luckiest of them? Different from a t-test on "
        "its returns, and it is the question that matters after a search. The "
        "bootstrap is BLOCKED because resampling individual days destroys the "
        "serial correlation that drives drawdowns and volatility clustering, "
        "making the null too narrow and the p-value too small.",
        RealityCheckInput,
    ),
    (
        "get_regime_stratified_performance",
        "Performance broken out by regime, because one Sharpe over a mixed "
        "sample describes none of them. Catches the strategy with an overall "
        "Sharpe of 1.2 that earned all of it in one 18-month window and was "
        "flat for the other eight years -- arithmetically correct and "
        "completely misleading, and nothing in a single Sharpe reveals it. "
        "Leads with what fraction of P&L came from the single best regime; "
        "above about 70% the strategy is a bet on that regime recurring.",
        RegimeStratifiedInput,
    ),
    (
        "analyze_parameter_decay",
        "Whether performance degrades SMOOTHLY as a parameter moves or falls "
        "off a cliff. A parameter whose neighbours perform almost as well "
        "describes a real effect with a broad optimum -- the exact value is "
        "not doing the work. A spike in a noisy objective is almost always "
        "the one setting that happened to fit the sample. Flags an optimum at "
        "the grid EDGE, where the true one may lie outside it or performance "
        "is monotone, which usually means the parameter stands in for "
        "something else.",
        ParameterDecayInput,
    ),
]

VALIDATION_TOOL_DISPATCH = {
    "get_deflated_sharpe_ratio": (get_deflated_sharpe_ratio, DeflatedSharpeInput),
    "estimate_backtest_overfitting": (estimate_backtest_overfitting, PBOInput),
    "build_purged_cv_splits": (build_purged_cv_splits, PurgedCVInput),
    "run_reality_check": (run_reality_check, RealityCheckInput),
    "get_regime_stratified_performance": (
        get_regime_stratified_performance,
        RegimeStratifiedInput,
    ),
    "analyze_parameter_decay": (analyze_parameter_decay, ParameterDecayInput),
}

__all__ = [
    "VALIDATION_TOOL_DEFS",
    "VALIDATION_TOOL_DISPATCH",
    "analyze_parameter_decay",
    "build_purged_cv_splits",
    "estimate_backtest_overfitting",
    "get_deflated_sharpe_ratio",
    "get_regime_stratified_performance",
    "run_reality_check",
]
