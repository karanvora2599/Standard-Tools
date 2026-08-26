"""
Error bars, and the questions that need them.

Almost every number this library returns is an estimate and almost none of
them arrive with a standard error. A Sharpe of 1.2 on two years of daily
data has a 95% interval running from roughly 0.2 to 2.2 -- consistent with a
mediocre strategy and with an excellent one. The interval is the number a
decision should be made on, and these produce it.

INPUTS ARE INLINE, as lists. These are as often run on a strategy's equity
curve, which no provider has, as on an asset's returns.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.analysis import inference as lib

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


StatisticName = Literal[
    "mean",
    "median",
    "std",
    "sharpe",
    "sortino",
    "skew",
    "kurtosis",
    "max_drawdown",
    "win_rate",
    "var_95",
    "cvar_95",
]


class BootstrapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: List[float] = Field(..., min_length=30, description="Periodic returns.")
    statistic: StatisticName = Field("sharpe")
    n_bootstrap: int = Field(2000, ge=100, le=50000)
    block_size: Optional[int] = Field(
        None,
        ge=1,
        description="Consecutive observations per resampled block. Defaults "
        "to n^(1/3). Setting it to 1 gives an IID bootstrap, whose interval "
        "is measurably too narrow on an autocorrelated series.",
    )
    confidence: float = Field(0.95, gt=0, lt=1)
    periods_per_year: int = Field(252, ge=1)
    seed: int = Field(0)


class CompareDistributionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_a: List[float] = Field(..., min_length=10)
    sample_b: List[float] = Field(..., min_length=10)
    label_a: str = Field("a")
    label_b: str = Field("b")


class CorrelationStabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: List[float] = Field(..., min_length=60)
    b: List[float] = Field(..., min_length=60, description="Aligned with `a`.")
    window: int = Field(63, ge=10)


class DecomposeReturnsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(..., min_length=30)
    periods_per_year: int = Field(252, ge=1)


class BootstrapResult(_Result):
    statistic: str = ""
    n_observations: int = 0
    n_bootstrap: int = 0
    block_size: int = 0
    confidence: Stat = None
    point_estimate: Stat = None
    lower: Stat = None
    upper: Stat = None
    interval_width: Stat = None
    bootstrap_mean: Stat = None
    bootstrap_std: Stat = None
    estimated_bias: Stat = None
    contains_zero: bool = False


class MomentSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    n: int = 0
    mean: Stat = None
    std: Stat = None
    skew: Stat = None
    kurtosis: Stat = None
    p01: Stat = None
    p05: Stat = None
    median: Stat = None
    p95: Stat = None
    p99: Stat = None


class MomentShift(BaseModel):
    model_config = ConfigDict(extra="allow")

    moment: str = ""
    change: Stat = None
    relative_change: Stat = None


class CompareDistributionsResult(_Result):
    n_a: int = 0
    n_b: int = 0
    ks_statistic: Stat = None
    p_value: Stat = None
    same_distribution_at_05: bool = True
    moments: Dict[str, MomentSummary] = Field(default_factory=dict)
    moment_shifts: List[MomentShift] = Field(default_factory=list)
    tail_ratio_p01: Stat = Field(
        None,
        description="How far the 1st percentile moved. KS is LEAST sensitive "
        "in the tails, so read this next to the p-value.",
    )


class CorrelationStabilityResult(_Result):
    n_observations: int = 0
    window: int = 0
    full_sample_correlation: Stat = None
    n_windows: int = 0
    n_independent_windows: int = 0
    mean_rolling: Stat = None
    min_rolling: Stat = None
    max_rolling: Stat = None
    std_rolling: Stat = None
    sign_flips: int = 0
    fraction_within_0_2: Stat = None
    stress_correlation: Stat = Field(
        None,
        description="Correlation conditional on the joint worst decile. This "
        "is the number a diversification claim has to survive.",
    )


class DecomposeReturnsResult(_Result):
    n_observations: int = 0
    arithmetic_mean: Stat = None
    geometric_mean: Stat = None
    volatility_drag: Stat = Field(
        None,
        description="Arithmetic minus geometric, which is roughly half the "
        "variance. Compounding earns the GEOMETRIC return.",
    )
    arithmetic_annualized: Stat = None
    geometric_annualized: Stat = None
    variance: Stat = None
    total_return: Stat = None
    total_without_best_5: Stat = None
    total_without_worst_5: Stat = None
    best_5_contribution: Stat = None
    worst_5_contribution: Stat = None
    n_positive: int = 0
    n_negative: int = 0
    win_rate: Stat = None
    mean_win: Stat = None
    mean_loss: Stat = None
    win_loss_ratio: Stat = None


def get_bootstrap_interval(input_data: BootstrapInput) -> BootstrapResult:
    return BootstrapResult(
        **lib.bootstrap_statistic(
            input_data.values,
            statistic=input_data.statistic,
            n_bootstrap=input_data.n_bootstrap,
            block_size=input_data.block_size,
            confidence=input_data.confidence,
            periods_per_year=input_data.periods_per_year,
            seed=input_data.seed,
        )
    )


def compare_distributions(
    input_data: CompareDistributionsInput,
) -> CompareDistributionsResult:
    return CompareDistributionsResult(
        **lib.compare_distributions(
            input_data.sample_a,
            input_data.sample_b,
            label_a=input_data.label_a,
            label_b=input_data.label_b,
        )
    )


def get_correlation_stability(
    input_data: CorrelationStabilityInput,
) -> CorrelationStabilityResult:
    return CorrelationStabilityResult(
        **lib.rolling_correlation_stability(
            input_data.a, input_data.b, window=input_data.window
        )
    )


def decompose_returns(input_data: DecomposeReturnsInput) -> DecomposeReturnsResult:
    return DecomposeReturnsResult(
        **lib.decompose_returns(
            input_data.returns, periods_per_year=input_data.periods_per_year
        )
    )


INFERENCE_TOOL_DEFS = [
    (
        "get_bootstrap_interval",
        "A confidence interval for a statistic, by BLOCK bootstrap. The point "
        "estimate is usually reported alone and usually should not be: a "
        "Sharpe of 1.2 on two years of daily data has a 95% interval from "
        "about 0.2 to 2.2, consistent with a mediocre strategy and an "
        "excellent one. Blocked rather than IID because resampling individual "
        "returns destroys serial correlation -- measured on AR(1) returns at "
        "phi=0.8, the IID interval is 2.24x too narrow for the Sharpe and "
        "1.63x for maximum drawdown, while at phi=0 the two agree so the "
        "correction costs nothing when it is not needed.",
        BootstrapInput,
    ),
    (
        "compare_distributions",
        "Whether two samples came from the same distribution -- not whether "
        "their means differ. In-sample against out-of-sample, this regime "
        "against that one, live against backtest: the question gets answered "
        "with a t-test, and a t-test misses every difference in SHAPE. A "
        "strategy whose out-of-sample mean matches and whose kurtosis has "
        "tripled is not performing as expected. Returns which MOMENT moved, "
        "and reports the tail separately because KS's power is concentrated "
        "near the median -- a normal against a t(3) gives KS p=0.45 while the "
        "1st percentile has moved by a factor of two.",
        CompareDistributionsInput,
    ),
    (
        "get_correlation_stability",
        "Whether a correlation is a property of the pair or an average over "
        "two different regimes. Two assets correlating at 0.0 over ten years "
        "may have correlated at +0.7 for five and -0.7 for five; the average "
        "is meaningless and a hedge sized on it is wrong in both regimes. "
        "Reports the sign-flip count, the range, and separately the "
        "correlation conditional on the joint worst decile -- because "
        "correlations move toward 1 when everything falls together, so a "
        "hedge computed on a calm sample fails precisely when it is needed.",
        CorrelationStabilityInput,
    ),
    (
        "decompose_returns",
        "Where compound growth actually came from. THE ARITHMETIC MEAN IS NOT "
        "WHAT YOU EARNED: compound growth is the arithmetic mean minus "
        "roughly half the variance, and for a volatile strategy that drag is "
        "most of the return -- 0.08% a day at 3% daily vol is 20% arithmetic "
        "and 10% compound. Also separates the contribution of the best and "
        "worst five days, because a strategy whose entire return disappears "
        "when five days are removed is a lottery ticket with good statistics.",
        DecomposeReturnsInput,
    ),
]

INFERENCE_TOOL_DISPATCH = {
    "get_bootstrap_interval": (get_bootstrap_interval, BootstrapInput),
    "compare_distributions": (compare_distributions, CompareDistributionsInput),
    "get_correlation_stability": (
        get_correlation_stability,
        CorrelationStabilityInput,
    ),
    "decompose_returns": (decompose_returns, DecomposeReturnsInput),
}

__all__ = [
    "INFERENCE_TOOL_DEFS",
    "INFERENCE_TOOL_DISPATCH",
    "compare_distributions",
    "decompose_returns",
    "get_bootstrap_interval",
    "get_correlation_stability",
]
