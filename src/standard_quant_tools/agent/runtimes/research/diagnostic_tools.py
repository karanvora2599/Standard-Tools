"""
Questions about a series that a summary statistic hides.

A Sharpe ratio, a mean and a standard deviation describe a series as if it
were an unordered bag of numbers. Every tool here uses the ORDER, because
the order is where the interesting failures live: when the edge stopped
working, whether the whole result is one calendar effect, whether returns
predict themselves, and whether a drawdown was one event or a long grind.

INPUTS ARE INLINE. Returns arrive as a list, a universe as a map of name to
series. Nothing here fetches prices, because these are as often run on a
strategy's equity curve -- which no provider has -- as on an asset's.

DATES ARE OPTIONAL EXCEPT WHERE THEY ARE THE QUESTION. `run_seasonality`
requires them and refuses without them: there is no calendar in a positional
index, and a seasonality test on invented dates returns confident nonsense.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.analysis import diagnostics as lib
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


def _dated(values: List[float], dates: Optional[List[str]], who: str) -> pd.Series:
    """
    A series with a real DatetimeIndex when dates were supplied.

    A length mismatch is refused by name rather than truncated: pandas would
    align on the shorter of the two and silently drop observations off the
    end, which changes the answer without changing anything visible.
    """
    if dates is None:
        return pd.Series(values, dtype=float)
    if len(dates) != len(values):
        raise ValidationError(
            f"{who}: {len(values)} values against {len(dates)} dates. They "
            "must be parallel -- aligning on the shorter one would drop "
            "observations without saying so."
        )
    try:
        index = pd.DatetimeIndex(pd.to_datetime(dates))
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{who}: could not parse `dates` -- {exc}") from None
    return pd.Series(values, index=index, dtype=float)


# ── inputs ──────────────────────────────────────────────────────────────


class LjungBoxInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: List[float] = Field(..., min_length=30)
    lags: Optional[int] = Field(
        None,
        ge=1,
        description="Lags to test jointly. Defaults to min(10, n/5) -- a test "
        "at 40 lags on 200 observations has no power against anything.",
    )
    squared: bool = Field(
        False,
        description="False tests predictability of DIRECTION, usually absent. "
        "True tests volatility clustering, almost always present -- and "
        "finding it is not a trading signal, it is why GARCH exists.",
    )


class SeasonalityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(..., min_length=30)
    dates: List[str] = Field(
        ...,
        description="ISO dates, one per return. REQUIRED -- there is no "
        "calendar in a positional index.",
    )
    by: Literal["weekday", "month", "day_of_month"] = Field("weekday")


class EntropyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: List[float] = Field(..., min_length=50)
    n_bins: int = Field(8, ge=2, le=200)
    embedding: int = Field(
        3,
        ge=2,
        le=7,
        description="Window length for the ordinal patterns. There are "
        "factorial(embedding) of them, so 7 needs a very long sample.",
    )


class SharpeStabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(..., min_length=60)
    window: int = Field(
        252, ge=20, description="Rolling window, for the displayed series."
    )
    periods_per_year: int = Field(252, ge=1)


class DrawdownProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: List[float] = Field(..., min_length=30)
    dates: Optional[List[str]] = Field(
        None, description="ISO dates, so episodes are reported with real dates."
    )
    threshold: float = Field(
        0.05, gt=0, lt=1, description="Minimum depth to count as an episode."
    )
    top_n: int = Field(5, ge=1, le=50)


class LeadLagInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Dict[str, List[float]] = Field(
        ..., description="Asset name -> return series, all the same length."
    )
    max_lag: int = Field(3, ge=1, le=20)
    min_correlation: float = Field(
        0.1,
        ge=0,
        lt=1,
        description="Correlations below this are not reported at all.",
    )


class StructuralBreakInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: List[float] = Field(..., min_length=20)
    break_index: int = Field(
        ...,
        ge=0,
        description="Position of the suspected break. It must come from "
        "OUTSIDE the data -- a regulation, a fee change, a go-live. A date "
        "chosen because the data looks different there is not a valid test.",
    )
    regressor: Optional[List[float]] = Field(
        None,
        description="With it, tests whether the RELATIONSHIP broke (a beta or "
        "hedge ratio). Without it, whether the mean moved.",
    )


# ── results ─────────────────────────────────────────────────────────────


class LagRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    lag: int = 0
    autocorrelation: Stat = None
    significant_alone: bool = False


class LjungBoxResult(_Result):
    n_observations: int = 0
    lags: int = 0
    on_squared_returns: bool = False
    statistic: Stat = None
    p_value: Stat = None
    significant_at_05: bool = False
    n_lags_individually_significant: int = 0
    by_lag: List[LagRow] = Field(default_factory=list)


class SeasonRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    period: str = ""
    n_observations: int = 0
    mean_return: Stat = None
    annualized: Stat = None
    win_rate: Stat = None
    t_statistic: Stat = None
    p_value_raw: Stat = None
    p_value_corrected: Stat = None
    significant_after_correction: bool = False


class SeasonalityResult(_Result):
    n_observations: int = 0
    by: str = ""
    n_periods: int = 0
    joint_f_statistic: Stat = None
    joint_p_value: Stat = Field(
        None,
        description="Read this FIRST. If it does not reject, no individual "
        "period should be reported however striking it looks.",
    )
    joint_significant: bool = False
    by_period: List[SeasonRow] = Field(default_factory=list)
    n_surviving_correction: int = 0


class EntropyResult(_Result):
    n_observations: int = 0
    n_bins: int = 0
    embedding: int = 0
    observations_per_bin: Stat = None
    shannon_entropy: Stat = None
    shannon_normalized: Stat = None
    permutation_entropy: Stat = None
    permutation_normalized: Stat = Field(
        None,
        description="1.0 is indistinguishable from random. Below ~0.9 there "
        "is ordering structure a LINEAR test would miss.",
    )
    n_patterns_observed: int = 0
    n_patterns_possible: int = 0


class SharpeStabilityResult(_Result):
    n_observations: int = 0
    window: int = 0
    n_windows: int = 0
    n_independent_windows: int = 0
    full_sample_sharpe: Stat = None
    mean_rolling_sharpe: Stat = None
    std_rolling_sharpe: Stat = None
    min_rolling_sharpe: Stat = None
    max_rolling_sharpe: Stat = None
    first_half_sharpe: Stat = None
    second_half_sharpe: Stat = None
    sharpe_difference: Stat = None
    difference_standard_error: Stat = None
    decay_p_value: Stat = Field(
        None,
        description="From a two-sample comparison of NON-OVERLAPPING halves, "
        "not from a regression on the rolling series -- consecutive windows "
        "share all but one observation and cannot support inference.",
    )
    decaying: bool = False
    trend_per_year: Stat = None
    n_blocks: int = 0
    fraction_of_windows_positive: Stat = None


class DrawdownEpisode(BaseModel):
    model_config = ConfigDict(extra="allow")

    start: str = ""
    trough: str = ""
    end: str = ""
    depth: Stat = None
    length_days: int = 0
    days_to_trough: int = 0
    recovery_days: Optional[int] = None
    recovered: bool = False


class DrawdownProfileResult(_Result):
    n_observations: int = 0
    threshold: Stat = None
    max_drawdown: Stat = None
    n_drawdowns: int = 0
    fraction_underwater: Stat = Field(
        None,
        description="Time underwater is usually the binding constraint on "
        "holding a strategy -- allocators redeem on duration, not depth.",
    )
    mean_recovery_days: Stat = None
    longest_drawdown_days: int = 0
    currently_in_drawdown: bool = False
    worst_drawdowns: List[DrawdownEpisode] = Field(default_factory=list)


class LeadLagPair(BaseModel):
    model_config = ConfigDict(extra="allow")

    leader: str = ""
    follower: str = ""
    lag: int = 0
    correlation: Stat = None
    p_value_raw: Stat = None
    p_value_corrected: Stat = None
    survives_correction: bool = False


class LeadLagResult(_Result):
    n_assets: int = 0
    n_observations: int = 0
    max_lag: int = 0
    n_tests: int = 0
    expected_false_positives_uncorrected: Stat = None
    n_surviving: int = 0
    surviving_pairs: List[LeadLagPair] = Field(default_factory=list)
    strongest_pairs: List[LeadLagPair] = Field(
        default_factory=list,
        description="Ranked by size, corrected or not. When n_surviving is "
        "zero this list is what noise looks like, not a finding.",
    )


class StructuralBreakResult(_Result):
    n_observations: int = 0
    break_index: int = 0
    break_date: str = ""
    tested: str = ""
    f_statistic: Stat = None
    p_value: Stat = None
    significant_at_05: bool = False
    pooled_coefficients: List[float] = Field(default_factory=list)
    before_coefficients: List[float] = Field(default_factory=list)
    after_coefficients: List[float] = Field(default_factory=list)
    mean_before: Stat = None
    mean_after: Stat = None


# ── tools ───────────────────────────────────────────────────────────────


def test_autocorrelation(input_data: LjungBoxInput) -> LjungBoxResult:
    return LjungBoxResult(
        **lib.ljung_box(
            pd.Series(input_data.series),
            lags=input_data.lags,
            squared=input_data.squared,
        )
    )


def run_seasonality_analysis(input_data: SeasonalityInput) -> SeasonalityResult:
    series = _dated(input_data.returns, input_data.dates, "run_seasonality_analysis")
    return SeasonalityResult(**lib.seasonality(series, by=input_data.by))


def get_entropy_measures(input_data: EntropyInput) -> EntropyResult:
    return EntropyResult(
        **lib.entropy_measures(
            pd.Series(input_data.series),
            n_bins=input_data.n_bins,
            embedding=input_data.embedding,
        )
    )


def get_sharpe_stability(input_data: SharpeStabilityInput) -> SharpeStabilityResult:
    return SharpeStabilityResult(
        **lib.rolling_sharpe_stability(
            pd.Series(input_data.returns),
            window=input_data.window,
            periods_per_year=input_data.periods_per_year,
        )
    )


def get_drawdown_profile(input_data: DrawdownProfileInput) -> DrawdownProfileResult:
    series = _dated(input_data.returns, input_data.dates, "get_drawdown_profile")
    return DrawdownProfileResult(
        **lib.drawdown_profile(
            series, threshold=input_data.threshold, top_n=input_data.top_n
        )
    )


def get_lead_lag_matrix(input_data: LeadLagInput) -> LeadLagResult:
    lengths = {name: len(values) for name, values in input_data.returns.items()}
    if len(set(lengths.values())) > 1:
        raise ValidationError(
            f"get_lead_lag_matrix: the series have different lengths "
            f"({lengths}). A lead-lag search on misaligned series measures "
            "the misalignment."
        )
    return LeadLagResult(
        **lib.lead_lag_matrix(
            pd.DataFrame(input_data.returns),
            max_lag=input_data.max_lag,
            min_correlation=input_data.min_correlation,
        )
    )


def test_structural_break(input_data: StructuralBreakInput) -> StructuralBreakResult:
    regressor = None
    if input_data.regressor is not None:
        if len(input_data.regressor) != len(input_data.series):
            raise ValidationError(
                f"test_structural_break: {len(input_data.series)} series "
                f"values against {len(input_data.regressor)} regressor values."
            )
        regressor = pd.Series(input_data.regressor)
    return StructuralBreakResult(
        **lib.structural_break_test(
            pd.Series(input_data.series),
            input_data.break_index,
            regressor=regressor,
        )
    )


DIAGNOSTIC_TOOL_DEFS = [
    (
        "test_autocorrelation",
        "A JOINT Ljung-Box test for autocorrelation across lags, rather than "
        "one test per lag. Check 20 lags individually at 5% on white noise "
        "and you expect one to fire; reporting that as 'returns are "
        "autocorrelated at lag 13' is an uncorrected multiple comparison and "
        "is how a great many spurious signals begin. Set squared=True to test "
        "volatility clustering instead of direction -- clustering is "
        "near-universal and is not a directional signal, it is why GARCH "
        "exists.",
        LjungBoxInput,
    ),
    (
        "run_seasonality_analysis",
        "Whether performance concentrates in a day of the week or a month of "
        "the year, corrected for having looked at all of them. Testing twelve "
        "months at 5% produces at least one 'significant' result on pure "
        "noise 46% of the time, which is where a good share of published "
        "calendar anomalies come from. A joint F-test is reported FIRST and "
        "every per-period p-value is Bonferroni corrected; if the joint test "
        "does not reject, no individual period should be reported however "
        "striking it looks.",
        SeasonalityInput,
    ),
    (
        "get_entropy_measures",
        "How predictable a series is WITHOUT assuming the predictability is "
        "linear. Every other statistical tool here -- autocorrelation, "
        "regression, Granger -- measures linear dependence and returns "
        "nothing on a series that is perfectly deterministic in a nonlinear "
        "way. Permutation entropy reads only the RANK ORDER inside each "
        "window, so it is invariant to any monotone transformation and robust "
        "to outliers; a value near 1.0 is indistinguishable from random.",
        EntropyInput,
    ),
    (
        "get_sharpe_stability",
        "Whether the edge DECAYED, or the full-sample Sharpe is the average "
        "of a good period and a dead one. A Sharpe of 1.0 made of 2.0 in the "
        "first half and 0.0 in the second is arithmetically correct and "
        "describes a dead strategy -- and the second half is the half that "
        "predicts tomorrow. The p-value comes from comparing two "
        "NON-OVERLAPPING halves, not from a regression on the rolling series, "
        "because consecutive rolling windows share all but one observation "
        "and cannot support inference.",
        SharpeStabilityInput,
    ),
    (
        "get_drawdown_profile",
        "Every drawdown, not just the worst. Maximum drawdown is one number "
        "describing one event and it says nothing about how often drawdowns "
        "happen, how long they last, or whether the worst was a one-day gap "
        "or a two-year grind -- and those determine whether a strategy is "
        "holdable far more than depth does. A 20% drawdown recovering in a "
        "month is survivable; one taking three years ends the mandate. Depth "
        "and duration are close to independent and both are reported.",
        DrawdownProfileInput,
    ),
    (
        "get_lead_lag_matrix",
        "Which series move first across a universe -- and why the answer is "
        "usually noise. Twenty assets at three lags is 1,140 correlations, of "
        "which about 57 clear an uncorrected 5% bar on data with NO lead-lag "
        "structure at all, and the strongest of those looks entirely "
        "convincing. Every pair carries a Bonferroni-corrected p-value "
        "against the full search size, and when nothing survives the result "
        "says so rather than presenting the top of the ranked list.",
        LeadLagInput,
    ),
    (
        "test_structural_break",
        "A Chow test for a break at a KNOWN date. The 'known' is "
        "load-bearing: a test at a date chosen because the data looks "
        "different there is not a valid test, because the hypothesis was "
        "picked using the data. Valid when the date comes from outside -- a "
        "regulation taking effect, a fee change, an index reconstitution, a "
        "strategy going live. For an unknown date use detect_change_points, "
        "which searches and reports the gain. With a `regressor` it tests "
        "whether the RELATIONSHIP broke (a beta or a hedge ratio) rather than "
        "whether the mean moved.",
        StructuralBreakInput,
    ),
]

DIAGNOSTIC_TOOL_DISPATCH = {
    "test_autocorrelation": (test_autocorrelation, LjungBoxInput),
    "run_seasonality_analysis": (run_seasonality_analysis, SeasonalityInput),
    "get_entropy_measures": (get_entropy_measures, EntropyInput),
    "get_sharpe_stability": (get_sharpe_stability, SharpeStabilityInput),
    "get_drawdown_profile": (get_drawdown_profile, DrawdownProfileInput),
    "get_lead_lag_matrix": (get_lead_lag_matrix, LeadLagInput),
    "test_structural_break": (test_structural_break, StructuralBreakInput),
}

__all__ = [
    "DIAGNOSTIC_TOOL_DEFS",
    "DIAGNOSTIC_TOOL_DISPATCH",
    "get_drawdown_profile",
    "get_entropy_measures",
    "get_lead_lag_matrix",
    "get_sharpe_stability",
    "run_seasonality_analysis",
    "test_autocorrelation",
    "test_structural_break",
]
