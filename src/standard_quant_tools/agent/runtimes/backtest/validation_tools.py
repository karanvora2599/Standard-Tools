"""
How much of a backtest is real: the multiple-testing tools.

Every tool here answers a question the backtest itself cannot. A backtest
reports what a strategy did on a sample. It cannot report how many other
strategies you tried before keeping this one, and that number changes what
the result means more than almost anything inside the backtest does.

INPUTS ARE INLINE OR A PUBLISHED PANEL. Returns come in as a list of
numbers, a trial grid as a map of configuration name to its return series
-- or, for the deflated Sharpe, PBO and the reality check, as `returns_ref`,
a `returns_panel` whose columns are the series, put on the dates they all
share with each column's losses reported in `streams`. Nothing here re-runs
a backtest -- these consume results the caller already has, from this
library's backtest tools or from anywhere else. That is deliberate: the
trials you need to declare include the ones you ran last week and the ones
you ran in a notebook, and no tool can enumerate those for you.

THE MODELS FORBID EXTRA FIELDS. A dropped `n_trials` would silently deflate
for one trial and report a strategy as significant that is not.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from standard_quant_tools.agent.models import Seed
from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.agent.runtimes.streams import (
    StreamAlignment,
    returns_ref_streams,
)
from standard_quant_tools.backtesting import overfitting as lib
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(
        default_factory=list,
        description="The conditions under which the numbers above are wrong.",
    )


#: `streams` on the three results that take a returns_ref.
_STREAMS_DESCRIPTION = (
    "returns_ref only: the dates the panel's columns share -- the rows "
    "used -- and what each column lost to them. Null for inline returns."
)

#: `returns_ref`'s description, up to what its columns are.
_RETURNS_REF = "An `sqt://returns_panel/...` reference in place of "


def _exactly_one(model: Any, inline: List[str], who: str) -> None:
    """Refuse both an inline form and returns_ref, or neither, by name."""
    given = [name for name in inline if getattr(model, name) is not None]
    if given and model.returns_ref is not None:
        raise ValueError(
            f"{who}: {' and '.join(given)} and returns_ref were both given. "
            "The returns are read from one of them; drop the other."
        )
    if not given and model.returns_ref is None:
        raise ValueError(
            f"{who} needs {' or '.join(inline)} inline, or returns_ref, a "
            "published returns_panel."
        )


# ── inputs ──────────────────────────────────────────────────────────────


class DeflatedSharpeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Optional[List[float]] = Field(
        None, min_length=20, description="Periodic returns as decimals."
    )
    returns_ref: Optional[str] = Field(
        None,
        description=f"{_RETURNS_REF}`returns`, read as one dated series; "
        "returns_column names it when the panel has several columns.",
    )
    returns_column: Optional[str] = Field(
        None, description="With returns_ref: the column that is the strategy."
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

    @model_validator(mode="after")
    def _one_series(self) -> "DeflatedSharpeInput":
        _exactly_one(self, ["returns"], "get_deflated_sharpe_ratio")
        if self.returns_column is not None and self.returns_ref is None:
            raise ValueError(
                "returns_column names a column of returns_ref and was given "
                "without it. Drop returns_column, or pass returns_ref."
            )
        return self


class PBOInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_returns: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Configuration name -> its return series. Every series "
        "must be the same length and aligned in time.",
    )
    returns_ref: Optional[str] = Field(
        None,
        description=f"{_RETURNS_REF}trial_returns: each column is a " "configuration.",
    )
    n_splits: int = Field(
        8,
        ge=4,
        le=16,
        description="Chunks the period is cut into. Must be EVEN -- the "
        "method forms every equal split of the chunks into two halves, so "
        "the count grows as C(n, n/2) and 16 is already 12,870 splits.",
    )

    @model_validator(mode="after")
    def _one_grid(self) -> "PBOInput":
        _exactly_one(self, ["trial_returns"], "estimate_backtest_overfitting")
        return self


#: The longest series one call splits: a year of one-minute bars. The
#: splitter is linear in the length (see the CHANGELOG entry of
#: 2026-09-28), so this bounds memory rather than time.
_MAX_CV_OBSERVATIONS = 100_000

#: Paths times observations one call may build. Every path carries a train
#: and a test index as long as the series, so memory is their product, and
#: C(n_splits, n_test_splits) grows fast: C(16, 8) is 12,870 paths. The
#: default six groups, two held out, at the observation ceiling is
#: 1,500,000; the ceiling keeps the largest legal call to about a third of
#: a second and roughly a hundred megabytes while it runs.
_MAX_CV_PATH_OBSERVATIONS = 3_000_000

#: Paths one call returns. Each is a row of the result, however short the
#: series, so the payload grows with the count and not with the length --
#: 12,870 paths is 2.5 MB of ranges for an agent to read back, and 6,000
#: keeps it near one.
_MAX_CV_PATHS = 6_000


class PurgedCVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_observations: int = Field(
        ...,
        ge=50,
        le=_MAX_CV_OBSERVATIONS,
        description="Length of the series to be split. At most 100,000 -- a "
        "year of one-minute bars; the paths are built whole, so the ceiling "
        "bounds the memory one call holds.",
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
        le=_MAX_CV_OBSERVATIONS,
        description="How many observations forward the label looks. A 5-day "
        "forward return is 5. This is what purging needs to know: training "
        "rows this close to a test block are purged on BOTH sides -- before "
        "it, because their labels reach into it, and after it, because they "
        "sit inside its labels' reach.",
    )

    @model_validator(mode="after")
    def _bounded_work(self) -> "PurgedCVInput":
        # Checked before the combination count is formed: C(n, k) is only
        # defined for k <= n, and the library refuses k >= n_splits by name.
        if self.n_test_splits >= self.n_splits:
            return self
        n_paths = math.comb(self.n_splits, self.n_test_splits)
        if n_paths > _MAX_CV_PATHS:
            raise ValueError(
                f"C({self.n_splits}, {self.n_test_splits}) = {n_paths:,} paths "
                f"is above the {_MAX_CV_PATHS:,} one call returns: every path "
                "is a row of the result, however short the series. Hold out "
                "fewer groups or cut fewer groups."
            )
        work = n_paths * self.n_observations
        if work > _MAX_CV_PATH_OBSERVATIONS:
            raise ValueError(
                f"C({self.n_splits}, {self.n_test_splits}) = {n_paths:,} paths "
                f"over {self.n_observations:,} observations is {work:,} "
                "path-observations, above the "
                f"{_MAX_CV_PATH_OBSERVATIONS:,} one call builds. Hold out fewer "
                "groups, cut fewer groups, or split a shorter series."
            )
        return self


class RealityCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Literal["best_candidate_vs_benchmark", "beats_every_alternative"] = Field(
        "best_candidate_vs_benchmark",
        description="The hypothesis. 'best_candidate_vs_benchmark' "
        "(White's Reality Check): does the best of candidate_returns beat "
        "benchmark_returns by more than the best of that many would by "
        "luck? 'beats_every_alternative': does strategy_returns beat EVERY "
        "series in alternative_returns? Each reads only its own inputs.",
    )
    candidate_returns: Optional[Dict[str, List[float]]] = Field(
        None,
        description="best_candidate_vs_benchmark: EVERY candidate the search "
        "considered, name -> return series, the one it kept among them. "
        "Leaving out the discarded ones removes the correction the test "
        "exists for.",
    )
    benchmark_returns: Optional[List[float]] = Field(
        None,
        min_length=30,
        description="best_candidate_vs_benchmark: the ONE series the "
        "candidates must beat, aligned with them. Omitted, the benchmark is "
        "zero.",
    )
    strategy_returns: Optional[List[float]] = Field(
        None,
        min_length=30,
        description="beats_every_alternative: the one strategy.",
    )
    alternative_returns: Optional[Dict[str, List[float]]] = Field(
        None,
        description="beats_every_alternative: every alternative the strategy "
        "must beat, name -> return series aligned with it. The p-value is "
        "the largest pairwise one: beating every alternative means beating "
        "the hardest.",
    )
    returns_ref: Optional[str] = Field(
        None,
        description=f"{_RETURNS_REF}the inline series. Under "
        "best_candidate_vs_benchmark every column but benchmark_column is a "
        "candidate; under beats_every_alternative every column but "
        "strategy_column is an alternative.",
    )
    benchmark_column: Optional[str] = Field(
        None,
        description="best_candidate_vs_benchmark with returns_ref: the "
        "benchmark's column. Omitted, the benchmark is zero.",
    )
    strategy_column: Optional[str] = Field(
        None,
        description="beats_every_alternative with returns_ref: the "
        "strategy's column (required).",
    )
    method: Literal["white", "hansen_spa"] = Field(
        "white",
        description="best_candidate_vs_benchmark only. 'hansen_spa' divides "
        "each candidate's outperformance by its standard error and keeps "
        "candidates far below the benchmark from widening the null (Hansen "
        "2005); White's form is conservative when the search held poor "
        "candidates.",
    )
    n_bootstrap: int = Field(1000, ge=100, le=20000)
    block_size: int = Field(
        20,
        ge=1,
        description="Consecutive days per bootstrap block. Blocks preserve "
        "serial correlation; resampling single days makes the null too "
        "narrow and the p-value too small.",
    )
    seed: Seed = Field(0)

    @model_validator(mode="before")
    @classmethod
    def _benchmark_is_one_series(cls, data: Any) -> Any:
        # The shape this field had before the question was explicit: a map
        # of alternatives, tested against the WEAKEST of them. Refused with
        # where each series belongs now, rather than as a type error.
        if isinstance(data, dict) and isinstance(data.get("benchmark_returns"), dict):
            raise ValueError(
                "benchmark_returns is ONE return series now, not a map of "
                "alternatives. To ask whether a strategy beats every "
                "alternative, pass question='beats_every_alternative' with "
                "strategy_returns and alternative_returns={name: series}. To "
                "ask whether the best of a search beats a benchmark, pass "
                "candidate_returns={name: series} and benchmark_returns as "
                "one series (or omit it for zero). The map form tested the "
                "strategy against its weakest alternative -- see the "
                "CHANGELOG entry of 2026-10-22."
            )
        return data

    @model_validator(mode="after")
    def _inputs_of_the_question(self) -> "RealityCheckInput":
        if self.returns_ref is not None:
            return self._columns_of_the_question()
        stray = [
            name
            for name in ("benchmark_column", "strategy_column")
            if getattr(self, name) is not None
        ]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} names a column of returns_ref and was "
                "given without it. Pass returns_ref, or the series inline."
            )
        if self.question == "best_candidate_vs_benchmark":
            needed, unread = ["candidate_returns"], [
                "strategy_returns",
                "alternative_returns",
            ]
        else:
            needed, unread = ["strategy_returns", "alternative_returns"], [
                "candidate_returns",
                "benchmark_returns",
            ]
        missing = [name for name in needed if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"question={self.question!r} needs {' and '.join(missing)}."
            )
        given = [name for name in unread if getattr(self, name) is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} is not read under question="
                f"{self.question!r}; best_candidate_vs_benchmark reads "
                "candidate_returns and benchmark_returns, "
                "beats_every_alternative reads strategy_returns and "
                "alternative_returns."
            )
        if self.question == "beats_every_alternative" and self.method != "white":
            raise ValueError(
                f"method={self.method!r} applies to question="
                "'best_candidate_vs_benchmark'; under beats_every_alternative "
                "each pair is tested on its own. Leave method at 'white'."
            )
        return self

    def _columns_of_the_question(self) -> "RealityCheckInput":
        """The roles of returns_ref's columns under the question asked."""
        inline = [
            name
            for name in (
                "candidate_returns",
                "benchmark_returns",
                "strategy_returns",
                "alternative_returns",
            )
            if getattr(self, name) is not None
        ]
        if inline:
            raise ValueError(
                f"{' and '.join(inline)} and returns_ref were both given. The "
                "series are read from one of them: with returns_ref, name the "
                "benchmark_column or strategy_column and every other column "
                "takes the question's other role."
            )
        if self.question == "best_candidate_vs_benchmark":
            if self.strategy_column is not None:
                raise ValueError(
                    "strategy_column is read under question="
                    "'beats_every_alternative'. Under best_candidate_vs_benchmark "
                    "every column of returns_ref but benchmark_column is a "
                    "candidate; drop strategy_column or change the question."
                )
        else:
            if self.benchmark_column is not None:
                raise ValueError(
                    "benchmark_column is read under question="
                    "'best_candidate_vs_benchmark'. Under beats_every_alternative "
                    "name the strategy in strategy_column; every other column "
                    "is an alternative."
                )
            if self.strategy_column is None:
                raise ValueError(
                    "question='beats_every_alternative' with returns_ref needs "
                    "strategy_column: which column is the strategy, every other "
                    "column being an alternative it must beat."
                )
            if self.method != "white":
                raise ValueError(
                    f"method={self.method!r} applies to question="
                    "'best_candidate_vs_benchmark'; under beats_every_alternative "
                    "each pair is tested on its own. Leave method at 'white'."
                )
        return self


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
    streams: Optional[StreamAlignment] = Field(None, description=_STREAMS_DESCRIPTION)
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
    streams: Optional[StreamAlignment] = Field(None, description=_STREAMS_DESCRIPTION)
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
    streams: Optional[StreamAlignment] = Field(None, description=_STREAMS_DESCRIPTION)
    question: str = Field(
        "", description="The hypothesis tested, as the input named it."
    )
    method: str = ""
    n_observations: int = 0
    n_candidates: Optional[int] = Field(
        None, description="best_candidate_vs_benchmark: candidates searched."
    )
    benchmark_given: Optional[bool] = Field(
        None,
        description="best_candidate_vs_benchmark: False when the benchmark "
        "was zero.",
    )
    best_candidate: Optional[str] = Field(
        None,
        description="best_candidate_vs_benchmark: the candidate the "
        "statistic is taken at -- the largest mean outperformance, or under "
        "hansen_spa the largest studentised one.",
    )
    n_alternatives: Optional[int] = None
    binding_alternative: Optional[str] = Field(
        None,
        description="beats_every_alternative: the alternative with the "
        "largest pairwise p-value, which is the test's p-value.",
    )
    per_alternative: Optional[Dict[str, Dict[str, Stat]]] = Field(
        None,
        description="beats_every_alternative: each alternative's annualised "
        "mean margin (outperformance) and one-sided p-value.",
    )
    n_bootstrap: int = 0
    block_size: int = 0
    observed_outperformance: Stat = Field(
        None,
        description="Annualised mean return over the benchmark of "
        "best_candidate, or under beats_every_alternative the smallest "
        "margin over any alternative.",
    )
    observed_statistic: Stat = Field(
        None,
        description="The test statistic: the annualised outperformance "
        "above, or under hansen_spa the studentised maximum (no units).",
    )
    p_value: Stat = None
    significant_at_05: bool = False
    bootstrap_p95: Stat = Field(
        None,
        description="The 95th percentile of the statistic under the null, "
        "in observed_statistic's units.",
    )


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
        description="Share of total P&L from the single best regime -- or, "
        "when the total is a loss, share of the loss from the single worst "
        "one. Above 0.7 the strategy is a bet on that regime recurring; "
        "above 1 the other regimes together went the other way.",
    )
    pnl_concentration_regime: Optional[str] = Field(
        None, description="The regime pnl_concentration is the share of."
    )
    pnl_concentration_of: Optional[Literal["profit", "loss"]] = Field(
        None,
        description="`profit` when the total P&L is positive, `loss` when it "
        "is negative; null when it is exactly zero.",
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


def _columns(
    streams: Dict[str, Any], who: str, min_overlap: int
) -> "tuple[Dict[str, List[float]], StreamAlignment, List[str]]":
    """
    A returns_panel's columns on the dates they all share, as the
    name -> list the inline inputs are, so the library is called exactly as
    it is for inline returns. The dates are honoured in the alignment and
    the order; the computation, like the inline one, is positional.
    """
    from standard_quant_tools.portfolio.streams import (
        align_return_streams,
        alignment_notes,
    )

    frame, report = align_return_streams(streams, min_overlap=min_overlap, what=who)
    return (
        {str(c): frame[c].tolist() for c in frame.columns},
        StreamAlignment(**report),
        alignment_notes(report, who),
    )


def _the_column(
    streams: Dict[str, Any], wanted: Optional[str], role: str, field: str, who: str
) -> Optional[str]:
    """`wanted` checked against the panel's columns, for a refusal by name."""
    if wanted is not None and wanted not in streams:
        raise ValidationError(
            f"{who}: {field}={wanted!r} is not a column of returns_ref; its "
            f"columns are {list(streams)[:12]}. Name the {role}'s column."
        )
    return wanted


def get_deflated_sharpe_ratio(
    input_data: DeflatedSharpeInput,
) -> DeflatedSharpeResult:
    who = "get_deflated_sharpe_ratio"
    returns = input_data.returns
    streams = None
    notes: List[str] = []
    if input_data.returns_ref is not None:
        panel = returns_ref_streams(input_data.returns_ref, what=who)
        column = _the_column(
            panel, input_data.returns_column, "strategy", "returns_column", who
        )
        if column is None and len(panel) != 1:
            raise ValidationError(
                f"{who}: returns_ref has {len(panel)} columns "
                f"({list(panel)[:12]}) and the deflated Sharpe is of ONE "
                "series. Name it in returns_column."
            )
        column = column or next(iter(panel))
        columns, streams, notes = _columns({column: panel[column]}, who, 20)
        (returns,) = columns.values()
    result = lib.deflated_sharpe_ratio(
        pd.Series(returns),
        n_trials=input_data.n_trials,
        trial_sharpes=input_data.trial_sharpes,
        benchmark_sharpe=input_data.benchmark_sharpe,
        periods_per_year=input_data.periods_per_year,
    )
    result["warnings"] = notes + list(result.get("warnings", []))
    return DeflatedSharpeResult(**result, streams=streams)


def estimate_backtest_overfitting(input_data: PBOInput) -> PBOResult:
    who = "estimate_backtest_overfitting"
    trials = input_data.trial_returns
    streams = None
    notes: List[str] = []
    if input_data.returns_ref is not None:
        # The library needs five rows per chunk; asked here, a short overlap
        # names the column that limits it.
        trials, streams, notes = _columns(
            returns_ref_streams(input_data.returns_ref, what=who),
            who,
            input_data.n_splits * 5,
        )
    result = lib.probability_of_backtest_overfitting(
        _aligned_frame(trials, who),
        n_splits=input_data.n_splits,
    )
    result["warnings"] = notes + list(result.get("warnings", []))
    return PBOResult(**result, streams=streams)


def _as_ranges(indices: List[int]) -> List[List[int]]:
    """
    Contiguous runs as half-open [start, end) pairs.

    Purging and the embargo both remove CONTIGUOUS blocks, so a training set
    is a union of a few runs however long the series is. Returning every
    index instead cost 37 KB for a 500-observation, 15-path split and would
    have grown linearly -- 360 KB at 5,000 observations, for a payload an
    agent then has to read back through a context window. The ranges carry
    the same information and are recovered with range(start, end).

    Found with array operations rather than a step per index: the splitter
    is linear in the length of the series, and a Python loop here would
    have become the slowest part of the call.
    """
    if not len(indices):
        return []
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    breaks = np.flatnonzero(np.diff(ordered) != 1)
    starts = ordered[np.concatenate(([0], breaks + 1))]
    ends = ordered[np.concatenate((breaks, [ordered.size - 1]))] + 1
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


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


def _reality_check_columns(input_data: RealityCheckInput, who: str):
    """
    returns_ref's columns in the roles the question gives them: the inline
    inputs they stand for, and the alignment report.
    """
    ref = input_data.returns_ref
    assert ref is not None
    panel = returns_ref_streams(ref, what=who)
    if input_data.question == "best_candidate_vs_benchmark":
        role = "candidate"
        named = _the_column(
            panel, input_data.benchmark_column, "benchmark", "benchmark_column", who
        )
    else:
        role = "alternative"
        named = _the_column(
            panel, input_data.strategy_column, "strategy", "strategy_column", who
        )
    if not [name for name in panel if name != named]:
        raise ValidationError(
            f"{who}: returns_ref has no column besides {named!r}, so there is "
            f"no {role} to test. Every other column is a {role}."
        )
    columns, streams, notes = _columns(panel, who, 30)
    others = {name: values for name, values in columns.items() if name != named}
    single = columns[named] if named is not None else None
    return others, single, streams, notes


def run_reality_check(input_data: RealityCheckInput) -> RealityCheckResult:
    who = "run_reality_check"
    streams = None
    notes: List[str] = []
    candidate_returns = input_data.candidate_returns
    benchmark_returns = input_data.benchmark_returns
    strategy_returns = input_data.strategy_returns
    alternative_returns = input_data.alternative_returns
    if input_data.returns_ref is not None:
        others, single, streams, notes = _reality_check_columns(input_data, who)
        if input_data.question == "best_candidate_vs_benchmark":
            candidate_returns, benchmark_returns = others, single
        else:
            alternative_returns, strategy_returns = others, single
    if input_data.question == "best_candidate_vs_benchmark":
        assert candidate_returns is not None
        candidates = _aligned_frame(candidate_returns, who)
        benchmark = None
        if benchmark_returns is not None:
            if len(benchmark_returns) != len(candidates):
                raise ValidationError(
                    f"{who}: benchmark_returns has "
                    f"{len(benchmark_returns)} values and each "
                    f"candidate {len(candidates)}. They must be aligned in "
                    "time, one value per period."
                )
            benchmark = pd.Series(benchmark_returns)
        result = lib.reality_check(
            question=input_data.question,
            candidate_returns=candidates,
            benchmark_returns=benchmark,
            method=input_data.method,
            n_bootstrap=input_data.n_bootstrap,
            block_size=input_data.block_size,
            seed=input_data.seed,
        )
    else:
        assert strategy_returns is not None
        assert alternative_returns is not None
        alternatives = _aligned_frame(alternative_returns, who)
        if len(strategy_returns) != len(alternatives):
            raise ValidationError(
                f"{who}: strategy_returns has {len(strategy_returns)} "
                f"values and each alternative {len(alternatives)}. They must "
                "be aligned in time, one value per period."
            )
        result = lib.reality_check(
            question=input_data.question,
            strategy_returns=pd.Series(strategy_returns),
            alternative_returns=alternatives,
            n_bootstrap=input_data.n_bootstrap,
            block_size=input_data.block_size,
            seed=input_data.seed,
        )
    result["warnings"] = notes + list(result.get("warnings", []))
    return RealityCheckResult(**result, streams=streams)


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
        "meaningless. Needs two or more configurations in trial_returns, one "
        "return series each; compare_models in the modeling runtime compares "
        "registered models.",
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
        "Is a result better than luck after a search, under the hypothesis "
        "named in question. The default, best_candidate_vs_benchmark, is "
        "White's Reality Check: does the best of every candidate searched "
        "beat one benchmark by more than the best of that many would by "
        "luck (method='hansen_spa' for Hansen's studentised form). "
        "beats_every_alternative asks whether one strategy beats EVERY "
        "alternative, and its p-value is the largest pairwise one. The "
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
