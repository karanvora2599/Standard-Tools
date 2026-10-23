"""
Value at risk as a forecast that can be checked: a one-step-ahead VaR and ES
series, and the tests that say whether a series of forecasts was right.

- `forecast_var_series` -- for one return stream and a rolling window, the
  VaR and ES of every date after the first window, each from the returns
  before it, by one of six methods; and the forecast for the period after
  the last return. With `run_id` the series is published as an
  `analytic_frame` (realized, var, es, confidence by date).
- `backtest_var_forecasts` -- that series, or dated forecasts from anywhere,
  against the returns that followed: the exception dates, Kupiec,
  Christoffersen, the Basel traffic light and Acerbi-Szekely Z2.

WHY BOTH ARE HERE, not the second in `backtest`. The backtest runtime's
listing is at its 32 KB budget with about 0.8 KB to spare, and the smallest
useful schema for the tester is over that, so registering it there made
`auto` thin `run_terminal_monte_carlo`, which it describes today. Here both
are larger than the largest tool `auto` still describes, so each is thinned
itself (its schema from `describe_tool`) and nothing else changes.

INLINE DATA OR A REFERENCE, NEVER TICKERS: a `returns_panel` reference or
{ISO date: return}, and the forecasts as the reference the first tool
publishes or as {ISO date: [return, VaR, ES]}. Nothing is fetched, so the
same input gives the same bits. The library functions are
`metrics.var_forecast.forecast_var_series` and
`metrics.var_backtest.backtest_var_series`.
"""

from __future__ import annotations

import math
from typing import Annotated, Dict, List, Literal, Optional

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.models import Seed
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.agent.runtimes.streams import (
    ReturnStream,
    columns_only_with_ref,
    exactly_one_source,
    resolve_return_streams,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics import var_backtest as backtest_lib
from standard_quant_tools.metrics import var_forecast as forecast_lib

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

#: The longest series one call takes, held in the validators rather than
#: advertised: a guard against a runaway payload, not a planning number.
MAX_OBSERVATIONS = 100_000

#: The columns of the published forecast frame, in order.
FORECAST_COLUMNS = ("realized", "var", "es", "confidence")

_WHO_FORECAST = "forecast_var_series"
_WHO_BACKTEST = "backtest_var_forecasts"


def _iso(stamp: pd.Timestamp) -> str:
    stamp = pd.Timestamp(stamp)
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


# ── inputs ──────────────────────────────────────────────────────────────


class VarForecastInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference, read as one dated stream."
    )
    column: Optional[str] = Field(
        None, description="returns_ref's column, if it has several."
    )
    returns: Optional[Dict[str, float]] = Field(
        None, description="{ISO date: return} inline, in place of returns_ref."
    )
    method: Literal[
        "historical",
        "normal",
        "cornish_fisher",
        "ewma",
        "filtered_historical",
        "evt_garch",
    ] = "historical"
    confidence: float = Field(0.99, gt=0.5, lt=1)
    window: int = Field(
        250,
        ge=20,
        description="Returns per forecast, all before its date; 100+ for the "
        "GARCH methods, ewma's warm-up.",
    )
    ewma_lambda: float = Field(0.94, gt=0, lt=1, description="ewma's decay.")
    tail_fraction: float = Field(
        0.1,
        gt=0,
        lt=0.5,
        description="evt_garch: share of standardized losses the tail is fitted to.",
    )
    refit_every: int = Field(
        20, ge=1, description="GARCH methods: refit every this many forecasts."
    )
    run_id: Optional[str] = Field(
        None,
        description="Publishes the series as forecasts_ref, which "
        "backtest_var_forecasts reads.",
    )
    name: str = Field("var_forecasts", description="Its name within run_id.")

    @model_validator(mode="before")
    @classmethod
    def _bounded(cls, data):
        if isinstance(data, dict):
            values = data.get("returns")
            if isinstance(values, dict) and len(values) > MAX_OBSERVATIONS:
                raise ValueError(
                    f"returns has {len(values)} dates, more than the "
                    f"{MAX_OBSERVATIONS} one call takes. Forecast a shorter span."
                )
            window = data.get("window")
            if isinstance(window, int) and window > MAX_OBSERVATIONS:
                raise ValueError(
                    f"window={window} is longer than any series one call takes "
                    f"({MAX_OBSERVATIONS})."
                )
            refit = data.get("refit_every")
            if isinstance(refit, int) and refit > MAX_OBSERVATIONS:
                raise ValueError(
                    f"refit_every={refit} is longer than any series one call "
                    f"takes ({MAX_OBSERVATIONS}); a GARCH fitted once is "
                    f"refit_every={MAX_OBSERVATIONS}."
                )
        return data

    @field_validator("returns")
    @classmethod
    def _dated_and_finite(
        cls, values: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """A return stream's own rules, checked here so they are refused at
        the boundary: every key a date, every value finite."""
        if values is not None:
            ReturnStream(values=values)
        return values

    @model_validator(mode="after")
    def _one_stream(self) -> "VarForecastInput":
        exactly_one_source(self, "returns", "returns_ref", _WHO_FORECAST)
        columns_only_with_ref(self, "column", "returns_ref")
        for field in ("confidence", "ewma_lambda", "tail_fraction"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number, got {value!r}.")
        return self


def _finite_rows(rows: Dict[str, List[float]]) -> Dict[str, List[float]]:
    lengths = {len(row) for row in rows.values()}
    if not lengths <= {2, 3} or len(lengths) != 1:
        raise ValueError(
            "every forecast is [return, VaR] or [return, VaR, ES], all of one "
            f"length; got lists of lengths {sorted(lengths)}."
        )
    bad = [d for d, row in rows.items() if not all(math.isfinite(x) for x in row)]
    if bad:
        raise ValueError(
            f"forecasts are not finite on {bad[:5]}; a date with no forecast is "
            "left out, not given a NaN."
        )
    return rows


class VarBacktestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forecasts_ref: Optional[str] = Field(
        None,
        description="forecast_var_series' forecasts_ref (an analytic_frame of "
        "realized, var, es, confidence by date).",
    )
    forecasts: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Inline: {ISO date: [return, VaR] or [return, VaR, ES]}, "
        "VaR and ES as positive losses.",
    )
    confidence: Optional[float] = Field(
        None,
        gt=0.5,
        lt=1,
        description="The forecasts' VaR level: required inline, read from "
        "forecasts_ref (and checked against it if given).",
    )
    n_simulations: int = Field(
        10_000, ge=100, le=100_000, description="Draws for Z2's p-value."
    )
    random_seed: Seed = Field(0, description="Seeds Z2's simulated p-value.")

    @model_validator(mode="before")
    @classmethod
    def _bounded(cls, data):
        if isinstance(data, dict):
            rows = data.get("forecasts")
            if isinstance(rows, dict) and len(rows) > MAX_OBSERVATIONS:
                raise ValueError(
                    f"forecasts has {len(rows)} dates, more than the "
                    f"{MAX_OBSERVATIONS} one call takes."
                )
        return data

    @model_validator(mode="after")
    def _one_source(self) -> "VarBacktestInput":
        exactly_one_source(self, "forecasts", "forecasts_ref", _WHO_BACKTEST)
        if self.forecasts is not None:
            _finite_rows(self.forecasts)
            if self.confidence is None:
                raise ValueError(
                    "confidence is required with inline forecasts: an exception "
                    "is counted against the rate 1 - confidence, and nothing in "
                    "the numbers says what it was. Pass the VaR level, e.g. 0.99."
                )
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence!r}.")
        return self


# ── results ─────────────────────────────────────────────────────────────


class VarForecastResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "next_es": "the fitted tail's shape is 1 or more, so the tail has no "
        "mean and the expected shortfall does not exist",
        "mean_es": "an ES in the series does not exist (the fitted tail's "
        "shape is 1 or more there)",
    }

    method: str = ""
    confidence: Stat = None
    window: int = 0
    n_returns: int = Field(0, description="Dated returns in the stream.")
    n_forecasts: int = Field(
        0, description="Dates forecast: every return after the first window."
    )
    first_forecast_date: Optional[str] = None
    last_forecast_date: Optional[str] = None
    next_var: Stat = Field(
        None,
        description="VaR for the period after the last return, from the last "
        "`window` returns: a loss, positive.",
    )
    next_es: Stat = Field(None, description="ES for that period, a loss.")
    mean_var: Stat = Field(None, description="Mean VaR over the forecast dates.")
    mean_es: Stat = None
    forecasts_ref: Optional[str] = Field(
        None,
        description="With run_id: the analytic_frame of realized, var, es and "
        "confidence by date, for backtest_var_forecasts.",
    )
    diagnostics: Dict[str, Stat] = Field(
        default_factory=dict,
        description="The method's own: min_tail_observations (historical, "
        "filtered_historical: the fewest returns an ES averages); "
        "garch_fits, garch_not_converged, garch_persistence (last fit), "
        "refit_every (GARCH methods); min_exceedances, gpd_shape_min/max, "
        "n_es_unbounded (evt_garch); skewness and excess_kurtosis min/max "
        "(cornish_fisher); ewma_lambda and ewma_seed_variance (ewma).",
    )
    warnings: List[str] = Field(default_factory=list)


class VarBacktestResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")

    confidence: Stat = None
    n_observations: int = Field(0, description="Forecast dates tested.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    n_exceptions: int = Field(0, description="Dates whose return was below -VaR.")
    expected_exceptions: Stat = Field(
        None, description="n_observations x (1 - confidence)."
    )
    exception_rate: Stat = None
    exception_dates: List[str] = Field(
        default_factory=list, description="The exception series: every date."
    )
    kupiec_lr: Stat = Field(
        None, description="Kupiec's proportion-of-failures LR, chi-squared(1)."
    )
    kupiec_p_value: Stat = None
    christoffersen_lr: Stat = Field(
        None,
        description="Christoffersen's independence LR (does an exception make "
        "the next likelier?), chi-squared(1).",
    )
    christoffersen_p_value: Stat = None
    transitions: Dict[str, int] = Field(
        default_factory=dict,
        description="n00, n01, n10, n11: dates in state i followed by state j "
        "(1 an exception).",
    )
    conditional_coverage_lr: Stat = Field(
        None, description="kupiec_lr + christoffersen_lr, chi-squared(2)."
    )
    conditional_coverage_p_value: Stat = None
    basel_zone: Optional[Literal["green", "yellow", "red"]] = Field(
        None,
        description="The traffic light on the last 250 forecasts at 99%: 0-4 "
        "exceptions green, 5-9 yellow, 10+ red.",
    )
    basel_exceptions: Optional[int] = Field(
        None, description="Exceptions in those 250."
    )
    basel_plus_factor: Stat = Field(
        None, description="0 green; 0.40-0.85 yellow; 1.00 red."
    )
    basel_multiplier: Stat = Field(None, description="3 + basel_plus_factor.")
    basel_first_date: Optional[str] = None
    z2: Stat = Field(
        None,
        description="Acerbi-Szekely Z2 for ES: 0 when the forecasts are right, "
        "negative when the tail was understated.",
    )
    z2_p_value: Stat = Field(
        None, description="Share of Z2 simulated under the forecasts below z2."
    )
    z2_simulations: Optional[int] = None
    z2_null_distribution: Optional[str] = Field(
        None, description="What each date's return is drawn from for the p-value."
    )
    warnings: List[str] = Field(default_factory=list)


# ── tools ───────────────────────────────────────────────────────────────


def _stream(input_data: VarForecastInput) -> pd.Series:
    """The one return stream, resolved and checked as every stream is:
    dated, sorted, finite, and returns rather than levels."""
    from standard_quant_tools.portfolio.streams import align_return_streams

    if input_data.returns_ref is not None:
        stream = ReturnStream(ref=input_data.returns_ref, column=input_data.column)
    else:
        stream = ReturnStream(values=input_data.returns)
    resolved = resolve_return_streams({"returns": stream}, what=_WHO_FORECAST)
    frame, _report = align_return_streams(resolved, min_overlap=1, what=_WHO_FORECAST)
    return frame["returns"]


def forecast_var_series(input_data: VarForecastInput) -> VarForecastResult:
    returns = _stream(input_data)
    if len(returns) > MAX_OBSERVATIONS:
        raise ValidationError(
            f"{_WHO_FORECAST}: returns_ref holds {len(returns)} returns, more than "
            f"the {MAX_OBSERVATIONS} one call forecasts. Forecast a shorter span."
        )
    out = forecast_lib.forecast_var_series(
        returns,
        method=input_data.method,
        confidence=input_data.confidence,
        window=input_data.window,
        ewma_lambda=input_data.ewma_lambda,
        tail_fraction=input_data.tail_fraction,
        refit_every=input_data.refit_every,
        who=_WHO_FORECAST,
    )
    warnings = list(out["warnings"])
    ref = None
    if input_data.run_id is not None:
        ref = handoff.publish(
            out["forecasts"],
            kind="analytic_frame",
            run_id=input_data.run_id,
            name=input_data.name,
            producer=f"research.{_WHO_FORECAST}",
        )
    else:
        warnings.append(
            "The series was not published: pass run_id to get forecasts_ref, "
            "which backtest_var_forecasts tests."
        )
    return VarForecastResult(
        method=out["method"],
        confidence=out["confidence"],
        window=out["window"],
        n_returns=out["n_returns"],
        n_forecasts=out["n_forecasts"],
        first_forecast_date=_iso(out["first_forecast_date"]),
        last_forecast_date=_iso(out["last_forecast_date"]),
        next_var=out["next_var"],
        next_es=out["next_es"],
        mean_var=out["mean_var"],
        mean_es=out["mean_es"] if out["mean_es"] is not None else float("nan"),
        forecasts_ref=ref,
        diagnostics=out["diagnostics"],
        warnings=warnings,
    )


def _inline_frame(rows: Dict[str, List[float]]) -> pd.DataFrame:
    from standard_quant_tools.agent.runtimes._shared import parse_date_keys

    parsed = parse_date_keys(rows, "forecasts", _WHO_BACKTEST)
    columns = list(FORECAST_COLUMNS[: len(next(iter(parsed.values())))])
    frame = pd.DataFrame(
        list(parsed.values()),
        index=pd.DatetimeIndex(list(parsed)),
        columns=columns,
        dtype="float64",
    )
    return frame.sort_index()


def _referenced_frame(ref: str, confidence: Optional[float]):
    """The published forecasts and their level: from the frame's own
    `confidence` column, checked against the one given."""
    frame = handoff.resolve_frame(
        ref,
        expect="analytic_frame",
        who=_WHO_BACKTEST,
        produced_by="forecast_var_series publishes one with run_id.",
    )
    missing = [c for c in ("realized", "var") if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"{_WHO_BACKTEST}: forecasts_ref has no {missing} column; its columns "
            f"are {[str(c) for c in frame.columns][:8]}. It needs 'realized' and "
            "'var' (and 'es' to test ES), as forecast_var_series publishes."
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    level = confidence
    if "confidence" in frame.columns:
        levels = sorted(set(frame["confidence"].dropna().astype(float)))
        if len(levels) != 1:
            raise ValidationError(
                f"{_WHO_BACKTEST}: forecasts_ref mixes VaR levels {levels[:5]}; "
                "one backtest is of one level. Test each level's series alone."
            )
        if confidence is not None and abs(levels[0] - confidence) > 1e-12:
            raise ValidationError(
                f"{_WHO_BACKTEST}: confidence={confidence} but forecasts_ref was "
                f"forecast at {levels[0]}. Counting its exceptions against another "
                "rate would test a claim the forecasts never made; drop "
                "confidence, or pass the series forecast at that level."
            )
        level = levels[0]
    if level is None:
        raise ValidationError(
            f"{_WHO_BACKTEST}: forecasts_ref carries no confidence column, so "
            "the VaR level is not known. Pass confidence."
        )
    keep = [c for c in ("realized", "var", "es") if c in frame.columns]
    return frame[keep].sort_index(), float(level)


def backtest_var_forecasts(input_data: VarBacktestInput) -> VarBacktestResult:
    if input_data.forecasts_ref is not None:
        frame, confidence = _referenced_frame(
            input_data.forecasts_ref, input_data.confidence
        )
    else:
        assert input_data.forecasts is not None and input_data.confidence is not None
        frame = _inline_frame(input_data.forecasts)
        confidence = float(input_data.confidence)
    out = backtest_lib.backtest_var_series(
        frame,
        confidence,
        random_seed=input_data.random_seed,
        n_simulations=input_data.n_simulations,
        who=_WHO_BACKTEST,
    )
    warnings = list(out["warnings"])
    christoffersen = out["christoffersen"]
    if christoffersen["null_reason"]:
        warnings.append(
            "christoffersen_lr, christoffersen_p_value and the conditional "
            f"coverage are null: {christoffersen['null_reason']}."
        )
    basel = out["basel"]
    if basel["null_reason"]:
        warnings.append(f"The basel_* fields are null: {basel['null_reason']}.")
    z2 = out["z2"]
    if z2["null_reason"]:
        warnings.append(f"z2 and z2_p_value are null: {z2['null_reason']}.")
    return VarBacktestResult(
        confidence=out["confidence"],
        n_observations=out["n_observations"],
        first_date=_iso(out["first_date"]),
        last_date=_iso(out["last_date"]),
        n_exceptions=out["n_exceptions"],
        expected_exceptions=out["expected_exceptions"],
        exception_rate=out["exception_rate"],
        exception_dates=[_iso(d) for d in out["exception_dates"]],
        kupiec_lr=out["kupiec"]["lr"],
        kupiec_p_value=out["kupiec"]["p_value"],
        christoffersen_lr=christoffersen["lr"],
        christoffersen_p_value=christoffersen["p_value"],
        transitions={k: christoffersen[k] for k in ("n00", "n01", "n10", "n11")},
        conditional_coverage_lr=out["conditional_coverage"]["lr"],
        conditional_coverage_p_value=out["conditional_coverage"]["p_value"],
        basel_zone=basel["zone"],
        basel_exceptions=basel["exceptions"],
        basel_plus_factor=basel["plus_factor"],
        basel_multiplier=basel["multiplier"],
        basel_first_date=(
            _iso(basel["first_date"]) if basel["first_date"] is not None else None
        ),
        z2=z2["z2"],
        z2_p_value=z2["p_value"],
        z2_simulations=z2["n_simulations"],
        z2_null_distribution=z2["null_distribution"],
        warnings=warnings,
    )


VAR_TOOL_DEFS = [
    (
        "forecast_var_series",
        "One-step-ahead VaR and expected shortfall for every date of a return "
        "stream, each from the rolling window before it, and the forecast for "
        "the next period. Methods: historical, normal, cornish_fisher (refused "
        "outside the region where the expansion is a quantile), ewma "
        "(RiskMetrics), filtered_historical and evt_garch (GARCH-filtered, the "
        "second with a generalized Pareto tail). With run_id the series is "
        "published for backtest_var_forecasts. Losses are positive.",
        VarForecastInput,
    ),
    (
        "backtest_var_forecasts",
        "Whether a series of VaR and ES forecasts was right against the "
        "returns that followed: Kupiec, Christoffersen, the Basel traffic light "
        "and Acerbi-Szekely Z2. Returns the exception dates, the coverage, "
        "independence and conditional-coverage tests, the zone with its plus "
        "factor on the last 250 forecasts at 99%, and Z2 for ES with a p-value "
        "simulated under the forecasts. Reads forecast_var_series' reference "
        "or dated forecasts inline; a statistic that is undefined is null with "
        "the reason.",
        VarBacktestInput,
    ),
]

VAR_TOOL_DISPATCH = {
    "forecast_var_series": (forecast_var_series, VarForecastInput),
    "backtest_var_forecasts": (backtest_var_forecasts, VarBacktestInput),
}

__all__ = [
    "VAR_TOOL_DEFS",
    "VAR_TOOL_DISPATCH",
    "VarBacktestInput",
    "VarBacktestResult",
    "VarForecastInput",
    "VarForecastResult",
    "backtest_var_forecasts",
    "forecast_var_series",
]
