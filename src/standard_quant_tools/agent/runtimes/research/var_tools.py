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
  Christoffersen, the Basel traffic light and Acerbi-Szekely Z2. Or a
  model's own VaR forecasts: a lower quantile column of its predictions
  (`q01` is a 99% VaR) against the outcome beside it, each entity of a
  panel tested alone (see "a model's quantiles as VaR forecasts" below).

WHY BOTH ARE HERE, not the second in `backtest`. The backtest runtime's
listing is at its 32 KB budget with about 0.8 KB to spare, and the smallest
useful schema for the tester is over that, so registering it there made
`auto` thin `run_terminal_monte_carlo`, which it describes today. Here both
are larger than the largest tool `auto` still describes, so each is thinned
itself (its schema from `describe_tool`) and nothing else changes.

INLINE DATA OR A REFERENCE, NEVER TICKERS: a `returns_panel` reference or
{ISO date: return}, and the forecasts as the reference the first tool
publishes, as {ISO date: [return, VaR, ES]}, or as a `predictions`
reference. Nothing is fetched, so the same input gives the same bits. The
library functions are `metrics.var_forecast.forecast_var_series` and
`metrics.var_backtest.backtest_var_series`.
"""

from __future__ import annotations

import math
from typing import Annotated, Dict, List, Literal, Optional, Sequence

import numpy as np
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

#: The longest target horizon, in bars, a predictions frame is thinned for.
MAX_HORIZON = 1000

#: The most entities one call tests separately: a row each in `entities`.
MAX_ENTITIES = 1000

#: The column a predictions frame records each outcome's last bar in, as
#: `modeling.dataset.alignment.LABEL_END_COL` names it.
LABEL_END_COLUMN = "label_end_date"

#: The fields that read a predictions frame, refused without one.
_PREDICTION_FIELDS = ("quantile_column", "target_column", "entity", "horizon")

#: The FDR at which `entities_rejected_bh` counts an entity's Kupiec test.
_ENTITY_FDR = 0.05

#: A target whose lag-1 autocorrelation within entities is above this, with
#: no horizon known, is said to look overlapping: (h - 1) / h is what an
#: h-bar return sampled every bar has, so 0.5 is a two-bar target.
_OVERLAP_AUTOCORRELATION = 0.5

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
    predictions_ref: Optional[str] = Field(
        None,
        description="A model's predictions in place of forecasts: its "
        "quantile_column is the VaR, its target_column the outcome.",
    )
    quantile_column: Optional[str] = Field(
        None, description="The quantile tested: 'q01' is a 99% VaR."
    )
    target_column: Optional[str] = Field(
        None, description="The outcome the quantile was fitted for; 'target'."
    )
    entity: Optional[str] = Field(
        None, description="One entity's series; default each entity, pooled."
    )
    horizon: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_HORIZON,
        description="Bars the target spans, if the frame has no "
        "label_end_date; rows are thinned to every horizon-th date.",
    )

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
        if self.predictions_ref is not None:
            return self._predictions_source()
        given = [name for name in _PREDICTION_FIELDS if getattr(self, name) is not None]
        if given:
            raise ValueError(
                f"{given} read the columns of predictions_ref and were given "
                "without it. Drop them, or pass predictions_ref."
            )
        if self.forecasts is None and self.forecasts_ref is None:
            raise ValueError(
                f"{_WHO_BACKTEST} needs exactly one of forecasts / forecasts_ref "
                "/ predictions_ref; got neither. forecasts is the data inline, "
                "forecasts_ref forecast_var_series' reference, predictions_ref "
                "a model's quantiles with their outcomes."
            )
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

    def _predictions_source(self) -> "VarBacktestInput":
        both = [
            n for n in ("forecasts", "forecasts_ref") if getattr(self, n) is not None
        ]
        if both:
            raise ValueError(
                f"{_WHO_BACKTEST} needs exactly one of forecasts / forecasts_ref "
                f"/ predictions_ref; got predictions_ref and {both}. Test one "
                "series of forecasts per call."
            )
        if self.quantile_column is None:
            raise ValueError(
                "quantile_column is required with predictions_ref: it names "
                "the quantile that is the VaR ('q01' for 99%, 'q05' for 95%), "
                "and a frame of several quantiles does not say which is meant."
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


class VarEntityBacktest(ExplainsNulls):
    """One entity's line of a pooled predictions backtest: the same numbers
    a call with `entity` set gives for it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "christoffersen_p_value": "independence is undefined for this entity: "
        "no exception, nothing but exceptions, or no transition out of one "
        "state (a call with entity set says which)",
    }

    n_observations: int = 0
    n_exceptions: int = 0
    expected_exceptions: Stat = None
    kupiec_p_value: Stat = None
    kupiec_p_value_bh: Stat = Field(
        None, description="Benjamini-Hochberg adjusted across the entities."
    )
    christoffersen_p_value: Stat = None
    basel_zone: Optional[Literal["green", "yellow", "red"]] = None


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
    quantile_column: Optional[str] = Field(
        None, description="With predictions_ref: the quantile tested as the VaR."
    )
    horizon: Optional[int] = Field(
        None,
        description="With predictions_ref: bars the target spans. Above 1, "
        "every horizon-th date of each entity was kept, so no two outcomes "
        "tested overlap.",
    )
    n_rows_before_thinning: Optional[int] = Field(
        None, description="The rows tested before that thinning."
    )
    entities: Dict[str, VarEntityBacktest] = Field(
        default_factory=dict,
        description="predictions_ref over several entities, no entity named: "
        "each entity's series tested alone. The top-level counts are their "
        "sums; the tests are per entity only.",
    )
    entities_rejected_bh: Optional[int] = Field(
        None,
        description="Entities whose Kupiec p-value, Benjamini-Hochberg "
        "adjusted across the entities, is at most 0.05.",
    )
    zone_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Entities by Basel zone; 'none' where the light is "
        "undefined (not 99%, or under 250 forecasts).",
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
    if input_data.predictions_ref is not None:
        return _backtest_predictions(input_data)
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
    return _series_result(out)


def _series_result(out: Dict, notes: Sequence[str] = (), **extra) -> VarBacktestResult:
    """The result of one series' backtest, as the forecasts paths have always
    returned it; `notes` follow its own warnings and `extra` sets the
    predictions fields."""
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
        warnings=warnings + list(notes),
        **extra,
    )


# ── a model's quantiles as VaR forecasts ────────────────────────────────
#
# A model fitted with `ModelSpec.quantiles` writes each quantile of the
# outcome into its out-of-sample predictions as a column (`q01`, `q05`), and
# a lower quantile IS a VaR forecast: q01 of a return is the 99% VaR, as a
# loss -q01, and an exception is an outcome below it. Read here with the
# column detection `score_prediction_intervals` uses, so a column one tool
# recognises the other does too.
#
# ONE ENTITY IS ONE SERIES, AND ONLY ONE SERIES IS TESTED. Over a panel each
# entity's series is backtested on its own and the Kupiec p-values are
# Benjamini-Hochberg adjusted across the entities; the counts are summed,
# and no test is run on the pooled exceptions. Entities' exceptions on one
# date are not independent draws -- a selloff breaches every long VaR at
# once -- so the pooled count is far more dispersed than a binomial with
# that many trials, and a pooled Kupiec would reject a correct model for
# what is only the market moving together.
#
# OVERLAPPING OUTCOMES ARE THINNED. A target that spans h bars, sampled
# every bar, shares h - 1 bars with the next row's, so one bad week is five
# exceptions in a row whatever the model did, and Christoffersen reads that
# clustering as a failure. Every h-th date of each entity is kept (h from
# the frame's label_end_date, or `horizon`), so no two outcomes tested
# overlap, and the result says how many rows that left.


def _predictions_frame(input_data: VarBacktestInput):
    """The predictions frame with its quantile, outcome and level checked:
    (frame, quantile column, target column, confidence)."""
    from standard_quant_tools.modeling.agent.statistics_tools import (
        _detect_quantile_columns,
    )

    frame = handoff.resolve_frame(
        input_data.predictions_ref,
        expect="predictions",
        who=_WHO_BACKTEST,
        produced_by="A model fitted with quantiles writes them (q01, q05) into "
        "its out-of-sample predictions; the outcome goes beside them.",
    )
    columns = [str(c) for c in frame.columns]
    target = input_data.target_column or "target"
    if target not in frame.columns:
        raise ValidationError(
            f"{_WHO_BACKTEST}: predictions_ref has no {target!r} column to count "
            f"exceptions against; it holds {columns[:12]}. An exception is a "
            "REALIZED outcome below the quantile, so pass target_column=<the "
            "outcome the quantile was fitted for>, or publish the predictions "
            "with that outcome beside them."
        )
    levels = _detect_quantile_columns(frame)
    if not levels:
        raise ValidationError(
            f"{_WHO_BACKTEST}: predictions_ref carries no quantile column; it "
            f"holds {columns[:12]}. A VaR is a lower quantile of a continuous "
            "outcome, written as 'q01' or 'q05' by a regression model fitted "
            "with quantiles; a classification model's probabilities and a "
            "point prediction have none."
        )
    column = str(input_data.quantile_column)
    if column not in levels:
        raise ValidationError(
            f"{_WHO_BACKTEST}: quantile_column={column!r} is not a quantile "
            f"column of predictions_ref, which holds {sorted(levels)}. Pass one "
            "of those, or refit the model with that quantile."
        )
    level = levels[column]
    if level >= 0.5:
        raise ValidationError(
            f"{_WHO_BACKTEST}: {column!r} is the {level:.4g} quantile, not a "
            "lower tail. A VaR is the loss at a lower quantile of the outcome "
            "('q01' for 99%, 'q05' for 95%); pass one below the median."
        )
    confidence = round(1.0 - level, 12)
    given = input_data.confidence
    if given is not None and abs(given - confidence) > 1e-12:
        raise ValidationError(
            f"{_WHO_BACKTEST}: confidence={given} but {column!r} is the "
            f"{level:.4g} quantile, a {confidence:.4g} VaR. Counting its "
            "exceptions against another rate would test a claim the model never "
            "made; drop confidence, or pass the quantile fitted at that level."
        )
    for name in ("date",):
        if name not in frame.columns:
            raise ValidationError(
                f"{_WHO_BACKTEST}: predictions_ref has no 'date' column; it holds "
                f"{columns[:12]}. Each forecast is tested against the outcome of "
                "its own date."
            )
    try:
        outcome = pd.to_numeric(frame[target], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{_WHO_BACKTEST}: {target!r} holds values that are not numbers "
            f"({exc}); a VaR is tested against a realized return."
        ) from exc
    observed = set(np.unique(outcome[np.isfinite(outcome)]).tolist())
    if observed == {0.0, 1.0}:
        raise ValidationError(
            f"{_WHO_BACKTEST}: {target!r} holds only 0 and 1, a class label "
            "rather than a return. These are a classification model's "
            "predictions; a VaR is a quantile of a continuous outcome and a "
            "classifier fits none. Test a regression model fitted with "
            "quantiles."
        )
    quantile = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype="float64")
    for name, values in ((target, outcome.to_numpy()), (column, quantile)):
        n_inf = int(np.isinf(values).sum())
        if n_inf:
            raise ValidationError(
                f"{_WHO_BACKTEST}: {name!r} is infinite on {n_inf} row(s). An "
                "infinity is a failed computation upstream, not a forecast or "
                "an outcome; a row with none to test is left empty (NaN)."
            )
    if input_data.entity is not None and "entity" not in frame.columns:
        raise ValidationError(
            f"{_WHO_BACKTEST}: entity={input_data.entity!r} but predictions_ref "
            f"has no 'entity' column; it holds {columns[:12]}, one series. Drop "
            "entity."
        )
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="raise").to_numpy(),
            "realized": outcome.to_numpy(),
            "var": -quantile,
        }
    )
    has_entity = "entity" in frame.columns
    out["entity"] = frame["entity"].astype(str).to_numpy() if has_entity else ""
    if LABEL_END_COLUMN in frame.columns:
        out[LABEL_END_COLUMN] = pd.to_datetime(frame[LABEL_END_COLUMN]).to_numpy()
    return out, column, target, confidence


def _horizon(frame: pd.DataFrame, given: Optional[int]) -> Optional[int]:
    """Bars the target spans: counted from label_end_date on each entity's
    own dates where the frame has it, else `given`; None when neither says.

    Counted on the rows whose end date is itself a date of the frame, and
    the largest count taken: a fold boundary can drop dates between a row
    and its end, which only ever makes a count smaller."""
    if LABEL_END_COLUMN not in frame.columns:
        return given
    counted: List[int] = []
    for _name, rows in frame.groupby("entity", sort=False):
        dates = pd.DatetimeIndex(rows["date"]).sort_values()
        ends = pd.DatetimeIndex(rows[LABEL_END_COLUMN])
        found = dates.get_indexer(ends)
        start = dates.get_indexer(pd.DatetimeIndex(rows["date"]))
        exact = found >= 0
        if exact.any():
            counted.append(int(np.max(found[exact] - start[exact])))
    if not counted:
        if given is not None:
            return given
        raise ValidationError(
            f"{_WHO_BACKTEST}: predictions_ref has a {LABEL_END_COLUMN} column "
            "but no row's end date is one of its dates, so the target's span "
            "cannot be counted in bars. Pass horizon=<bars the target spans>."
        )
    found_horizon = max(max(counted), 1)
    if given is not None and given != found_horizon:
        raise ValidationError(
            f"{_WHO_BACKTEST}: horizon={given} but the frame's "
            f"{LABEL_END_COLUMN} puts each outcome {found_horizon} bar(s) after "
            "its date. Drop horizon to use the frame's, or check which target "
            "these predictions are for."
        )
    return found_horizon


def _lag_one_autocorrelation(frame: pd.DataFrame) -> Optional[float]:
    """The outcome's lag-1 autocorrelation within entities, pooled over the
    pairs of consecutive dates; sums only, so no BLAS thread count reaches
    it."""
    left: List[np.ndarray] = []
    right: List[np.ndarray] = []
    for _name, rows in frame.groupby("entity", sort=True):
        values = rows.sort_values("date")["realized"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 2:
            left.append(values[:-1] - values[:-1].mean())
            right.append(values[1:] - values[1:].mean())
    if not left:
        return None
    x, y = np.concatenate(left), np.concatenate(right)
    scale = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return float(np.sum(x * y)) / scale if scale > 0 else None


def _bars(rho: float) -> float:
    """The span h whose overlapping returns have lag-1 autocorrelation rho,
    (h - 1) / h = rho, held below the longest horizon thinned for."""
    return min(1.0 / (1.0 - rho), float(MAX_HORIZON)) if rho < 1 else MAX_HORIZON


def _entity_series(frame: pd.DataFrame, horizon: int):
    """Each entity's (realized, var) by date, thinned to every horizon-th
    date of its own, then without the rows that have no outcome or quantile:
    ({entity: frame}, rows before thinning, rows without one)."""
    series: Dict[str, pd.DataFrame] = {}
    before = 0
    missing = 0
    for name, rows in frame.groupby("entity", sort=True):
        rows = rows.sort_values("date")
        before += len(rows)
        kept = rows.iloc[::horizon]
        usable = kept[np.isfinite(kept["realized"]) & np.isfinite(kept["var"])]
        missing += len(kept) - len(usable)
        series[str(name)] = pd.DataFrame(
            {
                "realized": usable["realized"].to_numpy(),
                "var": usable["var"].to_numpy(),
            },
            index=pd.DatetimeIndex(usable["date"]),
        )
    return series, before, missing


def _backtest_predictions(input_data: VarBacktestInput) -> VarBacktestResult:
    frame, column, target, confidence = _predictions_frame(input_data)
    repeated = frame.duplicated(subset=["entity", "date"])
    if repeated.any():
        sample = frame.loc[repeated, ["entity", "date"]].head(3)
        raise ValidationError(
            f"{_WHO_BACKTEST}: predictions_ref has {int(repeated.sum())} repeated "
            f"(entity, date) row(s), e.g. {sample.astype(str).to_dict('records')}. "
            "One forecast per entity and date: a frame from combinatorial "
            "cross-validation predicts a date once per path; test one path."
        )
    if input_data.entity is not None:
        entities = sorted(set(frame["entity"]))
        if input_data.entity not in entities:
            raise ValidationError(
                f"{_WHO_BACKTEST}: entity={input_data.entity!r} is not in "
                f"predictions_ref, whose entities include {entities[:8]}."
            )
    horizon = _horizon(frame, input_data.horizon)
    notes: List[str] = []
    if horizon is None:
        rho = _lag_one_autocorrelation(frame)
        if rho is not None and rho > _OVERLAP_AUTOCORRELATION:
            notes.append(
                f"{target!r} has a lag-1 autocorrelation of {rho:.2f} within "
                f"entities, what a target spanning about {_bars(rho):.0f} "
                "bars sampled every bar has. If it does, its outcomes overlap "
                "and its exceptions cluster by construction, which "
                "Christoffersen reads as a failure: pass horizon to thin them."
            )
        horizon = 1
    if input_data.entity is not None:
        frame = frame[frame["entity"] == input_data.entity]
    series, before, missing = _entity_series(frame, horizon)
    if len(series) > MAX_ENTITIES:
        raise ValidationError(
            f"{_WHO_BACKTEST}: predictions_ref holds {len(series)} entities, more "
            f"than the {MAX_ENTITIES} one call tests separately. Name one with "
            "entity, or publish a narrower frame."
        )
    tested = sum(len(s) for s in series.values())
    if horizon > 1:
        notes.append(
            f"The target spans {horizon} bars, so consecutive outcomes "
            f"overlap; every {horizon}th date of each entity was kept, "
            f"{tested + missing} of {before} rows, so no two outcomes tested "
            "share a bar."
        )
    if missing:
        notes.append(
            f"{missing} row(s) have no {target!r} or {column!r} (an outcome "
            "not yet realized, at the end of a sample) and were not tested."
        )
    extra = dict(quantile_column=column, horizon=horizon, n_rows_before_thinning=before)
    if len(series) == 1:
        (only,) = series.values()
        out = backtest_lib.backtest_var_series(
            only,
            confidence,
            random_seed=input_data.random_seed,
            n_simulations=input_data.n_simulations,
            who=_WHO_BACKTEST,
        )
        return _series_result(out, notes, **extra)
    return _pooled_result(series, confidence, input_data, notes, extra)


def _pooled_result(
    series: Dict[str, pd.DataFrame],
    confidence: float,
    input_data: VarBacktestInput,
    notes: List[str],
    extra: Dict,
) -> VarBacktestResult:
    """
    Each entity tested alone, the Kupiec p-values Benjamini-Hochberg
    adjusted across them, and the counts summed.

    NEVER ONE KUPIEC TEST OVER THE POOLED EXCEPTIONS. Kupiec's likelihood
    ratio is the binomial's: n independent trials at rate p. The entities of
    a panel are not independent on a date -- a selloff breaches every long
    VaR at once, a calm month breaches none -- so the pooled count of n x
    entities forecasts varies far more than that binomial says, and the
    test rejects a correct model for the cross-section moving together.
    Each entity's own series is a series of one-step forecasts that can be
    independent; across entities the family is adjusted for being many
    tests, and the zones are counted.
    """
    from standard_quant_tools.modeling.validation.comparison import bh_adjust

    outs: Dict[str, Dict] = {}
    short: List[str] = []
    for name, one in series.items():
        if len(one) < 2:
            short.append(name)
            continue
        outs[name] = backtest_lib.backtest_var_series(
            one,
            confidence,
            random_seed=input_data.random_seed,
            n_simulations=input_data.n_simulations,
            who=_WHO_BACKTEST,
        )
    if not outs:
        raise ValidationError(
            f"{_WHO_BACKTEST}: no entity of predictions_ref has two forecasts with "
            "an outcome to test, after thinning; a backtest needs at least two."
        )
    names = list(outs)
    adjusted = bh_adjust([outs[n]["kupiec"]["p_value"] for n in names])
    rows: Dict[str, VarEntityBacktest] = {}
    zones: Dict[str, int] = {}
    for name, p_adjusted in zip(names, adjusted):
        out = outs[name]
        zone = out["basel"]["zone"]
        zones[zone or "none"] = zones.get(zone or "none", 0) + 1
        rows[name] = VarEntityBacktest(
            n_observations=out["n_observations"],
            n_exceptions=out["n_exceptions"],
            expected_exceptions=out["expected_exceptions"],
            kupiec_p_value=out["kupiec"]["p_value"],
            kupiec_p_value_bh=p_adjusted,
            christoffersen_p_value=out["christoffersen"]["p_value"],
            basel_zone=zone,
        )
    n_obs = sum(out["n_observations"] for out in outs.values())
    n_exceptions = sum(out["n_exceptions"] for out in outs.values())
    raw = sum(1 for out in outs.values() if out["kupiec"]["p_value"] < _ENTITY_FDR)
    rejected = sum(1 for p in adjusted if p <= _ENTITY_FDR)
    p = 1.0 - confidence
    thin = [n for n in names if outs[n]["n_observations"] * p < 5]
    warnings = [
        f"{len(outs)} entities, each tested on its own series: the kupiec, "
        "christoffersen, conditional-coverage, basel and z2 fields are per "
        "entity, in entities, and null here. No test is run on the pooled "
        "exceptions: entities breach their VaR together on a bad day, so the "
        "pooled count is no binomial and a pooled Kupiec would reject a "
        "correct model for the market moving together.",
        f"Kupiec rejects at 5% for {raw} of {len(outs)} entities before "
        f"adjustment and for {rejected} after Benjamini-Hochberg across them "
        f"(about {len(outs) * _ENTITY_FDR:.1f} would reject at 5% by chance "
        "before adjustment if every model were right).",
        "exception_dates is empty: the dates are per entity; call with "
        "entity set for one entity's.",
    ]
    if thin:
        warnings.append(
            f"{len(thin)} entities expect fewer than 5 exceptions, so their "
            "p-values are rough and only a large error can reject."
        )
    if short:
        warnings.append(
            f"{len(short)} entities have fewer than two forecasts with an "
            f"outcome after thinning and were not tested: {short[:5]}."
        )
    first = min(out["first_date"] for out in outs.values())
    last = max(out["last_date"] for out in outs.values())
    return VarBacktestResult(
        confidence=confidence,
        n_observations=n_obs,
        first_date=_iso(first),
        last_date=_iso(last),
        n_exceptions=n_exceptions,
        expected_exceptions=n_obs * p,
        exception_rate=n_exceptions / n_obs,
        entities=rows,
        entities_rejected_bh=rejected,
        zone_counts=dict(sorted(zones.items())),
        warnings=warnings + notes,
        **extra,
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
        "the reason. Also a model's lower quantile as its VaR (q01 is 99%) "
        "from a predictions reference with the outcome: each entity tested "
        "alone, Kupiec BH-adjusted across entities, a multi-bar target "
        "thinned to non-overlapping dates.",
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
