"""
Reference-native research tools: the same arithmetic, on data from anywhere.

WHY THESE EXIST ALONGSIDE THE TICKER TOOLS. `analyze_stock_risk` computes a
Sharpe, and it can only compute one for a symbol this library can fetch.
The same question asked of a model's out-of-sample returns, an external
fund's monthly series, or a panel another agent already published had no
tool at all -- and the wrong fix is `calculate_sharpe_from_returns` beside
`calculate_sharpe`, which is how a surface ends up answering one question
under three names.

These take a `DataSource`: exactly one of a symbol, an `sqt://` reference,
or inline values. The tool is the QUESTION; the input says where the bytes
are.

THE EXISTING TICKER TOOLS ARE UNCHANGED. This is an addition, not a
migration -- `analyze_stock_risk` still takes `symbol=` and still means
what it meant.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools import metrics as M
from standard_quant_tools.agent.runtimes.data.models import DataSource, resolve_source
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.panel import technical_indicators_panel
from standard_quant_tools.portfolio.portfolio import fetch_ohlcv_panel_sync

from ..handoff import publish

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

#: name -> (callable, needs_equity_curve). Closed on purpose: this surface
#: is reachable from an agent, and an eval-shaped hole that accepted an
#: arbitrary expression would be a remote-code path wearing a statistics
#: costume.
_METRICS = {
    "cumulative_return": (M.cumulative_return, False),
    # `cagr` calls `cumulative_return`, and `calmar_ratio`'s own parameter
    # is named `equity_curve`. Registered False, they were handed a RETURN
    # series: cagr came back -0.4876 where the truth is +0.2337, sign
    # flipped, and calmar -0.218 against +1.650 -- both next to a
    # max_drawdown in the same response that was correct.
    "cagr": (M.cagr, True),
    "annualized_volatility": (M.annualized_volatility, False),
    "sharpe_ratio": (M.sharpe_ratio, False),
    "sortino_ratio": (M.sortino_ratio, False),
    "calmar_ratio": (M.calmar_ratio, True),
    "var_historical": (M.var_historical, False),
    "var_parametric": (M.var_parametric, False),
    "cvar": (M.cvar, False),
    "max_drawdown": (M.max_drawdown, True),
}

METRIC_NAMES = tuple(_METRICS)


class SeriesMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: DataSource = Field(
        ..., description="The RETURN series: a symbol, a reference, or values."
    )
    metrics: List[str] = Field(
        default_factory=lambda: ["sharpe_ratio", "max_drawdown"],
        description=f"Any of: {', '.join(METRIC_NAMES)}.",
    )
    risk_free_rate: float = Field(
        0.0,
        description=(
            "ANNUAL risk-free rate. Divided by periods_per_year internally, "
            "so do not pre-divide it."
        ),
    )
    periods_per_year: int = Field(
        252, gt=0, description="252 for daily, 12 for monthly."
    )


class SeriesMetricsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_observations: int = 0
    periods_per_year: int = 252
    values: Dict[str, Stat] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


def calculate_series_metrics(input_data: SeriesMetricsInput) -> SeriesMetricsResult:
    """Risk and return metrics for ANY return series, not just a ticker."""
    unknown = sorted(set(input_data.metrics) - set(_METRICS))
    if unknown:
        raise ValidationError(
            f"unknown metric(s) {unknown}; expected any of "
            f"{sorted(_METRICS)}. The set is closed rather than open "
            "because this surface is reachable from an agent."
        )
    if not input_data.metrics:
        raise ValidationError("no metrics requested; name at least one.")

    returns = resolve_source(input_data.series, what="calculate_series_metrics")
    equity = (1.0 + returns).cumprod()

    warnings: List[str] = []
    if len(returns) < 30:
        warnings.append(
            f"only {len(returns)} observations. A Sharpe on this little data "
            "has a standard error comparable to the estimate itself -- read "
            "it as a direction, not a number."
        )

    values: Dict[str, Any] = {}
    for name in input_data.metrics:
        fn, wants_equity = _METRICS[name]
        try:
            # `calmar_ratio` is NOT in this branch: it takes no
            # risk_free_rate, so it raised TypeError and fell through to a
            # handler that re-called it on the return series.
            if name in ("sharpe_ratio", "sortino_ratio"):
                values[name] = fn(
                    returns,
                    risk_free_rate=input_data.risk_free_rate,
                    periods_per_year=input_data.periods_per_year,
                )
            elif wants_equity:
                values[name] = fn(equity)
            elif name == "annualized_volatility":
                values[name] = fn(returns, periods_per_year=input_data.periods_per_year)
            elif name in ("cagr", "calmar_ratio"):
                values[name] = fn(equity, periods_per_year=input_data.periods_per_year)
            else:
                values[name] = fn(returns)
        except TypeError:
            # A metric whose signature does not take the annualization
            # arguments; call it plainly rather than guessing at kwargs.
            values[name] = fn(equity if wants_equity else returns)

    return SeriesMetricsResult(
        n_observations=int(len(returns)),
        periods_per_year=input_data.periods_per_year,
        values=values,
        warnings=warnings,
    )


class IndicatorPanelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: List[str] = Field(..., min_length=1)
    start_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    end_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    indicators: List[str] = Field(
        ...,
        min_length=1,
        description=("Any of: rsi, adx, atr, bollinger_bands, stochastic_oscillator."),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names this artifact within the run.")
    price_panel_ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://price_panel/...` from the data runtime. Given one, "
            "nothing is refetched -- the same bars are reused."
        ),
    )


class IndicatorPanelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refs: Dict[str, str] = Field(
        default_factory=dict, description="indicator -> `sqt://` reference."
    )
    indicators: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    rows: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


def compute_indicator_panel(input_data: IndicatorPanelInput) -> IndicatorPanelResult:
    """Whole-universe indicator HISTORY, published one reference per indicator."""
    if input_data.price_panel_ref:
        from ..handoff import resolve as _resolve

        try:
            stacked = _resolve(input_data.price_panel_ref)
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                f"{input_data.price_panel_ref!r} could not be resolved: {exc}"
            ) from exc
        if "entity" not in getattr(stacked, "columns", []):
            raise ValidationError(
                f"{input_data.price_panel_ref!r} has no `entity` column, so "
                "it is not a stacked universe panel. fetch_ohlcv_panel "
                "produces the shape this expects."
            )
        by_ticker = {
            str(sym): part.drop(columns=["entity"])
            for sym, part in stacked.groupby("entity")
        }
    else:
        by_ticker = fetch_ohlcv_panel_sync(
            list(input_data.tickers), input_data.start_date, input_data.end_date
        )

    panels = technical_indicators_panel(by_ticker, list(input_data.indicators))

    refs: Dict[str, str] = {}
    rows = 0
    start = end = None
    for indicator, frame in panels.items():
        refs[indicator] = publish(
            frame,
            kind="indicator_panel",
            run_id=input_data.run_id,
            name=f"{input_data.name}_{indicator}",
            producer="compute_indicator_panel",
        )
        rows = max(rows, int(len(frame)))
        if len(frame):
            index = pd.to_datetime(pd.Index(frame.index))
            start = str(index.min().date())
            end = str(index.max().date())

    return IndicatorPanelResult(
        refs=refs,
        indicators=sorted(panels),
        entities=sorted(by_ticker),
        rows=rows,
        start=start,
        end=end,
        warnings=[
            "The HISTORY is published, not the latest bar -- "
            "get_technical_panel is the tool for a snapshot. These "
            "references are for something that consumes the whole series."
        ],
    )


__all__ = [
    "IndicatorPanelInput",
    "IndicatorPanelResult",
    "METRIC_NAMES",
    "SeriesMetricsInput",
    "SeriesMetricsResult",
    "calculate_series_metrics",
    "compute_indicator_panel",
]
