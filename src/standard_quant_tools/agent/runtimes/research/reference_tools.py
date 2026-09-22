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
from typing import Annotated, Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools import metrics as M
from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.agent.runtimes.data.models import DataSource, resolve_source
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.panel import technical_indicators_panel
from standard_quant_tools.numeric_contract import require_aligned
from standard_quant_tools.portfolio.portfolio import fetch_ohlcv_panel_sync

from .._optional_ref import publish_if_requested as _publish_if_requested
from ..handoff import publish

logger = logging.getLogger(__name__)
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
    # The four below had no series door anywhere in the library, so a
    # strategy's own return series -- a backtest output, a synthetic path,
    # anything an agent produced itself -- could never be scored against a
    # benchmark or run through EVT: every other route required a listed
    # ticker. See the CHANGELOG entry of 2026-09-22.
    "information_ratio": (M.information_ratio, False),
    "treynor_ratio": (M.treynor_ratio, False),
    # A SERIES, not a scalar: it is published rather than inlined.
    "drawdown_series": (M.drawdown_series, True),
    # A dict of scalars, returned as `evt` rather than flattened into
    # `values` -- two of its keys are strings and the rest describe one
    # fitted tail, which reads as a block or not at all.
    "evt_tail_risk": (M.evt_tail_risk, False),
}

#: Metrics that measure a series AGAINST another one. Both estimate what
#: they need from the intersected window themselves, so the only thing this
#: layer owes them is a benchmark and the guarantee that it lines up.
_NEEDS_BENCHMARK = ("information_ratio", "treynor_ratio")

#: The metric whose answer is a per-bar series, and the one whose answer is
#: a block of scalars. Both are named here rather than special-cased inline
#: so the dispatch loop below stays readable as a table.
_SERIES_METRIC = "drawdown_series"
_BLOCK_METRIC = "evt_tail_risk"

#: The scalar keys of `evt_tail_risk`'s dict. `method` and
#: `tail_classification` are strings and travel as a warning line instead,
#: because `evt` is typed as numbers and a string in it would be dropped by
#: the `Stat` coercion rather than reported.
_EVT_SCALARS = (
    "confidence",
    "tail_fraction",
    "threshold",
    "n_exceedances",
    "n_obs",
    "shape_xi",
    "scale_beta",
    "var_evt",
    "cvar_evt",
)

METRIC_NAMES = tuple(_METRICS)


class SeriesMetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series: DataSource = Field(
        ..., description="The RETURN series: a symbol, a reference, or values."
    )
    benchmark: Optional[DataSource] = Field(
        None,
        description=(
            "The RETURN series to measure against, in the same three "
            "shapes as `series`. Required by information_ratio and "
            "treynor_ratio, which have no meaning without one; ignored by "
            "every other metric. It must cover exactly the same bars as "
            "`series` -- equal length is not alignment, and two series "
            "labelled with different dates would be paired positionally by "
            "one execution path and by label by another."
        ),
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
    run_id: Optional[str] = Field(
        None,
        description=(
            "With `name`, publishes the drawdown series as an "
            "`analytic_series` reference and returns it as `drawdown_ref`. "
            "Both or neither: one alone is refused, because a reference is "
            "addressed by both. Required when `drawdown_series` is "
            "requested, which answers with a per-bar series rather than a "
            "number."
        ),
    )
    name: Optional[str] = Field(
        None,
        description="Names the published series within the run. See `run_id`.",
    )


class SeriesMetricsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_observations: int = 0
    periods_per_year: int = 252
    values: Dict[str, Stat] = Field(default_factory=dict)
    benchmark_observations: Optional[int] = Field(
        None,
        description="Bars in the benchmark the ratios were measured "
        "against. Null when no benchmark was given.",
    )
    drawdown_ref: Optional[str] = Field(
        None,
        description="`sqt://analytic_series/...` for the per-bar drawdown "
        "from the running peak. `max_drawdown` is its minimum; WHEN the "
        "curve was under water is only visible here. Null unless "
        "`drawdown_series` was requested with `run_id` and `name`.",
    )
    evt: Optional[Dict[str, Stat]] = Field(
        None,
        description="`evt_tail_risk`'s fitted tail: threshold, "
        "n_exceedances, n_obs, shape_xi, scale_beta, var_evt, cvar_evt and "
        "the confidence/tail_fraction it was asked for. Extrapolated from "
        "a Generalized Pareto fit to the worst 5% of losses, so it is a "
        "different number from `var_historical`, which cannot see past the "
        "worst observation there is. Null unless the metric was requested.",
    )
    warnings: List[str] = Field(default_factory=list)


def calculate_series_metrics(input_data: SeriesMetricsInput) -> SeriesMetricsResult:
    """Risk and return metrics for ANY return series, not just a ticker.

    Four of the registry's names answer with something other than one
    number, and each says so in its own field rather than being collapsed
    into `values`: `information_ratio` and `treynor_ratio` need
    `benchmark`, `drawdown_series` comes back as `drawdown_ref`, and
    `evt_tail_risk` as the `evt` block.
    """
    unknown = sorted(set(input_data.metrics) - set(_METRICS))
    if unknown:
        raise ValidationError(
            f"unknown metric(s) {unknown}; expected any of "
            f"{sorted(_METRICS)}. The set is closed rather than open "
            "because this surface is reachable from an agent."
        )
    if not input_data.metrics:
        raise ValidationError("no metrics requested; name at least one.")

    wanted_benchmark = [m for m in input_data.metrics if m in _NEEDS_BENCHMARK]
    if wanted_benchmark and input_data.benchmark is None:
        raise ValidationError(
            f"{wanted_benchmark} measure a series AGAINST another one and "
            "there is no benchmark to measure against: pass `benchmark`, "
            "which takes the same three shapes as `series` (a symbol, an "
            "`sqt://` reference, or inline values). A benchmark is not "
            "defaulted to a market index here, because which index is the "
            "benchmark is the caller's decision and guessing it would "
            "answer a different question than the one asked."
        )
    if _SERIES_METRIC in input_data.metrics and (
        input_data.run_id is None and input_data.name is None
    ):
        raise ValidationError(
            f"{_SERIES_METRIC} answers with a per-bar SERIES, not a number, "
            "and a few hundred floats inline beside the scalars is not a "
            "readable payload. Pass `run_id` and `name` and it is published "
            "as an `analytic_series` reference returned as `drawdown_ref`; "
            "for the single deepest number, ask for `max_drawdown` instead."
        )

    returns = resolve_source(input_data.series, what="calculate_series_metrics")
    equity = (1.0 + returns).cumprod()

    warnings: List[str] = []
    benchmark = None
    if input_data.benchmark is not None:
        benchmark = resolve_source(
            input_data.benchmark, what="calculate_series_metrics benchmark"
        )
        # The "equal length is not alignment" check, applied BEFORE either
        # ratio runs. Both of them intersect the two indexes internally, so
        # a strategy and a benchmark describing different days would not
        # raise -- they would quietly be scored over whatever overlap
        # happened to exist, or over nothing at all.
        require_aligned(
            returns,
            benchmark,
            "series",
            "benchmark",
            "calculate_series_metrics",
        )
        if not wanted_benchmark:
            warnings.append(
                "a benchmark was given but none of the requested metrics "
                f"reads one. {list(_NEEDS_BENCHMARK)} are the metrics "
                "measured against a benchmark."
            )
    if len(returns) < 30:
        warnings.append(
            f"only {len(returns)} observations. A Sharpe on this little data "
            "has a standard error comparable to the estimate itself -- read "
            "it as a direction, not a number."
        )

    # The residual case `resolve_source` deliberately does NOT refuse: an
    # equity curve normalized to 1.0 that spends its life BELOW 1.0 -- a
    # losing strategy -- has a typical |value| under 1 and clears the hard
    # guard. What gives it away is the ratio: mean/std above 3 is an
    # annualized Sharpe over 47. A warning rather than a refusal because a
    # T-bill return series really does look like this.
    spread = float(returns.std(ddof=1))
    if spread > 0:
        ratio = abs(float(returns.mean())) / spread
        if ratio > 3.0:
            warnings.append(
                f"mean/std is {ratio:.1f}, an annualized Sharpe of about "
                f"{ratio * (input_data.periods_per_year ** 0.5):.0f}. For a "
                "return series that is implausible; the usual cause is a "
                "LEVEL series (an equity curve) passed where returns were "
                "expected. If these really are returns -- a cash or T-bill "
                "series can look like this -- the number stands."
            )

    values: Dict[str, Any] = {}
    drawdown_ref: Optional[str] = None
    evt: Optional[Dict[str, Any]] = None
    for name in input_data.metrics:
        fn, wants_equity = _METRICS[name]
        if name in _NEEDS_BENCHMARK:
            assert benchmark is not None  # refused above
            if name == "treynor_ratio":
                # Beta is estimated INSIDE, on the intersected window, so
                # the excess return and the beta describe the same bars.
                values[name] = fn(
                    returns,
                    benchmark,
                    risk_free_rate=input_data.risk_free_rate,
                    periods_per_year=input_data.periods_per_year,
                )
            else:
                values[name] = fn(
                    returns, benchmark, periods_per_year=input_data.periods_per_year
                )
            continue
        if name == _SERIES_METRIC:
            drawdown_ref = _publish_if_requested(
                fn(equity),
                kind="analytic_series",
                run_id=input_data.run_id,
                name=input_data.name,
                producer="research.calculate_series_metrics",
            )
            continue
        if name == _BLOCK_METRIC:
            # The library's own refusals travel out by name rather than
            # being caught: "fewer than 20 exceedances" is a statement
            # about the data and the remedy is in the message.
            try:
                computed = fn(returns)
            except ValidationError as exc:
                raise ValidationError(
                    f"calculate_series_metrics({_BLOCK_METRIC}): {exc}"
                ) from exc
            evt = {key: computed[key] for key in _EVT_SCALARS}
            warnings.append(
                f"evt_tail_risk fitted a {computed['tail_classification']} "
                f"tail by {computed['method']} to the worst "
                f"{computed['tail_fraction']:.0%} of losses "
                f"({computed['n_exceedances']} exceedances). shape_xi above "
                "0.1 means the tail is heavier than exponential, so the "
                "extrapolated VaR exceeds the historical quantile by "
                "design."
            )
            continue
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
        benchmark_observations=(None if benchmark is None else int(len(benchmark))),
        drawdown_ref=drawdown_ref,
        evt=evt,
        warnings=warnings,
    )


class IndicatorPanelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: List[str] = Field(..., min_length=1)
    start_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    end_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    indicators: List[
        Literal[
            "rsi",
            "adx",
            "atr",
            "atr_simple",
            "bollinger_bands",
            "stochastic_oscillator",
            "macd",
            "sma",
            "ema",
            "williams_r",
            "obv",
            "vwap",
            "parabolic_sar",
            "mfi",
        ]
    ] = Field(
        ...,
        min_length=1,
        description=(
            "Any of: rsi, adx, atr, atr_simple, bollinger_bands, "
            "stochastic_oscillator, macd, sma, ema, williams_r, obv, vwap, "
            "parabolic_sar, mfi. `atr` is WILDER's average true range and "
            "`atr_simple` the simple rolling mean of true range -- two "
            "different numbers, kept under two names so neither silently "
            "becomes the other. obv, vwap and mfi read Volume and are "
            "refused on a panel without it."
        ),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names this artifact within the run.")
    rsi_period: int = Field(14, gt=0, le=1000)
    adx_period: int = Field(14, gt=0, le=1000)
    atr_period: int = Field(14, gt=0, le=1000, description="Wilder ATR lookback.")
    atr_simple_period: int = Field(
        14, gt=0, le=1000, description="Simple-mean ATR lookback."
    )
    bollinger_period: int = Field(20, gt=0, le=1000)
    bollinger_num_std: float = Field(2.0, gt=0, le=100)
    stoch_k_period: int = Field(14, gt=0, le=1000)
    stoch_d_period: int = Field(3, gt=0, le=1000)
    macd_fast: int = Field(12, gt=0, le=1000, description="MACD fast EMA span.")
    macd_slow: int = Field(
        26,
        gt=0,
        le=1000,
        description="MACD slow EMA span; must exceed macd_fast, since an "
        "inverted pair is a sign-flipped indicator rather than an error.",
    )
    macd_signal: int = Field(9, gt=0, le=1000, description="MACD signal EMA span.")
    sma_period: int = Field(14, gt=0, le=1000)
    ema_period: int = Field(14, gt=0, le=1000)
    williams_period: int = Field(14, gt=0, le=1000)
    vwap_period: Optional[int] = Field(
        None,
        gt=0,
        le=1000,
        description="Rolling VWAP window. Null means the cumulative VWAP "
        "from the first shared bar.",
    )
    mfi_period: int = Field(14, gt=0, le=1000)
    sar_af_start: float = Field(
        0.02, gt=0, le=1, description="Parabolic SAR starting acceleration."
    )
    sar_af_step: float = Field(
        0.02, ge=0, le=1, description="Parabolic SAR acceleration increment."
    )
    sar_af_max: float = Field(
        0.2, gt=0, le=1, description="Parabolic SAR acceleration ceiling."
    )
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

    # Every parameter, forwarded. They were dropped here while
    # `get_technical_panel` forwarded all seven, so a PERSISTED rsi panel
    # -- the one a feature or a custom backtest consumes -- was always
    # RSI(14) no matter what was asked for.
    panels = technical_indicators_panel(
        by_ticker,
        list(input_data.indicators),
        rsi_period=input_data.rsi_period,
        adx_period=input_data.adx_period,
        atr_period=input_data.atr_period,
        bollinger_period=input_data.bollinger_period,
        bollinger_num_std=input_data.bollinger_num_std,
        stoch_k_period=input_data.stoch_k_period,
        stoch_d_period=input_data.stoch_d_period,
        macd_fast=input_data.macd_fast,
        macd_slow=input_data.macd_slow,
        macd_signal=input_data.macd_signal,
        sma_period=input_data.sma_period,
        ema_period=input_data.ema_period,
        williams_period=input_data.williams_period,
        vwap_period=input_data.vwap_period,
        mfi_period=input_data.mfi_period,
        sar_af_start=input_data.sar_af_start,
        sar_af_step=input_data.sar_af_step,
        sar_af_max=input_data.sar_af_max,
        atr_simple_period=input_data.atr_simple_period,
    )

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
