"""
The intermediate series `get_microstructure_metrics` computes and discards.

WHY THESE EXIST NOW AND NOT BEFORE. `get_microstructure_metrics` is a
SUMMARY: it signs the tape, computes a spread per trade, splits the
effective spread into what the liquidity provider kept and what the trade
moved, and then returns averages. The per-trade and per-quote series it
built along the way -- the thing an event study, a CUSUM detector or a
model would actually consume -- died inside the call.

They could not usefully be exposed until a tape could be fetched and
published. Now it can:

    fetch_tick_tape ──┐
                      ├──> classify_trade_direction ──> sqt://tick_tape/...
    fetch_quote_panel ┘                                        │
                                                               v
                                                  event study / CUSUM /
                                                  a model's features

EVERY ONE OF THESE NEEDS A TICK FEED, and most environments have none.
That is a precondition, not a crash: the refusal names
describe_data_capabilities rather than surfacing a NotImplementedError a
caller cannot tell from a bug.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, List, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.analysis import microstructure as lib
from standard_quant_tools.error import ValidationError

from ..handoff import publish, resolve

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


def _frame(ref: str, expect: str, what: str) -> pd.DataFrame:
    """Resolve a reference, refusing by name rather than from inside a loader."""
    try:
        data = resolve(ref, expect=expect)
    except (ValidationError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"{what}: {ref!r} could not be resolved as a {expect!r} "
            f"reference -- {exc}. fetch_tick_tape and fetch_quote_panel in "
            "the `data` runtime are what produce these."
        ) from exc
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValidationError(f"{what}: {ref!r} resolved to nothing usable.")
    return data


class ClassifyTradesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick_tape_ref: str = Field(
        ..., description="An `sqt://tick_tape/...` from fetch_tick_tape."
    )
    quote_panel_ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://quote_panel/...`. WITH quotes this is Lee-Ready, "
            "matching each trade against the quote PRECEDING it. Without "
            "them it falls back to the tick rule, which is materially worse "
            "-- say which one you used."
        ),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the signed tape within the run.")


class SignedTapeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    method: str = Field(
        ..., description="'lee_ready' with quotes, 'tick_rule' without."
    )
    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    n_unclassified: int = 0
    buy_volume_fraction: Stat = None
    warnings: List[str] = Field(default_factory=list)


def classify_trade_direction(input_data: ClassifyTradesInput) -> SignedTapeResult:
    """Sign a tape buyer- or seller-initiated, and publish the signed series."""
    trades = _frame(input_data.tick_tape_ref, "tick_tape", "classify_trade_direction")
    quotes = (
        _frame(input_data.quote_panel_ref, "quote_panel", "classify_trade_direction")
        if input_data.quote_panel_ref
        else None
    )

    signs = lib.sign_trades(trades, quotes)
    signed = trades.copy()
    # ASSIGN, then read the assigned column. `sign_trades` may return a
    # Series indexed differently from the tape it was given -- quotes that
    # do not cover every trade drop rows -- and comparing the raw result
    # against the tape's own columns raises an unalignable-indexer error
    # from inside pandas rather than producing a wrong number. Assignment
    # aligns on index and leaves NaN where no sign was produced, which is
    # the honest representation of a trade that could not be classified.
    signed["sign"] = signs
    aligned = signed["sign"]

    method = "lee_ready" if quotes is not None else "tick_rule"
    warnings: List[str] = []
    if quotes is None:
        warnings.append(
            "TICK RULE, not Lee-Ready: no quote panel was given, so the "
            "direction comes from the sign of the price change rather than "
            "from where the trade fell against the prevailing quote. It "
            "agrees with the true classification about 85% of the time on a "
            "liquid name and materially worse on an illiquid one, and the "
            "error attenuates every downstream estimate toward zero."
        )

    ref = publish(
        signed,
        kind="tick_tape",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="classify_trade_direction",
    )

    buys = int((aligned > 0).sum())
    sells = int((aligned < 0).sum())
    # A trade with no sign is UNCLASSIFIED whether that arrived as a zero
    # or as a NaN from the alignment above; both mean the same thing to a
    # caller and splitting them would be a distinction about plumbing.
    unclassified = int(((aligned == 0) | aligned.isna()).sum())
    volume = signed["size"] if "size" in signed.columns else None
    buy_fraction = None
    if volume is not None and float(volume.sum()) > 0:
        buy_fraction = float(volume[aligned > 0].sum() / volume.sum())
    if unclassified:
        warnings.append(
            f"{unclassified:,} of {len(signed):,} trades could not be "
            "signed -- they carry no direction rather than a neutral one, "
            "and any flow imbalance computed downstream is over the "
            "remainder."
        )

    return SignedTapeResult(
        ref=ref,
        method=method,
        n_trades=int(len(signed)),
        n_buys=buys,
        n_sells=sells,
        n_unclassified=unclassified,
        buy_volume_fraction=buy_fraction,
        warnings=warnings,
    )


class QuotedSpreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote_panel_ref: str = Field(
        ..., description="An `sqt://quote_panel/...` from fetch_quote_panel."
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the series within the run.")


class SpreadSeriesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str
    n_rows: int = 0
    columns: List[str] = Field(default_factory=list)
    mean_bps: Stat = None
    median_bps: Stat = None
    p95_bps: Stat = None
    warnings: List[str] = Field(default_factory=list)


def _bps_summary(frame: pd.DataFrame, column: str) -> tuple:
    """
    Summarize ONE named bps column.

    This used to scan for the first column whose name contained "bps" and
    summarize that. `effective_spread` merges the quoted `spread_bps` in
    before it assigns `effective_spread_bps`, so the first match was the
    QUOTED spread -- and `get_effective_spread_series`, whose whole promise
    is "what each trade ACTUALLY paid against the prevailing midpoint",
    reported the quoted number instead. Measured on a 20 bps quote with
    trades 1 bp off mid: reported mean_bps 20.0, true effective 2.0. It is
    the number the tool's own warning says not to charge.
    """
    if column not in frame.columns:
        return (None, None, None)
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if not len(series):
        return (None, None, None)
    return (
        float(series.mean()),
        float(series.median()),
        float(series.quantile(0.95)),
    )


def get_quoted_spread_series(input_data: QuotedSpreadInput) -> SpreadSeriesResult:
    """Spread and imbalance PER QUOTE, not averaged away."""
    quotes = _frame(
        input_data.quote_panel_ref, "quote_panel", "get_quoted_spread_series"
    )
    series = lib.quoted_spread(quotes)
    mean, median, p95 = _bps_summary(series, "spread_bps")
    ref = publish(
        series,
        kind="quote_panel",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="get_quoted_spread_series",
    )
    return SpreadSeriesResult(
        ref=ref,
        n_rows=int(len(series)),
        columns=[str(c) for c in series.columns],
        mean_bps=mean,
        median_bps=median,
        p95_bps=p95,
        warnings=[
            "QUOTED is what crossing would cost at an instant, not what "
            "trades paid. The effective spread is the one a backtest should "
            "be charging -- get_effective_spread_series computes it.",
            "Top of book only. Depth, queue position and resting size are "
            "not in this data and cannot be inferred from it.",
        ],
    )


class EffectiveSpreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tick_tape_ref: str = Field(..., description="An `sqt://tick_tape/...`.")
    quote_panel_ref: str = Field(
        ...,
        description=(
            "An `sqt://quote_panel/...`. Required: the effective spread is "
            "measured against the prevailing midpoint, which no tape holds."
        ),
    )
    realized_horizon_seconds: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Split the effective spread into REALIZED (what the liquidity "
            "provider kept) and IMPACT (what the trade moved) by comparing "
            "against the midpoint this many seconds later. The two imply "
            "opposite remedies, which is why the split is worth the "
            "argument."
        ),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the series within the run.")


def get_effective_spread_series(
    input_data: EffectiveSpreadInput,
) -> SpreadSeriesResult:
    """What each trade ACTUALLY paid against the prevailing midpoint."""
    trades = _frame(
        input_data.tick_tape_ref, "tick_tape", "get_effective_spread_series"
    )
    quotes = _frame(
        input_data.quote_panel_ref, "quote_panel", "get_effective_spread_series"
    )
    horizon = (
        pd.Timedelta(seconds=input_data.realized_horizon_seconds)
        if input_data.realized_horizon_seconds
        else None
    )
    series = lib.effective_spread(trades, quotes, horizon)
    mean, median, p95 = _bps_summary(series, "effective_spread_bps")

    warnings = [
        "EFFECTIVE is what trades paid against the prevailing midpoint, and "
        "it is the number a backtest should charge -- not the quoted spread."
    ]
    if horizon is None:
        warnings.append(
            "No realized_horizon_seconds, so the effective spread is NOT "
            "split into realized and impact. Those two imply opposite "
            "remedies -- impact says trade smaller, realized says trade "
            "somewhere else -- and without the split neither is visible."
        )

    ref = publish(
        series,
        kind="tick_tape",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="get_effective_spread_series",
    )
    return SpreadSeriesResult(
        ref=ref,
        n_rows=int(len(series)),
        columns=[str(c) for c in series.columns],
        mean_bps=mean,
        median_bps=median,
        p95_bps=p95,
        warnings=warnings,
    )


__all__ = [
    "ClassifyTradesInput",
    "EffectiveSpreadInput",
    "QuotedSpreadInput",
    "SignedTapeResult",
    "SpreadSeriesResult",
    "classify_trade_direction",
    "get_effective_spread_series",
    "get_quoted_spread_series",
]
