"""
Liquidity estimated from BARS, for the normal case where there is no tick
feed.

The four tools this runtime already had measure spreads from trades and
quotes and refuse to run without them. That refusal is right -- a quoted
spread is a quoted spread, and nothing computed from a daily bar is one. But
"no tick data" is the normal situation, and it does not make the questions go
away. Everything here recovers a liquidity measure from OHLCV and says what
it is a proxy FOR and how it fails.

THE ONE THING TO CARRY ACROSS ALL OF THEM: these are historical averages
under a model, not the spread you will pay. The cost at the moment you send
an order depends on the book at that moment, and no daily bar contains it.

INPUTS ARE INLINE -- prices and volumes as lists, not a ticker. The same
tools then work on a series the caller built, resampled, or simulated, which
is most of how a liquidity proxy actually gets used.

ONE EXCEPTION, AND IT IS THE POINT OF THE EXCEPTION. `estimate_kyle_lambda`
also takes a tick tape and a quote panel by reference, because its bars
estimate is circular by construction -- the bar's own return signs the
volume that return is then regressed on. A tape signs the flow from the
prints instead, and it is a tape rather than a list because a session of
them does not belong in a payload. See the CHANGELOG entry of 2026-09-22.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, List, Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.analysis import microstructure_estimators as lib
from standard_quant_tools.error import ValidationError

from .._optional_ref import publish_if_requested as _publish_if_requested
from ..handoff import resolve as _resolve

logger = logging.getLogger(__name__)
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

#: A session boundary, zero-padded on a 24-hour clock. '9:30' is refused
#: rather than read, because '9:3' parses under a looser pattern as 09:03.
_HHMM = re.compile(r"(?:[01]\d|2[0-3]):[0-5]\d")


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


def _referenced(ref: str, expect: str, who: str) -> pd.DataFrame:
    """
    Resolve a tape or a book, refusing by name and naming what produces one.

    Every failure here -- a malformed reference, the wrong kind, a run
    directory that was cleared -- becomes ONE refusal that ends with the
    tool a caller would run to get a good reference. The underlying
    messages say what is wrong with the string; none of them says where a
    tick tape comes from, and a caller holding a bad ref needs both.
    """
    try:
        data = _resolve(ref, expect=expect)
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"{who}: {ref!r} could not be resolved as a {expect!r} "
            f"reference -- {exc} fetch_tick_tape and fetch_quote_panel in "
            "the `data` runtime are what produce these."
        ) from exc
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValidationError(
            f"{who}: {ref!r} resolved to nothing usable. fetch_tick_tape "
            "and fetch_quote_panel in the `data` runtime are what produce "
            "these."
        )
    return data


def _ohlcv(
    who: str,
    *,
    close: Optional[List[float]] = None,
    volume: Optional[List[float]] = None,
    high: Optional[List[float]] = None,
    low: Optional[List[float]] = None,
    timestamps: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parallel lists into a frame, refusing a ragged one by name.

    Pandas would pad the short column with NaN and carry on, which drops
    observations from the estimate without changing anything visible -- and
    every estimator here is sensitive to how many observations it actually
    got.
    """
    columns = {
        name: values
        for name, values in (
            ("close", close),
            ("volume", volume),
            ("high", high),
            ("low", low),
        )
        if values is not None
    }
    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValidationError(
            f"{who}: the columns have different lengths ({lengths}). They "
            "must be parallel -- padding the short one would silently drop "
            "observations from the estimate."
        )
    index = None
    if timestamps is not None:
        if len(timestamps) != next(iter(lengths.values())):
            raise ValidationError(
                f"{who}: {len(timestamps)} timestamps against "
                f"{next(iter(lengths.values()))} observations."
            )
        try:
            index = pd.DatetimeIndex(pd.to_datetime(timestamps))
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"{who}: could not parse timestamps -- {exc}")
    return pd.DataFrame(columns, index=index)


# ── inputs ──────────────────────────────────────────────────────────────


class RollSpreadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prices: List[float] = Field(
        ..., min_length=30, description="Trade or close prices, oldest first."
    )
    window: Optional[int] = Field(
        None,
        ge=10,
        description="Rolling window for a time series of estimates. Omit for "
        "one estimate over the whole sample.",
    )


class CorwinSchultzInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: List[float] = Field(..., min_length=30)
    low: List[float] = Field(..., min_length=30)


class AmihudInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close: List[float] = Field(..., min_length=30)
    volume: List[float] = Field(..., min_length=30)
    window: int = Field(21, ge=2, description="Averaging window, in bars.")
    run_id: Optional[str] = Field(
        None,
        description=(
            "With `name`, publishes the rolling illiquidity series itself "
            "as an `analytic_series` reference and returns it as "
            "`rolling_ref`. Both or neither: one alone is refused, because "
            "a reference is addressed by both."
        ),
    )
    name: Optional[str] = Field(
        None, description="Names the published series within the run. See `run_id`."
    )


class KyleLambdaInput(BaseModel):
    """
    Two ways in, and they do not measure the same thing.

    A TAPE (`trades_ref`, ideally with `quotes_ref`) signs each print
    against the quote that preceded it, so the sign is evidence rather than
    a restatement of the answer. BARS (`close`, `volume`) have no sign in
    them: the only one available is the bar's own return, which is the very
    thing the regression explains, and the result says `circular=True`.
    Prefer the tape wherever one exists. See the CHANGELOG entry of
    2026-09-22.
    """

    model_config = ConfigDict(extra="forbid")

    close: Optional[List[float]] = Field(
        None,
        min_length=30,
        description="Bar closes, oldest first. With `volume`, the CIRCULAR "
        "path -- give `trades_ref` instead when a tape exists.",
    )
    volume: Optional[List[float]] = Field(
        None, min_length=30, description="Bar volumes, aligned with `close`."
    )
    trades_ref: Optional[str] = Field(
        None,
        description="An `sqt://tick_tape/...` from fetch_tick_tape. THE "
        "PATH TO PREFER: the tape is signed print by print and bucketed at "
        "`freq`, so lambda is estimated from a sign the regression does not "
        "already know. Mutually exclusive with `close`/`volume`.",
    )
    quotes_ref: Optional[str] = Field(
        None,
        description="An `sqt://quote_panel/...` from fetch_quote_panel, "
        "alongside `trades_ref`. WITH quotes the sign is Lee-Ready and the "
        "price that moves is the MIDPOINT, which keeps the bid-ask bounce "
        "out of the slope; without them the tape falls back to the tick "
        "rule on trade prices, which is materially worse. Requires "
        "`trades_ref`.",
    )
    freq: Literal["1s", "5s", "10s", "30s", "1min", "5min"] = Field(
        "1min",
        description="Bucket width for the tape path -- signed volume is "
        "summed and the price change taken over each bucket. Shorter "
        "buckets give more observations and a noisier price change; longer "
        "ones net opposing flow inside the bucket away. Ignored on the bars "
        "path, where the bars are the buckets.",
    )
    window: Optional[int] = Field(
        None, ge=20, description="Rolling window, for a spread of estimates."
    )

    @model_validator(mode="after")
    def _one_source_of_flow(self) -> "KyleLambdaInput":
        bars = self.close is not None or self.volume is not None
        if bars and self.trades_ref is not None:
            raise ValueError(
                "estimate_kyle_lambda: pass a tape (`trades_ref`) or bars "
                "(`close` and `volume`), not both. They are different "
                "estimates -- the tape's sign comes from the prints, the "
                "bars' from the return being explained -- and silently "
                "preferring one would hide which was measured. Drop "
                "`close`/`volume` to use the tape."
            )
        if self.trades_ref is None:
            if not bars:
                raise ValueError(
                    "estimate_kyle_lambda: nothing to measure. Pass "
                    "`trades_ref` (an `sqt://tick_tape/...` from "
                    "fetch_tick_tape, with `quotes_ref` from "
                    "fetch_quote_panel for a Lee-Ready sign), or `close` "
                    "and `volume` for the circular bars estimate."
                )
            if self.close is None or self.volume is None:
                missing = "close" if self.close is None else "volume"
                raise ValueError(
                    f"estimate_kyle_lambda: the bars path needs both "
                    f"`close` and `volume`; `{missing}` is missing. The "
                    "regression is a price change on a signed volume, so "
                    "neither side can be inferred from the other."
                )
            if self.quotes_ref is not None:
                raise ValueError(
                    "estimate_kyle_lambda: `quotes_ref` signs a trade tape "
                    "and does nothing to bars. Pass `trades_ref` as well, "
                    "or drop `quotes_ref`."
                )
        return self


class OrderFlowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close: List[float] = Field(..., min_length=30)
    volume: List[float] = Field(..., min_length=30)
    window: int = Field(5, ge=2, description="Bars per imbalance window.")


class VpinInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close: List[float] = Field(..., min_length=30)
    volume: List[float] = Field(..., min_length=30)
    n_buckets: int = Field(50, ge=5, le=1000, description="Equal-VOLUME buckets.")
    window: int = Field(50, ge=2, description="Buckets averaged per reading.")


class VolumeProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    volume: List[float] = Field(..., min_length=30)
    timestamps: List[str] = Field(
        ...,
        description="ISO timestamps WITH a time of day, one per bar. Daily "
        "bars are refused -- there is no intraday profile in daily data.",
    )
    n_buckets: int = Field(13, ge=3, le=100)
    exchange_timezone: str = Field(
        "America/New_York",
        description="IANA zone the session times below are read in. The "
        "default is US equities; a London tape is Europe/London and a Tokyo "
        "one Asia/Tokyo, and leaving the default on a non-US tape is refused "
        "rather than answered -- under the New York session a London day "
        "either falls outside it or keeps a couple of hours, and the "
        "open_share that comes back is measured on the wrong end of "
        "somebody else's afternoon.",
    )
    session_start: str = Field(
        "09:30",
        description="HH:MM, 24-hour, in exchange_timezone. The regular "
        "session's open: bars before it are extended hours and are excluded "
        "from the buckets, with their share reported separately.",
    )
    session_end: str = Field(
        "16:00",
        description="HH:MM, 24-hour, in exchange_timezone, exclusive. Must "
        "be later than session_start.",
    )
    index_timezone: Optional[str] = Field(
        None,
        description="IANA zone the `timestamps` are already in, when they "
        "carry no offset of their own. 'UTC' is the one to pass for a "
        "Databento extract. Omitted, a timestamp with an offset is honoured "
        "and a naive one is taken as already in exchange_timezone -- and a "
        "naive index that is really UTC is how an extended-hours feed gets "
        "bucketed from 4am, which reports an open_share of 0.00004.",
    )

    @field_validator("exchange_timezone", "index_timezone")
    @classmethod
    def _a_real_zone(cls, value: Optional[str]) -> Optional[str]:
        """By name, against the system's own zone database.

        A misspelled zone is not a near miss: `ZoneInfo` raises from inside
        the session conversion, several frames below this call, with a
        message naming no tool and no argument.
        """
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except Exception as exc:  # noqa: BLE001 -- any zone failure is one answer
            raise ValueError(
                f"{value!r} is not an IANA time zone ({exc}). Use a name from "
                "the tz database, such as 'America/New_York', "
                "'Europe/London', 'Asia/Tokyo' or 'UTC'."
            ) from exc
        return value

    @field_validator("session_start", "session_end")
    @classmethod
    def _a_clock_time(cls, value: str) -> str:
        if not _HHMM.fullmatch(value.strip()):
            raise ValueError(
                f"{value!r} is not a session time. Give HH:MM on a 24-hour "
                "clock, zero-padded -- '09:30', not '9:3' or '9:30am'."
            )
        return value.strip()

    @model_validator(mode="after")
    def _session_is_forward(self) -> "VolumeProfileInput":
        start = [int(part) for part in self.session_start.split(":")]
        end = [int(part) for part in self.session_end.split(":")]
        if end[0] * 60 + end[1] <= start[0] * 60 + start[1]:
            raise ValueError(
                f"session_end {self.session_end!r} must be later in the day "
                f"than session_start {self.session_start!r}. A session that "
                "crosses midnight is not something this profile can bucket."
            )
        return self


# ── results ─────────────────────────────────────────────────────────────


class RollRolling(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_windows: int = 0
    n_undefined: int = 0
    median_spread: Stat = None
    p25_spread: Stat = None
    p75_spread: Stat = None


class RollSpreadResult(_Result):
    n_observations: int = 0
    serial_covariance: Stat = None
    covariance_standard_error: Stat = None
    significant: Optional[bool] = Field(
        None,
        description="Whether the estimate is distinguishable from zero. On a "
        "series with NO spread the formula still returns a confident-looking "
        "number; this is what separates the two.",
    )
    smallest_detectable_spread: Stat = Field(
        None,
        description="Below this, any estimate is sampling noise that happened "
        "to land on the negative side of the covariance.",
    )
    spread_estimate: Stat = Field(
        None,
        description="Null when the serial covariance is positive -- 'could "
        "not measure' rather than 'was zero'.",
    )
    spread_bps: Stat = None
    half_spread_bps: Stat = None
    mean_price: Stat = None
    undefined_fraction: Stat = None
    rolling: Optional[RollRolling] = None


class CorwinSchultzResult(_Result):
    n_observations: int = 0
    n_estimates: int = 0
    spread_estimate: Stat = None
    spread_bps: Stat = None
    median_spread_bps: Stat = None
    negative_fraction: Stat = Field(
        None,
        description="Daily estimates that came out NEGATIVE and were floored "
        "at zero. 10-30% is normal; above a third the average is noise, "
        "because flooring turns a symmetric error into a one-sided bias.",
    )
    n_gap_adjusted: int = Field(
        0,
        description="Bar pairs that did not overlap at all, and so had the "
        "overnight gap removed before the two-bar range was measured. The "
        "estimator assumes a continuous price, so a large count means the "
        "answer rests on that adjustment rather than on the data. It moves "
        "raw_mean_bps and usually not spread_bps, because a gapping pair "
        "lands negative and is floored either way.",
    )
    raw_mean_bps: Stat = None


class AmihudResult(_Result):
    n_observations: int = 0
    window: int = 0
    current_illiquidity: Stat = None
    current_percentile: Stat = Field(
        None,
        description="READ THIS, not the raw value. The raw number's units "
        "are return-per-dollar, so it scales inversely with dollar volume "
        "and means nothing on its own.",
    )
    mean_illiquidity: Stat = None
    median_illiquidity: Stat = None
    trend_pct: Stat = None
    scaling: str = "1e6"
    mean_dollar_volume: Stat = None
    rolling_ref: Optional[str] = Field(
        None,
        description="`sqt://analytic_series/...` for the rolling ratio "
        "itself, `n_observations - window + 1` values. `current_percentile` "
        "and `trend_pct` are both comparisons within this series, and "
        "neither can be checked without it. Null unless `run_id` and "
        "`name` were given.",
    )


class KyleRolling(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_windows: int = 0
    median_lambda: Stat = None
    p25: Stat = None
    p75: Stat = None
    latest: Stat = None


class KyleLambdaResult(_Result):
    n_observations: int = 0
    kyle_lambda: Stat = Field(
        None, description="Price impact per unit of signed volume: market depth."
    )
    r_squared: Stat = Field(
        None,
        description="Check this. A lambda from a regression explaining 2% of "
        "the variance has a standard error larger than itself.",
    )
    impact_of_1pct_adv: Stat = None
    impact_of_1pct_adv_bps: Stat = None
    mean_price: Stat = None
    mean_volume: Stat = None
    circular: bool = Field(
        True,
        description="True when the flow was signed by the bar's own return, "
        "which is the variable the regression explains: lambda is positive "
        "by construction and r_squared measures nothing. READ THIS FIRST. "
        "Only a tape (`trades_ref`) makes it False.",
    )
    sign_source: str = Field(
        "return_sign",
        description="What signed the flow: 'lee_ready' (tape and quotes, "
        "each print matched against the quote preceding it), 'tick_rule' "
        "(tape without quotes, signed off the previous print's price) or "
        "'return_sign' (bars, the circular case).",
    )
    freq: Optional[str] = Field(
        None,
        description="The bucket the tape was summed into before the "
        "regression. Null on the bars path, where the bars are the buckets "
        "and their width is whatever the caller sampled.",
    )
    rolling: Optional[KyleRolling] = None


class OrderFlowResult(_Result):
    n_observations: int = 0
    n_non_overlapping: int = 0
    window: int = 0
    current_imbalance: Stat = None
    mean_imbalance: Stat = None
    std_imbalance: Stat = None
    persistence: Stat = Field(
        None,
        description="Measured on NON-OVERLAPPING windows. The overlapping "
        "version is roughly 1 - 1/window whatever the data does.",
    )
    overlapping_persistence: Stat = Field(
        None, description="The artefact, returned so the difference is visible."
    )
    next_day_correlation: Stat = None
    buy_volume_fraction: Stat = None


class VpinResult(_Result):
    n_buckets: int = 0
    bucket_volume: Stat = None
    window: int = 0
    current_vpin: Stat = None
    current_percentile: Stat = None
    mean_vpin: Stat = None
    max_vpin: Stat = None


class ProfileBucket(BaseModel):
    model_config = ConfigDict(extra="allow")

    bucket: int = 0
    start_time: str = ""
    share_of_volume: Stat = None
    mean_volume: Stat = None
    n_bars: int = 0


class VolumeProfileResult(_Result):
    n_bars: int = 0
    n_buckets: int = 0
    profile: List[ProfileBucket] = Field(default_factory=list)
    u_shaped: bool = False
    open_share: Stat = None
    close_share: Stat = None
    trough_share: Stat = None
    trough_bucket: int = 0
    open_to_trough_ratio: Stat = None
    session: List[str] = Field(
        default_factory=list,
        description="The session the buckets were measured over, as it was "
        "read: [start, end] in exchange_timezone.",
    )
    extended_hours_share: Stat = Field(
        None,
        description="Fraction of the tape's volume that fell OUTSIDE the "
        "session and was excluded from the buckets. Null when no zone was "
        "known, so no bar could be placed inside or outside a session: that "
        "is 'not measured', not 'none'. A large share on a feed you thought "
        "was regular-hours means the extract carries pre- and post-market "
        "and the profile below is the right one.",
    )


# ── tools ───────────────────────────────────────────────────────────────


def estimate_roll_spread(input_data: RollSpreadInput) -> RollSpreadResult:
    return RollSpreadResult(
        **lib.roll_spread(pd.Series(input_data.prices), window=input_data.window)
    )


def estimate_corwin_schultz_spread(
    input_data: CorwinSchultzInput,
) -> CorwinSchultzResult:
    frame = _ohlcv(
        "estimate_corwin_schultz_spread",
        high=input_data.high,
        low=input_data.low,
    )
    return CorwinSchultzResult(**lib.corwin_schultz_spread(frame))


def get_amihud_illiquidity(input_data: AmihudInput) -> AmihudResult:
    frame = _ohlcv(
        "get_amihud_illiquidity", close=input_data.close, volume=input_data.volume
    )
    computed = lib.amihud_illiquidity(frame, window=input_data.window)
    # This result model is extra="allow", so splatting the series would put
    # one float per bar inline beside the numbers that summarize it.
    # Published instead when the caller asked for it.
    rolling = computed.pop("rolling", None)
    ref = _publish_if_requested(
        rolling,
        kind="analytic_series",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="microstructure.get_amihud_illiquidity",
    )
    return AmihudResult(**computed, rolling_ref=ref)


def estimate_kyle_lambda(input_data: KyleLambdaInput) -> KyleLambdaResult:
    """Market depth from a signed tape, or -- circularly -- from bars."""
    if input_data.trades_ref is not None:
        trades = _referenced(input_data.trades_ref, "tick_tape", "estimate_kyle_lambda")
        quotes = (
            _referenced(input_data.quotes_ref, "quote_panel", "estimate_kyle_lambda")
            if input_data.quotes_ref is not None
            else None
        )
        computed = lib.kyle_lambda(
            trades=trades,
            quotes=quotes,
            freq=input_data.freq,
            window=input_data.window,
        )
    else:
        frame = _ohlcv(
            "estimate_kyle_lambda", close=input_data.close, volume=input_data.volume
        )
        computed = lib.kyle_lambda(frame, window=input_data.window)
    return KyleLambdaResult(**computed)


def get_order_flow_imbalance(input_data: OrderFlowInput) -> OrderFlowResult:
    frame = _ohlcv(
        "get_order_flow_imbalance", close=input_data.close, volume=input_data.volume
    )
    return OrderFlowResult(**lib.order_flow_imbalance(frame, window=input_data.window))


def estimate_vpin(input_data: VpinInput) -> VpinResult:
    frame = _ohlcv("estimate_vpin", close=input_data.close, volume=input_data.volume)
    return VpinResult(
        **lib.estimate_vpin(
            frame, n_buckets=input_data.n_buckets, window=input_data.window
        )
    )


def get_intraday_volume_profile(
    input_data: VolumeProfileInput,
) -> VolumeProfileResult:
    frame = _ohlcv(
        "get_intraday_volume_profile",
        volume=input_data.volume,
        timestamps=input_data.timestamps,
    )
    return VolumeProfileResult(
        **lib.intraday_volume_profile(
            frame,
            n_buckets=input_data.n_buckets,
            index_timezone=input_data.index_timezone,
            exchange_timezone=input_data.exchange_timezone,
            session=(input_data.session_start, input_data.session_end),
        )
    )


ESTIMATOR_TOOL_DEFS = [
    (
        "estimate_roll_spread",
        "The effective spread implied by BID-ASK BOUNCE, from trade prices "
        "alone (Roll 1984). Consecutive price changes mean-revert when trades "
        "arrive randomly at bid and ask, and the size of that reversal is the "
        "spread. IT RETURNS A SPREAD WHEN THERE IS NONE: on a simulated walk "
        "with a spread of exactly zero it produced 0.098 on a $100 stock, "
        "because the lag-1 autocovariance's standard error swamps the signal "
        "whenever the spread is small against volatility, and taking a root "
        "only when the covariance lands negative discards the other half of "
        "that noise. Read `significant` and `smallest_detectable_spread` "
        "before the estimate. On a trending series it returns null rather "
        "than zero, because 'could not measure' and 'was zero' are different "
        "facts.",
        RollSpreadInput,
    ),
    (
        "estimate_corwin_schultz_spread",
        "The spread implied by the HIGH-LOW RANGE (Corwin-Schultz 2012). A "
        "day's range contains both volatility and the spread; volatility "
        "scales with the square root of time and the spread does not, so one- "
        "and two-day ranges identify them separately with no quote data at "
        "all. It produces NEGATIVE estimates on 10-30% of days as a sampling "
        "artefact, floored at zero as the authors recommend -- read "
        "negative_fraction, because above about a third the flooring turns a "
        "symmetric error into a one-sided bias and the average is noise. "
        "Measured: a planted 100 bps spread came back at 103 bps, a planted "
        "20 bps at 56 bps with 44% negative.",
        CorwinSchultzInput,
    ),
    (
        "get_amihud_illiquidity",
        "How far the price moves per dollar traded (Amihud 2002) -- the most "
        "widely used liquidity proxy in the literature, because it needs "
        "nothing but daily bars. THE RAW NUMBER IS UNINTERPRETABLE: its units "
        "are return-per-dollar, so it scales inversely with dollar volume and "
        "a large cap's reading is orders of magnitude below a microcap's with "
        "neither meaning anything alone. Read the percentile against this "
        "name's own history. It is NOT a spread -- it conflates spread, book "
        "depth and the information content of trades, and a genuinely "
        "volatile stock scores as illiquid even with a deep book.",
        AmihudInput,
    ),
    (
        "estimate_kyle_lambda",
        "Market DEPTH: the price impact of a unit of signed order flow, from "
        "a regression of price change on signed volume. The one measure here "
        "with a direct trading interpretation -- multiply by the size you "
        "intend to trade for an estimate of the impact you will cause. IT "
        "MATTERS ENORMOUSLY WHICH WAY YOU CALL IT. From BARS (`close`, "
        "`volume`) the only sign available is the bar's own return, so "
        "sign(y) * volume is regressed on y: lambda is positive by "
        "construction, r_squared measures nothing, and the result says "
        "`circular=True`. From a TAPE (`trades_ref` from fetch_tick_tape, "
        "with `quotes_ref` from fetch_quote_panel) each print is signed "
        "Lee-Ready against the quote before it and bucketed at `freq`, and "
        "the sign is then evidence rather than a restatement of the answer. "
        "Measured side by side on the same live tape, the circular estimate "
        "was 3.2x the signed one while its r_squared looked 2.7x better. "
        "PREFER THE TAPE WHENEVER ONE EXISTS; use bars only when it does "
        "not, and read `circular`, `sign_source` and r_squared before sizing "
        "anything off the number.",
        KyleLambdaInput,
    ),
    (
        "get_order_flow_imbalance",
        "Signed volume imbalance from bars, with its own predictive test "
        "attached rather than presented as a signal to be trusted. "
        "`persistence` is measured on NON-OVERLAPPING windows: a rolling sum "
        "at window=5 shares four of five observations with the previous "
        "point, so its raw autocorrelation is about +0.76 on PURE NOISE (and "
        "+0.89 at window 10, +0.96 at 21, tracking 1-1/w). That number "
        "describes the window, not the flow, and it is returned separately as "
        "`overlapping_persistence` so the difference is visible.",
        OrderFlowInput,
    ),
    (
        "estimate_vpin",
        "Flow one-sidedness measured in VOLUME time rather than clock time "
        "(Easley, Lopez de Prado and O'Hara 2012) -- information arrives with "
        "volume, so the series is cut into equal-volume buckets. TWO HONEST "
        "CAVEATS. This is built from daily bars with tick-rule signing; the "
        "original is a trade-level measure where each bucket holds hundreds "
        "of trades, so what comes back is a defensible series of "
        "one-sidedness and not the VPIN of the paper. And VPIN is contested: "
        "Andersen and Bondarenko (2014) argue it is largely a transformation "
        "of volatility. Calling one-sided flow 'informed trading' is a model "
        "assumption, not a measurement.",
        VpinInput,
    ),
    (
        "get_intraday_volume_profile",
        "How volume distributes across the trading day, and what that implies "
        "for a participation schedule. The U-shape is the fact every "
        "execution schedule is built on: volume concentrates at the open and "
        "close with a midday trough routinely a third of the opening bucket, "
        "so a schedule spread evenly across the CLOCK over-participates at "
        "lunch -- paying impact into a thin book -- and under-participates at "
        "the close, missing the cheapest liquidity of the day. Needs INTRADAY "
        "bars with timestamps; daily bars are refused rather than aggregated "
        "into a meaningless single bucket. SET THE SESSION FOR THE VENUE THE "
        "BARS CAME FROM -- exchange_timezone, session_start, session_end -- "
        "because the default is US equity regular hours, and a London or "
        "Tokyo tape measured against it is bucketed over the wrong part of "
        "the day or refused outright. Pass index_timezone ('UTC' for a "
        "Databento extract) when the timestamps carry no offset, so "
        "extended-hours bars can be told from session ones instead of "
        "stretching the open bucket back to 4am.",
        VolumeProfileInput,
    ),
]

ESTIMATOR_TOOL_DISPATCH = {
    "estimate_roll_spread": (estimate_roll_spread, RollSpreadInput),
    "estimate_corwin_schultz_spread": (
        estimate_corwin_schultz_spread,
        CorwinSchultzInput,
    ),
    "get_amihud_illiquidity": (get_amihud_illiquidity, AmihudInput),
    "estimate_kyle_lambda": (estimate_kyle_lambda, KyleLambdaInput),
    "get_order_flow_imbalance": (get_order_flow_imbalance, OrderFlowInput),
    "estimate_vpin": (estimate_vpin, VpinInput),
    "get_intraday_volume_profile": (
        get_intraday_volume_profile,
        VolumeProfileInput,
    ),
}


class ShortfallFill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quantity: float = Field(..., gt=0, description="Shares or contracts filled.")
    price: float = Field(..., gt=0)
    fee: float = Field(0.0, description="Commission and fees for this fill.")


class ShortfallInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_price: float = Field(
        ...,
        gt=0,
        description="Price when the DECISION was made, not when the order "
        "reached the market. Passing the arrival price here sets the delay "
        "cost to zero by construction, and delay is often where the money "
        "went.",
    )
    arrival_price: float = Field(
        ..., gt=0, description="Price when the order first reached the market."
    )
    fills: List[ShortfallFill] = Field(..., min_length=1)
    target_quantity: float = Field(
        ..., gt=0, description="What you intended to trade. Direction is `side`."
    )
    final_price: float = Field(
        ...,
        gt=0,
        description="Price at which the unfilled remainder is valued.",
    )
    side: Literal["buy", "sell"] = Field("buy")


class ShortfallComponent(BaseModel):
    model_config = ConfigDict(extra="allow")

    component: str = ""
    dollars: Stat = None
    bps: Stat = None


class ShortfallResult(_Result):
    side: str = ""
    target_quantity: Stat = None
    filled_quantity: Stat = None
    unfilled_quantity: Stat = None
    fill_rate: Stat = None
    decision_price: Stat = None
    arrival_price: Stat = None
    average_fill_price: Stat = None
    final_price: Stat = None
    total_shortfall_dollars: Stat = None
    total_shortfall_bps: Stat = Field(
        None, description="POSITIVE is a cost. Both conventions exist."
    )
    components: List[ShortfallComponent] = Field(default_factory=list)
    largest_component: str = ""
    n_fills: int = 0


def get_implementation_shortfall(input_data: ShortfallInput) -> ShortfallResult:
    return ShortfallResult(
        **lib.implementation_shortfall(
            decision_price=input_data.decision_price,
            arrival_price=input_data.arrival_price,
            fills=[f.model_dump() for f in input_data.fills],
            target_quantity=input_data.target_quantity,
            final_price=input_data.final_price,
            side=input_data.side,
        )
    )


ESTIMATOR_TOOL_DEFS.append(
    (
        "get_implementation_shortfall",
        "What an execution ACTUALLY cost, decomposed after Perold. Every "
        "other cost tool here is a model run before the fact -- "
        "estimate_trade_cost predicts, get_capacity_report bounds, "
        "plan_rebalance schedules -- and this is the measurement those models "
        "should be checked against. Splits the gap between the decision price "
        "and what was achieved into DELAY (the price moved before the order "
        "reached the market, a workflow problem no algorithm recovers, and "
        "frequently the largest term), IMPACT (the part an algorithm "
        "controls), OPPORTUNITY (the shares never filled -- an algorithm that "
        "beats its benchmark by not completing has moved its cost here rather "
        "than saved it), and FEES. Positive is a cost.",
        ShortfallInput,
    )
)

ESTIMATOR_TOOL_DISPATCH["get_implementation_shortfall"] = (
    get_implementation_shortfall,
    ShortfallInput,
)


__all__ = [
    "get_implementation_shortfall",
    "ESTIMATOR_TOOL_DEFS",
    "ESTIMATOR_TOOL_DISPATCH",
    "estimate_corwin_schultz_spread",
    "estimate_kyle_lambda",
    "estimate_roll_spread",
    "estimate_vpin",
    "get_amihud_illiquidity",
    "get_intraday_volume_profile",
    "get_order_flow_imbalance",
]
