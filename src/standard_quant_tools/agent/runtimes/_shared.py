"""
Infrastructure every tool runtime needs.

Deliberately small. Only four things are genuinely shared across runtimes --
the C++ extension probe, the interval the backtest tools fetch bars at,
`_run_backtest`, which the execution and validation tools both call and
which is the reason those two categories live in ONE runtime rather than
two, and `parse_date_keys`, which every tool taking an inline date-keyed
map (the backtest and data runtimes both have them) reads its keys with.
Everything else belongs to exactly one runtime and lives there, so this
module cannot quietly become the place where cross-runtime coupling
accumulates.
"""

import logging
import math
import warnings
from typing import Any, Dict, List, Mapping

logger = logging.getLogger(__name__)

import pandas as pd

from standard_quant_tools.agent.models import (
    BacktestInput,
    BacktestResult,
    Trade,
)
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.data.bloomberg_provider import BloombergProvider
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import ValidationError

# The initializers come BEFORE the try, never after. Below it they ran
# unconditionally and overwrote a SUCCESSFUL import -- HAS_CPP was False on
# every machine, built extension or not, and the fused technical-indicator
# fast path in research/tools.py that reads these two names was dead code
# wherever it was imported from here.
_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass

#: The interval every backtest tool fetches bars at: `get_ohlcv`'s default,
#: which none of them overrides. Handed to the engine with the bars, so the
#: annualization is the fetched interval's (252 for daily) rather than a
#: guess from the spacing, and a holiday-gapped daily index never warns.
#: A tool that ever fetches another interval must pass that one instead.
FETCH_INTERVAL = "1d"

#: How many offending keys a refusal lists before it summarises the rest.
_KEYS_NAMED = 5


def _named(keys: List[Any]) -> str:
    shown = ", ".join(repr(k) for k in keys[:_KEYS_NAMED])
    more = len(keys) - _KEYS_NAMED
    return shown + (f" and {more} more" if more > 0 else "")


def parse_iso_date(value: Any, field: str, tool: str) -> pd.Timestamp:
    """One ISO date argument parsed, or a refusal naming the field.

    The single-value counterpart of `parse_date_keys`, for a date that sits
    beside a date-keyed map -- a contract's expiry next to its prices -- so
    both refuse the same inputs in the same words."""
    try:
        stamp = pd.to_datetime(str(value), format="ISO8601")
    except (TypeError, ValueError):
        stamp = pd.NaT
    if pd.isna(stamp):
        raise ValidationError(
            f"{tool}: {field}={value!r} is not an ISO date. Write it as "
            "'YYYY-MM-DD'."
        )
    return stamp


def parse_date_keys(
    mapping: Mapping[Any, Any],
    field: str,
    tool: str,
    *,
    finite: bool = False,
) -> Dict[pd.Timestamp, Any]:
    """
    An inline date-keyed map with every key parsed as an ISO date, or a
    refusal that names the keys that are not dates.

    The keys used to be parsed inside the computation -- one
    `pd.Timestamp(key)` at a time, or `pd.to_datetime` over the whole
    index -- so one bad key surfaced as pandas' own "Unknown datetime
    string format" or "doesn't match format" error, naming neither the
    argument nor the remedy. A map keyed by tickers instead of dates, the
    commonest shape of the mistake, failed the same way.

    Parsed as ISO 8601 ('2024-01-02', '2024-01-02T09:30'), the format every
    one of these fields documents. Also refused: two keys naming the same
    instant ('2024-01-02' and '2024-01-02T00:00:00'), which would silently
    keep only one value, and keys mixing time-zone-aware and naive stamps,
    which cannot share one index.

    `finite=True` also requires every value to be a finite number, for maps
    whose values feed arithmetic with no notion of a gap -- a price that is
    NaN or infinite there turns every later equity figure non-finite rather
    than being skipped. Maps where a missing value is meaningful leave it
    False, so their gaps stay governed by the numeric contract.

    Returns a new dict keyed by `pd.Timestamp`, in the input's order.
    """
    keys = list(mapping)
    if not keys:
        return {}
    with warnings.catch_warnings():
        # Mixed offsets come back as an object index with a FutureWarning;
        # that case is refused below, so the warning adds nothing.
        warnings.simplefilter("ignore", FutureWarning)
        try:
            parsed = pd.to_datetime(
                pd.Index([str(k) for k in keys], dtype=object),
                format="ISO8601",
                errors="coerce",
            )
        except (TypeError, ValueError):
            parsed = None
    if parsed is None or not isinstance(parsed, pd.DatetimeIndex):
        raise ValidationError(
            f"{tool}: the keys of {field} mix time-zone-aware and naive dates, "
            "so they cannot share one index. Write every key the same way, "
            "e.g. all as 'YYYY-MM-DD'."
        )
    bad = [key for key, stamp in zip(keys, parsed) if pd.isna(stamp)]
    if bad:
        raise ValidationError(
            f"{tool}: {len(bad)} key(s) of {field} are not ISO dates: "
            f"{_named(bad)}. {field} maps an ISO date ('YYYY-MM-DD') to its "
            "value; a map keyed by tickers or labels is a different input."
        )
    seen: Dict[pd.Timestamp, Any] = {}
    repeated: List[str] = []
    for key, stamp in zip(keys, parsed):
        if stamp in seen:
            repeated.append(f"{seen[stamp]!r} and {key!r}")
        else:
            seen[stamp] = key
    if repeated:
        raise ValidationError(
            f"{tool}: keys of {field} name the same date twice: "
            f"{_named(repeated)}. Only one value per date can be used; keep "
            "the one you mean."
        )
    out: Dict[pd.Timestamp, Any] = {}
    non_finite: List[Any] = []
    for key, stamp in zip(keys, parsed):
        value = mapping[key]
        if finite:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                non_finite.append(key)
                continue
            value = number
        out[stamp] = value
    if non_finite:
        raise ValidationError(
            f"{tool}: {field} has a missing or non-finite value on "
            f"{len(non_finite)} date(s): {_named(non_finite)}. Every value "
            "here enters the running account, where one NaN or infinity makes "
            "every later figure non-finite. Drop those dates or supply the "
            "value."
        )
    return out


def _run_backtest(
    input_data: BacktestInput,
    df: pd.DataFrame,
    signal_series: pd.Series,
    interval: str = FETCH_INTERVAL,
) -> BacktestResult:
    """Shared backtest execution used by all strategy-specific tools.

    `interval` is the interval `df` was fetched at, passed to the engine
    for its annualization.
    """
    logger.debug(
        "[backtest] %s  %s  %s → %s  capital=%.0f",
        input_data.strategy_type,
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.initial_capital,
    )
    results = run_strategy(
        df,
        signal_series,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
        interval=interval,
    )

    trade_log_raw = results.get("trade_log", pd.DataFrame())
    trades = None
    if isinstance(trade_log_raw, pd.DataFrame) and not trade_log_raw.empty:
        trades = [
            Trade(
                entry_date=str(r["entry_date"]),
                exit_date=str(r["exit_date"]),
                direction=str(r["direction"]),
                entry_price=float(r["entry_price"]),
                exit_price=float(r["exit_price"]),
                position_size=float(r.get("position_size", 1.0)),
                return_pct=float(r["return_pct"]),
            )
            for r in trade_log_raw.to_dict(orient="records")
        ]

    bt = BacktestResult(
        total_return=results["total_return"],
        annualized_volatility=results["annualized_volatility"],
        sharpe_ratio=results["sharpe_ratio"],
        sortino_ratio=results["sortino_ratio"],
        max_drawdown=results["max_drawdown"],
        calmar_ratio=results["calmar_ratio"],
        win_rate=results["win_rate"],
        profit_factor=results["profit_factor"],
        num_trades=results["num_trades"],
        avg_trade_return_pct=results["avg_trade_return_pct"],
        final_equity=results["final_equity"],
        equity_curve=results["equity_curve"].tolist(),
        trade_log=trades,
        # run_strategy emits a look-ahead caveat for fill_price="close" (a
        # signal derived from bar t's own Close cannot realistically be
        # filled at that same Close). Rebuilding the result here without it
        # meant the engine knew the simulation might contain look-ahead
        # while the agent-facing output said nothing -- exactly the silent
        # behaviour this library exists to prevent.
        warnings=list(results.get("warnings", [])),
    )
    logger.debug(
        # %s, not %.3f: the ratio is null where it is undefined.
        "[backtest] result  return=%.2f%%  sharpe=%s  maxdd=%.2f%%  trades=%d  win=%.0f%%",
        bt.total_return * 100,
        bt.sharpe_ratio,
        bt.max_drawdown * 100,
        bt.num_trades,
        bt.win_rate * 100,
    )
    return bt
