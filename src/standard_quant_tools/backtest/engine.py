import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.ranking import rank_rows, unrankable_note
from standard_quant_tools.backtest.screens import (  # noqa: F401
    SPLIT_SCREEN_THRESHOLD,
    require_sorted_unique_index,
    split_screen_warnings,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.trend import sma
from standard_quant_tools.metrics.annualization import resolve_periods_per_year
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio,
    has_no_dispersion,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_positive_price_series,
)
from standard_quant_tools.validation import require_finite_array

_VALID_FILL_PRICES = ("close", "next_open", "hl2_exploratory")

# Column order for _cpp_core.batch_run_strategy's flat (num_tests, 11) array
# return -- MUST stay in sync with bindings.cpp's batch_run_strategy binding,
# which writes exactly these 11 columns in exactly this order.
_BATCH_METRIC_COLUMNS = [
    "final_equity",
    "total_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "num_trades",
    "avg_trade_return_pct",
]

# ── Optional C++ fast path ────────────────────────────────────────────────────
from typing import Any as _Any

_cpp_core: _Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_trade_log(
    ref_prices: pd.Series,
    close_prices: pd.Series,
    executed: pd.Series,
    cost_per_unit: float = 0.0,
) -> pd.DataFrame:
    """
    Build a per-trade log from the same events and prices the equity curve
    uses.

    How far the two reconcile, exactly. At zero cost a unit-size lot's
    return_pct equals the equity curve's growth over the lot's bars under
    every fill_price (the next_open / hl2_exploratory legs compound, so a
    held bar earns the full close-to-close move). With costs they agree to
    first order only: return_pct charges each event's cost as a simple
    fraction of the lot's notional, while the equity curve deducts it from
    that bar's equity and compounds, so an entry cost forgoes the lot's
    growth and an exit cost is charged on the drifted notional. The gap is
    about cost * |lot return| per lot -- measured at 15 bps over five years
    of daily bars, at most 0.11 points on one lot and 0.07-0.15 points over
    the whole log, with the log on the high side. A lot sized other than 1
    also differs at zero cost, because a fractional position compounds
    differently from its simple return. Read performance from the equity
    curve; read the log for attribution.

    entry_price/exit_price use ref_prices — the same reference price series
    run_strategy's return calculation uses: Close[i-1] under
    fill_price="close" (since executed[i] = signals[i-1], a position that
    "appears" in `executed` at event date i actually earns its first
    return over Close[i-1] -> Close[i], so i-1's close is its true economic
    entry/exit point — the review's finding), or Open[i] / (High[i]+Low[i])/2
    directly under "next_open"/"hl2_exploratory", where the two-leg
    decomposition already prices entries/exits at that bar's own reference
    price (no shift needed there).

    A "trade" is one LOT: from the moment exposure leaves zero until it
    returns to zero. Same-sign resizes and partial reductions happen
    *inside* a trade rather than ending one. This mirrors
    backtest.cpp::apply_position_event exactly, and that shared definition
    is the point — the two used to disagree. The C++ kernel counted a
    resized lot as one trade while this function emitted two rows for it,
    so a single run_strategy result could report num_trades=1 (read from
    the native kernel) beside a two-row trade_log, with an
    avg_trade_return_pct that matched neither reading. Verified before the
    fix on a 1.0 -> 2.5 -> 0 sequence: native 1 trade / 17.4492% average,
    Python log 2 trades / 8.5113% average, from the identical inputs.

    Cost accounting follows the same shared model. Each position-changing
    event is charged abs(pdiff) * cost_per_unit — the amount actually
    transacted at that event, which is what run_strategy deducts from the
    equity curve. The old close-and-reopen reading of a resize charged
    2*(1.0 + 2.5) = 7 units of cost where the equity curve charged
    1.0 + 1.5 + 2.5 = 5, so trade-log P&L and equity P&L could not be
    reconciled for any strategy that scales a position. cost_per_unit is a
    cost per unit of *notional exposure traded*, so a 5x-leveraged trade
    pays 5x what a 1x trade pays.

    A lot still open at the final bar is flushed as a synthesized
    mark-to-market exit at the final Close (equity is marked to Close
    regardless of fill_price). No exit cost is charged for it, because no
    exit event occurred and the equity curve never deducted one either.

    entry_price/exit_price use ref_prices — the same reference price series
    run_strategy's return calculation uses: Close[i-1] under
    fill_price="close" (since executed[i] = signals[i-1], a position that
    "appears" in `executed` at event date i actually earns its first
    return over Close[i-1] -> Close[i], so i-1's close is its true economic
    entry/exit point), or Open[i] / (High[i]+Low[i])/2 directly under
    "next_open"/"hl2_exploratory", where the two-leg decomposition already
    prices entries/exits at that bar's own reference price (no shift
    needed there). For a lot that was resized, entry_price is the
    weighted-average cost basis across the whole lot rather than the price
    of its first leg — that is the price its reported return is actually
    measured against.

    position_size is the signed peak exposure the lot ever carried (2.5 for
    a lot that went 1.0 -> 2.5), not just its sign: run_strategy's own
    return calculation multiplies the raw price return by the executed
    signal value, so return_pct scales with size too. direction
    ("long"/"short") is a readable label derived from its sign.

    Vectorized detection of position changes; only iterates over trade
    events (orders-of-magnitude fewer than bars).
    """
    pos_diff = executed.diff()
    pos_diff.iloc[0] = executed.iloc[0]

    trade_event_idx = pos_diff[pos_diff != 0].index
    if len(trade_event_idx) == 0:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "direction",
                "entry_price",
                "exit_price",
                "position_size",
                "return_pct",
            ]
        )

    records: List[Dict[str, Any]] = []
    # The open lot: None when flat. Mirrors backtest.cpp's PositionState,
    # plus the reporting fields (entry_date / peak_size) the C++ side has
    # no need for because it only accumulates scalar stats.
    lot: Optional[Dict[str, Any]] = None

    def _close_record(exit_date: Any, exit_price: float, extra_pnl: float) -> None:
        """Emit the finished lot. extra_pnl is the P&L of the closing leg
        for a real exit (already folded into realized_pnl by the caller,
        so 0.0 there) or the mark-to-market P&L of the still-open remainder
        for the final-bar flush."""
        assert lot is not None
        peak = lot["peak_size"]
        net_pnl = lot["realized_pnl"] + extra_pnl - lot["cost_accrued"]
        records.append(
            {
                "entry_date": lot["entry_date"],
                "exit_date": exit_date,
                "direction": "long" if peak > 0 else "short",
                "entry_price": round(float(lot["cost_basis"]), 4),
                "exit_price": round(float(exit_price), 4),
                "position_size": round(float(peak), 4),
                "return_pct": round(float(net_pnl) * 100, 4),
            }
        )

    for date in trade_event_idx:
        # .loc, not []: bare [] on a Series is positional for an integer
        # index and label-based otherwise, so it silently changed meaning
        # with the index type pandas happened to infer.
        ref_price = float(ref_prices.loc[date])
        new_pos = float(executed.loc[date])
        pdiff = float(pos_diff.loc[date])

        if lot is not None and (pdiff > 0) != (lot["size"] > 0):
            # Opposite sign: reduce, fully close, or close-then-flip. Only
            # the quantity that actually offsets existing exposure is
            # closed here; a flip's fresh leg is opened by the block below.
            pos_sign = 1.0 if lot["size"] > 0 else -1.0
            closing_qty = min(abs(pdiff), abs(lot["size"]))
            lot["cost_accrued"] += closing_qty * cost_per_unit
            basis = lot["cost_basis"]
            if basis != 0.0:
                lot["realized_pnl"] += (
                    (ref_price - basis) / basis * (closing_qty * pos_sign)
                )
            lot["size"] -= closing_qty * pos_sign

            if lot["size"] == 0.0:
                _close_record(date, ref_price, 0.0)
                lot = None
        elif lot is not None:
            # Same sign: a resize/add. Blend the cost basis and charge only
            # the incremental amount transacted. This does NOT complete a
            # trade — the lot lives on.
            old_notional = lot["size"] * lot["cost_basis"]
            lot["size"] += pdiff
            lot["cost_basis"] = (old_notional + pdiff * ref_price) / lot["size"]
            lot["cost_accrued"] += abs(pdiff) * cost_per_unit
            if abs(lot["size"]) > abs(lot["peak_size"]):
                lot["peak_size"] = lot["size"]
            continue

        if lot is None and new_pos != 0.0:
            # Opening a fresh lot — either already flat, or the branch
            # above just fully closed the prior one (a flip). Uses the raw
            # target position, not a delta-derived value.
            lot = {
                "entry_date": date,
                "size": new_pos,
                "peak_size": new_pos,
                "cost_basis": ref_price,
                "cost_accrued": abs(new_pos) * cost_per_unit,
                "realized_pnl": 0.0,
            }

    # Flush a lot still open at the last bar (buy-and-hold, trend
    # strategies that never exit). Marked to the final Close, not
    # ref_prices, and charged no exit cost.
    if lot is not None:
        last_price = float(close_prices.iloc[-1])
        basis = lot["cost_basis"]
        mtm = (last_price - basis) / basis * lot["size"] if basis != 0.0 else 0.0
        _close_record(close_prices.index[-1], last_price, mtm)

    return pd.DataFrame(records)


def _compute_trade_stats(trade_log: pd.DataFrame) -> Dict[str, float]:
    if trade_log.empty:
        return {
            "win_rate": 0.0,
            # NaN, not 0.0: 0.0 is what "every trade lost" produces, and
            # nothing traded. backtest.cpp starts from the same NaN.
            "profit_factor": float("nan"),
            "num_trades": 0,
            "avg_trade_return_pct": 0.0,
        }

    num_trades = len(trade_log)
    winners = trade_log[trade_log["return_pct"] > 0]
    losers = trade_log[trade_log["return_pct"] <= 0]

    win_rate = len(winners) / num_trades
    gross_profit = float(winners["return_pct"].to_numpy(dtype=float).sum())
    gross_loss = float(np.abs(losers["return_pct"].to_numpy(dtype=float)).sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    return {
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "num_trades": num_trades,
        "avg_trade_return_pct": round(float(trade_log["return_pct"].mean()), 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core engine
# ──────────────────────────────────────────────────────────────────────────────


def _turnover_and_cost(signals: pd.Series, cost_per_unit: float) -> Dict[str, float]:
    """Position changed, summed over bars, and the cost that charged --
    the same lagged positions the returns are computed on."""
    executed = signals.shift(1).fillna(0.0)
    pos_diff = executed.diff().fillna(executed.iloc[0])
    turnover = float(pos_diff.abs().sum())
    return {
        "turnover": round(turnover, 6),
        "realized_cost_pct": round(turnover * float(cost_per_unit), 6),
    }


def _undefined_ratios_as_nan(
    sortino: Any,
    calmar: Any,
    profit_factor: Any,
    total_return: Any,
    annualized_vol: Any,
    num_trades: Any,
    risk_free_rate: float,
) -> tuple:
    """
    The 0/0 convention for a native result, applied at the boundary.

    backtest.cpp now returns NaN, not +inf, for a Sortino or Calmar over an
    empty denominator when the numerator is zero too, and NaN, not 0.0, for
    the profit factor of a run that never traded -- the same convention
    `risk_metrics` and `_compute_trade_stats` follow. An extension compiled
    before that change keeps the old values until it is rebuilt, and the old
    +inf ranked a do-nothing parameter set first, so the rule is enforced
    here as well, the way the Sharpe convention is below. Idempotent after a
    rebuild. Works on scalars and on whole grid columns alike.

    A Sortino is 0/0 only when every excess return is exactly zero: a book
    that never moved under a zero rate (a positive rate makes the first bar
    downside; a negative one makes every flat bar a gain, and +inf is right).
    A Calmar over no drawdown is 0/0 when the curve did not grow.
    """
    total_return = np.asarray(total_return, dtype=float)
    no_motion = (total_return == 0.0) & (np.asarray(annualized_vol, dtype=float) == 0.0)
    sortino = np.asarray(sortino, dtype=float)
    calmar = np.asarray(calmar, dtype=float)
    sortino = np.where(
        np.isinf(sortino) & no_motion & (risk_free_rate >= 0.0), np.nan, sortino
    )
    calmar = np.where(np.isinf(calmar) & (total_return <= 0.0), np.nan, calmar)
    profit_factor = np.where(
        np.asarray(num_trades) == 0, np.nan, np.asarray(profit_factor, dtype=float)
    )
    return sortino, calmar, profit_factor


def run_strategy(
    price_data: pd.DataFrame,
    signal_series: pd.Series,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    include_trade_log: bool = False,
    fill_price: str = "close",
    risk_free_rate: float = 0.0,
    adjusted: Optional[bool] = None,
    periods_per_year: Optional[int] = None,
    interval: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Vectorized backtesting engine with transaction costs.

    Args:
        price_data: DataFrame with 'Close' column (and 'Open' if fill_price="next_open").
            Its index must be sorted and unique: bar order is time order.
        signal_series: Series of 1 (long), 0 (flat), -1 (short), aligned to
            price_data by date. Its index must be unique.
        periods_per_year: bars per year for every annualized metric
            (volatility, Sharpe, Sortino, Calmar). A positive whole number.
        interval: the interval the bars were fetched at ("1d", "1wk",
            "1mo", ...), used for periods_per_year when that is not given.
            With neither, the number is read off the bar spacing (daily ->
            252, weekly -> 52, monthly -> 12, quarterly -> 4), and when the
            spacing cannot say (intraday bars, irregular spacing, a
            non-date index) 252 is used and a warning says so. The value
            used and where it came from are returned as `periods_per_year`
            and `periods_per_year_source`. This engine used 252 always, so
            weekly bars reported a volatility 2.2x too high and monthly bars
            a CAGR of 38%/yr against a true 1.6%.
        initial_capital: Starting capital.
        commission_pct: Commission per unit of position changed (default 0.1%).
        slippage_pct: Slippage per unit of position changed (default 0.05%).
        include_trade_log: If True, build and return per-trade log.
        risk_free_rate: Annualized risk-free rate as a decimal fraction
            (0.045 = 4.5%), subtracted per period before Sharpe and
            Sortino. Defaults to 0.0 — the value this engine always
            assumed — so an unset rate cannot move a number that was
            already reported. Passed identically to the native kernel and
            the Python fallback: a machine with the C++ extension built
            must not report a different ratio from one without it.
        adjusted: whether the bars are split- and dividend-adjusted, when
            the caller knows (a provider's DataSetMetadata.adjusted). Read
            from `price_data.attrs['adjusted']` when not passed. Used only
            to phrase the split screen's warning: this engine compounds
            every bar return, so an unadjusted split prints as a real
            -50% bar (a 10:1 split reported buy-and-hold at -62% against
            +276% true, findings D6).
        fill_price: "close" (default) — a signal known at bar t-1's close is
            assumed filled at that same close, earning bar t's full
            close-to-close return. "next_open" — decomposes each bar into
            an overnight leg (prior close -> this bar's open, priced at
            yesterday's position) and an intraday leg (this bar's open ->
            close, priced at today's position), so an entry only earns its
            own open-to-close move, an exit still bears the overnight gap
            it was held through before selling at the open, and a held
            position compounds the two legs, earning exactly the
            close-to-close move. (They used to be added, which left the
            equity curve 0.2-0.5 points below the fill-to-fill trade log
            over five years of daily bars at zero cost.)
            "hl2_exploratory" — identical two-leg decomposition, but using
            that bar's own (High + Low) / 2 ("HL2") as the reference fill
            price instead of Open. This is NOT a bid/ask midpoint quote —
            it requires knowing the bar's High and Low, which are only
            determined once the bar has already completed, so pricing a
            fill at a bar's own HL2 is look-ahead the same way fill_price=
            "close" is (see the warning below); the name says "exploratory"
            deliberately, so it's never mistaken for a real, tradable
            execution price. entry_price/exit_price in the trade log use
            this same fill-mode-aware reference price, not always Close —
            see _build_trade_log.

    Returns:
        Dict with performance metrics, equity curve, `periods_per_year` and
        `periods_per_year_source`, and optionally trade_log. When fill_price
        is "close" or "hl2_exploratory", result["warnings"] includes a
        look-ahead-bias caveat (see below). sortino_ratio and calmar_ratio
        are +inf over an empty denominator with a positive numerator and
        NaN when both are zero (a book that never moved); profit_factor is
        NaN when nothing traded.

    Raises:
        ValidationError: fill_price is not one of "close", "next_open",
            "hl2_exploratory"; a cost, the capital, the risk-free rate or
            periods_per_year is out of range; or price_data's index is
            unsorted or duplicated, or signal_series' index is duplicated.
    """
    if fill_price not in _VALID_FILL_PRICES:
        raise ValidationError(
            f"fill_price must be one of {_VALID_FILL_PRICES}, got {fill_price!r}"
        )
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValidationError(
            f"initial_capital must be positive and finite, got {initial_capital!r} "
            "— a zero/negative/non-finite value silently produces inf/nan in "
            "total_return and calmar_ratio instead of a meaningful result."
        )
    for name, value in (
        ("commission_pct", commission_pct),
        ("slippage_pct", slippage_pct),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValidationError(
                f"{name} must be non-negative and finite, got {value!r}"
            )
    # Up front, for both paths. The native binding refuses a NaN rate and
    # sharpe_ratio refuses one on the Python path, but only after the whole
    # simulation has run -- and a build that predated the binding check
    # returned a NaN Sharpe beside a +inf Sortino. Any sign is allowed:
    # negative policy rates are real.
    require_finite_scalar(risk_free_rate, "risk_free_rate", "run_strategy")

    # Columns each fill mode actually reads — checked up front so a missing
    # one is a clear error naming the mode that needs it, not a raw KeyError
    # from deep inside the return calculation.
    _required_cols = {
        "close": ("Close",),
        "next_open": ("Close", "Open"),
        "hl2_exploratory": ("Close", "High", "Low"),
    }[fill_price]
    missing_cols = [c for c in _required_cols if c not in price_data.columns]
    if missing_cols:
        raise ValidationError(
            f"price_data is missing column(s) {missing_cols} required for "
            f"fill_price={fill_price!r}"
        )

    # Bar order is time order on both paths below, and the intersection
    # keeps price_data's own order, so an unsorted price index is refused
    # here, before anything is computed from it. The signal is read onto
    # those bars by label, so only a repeated signal date matters.
    require_sorted_unique_index(price_data.index, "price_data", "run_strategy")
    require_sorted_unique_index(
        signal_series.index, "signal_series", "run_strategy", order_matters=False
    )

    # Fast path: skip the intersection + two .loc[] calls entirely when the
    # indices are already identical (the common case for a signal derived
    # directly from price_data) -- .equals() is a cheap array comparison,
    # intersection+loc is real allocation work neither index needs here.
    if price_data.index.equals(signal_series.index):
        idx = price_data.index
    else:
        idx = price_data.index.intersection(signal_series.index)
    prices = price_data.loc[idx, "Close"]
    signals = signal_series.loc[idx]

    # Finite-input contract, enforced once here for EVERY path.
    #
    # This used to live inside the C++ branch only, which made the contract
    # depend on whether the extension happened to be built: the same call with
    # the same data raised ValidationError with _sqt_core present and silently
    # produced NaN metrics without it. It also never covered fill_price=
    # "next_open"/"hl2_exploratory" at all, where a NaN reference price is
    # worse than merely NaN-poisoning the result -- pandas' cumprod() is
    # skipna=True, so the NaN bar's return is silently DROPPED from the
    # compounded equity curve and total_return is computed over a quietly
    # shortened series that still looks like a complete one.
    if len(idx) == 0:
        # After intersecting price dates with signal dates there may be
        # nothing left. Everything below assumes at least one bar, and the
        # failure surfaced far from here as an empty-slice error.
        raise ValidationError(
            "price_data and signal_series share no dates, so there is nothing "
            "to backtest. Check that the two are on the same calendar and "
            "cover overlapping ranges."
        )
    # Resolved once, on the bars actually backtested, and passed to BOTH
    # paths -- the native kernel took this number all along and was always
    # handed 252.0.
    ppy, ppy_source, ppy_warnings = resolve_periods_per_year(
        idx,
        periods_per_year=periods_per_year,
        interval=interval,
        where="run_strategy",
    )
    prices_arr = prices.to_numpy(dtype=np.float64)
    signals_arr = signals.to_numpy(dtype=np.float64)
    # STRICTLY POSITIVE, not merely finite. Every price column here feeds a
    # ratio -- pct_change for Close, open/prev_close and close/open for the
    # fill-aware paths -- so 0.0 divides by zero and a negative price flips
    # the sign of the return it produces. Both are perfectly finite, so the
    # finite check alone passed them: a single Close of -5.0 produced a
    # total_return of +0.397914 (a plausible profit computed through a
    # negative price), and a Close of 0.0 produced a silent -1.0 wipeout.
    require_positive_price_series(prices, "Close", "run_strategy", allow_nan=False)
    require_finite_array(signals_arr, "signals", "run_strategy")
    if fill_price == "next_open":
        require_positive_price_series(
            price_data.loc[idx, "Open"], "Open", "run_strategy", allow_nan=False
        )
    elif fill_price == "hl2_exploratory":
        require_positive_price_series(
            price_data.loc[idx, "High"], "High", "run_strategy", allow_nan=False
        )
        require_positive_price_series(
            price_data.loc[idx, "Low"], "Low", "run_strategy", allow_nan=False
        )

    n_bars = len(prices)
    logger.debug(
        "[run_strategy] bars=%d  capital=%.0f  commission=%.4f  slippage=%.4f  fill_price=%s",
        n_bars,
        initial_capital,
        commission_pct,
        slippage_pct,
        fill_price,
    )

    # `returns`/`executed` are NOT computed here anymore -- the C++ path
    # below needs neither (it recomputes both internally from raw prices/
    # signals), and building them unconditionally was pure waste whenever
    # the C++ kernel actually ran. Each is now computed only where it's
    # actually used: `executed` lazily inside the C++ branch (only if
    # include_trade_log requests a Python-side trade log) or unconditionally
    # at the top of the Python fallback branch below (where both are
    # genuinely needed for the return/cost calculation itself).

    warnings: List[str] = list(ppy_warnings)
    # ── The split screen (findings D6) ──────────────────────────────────
    # Nothing under backtest/ read the provider's `adjusted` flag, and a
    # split on unadjusted bars is a real -50% bar to this engine: LRCX's
    # 10:1 split reported buy-and-hold at -62.40% against +276.04% true,
    # and a short held through it printed a fictitious +93%. The engine
    # already walks every bar for the total-loss guard; this pass is free.
    warnings.extend(
        split_screen_warnings(
            prices,
            adjusted if adjusted is not None else price_data.attrs.get("adjusted"),
        )
    )
    if fill_price == "close":
        warnings.append(
            "fill_price='close': a signal known at bar t-1's close is assumed filled "
            "at that same close. If signal_series was derived from that bar's own "
            "Close (e.g. a same-day indicator/score), this is a look-ahead bias — the "
            "trade could not actually have been placed at that price in real time. "
            "Use fill_price='next_open' for a lookahead-free simulation."
        )
    elif fill_price == "hl2_exploratory":
        warnings.append(
            "fill_price='hl2_exploratory': fills at a bar's own (High + Low) / 2 — "
            "not a real bid/ask midpoint quote, and not knowable until that bar has "
            "already completed (High/Low are only determined in retrospect), so this "
            "is look-ahead the same way fill_price='close' is. Intended for "
            "exploratory analysis only; use fill_price='next_open' for a "
            "lookahead-free simulation."
        )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    # Pass raw signals — C++ applies the one-bar lag internally (executed[i] = signals[i-1]).
    # Do NOT pass `executed` here: it is already shifted, which would cause a 2-bar lag.
    #
    # EVERY fill mode runs here now. The kernel used to know only Close
    # prices, so "next_open" and "hl2_exploratory" always fell back to
    # Python -- which meant the MORE REALISTIC execution model was also the
    # slow one, and the native grid could not be used for it at all. The
    # kernel now takes an optional per-bar reference (fill) price and applies
    # the same two-leg overnight/intraday decomposition this module's Python
    # fallback does.
    # ── total-loss guard, BEFORE either path ─────────────────────────────
    # A bar return at or below -100% wipes the account out. Neither engine
    # models that: both compound `1 + r` unguarded, so equity goes NEGATIVE
    # and then keeps compounding. A 1x short (signal -1.0, inside the
    # documented {-1, 0, 1}) through a +200% bar gave
    # [10000, 10000, 10000, -10000, -11666, -12833] -- equity getting MORE
    # negative on bars where the short profits, max_drawdown -2.283 (deeper
    # than a total loss), and at 2x through a -60% bar a sharpe_ratio of
    # +2.5923 on a dead account.
    #
    # This refuses rather than truncating, because the engine has no
    # bankruptcy model to truncate INTO -- no margin call, no forced
    # liquidation, no borrow. Returning a number for a scenario it cannot
    # represent is what produced the +2.59 Sharpe. `run_portfolio_simulation`
    # models the account properly and is the tool for leveraged shorts.
    #
    # Computed from `prices` and `signals` directly because `returns` is
    # built after the dispatch below; it is the same one-bar lag both paths
    # apply.
    _wipeout = signals.shift(1).fillna(0.0) * prices.pct_change(
        fill_method=None
    ).fillna(0.0)
    if bool((_wipeout <= -1.0).any()):
        _at = _wipeout.index[_wipeout <= -1.0][0]
        raise ValidationError(
            f"run_strategy: the position loses "
            f"{float(_wipeout.loc[_at]) * 100:.1f}% of the account on the bar "
            f"at {_at}, which is a total loss. This engine compounds "
            f"(1 + r) with no bankruptcy model, so it would carry equity "
            f"negative from there and report a drawdown deeper than -100% "
            f"and a Sharpe computed on a dead account. Reduce the signal "
            f"magnitude, or use run_portfolio_simulation, which models cash "
            f"and margin."
        )

    if HAS_CPP and _cpp_core is not None:
        logger.debug("[run_strategy] using C++ kernel  fill_price=%s", fill_price)
        ref_arr = None
        if fill_price == "next_open":
            ref_arr = price_data.loc[idx, "Open"].to_numpy(dtype=np.float64)
        elif fill_price == "hl2_exploratory":
            ref_arr = (
                (price_data.loc[idx, "High"] + price_data.loc[idx, "Low"]) / 2.0
            ).to_numpy(dtype=np.float64)
        # prices_arr/signals_arr were built and validated above, for every
        # path — not just this one.
        r = _cpp_core.run_strategy(
            prices_arr,
            signals_arr,
            initial_capital,
            commission_pct,
            slippage_pct,
            float(ppy),
            ref_arr,
            risk_free_rate,
        )
        equity_curve = pd.Series(r["equity_curve"], index=idx)

        # THE ZERO-DISPERSION CONVENTION, applied at the boundary.
        # `backtest.cpp` returned 0.0 for a Sharpe with no dispersion where
        # `metrics/risk_metrics.py` returns NaN, and that is not cosmetic:
        # in `backtest_grid` a no-trade combination scored 0.0 natively and
        # sorted ABOVE genuinely losing combinations, while in Python it
        # sorts to the bottom as NaN -- the two backends ranked the same
        # grid differently.
        #
        # The kernel source is fixed too, but a compiled extension already
        # in the tree keeps the old value until it is rebuilt, so the
        # convention is enforced here as well. Idempotent: after a rebuild
        # the kernel returns NaN and this changes nothing.
        native_returns = equity_curve.pct_change(fill_method=None).dropna().to_numpy()
        r = dict(r)
        if native_returns.size and has_no_dispersion(native_returns):
            r["sharpe_ratio"] = float("nan")
        # The 0/0 convention for Sortino, Calmar and a no-trade profit
        # factor, enforced here for the same reason -- see the helper.
        sortino_v, calmar_v, pf_v = _undefined_ratios_as_nan(
            r["sortino_ratio"],
            r["calmar_ratio"],
            r["profit_factor"],
            r["total_return"],
            r["annualized_volatility"],
            r["num_trades"],
            risk_free_rate,
        )
        r["sortino_ratio"] = float(sortino_v)
        r["calmar_ratio"] = float(calmar_v)
        r["profit_factor"] = float(pf_v)
        # win_rate/profit_factor/num_trades/avg_trade_return_pct: read
        # straight from the native result. backtest.cpp's own trade-log
        # logic uses the identical convention _build_trade_log does
        # (entry_price = prices[i-1], entry_size = signal magnitude, cost
        # scaled by position size) and this session's own CI verification
        # work (TestNativeTradeStatsCorrectness, run against a real
        # compiled _sqt_core on live CI, not just locally) already
        # confirmed native and Python trade stats agree exactly -- so
        # rebuilding the full Python trade log here just to recompute
        # numbers the C++ kernel already returned was pure redundant work,
        # not a correctness requirement. The Python trade log itself is
        # still built below, but only when include_trade_log actually asks
        # for the DataFrame, not for its stats.
        result: Dict[str, Any] = {
            "final_equity": round(float(r["final_equity"]), 2),
            "total_return": round(float(r["total_return"]), 6),
            "annualized_volatility": round(float(r["annualized_volatility"]), 6),
            "sharpe_ratio": round(float(r["sharpe_ratio"]), 4),
            "sortino_ratio": round(float(r["sortino_ratio"]), 4),
            "max_drawdown": round(float(r["max_drawdown"]), 6),
            "calmar_ratio": round(float(r["calmar_ratio"]), 4),
            "num_trades": int(r["num_trades"]),
            "win_rate": round(float(r["win_rate"]), 4),
            "profit_factor": round(float(r["profit_factor"]), 4),
            "avg_trade_return_pct": round(float(r["avg_trade_return_pct"]), 4),
            "equity_curve": equity_curve,
            "warnings": warnings,
            "periods_per_year": ppy,
            "periods_per_year_source": ppy_source,
        }
        # Turnover and the cost it realized, which the Python path computes
        # on its way to the returns and this path recomputes here from the
        # same lagged positions; both were dropped on the floor before.
        result.update(_turnover_and_cost(signals, commission_pct + slippage_pct))
        if include_trade_log:
            executed = signals.shift(1).fillna(0.0)
            # Same reference-price convention the Python path uses: Close[i-1]
            # under "close" (a position appearing at bar i earns its first
            # return over Close[i-1] -> Close[i]), or that bar's own fill
            # price under the fill-aware modes, where the two-leg
            # decomposition already prices entries and exits there.
            if fill_price == "next_open":
                ref_for_log = price_data.loc[idx, "Open"]
            elif fill_price == "hl2_exploratory":
                ref_for_log = (
                    price_data.loc[idx, "High"] + price_data.loc[idx, "Low"]
                ) / 2.0
            else:
                ref_for_log = prices.shift(1)
            result["trade_log"] = _build_trade_log(
                ref_for_log,
                prices,
                executed,
                commission_pct + slippage_pct,
            )
        logger.debug(
            "[run_strategy] C++  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
            result["total_return"] * 100,
            result["sharpe_ratio"],
            result["num_trades"],
            result["max_drawdown"] * 100,
        )
        return result

    # ── Python fallback ───────────────────────────────────────────────────────
    logger.debug("[run_strategy] using Python fallback  fill_price=%s", fill_price)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    executed = signals.shift(1).fillna(0.0)
    cost_per_unit = commission_pct + slippage_pct
    pos_diff = executed.diff().fillna(executed.iloc[0])
    transaction_costs = pos_diff.abs() * cost_per_unit

    if fill_price in ("next_open", "hl2_exploratory"):
        # Two-leg decomposition, correct for entries, continuations, exits,
        # and same-bar flips alike:
        #   overnight leg (Close[t-1] -> ref_price[t]) priced at YESTERDAY's
        #     position (executed.shift(1)) — captures the gap a position
        #     still held overnight is exposed to, including on an exit bar
        #     (sold at today's reference price, so still exposed to the
        #     overnight gap but not today's remaining move).
        #   intraday leg (ref_price[t] -> Close[t]) priced at TODAY's
        #     position (executed) — captures a same-day entry's move from
        #     the reference price to the close, and a held-through day's
        #     remaining move.
        # The legs COMPOUND: the intraday leg is earned on the equity the
        # overnight leg left, so an unchanged position earns exactly the
        # close-to-close move. They used to be summed, which dropped the
        # product term -- over five years of daily bars at zero cost that
        # put the equity curve 0.2-0.5 points below the fill-to-fill trade
        # log, the curve being the approximate side. backtest.cpp's
        # gross_return_at compounds them identically. "next_open" uses that
        # bar's Open as the reference price; "hl2_exploratory" uses (High + Low) / 2 —
        # NOT a real bid/ask midpoint, and only knowable after the bar has
        # already completed (see the look-ahead warning above).
        if fill_price == "next_open":
            ref_prices = price_data.loc[idx, "Open"]
        else:
            ref_prices = (
                price_data.loc[idx, "High"] + price_data.loc[idx, "Low"]
            ) / 2.0
        overnight_leg = ((ref_prices - prices.shift(1)) / prices.shift(1)).fillna(0.0)
        intraday_leg = (prices - ref_prices) / ref_prices
        executed_prev = executed.shift(1).fillna(0.0)
        gross_returns = (1.0 + executed_prev * overnight_leg) * (
            1.0 + executed * intraday_leg
        ) - 1.0
        strategy_returns = gross_returns - transaction_costs
    else:
        strategy_returns = executed * returns - transaction_costs
        # executed[i] = signals[i-1], so a position "appearing" in `executed`
        # at bar i actually earns its first return over Close[i-1] -> Close[i]
        # — Close[i-1] is its true economic entry/exit reference, not Close[i]
        # (only used for the trade log below; the return calc above is
        # already correct as-is).
        ref_prices = prices.shift(1)
    # A BAR RETURN AT OR BELOW -100% WIPES THE ACCOUNT OUT; it does not
    # take equity negative. An unguarded cumprod does exactly that, and
    # then keeps compounding: a 1x short (signal -1.0, inside the
    # documented {-1, 0, 1}) through a +200% bar produced
    # [10000, 10000, 10000, -10000, -12500, -11250] -- equity getting MORE
    # negative on bars where the short profits, a max_drawdown of -2.283
    # (deeper than a total loss), and at 2x leverage through a -60% bar a
    # sharpe_ratio of +2.5923 on a dead account.
    #
    # Floored at zero and held there, which is what a broker does.
    # `run_portfolio_simulation` already models this; `run_strategy` did
    # not, and it is what run_custom_signal_backtest, run_backtest_compact,
    # run_signal_panel_backtest, run_strategy_matrix and every backtest_grid
    # run through.
    growth = (1 + strategy_returns).clip(lower=0.0)
    equity_curve = initial_capital * growth.cumprod()
    if bool((strategy_returns <= -1.0).any()):
        first = strategy_returns.index[strategy_returns <= -1.0][0]
        logger.warning(
            "[run_strategy] a bar return of %.4f at %s is a total loss; "
            "equity is floored at zero from there rather than compounding "
            "negative.",
            float(strategy_returns.loc[first]),
            first,
        )

    total_ret = cumulative_return(equity_curve)
    annual_vol = annualized_volatility(strategy_returns, ppy)
    sr = sharpe_ratio(strategy_returns, risk_free_rate, ppy)
    srt = sortino_ratio(strategy_returns, risk_free_rate, ppy)
    mdd = max_drawdown(equity_curve)
    cal = calmar_ratio(equity_curve, ppy)
    final_eq = (
        float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital
    )

    result = {
        "final_equity": round(final_eq, 2),
        "total_return": round(total_ret, 6),
        "annualized_volatility": round(annual_vol, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(srt, 4),
        "max_drawdown": round(mdd, 6),
        "calmar_ratio": round(cal, 4),
        "equity_curve": equity_curve,
        "warnings": warnings,
        # Already computed above for the returns; returned rather than
        # discarded (findings: 'computed at engine.py:604-605 and thrown
        # away').
        "turnover": round(float(pos_diff.abs().sum()), 6),
        "realized_cost_pct": round(float(transaction_costs.sum()), 6),
        "periods_per_year": ppy,
        "periods_per_year_source": ppy_source,
    }

    trade_log = _build_trade_log(ref_prices, prices, executed, cost_per_unit)
    result.update(_compute_trade_stats(trade_log))

    if include_trade_log:
        result["trade_log"] = trade_log

    logger.debug(
        "[run_strategy] Python  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
        result["total_return"] * 100,
        result["sharpe_ratio"],
        result["num_trades"],
        result["max_drawdown"] * 100,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Grid search — module-level worker (must be picklable for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────


def _fused_crossover_metrics(
    price_data: pd.DataFrame,
    keys: List[str],
    combos: List[tuple],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    ref_arr: Optional[np.ndarray],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Optional[np.ndarray]:
    """
    Fused path for two-moving-average crossover grids.

    Profiling a 300-combination x 5,000-bar SMA grid showed the batch kernel
    was solving the small half of the problem:

        python signal generation   121.4 ms   92.1%
        vstack into (combos,bars)    3.2 ms    2.4%
        native batch backtest        7.2 ms    5.4%

    and the grid computed 600 moving averages where only 35 UNIQUE periods
    existed -- every combination recomputing an average another combination
    had already produced.

    So this computes each unique SMA ONCE (through the same `sma` the
    strategy itself uses, so there is no second definition of the indicator
    to drift), hands the resulting (n_unique x n_bars) matrix to C++ with a
    (num_combos x 2) index pair per combination, and lets the kernel build
    each signal into one reusable buffer and backtest it immediately.

    Peak memory becomes O(n_unique * n_bars) instead of
    O(num_combos * n_bars): the 50,000-combination cap over 100,000 bars was
    a 40 GB signal matrix, and is now bounded by the number of distinct
    periods.

    Returns None when this grid is not a plain fast/slow crossover, in which
    case the caller falls back to the general path.
    """
    if sorted(keys) != ["fast_period", "slow_period"]:
        return None
    fast_i, slow_i = keys.index("fast_period"), keys.index("slow_period")

    periods = sorted(
        {int(c[fast_i]) for c in combos} | {int(c[slow_i]) for c in combos}
    )
    row_of = {p: i for i, p in enumerate(periods)}
    close = price_data["Close"]
    indicators = np.empty((len(periods), len(close)), dtype=np.float64)
    for period, row in row_of.items():
        indicators[row, :] = sma(close, period).to_numpy(dtype=np.float64)

    pair_idx = np.asarray(
        [[row_of[int(c[fast_i])], row_of[int(c[slow_i])]] for c in combos],
        dtype=np.int32,
    )
    return _cpp_core.batch_backtest_crossover(
        close.to_numpy(dtype=np.float64),
        indicators,
        pair_idx,
        initial_capital,
        commission_pct,
        slippage_pct,
        float(periods_per_year),
        ref_arr,
        risk_free_rate,
    )


def _run_grid_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for backtest_grid. Must live at module level to be
    picklable by ProcessPoolExecutor on Windows (spawn start method).

    Only reached via ProcessPoolExecutor when `strategy` was a registry
    name (see backtest_grid) — a raw user callable is never sent through
    this path, since arbitrary callables (lambdas, closures) are frequently
    unpicklable across the spawn boundary.
    """
    df = job["price_data"]
    signal_fn = STRATEGY_REGISTRY[job["strategy"]]
    signals = signal_fn(df, **job["params"])

    result = run_strategy(
        df,
        signals,
        initial_capital=job["initial_capital"],
        commission_pct=job["commission_pct"],
        slippage_pct=job["slippage_pct"],
        fill_price=job.get("fill_price", "close"),
        # A grid that ranked on a zero-rate Sharpe while the single run it
        # is compared against used a real one would pick a different
        # winner, and nothing in either result would say why.
        risk_free_rate=job.get("risk_free_rate", 0.0),
        # Resolved once by backtest_grid for the whole grid, so every row is
        # annualized alike and the per-row run does not re-infer (or re-warn).
        periods_per_year=job.get("periods_per_year"),
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.pop("periods_per_year", None)
    result.pop("periods_per_year_source", None)
    result.update(job["params"])
    return result


def _checked_custom_signal(signals: Any, label: str) -> pd.Series:
    """Refuse a custom callable's signal outside [-1, 1] by name."""
    series = pd.Series(signals)
    values = series.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.abs(finite).max()) > 1.0 + 1e-9:
        worst = float(finite[np.argmax(np.abs(finite))])
        raise ValidationError(
            f"backtest_grid: custom strategy {label!r} produced a signal of "
            f"{worst:+.4f}, outside [-1, 1]. run_strategy multiplies the lagged "
            "signal into the bar return, so a value of 2.0 is a 2x levered "
            "position and nothing in the result would say so. Scale the "
            "signal to [-1, 1], or run run_portfolio_simulation, which models "
            "leverage explicitly."
        )
    return series


def _run_signal_fn_job(
    price_data: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    params: Dict[str, Any],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
    fill_price: str = "close",
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sequential-only counterpart to _run_grid_job for a user-supplied signal
    callable. Always runs in the calling process (never via
    ProcessPoolExecutor), so signal_fn need not be picklable.

    `risk_free_rate` reaches `run_strategy` here as it does in
    `_run_grid_job`; it used to be dropped on this path alone, so a
    callable strategy's grid Sharpe disagreed with its single run.
    """
    signals = signal_fn(price_data, **params)
    result = run_strategy(
        price_data,
        signals,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        fill_price=fill_price,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.pop("periods_per_year", None)
    result.pop("periods_per_year_source", None)
    result.update(params)
    return result


def _distinct_in_order(values: List[Any]) -> List[Any]:
    """One entry per distinct value, first occurrence first.

    Hashable values go through a dict, which also collapses 10 and 10.0 --
    the same parameter to every strategy here. An unhashable value falls
    back to an equality scan.
    """
    try:
        return list(dict.fromkeys(values))
    except TypeError:
        kept: List[Any] = []
        for value in values:
            if not any(value is k or value == k for k in kept):
                kept.append(value)
        return kept


def _distinct_axes(
    param_grid: Dict[str, List],
) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Each axis with its repeated values removed, and how many each lost.

    A grid is the product of its axes, so {"fast": [10, 10, 20], "slow":
    [50, 50]} ran six backtests for two distinct combinations, and the row
    count is what a caller hands deflated_sharpe_ratio as n_trials: tripled
    trials and a dispersion shrunk by identical rows nearly doubled the
    selection-bias benchmark a real result is judged against.
    """
    axes = {key: _distinct_in_order(list(values)) for key, values in param_grid.items()}
    dropped = {
        key: len(param_grid[key]) - len(axes[key])
        for key in axes
        if len(param_grid[key]) != len(axes[key])
    }
    return axes, dropped


def _finish_grid(
    df_out: pd.DataFrame,
    sort_by: str,
    ascending: bool,
    periods_per_year: int,
    ppy_source: str,
    ppy_warnings: List[str],
    dropped: Dict[str, int],
) -> pd.DataFrame:
    """
    Rank the grid and stamp what the ranking and the run depended on.

    Shared by the native and the Python path, so the two cannot rank the
    same grid differently -- they did once, over a Sharpe of 0.0 against
    NaN. `attrs` carries: n_unrankable, n_combinations (distinct
    combinations run), duplicate_values_dropped, periods_per_year,
    periods_per_year_source and warnings.
    """
    ranked = rank_rows(df_out, sort_by, ascending)
    warnings = list(ppy_warnings)
    if dropped:
        warnings.append(
            "param_grid repeated value(s) "
            + ", ".join(f"{k}: {n}" for k, n in dropped.items())
            + "; each distinct combination was run once, and n_combinations "
            "counts distinct combinations."
        )
    n_unrankable = int(ranked.attrs.get("n_unrankable", 0))
    if n_unrankable:
        note = unrankable_note(n_unrankable, len(ranked), sort_by, "combination")
        warnings.append(note)
        logger.warning("[backtest_grid] %s", note)
    ranked.attrs.update(
        {
            "n_unrankable": n_unrankable,
            "n_combinations": len(ranked),
            "duplicate_values_dropped": dict(dropped),
            "periods_per_year": periods_per_year,
            "periods_per_year_source": ppy_source,
            "warnings": warnings,
        }
    )
    return ranked


def backtest_grid(
    price_data: pd.DataFrame,
    strategy: Union[str, Callable[..., pd.Series]],
    param_grid: Dict[str, List],
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    sort_by: str = "sharpe_ratio",
    ascending: bool = False,
    n_workers: Optional[int] = None,
    fill_price: str = "close",
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
    interval: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a backtest across every parameter combination in param_grid in parallel.

    Args:
        price_data:     OHLCV DataFrame (from provider.get_ohlcv).
        strategy:       Either one of the built-in registry names
                        ('sma_crossover', 'rsi_mean_reversion', 'macd_crossover',
                        'bollinger_reversion', 'donchian_breakout',
                        'momentum_timeseries', 'vwap_reversion', 'adx_trend'
                        — see backtest.strategies.STRATEGY_REGISTRY), or your
                        own signal-generating callable with signature
                        `(price_data: pd.DataFrame, **params)
                        -> pd.Series` (values in {-1, 0, 1}). A custom callable
                        still gets the full C++ batch-kernel speedup when
                        `_sqt_core` is built — only the metric computation runs
                        in C++, so it has no idea whether the signal came from
                        a built-in strategy or your own model.
        param_grid:     Dict mapping parameter name → list of values.
                        e.g. {'fast_period': [5, 10, 20], 'slow_period': [30, 50]}
                        A value repeated on an axis is run once (10 and 10.0
                        are the same value), so the row count is the number
                        of distinct combinations.
        initial_capital: Starting capital for every backtest.
        commission_pct: Commission per trade side (fraction).
        slippage_pct:   Slippage per trade side (fraction).
        sort_by:        Output column to rank results by (default: 'sharpe_ratio').
                        Only rows whose value is finite AND that traded are
                        ranked; the rest (a combination that never traded,
                        or a ratio over an empty denominator) follow every
                        ranked row in grid order, and their count is
                        `attrs["n_unrankable"]`.
        ascending:      Sort direction (default: False = best first).
        n_workers:      Worker processes for the PYTHON grid loop, defaulting
                        to os.cpu_count(); pass 1 to run sequentially.
                        HAS NO EFFECT ON A NORMAL INSTALL. The C++ batch path
                        below runs the whole sweep in one call with no
                        subprocessing, and `_sqt_core` is built by
                        `pip install`, so the pool is reached only where the
                        extension is absent or failed. Measured: zero pools
                        created at any setting with the extension present,
                        and a real pool without it.
                        Also forced to 1 for a custom callable when the
                        extension is not built — arbitrary callables
                        (lambdas, closures) are frequently unpicklable across
                        the ProcessPoolExecutor spawn boundary.
        fill_price:     "close" (default), "next_open", or "hl2_exploratory" — see
                        run_strategy. All three take the C++ batch path: it
                        passes the Open or HL2 reference array alongside the
                        closes. The note that used to sit here, that the
                        latter two "force the Python path" because "the C++
                        batch kernel only knows Close prices", was left
                        behind by the change that added `grid_ref_arr`.
        periods_per_year, interval: resolved once for the whole grid, as
                        run_strategy resolves them (explicit, then the
                        interval, then the bar spacing, then 252 with a
                        warning), and applied to every path.

    Returns:
        pd.DataFrame with one row per distinct parameter combination, ranked
        by sort_by. Columns include all metric keys plus the parameter
        names. `attrs` carries n_unrankable, n_combinations,
        duplicate_values_dropped, periods_per_year, periods_per_year_source
        and warnings.

    Raises:
        ValidationError: price_data's index is unsorted or duplicated, or a
            cost, the capital, the rate or periods_per_year is out of range.

    Example (built-in strategy)::

        df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")
        results = backtest_grid(
            df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
        )
        print(results[["fast_period", "slow_period", "sharpe_ratio", "total_return"]].head())

    Example (your own signal, still grid-searched and C++-accelerated)::

        def my_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
            # any proprietary alpha logic — the grid searcher doesn't care
            edge = my_model.score(price_data)
            return (edge > threshold).astype(int)

        results = backtest_grid(
            df,
            strategy=my_signal,
            param_grid={"threshold": [0.1, 0.2, 0.3]},
        )
    """
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValidationError(
            f"initial_capital must be positive and finite, got {initial_capital!r}"
        )
    for name, value in (
        ("commission_pct", commission_pct),
        ("slippage_pct", slippage_pct),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValidationError(
                f"{name} must be non-negative and finite, got {value!r}"
            )
    require_finite_scalar(risk_free_rate, "risk_free_rate", "backtest_grid")
    # Before either path, and before the bars are read for anything: the
    # native path used to backtest a reversed frame (signals computed on the
    # reversed series) and a duplicated bar without a word, where run_strategy
    # raised on the same input.
    require_sorted_unique_index(price_data.index, "price_data", "backtest_grid")
    ppy, ppy_source, ppy_warnings = resolve_periods_per_year(
        price_data.index,
        periods_per_year=periods_per_year,
        interval=interval,
        where="backtest_grid",
    )

    is_custom = callable(strategy)
    if is_custom:
        raw_signal_fn: Callable[..., pd.Series] = strategy  # type: ignore[assignment]
        strategy_label = getattr(strategy, "__name__", "custom_strategy")

        # A registry strategy emits {-1, 0, 1} by construction. A caller's
        # callable emits whatever it emits, and a value of 2.0 ran a levered
        # book through every combination with nothing saying so. Checked
        # at the one point every path -- fused, batch and sequential --
        # reads the callable.
        def signal_fn(price_data_, **params):
            return _checked_custom_signal(
                raw_signal_fn(price_data_, **params), strategy_label
            )

    else:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        signal_fn = STRATEGY_REGISTRY[strategy]
        strategy_label = strategy

    # Build every DISTINCT parameter combination -- see _distinct_axes.
    axes, dropped = _distinct_axes(param_grid)
    keys = list(axes.keys())
    combos = list(product(*[axes[k] for k in keys]))

    t0 = time.perf_counter()

    # ── C++ batch path ────────────────────────────────────────────────────────
    # Generate all signal arrays in Python, then ship the entire batch to C++
    # in a single call — no subprocess overhead, no per-combo boundary crossing.
    # NOT scoped to fill_price="close": `grid_ref_arr` below carries the Open
    # or HL2 series for the other two, so every fill price takes this path.
    if HAS_CPP and _cpp_core is not None:
        # Checked before the try/except below -- that except catches
        # Exception broadly (to fall back to the Python grid loop on any
        # C++ failure), which would otherwise silently swallow a
        # ValidationError and mask bad input behind a confusing fallback
        # instead of rejecting it.
        prices_arr = price_data["Close"].to_numpy(dtype=np.float64)
        # STRICTLY POSITIVE, matching run_strategy. This path had only the
        # finiteness check, so the positive-price contract added for the
        # single-call path did not cover the grid: a Close of -5.0 ran through
        # an entire parameter sweep and returned a full results table, because
        # -5.0 is perfectly finite. Found by auditing the fixes themselves for
        # parallel paths rather than by a new symptom.
        require_positive_price_series(
            price_data["Close"], "Close", "batch_run_strategy", allow_nan=False
        )
        require_finite_array(prices_arr, "prices", "batch_run_strategy")
        try:
            # Reference (fill) prices, resolved BEFORE either branch below
            # uses them. They were originally computed just above the batch
            # call, which put them AFTER the fused branch that reads them --
            # a NameError that the broad `except Exception` below caught and
            # turned into a silent fallback to the Python grid loop. The
            # fused path simply never ran, and nothing said so.
            grid_ref_arr = None
            if fill_price == "next_open":
                grid_ref_arr = price_data["Open"].to_numpy(dtype=np.float64)
            elif fill_price == "hl2_exploratory":
                grid_ref_arr = (
                    (price_data["High"] + price_data["Low"]) / 2.0
                ).to_numpy(dtype=np.float64)

            logger.debug(
                "[backtest_grid] strategy=%s  combos=%d  path=C++  sort_by=%s",
                strategy_label,
                len(combos),
                sort_by,
            )

            # Fused path first: for a plain fast/slow crossover grid this
            # skips signal generation and the (num_combos x n_bars) matrix
            # entirely -- see _fused_crossover_metrics for the profile that
            # motivated it.
            metrics_arr = None
            if not is_custom and strategy_label == "sma_crossover":
                metrics_arr = _fused_crossover_metrics(
                    price_data,
                    keys,
                    combos,
                    initial_capital,
                    commission_pct,
                    slippage_pct,
                    grid_ref_arr,
                    risk_free_rate,
                    ppy,
                )

            if metrics_arr is None:
                sig_rows = []
                for combo in combos:
                    params = dict(zip(keys, combo))
                    sig_rows.append(
                        signal_fn(price_data, **params).to_numpy(dtype=np.float64)
                    )
                signals_mat = np.ascontiguousarray(
                    np.vstack(sig_rows), dtype=np.float64
                )  # shape: (num_combos, n_bars)
                require_finite_array(signals_mat, "signals", "batch_run_strategy")

                # win_rate/profit_factor/num_trades/avg_trade_return_pct here come
                # straight from the native kernel's own trade-log logic, same as
                # run_strategy's single-call C++ path above -- unlike that path,
                # nothing overwrites them per-combo here (rebuilding a Python-side
                # trade log for every parameter combination in the grid would
                # defeat the point of the batch C++ path's speed). This used to
                # be a real gap (native entry price one bar off, no commission/
                # slippage in trade returns), but backtest.cpp's run_strategy
                # (which batch_run_strategy calls per test) now uses the same
                # fill-aware, cost-aware accounting as _build_trade_log directly,
                # so these native stats should already agree with the Python
                # recomputation without an override -- see
                # tests/test_backtest.py's TestNativeTradeStatsCorrectness for
                # the gated equivalence check.
                # A flat (num_tests, 11) NumPy array instead of a Python list of
                # dicts -- for a large grid (thousands of combos), building that
                # many Python dict objects just to immediately feed them into
                # pd.DataFrame(rows) was itself real, avoidable overhead. Column
                # order is a fixed contract with bindings.cpp -- see
                # _BATCH_METRIC_COLUMNS above.
                metrics_arr = _cpp_core.batch_run_strategy(
                    prices_arr,
                    signals_mat,
                    initial_capital,
                    commission_pct,
                    slippage_pct,
                    float(ppy),
                    grid_ref_arr,
                    risk_free_rate,
                )
            metrics_df = pd.DataFrame(metrics_arr, columns=_BATCH_METRIC_COLUMNS)
            metrics_df["num_trades"] = metrics_df["num_trades"].astype(int)
            # The same boundary rule run_strategy's native branch applies.
            (
                metrics_df["sortino_ratio"],
                metrics_df["calmar_ratio"],
                metrics_df["profit_factor"],
            ) = _undefined_ratios_as_nan(
                metrics_df["sortino_ratio"],
                metrics_df["calmar_ratio"],
                metrics_df["profit_factor"],
                metrics_df["total_return"],
                metrics_df["annualized_volatility"],
                metrics_df["num_trades"],
                risk_free_rate,
            )
            metrics_df["final_equity"] = metrics_df["final_equity"].round(2)
            metrics_df["total_return"] = metrics_df["total_return"].round(6)
            metrics_df["annualized_volatility"] = metrics_df[
                "annualized_volatility"
            ].round(6)
            metrics_df["sharpe_ratio"] = metrics_df["sharpe_ratio"].round(4)
            metrics_df["sortino_ratio"] = metrics_df["sortino_ratio"].round(4)
            metrics_df["max_drawdown"] = metrics_df["max_drawdown"].round(6)
            metrics_df["calmar_ratio"] = metrics_df["calmar_ratio"].round(4)
            metrics_df["win_rate"] = metrics_df["win_rate"].round(4)
            metrics_df["profit_factor"] = metrics_df["profit_factor"].round(4)
            metrics_df["avg_trade_return_pct"] = metrics_df[
                "avg_trade_return_pct"
            ].round(4)

            params_df = pd.DataFrame(combos, columns=keys)
            df_out = pd.concat([metrics_df, params_df.reset_index(drop=True)], axis=1)
            df_out = _finish_grid(
                df_out, sort_by, ascending, ppy, ppy_source, ppy_warnings, dropped
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if not df_out.empty and sort_by in df_out.columns:
                best = df_out.iloc[0]
                best_params = {k: best[k] for k in keys if k in best}
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  best %s=%.4f  params=%s",
                    elapsed_ms,
                    sort_by,
                    best[sort_by],
                    best_params,
                )
            else:
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  %d results",
                    elapsed_ms,
                    len(df_out),
                )
            return df_out

        except ValidationError:
            # Bad input (e.g. NaN/Inf in a generated signal array) is a
            # real problem to surface, not something to silently retry
            # via the Python fallback below.
            raise
        except Exception as exc:
            logger.warning(
                "[backtest_grid] C++ batch path failed (%s) — falling back to Python",
                exc,
            )

    # ── Python fallback ───────────────────────────────────────────────────────
    if is_custom:
        # A raw callable may not be picklable (lambda, closure, notebook-defined
        # function) — always run sequentially in this process rather than risk
        # an opaque PicklingError from a ProcessPoolExecutor worker. The C++
        # batch path above (when built) already handles custom callables at
        # full speed with no subprocessing involved, so this only gives up
        # parallelism in the one uncommon case: no C++ extension AND >1 workers
        # requested AND a custom strategy.
        if n_workers is not None and n_workers != 1:
            logger.debug(
                "[backtest_grid] custom strategy without C++ extension — "
                "forcing sequential execution (n_workers=%s ignored)",
                n_workers,
            )
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=1 (forced)  path=Python  sort_by=%s",
            strategy_label,
            len(combos),
            sort_by,
        )
        results = [
            _run_signal_fn_job(
                price_data,
                signal_fn,
                dict(zip(keys, combo)),
                initial_capital,
                commission_pct,
                slippage_pct,
                fill_price=fill_price,
                risk_free_rate=risk_free_rate,
                periods_per_year=ppy,
            )
            for combo in combos
        ]
    else:
        jobs = [
            {
                "price_data": price_data,
                "strategy": strategy,
                "params": dict(zip(keys, combo)),
                "initial_capital": initial_capital,
                "commission_pct": commission_pct,
                "slippage_pct": slippage_pct,
                "fill_price": fill_price,
                "risk_free_rate": risk_free_rate,
                "periods_per_year": ppy,
            }
            for combo in combos
        ]
        workers = n_workers if n_workers is not None else (os.cpu_count() or 4)
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=%d  path=Python  sort_by=%s",
            strategy_label,
            len(jobs),
            workers,
            sort_by,
        )

        if workers == 1 or len(jobs) == 1:
            results = [_run_grid_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run_grid_job, jobs))

    df_out = _finish_grid(
        pd.DataFrame(results),
        sort_by,
        ascending,
        ppy,
        ppy_source,
        ppy_warnings,
        dropped,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not df_out.empty and sort_by in df_out.columns:
        best = df_out.iloc[0]
        best_params = {k: best[k] for k in keys if k in best}
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  best %s=%.4f  params=%s",
            elapsed_ms,
            sort_by,
            best[sort_by],
            best_params,
        )
    else:
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  %d results", elapsed_ms, len(df_out)
        )

    return df_out
