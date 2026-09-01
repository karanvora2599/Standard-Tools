"""
True portfolio simulation engine: one shared cash balance, position sizing
relative to current account equity, and rebalancing at specific dates —
the piece run_signal_panel_backtest (panel.py) cannot do, since it gives
every ticker its own independent initial_capital and only blends the
resulting per-ticker return streams afterward.

Weights drift between rebalance dates instead of being re-applied every
bar: share counts stay constant, but equity still moves as prices move,
exactly like a real account. That drift is the entire reason this engine
exists alongside the vectorized per-ticker one.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.backtest.constraints import adv_participation
from standard_quant_tools.backtest.costs import (
    _cost_rate,
    impact_cost,
    margin_interest,
    per_share_commission,
    percentage_commission,
    short_borrow_cost,
)
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_VALID_FILL_PRICES = ("close", "next_open", "hl2_exploratory")
_VALID_COMMISSION_MODELS = ("pct", "per_share")


_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass

# Kernel status codes -> the message this engine has always raised. Kept in
# Python so the wording, and the fact that it names a date and a ticker, does
# not have to be duplicated in C++.
_PORTFOLIO_OK = 0
_PORTFOLIO_BAD_EXEC_PRICE = 1
_PORTFOLIO_INSOLVENT_AT_REBALANCE = 2
_PORTFOLIO_LEVERAGE_BREACH = 3
_PORTFOLIO_POSITION_BREACH = 4
_PORTFOLIO_INSOLVENT_AT_BAR = 5
_PORTFOLIO_BAD_DOLLAR_VOLUME = 6
_PORTFOLIO_ADV_BREACH = 7
_PORTFOLIO_BAD_VOLATILITY = 8

#: commission_model -> the kernel's enum.
_COMMISSION_CODES = {"pct": 0, "per_share": 1}

_FILL_CODES = {"close": 0, "next_open": 1, "hl2_exploratory": 2}


def _native_portfolio_sim(
    *,
    close_mat: np.ndarray,
    open_mat: Optional[np.ndarray],
    hl2_mat: Optional[np.ndarray],
    weights_mat: np.ndarray,
    rebalance_index: pd.Index,
    master_index: pd.Index,
    tickers: List[str],
    fill_price: str,
    commission_model: str,
    use_impact_model: bool,
    max_adv_participation: Optional[float],
    initial_capital: float,
    commission_pct: float,
    sell_commission_rate: float,
    slippage_pct: float,
    max_gross_leverage: float,
    max_position_pct: float,
    borrow_fee_bps: float,
    margin_interest_rate: float,
    per_share_rate: float = 0.0,
    min_commission: float = 0.0,
    impact_coefficient: float = 1.0,
    dollar_volume_mat: Optional[np.ndarray] = None,
    volatility_mat: Optional[np.ndarray] = None,
) -> Optional[
    Tuple[
        List[float],
        List[float],
        List[float],
        List[float],
        List[Dict[str, Any]],
        float,
    ]
]:
    """Run the bar loop natively, or return None if this configuration is not covered.

    Returning None rather than raising is the whole point: an unsupported
    option is not an error, it just means the Python loop runs instead.
    """
    if not (HAS_CPP and _cpp_core is not None):
        return None
    if fill_price not in _FILL_CODES:
        return None
    if commission_model not in _COMMISSION_CODES:
        return None
    # per_share commission, the impact model and the ADV cap USED to be
    # refused here, on the reasoning that each is a per-element decision
    # that would have to be restated in C++. They are per-element in the
    # kernel too -- its rebalance loop was already per-ticker with a
    # per-ticker error path -- so what they needed was arguments rather
    # than a different shape. Measured before: the kernel bought 1.3x on
    # the one configuration it covered while every configuration it
    # refused ran 6-21x slower with no acceleration at all.
    needs_volume = use_impact_model or (max_adv_participation is not None)
    if needs_volume and dollar_volume_mat is None:
        return None
    if use_impact_model and volatility_mat is None:
        return None

    n_bars = len(master_index)
    if n_bars == 0:
        return None

    if fill_price == "next_open":
        exec_mat = open_mat
    elif fill_price == "hl2_exploratory":
        exec_mat = hl2_mat
    else:
        exec_mat = close_mat
    if exec_mat is None:
        return None

    # Bar index each weights row triggers at. The upfront validation already
    # rejected a rebalance date that is not on the master calendar, so
    # get_indexer cannot return -1 here.
    rebal_bars = master_index.get_indexer(rebalance_index).astype(np.int64)
    if (rebal_bars < 0).any():
        return None

    # Calendar days since the previous bar, so a Friday->Monday gap accrues
    # three days of financing rather than one.
    if borrow_fee_bps > 0.0 or margin_interest_rate > 0.0:
        deltas = np.diff(master_index.to_numpy()).astype("timedelta64[D]")
        day_gaps = np.empty(n_bars, dtype=np.float64)
        day_gaps[0] = 1.0
        day_gaps[1:] = deltas.astype(np.float64)
    else:
        day_gaps = np.ones(n_bars, dtype=np.float64)

    res = _cpp_core.run_portfolio_simulation(
        np.ascontiguousarray(close_mat, dtype=np.float64),
        np.ascontiguousarray(exec_mat, dtype=np.float64),
        np.ascontiguousarray(weights_mat, dtype=np.float64),
        rebal_bars,
        day_gaps,
        initial_capital,
        commission_pct,
        sell_commission_rate,
        slippage_pct,
        max_gross_leverage,
        max_position_pct,
        borrow_fee_bps,
        margin_interest_rate,
        _FILL_CODES[fill_price],
        (
            np.ascontiguousarray(dollar_volume_mat, dtype=np.float64)
            if needs_volume
            else None
        ),
        (
            np.ascontiguousarray(volatility_mat, dtype=np.float64)
            if use_impact_model
            else None
        ),
        _COMMISSION_CODES[commission_model],
        per_share_rate,
        min_commission,
        use_impact_model,
        impact_coefficient,
        # The kernel reads <= 0 as "no cap"; None is how Python spells it.
        0.0 if max_adv_participation is None else float(max_adv_participation),
    )

    status = int(res["status"])
    if status != _PORTFOLIO_OK:
        _raise_portfolio_error(
            status,
            res,
            master_index,
            tickers,
            max_gross_leverage,
            max_position_pct,
            max_adv_participation,
        )

    n_exec = int(res["n_executed"])
    reb = res["rebalances"]
    # Which weight rows actually executed, in order. Under next_open a row
    # triggering on the final bar never executes -- there is no following
    # Open to fill at -- which is exactly what the Python loop does.
    executed_rows = [
        row
        for row, bar in enumerate(rebal_bars)
        if fill_price != "next_open" or bar + 1 < n_bars
    ][:n_exec]

    rebalance_log: List[Dict[str, Any]] = []
    for i, row in enumerate(executed_rows):
        trigger_date = rebalance_index[row]
        rebalance_log.append(
            {
                "date": (
                    str(trigger_date.date())
                    if hasattr(trigger_date, "date")
                    else str(trigger_date)
                ),
                "turnover_pct": round(float(reb[i, 0]), 6),
                "gross_leverage_after": round(float(reb[i, 1]), 6),
                "n_positions": int(reb[i, 2]),
            }
        )

    return (
        res["equity"].tolist(),
        res["cash"].tolist(),
        res["gross"].tolist(),
        res["net"].tolist(),
        rebalance_log,
        float(res["peak_position"]),
    )


def _raise_portfolio_error(
    status: int,
    res: Dict[str, Any],
    master_index: pd.Index,
    tickers: List[str],
    max_gross_leverage: float,
    max_position_pct: float,
    max_adv_participation: Optional[float] = None,
) -> None:
    """Re-raise a kernel status as the message the Python loop would have raised."""
    bar = int(res["bar"])
    value = float(res["value"])
    date = master_index[bar] if bar < len(master_index) else bar

    if status == _PORTFOLIO_BAD_EXEC_PRICE:
        ticker = tickers[int(res["ticker"])]
        raise ValidationError(
            f"rebalance {date} ticker {ticker!r}: execution price "
            f"{value!r} is nonpositive or non-finite — cannot size "
            "a nonzero target weight from it."
        )
    if status == _PORTFOLIO_INSOLVENT_AT_REBALANCE:
        raise ValidationError(
            f"rebalance {date}: account equity is {value!r} "
            "(zero or negative) after this rebalance's costs — insolvent; "
            "this engine does not model forced liquidation/margin calls, "
            "so the simulation cannot continue meaningfully."
        )
    if status == _PORTFOLIO_LEVERAGE_BREACH:
        raise ValidationError(
            f"rebalance {date}: realized gross leverage "
            f"{value:.4f} exceeds max_gross_leverage="
            f"{max_gross_leverage} (shares sized from this rebalance's "
            "equity do not match the requested weights)"
        )
    if status == _PORTFOLIO_POSITION_BREACH:
        raise ValidationError(
            f"rebalance {date}: realized position size "
            f"{value:.4f} exceeds max_position_pct={max_position_pct}"
        )
    if status == _PORTFOLIO_BAD_DOLLAR_VOLUME:
        ticker = tickers[int(res["ticker"])]
        raise ValidationError(
            f"rebalance {date} ticker {ticker!r}: average dollar volume is "
            f"{value!r} (missing or invalid) — max_adv_participation/use_impact_model "
            "require a valid 'Volume' baseline for every ticker actually traded."
        )

    if status == _PORTFOLIO_ADV_BREACH:
        ticker = tickers[int(res["ticker"])]
        raise ValidationError(
            f"rebalance {date} ticker {ticker!r}: ADV participation "
            f"{value:.4f} exceeds max_adv_participation={max_adv_participation}"
        )

    if status == _PORTFOLIO_BAD_VOLATILITY:
        ticker = tickers[int(res["ticker"])]
        raise ValidationError(
            f"rebalance {date} ticker {ticker!r}: impact-model volatility is "
            f"{value!r}. use_impact_model needs a finite non-negative "
            "volatility for every ticker actually traded; a rebalance inside "
            "the impact_lookback warm-up window has none yet."
        )

    if status == _PORTFOLIO_INSOLVENT_AT_BAR:
        raise ValidationError(
            f"{date}: account equity is {value!r} (zero or negative) — "
            "insolvent; this engine does not model forced liquidation/"
            "margin calls, so the simulation cannot continue meaningfully."
        )
    raise ValidationError(f"portfolio simulation failed with status {status}")


def run_portfolio_simulation(
    price_data: Dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    sell_commission_pct: Optional[float] = None,
    slippage_pct: float = 0.0005,
    max_gross_leverage: float = 1.0,
    max_position_pct: float = 1.0,
    fill_price: str = "close",
    commission_model: str = "pct",
    per_share_rate: float = 0.0,
    min_commission: float = 0.0,
    use_impact_model: bool = False,
    impact_coefficient: float = 1.0,
    impact_lookback: int = 20,
    borrow_fee_bps: float = 0.0,
    margin_interest_rate: float = 0.0,
    max_adv_participation: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Simulate a single shared-cash portfolio account rebalanced at the dates
    in target_weights.index.

    Args:
        price_data: Dict mapping ticker -> OHLCV DataFrame (must contain 'Close',
            plus 'Open' if fill_price="next_open", 'High'/'Low' if
            fill_price="hl2_exploratory", or 'Volume' if use_impact_model or
            max_adv_participation is set).
        target_weights: DataFrame indexed by rebalance date, one column per
            ticker in the universe, values are the target fraction of
            account equity (negative for short). Must be dense — every
            ticker must have a value at every rebalance date, mirroring
            run_signal_panel_backtest's existing "must have an entry for
            every ticker" contract.
        initial_capital: Starting cash.
        commission_pct: Commission per trade notional (fraction) when
            commission_model="pct" (default — today's existing behavior).
            Charged on BOTH sides unless sell_commission_pct overrides the
            sell side.
        sell_commission_pct: Optional separate commission rate for SALES.
            None (the default) charges commission_pct both ways, which is
            what this function did before this parameter existed. Set it
            when the two sides genuinely differ — the SEC Section 31 fee and
            the FINRA TAF are levied on sales only, so a round trip is not
            symmetric and a sell-heavy strategy is undercharged by one
            blended rate. Applies to commission only: the spread is crossed
            whichever way you go, so slippage_pct stays symmetric.
        slippage_pct: Spread cost per trade notional (fraction), applied
            regardless of commission_model.
        max_gross_leverage: Reject (raise ValidationError) any rebalance date
            whose sum(|weight|) exceeds this (default 1.0 = fully invested,
            no leverage). Bounds the TARGET weights / sizing basis — see
            "Post-trade enforcement" below for what this does and does not
            guarantee about realized, post-cost leverage.
        max_position_pct: Reject any single position whose |weight| exceeds
            this (default 1.0). Same target-weight/sizing-basis scope as
            max_gross_leverage above.
        fill_price: "close" (default) — a rebalance dated D executes at D's
            own Close, the same bar the target weight is "known" on (this is
            the pre-existing behavior, unchanged). "next_open" — the
            rebalance instead executes at the following bar's Open (one-bar
            delay, mirroring run_strategy's lookahead-free convention);
            raises ValidationError if the last rebalance date has no
            following bar to fill against. "hl2_exploratory" — executes on
            the same bar as "close" does, but at that bar's own (High+Low)/2
            ("HL2") instead of Close. This is NOT a bid/ask midpoint quote —
            it requires knowing the bar's High and Low, only determined once
            the bar has already completed, so it is look-ahead the same way
            "close" is (see the emitted warning); intended for exploratory
            analysis only, never for evaluating real strategy performance.
            Equity is always marked to Close regardless of fill_price — only
            the rebalance trade's own execution price changes.
        commission_model: "pct" (default) — commission_pct * trade notional.
            "per_share" — per_share_rate per share traded, floored at
            min_commission (backtest/costs.py: per_share_commission).
        use_impact_model: If True, adds a square-root market-impact cost
            (backtest/costs.py: impact_cost) on top of commission + spread,
            using a rolling impact_lookback-bar average dollar volume
            (Close * Volume) and return volatility per ticker. Requires a
            'Volume' column. Default False reproduces today's exact cost
            behavior.
        impact_coefficient, impact_lookback: Parameters of the impact model
            (ignored when use_impact_model=False).
        borrow_fee_bps: Annualized basis-point borrow fee accrued daily on
            any short position's notional (0.0 = no borrow cost, today's
            existing behavior).
        margin_interest_rate: Annualized rate accrued daily on negative cash
            (implied margin borrowing); 0.0 = no financing cost charged
            beyond the existing "cash went negative" warning.
        max_adv_participation: Reject (raise ValidationError) any rebalance
            trade whose notional exceeds this fraction of the ticker's own
            rolling average dollar volume. None (default) = no ADV
            constraint checked. Requires a 'Volume' column when set. When
            either this or use_impact_model is set, a missing/zero/non-finite
            volume baseline for a ticker being traded also raises
            ValidationError — fails closed (can't estimate liquidity means
            the trade is rejected), not open (silently treated as
            unconstrained).

    Post-trade enforcement, and what max_gross_leverage/max_position_pct
    actually bound: target_shares for a rebalance are sized from equity_now
    (the account's equity immediately BEFORE that rebalance's own costs are
    deducted), so gross_after (the dollar value of the resulting shares) is
    a fixed fraction — exactly sum(|weight|) — of equity_now by
    construction. After costs are deducted, realized post-cost leverage
    (gross_after / equity_after, the same ratio reported as
    gross_leverage_after in rebalance_log and in leverage_curve) is
    mechanically inflated above sum(|weight|) whenever this rebalance's own
    costs are nonzero — equity_after < equity_now while gross_after is
    unchanged. This is expected, unavoidable cost drag, not a limit
    violation, and is NOT rejected: max_gross_leverage/max_position_pct
    bound the TARGET weights (equivalently, gross_after relative to
    equity_now, the actual sizing basis) — they are not a hard cap on the
    post-cost ratio you'll see reported. What IS re-checked and CAN raise
    ValidationError is gross_after/equity_now (and the largest single
    position's weight/equity_now) exceeding the limit — a sizing
    self-consistency invariant that should be structurally guaranteed by
    the per-date weight validation above, so a violation here indicates an
    actual sizing bug, not ordinary cost drag. If you need a hard ceiling
    on realized, cost-inclusive leverage, monitor leverage_curve/
    rebalance_log's gross_leverage_after yourself, or reserve headroom by
    requesting a lower max_gross_leverage than your true risk limit (the
    unavoidable inflation from cost drag is bounded by
    max_gross_leverage / (1 - cost_fraction_of_equity), so it's small
    whenever per-rebalance costs are a small fraction of equity — as they
    are for typical bps-level commission/slippage).

    Zero-size trades (a ticker whose target share count is unchanged from
    the prior rebalance) are skipped entirely — no cost, no turnover, no
    ADV check — so a per_share_commission minimum floor can't charge a
    ticker that didn't actually trade.

    Returns:
        Dict with equity_curve, cash_curve, gross_exposure_curve,
        net_exposure_curve, leverage_curve (all pd.Series on the master
        trading-calendar index), rebalance_log (pd.DataFrame: date,
        turnover_pct, gross_leverage_after, n_positions), final_equity,
        final_cash, warnings (list[str]).
    """
    if fill_price not in _VALID_FILL_PRICES:
        raise ValidationError(
            f"fill_price must be one of {_VALID_FILL_PRICES}, got {fill_price!r}"
        )
    if commission_model not in _VALID_COMMISSION_MODELS:
        raise ValidationError(
            f"commission_model must be one of {_VALID_COMMISSION_MODELS}, got {commission_model!r}"
        )

    if not math.isfinite(initial_capital) or initial_capital <= 0:
        raise ValidationError(
            f"initial_capital must be a positive finite number, got {initial_capital!r}"
        )

    # Validated through costs.py's own _cost_rate so the engine and the
    # cost primitives agree on what a valid rate IS, and so every rate is
    # checked once here rather than re-checked per trade inside the cost
    # functions. This is strictly stricter than the hand-rolled
    # `math.isfinite(value) or value < 0` loop it replaces: that accepted
    # `True` (isfinite(True) is True and True < 0 is False, so a boolean
    # became a 100% commission rate) and raised a bare TypeError rather
    # than a ValidationError on a string. It also checks EVERY rate,
    # including the ones the currently-selected commission model does not
    # read -- a typo in per_share_rate should not go unmentioned just
    # because this run happens to use the pct model.
    _cost_params = {
        "commission_pct": commission_pct,
        "slippage_pct": slippage_pct,
        "per_share_rate": per_share_rate,
        "min_commission": min_commission,
        "borrow_fee_bps": borrow_fee_bps,
        "margin_interest_rate": margin_interest_rate,
        "impact_coefficient": impact_coefficient,
    }
    if sell_commission_pct is not None:
        # Added conditionally: None means "not supplied", not "a rate of
        # None", and the loop below would reject it as a non-number.
        _cost_params["sell_commission_pct"] = sell_commission_pct
    for name, value in _cost_params.items():
        _cost_rate(name, value)

    # Resolved to a concrete rate once, here, so the scalar loop, the
    # vectorized branch and the native kernel all read the same number
    # rather than each re-deciding what None means.
    sell_commission_rate = (
        commission_pct if sell_commission_pct is None else sell_commission_pct
    )
    if impact_lookback <= 0:
        raise ValidationError(f"impact_lookback must be > 0, got {impact_lookback}")

    if not math.isfinite(max_gross_leverage) or max_gross_leverage <= 0:
        raise ValidationError(
            f"max_gross_leverage must be a positive finite number, got {max_gross_leverage!r}"
        )
    if not math.isfinite(max_position_pct) or max_position_pct <= 0:
        raise ValidationError(
            f"max_position_pct must be a positive finite number, got {max_position_pct!r}"
        )
    if max_adv_participation is not None and (
        not math.isfinite(max_adv_participation) or max_adv_participation <= 0
    ):
        raise ValidationError(
            f"max_adv_participation must be a positive finite number, got {max_adv_participation!r}"
        )

    tickers = list(target_weights.columns)
    if not tickers:
        raise ValidationError("target_weights has no ticker columns — empty universe")
    if target_weights.empty:
        raise ValidationError(
            "target_weights has no rebalance dates — nothing to simulate"
        )
    if target_weights.index.has_duplicates:
        dupes = (
            target_weights.index[target_weights.index.duplicated()].unique().tolist()
        )
        raise ValidationError(
            "target_weights.index has duplicate rebalance date(s): "
            f"{[str(d) for d in dupes[:5]]}"
        )
    if not target_weights.index.is_monotonic_increasing:
        raise ValidationError("target_weights.index must be sorted in increasing order")
    if np.isinf(target_weights.to_numpy(dtype=float)).any():
        raise ValidationError(
            "target_weights contains an infinite value — every weight must be finite"
        )

    missing = [t for t in tickers if t not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

    # target_weights.index is already checked for duplicates above; price_data
    # was not. A duplicated bar makes .loc[date, "Close"] return a Series
    # instead of a scalar, so float() raises a bare TypeError from deep inside
    # the per-bar loop with no indication of which ticker or date caused it.
    for t in tickers:
        if price_data[t].index.has_duplicates:
            dupes = (
                price_data[t].index[price_data[t].index.duplicated()].unique().tolist()
            )
            raise ValidationError(
                f"price_data[{t!r}].index has duplicate date(s): "
                f"{[str(d) for d in dupes[:5]]} — every bar must be unique."
            )

    required_cols = {
        "close": ["Close"],
        "next_open": ["Close", "Open"],
        "hl2_exploratory": ["Close", "High", "Low"],
    }
    needs_volume = use_impact_model or (max_adv_participation is not None)
    for t in tickers:
        missing_cols = [
            c for c in required_cols[fill_price] if c not in price_data[t].columns
        ]
        if needs_volume and "Volume" not in price_data[t].columns:
            missing_cols.append("Volume")
        if missing_cols:
            raise ValidationError(
                f"price_data[{t!r}] is missing column(s) {missing_cols} required for "
                f"fill_price={fill_price!r}"
                + (", use_impact_model/max_adv_participation" if needs_volume else "")
            )

    # Master trading calendar: intersection of every ticker's own index, so
    # every bar has a valid price for every ticker — no NaN price lookups,
    # no silent stale-price trading on a day one ticker didn't trade.
    master_index = price_data[tickers[0]].index
    for t in tickers[1:]:
        master_index = master_index.intersection(price_data[t].index)
    master_index = master_index.sort_values()
    if master_index.empty:
        raise ValidationError(
            "price_data tickers share no common trading dates — empty master "
            "trading calendar, nothing to simulate"
        )

    # ── Dense price matrices, materialized ONCE ──────────────────────────
    # The per-bar loop below used to read every price through
    # `price_data[t].loc[date, "Close"]`, i.e. one pandas label lookup per
    # ticker per bar. Profiling 100 tickers x 2,000 bars showed 200,000 such
    # calls accounting for essentially the entire runtime, while the rebalance
    # state machine -- the part that actually does the simulating -- is only
    # 96 x 100 = 9,600 operations.
    #
    # So the bottleneck was never the accounting; it was addressing the data.
    # Each required column becomes one (n_bars x n_tickers) float64 matrix
    # aligned to master_index, and the loop indexes it positionally. The
    # arithmetic below is unchanged -- this moves where the numbers are READ
    # FROM, not how they are computed.
    #
    # Built HERE, before validation, because the validation below needs the
    # same master_index-aligned view of every price. Reindexing once and
    # validating the matrix does the alignment work a single time instead of
    # once for the check and again for the simulation.
    n_bars = len(master_index)
    n_tickers = len(tickers)
    ticker_pos = {t: i for i, t in enumerate(tickers)}

    def _build_matrices(columns: Sequence[str]) -> Dict[str, np.ndarray]:
        """One row alignment per TICKER, not per (ticker, column).

        `frame.loc[master_index, column]` takes pandas' 2-D tuple-key path and
        recomputes the same alignment for every column of the same ticker.
        Profiling 500 tickers x 504 bars put 92% of this function's runtime
        right here. Resolving the row positions once per ticker and taking
        them from each column's raw array does the same work once.

        master_index is the INTERSECTION of every ticker's index (built just
        above), so every label is present in every frame and `get_indexer`
        cannot return -1 -- which is what makes positional take safe here and
        would not be safe on a union calendar.
        """
        mats = {c: np.empty((n_bars, n_tickers), dtype=np.float64) for c in columns}
        # Checked once for the whole universe rather than per ticker: when
        # every frame is already on the master calendar -- one date range from
        # one provider, the common case -- there is no alignment to do at all,
        # and Index.equals is itself not free at 500+ tickers.
        aligned = all(price_data[t].index.equals(master_index) for t in ticker_pos)
        for t, i in ticker_pos.items():
            frame = price_data[t]
            if aligned:
                for c in columns:
                    mats[c][:, i] = frame[c].to_numpy(dtype=np.float64)
            else:
                take = frame.index.get_indexer(master_index)
                for c in columns:
                    mats[c][:, i] = frame[c].to_numpy(dtype=np.float64)[take]
        return mats

    price_matrices = _build_matrices(required_cols[fill_price])

    # Screen each column with one whole-matrix pass. Only when something is
    # actually wrong do we walk ticker-by-ticker, and that walk reproduces
    # the original ticker-major/column-inner search order so the offender
    # named in the message is the same one as before -- with several bad
    # tickers, WHICH one gets reported is observable.
    if any(
        not np.isfinite(mat).all() or (mat <= 0).any()
        for mat in price_matrices.values()
    ):
        for t in tickers:
            for col in required_cols[fill_price]:
                arr = price_matrices[col][:, ticker_pos[t]]
                bad = (~np.isfinite(arr)) | (arr <= 0)
                if bad.any():
                    raise ValidationError(
                        f"price_data[{t!r}][{col!r}] has nonpositive or non-finite "
                        f"value(s) on the trading calendar, e.g. at: "
                        f"{[str(d) for d in master_index[bad][:5]]}"
                    )

    missing_dates = [d for d in target_weights.index if d not in master_index]
    if missing_dates:
        raise ValidationError(
            "target_weights has rebalance date(s) with no price data for "
            f"every ticker in the universe: {[str(d) for d in missing_dates[:5]]}"
        )

    nan_mask = target_weights.isna()
    if nan_mask.to_numpy().any():
        bad = {
            str(date): [
                t for t in target_weights.columns if bool(nan_mask.loc[date, t])
            ]
            for date in target_weights.index
            if nan_mask.loc[date].any()
        }
        raise ValidationError(
            "target_weights contains NaN — every ticker must have a value at "
            f"every rebalance date (see: {dict(list(bad.items())[:5])})"
        )

    # ── Dense weight matrix, materialized ONCE ───────────────────────────
    # `tickers` is literally list(target_weights.columns), so the columns are
    # already in position order and no realignment is needed — the frame IS
    # the matrix. Both the validation below and every rebalance read rows of
    # it positionally, which removes the per-rebalance-date pandas row
    # extraction and Series construction (2,000 of each, and the single
    # largest remaining cost, on a daily-rebalanced backtest).
    weights_mat = target_weights.to_numpy(dtype=np.float64)
    abs_weights = np.abs(weights_mat)
    gross_by_date = abs_weights.sum(axis=1)

    # Screen every date at once, then reconstruct the message only for the
    # first offender. Checking gross before position size WITHIN that date
    # preserves which of the two errors a doubly-invalid row reports, and
    # taking the first offending row in index order preserves which date is
    # named when several are invalid — both observable in the raised message.
    offending = np.flatnonzero(
        (gross_by_date > max_gross_leverage + 1e-9)
        | (abs_weights > max_position_pct + 1e-9).any(axis=1)
    )
    if offending.size:
        i = int(offending[0])
        date = target_weights.index[i]
        gross = float(gross_by_date[i])
        if gross > max_gross_leverage + 1e-9:
            raise ValidationError(
                f"rebalance date {date}: gross leverage {gross:.4f} exceeds "
                f"max_gross_leverage={max_gross_leverage}"
            )
        row = target_weights.iloc[i]
        over = row[row.abs() > max_position_pct + 1e-9]
        raise ValidationError(
            f"rebalance date {date}: position(s) exceed "
            f"max_position_pct={max_position_pct}: {over.to_dict()}"
        )

    rebalance_dates = set(target_weights.index)
    # Rebalance date -> row of weights_mat, so the per-bar loop resolves a
    # rebalance with a dict lookup instead of target_weights.loc[date].
    rebalance_rows = {date: i for i, date in enumerate(target_weights.index)}

    if fill_price == "next_open" and rebalance_dates:
        last_rebalance = max(rebalance_dates)
        last_idx = master_index.get_loc(last_rebalance)
        if last_idx == len(master_index) - 1:
            raise ValidationError(
                "fill_price='next_open' requires a bar after the last rebalance "
                f"date ({last_rebalance}) to fill against, but it is the last "
                "bar in the master trading calendar."
            )

    # Rolling average dollar volume / volatility per ticker, needed only
    # for the impact model and/or the ADV participation check — computed
    # once upfront rather than per-trade.
    # Stored as (n_bars x n_tickers) matrices for the same reason the prices
    # are: the liquidity-aware modes consult them once per ticker per trade,
    # which is a pandas scalar label lookup on a daily-rebalanced backtest
    # for every one of those. Positional indexing by (bar, ticker position)
    # instead.
    dollar_volume_mat: Optional[np.ndarray] = None
    volatility_mat: Optional[np.ndarray] = None
    if needs_volume:
        dollar_volume_mat = np.empty((n_bars, n_tickers), dtype=np.float64)
        for t, i in ticker_pos.items():
            dv = (price_data[t]["Volume"] * price_data[t]["Close"]).reindex(
                master_index
            )
            dollar_volume_mat[:, i] = (
                dv.rolling(impact_lookback, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float64)
            )
    if use_impact_model:
        volatility_mat = np.empty((n_bars, n_tickers), dtype=np.float64)
        for t, i in ticker_pos.items():
            ret = price_data[t]["Close"].pct_change().reindex(master_index)
            volatility_mat[:, i] = (
                ret.rolling(impact_lookback, min_periods=1)
                .std()
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

    close_mat = price_matrices["Close"]
    open_mat = price_matrices["Open"] if fill_price == "next_open" else None
    hl2_mat = (
        (price_matrices["High"] + price_matrices["Low"]) / 2.0
        if fill_price == "hl2_exploratory"
        else None
    )

    cash = initial_capital
    # shares as a dense vector rather than a dict: the hot loop computes
    # dot products over it every bar, and a dict comprehension there was
    # rebuilding Python floats for work NumPy does in one call.
    shares_vec = np.zeros(n_tickers, dtype=np.float64)

    equity_records: List[float] = []
    cash_records: List[float] = []
    gross_records: List[float] = []
    net_records: List[float] = []
    rebalance_log: List[Dict[str, Any]] = []
    # Running peak of the largest single position, in currency. A scalar
    # rather than a curve: the peak is what a concentration limit is written
    # against, and returning another (n_bars,) series would cost every
    # consumer the payload for a number they reduce anyway.
    peak_position_value = 0.0

    def _valid_dollar_volume(
        t: str, pos: int, trigger_bar: int, exec_date: Any
    ) -> float:
        # Fail closed: a caller who explicitly enabled max_adv_participation
        # or use_impact_model asked for a liquidity-aware check — a missing
        # or invalid volume baseline means that check can't be performed,
        # which must reject the trade, not silently pass it as
        # unconstrained. costs.py's adv_participation/impact_cost stay
        # permissive (return 0.0) for callers who use them directly without
        # opting into this engine's safety feature.
        #
        # Look up at the TRIGGER bar, not the execution bar: for
        # fill_price="close"/"hl2_exploratory" the two are always equal (a
        # no-op here), but for "next_open" they differ -- the execution bar
        # is the one actually being filled, whose own full-day Volume/Close
        # isn't knowable yet at that bar's Open. The trigger bar is the last
        # bar that was fully complete when the rebalance decision was made,
        # so it's the correct, actually-known-in-advance baseline for a mode
        # whose entire documented purpose is being lookahead-free.
        dv = float(dollar_volume_mat[trigger_bar, pos])
        if not math.isfinite(dv) or dv <= 0:
            raise ValidationError(
                f"rebalance {exec_date} ticker {t!r}: average dollar volume is "
                f"{dv!r} (missing or invalid) — max_adv_participation/use_impact_model "
                "require a valid 'Volume' baseline for every ticker actually traded."
            )
        return dv

    def _trade_cost(
        t: str,
        pos: int,
        delta_shares: float,
        trade_notional: float,
        trigger_bar: int,
        exec_date: Any,
    ) -> float:
        if commission_model == "per_share":
            # Inlined for the reason the pct branch below was inlined, and
            # on the same evidence: `per_share_rate` and `min_commission`
            # are function parameters, validated once at entry through
            # `_cost_rate` with every other rate, and cannot change between
            # trades. `per_share_commission` re-checked both of them plus
            # the share count on every call.
            #
            # Same arithmetic -- max(abs(shares) * rate, minimum), with a
            # zero-share trade costing 0.0 rather than the per-ORDER floor
            # -- so a change to that formula must be mirrored here.
            commission = (
                0.0
                if delta_shares == 0.0
                else max(abs(delta_shares) * per_share_rate, min_commission)
            )
        else:
            # Inlined rather than calling percentage_commission(): its
            # validation is the expensive part, and the RATE was already
            # validated once per rebalance by the caller. Same arithmetic --
            # abs(notional) * rate -- so a change to that formula must be
            # mirrored here, which is why the formula is spelled out rather
            # than paraphrased.
            #
            # Deliberately dropped with it: percentage_commission's check
            # that the NOTIONAL is a finite number. Unlike the rate, that is
            # not caller input -- it is abs(delta) * price, built from a
            # price the upfront validation already proved finite and
            # positive on every bar of the master calendar, and a delta
            # derived from an equity the insolvency check already proved
            # positive. There is no path by which it arrives non-finite.
            commission = abs(trade_notional) * (
                sell_commission_rate if delta_shares < 0.0 else commission_pct
            )
        spread = abs(trade_notional) * slippage_pct
        impact = 0.0
        if use_impact_model:
            adv = _valid_dollar_volume(t, pos, trigger_bar, exec_date)
            # Same trigger-bar-not-exec-bar lookup as _valid_dollar_volume,
            # and for the same reason.
            vol = float(volatility_mat[trigger_bar, pos])
            # The ONE value here that varies per element, so the one that
            # still needs checking. `impact_lookback` uses min_periods=1, so
            # the first bar's rolling std is NaN, and a NaN volatility must
            # refuse rather than propagate into a plausible-looking cost.
            if not math.isfinite(vol) or vol < 0:
                raise ValidationError(
                    f"rebalance {exec_date} ticker {t!r}: impact-model "
                    f"volatility is {vol!r}. use_impact_model needs a finite "
                    "non-negative volatility for every ticker actually "
                    "traded; a rebalance inside the impact_lookback warm-up "
                    "window has none yet."
                )
            # Inlined from impact_cost -> sqrt_impact_bps, which between them
            # ran three `_cost_rate` calls per trade re-validating
            # `impact_coefficient` (a parameter, checked once at entry), the
            # volatility (checked immediately above) and a participation
            # ratio built from `adv`, which `_valid_dollar_volume` has just
            # proved finite and positive. Measured 3.75M of those calls on a
            # 500-name daily-rebalanced panel.
            #
            # Same arithmetic: sqrt_impact_bps multiplies by 1e4 to return
            # basis points and impact_cost divides it straight back out, so
            # the two cancel and neither appears here.
            impact = (
                abs(trade_notional)
                * impact_coefficient
                * vol
                * math.sqrt(abs(trade_notional) / adv)
            )
        return commission + spread + impact

    def _apply_rebalance(
        trigger_date: Any,
        trigger_bar: int,
        exec_date: Any,
        weights_arr: np.ndarray,
        exec_prices: np.ndarray,
    ) -> None:
        """`weights_arr` and `exec_prices` are both (n_tickers,) rows —
        of weights_mat and of the relevant price matrix respectively —
        positionally aligned with `tickers` / `shares_vec`."""
        nonlocal cash
        equity_now = cash + float(shares_vec @ exec_prices)

        # No cost-rate validation here. The rates are function parameters --
        # they cannot change between rebalances -- and every one of them is
        # validated once at entry. Re-checking them per trade (399,800
        # _cost_rate calls, 0.28 s, on a daily-rebalanced 100-name backtest)
        # revalidated the same three numbers over and over and could never
        # have reported anything the entry check had not already rejected.
        turnover_notional = 0.0
        # ── Vectorized fast path ─────────────────────────────────────────
        # Engages only for the default cost configuration. per_share
        # commission has a per-ORDER minimum, the impact model needs a
        # per-ticker volatility lookup, and the ADV constraint must raise
        # naming one ticker -- each is a genuinely per-element decision, so
        # they keep the explicit loop below rather than being bent into a
        # vector form that would have to restate their semantics.
        #
        # The arithmetic is identical to the loop: target = equity * w / p,
        # delta = target - held, notional = |delta| * p, cost = |notional| *
        # (commission + slippage). Cash accumulates as a sum, which is the
        # only cross-element dependency and is exactly what np.sum does.
        if (
            commission_model == "pct"
            and not use_impact_model
            and max_adv_participation is None
        ):
            prices_v = np.asarray(exec_prices, dtype=np.float64)
            bad = ~np.isfinite(prices_v) | (prices_v <= 0)
            if np.any(bad & (np.abs(weights_arr) > 1e-12)):
                first = int(np.flatnonzero(bad & (np.abs(weights_arr) > 1e-12))[0])
                raise ValidationError(
                    f"rebalance {exec_date} ticker {tickers[first]!r}: execution "
                    f"price {float(prices_v[first])!r} is nonpositive or "
                    "non-finite — cannot size a nonzero target weight from it."
                )
            safe_prices = np.where(bad, 1.0, prices_v)
            target = np.where(bad, 0.0, equity_now * weights_arr / safe_prices)
            delta = target - shares_vec
            # Same zero-size-trade rule as the loop: an unchanged target
            # costs nothing and generates no turnover.
            traded = np.abs(delta) > 1e-9
            notional = np.abs(delta) * prices_v
            turnover_notional = float(np.sum(np.where(traded, notional, 0.0)))
            # Direction-aware commission stays vectorized: the side is an
            # element-wise select on the sign of delta, not a per-element
            # decision that would force the loop. A sale is delta < 0 --
            # reducing a long or extending a short both pay the sell rate.
            rate = np.where(
                delta < 0.0,
                sell_commission_rate + slippage_pct,
                commission_pct + slippage_pct,
            )
            costs = np.where(traded, notional * rate, 0.0)
            cash -= float(np.sum(np.where(traded, delta * prices_v, 0.0)))
            cash -= float(np.sum(costs))
            shares_vec[:] = target
        else:
            for pos, t in enumerate(tickers):
                price = float(exec_prices[pos])
                weight = float(weights_arr[pos])
                if not math.isfinite(price) or price <= 0:
                    # Upfront validation already rejects nonpositive/non-finite
                    # prices anywhere on the master trading calendar, so this
                    # should be structurally unreachable -- kept as a defensive
                    # invariant, same pattern as the sizing self-consistency
                    # check below. A zero target weight needs no valid price to
                    # size (there's nothing to buy), so only raise when this
                    # ticker was actually being sized to a nonzero position.
                    if abs(weight) > 1e-12:
                        raise ValidationError(
                            f"rebalance {exec_date} ticker {t!r}: execution price "
                            f"{price!r} is nonpositive or non-finite — cannot size "
                            "a nonzero target weight from it."
                        )
                    target_shares = 0.0
                else:
                    target_shares = equity_now * weight / price
                delta = target_shares - shares_vec[pos]

                # Zero-size trade: target didn't change since the last
                # rebalance. Skip entirely — no cost, no turnover, no ADV
                # check — so a per_share_commission minimum floor (or any
                # future per-order minimum) can't charge a ticker that isn't
                # actually trading.
                if abs(delta) <= 1e-9:
                    shares_vec[pos] = target_shares
                    continue

                trade_notional = abs(delta) * price
                turnover_notional += trade_notional

                if max_adv_participation is not None:
                    adv = _valid_dollar_volume(t, pos, trigger_bar, exec_date)
                    participation = adv_participation(trade_notional, adv)
                    # NaN means the participation could not be estimated at all
                    # (no usable volume baseline). It must be checked BEFORE the
                    # comparison, because `nan > limit` is False — so an
                    # unmeasurable trade would otherwise pass a liquidity
                    # constraint that a merely large trade fails. A constraint
                    # the caller explicitly asked for should not be silently
                    # satisfied by absent data.
                    if not math.isfinite(participation):
                        raise ValidationError(
                            f"rebalance {exec_date} ticker {t!r}: ADV participation "
                            "could not be estimated (no usable dollar-volume "
                            "baseline for this ticker/date), so "
                            f"max_adv_participation={max_adv_participation} cannot "
                            "be enforced. Supply volume data for this ticker or "
                            "drop the constraint — it is not satisfied by default."
                        )
                    if participation > max_adv_participation + 1e-9:
                        raise ValidationError(
                            f"rebalance {exec_date} ticker {t!r}: ADV participation "
                            f"{participation:.4f} exceeds max_adv_participation={max_adv_participation}"
                        )

                cash -= delta * price
                cash -= _trade_cost(
                    t, pos, delta, trade_notional, trigger_bar, exec_date
                )
                shares_vec[pos] = target_shares

        # The three post-trade invariant checks below all interrogate the
        # same quantity -- the signed market value of each position -- so it
        # is formed once here rather than rebuilt by three separate
        # per-ticker generators (201,899 evaluations each on a
        # daily-rebalanced 100-name backtest, purely to re-multiply numbers
        # already computed).
        position_values = shares_vec * exec_prices
        abs_position_values = np.abs(position_values)
        equity_after = cash + float(position_values.sum())
        gross_after = float(abs_position_values.sum())

        # Insolvency: this engine models a cash-settled account with no
        # forced-liquidation/margin-call machinery, so zero or negative
        # equity has no meaningful next state to simulate — continuing
        # would let leverage_curve divide by a negative equity (producing a
        # nonsensical negative leverage value) and let downstream
        # annualized-return calculations raise on a negative base raised to
        # a fractional power. Fail fast instead of returning a result that
        # looks like a number but isn't economically meaningful.
        if equity_after <= 0:
            raise ValidationError(
                f"rebalance {exec_date}: account equity is {equity_after!r} "
                "(zero or negative) after this rebalance's costs — insolvent; "
                "this engine does not model forced liquidation/margin calls, "
                "so the simulation cannot continue meaningfully."
            )

        # Sizing self-consistency check: shares[t] were sized as
        # equity_now * weight / price, so gross_after == (sum of |weight|) *
        # equity_now exactly (costs never touch shares[t], only cash) — a
        # value the per-date weight validation above already guarantees is
        # <= max_gross_leverage * equity_now. Comparing gross_after to
        # equity_now (the actual sizing basis) re-asserts that invariant
        # against the realized trade rather than the raw input weights, so
        # it still catches a future sizing bug, without false-flagging the
        # ordinary and unavoidable fact that transaction costs shrink
        # equity_after below equity_now — which mechanically pushes the
        # *reported* gross_after/equity_after ratio (see "gross_leverage_
        # after" in rebalance_log) above the nominal limit on every trade at
        # or near the boundary, with no sizing bug involved. Comparing
        # against equity_after with a tight tolerance would reject that
        # normal, cost-driven drift on almost every fully-invested backtest.
        if equity_now > 0:
            realized_gross_leverage = gross_after / equity_now
            if realized_gross_leverage > max_gross_leverage + 1e-9:
                raise ValidationError(
                    f"rebalance {exec_date}: realized gross leverage "
                    f"{realized_gross_leverage:.4f} exceeds max_gross_leverage="
                    f"{max_gross_leverage} (shares sized from this rebalance's "
                    "equity do not match the requested weights)"
                )
            # Dividing every element by the same positive equity_now is
            # monotonic, so taking the max first and dividing once selects
            # the identical element and yields the identical float. The
            # explicit empty-portfolio guard preserves the `default=0.0`
            # this replaced -- ndarray.max() raises on an empty array.
            realized_max_position = (
                float(abs_position_values.max()) / equity_now if n_tickers else 0.0
            )
            if realized_max_position > max_position_pct + 1e-9:
                raise ValidationError(
                    f"rebalance {exec_date}: realized position size "
                    f"{realized_max_position:.4f} exceeds max_position_pct={max_position_pct}"
                )

        rebalance_log.append(
            {
                "date": (
                    str(trigger_date.date())
                    if hasattr(trigger_date, "date")
                    else str(trigger_date)
                ),
                "turnover_pct": (
                    round(turnover_notional / equity_after, 6)
                    if equity_after > 0
                    else 0.0
                ),
                "gross_leverage_after": (
                    round(gross_after / equity_after, 6) if equity_after > 0 else 0.0
                ),
                "n_positions": int(np.count_nonzero(np.abs(shares_vec) > 1e-9)),
            }
        )

    # ── Native fast path ──────────────────────────────────────────────────
    # The bar loop below is Python. It has already been optimized hard --
    # dense matrices materialized once, positional indexing, a vectorized
    # rebalance branch -- and still costs 124.6 us/bar at 500 tickers,
    # extrapolating to ~450 us/bar at 2,000. A walk-forward or a parameter
    # sweep multiplies that by fifty or a hundred.
    #
    # The kernel covers exactly the configuration _apply_rebalance's own
    # vectorized branch covers, and nothing else: percentage commission, no
    # impact model, no ADV constraint. The per-share model has a per-ORDER
    # minimum, the impact model needs a per-ticker volatility lookup, and the
    # ADV constraint has to raise naming one ticker -- each is a per-element
    # decision that would have to be restated in C++ to be supported, and
    # restating it is how two implementations drift.
    #
    # Anything else falls through to the loop, which is unchanged: the diff
    # that introduced this is an indent plus this guard.
    _native = _native_portfolio_sim(
        close_mat=close_mat,
        open_mat=open_mat,
        hl2_mat=hl2_mat,
        weights_mat=weights_mat,
        rebalance_index=target_weights.index,
        master_index=master_index,
        tickers=tickers,
        fill_price=fill_price,
        commission_model=commission_model,
        use_impact_model=use_impact_model,
        max_adv_participation=max_adv_participation,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        sell_commission_rate=sell_commission_rate,
        slippage_pct=slippage_pct,
        max_gross_leverage=max_gross_leverage,
        max_position_pct=max_position_pct,
        borrow_fee_bps=borrow_fee_bps,
        margin_interest_rate=margin_interest_rate,
        per_share_rate=per_share_rate,
        min_commission=min_commission,
        impact_coefficient=impact_coefficient,
        dollar_volume_mat=dollar_volume_mat,
        volatility_mat=volatility_mat,
    )
    if _native is not None:
        (
            equity_records,
            cash_records,
            gross_records,
            net_records,
            rebalance_log,
            peak_position_value,
        ) = _native
    else:
        # (trigger_date, trigger_bar, weights_arr) awaiting execution at the
        # *next* bar's Open
        # — only used when fill_price == "next_open".
        pending_rebalance: Any = None
        prev_date: Any = None

        for bar, date in enumerate(master_index):
            close_prices = close_mat[bar]

            # Daily-accrued financing costs (borrow fee on shorts, margin
            # interest on negative cash), based on the position/cash carried
            # into this bar from the previous one — before today's rebalance,
            # if any, changes them. Uses the actual elapsed CALENDAR days since
            # the prior bar (e.g. 3 for a Friday->Monday gap over a weekend),
            # not a hardcoded 1 — a fixed days=1 would under-accrue financing
            # across every weekend/holiday gap in the trading calendar.
            if borrow_fee_bps > 0.0 or margin_interest_rate > 0.0:
                days = float((date - prev_date).days) if prev_date is not None else 1.0
                daily_cost = margin_interest(cash, margin_interest_rate, days=days)
                if borrow_fee_bps > 0.0:
                    # The borrow fee is LINEAR in notional, so the per-ticker
                    # loop this replaces was scaling each short's notional by
                    # the same rate and adding them up. Summing the short book's
                    # notional first and scaling once is the same quantity (to
                    # within floating-point associativity) and avoids 200,000
                    # short_borrow_cost calls on a 100-name, 2,000-bar backtest,
                    # each re-validating the same two scalars.
                    short_notional = float(
                        np.abs(
                            np.where(shares_vec < 0, shares_vec * close_prices, 0.0)
                        ).sum()
                    )
                    if short_notional > 0.0:
                        daily_cost += short_borrow_cost(
                            short_notional, borrow_fee_bps, days=days
                        )
                cash -= daily_cost

            if fill_price == "next_open" and pending_rebalance is not None:
                trigger_date, trigger_bar, pending_weights = pending_rebalance
                _apply_rebalance(
                    trigger_date, trigger_bar, date, pending_weights, open_mat[bar]
                )
                pending_rebalance = None

            rebalance_row = rebalance_rows.get(date)
            if rebalance_row is not None:
                weights_arr = weights_mat[rebalance_row]
                if fill_price == "close":
                    _apply_rebalance(date, bar, date, weights_arr, close_prices)
                elif fill_price == "hl2_exploratory":
                    _apply_rebalance(date, bar, date, weights_arr, hl2_mat[bar])
                else:  # next_open — defer execution to the following bar's Open
                    pending_rebalance = (date, bar, weights_arr)

            # Equity is always marked to Close, regardless of fill_price — only
            # the rebalance trade's own execution price changes.
            # One elementwise product per bar instead of a per-ticker Python
            # sum. Net and gross exposure are both reductions over this same
            # vector, so it is formed once and reduced twice.
            position_values = shares_vec * close_prices
            position_value = float(position_values.sum())
            equity = cash + position_value

            # Insolvency (see the identical check in _apply_rebalance): a
            # position that drifts to zero/negative equity purely from price
            # moves between rebalances (no trade involved) is just as
            # meaningless to continue marking as one that goes insolvent AT a
            # rebalance — same fail-fast rationale.
            if equity <= 0:
                raise ValidationError(
                    f"{date}: account equity is {equity!r} (zero or negative) — "
                    "insolvent; this engine does not model forced liquidation/"
                    "margin calls, so the simulation cannot continue meaningfully."
                )

            equity_records.append(equity)
            cash_records.append(cash)
            abs_position_values = np.abs(position_values)
            gross_records.append(float(abs_position_values.sum()))
            net_records.append(position_value)
            # The largest single position held on this bar, in currency. Rides
            # the vector already formed above -- one more reduction over it,
            # not another pass over the book.
            if abs_position_values.size:
                peak_position_value = max(
                    peak_position_value, float(abs_position_values.max())
                )
            prev_date = date

    warnings: List[str] = []
    if fill_price == "close" and rebalance_dates:
        warnings.append(
            "fill_price='close': each rebalance executes at the same bar's own Close, "
            "the same price its target weight is dated on. If target_weights was derived "
            "from that bar's own Close (e.g. a same-day signal/score), this is a look-ahead "
            "bias — the trade could not actually have been placed at that price in real "
            "time. Use fill_price='next_open' for a lookahead-free simulation, or confirm "
            "target_weights was already known before this bar's Close (e.g. computed from "
            "the prior bar's data)."
        )
    if fill_price == "hl2_exploratory" and rebalance_dates:
        warnings.append(
            "fill_price='hl2_exploratory': each rebalance executes at the same bar's "
            "own (High + Low) / 2 ('HL2') — not a real bid/ask midpoint quote, and not "
            "knowable until that bar has already completed (High/Low are only "
            "determined in retrospect), so this is look-ahead the same way "
            "fill_price='close' is. Intended for exploratory analysis only; use "
            "fill_price='next_open' for a lookahead-free simulation."
        )
    if any(c < 0 for c in cash_records):
        warnings.append(
            "cash went negative at one or more bars — implied margin borrowing"
        )

    equity_curve = pd.Series(equity_records, index=master_index, name="equity")
    cash_curve = pd.Series(cash_records, index=master_index, name="cash")
    gross_exposure_curve = pd.Series(
        gross_records, index=master_index, name="gross_exposure"
    )
    equity_safe = equity_curve.where(equity_curve.abs() > 1e-9, other=1e-9)
    leverage_curve = (gross_exposure_curve / equity_safe).rename("leverage")

    logger.debug(
        "[portfolio_engine] tickers=%d  bars=%d  rebalances=%d  fill_price=%s  final_equity=%.2f",
        len(tickers),
        len(master_index),
        len(rebalance_log),
        fill_price,
        float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
    )

    return {
        "equity_curve": equity_curve,
        "cash_curve": cash_curve,
        "gross_exposure_curve": gross_exposure_curve,
        "net_exposure_curve": pd.Series(
            net_records, index=master_index, name="net_exposure"
        ),
        "leverage_curve": leverage_curve,
        "rebalance_log": pd.DataFrame(
            rebalance_log,
            columns=["date", "turnover_pct", "gross_leverage_after", "n_positions"],
        ),
        "final_equity": (
            float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital
        ),
        "final_cash": (
            float(cash_curve.iloc[-1]) if not cash_curve.empty else initial_capital
        ),
        # ── Peak-exposure diagnostics ────────────────────────────────────
        # The curves above answer "what did this portfolio look like
        # typically"; these four answer "how bad did it get", which is the
        # question a risk limit is actually written against. An average
        # gross exposure of 0.9 is perfectly compatible with a single day at
        # 2.4, and only one of those two numbers breaches a mandate.
        #
        # All four are scalars, not curves. Every input is already computed,
        # so none of them costs a pass over the data, and returning them as
        # series would tax every consumer with a payload it reduces anyway.
        "max_leverage": (
            round(float(leverage_curve.max()), 6) if not leverage_curve.empty else 0.0
        ),
        "max_gross_exposure": (
            round(float(gross_exposure_curve.max()), 6)
            if not gross_exposure_curve.empty
            else 0.0
        ),
        "peak_position_value": round(float(peak_position_value), 6),
        # Total return divided by the number of rebalances that actually
        # executed. Not a per-trade P&L -- a rebalance trades many tickers at
        # once -- but the honest read is "what did each turn of the portfolio
        # earn", which is what says whether the costs above were worth
        # paying. None when nothing executed, because 0 rebalances earning
        # 0.0 is a different statement from a flat result -- a DEFENSIVE
        # branch, not a reachable one: the engine already rejects both
        # routes to an empty log (empty target_weights, and a next_open
        # rebalance with no following bar). Kept, and pinned by a test, so
        # that relaxing either guard cannot silently produce a ZeroDivision.
        "return_over_rebalance": (
            round(
                (float(equity_curve.iloc[-1]) / initial_capital - 1.0)
                / len(rebalance_log),
                6,
            )
            if rebalance_log and not equity_curve.empty
            else None
        ),
        "warnings": warnings,
    }
