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
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.backtest.constraints import adv_participation
from standard_quant_tools.backtest.costs import (
    impact_cost,
    margin_interest,
    per_share_commission,
    percentage_commission,
    short_borrow_cost,
)
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_VALID_FILL_PRICES = ("close", "next_open", "midpoint")
_VALID_COMMISSION_MODELS = ("pct", "per_share")


def run_portfolio_simulation(
    price_data: Dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
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
            fill_price="midpoint", or 'Volume' if use_impact_model or
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
            following bar to fill against. "midpoint" — executes on the
            same bar as "close" does, but at that bar's (High+Low)/2 instead
            of Close, as a bid/ask-free proxy for a midquote fill. Equity is
            always marked to Close regardless of fill_price — only the
            rebalance trade's own execution price changes.
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

    tickers = list(target_weights.columns)
    missing = [t for t in tickers if t not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

    required_cols = {
        "close": ["Close"],
        "next_open": ["Close", "Open"],
        "midpoint": ["Close", "High", "Low"],
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

    for date, row in target_weights.iterrows():
        gross = float(row.abs().sum())
        if gross > max_gross_leverage + 1e-9:
            raise ValidationError(
                f"rebalance date {date}: gross leverage {gross:.4f} exceeds "
                f"max_gross_leverage={max_gross_leverage}"
            )
        over = row[row.abs() > max_position_pct + 1e-9]
        if not over.empty:
            raise ValidationError(
                f"rebalance date {date}: position(s) exceed "
                f"max_position_pct={max_position_pct}: {over.to_dict()}"
            )

    rebalance_dates = set(target_weights.index)

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
    dollar_volume: Dict[str, pd.Series] = {}
    volatility: Dict[str, pd.Series] = {}
    if needs_volume:
        for t in tickers:
            dv = (price_data[t]["Volume"] * price_data[t]["Close"]).reindex(
                master_index
            )
            dollar_volume[t] = dv.rolling(impact_lookback, min_periods=1).mean()
    if use_impact_model:
        for t in tickers:
            ret = price_data[t]["Close"].pct_change().reindex(master_index)
            volatility[t] = (
                ret.rolling(impact_lookback, min_periods=1).std().fillna(0.0)
            )

    cash = initial_capital
    shares: Dict[str, float] = {t: 0.0 for t in tickers}

    equity_records: List[float] = []
    cash_records: List[float] = []
    gross_records: List[float] = []
    net_records: List[float] = []
    rebalance_log: List[Dict[str, Any]] = []

    def _valid_dollar_volume(t: str, exec_date: Any) -> float:
        # Fail closed: a caller who explicitly enabled max_adv_participation
        # or use_impact_model asked for a liquidity-aware check — a missing
        # or invalid volume baseline means that check can't be performed,
        # which must reject the trade, not silently pass it as
        # unconstrained. costs.py's adv_participation/impact_cost stay
        # permissive (return 0.0) for callers who use them directly without
        # opting into this engine's safety feature.
        dv = float(dollar_volume[t].loc[exec_date])
        if not math.isfinite(dv) or dv <= 0:
            raise ValidationError(
                f"rebalance {exec_date} ticker {t!r}: average dollar volume is "
                f"{dv!r} (missing or invalid) — max_adv_participation/use_impact_model "
                "require a valid 'Volume' baseline for every ticker actually traded."
            )
        return dv

    def _trade_cost(
        t: str, delta_shares: float, trade_notional: float, exec_date: Any
    ) -> float:
        if commission_model == "per_share":
            commission = per_share_commission(
                delta_shares, per_share_rate, min_commission
            )
        else:
            commission = percentage_commission(trade_notional, commission_pct)
        spread = trade_notional * slippage_pct
        impact = 0.0
        if use_impact_model:
            adv = _valid_dollar_volume(t, exec_date)
            impact = impact_cost(
                trade_notional,
                adv,
                float(volatility[t].loc[exec_date]),
                impact_coefficient,
            )
        return commission + spread + impact

    def _apply_rebalance(
        trigger_date: Any,
        exec_date: Any,
        weights_row: pd.Series,
        exec_prices: Dict[str, float],
    ) -> None:
        nonlocal cash
        equity_now = cash + sum(shares[t] * exec_prices[t] for t in tickers)
        turnover_notional = 0.0
        for t in tickers:
            price = exec_prices[t]
            target_shares = (
                (equity_now * float(weights_row[t]) / price) if price > 0 else 0.0
            )
            delta = target_shares - shares[t]

            # Zero-size trade: target didn't change since the last
            # rebalance. Skip entirely — no cost, no turnover, no ADV
            # check — so a per_share_commission minimum floor (or any
            # future per-order minimum) can't charge a ticker that isn't
            # actually trading.
            if abs(delta) <= 1e-9:
                shares[t] = target_shares
                continue

            trade_notional = abs(delta) * price
            turnover_notional += trade_notional

            if max_adv_participation is not None:
                adv = _valid_dollar_volume(t, exec_date)
                participation = adv_participation(trade_notional, adv)
                if participation > max_adv_participation + 1e-9:
                    raise ValidationError(
                        f"rebalance {exec_date} ticker {t!r}: ADV participation "
                        f"{participation:.4f} exceeds max_adv_participation={max_adv_participation}"
                    )

            cash -= delta * price
            cash -= _trade_cost(t, delta, trade_notional, exec_date)
            shares[t] = target_shares

        equity_after = cash + sum(shares[t] * exec_prices[t] for t in tickers)
        gross_after = sum(abs(shares[t] * exec_prices[t]) for t in tickers)

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
            realized_max_position = max(
                (abs(shares[t] * exec_prices[t]) / equity_now for t in tickers),
                default=0.0,
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
                "n_positions": int(sum(1 for t in tickers if abs(shares[t]) > 1e-9)),
            }
        )

    # (trigger_date, weights_row) awaiting execution at the *next* bar's Open
    # — only used when fill_price == "next_open".
    pending_rebalance: Any = None

    for date in master_index:
        close_prices = {t: float(price_data[t].loc[date, "Close"]) for t in tickers}

        # Daily-accrued financing costs (borrow fee on shorts, margin
        # interest on negative cash), based on the position/cash carried
        # into this bar from the previous one — before today's rebalance,
        # if any, changes them.
        if borrow_fee_bps > 0.0 or margin_interest_rate > 0.0:
            daily_cost = margin_interest(cash, margin_interest_rate, days=1.0)
            for t in tickers:
                if shares[t] < 0:
                    daily_cost += short_borrow_cost(
                        abs(shares[t]) * close_prices[t], borrow_fee_bps, days=1.0
                    )
            cash -= daily_cost

        if fill_price == "next_open" and pending_rebalance is not None:
            trigger_date, weights_row = pending_rebalance
            open_prices = {t: float(price_data[t].loc[date, "Open"]) for t in tickers}
            _apply_rebalance(trigger_date, date, weights_row, open_prices)
            pending_rebalance = None

        if date in rebalance_dates:
            weights_row = target_weights.loc[date]
            if fill_price == "close":
                _apply_rebalance(date, date, weights_row, close_prices)
            elif fill_price == "midpoint":
                mid_prices = {
                    t: (
                        float(price_data[t].loc[date, "High"])
                        + float(price_data[t].loc[date, "Low"])
                    )
                    / 2.0
                    for t in tickers
                }
                _apply_rebalance(date, date, weights_row, mid_prices)
            else:  # next_open — defer execution to the following bar's Open
                pending_rebalance = (date, weights_row)

        # Equity is always marked to Close, regardless of fill_price — only
        # the rebalance trade's own execution price changes.
        equity = cash + sum(shares[t] * close_prices[t] for t in tickers)

        equity_records.append(equity)
        cash_records.append(cash)
        gross_records.append(sum(abs(shares[t] * close_prices[t]) for t in tickers))
        net_records.append(sum(shares[t] * close_prices[t] for t in tickers))

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
        "warnings": warnings,
    }
