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
from typing import Any, Dict, List

import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_VALID_FILL_PRICES = ("close", "next_open", "midpoint")


def run_portfolio_simulation(
    price_data: Dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    max_gross_leverage: float = 1.0,
    max_position_pct: float = 1.0,
    fill_price: str = "close",
) -> Dict[str, Any]:
    """
    Simulate a single shared-cash portfolio account rebalanced at the dates
    in target_weights.index.

    Args:
        price_data: Dict mapping ticker -> OHLCV DataFrame (must contain 'Close',
            plus 'Open' if fill_price="next_open" or 'High'/'Low' if
            fill_price="midpoint").
        target_weights: DataFrame indexed by rebalance date, one column per
            ticker in the universe, values are the target fraction of
            account equity (negative for short). Must be dense — every
            ticker must have a value at every rebalance date, mirroring
            run_signal_panel_backtest's existing "must have an entry for
            every ticker" contract.
        initial_capital: Starting cash.
        commission_pct, slippage_pct: Applied to the notional traded at each
            rebalance — same convention run_strategy already uses.
        max_gross_leverage: Reject (raise ValidationError) any rebalance date
            whose sum(|weight|) exceeds this (default 1.0 = fully invested,
            no leverage).
        max_position_pct: Reject any single position whose |weight| exceeds
            this (default 1.0).
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

    tickers = list(target_weights.columns)
    missing = [t for t in tickers if t not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

    required_cols = {"close": ["Close"], "next_open": ["Close", "Open"], "midpoint": ["Close", "High", "Low"]}
    for t in tickers:
        missing_cols = [c for c in required_cols[fill_price] if c not in price_data[t].columns]
        if missing_cols:
            raise ValidationError(
                f"price_data[{t!r}] is missing column(s) {missing_cols} required for fill_price={fill_price!r}"
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

    cash = initial_capital
    shares: Dict[str, float] = {t: 0.0 for t in tickers}
    cost_rate = commission_pct + slippage_pct

    equity_records: List[float] = []
    cash_records: List[float] = []
    gross_records: List[float] = []
    net_records: List[float] = []
    rebalance_log: List[Dict[str, Any]] = []

    def _apply_rebalance(trigger_date: Any, weights_row: pd.Series, exec_prices: Dict[str, float]) -> None:
        nonlocal cash
        equity_now = cash + sum(shares[t] * exec_prices[t] for t in tickers)
        turnover_notional = 0.0
        for t in tickers:
            price = exec_prices[t]
            target_shares = (equity_now * float(weights_row[t]) / price) if price > 0 else 0.0
            delta = target_shares - shares[t]
            trade_notional = abs(delta) * price
            turnover_notional += trade_notional
            cash -= delta * price
            cash -= trade_notional * cost_rate
            shares[t] = target_shares

        equity_after = cash + sum(shares[t] * exec_prices[t] for t in tickers)
        gross_after = sum(abs(shares[t] * exec_prices[t]) for t in tickers)
        rebalance_log.append({
            "date": str(trigger_date.date()) if hasattr(trigger_date, "date") else str(trigger_date),
            "turnover_pct": round(turnover_notional / equity_after, 6) if equity_after > 0 else 0.0,
            "gross_leverage_after": round(gross_after / equity_after, 6) if equity_after > 0 else 0.0,
            "n_positions": int(sum(1 for t in tickers if abs(shares[t]) > 1e-9)),
        })

    # (trigger_date, weights_row) awaiting execution at the *next* bar's Open
    # — only used when fill_price == "next_open".
    pending_rebalance: Any = None

    for date in master_index:
        close_prices = {t: float(price_data[t].loc[date, "Close"]) for t in tickers}

        if fill_price == "next_open" and pending_rebalance is not None:
            trigger_date, weights_row = pending_rebalance
            open_prices = {t: float(price_data[t].loc[date, "Open"]) for t in tickers}
            _apply_rebalance(trigger_date, weights_row, open_prices)
            pending_rebalance = None

        if date in rebalance_dates:
            weights_row = target_weights.loc[date]
            if fill_price == "close":
                _apply_rebalance(date, weights_row, close_prices)
            elif fill_price == "midpoint":
                mid_prices = {
                    t: (float(price_data[t].loc[date, "High"]) + float(price_data[t].loc[date, "Low"])) / 2.0
                    for t in tickers
                }
                _apply_rebalance(date, weights_row, mid_prices)
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
    if any(c < 0 for c in cash_records):
        warnings.append("cash went negative at one or more bars — implied margin borrowing")

    equity_curve = pd.Series(equity_records, index=master_index, name="equity")
    cash_curve = pd.Series(cash_records, index=master_index, name="cash")
    gross_exposure_curve = pd.Series(gross_records, index=master_index, name="gross_exposure")
    equity_safe = equity_curve.where(equity_curve.abs() > 1e-9, other=1e-9)
    leverage_curve = (gross_exposure_curve / equity_safe).rename("leverage")

    logger.debug(
        "[portfolio_engine] tickers=%d  bars=%d  rebalances=%d  fill_price=%s  final_equity=%.2f",
        len(tickers), len(master_index), len(rebalance_log), fill_price,
        float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
    )

    return {
        "equity_curve": equity_curve,
        "cash_curve": cash_curve,
        "gross_exposure_curve": gross_exposure_curve,
        "net_exposure_curve": pd.Series(net_records, index=master_index, name="net_exposure"),
        "leverage_curve": leverage_curve,
        "rebalance_log": pd.DataFrame(
            rebalance_log,
            columns=["date", "turnover_pct", "gross_leverage_after", "n_positions"],
        ),
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
        "final_cash": float(cash_curve.iloc[-1]) if not cash_curve.empty else initial_capital,
        "warnings": warnings,
    }
