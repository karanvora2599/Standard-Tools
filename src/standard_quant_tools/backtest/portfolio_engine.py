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


def run_portfolio_simulation(
    price_data: Dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    max_gross_leverage: float = 1.0,
    max_position_pct: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate a single shared-cash portfolio account rebalanced at the dates
    in target_weights.index.

    Args:
        price_data: Dict mapping ticker -> OHLCV DataFrame (must contain 'Close').
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

    Returns:
        Dict with equity_curve, cash_curve, gross_exposure_curve,
        net_exposure_curve (all pd.Series on the master trading-calendar
        index), rebalance_log (pd.DataFrame: date, turnover_pct,
        gross_leverage_after, n_positions), final_equity, final_cash,
        warnings (list[str]).
    """
    tickers = list(target_weights.columns)
    missing = [t for t in tickers if t not in price_data]
    if missing:
        raise ValidationError(f"price_data is missing OHLCV for: {missing}")

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

    cash = initial_capital
    shares: Dict[str, float] = {t: 0.0 for t in tickers}
    cost_rate = commission_pct + slippage_pct
    rebalance_dates = set(target_weights.index)

    equity_records: List[float] = []
    cash_records: List[float] = []
    gross_records: List[float] = []
    net_records: List[float] = []
    rebalance_log: List[Dict[str, Any]] = []

    for date in master_index:
        prices_t = {t: float(price_data[t].loc[date, "Close"]) for t in tickers}
        equity = cash + sum(shares[t] * prices_t[t] for t in tickers)

        if date in rebalance_dates:
            weights_row = target_weights.loc[date]
            turnover_notional = 0.0
            for t in tickers:
                price = prices_t[t]
                target_shares = (equity * float(weights_row[t]) / price) if price > 0 else 0.0
                delta = target_shares - shares[t]
                trade_notional = abs(delta) * price
                turnover_notional += trade_notional
                cash -= delta * price
                cash -= trade_notional * cost_rate
                shares[t] = target_shares

            equity = cash + sum(shares[t] * prices_t[t] for t in tickers)
            gross_after = sum(abs(shares[t] * prices_t[t]) for t in tickers)
            rebalance_log.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "turnover_pct": round(turnover_notional / equity, 6) if equity > 0 else 0.0,
                "gross_leverage_after": round(gross_after / equity, 6) if equity > 0 else 0.0,
                "n_positions": int(sum(1 for t in tickers if abs(shares[t]) > 1e-9)),
            })

        equity_records.append(equity)
        cash_records.append(cash)
        gross_records.append(sum(abs(shares[t] * prices_t[t]) for t in tickers))
        net_records.append(sum(shares[t] * prices_t[t] for t in tickers))

    warnings: List[str] = []
    if any(c < 0 for c in cash_records):
        warnings.append("cash went negative at one or more bars — implied margin borrowing")

    equity_curve = pd.Series(equity_records, index=master_index, name="equity")
    cash_curve = pd.Series(cash_records, index=master_index, name="cash")

    logger.debug(
        "[portfolio_engine] tickers=%d  bars=%d  rebalances=%d  final_equity=%.2f",
        len(tickers), len(master_index), len(rebalance_log),
        float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
    )

    return {
        "equity_curve": equity_curve,
        "cash_curve": cash_curve,
        "gross_exposure_curve": pd.Series(gross_records, index=master_index, name="gross_exposure"),
        "net_exposure_curve": pd.Series(net_records, index=master_index, name="net_exposure"),
        "rebalance_log": pd.DataFrame(
            rebalance_log,
            columns=["date", "turnover_pct", "gross_leverage_after", "n_positions"],
        ),
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
        "final_cash": float(cash_curve.iloc[-1]) if not cash_curve.empty else initial_capital,
        "warnings": warnings,
    }
