import pandas as pd
import numpy as np
from typing import Dict, Any, List
from standard_quant_tools.metrics.return_metrics import cumulative_return, cagr, annualized_volatility
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio
)

def _build_trade_log(prices: pd.Series, executed: pd.Series) -> pd.DataFrame:
    """
    Build a per-trade log from the executed position series.
    Vectorized detection of position changes; only iterates over trade events
    (orders-of-magnitude fewer than bars).
    """
    pos_diff = executed.diff()
    pos_diff.iloc[0] = executed.iloc[0]

    trade_event_idx = pos_diff[pos_diff != 0].index
    if len(trade_event_idx) == 0:
        return pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'direction',
            'entry_price', 'exit_price', 'return_pct'
        ])

    records: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] = {}

    for date in trade_event_idx:
        price = prices[date]
        new_pos = executed[date]

        if open_trade:
            direction = open_trade['direction']
            entry_price = open_trade['entry_price']
            exit_pnl = (price - entry_price) / entry_price * direction
            records.append({
                'entry_date': open_trade['entry_date'],
                'exit_date': date,
                'direction': 'long' if direction == 1 else 'short',
                'entry_price': round(entry_price, 4),
                'exit_price': round(price, 4),
                'return_pct': round(exit_pnl * 100, 4),
            })
            open_trade = {}

        if new_pos != 0:
            open_trade = {
                'entry_date': date,
                'entry_price': price,
                'direction': 1 if new_pos > 0 else -1,
            }

    return pd.DataFrame(records)


def _compute_trade_stats(trade_log: pd.DataFrame) -> Dict[str, float]:
    if trade_log.empty:
        return {'win_rate': 0.0, 'profit_factor': 0.0, 'num_trades': 0, 'avg_trade_return_pct': 0.0}

    num_trades = len(trade_log)
    winners = trade_log[trade_log['return_pct'] > 0]
    losers = trade_log[trade_log['return_pct'] <= 0]

    win_rate = len(winners) / num_trades

    gross_profit = float(winners['return_pct'].to_numpy(dtype=float).sum())
    gross_loss = float(np.abs(losers['return_pct'].to_numpy(dtype=float)).sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    return {
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4),
        'num_trades': num_trades,
        'avg_trade_return_pct': round(trade_log['return_pct'].mean(), 4),
    }


def run_strategy(
    price_data: pd.DataFrame,
    signal_series: pd.Series,
    initial_capital: float = 10_000.0,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    include_trade_log: bool = False,
) -> Dict[str, Any]:
    """
    Vectorized backtesting engine with transaction costs.

    Args:
        price_data: DataFrame with 'Close' column.
        signal_series: Series of 1 (long), 0 (flat), -1 (short).
        initial_capital: Starting capital.
        commission_pct: Round-trip commission per unit of position changed (default 0.1%).
        slippage_pct: Slippage per unit of position changed (default 0.05%).
        include_trade_log: If True, build and return per-trade log.

    Returns:
        Dict with performance metrics, equity curve, and optionally trade_log.
    """
    # --- Align ---
    idx = price_data.index.intersection(signal_series.index)
    prices = price_data.loc[idx, 'Close']
    signals = signal_series.loc[idx]

    # --- Core return calculation ---
    returns = prices.pct_change().fillna(0.0)

    # Shift by 1: signal at close of day t executes at open of day t+1
    executed = signals.shift(1).fillna(0.0)

    # --- Transaction costs (vectorized) ---
    # Cost applies proportionally to the size of position change:
    #   0→1 or 1→0: 1× cost; +1→-1 or -1→+1: 2× cost (full reversal)
    cost_per_unit = commission_pct + slippage_pct
    pos_diff = executed.diff().fillna(executed.iloc[0])
    transaction_costs = pos_diff.abs() * cost_per_unit

    # --- Net strategy returns ---
    strategy_returns = executed * returns - transaction_costs

    # --- Equity curve ---
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    # --- Performance metrics ---
    total_ret = cumulative_return(equity_curve)
    annual_vol = annualized_volatility(strategy_returns)
    sr = sharpe_ratio(strategy_returns)
    srt = sortino_ratio(strategy_returns)
    mdd = max_drawdown(equity_curve)
    cal = calmar_ratio(equity_curve)
    final_eq = equity_curve.iloc[-1] if not equity_curve.empty else initial_capital

    result: Dict[str, Any] = {
        'final_equity': round(final_eq, 2),
        'total_return': round(total_ret, 6),
        'annualized_volatility': round(annual_vol, 6),
        'sharpe_ratio': round(sr, 4),
        'sortino_ratio': round(srt, 4),
        'max_drawdown': round(mdd, 6),
        'calmar_ratio': round(cal, 4),
        'equity_curve': equity_curve,
    }

    # --- Trade-level stats ---
    trade_log = _build_trade_log(prices, executed)
    result.update(_compute_trade_stats(trade_log))

    if include_trade_log:
        result['trade_log'] = trade_log

    return result
