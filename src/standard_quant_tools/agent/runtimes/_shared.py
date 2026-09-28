"""
Infrastructure every tool runtime needs.

Deliberately small. Only three things are genuinely shared across runtimes --
the C++ extension probe, the interval the backtest tools fetch bars at, and
`_run_backtest`, which the execution and validation tools both call and
which is the reason those two categories live in ONE runtime rather than
two. Everything else belongs to exactly one
runtime and lives there, so this module cannot quietly become the place
where cross-runtime coupling accumulates.
"""

import logging
from typing import Any, Dict, List

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
        "[backtest] result  return=%.2f%%  sharpe=%.3f  maxdd=%.2f%%  trades=%d  win=%.0f%%",
        bt.total_return * 100,
        bt.sharpe_ratio,
        bt.max_drawdown * 100,
        bt.num_trades,
        bt.win_rate * 100,
    )
    return bt
