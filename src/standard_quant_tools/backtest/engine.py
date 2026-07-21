import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.metrics.return_metrics import cumulative_return, annualized_volatility
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, max_drawdown, calmar_ratio, sortino_ratio,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

# ── Optional C++ fast path ────────────────────────────────────────────────────
from typing import Any as _Any
_cpp_core: _Any = None
HAS_CPP = False
try:
    from standard_quant_tools import _sqt_core as _cpp_core  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

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
            "entry_date", "exit_date", "direction",
            "entry_price", "exit_price", "return_pct",
        ])

    records: List[Dict[str, Any]] = []
    open_trade: Dict[str, Any] = {}

    for date in trade_event_idx:
        price = prices[date]
        new_pos = executed[date]

        if open_trade:
            direction = open_trade["direction"]
            entry_price = open_trade["entry_price"]
            exit_pnl = (price - entry_price) / entry_price * direction
            records.append({
                "entry_date": open_trade["entry_date"],
                "exit_date": date,
                "direction": "long" if direction == 1 else "short",
                "entry_price": round(entry_price, 4),
                "exit_price": round(price, 4),
                "return_pct": round(exit_pnl * 100, 4),
            })
            open_trade = {}

        if new_pos != 0:
            open_trade = {
                "entry_date": date,
                "entry_price": price,
                "direction": 1 if new_pos > 0 else -1,
            }

    # Close any position still open at the last bar (e.g. buy-and-hold, trend strategies
    # that never exit). Mark the exit date and price as the final bar in the series.
    if open_trade:
        last_date = prices.index[-1]
        last_price = prices.iloc[-1]
        direction = open_trade["direction"]
        entry_price = open_trade["entry_price"]
        exit_pnl = (last_price - entry_price) / entry_price * direction
        records.append({
            "entry_date": open_trade["entry_date"],
            "exit_date": last_date,
            "direction": "long" if direction == 1 else "short",
            "entry_price": round(entry_price, 4),
            "exit_price": round(float(last_price), 4),
            "return_pct": round(exit_pnl * 100, 4),
        })

    return pd.DataFrame(records)


def _compute_trade_stats(trade_log: pd.DataFrame) -> Dict[str, float]:
    if trade_log.empty:
        return {"win_rate": 0.0, "profit_factor": 0.0, "num_trades": 0, "avg_trade_return_pct": 0.0}

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
        commission_pct: Commission per unit of position changed (default 0.1%).
        slippage_pct: Slippage per unit of position changed (default 0.05%).
        include_trade_log: If True, build and return per-trade log.

    Returns:
        Dict with performance metrics, equity curve, and optionally trade_log.
    """
    idx = price_data.index.intersection(signal_series.index)
    prices = price_data.loc[idx, "Close"]
    signals = signal_series.loc[idx]

    n_bars = len(prices)
    logger.debug(
        "[run_strategy] bars=%d  capital=%.0f  commission=%.4f  slippage=%.4f",
        n_bars, initial_capital, commission_pct, slippage_pct,
    )

    returns = prices.pct_change().fillna(0.0)
    executed = signals.shift(1).fillna(0.0)

    # ── C++ fast path ─────────────────────────────────────────────────────────
    # Pass raw signals — C++ applies the one-bar lag internally (executed[i] = signals[i-1]).
    # Do NOT pass `executed` here: it is already shifted, which would cause a 2-bar lag.
    if HAS_CPP and _cpp_core is not None:
        logger.debug("[run_strategy] using C++ kernel")
        prices_arr  = prices.to_numpy(dtype=np.float64)
        signals_arr = signals.to_numpy(dtype=np.float64)
        r = _cpp_core.run_strategy(prices_arr, signals_arr,
                                   initial_capital, commission_pct, slippage_pct)
        equity_curve = pd.Series(r["equity_curve"], index=idx)
        result: Dict[str, Any] = {
            "final_equity":          round(float(r["final_equity"]),         2),
            "total_return":          round(float(r["total_return"]),          6),
            "annualized_volatility": round(float(r["annualized_volatility"]), 6),
            "sharpe_ratio":          round(float(r["sharpe_ratio"]),          4),
            "sortino_ratio":         round(float(r["sortino_ratio"]),         4),
            "max_drawdown":          round(float(r["max_drawdown"]),          6),
            "calmar_ratio":          round(float(r["calmar_ratio"]),          4),
            "equity_curve":          equity_curve,
            "win_rate":              round(float(r["win_rate"]),              4),
            "profit_factor":         round(float(r["profit_factor"]),         4),
            "num_trades":            int(r["num_trades"]),
            "avg_trade_return_pct":  round(float(r["avg_trade_return_pct"]), 4),
        }
        if include_trade_log:
            result["trade_log"] = _build_trade_log(prices, executed)
        logger.debug(
            "[run_strategy] C++  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
            result["total_return"] * 100, result["sharpe_ratio"],
            result["num_trades"], result["max_drawdown"] * 100,
        )
        return result

    # ── Python fallback ───────────────────────────────────────────────────────
    logger.debug("[run_strategy] using Python fallback")
    cost_per_unit = commission_pct + slippage_pct
    pos_diff = executed.diff().fillna(executed.iloc[0])
    transaction_costs = pos_diff.abs() * cost_per_unit

    strategy_returns = executed * returns - transaction_costs
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    total_ret = cumulative_return(equity_curve)
    annual_vol = annualized_volatility(strategy_returns)
    sr = sharpe_ratio(strategy_returns)
    srt = sortino_ratio(strategy_returns)
    mdd = max_drawdown(equity_curve)
    cal = calmar_ratio(equity_curve)
    final_eq = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital

    result = {
        "final_equity": round(final_eq, 2),
        "total_return": round(total_ret, 6),
        "annualized_volatility": round(annual_vol, 6),
        "sharpe_ratio": round(sr, 4),
        "sortino_ratio": round(srt, 4),
        "max_drawdown": round(mdd, 6),
        "calmar_ratio": round(cal, 4),
        "equity_curve": equity_curve,
    }

    trade_log = _build_trade_log(prices, executed)
    result.update(_compute_trade_stats(trade_log))

    if include_trade_log:
        result["trade_log"] = trade_log

    logger.debug(
        "[run_strategy] Python  return=%.2f%%  sharpe=%.3f  trades=%d  maxdd=%.2f%%",
        result["total_return"] * 100, result["sharpe_ratio"],
        result["num_trades"], result["max_drawdown"] * 100,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Grid search — module-level worker (must be picklable for ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────

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
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.update(job["params"])
    return result


def _run_signal_fn_job(
    price_data: pd.DataFrame,
    signal_fn: Callable[..., pd.Series],
    params: Dict[str, Any],
    initial_capital: float,
    commission_pct: float,
    slippage_pct: float,
) -> Dict[str, Any]:
    """
    Sequential-only counterpart to _run_grid_job for a user-supplied signal
    callable. Always runs in the calling process (never via
    ProcessPoolExecutor), so signal_fn need not be picklable.
    """
    signals = signal_fn(price_data, **params)
    result = run_strategy(
        price_data, signals,
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )
    result.pop("equity_curve", None)
    result.pop("trade_log", None)
    result.update(params)
    return result


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
) -> pd.DataFrame:
    """
    Run a backtest across every parameter combination in param_grid in parallel.

    Args:
        price_data:     OHLCV DataFrame (from provider.get_ohlcv).
        strategy:       Either one of the built-in registry names
                        ('sma_crossover', 'rsi_mean_reversion', 'macd_crossover',
                        'bollinger_reversion'), or your own signal-generating
                        callable with signature `(price_data: pd.DataFrame, **params)
                        -> pd.Series` (values in {-1, 0, 1}). A custom callable
                        still gets the full C++ batch-kernel speedup when
                        `_sqt_core` is built — only the metric computation runs
                        in C++, so it has no idea whether the signal came from
                        a built-in strategy or your own model.
        param_grid:     Dict mapping parameter name → list of values.
                        e.g. {'fast_period': [5, 10, 20], 'slow_period': [30, 50]}
        initial_capital: Starting capital for every backtest.
        commission_pct: Commission per trade side (fraction).
        slippage_pct:   Slippage per trade side (fraction).
        sort_by:        Output column to rank results by (default: 'sharpe_ratio').
        ascending:      Sort direction (default: False = best first).
        n_workers:      Worker processes. Defaults to os.cpu_count().
                        Pass 1 to run sequentially (no subprocess overhead).
                        Ignored (forced to 1) for a custom callable when the
                        C++ extension is not built — arbitrary callables
                        (lambdas, closures) are frequently unpicklable across
                        the ProcessPoolExecutor spawn boundary.

    Returns:
        pd.DataFrame with one row per parameter combination, sorted by sort_by.
        Columns include all metric keys plus the parameter names.

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
    is_custom = callable(strategy)
    if is_custom:
        signal_fn: Callable[..., pd.Series] = strategy  # type: ignore[assignment]
        strategy_label = getattr(strategy, "__name__", "custom_strategy")
    else:
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(STRATEGY_REGISTRY)}"
            )
        signal_fn = STRATEGY_REGISTRY[strategy]
        strategy_label = strategy

    # Build all parameter combinations
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    t0 = time.perf_counter()

    # ── C++ batch path ────────────────────────────────────────────────────────
    # Generate all signal arrays in Python, then ship the entire batch to C++
    # in a single call — no subprocess overhead, no per-combo boundary crossing.
    if HAS_CPP and _cpp_core is not None:
        try:
            prices_arr = price_data["Close"].to_numpy(dtype=np.float64)

            # Build (num_combos × n_bars) signal matrix
            sig_rows = []
            for combo in combos:
                params = dict(zip(keys, combo))
                sig_rows.append(
                    signal_fn(price_data, **params).to_numpy(dtype=np.float64)
                )
            signals_mat = np.ascontiguousarray(
                np.vstack(sig_rows), dtype=np.float64
            )  # shape: (num_combos, n_bars)

            logger.debug(
                "[backtest_grid] strategy=%s  combos=%d  path=C++  sort_by=%s",
                strategy_label, len(combos), sort_by,
            )

            cpp_results = _cpp_core.batch_run_strategy(
                prices_arr, signals_mat,
                initial_capital, commission_pct, slippage_pct,
            )

            rows = []
            for combo, r in zip(combos, cpp_results):
                row = {
                    "final_equity":          round(float(r["final_equity"]),          2),
                    "total_return":          round(float(r["total_return"]),           6),
                    "annualized_volatility": round(float(r["annualized_volatility"]),  6),
                    "sharpe_ratio":          round(float(r["sharpe_ratio"]),           4),
                    "sortino_ratio":         round(float(r["sortino_ratio"]),          4),
                    "max_drawdown":          round(float(r["max_drawdown"]),           6),
                    "calmar_ratio":          round(float(r["calmar_ratio"]),           4),
                    "win_rate":              round(float(r["win_rate"]),               4),
                    "profit_factor":         round(float(r["profit_factor"]),          4),
                    "num_trades":            int(r["num_trades"]),
                    "avg_trade_return_pct":  round(float(r["avg_trade_return_pct"]),  4),
                }
                row.update(dict(zip(keys, combo)))
                rows.append(row)

            df_out = pd.DataFrame(rows)
            if sort_by in df_out.columns:
                df_out = df_out.sort_values(
                    sort_by, ascending=ascending
                ).reset_index(drop=True)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if not df_out.empty and sort_by in df_out.columns:
                best = df_out.iloc[0]
                best_params = {k: best[k] for k in keys if k in best}
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  best %s=%.4f  params=%s",
                    elapsed_ms, sort_by, best[sort_by], best_params,
                )
            else:
                logger.debug(
                    "[backtest_grid] ✓ %.0fms (C++)  %d results",
                    elapsed_ms, len(df_out),
                )
            return df_out

        except Exception as exc:
            logger.warning(
                "[backtest_grid] C++ batch path failed (%s) — falling back to Python", exc
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
                "forcing sequential execution (n_workers=%s ignored)", n_workers
            )
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=1 (forced)  path=Python  sort_by=%s",
            strategy_label, len(combos), sort_by,
        )
        results = [
            _run_signal_fn_job(
                price_data, signal_fn, dict(zip(keys, combo)),
                initial_capital, commission_pct, slippage_pct,
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
            }
            for combo in combos
        ]
        workers = n_workers if n_workers is not None else (os.cpu_count() or 4)
        logger.debug(
            "[backtest_grid] strategy=%s  combos=%d  workers=%d  path=Python  sort_by=%s",
            strategy_label, len(jobs), workers, sort_by,
        )

        if workers == 1 or len(jobs) == 1:
            results = [_run_grid_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run_grid_job, jobs))

    df_out = pd.DataFrame(results)
    if sort_by in df_out.columns:
        df_out = df_out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not df_out.empty and sort_by in df_out.columns:
        best = df_out.iloc[0]
        best_params = {k: best[k] for k in keys if k in best}
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  best %s=%.4f  params=%s",
            elapsed_ms, sort_by, best[sort_by], best_params,
        )
    else:
        logger.debug(
            "[backtest_grid] ✓ %.0fms (Python)  %d results", elapsed_ms, len(df_out)
        )

    return df_out
