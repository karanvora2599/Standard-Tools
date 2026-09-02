"""
The `backtest` runtime: run, optimize, validate and diagnose a strategy.

The library's eight built-in strategies and caller-supplied signals, plus
the walk-forward, regime-adaptive, robustness and cost-sweep machinery that
says whether a result is real. Execution and validation share `_run_backtest`
and are one workflow, which is why they are one runtime rather than two.

Nothing here constructs a portfolio or sizes a position -- that is the
`portfolio` runtime, and asking for it here is refused by name.
"""

import logging
import math
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "sma_crossover": {
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100],
    },
    "rsi_mean_reversion": {
        "period": [7, 14, 21],
        "oversold": [25, 30],
        "overbought": [65, 70],
    },
    "macd_crossover": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [7, 9],
    },
    "bollinger_reversion": {
        "period": [15, 20, 25],
        "num_std": [1.5, 2.0],
    },
    "donchian_breakout": {
        "entry_period": [10, 20, 40],
        "exit_period": [5, 10],
    },
    "momentum_timeseries": {
        "lookback": [30, 60, 90],
        "threshold": [0.0, 0.05],
    },
    "vwap_reversion": {
        "period": [10, 20, 40],
        "entry_threshold": [0.01, 0.02, 0.03],
    },
    "adx_trend": {
        "adx_period": [10, 14, 21],
        "adx_threshold": [20.0, 25.0, 30.0],
    },
}
_REGIME_STRATEGY_MAP: Dict[str, str] = {
    "trending": "sma_crossover",
    "mean_reverting": "rsi_mean_reversion",
    "random_walk": "macd_crossover",
}
# Single canonical default parameters for each strategy (used by compare_strategies)
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "sma_crossover": {"fast_period": 10, "slow_period": 50},
    "rsi_mean_reversion": {"period": 14, "oversold": 30, "overbought": 70},
    "macd_crossover": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_reversion": {"period": 20, "num_std": 2.0},
}

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.agent.models import (
    BacktestCompactInput,
    BacktestDiagnosticsInput,
    BacktestDiagnosticsResult,
    BacktestInput,
    BacktestOptInput,
    BacktestOptResult,
    BacktestResult,
    BacktestResultV2,
    BuyAndHoldInput,
    CompareCostModelsInput,
    CompareCostModelsResult,
    CompareStrategiesInput,
    CompareStrategiesResult,
    CostScenarioResult,
    CostSummary,
    CustomSignalBacktestInput,
    DrawdownEpisode,
    DrawdownTableInput,
    DrawdownTableResult,
    ExposureDiagnostics,
    ExposureSummary,
    MatrixCell,
    MonteCarloSimulationInput,
    MonteCarloSimulationResult,
    OptimizationRun,
    PairTradeBacktestInput,
    PairTradeBacktestResult,
    PerformanceSummary,
    PortfolioSimulationInput,
    PortfolioSimulationResult,
    RebalanceEvent,
    RegimeAdaptiveInput,
    RegimeAdaptiveResult,
    RegimeAdaptiveWalkForwardInput,
    RegimeAdaptiveWalkForwardResult,
    RegimeAdaptiveWalkForwardWindow,
    RiskSummary,
    RobustnessDiagnosticsInput,
    RobustnessDiagnosticsResult,
    SignalPanelBacktestInput,
    SignalPanelBacktestResult,
    SignalType,
    StrategyComparison,
    StrategyMatrixInput,
    StrategyMatrixResult,
    Trade,
    TradeDiagnostics,
    WalkForwardInput,
    WalkForwardResult,
    WalkForwardWindow,
)
from standard_quant_tools.agent.runtimes._shared import (
    _run_backtest,
)
from standard_quant_tools.backtest.artifacts import load_artifact, save_artifact
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.backtest.monte_carlo import simulate_forward_paths
from standard_quant_tools.backtest.pairs import run_pair_backtest as _pair_backtest_run
from standard_quant_tools.backtest.panel import (
    run_signal_panel_backtest as _signal_panel_backtest,
)
from standard_quant_tools.backtest.portfolio_engine import (
    run_portfolio_simulation as _portfolio_engine_run,
)
from standard_quant_tools.backtest.robustness import (
    block_bootstrap_ci as _block_bootstrap_ci,
)
from standard_quant_tools.backtest.robustness import (
    deflated_sharpe_ratio as _deflated_sharpe_ratio,
)
from standard_quant_tools.backtest.robustness import (
    parameter_sensitivity as _parameter_sensitivity,
)
from standard_quant_tools.backtest.sizing import (
    dollar_neutral,
    equal_weight_top_bottom,
    rank_weighted,
    vol_scaled,
    zscore_normalized,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.backtest.walk_forward import (
    longest_losing_streak,
    parameter_turnover,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.diagnostics import (
    drawdown_periods,
    exposure_stats,
    top_n_drawdowns,
    trade_excursions,
    trade_expectancy,
)
from standard_quant_tools.metrics.return_metrics import annualized_volatility, cagr
from standard_quant_tools.metrics.risk_metrics import (
    cvar,
    drawdown_series,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    var_historical,
)
from standard_quant_tools.portfolio.portfolio import (
    build_portfolio,
    fetch_ohlcv_panel_sync,
    fetch_returns_sync,
)


def _apply_signal_fill_policy(
    signal_series: pd.Series,
    price_index: pd.Index,
    policy: str,
) -> pd.Series:
    """
    A caller-submitted signal map only covers the dates it explicitly
    listed (e.g. a monthly rebalance signal against a daily price index).
    Reindexing onto the full price calendar HERE — not downstream, where
    run_strategy intersects the price and signal indices — is what keeps
    the backtest on the real daily bar time scale instead of silently
    collapsing to whatever sparse dates were submitted (which would also
    corrupt annualization, since every metric assumes periods_per_year=252
    daily bars).

    "hold" (default): forward-fill between submitted dates, flat (0.0)
        before the first one — correct for a target-position signal that's
        meant to persist until explicitly changed.
    "flat": no forward-fill — only the exact submitted dates carry a
        nonzero signal; every other bar is flat.
    "error": every price-calendar date must have an explicit entry.
    """
    if policy == "error":
        missing = price_index.difference(signal_series.index)
        if len(missing) > 0:
            raise ValidationError(
                f"signal_fill_policy='error': {len(missing)} price date(s) have no "
                f"corresponding signal entry (e.g. {[str(d.date()) for d in missing[:5]]})"
            )
        return signal_series.reindex(price_index)
    reindexed = signal_series.reindex(price_index)
    if policy == "hold":
        return reindexed.ffill().fillna(0.0)
    return reindexed.fillna(0.0)  # "flat"


def run_sma_backtest(input_data: BacktestInput) -> BacktestResult:
    """SMA crossover backtest: long when fast SMA > slow SMA."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["sma_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_rsi_backtest(input_data: BacktestInput) -> BacktestResult:
    """RSI mean-reversion backtest: enter long at oversold, exit at overbought."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["rsi_mean_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_macd_backtest(input_data: BacktestInput) -> BacktestResult:
    """MACD crossover backtest: long when MACD line crosses above signal line."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["macd_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_bollinger_backtest(input_data: BacktestInput) -> BacktestResult:
    """Bollinger Band mean-reversion: enter at lower band, exit at middle band."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["bollinger_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_buy_and_hold(input_data: BuyAndHoldInput) -> BacktestResult:
    """Buy-and-hold baseline: long the full period. Use to compare against active strategies."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = pd.Series(1.0, index=df.index)
    bt_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="buy_and_hold",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    return _run_backtest(bt_input, df, signals)


def compare_strategies(input_data: CompareStrategiesInput) -> CompareStrategiesResult:
    """
    Run all four strategies on the same symbol/period with a buy-and-hold baseline.
    Returns results sorted by sort_by (best first).
    Use this instead of calling the four backtest tools individually.
    """
    logger.debug(
        "[compare_strategies] %s  %s → %s  sort_by=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.sort_by,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    # Buy-and-hold baseline (long all bars, one trade)
    bh_signals = pd.Series(1.0, index=df.index)
    bh_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="buy_and_hold",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    bh = _run_backtest(bh_input, df, bh_signals)

    strategy_params: Dict[str, Dict[str, Any]] = {
        "sma_crossover": input_data.sma_parameters or _DEFAULT_PARAMS["sma_crossover"],
        "rsi_mean_reversion": input_data.rsi_parameters
        or _DEFAULT_PARAMS["rsi_mean_reversion"],
        "macd_crossover": input_data.macd_parameters
        or _DEFAULT_PARAMS["macd_crossover"],
        "bollinger_reversion": input_data.bollinger_parameters
        or _DEFAULT_PARAMS["bollinger_reversion"],
    }

    comparisons: List[StrategyComparison] = []
    for strat_name, params in strategy_params.items():
        signals = STRATEGY_REGISTRY[strat_name](df, **params)
        bt_input = BacktestInput(
            symbol=input_data.symbol,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            strategy_type=strat_name,
            parameters=params,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
        )
        bt = _run_backtest(bt_input, df, signals)
        comparisons.append(
            StrategyComparison(
                strategy=strat_name,
                parameters=params,
                total_return=bt.total_return,
                sharpe_ratio=bt.sharpe_ratio,
                sortino_ratio=bt.sortino_ratio,
                max_drawdown=bt.max_drawdown,
                calmar_ratio=bt.calmar_ratio,
                win_rate=bt.win_rate,
                num_trades=bt.num_trades,
                final_equity=bt.final_equity,
            )
        )

    # Higher is always better for all supported metrics (max_drawdown: -0.10 > -0.30)
    comparisons.sort(
        key=lambda c: getattr(c, input_data.sort_by, 0.0),
        reverse=True,
    )
    logger.debug(
        "[compare_strategies] winner=%s  sharpe=%.3f  return=%.2f%%  vs B&H=%.2f%%",
        comparisons[0].strategy,
        comparisons[0].sharpe_ratio,
        comparisons[0].total_return * 100,
        bh.total_return * 100,
    )

    return CompareStrategiesResult(
        symbol=input_data.symbol,
        sort_by=input_data.sort_by,
        best_strategy=comparisons[0].strategy,
        buy_and_hold_return=bh.total_return,
        strategies=comparisons,
    )


def run_regime_adaptive_backtest(
    input_data: RegimeAdaptiveInput,
) -> RegimeAdaptiveResult:
    """
    Classify the market regime via Hurst exponent, then automatically select
    and optimise the most appropriate strategy via parameter grid search.

    Regime → Strategy mapping:
      trending      → sma_crossover
      mean_reverting → rsi_mean_reversion
      random_walk   → macd_crossover
    """
    from standard_quant_tools.analysis.hurst import hurst_exponent as _hurst

    logger.debug(
        "[regime_adaptive] %s  %s → %s  hurst_method=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.hurst_method,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    returns = df["Close"].pct_change(fill_method=None).dropna()

    hurst_result = _hurst(returns, method=input_data.hurst_method)
    h = hurst_result["hurst"]
    regime = hurst_result["regime"]
    fit_r2 = hurst_result["fit_r_squared"]

    # "unknown" is returned when Hurst is NaN (insufficient data); default to MACD
    strategy_name = _REGIME_STRATEGY_MAP.get(regime, "macd_crossover")

    grid_map = {
        "sma_crossover": input_data.sma_param_grid,
        "rsi_mean_reversion": input_data.rsi_param_grid,
        "macd_crossover": input_data.macd_param_grid,
        "bollinger_reversion": input_data.bollinger_param_grid,
    }
    param_grid = grid_map[strategy_name] or _DEFAULT_PARAM_GRIDS[strategy_name]

    grid_df = backtest_grid(
        df,
        strategy=strategy_name,
        param_grid=param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by="sharpe_ratio",
        ascending=False,
        n_workers=input_data.n_workers,
        risk_free_rate=input_data.risk_free_rate,
    )

    best_row = grid_df.iloc[0]
    param_keys = list(param_grid.keys())
    best_params: Dict[str, Any] = {
        k: (
            int(best_row[k])
            if isinstance(param_grid[k][0], int)
            else float(best_row[k])
        )
        for k in param_keys
    }

    signals = STRATEGY_REGISTRY[strategy_name](df, **best_params)
    dummy_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type=strategy_name,
        parameters=best_params,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
    )
    bt_result = _run_backtest(dummy_input, df, signals)

    n_combos = 1
    for vals in param_grid.values():
        n_combos *= len(vals)

    logger.debug(
        "[regime_adaptive] H=%.4f  regime=%s  strategy=%s  best_params=%s  combos=%d",
        float(h) if not math.isnan(float(h)) else 0.0,
        regime,
        strategy_name,
        best_params,
        n_combos,
    )
    return RegimeAdaptiveResult(
        symbol=input_data.symbol,
        regime=regime,
        hurst=round(float(h) if not math.isnan(float(h)) else 0.0, 4),
        fit_r_squared=round(float(fit_r2) if not math.isnan(float(fit_r2)) else 0.0, 4),
        selected_strategy=strategy_name,
        best_parameters=best_params,
        grid_combinations=n_combos,
        backtest=bt_result,
    )


def run_regime_adaptive_walkforward_backtest(
    input_data: RegimeAdaptiveWalkForwardInput,
) -> RegimeAdaptiveWalkForwardResult:
    """
    Leakage-free counterpart to run_regime_adaptive_backtest. That tool
    computes Hurst, optimises parameters, and backtests all on the same
    full requested range — useful as a quick exploratory check, but not
    out-of-sample validated. This tool instead walks forward exactly like
    run_walk_forward_backtest: at each non-overlapping window, regime
    detection AND strategy/parameter selection happen strictly on
    train_df, then the frozen (strategy, params) choice is evaluated
    strictly on the following test_df.

    Unlike run_regime_adaptive_backtest's hardcoded regime -> strategy map,
    every window here grid-searches all four registered strategies
    in-sample and keeps whichever wins by sort_by — the regime/Hurst value
    is still computed and reported per window, but purely as diagnostic
    context, not as a hard selector. Reuses backtest/walk_forward.py's
    stitching helpers for the aggregate OOS metrics — same math as
    run_walk_forward_backtest, no new stitching logic.
    """
    from standard_quant_tools.analysis.hurst import hurst_exponent as _hurst

    logger.debug(
        "[regime_adaptive_wf] %s  %s → %s  train=%d  test=%d  hurst_method=%s  sort_by=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.train_bars,
        input_data.test_bars,
        input_data.hurst_method,
        input_data.sort_by,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    n = len(df)

    train_bars = input_data.train_bars
    test_bars = input_data.test_bars
    if n < train_bars + test_bars:
        raise ValueError(
            f"Not enough data for regime-adaptive walk-forward: need at least "
            f"{train_bars + test_bars} bars, got {n}."
        )

    # Only the original 4 strategies have dedicated override fields on
    # RegimeAdaptiveWalkForwardInput; the 4 newer STRATEGY_REGISTRY entries
    # always use _DEFAULT_PARAM_GRIDS below (.get(...) -> None -> falls
    # through to the default), same as any future registry addition would
    # without a matching Pydantic field added here.
    grid_overrides: Dict[str, Optional[Dict[str, List[Any]]]] = {
        "sma_crossover": input_data.sma_param_grid,
        "rsi_mean_reversion": input_data.rsi_param_grid,
        "macd_crossover": input_data.macd_param_grid,
        "bollinger_reversion": input_data.bollinger_param_grid,
    }

    windows: List[RegimeAdaptiveWalkForwardWindow] = []
    oos_signal_tails: List[pd.Series] = []
    cursor = 0
    first_test_start = train_bars

    while cursor + train_bars + test_bars <= n:
        train_df = df.iloc[cursor : cursor + train_bars]
        test_df = df.iloc[cursor + train_bars : cursor + train_bars + test_bars]
        full_slice = df.iloc[cursor : cursor + train_bars + test_bars]

        train_returns = train_df["Close"].pct_change(fill_method=None).dropna()
        hurst_result = _hurst(train_returns, method=input_data.hurst_method)
        h = hurst_result["hurst"]
        regime = hurst_result["regime"]
        fit_r2 = hurst_result["fit_r_squared"]

        best_overall: Optional[Dict[str, Any]] = None
        for strat_name in STRATEGY_REGISTRY:
            param_grid = (
                grid_overrides.get(strat_name) or _DEFAULT_PARAM_GRIDS[strat_name]
            )
            # fill_price MUST match the out-of-sample leg below. Without
            # it the grid defaulted to "close" while the OOS evaluation used
            # the caller's mode, so a walk-forward run selected parameters
            # under same-close execution and then scored them under next-open
            # execution -- the two halves answering different questions.
            #
            # Not cosmetic: measured across 25 random series with a realistic
            # overnight gap, the WINNING parameter pair differed between the
            # two fill modes on 7 of them. The out-of-sample number was then
            # not a fair test of the parameters that were actually chosen.
            grid_df = backtest_grid(
                train_df,
                strategy=strat_name,
                param_grid=param_grid,
                initial_capital=input_data.initial_capital,
                commission_pct=input_data.commission_pct,
                slippage_pct=input_data.slippage_pct,
                sort_by=input_data.sort_by,
                ascending=False,
                n_workers=1,
                fill_price=input_data.fill_price,
                risk_free_rate=input_data.risk_free_rate,
            )
            best_row = grid_df.iloc[0]
            metric_val = float(best_row.get(input_data.sort_by, float("-inf")))
            if best_overall is None or metric_val > best_overall["metric_val"]:
                param_keys = list(param_grid.keys())
                best_params: Dict[str, Any] = {
                    k: (
                        int(best_row[k])
                        if isinstance(param_grid[k][0], int)
                        else float(best_row[k])
                    )
                    for k in param_keys
                }
                best_overall = {
                    "strategy": strat_name,
                    "params": best_params,
                    "metric_val": metric_val,
                    "sharpe": float(best_row.get("sharpe_ratio", 0.0)),
                    "return": float(best_row.get("total_return", 0.0)),
                }

        assert best_overall is not None  # STRATEGY_REGISTRY is never empty
        strategy_name = best_overall["strategy"]
        best_params = best_overall["params"]

        # Warm-up-aware, same as run_walk_forward_backtest: signals generated
        # over train+test together, keeping only the test-bars tail.
        oos_signals_full = STRATEGY_REGISTRY[strategy_name](full_slice, **best_params)
        oos_signals = oos_signals_full.iloc[train_bars:]
        oos_signal_tails.append(oos_signals)

        # Window-scoped diagnostic only — see stitched_oos_* below for the
        # single continuous OOS backtest.
        oos = run_strategy(
            test_df,
            oos_signals,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
            risk_free_rate=input_data.risk_free_rate,
        )

        windows.append(
            RegimeAdaptiveWalkForwardWindow(
                window_index=len(windows),
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                regime=regime,
                hurst=round(float(h) if not math.isnan(float(h)) else 0.0, 4),
                fit_r_squared=round(
                    float(fit_r2) if not math.isnan(float(fit_r2)) else 0.0, 4
                ),
                selected_strategy=strategy_name,
                best_params=best_params,
                in_sample_sharpe=round(best_overall["sharpe"], 4),
                in_sample_return=round(best_overall["return"], 6),
                out_of_sample_sharpe=round(float(oos["sharpe_ratio"]), 4),
                out_of_sample_return=round(float(oos["total_return"]), 4),
                out_of_sample_max_drawdown=round(float(oos["max_drawdown"]), 4),
            )
        )
        cursor += test_bars

    oos_sharpes = [w.out_of_sample_sharpe for w in windows]
    oos_returns = [w.out_of_sample_return for w in windows]
    oos_mdd = [w.out_of_sample_max_drawdown for w in windows]
    pct_profitable = sum(1 for r in oos_returns if r > 0) / len(windows)

    strat_counts = Counter(w.selected_strategy for w in windows)
    most_common_strat, most_common_n = strat_counts.most_common(1)[0]
    strategy_stability = {
        "most_common": most_common_strat,
        "frequency": round(most_common_n / len(windows), 3),
    }

    stitched_signals = pd.concat(oos_signal_tails)
    full_oos_df = df.iloc[first_test_start : cursor + train_bars]
    stitched = run_strategy(
        full_oos_df,
        stitched_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
    )
    worst_window = min(windows, key=lambda w: w.out_of_sample_return)

    result = RegimeAdaptiveWalkForwardResult(
        symbol=input_data.symbol,
        n_windows=len(windows),
        windows=windows,
        avg_oos_sharpe=round(float(np.mean(oos_sharpes)), 4),
        avg_oos_return=round(float(np.mean(oos_returns)), 4),
        avg_oos_max_drawdown=round(float(np.mean(oos_mdd)), 4),
        pct_windows_profitable=round(pct_profitable, 4),
        strategy_stability=strategy_stability,
        stitched_oos_return=round(stitched["total_return"], 6),
        stitched_oos_sharpe=round(stitched["sharpe_ratio"], 4),
        stitched_oos_sortino=round(stitched["sortino_ratio"], 4),
        stitched_oos_max_drawdown=round(stitched["max_drawdown"], 6),
        stitched_oos_calmar=round(stitched["calmar_ratio"], 4),
        worst_oos_window=worst_window.window_index,
        longest_losing_window_streak=longest_losing_streak(oos_returns),
    )
    logger.debug(
        "[regime_adaptive_wf] windows=%d  stitched_sharpe=%.3f  stitched_return=%.2f%%  strategy_stability=%s",
        result.n_windows,
        result.stitched_oos_sharpe,
        result.stitched_oos_return * 100,
        strategy_stability,
    )
    return result


def run_walk_forward_backtest(input_data: WalkForwardInput) -> WalkForwardResult:
    """
    Walk-forward validation: repeatedly optimise parameters on an in-sample
    window (backtest_grid), then evaluate the best parameters on the next
    out-of-sample window. Returns per-window stats and aggregate OOS metrics.

    The OOS windows are non-overlapping; the training window slides forward
    by test_bars each step. Each window's OOS signals are generated from
    train_df + test_df together (so indicators get train_bars of warm-up
    before the OOS region starts, instead of computing over test_df alone —
    wrong for any indicator needing more history than test_bars provides),
    keeping only the test-bars tail. Those tails are then stitched
    chronologically into ONE continuous signal series and run through a
    SINGLE run_strategy call spanning the whole OOS region — one capital
    base, one compounding stream, real transaction costs at every actual
    transition including window boundaries (now just an ordinary bar, not a
    reset) — rather than resetting capital and re-running run_strategy
    independently per window. windows[i].out_of_sample_* stay window-scoped
    diagnostics (their own independent run_strategy call, still useful to
    see per-window) — stitched_oos_* are the economically correct aggregate,
    computed from the single continuous backtest above.
    """
    logger.debug(
        "[walk_forward] %s  strategy=%s  %s → %s  train=%d  test=%d  sort_by=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.start_date,
        input_data.end_date,
        input_data.train_bars,
        input_data.test_bars,
        input_data.sort_by,
    )
    if input_data.strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    n = len(df)

    train_bars = input_data.train_bars
    test_bars = input_data.test_bars
    if n < train_bars + test_bars:
        raise ValueError(
            f"Not enough data for walk-forward: need at least "
            f"{train_bars + test_bars} bars, got {n}."
        )

    windows: List[WalkForwardWindow] = []
    oos_signal_tails: List[pd.Series] = []
    cursor = 0
    first_test_start = train_bars

    while cursor + train_bars + test_bars <= n:
        train_df = df.iloc[cursor : cursor + train_bars]
        test_df = df.iloc[cursor + train_bars : cursor + train_bars + test_bars]
        full_slice = df.iloc[cursor : cursor + train_bars + test_bars]

        # fill_price MUST match the out-of-sample leg below. Without
        # it the grid defaulted to "close" while the OOS evaluation used
        # the caller's mode, so a walk-forward run selected parameters
        # under same-close execution and then scored them under next-open
        # execution -- the two halves answering different questions.
        #
        # Not cosmetic: measured across 25 random series with a realistic
        # overnight gap, the WINNING parameter pair differed between the
        # two fill modes on 7 of them. The out-of-sample number was then
        # not a fair test of the parameters that were actually chosen.
        grid_df = backtest_grid(
            train_df,
            strategy=input_data.strategy,
            param_grid=input_data.param_grid,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            sort_by=input_data.sort_by,
            ascending=False,
            n_workers=1,
            fill_price=input_data.fill_price,
            risk_free_rate=input_data.risk_free_rate,
        )

        best_row = grid_df.iloc[0]
        param_keys = list(input_data.param_grid.keys())
        # Coerce param types: DataFrame stores everything as float; restore original type.
        best_params: Dict[str, Any] = {
            k: (
                int(best_row[k])
                if isinstance(input_data.param_grid[k][0], int)
                else float(best_row[k])
            )
            for k in param_keys
        }
        is_sharpe = float(best_row.get("sharpe_ratio", 0.0))
        is_return = float(best_row.get("total_return", 0.0))

        # Warm-up-aware: generate signals over train+test together, keep only
        # the test-bars tail — a strategy needing more lookback than
        # test_bars alone provides now gets it, from this window's own
        # train_df (not reaching further back than cursor).
        oos_signals_full = STRATEGY_REGISTRY[input_data.strategy](
            full_slice, **best_params
        )
        oos_signals = oos_signals_full.iloc[train_bars:]
        oos_signal_tails.append(oos_signals)

        # Window-scoped diagnostic only — independently capitalized/reset,
        # not the aggregate (see stitched_oos_* below for the continuous one).
        oos = run_strategy(
            test_df,
            oos_signals,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
            risk_free_rate=input_data.risk_free_rate,
        )

        windows.append(
            WalkForwardWindow(
                window_index=len(windows),
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                best_params=best_params,
                in_sample_sharpe=round(is_sharpe, 4),
                in_sample_return=round(is_return, 6),
                out_of_sample_sharpe=round(float(oos["sharpe_ratio"]), 4),
                out_of_sample_return=round(float(oos["total_return"]), 4),
                out_of_sample_max_drawdown=round(float(oos["max_drawdown"]), 4),
            )
        )
        cursor += test_bars

    oos_sharpes = [w.out_of_sample_sharpe for w in windows]
    oos_returns = [w.out_of_sample_return for w in windows]
    oos_mdd = [w.out_of_sample_max_drawdown for w in windows]
    pct_profitable = sum(1 for r in oos_returns if r > 0) / len(windows)

    param_stability: Dict[str, Any] = {}
    for key in input_data.param_grid:
        counts = Counter(str(w.best_params[key]) for w in windows)
        most_common_val, most_common_n = counts.most_common(1)[0]
        param_stability[key] = {
            "most_common": most_common_val,
            "frequency": round(most_common_n / len(windows), 3),
        }

    # Continuous OOS backtest: one signal series spanning the whole OOS
    # region (windows are contiguous by construction, cursor += test_bars
    # each step), one run_strategy call, one capital base.
    stitched_signals = pd.concat(oos_signal_tails)
    full_oos_df = df.iloc[first_test_start : cursor + train_bars]
    stitched = run_strategy(
        full_oos_df,
        stitched_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
    )
    avg_is_sharpe = float(np.mean([w.in_sample_sharpe for w in windows]))
    avg_is_return = float(np.mean([w.in_sample_return for w in windows]))
    worst_window = min(windows, key=lambda w: w.out_of_sample_return)

    result_wf = WalkForwardResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        n_windows=len(windows),
        windows=windows,
        avg_oos_sharpe=round(float(np.mean(oos_sharpes)), 4),
        avg_oos_return=round(float(np.mean(oos_returns)), 4),
        avg_oos_max_drawdown=round(float(np.mean(oos_mdd)), 4),
        pct_windows_profitable=round(pct_profitable, 4),
        param_stability=param_stability,
        stitched_oos_return=round(stitched["total_return"], 6),
        stitched_oos_sharpe=round(stitched["sharpe_ratio"], 4),
        stitched_oos_sortino=round(stitched["sortino_ratio"], 4),
        stitched_oos_max_drawdown=round(stitched["max_drawdown"], 6),
        stitched_oos_calmar=round(stitched["calmar_ratio"], 4),
        is_to_oos_sharpe_decay=round(avg_is_sharpe - stitched["sharpe_ratio"], 4),
        is_to_oos_return_decay=round(avg_is_return - stitched["total_return"], 6),
        worst_oos_window=worst_window.window_index,
        longest_losing_window_streak=longest_losing_streak(oos_returns),
        parameter_turnover=parameter_turnover([w.best_params for w in windows]),
    )
    logger.debug(
        "[walk_forward] windows=%d  avg_OOS_sharpe=%.3f  avg_OOS_return=%.2f%%  profitable=%.0f%%",
        result_wf.n_windows,
        result_wf.avg_oos_sharpe,
        result_wf.avg_oos_return * 100,
        result_wf.pct_windows_profitable * 100,
    )
    return result_wf


def run_backtest_optimization(input_data: BacktestOptInput) -> BacktestOptResult:
    """
    Run a full parameter grid search for one strategy and return the top N combinations.
    Use this to find the best parameters before committing to a single backtest.
    """
    logger.debug(
        "[backtest_opt] %s  %s  %s → %s  sort_by=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.start_date,
        input_data.end_date,
        input_data.sort_by,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    grid_df = backtest_grid(
        price_data=df,
        strategy=input_data.strategy,
        param_grid=input_data.param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by=input_data.sort_by,
        ascending=False,
        n_workers=input_data.n_workers,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
    )

    n_combinations = len(grid_df)
    top_n = min(input_data.top_n, 20, n_combinations)
    top_df = grid_df.head(top_n)

    metric_cols = {
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
        "final_equity",
    }
    param_cols = [c for c in grid_df.columns if c not in metric_cols]

    top_results = [
        OptimizationRun(
            rank=rank,
            parameters={col: row[col] for col in param_cols if col in row},
            total_return=round(float(row.get("total_return", 0.0)), 6),
            sharpe_ratio=round(float(row.get("sharpe_ratio", 0.0)), 4),
            sortino_ratio=round(float(row.get("sortino_ratio", 0.0)), 4),
            calmar_ratio=round(float(row.get("calmar_ratio", 0.0)), 4),
            max_drawdown=round(float(row.get("max_drawdown", 0.0)), 6),
            num_trades=int(row.get("num_trades", 0)),
        )
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1)
    ]

    best_row = top_df.iloc[0]
    best_params = {col: best_row[col] for col in param_cols if col in best_row}
    logger.debug(
        "[backtest_opt] n=%d  best_params=%s  %s=%.4f",
        n_combinations,
        best_params,
        input_data.sort_by,
        float(best_row.get(input_data.sort_by, 0.0)),
    )
    return BacktestOptResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        n_combinations=n_combinations,
        sort_by=input_data.sort_by,
        best_params=best_params,
        best_sharpe=round(float(best_row.get("sharpe_ratio", 0.0)), 4),
        best_return=round(float(best_row.get("total_return", 0.0)), 6),
        top_results=top_results,
    )


def run_custom_signal_backtest(input_data: CustomSignalBacktestInput) -> BacktestResult:
    """
    Backtest a signal computed entirely outside this library (e.g. your own
    alpha model) on a single symbol. Unlike run_sma_backtest / run_rsi_backtest /
    run_macd_backtest / run_bollinger_backtest, this tool does not generate the
    signal — you supply it, and it reuses the same fast backtest engine.
    """
    logger.debug(
        "[custom_signal_backtest] %s  %s → %s  n_signal_points=%d",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        len(input_data.signals),
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    signal_series = pd.Series(
        {pd.Timestamp(d): v for d, v in input_data.signals.items()}
    ).sort_index()
    signal_series = _apply_signal_fill_policy(
        signal_series,
        df.index,
        input_data.signal_fill_policy,
    )

    bt_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="custom_signal",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    return _run_backtest(bt_input, df, signal_series)


def run_signal_panel_backtest(
    input_data: SignalPanelBacktestInput,
) -> SignalPanelBacktestResult:
    """
    Backtest a pre-computed signal panel (e.g. your own cross-sectional alpha
    model) across a ticker universe, combined into portfolio-level metrics.
    Fetches OHLCV internally and reuses run_strategy per ticker plus the
    existing portfolio module for the combination — no new backtest math.
    """
    # A reference is resolved to the same shape the inline field carries,
    # so everything downstream is identical either way -- the two entry
    # points must not be two code paths.
    if input_data.signal_panel_ref is not None:
        from standard_quant_tools.agent.runtimes import handoff

        resolved_panel = handoff.resolve(
            input_data.signal_panel_ref, expect="signal_panel"
        )
        input_data = input_data.model_copy(
            update={"signal_panel": resolved_panel, "signal_panel_ref": None}
        )
    logger.debug(
        "[signal_panel_backtest] tickers=%d  %s → %s",
        len(input_data.tickers),
        input_data.start_date,
        input_data.end_date,
    )
    # Concurrent fetch (bounded by the default executor's thread pool, ~32
    # in flight) rather than one blocking get_ohlcv() call per ticker in a
    # loop — for a large universe (e.g. the full S&P 500), the sequential
    # form used to mean minutes of pure network wait before anything else
    # even started. Needs the full OHLCV panel (Volume/High/Low, not just
    # Close-derived returns), hence fetch_ohlcv_panel_sync rather than the
    # cheaper fetch_returns_sync other multi-ticker tools in this module use.
    price_data = fetch_ohlcv_panel_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )

    signal_panel = pd.DataFrame(
        {
            t: _apply_signal_fill_policy(
                pd.Series(
                    {pd.Timestamp(d): v for d, v in input_data.signal_panel[t].items()}
                ).sort_index(),
                price_data[t].index,
                input_data.signal_fill_policy,
            )
            for t in input_data.tickers
        }
    ).sort_index()

    bench_returns = None
    if input_data.benchmark:
        bench_df = DataFactory.get_provider().get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )
        bench_returns = bench_df["Close"].pct_change(fill_method=None).dropna()

    weights_arg: Any = input_data.weights if input_data.weights is not None else None

    raw = _signal_panel_backtest(
        price_data,
        signal_panel,
        weights=weights_arg,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        benchmark_returns=bench_returns,
        include_trade_log=input_data.include_trade_log,
        fill_price=input_data.fill_price,
    )

    per_ticker: Dict[str, BacktestResult] = {}
    for ticker, res in raw["per_ticker"].items():
        trade_log_raw = res.get("trade_log", pd.DataFrame())
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
        per_ticker[ticker] = BacktestResult(
            total_return=res["total_return"],
            annualized_volatility=res["annualized_volatility"],
            sharpe_ratio=res["sharpe_ratio"],
            sortino_ratio=res["sortino_ratio"],
            max_drawdown=res["max_drawdown"],
            calmar_ratio=res["calmar_ratio"],
            win_rate=res["win_rate"],
            profit_factor=res["profit_factor"],
            num_trades=res["num_trades"],
            avg_trade_return_pct=res["avg_trade_return_pct"],
            final_equity=res["final_equity"],
            equity_curve=res["equity_curve"].tolist(),
            trade_log=trades,
        )

    portfolio_metrics_out = dict(raw["portfolio_metrics"])
    logger.debug(
        "[signal_panel_backtest] portfolio  sharpe=%.3f  return=%.2f%%",
        portfolio_metrics_out.get("sharpe_ratio", float("nan")),
        portfolio_metrics_out.get("annualized_return", 0.0) * 100,
    )

    return SignalPanelBacktestResult(
        tickers=input_data.tickers,
        per_ticker=per_ticker,
        portfolio_metrics=portfolio_metrics_out,
    )


def _metrics_with_day0_cost(
    equity_curve: pd.Series,
    initial_capital: float,
) -> Tuple[pd.Series, float, float, pd.Series]:
    """
    Returns (returns, total_return, annualized_return, equity_with_start)
    accounting for a same-bar-0 rebalance cost that would otherwise be
    invisible: equity_curve.iloc[0] from run_portfolio_simulation/
    run_pair_backtest is already net of that rebalance's costs, so naively
    using cumulative_return/cagr (baseline = equity_curve.iloc[0]) and
    equity_curve.pct_change(fill_method=None) (first return = NaN -> filled 0.0) silently
    drops the day-0 cost's entire effect on every downstream metric.
    Prepending a synthetic pre-trade observation at initial_capital makes
    the first real return capture it instead.

    equity_with_start is also returned (not just consumed internally) so
    callers can compute max_drawdown/calmar_ratio from the same complete
    curve: max_drawdown(equity_curve) alone would treat the already-
    post-cost equity_curve.iloc[0] as the peak, so a portfolio that lost
    5% entirely to day-0 rebalance costs and then stayed flat would
    otherwise report max_drawdown=0.0 (and calmar_ratio=inf) despite
    total_return correctly reporting -5%. calmar_ratio should still be
    computed as annualized_return (above, already using the correct
    initial_capital baseline and undistorted bar count) divided by
    max_drawdown(equity_with_start) -- not via the standalone calmar_ratio()
    metrics function, which would redundantly recompute cagr from
    equity_with_start's own iloc[0] baseline using len(equity_with_start)
    (one bar too many, from the synthetic point) as its bar count.
    """
    if equity_curve.empty:
        return equity_curve, 0.0, 0.0, equity_curve
    synthetic_index = equity_curve.index[0] - pd.Timedelta(days=1)
    equity_with_start = pd.concat(
        [
            pd.Series([initial_capital], index=[synthetic_index]),
            equity_curve,
        ]
    )
    returns = equity_with_start.pct_change(fill_method=None).dropna()
    total_return = float(equity_curve.iloc[-1]) / initial_capital - 1.0
    num_years = len(equity_curve) / 252
    annualized_return = (
        (1.0 + total_return) ** (1.0 / num_years) - 1.0 if num_years > 0 else 0.0
    )
    return returns, total_return, annualized_return, equity_with_start


def run_portfolio_simulation(
    input_data: PortfolioSimulationInput,
) -> PortfolioSimulationResult:
    """
    True shared-cash portfolio simulation: one account, position sizing
    relative to current equity, and rebalancing at the dates in
    target_weights — unlike run_signal_panel_backtest, which gives every
    ticker its own independent capital and only blends per-ticker return
    streams afterward. Reuses backtest/portfolio_engine.py for the
    simulation and the existing metrics functions for the summary; no new
    metric math.

    When signal_type='score', target_weights holds arbitrary per-ticker
    alpha scores instead of weights — converted via construction_method
    (backtest/sizing.py: rank_weighted, equal_weight_top_bottom,
    zscore_normalized, vol_scaled) before simulation.
    """
    # Resolved to the identical shape the inline field carries, so the two
    # entry points stay one code path. The kind is checked against
    # signal_type: a score panel simulated as target weights would size
    # every position at an alpha score, which is a plausible-looking
    # disaster rather than an error.
    if input_data.target_weights_ref is not None:
        from standard_quant_tools.agent.runtimes import handoff

        expected = (
            "score_panel"
            if input_data.signal_type == SignalType.SCORE
            else "weight_panel"
        )
        input_data = input_data.model_copy(
            update={
                "target_weights": handoff.resolve(
                    input_data.target_weights_ref, expect=expected
                ),
                "target_weights_ref": None,
            }
        )
    logger.debug(
        "[portfolio_simulation] tickers=%d  %s → %s  max_gross_leverage=%.2f",
        len(input_data.tickers),
        input_data.start_date,
        input_data.end_date,
        input_data.max_gross_leverage,
    )
    # Concurrent fetch (bounded by the default executor's thread pool, ~32
    # in flight) rather than one blocking get_ohlcv() call per ticker in a
    # loop — for a large universe (e.g. the full S&P 500), the sequential
    # form used to mean minutes of pure network wait before anything else
    # even started. Needs the full OHLCV panel (Volume/High/Low, not just
    # Close-derived returns), hence fetch_ohlcv_panel_sync rather than the
    # cheaper fetch_returns_sync other multi-ticker tools in this module use.
    price_data = fetch_ohlcv_panel_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )

    values_panel = pd.DataFrame(
        {
            t: pd.Series(
                {pd.Timestamp(d): v for d, v in input_data.target_weights[t].items()}
            )
            for t in input_data.tickers
        }
    ).sort_index()

    if input_data.signal_type == SignalType.SCORE:
        method = input_data.construction_method
        if method == "rank_weighted":
            target_weights = rank_weighted(
                values_panel, gross_leverage=input_data.gross_leverage
            )
        elif method == "equal_weight_top_bottom":
            # n_long/n_short non-None is enforced by PortfolioSimulationInput's
            # model validator when construction_method == "equal_weight_top_bottom".
            assert input_data.n_long is not None and input_data.n_short is not None
            target_weights = equal_weight_top_bottom(
                values_panel,
                n_long=input_data.n_long,
                n_short=input_data.n_short,
                gross_leverage=input_data.gross_leverage,
            )
        elif method == "zscore_normalized":
            target_weights = zscore_normalized(
                values_panel, gross_leverage=input_data.gross_leverage
            )
        else:  # vol_scaled — validated to be one of _CONSTRUCTION_METHODS by the input model
            returns_df = pd.DataFrame(
                {t: price_data[t]["Close"].pct_change(fill_method=None) for t in input_data.tickers}
            )
            target_weights = vol_scaled(
                values_panel,
                returns_df,
                lookback=input_data.vol_lookback,
                gross_leverage=input_data.gross_leverage,
            )
        if input_data.make_dollar_neutral:
            target_weights = dollar_neutral(target_weights)
        logger.debug(
            "[portfolio_simulation] converted SCORE signals via construction_method=%s  gross_leverage=%.2f",
            method,
            input_data.gross_leverage,
        )
    else:
        target_weights = values_panel

    raw = _portfolio_engine_run(
        price_data,
        target_weights,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        sell_commission_pct=input_data.sell_commission_pct,
        slippage_pct=input_data.slippage_pct,
        max_gross_leverage=input_data.max_gross_leverage,
        max_position_pct=input_data.max_position_pct,
        fill_price=input_data.fill_price,
        commission_model=input_data.commission_model,
        per_share_rate=input_data.per_share_rate,
        min_commission=input_data.min_commission,
        use_impact_model=input_data.use_impact_model,
        impact_coefficient=input_data.impact_coefficient,
        impact_lookback=input_data.impact_lookback,
        borrow_fee_bps=input_data.borrow_fee_bps,
        margin_interest_rate=input_data.margin_interest_rate,
        max_adv_participation=input_data.max_adv_participation,
    )

    equity_curve = raw["equity_curve"]
    returns, total_return, annualized_return, equity_with_start = (
        _metrics_with_day0_cost(
            equity_curve,
            input_data.initial_capital,
        )
    )
    day0_max_dd = float(max_drawdown(equity_with_start))
    day0_calmar = (
        annualized_return / abs(day0_max_dd) if day0_max_dd != 0.0 else float("inf")
    )

    ir: Optional[float] = None
    if input_data.benchmark:
        bench_df = DataFactory.get_provider().get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )
        bench_returns = bench_df["Close"].pct_change(fill_method=None).dropna()
        common = returns.index.intersection(bench_returns.index)
        if len(common) > 1:
            ir = round(
                float(
                    information_ratio(returns.loc[common], bench_returns.loc[common])
                ),
                4,
            )

    rebalance_events = [
        RebalanceEvent(
            date=str(r["date"]),
            turnover_pct=float(r["turnover_pct"]),
            gross_leverage_after=float(r["gross_leverage_after"]),
            n_positions=int(r["n_positions"]),
        )
        for r in raw["rebalance_log"].to_dict(orient="records")
    ]

    leverage_series = raw["leverage_curve"]
    avg_gross_leverage = (
        round(float(leverage_series.mean()), 4) if not leverage_series.empty else 0.0
    )
    max_gross_leverage_used = (
        round(float(leverage_series.max()), 4) if not leverage_series.empty else 0.0
    )

    logger.debug(
        "[portfolio_simulation] rebalances=%d  final_equity=%.2f  sharpe=%.3f  warnings=%s",
        len(rebalance_events),
        raw["final_equity"],
        float(sharpe_ratio(returns)),
        raw["warnings"],
    )

    return PortfolioSimulationResult(
        tickers=input_data.tickers,
        n_rebalances=len(rebalance_events),
        rebalance_log=rebalance_events,
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        annualized_volatility=round(float(annualized_volatility(returns)), 6),
        sharpe_ratio=round(float(sharpe_ratio(returns)), 4),
        sortino_ratio=round(float(sortino_ratio(returns)), 4),
        max_drawdown=round(day0_max_dd, 6),
        calmar_ratio=round(day0_calmar, 4),
        var_95=round(float(var_historical(returns, 0.95)), 6),
        cvar_95=round(float(cvar(returns, 0.95)), 6),
        information_ratio=ir,
        final_equity=round(float(raw["final_equity"]), 2),
        final_cash=round(float(raw["final_cash"]), 2),
        avg_gross_leverage=avg_gross_leverage,
        max_gross_leverage_used=max_gross_leverage_used,
        equity_curve=equity_curve.tolist(),
        warnings=list(raw["warnings"]),
    )


def run_pair_trade_backtest(
    input_data: PairTradeBacktestInput,
) -> PairTradeBacktestResult:
    """
    Backtest a cointegrated pair as one synchronized two-leg trade — unlike
    scan_pairs, which only screens for cointegration candidates and reports
    a current z-score signal. Reuses backtest/pairs.py, which itself reuses
    run_portfolio_simulation: both legs are columns of the same
    target_weights row, so they enter/exit together by construction and
    share one cash account.
    """
    logger.debug(
        "[pair_trade_backtest] %s/%s  hedge_ratio=%.4f  entry_z=%.2f  exit_z=%.2f",
        input_data.symbol_a,
        input_data.symbol_b,
        input_data.hedge_ratio,
        input_data.entry_z,
        input_data.exit_z,
    )
    provider = DataFactory.get_provider()
    price_data = {
        input_data.symbol_a: provider.get_ohlcv(
            input_data.symbol_a, input_data.start_date, input_data.end_date
        ),
        input_data.symbol_b: provider.get_ohlcv(
            input_data.symbol_b, input_data.start_date, input_data.end_date
        ),
    }

    raw = _pair_backtest_run(
        price_data,
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        hedge_ratio=input_data.hedge_ratio,
        entry_z=input_data.entry_z,
        exit_z=input_data.exit_z,
        zscore_window=input_data.zscore_window,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        gross_leverage=input_data.gross_leverage,
        fill_price=input_data.fill_price,
    )

    equity_curve = raw["equity_curve"]
    returns, total_return, annualized_return, equity_with_start = (
        _metrics_with_day0_cost(
            equity_curve,
            input_data.initial_capital,
        )
    )
    day0_max_dd = float(max_drawdown(equity_with_start))
    day0_calmar = (
        annualized_return / abs(day0_max_dd) if day0_max_dd != 0.0 else float("inf")
    )
    rebalance_events = [
        RebalanceEvent(
            date=str(r["date"]),
            turnover_pct=float(r["turnover_pct"]),
            gross_leverage_after=float(r["gross_leverage_after"]),
            n_positions=int(r["n_positions"]),
        )
        for r in raw["rebalance_log"].to_dict(orient="records")
    ]

    logger.debug(
        "[pair_trade_backtest] rebalances=%d  round_trips=%d  final_equity=%.2f  sharpe=%.3f",
        len(rebalance_events),
        raw["n_round_trips"],
        raw["final_equity"],
        float(sharpe_ratio(returns)),
    )

    return PairTradeBacktestResult(
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        hedge_ratio=raw["hedge_ratio"],
        n_rebalances=len(rebalance_events),
        n_round_trips=raw["n_round_trips"],
        rebalance_log=rebalance_events,
        entry_spread=raw["entry_spread"],
        current_spread=round(float(raw["current_spread"]), 6),
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        annualized_volatility=round(float(annualized_volatility(returns)), 6),
        sharpe_ratio=round(float(sharpe_ratio(returns)), 4),
        sortino_ratio=round(float(sortino_ratio(returns)), 4),
        max_drawdown=round(day0_max_dd, 6),
        calmar_ratio=round(day0_calmar, 4),
        final_equity=round(float(raw["final_equity"]), 2),
        final_cash=round(float(raw["final_cash"]), 2),
        equity_curve=equity_curve.tolist(),
        warnings=list(raw["warnings"]),
    )


def get_robustness_diagnostics(
    input_data: RobustnessDiagnosticsInput,
) -> RobustnessDiagnosticsResult:
    """
    Same-sample robustness checks for a grid search: how much better is the
    best trial than the pack (parameter_sensitivity), does the best trial's
    Sharpe survive correcting for having been selected as the max of
    n_trials attempts (deflated_sharpe_ratio), and a block-bootstrap
    confidence interval on the best trial's Sharpe ratio. This is NOT a
    substitute for run_walk_forward_backtest's out-of-sample validation —
    it quantifies confidence in a same-sample estimate, a different and
    complementary question ("how sure am I this number is real" vs. "would
    it have held up on unseen data").
    """
    logger.debug(
        "[robustness_diagnostics] %s  strategy=%s  sort_by=%s  n_combos=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.sort_by,
        {k: len(v) for k, v in input_data.param_grid.items()},
    )
    if input_data.strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    grid_df = backtest_grid(
        df,
        strategy=input_data.strategy,
        param_grid=input_data.param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by=input_data.sort_by,
        ascending=False,
        risk_free_rate=input_data.risk_free_rate,
    )
    sensitivity = _parameter_sensitivity(grid_df, metric_col=input_data.sort_by)

    best_row = grid_df.iloc[0]
    param_keys = list(input_data.param_grid.keys())
    best_params: Dict[str, Any] = {
        k: (
            int(best_row[k])
            if isinstance(input_data.param_grid[k][0], int)
            else float(best_row[k])
        )
        for k in param_keys
    }
    best_sharpe = float(best_row["sharpe_ratio"])
    sharpe_trials_std = (
        float(grid_df["sharpe_ratio"].std()) if len(grid_df) > 1 else 0.0
    )

    # grid_df's sharpe_ratio column is annualized (run_strategy -> sharpe_ratio's
    # default periods_per_year=252), but deflated_sharpe_ratio's formula requires
    # the non-annualized, per-period Sharpe (its z-score already scales by
    # sqrt(n_obs - 1) itself). De-annualize before the DSR call, then re-annualize
    # expected_max_sharpe for reporting so it's on the same scale as best_sharpe.
    _ANNUALIZATION = np.sqrt(252.0)
    dsr = _deflated_sharpe_ratio(
        observed_sharpe=best_sharpe / _ANNUALIZATION,
        sharpe_trials_std=sharpe_trials_std / _ANNUALIZATION,
        n_trials=len(grid_df),
        n_obs=len(df),
        skew=input_data.skew,
        kurtosis=input_data.kurtosis,
    )
    dsr["expected_max_sharpe"] = round(dsr["expected_max_sharpe"] * _ANNUALIZATION, 6)

    best_signals = STRATEGY_REGISTRY[input_data.strategy](df, **best_params)
    best_result = run_strategy(
        df,
        best_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        risk_free_rate=input_data.risk_free_rate,
    )
    best_returns = best_result["equity_curve"].pct_change(fill_method=None).fillna(0.0)

    bootstrap = _block_bootstrap_ci(
        best_returns,
        sharpe_ratio,
        n_iterations=input_data.n_bootstrap_iterations,
        block_size=input_data.bootstrap_block_size,
        confidence=input_data.bootstrap_confidence,
        seed=input_data.random_seed,
    )

    warnings: List[str] = []
    if len(grid_df) < 5:
        warnings.append(
            f"Only {len(grid_df)} trial(s) searched — parameter sensitivity and "
            "deflated Sharpe estimates are noisy with this few combinations."
        )

    logger.debug(
        "[robustness_diagnostics] trials=%d  best_sharpe=%.3f  expected_max=%.3f  dsr=%.3f",
        len(grid_df),
        best_sharpe,
        dsr["expected_max_sharpe"],
        dsr["deflated_sharpe_ratio"],
    )

    return RobustnessDiagnosticsResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        best_params=best_params,
        parameter_sensitivity=sensitivity,
        expected_max_sharpe=dsr["expected_max_sharpe"],
        deflated_sharpe_ratio=dsr["deflated_sharpe_ratio"],
        bootstrap_point_estimate=round(bootstrap["point_estimate"], 4),
        bootstrap_ci_lower=round(bootstrap["ci_lower"], 4),
        bootstrap_ci_upper=round(bootstrap["ci_upper"], 4),
        bootstrap_confidence=input_data.bootstrap_confidence,
        warnings=warnings,
    )


def run_monte_carlo_simulation(
    input_data: MonteCarloSimulationInput,
) -> MonteCarloSimulationResult:
    """
    Monte Carlo forward simulation: projects possible future equity paths
    from a portfolio's historical return distribution via moving-block
    bootstrap resampling. Forward-looking, unlike get_robustness_diagnostics
    (a same-sample confidence check) or run_walk_forward_backtest (which
    tests actual historical decisions) — this is a projection from
    historical statistics, not a prediction or a validation of any
    strategy's decisions.
    """
    logger.debug(
        "[monte_carlo] tickers=%s  %s → %s  horizon=%d  n_sim=%d",
        input_data.tickers,
        input_data.start_date,
        input_data.end_date,
        input_data.horizon_days,
        input_data.n_simulations,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    n_assets = returns_df.shape[1]
    weights = (
        np.full(n_assets, 1.0 / n_assets)
        if input_data.weights is None
        else input_data.weights
    )
    portfolio_returns = build_portfolio(returns_df, weights)

    # Materialized BEFORE execution, not left to the kernel. With
    # random_seed=None the native path seeded itself from steady_clock, so
    # the audit record faithfully stored `None` while the numbers came from
    # a value that was never captured anywhere — the run was unreproducible
    # and nothing said so. Drawing it here means the seed is in both the
    # result and the decision record, and re-passing it repeats the run.
    resolved_seed = (
        input_data.random_seed
        if input_data.random_seed is not None
        else int(np.random.SeedSequence().entropy % (2**32))
    )

    result = simulate_forward_paths(
        portfolio_returns,
        horizon_days=input_data.horizon_days,
        n_simulations=input_data.n_simulations,
        block_size=input_data.block_size,
        initial_capital=input_data.initial_capital,
        seed=resolved_seed,
    )

    return MonteCarloSimulationResult(
        tickers=input_data.tickers,
        horizon_days=input_data.horizon_days,
        n_simulations=input_data.n_simulations,
        random_seed=resolved_seed,
        terminal_median=round(result["terminal_median"], 2),
        terminal_p5=round(result["terminal_p5"], 2),
        terminal_p95=round(result["terminal_p95"], 2),
        prob_loss=round(result["prob_loss"], 4),
        terminal_var_95=round(result["terminal_var_95"], 6),
        terminal_cvar_95=round(result["terminal_cvar_95"], 6),
        equity_band_p5=[round(v, 2) for v in result["equity_band_p5"]],
        equity_band_p50=[round(v, 2) for v in result["equity_band_p50"]],
        equity_band_p95=[round(v, 2) for v in result["equity_band_p95"]],
    )


def run_backtest_compact(input_data: BacktestCompactInput) -> BacktestResultV2:
    """
    Compact counterpart to run_sma_backtest/run_rsi_backtest/run_macd_backtest/
    run_bollinger_backtest: instead of embedding the full equity_curve/
    trade_log inline (BacktestResult), saves them via backtest/artifacts.py
    and returns summary/risk/exposure/cost sub-reports plus URIs — closes
    the "agent tool result can contain the complete equity curve" gap for
    callers who opt into this shape. Reuses run_strategy and
    metrics/diagnostics.py's exposure_stats; no new backtest or metric math.
    """
    if input_data.strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )
    run_id = input_data.run_id or uuid.uuid4().hex
    logger.debug(
        "[backtest_compact] run_id=%s  %s  %s",
        run_id,
        input_data.symbol,
        input_data.strategy_type,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY[input_data.strategy_type](df, **input_data.parameters)

    results = run_strategy(
        df,
        signals,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
    )
    equity_curve = results["equity_curve"]
    returns = equity_curve.pct_change(fill_method=None).fillna(0.0)
    trade_log = results.get("trade_log", pd.DataFrame())
    executed = signals.shift(1).fillna(0.0)

    exposure = exposure_stats(executed, trade_log)

    pos_diff = executed.diff().fillna(executed.iloc[0])
    turnover = float(pos_diff.abs().sum())
    total_commission_pct = turnover * input_data.commission_pct
    total_slippage_pct = turnover * input_data.slippage_pct

    equity_curve_uri = save_artifact(equity_curve, run_id, "equity_curve")
    trades_uri = (
        save_artifact(trade_log, run_id, "trades") if not trade_log.empty else None
    )

    # The same bytes, addressed by KIND as well as by path. The raw URIs
    # above stay for existing callers and the MCP resource layer; these are
    # what every reference-taking tool in every runtime accepts, and what
    # lets a mismatched handoff fail by name instead of as a missing column.
    from standard_quant_tools.agent.runtimes import handoff

    equity_curve_ref = handoff.publish(
        equity_curve,
        "equity_curve",
        run_id,
        "equity_curve_ref",
        producer="backtest.run_backtest_compact",
        overwrite=True,
    )
    trades_ref = (
        handoff.publish(
            trade_log,
            "trade_log",
            run_id,
            "trades_ref",
            producer="backtest.run_backtest_compact",
            overwrite=True,
        )
        if not trade_log.empty
        else None
    )

    warnings: List[str] = []
    validation_status = "ok"
    if int(results["num_trades"]) < 5:
        warnings.append(
            f"Only {results['num_trades']} trade(s) — too few to draw reliable conclusions."
        )
        validation_status = "warning"

    logger.debug(
        "[backtest_compact] run_id=%s  trades=%d  sharpe=%.3f  status=%s",
        run_id,
        results["num_trades"],
        results["sharpe_ratio"],
        validation_status,
    )

    return BacktestResultV2(
        run_id=run_id,
        strategy_name=input_data.strategy_type,
        summary=PerformanceSummary(
            total_return=round(float(results["total_return"]), 6),
            annualized_return=round(float(cagr(equity_curve)), 6),
            annualized_volatility=round(float(results["annualized_volatility"]), 6),
            sharpe_ratio=round(float(results["sharpe_ratio"]), 4),
            sortino_ratio=round(float(results["sortino_ratio"]), 4),
            calmar_ratio=round(float(results["calmar_ratio"]), 4),
        ),
        risk=RiskSummary(
            max_drawdown=round(float(results["max_drawdown"]), 6),
            var_95=round(float(var_historical(returns, 0.95)), 6),
            cvar_95=round(float(cvar(returns, 0.95)), 6),
        ),
        exposure=ExposureSummary(**exposure),
        costs=CostSummary(
            total_commission_pct=round(total_commission_pct, 6),
            total_slippage_pct=round(total_slippage_pct, 6),
            total_cost_pct=round(total_commission_pct + total_slippage_pct, 6),
            num_trades=int(results["num_trades"]),
        ),
        equity_curve_uri=equity_curve_uri,
        trades_uri=trades_uri,
        equity_curve_ref=equity_curve_ref,
        trades_ref=trades_ref,
        warnings=warnings,
        validation_status=validation_status,
    )


def get_backtest_diagnostics(
    input_data: BacktestDiagnosticsInput,
) -> BacktestDiagnosticsResult:
    """
    Extended diagnostics for one of the library's built-in strategies:
    top drawdown episodes (with recovery time), trade expectancy/payoff/
    streaks with MAE/MFE, and exposure statistics — the detail Sharpe and
    total return alone don't surface. Reuses run_strategy for the backtest
    itself and the pure functions in metrics/diagnostics.py for everything
    else; no new backtest math.
    """
    logger.debug(
        "[backtest_diagnostics] %s  %s  %s → %s",
        input_data.symbol,
        input_data.strategy_type,
        input_data.start_date,
        input_data.end_date,
    )
    if input_data.strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY[input_data.strategy_type](df, **input_data.parameters)

    results = run_strategy(
        df,
        signals,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
        risk_free_rate=input_data.risk_free_rate,
    )
    equity_curve = results["equity_curve"]
    trade_log = results.get("trade_log", pd.DataFrame())
    # Same one-bar-lag convention run_strategy applies internally
    # (signals.shift(1)) — needed here to report the *held* position,
    # not the raw pre-lag signal.
    executed = signals.shift(1).fillna(0.0)

    top_dd = top_n_drawdowns(equity_curve, n=input_data.top_n_drawdowns)
    drawdown_episodes = [
        DrawdownEpisode(
            start=str(row["start"].date()),
            trough=str(row["trough"].date()),
            end=None if pd.isna(row["end"]) else str(row["end"].date()),
            depth=float(row["depth"]),
            duration_bars=int(row["duration_bars"]),
            recovery_bars=(
                None if pd.isna(row["recovery_bars"]) else int(row["recovery_bars"])
            ),
        )
        for _, row in top_dd.iterrows()
    ]

    expectancy = trade_expectancy(trade_log)
    excursions = trade_excursions(trade_log, df)
    avg_mae = float(excursions["mae_pct"].mean()) if not excursions.empty else 0.0
    avg_mfe = float(excursions["mfe_pct"].mean()) if not excursions.empty else 0.0

    exposure = exposure_stats(executed, trade_log)

    logger.debug(
        "[backtest_diagnostics] sharpe=%.3f  drawdowns=%d  expectancy=%.2f%%  time_in_market=%.0f%%",
        results["sharpe_ratio"],
        len(drawdown_episodes),
        expectancy["expectancy_pct"],
        exposure["time_in_market"] * 100,
    )

    return BacktestDiagnosticsResult(
        symbol=input_data.symbol,
        strategy_type=input_data.strategy_type,
        total_return=round(float(results["total_return"]), 6),
        sharpe_ratio=round(float(results["sharpe_ratio"]), 4),
        sortino_ratio=round(float(results["sortino_ratio"]), 4),
        max_drawdown=round(float(results["max_drawdown"]), 6),
        calmar_ratio=round(float(results["calmar_ratio"]), 4),
        num_trades=int(results["num_trades"]),
        top_drawdowns=drawdown_episodes,
        trade_diagnostics=TradeDiagnostics(
            expectancy_pct=expectancy["expectancy_pct"],
            avg_winner_pct=expectancy["avg_winner_pct"],
            avg_loser_pct=expectancy["avg_loser_pct"],
            payoff_ratio=expectancy["payoff_ratio"],
            max_consecutive_wins=expectancy["max_consecutive_wins"],
            max_consecutive_losses=expectancy["max_consecutive_losses"],
            avg_mae_pct=round(avg_mae, 4),
            avg_mfe_pct=round(avg_mfe, 4),
        ),
        exposure=ExposureDiagnostics(**exposure),
    )


def get_drawdown_table(input_data: DrawdownTableInput) -> DrawdownTableResult:
    """
    Every drawdown episode in a persisted equity curve, deepest first.

    `get_backtest_diagnostics` reports the top N episodes but re-runs the
    backtest to do it, from a symbol and a strategy rather than from the run
    that actually happened. That is slower and it is not necessarily the
    same run: a provider revision between the two calls diagnoses a curve
    nobody reported. This reads the artifact.

    `min_depth` exists because episode COUNT is dominated by one-bar noise
    on any long curve — a hundred 0.1% dips crowd out the four that mattered
    — while `n_episodes_total` still reports how many there were before
    filtering, so the cap is never silent.
    """
    frame = load_artifact(input_data.equity_curve_uri)
    equity = frame.squeeze("columns")
    if isinstance(equity, pd.DataFrame):
        raise ValidationError(
            f"{input_data.equity_curve_uri} has {len(frame.columns)} columns "
            f"({list(frame.columns)[:5]}); an equity curve is a single series. "
            "Pass a curve artifact, not a trade log or a panel."
        )
    if equity.empty:
        raise ValidationError(f"{input_data.equity_curve_uri} is empty")

    episodes_frame = drawdown_periods(equity)
    total = int(len(episodes_frame))

    kept = episodes_frame
    if input_data.min_depth > 0 and not kept.empty:
        kept = kept[kept["depth"].abs() >= input_data.min_depth]
    kept = kept.sort_values("depth").head(input_data.max_episodes)

    episodes = [
        DrawdownEpisode(
            start=str(row["start"]),
            trough=str(row["trough"]),
            end=(
                str(row["end"])
                if row["end"] is not None and not pd.isna(row["end"])
                else None
            ),
            depth=round(float(row["depth"]), 6),
            duration_bars=int(row["duration_bars"]),
            recovery_bars=(
                int(row["recovery_bars"])
                if row["recovery_bars"] is not None
                and not pd.isna(row["recovery_bars"])
                else None
            ),
        )
        for row in kept.to_dict(orient="records")
    ]

    dd = drawdown_series(equity)
    underwater_bars = int((dd < 0).sum())
    unrecovered = not episodes_frame.empty and pd.isna(
        episodes_frame.iloc[-1]["recovery_bars"]
    )

    logger.debug(
        "[drawdown_table] episodes=%d/%d underwater=%.1f%%",
        len(episodes),
        total,
        underwater_bars / len(equity) * 100 if len(equity) else 0.0,
    )
    return DrawdownTableResult(
        equity_curve_uri=input_data.equity_curve_uri,
        n_bars=int(len(equity)),
        n_episodes_total=total,
        n_episodes_returned=len(episodes),
        max_drawdown=round(float(max_drawdown(equity)), 6),
        episodes=episodes,
        currently_underwater=bool(unrecovered),
        time_underwater_pct=(
            round(underwater_bars / len(equity), 6) if len(equity) else 0.0
        ),
    )


#: Bisection bounds and tolerance for the breakeven commission solve. The
#: upper bound is a 100% commission, which no real venue charges -- it is
#: there so a strategy that survives anything reports "no crossing" rather
#: than a number pulled from the edge of the bracket.
_BREAKEVEN_HI = 1.0
_BREAKEVEN_TOL = 1e-6
_BREAKEVEN_MAX_ITER = 60


def compare_cost_models(
    input_data: CompareCostModelsInput,
) -> CompareCostModelsResult:
    """
    Run one strategy under several cost assumptions and report what each
    costs — plus the commission rate at which the edge disappears.

    The signal series is computed ONCE and priced repeatedly. That is not
    only an efficiency: it is what makes the comparison mean anything.
    Costs do not feed back into the signal (signals come from prices, never
    from equity), so every scenario trades on identical dates and the
    differences between rows are cost and nothing else. Running the same
    comparison as N separate backtest calls re-derives the same signal N
    times and invites a parameter to drift between them.

    That same independence makes total return strictly decreasing in the
    commission rate, so `breakeven_commission_pct` is found by BISECTION
    rather than a search — there is exactly one crossing when one exists.
    None is returned when the strategy loses money at zero cost (nothing to
    break even from) or still profits at a 100% commission (no crossing in
    the bracket).
    """
    if input_data.strategy_type not in STRATEGY_REGISTRY:
        raise ValidationError(
            f"Unknown strategy_type {input_data.strategy_type!r}. Available: "
            f"{sorted(STRATEGY_REGISTRY)}. Call list_strategies for each "
            "one's parameters."
        )

    logger.debug(
        "[compare_cost_models] %s %s scenarios=%d",
        input_data.symbol,
        input_data.strategy_type,
        len(input_data.scenarios),
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY[input_data.strategy_type](df, **input_data.parameters)

    def _run(commission_pct: float, slippage_pct: float) -> Dict[str, Any]:
        return run_strategy(
            df,
            signals,
            input_data.initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            include_trade_log=True,
            fill_price=input_data.fill_price,
            risk_free_rate=input_data.risk_free_rate,
        )

    gross = _run(0.0, 0.0)
    gross_return = float(gross["total_return"])

    rows: List[CostScenarioResult] = []
    for scenario in input_data.scenarios:
        result = _run(scenario.commission_pct, scenario.slippage_pct)
        rows.append(
            CostScenarioResult(
                label=scenario.label,
                commission_pct=scenario.commission_pct,
                slippage_pct=scenario.slippage_pct,
                total_return=round(float(result["total_return"]), 6),
                annualized_return=round(float(cagr(result["equity_curve"])), 6),
                sharpe_ratio=round(float(result["sharpe_ratio"]), 4),
                max_drawdown=round(float(result["max_drawdown"]), 6),
                n_trades=int(result["num_trades"]),
                cost_drag_vs_gross=round(
                    float(result["total_return"]) - gross_return, 6
                ),
            )
        )

    notes: List[str] = []
    breakeven: Optional[float] = None
    if input_data.solve_breakeven:
        slippage = input_data.scenarios[0].slippage_pct
        if gross_return <= 0:
            notes.append(
                "No breakeven commission: the strategy loses money before "
                "any costs are charged, so there is no edge for costs to "
                "consume. Fix the strategy, not the cost assumption."
            )
        elif float(_run(_BREAKEVEN_HI, slippage)["total_return"]) > 0:
            notes.append(
                "No breakeven commission inside the bracket: total return is "
                "still positive at a 100% commission. That normally means the "
                "strategy barely trades — check n_trades before reading this "
                "as robustness."
            )
        else:
            lo, hi = 0.0, _BREAKEVEN_HI
            for _ in range(_BREAKEVEN_MAX_ITER):
                if hi - lo < _BREAKEVEN_TOL:
                    break
                mid = (lo + hi) / 2.0
                if float(_run(mid, slippage)["total_return"]) > 0:
                    lo = mid
                else:
                    hi = mid
            breakeven = round((lo + hi) / 2.0, 6)
            notes.append(
                f"Breakeven solved at slippage_pct={slippage} (the first "
                "scenario's). A different slippage moves this number."
            )

    survives = all(row.total_return > 0 for row in rows)
    if not survives:
        losing = [row.label for row in rows if row.total_return <= 0]
        notes.append(f"Scenarios that lose money: {losing}.")

    return CompareCostModelsResult(
        symbol=input_data.symbol,
        strategy_type=input_data.strategy_type,
        n_bars=int(len(df)),
        gross_total_return=round(gross_return, 6),
        gross_sharpe_ratio=round(float(gross["sharpe_ratio"]), 4),
        scenarios=rows,
        breakeven_commission_pct=breakeven,
        survives_all_scenarios=survives,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────
# Strategy matrix — every strategy against every ticker, in one call
# ──────────────────────────────────────────────────────────────────


def run_strategy_matrix(input_data: StrategyMatrixInput) -> StrategyMatrixResult:
    """
    Every requested strategy against every requested ticker, ranked.

    The comparison an agent otherwise assembles from N x M separate calls,
    each with its own fetch. This fetches once per ticker and reuses the
    bars across every strategy, so the cost is one fetch per name rather
    than one per cell — and, more usefully, every cell is priced on exactly
    the same bars, which N separate calls cannot promise once a provider
    revises anything between them.

    A cell that cannot run is reported in `failures` rather than dropped. A
    silently missing cell reads as a strategy that was tested and lost,
    which is the opposite of what happened.
    """
    logger.debug(
        "[strategy_matrix] %d tickers x %d strategies",
        len(input_data.tickers),
        len(input_data.strategies),
    )
    unknown = sorted(set(input_data.strategies) - set(STRATEGY_REGISTRY))
    if unknown:
        raise ValidationError(
            f"unknown strategies {unknown}. Available: "
            f"{sorted(STRATEGY_REGISTRY)}. Call list_strategies for each "
            "one's parameters."
        )

    provider = DataFactory.get_provider()
    cells: List[MatrixCell] = []
    failures: Dict[str, str] = {}

    for ticker in input_data.tickers:
        try:
            bars = provider.get_ohlcv(
                ticker, input_data.start_date, input_data.end_date
            )
        except Exception as exc:
            for strategy in input_data.strategies:
                failures[f"{ticker}/{strategy}"] = f"no data: {exc}"
            continue
        if bars is None or bars.empty:
            for strategy in input_data.strategies:
                failures[f"{ticker}/{strategy}"] = "no bars returned"
            continue

        for strategy in input_data.strategies:
            parameters = input_data.parameters.get(strategy, {})
            try:
                signals = STRATEGY_REGISTRY[strategy](bars, **parameters)
                result = run_strategy(
                    bars,
                    signals,
                    input_data.initial_capital,
                    commission_pct=input_data.commission_pct,
                    slippage_pct=input_data.slippage_pct,
                    fill_price=input_data.fill_price,
                    risk_free_rate=input_data.risk_free_rate,
                )
            except Exception as exc:
                failures[f"{ticker}/{strategy}"] = str(exc)
                continue
            cells.append(
                MatrixCell(
                    ticker=ticker,
                    strategy=strategy,
                    total_return=round(float(result["total_return"]), 6),
                    sharpe_ratio=round(float(result["sharpe_ratio"]), 4),
                    max_drawdown=round(float(result["max_drawdown"]), 6),
                    num_trades=int(result["num_trades"]),
                    win_rate=round(float(result["win_rate"]), 4),
                )
            )

    if not cells:
        raise ValidationError(
            f"no cell in the matrix could be evaluated. Failures: {failures}"
        )

    key = input_data.sort_by
    if not hasattr(cells[0], key):
        raise ValidationError(
            f"sort_by={key!r} is not a reported metric; expected one of "
            f"{sorted(MatrixCell.model_fields)}"
        )
    cells.sort(key=lambda c: getattr(c, key), reverse=True)

    best_per_ticker: Dict[str, str] = {}
    for cell in cells:
        best_per_ticker.setdefault(cell.ticker, cell.strategy)

    notes: List[str] = []
    if failures:
        notes.append(
            f"{len(failures)} of {len(input_data.tickers) * len(input_data.strategies)} "
            "cell(s) could not be evaluated and are listed in `failures`, "
            "not omitted."
        )
    thin = [c for c in cells if c.num_trades < 5]
    if thin:
        notes.append(
            f"{len(thin)} cell(s) traded fewer than 5 times. Their ranking "
            "is noise — a Sharpe from three trades orders the table without "
            "meaning anything."
        )

    return StrategyMatrixResult(
        tickers=input_data.tickers,
        strategies=input_data.strategies,
        n_backtests=len(cells),
        cells=cells,
        best_overall=cells[0],
        best_per_ticker=best_per_ticker,
        failures=failures,
        notes=notes,
    )
