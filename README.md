# Standard Quant Tools for AI Agents

A high-performance, modular Python library for quantitative financial analysis. Designed to give AI agents and automated workflows **clean structured data**, **mathematical accuracy**, and **robust error handling**.

## Key Features

- **High Performance** — Optional C++ extension (`_sqt_core`) for Hurst/rolling Hurst (20–80×), RSI/ADX/Parabolic SAR (10–30×), Wilder's ATR (4–8×), Engle-Granger cointegration (5–15×), 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread` — 10–20×), backtest kernel (`run_strategy` — 3–8×), `batch_run_strategy` grid kernel (10–50×), `rolling_factor_loadings` incremental Cholesky (50–200×), `rolling_beta` incremental sums (10–40×), `bollinger_bands` fused mean+std (3–8×), `stochastic_oscillator` fused min+max (5–15×); NumPy single-pass ATR (5.6×); BLAS-backed portfolio covariance; async concurrent data fetching; persistent Parquet disk cache; `ProcessPoolExecutor` screener and parallel backtest grid
- **Agent-First Design** — All tools return Pydantic models; 34 LLM-callable tools with OpenAI/Anthropic function-calling schemas, including two bring-your-own-signal tools; descriptive errors for self-correction
- **Comprehensive Coverage** — 14 indicators, 13 risk/return metrics + 5 backtest diagnostics, 12 analysis functions, portfolio analysis, stock screener, 4 backtest strategies + parameter grid search, a shared-cash portfolio simulation engine with pluggable cost/constraint models, pairs backtest, and walk-forward/robustness diagnostics — grid search and the signal-panel backtester also accept your own signal-generating callable/matrix, not just the built-in strategies
- **Robust Infrastructure** — Retry logic with exponential backoff, TTL + Parquet caching, custom exception hierarchy, `@validate_series` decorator, decision-record audit trail (`sqt` CLI), optional C++/scipy/numba graceful fallback

---

## Installation

```bash
pip install .
# or
poetry install
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `yfinance`, `numba`, `aiohttp`, `cachetools`, `pydantic`, `statsmodels`, `scikit-learn`, `plotly`, `pyarrow`

---

## Quick Start

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators import sma, rsi, bollinger_bands, adx
from standard_quant_tools.metrics import sharpe_ratio, max_drawdown, calmar_ratio, var_historical

# Fetch data
provider = DataFactory.get_provider()
df = provider.get_ohlcv("NVDA", "2023-01-01", "2024-01-01")

# Technical analysis
df['RSI'] = rsi(df['Close'], 14)
df['SMA_50'] = sma(df['Close'], 50)
bb = bollinger_bands(df['Close'], 20, 2.0)
adx_df = adx(df['High'], df['Low'], df['Close'])

# Risk metrics
returns = df['Close'].pct_change().dropna()
equity = (1 + returns).cumprod() * 10_000
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max Drawdown: {max_drawdown(equity):.2%}")
print(f"VaR(95%): {var_historical(returns, 0.95):.4f}")
```

---

## Module Reference

### Data (`standard_quant_tools.data`)

| Function | Description | Returns |
|---|---|---|
| `get_ohlcv(symbol, start, end, interval)` | Historical OHLCV data | `pd.DataFrame` |
| `get_ohlcv_async(...)` | Non-blocking OHLCV fetch | `Awaitable[pd.DataFrame]` |
| `get_ticker_info(symbol)` | Company metadata | `TickerInfo` (Pydantic) |
| `get_financial_ratios(symbol)` | P/E, P/B, D/E, ROE, margins, etc. | `FinancialRatios` (Pydantic) |
| `get_metadata(symbol, interval)` | Dataset provenance: adjusted, survivorship-free, point-in-time, timezone | `DataSetMetadata` (Pydantic) |

**Caching:** Historical OHLCV calls are saved as Parquet files under `~/.cache/standard_quant_tools/ohlcv/`. Subsequent calls — even from a new Python process — load from disk rather than the network. Override the cache directory with `SQT_CACHE_DIR`.

**Data quality (`standard_quant_tools.data.quality`):** `detect_missing_bars`, `detect_stale_prices`, `detect_price_jumps` — heuristic checks on an already-fetched OHLCV frame (weekday gaps, frozen prices, large single-bar jumps). `detect_missing_bars` has no market-holiday calendar, so U.S. holidays show up as false-positive gaps — treat findings as leads to investigate, not confirmed defects. Exposed together with `get_metadata` via the `get_data_quality_report` agent tool.

---

### Technical Indicators (`standard_quant_tools.indicators`)

**Trend**

| Function | Description | Performance |
|---|---|---|
| `sma(series, period)` | Simple Moving Average | Pandas rolling |
| `ema(series, period)` | Exponential Moving Average | Pandas EWM |
| `macd(series, fast, slow, signal)` | MACD + Signal + Histogram | Pandas EWM |
| `adx(high, low, close, period)` | ADX + DI+ + DI− | **C++ extension** / Numba JIT / Python fallback |
| `parabolic_sar(high, low)` | Parabolic SAR + Trend direction | **C++ extension** / Numba JIT / Python fallback |
| `williams_r(high, low, close, period)` | Williams %R oscillator | Pandas rolling |

**Momentum**

| Function | Description | Performance |
|---|---|---|
| `rsi(series, period)` | RSI (Wilder's smoothing) | **C++ extension** / Numba JIT / Python fallback |
| `stochastic_oscillator(high, low, close)` | Stochastic %K and %D | **C++ extension** / Pandas rolling fallback |

**Volatility**

| Function | Description | Performance |
|---|---|---|
| `bollinger_bands(series, period, num_std)` | Upper / Middle / Lower bands | **C++ extension** / Pandas rolling fallback |
| `atr(high, low, close, period)` | Average True Range (SMA of TR) | **NumPy single-pass** (5.6× vs `pd.concat`) |
| `wilder_atr(high, low, close, period)` | Wilder's ATR (SMA seed + Wilder smoothing) | **C++ extension** / Python fallback |

**Volume**

| Function | Description |
|---|---|
| `obv(close, volume)` | On Balance Volume |
| `vwap(high, low, close, volume, period)` | VWAP (cumulative or rolling) |
| `mfi(high, low, close, volume, period)` | Money Flow Index |

---

### Metrics (`standard_quant_tools.metrics`)

**Return Metrics**

| Function | Description |
|---|---|
| `cumulative_return(series)` | Total return over period |
| `cagr(series)` | Compound Annual Growth Rate |
| `annualized_volatility(returns)` | Annualized standard deviation |

**Risk Metrics**

| Function | Description |
|---|---|
| `sharpe_ratio(returns, risk_free_rate)` | Excess return per unit of total risk |
| `sortino_ratio(returns, risk_free_rate)` | Excess return per unit of downside risk |
| `max_drawdown(series)` | Maximum peak-to-trough decline |
| `calmar_ratio(equity_curve)` | CAGR / \|max drawdown\| |
| `var_historical(returns, confidence)` | Historical Value at Risk |
| `var_parametric(returns, confidence)` | Gaussian VaR (scipy optional) |
| `cvar(returns, confidence)` | Conditional VaR / Expected Shortfall |
| `information_ratio(returns, benchmark_returns)` | Active return / tracking error |
| `treynor_ratio(returns, benchmark_returns)` | Excess return / beta |
| `drawdown_series(series)` | Full drawdown time series |

**Backtest Diagnostics**

| Function | Description |
|---|---|
| `drawdown_periods(equity_curve)` | One row per drawdown episode: peak, trough, recovery, depth, duration |
| `top_n_drawdowns(equity_curve, n)` | The n deepest drawdown episodes, worst first |
| `trade_expectancy(trade_log)` | Expectancy, avg winner/loser, payoff ratio, consecutive win/loss streaks |
| `trade_excursions(trade_log, price_data)` | Adds MAE/MFE (max adverse/favorable excursion) columns to a trade log |
| `exposure_stats(executed_signal, trade_log)` | Time in market, gross/net exposure, % long/short, avg holding period |

Computed entirely from data a backtest already produces (`equity_curve`, `trade_log`, the OHLCV frame) — no engine changes required. Exposed together via the `get_backtest_diagnostics` agent tool.

---

### Analysis (`standard_quant_tools.analysis`)

12 functions across five areas. Several functions have a **C++ fast path** via `_sqt_core`:
- `calculate_beta` — 2-variable OLS via closed-form normal equations (10–20× vs. `np.linalg.lstsq`)
- `rolling_beta` — incremental O(1)-per-bar sum updates (10–40× vs. two pandas rolling passes)
- `half_life` / `compute_spread` — same OLS kernel, same speedup
- `cointegration_test` — full Engle-Granger pipeline (5–15× vs. statsmodels)
- `hurst_exponent` / `rolling_hurst` — DFA + R/S + sliding window (20–80× / 30–100×)
- `rolling_factor_loadings` — incremental Cholesky rank-1 updates (50–200× vs. per-window `lstsq`)

#### Regression

```python
from standard_quant_tools.analysis import calculate_beta, rolling_beta

stats = calculate_beta(asset_returns, benchmark_returns)
# {'alpha': 0.0003, 'beta': 1.12, 'r_squared': 0.78}

rolling_df = rolling_beta(asset_returns, benchmark_returns, window=60)
# DataFrame with 'Rolling_Beta' column
```

#### Multi-Factor Regression

```python
from standard_quant_tools.analysis import multi_factor_regression, rolling_factor_loadings
import pandas as pd

factors = pd.DataFrame({"mkt": spy_rets, "smb": smb_rets, "hml": hml_rets})
result = multi_factor_regression(asset_returns, factors)

print(result["loadings"])          # {"mkt": 1.12, "smb": -0.3, "hml": 0.05}
print(result["t_stats"])           # includes alpha and all factor names
print(result["r_squared"])         # 0.78
print(result["adj_r_squared"])     # 0.77

rolling = rolling_factor_loadings(asset_returns, factors, window=60)
# DataFrame (dates × ["alpha", "mkt", "smb", "hml"])
```

#### Cointegration & Pairs Spread

```python
from standard_quant_tools.analysis import (
    cointegration_test, compute_spread, half_life, spread_zscore
)

result = cointegration_test(ko_prices, pep_prices)
# {'cointegrated': True, 'hedge_ratio': 0.83, 'p_value': 0.003,
#  'half_life_days': 14.2, 'critical_values': {'1%': -3.92, '5%': -3.35, '10%': -3.05}}

spread = compute_spread(ko_prices, pep_prices)           # pd.Series
hl     = half_life(spread)                               # 14.2 (bars)
z      = spread_zscore(spread, window=30)                # rolling z-score signal
```

#### PCA on Returns

```python
from standard_quant_tools.analysis import pca_returns, factor_contributions

result = pca_returns(returns_df, n_components=3)
print(result["explained_variance_ratio"])  # PC1: 0.42, PC2: 0.12, PC3: 0.08
print(result["loadings"])                  # (assets × 3) — eigenvector matrix
print(result["factor_returns"])            # (dates × 3) — orthogonal PC time series

contrib = factor_contributions(returns_df, n_components=3)
# DataFrame (assets × PCs): marginal R² each PC contributes per asset
```

#### Hurst Exponent

```python
from standard_quant_tools.analysis import hurst_exponent, rolling_hurst
from standard_quant_tools.analysis.hurst import HAS_CPP

print("C++ backend active:", HAS_CPP)     # True once _sqt_core is built

result = hurst_exponent(returns)          # pass RETURNS not prices
# {'hurst': 0.38, 'regime': 'mean_reverting', 'fit_r_squared': 0.97, 'method': 'dfa'}

rolling = rolling_hurst(returns, window=252, step=5)   # pd.Series of H values
```

| H value | Regime | Implication |
|---|---|---|
| > 0.55 | trending | Momentum strategies have an edge |
| 0.45 – 0.55 | random walk | No persistent signal from past prices |
| < 0.45 | mean-reverting | Contrarian / mean-reversion strategies |

> The C++ extension accelerates `hurst_exponent` by 20–80× and `rolling_hurst` by 30–100×. The API is identical with or without it — pure Python fallback is automatic. See [Development/build_guide.md](Development/build_guide.md).

---

### Backtesting (`standard_quant_tools.backtest`)

Vectorized engine with transaction costs, trade log, and full metric output.

```python
from standard_quant_tools.backtest.engine import run_strategy

signals = (df['Close'] > sma(df['Close'], 50)).astype(int)

result = run_strategy(
    df, signals,
    initial_capital=10_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    include_trade_log=True,
)
print(f"Sharpe: {result['sharpe_ratio']:.2f}  |  Win Rate: {result['win_rate']:.1%}  |  Trades: {result['num_trades']}")
```

**Output keys:** `final_equity`, `total_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `num_trades`, `avg_trade_return_pct`, `equity_curve`, `trade_log`

#### Parameter Grid Search

```python
from standard_quant_tools.backtest import backtest_grid

results = backtest_grid(
    price_data=df,
    strategy="sma_crossover",          # or rsi_mean_reversion / macd_crossover / bollinger_reversion
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    sort_by="sharpe_ratio",
    n_workers=4,                        # parallel ProcessPoolExecutor
)
print(results.head())   # 9 combinations ranked by Sharpe
```

`strategy` also accepts your own signal-generating callable — grid search, C++ speed, and `sort_by` ranking all work identically on your own alpha logic, not just the built-ins. For a pre-computed signal matrix across a ticker universe, see `run_signal_panel_backtest` in [Documentation/04_backtesting.md](Documentation/04_backtesting.md#grid-searching-your-own-signal).

#### Portfolio Simulation Engine

`run_signal_panel_backtest` gives every ticker its own independent capital and blends the return streams afterward. `run_portfolio_simulation` (`standard_quant_tools.backtest.portfolio_engine`) is the true-portfolio counterpart: one shared cash balance, position sizing relative to current account equity, and rebalancing at specific dates — weights drift between rebalances instead of being re-applied every bar.

```python
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation

result = run_portfolio_simulation(
    price_data={"AAPL": aapl_df, "MSFT": msft_df},
    target_weights=weights_df,           # DataFrame indexed by rebalance date, one column per ticker
    initial_capital=100_000,
    max_gross_leverage=1.0,
    max_position_pct=0.25,
    commission_model="pct",              # or "per_share"
    use_impact_model=True,
    max_adv_participation=0.1,
)
print(result['final_equity'], result['leverage_curve'].mean())
```

Pluggable building blocks compose into `run_portfolio_simulation` (or can be used standalone):

- `standard_quant_tools.backtest.costs` — `percentage_commission`, `per_share_commission`, `fixed_bps_spread`, `pct_of_range_spread`, `sqrt_impact_bps`, `impact_cost`, `short_borrow_cost`, `margin_interest`
- `standard_quant_tools.backtest.constraints` — `adv_participation`, `days_to_liquidate`, `sector_exposure`, `capacity_report`
- `standard_quant_tools.backtest.sizing` — `rank_weighted`, `equal_weight_top_bottom`, `zscore_normalized`, `vol_scaled`, `dollar_neutral` — turns a SCORE signal panel into a target-weight panel
- `standard_quant_tools.backtest.pairs` — `run_pair_backtest`, a two-leg pair trade as one dollar-neutral portfolio (reuses `run_portfolio_simulation`)
- `standard_quant_tools.backtest.artifacts` — `save_artifact` / `load_artifact`, a local Parquet store for equity curves/trade logs too large to embed inline in an agent-tool response

#### Robustness Diagnostics

`standard_quant_tools.backtest.robustness` answers "is this backtest result trustworthy, or a fluke of one sample path / one lucky parameter combination": `block_bootstrap_ci` (confidence interval on a point-estimate metric), `parameter_sensitivity` (best-vs-median gap on a grid search), and `deflated_sharpe_ratio` (corrects the best observed Sharpe for having been selected as the max of `n_trials` attempts). Complementary to, not a substitute for, out-of-sample walk-forward validation.

---

### Portfolio (`standard_quant_tools.portfolio`)

```python
from standard_quant_tools.portfolio import portfolio_metrics, correlation_matrix, fetch_returns_sync

returns_df = fetch_returns_sync(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-01-01')
weights = [0.4, 0.35, 0.25]

metrics = portfolio_metrics(returns_df, weights)
print(f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Portfolio VaR(95%): {metrics['var_95']:.4f}")

corr = correlation_matrix(returns_df)
```

---

### Screener (`standard_quant_tools.screener`)

```python
from standard_quant_tools.screener import screen_stocks

result = screen_stocks(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    filters={
        'pe_ratio_max': 35,
        'rsi_max': 50,
        'price_above_sma': 50,
        'beta_max': 1.5,
    },
    sort_by='rsi_14',
    ascending=True,
)
```

**Available filters:** `pe_ratio_max`, `pb_ratio_max`, `debt_equity_max`, `roe_min`, `profit_margin_min`, `div_yield_min`, `market_cap_min`, `rsi_max`, `rsi_min`, `price_above_sma`, `price_below_sma`, `beta_max`, `beta_min`

**Large universes:** Pass `n_workers` to split screening across CPU cores. ≤ 20 tickers run in a single async event loop; larger universes automatically use `ProcessPoolExecutor`.

**Beta filter optimisation:** When `beta_max` / `beta_min` filters are active, SPY data is pre-fetched once per batch instead of once per ticker — a single HTTP round-trip for the whole universe when `n_workers <= 1` (the default for ≤ 20 tickers), or once per worker process for larger multi-worker universes (still eliminating the N−1 redundant per-ticker fetches within each worker's batch).

```python
result = screen_stocks(sp500_tickers, filters={...}, n_workers=8)
```

---

### AI Agent Tools (`standard_quant_tools.agent`)

34 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas — including two tools that backtest a signal you computed yourself rather than one of the built-in indicator strategies.

For a single agent choosing among all 34, see `Implementation/`. For an **orchestrator-workers** architecture — a lead agent that delegates to six specialist sub-agents, each scoped to a small, non-overlapping tool subset — see `Multi_Agent_Implementation/` (Anthropic only for now). Splitting tools this way is a direct fix for tool-selection confusion between similar tools (e.g. a built-in strategy backtest vs. a bring-your-own-signal backtest): a worker that was never given the other tool cannot call it by mistake.

```python
from standard_quant_tools.agent.tools import (
    get_agent_tools, analyze_stock_risk,
    run_factor_regression, run_cointegration_test,
    run_pca_analysis, run_hurst_analysis,
    get_stock_fundamentals, run_backtest_optimization,
    get_advanced_indicators, get_rolling_beta,
    get_extended_risk_metrics,
    run_custom_signal_backtest, run_signal_panel_backtest,
    get_backtest_diagnostics,
)
from standard_quant_tools.agent.models import (
    AnalysisInput, FactorRegressionInput,
    CointegrationInput, PCAInput, HurstInput,
    FundamentalsInput, BacktestOptInput,
    AdvancedIndicatorsInput, RollingBetaInput, ExtendedRiskInput,
    CustomSignalBacktestInput, SignalPanelBacktestInput,
    BacktestDiagnosticsInput,
)

# Get tool schemas for your LLM
tools = get_agent_tools()  # 34 tools ready for function calling

# Risk analysis
result = analyze_stock_risk(AnalysisInput(symbol='NVDA', benchmark='SPY', period='1y'))
print(result.model_dump_json(indent=2))

# Multi-factor regression (SPY/IWM/IWD as mkt/smb/hml proxies)
result = run_factor_regression(FactorRegressionInput(
    symbol='AAPL',
    factor_tickers=['SPY', 'IWM', 'IWD'],
    factor_names=['mkt', 'smb', 'hml'],
    start_date='2022-01-01',
    end_date='2024-01-01',
    rolling_window=60,
))

# Pairs cointegration + live z-score signal
result = run_cointegration_test(CointegrationInput(
    symbol_a='KO', symbol_b='PEP',
    start_date='2022-01-01', end_date='2024-01-01',
    zscore_window=30,
))
print(result.signal)  # "long_a_short_b" | "short_a_long_b" | "neutral"

# PCA on a basket of stocks
result = run_pca_analysis(PCAInput(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
    start_date='2022-01-01', end_date='2024-01-01',
    n_components=3,
))

# Hurst exponent with rolling regime breakdown
result = run_hurst_analysis(HurstInput(
    symbol='SPY', start_date='2022-01-01', end_date='2024-01-01',
    method='dfa', rolling_window=252,
))
print(result.regime)   # "trending" | "random_walk" | "mean_reverting"
```

**Core backtest & analysis tools (14):** `run_sma_backtest`, `run_rsi_backtest`, `run_macd_backtest`, `run_bollinger_backtest`, `run_buy_and_hold`, `compare_strategies`, `analyze_stock_risk`, `get_technical_analysis`, `get_portfolio_analysis`, `run_screener`, `run_factor_regression`, `run_cointegration_test`, `run_pca_analysis`, `run_hurst_analysis`

**Advanced agentic tools (7):** `run_regime_adaptive_backtest`, `run_regime_adaptive_walkforward_backtest`, `scan_pairs`, `run_walk_forward_backtest`, `get_portfolio_risk_attribution`, `get_position_size`, `run_portfolio_simulation`

**Supplementary tools (6):** `get_stock_fundamentals`, `run_backtest_optimization`, `get_advanced_indicators`, `get_rolling_beta`, `get_extended_risk_metrics`, `get_backtest_diagnostics`

**Custom signal tools (2):** `run_custom_signal_backtest`, `run_signal_panel_backtest`

**Diagnostics, capacity & specialized backtests (5):** `run_pair_trade_backtest`, `get_robustness_diagnostics`, `get_capacity_report`, `get_data_quality_report`, `run_backtest_compact`

---

## Performance

### C++ Extension (`_sqt_core`)

The optional compiled C++ extension accelerates the highest-impact CPU-bound paths. The API is identical with or without it — pure Python fallback is automatic.

| Operation | Python fallback | C++ (`_sqt_core`) | Speedup |
|---|---|---|---|
| `hurst_exponent` single call (n = 500) | ~5–15 ms | ~0.1–0.5 ms | **20–80×** |
| `hurst_exponent` single call (n = 2 000) | ~25–80 ms | ~0.5–2 ms | **20–80×** |
| `rolling_hurst` (n = 2 000, window = 200, step = 1) | ~5–15 s | ~0.1–0.3 s | **30–100×** |
| `rolling_hurst` (n = 2 000, window = 252, step = 5) | ~1–3 s | ~0.05–0.15 s | **20–60×** |
| `run_regime_adaptive_backtest` (end-to-end) | ~10–20 s | ~0.5–2 s | **10–30×** |
| `rsi` (n = 2 000, period = 14) | ~0.5–2 ms | ~0.02–0.1 ms | **10–30×** |
| `adx` (n = 2 000, period = 14) | ~1–4 ms | ~0.05–0.2 ms | **10–30×** |
| `parabolic_sar` (n = 2 000) | ~0.5–2 ms | ~0.02–0.1 ms | **10–30×** |
| `wilder_atr` (n = 2 000, period = 14) | ~0.5–2 ms | ~0.05–0.2 ms | **4–8×** |
| `cointegration_test` (n = 500) | ~5–20 ms (statsmodels) | ~0.3–2 ms | **5–15×** |
| `calculate_beta` (n = 500) | ~0.3–0.8 ms (`lstsq`) | ~0.01–0.03 ms | **10–20×** |
| `half_life` (n = 500) | ~0.2–0.5 ms (`lstsq`) | ~0.008–0.02 ms | **10–20×** |
| `run_strategy` (n = 2 000) | ~1–3 ms (pandas) | ~0.1–0.4 ms | **3–8×** |
| `backtest_grid` (100 combos, batch kernel) | ~100–300 ms | ~5–20 ms | **10–50×** |
| `rolling_factor_loadings` (n = 500, window = 60, k = 3) | ~50–200 ms (lstsq loop) | ~0.5–3 ms | **50–200×** |
| `rolling_beta` (n = 2 000, window = 60) | ~1–3 ms (2× rolling) | ~0.05–0.2 ms | **10–40×** |
| `bollinger_bands` (n = 2 000, period = 20) | ~0.5–1.5 ms (2× rolling) | ~0.1–0.4 ms | **3–8×** |
| `stochastic_oscillator` (n = 2 000, k = 14) | ~0.6–1.8 ms (2× rolling) | ~0.1–0.3 ms | **5–15×** |

The rolling Hurst gain is the most significant: rather than re-entering Python for every bar, the entire sliding-window pass runs in one C++ function. RSI/ADX/PSAR gains are most visible when numba is unavailable (e.g. NumPy 2.x), where the alternative is an interpreted Python loop. `rolling_factor_loadings` achieves its dramatic speedup through incremental rank-1 XtX updates — each new bar costs O(k²) instead of a full O(n·k²) `lstsq` solve.

> These are projected figures based on algorithmic analysis of loop iterations vs. compiled throughput. The benchmark suite (`tests/cpp/bench_hurst.cpp` and `pytest -m benchmark`) confirms actual numbers once the extension is built. See [Development/build_guide.md](Development/build_guide.md) for build instructions.

---

### Python-Level Optimisations

Confirmed benchmarks on a 2 000-bar series (Python 3.12, NumPy 2.4):

| Optimisation | Before | After | Speedup | Notes |
|---|---|---|---|---|
| ATR true range | 2.8 ms (`pd.concat` + `.max`) | 0.49 ms (`np.maximum`) | **5.6×** | Single-pass; eliminates 3 Series + concat |
| Trade log serialization | 31 ms (`iterrows`, 500 trades) | 3.6 ms (`to_dict`) | **~9×** | Vectorized dict conversion |
| CVaR computation | 0.83 ms (two-pass) | 0.44 ms (one-pass) | **1.9×** | Single `np.percentile` + boolean mask |
| SPY beta screen | N HTTP requests | 1 request per worker | **~N/workers×** | SPY pre-fetched once per batch — 1 total for single-process runs, once per worker for `n_workers > 1` |
| Backtesting equity curve | — | NumPy cumprod | vectorized | `(1 + returns).cumprod()` |
| Portfolio covariance | — | BLAS `pandas.cov` | BLAS-backed | O(n·k²) via LAPACK |
| Screener (50+ tickers) | — | ProcessPoolExecutor | multi-core | Auto async→multiprocess threshold |

> **Numba note:** RSI, ADX, Parabolic SAR, and the RSI/Bollinger strategy state machines are decorated with `@njit` for ~50–100× speedup on their inner loops. This requires Numba with a compatible NumPy version (≤ 2.0). On NumPy 2.x (current default), Numba decorators are a no-op — but the C++ extension (`_sqt_core`) provides equivalent performance for RSI, ADX, and PSAR without any Numba dependency. All three fall back to pure Python automatically when neither C++ nor Numba is available.

---

## Error Handling

```python
from standard_quant_tools.error import DataNotFoundError, InvalidSymbolError, ValidationError

try:
    df = provider.get_ohlcv("INVALID", "2023-01-01", "2024-01-01")
except DataNotFoundError as e:
    print(f"No data: {e}")
except InvalidSymbolError as e:
    print(f"Bad symbol: {e}")
```

**Exception hierarchy:** `QuantError` → `DataProviderError` → `DataNotFoundError / InvalidSymbolError / APIError`

---

## Audit Trail & CLI (`standard_quant_tools.audit`, `sqt`)

Every call routed through `agent.tools.dispatch()` can produce an immutable JSONL decision record capturing its inputs, the market data it pulled (with content hashes), which execution path ran, and a hash of its output — enough to tell a stale/tampered cache apart from a genuine code change. Nothing runs automatically; set `SQT_AUDIT_ENABLED=0` to disable record writes, and override the storage directory with `SQT_AUDIT_DIR`.

The `sqt` command (installed with the package) inspects and verifies these records by `request_id`:

```bash
sqt replay <request_id>              # re-run the recorded call, report whether data/output still match
sqt compare <request_id_a> <id_b>    # diff two records' status/output/timing/provenance/inputs
sqt report <request_id>              # pretty-print one record in full
```

`sqt replay` exits 0 if the output reproduced exactly, 1 on a confirmed mismatch, 2 if the record has no output hash to compare against. See [Documentation/10_auditability.md](Documentation/10_auditability.md).

---

## Running Tests

```bash
# Unit tests (no network required)
pytest tests/ -m "not integration and not benchmark and not slow"

# Including slow tests (large-data cross-validation)
pytest tests/ -m "not integration"

# Integration tests (requires internet)
pytest tests/ -m integration

# C++ vs Python benchmark tests — prints timing and speedup (requires _sqt_core)
pytest tests/test_cpp_hurst.py -m benchmark -s -v

# C++ unit tests (requires _sqt_core built with SQT_BUILD_TESTS=ON)
ctest --test-dir build --config Release -V

# C++ performance benchmark binary (prints a timing table)
# Windows: build\tests\cpp\Release\bench_hurst.exe
# Linux / macOS: ./build/tests/cpp/bench_hurst

# With coverage
pytest tests/ -m "not integration" --cov=src/standard_quant_tools
```

**1033 Python tests total** (892 passing; 141 skipped pending C++ build, across 6 `test_cpp_*.py` files) · **76 C++ unit tests** (17 Hurst + 24 indicators + 18 cointegration + 17 backtest, run via `ctest`)

---

## Documentation

| File | Module |
|---|---|
| `Documentation/01_data_fetching.md` | Data providers, Parquet cache, error handling |
| `Documentation/02_indicators.md` | All 14 technical indicators with examples |
| `Documentation/03_metrics.md` | Risk/return metrics (VaR, Sharpe, Calmar, …) |
| `Documentation/04_backtesting.md` | Vectorized engine, trade log, custom signals, grid search |
| `Documentation/05_portfolio.md` | Multi-asset metrics, correlation, optimization |
| `Documentation/06_screener.md` | Filter reference, large-universe screening, example screens |
| `Documentation/07_agent_tools.md` | Core 14 LLM tools, full 34-tool registry, Pydantic models, end-to-end agent loop |
| `Documentation/08_analysis.md` | Multi-factor regression, cointegration, PCA, Hurst exponent (incl. C++ acceleration) |
| `Documentation/09_advanced_agent_tools.md` | 20 advanced/supplementary/custom-signal/diagnostic tools: regime-adaptive (full-sample and leakage-free walk-forward), pair scanner, walk-forward, risk attribution, position sizer, fundamentals, optimization, advanced indicators, rolling beta, extended risk, backtest diagnostics, true portfolio simulation, pair trade backtest, robustness diagnostics, capacity report, data quality report, compact backtest result |
| `Documentation/10_auditability.md` | Decision-record audit trail, replay verification, correlated logging, `sqt` CLI |
| `Documentation/11_data_quality.md` | Dataset provenance metadata, missing-bar/stale-price/price-jump detection |
| `Development/build_guide.md` | C++ extension build instructions (Windows / Linux / macOS) |
| `Development/performance_insights.md` | Algorithmic analysis: which components benefit from C++ and by how much |
