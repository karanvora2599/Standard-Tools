# Module reference

Every public module in one place: what it holds, and where the deep
documentation for it lives. This exists because the per-module guides
(`01_data_fetching.md` through `17_correctness.md`) each go deep on one
subject, and none of them answers "what is actually in this library".

The prose here is the orientation layer. When a section raises a question it
does not answer, the linked guide answers it.

## Data (`standard_quant_tools.data`)

Three providers implement the same `DataProvider` ABC — `DataFactory.get_provider("yfinance" | "bloomberg" | "polygon")` — so switching is a one-line change with zero changes downstream.

| Function | Description | Returns |
|---|---|---|
| `get_ohlcv(symbol, start, end, interval)` | Historical OHLCV data | `pd.DataFrame` |
| `get_ohlcv_async(...)` | Non-blocking OHLCV fetch | `Awaitable[pd.DataFrame]` |
| `get_ticker_info(symbol)` | Company metadata | `TickerInfo` (Pydantic) |
| `get_financial_ratios(symbol)` | P/E, P/B, D/E, ROE, margins, etc. | `FinancialRatios` (Pydantic) |
| `get_metadata(symbol, interval)` | Dataset provenance: adjusted, survivorship-free, point-in-time, timezone | `DataSetMetadata` (Pydantic) |

**`YFinanceProvider`** (default) — **Caching:** Historical OHLCV calls are saved as Parquet files under `~/.cache/standard_quant_tools/ohlcv/`. Subsequent calls — even from a new Python process — load from disk rather than the network. Override the cache directory with `SQT_CACHE_DIR`.

**`BloombergProvider`** — talks to a local, logged-in Bloomberg Terminal via Desktop API (`blpapi`, optional dependency). No API key: DAPI authenticates via the Terminal login itself; only `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` are configurable (via `.env` locally or CI secrets — see Config & secrets above), and neither is a secret. Daily/weekly/monthly bars only. See [Documentation/01_data_fetching.md](01_data_fetching.md#bloomberg-provider) for the full reference.

**`PolygonProvider`** — talks to Polygon.io's plain REST API, no vendor SDK required. Needs an API key (`SQT_POLYGON_API_KEY`, no default — free tier available at [polygon.io/dashboard/api-keys](https://polygon.io/dashboard/api-keys)). Supports `1m`/`5m`/`15m`/`30m`/`60m`/`1d`/`1wk`/`1mo`/`3mo` bars via the Aggregates endpoint; `get_financial_ratios` derives P/E, P/B, D/E, ROE, and margins from the most recent financials filing plus market cap (no forward estimates or dividend yield). See [Documentation/01_data_fetching.md](01_data_fetching.md#polygonio-provider) for the full reference.

**Data quality (`standard_quant_tools.data.quality`):** `detect_missing_bars`, `detect_stale_prices`, `detect_price_jumps` — heuristic checks on an already-fetched OHLCV frame (weekday gaps, frozen prices, large single-bar jumps). `detect_missing_bars` has no market-holiday calendar, so U.S. holidays show up as false-positive gaps — treat findings as leads to investigate, not confirmed defects. Exposed together with `get_metadata` via the `get_data_quality_report` agent tool.

---

## Technical Indicators (`standard_quant_tools.indicators`)

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

## Metrics (`standard_quant_tools.metrics`)

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

## Analysis (`standard_quant_tools.analysis`)

12 functions across five areas. Several functions have a **C++ fast path** via `_sqt_core` — numbers below are measured, not projected (see [Development/performance_insights.md](../Development/performance_insights.md) for the full methodology and an earlier round of unmeasured projections that turned out to overstate several of these, since corrected):
- `calculate_beta` — 2-variable OLS via closed-form normal equations (1.4× vs. `np.linalg.lstsq` — a real but modest win, not the 10–20× originally projected before this was actually benchmarked)
- `rolling_beta` — incremental O(1)-per-bar sum updates (4.7× vs. two pandas rolling passes), plus a further ~1.1–1.5× from an optional runtime AVX2+FMA dispatch path
- `half_life` / `compute_spread` — same OLS kernel, same modest (~1.1×) speedup
- `cointegration_test` — full Engle-Granger pipeline (23× vs. statsmodels at n=500; **86×** at n=2 000, because the ADF lag sweep now reads every candidate lag off a single nested factorization instead of factorizing once per lag)
- `scan_cointegrated_pairs` — every pair of a universe in one native call, parallel across pairs. A 2 000-ticker screen is ~5 min at 2 000 bars rather than ~9.8 h looping `cointegration_test`
- `hurst_exponent` / `rolling_hurst` — DFA + R/S + sliding window (83–131× / 274×)
- `rolling_factor_loadings` — per-window rank-revealing QR with column pivoting (2.3–10× vs. per-window `lstsq`, larger at shorter windows). This deliberately replaced a much faster incremental-Cholesky path that was **wrong**: its pivot test compared every factor column against the intercept column's diagonal, so factor values around 1e-6 made the whole window read as singular and it returned all-NaN where NumPy returned correct coefficients. Correctness first — see `Development/optimization_plan.md` §5.2 for the plan to recover the speed without giving the rank policy back

### Options Pricing, Greeks & Implied Volatility

`standard_quant_tools.analysis.options` — Black-Scholes-Merton pricing for **European options only**. Dependency-free (standard normal CDF/PDF via `math.erf`, not scipy).

```python
from standard_quant_tools.analysis.options import black_scholes_price, black_scholes_greeks, implied_volatility

price = black_scholes_price(spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, volatility=0.20, option_type="call")
greeks = black_scholes_greeks(42, 40, 0.5, 0.10, 0.20, "call")   # delta, gamma, vega, theta, rho, d1, d2

iv = implied_volatility(option_price=price, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, option_type="call")
print(iv["implied_volatility"], iv["converged"], iv["method"])  # 0.20, True, "newton"
```

`implied_volatility` solves via Newton-Raphson (vega as the derivative) with a bisection fallback over `[1e-6, 5.0]` when Newton fails to converge — the standard robust design for this exact problem. See [Documentation/12_options.md](12_options.md) for the full reference, including unit conventions for `vega`/`theta` and the no-arbitrage bound check `implied_volatility` runs before solving.

### Regression

```python
from standard_quant_tools.analysis import calculate_beta, rolling_beta

stats = calculate_beta(asset_returns, benchmark_returns)
# {'alpha': 0.0003, 'beta': 1.12, 'r_squared': 0.78}

rolling_df = rolling_beta(asset_returns, benchmark_returns, window=60)
# DataFrame with 'Rolling_Beta' column
```

### Multi-Factor Regression

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

### Cointegration & Pairs Spread

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

### PCA on Returns

```python
from standard_quant_tools.analysis import pca_returns, factor_contributions

result = pca_returns(returns_df, n_components=3)
print(result["explained_variance_ratio"])  # PC1: 0.42, PC2: 0.12, PC3: 0.08
print(result["loadings"])                  # (assets × 3) — eigenvector matrix
print(result["factor_returns"])            # (dates × 3) — orthogonal PC time series

contrib = factor_contributions(returns_df, n_components=3)
# DataFrame (assets × PCs): marginal R² each PC contributes per asset
```

### Hurst Exponent

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

> The C++ extension accelerates `hurst_exponent` by 83–131× and `rolling_hurst`
> by 274× (measured; see [16_performance.md](16_performance.md)). The API is identical with or without it — pure Python fallback is automatic. See [Development/build_guide.md](../Development/build_guide.md).

---

## Backtesting (`standard_quant_tools.backtest`)

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

### Parameter Grid Search

```python
from standard_quant_tools.backtest import backtest_grid

results = backtest_grid(
    price_data=df,
    strategy="sma_crossover",          # or any of the 8 STRATEGY_REGISTRY names
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    sort_by="sharpe_ratio",
    n_workers=4,                        # parallel ProcessPoolExecutor
)
print(results.head())   # 9 combinations ranked by Sharpe
```

**8 built-in strategies** (`backtest.strategies.STRATEGY_REGISTRY`): `sma_crossover`, `rsi_mean_reversion`, `macd_crossover`, `bollinger_reversion`, `donchian_breakout` (Turtle-style channel breakout), `momentum_timeseries` (trailing-return threshold, no state machine — the cheapest to evaluate), `vwap_reversion` (mean reversion to rolling VWAP — aimed at intraday/tick data), `adx_trend` (ADX-strength-filtered directional trend). The 4 newer ones don't have dedicated `run_*_backtest` tools — use them via `backtest_grid`, `get_backtest_diagnostics`, or `run_backtest_compact`, or call `STRATEGY_REGISTRY[name](df, **params)` directly. Every hysteresis-based strategy (`rsi_mean_reversion`, `bollinger_reversion`, `donchian_breakout`, `vwap_reversion`) runs its entry/exit tracking through a numba-JIT state machine — no interpreted Python loop regardless of series length; the other four need no per-bar state at all and are pure vectorized pandas/numpy. See [Documentation/04_backtesting.md](04_backtesting.md) for the full reference.

`strategy` also accepts your own signal-generating callable — grid search, C++ speed, and `sort_by` ranking all work identically on your own alpha logic, not just the built-ins. For a pre-computed signal matrix across a ticker universe, see `run_signal_panel_backtest` in [Documentation/04_backtesting.md](04_backtesting.md#grid-searching-your-own-signal).

### Portfolio Simulation Engine

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

### Robustness Diagnostics

`standard_quant_tools.backtest.robustness` answers "is this backtest result trustworthy, or a fluke of one sample path / one lucky parameter combination": `block_bootstrap_ci` (confidence interval on a point-estimate metric), `parameter_sensitivity` (best-vs-median gap on a grid search), and `deflated_sharpe_ratio` (corrects the best observed Sharpe for having been selected as the max of `n_trials` attempts). Complementary to, not a substitute for, out-of-sample walk-forward validation.

---

## Portfolio (`standard_quant_tools.portfolio`)

```python
from standard_quant_tools.portfolio import portfolio_metrics, correlation_matrix, fetch_returns_sync

returns_df = fetch_returns_sync(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-01-01')
weights = [0.4, 0.35, 0.25]

metrics = portfolio_metrics(returns_df, weights)
print(f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Portfolio VaR(95%): {metrics['var_95']:.4f}")

corr = correlation_matrix(returns_df)
```

### Portfolio Optimization

`standard_quant_tools.portfolio.optimize` — produces weights, rather than only scoring weights you already chose (`portfolio_metrics` above) or converting an existing alpha score into weights (`backtest.sizing`).

```python
from standard_quant_tools.portfolio import mean_variance_optimize, risk_parity_weights, black_litterman, build_bl_views

# Markowitz mean-variance — max_sharpe / min_volatility / target_return / target_volatility.
# allow_short=True with max_weight=None is closed-form (numpy only); anything
# constrained (long-only and/or a max_weight cap) uses scipy.
result = mean_variance_optimize(returns_df, objective="max_sharpe", allow_short=False, max_weight=0.4)
print(result["weights"], result["converged"])

# Risk parity — equal (or custom-budgeted) fractional contribution to variance.
cov = (returns_df.cov() * 252).to_numpy()
rp = risk_parity_weights(cov)
print(rp["weights"], rp["risk_contributions"])

# Black-Litterman — market-equilibrium prior blended with explicit views.
P, Q, omega = build_bl_views(["AAPL", "MSFT", "GOOGL"], [{"assets": {"AAPL": 1.0}, "view_return": 0.15}], cov)
bl = black_litterman(cov, market_weights=[0.4, 0.35, 0.25], P=P, Q=Q, omega=omega)
print(bl["posterior_returns"], bl["implied_weights"])
```

See [Documentation/05_portfolio.md](05_portfolio.md#portfolio-optimization) for the full reference, including exactly which cases are closed-form vs. require scipy, and the `run_portfolio_optimization` agent tool (`method="max_sharpe"|"min_volatility"|"target_return"|"target_volatility"|"risk_parity"|"black_litterman"`).

---

## Screener (`standard_quant_tools.screener`)

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

**Failure reporting:** A ticker that raised an exception (bad symbol, fetch error) is never indistinguishable from one that simply failed a filter — both used to collapse to being silently dropped. The returned `DataFrame` now carries this on `.attrs`: `failed_filters` maps ticker → the specific filter key it failed (genuine rejection), `failed_tickers` maps ticker → the exception message (fetch/computation error), and `failed_batches` (multi-worker runs only) lists any worker batch that failed outright.

**Large universes:** Pass `n_workers` to split screening across CPU cores. ≤ 20 tickers run in a single async event loop; larger universes automatically use `ProcessPoolExecutor`.

**Beta filter optimisation:** When `beta_max` / `beta_min` filters are active, SPY data is pre-fetched once per batch instead of once per ticker — a single HTTP round-trip for the whole universe when `n_workers <= 1` (the default for ≤ 20 tickers), or once per worker process for larger multi-worker universes (still eliminating the N−1 redundant per-ticker fetches within each worker's batch).

```python
result = screen_stocks(sp500_tickers, filters={...}, n_workers=8)
```

---

## AI Agent Tools (`standard_quant_tools.agent`)

46 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas — including two tools that backtest a signal you computed yourself rather than one of the built-in indicator strategies.

`Implementation/{Anthropic,OpenAI,Gemini}/` are single-agent reference scripts across all three providers — each narrows the tool list per request via a lightweight **router** (`standard_quant_tools.agent.router`) instead of handing the model all 46 tools on every call: one cheap classification call picks the 1-2 relevant tool categories before the real agent loop starts, no separate agent session required. Each provider folder also carries `Agent_Model_Builder.py`, the one script that drives the separate 8-tool modeling registry instead — it passes `registry="modeling"` and skips the router, since eight tools in one ordered pipeline have no selection ambiguity to remove. For a heavier, more thorough split, `Multi_Agent_Implementation/` (Anthropic only for now) is a full **orchestrator-workers** architecture — a lead agent that delegates to 9 specialist sub-agents, seven over the analysis registry and two over the modeling one, each with its own independent session scoped to a small, non-overlapping tool subset. The analysis workers build on the same category taxonomy (`TOOL_CATEGORY`), so a tool's categorization only needs to be correct in one place. Splitting tools this way is a direct fix for tool-selection confusion between similar tools (e.g. a built-in strategy backtest vs. a bring-your-own-signal backtest, or "run this strategy" vs. "optimize this strategy's parameters"): a worker/routed request that was never given the other tool cannot call it by mistake. See [Documentation/13_agent_orchestration.md](13_agent_orchestration.md).

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
tools = get_agent_tools()  # 46 tools ready for function calling

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

**Advanced agentic tools (8):** `run_regime_adaptive_backtest`, `run_regime_adaptive_walkforward_backtest`, `scan_pairs`, `run_walk_forward_backtest`, `get_portfolio_risk_attribution`, `run_portfolio_optimization`, `get_position_size`, `run_portfolio_simulation`

**Supplementary tools (6):** `get_stock_fundamentals`, `run_backtest_optimization`, `get_advanced_indicators`, `get_rolling_beta`, `get_extended_risk_metrics`, `get_backtest_diagnostics`

**Custom signal tools (2):** `run_custom_signal_backtest`, `run_signal_panel_backtest`

**Diagnostics, capacity & specialized backtests (5):** `run_pair_trade_backtest`, `get_robustness_diagnostics`, `get_capacity_report`, `get_data_quality_report`, `run_backtest_compact`

**Analytics tools (8):** `get_volatility_estimators`, `get_correlation_analysis`, `run_monte_carlo_simulation`, `run_stress_test`, `get_liquidity_metrics`, `run_garch_volatility_forecast`, `run_kalman_hedge_ratio`, `get_tail_risk_metrics`

**Options pricing tools (2):** `get_option_pricing`, `get_implied_volatility`

## Modeling Runtime (`standard_quant_tools.modeling`)

A second, independent 8-tool registry — `list_features`, `analyze_features`,
`build_model_dataset`, `run_model_experiment`, `score_model`,
`inspect_model`, `evaluate_model_portfolio`, `list_modeling_capabilities`
— for building walk-forward-validated statistical models from this
library's own features (21 built-in: technical, market, risk, volume,
statistical and PCA-derived factors), never merged into the 46-tool
`get_agent_tools()`/`TOOL_CATEGORY` surface above.

| Axis | What is available |
|---|---|
| **Targets** | `forward_return`, `forward_return_vol_scaled`, `forward_return_rank`, `forward_return_market_neutral` (regression); `forward_direction`, `triple_barrier` (classification) |
| **Tasks** | `regression`, `classification`, `ranking` — each behind a `ModelAdapter` that owns how its arrays are built, scored and measured |
| **Estimators** | 19 — 11 regression, 6 classification, 2 ranking. scikit-learn throughout, plus `lightgbm`/`xgboost` when installed, two quantile-regression forms, and LambdaRank rankers |
| **Validation** | walk-forward (rolling or expanding) and purged K-fold, both with a target-overlap purge |
| **Preprocessing** | pooled or cross-sectional normalization |
| **Weighting** | none, label uniqueness, time decay, or both |
| **Search** | optional grid or random search on each fold's training window |

Everything past the defaults is opt-in behind an explicit spec field, so an
existing `ModelSpec` predicts exactly what it predicted before.

Walk-forward validation purges training rows whose forward-return label
would resolve inside the test window — feature-side embargo alone does not
close that channel. Registered models are content-addressed, verified on
load, and self-contained; `score_model` refuses an `as_of` inside the
training window, since the deployed estimator is refit on the full panel.

Two of the options are corrections rather than variations, and the
documentation says so. **Cross-sectional normalization** exists because
pooled z-scoring leaves the market factor inside every feature, so a model
judged on cross-sectional IC can score well by learning "today was an up
day". **Sample weighting** exists because `effective_sample_size` was always
reported and never acted on: overlapping forward returns make consecutive
rows largely redundant. Neither is the default only because switching it
changes what every existing model predicts.

The provider and bar interval are named on the `DatasetSpec` (they were
previously implicit, so every dataset came from the default provider at its
default interval and no model recorded which), the universe is fetched
concurrently, and `build_model_dataset` returns the coverage and provenance
conditions that qualify the resulting metrics — a survivors-only universe, a
provider that revises history, a symbol covering part of the window, a
complete-case intersection that truncated the panel. Those travel onto the
trained model, so `inspect_model(view="lineage")` shows them next to the OOS
numbers.

See [Documentation/15_modeling.md](15_modeling.md) for the
full reference, including what is explicitly deferred (fundamentals need a
point-in-time provider first; time-varying universe membership needs
index-constituent history no shipped provider exposes, so survivorship bias
is disclosed rather than corrected).

---
