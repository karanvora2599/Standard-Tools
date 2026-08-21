# Standard Quant Tools for AI Agents

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

A high-performance, modular Python library for quantitative financial analysis. Designed to give AI agents and automated workflows **clean structured data**, **mathematical accuracy**, and **robust error handling**.

Maintained by [Karan Vora](mailto:kv2154@nyu.edu). Source: [github.com/karanvora2599/Standard-Tools](https://github.com/karanvora2599/Standard-Tools).

## Key Features

- **High Performance** — Optional C++ extension (`_sqt_core`), with real measured numbers (not projections) on this dev machine: `run_strategy` backtest kernel (~58× end-to-end once a wrapper redundancy was found and fixed), `batch_run_strategy` grid kernel (allocation-free summary metrics + OpenMP across parameter combinations — ~6–11× depending on grid size), Hurst/rolling Hurst (83–131× / 274× vs. the Python fallback — no numba path exists for these — plus a further ~5–11× from OpenMP + a one-pass DFA reformulation on top of that, still vs. the *original* C++ implementation), Engle-Granger cointegration (23× vs. statsmodels at n=500, 86× at n=2 000 — the ADF lag sweep now reads every candidate lag off one factorization), a batch pair-scan kernel that screens a 2 000-name universe in ~5 min instead of ~9.8 h, panel indicator entry points that compute a whole universe in one call (11.9×), `rolling_factor_loadings` per-window rank-revealing QR (2.3–10× depending on window), Wilder's ATR (28×), `rolling_beta` incremental sums (4.7×, plus a further ~1.1–1.5× from an optional runtime AVX2+FMA dispatch path), Monte Carlo forward simulation (moving-block bootstrap; 2× vs. uncompiled Python, plus a further ~2.0–2.4× from an optional OpenMP-parallel path on this 16-core machine). RSI/ADX/PSAR/GARCH/Kalman/Donchian/VWAP-reversion measure close to 1× against *warm numba* on a machine where numba works — their real value is eliminating numba's ~200ms–1.1s per-process JIT cold-start and the numpy-ABI fragility that broke numba once already for RSI/ADX/PSAR, not raw steady-state speed (see the Performance section below for the full, honest breakdown); NumPy single-pass ATR (5.6×); BLAS-backed portfolio covariance; async concurrent data fetching (including full-universe portfolio simulation, not just correlation/optimization); persistent Parquet disk cache; `ProcessPoolExecutor` screener and parallel backtest grid
- **Agent-First Design** — All tools return Pydantic models; 46 LLM-callable tools with OpenAI/Anthropic function-calling schemas, including two bring-your-own-signal tools; descriptive errors for self-correction
- **Comprehensive Coverage** — 14 indicators, 13 risk/return metrics + 5 backtest diagnostics, 12 analysis functions plus Black-Scholes-Merton option pricing/Greeks/implied volatility, portfolio analysis and optimization (Markowitz mean-variance, risk parity, Black-Litterman), stock screener, 8 backtest strategies + parameter grid search, a shared-cash portfolio simulation engine with pluggable cost/constraint models, pairs backtest, and walk-forward/robustness diagnostics — grid search and the signal-panel backtester also accept your own signal-generating callable/matrix, not just the built-in strategies
- **Robust Infrastructure** — Retry logic with exponential backoff, TTL + Parquet caching, custom exception hierarchy, `@validate_series` decorator, decision-record audit trail (`sqt` CLI), optional C++/scipy/numba graceful fallback
- **Audited for correctness** — Both tiers have been through a line-by-line correctness audit (41 findings fixed; see [Correctness & Backend Parity](#correctness--backend-parity)), followed by two reviews of the modeling runtime: the first found 7 critical issues (two leakage channels, a PCA start-vector degeneracy), the second found 20 more across modeling, the data layer and the numerics — a full-refit model that had seen prices past its own recorded cutoff, an `end_date` that meant different things per provider, a "cross-section" that could mix dates, and aliases that could forge feature provenance. None of it was catchable by the suite as it stood. Every finding was reproduced against a live interpreter before being fixed, and each is pinned by a regression test. 2451 Python tests + 9 C++ suites, all green.

---

## Installation

```bash
pip install .
# or
poetry install
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `yfinance`, `numba`, `aiohttp`, `cachetools`, `pydantic`, `statsmodels`, `scikit-learn`, `plotly`, `pyarrow`, `python-dotenv`

**Optional:** `pip install standard_quant_tools[bloomberg]` adds `blpapi` (Bloomberg's own SDK) for `BloombergProvider` — requires a running, logged-in Bloomberg Terminal; see [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#bloomberg-provider). `pip install standard_quant_tools[signing]` adds `cryptography` for Ed25519 audit-checkpoint signing — see [Audit Trail & CLI](#audit-trail--cli-standard_quant_toolsaudit-sqt) below. `PolygonProvider` needs no extra install — it's a plain REST API — just an API key (`SQT_POLYGON_API_KEY`); see [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#polygonio-provider). `pip install standard_quant_tools[polars]` adds optional `polars` interop for a growing subset of functions — pandas remains the default and required backend either way; see [Documentation/14_polars_support.md](Documentation/14_polars_support.md).

> **Note on the C++ extension:** `pip install .` now **builds `_sqt_core` when a C++ toolchain is available** (the backend is scikit-build-core, which drives the project's CMake build). Without a compiler the install still succeeds and you get the pure-Python package — every indicator, backtest and analysis function works through its Numba/pure-Python fallback, and `HAS_CPP` is `False`. That degradation is deliberate: the extension is an optional accelerator, so requiring a compiler to install would turn it into a hard dependency. Pass `-C cmake.define.SQT_REQUIRE_NATIVE=ON` to make a missing toolchain a hard error instead (what CI uses). For the in-place developer build, see [Development/build_guide.md](Development/build_guide.md).

> **Config & secrets:** copy [`.env.example`](.env.example) to `.env` (already `.gitignore`d) for any local provider configuration — currently `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` and `SQT_POLYGON_API_KEY`. `standard_quant_tools.config.load_env()` loads it automatically and is a harmless no-op when `.env` doesn't exist (the normal state in CI). In GitHub Actions / GitLab CI, set the same variable names as encrypted secrets and inject them as job-level environment variables instead — no `.env` file involved, no code changes needed either way.

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

Three providers implement the same `DataProvider` ABC — `DataFactory.get_provider("yfinance" | "bloomberg" | "polygon")` — so switching is a one-line change with zero changes downstream.

| Function | Description | Returns |
|---|---|---|
| `get_ohlcv(symbol, start, end, interval)` | Historical OHLCV data | `pd.DataFrame` |
| `get_ohlcv_async(...)` | Non-blocking OHLCV fetch | `Awaitable[pd.DataFrame]` |
| `get_ticker_info(symbol)` | Company metadata | `TickerInfo` (Pydantic) |
| `get_financial_ratios(symbol)` | P/E, P/B, D/E, ROE, margins, etc. | `FinancialRatios` (Pydantic) |
| `get_metadata(symbol, interval)` | Dataset provenance: adjusted, survivorship-free, point-in-time, timezone | `DataSetMetadata` (Pydantic) |

**`YFinanceProvider`** (default) — **Caching:** Historical OHLCV calls are saved as Parquet files under `~/.cache/standard_quant_tools/ohlcv/`. Subsequent calls — even from a new Python process — load from disk rather than the network. Override the cache directory with `SQT_CACHE_DIR`.

**`BloombergProvider`** — talks to a local, logged-in Bloomberg Terminal via Desktop API (`blpapi`, optional dependency). No API key: DAPI authenticates via the Terminal login itself; only `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` are configurable (via `.env` locally or CI secrets — see Config & secrets above), and neither is a secret. Daily/weekly/monthly bars only. See [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#bloomberg-provider) for the full reference.

**`PolygonProvider`** — talks to Polygon.io's plain REST API, no vendor SDK required. Needs an API key (`SQT_POLYGON_API_KEY`, no default — free tier available at [polygon.io/dashboard/api-keys](https://polygon.io/dashboard/api-keys)). Supports `1m`/`5m`/`15m`/`30m`/`60m`/`1d`/`1wk`/`1mo`/`3mo` bars via the Aggregates endpoint; `get_financial_ratios` derives P/E, P/B, D/E, ROE, and margins from the most recent financials filing plus market cap (no forward estimates or dividend yield). See [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#polygonio-provider) for the full reference.

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

12 functions across five areas. Several functions have a **C++ fast path** via `_sqt_core` — numbers below are measured, not projected (see [Development/performance_insights.md](Development/performance_insights.md) for the full methodology and an earlier round of unmeasured projections that turned out to overstate several of these, since corrected):
- `calculate_beta` — 2-variable OLS via closed-form normal equations (1.4× vs. `np.linalg.lstsq` — a real but modest win, not the 10–20× originally projected before this was actually benchmarked)
- `rolling_beta` — incremental O(1)-per-bar sum updates (4.7× vs. two pandas rolling passes), plus a further ~1.1–1.5× from an optional runtime AVX2+FMA dispatch path
- `half_life` / `compute_spread` — same OLS kernel, same modest (~1.1×) speedup
- `cointegration_test` — full Engle-Granger pipeline (23× vs. statsmodels at n=500; **86×** at n=2 000, because the ADF lag sweep now reads every candidate lag off a single nested factorization instead of factorizing once per lag)
- `scan_cointegrated_pairs` — every pair of a universe in one native call, parallel across pairs. A 2 000-ticker screen is ~5 min at 2 000 bars rather than ~9.8 h looping `cointegration_test`
- `hurst_exponent` / `rolling_hurst` — DFA + R/S + sliding window (83–131× / 274×)
- `rolling_factor_loadings` — per-window rank-revealing QR with column pivoting (2.3–10× vs. per-window `lstsq`, larger at shorter windows). This deliberately replaced a much faster incremental-Cholesky path that was **wrong**: its pivot test compared every factor column against the intercept column's diagonal, so factor values around 1e-6 made the whole window read as singular and it returned all-NaN where NumPy returned correct coefficients. Correctness first — see `Development/optimization_plan.md` §5.2 for the plan to recover the speed without giving the rank policy back

#### Options Pricing, Greeks & Implied Volatility

`standard_quant_tools.analysis.options` — Black-Scholes-Merton pricing for **European options only**. Dependency-free (standard normal CDF/PDF via `math.erf`, not scipy).

```python
from standard_quant_tools.analysis.options import black_scholes_price, black_scholes_greeks, implied_volatility

price = black_scholes_price(spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, volatility=0.20, option_type="call")
greeks = black_scholes_greeks(42, 40, 0.5, 0.10, 0.20, "call")   # delta, gamma, vega, theta, rho, d1, d2

iv = implied_volatility(option_price=price, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, option_type="call")
print(iv["implied_volatility"], iv["converged"], iv["method"])  # 0.20, True, "newton"
```

`implied_volatility` solves via Newton-Raphson (vega as the derivative) with a bisection fallback over `[1e-6, 5.0]` when Newton fails to converge — the standard robust design for this exact problem. See [Documentation/12_options.md](Documentation/12_options.md) for the full reference, including unit conventions for `vega`/`theta` and the no-arbitrage bound check `implied_volatility` runs before solving.

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
    strategy="sma_crossover",          # or any of the 8 STRATEGY_REGISTRY names
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    sort_by="sharpe_ratio",
    n_workers=4,                        # parallel ProcessPoolExecutor
)
print(results.head())   # 9 combinations ranked by Sharpe
```

**8 built-in strategies** (`backtest.strategies.STRATEGY_REGISTRY`): `sma_crossover`, `rsi_mean_reversion`, `macd_crossover`, `bollinger_reversion`, `donchian_breakout` (Turtle-style channel breakout), `momentum_timeseries` (trailing-return threshold, no state machine — the cheapest to evaluate), `vwap_reversion` (mean reversion to rolling VWAP — aimed at intraday/tick data), `adx_trend` (ADX-strength-filtered directional trend). The 4 newer ones don't have dedicated `run_*_backtest` tools — use them via `backtest_grid`, `get_backtest_diagnostics`, or `run_backtest_compact`, or call `STRATEGY_REGISTRY[name](df, **params)` directly. Every hysteresis-based strategy (`rsi_mean_reversion`, `bollinger_reversion`, `donchian_breakout`, `vwap_reversion`) runs its entry/exit tracking through a numba-JIT state machine — no interpreted Python loop regardless of series length; the other four need no per-bar state at all and are pure vectorized pandas/numpy. See [Documentation/04_backtesting.md](Documentation/04_backtesting.md) for the full reference.

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

#### Portfolio Optimization

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

See [Documentation/05_portfolio.md](Documentation/05_portfolio.md#portfolio-optimization) for the full reference, including exactly which cases are closed-form vs. require scipy, and the `run_portfolio_optimization` agent tool (`method="max_sharpe"|"min_volatility"|"target_return"|"target_volatility"|"risk_parity"|"black_litterman"`).

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

**Failure reporting:** A ticker that raised an exception (bad symbol, fetch error) is never indistinguishable from one that simply failed a filter — both used to collapse to being silently dropped. The returned `DataFrame` now carries this on `.attrs`: `failed_filters` maps ticker → the specific filter key it failed (genuine rejection), `failed_tickers` maps ticker → the exception message (fetch/computation error), and `failed_batches` (multi-worker runs only) lists any worker batch that failed outright.

**Large universes:** Pass `n_workers` to split screening across CPU cores. ≤ 20 tickers run in a single async event loop; larger universes automatically use `ProcessPoolExecutor`.

**Beta filter optimisation:** When `beta_max` / `beta_min` filters are active, SPY data is pre-fetched once per batch instead of once per ticker — a single HTTP round-trip for the whole universe when `n_workers <= 1` (the default for ≤ 20 tickers), or once per worker process for larger multi-worker universes (still eliminating the N−1 redundant per-ticker fetches within each worker's batch).

```python
result = screen_stocks(sp500_tickers, filters={...}, n_workers=8)
```

---

### AI Agent Tools (`standard_quant_tools.agent`)

46 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas — including two tools that backtest a signal you computed yourself rather than one of the built-in indicator strategies.

`Implementation/{Anthropic,OpenAI,Gemini}/` are single-agent reference scripts across all three providers — each narrows the tool list per request via a lightweight **router** (`standard_quant_tools.agent.router`) instead of handing the model all 46 tools on every call: one cheap classification call picks the 1-2 relevant tool categories before the real agent loop starts, no separate agent session required. For a heavier, more thorough split, `Multi_Agent_Implementation/` (Anthropic only for now) is a full **orchestrator-workers** architecture — a lead agent that delegates to 7 specialist sub-agents, each with its own independent session scoped to a small, non-overlapping tool subset. Both build on the same category taxonomy (`TOOL_CATEGORY`), so a tool's categorization only needs to be correct in one place. Splitting tools this way is a direct fix for tool-selection confusion between similar tools (e.g. a built-in strategy backtest vs. a bring-your-own-signal backtest, or "run this strategy" vs. "optimize this strategy's parameters"): a worker/routed request that was never given the other tool cannot call it by mistake. See [Documentation/13_agent_orchestration.md](Documentation/13_agent_orchestration.md).

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

### Modeling Runtime (`standard_quant_tools.modeling`)

A second, independent 6-tool registry — `list_features`, `build_model_dataset`,
`run_model_experiment`, `score_model`, `inspect_model` — for building
walk-forward-validated statistical models from this library's own features
(21 built-in: technical, market, risk, volume, statistical and PCA-derived
factors), never merged into the 46-tool `get_agent_tools()`/`TOOL_CATEGORY`
surface above.

Regression and classification are both reachable through the same five
tools (`TargetSpec(type="forward_return" | "forward_direction")`).
Walk-forward validation purges training rows whose forward-return label
would resolve inside the test window — feature-side embargo alone does not
close that channel. Registered models are content-addressed, verified on
load, and self-contained; `score_model` refuses an `as_of` inside the
training window, since the deployed estimator is refit on the full panel.

The provider and bar interval are named on the `DatasetSpec` (they were
previously implicit, so every dataset came from the default provider at its
default interval and no model recorded which), the universe is fetched
concurrently, and `build_model_dataset` returns the coverage and provenance
conditions that qualify the resulting metrics — a survivors-only universe, a
provider that revises history, a symbol covering part of the window, a
complete-case intersection that truncated the panel. Those travel onto the
trained model, so `inspect_model(view="lineage")` shows them next to the OOS
numbers.

See [Documentation/15_modeling.md](Documentation/15_modeling.md) for the
full reference, including what is explicitly deferred (fundamentals need a
point-in-time provider first; time-varying universe membership needs
index-constituent history no shipped provider exposes, so survivorship bias
is disclosed rather than corrected).

---

## Performance

### C++ Extension (`_sqt_core`)

The optional compiled C++ extension accelerates the highest-impact CPU-bound paths. The API is identical with or without it — pure Python fallback is automatic.

**Measured, not projected**, on a Windows 11 / MSVC 19.44 / Python 3.12 dev machine (16 logical cores) — each row toggles the same module's own `HAS_CPP` flag and times both paths back-to-back, so it's an apples-to-apples comparison, not separately-run numbers:

| Operation | vs. numba (warm)¹ | vs. numba JIT cold-start² | Notes |
|---|---|---|---|
| `hurst_exponent` DFA (n = 500) | **83×** (4.57ms → 0.05ms) | — | No numba path exists for Hurst — this is C++ vs. the pure-Python fallback directly. |
| `hurst_exponent` DFA (n = 2 000) | **131×** (12.3ms → 0.09ms) | — | Same. |
| `rolling_hurst` (n = 2 000, window = 200, step = 1) | **274×** (4.64s → 17ms) | — | Same — the standout number in this table, and it holds up under real measurement. |
| `rsi` (n = 2 000) | **5.3×** (0.47ms → 0.09ms) | 1109ms → 1.2ms first call | |
| `adx` (n = 2 000) | **0.9×** (essentially tied) | 1110ms → 1.2ms first call | Numba's *warm* ADX is already about as fast as C++ on this machine — see the note below. |
| `parabolic_sar` (n = 2 000) | **1.1×** (essentially tied) | ~similar order to ADX | |
| `wilder_atr` (n = 2 000) | **28×** (4.40ms → 0.15ms) | | |
| `bollinger_bands` (n = 2 000) | **1.6×** | | |
| `stochastic_oscillator` (n = 2 000) | **2.6×** | | |
| `cointegration_test` (n = 500, vs. statsmodels) | **23×** (8.3ms → 0.37ms) | — | Compares against statsmodels, not numba — statsmodels has no JIT path at all. |
| `cointegration_test` (n = 2 000, vs. statsmodels) | **86×** (86.9ms → 1.01ms) | — | The ratio grows with n because the kernel is no longer quadratic: the ADF lag sweep used to run one column-pivoted QR per candidate lag, `O(T·L³)` in total, and now reads every candidate's residual off one nested factorization, `O(T·L²)`. |
| `scan_cointegrated_pairs` (2 000 tickers, 2 000 bars) | **111×** (9.81 h → 5.31 min) | — | One native call over the whole pair set instead of ~2 M Python round trips, parallel across pairs. |
| `calculate_beta` (n = 500, vs. `lstsq`) | **1.4×** | — | |
| `half_life` (n = 500, vs. `lstsq`) | **1.1×** | — | |
| `run_strategy` (n = 2 000, `include_trade_log=False`) | **~58×** (26.8ms → 0.46ms) | — | A wrapper-redundancy bug, not a kernel problem — see note below. Was ~1.0× before the fix. |
| `batch_run_strategy` (n = 2 000, num_tests = 2 000) | **~11×** (51.6ms → 4.6ms) | — | Allocation-free summary kernel + OpenMP across parameter combinations (16 cores); ranges ~6–11× depending on grid size — see `Development/performance_insights.md`. |
| `rolling_beta` (n = 2 000, window = 60) | **4.7×**, plus a further ~1.1–1.5× from optional AVX2+FMA dispatch | — | |
| `rolling_factor_loadings` (n = 500, window = 60, k = 3) | **5.5×** (8.9ms → 1.6ms) | — | Was 26× when this used an incremental Cholesky update. That path was removed because it was wrong on small-magnitude factors (all-NaN where NumPy answered correctly); the replacement is a per-window rank-revealing QR. 10.0× at n=2 000/window=60, 2.3× at window=252 — the gap narrows as the window grows, since cost is `O(n·window·p²)`. |
| `technical_indicators_panel` (500 tickers × 1 000 bars, 5 indicators) | **11.9×** (1 727.6ms → 144.7ms) | — | vs. looping the per-ticker Python wrappers. The pybind11 boundary was never the cost (2.7 µs/call, 14%) — the per-ticker pandas round trip was, at 318 µs against 19 µs of kernel. |
| `run_portfolio_simulation` (1 000 tickers × 2 000 bars) | **5.3×** (188.7ms → 35.8ms) | — | Most of it was *not* the bar loop: profiling put 92% in building the dense price matrices, one pandas `.loc` per (ticker, column). The native bar-loop kernel adds a further 1.7–3.3× on top. |
| `rolling_hurst` (n = 2 000, window = 200) | **274×** vs. Python, plus a further ~10.5× from OpenMP + a one-pass DFA reformulation on top of the *original* C++ implementation (measured independently, at the same n/window) | — | Combining the two independently-measured ratios gives roughly ~2 900× vs. the pure-Python fallback at this size — not itself a single direct measurement, but both factors are real. |
| `simulate_forward_paths` (n_simulations = 5 000, horizon = 60) | **2.0×** (74.8ms → 37.7ms) | — | No numba path ever existed for this one — was pure uncompiled Python. See OpenMP note below for the parallel path's own measured speedup. |
| `garch11_variance_recursion` (n = 2 000, warm steady-state) | **0.8×** (10.8ms → 12.9ms, i.e. slightly *slower*) | 219ms → 4.8ms first call | The whole point of this port is the cold-start column, not this one — see below. |
| `kalman_filter_*`, `donchian_state_machine`, `vwap_reversion_state_machine` | not separately re-measured | same cold-start pattern as GARCH/ADX above | |

¹ **This is C++ vs. numba, not C++ vs. interpreted Python** — numba is fully functional on this dev machine (NumPy 2.0.2), so the "Python fallback" path for RSI/ADX/PSAR/GARCH/Kalman/signal-state-machines actually means *numba-JIT-compiled*, already close to C speed once warm. On a machine where numba is broken or unavailable (e.g. NumPy 2.4+, which is what originally motivated porting RSI/ADX/PSAR to C++ in the first place), the true comparison is C++ vs. an *interpreted* Python loop, which would show much larger gains than this table — those older, unmeasured "10–30×"-style estimates are directionally right for that scenario, just not what this table reports. Hurst and cointegration have no numba path at all, so their numbers above are already the "real" comparison either way.

² Measured via a genuinely fresh subprocess per number (`time.perf_counter()` around the very first call, nothing warmed up beforehand) — this is the number that actually matters for a single one-off agent-tool call in a new process, which is the primary reason GARCH/Kalman/Donchian/VWAP-reversion were ported at all (see `Development/performance_insights.md`).

**Two honest findings from actually measuring this**, worth calling out rather than hiding:
- **`run_strategy` originally showed only ~1.0× end-to-end**, not the then-documented 3–8×, even though the raw C++ kernel genuinely was faster in isolation (confirmed by `tests/cpp/bench_backtest.cpp`'s native-only numbers below). The gap was never the kernel — it was the Python wrapper: `pct_change`/`shift` computed unconditionally before the C++ dispatch check even though the C++ path never used them, and an unconditional Python trade-log rebuild that overwrote already-correct native stats every call. **Since fixed** (removing both, and only building the Python trade log when a caller actually asks for it via `include_trade_log=True`) — the real, current number is **~58×** (26.8ms → 0.46ms), reflected in the table above. `batch_run_strategy` never had this specific bug (its consumer already read native stats directly), but has since gained its own further ~6–11× from an allocation-free summary kernel plus OpenMP across the parameter grid.
- **OpenMP's measured speedup for `simulate_forward_paths` is ~2.0–2.4×** on this 16-core machine (min-of-7-runs across separate process invocations, `n_simulations=200 000`) — not the near-linear-with-cores scaling the per-path independence would suggest in theory. MSVC's OpenMP support here is version 2.0 (an older spec) — some of that gap was expected going in. A later pass eliminating each path's small per-path RNG/buffer allocations moved this scaling ratio only within noise (~2.4×→~2.1×, both real measurements) — the allocation being eliminated turned out not to be the dominant cost at this problem size, a legitimate change worth keeping regardless (fewer allocations is never worse) but not the win that framing initially suggested.

Raw C++-only (no Python involved) numbers from `tests/cpp/bench_hurst.cpp` and `tests/cpp/bench_backtest.cpp`, run via `ctest`:

| Operation | Time |
|---|---|
| `hurst_dfa` (n = 2 000) | 0.107 ms |
| `rolling_hurst` DFA (n = 2 000, window = 200, step = 1) | 16.9 ms |
| `rolling_hurst` DFA (n = 5 000, window = 252, step = 1) | 60.8 ms |
| `run_strategy` long-only, all costs (n = 2 000) | 0.017 ms |
| `run_strategy` mixed L/F/S signals, all costs (n = 5 000) | 0.089 ms |

The rolling Hurst gain is the most significant and the most robust to how you measure it: rather than re-entering Python for every bar, the entire sliding-window pass runs in one C++ function, with no numba equivalent to compare against either way.

`rolling_factor_loadings` is the one entry in this table that got **slower on purpose**. It used incremental rank-1 XtX updates — O(k²) per bar instead of a full O(n·k²) `lstsq` — and that was 26×. It was also wrong: the pivot test compared every column against the single largest diagonal of XtX, which belongs to the intercept column and equals the window length, so factors around 1e-6 made every window read as singular and the kernel returned all-NaN where the NumPy fallback returned correct coefficients. It now runs a column-pivoted QR per window, which ranks each column by its own norm and gives a scale-invariant answer, at 2.3–10×. Recovering the speed via QR update/downdate is planned but not attempted — see `Development/optimization_plan.md` §5.2, including why the analogous Cholesky attempt was reverted.

**Deeper native optimization pass** (on top of the module-level wins above): `run_strategy`/`batch_run_strategy` and `rolling_hurst` now parallelize across independent work (parameter combinations, rolling windows) via OpenMP; several kernels' Python/C++ boundary crossings were converted to direct-write into a pre-allocated NumPy buffer instead of allocate-then-copy; `rolling_beta` gained an optional runtime-dispatched AVX2+FMA reduction path (falls back safely to the portable scalar kernel on older CPUs); the build enables LTO/IPO automatically and supports an opt-in, local-only PGO workflow. One optimization (a rank-1 Cholesky *factor* update/downdate, intended to replace `rolling_factor_loadings`'s O(p³) per-step refactor with O(p²)) was implemented, gated against the existing path on real before/after data, found to break down numerically on near-singular inputs, and reverted rather than shipped — documented in `CHANGELOG.md` alongside the items that did ship.

See [Development/performance_insights.md](Development/performance_insights.md) for the full methodology, every number above with its exact benchmark script, and a running log of real edge-case bugs found and fixed while actually building, running, and benchmarking this codebase — not assumed from reading the code (a degenerate-input NaN in the cointegration ADF test, a half-life NaN-vs-inf gap, an input-validation gap in the Monte Carlo binding, incorrect hand-written C++ test expectations that had never been compiled before, and a Linux-CI-only flake in an audit-trail test caused by an unfiltered directory glob, among others).

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
| Portfolio simulation (100 tickers × 2 000 bars, monthly) | 1 503 ms (per-ticker `.loc`) | 32 ms (dense matrices) | **47×** | 200 000 pandas label lookups replaced by positional indexing; 500 tickers → **78×** |

> **Portfolio simulator note:** `run_portfolio_simulation` holds prices, target weights and liquidity baselines as dense `(n_bars × n_tickers)` matrices and executes the default cost configuration as array arithmetic. The vectorized rebalance is deliberately narrow — `per_share` commission, the impact model and the ADV constraint each need a per-element decision (a per-order minimum, a per-ticker volatility lookup, an error naming one ticker) and keep the explicit loop, selected automatically by cost model. Both routes are held to the same numbers by tests: agreement with the pre-vectorization implementation is within 1.7e-15 relative across every configuration, with `rebalance_log` identical, the residual being pairwise-vs-sequential summation rather than a different formula. The speedup grows with universe size because the removed cost scaled with tickers × bars. See [Documentation/04_backtesting.md](Documentation/04_backtesting.md).

> **Numba note:** RSI, ADX, Parabolic SAR, GARCH's variance recursion, the Kalman filter, and every backtest-strategy state machine (RSI/Bollinger/Donchian/VWAP-reversion) are decorated with `@njit`. This requires Numba with a compatible NumPy version (≤ 2.0, or wherever Numba's own ABI support currently ends). On an incompatible NumPy version, Numba decorators are a no-op and the code falls back to interpreted Python, where C++ genuinely wins big (the original ~10–30× estimates for RSI/ADX/PSAR describe this scenario). On a machine where Numba *is* working (like the one that produced the measured table below), it's already close to C speed once warm — real measurement shows C++ landing anywhere from a tie to a modest win against it, not a blowout. What C++ reliably wins either way: no per-process JIT compile tax (measured at ~200ms–1.1s on the first call in a fresh process, gone entirely with C++) and no numpy-ABI fragility risk (the exact failure mode that motivated porting RSI/ADX/PSAR to C++ in the first place). Every one of these falls back to pure Python automatically when neither C++ nor Numba is available.

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

**Exception hierarchy:** `QuantError` → `DataProviderError` → `DataNotFoundError / InvalidSymbolError / APIError / NonRetryableAPIError`

`ValidationError` (a `QuantError`, not a `DataProviderError`) is raised for
caller-side input problems — a bad period, a non-finite price, mismatched
series lengths — and is **never retried or re-typed** by the `retry`
decorator. See [01_data_fetching.md](Documentation/01_data_fetching.md) for
the full retry classification table.

---

## Correctness & Backend Parity

Most functions here have two implementations: a C++ kernel in `_sqt_core`
and a Python/NumPy fallback used when the extension isn't built. **The two
are contractually required to return the same answer**, and that requirement
is now tested directly rather than assumed.

Both tiers went through a line-by-line correctness audit (31 findings in the
Python tier, 10 in the C++ tier — full write-ups in
[CHANGELOG.md](CHANGELOG.md)). Three themes are worth knowing as a user:

1. **Backend divergences.** Five cases were found where the same call
   returned a different answer depending on whether `_sqt_core` was built —
   `stochastic_oscillator` on a flat window, `cointegration_test`'s
   `autolag` handling, `hurst_exponent`'s regime post-processing,
   `rolling_factor_loadings` on an underdetermined window, and
   `profit_factor` when no trade wins or loses. All are fixed and pinned by
   tests that assert the two backends against *each other* — a test pinning
   only one side cannot see a divergence, which is exactly how several of
   these survived.

2. **Input validation is now uniform across tiers.** Non-finite inputs,
   invalid periods, and mismatched series lengths raise `ValidationError`
   at the Python boundary regardless of which tier executes. Previously
   several checks lived inside the C++ branch only, so the same bad input
   raised with the extension present and silently produced NaN without it.
   The most consequential case: `run_strategy(fill_price="next_open")` never
   validated `Open`, and because `cumprod` skips NaN, a gap there silently
   *dropped* that bar's P&L rather than surfacing — see
   [04_backtesting.md](Documentation/04_backtesting.md#input-validation-contract).

3. **Memory safety in the Numba tier.** `@njit` compiles with bounds
   checking disabled. Two kernels (`_adx_numba`, `_psar_numba`) could write
   or read past their output arrays on short/empty input, returning
   plausible numbers instead of raising. Both are guarded, and all three
   execution tiers now agree on those inputs.

4. **Leakage in the modeling runtime.** A separate review of
   `standard_quant_tools.modeling` found seven critical issues the suite as
   it stood could not have caught, two of them look-ahead channels. Walk-
   forward validation was given only an integer `embargo`, never the target
   horizon, so training labels built from test-period prices survived the
   split — and the existing engine tests happened to pass
   `embargo == horizon`, which accidentally satisfied the missing invariant
   and hid it. Training rows are now purged by a per-row label end *date*:
   `horizon` counts an entity's own bars, so on a sparse calendar an integer
   offset under-purges exactly where it matters. Separately,
   `FeatureSpec.params` was unvalidated, and pandas reads a negative
   `pct_change` period as a *forward* window — so a negative lookback made a
   feature read future prices while its `pit_safe` label, and the
   point-in-time gate, stayed satisfied. Both are pinned by regression
   tests; see [15_modeling.md](Documentation/15_modeling.md) and
   [CHANGELOG.md](CHANGELOG.md) for the rest.

5. **The second modeling audit — 20 more items.** A follow-up review swept
   the modeling stack, the data layer beneath it, and the numerics both rest
   on. The unifying failure mode is a result that is plausible, internally
   consistent, and wrong:

   - A horizon-`h` label reads `Close[t+h]`, so a full-refit model has seen
     prices past its recorded `train_end_date`. Manifests now carry
     `training_information_cutoff` and `score_model` gates on it.
   - `end_date` was exclusive on yfinance and inclusive on Polygon and
     Bloomberg, so the default provider silently dropped the final bar. The
     ABC now specifies **inclusive** and all three providers trim to it.
   - `score_model` returned a "cross-section" that could mix dates, because
     each entity contributed its own latest surviving bar. Now one
     `effective_score_date`, with `stale_entities` and `staleness_days`.
   - A calendar gap in OOS predictions compressed the price axis: a
     boundary bar carried **26×** a normal daily return.
   - An alias could make a feature record *another* feature's implementation
     hash — the field whose whole job is answering what produced a column.
   - ICIR was computed as a mean of per-fold ICIRs, discarding exactly the
     between-fold variation it exists to measure.
   - Volatility features annualized with `sqrt(252)` at every interval.
   - The Python and C++ backtests disagreed on where a trade ends, so a
     resized position produced `num_trades=1` beside a two-row trade log.

   Every item was reproduced against a live interpreter before being fixed
   and is pinned by a regression test.

6. **The portfolio, screener and agent-tools audit — 10 more items.** The
   three packages the earlier passes had only touched incidentally. The
   sharpest two both produced a confident number that was not merely
   imprecise but inverted or fictional:

   - With observations ≤ assets a sample covariance is singular *by
     construction*, and the constrained optimizer answered by finding a
     direction in its null space — reporting a portfolio at ~0% volatility
     that carried **23% annualized volatility out of sample**.
   - `max_sharpe` returned the *minimum*-Sharpe portfolio whenever the
     risk-free rate reached the minimum-variance return, because
     normalizing the tangency solution by a negative sum flips it onto the
     inefficient branch.
   - A beta that could not be estimated was reported as `0.0`, so a ticker
     with no overlapping history **passed** a `beta_max` screen.
   - A NaN filter bound made an oversold screen a no-op that admitted
     RSI 100, since NaN fails every comparison.

   Both optimizer findings also split the two solver paths, which now share
   one gate.

7. **A full-codebase audit, Pass 1 — the older quant runtime.** A fresh
   review found the modeling runtime is no longer the weak point; the
   remaining risk sat in backtesting, metrics, data normalization and the
   audit trail, which never gained the input/output contracts modeling now
   enforces. The temporal and integrity findings are fixed:

   - **Deleting a model's `manifest.json` bypassed every integrity check.**
     It is the package's commit point, so removing it — strictly easier than
     forging a hash inside it — made a tampered `model.joblib`
     **deserialize** where it had previously been refused. `joblib.load`
     executes code from the file it is handed.
   - **A negative strategy lookback read future prices.** Pandas treats a
     negative `pct_change` period as a *forward* window, so
     `momentum_timeseries(lookback=-20)` computed bar 25's signal from bar
     45's price. Not one of the eight strategies validated a parameter; all
     now share one contract.
   - **A sparse signal panel deleted trading days**, distorting annualized
     volatility by **32×** on identical prices.
   - **Intraday bars from different exchanges looked simultaneous** —
     London 15:00 BST and New York 15:00 EDT are five hours apart and were
     indexed identically. Intraday is canonical UTC now.
   - **A corrupted audit trail silently restarted at genesis** instead of
     refusing to extend a damaged chain.
   - **"Unknown" stopped meaning free**: a ticker with no volume data used
     to score `$0` market impact against `$3bn` for one with real data.

8. **Pass 2 — one shared numerical contract.** Around 40 of the audit's
   findings were a single problem wearing different clothes:
   `@validate_series` checked emptiness and nothing else, so the same invalid
   input gave `nan` from one metric, `+inf` from another, and an
   `IndexError` from a third. Worst of all, `max_drawdown` on a series
   containing one infinity returned **-1.70** — a drawdown that looks
   measured. `standard_quant_tools.numeric_contract` now states the rules
   once: infinities and all-NaN are rejected everywhere, partial NaN is
   deliberately still allowed (warm-up windows are legitimate), prices must
   be strictly *positive* rather than merely finite, and `periods_per_year`
   is validated wherever it multiplies. Cost primitives no longer accept
   negative rates, which returned negative costs — a backtest paid to trade.

9. **Passes 3–5 — solvers, schemas and audit policy.** A solver reporting
   success is not a valid answer: a covariance with condition number
   **3.8e+14** (full rank, so the rank check passed) produced a maximum
   weight of **197,838× capital** with `converged: True`, and a long-only
   `target_return=99.0` returned tidy weights achieving **0.2443**. Returned
   weights are now checked against their own constraints. The classic agent
   schemas gained the Literals and bounds the modeling schemas already had —
   including `sort_by`, where an unrecognized metric had been **silently
   ignored**, and a combinatorial budget on `param_grid`. The audit trail
   gained a fail-closed mode, refuses to replay a redacted record (redaction
   and exact replay are in tension by construction), and treats a
   previously-failed call as a first-class replay outcome.

If you have audit records written before this release, note that
`content_hash` values are not comparable across the change — see the format
note in [10_auditability.md](Documentation/10_auditability.md). The
tamper-evident record chain is unaffected and still verifies.

---

## Audit Trail & CLI (`standard_quant_tools.audit`, `sqt`)

Every call routed through `agent.tools.dispatch()` can produce an immutable JSONL decision record capturing its inputs, the market data it pulled (with content hashes), which execution path ran, and a hash of its output — enough to tell a stale/tampered cache apart from a genuine code change. Records are hash-chained (`prev_record_hash` / `record_hash`) **across every calendar day**, not just within one day's file — a first-of-day record commits to the previous active day's last hash via an independent chain index (`_chain_index.jsonl`), so deleting an entire day's file is detectable too, not just editing one record. `verify_audit_log_integrity()` checks one file; `verify_audit_trail_integrity()` checks the full cross-day trail. Every write (record or index entry) is `fsync`'d before its lock is released, and JSONL writes are guarded by a cross-process file lock. Nothing runs automatically; set `SQT_AUDIT_ENABLED=0` to disable record writes, and override the storage directory with `SQT_AUDIT_DIR`.

This is an *engineering control* — tamper detection, not tamper prevention or regulatory certification. See [Documentation/10_auditability.md](Documentation/10_auditability.md#auditability) for what it can and can't certify.

The `sqt` command (installed with the package) inspects and verifies these records by `request_id`:

```bash
sqt replay <request_id>              # re-run the recorded call, report whether data/output still match
sqt compare <request_id_a> <id_b>    # diff two records' status/output/timing/provenance/inputs
sqt report <request_id>              # pretty-print one record in full
sqt verify [--file PATH]             # check hash-chain integrity (full trail, or one file with --file)
sqt hold <date> [--reason TEXT]      # legal/retention hold on a calendar day, protects it from gc
sqt release-hold <date>              # remove a hold
sqt gc [--confirm]                   # delete day files past SQT_AUDIT_RETENTION_DAYS (dry-run by default)
sqt seal <date>                      # chmod a day file read-only (operational safeguard, not WORM)
sqt export --start D --end D --out F # zip a date range + manifest + standalone verifier for an auditor
sqt keygen [--out DIR]                # generate an Ed25519 keypair (local development only)
sqt anchor <date> [--key PATH]        # sign a checkpoint anchoring a day's chain endpoint
sqt verify --checkpoint <date> --pubkey PATH   # verify a checkpoint's signature (public key only)
```

`sqt replay` exits 0 if the output reproduced exactly, 1 on a confirmed mismatch, 2 if the record has no output hash to compare against. `sqt verify` exits 0 if clean, 1 if any problems are found. A dependency-free standalone verifier (`scripts/verify_audit_log.py`) is also available for external auditors who don't want to install the package. `SQT_AUDIT_REDACT_FIELDS` (comma-separated dotted field paths) replaces matching `input` fields — and, best-effort, an `error_message` that echoes one back — with a non-reversible content-hash placeholder before a record is written; set `SQT_AUDIT_REDACT_SALT` to a long random secret so that placeholder isn't brute-forceable offline for a small value space (an unset salt still works but logs a one-time warning).

**Checkpoint signing** (Ed25519, optional `pip install standard_quant_tools[signing]`) closes the one gap the hash chain can't on its own: a wholesale, internally-consistent rewrite of an entire day file. `checkpoint_and_sign()`/`sqt anchor` signs `{date, final_record_hash, index_hash}`; `verify_checkpoint_signature()`/`sqt verify --checkpoint` verifies it with only the public key. `generate_keypair()`/`sqt keygen` are for local development only — a real deployment should route signing through an HSM/KMS via a `signer` callback instead of a bare key file. Storage itself is behind a pluggable `AuditStorageBackend` (`LocalFilesystemBackend` is the only implementation shipped — a seam for a future WORM backend, not a WORM backend itself). See [Documentation/10_auditability.md](Documentation/10_auditability.md) for the full picture, including what none of this certifies by itself.

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
pytest tests/cpp_bindings/test_cpp_hurst.py -m benchmark -s -v

# C++ unit tests (requires _sqt_core built with SQT_BUILD_TESTS=ON)
ctest --test-dir build --config Release -V

# C++ performance benchmark binary (prints a timing table)
# Windows: build\tests\cpp\Release\bench_hurst.exe
# Linux / macOS: ./build/tests/cpp/bench_hurst

# Performance harnesses — minutes to run, so not part of the suite.
# Every figure in Development/optimization_plan.md comes from one of these.
SQT_NUM_THREADS=1 python tests/bench/bench_kernels.py   # per-kernel scaling, serial
python tests/bench/bench_kernels.py                     # ... and parallel
python tests/bench/bench_universe.py                    # 2,000-ticker shapes

# With coverage
pytest tests/ -m "not integration" --cov=src/standard_quant_tools
```

**2996 Python tests total** — 2994 passing, 2 skipped, with `_sqt_core` built. (Both skips are environmental: one needs `ANTHROPIC_API_KEY`, the other exercises a failure path that the input under test does not trigger.) Without the C++ extension the `tests/cpp_bindings/` files skip instead (they are gated on the extension being importable), and the rest still pass: every C++ path has a Python fallback, and both are held to the same contract (see [Correctness & Backend Parity](#correctness--backend-parity)).

`tests/` mirrors `src/standard_quant_tools/` — one directory per package (`agent/`, `analysis/`, `audit/`, `backtest/`, `data/`, `indicators/`, `metrics/`, `modeling/`, `portfolio/`, `screener/`), plus `core/` for cross-cutting suites, `cpp/` for the C++ gtest sources CMake compiles, and `cpp_bindings/` for the Python-side backend-parity tests. Run one group with `pytest tests/backtest`.

**9 C++ test executables** run via `ctest` (Hurst, indicators, cointegration, backtest, Monte Carlo, GARCH, signal state machines, rolling regression, plus a randomized-input cointegration fuzz harness) — **67,688** assertion-level checks between them, 50,234 of which come from the fuzz harness alone.

Note what the fuzz harness does *not* generate: non-finite inputs. Every shape it builds is
finite, which is why a NaN-handling defect once survived it alongside every other suite. The
NaN/Inf data contract is covered separately, in
`tests/cpp_bindings/test_cpp_nan_data_contract.py`.

> **If you build the extension yourself, don't override `CMAKE_CXX_FLAGS`.**
> Passing `-DCMAKE_CXX_FLAGS=...` *replaces* the project's defaults rather
> than appending to them, which silently drops `/EHsc` on MSVC. The result
> builds and links cleanly, emits only a C4530 warning that is easy to
> dismiss, and then takes an access violation the first time a kernel throws
> across the Python boundary. All build output also lands in
> `src/standard_quant_tools/`, so a second configure directory will overwrite
> the extension your main build produced. Use the documented
> `cmake -B build` invocation in
> [Development/build_guide.md](Development/build_guide.md).

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
| `Documentation/07_agent_tools.md` | Core 14 LLM tools, full 46-tool registry, Pydantic models, end-to-end agent loop |
| `Documentation/08_analysis.md` | Multi-factor regression, cointegration, PCA, Hurst exponent (incl. C++ acceleration) |
| `Documentation/09_advanced_agent_tools.md` | 31 advanced/supplementary/custom-signal/analytics/options/diagnostic tools: regime-adaptive (full-sample and leakage-free walk-forward), pair scanner, walk-forward, risk attribution, portfolio optimization, position sizer, fundamentals, optimization, advanced indicators, rolling beta, extended risk, backtest diagnostics, true portfolio simulation, pair trade backtest, robustness diagnostics, capacity report, data quality report, compact backtest result, volatility estimators, correlation analysis, Monte Carlo simulation, stress test, liquidity metrics, GARCH volatility forecast, Kalman hedge ratio, EVT tail risk, option pricing/Greeks, implied volatility |
| `Documentation/10_auditability.md` | Decision-record audit trail, replay verification (both tool registries), correlated logging, `sqt` CLI |
| `Documentation/11_data_quality.md` | Dataset provenance metadata, missing-bar/stale-price/price-jump detection |
| `Documentation/12_options.md` | Black-Scholes-Merton option pricing, Greeks, implied volatility (European options only) |
| `Documentation/13_agent_orchestration.md` | Tool-category taxonomy, the lightweight router, and the multi-agent orchestrator-workers architecture |
| `Documentation/14_polars_support.md` | Optional Polars interop (`pip install standard_quant_tools[polars]`): what's supported today, the conversion-boundary design, and the phased roadmap |
| `Documentation/15_modeling.md` | The separate 6-tool modeling runtime: 21-feature catalog, regression/classification targets, leakage-purged walk-forward validation, the content-addressed model registry, the model→backtest bridge, portfolio evaluation of OOS predictions, and what's explicitly deferred |
| `Development/build_guide.md` | C++ extension build instructions (Windows / Linux / macOS) |
| `Development/performance_insights.md` | Algorithmic analysis: which components benefit from C++ and by how much |

---

## Contributing

Bug reports, doc fixes, and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)
for the development workflow, code conventions, and PR expectations.

## Security

Found a security issue? Please don't open a public issue — see
[SECURITY.md](SECURITY.md) for how to report it privately.

## Changelog

Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) for the full text.

```
Copyright 2026 Karan Vora

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
