# Standard Quant Tools for AI Agents

A high-performance, modular Python library for quantitative financial analysis. Designed to give AI agents and automated workflows **clean structured data**, **mathematical accuracy**, and **robust error handling**.

## Key Features

- **High Performance** — Optional C++ extension (`_sqt_core`) for Hurst/rolling Hurst (20–80×), RSI/ADX/Parabolic SAR (10–30×); NumPy single-pass ATR (5.6×); BLAS-backed portfolio covariance; vectorized backtesting engine; async concurrent data fetching; persistent Parquet disk cache; `ProcessPoolExecutor` screener and parallel backtest grid
- **Agent-First Design** — All tools return Pydantic models; 17 LLM-callable tools with OpenAI/Anthropic function-calling schemas; descriptive errors for self-correction
- **Comprehensive Coverage** — 14 indicators, 10 risk/return metrics, 12 analysis functions, portfolio analysis, stock screener, 4 backtest strategies + parameter grid search
- **Robust Infrastructure** — Retry logic with exponential backoff, TTL + Parquet caching, custom exception hierarchy, `@validate_series` decorator, optional C++/scipy/numba graceful fallback

---

## Installation

```bash
pip install .
# or
poetry install
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `yfinance`, `numba`, `aiohttp`, `cachetools`, `pydantic`, `statsmodels`, `pyarrow`

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

**Caching:** Historical OHLCV calls are saved as Parquet files under `~/.cache/standard_quant_tools/ohlcv/`. Subsequent calls — even from a new Python process — load from disk rather than the network. Override the cache directory with `SQT_CACHE_DIR`.

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
| `stochastic_oscillator(high, low, close)` | Stochastic %K and %D | Pandas rolling |

**Volatility**

| Function | Description | Performance |
|---|---|---|
| `bollinger_bands(series, period, num_std)` | Upper / Middle / Lower bands | Pandas rolling |
| `atr(high, low, close, period)` | Average True Range | **NumPy single-pass** (5.6× vs `pd.concat`) |

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
| `information_ratio(returns, benchmark)` | Active return / tracking error |
| `treynor_ratio(returns, benchmark)` | Excess return / beta |
| `drawdown_series(series)` | Full drawdown time series |

---

### Analysis (`standard_quant_tools.analysis`)

10 functions across four areas. All pure NumPy / Pandas — no heavy solver dependencies.

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

**Beta filter optimisation:** When `beta_max` / `beta_min` filters are active, SPY data is fetched **once per call** (not once per ticker), eliminating N−1 redundant HTTP round-trips for an N-ticker beta screen.

```python
result = screen_stocks(sp500_tickers, filters={...}, n_workers=8)
```

---

### AI Agent Tools (`standard_quant_tools.agent`)

12 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas.

```python
from standard_quant_tools.agent.tools import (
    get_agent_tools, analyze_stock_risk,
    run_factor_regression, run_cointegration_test,
    run_pca_analysis, run_hurst_analysis,
)
from standard_quant_tools.agent.models import (
    AnalysisInput, FactorRegressionInput,
    CointegrationInput, PCAInput, HurstInput,
)

# Get tool schemas for your LLM
tools = get_agent_tools()  # 12 tools ready for function calling

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

**Original tools (12):** `run_sma_backtest`, `run_rsi_backtest`, `run_macd_backtest`, `run_bollinger_backtest`, `analyze_stock_risk`, `get_technical_analysis`, `get_portfolio_analysis`, `run_screener`, `run_factor_regression`, `run_cointegration_test`, `run_pca_analysis`, `run_hurst_analysis`

**Advanced agentic tools (5):** `run_regime_adaptive_backtest`, `scan_pairs`, `run_walk_forward_backtest`, `get_portfolio_risk_attribution`, `get_position_size`

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

The rolling Hurst gain is the most significant: rather than re-entering Python for every bar, the entire sliding-window pass runs in one C++ function. RSI/ADX/PSAR gains are most visible when numba is unavailable (e.g. NumPy 2.x), where the alternative is an interpreted Python loop.

> These are projected figures based on algorithmic analysis of loop iterations vs. compiled throughput. The benchmark suite (`tests/cpp/bench_hurst.cpp` and `pytest -m benchmark`) confirms actual numbers once the extension is built. See [Development/build_guide.md](Development/build_guide.md) for build instructions.

---

### Python-Level Optimisations

Confirmed benchmarks on a 2 000-bar series (Python 3.12, NumPy 2.4):

| Optimisation | Before | After | Speedup | Notes |
|---|---|---|---|---|
| ATR true range | 2.8 ms (`pd.concat` + `.max`) | 0.49 ms (`np.maximum`) | **5.6×** | Single-pass; eliminates 3 Series + concat |
| Trade log serialization | 31 ms (`iterrows`, 500 trades) | 3.6 ms (`to_dict`) | **~9×** | Vectorized dict conversion |
| CVaR computation | 0.83 ms (two-pass) | 0.44 ms (one-pass) | **1.9×** | Single `np.percentile` + boolean mask |
| SPY beta screen | N HTTP requests | 1 HTTP request | **N×** | SPY fetched once per `screen_stocks()` call |
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

**544 Python unit tests** (494 passing; 50 skipped pending C++ build) · **6 integration tests** · **6 benchmark tests** · **35 C++ unit tests** (19 Hurst + 16 indicators)

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
| `Documentation/07_agent_tools.md` | Original 12 LLM tools, Pydantic models, end-to-end agent loop |
| `Documentation/08_analysis.md` | Multi-factor regression, cointegration, PCA, Hurst exponent (incl. C++ acceleration) |
| `Documentation/09_advanced_agent_tools.md` | 5 advanced tools: regime-adaptive, pair scanner, walk-forward, risk attribution, position sizer |
| `Development/build_guide.md` | C++ extension build instructions (Windows / Linux / macOS) |
| `Development/performance_insights.md` | Algorithmic analysis: which components benefit from C++ and by how much |
