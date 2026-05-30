# Standard Quant Tools for AI Agents

A high-performance, modular Python library for quantitative financial analysis. Designed to give AI agents and automated workflows **clean structured data**, **mathematical accuracy**, and **robust error handling**.

## Key Features

- **High Performance** — Numba JIT for RSI, ADX, Parabolic SAR; NumPy BLAS-backed portfolio covariance; vectorized backtesting engine; async concurrent data fetching
- **Agent-First Design** — All tools return Pydantic models; 8 LLM-callable tools with OpenAI/Anthropic function-calling schemas; descriptive errors for self-correction
- **Comprehensive Coverage** — 14 indicators, 10 risk/return metrics, portfolio analysis, stock screener, 4 backtest strategies, regression analysis
- **Robust Infrastructure** — Retry logic with exponential backoff, TTL caching, custom exception hierarchy, `@validate_series` decorator, optional scipy/numba graceful fallback

---

## Installation

```bash
pip install .
# or
poetry install
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `yfinance`, `numba`, `aiohttp`, `cachetools`, `pydantic`

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

### Technical Indicators (`standard_quant_tools.indicators`)

**Trend**

| Function | Description | Performance |
|---|---|---|
| `sma(series, period)` | Simple Moving Average | Pandas rolling |
| `ema(series, period)` | Exponential Moving Average | Pandas EWM |
| `macd(series, fast, slow, signal)` | MACD + Signal + Histogram | Pandas EWM |
| `adx(high, low, close, period)` | ADX + DI+ + DI− | **Numba JIT** (Wilder smoothing) |
| `parabolic_sar(high, low)` | Parabolic SAR + Trend direction | **Numba JIT** |
| `williams_r(high, low, close, period)` | Williams %R oscillator | Pandas rolling |

**Momentum**

| Function | Description | Performance |
|---|---|---|
| `rsi(series, period)` | RSI (Wilder's smoothing) | **Numba JIT** |
| `stochastic_oscillator(high, low, close)` | Stochastic %K and %D | Pandas rolling |

**Volatility**

| Function | Description |
|---|---|
| `bollinger_bands(series, period, num_std)` | Upper / Middle / Lower bands |
| `atr(high, low, close, period)` | Average True Range |

**Volume**

| Function | Description |
|---|---|
| `obv(close, volume)` | On Balance Volume |
| `vwap(high, low, close, volume, period)` | VWAP (cumulative or rolling) |
| `mfi(high, low, close, volume, period)` | Money Flow Index |

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

### Analysis (`standard_quant_tools.analysis`)

```python
from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta

stats = calculate_beta(asset_returns, benchmark_returns)
# {'alpha': ..., 'beta': ..., 'r_squared': ...}  — NumPy OLS, ~100x faster than statsmodels

rolling_df = rolling_beta(asset_returns, benchmark_returns, window=60)
# DataFrame with 'Rolling_Beta' column
```

### Backtesting (`standard_quant_tools.backtest`)

Vectorized engine with transaction costs, trade log, and full metric output.

```python
from standard_quant_tools.backtest.engine import run_strategy
import numpy as np

signals = (df['Close'] > sma(df['Close'], 50)).astype(int)

result = run_strategy(
    df, signals,
    initial_capital=10_000,
    commission_pct=0.001,   # 0.1% per trade
    slippage_pct=0.0005,    # 0.05% per trade
    include_trade_log=True,
)

print(f"Sharpe: {result['sharpe_ratio']}")
print(f"Calmar: {result['calmar_ratio']}")
print(f"Win Rate: {result['win_rate']:.1%}")
print(f"Trades: {result['num_trades']}")
```

**Output keys:** `final_equity`, `total_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `num_trades`, `avg_trade_return_pct`, `equity_curve`, `trade_log`

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
print(result)
```

**Available filters:** `pe_ratio_max`, `pb_ratio_max`, `debt_equity_max`, `roe_min`, `profit_margin_min`, `div_yield_min`, `market_cap_min`, `rsi_max`, `rsi_min`, `price_above_sma`, `price_below_sma`, `beta_max`, `beta_min`

### AI Agent Tools (`standard_quant_tools.agent`)

8 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas.

```python
from standard_quant_tools.agent.tools import get_agent_tools, analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

# Get tool schemas for your LLM
tools = get_agent_tools()  # 8 tools ready for function calling

# Call directly
result = analyze_stock_risk(AnalysisInput(symbol='NVDA', benchmark='SPY', period='1y'))
print(result.model_dump_json(indent=2))
```

**Available tools:** `run_sma_backtest`, `run_rsi_backtest`, `run_macd_backtest`, `run_bollinger_backtest`, `analyze_stock_risk`, `get_technical_analysis`, `get_portfolio_analysis`, `run_screener`

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
# Unit tests (no network)
pytest tests/ -m "not integration"

# Integration tests (requires internet)
pytest tests/ -m integration

# With coverage
pytest tests/ -m "not integration" --cov=src/standard_quant_tools
```

See `Documentation/` for detailed guides on each module.
