# Standard Quant Tools for AI Agents

A high-performance, modular, and agent-centric Python library for quantitative financial analysis. Designed to provide **clean, structured data** and **robust error handling** for AI agents (LLMs) and automated HFT-like workflows.

## Key Features

- **High Performance**:
  - **Numba Acceleration**: JIT-compiled technical indicators (e.g., RSI) for low-latency calculation.
  - **NumPy Optimization**: OLS regression using linear algebra (`numpy.linalg`) instead of heavy statistical packages, offering ~100x speedup.
  - **AsyncIO Support**: Non-blocking data fetching for high-throughput applications.
  - **Smart Caching**: In-memory LRU caching to prevent redundant API calls.
- **Agent-First Design**:
  - **Structured Outputs**: All tools return Pydantic models or clean dictionaries, not arbitrary objects.
  - **Error Recovery**: Descriptive error messages designed for LLM self-correction.
- **Robustness**:
  - **Retry Logic**: Automatic retries with exponential backoff for network resilience.
  - **Custom Exceptions**: Granular error handling with `QuantError`, `DataNotFoundError`, etc.
  - **Input Validation**: Decorators (`@validate_series`) ensure data integrity before calculation.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/standard-quant-tools.git
cd standard-quant-tools

# Install dependencies (ensure you have poetry or pip)
pip install . 
# Or with Poetry
poetry install
```

**Requirements**:

- Python 3.10+
- `pandas`, `numpy`, `yfinance`
- `numba` (Performance)
- `aiohttp` (Async I/O)
- `cachetools` (Caching)

---

## Quick Start

### 1. Synchronous Data Fetching

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators.trend import sma

# Initialize Provider (Free Tier YFinance by default)
provider = DataFactory.get_provider("yfinance")

# Fetch Data (Cached automatically)
df = provider.get_ohlcv("NVDA", "2023-01-01", "2024-01-01")

# Calculate SMA
df['SMA_20'] = sma(df['Close'], period=20)
print(df.tail())
```

### 2. High-Performance Async Data Fetching

For building responsive applications or fetching multiple tickers in parallel:

```python
import asyncio
from standard_quant_tools.data.factory import DataFactory

async def fetch_portfolio(tickers):
    provider = DataFactory.get_provider("yfinance")
    tasks = [
        provider.get_ohlcv_async(ticker, "2023-01-01", "2024-01-01") 
        for ticker in tickers
    ]
    results = await asyncio.gather(*tasks)
    return results

# Run async loop
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN"]
data_frames = asyncio.run(fetch_portfolio(tickers))
```

### 3. Robust Error Handling

The library uses a custom exception hierarchy for precise control.

```python
from standard_quant_tools.error import DataNotFoundError, InvalidSymbolError
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider("yfinance")

try:
    df = provider.get_ohlcv("INVALID_TICKER", "2023-01-01", "2024-01-01")
except DataNotFoundError:
    print("Ticker not found! Please check the symbol.")
except InvalidSymbolError:
    print("Symbol format is incorrect.")
```

---

## comprehensive Module Reference

### 1. Data (`src.data`)

Handles data fetching with built-in resilience.

#### `get_ohlcv(symbol, start_date, end_date, interval='1d')`

Fetches historical Open-High-Low-Close-Volume data.

- **Returns**: `pd.DataFrame` with standard columns.
- **Example**:

  ```python
  df = provider.get_ohlcv("TSLA", "2023-01-01", "2024-01-01")
  ```

#### `get_ohlcv_async(symbol, start_date, end_date, interval='1d')`

Non-blocking version of `get_ohlcv`. Ideal for batch processing.

- **Returns**: `Awaitable[pd.DataFrame]`

#### `get_ticker_info(symbol)`

Fetches company metadata.

- **Returns**: `TickerInfo` (Pydantic model: name, sector, industry, etc.)
- **Example**:

  ```python
  info = provider.get_ticker_info("AAPL")
  print(f"{info.name} is in {info.sector}")
  ```

#### `get_financial_ratios(symbol)`

Fetches key fundamental ratios.

- **Returns**: `FinancialRatios` (P/E, P/B, Debt/Eq, etc.)
- **Example**:

  ```python
  ratios = provider.get_financial_ratios("MSFT")
  print(f"Forward P/E: {ratios.forward_pe}")
  ```

### 2. Technical Indicators (`src.indicators`)

Optimized implementations of standard indicators.

#### Trend (`standard_quant_tools.indicators.trend`)

- **`sma(series, period)`**: Simple Moving Average.
- **`ema(series, period)`**: Exponential Moving Average.
- **`macd(series, fast=12, slow=26, signal=9)`**: Moving Average Convergence Divergence.
  - **Returns**: DataFrame with `MACD`, `Signal`, `Histogram`.

  ```python
  from standard_quant_tools.indicators.trend import macd
  macd_df = macd(df['Close'])
  ```

#### Momentum (`standard_quant_tools.indicators.momentum`)

- **`rsi(series, period=14)`**: Relative Strength Index.
  - **Optimization**: Uses **Numba JIT** for massive speedup on large arrays.

  ```python
  from standard_quant_tools.indicators.momentum import rsi
  rsi_vals = rsi(df['Close'])
  ```

- **`stochastic_oscillator(high, low, close)`**: Stochastic Oscillator.
  - **Returns**: DataFrame with `Stoch_K`, `Stoch_D`.

#### Volatility (`standard_quant_tools.indicators.volatility`)

- **`bollinger_bands(series, period=20, num_std=2)`**: Bollinger Bands.
  - **Returns**: DataFrame with `BB_Upper`, `BB_Middle`, `BB_Lower`.
- **`atr(high, low, close, period=14)`**: Average True Range.

### 3. Financial Metrics (`src.metrics`)

Standard performance metrics for portfolio evaluation.

#### Return Metrics (`standard_quant_tools.metrics.return_metrics`)

- **`cumulative_return(series)`**: Total return over the period.
- **`cagr(series)`**: Compound Annual Growth Rate.
- **`annualized_volatility(returns)`**: Standard deviation scaled to year.

#### Risk Metrics (`standard_quant_tools.metrics.risk_metrics`)

- **`sharpe_ratio(returns, risk_free_rate)`**: Excess return per unit of risk.
- **`sortino_ratio(returns, risk_free_rate)`**: Excess return per unit of *downside* risk.
- **`max_drawdown(series)`**: Maximum peak-to-trough decline.

```python
from standard_quant_tools.metrics.risk_metrics import sharpe_ratio, max_drawdown
sr = sharpe_ratio(daily_returns)
mdd = max_drawdown(price_series)
```

### 4. Regression & Analysis (`src.analysis`)

**High-Performance Regression Module**.

#### `calculate_beta(asset_returns, benchmark_returns)`

Calculates static Alpha and Beta using **NumPy** linear algebra optimization (~100x faster than statsmodels).

- **Returns**: Dict `{'alpha', 'beta', 'r_squared'}`.

```python
from standard_quant_tools.analysis.regression import calculate_beta
stats = calculate_beta(stock_returns, market_returns)
```

#### `rolling_beta(asset_returns, benchmark_returns, window=60)`

Calculates Beta over a rolling window to track changing risk exposure.

- **Returns**: DataFrame with `Rolling_Beta`.

### 5. Backtesting Engine (`src.backtest`)

A **vectorized** backtesting engine. It calculates equity curves entire arrays at once, avoiding slow loops.

#### `run_strategy(price_data, signal_series, initial_capital)`

- **Inputs**:
  - `price_data`: DataFrame with 'Close'.
  - `signal_series`: Series of 1 (Long), 0 (Flat), -1 (Short).
- **Returns**: Dictionary with `total_return`, `final_equity`, `max_drawdown`, `equity_curve`.

```python
from standard_quant_tools.backtest.engine import run_strategy
# Strategy: Long if Close > SMA(50)
signals = (df['Close'] > df['SMA_50']).astype(int)
results = run_strategy(df, signals, initial_capital=10000)
```

### 6. AI Agent Tools (`src.agent`)

**LLM-Ready Functions**. Wrappers designed for OpenAI Function Calling.

#### `get_agent_tools()`

Returns a JSON-serializable list of tool definitions (schema) that can be passed directly to an LLM API.

#### `analyze_stock_risk`

- **Purpose**: Quick risk profile.
- **Input Model**: `AnalysisInput(symbol="...", benchmark="...")`
- **Output Model**: `AnalysisResult(alpha=..., beta=..., sharpe=...)`

#### `run_sma_backtest`

- **Purpose**: Test a moving average crossover strategy.
- **Input Model**: `BacktestInput(symbol="...", strategy_type="sma_crossover", ...)`
- **Output Model**: `BacktestResult`

**Example Usage**:

```python
from standard_quant_tools.agent.tools import analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

input_data = AnalysisInput(symbol="GOOGL", benchmark="SPY", period="1y")
result = analyze_stock_risk(input_data)
# result.model_dump_json() provides clean JSON for the agent
```
