# Standard Quant Tools for AI Agents

A modular, agent-centric Python library for quantitative financial analysis. Designed to provide clean, structured data and robust error handling for AI agents (LLMs) and automated workflows.

## Installation

```bash
pip install .
# OR with poetry
poetry install
```

## Quick Start

```python
from standard_quant_tools.src.data.factory import DataFactory
from standard_quant_tools.src.indicators.trend import sma

# 1. Get Data
provider = DataFactory.get_provider("yfinance")
df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")

# 2. Calculate Indicators
df['SMA_20'] = sma(df['Close'], period=20)
print(df.tail())
```

---

## Module Reference

### 1. Data Module (`src.data`)

This module handles data fetching from various sources with a unified API.

**Factory Usage:**

```python
from standard_quant_tools.src.data.factory import DataFactory
provider = DataFactory.get_provider("yfinance")
```

**Methods:**

* **`get_ohlcv(symbol, start_date, end_date)`**
  * Returns: `pd.DataFrame` with `Open`, `High`, `Low`, `Close`, `Volume`.
  * Example:

        ```python
        df = provider.get_ohlcv("NVDA", "2023-01-01", "2024-01-01")
        ```

* **`get_financial_ratios(symbol)`**
  * Returns: `FinancialRatios` object (P/E, P/B, Debt/Eq, etc.).
  * Example:

        ```python
        ratios = provider.get_financial_ratios("MSFT")
        print(f"Forward P/E: {ratios.forward_pe}")
        ```

### 2. Technical Indicators (`src.indicators`)

Modular functions that accept pandas Series and return Series or DataFrames.

**Trend:**

```python
from standard_quant_tools.src.indicators.trend import sma, ema, macd

# Simple Moving Average
ma = sma(df['Close'], period=50)

# MACD (Returns DataFrame with MACD, Signal, Histogram)
macd_df = macd(df['Close'])
```

**Momentum:**

```python
from standard_quant_tools.src.indicators.momentum import rsi, stochastic_oscillator

# RSI
rsi_values = rsi(df['Close'], period=14)
```

**Volatility:**

```python
from standard_quant_tools.src.indicators.volatility import bollinger_bands, atr

# Bollinger Bands (Returns DataFrame with Upper, Middle, Lower)
bb = bollinger_bands(df['Close'])
```

### 3. Financial Metrics (`src.metrics`)

Calculate risk and return metrics for performance evaluation.

```python
from standard_quant_tools.src.metrics.return_metrics import cagr, cumulative_return
from standard_quant_tools.src.metrics.risk_metrics import sharpe_ratio, max_drawdown

# Assuming 'returns' is a pandas Series of daily percentage returns
total_ret = cumulative_return(returns)
sharpe = sharpe_ratio(returns, risk_free_rate=0.04)
mdd = max_drawdown(price_series)

print(f"Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}")
```

### 4. Regression & Analysis (`src.analysis`)

Advanced statistical analysis of asset behavior relative to a benchmark.

**Static Alpha/Beta:**

```python
from standard_quant_tools.src.analysis.regression import calculate_beta

# Calculate Beta of Asset vs Benchmark
# Returns dict: {'alpha': ..., 'beta': ..., 'r_squared': ...}
stats = calculate_beta(asset_returns, benchmark_returns)
print(f"Beta: {stats['beta']:.2f}, R-Squared: {stats['r_squared']:.2f}")
```

**Rolling Beta:**

```python
from standard_quant_tools.src.analysis.regression import rolling_beta

# See how risk exposure changes over time
rb = rolling_beta(asset_returns, benchmark_returns, window=60)
```

### 5. Backtesting Engine (`src.backtest`)

A fast, vectorised backtesting engine for validating strategies.

```python
from standard_quant_tools.src.backtest.engine import run_strategy

# 1. Define Signal (1 = Long, 0 = Cash/Flat)
signals = pd.Series(0, index=df.index)
signals[df['Close'] > df['SMA_200']] = 1

# 2. Run Backtest
results = run_strategy(df, signals, initial_capital=10000)

print(f"Final Equity: ${results['final_equity']:,.2f}")
print(f"Total Return: {results['total_return']:.2%}")
```

### 6. AI Agent Tools (`src.agent`)

**Ready-to-use tools** designed for valid function calling with LLMs (OpenAI, Anthropic, etc.). Inputs and outputs are strictly typed Pydantic models.

**Get Tool Definitions:**

```python
from standard_quant_tools.src.agent.tools import get_agent_tools

# Returns JSON schema for direct API usage
tools_schema = get_agent_tools()
```

**Tool 1: `analyze_stock_risk`**

* **Purpose**: Get a quick risk profile of a stock.
* **Input**: `AnalysisInput(symbol="TSLA", benchmark="SPY")`
* **Output**: JSON object with Alpha, Beta, Sharpe Ratio.

**Tool 2: `run_sma_backtest`**

* **Purpose**: Test a simple moving average crossover strategy.
* **Input**: `BacktestInput(...)`
* **Output**: Performance summary (Return, Volatility, Max Drawdown).

**Example Agent Call:**

```python
from standard_quant_tools.src.agent.tools import analyze_stock_risk
from standard_quant_tools.src.agent.models import AnalysisInput

input_data = AnalysisInput(symbol="GOOGL", benchmark="SPY", period="1y")
result = analyze_stock_risk(input_data)
# result is an AnalysisResult object, easily serialized to JSON
print(result.model_dump_json(indent=2))
```
