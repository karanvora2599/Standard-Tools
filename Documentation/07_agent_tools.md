# Agent Tools

The agent module wraps every major capability in Pydantic-typed functions designed for LLM function calling. Each tool accepts a single Pydantic input model and returns a Pydantic output model — both are directly JSON-serializable.

---

## Tool Registry

```python
from standard_quant_tools.agent.tools import get_agent_tools

tools = get_agent_tools()
# Returns a list of 8 dicts in OpenAI / Anthropic function-calling format.
# Each dict: {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}
print(len(tools))   # 8
print(tools[0]['function']['name'])   # 'run_sma_backtest'
```

Pass directly to the OpenAI client:

```python
import openai
from standard_quant_tools.agent.tools import get_agent_tools

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Backtest a 10/30 SMA crossover on TSLA from 2021 to 2024"}],
    tools=get_agent_tools(),
    tool_choice="auto",
)
```

Pass to the Anthropic client:

```python
import anthropic
from standard_quant_tools.agent.tools import get_agent_tools

client = anthropic.Anthropic()
# Anthropic uses "tools" key directly — same schema format
response = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    tools=get_agent_tools(),
    messages=[{"role": "user", "content": "Screen FAANG stocks for value opportunities"}],
)
```

---

## Tool 1 — SMA Crossover Backtest

```python
from standard_quant_tools.agent.tools import run_sma_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_sma_backtest(BacktestInput(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 30},
    initial_capital=10_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
))

print(f"Total Return   : {result.total_return:.1%}")
print(f"Sharpe Ratio   : {result.sharpe_ratio:.2f}")
print(f"Max Drawdown   : {result.max_drawdown:.1%}")
print(f"Win Rate       : {result.win_rate:.1%}")
print(f"Num Trades     : {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast_period` | int | 10 | Fast SMA window |
| `slow_period` | int | 30 | Slow SMA window |

**Signal logic:** Long (`1`) when fast SMA > slow SMA; flat (`0`) otherwise.

---

## Tool 2 — RSI Mean Reversion Backtest

```python
from standard_quant_tools.agent.tools import run_rsi_backtest

result = run_rsi_backtest(BacktestInput(
    symbol="SPY",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="rsi_mean_reversion",
    parameters={"period": 14, "oversold": 30, "overbought": 70},
))

print(f"Calmar Ratio   : {result.calmar_ratio:.2f}")
print(f"Profit Factor  : {result.profit_factor:.2f}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 14 | RSI lookback period |
| `oversold` | float | 30 | Enter long below this RSI level |
| `overbought` | float | 70 | Exit position above this RSI level |

**Signal logic:** Enter long when RSI drops below `oversold`; hold until RSI rises above `overbought`. Stateful — position persists between entry and exit conditions.

---

## Tool 3 — MACD Crossover Backtest

```python
from standard_quant_tools.agent.tools import run_macd_backtest

result = run_macd_backtest(BacktestInput(
    symbol="QQQ",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="macd_crossover",
    parameters={"fast": 12, "slow": 26, "signal": 9},
))
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast` | int | 12 | Fast EMA period |
| `slow` | int | 26 | Slow EMA period |
| `signal` | int | 9 | Signal line EMA period |

**Signal logic:** Long when MACD line > signal line; flat otherwise.

---

## Tool 4 — Bollinger Band Mean Reversion Backtest

```python
from standard_quant_tools.agent.tools import run_bollinger_backtest

result = run_bollinger_backtest(BacktestInput(
    symbol="GLD",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="bollinger_reversion",
    parameters={"period": 20, "num_std": 2.0},
))
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | Bollinger Band SMA window |
| `num_std` | float | 2.0 | Band width in standard deviations |

**Signal logic:** Enter long when price closes at or below the lower band; exit when price reaches the middle band (SMA). Stateful hold between entry and exit.

---

## BacktestInput / BacktestResult Models

```python
from standard_quant_tools.agent.models import BacktestInput, BacktestResult, Trade

# Full BacktestInput fields:
inp = BacktestInput(
    symbol="TSLA",
    start_date="2021-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 30},
    initial_capital=50_000,
    commission_pct=0.001,    # 0.1% per trade side
    slippage_pct=0.0005,     # 0.05% per trade side
)

# BacktestResult fields:
result: BacktestResult = run_sma_backtest(inp)
result.total_return           # float — net return fraction
result.annualized_volatility  # float
result.sharpe_ratio           # float
result.sortino_ratio          # float
result.max_drawdown           # float (negative)
result.calmar_ratio           # float
result.win_rate               # float (0–1)
result.profit_factor          # float (gross profit / gross loss)
result.num_trades             # int
result.avg_trade_return_pct   # float
result.final_equity           # float
result.equity_curve           # List[float] — day-by-day portfolio value
result.trade_log              # Optional[List[Trade]]

# Serialize for LLM consumption:
import json
print(json.dumps(result.model_dump(), indent=2))

# Inspect trade log:
if result.trade_log:
    for trade in result.trade_log[:5]:
        print(f"{trade.entry_date} → {trade.exit_date} | {trade.return_pct:.2%}")
```

---

## Tool 5 — Stock Risk Analysis

```python
from standard_quant_tools.agent.tools import analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

result = analyze_stock_risk(AnalysisInput(
    symbol="NVDA",
    benchmark="SPY",   # default
    period="1y",       # '6mo', '1y', '2y'
))

print(f"Alpha            : {result.alpha:.6f}")
print(f"Beta             : {result.beta:.4f}")
print(f"R²               : {result.r_squared:.4f}")
print(f"Sharpe Ratio     : {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio    : {result.sortino_ratio:.2f}")
print(f"Max Drawdown     : {result.max_drawdown:.1%}")
print(f"VaR (95%)        : {result.var_95:.4f}")
print(f"CVaR (95%)       : {result.cvar_95:.4f}")
print(f"Information Ratio: {result.information_ratio:.2f}")
```

**Period strings:** `'6mo'`, `'1y'`, `'2y'`, `'3y'`, or `'Nd'` for N days.

---

## Tool 6 — Technical Analysis

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

result = get_technical_analysis(TechnicalInput(
    symbol="MSFT",
    start_date="2023-01-01",
    end_date="2024-01-01",
    indicators=["rsi", "macd", "bollinger", "adx", "obv", "sma"],
))

print(f"Last close: {result.last_close}")
print(f"RSI(14)   : {result.last_values.get('rsi_14')}")
print(f"ADX       : {result.last_values.get('adx')}")
print(f"MACD bull : {result.signals.get('macd_bullish')}")
print(f"Strong trend: {result.signals.get('strong_trend')}")
```

**Available indicators:**

| Key | Values returned | Signals generated |
|---|---|---|
| `sma` | `sma_20`, `sma_50`, `sma_200` | `price_above_sma_N` |
| `ema` | `ema_12`, `ema_26` | — |
| `macd` | `macd`, `macd_signal`, `macd_histogram` | `macd_bullish` |
| `rsi` | `rsi_14` | `rsi_oversold`, `rsi_overbought` |
| `stochastic` | `stoch_k`, `stoch_d` | `stoch_oversold` |
| `bollinger` | `bb_upper`, `bb_middle`, `bb_lower` | `price_near_lower_band`, `price_near_upper_band` |
| `atr` | `atr_14` | — |
| `obv` | `obv` | `obv_rising` |
| `vwap` | `vwap` | `price_above_vwap` |
| `adx` | `adx`, `di_plus`, `di_minus` | `strong_trend`, `bullish_di` |
| `williams_r` | `williams_r` | `williams_r_oversold`, `williams_r_overbought` |

---

## Tool 7 — Portfolio Analysis

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

result = get_portfolio_analysis(PortfolioInput(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],  # must sum to 1.0
    start_date="2023-01-01",
    end_date="2024-01-01",
    benchmark="SPY",
))

print(f"Annualized Return : {result.annualized_return:.2%}")
print(f"Sharpe Ratio      : {result.sharpe_ratio:.2f}")
print(f"Max Drawdown      : {result.max_drawdown:.2%}")
print(f"Information Ratio : {result.information_ratio:.2f}")

# Correlation matrix (nested dict of ticker → ticker → float)
for ticker, row in result.correlation_matrix.items():
    for other, corr in row.items():
        if ticker < other:
            print(f"{ticker} / {other}: {corr:.2f}")
```

**All metric fields on `PortfolioResult`:**

```python
result.tickers               # List[str]
result.weights               # List[float]
result.annualized_return     # float
result.annualized_volatility # float
result.sharpe_ratio          # float
result.sortino_ratio         # float
result.max_drawdown          # float (negative)
result.calmar_ratio          # float
result.var_95                # float
result.cvar_95               # float
result.information_ratio     # float
result.total_return          # float
result.correlation_matrix    # Dict[str, Dict[str, float]]
```

---

## Tool 8 — Stock Screener

```python
from standard_quant_tools.agent.tools import run_screener
from standard_quant_tools.agent.models import ScreenerInput

result = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN"],
    filters={
        "pe_ratio_max": 35,
        "rsi_max": 55,
        "beta_max": 1.5,
        "market_cap_min": 100_000_000_000,  # $100B
    },
    sort_by="rsi_14",
    ascending=True,  # most oversold first
))

print(f"Passed: {result.num_passed} / {len(result.tickers_passed)}")
print(f"Tickers: {result.tickers_passed}")
for row in result.results:
    print(row)

# Serialize for LLM:
import json
print(json.dumps(result.model_dump(), indent=2))
```

---

## End-to-End LLM Agent Example

```python
import json
import anthropic
from standard_quant_tools.agent.tools import get_agent_tools, run_sma_backtest, run_screener
from standard_quant_tools.agent.models import BacktestInput, ScreenerInput

TOOL_MAP = {
    "run_sma_backtest": run_sma_backtest,
    "run_rsi_backtest": run_rsi_backtest,
    "run_macd_backtest": run_macd_backtest,
    "run_bollinger_backtest": run_bollinger_backtest,
    "analyze_stock_risk": analyze_stock_risk,
    "get_technical_analysis": get_technical_analysis,
    "get_portfolio_analysis": get_portfolio_analysis,
    "run_screener": run_screener,
}

INPUT_MAP = {
    "run_sma_backtest": BacktestInput,
    "run_rsi_backtest": BacktestInput,
    "run_macd_backtest": BacktestInput,
    "run_bollinger_backtest": BacktestInput,
    "analyze_stock_risk": AnalysisInput,
    "get_technical_analysis": TechnicalInput,
    "get_portfolio_analysis": PortfolioInput,
    "run_screener": ScreenerInput,
}

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Screen FAANG for PE < 30 and RSI < 50, then backtest an SMA crossover on the top result"}]

while True:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        tools=get_agent_tools(),
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        print(response.content[0].text)
        break

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            fn = TOOL_MAP[block.name]
            model_cls = INPUT_MAP[block.name]
            inp = model_cls(**block.input)
            result = fn(inp)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result.model_dump()),
            })

    messages.append({"role": "user", "content": tool_results})
```

---

## Model Summary

| Input Model | Required fields | Optional fields |
|---|---|---|
| `BacktestInput` | `symbol`, `start_date`, `end_date`, `strategy_type` | `parameters`, `initial_capital`, `commission_pct`, `slippage_pct` |
| `AnalysisInput` | `symbol` | `benchmark` (SPY), `period` (1y) |
| `TechnicalInput` | `symbol`, `start_date`, `end_date` | `indicators` (default: rsi/macd/bollinger/atr) |
| `PortfolioInput` | `tickers`, `weights`, `start_date`, `end_date` | `benchmark` (SPY) |
| `ScreenerInput` | `tickers`, `filters` | `start_date`, `end_date`, `sort_by`, `ascending` |
