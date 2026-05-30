# Agent Tools

The agent module exposes every major library capability as **Pydantic-typed, LLM-callable functions**. Each tool takes one input model, does real computation, and returns one output model — both are directly JSON-serializable for function calling.

**Why use the agent module instead of calling the library directly?**

| Direct call | Agent tool |
|---|---|
| Returns `pd.DataFrame`, `pd.Series`, plain dicts | Returns Pydantic model — serializable without post-processing |
| Requires knowing the full library API | Single function per capability |
| Multiple imports, multiple calls | One import, one call |
| No schema for LLM | Schema auto-generated from Pydantic model |

---

## Quick Start

```python
from standard_quant_tools.agent.tools import (
    get_agent_tools,
    run_sma_backtest, run_rsi_backtest, run_macd_backtest, run_bollinger_backtest,
    analyze_stock_risk,
    get_technical_analysis,
    get_portfolio_analysis,
    run_screener,
)
from standard_quant_tools.agent.models import (
    BacktestInput, AnalysisInput, TechnicalInput, PortfolioInput, ScreenerInput,
)

# Call any tool directly
from standard_quant_tools.agent.tools import analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

result = analyze_stock_risk(AnalysisInput(symbol="AAPL"))
print(result.model_dump_json(indent=2))
```

---

## Tool Registry

`get_agent_tools()` returns 8 tool definitions in the format both OpenAI and Anthropic expect. The schemas are derived automatically from Pydantic — no manual JSON authoring.

```python
from standard_quant_tools.agent.tools import get_agent_tools

tools = get_agent_tools()
print(len(tools))  # 8

# Each tool follows the OpenAI function-calling format:
# {"type": "function", "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}
for t in tools:
    print(t["function"]["name"], "—", t["function"]["description"])
# run_sma_backtest — SMA crossover backtest.
# run_rsi_backtest — RSI mean-reversion backtest.
# run_macd_backtest — MACD crossover backtest.
# run_bollinger_backtest — Bollinger Band mean-reversion backtest.
# analyze_stock_risk — Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.
# get_technical_analysis — Compute configurable technical indicators.
# get_portfolio_analysis — Multi-asset portfolio metrics.
# run_screener — Filter a stock universe by fundamental and technical criteria.

# Inspect the parameter schema for any tool:
import json
print(json.dumps(tools[4]["function"]["parameters"], indent=2))
```

### Wiring up OpenAI

```python
import json
import openai
from standard_quant_tools.agent.tools import (
    get_agent_tools,
    run_sma_backtest, run_rsi_backtest, run_macd_backtest, run_bollinger_backtest,
    analyze_stock_risk, get_technical_analysis, get_portfolio_analysis, run_screener,
)
from standard_quant_tools.agent.models import (
    BacktestInput, AnalysisInput, TechnicalInput, PortfolioInput, ScreenerInput,
)

TOOL_FN   = {
    "run_sma_backtest":      run_sma_backtest,
    "run_rsi_backtest":      run_rsi_backtest,
    "run_macd_backtest":     run_macd_backtest,
    "run_bollinger_backtest":run_bollinger_backtest,
    "analyze_stock_risk":    analyze_stock_risk,
    "get_technical_analysis":get_technical_analysis,
    "get_portfolio_analysis":get_portfolio_analysis,
    "run_screener":          run_screener,
}
INPUT_MODEL = {
    "run_sma_backtest":      BacktestInput,
    "run_rsi_backtest":      BacktestInput,
    "run_macd_backtest":     BacktestInput,
    "run_bollinger_backtest":BacktestInput,
    "analyze_stock_risk":    AnalysisInput,
    "get_technical_analysis":TechnicalInput,
    "get_portfolio_analysis":PortfolioInput,
    "run_screener":          ScreenerInput,
}

client   = openai.OpenAI()
messages = [{"role": "user", "content": "What is NVDA's current risk profile vs SPY?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=get_agent_tools(),
        tool_choice="auto",
    )
    msg = response.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print(msg.content)
        break

    for tc in msg.tool_calls:
        fn        = TOOL_FN[tc.function.name]
        model_cls = INPUT_MODEL[tc.function.name]
        result    = fn(model_cls(**json.loads(tc.function.arguments)))
        messages.append({
            "role":         "tool",
            "tool_call_id": tc.id,
            "content":      result.model_dump_json(),
        })
```

### Wiring up Anthropic

```python
import json
import anthropic
from standard_quant_tools.agent.tools import get_agent_tools, run_screener, run_sma_backtest
from standard_quant_tools.agent.models import ScreenerInput, BacktestInput

# (build TOOL_FN and INPUT_MODEL the same way as above)

client   = anthropic.Anthropic()
messages = [{"role": "user", "content": "Screen mega-cap tech for RSI < 45, then backtest SMA 10/50 on any that pass"}]

while True:
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        tools=get_agent_tools(),
        messages=messages,
    )

    if response.stop_reason != "tool_use":
        # Final text response
        for block in response.content:
            if hasattr(block, "text"):
                print(block.text)
        break

    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            fn        = TOOL_FN[block.name]
            model_cls = INPUT_MODEL[block.name]
            result    = fn(model_cls(**block.input))
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result.model_dump_json(),
            })

    messages.append({"role": "user", "content": tool_results})
```

### Recommended system prompt

Tell the model what tools are available and how to use them together:

```python
SYSTEM = """
You are a quantitative analyst assistant with access to 8 financial tools:

1. run_sma_backtest / run_rsi_backtest / run_macd_backtest / run_bollinger_backtest
   — Backtest a trading strategy on a single stock. Always use a minimum of 2 years
     of data for reliable statistics. Prefer Sharpe > 1.0 and max drawdown < 30%.

2. analyze_stock_risk
   — Get the full risk profile of a stock vs a benchmark. Use this before recommending
     a position. A beta > 1.5 means the stock amplifies market moves.

3. get_technical_analysis
   — Fetch the latest indicator readings for a stock. Use to confirm entry conditions
     before backtesting. RSI < 30 = oversold, RSI > 70 = overbought.

4. get_portfolio_analysis
   — Analyze a weighted basket of stocks. Weights must sum to 1.0. The correlation
     matrix shows diversification quality; pairs near 1.0 are redundant.

5. run_screener
   — Filter a list of tickers by fundamental and/or technical criteria. Always screen
     first, then analyze or backtest the survivors.

When a user asks a question that requires real data, always call the relevant tool
rather than guessing. Chain tools logically: screen → analyze → backtest.
"""
```

---

## Tool 1 — SMA Crossover Backtest

**When to use:** Testing trend-following strategies. Best on volatile assets with clear trends (individual stocks, sector ETFs). Works poorly on mean-reverting instruments.

**Signal logic:** Long when the fast SMA is above the slow SMA. Each bar where the fast SMA crosses above the slow SMA generates a new long entry; a cross below exits to flat.

```python
from standard_quant_tools.agent.tools import run_sma_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_sma_backtest(BacktestInput(
    symbol="TSLA",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 50},
    initial_capital=10_000,
    commission_pct=0.001,   # 0.1% per trade side (typical retail broker)
    slippage_pct=0.0005,    # 0.05% slippage
))

print(f"Total Return   : {result.total_return:.1%}")
print(f"Sharpe Ratio   : {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio  : {result.sortino_ratio:.2f}")
print(f"Max Drawdown   : {result.max_drawdown:.1%}")
print(f"Calmar Ratio   : {result.calmar_ratio:.2f}")
print(f"Win Rate       : {result.win_rate:.1%}")
print(f"Profit Factor  : {result.profit_factor:.2f}")
print(f"Num Trades     : {result.num_trades}")
print(f"Avg Trade P&L  : {result.avg_trade_return_pct:.2f}%")
print(f"Final Equity   : ${result.final_equity:,.2f}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast_period` | int | 10 | Fast SMA window (e.g. 10, 20) |
| `slow_period` | int | 30 | Slow SMA window (e.g. 50, 100, 200) |

**Common configurations:**

| Style | Fast | Slow | Notes |
|---|---|---|---|
| Swing | 10 | 30 | Reactive, more trades |
| Classic | 20 | 50 | Balanced |
| Position | 50 | 200 | Golden cross, fewer trades |

**Interpreting results:**

- `sharpe_ratio > 1.0` is good; `> 2.0` is excellent
- `max_drawdown` between −10% and −25% is typical for a trend-following system
- `calmar_ratio > 1.0` means the annual return exceeds the worst drawdown
- `profit_factor > 1.5` is a healthy edge; `< 1.0` means the strategy loses money
- Low `win_rate` (40–50%) is normal for trend strategies — a few large winners offset many small losses

---

## Tool 2 — RSI Mean Reversion Backtest

**When to use:** Testing mean-reversion strategies on liquid, range-bound instruments (broad market ETFs like SPY/QQQ, blue-chip dividend stocks). Works poorly on trending momentum stocks.

**Signal logic:** Enter long when RSI drops below `oversold`. Hold the position until RSI rises above `overbought`. Stateful — if RSI never reaches `overbought`, the position stays open.

```python
from standard_quant_tools.agent.tools import run_rsi_backtest
from standard_quant_tools.agent.models import BacktestInput

# Conservative: wait for deep oversold, exit quickly
result = run_rsi_backtest(BacktestInput(
    symbol="SPY",
    start_date="2018-01-01",
    end_date="2024-01-01",
    strategy_type="rsi_mean_reversion",
    parameters={"period": 14, "oversold": 30, "overbought": 60},
))

print(f"Win Rate       : {result.win_rate:.1%}")   # RSI strategies often have high win rates
print(f"Calmar Ratio   : {result.calmar_ratio:.2f}")
print(f"Profit Factor  : {result.profit_factor:.2f}")
print(f"Num Trades     : {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 14 | RSI lookback (7 = sensitive, 21 = smooth) |
| `oversold` | float | 30 | Enter long threshold (lower = rarer trades) |
| `overbought` | float | 70 | Exit threshold (lower = shorter holds) |

**Common configurations:**

| Style | Period | Oversold | Overbought | Characteristics |
|---|---|---|---|---|
| Aggressive | 7 | 35 | 65 | Many trades, quick exits |
| Classic | 14 | 30 | 70 | Balanced frequency |
| Conservative | 21 | 25 | 60 | Rare trades, high conviction |

**Inspecting individual trades:**

```python
if result.trade_log:
    # Find best and worst trades
    sorted_trades = sorted(result.trade_log, key=lambda t: t.return_pct, reverse=True)
    print("Best trade:", sorted_trades[0])
    print("Worst trade:", sorted_trades[-1])

    # Average holding period
    from datetime import datetime
    holds = [
        (datetime.fromisoformat(t.exit_date) - datetime.fromisoformat(t.entry_date)).days
        for t in result.trade_log
        if t.exit_date and t.entry_date
    ]
    if holds:
        print(f"Avg holding period: {sum(holds)/len(holds):.0f} days")
```

---

## Tool 3 — MACD Crossover Backtest

**When to use:** Momentum-following on medium timeframes. Good for stocks with strong trends that develop over weeks. Better on volatile assets than SMA crossover because MACD reacts to acceleration, not just direction.

**Signal logic:** Long when the MACD line (fast EMA − slow EMA) is above the signal line (EMA of MACD). Flat otherwise. No stateful holding — the signal updates each bar.

```python
from standard_quant_tools.agent.tools import run_macd_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_macd_backtest(BacktestInput(
    symbol="QQQ",
    start_date="2019-01-01",
    end_date="2024-01-01",
    strategy_type="macd_crossover",
    parameters={"fast": 12, "slow": 26, "signal": 9},
))

# Compare against buy-and-hold
print(f"Strategy return: {result.total_return:.1%}")
print(f"Strategy Sharpe: {result.sharpe_ratio:.2f}")
print(f"Trades: {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `fast` | int | 12 | Fast EMA (commonly 8 or 12) |
| `slow` | int | 26 | Slow EMA (commonly 21 or 26) |
| `signal` | int | 9 | Signal line EMA (commonly 9) |

**Less common but useful configurations:**

```python
# Faster MACD — more sensitive, more trades
parameters={"fast": 8, "slow": 21, "signal": 5}

# Weekly-equivalent MACD on daily bars
parameters={"fast": 5, "slow": 35, "signal": 5}
```

---

## Tool 4 — Bollinger Band Mean Reversion Backtest

**When to use:** Range-bound, oscillating instruments. Highly effective on commodity ETFs (GLD, USO), bond ETFs (TLT), and defensive sectors. Poor on strongly trending stocks.

**Signal logic:** Enter long when the closing price touches or crosses below the lower band. Exit when price returns to the middle band (the 20-day SMA). Position is held between these events regardless of how many bars it takes.

```python
from standard_quant_tools.agent.tools import run_bollinger_backtest
from standard_quant_tools.agent.models import BacktestInput

# Wider bands = rarer but higher-confidence entries
result = run_bollinger_backtest(BacktestInput(
    symbol="GLD",
    start_date="2015-01-01",
    end_date="2024-01-01",
    strategy_type="bollinger_reversion",
    parameters={"period": 20, "num_std": 2.0},
    initial_capital=50_000,
    commission_pct=0.0005,  # Lower commission for liquid ETF
))

print(f"Total Return : {result.total_return:.1%}")
print(f"Max Drawdown : {result.max_drawdown:.1%}")
print(f"Win Rate     : {result.win_rate:.1%}")  # Typically high for mean reversion
print(f"Num Trades   : {result.num_trades}")
```

**Parameters:**

| Key | Type | Default | Description |
|---|---|---|---|
| `period` | int | 20 | SMA window (also the basis for bands) |
| `num_std` | float | 2.0 | Band width in standard deviations |

**Band width effect on trading frequency:**

| `num_std` | Behaviour |
|---|---|
| 1.5 | More entries, shallower oversold |
| 2.0 | Classic setting |
| 2.5 | Rare but high-conviction entries |
| 3.0 | Extreme oversold only — very few trades |

---

## BacktestInput / BacktestResult — Full Reference

```python
from standard_quant_tools.agent.models import BacktestInput, BacktestResult, Trade

# All BacktestInput fields with descriptions:
inp = BacktestInput(
    symbol="AAPL",               # Required: ticker symbol
    start_date="2020-01-01",     # Required: ISO date string
    end_date="2024-01-01",       # Required: ISO date string
    strategy_type="sma_crossover",  # Required: strategy key (informational only)
    parameters={                 # Optional: strategy-specific params (see each tool)
        "fast_period": 10,
        "slow_period": 50,
    },
    initial_capital=10_000.0,    # Optional: default $10,000
    commission_pct=0.001,        # Optional: fraction per trade side (default 0.1%)
    slippage_pct=0.0005,         # Optional: fraction per trade side (default 0.05%)
)

# All BacktestResult fields:
result: BacktestResult = run_sma_backtest(inp)
result.total_return           # Net return as fraction (0.42 = +42%)
result.annualized_volatility  # Return std × √252
result.sharpe_ratio           # Annualized excess return / volatility
result.sortino_ratio          # Annualized excess return / downside volatility
result.max_drawdown           # Worst peak-to-trough (e.g. -0.23 = -23%)
result.calmar_ratio           # CAGR / |max_drawdown|; higher = better
result.win_rate               # Fraction of trades that were profitable (0–1)
result.profit_factor          # Gross profit / gross loss; > 1.5 is healthy
result.num_trades             # Number of completed round-trip trades
result.avg_trade_return_pct   # Average per-trade P&L in percent
result.final_equity           # Portfolio value at end date
result.equity_curve           # List[float] — daily portfolio value
result.trade_log              # Optional[List[Trade]] — per-trade details

# Quick benchmarking table — what's "good":
# Sharpe > 1.0     acceptable  |  > 2.0 excellent
# Max drawdown < 20%  comfortable  |  > 40% risky
# Win rate depends on strategy: 40% is fine for trend, 65%+ for mean reversion
# Profit factor > 1.5  good edge  |  < 1.0 losing strategy

# Serialize everything for LLM consumption:
import json
payload = result.model_dump()
payload.pop("equity_curve")  # Large list — omit if sending to LLM
print(json.dumps(payload, indent=2))
```

**Working with the equity curve:**

```python
import pandas as pd

# Reconstruct a dated equity curve for plotting
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()
df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")

result = run_sma_backtest(BacktestInput(
    symbol="AAPL", start_date="2020-01-01", end_date="2024-01-01",
    strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 50},
))

equity = pd.Series(result.equity_curve, index=df.index[:len(result.equity_curve)])
buy_hold = 10_000 * (1 + df["Close"].pct_change().fillna(0)).cumprod()

print(f"Strategy final  : ${equity.iloc[-1]:,.0f}")
print(f"Buy & hold final: ${buy_hold.iloc[-1]:,.0f}")
```

---

## Tool 5 — Stock Risk Analysis

**When to use:** Pre-trade due diligence, risk profiling for an LLM to explain a stock's characteristics. Compare multiple stocks on the same risk dimensions before building a portfolio.

```python
from standard_quant_tools.agent.tools import analyze_stock_risk
from standard_quant_tools.agent.models import AnalysisInput

result = analyze_stock_risk(AnalysisInput(
    symbol="NVDA",
    benchmark="SPY",   # Default. Use "QQQ" for tech comparison, "GLD" for commodity
    period="2y",       # "6mo", "1y", "2y", "3y", or "Nd" for N days
))

print(f"Alpha            : {result.alpha:.4f}")        # Daily excess return vs benchmark
print(f"Beta             : {result.beta:.2f}")          # < 1 = defensive, > 1 = amplified
print(f"R² (vs SPY)      : {result.r_squared:.2%}")    # How much variance is market-driven
print(f"Sharpe Ratio     : {result.sharpe_ratio:.2f}") # > 1.0 good, > 2.0 excellent
print(f"Sortino Ratio    : {result.sortino_ratio:.2f}")
print(f"Max Drawdown     : {result.max_drawdown:.1%}") # Negative number
print(f"VaR (95%, daily) : {result.var_95:.3%}")       # Max daily loss 95% of the time
print(f"CVaR (95%, daily): {result.cvar_95:.3%}")      # Expected loss when VaR is breached
print(f"Information Ratio: {result.information_ratio:.2f}")
```

**Interpreting each metric:**

| Metric | What it means | Typical range |
|---|---|---|
| `alpha` | Daily excess return above benchmark (raw, not annualized) | −0.002 to +0.002 |
| `beta` | Sensitivity to benchmark. 1.5 = stock moves 1.5× the market | 0.3 (utilities) to 2.5 (high-growth) |
| `r_squared` | How much of price movement is explained by the benchmark | 0.1 (idiosyncratic) to 0.9 (ETF-like) |
| `sharpe_ratio` | Risk-adjusted return (annualized) | 0.5–1.0 good for equities |
| `sortino_ratio` | Like Sharpe but only penalizes downside | Always ≥ Sharpe |
| `var_95` | Largest expected daily loss, 95% confidence | 0.01–0.05 (1%–5%) |
| `cvar_95` | Average loss on the 5% worst days | Always ≥ VaR |
| `information_ratio` | Active return vs benchmark / tracking error | > 0.5 strong |

**Comparing multiple stocks:**

```python
stocks  = ["AAPL", "NVDA", "TSLA", "MSFT", "JPM"]
results = [
    analyze_stock_risk(AnalysisInput(symbol=s, period="1y"))
    for s in stocks
]

# Build a comparison table
rows = [{
    "symbol":  r.symbol,
    "beta":    r.beta,
    "sharpe":  r.sharpe_ratio,
    "drawdown":f"{r.max_drawdown:.1%}",
    "var_95":  f"{r.var_95:.2%}",
    "ir":      r.information_ratio,
} for r in results]

import pandas as pd
print(pd.DataFrame(rows).sort_values("sharpe", ascending=False).to_string(index=False))
```

**Using a non-SPY benchmark:**

```python
# Tech-relative analysis: beats QQQ?
result_tech = analyze_stock_risk(AnalysisInput(symbol="NVDA", benchmark="QQQ", period="2y"))

# Inflation hedge: beats gold?
result_gold = analyze_stock_risk(AnalysisInput(symbol="GLD", benchmark="TLT", period="3y"))
```

---

## Tool 6 — Technical Analysis

**When to use:** Snapshot of current market conditions on a stock. Use to confirm an entry idea or quickly survey which signals are active. The LLM can interpret the `signals` dict and explain what it sees.

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

result = get_technical_analysis(TechnicalInput(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-06-01",
    indicators=["rsi", "macd", "bollinger", "adx", "sma", "obv", "atr"],
))

print(f"Last close  : ${result.last_close}")
print()
print("--- Indicator values ---")
for k, v in sorted(result.last_values.items()):
    print(f"  {k:<20}: {v}")
print()
print("--- Active signals ---")
for k, v in sorted(result.signals.items()):
    print(f"  {k:<30}: {v}")
```

**Available indicators:**

| Key | Values returned | Signals generated | Notes |
|---|---|---|---|
| `sma` | `sma_20`, `sma_50`, `sma_200` | `price_above_sma_20/50/200` | Trend filter |
| `ema` | `ema_12`, `ema_26` | — | Faster trend |
| `macd` | `macd`, `macd_signal`, `macd_histogram` | `macd_bullish` | Momentum |
| `rsi` | `rsi_14` | `rsi_oversold` (< 30), `rsi_overbought` (> 70) | Mean reversion |
| `stochastic` | `stoch_k`, `stoch_d` | `stoch_oversold` (K/D < 20) | Short-term momentum |
| `bollinger` | `bb_upper`, `bb_middle`, `bb_lower` | `price_near_lower_band`, `price_near_upper_band` | Volatility |
| `atr` | `atr_14` | — | Volatility in price units |
| `obv` | `obv` | `obv_rising` | Volume-price confirmation |
| `vwap` | `vwap` | `price_above_vwap` | Intraday fair value |
| `adx` | `adx`, `di_plus`, `di_minus` | `strong_trend` (ADX > 25), `bullish_di` (DI+ > DI−) | Trend strength |
| `williams_r` | `williams_r` | `williams_r_oversold` (< −80), `williams_r_overbought` (> −20) | Oscillator |

**Screening for entry conditions programmatically:**

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput

def is_oversold_in_trend(symbol: str, start: str, end: str) -> bool:
    """True when: RSI oversold + price above 50-day SMA + strong uptrend."""
    result = get_technical_analysis(TechnicalInput(
        symbol=symbol, start_date=start, end_date=end,
        indicators=["rsi", "sma", "adx"],
    ))
    s = result.signals
    return (
        s.get("rsi_oversold", False)
        and s.get("price_above_sma_50", False)
        and s.get("strong_trend", False)
        and s.get("bullish_di", False)
    )

candidates = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]
for ticker in candidates:
    if is_oversold_in_trend(ticker, "2023-01-01", "2024-01-01"):
        print(f"{ticker}: oversold pullback in a strong uptrend — potential entry")
```

**Building a multi-stock technical dashboard:**

```python
from standard_quant_tools.agent.tools import get_technical_analysis
from standard_quant_tools.agent.models import TechnicalInput
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]
rows = []

for ticker in tickers:
    r = get_technical_analysis(TechnicalInput(
        symbol=ticker,
        start_date="2023-06-01",
        end_date="2024-01-01",
        indicators=["rsi", "macd", "adx"],
    ))
    rows.append({
        "ticker":        ticker,
        "close":         r.last_close,
        "rsi_14":        r.last_values.get("rsi_14"),
        "adx":           r.last_values.get("adx"),
        "macd_bullish":  r.signals.get("macd_bullish"),
        "strong_trend":  r.signals.get("strong_trend"),
        "rsi_oversold":  r.signals.get("rsi_oversold"),
    })

df = pd.DataFrame(rows).set_index("ticker")
print(df.to_string())
```

---

## Tool 7 — Portfolio Analysis

**When to use:** Evaluate a weighted basket of assets together. Identify diversification quality via correlation, compare equal-weight vs custom weights, understand portfolio-level risk metrics.

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

result = get_portfolio_analysis(PortfolioInput(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],  # Must sum to 1.0
    start_date="2022-01-01",
    end_date="2024-01-01",
    benchmark="SPY",                           # For Information Ratio
))

print(f"Annualized Return     : {result.annualized_return:.2%}")
print(f"Annualized Volatility : {result.annualized_volatility:.2%}")
print(f"Sharpe Ratio          : {result.sharpe_ratio:.2f}")
print(f"Sortino Ratio         : {result.sortino_ratio:.2f}")
print(f"Max Drawdown          : {result.max_drawdown:.2%}")
print(f"Calmar Ratio          : {result.calmar_ratio:.2f}")
print(f"VaR (95%, daily)      : {result.var_95:.3%}")
print(f"CVaR (95%, daily)     : {result.cvar_95:.3%}")
print(f"Information Ratio     : {result.information_ratio:.2f}")
print(f"Total Return          : {result.total_return:.2%}")
```

**Extracting the correlation matrix:**

```python
import pandas as pd

corr_dict = result.correlation_matrix
corr_df   = pd.DataFrame(corr_dict)

print("Correlation matrix:")
print(corr_df.round(2).to_string())

# Find the most and least correlated pairs
pairs = [
    (t1, t2, corr_df.loc[t1, t2])
    for i, t1 in enumerate(result.tickers)
    for t2 in result.tickers[i+1:]
]
pairs.sort(key=lambda x: x[2])
print(f"\nLowest correlation (best diversification): {pairs[0][0]} / {pairs[0][1]} = {pairs[0][2]:.2f}")
print(f"Highest correlation (most redundant)     : {pairs[-1][0]} / {pairs[-1][1]} = {pairs[-1][2]:.2f}")
```

**Comparing equal-weight vs custom weights:**

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
n       = len(tickers)

# Equal weight
eq = get_portfolio_analysis(PortfolioInput(
    tickers=tickers,
    weights=[1/n] * n,
    start_date="2022-01-01",
    end_date="2024-01-01",
))

# Overweight NVDA (the best performer in the period)
custom = get_portfolio_analysis(PortfolioInput(
    tickers=tickers,
    weights=[0.15, 0.15, 0.15, 0.15, 0.40],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

print(f"Equal-weight  Sharpe : {eq.sharpe_ratio:.2f} | Return: {eq.annualized_return:.2%}")
print(f"Custom-weight Sharpe : {custom.sharpe_ratio:.2f} | Return: {custom.annualized_return:.2%}")
```

**Defensive vs aggressive portfolio:**

```python
# Defensive: bonds + gold + dividend stocks
defensive = get_portfolio_analysis(PortfolioInput(
    tickers=["TLT", "GLD", "VYM", "KO", "JNJ"],
    weights=[0.30, 0.20, 0.20, 0.15, 0.15],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

# Aggressive: concentrated growth tech
aggressive = get_portfolio_analysis(PortfolioInput(
    tickers=["NVDA", "TSLA", "META", "AMZN", "AMD"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],
    start_date="2022-01-01",
    end_date="2024-01-01",
))

print(f"Defensive — Sharpe: {defensive.sharpe_ratio:.2f} | MDD: {defensive.max_drawdown:.1%}")
print(f"Aggressive — Sharpe: {aggressive.sharpe_ratio:.2f} | MDD: {aggressive.max_drawdown:.1%}")
```

---

## Tool 8 — Stock Screener

**When to use:** Starting a research workflow. Narrow a large universe down to candidates before spending compute on analysis or backtesting. All filters are applied concurrently — screening 50 tickers takes roughly the same time as screening 5.

```python
from standard_quant_tools.agent.tools import run_screener
from standard_quant_tools.agent.models import ScreenerInput
import json

result = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN",
             "JPM", "BAC", "V", "MA", "UNH", "JNJ", "PG", "KO"],
    filters={
        "pe_ratio_max":      35,
        "profit_margin_min": 0.15,   # At least 15% net margin
        "rsi_max":           50,     # Not already overbought
        "market_cap_min":    50_000_000_000,  # $50B+
        "beta_max":          1.5,    # Not excessively volatile
    },
    sort_by="rsi_14",
    ascending=True,  # Most oversold first
))

print(f"Passed: {result.num_passed} / {len(result.tickers_passed)} tickers")
print(f"Survivors: {result.tickers_passed}")

# Send to LLM
print(json.dumps(result.model_dump(), indent=2))
```

**All supported filters:**

| Filter | Type | Description | Example |
|---|---|---|---|
| `pe_ratio_max` | float | Forward P/E upper bound | `25` |
| `pb_ratio_max` | float | Price-to-Book upper bound | `5.0` |
| `debt_equity_max` | float | Debt-to-Equity upper bound | `150` |
| `roe_min` | float | Return on Equity minimum (decimal) | `0.15` = 15% |
| `profit_margin_min` | float | Net profit margin minimum (decimal) | `0.10` = 10% |
| `div_yield_min` | float | Dividend yield minimum (decimal) | `0.02` = 2% |
| `market_cap_min` | int | Market cap minimum (USD) | `10_000_000_000` |
| `rsi_max` | float | RSI(14) upper bound | `40` = oversold screen |
| `rsi_min` | float | RSI(14) lower bound | `60` = momentum screen |
| `price_above_sma` | int | Close must be above SMA(N) | `50` |
| `price_below_sma` | int | Close must be below SMA(N) | `200` |
| `beta_max` | float | Beta vs SPY upper bound | `1.2` |
| `beta_min` | float | Beta vs SPY lower bound | `0.5` |

**Pre-built screen recipes:**

```python
# Value screen: cheap, profitable, low leverage
value = ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "WMT", "KO", "JNJ"],
    filters={"pe_ratio_max": 20, "pb_ratio_max": 3.0, "roe_min": 0.15, "debt_equity_max": 100},
    sort_by="forward_pe", ascending=True,
)

# Momentum screen: already running, confirmed by SMA structure
momentum = ScreenerInput(
    tickers=["NVDA", "META", "MSFT", "AAPL", "GOOGL", "AMD", "AVGO"],
    filters={"rsi_min": 55, "price_above_sma": 50, "price_above_sma": 200},
    sort_by="rsi_14", ascending=False,
)

# Oversold quality: temporary weakness in strong businesses
oversold_quality = ScreenerInput(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "V", "MA", "UNH"],
    filters={"rsi_max": 40, "profit_margin_min": 0.15, "beta_max": 1.5, "market_cap_min": 50_000_000_000},
    sort_by="rsi_14", ascending=True,
)

# Defensive dividend: income + low volatility
dividend = ScreenerInput(
    tickers=["KO", "PEP", "JNJ", "PG", "MCD", "T", "VZ", "O"],
    filters={"div_yield_min": 0.025, "beta_max": 0.8, "debt_equity_max": 200},
    sort_by="dividend_yield", ascending=False,
)

# Run any of them:
result = run_screener(oversold_quality)
```

---

## Chaining Tools — Multi-Step Workflows

The real power of the agent module is composing tools into workflows. Here are three common patterns.

### Pattern 1: Screen → Analyze → Decide

```python
from standard_quant_tools.agent.tools import run_screener, analyze_stock_risk
from standard_quant_tools.agent.models import ScreenerInput, AnalysisInput

# Step 1: find oversold large-cap stocks
screen = run_screener(ScreenerInput(
    tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
             "V", "MA", "UNH", "JNJ", "PG", "HD", "KO", "WMT"],
    filters={"rsi_max": 40, "profit_margin_min": 0.12, "market_cap_min": 100_000_000_000},
    sort_by="rsi_14", ascending=True,
))
print(f"Screened to: {screen.tickers_passed}")

# Step 2: risk-profile each survivor
if screen.tickers_passed:
    risk_profiles = [
        analyze_stock_risk(AnalysisInput(symbol=t, period="1y"))
        for t in screen.tickers_passed
    ]
    # Keep only low-beta (defensive) candidates
    candidates = [r for r in risk_profiles if r.beta < 1.0 and r.sharpe_ratio > 0.5]
    for r in sorted(candidates, key=lambda x: x.sharpe_ratio, reverse=True):
        print(f"{r.symbol}: Sharpe={r.sharpe_ratio:.2f}, Beta={r.beta:.2f}, MDD={r.max_drawdown:.1%}")
```

### Pattern 2: Screen → Backtest Each Survivor

```python
from standard_quant_tools.agent.tools import run_screener, run_rsi_backtest
from standard_quant_tools.agent.models import ScreenerInput, BacktestInput

universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
            "JPM", "V", "MA", "UNH", "JNJ", "PG"]

screen = run_screener(ScreenerInput(
    tickers=universe,
    filters={"rsi_max": 45, "pe_ratio_max": 40},
    sort_by="rsi_14", ascending=True,
))

if screen.tickers_passed:
    results = []
    for ticker in screen.tickers_passed:
        bt = run_rsi_backtest(BacktestInput(
            symbol=ticker,
            start_date="2021-01-01",
            end_date="2024-01-01",
            strategy_type="rsi_mean_reversion",
            parameters={"period": 14, "oversold": 30, "overbought": 65},
        ))
        results.append({
            "ticker":    ticker,
            "sharpe":    bt.sharpe_ratio,
            "win_rate":  bt.win_rate,
            "drawdown":  bt.max_drawdown,
            "trades":    bt.num_trades,
        })

    import pandas as pd
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print(df.to_string(index=False))
```

### Pattern 3: Technical Analysis → Conditional Backtest

```python
from standard_quant_tools.agent.tools import get_technical_analysis, run_sma_backtest
from standard_quant_tools.agent.models import TechnicalInput, BacktestInput

ticker = "NVDA"

# Step 1: check current market structure
tech = get_technical_analysis(TechnicalInput(
    symbol=ticker,
    start_date="2023-01-01",
    end_date="2024-01-01",
    indicators=["adx", "rsi", "sma"],
))

is_trending = tech.signals.get("strong_trend", False)
above_200   = tech.signals.get("price_above_sma_200", False)

print(f"{ticker}: strong trend={is_trending}, above 200-SMA={above_200}")

# Step 2: choose strategy based on regime
if is_trending and above_200:
    print("Trend regime detected — backtesting SMA crossover")
    result = run_sma_backtest(BacktestInput(
        symbol=ticker, start_date="2021-01-01", end_date="2024-01-01",
        strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 50},
    ))
else:
    print("Range-bound regime detected — backtesting RSI mean reversion")
    from standard_quant_tools.agent.tools import run_rsi_backtest
    result = run_rsi_backtest(BacktestInput(
        symbol=ticker, start_date="2021-01-01", end_date="2024-01-01",
        strategy_type="rsi_mean_reversion", parameters={"period": 14, "oversold": 30, "overbought": 70},
    ))

print(f"Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
```

---

## Complete End-to-End Agent Loop

A production-ready loop that handles multiple tool calls per turn, serializes results, and terminates cleanly. Works with both OpenAI and Anthropic with minor changes.

```python
import json
from typing import Any

import anthropic

from standard_quant_tools.agent.tools import (
    get_agent_tools,
    run_sma_backtest, run_rsi_backtest, run_macd_backtest, run_bollinger_backtest,
    analyze_stock_risk, get_technical_analysis, get_portfolio_analysis, run_screener,
)
from standard_quant_tools.agent.models import (
    BacktestInput, AnalysisInput, TechnicalInput, PortfolioInput, ScreenerInput,
)

# ── Tool dispatch ─────────────────────────────────────────────────────────────

TOOL_FN: dict[str, Any] = {
    "run_sma_backtest":       run_sma_backtest,
    "run_rsi_backtest":       run_rsi_backtest,
    "run_macd_backtest":      run_macd_backtest,
    "run_bollinger_backtest": run_bollinger_backtest,
    "analyze_stock_risk":     analyze_stock_risk,
    "get_technical_analysis": get_technical_analysis,
    "get_portfolio_analysis": get_portfolio_analysis,
    "run_screener":           run_screener,
}

INPUT_MODEL: dict[str, Any] = {
    "run_sma_backtest":       BacktestInput,
    "run_rsi_backtest":       BacktestInput,
    "run_macd_backtest":      BacktestInput,
    "run_bollinger_backtest": BacktestInput,
    "analyze_stock_risk":     AnalysisInput,
    "get_technical_analysis": TechnicalInput,
    "get_portfolio_analysis": PortfolioInput,
    "run_screener":           ScreenerInput,
}


def dispatch_tool(name: str, raw_input: dict) -> str:
    """Call the named tool and return a JSON string for the LLM."""
    fn        = TOOL_FN[name]
    model_cls = INPUT_MODEL[name]
    result    = fn(model_cls(**raw_input))

    # Strip large lists (equity_curve) before sending back — saves tokens
    payload = result.model_dump()
    payload.pop("equity_curve", None)
    return json.dumps(payload)


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM = """
You are a quantitative investment analyst. You have 8 tools:
- run_sma_backtest, run_rsi_backtest, run_macd_backtest, run_bollinger_backtest
- analyze_stock_risk
- get_technical_analysis
- get_portfolio_analysis
- run_screener

Always start by screening if the user mentions a broad universe.
Prefer at least 2 years of data for backtests.
Interpret numeric results clearly — translate Sharpe ratios, drawdowns, and
betas into plain English recommendations.
"""


def run_agent(user_message: str, max_turns: int = 10) -> str:
    client   = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=SYSTEM,
            tools=get_agent_tools(),
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Final answer — extract text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Execute every tool call in this turn
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → Calling {block.name}({list(block.input.keys())})")
                try:
                    content = dispatch_tool(block.name, block.input)
                except Exception as e:
                    content = json.dumps({"error": str(e)})

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     content,
                })

        messages.append({"role": "user", "content": tool_results})

    return "Max turns reached."


# ── Example queries ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        "Screen FAANG + NVDA for PE < 35 and RSI < 50. For any that pass, run an RSI mean-reversion backtest from 2021 to 2024 and tell me which had the best risk-adjusted return.",
        "Compare the risk profiles of NVDA and JNJ vs SPY over the past 2 years. Which is more suitable for a conservative portfolio?",
        "Analyze an equal-weight portfolio of AAPL, MSFT, GOOGL, and AMZN from 2022 to 2024. How diversified is it?",
    ]

    for q in queries[:1]:  # Run one query as demonstration
        print(f"\nQuery: {q}\n{'─' * 60}")
        answer = run_agent(q)
        print(f"\nAnswer:\n{answer}")
```

---

## Model Summary

### Input Models

| Model | Required | Optional (with defaults) |
|---|---|---|
| `BacktestInput` | `symbol`, `start_date`, `end_date`, `strategy_type` | `parameters={}`, `initial_capital=10000`, `commission_pct=0.001`, `slippage_pct=0.0005` |
| `AnalysisInput` | `symbol` | `benchmark="SPY"`, `period="1y"` |
| `TechnicalInput` | `symbol`, `start_date`, `end_date` | `indicators=["rsi","macd","bollinger","atr"]` |
| `PortfolioInput` | `tickers`, `weights`, `start_date`, `end_date` | `benchmark="SPY"` |
| `ScreenerInput` | `tickers`, `filters` | `start_date`, `end_date`, `sort_by=None`, `ascending=True` |

### Output Models

| Model | Key fields |
|---|---|
| `BacktestResult` | `total_return`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `num_trades`, `avg_trade_return_pct`, `final_equity`, `equity_curve`, `trade_log` |
| `AnalysisResult` | `symbol`, `benchmark`, `alpha`, `beta`, `r_squared`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_95`, `cvar_95`, `information_ratio` |
| `TechnicalResult` | `symbol`, `last_close`, `last_values` (dict), `signals` (dict) |
| `PortfolioResult` | `tickers`, `weights`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `information_ratio`, `total_return`, `correlation_matrix` |
| `ScreenerResult` | `num_passed`, `tickers_passed`, `results` (list of dicts) |
| `Trade` | `entry_date`, `exit_date`, `direction`, `entry_price`, `exit_price`, `return_pct` |
