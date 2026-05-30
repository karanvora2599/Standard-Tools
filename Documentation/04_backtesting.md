# Backtesting

The backtesting engine is **fully vectorized** — it computes the entire equity curve in one NumPy operation instead of looping bar-by-bar. This makes it orders of magnitude faster than event-driven backtesting for signal-based strategies.

---

## Core Concepts

**Signal series:** A `pd.Series` aligned to the OHLCV index with values:
- `1` = long position (buy)
- `0` = flat (no position)
- `-1` = short position (sell short)

**Lookahead prevention:** Signals are automatically shifted by 1 bar — a signal generated at close of day *t* executes at the close of day *t+1*.

**Transaction costs:** Applied each time the position changes. Going from flat (0) to long (+1) costs `1 × (commission + slippage)`. Going from long (+1) to short (−1) costs `2 ×` (full reversal).

---

## Running a Backtest

```python
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.indicators import sma
from standard_quant_tools.data.factory import DataFactory
import pandas as pd
import numpy as np

provider = DataFactory.get_provider()
df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")

# SMA crossover signal
fast = sma(df['Close'], 20)
slow = sma(df['Close'], 50)
signals = pd.Series(np.where(fast > slow, 1, 0), index=df.index)

result = run_strategy(
    df,
    signals,
    initial_capital=10_000,
    commission_pct=0.001,    # 0.1% per trade side
    slippage_pct=0.0005,     # 0.05% per trade side
    include_trade_log=True,
)

print(f"Final Equity   : ${result['final_equity']:,.2f}")
print(f"Total Return   : {result['total_return']:.1%}")
print(f"Sharpe Ratio   : {result['sharpe_ratio']:.2f}")
print(f"Max Drawdown   : {result['max_drawdown']:.1%}")
print(f"Calmar Ratio   : {result['calmar_ratio']:.2f}")
print(f"Win Rate       : {result['win_rate']:.1%}")
print(f"Profit Factor  : {result['profit_factor']:.2f}")
print(f"Num Trades     : {result['num_trades']}")
```

---

## Trade Log

When `include_trade_log=True`, the result includes a `pd.DataFrame` with one row per completed trade.

```python
trade_log = result['trade_log']
print(trade_log.columns)
# ['entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price', 'return_pct']

# Best and worst trades
print(trade_log.nlargest(3, 'return_pct'))
print(trade_log.nsmallest(3, 'return_pct'))

# Average holding period
trade_log['holding_days'] = (pd.to_datetime(trade_log['exit_date'])
                             - pd.to_datetime(trade_log['entry_date'])).dt.days
print(f"Avg holding: {trade_log['holding_days'].mean():.0f} days")
```

---

## Pre-Built Agent Backtest Strategies

All 4 strategies use the same `run_strategy` engine under the hood and return a `BacktestResult` Pydantic model.

### SMA Crossover

```python
from standard_quant_tools.agent.tools import run_sma_backtest
from standard_quant_tools.agent.models import BacktestInput

result = run_sma_backtest(BacktestInput(
    symbol="TSLA",
    start_date="2021-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 30},
    initial_capital=50_000,
    commission_pct=0.001,
))
print(f"Sharpe: {result.sharpe_ratio:.2f}, Trades: {result.num_trades}")
```

### RSI Mean Reversion

```python
from standard_quant_tools.agent.tools import run_rsi_backtest

result = run_rsi_backtest(BacktestInput(
    symbol="SPY",
    start_date="2020-01-01",
    end_date="2024-01-01",
    strategy_type="rsi_mean_reversion",
    parameters={"period": 14, "oversold": 30, "overbought": 70},
))
print(f"Win Rate: {result.win_rate:.1%}, Calmar: {result.calmar_ratio:.2f}")
```

### MACD Crossover

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

### Bollinger Band Mean Reversion

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

---

## Parameter Grid Search

`backtest_grid` runs every combination in `param_grid` in parallel using `ProcessPoolExecutor` and returns a ranked DataFrame.

```python
from standard_quant_tools.backtest import backtest_grid
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()
df = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")

results = backtest_grid(
    price_data=df,
    strategy="sma_crossover",          # or rsi_mean_reversion / macd_crossover / bollinger_reversion
    param_grid={
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100, 200],
    },
    initial_capital=10_000,
    commission_pct=0.001,
    sort_by="sharpe_ratio",            # default: best Sharpe first
    n_workers=4,                       # default: os.cpu_count()
)

# 3 × 4 = 12 combinations, sorted best → worst Sharpe
print(results[["fast_period", "slow_period", "sharpe_ratio", "total_return", "max_drawdown"]].head())
```

All four strategies are supported:

```python
results = backtest_grid(df, strategy="rsi_mean_reversion",
    param_grid={"period": [7, 14, 21], "oversold": [25, 30], "overbought": [65, 70]})

results = backtest_grid(df, strategy="macd_crossover",
    param_grid={"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]})

results = backtest_grid(df, strategy="bollinger_reversion",
    param_grid={"period": [15, 20, 25], "num_std": [1.5, 2.0, 2.5]})
```

Pass `n_workers=1` to run sequentially (no subprocess overhead — useful in notebooks).

---

## Custom Signal Generation

You can plug any signal series into `run_strategy`:

```python
from standard_quant_tools.indicators import rsi, bollinger_bands, adx
import pandas as pd
import numpy as np

df = provider.get_ohlcv("MSFT", "2022-01-01", "2024-01-01")

# Multi-condition signal:
# - RSI < 40 (oversold)
# - Price below BB lower (stretched)
# - ADX > 25 (trending market)
rsi_vals  = rsi(df['Close'], 14)
bb        = bollinger_bands(df['Close'], 20, 2.0)
adx_df    = adx(df['High'], df['Low'], df['Close'])

long_cond = (rsi_vals < 40) & (df['Close'] < bb['BB_Lower']) & (adx_df['ADX'] > 25)
exit_cond = rsi_vals > 60

# Stateful signal: hold once entered until exit condition
values = np.zeros(len(df))
in_pos = False
for i in range(len(df)):
    if not in_pos and bool(long_cond.iloc[i]):
        in_pos = True
    elif in_pos and bool(exit_cond.iloc[i]):
        in_pos = False
    values[i] = 1.0 if in_pos else 0.0

signals = pd.Series(values, index=df.index)
result = run_strategy(df, signals, initial_capital=10_000, commission_pct=0.001)
```

---

## Understanding the Output

| Key | Type | Description |
|---|---|---|
| `final_equity` | float | Portfolio value at end |
| `total_return` | float | Net return as fraction (0.42 = +42%) |
| `annualized_volatility` | float | Return std × √252 |
| `sharpe_ratio` | float | Annualized excess return / vol |
| `sortino_ratio` | float | Annualized excess return / downside vol |
| `max_drawdown` | float | Worst peak-to-trough decline (negative) |
| `calmar_ratio` | float | CAGR / \|max drawdown\| |
| `win_rate` | float | Fraction of profitable trades |
| `profit_factor` | float | Gross profit / gross loss |
| `num_trades` | int | Number of completed round-trips |
| `avg_trade_return_pct` | float | Average trade P&L in % |
| `equity_curve` | pd.Series | Day-by-day portfolio value |
| `trade_log` | pd.DataFrame | Per-trade entry/exit details |
