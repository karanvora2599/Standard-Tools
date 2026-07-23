# Backtesting

The backtesting engine is **fully vectorized** — it computes the entire equity curve in one NumPy operation instead of looping bar-by-bar. This makes it orders of magnitude faster than event-driven backtesting for signal-based strategies.

**Signal generators** (`rsi_mean_reversion`, `bollinger_reversion`) use a Numba JIT-compiled state machine for their entry/exit tracking loops, providing ~50–100× speedup over a plain Python loop on long series. This is especially impactful inside `backtest_grid` where the same loop runs thousands of times across parameter combinations. The JIT path requires `numba` installed with NumPy ≤ 2.0; pure Python is used otherwise.

---

## C++ Acceleration

When the `_sqt_core` extension is built, `run_strategy` automatically routes to a compiled C++ kernel — **3–8× faster** than the NumPy/pandas vectorized fallback on typical series lengths. The signal generators and `backtest_grid`'s parallel executor remain in Python; only the per-bar return computation, equity curve, and all metrics run in C++.

```python
from standard_quant_tools.backtest.engine import HAS_CPP

print(f"C++ kernel active: {HAS_CPP}")
```

**What the C++ kernel computes in a single pass:**
- One-bar lag execution: `executed[i] = signals[i-1]`
- Strategy returns with transaction costs applied on position changes
- Equity curve (cumulative product of `1 + strategy_return`)
- All performance metrics: total return, annualized vol, Sharpe, Sortino, max drawdown, Calmar
- Trade statistics: num trades (closed only), win rate, profit factor, avg trade return

**`backtest_grid` batch kernel:** When `_sqt_core` is built, `backtest_grid` uses an additional C++ batch path. All signal arrays for every parameter combination are generated in Python (the strategy logic is Python), stacked into a single 2D matrix, and passed to `_sqt_core.batch_run_strategy` in **one call**. This eliminates Python re-entry overhead between combinations and is significantly faster than the previous approach of calling the C++ kernel once per combination from a `ProcessPoolExecutor` worker.

The optional per-trade log (`include_trade_log=True`) still runs in Python — it requires DatetimeIndex-aware iteration to produce labeled entry/exit dates. All numeric results are identical to the Python fallback.

| Scenario | Python (pandas) | C++ single calls | C++ batch kernel | Speedup (batch) |
|---|---|---|---|---|
| `run_strategy` (n=2000) | ~1–3 ms | ~0.1–0.4 ms | — | **3–8×** |
| Grid (100 combos) | ~100–300 ms | ~10–40 ms | ~5–20 ms | **10–50×** |
| Grid (1000 combos) | ~1–3 s | ~0.1–0.3 s | ~50–200 ms | **10–50×** |

See [build_guide.md](../Development/build_guide.md) for build instructions.

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

## Execution Timing (`fill_price`)

By default (`fill_price="close"`), a signal known at bar *t-1*'s close is
assumed filled at that same close, earning bar *t*'s full close-to-close
return — the standard lookahead-free convention (see Core Concepts above).
This is optimistic in one sense: it assumes you can transact right at the
closing price the instant you observe it.

`fill_price="next_open"` is more conservative: entries and exits are priced
off the bar's own `Open` instead. `fill_price="midpoint"` uses that bar's
`(High + Low) / 2` instead — a bid/ask-free proxy for a midquote fill, for
data sources that don't carry a real bid/ask. Both decompose each bar into
two legs:

- **Overnight leg** (prior close → this bar's reference price), priced at
  *yesterday's* position — an exit still bears the gap risk of the position
  it was held through overnight, before selling at today's reference price.
- **Intraday leg** (this bar's reference price → close), priced at *today's*
  position — a same-day entry only earns its own reference-to-close move.

A held (unchanged) position sums these two legs rather than compounding them
— a second-order, daily-bar-negligible difference from pure close-to-close
(their product is the only gap, e.g. two 0.5% legs differ from true
compounding by ~0.0025%).

```python
result_close     = run_strategy(df, signals, fill_price="close")       # default
result_next_open = run_strategy(df, signals, fill_price="next_open")   # more conservative
result_midpoint  = run_strategy(df, signals, fill_price="midpoint")    # bid/ask-free midquote proxy
```

`backtest_grid` accepts the same `fill_price` argument and forces the
Python execution path when it isn't `"close"` — the compiled C++ kernel
only knows `Close` prices, so `next_open`/`midpoint` always run in Python
regardless of whether `_sqt_core` is built.

**Known limitation:** the trade log's `entry_price`/`exit_price` always
report the bar's `Close`, in every mode — only the aggregate P&L (equity
curve, Sharpe, total return, everything else) reflects `next_open`/
`midpoint` fills correctly. Treat the trade log's prices as a reference
marker, not the literal assumed fill price, when using either mode.

On the agent-tool side, `fill_price` is exposed on `BacktestInput`,
`BuyAndHoldInput`, `CompareStrategiesInput`, `CustomSignalBacktestInput`,
`SignalPanelBacktestInput`, `BacktestOptInput`, `BacktestDiagnosticsInput`,
`WalkForwardInput`, `RegimeAdaptiveWalkForwardInput`, and
`PortfolioSimulationInput` — every tool whose out-of-sample/simulated leg
ultimately calls `run_strategy` or `run_portfolio_simulation`. The one
holdout is `run_regime_adaptive_backtest` (the older, in-sample exploratory
tool, kept deliberately simple — see its own docstring), which still always
uses `"close"`.

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

`backtest_grid` runs every combination in `param_grid` in parallel using `ProcessPoolExecutor` and returns a ranked DataFrame. Each worker runs the full signal-generation + backtest pipeline independently, so the Numba JIT state machines (where available) multiply the benefit: JIT speedup × core count.

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

## Grid-Searching Your Own Signal

`backtest_grid`'s `strategy` argument accepts either a built-in registry name
(`"sma_crossover"`, etc.) **or your own signal-generating callable** — the
grid searcher, C++ batch kernel, and sort/rank logic don't care where the
signal came from.

```python
from standard_quant_tools.backtest import backtest_grid
import pandas as pd

def my_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
    """Any proprietary alpha logic — this is not part of the library."""
    edge = my_model.score(price_data)          # your model, not this library's
    return (edge > threshold).astype(int)      # 1 = long, 0 = flat, -1 = short

results = backtest_grid(
    df,
    strategy=my_signal,
    param_grid={"threshold": [0.05, 0.10, 0.15, 0.20]},
    sort_by="sharpe_ratio",
)
print(results[["threshold", "sharpe_ratio", "total_return", "num_trades"]])
```

A custom callable still gets the full C++ batch-kernel speedup when
`_sqt_core` is built: the C++ path always calls `signal_fn(price_data, **params)`
in-process to build the signal matrix before shipping it to C++ in one call —
it never inspects *how* the signal was produced.

For LLM/JSON tool-calling rather than a direct Python callable, see the
`run_custom_signal_backtest` agent tool in
[09_advanced_agent_tools.md](09_advanced_agent_tools.md#11-custom-signal-backtest) —
it accepts a pre-computed `{date: value}` signal map instead of a callable
(a raw Python function can't cross a JSON tool-calling boundary).

**One constraint:** without the C++ extension built, `backtest_grid`'s
Python fallback parallelises across parameter combinations via
`ProcessPoolExecutor`, which requires picklable, importable functions.
Lambdas, closures, and locally-defined functions are frequently *not*
picklable, so a custom callable always runs **sequentially** in that fallback
path — `n_workers` is silently ignored for custom callables when C++ isn't
built. Built-in string strategies are unaffected and keep parallelising
exactly as before. Define your signal function at module level (not inside
another function) if you want it to remain picklable for other uses.

---

## Multi-Ticker Signal Panel Backtest

`run_signal_panel_backtest` is the entry point for a **pre-computed signal
matrix across a ticker universe** — e.g. the output of your own cross-sectional
alpha model — without assuming anything about how the signal was generated.
It reuses `run_strategy` per ticker (full C++ speed where available) and
combines the realized returns via the existing portfolio module
(`build_portfolio` / `portfolio_metrics`) — no new backtest math.

```python
from standard_quant_tools.backtest import run_signal_panel_backtest
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL"]
price_data = {t: provider.get_ohlcv(t, "2022-01-01", "2024-01-01") for t in tickers}

# Your own signal panel: index = dates, columns = tickers, values in {-1, 0, 1}
signal_panel = pd.DataFrame({
    t: my_model.signal(price_data[t]) for t in tickers
})

result = run_signal_panel_backtest(
    price_data,
    signal_panel,
    weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},   # optional — default equal weight
)

# Per-ticker backtest results (same shape as run_strategy's output)
print(result["per_ticker"]["AAPL"]["sharpe_ratio"])

# Portfolio-level combination
print(f"Portfolio Sharpe : {result['portfolio_metrics']['sharpe_ratio']:.2f}")
print(f"Portfolio Return : {result['portfolio_metrics']['total_return']:.1%}")
print(result["portfolio_returns"].tail())
```

**Output:**

| Key | Type | Description |
|---|---|---|
| `tickers` | `List[str]` | Universe, in `signal_panel`'s column order |
| `per_ticker` | `Dict[str, dict]` | One `run_strategy`-shaped result per ticker |
| `portfolio_returns` | `pd.Series` | Daily weighted portfolio returns |
| `portfolio_metrics` | `dict` | Same shape as `portfolio.portfolio_metrics()` output |

**Notes:**
- `weights` accepts a list (matching `signal_panel`'s column order) or a `{ticker: weight}` dict; defaults to equal weight. Must sum to 1.0 — validated by the existing `build_portfolio` check.
- Per-ticker equity curves are aligned to their **common date range** (inner join) before combining into the portfolio — a ticker whose `price_data` doesn't fully cover `signal_panel`'s dates will shrink the portfolio's effective range.
- Pass `benchmark_returns=` to get an `information_ratio` in `portfolio_metrics`, and `include_trade_log=True` to get a per-ticker trade log in `per_ticker[ticker]["trade_log"]`.

For LLM/JSON tool-calling, see the `run_signal_panel_backtest` agent tool in
[09_advanced_agent_tools.md](09_advanced_agent_tools.md#12-signal-panel-backtest) —
same idea, JSON-shaped input for function calling.

---

## True Portfolio Simulation (Shared Cash)

`run_signal_panel_backtest` above is fast and useful for research, but it
isn't a real portfolio: every ticker gets its **own independent
`initial_capital`**, and only the resulting *return streams* are blended
afterward via fixed weights. `run_portfolio_simulation`
(`backtest/portfolio_engine.py`) is the true alternative — **one shared cash
balance**, positions sized against **current account equity**, and explicit
**rebalancing** at whichever dates you choose.

```python
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL"]
price_data = {t: provider.get_ohlcv(t, "2022-01-01", "2024-01-01") for t in tickers}

# Rebalance calendar: index = rebalance dates only (not every bar), one
# column per ticker, values = target fraction of account equity.
target_weights = pd.DataFrame(
    {"AAPL": [0.4, 0.3], "MSFT": [0.3, 0.3], "GOOGL": [0.2, 0.3]},
    index=pd.to_datetime(["2022-01-03", "2022-07-01"]),
)

result = run_portfolio_simulation(
    price_data, target_weights,
    initial_capital=100_000.0,
    max_gross_leverage=1.0,   # reject any rebalance date requesting more than fully invested
)

print(f"Final equity : ${result['final_equity']:,.2f}")
print(f"Final cash   : ${result['final_cash']:,.2f}")
print(result["rebalance_log"])           # date, turnover_pct, gross_leverage_after, n_positions
print(result["equity_curve"].tail())      # drifts between rebalances, doesn't jump
```

**Why this is a different engine, not a flag on `run_signal_panel_backtest`:**
between rebalance dates, share counts stay fixed but `equity_curve` still
moves bar-to-bar as prices move — weights **drift** with the market exactly
like a real account. `run_signal_panel_backtest`'s fixed per-bar weighted
blend can't represent that, because it never tracks share counts or cash at
all.

**Output:** `equity_curve`, `cash_curve`, `gross_exposure_curve`,
`net_exposure_curve`, `leverage_curve` (all `pd.Series` — `leverage_curve`
is `gross_exposure_curve / equity_curve`, the continuous version of
`rebalance_log`'s point-in-time `gross_leverage_after`), `rebalance_log`
(`pd.DataFrame`: `date`, `turnover_pct`, `gross_leverage_after`,
`n_positions`), `final_equity`, `final_cash`, `warnings` (e.g. flags if cash
ever went negative — implied margin borrowing).

**Validation (raises `ValidationError` — same self-correcting-error pattern
as everywhere else in this library):** every ticker must be present at every
rebalance date (`target_weights` must be dense); `sum(|weight|)` per
rebalance date can't exceed `max_gross_leverage` (default `1.0` = fully
invested, no leverage); no single `|weight|` can exceed `max_position_pct`
(default `1.0`); every rebalance date must fall on a day all tickers have
price data for (the master trading calendar is the **intersection** of every
ticker's own index).

**Execution timing (`fill_price`):** like `run_strategy`, accepts `"close"`
(default — a rebalance dated D executes at D's own Close), `"next_open"`
(the rebalance instead executes at the *following* bar's Open — one-bar
delay; raises `ValidationError` if the last rebalance date has no following
bar to fill against), or `"midpoint"` (same bar as `"close"`, but at that
bar's `(High + Low) / 2` instead — a bid/ask-free proxy for a midquote
fill). Equity is always marked to Close regardless of `fill_price` — only
the rebalance trade's own execution price changes.

```python
result = run_portfolio_simulation(
    price_data, target_weights, fill_price="next_open",
)
```

**Scope, stated explicitly:** costs are the same flat
`commission_pct`/`slippage_pct` every other tool uses (no per-share/ADV/
impact model yet); short-sale proceeds are credited to cash in full with no
margin/haircut modeling. Both are natural follow-on work, not required for
the shared-cash architecture itself to be correct.

**Feeding it SCORE signals:** `target_weights` above assumes you already
have per-ticker target weights. If you instead have an arbitrary
cross-sectional alpha score per ticker (a `SignalType.SCORE` panel — see
[Custom Signal Backtest](#custom-signal-generation)), convert it to weights
first with `backtest/sizing.py`'s construction functions —
`rank_weighted`, `equal_weight_top_bottom`, `zscore_normalized`,
`vol_scaled`, and the post-processing helper `dollar_neutral` — before
passing the result in as `target_weights`:

```python
from standard_quant_tools.backtest.sizing import zscore_normalized

target_weights = zscore_normalized(my_alpha_scores, gross_leverage=1.0)
result = run_portfolio_simulation(price_data, target_weights)
```

Beta-neutral, sector-neutral, risk-parity, and optimizer-generated weights
are not implemented — each needs infrastructure this repo doesn't have yet
(per-ticker beta/sector metadata, a QP solver).

For LLM/JSON tool-calling, see the `run_portfolio_simulation` agent tool in
[09_advanced_agent_tools.md](09_advanced_agent_tools.md#15-true-portfolio-simulation) —
same idea, JSON-shaped `{ticker: {date: weight}}` input for function calling.

---

## Pair Trade Backtest (Synchronized Two-Leg Execution)

`scan_pairs` (Feature 2 in the agent tools) screens a ticker universe for
cointegrated candidates and reports a current z-score signal, but the
plain per-symbol `run_strategy` can't execute a pair trade as one
synchronized position — each leg would need its own independent state
machine, with no guarantee both legs enter/exit together.
`run_pair_backtest` (`backtest/pairs.py`) closes that gap by treating a
pair trade as a **2-asset portfolio with a dollar-neutral weight vector**:
both legs are columns of the same `target_weights` row passed to
`run_portfolio_simulation`, so they can never fall out of sync — no new
execution engine, just a different way to build the weight panel.

```python
from standard_quant_tools.backtest.pairs import run_pair_backtest

price_data = {"KO": provider.get_ohlcv("KO", "2022-01-01", "2024-01-01"),
              "PEP": provider.get_ohlcv("PEP", "2022-01-01", "2024-01-01")}

result = run_pair_backtest(
    price_data, symbol_a="KO", symbol_b="PEP",
    hedge_ratio=0.85,   # typically from run_cointegration_test / cointegration_test
    entry_z=2.0, exit_z=0.5,
    initial_capital=100_000.0,
)

print(f"Round trips  : {result['n_round_trips']}")
print(f"Final equity : ${result['final_equity']:,.2f}")
print(result["rebalance_log"])
```

**Position logic:** long the spread (long `symbol_a`, short `symbol_b`)
when the z-scored spread (`analysis/cointegration.py`'s `compute_spread` +
`spread_zscore`, same functions `scan_pairs` already uses) falls to or
below `-entry_z`; short the spread on the mirror condition; exit to flat
once the z-score reverts inside `exit_z`. Each leg's weight is sized so the
dollar ratio matches `hedge_ratio`: `weight_a = gross_leverage / (1 +
|hedge_ratio|)`, `weight_b = hedge_ratio * weight_a` (sign flips with
direction) — together they sum to `gross_leverage` at every entry.

**Output:** everything `run_portfolio_simulation` returns (`equity_curve`,
`cash_curve`, `leverage_curve`, `rebalance_log`, `final_equity`,
`final_cash`, `warnings`), plus `hedge_ratio`, `entry_spread` (spread value
at the most recent entry, `None` if the spread never crossed `entry_z`),
`current_spread`, `n_round_trips` (completed entry → exit cycles), and
`state` (`pd.Series` — the daily long/short/flat spread position).

**Validation:** raises `ValidationError` if either symbol is missing from
`price_data`, or if the spread never crosses `entry_z` (nothing to
backtest) — the same self-correcting-error pattern used everywhere else in
this library.

For LLM/JSON tool-calling, see the `run_pair_trade_backtest` agent tool in
[09_advanced_agent_tools.md](09_advanced_agent_tools.md#16-pair-trade-backtest).


---

## Robustness Diagnostics

Sharpe and total return alone don't tell you whether a grid-search result
is real or a fluke of one lucky parameter combination among many tried.
`backtest/robustness.py` adds three independent, same-sample checks — none
of them a substitute for `run_walk_forward_backtest`'s out-of-sample
validation, which answers a different question ("would this have held up
on unseen data" vs. "how sure am I this in-sample number is real").

```python
from standard_quant_tools.backtest.robustness import (
    block_bootstrap_ci, parameter_sensitivity, deflated_sharpe_ratio,
)
from standard_quant_tools.backtest import backtest_grid
from standard_quant_tools.metrics.risk_metrics import sharpe_ratio

grid_df = backtest_grid(df, strategy="sma_crossover",
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]})

# 1. How much better is the best row than the pack?
sensitivity = parameter_sensitivity(grid_df, metric_col="sharpe_ratio")

# 2. Does the best Sharpe survive correcting for having been selected as
#    the max of len(grid_df) attempts?
dsr = deflated_sharpe_ratio(
    observed_sharpe=grid_df.iloc[0]["sharpe_ratio"],
    sharpe_trials_std=grid_df["sharpe_ratio"].std(),
    n_trials=len(grid_df), n_obs=len(df),
)

# 3. Confidence interval on the best trial's own Sharpe ratio, via block
#    bootstrap on its daily returns (preserves short-range autocorrelation
#    an i.i.d. resample would destroy).
ci = block_bootstrap_ci(best_trial_returns, sharpe_ratio, block_size=20, seed=42)
```

**`parameter_sensitivity(grid_df, metric_col="sharpe_ratio")`:** returns
`n_trials`, `best`, `median`, `best_minus_median`, `best_minus_rank2`, and
`best_minus_top5_mean`. A large gap on a small grid is a red flag for
overfitting — the "best" combination may just be the one that happened to
fit this particular sample.

**`deflated_sharpe_ratio(observed_sharpe, sharpe_trials_std, n_trials, n_obs, skew=0.0, kurtosis=3.0)`:**
implements Bailey & López de Prado's Deflated Sharpe Ratio (2014).
`sharpe_trials_std` is the standard deviation of the Sharpe ratios actually
observed across the grid search (`grid_df["sharpe_ratio"].std()`) — a
measured quantity from the real search, not an assumed theoretical one.
Returns `expected_max_sharpe` (the bar `observed_sharpe` must clear, given
`n_trials` independent attempts) and `deflated_sharpe_ratio` (a probability
in `[0, 1]`: how likely the true Sharpe ratio exceeds zero after that
correction). `n_trials <= 1` skips the correction entirely
(`expected_max_sharpe = 0.0`) — no selection bias to correct for with a
single trial. Implemented with a self-contained normal CDF/inverse-CDF
(`math.erf` + Acklam's rational approximation) rather than a hard `scipy`
dependency.

**`block_bootstrap_ci(returns, metric_fn, n_iterations=1000, block_size=20, confidence=0.95, seed=None)`:**
resamples overlapping blocks of `block_size` consecutive returns with
replacement (not an i.i.d. resample, which would destroy short-range
autocorrelation), recomputes `metric_fn` on each resample, and reports the
percentile-based confidence interval. `metric_fn` can be any callable
`pd.Series -> float` — `sharpe_ratio` from
`metrics/risk_metrics.py` is the typical choice, but total-return or any
other metric function works identically.

For LLM/JSON tool-calling, see the `get_robustness_diagnostics` agent tool
in [09_advanced_agent_tools.md](09_advanced_agent_tools.md#17-robustness-diagnostics)
— runs a grid search internally and reports all three checks on the best
trial in one call.

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
