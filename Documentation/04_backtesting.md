# Backtesting

The backtesting engine is **fully vectorized** — it computes the entire equity curve in one NumPy operation instead of looping bar-by-bar. This makes it orders of magnitude faster than event-driven backtesting for signal-based strategies.

**Signal generators** (`rsi_mean_reversion`, `bollinger_reversion`, `donchian_breakout`, `vwap_reversion`) use a Numba JIT-compiled state machine for their entry/exit tracking loops, providing ~50–100× speedup over a plain Python loop on long series. This is especially impactful inside `backtest_grid` where the same loop runs thousands of times across parameter combinations. The JIT path requires `numba` installed with NumPy ≤ 2.0; pure Python is used otherwise. (`momentum_timeseries` and `adx_trend` need no state machine at all — every bar's signal depends only on that bar's own already-vectorized values — so they're plain pandas/numpy regardless of numba availability.)

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

The optional per-trade log (`include_trade_log=True`) still runs in Python — it requires DatetimeIndex-aware iteration to produce labeled entry/exit dates.

**Trade-stat parity (single-call vs. batch) — verified.** `run_strategy`'s single-call C++ path and `backtest_grid`'s batch kernel used to disagree on `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct`: the native kernel's own trade-log construction recorded entries one bar late and excluded commission/slippage, so `run_strategy` masked this by overwriting those four fields with a correct Python recomputation (`_build_trade_log`), while `backtest_grid`'s batch path had no such override and returned the native kernel's uncorrected numbers as-is. `backtest.cpp`'s native trade-log construction was rewritten to match `_build_trade_log`'s accounting exactly (entry size = signal magnitude rather than just its sign, `prices[i-1]` as the reference price, correct commission/slippage deduction).

Agreement is now **confirmed against a real compiled `_sqt_core`**: all three tests in `tests/backtest/test_backtest.py::TestNativeTradeStatsCorrectness` pass with the extension built, covering the single-call path, the batch path, and a native-vs-Python-recomputation cross-check. (An earlier revision of this section deferred that confirmation to CI because the fix had been written without a C++ toolchain available; that is no longer the case.)

**What counts as one trade.** A trade is one **lot**: from the moment exposure leaves zero until it returns to zero. Same-sign resizes and partial reductions happen *inside* a trade rather than ending one, and cost is charged per event on the amount actually transacted — the same `sum(abs(pos_diff))` the equity curve charges. For a lot that was resized, `entry_price` is the weighted-average cost basis across the whole lot and `position_size` is its peak exposure.

Both implementations now share that definition. They did not always: `backtest.cpp` moved to weighted-average cost basis in the earlier C++ pass while `engine.py`'s `_build_trade_log` emitted a completed trade for *every* position-changing event, so one result dict could report `num_trades=1` beside a two-row `trade_log`. On a 1.0 → 2.5 → 0 sequence that was native 1 trade averaging 17.4492% against a Python log of 2 trades averaging 8.5113%, from identical inputs.

This was documented at the time as a *resize* problem. It was broader than that, and worth knowing if you read any trade log written by an older version: a partial **reduce** diverged identically. `2.0 → 1.0` is opposite-sign without being a full close, so the old code booked it as a completed trade — on `0 → 2.0 → 1.0 → 0`, containing no resize at all, native reported 1 trade at 12.8078% against an old log of 2 rows averaging 6.1583%. On a realistic 100-bar signal series the old log produced **67 rows against the kernel's 50**, with the average trade return off by **0.087pp**. Fixed in the second modeling audit's item 20; `TestNativeTradeStatsCorrectness` now *includes* resizes and partial reduces in its cross-check and asserts `num_trades == len(trade_log)`.

## "Unknown" is never reported as the benign value

Several places used a valid-looking number as a failure sentinel, and each
biased toward the reassuring answer. They now return `NaN`, which cannot be
mistaken for a measurement:

| Function | Missing input used to give | Why that was backwards |
|---|---|---|
| `adv_participation` | `0.0` | 0.0 participation is the score of a trade so small it barely moves the market. A billion-dollar order in a name with **no volume data** ranked as the easiest trade in the universe — against `100.0` (100× ADV) for a name with real data |
| `impact_cost` | `0.0` | `$0` impact against **$3bn** for the same order with a real ADV, so a capacity report routed size into exactly the names it knew least about |
| `calculate_beta` | `beta: 0.0` | Indistinguishable from a genuinely market-neutral asset, so `treynor_ratio` turned "no overlapping benchmark data" into a plausible risk-adjusted return |

Because every comparison against `NaN` is `False`, code that *gates* on these
must test finiteness explicitly rather than relying on a comparison — a
`max_adv_participation` limit would otherwise be silently satisfied by absent
data. `run_portfolio_simulation` rejects an unestimable participation by name
rather than letting it pass a constraint that a merely large trade fails.

The same applies to guards written as comparisons: `days_to_liquidate` checked
`avg_daily_volume <= 0`, which `NaN` does not satisfy, so a `NaN` volume
produced a `NaN` answer that looked computed. It now checks finiteness first.

## Signal panels keep the full trading calendar

`run_strategy` intersects price dates with signal dates and then takes
`pct_change()` over **what remains**. A signal series sparser than the price
series therefore does not read as "hold" — the intervening trading days
disappear from the price axis entirely and the bars either side become
adjacent, so a monthly signal turns Jan 31 → Feb 28 into a single "bar"
carrying a month of price movement.

Measured on a 120-bar daily series driven by identical exposure, once with a
daily signal and once with the same signal sampled monthly:

| | bars used | annualized volatility |
|---|---|---|
| daily signal | 120 | 0.0241 |
| monthly signal | 4 | 0.7735 |

A **32× distortion of risk from the same prices**. Total return can still look
correct, which is what made this easy to miss; per-bar volatility, Sharpe and
drawdown are all wrong.

`backtest.panel.run_signal_panel_backtest` now reindexes every ticker's signal
onto that ticker's own full price calendar before running, controlled by
`signal_calendar_policy`:

- **`hold`** (default) — carry the last signal forward. This is what a
  rebalance schedule means: a monthly signal is a position held *through* the
  month, not one that exists on a single day.
- **`flat`** — `0.0` between signal dates, i.e. in the market only on dates
  that carry a signal.
- **`error`** — refuse, naming how many bars lack a signal.

`hold` deliberately does **not** back-fill before the first signal. No view
had been expressed yet, and back-filling one would be look-ahead.

**Cross-backend parity generally.** Both audit passes (see `CHANGELOG.md`) specifically hunted for cases where the same call returns a different answer depending on whether `_sqt_core` is built. Four were found and fixed — `stochastic_oscillator` on a zero-range window, `cointegration_test`'s `autolag` handling, `hurst_exponent`'s regime post-processing, and `rolling_factor_loadings` on an underdetermined window — plus `profit_factor`'s 0/0 case described under [Understanding the Output](#understanding-the-output) below. Each is now pinned by a test asserting the two backends against *each other* rather than against a constant, since a test pinning only one side cannot see a divergence.

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

## Every fill mode runs natively

`next_open` and `hl2_exploratory` used to force the Python path, because the
compiled kernel only knew Close prices — so the **more realistic execution
model was also the slow one**, and grid search could not use it at all. The
kernel now takes an optional per-bar reference (fill) price and applies the
same two-leg overnight/intraday decomposition, measured on 20,000 bars:

| `fill_price` | native | Python fallback | speedup |
|---|---|---|---|
| `close` | 2.60 ms | 296.86 ms | **114×** |
| `next_open` | 4.20 ms | 307.88 ms | **73×** |
| `hl2_exploratory` | 6.39 ms | 321.08 ms | **50×** |

A `next_open` grid now costs about what a `close` grid costs.

## Crossover grids are fused, and never build a signal matrix

Profiling a 300-combination × 5,000-bar SMA grid showed the batch kernel was
solving the small half of the problem:

```
python signal generation   121.4 ms   92.1%
vstack into (combos,bars)    3.2 ms    2.4%
native batch backtest        7.2 ms    5.4%
```

It also computed 600 moving averages where only **35 unique periods** existed.

`backtest_grid` now computes each unique indicator once — through the same
`sma` the strategy itself uses, so there is no second definition to drift —
and C++ builds each combination's signal into a single reusable buffer and
backtests it immediately:

| grid | before | after | speedup |
|---|---|---|---|
| 300 combos × 5,000 bars | 167.2 ms | 12.2 ms | **13.7×** |
| 2,000 combos × 10,000 bars | 1,370.6 ms | 79.9 ms | **17.2×** |

Results are bit-identical to the general path. Peak memory becomes
`O(unique_periods × bars)` rather than `O(combos × bars)` — at the 50,000
combination cap over 100,000 bars, **40 GB → 72 MB**.

Grids that are not a plain fast/slow crossover fall back to the general path
automatically.

### Controlling parallelism

Native kernels used to parallelize whenever there was more than one task,
which oversubscribes badly when Standard Tools is itself running inside a
`ProcessPoolExecutor`, several agents, or replicated containers. The decision
is now based on total work, and two environment variables govern it:

| variable | default | meaning |
|---|---|---|
| `SQT_NUM_THREADS` | unset | Ceiling on threads any kernel may use. **Set to `1` inside a process pool.** |
| `SQT_OMP_MIN_WORK` | `50000` | Minimum tasks × elements before a region goes parallel at all. |

## Annualization is a parameter, not an assumption

Every annualized metric — volatility, Sharpe, Sortino, Calmar — scales by a
bars-per-year factor. The native kernel hard-coded `252`, which is right for
daily equity bars and wrong for everything else the data layer now supports:
1h, 5m, 1m and 24/7 markets. An hourly backtest reported a "Sharpe" annualized
as though its bars were trading days.

`periods_per_year` is now a parameter on `run_strategy`,
`run_strategy_summary` and `batch_run_strategy`, defaulting to `252` so
existing callers are unchanged. Python resolves the calendar; the kernel stays
calendar-agnostic.

**Calmar counts intervals, on both backends.** N level observations span N−1
return intervals. Python was corrected first, which left the native kernel
disagreeing about the same backtest by **4.01% on a 21-bar series** (1.79% at
63, 0.51% at 252) — negligible on long histories, material on exactly the
short windows a walk-forward fold uses. A wiped-out strategy now also reports
−1.0 on both backends rather than `0.0` natively, which read as *neutral*
rather than as a total loss.

## Walk-forward optimizes and evaluates under the same execution model

`backtest_grid` defaults to `fill_price="close"`, and the walk-forward tools
did not pass the caller's mode into it — while the out-of-sample leg honoured
it. A run requesting `next_open` therefore selected parameters under
same-close execution and scored them under next-open execution.

Measured across 25 random series with a realistic overnight gap, the **winning
parameter pair differed between the two fill modes on 7 of them**, so the
out-of-sample number was not a test of the parameters actually chosen. Both
walk-forward tools now pass `fill_price` into the in-sample grid.

> `next_open` and `hl2_exploratory` force the Python path, because the C++
> batch kernel only knows Close prices — so the more realistic execution mode
> is currently the slower one. A fill-aware native kernel is the natural next
> step and would remove that trade-off.

## Strategy parameters are validated

Every entry in `STRATEGY_REGISTRY` resolves its parameters through
`backtest/strategy_params.py` before it computes a single signal. This is the
same contract the modeling runtime applies to features, applied to the eight
classic strategies — which previously validated **nothing**.

The reason it matters is not tidiness:

```python
momentum_timeseries(lookback=-20)     # rejected now
```

reached `Close.pct_change(periods=-20)` unchecked, and pandas reads a
*negative* period as a **forward** window. Standing at bar 25 it returns
`close[25] / close[45] - 1` — so the signal for a bar was computed from a
price twenty bars into its own future. An ordinary-looking integer produced a
backtest with look-ahead built in, and it was reachable from the agent surface
because `BacktestInput.parameters` was an unconstrained `Dict[str, Any]`.

The contract:

| Rule | Why |
|---|---|
| Windows are positive integers | A negative period is a forward window (above); zero is degenerate |
| Thresholds must be finite | NaN fails every comparison, so it does not tighten a strategy — it silently makes it inert, which looks exactly like a strategy that honestly found no trades |
| Values stay in declared ranges | A negative Bollinger `num_std` puts the "upper" band below the "lower" one, inverting every entry and exit while still producing plausible output |
| Unknown parameter names are rejected | Every signature ends in `**_`, so a typo was swallowed and the strategy silently ran its default while the caller believed it had configured something |
| Relations hold (`fast < slow`, `oversold < overbought`) | Each value can be individually valid while the pair is nonsense |

Validation is attached to `STRATEGY_REGISTRY` itself, not to each call site,
so it cannot be reached around — including from the `ProcessPoolExecutor`
grid worker, which rebuilds its call in a child process.

> **Cross-parameter relations are the one exception to "always on".** They are
> enforced where a single configuration is deliberately requested (the agent
> tools), and skipped inside the registry, because a parameter grid
> legitimately sweeps a rectangle containing `fast >= slow` and
> `backtest_grid` does not catch per-combination errors — enforcing them
> there would abort a whole sweep over points a search should simply score
> badly and move past.

## Look-ahead warnings reach the caller

`BacktestResult.warnings` carries the caveats `run_strategy` raises, most
importantly the `fill_price="close"` one: a signal derived from bar *t*'s own
close cannot realistically be filled at that same close. The engine has always
emitted it, but the agent-facing model had no field for it and rebuilt the
result without it — so the engine knew a simulation might contain look-ahead
while the output an LLM reads said nothing. `fill_price` and `strategy_type`
are `Literal`-typed, so an unsupported value is rejected at the boundary
rather than deep inside dispatch.

**Validation:** `run_strategy` raises `ValidationError` if `initial_capital`
isn't finite and `> 0`, or if `commission_pct`/`slippage_pct` isn't finite
and `>= 0` — the same self-correcting-error pattern used everywhere else in
this library. Previously a zero, negative, or non-finite `initial_capital`
was accepted silently and produced `inf`/`nan` in `total_return`/
`calmar_ratio` instead of raising.

---

## Execution Timing (`fill_price`)

By default (`fill_price="close"`), a signal known at bar *t-1*'s close is
assumed filled at that same close, earning bar *t*'s full close-to-close
return — the standard lookahead-free convention (see Core Concepts above).
This is optimistic in one sense: it assumes you can transact right at the
closing price the instant you observe it.

`fill_price="next_open"` is more conservative: entries and exits are priced
off the bar's own `Open` instead. `fill_price="hl2_exploratory"` uses that
bar's own `(High + Low) / 2` ("HL2") instead — this is **not** a real
bid/ask midpoint quote, and it requires knowing the bar's High and Low,
which are only determined once that bar has already completed, so pricing
a fill at a bar's own HL2 is look-ahead the same way `fill_price="close"`
is (a warning is emitted, same as for `"close"`) — the name says
"exploratory" deliberately, so it's never mistaken for a real, tradable
execution price. Both `"next_open"` and `"hl2_exploratory"` decompose each
bar into two legs:

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
result_close = run_strategy(df, signals, fill_price="close")             # default
result_next_open = run_strategy(df, signals, fill_price="next_open")     # more conservative
result_hl2 = run_strategy(df, signals, fill_price="hl2_exploratory")     # exploratory only — see above
```

`backtest_grid` accepts the same `fill_price` argument and forces the
Python execution path when it isn't `"close"` — the compiled C++ kernel
only knows `Close` prices, so `next_open`/`hl2_exploratory` always run in
Python regardless of whether `_sqt_core` is built.

**Validation:** `run_strategy` raises `ValidationError` if `fill_price` isn't
one of `"close"`, `"next_open"`, `"hl2_exploratory"` — the same
self-correcting-error pattern used everywhere else in this library.

**Trade log price convention:** `entry_price`/`exit_price` use the same
reference price the equity curve's P&L is actually computed from, per
mode — not always `Close`. Under `fill_price="close"`, `executed[i] =
signals[i-1]`, so a position that "appears" in the executed-position
series at bar *i* actually earns its first return over `Close[i-1] ->
Close[i]`: the trade log reports `Close[i-1]` as that entry (and, at exit,
the analogous `Close[j-1]` for the closing event at bar *j*) — not
`Close[i]`/`Close[j]`, which would be one bar later than the price the
equity curve actually used. Under `next_open`/`hl2_exploratory`, the
two-leg decomposition already prices entries/exits at that bar's own
reference price (`Open`/HL2), so no shift is needed there — `entry_price`/
`exit_price` equal that bar's reference price directly.

For a lot that changed size during its life, `entry_price` is the
**weighted-average cost basis** across the whole lot rather than any single
bar's reference price — see [What counts as one trade](#c-acceleration)
above. That is the price its `return_pct` is actually measured against, but
it is a computed basis, not a level that necessarily ever traded: a lot
opened at 100 and doubled at 110 reports `entry_price` 105 even though no
bar printed 105. Read it as a basis, not as a fill.

`return_pct` is also net of commission+slippage: `cost_per_unit` is charged
per position-changing event on `abs(pos_diff)`, the amount actually
transacted, which is the same total the equity curve deducts. For a simple
open-and-close lot that is the familiar 2× `abs(position_size)`; for a lot
that resized or was partially trimmed it is the sum over its events, and
for a position still open at the final bar the closing charge is absent
entirely, since no real exit event/cost was ever applied to the equity
curve either. The result matches, up to a small second-order compounding
residual, the equity curve's own return over the trade's span. A position still open at the final bar (no
exit signal) is marked at that bar's `Close` regardless of `fill_price`,
matching how the equity curve itself is always marked to `Close`.

On the agent-tool side, `fill_price` is exposed on `BacktestInput`,
`BuyAndHoldInput`, `CompareStrategiesInput`, `CustomSignalBacktestInput`,
`SignalPanelBacktestInput`, `BacktestOptInput`, `BacktestDiagnosticsInput`,
`WalkForwardInput`, `RegimeAdaptiveWalkForwardInput`, `PortfolioSimulationInput`,
`PairTradeBacktestInput`, and `BacktestCompactInput` — every tool whose
out-of-sample/simulated leg ultimately calls `run_strategy`,
`run_portfolio_simulation`, or `run_pair_backtest`. The one holdout is
`run_regime_adaptive_backtest` (the older, in-sample exploratory tool, kept
deliberately simple — see its own docstring), which still always uses
`"close"`.

### Input validation contract

`run_strategy` rejects non-finite (NaN/Inf) input **on every path**, and
the check covers whichever price columns the chosen `fill_price` actually
reads:

| `fill_price` | Columns required and validated |
|---|---|
| `"close"` | `Close` |
| `"next_open"` | `Close`, `Open` |
| `"hl2_exploratory"` | `Close`, `High`, `Low` |

A missing column raises a `ValidationError` naming both the column and the
fill mode, rather than surfacing as a bare `KeyError` from inside the return
calculation. `signal_series` is validated for finiteness too.

This contract used to be enforced only inside the `fill_price="close"` C++
branch, which had two consequences worth knowing if you are upgrading:

- **The same call behaved differently depending on whether `_sqt_core` was
  built** — raising with the extension present, silently producing NaN
  metrics without it.
- **`"next_open"`/`"hl2_exploratory"` were never validated at all.** A NaN
  `Open` did not merely poison the result: because `Series.cumprod()` skips
  NaN, that bar's P&L was silently *dropped* from the compounded curve, so
  `total_return` was computed over a quietly shortened series that still
  looked complete. Any historical result produced from a frame with gaps in
  `Open`/`High`/`Low` under those fill modes is worth re-running.

---

## Trade Log

When `include_trade_log=True`, the result includes a `pd.DataFrame` with one row per completed trade.

```python
trade_log = result['trade_log']
print(trade_log.columns)
# ['entry_date', 'exit_date', 'direction', 'entry_price', 'exit_price', 'position_size', 'return_pct']

# Best and worst trades
print(trade_log.nlargest(3, 'return_pct'))
print(trade_log.nsmallest(3, 'return_pct'))

# Average holding period
trade_log['holding_days'] = (pd.to_datetime(trade_log['exit_date'])
                             - pd.to_datetime(trade_log['entry_date'])).dt.days
print(f"Avg holding: {trade_log['holding_days'].mean():.0f} days")
```

**`position_size`:** the actual signal value held during the trade (its
sign gives `direction`) — exactly `1.0`/`-1.0` for a `DIRECTION`-type
signal, but a fractional or leveraged number (e.g. `2.5`) for a `SCORE`-type
signal, since `run_strategy` multiplies the signal value directly into
`strategy_return = lagged_signal * market_return`. `return_pct` already
scales with `position_size` — it is not silently treated as if every trade
were exactly 1x/-1x.

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

### Donchian Breakout, Momentum, VWAP Reversion, ADX Trend

Four more built-in strategies, registered in `backtest.strategies.STRATEGY_REGISTRY` alongside the original four. They don't have dedicated `run_*_backtest` tools (only `run_sma_backtest`/`run_rsi_backtest`/`run_macd_backtest`/`run_bollinger_backtest` do) — use them via `backtest_grid`, `get_backtest_diagnostics`, or `run_backtest_compact`, which all accept any registered strategy name as a plain string:

```python
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

signals = STRATEGY_REGISTRY["donchian_breakout"](df, entry_period=20, exit_period=10)
result = run_strategy(df, signals, initial_capital=10_000)
```

- **`donchian_breakout`** (`entry_period=20`, `exit_period=10`) — Turtle-style channel breakout. Long on a new `entry_period`-bar high (measured against the *prior* bars via `.shift(1)`, not today's own high — a genuine breakout past the already-established channel, not a tautology); flat again on a new `exit_period`-bar low. `entry_period > exit_period` is the classic asymmetric design (slower entry, faster exit).
- **`momentum_timeseries`** (`lookback=90`, `threshold=0.0`) — time-series (absolute) momentum: long when the trailing `lookback`-bar return exceeds `threshold`. No per-bar state at all — a single vectorized `pct_change(periods=lookback)` call, the cheapest strategy in the registry on very large series.
- **`vwap_reversion`** (`period=20`, `entry_threshold=0.02`) — mean reversion to a rolling VWAP rather than a plain price mean (contrast with `bollinger_reversion`). Aimed at intraday/tick data, where VWAP is the standard fair-value benchmark: enter long when Close drops `entry_threshold` below its own trailing VWAP, exit once it recovers to VWAP.
- **`adx_trend`** (`adx_period=14`, `adx_threshold=25.0`) — trend-strength-filtered directional strategy: long only when ADX confirms a genuinely trending market *and* `+DI > -DI`. No state machine — a single vectorized boolean condition on the `adx()` indicator's own output.

**Performance, stated explicitly (all four were written with per-tick/million-row series in mind):** `donchian_breakout` and `vwap_reversion` use the same numba-JIT entry/exit state-machine pattern as `rsi_mean_reversion`/`bollinger_reversion` — no interpreted Python loop over the series regardless of length. `momentum_timeseries` and `adx_trend` need no state machine at all (a bar's signal depends only on that bar's own already-vectorized indicator/rolling values), so they're pure pandas/numpy — O(n), not O(n·window). All four were benchmarked in `tests/backtest/test_strategies.py`'s `TestScalesToLargeSeries` on 500k-bar synthetic series and complete in well under a second.

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
    strategy="sma_crossover",          # or any other STRATEGY_REGISTRY name
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

All eight registered strategies are supported:

```python
results = backtest_grid(df, strategy="rsi_mean_reversion",
    param_grid={"period": [7, 14, 21], "oversold": [25, 30], "overbought": [65, 70]})

results = backtest_grid(df, strategy="macd_crossover",
    param_grid={"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]})

results = backtest_grid(df, strategy="bollinger_reversion",
    param_grid={"period": [15, 20, 25], "num_std": [1.5, 2.0, 2.5]})

results = backtest_grid(df, strategy="donchian_breakout",
    param_grid={"entry_period": [10, 20, 40], "exit_period": [5, 10, 20]})

results = backtest_grid(df, strategy="momentum_timeseries",
    param_grid={"lookback": [30, 60, 90], "threshold": [0.0, 0.05]})

results = backtest_grid(df, strategy="vwap_reversion",
    param_grid={"period": [10, 20, 40], "entry_threshold": [0.01, 0.02, 0.03]})

results = backtest_grid(df, strategy="adx_trend",
    param_grid={"adx_period": [10, 14, 21], "adx_threshold": [20.0, 25.0, 30.0]})
```

Pass `n_workers=1` to run sequentially (no subprocess overhead — useful in notebooks).

**Validation:** `backtest_grid` applies the same checks as `run_strategy`
(see [Running a Backtest](#running-a-backtest) above) — `initial_capital`
must be finite and `> 0`, and `commission_pct`/`slippage_pct` must each be
finite and `>= 0` — raising `ValidationError` up front rather than letting a
bad value silently produce `inf`/`nan` metrics across every combination in
the grid.

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

`backtest/strategy.py`'s `VectorizedStrategy` is the formal type for
`my_signal` above — every `STRATEGY_REGISTRY` entry already satisfies it
structurally, so annotating a custom callable as `VectorizedStrategy`
documents the contract without changing any behavior:

```python
from standard_quant_tools.backtest.strategy import VectorizedStrategy

def my_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
    ...

_: VectorizedStrategy = my_signal  # type-checker verifies the signature matches
```

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
- `weights` accepts a list (matching `signal_panel`'s column order) or a `{ticker: weight}` dict; defaults to equal weight. Validation happens in `run_signal_panel_backtest` itself, up front: the weights must sum to 1.0, a dict must have an entry for every ticker and no extras, and a list must match the ticker count. Earlier revisions of this page claimed the sum was "validated by the existing `build_portfolio` check" — it was not, and nothing enforced it: a dict missing a ticker raised a bare `KeyError` naming only that ticker, a wrong-length list silently misaligned weights against columns, and weights summing to anything else produced a scaled portfolio that still looked valid.
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
`n_positions`), `final_equity`, `final_cash`, `warnings` (`list[str]`).
`warnings` **always** includes a look-ahead-bias notice when `fill_price`
is left at its default `"close"` and at least one rebalance occurs — each
rebalance executes at the same bar's own Close that its target weight is
dated on, which is only lookahead-free if `target_weights` was itself
derived from data known before that bar's Close; use `fill_price="next_open"`
for a lookahead-free simulation to silence it. It also flags if cash ever
went negative (implied margin borrowing).

**Validation (raises `ValidationError` — same self-correcting-error pattern
as everywhere else in this library):**
- `target_weights` must have at least one ticker column (empty universe
  rejected) and at least one rebalance date.
- `target_weights.index` must be free of duplicate dates and sorted in
  increasing order — an unsorted or duplicated rebalance calendar is
  rejected outright rather than silently re-sorted or silently only
  honoring one of the duplicates.
- `target_weights` must be dense — every ticker must be present at every
  rebalance date, and no cell may be `NaN` (a `NaN` used to silently
  corrupt the equity curve; it now fails fast with the offending
  dates/tickers listed) or infinite.
- `sum(|weight|)` per rebalance date can't exceed `max_gross_leverage`
  (default `1.0` = fully invested, no leverage); no single `|weight|` can
  exceed `max_position_pct` (default `1.0`). See "Post-trade enforcement"
  below for what these two limits do and do not guarantee once costs are
  applied.
- Every rebalance date must fall on a day all tickers have price data for
  (the master trading calendar is the **intersection** of every ticker's
  own index), and every price on that calendar (for the columns
  `fill_price` actually needs) must be finite and strictly positive —
  a `NaN`/`inf`/zero/negative price anywhere on the calendar is rejected
  upfront rather than surfacing as a corrupted downstream number.
- `initial_capital` and every cost parameter (`commission_pct`,
  `slippage_pct`, `per_share_rate`, `min_commission`, `borrow_fee_bps`,
  `margin_interest_rate`, `impact_coefficient`) must be finite and
  non-negative (`initial_capital` strictly positive); `max_gross_leverage`,
  `max_position_pct`, `impact_lookback`, and `max_adv_participation` (when
  set) must be finite and strictly positive.

**Post-trade enforcement — target weights vs. realized, post-cost state:**
`max_gross_leverage`/`max_position_pct` bound the *target* weights (validated
upfront, as above), and each rebalance's `target_shares` are sized from
`equity_now` — account equity immediately **before** that rebalance's own
costs are deducted — so the resulting gross exposure is exactly
`sum(|weight|) * equity_now` by construction. Once that rebalance's costs are
deducted, `equity_after < equity_now` while the share positions (hence gross
exposure) are unchanged, which mechanically pushes the **realized**,
post-cost ratio `gross_after / equity_after` — reported as
`gross_leverage_after` in `rebalance_log` and continuously in
`leverage_curve` — slightly *above* `sum(|weight|)`. This is expected,
unavoidable cost drag, not a limit violation, and is **not** rejected. What
**is** re-checked after every rebalance, and **can** raise `ValidationError`,
is `gross_after / equity_now` (and the largest single position's
`weight / equity_now`) exceeding the limit — a sizing self-consistency
invariant that should already be guaranteed by the per-date weight
validation above, so a violation here indicates an actual sizing bug, not
ordinary cost drag. If you need a hard ceiling on realized, cost-inclusive
leverage, monitor `leverage_curve`/`rebalance_log`'s `gross_leverage_after`
yourself, or request a `max_gross_leverage` a little below your true risk
limit to absorb the cost-drag headroom.

**Insolvency:** this engine models a cash-settled account with no
forced-liquidation/margin-call machinery, so account equity reaching zero or
negative — whether from a rebalance's own costs or simply from price moves
between rebalances — has no meaningful next state to simulate. Both cases
raise `ValidationError` immediately rather than continuing with a
`leverage_curve` divide-by-negative-equity or an annualized-return
calculation raising on a negative base.

**Execution timing (`fill_price`):** like `run_strategy`, accepts `"close"`
(default — a rebalance dated D executes at D's own Close), `"next_open"`
(the rebalance instead executes at the *following* bar's Open — one-bar
delay; raises `ValidationError` if the last rebalance date has no following
bar to fill against), or `"hl2_exploratory"` (same bar as `"close"`, but at
that bar's own `(High + Low) / 2` instead — **not** a real bid/ask midpoint
quote, and only knowable after that bar has already completed, so this is
look-ahead the same way `"close"` is; a warning is emitted, same as for
`"close"`). Equity is always marked to Close regardless of `fill_price` —
only the rebalance trade's own execution price changes.

```python
result = run_portfolio_simulation(
    price_data, target_weights, fill_price="next_open",
)
```

**Pluggable cost models (`backtest/costs.py`):** beyond the default flat
`commission_pct`/`slippage_pct`, `run_portfolio_simulation` accepts:

```python
result = run_portfolio_simulation(
    price_data, target_weights,
    commission_model="per_share", per_share_rate=0.005, min_commission=1.0,
    use_impact_model=True, impact_coefficient=1.0, impact_lookback=20,
    borrow_fee_bps=50.0,          # annualized, accrued daily on short notional
    margin_interest_rate=0.06,    # annualized, accrued daily on negative cash
)
```

- `commission_model`: `"pct"` (default, unchanged) or `"per_share"` —
  `per_share_rate` per share traded, floored at `min_commission`.
- `use_impact_model`: adds a square-root market-impact cost
  (`impact_bps = impact_coefficient * volatility * sqrt(participation)`,
  `participation = trade notional / rolling avg dollar volume`) on top of
  commission + spread. Requires a `'Volume'` column — no other new data
  dependency, since `Close * Volume` is already computable from any OHLCV
  frame.
- `borrow_fee_bps` / `margin_interest_rate`: financing costs accrued on
  short notional / negative cash respectively, charged once per bar based
  on the position/cash carried in from the previous bar. The accrual uses
  the **actual elapsed calendar days** since the prior bar (`(date -
  prev_date).days` — e.g. 3 over a Friday→Monday weekend gap, or more
  across a holiday), not a hardcoded `days=1.0` — a fixed 1-day assumption
  would under-accrue financing across every weekend/holiday gap in the
  trading calendar. Both default to `0.0` (today's exact behavior — no
  financing cost beyond the existing "cash went negative" warning).

**Liquidity constraint:** pass `max_adv_participation=0.1` to reject (raise
`ValidationError`) any rebalance trade whose notional exceeds 10% of the
ticker's own rolling average dollar volume — same fail-fast pattern as
`max_gross_leverage`/`max_position_pct`. Requires a `'Volume'` column. Built
on `backtest/constraints.py`'s `adv_participation()` — see the "Liquidity &
Capacity Diagnostics" section below for the full module, including the
standalone `capacity_report()` (not wired into the simulation itself).

**Scope, stated explicitly:** short-sale proceeds are credited to cash in
full with no margin haircut modeling beyond the flat `margin_interest_rate`
accrual above; sector-exposure constraints (as opposed to reporting — see
`get_capacity_report`) aren't enforced by the engine itself. Neither is
required for the shared-cash architecture itself to be correct.

### Position Sizing (`backtest/sizing.py`)

`target_weights` above assumes you already have per-ticker target weights.
If you instead have an arbitrary cross-sectional alpha score per ticker (a
`SignalType.SCORE` panel — see
[Custom Signal Backtest](#custom-signal-generation)), convert it to weights
first. Every function takes/returns a `pd.DataFrame` of the same shape
(dates × tickers) and scales each row's gross exposure (`sum(|weight|)`) to
`gross_leverage`; all raise `ValidationError` on an empty or NaN-containing
`scores` panel.

| Function | Signature | Behavior |
|---|---|---|
| `rank_weighted` | `(scores, gross_leverage=1.0)` | Weight ∝ cross-sectional rank, centered on the row's mean rank (long top, short bottom); `sum(weight) ≈ 0` automatically. |
| `equal_weight_top_bottom` | `(scores, n_long, n_short, gross_leverage=1.0)` | Equal-weight the top `n_long` and bottom `n_short` names each row; everything else gets 0. When **both** sides are active, `gross_leverage` is split 50/50 between them. When only one side is active (`n_long=0` or `n_short=0`), that one side gets the **full** `gross_leverage` — not half of it — so a long-only or short-only request isn't silently sized at half the requested gross exposure. Raises if `n_long + n_short` exceeds the ticker count, or if both are 0. |
| `zscore_normalized` | `(scores, gross_leverage=1.0)` | Weight ∝ cross-sectional z-score; a row with zero cross-sectional std gets all-zero weight rather than a division-by-zero blowup. |
| `vol_scaled` | `(scores, returns_df, lookback=20, gross_leverage=1.0)` | Divide each score by its trailing realized volatility, then apply the same normalization as `zscore_normalized`. The rolling-std window runs on `returns_df`'s own (daily) frequency **first**, and is only reindexed onto `scores.index` afterward — reindexing first would silently turn a "`lookback`-bar" volatility window into `lookback` *score-date* observations (e.g. a nominal 20-bar window quietly becoming ~20 months of history when scores are submitted monthly against daily returns). Dates without `lookback` observations of trailing volatility yet get zero weight for that name rather than a division blowup. |
| `dollar_neutral` | `(weights)` | Post-process any weight panel so `sum(weight) == 0` per row, by subtracting each row's mean weight — then **rescaling back to that row's original `sum(\|weight\|)`**, since mean-centering alone shrinks gross exposure and would otherwise silently drift a portfolio's leverage away from what it was sized to. |

```python
from standard_quant_tools.backtest.sizing import zscore_normalized

target_weights = zscore_normalized(my_alpha_scores, gross_leverage=1.0)
result = run_portfolio_simulation(price_data, target_weights)
```

Beta-neutral, sector-neutral, risk-parity, and optimizer-generated weights
are not implemented — each needs infrastructure this repo doesn't have yet
(per-ticker beta/sector metadata, a QP solver).

### Cost Model Building Blocks (`backtest/costs.py`)

Pure functions `run_portfolio_simulation`'s cost parameters (above) are
built from — import them directly for a custom cost calculation outside the
engine.

| Function | Signature | Behavior |
|---|---|---|
| `percentage_commission` | `(notional, rate)` | `abs(notional) * rate` — today's default model. |
| `per_share_commission` | `(shares, rate_per_share, minimum=0.0)` | Flat rate per share, floored at `minimum`. `shares=0` costs `0.0`, **not** `minimum` — the floor is a per-*order* minimum, and no order exists when nothing trades. |
| `fixed_bps_spread` | `(notional, bps)` | Spread cost as a fixed number of basis points of notional. Not wired into `run_portfolio_simulation` (which uses `slippage_pct` instead) — available for standalone use. |
| `pct_of_range_spread` | `(notional, high, low, close, pct)` | Spread cost as a fraction of the bar's own `(High - Low)` range, scaled to notional via `Close`. Raises `ValidationError` if `close <= 0`. Not wired into the engine. |
| `sqrt_impact_bps` | `(participation, volatility, coefficient=1.0)` | Square-root impact model, returned in **basis points**: `coefficient * volatility * sqrt(participation) * 10_000` (`impact_cost` divides the 1e4 back out). Raises `ValidationError` if `participation < 0`. |
| `impact_cost` | `(notional, avg_dollar_volume, volatility, coefficient=1.0)` | Dollar impact cost combining `sqrt_impact_bps` with trade notional; returns `0.0` if `avg_dollar_volume <= 0` (no volume baseline). |
| `short_borrow_cost` | `(notional, annual_bps, days=1.0)` | Daily-accrued borrow fee on short notional. |
| `margin_interest` | `(cash, annual_rate, days=1.0)` | Daily-accrued interest on negative cash; `0.0` when `cash >= 0`. |

```python
from standard_quant_tools.backtest.costs import per_share_commission

cost = per_share_commission(shares=500, rate_per_share=0.005, minimum=1.0)
```

### Liquidity & Capacity Diagnostics (`backtest/constraints.py`)

| Function | Signature | Behavior |
|---|---|---|
| `adv_participation` | `(notional, avg_dollar_volume)` | Fraction of average dollar volume a trade's notional represents; `0.0` if `avg_dollar_volume <= 0`. What `max_adv_participation` above checks under the hood. |
| `days_to_liquidate` | `(shares, avg_daily_volume, max_participation)` | Estimated trading days to unwind a position without exceeding `max_participation` of average daily volume. Raises `ValidationError` if `avg_daily_volume <= 0` or `max_participation <= 0`. |
| `sector_exposure` | `(weights, sectors)` | Aggregate portfolio weight by sector; tickers missing from `sectors` are bucketed into `"Unknown"` rather than dropped. |
| `capacity_report` | `(tickers, avg_dollar_volumes, target_weights, max_participation)` | Per-ticker max account size deployable at `max_participation` of its own ADV, given its target weight. Returns `per_ticker`, `binding_ticker` (tightest constraint, `None` if every weight is 0), `max_account_size`. Raises `ValidationError` on missing tickers or `max_participation <= 0`. |

```python
from standard_quant_tools.backtest.constraints import capacity_report

report = capacity_report(
    tickers=["AAPL", "MSFT"],
    avg_dollar_volumes={"AAPL": 8e9, "MSFT": 6e9},
    target_weights={"AAPL": 0.4, "MSFT": 0.3},
    max_participation=0.1,
)
print(report["max_account_size"], report["binding_ticker"])
```

See also `get_capacity_report`
([09_advanced_agent_tools.md §18](09_advanced_agent_tools.md#18-capacity-report))
for the LLM/JSON tool-calling form of `capacity_report`.

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
pair trade as a **2-asset portfolio with a share-ratio-hedged, price-scaled
weight vector** (only dollar-neutral when `|hedge_ratio| * Close_b ≈
Close_a`, not in general): both legs are columns of the same
`target_weights` row passed to `run_portfolio_simulation`, so they can
never fall out of sync — no new execution engine, just a different way to
build the weight panel.

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
once the z-score reverts inside `exit_z`. `hedge_ratio` is a **share**
ratio (`spread = Close_a - hedge_ratio * Close_b` — 1 share of `symbol_a`
hedged by `hedge_ratio` shares of `symbol_b`), not a dollar-weight ratio,
so converting it to dollar weights needs the price the trade will
**actually execute at** — Close on the trigger date itself under
`fill_price="close"`, but the **following bar's Open** under the default
`"next_open"` (sizing off the trigger date's Close and then executing a
bar later silently breaks the share ratio unless both legs happen to gap
overnight by the same percentage — recomputed at every transition since
the two legs' prices drift apart over time either way):

```text
denom     = exec_price_a + |hedge_ratio| * exec_price_b
weight_a  = gross_leverage * exec_price_a / denom
weight_b  = sign(hedge_ratio) * gross_leverage * |hedge_ratio| * exec_price_b / denom
```

— together `|weight_a| + |weight_b|` sum to `gross_leverage` at every
entry, and `shares_b / shares_a == hedge_ratio` by construction. The
resulting `max(|weight_a|, |weight_b|)` across all transition dates is
passed to `run_portfolio_simulation` as its `max_position_pct` (instead of
the engine's own default `1.0`), so a large `|hedge_ratio|` or
`gross_leverage > 1.0` doesn't spuriously trip the position-size check on
an otherwise-valid pair trade.

**`fill_price`** defaults to `"next_open"`, not `"close"`: the z-score
signal that decides a transition is itself computed from that same bar's
Close, so executing at that same Close would be look-ahead — the trade
could not actually have been placed at the exact price its own signal was
computed from. Pass `fill_price="close"` only for explicit same-bar/
exploratory analysis, mirroring `run_strategy`'s and
`run_portfolio_simulation`'s own execution-timing convention.

**`zscore_window`** (default `30`): rolling window, in bars, for
`spread_zscore` — every signal only uses spread history available up to
that bar, so the backtest is causal/lookahead-free. Passing `None` reverts
to a full-sample static z-score computed once over the *entire* series
(including bars after the signal date), which leaks future spread
statistics into historical signals and produces an optimistically-biased
backtest; it's an explicit opt-in, intended for exploratory analysis only —
never use it to evaluate real strategy performance.

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

Trials whose `metric_col` is NaN or infinite are **excluded from the
ranking** (and from `n_trials`), with a warning — a grid row whose returns
had zero variance is the common source. This matters because `np.sort`
places NaN last, so a descending sort would otherwise put it *first* and
report NaN as the best trial, poisoning every gap. `ValidationError` is
raised if no finite values remain.

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
| `profit_factor` | float | Gross profit / gross loss. `inf` whenever gross loss is zero (no losing trades) — including the degenerate case where gross profit is *also* zero, e.g. every trade returning exactly 0.00%. Both backends agree on this; the C++ kernel previously returned `0.0` for that 0/0 case while Python returned `inf`. |
| `num_trades` | int | Number of completed round-trips |
| `avg_trade_return_pct` | float | Average trade P&L in % |
| `equity_curve` | pd.Series | Day-by-day portfolio value |
| `trade_log` | pd.DataFrame | Per-trade entry/exit details |
