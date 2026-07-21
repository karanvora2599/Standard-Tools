# Advanced Agent Tools

Twelve high-level agentic tools that compose the library's existing primitives into single, LLM-callable operations. Each collapses a multi-step reasoning workflow into one structured function call with a Pydantic output model.

> **See also:** [07_agent_tools.md](07_agent_tools.md) covers the 14 core tools (including `run_buy_and_hold` and `compare_strategies`), the full `get_agent_tools()` registry (all 26), `dispatch()` wiring, and the complete Model Summary.

---

## Tool Summary

**Advanced tools (5)**

| Tool | What it does | Key output fields |
|---|---|---|
| `run_regime_adaptive_backtest` | Classify regime via Hurst, auto-select and optimise strategy | `regime`, `selected_strategy`, `best_parameters`, `backtest` |
| `scan_pairs` | Find cointegrated pairs in a universe, ranked by half-life | `pairs[].half_life_days`, `pairs[].signal` |
| `run_walk_forward_backtest` | Optimise in-sample, validate out-of-sample across rolling windows | `avg_oos_sharpe`, `pct_windows_profitable`, `param_stability` |
| `get_portfolio_risk_attribution` | Deep risk decomposition: MCR, PCA, optional factor model | `asset_risk_contributions`, `pca_variance_explained` |
| `get_position_size` | ATR stop-loss sizing with optional Kelly criterion | `shares_fixed_risk`, `kelly_fraction`, `recommended_shares` |

**Supplementary tools (5)**

| Tool | What it does | Key output fields |
|---|---|---|
| `get_stock_fundamentals` | Fetch company metadata and key financial ratios | `trailing_pe`, `price_to_book`, `debt_to_equity`, `return_on_equity`, `market_cap` |
| `run_backtest_optimization` | Exhaustive parameter grid search, return top N ranked by metric | `top_results[].parameters`, `top_results[].sharpe_ratio` |
| `get_advanced_indicators` | Parabolic SAR trend, Wilder ATR volatility, MFI volume signal | `sar_trend`, `wilder_atr_pct`, `mfi_signal` |
| `get_rolling_beta` | Rolling OLS beta to detect drift vs a benchmark | `current_beta`, `beta_trend`, `beta_6m_ago` |
| `get_extended_risk_metrics` | Calmar, Treynor, parametric VaR 95/99, historical VaR 99, CVaR 99 | `calmar_ratio`, `treynor_ratio`, `var_parametric_95` |

**Custom signal tools (2)**

| Tool | What it does | Key output fields |
|---|---|---|
| `run_custom_signal_backtest` | Backtest a signal computed outside this library on one symbol | Same shape as `BacktestResult` — `sharpe_ratio`, `total_return`, `num_trades`, ... |
| `run_signal_panel_backtest` | Backtest a pre-computed signal panel across a ticker universe, combined into portfolio metrics | `per_ticker[ticker]`, `portfolio_metrics.sharpe_ratio` |

---

## 1. Regime-Adaptive Strategy Selector

`run_regime_adaptive_backtest` computes the Hurst exponent on the symbol's return series, maps the result to the most appropriate strategy class, optimises parameters via grid search, and returns the best backtest alongside the regime classification — all in one call.

> **Performance:** The dominant cost in this tool is the `hurst_exponent` call on the full return series. With the optional C++ extension (`_sqt_core`) built, this step runs 20–80× faster, reducing total wall-clock time for the tool from ~10–20 s to ~0.5–2 s on a 2 000-bar series. See [Development/build_guide.md](../Development/build_guide.md).

**Regime → Strategy mapping:**

| Hurst | Regime | Selected strategy |
|---|---|---|
| > 0.55 | trending | `sma_crossover` |
| 0.45–0.55 | random_walk | `macd_crossover` |
| < 0.45 | mean_reverting | `rsi_mean_reversion` |
| NaN (< 40 returns) | unknown | `macd_crossover` (safe default) |

> **"unknown" regime:** When the return series is too short for a reliable Hurst estimate (fewer than ~40 observations), `hurst_exponent` returns `"unknown"`. The tool does **not** raise in this case — it logs the regime as `"unknown"` and falls back to `macd_crossover`, the most neutral strategy. Check `result.regime` before trusting the strategy selection for short date ranges.

```python
from standard_quant_tools.agent.tools import run_regime_adaptive_backtest
from standard_quant_tools.agent.models import RegimeAdaptiveInput

result = run_regime_adaptive_backtest(RegimeAdaptiveInput(
    symbol="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    initial_capital=50_000,
))

print(f"Regime            : {result.regime}")
print(f"Hurst             : {result.hurst:.3f}  (fit R² = {result.fit_r_squared:.3f})")
print(f"Selected strategy : {result.selected_strategy}")
print(f"Best params       : {result.best_parameters}")
print(f"Grid combos tested: {result.grid_combinations}")
print(f"Sharpe (best OIS) : {result.backtest.sharpe_ratio:.2f}")
print(f"Max drawdown      : {result.backtest.max_drawdown:.1%}")
```

**Custom parameter grids:** Pass any of `sma_param_grid`, `rsi_param_grid`, `macd_param_grid`, or `bollinger_param_grid` to override the defaults.

```python
result = run_regime_adaptive_backtest(RegimeAdaptiveInput(
    symbol="NVDA",
    start_date="2022-01-01",
    end_date="2024-01-01",
    sma_param_grid={"fast_period": [10, 20], "slow_period": [50, 100, 200]},
    rsi_param_grid={"period": [10, 14], "oversold": [25, 30], "overbought": [70, 75]},
))
```

**Default parameter grids:**

| Strategy | Grid |
|---|---|
| `sma_crossover` | fast=[5,10,20], slow=[30,50,100] → 9 combos |
| `rsi_mean_reversion` | period=[7,14,21], oversold=[25,30], overbought=[65,70] → 12 combos |
| `macd_crossover` | fast=[8,12], slow=[21,26], signal=[7,9] → 8 combos |
| `bollinger_reversion` | period=[15,20,25], num_std=[1.5,2.0] → 6 combos |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `regime` | `str` | `"trending"`, `"random_walk"`, `"mean_reverting"`, or `"unknown"` |
| `hurst` | `float` | Hurst exponent H (NaN when data is insufficient for a reliable estimate) |
| `fit_r_squared` | `float` | Quality of the Hurst log-log scaling fit |
| `selected_strategy` | `str` | Strategy name chosen for this regime |
| `best_parameters` | `dict` | Best parameter set from grid search |
| `grid_combinations` | `int` | Number of parameter combinations tested |
| `backtest` | `BacktestResult` | Full backtest result for the best parameters |

**Multi-symbol regime scan — routing a universe to the right strategy:**

```python
from standard_quant_tools.agent.tools import run_regime_adaptive_backtest
from standard_quant_tools.agent.models import RegimeAdaptiveInput

symbols = ["AAPL", "TSLA", "GLD", "TLT", "XOM", "JPM"]
start, end = "2022-01-01", "2024-01-01"

scan_results = []
for sym in symbols:
    r = run_regime_adaptive_backtest(RegimeAdaptiveInput(
        symbol=sym, start_date=start, end_date=end,
        initial_capital=50_000,
    ))
    scan_results.append({
        "symbol":   sym,
        "hurst":    r.hurst,
        "regime":   r.regime,
        "strategy": r.selected_strategy,
        "params":   r.best_parameters,
        "sharpe":   r.backtest.sharpe_ratio,
        "mdd":      r.backtest.max_drawdown,
    })

print(f"{'Symbol':<8} {'H':>6}  {'Regime':<16} {'Strategy':<22} {'Sharpe':>7} {'MDD':>7}")
print("-" * 75)
for row in sorted(scan_results, key=lambda x: -x["sharpe"]):
    print(f"{row['symbol']:<8} {row['hurst']:>6.3f}  {row['regime']:<16} "
          f"{row['strategy']:<22} {row['sharpe']:>7.2f} {row['mdd']:>7.1%}")
```

**Custom parameter grid — fine-tuning the search space:**

```python
from standard_quant_tools.agent.tools import run_regime_adaptive_backtest
from standard_quant_tools.agent.models import RegimeAdaptiveInput

# Widen the SMA grid to include 200-day MA, tighten RSI thresholds
result = run_regime_adaptive_backtest(RegimeAdaptiveInput(
    symbol="NVDA",
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_capital=100_000,
    sma_param_grid={
        "fast_period": [10, 20, 50],
        "slow_period": [100, 150, 200],
    },
    rsi_param_grid={
        "period":     [7, 10, 14],
        "oversold":   [20, 25, 30],
        "overbought": [70, 75, 80],
    },
    macd_param_grid={
        "fast_period":   [8, 12],
        "slow_period":   [21, 26],
        "signal_period": [7, 9],
    },
))

print(f"Regime: {result.regime}  (H={result.hurst:.3f})")
print(f"Best strategy: {result.selected_strategy}")
print(f"Best params  : {result.best_parameters}")
print(f"Grid tested  : {result.grid_combinations} combinations")
print(f"Sharpe       : {result.backtest.sharpe_ratio:.2f}")
print(f"Max drawdown : {result.backtest.max_drawdown:.1%}")
print(f"Win rate     : {result.backtest.win_rate:.1%}")
```

**Comparing regime-adaptive vs a fixed strategy:**

```python
from standard_quant_tools.agent.tools import run_regime_adaptive_backtest, run_sma_backtest
from standard_quant_tools.agent.models import RegimeAdaptiveInput, BacktestInput

sym = "SPY"
start, end = "2018-01-01", "2024-01-01"

# Fixed-strategy baseline: always SMA 10/50
fixed = run_sma_backtest(BacktestInput(
    symbol=sym, start_date=start, end_date=end,
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 50},
    initial_capital=100_000,
))

# Regime-adaptive (auto-selects strategy + params)
adaptive = run_regime_adaptive_backtest(RegimeAdaptiveInput(
    symbol=sym, start_date=start, end_date=end,
    initial_capital=100_000,
))

print(f"Strategy          {'Fixed SMA':>12}  {'Adaptive':>12}")
print(f"Regime / selected {'SMA 10/50':>12}  {adaptive.selected_strategy:>12}")
print(f"Sharpe ratio      {fixed.sharpe_ratio:>12.2f}  {adaptive.backtest.sharpe_ratio:>12.2f}")
print(f"Total return      {fixed.total_return_pct:>11.1f}%  {adaptive.backtest.total_return_pct:>11.1f}%")
print(f"Max drawdown      {fixed.max_drawdown:>11.1%}  {adaptive.backtest.max_drawdown:>11.1%}")
print(f"Trades            {fixed.num_trades:>12}  {adaptive.backtest.num_trades:>12}")

improvement = adaptive.backtest.sharpe_ratio - fixed.sharpe_ratio
if improvement > 0:
    print(f"\nAdaptive outperforms by {improvement:.2f} Sharpe points")
else:
    print(f"\nFixed strategy wins by {-improvement:.2f} Sharpe points (regime may be stable)")
```

---

## 2. Cointegration Pair Scanner

`scan_pairs` tests all O(n²/2) ticker combinations for cointegration, filters by p-value and half-life bounds, and returns the top N pairs sorted by half-life (shortest first = fastest mean reversion = most tradeable). Each ticker's prices are fetched **once** before testing begins.

> **Performance:** Each pair test calls `cointegration_test`, which uses the C++ extension (`_sqt_core`) when available — **5–15× faster** than the statsmodels fallback. For a universe of 10 tickers (45 pairs), the C++ path reduces total scan time from ~15–20 s to ~1–3 s.

```python
from standard_quant_tools.agent.tools import scan_pairs
from standard_quant_tools.agent.models import PairScannerInput

result = scan_pairs(PairScannerInput(
    tickers=["KO", "PEP", "MCD", "YUM", "SBUX", "WEN"],
    start_date="2021-01-01",
    end_date="2024-01-01",
    max_pairs=5,
    min_half_life=5.0,    # at least 5 bars — faster is too noisy
    max_half_life=126.0,  # no more than ~6 months
    p_value_threshold=0.05,
    zscore_window=30,
))

print(f"Pairs tested      : {result.n_pairs_tested}")
print(f"Cointegrated      : {result.n_pairs_cointegrated}")
print(f"Returned          : {result.n_pairs_returned}")

for pair in result.pairs:
    print(f"{pair.symbol_a}/{pair.symbol_b}  "
          f"p={pair.p_value:.3f}  hl={pair.half_life_days:.1f}d  "
          f"z={pair.current_zscore:+.2f}  signal={pair.signal}")
```

**Signal values:** `"long_a_short_b"` (z < −2), `"short_a_long_b"` (z > +2), `"neutral"` (|z| < 2).

**Half-life guidance:**

| Half-life | Tradability |
|---|---|
| < 5 bars | Too fast — slippage and transaction costs dominate |
| 5–30 bars | Sweet spot for daily-bar mean reversion |
| 30–126 bars | Slower — needs patient sizing and looser entry thresholds |
| > 126 bars | Marginal — consider longer data or different pairs |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `n_pairs_tested` | `int` | Total combinations evaluated |
| `n_pairs_cointegrated` | `int` | Pairs passing p-value and half-life filters |
| `n_pairs_returned` | `int` | Pairs in the result (capped at `max_pairs`) |
| `pairs` | `List[PairResult]` | Sorted by `half_life_days` ascending |

Each `PairResult` contains: `symbol_a`, `symbol_b`, `p_value`, `hedge_ratio`, `half_life_days`, `adf_statistic`, `current_zscore`, `signal`.

**Full workflow — scan, select top pair, and get position size:**

```python
from standard_quant_tools.agent.tools import scan_pairs, get_position_size
from standard_quant_tools.agent.models import PairScannerInput, PositionSizerInput

# Step 1: scan the consumer staples sector for cointegrated pairs
scan = scan_pairs(PairScannerInput(
    tickers=["KO", "PEP", "MCD", "YUM", "SBUX", "WEN", "DPZ", "QSR"],
    start_date="2021-01-01",
    end_date="2024-01-01",
    max_pairs=10,
    min_half_life=5.0,
    max_half_life=60.0,
    p_value_threshold=0.05,
    zscore_window=30,
))

print(f"Tested {scan.n_pairs_tested} pairs  →  {scan.n_pairs_cointegrated} cointegrated  →  {scan.n_pairs_returned} returned")
print()
for p in scan.pairs:
    print(f"{p.symbol_a}/{p.symbol_b:<6}  p={p.p_value:.3f}  hl={p.half_life_days:.1f}d  "
          f"hedge={p.hedge_ratio:.3f}  z={p.current_zscore:+.2f}  {p.signal}")

# Step 2: trade the top pair if there's an active signal
if scan.pairs:
    top = scan.pairs[0]   # shortest half-life = fastest mean reversion
    if top.signal != "neutral":
        print(f"\nActive signal on {top.symbol_a}/{top.symbol_b}: {top.signal}")

        # Step 3: size the long leg
        long_sym = top.symbol_a if "long_a" in top.signal else top.symbol_b
        pos = get_position_size(PositionSizerInput(
            symbol=long_sym,
            start_date="2023-06-01",
            end_date="2024-01-01",
            account_equity=200_000,
            risk_per_trade_pct=0.005,   # 0.5% per leg — pairs trade has two legs
            atr_period=14,
            atr_multiplier=2.0,
        ))
        print(f"Long leg ({long_sym}): {pos.recommended_shares} shares  "
              f"(${pos.recommended_position_value:,.0f}  stop=${pos.stop_distance:.2f})")
    else:
        print(f"\n{top.symbol_a}/{top.symbol_b} cointegrated but z-score neutral — waiting for entry")
```

**Sector-focused scan — why sector pairs are more robust:**

```python
from standard_quant_tools.agent.tools import scan_pairs
from standard_quant_tools.agent.models import PairScannerInput

# Within-sector pairs share macro drivers, reducing the chance of spurious cointegration.
# Cross-sector pairs often fail out-of-sample because the shared factor disappears.

sectors = {
    "Energy":    ["XOM", "CVX", "COP", "EOG", "SLB", "OXY"],
    "Banks":     ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Tech":      ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
}

all_pairs = []
for sector, tickers in sectors.items():
    result = scan_pairs(PairScannerInput(
        tickers=tickers,
        start_date="2021-01-01",
        end_date="2024-01-01",
        max_pairs=3,
        min_half_life=5.0,
        max_half_life=90.0,
    ))
    for pair in result.pairs:
        all_pairs.append((sector, pair))

# Rank all sector pairs by half-life
all_pairs.sort(key=lambda x: x[1].half_life_days)

print(f"{'Sector':<10} {'Pair':<14} {'p-val':>6}  {'HL':>6}  {'z-score':>8}  {'Signal'}")
print("-" * 65)
for sector, pair in all_pairs[:10]:
    print(f"{sector:<10} {pair.symbol_a}/{pair.symbol_b:<8} "
          f"{pair.p_value:>6.3f}  {pair.half_life_days:>6.1f}d  "
          f"{pair.current_zscore:>8.2f}  {pair.signal}")
```

**Monitoring active pairs — rolling cointegration stability check:**

```python
from standard_quant_tools.agent.tools import scan_pairs
from standard_quant_tools.agent.models import PairScannerInput

# Re-run the scan monthly to detect when a pair breaks down (p-value rises above 0.10)
import datetime

today = datetime.date.today()
windows = [
    (today - datetime.timedelta(days=d+365), today - datetime.timedelta(days=d))
    for d in [0, 90, 180, 270]    # 4 rolling 1-year windows
]

pair_sym_a, pair_sym_b = "KO", "PEP"

print(f"Cointegration stability: {pair_sym_a}/{pair_sym_b}")
print(f"{'Window end':<14} {'p-value':>8}  {'Half-life':>10}  {'Status'}")
print("-" * 50)
for win_start, win_end in windows:
    r = scan_pairs(PairScannerInput(
        tickers=[pair_sym_a, pair_sym_b],
        start_date=str(win_start),
        end_date=str(win_end),
        p_value_threshold=1.0,   # include all so we can track the p-value trend
    ))
    if r.pairs:
        p = r.pairs[0]
        status = "OK" if p.p_value < 0.05 else ("WEAKENING" if p.p_value < 0.10 else "BROKEN")
        print(f"{str(win_end):<14} {p.p_value:>8.3f}  {p.half_life_days:>10.1f}d  {status}")
    else:
        print(f"{str(win_end):<14} {'—':>8}  {'—':>10}  NO PAIRS")
```

---

## 3. Walk-Forward Backtest

`run_walk_forward_backtest` is the gold standard for strategy validation. It repeatedly:
1. Runs `backtest_grid` on a training window to find the best parameters.
2. Evaluates those parameters on the next (unseen) out-of-sample window.
3. Slides forward by `test_bars` and repeats.

The OOS windows are non-overlapping, so each bar is tested exactly once. This gives an unbiased estimate of strategy performance.

```python
from standard_quant_tools.agent.tools import run_walk_forward_backtest
from standard_quant_tools.agent.models import WalkForwardInput

result = run_walk_forward_backtest(WalkForwardInput(
    symbol="SPY",
    start_date="2016-01-01",
    end_date="2024-01-01",
    strategy="sma_crossover",
    param_grid={
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100],
    },
    train_bars=252,   # ~1 year in-sample
    test_bars=63,     # ~1 quarter OOS
    sort_by="sharpe_ratio",
))

print(f"Windows           : {result.n_windows}")
print(f"Avg OOS Sharpe    : {result.avg_oos_sharpe:.2f}")
print(f"Avg OOS Return    : {result.avg_oos_return:.1%}")
print(f"Avg OOS Max DD    : {result.avg_oos_max_drawdown:.1%}")
print(f"% Windows +ve     : {result.pct_windows_profitable:.0%}")

# Parameter stability: how consistently does the same param win in-sample?
for param, info in result.param_stability.items():
    print(f"{param}: most common={info['most_common']}  frequency={info['frequency']:.0%}")

# Per-window breakdown
for win in result.windows:
    print(f"[{win.test_start} → {win.test_end}]  "
          f"IS Sharpe={win.in_sample_sharpe:.2f}  "
          f"OOS Sharpe={win.out_of_sample_sharpe:.2f}  "
          f"OOS Return={win.out_of_sample_return:.1%}  "
          f"params={win.best_params}")
```

**Interpreting results:**

| Metric | Good | Concerning |
|---|---|---|
| `avg_oos_sharpe` | > 0.5 | < 0 |
| `pct_windows_profitable` | > 60% | < 40% |
| `param_stability[x].frequency` | > 60% (stable) | < 30% (unstable/overfitting) |

> **Minimum data requirement:** `train_bars + test_bars` bars. For 252 + 63 daily bars, you need ≈315 trading days (~15 months) minimum; more windows are produced with longer history.

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `n_windows` | `int` | Number of walk-forward windows |
| `windows` | `List[WalkForwardWindow]` | Per-window detail |
| `avg_oos_sharpe` | `float` | Mean OOS Sharpe across all windows |
| `avg_oos_return` | `float` | Mean OOS total return per window |
| `avg_oos_max_drawdown` | `float` | Mean OOS max drawdown per window |
| `pct_windows_profitable` | `float` | Fraction of OOS windows with positive return |
| `param_stability` | `dict` | Most common winning parameter per key + frequency |

**Detecting overfitting — is-sample Sharpe that vanishes out-of-sample:**

```python
from standard_quant_tools.agent.tools import run_walk_forward_backtest
from standard_quant_tools.agent.models import WalkForwardInput

result = run_walk_forward_backtest(WalkForwardInput(
    symbol="TSLA",
    start_date="2018-01-01",
    end_date="2024-01-01",
    strategy="sma_crossover",
    param_grid={"fast_period": [3, 5, 10, 20], "slow_period": [15, 30, 50, 100]},
    train_bars=252,
    test_bars=63,
))

print(f"Windows: {result.n_windows}   Avg OOS Sharpe: {result.avg_oos_sharpe:.2f}")
print()
print(f"{'Window':<28} {'IS Sharpe':>10}  {'OOS Sharpe':>11}  {'Decay':>8}")
print("-" * 65)
for i, win in enumerate(result.windows):
    decay = win.in_sample_sharpe - win.out_of_sample_sharpe
    flag  = " ← overfit" if decay > 1.0 else ""
    print(f"[{win.test_start} → {win.test_end}]  {win.in_sample_sharpe:>10.2f}  "
          f"{win.out_of_sample_sharpe:>11.2f}  {decay:>7.2f}{flag}")

# Summary diagnosis
avg_is  = sum(w.in_sample_sharpe  for w in result.windows) / result.n_windows
avg_oos = result.avg_oos_sharpe
print()
if avg_is - avg_oos > 0.8:
    print(f"⚠  Large IS→OOS Sharpe decay ({avg_is:.2f} → {avg_oos:.2f}): likely overfitting")
    print("   Try a coarser parameter grid or longer training window")
elif avg_oos < 0:
    print(f"⚠  Negative avg OOS Sharpe ({avg_oos:.2f}): strategy has no real edge")
else:
    print(f"✓ Moderate IS→OOS decay ({avg_is:.2f} → {avg_oos:.2f}): strategy appears robust")
```

**Parameter stability — catching strategies that only win by luck:**

```python
# Parameter stability tells you whether a strategy requires a specific setting or is robust
result = run_walk_forward_backtest(WalkForwardInput(
    symbol="SPY",
    start_date="2016-01-01",
    end_date="2024-01-01",
    strategy="rsi_mean_reversion",
    param_grid={
        "period":     [7, 10, 14, 21],
        "oversold":   [25, 30, 35],
        "overbought": [65, 70, 75],
    },
    train_bars=252, test_bars=63,
))

print("Parameter stability (how often the same setting wins in-sample):\n")
for param, info in result.param_stability.items():
    freq = info["frequency"]
    val  = info["most_common"]
    bar  = "█" * int(freq * 20)
    status = "STABLE" if freq > 0.6 else ("MODERATE" if freq > 0.4 else "UNSTABLE")
    print(f"  {param:<16}: {val!s:<6} wins {freq:.0%} of windows  {bar}  [{status}]")

# Unstable parameters are red flags: the strategy needs a specific lucky setting
# to look good in-sample — this setting will likely not persist OOS.
```

**Two-strategy comparison — which survives out-of-sample?**

```python
from standard_quant_tools.agent.tools import run_walk_forward_backtest
from standard_quant_tools.agent.models import WalkForwardInput

sym, start, end = "AAPL", "2016-01-01", "2024-01-01"

sma_wf = run_walk_forward_backtest(WalkForwardInput(
    symbol=sym, start_date=start, end_date=end,
    strategy="sma_crossover",
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    train_bars=252, test_bars=63,
))

rsi_wf = run_walk_forward_backtest(WalkForwardInput(
    symbol=sym, start_date=start, end_date=end,
    strategy="rsi_mean_reversion",
    param_grid={"period": [7, 14, 21], "oversold": [25, 30], "overbought": [70, 75]},
    train_bars=252, test_bars=63,
))

print(f"{'Metric':<28} {'SMA Crossover':>15}  {'RSI Mean-Rev':>14}")
print("-" * 62)
print(f"{'Avg OOS Sharpe':<28} {sma_wf.avg_oos_sharpe:>15.2f}  {rsi_wf.avg_oos_sharpe:>14.2f}")
print(f"{'Avg OOS Return':<28} {sma_wf.avg_oos_return:>15.1%}  {rsi_wf.avg_oos_return:>14.1%}")
print(f"{'Avg OOS Max DD':<28} {sma_wf.avg_oos_max_drawdown:>15.1%}  {rsi_wf.avg_oos_max_drawdown:>14.1%}")
print(f"{'% Windows Profitable':<28} {sma_wf.pct_windows_profitable:>15.0%}  {rsi_wf.pct_windows_profitable:>14.0%}")

winner = "SMA Crossover" if sma_wf.avg_oos_sharpe > rsi_wf.avg_oos_sharpe else "RSI Mean-Rev"
print(f"\nRecommended strategy: {winner}")
```

---

## 4. Portfolio Risk Attribution

`get_portfolio_risk_attribution` goes beyond basic portfolio metrics to provide a complete risk decomposition:
- **Marginal Risk Contribution (MCR):** how much each asset contributes to total portfolio volatility (values sum to 1.0).
- **PCA variance decomposition:** what fraction of the asset universe's variance is captured by each latent factor, and how much the portfolio loads onto each factor.
- **Factor regression** (optional): multi-factor OLS on the aggregate portfolio returns.

```python
from standard_quant_tools.agent.tools import get_portfolio_risk_attribution
from standard_quant_tools.agent.models import RiskAttributionInput

result = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM", "GLD"],
    weights=[0.25, 0.20, 0.20, 0.20, 0.15],
    start_date="2021-01-01",
    end_date="2024-01-01",
    benchmark="SPY",
    n_components=3,
    factor_tickers=["SPY", "TLT", "GLD"],   # optional
    factor_names=["equity", "bonds", "gold"],
))

# Portfolio-level metrics
print(f"Annualised return : {result.annualized_return:.1%}")
print(f"Annualised vol    : {result.annualized_volatility:.1%}")
print(f"Sharpe ratio      : {result.sharpe_ratio:.2f}")
print(f"Max drawdown      : {result.max_drawdown:.1%}")
print(f"VaR (95%)         : {result.var_95:.4f}")
print(f"CVaR (95%)        : {result.cvar_95:.4f}")

# Which assets drive portfolio risk?
print("\nMarginal Risk Contributions (sum = 1.0):")
for ticker, mcr in sorted(result.asset_risk_contributions.items(),
                           key=lambda x: -x[1]):
    print(f"  {ticker}: {mcr:.1%}")

# How much of portfolio variance comes from each latent factor?
print("\nPCA variance explained:")
for pc, evr in result.pca_variance_explained.items():
    exp = result.portfolio_pc_exposures[pc]
    print(f"  {pc}: EVR={evr:.1%}  portfolio loading={exp:+.3f}")

# Factor model (if factor_tickers was provided)
if result.factor_loadings:
    print(f"\nFactor alpha (daily): {result.factor_alpha:.6f}")
    print(f"Factor R²           : {result.factor_r_squared:.3f}")
    for factor, loading in result.factor_loadings.items():
        print(f"  {factor}: {loading:.3f}")
```

**Marginal Risk Contribution (MCR) formula:**

For portfolio weights **w** and covariance matrix **Σ** (annualised):

```
MCR_i = (Σ w)_i × w_i / (w' Σ w)
```

MCR values sum to exactly 1.0 and represent each asset's fractional contribution to total portfolio variance. A high MCR for a single asset indicates concentration risk.

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `asset_risk_contributions` | `dict[str, float]` | Fractional MCR per asset (sums to 1.0) |
| `pca_variance_explained` | `dict[str, float]` | EVR per PC across the asset universe |
| `portfolio_pc_exposures` | `dict[str, float]` | Portfolio's weighted loading on each PC |
| `factor_loadings` | `dict[str, float]` | Factor OLS loadings (optional) |
| `factor_r_squared` | `float` | Factor model R² (optional) |
| `factor_alpha` | `float` | Factor model alpha, daily (optional) |

**Identifying and fixing concentration risk:**

```python
from standard_quant_tools.agent.tools import get_portfolio_risk_attribution
from standard_quant_tools.agent.models import RiskAttributionInput

# Current portfolio — appears diversified by ticker count
result = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],
    start_date="2022-01-01",
    end_date="2024-01-01",
    n_components=3,
))

print("MCR per asset (contribution to total portfolio risk):")
for ticker, mcr in sorted(result.asset_risk_contributions.items(), key=lambda x: -x[1]):
    bar = "█" * int(mcr * 50)
    flag = " ← dominant" if mcr > 0.30 else ""
    print(f"  {ticker:<8} {mcr:>6.1%}  {bar}{flag}")

print(f"\nPC1 explains {result.pca_variance_explained.get('PC1', 0):.0%} of all variance")
if result.pca_variance_explained.get("PC1", 0) > 0.70:
    print("⚠  Single latent factor dominates — despite 5 names, this is a ~1-factor portfolio")
    print("   Consider adding GLD, TLT, XLE, or VNQ to introduce uncorrelated factor exposure")

# Fix: add uncorrelated assets
result_fixed = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=["AAPL", "MSFT", "GOOGL", "GLD", "TLT"],
    weights=[0.25, 0.20, 0.20, 0.20, 0.15],
    start_date="2022-01-01",
    end_date="2024-01-01",
    n_components=3,
))

print("\nAfter adding GLD + TLT:")
print(f"  PC1 explains: {result_fixed.pca_variance_explained.get('PC1', 0):.0%}  "
      f"(was {result.pca_variance_explained.get('PC1', 0):.0%})")
print(f"  Max MCR     : {max(result_fixed.asset_risk_contributions.values()):.1%}  "
      f"(was {max(result.asset_risk_contributions.values()):.1%})")
```

**Risk-parity insight — what weights would equalise MCR?**

```python
from standard_quant_tools.agent.tools import get_portfolio_risk_attribution
from standard_quant_tools.agent.models import RiskAttributionInput

# Start with equal weights, then iteratively reduce weights for high-MCR assets
tickers = ["AAPL", "MSFT", "TLT", "GLD", "XOM"]
weights = [0.20, 0.20, 0.20, 0.20, 0.20]   # equal weight baseline

result = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=tickers, weights=weights,
    start_date="2021-01-01", end_date="2024-01-01",
))

print("Equal-weight baseline MCR:")
for t, mcr in result.asset_risk_contributions.items():
    print(f"  {t}: {mcr:.1%}")

# Simple heuristic: target = 1/n; reduce overweighted contributors
n = len(tickers)
target_mcr = 1.0 / n
adj_weights = dict(zip(tickers, weights))
mcr_map = result.asset_risk_contributions

for t in tickers:
    ratio = target_mcr / mcr_map[t]
    adj_weights[t] = weights[tickers.index(t)] * ratio   # scale toward parity

# Re-normalise
total = sum(adj_weights.values())
adj_weights = {t: w / total for t, w in adj_weights.items()}

print("\nHeuristic risk-parity weights:")
for t, w in adj_weights.items():
    print(f"  {t}: {w:.1%}")

# Verify with a second call
result2 = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=tickers, weights=list(adj_weights.values()),
    start_date="2021-01-01", end_date="2024-01-01",
))
print("\nMCR after adjustment:")
for t, mcr in result2.asset_risk_contributions.items():
    print(f"  {t}: {mcr:.1%}")
```

**Before/after comparison — adding an uncorrelated asset:**

```python
from standard_quant_tools.agent.tools import get_portfolio_risk_attribution
from standard_quant_tools.agent.models import RiskAttributionInput

# Portfolio BEFORE adding diversifier
before = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM"],
    weights=[0.30, 0.30, 0.25, 0.15],
    start_date="2020-01-01", end_date="2024-01-01",
    n_components=3,
    factor_tickers=["SPY", "TLT"],
    factor_names=["equity", "bonds"],
))

# Portfolio AFTER adding GLD (gold) as a diversifier
after = get_portfolio_risk_attribution(RiskAttributionInput(
    tickers=["AAPL", "MSFT", "GOOGL", "JPM", "GLD"],
    weights=[0.25, 0.25, 0.20, 0.12, 0.18],
    start_date="2020-01-01", end_date="2024-01-01",
    n_components=3,
    factor_tickers=["SPY", "TLT"],
    factor_names=["equity", "bonds"],
))

print(f"{'Metric':<30} {'Before':>10}  {'After':>10}  {'Delta':>8}")
print("-" * 65)
metrics = [
    ("Annualised return",  before.annualized_return,    after.annualized_return,    True),
    ("Annualised vol",     before.annualized_volatility, after.annualized_volatility, False),
    ("Sharpe ratio",       before.sharpe_ratio,          after.sharpe_ratio,          True),
    ("Max drawdown",       before.max_drawdown,          after.max_drawdown,          False),
    ("VaR (95%)",          before.var_95,                after.var_95,                False),
]
for name, b_val, a_val, higher_is_better in metrics:
    delta = a_val - b_val
    sign  = "+" if delta > 0 else ""
    arrow = "▲" if (delta > 0) == higher_is_better else "▼"
    if name in ("Sharpe ratio",):
        print(f"{name:<30} {b_val:>10.2f}  {a_val:>10.2f}  {sign}{delta:>6.2f} {arrow}")
    else:
        print(f"{name:<30} {b_val:>10.1%}  {a_val:>10.1%}  {sign}{delta:>6.1%} {arrow}")

pc1_before = before.pca_variance_explained.get("PC1", 0)
pc1_after  = after.pca_variance_explained.get("PC1", 0)
print(f"\nPC1 variance explained: {pc1_before:.0%} → {pc1_after:.0%}  "
      f"({'more diversified' if pc1_after < pc1_before else 'more concentrated'})")
```

---

## 5. ATR-Based Position Sizer

`get_position_size` computes a risk-adjusted position size using an ATR-based stop-loss. Optionally applies the Kelly criterion when strategy statistics (win rate, avg win/loss) are known.

**Fixed-risk sizing formula:**

```
dollar_risk    = account_equity × risk_per_trade_pct
stop_distance  = ATR × atr_multiplier
shares         = floor(dollar_risk / stop_distance)
```

**Kelly criterion:**

```
b = avg_win_pct / avg_loss_pct     (win/loss ratio)
f = (b × win_rate − (1 − win_rate)) / b
```

Half-Kelly is recommended (`f × 0.5`) — full Kelly is theoretically optimal but has severe drawdown consequences in practice.

```python
from standard_quant_tools.agent.tools import get_position_size
from standard_quant_tools.agent.models import PositionSizerInput

# ATR-based sizing only
result = get_position_size(PositionSizerInput(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01",
    account_equity=100_000,
    risk_per_trade_pct=0.01,   # risk 1% per trade
    atr_period=14,
    atr_multiplier=2.0,        # stop = 2 × ATR
))

print(f"Last close        : ${result.last_close:.2f}")
print(f"ATR(14)           : ${result.atr:.2f}  ({result.atr_pct:.2f}% of price)")
print(f"Stop distance     : ${result.stop_distance:.2f}")
print(f"Shares (fixed)    : {result.shares_fixed_risk}")
print(f"Position value    : ${result.position_value_fixed_risk:,.0f}")
print(f"Portfolio %       : {result.portfolio_pct_fixed_risk:.1%}")
print(f"Max loss at stop  : ${result.max_loss_fixed_risk:.0f}")
```

```python
# With Kelly inputs (from a known strategy's track record)
result = get_position_size(PositionSizerInput(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01",
    account_equity=100_000,
    risk_per_trade_pct=0.01,
    win_rate=0.55,
    avg_win_pct=0.05,    # average winning trade = +5%
    avg_loss_pct=0.025,  # average losing trade  = −2.5%
))

print(f"Kelly fraction    : {result.kelly_fraction:.3f}")
print(f"Half-Kelly shares : {result.shares_half_kelly}")
print(f"Recommended       : {result.recommended_sizing}")
print(f"Recommended shares: {result.recommended_shares}")
print(f"Recommended value : ${result.recommended_position_value:,.0f}")
```

**Recommendation logic:**
- `"fixed_risk"`: default, or when Kelly fraction ≤ 0 (negative edge).
- `"half_kelly"`: when Kelly > 0 and half-Kelly shares > 0 (positive expected value strategy).

**Sizing multiple positions across a portfolio — enforcing total exposure limits:**

```python
from standard_quant_tools.agent.tools import get_position_size
from standard_quant_tools.agent.models import PositionSizerInput

account_equity   = 250_000
max_total_risk   = 0.05     # never risk more than 5% of account across all open trades
risk_per_trade   = 0.01     # 1% per position

# Candidate symbols to enter (from a screener or pair scan)
candidates = [
    {"symbol": "AAPL", "win_rate": 0.55, "avg_win": 0.04, "avg_loss": 0.02},
    {"symbol": "TSLA", "win_rate": 0.52, "avg_win": 0.06, "avg_loss": 0.03},
    {"symbol": "GLD",  "win_rate": 0.50, "avg_win": 0.03, "avg_loss": 0.02},
    {"symbol": "XOM",  "win_rate": 0.58, "avg_win": 0.035,"avg_loss": 0.02},
]

allocations = []
total_risk_used = 0.0

for c in candidates:
    pos = get_position_size(PositionSizerInput(
        symbol=c["symbol"],
        start_date="2023-06-01",
        end_date="2024-01-01",
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade,
        win_rate=c["win_rate"],
        avg_win_pct=c["avg_win"],
        avg_loss_pct=c["avg_loss"],
        atr_period=14,
        atr_multiplier=2.0,
    ))

    position_risk_pct = pos.max_loss_fixed_risk / account_equity
    if total_risk_used + position_risk_pct > max_total_risk:
        print(f"{c['symbol']}: SKIP — would push total risk to "
              f"{total_risk_used + position_risk_pct:.1%} (limit {max_total_risk:.0%})")
        continue

    total_risk_used += position_risk_pct
    allocations.append((c["symbol"], pos))
    print(f"{c['symbol']:<8} {pos.recommended_shares:>5} shares  "
          f"${pos.recommended_position_value:>8,.0f}  "
          f"risk={position_risk_pct:.2%}  "
          f"({pos.recommended_sizing}  kelly={pos.kelly_fraction:.3f})")

print(f"\nTotal risk committed: {total_risk_used:.2%} of ${account_equity:,.0f}")
```

**Deriving Kelly inputs directly from a backtest result:**

```python
from standard_quant_tools.agent.tools import run_sma_backtest, get_position_size
from standard_quant_tools.agent.models import BacktestInput, PositionSizerInput

# Step 1: run the backtest to get win-rate and avg trade statistics
bt = run_sma_backtest(BacktestInput(
    symbol="SPY",
    start_date="2018-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters={"fast_period": 10, "slow_period": 50},
    initial_capital=100_000,
))

print(f"Backtest: Sharpe={bt.sharpe_ratio:.2f}  WinRate={bt.win_rate:.1%}  "
      f"AvgTrade={bt.avg_trade_return_pct:.2f}%  Trades={bt.num_trades}")

# Step 2: use the backtest's win-rate as Kelly inputs
# avg_win / avg_loss requires per-trade statistics; approximate from total figures
avg_win_pct  = max(abs(bt.avg_trade_return_pct) * 1.5, 0.005) / 100  # rough: winners ≈ 1.5× avg
avg_loss_pct = max(abs(bt.avg_trade_return_pct) * 0.7, 0.003) / 100  # losers  ≈ 0.7× avg

pos = get_position_size(PositionSizerInput(
    symbol="SPY",
    start_date="2023-01-01",
    end_date="2024-01-01",
    account_equity=100_000,
    risk_per_trade_pct=0.01,
    win_rate=bt.win_rate,
    avg_win_pct=avg_win_pct,
    avg_loss_pct=avg_loss_pct,
    atr_period=14,
    atr_multiplier=2.0,
))

print(f"\nKelly fraction : {pos.kelly_fraction:.3f}")
print(f"Half-Kelly     : {pos.shares_half_kelly} shares")
print(f"Fixed-risk     : {pos.shares_fixed_risk} shares")
print(f"Recommended    : {pos.recommended_shares} shares ({pos.recommended_sizing})")
```

**Volatility-regime adjustment — widening stop in high-ATR environments:**

```python
from standard_quant_tools.agent.tools import get_position_size
from standard_quant_tools.agent.models import PositionSizerInput

# In low-volatility regimes: tighten the stop (smaller ATR multiplier)
# In high-volatility regimes: widen the stop to avoid being stopped out by noise
# The fixed-risk formula automatically adjusts shares — more volatile = fewer shares

sym = "TSLA"
account = 100_000
risk_pct = 0.01

scenarios = [
    ("Low vol (tight stop)",    1.0),   # stop = 1× ATR
    ("Normal (standard stop)",  2.0),   # stop = 2× ATR  ← default
    ("High vol (wide stop)",    3.0),   # stop = 3× ATR
]

print(f"Position sizing for {sym} (account=${account:,}, risk={risk_pct:.0%}/trade)")
print()
print(f"{'Scenario':<28} {'ATR mult':>9}  {'Shares':>7}  {'Value':>10}  {'Max loss':>9}  {'Port %':>7}")
print("-" * 80)

for label, mult in scenarios:
    pos = get_position_size(PositionSizerInput(
        symbol=sym,
        start_date="2023-01-01",
        end_date="2024-01-01",
        account_equity=account,
        risk_per_trade_pct=risk_pct,
        atr_period=14,
        atr_multiplier=mult,
    ))
    print(f"{label:<28} {mult:>9.1f}  {pos.shares_fixed_risk:>7}  "
          f"${pos.position_value_fixed_risk:>9,.0f}  "
          f"${pos.max_loss_fixed_risk:>8,.0f}  "
          f"{pos.portfolio_pct_fixed_risk:>7.1%}")

# Key insight: dollar risk stays constant (always = account × risk_pct).
# The shares and position value shrink as ATR/multiplier grows.
# This is the correct response to increased volatility — exposure goes down, not up.
```

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `last_close` | `float` | Most recent closing price |
| `atr` | `float` | ATR value at last bar |
| `atr_pct` | `float` | ATR as % of price |
| `stop_distance` | `float` | atr_multiplier × ATR in dollars |
| `shares_fixed_risk` | `int` | Fixed-risk position size in shares |
| `position_value_fixed_risk` | `float` | Position dollar value (fixed-risk) |
| `portfolio_pct_fixed_risk` | `float` | Position as % of account |
| `max_loss_fixed_risk` | `float` | Max dollar loss if stop is hit |
| `kelly_fraction` | `float?` | Full Kelly fraction (0 if negative edge) |
| `shares_half_kelly` | `int?` | Half-Kelly position in shares |
| `recommended_sizing` | `str` | `"fixed_risk"` or `"half_kelly"` |
| `recommended_shares` | `int` | Final recommended position size |
| `recommended_position_value` | `float` | Final recommended dollar value |

---

## Using multiple tools together

These tools are designed to chain naturally in an agent loop:

```python
# Step 1: Classify regime and run optimised backtest
regime_result = run_regime_adaptive_backtest(RegimeAdaptiveInput(
    symbol="AAPL", start_date="2021-01-01", end_date="2024-01-01",
))

# Step 2: Use backtest statistics to size the position
pos = get_position_size(PositionSizerInput(
    symbol="AAPL",
    start_date="2023-06-01",
    end_date="2024-01-01",
    account_equity=250_000,
    win_rate=regime_result.backtest.win_rate,
    avg_win_pct=max(regime_result.backtest.avg_trade_return_pct / 100, 0.001),
    avg_loss_pct=abs(min(regime_result.backtest.avg_trade_return_pct / 100, -0.001)),
))

print(f"Regime: {regime_result.regime} → {regime_result.selected_strategy}")
print(f"Params: {regime_result.best_parameters}")
print(f"OIS Sharpe: {regime_result.backtest.sharpe_ratio:.2f}")
print(f"Recommended position: {pos.recommended_shares} shares "
      f"(${pos.recommended_position_value:,.0f}, {pos.recommended_sizing})")
```

```python
# Pairs workflow: scan → trade top pair
pairs = scan_pairs(PairScannerInput(
    tickers=["KO", "PEP", "MCD", "YUM", "SBUX"],
    start_date="2021-01-01", end_date="2024-01-01",
    max_pairs=3,
))

if pairs.pairs:
    top = pairs.pairs[0]
    print(f"Best pair: {top.symbol_a}/{top.symbol_b}  "
          f"signal={top.signal}  z={top.current_zscore:+.2f}  "
          f"half-life={top.half_life_days:.0f}d")
```

```python
# Walk-forward → position size pipeline
wf = run_walk_forward_backtest(WalkForwardInput(
    symbol="SPY",
    start_date="2018-01-01", end_date="2024-01-01",
    strategy="rsi_mean_reversion",
    param_grid={"period": [7, 14], "oversold": [25, 30], "overbought": [70, 75]},
    train_bars=252, test_bars=63,
))

if wf.avg_oos_sharpe > 0.5 and wf.pct_windows_profitable > 0.6:
    best_p = wf.windows[-1].best_params  # most recent window's params
    pos = get_position_size(PositionSizerInput(
        symbol="SPY",
        start_date="2023-06-01", end_date="2024-01-01",
        account_equity=500_000,
        win_rate=wf.pct_windows_profitable,
    ))
    print(f"Strategy validated (avg OOS Sharpe={wf.avg_oos_sharpe:.2f})")
    print(f"Position: {pos.recommended_shares} shares ({pos.recommended_sizing})")
```

---

## 6. Stock Fundamentals

`get_stock_fundamentals` fetches company metadata (sector, industry, country, employees) and key financial ratios from the data provider. Call this early in any fundamental-driven workflow to establish the valuation context before running backtests or risk analysis.

```python
from standard_quant_tools.agent.tools import get_stock_fundamentals
from standard_quant_tools.agent.models import FundamentalsInput

result = get_stock_fundamentals(FundamentalsInput(symbol="AAPL"))

print(f"Name       : {result.name}")
print(f"Sector     : {result.sector}")
print(f"Industry   : {result.industry}")
print(f"Country    : {result.country}")
print(f"Employees  : {result.full_time_employees}")
print(f"Market cap : ${result.market_cap:,.0f}" if result.market_cap else "Market cap : N/A")
print(f"PE (trail) : {result.trailing_pe}")
print(f"PE (fwd)   : {result.forward_pe}")
print(f"P/B        : {result.price_to_book}")
print(f"D/E        : {result.debt_to_equity}")
print(f"ROE        : {result.return_on_equity}")
print(f"Profit mrg : {result.profit_margins}")
print(f"Div yield  : {result.dividend_yield}")
```

**FundamentalsInput fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `symbol` | str | yes | Ticker symbol |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Ticker symbol |
| `name` | `str` | Company long name |
| `sector` | `str` | GICS sector |
| `industry` | `str` | Industry sub-group |
| `country` | `str?` | Country of domicile |
| `full_time_employees` | `int?` | Full-time headcount |
| `market_cap` | `int?` | Market capitalisation in USD |
| `trailing_pe` | `float?` | Trailing twelve-month P/E |
| `forward_pe` | `float?` | Forward P/E (consensus estimate) |
| `price_to_book` | `float?` | Price-to-book ratio |
| `debt_to_equity` | `float?` | Total debt / equity |
| `return_on_equity` | `float?` | ROE as a decimal (e.g. 0.25 = 25%) |
| `profit_margins` | `float?` | Net profit margin as decimal |
| `dividend_yield` | `float?` | Dividend yield as decimal |

All ratio fields may be `None` when the data provider does not report them (e.g. pre-revenue companies, REITs without standard PE).

**Comparing fundamentals across peers:**

```python
from standard_quant_tools.agent.tools import get_stock_fundamentals
from standard_quant_tools.agent.models import FundamentalsInput

peers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN"]
rows = [get_stock_fundamentals(FundamentalsInput(symbol=s)) for s in peers]

print(f"{'Ticker':<8} {'PE':>7} {'Fwd PE':>8} {'P/B':>6} {'D/E':>6} {'ROE':>7} {'Margin':>8}")
print("-" * 55)
for r in rows:
    pe   = f"{r.trailing_pe:.1f}"  if r.trailing_pe  else "N/A"
    fpe  = f"{r.forward_pe:.1f}" if r.forward_pe else "N/A"
    pb   = f"{r.price_to_book:.1f}"  if r.price_to_book  else "N/A"
    de   = f"{r.debt_to_equity:.1f}" if r.debt_to_equity else "N/A"
    roe  = f"{r.return_on_equity:.0%}" if r.return_on_equity else "N/A"
    mrg  = f"{r.profit_margins:.0%}"    if r.profit_margins   else "N/A"
    print(f"{r.symbol:<8} {pe:>7} {fpe:>8} {pb:>6} {de:>6} {roe:>7} {mrg:>8}")
```

---

## 7. Backtest Parameter Optimization

`run_backtest_optimization` runs an exhaustive parameter grid search for a single strategy and returns the top N combinations ranked by a chosen metric. Use this before committing to a single `run_*_backtest` call to identify which parameter settings perform best over the chosen period.

```python
from standard_quant_tools.agent.tools import run_backtest_optimization
from standard_quant_tools.agent.models import BacktestOptInput

result = run_backtest_optimization(BacktestOptInput(
    symbol="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    strategy="sma_crossover",
    param_grid={
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100],
    },
    top_n=5,
    sort_by="sharpe_ratio",
))

print(f"Strategy    : {result.strategy}")
print(f"Combos tested: {result.n_combinations}")
print(f"Ranked by   : {result.sort_by}")
print(f"Best params : {result.best_params}")
print()
for run in result.top_results:
    print(f"  #{run.rank}  params={run.parameters}  "
          f"Sharpe={run.sharpe_ratio:.2f}  Return={run.total_return:.1%}  "
          f"MDD={run.max_drawdown:.1%}  Trades={run.num_trades}")
```

**BacktestOptInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Ticker symbol |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `strategy` | str | — | `"sma_crossover"`, `"rsi_mean_reversion"`, `"macd_crossover"`, `"bollinger_reversion"` |
| `param_grid` | dict | — | Mapping of parameter name → list of values to test |
| `initial_capital` | float | 10000 | Starting capital |
| `sort_by` | str | `"sharpe_ratio"` | Ranking metric: `"sharpe_ratio"`, `"total_return"`, `"calmar_ratio"`, `"sortino_ratio"`, `"max_drawdown"` |
| `top_n` | int | 5 | Number of top combinations to return (capped at 20) |
| `n_workers` | int | 1 | CPU workers for parallel grid search |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Ticker |
| `strategy` | `str` | Strategy name |
| `n_combinations` | `int` | Total parameter combinations tested |
| `sort_by` | `str` | Metric used to rank |
| `best_params` | `dict` | Parameter combination of the top-ranked run |
| `best_sharpe` | `float` | Sharpe ratio of the top-ranked run |
| `best_return` | `float` | Total return of the top-ranked run |
| `top_results` | `List[OptimizationRun]` | Top N results, sorted best first |

Each `OptimizationRun` has:

| Field | Type | Description |
|---|---|---|
| `rank` | `int` | 1 = best |
| `parameters` | `dict` | Parameter combination |
| `total_return` | `float` | Total return (decimal) |
| `sharpe_ratio` | `float` | Sharpe ratio for this run |
| `sortino_ratio` | `float` | Sortino ratio for this run |
| `calmar_ratio` | `float` | Calmar ratio for this run |
| `max_drawdown` | `float` | Max drawdown (negative decimal) |
| `num_trades` | `int` | Number of round-trip trades |

**Finding the best SMA parameters then running the full backtest:**

```python
from standard_quant_tools.agent.tools import run_backtest_optimization, run_sma_backtest
from standard_quant_tools.agent.models import BacktestOptInput, BacktestInput

opt = run_backtest_optimization(BacktestOptInput(
    symbol="NVDA",
    start_date="2021-01-01",
    end_date="2024-01-01",
    strategy="sma_crossover",
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    top_n=1,
    sort_by="sharpe_ratio",
))

best = opt.top_results[0]
print(f"Best params: {best.parameters}  Sharpe={best.sharpe_ratio:.2f}")

# Now run with full trade log for the best params
bt = run_sma_backtest(BacktestInput(
    symbol="NVDA",
    start_date="2021-01-01",
    end_date="2024-01-01",
    strategy_type="sma_crossover",
    parameters=best.parameters,
))
print(f"Full run: Sharpe={bt.sharpe_ratio:.2f}  Trades={bt.num_trades}  "
      f"WinRate={bt.win_rate:.1%}  MDD={bt.max_drawdown:.1%}")
```

---

## 8. Advanced Technical Indicators

`get_advanced_indicators` computes three indicators that are not included in `get_technical_analysis`:

- **Parabolic SAR** — a dynamic trailing stop used to identify trend direction and potential reversals
- **Wilder ATR** — the original Wilder smoothed Average True Range, a measure of true price volatility
- **MFI** — Money Flow Index, a volume-weighted RSI that signals overbought/oversold conditions

```python
from standard_quant_tools.agent.tools import get_advanced_indicators
from standard_quant_tools.agent.models import AdvancedIndicatorsInput

result = get_advanced_indicators(AdvancedIndicatorsInput(
    symbol="AAPL",
    start_date="2022-01-01",
    end_date="2024-01-01",
))

print(f"Symbol      : {result.symbol}")
print(f"Last close  : {result.last_close:.2f}")
print(f"SAR trend   : {result.sar_trend}")       # "bullish" or "bearish"
print(f"SAR signal  : {result.sar_signal}")      # "buy" or "sell"
print(f"SAR value   : {result.sar_value:.2f}")
print(f"Wilder ATR  : {result.wilder_atr:.2f}  ({result.wilder_atr_pct:.2%} of price)")
print(f"MFI         : {result.mfi:.1f}")
print(f"MFI signal  : {result.mfi_signal}")      # "overbought", "oversold", or "neutral"
```

**AdvancedIndicatorsInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Ticker symbol |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `sar_af_start` | float | 0.02 | SAR: initial acceleration factor |
| `sar_af_max` | float | 0.2 | SAR: maximum acceleration factor |
| `atr_period` | int | 14 | Wilder ATR period in bars |
| `mfi_period` | int | 14 | MFI lookback period in bars |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Ticker |
| `last_close` | `float` | Most recent closing price |
| `sar_trend` | `str` | `"bullish"` (price above SAR) or `"bearish"` (price below SAR) |
| `sar_signal` | `str` | `"buy"` or `"sell"` — mirrors `sar_trend` |
| `sar_value` | `float` | Most recent SAR value |
| `wilder_atr` | `float` | Most recent Wilder ATR value in price units |
| `wilder_atr_pct` | `float` | ATR as a fraction of closing price |
| `mfi` | `float` | Most recent MFI reading (0–100) |
| `mfi_signal` | `str` | `"overbought"` (≥80), `"oversold"` (≤20), `"neutral"` |

**Interpreting the signals:**

| Indicator | Signal | Interpretation |
|---|---|---|
| SAR trend | `"bullish"` | Price is above SAR — trend is up; SAR acts as trailing support |
| SAR trend | `"bearish"` | Price is below SAR — trend is down; SAR acts as trailing resistance |
| MFI | ≥ 80 | Overbought: buying pressure may be exhausted; watch for reversal |
| MFI | ≤ 20 | Oversold: selling pressure may be exhausted; watch for recovery |
| ATR % | > 3–4% | Elevated volatility — widen stops and reduce position size |
| ATR % | < 1% | Low volatility — tight stops are feasible; breakout potential building |

**Multi-stock signal scan:**

```python
from standard_quant_tools.agent.tools import get_advanced_indicators
from standard_quant_tools.agent.models import AdvancedIndicatorsInput

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]

print(f"{'Ticker':<8} {'SAR':>8} {'ATR%':>7} {'MFI':>6} {'MFI Signal'}")
print("-" * 45)
for t in tickers:
    r = get_advanced_indicators(AdvancedIndicatorsInput(
        symbol=t, start_date="2023-01-01", end_date="2024-01-01",
    ))
    print(f"{r.symbol:<8} {r.sar_trend:>8} {r.wilder_atr_pct:>7.2%} "
          f"{r.mfi:>6.1f} {r.mfi_signal}")
```

---

## 9. Rolling Beta

`get_rolling_beta` computes a rolling OLS beta against a benchmark using a sliding window. Unlike the static beta in `analyze_stock_risk`, this shows how sensitivity to the market has evolved over time — useful for detecting structural shifts (e.g. a stock becoming more/less market-correlated after a strategic pivot or sector rotation).

```python
from standard_quant_tools.agent.tools import get_rolling_beta
from standard_quant_tools.agent.models import RollingBetaInput

result = get_rolling_beta(RollingBetaInput(
    symbol="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    benchmark="SPY",
    window=60,
))

print(f"Symbol        : {result.symbol}")
print(f"Benchmark     : {result.benchmark}")
print(f"Window        : {result.window} bars")
print(f"Current beta  : {result.current_beta:.3f}")
print(f"1m ago        : {result.beta_1m_ago:.3f}" if result.beta_1m_ago else "1m ago: N/A")
print(f"3m ago        : {result.beta_3m_ago:.3f}" if result.beta_3m_ago else "3m ago: N/A")
print(f"6m ago        : {result.beta_6m_ago:.3f}" if result.beta_6m_ago else "6m ago: N/A")
print(f"Beta trend    : {result.beta_trend}")   # "increasing", "decreasing", or "stable"
print(f"Observations  : {result.n_obs}")
```

**RollingBetaInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Asset ticker |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `benchmark` | str | `"SPY"` | Benchmark ticker |
| `window` | int | 60 | Rolling window in trading days |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Asset ticker |
| `benchmark` | `str` | Benchmark ticker |
| `window` | `int` | Rolling window in bars |
| `current_beta` | `float` | Beta at the most recent window |
| `beta_1m_ago` | `float?` | Beta ~22 bars before the end (None if insufficient data) |
| `beta_3m_ago` | `float?` | Beta ~66 bars before the end |
| `beta_6m_ago` | `float?` | Beta ~132 bars before the end |
| `beta_trend` | `str` | `"increasing"` / `"decreasing"` / `"stable"` relative to 22 bars ago |
| `beta_min` | `float` | Minimum rolling beta over the full period |
| `beta_max` | `float` | Maximum rolling beta over the full period |
| `beta_mean` | `float` | Mean rolling beta over the full period |
| `n_obs` | `int` | Number of return observations used |

**Beta trend rule:** `"increasing"` if `current_beta - beta_1m_ago > 0.1`; `"decreasing"` if `< -0.1`; otherwise `"stable"`.

**Comparing rolling beta across a portfolio:**

```python
from standard_quant_tools.agent.tools import get_rolling_beta
from standard_quant_tools.agent.models import RollingBetaInput

holdings = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]

print(f"{'Ticker':<8} {'β now':>7} {'β 6m':>7} {'Drift':>7} {'Trend'}")
print("-" * 42)
for t in holdings:
    r = get_rolling_beta(RollingBetaInput(
        symbol=t, start_date="2022-01-01", end_date="2024-01-01",
    ))
    b6 = f"{r.beta_6m_ago:.3f}" if r.beta_6m_ago else "  N/A"
    drift = f"{r.current_beta - r.beta_6m_ago:+.3f}" if r.beta_6m_ago else "  N/A"
    print(f"{r.symbol:<8} {r.current_beta:>7.3f} {b6:>7} {drift:>7} {r.beta_trend}")

# Flag any stock where beta has shifted by more than 0.2 over 6 months
print("\nBeta-drift alerts (|Δβ| > 0.2 vs 6m ago):")
for t in holdings:
    r = get_rolling_beta(RollingBetaInput(
        symbol=t, start_date="2022-01-01", end_date="2024-01-01",
    ))
    if r.beta_6m_ago and abs(r.current_beta - r.beta_6m_ago) > 0.2:
        print(f"  {t}: β drifted from {r.beta_6m_ago:.2f} → {r.current_beta:.2f}")
```

---

## 10. Extended Risk Metrics

`get_extended_risk_metrics` returns risk metrics that complement `analyze_stock_risk`. Where `analyze_stock_risk` focuses on alpha/beta, Sharpe, and VaR at 95%, this tool adds CAGR, Calmar ratio, Treynor ratio, parametric VaR at both 95% and 99%, historical VaR at 99%, and CVaR at 99%.

```python
from standard_quant_tools.agent.tools import get_extended_risk_metrics
from standard_quant_tools.agent.models import ExtendedRiskInput

result = get_extended_risk_metrics(ExtendedRiskInput(
    symbol="AAPL",
    start_date="2021-01-01",
    end_date="2024-01-01",
    benchmark="SPY",
))

print(f"Symbol              : {result.symbol}")
print(f"Annualized return   : {result.annualized_return:.2%}")
print(f"Beta                : {result.beta:.3f}")
print(f"Calmar ratio        : {result.calmar_ratio:.2f}")
print(f"Treynor ratio       : {result.treynor_ratio:.4f}")
print(f"Parametric VaR 95%  : {result.var_parametric_95:.2%}")
print(f"Parametric VaR 99%  : {result.var_parametric_99:.2%}")
print(f"Historical VaR 99%  : {result.var_historical_99:.2%}")
print(f"CVaR 99%            : {result.cvar_99:.2%}")
```

**ExtendedRiskInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Ticker symbol |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `benchmark` | str | `"SPY"` | Benchmark ticker (required for Treynor ratio) |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `symbol` | `str` | Ticker |
| `benchmark` | `str` | Benchmark ticker |
| `annualized_return` | `float` | CAGR of the asset's equity curve, as a decimal |
| `calmar_ratio` | `float` | CAGR / |max drawdown| — higher is better |
| `treynor_ratio` | `float` | (Return − Risk-free) / Beta — excess return per unit of market risk |
| `var_parametric_95` | `float` | 1-day parametric VaR at 95% confidence (negative = loss) |
| `var_parametric_99` | `float` | 1-day parametric VaR at 99% confidence |
| `var_historical_99` | `float` | 1-day historical VaR at 99% (5th-worst percentile of daily returns) |
| `cvar_99` | `float` | 1-day CVaR at 99% — expected loss in the worst 1% of days |
| `beta` | `float` | OLS beta vs `benchmark` (same computation as `analyze_stock_risk`) |

**Interpreting the metrics:**

| Metric | Good value | Poor value | Notes |
|---|---|---|---|
| Calmar ratio | > 1.0 | < 0.5 | CAGR earned per unit of drawdown risk |
| Treynor ratio | > 0.10 | < 0.05 | Annualised excess return per unit of beta |
| VaR 99% (parametric) | < −2% | < −5% | Single-day loss expected 1% of the time |
| CVaR 99% | < −3% | < −8% | Average loss on the worst 1% of days |

**Full risk picture — combining with analyze_stock_risk:**

```python
from standard_quant_tools.agent.tools import analyze_stock_risk, get_extended_risk_metrics
from standard_quant_tools.agent.models import AnalysisInput, ExtendedRiskInput

ticker = "NVDA"
start, end = "2021-01-01", "2024-01-01"

# Core risk
core = analyze_stock_risk(AnalysisInput(symbol=ticker))
# Extended risk
ext  = get_extended_risk_metrics(ExtendedRiskInput(
    symbol=ticker, start_date=start, end_date=end,
))

print(f"=== {ticker} Complete Risk Profile ===")
print(f"Alpha            : {core.alpha:.4f}")
print(f"Beta             : {core.beta:.3f}")
print(f"Sharpe           : {core.sharpe_ratio:.2f}")
print(f"Sortino          : {core.sortino_ratio:.2f}")
print(f"Max drawdown     : {core.max_drawdown:.1%}")
print(f"VaR 95% (core)   : {core.var_95:.2%}")
print()
print(f"Annualized return: {ext.annualized_return:.2%}")
print(f"Calmar ratio     : {ext.calmar_ratio:.2f}")
print(f"Treynor ratio    : {ext.treynor_ratio:.4f}")
print(f"Param VaR 95%    : {ext.var_parametric_95:.2%}")
print(f"Param VaR 99%    : {ext.var_parametric_99:.2%}")
print(f"Hist  VaR 99%    : {ext.var_historical_99:.2%}")
print(f"CVaR 99%         : {ext.cvar_99:.2%}")
```

---

## 11. Custom Signal Backtest

`run_custom_signal_backtest` does **not** generate a signal — unlike every
`run_*_backtest` tool above, which computes its signal from a named indicator
(SMA, RSI, MACD, Bollinger). This tool backtests a signal *you* (or an
upstream model) already computed, reusing the same fast engine
(`standard_quant_tools.backtest.engine.run_strategy`, with the C++ kernel
when built) without assuming anything about how the signal was produced.

**When to use:** the user references or supplies their own alpha logic —
"backtest my momentum score", "here's my model's output, evaluate it" —
rather than asking for one of the built-in strategies. Never substitute a
built-in strategy when the user's intent is to test their own signal.

```python
from standard_quant_tools.agent.tools import run_custom_signal_backtest
from standard_quant_tools.agent.models import CustomSignalBacktestInput

# `signals` would normally come from your own model — a toy example here:
signals = {
    "2023-01-03": 1.0, "2023-01-04": 1.0, "2023-01-05": 0.0,
    "2023-01-06": 0.0, "2023-01-09": -1.0,
    # ... one entry per trading day in [start_date, end_date]
}

result = run_custom_signal_backtest(CustomSignalBacktestInput(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01",
    signals=signals,
))

print(f"Sharpe Ratio  : {result.sharpe_ratio:.2f}")
print(f"Total Return  : {result.total_return:.1%}")
print(f"Num Trades    : {result.num_trades}")
```

**CustomSignalBacktestInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `symbol` | str | — | Ticker symbol |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `signals` | `Dict[str, float]` | — | `{date: value}`, value in `{1, 0, -1}` (long/flat/short). Dates without a matching OHLCV bar are ignored, same as extra OHLCV bars with no signal entry. |
| `initial_capital` | float | `10000` | Starting capital |
| `commission_pct` | float | `0.001` | Commission per trade (fraction) |
| `slippage_pct` | float | `0.0005` | Slippage per trade (fraction) |

**Output:** `BacktestResult` — identical shape to `run_sma_backtest` / `run_rsi_backtest` / etc. See the full reference in [07_agent_tools.md](07_agent_tools.md#backtestinput--backtestresult--full-reference).

**Chaining into position sizing — exactly the "let my own agent optimize risk parameters" workflow:**

```python
from standard_quant_tools.agent.tools import run_custom_signal_backtest, get_position_size
from standard_quant_tools.agent.models import CustomSignalBacktestInput, PositionSizerInput

bt = run_custom_signal_backtest(CustomSignalBacktestInput(
    symbol="AAPL", start_date="2022-01-01", end_date="2024-01-01",
    signals=my_model_signals,   # your own model's output
))

# Use the resulting win-rate / trade stats to size the next position —
# the library never had to know how the signal itself was generated.
pos = get_position_size(PositionSizerInput(
    symbol="AAPL", start_date="2023-06-01", end_date="2024-01-01",
    account_equity=100_000,
    win_rate=bt.win_rate,
    avg_win_pct=max(bt.avg_trade_return_pct / 100, 0.001),
    avg_loss_pct=abs(min(bt.avg_trade_return_pct / 100, -0.001)),
))
print(f"Sharpe: {bt.sharpe_ratio:.2f}  →  Recommended: {pos.recommended_shares} shares")
```

---

## 12. Signal Panel Backtest

`run_signal_panel_backtest` extends the same idea to a **ticker universe**:
pass a pre-computed signal matrix (e.g. a cross-sectional alpha model's
output) and get back per-ticker backtest results plus portfolio-level
metrics. It fetches OHLCV internally, runs `run_strategy` per ticker, and
combines the realized returns via the existing portfolio module
(`build_portfolio` / `portfolio_metrics`) — no new backtest math, and no
assumption about how the signal was generated.

```python
from standard_quant_tools.agent.tools import run_signal_panel_backtest
from standard_quant_tools.agent.models import SignalPanelBacktestInput

tickers = ["AAPL", "MSFT", "GOOGL"]

# signal_panel would normally come from your own cross-sectional model
signal_panel = {
    "AAPL": {"2023-01-03": 1.0, "2023-01-04": 1.0, "2023-01-05": 0.0},
    "MSFT": {"2023-01-03": 0.0, "2023-01-04": 1.0, "2023-01-05": 1.0},
    "GOOGL": {"2023-01-03": -1.0, "2023-01-04": 0.0, "2023-01-05": 0.0},
    # ... one entry per trading day in [start_date, end_date] for each ticker
}

result = run_signal_panel_backtest(SignalPanelBacktestInput(
    tickers=tickers,
    start_date="2023-01-01",
    end_date="2024-01-01",
    signal_panel=signal_panel,
    weights={"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.25},   # optional — default equal weight
    benchmark="SPY",   # optional — adds information_ratio to portfolio_metrics
))

# Per-ticker results — same shape as run_custom_signal_backtest's output
for t in tickers:
    r = result.per_ticker[t]
    print(f"{t}: Sharpe={r.sharpe_ratio:.2f}  Return={r.total_return:.1%}  Trades={r.num_trades}")

# Portfolio-level combination
pm = result.portfolio_metrics
print(f"\nPortfolio Sharpe : {pm['sharpe_ratio']:.2f}")
print(f"Portfolio Return : {pm['total_return']:.1%}")
print(f"Portfolio VaR 95%: {pm['var_95']:.4f}")
```

**SignalPanelBacktestInput fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `tickers` | `List[str]` | — | Universe; must match `signal_panel`'s outer keys |
| `start_date` | str | — | ISO date |
| `end_date` | str | — | ISO date |
| `signal_panel` | `Dict[str, Dict[str, float]]` | — | `{ticker: {date: value}}`, value in `{1, 0, -1}` |
| `weights` | `Dict[str, float]?` | `None` | Per-ticker weight, must sum to 1.0. Defaults to equal weight. |
| `initial_capital` | float | `10000` | Starting capital applied per ticker |
| `commission_pct` | float | `0.001` | Commission per trade (fraction) |
| `slippage_pct` | float | `0.0005` | Slippage per trade (fraction) |
| `benchmark` | str? | `None` | Optional benchmark ticker — adds `information_ratio` to `portfolio_metrics` |
| `include_trade_log` | bool | `False` | If `True`, include a per-trade log for each ticker |

**Output reference:**

| Field | Type | Description |
|---|---|---|
| `tickers` | `List[str]` | Universe, in `signal_panel`'s order |
| `per_ticker` | `Dict[str, BacktestResult]` | One full backtest result per ticker |
| `portfolio_metrics` | `dict` | Same shape as `portfolio.portfolio_metrics()`: `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_95`, `cvar_95`, `total_return`, `tickers`, `weights`, plus `information_ratio` when `benchmark` was set |

**Validation:** `signal_panel` must have an entry for every ticker in `tickers`; if `weights` is given, its keys must exactly match `tickers` and sum to 1.0 — both raise a Pydantic `ValidationError` with the offending ticker(s) named directly, so the calling agent can retry with a corrected payload.

**Note on scale:** per-ticker equity curves are aligned to their common date range (inner join) before being combined into the portfolio — a ticker whose signal/price data doesn't fully cover the requested range will shrink the portfolio's effective date range. For very large universes, prefer calling this tool once per rebalance period rather than once per ticker.
```
