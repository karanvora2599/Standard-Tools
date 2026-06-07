# Advanced Agent Tools

Five high-level agentic tools that compose the library's existing primitives into single, LLM-callable operations. Each collapses a multi-step reasoning workflow into one structured function call with a Pydantic output model.

> **See also:** [07_agent_tools.md](07_agent_tools.md) covers the 12 core tools, the full `get_agent_tools()` registry (all 17), OpenAI/Anthropic wiring, and the complete Model Summary.

---

## Tool Summary

| Tool | What it does | Key output fields |
|---|---|---|
| `run_regime_adaptive_backtest` | Classify regime via Hurst, auto-select and optimise strategy | `regime`, `selected_strategy`, `best_parameters`, `backtest` |
| `scan_pairs` | Find cointegrated pairs in a universe, ranked by half-life | `pairs[].half_life_days`, `pairs[].signal` |
| `run_walk_forward_backtest` | Optimise in-sample, validate out-of-sample across rolling windows | `avg_oos_sharpe`, `pct_windows_profitable`, `param_stability` |
| `get_portfolio_risk_attribution` | Deep risk decomposition: MCR, PCA, optional factor model | `asset_risk_contributions`, `pca_variance_explained` |
| `get_position_size` | ATR stop-loss sizing with optional Kelly criterion | `shares_fixed_risk`, `kelly_fraction`, `recommended_shares` |

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
| `regime` | `str` | `"trending"`, `"random_walk"`, or `"mean_reverting"` |
| `hurst` | `float` | Hurst exponent H |
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
