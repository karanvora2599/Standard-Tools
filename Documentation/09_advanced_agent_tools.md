# Advanced Agent Tools

Five high-level agentic tools that compose the library's existing primitives into single, LLM-callable operations. Each collapses a multi-step reasoning workflow into one structured function call with a Pydantic output model.

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

---

## 2. Cointegration Pair Scanner

`scan_pairs` tests all O(n²/2) ticker combinations for cointegration, filters by p-value and half-life bounds, and returns the top N pairs sorted by half-life (shortest first = fastest mean reversion = most tradeable). Each ticker's prices are fetched **once** before testing begins.

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
