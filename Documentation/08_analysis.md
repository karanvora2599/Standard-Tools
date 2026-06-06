# Analysis

The analysis module provides statistical tools for understanding return series, factor exposures, and market structure. All functions are pure NumPy / Pandas — no heavy dependencies beyond what the core library already requires. `scipy` is used for precise p-values when available and falls back gracefully to a `math.erf`-based normal approximation otherwise.

---

## Multi-Factor Regression

`multi_factor_regression` runs OLS on N factors simultaneously — the generalisation of the single-factor beta calculation already in the library. It returns factor loadings, inferential statistics (t-stats, p-values), and goodness-of-fit metrics in one call.

### When to use
- Decompose an asset's return into known risk premia (market, size, value, momentum, quality …)
- Measure Jensen's alpha after controlling for multiple systematic factors
- Compare factor exposures across stocks or time periods
- Feed factor loadings directly to an LLM agent for natural-language interpretation

### Basic usage

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.analysis import multi_factor_regression

provider = DataFactory.get_provider()

# Fetch asset and factors
aapl  = provider.get_ohlcv("AAPL",  "2021-01-01", "2024-01-01")["Close"].pct_change().dropna()
spy   = provider.get_ohlcv("SPY",   "2021-01-01", "2024-01-01")["Close"].pct_change().dropna()
iwm   = provider.get_ohlcv("IWM",   "2021-01-01", "2024-01-01")["Close"].pct_change().dropna()
tlt   = provider.get_ohlcv("TLT",   "2021-01-01", "2024-01-01")["Close"].pct_change().dropna()

import pandas as pd
factors = pd.DataFrame({"mkt": spy, "smb_proxy": iwm, "bond": tlt})

result = multi_factor_regression(aapl, factors)

print(f"Alpha (daily)     : {result['alpha']:.6f}")
print(f"Market loading    : {result['loadings']['mkt']:.3f}")
print(f"Size proxy loading: {result['loadings']['smb_proxy']:.3f}")
print(f"Bond loading      : {result['loadings']['bond']:.3f}")
print(f"R²                : {result['r_squared']:.3f}")
print(f"Adj. R²           : {result['adj_r_squared']:.3f}")
```

### Interpreting t-stats and p-values

```python
for factor, t in result["t_stats"].items():
    p = result["p_values"][factor]
    sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {factor:<15} t={t:+.2f}  p={p:.3f} {sig}")
```

| Significance | p-value threshold |
|---|---|
| `***` | < 0.01 |
| `**`  | < 0.05 |
| `*`   | < 0.10 |
| (none)| ≥ 0.10 |

### Fama-French 3-Factor example

```python
# Build approximate FF3 factors from ETF proxies
factors_ff3 = pd.DataFrame({
    "mkt":  spy,          # market excess return proxy
    "smb":  iwm - spy,    # small-minus-big proxy (IWM - SPY)
    "hml":  provider.get_ohlcv("IWD", "2021-01-01", "2024-01-01")["Close"].pct_change().dropna(),
})

tsla = provider.get_ohlcv("TSLA", "2021-01-01", "2024-01-01")["Close"].pct_change().dropna()
result = multi_factor_regression(tsla, factors_ff3)

print(f"TSLA market beta  : {result['loadings']['mkt']:.2f}")
print(f"TSLA size loading : {result['loadings']['smb']:.2f}")
print(f"TSLA value loading: {result['loadings']['hml']:.2f}")
print(f"R²                : {result['r_squared']:.3f}")
```

### Output reference

| Key | Type | Description |
|---|---|---|
| `alpha` | `float` | Regression intercept (raw daily units, not annualised) |
| `loadings` | `dict[str, float]` | One entry per factor column |
| `t_stats` | `dict[str, float]` | Includes `"alpha"` plus all factor names |
| `p_values` | `dict[str, float]` | Two-tailed p-values; `"alpha"` included |
| `r_squared` | `float` | Fraction of variance explained |
| `adj_r_squared` | `float` | R² penalised for number of factors |
| `n_obs` | `int` | Number of observations used after index alignment |

> **Index alignment** — `asset_returns` and `factor_returns` are automatically aligned on their common index before fitting. Mismatched lengths or sparse data are handled without errors.

> **Insufficient data** — when `n_obs < n_factors + 2`, all numeric fields are returned as `nan` rather than raising.

---

## Rolling Factor Loadings

`rolling_factor_loadings` applies the same OLS over a sliding window, producing a time-series of factor exposures. Useful for detecting regime shifts in factor sensitivity.

### Basic usage

```python
from standard_quant_tools.analysis import rolling_factor_loadings

rolling = rolling_factor_loadings(aapl, factors, window=60)

# rolling is a DataFrame: index = dates, columns = ["alpha", "mkt", "smb_proxy", "bond"]
print(rolling.tail())
```

### Detecting regime shifts

```python
import plotly.graph_objects as go

fig = go.Figure()
for col in ["mkt", "smb_proxy"]:
    fig.add_trace(go.Scatter(
        x=rolling.index,
        y=rolling[col],
        name=f"Rolling {col} loading",
    ))
fig.update_layout(title="60-Day Rolling Factor Loadings — AAPL")
fig.show()
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `asset_returns` | `pd.Series` | required | Daily return series |
| `factor_returns` | `pd.DataFrame` | required | One column per factor |
| `window` | `int` | `60` | Lookback window in bars |

### Output

A `pd.DataFrame` with the same index as `asset_returns` (after alignment) and columns `["alpha", factor1, factor2, ...]`. The first `window - 1` rows are `NaN`.

---

## Combining with other modules

### Factor exposure → Screener pipeline

```python
from standard_quant_tools.screener import screen_stocks
from standard_quant_tools.analysis import multi_factor_regression

# Step 1: screen for liquid large-caps
universe = screen_stocks(
    tickers=["AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL"],
    filters={"market_cap_min": 100_000_000_000, "pe_ratio_max": 50},
)

# Step 2: run factor model on each survivor
spy_ret = provider.get_ohlcv("SPY", "2022-01-01", "2024-01-01")["Close"].pct_change().dropna()
factor_df = pd.DataFrame({"mkt": spy_ret})

for ticker in universe.index:
    rets = provider.get_ohlcv(ticker, "2022-01-01", "2024-01-01")["Close"].pct_change().dropna()
    r = multi_factor_regression(rets, factor_df)
    print(f"{ticker}: beta={r['loadings']['mkt']:.2f}, alpha={r['alpha']:.5f}, R²={r['r_squared']:.2f}")
```

### Via Agent Tool

The analysis module is not yet wrapped as a standalone agent tool, but its results are plain dicts and DataFrames — JSON-serialise the dict with `json.dumps(result)` before passing to an LLM.

```python
import json
from standard_quant_tools.analysis import multi_factor_regression

result = multi_factor_regression(asset_returns, factor_returns)

# Prepare LLM-friendly payload (exclude n_obs if not needed)
payload = {k: v for k, v in result.items() if k != "n_obs"}
print(json.dumps(payload, indent=2))
```

---

---

## Cointegration & Pairs Spread Analysis *(C++ / statsmodels)*

Two price series are **cointegrated** when a linear combination of them is stationary, even though each series individually follows a random walk. This is the statistical foundation of pairs trading.

`cointegration_test` uses the **C++ extension** (`_sqt_core`) when available — a self-contained Engle-Granger implementation (OLS + ADF + MacKinnon 2010 response surface) with no dependency on `statsmodels`. The C++ path is **5–15× faster** on typical series lengths (n = 250–1 000). The statsmodels fallback is used automatically when the extension is not built; the API and return format are identical either way.

The toolkit provides four functions covering the full pairs workflow:

| Function | Purpose |
|---|---|
| `cointegration_test` | Engle-Granger test — is the pair cointegrated? |
| `compute_spread` | Compute the hedged residual (the tradeable spread) |
| `half_life` | How quickly does the spread mean-revert? |
| `spread_zscore` | Normalise the spread into a trading signal |

### Cointegration test

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.analysis import cointegration_test

provider = DataFactory.get_provider()

ko  = provider.get_ohlcv("KO",  "2020-01-01", "2024-01-01")["Close"]
pep = provider.get_ohlcv("PEP", "2020-01-01", "2024-01-01")["Close"]

result = cointegration_test(ko, pep)

print(f"Cointegrated     : {result['cointegrated']}")
print(f"Hedge ratio      : {result['hedge_ratio']:.4f}")
print(f"ADF statistic    : {result['adf_statistic']:.4f}")
print(f"p-value          : {result['p_value']:.4f}")
print(f"Half-life (days) : {result['half_life_days']:.1f}")
print(f"Critical values  : {result['critical_values']}")
```

The test uses **MacKinnon (2010) p-values** — these are stricter than standard ADF critical values because they account for the fact that we are testing residuals from a fitted regression, not the original series.

To check which execution path is active:

```python
from standard_quant_tools.analysis.cointegration import HAS_CPP
print("C++ cointegration active:", HAS_CPP)
# True  → compiled extension found; OLS + ADF computed in C++
# False → extension not built; statsmodels fallback used automatically
```

#### Output reference

| Key | Type | Description |
|---|---|---|
| `cointegrated` | `bool` | `True` when `p_value < 0.05` |
| `hedge_ratio` | `float` | OLS coefficient: `series_a ≈ α + hedge_ratio × series_b` |
| `adf_statistic` | `float` | ADF t-statistic on the spread |
| `p_value` | `float` | MacKinnon cointegration p-value |
| `critical_values` | `dict` | `{"1%": ..., "5%": ..., "10%": ...}` — sample-size adjusted |
| `half_life_days` | `float` | AR(1) mean-reversion half-life in bars |
| `n_obs` | `int` | Observations used after index alignment |

---

### Computing the spread

```python
from standard_quant_tools.analysis import compute_spread

# Auto-estimate hedge ratio via OLS (zero-mean residual by construction)
spread = compute_spread(ko, pep)

# Or supply a known/previously fitted ratio
spread = compute_spread(ko, pep, hedge_ratio=result["hedge_ratio"])

print(spread.describe())
```

`compute_spread` returns a `pd.Series` named `"spread"`, aligned to the common index of the two inputs.

---

### Half-life of mean reversion

```python
from standard_quant_tools.analysis import half_life

hl = half_life(spread)
print(f"Half-life: {hl:.1f} bars")
# Rule of thumb:
#   < 5 bars  → too fast, hard to trade (slippage kills it)
#   5–30 bars → sweet spot for daily-bar mean reversion
#   > 60 bars → slow; may need patience or tighter entry thresholds
```

Internally fits `ΔS_t = α + β · S_{t-1} + ε` via OLS and computes `−ln(2) / β`.
Returns `float('inf')` when the spread is not mean-reverting (β ≥ 0).

---

### Z-score signal

```python
from standard_quant_tools.analysis import spread_zscore

# Static (full-sample) normalisation — best for research/backtesting
z_static = spread_zscore(spread)

# Rolling normalisation — required for live trading (avoids lookahead)
z_rolling = spread_zscore(spread, window=30)

# Typical entry/exit thresholds
entry_long  = z_rolling < -2.0   # buy series_a, sell series_b
entry_short = z_rolling >  2.0   # sell series_a, buy series_b
exit_signal = z_rolling.abs() < 0.5
```

---

### Full pairs trading workflow

```python
import pandas as pd
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.analysis import (
    cointegration_test, compute_spread, half_life, spread_zscore
)
from standard_quant_tools.backtest.engine import run_strategy

provider = DataFactory.get_provider()
start, end = "2020-01-01", "2024-01-01"

# 1. Load price series
ko  = provider.get_ohlcv("KO",  start, end)["Close"]
pep = provider.get_ohlcv("PEP", start, end)["Close"]

# 2. Test for cointegration
result = cointegration_test(ko, pep)
if not result["cointegrated"]:
    raise ValueError("Pair is not cointegrated — do not trade")

print(f"Hedge ratio: {result['hedge_ratio']:.3f}, Half-life: {result['half_life_days']:.1f} days")

# 3. Build spread and z-score
spread = compute_spread(ko, pep, hedge_ratio=result["hedge_ratio"])
hl = half_life(spread)
window = max(int(hl * 2), 10)          # rolling window = 2× half-life
z = spread_zscore(spread, window=window)

# 4. Generate mean-reversion signals on series_a (KO)
import numpy as np
entry_thresh, exit_thresh = 2.0, 0.5
signals = pd.Series(0.0, index=z.index)
in_pos = 0
for i in range(len(z)):
    if z.isna().iloc[i]:
        continue
    zv = z.iloc[i]
    if in_pos == 0:
        if zv < -entry_thresh:
            in_pos = 1   # spread too low → long KO
        elif zv > entry_thresh:
            in_pos = -1  # spread too high → short KO
    elif in_pos != 0 and abs(zv) < exit_thresh:
        in_pos = 0
    signals.iloc[i] = float(in_pos)

# 5. Backtest on series_a's OHLCV
ohlcv_ko = provider.get_ohlcv("KO", start, end)
bt = run_strategy(ohlcv_ko, signals, initial_capital=10_000, commission_pct=0.001)
print(f"Sharpe: {bt['sharpe_ratio']:.2f}, Max DD: {bt['max_drawdown']:.1%}")
```

---

### Screening for cointegrated pairs

```python
from itertools import combinations
import pandas as pd
from standard_quant_tools.analysis import cointegration_test

tickers = ["KO", "PEP", "MCD", "WEN", "YUM", "SBUX"]
provider = DataFactory.get_provider()

prices = {t: provider.get_ohlcv(t, "2021-01-01", "2024-01-01")["Close"]
          for t in tickers}

rows = []
for a, b in combinations(tickers, 2):
    r = cointegration_test(prices[a], prices[b])
    if r["cointegrated"]:
        rows.append({
            "pair": f"{a}/{b}",
            "hedge_ratio": round(r["hedge_ratio"], 3),
            "p_value": round(r["p_value"], 4),
            "half_life": round(r["half_life_days"], 1),
        })

pairs_df = pd.DataFrame(rows).sort_values("half_life")
print(pairs_df)
```

---

---

## PCA on Returns

`pca_returns` decomposes a multi-asset return matrix into orthogonal principal components (PCs) using full SVD — pure NumPy, no sklearn required. `factor_contributions` then quantifies how much each PC explains for each individual asset.

### When to use

| Question | Tool |
|---|---|
| "What are the dominant hidden risk factors in my universe?" | `pca_returns` |
| "How correlated is NVDA to the market's first risk factor?" | `pca_returns` → loadings |
| "Does adding more PCs actually explain more of AAPL's variance?" | `factor_contributions` |
| "How many PCs do I need to capture 90% of portfolio variance?" | `cumulative_variance_ratio` |

### Basic usage

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.analysis import pca_returns
import pandas as pd

provider = DataFactory.get_provider()
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "GS", "BAC"]
start, end = "2021-01-01", "2024-01-01"

returns = pd.DataFrame({
    t: provider.get_ohlcv(t, start, end)["Close"].pct_change()
    for t in tickers
}).dropna()

result = pca_returns(returns, n_components=3)

print(result["explained_variance_ratio"])
# PC1    0.421
# PC2    0.118
# PC3    0.083

print(result["cumulative_variance_ratio"])
# PC1    0.421
# PC2    0.539
# PC3    0.622

print(result["loadings"])
#        PC1    PC2    PC3
# AAPL  0.35  -0.12   0.08
# MSFT  0.34  -0.14  ...
# ...
```

### Reading the output

#### Explained variance ratio

```python
evr = result["explained_variance_ratio"]
# How many PCs to reach 80% explained variance?
n_for_80 = (result["cumulative_variance_ratio"] < 0.80).sum() + 1
print(f"Need {n_for_80} PCs to explain 80% of variance")
```

#### Loadings (eigenvectors)

Each column of the loadings matrix is a principal component direction in asset space. A high positive loading on PC1 means the asset moves strongly with the first risk factor.

```python
loadings = result["loadings"]

# Which assets load most heavily on PC1?
print(loadings["PC1"].sort_values(ascending=False))

# Assets that load in opposite directions on PC2 tend to hedge each other
hedges = loadings["PC2"].sort_values()
print("PC2 shorts:", hedges.head(3).index.tolist())
print("PC2 longs :", hedges.tail(3).index.tolist())
```

#### Factor returns (PC time series)

```python
factor_rets = result["factor_returns"]
# factor_rets["PC1"] is the time series of the first risk factor
# Use it as a systematic benchmark or for regime analysis

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=factor_rets.index, y=factor_rets["PC1"], name="PC1"))
fig.add_trace(go.Scatter(x=factor_rets.index, y=factor_rets["PC2"], name="PC2"))
fig.update_layout(title="First Two Risk Factors")
fig.show()
```

### Output reference

| Key | Type | Description |
|---|---|---|
| `explained_variance_ratio` | `pd.Series` | Fraction of total variance per PC, indexed "PC1", "PC2", ... |
| `cumulative_variance_ratio` | `pd.Series` | Running sum of EVR |
| `loadings` | `pd.DataFrame` | Shape (n_assets × n_components). Each column is a unit-norm eigenvector. |
| `factor_returns` | `pd.DataFrame` | Shape (n_dates × n_components). PC time series; pairwise correlations are exactly 0. |
| `n_components` | `int` | Actual number of PCs returned (capped at min(n_assets, n_obs)) |
| `n_obs` | `int` | Rows used after dropping NaN |

> **Sign convention** — SVD eigenvectors have arbitrary signs. `pca_returns` normalises each PC so its largest-magnitude loading is positive, making factors easier to interpret.

> **Standardisation** — `standardize=True` (default) divides each asset column by its standard deviation before fitting. This ensures high-volatility assets don't dominate the decomposition. Set `standardize=False` only when columns are already on a comparable scale (e.g. z-scored returns).

---

### Factor contributions per asset

`factor_contributions` answers: *"for asset X, how much does each PC explain?"*

If you have already called `pca_returns`, pass the result directly to avoid running SVD twice:

```python
from standard_quant_tools.analysis import pca_returns, factor_contributions

result = pca_returns(returns, n_components=3)
contrib = factor_contributions(returns, n_components=3, pca_result=result)  # no second SVD
```

Without the pre-computed result:

```python
contrib = factor_contributions(returns, n_components=3)
print(contrib)
#        PC1    PC2    PC3
# AAPL  0.38   0.09   0.04
# MSFT  0.36   0.11   0.03
# NVDA  0.28   0.05   0.12   # more idiosyncratic — less explained by factors
# JPM   0.41   0.18   0.02   # highly systematic
# ...

# Total systematic R² for each asset
print(contrib.sum(axis=1).sort_values())

# Which assets are most idiosyncratic (least explained by top 3 PCs)?
print("Most idiosyncratic:", contrib.sum(axis=1).nsmallest(3).index.tolist())
```

Each cell is the **marginal R²** added by including that PC. Values are additive: `contrib["PC1"] + contrib["PC2"] + contrib["PC3"]` equals the total R² from regressing the asset on the first 3 PCs.

---

### Combining PCA with the portfolio module

PCA and portfolio analysis work naturally together. Factor returns can serve as benchmark series for the metrics module.

```python
from standard_quant_tools.analysis import pca_returns
from standard_quant_tools.metrics import sharpe_ratio, information_ratio
import pandas as pd

result = pca_returns(returns, n_components=2)
pc1 = result["factor_returns"]["PC1"]

# Treat PC1 as the "market" — compute each asset's IR vs the first factor
for ticker in returns.columns:
    ir = information_ratio(returns[ticker].dropna(), pc1, periods_per_year=252)
    print(f"{ticker}: IR vs PC1 = {ir:.2f}")
```

### Portfolio risk decomposition

```python
import numpy as np

# What fraction of equal-weight portfolio variance comes from PC1?
weights = np.ones(len(tickers)) / len(tickers)
loadings = result["loadings"].to_numpy()

# Portfolio loading on each PC = weighted sum of asset loadings
port_loadings = weights @ loadings  # (n_components,)
evr = result["explained_variance_ratio"].values

# Approximate variance attribution (valid when PCs are orthogonal, which they are)
pc_contrib_to_portfolio = port_loadings ** 2 * evr
pc_contrib_to_portfolio /= pc_contrib_to_portfolio.sum()

for i, name in enumerate(result["explained_variance_ratio"].index):
    print(f"{name}: {pc_contrib_to_portfolio[i]:.1%} of equal-weight portfolio variance")
```

---

---

## Hurst Exponent

The Hurst exponent H classifies the long-memory scaling behaviour of a return series. It is the single most useful number for deciding which *class* of strategy to apply to a market.

| H value | Regime | Strategy implication |
|---|---|---|
| H > 0.55 | **Trending** | Momentum strategies — recent direction tends to continue |
| 0.45 ≤ H ≤ 0.55 | **Random walk** | No persistent edge from past prices alone |
| H < 0.45 | **Mean-reverting** | Contrarian / mean-reversion strategies — overshoots tend to reverse |

> **Input must be returns, not prices.** Pass `close.pct_change().dropna()` or log-returns — not the price series itself. The algorithm works on the scaling of cumulative return fluctuations.

---

### C++ Acceleration

The Hurst module ships with an optional compiled C++ backend (`_sqt_core`). When the extension is built it is used automatically — the Python API is identical either way.

| Operation | Python fallback | C++ (`_sqt_core`) | Speedup |
|---|---|---|---|
| `hurst_exponent` single call (n = 500) | ~5–15 ms | ~0.1–0.5 ms | **20–80×** |
| `rolling_hurst` (2 000 bars, window = 200) | ~5–15 s | ~0.1–0.3 s | **30–100×** |

The rolling gain is the most significant: rather than calling back into Python for every bar, the entire sliding-window pass executes inside one C++ function. This makes `rolling_hurst` practical for live pipelines and for `run_regime_adaptive_backtest`, which calls it internally.

#### Check which path is active

```python
from standard_quant_tools.analysis.hurst import HAS_CPP
print("C++ backend active:", HAS_CPP)
# True  → compiled extension found; all calls use the fast C++ path
# False → extension not built; pure Python fallback is used automatically
```

#### Build the extension

Full platform instructions are in [Development/build_guide.md](../Development/build_guide.md). Quick start:

```bash
# Prerequisites (once)
pip install pybind11 cmake ninja

# Build (run from the project root)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Verify
python -c "from standard_quant_tools.analysis.hurst import HAS_CPP; print('C++ active:', HAS_CPP)"
```

**Windows note:** Open "x64 Native Tools Command Prompt for VS 2022" before running cmake, or use the Visual Studio generator from any terminal:

```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The compiled file (`_sqt_core.pyd` on Windows, `_sqt_core.*.so` on Linux/macOS) is written directly into the package directory — no install step needed.

---

### Method: DFA vs R/S

Two methods are available:

| Method | `method=` | Notes |
|---|---|---|
| Detrended Fluctuation Analysis | `"dfa"` (default) | Less biased for typical daily bar counts (200–2000). Recommended. |
| Rescaled Range | `"rs"` | Classic method; biased upward for small samples. Available for comparison. |

Both methods are implemented in the C++ extension and in the Python fallback. DFA is the better default for financial time series of typical length.

---

### Basic usage

The API is the same regardless of whether the C++ extension is built.

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.analysis import hurst_exponent

provider = DataFactory.get_provider()
close = provider.get_ohlcv("AAPL", "2020-01-01", "2024-01-01")["Close"]
returns = close.pct_change().dropna()

result = hurst_exponent(returns)

print(f"H              : {result['hurst']:.3f}")
print(f"Regime         : {result['regime']}")
print(f"Fit R²         : {result['fit_r_squared']:.3f}")
print(f"Method         : {result['method']}")
print(f"n_obs          : {result['n_obs']}")
```

Using R/S instead of DFA:

```python
result_rs = hurst_exponent(returns, method="rs")
print(f"H (R/S)  : {result_rs['hurst']:.3f}")
print(f"H (DFA)  : {result['hurst']:.3f}")
# R/S tends to read slightly higher; DFA is preferred for short series
```

Restricting the scaling range:

```python
# Useful when you want to focus on a specific time-scale band
result = hurst_exponent(returns, method="dfa", min_window=20, max_window=100)
```

### Output reference

| Key | Type | Description |
|---|---|---|
| `hurst` | `float` | Estimated H value (typically 0 < H < 1) |
| `regime` | `str` | `"trending"`, `"random_walk"`, or `"mean_reverting"` |
| `fit_r_squared` | `float` | R² of the log-log scaling fit. Values > 0.90 indicate a reliable estimate. |
| `method` | `str` | Method used (`"dfa"` or `"rs"`) |
| `n_obs` | `int` | Observations used after dropping NaN |

> **Reliability guide** — `fit_r_squared` tells you how cleanly the series follows a power-law at the tested window sizes. Below 0.85, treat the H estimate with caution. Insufficient data (fewer than `min_window × 4` observations) returns `hurst=nan` and `regime="unknown"` rather than raising.

---

### Screening assets by regime

```python
import pandas as pd
from standard_quant_tools.analysis import hurst_exponent
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()
tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "GLD", "TLT", "BTC-USD"]
start, end = "2021-01-01", "2024-01-01"

rows = []
for ticker in tickers:
    try:
        rets = provider.get_ohlcv(ticker, start, end)["Close"].pct_change().dropna()
        r = hurst_exponent(rets)
        rows.append({
            "ticker":  ticker,
            "hurst":   round(r["hurst"], 3),
            "regime":  r["regime"],
            "fit_r2":  round(r["fit_r_squared"], 3),
        })
    except Exception:
        pass

df = pd.DataFrame(rows).sort_values("hurst")
print(df)

# Split by regime
trending = df[df["regime"] == "trending"]["ticker"].tolist()
mean_rev = df[df["regime"] == "mean_reverting"]["ticker"].tolist()
print(f"Trending:       {trending}")
print(f"Mean-reverting: {mean_rev}")
```

With the C++ extension this loop runs in under a second for 8 tickers on 3 years of data. Without it, each `hurst_exponent` call takes ~5–15 ms, so the loop still completes in well under a second at this scale — the C++ gain becomes dominant only in `rolling_hurst` and in screening hundreds of tickers.

---

### Rolling Hurst — regime shift detection

`rolling_hurst` computes H over a sliding window, making it possible to detect when a market switches regimes. This is where the C++ extension provides its largest benefit: without it, a 2 000-bar series at `window=252` takes 5–15 seconds; with it, the same call takes under 300 ms.

```python
from standard_quant_tools.analysis import rolling_hurst

returns = provider.get_ohlcv("SPY", "2018-01-01", "2024-01-01")["Close"].pct_change().dropna()

# window=252 (one trading year), step=5 to compute every 5 bars
rolling = rolling_hurst(returns, window=252, step=5)

# Identify regime periods
import numpy as np
trending_mask = rolling > 0.55
mean_rev_mask = rolling < 0.45

print(f"Trending bars  : {trending_mask.sum()}")
print(f"Mean-rev bars  : {mean_rev_mask.sum()}")

# Fraction of time each regime was active
total_valid = rolling.dropna()
print(f"Fraction trending   : {(total_valid > 0.55).mean():.1%}")
print(f"Fraction mean-rev   : {(total_valid < 0.45).mean():.1%}")
print(f"Fraction random walk: {((total_valid >= 0.45) & (total_valid <= 0.55)).mean():.1%}")
```

> **`step` parameter without C++** — setting `step > 1` skips bars and fills them with `NaN`, reducing total calls proportionally. Even without the C++ extension, `step=5` makes a 2 000-bar series ~5× faster. With the C++ extension the entire pass runs in one shot regardless of `step`, so `step` becomes a resolution choice rather than a performance lever.

#### Visualising regime shifts

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=rolling.index, y=rolling, name="Rolling H (252d, step=5)"))
fig.add_hline(y=0.55, line_dash="dash", line_color="green",
              annotation_text="Trending threshold")
fig.add_hline(y=0.45, line_dash="dash", line_color="red",
              annotation_text="Mean-revert threshold")
fig.add_hline(y=0.50, line_dash="dot", line_color="gray",
              annotation_text="Random walk")
fig.update_layout(title="Rolling Hurst Exponent — SPY", yaxis_title="H")
fig.show()
```

---

### Using Hurst to select a strategy

The Hurst regime output feeds naturally into the backtesting module.

```python
from standard_quant_tools.analysis import hurst_exponent
from standard_quant_tools.agent.tools import run_rsi_backtest, run_sma_backtest
from standard_quant_tools.agent.models import BacktestInput

provider = DataFactory.get_provider()
ticker, start, end = "MSFT", "2021-01-01", "2024-01-01"
rets = provider.get_ohlcv(ticker, start, end)["Close"].pct_change().dropna()
result = hurst_exponent(rets)

inp = BacktestInput(symbol=ticker, start_date=start, end_date=end,
                    strategy_type="", parameters={}, initial_capital=10_000)

if result["regime"] == "trending":
    inp.strategy_type = "sma_crossover"
    inp.parameters = {"fast_period": 20, "slow_period": 50}
    bt = run_sma_backtest(inp)
elif result["regime"] == "mean_reverting":
    inp.strategy_type = "rsi_mean_reversion"
    inp.parameters = {"period": 14, "oversold": 30, "overbought": 70}
    bt = run_rsi_backtest(inp)
else:
    print("Random walk regime — no edge from technical signals")
    bt = None

if bt:
    print(f"Regime: {result['regime']}  →  Sharpe: {bt.sharpe_ratio:.2f}")
```

The `run_regime_adaptive_backtest` agent tool automates this entire flow — it calls `rolling_hurst` internally to select the best strategy for the current regime and runs a parameter grid. With the C++ extension active, that call returns in seconds rather than minutes.

---

### Parameters

#### `hurst_exponent`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `series` | `pd.Series` | required | Return series (not price levels) |
| `method` | `str` | `"dfa"` | `"dfa"` or `"rs"` |
| `min_window` | `int` | `10` | Smallest sub-window for the scaling analysis |
| `max_window` | `int` | `None` | Largest sub-window. `None` = auto (`n//4` for DFA, `n//2` for R/S). |

#### `rolling_hurst`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `series` | `pd.Series` | required | Return series (not price levels) |
| `window` | `int` | `200` | Lookback in bars. Minimum ~100 for reliable estimates. |
| `step` | `int` | `1` | Compute every N bars; intermediate positions are `NaN`. With C++ active this is purely a resolution choice — the C++ pass always runs in O(n) regardless. Without C++, `step > 1` reduces calls proportionally. |
| `method` | `str` | `"dfa"` | Passed to the underlying `hurst_exponent` call |
| `min_window` | `int` | `10` | Passed to the underlying `hurst_exponent` call |

#### Return values

`rolling_hurst` returns a `pd.Series` indexed like the input series (after dropping NaN). The first `window - 1` rows are `NaN`. Skipped bars (when `step > 1`) are also `NaN`.

---

### Implementation notes

**Python fallback** — `_dfa` and `_rs` are pure NumPy/Python functions. They produce numerically identical results to the C++ implementation (verified in the test suite with `atol=1e-10`). The fallback is always available and requires no additional dependencies beyond NumPy and pandas.

**C++ path** — when `_sqt_core` is importable, `hurst_exponent` calls `_cpp.hurst_dfa` or `_cpp.hurst_rs` directly with the raw `float64` array. `rolling_hurst` calls `_cpp.rolling_hurst`, which executes the entire sliding-window pass in one C++ function — no Python re-entry per bar. The result is copied back to a `pd.Series` with the original index.

**Numerical stability** — both implementations clip the final H estimate to `[0.0, 1.5]` and return `hurst=nan` when fewer than `min_window × 4` observations are available, when fewer than 3 valid window sizes survive, or when any fluctuation value is non-positive.
