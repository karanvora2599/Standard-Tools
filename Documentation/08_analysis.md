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

*More tools coming: cointegration & pairs spread analysis, PCA on returns, Hurst exponent.*
