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

## Cointegration & Pairs Spread Analysis

Two price series are **cointegrated** when a linear combination of them is stationary, even though each series individually follows a random walk. This is the statistical foundation of pairs trading.

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

*More tools coming: PCA on returns, Hurst exponent.*
