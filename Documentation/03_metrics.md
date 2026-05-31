# Metrics

All metric functions accept `pd.Series` and return a single `float`. They are decorated with `@validate_series` which raises `ValidationError` on empty input.

---

## Return Metrics

```python
from standard_quant_tools.metrics import cumulative_return, cagr, annualized_volatility

returns = df['Close'].pct_change().dropna()
equity  = (1 + returns).cumprod() * 10_000

total  = cumulative_return(equity)   # e.g. 0.42 = +42%
annual = cagr(equity)                # e.g. 0.18 = 18% per year
vol    = annualized_volatility(returns)  # e.g. 0.22 = 22% annualized

print(f"Total Return : {total:.1%}")
print(f"CAGR         : {annual:.1%}")
print(f"Annual Vol   : {vol:.1%}")
```

---

## Sharpe & Sortino Ratios

```python
from standard_quant_tools.metrics import sharpe_ratio, sortino_ratio

# Risk-free rate: 5% annual → 0.05/252 per day
rf_daily = 0.05 / 252

sr  = sharpe_ratio(returns, risk_free_rate=rf_daily)
srt = sortino_ratio(returns, risk_free_rate=rf_daily)

print(f"Sharpe  : {sr:.2f}")   # > 1.0 = good, > 2.0 = excellent
print(f"Sortino : {srt:.2f}")  # Sortino ≥ Sharpe when returns are right-skewed
```

**Sortino vs Sharpe:** Sortino only penalizes downside deviation, making it more appropriate for strategies with asymmetric returns.

---

## Drawdown Metrics

```python
from standard_quant_tools.metrics import max_drawdown, calmar_ratio, drawdown_series

mdd = max_drawdown(equity)        # e.g. -0.23 = 23% peak-to-trough drop
cal = calmar_ratio(equity)        # CAGR / |MDD| — higher = better recovery vs risk
dd  = drawdown_series(equity)     # full time series of drawdown for plotting

print(f"Max Drawdown : {mdd:.1%}")
print(f"Calmar Ratio : {cal:.2f}")

# Find the worst drawdown period
worst_start = equity[dd == mdd].index[0]
print(f"Worst drawdown started: {worst_start.date()}")
```

---

## Value at Risk & CVaR

Both metrics express the **daily loss** at a given confidence level.

```python
from standard_quant_tools.metrics import var_historical, var_parametric, cvar

var95  = var_historical(returns, confidence=0.95)   # no normality assumption
var95p = var_parametric(returns, confidence=0.95)   # assumes Gaussian distribution
cvar95 = cvar(returns, confidence=0.95)             # expected loss beyond VaR

print(f"Historical VaR(95%) : {var95:.4f}  ({var95*100:.2f}% of portfolio)")
print(f"Parametric VaR(95%) : {var95p:.4f}")
print(f"CVaR/ES(95%)        : {cvar95:.4f}")  # always >= VaR
```

**Use historical VaR** unless you have a specific reason to assume normality — most financial return distributions have fat tails.

> **Performance:** `cvar` computes the quantile threshold and tail mean in a single NumPy pass (~1.9× faster than a two-pass approach that calls `var_historical` first, then filters). This matters when computing CVaR across many assets or rolling windows.

---

## Benchmark-Relative Metrics

```python
from standard_quant_tools.metrics import information_ratio, treynor_ratio

# Fetch benchmark
spy_df = provider.get_ohlcv("SPY", "2023-01-01", "2024-01-01")
bench  = spy_df['Close'].pct_change().dropna()

ir  = information_ratio(returns, bench)    # active return / tracking error
tr  = treynor_ratio(returns, bench)        # excess return / beta

print(f"Information Ratio : {ir:.2f}")   # > 0.5 = strong active management
print(f"Treynor Ratio     : {tr:.4f}")
```

---

## Complete Strategy Evaluation

```python
import pandas as pd
import numpy as np
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.metrics import (
    cumulative_return, cagr, annualized_volatility,
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    var_historical, cvar, information_ratio,
)

provider = DataFactory.get_provider()
asset_df = provider.get_ohlcv("NVDA", "2022-01-01", "2024-01-01")
bench_df = provider.get_ohlcv("SPY",  "2022-01-01", "2024-01-01")

asset_ret = asset_df['Close'].pct_change().dropna()
bench_ret = bench_df['Close'].pct_change().dropna()
equity    = (1 + asset_ret).cumprod() * 10_000

report = {
    "Total Return"      : f"{cumulative_return(equity):.1%}",
    "CAGR"              : f"{cagr(equity):.1%}",
    "Annual Volatility" : f"{annualized_volatility(asset_ret):.1%}",
    "Sharpe Ratio"      : f"{sharpe_ratio(asset_ret):.2f}",
    "Sortino Ratio"     : f"{sortino_ratio(asset_ret):.2f}",
    "Max Drawdown"      : f"{max_drawdown(equity):.1%}",
    "Calmar Ratio"      : f"{calmar_ratio(equity):.2f}",
    "VaR (95%)"         : f"{var_historical(asset_ret, 0.95):.4f}",
    "CVaR (95%)"        : f"{cvar(asset_ret, 0.95):.4f}",
    "Information Ratio" : f"{information_ratio(asset_ret, bench_ret):.2f}",
}
for k, v in report.items():
    print(f"{k:<22} {v}")
```
