# Metrics

All metric functions accept `pd.Series`. Most return a single `float` — the exception is `drawdown_series`, which returns a full `pd.Series` (one drawdown value per bar). The `risk_metrics` functions (`sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `var_historical`, `var_parametric`, `cvar`, `information_ratio`, `treynor_ratio`) are decorated with `@validate_series`, which raises `ValidationError` on empty input. The `return_metrics` functions (`cumulative_return`, `cagr`, `annualized_volatility`) and `drawdown_series` are **not** decorated: `cumulative_return`/`cagr` return `0.0` on an empty series, while `annualized_volatility`/`drawdown_series` return `nan`/an empty `Series` rather than raising.

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

# risk_free_rate is the ANNUAL rate — both functions divide it by
# periods_per_year internally to get the per-period risk-free rate.
# Do NOT pre-divide it yourself (e.g. don't pass 0.05/252), or the
# risk-free adjustment gets divided by periods_per_year twice and
# becomes negligible.
rf_annual = 0.05

sr  = sharpe_ratio(returns, risk_free_rate=rf_annual)
srt = sortino_ratio(returns, risk_free_rate=rf_annual)

print(f"Sharpe  : {sr:.2f}")   # > 1.0 = good, > 2.0 = excellent
print(f"Sortino : {srt:.2f}")  # Sortino ≥ Sharpe when returns are right-skewed
```

**Formulas:**
- `sharpe_ratio` = `mean(returns - risk_free_rate/periods_per_year) / std(returns) * sqrt(periods_per_year)`. `std` is computed on the raw `returns` (equivalent to the std of the excess returns, since subtracting a constant doesn't change dispersion).

> These are the definitions the backtest engine uses too. `run_strategy`,
> `backtest_grid` and every Sharpe-reporting tool take a `risk_free_rate`
> and apply it exactly as above — in the C++ kernel as well as in Python,
> with parity asserted at several rates. See
> [04_backtesting.md](04_backtesting.md#the-risk-free-rate).
- `sortino_ratio` = `(mean(excess_returns) * periods_per_year) / downside_deviation`, where `excess_returns = returns - risk_free_rate/periods_per_year` and `downside_deviation = sqrt(mean(min(excess_returns, 0)**2)) * sqrt(periods_per_year)`. Note the denominator is the RMS of `min(excess_return, 0)` averaged over **all** N periods (zero contribution from winning bars), not just the subset of losing periods — the Sortino & Price (1994) convention. This gives a larger, more conservative denominator than dividing by the count of negative-return bars only, which some other libraries do. Returns `inf` when downside deviation is zero or `nan`.

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

> **Confidence validation:** all three functions raise `ValidationError` unless `0.0 < confidence < 1.0`. Passing `confidence=1.5` or `confidence=-0.2` (or `0.0`/`1.0` themselves) raises rather than silently producing a nonsensical result.

> **`var_parametric` without scipy:** when scipy is not installed, `var_parametric` uses a precomputed z-table covering exactly four confidence levels — `0.90`, `0.95`, `0.99`, `0.999`. Calling it with any other confidence level (e.g. `0.975`) and no scipy available raises `ValidationError` rather than silently substituting the 95% z-score. Install scipy to support arbitrary confidence levels.

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

> **Index alignment in `treynor_ratio`** — both the beta denominator and the excess-return numerator are computed on `returns.loc[common_idx]`, where `common_idx = returns.index.intersection(benchmark_returns.index)`. `beta` comes from `calculate_beta` on that same aligned slice, so the numerator and denominator always cover the identical date range, even when `returns` and `benchmark_returns` don't already share an identical index. `information_ratio` uses the same common-index-first approach for its active returns.

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
