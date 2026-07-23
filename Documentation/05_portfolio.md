# Portfolio Analysis

The portfolio module handles multi-asset analysis using NumPy matrix operations (covariance via BLAS, portfolio returns via `@` matrix multiply). Data fetching is async-first.

---

## Fetching Multi-Asset Returns

```python
from standard_quant_tools.portfolio import fetch_returns_sync

# Concurrent fetch: all tickers in ~1 network round-trip
returns_df = fetch_returns_sync(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    start_date="2023-01-01",
    end_date="2024-01-01",
)
print(returns_df.shape)   # (252, 5) — 252 trading days, 5 tickers
print(returns_df.head())
```

---

## Portfolio Returns

```python
from standard_quant_tools.portfolio import build_portfolio

weights = [0.30, 0.25, 0.20, 0.15, 0.10]  # must sum to 1.0
port_returns = build_portfolio(returns_df, weights)

print(f"Mean daily return : {port_returns.mean():.4f}")
print(f"Annual return     : {port_returns.mean() * 252:.2%}")
```

---

## Full Portfolio Metrics

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.portfolio import portfolio_metrics

# Fetch benchmark for Information Ratio
provider = DataFactory.get_provider()
spy_df = provider.get_ohlcv("SPY", "2023-01-01", "2024-01-01")
bench_ret = spy_df['Close'].pct_change().dropna()

metrics = portfolio_metrics(
    returns_df,
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],
    benchmark_returns=bench_ret,
)

print(f"Annualized Return     : {metrics['annualized_return']:.2%}")
print(f"Annualized Volatility : {metrics['annualized_volatility']:.2%}")
print(f"Sharpe Ratio          : {metrics['sharpe_ratio']:.2f}")
print(f"Sortino Ratio         : {metrics['sortino_ratio']:.2f}")
print(f"Max Drawdown          : {metrics['max_drawdown']:.2%}")
print(f"Calmar Ratio          : {metrics['calmar_ratio']:.2f}")
print(f"VaR (95%)             : {metrics['var_95']:.4f}")
print(f"CVaR (95%)            : {metrics['cvar_95']:.4f}")
print(f"Information Ratio     : {metrics['information_ratio']:.2f}")
```

---

## Correlation Matrix

```python
from standard_quant_tools.portfolio import correlation_matrix

corr = correlation_matrix(returns_df)
print(corr.round(2))

# Find the lowest-correlated pair (best diversification)
import numpy as np
mask = np.triu(np.ones_like(corr), k=1).astype(bool)
flat = corr.where(mask).stack()
best_pair = flat.idxmin()
print(f"Best diversification pair: {best_pair} (corr = {flat.min():.2f})")
```

---

## Equal-Weight vs Optimized Weights

```python
import numpy as np
from standard_quant_tools.portfolio import portfolio_metrics, build_portfolio

n = returns_df.shape[1]

# Equal weight
eq_metrics = portfolio_metrics(returns_df, [1/n] * n)
print(f"Equal weight Sharpe : {eq_metrics['sharpe_ratio']:.2f}")

# Max-Sharpe approximation: weight by inverse volatility
inv_vol = 1 / returns_df.std()
iv_weights = (inv_vol / inv_vol.sum()).tolist()
iv_metrics = portfolio_metrics(returns_df, iv_weights)
print(f"Inv-vol Sharpe      : {iv_metrics['sharpe_ratio']:.2f}")
```

---

## Via Agent Tool

```python
from standard_quant_tools.agent.tools import get_portfolio_analysis
from standard_quant_tools.agent.models import PortfolioInput

result = get_portfolio_analysis(PortfolioInput(
    tickers=["AAPL", "MSFT", "GOOGL"],
    weights=[0.4, 0.35, 0.25],
    start_date="2023-01-01",
    end_date="2024-01-01",
    benchmark="SPY",
))

# result is a Pydantic model — serialize directly for LLM
import json
print(json.dumps(result.model_dump(), indent=2))
```

---

## Async Fetching (Advanced)

For time-sensitive applications, call the async version directly:

```python
import asyncio
from standard_quant_tools.portfolio import fetch_returns_async

async def main():
    returns_df = await fetch_returns_async(
        ["AAPL", "MSFT", "GOOGL", "TSLA"],
        start_date="2023-01-01",
        end_date="2024-01-01",
    )
    return returns_df

returns_df = asyncio.run(main())
```
