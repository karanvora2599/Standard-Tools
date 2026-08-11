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

## Portfolio Optimization

`standard_quant_tools.portfolio.optimize` — unlike everything above, which
*scores* weights you already chose, this module *produces* weights. Three
families:

### Markowitz Mean-Variance

```python
from standard_quant_tools.portfolio import mean_variance_optimize

result = mean_variance_optimize(
    returns_df,
    objective="max_sharpe",   # or "min_volatility" / "target_return" / "target_volatility"
    risk_free_rate=0.0,
    allow_short=False,        # long-only by default
    max_weight=0.4,           # optional per-asset cap
)
print(result["weights"])              # {"AAPL": 0.4, "MSFT": 0.35, ...}
print(result["expected_volatility"])  # annualized
print(result["converged"])
```

`allow_short=True` with `max_weight=None` (the fully unconstrained case) is solved in **closed form** via the standard two-fund efficient-frontier parametrization (Merton 1972) — numpy only, no solver dependency, and `converged` is always `True`. Any other combination (`allow_short=False` and/or a `max_weight` cap) requires **scipy** (SLSQP) and reports the solver's own success flag as `converged` — a request that's actually infeasible (e.g. a `target_return` no long-only portfolio can reach) comes back with `converged=False` rather than a silently wrong answer.

`objective="target_return"`/`"target_volatility"` need the matching `target_return`/`target_volatility` argument (annualized). A `target_volatility` below the global minimum-variance portfolio's own volatility is infeasible and raises `ValidationError` immediately.

### Risk Parity

```python
from standard_quant_tools.portfolio import risk_parity_weights

cov = (returns_df.cov() * 252).to_numpy()
result = risk_parity_weights(cov)              # equal risk contribution
# or: risk_parity_weights(cov, risk_budget=np.array([0.5, 0.3, 0.2]))

print(result["weights"])              # np.ndarray
print(result["risk_contributions"])   # fractional, sums to 1
print(result["converged"])
```

Solved via a damped multiplicative fixed-point iteration — a **documented heuristic**, not a globally-convergence-proven algorithm like the mean-variance closed form. It converges reliably in practice for well-conditioned covariance matrices (verified in `tests/portfolio/test_portfolio_optimize.py`: a diagonal covariance converges exactly to the closed-form inverse-volatility weights), but `converged` reflects whether the iteration actually reached its tolerance within `max_iterations`, not an assumption — check it.

### Black-Litterman

```python
from standard_quant_tools.portfolio import black_litterman, build_bl_views

cov = (returns_df.cov() * 252).to_numpy()
market_weights = np.array([0.4, 0.35, 0.25])

# Absolute view: "AAPL will return 15%/yr"; relative views (long/short
# coefficients in "assets") work the same way.
P, Q, omega = build_bl_views(
    tickers=["AAPL", "MSFT", "GOOGL"],
    views=[{"assets": {"AAPL": 1.0}, "view_return": 0.15, "confidence": 0.7}],
    cov_matrix=cov,
)
result = black_litterman(cov, market_weights, P, Q, risk_aversion=2.5, tau=0.05, omega=omega)

print(result["posterior_returns"])   # blended prior + views
print(result["implied_weights"])     # sums to 1
```

The market-equilibrium prior (`pi = risk_aversion * cov @ market_weights`) is blended with your views via the standard He & Litterman (1999) formula. `confidence` in `build_bl_views` (default `1.0`, the standard He-Litterman uncertainty) is a **documented simplification** of Idzorek's (2005) full confidence-scaling method, not a reimplementation of it — lower values widen that view's uncertainty proportionally, letting it move the posterior less.

### Via Agent Tool

```python
from standard_quant_tools.agent.tools import run_portfolio_optimization
from standard_quant_tools.agent.models import PortfolioOptimizationInput

result = run_portfolio_optimization(PortfolioOptimizationInput(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01", end_date="2024-01-01",
    method="risk_parity",
))
print(result.weights, result.risk_contributions)
```

`method` selects among all six: `"max_sharpe"`, `"min_volatility"`, `"target_return"`, `"target_volatility"`, `"risk_parity"`, `"black_litterman"` — see [09_advanced_agent_tools.md](09_advanced_agent_tools.md) and [07_agent_tools.md](07_agent_tools.md) for the full input/output model reference, including `black_litterman`'s `views`/`market_weights` fields.

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
