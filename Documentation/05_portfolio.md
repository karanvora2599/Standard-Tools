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

`result["warnings"]` carries the optimizer's own caveats — currently the small-sample covariance warning, and empty when the window is long enough.

#### What the optimizer refuses to answer

Three conditions produce a `ValidationError` rather than a number, because in each case the number would look authoritative and be meaningless.

**You need more observations than assets.** A sample covariance built from *n* observations has rank at most *n − 1*, so with observations ≤ assets it is singular *by construction*. That is not a numerical nuisance — it hands the optimizer a whole null space of directions with exactly zero in-sample variance. Measured on 5 observations of 6 assets, the SLSQP path reported `expected_volatility` of 1.19e-07 with `converged=True`, for weights whose actual out-of-sample volatility was **23.1%**. The closed-form path had always caught this (its matrix inverse fails); the constrained path inverts nothing and did not. Both now check the same condition before either solver runs, so they cannot disagree about whether an input is solvable. Perfect collinearity (a duplicated ticker, a share class tracking another exactly) is rejected on the same grounds even when there are plenty of rows.

**`max_sharpe` needs a risk-free rate below the minimum-variance return.** The closed-form tangency portfolio normalizes `Σ⁻¹(μ − rf·1)` by its own sum, `B − rf·A`. The resulting excess return is `(μ−rf)'Σ⁻¹(μ−rf)` divided by that sum — and the numerator is a quadratic form in a positive-definite Σ, so it is *always positive*. The sign is therefore entirely the denominator's, and once `rf` reaches the global minimum-variance portfolio's expected return (`B/A`) the normalization flips you onto the **inefficient** branch. An objective named `max_sharpe` then returned the *minimum*-Sharpe portfolio with `converged=True`: on μ=[0.10, 0.08], Σ=[[.04,.01],[.01,.05]], rf=0.20, Sharpe **−0.66**. The supremum genuinely is not attained in that regime, so it is reported. Bounded requests (`allow_short=False` and/or `max_weight` set) still solve — bounds make the feasible set compact — so the restriction is specific to the unconstrained closed form.

**A solver reporting success is not a valid answer.** `result.success` is the
solver's opinion of its own run, not a statement that the returned vector
satisfies the constraints it was given, so the weights are now checked
independently — sum to 1, inside their bounds, and actually meeting a requested
`target_return` / `target_volatility`. A long-only `target_return=99.0`
previously returned weights that looked entirely well-formed: `sum(w)=1.0000`
with an achieved return of **0.2443**. A violation sets `converged=False` and
names what was missed in `warnings`.

**Ill-conditioning is reported even at full rank.** The rank check catches an
exactly-degenerate covariance; it does not catch two assets that are merely
*almost* identical, which is the far more common real case — a share class
pair, an ETF and its largest holding. Measured on three assets where two differ
by 1e-9 of noise: rank 3/3, condition number **3.827e+14**, and a maximum
weight of **197,838× capital** reported as converged. Mean-variance inverts the
covariance, so that amplification lands directly in the weights. This is a
warning rather than an error — an ill-conditioned covariance is still the
caller's data — but nothing about the returned weights said so on their own.

**Returns must be finite.** `dropna()` removes NaN but not `±inf`, so an infinite return propagated into the covariance and came back as `{ticker: nan}` weights flagged `converged=True`. A zero or negative price feeding `pct_change` is the usual cause.

`max_weight` feasibility is checked whether or not shorting is allowed. Shorting lowers the per-asset *floor*, not the cap, so `sum(w) ≤ n × max_weight` either way and `sum(w) == 1` is still unreachable when `n × max_weight < 1` — that case used to return weights summing to 0.6 with only `converged=False` to indicate it.

### Input contracts for every optimizer

`risk_parity_weights` and `black_litterman` validate their matrices and scalars
before iterating. This matters more than usual here because the natural guard
is a comparison and **NaN satisfies no comparison**: a NaN covariance did not
trip risk parity's `portfolio_variance <= 0` degeneracy check, so it flowed
through every iteration and emerged as `{nan, nan}` weights with no error. The
same held for Black-Litterman, where one non-finite entry in any of
`cov_matrix`, `market_weights`, `P`, `Q`, `omega`, `tau` or `risk_aversion`
made the entire posterior NaN.

Covariances must also be **symmetric** — an asymmetric matrix was accepted and
silently used as though it were a covariance — and `build_bl_views` rejects
duplicate tickers, since the ticker→column map keeps the *last* index and a
view on a repeated name would silently attach to the wrong slot.

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

---

## Allocation without expected returns

`run_portfolio_optimization` maximizes a mean-variance objective, which is
the right tool when you genuinely have return forecasts and the wrong one
otherwise.

**Mean-variance is an error maximizer.** It puts weight where expected
return is high and covariance is low, which is exactly where estimation
error is most likely to have put them. With 50 assets and two years of data
you are estimating 1,275 covariance parameters from 500 observations, and
the optimizer finds the corner of that estimate where the noise happens to
align.

Three tools avoid the noisiest input entirely.

### `optimize_risk_parity`

Weights at which every asset contributes the **same amount of risk** — not
the same weight. An equally weighted portfolio of a bond fund and a biotech
stock is a biotech portfolio; the equity contributes almost all the
variance.

It uses no expected returns at all, which is the reason to prefer it in most
real situations: the standard error on a mean return estimated from two
years of daily data is roughly the size of the estimate, and mean-variance
is maximally sensitive to precisely that input.

Solved by cyclical coordinate descent, which needs no matrix inversion.
Convergence is **reported** — a solution that did not converge comes back
with `converged: false` rather than silently, because the iterate at that
point is not a risk parity portfolio and using it as one is worse than not
having it.

Two closed-form cases pin the solver: with a diagonal covariance matrix the
answer is exactly inverse-volatility, and the risk shares equalize to within
4.5e-12 in nine iterations on a correlated one.

`budget` allows an unequal **risk** budget — `[0.5, 0.3, 0.2]` gives the
first asset half the portfolio's risk. Most real mandates are written that
way rather than as equal contributions.

**What it assumes:** equal risk contribution implicitly bets that Sharpe
ratios are similar across assets. Where they are not, it over-weights the
low-Sharpe ones.

### `optimize_hierarchical_risk_parity`

Lopez de Prado's HRP: allocation that **never inverts** the covariance
matrix.

Inversion is where an ill-conditioned estimate does its damage. The smallest
eigenvalue becomes the largest after inversion, so the direction the data
says least about becomes the one the portfolio bets most on — and with 50
assets on 500 observations that smallest eigenvalue is essentially noise.

HRP clusters assets by correlation distance into a tree, orders them so
similar assets sit adjacent, then walks the tree splitting capital between
each pair of branches in inverse proportion to their variance. No inversion
happens anywhere.

**The trade-off is real and worth stating: HRP has no optimality property.**
It does not maximize anything. It is more robust out of sample than
mean-variance in most published comparisons and it is not the highest-Sharpe
portfolio under any model. It buys stability by giving up the claim to be
optimal.

### `optimize_max_diversification`

Maximizes the **diversification ratio** — the weighted average of the
assets' volatilities over the portfolio's own. It is 1.0 when everything is
perfectly correlated (combining the assets bought nothing) and grows as
correlations fall, so maximizing it maximizes the volatility that *cancels*.

**Not the same as minimum variance**, which is the usual confusion. Minimum
variance piles into the lowest-volatility assets because low volatility is
what it minimizes; on a set containing one very quiet asset it concentrates
there and is not diversified in any ordinary sense. Maximum diversification
normalizes by each asset's own volatility first, so an asset is rewarded for
being *uncorrelated* rather than for being quiet.

It does invert the correlation matrix, and reports the condition number for
exactly that reason.

## What the portfolio is actually exposed to

### `get_factor_exposure_budget`

Answers the failure that sinks more portfolios than any optimizer: **"I hold
40 names, so I am diversified."** Forty names with the same factor loading
is one position with extra transaction costs.

Risk is decomposed, not just exposure. A large loading on a low-variance
factor is not a large risk, and a small loading on a volatile one can be.
With `factor_covariance` supplied, the result reports each factor's share of
total portfolio **variance** — the number that answers "what am I actually
taking risk on". Without it, only exposures can be reported, and the result
says so rather than implying they are the risk.

The **residual** matters as much as the factors. A portfolio whose variance
is 90% explained by three factors is a factor bet; one where the factors
explain 20% is a stock-picking portfolio, and its risk lives somewhere this
decomposition cannot see.

### `analyze_concentration`

Turns "how concentrated is this" into numbers with known interpretations.
**Effective N** — the inverse Herfindahl — is the count of equally weighted
positions that would give the same concentration. A 100-position portfolio
with an effective N of 12 holds 100 names and has the concentration of 12,
and that is the number to quote.

Long-short books are measured on **gross** weights. Weights summing to zero
make a share-of-total meaningless, and squaring signed weights loses the
direction; gross is the economically relevant denominator, so a
market-neutral book has an effective N in the tens rather than an undefined
one.

### `get_marginal_risk_contribution`

For a portfolio you **already hold**: not "what should I hold" but "where is
my risk coming from, and what does adding to this position cost me". An
optimizer answers neither.

Marginal risk is the derivative of portfolio volatility with respect to the
weight — the cost of the next unit. Contribution is weight times marginal,
and these sum **exactly** to portfolio volatility, which is what makes it a
decomposition rather than an allocation of blame. The diagnostic is the
contribution share against the weight share: an asset at 5% of the portfolio
carrying 30% of the risk is the position to look at first.

**A negative marginal contribution is the interesting case.** It means
adding to that position *reduces* portfolio risk, which happens when the
asset is negatively correlated with the rest of the book. Those positions
are hedges whether or not they were intended as such.

## Risk that admits you cannot exit at the mark

### `get_liquidity_adjusted_var`

The standard number assumes instant exit, and that assumption does more work
than anyone acknowledges. A 1-day 95% VaR is a statement about a position
you could close today. A position that takes 15 days to liquidate at a sane
participation rate is exposed for 15 days, and its risk is larger by roughly
√15 — a factor of four.

Two parts, and they are different things:

- **Holding-period extension.** Risk scales with the square root of the
  liquidation horizon. This is the larger effect and the one usually missed.
- **Liquidation cost.** Getting out moves the price against you. This is an
  expected cost rather than a risk, and it is reported **separately** —
  adding a cost to a quantile produces a number that is neither.

The correlation argument matters enormously. At zero the position risks add
in quadrature, which understates a real portfolio. At 1.0 they add linearly,
which is the crisis case — and crisis is exactly when liquidation horizons
matter, so an honest stress uses a correlation well above the historical
average.

### `run_portfolio_scenarios`

What a portfolio does under **named** shocks rather than under a
distribution.

A 99% VaR is a quantile of a distribution fitted to history, and its central
weakness is that the event you care about is usually not in that history. A
named scenario — "rates +200bp, equities −20%, credit spreads double" —
makes the assumption explicit and arguable, which a quantile does not. The
two answer different questions and a risk process needs both.

Assets in the portfolio but absent from a scenario are treated as
**unchanged**, and the coverage is reported: a scenario touching three of
forty positions produces a loss that is a lower bound, which is worth
knowing before it is presented as a worst case.

## Related

- [22_microstructure.md](22_microstructure.md) — what it costs to get there
- [24_overfitting.md](24_overfitting.md) — whether the signal driving the weights is real


## Constructing weights, as a step you can stop at

`construct_weights_from_scores` turns alpha scores into portfolio weights
and stops. Rank, top/bottom, z-score or volatility-scaled construction,
optionally dollar-neutralised, published as an `sqt://weight_panel`.

This matters most between modeling and backtesting:

```
predictions ──> construct_weights_from_scores ──> LOOK AT THE WEIGHTS
                                               ──> only then simulate
```

A model's predictions become weights become a P&L, and when the P&L looks
wrong there is otherwise no way to tell whether the signal or the
construction did it. The construction rules always existed in
`backtest/sizing.py`; what was missing was a way to stop in the middle.

**These are TARGET weights, not a P&L.** What they earn depends on costs,
fills and rebalancing, which `run_portfolio_simulation` applies and this
tool deliberately does not.

`dollar_neutral=true` is applied AFTER the method, so it changes net
exposure and leaves the cross-sectional ordering alone — gross may then
differ from the leverage you asked for.

## Covariance before optimization

`estimate_covariance` produces the matrix an optimizer consumes, separately
from consuming it. A sample covariance on a short window is badly
conditioned, and feeding one straight into a mean-variance solver produces
confident weights built on noise — inspecting the estimate first is what
makes that visible rather than showing up as an implausible allocation.
