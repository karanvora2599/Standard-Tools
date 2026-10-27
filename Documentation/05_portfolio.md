<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
  <img src="assets/header-light.svg" alt="Standard Quant Tools documentation" width="100%">
</picture>

<sub>[Standard Quant Tools](../README.md) · [Documentation](README.md)</sub>

# <img src="assets/chip-05.svg" height="28" alt=""> Portfolio Analysis

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

**A missing return is refused, not passed through.** A `NaN` in
`returns_df` used to go straight into the weighted sum, and the three
calculations downstream each read the resulting `NaN` their own way — on a
live panel, 20 missing days added **+290 bp of CAGR** and +0.14 of Sharpe.
`build_portfolio` now refuses by default, naming how many returns are
missing and in which columns, and takes a policy when you have decided what
a hole means:

```python
build_portfolio(returns_df, weights, missing="zero")  # the asset earned nothing that day; its weight sat in cash
build_portfolio(returns_df, weights, missing="drop")  # remove every date any asset is missing
```

Neither is the default because neither is neutral: `"zero"` keeps the
calendar and understates variance, `"drop"` keeps the arithmetic and
shortens the history.

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

`allow_short=True` with `max_weight=None` (the fully unconstrained case) is solved in **closed form** via the standard two-fund efficient-frontier parametrization (Merton 1972) — numpy only, no solver dependency, and `converged` is always `True`. Every other combination (`allow_short=False` and/or a `max_weight` cap) requires **scipy**, except `target_volatility` (below), and how it is solved depends on the objective:

- **`min_volatility`, `target_return`, and long-only `max_sharpe` with no binding cap** are convex quadratic programmes; `max_sharpe` becomes one after y = w / ((μ − rf)'w): minimize y'Σy subject to (μ − rf)'y = 1 and y ≥ 0, then w = y / 1'y. They are solved **exactly** by a primal-dual active-set method. Each pass fixes the weights it judges to be at a bound and solves the KKT system on the rest with one linear solve; when a pass changes nothing, the KKT conditions hold to rounding, which for a convex problem makes the answer the optimum. At 235 assets this takes 2–6 ms, where SLSQP took 0.9–14 s and stopped up to 1.6e-5 short of the optimum in the weights (on 235 zero-drift noise series, `max_sharpe` ran 9 s and ended with status 8, 2e-9 from it). Measured on 240 random problems, every answer certified, and once SLSQP's own constraint residuals are charged at the shadow prices no objective got worse by more than 4e-15 relative.
- **Capped or shorting `max_sharpe`** runs SLSQP, then is solved exactly on SLSQP's active set (150 of 150 random problems certified).
- **`target_volatility`** is solved on the frontier itself. It maximizes return subject to w'Σw = σ², a quadratic *equality*, which is not convex: SLSQP, which solved it until 2026-10-23, stopped at whichever local maximum it reached first, and over 60 random long-only problems of six assets 6 came back up to 16 percentage points of return below the frontier at the requested volatility, with `converged=True`. Now Markowitz's critical line algorithm (`portfolio.critical_line`, below) traces the corner portfolios of the frontier inside the bounds; volatility rises monotonically along them, so exactly one piece between two adjacent corners holds the target, and on that piece the variance is a quadratic in the mix of the two corners whose root in [0, 1] is the answer. It is certified by the frontier's own KKT conditions at the interpolated risk tolerance λ: the minimum-variance portfolio at its own return under the bounds, which with λ ≥ 0 makes it the maximum-return portfolio at its volatility. Two assets at 25% and 20% volatility correlated −0.5, long-only at a 13% target, returned 5.20% and now return 7.98%. This path needs no scipy. The closed form (`allow_short=True`, `max_weight=None`) is unchanged.

A request that's actually infeasible (e.g. a `target_return` no long-only portfolio can reach) comes back with `converged=False` rather than a silently wrong answer, with the attainable range named in the warning.

`objective="target_return"`/`"target_volatility"` need the matching `target_return`/`target_volatility` argument (annualized). A `target_volatility` below the minimum-variance portfolio's own volatility is infeasible and raises `ValidationError` immediately. With bounds, so does one above the highest-return portfolio's volatility: portfolios that risky exist inside the bounds, but every one earns less than the highest-return portfolio does at less risk, so the answer would be a dominated one. With bounds, both refusals name the attainable range `[minimum volatility, highest-return portfolio's volatility]`; the closed form's names the minimum volatility.

`result["warnings"]` carries the optimizer's own caveats: the small-sample covariance warning, the ill-conditioning warning below, and anything the independent weight check found. `result["condition_number"]` is reported as a number at every level, not only above a threshold. It is computed on one BLAS thread — here, and for the portfolio optimization tool's risk parity and Black-Litterman alike — and so is the covariance it is the condition number of, so the same returns report the same `condition_number` on any machine with the same BLAS, where at 235 assets it used to change in the last bits with the number of threads.

`result["solver"]` holds:

- `method` (`"closed_form"`, `"active_set"`, `"critical_line"` for bounded `target_volatility`, or `"SLSQP"`) and `iterations` (for the active-set method, passes of one linear solve each; for the critical line, the number of corner portfolios; none on the closed-form path). The critical line's `message` names the two corners whose mix the answer is, their risk tolerances and the share of each.
- `status`, `message`, `objective`, `n_function_evals` and the equality `multipliers`.
- `certificate`: the KKT residuals of the returned weights, estimated from the weights alone — `stationarity`, `dual_infeasibility`, `equality_residual` and `bound_violation`, each relative to the size of its terms, and the `tolerance` (1e-12).
- `certified`: `True` when all four are within the tolerance; `False` for an SLSQP answer the exact solve could not certify; `None` where no certificate applies (the closed form, an infeasible target). For `"critical_line"` the four residuals are those of minimizing ½w'Σw − λμ'w under the budget and the bounds at the λ the message names, with λ ≥ 0 required as well.
- `fallback`: why the fast method's cold start did not produce the answer, when it did not.

The agent tool's `SolverReport` carries `certificate`, `certified` and `fallback` too.

**When the fast method fails.** It can revisit a guess, or guess too few free weights for the equality constraints. Measured, that happened on capped or shorting `target_return` near the top of the attainable range (112 of 3,600 random problems), and never on `min_volatility` or long-only `max_sharpe`. The chain is then:

1. The primal active-set method from a feasible point: one bound per pass, no cycling, no singular systems. It solved all 112, in a median 0.7 ms.
2. SLSQP as before, then the fast method started from its answer.
3. SLSQP's own answer, with `certified=False` and a warning carrying its residuals.

Every exact stage ends with the same KKT solve on its final active set, so stages reaching the same active set give the same bits. The exact solve and its certificate run on one BLAS thread, so the same input gives the same bits on any core count. So do the covariance product and the closed form's inverse (`allow_short=True` with no `max_weight`), whose weights moved by up to 3e-13 relative between one and two threads before 2026-10-04.

**A stopped solver says what happened.** `warnings` names SLSQP's status and message and the residual against each requested constraint (sum to 1, bounds, target), plus the KKT residuals where they apply. Status 8 is a line search that could not improve at the requested precision; measured, its weights met sum-to-1 to 2e-11 and sat within 2e-9 of the optimum — it does not mean the constraints are infeasible. It used to be logged as weights that "may violate the sum-to-1 constraint", and the agent tool added "constraints may be infeasible"; neither was true.

#### What the optimizer refuses to answer

Three conditions produce a `ValidationError` rather than a number, because in each case the number would look authoritative and be meaningless.

**You need more observations than assets.** A sample covariance built from *n* observations has rank at most *n − 1*, so with observations ≤ assets it is singular *by construction*. That is not a numerical nuisance — it hands the optimizer a whole null space of directions with exactly zero in-sample variance. Measured on 5 observations of 6 assets, the SLSQP path reported `expected_volatility` of 1.19e-07 with `converged=True`, for weights whose actual out-of-sample volatility was **23.1%**. The closed-form path had always caught this (its matrix inverse fails); the constrained path inverts nothing and did not. Both now check the same condition before either solver runs, so they cannot disagree about whether an input is solvable. Perfect collinearity (a duplicated ticker, a share class tracking another exactly) is rejected on the same grounds even when there are plenty of rows.

**`max_sharpe` needs a risk-free rate below the minimum-variance return.** The closed-form tangency portfolio normalizes `Σ⁻¹(μ − rf·1)` by its own sum, `B − rf·A`. The resulting excess return is `(μ−rf)'Σ⁻¹(μ−rf)` divided by that sum — and the numerator is a quadratic form in a positive-definite Σ, so it is *always positive*. The sign is therefore entirely the denominator's, and once `rf` reaches the global minimum-variance portfolio's expected return (`B/A`) the normalization flips you onto the **inefficient** branch. An objective named `max_sharpe` then returned the *minimum*-Sharpe portfolio with `converged=True`: on μ=[0.10, 0.08], Σ=[[.04,.01],[.01,.05]], rf=0.20, Sharpe **−0.66**. The supremum genuinely is not attained in that regime, so it is reported.

**Bounded requests are refused when no portfolio inside the bounds beats the rate.** Bounds (`allow_short=False` and/or `max_weight` set) make the feasible set compact, so a maximum always *exists* — but when every attainable excess return is negative, the maximum of a negative Sharpe is the portfolio that divides the shortfall by the most volatility. Long-only with rf=10% over assets returning 4%/10% vol, 3%/30% and 5%/20% returned 100% in the 3% asset, dominated by both others, with `converged=True` and no warning; at `max_weight=5` the same regime returned a 5x-levered Sharpe of −1.06. The bounded path now computes the highest expected return any portfolio inside the bounds can earn (a linear programme, solved exactly) and refuses when the rate is at or above it, naming that return. When a positive-excess portfolio does exist but the solver ends in the negative region from its equal-weight start, it restarts from the highest-return portfolio; if it still ends there, `converged` is false and a warning says why.

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

Covariances must also be **symmetric**, under one rule for the whole library:
the numeric contract's `require_finite_covariance`, which compares the matrix
against its own transpose at `rtol=1e-9`, `atol=1e-12` and names the largest
`|A − A'|` it found. `portfolio/construction.py` used to carry its own copy of
that check at `rtol=1e-8` — two doors onto one rule, agreeing by inspection
rather than by construction, and not quite agreeing: the local copy took an
asymmetry ten times larger and let an infinity through to the eigenvalue code.
It calls the contract now. Before either existed, an asymmetric matrix was
accepted and silently used as though it were a covariance. `build_bl_views`
rejects
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

**There is one risk-parity solver, and this is not where it lives.** This
function validates and delegates to `portfolio/construction.py`'s cyclical
coordinate descent, which is what `optimize_risk_parity` has always called.
There used to be two, both reachable from the portfolio runtime, targeting the
same fixed point (`w_i (Σw)_i == b_i (w'Σw)`) and agreeing to about 5e-11 —
when both converged. Over 300 sample covariances built the way callers build
them, the damped multiplicative fixed point that used to sit here failed on
8 of 300, worst risk-share error **0.94** (an asset carrying ninety-four
percentage points more risk than its target); the coordinate descent failed on
none, at error 0.0000. The failures clustered on negative correlations and
near-singular sample matrices, not on volatility dispersion.

The validation stays here and is the stricter of the two: a `risk_budget` that
does not sum to 1 is refused rather than renormalized. `converged` still
reports whether the iteration reached its tolerance — check it, because the
iterate at that point is not a risk parity portfolio and using it as one is
worse than not having it.

### Black-Litterman

The tool returns the reason as well as the weights: the implied
equilibrium returns, the posterior returns and volatilities, and per view
how much of the stated spread the posterior absorbed (`absorbed_fraction`,
`None` when the view states the equilibrium), which is the one number that
says whether a view did anything. Every optimization also reports its
solver's iterations, status, objective and multipliers (`solver`, with
`method="closed_form"` on the unconstrained path) and the covariance
condition number as a number, not only as a sentence above a threshold.


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

**A net-short holding is refused, not flipped.** The weights are the
posterior's unconstrained optimal holding (δ Σ_post)⁻¹ μ_post rescaled to
sum to 1. That holding's posterior expected return is a positive quadratic
form, so when its sum is negative the rescaling flips every position and
the sign of the return with it: bearish views on all three assets of a
universe came back as a 100%-long book at a posterior Sharpe of −1.16 with
`converged=True`. `black_litterman` now raises `ValidationError` naming the
sum, the return the flip would have turned negative and the posterior
returns — as `max_sharpe` refuses a tangency portfolio with a negative
normalizing sum. State a bearish view as a relative one (one asset against
another), lower its confidence, or act on the posterior returns directly.
The portfolio optimization tool's `converged` for Black-Litterman, which
was the literal `True`, is now checked from the weights (finite, summing
to 1).

## The whole frontier, not one point of it

`get_efficient_frontier` returns the curve itself: the trade-off between
expected return and volatility across a universe, plus the two portfolios
on it that have names.

```python
from standard_quant_tools.agent.tools import get_efficient_frontier
from standard_quant_tools.agent.models import EfficientFrontierInput

curve = get_efficient_frontier(EfficientFrontierInput(
    tickers=["AAPL", "MSFT", "GOOGL", "XOM"],
    start_date="2022-01-01", end_date="2024-01-01",
    n_points=25,
    risk_free_rate=0.03,
))
print(curve.min_variance.weights)                 # needs no return forecast
print(curve.tangency.expected_return)             # None when no such portfolio exists
print([(p.expected_return, p.volatility) for p in curve.points])
```

`run_portfolio_optimization(method="target_return")` answers for ONE target
return. Tracing a curve through it is a tool call and an agent turn per
point, and it leaves the caller guessing which target returns the universe
can even support — the default span here runs from the global
minimum-variance portfolio's own return to the highest single-asset mean,
which is the stretch anybody is choosing among. Pass `return_range` to
choose the span deliberately.

**By default the weights are unbounded and sum to one.** That is not a
relaxation for convenience; it is the condition under which the Merton
frontier constants describe the frontier at all. A weight may be negative
(a short) or above 1 (levered by that short). Under
`allow_short=True, max_weight=None` `run_portfolio_optimization` takes the
same closed-form branch this tool does, so the two agree to the last place
either of them publishes. Every point is exact — none is an optimizer run —
so a fine curve costs no more turns than a coarse one, and `return_range`
may extend past the default span: it extrapolates the same closed form,
including onto the inefficient branch below the minimum-variance return.

**A long-only or capped frontier** (`allow_short=False`, or a `max_weight`;
with `allow_short=True` the cap is also a short floor at −`max_weight`, as
in `run_portfolio_optimization`) is a different curve with no closed form,
and the tool traces it exactly with Markowitz's critical line algorithm
(Markowitz 1956; Niedermayer & Niedermayer 2006; Bailey & López de Prado
2013), `method="critical_line"`. At a fixed set of assets held at a bound,
the solution of minimize ½w'Σw − λμ'w under the budget and the bounds is
linear in the risk tolerance λ, so as λ falls from infinity (the
highest-return portfolio) to zero (the minimum-variance one) the frontier
is a chain of straight pieces in the weights, joined at **corner
portfolios** where an asset reaches a bound or leaves one. `corners` lists
them, highest return first, each with its `risk_tolerance` (λ: twice it is
the frontier's slope of variance against return there), its return,
volatility and weights, and the `event` that happens there ("BBB leaves its
lower bound", "AAA reaches its upper bound", the minimum-variance end).
Every corner is certified by its KKT conditions at its own λ — the
minimum-variance portfolio at its return under the bounds — and
`certified` and `largest_kkt_residual` say so for the whole curve. Each of
the `n_points` points is the mix of the two corners whose returns bracket
it, exact to the rounding of that mix. The default span runs from the
minimum-variance portfolio's return to the highest attainable one; a
`return_range` reaching outside that span is refused with the span named,
since a bounded frontier has no points beyond it. `tangency` is the maximum
Sharpe ratio over the pieces, exact on each (the stationary point of a
ratio of a linear to the root of a quadratic is the root of a linear
equation), and null with a warning when no portfolio inside the bounds
earns more than the rate. Every point agrees with
`run_portfolio_optimization`'s exact `target_return` solve at the same
return and bounds, and a point's volatility asked of its
`target_volatility` returns that point: the optimizer now solves on the
same corners.

**Degenerate frontiers are handled or refused by name.** When expected
returns tie at the highest-return end, every split among the tied assets
earns the most, and the frontier starts at the split of least variance (a
small exact quadratic programme over them). When the highest-return
portfolio holds no asset strictly inside its bounds, the first two assets
to move are found together. Coincident events are taken one at a time at
the same λ, and a sequence of them that cycles is refused naming the
assets. A covariance that is singular on the free assets — a riskless
asset, or two collinear ones — is refused naming them, since every piece is
a solve on it; a frontier on which every expected return is equal is the
one minimum-variance portfolio, with a warning. The trace is sized for up
to 500 assets and refuses more. Its linear algebra runs on one BLAS thread,
so the same input gives the same bits on any machine. The library function
is `standard_quant_tools.portfolio.critical_line.critical_line(mu, cov,
lower, upper, budget=1.0, names=None)`, returning the corners and
`at_return`, `at_volatility` and `max_sharpe` for the points between them.

`tangency` is the maximum-Sharpe portfolio at `risk_free_rate`, and on the
closed form it is **null with a warning** when the rate sits at or above
the minimum-variance portfolio's own return: there is then no maximum-Sharpe
portfolio on the efficient branch, and normalizing the same algebra anyway
lands on the inefficient one and reads as an ordinary answer. The rate
itself is a decimal within ±10 (1,000%), bounded on magnitude and never on
sign; a larger magnitude is a unit error and is refused rather than
answered with a null tangency. `condition_number` is reported at every
level rather than only above the warning threshold — the closed form
inverts that covariance for every point it returns, and the critical line
solves on its principal blocks.

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

`max_weight` and `allow_short` apply to the four mean-variance methods
only, and are refused with `risk_parity` and `black_litterman`, naming the
method: neither reads them, and Black-Litterman with `max_weight=0.2` and
`allow_short=False` used to return weights of 1.787 and −1.120.
`allow_short` is refused whenever it is given with those methods, `False`
included — Black-Litterman's holding can short whatever it says.

## Constrained optimization on your own numbers

`optimize_portfolio` optimizes on expected returns and a covariance you
supply, under the limits a mandate is actually written in, and solves the
problem exactly. It takes no tickers and fetches nothing: the covariance is
rows with `assets`, the `matrix` map `estimate_covariance` returns, or its
`covariance_ref`, so the same arguments always give the same answer.

```python
from standard_quant_tools.portfolio.general_optimizer import optimize_portfolio

out = optimize_portfolio(
    covariance,                       # a named DataFrame, annualized
    expected_returns=mu,              # {asset: return}, same units
    objective="max_utility",
    risk_aversion=4.0,
    upper_bounds=0.05,                # every asset; or a map, asset -> cap
    groups={"tech": tech_names, "energy": energy_names},
    group_max={"tech": 0.25},
    factor_loadings=loadings,         # assets x factors
    exposure_min={"value": 0.0}, exposure_max={"value": 0.0},  # neutral
    max_volatility=0.12,
)
out["weights"], out["certified"], out["binding_constraints"]
```

| `objective` | maximizes or minimizes | needs |
|---|---|---|
| `min_variance` | w'Sw | — |
| `max_utility` | mu'w − (λ/2) w'Sw | `expected_returns` |
| `max_sharpe` | (mu − rf)'w / σ | `expected_returns`, a positive budget held in full, no cap |
| `min_tracking_error` | (w − b)'S(w − b) | `benchmark_weights` |
| `max_active_utility` | mu'(w − b) − (λ/2)(w − b)'S(w − b) | both |

The limits: per-asset bounds (the tool's `min_weight` / `max_weight` for
every asset, overridden per asset by `lower_bounds` / `upper_bounds`; the
library function takes a scalar, a map or a sequence in `lower_bounds` /
`upper_bounds`; equal bounds fix a weight, and the default is long-only
with no cap), a `budget` the weights sum to (`budget_type="equal"`) or stay
under (`"at_most"`, where the uninvested rest earns nothing), with 0 for a
dollar-neutral book; `groups` with `group_min` / `group_max` on each
group's summed weight; `factor_loadings` with `exposure_min` /
`exposure_max` on B'w, where equal values hold the exposure there (0 is
neutrality); and `max_volatility` and `max_tracking_error` (which needs
`benchmark_weights`), in the covariance's units.

**Exact, and certified from the weights.** Every linear limit is a row of
the library's active-set solver, which ends with one linear solve on the
final active set. The certificate is computed from the answer, not read
from the solver: stationarity, the multipliers' signs, every constraint's
violation and complementary slackness, each at most 1e-12. `certified` is
True only when all four hold; an answer that is not certified keeps its
weights and names the condition that failed.

**What each limit costs.** `binding_constraints` lists every constraint at
its bound with its `shadow_price`: the derivative of `objective_value`
(variance, utility, or Sharpe ratio) with respect to that bound. A cap of
5% on a name with price 0.31 says raising it to 6% adds about 0.0031 of
utility. At a vertex the limits fix by themselves, the price is one valid
multiplier, between the rates for raising and lowering the bound.

**Caps are found, not approximated.** A volatility or tracking-error cap is
quadratic, so its multiplier is found by a search that hits the cap to the
last bits. A cap the unconstrained answer already meets is reported as not
binding; one below the lowest risk the other limits allow is refused naming
that floor. Capping a utility on its own risk is the same as raising its
risk aversion, and `effective_risk_aversion` reports the value.

**Infeasible limits are refused by name.** A linear programme (scipy's
HiGHS, so the optimizer needs scipy) finds a feasible starting point or
proves there is none, and the refusal names the constraints in its proof:
caps of 10% on six names under a fully invested budget name the budget and
the six caps.

**Max Sharpe** is solved in y = w / τ, where it is convex; it needs some
portfolio with a positive excess return, and it is refused when the ratio
is approached only by levering a long-short book without limit (the
unbounded tangency on the inefficient branch) rather than normalized onto
the wrong portfolio. It takes no volatility or tracking-error cap: under a
binding volatility cap the highest Sharpe ratio is the highest return at
the cap, which `max_utility` with `max_volatility` finds exactly.

**A covariance that is not positive definite is refused, not repaired.**
Over the assets free to move, a singular or near-singular covariance —
fewer observations than assets, a duplicated or collinear asset — has a
direction with no variance, and the optimum along it is not unique. The
refusal names the smallest eigenvalue and the fixes: a shrunk estimate
(`estimate_covariance` with `method="ledoit_wolf"`), dropping the asset, or
more observations. Above a condition number of 1e10 the answer is still
given, with a warning.

The optimizer is sized for up to 500 assets and refuses a larger universe
by name; at 500 assets with sector and factor limits a solve takes about a
second.

---

## Rebalancing with trading costs

`optimize_portfolio` answers which portfolio to hold. Asked again a week
later with forecasts that moved a little, it answers with a portfolio that
moved a little in every name, and a book that follows it trades everything
every week. `optimize_rebalance` answers what a rebalance asks -- what to
trade from the weights held -- with the costs of trading in the objective:

    maximize  mu'w - (gamma/2) w'Sw - sum kappa_i |w_i - w0_i|
                                    - sum eta_i (w_i - w0_i)^2

`linear_cost` (kappa) is the cost per unit of weight traded, a decimal
(0.0015 is 15 bps: spread and commission); `quadratic_cost` (eta) prices the
square of the trade, as impact grows with size. Both are one number or one
per asset. The limits are `optimize_portfolio`'s -- bounds, a budget, groups,
factor exposures, read by the same code -- plus `max_turnover` (sum
|w - w0|), `max_gross` (sum |w|) and `max_volatility`.

```python
from standard_quant_tools.portfolio.rebalance_optimizer import optimize_rebalance

out = optimize_rebalance(
    covariance,                    # a named DataFrame, annualized
    expected_returns=mu,           # {asset: return}, same units
    current_weights=held,          # {asset: weight}; left out is 0
    risk_aversion=4.0,
    linear_cost=0.0015,            # 15 bps per unit traded
    upper_bounds=0.05,
    max_turnover=0.20,
)
out["trades"], out["no_trade"], out["turnover"], out["binding_constraints"]
```

**A name inside its no-trade band keeps its weight exactly.** Trading costs
make a band around each current weight inside which no trade earns its
cost; a weight there comes back as the weight held, to the bit, and is
listed in `no_trade`. The band widens with the turnover cap's price: when
the cap binds, a trade must earn its cost plus what that unit of turnover
is worth elsewhere. Every binding constraint carries its `shadow_price` in
utility net of costs -- the turnover cap's is what one more unit of
turnover would add -- and the KKT certificate is computed from the weights,
as `optimize_portfolio`'s is.

**What the costs held back.** The tool reports the frictionless target
beside the answer -- the same problem with no costs and no turnover cap,
which is `optimize_portfolio`'s `max_utility` -- with the turnover and cost
of trading to it, its utility after that cost, and the distance and the
tracking volatility between it and the answer. Without a turnover cap the
answer's utility is at least the target's after costs; with a cap that
forbids the target it can be less, and that difference is the cap's, not
the costs'.

**Exact, and fast enough to call at every rebalance date.** The solve works
on each weight's pieces between its breakpoints (its bounds, its current
weight, zero under a gross cap), where the costs are linear: a primal-dual
active-set pass, then the primal active-set method, each ending in one
linear solve on the final active set. At 200 assets with a binding turnover
cap it takes about 25 ms, at 500 about 0.1 s; a cold solve spends two
passes on each name it trades to a bound. In a walk-forward, pass the
previous date's answer as `start=`: its states are the first guess, and the
answer is the same. A book that breaks its limits (a weight drifted over its
cap) reports `required_turnover`, the least any answer must trade, and a
turnover cap below it is refused naming the turnover cap among the
constraints that conflict.

A model's predictions are sized through it at every rebalance date by
`PredictionTransformSpec(method="mean_variance")`, from the book the account
holds, with the model's own trailing IC turning scores into expected
returns; see
[15_modeling.md](15_modeling.md#sizing-by-the-optimizer-methodmean_variance).

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

**A covariance is accepted in either form this library writes it.** The
tools that take a covariance — `optimize_risk_parity`,
`optimize_max_diversification`, `get_marginal_risk_contribution`,
`optimize_portfolio` (above) and the optional sigmas of
`run_portfolio_scenarios` — take `covariance` as rows
parallel to `assets`, or as the map of asset → {asset: covariance} that
`estimate_covariance` returns as `matrix`. Pass that tool's `assets` and
`matrix` on unchanged. The map is read by **name** and laid out in `assets`
order; names that are not exactly `assets`, or a row that lacks a column,
are refused naming them (a missing covariance is not zero), and symmetry and
finiteness are checked by the numeric contract exactly as for rows. The map
used to be refused as `list_type`, and the rows had to be rebuilt by hand,
in an order the caller had to get right.

**Or as a reference.** The same five tools take `covariance_ref`, the
`sqt://covariance/...` reference `estimate_covariance` publishes when it is
given a `run_id`, in place of `covariance`. Exactly one of the rows, the map
and the reference is read; two are refused, and an inline matrix still needs
`assets`. A reference names its own assets, so `assets` is optional beside
it: given, it must name exactly the reference's assets (each missing or
extra name is listed) and fixes the order of the result. A reference of
another kind — a `returns_panel` handed over by mistake — is refused naming
both kinds. The answers are the inline matrix's to the bit; the matrix never
passes through the conversation, and its annualization and estimator travel
with it.

**A null number says why.** Every construction tool's result writes one
line into its `warnings` for each null number, naming the field and the
reason: a factor variance with no factor covariance, a `top_10_share` of
a book of four names, a scenario's `sigma_move` with no covariance to size
it, a liquidity-VaR `adjustment_multiple` of a zero position. A weight map
carries a number or a null, never a NaN.

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
way rather than as equal contributions. Each entry must be finite and
positive: a NaN passed the positivity check (it compares False against
every bound) and an infinity survived it, and both came back as NaN
weights. `tolerance` must be finite too — an infinite one is met by the
first sweep, which reported the inverse-volatility start as converged.

**What it assumes:** equal risk contribution implicitly bets that Sharpe
ratios are similar across assets. Where they are not, it over-weights the
low-Sharpe ones.

### `optimize_hierarchical_risk_parity`

Lopez de Prado's HRP: allocation that **never inverts** the covariance
matrix. Takes `periods_per_year`, which the library always did and the
tool now passes.

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

**The returns carry their dates.** Each series in `returns` may be a
date-keyed map {ISO date: return} or an undated list. Undated lists were the
only form, and two series of equal length from different calendars — one a
fortnight later, one with another market's holidays — were accepted as
aligned and correlated each day against some other day. Dated series are
checked: under the default `alignment="exact"` they must carry the same
dates, and a mismatch is refused with what each series lacks (the
equal-length case named as such); `alignment="intersect"` uses the dates
every series shares, and `alignment` on the result gives the shared span and,
per series, how many returns were supplied and dropped. The two forms cannot
be mixed. Undated lists still work — `alignment` is null and a warning says
their alignment could not be checked — but have no dates to intersect, so
they refuse `alignment="intersect"`.

**Or a published panel.** `returns_ref` takes a `returns_panel` reference in
place of `returns`; its columns are the assets. A blank cell is a date that
column has no return on, and `alignment` applies exactly as it does to dated
series: under `"exact"` a panel whose columns do not carry the same dates is
refused with what each lacks, under `"intersect"` the dates every column
shares are used and `alignment` on the result says what each column lost.
The columns are checked as return streams first, so a panel of prices is
refused as levels rather than allocated. On the same returns the weights are
the inline answer's, bit for bit.

**The weights do not depend on the order you listed the assets in.** The
clustering's tie-breaks follow column position, and forty permutations of
one universe moved single weights by up to **8.7 pp** — the same names,
different numbers, with nothing to say which permutation was the answer.
Columns are now sorted by name before anything reads them, so the weights
are a function of the universe and its returns alone.

**The correlation matrix is compiled, to the bit.** `frame.corr()` was
78–86% of the call. With the extension built, HRP takes its correlation
matrix from the `pearson_correlation` kernel — pandas' own per-pair
arithmetic, clipped to [-1, 1] exactly when the installed pandas clips — and
is 4.5× faster at 235 assets (341 → 77 ms on pandas 2.3, 295 → 63 ms on
pandas 3.0), with weights, cluster order and risk contributions identical to
the `frame.corr()` path. If the installed pandas answers a probe differently
from the kernel, `frame.corr()` computes the matrix itself.

**A zero-variance asset is refused by name.** HRP splits capital in inverse
proportion to variance, and it had no guard: an exactly constant column (a
cash sleeve, a halted name) divided by zero and every weight came back NaN —
nulls at the agent surface, with only the usual warnings — while a column
constant at a non-zero level (1bp a day) has a float-residue variance of
about 1e-40 and took **100%** of the portfolio. A column that is constant on
its own scale, or whose variance is at or below 1e-20 of the largest in the
universe (a volatility ratio of 1e-10), is now refused naming the asset. A
genuinely quiet asset — T-bills next to equities, around 1e-6 of the largest
variance — is not affected. `optimize_risk_parity`,
`optimize_max_diversification` and every other tool that takes a covariance
apply the same relative floor to its diagonal; they refused only an exact
zero, and gave the 1bp sleeve 100% with `converged=True`.

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

By default it is the closed form w ∝ Σ⁻¹σ, which may short; it inverts the
correlation matrix and reports the condition number for exactly that
reason. **With `long_only=True`** (and optionally a per-asset `max_weight`,
which needs `long_only=True` and is refused when `max_weight × n < 1`) the
weights are also non-negative and capped, and the maximum is solved exactly
and certified. Homogenised, it is the maximum-Sharpe problem with the
volatilities as the returns and a zero rate: minimize y'Σy subject to
σ'y = 1 and 0 ≤ yᵢ ≤ cap · 1'y, then w = y / 1'y. The set of assets at
zero and at the cap comes from the critical line (the maximum ratio is the
tangency portfolio of the frontier whose returns are the volatilities), and
the homogenised KKT system is then solved on that set exactly; started
cold, or from the unconstrained holding clipped to the bounds, the
homogenised solve stopped without an answer on 3 to 12 in 100 random
capped problems. The answer is certified against the ratio's own KKT
conditions (`certified`, `kkt_residual`, `method` — `"active_set"`, or
`"critical_line"` when the homogenised solve on that set does not certify —
and `solver_message`). Without a cap the result has the core property of
the maximum-diversification portfolio (Choueifaty & Coignard 2008): every
held asset's correlation with the portfolio is exactly 1 / DR, and every
asset left at zero has at least that.

**A holding with a negative sum is refused by the closed form, and answered
long-only.** The maximum is the direction Σ⁻¹σ, rescaled to sum to 1, and
that sum can be negative — when a low-volatility asset is strongly
correlated with a combination of the others and the holding shorts it
heavily. Dividing by it flips every position and turns the maximum ratio
√(σ'Σ⁻¹σ) into its negative: three assets at volatilities 3.17, 4.59 and
0.80 with correlations −0.474, 0.792 and 0.144 (condition number 204) came
back at a diversification ratio of −5.32. No fully-invested portfolio
attains the unconstrained maximum then — the ratio only approaches it as
the positions grow without bound — so the default request is refused,
naming the holding (scaled to a gross of 1), the ratio it would have
flipped, and `long_only=True` as the fix. A long-only maximum always
exists: on the same three assets it drops the third and holds the first two
at inverse-volatility weights (0.5910, 0.4090), at a ratio of
√(2 / (1 − 0.474)) = 1.9499.

### An indefinite covariance is repaired, and the repair is named

Nothing checked positive semi-definiteness between a caller's covariance
and the optimizers, and a pairwise estimate over a ragged panel —
`cov(min_periods=30)` on real prices with one short history — is not a
covariance of anything: its smallest eigenvalue was **−2.77e-03**, and
`optimize_max_diversification` returned a *negative* weighted average
volatility with zero warnings. Every tool that takes a covariance
(`optimize_risk_parity`, `optimize_max_diversification`,
`get_marginal_risk_contribution`, `get_factor_exposure_budget`,
`run_portfolio_scenarios`) now projects an indefinite matrix onto the
nearest positive semi-definite one by flooring its eigenvalues (Higham's
clipping, re-symmetrized) and warns with the eigenvalue that was floored;
the weights returned are for the repaired matrix. The warning is the thing
to act on: estimate the covariance on complete rows (`estimate_covariance`,
below) and the repair never runs. `optimize_portfolio` is the exception:
it refuses a covariance that is not positive definite on the assets free to
move, naming its smallest eigenvalue, because its optimum would then not be
unique.

Deciding that no repair is needed costs a Cholesky factorization, not an
eigendecomposition. A factorization that completes on a matrix of up to 670
assets proves the eigenvalue test would pass — its backward error is within
half the 1e-10 tolerance there — and the matrix is used untouched; only when
it fails, or the matrix is larger, do the eigenvalues decide, as before. The
answer is the same: on 1,500 matrices built across the threshold the
factorization never completed where the eigenvalues repaired, and every
output matched. The check takes 0.4 ms at 235 assets instead of 18–25 ms.
These factorizations, and `optimize_max_diversification`'s pseudo-inverse
and condition number, run on one BLAS thread, so their bits do not depend on
the machine's core count; against the earlier multi-threaded default they
moved once, in the last bits — weights by at most 2e-12 relative.

Each of them also needs every asset named once. The results are keyed by
name, so a repeated name collapsed two rows of the matrix into one key
while both were still used: `optimize_risk_parity` over `["A", "A"]`
answered `{"A": 0.4}`, a book whose weights summed to 0.4. A repeat in
`assets` is refused at the schema, and the library functions in
`portfolio.construction` refuse a covariance whose row or column labels
repeat — and `hierarchical_risk_parity` a repeated return column — by name,
so a direct caller meets the same rule.

## What the portfolio is actually exposed to

### `get_factor_exposure_budget`

Answers the failure that sinks more portfolios than any optimizer: **"I hold
40 names, so I am diversified."** Forty names with the same factor loading
is one position with extra transaction costs.

Risk is decomposed, not just exposure. A large loading on a low-variance
factor is not a large risk, and a small loading on a volatile one can be.
With `factor_covariance` supplied, the result reports the **factor**
variance x'Fx of the exposures x and each factor's share of it
(`factor_variance_shares`, summing to 1). Without it, only exposures can
be reported, and the result says so rather than implying they are the
risk.

**The specific risk matters as much as the factors, and it is not in the
factor variance.** A portfolio whose variance is 90% factor is a factor
bet; one where the factors carry 20% is a stock-picking portfolio. Shares
of the factor variance cannot tell the two apart — they sum to 1 either
way — and they used to be described as shares of portfolio variance: two
names with a beta of 1 to a market factor at 1% volatility and 40%
specific volatility each read "100% ... it is one bet" when the market
carries 0.12% of their variance. Pass `specific_variance` — each held
asset's idiosyncratic return variance, in the units of
`factor_covariance`, taken as uncorrelated across names — and the result
also reports the portfolio's `specific_variance` Σ w_i² s_i, the
`total_variance`, each factor's share of the **total**
(`total_variance_shares`) and the `specific_variance_share`, which
together sum to 1. "One bet" is said only of a factor that carries more
than 60% of the total. Without `specific_variance` those fields are null
and a warning says the factor shares are not the portfolio's total risk.
A name at zero weight may be left out of `specific_variance`; a held name
may not, and a negative or non-finite variance is refused.
`specific_variance` without `factor_covariance` is refused too: there is
no factor variance to add it to.

Weights and loadings must be finite, and every held asset needs a loading
on every factor: a gap is refused by name (`C/size`) rather than read as
zero, because absent is not zero — it used to make that factor's exposure
NaN for the whole book. Give `0.0` where an asset genuinely has no
loading, or narrow `factors` to the ones every asset has. `factors`, when
given, names at least one factor and each once.

`portfolio.construction.factor_exposure_budget`, the library function
behind the tool, keeps the same rules for a direct caller: a weight that
is not finite, a loading of a held asset that is missing or not finite
(named as `asset/factor`), an asset or factor that labels two rows or
columns of the loadings, a weight named twice, and an exposure or gross
exposure that overflows the float range are each refused as a
`ValidationError` by name. A gap in an asset that is not held never enters
the sum and is left alone.

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

Weights must be finite. An infinite weight made the gross exposure
infinite and the effective N a division by zero, and a NaN weight was
dropped without a word; both are refused at the schema now, and so is a
set of finite weights whose gross exposure overflows. The library's
`concentration_analysis` refuses the same inputs by name for a direct
caller, and a name listed twice, instead of dropping the NaN or dividing
by zero.

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

Weights must be finite. A NaN weight made the volatility and every row NaN,
which reached the agent as nulls with no reason, and an infinite one gave
NaN contributions under an infinite volatility; the schema and
`marginal_risk_contribution` both refuse it by name. A book whose variance
passes the float range from finite weights is refused too, rather than
reported with every marginal at 0.0.

### `decompose_factor_risk`

The same question as `get_marginal_risk_contribution` — where is the risk
in a book already held — asked of a **factor model** rather than of a
covariance matrix, so the answer can say how much of the risk is the
factors and how much the names' own. The model is the caller's: loadings
B (`factor_loadings`, asset → {factor: loading}), a factor covariance F
(`factor_covariance`, factor → {factor: covariance}, read by name) and
specific variances D (`specific_variance`, asset → variance), with
Σ = BFB′ + D (Grinold and Kahn, *Active Portfolio Management*, ch. 3).

For the book (`portfolio`) and, given `benchmark_weights`, for the active
weights w − b (`active`), the result reports:

- the volatility, the factor variance x′Fx of the exposures x = B′w, the
  specific variance Σ w_i² d_i, and each one's share of the **total**;
- `exposures` (and `benchmark_exposures`);
- `by_factor`: each factor's Euler contribution x_k (Fx)_k / σ, and the
  same number as **exposure × factor volatility × correlation** — x_k,
  σ_k = √F_kk and ρ_k = (Fx)_k / (σ_k σ), the factor's correlation with
  the book (Menchero and Davis 2011) — which separates how much of a factor
  is held from how much it moves from how much it moves *with* the book;
- `specific_contribution`, the names' own risk, V_s / σ;
- `by_asset`: each asset's marginal (Σw)_i / σ, its contribution, its
  share, and the contribution split into the part that comes through the
  factors and its own specific part;
- `factor_specific_covariance`, **zero by construction**: the model takes
  specific returns as uncorrelated with the factors and with each other,
  which is what makes variance = factor + specific. It is stated rather
  than measured; a book whose residuals do correlate (two share classes, a
  merger pair) breaks it, and nothing computed from B, F and D can see
  that.

The factor contributions plus the specific contribution sum to the
volatility, and so do the asset contributions, to 1e-12; the active side
does the same for `tracking_error`. Units are the inputs': F and D in
annual variance give annual volatility and tracking error, and nothing is
annualised.

**Specific variance is required.** Without D every name is riskless
beyond its factors, and the factor share is overstated without limit —
two names with a beta of 1 to a market at 1% volatility and 40% specific
volatility each are 100% market without it and 0.12% market with it.

Every name is checked against every other, and a disagreement is refused
by name: the covariance must be square under its factor names, symmetric,
finite and give every factor a variance; each asset needs a loading on
every factor (0.0 where it has none) and on no factor the covariance
lacks; `specific_variance` and `factor_loadings` must name the same
assets; and every asset in `weights` or `benchmark_weights` needs
loadings — a held name with none would be left out of the risk, which
understates it by exactly that position. An indefinite factor covariance
is repaired as the allocators' is, with the warning. A book equal to its
benchmark has a tracking error of exactly 0, and every number that divides
by it is null with the reason. `portfolio.factor_risk.decompose_factor_risk`
is the library function, and also takes the loadings and the covariance as
frames.

Planted: two names with betas 1.2 and 0.8 to one factor at variance 0.04
and specific variances 0.09 and 0.0625, the book 50/50 and the benchmark
30/70. The active exposure is 0.08, the factor TE² 0.000256, the specific
TE² 0.0061, the tracking error √0.006356 = 0.0797 and the factor 4.0% of
it.

The model can come from the data: `estimate_factor_model` in the
`research` runtime fits B by OLS on factor returns the caller supplies,
takes F as the factors' sample covariance and D as each asset's residual
variance over n − p, and returns them as `factor_loadings`,
`factor_covariance` and `specific_variance`, ready to pass here. Given a
`run_id` it also publishes Σ = BFB′ + D as a `covariance` reference, which
`optimize_portfolio` and the allocators take as `covariance_ref`. Series
built so their sample moments are this section's planted book give back
its tracking error of 0.0797 and factor share of 4.0%.

## Risk that admits you cannot exit at the mark

### `get_liquidity_adjusted_var`

The standard number assumes instant exit, and that assumption does more work
than anyone acknowledges. A 1-day 95% VaR is a statement about a position
you could close today. A position that takes 15 days to liquidate at a sane
participation rate is exposed for 15 days, and its risk is larger by roughly
√15 — a factor of four. The `impact_coefficient` that scales the
liquidation cost with the square root of participation is an input; it
moves that cost across three orders of magnitude, so the cost is a model
output, not a measurement.

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

It is one correlation between the assets' **returns**, shared by every pair,
and two things follow. First, it has a floor: an equicorrelation matrix over
*n* positions is a correlation matrix only for ρ ≥ −1/(n−1). Below that the
implied variance is negative and was clamped to zero — five $1m positions in
30%-vol names at ρ = −0.30 reported a VaR of exactly **$0.00**. A correlation
below the floor (−0.25 for five positions) is refused naming it; at the floor
itself the positions offset exactly and a warning says the zero is a
statement about the assumption. Second, a short position's P&L is the
negative of its asset's return, so the aggregate uses **signed** position
risks: {+$1m, −$1m} at ρ = +0.9 is a hedge carrying √0.2 of one leg's risk.
The rows used to be aggregated as magnitudes, which scored that hedge as a
long/long doubling (√3.8). The per-position rows still report magnitudes.

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

Weights and shocks must be finite. A NaN weight used to be dropped without
a word, so every scenario described a smaller book than the one given, and
a NaN shock made its scenario's return NaN, sorted anywhere among the
others. Both are refused by name — the shock naming its scenario — at the
schema and by `portfolio_scenarios`, and so is a scenario return past the
float range.

### `decompose_tail_risk`

Which positions the expected shortfall and the VaR come from — on
**scenarios**, not on a covariance, so a fat-tailed or option-bearing book
is decomposed in the scenarios it actually loses in. The input is a
scenario matrix inline (`scenarios`: asset → its value in each scenario,
scenario *i* at index *i*, exactly the weights' assets) or a published
`returns_panel` (`returns_ref`: its dates are the scenarios, aligned on the
dates the weighted columns share and reported in `streams`; columns
without a weight are not read). The portfolio's P&L in a scenario is
Σ w_i X_si — a return with returns and book fractions, currency with P&L
per unit and quantities (1.0 for a column already one position's P&L) —
summed asset by asset in the weights' order.

**VaR and ES are losses, positive**, as everywhere in the library: on the
same P&L they are `var_historical`'s and `cvar`'s own bits — the empirical
(1 − confidence) quantile with numpy's linear interpolation, and the mean
of the scenarios at or below it.

- **Component ES is exact** (Tasche 1999). The ES is a mean over a tail
  set that scaling the weights leaves unchanged, so it is homogeneous of
  degree one, and each asset's component is its weight times minus its
  own mean in the portfolio's tail. The components sum to the ES to
  rounding (`sum_of_es_components`), and `es_share` to 1.
- **Component VaR is smoothed** (Epperlein and Smillie 2006). VaR's
  marginal is the expectation of −X_i given that the portfolio sits exactly
  at its VaR — a single scenario in a sample. A Gaussian kernel weights
  every scenario by its distance from the VaR; the components then sum
  exactly to the kernel-smoothed VaR, reported as `var_kernel` beside the
  empirical `var` so the smoothing's bias is visible rather than rescaled
  away. `var_share` is a share of `var_kernel`.
- **The bandwidth** is Silverman's rule on the portfolio's P&L,
  0.9 min(sd, IQR/1.34) n^(−1/5) (the standard deviation alone when the
  IQR is zero), unless `bandwidth` is given in P&L units; it is reported
  with `bandwidth_rule` and `kernel_effective_scenarios`, (ΣK)² / ΣK², and
  a result resting on fewer than ten effective scenarios says so.
- **The tail count is always reported** (`n_tail_observations`), with the
  library's shared warning below ten: ES at 99% on 250 scenarios is the
  mean of three, and so is every component.
- A **negative component** is an asset that gains, on average, when the
  book is in its tail: a hedge, named in `n_hedges` and a warning.

Refused by name: weights and scenarios that name different assets, lists
of different lengths, a non-finite value, fewer than ten scenarios, a
portfolio P&L that is the same in every scenario, a weight naming a column
the panel lacks, a bandwidth that is not finite and positive, and both or
neither of `scenarios` and `returns_ref`. The tool's `confidence` is at
least 0.5 and below 1. The library function is
`portfolio.tail_risk.decompose_tail_risk`, which takes the matrix as a
frame and any confidence strictly between 0 and 1; `silverman_bandwidth`
is the rule.

Under a normal with zero mean the components converge, as the scenarios
grow, to w_i (Σw)_i / σ_p times z for VaR and φ(z) / (1 − c) for ES: at two
million scenarios the largest error is under 0.5% (ES) and 1% (VaR) of the
total.

## Related

- [22_microstructure.md](22_microstructure.md) — what it costs to get there
- [24_overfitting.md](24_overfitting.md) — whether the signal driving the weights is real


## Constructing weights, as a step you can stop at

`construct_weights_from_scores` turns alpha scores into portfolio weights
and stops. Rank, top/bottom, z-score or volatility-scaled construction,
optionally dollar-neutralised, published as an `sqt://weight_panel`.

This matters most between modeling and backtesting:

```
predictions ──> convert_reference(to_kind='score_panel', task=...)
            ──> construct_weights_from_scores ──> LOOK AT THE WEIGHTS
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

**Each reference must be the kind it names.** `scores_ref` is resolved as
an `sqt://score_panel` and `returns_ref` as an `sqt://returns_panel`, and
any other kind is refused naming both. Without the check, the
`weight_panel` this tool publishes was accepted back as scores and
transformed a second time — under `vol_scaled` the weights came out
divided by volatility twice, a plausible vector that was not the one
asked for — and a price panel passed as returns scaled by the volatility
of price levels. A `predictions` reference is refused with the
conversion that turns it into scores: it is a long (date, entity,
prediction) frame, and a classifier's predictions are probabilities whose
sign is a direction only after `convert_reference(to_kind='score_panel',
task='classification')` recentres them, which needs the task this tool
is not told.

## Covariance before optimization

`estimate_covariance` produces the matrix an optimizer consumes, separately
from consuming it. A sample covariance on a short window is badly
conditioned, and feeding one straight into a mean-variance solver produces
confident weights built on noise — inspecting the estimate first is what
makes that visible rather than showing up as an implausible allocation.

**It says what it dropped.** The estimate uses complete rows only, so one
asset with a short history removes every date it lacks for *all* of them:
on a live panel one such history silently removed **400 of 512 rows** with
`warnings: []`, and the risk-parity weights built on the remainder moved by
12.4% of NAV. The result now carries `n_rows_dropped`, and when it is not
zero a warning names the count and the asset with the shortest history, so
the choice — drop the asset, or start the window where every asset has
data — is the caller's rather than the estimator's.

**It says how many observations the EWMA estimate rests on.** Exponential
weighting lowers the effective sample size, `1 / Σw²`: a 60-observation
half-life on 252 rows rests on about 155 effective observations, not 252.
The result reports `effective_observations` (the row count for `sample` and
`ledoit_wolf`), `observations_per_parameter` is computed from it, and a
warning fires when there are no more effective observations than assets —
the estimate is rank-deficient by construction then, whatever the row count
says. Two half-lives used to be mishandled. `halflife=0` was read as the
default 60 (`halflife or 60`), while −5 was refused; 0 is now refused and
only an absent half-life takes the default. And a half-life short enough to
put all the weight on the last row — fewer than 2 effective observations,
below about 0.63 — is refused before the estimate: below about 0.019 the
unbiasing denominator `1 − Σw²` was exactly zero and a raw numpy
`LinAlgError` escaped, and between 0.019 and 0.05 a matrix with a condition
number of 1e12–1e17 came back without a warning.

The matrix, its condition number and its smallest eigenvalue are computed
on one BLAS thread, the product that builds the matrix as well as the
eigendecomposition, so the same returns give the same bits on any machine
with the same BLAS. Until 2026-10-04 the product kept the caller's threads:
an EWMA matrix's last bits followed the core count, by up to 1.8e-12
relative at 235 assets, and on CI's OpenBLAS a sample or Ledoit-Wolf
matrix's did too. A Ledoit-Wolf estimate no longer has scikit-learn compute
the precision matrix it stores by default, which nothing read: that was
15–27 ms of a 23–33 ms fit at 235 assets, and skipping it leaves the bits
identical.

**From a published panel, and published in turn.** `returns_ref` takes a
`returns_panel` reference in place of `tickers` and their window (exactly one
of the two; the window is refused beside a reference). Its columns are the
assets, aligned as return streams on the dates they all share; `streams` on
the result gives, per column, the returns supplied and the ones dropped
before, after and inside the common window, and `n_rows_dropped` counts the
panel's dates some column lacked — on a 300-day panel with 20, 10 and 5
blank days in three columns, the estimate rests on 265 rows and 35 are
dropped. Given a `run_id`, the matrix is published as
`sqt://covariance/<run_id>/covariance` and returned as `covariance_ref`, which
the allocators above take; the reference records the assets, the matrix,
whether it is annualized, at how many periods a year, the method and the
observations. A second publish at the same `run_id` is refused, so the tool
is not advertised as safe to repeat. Without a `run_id` nothing is written
and the result is what it was.

## Return streams on one calendar

`portfolio.streams` is where the library puts several dated return series on
the dates they share — the step every allocation across strategies, sleeves
or funds starts from, and the one an undated list cannot take: two series of
equal length from different calendars look aligned and correlate each day
against some other day.

```python
from standard_quant_tools.portfolio.streams import align_return_streams

frame, report = align_return_streams({"us": us, "eu": eu}, min_overlap=60)
```

Each stream is a Series on a DatetimeIndex. The frame holds the dates every
stream has, in date order, each value the stream's own. The report gives the
window every stream covers (`window_start`, the latest first date, and
`window_end`, the earliest last date) and the streams that set it, and per
stream the returns supplied, its first and last date, the rows kept, and the
returns dropped before the window, after it, and inside it on a date another
stream lacks. Two ten-return streams, one on business days from 2024-01-02
and one starting two days later without 2024-01-08, share seven dates: the
first loses two before the window and one inside it, the second three after
it.

Refused, each by name: fewer shared dates than `min_overlap`, naming the
stream whose removal would share the most and the range it covers; a date
that appears twice; a missing or non-finite value (a gap is a date the
stream does not have, not a NaN on it); a level series passed as returns
(the rule `refuse_if_not_returns`: a typical |value| above 1, an equity
curve or a price); undated values; and dates in different time zones.

`streams_from_panel` reads a wide returns frame as streams, a blank cell a
date that column was not observed on. `combine_return_streams` resolves
streams — Series, or `ReturnStream` specs: a `returns_panel` reference and
its column, {ISO date: return}, or a symbol with both ends of its window —
aligns them and publishes the panel as a `returns_panel` at a `run_id`. The
`allocation` runtime's tool of the same name is its door for an agent (see
[31_allocation.md](31_allocation.md)); `allocate_across_sleeves` and
`evaluate_allocation_rule` there read the panel it publishes, or dated
values, and fetch nothing.

## Planning the transition

`plan_rebalance` schedules the path from current to target weights under a
participation cap, priced by `adv` (average daily dollar volume per name).
A name missing from `adv` used to be read as having **zero** liquidity: it
never traded, and the warning did not say why. It is now refused by name —
supply that name's ADV, or omit `adv` altogether to plan without
participation limits, which the result then reports as an unpriced
transition rather than a free one. A NaN or infinite entry in
`current_weights`, `target_weights` or `adv` is refused by key: a holding
has no gap to skip, and it used to come back as a schedule of nulls. A plan
already at target reports a `residual_distance` of 0.0.

`portfolio_value` must be a finite positive number, at the tool and in
`portfolio.rebalance.plan_rebalance`: NaN passed the `<= 0` check and the
plan came back with a NaN cost and `converged=True`, and an infinite value
— which the tool's `gt=0` admitted — gave a NaN cost. The library also
refuses, by key, a NaN or infinite holding or ADV (the tool's schema
already did), and a non-finite or negative `impact_coefficient`.

## The cost of one trade

`estimate_trade_cost` prices one hypothetical trade under a composed cost
model — commission, spread, square-root impact and, on a short, borrow —
itemized leg by leg from the same functions a backtest applies per fill.

`breakeven_move_bps` is `2 x execution_bps + holding_bps`: commission,
spread and impact are paid on the way in and again on the way out, while a
short's borrow accrues once over `holding_days`. A 90-day short at 7 bps of
execution and 300 bps of borrow needs 87.97 bps (it reported 161.95). Borrow
is charged on `side="sell"` only. Margin interest on a negative
`margin_cash` is the account's, reported as `account_margin_interest` in
currency and kept out of both totals. `day_count` (`"ACT/365F"`, the
default and the arithmetic it always used, or `"ACT/360"`) is the basis the
borrow and the account margin accrue on over `holding_days`: the same
90-day, 300 bps short holds 75.00 bps of borrow on ACT/360 against 73.97 on
ACT/365F. 30/360 is refused with the reason.

---

<sub>[Standard Quant Tools](../README.md) · [Documentation](README.md) · by Karan Vora · Apache 2.0</sub>
