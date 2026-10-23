# Module reference

Every public module in one place: what it holds, and where the deep
documentation for it lives. This exists because the per-module guides
(`01_data_fetching.md` through `30_build_guide.md`) each go deep on one
subject, and none of them answers "what is actually in this library".

The prose here is the orientation layer. When a section raises a question it
does not answer, the linked guide answers it.

## Data (`standard_quant_tools.data`)

Four providers implement the same `DataProvider` ABC — `DataFactory.get_provider("yfinance" | "bloomberg" | "polygon" | "databento")` — so switching is a one-line change with zero changes downstream.

| Function | Description | Returns |
|---|---|---|
| `get_ohlcv(symbol, start, end, interval)` | Historical OHLCV data | `pd.DataFrame` |
| `get_ohlcv_async(...)` | Non-blocking OHLCV fetch | `Awaitable[pd.DataFrame]` |
| `get_ticker_info(symbol)` | Company metadata | `TickerInfo` (Pydantic) |
| `get_financial_ratios(symbol)` | P/E, P/B, D/E, ROE, margins, etc. | `FinancialRatios` (Pydantic) |
| `get_metadata(symbol, interval)` | Dataset provenance: adjusted, survivorship-free, point-in-time, timezone | `DataSetMetadata` (Pydantic) |

**`YFinanceProvider`** (default) — **Caching:** Historical OHLCV calls are saved as Parquet files under `~/.cache/standard_quant_tools/ohlcv/`, by every provider rather than this one alone (`data._cache`). Subsequent calls — even from a new Python process — load from disk rather than the network. Override the cache directory with `SQT_CACHE_DIR` (read once, at first use; blank means the default, a relative path is refused). An entry is served only if it passes the checks a live answer gets — the five OHLCV columns, no null `Close`, bars inside the requested window — and is otherwise evicted and refetched. `sqt cache gc` lists (and with `--confirm` deletes) files of a dead format generation and temp files an interrupted write left behind over an hour ago, and nothing else.

**`BloombergProvider`** — talks to a local, logged-in Bloomberg Terminal via Desktop API (`blpapi`, optional dependency). No API key: DAPI authenticates via the Terminal login itself; only `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` are configurable (via `.env` locally or CI secrets — see Config & secrets above), and neither is a secret. Daily/weekly/monthly bars only. See [Documentation/01_data_fetching.md](01_data_fetching.md#bloomberg-provider) for the full reference.

**`PolygonProvider`** — talks to Polygon.io's plain REST API, no vendor SDK required. Needs an API key (`SQT_POLYGON_API_KEY`, no default — free tier available at [polygon.io/dashboard/api-keys](https://polygon.io/dashboard/api-keys)). Supports `1m`/`5m`/`15m`/`30m`/`60m`/`1d`/`1wk`/`1mo`/`3mo` bars via the Aggregates endpoint; `get_financial_ratios` derives P/E, P/B, D/E, ROE, and margins from the most recent financials filing plus market cap (no forward estimates or dividend yield). See [Documentation/01_data_fetching.md](01_data_fetching.md#polygonio-provider) for the full reference.

**Data quality (`standard_quant_tools.data.quality`):** `detect_missing_bars`, `detect_stale_prices`, `detect_price_jumps`, `detect_volume_anomalies` — heuristic checks on an already-fetched OHLCV frame (missing sessions, frozen prices, large single-bar jumps, zero or thin volume); `detect_duplicate_timestamps`, `detect_out_of_order_timestamps`, `detect_ohlc_inconsistencies` — integrity checks whose findings are data errors; `detect_sample_feed` — provenance, from the dataset stamped on the frame, since no volume statistic can tell a feed thin on every bar from the tape; `detect_split_like_moves` — a move beyond 35%, or a fall the size of a 3:2 split, named after the split ratio it fits, the screen the backtest and the dataset build share. `detect_missing_bars` asks `exchange_calendars` for the venue's sessions (`calendar="XNYS"` by default), so a market holiday is not a gap; without that package, or for a code it does not know, it falls back to the weekday rule and every entry says which basis it used. Exposed together with `get_metadata` via the `get_data_quality_report` agent tool, which takes the source, the calendar and the volume-anomaly thresholds.

---

## Technical Indicators (`standard_quant_tools.indicators`)

**Trend**

| Function | Description | Performance |
|---|---|---|
| `sma(series, period)` | Simple Moving Average | Pandas rolling |
| `ema(series, period)` | Exponential Moving Average | Pandas EWM |
| `macd(series, fast, slow, signal)` | MACD + Signal + Histogram | Pandas EWM |
| `adx(high, low, close, period)` | ADX + DI+ + DI− | **C++ extension** / Numba JIT / Python fallback |
| `parabolic_sar(high, low)` | Parabolic SAR + Trend direction | **C++ extension** / Numba JIT / Python fallback |
| `williams_r(high, low, close, period)` | Williams %R oscillator | Pandas rolling |

**Momentum**

| Function | Description | Performance |
|---|---|---|
| `rsi(series, period)` | RSI (Wilder's smoothing) | **C++ extension** / Numba JIT / Python fallback |
| `stochastic_oscillator(high, low, close)` | Stochastic %K and %D | **C++ extension** / Pandas rolling fallback |

**Volatility**

| Function | Description | Performance |
|---|---|---|
| `bollinger_bands(series, period, num_std)` | Upper / Middle / Lower bands | **C++ extension** / Pandas rolling fallback |
| `atr(high, low, close, period)` | Average True Range (SMA of TR) | **NumPy single-pass** (5.6× vs `pd.concat`) |
| `wilder_atr(high, low, close, period)` | Wilder's ATR (SMA seed + Wilder smoothing) | **C++ extension** / Python fallback |

**Volume**

| Function | Description |
|---|---|
| `obv(close, volume)` | On Balance Volume |
| `vwap(high, low, close, volume, period)` | VWAP (cumulative or rolling) |
| `mfi(high, low, close, volume, period)` | Money Flow Index |

---

## Metrics (`standard_quant_tools.metrics`)

**Return Metrics**

| Function | Description |
|---|---|
| `cumulative_return(series)` | Total return over period |
| `cagr(series)` | Compound Annual Growth Rate |
| `annualized_volatility(returns)` | Annualized standard deviation |

**Risk Metrics**

| Function | Description |
|---|---|
| `sharpe_ratio(returns, risk_free_rate)` | Excess return per unit of total risk |
| `sortino_ratio(returns, risk_free_rate)` | Excess return per unit of downside risk |
| `max_drawdown(series)` | Maximum peak-to-trough decline |
| `calmar_ratio(equity_curve)` | CAGR / \|max drawdown\| |
| `var_historical(returns, confidence)` | Historical Value at Risk |
| `var_parametric(returns, confidence)` | Gaussian VaR (scipy optional) |
| `cvar(returns, confidence)` | Conditional VaR / Expected Shortfall |
| `information_ratio(returns, benchmark_returns)` | Active return / tracking error |
| `treynor_ratio(returns, benchmark_returns)` | Excess return / beta |
| `drawdown_series(series)` | Full drawdown time series |

**Backtest Diagnostics**

| Function | Description |
|---|---|
| `drawdown_periods(equity_curve)` | One row per drawdown episode: peak, trough, recovery, depth, duration |
| `top_n_drawdowns(equity_curve, n)` | The n deepest drawdown episodes, worst first |
| `trade_expectancy(trade_log)` | Expectancy, avg winner/loser, payoff ratio, consecutive win/loss streaks |
| `trade_excursions(trade_log, price_data)` | Adds MAE/MFE (max adverse/favorable excursion) columns to a trade log |
| `exposure_stats(executed_signal, trade_log)` | Time in market, gross/net exposure, % long/short, avg holding period |

Computed entirely from data a backtest already produces (`equity_curve`, `trade_log`, the OHLCV frame) — no engine changes required. Exposed together via the `get_backtest_diagnostics` agent tool.

**VaR as a forecast, and its backtest**

| Function | Description |
|---|---|
| `forecast_var_series(returns, *, method="historical", confidence=0.99, window=250, ewma_lambda=0.94, tail_fraction=0.1, refit_every=20)` | One-step-ahead VaR and ES for every date after the first window, each from the `window` returns before it — `historical`, `normal`, `cornish_fisher`, `ewma`, `filtered_historical` or `evt_garch` — and the next period's |
| `backtest_var_series(frame, confidence, *, random_seed=0, n_simulations=10000)` | The exception dates, Kupiec, Christoffersen and conditional coverage, the Basel traffic light, and Acerbi-Szekely Z2 with a simulated p-value; an undefined statistic is None with a `null_reason` |

Exposed as `forecast_var_series` and `backtest_var_forecasts` in the
`research` runtime; see [03_metrics.md](03_metrics.md) and
[23_inference.md](23_inference.md).

---

## Analysis (`standard_quant_tools.analysis`)

16 functions plus the options modules. Several have a **C++ fast path** via `_sqt_core` — numbers below are measured, not projected (see the CHANGELOG for the full methodology and an earlier round of unmeasured projections that turned out to overstate several of these, since corrected):
- `calculate_beta` — 2-variable OLS via closed-form normal equations (1.4× vs. `np.linalg.lstsq` — a real but modest win, not the 10–20× originally projected before this was actually benchmarked)
- `rolling_beta` — incremental O(1)-per-bar sum updates (4.7× vs. two pandas rolling passes), plus a further ~1.1–1.5× from an optional runtime AVX2+FMA dispatch path
- `half_life` / `compute_spread` — same OLS kernel, same modest (~1.1×) speedup
- `cointegration_test` — full Engle-Granger pipeline (25–30× vs. statsmodels at n=500; **37–41×** at n=2 000, because the ADF lag sweep now reads every candidate lag off a single nested factorization instead of factorizing once per lag)
- `scan_cointegrated_pairs` — every pair of a universe, in both orders, in one native call, parallel across pairs: 49–51 µs a pair at 500 bars and 231–249 µs at 2 000 on 200 names, against 0.23–0.25 ms and 0.82–0.84 ms for one `cointegration_test` call per pair in a Python loop
- `hurst_exponent` / `rolling_hurst` — DFA + R/S + sliding window (83–131× / 274×)
- `rolling_factor_loadings` — per-window rank-revealing QR with column pivoting (2.3–10× vs. per-window `lstsq`, larger at shorter windows). This deliberately replaced a much faster incremental-Cholesky path that was **wrong**: its pivot test compared every factor column against the intercept column's diagonal, so factor values around 1e-6 made the whole window read as singular and it returned all-NaN where NumPy returned correct coefficients. Recovering the speed via a QR update/downdate is not attempted; see [16_performance.md](16_performance.md)

### Options Pricing, Greeks & Implied Volatility

`standard_quant_tools.analysis.options` — Black-Scholes-Merton pricing for **European options only**. Dependency-free (standard normal CDF/PDF via `math.erf`, not scipy).

```python
from standard_quant_tools.analysis.options import black_scholes_price, black_scholes_greeks, implied_volatility

price = black_scholes_price(spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, volatility=0.20, option_type="call")
greeks = black_scholes_greeks(42, 40, 0.5, 0.10, 0.20, "call")   # delta, gamma, vega, theta, rho, d1, d2

iv = implied_volatility(option_price=price, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, option_type="call")
print(iv["implied_volatility"], iv["converged"], iv["method"])  # 0.20, True, "newton"
```

`implied_volatility` solves via Newton-Raphson (vega as the derivative) and declares convergence on the **volatility step** it just took (`tol_sigma`, the price tolerance scaled by vega), handing over to a bisection over `[1e-6, 5.0]` when vega falls below a floor or a step leaves the bracket. It used to test an absolute price tolerance *before* taking a step, so where vega is small it returned its initial guess as "converged" — 0.2 for true vols of 3.00, 1.20 and 0.45 on short-dated puts. The result reports `price_error` and `at_bound`; a price bit-for-bit at intrinsic (which the pricer itself produces deep in the money) is admitted and returned as the largest volatility that still reproduces it, with `at_bound=True`, and a price of 0.0 is refused as underflow rather than searched. See [Documentation/12_options.md](12_options.md) for the full reference, including unit conventions for `vega`/`theta` and the no-arbitrage bound check `implied_volatility` runs before solving.

#### Whole chains (`standard_quant_tools.analysis.options_batch`)

The same functions over a chain, in one call, with a **C++ fast path** (OpenMP across contracts) and a numpy fallback that returns the same results. Arguments broadcast; `is_call` is a boolean per contract.

| Function | Description |
|---|---|
| `implied_volatility_batch(option_price, spot, strike, time_to_expiry, risk_free_rate, dividend_yield=0.0, is_call=True, ...)` | `implied_volatility`, contract for contract — same algorithm, bounds, tolerances and results. A quote no volatility reproduces is a per-contract `reason` (`price_not_positive`, `price_not_finite`, `below_lower_bound`, `above_upper_bound`, `no_root_in_bracket`) with NaN volatility; an input outside the pricing domain refuses the batch naming the contract |
| `black_scholes_greeks_batch(spot, strike, time_to_expiry, volatility, risk_free_rate, dividend_yield=0.0, is_call=True, *, grid=False)` | `option_greeks`' price and full greek set in its units, per contract or, with `grid=True`, every contract at every spot — `(contracts, spots)` |
| `zero_gamma_spot(strike, time_to_expiry, volatility, quantity, *, spot_low, spot_high, ...)` | Where the book's net signed gamma crosses zero in the bracket (grid scan, then Brent), or `None` with the reason when it does not |

476 implied vols in 0.2–0.3 ms against 14–27 ms one call at a time, and 29,036 greeks (61 spots × 476 contracts) in 0.3 ms against 100 ms — see [16_performance.md](16_performance.md#option-chains).

`analysis.option_book.aggregate_option_book(options, spots, rates, *,
dividend_yields=None, underlying_units=None, betas=None,
valuation_date=None, spot_shocks=None)` prices a book's option lines
through `black_scholes_greeks_batch` and reports dollar delta, gamma per
1%, vega, theta per calendar day and rho per line, per underlying and in
total, the beta-weighted delta, and delta-gamma P&L against full
revaluation on a grid of spot shocks. Each option line needs its
`multiplier`.

### Regression

```python
from standard_quant_tools.analysis import calculate_beta, rolling_beta

stats = calculate_beta(asset_returns, benchmark_returns)
# {'alpha': 0.0003, 'beta': 1.12, 'r_squared': 0.78}

rolling_df = rolling_beta(asset_returns, benchmark_returns, window=60)
# DataFrame with 'Rolling_Beta' column
```

`calculate_beta` answers the same on both backends where the fit is degenerate: fewer than
two overlapping points or a constant benchmark (the slope is unidentified) give NaN for
`alpha`, `beta` and `r_squared`; a constant asset gives `beta` 0 and `r_squared` NaN (no
variance to explain). The NumPy path used to return lstsq's minimum-norm solution for a
constant benchmark. `rolling_beta` gives each window its own beta on both backends,
including the windows after a large print has left (the NumPy fallback centres each window
on its own means rather than using pandas' online rolling sums, which lost digits there);
a window whose benchmark is flat is NaN.

### Multi-Factor Regression

```python
from standard_quant_tools.analysis import multi_factor_regression, rolling_factor_loadings
import pandas as pd

factors = pd.DataFrame({"mkt": spy_rets, "smb": smb_rets, "hml": hml_rets})
result = multi_factor_regression(asset_returns, factors)

print(result["loadings"])          # {"mkt": 1.12, "smb": -0.3, "hml": 0.05}
print(result["t_stats"])           # includes alpha and all factor names
print(result["r_squared"])         # 0.78
print(result["adj_r_squared"])     # 0.77

rolling = rolling_factor_loadings(asset_returns, factors, window=60)
# DataFrame (dates × ["alpha", "mkt", "smb", "hml"])
```

Linearly dependent factors (a duplicate, a sum or multiple of others, a constant) raise
`ValidationError` naming the factor to drop; `rolling_factor_loadings` leaves such a
window NaN instead.

`estimate_factor_model(asset_returns, factor_returns, *, intercept=True,
periods_per_year=252)` (`analysis/factor_model.py`) fits every asset at
once on the same dates and returns the loadings B, the factor covariance
F and the specific variances D (SSR / (n − p)), each with a standard
error, the alphas, the in-sample R², the implied covariance B F B′ + D and
the largest residual correlation between two assets.

### Cointegration & Pairs Spread

```python
from standard_quant_tools.analysis import (
    cointegration_test, compute_spread, half_life, spread_zscore
)

result = cointegration_test(ko_prices, pep_prices)     # at least 20 aligned bars
# {'cointegrated': True, 'hedge_ratio': 0.83, 'p_value': 0.003,
#  'half_life_days': 14.2, 'half_life_mean_reverting': True, ...,
#  'critical_values': {'1%': -3.92, '5%': -3.35, '10%': -3.05}}

spread = compute_spread(ko_prices, pep_prices)           # pd.Series
hl     = half_life(spread)                               # 14.2 (bars)
z      = spread_zscore(spread, window=30)                # rolling z-score signal

from standard_quant_tools.analysis.cointegration import (
    half_life_statistics, scan_cointegrated_pairs,
)
half_life_statistics(spread, fitted_residual=True)       # half-life + Dickey-Fuller gate
scan_cointegrated_pairs(prices)   # both regression orders, BH-adjusted p, cointegrated_fdr
```

### PCA on Returns

```python
from standard_quant_tools.analysis import pca_returns, factor_contributions

result = pca_returns(returns_df, n_components=3)
print(result["explained_variance_ratio"])  # PC1: 0.42, PC2: 0.12, PC3: 0.08
print(result["loadings"])                  # (assets × 3) — eigenvector matrix
print(result["factor_returns"])            # (dates × 3) — orthogonal PC time series

contrib = factor_contributions(returns_df, n_components=3)
# DataFrame (assets × PCs): marginal R² each PC contributes per asset
```

### Hurst Exponent

```python
from standard_quant_tools.analysis import hurst_exponent, rolling_hurst
from standard_quant_tools.analysis.hurst import HAS_CPP

print("C++ backend active:", HAS_CPP)     # True once _sqt_core is built from these sources

result = hurst_exponent(returns)          # pass RETURNS not prices
# {'hurst': 0.38, 'regime': 'mean_reverting', 'regime_band': 0.07,
#  'fit_r_squared': 0.97, 'method': 'dfa', ...}

rolling = rolling_hurst(returns, window=252, step=5)   # pd.Series of H values
```

| H value | Regime | Implication |
|---|---|---|
| > 0.5 + band | trending | Momentum strategies have an edge |
| within the band | random walk | No persistent signal from past prices |
| < 0.5 − band | mean-reverting | Contrarian / mean-reversion strategies |

The band (`regime_band`) is 1.645 white-noise standard deviations of the estimate at the
series length, at least 0.05, so it is wider for short series. R/S is corrected for its
small-sample bias (`hurst_raw` is the uncorrected slope).

> The C++ extension accelerates `hurst_exponent` by 83–131× and `rolling_hurst`
> by 274× (measured; see [16_performance.md](16_performance.md)). The API is identical with or without it — pure Python fallback is automatic. See [30_build_guide.md](30_build_guide.md).

---

## Backtesting (`standard_quant_tools.backtest`)

Vectorized engine with transaction costs, trade log, and full metric output.

```python
from standard_quant_tools.backtest.engine import run_strategy

signals = (df['Close'] > sma(df['Close'], 50)).astype(int)

result = run_strategy(
    df, signals,
    initial_capital=10_000,
    commission_pct=0.001,
    slippage_pct=0.0005,
    include_trade_log=True,
)
print(f"Sharpe: {result['sharpe_ratio']:.2f}  |  Win Rate: {result['win_rate']:.1%}  |  Trades: {result['num_trades']}")
```

**Output keys:** `final_equity`, `total_return`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `num_trades`, `avg_trade_return_pct`, `equity_curve`, `trade_log`

### Parameter Grid Search

```python
from standard_quant_tools.backtest import backtest_grid

results = backtest_grid(
    price_data=df,
    strategy="sma_crossover",          # or any of the 8 STRATEGY_REGISTRY names
    param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50, 100]},
    sort_by="sharpe_ratio",
    n_workers=4,                        # parallel ProcessPoolExecutor
)
print(results.head)   # 9 combinations ranked by Sharpe
```

**8 built-in strategies** (`backtest.strategies.STRATEGY_REGISTRY`): `sma_crossover`, `rsi_mean_reversion`, `macd_crossover`, `bollinger_reversion`, `donchian_breakout` (Turtle-style channel breakout), `momentum_timeseries` (trailing-return threshold, no state machine — the cheapest to evaluate), `vwap_reversion` (mean reversion to rolling VWAP — aimed at intraday/tick data), `adx_trend` (ADX-strength-filtered directional trend). The 4 newer ones don't have dedicated `run_*_backtest` tools — use them via `backtest_grid`, `get_backtest_diagnostics`, or `run_backtest_compact`, or call `STRATEGY_REGISTRY[name](df, **params)` directly. Every hysteresis-based strategy (`rsi_mean_reversion`, `bollinger_reversion`, `donchian_breakout`, `vwap_reversion`) runs its entry/exit tracking through a numba-JIT state machine — no interpreted Python loop regardless of series length; the other four need no per-bar state at all and are pure vectorized pandas/numpy. See [Documentation/04_backtesting.md](04_backtesting.md) for the full reference.

`strategy` also accepts your own signal-generating callable — grid search, C++ speed, and `sort_by` ranking all work identically on your own alpha logic, not just the built-ins. For a pre-computed signal matrix across a ticker universe, see `run_signal_panel_backtest` in [Documentation/04_backtesting.md](04_backtesting.md#grid-searching-your-own-signal).

### Portfolio Simulation Engine

`run_signal_panel_backtest` gives every ticker its own independent capital and blends the return streams afterward. `run_portfolio_simulation` (`standard_quant_tools.backtest.portfolio_engine`) is the true-portfolio counterpart: one shared cash balance, position sizing relative to current account equity, and rebalancing at specific dates — weights drift between rebalances instead of being re-applied every bar.

```python
from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation

result = run_portfolio_simulation(
    price_data={"AAPL": aapl_df, "MSFT": msft_df},
    target_weights=weights_df,           # DataFrame indexed by rebalance date, one column per ticker
    initial_capital=100_000,
    max_gross_leverage=1.0,
    max_position_pct=0.25,
    commission_model="pct",              # or "per_share"
    use_impact_model=True,
    max_adv_participation=0.1,
)
print(result['final_equity'], result['leverage_curve'].mean)
```

`max_adv_participation` is a cap, not a kill switch: a trade over it is sized down to the cap and the shortfall is recorded per rebalance (`n_capped`, `capped_notional`, `capped` in `rebalance_log`, plus a summary warning). It used to refuse the whole simulation, so a capacity study could not be expressed.

Pluggable building blocks compose into `run_portfolio_simulation` (or can be used standalone):

- `standard_quant_tools.backtest.costs` — `percentage_commission`, `per_share_commission`, `fixed_bps_spread`, `pct_of_range_spread`, `sqrt_impact_bps`, `impact_cost`, `short_borrow_cost`, `margin_interest`
- `standard_quant_tools.backtest.constraints` — `adv_participation`, `days_to_liquidate`, `liquidation_days`, `sector_exposure`, `capacity_report`
- `standard_quant_tools.backtest.sizing` — `rank_weighted`, `equal_weight_top_bottom`, `zscore_normalized`, `vol_scaled`, `dollar_neutral` — turns a SCORE signal panel into a target-weight panel
- `standard_quant_tools.backtest.pairs` — `run_pair_backtest`, a two-leg pair trade as one dollar-neutral portfolio (reuses `run_portfolio_simulation`)
- `standard_quant_tools.backtest.artifacts` — `save_artifact` / `load_artifact`, a local Parquet store for equity curves/trade logs too large to embed inline in an agent-tool response

### Robustness Diagnostics

`standard_quant_tools.backtest.robustness` answers "is this backtest result trustworthy, or a fluke of one sample path / one lucky parameter combination": `block_bootstrap_ci` (confidence interval on a point-estimate metric), `parameter_sensitivity` (best-vs-median gap on a grid search), and `deflated_sharpe_ratio` (corrects the best observed Sharpe for having been selected as the max of `n_trials` attempts). Complementary to, not a substitute for, out-of-sample walk-forward validation.

---

## Portfolio (`standard_quant_tools.portfolio`)

```python
from standard_quant_tools.portfolio import portfolio_metrics, correlation_matrix, fetch_returns_sync

returns_df = fetch_returns_sync(['AAPL', 'MSFT', 'GOOGL'], '2023-01-01', '2024-01-01')
weights = [0.4, 0.35, 0.25]

metrics = portfolio_metrics(returns_df, weights)
print(f"Portfolio Sharpe: {metrics['sharpe_ratio']:.2f}")
print(f"Portfolio VaR(95%): {metrics['var_95']:.4f}")

corr = correlation_matrix(returns_df)
```

### Portfolio Optimization

`standard_quant_tools.portfolio.optimize` — produces weights, rather than only scoring weights you already chose (`portfolio_metrics` above) or converting an existing alpha score into weights (`backtest.sizing`).

```python
from standard_quant_tools.portfolio import mean_variance_optimize, risk_parity_weights, black_litterman, build_bl_views

# Markowitz mean-variance — max_sharpe / min_volatility / target_return / target_volatility.
# allow_short=True with max_weight=None is closed-form (numpy only); anything
# constrained (long-only and/or a max_weight cap) uses scipy, except
# target_volatility, which is solved on the critical line (numpy only).
result = mean_variance_optimize(returns_df, objective="max_sharpe", allow_short=False, max_weight=0.4)
print(result["weights"], result["converged"])

# Risk parity — equal (or custom-budgeted) fractional contribution to variance.
cov = (returns_df.cov * 252).to_numpy
rp = risk_parity_weights(cov)
print(rp["weights"], rp["risk_contributions"])

# Black-Litterman — market-equilibrium prior blended with explicit views.
P, Q, omega = build_bl_views(["AAPL", "MSFT", "GOOGL"], [{"assets": {"AAPL": 1.0}, "view_return": 0.15}], cov)
bl = black_litterman(cov, market_weights=[0.4, 0.35, 0.25], P=P, Q=Q, omega=omega)
print(bl["posterior_returns"], bl["implied_weights"])
```

See [Documentation/05_portfolio.md](05_portfolio.md#portfolio-optimization) for the full reference, including exactly which cases are closed-form vs. require scipy, and the `run_portfolio_optimization` agent tool (`method="max_sharpe"|"min_volatility"|"target_return"|"target_volatility"|"risk_parity"|"black_litterman"`).

`standard_quant_tools.portfolio.critical_line` — the efficient frontier under a budget and per-asset bounds, exactly: Markowitz's critical line algorithm. `critical_line(mu, cov, lower, upper, budget=1.0, names=None)` returns the corner portfolios (each with its risk tolerance λ, the events that happen there and a KKT certificate), and `at_return`, `at_volatility` and `max_sharpe` for the exact points between them. `run_portfolio_optimization`'s bounded `target_volatility`, `get_efficient_frontier`'s bounded curve and the long-only `optimize_max_diversification` are built on it.

`standard_quant_tools.portfolio.general_optimizer.optimize_portfolio(covariance, *, assets=None, expected_returns=None, objective="min_variance", risk_aversion=1.0, risk_free_rate=0.0, benchmark_weights=None, lower_bounds=0.0, upper_bounds=None, budget=1.0, budget_type="equal", groups=None, group_min=None, group_max=None, factor_loadings=None, exposure_min=None, exposure_max=None, max_volatility=None, max_tracking_error=None)` — exact constrained optimization on expected returns and a covariance the caller supplies: `min_variance`, `max_utility`, `max_sharpe`, `min_tracking_error` or `max_active_utility` under per-asset bounds, a budget (equal or at most; 0 is dollar-neutral), group and factor-exposure limits and volatility or tracking-error caps. It returns the weights, every binding constraint with its shadow price, and a KKT certificate computed from the weights; an infeasible set of limits is refused naming the ones that conflict. Up to 500 assets; needs scipy. The `optimize_portfolio` agent tool and `optimize_replication_basket` solve through it. Deep guide: [05_portfolio.md](05_portfolio.md#constrained-optimization-on-your-own-numbers).

---

## Screener (`standard_quant_tools.screener`)

```python
from standard_quant_tools.screener import screen_stocks

result = screen_stocks(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
    filters={
        'pe_ratio_max': 35,
        'rsi_max': 50,
        'price_above_sma': 50,
        'beta_max': 1.5,
    },
    sort_by='rsi_14',
    ascending=True,
)
```

**Available filters:** `pe_ratio_max`, `pb_ratio_max`, `debt_equity_max`, `roe_min`, `profit_margin_min`, `div_yield_min`, `market_cap_min`, `rsi_max`, `rsi_min`, `price_above_sma`, `price_below_sma`, `beta_max`, `beta_min`

**Failure reporting:** A ticker that raised an exception (bad symbol, fetch error) is never indistinguishable from one that simply failed a filter — both used to collapse to being silently dropped. The returned `DataFrame` now carries this on `.attrs`: `failed_filters` maps ticker → the specific filter key it failed (genuine rejection), `failed_tickers` maps ticker → the exception message (fetch/computation error), and `failed_batches` (multi-worker runs only) lists any worker batch that failed outright.

**Large universes:** One process by default, whatever the size: fetches run on up to 64 threads. `n_workers > 1` splits the universe across a `ProcessPoolExecutor`, which is usually slower (each worker re-imports the library and re-reads its bars) and is declined while a decision record is open or a Databento request gate is registered.

**Beta filter optimisation:** When `beta_max` / `beta_min` filters are active, SPY data is pre-fetched once per batch instead of once per ticker — a single HTTP round-trip for the whole universe in the default single-process run, or once per worker process when `n_workers > 1` (still eliminating the N−1 redundant per-ticker fetches within each worker's batch).

```python
result = screen_stocks(sp500_tickers, filters={...}, n_workers=8)
```

---

## AI Agent Tools (`standard_quant_tools.agent`)

213 LLM-callable tools with Pydantic input/output models and OpenAI/Anthropic function-calling schemas — including two tools that backtest a signal you computed yourself rather than one of the built-in indicator strategies.

`Implementation/{Anthropic,OpenAI,Gemini}/` are single-agent reference scripts across all three providers — each narrows the tool list per request via a lightweight **router** (`standard_quant_tools.agent.router`) instead of handing the model all 213 tools on every call: one cheap classification call picks the 1-2 relevant tool categories before the real agent loop starts, no separate agent session required. Each provider folder also carries `Agent_Model_Builder.py`, the one script that drives the separate 38-tool modeling runtime instead — it passes `registry="modeling"` and skips the router, since one ordered pipeline has no selection ambiguity to remove. For a heavier, more thorough split, `Multi_Agent_Implementation/` (Anthropic only for now) is a full **orchestrator-workers** architecture — a lead agent that delegates to 18 specialist sub-agents, fifteen over the analysis runtimes, two over the modeling one and one over `feature_lab`, each with its own independent session scoped to a small, non-overlapping tool subset. The analysis workers build on the same category taxonomy (`TOOL_CATEGORY`), so a tool's categorization only needs to be correct in one place. Splitting tools this way is a direct fix for tool-selection confusion between similar tools (e.g. a built-in strategy backtest vs. a bring-your-own-signal backtest, or "run this strategy" vs. "optimize this strategy's parameters"): a worker/routed request that was never given the other tool cannot call it by mistake. See [Documentation/13_agent_orchestration.md](13_agent_orchestration.md).

```python
from standard_quant_tools.agent.tools import (
    get_agent_tools, analyze_stock_risk,
    run_factor_regression, run_cointegration_test,
    run_pca_analysis, run_hurst_analysis,
    get_stock_fundamentals, run_backtest_optimization,
    get_advanced_indicators, get_rolling_beta,
    get_extended_risk_metrics,
    run_custom_signal_backtest, run_signal_panel_backtest,
    get_backtest_diagnostics,
)
from standard_quant_tools.agent.models import (
    AnalysisInput, FactorRegressionInput,
    CointegrationInput, PCAInput, HurstInput,
    FundamentalsInput, BacktestOptInput,
    AdvancedIndicatorsInput, RollingBetaInput, ExtendedRiskInput,
    CustomSignalBacktestInput, SignalPanelBacktestInput,
    BacktestDiagnosticsInput,
)

# Get tool schemas for your LLM
tools = get_agent_tools  # 213 tools ready for function calling

# Risk analysis
result = analyze_stock_risk(AnalysisInput(symbol='NVDA', benchmark='SPY', period='1y'))
print(result.model_dump_json(indent=2))

# Multi-factor regression (SPY/IWM/IWD as mkt/smb/hml proxies)
result = run_factor_regression(FactorRegressionInput(
    symbol='AAPL',
    factor_tickers=['SPY', 'IWM', 'IWD'],
    factor_names=['mkt', 'smb', 'hml'],
    start_date='2022-01-01',
    end_date='2024-01-01',
    rolling_window=60,
))

# Pairs cointegration + live z-score signal
result = run_cointegration_test(CointegrationInput(
    symbol_a='KO', symbol_b='PEP',
    start_date='2022-01-01', end_date='2024-01-01',
    zscore_window=30,
))
print(result.signal)  # "long_a_short_b" | "short_a_long_b" | "neutral"

# PCA on a basket of stocks
result = run_pca_analysis(PCAInput(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN'],
    start_date='2022-01-01', end_date='2024-01-01',
    n_components=3,
))

# Hurst exponent with rolling regime breakdown
result = run_hurst_analysis(HurstInput(
    symbol='SPY', start_date='2022-01-01', end_date='2024-01-01',
    method='dfa', rolling_window=252,
))
print(result.regime)   # "trending" | "random_walk" | "mean_reverting"
```

Every tool by name, with the description the model reads and every
argument, is in [20_tool_index.md](20_tool_index.md) — generated from the
live registry, so it cannot drift. The prose is split between
[07_agent_tools.md](07_agent_tools.md) (the core fourteen and the agent
loop) and [09_advanced_agent_tools.md](09_advanced_agent_tools.md) (the
advanced, supplementary, custom-signal, analytics, options and diagnostic
tools).

## Modeling Runtime (`standard_quant_tools.modeling`)

A second, independent 38-tool runtime for building walk-forward-validated
statistical models from this library's own features (31 built-in:
technical, market, risk, volume, statistical, network, fundamental and
PCA-derived factors) or from a matrix computed elsewhere and registered
without copying it — never merged into the 213-tool
`get_agent_tools`/`TOOL_CATEGORY` surface above. A sibling `feature_lab`
runtime holds 11 more, for interrogating those features before a model
exists. Every tool with its arguments is in
[20_tool_index.md](20_tool_index.md); every feature, estimator, target and
spec option is generated into
[29_modeling_reference.md](29_modeling_reference.md).

| Axis | What is available |
|---|---|
| **Targets** | 18 — six buildable from prices (`forward_return`, `forward_return_vol_scaled`, `forward_return_rank`, `forward_return_market_neutral`, `forward_direction`, `triple_barrier`), the other twelve arriving through `register_external_panel` |
| **Tasks** | `regression`, `classification`, `ranking`, `survival` — each behind a `ModelAdapter` that owns how its arrays are built, scored and measured |
| **Estimators** | 19 always available, 8 more when their library is installed. scikit-learn throughout, plus `lightgbm`/`xgboost`, quantile-regression forms, LambdaRank rankers and a panel MLP |
| **Validation** | walk-forward (rolling or expanding), purged K-fold and combinatorial purged CV, all with a target-overlap purge |
| **Preprocessing** | pooled or cross-sectional normalization |
| **Weighting** | none, label uniqueness, time decay, or both |
| **Search** | optional grid, random or TPE search on each fold's training window |

Everything past the defaults is opt-in behind an explicit spec field, so an
existing `ModelSpec` predicts exactly what it predicted before.

Walk-forward validation purges training rows whose forward-return label
would resolve inside the test window — feature-side embargo alone does not
close that channel. Registered models are content-addressed, verified on
load, and self-contained; `score_model` refuses an `as_of` inside the
training window, since the deployed estimator is refit on the full panel.

Two of the options are corrections rather than variations, and the
documentation says so. **Cross-sectional normalization** exists because
pooled z-scoring leaves the market factor inside every feature, so a model
judged on cross-sectional IC can score well by learning "today was an up
day". **Sample weighting** exists because `effective_sample_size` was always
reported and never acted on: overlapping forward returns make consecutive
rows largely redundant. Neither is the default only because switching it
changes what every existing model predicts.

The provider and bar interval are named on the `DatasetSpec` (they were
previously implicit, so every dataset came from the default provider at its
default interval and no model recorded which), the universe is fetched
concurrently, and `build_model_dataset` returns the coverage and provenance
conditions that qualify the resulting metrics — a survivors-only universe, a
provider that revises history, a symbol covering part of the window, a
complete-case intersection that truncated the panel. Those travel onto the
trained model, so `inspect_model(view="lineage")` shows them next to the OOS
numbers.

See [Documentation/15_modeling.md](15_modeling.md) for the
full reference, including what is explicitly deferred (Polygon is the one
point-in-time fundamentals source, so yfinance and Bloomberg still report
`point_in_time=False`; time-varying universe membership needs
index-constituent history no shipped provider exposes, so survivorship bias
is disclosed rather than corrected; sequence and graph models wait for a
measured case).

---

## Derivatives (`standard_quant_tools.analysis.derivatives`)

What you do with an option price once you have one. `pricing.py` answers
"what is this worth"; this answers everything after that.

| Function | Description |
|---|---|
| `option_greeks(...)` | Full greek set including vanna, volga, charm, speed — validated against central finite differences |
| `analyze_strategy(legs,...)` | Payoff, breakevens and aggregate greeks of an arbitrary multi-leg position; legs with different expiries are valued at the first one, later legs marked by Black-Scholes |
| `fit_volatility_smile(...)` | Quadratic in log-moneyness, with a Durrleman arbitrage check |
| `volatility_cone(prices,...)` | Realized-vol percentiles by horizon, with the independent-window count |
| `analyze_vol_term_structure(...)` | Contango/backwardation and the forward vols a calendar spread prices |
| `check_put_call_parity(...)` | The model-free identity, with the implied dividend and forward for diagnosis |
| `implied_forward_price(...)` | Carry forward with financing, dividend and borrow separated |
| `expected_move(...)` | One-sd move and the straddle approximation, with the historical exceedance rate |
| `simulate_delta_hedge(...)` | Hedged P&L distribution; error scales as 1/sqrt(n_hedges). Every path's delta at a rebalance comes from one batched greek call |
| `option_risk_scenarios(...)` | Full revaluation over spot x vol, the whole grid in one batched call |

Deep guide: [21_derivatives.md](21_derivatives.md)

---

## Microstructure estimators (`standard_quant_tools.analysis.microstructure_estimators`)

Liquidity recovered from OHLCV, for the normal case where there is no tick
feed. Each names what it is a proxy for and how it fails. Dated bars out of
time order are sorted by their stamps (and the result says so); a repeated
stamp is refused.

| Function | Description |
|---|---|
| `roll_spread(prices,...)` | Effective spread from bid-ask bounce, with a `smallest_detectable_spread` floor |
| `corwin_schultz_spread(ohlc)` | Spread from the high-low range; reports the negative fraction |
| `amihud_illiquidity(ohlcv,...)` | Price move per dollar traded, reported as a percentile |
| `kyle_lambda(ohlcv,...)` / `kyle_lambda(trades=, quotes=, freq=)` | Market depth from signed order flow. From bars the sign is the bar's own return, so the result says `circular=True`; from a tape with quotes the flow is Lee-Ready signed and the midpoint change is regressed on it (`circular=False`); a tape without quotes is tick-rule signed off the trade prices it explains and says `circular=True` |
| `order_flow_imbalance(ohlcv,...)` | Signed imbalance, with non-overlapping persistence |
| `estimate_vpin(ohlcv,...)` | Flow one-sidedness in volume time; the trailing residue bucket is dropped and reported as `residual_volume`, and a bar that did not move is split half to each side (`undirected_volume_share`) |
| `intraday_volume_profile(bars, index_timezone=...)` | The U-shape, for scheduling — bucketed over the regular session (its edges, `bucket_span`), with `extended_hours_share` when the index carries a zone; every bucket is returned, empty ones with a zero share (`n_empty_buckets`), and `trough_bucket` is a bucket id |
| `implementation_shortfall(...)` | Perold decomposition: delay, impact, opportunity, fees |

Deep guide: [22_microstructure.md](22_microstructure.md)

---

## Inference and diagnostics (`standard_quant_tools.analysis.inference`, `.diagnostics`, `.structure`, `.stationarity`)

Error bars on estimates normally reported without any, and structure a mean
and standard deviation cannot see.

| Function | Module | Description |
|---|---|---|
| `bootstrap_statistic(...)` | `inference` | Block-bootstrap confidence interval for eleven statistics |
| `compare_distributions(a, b)` | `inference` | KS plus a moment and tail comparison |
| `rolling_correlation_stability(...)` | `inference` | Sign flips, range, and the correlation on days both series sit in their worst decile (null below 20 such days) |
| `decompose_returns(...)` | `inference` | Arithmetic vs geometric, and the volatility drag |
| `test_normality(values)` | `inference` | Jarque-Bera plus the tail ratio that actually matters |
| `estimate_tail_index(...)` | `inference` | Hill estimator; which moments exist |
| `ljung_box(series,...)` | `diagnostics` | Joint autocorrelation test across lags |
| `seasonality(returns,...)` | `diagnostics` | Calendar effects, Bonferroni corrected |
| `rolling_sharpe_stability(...)` | `diagnostics` | Did the edge decay, tested on non-overlapping halves |
| `drawdown_profile(returns,...)` | `diagnostics` | Every drawdown, with depth and duration separated |
| `lead_lag_matrix(returns,...)` | `diagnostics` | Cross-asset search, corrected for its own size |
| `structural_break_test(...)` | `diagnostics` | Chow test at a known date |
| `entropy_measures(series,...)` | `diagnostics` | Nonlinear structure a linear test would miss |
| `detect_change_points(...)` | `structure` | Binary segmentation for an unknown break date |
| `partial_correlation(...)` | `structure` | What survives removing the common drivers |
| `granger_causality(...)` | `structure` | Temporal precedence, Bonferroni corrected |
| `tail_dependence(...)` | `structure` | Do these move together where it matters |
| `run_stationarity_tests(...)` | `stationarity` | ADF, KPSS and variance ratio with a four-way verdict |
| `detect_regimes(...)` | `stationarity` | Volatility regimes by Gaussian mixture |

Deep guide: [23_inference.md](23_inference.md)

---

## Overfitting and trade analysis (`standard_quant_tools.backtesting.overfitting`, `.trade_analysis`)

How much of a backtest is real.

| Function | Module | Description |
|---|---|---|
| `deflated_sharpe_ratio(...)` | `overfitting` | Sharpe deflated for the number of trials |
| `probability_of_backtest_overfitting(...)` | `overfitting` | PBO across every equal split |
| `combinatorial_purged_cv(...)` | `overfitting` | Purged and embargoed train/test paths |
| `reality_check(*, question, ...)` | `overfitting` | White's Reality Check (or Hansen's SPA) of the best candidate against one benchmark, or an intersection-union test that one strategy beats every alternative; block bootstrapped, series by keyword |
| `regime_stratified_performance(...)` | `overfitting` | P&L concentration by regime |
| `parameter_decay(...)` | `overfitting` | Plateau or spike |
| `monte_carlo_trade_paths(...)` | `trade_analysis` | Drawdown distribution from reshuffled trades |
| `analyze_trade_clustering(...)` | `trade_analysis` | Runs test on wins and losses |
| `compare_against_random(...)` | `trade_analysis` | Do the entries beat a coin |
| `exposure_attribution(...)` | `trade_analysis` | Timing versus being invested |
| `break_even_cost(...)` | `trade_analysis` | The cost at which the edge disappears |

Deep guide: [24_overfitting.md](24_overfitting.md)

---

## Portfolio construction (`standard_quant_tools.portfolio.construction`)

Allocation that does not lean on expected returns, and what a portfolio is
actually exposed to.

| Function | Description |
|---|---|
| `risk_parity(covariance,...)` | Equal risk contribution, or an explicit risk budget |
| `hierarchical_risk_parity(returns)` | Allocation without inverting the covariance matrix; columns are sorted by name first, so the weights do not depend on the order the universe was listed in |
| `max_diversification(covariance, *, long_only=False, max_weight=None)` | Maximizes the diversification ratio: the closed form by default (which may short), or with `long_only=True` (and an optional cap) the long-only maximum, solved exactly and certified |
| `factor_exposure_budget(...)` | What the portfolio is a bet on, once names collapse into factors |
| `concentration_analysis(weights)` | Effective N, Herfindahl, top-k share |
| `marginal_risk_contribution(...)` | Where the risk in a held portfolio comes from |
| `liquidity_adjusted_var(...)` | VaR that admits you cannot exit at the mark |
| `portfolio_scenarios(...)` | Named shocks rather than a distribution |

`standard_quant_tools.portfolio.factor_risk.decompose_factor_risk(weights, factor_loadings, factor_covariance, specific_variance, *, benchmark_weights=None)` splits a book's volatility — and, given a benchmark, its tracking error — under Σ = BFB′ + D into factor and specific parts, by factor (exposure × factor volatility × correlation) and by asset, with Euler contributions that sum to the volatility; specific variances are required and every name is checked against every other. `standard_quant_tools.portfolio.tail_risk.decompose_tail_risk(scenarios, weights, *, confidence=0.95, bandwidth=None)` splits the ES (exactly, by Euler) and the VaR (by a Gaussian kernel, summing to the reported `var_kernel`) of a scenario matrix by asset; VaR and ES are `var_historical`'s and `cvar`'s on the same P&L, and the tail count is always reported. `silverman_bandwidth(values)` is the default bandwidth rule.

`standard_quant_tools.portfolio.position_sizing.size_position(account_equity, last_close, last_atr, ...)` sizes one position behind an ATR stop and, when the win statistics are given, at half the growth-optimal Kelly position f* = p/L − q/W, capped at `max_leverage` times equity (1 by default); it reports f* and the risk fraction f* × L, and sizes zero shares without an edge. It is the arithmetic `get_position_size` calls, and refuses a stop, share count or position value past the float range by name. `kelly_notional_fraction(p, W, L)` and `kelly_log_growth(f, p, W, L)` are the two formulas.

Deep guide: [05_portfolio.md](05_portfolio.md#allocation-without-expected-returns)

---

## Positions and base currency (`standard_quant_tools.finance`)

Holdings valued in one base currency: the arithmetic that financing,
margin, currency exposure and carry on a set of positions start from. Pure
functions of their arguments — no price or rate is fetched.

| Function | Description |
|---|---|
| `position_values(positions, base_currency, fx_rates=None)` | Each line valued, in input order: local notional = quantity × price × multiplier in its own currency, base notional = local × its rate, and its side (the sign of the quantity) |
| `exposure_summary(positions, base_currency, fx_rates=None)` | Gross, net, long and short in the base currency, and per currency its net, gross, long, short, local net and share of gross; rates given for currencies not held are listed as unused |
| `currency_exposure(positions, base_currency, fx_rates=None, *, hedges=None, fx_returns=None)` | Exposure per currency in the base currency after FX hedges given as `{pair: notional of its first currency}`, each currency's hedge ratio, and — with each position's `local_return` and `fx_returns` — the local / FX / cross split of every position's base-currency return |
| `book_carry(positions, base_currency, fx_rates=None)` | Each line's annual carry and the set's: (S - F)/(F T) for a line with `spot` and `years_to_expiry`, income less financing (less `borrow_fee_bps` when short) for a line with `income_rate` and `financing_rate` |
| `fx_forward_hedge(*, pair, spot, days, base_currency, base_rate, foreign_rate, base_day_count, foreign_day_count, basis_bps=0.0, pip_size=None, start_date=None, asset_return=None, spot_at_maturity=None, hedge_ratio=1.0)` | The covered-interest-parity forward, its points, the hedge's annual pickup or cost and, with a period, the hedged and unhedged returns |
| `covered_interest_forward(spot, *, first_rate, first_years, second_rate, second_years)` | S (1 + r_second t_second) / (1 + r_first t_first), the growth ratio formed first so equal legs give S exactly |
| `currency_pair(value)` | A market quote's two ISO 4217 codes, 'EURUSD' or 'EUR/USD' to ('EUR', 'USD') |
| `currency_code(value, argument=...)` | An ISO 4217 code, upper-cased, or a refusal naming the nearest valid code |
| `exact_product(*factors)` | A product rounded once from its factors' exact values |

`positions` are objects or mappings with `instrument`, `quantity`, `price`,
`currency` and optionally `multiplier` (default 1). `fx_rates` gives units
of the base currency per ONE unit of each other currency held: with base
USD and 1 EUR = 1.10 USD, `{"EUR": 1.10}`. Every non-base currency held
needs a rate; the base's own rate, when given, is exactly 1.

Each notional is the correctly rounded product of its factors, so 3
contracts × 50 is the same bits as 150 × 1, and every total is `math.fsum`,
so reordering the positions moves no total by a bit. Long minus short is
net at any price. A negative price (crude settled at -37.63) gives a long a
negative notional, and long + short is then not gross.

Refused, naming the argument and the value: a zero or non-finite quantity,
a non-finite price, a multiplier that is not finite and positive, an
unknown currency (with the nearest code), a withdrawn one (with its
successor), a price in pence, cents or agorot ('GBp', 'GBX', 'ZAc', 'ILA'
— give it in the main unit), a missing rate, a rate that is not finite and
positive, a base rate other than 1, one instrument on two lines in one
currency (net them yourself), and more than 5,000 lines. The currency list
is ISO 4217 List One as of 2026-10-22, kept in the library.

A hedge in `currency_exposure` is a pair and a notional of its first
currency, positive bought; its second leg is the equal and opposite
base-currency value at the caller's rates, so the hedges sum to nothing in
the base currency and only move exposure between currencies. The return
split is r_L + r_FX + r_L x r_FX, its `math.fsum` the base return, and a
base-currency line has no FX or cross term. `book_carry` refuses a line
with both sets of carry terms, neither or half of one, a funded short with
no borrow fee and a short future with one. `fx_forward_hedge` adds
`basis_bps` to the foreign leg (the pair's currency that is not
`base_currency`); a pip is 0.0001, or 0.01 when the quote currency is JPY.

The tool input for the same thing is `Holdings` — `positions`,
`base_currency`, `fx_rates` — with a `Position` per line, in
`standard_quant_tools.agent.runtimes._positions`. A tool subclasses it or
takes one as a field, and its checks run when the input is built. A
`Position` may carry `borrow_fee_bps`, read only while the line is short;
it overrides the account's per-instrument and default borrow.

### Financing and margin on positions

| Function | Description |
|---|---|
| `financing_cost(positions, *, equity, financing, base_currency, fx_rates=None, days=None, start=None, end=None)` | Debit and credit interest, each short's borrow, the rebate on segregated proceeds, the net cost, the cash drag and the effective rate on gross and on equity, over `days` (scalar benchmark) or `start` to `end`, under a `FinancingSpec` (or its input model or a mapping of it) |
| `margin_requirement(positions, base_currency="USD", fx_rates=None, house_rates=None)` | Reg T initial and FINRA 4210 maintenance margin per line, each naming its rule, and the account's initial requirement (the greater of Reg T, maintenance and the $2,000 minimum equity) |
| `margin_runway(equity, *, equity_changes=None, initial_margin=None, maintenance_margin=None, positions=None, ..., returns=None, restore_to="initial", cure_days=15)` | A stress path walked day by day: the first maintenance call and its amount, the day forced deleveraging starts, and the least cash today that avoids every call |

`financing_cost` takes cash to be equity less the lines' net and charges
simple interest on today's balances: balance x sum(rate x days) / basis,
each period at the rate in force on its first date. A position's own
`borrow_fee_bps` overrides the spec's `borrow_bps` for its instrument,
which overrides `default_borrow_bps`; a short with none is refused, and a
short at a negative price is refused. A horizon in days needs a scalar
benchmark. With no spreads, a book k times its equity long returns
kR - (k - 1)f on its equity for every k.

`margin_requirement` reads, on each position, `kind` ('equity' by
default, 'etf', 'call', 'put'), `leverage` (an ETF's factor, 1 to 10) and
an option's `strike` and `underlying_price`; a plain position is a stock.
It refuses a negative price, a stock or ETF line with a multiplier other
than 1, a leverage factor off an ETF, an option without its terms, an
option on a leveraged ETF, and a base currency other than USD. Every
requirement is account equity, so the lines add up. `margin_runway`
compares equity with maintenance on each day: a call when equity is below
it, cured when equity reaches the call's target within `cure_days`, and
deleveraging on the deadline otherwise. A path of equity changes holds the
requirements fixed; a path of returns recomputes them (stocks and ETFs
only, FX held). The equity path is the exact sum of the changes rounded
once a day.

The tool inputs are `MarginPosition` (a `Position` with those fields) and
`MarginHoldings` (positions, base currency, FX rates, `house_rates`), in
`standard_quant_tools.agent.runtimes.finance.margin_models`.

---

## The time value of money (`standard_quant_tools.finance`)

| Function | Description |
|---|---|
| `periodic_cash_flows(flows, *, periods_per_year=1, compounding=None)` | A schedule of flows at regular periods: flow k at k / periods_per_year years, flow 0 now; compounding 'periodic' by default |
| `dated_cash_flows(flows, *, day_count, compounding=None, periods_per_year=1)` | A schedule of {date: amount} timed from the first date under a required day count; compounding 'annual' (XIRR's) by default |
| `analyze_cash_flows(schedule, *, rate=None, finance_rate=None, reinvest_rate=None)` | NPV, every real IRR (the IRR only when unique, with `irr_status`), Norstrom's condition, MIRR, Macaulay and modified duration, convexity and DV01 |
| `rate_roots(times, amounts)` | Every real root of sum amounts_i exp(-u times_i), as continuously compounded rates, with the method ('exact' Rolle isolation up to 20 sign changes, 'grid' beyond) and the Cauchy-bounded range searched |
| `estimate_cost_of_capital(*, risk_free_rate, beta, equity_risk_premium, equity_value, tax_rate, debt_value=0.0, cost_of_debt=None, beta_standard_error=None, target_debt_to_equity=None)` | CAPM cost of equity, after-tax cost of debt, WACC on market values, Hamada's unlevered beta, the relevered beta and WACC at a target D/E, and the sensitivity at beta -/+ one standard error |
| `unlever_beta(levered_beta, debt_to_equity, tax_rate)`, `relever_beta(...)` | Hamada's two formulas |
| `dcf_valuation(cash_flows, *, discount_rate=None, terminal_method="gordon", terminal_growth=None, exit_multiple=None, terminal_metric=None, target_value=None, solve_for=None, sensitivity_rates=None, sensitivity_terminal=None)` | A DCF with a Gordon or exit-multiple terminal value and its share, the reverse DCF for the discount rate, terminal growth or exit multiple a target implies, and the sensitivity grid |

Every rate is the caller's and a decimal; nothing is fetched. A number the
call did not ask for (an NPV without a rate, the other terminal method's
inputs) is None; one it asked for that is not defined (an IRR the flows
give no single value of, a duration of a zero present value) is NaN. An
IRR is a root of an exponential sum and can have none, one or several:
the search isolates every one inside bounds read from the amounts (the
module docstring of `finance.cash_flows` has the method) and reports the
IRR only when there is exactly one.

The agent tools over these are the `finance` runtime's; see
[32_finance.md](32_finance.md).

---

## Capital allocation (`standard_quant_tools.allocation`)

Capital across return streams rather than across tickers. Every function
takes dated return streams (aligned with
`portfolio.streams.align_return_streams`) or plain arrays, and fetches
nothing.

| Function | Description |
|---|---|
| `SleeveRule(method="risk_parity", covariance_method="ledoit_wolf", ...)` | An allocation rule and its constraints: `equal`, `inverse_volatility`, `risk_parity`, `hrp`, `min_variance`, `max_sharpe` or `shrunk_sharpe` on one of `estimate_covariance`'s estimators, then `max_weight`, `target_volatility` (up to `max_leverage`) and a `capacity` cap, in that order |
| `allocate_across_sleeves(frame, rule, *, cash_rate=0.0, periods_per_year=252, n_bootstrap=500, block_size=None, random_seed=0)` | The rule's weights, risk contributions and effective N on the aligned sleeve returns, with each weight's 5th and 95th percentile over a circular block bootstrap and the mean L1 distance of a resampled allocation from the point one |
| `fit_sleeve_rule(frame, rule, *, cash_rate=0.0, periods_per_year=252, capital=None)` | One fit of the rule, without the bootstrap |
| `evaluate_allocation_rule(frame, rule, *, n_trials, benchmark_rule="equal", estimation_window=252, rebalance_every=21, cost_bps=0.0, ...)` | The rule walked forward against a benchmark rule and against itself fitted on the full sample: out-of-sample Sharpe, return, volatility, drawdown, turnover and cost, the Sharpe difference with a block-bootstrap interval, the look-ahead gap and the deflated Sharpe ratio at `n_trials` |
| `walk_rule_forward(frame, rule, *, estimation_window, rebalance_every, cost_bps=0.0, cash_rate=0.0, ...)` | The out-of-sample path alone: each refit on the window before it, weights drifting between refits |
| `kelly_report(returns, *, periods_per_year, risk_free_rate=0.0, fractions=(0.25, 0.5, 1.0), drawdown_levels=(0.2, 0.5), ...)` | Kelly leverage f* = mu / sigma^2 (Sigma^-1 mu for several streams), its standard error, the chance it is more than twice the truth, Kan and Zhou's shrink factor, the empirical Kelly, and growth and drawdown odds at fractions of it |
| `apply_overlay(returns, *, method, periods_per_year, target_volatility=0.1, multiplier=3.0, floor=0.8, max_leverage=1.0, ...)` | A volatility target, CPPI or TIPP over one stream, each exposure decided from the returns before it: the overlaid stream, its exposure path, and the floor breaches |
| `capacity_curve(returns, weights, *, adv, daily_volatility, spread_bps, periods_per_year, impact_coefficient=1.0, aum=None, ...)` | Net return and Sharpe against AUM under spread plus square-root impact on every weight change, the break-even AUM, the AUM at half the gross Sharpe and the instruments whose impact binds |
| `brinson_fachler(periods, portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns, *, linking="carino")` | Allocation, selection and interaction per segment and period, linked by Carino or Menchero so they sum to the cumulative active return |

### Book returns

`measure_book_returns(valuations, flows=None, *, day_count="ACT/365F",
flows_in_valuation=False, large_flow_share=0.10)` returns the true
time-weighted return (NaN, with the dates named, when a flow date has no
valuation), Modified Dietz per sub-period linked geometrically, and the
money-weighted return — the IRR of the owner's cash by
`finance.cash_flows.rate_roots`, NaN unless unique — of a portfolio with
external flows (GIPS 2020). The annualized figures are None for a period
under a year; flows above `large_flow_share` of the value are listed.

The agent tools over these are the `allocation` runtime's; see
[31_allocation.md](31_allocation.md).
