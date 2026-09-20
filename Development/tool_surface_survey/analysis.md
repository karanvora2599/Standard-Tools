# Survey: `src/standard_quant_tools/analysis/` — implemented vs. exposed

Scope: the 21 modules under `analysis/` (10,850 lines, `__init__.py` included).
Tool surface checked by grep against every file under `agent/runtimes/*/`
(`tools.py` plus the split files `book_tools.py`, `estimator_tools.py`,
`event_tools.py`, `series_tools.py`, `diagnostic_tools.py`,
`inference_tools.py`, `construction_tools.py`, `weight_tools.py`,
`futures_tools.py`, `terminal_mc_tools.py`, `trade_tools.py`,
`validation_tools.py`), `modeling/agent/tools.py`,
`modeling/agent/feature_tools.py`, and the input/result models in
`agent/models.py` and `agent/runtimes/*/models.py`. Tool names are as they
appear in `Documentation/20_tool_index.md` (211 tools).

## Headline

| | Count |
|---|---:|
| Public functions / classes inventoried | **86** |
| Reached directly by at least one tool (the tool calls it and returns its result) | **69** |
| Reached only indirectly (called inside a composite a tool wraps; the standalone capability, a parameter, or an output field is hidden) | **14** |
| Never reached from any tool | **3** (`Channel`, `available_channels`, `declared_channels`) |
| Private helpers with real capability (listed separately) | 15 |

This slice is well covered at the function level: almost every public
estimator has a tool. The gaps are of a different kind, and they are where the
proposals below come from:

1. **Capabilities that exist only as a side effect of a composite.** `cusum`
   runs only inside `detect_liquidity_events`; `half_life` only inside
   `cointegration_test`; `variance_ratio` only at the fixed periods (2, 4, 8);
   `microprice` only as a window mean. An agent cannot ask the standalone
   question.
2. **Outputs the library computes and the tool result model drops.**
   `detect_regimes` produces per-observation `labels` and the tool discards
   them; `variance_ratio` reports `differencing` (level vs log) and the
   `VarianceRatio` result model has no such field; `andrews_bandwidth` is
   computed and never reported.
3. **Parameters the library takes and the tool does not pass.** `autolag`,
   `max_lag`, explicit `pairs` (cointegration); `slack` (CUSUM); `vr_periods`
   and KPSS `lags` (stationarity); `step` (rolling Hurst); `model` for implied
   vol (only Black-Scholes is invertible from the tool).
4. **Declared-but-empty capability.** `liquidity_events.CHANNELS` declares
   seven order-book channels with `compute=None` and a refusal text saying no
   provider serves a book, while `order_book.py` (same package) now computes
   exactly those quantities. The registry was never wired to it.

---

## 1. Inventory

Legend for "exposed by": tool names in `code`; **direct** = the tool calls this
function and shapes its result; **indirect** = reached only inside another
function; **none** = unreachable from any tool. Runtime in parentheses.

### 1.1 `_series.py` (87 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `_series` | `clean_series(series, name, func, *, minimum, as_array, note)` | One shared "is this series usable" gate: rejects +/-inf via `numeric_contract.require_finite_series`, drops NaN, enforces a per-caller minimum length. Replaced four divergent private `_clean` copies. | Series/sequence -> `pd.Series` or `np.ndarray`; raises `ValidationError` | **indirect** — every tool backed by `structure`, `stationarity`, `diagnostics`, `inference` passes through it. Not a tool candidate. |

### 1.2 `regression.py` (137 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `regression` | `calculate_beta(asset_returns, benchmark_returns)` | Static OLS alpha/beta/R2 on the shared index; NaN (not 0.0) when <2 overlapping obs. C++ `ols2` fast path. | two `pd.Series` -> `{alpha, beta, r_squared}` | **direct**: `analyze_stock_risk`, `get_extended_risk_metrics` (research) |
| `regression` | `rolling_beta(asset_returns, benchmark_returns, window=60)` | Rolling OLS beta; C++ incremental path, pandas fallback; zero-variance windows -> NaN. | two `pd.Series`, window -> `DataFrame[Rolling_Beta]` | **direct**: `get_rolling_beta` (research) |

### 1.3 `correlation.py` (145 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `correlation` | `diversification_ratio(returns_df, weights=None)` | Choueifaty-Coignard DR = sum(w*sigma) / sigma_p; equal-weight default; NaN if sigma_p=0. | wide returns frame, weights -> float | **direct**: `get_correlation_analysis` (research) |
| `correlation` | `pairwise_correlation_summary(returns_df)` | Full correlation matrix + mean pairwise corr + most/least correlated pair (vectorised upper-triangle). Lazy-imports `portfolio.correlation_matrix` to dodge an import cycle. | wide returns frame -> dict{correlation_matrix, avg, highest_pair, lowest_pair} | **direct**: `get_correlation_analysis` (research) |

### 1.4 `rally.py` (227 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `rally` | `_return_zscore(close, lookback, zscore_window)` (private, real capability) | Vol-normalised trailing-return z-score against its own rolling history. Docstring says it exists nowhere else in the codebase. | Close series -> z-score series | **indirect** via `detect_rally` |
| `rally` | `detect_rally(df, lookback, zscore_window, adx_period, adx_threshold, breakout_period, hurst_method, auto_tune_adx_threshold, auto_tune_percentile)` | Five-signal rally vote (return z>1, ADX>threshold, DI+>DI-, Hurst trending, Donchian new high); `is_rally` at >=3/5. Auto-tuned ADX threshold from the symbol's own ADX percentile. | OHLCV frame + params -> dict(is_rally, rally_score, signal components, adx_threshold_used, auto_tuned) | **direct**: `get_rally_signal` (research) — every parameter is exposed |

### 1.5 `multi_factor.py` (255 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `multi_factor` | `multi_factor_regression(asset_returns, factor_returns)` | OLS on N factors with intercept; t-stats, p-values (scipy t-dist or normal approx), R2/adj-R2. NaN block when n < k+2. **No HAC/Newey-West** standard errors. | Series + factor frame -> dict(alpha, loadings, t_stats, p_values, r_squared, adj_r_squared, n_obs) | **direct**: `run_factor_regression` (research), `get_portfolio_risk_attribution` (portfolio) |
| `multi_factor` | `rolling_factor_loadings(asset_returns, factor_returns, window=60)` | Rolling OLS loadings; C++ Cholesky path; all-NaN when window < k+2 or rank-deficient. | -> DataFrame[alpha, factor...] | **direct**: `run_factor_regression` when `rolling_window` is set |

### 1.6 `hurst.py` (301 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `hurst` | `hurst_exponent(series, method="dfa"/"rs", min_window, max_window)` | Hurst H via DFA or R/S with log-log OLS; regime label (trending >0.55, mean_reverting <0.45); C++ path; accepts polars. | return series -> dict(hurst, regime, fit_r_squared, method, n_obs) | **direct**: `run_hurst_analysis` (research), `run_regime_adaptive_backtest`, `run_regime_adaptive_walkforward_backtest` (backtest); indirect via `detect_rally` |
| `hurst` | `rolling_hurst(series, window=200, step=1, method, min_window)` | Rolling H; single C++ pass or Python loop. | -> `pd.Series` | **direct**: `run_hurst_analysis` (`rolling_window`; `step` NOT exposed); also a modeling feature (`modeling/features/statistical.py`) reachable via `build_model_dataset` |
| `hurst` | `_dfa`, `_rs`, `_ols_slope_r2`, `_classify`, `_log_sizes` | Python fallback kernels and the regime thresholds. | | indirect |

### 1.7 `garch.py` (308 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `garch` | `garch_volatility_forecast(returns, forecast_horizon=10, periods_per_year=252)` | GARCH(1,1), normal innovations, constant mean, L-BFGS-B MLE (C++ fused NLL+gradient or numba recursion). Returns params, persistence, convergence, LL/AIC/BIC, current/long-run/forecast annualised vol. **Discards** the conditional-variance path and standardised residuals; no residual diagnostics. | return series -> dict | **direct**: `run_garch_volatility_forecast` (research). `periods_per_year` NOT exposed (daily-only). |
| `garch` | `_garch11_variance_recursion[_numba]`, `_garch11_neg_loglik[_and_grad]`, `_require_scipy` | Kernels. The recursion produces the full sigma2_t path that the public function throws away after reading `sigma2[-1]`. | | indirect |

### 1.8 `order_events.py` (308 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `order_events` | `queue_positions(events)` | Size ahead of each new order at its (side, price) level, single pass with a running accumulator; CLEAR resets; MODIFY ignored. | event frame -> dict(n_adds, mean/median_queue_ahead, share_joining_empty) | **indirect** via `order_event_metrics` (its full output is included in the composite) |
| `order_events` | `order_lifetimes(events)` | Seconds from ADD to CANCEL/FILL; left-censored (no ADD) counted separately; right-censored count. | -> dict(filled{n,mean,median}, cancelled{...}, still_resting, terminated_without_an_add) | **indirect** via `order_event_metrics` |
| `order_events` | `event_rates(events)` | Events/second total and by action; cancel-to-add, cancel-to-trade; None (not 0) when no clock span. | -> dict | **indirect** via `order_event_metrics` |
| `order_events` | `order_event_metrics(events, *, name)` | Composite of the three above plus vocabulary/CLEAR/MODIFY/censoring warnings. | -> dict(n_events, queue, lifetimes, rates, warnings) | **direct**: `get_order_event_metrics` (microstructure) |
| `order_events` | constants `ORDER_EVENT_COLUMNS`, `ACTION_MEANINGS`, `ADD/CANCEL/MODIFY/FILL/TRADE/CLEAR`, `BID/ASK` | Canonical column and action vocabulary. | | used by `data/external_validation.py` (-> `validate_external_dataset`, data runtime) and `event_tools.py` |

### 1.9 `pca.py` (374 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `pca` | `_top_k_pc_power_iteration(arr, n_comp, tol, max_iter)` (private, real capability) | Top-k PCs by power iteration + Hotelling deflation without forming the covariance; fixed-seed start; residual-verified, returns `converged`. | centred matrix -> (vecs, vals, total_var, converged) | **indirect** via `pca_returns(method="power_iteration")` |
| `pca` | `pca_returns(returns_df, n_components, standardize=True, method="svd")` | PCA via SVD or power iteration (falls back to SVD when unverified); sign convention fixed; rejects inf and duplicate column names. Raises `ValueError` (not `ValidationError`) for n_obs<2 / bad method. | wide returns -> dict(explained_variance_ratio, cumulative, loadings, factor_returns, n_components, n_obs) | **direct**: `run_pca_analysis` (research; `standardize`, `method` exposed), `get_portfolio_risk_attribution` (portfolio) |
| `pca` | `factor_contributions(returns_df, n_components=3, pca_result=None)` | Marginal R2 of each PC per asset by sequential regression. | -> DataFrame(assets x PCs) | **direct**: `run_pca_analysis` |

### 1.10 `pricing.py` (387 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `pricing` | `price_option(*, spot, strike, time_to_expiry, volatility, risk_free_rate, option_type, model, dividend_yield, american, steps)` | One `PricingSpec`-style entry over four models: `black_scholes`, `black_76` (forward, rate only in discount, corrected rho sign), `bachelier` (normal; negative spot/strike allowed; vol is ABSOLUTE), `binomial` (CRR; only American-capable; lattice delta/gamma, no vega/rho). Magnitude bounds against exp overflow. | -> dict(price, delta, gamma, vega, rho, d1, d2, model[, american, steps, notes]) | **direct**: `get_option_pricing` (derivatives) — used for any `model != black_scholes` or `american=True`; the BS default path goes through `options.py` instead |
| `pricing` | `_validate`, `_rho`, `_black_scholes`, `_bachelier`, `_binomial` | Per-model kernels. | | indirect |
| `pricing` | `MODELS`, `AMERICAN_CAPABLE`, `DEFAULT_BINOMIAL_STEPS` | Registry constants. | | not surfaced by a listing tool; `OptionPricingInput.model` is a free `str`, validated inside |

### 1.11 `options.py` (405 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `options` | `black_scholes_price(spot, strike, T, r, sigma, option_type, q)` | BSM European price; bounded inputs; European only, T>0. | -> float | **direct**: `get_option_pricing` (default path); indirect in `implied_volatility` |
| `options` | `black_scholes_greeks(...)` | delta, gamma, vega (per 1.0 vol), theta (per year), rho, d1, d2 — raw units, scaled by the tool. | -> dict | **direct**: `get_option_pricing`; indirect in `implied_volatility` (vega for Newton) |
| `options` | `implied_volatility(option_price, spot, strike, T, r, option_type, q, initial_guess, tol, max_iterations)` | Newton-Raphson with bisection fallback on [1e-6, 5]; no-arbitrage bound check first. **BSM only** — cannot invert Black-76/Bachelier/binomial prices. | -> dict(implied_volatility, converged, iterations, method) | **direct**: `get_implied_volatility` (derivatives) |
| `options` | `_validate_option_inputs`, `_d1_d2` | | | indirect |

### 1.12 `microstructure.py` (438 lines) — tick data

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `microstructure` | `quoted_spread(quotes)` | Per-quote mid, spread, spread_bps, size imbalance; drops crossed/non-positive quotes. | quote frame (DatetimeIndex) -> DataFrame | **direct**: `get_quoted_spread_series` (microstructure); indirect via `sign_trades`, `effective_spread`, `microstructure_summary` |
| `microstructure` | `sign_trades(trades, quotes=None)` | Lee-Ready with strict-prior quote (`allow_exact_matches=False`), zero-tick rule; unclassifiable trades dropped not defaulted. | -> Series of +1/-1 | **direct**: `classify_trade_direction`; indirect via `effective_spread`, `microstructure_summary`, and `liquidity_events._signed_volume` |
| `microstructure` | `effective_spread(trades, quotes, realized_horizon=None)` | Per-trade effective spread bps; with horizon, realized spread and price impact decomposition. | -> DataFrame | **direct**: `get_effective_spread_series` (`realized_horizon_seconds` exposed); indirect via `get_microstructure_metrics`, `detect_liquidity_events` (effective_spread channel) |
| `microstructure` | `microstructure_summary(trades, quotes=None, realized_horizon=None)` | One-symbol size-weighted liquidity profile: VWAP, quoted/effective/realized/impact bps, buy-volume fraction. | -> dict | **direct**: `get_microstructure_metrics`, `check_spread_proxy` (both live in `portfolio/tools.py`, listed under the microstructure runtime) |
| `microstructure` | `trade_size_profile(trades, buckets=5)` | Volume share by trade-size quantile bucket. | -> dict | **direct**: `get_trade_profile` |
| `microstructure` | `intraday_volume_profile(trades, freq="30min")` | Volume share by time-of-day bucket from TRADES. **Same name** as the bar-based one in `microstructure_estimators`. | -> dict(buckets, peak_time, peak_volume_fraction) | **direct**: `get_trade_profile` |
| `microstructure` | `_require_frame`, `_classified` | | | indirect |

### 1.13 `stationarity.py` (456 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `stationarity` | `adf_statistic(values, lags=1)` | ADF t-stat with constant, hand-rolled OLS. | array -> float | **indirect** via `run_stationarity_tests` (`lags` exposed as the tool's only stationarity parameter) |
| `stationarity` | `andrews_bandwidth(residual)` | Andrews (1991) data-driven Bartlett bandwidth capped at Schwert l12 (measured: fixed rule over-rejects 23-40% on persistent AR(1)). | -> int | **indirect** via `kpss_statistic`; the bandwidth value is never reported |
| `stationarity` | `kpss_statistic(values, lags=None)` | KPSS level-stationarity statistic with automatic bandwidth. | -> float | **indirect** via `run_stationarity_tests`; `lags` override NOT exposed |
| `stationarity` | `variance_ratio(values, period=2)` | Lo-MacKinlay heteroskedasticity-robust VR; log differences only when all values >0, else level differences (spread-safe); returns `differencing`. | -> dict(variance_ratio, z_statistic, p_value, period, differencing) | **indirect** via `run_stationarity_tests` at fixed periods (2,4,8); `vr_periods` NOT exposed; **`differencing` is dropped by the `VarianceRatio` result model** |
| `stationarity` | `run_stationarity_tests(series, *, lags=1, vr_periods=(2,4,8))` | ADF + KPSS + VRs with the four-way verdict (stationary / non_stationary / inconclusive / contradictory) and warnings. | -> dict | **direct**: `run_stationarity_tests` (research; `on=price|returns`, `lags`) |
| `stationarity` | `detect_regimes(series, *, n_regimes=2, max_iterations=100)` | Gaussian-mixture EM (not HMM), quantile init (deterministic; the old seed was a no-op and was removed), regimes sorted by vol, persistence/switch count. | -> dict(labels[per obs], regimes, persistence, n_switches, current_regime, warnings) | **direct**: `detect_regimes` (research) — **`labels` dropped by `RegimeDetectionResult`** |
| `stationarity` | `_clean`, `_stationarity_warnings`, `_gaussian` | | | indirect |

### 1.14 `structure.py` (519 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `structure` | `detect_change_points(series, *, max_breaks=3, min_segment=20, penalty=None)` | Binary segmentation on mean shift with a series-scaled BIC penalty (k=3, measured 0/200 false breaks); refuses an absolute penalty no split can clear; reports gain per break and segment stats. | -> dict(breaks, segments, penalty, penalty_was_derived, warnings) | **direct**: `detect_change_points` (research; all params + `on`); also used by `delta_one/basis.py` -> `analyze_basis_history` |
| `structure` | `partial_correlation(frame, x, y, controlling_for)` | Residual correlation after regressing both on controls. | -> dict(raw, partial, explained_away) | **direct**: `get_partial_correlation` |
| `structure` | `granger_causality(cause, effect, *, max_lag=5)` | F-test per lag, closed-form F tail; Bonferroni-corrected flag (measured 15% at nominal 5% uncorrected). | -> dict(by_lag, best_lag, p_value, uncorrected_p_value, significant_at_05) | **direct**: `test_granger_causality` |
| `structure` | `tail_dependence(x, y, *, quantile=0.05)` | Empirical lower/upper tail co-exceedance probability with tail-count honesty. | -> dict | **direct**: `analyze_tail_dependence` |
| `structure` | `_best_split`, `_describe_segments`, `_change_point_warnings`, `_lagged_design` | Prefix-sum O(1) split search, etc. | | indirect |

### 1.15 `liquidity_events.py` (641 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `liquidity_events` | `Channel` (frozen dataclass) | name, requires, description, compute, refused_because; `.available`, `.why_unavailable()`. | | **none** (registry data) |
| `liquidity_events` | `CHANNELS` (dict of 15 `Channel`) | 6 computable from trades/quotes (`mid_return`, `spread`, `trade_intensity`, `signed_volume`, `realized_vol`, `effective_spread`) + `mid_price` (declared and refused: level CUSUM fires on drift) + **8 declared with `compute=None`** needing an order book (`microprice`, `book_imbalance`, `l5_imbalance`, `ofi`, `bid_depth`, `ask_depth`, `depth_slope`, `cancel_rate`). | | reached only when a name is looked up inside `detect_liquidity_events` |
| `liquidity_events` | `available_channels()` / `declared_channels()` | Sorted names of computable / all channels. | -> List[str] | **none** — no tool lists channels; an agent learns the vocabulary only from the `unknown channel` error text |
| `liquidity_events` | `cusum(series, *, slack=0.5, threshold=9.0, reference_fraction=0.3)` | Two-sided CUSUM standardised against a REFERENCE window (first 30%); threshold 9.0 calibrated to ~5% whole-window false alarm on iid noise (5.0 gave 36-82%); degenerate/constant baseline handling; reports first crossing, peak, direction, severity, `shift` in channel units. | any Series -> dict | **indirect** via `detect_liquidity_events` (`slack` NOT exposed) and `delta_one.basis.detect_basis_dislocation` -> `detect_basis_dislocation` (delta_one). **No standalone tool.** |
| `liquidity_events` | `detect_liquidity_events(*, channels, trades, quotes, freq, slack, threshold, reference_fraction)` | Runs CUSUM over requested channels; unavailable channels reported with reason, never dropped; cross-channel warnings. | -> dict(channels_run, unavailable, results, n_triggered, worst_channel, summary, warnings) | **direct**: `detect_liquidity_events` (implemented in `portfolio/tools.py`, listed under microstructure) |
| `liquidity_events` | `_mid_return`, `_spread`, `_trade_intensity`, `_signed_volume`, `_realized_vol`, `_effective_spread`, `_resample`, `_severity`, `_warnings` | Channel compute functions (each a real per-bucket series builder). | | indirect via the registry |
| `liquidity_events` | `DEFAULT_SLACK`, `DEFAULT_THRESHOLD`, `DEFAULT_REFERENCE_FRACTION`, `DEGENERATE_BASELINE_CV` | Calibrated constants; imported by `delta_one/streaming.py`. | | |

### 1.16 `cointegration.py` (718 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `cointegration` | `_degenerate_pair_reason(a, b)` (private, real capability) | Detects a constant leg or an exact affine pair (dual listing, ETF vs sole holding, duplicated column) so the two backends stop inventing opposite verdicts (C++ p=0.26 vs statsmodels p=0.0). | -> reason str or None | **indirect** in `cointegration_test` (raises) and `scan_cointegrated_pairs` (blank row) |
| `cointegration` | `cointegration_test(series_a, series_b, autolag="aic")` | Engle-Granger: OLS hedge ratio, ADF on residual with MacKinnon p-values (C++ `engle_granger` or statsmodels `coint`), half-life. | -> dict(cointegrated, hedge_ratio, adf_statistic, p_value, critical_values, half_life_days, n_obs) | **direct**: `run_cointegration_test`, `scan_pairs` (fallback path). `autolag` NOT exposed by either tool. |
| `cointegration` | `compute_spread(a, b, hedge_ratio=None)` | a - beta*b (OLS-fitted incl. intercept when None; raw when given). | -> Series | **direct**: `run_cointegration_test`, `scan_pairs`; also `delta_one` |
| `cointegration` | `half_life(spread)` | Discrete AR(1) half-life log(0.5)/log|phi| (bias table vs -ln2/b documented); inf when non-reverting; 0 when phi=0. | -> float | **indirect**: inside `cointegration_test` (statsmodels path only — the C++ path returns the kernel's own `half_life`; whether the kernel applies the same discrete correction is not verified here) and `delta_one.basis.basis_history` -> `analyze_basis_history`. **Not available for a Kalman spread or any caller-supplied series.** |
| `cointegration` | `spread_zscore(spread, window=None)` | Rolling or full-sample z-score; NaN on zero-variance windows; documents the look-ahead of `window=None`. | -> Series | **direct**: `run_cointegration_test`, `run_kalman_hedge_ratio`, `scan_pairs` (always rolling); `delta_one` |
| `cointegration` | `scan_cointegrated_pairs(prices, pairs=None, autolag="aic", max_lag=-1)` | Batch Engle-Granger in one native call (9.8 h -> 5 min at 2,000 names); common-index alignment; degenerate pairs blanked. Python fallback loops `cointegration_test` and leaves `intercept`=NaN, `optimal_lag`=0. | -> DataFrame[intercept, hedge_ratio, adf_statistic, optimal_lag, p_value, cv_*, half_life_days, n_obs, cointegrated] | **direct**: `scan_pairs` (long-running). `pairs`, `autolag`, `max_lag` NOT exposed; `intercept`/`optimal_lag` not surfaced. |
| `cointegration` | `kalman_hedge_ratio(a, b, delta=1e-4, observation_noise=1e-3, include_intercept=True)` | Random-walk-beta Kalman filter (1- or 2-state; numba or C++). | -> DataFrame[Hedge_Ratio, Intercept, Spread, Kalman_Gain] | **direct**: `run_kalman_hedge_ratio` (all params exposed). Result reports current beta, beta std, spread z; **no half-life of the Kalman spread, no drift/gain diagnostics beyond std** |
| `cointegration` | `_kalman_filter_1state`, `_kalman_filter_2state` | numba kernels (also return innovation path, discarded). | | indirect |

### 1.17 `inference.py` (883 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `inference` | `bootstrap_statistic(values, *, statistic, n_bootstrap, block_size, confidence, periods_per_year, seed)` | Block bootstrap CI (block = n^(1/3) default; IID warned) over a closed `STATISTICS` set; bias and contains-zero flags. | -> dict | **direct**: `get_bootstrap_interval` (research; all params) |
| `inference` | `compare_distributions(a, b, *, label_a, label_b)` | Two-sample KS (Kolmogorov asymptotic), moment shifts, p01 tail ratio, low-power warning. | -> dict | **direct**: `compare_distributions` |
| `inference` | `rolling_correlation_stability(a, b, *, window=63)` | Rolling corr range, sign flips, fraction within 0.2, joint-worst-decile "stress" correlation. | -> dict | **direct**: `get_correlation_stability` |
| `inference` | `decompose_returns(returns, *, periods_per_year)` | Arithmetic vs geometric, vol drag, total without best/worst 5 (by rank), win/loss profile. | -> dict | **direct**: `decompose_returns` |
| `inference` | `test_normality(values)` | Jarque-Bera with population moments, 3-sigma/4-sigma tail counts vs normal expectation. | -> dict | **direct**: `test_normality` |
| `inference` | `estimate_tail_index(values, *, tail, threshold_quantile)` | Hill estimator across five thresholds, SE alpha/sqrt(k), instability flag; documented 20-30% low bias on t-dist. | -> dict | **direct**: `estimate_tail_index` |
| `inference` | `_statistic`, `_block_indices` (pass-through shim to `_resampling.block_indices`), `_clean`; `STATISTICS`, `TRADING_DAYS` | | | indirect |

### 1.18 `diagnostics.py` (1,100 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `diagnostics` | `ljung_box(series, *, lags=None, squared=False)` | Joint autocorrelation test; per-lag rho with 2/sqrt(n) band; `squared=True` for vol clustering. | -> dict | **direct**: `test_autocorrelation` |
| `diagnostics` | `entropy_measures(series, *, n_bins=8, embedding=3)` | Normalised Shannon (binned) + Bandt-Pompe permutation entropy. | -> dict | **direct**: `get_entropy_measures` |
| `diagnostics` | `seasonality(returns, *, by="weekday"/"month"/"day_of_month")` | Joint one-way ANOVA F first; per-period Welch t with Bonferroni; needs DatetimeIndex. | -> dict | **direct**: `run_seasonality_analysis` |
| `diagnostics` | `rolling_sharpe_stability(returns, *, window=252, periods_per_year)` | Half-vs-half Sharpe test with Lo (2002) SE (calibrated 3% size, 62% power); rolling series descriptive only; block-fitted trend. | -> dict | **direct**: `get_sharpe_stability` |
| `diagnostics` | `drawdown_profile(returns, *, threshold=0.05, top_n=5)` | Every drawdown episode (start = first underwater bar — one bar later than `metrics.diagnostics.drawdown_periods`, pinned by test), recovery days, fraction underwater. | -> dict | **direct**: `get_drawdown_profile` |
| `diagnostics` | `lead_lag_matrix(returns, *, max_lag=3, min_correlation=0.1)` | One cross-correlation matmul per lag; Bonferroni against the full search size; leads with survivors. | -> dict | **direct**: `get_lead_lag_matrix` |
| `diagnostics` | `structural_break_test(series, break_index, *, regressor=None)` | Chow test at a KNOWN index; mean-shift or relationship (intercept+slope) form. | -> dict | **direct**: `test_structural_break` |
| `diagnostics` | `_lower_gamma`, `_chi2_sf` | Regularised incomplete gamma / chi-square tail — the one distribution tail still implemented locally after beta/F moved to `_special`. | | indirect (ljung_box) |
| `diagnostics` | `_sharpe_variance`, `_episode`, `_clean`; `WEEKDAYS`, `TRADING_DAYS` | | | indirect |

### 1.19 `microstructure_estimators.py` (1,181 lines) — bar data

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `microstructure_estimators` | `roll_spread(prices, *, window=None)` | Roll (1984) 2*sqrt(-cov); `significant` and `smallest_detectable_spread` (measured: zero-spread RW returned 10 bps noise); None not 0 on positive covariance; rolling median form. | -> dict | **direct**: `estimate_roll_spread` |
| `microstructure_estimators` | `require_tradeable_bars(high, low, caller)` | Refuse non-positive low or high<low (shared kernel guard). | -> None/raises | **indirect** (both CS shapes; `backtest.liquidity` -> `get_liquidity_metrics`, `check_spread_proxy`) |
| `microstructure_estimators` | `overnight_gap_shift(high_prev, low_prev, high, low)` | Per-bar shift to remove the overnight gap from the two-bar range. | arrays -> array | **indirect** via `corwin_schultz_pairs` |
| `microstructure_estimators` | `corwin_schultz_pairs(...)` | The one CS kernel (unfloored) shared by the dict and series shapes. | -> array/Series | **indirect** |
| `microstructure_estimators` | `corwin_schultz_spread(ohlc)` | Aggregate CS with `negative_fraction`, `n_gap_adjusted`, `raw_mean_bps`. | OHLC frame -> dict | **direct**: `estimate_corwin_schultz_spread` |
| `microstructure_estimators` | `amihud_illiquidity(ohlcv, *, window=21)` | Amihud x1e6 with current percentile vs own history and half-vs-half trend. | -> dict | **direct**: `get_amihud_illiquidity`. (`get_liquidity_metrics` uses the *other* Amihud in `backtest/liquidity.py` — see section 4.) |
| `microstructure_estimators` | `kyle_lambda(ohlcv, *, window=None)` | Delta-price on tick-rule-signed volume; R2; impact of 1% ADV in bps; rolling summary. | -> dict | **direct**: `estimate_kyle_lambda` |
| `microstructure_estimators` | `order_flow_imbalance(ohlcv, *, window=5)` | Tick-rule imbalance, next-day correlation, persistence on non-overlapping windows (overlapping artefact reported beside it). | -> dict | **direct**: `get_order_flow_imbalance` |
| `microstructure_estimators` | `estimate_vpin(ohlcv, *, n_buckets=50, window=50)` | Volume-bucket VPIN from bars (labelled as not the paper's VPIN; Andersen-Bondarenko critique stated). | -> dict | **direct**: `estimate_vpin` |
| `microstructure_estimators` | `intraday_volume_profile(bars, *, n_buckets=13)` | Bar-based U-shape profile over the observed minute span; refuses daily bars. | -> dict(profile, u_shaped, open/close/trough shares) | **direct**: `get_intraday_volume_profile` |
| `microstructure_estimators` | `implementation_shortfall(*, decision_price, arrival_price, fills, target_quantity, final_price, side)` | Perold decomposition: delay / impact / opportunity / fees, positive = cost. | -> dict | **direct**: `get_implementation_shortfall` |
| `microstructure_estimators` | `_require_columns`, `_enough`; `MIN_OBSERVATIONS` | | | indirect |

### 1.20 `derivatives.py` (1,475 lines)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `derivatives` | `option_greeks(*, spot, strike, T, vol, r, option_type, q)` | First- and second-order BS greeks (vanna, volga, charm, speed) with per-greek units. | -> dict | **direct**: `get_option_greeks`; indirect in `analyze_strategy`, `simulate_delta_hedge` |
| `derivatives` | `analyze_strategy(legs, *, spot, r, q, spot_range)` | Multi-leg payoff at expiry, numeric breakevens, aggregate greeks today, unbounded-edge flags. | -> dict | **direct**: `analyze_option_strategy` |
| `derivatives` | `fit_volatility_smile(strikes, ivs, *, forward, T)` | Quadratic in log-moneyness; Durrleman negative-density check; no extrapolation. | -> dict(atm_vol, skew, curvature, r_squared, arbitrage_violations, fitted) | **direct**: `fit_volatility_smile` |
| `derivatives` | `volatility_cone(prices, *, horizons, current_implied)` | Realised-vol percentiles per horizon with independent-window counts; IV percentile if supplied. | -> dict | **direct**: `get_volatility_cone` |
| `derivatives` | `analyze_vol_term_structure(implied_by_expiry)` | Forward vols between expiries; negative forward variance flagged as calendar arbitrage; contango/backwardation. | -> dict | **direct**: `analyze_vol_term_structure` |
| `derivatives` | `check_put_call_parity(...)` | Model-free parity residual in bps of strike; implied q and forward to diagnose cause. | -> dict | **direct**: `check_put_call_parity` |
| `derivatives` | `implied_forward_price(*, spot, T, r, q, borrow_rate)` | Carry forward with financing/dividend/borrow decomposed; product-exponent bound. | -> dict | **direct**: `get_implied_forward` |
| `derivatives` | `expected_move(*, spot, implied_vol, days, realized_moves)` | 1-sigma move and 0.8-sigma straddle approximation; historical exceedance if past moves given. | -> dict | **direct**: `get_expected_move` |
| `derivatives` | `simulate_delta_hedge(...)` | Short-option, discrete delta hedge under realised != implied vol; path dispersion, costs, continuous-hedge reference. | -> dict | **direct**: `simulate_delta_hedge` |
| `derivatives` | `option_risk_scenarios(...)` | Full revaluation grid over spot x vol shocks with time decay. | -> dict | **direct**: `get_option_risk_scenarios` |
| `derivatives` | `_positive`, `_bounded`, `_option_inputs`, `_bounded_exponent`, `_find_breakevens`, `_durrleman_violations`; `CONE_HORIZONS`, `MIN_SMILE_STRIKES`, `MAX_*` | Validation and kernels. `_positive` is imported by `delta_one/basis.py` (private cross-package import). | | indirect |

### 1.21 `order_book.py` (496 lines) — L2 snapshots

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| `order_book` | `microprice(bid_price, bid_size, ask_price, ask_size)` | Opposite-side-size-weighted touch price; NaN on empty book; scalar or array. | -> float/array | **indirect** via `book_metrics` (window MEAN only) |
| `order_book` | `book_metrics(book, *, levels=None)` | Per-snapshot spread, mid, microprice, microprice lean, touch and cumulative imbalance, depth slope (regression through origin), crossed-snapshot exclusion; refuses one-level books' slope. Averages everything over the window. | snapshot frame -> dict of means + warnings | **direct**: `get_order_book_metrics` (microstructure; `levels`) |
| `order_book` | `book_dynamics(book)` | Cont-Kukanov-Stoikov OFI at the touch between consecutive snapshots; update/mid-change rates. Per-pair contribution array is summed, not returned. | -> dict(ofi, ofi_per_update, ofi_per_second, updates_per_second, mid_changes...) | **direct**: `get_order_book_metrics` (`include_dynamics`) |
| `order_book` | `depth_profile(book, *, levels=None)` | Mean size and distance (bps) per level. | -> dict | **direct**: `get_order_book_metrics` (`include_profile`) |
| `order_book` | `_levels_present`, `_depth_slope`, `_mean` | | | indirect |

### 1.22 Private helpers carrying capability nothing public wraps on its own

| module | helper | what it can do that no tool asks for |
|---|---|---|
| `garch` | `_garch11_variance_recursion` | Full conditional-variance path sigma2_t (the fitted vol series) — thrown away after `sigma2[-1]`. |
| `cointegration` | `_kalman_filter_*state` | Innovation path e_t (prediction errors) — returned by the kernel, discarded by `kalman_hedge_ratio`. |
| `order_book` | `book_dynamics` internals | Per-pair OFI contribution series — summed before return. |
| `order_book` | `book_metrics` internals | Per-snapshot microprice / lean / imbalance arrays — averaged before return. |
| `liquidity_events` | `_mid_return` ... `_effective_spread` | Per-bucket channel series — consumed by CUSUM, never returned. |
| `stationarity` | `andrews_bandwidth` | The bandwidth actually used by KPSS — not reported. |
| `diagnostics` | `_chi2_sf` / `_lower_gamma` | Chi-square tail — a `_special` candidate. |

---

## 2. Exposure summary by module

| module | public | direct | indirect only | none |
|---|---:|---:|---:|---:|
| `_series` | 1 | 0 | 1 | 0 |
| `regression` | 2 | 2 | 0 | 0 |
| `correlation` | 2 | 2 | 0 | 0 |
| `rally` | 1 | 1 | 0 | 0 |
| `multi_factor` | 2 | 2 | 0 | 0 |
| `hurst` | 2 | 2 | 0 | 0 |
| `garch` | 1 | 1 | 0 | 0 |
| `order_events` | 4 | 1 | 3 | 0 |
| `pca` | 2 | 2 | 0 | 0 |
| `pricing` | 1 | 1 | 0 | 0 |
| `options` | 3 | 3 | 0 | 0 |
| `microstructure` | 6 | 6 | 0 | 0 |
| `stationarity` | 6 | 2 | 4 | 0 |
| `structure` | 4 | 4 | 0 | 0 |
| `liquidity_events` | 5 | 1 | 1 | 3 |
| `cointegration` | 6 | 5 | 1 | 0 |
| `inference` | 6 | 6 | 0 | 0 |
| `diagnostics` | 7 | 7 | 0 | 0 |
| `microstructure_estimators` | 11 | 8 | 3 | 0 |
| `derivatives` | 10 | 10 | 0 | 0 |
| `order_book` | 4 | 3 | 1 | 0 |
| **total** | **86** | **69** | **14** | **3** |

---

## 3. Unexposed capability, grouped by theme

### Theme A — Change detection on an arbitrary series
- `cusum` standalone (any series: rolling Sharpe, turnover, spread proxy, fill rate, factor loading, basis). Only reachable through the liquidity-channel registry or the delta-one basis monitor. `slack` never exposed.
- The CUSUM crossing date is the natural input to `structural_break_test`, which requires a KNOWN date — but a data-chosen date invalidates the Chow p-value (its own docstring says so). A combined tool must say this.

### Theme B — Mean-reversion speed as a first-class question
- `half_life` on a caller-supplied or Kalman spread; `variance_ratio` at chosen periods with the `differencing` field; `kpss_statistic(lags=...)`; `andrews_bandwidth` reported. Today "how fast does it revert" is only a by-product of `run_cointegration_test` and is absent from `run_kalman_hedge_ratio` entirely, so the tool that says the OLS ratio is stale cannot say what the corrected spread's half-life is.

### Theme C — Regime labels as data, not a summary
- `detect_regimes` computes per-observation labels; the tool returns only counts and persistence. Nothing downstream can condition a statistic on regime (regime-stratified Sharpe, drawdown, distribution comparison) from this definition of regime.

### Theme D — Fitted paths and residuals
- GARCH conditional-vol path and standardised residuals (fit diagnostics via `ljung_box(squared=True)` and `test_normality`, both already in the package).
- Kalman innovation series.
- Per-snapshot order-book series (microprice, lean, imbalance, per-pair OFI) — the docstrings make predictive claims ("touch imbalance predicts the next tick", "OFI is the quantity the literature regresses a price change on") that no tool lets an agent test, because only means leave the function.

### Theme E — Registry discovery
- `CHANNELS` / `available_channels` / `declared_channels` — the channel vocabulary is unlistable. Same for `pricing.MODELS` (which models exist, which are American-capable, what `volatility` means per model).
- The eight order-book channels are declared as unavailable "because no provider serves a book", while `order_book.py` implements the arithmetic. Wiring is missing, not math.

### Theme F — Parameters and fields the tools hide
Listed per tool in section 5.

---

## 4. Dead code, duplicates, docstring/code mismatches

1. **Two Amihud implementations that do not share a kernel.**
   `analysis.microstructure_estimators.amihud_illiquidity` (frame -> dict with percentile/trend; window default 21; filters non-positive close/volume rows) vs `backtest.liquidity.amihud_illiquidity` (two Series -> rolling Series; window default 20; non-positive volume -> NaN for that day). `get_amihud_illiquidity` (microstructure) uses the former; `get_liquidity_metrics` (portfolio) uses the latter. Two tools answer "Amihud for AAPL over this window" with numbers that differ in row filtering and default window, under the same name. Corwin-Schultz was deduplicated onto one kernel with a docstring explaining why both shapes remain; Amihud was not.
2. **Two functions named `intraday_volume_profile`** (`microstructure.py`: tick trades, `freq` string, time-of-day keys; `microstructure_estimators.py`: bars, `n_buckets` over the observed minute span, `u_shaped` flag). Both exposed (`get_trade_profile` vs `get_intraday_volume_profile`). Different outputs, same name, same package — a rename (`trade_volume_profile` / `bar_volume_profile`) would remove a real confusion risk for anyone importing from `analysis`.
3. **`scan_cointegrated_pairs` docstring promises `intercept` and `optimal_lag` columns; the pure-Python fallback fills NaN and 0** (`intercept is not exposed by cointegration_test`). On a build without `_sqt_core`, the result shape is honoured and the content is not.
4. **`cointegration_test` half-life on the C++ path** comes from the kernel (`raw["half_life"]`), on the statsmodels path from `half_life()`, whose docstring documents a deliberate switch from -ln2/b to the discrete log(0.5)/log(phi) formula with a bias table. Whether the C++ kernel applies the same correction is not established in this slice; if it does not, the two backends rank pairs differently — exactly the failure the `_degenerate_pair_reason` guard was added for. Worth a pinned test.
5. **`liquidity_events` L2 channels**: `Channel.why_unavailable` says "no provider in this library serves an order book yet"; `DataProvider.get_order_book` exists, `get_order_book_metrics` consumes it, and `order_book.py` computes microprice, imbalance, OFI and depth slope. The registry entries still have `compute=None`. The module docstring's "adding `depth_slope` is a table entry" is now the literal remaining work.
6. **`diagnostics._lower_gamma` / `_chi2_sf`** are the last locally-implemented distribution tail after `betainc`/`betacf`/`f_sf` were centralised in `_special`; the module header still says "computed from incomplete gamma and beta functions implemented here". Half true.
7. **`inference._block_indices`** is a one-line shim around `_resampling.block_indices`, kept with a comment explaining its history; nothing outside the module references it. Dead wrapper.
8. **Three `TRADING_DAYS` re-exports** (`inference`, `diagnostics`, `derivatives`) of `constants.TRADING_DAYS_PER_YEAR`, each "kept because imported by name".
9. **`microstructure_estimators.py` lines 74-75**: a dangling comment `#: Trading days per year, for annualizing anything that needs it.` with no constant under it.
10. **`delta_one/basis.py` imports `analysis.derivatives._positive`** (a private helper) — a cross-package private import; `delta_one/_numbers.py` already has `positive`.
11. **`pca_returns` raises `ValueError`** for n_obs<2 and for an unknown `method`, while every sibling raises `ValidationError`. A tool boundary that catches `ValidationError` for a structured refusal will surface these as a raw exception.
12. **`calculate_beta`** returns `r_squared=0.0` when `ss_tot == 0` in the numpy path, contradicting its own NaN-not-zero policy stated two paragraphs above.
13. **`multi_factor_regression`** has no HAC standard errors; p-values assume iid residuals. Not a bug, but every tool built on it should say so (currently `run_factor_regression` and `get_portfolio_risk_attribution` do not, from the input models inspected).
14. **`hurst_exponent` accepts polars; `rolling_hurst` does not** (calls `.dropna()` / `.index` on the input). The polars-support doc promise is asymmetric within one module.
15. **`detect_regimes`** — the `seed` parameter was removed from the library; the `RegimeDetectionInput` inspected has only `n_regimes`, so the tool is consistent, but any older description text mentioning a seed would be stale.
16. **Test coverage by name**: `clean_series`, `corwin_schultz_pairs`, `overnight_gap_shift` have zero direct test references; they are exercised only through callers. `andrews_bandwidth`, `kpss_statistic`, `available_channels`, `declared_channels`, `sign_trades`, `trade_size_profile` each have exactly one.

---

## 5. Where an existing tool should gain a parameter or field (no new tool)

| tool (runtime) | add | backing | why | effort |
|---|---|---|---|---|
| `run_stationarity_tests` (research) | input `vr_periods: List[int]`, `kpss_lags: Optional[int]`; result `differencing` on each `VarianceRatio`, `kpss_bandwidth` | `variance_ratio`, `kpss_statistic`, `andrews_bandwidth` | An agent testing a spread cannot currently see whether VR used level or log differences — the module measured VR(2) at 0.62 vs 0.86 on the same series depending on that choice. | S |
| `detect_regimes` (research) | result `labels` (or `regime_by_date`) | `detect_regimes["labels"]` | Without labels the regime cannot condition anything. | S |
| `run_kalman_hedge_ratio` (research) | result `half_life_days`, `hedge_ratio_start/end`, `hedge_ratio_drift_pct`, `mean_kalman_gain_last_n` | `half_life`, `kalman_hedge_ratio` columns | The tool exists to say the static ratio is stale; it should say how stale and whether the filter is still reacting. | S |
| `run_cointegration_test`, `scan_pairs` (research) | input `autolag: "aic"/"bic"`, `max_lag`; `scan_pairs` input `pairs: Optional[List[[a,b]]]`; result `intercept`, `optimal_lag` | `cointegration_test`, `scan_cointegrated_pairs` | Explicit pairs turns an O(N^2) scan into the agent's shortlist; lag choice changes p-values. | S |
| `detect_liquidity_events` (microstructure) | input `slack`; `channels` accepts `"all_available"` | `cusum`, `available_channels` | Slack is the sensitivity knob the docstring discusses; "all available" avoids the refusal round-trip. | S |
| `run_hurst_analysis` (research) | input `step` | `rolling_hurst(step=)` | Rolling H at step=1 over years is 30-100x the work of step=5 for the same picture. | S |
| `run_garch_volatility_forecast` (research) | input `periods_per_year`, `include_path: bool`; result `conditional_vol_path` (last N), `variance_half_life_bars` = ln0.5/ln(alpha+beta) | `garch_volatility_forecast` + a small library change to return `sigma2` | Intraday bars are currently annualised as daily; the fitted path is the thing a risk model consumes. | M |
| `get_implied_volatility` (derivatives) | input `model` (`black_76`, `bachelier`, `binomial`) | generic bisection over `price_option` (monotone in vol) | `get_option_pricing` prices four models; `get_implied_volatility` inverts one. | M |
| `get_order_book_metrics` (microstructure) | input `include_series: bool` -> publishes an `sqt://` series artifact of per-snapshot microprice/lean/imbalance/OFI | `book_metrics`, `book_dynamics` internals refactored to return arrays | See proposal 6.6. | M |
| `get_liquidity_metrics` (portfolio) | switch backing to `analysis.microstructure_estimators` (percentile, `negative_fraction`) | | Removes the duplicate Amihud and gives the tool the honesty fields the other tool already has. | S |
| `get_option_pricing` (derivatives) | `model` as a `Literal` from `pricing.MODELS`; result echoes `volatility_convention` (relative vs absolute) | `MODELS`, `AMERICAN_CAPABLE` | The Bachelier vol-unit trap is documented in the library and invisible at the tool. | S |

---

## 6. Proposed new tools

Ordering is by value. "Decision" states the question an agent asks that the
tool answers, per the repo rule that a tool earns its place by being a
decision rather than plumbing (`Documentation/15_modeling.md`, the paragraph
ending "...plumbing. Choosing features is a decision").

### 6.1 `assess_mean_reversion` — research — **M**
- **Question**: How fast does this spread (or any series) revert, is that speed measured or assumed, and what z-score window / holding period does it justify?
- **Inputs**: one of (a) `values` + optional `dates`; (b) `symbol_a`, `symbol_b`, `start_date`, `end_date`, `hedge: "ols" | "kalman" | float`, plus `delta`/`observation_noise` for Kalman; (c) an `sqt://` series reference. Options: `vr_periods` (default 2,4,8,16), `adf_lags`, `kpss_lags`.
- **Backing**: `compute_spread` / `kalman_hedge_ratio` (Spread column), `half_life`, `adf_statistic`, `kpss_statistic` + `andrews_bandwidth`, `variance_ratio` (with `differencing`), `run_stationarity_tests` verdict logic, `spread_zscore` for a suggested window.
- **Outputs**: `half_life_bars`, `phi`, `reverting: bool`, `half_life_within_sample: bool` (false when half-life > n/4), stationarity verdict, VR table with differencing, `suggested_zscore_window` (about 2-3 half-lives, clipped), for Kalman: `half_life_ols` vs `half_life_kalman` and hedge-ratio drift.
- **Why a decision**: holding period and z-window sizing hinge on this number; today it is a side field of one tool and absent from the Kalman tool.
- **Warnings the tool must state**: discrete AR(1) half-life (not -ln2/b); `inf` means "did not revert in-sample", not "slow"; an OLS spread is fitted on the full window (in-sample; the z-scores are look-ahead unless rolling); the Kalman spread is filtered (causal) but `delta` is a tuning knob that sets the answer; VR on a level-crossing spread uses level differences; sample floor — a half-life longer than a quarter of the sample is not a measurement.

### 6.2 `detect_level_shift` — research — **S**
- **Question**: Did this series' level change, when, in which direction, and by how much in its own units?
- **Inputs**: `values` + optional `dates`, or an `sqt://` series / backtest-artifact metric reference (`rolling_sharpe`, `turnover`, `exposure`...); `slack`, `threshold`, `reference_fraction`.
- **Backing**: `cusum`.
- **Outputs**: `triggered`, `first_crossing`, `peak_at`, `direction`, `severity`, `shift`, `shift_in_reference_sd`, `degenerate_baseline`, `n_reference`.
- **Why a decision**: "has the edge / cost / exposure moved" on any monitored series, not only tick liquidity channels; it is the pre-question to `test_structural_break` and `detect_change_points`.
- **Warnings**: detection lag by construction; the reference window must precede the shock (a shock inside it hides itself); threshold 9.0 is calibrated to ~5% whole-window false alarms on iid noise — autocorrelated series alarm more often; `shift`, not `peak_statistic`, is the size; **the crossing date is data-chosen, so passing it to `test_structural_break` does not yield a valid Chow p-value** (state this explicitly, since the two tools will be used together).

### 6.3 `get_regime_conditioned_stats` — research — **M**
- **Question**: Does this return series (a strategy's P&L, a factor) have its edge in the calm regime, the stressed regime, or only on average?
- **Inputs**: `regime_series` (values+dates or symbol) with `n_regimes`; `target_returns` (values+dates, backtest artifact ref, or the same series); `statistic` from `STATISTICS`; `n_bootstrap`.
- **Backing**: `detect_regimes` (labels), then per-regime `decompose_returns`, `bootstrap_statistic`, `drawdown_profile`; `compare_distributions` between the calm and the most volatile regime.
- **Outputs**: per-regime n, geometric return, bootstrap CI of the statistic, max drawdown; KS comparison calm-vs-stressed; `edge_survives_stress: bool` (CI excludes zero in the top-vol regime).
- **Why a decision**: `get_regime_stratified_performance` (backtest) stratifies by its own regime definition on a backtest artifact only; this works on any series and any regime definition the mixture produces. Cheaper first step: the `labels` field in section 5.
- **Warnings**: mixture labels are fitted on the FULL sample — the label at t uses data after t (look-ahead; fine for description, not for a signal); the mixture flips on single observations (report `persistence`); regime samples are non-contiguous so a block bootstrap within a regime preserves less serial structure than it claims; regime 0 = calm by sorting, not by economics; small-regime CIs will be wide.

### 6.4 `check_garch_fit` — research — **M** (or fold into `run_garch_volatility_forecast` via `include_diagnostics`)
- **Question**: Can this GARCH forecast be trusted — did the model absorb the volatility clustering, and are normal innovations adequate?
- **Inputs**: as `run_garch_volatility_forecast`, plus `ljung_box_lags`.
- **Backing**: `garch_volatility_forecast` + `_garch11_variance_recursion` (library change: return `sigma2`), `ljung_box(squared=True)` on standardised residuals, `test_normality` on standardised residuals.
- **Outputs**: `converged`, `persistence`, `variance_half_life_bars`, `standardized_residual_ljung_box_p` (should be > 0.05), `standardized_excess_kurtosis`, `tail_ratio_3_sigma`, `innovation_assumption_adequate: bool`, `conditional_vol_path` (last N).
- **Why a decision**: use the forecast, or fall back to a realised-vol estimator. The library states EGARCH/GJR/Student-t are not built; this tool tells the agent when that matters.
- **Warnings**: GARCH(1,1) normal only; `long_run_annualized_vol` is a clamp artefact when `converged=False`; residual whiteness on squares is necessary not sufficient; the 100-observation floor is a minimum, not a comfortable sample.

### 6.5 `list_liquidity_channels` — microstructure — **S**
- **Question**: Which liquidity channels can I run with the data I have (trades only / quotes only / book), and what does each measure?
- **Inputs**: optional `have: List["trades","quotes","orderbook"]`.
- **Backing**: `CHANNELS`, `available_channels`, `declared_channels`, `Channel.requires/description/why_unavailable`.
- **Outputs**: rows of `{name, requires, description, available, runnable_with_inputs, reason_if_not}`.
- **Why a decision**: choosing channels before a fetch; today the vocabulary is only discoverable through a refusal. Alternative with no new tool: `channels="all_available"` on `detect_liquidity_events` (section 5). Pair with `describe_data_capabilities` (meta) so the agent can match channels to provider capability.
- **Warnings**: eight channels are declared-not-computable pending the order-book wiring (section 4.5); `mid_price` is refused by design.

### 6.6 `get_order_book_series` — microstructure — **M**
- **Question**: Does book imbalance / microprice lean / OFI lead the mid on this feed — i.e. is the book informative enough to build a signal on?
- **Inputs**: `ref` (`sqt://order_book/...`) or `snapshots`, `levels`, `run_id`, `name`.
- **Backing**: `microprice`, `book_metrics` per-snapshot arrays, `book_dynamics` per-pair contribution (library refactor to return series), then publish an `sqt://` series artifact and optionally run `test_granger_causality` / `get_lead_lag_matrix` between `ofi` and `mid_return` inside the same call.
- **Outputs**: artifact ref plus `ofi_leads_mid_p_value` (Bonferroni), `microprice_lean_autocorr`, counts of crossed/dropped snapshots.
- **Why a decision**: the module docstrings assert predictive relationships; the tool surface returns only window means, so the assertion cannot be checked. This also supplies the compute functions for the eight declared L2 channels in `liquidity_events` (the series are exactly what CUSUM needs).
- **Warnings**: snapshot frequency may be the sampling rate not the update rate (rates reported beside counts); OFI is touch-level only; crossed snapshots are dropped, not interpolated; Granger on tick-frequency series with thousands of points will "reject" on trivial effects — report the correlation size beside the p-value.

### 6.7 `audit_pair_relationship` — research — **M** (mostly composition of exposed pieces; included because it is the decision the individual tools circle)
- **Question**: Is this pair's relationship stable enough to trade, and where did it break?
- **Inputs**: `symbol_a`, `symbol_b`, dates, optional `known_break_date`.
- **Backing**: `cointegration_test`, `half_life` (via 6.1), `kalman_hedge_ratio` (drift of beta), `rolling_correlation_stability`, `tail_dependence`, `detect_change_points` on the spread, `structural_break_test(regressor=b)` only when `known_break_date` is supplied.
- **Outputs**: one verdict (`tradeable`, `stale_ratio`, `broken`, `insufficient_data`) with the component evidence and the date of the strongest spread break.
- **Warnings**: change-point dates are searched, so no Chow p-value is reported for them; the cointegration p-value is in-sample; correlation stability windows overlap (independent count reported); tail dependence at 5% on a year of data is about 12 points.

### 6.8 `describe_pricing_models` — derivatives — **S**
- **Question**: Which option model should I use for this underlying (can it go negative? is it a future? American?) and what does `volatility` mean for it?
- **Backing**: `pricing.MODELS`, `AMERICAN_CAPABLE`, `DEFAULT_BINOMIAL_STEPS`, the per-model docstrings.
- **Outputs**: table `{model, underlying_type, allows_negative, american_capable, volatility_convention, greeks_returned}`.
- **Why a decision**: the Bachelier absolute-vol trap and the Black-76 double-carry trap are documented in the library and invisible from the tool schema (`model: str`). Could be folded into `describe_tool(get_option_pricing)` if the meta runtime allows tool-specific enrichments.

---

## 7. Effort summary

| proposal | effort | needs library change |
|---|---|---|
| 6.1 `assess_mean_reversion` | M | no (composition) |
| 6.2 `detect_level_shift` | S | no |
| 6.3 `get_regime_conditioned_stats` | M | no (`labels` already returned) |
| 6.4 `check_garch_fit` | M | yes — return `sigma2` path |
| 6.5 `list_liquidity_channels` | S | no |
| 6.6 `get_order_book_series` | M | yes — return per-snapshot arrays |
| 6.7 `audit_pair_relationship` | M | no |
| 6.8 `describe_pricing_models` | S | no |
| section 5 parameter/field additions | S each (two M) | mostly no |
| Wire L2 channels in `liquidity_events.CHANNELS` to `order_book` | M | yes (depends on the 6.6 refactor) |
| Deduplicate Amihud onto one kernel | S | yes |
