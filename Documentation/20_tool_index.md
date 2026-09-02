# Tool index

Every tool in the library, by runtime, with the description the model
actually sees. **Generated from the live registry** by
`Development/generate_tool_index.py` -- a test regenerates it and fails if
this file has drifted, so a tool added without regenerating breaks the
suite in the commit that added it.

The descriptions here are not a parallel prose layer. They are the exact
strings a model reads when choosing a tool, which means a description that
reads badly here reads badly to the model too, and the fix belongs in the
tool definition.

## How to read this

A **runtime** is an execution boundary, not a category. Each holds its own
dispatch table, so a tool from another runtime is *unroutable* rather than
discouraged -- and the refusal names the runtime that owns it, so a scoping
mistake is recoverable and a hallucinated name is not mistaken for one.
See [19_runtimes.md](19_runtimes.md) for why, and how results still cross
between them.

A **category** narrows *within* a runtime. `--categories microstructure` is
not the same as `--runtime microstructure`, and the difference matters when
scoping an MCP session -- see [18_mcp.md](18_mcp.md).

Two tools (`run_backtest_optimization`, `scan_pairs`) are long-running and
are served only with `--enable-long-running`, so a default MCP session
advertises 155 of the 198 below.


## The runtimes

| Runtime | Tools | Schema cost | Categories | Deep documentation |
|---|---:|---:|---|---|
| `research` | 42 | 47 KB | `screener`, `analysis`, `quant_research` | [08_analysis.md](08_analysis.md), [23_inference.md](23_inference.md) |
| `backtest` | 34 | 76 KB | `backtest_execution`, `backtest_validation`, `custom_signal` | [04_backtesting.md](04_backtesting.md), [24_overfitting.md](24_overfitting.md) |
| `meta` | 19 | 14 KB | `discovery`, `provenance` | [27_meta.md](27_meta.md), [10_auditability.md](10_auditability.md) |
| `portfolio` | 18 | 31 KB | `portfolio_risk` | [05_portfolio.md](05_portfolio.md) |
| `delta_one` | 17 | 35 KB | *(one surface)* | [28_delta_one.md](28_delta_one.md) |
| `modeling` | 17 | 46 KB | *(one surface)* | [15_modeling.md](15_modeling.md) |
| `microstructure` | 16 | 20 KB | *(one surface)* | [22_microstructure.md](22_microstructure.md) |
| `data` | 14 | 15 KB | *(one surface)* | [26_data.md](26_data.md) |
| `derivatives` | 12 | 17 KB | *(one surface)* | [21_derivatives.md](21_derivatives.md) |
| `feature_lab` | 9 | 11 KB | *(one surface)* | [15_modeling.md](15_modeling.md) |
| **Total** | **198** | | | |

---

## `research` — Research

Describe an asset or a universe: screen it, profile its risk and technicals, and analyze its statistical structure (factors, cointegration, PCA, Hurst, correlation). Does not run strategies.

### `analysis`

#### `analyze_stock_risk`

Full risk profile of one asset against a benchmark: alpha, beta, Sharpe, VaR and CVaR in a single call. The starting point for 'what is this thing like', and the place most analyses begin. Every number here is a point estimate over the whole sample -- use get_bootstrap_interval for what the Sharpe's error bar actually is, and get_sharpe_stability to check the edge did not decay inside the window.

**Required:** `symbol`  
**Optional:** `benchmark`, `period`, `risk_free_rate`

#### `calculate_series_metrics`

Risk and return metrics for ANY return series -- a symbol, an `sqt://` reference from another runtime, or values passed inline. The same arithmetic analyze_stock_risk applies to a ticker, available for a model's out-of-sample returns, an external fund's series, or a panel another agent already computed. The metric set is closed rather than open, because this surface is reachable from an agent and an arbitrary-expression argument would be a code path wearing a statistics costume.

**Required:** `series`  
**Optional:** `metrics`, `risk_free_rate`, `periods_per_year`

#### `compute_indicator_panel`

Indicator HISTORY for a whole universe, published one `sqt://` reference per indicator. get_technical_panel answers what the indicators are NOW; this answers what they have been, which is what a signal, a feature or a custom backtest actually consumes. Pass a price_panel_ref from the data runtime and nothing is refetched -- the same bars are reused.

**Required:** `tickers`, `start_date`, `end_date`, `indicators`, `run_id`, `name`  
**Optional:** `price_panel_ref`

#### `get_advanced_indicators`

Compute Parabolic SAR (trend), Wilder ATR (volatility), and MFI (volume-flow oscillator).

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `mfi_period`, `atr_period`, `sar_af_start`, `sar_af_step`, `sar_af_max`

#### `get_data_quality_report`

Dataset provenance (adjusted/survivorship-free/point-in-time guarantees) plus missing-bar/stale-price/price-jump detection on a symbol's OHLCV.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `stale_run_length`, `jump_threshold`

#### `get_extended_risk_metrics`

Extended risk: Calmar ratio, Treynor ratio, parametric VaR 95/99, historical VaR 99, CVaR 99.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `benchmark`

#### `get_portfolio_analysis`

Risk and return metrics for a basket held at fixed weights: portfolio volatility, correlation structure, and contribution by position. Describes a portfolio you specify rather than choosing one -- run_portfolio_optimization and optimize_risk_parity choose, and get_marginal_risk_contribution answers the follow-up question of where the risk in this basket actually comes from.

**Required:** `tickers`, `weights`, `start_date`, `end_date`  
**Optional:** `benchmark`, `risk_free_rate`

#### `get_rally_signal`

Detect a rally via 5 confirming signals: return z-score, ADX trend strength, DI+/DI- direction, Hurst trending regime, and new-high breakout.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `lookback`, `zscore_window`, `adx_period`, `adx_threshold`, `breakout_period`, `hurst_method`, `auto_tune_adx_threshold`, `auto_tune_percentile`

#### `get_rolling_beta`

Compute rolling OLS beta to detect beta drift over time vs a benchmark.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `benchmark`, `window`

#### `get_tail_risk_metrics`

Extreme Value Theory tail risk (Peaks-Over-Threshold GPD fit): VaR/CVaR extrapolated from the fitted tail, compared against the naive historical quantile.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `confidence`, `tail_fraction`, `method`

#### `get_technical_analysis`

Technical indicators for ONE ticker, with the parameters you choose -- RSI, MACD, Bollinger, moving averages and the rest. Use get_technical_panel instead when the question spans a universe: it computes the same indicators for every ticker in one native call, and looping this tool per ticker is the slow way to the same answer.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `indicators`

#### `get_technical_panel`

Indicators (RSI/ADX/ATR/Bollinger/Stochastic) for a whole ticker universe in one native call, reported at the latest bar. Use instead of one get_technical_analysis call per ticker when screening.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `indicators`, `rsi_period`, `adx_period`, `atr_period`, `bollinger_period`, `bollinger_num_std`, `stoch_k_period`, `stoch_d_period`, `persist_run_id`

#### `get_volatility_estimators`

Realized volatility via Parkinson, Garman-Klass, and Yang-Zhang estimators vs. plain close-to-close.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `period`

#### `run_garch_volatility_forecast`

GARCH(1,1) conditional volatility: fits how variance evolves over time and forecasts it forward, unlike get_volatility_estimators' backward-looking realized estimates.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `forecast_horizon`

### `quant_research`

#### `analyze_tail_dependence`

Whether two assets move together IN THE TAIL, which is the only regime a diversification claim has to survive. A full-sample correlation of 0.3 is compatible with two assets that are independent day to day and fall together every time it matters. Read n_tail_observations alongside the estimate: at a 1% quantile on a year of data that is two or three points.

**Required:** `x`, `y`, `start_date`, `end_date`  
**Optional:** `quantile`

#### `compare_distributions`

Whether two samples came from the same distribution -- not whether their means differ. In-sample against out-of-sample, this regime against that one, live against backtest: the question gets answered with a t-test, and a t-test misses every difference in SHAPE. A strategy whose out-of-sample mean matches and whose kurtosis has tripled is not performing as expected. Returns which MOMENT moved, and reports the tail separately because KS's power is concentrated near the median -- a normal against a t(3) gives KS p=0.45 while the 1st percentile has moved by a factor of two.

**Required:** `sample_a`, `sample_b`  
**Optional:** `label_a`, `label_b`

#### `decompose_returns`

Where compound growth actually came from. THE ARITHMETIC MEAN IS NOT WHAT YOU EARNED: compound growth is the arithmetic mean minus roughly half the variance, and for a volatile strategy that drag is most of the return -- 0.08% a day at 3% daily vol is 20% arithmetic and 10% compound. Also separates the contribution of the best and worst five days, because a strategy whose entire return disappears when five days are removed is a lottery ticket with good statistics.

**Required:** `returns`  
**Optional:** `periods_per_year`

#### `detect_change_points`

When the process generating a series CHANGED, by binary segmentation on the mean. run_hurst_analysis says what KIND of process a series is; this says when it stopped being that one, which the first cannot -- a single Hurst exponent over a sample containing a break describes neither regime. Read `gain` on each break: a marginal call then looks marginal instead of looking like a boundary.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `max_breaks`, `min_segment`, `penalty`, `on`

#### `detect_regimes`

Label each observation with a volatility regime, by a Gaussian mixture. A MIXTURE rather than a hidden Markov model: it has no transition matrix, so it flips on single observations where an HMM would smooth, and `persistence` reports how often it does -- below about 0.8 the labels describe noise. Regimes come back sorted by volatility so regime 0 is always the calm one.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `n_regimes`, `seed`

#### `estimate_tail_index`

How fat the tail is, by the Hill estimator. Alpha says which moments EXIST: below 4 the kurtosis is infinite, so a sample kurtosis is an artefact of the sample size; below 2 the variance is infinite and every volatility, Sharpe and correlation on the series is meaningless. The threshold choice is the whole problem and has no good answer, so alpha is reported across several -- if it swings, the threshold is doing the work and there is no tail index to report. Measured as biased LOW (2.39 for a true 3.0, 3.48 for a true 5.0), so read it as a lower bound on tail thinness rather than as a measurement.

**Required:** `values`  
**Optional:** `tail`, `threshold_quantile`

#### `get_bootstrap_interval`

A confidence interval for a statistic, by BLOCK bootstrap. The point estimate is usually reported alone and usually should not be: a Sharpe of 1.2 on two years of daily data has a 95% interval from about 0.2 to 2.2, consistent with a mediocre strategy and an excellent one. Blocked rather than IID because resampling individual returns destroys serial correlation -- measured on AR(1) returns at phi=0.8, the IID interval is 2.24x too narrow for the Sharpe and 1.63x for maximum drawdown, while at phi=0 the two agree so the correction costs nothing when it is not needed.

**Required:** `values`  
**Optional:** `statistic`, `n_bootstrap`, `block_size`, `confidence`, `periods_per_year`, `seed`

#### `get_correlation_analysis`

Correlation matrix, avg pairwise correlation, most/least correlated pair, and diversification ratio for a universe.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `weights`

#### `get_correlation_stability`

Whether a correlation is a property of the pair or an average over two different regimes. Two assets correlating at 0.0 over ten years may have correlated at +0.7 for five and -0.7 for five; the average is meaningless and a hedge sized on it is wrong in both regimes. Reports the sign-flip count, the range, and separately the correlation conditional on the joint worst decile -- because correlations move toward 1 when everything falls together, so a hedge computed on a calm sample fails precisely when it is needed.

**Required:** `a`, `b`  
**Optional:** `window`

#### `get_drawdown_profile`

Every drawdown, not just the worst. Maximum drawdown is one number describing one event and it says nothing about how often drawdowns happen, how long they last, or whether the worst was a one-day gap or a two-year grind -- and those determine whether a strategy is holdable far more than depth does. A 20% drawdown recovering in a month is survivable; one taking three years ends the mandate. Depth and duration are close to independent and both are reported.

**Required:** `returns`  
**Optional:** `dates`, `threshold`, `top_n`

#### `get_entropy_measures`

How predictable a series is WITHOUT assuming the predictability is linear. Every other statistical tool here -- autocorrelation, regression, Granger -- measures linear dependence and returns nothing on a series that is perfectly deterministic in a nonlinear way. Permutation entropy reads only the RANK ORDER inside each window, so it is invariant to any monotone transformation and robust to outliers; a value near 1.0 is indistinguishable from random.

**Required:** `series`  
**Optional:** `n_bins`, `embedding`

#### `get_lead_lag_matrix`

Which series move first across a universe -- and why the answer is usually noise. Twenty assets at three lags is 1,140 correlations, of which about 57 clear an uncorrected 5% bar on data with NO lead-lag structure at all, and the strongest of those looks entirely convincing. Every pair carries a Bonferroni-corrected p-value against the full search size, and when nothing survives the result says so rather than presenting the top of the ranked list.

**Required:** `returns`  
**Optional:** `max_lag`, `min_correlation`

#### `get_partial_correlation`

The correlation between two assets once the common drivers are removed from BOTH. Two stocks in one sector correlate at 0.7 and it says almost nothing; take out the market and the sector and what is left is the part actually about those two companies. That residual is what a pair trade lives on, and the raw correlation systematically overstates it.

**Required:** `x`, `y`, `controlling_for`, `start_date`, `end_date`

#### `get_sharpe_stability`

Whether the edge DECAYED, or the full-sample Sharpe is the average of a good period and a dead one. A Sharpe of 1.0 made of 2.0 in the first half and 0.0 in the second is arithmetically correct and describes a dead strategy -- and the second half is the half that predicts tomorrow. The p-value comes from comparing two NON-OVERLAPPING halves, not from a regression on the rolling series, because consecutive rolling windows share all but one observation and cannot support inference.

**Required:** `returns`  
**Optional:** `window`, `periods_per_year`

#### `run_cointegration_test`

Engle-Granger cointegration: hedge ratio, half-life, spread z-score signal.

**Required:** `symbol_a`, `symbol_b`, `start_date`, `end_date`  
**Optional:** `zscore_window`

#### `run_factor_regression`

Multi-factor OLS regression: alpha, loadings, t-stats, p-values, R².

**Required:** `symbol`, `factor_tickers`, `start_date`, `end_date`  
**Optional:** `factor_names`, `rolling_window`

#### `run_hurst_analysis`

Hurst exponent (DFA/R-S): regime classification and optional rolling breakdown.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `min_window`, `max_window`, `method`, `rolling_window`

#### `run_kalman_hedge_ratio`

Time-varying hedge ratio via a Kalman filter — a staleness diagnostic companion to run_cointegration_test's static OLS hedge ratio.

**Required:** `symbol_a`, `symbol_b`, `start_date`, `end_date`  
**Optional:** `observation_noise`, `include_intercept`, `delta`, `zscore_window`

#### `run_pca_analysis`

PCA on multi-asset returns: explained variance, loadings, factor contributions.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `standardize`, `method`, `n_components`

#### `run_seasonality_analysis`

Whether performance concentrates in a day of the week or a month of the year, corrected for having looked at all of them. Testing twelve months at 5% produces at least one 'significant' result on pure noise 46% of the time, which is where a good share of published calendar anomalies come from. A joint F-test is reported FIRST and every per-period p-value is Bonferroni corrected; if the joint test does not reject, no individual period should be reported however striking it looks.

**Required:** `returns`, `dates`  
**Optional:** `by`

#### `run_stationarity_tests`

ADF, KPSS and the variance ratio, with the four-way verdict spelled out. The two tests have OPPOSITE nulls, which is the whole reason to run both: failing to reject ADF is not evidence of a unit root, and the verdict separates 'the data says non-stationary' from 'the data says nothing'. 'contradictory' usually means a structural break rather than either answer.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `on`, `lags`

#### `scan_pairs`

Scan a ticker universe for cointegrated pairs, ranked by half-life.  
*Long-running: served only with `--enable-long-running`.*

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `max_pairs`, `min_half_life`, `max_half_life`, `p_value_threshold`, `zscore_window`

#### `test_autocorrelation`

A JOINT Ljung-Box test for autocorrelation across lags, rather than one test per lag. Check 20 lags individually at 5% on white noise and you expect one to fire; reporting that as 'returns are autocorrelated at lag 13' is an uncorrected multiple comparison and is how a great many spurious signals begin. Set squared=True to test volatility clustering instead of direction -- clustering is near-universal and is not a directional signal, it is why GARCH exists.

**Required:** `series`  
**Optional:** `lags`, `squared`

#### `test_granger_causality`

Whether one series helps predict another beyond that series' own past. NOT causality: a common driver produces it and so does a faster-updating proxy for the same information. It establishes temporal precedence in a linear model, which is necessary for a tradeable lead and nowhere near sufficient. Every lag is tested and the smallest p-value reported, so treat it as a screen rather than a test result.

**Required:** `cause`, `effect`, `start_date`, `end_date`  
**Optional:** `max_lag`

#### `test_normality`

Whether a return series is normal -- and it is not, which is the point. Almost every risk number assumes normality somewhere: parametric VaR multiplies a standard deviation by 1.645, a Sharpe's confidence interval uses a normal approximation, and every '2-sigma move' is a normal statement. The interesting output is not the p-value but the TAIL RATIO -- how many observations fall beyond three sigma against the 0.27% a normal predicts. Three to five times that is common on daily returns and it is what makes a parametric VaR understate the loss. On a long sample the test always rejects, so read the tail counts, which measure size, not the p-value, which measures length.

**Required:** `values`

#### `test_structural_break`

A Chow test for a break at a KNOWN date. The 'known' is load-bearing: a test at a date chosen because the data looks different there is not a valid test, because the hypothesis was picked using the data. Valid when the date comes from outside -- a regulation taking effect, a fee change, an index reconstitution, a strategy going live. For an unknown date use detect_change_points, which searches and reports the gain. With a `regressor` it tests whether the RELATIONSHIP broke (a beta or a hedge ratio) rather than whether the mean moved.

**Required:** `series`, `break_index`  
**Optional:** `regressor`

### `screener`

#### `get_stock_fundamentals`

Fetch company metadata and key financial ratios (PE, P/B, debt/equity, ROE, market cap).

**Required:** `symbol`

#### `run_screener`

Filter a stock universe by fundamental and technical criteria.

**Required:** `tickers`, `filters`  
**Optional:** `start_date`, `end_date`, `sort_by`, `ascending`, `min_beta_obs`

---

## `backtest` — Backtest

Run, optimize, validate and diagnose a trading strategy — the library's built-in ones or a signal the caller computed themselves. Does not construct portfolios or size positions.

### `backtest_execution`

#### `compare_strategies`

Run all four strategies on the same symbol and return ranked results vs buy-and-hold.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `initial_capital`, `commission_pct`, `slippage_pct`, `sort_by`, `sma_parameters`, `rsi_parameters`, `macd_parameters`, `bollinger_parameters`, `fill_price`

#### `run_backtest_compact`

Compact backtest result: summary/risk/exposure/cost sub-reports plus equity-curve/trade-log artifact URIs, instead of embedding the full data inline like run_sma_backtest etc.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `run_id`, `risk_free_rate`

#### `run_bollinger_backtest`

Backtest a Bollinger Band reversion rule: buy the lower band, sell the upper. Mean-reverting like the RSI version but with a volatility-scaled threshold, so it trades less in calm markets and more in volatile ones. That scaling is the reason to prefer it to a fixed threshold, and the reason its trade count varies so much across periods.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_buy_and_hold`

Buy-and-hold baseline: long the full period. Use as a passive benchmark.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_futures_backtest`

Simulate a FUTURES account, whose books the shared-cash engine cannot keep. Buying ten ES at 6200 does not cost 10 x 6200 x 50 of cash, it costs margin; the position then has no market value, because its profit arrives as daily variation margin credited to cash; and a short future pays no borrow. Equity here is cash plus posted margin and the contracts contribute nothing, so the leverage reported is economic exposure over equity rather than the gross-market-value ratio, and the two are not comparable. Margin calls reduce the position rather than being financed away.

**Required:** `prices`, `target_contracts`, `multiplier`  
**Optional:** `initial_capital`, `initial_margin`, `maintenance_margin`, `commission_per_contract`, `slippage_points`, `collateral_rate`, `contract_map`, `allow_fractional`

#### `run_macd_backtest`

Backtest a MACD signal-line crossover. Trend-following like the SMA version but on the difference of two exponential averages, so it turns faster and trades more -- which makes it the one most sensitive to transaction costs. Run estimate_break_even_cost on the result: a MACD strategy that breaks even near its assumed cost is a cost assumption rather than an edge.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_pair_trade_backtest`

Backtest a cointegrated pair as one synchronized two-leg trade — both legs enter/exit together and share one cash account, unlike scan_pairs which only screens candidates.

**Required:** `symbol_a`, `symbol_b`, `start_date`, `end_date`, `hedge_ratio`  
**Optional:** `entry_z`, `exit_z`, `zscore_window`, `initial_capital`, `commission_pct`, `slippage_pct`, `gross_leverage`, `fill_price`, `risk_free_rate`

#### `run_portfolio_simulation`

True shared-cash portfolio simulation with rebalancing at target-weight dates — unlike run_signal_panel_backtest, positions share one account instead of each ticker getting its own capital.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `target_weights_ref`, `target_weights`, `signal_type`, `construction_method`, `gross_leverage`, `n_long`, `n_short`, `vol_lookback`, `make_dollar_neutral`, `initial_capital`, `commission_pct`, `sell_commission_pct`, `slippage_pct`, `max_gross_leverage`, `max_position_pct`, `fill_price`, `commission_model`, `per_share_rate`, `min_commission`, `use_impact_model`, `impact_coefficient`, `impact_lookback`, `borrow_fee_bps`, `margin_interest_rate`, `max_adv_participation`, `benchmark`, `risk_free_rate`

#### `run_rsi_backtest`

Backtest an RSI mean-reversion rule: buy oversold, sell overbought. The counterpart to the crossover strategies -- it profits when prices revert and loses in a trend, so comparing the two on the same period says more about the period than either does alone. run_stationarity_tests and run_hurst_analysis say in advance which regime the sample is in.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_sma_backtest`

Backtest a moving-average crossover: long when the fast average crosses above the slow one, flat or short when it crosses back. The simplest trend-following rule there is, which makes it the right BASELINE -- a more elaborate strategy that cannot beat it has not earned its complexity. One run at one parameter pair; run_backtest_optimization searches, and run_walk_forward_backtest checks the search survived out of sample.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_strategy_matrix`

Every requested strategy against every requested ticker in one call, ranked. Fetches once per ticker and reuses the bars across strategies, so every cell is priced on identical data — which N separate calls cannot promise.

**Required:** `tickers`, `strategies`, `start_date`, `end_date`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `sort_by`, `risk_free_rate`

### `backtest_validation`

#### `analyze_parameter_decay`

Whether performance degrades SMOOTHLY as a parameter moves or falls off a cliff. A parameter whose neighbours perform almost as well describes a real effect with a broad optimum -- the exact value is not doing the work. A spike in a noisy objective is almost always the one setting that happened to fit the sample. Flags an optimum at the grid EDGE, where the true one may lie outside it or performance is monotone, which usually means the parameter stands in for something else.

**Required:** `parameter_values`, `performance`  
**Optional:** `metric_name`

#### `analyze_trade_clustering`

Whether wins and losses arrive in RUNS, measured in the original order. A 55% win rate is survivable if the losses are scattered and unholdable if they arrive eleven in a row, and the win rate is identical in both cases. A runs test gives the z-score: negative is clustering, positive is alternation (rarer, and usually a strategy reacting to its own last outcome, which is worth checking for a state-carrying bug). Read it alongside run_monte_carlo_trade_paths, whose reshuffling DESTROYS exactly the clustering measured here -- so its drawdown distribution is optimistic by this much.

**Required:** `trade_returns`

#### `build_purged_cv_splits`

Train/test index sets that do not leak, for a label that looks forward. A label built from a 5-day forward return at time t is a function of prices through t+5, so plain k-fold puts the test period's answer inside the training labels at every fold boundary -- which is why a model shows 0.6 AUC in cross-validation and 0.5 in production. Purging drops the overlapping training observations and the embargo drops the stretch after each test block. Combinatorial rather than sequential, so out-of-sample performance gets a DISTRIBUTION instead of one number with no error bar.

**Required:** `n_observations`  
**Optional:** `n_splits`, `n_test_splits`, `embargo_pct`, `label_horizon`

#### `compare_against_random`

Whether the strategy beats a coin that traded the same instrument the same number of times. Comparing against zero or against buy-and-hold is the usual test and neither is the right one after a search: a strategy can beat zero purely by holding a rising asset with no timing skill at all. The null keeps the trade MAGNITUDES and the win rate and randomizes the signs, so it tests whether the sequencing and sizing add anything -- not whether the win rate does. Measured as CONSERVATIVE: it fired on 1 of 150 skill-free strategies against a nominal 5%, so a non-rejection is weaker evidence than it looks.

**Required:** `trade_returns`  
**Optional:** `n_simulations`, `seed`

#### `compare_cost_models`

Run one strategy under several cost assumptions on a single fetched signal series, and solve for the commission rate at which its total return reaches zero. Answers 'does this survive costs' in one call.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`, `scenarios`  
**Optional:** `parameters`, `initial_capital`, `fill_price`, `solve_breakeven`, `risk_free_rate`

#### `estimate_backtest_overfitting`

PBO: how often the configuration that wins in-sample loses out-of-sample, across every equal split of the period. It measures your SELECTION PROCEDURE, not the strategy -- 0.5 means picking the in-sample best is no better than picking at random, and above 0.5 means the grid is being fitted to noise. Reports the median pairwise correlation between configurations, because a hundred settings correlated at 0.99 are one strategy and the PBO on them is meaningless.

**Required:** `trial_returns`  
**Optional:** `n_splits`

#### `estimate_break_even_cost`

The per-trade cost at which this edge disappears -- the number every backtest should report and almost none do. What decides whether a result survives contact with a real broker is not whether it is profitable at the assumed cost but how far above it the break-even sits. A strategy breaking even at 8bp when you modelled 5bp has 1.6x of headroom, and one bad fill or a widening spread eats it; one breaking even at 80bp is robust to both. Under about 2x, the backtest is a statement about the cost ASSUMPTION rather than about the strategy. Models a FLAT charge and not impact, so a strategy with headroom here can still fail on capacity.

**Required:** `trade_returns`  
**Optional:** `current_cost_bps`

#### `get_backtest_diagnostics`

Extended diagnostics for a built-in strategy: top drawdown episodes, trade expectancy/payoff/streaks with MAE/MFE, and exposure stats.

**Required:** `symbol`, `start_date`, `end_date`, `strategy_type`  
**Optional:** `parameters`, `initial_capital`, `commission_pct`, `slippage_pct`, `top_n_drawdowns`, `fill_price`, `risk_free_rate`

#### `get_deflated_sharpe_ratio`

The probability a Sharpe ratio is real GIVEN how many strategies were tried. Twenty strategies with no edge whatsoever, on two years of daily data, produce a best annualized Sharpe of 1.34 -- measured, not argued. Reporting that without saying 'and I tried 19 others' is false while every individual number is true. Skew and kurtosis widen the Sharpe's sampling distribution, so a short-volatility payoff is penalised here, correctly. Pass trial_sharpes for the accurate version: their VARIANCE sets the threshold, and 100 near-identical settings deflate far less than 100 different ideas.

**Required:** `returns`, `n_trials`  
**Optional:** `trial_sharpes`, `benchmark_sharpe`, `periods_per_year`

#### `get_drawdown_table`

Every drawdown episode in a persisted equity curve (peak, trough, recovery, depth, duration), deepest first, from an equity_curve_uri rather than by re-running the backtest.

**Required:** `equity_curve_uri`  
**Optional:** `min_depth`, `max_episodes`

#### `get_exposure_attribution`

How much of the return came from being RIGHT versus from being INVESTED. A strategy's return is exposure times the market's move, and splitting it shows whether the P&L came from timing -- holding more before up moves -- or simply from average exposure to an asset that rose, which is beta and available for nothing. The timing term is the covariance between exposure and the subsequent return; it is usually far smaller than expected and often NEGATIVE in strategies that look profitable. Also reports time in market, because a Sharpe computed on a 20%-invested strategy ignores what the idle capital earns and is not comparable with a fully-invested one's.

**Required:** `returns`, `exposure`  
**Optional:** `periods_per_year`

#### `get_regime_stratified_performance`

Performance broken out by regime, because one Sharpe over a mixed sample describes none of them. Catches the strategy with an overall Sharpe of 1.2 that earned all of it in one 18-month window and was flat for the other eight years -- arithmetically correct and completely misleading, and nothing in a single Sharpe reveals it. Leads with what fraction of P&L came from the single best regime; above about 70% the strategy is a bet on that regime recurring.

**Required:** `returns`, `regimes`  
**Optional:** `periods_per_year`

#### `get_robustness_diagnostics`

Same-sample robustness checks for a grid search: parameter sensitivity, Deflated Sharpe Ratio, and a block-bootstrap confidence interval on the best trial's Sharpe ratio.

**Required:** `symbol`, `start_date`, `end_date`, `strategy`, `param_grid`  
**Optional:** `initial_capital`, `commission_pct`, `slippage_pct`, `sort_by`, `n_bootstrap_iterations`, `bootstrap_block_size`, `bootstrap_confidence`, `random_seed`, `skew`, `kurtosis`, `risk_free_rate`

#### `run_backtest_optimization`

Grid-search strategy parameters and return the top N combinations ranked by a chosen metric.  
*Long-running: served only with `--enable-long-running`.*

**Required:** `symbol`, `strategy`, `start_date`, `end_date`, `param_grid`  
**Optional:** `initial_capital`, `commission_pct`, `slippage_pct`, `sort_by`, `top_n`, `n_workers`, `fill_price`, `risk_free_rate`

#### `run_monte_carlo_simulation`

Monte Carlo forward simulation of a portfolio's future equity paths via moving-block bootstrap of its historical returns.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `weights`, `horizon_days`, `n_simulations`, `block_size`, `initial_capital`, `random_seed`

#### `run_monte_carlo_trade_paths`

The distribution of outcomes the same edge could have produced, by RESHUFFLING the trades. A strategy with a 20% backtested drawdown does not have a 20% drawdown -- it has a distribution of them, and the backtested one is a single draw that routinely sits near the middle while the 95th percentile is half again as deep. Sizing so the backtested drawdown is survivable is sizing on the median outcome, and half of all realizations are worse. Reshuffles rather than resamples with replacement, so every path holds the same trades and ends at the same total return -- which isolates SEQUENCE risk from edge uncertainty instead of mixing the two.

**Required:** `trade_returns`  
**Optional:** `n_paths`, `seed`, `starting_equity`

#### `run_reality_check`

White's Reality Check: is this strategy better than the best of the alternatives, or the luckiest of them? Different from a t-test on its returns, and it is the question that matters after a search. The bootstrap is BLOCKED because resampling individual days destroys the serial correlation that drives drawdowns and volatility clustering, making the null too narrow and the p-value too small.

**Required:** `strategy_returns`, `benchmark_returns`  
**Optional:** `n_bootstrap`, `block_size`, `seed`

#### `run_regime_adaptive_backtest`

Classify market regime via Hurst, auto-select and optimise the best strategy.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `initial_capital`, `commission_pct`, `slippage_pct`, `hurst_method`, `sma_param_grid`, `rsi_param_grid`, `macd_param_grid`, `bollinger_param_grid`, `n_workers`, `risk_free_rate`

#### `run_regime_adaptive_walkforward_backtest`

Leakage-free regime-adaptive backtest: regime/strategy/parameter selection per walk-forward window, evaluated strictly out-of-sample.

**Required:** `symbol`, `start_date`, `end_date`  
**Optional:** `train_bars`, `test_bars`, `initial_capital`, `commission_pct`, `slippage_pct`, `hurst_method`, `sma_param_grid`, `rsi_param_grid`, `macd_param_grid`, `bollinger_param_grid`, `sort_by`, `fill_price`, `risk_free_rate`

#### `run_terminal_monte_carlo`

Monte Carlo that keeps only where the paths ENDED, so the simulation count is capped by wall-clock rather than by memory -- a million paths over a year is about 2 GB of path matrix for a handful of terminal quantiles, and this avoids allocating it. Same block bootstrap as run_monte_carlo_simulation. It says nothing about the journey: worst drawdown along the way and time underwater are not in here, and a benign distribution of endpoints can be reached by paths nobody could hold.

**Required:** `returns`, `horizon_days`  
**Optional:** `n_simulations`, `block_size`, `initial_capital`, `seed`

#### `run_walk_forward_backtest`

Walk-forward validation: optimise in-sample, evaluate out-of-sample, return OOS stats.

**Required:** `symbol`, `start_date`, `end_date`, `strategy`, `param_grid`  
**Optional:** `train_bars`, `test_bars`, `initial_capital`, `commission_pct`, `slippage_pct`, `sort_by`, `fill_price`, `risk_free_rate`

### `custom_signal`

#### `run_custom_signal_backtest`

Backtest a signal computed outside this library (your own alpha model) on one symbol.

**Required:** `symbol`, `start_date`, `end_date`, `signals`  
**Optional:** `signal_type`, `max_abs_weight`, `signal_fill_policy`, `initial_capital`, `commission_pct`, `slippage_pct`, `fill_price`, `risk_free_rate`

#### `run_signal_panel_backtest`

Backtest a pre-computed signal panel across a ticker universe, combined into portfolio metrics.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `signal_panel`, `signal_panel_ref`, `weights`, `signal_fill_policy`, `initial_capital`, `commission_pct`, `slippage_pct`, `benchmark`, `include_trade_log`, `fill_price`, `signal_type`, `max_abs_weight`

---

## `meta` — Discovery & Provenance

Questions about the library and the session rather than about a market: what this library accepts and what the data provider can serve, and what a past tool call did and whether it still reproduces.

### `discovery`

#### `compare_artifacts`

A field-by-field diff of two result objects, ordered by the size of the change. Answers the question that follows every re-run: did anything move, and what. Separates STRUCTURAL differences -- a field present in one and not the other -- from numerical ones, because the first usually means a version change and should be resolved before the second is read. An identical result is evidence of reproducibility and not proof of it: identical output from an identical cached input says nothing about the computation.

**Required:** `a`, `b`  
**Optional:** `tolerance`, `label_a`, `label_b`

#### `compare_data_sources`

Fetch the same fundamentals from two providers and report where they disagree, separating a SCALE difference (a constant ratio -- a missed unit conversion, fixable by arithmetic) from a DEFINITION difference (systematic with no constant ratio -- the two are computing different quantities and no conversion exists) from noise. FinancialRatios already documents that Polygon derives debt_to_equity from total liabilities and yfinance reports it as a percentage; this checks it rather than leaving it in a docstring. Fetches from both providers.

**Required:** `symbols`  
**Optional:** `left`, `right`, `fields`

#### `convert_reference`

Turn one kind of published value into another and publish the result: raw model predictions into a signal panel, scores into portfolio weights. This is what lets a producer and a consumer that were never written for each other compose.

**Required:** `ref`, `to_kind`, `run_id`, `name`  
**Optional:** `deadband`, `proba_threshold`, `long_only`, `task`, `construction_method`, `gross_leverage`

#### `describe_data_capabilities`

What a data provider can serve — tick trades, top-of-book quotes, async OHLCV, supported intervals, and its adjusted/survivorship/point-in-time guarantees. Fetches no market data. Call this before a tool that needs a capability the active provider may not have.

*No required arguments.*  
**Optional:** `source`

#### `describe_reference`

What a handoff reference points at — its content kind, shape, date span and which runtime published it. References are how bulk values cross runtimes without passing through the conversation.

**Required:** `ref`

#### `describe_runtime`

What each runtime is for, which categories it owns, and which tools it holds. A runtime is an EXECUTION boundary rather than a hint: a tool from another runtime is unroutable, not merely discouraged. Use this before scoping a session, and after a refusal that named a runtime you did not expect.

*No required arguments.*  
**Optional:** `runtime`, `include_tool_names`

#### `describe_temporal_contract`

What a data source can say about WHEN its facts became knowable, asked BEFORE fetching anything. A quarterly filing describes 30 September and is published on 25 October, so a model that joins it on the quarter end carries three weeks of hindsight per row. Read pit_safe first — False means do not build this dataset from this source — then reproduces_history, which is stricter: a snapshot source joins without leaking the future and still shows a backtest restated numbers nobody had. Fetches nothing.

*No required arguments.*  
**Optional:** `source`, `frame_kind`

#### `describe_tool`

One tool's full contract — arguments, result fields, owning runtime, and whether calling it fetches data or writes an artifact. Works for tools this caller is not scoped to; describing a tool is not calling it.

**Required:** `tool_name`  
**Optional:** `include_schema`

#### `estimate_tool_cost`

What each runtime costs a client's context, in bytes and approximate tokens, before any call is made. Choosing a scope is a real decision with a real cost and this makes it visible -- the numbers come from the live registry rather than from documentation, so they are current whenever a schema changes. Output schemas are excluded by default because the server omits them by default; counting them would report a cost nobody pays.

*No required arguments.*  
**Optional:** `runtimes`, `include_output_schemas`

#### `list_reference_kinds`

Every content kind a handoff reference can carry and what converts to what — the map of which producer outputs can reach which consumer inputs. Offline.

*No required arguments.*

#### `list_strategies`

Every built-in strategy's parameter contract: names, kinds, defaults, bounds and cross-parameter relations. Offline. Call this before guessing a strategy's parameters.

*No required arguments.*  
**Optional:** `strategy_type`

#### `list_stress_scenarios`

The named historical crash windows run_stress_test accepts, with each window's dates. Offline.

*No required arguments.*

#### `validate_tool_call`

Check arguments against a tool's schema WITHOUT calling it, including the strategy parameter contract that the JSON schema cannot express. Catches a hallucinated or out-of-range argument before it costs a fetch and a run.

**Required:** `tool_name`  
**Optional:** `arguments`

### `provenance`

#### `compare_decisions`

Diff two recorded calls — tool, inputs, output hash, git commit — and say which of the candidate causes the evidence supports.

**Required:** `request_id_a`, `request_id_b`

#### `describe_artifact`

Shape, date span, per-column statistics and both ends of a persisted Parquet artifact, by URI. Read what a run produced instead of re-running it.

**Required:** `uri`  
**Optional:** `preview_rows`

#### `explain_decision`

What one recorded tool call did: inputs, the market data it read with the content hashes those inputs had at the time, which execution path ran (C++/Numba/Python), timing, and the git commit and package version it ran under.

**Required:** `request_id`

#### `export_audit_bundle`

Package a date range of the audit log plus its chain index and manifest into one zip. Writes a new file; modifies no existing record.

**Required:** `start_date`, `end_date`, `out_path`

#### `replay_decision`

Re-run a recorded call and classify the result: reproduced, data_changed (the inputs were revised, so a different answer is expected), code_changed (inputs identical, output differs — the only case implicating the library), or not_comparable.

**Required:** `request_id`

#### `verify_audit_integrity`

Check the audit log's tamper-evident hash chain, for one day or the whole trail, optionally including that day's Ed25519 checkpoint signature. Read-only.

*No required arguments.*  
**Optional:** `date`, `public_key_path`

---

## `portfolio` — Portfolio & Execution

Turn a view into a position and price what it costs: optimal weights, risk attribution, sizing, stress tests, capacity, and liquidity measured from bars or from ticks.

#### `analyze_concentration`

How concentrated a portfolio is, in numbers with known interpretations. Effective N is the count of equally weighted positions that would give the same concentration -- a 100-position book with an effective N of 12 holds 100 names and has the concentration of 12, and that is the number to quote. Long-short books are measured on GROSS weights, because net makes the denominator near-zero and the shares meaningless.

**Required:** `weights`

#### `construct_weights_from_scores`

Turn alpha scores into portfolio weights and STOP, so the weights can be looked at before anything is simulated. Rank, top/bottom, z-score or volatility-scaled construction, optionally dollar-neutralised. This is the step that is otherwise buried inside a larger operation: when a model's backtest looks wrong, seeing the weights is what separates a bad signal from bad construction. Returns TARGET weights, not a P&L.

**Required:** `scores_ref`, `run_id`, `name`  
**Optional:** `method`, `gross_leverage`, `n_long`, `n_short`, `returns_ref`, `vol_lookback`, `dollar_neutral`

#### `estimate_covariance`

A covariance matrix plus the diagnostics that say whether to trust it. Shrinkage is the ANSWER to the conditioning warnings the optimizer already emits rather than a caveat about them: a covariance over N assets has N(N+1)/2 parameters, and mean-variance optimization is an error-maximizer over its worst-estimated directions. Read observations_per_parameter and condition_number before the matrix. Returns it annualized.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `method`, `halflife`, `periods_per_year`

#### `estimate_trade_cost`

Itemized cost of one hypothetical trade under a composed cost model: commission (pct/per-share/directional/maker-taker), spread (fixed bps or a fraction of the bar's range), square-root market impact, short borrow and margin interest. No market data needed.

**Required:** `notional`  
**Optional:** `side`, `commission_model`, `commission_pct`, `shares`, `per_share_rate`, `min_commission`, `buy_rate`, `sell_rate`, `taker_rate`, `maker_rate`, `is_maker`, `spread_model`, `spread_bps`, `bar_high`, `bar_low`, `bar_close`, `range_pct`, `avg_dollar_volume`, `volatility`, `impact_coefficient`, `short_borrow_bps`, `holding_days`, `margin_cash`, `margin_annual_rate`

#### `get_capacity_report`

How much account size a target-weight portfolio can support before positions become too large relative to each ticker's own trading volume, plus days-to-liquidate and sector exposure.

**Required:** `tickers`, `start_date`, `end_date`, `target_weights`  
**Optional:** `max_participation`, `adv_lookback`, `include_sector_exposure`

#### `get_factor_exposure_budget`

What the portfolio is actually betting on, once the names collapse into factors. Answers the failure that sinks more portfolios than any optimizer: 'I hold 40 names so I am diversified' -- 40 names with the same loading are one position with extra transaction costs. Supply factor_covariance to get each factor's share of portfolio VARIANCE, which is the number that answers what you are taking risk on; without it only exposures can be reported, and a large loading on a quiet factor is not a large risk.

**Required:** `weights`, `factor_loadings`  
**Optional:** `factor_covariance`, `factors`

#### `get_liquidity_adjusted_var`

VaR that accounts for not being able to exit at the mark. A 1-day 95% VaR describes a position you could close today; one that takes 15 days to liquidate at a sane participation rate is exposed for 15 days and carries roughly sqrt(15) times the risk -- a factor of four, and the part usually missed. The liquidation COST is reported separately from the quantile on purpose: cost is an expectation and VaR is a quantile, and adding them produces a number that is neither.

**Required:** `positions`, `volatilities`, `daily_volumes`  
**Optional:** `confidence`, `participation_rate`, `correlation`

#### `get_liquidity_metrics`

Amihud illiquidity ratio and Corwin-Schultz spread estimator per ticker — OHLCV-derived proxies for market depth and bid/ask spread.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `window`

#### `get_marginal_risk_contribution`

Where the risk in a portfolio you ALREADY hold is coming from, and what one more unit of each position costs. Marginal risk is the derivative of portfolio volatility with respect to the weight; contribution is weight times marginal, and these sum exactly to portfolio volatility, which makes it a decomposition rather than an allocation of blame. The diagnostic is the contribution share against the weight share: an asset at 5% of the book carrying 30% of the risk is the position to look at. A NEGATIVE marginal contribution means adding reduces risk -- that position is a hedge whether or not it was meant as one.

**Required:** `weights`, `assets`, `covariance`

#### `get_portfolio_risk_attribution`

Deep portfolio risk decomposition: MCR per asset, PCA attribution, optional factor model.

**Required:** `tickers`, `weights`, `start_date`, `end_date`  
**Optional:** `benchmark`, `n_components`, `factor_tickers`, `factor_names`, `risk_free_rate`

#### `get_position_size`

How large to trade, from a stop distance measured in ATR and an account risk budget. Answers the question a signal does not: a correct direction sized wrong loses money. Kelly is optional and is a CEILING rather than a target -- full Kelly maximizes long-run growth and produces drawdowns almost nobody holds through, which is why half-Kelly is the common practice.

**Required:** `symbol`, `start_date`, `end_date`, `account_equity`  
**Optional:** `risk_per_trade_pct`, `atr_period`, `atr_multiplier`, `win_rate`, `avg_win_pct`, `avg_loss_pct`

#### `optimize_hierarchical_risk_parity`

Allocation that never INVERTS the covariance matrix. Inversion is where an ill-conditioned estimate does its damage -- the smallest eigenvalue becomes the largest, so the direction the data says least about becomes the one the portfolio bets most on, and with 50 assets on 500 observations that eigenvalue is noise. HRP clusters by correlation, orders the assets so similar ones sit adjacent, and splits capital down the tree. It has NO optimality property and does not maximize anything; it buys robustness by giving that up.

**Required:** `returns`

#### `optimize_max_diversification`

The portfolio that maximizes the DIVERSIFICATION RATIO -- the weighted average of the assets' volatilities over the portfolio's own. It is 1.0 when everything is perfectly correlated and grows as correlations fall, so maximizing it maximizes the volatility that CANCELS. Not the same as minimum variance, which piles into the quietest assets because quiet is what it minimizes; this normalizes by each asset's own volatility first, so an asset is rewarded for being UNCORRELATED rather than for being quiet. It does invert the correlation matrix and reports the condition number for that reason.

**Required:** `assets`, `covariance`

#### `optimize_risk_parity`

Weights at which every asset contributes the SAME AMOUNT OF RISK -- not the same weight. An equally weighted portfolio of a bond fund and a biotech stock is a biotech portfolio; the equity contributes nearly all the variance. Uses NO expected returns, which is the reason to prefer it over mean-variance in most real situations: the standard error on a mean return from two years of daily data is about the size of the estimate, and mean-variance is maximally sensitive to exactly that input. Pass `budget` for an unequal risk budget, which is how most mandates are actually written.

**Required:** `assets`, `covariance`  
**Optional:** `budget`, `max_iterations`, `tolerance`

#### `plan_rebalance`

A day-by-day path from the weights you hold to the weights you want. Every optimizer here returns a target vector and implicitly assumes you arrive instantly and for free; trading fast costs market impact and trading slow means holding the portfolio you were trying to leave, so this returns the SCHEDULE and both costs rather than one number. Surfaces what nothing else does: a target weight the market cannot supply, with the number of days it would really take.

**Required:** `current_weights`, `target_weights`, `portfolio_value`  
**Optional:** `adv`, `max_participation`, `max_days`, `urgency`, `impact_coefficient`

#### `run_portfolio_optimization`

Produce portfolio weights via Markowitz mean-variance (max_sharpe/min_volatility/target_return/target_volatility), risk parity, or Black-Litterman — unlike get_portfolio_analysis, which only scores weights already chosen.

**Required:** `tickers`, `start_date`, `end_date`  
**Optional:** `method`, `risk_free_rate`, `target_return`, `target_volatility`, `allow_short`, `max_weight`, `risk_budget`, `market_weights`, `views`, `risk_aversion`, `tau`, `periods_per_year`

#### `run_portfolio_scenarios`

What a portfolio does under NAMED shocks rather than under a distribution. A 99% VaR is a quantile of a distribution fitted to history, and its central weakness is that the event you care about is usually not in that history; a named scenario makes the assumption explicit and arguable. Positions absent from a scenario are treated as unchanged and the COVERAGE is reported, because a scenario touching three of forty positions produces a loss that is a lower bound rather than a worst case.

**Required:** `weights`, `scenarios`  
**Optional:** `assets`, `covariance`

#### `run_stress_test`

Replay a portfolio's weights against a named historical crash window (or custom date range) using real historical returns.

**Required:** `tickers`  
**Optional:** `weights`, `scenario`, `custom_start_date`, `custom_end_date`

---

## `delta_one` — Delta One

Which instrument is the cheapest way to own or hedge this exposure. Carry and basis against a quoted future, the term structure and what rolling along it costs, a portfolio beta translated into a number of contracts, whether that hedge historically worked, a basket against its index, and every way of expressing one position -- cash, ETF, future, swap, synthetic -- ranked on one annualized number. Takes quotes and contract specifications as arguments; there is no futures data provider.

#### `analyze_basis_history`

Where today's basis sits inside its own history -- z-score, percentile, half-life and the distribution it came from. A basis of 38 bps means nothing until you know the series has spent a year between -5 and +25, which is the difference between a number and a trade. Measured in bps of spot rather than points, because points are not comparable through time on anything that has moved. Without a window the z-score is full-sample and looks ahead.

**Required:** `spot_prices`, `futures_prices`  
**Optional:** `window`, `time_to_expiry`

#### `analyze_cash_futures_basis`

A quoted future against its carry-fair value, with the mispricing attributed to financing, dividend or borrow. Returns the basis in POINTS, as an annualized rate, and as the financing the quote implies -- three views because points cannot be compared between a March and a December contract and the annualized spread can. A future 40 bps rich is usually expensive funding rather than edge, so the implied financing rate is the number to check against SOFR before trading it.

**Required:** `spot`, `future_price`, `time_to_expiry`, `risk_free_rate`  
**Optional:** `dividend_yield`, `borrow_rate`, `tolerance_bps`

#### `analyze_dividend_points`

Index dividends as POINTS to a contract's expiry, dated and attributed by name. A continuous yield is the approximation this replaces: index dividends arrive in dense seasonal clusters, so a June and a September contract straddle the whole season and pricing both off one q puts one badly wrong. Supply a quoted future to get the market's own dividend number alongside the forecast -- a gap between them is usually a position rather than an error on either side.

**Required:** `constituents`, `divisor`, `as_of`, `expiry`  
**Optional:** `spot`, `future_price`, `financing_rate`, `time_to_expiry`

#### `analyze_etf_fair_value`

An ETF's premium or discount, and what survives the cost of capturing it. A visible 40 bp premium is almost never 40 bps of edge: the creation unit is an indivisible block, creating means crossing the BASKET's spreads rather than the fund's, and the NAV compared against is usually last night's struck value rather than a live one, so an intraday premium is mostly the market's move since the strike. The net figure after round-trip costs is the number that decides anything.

**Required:** `etf_price`, `nav`  
**Optional:** `nav_is_intraday`, `basket_value`, `cash_component`, `creation_unit_shares`, `creation_fee`, `etf_spread_bps`, `basket_spread_bps`, `tolerance_bps`

#### `analyze_futures_curve`

The futures term structure, and the FORWARD carry between expiries that a calendar spread actually prices. A trader seeing the near contract at 205 bps and the far at 210 is not being offered 210 for the period between them; they are offered whatever makes the two consistent, and trading off the quoted levels can reverse the sign of the position. Contango here describes the PRICE curve -- unrelated to the vol term structure that uses the same word.

**Required:** `contracts`  
**Optional:** `spot`

#### `analyze_hedge_effectiveness`

Whether a hedge ratio actually removed risk, measured on realized returns rather than assumed from a beta. Reports volatility, beta and drawdown before and after, and the ROLLING ratio, which is the diagnostic that matters: a hedge whose ratio averaged 1.0 while ranging from 0.4 to 1.7 was never a hedge, it was two positions that averaged out, and its in-sample volatility reduction will not repeat. A hedge that raised volatility is almost always a sign error on the ratio.

**Required:** `portfolio_returns`, `hedge_returns`, `hedge_ratio`  
**Optional:** `window`, `periods_per_year`

#### `analyze_index_basket`

Value a basket of constituents against the index it replicates, attributing the spread name by name. A basket printing 40 bps from its index is far more often ONE constituent that has not traded than an arbitrage across all of them, so contributions come back sorted and a suspiciously unchanged price is flagged. Share-based with a divisor reproduces the index level; weight-based without one reproduces its returns but not its level, and conflating the two is how a basket comes out off by a constant.

**Required:** `constituents`  
**Optional:** `index_level`, `divisor`

#### `analyze_index_rebalance`

The buying and selling an index change forces on passive money, sized as DAYS OF VOLUME first and currency second. $2.8bn is nothing in a name trading $2bn a day and is a crisis in one trading $450m, and only the ratio says which. Index flow conventionally clears in a single closing auction that is 10-20% of the day, so leaving auction_fraction at 1.0 understates the binding participation by five to ten times. It sizes the flow and deliberately does not predict the price move.

**Required:** `old_weights`, `new_weights`, `indexed_assets`  
**Optional:** `adv`, `auction_fraction`

#### `analyze_roll`

What moving a position from one contract into the next actually costs, with the break-even rate it then has to out-earn. Distinct from the curve because this one has a size: the position is SIGNED, and a short rolled up a contango curve collects the step a long pays. Roll yield is reported as what it is -- a price step expressed as a rate, not a return, which a long gives up if spot does not move. Different multipliers are resized by money, not contract count.

**Required:** `front_price`, `next_price`, `contracts_held`, `multiplier`, `days_to_front_expiry`  
**Optional:** `next_multiplier`, `cost_per_contract`, `spread_ticks`, `tick_value`

#### `analyze_total_return_future`

Read a TRF quote as the financing spread it embeds, and compare it to what a swap charges. This is what answers 'regular futures imply 50 bp of funding and the TRF implies 95 -- where does the 45 go' as a calculation rather than a reconstruction. quote_convention is REQUIRED because some contracts quote the spread in bps and others quote a level, the two are not convertible without knowing which you have, and assuming wrong misprices by the entire financing leg.

**Required:** `quote`, `quote_convention`, `underlying_price`, `time_to_expiry`, `reference_rate`  
**Optional:** `dividend_yield`, `comparison_spread_bps`

#### `compare_delta_one_expressions`

Rank several ways of holding one exposure -- cash, ETF, future, forward, synthetic or swap -- on one annualized basis-point number. The HORIZON reorders them, and that is the answer rather than an artefact: execution is paid once and carry accrues, so a 2 bp round trip is 24 bp a year over one month and 1 bp over two years, and the cheapest instrument to hold is routinely not the cheapest to hold briefly. Every omitted cost term defaults to zero, so a surprisingly cheap row is usually an unpriced one.

**Required:** `expressions`, `notional`, `horizon_years`  
**Optional:** `direction`

#### `detect_basis_dislocation`

Whether a basis has STRUCTURALLY shifted rather than merely moved, by CUSUM and change-point detection. A z-score asks how unusual today is, and a basis that drifts two sigma wide and stays there never has a remarkable day -- CUSUM accumulates, so a sustained shift crosses when no single observation would. That is the difference between 'the basis is wide' and 'the basis is not the same basis any more', and only the second is a reason to re-examine the carry behind a position. A crossing is usually a roll or a dividend rather than a dislocation.

**Required:** `spot_prices`, `futures_prices`  
**Optional:** `time_to_expiry`, `reference_fraction`, `threshold`, `slack`, `max_breaks`

#### `monitor_spread_stream`

Watch a spread on a LIVE feed, one stateful call at a time. A tool call returns, so there is nowhere for a subscription to live -- the state comes back in the result and goes in on the next call, which means a monitor can be paused, serialized and resumed elsewhere without losing its baseline. ONE tool covers live basis, ETF NAV, index arbitrage, roll spread and any cross-instrument spread, because those are three formulas rather than five. Accumulators are carried, so a hundred ticks in one call and a hundred calls of one tick agree exactly.

**Required:** `primary_prices`, `reference_prices`  
**Optional:** `channel`, `state`, `time_to_expiry`, `reset`, `keep_baseline_on_reset`, `label`, `warmup`, `threshold`, `slack`

#### `optimize_replication_basket`

The smallest basket that tracks a benchmark, minimizing the variance of the DIFFERENCE rather than the portfolio's own variance. Those are different portfolios: minimum-variance picks a defensive corner of a universe, minimum-tracking-error picks whatever most resembles the index. A max_names limit is enforced by thresholding because SLSQP cannot express an integer constraint and this library has no mixed-integer solver, so the answer is a GOOD basket of that size rather than a provably best one. Tracking error is in sample.

**Required:** `returns`, `benchmark_returns`  
**Optional:** `max_names`, `long_only`, `max_weight`, `weight_caps`, `covariance_method`, `periods_per_year`

#### `price_total_return_swap`

Mark a total return swap with the equity and financing legs separated. The payoff is simple -- receive price change plus dividends, pay a rate plus a spread -- and the CONVENTIONS are where the money is: ACT/360 accrues about 1.4% more financing than ACT/365F on the same period, and a zero dividend argument silently turns this into a price-return swap, understating the equity leg by 200 bps a year on a 2% yielder.

**Required:** `notional`, `initial_price`, `current_price`, `financing_rate`  
**Optional:** `spread_bps`, `dividends`, `start_date`, `valuation_date`, `time_elapsed`, `day_count`, `direction`

#### `size_futures_hedge`

The futures position that neutralizes a portfolio's beta, reported exact, rounded, and with the residual that rounding leaves. A -903.2 contract hedge is not available and -903 is; the 0.2 left over is $70,000 of unhedged beta and it decides whether the hedge is finished. Sizes on dollar beta, so a 1.12-beta book sells 12% more notional than it holds. Hedging with a different index needs that contract's own beta, or the hedge is short by the ratio.

**Required:** `portfolio_value`, `portfolio_beta`, `future_price`, `multiplier`  
**Optional:** `future_beta`, `objective`, `existing_contracts`

#### `solve_forward_carry`

Recover whichever of financing, dividend or borrow a quoted forward implies, given the other two. One inverse rather than three tools, because they are one rearrangement of ln(F/S)/T = r - q - b and three near-identical names would be a coin flip for a model. The answer is CONDITIONAL and absorbs every error in the two rates you supply: an implied borrow computed against a wrong dividend is wrong by that whole dividend and looks entirely plausible.

**Required:** `spot`, `forward`, `time_to_expiry`, `solve_for`  
**Optional:** `risk_free_rate`, `dividend_yield`, `borrow_rate`

---

## `modeling` — Modeling

Build, validate and score a statistical model from this library's own features: dataset construction, leakage-purged walk-forward fitting, the model registry, and evaluation of out-of-sample predictions. One ordered pipeline rather than a set of interchangeable tools.

#### `analyze_features`

Score a built dataset's FEATURES before fitting anything: coverage, turnover, cross-sectional IC and ICIR, decile spread and monotonicity, which features are near-duplicates of one another, and a lead-lag causality screen for features whose information arrives too early. Use this to choose features; use inspect_model(view='feature_importance') to see what one fitted model then leaned on.

**Required:** `dataset_id`  
**Optional:** `features`, `n_quantiles`, `cluster_threshold`, `include_leakage`, `leakage_max_shift`

#### `build_model_dataset`

Fetch OHLCV, compute requested features/target, persist the panel.

**Required:** `spec`

#### `check_leakage`

Ask whether a set of features is temporally safe to fit on — before building a dataset with them. Optionally reports a built dataset's recorded point-in-time coverage too.

*No required arguments.*  
**Optional:** `feature_ids`, `dataset_id`

#### `compare_models`

Rank registered models side by side on their out-of-sample metrics. Models are ranked within their own task, never across tasks, because those metrics are not on a common scale.

**Required:** `model_ids`  
**Optional:** `metric`

#### `evaluate_model_portfolio`

Evaluate a model's out-of-sample predictions as a shared-cash portfolio: transform predictions into target weights and simulate them with costs, returning Sharpe, drawdown, turnover and exposure.

**Required:** `model_id`  
**Optional:** `transform`, `portfolio`

#### `explain_dataset_row_loss`

Which column cost which training rows, and which are free to drop. Reports n_missing beside n_sole_missing, and the second is the actionable one: a 252-day feature sitting behind a 500-day one has n_missing in the hundreds of thousands and n_sole_missing of zero, so removing it gives back nothing. Reading only the first number produces a decision that feels informed and changes nothing, which is why "you lost 44% of the data" is not an answer.

**Required:** `dataset_id`

#### `inspect_model`

Inspect a registered model's summary/importance/validation/lineage.

**Required:** `model_id`  
**Optional:** `view`

#### `join_point_in_time`

Attach point-in-time records to a built dataset, each panel row getting the most recent record AVAILABLE by then. Strictly backward and inclusive: a filing released before a bar's close is usable on it, one released after is not, and a row with nothing available yet gets NaN rather than zero or the eventual value. A restatement is a second row with the same event_time and a later available_time, and the join returns whichever version was current at each date.

**Required:** `dataset_id`, `records`  
**Optional:** `fields`, `entity_scoped`, `prefix`, `max_staleness_days`

#### `list_datasets`

Every built dataset panel, newest first, with row/entity/feature counts and date span.

*No required arguments.*  
**Optional:** `limit`

#### `list_features`

Which features this library can build, what each one measures, and what it costs to compute. Call it BEFORE build_model_dataset rather than guessing names -- a feature name that does not exist is a failed call and an error round trip, which costs more than reading the catalogue does. The feature_lab runtime then answers what each one is worth once the dataset exists.

*No required arguments.*  
**Optional:** `category`

#### `list_modeling_capabilities`

What this modeling runtime can do: tasks, estimators and the capabilities of each (sample weights, probabilities, query groups, coefficients, feature importance), features, target types, validation schemes, preprocessing and weighting options, and which optional libraries are installed. Call this before choosing a model rather than assuming an estimator is available.

*No required arguments.*  
**Optional:** `include_estimators`

#### `list_models`

Every registered model, newest first, with its task, estimator, headline out-of-sample metric and source dataset. Call this when you need a model_id you do not already hold.

*No required arguments.*  
**Optional:** `task`, `limit`

#### `run_model_experiment`

Fit + walk-forward validate + register a model from a persisted dataset.

**Required:** `dataset_id`, `spec`

#### `score_model`

Run a registered model forward and get its predictions for a universe as of a date. The step that turns a fitted model into something a backtest can consume, and the one where point-in-time discipline matters most: the `as_of` date is what stops the model seeing features that did not exist yet. Raw probabilities from a tree ensemble are NOT calibrated, so a 0.9 threshold may select no rows at all -- check the distribution before thresholding.

**Required:** `model_id`, `as_of`, `universe`  
**Optional:** `lookback_days`, `max_staleness_days`

#### `score_predictions`

Score a predictions reference against its realized outcome — accuracy metrics, cross-sectional IC and ICIR, a predict-the-mean baseline, and an effective sample size adjusted for overlapping forward returns. Works on predictions this library never produced.

**Required:** `predictions_ref`, `task`  
**Optional:** `target_column`, `prediction_column`, `ic_method`, `ndcg_cutoffs`

#### `validate_model_spec`

Check a ModelSpec before spending an experiment on it: that the estimator exists for the task, that its parameters are accepted, and how many fits the spec implies once a search grid multiplies through every fold. Fetches nothing and fits nothing.

**Required:** `spec`  
**Optional:** `dataset_id`

#### `validate_pit_records`

Check point-in-time records BEFORE joining them onto anything. The error worth catching is the two timestamps the wrong way round: event_time is when a fact is ABOUT, available_time is when it could first be ACTED ON, and swapped they make every model look prescient. Also reports median_publication_lag_days -- exactly how much hindsight a naive join on event_time would have handed you. Fetches nothing.

**Required:** `records`  
**Optional:** `entity_scoped`

---

## `microstructure` — Microstructure

What the market will charge you to trade, at two data fidelities. Four tools MEASURE spreads and order flow from ticks and refuse without a tick feed; seven ESTIMATE the same quantities from OHLCV, which is the normal case, each saying what it is a proxy for and how it fails.

#### `check_spread_proxy`

Measures the spread from ticks and compares it against the OHLCV estimate, so the proxy's error on THIS name is a number rather than an assumption. The Corwin-Schultz estimate is what a bar-only session would have used.

**Required:** `symbol`, `start`, `end`, `bar_start_date`, `bar_end_date`  
**Optional:** `window`, `limit`, `source`

#### `classify_trade_direction`

Sign a tick tape buyer- or seller-initiated and publish the signed series, which is what an event study, a CUSUM detector or a model consumes -- get_microstructure_metrics computes this internally and returns only averages. WITH a quote panel it is Lee-Ready, matching each trade against the quote PRECEDING it; without one it falls back to the tick rule, which agrees about 85% of the time on a liquid name and worse on an illiquid one. The result says which was used, because every downstream estimate inherits that error.

**Required:** `tick_tape_ref`, `run_id`, `name`  
**Optional:** `quote_panel_ref`

#### `detect_liquidity_events`

CUSUM change detection over tick-derived liquidity channels -- when the spread, depth, trade intensity or signed flow regime CHANGED, rather than what it is on average. Declares every channel it knows about, including the ones this feed cannot supply.

**Required:** `symbol`, `start_date`, `end_date`, `channels`  
**Optional:** `freq`, `threshold`, `reference_fraction`, `source`

#### `estimate_corwin_schultz_spread`

The spread implied by the HIGH-LOW RANGE (Corwin-Schultz 2012). A day's range contains both volatility and the spread; volatility scales with the square root of time and the spread does not, so one- and two-day ranges identify them separately with no quote data at all. It produces NEGATIVE estimates on 10-30% of days as a sampling artefact, floored at zero as the authors recommend -- read negative_fraction, because above about a third the flooring turns a symmetric error into a one-sided bias and the average is noise. Measured: a planted 100 bps spread came back at 103 bps, a planted 20 bps at 56 bps with 44% negative.

**Required:** `high`, `low`

#### `estimate_kyle_lambda`

Market DEPTH: the price impact of a unit of signed order flow, from a regression of price change on signed volume. The one measure here with a direct trading interpretation -- multiply by the size you intend to trade for an estimate of the impact you will cause. The signing comes from the TICK RULE rather than from matching trades against quotes, which is right about 85% of the time on liquid names and worse on illiquid ones; misclassification attenuates the slope toward zero, so this understates impact and understates it most exactly where impact is largest. Check r_squared before sizing anything off it.

**Required:** `close`, `volume`  
**Optional:** `window`

#### `estimate_roll_spread`

The effective spread implied by BID-ASK BOUNCE, from trade prices alone (Roll 1984). Consecutive price changes mean-revert when trades arrive randomly at bid and ask, and the size of that reversal is the spread. IT RETURNS A SPREAD WHEN THERE IS NONE: on a simulated walk with a spread of exactly zero it produced 0.098 on a $100 stock, because the lag-1 autocovariance's standard error swamps the signal whenever the spread is small against volatility, and taking a root only when the covariance lands negative discards the other half of that noise. Read `significant` and `smallest_detectable_spread` before the estimate. On a trending series it returns null rather than zero, because 'could not measure' and 'was zero' are different facts.

**Required:** `prices`  
**Optional:** `window`

#### `estimate_vpin`

Flow one-sidedness measured in VOLUME time rather than clock time (Easley, Lopez de Prado and O'Hara 2012) -- information arrives with volume, so the series is cut into equal-volume buckets. TWO HONEST CAVEATS. This is built from daily bars with tick-rule signing; the original is a trade-level measure where each bucket holds hundreds of trades, so what comes back is a defensible series of one-sidedness and not the VPIN of the paper. And VPIN is contested: Andersen and Bondarenko (2014) argue it is largely a transformation of volatility. Calling one-sided flow 'informed trading' is a model assumption, not a measurement.

**Required:** `close`, `volume`  
**Optional:** `n_buckets`, `window`

#### `get_amihud_illiquidity`

How far the price moves per dollar traded (Amihud 2002) -- the most widely used liquidity proxy in the literature, because it needs nothing but daily bars. THE RAW NUMBER IS UNINTERPRETABLE: its units are return-per-dollar, so it scales inversely with dollar volume and a large cap's reading is orders of magnitude below a microcap's with neither meaning anything alone. Read the percentile against this name's own history. It is NOT a spread -- it conflates spread, book depth and the information content of trades, and a genuinely volatile stock scores as illiquid even with a deep book.

**Required:** `close`, `volume`  
**Optional:** `window`

#### `get_effective_spread_series`

What each trade ACTUALLY paid against the prevailing midpoint, per trade, published as a reference. Pass realized_horizon_seconds to split it into the REALIZED half the liquidity provider kept and the IMPACT half the trade moved -- those imply opposite remedies, impact says trade smaller and realized says trade somewhere else, and without the split neither is visible.

**Required:** `tick_tape_ref`, `quote_panel_ref`, `run_id`, `name`  
**Optional:** `realized_horizon_seconds`

#### `get_implementation_shortfall`

What an execution ACTUALLY cost, decomposed after Perold. Every other cost tool here is a model run before the fact -- estimate_trade_cost predicts, get_capacity_report bounds, plan_rebalance schedules -- and this is the measurement those models should be checked against. Splits the gap between the decision price and what was achieved into DELAY (the price moved before the order reached the market, a workflow problem no algorithm recovers, and frequently the largest term), IMPACT (the part an algorithm controls), OPPORTUNITY (the shares never filled -- an algorithm that beats its benchmark by not completing has moved its cost here rather than saved it), and FEES. Positive is a cost.

**Required:** `decision_price`, `arrival_price`, `fills`, `target_quantity`, `final_price`  
**Optional:** `side`

#### `get_intraday_volume_profile`

How volume distributes across the trading day, and what that implies for a participation schedule. The U-shape is the fact every execution schedule is built on: volume concentrates at the open and close with a midday trough routinely a third of the opening bucket, so a schedule spread evenly across the CLOCK over-participates at lunch -- paying impact into a thin book -- and under-participates at the close, missing the cheapest liquidity of the day. Needs INTRADAY bars with timestamps; daily bars are refused rather than aggregated into a meaningless single bucket.

**Required:** `volume`, `timestamps`  
**Optional:** `n_buckets`

#### `get_microstructure_metrics`

Quoted and effective spread MEASURED from trades and quotes, with the effective spread split into the realized (liquidity-provider) and impact (price-move) halves, plus Lee-Ready signed order flow. Needs a tick feed and refuses without one rather than approximating from bars.

**Required:** `symbol`, `start`, `end`  
**Optional:** `realized_horizon_seconds`, `limit`, `source`

#### `get_order_book_metrics`

Depth-book statistics a top-of-book quote cannot give: the microprice, imbalance at the touch AND cumulatively, and how fast liquidity thins with distance. The midpoint ignores size, so a book with 5,000 bid and 100 offered reads the same as its mirror and the second is about to trade higher. Touch and cumulative imbalance routinely disagree, and a book bid at the touch with weight behind the offer is exactly the one that ticks up and fills badly. Takes the book as an argument; no provider here serves depth.

**Required:** `snapshots`  
**Optional:** `levels`, `include_profile`

#### `get_order_flow_imbalance`

Signed volume imbalance from bars, with its own predictive test attached rather than presented as a signal to be trusted. `persistence` is measured on NON-OVERLAPPING windows: a rolling sum at window=5 shares four of five observations with the previous point, so its raw autocorrelation is about +0.76 on PURE NOISE (and +0.89 at window 10, +0.96 at 21, tracking 1-1/w). That number describes the window, not the flow, and it is returned separately as `overlapping_persistence` so the difference is visible.

**Required:** `close`, `volume`  
**Optional:** `window`

#### `get_quoted_spread_series`

Spread and imbalance PER QUOTE rather than averaged into one number, published as a reference. QUOTED is what crossing would cost at an instant and is NOT what trades paid -- the effective spread is the one a backtest should be charging. Top of book only: depth and queue position are not in this data.

**Required:** `quote_panel_ref`, `run_id`, `name`

#### `get_trade_profile`

How volume distributes across trade sizes and times of day, from tick data. Answers whether the liquidity is in a few large prints or many small ones, which decides whether a large order can hide.

**Required:** `symbol`, `start`, `end`  
**Optional:** `size_buckets`, `intraday_freq`, `limit`, `source`

---

## `data` — Data

Get the data and publish it as an `sqt://` reference every other runtime can read: OHLCV for one name or a whole universe, return panels, tick tapes and quote panels, provider guarantees, temporal contracts, and bundles that pair frames with what their sources promise. Fetches; does not analyze.

#### `build_continuous_futures_series`

Stitch a chain of futures contracts into one continuous series, and publish TWO references rather than one. The adjusted series is a research instrument and is not a price -- back-adjustment changes every historical level, and a difference-adjusted series can go negative on a contract that never traded below zero -- so sizing a position from it means sizing against a number nobody could have transacted at. The second reference carries which contract was actually active each date and what it actually traded at.

**Required:** `contracts`, `run_id`, `name`  
**Optional:** `roll_rule`, `adjustment`, `days_before_expiry`

#### `build_data_bundle`

Name several already-published frames as one unit and publish the manifest as a data_bundle reference. A bundle holds references rather than copies, so it cannot diverge from the frames it names, and it pairs each frame with what its source can say about timing -- which is the pairing a point-in-time join depends on and which a bare frame throws away.

**Required:** `frames`, `run_id`, `name`

#### `compare_ratio_frames`

Two providers' ratios side by side, with each disagreement CLASSIFIED rather than merely measured: a unit mismatch is fixable by rescaling, a definition difference is not, and averaging across the second kind produces a number neither provider would stand behind. Takes the values as arguments, so it works for sources this library cannot fetch.

**Required:** `left`, `right`  
**Optional:** `left_name`, `right_name`, `fields`

#### `describe_data_bundle`

What frames a bundle names, how many rows and columns each has, and what each source can promise about revisions and point-in-time availability. Use it to see what a bundle actually contains before building a dataset on it, rather than after a model has already been fitted on whatever was in there.

**Required:** `ref`

#### `fetch_financial_ratios`

Fetch a company's financial ratios and flag the ones that are implausible on their face -- a negative price-to-book, a dividend yield above a plausible ceiling. The flag is a weak signal in one direction only: it catches values that are obviously wrong, never values that are merely incorrect.

**Required:** `symbol`

#### `fetch_ohlcv`

Fetch one symbol's OHLCV bars and publish them as an `sqt://` price_panel reference rather than returning the rows inline. Reach for this when the bars themselves are the thing another tool needs -- an indicator series, a custom signal, a panel join -- instead of going through an analysis tool that wants to do something else with them. The reference is what crosses runtimes; the frame never has to enter the conversation.

**Required:** `start_date`, `end_date`, `run_id`, `name`, `symbol`  
**Optional:** `interval`

#### `fetch_ohlcv_panel`

Fetch a whole universe's OHLCV in one call and publish it stacked long, with an `entity` column, as a price_panel reference. Tickers that returned nothing are named in `warnings` and are ABSENT from the panel rather than present as NaN, which matters because a complete-case join downstream will not see them at all.

**Required:** `start_date`, `end_date`, `run_id`, `name`, `tickers`  
**Optional:** `interval`

#### `fetch_quote_panel`

Fetch top-of-book quotes and publish them as a quote_panel reference, which is what signing trades by the Lee-Ready rule needs alongside a tape. Top of book ONLY: no shipped provider exposes depth, so queue position and resting size at a level are not recoverable from this and should not be inferred from it.

**Required:** `start_date`, `end_date`, `run_id`, `name`, `symbol`  
**Optional:** `limit`

#### `fetch_returns_panel`

Fetch a universe and publish a wide date-by-ticker frame of returns as a returns_panel reference. This is the shape most panel analysis wants -- PCA, correlation, factor regressions and portfolio construction all consume it directly -- so computing it once and handing over the reference avoids every consumer rebuilding it from prices.

**Required:** `start_date`, `end_date`, `run_id`, `name`, `tickers`  
**Optional:** `interval`

#### `fetch_tick_tape`

Fetch individual trades and publish them as a tick_tape reference, for the microstructure tools that measure rather than estimate. Needs a provider with a tick feed. A tape is large, so `limit` caps it -- and when the cap is hit the result says so, because a truncated tape makes every rate and total computed from it understate the real one.

**Required:** `start_date`, `end_date`, `run_id`, `name`, `symbol`  
**Optional:** `limit`

#### `get_dataset_metadata`

What the active provider GUARANTEES about the data it serves: whether prices are adjusted, whether the universe is survivorship-free, whether values are point-in-time, and which timezone stamps them. Read this before trusting a backtest over history, because a provider that is not point-in-time will hand you restated values under their original dates.

**Required:** `symbol`  
**Optional:** `interval`

#### `infer_temporal_contract`

Read a published frame's own columns and report what they imply about when each row became knowable. For data this library did not fetch -- a vendor extract, another system's output -- where no provider contract exists. Inference reads COLUMNS, so it can only say what is present and never what a source guarantees; prefer get_dataset_metadata whenever the data came from a known provider.

**Required:** `ref`  
**Optional:** `source`, `frame_kind`, `entity_scoped`

#### `validate_data_bundle`

Whether a bundle is safe to model on, returned as a verdict with the blocking reasons rather than raised as an error, because the answer is usually yes-with-caveats and a caller needs the caveats to decide. `require_pit` defaults to false: no shipped provider reports point-in-time for every frame kind, so requiring it refuses almost everything -- set it when a leakage-free join is the point.

**Required:** `ref`  
**Optional:** `require_pit`

#### `validate_financial_ratios`

Check ratios you already hold -- from a vendor, a spreadsheet, another system -- for values that are implausible on their face, without fetching anything. The same check fetch_financial_ratios applies, available for data this library has no provider for.

**Required:** `ratios`

---

## `derivatives` — Derivatives

Price an option and understand what holding it does to you: the second-order greeks, multi-leg payoffs, the consistency of a quoted surface, what the market is pricing as a move, and what a delta hedge costs to run. Takes quotes as arguments rather than fetching a chain -- this library has no options data provider, and a tool that pretended to would compute a chain that does not exist.

#### `analyze_option_strategy`

The payoff, breakevens and aggregate greeks of an arbitrary multi-leg position -- any combination of calls, puts and stock, rather than a fixed menu of named structures. Breakevens are found numerically, and an unbounded loss is REPORTED as unbounded: a short call has no worst case, so returning the edge of the scanned range as 'max loss' would be a finite number standing in for an infinite risk.

**Required:** `legs`, `spot`  
**Optional:** `risk_free_rate`, `dividend_yield`, `spot_range`

#### `analyze_vol_term_structure`

Contango or backwardation, and the FORWARD volatilities between expiries -- which is what a calendar spread actually prices. A trader seeing 30-day IV at 25 and 60-day at 28 is not being offered 28 for the second month; they are being offered 30.6, whatever makes total variance add up. Trading off the quoted levels can reverse the sign of the position. Negative forward variance is reported as the calendar arbitrage it is.

**Required:** `implied_by_expiry`

#### `check_put_call_parity`

Whether a call and a put on the same strike are mutually consistent, and what the violation is worth. This is a MODEL-FREE identity -- it follows from the payoffs alone and holds under any distribution -- which makes it the right first check on a quoted chain. The result names the likely causes before the arbitrage, because a break is a stale quote, a mismatched timestamp, a wrong dividend or a hard borrow far more often than it is free money; the implied dividend and forward are returned so the cause is identifiable rather than merely flagged.

**Required:** `call_price`, `put_price`, `spot`, `strike`, `time_to_expiry`, `risk_free_rate`  
**Optional:** `dividend_yield`, `tolerance_bps`

#### `fit_volatility_smile`

Fit a quoted smile as a quadratic in LOG-MONEYNESS and check it for arbitrage. Log-moneyness rather than strike because the smile is roughly symmetric in log(K/F) and emphatically not in K -- a parabola in strike puts its vertex at a fixed price, so the same shape refit after a 10% rally reports a different skew. Durrleman's condition is evaluated across the fitted range: a violation means the quotes admit a butterfly arbitrage, which in practice means one of them is stale.

**Required:** `strikes`, `implied_vols`, `forward`, `time_to_expiry`

#### `get_expected_move`

The move the option market is pricing over a horizon, with the standard misreading pre-empted. The number is ONE STANDARD DEVIATION and it gets quoted as 'the expected move' and then read as a bound -- under the model's own assumptions it is exceeded about 32% of the time, one earnings print in three. Both conventions are returned (the straddle approximation is 80% of the 1-sd move) because confusing them misprices event trades. Pass realized_moves for the historical exceedance rate rather than the lognormal one.

**Required:** `spot`, `implied_vol`, `days`  
**Optional:** `realized_moves`

#### `get_implied_forward`

The forward implied by carry, with financing, dividend and borrow broken out separately. When a quoted future disagrees with the computed forward the question is always WHICH term is wrong, and a single number cannot answer it. Borrow is kept apart from the dividend on purpose: a listed name's dividend is a known cash amount, while borrow floats and can move hundreds of basis points in a day on a squeezed stock.

**Required:** `spot`, `time_to_expiry`, `risk_free_rate`  
**Optional:** `dividend_yield`, `borrow_rate`

#### `get_implied_volatility`

The volatility that reproduces an observed option price. Solved by bisection on a monotone function, so it either converges or says it did not -- a price below intrinsic has no implied vol at all, and that is a refusal rather than a number.

**Required:** `option_price`, `spot`, `strike`, `time_to_expiry`, `risk_free_rate`  
**Optional:** `option_type`, `dividend_yield`

#### `get_option_greeks`

The SECOND-ORDER greeks -- vanna, volga, charm and speed -- that explain why a delta-hedged book still loses money. Delta and gamma tell you today's risk; these tell you how that risk CHANGES. Vanna is why a vol spike forces a rehedge on a delta-flat book, volga is why short wings lose more than the vega number suggested, and charm is why a Friday delta-flat book opens Monday short with no move in the underlying. Units are stated per greek because there is no convention and the mismatch is a real source of error.

**Required:** `spot`, `strike`, `time_to_expiry`, `volatility`  
**Optional:** `risk_free_rate`, `option_type`, `dividend_yield`

#### `get_option_pricing`

Price a European option and return its first-order greeks, under Black-Scholes, Black-76, Bachelier or a binomial tree. `volatility` means different things to different models and no type system catches it: the lognormal models take a RELATIVE vol (a fraction of the underlying per year), Bachelier takes an ABSOLUTE one in the underlying's own units, and passing 0.30 to Bachelier on an $80 future means 30 cents of annual vol rather than 30%.

**Required:** `spot`, `strike`, `time_to_expiry`, `risk_free_rate`, `volatility`  
**Optional:** `option_type`, `dividend_yield`, `model`, `american`, `binomial_steps`

#### `get_option_risk_scenarios`

A full REVALUATION grid over spot and volatility, not a delta-gamma approximation of one. Under a 20% move the Taylor estimate overstates a long call's gain by 5%, and by 11% at 30% -- the error grows with the cube of the move, which is why a stress test built on greeks understates a real gap. The two axes are shocked independently and the market does not move that way: read the down-spot/up-vol diagonal, not a row.

**Required:** `spot`, `strike`, `time_to_expiry`, `volatility`  
**Optional:** `risk_free_rate`, `option_type`, `quantity`, `spot_shocks`, `vol_shocks`, `days_forward`

#### `get_volatility_cone`

Where today's implied vol sits inside this name's own history of REALIZED vol, horizon by horizon. 'IV is 30' means nothing until you know this underlying's 30-day realized vol has spent two years between 18 and 55. Reports independent_windows next to each percentile, because rolling windows overlap and the observation count overstates the confidence by roughly the horizon.

**Required:** `prices`  
**Optional:** `horizons`, `current_implied`

#### `simulate_delta_hedge`

What a delta-hedged short option actually earns when the vol you sold at is not the vol that shows up. The expectation is known in closed form; the DISPERSION is not, and it is what decides whether the trade is sized correctly. Discrete hedging error scales as 1/sqrt(n_hedges), so going from daily to twice-daily cuts the standard deviation by 29% rather than by half while doubling the transaction cost -- that tradeoff is the reason to simulate rather than compute.

**Required:** `spot`, `strike`, `time_to_expiry`, `implied_vol`, `realized_vol`  
**Optional:** `risk_free_rate`, `option_type`, `n_hedges`, `n_paths`, `transaction_cost_bps`, `seed`

---

## `feature_lab` — Feature Lab

Interrogate the FEATURES of a built dataset, before and independently of fitting anything: what each one measures and predicts, which are restatements of one another, whether one has drifted or only worked in one regime, whether its IC is larger than what this panel's noise produces, and what each is worth to a fitted model. Exploratory and repeatable, where `modeling` is one ordered pipeline.

#### `compare_feature_sets`

Two feature sets measured on the same panel, with the cost of the difference attached: per-set IC, independent-signal count and condition number, what is unique to each side, and the per-feature IC table. Not a single score, because a larger set almost always has a higher maximum IC and almost always more collinearity, and one number hides half of that trade.

**Required:** `dataset_id`, `left`, `right`  
**Optional:** `cluster_threshold`

#### `get_feature_drift`

Whether a feature is still the same measurement, and still predicts, either side of a date. Returns PSI and a two-sample KS for the distribution, plus the IC computed separately on each half. The two fail differently and need different fixes: distribution drift with a stable IC is a preprocessing problem, while a stable distribution with a collapsed IC means the edge is gone.

**Required:** `dataset_id`, `feature`  
**Optional:** `split_date`, `method`

#### `get_feature_ic_decay`

How one feature's IC behaves when the feature is displaced in time. Answers two questions: whether it leaks (an IC that spikes at shift 0 and collapses on both sides already contains the answer) and whether it is tradeable (how much IC survives one bar of staleness). Returns the curve as ordered points with the peak named.

**Required:** `dataset_id`, `feature`  
**Optional:** `max_shift`, `method`

#### `get_feature_redundancy`

Which features are restatements of one another, and which one to keep. RSI, 20-day momentum, MACD and stochastic are one momentum cluster, not four independent sources of alpha. Returns each cluster with a representative chosen by strongest rank IC, the drop list already worked out, and the collinearity diagnostics (VIF, condition number) that say whether linear coefficients on this panel mean anything.

**Required:** `dataset_id`  
**Optional:** `features`, `cluster_threshold`

#### `get_feature_regime_stability`

The feature's IC inside each of several CONTIGUOUS time blocks, never shuffled -- a feature's usual problem is that it worked in one regime, and interleaved folds average exactly that away. Returns per-block IC plus sign consistency against the full-sample IC. Read both: consistent sign with collapsing magnitude is decay, and sign consistency stays at 1.0 through it.

**Required:** `dataset_id`, `feature`  
**Optional:** `n_blocks`, `method`

#### `profile_feature`

Profile ONE feature of a built dataset: coverage, turnover, autocorrelation, cross-sectional IC and ICIR, quantile spread and monotonicity. The single-feature counterpart to analyze_features — reach for it when a specific feature is in question, since it costs one feature's work instead of the whole panel's and returns every number named rather than nested in a report dict.

**Required:** `dataset_id`, `feature`  
**Optional:** `n_quantiles`, `include_ic_decay`, `max_shift`

#### `run_feature_ablation`

Refit the model without each feature in turn and report what each one was worth. The only feature tool that asks a MODEL-RELATIVE question: a strong feature that duplicates another contributes nothing marginal, and a mediocre feature that is the sole source of some information can be the one holding the model up. Neither shows in a per-feature report or in tree importance. EXPENSIVE -- one baseline plus one refit per feature across every fold, so 40 features at 8 folds is 328 fits. The count is computed before anything is fit and the run is REFUSED past max_fits, so narrow `features` to the candidates you actually doubt.

**Required:** `dataset_id`, `spec`  
**Optional:** `features`, `metric`, `max_fits`

#### `run_feature_permutation_test`

How often noise on THIS panel produces an IC as large as the observed one, in either direction. Shuffles the feature within each date, which states the null exactly -- the feature carries no cross-sectional information within a date -- and returns a TWO-SIDED empirical p-value, so a strongly negative IC is significant rather than ignored. null_p95_abs is the IC this panel yields from noise alone 5% of the time, which is the defensible floor for select_features(min_abs_rank_ic=...). Cost is linear in n_permutations.

**Required:** `dataset_id`, `feature`  
**Optional:** `n_permutations`, `method`, `random_seed`

#### `select_features`

Choose a feature set from a built dataset: keep one feature per redundancy cluster, drop what falls below an IC floor, and return a reason for every exclusion. Deliberately has no greedy search -- a selector scored on the panel it selects from manufactures overfit that looks like evidence. Redundancy is resolved before the IC floor, because a cluster is one signal and the question is whether THAT signal clears the floor.

**Required:** `dataset_id`  
**Optional:** `features`, `cluster_threshold`, `min_abs_rank_ic`, `max_features`
