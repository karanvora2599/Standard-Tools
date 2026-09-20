# Survey: `backtest/`, `backtesting/`, `portfolio/`, `screener/` — implemented vs. exposed

Slice: `src/standard_quant_tools/{backtest,backtesting,portfolio,screener}/` — 11,307 lines across 28 modules
(backtest 6,214 · backtesting 1,480 · portfolio 2,998 · screener 615).

Method: every module was read in full. Exposure was determined by grepping import lines and call sites in
every file under `agent/runtimes/**` (not only `tools.py` — the backtest runtime also has `trade_tools.py`,
`validation_tools.py`, `futures_tools.py`, `terminal_mc_tools.py`; the portfolio runtime has
`construction_tools.py`, `weight_tools.py`; meta has `convert.py`; plus `agent/runtimes/handoff.py`,
`agent/runtimes/_shared.py`), `modeling/agent/tools.py`, `modeling/portfolio_eval.py`, `modeling/bridge.py`
and `modeling/artifacts.py`. Tool parameter lists were cross-checked against `Documentation/20_tool_index.md`.
Nothing in the repository was modified.

## Headline counts

| | Count |
|---|---:|
| Public functions / classes / registries inventoried | **85** |
| Private helpers carrying real capability (listed separately) | 15 |
| Directly reached by at least one tool (call by name or via `lib.`) | **74** |
| Reached only indirectly (inside another library function; the number itself never surfaced) | **8** |
| Not reached by any tool | **3** |
| Directly reached but only in a **shape-limited** form (see §5) | 9 of the 74 |

The three unreached items are `walk_forward.stitch_oos_returns`, `walk_forward.compute_stitched_metrics`
(both "kept as general-purpose utilities", exercised only by tests) and the `strategy.VectorizedStrategy`
Protocol (a type, nothing to expose). The real gaps in this slice are not unexposed *functions* — the surface
is unusually complete — but unexposed *combinations* and *shapes*: results that exist as `sqt://` artifacts
which the judgement tools cannot consume, an optimizer that never sees the shrunk covariance the library can
produce, a frontier that is solved but never traced, and strategies that can only be run one ticker per
cash account.

## 1. Which tool modules reach this slice

| Tool module | Slice functions it imports / calls |
|---|---|
| `agent/runtimes/_shared.py` | `engine.run_strategy` (the `_run_backtest` helper every single-name strategy tool uses) |
| `agent/runtimes/backtest/tools.py` | `artifacts.{save,load}_artifact`, `engine.{run_strategy,backtest_grid}`, `monte_carlo.simulate_forward_paths`, `pairs.run_pair_backtest`, `panel.run_signal_panel_backtest`, `portfolio_engine.run_portfolio_simulation`, `robustness.{block_bootstrap_ci,deflated_sharpe_ratio,parameter_sensitivity}`, `sizing.*` (all 5), `strategies.{RUNNABLE,STRATEGY_REGISTRY}`, `walk_forward.{longest_losing_streak,parameter_turnover}`, `portfolio.portfolio.{build_portfolio,fetch_ohlcv_panel_sync,fetch_returns_sync}` |
| `agent/runtimes/backtest/trade_tools.py` | `backtesting.trade_analysis` (all 5 public functions) |
| `agent/runtimes/backtest/validation_tools.py` | `backtesting.overfitting` (all 6 public functions) |
| `agent/runtimes/backtest/futures_tools.py` | `futures_engine.run_futures_simulation`, `futures_hedge_backtest.run_futures_hedge_backtest` |
| `agent/runtimes/backtest/terminal_mc_tools.py` | `monte_carlo.simulate_forward_paths_terminal` |
| `agent/runtimes/portfolio/tools.py` | `constraints.{capacity_report,days_to_liquidate,sector_exposure}`, `costs.*` (9 public cost functions), `liquidity.{amihud_illiquidity,corwin_schultz_spread}`, `stress_test.{replay_stress_scenario,scenario_dates}`, `optimize.{annualized_mean_cov,black_litterman,build_bl_views,mean_variance_optimize,risk_parity_weights,_check_covariance_estimable,_small_sample_warnings}`, `rebalance.plan_rebalance`, `covariance.estimate_covariance`, `portfolio.fetch_returns_sync` |
| `agent/runtimes/portfolio/construction_tools.py` | `portfolio.construction` (all 8 public functions) |
| `agent/runtimes/portfolio/weight_tools.py` | `backtest.sizing` (all 5) |
| `agent/runtimes/research/tools.py` | `artifacts.save_artifact`, `portfolio.{fetch_returns_sync,fetch_ohlcv_panel_sync,portfolio_metrics,correlation_matrix}`, `screener.screen_stocks` |
| `agent/runtimes/research/reference_tools.py` | `portfolio.fetch_ohlcv_panel_sync` |
| `agent/runtimes/data/tools.py` | `portfolio.{fetch_ohlcv_panel_sync,fetch_returns_sync}` |
| `agent/runtimes/meta/tools.py` | `artifacts.load_artifact`, `strategy_params.{STRATEGY_PARAM_SCHEMA,_RELATIONS,_MAX_WINDOW_BARS,resolve_strategy_params}`, `stress_test.list_stress_scenarios` |
| `agent/runtimes/meta/convert.py` | `artifacts.save_artifact`, `sizing.{rank_weighted,vol_scaled,zscore_normalized}` (not `equal_weight_top_bottom`, not `dollar_neutral`) |
| `agent/runtimes/handoff.py` | `artifacts.{save_artifact,load_artifact,_runs_dir,_resolved_within_runs_dir,_validate_identifier}` — every `*_ref` parameter in every runtime resolves through this |
| `modeling/portfolio_eval.py` → `evaluate_model_portfolio` (modeling) | `portfolio_engine.run_portfolio_simulation`, `sizing.{equal_weight_top_bottom,rank_weighted,vol_scaled,zscore_normalized}` |
| `modeling/bridge.py` → `convert_reference`, `run_signal_panel_backtest(signal_panel_ref=…)` | produces the `{-1,0,+1}` panel that `panel.run_signal_panel_backtest` consumes; imports nothing from the slice directly |
| `modeling/artifacts.py` (every modeling tool that persists/reads) | `artifacts.{save_artifact,load_artifact}` |

## 2. Inventory — `backtest/`

Legend for the *exposed by* column: **direct** = a tool calls it by name; **indirect** = only reached inside
another library function; **—** = not reached. "(shape-limited)" = reachable but only in one fixed
configuration; see §5.

| module | function / class | purpose | inputs → outputs | exposed by |
|---|---|---|---|---|
| artifacts.py | `save_artifact(data, run_id, name, overwrite=False)` | Atomic Parquet write under `SQT_RUNS_DIR/<run_id>/<name>.parquet`; Series → 1-col frame; refuses empty / existing unless `overwrite` | Series/DataFrame → path str | **direct**: `run_backtest_compact` (equity_curve, trades), `get_technical_panel`, `handoff.publish` (all runtimes), meta `convert_reference`, modeling `build_model_dataset`/`join_point_in_time`/`evaluate_model_portfolio` |
| artifacts.py | `load_artifact(uri)` | Read back; path-traversal guarded to runs dir | uri → DataFrame | **direct**: `get_drawdown_table`, meta `describe_artifact`, `handoff.resolve` (every `*_ref`), modeling `run_model_experiment`, `monitor_model`, `evaluate_model_portfolio` |
| constraints.py | `adv_participation(notional, avg_dollar_volume)` | Fraction of ADV a trade takes; NaN (not 0) when ADV unusable | floats → float/NaN | **indirect**: inside `run_portfolio_simulation` Python loop when `max_adv_participation` set (native kernel re-implements). No tool reports per-trade participation from it. |
| constraints.py | `days_to_liquidate(shares, avg_daily_volume, max_participation)` | Days to unwind at a participation cap; raises on non-finite/≤0 | floats → float | **direct**: `get_capacity_report` |
| constraints.py | `sector_exposure(weights, sectors)` | Sum weight by sector, "Unknown" bucket | dicts → dict | **direct**: `get_capacity_report(include_sector_exposure)` |
| constraints.py | `capacity_report(tickers, avg_dollar_volumes, target_weights, max_participation)` | Per-ticker max account size, binding ticker, overall capacity | → dict | **direct**: `get_capacity_report` |
| costs.py | `percentage_commission`, `per_share_commission`, `fixed_bps_spread`, `pct_of_range_spread`, `impact_cost`, `short_borrow_cost`, `margin_interest`, `directional_commission`, `maker_taker_cost` | Composable cost primitives (validated via `_cost_rate`; negative rates refused except maker rebate) | scalars → float ($) | **direct**: `estimate_trade_cost` (all 9). Also inside `run_portfolio_simulation` (pct/per-share/impact inlined; `margin_interest`, `short_borrow_cost` called) |
| costs.py | `sqrt_impact_bps(participation, volatility, coefficient)` | Square-root impact in bps (`coef·vol·√part·1e4`) | → float (bps) | **indirect**: via `impact_cost` (→ `estimate_trade_cost`, `get_liquidity_adjusted_var`). `plan_rebalance` re-parameterises it (documented identity). No tool returns the bps figure itself. |
| costs.py | `_cost_rate(name, value, allow_negative)` (private) | The one validator every rate passes; imported by `portfolio_engine` | → float | internal (validation only) |
| engine.py | `run_strategy(price_data, signal_series, initial_capital, commission_pct, slippage_pct, include_trade_log, fill_price, risk_free_rate)` | Vectorised single-asset engine; C++ kernel for all three fill modes; refuses total-loss bars; look-ahead warnings for `close`/`hl2_exploratory` | OHLCV + {-1,0,1} → dict(final_equity, total_return, ann_vol, sharpe, sortino, max_dd, calmar, win_rate, profit_factor, num_trades, avg_trade_return_pct, equity_curve, warnings[, trade_log]) | **direct**: `_shared._run_backtest` → `run_sma/rsi/macd/bollinger_backtest`, `run_buy_and_hold`, `compare_strategies`, `run_custom_signal_backtest`; `run_regime_adaptive_backtest`, `run_regime_adaptive_walkforward_backtest`, `run_walk_forward_backtest`, `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models`, `run_strategy_matrix`, `get_robustness_diagnostics`; indirectly via `panel.run_signal_panel_backtest` |
| engine.py | `backtest_grid(price_data, strategy, param_grid, …, sort_by, ascending, n_workers, fill_price, risk_free_rate)` | Parameter sweep; C++ batch kernel; fused SMA-crossover path; `strategy` may be a registry name **or any callable** | → DataFrame (metrics × params, sorted) | **direct**: `run_backtest_optimization`, `run_walk_forward_backtest`, `run_regime_adaptive_*` (×2), `get_robustness_diagnostics`. The **custom-callable path is not exposed** (JSON boundary) — see §6 P8. |
| engine.py | `_build_trade_log(ref_prices, close_prices, executed, cost_per_unit)` (private) | Lot-based trade log reconciled to equity P&L (resizes stay in-lot; MTM flush) | → DataFrame | internal; also imported by `metrics.diagnostics` (→ `get_backtest_diagnostics`) |
| engine.py | `_compute_trade_stats`, `_fused_crossover_metrics`, `_run_grid_job`, `_run_signal_fn_job` (private) | Trade stats; fused unique-SMA grid; pool workers | — | internal. **Bug**: `_run_signal_fn_job` does not forward `risk_free_rate` (custom-callable Python-fallback grids rank on a zero-rate Sharpe while the registry path uses the real one — exactly the inconsistency `_run_grid_job`'s comment says it fixed). |
| futures_engine.py | `run_futures_simulation(prices, target_contracts, multiplier, initial_capital, initial_margin, maintenance_margin, commission_per_contract, slippage_points, collateral_rate, contract_map, allow_fractional)` | Margin/variation-margin futures account; rolls via `contract_map`; margin calls reduce position; equity = cash + margin | → dict(equity/cash/margin/position/exposure/leverage curves, totals, margin_calls, rolls, warnings) | **direct**: `run_futures_backtest`; via `run_futures_hedge_backtest` |
| futures_hedge_backtest.py | `run_futures_hedge_backtest(portfolio_values, future_prices, multiplier, portfolio_beta, future_beta, initial_margin, …, rehedge, drift_band, allow_fractional)` | Cash book + futures hedge carried together; re-hedge rules daily/weekly/monthly/drift; two P&L streams kept apart; `hedge_effectiveness` from `delta_one.hedging` | → dict(cash_pnl, hedge_pnl, combined_pnl, vol reduction, residual_beta, effective_hedge_ratio, margin stats, warnings) | **direct**: `run_futures_hedge_backtest` |
| futures_hedge_backtest.py | `_rehedge_dates(index, rule, residual_fraction, band)` (private) | Which bars re-size | → bool array | internal |
| liquidity.py | `amihud_illiquidity(returns, dollar_volume, window)` | Amihud ratio ×1e6, rolling mean; non-positive volume → NaN | → Series | **direct**: `get_liquidity_metrics`, `check_spread_proxy` |
| liquidity.py | `corwin_schultz_spread(high, low, window)` | Per-bar CS spread (series shape of the estimator in `analysis.microstructure_estimators`), clipped ≥0 | → Series (fraction) | **direct**: `get_liquidity_metrics`, `check_spread_proxy` |
| monte_carlo.py | `simulate_forward_paths(returns, horizon_days, n_simulations, block_size, initial_capital, seed)` | Moving-block bootstrap of a return series → full path matrix; terminal stats + p5/p50/p95 equity bands | → dict | **direct (shape-limited)**: `run_monte_carlo_simulation` — only from `tickers+weights` (historical `build_portfolio`), never from a strategy equity/return **ref** |
| monte_carlo.py | `simulate_forward_paths_terminal(…)` | Same bootstrap, terminal-only, O(n_sim) memory | → dict (no bands) | **direct**: `run_terminal_monte_carlo` — takes a `DataSource` (symbol / `sqt://` ref / values) |
| pairs.py | `run_pair_backtest(price_data, symbol_a, symbol_b, hedge_ratio, entry_z, exit_z, zscore_window, initial_capital, commission_pct, slippage_pct, gross_leverage, fill_price)` | Two-leg pair as a 2-asset `run_portfolio_simulation`; share-ratio → dollar weights at the actual execution price; rolling z-score default (full-sample z leaks) | → portfolio-sim result + hedge_ratio, entry_spread, current_spread, n_round_trips, state | **direct**: `run_pair_trade_backtest` (all params forwarded) |
| pairs.py | `_spread_state(z, entry_z, exit_z)` (private) | Stateful z-score entry/exit machine (NaN holds state) | → Series {-1,0,1} | internal to `run_pair_backtest`; nothing exposes the *current* spread state without running a full backtest |
| pairs.py | `_execution_prices_for_weights(…)` (private) | Fill-mode-aware price for sizing the hedge leg | → (pa, pb) | internal |
| panel.py | `run_signal_panel_backtest(price_data, signal_panel, weights, initial_capital, commission_pct, slippage_pct, benchmark_returns, include_trade_log, fill_price, signal_calendar_policy, risk_free_rate)` | Per-ticker `run_strategy` on a {-1,0,1} panel (each ticker its OWN capital), then `portfolio_metrics`; reindexes sparse signals onto each ticker's calendar (hold/flat/error) | → dict(tickers, per_ticker, portfolio_returns, portfolio_metrics) | **direct**: `run_signal_panel_backtest` (`signal_fill_policy` ↔ `signal_calendar_policy`; accepts `signal_panel_ref` from modeling `convert_reference`) |
| panel.py | `_align_signal_to_calendar(signal, price_index, policy, ticker)` (private) | The reindex; duplicated by `backtest/tools.py::_apply_signal_fill_policy` | → Series | internal (see §8) |
| portfolio_engine.py | `run_portfolio_simulation(price_data, target_weights, initial_capital, commission_pct, sell_commission_pct, slippage_pct, max_gross_leverage, max_position_pct, fill_price, commission_model, per_share_rate, min_commission, use_impact_model, impact_coefficient, impact_lookback, borrow_fee_bps, margin_interest_rate, max_adv_participation)` | One shared cash account rebalanced at `target_weights.index`; drift between rebalances; C++ kernel for pct-commission/no-impact/no-ADV configs; insolvency refuses; ADV cap fails closed | → dict(equity/cash/gross/net/leverage curves, rebalance_log, final_equity, final_cash, max_leverage, max_gross_exposure, peak_position_value, return_over_rebalance, warnings) | **direct**: `run_portfolio_simulation` (all 18 params forwarded; `target_weights_ref` or inline, or built from a score panel via `construction_method`), `run_pair_trade_backtest` (via pairs), modeling `evaluate_model_portfolio` |
| portfolio_engine.py | `_native_portfolio_sim`, `_raise_portfolio_error` (private) | Kernel dispatch + status → message | — | internal |
| robustness.py | `block_bootstrap_ci(returns, metric_fn, n_iterations, block_size, confidence, seed)` | Block-bootstrap CI for ANY metric callable | → dict(point, ci_lower, ci_upper, …) | **direct (shape-limited)**: `get_robustness_diagnostics` only, `metric_fn` fixed to `sharpe_ratio`, on the re-run best grid row. (Research `get_bootstrap_interval` uses `analysis.inference`, not this.) |
| robustness.py | `parameter_sensitivity(grid_df, metric_col)` | best − median / rank-2 / top-5 gaps from a grid | → dict | **direct (shape-limited)**: `get_robustness_diagnostics` only — always re-runs the whole grid; cannot be applied to a grid already run by `run_backtest_optimization` |
| robustness.py | `expected_max_sharpe(trials_std, n_trials)` | Bailey–LdP expected max of N null Sharpes | → float | **indirect**: both DSR implementations |
| robustness.py | `sharpe_standard_error_factor(observed_sharpe, skew, kurtosis)` | Non-normality SE factor | → float | **indirect**: both DSR implementations |
| robustness.py | `deflated_sharpe_ratio(observed_sharpe, sharpe_trials_std, n_trials, n_obs, skew, kurtosis)` | Grid-form DSR (takes the trial Sharpe std) | → dict | **direct (shape-limited)**: `get_robustness_diagnostics` only |
| sizing.py | `rank_weighted(scores, gross_leverage)` | Centred cross-sectional rank weights, Σ|w| = gross | date×ticker → date×ticker | **direct**: `run_portfolio_simulation(construction_method)`, `construct_weights_from_scores`, meta `convert_reference`, modeling `evaluate_model_portfolio` |
| sizing.py | `equal_weight_top_bottom(scores, n_long, n_short, gross_leverage)` | Top/bottom N equal weight; long-only gets full gross | → panel | **direct**: `run_portfolio_simulation`, `construct_weights_from_scores`, `evaluate_model_portfolio`. **Not** in `convert_reference`. |
| sizing.py | `zscore_normalized(scores, gross_leverage)` | z-score weights; degenerate rows → 0 | → panel | **direct**: same four as `rank_weighted` |
| sizing.py | `vol_scaled(scores, returns_df, lookback, gross_leverage)` | Score / trailing vol, rolled on the returns calendar first | → panel | **direct**: `run_portfolio_simulation`, `construct_weights_from_scores(returns_ref)`, `evaluate_model_portfolio`; `convert_reference` refuses it (no returns in a score panel) |
| sizing.py | `dollar_neutral(weights)` | Mean-centre then rescale to original gross | → panel | **direct**: `run_portfolio_simulation(make_dollar_neutral)`, `construct_weights_from_scores(dollar_neutral)`. **Not** in `convert_reference`. |
| strategies.py | `STRATEGY_REGISTRY` (8, each wrapped by `_validating` → `resolve_strategy_params(check_relations=False)`) | Signal generators; see §4 | OHLCV, **params → Series {0,1} | **direct**: see §4 |
| strategies.py | `BASELINE_REGISTRY` = {`buy_and_hold`} | Long every bar, refuses params | → Series of 1.0 | **direct** via `RUNNABLE`: `run_buy_and_hold`; `compare_strategies` builds it separately. **Rejected** by `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models`, `run_strategy_matrix` (they gate on `STRATEGY_REGISTRY`) |
| strategies.py | `RUNNABLE` = registry ∪ baselines | Dispatch set | — | **direct**: `_dispatch_backtest` (4 strategy tools + `run_buy_and_hold`) |
| strategies.py | `_rsi/_bollinger/_donchian/_vwap_reversion_state_machine` (numba/C++), `_*_signals` raw fns (private) | Hysteresis kernels; unvalidated signal fns | — | internal by design (module says "never call these directly") |
| strategy.py | `VectorizedStrategy` (Protocol) | Type for `(df, **params) -> Series`; explicitly NOT an event-driven engine | — | **—** (type only; tests only) |
| strategy_params.py | `STRATEGY_PARAM_SCHEMA`, `_RELATIONS`, `_MAX_WINDOW_BARS`, `_Param` | Declared per-strategy contract (windows ≥1, finite thresholds, relations) | — | **direct**: meta `list_strategies`, `validate_tool_call` |
| strategy_params.py | `resolve_strategy_params(strategy, params, check_relations=True)` | Validate + merge defaults; rejects unknown names, negative windows (forward look-ahead), NaN thresholds | → dict | **direct**: meta `validate_tool_call` (relations ON). Inside every registry call (relations OFF). **No backtest tool enforces relations itself** — see §8. |
| stress_test.py | `list_stress_scenarios()` | 6 named windows (1987, dotcom, GFC, volmageddon, covid, 2022) | → dict | **direct**: meta `list_stress_scenarios` |
| stress_test.py | `scenario_dates(name)` | Lookup | → (start, end) | **direct**: `run_stress_test` |
| stress_test.py | `replay_stress_scenario(returns_df, weights)` | Weighted replay over a sliced window; MDD seeded with 1.0; `_pct` fields are FRACTIONS | → dict | **direct**: `run_stress_test` (named scenario or `custom_start_date/custom_end_date`) |
| walk_forward.py | `stitch_oos_returns(window_returns)` | concat+sort non-overlapping window return series | list[Series] → Series | **—** (tests only) |
| walk_forward.py | `compute_stitched_metrics(oos_returns, initial_capital)` | One compounded curve → total_return, sharpe, sortino, max_dd, calmar | → dict | **—** (tests only; the walk-forward tools re-run `run_strategy` over the OOS span instead) |
| walk_forward.py | `longest_losing_streak(window_returns)` | Consecutive losing windows | list[float] → int | **direct**: `run_walk_forward_backtest`, `run_regime_adaptive_walkforward_backtest` |
| walk_forward.py | `parameter_turnover(window_params)` | Fraction of window transitions whose best params changed | list[dict] → float | **direct**: `run_walk_forward_backtest` |

## 3. Inventory — `backtesting/`, `portfolio/`, `screener/`

| module | function / class | purpose | inputs → outputs | exposed by |
|---|---|---|---|---|
| backtesting/overfitting.py | `deflated_sharpe_ratio(returns, n_trials, trial_sharpes, benchmark_sharpe, periods_per_year)` | Returns-form DSR: computes skew/kurtosis from the series; uses trial-Sharpe variance when supplied | Series → dict(observed, expected_max_from_luck, threshold, probability, significant_at_95, …) | **direct**: `get_deflated_sharpe_ratio` (inline `returns: List[float]` only) |
| backtesting/overfitting.py | `probability_of_backtest_overfitting(trial_returns, n_splits)` | CSCV PBO over a configurations×periods frame; median pairwise correlation reported | DataFrame → dict | **direct**: `estimate_backtest_overfitting` (inline dict of lists) |
| backtesting/overfitting.py | `combinatorial_purged_cv(n_observations, n_splits, n_test_splits, embargo_pct, label_horizon)` | Purged + embargoed combinatorial train/test index sets | ints → dict(paths[…]) | **direct**: `build_purged_cv_splits` |
| backtesting/overfitting.py | `reality_check(strategy_returns, benchmark_returns, n_bootstrap, block_size, seed)` | White's Reality Check, block bootstrap | Series, DataFrame → dict(p_value, …) | **direct**: `run_reality_check` (inline lists) |
| backtesting/overfitting.py | `regime_stratified_performance(returns, regimes, periods_per_year)` | Per-regime Sharpe/P&L share; concentration flag | Series, Series → dict | **direct**: `get_regime_stratified_performance` (inline) |
| backtesting/overfitting.py | `parameter_decay(parameter_values, performance, metric_name)` | 1-D spike ratio / plateau / edge test | lists → dict | **direct**: `analyze_parameter_decay` (inline) |
| backtesting/overfitting.py | `_clean_returns`, `_sharpe`, `_block_ends` (private) | validators/helpers | — | internal |
| backtesting/trade_analysis.py | `monte_carlo_trade_paths(trade_returns, n_paths, seed, starting_equity)` | Reshuffle (not resample) trades → drawdown distribution | list → dict | **direct**: `run_monte_carlo_trade_paths` (inline `List[float]`) |
| backtesting/trade_analysis.py | `analyze_trade_clustering(trade_returns)` | Runs test on win/loss order; streaks | list → dict | **direct**: `analyze_trade_clustering` (inline) |
| backtesting/trade_analysis.py | `compare_against_random(trade_returns, n_simulations, seed)` | Random-sign null at the same win rate | list → dict | **direct**: `compare_against_random` (inline) |
| backtesting/trade_analysis.py | `exposure_attribution(returns, exposure, periods_per_year)` | E[e·r] = E[e]E[r] + Cov: passive vs timing | lists → dict | **direct**: `get_exposure_attribution` (inline) |
| backtesting/trade_analysis.py | `break_even_cost(trade_returns, current_cost_bps)` | Flat per-trade cost at which the edge vanishes; headroom multiple; cost ladder | list → dict | **direct**: `estimate_break_even_cost` (inline) |
| portfolio/covariance.py | `estimate_covariance(returns, method∈{sample,ledoit_wolf,ewma,ewma_shrunk}, halflife, periods_per_year)` | Annualised covariance + diagnostics (obs/parameter, condition number, shrinkage intensity) | DataFrame → dict(matrix, …) | **direct**: `estimate_covariance`. **Not consumed by `run_portfolio_optimization`** (which always uses the sample covariance via `annualized_mean_cov`) — see §6 P3. |
| portfolio/covariance.py | `_ewma_covariance`, `_shrink_to_identity`, `_warnings` (private) | weighted-mean EWMA; N/T shrinkage | — | internal |
| portfolio/optimize.py | `annualized_mean_cov(returns_df, periods_per_year)` | (μ, Σ) annualised | → tuple | **direct**: `run_portfolio_optimization`, `get_portfolio_risk_attribution` |
| portfolio/optimize.py | `mean_variance_optimize(returns_df, objective∈{max_sharpe,min_volatility,target_return,target_volatility}, risk_free_rate, target_return, target_volatility, allow_short, max_weight, periods_per_year)` | Markowitz: closed-form Merton frontier when unconstrained, SLSQP otherwise; independent constraint verification; conditioning + small-sample warnings | → dict(weights, expected_return/vol, sharpe, converged, warnings) | **direct**: `run_portfolio_optimization` (all params; `tickers+dates` only, no `returns_ref`) |
| portfolio/optimize.py | `risk_parity_weights(cov_matrix, risk_budget, max_iterations, tol)` | Strict-validation wrapper that delegates to `construction.risk_parity` | → dict(weights ndarray, contributions, converged) | **direct**: `run_portfolio_optimization(method="risk_parity")` |
| portfolio/optimize.py | `black_litterman(cov_matrix, market_weights, P, Q, risk_aversion, tau, omega)` | He–Litterman posterior + implied weights | → dict | **direct**: `run_portfolio_optimization(method="black_litterman")` |
| portfolio/optimize.py | `build_bl_views(tickers, views, cov_matrix, tau)` | Dict views → (P, Q, Ω) with confidence scaling | → tuple | **direct**: `run_portfolio_optimization` |
| portfolio/optimize.py | `_frontier_stats`, `_frontier_weights`, `_solve_unconstrained`, `_solve_constrained` (private) | Merton A/B/C/D constants; **w(r) for any target return** (the whole frontier); SLSQP | — | internal. The frontier *trace* is solved but never returned — see §6 P4. |
| portfolio/optimize.py | `_check_covariance_estimable`, `_small_sample_warnings` (private) | n_obs ≤ n_assets refusal; 10-obs-per-asset warning | — | **direct** (private cross-module import): `run_portfolio_optimization` |
| portfolio/optimize.py | `_conditioning_warnings`, `_verify_solution`, `_require_scipy`, `_require_finite_scalar` (private) | cond > 1e10 warning; feasibility re-check | — | internal |
| portfolio/construction.py | `risk_parity(covariance, max_iterations, tolerance, budget)` | Cyclical coordinate descent ERC; budget renormalised | cov → dict(weights, risk_shares, converged, …) | **direct**: `optimize_risk_parity` (inline `covariance`), via `risk_parity_weights` |
| portfolio/construction.py | `hierarchical_risk_parity(returns, periods_per_year)` | LdP HRP; single-linkage without scipy; no inversion | returns DataFrame → dict | **direct**: `optimize_hierarchical_risk_parity` — requires **inline `returns` matrix** (no tickers/dates, no `returns_ref`) |
| portfolio/construction.py | `factor_exposure_budget(weights, factor_loadings, factor_covariance)` | Exposures and (with Σ_f) variance shares; unmapped names counted | → dict | **direct**: `get_factor_exposure_budget` |
| portfolio/construction.py | `concentration_analysis(weights)` | Effective N, Herfindahl, top-k on GROSS | → dict | **direct**: `analyze_concentration` |
| portfolio/construction.py | `liquidity_adjusted_var(positions, volatilities, daily_volumes, confidence, participation_rate, correlation, impact_coefficient)` | VaR × √(liquidation days) + separate impact cost | → dict | **direct**: `get_liquidity_adjusted_var` (`impact_coefficient` not exposed) |
| portfolio/construction.py | `max_diversification(covariance)` | Max diversification ratio (pinv of correlation) | → dict | **direct**: `optimize_max_diversification` |
| portfolio/construction.py | `marginal_risk_contribution(weights, covariance)` | Marginal / contribution / share per asset; hedges flagged | → dict | **direct**: `get_marginal_risk_contribution` |
| portfolio/construction.py | `portfolio_scenarios(weights, scenarios, covariance)` | Named shocks; coverage; σ-move | → dict | **direct**: `run_portfolio_scenarios` |
| portfolio/construction.py | `_covariance_frame`, `_portfolio_volatility`, `_risk_contributions`, `_cluster_variance`, `_quasi_diagonal_order`, `_normal_quantile` (private) | validation; ERC decomposition; O(n) single-linkage; bisection ppf | — | internal (see §8 on `_normal_quantile`) |
| portfolio/construction.py | `MIN_OBS_PER_PARAMETER = 2.0` | exported constant | — | **dead**: defined, in `__all__`, referenced nowhere |
| portfolio/portfolio.py | `build_portfolio(returns_df, weights)` | Σ w·r per day; weights must sum to 1 | → Series | **direct**: `run_monte_carlo_simulation`; indirect: `run_stress_test`, `run_signal_panel_backtest` |
| portfolio/portfolio.py | `portfolio_metrics(returns_df, weights, risk_free_rate, periods_per_year, benchmark_returns)` | Return/risk/ratio metrics on a static-weight portfolio | → dict | **direct**: `get_portfolio_analysis`; indirect: `run_signal_panel_backtest` |
| portfolio/portfolio.py | `correlation_matrix(returns_df)` | Pearson | → DataFrame | **direct**: `get_portfolio_analysis` (via `correlation_matrix_to_dict`) |
| portfolio/portfolio.py | `fetch_returns_sync` / `fetch_returns_async` | Concurrent OHLCV fetch → per-ticker pct_change THEN assemble (no fabricated 0% on halted days) | → DataFrame | **direct** (sync): `fetch_returns_panel` (data), `run_portfolio_optimization`, `estimate_covariance`, `get_portfolio_analysis`, `get_correlation_analysis`, `run_monte_carlo_simulation`, … ; async: **indirect** only |
| portfolio/portfolio.py | `fetch_ohlcv_panel_sync` / `_async` | Concurrent full-OHLCV fetch | → dict[ticker, DataFrame] | **direct** (sync): `fetch_ohlcv_panel` (data), `run_signal_panel_backtest`, `run_portfolio_simulation`, `get_technical_panel`, `compute_indicator_panel`; async: **indirect** |
| portfolio/rebalance.py | `plan_rebalance(current_weights, target_weights, portfolio_value, adv, max_participation, max_days, urgency, impact_coefficient)` | Day-by-day transition schedule under a participation cap; sqrt impact (bps-at-full-participation parameterisation); unreachable names with days needed | → dict(schedule, total_cost_bps/$, converged, unreachable, warnings) | **direct**: `plan_rebalance` |
| screener/screener.py | `screen_stocks(tickers, filters, start_date, end_date, sort_by, ascending, n_workers, min_beta_obs)` | 13 closed filter keys (7 fundamental, 6 technical incl. beta vs SPY with min-overlap guard); process pool >20 tickers; `attrs` carry failed_filters / failed_tickers / failed_batches | → DataFrame | **direct**: `run_screener` (surfaces all three failure maps) |
| screener/screener.py | `screen_stocks_async(…)` | Single-process async core | → DataFrame | **indirect** via `screen_stocks` |
| screener/screener.py | `_fetch_ticker_data`, `_screen_batch`, validators (private) | per-ticker evaluation with 3-state outcome | — | internal |

## 4. Strategy registry — what exists, what is exposed where

| Strategy (`STRATEGY_REGISTRY`) | Params (schema) | Dedicated tool | Reachable via generic tools |
|---|---|---|---|
| `sma_crossover` | fast_period, slow_period (fast < slow) | `run_sma_backtest` | `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models`, `run_strategy_matrix`, `run_backtest_optimization`, `run_walk_forward_backtest`, `get_robustness_diagnostics`, `compare_strategies`, `run_regime_adaptive_*` |
| `rsi_mean_reversion` | period, oversold, overbought (oversold < overbought) | `run_rsi_backtest` | same |
| `macd_crossover` | fast, slow, signal (fast < slow) | `run_macd_backtest` | same |
| `bollinger_reversion` | period, num_std | `run_bollinger_backtest` | same |
| `donchian_breakout` | entry_period, exit_period | — | generic tools only |
| `momentum_timeseries` | lookback, threshold | — | generic tools only |
| `vwap_reversion` | period, entry_threshold | — | generic tools only |
| `adx_trend` | adx_period, adx_threshold | — | generic tools only |
| `buy_and_hold` (`BASELINE_REGISTRY`) | none | `run_buy_and_hold` | `compare_strategies` (built separately). **Not** accepted by `run_strategy_matrix`, `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models` |

All eight strategies are long/flat only (`{0,1}`); no registry strategy emits `-1`. The engine and the panel
accept `-1`, and `run_custom_signal_backtest` / `run_signal_panel_backtest` are the only routes to a short
signal. `compare_strategies` compares only the four with dedicated tools (its inputs are
`sma_/rsi_/macd_/bollinger_parameters`), so half the registry is missing from the one "which rule?" tool;
`run_strategy_matrix` covers all eight but per ticker.

## 5. Gap analysis — grouped by theme

**A. Judgement tools cannot consume the artifacts the execution tools produce (plumbing gap, high leverage).**
`run_backtest_compact` persists `equity_curve_uri` and `trades_uri`; `run_terminal_monte_carlo` takes a
`DataSource` (symbol / `sqt://` / values) and `get_drawdown_table` takes `equity_curve_uri`. But every
`backtesting/`-backed tool — `get_deflated_sharpe_ratio`, `run_reality_check`,
`get_regime_stratified_performance`, `estimate_backtest_overfitting`, `run_monte_carlo_trade_paths`,
`analyze_trade_clustering`, `compare_against_random`, `estimate_break_even_cost`, `get_exposure_attribution`
— declares `List[float]` / `Dict[str, List[float]]` inputs (verified in `trade_tools.py` and
`validation_tools.py`). An agent must `read_reference` the artifact and paste thousands of numbers back
through the context window. `run_monte_carlo_simulation` (the variant with equity bands) has the same
limitation in reverse: it takes only `tickers+weights`, never a strategy's return series.

**B. Robustness only by re-running.** `robustness.parameter_sensitivity`, `robustness.deflated_sharpe_ratio`
and `robustness.block_bootstrap_ci` are reachable only inside `get_robustness_diagnostics`, which re-runs the
entire grid and the best row. `run_backtest_optimization` returns at most `top_n ≤ 20` rows and does not
persist the grid, so sensitivity/DSR cannot be computed on a grid the agent already paid for, and
`analyze_parameter_decay` can only be fed the truncated top-20 (which biases its spike/plateau reading).

**C. The covariance the library can estimate is not the covariance the optimizer uses.**
`estimate_covariance` (Ledoit-Wolf, EWMA, shrunk EWMA, with condition-number diagnostics) is exposed, and
its module docstring says shrinkage "is the ANSWER to [the optimizer's] warning" — but
`run_portfolio_optimization` (all five methods) computes the sample covariance internally via
`annualized_mean_cov` and offers no `covariance_method`. The construction tools (`optimize_risk_parity`,
`optimize_max_diversification`, `get_marginal_risk_contribution`) do take an inline matrix, so the agent can
chain by hand for those three; the mean-variance and Black-Litterman paths cannot be shrunk at all.

**D. The efficient frontier is solved but never traced.** `_frontier_weights` yields w(r) for any target
return in closed form; `_solve_constrained` does the bounded case. Only single points (four objectives) are
exposed. "Where on the curve should this mandate sit / how much return does each unit of risk buy" cannot
be asked.

**E. Strategies run per ticker, never as one portfolio.** `run_strategy_matrix` gives each (ticker,
strategy) its own capital; `run_signal_panel_backtest` also gives each ticker its own capital (documented).
`run_portfolio_simulation` is the shared-cash engine, but it takes weights, not a strategy name. There is no
route from "registry strategy + universe" to "one cash account with drift, ADV cap and borrow cost", which
is the question a rule's *portfolio* viability turns on.

**F. Walk-forward stitching utilities are orphaned.** `stitch_oos_returns` / `compute_stitched_metrics`
are reached by no tool. Their natural consumer — combining out-of-sample paths that were produced
separately (CPCV paths from `build_purged_cv_splits`, per-symbol walk-forwards, modeling OOS folds) — has no
tool either.

**G. Baseline excluded from the comparison tools.** `buy_and_hold` cannot be passed to
`run_strategy_matrix`, `run_backtest_compact`, `get_backtest_diagnostics` or `compare_cost_models`, so a
matrix "compared against nothing" is exactly what `_buy_and_hold_signals`'s docstring says it was moved into
the registry to prevent.

**H. Capacity is static.** `get_capacity_report` answers "max AUM for THIS weight vector"; `plan_rebalance`
answers "how long to get there". Neither uses a simulation's `rebalance_log` (turnover per rebalance) to
answer "at what AUM does this *strategy* — with its actual turnover — start paying for itself in impact".

**I. Sizing methods unevenly exposed through `convert_reference`.** Only `rank_weighted`, `zscore_normalized`,
`vol_scaled` (refused) — `equal_weight_top_bottom` (the quantile-portfolio sizer the modeling runtime uses by
default) and `dollar_neutral` are not offered when converting a score panel to a weight panel via meta.

**J. Custom-callable grid.** `backtest_grid(strategy=<callable>)` — a threshold sweep over a
user-supplied signal function — is unreachable across the JSON boundary; there is no threshold/deadband
grid over a score reference either.

**K. Screener.** `screen_stocks` is fully exposed; the filter vocabulary is closed (13 keys) and
technical filters are limited to RSI(14), SMA(N) and beta-vs-SPY. Nothing in the screener module is
unexposed; the gap is library scope, not surface.

## 6. Proposed new tools

Ranking reflects (decision value) × (unexposed capability combined) ÷ (effort). Effort: S ≤ ½ day,
M ≈ 1–2 days, L > 2 days, including Pydantic models, dispatch registration, tests and regenerating
`20_tool_index.md` (a test fails otherwise).

### P1. `run_strategy_portfolio_backtest` — backtest runtime, `backtest_execution`
- **Inputs:** `tickers`, `start_date`, `end_date`, `strategy_type` (any `RUNNABLE` name), `parameters`,
  `weighting` ∈ {equal, inverse_vol}, `rebalance` ∈ {on_signal_change, weekly, monthly}, `gross_leverage`,
  plus the `run_portfolio_simulation` cost/limit block (`commission_model`, `use_impact_model`,
  `max_adv_participation`, `borrow_fee_bps`, `fill_price` default `next_open`).
- **Outputs:** the portfolio-simulation result (equity curve ref, rebalance_log ref, max_leverage,
  peak_position_value, return_over_rebalance, warnings) **plus** the per-ticker `run_signal_panel_backtest`
  table for the same signals, side by side, with the gap between "independent accounts" and "one account"
  named explicitly (turnover paid, cash drag while flat, ADV rejections).
- **Backing:** `STRATEGY_REGISTRY`/`RUNNABLE` → per-ticker signals; `sizing` (equal / `vol_scaled`) →
  `target_weights` at rebalance dates; `portfolio_engine.run_portfolio_simulation`;
  `panel.run_signal_panel_backtest` for the comparison row; `fetch_ohlcv_panel_sync`.
- **Decision, not plumbing:** "Does this rule survive as a *book* — one cash balance, positions that drift,
  a cap on how much of the volume I can take — or only as twenty hypothetical accounts?" No existing tool
  answers it; the agent cannot compose it because no tool emits a strategy's signal panel as a weight ref.
- **Warnings the tool must state:** `fill_price='close'` is look-ahead (engine warning propagated);
  every ticker sits flat during its own indicator warm-up, so early equity is cash; equal weighting across
  names in signal is a choice, not the strategy; universe is as-of-today (survivorship) unless the caller
  passes a point-in-time list; the independent-accounts row is not a valid portfolio number and is shown
  only to size the difference; ADV/impact need `Volume` and a lookback that is itself a model.
- **Effort:** M.

### P2. `assess_backtest_result` — backtest runtime, `backtest_validation`
- **Inputs:** `equity_curve_ref` (from `run_backtest_compact`, `run_portfolio_simulation`, modeling
  `evaluate_model_portfolio`), optional `trades_ref`, `n_trials` (+ optional `trial_sharpes`),
  `regimes_ref` or `regime_source` (e.g. `detect_regimes` output), `current_cost_bps`, `seed`.
- **Outputs:** one report: DSR (returns-form, `overfitting.deflated_sharpe_ratio`), block-bootstrap CI on
  Sharpe **and** max drawdown (`robustness.block_bootstrap_ci` with two `metric_fn`s), regime-stratified P&L
  concentration, and — when `trades_ref` is given — trade-reshuffle drawdown distribution, runs-test
  clustering, random-sign comparison and break-even cost. Each block carries its own `warnings`.
- **Backing:** `artifacts.load_artifact` via `handoff.resolve`; `overfitting.{deflated_sharpe_ratio,
  regime_stratified_performance}`; `robustness.block_bootstrap_ci`; `trade_analysis.{monte_carlo_trade_paths,
  analyze_trade_clustering, compare_against_random, break_even_cost}`.
- **Decision, not plumbing:** "I have this run's artifacts — should I believe it?" Today that is seven tool
  calls, each requiring the agent to copy a list of floats out of `read_reference`. The one decision is
  "trust / do not trust / trust at half size", and the pieces disagree in informative ways (a DSR of 0.97
  next to 80% of P&L from one regime) that only a joint read surfaces.
- **Warnings:** DSR deflates only the trials declared; bootstrap assumes stationarity; reshuffle destroys
  inter-trade dependence (optimistic on clustering, and the clustering block says by how much); regime
  labels chosen after seeing the equity curve are themselves a trial; break-even is flat cost, not
  impact; `n_trials=1` is almost never true.
- **Effort:** M (the pieces exist; the work is `DataSource` inputs and one result model).
- **Cheaper alternative:** make every `List[float]` input on the nine `backtesting/`-backed tools a
  `DataSource` (as `run_terminal_monte_carlo` already does). S per tool, and it removes theme A entirely
  even without P2.

### P3. `optimize_with_estimated_covariance` — or, preferably, a parameter on `run_portfolio_optimization`
- **Inputs (parameter form):** `run_portfolio_optimization` gains `covariance_method` ∈
  {`sample` (default, unchanged), `ledoit_wolf`, `ewma`, `ewma_shrunk`} and `halflife`.
- **Outputs:** unchanged, plus `covariance_diagnostics` (condition number, obs/parameter, shrinkage
  intensity) and, when the method is not `sample`, the same solve on the sample matrix reported as
  `sample_covariance_weights` so the agent sees how much the answer moved.
- **Backing:** `covariance.estimate_covariance` → matrix; `mean_variance_optimize` needs a small refactor
  to accept a precomputed (μ, Σ) (it currently derives both from `returns_df`); `risk_parity_weights`,
  `black_litterman` already take a matrix.
- **Decision, not plumbing:** "Are these weights a property of the data or of the noise in the smallest
  eigenvalue?" The library states that shrinkage is the answer to its own conditioning warning, then makes
  the answer unreachable from the tool that raises the warning. Showing sample vs shrunk side by side
  turns a caveat into a measured sensitivity.
- **Warnings:** shrinkage intensity near 1 means the data supported almost no correlation structure;
  EWMA lowers effective sample size (conditioning gets worse, not better); the mean vector is still the
  sample mean and is the noisier input — shrinking Σ does not fix μ; long-only/max_weight paths need scipy.
- **Effort:** M (refactor of `mean_variance_optimize` to accept `(mu, cov)`; keep `returns_df` path).

### P4. `trace_efficient_frontier` — portfolio runtime
- **Inputs:** `tickers`, `start_date`, `end_date` (or `returns_ref`), `n_points` (default 20),
  `allow_short`, `max_weight`, `risk_free_rate`, `covariance_method` (from P3), `periods_per_year`.
- **Outputs:** frontier rows `{target_return, volatility, sharpe, weights, converged}` from the GMV up to the
  max-return corner; the tangency point; marginal return per unit of volatility between adjacent points;
  the conditioning/small-sample warnings; where a supplied `current_weights` vector sits relative to the
  curve (distance in vol at the same return).
- **Backing:** `optimize._frontier_stats` + `_frontier_weights` (unconstrained, closed form, numpy only);
  `_solve_constrained` looped over target returns when bounded; `_conditioning_warnings`,
  `_small_sample_warnings`, `_verify_solution`.
- **Decision, not plumbing:** "Where should this mandate sit?" is a curve question; the four single-point
  objectives answer it only if the agent already knows the answer. The marginal-return column is what a
  risk committee actually reads.
- **Warnings:** the frontier is an in-sample fit — points to the right of the tangency are increasingly
  estimates of μ, the noisiest input; constrained frontiers can have infeasible target returns (reported
  as non-converged rather than dropped); `max_sharpe` is undefined when `rf ≥ B/A` (the library already
  refuses this — the trace should mark the region rather than fail).
- **Effort:** M.

### P5. `estimate_strategy_capacity` — portfolio runtime
- **Inputs:** `target_weights_ref` (rebalance-dated weight panel — the same artifact
  `run_portfolio_simulation` / `construct_weights_from_scores` / `evaluate_model_portfolio` write) or a
  `rebalance_log_ref`, `tickers`, `start_date`, `end_date`, `max_participation`, `adv_lookback`,
  `impact_coefficient`, `return_drag_tolerance` (fraction of gross return impact may consume).
- **Outputs:** per-rebalance binding ticker and max AUM (`capacity_report` evaluated at every rebalance
  date with that date's trailing ADV); the distribution (min / p10 / median) — the *min* is the capacity;
  the AUM at which cumulative sqrt-impact equals `return_drag_tolerance` of the backtested return; days to
  liquidate the largest position at the binding date.
- **Backing:** `constraints.capacity_report`, `constraints.days_to_liquidate`, `constraints.adv_participation`,
  `costs.sqrt_impact_bps`/`impact_cost`, `fetch_ohlcv_panel_sync`, `artifacts.load_artifact`.
- **Decision, not plumbing:** "At what size does this stop being the strategy that was backtested?"
  `get_capacity_report` answers for one weight vector; a strategy's capacity is the *worst* rebalance, and
  the impact-drag AUM is the number that says whether the backtested return exists at fund size.
- **Warnings:** ADV is a trailing mean of Close×Volume, not executable liquidity; the sqrt model's
  coefficient is a floor-ish default (see `rebalance.py`'s own note); capacity assumes every rebalance is
  traded at the cap in one day — `plan_rebalance` spreads it; thin names dominate the min, so one illiquid
  name with a 1% weight sets the whole book's capacity (report which).
- **Effort:** M.

### P6. `stitch_oos_paths` — backtest runtime, `backtest_validation`
- **Inputs:** `window_returns` as a list of `DataSource`s (refs or values) with optional
  `window_params` (list of dicts) and `window_labels`; `initial_capital`; `require_non_overlapping`
  (default true).
- **Outputs:** `compute_stitched_metrics` on the chronologically stitched series; per-window table;
  `longest_losing_streak`; `parameter_turnover`; the overlap check result.
- **Backing:** `walk_forward.{stitch_oos_returns, compute_stitched_metrics, longest_losing_streak,
  parameter_turnover}` — the two orphaned functions plus the two exposed ones.
- **Decision, not plumbing:** "Across the out-of-sample paths I produced separately (CPCV paths from
  `build_purged_cv_splits`, modeling OOS folds, per-window custom-signal backtests), what did the WHOLE
  out-of-sample record earn compounded, and how many windows in a row lost?" The walk-forward tool does
  this internally for registry strategies only.
- **Warnings:** concatenation carries no cost at window boundaries (the walk-forward tool re-runs the engine
  across the boundary precisely because this does not); windows must not overlap — paths from different
  symbols over the same dates are a *panel*, not a sequence, and are refused unless the flag is cleared;
  averaging per-window Sharpes is exactly what this exists to avoid, so per-window rows are shown but the
  headline is the compounded one.
- **Effort:** S.

### P7. `compare_portfolio_cost_models` — backtest runtime, `backtest_validation`
- **Inputs:** everything `run_portfolio_simulation` takes, plus `scenarios`: a list of named cost blocks
  (`commission_model`, `per_share_rate`, `use_impact_model`, `impact_coefficient`, `borrow_fee_bps`,
  `margin_interest_rate`, `max_adv_participation`) and `solve_breakeven`.
- **Outputs:** one row per scenario (final equity, Sharpe, max drawdown, total cost paid, rebalances
  rejected by ADV), the gross (zero-cost) row, and the flat per-rebalance cost at which return reaches
  zero (`break_even_cost` over `return_over_rebalance`-style per-rebalance returns).
- **Backing:** `run_portfolio_simulation` (N runs), `trade_analysis.break_even_cost`.
- **Decision, not plumbing:** `compare_cost_models` exists for a single name and pct/bps costs; the
  multi-asset, per-share + impact + borrow + ADV cost model is the one that kills long/short and
  small-cap strategies, and it is exposed only one scenario at a time.
- **Warnings:** impact coefficient and lookback are models; ADV rejections mean the scenario could not be
  traded as specified (that is the finding, not an error); borrow fees are a flat bps, real borrow is
  name-specific; same look-ahead caveats as the engine.
- **Effort:** M.

### P8. `run_threshold_grid_on_scores` — backtest runtime, `custom_signal` (lower priority)
- **Inputs:** `scores_ref` (single-name score series or one column of a panel), `symbol`, dates,
  `thresholds`/`deadbands` list, engine cost block.
- **Outputs:** `backtest_grid`-shaped table over thresholds, **always** accompanied by
  `parameter_sensitivity`, `parameter_decay` and grid-form DSR (so the sweep cannot be read without its
  multiple-testing bill), and a `pbo` row if enough bars.
- **Backing:** `backtest_grid(strategy=<threshold callable built from the score>)` — the unreachable
  custom-callable path; `robustness.{parameter_sensitivity, deflated_sharpe_ratio}`;
  `overfitting.{parameter_decay, probability_of_backtest_overfitting}`.
- **Decision, not plumbing:** "Which threshold?" — but the repo's own stance (see `select_features`) is
  that a selector scored on the sample it selects from manufactures overfit. The tool earns its place only
  because it refuses to return the sweep without the deflation. If that coupling is not wanted, do not
  build it.
- **Warnings:** every threshold is a trial; the score's own construction (modeling OOS vs in-sample) decides
  whether the whole grid is leakage; `fill_price='next_open'` mandatory when the score is same-bar.
- **Effort:** M.

## 7. Parameter additions to existing tools (cheaper than new tools)

| Tool | Add | Backed by | Why |
|---|---|---|---|
| `get_deflated_sharpe_ratio`, `run_reality_check`, `get_regime_stratified_performance`, `estimate_backtest_overfitting`, `run_monte_carlo_trade_paths`, `analyze_trade_clustering`, `compare_against_random`, `estimate_break_even_cost`, `get_exposure_attribution` | Accept `DataSource` (symbol / `sqt://` ref / values) instead of `List[float]` | `handoff.resolve` + `artifacts.load_artifact` (already used by `run_terminal_monte_carlo`) | Closes theme A without any new tool; `trades_uri` from `run_backtest_compact` becomes consumable. Effort S each. |
| `run_monte_carlo_simulation` | `returns: DataSource` as an alternative to `tickers+weights` | `monte_carlo.simulate_forward_paths` | The banded MC on a strategy's own equity path; today only the terminal variant can do it. S. |
| `run_backtest_optimization` | `include_robustness: bool` → `parameter_sensitivity`, grid-form DSR (with the annualisation fix `get_robustness_diagnostics` applies) and 1-D `parameter_decay` per axis, computed on the **full** grid before `top_n` truncation; and/or `grid_ref` persisting the full grid | `robustness.{parameter_sensitivity, deflated_sharpe_ratio}`, `overfitting.parameter_decay`, `save_artifact` | Theme B: no second grid run. S–M. |
| `run_portfolio_optimization` | `covariance_method`, `halflife` | `covariance.estimate_covariance` | Theme C (P3 in parameter form). M. |
| `run_strategy_matrix`, `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models` | Gate on `RUNNABLE` instead of `STRATEGY_REGISTRY` | `strategies.RUNNABLE` | Theme G: the baseline row. S. Searches (`run_backtest_optimization`, walk-forward) should stay on `STRATEGY_REGISTRY`, as the module comment argues. |
| `compare_strategies` | `strategies: List[str]` + `parameters: Dict[str, Dict]` instead of four fixed `*_parameters` fields | `STRATEGY_REGISTRY` | Half the registry (donchian, momentum, vwap, adx) is absent from the "which rule?" tool. S. |
| `convert_reference` (score_panel → weight_panel) | `construction_method` ∈ + `equal_weight_top_bottom` (`n_long`, `n_short`), `dollar_neutral: bool` | `sizing.{equal_weight_top_bottom, dollar_neutral}` | Theme I. S. |
| `optimize_hierarchical_risk_parity` | `tickers`/dates or `returns_ref` alternative to inline `returns` | `fetch_returns_sync`, `handoff.resolve` | The only construction tool that needs a full returns matrix inline. S. |
| `get_liquidity_adjusted_var` | `impact_coefficient` | `construction.liquidity_adjusted_var` already takes it | Exposed function has a parameter the tool hides. S. |
| `get_capacity_report` | `as_of_dates` / `rebalance_dates` (evaluate at several dates) | `capacity_report` in a loop | Half of P5 for the price of a loop. S. |
| `run_pair_trade_backtest` | `signal_only: bool` → return the `_spread_state` series and current state without simulating | `pairs._spread_state`, `analysis.cointegration.spread_zscore` | "Is the pair in a trade right now?" without a full backtest. S. |

## 8. Dead code, duplication, docstring-vs-code

1. **Duplicate signal-calendar alignment.** `panel._align_signal_to_calendar` and
   `backtest/tools.py::_apply_signal_fill_policy` implement the same hold/flat/error reindex. The library
   version's docstring says "the agent wrapper already did this; the library function … did not" — now both
   do, and `run_signal_panel_backtest`'s tool path applies the tool version then the library version. Same
   semantics today; two places to drift.
2. **Two deflated-Sharpe implementations, both exposed.** `backtest.robustness.deflated_sharpe_ratio`
   (grid form: takes observed per-period Sharpe + trial std) feeds `get_robustness_diagnostics`;
   `backtesting.overfitting.deflated_sharpe_ratio` (returns form: computes skew/kurtosis, takes
   `trial_sharpes`) feeds `get_deflated_sharpe_ratio`. They now share `expected_max_sharpe` and
   `sharpe_standard_error_factor` (pinned by `tests/test_single_definition.py`), so the maths is one
   definition, but the annualisation conventions differ: the grid form needs de-annualised inputs (the tool
   divides by √252 and re-multiplies `expected_max_sharpe` by hand), the returns form takes
   `periods_per_year`. A caller reading both results gets `expected_max_sharpe` (annualised by the tool) and
   `expected_max_sharpe_from_luck` (annualised by the library) — same quantity, two names.
3. **Two block-bootstrap CIs.** `robustness.block_bootstrap_ci` (backtest runtime, Sharpe only, inside
   `get_robustness_diagnostics`) and `analysis.inference` behind research `get_bootstrap_interval`
   (`statistic` selectable). Both use `_resampling.block_indices`, so the resampling is one definition;
   the CI wrappers are two.
4. **Two risk-parity entry points, one solver.** `optimize.risk_parity_weights` now delegates to
   `construction.risk_parity` (the docstring records the 8/300 non-convergence of the old fixed point).
   Both are exposed (`run_portfolio_optimization(method="risk_parity")` and `optimize_risk_parity`) with
   different budget semantics (refuse vs renormalise) — documented, deliberate, but still two tools for one
   question.
5. **`_normal_quantile` in `construction.py`** is a 200-step bisection on `erf` while `_special.norm_ppf`
   exists and is what `robustness.py`/`overfitting.py` use ("this had 2 copies across the library" — this
   is a third, private one).
6. **`MIN_OBS_PER_PARAMETER`** (`construction.py`) is defined, exported in `__all__`, and never read.
7. **`walk_forward.stitch_oos_returns` / `compute_stitched_metrics`** — the module docstring explicitly
   says "not dead code, just not what the two OOS aggregate fields are computed from anymore". They have
   tests and no caller (P6 is the caller they lack).
8. **`_buy_and_hold_signals` docstring vs `BASELINE_REGISTRY` comment.** The function's docstring says it
   lives in the registry "so that the grid, the strategy matrix, walk-forward and the optimiser can all use
   the same baseline". The registry comment forty lines later says baselines are deliberately *excluded*
   from `STRATEGY_REGISTRY` so a search cannot choose to hold — and `run_strategy_matrix`, which is not a
   search, gates on `STRATEGY_REGISTRY` and rejects it. The comment is right; the docstring promises more
   than the code delivers.
9. **`strategy_params.resolve_strategy_params` docstring: "relations are enforced where a single
   configuration is deliberately requested (the agent tools)".** Grep shows `check_relations=True` is
   called only from meta `validate_tool_call`. `run_sma_backtest(fast_period=50, slow_period=10)` runs
   through `RUNNABLE` (relations off) and returns a result. The per-value checks (the leakage-relevant
   ones) do hold everywhere; the cross-parameter promise holds only if the agent pre-flights.
10. **`engine._run_signal_fn_job` drops `risk_free_rate`** (see §2). Only reachable when the C++ extension
    is absent and `strategy` is a callable — i.e. not from any tool today — but it is the exact
    grid-vs-single-run inconsistency the neighbouring `_run_grid_job` comment claims to have fixed.
11. **`combinatorial_purged_cv` has a redundant embargo check**: lines 479–483 test only
    `test_index.max()`, immediately followed by lines 484–488 testing every `_block_ends(test_index)`,
    which includes the max. The first branch can never purge something the second would not.
12. **Unit inconsistency, documented but live:** `stress_test.replay_stress_scenario` returns
    `max_drawdown_pct` as a fraction; `futures_engine.run_futures_simulation` returns `max_drawdown_pct`
    ×100. Both agent-reachable; both docstrings now say so; the names still collide.
13. **`portfolio_engine` inlines cost arithmetic** from `percentage_commission`, `per_share_commission`,
    `impact_cost`/`sqrt_impact_bps` for speed, with comments that each "must be mirrored" on change. Three
    formulas with two homes each — deliberate, measured, and a maintenance hazard the comments own.
14. **`STRATEGY_REGISTRY` is long/flat only.** `run_strategy`'s docstring documents `-1` and the
    `total-loss guard` exists for shorts, but no registry strategy can produce one; `compare_strategies`,
    `run_strategy_matrix` etc. therefore never exercise the short path.

## 9. Evidence trail (grep summary)

- Slice imports in the tool surface: `grep -rnE 'from standard_quant_tools\.(backtest|backtesting|portfolio|screener)' src/standard_quant_tools/{agent,modeling}` — 30 import sites across 17 files (listed in §1).
- Unreached names confirmed by `grep -rn --include=*.py '\b<name>\b' src tests`: `stitch_oos_returns`,
  `compute_stitched_metrics` (walk_forward.py + `tests/backtest/test_backtest_walk_forward.py` only);
  `VectorizedStrategy` (tests only); `MIN_OBS_PER_PARAMETER` (definition + `__all__` only).
- `resolve_strategy_params` with relations: only `agent/runtimes/meta/tools.py:1107`.
- `expected_max_sharpe` / `sharpe_standard_error_factor`: `robustness.py`, `overfitting.py:56-57` (import),
  `tests/test_single_definition.py`.
- `List[float]` inputs on judgement tools: `trade_tools.py:42,55,67,75,289`; `validation_tools.py:53,82,129,130,150`.
- `run_backtest_optimization` truncation: `backtest/tools.py` `top_n = min(input_data.top_n, 20, n_combinations)`; no `save_artifact`/`publish` in that tool.
- `convert_reference` sizers: `meta/convert.py:147-150` imports `rank_weighted, vol_scaled, zscore_normalized` only.
