# LLM tool surface cross-reference — `standard_quant_tools`

Source of truth: every tool function body under `src/standard_quant_tools/agent/runtimes/<runtime>/` (including the sub-modules each runtime's `__init__.py` concatenates into `TOOL_DEFS`), `src/standard_quant_tools/modeling/agent/tools.py` (`MODELING_TOOL_DISPATCH`), and `src/standard_quant_tools/modeling/agent/feature_tools.py` (`FEATURE_TOOL_DISPATCH`). Descriptions in `Documentation/20_tool_index.md` were used only for the tool list, never for the call attribution.

**Counts (from the dispatch tables):** research 42 (29 in `tools.py`/`reference_tools.py` + 7 `diagnostic_tools.py` + 6 `inference_tools.py`), backtest 35 (21 `tools.py` + 1 `terminal_mc_tools.py` + 6 `validation_tools.py` + 5 `trade_tools.py` + 2 `futures_tools.py`), modeling 22, meta 20 (17 `tools.py` + 3 `scope_tools.py`), data 18, portfolio 18 (10 `tools.py`/`weight_tools.py` + 8 `construction_tools.py`), delta_one 18, microstructure 17 (4 tick tools defined in `portfolio/tools.py` + 3 `series_tools.py` + 8 `estimator_tools.py` + 1 `book_tools.py` + 1 `event_tools.py`), derivatives 12, feature_lab 9. **Total 211.**

## Legend

| Column | Meaning |
|---|---|
| **Calls** | The library functions that do the numerical work. Pydantic input/result models, `handoff.publish/resolve` plumbing, `_rounded`/`_json_safe` helpers and logging are omitted. `provider.*` means `DataFactory.get_provider().<method>` (yfinance / polygon / bloomberg / databento). |
| **Reads** | `fetch` = provider market-data call inside the tool; `panel-fetch` = `portfolio.portfolio.fetch_returns_sync` / `fetch_ohlcv_panel_sync` (concurrent multi-ticker provider fetch); `ref:<kind>` = an `sqt://<kind>/...` handoff reference; `uri` = a Parquet path from `backtest.artifacts` / `modeling.artifacts`; `dataset_id` / `model_id` = a runs-directory entry; `inline` = arrays/dicts in the call; `audit` = the audit JSONL log; `catalog` = the live tool registry; `file` = a caller-supplied path on disk. |
| **Writes** | `none`; `ref:<kind>` = publishes a handoff reference; `uri` = saves a Parquet artifact; `registry` = modeling runs dir (dataset/model/promotion log); `file` = a file the caller named. |

---

## `research` (42 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `analyze_stock_risk` | Alpha/beta/Sharpe/Sortino/MDD/VaR/CVaR/IR of one symbol vs a benchmark over a period string | `analysis.regression.calculate_beta`; `metrics.risk_metrics.sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_historical`, `cvar`, `information_ratio` | fetch (`get_ohlcv` symbol + benchmark, period parsed by `_parse_period`) | none |
| `get_technical_analysis` | Last-bar indicator values + boolean signals for one symbol | `indicators.trend.sma`, `ema`, `macd`, `adx`, `williams_r`; `indicators.momentum.rsi`, `stochastic_oscillator`; `indicators.volatility.atr`, `bollinger_bands`; `indicators.volume.obv`, `vwap`; fused `_cpp_core.technical_indicators` when ≥2 of rsi/adx/bollinger/stochastic and C++ present | fetch | none |
| `get_portfolio_analysis` | Metrics of a fixed-weight basket vs a benchmark | `portfolio.portfolio.portfolio_metrics`, `portfolio.portfolio.correlation_matrix` | panel-fetch (returns) + fetch (benchmark) | none |
| `run_screener` | Filter a universe by fundamental/technical criteria | `screener.screener.screen_stocks` (internally `calculate_beta`, `rsi`, `sma`, provider ratios) | fetch (inside `screen_stocks`) | none |
| `run_factor_regression` | OLS multi-factor regression of one symbol on factor tickers | `analysis.multi_factor.multi_factor_regression`, `rolling_factor_loadings` | fetch (symbol + each factor) | none |
| `run_cointegration_test` | Engle-Granger on a pair: hedge ratio, half-life, z-score signal | `analysis.cointegration.cointegration_test`, `compute_spread`, `spread_zscore` | fetch ×2 | none |
| `run_kalman_hedge_ratio` | Time-varying hedge ratio for a pair | `analysis.cointegration.kalman_hedge_ratio`, `spread_zscore` | fetch ×2 | none |
| `run_pca_analysis` | PCA of a universe's returns: explained variance, loadings, contributions | `analysis.pca.pca_returns`, `factor_contributions` | fetch (per ticker) | none |
| `get_correlation_analysis` | Correlation matrix, avg pairwise, extreme pairs, diversification ratio | `analysis.correlation.pairwise_correlation_summary`, `diversification_ratio` | panel-fetch (returns) | none |
| `run_hurst_analysis` | Hurst exponent (DFA/RS) + optional rolling regime fractions | `analysis.hurst.hurst_exponent`, `rolling_hurst` | fetch | none |
| `get_rally_signal` | Five-signal rally detector for one symbol | `analysis.rally.detect_rally` (internally `hurst_exponent`, `adx`) | fetch | none |
| `get_volatility_estimators` | Parkinson / Garman-Klass / Yang-Zhang vs close-to-close vol | `metrics.volatility_estimators.parkinson_volatility`, `garman_klass_volatility`, `yang_zhang_volatility`; `metrics.return_metrics.annualized_volatility` | fetch | none |
| `run_garch_volatility_forecast` | GARCH(1,1) fit and forward vol forecast | `analysis.garch.garch_volatility_forecast` | fetch | none |
| `scan_pairs` | All-pairs cointegration scan ranked by half-life | `analysis.cointegration.scan_cointegrated_pairs` (batch, identical indexes) else `cointegration_test` per pair; `compute_spread`, `spread_zscore` | fetch (per ticker, failures isolated) | none |
| `get_stock_fundamentals` | Company metadata + key ratios | none (pure provider: `provider.get_ticker_info`, `provider.get_financial_ratios`) | fetch | none |
| `get_advanced_indicators` | Parabolic SAR, Wilder ATR, MFI at last bar | `indicators.trend.parabolic_sar`; `indicators.volatility.wilder_atr`; `indicators.volume.mfi` | fetch | none |
| `get_rolling_beta` | Rolling OLS beta and its drift | `analysis.regression.rolling_beta` | fetch ×2 | none |
| `get_extended_risk_metrics` | Calmar, Treynor, parametric VaR 95/99, hist VaR 99, CVaR 99, CAGR | `metrics.return_metrics.cagr`; `metrics.risk_metrics.calmar_ratio`, `treynor_ratio`, `var_parametric`, `var_historical`, `cvar`; `analysis.regression.calculate_beta` | fetch ×2 | none |
| `get_tail_risk_metrics` | EVT (POT/GPD) VaR & CVaR vs empirical quantile | `metrics.risk_metrics.evt_tail_risk`, `var_historical` | fetch | none |
| `get_data_quality_report` | Provider guarantees + missing bars / stale runs / price jumps | `data.quality.detect_missing_bars`, `detect_stale_prices`, `detect_price_jumps`; `provider.get_metadata` | fetch + metadata | none |
| `get_technical_panel` | RSI/ADX/ATR/Bollinger/Stochastic for a universe at the latest bar | `indicators.panel.technical_indicators_panel` | panel-fetch (OHLCV) | uri (optional, `backtest.artifacts.save_artifact` per indicator when `persist_run_id`) |
| `detect_change_points` | Binary-segmentation mean breaks in price or returns | `analysis.structure.detect_change_points` | fetch (via `_price_series`) | none |
| `get_partial_correlation` | Correlation of x,y after removing controls | `analysis.structure.partial_correlation` | fetch (x, y, each control) | none |
| `test_granger_causality` | Lag-screened Granger test, cause → effect | `analysis.structure.granger_causality` | fetch ×2 | none |
| `analyze_tail_dependence` | Conditional tail co-movement of a pair | `analysis.structure.tail_dependence` | fetch ×2 | none |
| `run_stationarity_tests` | ADF + KPSS + variance ratio with verdict | `analysis.stationarity.run_stationarity_tests` | fetch | none |
| `detect_regimes` | Gaussian-mixture volatility regimes | `analysis.stationarity.detect_regimes` | fetch | none |
| `calculate_series_metrics` | Chosen metrics on ANY return series | closed table over `metrics.cumulative_return`, `cagr`, `annualized_volatility`, `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `var_historical`, `var_parametric`, `cvar`, `max_drawdown`; `data.models.resolve_source` | `DataSource`: fetch (symbol, full history) **or** ref (any tabular kind) **or** inline values | none |
| `compute_indicator_panel` | Indicator HISTORY for a universe, one ref per indicator | `indicators.panel.technical_indicators_panel` | ref:price_panel (stacked) **or** panel-fetch (OHLCV) | ref:indicator_panel (one per indicator) |
| `test_autocorrelation` | Joint Ljung-Box (returns or squared) | `analysis.diagnostics.ljung_box` | inline | none |
| `run_seasonality_analysis` | Weekday/month/day-of-month effects, Bonferroni + joint F | `analysis.diagnostics.seasonality` | inline (+ dates required) | none |
| `get_entropy_measures` | Shannon + permutation entropy | `analysis.diagnostics.entropy_measures` | inline | none |
| `get_sharpe_stability` | Rolling Sharpe, half-sample decay test | `analysis.diagnostics.rolling_sharpe_stability` | inline | none |
| `get_drawdown_profile` | Every drawdown episode above a threshold | `analysis.diagnostics.drawdown_profile` | inline (+ optional dates) | none |
| `get_lead_lag_matrix` | Cross-lag correlations across a universe, Bonferroni | `analysis.diagnostics.lead_lag_matrix` | inline map | none |
| `test_structural_break` | Chow test at a known index, mean or relationship | `analysis.diagnostics.structural_break_test` | inline | none |
| `get_bootstrap_interval` | Block-bootstrap CI for a named statistic | `analysis.inference.bootstrap_statistic` | inline | none |
| `compare_distributions` | KS + moment shifts + tail ratio between two samples | `analysis.inference.compare_distributions` | inline | none |
| `get_correlation_stability` | Rolling correlation, sign flips, stress correlation of a pair | `analysis.inference.rolling_correlation_stability` | inline | none |
| `decompose_returns` | Arithmetic vs geometric, vol drag, best/worst-5 contribution | `analysis.inference.decompose_returns` | inline | none |
| `test_normality` | Jarque-Bera + tail counts beyond 3σ/4σ | `analysis.inference.test_normality` | inline | none |
| `estimate_tail_index` | Hill tail index across thresholds | `analysis.inference.estimate_tail_index` | inline | none |

---

## `backtest` (35 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `run_sma_backtest` | One strategy run (dispatches on `strategy_type`, default SMA) | `backtest.strategies.RUNNABLE[strategy_type]` → signals; `_shared._run_backtest` → `backtest.engine.run_strategy` | fetch | none |
| `run_rsi_backtest` | Same as above, default RSI mean-reversion | identical to `run_sma_backtest` (`_dispatch_backtest`) | fetch | none |
| `run_macd_backtest` | Same, default MACD | identical (`_dispatch_backtest`) | fetch | none |
| `run_bollinger_backtest` | Same, default Bollinger | identical (`_dispatch_backtest`) | fetch | none |
| `run_buy_and_hold` | Constant-long baseline | `_run_backtest` → `backtest.engine.run_strategy` on `pd.Series(1.0)` | fetch | none |
| `compare_strategies` | The four registry strategies + B&H on one symbol, ranked | `backtest.strategies.STRATEGY_REGISTRY[...]` ×4; `_run_backtest` → `run_strategy` ×5 | fetch (once) | none |
| `run_regime_adaptive_backtest` | Hurst regime → strategy map → grid search → run | `analysis.hurst.hurst_exponent`; `backtest.engine.backtest_grid`; `STRATEGY_REGISTRY`; `_run_backtest` → `run_strategy` | fetch | none |
| `run_regime_adaptive_walkforward_backtest` | Per-window Hurst + grid over ALL registry strategies, OOS stitched | `hurst_exponent` (per window); `backtest_grid` (per strategy per window); `STRATEGY_REGISTRY`; `backtest.engine.run_strategy` (per window + stitched); `backtest.walk_forward.longest_losing_streak` | fetch | none |
| `run_walk_forward_backtest` | Walk-forward grid optimisation + stitched OOS run | `backtest_grid` per window; `STRATEGY_REGISTRY`; `run_strategy` (per window + stitched); `walk_forward.longest_losing_streak`, `parameter_turnover` | fetch | none |
| `run_backtest_optimization` | Full grid search, top-N | `backtest.engine.backtest_grid` | fetch | none |
| `run_custom_signal_backtest` | Caller-supplied signal on one symbol | local `_apply_signal_fill_policy`; `_run_backtest` → `run_strategy` | fetch + inline signals dict | none |
| `run_signal_panel_backtest` | Caller-supplied signal panel across a universe | `backtest.panel.run_signal_panel_backtest` (per-ticker `run_strategy` + portfolio blend) | panel-fetch (OHLCV) + inline panel **or** ref:signal_panel; optional benchmark fetch | none |
| `run_portfolio_simulation` | Shared-cash portfolio sim from target weights or scores | score mode: `backtest.sizing.rank_weighted` / `equal_weight_top_bottom` / `zscore_normalized` / `vol_scaled`, `dollar_neutral`; `backtest.portfolio_engine.run_portfolio_simulation`; `metrics.risk_metrics.max_drawdown`, `sharpe_ratio`, `sortino_ratio`, `var_historical`, `cvar`, `information_ratio`; `metrics.return_metrics.annualized_volatility` | panel-fetch (OHLCV) + inline weights **or** ref:weight_panel / ref:score_panel; optional benchmark fetch | none |
| `run_pair_trade_backtest` | Two-leg pair trade in one cash account | `backtest.pairs.run_pair_backtest` (→ `portfolio_engine.run_portfolio_simulation`); `max_drawdown`, `sharpe_ratio`, `sortino_ratio`, `annualized_volatility` | fetch ×2 | none |
| `get_robustness_diagnostics` | Parameter sensitivity + DSR + bootstrap CI on best grid trial | `backtest_grid`; `backtest.robustness.parameter_sensitivity`, `deflated_sharpe_ratio`, `block_bootstrap_ci`; `STRATEGY_REGISTRY`; `run_strategy`; `metrics.risk_metrics.sharpe_ratio` (as the bootstrapped statistic) | fetch | none |
| `run_monte_carlo_simulation` | Block-bootstrap forward equity paths for a basket | `portfolio.portfolio.build_portfolio`; `backtest.monte_carlo.simulate_forward_paths` | panel-fetch (returns) | none |
| `run_backtest_compact` | Backtest with curve/trades persisted, summary returned | `STRATEGY_REGISTRY`; `run_strategy`; `metrics.diagnostics.exposure_stats`; `metrics.return_metrics.cagr`; `var_historical`, `cvar` | fetch | uri (`backtest.artifacts.save_artifact` equity_curve, trades) + ref:equity_curve, ref:trade_log |
| `get_backtest_diagnostics` | Top drawdowns, expectancy, MAE/MFE, exposure for a registry strategy | `STRATEGY_REGISTRY`; `run_strategy`; `metrics.diagnostics.top_n_drawdowns`, `trade_expectancy`, `trade_excursions`, `exposure_stats` | fetch | none |
| `get_drawdown_table` | All drawdown episodes of a persisted curve | `backtest.artifacts.load_artifact`; `metrics.diagnostics.drawdown_periods`; `metrics.risk_metrics.drawdown_series`, `max_drawdown` | uri (equity_curve_uri) | none |
| `compare_cost_models` | One signal priced under N cost scenarios + breakeven commission by bisection | `STRATEGY_REGISTRY`; `run_strategy` (1 gross + N scenarios + ≤60 bisection runs); `cagr` | fetch | none |
| `run_strategy_matrix` | Every strategy × every ticker, ranked | `STRATEGY_REGISTRY`; `run_strategy` per cell | fetch (per ticker) | none |
| `run_terminal_monte_carlo` | Terminal-only block bootstrap | `backtest.monte_carlo.simulate_forward_paths_terminal`; `data.models.resolve_source` | `DataSource`: fetch **or** ref **or** inline | none |
| `get_deflated_sharpe_ratio` | DSR given number/variance of trials | `backtesting.overfitting.deflated_sharpe_ratio` | inline | none |
| `estimate_backtest_overfitting` | PBO across combinatorial splits | `backtesting.overfitting.probability_of_backtest_overfitting` | inline map | none |
| `build_purged_cv_splits` | Purged/embargoed combinatorial CV index ranges | `backtesting.overfitting.combinatorial_purged_cv` | inline (n_observations) | none |
| `run_reality_check` | White's reality check vs alternatives | `backtesting.overfitting.reality_check` | inline | none |
| `get_regime_stratified_performance` | Performance by caller-supplied regime label | `backtesting.overfitting.regime_stratified_performance` | inline | none |
| `analyze_parameter_decay` | Smooth vs spiky 1-D parameter surface | `backtesting.overfitting.parameter_decay` | inline | none |
| `run_monte_carlo_trade_paths` | Reshuffle trade returns → drawdown distribution | `backtesting.trade_analysis.monte_carlo_trade_paths` | inline | none |
| `analyze_trade_clustering` | Runs test on win/loss sequence | `backtesting.trade_analysis.analyze_trade_clustering` | inline | none |
| `compare_against_random` | Sign-randomised null for per-trade Sharpe | `backtesting.trade_analysis.compare_against_random` | inline | none |
| `get_exposure_attribution` | Passive vs timing decomposition | `backtesting.trade_analysis.exposure_attribution` | inline | none |
| `estimate_break_even_cost` | Per-trade cost at which edge vanishes, headroom | `backtesting.trade_analysis.break_even_cost` | inline | none |
| `run_futures_backtest` | Margin/variation-margin futures account sim | `backtest.futures_engine.run_futures_simulation` | inline (price + target-contract dicts) | none |
| `run_futures_hedge_backtest` | Cash book + futures hedge, separate P&L streams | `backtest.futures_hedge_backtest.run_futures_hedge_backtest` | inline | none |

---

## `modeling` (22 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `list_features` | Feature catalogue | `modeling.features.registry.list_features` | registry (in-memory) | none |
| `build_model_dataset` | Fetch universe, compute features/target, persist panel | `modeling.dataset.builder.build_dataset` (fetches OHLCV internally) | fetch (inside builder) | registry (`modeling.artifacts.save_artifact` panel.parquet; `save_json` dataset_spec, dataset_meta) → `dataset_id` |
| `run_model_experiment` | Fit + walk-forward validate + register | `_load_dataset_panel` (`artifacts.load_artifact` + `audit.hashing.hash_dataframe`); `dataset.builder.dataset_spec_hash`; `modeling.engine.run_experiment` (fits, `save_model`); `backtest.artifacts.load_artifact` | dataset_id | registry (model dir: manifest, estimator, oos predictions, monitoring refs) + ref:predictions |
| `score_model` | Predictions for a universe as of a date | `modeling.scoring.score_model` (→ `dataset.builder.build_dataset(include_target=False)`, `registry.load_model`, `estimator.predict`, preprocessing) | model_id + fetch (inside scoring) | uri (predictions_*.parquet, features_*.parquet) |
| `inspect_model` | Summary / importance / validation / lineage views | `registry.model_registry.load_manifest`; `registry.lifecycle.current_stage`, `promotions`; `registry.package.verify_model_package` | model_id | none |
| `evaluate_model_portfolio` | OOS predictions → weights → shared-cash sim | `modeling.portfolio_eval.evaluate_model_portfolio` (→ `backtest.sizing.*`, `backtest.portfolio_engine.run_portfolio_simulation`, `dataset.fetch.fetch_universe_ohlcv`) | model_id + fetch (inside eval) | uri (weights, equity curve via `artifacts.save_artifact`) |
| `list_models` | Registered models newest first | `artifacts._runs_dir` glob; `load_manifest`; `lifecycle.current_stage` | registry | none |
| `promote_model` | Lifecycle stage change with reason | `registry.lifecycle.promote`, `promotions` | model_id | registry (append-only promotion log) |
| `monitor_model` | PSI/KS feature drift, prediction drift, realized IC | `load_manifest`; `registry.model_registry.load_monitoring_reference`; `modeling.monitoring.drift_report`, `prediction_drift`, `realized_ic`; `artifacts.load_artifact` | model_id + uri (predictions, features, outcomes) | none |
| `list_datasets` | Built datasets newest first | `artifacts._runs_dir` glob; `artifacts.load_json` | registry | none |
| `compare_models` | Rank within task; optional paired Holm-adjusted test | `load_manifest`; paired: `ensemble.load_oos_predictions`, `_load_dataset_panel`, `validation.comparison.paired_comparison`, `holm_adjust` | model_id(s) + dataset panels | none |
| `check_leakage` | PIT safety of a feature set | `dataset.leakage.check_point_in_time_safety`; `features.registry.get_feature`; optional `_load_dataset_panel` | registry (+ dataset_id) | none |
| `validate_model_spec` | Estimator/params/search/budget check, fit count | `estimators.registry.validate_params`, `validate_param_value`, `allowed_params`, `quantile_support`, `quantile_estimators`; `validation.search.optuna_available`, `n_search_candidates`; `plan.plan_experiment` / `plan.fit_count`; `validation.walk_forward.build_splitter`; `_load_dataset_meta` | inline spec (+ dataset_meta.json) | none |
| `score_predictions` | Metrics, cross-sectional IC, baseline, ESS for any predictions ref | `validation.metrics.regression_metrics`, `baseline_regression_metrics`, `classification_metrics`, `cross_sectional_ic`, `summarize_cross_sectional_ic`, `effective_sample_size`; `validation.ranking.ranking_metrics` | ref:predictions | none |
| `analyze_features` | Whole-panel feature report | `modeling.analysis.build_feature_report` | dataset_id | none |
| `validate_pit_records` | Timestamp sanity of PIT records | `dataset.point_in_time.validate_pit_frame` | inline records | none |
| `join_point_in_time` | As-of join of PIT records onto a dataset | `dataset.point_in_time.validate_pit_frame`, `asof_join`, `coverage_report` | dataset_id + inline records | uri (`artifacts.save_artifact` pit_joined) |
| `build_model_ensemble` | Combine OOS predictions of several models | `modeling.ensemble.combine_predictions` (→ `load_oos_predictions`, `rank_within_date`) | model_ids | ref:predictions |
| `register_external_panel` | Register an externally built feature matrix by reference | `dataset.external_panel.load_external_panel`; `audit.hashing.hash_dataframe`; `dataset.builder.dataset_spec_hash` | file | registry (dataset_spec.json, dataset_meta.json; no panel copy) |
| `analyze_model_errors` | Residuals, calibration, attribution by entity/period/decile | `load_manifest`; `ensemble.load_oos_predictions`; `_load_dataset_panel`; `modeling.diagnostics.residual_summary`, `calibration`, `error_attribution`, `worst_buckets`, `residual_autocorrelation`, `heteroskedasticity` | model_id + dataset panel | none |
| `explain_dataset_row_loss` | n_missing vs n_sole_missing per column | `dataset.alignment.attribute_drops` | dataset_id | none |
| `list_modeling_capabilities` | Tasks, estimators, features, validation schemes, optional deps | `modeling.capabilities.modeling_capabilities` | registries | none |

---

## `feature_lab` (9 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `profile_feature` | Distribution + predictive stats of ONE feature (optional IC-decay curve) | `analysis.feature_report.feature_distribution_stats`, `feature_predictive_stats`; optional `lead_lag_ic_curve` | dataset_id | none |
| `get_feature_redundancy` | Correlation clusters with a representative each, VIF, condition number | `analysis.feature_report.redundancy_report`, `feature_predictive_stats` | dataset_id | none |
| `get_feature_ic_decay` | IC vs time shift of the feature | `analysis.feature_report.lead_lag_ic_curve` | dataset_id | none |
| `select_features` | Drop redundant, drop below IC floor | `analysis.feature_selection.select_features` | dataset_id | none |
| `compare_feature_sets` | Two feature sets on one panel | `analysis.feature_selection.compare_feature_sets` | dataset_id | none |
| `get_feature_drift` | PSI/KS and IC either side of a date | `analysis.feature_stability.feature_drift` | dataset_id | none |
| `get_feature_regime_stability` | IC in contiguous time blocks | `analysis.feature_stability.feature_stability` | dataset_id | none |
| `run_feature_permutation_test` | Two-sided within-date permutation p-value for IC | `analysis.feature_stability.permutation_test_ic` | dataset_id | none |
| `run_feature_ablation` | Leave-one-out refit per feature | `modeling.engine.build_splitter`, `modeling.engine.run_experiment` (register=False, `cache.FoldCache`); `analysis.feature_ablation.estimate_ablation_fits`, `ablation_contributions`, `summarize_ablation` | dataset_id + inline ModelSpec | none |

---

## `meta` (20 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `explain_decision` | One audit record's inputs, data hashes, execution path | `cli.find_record` (no computation) | audit | none |
| `replay_decision` | Re-run a recorded call, four-way verdict | `audit.replay.verify_replay` (re-dispatches the recorded tool; may fetch) | audit (+ whatever the replayed tool reads) | none |
| `compare_decisions` | Diff two audit records | `cli.find_record` ×2; `cli.cmd_compare` | audit | none |
| `verify_audit_integrity` | Hash-chain check, optional Ed25519 checkpoint | `audit.verify.verify_audit_trail_integrity` / `verify_audit_log_integrity`; `audit.signing.verify_checkpoint_signature` | audit (+ public key file) | none |
| `export_audit_bundle` | Zip a date range of the log | `audit.export.export_bundle` | audit | file (zip; refuses overwrite) |
| `describe_artifact` | Shape, span, per-column stats, head/tail, sha256 of a Parquet artifact | `backtest.artifacts.load_artifact`; `hashlib.sha256` | uri | none |
| `list_strategies` | Strategy parameter contracts | `backtest.strategy_params.STRATEGY_PARAM_SCHEMA`, `_RELATIONS`, `_MAX_WINDOW_BARS` (constants) | none | none |
| `list_stress_scenarios` | Named crash windows | `backtest.stress_test.list_stress_scenarios` | none | none |
| `describe_data_capabilities` | What a provider class can serve | `DataFactory.get_provider`; method-override probes; `provider.get_metadata("AAPL")` | provider metadata (no bars) | none |
| `describe_reference` | Kind/producer/shape of an `sqt://` ref | `handoff.describe` | ref (any kind) | none |
| `read_reference` | Up to 64 rows of a tabular ref | `handoff.parse`, `handoff.resolve` | ref (tabular) | none |
| `list_reference_kinds` | All kinds and conversion edges | `handoff.kinds`; `meta.convert.CONVERSIONS` | none | none |
| `convert_reference` | Kind-to-kind conversion and publish | `meta.convert.convert` — pairs: predictions→signal_panel (sign), predictions→score_panel, score_panel→weight_panel (`backtest.sizing.rank_weighted` / `zscore_normalized`; `vol_scaled` refused), signal_panel→score_panel, equity_curve→returns_panel | ref | ref:<to_kind> |
| `describe_tool` | One tool's contract from the catalog | `mcp.catalog.build_catalog` | catalog | none |
| `validate_tool_call` | Pydantic + strategy-contract check without calling | `agent.tools._TOOL_DISPATCH` ∪ `MODELING_TOOL_DISPATCH` (feature_lab **not** included); `model_cls(**args)`; `backtest.strategy_params.resolve_strategy_params` | catalog | none |
| `describe_temporal_contract` | PIT guarantees of a provider's frame kind | `provider.get_temporal_contract` | provider contract | none |
| `compare_data_sources` | Same ratios from two providers, scale/definition/agree | `provider.get_financial_ratios` ×providers×symbols; `data.comparison.compare_ratio_sources` | fetch (two providers) | none |
| `estimate_tool_cost` | Schema bytes/tokens per runtime | `mcp.catalog.build_catalog`, `select_runtimes`, `CatalogEntry.cost_bytes` | catalog | none |
| `describe_runtime` | Runtime labels, categories, tool names | `agent.runtimes.all_runtimes` | catalog | none |
| `compare_artifacts` | Flattened field-by-field diff of two result dicts | local `_flatten` (no library call) | inline | none |

---

## `data` (18 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `fetch_ohlcv` | One symbol's bars → ref | `provider.get_ohlcv` | fetch | ref:price_panel |
| `fetch_ohlcv_panel` | Universe bars stacked long → ref | `portfolio.portfolio.fetch_ohlcv_panel_sync` | panel-fetch | ref:price_panel (with `entity` column) |
| `fetch_returns_panel` | Wide date×ticker returns → ref | `portfolio.portfolio.fetch_returns_sync` | panel-fetch | ref:returns_panel |
| `fetch_tick_tape` | Trades → ref | `provider.get_trades` | fetch (tick feed) | ref:tick_tape |
| `fetch_quote_panel` | Top-of-book quotes → ref | `provider.get_quotes` | fetch (quote feed) | ref:quote_panel |
| `fetch_financial_ratios` | Ratios with plausibility flags | `provider.get_financial_ratios`; `data.ratios.implausible_value_warnings` | fetch | none |
| `get_dataset_metadata` | Provider guarantees for a symbol/interval | `provider.get_metadata` | provider metadata | none |
| `infer_temporal_contract` | Infer PIT contract from a frame's columns | `data.temporal.contract_for_frame` | ref | none |
| `prepare_vendor_extract` | Databento raw export → library schema Parquet | `data.external.inspect`, `check_schema`; `data.databento.normalize_book` / `normalize_mbo` / `normalize_quotes` / `normalize_trades`, `book_depth`, `level_is_empty`, `split_empty_levels`, `looks_like_databento`; `pyarrow.parquet.ParquetWriter` | file (batched) | file (new Parquet; refuses overwrite) |
| `register_external_dataset` | Register on-disk data by pointer | `handoff.publish_external` (schema check only) | file (schema) | ref (external registration, no copy) |
| `describe_external_dataset` | Schema, fingerprint, changed-since-registration, preview | `handoff.describe`, `handoff.resolve`; `data.external.book_levels` | ref (external) | none |
| `validate_external_dataset` | Batched row-level validation verdict | `data.external_validation.validate_external` | ref (external, batches) | none |
| `build_data_bundle` | Name several refs as one unit | `data.bundle.DataBundle.add`, `.describe` | refs | ref:data_bundle (manifest) |
| `describe_data_bundle` | What a bundle names and promises | `DataBundle.describe` (after resolving every member) | ref:data_bundle + members | none |
| `validate_data_bundle` | Usable / blocking verdict, optional require_pit | `data.bundle.validate_bundle` | ref:data_bundle + members | none |
| `validate_financial_ratios` | Plausibility check on caller-supplied ratios | `data.ratios.implausible_value_warnings` | inline | none |
| `compare_ratio_frames` | Two caller-supplied ratio dicts, gap classified | `data.comparison.compare_ratio_sources` | inline | none |
| `build_continuous_futures_series` | Roll-stitched research series + tradeable contract map | `data.continuous.build_continuous_futures` | inline contract chain | ref:price_panel ×2 (`_research`, `_tradeable`) |

---

## `portfolio` (18 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `run_portfolio_optimization` | Mean-variance / risk parity / Black-Litterman weights | `portfolio.optimize.mean_variance_optimize` **or** `risk_parity_weights` **or** `build_bl_views` + `black_litterman`; `annualized_mean_cov`, `_check_covariance_estimable`, `_small_sample_warnings` | panel-fetch (returns) | none |
| `get_portfolio_risk_attribution` | Portfolio metrics, MCR, PCA exposure, optional factor regression | `metrics.return_metrics.cagr`, `annualized_volatility`; `metrics.risk_metrics.sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `var_historical`, `cvar`, `information_ratio`; local MCR arithmetic (`cov@w*w/var`); `analysis.pca.pca_returns`; `analysis.multi_factor.multi_factor_regression` | fetch (per ticker + benchmark + factors) | none |
| `run_stress_test` | Replay weights over a named crash window | `backtest.stress_test.scenario_dates`, `replay_stress_scenario` | fetch (per ticker over window) | none |
| `get_position_size` | ATR-stop fixed-risk shares + optional half-Kelly | `indicators.volatility.atr`; local Kelly arithmetic | fetch | none |
| `get_capacity_report` | Max account size per ADV participation, days-to-liquidate, sector exposure | `backtest.constraints.capacity_report`, `days_to_liquidate`, `sector_exposure`; `provider.get_ticker_info` | fetch (per ticker + ticker info) | none |
| `get_liquidity_metrics` | Amihud + Corwin-Schultz per ticker (latest) | `backtest.liquidity.amihud_illiquidity`, `corwin_schultz_spread` | fetch (per ticker) | none |
| `estimate_trade_cost` | Itemised cost of one hypothetical trade | `backtest.costs.percentage_commission`, `per_share_commission`, `directional_commission`, `maker_taker_cost`, `fixed_bps_spread`, `pct_of_range_spread`, `impact_cost`, `short_borrow_cost`, `margin_interest` | inline | none |
| `plan_rebalance` | Day-by-day path from held to target weights | `portfolio.rebalance.plan_rebalance` | inline | none |
| `estimate_covariance` | Sample / Ledoit-Wolf / EWMA covariance + conditioning diagnostics | `portfolio.covariance.estimate_covariance` | panel-fetch (returns) | none (matrix returned inline) |
| `construct_weights_from_scores` | Scores → weights and stop | `backtest.sizing.rank_weighted` / `zscore_normalized` / `equal_weight_top_bottom` / `vol_scaled`; `dollar_neutral` | ref:score_panel or ref:predictions (+ ref:returns_panel for vol_scaled) | ref:weight_panel |
| `optimize_risk_parity` | Equal/budgeted risk contribution weights | `portfolio.construction.risk_parity` | inline covariance | none |
| `optimize_hierarchical_risk_parity` | HRP without inverting the covariance | `portfolio.construction.hierarchical_risk_parity` | inline returns map | none |
| `get_factor_exposure_budget` | Factor exposures and variance shares | `portfolio.construction.factor_exposure_budget` | inline | none |
| `analyze_concentration` | Herfindahl, effective N, top-k shares | `portfolio.construction.concentration_analysis` | inline | none |
| `get_liquidity_adjusted_var` | VaR scaled by liquidation horizon | `portfolio.construction.liquidity_adjusted_var` | inline | none |
| `optimize_max_diversification` | Max diversification-ratio weights | `portfolio.construction.max_diversification` | inline covariance | none |
| `get_marginal_risk_contribution` | Marginal and total risk contribution of held weights | `portfolio.construction.marginal_risk_contribution` | inline covariance + weights | none |
| `run_portfolio_scenarios` | Named asset shocks, coverage, sigma moves | `portfolio.construction.portfolio_scenarios` | inline | none |

---

## `microstructure` (17 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `get_microstructure_metrics` | Quoted/effective/realized spread + impact + signed flow, averaged | `analysis.microstructure.microstructure_summary` | fetch (`provider.get_trades`, `get_quotes`; refuses providers without ticks) | none |
| `get_trade_profile` | Volume by trade-size quantile and time of day | `analysis.microstructure.trade_size_profile`, `intraday_volume_profile` | fetch (trades) | none |
| `detect_liquidity_events` | CUSUM change detection over spread/flow/mid channels | `analysis.liquidity_events.detect_liquidity_events` | fetch (trades, quotes; missing channels reported) | none |
| `check_spread_proxy` | Tick-measured effective spread vs Corwin-Schultz from bars | `analysis.microstructure.microstructure_summary`; `backtest.liquidity.corwin_schultz_spread`, `amihud_illiquidity` | fetch (trades + quotes) + fetch (bars) | none |
| `classify_trade_direction` | Lee-Ready (with quotes) or tick-rule signing, published | `analysis.microstructure.sign_trades` | ref:tick_tape (+ ref:quote_panel) | ref:tick_tape (signed) |
| `get_quoted_spread_series` | Spread and imbalance per quote | `analysis.microstructure.quoted_spread` | ref:quote_panel | ref:quote_panel |
| `get_effective_spread_series` | Effective (and realized/impact) spread per trade | `analysis.microstructure.effective_spread` | ref:tick_tape + ref:quote_panel | ref:tick_tape |
| `estimate_roll_spread` | Roll (1984) spread from serial covariance | `analysis.microstructure_estimators.roll_spread` | inline prices | none |
| `estimate_corwin_schultz_spread` | Corwin-Schultz from high/low | `analysis.microstructure_estimators.corwin_schultz_spread` | inline high/low | none |
| `get_amihud_illiquidity` | Amihud ratio + own-history percentile | `analysis.microstructure_estimators.amihud_illiquidity` | inline close/volume | none |
| `estimate_kyle_lambda` | Price impact per signed volume (tick rule) | `analysis.microstructure_estimators.kyle_lambda` | inline close/volume | none |
| `get_order_flow_imbalance` | Bar-based signed volume imbalance, non-overlapping persistence | `analysis.microstructure_estimators.order_flow_imbalance` | inline close/volume | none |
| `estimate_vpin` | Volume-bucket flow one-sidedness | `analysis.microstructure_estimators.estimate_vpin` | inline close/volume | none |
| `get_intraday_volume_profile` | U-shape volume buckets from intraday bars | `analysis.microstructure_estimators.intraday_volume_profile` | inline volume + timestamps | none |
| `get_implementation_shortfall` | Perold decomposition of one execution | `analysis.microstructure_estimators.implementation_shortfall` | inline fills | none |
| `get_order_book_metrics` | Microprice, touch/cumulative imbalance, depth slope, OFI dynamics, per-level profile | `analysis.order_book.book_metrics`; optional `book_dynamics`, `depth_profile`; `data.external.book_levels` | inline snapshots **or** ref:order_book_panel (batched, capped) | none |
| `get_order_event_metrics` | Queue-ahead, order lifetimes, cancel ratios, event rates | `analysis.order_events.order_event_metrics` | inline events **or** ref:order_event_panel (batched, capped) | none |

---

## `derivatives` (12 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `get_option_pricing` | Price + greeks under a chosen model | BS European: `analysis.options.black_scholes_price`, `black_scholes_greeks` (vega/theta/rho rescaled here); otherwise `analysis.pricing.price_option` (black_76 / binomial / American) | inline | none |
| `get_implied_volatility` | Solve BSM IV from a price | `analysis.options.implied_volatility` | inline | none |
| `get_option_greeks` | Greeks in trader units | `analysis.derivatives.option_greeks` | inline | none |
| `analyze_option_strategy` | Multi-leg payoff, breakevens, net greeks | `analysis.derivatives.analyze_strategy` | inline legs | none |
| `fit_volatility_smile` | Parabolic smile fit, skew/curvature | `analysis.derivatives.fit_volatility_smile` | inline strikes/IVs | none |
| `get_volatility_cone` | Realized-vol percentiles by horizon vs current implied | `analysis.derivatives.volatility_cone` | inline price series | none |
| `analyze_vol_term_structure` | Slope/shape of IV by expiry | `analysis.derivatives.analyze_vol_term_structure` | inline map | none |
| `check_put_call_parity` | Parity residual in bps | `analysis.derivatives.check_put_call_parity` | inline | none |
| `get_implied_forward` | Forward from carry inputs | `analysis.derivatives.implied_forward_price` | inline | none |
| `get_expected_move` | ±1σ move from IV, vs realized | `analysis.derivatives.expected_move` | inline | none |
| `simulate_delta_hedge` | Monte Carlo delta-hedge P&L implied vs realized vol | `analysis.derivatives.simulate_delta_hedge` | inline | none |
| `get_option_risk_scenarios` | Spot×vol×time scenario P&L grid | `analysis.derivatives.option_risk_scenarios` | inline | none |

---

## `delta_one` (18 tools)

| Tool | Purpose | Calls | Reads | Writes |
|---|---|---|---|---|
| `analyze_cash_futures_basis` | Fair basis vs quoted, rich/cheap | `delta_one.basis.cash_futures_basis` | inline | none |
| `solve_forward_carry` | Solve one carry input from spot/forward | `delta_one.carry.solve_carry` | inline | none |
| `analyze_basis_history` | Basis time series, rolling stats | `delta_one.basis.basis_history` | inline series | none |
| `analyze_futures_curve` | Contango/backwardation, roll yields across a chain | `delta_one.futures.futures_curve` | inline contracts | none |
| `analyze_roll` | Cost of rolling front→next | `delta_one.futures.roll_analysis` | inline | none |
| `size_futures_hedge` | Contracts to hedge a beta-weighted book | `delta_one.hedging.futures_hedge` | inline | none |
| `analyze_hedge_effectiveness` | Ex-post variance reduction, residual beta | `delta_one.hedging.hedge_effectiveness` | inline return series | none |
| `analyze_index_basket` | Basket vs index level, divisor arithmetic | `delta_one.baskets.index_basket` | inline constituents | none |
| `compare_delta_one_expressions` | Normalised all-in cost of futures/ETF/swap/basket | `delta_one.expressions.compare_expressions` | inline | none |
| `optimize_replication_basket` | Sparse basket tracking a benchmark | `delta_one.replication.optimize_replication_basket` | inline returns map + benchmark | none |
| `analyze_etf_fair_value` | ETF vs NAV/basket, creation arbitrage | `delta_one.etf.etf_fair_value` | inline | none |
| `price_total_return_swap` | TRS legs valuation | `delta_one.swaps.price_total_return_swap` | inline | none |
| `analyze_total_return_future` | Implied financing from a TRF quote | `delta_one.swaps.total_return_future` | inline | none |
| `analyze_dividend_points` | Index dividend points to expiry, implied vs forecast | `delta_one.dividends.dividend_points` | inline constituents | none |
| `analyze_index_rebalance` | Flow vs ADV from weight changes | `delta_one.rebalance.index_rebalance_flow` | inline | none |
| `detect_basis_dislocation` | CUSUM breaks in a basis series | `delta_one.basis.detect_basis_dislocation` | inline series | none |
| `monitor_spread_stream` | Incremental spread monitor with carried state | `delta_one.streaming.new_spread_monitor` / `reset_spread_monitor`; `update_spread_monitor` | inline batch + state blob | none (state returned to caller) |
| `scan_basis_dislocations` | Rank many spot/future pairs by dislocation | `delta_one.scan.basis_scan` | inline pairs | none |

---

# Analysis 1 — Backing-function reuse

## Library functions called by more than one tool

| Function | Tools (count) |
|---|---|
| `backtest.engine.run_strategy` (direct or via `_shared._run_backtest`, `backtest.panel`, `backtest.pairs`) | run_sma/rsi/macd/bollinger_backtest, run_buy_and_hold, compare_strategies, run_regime_adaptive_backtest, run_regime_adaptive_walkforward_backtest, run_walk_forward_backtest, run_custom_signal_backtest, run_signal_panel_backtest, get_robustness_diagnostics, run_backtest_compact, get_backtest_diagnostics, compare_cost_models, run_strategy_matrix (**16**) |
| `backtest.strategies.STRATEGY_REGISTRY` / `RUNNABLE` | the four named backtests, compare_strategies, both regime-adaptive tools, run_walk_forward_backtest, get_robustness_diagnostics, run_backtest_compact, get_backtest_diagnostics, compare_cost_models, run_strategy_matrix (**13**) |
| `backtest.engine.backtest_grid` | run_regime_adaptive_backtest, run_regime_adaptive_walkforward_backtest, run_walk_forward_backtest, run_backtest_optimization, get_robustness_diagnostics (**5**) |
| `backtest.sizing.rank_weighted` / `zscore_normalized` / `equal_weight_top_bottom` / `vol_scaled` / `dollar_neutral` | run_portfolio_simulation, construct_weights_from_scores, convert_reference (score_panel→weight_panel), evaluate_model_portfolio (via `portfolio_eval`) (**4**) |
| `backtest.portfolio_engine.run_portfolio_simulation` | run_portfolio_simulation, run_pair_trade_backtest (via `backtest.pairs`), evaluate_model_portfolio (via `portfolio_eval`) (**3**) |
| `metrics.risk_metrics.var_historical` | analyze_stock_risk, get_extended_risk_metrics, get_tail_risk_metrics, get_portfolio_risk_attribution, run_portfolio_simulation, run_backtest_compact, calculate_series_metrics (**7**) |
| `metrics.risk_metrics.sharpe_ratio` | analyze_stock_risk, get_portfolio_risk_attribution, run_portfolio_simulation, run_pair_trade_backtest, get_robustness_diagnostics, calculate_series_metrics (**6**) |
| `metrics.risk_metrics.max_drawdown` | analyze_stock_risk, get_portfolio_risk_attribution, run_portfolio_simulation, run_pair_trade_backtest, get_drawdown_table, calculate_series_metrics (**6**) |
| `metrics.risk_metrics.cvar` | analyze_stock_risk, get_extended_risk_metrics, get_portfolio_risk_attribution, run_portfolio_simulation, run_backtest_compact, calculate_series_metrics (**6**) |
| `metrics.risk_metrics.sortino_ratio` | analyze_stock_risk, get_portfolio_risk_attribution, run_portfolio_simulation, run_pair_trade_backtest, calculate_series_metrics (**5**) |
| `metrics.return_metrics.cagr` | get_extended_risk_metrics, get_portfolio_risk_attribution, run_backtest_compact, compare_cost_models, calculate_series_metrics (**5**) |
| `metrics.return_metrics.annualized_volatility` | get_volatility_estimators, get_portfolio_risk_attribution, run_portfolio_simulation, run_pair_trade_backtest, calculate_series_metrics (**5**) |
| `metrics.risk_metrics.information_ratio` | analyze_stock_risk, get_portfolio_risk_attribution, run_portfolio_simulation (**3**) |
| `metrics.risk_metrics.calmar_ratio`, `var_parametric` | get_extended_risk_metrics, calculate_series_metrics (**2** each) |
| `analysis.regression.calculate_beta` | analyze_stock_risk, get_extended_risk_metrics, run_screener (inside `screen_stocks`) (**3**) |
| `analysis.hurst.hurst_exponent` | run_hurst_analysis, run_regime_adaptive_backtest, run_regime_adaptive_walkforward_backtest, get_rally_signal (inside `detect_rally`) (**4**) |
| `analysis.cointegration.cointegration_test`, `compute_spread`, `spread_zscore` | run_cointegration_test, scan_pairs; `spread_zscore` also in run_kalman_hedge_ratio (**2–3**) |
| `analysis.pca.pca_returns` | run_pca_analysis, get_portfolio_risk_attribution (**2**) |
| `analysis.multi_factor.multi_factor_regression` | run_factor_regression, get_portfolio_risk_attribution (**2**) |
| `indicators.panel.technical_indicators_panel` | get_technical_panel, compute_indicator_panel (**2**) |
| `indicators.volatility.atr` | get_technical_analysis, get_position_size (**2**) |
| `backtest.liquidity.amihud_illiquidity`, `corwin_schultz_spread` | get_liquidity_metrics, check_spread_proxy (**2**) |
| `analysis.microstructure.microstructure_summary` | get_microstructure_metrics, check_spread_proxy (**2**) |
| `metrics.diagnostics.exposure_stats` | run_backtest_compact, get_backtest_diagnostics (**2**) |
| `portfolio.portfolio.fetch_returns_sync` | get_portfolio_analysis, get_correlation_analysis, run_monte_carlo_simulation, run_portfolio_optimization, estimate_covariance, fetch_returns_panel (**6**) |
| `portfolio.portfolio.fetch_ohlcv_panel_sync` | get_technical_panel, compute_indicator_panel, run_signal_panel_backtest, run_portfolio_simulation, fetch_ohlcv_panel (**5**) |
| `data.comparison.compare_ratio_sources` | compare_ratio_frames (data, inline), compare_data_sources (meta, fetch) (**2**) |
| `data.ratios.implausible_value_warnings` | fetch_financial_ratios, validate_financial_ratios (**2**) |
| `modeling.agent.tools._load_dataset_panel` (`artifacts.load_artifact` + `hash_dataframe`) | run_model_experiment, check_leakage, analyze_features, join_point_in_time, explain_dataset_row_loss, compare_models (paired), analyze_model_errors, and all 9 feature_lab tools (**16**) |
| `modeling.registry.model_registry.load_manifest` | inspect_model, list_models, compare_models, monitor_model, analyze_model_errors (+ inside score_model, evaluate_model_portfolio) (**5+**) |
| `modeling.registry.lifecycle.current_stage` / `promotions` | inspect_model, list_models, monitor_model / inspect_model, promote_model (**3 / 2**) |
| `modeling.ensemble.load_oos_predictions` | build_model_ensemble (inside `combine_predictions`), compare_models (paired), analyze_model_errors (**3**) |
| `modeling.engine.run_experiment` | run_model_experiment, run_feature_ablation (**2**) |
| `modeling.analysis.feature_report.feature_predictive_stats` | profile_feature, get_feature_redundancy (+ inside `build_feature_report` for analyze_features) (**2–3**) |
| `modeling.analysis.feature_report.lead_lag_ic_curve` | profile_feature, get_feature_ic_decay (**2**) |
| `handoff.describe` | describe_reference, describe_external_dataset, validate_external_dataset (**3**) |
| `mcp.catalog.build_catalog` | describe_tool, estimate_tool_cost (**2**) |
| `cli.find_record` | explain_decision, replay_decision, compare_decisions (**3**) |
| `DataFactory.get_provider().get_ohlcv` | ~45 tools across research, backtest, portfolio, microstructure (check_spread_proxy), data |

## Duplicate implementations of the same computation (different modules, called by different tools)

These are worth knowing because two tools that look like the same question in two shapes may not agree numerically:

| Computation | Implementation A (tool) | Implementation B (tool) |
|---|---|---|
| Amihud illiquidity | `backtest.liquidity.amihud_illiquidity` (get_liquidity_metrics, check_spread_proxy) | `analysis.microstructure_estimators.amihud_illiquidity` (get_amihud_illiquidity) |
| Corwin-Schultz spread | `backtest.liquidity.corwin_schultz_spread` (get_liquidity_metrics, check_spread_proxy) | `analysis.microstructure_estimators.corwin_schultz_spread` (estimate_corwin_schultz_spread) |
| Intraday volume profile | `analysis.microstructure.intraday_volume_profile` (get_trade_profile, from ticks) | `analysis.microstructure_estimators.intraday_volume_profile` (get_intraday_volume_profile, from bars) |
| Deflated Sharpe ratio | `backtest.robustness.deflated_sharpe_ratio` (get_robustness_diagnostics) | `backtesting.overfitting.deflated_sharpe_ratio` (get_deflated_sharpe_ratio) |
| Block bootstrap | `backtest.robustness.block_bootstrap_ci` (get_robustness_diagnostics) | `analysis.inference.bootstrap_statistic` (get_bootstrap_interval); also `backtest.monte_carlo.simulate_forward_paths[_terminal]` and `backtesting.overfitting.reality_check` carry their own |
| Drawdown episodes | `metrics.diagnostics.top_n_drawdowns` (get_backtest_diagnostics) | `metrics.diagnostics.drawdown_periods` (get_drawdown_table); `analysis.diagnostics.drawdown_profile` (get_drawdown_profile) — three |
| Risk parity | `portfolio.optimize.risk_parity_weights` (run_portfolio_optimization method=risk_parity) | `portfolio.construction.risk_parity` (optimize_risk_parity) |
| Marginal risk contribution | inline `cov@w*w/var` in get_portfolio_risk_attribution | `portfolio.construction.marginal_risk_contribution` (get_marginal_risk_contribution) |
| Black-Scholes greeks | `analysis.options.black_scholes_greeks` (get_option_pricing, rescaled in the tool) | `analysis.derivatives.option_greeks` (get_option_greeks) |
| Implied forward / carry | `analysis.derivatives.implied_forward_price` (get_implied_forward) | `delta_one.carry.solve_carry`, `delta_one.basis.cash_futures_basis` |
| Feature drift (PSI/KS) | `modeling.monitoring.drift_report` (monitor_model, vs training reference) | `modeling.analysis.feature_stability.feature_drift` (get_feature_drift, either side of a date) |
| Change detection | `analysis.structure.detect_change_points` (binary segmentation), `analysis.diagnostics.structural_break_test` (Chow), `analysis.liquidity_events.detect_liquidity_events` (CUSUM on ticks), `delta_one.basis.detect_basis_dislocation` + `delta_one.streaming` (CUSUM on basis) — four families |
| Parameter surface | `backtest.robustness.parameter_sensitivity` (get_robustness_diagnostics) | `backtesting.overfitting.parameter_decay` (analyze_parameter_decay) |
| Break-even cost | bisection over `run_strategy` in compare_cost_models | `backtesting.trade_analysis.break_even_cost` (estimate_break_even_cost) |

## Single-function wrappers (one library computation call; argument conversion + result typing only)

**119 tools.** Listed so the count is auditable:

- **research (25):** test_autocorrelation, run_seasonality_analysis, get_entropy_measures, get_sharpe_stability, get_drawdown_profile, get_lead_lag_matrix, test_structural_break, get_bootstrap_interval, compare_distributions, get_correlation_stability, decompose_returns, test_normality, estimate_tail_index, detect_change_points, get_partial_correlation, test_granger_causality, analyze_tail_dependence, run_stationarity_tests, detect_regimes, run_garch_volatility_forecast, get_rally_signal, run_screener, get_rolling_beta, get_technical_panel, compute_indicator_panel
- **backtest (17):** run_buy_and_hold, run_backtest_optimization, run_signal_panel_backtest, run_terminal_monte_carlo, get_deflated_sharpe_ratio, estimate_backtest_overfitting, build_purged_cv_splits, run_reality_check, get_regime_stratified_performance, analyze_parameter_decay, run_monte_carlo_trade_paths, analyze_trade_clustering, compare_against_random, get_exposure_attribution, estimate_break_even_cost, run_futures_backtest, run_futures_hedge_backtest
- **portfolio (10):** plan_rebalance, estimate_covariance, optimize_risk_parity, optimize_hierarchical_risk_parity, get_factor_exposure_budget, analyze_concentration, get_liquidity_adjusted_var, optimize_max_diversification, get_marginal_risk_contribution, run_portfolio_scenarios
- **microstructure (14):** get_microstructure_metrics, detect_liquidity_events, classify_trade_direction, get_quoted_spread_series, get_effective_spread_series, estimate_roll_spread, estimate_corwin_schultz_spread, get_amihud_illiquidity, estimate_kyle_lambda, get_order_flow_imbalance, estimate_vpin, get_intraday_volume_profile, get_implementation_shortfall, get_order_event_metrics
- **derivatives (11):** all except get_option_pricing (which branches between two pricers)
- **delta_one (17):** all except monitor_spread_stream (constructor/reset + update)
- **data (6):** infer_temporal_contract, validate_external_dataset, validate_data_bundle, validate_financial_ratios, compare_ratio_frames, build_continuous_futures_series
- **meta (5):** replay_decision, export_audit_bundle, list_stress_scenarios, describe_temporal_contract, convert_reference
- **modeling (8):** list_features, score_model, evaluate_model_portfolio, list_modeling_capabilities, explain_dataset_row_loss, analyze_features, promote_model, validate_pit_records
- **feature_lab (6):** get_feature_ic_decay, select_features, compare_feature_sets, get_feature_drift, get_feature_regime_stability, run_feature_permutation_test

A further **27 tools call no numerical library function at all** (fetch-and-publish, registry reads, catalog introspection, own diff code): get_stock_fundamentals, fetch_ohlcv, fetch_ohlcv_panel, fetch_returns_panel, fetch_tick_tape, fetch_quote_panel, get_dataset_metadata, register_external_dataset, describe_external_dataset, build_data_bundle, describe_data_bundle, describe_reference, read_reference, list_reference_kinds, describe_runtime, describe_tool, estimate_tool_cost, explain_decision, compare_decisions, list_strategies, describe_artifact, describe_data_capabilities, compare_artifacts, validate_tool_call, inspect_model, list_models, list_datasets.

## Merge candidates (thin wrappers that differ only by a parameter)

1. **`run_sma_backtest` / `run_rsi_backtest` / `run_macd_backtest` / `run_bollinger_backtest`** — all four call `_dispatch_backtest(input_data, default)` and already honour `strategy_type`; they are four names for one function. `run_buy_and_hold` is the same engine with a constant signal (`RUNNABLE` already contains `buy_and_hold`). One `run_backtest(strategy_type=...)` would replace five tools with no loss.
2. **The 13 inline series tests in research** (`diagnostic_tools` + `inference_tools`) each wrap one `analysis.diagnostics.*` / `analysis.inference.*` call with the same `List[float]` (+ optional dates) input. `test_autocorrelation`, `test_normality`, `test_structural_break`, `estimate_tail_index` are a natural `test_series(test=...)`; `get_sharpe_stability`, `decompose_returns`, `get_drawdown_profile`, `get_entropy_measures`, `get_bootstrap_interval` a `profile_series(measure=...)`.
3. **The 5 trade-return tools in backtest** (`trade_tools.py`) all take `trade_returns: List[float]` and wrap one `backtesting.trade_analysis.*` call: `analyze_trades(analysis=...)`.
4. **The 8 bar estimators in microstructure** all take close/volume (or high/low) lists and wrap one `analysis.microstructure_estimators.*` call: `estimate_liquidity(estimator=...)`. Note `get_liquidity_metrics` (portfolio) is already the fetch-shaped parameterised version of two of them.
5. **The 8 `portfolio.construction` tools** take an inline covariance/weights and wrap one call each; `optimize_risk_parity` / `optimize_max_diversification` / `optimize_hierarchical_risk_parity` are `optimize_weights(method=...)`, and `run_portfolio_optimization` already has a `method` switch for the fetch-shaped equivalents.
6. **`get_technical_panel` / `compute_indicator_panel`** — same `technical_indicators_panel` call; the difference is "latest bar inline" vs "history as refs". One tool with `publish: bool`.
7. **`compare_ratio_frames` (data) / `compare_data_sources` (meta)** — same `compare_ratio_sources`; one takes dicts, the other fetches. A `DataSource`-style union would collapse them.
8. **`fetch_financial_ratios` / `validate_financial_ratios` / `get_stock_fundamentals`** — the same ratios object with or without `implausible_value_warnings`.
9. **`analyze_stock_risk` / `get_extended_risk_metrics`** are fixed bundles of the same `metrics.*` functions `calculate_series_metrics` exposes by name; the only thing they add is the benchmark-relative metrics (`calculate_beta`, `treynor_ratio`, `information_ratio`), which `calculate_series_metrics` cannot compute because it has no benchmark source.
10. **`run_monte_carlo_simulation` / `run_terminal_monte_carlo`** — same bootstrap kernel family (`simulate_forward_paths` vs `_terminal`), the first fetch-shaped (tickers + weights), the second `DataSource`-shaped. A `keep_paths: bool` on the second would retire the first.
11. **`profile_feature` / `get_feature_redundancy` / `get_feature_ic_decay`** are typed slices of what `analyze_features` returns as one untyped report; the module docstring says so.
12. **`describe_reference` / `describe_external_dataset` / `describe_artifact` / `describe_data_bundle`** describe the same kind of thing (a stored frame) addressed three ways (ref, external ref, Parquet URI).

---

# Analysis 2 — Shape patterns

## Shape taxonomy

| Shape | Description | Tools |
|---|---|---|
| **A. Symbol(s) + date range → fetch** | The tool calls the provider itself; result inline. | research 27 (every tool in `research/tools.py`), backtest 20 (`tools.py` minus get_drawdown_table; run_signal_panel_backtest and run_portfolio_simulation fetch prices but also accept refs), portfolio 7 (run_portfolio_optimization, get_portfolio_risk_attribution, run_stress_test, get_position_size, get_capacity_report, get_liquidity_metrics, estimate_covariance), microstructure 4 (tick tools), meta 1 (compare_data_sources), modeling 2 (build_model_dataset, score_model via as_of + universe), data 5 (fetch_* — fetch then publish rather than return) |
| **B. Artifact reference** (`sqt://`, Parquet URI, `dataset_id`/`model_id`) | The tool resolves something a previous tool produced. | data 7 (infer_temporal_contract, describe/validate_external_dataset, build/describe/validate_data_bundle, plus fetch tools as *producers*), research 1 (compute_indicator_panel optional), backtest 1 (get_drawdown_table URI), portfolio 1 (construct_weights_from_scores), microstructure 3 (classify_trade_direction, get_quoted_spread_series, get_effective_spread_series), meta 4 (describe_reference, read_reference, convert_reference, describe_artifact), modeling 18 (everything keyed by dataset_id/model_id, plus score_predictions on ref:predictions and monitor_model on URIs), feature_lab 9 (dataset_id) |
| **C. Inline arrays / dicts** | Numbers travel through the call. | research 13 (diagnostic + inference), backtest 13 (validation 6, trade 5, futures 2), portfolio 10 (8 construction + plan_rebalance + estimate_trade_cost), microstructure 8 (estimators + implementation shortfall), derivatives 12, delta_one 18, data 3 (validate_financial_ratios, compare_ratio_frames, build_continuous_futures_series), meta 2 (compare_artifacts, validate_tool_call), modeling 2 (validate_pit_records, validate_model_spec) |
| **D. `DataSource` union (symbol OR ref OR values)** | The three shapes in one field. | **only 2 tools:** calculate_series_metrics, run_terminal_monte_carlo |
| **E. Dual inline-or-ref** | Two fields, exactly one required. | run_signal_panel_backtest (`signal_panel` / `signal_panel_ref`), run_portfolio_simulation (`target_weights` / `target_weights_ref`), get_order_book_metrics (`snapshots` / `ref`), get_order_event_metrics (`events` / `ref`), compute_indicator_panel (`tickers` / `price_panel_ref`) |

Whole runtimes are single-shape: **derivatives and delta_one are 100% inline (30 tools, nothing fetches, nothing takes a ref)**; feature_lab is 100% `dataset_id`; meta's provenance half is 100% audit-log.

## Same computation in two shapes inside one runtime

| Runtime | Fetch / symbol shape | Inline or ref shape | Same underlying maths? |
|---|---|---|---|
| research | analyze_stock_risk, get_extended_risk_metrics (symbol + benchmark) | calculate_series_metrics (DataSource) | yes, same `metrics.*` functions — but the benchmark-relative half is missing from the inline shape |
| research | get_technical_panel (latest bar) | compute_indicator_panel (fetch or ref → history refs) | yes, `technical_indicators_panel` |
| research | get_correlation_analysis (universe matrix) | get_correlation_stability (one pair inline) | different modules |
| research | run_hurst_analysis rolling regime fractions (fetch) | get_sharpe_stability (inline) | different measures of the same "did it change" question |
| research | detect_change_points (fetch; unknown date) | test_structural_break (inline; known index) | intentionally different tests, no shared shape |
| research | analyze_tail_dependence (fetch pair) | estimate_tail_index / test_normality (inline single series) | different questions, but tail work is split by shape |
| research | get_drawdown_* — **none** fetch-shaped | get_drawdown_profile (inline) | the fetch-shaped drawdown table lives in backtest (get_drawdown_table, URI) |
| backtest | run_monte_carlo_simulation (tickers) | run_terminal_monte_carlo (DataSource) | same block bootstrap, different storage |
| backtest | get_robustness_diagnostics (fetch + grid → DSR, sensitivity, bootstrap CI) | get_deflated_sharpe_ratio, analyze_parameter_decay, get_bootstrap_interval (research) — all inline | **different implementations** (`backtest.robustness` vs `backtesting.overfitting` / `analysis.inference`) |
| backtest | get_backtest_diagnostics top drawdowns (fetch + rerun) | get_drawdown_table (URI) | different functions (`top_n_drawdowns` vs `drawdown_periods`) |
| backtest | compare_cost_models breakeven commission (fetch; bisection over run_strategy) | estimate_break_even_cost (inline trade returns) | different definitions (commission % on a signal vs flat bps per trade) |
| backtest | run_custom_signal_backtest (fetch + inline signal dict) | run_signal_panel_backtest (fetch + inline or ref) | single-symbol has no ref path; panel has both |
| backtest | run_regime_adaptive_backtest (detects regime itself) | get_regime_stratified_performance (caller supplies labels) | the runtime detects regimes in one tool and cannot pass them to the other |
| portfolio | run_portfolio_optimization(method=risk_parity) (fetch) | optimize_risk_parity (inline covariance) | **two implementations** |
| portfolio | get_portfolio_risk_attribution MCR (fetch) | get_marginal_risk_contribution (inline) | two implementations |
| portfolio | run_stress_test (fetch; named windows) | run_portfolio_scenarios (inline shocks) | different question (historical replay vs hypothetical) |
| portfolio | get_liquidity_metrics (fetch; latest Amihud/CS) | get_liquidity_adjusted_var (inline ADV) | different question |
| portfolio | estimate_covariance (fetch → matrix returned **inline**) | 5 construction tools take that matrix inline | the handoff is through the conversation, not a ref |
| microstructure | get_microstructure_metrics (tick fetch; averages) | get_quoted_spread_series / get_effective_spread_series (refs; per-event series) | same `analysis.microstructure` module, summary vs series |
| microstructure | get_trade_profile intraday buckets (tick fetch) | get_intraday_volume_profile (inline bars) | two implementations |
| microstructure | check_spread_proxy's Corwin-Schultz + Amihud (bar fetch) | estimate_corwin_schultz_spread, get_amihud_illiquidity (inline) | two implementations |
| microstructure | detect_liquidity_events (tick fetch; CUSUM on spread/flow) | get_order_book_metrics OFI (inline/ref book) | different flow definitions (bar signed volume vs Cont-Kukanov-Stoikov) |
| data / meta | compare_data_sources (meta; fetch two providers) | compare_ratio_frames (data; inline) | same `compare_ratio_sources` |
| data | fetch_financial_ratios (fetch + check) | validate_financial_ratios (inline check) | same `implausible_value_warnings` |
| modeling / feature_lab | analyze_features (dataset_id; whole report) | profile_feature, get_feature_redundancy, get_feature_ic_decay (dataset_id; one question) | same `feature_report` functions, same shape, different granularity |
| modeling / feature_lab | monitor_model drift (model_id + scored URI) | get_feature_drift (dataset_id + split date) | two PSI/KS implementations |

---

# Analysis 3 — Gaps by runtime

Each gap names the existing computation it extends and why the next question cannot be answered inside the runtime today.

## research

1. **Cross-sectional ranking of per-symbol statistics.** run_hurst_analysis, get_rolling_beta, get_volatility_estimators, run_garch_volatility_forecast, get_tail_risk_metrics, estimate_tail_index and analyze_stock_risk each answer for one symbol. Only run_screener (fixed filter vocabulary) and get_technical_panel (indicators) operate on a universe. "Which of these 30 names has the fattest tail / most beta drift / highest Hurst" is 30 calls with no ranked table.
2. **Stability over time of fitted relationships.** run_factor_regression returns a 20-point rolling tail but no test of whether loadings drifted (test_structural_break exists but only inline and needs a known index; get_rolling_beta reports drift for one beta only). run_cointegration_test gives one hedge ratio and one ADF; nothing reports rolling cointegration p-values or whether the pair stayed cointegrated across sub-windows, even though run_kalman_hedge_ratio computes the whole `Hedge_Ratio` series and discards everything but the last value and its std.
3. **Forecast evaluation.** run_garch_volatility_forecast fits and forecasts; no tool compares a forecast with the subsequently realized vol that get_volatility_estimators can compute — no GARCH backtest.
4. **Universe-level correlation regime.** get_correlation_analysis is one matrix at one window; get_correlation_stability is one pair, inline. Nothing reports rolling average pairwise correlation or rolling diversification ratio for a universe.
5. **The 13 inline tools cannot read refs.** get_sharpe_stability, get_drawdown_profile, decompose_returns, get_bootstrap_interval, compare_distributions, etc. take `List[float]` only. A run_backtest_compact result publishes `sqt://equity_curve`, and convert_reference can turn it into `returns_panel`, but the only consumer that accepts a ref is calculate_series_metrics; getting an equity curve into get_sharpe_stability means read_reference (capped at 64 rows) or pasting it. The `DataSource` union exists (2 tools) and is the obvious fix.
6. **Benchmark-relative diagnostics.** decompose_returns, get_sharpe_stability, get_bootstrap_interval have no benchmark; analyze_stock_risk computes information_ratio and calculate_beta but as one number, so "did alpha decay" has no tool.
7. **Scans beyond cointegration.** scan_pairs scans a fetched universe for cointegration; get_lead_lag_matrix and test_granger_causality do the lead-lag equivalent but the former is inline and the latter one pair. analyze_tail_dependence and get_partial_correlation are pairwise with no universe scan.
8. **PCA outputs are dead-ends.** run_pca_analysis returns loadings inline; no tool publishes PC return series as a ref so they could feed run_factor_regression as factors, and no rolling explained-variance (factor-concentration regime) exists.
9. **get_rally_signal and get_data_quality_report are single-symbol** with no universe form, though `detect_rally` and the `data.quality` functions are trivially loopable.

## backtest

1. **Statistical comparison of two runs.** compare_strategies and run_strategy_matrix rank by point estimates; run_reality_check, compare_distributions and get_bootstrap_interval exist but only inline. No tool takes two `sqt://equity_curve` refs and says whether the difference exceeds noise.
2. **Custom signals get one run and nothing else.** get_backtest_diagnostics, compare_cost_models, run_walk_forward_backtest and get_robustness_diagnostics all require `strategy_type ∈ STRATEGY_REGISTRY`. A caller-supplied signal (run_custom_signal_backtest / run_signal_panel_backtest) cannot be cost-swept, diagnosed for MAE/MFE, or walked forward.
3. **The published trade log has no consumer.** run_backtest_compact publishes `sqt://trade_log`; run_monte_carlo_trade_paths, analyze_trade_clustering, compare_against_random and estimate_break_even_cost need `trade_returns: List[float]` inline. get_drawdown_table reads the curve URI; nothing reads the trades URI/ref.
4. **Grid results cannot reach the surface tools.** run_backtest_optimization returns top-N rows; analyze_parameter_decay needs a 1-D `parameter_values`/`performance` pair inline and get_deflated_sharpe_ratio needs `trial_sharpes` inline — the agent must re-key the grid by hand, and 2-D surfaces have no tool.
5. **Regime labels do not cross tools.** run_regime_adaptive_backtest computes a Hurst regime per window and detect_regimes (research) labels each observation, but get_regime_stratified_performance needs labels pasted inline; no tool stratifies a backtest by the regimes the library itself detected.
6. **Only run_backtest_compact publishes.** run_portfolio_simulation, run_pair_trade_backtest, run_walk_forward_backtest and run_signal_panel_backtest return equity curves inline (full lists) and publish nothing, so their curves cannot feed get_drawdown_table or calculate_series_metrics without pasting.
7. **Cost models are pct-only in the sweeps.** compare_cost_models varies commission/slippage pct; the square-root impact model (`backtest.costs.impact_cost`, used by estimate_trade_cost and run_portfolio_simulation's `use_impact_model`) and the ADV-participation constraint are not sweepable, so "does this survive impact" has no single call.
8. **Futures tools take dicts, not refs.** build_continuous_futures_series (data) publishes the tradeable map as `sqt://price_panel`, but run_futures_backtest and run_futures_hedge_backtest take `prices: Dict[str, float]` inline; the ref must be read back through read_reference (64 rows).
9. **No walk-forward for portfolio or pair strategies.** Walk-forward exists for single-symbol registry strategies; run_pair_trade_backtest and run_portfolio_simulation have no out-of-sample counterpart (e.g. re-estimating hedge ratio / weights per window).

## modeling

1. **IC stability across folds.** inspect_model(view="validation") returns per-fold detail, but no tool tests whether OOS IC decays over folds the way get_sharpe_stability tests a Sharpe; compare_models(paired) compares mean per-date IC only.
2. **Live scores are outside the handoff graph.** score_model writes `predictions_*.parquet` (a URI) rather than an `sqt://predictions` ref, so its output cannot go through convert_reference → construct_weights_from_scores → run_portfolio_simulation; evaluate_model_portfolio only accepts a model_id and uses OOS predictions. There is no "simulate what the live scores would trade".
3. **Ensembles cannot be evaluated economically.** build_model_ensemble publishes `sqt://predictions`; score_predictions can score it, but evaluate_model_portfolio, monitor_model and analyze_model_errors take a model_id, so an ensemble has no portfolio evaluation or error analysis.
4. **Search surface is invisible.** validate_model_spec counts fits and run_model_experiment picks the best candidate; nothing reports the inner search's candidate-by-candidate results (the analyze_parameter_decay question for estimator hyperparameters).
5. **Refit / refresh.** There is no tool that re-runs a registered model's bundled `dataset_spec` + `ModelSpec` over a later window; refreshing means build_model_dataset + run_model_experiment by hand and then compare_models, with no lineage link between the two model_ids.
6. **Calibration is diagnosed, never applied.** analyze_model_errors reports calibration slope and ECE; no tool produces a recalibrated prediction ref or a scaling factor for construct_weights_from_scores.
7. **score_predictions has no survival branch** while `_HEADLINE_METRIC` and the engine support a survival task; survival predictions from register_external_panel datasets can be ranked by compare_models but not scored from a ref.
8. **Feature-importance drift** across folds is aggregated into `feature_importance_summary`; per-fold importance exists in the engine (`fold_feature_importance`) but is not exposed, so "did the model start leaning on a different feature" is unanswerable.
9. **validate_tool_call does not cover feature_lab** (`_TOOL_DISPATCH` ∪ `MODELING_TOOL_DISPATCH` only), so the runtime's own pre-flight check cannot validate a run_feature_ablation call — the most expensive call in feature_lab.

## feature_lab

1. **Everything is one feature or one panel.** get_feature_drift, get_feature_regime_stability and run_feature_permutation_test are per feature; "which of the 40 features drifted / lost IC / fails the permutation null" is 40 calls each, while `feature_stability` / `permutation_test_ic` are trivially loopable.
2. **The permutation null is not wired into selection.** run_feature_permutation_test returns `null_p95_abs` as "the honest floor for select_features(min_abs_rank_ic=...)", but select_features takes a fixed floor; nothing runs the null per feature and selects against it.
3. **No cross-dataset comparison.** compare_feature_sets compares two sets on one panel; the same feature's IC on two datasets (different universe, interval or horizon) has no tool, though `feature_predictive_stats` on two panels is all it needs.
4. **Horizon is fixed.** get_feature_ic_decay shifts the feature against one target; a multi-target dataset (register_external_panel `targets`) can hold several horizons but no tool reports IC by horizon without a run_model_experiment per target.
5. **Interaction / incremental IC.** get_feature_redundancy is pairwise correlation; the only conditional measure is run_feature_ablation, which needs a full ModelSpec and refits. A cheap "IC of B after residualising on A" is absent.
6. **IC series are never published.** get_feature_regime_stability returns block ICs inline; no rolling IC ref that research's inline diagnostics (get_sharpe_stability-style) could consume.

## meta

1. **No listing of decisions.** explain_decision, replay_decision and compare_decisions need a request_id the caller already holds; there is no "recent calls of tool X / for symbol Y" — the same gap list_models closed for modeling.
2. **No lineage walk.** describe_reference reports `producer` but not the refs that producer consumed; "what fetched the panel this weight_panel was built from" is not answerable, though every tool call is in the audit log with its inputs.
3. **compare_artifacts is dict-only.** Two `sqt://` refs or two Parquet URIs cannot be diffed numerically; describe_artifact gives a content hash (equal/unequal) and nothing in between.
4. **Dead-end and missing conversions.** compute_indicator_panel publishes `indicator_panel`, which is accepted by no tool and appears in no `CONVERSIONS` edge; there is no `price_panel → returns_panel` (fetch_returns_panel refetches instead), no `returns_panel → equity_curve`, and score_panel→weight_panel refuses `vol_scaled` because the returns are not carried.
5. **Cost is schema bytes only.** estimate_tool_cost prices the listing; the audit log records `duration_ms` per call (explain_decision returns it) but no tool aggregates expected run time per tool.
6. **replay_decision replays one record**; no batch "replay everything from date D and report which changed" though verify_replay and the day files are both available.

## data

1. **No transformation of published refs.** The fetch tools publish, but there is no align (two refs onto one calendar), resample (daily→weekly), slice (date/entity subset) or join tool; convert_reference covers five kind pairs and none of these. A returns_panel over 500 names cannot be narrowed to 20 without refetching.
2. **No quality verdict for a published price panel.** get_data_quality_report (research) works on a fresh single-symbol fetch; validate_external_dataset is external-only; validate_data_bundle checks PIT/provenance, not gaps or stale prices. A `sqt://price_panel` from fetch_ohlcv_panel has no quality check.
3. **Cross-provider comparison is ratios-only.** compare_data_sources/compare_ratio_frames use `compare_ratio_sources`; two price_panel refs for the same symbol from yfinance and polygon cannot be compared, though describe_reference can describe each.
4. **Universe source.** fetch_ohlcv_panel and fetch_returns_panel need an explicit ticker list; nothing produces one (index constituents, sector membership via `provider.get_ticker_info.sector` which get_capacity_report already reads).
5. **Bundles are checked for PIT, not coverage.** describe/validate_data_bundle report pit_safe and reproduces_history; whether the member frames share entities and dates is not reported.
6. **Tick fetches are single-symbol** (fetch_tick_tape, fetch_quote_panel) with no panel form, unlike OHLCV.
7. **build_continuous_futures_series takes inline chains** and publishes; there is no ref path for contract price series and no consumer of the `_research` series inside the data runtime.

## portfolio

1. **The covariance never becomes a ref.** estimate_covariance returns an N×N matrix inline and optimize_risk_parity / optimize_max_diversification / get_marginal_risk_contribution / run_portfolio_scenarios / get_factor_exposure_budget take it inline again — a 50-name matrix travels through the conversation twice. There is no `sqt://covariance` kind.
2. **Optimizer output cannot be simulated.** run_portfolio_optimization and the optimize_* tools return one weight vector inline; run_portfolio_simulation needs per-date `target_weights` or a `weight_panel` ref, and only construct_weights_from_scores publishes one. No "rolling re-optimisation backtest" (re-estimate covariance and re-solve at each rebalance date).
3. **Weight sensitivity.** estimate_covariance reports conditioning and the optimizer warns, but nothing measures how weights move with the estimation window or shrinkage method — the direct test of the concern the docstrings raise.
4. **Ex-post attribution.** get_portfolio_risk_attribution decomposes ex-ante variance (MCR, PCA, factors); no tool attributes realized P&L over a period by asset or factor. get_exposure_attribution (backtest) is single-asset.
5. **Cost model and schedule are disconnected.** estimate_trade_cost composes nine cost legs; plan_rebalance takes only an impact coefficient; get_capacity_report's ADV is fetched while plan_rebalance and get_liquidity_adjusted_var take ADV dicts inline. No tool prices a rebalance schedule under the full cost model or checks it against capacity.
6. **Factor stress.** run_stress_test shocks by historical replay and run_portfolio_scenarios shocks assets; get_factor_exposure_budget computes factor exposures but no scenario tool shocks factors.
7. **Sizing is single-symbol.** get_position_size is ATR/Kelly for one name; portfolio-level vol targeting exists only as `vol_scaled` inside construct_weights_from_scores.

## microstructure

1. **No cross-symbol ranking.** All 8 estimators take one series; get_liquidity_metrics (portfolio) fetches per ticker but reports only Amihud and CS at the last bar. "Rank this universe by Kyle lambda / VPIN" is N calls.
2. **Estimators return summaries, not series.** estimate_kyle_lambda and estimate_roll_spread compute rolling windows and return median/p25/p75; only the three series tools publish refs. The rolling series cannot reach detect_change_points or get_sharpe_stability-style diagnostics.
3. **Tick-vs-bar reconciliation stops at the spread.** check_spread_proxy compares measured effective spread with Corwin-Schultz; the same check for impact (get_effective_spread_series' impact half vs estimate_kyle_lambda's `impact_of_1pct_adv`) and for flow (classify_trade_direction's `buy_volume_fraction` vs get_order_flow_imbalance's tick-rule fraction) does not exist.
4. **The signed tape has no consumer.** classify_trade_direction publishes a signed `sqt://tick_tape`; get_order_flow_imbalance and estimate_vpin take bar close/volume inline, so the better (Lee-Ready) signing cannot feed the flow estimators.
5. **Implementation shortfall is one order.** get_implementation_shortfall takes one fill list inline; no aggregation over a fill log ref, and no comparison of realized shortfall against estimate_trade_cost's prediction, which is exactly what the tool's description says it is for.
6. **Book-based event detection.** detect_liquidity_events runs CUSUM on trade/quote channels; get_order_book_metrics computes depth, imbalance and OFI but no change detector runs on them.
7. **No event study.** After detect_liquidity_events finds a break time, nothing computes spread/flow/returns around it.

## derivatives

1. **Surface-level tools.** fit_volatility_smile is one expiry; analyze_vol_term_structure is one strike (ATM). No cross-expiry×strike surface, no calendar/butterfly arbitrage check across the smiles that fit_volatility_smile fits individually.
2. **Book-level risk.** get_option_greeks and get_option_risk_scenarios are one option; analyze_option_strategy nets greeks for a few legs at one spot. No aggregation of a positions list into book greeks with a scenario grid.
3. **Implied vs realized.** get_volatility_cone takes inline prices (the runtime never fetches), research's get_volatility_estimators fetches realized vol, and get_expected_move accepts `realized_moves` inline; no tool computes the variance risk premium (IV vs subsequently realized) over history.
4. **Historical delta hedge.** simulate_delta_hedge is Monte Carlo on assumed vols; there is no hedge backtest along an actual price path with actual IV inputs.
5. **Dividends as a schedule.** Every tool takes `dividend_yield`; discrete dividends exist in delta_one (analyze_dividend_points) but not here, and `price_option`'s American branch has no dividend schedule.
6. **Option-implied forward vs futures.** get_implied_forward and delta_one's analyze_cash_futures_basis compute the same forward from two directions; no tool reconciles them.

## delta_one

1. **Nothing fetches or reads refs.** analyze_basis_history, detect_basis_dislocation, analyze_hedge_effectiveness and optimize_replication_basket take pasted series; build_continuous_futures_series (data) publishes exactly the futures series they need as `sqt://price_panel`, and fetch_returns_panel the constituent returns, but no delta_one tool accepts a ref.
2. **Hedge ratio is never estimated.** size_futures_hedge takes `portfolio_beta` and `future_beta` as inputs; analyze_hedge_effectiveness measures ex post; the minimum-variance hedge ratio regression (the delta-one analogue of run_kalman_hedge_ratio / get_rolling_beta in research) has no tool.
3. **Replication is a snapshot.** optimize_replication_basket returns weights; tracking error of that basket over time, its rebalance cost, and a comparison with analyze_etf_fair_value's ETF are all absent.
4. **Curve history.** analyze_futures_curve is one date; no roll-yield or contango time series, so "has the curve regime changed" cannot be asked even though detect_basis_dislocation exists for the basis.
5. **Expression costs are not derived.** compare_delta_one_expressions takes each expression's financing/spread/fee inline; analyze_roll, price_total_return_swap, analyze_total_return_future and analyze_etf_fair_value compute those very inputs but nothing chains them.
6. **Monitor state travels through the conversation.** monitor_spread_stream returns its state blob for the caller to pass back; nothing persists it as a ref, so a long-running monitor is bounded by context.
7. **scan_basis_dislocations results are inline** and per pair; no ranked history or published panel for the pairs it flags.
