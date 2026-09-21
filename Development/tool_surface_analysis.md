# What is implemented, what is exposed, and what to build next

A survey of the library's functions against its 211-tool surface, run on
2026-09-20 as six parallel read-only passes -- one per package slice and one
over every tool body -- and synthesized here. The six reports were the
evidence; every claim below cites the report that made it, and every
count is the survey's count on that date. The reports were removed on
2026-09-21 once this file held their conclusions, and the last commit
that carries them is `e005a3a`
(`git show e005a3a:Development/tool_surface_survey/<report>`).

**Status:** Wave 1 (section 6) landed on 2026-09-20. Waves 2 and 3
(section 7) and the proposals in sections 2-5 are not built, and this
file stays for them.

| Report | Slice | Lines read |
|---|---|---|
| `tool_map.md` | every tool body, mapped to the library functions it calls | 211 tools |
| `analysis.md` | `analysis/` (21 modules) | ~10,850 |
| `modeling.md` | `modeling/` except the tool layer | ~21,000 |
| `backtest_portfolio_screener.md` | `backtest/`, `backtesting/`, `portfolio/`, `screener/` | ~11,000 |
| `data_audit_mcp.md` | `data/`, `audit/`, `mcp/`, top-level modules | ~13,900 |
| `delta_one_indicators_metrics.md` | `delta_one/`, `indicators/`, `metrics/` | ~6,965 |

The rule the survey applied is the repository's own: a tool earns its place
by being a **decision the agent makes, not plumbing**. Every proposal below
names the question it answers, the functions that already answer it, and the
caveats the tool would have to state.

---

## 1. The surface in numbers

- **211 tools across 10 runtimes**: research 42, backtest 35, modeling 22,
  meta 20, data 18, portfolio 18, delta_one 18, microstructure 17,
  derivatives 12, feature_lab 9.
- **119 are single-function wrappers** -- one library call, argument
  conversion and result typing -- and **27 call no numerical function at
  all** (fetch-and-publish, registry reads, catalog introspection). The
  remaining 65 compose several computations. (`tool_map.md`, "Single-function
  wrappers".)
- **Shape is uneven.** Research and backtest fetch by symbol (47 tools);
  modeling and feature_lab address artifacts by id (27); **derivatives and
  delta_one are 100% inline** -- 30 tools, nothing fetches, nothing reads a
  reference; and the `DataSource` union (symbol *or* reference *or* values)
  that solves the shape problem exists on **exactly two tools**
  (`calculate_series_metrics`, `run_terminal_monte_carlo`). (`tool_map.md`,
  "Shape taxonomy".)
- **Exposure by slice.** analysis: 69 of 86 public functions reached
  directly, 14 indirectly, 3 never. modeling: 218 of 235 rows exposed
  directly or through a spec field, 17 not, plus 13 exposed functions with
  an unreachable mode. backtest/portfolio: 74 of 85 reached, 3 never
  (`stitch_oos_returns`, `compute_stitched_metrics`, `VectorizedStrategy`).
  data/audit/mcp: 79 of 293 items reached by a tool, 64 CLI/MCP-only, 21
  unreachable. delta_one/indicators/metrics: 57 of 66 reached, 2 with no
  route (`ContractSpec`, `day_count`).

The library is, by these counts, largely exposed. The gaps are of three
kinds, and they matter more than the counts: capabilities that are computed
and then **discarded** before the result leaves the tool; capabilities that
exist in one **shape** (inline) when the agent holds them in another (a
published reference); and **decisions** that need several exposed pieces
joined, where the joining is today done by the agent copying numbers
between calls.

---

## 2. What is implemented and not exposed

Condensed from the six reports; each item names its report.

**Computed then discarded.**
- Per-trade MAE/MFE (`trade_excursions`) -- only two means leave the tool;
  the distribution that sizes a stop is thrown away. The rolling
  volatility-estimator series -- one scalar surfaced. `trade_expectancy` on
  a persisted trade log -- only by re-running the strategy. (delta_one/metrics)
- `detect_regimes` per-observation labels -- the tool returns counts and
  persistence, so nothing downstream can condition on a regime. GARCH
  conditional-volatility path and standardized residuals. Kalman innovation
  series and the whole hedge-ratio path (only the last value and its std
  leave). Per-snapshot order-book series (microprice, imbalance, OFI) -- the
  docstrings make predictive claims only means can never test. (analysis)
- The experiment plan as a document: per-fold spans, purge counts, inner
  folds, hashes -- `validate_model_spec` surfaces only `n_fits`. Feature
  provenance, implementation hashes, content hashes, the monitoring profile
  -- persisted on every manifest and never shown. `build_dataset`'s
  temporal-bundle verdict -- computed and dropped by the tool. (modeling)
- `run_backtest_optimization` truncates to `top_n` and persists nothing, so
  sensitivity, deflated Sharpe and parameter decay can only be computed by
  re-running the grid inside `get_robustness_diagnostics`. (backtest)

**Exists, but in the wrong shape for the artifacts the surface produces.**
- Nine `backtesting/`-backed judgement tools (deflated Sharpe, reality check,
  regime stratification, overfitting, trade-path Monte Carlo, clustering,
  random comparison, break-even cost, exposure attribution) take
  `List[float]`; `run_backtest_compact` publishes `sqt://equity_curve` and
  `sqt://trade_log`, and the only way across is `read_reference` (capped at
  64 rows) or pasting thousands of numbers. (backtest)
- Thirteen inline research diagnostics have the same problem in the other
  direction; the two `DataSource` tools show the fix. (tool_map)
- `estimate_covariance` returns an N×N matrix inline and five construction
  tools take it inline again; there is no `sqt://covariance` kind. (backtest/portfolio)
- Every data-runtime tool is hard-wired to yfinance: `fetch_tick_tape` and
  `fetch_quote_panel` can never succeed outside a test, because yfinance has
  no tick feed. Databento's `get_order_book` / `get_order_events` and
  Polygon's `get_point_in_time_records` have no fetch tool. (data)

**Reachable only through one narrow path.**
- Package trust -- `sign_manifest`, `verify_model_package(require_signature,
  public_key)`, `load_manifest(require_signature=True)`, mirror and pull --
  is Python-only; `inspect_model(view="lineage")` verifies with defaults, so
  an unsigned package reads `ok=True`, and `promote_model` does not gate on
  verification. (modeling)
- `transform_predictions_to_weights` (gross and net exactly, per-position
  cap, rebalance schedule, uncertainty scaling) and the shared-cash simulator
  are reachable only through a registered `model_id`; an ensemble or an
  external prediction set cannot be sized or simulated. (modeling)
- `oos_predictions_to_signal_panel`'s verified mode (digest check, cpcv
  refusal, skipped-fold refusal) is bypassed by the one tool that reaches it,
  `meta.convert_reference` in URI mode. (modeling)
- `half_life`, `variance_ratio(differencing)`, `kpss_statistic(lags)`,
  `cusum` on an arbitrary series -- each reachable only as a by-product of
  a tool asking a different question. (analysis)
- `ContractSpec` arithmetic and the four day-count conventions -- no route
  at all; the library's own docstring quantifies the stake of a convention
  mismatch and no tool can produce that number. (delta_one)
- The decision-record vocabulary: `explain_decision`, `replay_decision` and
  `compare_decisions` need a `request_id` that no dispatcher, runtime or MCP
  server ever returns to the caller. (data/audit)

**Deliberately unexposed, and correctly so.** `register_feature`,
`register_target`, `register_estimator`, `register_preprocessor`: handing an
LLM an arbitrary callable is the `exec()` path the specs exist to prevent.
Hold/release/gc/seal on the audit trail stay CLI by stated policy. (modeling, data/audit)

---

## 3. New tools worth building

Ranked across all six reports by the value of the decision, the amount of
unexposed capability it joins, and the effort. Effort is the survey's
estimate: S under half a day, M one to two days, L more, each including
models, dispatch, tests and regenerating the tool index (a test fails
otherwise). "Alternative" is where a parameter on an existing tool would do
the same job; the report marks several proposals as better served that way.

| # | Tool | Runtime | The question it answers | Backing already in the library | Effort |
|---|---|---|---|---|---|
| 1 | `assess_backtest_result` | backtest | I have this run's artifacts; should I believe it, and at what size? | `overfitting.{deflated_sharpe_ratio, regime_stratified_performance}`, `robustness.block_bootstrap_ci`, `trade_analysis.{monte_carlo_trade_paths, analyze_trade_clustering, compare_against_random, break_even_cost}`, `artifacts.load_artifact` | M |
| 2 | `analyze_trade_log` | backtest | Where should the stop be, and is my exit leaving money on the table? | `metrics.diagnostics.{trade_expectancy, trade_excursions, exposure_stats}` on a persisted `sqt://trade_log` | M |
| 3 | `list_decisions` (+ return `request_id` from every dispatch) | meta | Which past call do I explain, replay or compare? | `cli._iter_records`, `paths._iter_day_files`, `replay._redacted_input_fields` | S |
| 4 | `validate_dataset_spec` | modeling | What will this spec cost and refuse before I fetch a universe? | `resolve_params`, `resolved_lookback`, `deepest_lag` (both dead today), `check_point_in_time_safety`, `gate_point_in_time`, `provider_guarantee_warnings`, `interval_warnings`, `validate_calendar_name` | M |
| 5 | `plan_model_experiment` (or `validate_model_spec(include_plan=True)`) | modeling | Show me the fold schedule, purge counts and inner folds this spec implies on this dataset | `plan.plan_experiment(...).to_dict()` | S |
| 6 | `compare_predictions` | modeling | Does the ensemble beat its best member on the same rows? Did the model add anything over its best feature? | `validation/comparison.{paired_comparison, holm_adjust, diebold_mariano}`, `ensemble.load_oos_predictions` | M |
| 7 | `build_target_weights` + `evaluate_model_portfolio(predictions_ref=...)` | modeling / meta | Size and simulate predictions that are not a registered model's | `portfolio_eval.{predictions_to_score_panel, transform_predictions_to_weights, scale_by_uncertainty, select_rebalance_dates}` | M |
| 8 | `attest_model_package` (+ `promote_model(require_verified_package=)`) | modeling | Is this package whole and signed by a key I trust, before it is promoted or copied? | `package.{verify_model_package, mirror_model_package, pull_model_package}`, `signing.{sign_manifest, verify_manifest_signature}` | S/M |
| 9 | `assess_mean_reversion` | research | How fast does this spread revert, is that measured or assumed, and what z-window does it justify? | `compute_spread` / `kalman_hedge_ratio`, `half_life`, `adf_statistic`, `kpss_statistic` + `andrews_bandwidth`, `variance_ratio`, `spread_zscore` | M |
| 10 | `detect_level_shift` | research | Did this series' level change, when, which way, and by how much in its own units? | `cusum` on any series | S |
| 11 | `fetch_order_book`, `fetch_order_events`, `fetch_point_in_time_records` | data | Get depth, order events and point-in-time filings into the audit trail and the handoff graph | `DatabentoProvider.{get_order_book, get_order_events}`, `PolygonProvider.get_point_in_time_records`, `normalize_book`, `normalize_mbo`, `validate_pit_frame`, `observed_revisions` | M each |
| 12 | `compare_expressions_from_quotes` | delta_one | Given these live quotes, which is the cheapest way to hold the exposure? | `carry.{solve_carry, observed_carry_rate}`, `swaps.total_return_future`, `expressions.compare_expressions`, put-call parity from `analysis.derivatives` | M |
| 13 | `trace_efficient_frontier` | portfolio | Where on the curve should this mandate sit; what does each unit of risk buy? | `_frontier_weights`, `_solve_constrained`, `estimate_covariance` | M |
| 14 | `run_strategy_portfolio_backtest` | backtest | Does this rule survive as one book -- one cash balance, drift, an ADV cap -- or only as twenty accounts? | `STRATEGY_REGISTRY` signals, `sizing`, `portfolio_engine.run_portfolio_simulation`, `panel.run_signal_panel_backtest` | M |
| 15 | `get_order_book_series` (+ wire the eight declared L2 channels) | microstructure | Does book imbalance or OFI lead the mid on this feed? | `microprice`, `book_metrics`, `book_dynamics` refactored to return series; `test_granger_causality` | M |
| 16 | `compare_tail_risk_models` | research | Which VaR do I size on at 99.5% with 500 days of history? | `var_historical`, `var_parametric`, `cvar`, `evt_tail_risk` on a `DataSource` | S/M |
| 17 | `evaluate_tracking_basket` | delta_one | Is the basket I hold still tracking, and would re-optimizing be worth the turnover? | `hedging.tracking_error`, `information_ratio`, `rolling_beta`, `max_drawdown` on the active curve, `optimize_replication_basket` as reference | S/M |
| 18 | `get_regime_conditioned_stats` | research | Is the edge in the calm regime, the stressed regime, or only on average? | `detect_regimes` labels, `decompose_returns`, `bootstrap_statistic`, `drawdown_profile`, `compare_distributions` | M |
| 19 | `stitch_oos_paths` | backtest | What did the whole out-of-sample record earn, compounded, across paths produced separately? | `walk_forward.{stitch_oos_returns, compute_stitched_metrics, longest_losing_streak, parameter_turnover}` (the two orphaned functions) | S |
| 20 | `describe_audit_trail` | meta | Is the trail being written, what is held, what would `gc` delete? | `paths.*`, `retention.{is_held, gc_candidates}` dry run, checkpoint sidecars | S |
| 21 | `check_garch_fit` (or `run_garch_volatility_forecast(include_diagnostics=)`) | research | Can this GARCH forecast be trusted, or do I fall back to realized vol? | `garch_volatility_forecast` + returning `sigma2`, `ljung_box(squared=True)`, `test_normality` | M |
| 22 | `compare_financing_accrual` | delta_one | Which day-count convention is the counterparty using, and what does a mismatch cost? | `daycount.{day_count, year_fraction, CONVENTIONS}` | S |
| 23 | `diagnose_label_overlap` | feature_lab | What do `WeightingSpec.method` and `half_life_days` do to my training window before I choose them? | `validation/weights.py`, `effective_sample_size`, the plan's purge counts | S/M |
| 24 | `preview_preprocessing` | feature_lab | What does this pipeline's fitted state look like -- PCA variance ratio, winsor bounds -- before a full experiment? | `preprocessing/pipeline.py`, `cache.column_wise_pipeline`, `feature_distribution_stats` | M |
| 25 | `list_run_artifacts`, `read_external_window`, `describe_ratio_definitions`, `list_liquidity_channels`, `describe_pricing_models`, `estimate_strategy_capacity`, `compare_portfolio_cost_models` | meta / data / microstructure / derivatives / portfolio / backtest | Discovery and one-question tools each report argues for on its own terms | see reports | S–M |

Three observations across the table:

- **The highest-value items are joins, not new math.** Numbers 1, 2, 6, 7,
  9, 12 and 14 compute nothing the library does not already compute; they
  put several exposed pieces behind the one question an agent actually asks,
  and stop the agent from carrying floats between calls through the context
  window.
- **The modeling runtime's gaps are at its trust boundaries.** Signing,
  verification, mirroring and pulling shipped in phases 9 and 10 as library
  functions; number 8 is what puts a tool at the boundary the docstrings
  name, and `promote_model` should read it.
- **Three fetch tools close a documented contradiction.** `26_data.md` still
  says no shipped provider serves depth, which stopped being true when the
  Databento provider landed; `describe_temporal_contract` says Polygon's
  revisions are unknown "until `observed_revisions` on a pulled history
  upgrades it", and nothing pulls a history. Number 11 is that pull.

---

## 4. Parameters that beat a new tool

The reports found more leverage in widening existing tools than in adding
ones. Each row is one field.

| Existing tool | Add | Why | Effort |
|---|---|---|---|
| the nine `backtesting/`-backed judgement tools, `run_monte_carlo_simulation` | `DataSource` in place of `List[float]` | closes the artifact-to-judgement gap without any new tool; `run_terminal_monte_carlo` shows the pattern | S each |
| `run_backtest_optimization` | `include_robustness`, or a persisted `grid_ref` | sensitivity, deflated Sharpe and decay on the grid already paid for, before `top_n` truncation | S–M |
| `run_portfolio_optimization` | `covariance_method` ∈ {sample, ledoit_wolf, ewma, ewma_shrunk}, `halflife` | the library says shrinkage is the answer to the optimizer's own warning, then makes it unreachable from the tool that warns | M |
| `run_strategy_matrix`, `run_backtest_compact`, `get_backtest_diagnostics`, `compare_cost_models` | gate on `RUNNABLE`, not `STRATEGY_REGISTRY` | the buy-and-hold baseline is excluded from exactly the comparisons it was moved into the registry for | S |
| `compare_strategies` | `strategies: List[str]` + params | half the registry (donchian, momentum, vwap, adx) is absent from the "which rule?" tool | S |
| `detect_regimes` | result `labels` | without labels the regime cannot condition anything | S |
| `run_stationarity_tests` | `vr_periods`, `kpss_lags`; result `differencing`, `kpss_bandwidth` | the module measured VR(2) at 0.62 vs 0.86 on one series depending on a choice the tool hides | S |
| `run_kalman_hedge_ratio` | `half_life_days`, hedge-ratio start/end/drift, mean Kalman gain | the tool exists to say the static ratio is stale and cannot say how stale | S |
| `run_cointegration_test`, `scan_pairs` | `autolag`, `max_lag`; explicit `pairs` on the scan | lag choice changes p-values; a shortlist turns O(N²) into the agent's list | S |
| `get_technical_analysis` | the indicator periods `get_technical_panel` already takes; `atr_kind` | two tools compute the same indicators with different knobs and, for ATR, different definitions | S |
| `calculate_series_metrics` | `benchmark: DataSource`; `information_ratio`, `treynor_ratio`, `evt_tail_risk`, drawdown episodes, `confidence` | the one `DataSource` tool cannot compute a benchmark-relative metric or an EVT tail | S |
| `get_drawdown_table` | `series: DataSource`; `start_convention` | two tools answer the drawdown question with two start conventions; one with a parameter, not a third | S |
| `score_predictions` | `horizon`, `outcomes_dataset_id`, a survival branch, quantile/interval columns | see the defects below: the effective-sample-size adjustment cannot run and survival is mis-scored | S |
| `inspect_model` | `view="provenance"` with an `implementation_drift` block; `require_signature`/`public_key` on the lineage view | the retrain-or-score-with-old-code decision is made before a scoring call, not learned from its error | S |
| `validate_pit_records`, `build_model_dataset`, `check_leakage` | `observed_revisions` numbers; persist and return the `temporal_bundle` verdict | a single boolean hides how much of a record set is restated | S |
| `list_modeling_capabilities` | `calendar`, `interval` → resolved annualization and the calendar names | naming a venue changes every annualized number by a fixed factor and the agent cannot discover the codes | S |
| every `feature_lab` tool, `analyze_features` | `target: Optional[str]` | a multi-horizon panel's other labels are invisible to the lab | S |
| `meta.convert_reference` | `model_id` for predictions→signal_panel | the one path to the bridge bypasses its verified mode | S |
| `convert_reference` (score_panel→weight_panel) | `equal_weight_top_bottom`, `dollar_neutral` | the sizer the modeling runtime uses by default is not offered | S |
| `detect_liquidity_events` | `slack`; `channels="all_available"` | the sensitivity knob the docstring discusses; no refusal round-trip | S |
| `size_futures_hedge`, `analyze_roll`, `price_total_return_swap` | a `ContractSpec` mapping; `also_report_conventions` | contract economics derived from the spec rather than re-typed; the accrual gap in currency | S |
| the seven fetching data tools | `source` | the single highest-value change in the data runtime: today they cannot choose a provider | S |

---

## 5. Consolidation: the same computation under several names

The tool-map report counted 119 single-function wrappers and grouped the
ones that differ only by a parameter. Merging is a **surface** decision --
every runtime's tool count is pinned in tests, README rows and the generated
index, and a runtime holds at least eight tools on both sides of any split --
so this is listed as an option with its cost, not a recommendation to act on
alone.

| Today | Could be | Saves |
|---|---|---|
| `run_sma_backtest`, `run_rsi_backtest`, `run_macd_backtest`, `run_bollinger_backtest`, `run_buy_and_hold` | `run_backtest(strategy_type=...)` -- all four already call one `_dispatch_backtest` | 4 |
| 13 inline series tests in research (`diagnostic_tools`, `inference_tools`) | `test_series(test=...)` and `profile_series(measure=...)` | ~9 |
| 5 trade-return tools in backtest | `analyze_trades(analysis=...)` | 4 |
| 8 bar estimators in microstructure | `estimate_liquidity(estimator=...)` (`get_liquidity_metrics` is already the fetch-shaped version of two) | 7 |
| `optimize_risk_parity`, `optimize_max_diversification`, `optimize_hierarchical_risk_parity` | `optimize_weights(method=...)` | 2 |
| `get_technical_panel` / `compute_indicator_panel` | one tool with `publish: bool` | 1 |
| `run_monte_carlo_simulation` / `run_terminal_monte_carlo` | `keep_paths: bool` on the `DataSource` one | 1 |
| `compare_ratio_frames` / `compare_data_sources` | one `DataSource`-shaped comparison | 1 |

The other kind of duplication is silent and is worth fixing regardless of
the surface: **the same computation implemented twice in different modules,
called by different tools, with different conventions.** Amihud (row
filtering and default window differ between `backtest.liquidity` and
`analysis.microstructure_estimators`), Corwin-Schultz, intraday volume
profile (two functions with one name), the deflated Sharpe ratio (two
annualization conventions), block bootstrap (four carriers), risk parity,
marginal risk contribution, Black-Scholes greeks, drawdown episodes (three,
with two start conventions), feature drift PSI/KS, and four families of
change detection. Two tools that look like one question in two shapes may
not agree numerically. (`tool_map.md`, "Duplicate implementations"; each
slice report names its own.)

---

## 6. Defects and drift the survey found

Actionable as they stand, independent of any new tool. Report and location
in parentheses. **Status:** all eleven were fixed on 2026-09-20 in the
commit that carries this sentence (Wave 1), with a planted test for each;
item 9 is corrected below, because the survey over-counted.

1. `score_predictions` calls `effective_sample_size` with `horizon=1`, so
   the "effective sample size adjusted for overlapping forward returns" it
   reports equals `n_observations`; and `task="survival"` falls through to
   ranking metrics, so a duration is scored with NDCG. (modeling, §4.6)
2. `hedge_effectiveness`'s `tracking_error` field is arithmetically identical
   to `volatility_after` (`hedging.py:287` vs `:264`); `index_basket`'s
   `missing_symbols` can never be non-empty; `basis_scan`'s docstring
   promises a `multiplier` the code never reads and the schema forbids;
   `reset_spread_monitor` leaves `degenerate_n/mean/m2` uncleared; 24 unused
   `_numbers` imports across `delta_one`. (delta_one, §9)
3. `request_id` is minted by `_run_and_record` and returned to no one, so
   the three provenance tools need an id the agent cannot obtain; the MCP
   server docstring claims a per-call request-id context it does not set.
   (data/audit, findings 2 and 6)
4. `DatabentoProvider` never calls `record_data_access`, and neither do any
   provider's `get_trades`/`get_quotes` or Polygon's PIT records: a decision
   record for such a call has empty `data_sources` and can never replay as
   `data_changed`. (data/audit, finding 4)
5. `audit.replay._resolve_tool` and `meta.validate_tool_call` search the
   eight runtimes plus `MODELING_TOOL_DISPATCH` and not
   `FEATURE_TOOL_DISPATCH`: feature_lab's records are unreplayable and its
   most expensive call cannot be pre-validated. (data/audit, finding 5)
6. `liquidity_events` declares eight order-book channels unavailable
   "because no provider serves a book" while `order_book.py` computes them
   and a provider now serves one; `26_data.md` says the same stale thing.
   (analysis, §4.5; data/audit, finding 3)
7. `scan_cointegrated_pairs` promises `intercept` and `optimal_lag` columns;
   the pure-Python fallback fills NaN and 0. Whether the C++ half-life
   applies the same discrete-AR(1) correction as the Python one is not
   pinned by a test, and the two backends would rank pairs differently if
   not. (analysis, §4.3–4.4)
8. `calculate_beta` returns `r_squared=0.0` when `ss_tot == 0`, against its
   own NaN-not-zero policy; `pca_returns` raises `ValueError` where every
   sibling raises `ValidationError`, so a tool boundary surfaces it raw;
   `multi_factor_regression` has no HAC errors and the two tools built on
   it do not say so. (analysis, §4.11–4.13)
9. Dead code, corrected on inspection: `holdout_split` (no caller, no
   test) and `MIN_OBS_PER_PARAMETER` (exported, never read) were dead and
   are removed, as is the dangling constant comment in
   `microstructure_estimators.py`. The survey also listed
   `parse_lag_column`, `deepest_lag`, `resolved_lookback`,
   `inference._block_indices` and `local_store`; the first four have
   tests or a caller (`comparison.py` uses `_block_indices`) and
   `local_store` is the store's documented entry point, so they stay.
   The three `TRADING_DAYS` re-exports are kept on purpose, by name.
10. Docstrings that promise more than the code: `_buy_and_hold_signals`,
    `resolve_strategy_params`, `engine._run_signal_fn_job` drops
    `risk_free_rate`, a redundant embargo check in
    `combinatorial_purged_cv`. (backtest, §8)
11. "atr" means a simple-mean ATR in `get_technical_analysis` and Wilder's
    ATR in the panel tools, under one name. (delta_one/indicators, §9.5)

---

## 7. Sequencing

Three waves, each independently shippable, each ending with the tool index
regenerated and the pinned counts updated.

**Wave 1 -- fix what is wrong and widen what exists (no new tools).** The
eleven defects above; `DataSource` on the nine judgement tools and
`run_monte_carlo_simulation`; `source` on the data fetch tools;
`request_id` in every dispatch result; `labels` on `detect_regimes`;
`covariance_method` on the optimizer; the `score_predictions` and
`inspect_model` additions; `RUNNABLE` gating. Mostly S, all within the
existing surface, and it removes the largest class of agent friction (floats
carried through the context window).

**Wave 2 -- the joins.** `assess_backtest_result`, `analyze_trade_log`,
`list_decisions`, `validate_dataset_spec`, `plan_model_experiment`,
`compare_predictions`, `build_target_weights` + `evaluate_model_portfolio(predictions_ref)`,
`attest_model_package` with the `promote_model` gate, `assess_mean_reversion`,
`detect_level_shift`, `compare_expressions_from_quotes`,
`trace_efficient_frontier`. Twelve tools, every one a decision that today
takes four to seven calls and hand-copied numbers.

**Wave 3 -- new reach.** The three data fetch tools (depth, order events,
point-in-time records) with `record_data_access` wired for every provider
method; `get_order_book_series` and the L2 channels; `run_strategy_portfolio_backtest`;
the remaining research, delta_one and feature_lab proposals. These add
capability rather than join it, and each needs a provider or a library
refactor (`sigma2` from GARCH, per-snapshot arrays from the book functions).

Consolidation (section 5) is a separate decision: it changes counts that
tests, the README and the MCP budget all pin, and the per-runtime schema
budget is the argument for doing it, not the wrapper count.
