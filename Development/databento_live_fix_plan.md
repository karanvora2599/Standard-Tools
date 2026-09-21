# Fixing what the live market found: the plan

The companion to `databento_live_findings.md`. That document records what
two live passes found and deliberately applied no fix; this one decides
what to change, in what order, how each change is verified without a key,
and what is left alone and why. Findings are cited by their number there
(`D1`..`D20`) or by section.

**Status: phases 1 (the data path), 2 (the deployed model is the
validated model), 3 (selection and inference) and 4 (backtest and
portfolio) are implemented, 2026-09-20; phases 5-7 are the plan.** In
phase 4 the native portfolio kernel keeps refusing a trade over the ADV
cap; the engine catches that refusal and runs the Python loop, which
sizes the trade down, so a capped configuration is correct and merely
slower until the kernel caps too. Phase 3 went one step past its
wording on the purge: besides saying `not_applicable` when no label end
exists, an external panel registered with a horizon and no
`label_end_column` now gets its label end derived from the horizon, so
the purge the two docstrings described actually runs there.
Phase 2 chose the calendar-anchored form of D17 over the end-anchored
one: a refit grid fixed by each bar's date keeps BOTH properties -- a
value is unchanged when leading bars are dropped and when trailing bars
are truncated -- where an end anchor would have traded the second for
the first. Each phase
lands as its own commit with the offline suites green, and each names the
live check the owner should run afterwards with the engine's interpreter
and a key (findings §13), because no live check runs here.

---

## 1. Principles

**A fix that changes a number is a decision, and the CHANGELOG records it
as one.** Several of these change results the library has already
produced -- the daily close, the default feed's volume, the deployed
estimator's parameters. Each such change is named in the CHANGELOG entry
with the direction of the change, so a reader who sees a number move can
find why.

**The stub must be able to lie in the way the vendor does.** D1 hid behind
a stub that returned five rows whatever window was asked. Every provider
fix here comes with a stub that honours the request -- slices to
`[start, end)`, refuses an unfinalized tail on the schema that has one,
answers per dataset -- so the offline test can fail for the reason the
live one did.

**Both directions.** A refusal is tested with the input it refuses and
with the input it must let through. A calibrated test is tested on the
regime it was miscalibrated in AND on i.i.d. noise, where it must still
be calibrated.

**Refuse by name, at the seam.** Where the honest answer is "this library
cannot do that" -- a futures root that is also a ticker, a cross-sectional
model on a universe it never saw, a channel that crashes on real ticks --
the refusal names the input, the reason and the remedy, before any
network call or fit.

**Scope is the finding.** A phase fixes what the findings measured. It
does not redesign the provider, the engine or the estimators around them.
Where a finding names a redesign (the futures engine's roll accounting,
the MBO counters), the phase does the measured fix and records the rest.

---

## 2. Phase 1 -- The data path (D1, D2, D3, D5, D11, D12, plumbing)

The root of most of the rest: the provider that serves the live market is
the one that bypasses every seam the other three providers share.

### 2.1 D1 -- A daily request returns tomorrow

`_to_utc(end_of_day=True)` already pushes a bare end to the next midnight;
the `ohlcv-1d` branch of `_get_range` then adds another day. Databento's
day-granular end is exclusive, so the request asks for one bar past the
inclusive end, and nothing trims.

- `_get_range`: the daily attempt starts at the day boundary the end
  already names (`ceil` to a whole day, no extra day). The walk-back stays.
- `get_ohlcv`: `trim_to_inclusive_end(frame, end_date, interval)` after
  shaping, exactly as polygon, yfinance and bloomberg do. The contract then
  holds by construction on this provider too.
- The 422-per-request cost goes with it: the first attempt no longer asks
  past the edge.
- **Stub:** `_Timeseries.get_range` slices `default` to `[start, end)` on
  its index; a rule can refuse an end past a configurable
  `finalized_through` with the vendor's `available_end` text. The five-row
  assertion becomes "the bars asked for, and none after".

Verify live: `get_ohlcv("AAPL", d, d)` returns one bar dated `d`, close
matching the tape's close on `d`, on a weekday whose next day is a session.

### 2.2 D2 and the seam -- Databento bypasses the normaliser, the cache and the retry layer

Every other provider's frame passes `_normalize_ohlcv_index`; Databento's
does not, which is why its index is tz-aware UTC where every consumer
builds tz-naive, and why `build_dataset` raises at
`coverage.py:193` and again at `targets/builtin.py:135`.

- `get_ohlcv` returns through `_normalize_ohlcv_index(frame, interval)`:
  daily bars keyed by their local trading date (the session date; for the
  US feeds the UTC day boundary the vendor uses is the session), intraday
  converted to UTC and stripped, exactly the normaliser's documented rule.
- `Volume` cast to `int64` at the seam (the `uint64` diff trap: `1.8e19`
  for -1,150,414).
- Defence in depth where the crash surfaced: `entity_coverage_warnings`
  compares tz-normalised timestamps, and `horizon_label_end` allocates its
  NaT series from the index's own dtype rather than assuming naive `ns`.
- The Parquet cache and the retry layer are wired the way yfinance wires
  them: the session cache keyed on `(provider, instance, symbol, start,
  end, interval, dataset)`, the disk cache key including the dataset
  (§9's "the cache key omits the dataset" must land before D12), and the
  audit record still written on a cache hit.
- Round-trip fidelity: the normaliser pins the index to `datetime64[ns]`,
  so a frame read back from Parquet hashes like the one that was written.

Verify live: `build_dataset(DatasetSpec(provider="databento", ...))`
builds; three identical requests make one metered fetch.

### 2.3 D3 -- The default feed is a venue sample, not the tape

`EQUS.MINI` is documented as the consolidated tape and serves 2.3-3.6% of
consolidated volume with an after-hours last print as its close.
`EQUS.SUMMARY` matches the tape exactly but starts 2024-07-01 and carries
only `ohlcv-1d`.

- `_bar_datasets(schema, start)` becomes schema- and range-aware: for
  `ohlcv-1d` inside `EQUS.SUMMARY`'s range, `EQUS.SUMMARY` first; otherwise
  the existing order. The constants say what each feed IS
  (`DATASET_SUMMARY`, `SUMMARY_START`), and the docstring that called
  `EQUS.MINI` "the consolidated tape" says "a sample feed: a fraction of
  consolidated volume and a UTC-day close".
- The served dataset travels with the frame (`frame.attrs["dataset"]`) and
  `get_metadata(symbol, interval)` reports which feed answers a daily
  request today and that the intraday feeds are venue samples. This is the
  data-side half of D16; the modeling half is phase 2.
- **What this does not do:** it does not make intraday consolidated,
  because no such feed exists in the entitlement. The intraday answer is a
  venue feed with its share stated, and the metadata states it.

Verify live: a daily close from `get_ohlcv` on a 2025 date equals the
tape's close; `frame.attrs["dataset"] == "EQUS.SUMMARY"`.

### 2.4 D5 -- A futures root resolves to the equity

`_EQUITY_RE` admits `ES` and `CL` as equity tickers, `GLBX.MDP3` is never
a candidate, and `stype_in` is hard-wired to `raw_symbol`, so a futures
request silently returns Colgate-Palmolive.

- `to_raw_symbol` becomes `resolve_symbol(symbol) -> (raw, stype_in,
  family)`: `ES.c.0` / `ES.c.1` are `continuous` on the futures dataset,
  `ES.FUT` is `parent`, `ESZ6` is a contract `raw_symbol`, an OSI option
  string is `raw_symbol` on the options dataset, and a plain ticker is an
  equity.
- **A bare root that is also an equity ticker is ambiguous and refused**,
  naming both readings and the spelling for each.
- `_fetch` takes the dataset list from the family; `get_metadata` says
  which families the provider reaches and with which entitlement.
- The futures and options datasets are constants with env overrides
  (`DATABENTO_FUTURES_DATASET`, `DATABENTO_OPTIONS_DATASET`).

Verify live: `get_ohlcv("ES.c.0")` returns a four-figure close;
`get_ohlcv("ES")` is refused as ambiguous; `get_ohlcv("CL")` too.

### 2.5 D11 and D12 -- one object, two tapes, and a feed chosen per window

- One preference list per family, shared by bars and ticks, so the trades
  and the minute bars for the same window come from the same feed unless
  the caller pins one.
- The served dataset is reported (2.3), so two adjacent windows served by
  different feeds are visible to the caller, and `build_dataset` records
  it per entity (phase 2).

### 2.6 Also in the data layer

- `data/quality.py` reads `Volume`: zero-volume bars and a per-bar volume
  that is a small fraction of its trailing median are findings.
- `detect_missing_bars` uses the exchange calendar when
  `exchange_calendars` is present (it is a dependency already), so a
  holiday is not a gap; the weekday heuristic stays as the fallback and
  says so.
- `get_temporal_contract` on Databento says `revisions="unknown"` to agree
  with `point_in_time=False` rather than claiming the data is never
  restated.
- `DataSetMetadata.timezone` is defined as the zone the index is
  EXPRESSED in after normalisation (naive session dates for daily; naive
  UTC instants for intraday), and yfinance's report says that rather than
  `America/New_York`.
- Databento's per-session `degraded` flag: deferred -- it needs the
  `statistics` schema and an entitlement check this session cannot run.

---

## 3. Phase 2 -- The deployed model is the validated model (D14, D15, D16, D17, D18, D20)

### 3.1 D14 -- The refit ignores the searched hyperparameters

`fold_params` is reassigned per fold from the inner search and never
reaches the refit, the quantile models or the conformal radius, which all
instantiate from `model_spec.estimator.params`.

- One variable, `deployed_params`, feeds all three call sites.
- With a search: one final inner search on the full panel, under the same
  purge and embargo, chooses `deployed_params`; its fits are counted by
  `plan_experiment` (`n_fits_refit` grows by the candidate count times the
  inner folds) and refused over budget like every other fit. Without
  enough dates for inner folds, the LAST fold's selection is used.
- The manifest records `estimator_params` as the deployed values,
  `deployed_params_source` (`"full_panel_search"` / `"last_fold"` /
  `"spec"`), and the per-fold selections already in `validation_report`.
- `inspect_model(view="summary")` shows `estimator_params`.
- A test pins that the deployed estimator's parameters are in the grid and
  agree with the final search; on the findings' ridge example the deployed
  alpha is one of {0.001, 10000}, never 1.0.

### 3.2 D15 -- A cross-sectional model is refit on the scoring universe

`cross_sectional_standardize` fits nothing, so scoring a subset changes
every row's score; the universe pin fires only for universe-scope
features.

- The manifest records the training cross-section width per date
  (`training_cross_section`: min/median/max entities per date).
- `score_model` refuses a cross-sectional model on a universe that is not
  the trained one unless `universe_policy="allow"` is passed, in the same
  voice as the universe-scope refusal; with `allow`, the result carries a
  `warnings` field saying the transform was refit on the scoring
  cross-section and how far its width is from the training one.
- `ScoreModelResult.warnings` is added (it had none), and the scoring
  docstring stops promising the opposite.

### 3.3 D16 -- The lineage cannot name the feed

- `fetch_universe_ohlcv` keeps each frame's `attrs["dataset"]`;
  `build_dataset` returns `data_sources: {entity: "<provider>:<dataset>"}`
  and the tool persists it into `dataset_meta.json`; `ModelManifest.data_sources`
  carries it forward. It is NOT part of `dataset_spec_hash` (an observation,
  not a request) but IS part of `dataset_hash`'s companion record, so two
  models built from two feeds say so.

### 3.4 D17 -- Universe-scope refits are anchored on the frame's first bar

- `_pca_loading`, `_pca_factor_return`, `_rolling_network` anchor the refit
  schedule on the LAST bar (`end` runs `n, n - refit_every, ...`), so the
  last bar is always a refit and dropping leading bars leaves every value
  at a date unchanged. The property is pinned: recomputing after dropping
  `k` leading bars is bit-identical at every date for every `k`, not only
  multiples of `refit_every`.

### 3.5 D18 -- A column absent from a fold's training rows trains as a constant

- The engine measures, per fold, each feature's missing rate in the
  training rows and records `missing_rate_by_fold` in the validation
  report. A column that is 100% NaN in a fold's training rows is refused
  by name with the fold and the remedy (drop the feature, shorten the
  universe, or use a scheme whose folds cover it), before the impute step
  turns it into a constant.
- `hist_gradient_boosting` on an all-NaN column is therefore refused here
  rather than dying inside numpy.

### 3.6 D20 -- The delisting diagnostic names the wrong symbol

- `intersection_warnings` names the entity whose absence is binding: the
  one that, removed, recovers the most dates -- computed, not inferred
  from the latest start.

### 3.7 Also

- `capabilities()` reports `coefficients` for `sgd`.
- `combine_predictions(method="mean")` across regression and ranking is
  refused by name; `rank_mean` stays the default.
- `preprocessing_stats.json` writes `{"legacy": false, "note": ...}` for a
  pipeline the legacy form cannot express, so `{}` is never read as "no
  preprocessing".

---

## 4. Phase 3 -- Selection and inference (D4, D10, D19, and four CPCV/paired items)

### 4.1 D4 -- `select_features` scores on the whole panel

- `select_features` takes `selection_end` (a date) or `holdout_fraction`
  (default 0.3): redundancy and the IC floor are measured on dates up to
  the cutoff only, and the result reports each selected feature's IC on
  the held-out dates beside the selection IC. The result carries
  `selection_window` and a warning that the holdout IC is the honest
  number.
- The tool input gains the same fields; `run_model_experiment` cannot know
  how its features were chosen, so the selection result is what records
  it.
- Test: the findings' construction -- noise columns, top-5 by full-panel
  IC vs blind -- and the holdout IC of the selected noise features is
  indistinguishable from zero.

### 4.2 D10 -- The permutation null is miscalibrated for autocorrelated features

- The null preserves each entity's serial correlation: a per-entity
  circular shift of the feature's series by a random offset, instead of a
  within-date shuffle. The link to the target is destroyed; the feature's
  autocorrelation and the target's are not.
- `permutation_test_ic` reports `null="circular_shift"` and the lag-1
  autocorrelation of the per-date IC, so a reader sees the regime.
- Test both directions: i.i.d. features stay calibrated (≈5%); an AR(1)
  feature at φ=0.99 against an overlapping 5-bar label rejects near 5%
  where the within-date shuffle rejected several times that.

### 4.3 D19 -- `check_leakage` says a copy of the target is safe

- With a `dataset_id`, the lead-lag screen (`lead_lag_ic_curve`) runs on
  every requested feature and a feature whose contemporaneous IC exceeds
  the screen's threshold is a finding. Without one, `safe` is reported
  with `scope="declared_temporal_support_only"`.

### 4.4 The CPCV and paired items

- `compare_models(method="paired")` refuses a CPCV model by name, as the
  bridge, the portfolio path and the ensemble do.
- The purge report says `purge: "not_applicable"` when the panel has no
  `label_end_date`, and `0` only when it ran; the two docstrings that
  claim a horizon-based purge are corrected.
- A CPCV fold record names its test BLOCKS (start, end per block), not
  one span containing training dates.
- `paired_comparison` reports `n_ties` and computes `hit_rate` over
  decided days, so two identical models read as a tie.

---

## 5. Phase 4 -- Backtest and portfolio (D6 and the liquidity surface)

- **D6.** `run_strategy` screens bar-to-bar returns for `|r| > 0.35` and
  warns with the dates and the provider's `adjusted` flag when it is
  known; the same warning reaches every tool built on it.
- `run_strategy` returns `turnover` and `realized_cost`, both already
  computed and dropped.
- A custom `strategy=` callable's signal is range-checked to `[-1, 1]`.
- `max_adv_participation` sizes a trade DOWN to the cap and records the
  shortfall per rebalance, rather than raising.
- Every optimiser that takes a covariance checks it is PSD and repairs
  (nearest PSD, eigenvalue floor) with a warning naming the smallest
  eigenvalue; `estimate_covariance` warns how many rows a short history
  removed.
- `build_portfolio` refuses NaN or fills under a named policy.
- HRP sorts columns before clustering so the result is order-invariant.
- `plan_rebalance` names an entity missing from `adv` rather than treating
  it as untradeable.
- The screener takes a provider (with the data runtime's `source`
  parameter, which is the Wave 1 widening the survey named).

---

## 6. Phase 5 -- Options and futures (D9, D13)

- **D9.** The IV solver's convergence test is on vol, not on price: a step
  is taken before convergence is declared, the tolerance is
  `|Δsigma| < tol_sigma` or a price tolerance scaled by vega, and the
  bisection fallback the docstring promises is used when vega is below a
  floor. Test: the findings' four cases return the true vol, and the
  700-case grid has no case off by more than 1e-4 that reports converged.
- **D13.** The no-arbitrage bound admits equality within a tolerance, and
  a price the pricer underflowed to 0.0 is refused with that reason.
- `fit_volatility_smile` names the field `moneyness`.
- `analyze_strategy` scans the loss from spot 0 to twice the highest
  strike so max_loss is the true worst case and "unbounded" is reserved
  for a genuinely unbounded side.
- `price_option(model="bachelier", dividend_yield=...)` refuses the
  dividend it cannot use.
- `roll_analysis` refuses `spread_ticks` without `tick_value`.
- `rehedge="drift"` measures the residual of the HELD hedge.
- The three carry decompositions and the futures engine's roll accounting
  and margin fill are investigated in this phase and fixed where the cause
  is local; where it is not, the finding is recorded with the measurement.

---

## 7. Phase 6 -- Microstructure (D7, D8)

- **D7.** Duplicate timestamps: the three sites align by position
  (`searchsorted` / `merge_asof` on a reset index) instead of `.loc` on a
  label index; `_signed_volume` cannot grow past its input;
  `detect_liquidity_events` isolates every channel's failure, not only
  `ValidationError`, and reports the channel that failed.
- **D8.** `kyle_lambda` takes signed volume from a trade classification
  when trades and quotes are supplied (the Lee-Ready path that is 99.7%
  accurate), and with bars only it says in its result that the sign is the
  return's own sign, that lambda is then positive by construction, and
  that `r2` measures nothing -- and it returns `circular=True`. The
  findings' shuffle control is the test: with signed volume, destroying
  the relationship drives lambda to zero.
- `intraday_volume_profile` buckets the regular session when the index
  carries a timezone, and reports the extended-hours share separately.
- `estimate_vpin` drops the residue bucket.
- `roll_spread`'s significance guard runs in the windowed branch.
- The CUSUM threshold: the detector reports the channel's lag-1
  autocorrelation and the false-alarm rate that implies under an AR(1)
  null, and takes `threshold` from a block-calibrated table when asked.
- MBO: `cancel_to_trade` counts a trade once (T and F are one fill),
  fills are not cancellations, `terminated_without_an_add` respects a
  snapshot-bearing window, `events_per_second` excludes snapshots, and
  `queue_positions` is compared against `mbp-10` in a test using the
  findings' sequence numbers.

---

## 8. Phase 7 -- Plumbing

- The disk cache's dead generation: `sqt cache gc` removes files whose
  format version is not current; nothing else is evicted.
- The session cache does not serve an unsettled bar as final: the
  historical guard applies to it too, and the disk guard compares against
  the UTC date.
- One containment check, used by all four call sites, with the Windows
  `\\?\` prefix handled once.
- The data runtime's tools take `source`; the error message for tick data
  names the providers that serve it.
- `describe_data_capabilities` reports an unconfigured Databento as
  `available=False` with the reason.

---

## 9. What is not changed, and why

- **EQUS.MINI is not removed.** It is the only sub-daily feed with the
  consolidated symbol set, and the honest treatment is to name what it is.
- **The default provider stays yfinance.** The findings are about the
  Databento path; changing the default would change every number for
  every user who never asked for it.
- **No estimator is re-tuned.** The refit change (D14) makes the deployed
  model the validated one; it does not change how validation selects.
- **The MBO counters are corrected, not redesigned.** A queue-position
  model needs a book replay this library does not have.

---

## 10. Verification

Offline, every phase: the modeling, data, surface, docs, agent and MCP
suites, then the remaining directories, green before the commit. The stub
client honours windows and finalization from phase 1 on, so the daily
off-by-one and its cousins are reproducible without a key.

Live, per phase, by the owner (findings §13 has the command):

| phase | check |
|---|---|
| 1 | daily close on `d` equals the tape's; `attrs["dataset"]`; `build_dataset(provider="databento")` builds; `ES.c.0` prices, `ES` refused |
| 2 | a searched model's `estimator_params` are in the grid; scoring a subset universe on a cross-sectional model is refused |
| 3 | `permutation_test_ic` on `market.momentum` gives p near the block-bootstrap value |
| 4 | LRCX buy-and-hold across its split carries the split warning |
| 5 | the four IV cases return the true vol |
| 6 | `microstructure_summary` on a live AAPL minute returns |
