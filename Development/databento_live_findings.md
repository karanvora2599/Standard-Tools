# Databento against the live market: what two passes found

A record of the live testing this library has had against Databento, the
defects it found, the claims it confirmed, and what each fix costs.

**Status.** Two passes. The first (2026-09-20, commit `941a730`) added
`tests/data/test_databento_live.py` and
`tests/data/test_databento_pipeline_live.py` — 47 tests, 46 passing and one
strict xfail. The second, later the same day, ran **twelve parallel
investigations** over the areas the first pass named as uncovered, and is
the bulk of this document. Four more regression tests landed as `34899bb`;
the live suite is now **29 passed, 5 xfailed**. Offline suite unchanged at
**7,267 passed** with the same two pre-existing failures (section 11).

**No fix is applied.** Several of these change numbers the library has
already produced, and that is a decision to take deliberately rather than
inside a test commit.

**Provenance.** Findings marked ✅ I reproduced personally against the live
feed, in this session, before writing them down. The rest come from the
parallel investigations, each of which reports its own repro; they are
recorded as measured but not independently re-run by me.

---

## 1. The one-paragraph version

The **arithmetic in this library is overwhelmingly right.** Optimisers hit
their optima under 900,000 perturbations, covariance estimators match
Ledoit–Wolf to 1e-17, Black-Scholes matches a scipy reference to 1e-13,
greeks match finite differences, the purge and embargo remove exactly the
rows they should, the continuous-futures stitching matches an independent
vendor construction to 3 bp over a year, and the Lee-Ready trade classifier
is **99.7% accurate against the venue's own aggressor flag**.

**What is wrong is the data path and the seams.** The default feed is not
the tape it is documented to be, in price as well as in volume. A daily
request returns tomorrow. A futures ticker returns an equity. The modeling
runtime cannot ingest the live provider at all. And several estimators are
either circular by construction or crash on any real tick tape.

---

## 2. Corrections to the first pass

Two things the first pass got wrong, both found by the second.

**F1 said EQUS.MINI carried "correct prices". It does not.** That claim was
based on comparing close *levels*, where the median error is about a tenth
of a percent and looks like a last-print difference. It is not. EQUS.MINI's
daily bar spans the UTC day and its close is the **last print in that day**,
which on a busy afternoon is an after-hours trade. Three investigations
converged on this independently:

| measurement | EQUS.MINI vs the consolidated tape |
|---|---|
| ✅ AAPL 2025-04-02 close (tariffs, after the bell) | **208.00** vs **223.89** — 7.1% |
| ✅ AAPL 2026-07-30 close | 312.75 vs 333.43 — 6.2% |
| ✅ daily-return RMS error, AAPL, 249 sessions | **74.0 bp** (max 647 bp) |
| ✅ same, MSFT / PLUG / AMC | 117.9 / **287.2** / 123.1 bp |
| close deviation, 513 sessions, 12 names | median 13 bp, p95 31–161 bp, **max 1,308 bp** |
| sessions deviating >25 bp | NVDA **280/513**, TSLA 278/513, AMZN 159/513 |
| annualised vol error | −5.7% to +5.4% relative |
| beta vs SPY | wrong by 0.7–12.7%; **JNJ's beta flips sign** (+0.046 → −0.016) |

74 bp RMS is roughly half of AAPL's daily volatility. The level error hides;
the return error is what every downstream number is built on.

**The fix is not a rename.** ✅ EQUS.SUMMARY matches the tape exactly — 1.0000
volume ratio and 0.000000 close error on every symbol tried — but it **only
starts 2024-07-01**, where EQUS.MINI reaches back to 2023-03-28 and
XNAS.ITCH to 2018-05-01. It also carries only `ohlcv-1d`, `definition` and
`statistics`. So `_bar_datasets` has to become schema- and range-aware, and
intraday needs a separate answer where the honest one is a venue feed with
its share stated rather than a sample feed presented as a tape.

---

## 3. What to fix first

Ordered by how badly a wrong answer propagates, not by how hard it is.

| # | finding | area | why it is first |
|---|---|---|---|
| D1 | ✅ A daily request returns tomorrow | data | lookahead into every as-of query |
| D2 | ✅ Modeling cannot ingest the live provider | modeling | hard crash, no workaround |
| D3 | ✅ EQUS.MINI is not the tape, in price or volume | data | silently wrong everywhere |
| D4 | Feature selection scores on the whole panel | modeling | manufactures 70% of a real headline from noise |
| D14 | ✅ The deployed model is not the one that was validated | modeling | the reported metrics describe a model you cannot obtain |
| D5 | ✅ A futures ticker returns an equity | data | silently the wrong instrument |
| D6 | ✅ Splits compound as returns | backtest | −62% reported against +276% |
| D7 | ✅ Three estimators crash on any real tick tape | microstructure | the feed they are for |
| D8 | ✅ Kyle's lambda is circular | microstructure | survives destroying the relationship |
| D9 | ✅ The IV solver returns its own guess, "converged" | options | silently wrong vol |
| D10 | Permutation IC test rejects a true null 27–35% | modeling | both "significant" features are noise |

---

## 4. The data layer

### D1. A `1d` request returns one bar more than it asked for, and that bar is the future — DEFECT ✅

`data/databento_provider.py:347-348`. `_to_utc(end_of_day=True)` has
already pushed a bare end date to next-midnight, which is how the inclusive
contract is honoured. The `ohlcv-1d` branch then adds another day, and
Databento's day-granular end is exclusive. Nothing trims: `get_ohlcv` never
calls `trim_to_inclusive_end`, which polygon (`:671`), yfinance (`:271`)
and bloomberg (`:378`) all do.

```
asked 2026-09-16..2026-09-16  -> 2 rows, last index 2026-09-17, last Close 337.00
asked 2026-09-14..2026-09-16  -> 4 rows, last index 2026-09-17
asked 2026-09-10..2026-09-16  -> 6 rows, last index 2026-09-17
asked 2026-09-18..2026-09-18  -> 1 row   (only because 09-19 is a Saturday)
```

The correct close as of 2026-09-16 is **332.85**; 337.00 is the 17th. Any
as-of query, walk-forward, or signal-at-close path reads tomorrow. ✅ The
intraday schemas take the other branch and are exactly right, which
localises it.

It also costs a **guaranteed-to-fail 422 on every daily request** — the
first attempt asks two days past the edge, is refused, and walks back,
burning a round trip and one of six finalization attempts every time.

**Why no test caught it:** `tests/data/test_databento_provider.py:292` and
`:260` assert against a stub whose `_bars()` returns a fixed five-row frame
*regardless of the requested window*, so `assert len(frame) == 5` asserts
the stub's own length.

**Fix:** round the end up to a whole day instead of adding one, and call
`trim_to_inclusive_end`. Then make the stub slice to `[start, end)`.

### D3. The default feed is a venue sample presented as the consolidated tape — DEFECT ✅

Covered in section 2. `data/databento.py:84` sets
`DATASET_CONSOLIDATED = "EQUS.MINI"` and `:88-90` states "EQUS.MINI is the
consolidated tape". Volume is 2.3–3.6% of consolidated; the close is an
after-hours print.

✅ **The Carbon engine has the same defect from the same cause** — it pins
`DATABENTO_OHLCV_DATASET=EQUS.MINI`; measured through its own bars provider,
volume 0.0315 of consolidated.

### D5. A futures root that is also an equity ticker silently returns the equity — DEFECT ✅

`data/databento_provider.py:88` (`_EQUITY_RE`), `:259-280`, enforced at
`:374`. GLBX.MDP3 is never in the candidate list and `stype_in` is hardwired
to `raw_symbol`, so `continuous` and `parent` symbology are unreachable.

```
get_ohlcv("ES") -> 4 rows, close 67.22    (CME had ES at ~7725 that session)
get_ohlcv("CL") -> 4 rows, close 87.01    (Colgate-Palmolive, not crude)
get_ohlcv("GC") -> APIError naming three EQUITY datasets
'ES.c.0' / 'ESZ6' / 'ES.FUT' -> ValidationError
```

No warning, no error, a four-figure-cheaper instrument. The same door is
shut for OPRA: every option spelling is refused before any network call, so
**the library cannot reach options or futures at all**, and `get_metadata`
does not say so.

**Fix:** recognise the futures and OSI shapes and route them to the right
dataset and `stype_in`; **refuse bare roots that are also equity tickers as
ambiguous** rather than resolving them to the equity.

### D11. One provider object serves bars and trades from different tapes — DEFECT

`databento_provider.py:250` (bars: `EQUS.MINI, XNAS.BASIC, XNAS.ITCH`) vs
`:516` (trades: `XNAS.ITCH, XNAS.BASIC`). Measured, same symbol, same
five minutes, one object: trades sum **65,579** shares against the minute
bars' **20,530** — 3.19x, mismatching in every minute. On a single dataset
the reconciliation is **exact to the share, per minute, on both feeds**, so
this is not trade-condition filtering; it is two tapes.

### D12. The serving dataset is chosen per window and then discarded — DEFECT

`:424, 484, 505, 545, 583` all do `frame, _dataset = self._fetch(...)`.
`get_metadata` (`:619-643`) is a static self-report with no dataset field.
Two adjacent windows get different feeds (✅ measured: `2023-02-01..03-24`
served by XNAS.ITCH, `2023-03-29..05-31` by EQUS.MINI) whose volumes differ
tenfold, and the caller cannot tell. A backtest that fetches per year
splices two feeds and inherits a structural break the market never had.

### Also in the data layer

- **`data/quality.py` never reads Volume** (`grep` → 0 hits), so it gives
  identical verdicts on a 3%-of-volume frame and the real tape.
- **`data/comparison.py` compares fundamentals only.** Run live against
  Databento it returns `n_entities_compared: 0, warnings: []` — which reads
  as "checked, nothing found" while the caller holds 3% of the tape.
- **`detect_missing_bars` is 100% false positives** on live US equity data:
  21 flagged over 500 sessions, **0 real**, all of them genuine holidays.
  The stated reason (avoiding a calendar dependency) is obsolete —
  `exchange_calendars` is already a dependency at `modeling/calendar.py:47`.
- **Databento's own per-session quality flag is never read.** 15 sessions
  marked `degraded` by the vendor were served unmarked.
- **Following `DataSetMetadata.timezone` yields zero aligned rows.**
  yfinance declares `America/New_York` and returns a **tz-naive** index;
  doing what the metadata says gives 0 of 34 joined rows, and ignoring it
  gives 34.
- **`Volume` is `uint64` on Databento and `int64` on yfinance**, so
  `Volume.diff()` returns `1.8446744e19` instead of −1,150,414. Silent,
  finite, correct dtype.
- **`get_temporal_contract` says the data is never restated** while
  `get_metadata` on the same object says `point_in_time=False`.
- **`survivorship_free=True` is VERIFIED SOUND** — five real delisted names
  (ATVI, SGEN, HZNP, VMW, SAVE) return full history at prices matching their
  deal levels, where yfinance returns `DataNotFoundError` for all five.

---

## 5. Modeling

The feature the owner considers most underrated, and the one with both the
hardest blocker and the subtlest leak.

### D2. The modeling runtime cannot build a dataset from the live provider — DEFECT ✅

`modeling/dataset/coverage.py:193`:

```
build_dataset(DatasetSpec(provider="databento", ...))
  -> TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects
     at  if actual_start - requested_start > tolerance:
build_dataset(DatasetSpec(provider="yfinance", ...))   -> builds
```

`union_dates` comes from the provider index, which Databento returns
**tz-aware UTC**; `pd.Timestamp(spec.start)` is naive. Every modeling
dataset on the live provider raises before returning anything.

**Root cause is shared with three other findings.** Databento is the only
provider that touches neither `_cache` nor `_retry` (`grep` → 0 hits; the
other three import both), so it never passes `_normalize_ohlcv_index`, the
single choke point that makes every other provider tz-naive. That one gap
produces this crash, the silent `reindex` → all-NaN, and the metadata
timezone trap above.

**Fix:** normalise both sides in `entity_coverage_warnings`, and better,
route Databento through the normaliser at the provider seam.

### D4. `select_features` scores on the whole panel, holdout included — HAZARD

`modeling/analysis/feature_selection.py:43-141`. `min_abs_rank_ic` and
`max_features` filter on target correlation measured over **every** date,
including the ones `run_model_experiment` will later hold out.
`SelectFeaturesResult` carries no warning and `validation_report` has no
field recording that its features were chosen this way.

Measured: 60 columns of pure i.i.d. noise added to the live panel, top 5
selected by full-panel IC, then the same walk-forward run on the selected
five and on five chosen blind:

| | selected | blind |
|---|---|---|
| mean OOS `cs_rank_ic_mean` over 5 seeds | **+0.04510** | **+0.00163** |
| seeds where selected beat blind | **5 of 5** | |

The real four-feature price model scores **+0.0635** on the same folds. So
**about 70% of the real model's headline is manufacturable from pure noise**
by following the library's own documented workflow — and the run reports
`n_train_rows_purged_overlap: 280` and looks perfectly disciplined. The
engine's split is not at fault; the leak is upstream of it and nothing
records it.

### D10. `permutation_test_ic` rejects a true null 27–35% of the time — DEFECT

`modeling/analysis/feature_stability.py:371`, null at `:310`. It shuffles
within each date, which destroys the cross-sectional link **and** the
feature's serial correlation, so the null's per-date ICs are independent
while the observed ones are not.

| null feature | on the live panel, real h=5 label | ✅ my replication, i.i.d. target |
|---|---|---|
| i.i.d. (the control) | 3.3% ✔ | 1.7% ✔ |
| AR(1) φ=0.95 | **27.5%** | 5.0% |
| AR(1) φ=0.99 | **35.0%** | **11.7%** |

✅ I replicated this independently and got the same direction at a milder
magnitude. The gap is instructive rather than a disagreement: my
construction uses an i.i.d. target, while the investigation used the real
overlapping five-bar label, which correlates consecutive per-date ICs on
the *target* side as well and compounds the effect. The realistic setup is
theirs, so 11.7% is a floor on the error, not a ceiling.

Every real feature in the live panel sits in that regime (per-date IC lag-1
autocorrelation +0.60, +0.62, +0.63, +0.21). Both features the tool calls
significant on real prices are inside the noise once that is accounted for:
`market.momentum` p=0.0200 against a block-bootstrap 0.2145, `technical.rsi`
p=0.0020 against 0.0745 — a 10x and a 37x error.

The suite does not catch it because its calibration test draws i.i.d. noise,
the one regime where the test *is* calibrated.

**The fix is already in-tree:** `validation/comparison.py` uses block
resampling and was verified correctly sized (6.7% at φ=0, 5.8% at φ=0.9).

### Also in modeling

- **`compare_models(method="paired")` accepts a CPCV model** and joins on a
  25x cartesian product (19,520 rows for 3,904 honest ones), producing a
  "significant" p=0.0130. Every other consumer refuses CPCV by name —
  `bridge.py:81`, `portfolio_eval.py:657`, `ensemble.py:233` — this one path
  does not.
- **The outer report cannot distinguish "nothing overlapped" from "the purge
  never ran".** Without a `label_end_date` column the purge is a no-op and
  writes `n_train_rows_purged_overlap: 0`, the same value a clean run gives.
  Measured: 280 training rows whose label lands inside the test window,
  reported as 0. Two docstrings claim a horizon-based purge that does not
  exist.
- **A CPCV fold record names a contiguous test window containing training
  dates** — fold 1 spans 1,912 rows against `n_test_rows: 1,304`.
- **`paired_comparison` reports `hit_rate = 0.000` for two identical
  models**, which reads as "A won every day" when the truth is a tie.

### D14. The full-panel refit ignores the hyperparameters the search selected — DEFECT ✅

`modeling/engine.py:1214-1219`, and the same at `:1265` (quantile models) and
`:1273` (the conformal radius).

✅ Confirmed at the source. Each fold sets `fold_params = model_spec.estimator.params`
and then **reassigns it** from `search_best_params` when a search exists
(`:866`). The refit instantiates from `model_spec.estimator.params` — the
**base** values. `fold_params` never reaches it. `save_model` then writes
those base values into the manifest, so the only recorded description of the
deployed estimator is a configuration the search may never have scored.

| spec | selected per fold | deployed |
|---|---|---|
| `random_forest`, grid `max_depth [6, 8]` | 8,8,8,8,6,8,8 | **max_depth 1** (the base) |
| `ridge`, grid `alpha [0.001, 100, 10000]` | 0.001 ×4, 10000 ×3 | **alpha 1.0**, which is **not in the grid** |

The forest's deployed predictions correlate with the correctly-refitted ones
at Spearman **0.2971**. The ridge coefficients differ by a factor of 4.7. The
reported `cs_rank_ic_mean` describes folds fitted at α ∈ {0.001, 10000}; the
artifact you can actually score is α = 1.0.

Nothing warns. `inspect_model(view="summary")` does not even return
`estimator_params`, and no test asserts the deployed estimator carries the
searched values.

This is the rule the same file states one field over for weighting
(`:1246-1254`, "weighted the same way the folds were") and for preprocessing
(`:1184-1196`) — and it is the plan's own constraint that "the deployed
pipeline is the validated pipeline".

**Fix:** carry a deliberate choice into the refit and record it — the last
fold's `best_params` with a `deployed_params_source` field, or one final
inner search on the whole panel. Structurally, make the three call sites read
one variable, and refuse to deploy a configuration no fold ever scored.

### D15. A cross-sectional model's deployed transform is refit on the scoring universe — DEFECT

`modeling/scoring.py:356-372`, against the module docstring at `:9-14`, which
promises the opposite: that it applies registered stats "not freshly fit
stats on the scoring universe — otherwise the same input row could score
differently depending on which other tickers happened to be in the scoring
call."

For a `cross_sectional` model that is false by construction, because
`cross_sectional_standardize` fits nothing and standardises within whatever
rows the call contains. Measured through the real `score_model`, narrowing
the universe from the trained 8 names to 3:

| model | worst rank agreement vs the full universe | largest move of one row |
|---|---|---|
| ridge, cross-sectional | Spearman **−0.500** | 284% |
| random forest, cross-sectional | Spearman **−1.000** (fully inverted) | 544% |
| pooled sibling, same call | invariant (4.3e-19) | — |

The universe-pin guard at `:246-263` only fires for `FeatureScope.UNIVERSE`
features, and `ScoreModelResult` has no `warnings` field at all. This is the
same failure as the plan's F1, arriving through the universe rather than
through the code path.

**Fix:** record the training cross-section width per date, and refuse — or
at minimum warn — when a cross-sectional model is scored on a universe that
is not the trained one, in the same voice as the existing refusal at `:252`.
And correct the docstring.

### D16. The lineage cannot name the feed the prices came from — DEFECT

`specs.py:463-482`; `DatasetSpec.provider` is the only source field. The
provider picks a dataset at fetch time and records it only into an audit
decision record that the modeling build never opens. Searching the manifest,
`dataset_meta.json` and `dataset_spec.json` for every dataset name: all
absent.

The same spec built twice, pinned to two different datasets:

```
              dataset_spec_hash        dataset_hash        cs_rank_ic
EQUS.MINI     99c83bd0...(identical)   f159a045e098f4e1    0.04583
XNAS.ITCH     99c83bd0...(identical)   264ee0b6959cefbe    0.05583
```

Two models with **identical recorded spec identity**, 22% apart on the
headline metric and about 20% apart on every coefficient. `dataset_hash`
proves they differ but cannot say why, and nothing in the package can
reproduce either.

**Fix:** carry the dataset the provider already knows into `build_dataset`'s
result and persist it as `ModelManifest.data_source`, per entity since the
fallback is per symbol. Keep it out of `dataset_spec_hash` — it is an
observation, not a request.

### Also in the modeling lifecycle

- **`capabilities()` says `sgd` has no coefficients** while the run reports
  signed coefficient importance for it with perfect sign consistency. The
  direction is under-promise, so nothing breaks, but an agent choosing an
  interpretable model is told wrong.
- **`combine_predictions(method="mean")` across regression and ranking is
  the ranker**, silently: standard deviations differ 37x, the blend
  correlates with the ranker at 0.9996 and the regressor at 0.4509, and
  `warnings` is empty. The module docstring explains exactly why this is
  wrong. `rank_mean` is the default and is the mitigation.
- **`preprocessing_stats.json` is `{}`** for every pipeline the legacy form
  cannot express, and `apply_preprocessing(X, {})` is the identity. The live
  path never reaches it.
- **Rewriting `manifest.json` alone rewrites a model's recorded track
  record** — no self-hash, and `inspect_model` reads `oos_metrics` straight
  out of it. The remedy exists and works: signing catches it.

### Modeling: claimed fixes verified as HOLDING

This is the good news, and it is substantial. The plan's claims were tested
against real prices rather than taken on trust:

- **The inner search purges and embargoes.** 66 inner folds at outer embargo
  0/5/10: **zero** training rows whose label ends inside the inner test
  window, every fold. Counts are arithmetically right, not accidentally
  zero (40 purged = 5 dates × 8 entities for h=5).
- **CPCV is correct.** Across six (n,k,embargo) settings: folds = C(n,k)
  exactly, all test sets distinct, train ∩ test empty, every block appearing
  in exactly C(n−1,k−1) test sets, **zero** per-block purge violations, no
  training position within the embargo of a block boundary.
- **The path distribution reproduces exactly** by independent enumeration,
  and correctly reports 14 of 15 when a path is skipped.
- **Paired comparison is genuinely paired and correctly sized.** Model
  against itself → p=1.0000; leaky vs noise → p=0.0010; Newey-West matches
  statsmodels identically; `holm_adjust` reproduces R's `p.adjust`.
- **Validation detects overfitting.** Leaky feature → IC +0.998; four noise
  features → −0.019; real features → +0.064.
- **Every metric recomputed independently** from persisted OOS predictions
  agrees to |Δ| = 0.00e+00.
- **Determinism holds**: same spec and seed → identical folds, identical
  `node_hash`, max prediction difference 0.0.

From the lifecycle investigation, on the same live panel:

- **The plan's F1 — "validated cross-sectionally, deployed pooled" — holds
  exactly.** The fold transform recomputed independently versus the one
  `score_model` applies: **max |ΔX| = 0.000e+00**, **max |Δpred| = 0.000e+00**,
  Spearman 1.000000. And the measurement discriminates: the pre-fix pooled
  deployment reproduces the plan's own numbers on this panel (Spearman 0.8372
  against the plan's 0.84).
- **F2 — ranking models score.** A registered ranker with no `predict_proba`
  trained, registered and scored cleanly through `RankingAdapter`.
- **F8 — ensembles.** All four methods match an independent recomputation at
  **max |d| = 0.0**.
- **Importance is labelled by the pipeline's output columns**, verified on a
  PCA pipeline where the keys are `pc1..pc3` rather than the dataset's four.
- **Round-trip fidelity is exact, not float-noise**: persisted state
  byte-identical, max |Δpred| = 0.0.
- **Determinism is exact**: two runs, nine content hashes equal including
  `model.joblib`'s bytes.
- **Artifact integrity holds on every artifact tampered**, and `model.joblib`
  is refused *before* `joblib.load`, so no pickle executes. Signing catches a
  post-signature manifest edit.
- **The distributional lifecycle reconciles**: independently recomputed
  coverage 0.857 and 0.818 against reported 0.85714 and 0.81786.
- **Degenerate models report honestly** — collapsed estimators report an IC of
  zero rather than a flattering number.

---

## 6. The backtest engine

### D6. Splits are compounded as returns, silently — DEFECT ✅

Nothing under `backtest/` reads the `adjusted` flag (`grep` finds only a
local variable in `sizing.py`), though the provider reports `adjusted=False`
and documents that a split is a real −50% bar.

| symbol | split | reported buy-and-hold | true | warnings |
|---|---|---|---|---|
| ✅ LRCX | 10:1 | **−62.40%** | **+276.04%** | fill_price only |
| ✅ ORLY | 15:1 | −92.56% | +11.67% | fill_price only |
| ✅ NFLX | 10:1 | −89.34% | +6.58% | fill_price only |
| IBKR | 4:1 | −28.83% | +184.68% | fill_price only |

A short held through a split prints a fictitious **+93%** profit. ✅ Control:
on a non-split name the engine's arithmetic is right.

**Fix:** screen bar returns for |r| > 0.35 and warn. The engine already
walks every bar for its total-loss guard, so the pass is free.

### The liquidity surface, poisoned by D3's volume

With **identical prices** and only `Volume` swapped between EQUS.MINI and
the real tape:

| measure | error |
|---|---|
| ADV participation | **14–33x** overstated |
| capacity at a 5% cap | $5.8m allowed vs **$81.7m** real — 14x understated |
| `capacity_report` max account | $4.1m vs **$99.6m** — 24x |
| `days_to_liquidate` | 20–37x overstated |
| impact-model drag at $100m | 0.393pp vs 0.089pp — 4.4x |
| `liquidity_adjusted_var` | 3.3x VaR, 5.5x cost, **10 fabricated "cannot exit"** warnings |
| a $1bn ADV screen | keeps **1 of 12** names; the real tape keeps **12** |

### Also in the backtest and portfolio surface

- **`max_adv_participation` is a kill switch, not a cap** — it raises rather
  than sizing the trade down, so the two states are "no effect" and "no
  result"; a capacity study cannot be expressed.
- **`run_strategy` returns no turnover and no realised cost**, both of which
  are computed at `engine.py:604-605` and thrown away.
- **A custom `strategy=` callable is never range-checked**, so a signal of
  2.0 silently runs a levered book.
- **Nothing checks PSD** between a caller's covariance and the optimisers.
  A ragged real panel through `cov(min_periods=30)` gives min eigenvalue
  −2.77e-03; `max_diversification` then returns a **negative** weighted
  average volatility with zero warnings.
- **`estimate_covariance` silently drops incomplete rows** — one short
  history truncated 400 of 512 rows with `warnings: []`, moving risk-parity
  weights by 12.4% of NAV.
- **`build_portfolio` lets NaN through** and three downstream calculations
  treat it three different ways; 20 missing days added **+290 bp of CAGR**
  and +0.14 of Sharpe.
- **HRP is not invariant to column order** — 40 permutations of the same
  universe moved single weights by up to 8.7 pp.
- **`plan_rebalance` treats a name missing from `adv` as having zero
  liquidity**, so it never trades, and the warning does not say why.
- **The screener is hard-wired to yfinance** and cannot reach Databento.

### The backtest engine: SOUND

- **Execution lag is exactly one bar, proved both directions.** A signal
  clairvoyant by one bar returns Sharpe 13.82; the same-bar signal, which
  contains no future information, returns −0.259.
- **Costs are arithmetically exact** at four settings on both the C++ and
  Python paths; the portfolio engine matches an independent reimplementation
  to **4.1e-16 relative**.
- **Every reported metric reconciles** with plain pandas from the returned
  equity curve.
- **C++ and Python agree to the last decimal** across 20 grid combos × 3
  fill modes and 9 portfolio configurations.
- **All five sizing rules match closed form**; risk parity's realised risk
  shares match to 9.8e-12; min-variance and max-Sharpe survive 1.5 million
  perturbations with **zero** improvements found.
- **The hedge sign convention is right** and was checked explicitly by grid
  search: the minimum-variance ratio is −1.168000 against −beta of −1.167920.

---

## 7. Options, futures and derivatives

### D9. The IV solver returns its initial guess and reports convergence — DEFECT ✅

`analysis/options.py:339-353`. The convergence test is an **absolute price
tolerance** applied before any step is taken, so where vega is small a vol
wrong by hundreds of points still prices inside 1e-6.

```
K=250 T=0.000274 put  true_iv=3.00  returned=0.200000  converged=True  iters=1
K=500 T=0.002700 put  true_iv=1.20  returned=0.200000  converged=True  iters=1
K=650 T=0.050000 put  true_iv=0.45  returned=0.200000  converged=True  iters=1
```

✅ Four of four returned exactly the default `initial_guess=0.2`. Over a
700-case grid, 549 reported convergence and **28 were off by more than 0.01
vol, 17 by more than 0.10, worst 2.80**. The bisection fallback the
docstring promises for small-vega cases is never reached.

### D13. The no-arbitrage bound refuses prices its own pricer produced — DEFECT ✅

`analysis/options.py:318` uses a strict `<` on both sides. For a deep-ITM
call `black_scholes_price` returns a value **bit-for-bit equal** to the
intrinsic lower bound:

```
K=200 T=0.25 sig=0.08: price = lower = 134.9884850488  diff = 0.00e+00 -> ValidationError
```

77 of 700 cases refused on the bound; 74 more because the pricer underflowed
to 0.0, and one returned a **negative** option price.

### Also in options and futures

- **`fit_volatility_smile` reports moneyness in a field named `strike`** —
  a trader is told the arbitrage is at "k=1.00" while `strike_range` in the
  same payload says [300, 370].
- **`analyze_strategy` max_loss is half the true worst case** and labels a
  bounded loss "unbounded", because the scan starts at half the lowest
  strike rather than zero.
- **Three carry decompositions do not sum to the basis they decompose** —
  up to 45% off — in `implied_forward_price`, `cash_futures_basis` and their
  shared root in `derivatives.py:1062`.
- **`price_option(model="bachelier", dividend_yield=…)` silently discards
  the dividend**, with `notes = None`.
- **`roll_analysis` silently drops 83% of the spread cost** when
  `spread_ticks` is given without `tick_value`.
- **`rehedge="drift"` measures the residual of a hypothetical fresh hedge**,
  not the one held, so the band is bounded by half a contract and can never
  fire. Measured: the rule sat through **81.8% residual beta** on a 5% band.
- **The futures engine books zero P&L on every roll day** ($7,025 per
  contract per year, 5.5 pp of return) and **fills a target it cannot margin
  then liquidates it in the same bar**, charging both legs — 239 margin calls
  and 52.6% of starting capital in fees, in a year when ES rose 18.4%.

### Options and futures: SOUND

- **Black-Scholes to 1.14e-13** over 864 cases against an independent scipy
  reference; **all greeks match finite differences**, including vanna,
  volga, charm and speed, with no sign errors and correct scalings.
- **Put-call parity on REAL OPRA quotes**: eight strikes, every one inside
  half the combined bid-ask spread, worst violation 4.3 bp of spot. This is
  the strongest available real-world check and the function passes it.
- **IV on real OPRA mids**: all 16 contracts, max difference from scipy
  `brentq` **4.6e-09**.
- **Binomial converges as clean O(1/n)**; the "within about a cent at 200
  steps" claim is exact.
- **Continuous-futures stitching matches Databento's own `ES.v.0`** to
  **2.91 bp over a year**, with the entire disagreement being roll timing;
  ratio adjustment reproduces the old contract's roll-day return to 0.0000 bp.
- **Contract multipliers are not hardcoded** — verified against live CME
  `definition` records for ES, CL and GC.
- **Day count is consistent** — every core function takes years, and the
  three internal conversions are all calendar/365 and mutually consistent.

---

## 8. Microstructure and order events

### D7. Three functions crash on any real tick tape — DEFECT ✅

`analysis/microstructure.py:224-227` and `:311`,
`analysis/liquidity_events.py:205`. All three do
`.loc[<index with duplicate labels>]`, and real trades share timestamps —
**32% of prints in a live AAPL minute**, in every window sampled.

```
✅ effective_spread(t, q)          -> ValueError: cannot reindex on an axis with duplicate labels
✅ microstructure_summary(t)       -> IndexError: boolean index did not match indexed array
✅ microstructure_summary(t, q)    -> ValueError: cannot reindex ...
```

The third is worse because it **does not raise**: `_signed_volume` returns a
longer, wrong Series by label alignment. Measured on a tape whose total
volume is 64,780 shares, it reported a net imbalance of **−229,340** — 5.5x
the truth and 3.5x larger than everything that traded. And
`detect_liquidity_events` catches only `ValidationError`, so one channel's
`ValueError` kills all six — and the failing channel is in
`available_channels()`, so **the obvious call is the one that dies**.

### D8. Kyle's lambda regresses on the sign of its own dependent variable — DEFECT ✅

`analysis/microstructure_estimators.py:563`:
`signed_volume = np.sign(price_change) * frame["volume"]`, then regressed on
`price_change`. Since x = sign(y)·V, lambda is positive by construction.

```
✅ real AAPL bars                     : lambda 1.54e-06  r2 0.609
✅ returns shuffled, volume permuted  : lambda 1.19e-06  r2 0.328   <- true lambda is ZERO
   (4 trials, all 1.19-1.69e-06)
✅ non-circular control, same bars    : lambda -1.73e-10  r2 0.0004
```

Destroying every real relationship leaves ~80% of the estimate intact. The
docstring claims the opposite failure mode — that misclassification
*understates* impact — and the guard against a non-positive lambda protects
against something that cannot happen.

### Also in microstructure

- ✅ **`intraday_volume_profile` reports a 0% open and close** on a real
  feed, because it buckets over the observed extended-session range:
  `open_share 0.00004`, `close_share 0.00000`, `u_shaped False` — and warns
  the caller that *their data* is unusual. Restricted to regular hours the
  same bars give 0.234 / 0.153 and `u_shaped True`.
- ✅ **`estimate_vpin` appends a phantom bucket** from a float residue whose
  VPIN is exactly 1.0 and which lands last, dominating `current_vpin`
  (measured: 51 buckets reported for 50 requested).
- **`roll_spread`'s significance guard is dead in the windowed branch** —
  it reported 59.2 bp against its own detection floor of 161.4 bp with
  `significant: None`.
- **The CUSUM threshold fires on 43% of quiet real channel-windows**,
  because it is calibrated on i.i.d. noise and real channels are
  autocorrelated (spread channel lag-1 +0.671, 8/10 false alarms). The
  constant itself is sound — an i.i.d. control gives 7.0%.
- **MBO:** `cancel_to_trade` counts every execution twice (T and F are the
  same trade); every XNAS fill is also counted as a cancellation, flipping
  `cancel_to_add` across 1.0; `terminated_without_an_add` is **54.5% false**
  on a CME reopen; `events_per_second` is off by **16,000x** on a
  snapshot-bearing window; `queue_positions` understates the real queue by
  33–79% against the `mbp-10` book for the same sequence numbers.

### Microstructure: SOUND

- **Lee-Ready is 99.7% accurate against the venue's own aggressor flag**,
  on two independent windows, with zero buy-classified-as-sell errors. The
  tick-rule-only path scores 0.95 and 0.90, so the docstring's "about 85%"
  is conservative.
- **`effective_spread`'s realized-horizon alignment is exact** — the
  forward-asof trick matches an independent `searchsorted` to 0.0 across
  1,022 trades, and `effective = realized + impact` holds exactly.
- **Amihud, Corwin-Schultz pairs, order-flow imbalance** all reproduce
  independently; the OFI overlapping-window trap is handled as documented.
- **Price scale holds across three schemas** — MBO, trades and daily bars
  agree, and no sentinel survived into output across 57,000+ records.

---

## 9. The plumbing

- **Databento touches neither the cache nor the retry layer** — `grep` → 0
  hits, where the other three providers import both. Three identical live
  requests made three metered fetches. This is the root cause of D2.
- **The cache key omits the dataset.** Four feeds 30.8x apart in volume
  collapse to one filename. Latent only because Databento bypasses the
  cache — and it must be fixed *before* anyone fixes D12.
- **A cached frame does not round-trip**: parquet changes the index
  resolution from `datetime64[s]` to `[ms]`, so `hash_dataframe` differs and
  a replay reports `data_changed` for byte-identical data.
- **The session cache has no historical guard**, so an unsettled bar is
  served as final for up to an hour; and the disk guard compares against
  the local date, so east of UTC+5:30 a mid-session bar is written
  permanently.
- **Nothing evicts a cache entry.** The real cache holds 1,574 files / 47 MB,
  **501 of them a dead generation** that will never be read.
- **Concurrency**: a cold runs directory produces spurious path-traversal
  refusals on Windows because the `\\?\` prefix handling exists in one of
  four copies of the containment check, and the publish path is not the one.
- **The agent data runtime can only ever reach yfinance**, and its error
  message says "Only PolygonProvider does" for tick data — ✅ false, since
  `DatabentoProvider` implements both `get_trades` and `get_quotes`.
- **`describe_data_capabilities` reports an unconfigured Databento as
  `available=True`**, because its constructor defers the key check.

---

## 10. Checked and found sound

Beyond the per-area lists above: the retry layer's non-retryable contract
holds exactly; cache path-traversal defences are correct; artifact-store key
validation rejects every traversal form tried; `classify_divergence`,
`data/ratios.py` unit handling, and the `TemporalContract` machinery all
behave as documented; `DataBundle` container semantics are correct; and the
vendor's own aggregation identities are **perfect** — `1s → 1m → 1h → 1d`
reproduces to the share on every dataset tested, in both DST regimes.

---

## 11. Pre-existing failures, unrelated

`tests/surface/test_adversarial_inputs.py::TestTheBaselineHolds` fails two
tests, before and after both passes, unchanged: four modeling tools reject
their own synthesized input on a cross-field validator, and 205 of 209 tools
synthesize with 0 declared unsynthesizable. Collection counts drift between
runs because that file synthesizes tests from the live tool surface.

---

## 12. Traps that make a live test lie

Each produces a **passing** test that checks nothing.

- **Inherited dataset pins.** A sibling project's `.env` sets all three
  Databento dataset variables. A first pass of this work concluded the
  `dataset=` constructor argument was ignored — it was not; the inherited
  override was winning. Both suites now clear all three.
- **yfinance `end` is exclusive; Databento's is inclusive.** The same two
  dates name windows differing by one session.
- **yfinance is tz-naive; Databento is UTC-aware.** Joining without
  normalising silently yields an empty frame, and a test on an empty join
  passes by comparing nothing to nothing. Both suites assert a minimum
  joined row count first.
- **Window end-date luck.** The original suite's window ends on a Friday,
  which is the one case where D1's extra bar does not appear.

---

## 13. Cost

Across all twelve investigations plus my own verification: roughly **75 MB**
of DBN moved, dominated by one MBO window (50.8 MB) and one OPRA definition
pull (1.3 MB). `metadata.get_cost` returned **0.00** on every preflight —
this account is a subscription, not metered per gigabyte. Every OPRA and MBO
fetch was preflighted with `get_billable_size` first.

Run the live suites with the **engine's** interpreter; this repo's
`.venv311` does not have the package installed:

```
cd "C:/Users/karan/Documents/Projects/Standard Tools"
DATABENTO_API_KEY=... \
"C:/Users/karan/Documents/Projects/Carbon Redifined/services/engine/.venv/Scripts/python.exe" \
  -m pytest tests/data/test_databento_live.py tests/data/test_databento_pipeline_live.py \
  -q -m integration -p no:randomly
```

---

## 14. Not covered

- **Modeling dataset construction and features** — leakage checks, the
  missing-data policies, feature alignment and target construction. An
  investigation was running when this was written; its findings are not here.
- **ICE and Eurex venues**, and options/futures end to end, all blocked by
  D5 rather than untested by choice.
- **A live restatement of a Databento bar**, which needs two pulls
  separated by a correction event.
- **`optuna`-backed TPE search**, not installed in this environment.
- **Bloomberg and Polygon** column and dtype contracts — no terminal, no key.
