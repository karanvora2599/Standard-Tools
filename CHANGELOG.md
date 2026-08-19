# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/) —
while the major version is `0`, breaking changes may still land in a minor
bump, consistent with SemVer's pre-1.0 clause.

## [Unreleased]

### Fixed (full-codebase audit, Pass 2 — one shared numerical contract)

The audit's own diagnosis was that roughly 40 of its findings were a single
problem wearing different clothes. `@validate_series` checked emptiness and
nothing else — its all-NaN check sat in the body as commented-out code, and
there was no infinity check at all — so every metric wearing that decorator
had its own accidental behaviour for the same invalid input:

```
sharpe_ratio(all-NaN)       -> nan
sortino_ratio(all-NaN)      -> +inf        (reads as "no losing bars")
var_historical(all-NaN)     -> IndexError
max_drawdown(contains inf)  -> -1.703437775179145
```

That last one is why the fix belongs in the shared decorator rather than in
each function: an infinity does not stay visibly wrong. It came back as a
drawdown that looks measured. Suite: 2562 → 2613 passed, 1 skipped.

**New `standard_quant_tools.numeric_contract`** — one set of helpers for
every public numerical boundary: `require_finite_series`,
`require_positive_price_series`, `require_positive_start_level`,
`require_aligned`, `require_positive_int`, `require_finite_scalar`,
`require_periods_per_year`, `require_finite_covariance`.

Three rules, each drawn deliberately:

- **Non-finite input is never information.** `±inf` in a price or return
  series has no economic reading — it is a division that should not have
  happened upstream. Rejected everywhere.
- **All-NaN is not a series.** It carries no observation at all. Rejected.
- **Partial NaN is allowed by default.** This is the contract's deliberate
  limit. Warm-up windows, a ticker that lists mid-sample, a benchmark on a
  different holiday calendar all produce legitimate gaps, and many callers
  drop them internally on purpose. Making them fatal would break correct code
  to catch a problem it has already handled. Pass `allow_nan=False` where a
  gap genuinely cannot be tolerated.

**Prices must be strictly positive, not merely finite.** `0.0` and `-5.0` are
perfectly finite and are not prices. `run_strategy` checked finiteness only,
so a single `Close` of **-5.0 produced a total return of +0.397914** — a
plausible profit computed through a negative price — while a `0.0` close
produced a silent total wipeout. It also now rejects a price/signal pair that
share no dates, which previously surfaced as an empty-slice error far from
the cause.

**Level series get a weaker, correct rule.** `require_positive_start_level`
constrains only the OPENING value, because a leveraged position can genuinely
be wiped out and an equity curve legitimately reaches zero or goes negative at
its tail. What must hold is that the *denominator* is positive: cumulative
return divides by the first value, and the drawdown ratio divides by a running
maximum seeded from it. A non-positive open made `max_drawdown` return
**-1.0048519736842105** — a drawdown deeper than total loss.

**Sortino no longer conflates two opposite states.** `+inf` meant both "the
strategy never had a losing bar" and "the deviation could not be computed" —
the single most flattering possible misreading of unusable data. The genuine
no-downside case still returns `+inf`; an incomputable one does not.

**CAGR counts intervals, not observations.** N levels contain N-1 returns, so
`len(series) / periods_per_year` overstated the elapsed time and understated
the growth rate. Negligible over a decade of daily bars; over 21 observations
it is a 5% error in the exponent's denominator, growing as the window shortens
— exactly where a CAGR is already least reliable.

**`periods_per_year` is validated wherever it is used.** It is a bare
multiplier, so an invalid value produced a confidently wrong number rather
than an error: `-252` returned a CAGR of **-0.5350151890419428**, which reads
as an ordinary annual loss. Zero raised a bare `ZeroDivisionError` from inside
the arithmetic.

**Cost primitives reject credits.** Every function in `backtest/costs.py` is a
bare arithmetic expression, so a negative rate returned a *negative cost* —
indistinguishable downstream from a rebate:

```
percentage_commission(1e6, rate=-0.001)  -> -1000.0
fixed_bps_spread(1e6, bps=-10)           -> -1000.0
short_borrow_cost(1e6, annual_bps=-500)  -> -4109.59
```

A backtest charging negative commission earns money by trading, flattering
exactly the strategies that turn over most. NaN is checked before the sign,
since `value < 0` is False for NaN. `pct_of_range_spread` also rejects an
inverted bar (`high < low`).

**Sizing hygiene.** The score panel rejected NaN but not infinity — and
infinity is worse here, because it makes a column's mean and standard
deviation NaN, so *every* weight in that cross-section becomes NaN rather than
just the offending one. `gross_leverage` is now validated too: it scales the
whole vector, so a negative value flips every position — turning the strategy
into its own opposite — while each individual weight still looks well-formed.

**Diagnostics semantics.**

- A trade returning exactly `0.0` is neither a win nor a loss. It used to fall
  into `losses` via `~is_win`, dragging `avg_loser` toward zero and extending
  `max_consecutive_losses` through trades that were actually flat. On a
  win/breakeven/loss triple it reported `avg_loser -0.5` and **2** consecutive
  losses; it now reports `-1.0` and **1**.
- A NaN position satisfies `!= 0`, so a missing position counted as time *in*
  the market while making every exposure average NaN.
- An unmeasurable excursion (empty price window, unusable entry price) is NaN
  rather than `0.0` — which reads as "this trade never moved against me",
  the most flattering answer available for a trade whose prices are missing.

### Fixed (full-codebase audit, Pass 1 — temporal correctness and integrity)

A fresh review of the whole repository, taken independently of the earlier
passes. Its central finding was that the modeling runtime is no longer the
weakest part of the codebase — the remaining risk had shifted to the older
quant runtime, which never gained the deterministic input/output contracts
the modeling layer now enforces.

This pass fixes the subset that produces a temporally wrong answer, a
security hole, or a silently benign reading of missing data. Every item was
reproduced against a live interpreter before being fixed and is pinned by a
regression test in `tests/core/test_pass1_temporal_integrity.py`. Suite:
2516 → 2562 passed, 1 skipped.

**Deleting a model's manifest bypassed every integrity check.**
`_expected_hash()` caught a `ValidationError` from `load_manifest()` and
returned `None`, and `verify_file()` treats `expected=None` as "skip
verification". `manifest.json` is the package's commit point — written last,
holding every other artifact's digest — so removing it downgraded all of them
at once. Measured on a registered model whose `model.joblib` had been
swapped: with the manifest present the load was refused; with the manifest
deleted the tampered file was **deserialized**. Removing a file is strictly
easier than forging a hash inside it, so the bypass was cheaper than the
attack it existed to stop, and `joblib.load` executes code from the file it
is handed. The manifest error now propagates. A *valid* manifest that simply
predates content hashing still yields `expected=None`, so genuinely legacy
models keep loading.

**A negative strategy lookback read future prices.** Not one of the eight
registered strategies validated a single parameter, and
`momentum_timeseries(lookback=-20)` reached `Close.pct_change(periods=-20)`,
where pandas reads *forward*. Standing at bar 25 it returns
`close[25]/close[45] - 1`, so a bar's signal is computed from a price 20 bars
into its own future. Reachable from the agent surface, since
`BacktestInput.parameters` was an unconstrained `Dict[str, Any]`.

New `backtest/strategy_params.py` gives the classic registry the contract the
modeling runtime already had: positive-integer windows, finite thresholds,
declared ranges, cross-parameter relations, and rejection of unknown names
(every signature ends in `**_`, so a typo silently ran the default while the
caller believed it had configured something). It is applied by wrapping
`STRATEGY_REGISTRY` itself rather than at each of the ~10 call sites, so it
cannot be reached around — including from the `ProcessPoolExecutor` grid
worker, which rebuilds its call in a child process. Cross-parameter relations
are enforced where a single configuration is deliberately requested but not
inside the registry, because a parameter grid legitimately sweeps
`fast >= slow` pairs and `backtest_grid` does not catch per-combination
errors.

**The engine's look-ahead warning never reached the agent.** `run_strategy`
has always emitted a caveat for `fill_price="close"` (a signal derived from
bar *t*'s own close cannot realistically be filled at that same close), but
`BacktestResult` had no `warnings` field and `_run_backtest` rebuilt the
result without it. The engine knew the simulation might contain look-ahead
while the LLM-facing output said nothing. `fill_price` and `strategy_type`
are `Literal`s now; the latter's description listed four of the eight
registered strategies, so half the registry was undiscoverable from the
schema.

**A sparse signal panel deleted trading days.** `run_strategy` intersects
price dates with signal dates and then takes `pct_change()` over what
remains, so a monthly signal against daily prices does not read as "hold" —
the intervening days vanish and the bars either side become adjacent.
Measured on a 120-bar series driven by identical exposure: annualized
volatility **0.0241 with a daily signal against 0.7735 with the same signal
sampled monthly**, a 32× distortion of risk from the same prices. Total
return can still look right, which is what made it easy to miss. The agent
wrapper already applied a fill policy; the public
`backtest.panel.run_signal_panel_backtest` beneath it did not, and now takes
`signal_calendar_policy` (`hold` / `flat` / `error`). `hold` does not
back-fill before the first signal, since no view had been expressed yet.

**Intraday bars from different exchanges looked simultaneous.**
`tz_localize(None)` was applied without converting first, which keeps the
local wall clock: London 15:00 BST (14:00 UTC) and New York 15:00 EDT (19:00
UTC) both became naive 15:00 and indexed identically, so any cross-market
correlation, PCA or panel silently paired them as one instant. Intraday is
now canonicalized to **UTC** before the timezone is dropped — Polygon's
parser included, which had been emitting naive New York time. Daily and
coarser deliberately do *not* convert: a daily bar is identified by its local
session date, and converting first would shift Tokyo's 2024-06-03 to
2024-06-02. Cache format bumped to `v3`, since every `v2` intraday file holds
local wall-clock times.

**A corrupted audit trail silently restarted itself.** An unparsable last
line returned `None`, which the caller turned into the genesis hash — so the
writer began a new chain and kept appending as though the trail had just
started. Reproduced exactly that way. "The file does not exist" and "the file
exists and I cannot read its tail" are different states: the first is a
legitimate genesis, the second means the log is already damaged, and
extending it destroys its evidential value. Now raises the new
`AuditIntegrityError`. The cross-midnight race is closed too — the previous
day's tail is read while holding *that day's* lock, so a writer appending at
23:59:59 cannot be missed by one creating the new day's file at 00:00:00.

**"Unknown" stopped meaning the benign case.** Three places used a
valid-looking number as a failure sentinel, each biased toward the
reassuring answer:

- `calculate_beta` returned `beta: 0.0` when fewer than two observations
  overlapped — indistinguishable from a genuinely market-neutral asset. It
  returns NaN now, and `treynor_ratio` no longer turns "no overlapping
  benchmark data" into a plausible risk-adjusted return.
- `adv_participation` and `impact_cost` returned `0.0` for an unusable volume
  baseline and called it conservative. It is the opposite:
  `adv_participation(1e9, adv=0)` scored **0.0** where a real baseline scores
  **100.0** (100× ADV), and `impact_cost` scored **$0** against **$3bn** — so
  the ticker with no liquidity data ranked as the cheapest in the universe to
  trade. Both return NaN. The `max_adv_participation` gate now rejects an
  unestimable participation explicitly, since `nan > limit` is False and would
  otherwise let an unmeasurable trade pass a constraint a merely large trade
  fails.
- `days_to_liquidate` guarded with `<= 0`, which NaN does not satisfy, so a
  NaN volume produced a NaN answer that looked computed.

**Optimizer scalars are finite-checked before any comparison.** Every domain
guard in `mean_variance_optimize` is written as a comparison, and NaN makes
all of them False — so `if target_volatility <= 0` never fired for NaN, and
`risk_free_rate`, `target_return` and `target_volatility` each produced
`{ticker: nan}` weights reported with `converged: True`. `periods_per_year`
must now be a positive integer.

### Added

- **`standard_quant_tools.error.AuditIntegrityError`** — raised when the
  audit trail's own hash chain is damaged. Distinct from `ValidationError`
  because it is not a statement about the caller's input: it says the
  tamper-evident log on disk can no longer be extended honestly.

### Fixed (portfolio, screener and agent-tools audit — 10 items)

A line-by-line pass over `portfolio/`, `screener/` and `agent/tools.py`, the
three packages the earlier audits had only touched incidentally. Same method:
every finding reproduced against a live interpreter before being fixed, each
pinned by a regression test. Suite: 2452 → 2493 passed, 1 skipped.

**Portfolio optimization**

- **`max_sharpe` could return the *minimum*-Sharpe portfolio.** The
  closed-form tangency solution normalizes `Σ⁻¹(μ − rf·1)` by its own sum,
  `B − rf·A`. The resulting excess return is `(μ−rf)'Σ⁻¹(μ−rf)` over that
  sum — a quadratic form in a positive-definite Σ, so the numerator is
  *always* positive and the sign is entirely the denominator's. Once `rf`
  reaches the global minimum-variance return `B/A` the normalization flips
  onto the inefficient branch. Only `abs(denom) < 1e-14` was guarded, which
  catches the un-normalizable case and misses the inverted one. Measured on
  μ=[0.10,0.08], Σ=[[.04,.01],[.01,.05]], rf=0.20: Sharpe **−0.66** with
  `converged=True`. It also split the backends — closed-form −3.0707 against
  scipy +0.1423 on identical inputs. Now rejected with the threshold named;
  bounded requests still solve, since bounds make the feasible set compact.
- **A rank-deficient covariance produced a "zero-risk" portfolio.** With
  observations ≤ assets the sample covariance is singular *by construction*
  (rank ≤ n−1), handing the optimizer a null space of zero-variance
  directions. The closed-form path caught it (its inverse fails); the SLSQP
  path inverts nothing and did not. On 5 observations of 6 assets it
  returned `expected_volatility` 1.19e-07, in-sample `w'Σw` = 1.4e-14, and
  `converged=True` — for weights carrying **23.1% annualized volatility out
  of sample**. Both paths now check the same condition before either solver
  runs, so they cannot disagree about solvability, and perfect collinearity
  is rejected on the same grounds. The gate also covers `risk_parity` and
  `black_litterman`, which bypass `mean_variance_optimize` entirely.
- **Infinite returns produced NaN weights reported as converged.**
  `dropna()` removes NaN but not `±inf`.
- **`max_weight` feasibility was only checked for long-only.** Shorting
  lowers the per-asset floor, not the cap, so `sum(w) == 1` is equally
  unreachable when `n × max_weight < 1`; `allow_short=True` with n=2 and
  max_weight=0.3 returned weights summing to **0.6**.
- **Small samples are now warned about.** Same process, 5 assets: 6
  observations report an annualized volatility of 0.0039 where 250 report
  0.1376 — a ~22× understatement, previously indistinguishable. A warning
  rather than an error, since a short window is a legitimate request.
- **`build_bl_views` raised a raw `KeyError`** on a malformed view dict.
  These are agent-reachable, so the error is what an LLM self-corrects from.

**Screener**

- **A beta that could not be estimated was reported as `0.0`.**
  `calculate_beta` returns all-zeros below two overlapping points — a
  sentinel indistinguishable from a real answer, since 0.0 is a legitimate
  beta. The screener *filtered* on it, so a ticker with no overlap with the
  benchmark **passed** `beta_max=0.5`: "could not be estimated" read as
  "very low beta", backwards for the defensive screen that bound expresses.
  A minimum overlap is now required and a shortfall reported as an error.
  The floor is a `min_beta_obs` parameter on `screen_stocks`,
  `screen_stocks_async` and `ScreenerInput` (default
  `DEFAULT_MIN_BETA_OBS` = 20) — a judgment call, not a mathematical bound,
  so weekly bars or a deliberate recent-listing screen can lower it. It is
  bounded below at 2, which is *not* a matter of taste: below two
  overlapping points the sentinel and a real beta of 0.0 are the same
  number. Threaded through the `ProcessPoolExecutor` worker tuple as well,
  since a parameter missing from that tuple silently reverts to its default
  in the child and would make the same request screen differently at
  `n_workers=1` than at `n_workers=8`.
- **Filter *values* went unvalidated while only keys were checked.**
  `rsi_max=float("nan")` made every comparison False, so an oversold screen
  silently became a no-op admitting RSI 100 — a filter that rejects nothing
  looks exactly like a filter nothing failed. Wrong types and out-of-range
  windows raised inside the per-ticker handler, turning one malformed filter
  into *N* identical per-ticker errors that never named the filter.
- **A crashed worker batch lost its tickers.** `failed_batches` recorded the
  exception but not which symbols went with it, so those tickers were absent
  from the results, from `failed_filters` and from `failed_tickers` alike —
  indistinguishable from never having been asked for. The batch's tickers
  are now named, and each also appears individually in `failed_tickers`.

**Agent tools**

- **Duplicate tickers desynchronized a result's own fields.** The returns
  frame is built as `{ticker: close}`, so a repeat collapses to one column;
  `['AAA','BBB','AAA']` came back with `tickers` listing three symbols and
  `weights` holding two. Rejected at the boundary rather than de-duplicated
  (a repeat leaves the caller's intent genuinely ambiguous), and weights are
  now labelled from the solved columns so the two cannot drift apart again.
- The optimizer's `warnings` now reach the caller instead of being dropped
  at the tool boundary.

### Fixed (second modeling audit — 20 items)

A second full review of the modeling stack, the data layer beneath it and
the numerics both rest on, worked in a fixed order from the findings most
capable of producing a confidently wrong answer down to hardening. Every
item was reproduced against a live interpreter before being fixed and is
pinned by a regression test that records the *reason*, not just the
behaviour. Suite: 2343 → 2451 passed, 1 skipped.

The common thread is the failure class this library exists to remove: a
result that is plausible, internally consistent, and wrong, with nothing in
the output to say so.

**Temporal correctness (items 1–5, 9)**

- **Full-refit information cutoff used the feature date, not the label
  date.** A row dated `t` with a horizon-`h` forward-return target reads
  `Close[t+h]` to build its label, so the estimator has indirectly seen
  prices through `max(label_end_date)`. Measured on a 120-bar / h=20 panel:
  feature end 2026-05-20 vs label end 2026-06-17 — a 28-day window in which
  `score_model` accepted an `as_of` whose future the model had already
  consumed. Manifests now record `training_information_cutoff` and
  `score_model` gates on it. Models registered earlier still score under the
  old guard; they are detectable (`training_information_cutoff is None`) but
  not retroactively safe.
- **`end_date` meant different things per provider.** yfinance's `end=` is
  exclusive, Polygon's and Bloomberg's are inclusive, and `data/base.py`
  never stated which the ABC required — so the same call returned a
  different window depending on who served it, and silently dropped the
  final bar on the default provider. Resolved toward **inclusive** at the
  ABC, with all three providers trimming through the shared
  `trim_to_inclusive_end` so the contract holds by construction rather than
  by trusting each vendor's documented boundary. Cache format bumped to
  `v2`; v1 files were written under the exclusive behaviour and are never
  looked up again.
- **Intraday timestamps were destroyed by `_normalize_ohlcv_index`.** An
  unconditional `idx.normalize()` collapsed four hourly bars into four
  copies of one date. It ran on the live fetch *and* on both providers'
  Parquet cache reads, so it also made the same request answer differently
  live vs cached. Normalization is now interval-aware; daily and coarser are
  bit-identical to before.
- **A scored "cross-section" could mix dates.** `score_model` took each
  entity's own most recent surviving row, so a halted or short-history
  symbol contributed an older bar inside what the response called one
  `as_of` cross-section — and `missing_entities` never caught it, because
  the entity was present. `effective_score_date` is now enforced across the
  cross-section, with excluded entities reported in `stale_entities` (kept
  separate from `missing_entities`: "no data" and "older data" have
  different causes and different fixes). `staleness_days` is always
  reported; `max_staleness_days` is opt-in, since how much staleness is
  decision-useful is a property of the strategy.
- **Universe-scope features did not pin the universe.** `score_model`
  permits a different scoring universe, which is right for entity-scope
  features and wrong for `factors.pca_loading` /
  `factors.pca_factor_return`: those are computed from the whole universe's
  return matrix, so scoring [AAA, BBB] a model trained on [AAA, BBB, CCC]
  feeds the estimator a different PCA basis under the same column name. Now
  required to match exactly (as sets) — but only when a universe-scope
  feature is actually present, so the permission survives where it is sound.
- **Calendar gaps in OOS predictions compressed the price axis.**
  `run_strategy` intersects prices down to the signal index and then takes
  `pct_change()` over what remains, so an absent span does not read as
  "flat" — the bars either side become adjacent. Measured on a 90-day
  series with February missing: the boundary bar carried **26×** a normal
  daily return. A skipped walk-forward fold is now rejected (its dates are
  absent from every entity, so nothing can be densified against — only the
  caller knows the missing calendar); an entity-level gap is filled with
  0.0 on the panel's shared calendar, which is the honest fill.

**Provenance and integrity (items 10–14)**

- **Aliases destroyed feature provenance — and could forge it.** Panel
  columns are `FeatureSpec.output_name`, i.e. the alias, and those names
  were looked up in `FEATURE_REGISTRY`. An aliased feature recorded
  `"unavailable"`; an alias that happened to name *another* registered
  feature recorded that feature's hash. Verified: `alias="technical.rsi"` on
  a momentum feature recorded RSI's hash `2f6444a367010516` rather than
  momentum's `a3f025e590b1bbb3`. Not a missing record — an actively wrong
  one, in the field whose whole job is answering what produced a column.
  Provenance now resolves from the spec's own entries into
  `feature_provenance`, with the alias as the KEY and never a lookup.
- **The recorded hashes were never checked at scoring time.** Now compared
  before scoring, with the mismatch named per column. Scoped honestly: the
  hash covers the feature function's own source, not its transitive
  dependencies.
- **The OOS predictions artifact was loaded without verifying its digest.**
  The bridge's structural validation is shape-based, so flipping the sign of
  the prediction column passes all of it and produces a clean, entirely
  plausible backtest of numbers the model never emitted. The digest was
  already in the manifest, unused. Direct-URI mode stays explicitly
  unverified — with no manifest there is no root of trust — and now says so.
- **`dataset_spec.json` was never verified against its `spec_hash`**, even
  though the spec is the more dangerous of the pair to tamper with: the
  panel is only read during training, while the spec is copied into the
  model and defines what `score_model` rebuilds features from for the rest
  of that model's life.
- **Score artifacts were mutable.** The name covered only (date, universe)
  and was written with `overwrite=True`, so re-scoring after a provider
  revised its data replaced the file in place — and an audit record written
  earlier still pointed at that URI, which now returned different bytes. The
  filename now carries a content digest, returned as `predictions_hash`.
- **Persisted JSON was not valid JSON.** `save_json` used `allow_nan=True`,
  writing bare `NaN`/`Infinity` tokens (the runtime legitimately produces
  NaN: AUC on a single-class fold, ICIR with no dispersion). Now routed
  through the same `sanitize_for_json` the agent boundary uses, with
  `allow_nan=False` so any future path fails at the write.

**Statistics (items 15–16)**

- **The regression baseline cheated.** `baseline_regression_metrics` built
  its constant from the *test* fold's own mean, so the model was judged
  against a standard no real forecaster could meet. The constant now comes
  from the training fold, and `baseline_is_oracle` reports which is in
  force.
- **Cross-sectional IC dispersion was averaged across folds, not pooled.**
  `mean(fold stds) ≠ std(pooled ICs)` and `mean(fold ICIRs) ≠ mean(ICs) /
  std(ICs)`. A fold's std measures dispersion *within* that fold's dates
  only, discarding exactly the between-fold variation ICIR exists to
  measure. Demonstrated in the tests: two folds each internally rock-steady
  (std < 0.02) but centred at +0.20 and −0.20 averaged to a "dependable"
  ICIR, while the pooled series has ~zero mean and std > 0.15.

**Numerics (items 17–18)**

- **Power iteration reported success on a null direction.** `pca.py` skipped
  its convergence check whenever the eigenvalue came out ≈ 0, treating a
  zero matvec as a legitimate zero eigenvalue. That is only legitimate when
  no remaining variance exists to find; otherwise the iteration landed on a
  null direction while real structure was still there, and the SVD fallback
  never fired. The check now discriminates on remaining variance and on the
  residual `‖Av − λv‖`. Verified against a rank-1 panel constructed
  orthogonal to the fixed start vector, and confirmed to add **no** SVD
  fallbacks (0 before, 0 after) across the modeling and analysis suites.
- **Volatility features annualized with a hardcoded daily constant.**
  Yang-Zhang, Parkinson and Garman-Klass all multiplied by `sqrt(252)`
  regardless of interval, so weekly bars were reported at roughly 2.2× their
  true annualized volatility. `FeatureContext` now carries the interval and
  the features scale by its own constant; intraday raises rather than
  guessing, because session length is venue-specific and not derivable
  without an exchange calendar. A missing interval still means daily, so
  existing callers are unaffected.

**Boundaries and resource limits (item 19)**

Estimator parameters already carried compute ceilings; the same reasoning
had not reached feature parameters, request sizes, or the RNG seed.

- Integer-valued feature params are enforced from the **default's type**
  rather than a name vocabulary — `refit_every=1.5` previously reached
  `range()` and raised a bare `TypeError` naming nothing.
- `_MAX_WINDOW_BARS` ceiling on feature windows; `universe` capped at 1000
  symbols; `random_seed` bounded to `[0, 2**32 - 1]`.
- Reserved panel column names (`date`, `entity`, `target`,
  `label_end_date`) are now rejected on `FeatureDefinition.id` as well as on
  `FeatureSpec.alias`, from one shared `RESERVED_PANEL_COLUMNS` — two
  independent copies is how the id path drifted from the alias path.
- `oos_predictions_to_signal_panel` validates `task` at runtime (the
  `Literal` is a static hint; `task="banana"` fell through into
  classification handling) and rejects a non-finite `deadband` (NaN
  compares False against everything, silently disabling it; inf compares
  True, silently flattening every prediction to 0).
- `provider_guarantee_warnings(None)` returns an explicit "could not be
  determined" warning instead of `[]` — a failed metadata fetch and a clean
  bill of health were indistinguishable.
- Fold records report `train_end` as the range **actually fit** after
  label-overlap purging, alongside `scheduled_train_end`; their difference
  is the purge extent.
- The documented ENTITY/UNIVERSE custom-feature output contracts are now
  enforced and name the offending feature. The entity contract is
  deliberately a **subset** of the entity's index, not equality: a feature
  legitimately returns fewer rows than it consumes (`risk.rolling_beta`
  loses the first bar to `pct_change`), and panel assembly is index-aligned.

**Native/Python parity (item 20)**

- **The two backtest implementations disagreed on where a trade ends.**
  `backtest.cpp` defines a trade as one **lot** — exposure leaving zero
  until it returns to zero — while `engine.py`'s `_build_trade_log` emitted
  a completed trade for *every* position-changing event. With the native
  kernel present, one result dict reported `num_trades=1` beside a two-row
  `trade_log`. Measured on a 1.0 → 2.5 → 0 sequence: native 1 trade
  averaging 17.4492%, Python log 2 trades averaging 8.5113%, from identical
  inputs.

  **The "same-sign resize" framing of the original Known Issues entry
  understated this, and that framing is itself corrected here.** A partial
  *reduce* diverged identically and was never named — `2.0 → 1.0` is
  opposite-sign without being a full close, so the old code booked it as a
  completed trade too. On `0 → 2.0 → 1.0 → 0`, which contains no same-sign
  resize at all: native 1 trade at 12.8078%, old log 2 rows averaging
  6.1583%. Nor was the cost one spurious row per event: on the 100-bar
  random-signal fixture the cross-check test uses, the old log produced
  **67 rows against the kernel's 50** (7 resizes + 10 partial reduces = 17
  spurious completions) and an average trade return off by **0.087pp**
  (−0.5384% vs −0.6254%). The error compounds across a realistic series.

  `_build_trade_log` now mirrors `apply_position_event` exactly, so cost is
  charged per event on the amount actually transacted — the same
  `sum(abs(pdiff))` the equity curve charges — and trade-log P&L reconciles
  with equity P&L for strategies that scale *or trim* a position. No C++
  change was needed; the kernel was already correct. This closes the "Known
  Issues" entry opened by the earlier C++ pass.

### Changed (test suite layout)

`tests/` now mirrors `src/standard_quant_tools/` — one directory per
package — instead of 70 files in a flat root alongside the two that were
already grouped (`cpp/`, `modeling/`):

```
tests/
  conftest.py   shared fixtures, visible to every subdirectory
  agent/ analysis/ audit/ backtest/ data/ indicators/
  metrics/ modeling/ portfolio/ screener/
  core/         cross-cutting: errors, compat shims, regression suites
  cpp/          C++ gtest sources compiled by CMake — not collected by pytest
  cpp_bindings/ Python-side parity tests for the compiled extension
```

Placement was decided by what each file actually imports, not by its name;
`test_liquidity.py` and `test_stress_test.py` both live under `backtest/`
despite reading like metrics, because that is where the code under test
lives. `tests/cpp/` keeps its name and contents — CMake and
`build-cpp.yml` reference that path — so the Python-side extension tests
went to `cpp_bindings/` rather than colliding with it.

All 70 moves are recorded as git renames, so history follows the files.
`testpaths = ["tests"]` already recursed, and the suite reports the same
2343 passed / 1 skipped before and after.

Three tests reached outside the suite for non-importable files (the
standalone audit verifier script, the reference agent implementations),
each with its own `Path(__file__).parent.parent`, which encodes how deep
that file happens to sit — moving them one level down broke all three at
once. `REPO_ROOT` is now defined once in `tests/__init__.py` and imported,
so a future move updates one line instead of N, where N is not
discoverable until the tests fail.

### Added (modeling: per-feature drop attribution)

Feature/target alignment drops rows — every feature consumes its lookback
window and a forward-return target consumes its horizon — and that loss
was reported as a final row count and nothing else. A count cannot
separate "this is the warm-up I asked for" from "one feature is silently
costing me two thirds of my panel", and it cannot say which feature.

- **`BuildModelDatasetResult.drop_attribution`** — rows before and after
  alignment, per-entity drop counts, and two counts per column:
  `n_missing` (rows where that column was NaN) and `n_sole_missing` (rows
  where it was the ONLY thing missing).

  The pair matters. Warm-up windows overlap, so per-column `n_missing`
  sums to far more than the rows actually lost, and a short-lookback
  feature sitting entirely inside a longer one looks equally guilty. Only
  `n_sole_missing` says what removing that one feature would give back —
  in a measured example, `technical.rsi` was missing in 42 rows and
  recoverable in none of them, because its 14-bar warm-up sits inside
  `risk.rolling_drawdown`'s 252-bar one, which was solely responsible for
  515.

  The target is attributed separately from the features: its cause (the
  forward horizon) and remedy (a shorter horizon, or more data) differ,
  and unlike a feature it cannot be removed.

- **Warnings** when alignment costs more than 30% of rows, or when a single
  feature is solely responsible for more than 10% — not on every dataset,
  since a warning that always fires trains the reader to skip the ones that
  matter. When no column is ever the sole cause, the warning says so rather
  than showing an empty breakdown, which would read as a bug.

- **The empty-panel error now explains itself**, listing rows missing per
  column. "No rows survive feature/target alignment" previously left the
  caller to guess which feature was too long for their window.

- **`entities` now reports what reached the panel**, not what was fetched.
  The two differ whenever a symbol's history is shorter than the feature
  lookbacks plus the target horizon, and reporting the fetched list made a
  dataset look like it covered a universe the model never saw a single row
  of. The fetched list remains available as `entities_fetched`, and any
  symbol that dropped out is named in `warnings`.

`stack_long`/`stack_features_only` now return `(panel, attribution)`. The
tuple is deliberate: an external caller breaks loudly on unpacking rather
than silently receiving un-dropped rows. The aligned panel itself is
unchanged, which a test pins — attribution is additive, and any change
there would move every downstream hash and metric.

### Fixed (modeling: degenerate windows, warm-up, and signed importance)

Four features answered confidently where they had no information, and the
diagnostic meant to catch unstable features rated the least stable one
perfectly stable.

- **`market.new_high_breakout` fabricated its entire warm-up.**
  `breakout_high` is NaN for the first `period` bars and `NaN > x` is
  False, so `.astype(float)` emitted **0.0** — "no breakout occurred" —
  for every bar before the comparison window existed. It was the only
  feature in the catalog that never produced NaN, so it never let
  alignment drop its own warm-up: a dataset built on it began `period`
  bars early, with fabricated negatives in exactly the rows a breakout
  model cares about most, and its declared `lookback=20` described nothing
  observable. Verified against the old code: the panel started at bar 0
  with 115 rows where it should have had 95.

- **`risk.atr_pct` produced ±inf, which rejected the whole panel.** The
  division by `Close` was unguarded, so a single price of exactly 0.0 (a
  bad print, a delisted stub, a provider filling a gap with zero) gave an
  inf — and `build_dataset`'s finite-value guard rejects the ENTIRE panel
  on a non-finite value, so one bad bar in one symbol failed the whole
  build with an error naming the feature rather than the data. Exactly the
  failure mode `volume.obv_roc` was already fixed for. Verified: a
  two-symbol build with one zero print raised instead of building; it now
  drops the affected rows and keeps both symbols.

- **`risk.bollinger_pct_b` dropped halted symbols instead of describing
  them.** A flat window collapses both bands onto the mean, making %B a
  0/0 that came out NaN and was silently dropped by alignment. When the
  window is flat, Close equals that mean exactly, so 0.5 — the middle band
  — is what %B is *defined* to be there, not a fallback. Warm-up stays
  NaN: conflating "the bands collapsed" with "there are not yet `period`
  bars" would repeat the breakout bug. Only the exactly-degenerate case
  needed handling; a near-flat window followed by a jump is well behaved,
  because the jump enters the standard deviation that scales it (%B peaks
  near 1.56, not at infinity).

- **`volume.vwap_deviation`** got the same denominator guard. Flagged
  honestly as defensive rather than a reproduced failure: a zero-volume
  window already yielded NaN here, because VWAP is itself 0/0 there.

- **Feature importance discarded the sign, which silently inverted the
  stability metric.** `fold_feature_importance` returned `|coef|`, so the
  cross-fold `std` — whose stated purpose is showing "whether a feature's
  importance is stable or an artifact of one fold" — was computed on
  magnitudes. A feature alternating `+0.5, −0.5, +0.5, −0.5` across folds,
  the maximally unstable case and a textbook sign of fitting noise, is
  `|0.5|` every fold: **std exactly 0.0, reported as perfectly stable.**
  Added `signed_mean`, `signed_std` and `sign_consistency`; `mean`/`std`
  keep their meaning so existing manifests stay comparable. All three are
  NaN for tree estimators, whose importances have no direction —
  deliberately NaN rather than a plausible default. Exact-zero
  coefficients (routine under L1) do not vote on direction.

  Also fixed a latent misattribution: multiclass `coef_` is
  `(n_classes, n_features)`, which ravels to `n_classes * n_features`
  values, and `zip()` kept the first `n_features` — reporting class 0's
  coefficients as THE importances and dropping every other class without a
  word. Not reachable through the tool surface today (`forward_direction`
  is binary), but `register_estimator` accepts custom estimators, and a
  wrong attribution is worse than an absent one.

One existing test was rewritten: it reproduced
`market.new_high_breakout`'s implementation expression verbatim, warm-up
included, so it pinned the bug rather than the look-ahead-safety claim it
documented.

### Added (modeling: data/runtime architecture)

The modeling runtime was built against whatever `DataFactory.get_provider()`
returned, at whatever interval that defaulted to, one symbol at a time —
none of which was a decision anyone had made or recorded.

- **`DatasetSpec.provider`** (`"yfinance"` | `"polygon"` | `"bloomberg"`)
  and **`DatasetSpec.interval`** (default `"1d"`). Both were previously
  implicit: the builder called `DataFactory.get_provider()` with no
  arguments, so the runtime was a yfinance-daily system by accident and a
  model's lineage could not say what it had been trained on. Because they
  live on the spec they are covered by `spec_hash` and bundled into the
  model, so scoring reuses the same source and interval rather than
  silently substituting the default.

  Credentials are deliberately not spec fields: the spec is written to
  disk, hashed into model lineage and embedded in decision records, so an
  `api_key` here would leak the key into all three. The interval VALUE is
  validated by the selected provider, which owns the authoritative list —
  they genuinely differ, and duplicating a union of them would only drift.

- **Concurrent universe fetch** (`modeling/dataset/fetch.py`), replacing
  the serial dict comprehension; every provider already exposed
  `get_ohlcv_async` and the rest of the library already fetched universes
  concurrently. Bounded by `SQT_MODELING_FETCH_CONCURRENCY` (default 8).
  Three failure modes a bare `asyncio.gather` would have introduced are
  handled: it propagates only the FIRST exception and abandons the rest
  (so all failures are now collected and reported together, sorted, rather
  than one bad ticker per run in nondeterministic order); `asyncio.run`
  refuses to nest, which would have made `build_dataset` unusable from a
  notebook or async agent runtime (falls back to sequential, as it does for
  a duck-typed provider implementing only `get_ohlcv`); and both paths
  report failures identically, so the error does not depend on which ran.

- **Coverage and provenance diagnostics** (`modeling/dataset/coverage.py`),
  finally populating `BuildModelDatasetResult.warnings` — a field that had
  existed since the first version of the tool surface and was never written
  to by anything. Reported: a provider that guarantees neither point-in-time
  data nor a survivorship-free universe (`DataSetMetadata.point_in_time` /
  `survivorship_free` were recorded honestly by every provider and read by
  nothing); a symbol covering materially less of the window than its
  presence in `universe` suggests; a requested window that came back
  shorter than asked for; the complete-case intersection that universe-scope
  PCA features require, which lets one short history truncate the panel for
  every entity; and a non-daily interval against daily-calibrated feature
  defaults.

  These are warnings rather than errors because every provider this package
  ships reports both guarantees as false — failing on that would make the
  runtime unusable against its own default data source while teaching the
  caller nothing. The distinction is now stated explicitly in the docs: a
  `CURRENT_ONLY` feature is *rejected* because a PIT-safe alternative
  exists, while a revising provider is *disclosed* because none does.

- **`ModelManifest.dataset_warnings`**, surfaced by
  `inspect_model(view="lineage")`. The caveats belong next to the metrics
  they qualify, and the build-time tool response is transient — lineage
  previously reported hashes and a commit sha while staying silent about a
  survivors-only universe. An empty list on an older model is
  indistinguishable from "no warnings" by design.

Time-varying universe membership remains deferred: it needs
index-constituent history no shipped provider exposes. What is built is the
diagnosis, not a correction.

Also fixed two tests that were passing for the wrong reason. They wired
only `get_ohlcv` on an unspecced `MagicMock`, so `await`ing the
auto-created `get_ohlcv_async` attribute raised `TypeError`, that
`TypeError` was collected as a per-symbol fetch failure, and an assertion
that the error named a symbol passed while exercising nothing it meant to.
Provider mocks now drive both paths from one function.

### Fixed (modeling correctness review — P0 + P1)

An external line-by-line review of the modeling runtime raised findings the
2,099-test suite did not exercise. Every P0 was reproduced against a live
interpreter before being fixed, and each is pinned by a regression test.
Suite across the pass: **2,099 → 2,248 passing**, 1 skipped.

**Leakage (P0).** `target[t]` reads `Close[t+horizon]`, but `WalkForwardSplit`
is never given the horizon — only an integer `embargo` — so with
`horizon=20, embargo=0` the last 20 training labels were built from
test-period prices. The existing engine tests happened to use
`embargo == horizon`, which accidentally satisfied the missing invariant
and hid it. Training rows are now purged by a per-row `label_end_date`
recorded at build time rather than an integer offset: `horizon` counts an
entity's OWN bars, so on a sparse or heterogeneous calendar `t+horizon`
entity bars is a different date than `t+horizon` panel dates, and an
integer embargo under-purges exactly there.

Separately, `FeatureSpec.params` was unrestricted and splatted straight
into the feature. `market.momentum`/`volume.obv_roc` pass `lookback` to
`pct_change`, and pandas reads a negative period as a FORWARD window — so
`lookback=-20` made the feature at *t* read `Close[t+20]` while its
`PIT_SAFE` label, and therefore the point-in-time gate, stayed satisfied.
`features/params.py` now validates resolved parameter values centrally.

**Wrong answers (P0).** PCA power iteration started from the uniform
vector, which is exactly orthogonal to a `[1,−1]` spread factor: the first
matvec was zero and it returned the zero-eigenvalue direction as PC1 —
explained variance 0.0001 where SVD gave 0.9999. Fixed with a fixed-seed
non-degenerate start plus a residual check that falls back to SVD.
`volume.obv_roc` divided by an OBV series seeded at exactly 0, producing
`±inf` on ordinary data (25 of 60 rows) and causing the dataset builder to
reject the whole panel; reformulated as OBV change normalized by traded
volume. `score_model` never compared `as_of` against the training window,
so it would return a future-trained prediction dressed as a historical one.

**Provenance (P1).** The model directory was a collection of
individually-atomic files, not a verifiable package: an edited
`dataset_spec.json`, tampered `preprocessing_stats.json` or swapped
`model.joblib` all went undetected. Every artifact now carries a content
hash verified on load — `model.joblib` before `joblib.load`, since
deserialization executes code from the file. Models bundle their own
training spec (scoring no longer depends on the dataset directory
surviving), `manifest.json`/`dataset_meta.json` are written last as commit
points, and modeling stopped duplicating the column-blind
`hash_pandas_object` hashing the audit package had already been fixed for.

**Validation statistics (P1).** Pooled IC across every `(entity, date)` row
conflates cross-sectional skill with market timing — a model with zero
ranking ability can post a pooled IC above 0.9 by tracking the market
factor (constructed and pinned in a test). Per-date cross-sectional IC,
ICIR and hit rate were added alongside it, plus a predict-the-mean
baseline, overlap-adjusted effective sample size, prediction-count-weighted
fold averaging, per-fold metrics, skip accounting, and a `min_folds`
default of 2 (one surviving fold is a single split, not walk-forward
validation).

**Agent safety (P1).** Estimator parameters were name-allowlisted but
value-unbounded, so `n_estimators=10_000_000` in one tool call was a
resource-exhaustion path; typed bounds now apply, generous enough that
realistic requests pass. `penalty` was exposed without `solver`, so
incompatible pairs failed inside sklearn; both are now validated together.
`register_estimator` requires `overwrite=True`, matching `register_feature`.

**Capability gaps (P1).** `ModelSpec.task` accepted `"classification"` while
`TargetSpec` could only build a continuous return, so a binary target was
reachable only by mutating the panel by hand — `forward_direction` makes
classification constructible through the five-tool surface, with task/target
compatibility enforced both ways. `FeatureSpec.alias` makes multi-horizon
specs (`momentum(20)` + `momentum(252)`) expressible; uniqueness is enforced
on the output column rather than the feature id. `FeatureDefinition.requires`
is enforced instead of informational.

**Audit replay (P1).** `verify_replay` hardcoded the 46-tool registry, so a
modeling record could not be replayed at all. It now resolves against both
surfaces and compares semantically: modeling mints a fresh id per run and
embeds it in artifact paths, so a byte-identical re-run never matches
literally, and reporting that as a mismatch would look like evidence of
drift. The modeling test fixture also disabled audit entirely; it is now
redirected to a temp directory so the integration is actually exercised.

**Documentation.** `Documentation/15_modeling.md` rewritten against current
behavior; README test counts and modeling summary updated; the
"leakage-safe by construction" phrasing removed from `bridge.py` and the
result model, since that guarantee rests on the target-overlap purge rather
than on walk-forward splitting alone.

### Added (modeling: model→backtest bridge, feature/estimator expansion)

- **`modeling.bridge.oos_predictions_to_signal_panel`** — a trained
  model's out-of-sample predictions can now actually be backtested as a
  strategy, closing a real gap: `score_model` produced a predictions
  Parquet and stopped, with nothing turning it into a
  `run_signal_panel_backtest` call. Deliberately a plain Python function,
  **not a 6th agent tool** — the 5-tool modeling surface stays exactly 5;
  this is the "artifacts, not tool calls" boundary between the modeling
  registry and the existing 46-tool `agent` registry. Two findings drove
  the design: (1) `score_model`'s single as-of snapshot is the wrong data
  source — using its final, fully-trained model to "predict" historical
  dates would be leakage — so the bridge reads `run_model_experiment`'s
  walk-forward out-of-sample fold predictions instead (leakage-safe by
  construction, now persisted as a new `oos_predictions.parquet` artifact
  and exposed via `RunModelExperimentResult.oos_predictions_uri` /
  `ModelManifest.oos_predictions_uri`); (2) `run_signal_panel_backtest`
  never normalizes `SignalType.SCORE` — it's a raw leverage multiplier —
  so a raw `0.02` forward-return prediction passed through as `SCORE`
  would become an economically meaningless ~2%-leveraged position. The
  bridge converts to `SignalType.DIRECTION` instead (sign of the
  prediction, or a thresholded classifier probability), units-invariant
  regardless of prediction scale.
- **12 new features** (9 → 21), all thin wrappers over existing,
  already-implemented primitives (no new indicator math):
  `technical.macd_histogram`, `technical.stochastic_k`,
  `technical.williams_r`, `market.psar_trend`, `risk.atr_pct`,
  `risk.bollinger_pct_b`, `risk.parkinson_volatility`,
  `risk.garman_klass_volatility`, `risk.rolling_drawdown`, `volume.mfi`,
  `volume.obv_roc`, `volume.vwap_deviation` (new
  `modeling/features/volume.py` — the first feature file needing the
  OHLCV panel's `Volume` column, so `tests/modeling/conftest.py`'s
  synthetic fixture gained one). `risk.rolling_drawdown` is deliberately
  **not** a direct wrap of `metrics.risk_metrics.drawdown_series` — that
  function's whole-series `cummax()` gives a stale all-time peak inside a
  multi-year training window; the feature uses a bounded
  `.rolling(window).max()` peak instead.
- **3 new estimators**: `random_forest` for regression (closing an
  asymmetry — it already existed for classification only) and
  `gradient_boosting` for both tasks (the classic, non-histogram GBM).
  Regression 5→7, classification 3→4 — 11 registry entries in total.
  (An earlier revision of this entry said "4 new estimators" and
  "classification 3→5"; the registry has four classification entries:
  `logistic`, `hist_gradient_boosting`, `random_forest`,
  `gradient_boosting`.) Still an explicit allowlist, still
  `scikit-learn>=1.3.0` only — no new dependency.

28 new tests (modeling suite 99 → 127), full suite 2099 passed / 1
skipped, zero regressions.

### Added (performance)

- **`analysis.pca.pca_returns` gained a `method: "svd" | "power_iteration"`
  parameter** (default `"svd"`, exact prior behavior for every existing
  caller). Investigated whether `modeling`'s rolling PCA features
  (`factors.pca_loading`/`factors.pca_factor_return`, which always request
  `n_components=1` and refit repeatedly over a sliding window) were a good
  candidate for a new `_sqt_core` C++ kernel. Finding: `pca_returns`'s
  "slow path" already calls into compiled LAPACK via `np.linalg.svd`, so a
  hand-rolled C++ full-SVD would not reliably beat it — the actual waste is
  algorithmic (full SVD computes every singular triplet regardless of how
  many are wanted). Added `method="power_iteration"` — power iteration +
  deflation applied directly to the return matrix (never forms the
  `n_assets × n_assets` covariance matrix explicitly, since for a wide
  matrix, n_assets > n_obs, that costs more than SVD itself and would
  defeat the point) — computing only the requested components. Wired into
  `modeling/features/factors.py`'s two PCA features. Benchmarked on
  synthetic factor-structured data: **12–45× faster** depending on universe
  size (500-name universe, ~120 refits: 7.0s → 0.16s). Parity with SVD is
  exact for any well-separated eigenvalue (true of PC1 for real market
  data — the only component either `factors.py` feature ever requests);
  near-degenerate eigenvalues beyond PC1 can yield a different orthonormal
  basis within that subspace between methods, an inherent PCA property
  documented in the `method` parameter's docstring, not a bug in either
  path. A C++ kernel (`rolling_top1_pca`, incremental covariance
  maintenance + warm-started power iteration across refits) was scoped in
  detail but explicitly not built — Tier 0 alone made PCA feature
  computation negligible next to the OHLCV fetch it's part of, so building
  a C++ kernel now would be speculative, not evidence-driven.

### Known Issues

> **RESOLVED** by the second modeling audit's item 20 (see the top of this
> file). `engine.py`'s `_build_trade_log()` now mirrors `backtest.cpp`'s
> weighted-average cost-basis accounting, so `len(trade_log)` and
> `num_trades` agree and the native/Python cross-check test deliberately
> *includes* the cases it used to exclude. Note that the entry below is
> **narrower than the actual defect**: it names only the same-sign resize,
> but a partial *reduce* diverged the same way and went unmentioned for as
> long as this entry stood. Kept as the record of what was knowingly
> shipped, and of what the record itself missed.

- **Correctness/portability pass, item 14 of 20 (native/Python trade-log
  divergence for resize scenarios):** `backtest.cpp`'s `run_strategy()`/
  `run_strategy_summary()` now track a genuine weighted-average cost basis
  for the trade log (see the Fixed entry below), but `engine.py`'s
  `_build_trade_log()` (the Python reference implementation, used to build
  the optional `trade_log` DataFrame when `run_strategy(..., 
  include_trade_log=True)` is called) has **not** been updated to match --
  it still treats a same-sign resize as closing-then-reopening two separate
  trades. `run_strategy()`'s scalar stats (`num_trades`, `win_rate`,
  `profit_factor`, `avg_trade_return_pct`) are read directly from the
  native kernel (already fixed), but the returned `trade_log` DataFrame
  (when requested) is still built by the unfixed Python path -- for any
  signal sequence containing a same-sign resize, the DataFrame's row count
  can now disagree with `result["num_trades"]`. Out of scope for this
  native-only pass; tracked here rather than silently shipped unnoticed.
  `tests/test_backtest.py::TestNativeTradeStatsCorrectness::
  test_run_strategy_native_matches_python_recomputed_stats` documents this
  explicitly and excludes resize scenarios from its native/Python
  cross-check accordingly.

### Added

- **Correctness/portability pass, item 20b of 20 (final item -- closes out
  the 20-finding pass):** new `.github/workflows/nightly-tsan.yml` --
  scheduled (`03:00 UTC daily`, plus `workflow_dispatch` for manual runs)
  ThreadSanitizer build+test job, separate from `build-cpp.yml` since TSan
  is meaningfully slower than a normal cycle and only useful periodically,
  not on every push. Explicitly depends on item 2's earlier
  `isa_dispatch.cpp` atomicity fix (`g_override_value`'s independent
  atomics for the override's `avx2`/`fma` bits) -- adding this job before
  that fix would have immediately flagged that race with zero signal about
  anything else; the fix landed first in this same pass. `continue-on-error:
  true`, mirroring `build-and-test-sanitizers`' own established "unproven
  sanitizer config" precedent -- unlike ASan, no `LD_PRELOAD` gymnastics
  are needed to import the TSan-instrumented extension (TSan's runtime
  loads as a normal shared-library dependency, unlike ASan's global-
  allocator interception). Verification is the scheduled CI run itself
  once this lands (no local Linux/TSan toolchain available in this
  session) -- YAML syntax validated locally via `yaml.safe_load`.

  This closes out the full correctness/scale/numerical-stability/
  portability/CI review (all 20 findings, "everything" scope as selected).
  See this file's "Not shipped" section for the one item this session's
  broader work deliberately did NOT ship (the rank-1 Cholesky update/
  downdate, reverted after failing its own numerical-stability gate, and
  explicitly re-validated as the correct call by this same review), and
  the "Known Issues" entry above for the one honestly-scoped gap this pass
  surfaced (native/Python trade-log divergence for backtest resize
  scenarios).
- **Correctness/portability pass, item 20a of 20:** new
  `tests/cpp/fuzz_cointegration.cpp` -- a randomized-input stress test for
  `cointegration.cpp`'s `gauss_elim` and `rolling_regression.cpp`'s
  `cholesky_solve`. Both are anonymous-namespace internals, not directly
  linkable from an external test binary, so this fuzzes them indirectly
  through the public functions that call them: `sqt::ols2`/
  `sqt::engle_granger` (exercise `gauss_elim`) and
  `sqt::rolling_factor_loadings` (exercises `cholesky_solve`). ~1,050
  randomized trials across 7 deliberately varied input shapes (ordinary
  random walk, huge baseline + small variation, near-constant, huge
  dynamic range, all-zero, strongly trending, plain white noise) plus 50
  below-minimum-length edge cases, asserting (1) no crash/UB and (2)
  structural invariants whenever a function reports success: `ols2`
  residuals sum to ~0 (relative tolerance, since these shapes span many
  orders of magnitude by design), `r_squared` in `[0,1]`, `engle_granger`'s
  `p_value` in `[0,1]` and critical values ordered
  `cv_1pct < cv_5pct < cv_10pct`, `rolling_factor_loadings` never produces
  `+/-inf` (only `NaN` or finite). Fixed seed for deterministic default CI
  runs. Registered as a normal ctest (`cpp_fuzz_cointegration`) -- gets
  ASan/UBSan coverage for free via the existing
  `build-and-test-sanitizers` CI job, no separate sanitizer-specific
  wiring needed. Deliberately a lightweight in-repo harness (reusing this
  project's own `pseudo_random`-style PRNG convention already used
  elsewhere in `tests/cpp/`) rather than libFuzzer/AFL++ -- proportionate
  to the review's own "lower priority" framing for this item. All 49,980
  assertions pass locally; full native ctest (9/9) + full pytest (1868
  passed) green.
- **Correctness/portability pass, item 19 of 20:** `build-cpp.yml`'s
  `build-and-test` job now builds/tests `_sqt_core` on a
  `[ubuntu-latest, windows-latest, macos-latest]` matrix (`fail-fast:
  false`) instead of Linux only -- every job that builds or `ctest`s the
  native extension used to run exclusively on `ubuntu-latest`, despite
  local development happening on Windows/MSVC and the codebase having
  compiler-specific branches (`if(MSVC)`/`else()`) throughout its
  CMakeLists.txt that were never exercised in CI. macOS ships no OpenMP
  runtime, so `SQT_HAS_OPENMP` stays undefined there -- genuine coverage
  of the codebase's own already-documented serial fallback path, not just
  an architecture/compiler check. `windows-latest`/`macos-latest` are new,
  unverified legs (this environment has no way to trigger and observe a
  live GitHub Actions run) -- soft-gated via
  `continue-on-error: ${{ matrix.os != 'ubuntu-latest' }}`, mirroring the
  existing `build-and-test-sanitizers` job's own established "unproven
  config, continue-on-error until confirmed green" precedent; the
  existing, already-proven `ubuntu-latest` leg stays a hard gate. The
  sanitizer job and the parity-check job stay Linux-only (ASan/UBSan flag
  syntax differs meaningfully on MSVC, out of scope here). Verification is
  the CI run itself once this lands -- YAML syntax validated locally via
  `yaml.safe_load`, and the CMake files audited for any Ninja-generator-
  specific assumption that could break under `windows-latest`'s default
  Visual Studio generator (none found -- the existing `$<CONFIG:...>`
  generator expressions and per-config `RUNTIME_OUTPUT_DIRECTORY_*`
  properties already support both single- and multi-config generators).
- **Correctness/portability pass, item 18 of 20:** new strict/zero-copy
  `_zerocopy` sibling bindings for the six highest-value large-array entry
  points: `rolling_beta_zerocopy`, `rolling_factor_loadings_zerocopy`,
  `simulate_forward_paths_zerocopy`, `batch_run_strategy_zerocopy`,
  `technical_indicators_zerocopy`, `rolling_hurst_zerocopy`. The existing
  `Array1D` binding type uses `forcecast`, silently copying any input
  that isn't already exactly float64 + C-contiguous -- a real, avoidable
  cost for a caller who already has correctly-typed arrays. Each
  `_zerocopy` sibling takes an untyped `py::array` and validates dtype/
  layout manually via new `require_strict_f64_1d`/`require_strict_f64_2d`
  helpers, raising a clear `ValueError` (not pybind11's own generic
  "incompatible function arguments" message) on a mismatch, then casts
  without `forcecast` -- a correctly-typed input is used in place with
  zero copy. Existing default bindings are unchanged -- fully additive.
  New `tests/test_cpp_zerocopy_bindings.py`: each variant verified to
  produce output identical to its non-strict counterpart for correctly-
  typed input, and to raise a clear error (not a copy) for wrong dtype/
  non-contiguous input.
- **Correctness/portability pass, item 17 of 20:** new
  `simulate_forward_paths_terminal()` (native `simulate_forward_paths_terminal`/
  `simulate_forward_paths_terminal_into`, pybind11 binding, and
  `backtest/monte_carlo.py` Python wrapper) -- a memory-bounded variant of
  `simulate_forward_paths()` that never materializes the full
  `(n_simulations, horizon_days)` path matrix, only each path's terminal
  equity. For a large `n_simulations x horizon_days` (e.g. 1,000,000 x 252
  would be a ~2GB full path matrix), this avoids that allocation entirely.
  Identical RNG/block-bootstrap core (same per-path seed derivation, same
  block-draw/concatenate/cumprod logic, in the same order) -- for identical
  `(seed, inputs)`, `simulate_forward_paths_terminal(...)[i]` equals
  `simulate_forward_paths(...)[i, -1]` exactly, verified via a new
  `test_terminal_matches_full_matrix_last_column_exactly` in
  `tests/cpp/test_monte_carlo.cpp` (exact `==`) and
  `TestSimulateForwardPathsTerminal::test_matches_full_matrix_terminal_stats_exactly`
  in `tests/test_monte_carlo.py`. Trade-off: no per-day
  `equity_band_p5`/`p50`/`p95` in the result (those require the full
  per-day matrix this variant never builds) -- only the terminal-
  distribution stats (`terminal_median`, `terminal_p5`, `terminal_p95`,
  `prob_loss`, `terminal_var_95`, `terminal_cvar_95`). Purely additive --
  the existing `simulate_forward_paths()` is unchanged.
- **Correctness/portability pass, item 13 of 20 (NaN/Inf input contract,
  remaining `_cpp_core.*` wrappers):** mechanical sweep wiring
  `require_finite_array()` into every remaining Python wrapper that
  dispatches to a `_cpp_core.*` kernel with no prior NaN/Inf check:
  `adx`, `wilder_atr`, `bollinger_bands`, `stochastic_oscillator`
  (indicators); `calculate_beta`, `rolling_beta`, `rolling_factor_loadings`
  (regression/multi-factor); `cointegration_test`, `compute_spread`,
  `half_life` (cointegration); `run_strategy`, `backtest_grid` (backtest
  engine). Deliberately **excluded** `hurst_exponent`/`rolling_hurst` --
  both already have documented, intentional NaN-tolerant behavior (silent
  `dropna()` via `to_clean_numpy()`/manual `.dropna()`), and overriding
  that with a hard rejection would be a real behavior change outside this
  pass's scope, not a bug fix.

  Several wrappers (`bollinger_bands`, `stochastic_oscillator`,
  `rolling_beta`, `rolling_factor_loadings`) wrap their C++ call in a
  broad `try: ... except Exception: fall back to pandas/python`, which
  would otherwise silently swallow a `ValidationError` raised inside the
  `try` block and mask bad input behind a confusing fallback instead of
  rejecting it -- every check in this sweep is placed **before** the
  corresponding `try` block (or, in `backtest_grid`'s case where the
  matrix is genuinely built inside the `try`, guarded by an explicit
  `except ValidationError: raise` ahead of the broad `except Exception`).

  New tests across `tests/test_indicators_trend.py`,
  `tests/test_indicators_volatility.py` (including a new `TestWilderATR`
  class -- no prior Python-level coverage existed for that wrapper),
  `tests/test_indicators_momentum.py`, `tests/test_multi_factor.py`,
  `tests/test_cointegration.py`, `tests/test_cpp_regression.py`,
  `tests/test_analysis.py`, and `tests/test_backtest.py`.
- **Correctness/portability pass, item 12 of 20 (NaN/Inf input contract,
  Monte Carlo):** `backtest/monte_carlo.py::simulate_forward_paths()`
  validated `initial_capital`'s finiteness (in the native kernel) but
  never checked `values` (the historical returns being resampled from)
  itself -- a single NaN/Inf poisons `equity` permanently for every
  path/bar downstream of when it's sampled
  (`equity *= (1.0 + values[start+k])`), with no explicit check anywhere
  in the native kernel, header, or binding. Wired `require_finite_array()`
  in right after `values = returns.to_numpy(...)`, before dispatch to
  either the C++ or pure-Python fallback path. New tests:
  `test_nan_in_returns_raises`/`test_inf_in_returns_raises` in
  `tests/test_monte_carlo.py`.
- **Correctness/portability pass, item 11 of 20 (NaN/Inf input contract,
  GARCH):** `analysis/garch.py::garch_volatility_forecast()` already
  called `returns.dropna()`, stripping NaN, but not `+/-Inf` --
  `garch11_variance_recursion_into`'s floor-clamp (`mean < kMinSigma2`)
  is false for both NaN and Inf, so an Inf would otherwise silently
  propagate through the entire native recursion uncaught (confirmed: all
  three of `garch11_variance_recursion_into`,
  `garch11_neg_loglik`/`garch11_neg_loglik_grad` share this pattern).
  Wired `require_finite_array()` in right after `dropna()`, before the
  mean/residual computation (catching an Inf at its source, before
  Inf-arithmetic could turn it into a masking NaN first). New test:
  `test_inf_in_returns_raises` in `tests/test_garch.py`.
- **Correctness/portability pass, items 10/13 of 20 (NaN/Inf input
  contract, first two call sites):** new `require_finite_array()` in
  `validation.py`, raising the existing `ValidationError` -- extends the
  same convention already used for `parabolic_sar`'s `af_*` params and
  `stochastic_oscillator`'s `d_period`. Core numeric kernels require
  finite observations unless their documented semantics explicitly
  support NaN warm-up values; this is the single enforcement point for
  that contract, called once at the Python/API boundary rather than
  duplicated inside each native kernel. Wired into
  `indicators/momentum.py::rsi()` -- deliberately at the Python boundary,
  not inside `rsi_into` itself, which has two internally-inconsistent NaN
  behaviors (its seed loop's `if/else` propagates a NaN into `avg_loss`
  via `-= NaN`; its forward-pass ternaries silently treat NaN as zero
  movement) that this fix doesn't attempt to reconcile -- enforcing
  finiteness before either C++/numba path is reached makes that internal
  inconsistency unreachable rather than papering over it. New tests:
  `test_nan_in_input_raises`/`test_inf_in_input_raises` in
  `tests/test_indicators_momentum.py`. Remaining `_cpp_core.*`-dispatching
  wrappers land in follow-up commits.
- **Correctness/portability pass, item 9 of 20:** new adversarial
  large-baseline/large-`max_lag` regression tests for the cointegration
  kernels, mirroring `rolling_beta`'s own large-baseline test pattern --
  `test_large_baseline_no_catastrophic_cancellation` /
  `test_large_baseline_hedge_ratio_recovered` /
  `test_max_lag_above_old_silent_cap_is_honored` in
  `tests/test_cpp_cointegration.py`, plus native mirrors
  (`test_ols2_large_baseline_no_catastrophic_cancellation`,
  `test_eg_large_baseline_hedge_ratio_recovered`,
  `test_adf_max_lag_above_old_silent_cap_is_honored`) in
  `tests/cpp/test_cointegration.cpp` calling `sqt::ols2`/
  `sqt::engle_granger`/`sqt::adf_test` directly. These pin the items 5/6/7
  fixes above against regression.

### Fixed

#### Modeling runtime reliability pass (14 findings)

A follow-up correctness/reliability pass on the new
`standard_quant_tools.modeling` package (see the Added entry below),
prompted by "too basic when it comes to reliability and error handling."
Suite after the pass: **99 modeling tests / 2063 total passed, 1
skipped**, plus a repeated live end-to-end run against real Yahoo
Finance data confirming the fixes are additive/defensive and don't
change happy-path numerics.

**Silent data corruption**

- **Duplicate feature ids in `DatasetSpec.features` silently overwrote
  columns.** Requesting `technical.rsi` twice (even with different
  params) produced no error — `dataset.builder`'s `columns[fs.id] = ...`
  assignment just let the second call clobber the first, so a caller
  believing they'd requested two features silently got one. Fixed with a
  `DatasetSpec` field validator rejecting duplicate feature ids.
- **Duplicate universe symbols** were similarly unvalidated (harmless in
  `build_dataset`, since a dict comprehension naturally deduplicates, but
  an accident, not a guarantee). Fixed with the same validator pattern.
- **`scoring.score_model` reconstructed its scoring-time `DatasetSpec`
  via `original_spec.model_copy(update=...)`.** Pydantic v2's
  `model_copy` does **not** re-run validators, so it silently bypassed
  both checks above (and the start-before-end check below) for every
  `score_model` call, even though the exact same `DatasetSpec` class
  enforces them everywhere else. Fixed by reconstructing via
  `DatasetSpec(**{...})` instead, which re-validates.

**Crashes with unhelpful messages instead of clear errors**

- **`DatasetSpec.start >= end`** was never checked; a caller error there
  surfaced only much later as an opaque empty-panel or provider error.
  Now rejected immediately with a clear message.
- **Malformed date strings** (`DatasetSpec.start/end`, `ScoreModelInput.as_of`)
  raised a raw `dateutil`/pandas parse error instead of this codebase's
  `ValidationError`. Fixed via a shared `_parse_date` helper used both
  inside Pydantic validators (where it needs to raise plain `ValueError`
  for Pydantic to wrap) and directly from `scoring.py` (where the
  `ValueError` is now caught and re-raised as `ValidationError`, matching
  every other failure mode `score_model` raises).
- **`features/factors.py`'s PCA features used `window`/`refit_every`
  directly as a `range()` step with no validation** — `refit_every=0`
  crashed with Python's cryptic `range() arg 3 must not be zero` instead
  of a clear, attributable error, and `window<2` could feed
  `pca_returns` an underdetermined single-observation slice. Both
  parameters are now validated up front.
- **`task="classification"` against the only target `TargetSpec`
  currently builds (a continuous forward return) reached sklearn and
  failed deep inside `.fit()` with "Unknown label type: continuous."**
  `engine.run_experiment` now validates the target is binary `{0, 1}`
  before attempting any fold, with a message explaining `TargetSpec`
  doesn't yet build classification-ready targets directly.
- **A walk-forward fold whose training window happened to land entirely
  on one class of a binary target crashed the whole experiment** (a
  classifier can't fit on one class). Now skipped, the same discipline
  already used for an empty train/test slice, as long as at least one
  fold ends up with both classes in train.
- **Classification probability extraction assumed `predict_proba(...)[:, 1]`
  is always the positive class.** `estimator.classes_` doesn't guarantee
  that column ordering, and a fold whose estimator only ever saw one
  class returns a single-column `predict_proba`, so `[:, 1]` could
  silently score the wrong class or raise a raw `IndexError`. Fixed with
  a shared `validation.metrics.positive_class_proba` helper (used by both
  `engine.py` and `scoring.py`) that looks the class index up explicitly.
- **A provider fetch failure for one symbol in a multi-symbol universe**
  (network error, delisted ticker, rate limit) propagated with no
  indication of *which* symbol caused it. `dataset.builder` now wraps
  every fetch (universe symbols and the benchmark, which previously had
  no empty-data check at all — only universe symbols did) in a
  `ValidationError` naming the symbol, chaining the original exception.
- **`estimators.registry.validate_params` raised a raw `KeyError`** for
  an unregistered `(task, name)` if called independently of
  `get_estimator_class` (which already reported the identical condition
  as a clear `ValidationError`). Now consistent.

**Silent partial failure**

- **`dropna()` removes `NaN` but not `+/-inf`** — a degenerate feature
  computation could feed `inf` straight into sklearn. `dataset.builder`
  now runs `require_finite_array` over every feature/target column
  before returning the panel, the same enforcement point this codebase
  already uses pervasively elsewhere.
- **`score_model` silently dropped universe entities with no scoreable
  row** (e.g. insufficient history within `lookback_days`) from the
  result with no indication anything was missing. `ScoreModelResult`
  gained a `missing_entities` field, populated by comparing the
  requested universe against what actually scored.

#### C++-codebase audit pass (10 findings)

A line-by-line correctness audit of the native tier (~5,000 lines across
`_cpp/src`, `_cpp/include`, and `bindings.cpp`), the counterpart to the
Python audit below. Built clean under MSVC `/W4 /permissive-`. Suites after
the pass: **9/9 C++ (ctest)** and **1964 passed, 1 skipped** (Python).

The native tier was in materially better shape than the Python tier —
no memory-safety defects, no data races, and no incorrect math. The real
findings were two cross-backend divergences and one invariant the code
documented but did not hold.

**Cross-backend divergences (same call, different answer per build)**

- **`rolling_factor_loadings` disagreed for `window < k+2`.**
  `rolling_regression.cpp` bails to all-NaN when the window has fewer
  observations than the `k+1` coefficients being estimated, but
  `analysis/multi_factor.py`'s fallback handed the underdetermined system to
  `numpy.linalg.lstsq`, which returns its minimum-norm solution instead. The
  same call therefore produced NaN or numbers depending only on whether
  `_sqt_core` was built. Resolved toward the C++ behavior — a minimum-norm
  solution to an underdetermined system is an artifact of the solver, not an
  estimated factor loading — via a short-circuit ahead of the path dispatch
  so both backends answer identically.
  `tests/test_multi_factor.py::test_window_equals_1_factor_loads_trivially`
  is why this went unnoticed: it asserted only `result.shape`, so it passed
  on both paths. Replaced with value assertions covering both the
  underdetermined case and the smallest determined window.
- **`profit_factor` disagreed when every trade returns exactly 0.0.**
  `backtest.cpp` used `(gross_loss > 0) ? win/loss : (gross_win > 0 ? inf : 0.0)`,
  returning `0.0` when gross_win and gross_loss are both zero (a flat price
  series with zero costs), where `engine.py`'s `_compute_trade_stats` returns
  `inf`. Resolved toward Python — which also makes the kernel consistent with
  its OWN documented rule, since `tests/cpp/test_backtest.cpp` already pinned
  "no losing trades -> inf". Fixed in both `run_strategy` and
  `run_strategy_summary` (separate copies of the expression, so
  `batch_run_strategy` would otherwise have kept disagreeing).

**Exceptions escaping OpenMP parallel regions (undefined behavior)**

`hurst.cpp` carried an explicit comment asserting that no exception could be
thrown inside `rolling_hurst_into`'s parallel region — "throwing across an
`#pragma omp for` boundary is undefined behavior / terminates the process."
That claim was false: two throw sites were live inside it
(`numerics::clamp_near_zero_sumsq` via `dfa_onepass`, reachable on exactly
the ill-conditioned input it exists to detect, and
`numerics::checked_narrow_to_int` via `hurst_exponent_scratch`), and
`batch_run_strategy` had the same shape via `run_strategy_summary`. Fixed by
hoisting the loop-invariant narrowing checks out of both regions (making the
inner copies unreachable rather than merely improbable) and converting the
negative-SSE condition into a per-thread flag combined with
`reduction(||:)`, rethrown as a real exception after the region — so a
genuine numerical bug still surfaces instead of killing the process. The
misleading comment now states what is actually guaranteed, and why.

**Consistency against the project's own `numerics.hpp`**

`numerics.hpp` exists to replace ad-hoc absolute thresholds with a
relative-epsilon convention, but several kernels predated or bypassed it:

- `rolling_regression.cpp`'s `rolling_beta_into` used a fixed
  `abs(denom) > 1e-14` while `cholesky_solve` in the *same file* already used
  the relative test; both now use `is_negligible_pivot`.
- `hurst.cpp`'s `ols_slope_r2` used fixed `1e-14` twice, and returned
  `{0.0, 0.0}` as its "couldn't fit" sentinel. 0.0 is a perfectly valid slope
  that `classify()` labels `"mean_reverting"`, so an unfittable series was
  reported as confidently mean-reverting; it now returns NaN, which
  `hurst_exponent` already maps to the `"unknown"` regime.
- `cointegration.cpp`'s `ols2` tested `det = s1*sxxd - sxd^2` against a scale
  of `sxxd` alone. `det` grows with the observation count as well as the
  spread of x, so the singularity check was roughly n times too lenient — a
  genuinely near-singular system passed on any long series. Scale is now
  `s1*sxxd`.
- `indicators.cpp`'s `bollinger_bands_into` clamped ANY negative variance to
  zero (`var > 0.0 ? sqrt(var) : 0.0`), collapsing the bands onto the moving
  average with no signal — the exact silent failure the shift-by-reference
  centering directly above it was added to prevent, left undetectable if it
  ever recurred. Now `clamp_near_zero_sumsq`, which clamps genuine noise and
  throws on anything larger.
- `isa_dispatch.cpp`'s `force_isa_features_for_testing` did not apply the
  `avx2 && fma` conflation that real detection applies, so forcing
  `{avx2=true, fma=false}` would route `rolling_beta_into` into the
  `_mm256_fmadd_pd` kernel — an illegal instruction, from the very function
  whose job is to prevent one. Test-only (not exposed through the bindings).
- `monte_carlo.cpp`'s `simulate_forward_paths` computed its output size
  without `numerics::checked_mul`, which exists for exactly that.

**Investigated and confirmed NOT bugs** (recorded so this ground isn't
re-covered): `SQT_RESTRICT` is sound — it is applied to inputs Python callers
CAN alias (e.g. `stochastic_oscillator(s, s, s)`), but every `out` across all
39 `mutable_data()` sites is freshly allocated (`_zerocopy` refers to inputs
only), and read-only aliasing among `const` restrict pointers is not UB.
OpenMP data races: none — `monte_carlo` declares `gen`/`dist` per-thread with
per-path derived seeds (so results are thread-count independent) and
`batch_run_strategy` writes distinct pre-sized indices. GARCH's analytic
gradient recurrences were verified term by term, including that
`new_g_beta` reads `sigma2_prev` before it is updated. The MacKinnon p-value
coefficients match statsmodels' N=2/`"c"` response-surface values, with the
correct branch direction. `run_strategy`/`run_strategy_summary`'s
bit-identity contract holds. `rolling_beta_reduce_avx2` correctly assigns
rather than accumulates, paired with the caller not pre-zeroing on that
branch. Binding-level length/dtype validation is consistent across every
multi-array entry point.

#### Python-codebase audit pass (31 findings)

A line-by-line correctness audit of the Python tier (`src/`, plus the
`Implementation/` and `Multi_Agent_Implementation/` trees), complementing
the C++-focused 20-finding pass above. Every finding below was reproduced
against a live interpreter before being fixed, and each is pinned by a
regression test in the new `tests/test_bugfix_regressions.py` (39 tests).
Full suite after the pass: **1961 passed, 1 skipped** (baseline before:
1921 passed, 1 skipped).

**Memory safety**

- **`indicators/trend.py` — out-of-bounds heap writes in `_adx_numba`.**
  The kernel wrote `result[period, ...]`, `dx_vals[period]` and
  `result[2*period-1, ...]` without ever bounding them against `n`. Numba's
  `@njit` compiles with bounds checking DISABLED, so for `n <= period` (e.g.
  `adx(..., period=14)` on 10 bars) these wrote roughly 96 bytes past the
  end of the output buffer and returned "successfully". The same source run
  as pure Python (numba absent) raised `IndexError`, and the C++ kernel
  returned all-NaN — three dispatch paths, three behaviors, one of them
  memory-unsafe. Added an early all-NaN return for `n <= period`, matching
  the C++ kernel. `_psar_numba` had the same class of defect (unconditional
  `low[0]`/`high[0]` bootstrap read on an empty array) and is guarded the
  same way. `adx()`/`parabolic_sar()`/`atr()`/`williams_r()`/`vwap()`/`mfi()`
  now also reject mismatched input lengths up front, which was the other
  route into an out-of-bounds read under `@njit`.

**Silently-wrong results**

- **`backtest/engine.py` — `run_strategy`'s finite-input contract depended
  on whether the C++ extension was built.** `require_finite_array` ran only
  inside the `fill_price="close"` C++ branch, so identical input raised
  `ValidationError` with `_sqt_core` present and silently produced NaN
  metrics without it. Worse, `fill_price="next_open"`/`"hl2_exploratory"`
  were never validated at all: `intraday_leg` lacked the `.fillna(0.0)` its
  sibling `overnight_leg` had, and because `Series.cumprod()` is
  `skipna=True` a NaN `Open` did not poison the curve — it silently DROPPED
  that bar's P&L, leaving a NaN hole and a `total_return` computed over a
  quietly shortened series that still looked complete. Validation now runs
  once for every path and covers the reference-price columns each fill mode
  actually reads; missing columns are named explicitly instead of surfacing
  as a raw `KeyError`.
- **`metrics/risk_metrics.py` — `evt_tail_risk` extrapolated below its own
  threshold.** Peaks-Over-Threshold is only valid above the fitted
  threshold, but nothing enforced `confidence > 1 - tail_fraction`. With the
  documented default `tail_fraction=0.05`, `confidence=0.90` gives an
  exceedance probability of 2.0, making `tail_prob**(-xi) - 1` negative and
  returning a "VaR" *below* the threshold — a wrong number, not an imprecise
  one. Now rejected with a message pointing at `var_historical`/`cvar` for
  in-sample quantiles. The docstring also claimed a (0.5, 1.0) bound that
  `_check_confidence` never enforced; corrected. `_fit_gpd_pwm`'s unguarded
  `b0 - 2*b1` division is now caught as a degenerate fit.
- **`backtest/robustness.py` — `parameter_sensitivity` could rank NaN as the
  best trial.** `np.sort` places NaN last, so `[::-1]` placed it *first*: a
  single NaN metric (a grid row with zero-variance returns is the common
  source) became `best` and made every reported gap NaN. Non-finite trials
  are now excluded from the ranking with a warning, and an all-NaN column
  raises instead of returning nonsense.
- **`audit/hashing.py` — two provenance-hash collisions.**
  `hash_dataframe` used `pd.util.hash_pandas_object`, a per-row digest that
  never sees column labels, so two frames holding identical numbers under
  entirely different column names (a `Close`/`Open` frame and a
  `Volume`/`Adj` frame) produced the *same* fingerprint. Column names,
  dtypes and order are now part of the hash. Separately, `hash_payload`'s
  `default=str` routed ndarrays through numpy's abbreviating repr, so two
  10,000-element arrays differing only in the middle hashed identically; the
  fallback encoder is now lossless. **`hash_payload`'s output is unchanged
  for records made only of native JSON types**, which is every
  `DecisionRecord`/chain-index entry — so the tamper-evident record chain
  built on it still verifies across this change (pinned by
  `test_chain_hash_unchanged_for_plain_json_records`). `hash_dataframe` IS a
  format change: replaying a record captured by an older version will report
  a `data_source` mismatch even when the data is unchanged.

**`data/_retry.py` — three defects in one decorator**

- `ValidationError` (and every other non-`APIError` `QuantError`) was caught
  by the broad `except Exception` and re-raised as `APIError`, so a caller's
  `except ValidationError` never fired.
- `retry(times=0)` silently returned `None` **without ever calling the
  wrapped function** — the loop body never executed and the trailing
  `if last_exc` was falsy. `times < 1` now raises at decoration time.
- Raw network exceptions (`ConnectionError`, `TimeoutError`, and anything
  else outside this package's hierarchy) hit the catch-all and were wrapped
  and raised on the **first** attempt, so the single most common transient
  failure mode was never actually retried. Classification now happens inside
  one handler rather than across overlapping `except` clauses — the previous
  clause ordering also silently decided the wrong outcome for
  `NonRetryableAPIError`, which is itself an `APIError`.

**Cross-path divergences (same call, different answer per build)**

- `indicators/momentum.py` — `stochastic_oscillator` on a zero-range window:
  the C++ kernel returned `0.0`, the pandas fallback `NaN`. The fallback now
  matches the compiled kernel, with warm-up bars still NaN.
- `analysis/cointegration.py` — `cointegration_test(autolag=...)`: the C++
  path mapped anything that wasn't exactly `"bic"` onto AIC while the
  statsmodels fallback passed the string straight through to `coint()`, so a
  typo ran a *different* lag-selection criterion depending on the build.
  Now validated against `{"aic", "bic"}`.
- `analysis/hurst.py` — the C++ path returned the kernel's dict verbatim,
  skipping the `clip(0, 1.5)` and the `_classify` regime thresholds the
  Python fallback applies. Both paths now share the same post-processing.

**Validation-consistency gaps**

- `require_finite_array` added to `parabolic_sar`, `atr`,
  `multi_factor_regression` and `kalman_hedge_ratio` — each had siblings in
  the same module already enforcing the contract while these quietly
  propagated NaN into a result that looked complete (`parabolic_sar` on
  `[1, nan, 3]` returned `[1.0, 1.0, 1.0]`).
- `backtest/panel.py` — `run_signal_panel_backtest`'s docstring has always
  required `weights` to cover every ticker and sum to 1.0; nothing enforced
  it. A dict missing a ticker raised a bare `KeyError`, a wrong-length list
  silently misaligned weights against columns, and weights summing to
  anything else produced a scaled portfolio that still looked valid.
- `backtest/portfolio_engine.py` — `target_weights.index` was checked for
  duplicates but `price_data` was not; a duplicated bar made
  `.loc[date, "Close"]` return a Series and `float()` raise a bare
  `TypeError` from deep inside the per-bar loop.
- Missing period/window validation added across `sma`, `ema`, `macd` (which
  also now rejects `fast >= slow`), `bollinger_bands`, `williams_r`, `vwap`,
  `mfi`, `rolling_beta` and `block_bootstrap_ci`. `rolling_factor_loadings`
  keeps its documented `window=1` minimum-norm behavior and only rejects
  `window <= 0`.

**Edge cases**

- `metrics/return_metrics.py` — `cagr` on a wiped-out equity curve
  (`total_ret <= -1`, reachable since `run_strategy` applies no bankruptcy
  floor) computed `(1 + total_ret) ** (1/years)`, yielding NaN plus a
  `RuntimeWarning` that then propagated silently into `calmar_ratio`. Now
  reports `-1.0` (total loss) with a warning.
- `backtest/costs.py` — `per_share_commission(0, ...)` returned the
  per-order `minimum`, inventing a commission for a trade that never
  happened.
- `data/_cache.py` — `"/"` was encoded by replacing it with `"-"`, making
  `BRK/B` and `BRK-B` (two genuinely different symbols) resolve to the same
  cache file, so one symbol could be served the other's cached bars.
- `indicators/volume.py` — `mfi` returned `0.0` ("maximally oversold") for a
  window with no money flow at all: the second unconditional `.where()`
  overwrote the first one's `100.0`. Now NaN, which is what an undefined
  ratio actually is.
- `metrics/diagnostics.py` — `exposure_stats` guarded `idx.get_loc` against
  `KeyError` only; on a non-unique index it returns a slice or mask and
  `int()` raises `TypeError`, crashing instead of skipping the trade.
- `agent/tools.py` — `_sanitize_for_json` walked only dicts and lists and
  tested `isinstance(obj, float)`, so a non-finite value inside a tuple, or
  an `np.float32`/ndarray, survived to `json.dumps` and emitted the
  non-standard `Infinity`/`NaN` tokens that strict parsers reject.
  (`np.float64` subclasses `float` and was already covered.)
- `backtest/engine.py` — `_build_trade_log` indexed with bare `[]`, which is
  positional for an integer index and label-based otherwise; now `.loc`.
- `portfolio/optimize.py` — a non-converged SLSQP result is still returned
  (callers may want the iterate) but now logs a warning with the actual
  weight sum, rather than leaving the violated sum-to-1 constraint buried in
  a boolean.
- Typing/cleanup: implicit `Optional` corrected in `error.py` and
  `validation.py`; unused imports removed from `agent/tools.py`,
  `data/base.py`, `data/yfinance_provider.py`, `portfolio/portfolio.py`;
  dead unreachable `series.empty` branch removed from `rsi` (its
  `@validate_series()` decorator already rejects empty input);
  `sqrt_impact_bps`'s docstring formula corrected to include the 1e4 bps
  conversion the code applies.

**Investigated and confirmed NOT bugs** (recorded so the same ground isn't
re-covered): all three pylint E-level hits in `agent/models.py`/`tools.py`
are pydantic/`or`-narrowing false positives; `pandas.DataFrame.attrs`
survives the `ProcessPoolExecutor` pickle boundary, so `screen_stocks`'
`failed_filters`/`failed_tickers` aggregation is sound;
`portfolio_engine.py`'s `prev_date` IS updated (line 605), so calendar-day
financing accrual is correct; the `not v != v` NaN idiom is correct; and the
Black-Scholes Greeks, Corwin-Schultz, Yang-Zhang, Merton two-fund frontier
and Deflated-Sharpe implementations were each checked against their
published forms and are correct as written.

#### C++ correctness/portability pass (20 findings)

- **Correctness/portability pass, item 16 of 20:** `indicators.cpp`'s
  `wilder_atr_into` allocated a full `std::vector<double> tr(n)` temp
  buffer despite every `TR[i]` depending only on `high[i]`/`low[i]`/
  `close[i-1]` (and `high[0]`/`low[0]` for bar 0) -- no lookback beyond
  that. Fused to O(1) auxiliary memory via an inline `tr_at(i)` helper
  used by both the seed and forward-smoothing loops, mirroring
  `adx_into`'s existing precedent in the same file exactly (same
  technique, same file, already applied to DM/TR there). Pure refactor --
  same arithmetic, same order, not a reassociation -- verified bit-
  identical against an independent unfused array-based reference
  implementation via a new
  `test_wilder_atr_matches_unfused_array_reference_exactly` in
  `tests/cpp/test_indicators.cpp` (exact `==`, not `CHECK_NEAR`). Full
  native ctest (8/8) + full pytest (1851 passed) green.
- **Correctness/portability pass, item 14 of 20 (highest-risk item in this
  pass):** `backtest.cpp`'s `run_strategy()`/`run_strategy_summary()` trade
  log used to treat a same-sign position RESIZE (e.g. size 1.0 -> 2.5) as
  closing the 1.0-sized trade and opening a fresh 2.5-sized one, each
  independently costed at `2*abs(own size)*cost_per_unit` -- double-counting
  cost relative to what the equity curve itself charges for that one event
  (`abs(pdiff)*cost_per_unit`). This was an explicitly documented, tested
  approximation, not a hidden bug -- now replaced with a genuine
  weighted-average cost basis: a new shared `PositionState` +
  `apply_position_event()`/`flush_open_lot()` (anonymous-namespace helpers
  used identically by both `run_strategy()` and `run_strategy_summary()`)
  track `size`/`cost_basis`/`cost_accrued`/`realized_pnl_accum` across a
  lot's whole life, so a resize is now a partial ADD that blends cost basis
  and charges only the incremental amount actually transacted, and a lot's
  final trade-log cost always equals `sum(abs(pdiff))*cost_per_unit` summed
  over every event that touched it -- matching the equity curve exactly.

  A full open-then-close (via a real event or the final-bar flush) and a
  sign-flip (close-then-reopen in one event) are **unchanged** in total
  cost/pnl from the old model and remain bit-identical-by-construction
  (verified: all pre-existing pinned tests in `tests/cpp/test_backtest.cpp`
  pass unmodified except the resize test below). Only a same-sign resize's
  accounting actually changes -- for the resize case in
  `test_trade_log_resize_cost_is_documented_approximation` (renamed
  `test_trade_log_resize_cost_is_weighted_cost_basis`), the whole
  open→resize→close sequence is now correctly ONE continuous trade (was 2),
  with total cost `5*cost_per_unit` (was `7*cost_per_unit`) -- this is the
  fix, not a regression. New
  `test_trade_log_cost_matches_equity_curve_cost_property` pins the general
  invariant (trade-log total cost == equity-curve total cost, for any
  signal sequence, via a costed-vs-cost-free differential) rather than only
  the one hand-verified case.

  See the "Known Issues" entry above for a real, honestly-scoped gap this
  surfaced: `engine.py`'s Python-side `_build_trade_log()` (used only for
  the optional `trade_log` DataFrame, not for `run_strategy()`'s own scalar
  stats) has not been updated to match, so that DataFrame can now disagree
  with `result["num_trades"]` for resize scenarios specifically.

  Verified: full native ctest (8/8, including the updated/new backtest
  tests) + full pytest (1851 passed) green.
- **Correctness/portability pass, item 8 of 20:** `hurst.cpp`'s
  `dfa_onepass` (the one-pass DFA reformulation shipped in the prior
  performance pass) computed each chunk's sum-of-squared-residuals via a
  sum-of-squares-style accumulation (`Syy - a*Sy - b*S_jy`) with no guard
  against it drifting slightly negative under floating-point cancellation
  before feeding a `sqrt` -- never observed to actually trigger, but no
  guard existed either. Routed through the new
  `numerics::clamp_near_zero_sumsq`: clamps to exactly `0.0` only when the
  negative magnitude is negligible relative to `Syy` (the dominant raw
  term feeding the subtraction); otherwise throws, surfacing a real bug
  instead of silently hiding it with a blind `max(x, 0)`. **Hard gate,
  passed cleanly**: the existing ill-conditioned adversarial test
  (`test_dfa_onepass_tolerance_ill_conditioned`, strongly-trending and
  near-constant series) plus a new, deliberately more extreme combined
  fixture (strong trend *and* tiny chunk-local variance together) both
  exercise the clamp path with zero throws -- this item ships as designed,
  no escape-hatch revert needed. Full native ctest (8/8) + full pytest
  (1830 passed) green.
- **Correctness/portability pass, item 5 of 20:** `cointegration.cpp`'s
  `ols2()` (backing `calculate_beta`, `half_life`, `compute_spread`, and
  `engle_granger`'s hedge-ratio step) accumulated raw, uncentered sums
  (`Σx`, `Σx²`, `Σxy`, ...) -- the same catastrophic-cancellation bug
  class already fixed in `rolling_beta_into` and `bollinger_bands_into`
  for exactly this reason. Confirmed empirically before fixing: for a
  ~1e9-baseline `x` with genuine unit variance and a well-posed linear
  relationship, the raw formula's `det = s1*sxx - sx*sx` computed to
  *exactly* `0.0` (total cancellation between two ~1e20-magnitude terms),
  making the pre-existing absolute `1e-14` singularity guard falsely
  declare the pair singular -- `ols2` silently returned all-`NaN` for a
  regression that isn't singular at all, just poorly conditioned by the
  baseline. Fixed via the same shift-by-reference-point technique as
  `rolling_beta` -- here a single shift by `x[0]`/`y[0]` suffices (`ols2`
  is a one-shot fit, no sliding window, so no periodic re-centering is
  needed) -- with the un-shifted intercept recovered algebraically at the
  end. Also replaced the `det` singularity check's fixed absolute `1e-14`
  threshold with a relative one (`numerics::is_negligible_pivot`, scaled
  to the shifted matrix's own magnitude), consistent with this pass's
  other singularity-threshold fixes. Verified against the same adversarial
  case: the fixed implementation now recovers slope/intercept matching an
  independent `numpy.polyfit` reference to 4+ significant figures, instead
  of `NaN`. Full native ctest (8/8) + full pytest (1830 passed) green,
  confirming the existing (tolerance-based, not exact-equality)
  `ols2`/`engle_granger` test suite is unaffected for well-conditioned
  inputs. Dedicated pinned adversarial tests land in a follow-up commit.
- **Correctness/portability pass, item 6 of 20 (second consumer):**
  `rolling_regression.cpp`'s `cholesky_solve` used the same fixed absolute
  `s <= 1e-14` singularity threshold on its Cholesky diagonal as
  `cointegration.cpp`'s `gauss_elim` (fixed above) -- a threshold that
  doesn't scale with the design matrix's own magnitude. Now a relative
  threshold (`s <= 1e-12 * max(diagonal_scale, 1.0)`), scale computed once
  from the matrix's own original diagonal before decomposition begins.
  Deliberately NOT routed through the shared `numerics::is_negligible_pivot`
  helper used elsewhere -- that helper tests `|value|` (correct for
  Gaussian-elimination pivots, which can legitimately be negative), whereas
  a Cholesky diagonal entry must be positive before its `sqrt` immediately
  below, so a large-magnitude *negative* value (definitely not positive-
  definite) must still fail this check exactly as the original threshold
  did. New adversarial tests in `tests/cpp/test_rolling_regression.cpp`:
  a well-conditioned single-factor window at ~1e6 magnitude still recovers
  the exact true coefficients (proving the relative threshold isn't overly
  strict), and a genuinely singular (duplicate-column) window at the same
  magnitude still correctly produces `NaN` (proving it isn't accidentally
  *more* permissive at scale than the old fixed threshold was). Existing
  well-conditioned tests confirmed bit-identical against the full native +
  Python suite.
- **Correctness/portability pass, items 6/7/15 of 20:** `adf_test()`
  (backing `engle_granger`) silently clamped any `max_lag` request above
  14 (`kMaxK - 2`, a fixed max-regressor-count constant) with no error or
  warning -- a caller asking for 30 lags of Δy silently got at most 12.
  `kMaxK` is now removed entirely: the ADF regression's `XtX`/`Xty`/`xrow`
  buffers are dynamically sized per candidate lag (`std::vector`, not a
  fixed `double[16*16]`), and the loop's already-existing data-driven
  `if (T < p + 3) break;` is the sole limiter -- a requested `max_lag` is
  now honored up to what the data can actually support, never silently
  truncated below that. Also replaced `gauss_elim`'s fixed absolute
  `< 1e-14` singularity/pivot threshold (shared by both the beta-solve and
  `(X'X)^-1`-diagonal solve inside `ols_normal_eq`) with a relative-epsilon
  threshold (new `numerics::is_negligible_pivot`, scaled to the original
  matrix's own magnitude) -- the same class of large-baseline-tolerance
  gap as the raw-moment cancellation fixes below, just on the singularity
  side rather than the arithmetic side. Structural changes only (same
  formulas, differently-sized/typed buffers) -- verified bit-identical
  against the full existing native + Python test suite; no engle_granger
  and no ADF pinned value changed.
- **Correctness/portability pass, item 4 of 20:** MSVC's `/wd4244`/`/wd4267`
  narrowing-warning suppression was applied target-wide in both
  `_cpp/CMakeLists.txt` and `tests/cpp/CMakeLists.txt`, silencing exactly
  the class of warning that would have caught the `size_t`->`int`
  narrowing bugs fixed in the item above -- in every kernel file, not just
  `bindings.cpp` (the one place pybind11's own API genuinely forces some
  narrowing). Now scoped to `bindings/bindings.cpp` only in the main
  extension target, and removed entirely from every target in
  `tests/cpp/CMakeLists.txt` (none of those link pybind11). A full clean
  MSVC rebuild under the now-unsuppressed `/W3` produces zero
  `C4244`/`C4267` warnings outside `bindings.cpp`, confirming the
  preceding narrowing sweep was thorough.
- **Correctness/portability pass, item 3 of 20:** eliminated `size_t`->`int`
  narrowing (`static_cast<int>(n)` and friends) across `hurst.cpp`,
  `indicators.cpp`, `backtest.cpp`, and `rolling_regression.cpp` -- for a
  series with more than ~2.1 billion elements (`n > INT_MAX`), these casts
  silently wrapped instead of erroring, corrupting loop bounds, comparisons,
  and buffer indices. Bar-count/index variables in these files are now
  `std::size_t` throughout; OpenMP-parallelized loops (`rolling_hurst_into`,
  `stochastic_oscillator_into`) use `long long` induction variables instead
  (MSVC's OpenMP 2.0 canonical-for-loop form requires a signed type, and
  `long long` covers the full practical range where `int` didn't -- matching
  the precedent already set by `backtest.cpp`'s `batch_run_strategy`).
  Values narrowed into a public struct field (`BacktestResult::num_trades`)
  now go through `numerics::checked_narrow_to_int`, which throws instead of
  silently wrapping if that count itself somehow exceeded `INT_MAX`. All
  changes in `hurst.cpp`/`indicators.cpp`/`backtest.cpp` are pure
  reassociation-free refactors (same arithmetic, wider index types) verified
  bit-identical via the existing exact-equality native test suite;
  `rolling_regression.cpp`'s `build_normal_equations`/slide-loop indices
  changed the same way with no formula change.
- **Correctness/portability pass (native ISA dispatch), item 1/2 of 20:**
  `isa_dispatch.cpp` previously (a) would not compile on non-x86
  architectures at all (its CPUID logic was unguarded), and (b) checked
  only CPUID's AVX2/FMA hardware-support bits, not whether the OS had
  actually enabled AVX register-state saving (OSXSAVE + XGETBV/XCR0) --
  some hypervisors/sandboxes report AVX2 hardware support via CPUID while
  leaving that OS-level bit unset, which would have made `rolling_beta`'s
  AVX2 dispatch path unsafe on such a machine. Both fixed: detection now
  collapses to `{false, false}` on non-x86 (guarded behind a new
  `SQT_ARCH_X86` macro) and additionally requires XCR0 bits 1+2 (SSE+AVX
  state) before ever reporting AVX2 available. Also fixed a latent data
  race: `detect_isa_features()`'s test-only override previously stored an
  `IsaFeatures` struct as one non-atomic global paired with only an atomic
  "active" flag -- a real race a ThreadSanitizer build would flag, even
  though today's tests only exercise it sequentially. `detect_isa_features()`
  now returns `IsaFeatures` by value (was `const IsaFeatures&`, source-
  compatible with the one existing call site in `rolling_regression.cpp`),
  backed by independent atomics for the override's `avx2`/`fma` bits.
- **Correctness/portability pass, item 1 of 20:** the AVX2+FMA translation
  unit (`rolling_beta_avx2.cpp`) was unconditionally compiled with
  `-mavx2;-mfma` / `/arch:AVX2` on every platform, including non-x86 --
  those flags are x86-only and a hard compile error on e.g. ARM/Apple
  Silicon toolchains. Both `_cpp/CMakeLists.txt` and the duplicated block
  in `tests/cpp/CMakeLists.txt` now gate those flags behind
  `CMAKE_SYSTEM_PROCESSOR` matching an x86/x64 pattern; the file itself
  also gained a source-level `#if` architecture guard so it compiles to a
  portable (unreachable -- `isa_dispatch` always reports AVX2 unavailable
  on non-x86) stub on any other architecture, since flag-gating alone
  doesn't make AVX2 intrinsics compile without the matching codegen flags.

### Changed

- **Breaking:** `fill_price="midpoint"` renamed to `fill_price="hl2_exploratory"`
  everywhere (`run_strategy`, `run_portfolio_simulation`, `run_pair_backtest`,
  their agent-tool input models, and docs) — it was never a real bid/ask
  midpoint (just `(High+Low)/2`), and the old name implied a market-quote
  guarantee it didn't have. Every reference now carries an explicit
  look-ahead-bias caveat.
- **Breaking:** `CustomSignalBacktestInput`/`SignalPanelBacktestInput`'s
  `signal_type` now defaults to `DIRECTION` (values must be exactly -1/0/1)
  instead of `SCORE` (unrestricted float, multiplied directly into position
  size) — `SCORE` is raw leverage, not a bounded confidence value, and was an
  unsafe default for anyone passing an un-normalized signal.
- `run_pair_trade_backtest`'s `fill_price` now defaults to `"next_open"`
  instead of `"close"` — the z-score signal deciding a transition is computed
  from that same bar's Close, so executing at that same Close was look-ahead
  bias by default. `"close"` is still available for explicit same-bar/
  exploratory analysis.
- `run_backtest_optimization` (the `backtest_grid` agent-tool wrapper) now
  threads `commission_pct`/`slippage_pct` into every grid combination
  instead of silently ignoring them — `backtest_grid` itself already did
  this correctly; the gap was specific to the agent-tool wrapper.
- **Breaking (Tier 4 item 12 of the C++ code review):** all four
  hysteresis signal state machines — `_rsi_state_machine`,
  `_bollinger_state_machine`, `_donchian_state_machine`,
  `_vwap_reversion_state_machine` in `backtest/strategies.py`, plus the C++
  ports `donchian_state_machine`/`vwap_reversion_state_machine` in
  `signal_state_machines.cpp` — now carry the currently-held position
  through a NaN (rolling-warmup) bar in their *output*, instead of
  hardcoding `0.0` regardless of whether a position was actually open. The
  internal `in_pos` state was never touched by a NaN bar in either version
  (that part was already correct); only the emitted value for that bar was
  wrong, previously showing a phantom close/reopen blip in a position
  series that a real caller (or anything downstream reading these signals
  as an actual position, not just a steady-state indicator) would not
  expect — the position was never actually closed. This changes real
  output values for the `donchian_breakout`/`vwap_reversion` (and any
  RSI-/Bollinger-hysteresis-based) strategies wherever NaN warmup bars
  occur alongside an already-open position; confirmed with the user before
  implementing, given the behavior was previously documented as
  intentional in both the Python and C++ docstrings. Updated docstrings in
  both languages and the affected native/Python tests
  (`tests/cpp/test_signals.cpp`, `tests/test_cpp_signals.py`) accordingly,
  including new coverage for the previously-untested "NaN bar while a
  position is already open" case on the VWAP side.
- **Internal:** `src/standard_quant_tools/audit.py` (~1060 lines after
  audit-trail hardening phases 1–2) was split into a package,
  `standard_quant_tools/audit/` (hashing, context, provenance, paths,
  models, storage, writer, verify, redaction, retention, export, signing,
  dispatch, replay), ahead of phase 3 adding more surface area.
  `__init__.py` re-exports the full previous public + semi-private surface,
  so this is a pure internal reorganization — no call site anywhere in the
  codebase or its tests needed to change, and no behavior changed.

### Added

- **Agent tool orchestration: category taxonomy, a lightweight router, and a
  hardened multi-agent orchestrator.** Tool metadata used to be
  hand-duplicated across `get_agent_tools()`'s `tool_defs`, `_TOOL_DISPATCH`,
  and a hardcoded `WORKER_AGENTS` tool-list in
  `Multi_Agent_Implementation/worker_agents.py`, drifting apart silently
  (README/comments variously claimed 34, 42, or 45 tools against a real
  registry of 45). `standard_quant_tools.agent.tools.TOOL_CATEGORY` is now
  the single source of truth — every tool mapped to one of 7 categories
  (`screener`, `analysis`, `quant_research`, `backtest_execution`,
  `backtest_validation`, `custom_signal`, `portfolio_risk`; the former
  16-tool `backtest` bucket split into execution vs. validation, since
  "run this strategy" and "optimize this strategy's parameters" are
  different jobs). `get_agent_tools()` gained an optional `categories`
  filter param, backward compatible (`None` = every tool). Fixed
  `agent/__init__.py`'s stale `__all__`, which predated ~16 real tools.

  New `standard_quant_tools.agent.router`: a provider-agnostic tool-category
  classifier — one cheap completion call narrows the tool list to 1-2
  categories before the real agent loop starts, without spinning up a
  separate agent session. Fails open by design (returns every category on
  any malformed/empty/unparseable response or API error) — a router that
  wrongly excludes a needed tool is worse than today's unfiltered list.
  `route_request()` + an optional `categories` param on `run_agent()` wired
  into every `Implementation/{Anthropic,OpenAI,Gemini}/Agent_*.py` script
  (27 scripts across 3 providers), replacing "hand every tool to the model
  on every call" with "narrow first, then call."

  `Multi_Agent_Implementation/worker_agents.py`'s `WORKER_AGENTS` now
  *derives* each worker's tool list from `TOOL_CATEGORY` instead of a
  hand-duplicated literal list (7 workers now, up from 6, matching the
  execution/validation split); `Agent_Orchestrator.py`'s delegate-tool set
  and system prompt are generated from `WORKER_AGENTS.keys()`/`len()`
  rather than hardcoded counts. Fixed a missing duplicate-log-handler guard
  in `Multi_Agent_Implementation/_agent_utils.py` (present in
  `Implementation/Anthropic/_agent_utils.py`, absent here) that would have
  gotten worse as delegation fans out across more workers.

  New `tests/test_router.py` (unit tests + an `@pytest.mark.integration`
  routing-accuracy eval — the first actual measurement of routing
  correctness in this codebase, vs. the pre-existing multi-agent test's
  coverage/disjointness-only checks) and expanded
  `tests/test_multi_agent_tool_coverage.py` for the 7-worker split. New
  [Documentation/13_agent_orchestration.md](Documentation/13_agent_orchestration.md).

- **3 new agent tools: GARCH volatility forecasting, Kalman dynamic hedge
  ratio, EVT tail risk** (42 → 45 tools). All three model time-varying
  dynamics or fat tails — a gap the analytics layer's existing static/
  point-in-time tools (cointegration, correlation, realized-vol estimators,
  historical VaR/CVaR) didn't cover:
  - `run_garch_volatility_forecast` (`analysis/garch.py`) — fits GARCH(1,1)
    conditional volatility and forecasts it forward, unlike
    `get_volatility_estimators`' backward-looking realized measures. The
    variance recursion is numba-`@njit`'d (inherently sequential, same tool
    `backtest/strategies.py`'s state machines already use); MLE fitting via
    `scipy.optimize` handles millions of bars in well under a second thanks
    to the JIT'd recursion. Requires scipy — no meaningful scipy-free
    fallback for a maximum-likelihood fit.
  - `run_kalman_hedge_ratio` (`analysis/cointegration.py`) — re-estimates a
    pair's hedge ratio every bar via a Kalman filter, a time-varying
    diagnostic companion to `run_cointegration_test`'s static OLS
    `hedge_ratio`. Hand-unrolled 2×2 numba recursion, verified to converge
    to `cointegration_test`'s static hedge ratio as the `delta` tuning
    parameter shrinks toward 0. Deliberately **not** wired into
    `run_pair_trade_backtest`, which still trades a single static hedge
    ratio for the whole window — a real, separate follow-up.
  - `get_tail_risk_metrics` (`metrics/risk_metrics.py`) — Extreme Value
    Theory tail risk via Peaks-Over-Threshold: fits a Generalized Pareto
    Distribution to the worst tail of daily losses and extrapolates
    VaR/CVaR from that fitted tail, reported alongside the naive
    `var_historical` figure for direct contrast. Default fitting method is
    probability-weighted moments (closed-form, pure numpy, zero
    optional-dependency surface); `method="mle"` requires scipy.

  All three follow the established pattern exactly: new Pydantic
  Input/Result models, registration in both `get_agent_tools()` and
  `_TOOL_DISPATCH`, worker assignment + updated system prompt in
  `Multi_Agent_Implementation/worker_agents.py` (verified against
  `test_multi_agent_tool_coverage.py`), and hand-verified pure-function
  tests (GARCH against a simulated known-parameter process; Kalman against
  a hand-computed toy recursion and convergence to static OLS; EVT against
  a known-generating GPD via inverse-CDF sampling) plus structural
  agent-tool tests. See
  [Documentation/09_advanced_agent_tools.md](Documentation/09_advanced_agent_tools.md),
  Tools 26–28.

  Found and fixed a real bug while implementing this: the initial EVT
  probability-weighted-moments estimator had its order-statistic weights
  backwards (weighting by `F(x)` instead of `1-F(x)`), which silently fit
  the wrong tail shape — caught by the known-generating-GPD hand
  verification before it shipped, not by the unit tests alone.

- **4 new backtest strategies** (`backtest/strategies.py`, `STRATEGY_REGISTRY`
  now has 8 entries, up from 4): `donchian_breakout` (Turtle-style channel
  breakout, entry/exit channels use `.shift(1)` so it's a genuine breakout
  past the already-established channel, not a same-bar tautology),
  `momentum_timeseries` (trailing-return threshold, fully vectorized —
  `pandas.Series.pct_change`, no per-bar state at all), `vwap_reversion`
  (mean reversion to a rolling VWAP rather than a plain price mean, aimed
  at intraday/tick data), and `adx_trend` (ADX-strength-filtered
  directional trend, a single vectorized boolean condition on the existing
  `adx()` indicator's output). Every hysteresis-based strategy
  (`donchian_breakout`, `vwap_reversion`, matching the existing
  `rsi_mean_reversion`/`bollinger_reversion` pattern) runs its entry/exit
  tracking through a numba-JIT state machine — verified to complete in
  well under a second on 500k-bar synthetic series in
  `tests/test_strategies.py::TestScalesToLargeSeries`, with no interpreted
  Python loop over the series regardless of length. The other two need no
  state machine at all. All four are immediately usable through every
  entry point that already accepted a `STRATEGY_REGISTRY` name generically
  (`backtest_grid`, `get_backtest_diagnostics`, `run_backtest_compact`,
  `run_backtest_optimization`, `run_walk_forward_backtest`,
  `get_robustness_diagnostics`) — updated their Pydantic field
  descriptions accordingly. They do **not** get dedicated `run_*_backtest`
  tools (only the original 4 do) and are **not** added to
  `compare_strategies`' fixed four-strategy comparison or
  `run_regime_adaptive_backtest`'s curated 3-way regime→strategy map —
  both deliberate scope boundaries, not oversights. See
  [Documentation/04_backtesting.md](Documentation/04_backtesting.md).

  Registering the 4 new strategies surfaced a real, unrelated gap:
  `run_regime_adaptive_walkforward_backtest` (unlike the single-shot
  `run_regime_adaptive_backtest` above) iterates the *entire*
  `STRATEGY_REGISTRY` every window trying all of them, so it immediately
  `KeyError`'d on the first new strategy name via
  `_DEFAULT_PARAM_GRIDS[strat_name]` — that dict only had the original 4
  entries. Fixed by adding default grids for all 4 new strategies and
  changing `grid_overrides[strat_name]` to `grid_overrides.get(strat_name)`
  (the per-strategy override fields on `RegimeAdaptiveWalkForwardInput`
  only exist for the original 4; newer registry entries fall through to
  their default grid, same as any future addition would without a
  matching Pydantic field) — caught by
  `tests/test_new_agent_tools.py::TestRegimeAdaptiveWalkForwardBacktest`,
  not discovered after the fact.

- **Portfolio optimization** (`portfolio/optimize.py`): `mean_variance_optimize`
  (Markowitz mean-variance — `max_sharpe`/`min_volatility`/`target_return`/
  `target_volatility`), `risk_parity_weights`, and `black_litterman` (plus
  `build_bl_views`, a convenience for turning a plain-dict view list into the
  `(P, Q, Omega)` matrices `black_litterman` expects). The unconstrained
  mean-variance case (`allow_short=True`, `max_weight=None`) is solved in
  closed form via the standard Merton (1972) two-fund efficient-frontier
  parametrization — numpy only, no solver dependency, `converged` is always
  `True`. Any long-only and/or weight-capped request uses scipy (SLSQP),
  following the same "scipy optional, clear error if needed and missing"
  convention `metrics.risk_metrics.var_parametric` already established; a
  genuinely infeasible constrained request (e.g. an unreachable
  `target_return` under a `max_weight` cap) reports `converged=False` rather
  than a silently wrong answer. `risk_parity_weights` is a documented
  heuristic (damped multiplicative fixed-point iteration) — not a
  globally-convergence-proven algorithm like the mean-variance closed form —
  and reports its own `converged` flag honestly; verified against a
  diagonal-covariance closed-form case (inverse-volatility weighting) in
  tests. New agent tool `run_portfolio_optimization`
  (`PortfolioOptimizationInput`/`Result`, `BLViewInput`), registered in
  `get_agent_tools()`/`dispatch()` and assigned to the multi-agent
  orchestrator's Portfolio Risk & Sizing worker. This closes the gap
  `backtest/sizing.py`'s own docstring flagged: every other portfolio-facing
  tool only *scored* weights already chosen; nothing *produced* them. See
  [Documentation/05_portfolio.md](Documentation/05_portfolio.md#portfolio-optimization).

- **Options pricing, Greeks & implied volatility** (`analysis/options.py`):
  `black_scholes_price`/`black_scholes_greeks` (Black-Scholes-Merton,
  European options only, `dividend_yield` covers the Merton 1973 continuous-
  dividend extension) and `implied_volatility` (Newton-Raphson with a
  bisection fallback over a practical `[1e-6, 5.0]` bracket, plus a
  no-arbitrage bound check before solving). Dependency-free: the standard
  normal CDF/PDF are computed via `math.erf` (stdlib), not scipy. Every
  Greek is cross-validated in tests against a finite-difference derivative of
  `black_scholes_price` itself (not just checked against the textbook
  formula), and pricing matches Hull's published reference example exactly.
  Two new agent tools, `get_option_pricing` (price + all five Greeks in one
  call) and `get_implied_volatility`
  (`OptionPricingInput`/`Result`/`OptionGreeks`,
  `ImpliedVolatilityInput`/`Result`), registered in
  `get_agent_tools()`/`dispatch()` and assigned to the multi-agent
  orchestrator's Technical & Risk Analysis worker (Greeks are risk
  sensitivities). `get_agent_tools()` now returns 42 tools, up from 39. See
  [Documentation/12_options.md](Documentation/12_options.md) (new file).

- `data.polygon_provider.PolygonProvider`: a third `DataProvider`
  implementation, backed by Polygon.io's plain REST API — no vendor SDK to
  install, just an API key (`SQT_POLYGON_API_KEY`, no default; get a free
  one at https://polygon.io/dashboard/api-keys). Supports `1m`/`5m`/`15m`/
  `30m`/`60m`/`1d`/`1wk`/`1mo`/`3mo` bars via the Aggregates (Bars) endpoint
  (other intervals raise `ValidationError`); `get_financial_ratios` derives
  `trailing_pe`/`price_to_book`/`debt_to_equity`/`return_on_equity`/
  `profit_margins` from the most recent Financials vX filing combined with
  `market_cap` from Ticker Details v3 — `forward_pe` and `dividend_yield`
  are always `None` (no forward estimates or dividend-history aggregation
  in scope). Wired into `DataFactory.get_provider("polygon", api_key=...)`,
  replacing the old `NotImplementedError` stub. See
  [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#polygonio-provider).
- **Audit trail hardening, phase 3 (Ed25519 checkpoint signing + pluggable
  storage backend):** `audit.generate_keypair()`/`checkpoint_and_sign()`/
  `verify_checkpoint_signature()` add an optional external anchor closing
  the one gap the hash chain can't close on its own — an attacker who
  consistently rewrites an entire day file *and* its chain-index entry to
  stay internally self-consistent. A signed checkpoint
  (`{date, final_record_hash, index_hash, signed_at_utc}`) is verifiable
  with only the public key, no trust in the JSONL files' own consistency
  required. Requires the new optional `cryptography` dependency
  (`pip install standard_quant_tools[signing]`, a new `signing` extra in
  `pyproject.toml`); every other audit-trail feature keeps working without
  it, and calling a signing function without it installed raises a clear
  `ImportError` instead of a confusing traceback (same pattern as the
  `bloomberg` extra). Signing key: pass a `signer` callback (routed through
  an HSM/KMS) for anything beyond local development, or `key_path`/
  `SQT_AUDIT_SIGNING_KEY_PATH` pointing at a raw key file —
  `generate_keypair()`/`sqt keygen` are explicitly labeled local-development
  only, not a production key-custody solution. New `sqt keygen`/
  `sqt anchor <date>`/`sqt verify --checkpoint <date> --pubkey PATH` CLI
  subcommands.

  Also introduces a pluggable `AuditStorageBackend` interface behind
  `AuditWriter`; `LocalFilesystemBackend` (the only implementation shipped)
  is a like-for-like move of the previous direct-filesystem behavior behind
  that interface, not a new capability — it's a seam so a future WORM
  backend (S3 Object Lock, Azure Immutable Blob) could be substituted later
  without touching `AuditWriter`'s chain-hashing/locking logic. Building
  that backend is explicitly out of scope for this round.

  28 new tests across `tests/test_audit_signing.py` (18) and
  `tests/test_audit_storage.py` (5, including a fake in-memory backend that
  proves the interface is a real seam, not a passthrough wrapper) plus 5 new
  `sqt keygen`/`sqt anchor`/`sqt verify --checkpoint` CLI tests in
  `tests/test_cli.py`. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#checkpoint-signing-ed25519).

- **Audit trail hardening, phase 2 (retention, legal hold, sealing,
  redaction, export bundle):** `audit.hold_day()`/`release_hold()`/
  `is_held()` place/remove a legal/retention hold sidecar
  (`<date>.jsonl.hold`) on a calendar day. `gc_candidates()`/`gc()` delete
  day files past `SQT_AUDIT_RETENTION_DAYS` (or an explicit
  `retention_days` param) — held days are always excluded, deletion never
  happens automatically (`dry_run=True` by default, only ever triggered
  explicitly via `sqt gc --confirm`), and an unset retention window means
  never delete. Deleting a day file this way is real and permanent, and —
  by design, not by bug — `verify_audit_trail_integrity()` will correctly
  report it as "likely deleted" afterward, same as it would for tampering;
  the chain has no way to tell the two apart, so treat your own
  gc-invocation log as the record of *why*. `seal_day()` chmod's a day file
  read-only as an operational safeguard against accidental writes —
  explicitly not WORM. `SQT_AUDIT_REDACT_FIELDS` (comma-separated dotted
  field paths) replaces matching `input` fields with a non-reversible
  content-hash placeholder before a record is written, so redacted values
  stay comparable across records without the raw value ever touching disk.
  `export_bundle()` zips a date range of day files, the chain index, a
  manifest (per-file SHA-256, record counts, provenance), a copy of
  `scripts/verify_audit_log.py`, and verification instructions into one
  auditor-ready archive. New `sqt hold`/`sqt release-hold`/`sqt gc`/
  `sqt seal`/`sqt export` CLI subcommands. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#retention-legal-hold-sealing-and-export).
- **Audit trail hardening, phase 1 (cross-day chain continuity, durability,
  `sqt verify`):** decision records were previously hash-chained only
  *within* one day's JSONL file — deleting an entire day's file outright was
  undetectable. `audit.py` now maintains an independent, self-hash-chained
  witness log (`_chain_index.jsonl`) at the audit-dir root, one entry per
  calendar day with any activity; the first record of a new day commits to
  the previous active day's last hash via this index (correctly bridging
  gaps like weekends without a false positive), so an attacker now has to
  rewrite both the day file and the index, consistently, to hide a deletion.
  New `verify_audit_trail_integrity()` checks the full trail (the index's
  own chain, index-vs-on-disk day files in both directions, and each day
  file reseeded with the index's claimed starting hash); the existing
  `verify_audit_log_integrity()` gained an optional `expected_prev_hash`
  param (default unchanged) so it can be seeded that way. Every write — a
  decision record or a chain-index entry — is now followed by `f.flush()` +
  `os.fsync(f.fileno())` before its lock is released, unconditionally, so a
  record isn't lost to a crash immediately after `dispatch()` returns.
  New `sqt verify [--file PATH]` CLI subcommand (full trail by default,
  single file with `--file`; exit 0 clean / 1 problems found). New
  `scripts/verify_audit_log.py`: a deliberate, stdlib-only reimplementation
  of the same hashing/chain-walking logic (no `pydantic`/`pandas`/`numpy`,
  no package install) so an external auditor can verify an exported log
  bundle independently; `tests/test_standalone_verifier.py` is a parity
  test that fails if the two implementations' hash output ever diverges.
  Pre-existing audit directories need no migration — old day files stay
  independently valid, and cross-day linkage begins transparently at the
  next new-day write. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#what-this-can-and-cannot-certify)
  for what this does and does not certify — it is a tamper-*detection*
  control, not tamper prevention or regulatory certification by itself.
- `data.bloomberg_provider.BloombergProvider`: a second `DataProvider`
  implementation, backed by a local Bloomberg Terminal via Desktop API
  (`blpapi`, a new optional dependency — `pip install
  standard_quant_tools[bloomberg]`). No API key (DAPI authenticates via the
  Terminal login); `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` are the only
  configurable, non-secret connection settings. Daily/weekly/monthly bars
  only (intraday raises a clear `ValidationError`, not wrong data). Wired
  into `DataFactory.get_provider("bloomberg")`, replacing the old
  `NotImplementedError` stub. See
  [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#bloomberg-provider).
- `standard_quant_tools.config.load_env()`: a single choke point for
  loading `.env` (via the new `python-dotenv` core dependency) into
  `os.environ`, idempotent per process, never overriding a real environment
  variable — the same mechanism whether config comes from a local `.env`
  file or CI/CD secrets (GitHub Actions / GitLab CI) injected as real env
  vars. `.env.example` documents every variable and both platforms' secrets
  syntax.
- `data/_retry.py`: extracted the retry-with-backoff decorator out of
  `yfinance_provider.py` into a shared module so `BloombergProvider` doesn't
  duplicate it; `yfinance_provider.py`'s behavior is unchanged (verified —
  same tests, same results).
- `audit.py`: a hash-chain (`prev_record_hash`/`record_hash` on every JSONL
  decision record) and `verify_audit_log_integrity()`, so the audit log
  itself is tamper-evident, not just each record's replay. JSONL writes are
  now guarded by a cross-process advisory lock (`msvcrt` on Windows,
  `fcntl.flock` on POSIX; falls back to unlocked with a debug log if neither
  is available, rather than blocking a tool call on a missing OS primitive).
- `verify_replay()` now reports data sources that disappeared between the
  original call and the replay (previously silently dropped from the
  comparison).
- `screener.py` now reports fetch/filter failures via `DataFrame.attrs`
  (`failed_filters`, `failed_tickers`, and `failed_batches` for the
  multi-worker path) instead of returning `None` — previously
  indistinguishable from a ticker that legitimately didn't pass a filter.
- Project governance: Apache 2.0 `LICENSE`/`NOTICE`, `SECURITY.md`,
  `CONTRIBUTING.md`, this `CHANGELOG.md`, license/URL metadata in
  `pyproject.toml`, and a local `v0.1.0` release tag.
- `black`/`isort` now actually pass in CI — added shared `[tool.black]`/
  `[tool.isort]` config (`profile = "black"`) and reformatted the full
  `src/`/`tests/` tree, which had never matched the CI check before.
- `_sqt_core` (the optional C++ extension) gained four more kernels, found
  by auditing everything added to the library since its last porting pass:
  `simulate_forward_paths` (Monte Carlo moving-block bootstrap — the only
  genuinely unaccelerated loop found, not even numba-decorated, and
  embarrassingly parallel, so it also gets an optional OpenMP path on top
  of the usual compiled-vs-interpreted speedup),
  `garch11_variance_recursion`, `kalman_filter_1state`/`kalman_filter_2state`
  (added to the existing `cointegration.cpp` rather than a new file), and
  `donchian_state_machine`/`vwap_reversion_state_machine`. The latter three
  were already numba-JIT'd and confirmed fast once warm — ported for the
  same permanent reason already documented for RSI/ADX/PSAR: no JIT
  cold-start latency on a fresh process (measured at 200ms–1.1s, not the
  initial ~300–500ms estimate — see below), and immunity to future numpy
  ABI breakage. Every port keeps the existing pure-Python/numba fallback as
  the default when `_sqt_core` isn't built, and follows the same
  `HAS_CPP`/`_cpp_core` guard pattern as the rest of the extension. All four
  were subsequently built and their full test suites actually run (see the
  build-verification entry below) — real numbers, not projections, are in
  `Development/performance_insights.md`.
  **Behavior note:** the Monte Carlo C++ path's RNG does not reproduce
  NumPy's PCG64 bit stream, so `random_seed` is only reproducible *within*
  one backend — the same seed gives different concrete numbers depending on
  whether `_sqt_core` is built (still bit-identical on repeat calls within
  one backend). See `Development/performance_insights.md` and
  `Development/build_guide.md` for the full detail.

- **C++ hardening, Tier 3 item 9 of an independent code review:** every
  `_sqt_core` binding (all ~21 `m.def(...)` entries in `bindings.cpp`) now
  releases the GIL (`py::gil_scoped_release`) around just the `sqt::` kernel
  call itself — extracting raw pointers/sizes/plain-C++ arguments from the
  `py::` types first (while still holding the GIL, since buffer access and
  argument casting are Python-API calls), then letting multiple Python
  threads run the actual C++ computation concurrently instead of
  serializing on the GIL for work that never touched a Python object once
  argument extraction was done. Added `tests/test_cpp_gil_release.py`: a
  concurrency smoke-test suite (multiple threads hammering `rsi`,
  `run_strategy`, `hurst_dfa`, `bollinger_bands`, and a mixed-kernel
  scenario at once), each thread's result checked against its own
  single-threaded reference rather than attempting to prove GIL-release
  timing from Python.
- **C++ hardening, Tier 3 item 10:** `-march=native`/`/arch:AVX2` (tuning
  codegen for the exact build machine's CPU, not portable to a
  different/older one) is now opt-in via a new `SQT_NATIVE_ARCH` CMake
  option (default `OFF`) instead of always-on in Release builds — applies to
  both `_cpp/CMakeLists.txt` (the actual extension) and `tests/cpp/
  CMakeLists.txt`'s `bench_hurst`/`bench_backtest` targets. A default build
  (what CI and a fresh clone both use) now produces portable codegen; pass
  `-DSQT_NATIVE_ARCH=ON` for the extra local-dev speed this session's own
  measured benchmarks in `performance_insights.md` were built with (no
  re-benchmarking needed — the numbers already reflect `SQT_NATIVE_ARCH=ON`).
  Verified both configurations build clean and pass the full native ctest
  suite + Python suite.

### Added

- **Deep native optimization, item L: runtime ISA dispatch demo (AVX2+FMA,
  `rolling_beta` only).** New `include/sqt/isa_dispatch.hpp` +
  `src/isa_dispatch.cpp`: lazily-detected, thread-safe (C++11 magic static)
  `IsaFeatures{avx2, fma}` via CPUID (`__cpuid`/`__cpuidex` on MSVC,
  `__get_cpuid`/`__get_cpuid_count` on GCC/Clang), plus a test-only override
  hook (`force_isa_features_for_testing`/`reset_isa_features_override_for_testing`)
  — the only practical way to exercise the "runs correctly on a non-AVX2
  CPU" path without physical access to one. Deliberately scoped to **one
  kernel, AVX2 only** (not AVX-512, per the review's own caveat that
  AVX-512 isn't automatically faster) — `rolling_beta_into`'s 4-accumulator
  window reduction (`Sx`, `Sy`, `Sxy`, `Sxx`), chosen as the same reduction
  item C's SIMD-pragma attempt already targeted. New
  `src/rolling_beta_avx2.cpp` (`rolling_beta_reduce_avx2`) in its own
  translation unit, compiled unconditionally with AVX2+FMA codegen enabled
  via `set_source_files_properties` (`/arch:AVX2` MSVC, `-mavx2 -mfma`
  GCC/Clang) — **independent of the opt-in `SQT_NATIVE_ARCH` flag**, since
  MSVC has no per-function ISA-target attribute (unlike GCC/Clang's
  `__attribute__((target(...)))`), so isolating the intrinsics into their
  own file is the only portable way to keep the rest of the module's
  codegen safe on non-AVX2 CPUs when `SQT_NATIVE_ARCH=OFF`. Runtime safety
  comes entirely from `isa_dispatch.cpp`'s CPUID check gating every call
  into this file, not from the compile flag. `rolling_beta_into` dispatches
  once per call (not per window) based on `detect_isa_features().avx2`.
  **Not bit-identical to the scalar path** (SIMD lane accumulation reorders
  the sum) — tolerance-gated (`1e-6` absolute, `.hurst`-style bounded
  quantity) against the scalar path forced via the test override hook,
  across normal data, a large-baseline case (same cancellation-risk shape
  as `rolling_beta`'s existing large-baseline fix), a window not a multiple
  of 4 (exercises the AVX2 kernel's scalar tail), and window==n. A separate
  forced-scalar-path test confirms the scalar fallback alone still recovers
  a known slope exactly. Measured (min of 15 runs, real dispatch vs. the
  same test-forced scalar path, not a projection): **n=2000/window=60:
  ~1.50×**; **n=20000/window=60: ~1.10×** — modest, honestly reported gains
  for a single reduction kernel, not the dramatic win a wholesale
  multi-kernel AVX2/AVX-512 rewrite might chase (explicitly out of this
  item's scope, per its own spec, to avoid the scope creep the review's own
  caveat about AVX-512 warned against).
- **Deep native optimization, item K: opt-in, local-only PGO (Profile-Guided
  Optimization) build workflow.** New `SQT_PGO_GENERATE`/`SQT_PGO_USE`
  CMake options (default `OFF`, mutually exclusive — `FATAL_ERROR` if both
  set), mirroring `SQT_NATIVE_ARCH`'s existing "opt-in for local max speed"
  philosophy. MSVC: `/GL` + `/LTCG:PGInstrument` / `/LTCG:PGOptimize`.
  GCC/Clang: `-fprofile-generate` / `-fprofile-use -fprofile-correction`.
  Documented the 2-step local workflow in `Development/build_guide.md`
  (instrumented build → train against a representative workload → optimized
  rebuild), including a real gotcha discovered while writing it: every
  CMake build directory in this repo writes `_sqt_core` to the same
  absolute package path regardless of which directory produced it, so a
  PGO experiment silently overwrites your normal working extension unless
  you use a separate build directory and rebuild the normal one afterward
  — confirmed by actually doing this, not just reasoned about (an
  `SQT_PGO_GENERATE=ON` test build in a separate `build-pgo-test/` dir did
  overwrite the real extension, caught by `python -c "from
  standard_quant_tools import _sqt_core"` still loading successfully but
  being the wrong build, then restored by rebuilding `build/` normally).
  **Explicitly not wired into any CI workflow** — same reasoning
  `SQT_NATIVE_ARCH` already documents for itself, now doubled by PGO's own
  two-build-step requirement not fitting a simple CI pipeline.
  **Verification:** the default-OFF path (unaffected — confirmed full
  native ctest + full pytest green on a normal build after the CMake
  changes) is the real gate here; separately confirmed `SQT_PGO_GENERATE=ON`
  actually configures and builds successfully on this project's MSVC
  toolchain (not just assumed from the flag names).

### Not shipped

- **Deep native optimization, item J: rank-1 Cholesky update/downdate for
  `rolling_factor_loadings` — attempted, hard numerical-stability gate
  failed, reverted.** Implemented the standard Givens-rotation-based
  Cholesky update/downdate (Golub & Van Loan §6.5.4 — the same algorithm
  LINPACK's `dchud`/`dchdd` and MATLAB's `cholupdate` implement) to
  maintain the Cholesky factor `L` directly in O(p²) as bars enter/leave
  the rolling window, instead of `cholesky_solve()`'s O(p³) full refactor
  every step (with a downdate-failure fallback reusing the existing
  `refresh` cadence). Gated it — per this project's established
  hard-gate-with-escape-hatch pattern — behind a comparison against the
  existing full-refactor-per-step path (same-machine `git stash`/`git
  stash pop`, real before/after output, not a re-derived reference) across
  8 configs: `k` = 3, 10, 30, 50 on well-conditioned random data, `k` = 5
  and 10 on deliberately near-singular/collinear factor data (one column a
  near-duplicate of another), and a large-baseline-offset case (`+1e6` on
  every factor value, small relative variation per window — the same
  shape of numerical stress that motivated `rolling_beta`'s own
  large-baseline fix). **Result: well-conditioned data agreed to
  ~1e-13–3e-10 relative tolerance (excellent) across every `k` tested, but
  the near-singular/collinear cases showed max relative differences of
  ~30× and ~1.2× (i.e., a real, not marginal, numerical breakdown), and the
  large-baseline case showed a ~5.3% relative difference.** This matches
  exactly the risk this item's own spec flagged going in — Cholesky
  *downdate* is a well-known harder numerical problem than update, most
  fragile precisely where the periodic full-refactor safety net matters
  most. **Per the documented escape hatch, this was reverted — not
  shipped.** `rolling_factor_loadings_into` still benefits from items A/B
  above (dead upper-triangle removal, `cholesky_solve` scratch reuse),
  unconditionally safe and independent of this item. The implementation
  and its gate results are documented here rather than silently dropped,
  matching this project's "record the real outcome, including a
  disappointing one" standard (e.g. the GARCH gradient's documented
  tolerance-loosening, Phase 3's LTO/IPO null result).
- **Deep native optimization, Phase 5: `SQT_RESTRICT` portable `restrict`
  qualifier across every `_into` kernel.** New `include/sqt/platform.hpp`
  (`__restrict` on MSVC, `__restrict__` on GCC/Clang). Applied to all 12
  confirmed `_into`-style functions' pointer parameters (both `.hpp`
  declaration and `.cpp` definition): `rolling_hurst_into`,
  `rolling_factor_loadings_into`, `rolling_beta_into`,
  `simulate_forward_paths_into`, `garch11_variance_recursion_into`,
  `donchian_state_machine_into`, `vwap_reversion_state_machine_into`,
  `rsi_into`, `adx_into`, `parabolic_sar_into`, `wilder_atr_into`,
  `bollinger_bands_into`, `stochastic_oscillator_into`. Audited every call
  site in `bindings.cpp` first (not assumed): each one's `out` buffer is
  always a freshly-constructed `py::array_t<double>` immediately before the
  call, never derived from or aliased with any input array — the
  non-aliasing contract `restrict` promises genuinely holds. `backtest.cpp`'s
  `run_strategy_summary`/`batch_run_strategy` were deliberately left out of
  this item's scope — they have no output-pointer parameter at all (return
  by value), so the usual "protect a written buffer from being
  conservatively treated as possibly-aliased with the inputs" restrict use
  case doesn't apply the same way there. Full native ctest + full pytest
  passed unchanged as the correctness gate (pure codegen hint — no behavior
  change possible if the aliasing audit is correct). Measured honestly, not
  assumed: `rsi`/`adx`/`rolling_factor_loadings`/`run_strategy` (n=2000)
  showed **no measurable difference** on this MSVC build — consistent with
  MSVC's optimizer historically extracting less benefit from `__restrict`
  than GCC/Clang; kept anyway as a correctness-neutral hint that may help on
  other compilers, matching this item's own documented expectation rather
  than an assumed win.
- **Deep native optimization, Phase 4 (`hurst.cpp`): one-pass DFA
  reformulation.** New internal `dfa_onepass()`, used only by
  `hurst_exponent_scratch()`'s "dfa" branch — the public `dfa()`/`dfa_impl()`
  stay on the original 3-pass arithmetic permanently, so this genuine
  reassociation never touches the standalone-tested public function.
  Collapses `dfa_impl()`'s 3 passes per chunk (mean, cross-product, residual
  sum-of-squares) into 1, using two algebraic identities that are exact at
  the OLS optimum, not approximations: the cross-product's `seg_mean`
  cross-term cancels algebraically (`cross = Σ(j·y) - x_mean·Σy`, since
  `Σ(j-x_mean) == 0` on the fixed integer grid), and the residual
  sum-of-squares reduces to the standard OLS sufficient-statistics identity
  `SSE = Σy² - a·Σy - b·Σ(j·y)`. `x_var` also replaced with its closed form
  `(sz²-1)/12` instead of a per-window-size loop. **Hard numerical-stability
  gate, not assumed bit-identical** (sum-of-squares-style accumulation is,
  in general, less robust to catastrophic cancellation than the original's
  deviation-from-mean style): tested `rolling_hurst`'s "dfa" output against
  the unchanged public `hurst_exponent()` at `rel`≈`1e-9` absolute tolerance
  on ordinary series, plus a dedicated adversarial test at `1e-6` absolute
  tolerance against deliberately ill-conditioned inputs (a strongly-trending
  series — large-magnitude, near-linear cumulative sum after DFA's own
  Step-1 transform — and a near-constant series with tiny variance). **Gate
  passed cleanly** on every tested case, so this is wired in (the
  alternative — keeping Phase 3b's scratch-reuse-only 3-pass path — was the
  documented fallback if it hadn't). Measured (min of 7 runs, isolating this
  item's effect on top of Phase 3b's OpenMP+scratch baseline): **n=1000:
  0.90ms → 0.78ms, ~1.15×**; **n=2000: 2.68ms → 1.45ms, ~1.85×**;
  **n=5000: 6.92ms → 3.81ms, ~1.82×** — combined with Phase 3b,
  `rolling_hurst` is now **~5.2×, ~10.5×, ~10.7×** faster than the original
  fully-serial 3-pass-per-chunk baseline at these three sizes respectively.
- **Deep native optimization, Phase 3b (`hurst.cpp`): OpenMP across
  `rolling_hurst`'s window loop + scratch-buffer reuse.** `dfa()` split
  into a shared `dfa_impl(..., y_scratch)` — `y_scratch == nullptr`
  reproduces the exact original always-allocate-locally behavior, so the
  public, standalone-tested `dfa()` is now a one-line wrapper with
  byte-identical behavior in every way that matters. New internal
  `hurst_exponent_scratch()` mirrors `hurst_exponent()` but reuses a
  per-thread `RollingHurstScratch{ y }` buffer across every window that
  thread processes (the "rs" method branch is unchanged — not worth the
  complexity for its small `n_points`-bounded vectors). `rolling_hurst_into`
  now runs the window loop under `#pragma omp parallel` + `#pragma omp for`,
  one scratch buffer constructed per thread. The loop originally incremented
  by a runtime `step` value (not unit stride); rather than assume OpenMP's
  canonical-loop-form permits this cleanly on every targeted compiler (no
  local precedent — `monte_carlo.cpp`'s only prior OpenMP loop is
  unit-stride), rewrote it as a counted loop (`idx` in `[0, count)`,
  `i = window-1 + idx*step`) — confirmed to build correctly on this
  project's MSVC toolchain either way, so this was a deliberate
  robustness choice, not a workaround for an actual failure. Verified two
  ways: (1) `rolling_hurst`'s output exactly matches calling the unchanged
  public `hurst_exponent()` directly on the same window slice, for both
  "dfa" and "rs" methods plus a non-evenly-dividing step size that
  exercises the counted rewrite's boundary math — isolates the
  scratch-reuse and OpenMP risk surfaces from each other by checking
  against an independent, already-trusted code path rather than only
  self-consistency; (2) exact reproducibility across
  `OMP_NUM_THREADS=1/2/4/8`. Measured (min of 7 runs, same-machine
  before/after, 16 logical cores): **n=1000/window=100: 4.05ms → 0.90ms,
  ~4.5×**; **n=2000/window=200: 15.28ms → 2.68ms, ~5.7×**;
  **n=5000/window=200: 40.77ms → 6.92ms, ~5.9×**.
- **Deep native optimization, Phase 3 (build): LTO/IPO enabled for Release
  builds.** `_cpp/CMakeLists.txt` now runs `CheckIPOSupported` and applies
  `INTERPROCEDURAL_OPTIMIZATION_RELEASE` automatically when the toolchain
  supports it — unlike `SQT_NATIVE_ARCH`, this carries no "illegal
  instruction on a different CPU" portability risk (link-time only, doesn't
  change the target ISA), so it's not gated behind an opt-in flag. Scoped to
  Release only, same as the existing `/O2`-vs-`/Od` split. Full native
  ctest + full pytest passed unchanged as the actual correctness gate (LTO
  can in principle shift FP instruction selection under whole-program
  visibility; no regression surfaced). Measured honestly, not assumed:
  clean-build time on this (small, 9-source-file) extension is unaffected
  either way (~5.7-6.0s, noise-level difference); a handful of representative
  kernels (`rsi`, `adx`, `rolling_factor_loadings`, `run_strategy`, n=2000)
  showed **no measurable runtime difference** (~1.0× across the board) —
  each kernel's hot loop already lives entirely within its own translation
  unit, so there wasn't much cross-TU inlining opportunity for LTO to
  exploit in this codebase's current structure. Kept anyway since it's a
  free, correctness-neutral toolchain improvement with no measured downside,
  matching the review's own framing ("percentages, not multiples... low-effort").
- **Deep native optimization, Phase 2 (`backtest.cpp`): allocation-free
  summary kernel + OpenMP across the batch grid.** New `run_strategy_summary()`
  computes `run_strategy()`'s 11 scalar metrics with zero heap allocation at
  all (no `equity_curve`, no `strat_ret`, no `trade_rets` vector), exploiting
  a fact discovered during verification: `strat_ret[i]` has no true
  loop-carried dependency (`exec_i = signals[i-1]` and the `prev_exec`
  needed for `pos_diff` equals `signals[i-2]`, or 0.0 for `i==1`, both
  directly index-derivable) — only the trade-log open/close bookkeeping is
  a genuine sequential state machine. Two passes: pass 1 fuses that state
  machine with running equity/peak/drawdown/mean tracking (trade stats
  accumulated as running scalars instead of a `trade_rets` vector); pass 2
  recomputes `strat_ret[i]` on demand, now that the mean is known, to get
  variance and downside deviation. Verified bit-identical against
  `run_strategy()`'s 11 fields across 40 random `(n, prices, signals,
  commission, slippage)` trials plus edge cases (`n==0`, `n==1`, all-flat,
  all-short, leveraged/non-±1 signals, zero-price bars) — the design
  guarantees this by construction (same formulas, same op order, index-0's
  implicit `strat_ret[0]=0.0` contribution to the variance sum seeded
  directly since `0.0 + x == x` exactly in IEEE 754), and the new test is
  what actually proved it held.
  `batch_run_strategy` now calls `run_strategy_summary` directly (no more
  manual `equity_curve.clear()/shrink_to_fit()` after the fact) and runs
  every test index in parallel via `#pragma omp parallel for` — each call is
  a pure function of its own `(prices, signals_flat + t*n, n, ...)` slice
  with no shared mutable state, so (unlike `simulate_forward_paths_into` in
  `monte_carlo.cpp`, which needs a thread-local RNG) no per-thread setup is
  needed, just the simpler combined form. `results` switched from
  `reserve()+push_back()` to `resize()`+indexed writes first, since
  `push_back` on a shared vector is not thread-safe across concurrent
  writers. Verified exact reproducibility of `batch_run_strategy`'s output
  across `OMP_NUM_THREADS=1/2/4/8` (every row is fully independent, unlike
  Monte Carlo's per-path-seed reproducibility, so output must be identical
  regardless of thread count, not just per-path-deterministic). Measured
  (`batch_run_strategy`, min of 7 runs, same-machine before/after, 16
  logical cores): **n=500/num_tests=500: 3.26ms → 0.54ms, ~6.0×**;
  **n=2000/num_tests=2000: 51.55ms → 4.55ms, ~11.3×**;
  **n=2000/num_tests=10000: 255.25ms → 29.81ms, ~8.6×**.
- **Deep native optimization, Phase 1 (`rolling_regression.cpp`):** three
  changes to `rolling_factor_loadings`'s per-bar Cholesky solve, following a
  third-party review of what's left in the native layer after the
  performance-architecture pass above. (1) `build_normal_equations()` and
  the rank-1 XtX update/downdate loop computed all p² entries of the
  symmetric normal-equations matrix; `cholesky_solve()`'s decomposition loop
  only ever reads the lower triangle (`j <= i`), so the upper triangle was
  provably dead work — removed outright (`c < p` → `c <= r`), no mirror step
  needed since nothing downstream ever reads those entries. Verified
  bit-identical two ways: a same-machine `git stash`/`git stash pop`
  comparison of `rolling_factor_loadings()`'s full output array (exact `==`,
  not tolerance) on a fixed random `(n=400, k=5, window=30)` input, and a
  new from-scratch independent-reference regression test (dense Gaussian
  elimination on the full normal equations, sharing no code with the
  production lower-triangle-only path). (2) `cholesky_solve()` allocated a
  fresh `L`/`z` vector on every single call — one call per bar in the
  rolling window, so `(n-window+1)` allocations per series. Now takes
  caller-owned `L_scratch`/`z_scratch` buffers, sized once outside the
  loop and reused across every call; traced the read pattern by hand and
  confirmed the old `L(p*p, 0.0)` zero-fill was never actually load-bearing
  (every read of `L` is to an entry the same call already wrote earlier in
  its own iteration order), so the scratch buffer is reused with no re-zero
  needed either — also verified bit-identical via the same two methods.
  (3) Added `#pragma omp simd reduction(+:Sx,Sy,Sxy,Sxx)` above
  `rolling_beta_into`'s 4-accumulator reduction loop as a vectorization
  hint. **First attempt broke the MSVC build**: MSVC's default `/openmp`
  only implements OpenMP 2.0, which doesn't recognize `omp simd` (that's
  4.0+) — this is a hard `C7660` compile error requiring
  `/openmp:experimental`, not the silently-ignored no-op initially assumed;
  scoped the pragma to non-MSVC compilers only (`!defined(_MSC_VER)`)
  rather than pulling in a project-wide experimental-flag change for one
  hint whose payoff is itself unproven. Also added `tests/cpp/test_rolling_regression.cpp`
  (new `sqt_rolling_regression_impl`/`cpp_rolling_regression` CMake target) —
  `rolling_regression.cpp` previously had no native-level test coverage at
  all, only the existing Python-level `tests/test_cpp_regression.py`.
  Measured (`rolling_factor_loadings`, n=2000, window=60, min of 9 runs,
  same-machine before/after): **k=3 (this library's own typical/tested
  factor count) 0.269ms → 0.150ms, ~1.79×** — the allocator overhead from
  item (2) turned out to dominate total cost at this library's actual
  problem size, a bigger and more directly-relevant win than the review's
  own "matters more at k=10-50" framing suggested; **k=10: 1.058ms →
  0.811ms, ~1.30×**; **k=30: 7.452ms → 6.833ms (best of 2 runs), ~1.09×** —
  as p grows, `cholesky_solve`'s O(p³) decomposition dominates total cost
  more, so the O(1)/O(p²) savings from items (1)/(2) become proportionally
  smaller, not larger.
- **Performance architecture, item 6:** two changes, per the review's own
  final priority item. (1) `batch_run_strategy` (`bindings.cpp`) returned
  `py::list` of `py::dict`, one per grid combo; `backtest_grid`
  (`engine.py`) then rebuilt a Python dict per row before handing them to
  `pd.DataFrame`. Changed the binding to return a single `(num_tests, 11)`
  `py::array_t<double>` (fixed column order, `_BATCH_METRIC_COLUMNS` in
  `engine.py`) and `backtest_grid` to build the metrics `DataFrame`
  directly via `pd.DataFrame(arr, columns=_BATCH_METRIC_COLUMNS)`, then
  concat the parameter-combo columns — no per-row dict ever built. Isolated
  micro-benchmark: the binding call itself (array vs list-of-dict
  construction in C++) **~1.21×**; the Python-side `DataFrame`-construction
  step alone (array→DataFrame vs `num_tests` dicts→DataFrame) **~7×**. At a
  1,200-combo end-to-end `backtest_grid()` (n=1,500 bars, the review's own
  "1,000+ combos" scale), the two measured within noise of each other
  (~0.26s either way) — at that grid size the C++ kernel itself (1,200 full
  backtests) dominates wall time, so the marshaling-layer win, while real,
  is a small fraction of the total; it matters more for cheaper
  strategies/shorter series or larger combo counts relative to series
  length, not uniformly at every grid size. (2) New fused
  `sqt::technical_indicators(high, low, close, config)` (`indicators.cpp`)
  computes whichever of {RSI, ADX, ATR, Bollinger Bands, Stochastic
  Oscillator} the caller requests in one native call instead of up to 5
  separate ones — pure orchestration over the same already-tested `*_into`
  kernels from item 5, no new algorithm logic. New `technical_indicators`
  pybind11 binding (`py::dict` of arrays, conditional keys). Wired as an
  additive fast path into `agent/tools.py`'s technical-analysis tool: when
  2+ of {rsi, adx, bollinger, stochastic} are requested (and C++ is
  available), one fused call replaces up to 4 separate Python-wrapper round
  trips; the plain `atr` indicator is deliberately excluded from the fused
  path since the tool's `atr()` uses a simple rolling mean while the fused
  call's ATR field is Wilder-smoothed — a different algorithm, not the same
  one computed faster — so fusing it would have silently changed the tool's
  output. Individual indicator wrappers (`rsi()`, `adx()`, etc.) are
  unchanged and still used standalone elsewhere, and as the fallback when
  fewer than 2 fusable indicators are requested. Verified the fused path
  produces byte-identical `last_values`/`signals` to the per-indicator
  fallback (forced via a `HAS_CPP` monkeypatch) in
  `tests/test_agent_tools.py`. Measured at the actual integration point
  (`get_technical_analysis`, n=2,000 bars, all 4 fusable indicators
  requested): **~4.6×** (1,467µs → 314µs, median of 9 runs) — the win here
  is eliminating 3 of 4 redundant Python-wrapper layers (validation,
  logging, numpy conversion, per-call pandas construction), not a faster
  native kernel; at the raw C++-binding level alone the 4 individual
  bindings vs. 1 fused call measure ~1.0× (n=2,000, ~100µs either way — the
  pybind11 call overhead itself is negligible at this size next to the
  kernels' own O(n) work), consistent with the review's own framing that
  the win comes from removing Python-side glue, not from a faster inner
  loop.
- **Performance architecture, item 5:** ~16 of `bindings.cpp`'s ~21
  bindings shared the pattern `std::vector<double> result = sqt::foo(...);
  py::array_t<double> out(...); std::copy(result.begin(), result.end(),
  out.mutable_data());` — a `std::vector` allocation plus a full copy into
  a second, separately-allocated NumPy array, on every call. Added a
  buffer-writing `*_into` overload alongside 13 of the ~16 identified
  vector-returning `sqt::` functions (`rsi`, `adx`, `parabolic_sar`,
  `wilder_atr`, `bollinger_bands`, `stochastic_oscillator`,
  `rolling_hurst`, `rolling_beta`, `rolling_factor_loadings`,
  `simulate_forward_paths`, `garch11_variance_recursion`,
  `donchian_state_machine`, `vwap_reversion_state_machine`) — the
  existing vector-returning form becomes a thin wrapper (allocate, call
  `_into`, return), so every native test keeps calling the unchanged API
  with zero test churn. `bindings.cpp` now allocates the NumPy output
  array first and passes its buffer straight into the `_into` call: one
  allocation, zero copies. `simulate_forward_paths_into` needed a small
  contract change from the vector-returning form (returns `bool` for
  "was `out` actually written" instead of signaling invalid input via an
  empty vector, since a pre-sized buffer can't itself be "empty") — the
  vector-returning wrapper still preserves the original empty-on-invalid
  contract exactly.
  **Deliberately scoped out**: `run_strategy`'s `equity_curve` field and
  the two Kalman filters' 3-4 output arrays each — these return
  multi-field structs, not a single `std::vector`, so the same pattern
  would need multiple output-buffer parameters per call; lower value
  (Kalman filters aren't hot-loop calls, and `run_strategy`'s own copy is
  already dwarfed by item 1's ~58× wrapper fix) for real added
  complexity, left as a known, documented gap. Measured on two of the
  cheapest kernels at small n (where a copy is proportionally largest):
  `rsi` (n=100) **~1.6×** (0.00429ms→0.00262ms), `adx` (n=100) **~1.9×**
  (0.00886ms→0.00477ms) — same-machine git-stash-verified.
- **Performance architecture, item 4:** `adx()` (`indicators.cpp`)
  allocated 4 full n-sized temporary arrays (`dm_plus`, `dm_minus`, `tr`,
  `dx_vals`) beyond its own output array. Traced Wilder's recursion by
  hand: it only ever needs the immediately-previous smoothed sum plus the
  *current* bar's raw TR/DM value (computable inline, no lookback array
  needed), and the DX/ADX seed windows only need a running sum of the
  values seen so far, not the individual values — so the whole function
  genuinely reduces to O(1) auxiliary memory, not just "smaller."
  Rewrote as a single fused pass preserving the exact same order of
  floating-point operations as the original 4-pass version (addition
  isn't associative, so order — not just which values get summed —
  determines the result). Verified bit-identical output two ways: every
  existing test passed unchanged with zero tolerance widening, and a new
  exact-equality regression pin (`tests/cpp/test_indicators.cpp`) was
  confirmed to match against *both* the pre- and post-rewrite
  implementation via `git stash` in both directions. Measured speed:
  negligible at n=2000 (~1.02–1.07×, within noise — fixed Python/pybind
  call overhead dominates at this size) but a real **~1.21×** at n=50000
  (min 3.18ms→2.63ms) once the eliminated arrays are large enough
  (~1.6MB total) for memory bandwidth/allocation cost to matter against
  the O(n) arithmetic. Memory footprint (5 allocations → 1) improves
  unconditionally regardless of n.
- **Performance architecture, item 3:** `garch_volatility_forecast`'s
  scipy L-BFGS-B fit called `_garch11_neg_loglik` every iteration, which
  dispatched to the C++ recursion for a full `sigma2` array, copied it out
  of C++, then reduced it to one scalar in NumPy — a full array round-trip
  every iteration purely to throw the array away. New
  `garch11_neg_loglik` (C++) fuses the recursion and the NLL reduction
  into one native call returning a single `double`; new
  `garch11_neg_loglik_grad` additionally computes the analytic gradient
  w.r.t. `(omega, alpha, beta)` in the same fused pass, wired via scipy's
  `jac=True` convention so L-BFGS-B stops needing 6 extra
  finite-difference NLL evaluations per iteration. The analytic gradient
  was verified against central differences across 5 random input grids
  before being trusted (`tests/cpp/test_garch.cpp`) — per the plan's own
  gate, this was only wired into the optimizer after that check passed
  cleanly (the first attempt used a single absolute step size across all
  three parameters and failed on `omega`, not because the gradient was
  wrong, but because `omega`'s tiny ~1e-6 scale needs a much smaller step
  than `alpha`/`beta`'s ~0.05–0.95 range; per-parameter-scaled steps fixed
  the numerical reference itself). `garch11_variance_recursion` alone
  (just the recursion, no fusion) still measures 0.8× vs warm numba — the
  fusion is what actually pays off. Measured end-to-end
  `garch_volatility_forecast()`: **~7.8×** (7.928ms → 1.016ms, n=1000,
  same-machine git stash/pop before/after). `jac=True` can converge to a
  very slightly different point than finite-difference gradients near a
  flat likelihood surface (real for GARCH), so
  `TestGarchForecastEndToEndParity` was loosened from bit-identical
  (`abs=1e-10`) to `rel=1e-2` on fitted parameters plus a tight `rel=1e-3`
  check on the two fits' own log-likelihoods — the actual invariant that
  matters.
- **Performance architecture, item 2:** `simulate_forward_paths`
  (`monte_carlo.cpp`) constructed a fresh `std::mt19937_64` and allocated a
  `resampled` heap buffer on *every single simulated path* inside the
  OpenMP-parallel loop — 200,000 heap allocations/frees at
  `n_simulations=200000`. Hoisted the RNG/distribution to one instance per
  OpenMP thread (reseeded per path via `gen.seed(path_seed)`, not
  reconstructed — identical reproducibility, since seeding fully
  reinitializes a Mersenne Twister's state either way and no two threads
  ever touch the same `gen`), and removed `resampled` entirely by writing
  sampled values directly into the output row as they're drawn. Did **not**
  swap the RNG family (still `mt19937_64`) — that would break bit-exact
  reproducibility for existing seeds, a separate decision out of scope
  here. Measured (min-of-7-runs, separate process invocations, honest
  about the noise): 1-thread 284.5ms→239.1ms (~1.19×), unconstrained
  117.4ms→113.7ms (~1.03×) at `n_simulations=200000` — real but modest;
  the eliminated per-path allocation was small (~480 bytes) and evidently
  wasn't the dominant cost at this problem size, unlike what the review's
  framing suggested. Kept as a correct change regardless (fewer
  allocations is never worse) with the real numbers recorded, not
  oversold.
- **Performance architecture, item 1 of an independent review of the C++/
  Python boundary:** `run_strategy()` (`backtest/engine.py`) measured
  ~1.0× end-to-end against its own pure-C++ kernel time (68ms wrapper vs.
  0.017ms native kernel) despite the kernel itself being fast — the
  wrapper computed `prices.pct_change()`/`signals.shift(1)` unconditionally
  before even checking whether the C++ path would run (never used on that
  path — the kernel recomputes both internally), and after the kernel
  returned, unconditionally rebuilt the entire Python trade log
  (`_build_trade_log`/`_compute_trade_stats`) purely to overwrite native
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` fields
  that were already correct — confirmed correct by this session's own CI
  verification work (`TestNativeTradeStatsCorrectness` passing against a
  real compiled `_sqt_core` on live CI), which is exactly the precondition
  an existing code comment had flagged as needed before removing the
  override. Both are now gone: the pandas calls are computed only where
  actually used (Python fallback path, or lazily inside the C++ path only
  when `include_trade_log=True` asks for the DataFrame), and the C++
  path's summary stats flow straight from the native result, unmodified.
  Also added an `index.equals()` fast path ahead of the existing
  `intersection()`+`.loc[]` calls for the common case where `price_data`
  and `signal_series` already share an index. Measured end-to-end
  (n=2000, `include_trade_log=False`, the common case): **26.8ms → 0.46ms,
  ~58×** — real numbers, stashed/unstashed the fix to measure the same
  benchmark before and after on the same machine, not a projection.
- **C++ hardening, Tier 4 item 13:** `stochastic_oscillator`
  (`indicators.cpp`) rewritten from an O(n·k_period) full-window rescan
  (re-scanning the entire `[i-k_period+1, i]` window on every single bar
  despite an inline comment claiming O(1)-amortized behavior a different,
  never-actually-implemented technique would have provided) to a genuine
  O(n) sliding max(high)/min(low) via two monotonic deques of indices —
  the standard sliding-window-extrema technique. Removed the stale,
  inaccurate complexity comment. Added native test coverage that didn't
  exist before at all (`tests/cpp/test_indicators.cpp`), including an
  independent brute-force O(n·k) reference oracle (deliberately
  implemented separately from the real function, not just a copy of it)
  and adversarial monotonic-rising/falling and mid-window-spike cases —
  the specific patterns that expose an off-by-one in a monotonic deque's
  front-eviction logic, as opposed to just its back-insertion logic. Added
  matching adversarial Python-level tests
  (`tests/test_cpp_new_indicators.py`) against an independent pandas
  `.rolling().min()/.max()` reference.
- `build-cpp.yml`'s ASan/UBSan job's "Verify extension loaded" step never
  actually verified anything — it imported the ASan-instrumented `_sqt_core`
  without the `LD_PRELOAD=$(gcc -print-file-name=libasan.so)` the very next
  step already correctly sets for the same import, so it always failed
  immediately with "ASan runtime does not come first in initial library
  list" regardless of whether the build itself was healthy. Confirmed via
  an actual failed CI run's logs (fetched with the repo's own stored git
  credential, since the anonymous GitHub API blocks job-log downloads even
  on public repos). Added the same `LD_PRELOAD` export this step was
  missing. This is what let item 8's `-DSQT_BUILD_TESTS=ON` + `ctest` fix
  be verified for real: the native `ctest` suite under ASan/UBSan now
  genuinely passes (confirmed on a live CI run, not just locally on
  Windows/MSVC where sanitizers aren't available at all).
- **C++ hardening, Tier 1-2 (items 1-5 of an independent code review of the
  entire `_cpp` surface at commit `d52e9f2`), each verified against the real
  compiled `_sqt_core` before and after:**
  1. `cointegration.cpp`'s `mackinnon_pvalue` used a 13-point lookup table
     with log-linear interpolation, documented as +-0.01-0.02 accurate —
     independently reproduced the exact algorithm and found errors up to
     0.08 vs. `statsmodels.tsa.stattools.mackinnonp` mid-distribution.
     Replaced with the real MacKinnon (2010) regression-surface algorithm
     (quadratic/cubic polynomial + normal CDF, coefficients extracted from
     `statsmodels`' own `tsa/adfvalues.py` for `regression="c", N=2`),
     verified to machine precision (1e-9) across a swept range of ADF
     statistics.
  2. `Array1D` (`py::array_t<double, c_style|forcecast>`) enforced dtype and
     contiguity but not `ndim` — a 2-D array silently passed through every
     binding and produced garbage (or a native crash) rather than a clear
     error. Added `require_1d()`, called at the top of all 20 `m.def(...)`
     lambdas (37 call sites) taking an `Array1D` parameter.
  3. `bollinger_bands`/`rolling_beta` used raw-moment sliding sums
     (`Sxx - Sx*Sx/W`-style formulas), which suffer catastrophic
     cancellation on a large-baseline series — e.g. a ~1e9-level price
     series previously produced a near-zero variance instead of the true
     small value, and `rolling_beta`'s denominator could collapse to
     exactly zero. Rewrote both with a shifted-window + periodic-recompute
     technique (subtract each window's own first value before accumulating,
     full recompute every `window` bars) — the same idiom already used by
     `rolling_factor_loadings` elsewhere in this codebase.
  4. `backtest.cpp`'s native trade-log cost deduction (and the identical
     logic in `_build_trade_log`, `backtest/engine.py`) was a flat
     `2*cost_per_unit`/`1*cost_per_unit` regardless of the position's actual
     size — a 5x-leveraged SCORE-type trade paid the exact same cost as a
     1x trade even though the equity curve's own `strat_ret` already scales
     cost by `abs(pos_diff)`, silently under-costing every leveraged
     (non-+/-1) position's reported `return_pct`/`avg_trade_return_pct`.
     Cost is now scaled by `abs(position_size)` per leg in both
     implementations, matching the equity curve's convention for the common
     case (full close/reopen, including leveraged round trips). A same-sign
     *resize* (e.g. 1.0 -> 2.5 in one event) remains a documented
     approximation — costed as closing the old size and opening the new one
     independently, which doesn't exactly reconcile with the equity curve's
     single smaller `abs(pos_diff)`-sized cost for that event; a fully exact
     reconciliation would require tracking continuous positions with a
     weighted-average cost basis, a bigger redesign that changes reported
     `num_trades` for resize-using strategies and was left out of scope here.
  5. `hurst.cpp`'s `hurst_exponent` accepted any `method` string, silently
     treating anything other than exactly `"dfa"` as `"rs"` — the Python
     wrapper (`analysis/hurst.py`) already validated this at its own layer,
     but `_sqt_core` is directly importable, so a caller bypassing the
     wrapper got a silently wrong estimator instead of an error. Both
     `hurst_exponent` and `rolling_hurst` now reject any method other than
     `"dfa"`/`"rs"` with `std::invalid_argument` (validated eagerly in
     `rolling_hurst`, before its sliding-window loop, so a too-short input
     that would otherwise run zero iterations still raises). Also added an
     explicit `std::isnan(h)` guard before `std::clamp`/regime
     classification — `std::clamp`'s behavior on a NaN input is unspecified
     by the standard, and relying on classify()'s threshold comparisons
     (all false for NaN) to coincidentally fall through to a safe-looking
     label was fragile.
- Root `CMakeLists.txt`'s `cmake_minimum_required` bumped from `3.15` to
  `3.19` — `3.15` was never actually sufficient: `find_package(...
  Development.Module)` requires `3.18`, and `_cpp/CMakeLists.txt`'s
  multi-value `$<CONFIG:Release,RelWithDebInfo>:...>` generator expressions
  require `3.19`. A fresh `3.15`-`3.17` CMake install would have failed at
  configure time regardless of what the stated minimum claimed.
- `.github/workflows/build-cpp.yml` never actually ran the native `tests/cpp/**`
  suite — `SQT_BUILD_TESTS=ON` wasn't passed to either `build-and-test`'s or
  `build-and-test-sanitizers`'s `cmake -B build` invocation, so the compiled
  test executables never existed, and there was no `ctest` step to run them
  even if they had. A native-only regression (like several fixed in this
  release) could land without CI ever compiling or exercising the code that
  changed. Both jobs now pass `-DSQT_BUILD_TESTS=ON` and run
  `ctest --test-dir build --output-on-failure` immediately after building,
  before the Python `pytest` step. Also added `tests/cpp/**` to the
  workflow's `paths:` triggers (previously only `_cpp/**`/`CMakeLists.txt`),
  so a native-test-only change still triggers this workflow.
- `tests/cpp/test_indicators.cpp` failed to compile on GCC/Linux —
  `std::max({...})` (the initializer-list overload) is declared in
  `<algorithm>`, which this file never included; MSVC's headers transitively
  pull it in via other standard headers, so this went undetected until the
  `build-cpp.yml` fix above actually compiled `tests/cpp/**` on Linux for the
  first time. Added the missing `#include <algorithm>`. While auditing for
  the same class of bug, also added `#include <algorithm>`/`#include
  <stdexcept>` to `bindings.cpp` (uses `std::copy` and `throw
  std::invalid_argument` ~20+ times, currently working only because
  pybind11's own headers happen to pull both in transitively) — not
  currently broken, but relying on transitive includes from a third-party
  header is fragile the same way the `test_indicators.cpp` bug was.
- `portfolio_engine.py`: `max_gross_leverage`/`max_position_pct` are now
  enforced against realized post-cost state, not just pre-trade intent;
  added insolvency checks (a rebalance that leaves the account with
  zero/negative equity now raises instead of silently continuing); financing
  (borrow fee, margin interest) now accrues on actual elapsed calendar days
  instead of a hardcoded 1-day assumption; added validation for an empty
  universe, duplicate/unsorted rebalance dates, and non-finite weights/prices.
- `sizing.py`: fixed `vol_scaled`'s rolling-window frequency mismatch,
  `equal_weight_top_bottom`'s long/short-only allocation, and
  `dollar_neutral`'s gross-leverage drift.
- `risk_metrics.py`: `var_historical`/`var_parametric`/`cvar` now validate
  `confidence` is a valid probability bound; fixed `var_parametric`'s silent
  fallback when scipy isn't available; fixed `treynor_ratio`'s misaligned
  numerator/denominator index (the excess-return numerator previously used
  the full unaligned series while beta used only the intersected dates).
- `yfinance_provider.py`: path-traversal containment on the Parquet cache
  path (symbol/date/interval), the audit trail now fires on session-cache
  hits (not just misses), cache-hit results are copied so callers can't
  mutate shared cached state, corrupt Parquet files on disk are detected and
  evicted/refetched instead of failing or serving bad data, and atomic-write
  temp filenames are now thread-unique.
- `dispatch()` sanitizes `inf`/`nan` to `None` before returning a result,
  since raw `json.dumps()` would otherwise emit non-standard tokens.
- `run_strategy` (`backtest/engine.py`) now always recomputes
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` in Python
  (`_build_trade_log`/`_compute_trade_stats`) instead of trusting the C++
  kernel's own native trade-log values, which used to record each entry one
  bar late and exclude commission/slippage. This Python-side override
  remains in place as a safety net even after the underlying native bug was
  also fixed directly (see below) — see Known Issues for the exact pending
  verification status.
- Fixed a day-0 drawdown edge case (see git history for the exact commit).
- `_cpp/src/backtest.cpp`'s `run_strategy` native trade-log construction
  rewritten to match `_build_trade_log`'s accounting exactly (entry size =
  signal magnitude not just sign, `prices[i-1]` as the reference price,
  correct commission/slippage deduction) — this is the fix for the exact bug
  the Python-side override above works around, now applied at the native
  level too, including `backtest_grid`'s batch path (`batch_run_strategy`)
  which had no equivalent Python override. **Not yet verified against a
  real compiled `_sqt_core`** (no C++ toolchain available where this was
  written) — see Known Issues.
- `stochastic_oscillator`: `k_period<=0`/`d_period<=0` now raise
  `ValidationError` in both the C++ kernel and its Python wrapper —
  `d_period<=0` previously reached the native kernel unchecked, causing an
  out-of-bounds vector read (an uncatchable segfault, not a Python
  exception), not just a wrong result.
- `hurst_exponent`/`rolling_hurst`: `method` must now be exactly `"dfa"` or
  `"rs"` (raises `ValidationError` otherwise, in both paths) —
  previously any other string was silently treated as `"rs"` while the
  result's own `"method"` field echoed back the typo, making the mistake
  invisible. `HurstInput.method`, `RegimeAdaptiveInput.hurst_method`, and
  `RegimeAdaptiveWalkForwardInput.hurst_method` are now
  `Literal["dfa", "rs"]` instead of a bare `str` so a bad value is rejected
  by Pydantic before it ever reaches the function.
- `parabolic_sar`: `af_start`/`af_step`/`af_max` are now validated (finite;
  `af_start>0`; `af_step>=0`; `af_max>0`; `af_max>=af_start`) in both the
  C++ kernel and the Python wrapper — a nonsensical combination previously
  produced a silently meaningless SAR series instead of raising.
- `run_strategy`/`backtest_grid`: `initial_capital`, `commission_pct`, and
  `slippage_pct` are now validated (finite, correct sign) before reaching
  the native kernel — a zero/negative/non-finite `initial_capital`
  previously produced silent `inf`/`nan` in `total_return`/`calmar_ratio`
  instead of raising.
- The four provider example agent loops (`Implementation/*/_agent_utils.py`)
  fixed duplicate logging handlers on repeated setup, malformed tool-call
  JSON silently becoming `{}`, missing request/tool timeouts, non-strict
  JSON allowing `NaN`/`Infinity` tokens, and narrative text being discarded
  after each tool round.
- CI: dropped the unused `pytest-freezegun` dependency (it imported
  `distutils`, which Python 3.12 removed, and nothing in the suite actually
  used it) and added `anthropic` to the `test` extras, since
  `test_multi_agent_tool_coverage.py` transitively imports it.
- `garch_volatility_forecast`: the one-step-ahead forecast seed never
  incorporated the most recent observed return (`current_var` stopped one
  recursion step short), so `forecast_annualized_vol[0]` silently diverged
  from `current_annualized_vol` and every later forecast step compounded a
  spurious extra decay. Fixed by computing the true T+1 variance explicitly
  and re-indexing the forecast horizon from `h=0`.
- `audit/paths.py`: the Windows advisory file lock (`msvcrt.locking`) raised
  `OSError` after its own ~10s internal retry and was silently swallowed by
  a blanket `except Exception`, letting `AuditWriter.write()` proceed
  completely unlocked under contention (and leaking the file handle). Now
  retries indefinitely, matching POSIX `fcntl.flock`'s existing blocking
  behavior, and closes the handle on failure.
- `PositionSizerInput`: `win_rate`/`avg_win_pct`/`avg_loss_pct` had no range
  validation (unlike the sibling `risk_per_trade_pct`), so an impossible
  input (e.g. `avg_loss_pct=0`, a Kelly-formula divisor) could reach the
  sizing math instead of being rejected up front.
- `data/_cache.py`: the shared in-process session cache (`cachetools.TTLCache`)
  had no locking despite being read/written from multiple threads via each
  provider's async path; added a module-level lock around get/set.
- Audit redaction: exception messages echoing a redacted field's raw value
  were never redacted (only `input` was), and the redaction placeholder
  itself was an unsalted 8-hex-char hash, brute-forceable offline for small
  value spaces (SSNs, PINs). Added `redact_text()` for error messages
  (sharing one `_placeholder_for()` helper with `input` redaction so both
  produce the same placeholder) and an optional `SQT_AUDIT_REDACT_SALT`
  env var, with a one-time warning when it's unset.
- `portfolio_engine.py`: `fill_price="next_open"` still looked up that
  day's own ADV/volatility for cost/impact modeling — not yet knowable at
  that bar's Open. `_valid_dollar_volume`/`_trade_cost` now index at
  `trigger_date` instead of `exec_date` (a no-op for `close`/
  `hl2_exploratory`, where the two are already equal).
- The retry decorator treated HTTP 401/403 (permanent, e.g. an invalid API
  key) identically to 429/5xx (transient), burning through a rate-limited
  API's request budget on every call until the key was fixed. Added
  `NonRetryableAPIError` (a subclass of `APIError`, so existing `except
  APIError` sites are unaffected); `PolygonProvider` now raises it for
  401/403 specifically, and the retry decorator never retries it.
- `agent/__init__.py` was missing re-exports for ~46 Pydantic models defined
  in `models.py` (e.g. `Trade`, `PortfolioOptimizationInput`,
  `OptionPricingResult`), so `from standard_quant_tools.agent import
  SomeInput` silently `ImportError`'d for those classes even though the
  models themselves worked fine. Added a regression-guard test
  (`TestAgentModelExports`) so this can't drift silently again.
- `YFinanceProvider` hard-failed with `ValidationError` on a symbol whose
  characters couldn't be safely encoded into a cache filename, where
  `PolygonProvider` already degraded gracefully by skipping the disk cache
  for that call. Both providers now use `_safe_parquet_path` consistently
  on the read *and* write side (the write-side call in `PolygonProvider`
  itself was missing the same `None` guard the read side already had).
- `CorrelationAnalysisInput.weights`/`MonteCarloSimulationInput.weights`
  (both optional — `None` means equal weighting) had no validation when
  provided, unlike the required `weights` on sibling models
  (`PortfolioInput`, `RiskAttributionInput`). Added the same length/sum-to-1
  check, guarded on `weights is not None`.
- `spread_zscore`'s rolling branch and `rolling_beta`'s pandas fallback both
  divided by a rolling std/variance with no zero-guard — a flat spread or
  constant benchmark window produced `inf`/`-inf` instead of raising or
  producing an explicit missing value. Both now NaN out that window instead
  (not a literal `0.0`, which would be indistinguishable from a legitimate
  zero mid-series).
- Test isolation: `tests/test_polygon_provider.py` and `tests/test_data.py`
  didn't redirect the real persistent Parquet disk cache to a temp
  directory (unlike `test_parquet_cache.py`/`test_audit.py`, which already
  did), so a cache entry written by an earlier test/run could leak into a
  later test in the same run — the root cause of an intermittent CI "Run
  tests" failure. Added the same `autouse=True` `redirect_cache` fixture to
  both files.
- `run_portfolio_simulation`/`run_signal_panel_backtest` fetched every
  ticker with a blocking `provider.get_ohlcv()` call inside a plain `for`
  loop — for a large universe (e.g. the full S&P 500) this meant minutes of
  pure sequential network wait before the simulation itself even started,
  unlike every other multi-ticker tool in the module, which already fetches
  concurrently. Added `fetch_ohlcv_panel_async`/`fetch_ohlcv_panel_sync`
  (same `asyncio.gather` concurrency as the existing `fetch_returns_*`
  helpers, but preserving the full OHLCV panel — Volume/High/Low, not just
  Close-derived returns — since the transaction-cost model needs it) and
  wired both tools to use it. Verified against live yfinance: 20 uncached
  tickers fetched concurrently in ~2.1s vs. ~2.4s for 10 tickers
  sequentially beforehand.
- `_sqt_core` was built and its full test suite actually run for the first
  time this session (previously blocked by a missing Windows SDK — `cl.exe`
  was present, `rc.exe`/`mt.exe` were not; see
  `Development/build_guide.md`'s troubleshooting section). This found 5
  real, previously-undetectable bugs:
  - `simulate_forward_paths`'s pybind11 binding didn't raise for
    `horizon_days<=0`/`n_simulations<=0` — the result-size validation
    degenerated to `0==0` for exactly those inputs, silently passing them
    through instead of raising `ValueError`. Fixed with an explicit upfront
    check.
  - `adf_test` (cointegration ADF/Engle-Granger) returned `NaN` for a
    degenerate, (near-)perfectly-collinear input — every regressor has zero
    variance, so the per-lag OLS solve is singular for every candidate lag —
    instead of matching statsmodels' own convention for this exact case
    (`adf_statistic=-inf, p_value≈0`, verified empirically against
    statsmodels). Fixed with an upfront degenerate-input check.
  - `ar1_halflife` returned `NaN` instead of `+inf` for a zero-variance
    lagged predictor, because `beta >= 0.0` is `false` for `NaN` under
    IEEE 754 — the same "not mean-reverting" case a non-negative beta
    already gets was falling through a different comparison path. Fixed by
    testing `!(beta < 0.0)` instead.
  - 4 of `tests/cpp/test_backtest.cpp`'s own hand-written trade-log test
    expectations were wrong — written without ever compiling or running
    them, based on a mistaken `prices[i]`-vs-`prices[i-1]` reference-price
    assumption. The actual native trade-log implementation (the
    `backtest_grid` fix from 0.1.0, described in Known Issues below) was
    already correct; only the tests needed fixing.
  - A native/Python trade-stats parity test used a tolerance tight enough to
    fail on Python's own intentional `round(..., 4)` display rounding, not a
    real discrepancy. Loosened from `abs=1e-9` to `abs=5e-5`.

### Known Issues

- **Resolved:** the native trade-stat parity gap described in earlier drafts
  of this section (`backtest_grid`'s C++ batch kernel returning uncorrected
  trade stats) is now **confirmed correct**, not just implemented. A missing
  Windows SDK component (`cl.exe` was present; `rc.exe`/`mt.exe` were not)
  was found and fixed, `_sqt_core` was built for the first time, and
  `tests/test_backtest.py::TestNativeTradeStatsCorrectness` plus the full
  native `ctest` suite (110 test cases) were actually run. The native/Python
  trade-stat accounting genuinely agrees — `backtest_grid`'s C++-path
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` (and
  anything built on top of it, e.g. `run_walk_forward_backtest`/
  `run_backtest_optimization`) can now be treated as trustworthy. See
  Fixed below for the 5 bugs this build-and-test pass actually found (none
  of them in the trade-stat fix itself).

## [0.1.0] - 2026-07-24

Initial documented release. `main` had no prior tags — this release
consolidates everything built since the first commit into one baseline.

### Added

**Data layer** (`standard_quant_tools.data`)
- `YFinanceProvider`: `get_ohlcv` / `get_ohlcv_async`, `get_ticker_info`,
  `get_financial_ratios`, `get_metadata` (dataset provenance), with retry
  with exponential backoff, an in-process TTL session cache, and a
  persistent Parquet disk cache for historical OHLCV (`SQT_CACHE_DIR`).
- `data.quality`: heuristic data-quality checks — `detect_missing_bars`,
  `detect_stale_prices`, `detect_price_jumps`.

**Indicators** (`standard_quant_tools.indicators`) — 14 functions across
trend (SMA, EMA, MACD, ADX+DI, Parabolic SAR, Williams %R), momentum (RSI,
Stochastic), volatility (Bollinger Bands, ATR, Wilder's ATR), and volume
(OBV, VWAP, MFI), each with a C++ extension → Numba JIT → pure-Python
fallback chain.

**Metrics** (`standard_quant_tools.metrics`) — 18 functions: return metrics
(cumulative return, CAGR, annualized volatility), risk/ratio metrics
(Sharpe, Sortino, Calmar, historical/parametric VaR, CVaR, Information
Ratio, Treynor, max drawdown), and backtest diagnostics (drawdown episodes,
trade expectancy, MAE/MFE excursions, exposure stats).

**Analysis** (`standard_quant_tools.analysis`) — 12 functions: OLS beta /
rolling beta, Engle-Granger cointegration + spread/half-life/z-score,
multi-factor regression + rolling factor loadings, PCA on returns, and
Hurst exponent (DFA / R-S / rolling), several with C++ fast paths.

**Backtesting** (`standard_quant_tools.backtest`)
- Vectorized single-ticker engine (`run_strategy`) with transaction costs,
  trade log, and three execution-timing modes (`close`/`next_open`/a
  same-bar approximate-fill mode, renamed `hl2_exploratory` — see Unreleased).
- Parameter grid search (`backtest_grid`) and walk-forward / regime-adaptive
  (leakage-free) backtesting.
- Multi-ticker signal-panel backtesting (`run_signal_panel_backtest`).
- A shared-cash portfolio simulation engine (`portfolio_engine.py`) with
  pluggable cost models (`costs.py`: percentage/per-share commission,
  spread, square-root market impact, short borrow, margin interest),
  liquidity/capacity constraints (`constraints.py`), and position-sizing
  helpers that turn a score panel into a target-weight panel (`sizing.py`).
- Two-leg pair-trade backtesting (`pairs.py`), reusing the portfolio engine
  so both legs share one cash account and rebalance together.
- Robustness diagnostics (`robustness.py`): block-bootstrap confidence
  intervals, parameter sensitivity, and Deflated Sharpe Ratio.
- A local Parquet artifact store (`artifacts.py`) for equity curves/trade
  logs too large to embed inline in an agent-tool response.
- 4 built-in strategies (SMA crossover, RSI mean-reversion, MACD crossover,
  Bollinger reversion), plus support for bring-your-own signal callables in
  grid search and the signal-panel backtester.

**Portfolio & Screener**
- `standard_quant_tools.portfolio`: multi-asset portfolio metrics, risk
  attribution (marginal contribution to risk, PCA-based, factor-based),
  correlation matrix.
- `standard_quant_tools.screener`: async filter-based stock screener with
  automatic `ProcessPoolExecutor` fan-out for universes over 20 tickers.

**Agent tools** (`standard_quant_tools.agent`) — 34 LLM-callable tools with
Pydantic input/output models and OpenAI/Anthropic function-calling schemas,
covering backtesting, risk/technical/portfolio analysis, screening, factor
regression, cointegration, PCA, Hurst analysis, regime-adaptive and
walk-forward backtests, pair scanning, position sizing, bring-your-own-signal
backtests, portfolio simulation, pair-trade backtests, robustness
diagnostics, capacity reports, and data-quality reports.

**Auditability** (`standard_quant_tools.audit`, `sqt` CLI)
- Every `dispatch()` call can write a tamper-evident JSONL decision record
  (inputs, market-data content hashes, execution path, output hash, latency).
- `verify_replay()` re-runs a recorded call and distinguishes stale/tampered
  cache from a genuine code change.
- The `sqt` CLI (`sqt replay` / `sqt compare` / `sqt report`) inspects and
  verifies decision records by `request_id` from the command line.

**Performance**
- Optional C++ extension (`_sqt_core`, pybind11 + CMake) accelerating Hurst,
  RSI/ADX/Parabolic SAR, Wilder's ATR, Engle-Granger cointegration, 2-variable
  OLS, the backtest kernel and grid-search batch kernel, rolling factor
  loadings, rolling beta, Bollinger Bands, and the Stochastic Oscillator.
  The API is identical with or without it; every path falls back to
  Numba/pure-Python transparently when the extension isn't built.

### Fixed

Notable correctness fixes folded into this baseline (see git history for
full detail):
- Look-ahead bias in the pairs-backtest z-score default (now a rolling
  window by default instead of a full-sample static z-score) and in the
  regime-adaptive walk-forward backtest.
- `run_portfolio_simulation` now rejects `NaN` target weights immediately
  instead of silently propagating them through the equity curve, and surfaces
  an explicit look-ahead-bias warning when using same-bar (`close`) fills.
- `sqt replay` now exits non-zero on a confirmed output mismatch instead of
  always exiting `0`.
- De-annualized the Sharpe ratio fed into the Deflated Sharpe Ratio formula
  in `get_robustness_diagnostics` (previously inflated the statistic).
- `save_artifact` now rejects a reused `(run_id, name)` unless `overwrite=True`,
  validates both against a path-traversal-safe identifier pattern, and writes
  atomically.

[Unreleased]: https://github.com/karanvora2599/Standard-Tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/karanvora2599/Standard-Tools/releases/tag/v0.1.0
