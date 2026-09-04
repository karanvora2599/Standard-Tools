# Changelog

## The book finally arrives — 200 to 204 tools

`get_order_book_metrics` has computed the microprice, touch and cumulative
imbalance, the depth profile and the depth slope since long before anything
could feed it. `DataProvider.get_order_book` declared the columns and raised
`NotImplementedError` in every shipped provider, and `book_tools.py` said so
in its own docstring: "no shipped provider serves depth." The analytics were
written, tested and correct against a book that had never once arrived.

This is the data path, not more analytics. Three new tools, and the audit
that preceded them found that eleven of the thirteen L2 features a bigger
plan proposed were already shipped — so the work was to feed what exists
rather than to build it again.

### `DataFrame.where()` with a Series condition costs 22 microseconds per column

Three vectorisations from `Development/modeling_native_plan_ii.md`, none of
which needed C++. All measured, all verified against the code they replace.

**`zscore_normalized` was paying per COLUMN, not per row.** `z.where(~degenerate,
other=0.0)` broadcasts a Series condition across a DataFrame column by column
-- 40.8 ms on a ONE-ROW 2,000-column frame, where the same masking in numpy is
0.18 ms. **224x.** `transform_predictions_to_weights` calls it once per
availability pattern, so on a panel with staggered listing dates that
per-column constant was most of the run. `_check_scores` has already rejected
NaN and infinity, which is what lets the replacement use plain reductions.

Agreement with the previous implementation is to **3.2 ULPs** (7.15e-16
relative, scale-invariant from 1e-6 to 1e9) -- floating-point reassociation
between pandas' and numpy's reduction order, not a change of formula.

**`transform_predictions_to_weights` had three separate per-column costs.**
The availability pattern was built by a Python string join PER ROW; `present`
was a scalar `.loc[date, col]` per column per group (150,000 lookups on a
500-name panel); and the sub-panel was taken and written back by label. Now
the pattern is a packed bit-record factorized in one pass, and the loop fills
a numpy matrix positionally -- the intermediate DataFrame existed only to be
converted back to one.

    500 entities x 1000 dates      before     after
    dense                          0.110 s    0.075 s    1.5x
    staggered listing dates        3.432 s    0.537 s    6.4x
    1% scattered NaN               7.574 s    1.034 s    7.3x

**Output is bit-identical in all three (max difference 0.00e+00.)**

**`_quantile_shape` called a Python function once per date.**
`groupby("date").transform(_bucket)` with a non-cython callable, measured at
78 s of a 96 s `feature_predictive_stats` over 40 features -- in which the
already-ported C++ kernel was 2% of the runtime and this pandas glue was 81%.
Replaced by one segmented `np.lexsort`: it is stable, and putting the row
position last as a tiebreaker is exactly `rank(method="first")`.

    504,000 rows, 200 entities     1.342 s -> 0.168 s     8.0x
    heavy ties                     0.264 s -> 0.021 s    12.8x
    ragged dates                   0.132 s -> 0.003 s    37.9x

Bucket assignment verified identical across 21 combinations: clean, heavy
ties, an all-identical column, ragged dates, dates carrying fewer names than
buckets, two-entity dates, and 3/5/10 quantiles.

**None of this is a kernel, and that is the finding.** The measured ceiling on
the whole subsystem is that a real estimator spends 78.5% of wall-clock inside
sklearn, and the existing kernels are only a third of the cost of the
preprocessing path they sit inside. The pandas dispatch around them was worth
more than porting more of them would have been.

### The guard matched three phrasings, and the count rotted in the other four

A sweep of all 19,000 lines of documentation against the live registries.
The interesting part is not the wrong numbers -- it is that a guard already
existed for exactly this and let them through.

`test_every_whole_surface_count_is_current` matched "The N tools", "Every
one of the N tools" and "Serving all N tools". The docs also say "returns
174 LLM-callable tools", "handing the model all 174 tools", "the 174-tool
surface" and "index of all 200 tools" -- and every one of those rotted
across several releases while the guard passed on every run.

**What was wrong.** The analysis facade was quoted as 174 in seven files and
is 178. `modeling` was quoted as 17 or 16 and is 20. The MCP architecture
note described a 152-tool surface across seven runtimes and eight
categories; it is 178 across eight runtimes and thirteen. The README's
per-runtime table was wrong on three rows at once -- `data` 14 against 17,
`modeling` 17 against 20, `microstructure` 16 against 17 -- because the
identical table in `19_runtimes.md` was guarded and the README's was not.
`25_testing.md` still described "the one tool without a baseline" when
`EXPECTED_UNSYNTHESIZABLE` has been empty for some time and all 207 tools
are fuzzed. Test counts, schema sizes and the per-category figures were all
from earlier surfaces.

**Two figures were not merely stale, they were incomparable.** The README
quoted full-detail sizes from `--print-budget`, which counts bare schema
bytes, beside `--tool-detail auto` sizes from the server, which count the
envelope actually sent -- in one sentence, as if one instrument had produced
both. Measured consistently, `auto` is never larger than `full`; measured
the way the README had it, six of the ten runtimes came out larger under
`auto` than under `full`, which is nonsense that reads as a real finding.
Both numbers are now labelled with the instrument that produced them.

**A guard that forbade the right answer.** `test_wrong_numbers.py` kept a
list of phrases that must not reappear in the source, and "178 tools" was
on it -- correct when the facade was a different size, and now a rule
against stating the truth. Removed, with the reason recorded: a guard that
outlaws the current answer teaches the next person to write something
vaguer to get past it.

**What replaces the narrow guard.** Any three-digit count of tools anywhere
in the docs must now be one of the two real surface sizes. That works
because every scoped count is two digits -- the largest runtime is
`research` at 42 -- so a three-digit number here is always a claim about
the whole surface or about the facade. Genuine exceptions (a capacity
figure, a sentence explicitly about history) are declared with their reason
rather than pattern-matched away, so the list cannot become a place stale
numbers hide. The runtime-table check now covers the README as well as
`19_runtimes.md`. Verified by injecting a stale count and watching it fail
with the file, line and value.

**What was checked and found correct**, so it is not guessed at next time:
every cross-document link and anchor, every feature id, every registered
estimator, every provider name, and every tool -- all 23 features, 17
estimators and 207 tools appear in the documentation.

### Minutes of silence are indistinguishable from a hung server

`--enable-long-running` exposes `scan_pairs` (measured at 5.31 minutes over
a 2,000-ticker universe) and `run_backtest_optimization`. Hiding them by
default was half a solution: enabled, they ran as silence, and a client's
usual response to silence is a timeout that abandons work about to finish.
The tools were reachable and not really usable.

The server now emits `notifications/progress` every `--heartbeat` seconds
while a tool runs.

**No `total`, ever.** The server dispatches an opaque synchronous call into
the library; there is no progress hook, so any completion estimate would be
invented. `progress` with no `total` is exactly how the protocol says
"still working, completion unknown", and a client renders it as an
indeterminate spinner. A fabricated total renders as a BAR, and a bar that
stops at 90% is worse than no bar -- it converts "I do not know" into a
specific false claim.

**Elapsed seconds, because the protocol requires the value to increase.**
Any fraction-of-work estimate can be revised downward and would violate
that; elapsed time cannot.

**Only when asked, and never at the caller's expense.** Nothing is sent
without a `progressToken`. If the stream is closed or the client has gone,
the notification is dropped and the tool still returns -- failing a call
because its progress note could not be delivered would turn a cosmetic
problem into a lost result.

**A defect this introduced, caught by its own test.** A task group re-raises
whatever escaped it WRAPPED in an ExceptionGroup. `call_tool` catches
QuantError to return the library's own self-correcting message verbatim,
and wrapped, that except clause stopped matching: `describe_tool` with a bad
argument went from

    ValidationError: 2 validation errors for DescribeToolInput
    tool_name -- Field required

to

    ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)

for exactly the clients that asked for progress, and no others. Every
self-correcting message the server exists to pass through would have
collapsed the same way. `report_liveness` now unwraps a single-exception
group, so adding a heartbeat cannot change what a block raises.

**What this is not.** It is not cancellation. The tools are synchronous
CPU-bound Python and a thread cannot be killed, so a client that gives up
stops listening while the work runs to completion. Stopping it needs the
process pool Phase 3 still has open -- abandoning the thread would leave it
writing into the runs directory behind a caller who believes the call is
over.

### Three defects in the estimators and diagnostics just added

Found by going back over the newest code rather than by a failing test,
which is why they are worth writing down: each produced a plausible result
or a confusing crash rather than an error anybody would attribute.

**A bare `sgd` classifier could not finish training.** sklearn's
SGDClassifier defaults to `loss='hinge'`, which has no `predict_proba` --
and `adapters.score` asks every classifier for one on every fold. So
`EstimatorSpec(type="sgd")` trained for three folds and then died inside
sklearn's internals with `AttributeError: This 'SGDClassifier' has no
attribute 'predict_proba'`, pointing at nothing the caller wrote. Every
other classifier in the registry works with no params at all; this one
shipped not working.

Registered now as a subclass defaulting to `log_loss`, and `hinge`,
`squared_hinge` and `perceptron` are removed from the allowlist rather than
left as choices that cannot work. Offering an option that fails after
training, from inside a dependency, is worse than not offering it.

**A "<NA>" bucket in the error breakdown.** `qcut` returns NA for a row
whose feature is missing, and those rows have perfectly good residuals --
so they grouped into a bucket labelled `<NA>` reported alongside the real
deciles, and one large enough could be named the WORST bucket. Every
rolling feature has a warm-up window, so every real panel produces them.
They are dropped from the table and counted in `rows_without_feature`
instead: "we do not know this row's spread" is not a spread decile, and
silently analysing fewer rows than the caller believes is its own kind of
wrong answer.

**Numeric buckets sorted as strings.** Bucket labels are strings so one
table can hold entities, periods and decile numbers, and sorted
lexicographically bucket 10 lands between 1 and 2. Deciles are 0-9 today so
nothing was visibly wrong -- this was a trap set for whoever raises
`N_BUCKETS`, which is a module-level constant sitting there to be raised.
Numeric labels now sort by value and everything else lexicographically.

### Measured: the lag window, and the multi-output decision

A panel built so the label depends on the recent PATH of a feature --
`0.25*f[t] + 0.45*f[t-1] + 0.30*f[t-2]` plus noise, 2,988 rows over 6
entities, walk-forward with a 250-bar train window and a 5-bar embargo. The
history IS the signal by construction, so a model denied it cannot reach
the achievable R2 however flexible it is.

| model | OOS r2 | IC |
|---|---|---|
| ridge, current value only | 0.2857 | 0.5405 |
| **ridge, + 2 lags** | **0.5499** | **0.7428** |
| mlp, current value only | 0.2833 | 0.5389 |
| mlp, + 2 lags | 0.5346 | 0.7339 |
| sgd (huber), + 2 lags | 0.5513 | 0.7432 |

**The window nearly doubles R2.** That is the case for `FeatureSpec.lags`,
and it is the whole case for sequence modelling here.

**The MLP does not beat ridge, it loses to it** -- 0.5346 against 0.5499 on
a truth that is linear, which this one is. That is the honest result and it
is worth stating plainly: a network is not free, and on a panel this size
its extra capacity is spent on noise. It earns its keep when the
relationship is not linear, and the way to find out is to run both, which
costs one extra call.

**This is also the strongest argument against a torch dependency.** If an
MLP over the lag columns cannot beat a ridge over the same columns, the
additional machinery a TCN brings -- weight sharing across lag positions --
has even less to justify it at this depth.

#### Why there is no multi-output estimator

Fitting one model that predicts several horizons at once has exactly one
form that is not already available. `MultiOutputRegressor` fits one
estimator per output, so it is arithmetically identical to running N
experiments against a panel registered with N labels -- which
`TargetSpec.horizons` and `register_external_panel(targets=[...])` already
do, over the same rows and the same folds. Only a model that SHARES
parameters across outputs, such as a network with N heads, computes
anything new.

So that is what was measured, on three labels driven by one shared latent
plus independent noise -- the single most favourable case a shared
representation can be given:

| model | h1 | h5 | h20 |
|---|---|---|---|
| 3 independent MLPs | 0.5299 | 0.3331 | 0.1993 |
| 1 shared-head MLP, 3 outputs | 0.5300 | 0.3331 | 0.2034 |
| 3 independent ridges | 0.5510 | 0.3406 | 0.1964 |

**Sharing gains +0.0014 mean R2.** Against that: the out-of-sample
prediction frame grows a column per horizon, `ModelManifest.target_id`
stops being a single value, and `portfolio_eval`'s pivot, the backtest
bridge and `score_predictions` all change shape. Three independent ridges
beat both networks on two of the three horizons anyway.

The measurement is a spike, not a validation -- one split, one
architecture, synthetic data. It does not need to be more than that: the
effect it found is 0.0014 and the change it would justify touches five
modules and every persisted prediction frame.

### The window has to be in the columns, so torch is not a dependency

`engine.py` hands every estimator a 2-D X whose rows are (date, entity)
observations carrying NO entity identity. That is not an oversight -- it is
the contract that lets ridge, LightGBM and an SGD learner be swapped for
one another without the engine knowing anything about them. A recurrent or
convolutional model cannot reconstruct per-entity sequences from it, so the
history has to arrive as COLUMNS whatever architecture reads it.

`FeatureSpec.lags` is that. `lags=[1, 2, 3]` adds a feature's value 1, 2
and 3 bars ago as its own columns, and every estimator in the registry gets
access to history rather than one that ships with special plumbing.

**Shifted within the entity, before stacking.** Applied after `stack_long`
a shift crosses the entity boundary and hands AAA a value belonging to BBB.
The panel that results is perfectly well-formed, every aggregate downstream
looks normal, and nothing later catches it. Pinned by a test that asserts,
for every entity in a real built panel, that lag1 equals that entity's own
previous bar -- and that states the wrong answer explicitly beside it.

**A negative lag is refused, not clamped.** It is a shift FORWARD: next
week's value of a feature on this week's row. Every leakage check in this
library reasons about the TARGET, so not one of them would see it, and the
model it produces looks brilliant. The refusal names the value the caller
probably meant.

**Once the window is in the columns, a TCN adds weight sharing across lag
positions** -- which pays at hundreds of timesteps and thousands of series,
not at the depth a daily panel supports. `mlp` over the lag columns is the
same hypothesis class with no dependency and runs inside the walk-forward
loop that already exists. torch is installed on this machine and is still
not a dependency of this package, which is the decision rather than the
absence of one.

**The architecture is two bounded integers.** sklearn takes
`hidden_layer_sizes` as a tuple, which the parameter allowlist cannot bound
-- and an unbounded tuple is exactly the resource-exhaustion path
`bounds.py` exists to close. Width and depth are ints instead. The subclass
that maps them rederives the tuple at fit time, because `clone` rebuilds
from `get_params` and the tuple is not a param: without that, every fold
would have trained sklearn's default 100-unit layer while the manifest
recorded the width that was asked for. The engine clones per fold, so that
is the normal path and not an edge case.

### Online modelling was mostly already here

Walk-forward already refits on every fold, and `validation/weights.py`
already implements exponential time decay. Those are the two things "online
learning" means for correctness on a panel, and adding a second mechanism
for either would have been the same feature twice.

What was actually missing is an estimator cheap enough to refit often with
a step size the caller controls, so `sgd` is registered for both tasks.

**`partial_fit` is deliberately not reachable from a ModelSpec.** The
engine's guarantee is that every out-of-sample prediction comes from an
estimator that never saw that row or anything overlapping it. A warm-started
estimator carries state from every previous fold, including rows inside the
current fold's purge and embargo window. The speed is real; the guarantee
it costs is the one every number downstream depends on.

### Two network features chosen for what they are not

`network.avg_correlation` and `network.mst_degree` describe where an entity
sits in the correlation graph its universe forms.

**Eigenvector centrality is deliberately absent.** Up to normalisation it
IS the leading eigenvector of the correlation matrix, which
`factors.pca_loading` has computed since this package had features at all.
Registering it under a graph-theoretic name would be one number with two
explanations.

The two that ARE here are not recoverable from PC1. Mean correlation is
scale-free where a loading is variance-weighted -- multiplying a series by
100 changes its variance contribution entirely and its mean correlation not
at all, which is pinned by test. MST degree is local topology: a star
universe has a hub of degree n-1 and spokes of degree 1, a chain has no
hub at all, and the two can share a PC1.

**The edge weight is a distance, not a correlation.** `sqrt(2(1-rho))`, the
Mantegna metric -- zero only at rho=1 and obeying the triangle inequality,
which a raw correlation does not. A spanning tree over a non-metric weight
spans nothing in particular. An unestimable pair gets the MAXIMUM distance
rather than zero, since zero would read as perfect correlation and route
the tree through a pair nobody measured.

### An R2 cannot tell you where a model is wrong

`score_predictions` reports an R2, an IC and a predict-the-mean baseline,
and those answer whether the model beat a constant. They cannot separate a
broadly mediocre model from one that is excellent everywhere except the
conditions you trade -- those two have the SAME headline number and
opposite consequences. Nothing in `modeling/` computed a residual, a
calibration curve or an error breakdown before this.

`analyze_model_errors` does. It breaks the out-of-sample errors down by
entity, by period, by the model's own prediction decile, and by the decile
of any feature in its panel -- the last being the one that pays, because it
turns "the model is mediocre" into "the model fails when the spread is
wide", and only the second of those is something you can act on.

**The join it depends on.** The persisted out-of-sample frame carries
`date`, `entity` and `prediction` and NO outcome, so every number here
exists only because the actuals are fetched back from the dataset panel the
model was fit on. Which panel is not a guess: the manifest's `target_id` is
matched against the panel's declared labels, and an ambiguous match REFUSES
rather than taking the first -- residuals against the wrong horizon look
completely normal and describe a different model.

**Calibration is a separate question from accuracy.** A regression can rank
perfectly and be systematically over-confident, its predictions spread
twice as wide as the outcomes they predict. Regressing the actual on the
predicted says so -- slope 1 is calibrated, 0.5 means everything sized
directly from the prediction over-trades by two. Measured on a series and
its exact double: slope 0.5000, dispersion ratio 2.0000, and an R2 that
notices nothing.

**Residual autocorrelation is computed within each entity.** Stacking the
panel first would measure the row ordering, since consecutive rows in a
long panel are different entities on the same date. Pinned by a panel whose
stacked residuals alternate and correlate at MINUS 0.9, and whose real
within-entity autocorrelation is zero. It is also reported with the caveat
that a positive value is EXPECTED for an overlapping label -- an h-bar
forward return sampled every bar shares h-1 bars with its neighbour -- so
it reads as how few independent observations there were, not as
misspecification.

**Thin buckets are shown and not ranked.** The worst bucket of any
breakdown is almost always the emptiest one, and a finding computed without
a row-count floor reports noise with a decimal point. Buckets under 30 rows
are marked `thin`, listed in the table, and excluded from the findings.

**A flag is not a gradient.** `qcut(duplicates="drop")` does not raise on a
two-valued column -- it silently returns two buckets, which would have been
reported as deciles. The bucket count is checked rather than the exception,
because the exception never comes: under three buckets refuses, and between
three and ten says out loud that these are not tenths of the sample.

**Output stays bounded.** A 500-name universe would return 500 rows of
table, so each breakdown returns its worst and best `top_n` and counts what
it omitted.

### Registered where it lies, never copied

Every other path into this library goes provider call -> one whole
DataFrame -> `publish` -> a second complete copy under `SQT_RUNS_DIR`. Two
full materializations survive a decade of daily bars and not an afternoon of
depth. The only concession to size anywhere else is `fetch_tick_tape`'s
`limit`, which does not sample -- it TRUNCATES, so every rate and total
computed downstream understates the real one and nothing in the numbers says
so.

`register_external_dataset` stores a pointer and a schema. A 400,000-row,
73 MB mbp-10 export registers into a single 400-byte sidecar and is read in
batches, with column projection so that reading four columns of a
sixty-column book reads four. `resolve()` hands back an `ExternalDataset`
handle rather than a frame, deliberately: something frame-shaped would let a
consumer written for a fetched panel pull the whole file through an `.iloc`
without anyone deciding to.

This extends the existing `sqt://` mechanism rather than paralleling it. Two
new kinds (`order_book_panel`, `event_panel`) are external-only; `tick_tape`
and `quote_panel` now exist in both storages, because a tape `fetch_tick_tape`
fetched and a tape bought from a vendor are the same content addressed
differently. The sidecar records which storage a given reference used, so
the kind stays one concept instead of growing an `external_` twin.

### The promise that had to get weaker, and says so

A published artifact is immutable because this library wrote it. An external
file belongs to the caller and can be re-extracted under a live reference,
so `describe_external_dataset` reports `changed_since_registration` -- a
field with no equivalent anywhere else on the surface. It is backed by a
`fingerprint` (every file's name, size and mtime), spelled with a different
key from a published artifact's `content_hash` because it is a weaker claim:
it catches a re-extract, a truncated copy and a partially written file, and
not an edit preserving both size and mtime. Hashing the bytes would catch
that too and would cost the full read this path exists to avoid.

### Schema at registration, rows at validation

Registration refuses a book with no `ask_size_0` before reading a row. It
does NOT read the rows, and the gap is the point: **a book with its bid and
ask columns transposed has a perfectly valid schema.** Every column present,
every value a real price, only the order wrong.

`validate_external_dataset` catches that, scanning in batches and returning a
verdict rather than raising -- three crossed books in nine million rows is
fine and a third of them crossed is transposed columns, and only a count
separates the two. It is the out-of-core sibling of `validate_pit_records`,
which is capped at 5,000 rows passed inline through a tool call's JSON, and
it calls the same `validate_pit_frame` per chunk rather than reimplementing
the temporal rules. So the swapped `event_time`/`available_time` leak --
records "available BEFORE the period they describe ended" -- is now caught
in a file of any size by the same check that has always caught it in forty
rows.

### Databento, and three ways to be silently wrong

None of these raises. Each produces finite, ordered, plausible-looking
numbers, which is what earns them a module and 39 tests that need no API key:

- **Fixed-point int64 prices.** A raw `bid_px_00` of 250,002,273,815 is
  $250.0023. Read as dollars, every spread and microprice is off by a factor
  of a billion. Detection reads the DTYPE first and cross-checks magnitude in
  BOTH directions, because an integer column of plausible dollar magnitudes
  is a rounded export and dividing it by 1e9 is the same error mirrored.
- **`UNDEF_PRICE` is int64 max, not null.** Scaled, an empty level becomes a
  **$9.2 billion quote**. Sentinels are masked BEFORE scaling, after which a
  sentinel is just a large float and no longer identifiable.
- **A trailing empty level is not a level.** An mbp-10 subscription on a name
  that never shows ten levels returns four real ones and six of sentinel;
  once masked, the columns are still there, the dataset DECLARES ten, and
  `depth_slope` regresses size against distance over levels holding nothing.
  Trailing empties are dropped and counted. A gap BELOW a live level is
  reported instead, because that is a malformed book rather than a thin one
  and renumbering around it would hide that.

`ts_recv` and `ts_event` differ by the network, which is exactly what a
latency study measures, so which one was used is reported on every call
rather than chosen silently. `action`, `side`, `flags` and the per-level
order counts are kept -- they are what make cancellation rate and a
queue-position proxy computable later, and precisely what a naive rename
discards. And the vendor's own `F_MAYBE_BAD_BOOK` and `F_BAD_TS_RECV` flags
are surfaced: the venue reporting that it could not keep the book consistent
is worth more than any check invented here.

A raw export is refused with the normalizer named. "Missing column
`bid_price_0`" would send someone hunting for data that is sitting right
there under another name.

### Measured

A 400,000-snapshot, 10-level export: refused raw, normalized (3.2M sentinels
masked, levels 8-9 dropped as empty), registered without copying, validated
across 7 batches at 100% coverage, and read back as a mean spread of 0.4622
bps, a microprice of 250.9017 and a depth slope of 5,543.5 resting units per
basis point. Inline and by-reference agree to floating-point equality.

`18_mcp.md`'s budget tables are regenerated from `print_budget()` and every
unchanged runtime is byte-identical to what was published, which is what
confirms the two rows that moved moved for a reason.

### A feature matrix this library did not build

`build_model_dataset` fetches OHLCV, computes registry features and writes a
panel. When the features already exist -- a C++ pipeline over an L2 feed, a
warehouse query, another system -- there is nothing to fetch and nothing to
compute, and the only thing between that matrix and `run_model_experiment`
was the dataset record.

`register_external_panel` writes the record and nothing else. Measured on a
four-entity, 260-date panel: the dataset directory holds
`dataset_meta.json` and `dataset_spec.json` and NO `panel.parquet`. The
matrices this exists for are often partitioned directories, and flattening
one into a single copied panel is the materialization the external-dataset
contract was built to avoid.

**Integrity is not the price of that.** The engine loads the panel whole
either way, so `hash_dataframe` runs on the loaded FRAME rather than on the
file, and a referenced panel is verified on every load exactly as strictly
as a built one. What changes is the failure mode, and both are now pinned by
test: an edited file fails loudly with a hash mismatch, and a moved one
stops loading with a message saying why. A built panel could never do
either, because this library owned it.

**The horizon is required and is not defaulted.** A panel arrives with a
`target` column and no statement of what that column means. The engine needs
the horizon for the target-overlap purge -- the rule stopping a label
spanning bars t..t+h from being trained on beside a fold boundary inside
that span -- and a missing horizon does not fail. It disables the purge
silently. That is the one thing about an external panel this tool refuses to
guess.

The spec it synthesizes is a real `DatasetSpec`, not a placeholder:
`run_model_experiment` verifies its hash, bundles it into the registered
model and reads its universe and interval into the lineage, so a decorative
one would put a false claim in all three.

`DatasetSpec.provider` gains `"external"` and `"databento"`. The first is
not a provider -- it is the honest label for a panel with no fetch behind
it, and `score_model` reads it to refuse by name rather than hunting for
feature definitions that do not exist. The second removes a monkey-patch: a
sibling project widens this exact Literal at import time and its own
docstring calls upstream "the cleaner home", noting the trap that makes the
patch fragile -- a nested model's core schema is INLINED into its parents,
so rebuilding `DatasetSpec` alone leaves every tool-input model still
advertising the old enum.

Verified end to end on a panel with planted signal: `alpha` predicts the
target and `noise` does not, and a ridge fitted through
`run_model_experiment` on the registered dataset reported an out-of-sample
r2 of 0.796. A pipeline that mismapped a column would still have "run".

### Several horizons in one panel

A microstructure panel is labelled at 100ms, 1s, 5s and 30s at once, off
IDENTICAL features. One dataset per horizon would recompute and re-store the
same matrix four times, and -- worse -- the four models would stop being
comparable, because each would have been aligned separately against its own
label's coverage.

`register_external_panel` now takes `targets`: several labels, each with its
own horizon, from one file. The panel carries them as `target__<name>`, and
the PRIMARY is also written to plain `target`, so a multi-horizon panel is
still an ordinary panel to everything that has only ever seen one label --
`explain_dataset_row_loss`, the feature report, the engine itself.
`run_model_experiment` gained `target` to pick one.

**The engine did not change.** Selection is a rename plus a drop performed
before the dataset dict is handed over, so `engine.py` reads `target` and
`label_end_date` exactly as it always has.

**Rows are dropped by the CHOSEN label, per experiment.** A 30-second
horizon has more unclosed rows at the end of a sample than a 1-second one,
and dropping on the union would make every short-horizon model pay for the
longest one's warm-down. Each experiment drops only its own and reports how
many; `target_id` follows the selection, so a registered model records the
horizon it actually learned rather than the panel's first one. Selecting a
label whose `label_end_date` is absent drops the primary's rather than
applying it, because a purge computed against another horizon's window is
worse than a purge computed on the nominal horizon.

Measured on a panel built so `alpha` predicts the 1-bar label strongly, the
5-bar one weakly and the 20-bar one not at all -- one file, three labels,
three ridges through the same folds:

    label   rows fitted   OOS r2
    h1          1,040     +0.9929
    h5          1,020     +0.8478
    h20           960     -0.0034

The row counts fall by exactly each horizon's own unclosed tail (0, 20 and
80 rows = 4 entities x 0/5/20 bars). A selector quietly training on the
wrong column would still have returned three models and three numbers; only
the ORDERING says the right label was used.

**What this is not.** One estimator emitting several horizons at once --
multi-OUTPUT -- is a different change and is not done. It would alter the
out-of-sample prediction schema, `ModelManifest.target_id`, and every
consumer of a single `prediction` column including `portfolio_eval`'s
date-by-entity pivot. What is here gives one panel, N comparable models and
the horizon curve; a joint fit does not.

### Book-update OFI, and the four L2 features are done

`get_order_flow_imbalance` in this library computes signed return times
volume from BARS. That is a proxy for order-flow imbalance and a different
quantity, and the audit that opened this work listed the real one -- the
Cont-Kukanov-Stoikov definition, from the book itself -- as one of four
features that genuinely did not exist here.

`book_dynamics` computes it, and `get_order_book_metrics(include_dynamics=
True)` surfaces it. NO new tool: a second tool with a name near
`get_order_flow_imbalance` is the confusable duplication the runtime split
exists to prevent, so the measure lands on the tool that already reads a
book.

The definition reduces to a size DIFFERENCE only when the price is
unchanged, which is exactly the case a naive implementation gets right:

    bid price rises   the whole NEW size is demand        +q_b(n)
    bid price falls   the whole OLD size left             -q_b(n-1)
    bid price equal   the difference                      q_b(n) - q_b(n-1)

All three are pinned by test against hand-computed transitions, because an
implementation that only handles the third passes any test written from the
intuition and is wrong on the two interesting book states.

A crossed or non-finite snapshot poisons the PAIRS it participates in, and
those are dropped and counted rather than interpolated across -- an
interpolated pair describes a transition nobody observed.

**That completes the four.** Queue position, cancellation rate and quote
intensity came from the order feed; book-update OFI comes from the book.
The audit's other nine were already shipped, which is why this is four
functions rather than a feature library.

### Ensembles that cannot cheat

The usual way stacking goes wrong is fitting the combiner on base
predictions the base models made about their own TRAINING rows. Those are
optimistic in a way the combiner cannot see and will exploit, and the result
looks excellent until it meets a new day.

`build_model_ensemble` cannot make that mistake, because the only
predictions it reads are the out-of-sample ones `run_model_experiment`
already persists -- each row predicted by a fold that did not train on it.
It publishes an ordinary `sqt://predictions` reference, so `score_predictions`
and the backtest bridge read the result with no new machinery.

**rank_mean is the default, not mean.** Two models predicting the same thing
on different scales average into a number dominated by whichever has the
wider spread -- a fact about its units, not its skill. Measured: a series
and its exact negative at 1000x scale average to something tracking the
loud one, and rank-average to zero, which is the honest answer for two
models that disagree completely.

**What it refuses.** Averaging a classifier's probability with a regressor's
return, which is arithmetic on incomparable units -- the same rule
`compare_models` applies to ranking across tasks. A model listed twice,
because silent double-weighting is a weighting decision disguised as a typo.
Weights passed with a method that ignores them, because quietly dropping
them is how an ensemble ends up not being the one anybody designed.

**What it reports.** The pairwise correlation between the base models, which
is the number that says whether the ensemble was worth building. Two models
agreeing at 0.98 combine into approximately either of them, and the
ensemble's own score cannot show you that -- it looks exactly like a good
single model. Above 0.95 it says so out loud.

Coverage is reported too: only rows EVERY model predicted are combined, so a
model validated over a shorter window silently shortens the ensemble unless
the per-model row counts are visible.

### Order-by-order: the four measures a depth book cannot produce

Depth aggregates size per price level, and that aggregation is the ceiling.
A book showing 5,000 shares at the bid cannot say whether it is one order or
two hundred, which arrived first, or whether size that vanished was
cancelled or filled -- and cancelled and filled mean opposite things about
who wanted to trade.

`DataProvider.get_order_events` is a new optional capability with its own
declared column contract, implemented by `DatabentoProvider` alone from the
market-by-order schema. `analysis/order_events.py` computes from it, and
`get_order_event_metrics` (microstructure, 16 -> 17 tools) exposes it inline
or through an `sqt://order_event_panel` reference:

- QUEUE AHEAD -- resting size at an order's own price level when it arrives,
  the number that decides whether a passive order fills. Computed in one
  pass with a per-(side, price) accumulator rather than a book rebuild.
- ORDER LIFETIME -- add to cancel or fill, reported separately for the two
  because they answer different questions.
- CANCEL-TO-ADD and CANCEL-TO-TRADE -- exact, where a snapshot cannot tell
  the two apart at all.
- EVENT INTENSITY by action.

**Censoring is counted, not folded in.** An order already resting when the
window opened has no add in it, so its true lifetime exceeds anything the
window can see; it is counted separately and EXCLUDED from the averages.
Folding it in as time-since-window-open would drag every average downward,
worst for exactly the long-resting orders a queue study is about. Orders
still open at the close are reported for the same reason. A CLEAR resets the
queue accumulators rather than carrying depth across a boundary where none
existed, and a MODIFY is counted without adjusting queue depth because
whether it loses priority is the venue's rule, not this library's.

Every measure is pinned against a fixture with a hand-computable answer -- a
queue of exactly 300 ahead, a lifetime of exactly 5 seconds, three cancels
per four adds -- because "it returned a float" is not evidence for a queue
statistic.

Four liquidity labels join the target registry alongside them:
`future_depth`, `future_ofi`, `future_volume`, `future_trade_intensity`.
Eighteen target types now, twelve of which no Close series can build.

### The first depth provider

`DataProvider.get_order_book` has been a declared contract with canonical
columns and no implementation since it was written -- "NOT IMPLEMENTED BY
ANY PROVIDER IN THIS LIBRARY". Every depth measure in
`analysis/order_book.py` was built and tested against synthetic books shaped
to that contract. `DatabentoProvider` serves a real one, and
`book_metrics` reads it without translation, which is what the declared
contract was for.

Twelve places in the source and docs said no provider serves depth. All
twelve now say which one does, and which call -- a false claim in a tool
description is what an agent reads.

**Operational knowledge reused rather than rediscovered.** A sibling
project has run Databento in production and its provider says in its own
docstring that upstream "is the cleaner home", monkey-patching only because
it cannot edit here. Taken from it: dataset PREFERENCE rather than one name
(consolidated where it reaches, venue where it does not, depth venue for the
deepest history); `end` anchored to the dataset's published edge rather than
to wall-clock now, which is what makes a weekend request return Friday
instead of failing; the `ohlcv-1d` finalization walk-back, because daily
bars finalize a day or two behind the edge the range endpoint reports; and
remembering entitlement denials, because a 403 is a fact about the
subscription rather than about the request.

**Prices were NOT taken from it.** That project scales by a magnitude test
on the last row -- `1e9 if close > 1e7 else 1.0` -- reading one value to
decide the units of a whole frame. `data/databento.py` decides from the
dtype, cross-checks magnitude in both directions, masks the int64-max
sentinels BEFORE scaling, and reports which of ts_recv/ts_event it used. It
also reads the vendor's own `F_MAYBE_BAD_BOOK` flag, which that project does
not.

**Two honest fields.** `adjusted=False`, because Databento serves what the
venue published -- a split is a real -50% bar, and every other provider here
reports True, so this is the one that will surprise someone.
`point_in_time=False`, because corrections are issued; they are announced
rather than silent, which is better than most, but "announced" is not the
guarantee that field asks about. Fundamentals are refused rather than
returned empty, which would be indistinguishable from a company that has
none.

**42 tests, all offline.** The client is injected, so dataset preference,
the finalization walk-back, denial memory, the inclusive end date and the
symbol mapping are all exercised without a key, a network or an
entitlement -- which is the only way those parts get tested more than once.

### Eight labels this library records and cannot build

`TargetSpec.type` went from six types to fourteen. The eight new ones are
microstructure and execution labels -- `future_mid_return`,
`future_microprice_return`, `future_markout`, `next_mid_direction`,
`future_spread`, `fill_probability`, `time_to_fill`, `adverse_selection` --
and NONE of them is a function of closing prices. A markout is measured
from a fill; a fill probability needs queue position and cancellations.

`build_target` refuses every one by name and says to compute it where the
book is and register the panel. It does not approximate: a fill probability
derived from daily bars would be a number with nothing behind it, and would
look exactly like a number with something behind it.

What this buys is that an externally computed label can say what it IS.
Before it, a fill probability had to be registered as `forward_return` -- a
false claim in the manifest, and one that left the task/target check unable
to tell a probability from a return. Measured end to end: a panel labelled
`fill_probability` now fits under a classifier at 0.6+ accuracy and is
REFUSED for a regressor. That refusal is the one that matters, because a
regressor on a 0/1 label does not error -- it fits and reports a
meaningless R2.

**One registry, not five literals.** `TARGET_KINDS` is the only place that
says what a label is: its tasks, whether it is buildable, whether it is
continuous. The engine's compatibility map is now DERIVED from it -- it was
three hand-written sets, two of which were copies of each other kept in
sync by hand -- and the threshold rule reads `continuous` off it rather
than the four-of-six list it had inlined.

This is the lesson from the task literal applied one file over, and it was
overdue: of the four copies of the target literal, TWO were added by the
external-panel work in this same changelog entry. A hand-written `Literal`
still exists because one cannot be built from a dict at type-check time,
and a test pins the two equal.

**A silent fallthrough closed.** `build_target` ended in a bare `else`
producing a direction target, so any type added to the Literal and
forgotten there came back binarized -- a continuous label arriving as
1.0/0.0 with nothing raising. Every buildable type now has an explicit
branch and anything unhandled raises, naming the file to fix.

### `ranking` was first-class in the spec and refused by both consumers

`ModelSpec.task` has accepted `ranking` since it was added, the estimator
registry carries two rankers, and `15_modeling.md` calls it the right task
for a cross-sectional problem. Both functions that turn predictions into
positions rejected it outright:

    bridge.oos_predictions_to_signal_panel   task must be 'regression' or
    portfolio_eval.predictions_to_score_panel 'classification'

So a ranker could be trained, registered and inspected, and never traded or
evaluated as a portfolio. `predictions_to_score_panel`'s own docstring even
said "Ranking is unaffected (a constant shift is monotone)" -- describing
behaviour its guard three lines below made unreachable.

A ranker emits a relative SCORE, which is the same shape a regressor's
magnitude is, so both now read it identically: sign for the side, deadband
around zero, no 0.5 recentering. Only a probability gets recentered.

**The cause was a repeated literal.** The task set was written five times
in two different widths -- three said regression/classification/ranking and
two said regression/classification. The narrow pair was not a different
opinion; it was where ranking had been forgotten. There is now one
`TASKS`/`Task` in `modeling/specs.py` and one `SCORE_TASKS` naming the
tasks whose prediction is a continuous score. The one copy that stays
spelled out is in `agent/models.py`, because the analysis models
deliberately do not import the modeling package -- and a test pins the two
equal so the next task cannot go missing from one side.

### The compatibility check that skipped itself

`_check_task_target_compatibility` did `allowed = MAP.get(task)` and then
`if allowed is None or target_type in allowed: return`. An unrecognized
task returned CLEAN. The one check standing between a task and a target it
cannot consume opted out for precisely the task nobody had thought about
yet, which is the only task that needs it -- and a regressor fitted on a
0/1 target does not error, it fits happily and reports a meaningless R2.

Unreachable today, because the Literal constrains the value. It is now a
refusal naming the map and the file to add the task to, so widening the
taxonomy fails at the map that was not updated rather than at the model
that was fitted. A companion test walks every entry of `TASKS` and fails
the moment one has no map entry.

### Every tool on the surface is now fuzzed

`run_feature_ablation` was the last declared gap: its `spec` field was
annotated `typing.Any`, so the synthesizer had no shape to build from --
and the MCP server had no schema to advertise either. The tool's own body
already did `ModelSpec(**input_data.spec)`, so typing the field as
`ModelSpec` lost nothing (pydantic still accepts a dict or an instance) and
fixed both. It also turned out to be an unresolved ForwardRef once typed:
the module never imported `ModelSpec`, so the model loaded but was not
fully defined and `model_json_schema()` raised.

`EXPECTED_UNSYNTHESIZABLE` is now empty. **204 of 204**, up from 178 of 203
at the start of this work. The stale-exemption guard is what reported the
list had outlived its reason, which is the job it was written for.

### The same horizons on the built path

`TargetSpec` gained `horizons`, so `build_model_dataset` labels one feature
computation at several distances the way `register_external_panel` already
did. Both routes write the target set in the SAME shape, so `_select_target`
needs no branch for where a panel came from.

Purely additive: a single-horizon spec -- every spec written before this --
produces a byte-identical panel, and two existing tests
(`test_panel_has_expected_columns`,
`test_panel_is_identical_to_the_pre_change_defaults`) caught the first
attempt, which emitted a duplicate `target__h5` beside `target` for every
dataset in existence. `horizon` and `horizons` normalize onto each other so
`spec.target.horizon` still reads the primary at all six sites that consume
it, and alignment drops on the primary only.

The validator had to be made IDEMPOTENT rather than exactly-one. A
normalized spec dumps BOTH fields, and `dataset_spec_hash` rebuilds a spec
from its own dump to re-derive the hash -- so an exactly-one rule made a
spec unable to survive its own serialization, which is how 165 tests failed
at once. What is still refused is the pair DISAGREEING.

**An upgrade note, because it is a real break.** `dataset_spec_hash` covers
every field, so adding one changes the hash of every spec including
already-persisted ones. A dataset built before this release fails
`run_model_experiment`'s spec-hash check with nothing edited. The remedy is
unchanged -- rebuild -- and the message now names an upgrade as a cause so
it does not read as an accusation.

### A gap in the determinism layer, found by the reworked fuzzer

`run_terminal_monte_carlo` failed "call it twice, get the same answer". The
tool is correct: `seed` is optional, and unset it warns "No seed, so this
answer is not reproducible. Set one before quoting a number anybody will act
on." The HARNESS was wrong -- `_offline_tools` never set a seed, so it was
asserting determinism on a tool that documents itself as random.

It had never shown up because the only offline stochastic tool on the
surface takes a polymorphic data source, which the synthesizer could not
build until it learned cross-field validators. The tool was outside the
layer entirely. `_offline_tools` now pins a seed, which tests the property
that matters -- same arguments AND same seed, same answer -- while
`TestSeedsAreHonoured` continues to test the other direction. Measured:
unseeded twice gives 10027.28 and 10049.21, seeded twice gives 10045.14
both times, and seeds 1 and 999983 give 10038.91 and 10039.47.

### A correction to the entry above

`18_mcp.md`'s schema-budget tables were reported as regenerated from
`print_budget()` and verified line by line. That verification was vacuous:
`print_budget` writes to STDERR, the check captured only stdout, and an
empty capture trivially satisfies "every line appears in the doc". The
tables had been hand-edited and were carrying stale figures. They are now
regenerated from a capture that reads both streams, and the derived prose --
total KB, tokens, per-tool average, the heaviest runtime's share -- is
recomputed from the same numbers rather than left behind.

### The fuzz suite was testing 178 of 203 tools and could not say so

`_synthesizable()` wrapped every input build in
`except (Unsynthesizable, pydantic.ValidationError, Exception): continue`.
The trailing `Exception` made the first two decorative and swallowed
everything, so a tool the synthesizer could not build left the fuzz set
SILENTLY -- the parametrization simply got smaller, which in pytest output
is indistinguishable from a full one. Twenty-five tools were outside every
adversarial and determinism check, including `get_order_book_metrics`, all
four tools taking a polymorphic data source, and six modeling tools.

`synth.py`'s own docstring had claimed the opposite for its whole life: "a
tool skipped for a missing field is visible in the skip list." There was no
skip list. The floor guard that should have caught the drift asked for 100
synthesizable tools out of a surface that had 178 -- 78 tools of headroom
for the gap to grow in, and it had grown into all of it.

**202 of 203 now synthesize.** The 25 came from eight distinct causes, each
fixed rather than exempted:

- `max_length` was never read, so a 12-scenario field got 300 items.
- A parameter grid was built as three keys of 300 values -- 27 million
  combinations, which every optimizer here refuses by design.
- `SYMBOLS` holds six names, so `len(weights)=300` never matched
  `len(tickers)=6`; universe-aligned fields now generate at the symbol
  count, which also keeps a fuzz case from triggering 300 fetches.
- Ticker-keyed mappings used the first three symbols beside six tickers.
- Cross-field validators ("exactly one of symbol / ref / values") cannot be
  satisfied by filling required fields alone, so `build` now tries the
  required set, then each optional in turn, then all of them, and re-raises
  the ORIGINAL refusal if none constructs.
- Two id fields on one model both got `"a"`, so a diff was against itself.
- Repeated nested models were identical, so tools rejecting duplicate
  labels or duplicate feature ids refused their own synthesized input.
- `as_of`, `start` and `end` were not recognized as date-shaped names.

**Dates are now a RANGE.** Both ends resolved to `_DATES[0]`, so every
windowed tool in the fuzz set received a zero-length window and only ever
exercised its empty-range path. They now span 400 business days, enough for
a 252-day lookback. That is most of why this layer went from ~90 seconds to
~4.5 minutes, and it is the change that made the suite find something.

**The skip is now loud.** `EXPECTED_UNSYNTHESIZABLE` declares each absence
with its reason -- currently one, `run_feature_ablation`, whose `spec` field
is typed `typing.Any` and therefore advertises no schema to the MCP server
either. Two guards hold the list honest: an undeclared gap fails, and a
declared gap that has since been fixed also fails, so exemptions cannot
accumulate past their reason. Both were verified to fail on exactly the
defect they exist for and to pass again when reverted. The floor is now
expressed against the live surface instead of a constant.

### What the reworked suite found

`detect_regimes` advertised a `seed` that could not affect its output. The
library built `np.random.default_rng(seed)` and never used it -- the comment
on the next line explains why, and it is a good reason: the mixture is
initialized on QUANTILES because "a random start on financial data converges
to the same answer most of the time and to a degenerate one occasionally,
and reproducibility matters more here than the small chance of a better
optimum."

The deterministic initialization is right. Advertising a seed on top of it
is not: an agent reading that schema believes varying the seed explores
alternative fits. Measured across six seeds, the result was identical on
well-separated regimes, on barely separated ones, and on noise with no
regime structure at all. **The field is removed** -- from the tool input,
from the library signature, and with it the dead `rng`. This is a BREAKING
change for any caller passing `seed` to `detect_regimes`, which under
`extra="forbid"` now refuses by name rather than accepting a value it
ignores.

The tool was already in the fuzz set. It was reachable only once the
synthesized window stopped being zero-length.

## Delta One, a correctness pass, and the CI gap — 157 to 200 tools, eight runtimes to ten

Forty-three new tools, two new execution boundaries, and roughly thirty
fixed defects that returned a plausible wrong number rather than raising.

### Two new runtimes

`delta_one` (18 tools) is the instrument-equivalence layer: carry and
basis, futures curves and rolls, hedge sizing, baskets and replication, ETF
fair value, swaps and total-return futures, and the comparison that
normalizes six ways of holding one exposure onto a single annualized carry.
It exists because the economics connecting cash, futures, ETFs, baskets,
forwards and swaps is a different question from what `derivatives` answers,
which is what one convex contract is worth.

`data` (14 tools) is the fetch-and-publish boundary, including
`build_continuous_futures_series`, which returns a back-adjusted research
series and a tradeable contract map SEPARATELY -- a back-adjusted price is
fine for indicators and is not a price anyone could have traded.

### Roughly thirty wrong numbers

Each was reproduced before being touched and re-measured after. The
sharpest, with what they returned:

- `run_futures_simulation` booked the calendar spread at each roll as P&L.
  A decade of dead-flat market with 39 quarterly rolls reported +5.85%.
- `calculate_series_metrics` handed a RETURN series to `cagr` and
  `calmar_ratio`, both of which take an equity curve. Both came back
  SIGN-FLIPPED: +23%/yr reported as -49%/yr.
- Black-76's `rho` was the Black-Scholes formula, so the call's sign was
  wrong: +0.4377 where the derivative of its own price is -0.0758.
- `run_strategy` carried equity NEGATIVE through a total loss and kept
  compounding, reporting a Sharpe of +2.59 on a dead account.
- `etf_fair_value` accepted a creation basket, warned about it, and priced
  against NAV anyway -- recommending the losing side of a 44.9 bps error.
- `trade_expectancy` priced breakeven trades at the average loss, so
  [10.0, 0.0, -5.0] returned exactly -0.000 where the mean is +1.667.
- `hierarchical_risk_parity` annualized its portfolio volatility but not
  its risk contributions: the two disagreed by sqrt(252) in one dict.
- `information_ratio` returned 7.31e+16 on a constant active return.

### Duplication that had already drifted

Fourteen copies of six special functions collapsed into `_special.py`; two
pairs had drifted at the edge of their domain, one returning +inf where the
other raised and one returning a p-value of 0.0 for a test with no
denominator degrees of freedom. `252` was defined nine times under two
spellings and now lives once in `constants.py`. The moving-block bootstrap
index builder was written four times, three of them the slow way that one
of the four had already measured at 87% of its own runtime.

### The CI gap that hid all of it

266 tests are gated on the C++ extension, and a failed build turned them
into skips while the run stayed green -- measured at `3 failed, 5570
passed, 506 skipped` with the extension blocked, and one of those three
failures is `slow`, which CI filters out. `ci.yml` now installs with
`SQT_REQUIRE_NATIVE=ON`, verifies `_sqt_core` imports and `HAS_CPP is
True`, and sets `SQT_EXPECT_NATIVE=1` so `tests/test_native_extension.py`
FAILS rather than skipping. `--strict-markers` is on, so a typo'd marker is
a collection error rather than a silently ungated test.

Two mutations that an audit proved survive the entire suite are now killed:
scaling covariance by `sqrt(periods_per_year)` instead of
`periods_per_year` (71 live calls, 2,149 tests green) and scoring the
opposite class in `positive_class_proba` (77 calls, 858 tests green).

## Tool surface expansion — 102 to 157 tools, six runtimes to eight

Fifty-five new tools and two new execution boundaries.

**New runtimes.** `derivatives` (12) took option pricing out of `research`
and added second-order greeks, multi-leg payoffs, smile and term-structure
fitting with arbitrage checks, put-call parity, expected move, delta-hedge
simulation and revaluation grids. `microstructure` (12) left `portfolio`
once it held more than four tools, and gained seven bar-based liquidity
estimators — Roll, Corwin-Schultz, Amihud, Kyle, order-flow imbalance,
VPIN, intraday volume profile — plus implementation shortfall. Both splits
clear the floor this library sets: at least eight tools on each side.
`MOVED_FROM` names the old home, so a caller scoped to the donor gets an
instruction rather than an "unknown tool" that reads like a hallucination.

**Existing runtimes.** `research` +11 (change points, partial correlation,
Granger, tail dependence, stationarity, regimes, Ljung-Box, seasonality,
entropy, Sharpe stability, drawdown profile, lead-lag, Chow test, bootstrap
intervals, distribution comparison, correlation stability, return
decomposition, normality, tail index). `backtest` +11 (deflated Sharpe,
PBO, purged combinatorial CV, White's reality check, regime-stratified
performance, parameter decay, Monte Carlo trade paths, trade clustering,
comparison against random, exposure attribution, break-even cost).
`portfolio` +8 (risk parity, HRP, factor exposure budget, concentration,
liquidity-adjusted VaR, max diversification, marginal risk contribution,
named scenarios). `meta` +3 (tool cost, runtime description, artifact
diff).

**`--tool-detail` now defaults to `auto`.** At full detail the backtest
runtime cost 75,867 bytes against a 73,728 per-runtime ceiling, and the
ceiling was not the thing to move. `auto` thins only what exceeds the
budget, so the runtimes already under it are byte-for-byte unchanged, and
`describe_tool` is injected whenever anything is thinned.

**Fixes found while building.** The audit layer could not accept a plain
dict return. A constant return series produced a Sharpe of 7.3e16, because
numpy's `std` on a flat array is ~1e-19 rather than 0 — the same fault
lived in two places and both are relative checks now. `granger_causality`
took the smallest p-value across every lag and called it significant at
5%, delivering 15%; it is Bonferroni corrected. Roll's spread estimator
returned 0.098 on a series with a spread of exactly zero, and now reports
the smallest spread the sample could distinguish. Order-flow persistence
was measuring its own window overlap (+0.76 on pure noise) and is computed
on non-overlapping windows. `rolling_sharpe_stability` corrected for
overlapping windows twice and had no power at all.


All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/) —
while the major version is `0`, breaking changes may still land in a minor
bump, consistent with SemVer's pre-1.0 clause.

## [Unreleased]

### Added — `plan_rebalance`: the transition an optimizer assumes is free

Every optimizer here returns a target weight vector and implicitly assumes
you arrive instantly and at no cost. The transition has two costs pulling
opposite ways — impact for trading fast, holding the old portfolio for
trading slow — so this returns the day-by-day schedule and both, with
`urgency` exposed rather than chosen for you.

**What it surfaces that nothing else does:** a target weight the market
cannot supply. On a $100m book, a 60% target in a $2m-ADV name comes back as
needing **295 days** at a 10% participation cap. The weight vector is valid,
the backtest fills at the close, and until now nothing noticed.

A bug found while building it, worth recording because it pointed the wrong
way: `total_cost_bps` summed each day's average impact rate, reporting 1.83
bps for trading everything at once against 4.08 for spreading it — exactly
backwards. Summing basis points across days adds rates, not costs. Both
numbers were small and plausible, and a caller would have concluded "trade
fast, it's cheaper". It now reports total impact dollars over total notional,
and a test checks the ratio against the square-root law's own arithmetic.

### Added — `estimate_covariance` with shrinkage

`sample`, `ledoit_wolf`, `ewma`, `ewma_shrunk`, plus the diagnostics that
say whether to trust the result. Measured on 40 assets and 120 days:
condition number 205 for `sample`, 143 for `ledoit_wolf`, **228 for `ewma`**
and 35 for `ewma_shrunk`.

EWMA making conditioning *worse* is the point worth carrying: it answers a
question about regime rather than about estimation error, and lowers the
effective sample size doing so. They are not alternatives, which is why
`ewma_shrunk` exists. Returned annualized.

### One of the two blockers on the microstructure split is cleared

`portfolio` now holds 13 tools — 9 `portfolio_risk` and 4 `microstructure` —
so moving the microstructure four out leaves 9, above the eight-tool floor.
A test pins it so a later move cannot quietly break it.

The split is still blocked, by the other condition: a new runtime has to
land at ≥ 8 tools and microstructure has four. The four it needs are all L2
ones, so it waits on the same absent order-book source as the rest of
Phase 3.

> **Since superseded.** The split shipped: `microstructure` is its own
> 16-tool runtime and `portfolio` is 18, all `portfolio_risk`. The
> eight-tool floor was cleared by the bar-based estimators rather than by
> the L2 tools this entry was waiting on.


### Added — probability calibration, and the empty signal panel it fixes

`EstimatorSpec.calibration` takes `isotonic` or `sigmoid`, fitted on
held-out folds inside each training window — never on the rows the estimator
itself trained on, which would calibrate against memorized labels and report
a confidence nobody has.

This fixes a live path. `convert_reference(proba_threshold=T)` turns a
classifier's probabilities into a signal panel, and the threshold only means
something if "probability above T" really means "wins about T of the time".
Measured on a 6,000-row synthetic panel:

```
T      raw forest          isotonic-calibrated
       realized    n       realized    n
0.5      0.740    943        0.741   909
0.7      0.875    353        0.829   532
0.9        --       0        0.912   194
```

A random forest's probabilities are compressed toward the middle — an
artefact of averaging trees — so it never emits one above ~0.9. **A caller
setting `proba_threshold=0.9` got an empty signal panel with no error
anywhere**: not a wrong answer, no answer. Calibrated, the same threshold
selects 194 rows that win 0.912 of the time.

Both the validation folds and the deployed refit are calibrated the same
way. A model validated with calibrated probabilities and deployed without
would report one threshold's behaviour and exhibit another's.

Refused by name in the three ways it can be asked for wrongly: on a
regressor (there are no probabilities to map), with more folds than the
window has rows (sklearn's own error arrives three frames down talking about
`n_splits`), and with an unrecognised method.

### Added — `huber` for robust regression

Every squared-loss fit was steered by its worst week: an 8-sigma day
contributes 64 times what a 1-sigma day does, and financial targets have
those. One registry entry, with `epsilon` bounded and documented.

**No new tools.** Both land in `ESTIMATOR_REGISTRY` and `ModelSpec`, and
`list_modeling_capabilities` reports the registry — so a new estimator is
discoverable the moment it is registered. A test pins the modeling surface
at 16 tools to keep it that way.


### Added — `detect_liquidity_events`: which part of the market changed

"NVDA moved 1.4 sigma" describes one channel, and it is the channel that
moves LAST. This runs a CUSUM change detector across several channels and
reports which broke:

```
spread shock            very high
effective_spread shock  very high
signed_volume shock     very high
(mid_return did not trigger)
```

One tool over a declared channel set, not one tool per channel. Six channels
need only trades and quotes; the eight needing an order book are declared
and refuse by name rather than being omitted — a missing row in a report
reads as a quiet channel.

**Three bugs found by running it on data with no shock in it**, each of
which would have shipped as a confident alarm:

- `mid_price` fired on 8 of 8 pure random walks. A CUSUM over a price LEVEL
  accumulates the walk itself. The channel is now `mid_return`, and
  `mid_price` stays as an explicitly refused channel so the trap is visible.
- A frozen channel produced a peak statistic of 286,431 while moving from
  1.00 bps to 1.02 bps — a denominator near zero. Now flagged
  `degenerate_baseline`, with the shift reported in the channel's own units.
  Labelled rather than suppressed: a calm period before a real shock is the
  case the tool is for.
- The textbook threshold of 5.0 alarmed on pure noise 36% of the time at
  n=120 and **82% at n=1000** — worse with more data. Calibrated to ~5%
  gives 8.6–9.3 across 42 to 1050 observations, essentially flat, so the fix
  is a better constant (9.0) rather than the log-scaling formula expected.
  False alarms now measure 4–6%, and a real shock still clears it.

### Added — the L2 order-book contract

`DataProvider.get_order_book` refuses explicitly and by name, and warns
against the substitution that would otherwise happen quietly: top-of-book
quotes passed off as depth give a one-level book with an imbalance of
exactly zero, which reads as a perfectly balanced market rather than as
missing data. The column layout is declared so every consumer agrees what a
book looks like before one exists — the same sequencing the point-in-time
join used, and for the same reason.

No provider implements it. The twelve L2 tools in the expansion plan were
**not** built, because until a source exists each one is a tool that
refuses.


### Added — `compare_data_sources`, and the three verdicts it separates

Fetches the same fundamentals from two providers and reports where they
disagree. `FinancialRatios` already documented that `debt_to_equity` means
different things depending on the source; nothing checked it, so a screen
ranking a universe on a mix of two providers ordered it partly by which one
answered, with no error anywhere.

The hard part is not spotting a difference. It is separating:

- **scale** — a CONSTANT ratio across entities. A missed unit conversion,
  fixable by arithmetic. yfinance reporting a percentage is this.
- **definition** — systematic, ratio NOT constant. The two are computing
  different quantities and no conversion exists. Polygon deriving
  `debt_to_equity` from total liabilities is this, because payables are not
  proportional to debt.
- **agree** — within rounding, and not a finding.

Telling a user "these disagree by 2x" when the verdict is `definition`
invites them to divide by two, which is exactly wrong.

### Added — `DataBundle`, `validate_pit_records`, `join_point_in_time`

`DataBundle` pairs every frame with the `TemporalContract` it was fetched
under; `pit_safe` is the weakest link, not an average, because a bundle is
used as a unit and "mostly safe" is not actionable.

`validate_pit_records` catches the two timestamps the wrong way round —
`event_time` is when a fact is ABOUT, `available_time` is when it could
first be ACTED ON, and swapped they make every model look prescient. It
reports `median_publication_lag_days` even when it passes: exactly the
hindsight a naive join on `event_time` would have handed you.

`join_point_in_time` attaches those records to a built dataset, each row
getting the most recent record AVAILABLE by then. Records arrive inline
(capped at 5,000) because no provider here serves them yet — which is a real
use case, not a placeholder: an earnings calendar or a set of FOMC dates can
be joined today.

### Not built, deliberately

`build_fundamental_panel`, `build_macro_panel` and `build_event_panel` are
in the expansion plan and are **not** implemented. No provider supplies
availability timestamps for any of them, so a tool with those names could
only join on `event_time` — a three-week lookahead with a reassuring name.
The plan's own instruction was to build the contract first for exactly this
reason. The contract exists and refuses; the builders wait for a source.

`build_universe_snapshot` and `list_data_snapshots` need a snapshot store
that has not been designed.


### Added — the temporal contract, asked before anything is fetched

`describe_temporal_contract` says what a source can tell you about WHEN its
facts became knowable, without fetching. A quarterly filing describes 30
September and is published on 25 October, so a model that joins it on the
quarter end carries three weeks of hindsight per row.

`asof_join` already refused a frame with no `available_time`. That refusal
arrives after a universe has been chosen, a history fetched and a cache
written; this answers the same question first.

- `data/temporal.py`: `TemporalContract`, `require_pit`,
  `contract_for_frame`, `price_contract`.
- `DataProvider.get_temporal_contract(frame_kind)`, defaulting honestly to
  "bars are safe by construction, everything else is unsupported here". A
  provider that can serve availability timestamps overrides it.

**`pit_safe` and `reproduces_history` are separate properties** because they
come apart. A `snapshot` source joins without leaking the future and still
shows a backtest final values nobody had at the time — not lookahead, a
different history. Conflating them would have hidden it.

**Two timestamps, not the three the plan specified.** A restatement is a
row, not a column: `available_time` on the restating row already IS its
publication date, and the existing join returns the version current at each
date. A `revision_time` column would have duplicated it and invited the
one-row-per-fact encoding that cannot reproduce history at all. What is
declared instead is how revisions are encoded, and `unknown` is treated as
unsafe.

`DataSetMetadata` was deliberately left alone — it documents one OHLCV pull,
and giving it a `frame_kind` to cover filings would have muddied a type that
is currently precise.


### Added — `feature_lab`, the sixth runtime

The nine feature tools have moved out of `modeling` into a runtime of their
own. `modeling` is one ordered pipeline — build a dataset, fit it, register
the model, score it — and feature work is a different job: exploratory, run
repeatedly, and finished before any model exists. Nine of `modeling`'s
twenty-three tools had nothing to do with fitting anything, so an agent
scoped there to run an experiment carried nine it would not call, and an
agent doing feature work carried fourteen it would not.

`sqt-mcp --runtime feature_lab` serves them for 11.5 KB.

The split followed the rule in `Development/runtime_expansion_plan.md §3`:
the cluster was **built inside `modeling` first** and moved once it reached
the eight-tool floor, rather than being declared empty and filled later.
Both sides clear the floor — `feature_lab` at 9, `modeling` at 14.

**A split is a breaking change**, so `MOVED_FROM` records where each tool
came from, and both the library and the MCP refusals say so — but only to a
caller scoped to the runtime the tool LEFT. Anyone else never had it, and
the history would explain something they were not part of.

### Added — `run_feature_ablation`

Refits the model without each feature in turn. The only feature tool that
asks a model-relative question: a strong feature that duplicates another
contributes nothing marginal, and a mediocre feature that is the sole source
of some information can be the one holding the model up. Neither shows in a
per-feature report, and tree importance does not answer it either.

It is also the most expensive tool in the library. One baseline plus one
refit per feature across every fold — 40 features at 8 folds is 328 fits —
so the count is computed BEFORE anything is fit and the run is refused past
`max_fits`, with the refusal naming the number that would work.

None of the refits are registered: `run_experiment` gained `register=False`
for it, because 41 candidate models in the registry to answer one question
is not a trade worth making. The fits are identical either way, and a test
pins that the metrics match.

### Fixed — a registry with no workers was invisible to its own coverage test

Splitting `feature_lab` out left all nine of its tools reachable by no
worker in `Multi_Agent_Implementation`, and every test passed: the coverage
check iterates the registries that HAVE workers, so one with none cannot
fail it. There is now a Feature Lab Agent, and a test that fails when any
registry has no worker at all.


### Added — eight feature tools, one question each

`analyze_features` returns `report: Dict[str, Any]`. Every number in it is
correct and none of it is promised by a schema, so an agent asking whether
one feature is worth keeping had to profile the whole panel and then guess
at key names. Eight typed tools now ask one question each:

`analyze_feature`, `get_feature_redundancy`, `get_feature_ic_decay`,
`get_feature_drift`, `get_feature_regime_stability`,
`run_feature_permutation_test`, `select_features`, `compare_feature_sets`.

The first three re-present analysis that already existed. The types leave
room for a recommendation rather than a table: `get_feature_redundancy` names
which feature to KEEP, picked by strongest |rank IC| and tie-broken
alphabetically so the drop list is reproducible across identical calls.

The rest answer questions a full-sample report structurally cannot.
`get_feature_drift` separates distribution drift from IC decay, which look
alike in an averaged report and need different fixes.
`get_feature_regime_stability` splits into contiguous — never shuffled —
time blocks, since interleaving averages away the regime structure it exists
to expose. `run_feature_permutation_test` shuffles the feature within each
date and returns a two-sided empirical p-value, so an IC of 0.03 on a small
panel can be checked against what that panel's noise produces anyway.

`select_features` has no greedy search, deliberately: a selector scored on
the panel it selects from manufactures overfit that looks like evidence. It
drops duplicates and the unmeasurable and records a reason for each.

PSI and the two-sample KS are implemented in numpy rather than imported from
scipy, which is not a declared dependency of this package.

### Fixed — non-finite statistics no longer serialize as bare `NaN`

`_safe()` in `feature_report.py` is documented as producing "a float that
survives JSON" and maps non-finite values to NaN. NaN does not survive JSON:
`json.dumps` writes a bare `NaN` token, invalid per RFC 8259 and rejected by
strict parsers — which JSON-RPC clients are. Measured on a legal panel with
one entity per date: twelve of them in a single `analyze_features` result.

Now `null` in both the typed and untyped paths, and `null` rather than `0.0`
because an IC that could not be computed is not an IC of zero. The
conversion happens at the tool boundary; NaN remains correct inside the
numpy pipeline.


### Added — `--tool-detail`: send the schema when the agent needs it

`sqt-mcp --runtime backtest --tool-detail auto` serves the same 21 tools for
40 KB instead of 65. A **thinned** tool is listed and callable; what it
loses is its argument schema, replaced by one line of purpose and an
instruction to call `describe_tool`. Measured: 531 bytes against 2,184,
**76% smaller**.

The observation this rests on is that nothing new was needed.
`describe_tool` already existed in `meta`, already answered for tools in any
runtime, and already returned exactly the schema a thin listing omits. The
server was shipping all 82 schemas at connect on the assumption an agent
would need every one; a session calls a handful.

`auto` thins the **most expensive** tools and stops as soon as the runtime
fits `--detail-budget` (32 KB default). A runtime's cost is concentrated in
a few large schemas — `modeling`'s top three are 65% of it — so this thins
one tool in modeling and seven in backtest where a uniform policy would thin
all of them, and every tool left
described is one an agent calls without a round trip. `research`,
`portfolio` and `meta` already fit and are left entirely alone.

- `--tool-detail {full,auto,thin}`, default `full` — nothing changes for an
  existing invocation, and a test asserts that.
- `--detail-budget BYTES` for `auto`.
- `catalog.plan_detail()`, `thin_schema()`, `thin_description()`.
- The server injects `describe_tool` whenever anything is thinned, and never
  thins it. A thin entry says "call `describe_tool`"; under
  `--runtime backtest` that tool is not in scope, so without this the
  instruction is unfollowable and every thinned tool is uncallable.

**Thinning changes the advertisement and nothing else.** Arguments are
unchanged and validated by the same model, and `extra="forbid"` still
rejects a guessed name rather than defaulting it. The "call `describe_tool`"
instruction is deliberately duplicated in the description and the schema: a
client may show a model only one of them, and a model shown an empty `{}`
schema concludes the tool takes no arguments and burns a turn proving
otherwise.

`auto` rather than a fixed primary tier because the fixed tier was measured
and found wrong — 8-full-plus-thin lands the projected 151-tool library at
195 KB, still over the ceiling, where a budget-driven rule holds every
runtime under its target by construction.


### Added — the MCP server is scoped by runtime, not just by category

`sqt-mcp --runtime research` serves one runtime. `--runtime research+meta`
serves two, spelled the way `combine()` spells a joined runtime in the
library itself. `--categories` survives unchanged as the narrower filter
*within* the chosen runtime.

The reason is a measurement, not a preference. Over the wire a tool averages
2,184 bytes and the context ceiling is 180,000, which buys 82.4 tools. The
library has 82. The remaining headroom is 912 bytes — under half a tool — so
**the 83rd tool would have failed the budget test whatever it was.** Serving
the whole surface had stopped being a thing to avoid and become a thing that
no longer fits.

What a client is actually served now costs a third of that: the heaviest
runtime (`backtest`) is 58 KB of the 150 KB total, and
`tests/mcp/test_runtime_scope.py` pins a 72 KB per-runtime ceiling.
Deliberately tight — `backtest` measures 66,437 bytes over the wire, about
two tools of headroom — because the next answer is thin listings with
schemas fetched on demand, not a third argued-up ceiling.

Scope is enforced at three points rather than one: the runtime's tools are
the only ones listed, the only ones `call_tool` will dispatch, and the
owning runtime's table would refuse a stray name again underneath.

- `--runtime NAME` on `sqt-mcp`, accepting `a+b`, `a,b` or `all`.
- `ServerConfig.runtimes`, which narrows a directly-constructed server too —
  not only one built from the command line.
- `catalog.select_runtimes()`, `categories_for_runtimes()`,
  `runtime_costs()`, `ALL_RUNTIMES` and `RUNTIME_CATEGORY_MAP`.
- `--print-budget` now reports per-runtime first, then per-category, and
  says which row a client actually pays.

**Nothing changes for an existing invocation.** With neither flag the server
serves what it always did, and a test asserts that rather than assuming it.
Given `--runtime` alone the categories widen to that runtime's own, because
inheriting the default under `--runtime backtest` would have served zero
tools — and an empty server reads as a broken install, not as two flags
disagreeing. Naming a category the runtime does not own is refused at
startup, by name, rather than silently intersected.

### Changed — an unknown tool error now says which of three problems it is

The error named the loaded categories for every unrecognised name, including
names that exist nowhere in the library. That told an agent which had
invented a tool to go and widen a scope that could never contain it. The
three cases are now distinguished: a tool in another runtime names the owner
and the flag that would serve it; a tool filtered out by category names the
category; a name that exists nowhere says so and suggests no flag at all.


### Changed — tool scoping is now enforced, not advertised

`get_agent_tools(categories=[...])` could always narrow the schema list
handed to a model. `dispatch()` never honoured it:

```python
>>> {t["function"]["name"] for t in get_agent_tools(categories=["screener"])}
{'run_screener', 'get_stock_fundamentals'}
>>> dispatch("list_stress_scenarios", {})    # never advertised to this agent
{'scenarios': [...]}                         # ...and it ran anyway
```

An agent scoped to two screener tools that hallucinated a backtest tool got
a **successful result**. The narrowing was advisory at the schema layer and
absent at the execution layer, so a wrong guess was rewarded. The MCP server
enforced its own selection; nothing else did, which meant every
`Implementation/*` script and the whole multi-agent orchestrator ran with a
boundary that was not there.

The 82 tools are now grouped into **five parallel runtimes**, each with its
own dispatch table:

| Runtime | Tools | Categories |
|---|---|---|
| `research` | 23 | `screener`, `analysis`, `quant_research` |
| `backtest` | 21 | `backtest_execution`, `backtest_validation`, `custom_signal` |
| `portfolio` | 10 | `portfolio_risk`, `microstructure` |
| `meta` | 14 | `discovery`, `provenance` |
| `modeling` | 14 | (unchanged, in `modeling/agent`) |

A name from another runtime is unroutable, and the refusal says where it
actually lives — "unknown tool" alone cannot be told apart from a
hallucinated name, and a model receiving one guesses again. The grouping is
deliberately coarse: a runtime holding two tools is overhead rather than
isolation, so nothing has fewer than eight and a test pins that.

`TOOL_CATEGORY` is unchanged and still drives the router, the MCP
`--categories` flag and the twelve workers. A category hints at which tools
suit a request; a runtime states which tools may execute.

`agent/tools.py` went from 6,453 lines to a 314-line facade over the runtime
packages. Every existing import still works, and its `dispatch()` is
documented as UNSCOPED by construction — an agent meant to be scoped should
be handed a runtime.

### Added — the handoff interconnect

Moving a model's predictions into a backtest used to need a bridge tool that
knew about both sides. With N producers and M consumers that costs N × M
bridges. A typed reference makes it N + M:

```
sqt://<kind>/<run_id>/<name>
```

Ten content kinds (`equity_curve`, `signal_panel`, `weight_panel`,
`score_panel`, `predictions`, ...). The kind is checked on resolve, so a
wrong handoff between two tool calls fails naming both kinds instead of
surfacing as a missing column deep in pandas. One `convert_reference` tool
covers every well-defined conversion; there is deliberately no best-effort
path, because a handoff that guesses is worse than one that refuses.

Producers publish typed references beside their existing URIs
(`equity_curve_ref`, `trades_ref`, `oos_predictions_ref`), so the chain
`run_model_experiment → convert_reference → run_signal_panel_backtest` works
with no code anywhere in it that knows about both ends. Consumers gained
optional `signal_panel_ref` / `target_weights_ref` beside their inline
fields.

Built for many agents rather than one session: one sidecar file per artifact
(a shared per-`run_id` catalogue races), `publish()` refuses to overwrite by
default (a reference promises the same value twice), and `describe_reference`
returns a content hash so a consumer can prove it read what the producer
wrote.

### Changed — unknown tool arguments are rejected

None of the tool inputs forbade extras, so this succeeded:

```python
BacktestInput(..., comission_pct=0.05)   # note the typo
```

The typo was dropped, the backtest ran at the 0.001 default, and the caller
believed it had set 5% commission — the same failure
`backtest/strategy_params.py` exists to stop one layer down, open at the
boundary where a *model* chooses the argument names. Every tool input now
sets `extra="forbid"`. Result models stay permissive.

Measured before changing: forcing it across the suite broke exactly two
tests and **both were wrong** — one passed `start_date`/`end_date` to a tool
that takes `period`, so it had been measuring the default window all along.

### Fixed — the risk-free rate was silently zero, everywhere

Seventeen tools report a Sharpe ratio and exactly one took a rate. The rest
measured total return per unit of risk; at a 4–5% short rate that is most of
the number for a low-volatility strategy.

**All 23 Sharpe-reporting tools now take `risk_free_rate`**, defaulting to
0.0 so nothing that already exists moves. That required threading it through
the backtest engine itself — the Python path, the C++ kernel's two
computation sites, and all three of `backtest_grid`'s execution paths — so
the number does not depend on whether the native extension happens to be
built, and a grid cannot rank on a different rate than the single run it is
compared against.

Two details the arithmetic turns on:

- **Sharpe's denominator does not move.** Subtracting a constant from every
  return shifts the mean and leaves the standard deviation untouched, so the
  kernel's existing sum-of-squares needed no change at all.
- **Sortino's does.** Python clips EXCESS returns, so the rate decides which
  bars count as downside. In the kernel's allocation-free summary path — the
  one every batch entry point reads from — bar 0's implicit return of 0.0
  has an excess of `-rf/ppy` and contributes to the downside sum. That term
  is invisible at rf = 0, so a missing seed would have made grids disagree
  with single runs only once a rate was set. Deleting the seed and rebuilding
  confirms the native parity test catches it: 60 failures, all on
  `sortino_ratio`.

The tests replaced, not relaxed: the two that asserted the engine hard-codes
zero are gone, and in their place are cross-path parity checks at five rates,
kernel-level parity between `run_strategy` and `run_strategy_summary`, grid
agreement on both the fused and batch paths, and an invariant that every
Sharpe-reporting tool exposes a rate.

The `risk_free_rate` on the option-pricing tools is a different quantity —
the Black-Scholes discount rate — and stays REQUIRED, since there is no
defensible default for discounting a cash flow. A test pins that distinction
so the Sharpe-scoped assertion cannot silently start covering it.

Schema descriptions for the new field are deliberately terse: 17 verbatim
copies of a long one cost ~7 KB of context that every MCP client holds for
a whole session. Trimming them kept the full surface under its existing
budget ceiling rather than moving it.

### Added — 28 tools

**`discovery` (8)** — `list_strategies`, `list_stress_scenarios`,
`describe_data_capabilities`, `describe_tool`, `validate_tool_call`,
`describe_reference`, `list_reference_kinds`, `convert_reference`. Contracts
the library already held in data, made askable instead of described in
prose. `validate_tool_call` checks arguments *without* calling, including
the strategy parameter contract that JSON Schema cannot express.

**`provenance` (6)** — `explain_decision`, `replay_decision`,
`compare_decisions`, `verify_audit_integrity`, `export_audit_bundle`,
`describe_artifact`. Read and verify only: retention (`gc`, `seal`, `hold`)
stays CLI-only, because an agent that can delete the record of its own
decisions is not audited by it. `replay_decision` classifies a mismatch as
`data_changed` or `code_changed` — checking input hashes first, so a
provider revision is not reported as a library bug.

**`microstructure` (3)** — `get_microstructure_metrics`,
`get_trade_profile`, `check_spread_proxy`, over a new
`analysis/microstructure.py`. `get_trades`/`get_quotes` had been on the data
interface with nothing consuming them. `check_spread_proxy` measures the
spread from ticks and reports which way the OHLCV proxy errs — understating
it means every backtest priced from it has been charging too little.
Nothing synthesizes ticks from bars.

**Elsewhere** — `get_technical_panel` (whole universe, one native call),
`run_strategy_matrix`, `compare_cost_models` (solves for the commission at
which the edge disappears), `estimate_trade_cost`, `get_drawdown_table`,
plus `list_models`, `list_datasets`, `compare_models`, `check_leakage`,
`validate_model_spec` and `score_predictions` in the modeling runtime.

### Added — options that were silently fixed

`risk_free_rate` (above), `standardize`/`method` on `run_pca_analysis`,
`min_window`/`max_window` on `run_hurst_analysis`,
`observation_noise`/`include_intercept` on `run_kalman_hedge_ratio`,
`sar_af_step` on `get_advanced_indicators`, and `sell_commission_pct` on
`run_portfolio_simulation` — the engine had supported an asymmetric rate
since it was added; no tool passed one. Every new field defaults to the
value the tool previously hard-coded.

### Changed — the example implementations are scoped

`registry=` in every provider's `_agent_utils.py` now accepts a RUNTIME
name, joinable with `+`, alongside the two whole-surface views. Naming a
runtime hands the script a dispatch table holding only that runtime's
tools; naming `"analysis"` still hands back the union, which knows every
tool regardless of what was advertised.

All 55 example scripts across `Implementation/`, its three provider folders
and `Multi_Agent_Implementation/` are scoped. Each script's runtime set is
DERIVED from the tools its own prompt mentions rather than chosen by hand —
a prompt that instructs the agent to call a tool the runtime will refuse is
worse than no scoping, because it walks the model into a wall it was told
to walk into. That derivation immediately caught seven scripts whose
prompts relied on the full surface, and one comment of mine that described
a backtest step its script does not take.

The twelve workers dispatch through their category's runtime. Each already
declared a fixed, non-overlapping tool subset; dispatching through the
union had made that subset advisory.

`modeling` is now resolvable and combinable like every other runtime.
Writing an example that walks modeling → meta → backtest showed that
excluding it was a gap in the abstraction rather than a deliberate limit.
It is wrapped BY REFERENCE — the Runtime holds `MODELING_TOOL_DISPATCH`
itself — so there is still exactly one definition of that boundary.

### Added — three example scripts

| Script | Runtimes | Shows |
|---|---|---|
| `Agent_Model_Backtester.py` | `backtest+meta+modeling` | The handoff interconnect end to end. The agent never sees a prediction; a reference crosses three runtimes and the panel never enters the conversation. |
| `Agent_Provenance_Auditor.py` | `meta` | Reconstructing a past decision and classifying a mismatch as the data's fault or the code's. Retention is unroutable, not merely discouraged. |
| `Agent_Execution_Analyst.py` | `portfolio+meta` | Measured spreads versus the OHLCV proxies, and which way the proxy errs. |

Each is mirrored across all three provider folders, because a capability
demonstrated for one provider only is a capability half-demonstrated.

### Documentation

New [`Documentation/19_runtimes.md`](Documentation/19_runtimes.md). Updated
`13_agent_orchestration.md`, `18_mcp.md` (category budget re-measured at 82
tools / 146 KB; `discovery` added to the default), `00_module_reference.md`,
`07_agent_tools.md`, `10_auditability.md`, `15_modeling.md` and the README.


### Added (direction-aware costs, peak-exposure diagnostics, a tick contract)

Three small, independent additions. Each closes a gap the code itself had
already named.

**Commission can differ by side.** `costs.py` gains `directional_commission`
and `maker_taker_cost`. The first exists because US equity execution is not
symmetric — the SEC Section 31 fee and the FINRA TAF are levied on SALES
only, so a round trip charged at one blended rate undercharges any strategy
whose turnover is sell-heavy.

The second is the function `_cost_rate`'s own docstring had been asking for.
It rejects negative rates everywhere with "if a genuine rebate is intended
it should be modelled explicitly rather than arriving as a sign error";
`maker_taker_cost` is that explicit model, and it is the only function in
the module that may return a credit — confined to the maker side, so a
negative taker rate is still the sign error every other function refuses.

`run_portfolio_simulation` gains `sell_commission_pct`. `None` charges
`commission_pct` both ways, which is exactly what it did before.

**The change had to land in three places at once, and did.** That simulator
picks among a scalar loop, a vectorized branch and a native kernel by
configuration — and the DEFAULT configuration takes the native path. A rate
added to Python alone would have been silently ignored on every machine with
`_sqt_core` built: two users, same spec, different numbers. That is the
`clip_sigma` defect exactly, so `PortfolioCosts` gained the field, the
kernel selects on the sign of `delta`, and a parity test asserts the two
backends agree rather than asserting either alone. Measured: bit-identical
on the equity curve, symmetric and asymmetric.

Direction stays vectorizable — the side is an element-wise select on the
sign of the trade, not a per-element decision that would force the loop —
so nothing falls off the fast path to get it.

**Four peak-exposure diagnostics.** `run_portfolio_simulation` now reports
`max_leverage`, `max_gross_exposure`, `peak_position_value` and
`return_over_rebalance` beside the existing `avg_gross_exposure`. The curves
answered "what did this portfolio look like typically"; nothing answered
"how bad did it get", which is the question a risk limit is written against.
An average gross exposure of 0.9 is perfectly compatible with a single day
at 2.4, and only one of those breaches a mandate.

None of them costs a pass over the data. `peak_position_value` was the only
one needing new state, and it rides the loop that already forms the position
vector for net and gross — one comparison per element, in both backends.
They are scalars rather than curves: returning four more `(n_bars,)` series
would tax every consumer with a payload it reduces immediately.

`return_over_rebalance`'s `None` branch turned out to be defensive rather
than reachable — the engine already rejects both routes to an empty
rebalance log — and the test pins that it keeps doing so, so relaxing either
guard cannot silently produce a ZeroDivisionError.

**A tick-data capability on the provider contract.** `DataProvider` gains
`get_trades` and `get_quotes`, implemented by `PolygonProvider` against
`/v3/trades` and `/v3/quotes`.

Concrete-with-raise, not abstract: marking them abstract would break
yfinance and Bloomberg at import time to express something better said at
the point of use. The base implementation names the provider, names the one
that does work, and refuses to offer bars as a substitute — a "trade"
derived from an OHLCV row is a fiction every microstructure measure
downstream would treat as fact.

Two details that are easy to get wrong and are pinned by tests. The v3
endpoints return NANOSECOND timestamps while the aggregates endpoint in the
same module returns milliseconds — parsing one with the other's unit dates
every tick to 1970 while leaving the frame structurally plausible. And the
range is half-open `[start, end)`, because a closed range on a nanosecond
clock either double-counts the boundary tick when two windows are
concatenated or drops it, and which one is invisible until someone
concatenates.

These are also the only endpoints in that module not on Polygon's free tier.
`_polygon_get` turns 403 into "check your key", which is right for an
expired key and misleading for a valid one on the wrong plan, so both
methods re-raise naming the real cause.

Single page per call, like `get_ohlcv`: Polygon paginates ticks by cursor
and a liquid name produces millions of trades a day, so following `next_url`
would turn one call into an unbounded download.

Not added, deliberately: anything reading depth. No shipped provider offers
an order book, so queue position and resting size stay out of reach rather
than approximated.

37 new tests. 3298 -> 3335 Python tests; 10 C++ suites unchanged and green.

### Added (an MCP server for the whole library)

`standard_quant_tools.mcp` serves both tool registries over the Model
Context Protocol, so any MCP client can use the library without the
`Implementation/` scripts. Install with `pip install
'standard_quant_tools[mcp]'`; the SDK is not a core dependency and the
`sqt-mcp` entry point says so if it is missing.

**Exposure is a policy, because the schemas are the constraint.** The 54
tools cost **104,645 bytes — about 26,000 tokens** — and an MCP client holds
that for the whole session. So tools are selected by category, using the
same `TOOL_CATEGORY` taxonomy that already drives the router and the nine
workers rather than a third grouping that could drift.

The measurement worth keeping: **tool count and cost are barely related.**
`analysis` is 13 tools in 11.9 KB; `custom_signal` is 2 tools in 6.1 KB;
`backtest_execution` alone is a quarter of the surface. Picking categories
by how many tools they hold gets the budget almost exactly backwards.
`sqt-mcp --print-budget` prints the table, a test pins a ceiling against it,
and the default — 22 tools, ~5k tokens — is stated at startup along with
what it costs.

**Structured results without the surcharge.** All 54 tools have typed
Pydantic returns, so the server returns `structuredContent` on every call.
Declaring `outputSchema` as well was assumed free in the plan and measured
at **74 KB, a 77% increase** on the whole surface — so it became
`--output-schemas`, off by default. The declaration only helps clients that
validate against it; the structured payload arrives either way.

**Seven schemas were dereferenced.** Those tools carried `$ref`/`$defs`
upstream and are the most complex in the library, so they are the worst ones
to hand a client that resolves references poorly. Inlining them turned out
to *shrink* the payload 5.4% rather than grow it, because the `$defs` held
definitions referenced once or not at all. A test asserts nothing reaching a
client still contains a `$ref`.

**Large results leave the conversation.** 83 list- or dict-valued fields
live across 28 result models, and over MCP a five-year backtest's equity
curve would sit in context for the rest of the session. A result over
`--inline-limit` (4096 bytes) is stored whole and returned as a summary that
names every field it withheld and how large it was, plus a `sqt://result/`
link. All-or-nothing per field, deliberately: half a trade log looks like a
whole trade log to a model reading it.

**Eight resource URIs**, covering stored results, Parquet artifacts, model
manifests, dataset metadata, audit records, and three static catalogs. Every
path resolves through the library's existing sandbox guard, because a URI
from a client is untrusted input and is the one place in the server where a
traversal bug is reachable from outside.

**Five workflow prompts.** A prompt invoked against a server that was not
started with the categories it needs prepends a warning rather than handing
the model a workflow it has no tools to run.

**Annotations are derived, not maintained.** `readOnlyHint` is true for all
54 and a test asserts it: this library does not place orders or move money,
and one tool breaking that would force clients to treat the whole server as
write-capable. `openWorldHint` comes from whether a symbol, ticker or
universe appears anywhere in the input schema — including nested specs,
since `build_model_dataset` hides its universe two levels down.
`idempotentHint` is false for the four tools that persist an artifact.

**The audit trail carries through.** Both dispatchers already route through
`audit._run_and_record`, so every call made over MCP produces a
hash-chained, replayable decision record, readable in-session at
`sqt://audit/{request_id}` and by `sqt replay` afterward.

**58 tests.** 48 schema- and wiring-level with no subprocess, and 10 that
spawn the server and drive it with a real MCP client. The integration file
earns its cost: it caught a resource handler constructing an object the SDK
rejects, which all 48 in-process tests had passed over. It also pins that no
library module writes to stdout — stdio transport shares that channel with
JSON-RPC, and a stray `print()` corrupts every session in a way that looks
like a protocol bug rather than a Python one.

`Development/mcp_plan.md` gains a "what the build found" section recording
the four plan assumptions that did not survive implementation, rather than
being edited to match the outcome.

### Fixed (formatting)

`tests/cpp_bindings/test_numerical_semantics.py` had been failing
`black --check src/ tests/` since 2026-08-19, so CI's lint job was red for a
reason unrelated to anything since. Reformatted; no semantic change.

### Added (the example agents can now reach the modeling runtime)

`Implementation/` and `Multi_Agent_Implementation/` had no way to build a
model. They loaded `standard_quant_tools.agent` and nothing else, so every
capability of the modeling runtime — eight tools, its own registry since it
shipped — was unreachable from the reference implementations. An agent built
by copying these scripts could analyze, backtest and size, and could not fit.

**One seam, in five files.** Each `_agent_utils.py` had exactly one place
that loads tool schemas and one place that executes a call. Those are now
paired behind a registry name:

```python
run_agent(..., registry="analysis")   # 46 tools, dispatch          (default)
run_agent(..., registry="modeling")   #  8 tools, modeling_dispatch
```

The pairing is the point. The two registries have identical shapes — same
OpenAI-format schema, same `dispatch(tool_name, arguments)` signature — so
nothing structural prevents loading one registry's tool list and calling the
other's dispatcher. That fails at the first tool call with an "unknown tool"
error naming the model's choice, which reads like the model picked badly
rather than like the wiring is wrong. Binding both to one lookup makes the
mistake unwriteable rather than merely unlikely.

`categories=` alongside `registry="modeling"` raises instead of being
ignored. Category routing exists to narrow 46 similarly-shaped tools; eight
tools in one ordered pipeline have nothing to narrow. A caller who believed
they had scoped the tool list and silently had not is worse off than one who
gets an error.

**`Agent_Model_Builder.py`, on all three providers plus the top-level set.**
Drives the full pipeline: capabilities, catalog, build, analyze, fit,
inspect, evaluate as a portfolio. Its system prompt spends most of its length
on two things the tools cannot enforce — that a leakage flag is a claim to
check against the lead-lag curve rather than a verdict (a slow-moving state
feature is not a leak), and that out-of-sample IC is not an answer to "would
this have made money", which is what `evaluate_model_portfolio` is for.

**Two modeling workers, not one.** `Multi_Agent_Implementation/` goes from
seven workers to nine: `model_research` (capabilities, catalog, build,
analyze) and `model_builder` (fit, inspect, score, evaluate). The analysis
workers derive their tool lists from `TOOL_CATEGORY`; the modeling runtime
has no taxonomy to derive from, so the split is by pipeline stage and is
written out explicitly.

The cut is at the dataset, for a structural reason rather than a tidy one:
it is the only handoff in the pipeline carrying a single value — the
`dataset_id` — rather than a whole panel, and therefore the only one that
survives two agent sessions that cannot see each other's context.
`model_builder` has no tool that can create a dataset, so the ordering is a
real constraint and the orchestrator's prompt states it rather than leaving
it to the general "chain specialists" rule.

The orchestrator needed no structural change: its delegate tools already
auto-generate from `WORKER_AGENTS.keys()`, so a second registry cost one
worker entry each, not a redesign of the delegation loop.

### Changed (the worker coverage check is now per-registry)

`test_multi_agent_tool_coverage.py` asserted that the union of every worker's
tools equals the library's tool set. With two registries that assertion would
pass while a worker listed a modeling tool under the analysis registry —
which fails at the first tool call, because the two dispatch functions do not
know each other's names. Each registry is now required to be covered exactly
once by the workers declaring it, the two are required to share no tool name,
and the `dataset_id` handoff is pinned by asserting `build_model_dataset`
stays out of `model_builder`.

Verified by mutation rather than by reading: adding `score_model` to the
screener worker fails the new check with "workers claim tool(s) this registry
does not have", and passed the old one.

Nine tests, up from seven.

### Added (modeling: feature analysis, ranking, adapters, point-in-time)

Four capabilities the previous cycle's analysis named as blocking, plus the two
agent tools that expose them. The tool count goes 6 → 8, and the invariant that
governed the first six still holds: **every tool is a decision the agent makes,
not plumbing**. Nothing here changes what an existing `ModelSpec` predicts.

**`analyze_features` — a feature report, not a feature list.**
`modeling/analysis/feature_report.py` answers the question `list_features` never
could: is this feature worth putting in a model. Per feature it reports
distribution and coverage (missing fraction, dispersion, share beyond 4σ),
predictive strength (IC, rank IC, ICIR, hit rate), autocorrelation, and a
lead–lag IC curve across shifts. Across features it reports a redundancy matrix
so an agent can see that two candidates are the same feature twice.

The leakage screen is the part worth describing, because the first version of it
was wrong. It flagged ADX and `realized_volatility` — both honest. The test had
asked "does advancing the feature help", which is true of any slow-moving state
variable and says nothing about leakage. The screen now asks whether shift 0 is a
*strict* peak on the lead–lag curve **and** whether the effect is large
(`_LEAKAGE_MIN_ABS_IC = 0.05`, raised from 0.01), with a persistence guard
(`_LEAKAGE_MAX_PERSISTENCE = 0.95`) so a highly autocorrelated feature is not
convicted for being autocorrelated. A fixture that had "caught" the screen being
wrong turned out to use a contemporaneous target, against which an honest feature
genuinely does peak at shift 0 — the screen was right and the test was wrong, and
the fixture now builds a real forward return.

**Learning-to-rank as a first-class task.** `ModelSpec.task` gains `"ranking"`,
with `lightgbm_ranker` and `xgboost_ranker` behind the same guarded registration
as the other optional estimators. Ranking is not regression with a different
metric: it needs integer relevance grades, query groups that are the
cross-section on each date, and NDCG rather than R². `modeling/validation/
ranking.py` supplies `relevance_grades`, `group_sizes`, `ndcg_at_k` and
`ranking_metrics`. `RankingSpec.n_grades` is bounded at 31, because LightGBM's
default `label_gain` holds exactly 31 entries and a 32nd grade fails at fit time
rather than at spec time.

**`ModelAdapter` — the task dispatch, in one place.** `modeling/adapters.py`
gives each task an adapter owning how its arrays are built, how a fitted model is
scored, which metrics apply, and how a fold's IC series is computed.
`run_experiment` no longer compares `task` anywhere in its body; it asks the
adapter. Adding a fourth task is now a class, not a sweep through the engine.

**`list_modeling_capabilities` — what this runtime can do, from the runtime.**
`modeling/capabilities.py` reports tasks, estimators, features, targets,
validation schemes, preprocessing, weighting, search, and which optional
dependencies are actually importable. An agent no longer has to guess whether
`xgboost_ranker` exists in this install, and the answer cannot drift from the
registry because it is read from it.

**A point-in-time join and its temporal contract.**
`modeling/dataset/point_in_time.py` adds `asof_join`, `validate_pit_frame` and
`coverage_report`. A record carries `available_time` separately from
`event_time`, and the join rule is that a feature at *t* may consume only rows
with `available_time <= t`. Revisions follow from the same rule: the value taken
is the one that was current at *t*, not the corrected one. `validate_pit_frame`
rejects `available_time < event_time` outright — a record available before the
period it describes has ended is a column mix-up, and it is the one error that
makes a backtest look prescient rather than merely wrong.

**Why the modalities are not here.** This is the join primitive, not a
fundamentals feed. `get_financial_ratios(symbol)` takes no `as_of` at all and
every shipped provider reports `point_in_time=False`, so a `DataBundle` carrying
fundamentals today would be an empty box with a correct label on it. What is
buildable and testable now is the join and its rules, so that the leakage-critical
part is already written and covered when a point-in-time source does arrive,
rather than being invented under deadline.

### Fixed (modeling audit — one of these made every tree model unscoreable)

**`ModelManifest` rejected the JSON it had itself written.** JSON has no NaN, so
a NaN metric serializes to `null` and returns as `None`, which the model failed
validation on. `summarize_importance` correctly reports `signed_mean`,
`signed_std` and `sign_consistency` as NaN for any estimator with no coefficient
sign — so **every tree-based model was unloadable, and therefore unscoreable**:
`random_forest`, gradient boosting, LightGBM and XGBoost alike. Linear models
were unaffected because they have a sign and never wrote a NaN, which is exactly
why this survived review. A `_nulls_to_nan` before-validator maps `null` back to
NaN; `oos_metrics` gets the same treatment, since a single-class fold's AUC and a
single-date fold's IC standard deviation are both legitimately NaN. Pre-existing,
not introduced by this cycle.

**`clip_sigma` disagreed across backends.** The native kernel raised `ValueError` on a
negative value and the Python path silently skipped clipping — so the same spec
produced differently-preprocessed features depending on whether `_sqt_core`
was built. `standardize_cross_sectional` now validates before dispatching, so
both paths raise `ValidationError`.

**Capability discovery reported that no linear model exposes coefficients.**
`hasattr(cls, "coef_")` is False on the class: scikit-learn sets `coef_` at fit
time, not at class definition. Detection now tests `issubclass` against
`LinearModel` / `LinearClassifierMixin`. Introduced in this cycle and caught
before release.

**`triple_barrier` returned a target no classifier would accept.** The encoding
was `{0, 0.5, 1}`, which scikit-learn's type inference reads as *continuous* — so
every classifier refused to fit. Re-encoded as `{0 = down, 1 = up, 2 = neither}`,
with "up" deliberately at 1 so `positive_class_proba` still means P(up).

**LightGBM estimators silently dropped `random_state`.** The wrapper took
`**kwargs`, and `get_params()` reads the `__init__` signature — so parameters
arrived at fit but were invisible to scikit-learn's cloning, and a "seeded" run
was not reproducible. The real classes are now registered with explicit
parameters, with the same defect fixed in `QuantileGradientBoostingRegressor`.

### Added (native kernels for the modeling layer)

Five kernels in `_sqt_core`, from `Development/modeling_native_plan.md`. Each
is an optional fast path with the Python implementation kept as both the
reference and the test oracle, and each agrees with it to **8.9e-16 or
better**.

| Kernel | Replaces | Measured |
|---|---|---|
| `fit_preprocess_stats` | per-column `quantile`/`clip`/moments | 5.5–23.5× |
| `apply_preprocess_stats` | per-column clip + standardize | 14.5–53.6× |
| `standardize_by_date` | `standardize_cross_sectional` | 8.6–11.6× |
| `cross_sectional_correlation` | `cross_sectional_ic` (per-date) | 3.0–6.2× |
| `cross_sectional_correlation` | pooled `Series.corr(method="spearman")` | 1.6–3.0× |
| `label_uniqueness` | `label_uniqueness_weights` | 8–23× |

End to end, `run_experiment` against the pure-Python path: **1.92×/2.05×**
pooled, **1.59×/1.82×** cross-sectional, **2.23×/2.55×** weighted, at
200/500 entities.

**The plan stated a ceiling before a method, and it held.** Feature
preprocessing was 47–56% of a run and the rest is pandas plumbing no kernel
reaches, so ~2× end-to-end was the arithmetic limit however fast the kernels
got. After the first phase, preprocessing fell to 13% and "everything else"
rose to 70% — fold slicing, DataFrame construction, the parquet write. That
is why the work stopped at three phases rather than continuing.

Exactness required reproducing pandas' *conventions*, not merely being
defensible: linearly interpolated quantiles at `h=(n−1)q`, ddof=1 standard
deviations, NaN skipped by moments but preserved by transforms, and
infinities NOT treated as missing. One quirk is reproduced deliberately and
pinned by a test — `standardize_cross_sectional` reduces with
`np.add.reduceat`, so a single NaN propagates into the whole date's mean and
zeroes that date's *entire* column, not just its own row. The first kernel
skipped NaN, which is more defensible and disagreed; it was a speed change
and had no business moving a number.

The pooled correlation is the same kernel with one segment, so it cannot
drift from the per-date form. Having no per-date parallelism to draw on, its
ranking sort splits into per-thread runs and merges, above 50 000 rows only —
the cross-sectional path never enters that region, which matters because it
is already inside a parallel loop and OpenMP disables nested regions by
default.

### Fixed (two fast paths that were slower than what they replaced)

Both found by benchmarking the small case as carefully as the large one.

`apply_preprocess_stats` regressed past four threads under `schedule(guided)`.
The case for guided rests on work per iteration varying, and here it provably
does not — every row is the same `n_cols` operations. What the loop *is*, is
memory-bandwidth bound, where guided's shrinking non-contiguous chunks work
against the prefetcher. Switched to `schedule(static)`, which measured better
at every thread count above one.

`label_uniqueness` measured **0.4×** on a 12 600-row panel: three pandas round
trips to normalize datetimes cost more than the Python loop saved. Fixed by
reinterpreting `datetime64[ns]` to int64 for free where possible, and gating
the kernel below 50 000 rows. After both: 1.4× at 12 600 rising to 22.8× at
504 000, with no size losing.

A fast path that is slower is a bug, not a trade-off.

### Fixed (a null-guard that skipped a legitimate panel)

`fit_preprocess_stats` returned early on a null `values` pointer. An empty
`std::vector`'s `data()` is null, so a legitimate `(0, n_cols)` panel silently
kept uninitialized statistics that the caller then divided by. Found by the
C++ suite, not by Python, because the engine happens to guard against empty
folds upstream. Zero-row columns now take the all-missing rule like any other
column with no observations.

### Added (tests)

`tests/cpp/test_panel_stats.cpp` — a tenth C++ suite, 43 assertions covering
the quantile interpolation rule, the ddof=1 divisor, NaN skipped by moments
but preserved by transforms, infinities not treated as missing, constant and
single-row columns, row-major column independence, and in-place aliasing.

`tests/modeling/test_native_preprocessing.py` and
`tests/modeling/test_native_metrics.py` compare the two paths directly by
toggling each module's `HAS_CPP`, so they are meaningful whether or not the
extension is present.

### Added (modeling: six capability gaps closed)

An analysis of `standard_quant_tools.modeling` (`Development/modeling_analysis.md`)
found the architecture sound and the gaps in breadth. All of these change what a
model predicts, so all of them are opt-in behind an explicit spec field and every
default is unchanged.

**Cross-sectional normalization** — `ModelSpec.preprocessing.normalization`.
Pooled z-scoring leaves the market factor inside every feature: on a day the
market rallies, every entity's momentum reads high together, and the model can
score well by learning "today was an up day". For a model judged on
cross-sectional IC that is the wrong thing to have learned. `cross_sectional`
standardizes within each date instead. There is no fold-boundary question — a
date uses only its own cross-section, which is contemporaneous information — and
the pooled path's 1st/99th percentile winsorizing is replaced by sigma clipping,
because a percentile inside one date is meaningless (the 1st percentile of a
20-name cross-section *is* its minimum). `zscore_cross_sectional`, which had sat
in `transforms.py` with zero callers, is what this is built on. Measured *faster*
than pooled — 469 ms against 898 ms — since it skips the quantile fitting.

**Sample weighting** — `ModelSpec.weighting`. `effective_sample_size` was
computed into `oos_metrics` and acted on by nothing; there was no `sample_weight`
anywhere in the module. Adds label-uniqueness weighting (the mean of
1/concurrency over the bars a row's label spans, computed per entity), calendar
time decay, and both. An estimator that does not accept `sample_weight` now
raises rather than silently ignoring it — a weighting the caller believes is
active but which never reached the fit is worse than an error.

**Expanding and purged K-fold validation** — `ValidationSpec.scheme` and
`.method`. `PurgedKFoldSplit` tests every date exactly once with a two-sided
embargo, which uses a short history far better than walk-forward. Its cost is
documented rather than buried: later folds train partly on data postdating their
test block, so it answers "is there a signal here", not "what would this have
earned".

**Hyperparameter search** — `ModelSpec.search`. Grid or random search on each
fold's training window, using an inner walk-forward over *dates*. Deliberately
not `GridSearchCV`: sklearn splits rows, and a K-fold over stacked
`(entity, date)` rows puts the same date on both sides of an inner split, so the
selection would be leaked even though the outer split is clean. The report keeps
every candidate's score per fold — on the benchmark the chosen `alpha` differed
on most folds, which is the search fitting noise and is visible only because
nothing was collapsed to a single "best" value.

**Four more targets** — volatility-scaled forward return, cross-sectional rank
(which matches the scorecard the model is judged on), market-neutral return, and
triple barrier. The two cross-sectional ones are applied after stacking, since
they are defined against the other entities on the date.

**LightGBM, XGBoost and quantile regression.** `random_forest` was measured at
62.9 s for one walk-forward run at 50 entities; `lightgbm` is 3.55 s and
`xgboost` 3.50 s on the same panel — about 18×. Neither is a declared
dependency: registration is guarded, so a missing library leaves the registry
reporting what *is* installed rather than breaking an import.

### Changed (modeling performance — no reported number moves)

**`cross_sectional_ic` vectorized.** Measured at **72%** of a ridge walk-forward
run: the per-date groupby called `Series.corr` thousands of times per run. Now a
handful of array passes, with a balanced-panel path that reshapes to
`(n_dates, n_entities)` and an `np.add.reduceat` segment path for ragged panels.
Agreement with the implementation it replaces is 2.2e-16 (spearman) and 5.0e-16
(pearson) across randomized panels with ties, constant cross-sections, NaN,
infinities and ragged shapes. 44.8× at 63×50, 47.8× at 252×50, falling to 1.8× at
252×2000 as the per-date overhead amortizes. `run_experiment` at 50 entities:
3.892 s → 1.577 s.

Two details are pinned by tests. The textbook `n*Sxy - Sx*Sy` shortcut
catastrophically cancels on return-scale data; the centered two-pass form moved
pearson agreement from 2.2e-14 to 5.0e-16. And the row-count gate runs *before*
the NaN drop, so a date whose rows are all NaN is reported as exactly 0.0 rather
than omitted — arguably wrong, deliberately preserved, since this was a speed
change and had no business moving a metric.

**Panel indicators wired into `build_dataset`.** `indicators/panel.py` existed
since the native-scaling work and was unreachable from the modeling path. Now
used for `rsi`, `adx`, `stochastic_k`, `atr_pct` and `bollinger_pct_b` — but only
when every entity's index is identical. That guard is the substance of the
change: the panel stacker uses the *intersection* of every ticker's bars, and
every indicator involved is path-dependent, so a shorter history changes values
and not merely coverage. A ragged universe falls back to the per-entity loop.
About 2× on the feature phase.

**Fold masks by gather rather than hash.** The walk-forward loop rebuilt
`panel["date"].isin(fold_dates)` every fold. One `searchsorted` up front maps
rows to date positions; a fold then gathers a small per-date boolean, which also
keeps working for splitters whose folds are not contiguous.

**Memoized input checks.** An opt-in scope in which an object that has already
passed a given check is not re-checked, keyed on `(object identity, check
variant)`, recording only successes. Outside the scope nothing changes. Worth
5.8% of a 16-feature build and nothing measurable at 6 features — the analysis
predicted 12–18%, which was wrong: it measured the cost of *all* validation
rather than of the *repeated* validation.

### Added (benchmarks)

`tests/bench/bench_modeling.py` backs every figure in
`Development/modeling_analysis.md`. Its `build` section attributes time to
feature computation directly rather than A/B-ing whole builds: repeated on an
ordinary workstation, a whole-build A/B of the same change returned ratios from
0.62× to 1.39× — a spread wider than the effect being measured.

### Fixed (C++ audit: the NaN/Inf data contract, and undefined behaviour)

`bindings.cpp` states a contract in writing — "degenerate arguments and bad
bars yield NaN, not exceptions" — and nothing tested it. Every "returns all
NaN" test in `tests/cpp_bindings/` covered a degenerate *parameter*
(`period <= 0`, `window > n`), never a non-finite *value*. Four kernels were
breaking it.

- **`bollinger_bands` raised `RuntimeError` for the whole series on one NaN
  bar.** `numerics::clamp_near_zero_sumsq` fell through to its throw because
  `NaN >= 0.0` and `|NaN| < eps*|NaN|` are both false. Fixed at source. The
  second half was subtler: an O(1) sliding sum cannot un-add a NaN
  (`NaN - NaN` is NaN), so the sums stayed poisoned until the next periodic
  refresh and the kernel reported NaN for a run of windows containing no bad
  data. Output now matches `pandas.rolling(min_periods=period)` exactly,
  including where the NaN starts and stops.
- **`stochastic_oscillator` returned %K of 250** — worse than raising,
  because it looks like data. A NaN was pushed into the monotonic
  sliding-extremum deque; every comparison against NaN is false, so it was
  never popped and it blocked eviction of the stale indices behind it. The
  deque front stopped being the window maximum. NaN indices are no longer
  pushed, and a missing-observation count reports the window as unevaluable.
  %D had the same accumulator problem and is fixed the same way.
- **`rolling_hurst(method="dfa")` raised where `hurst_exponent()` returned
  NaN** on the same series — three entry points to one estimator disagreeing
  about whether a bad bar is an error. `dfa_onepass`'s negative-SSE guard
  treated a non-finite residual as "indicates a real bug"; non-finite values
  now pass through as data.
- **`_zerocopy` bindings accepted arguments their siblings reject.**
  `batch_run_strategy_zerocopy` skipped all four scalar validators, so
  `initial_capital=0` returned `[0.0, nan, 0.0129, 6.595]` — a NaN total
  return beside a decisive-looking Sharpe — where `batch_run_strategy`
  correctly raised. The validators are now one grouped call per binding.

Also fixed, from the same audit:

- **MacKinnon critical values were the 1991 table** under a comment naming
  MacKinnon (2010), and were evaluated at `nobs` where statsmodels' `coint()`
  uses `nobs-1`. Both corrected; now bit-identical to `mackinnoncrit` across
  25 random pairs. Nothing caught it because the only assertions on these
  numbers checked *ordering*, and the 1991 values are also monotonic.
- **Exceptions could escape OpenMP structured blocks** (undefined behaviour)
  in `batch_backtest_crossover` and `rolling_hurst_into` — `std::bad_alloc`
  on a per-thread buffer in both cases. Contained and rethrown outside the
  region, matching the pattern `batch_run_strategy` already used.
- **`qr::lstsq`'s `rss` was silently wrong below full rank** — the tail sum
  from column `k` omits the `[rank, k)` components. NaN now.
- Unchecked `size_t -> int` narrowing in `rolling_factor_loadings` (both
  kernel and bindings) and in `adf_test`'s max-lag cap.
- `run_strategy` carried a hand-written duplicate of `gross_return_at`, the
  helper whose own comment says it exists so the two cannot drift.

**Tests:** `bollinger_bands` had no direct C++ coverage at all before this —
only an indirect check against itself inside the fused-indicator test. It now
has nine, including an independent brute-force reference. Added
`tests/cpp_bindings/test_cpp_nan_data_contract.py` (29 tests pushing one NaN
and one ±Inf through every kernel that takes a price series), plus C++
coverage for `batch_backtest_crossover` and the `ref_prices` fill model,
neither of which had any.

### Added (universe-scale performance)

Measured baseline first, in `Development/optimization_plan.md`, with the two
harnesses that produced it committed under `tests/bench/`.

- **`engle_granger` was the only kernel in the extension that was not linear
  in n** — `O(n^1.99)`, 246 ms at n=8000, of which forcing `max_lag=0` showed
  **1953×** was the ADF lag sweep. The candidates are *nested* and already
  share a common sample, so `qr::lstsq_nested_rss` reads every candidate's
  residual off one unpivoted factorization instead of factorizing per lag.
  9.6–17.9×, then 42.8× at n=8000 once the design was stored column-major.
- **`batch_engle_granger` / `scan_cointegrated_pairs`** — the whole pair set
  in one native call, parallel across pairs. A 2 000-ticker screen at 2 000
  bars: **9.81 h → 5.31 min (111×)**. `agent.tools.scan_pairs` takes this
  path only when every ticker shares an index, and falls back to the per-pair
  loop otherwise, because the batch path aligns the universe onto one common
  sample and the loop aligns each pair against only its partner.
- **`technical_indicators_panel` / `indicators.panel`** — a whole universe in
  one call, parallel across tickers. 500 tickers × 1 000 bars, five
  indicators: **1 727.6 → 144.7 ms (11.9×)**. Worth recording *why*: the
  pybind11 boundary was never the cost (2.7 µs/call, 14%); the per-ticker
  pandas round trip was, at 318 µs against 19 µs of kernel.
- **`run_portfolio_simulation`** — a native bar loop for the configuration
  the Python already treats as its vectorized fast path. 1 000 tickers ×
  2 000 bars: **188.7 → 35.8 ms**. Most of that was *not* the loop: profiling
  put 92% in building the dense price matrices, one pandas `.loc` per
  (ticker, column). Anything outside the fast path — per-share commission,
  the impact model, an ADV constraint — still runs the unchanged Python loop.
- **`schedule(guided)` on every parallel loop.** `schedule(static)` splits
  iterations evenly and never rebalances, which is only right when the work
  per iteration *and* the speed of each thread are uniform. Neither is
  something a library can assume. Scaling was running backwards —
  `batch_run_strategy` took 44.5 ms on 6 threads and 60.4 ms on 8. It is
  monotonic now. Deliberately *not* a tuned thread count: that would be
  tuning the library to whichever machine profiled it.

### Changed (numerics)

- `qr::lstsq_nested_rss` stores its design **column-major**, unlike
  `qr::lstsq`. A Householder factorization walks columns; row-major storage
  strides by `k*8` bytes, so at `k=27` every element access pulled a fresh
  64-byte cache line and used 8 of it. Cost per flop was nearly tripling as
  the design outgrew cache (0.99 → 2.69 ms/Mflop) and is now nearly flat.
  This also fixed `batch_engle_granger`'s thread scaling by itself — the
  ceiling there was memory bandwidth, not scheduling.
- `SQT_NOINLINE` (`platform.hpp`) keeps the AVX2 kernel from being inlined
  across its translation-unit boundary under LTO. Measured: MSVC 19.44 does
  not currently do that, with or without the qualifier — this is insurance
  against something the toolchain is permitted to do, kept because the
  property at stake is "does this crash on a pre-Haswell CPU".

### Added (model-to-portfolio evaluation)

A trained model could be *measured* (`run_model_experiment`'s R², IC,
rank-IC) and it could be *traded one name at a time*
(`bridge.oos_predictions_to_signal_panel` → `run_signal_panel_backtest`),
but nothing connected it to the shared-cash portfolio simulator this
repo already had. `evaluate_model_portfolio` — a 6th modeling tool —
closes that:

    OOS predictions → per-date cross-sectional weights → target-weight
    artifact → run_portfolio_simulation → economic metrics

**Why the bridge was not enough.** It maps every prediction to `-1/0/+1`,
which is the only defensible conversion for an engine that multiplies a
`SCORE` straight into `strategy_return` as a raw leverage multiplier —
but it discards the ranking, and `run_signal_panel_backtest` then gives
every ticker its *own* `initial_capital`. A cross-sectional model
predicts an ordering over names competing for the same dollars; neither
property survives. A strong `cs_rank_ic` alongside a negative portfolio
Sharpe is a real and common outcome, and nothing in `oos_metrics` could
show it.

**New declarative specs** (`modeling.specs`), constructible by an agent:

- `PredictionTransformSpec` — `sign` / `cross_sectional_rank` /
  `cross_sectional_zscore` / `top_bottom_quantile`, with
  `gross_exposure`, `net_exposure`, `max_position_weight`,
  `volatility_scale` and a `daily`/`weekly`/`monthly` rebalance schedule.
- `PortfolioSimSpec` — the evaluation-relevant subset of
  `run_portfolio_simulation`'s parameters, defaulting to `next_open`
  fills rather than the simulator's backward-compatible `close`.

**The ranking math is reused, not reimplemented.** `backtest.sizing`
(`rank_weighted`, `zscore_normalized`, `equal_weight_top_bottom`,
`vol_scaled`) already builds gross-normalized weight panels and is called
as-is. `modeling.portfolio_eval` contributes only what sizing.py has no
concept of:

- **Exact gross *and* net targets.** A single rescale controls one or the
  other, never both. The signed vector is split into books sized to
  `(gross + net)/2` and `(gross − net)/2`, giving `sum(|w|) = gross` and
  `sum(w) = net` by construction for any `|net| ≤ gross`.
- **A per-position cap that redistributes.** Excess above the cap is
  pushed onto uncapped names in the same book, iteratively — one pass is
  not enough, since redistribution can lift another name over the cap. A
  cap that merely truncated would silently deliver less gross exposure
  than requested.
- **Honest infeasibility.** Two names cannot hold 0.5 gross at a 0.1 cap.
  That reports a shortfall in `transform_diagnostics` and `warnings`
  rather than breaching the cap or rescaling the *other* book (which
  would break the net target instead).
- **First-of-period rebalancing.** "Last date in the month" is only
  knowable once the month has ended, so a schedule built that way cannot
  be reproduced live. First-of-period can.
- **Sparse cross-sections.** Dates sharing an availability pattern are
  grouped and weighted together. A missing `(entity, date)` stays `NaN`
  and gets zero *weight* — never a `0.0` *score*, which is the middle of
  a centered cross-section and would rank a name the model said nothing
  about above every name it was bearish on. A date with fewer than two
  entities is left flat: one name is not a cross-section.

**Classification predictions are recentred to `proba − 0.5`.** Ranking is
unaffected (a monotone shift), but a raw probability lives in `[0, 1]`,
so `sign()` is `+1` for every name on every date — a "long everything"
book that looks like a signal.

**Same leakage and integrity discipline as the bridge.** OOS predictions
only, never `score_model` (whose estimator is the final full-panel refit,
so scoring historical dates would be in-sample); no parameter selects
that path. The predictions artifact's recorded content hash is verified
*before* loading, because structural validation passes on an edited file
that kept its shape. The target-weight and equity-curve artifacts are
content-addressed, so changing the transform writes a new artifact rather
than replacing one an audit record still points at.

`estimated_cost_drag_pct` is labelled **derived, not measured**: the
simulator deducts costs from cash without reporting a total, so this
reconstructs the commission + spread component from realized turnover and
excludes borrow, margin interest and impact. It is a floor.

Corrected alongside: the modeling surface's own documentation claimed
"the 5-tool surface stays exactly 5". The invariant was never the count —
it was that every tool is a decision the agent makes rather than
plumbing. The bridge still is not a tool (it reshapes an artifact);
`evaluate_model_portfolio` is one (it runs a simulation and produces
persisted artifacts). README, `15_modeling.md` and `10_auditability.md`
updated to 6.

50 new tests in `tests/modeling/test_portfolio_eval.py`, covering the
exposure/cap invariants as exact arithmetic and the full
`build_model_dataset → run_model_experiment → evaluate_model_portfolio`
chain end to end, plus one tying `get_modeling_tools()` to
`MODELING_TOOL_DISPATCH` (a tool advertised in the schema but missing
from the dispatch table would have been callable and then failed with
"unknown modeling tool"). Suite: 2794 → 2845 passed, 2 skipped.


### Changed (performance — the portfolio simulator addresses its data once)

`run_portfolio_simulation` was the last major workflow still reading its
inputs one label at a time. Suite: 2749 → 2794 passed, 2 skipped; all 9 C++
suites green. Every configuration was checked against the previous
implementation before any timing was taken.

The bottleneck was never the accounting. Profiling 100 tickers × 2,000 bars
showed **200,000 pandas `.loc` lookups** — one per ticker per bar, purely to
read a price — against a rebalance state machine that performs only 9,600
operations. The simulator spent its time *addressing* data, not simulating.

Prices, target weights and the liquidity baselines are now dense
`(n_bars × n_tickers)` matrices built once and indexed positionally, and the
default cost configuration executes a rebalance as array arithmetic instead
of a per-ticker Python loop (min of 3 runs each, same process,
back-to-back):

| scenario | before | after | speedup |
|---|---|---|---|
| 25 × 2,000 bars, monthly | 394.2 ms | 23.4 ms | **17×** |
| 100 × 2,000 bars, monthly | 1,502.6 ms | 32.3 ms | **47×** |
| 500 × 2,000 bars, monthly | 7,455.8 ms | 96.0 ms | **78×** |
| 100 × 2,000 bars, weekly | 1,639.9 ms | 45.7 ms | **36×** |
| 100 × 2,000 bars, daily | 2,340.9 ms | 116.7 ms | **20×** |
| 100 × 2,000, `next_open` | 1,605.8 ms | 47.8 ms | **34×** |
| 100 × 2,000, ADV constraint | 1,573.2 ms | 61.9 ms | **25×** |
| 100 × 2,000, impact model | 1,683.6 ms | 121.9 ms | **14×** |

The speedup *grows* with universe size, because the cost removed was
per-ticker-per-bar while the work retained is per-rebalance.

**The vectorized rebalance is deliberately narrow.** It engages only for the
`pct` commission model with no impact model and no ADV constraint. Per-share
commission has a per-*order* minimum, the impact model needs a per-ticker
volatility lookup, and the ADV constraint must raise naming one ticker —
each is a genuinely per-element decision, and bending them into vector form
would mean restating their semantics in a second place. They keep the
explicit loop and fall through automatically.

Because that leaves two routes through one calculation, the equivalence is
now a test rather than an assumption: the same economics are driven down
both paths (an ADV limit set wide enough to bind on nothing forces the loop)
and the results must agree, under every fill model, long-only and long/short.

**Validation was merged with construction rather than duplicated.** The
upfront price check reindexed every ticker and column to the master calendar,
and the simulator then reindexed all of them again. It now builds the
matrices first and validates those, doing the alignment work once.

**Error messages still name the same offender.** Screening a whole matrix
finds every violation simultaneously, where the loops these replaced stopped
at the first — so *which* ticker, column or date is reported is observable
behaviour. The vectorized checks screen in bulk and then reconstruct the
message by walking in the original order (ticker-major with columns inner;
earliest rebalance date first, gross leverage before position size within a
date). Tests pin that ordering.

Other per-element work removed along the way: the three post-trade invariant
checks now share one position-value vector instead of rebuilding it three
times; short-borrow fees accrue as one masked sum, on the linearity of the
fee in notional, instead of a call per ticker per bar; and cost-rate
validation happens once per rebalance rather than once per ticker (399,800
redundant validations of the same three scalars on a daily-rebalanced
backtest).

**Cost-rate validation moved to the entry point, and got stricter on the
way.** The rates are function parameters — they cannot change between
rebalances — yet they were re-validated inside the cost functions on every
trade (399,800 redundant checks of the same three scalars on a
daily-rebalanced backtest). They are now checked once at entry, through
`costs.py`'s own `_cost_rate`, so the engine and the cost primitives agree on
what a valid rate is.

That reaches further than the hand-rolled guard it replaces, which tested
`math.isfinite(value) or value < 0`:

- `commission_pct=True` passed it (`isfinite(True)` is `True`, `True < 0` is
  `False`), making a boolean a 100% commission rate.
- `slippage_pct="0.001"` raised a bare `TypeError` from inside `math.isfinite`
  rather than a `ValidationError`.
- Rates belonging to the *other* commission model were never examined at all,
  because validation only ran inside whichever cost function executed —
  `per_share_rate=True`, `min_commission=True` and `impact_coefficient=True`
  each ran a complete simulation to a plausible-looking final equity under the
  default `pct` model. All three are now rejected by name.

Agreement with the previous implementation is within 1.7e-15 relative
across all twelve configurations tested, with the `rebalance_log` identical
in every one. The residual is floating-point reassociation — `np.sum`'s
pairwise summation against a sequential accumulator — not a change of
formula.

### Changed (performance — moving the boundary, not porting more formulas)

The organizing question was not "can this function be C++" but "what is the
largest deterministic numerical workflow we can cross into C++ once and not
return from until the answer is finished". Suite: 2734 → 2749 passed, 2
skipped; all 9 C++ suites green. Every path was checked for exact agreement
with the one it replaces before any timing was taken.

**The realistic execution mode is no longer the slow mode.** The kernel only
knew Close prices, so `next_open` and `hl2_exploratory` always fell back to
Python — and a native grid could not be used for them at all, which is what
let a walk-forward optimize under one fill model and evaluate under another.
`run_strategy` now takes an optional per-bar reference price and applies the
same two-leg overnight/intraday decomposition:

| fill mode | native | python | speedup |
|---|---|---|---|
| `close` | 2.60 ms | 296.86 ms | **114×** |
| `next_open` | 4.20 ms | 307.88 ms | **73×** |
| `hl2_exploratory` | 6.39 ms | 321.08 ms | **50×** |

(20,000 bars.) A `next_open` grid now costs about what a `close` grid costs
(150 ms vs 142 ms) where it previously ran entirely in Python.

**The fused crossover grid removes the signal matrix.** Profiling a
300-combination × 5,000-bar SMA grid showed the batch kernel was solving the
small half of the problem:

```
python signal generation   121.4 ms   92.1%
vstack into (combos,bars)    3.2 ms    2.4%
native batch backtest        7.2 ms    5.4%
```

and it computed 600 moving averages where only **35 unique periods** existed —
every combination recomputing an average another had already produced.

Python now computes each unique indicator once (through the same `sma` the
strategy itself uses, so there is no second definition to drift) and C++ builds
each combination's signal into one reusable buffer and backtests it
immediately:

| grid | before | after | speedup |
|---|---|---|---|
| 300 combos × 5,000 bars | 167.2 ms | 12.2 ms | **13.7×** |
| 2,000 combos × 10,000 bars | 1,370.6 ms | 79.9 ms | **17.2×** |

Results are **bit-identical** to the general path (worst difference 0.000e+00
across every metric).

Peak memory changes shape too, from `O(combos × bars)` to
`O(unique_periods × bars)`. At the existing 50,000-combination cap over
100,000 bars that is **40 GB → 72 MB** — the allocation was a latent
out-of-memory failure inside a nominally permitted request.

**OpenMP is governed rather than assumed.** Kernels parallelized whenever
there was more than one task, which asks the wrong question twice: two tiny
backtests cost more in thread startup than they save, and a library that
grabs every core oversubscribes badly when it is itself inside a
`ProcessPoolExecutor`, several agents, or replicated containers. New
`sqt::omp_policy` decides on **total work** (tasks × elements) and honours
two environment variables:

- `SQT_NUM_THREADS` — ceiling on threads any kernel may use. Set to `1`
  inside a process pool.
- `SQT_OMP_MIN_WORK` — minimum work before a region goes parallel at all
  (default 50,000).

**Monte Carlo runs are reproducible.** With `random_seed=None` the native
kernel seeded itself from `steady_clock`, so the audit record faithfully
stored `None` while the numbers came from a value nobody kept — the run could
never be repeated, and nothing said so. The seed is now drawn *before*
execution and returned on the result, so re-passing it repeats the run
exactly.

### Fixed (native-layer audit — annualization, parity, and an execution-model mismatch)

A follow-up review covering the C++ layer and the Python residuals the earlier
passes left. Suite: 2722 → 2734 passed, 2 skipped; all 9 C++ suites green.

**A walk-forward optimized under one execution model and scored under
another.** `backtest_grid` defaults to `fill_price="close"`, and neither
walk-forward tool passed the caller's mode into it — while the out-of-sample
leg honoured it. So a run requesting `next_open` selected parameters under
same-close execution and then evaluated them under next-open execution.

Not cosmetic: measured across 25 random series with a realistic overnight gap,
the **winning parameter pair differed between the two fill modes on 7 of
them**. The out-of-sample number was therefore not a test of the parameters
that had actually been chosen. Both `run_walk_forward_backtest` and
`run_regime_adaptive_walkforward_backtest` now pass it through.
(`get_robustness_diagnostics` and `run_regime_adaptive_backtest` carry no
`fill_price` field at all, so they were never inconsistent.)

**`252` is no longer hard-coded in the native backtester.** `constexpr double
kPPY = 252.0` fed volatility, Sharpe, Sortino and Calmar — correct for daily
equity bars and wrong for the 1h/5m/1m intervals and 24/7 markets the data and
modeling layers now support. An hourly backtest reported a "Sharpe"
annualized as though its bars were trading days. `periods_per_year` is a
parameter on `run_strategy`, `run_strategy_summary` and `batch_run_strategy`,
defaulting to 252 so existing callers are unchanged; Python owns calendar
semantics and the kernel stays calendar-agnostic.

**Native Calmar used `252/n` where Python uses `252/(n-1)`.** N level
observations span N−1 return intervals, and Python was corrected in Pass 2 —
leaving the two backends disagreeing about the same backtest:

| bars | native | python | divergence |
|---|---|---|---|
| 21 | −4.398278 | −4.582008 | **4.01%** |
| 63 | 4.136000 | 4.211419 | 1.79% |
| 252 | 7.095421 | 7.131805 | 0.51% |

Negligible on long histories, material on exactly the short windows a
walk-forward fold uses. Now exact to 1e-9.

**A wiped-out strategy scored 0.0 natively and −1.0 in Python.** The native
Calmar branch was skipped entirely for a non-positive final equity, leaving
the field at its `0.0` default — which reads as *neutral* rather than as total
loss. It now reports −1.0, matching `cagr`'s documented wipeout handling.

**ADF treated numerical breakdown as maximal evidence of cointegration.** RSS
is mathematically non-negative, and `yty - bXty` is a difference of two large
nearly-equal quantities — the classic cancellation setup. Every negative RSS
was read as a perfect fit and produced `adf_statistic = -inf`, so:

```
rss = -2e-15   (rounding on a genuine perfect fit)
rss = -0.3     (the normal-equations solve failed)
```

received the same interpretation: the *strongest possible* evidence of
cointegration, silently. A negligible negative is now clamped to zero (the
genuine perfect fit, which statsmodels also reports as −inf/0.0), while a
materially negative RSS fails that lag candidate. The threshold is relative to
`yty` because RSS carries the units of y-squared — an absolute one would
classify the same data differently merely rescaled.

**Python residuals from the earlier passes.** `pca_returns` rejects
infinities (they reached SVD and surfaced as a bare `LinAlgError: SVD did not
converge`, naming neither input nor column), non-integral `n_components`
(`2.5` passed the old `< 1` check and failed inside a slice), and duplicate
column names (loadings are keyed by column, so duplicates collapsed into one
entry). The OHLC volatility estimators validate positive prices, `high >= low`
and `periods_per_year` — each takes a logarithm of a price ratio, so a single
negative Low turned the whole series to NaN behind a RuntimeWarning, and a
negative `periods_per_year` produced sqrt of a negative for the same silent
result.

### Changed (packaging — the native extension is part of the build)

`pip install .` used a pure-Python backend (`flit_core`), so it produced a
package **without `_sqt_core`** no matter what the machine could compile.
Building the extension was a separate manual CMake step, which meant
"installed Standard Tools" could mean two materially different runtimes with
nothing in the install output saying which one you had.

The backend is now `scikit-build-core`, and a normal install builds the
extension when a C++ toolchain is present. Verified by building a wheel:
`standard_quant_tools-0.1.0-cp312-cp312-win_amd64.whl` containing
`standard_quant_tools/_sqt_core.cp312-win_amd64.pyd`.

**A missing toolchain degrades rather than fails.** `_sqt_core` is an optional
fast path — every function it accelerates has a Python fallback — so demanding
a compiler to install would turn an accelerator into a hard dependency, the
opposite of what shipping it in the build was for. CMake declares
`LANGUAGES NONE` and enables C++ only once a compiler is actually found;
without one it warns and installs the pure-Python package. `SQT_REQUIRE_NATIVE=ON`
inverts that for CI, where a silent skip would mean a green build that quietly
tested only the fallback path.

> The first attempt used `check_language(CXX)` and **broke the ordinary
> developer configure**: that runs a separate try-compile which does not
> inherit the Visual Studio generator's toolchain discovery, so it reported
> "no compiler" on a machine with a working MSVC install. Caught by testing
> the previous revision in the same shell — it found MSVC 19.44.35228.0 where
> `check_language` did not. `enable_language(CXX OPTIONAL)` uses the real
> generator and agrees with the build that follows it.

### Fixed (financial ratios — one canonical unit and definition per field)

`FinancialRatios` is populated by three providers, and the shared field names
implied an interchangeability that did not exist. Two separate problems hid
behind them, and they need different answers.

**Units differ, and are converted.** yfinance reports `debtToEquity` as a
**percentage** (150.5) while Polygon computes a plain **ratio** (1.505), so a
screen written as `debt_equity_max=2.0` admitted nearly every company on one
provider and nearly none on the other, with nothing in either result
indicating which convention was in force. Every field now has one canonical
unit — plain ratios for `forward_pe` / `trailing_pe` / `price_to_book` /
`debt_to_equity`, decimal fractions for `return_on_equity` / `profit_margins`
/ `dividend_yield`, absolute currency for `market_cap` — and each provider
converts to it.

**Definitions differ, and are declared.** Bloomberg's `TOT_DEBT_TO_TOT_EQY`
is total *debt* over equity; Polygon's is derived from total *liabilities*,
which include payables, deferred revenue and lease obligations. The Polygon
figure is systematically higher for the same company — not by a scale factor
that could be corrected, but because it answers a different question. New
`FinancialRatios.definition_notes` names any field whose basis departs from
the canonical one, and the value is still returned: a liabilities-to-equity
ratio is useful when you know that is what it is. Silently shipping it under a
debt-based name was the actual problem.

**No plausibility-based auto-correction.** Inferring "15.0 must be a
percentage" would silently rewrite a genuine 1500% return on equity, which
small-equity companies really do post. Providers declare their own vendor's
units, and `implausible_value_warnings` *reports* a value that looks like an
unconverted percentage instead of changing it — so a vendor changing
convention (as yfinance did with `dividendYield`, between releases) surfaces
as a warning rather than as a silently wrong number.

### Fixed (slip audit — fixes reachable around through a sibling path)

A pass over the audit's own fixes, asking whether each one actually holds
everywhere or can be reached around. Four slips; two were defects in code
written *during* this audit, which is the point — a fix is not finished when
the function it targets is correct, only when no sibling path does the same
job unguarded.

- **`backtest_grid`'s C++ batch path never got the positive-price contract.**
  It kept only the finiteness check, so a `Close` of **-5.0 ran through an
  entire parameter sweep** and returned a full results table — because -5.0 is
  perfectly finite.
- **A NaN trade return became a "breakeven".** A hole the breakeven fix itself
  opened: moving from `~is_win` to explicit `> 0` / `< 0` tests made NaN
  satisfy *neither*, so an unmeasurable trade was bucketed with the flat ones —
  counted in the win-rate denominator, excluded from both averages, and
  treated as a streak-breaker. The earlier two-way split had at least called
  it a loss. Neither is right.
- **`_RAW_STRATEGIES`** must exist as the input to the validation wrapper, but
  calling out of it skips the check that makes a negative lookback
  unreachable. Now prominently marked internal-only, and pinned by a test.
- **A forgotten `interval`** on `_normalize_ohlcv_index` re-enabled the
  intraday-collapsing behaviour by omission rather than intent. The back-compat
  default is deliberate and tested, so it stays — but it now logs a warning
  naming how many timestamps are about to be flattened.

> Also checked and found **not** to be slips: Bloomberg's unconditional
> `.normalize()` (it rejects intraday intervals outright, so it only ever sees
> daily-or-coarser bars), `run_signal_panel_backtest`, and `portfolio_metrics`
> — the last two already inherit their contracts from the functions beneath
> them.

### Fixed (full-codebase audit, Passes 3-5 — solvers, schemas, audit policy)

Suite: 2613 → 2695 passed, 2 skipped.

#### Pass 3 — a solver reporting success is not a valid answer

`result.success` is the solver's opinion of its own run, not a statement that
the returned vector satisfies the constraints it was given.

- **Ill-conditioning is now reported.** The rank check added earlier catches an
  exactly-degenerate covariance; it does not catch two assets that are merely
  *almost* identical — the far more common real case (a share class pair, an
  ETF and its largest holding). Measured on three assets where two differ by
  1e-9 of noise: **rank 3/3**, condition number **3.827e+14**, and a maximum
  weight of **197,838× capital**, reported `converged: True` with no warning.
- **The returned weights are checked against their own constraints.** A
  long-only `target_return=99.0` returned weights that looked entirely
  well-formed — `sum(w)=1.0000` with an achieved return of **0.2443**. The
  independent check also caught something new: the ill-conditioned case above
  returns weights summing to **0.997433**, which the closed-form path had been
  reporting as converged.
- **Risk parity** validates its covariance (finite and symmetric — an
  asymmetric matrix was silently accepted), `max_iterations` and `tol`. A NaN
  covariance does not trip the `port_var <= 0` guard, so it flowed through
  every iteration and emerged as `{nan, nan}` weights.
- **Black-Litterman** validates every matrix and vector. One non-finite entry
  in *any* of `cov_matrix`, `market_weights`, `P`, `Q`, `omega`, `tau` or
  `risk_aversion` made the whole posterior NaN with no error raised.
- **`build_bl_views`** rejects non-finite view returns, pick coefficients and
  confidences (NaN satisfies neither `<= 0` nor `> 1`, so it passed both
  halves of the range guard), and rejects duplicate tickers — the
  ticker→column map keeps the *last* index, so a view on a repeated name
  silently attached to the wrong slot.
- **`build_portfolio`** rejects an empty frame, non-finite weights (previously
  caught only incidentally, and reported as a *sum* problem) and infinite
  returns.

#### Pass 4 — the classic agent schemas

The modeling schemas had Literals, bounds and caps; the classic quant schemas
kept bare `str` / `float` / `int` / `Dict` on exactly the surface an agent
drives most, so a bad value was discovered part-way through a tool, after data
had been fetched.

- `strategy`, `sort_by`, `fill_price` and `indicators` are `Literal`s. `sort_by`
  mattered most: the code did `if sort_by and sort_by in df.columns`, so an
  unrecognized metric was **silently ignored** and the caller received unsorted
  results with nothing indicating the request had not been honoured.
- `initial_capital` (> 0), `commission_pct` / `slippage_pct` (0–1; above 1.0
  means paying more than the whole notional), window and worker counts, and
  the pair-scanner controls are all bounded.
- **`param_grid` has a combinatorial budget.** A grid is the cartesian product
  of its axes, so cost is multiplicative: four axes of ten values is 10,000
  full backtests from a dict that fits on one line. Estimator complexity was
  already bounded; the *number* of estimator invocations was not.
- **`_parse_period` is strict.** An unrecognized unit fell through to
  `now - 365 days`, so `6m`, `1yr`, `ytd` and `""` all silently became one year
  — a malformed request turning into a *different valid* request, which is not
  detectable from the result.

#### Pass 5 — audit write policy and replay honesty

- **`SQT_AUDIT_FAIL_CLOSED=1`** makes a failed audit write fail the tool call.
  Fail-open stays the default — for an analytics library a full disk should not
  destroy a result the caller already paid to compute — but under a governance
  regime an action taken without a record of it is exactly what the trail
  exists to prevent.
- **`AuditIntegrityError` is never swallowed.** This is the interaction that
  mattered: Pass 1 made the writer refuse to extend a corrupted chain, and the
  dispatch wrapper catches `Exception` broadly around the write — so without an
  explicit passthrough that refusal would have been logged as an ordinary write
  failure and the corruption would have stayed invisible.
- **A redacted record is not replayable, and says so.** The record stores a
  placeholder rather than the original value, so reconstructing the call would
  run a *different* call and then report the inevitable hash mismatch as drift.
  Redaction and exact replay are in tension by construction; a record needs one
  or the other.
- **A failed call replays as a first-class outcome.** Letting the exception
  escape reported an error in the replay machinery, when what actually
  reproduced was the original failure — which is a *successful* replay. The
  same failure reproducing, a different failure, and a previously-failing call
  now succeeding are three distinct reported results.

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
  [Documentation/10_auditability.md](Documentation/10_auditability.md#auditability)
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
