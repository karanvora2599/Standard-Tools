# Native acceleration for the modeling layer

A plan grounded in where the modeling runtime spends its time **now**, after
the Python-level pass. The bottleneck moved, and it moved somewhere a C++
kernel can actually reach — but not as far as the headline numbers suggest,
and this document says how far before it says how.

**Status: phases 1-3 are implemented and merged.** Section 6 records what
each was worth, including the two things the plan had wrong. Reproduce every
figure with `python tests/bench/bench_modeling.py`.

## 1. The bottleneck moved

`cross_sectional_ic` was 72% of a ridge walk-forward run. Vectorizing it
worked, and the consequence is that it is no longer the problem:

| Component | Share of a run, 50 entities | Share, 200 entities |
|---|---:|---:|
| `_preprocess` (fit + apply) | **55.9%** | **46.9%** |
| everything else | 37.9% | 46.9% |
| `cross_sectional_ic` | 3.4% | 4.3% |
| `estimator.fit` | 2.8% | 2.0% |

Measured by wall clock with the profiler off, since cProfile inflates
exactly the numpy-heavy code being judged.

### Where preprocessing's time actually goes

Per fold, one 252-date training window, 8 feature columns:

| Operation | 252×50 | 252×500 | 252×2000 | C++? |
|---|---:|---:|---:|---|
| `quantile` ×2 (the winsorize bounds) | 9.3 ms | 26.5 ms | **166.5 ms** | yes |
| `clip` | 4.7 ms | 15.0 ms | **60.2 ms** | yes |
| `panel[mask]` fold slicing | 0.5 ms | 4.6 ms | 16.0 ms | no |
| `df[features]` | 0.5 ms | 2.1 ms | 9.1 ms | no |
| `.to_numpy()` | 0.5 ms | 2.2 ms | 6.7 ms | no |

At universe scale the C++-able work is **227 ms against 32 ms** of pandas
plumbing — seven to one. That ratio is the whole case for this work.

### The kernels, measured in isolation

| Kernel | 252×2000 (504k rows) | 1000×2000 (2M rows) | µs/row |
|---|---:|---:|---:|
| `label_uniqueness_weights` | 2,162 ms | **11,370 ms** | 5.7 |
| `fit_preprocessing` | 742 ms | **3,117 ms** | 1.6 |
| `apply_cross_sectional_target` (rank) | 482 ms | 2,824 ms | 1.4 |
| `standardize_cross_sectional` | 327 ms | 1,380 ms | 0.7 |
| `cross_sectional_ic` spearman | 322 ms | 1,373 ms | 0.7 |
| `apply_preprocessing` | 226 ms | 1,001 ms | 0.5 |
| `cross_sectional_ic` pearson | 102 ms | 266 ms | 0.13 |

`label_uniqueness_weights` is the worst per row by a factor of three, and it
is the only one with a Python loop left in it — one iteration per entity.

## 2. What a C++ kernel can and cannot do here

**Stating the ceiling first, because the arithmetic is unforgiving.**
Preprocessing is ~50% of a run. Making it *infinitely* fast caps the
end-to-end speedup at **2×**. Making it 10× faster gives
`1 / (0.5 + 0.5/10)` = **1.8×**. Any claim above that for the whole run is
wrong before it is measured.

The reason is visible in the profile: it is **flat**. The top entries are
`isinstance`, `Series.__init__`, `__finalize__` — pandas object-model
overhead spread across thousands of small calls. There is no second hot loop
waiting to be found, and no kernel removes a Series constructor.

**Reachable by C++** — contiguous float64, no Python objects in the loop:

- per-column quantile, clip, mean, std (`fit`/`apply_preprocessing`)
- per-date reductions and ranking (`cross_sectional_ic`,
  `standardize_cross_sectional`, the rank target)
- per-entity label concurrency (`label_uniqueness_weights`)

**Not reachable** — and no amount of C++ changes it:

- `panel[mask]` fold slicing, `df[features]`, `.to_numpy()` (pandas)
- the parquet write of OOS predictions
- `estimator.fit` (sklearn's problem; already addressed by adding LightGBM)
- dataset assembly, `stack_long`, alignment

So: **target the 50%, expect ~1.8× end-to-end at universe scale, and expect
the kernels themselves to be 5–15×.** The kernel multiple is the honest
headline; the end-to-end number is the one that matters to a user.

## 3. Phases

Ordered so each is independently shippable and independently measurable.

### Phase 1 — `panel_preprocess`: fused fit and apply

The single largest item. `fit_preprocessing` currently runs, per column:
`quantile(0.01)`, `quantile(0.99)`, `clip`, `mean`, `std` — five passes and
two full partitions, each through pandas, each allocating.

Kernel design:

- **fit**: one strided copy of the column into a thread-local buffer, two
  `nth_element` calls for the quantile positions, then one fused pass for
  clip + mean + sum-of-squares. Parallel over **columns**.
- **apply**: a single pass over the row-major panel, inner loop over
  columns, applying `clip → (x - mean) / std` from a small parameter array
  that stays in L1. No temporaries at all. Parallel over **row blocks**.

Two exactness obligations, both of which are the actual work:

- `pandas.Series.quantile` uses **linear interpolation**: for sorted `x` and
  `h = (n-1)·q`, the result is `x[⌊h⌋] + (h-⌊h⌋)·(x[⌊h⌋+1] - x[⌊h⌋])`. The
  kernel must reproduce that, not `nth_element` alone.
- `std` is **ddof=1** in pandas and ddof=0 in numpy. `fit_preprocessing`
  uses `Series.std()`, so ddof=1.

Expected: 5–10× on the preprocessing phase; ~1.5–1.8× end-to-end.

### Phase 2 — `cross_sectional_panel`: per-date statistics

Three Python functions share one shape — segment rows by date, do something
within each segment, write back — so they share one kernel family:

| Python | Kernel op |
|---|---|
| `cross_sectional_ic(method="pearson")` | per-date correlation |
| `cross_sectional_ic(method="spearman")` | per-date rank + correlation |
| `standardize_cross_sectional` | per-date centre, scale, clip |
| rank target (`apply_cross_sectional_target`) | per-date average rank |

The win is not the sort. Measured, the sort is only **24%** of the spearman
IC; the other 76% is the rank-assignment machinery — tie-run detection,
`bincount`, `put_along_axis` — which allocates roughly six full-panel
temporaries. A fused per-date kernel does the rank and the correlation in
one pass over a block that fits in cache, and allocates nothing.

Parallel over **dates**, which is embarrassingly parallel and gives every
thread identical work when the panel is balanced.

Expected: 5–15× on spearman, 2–4× on pearson (already near memory
bandwidth), 4–8× on standardize.

### Phase 3 — `label_uniqueness`

Worst per-row cost in the module and the only remaining Python loop. Per
entity: build a concurrency difference array, prefix-sum it, then average
`1/concurrency` over each label's span. All integer and float array work,
trivially parallel over entities.

Expected: 20–50×. It is 5.7 µs/row today, which is two orders of magnitude
off what the arithmetic costs.

### Phase 4 — wiring, fallbacks, tests, benchmarks

Every kernel is an **optional fast path**, matching the rest of the package:
the native module may be absent, and the Python implementation stays as the
reference. That is also the test oracle — each kernel is asserted equal to
the Python function it replaces, not to a hand-computed constant.

## 4. Risks, and how each is handled

**Exactness is the whole game.** These functions feed reported metrics and
model inputs. Every kernel gets a test asserting agreement with the Python
path to floating-point tolerance across randomized panels including ties,
constant cross-sections, NaN, infinities and ragged shapes — the same bar
the IC vectorization was held to.

**Layout beats scheduling.** The lesson from the native-scaling pass was
that a column-major layout change did more for parallel scaling than the
OpenMP scheduling fix, and it was not in that plan at all. So: per-column
work copies to a contiguous thread-local buffer rather than striding, and
per-date work exploits that a date's rows are contiguous in a date-sorted
panel. Layout is decided per kernel, deliberately, and measured.

**Parallelism must be machine-agnostic.** `schedule(guided)` and
`omp_policy::worth_parallel` throughout, per the existing convention. The
verification bar is the established one: *adding a thread never makes a
kernel slower*. No thread-count tuned to this box.

**NaN semantics must match, not merely be sensible.** `Series.quantile`
skips NaN; `Series.std` skips NaN with ddof=1; `Series.corr` drops NaN
pairwise. Each kernel matches its Python counterpart's rule, and the tests
cover it.

**A fast path that is slower is a bug.** Every kernel is benchmarked against
the Python path it replaces, at 50 / 500 / 2000 entities, and any size where
it loses is either fixed or guarded with a threshold.

## 5. What this plan deliberately does not do

- **Port the estimator.** sklearn's fit is 2–3% of a run, and the answer to
  a slow estimator was adding LightGBM, not rewriting one.
- **Replace pandas in the fold loop.** The slicing is 32 ms against 227 ms
  of arithmetic at 2000 entities. Rewriting the DataFrame handling would be
  a large change for the smaller half.
- **Touch `build_dataset`'s assembly.** Its profile is Series construction,
  which is the same flat-overhead problem and not a kernel.

Each of those is a real cost. None is the biggest one, and the point of
measuring first is to spend the effort where the measurement points.

## 6. What the phases were actually worth

| Kernel | Measured | vs. predicted |
|---|---|---|
| `fit_preprocess_stats` | **5.5-23.5x** | 5-10x — at the top of it |
| `apply_preprocess_stats` | **14.5-53.6x** | 5-10x — well above |
| `standardize_by_date` | **8.6-11.6x** | 4-8x — above |
| cross-sectional IC, spearman | **4.9-6.2x** | 5-15x — bottom of it |
| cross-sectional IC, pearson | **3.0-4.2x** | 2-4x — as predicted |
| pooled rank IC | **1.6-3.0x** | *not in the plan* |
| `label_uniqueness` | **8-23x** | 20-50x — below it |

End to end, `run_experiment` against the pure-Python path:

| Universe | Pooled (default) | Cross-sectional | + uniqueness weights |
|---|---:|---:|---:|
| 200 entities, 293,600 rows | 1.92x | 1.59x | **2.23x** |
| 500 entities, 734,000 rows | 2.05x | 1.82x | **2.55x** |

Every kernel agrees with the Python it replaces to 8.9e-16 or better.

### The plan got two things wrong

**It missed the largest metric cost entirely.** The pooled rank IC was not on
the list, and re-measuring after phase 1 showed it is 41% of
`regression_metrics` at 25,000 rows and 51% at 250,000 — larger than the
per-date IC the plan did name. It sorts the whole test fold through scipy on
every fold. The lesson is the one phase 1 already demonstrated: the profile
after a change is not the profile before it, and the discipline is to
re-measure between phases rather than work down a list written once.

**It expected 20-50x from `label_uniqueness` and got 8-23x.** The estimate
came from the per-row cost, which was two orders of magnitude off what the
arithmetic should take. That was true, but the kernel still has to convert
timestamps, bucket by entity, and normalize — and the first version was
actually *slower* than Python on a 12,600-row panel, because the argument
conversion cost more than the loop saved. Fixed by taking the cheap
reinterpret path when the input is already `datetime64[ns]`, and by gating
the kernel below 50,000 rows. A fast path that is slower is a bug.

### What the ceiling arithmetic got right

Section 2 predicted ~2x end to end and explained why: preprocessing was ~50%
of a run, and the rest is pandas plumbing no kernel reaches. Measured, the
pooled default lands at 1.92-2.05x. After phase 1 the attribution shifted
exactly as the ceiling implied — preprocessing fell from 47% to 13% of a run,
and "everything else" rose to **70%**. That 70% is fold slicing, DataFrame
construction and the parquet write, and it is now what bounds the number.

Stating the ceiling before the method was worth more than any individual
kernel: it is why this stopped at three phases instead of chasing the
remaining 70% with tools that cannot reach it.

### Where the remaining time goes

At 200 entities, after all three phases: `_predict_fold` 34%, `_preprocess`
13%, `DataFrame.__getitem__` 13%, the final refit 8%, `estimator.fit` 6%,
the parquet write 5%. Nothing left in that list is a numeric loop.
