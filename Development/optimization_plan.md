# C++ Optimization and Performance Plan

**Date:** 2026-08-21
**Scope:** `_sqt_core` and the Python layers that call it, with an explicit target of a **2,000-ticker universe**.
**Status:** phases 1-3 shipped, phase 4-5 partly. Per-item status is marked inline;
§7 has the summary. Where a measurement contradicted this document's own prediction, the
prediction is kept alongside the result rather than quietly replaced.

This supersedes the forward-looking parts of `performance_insights.md`, which documents
what was ported and what it gained. That document's per-kernel numbers are still broadly
valid; several of its *descriptions* are now stale (the incremental-Cholesky
`rolling_factor_loadings` it describes was replaced by per-window QR, deliberately trading
15–60× of speed for a correct rank policy).

Every number below is **measured on this machine** (MSVC 19.44, `/O2 /arch:AVX2`, LTO,
OpenMP 2.0, i7-13620H — 6 P-cores + 4 E-cores, 10 physical / 16 logical) unless explicitly
labelled an extrapolation. Reproduce with the harnesses in "Verification" at the end.

---

## 1. Executive summary

The native kernels are individually fast. The performance problem is **not** inside them —
it is in three places the porting effort never reached:

| | Where the time is | Measured | Fix |
|---|---|---|---|
| **A** | ~~ADF lag selection is `O(T · L³)`~~ ✅ **shipped** | was **O(n^1.99)**, 246 ms at n=8000 → now **13.8 ms**, 9.6–17.9× | One nested QR replaced `L` independent ones |
| **B** | Universe work loops in **Python**, one call per ticker or per pair | 2,000-ticker pair scan: **9.8 hr serial / 37 min on 16 cores** | Batch entry points that cross the boundary once |
| **C** | Parallel scaling is non-monotonic | `batch_run_strategy` is *slower* on 8 threads (60.4 ms) than on 6 (44.5 ms) | `schedule(guided)` — `static` never rebalances, on any machine |

Fixing A and B together takes the flagship 2,000-ticker cointegration scan from
**9.8 hours to 5.31 minutes** — measured, 111×. That single item is worth more than every
remaining kernel micro-optimization in this document combined.

The honest counterpoint, stated up front so it is not buried: **the per-bar indicator
kernels are done.** RSI is 1.66 ms over 200,000 bars. There is no meaningful headroom
there, and `performance_insights.md` already measured most of them at ~1× against warm
numba. Do not spend time on them.

---

## 2. Measured baseline

### 2.1 Per-kernel, serial (`SQT_NUM_THREADS=1`)

Raw binding time, min of 7. `n^x` is the empirical exponent between the two largest sizes.

| Group | Kernel | small | mid | large | scaling |
|---|---|---:|---:|---:|---|
| indicator | `rsi(14)` | 0.011 ms @2k | 0.105 @20k | 1.656 @200k | n^1.20 |
| indicator | `wilder_atr(14)` | 0.041 | 0.414 | 3.824 | n^0.97 |
| indicator | `adx(14)` | 0.061 | 0.528 | 6.286 | n^1.08 |
| indicator | `parabolic_sar` | 0.006 | 0.110 | 2.020 | n^1.26 |
| indicator | `bollinger(20)` | 0.030 | 0.281 | 3.942 | n^1.15 |
| indicator | `stochastic(14,3)` | 0.059 | 0.693 | 7.946 | n^1.06 |
| indicator | `technical_indicators` (all 5) | 0.205 | 2.812 | 28.993 | n^1.01 |
| regression | `rolling_beta(w=60)` | 0.011 | 0.101 | 1.268 | n^1.10 |
| regression | `rolling_factor_loadings(w=60,k=3)` | 3.091 @2k | 7.346 @5k | **29.028** @20k | n^0.99 |
| regression | `rolling_factor_loadings(w=60,k=10)` | 13.037 | 37.235 | **128.491** | n^0.89 |
| regression | `rolling_factor_loadings(w=252,k=3)` | 9.723 | 26.928 | **120.417** | n^1.08 |
| cointegration | **`engle_granger`** | 1.100 @500 | 17.213 @2k | **270.230** @8k | **n^1.99** |
| cointegration | `ols2` | 0.005 | 0.034 | 0.738 | n^1.34 |
| cointegration | `kalman_2state` | 0.028 | 0.626 | 4.585 | n^0.86 |
| hurst | `hurst_dfa` | 0.036 @1k | 0.162 @5k | 0.633 @20k | n^0.98 |
| hurst | `rolling_hurst(w=200)` | 9.358 @2k | 28.225 @5k | 113.793 @20k | n^1.01 |
| backtest | `run_strategy` | 0.031 | 0.291 | 3.825 | n^1.12 |
| backtest | `batch_run_strategy(×5000)` | 44.537 @500 | 172.323 @2k | — | n^0.98 |
| backtest | `batch_backtest_crossover(×2450)` | 10.758 @500 | 48.013 @2k | — | n^1.08 |
| montecarlo | `simulate_forward_paths(h=252)` | 1.924 @1k | 12.570 @10k | 65.217 @50k | n^1.02 |
| garch | `garch11_neg_loglik_grad` | 0.015 | 0.137 | 1.557 | n^1.05 |

**Read the `engle_granger` row again.** Everything else is linear. That one is quadratic.

### 2.2 Thread scaling — and why 16 cores are not 16 cores

The CPU is an **i7-13620H: 6 P-cores (12 threads via SMT) + 4 E-cores (4 threads)**,
10 physical / 16 logical — a *hybrid* part, where E-cores run roughly half the throughput
of a P-core.

That matters for reading the absolute numbers, and it is **not** what the fix in §5.1 is
tuned to. A hybrid CPU is simply an environment that makes an existing load-imbalance bug
easy to see; the same bug shows up under an SMT pairing, a cgroup CPU quota, a busy shared
machine, or merely uneven work per iteration. Treat the table below as evidence that
`schedule(static)` does not rebalance — a fact about the scheduling clause, true on every
machine — rather than as a tuning target for this one.

A realistic aggregate ceiling is therefore about
`6×1.0 (P) + 6×0.25 (SMT siblings) + 4×0.55 (E) ≈ 9.7×`, **not 16×**.

| Threads | `rolling_hurst` | `batch_run_strategy` |
|---:|---:|---:|
| 1 | 111.2 ms | 167.4 ms |
| 2 | 62.8 | 92.7 |
| 4 | 42.0 | **79.0** ← barely better than 2 |
| 6 | 36.3 | 44.5 |
| 8 | 34.3 | **60.4** ← *worse than 6* |
| 10 | **34.8** ← *worse than 8* | 40.2 |
| 12 | **26.2 (best, 4.2×)** | 33.8 |
| 14 | **29.5** ← *worse than 12* | 30.9 |
| 16 | 28.3 (3.9×) | 29.6 (5.7×) |

Two measured facts, both actionable:

1. **Both kernels are non-monotonic.** `batch_run_strategy` at 8 threads (60.4 ms) is
   *slower* than at 6 (44.5 ms). `rolling_hurst` at 14 is slower than at 12. Adding a
   thread makes things worse at several points — the signature of `schedule(static)`
   handing an equal chunk count to unequal cores, so the whole region waits on whichever
   chunk landed on an E-core.
2. **`rolling_hurst` is fastest at 12 threads, not 16** — and 12 is exactly the P-core
   thread count. Using all 16 logical processors is a 8% loss on that kernel.

Against the ~9.7× realistic ceiling, the best measured results are 4.2× and 5.7×, i.e.
**43–58% efficiency**. The remaining headroom is roughly **1.7–2.3×**, not the 3× a naive
16-core comparison would suggest.

### 2.3 Universe scale

**Pairwise cointegration scan** (`itertools.combinations`, one `cointegration_test` per pair):

| n_bars | per pair (raw) | per pair (wrapper) | 500 tickers (124,750 pairs) | 2,000 tickers (1,999,000 pairs) |
|---:|---:|---:|---|---|
| 500 | 1.08 ms | 1.85 ms (+71%) | 3.85 min / 14.4 s @16 | 61.7 min / 3.85 min @16 |
| 1000 | 5.33 ms | 6.09 ms (+14%) | 12.7 min / 47.4 s @16 | 3.38 hr / 12.7 min @16 |
| 2000 | 18.45 ms | 17.67 ms | 36.7 min / 2.30 min @16 | **9.81 hr / 36.8 min @16** |

(Per-pair cost is measured; the universe totals are that cost × pair count. The "@16"
column divides by 16, which §2.2 shows is optimistic by roughly **3×** — measured scaling
tops out at 4.2–5.7×, against a realistic ceiling of ~9.7× on this hybrid part. Read those
columns as a lower bound on the time, not a target.)

**Portfolio simulation** (`run_portfolio_simulation`, 504 bars, 101 rebalances):

| tickers | total | per bar |
|---:|---:|---:|
| 50 | 12.75 ms | 25.3 µs |
| 200 | 30.36 ms | 60.2 µs |
| 500 | 62.82 ms | 124.6 µs |

Linear in tickers above a ~15 µs/bar fixed Python cost. Extrapolating to 2,000 tickers:
**~450 µs/bar ≈ 0.9 s** for a 2,000-bar backtest, per backtest. A 50-fold walk-forward is
~45 s; a 100-point parameter sweep over that is over an hour.

**Predictions → weights** (`transform_predictions_to_weights`, 504 dates):

| entities | `cross_sectional_rank` | `top_bottom_quantile` (q=0.2) |
|---:|---:|---:|
| 100 | 92.3 ms (183 µs/date) | 1046.7 ms (2077 µs/date) |
| 2000 | 783.8 ms (1555 µs/date) | **2445.3 ms (4852 µs/date)** |

**A caveat specific to this table.** Repeated runs of this measurement varied by up to
**3×** (an earlier run gave 611 and 1506 µs/date for the 2,000-entity row). The C++ kernel
figures in §2.1 did *not* — re-measuring `rsi(200k)` and `engle_granger(8k)` after the same
sustained load drifted only 1.09× and 0.90×, so this is not thermal throttling of the
machine. It is variance in the pandas/Python layer itself, which is exactly the layer this
item proposes to remove.

The numbers above are the ones `tests/bench/bench_universe.py` reproduces, and they are the
*pessimistic* end of the observed range. Treat the **shape** as the finding — hundreds of µs
to several ms per date, one to three orders of magnitude above any kernel in §2.1 — and not
the third significant figure.

**Monte Carlo** — already good:

| sims | horizon | terminal-only | full matrix |
|---:|---:|---:|---|
| 10,000 | 252 | 1.18 ms | 3.81 ms (0.02 GB) |
| 100,000 | 252 | 17.15 ms | 37.89 ms (0.20 GB) |
| 1,000,000 | 252 | **213.8 ms** | 2.02 GB — not attempted |

252 M path-steps in 214 ms ≈ 1.2 G steps/s across all 16 logical processors. This kernel
is **not** a problem. Its only issue is that the full-matrix variant is unusable above ~500k sims
because the *output* is 2 GB (§5.3).

### 2.4 Where the Python tax actually is

Measured at 2,000 tickers × 2,000 bars:

| Path | Total | Per ticker |
|---|---:|---:|
| 2000 × raw binding `c.rsi()` | 38.0 ms | 19.0 µs |
| 1 × contiguous 4 M-bar `c.rsi()` | 32.7 ms | — |
| **pybind11 dispatch overhead** | 5.3 ms | **2.7 µs (14%)** |
| 2000 × Python wrapper `indicators.rsi()` | **636 ms** | **318 µs (16.7×)** |

This is the single most important line in the document for scoping work item **B**.
The C++ call boundary is *not* the problem — 2.7 µs per call. The **Python wrapper is**:
pandas `Series` → NumPy conversion, `require_finite_array`, logging, and `Series`
reconstruction cost 16.7× the kernel itself. A batch kernel that keeps the per-ticker
pandas layer wins ~14%. A batch *entry point* that removes it wins ~94%.

---

## 3. P0 — Cointegration at universe scale

The flagship item. Three changes, independently valuable, multiplicative together.

### 3.1 One nested QR for ADF lag selection — ✅ SHIPPED

> **Result: 9.6–17.9× measured, growing with n.** Not the 26–36× predicted below; that
> estimate was wrong and the arithmetic is corrected under "What it actually delivered".
> statsmodels parity holds to 8.2e-13 on the ADF statistic across 200 random pairs, with
> zero disagreements on the `cointegrated` flag.

**Problem.** `adf_test` scores every candidate lag `p ∈ [0, L]` by building a fresh
`(T × k)` design matrix and running a full column-pivoted QR, `k = ntrend + 1 + p`. Total
cost is `Σₚ O(T·(p+1)²) = O(T·L³)`. With Schwert's rule `L = ceil(12·(n/100)^¼)`, `L` grows
as `n^¼`, so the whole thing is `O(n^1.75)` — matching the measured `n^1.99`.

**Evidence.** Forcing `max_lag` to a constant removes essentially all of the cost:

| n | auto `L` | `max_lag=0` | `max_lag=5` | auto | auto / ml=0 |
|---:|---:|---:|---:|---:|---:|
| 500 | 18 | 0.016 ms | 0.082 ms | 1.312 ms | 81× |
| 2000 | 26 | 0.034 | 0.465 | 17.998 | 534× |
| 8000 | 36 | 0.126 | 1.293 | **245.931** | **1953×** |

**The fix.** The candidate designs are **perfectly nested**. `fit_lag` emits columns in the
order `[const?, y_{t-1}, Δy_{t-1}, …, Δy_{t-p}]`, and the selection pass already holds every
candidate to a *common sample* (`start_t = max_lag + 1`). So model `p` is exactly the first
`ntrend + 1 + p` columns of model `L`, on identical rows.

Therefore: build the design **once** at `p = L`, run **one** Householder QR without column
pivoting, and read every candidate's residual sum of squares straight off the transformed
response:

```
RSS(p) = Σ_{i ≥ ntrend+1+p} (Q'b)_i²
```

One `O(T·L²)` factorization replaces `L` factorizations totalling `O(T·L³)`.

**Expected gain (as originally estimated — see below for what was wrong).** `L×` on the
selection pass, **26× at n=2000, 36× at n=8000**.

**What it actually delivered.**

| n | before | after | speedup |
|---:|---:|---:|---:|
| 500 | 1.100 ms | 0.115 ms | 9.6× |
| 1000 | 5.330 | 0.506 | 10.5× |
| 2000 | 17.998 | 1.622 | 11.1× |
| 4000 | 63.767 | 6.274 | 10.2× |
| 8000 | 245.931 | 13.776 | **17.9×** |

**The `L×` estimate was wrong, and it is worth recording why.** The old sweep costs
`Σₚ O(T·(p+1)²) ≈ O(T·L³/3)`, not `O(T·L³)`. Against the new `O(T·L²)` that is a factor of
`L/3`, not `L` — 8.7× at L=26 and 12× at L=36. The measured numbers *beat* that corrected
figure (11.1× and 17.9×) because the change also removed column pivoting from the sweep,
and pivoting recomputes remaining column norms at every step, which is itself `O(T·k²)` on
top of the reflections.

So: right idea, right mechanism, arithmetic off by 3×. The honest summary is
**~10× at the sizes a pair scan actually uses, rising to ~18× at n=8000.**

**A new bottleneck surfaced.** With the sweep no longer dominating, the single large QR is,
and it is now *memory*-bound rather than flop-bound at large n:

| n | design matrix | `T·k²` | ms per Mflop |
|---:|---:|---:|---:|
| 8000 | 2.37 MB | 11.0 M | 0.99 |
| 12000 | 3.94 MB | 20.2 M | 1.31 |
| 16000 | 5.63 MB | 31.0 M | 2.21 |
| 24000 | 9.41 MB | 57.6 M | 2.69 |

Cost per flop nearly triples as the design outgrows cache. The cause is layout: the design
is **row-major** while the factorization is **column-oriented**, so every reflection strides
through memory with stride `k`. Storing it column-major would make every inner loop
unit-stride. This does not affect the pair-scan use case — at n=500–2000 the design is
76 KB–600 KB and fully cache-resident — so it is logged as a follow-up, not done here.

**Risks and design notes.**
- Pivoting must be dropped for the nested read-off, since it reorders columns. The
  *reported* fit at the winning lag still runs the existing pivoted `qr::lstsq` on its own
  longer sample, so the rank policy and the t-statistic path are untouched.
- Rank deficiency in the unpivoted sweep must be detected from the `R` diagonal and that
  lag excluded, matching today's `if (!sol.full_rank) continue`.
- **Gate:** the selected lag, `ic_min` and `adf_statistic` must match the current
  implementation *exactly* on the 200-series statsmodels-parity corpus already used in
  `tests/cpp_bindings/test_cpp_cointegration.py`. This is a pure reassociation of the same
  arithmetic, so exact agreement is the right bar, not a tolerance.

**Effort:** ~1 day. **Risk:** low — self-contained in `adf_test`, strong existing gate.

### 3.2 A batch pair-scan kernel — ✅ SHIPPED

> **Result: the 2,000-ticker scan went from 9.81 hr to 5.31 min at 2,000 bars (111×),
> and from 61.7 min to 46.7 s at 500 bars (79×).** Output is bit-identical to a loop of
> `engle_granger()` calls, field by field.
>
> An unforeseen second win did most of the parallel scaling. The batch kernel initially
> reached only 3.0–3.8× and got *slower* past 6 threads, which was neither scheduling
> (guided was already in place) nor allocation (4× more compute per pair did not help).
> It was memory: the QR design was stored **row-major** while the factorization walks
> **columns**, so every element access pulled a fresh cache line and used 8 bytes of it.
> Column-major storage made the inner loops unit-stride, flattened cost-per-flop
> (0.99→0.52 ms/Mflop at n=8000, 2.69→0.77 at n=24000), and by itself fixed the thread
> scaling — monotonic to 16 threads, 5.9× instead of 2.4×.
>
> That is worth recording as a correction: §5.1 diagnosed non-monotonic scaling as a
> *scheduling* problem. For this kernel it was bandwidth. Both fixes are machine-agnostic,
> but for different reasons.

**Problem.** `agent/tools.py:2006` loops `for a, b in combinations(valid_tickers, 2)` in
Python, calling `cointegration_test` per pair. At 2,000 tickers that is 1,999,000 iterations,
each paying the wrapper tax (§2.4) and none of them parallel.

**The fix.** A single native entry point:

```cpp
// prices: (n_tickers × n_bars) row-major, already aligned on a common index
// pairs:  (n_pairs × 2) indices into prices
// out:    (n_pairs × 8) — hedge_ratio, adf_stat, p_value, cv1/5/10, half_life, n_obs
void batch_engle_granger(const double* prices, std::size_t n_tickers,
                         std::size_t n_bars, const int* pairs,
                         std::size_t n_pairs, int max_lag, bool use_aic,
                         double* out);
```

with `#pragma omp parallel for` over pairs. Each pair is fully independent — no shared
state, no RNG — so this is the same shape as `batch_run_strategy`, which already exists and
works.

**Expected gain.** Removes 2 M × (wrapper + binding) and parallelizes. Against the measured
5.1× real-world OpenMP scaling: **~16×** end-to-end on top of §3.1.

**Effort:** ~1 day (kernel + binding + a Python-side `scan_cointegrated_pairs`).
**Risk:** low.

### 3.3 A cheap pre-filter before the ADF

**Problem.** Every pair pays the full Engle–Granger cost, including the pairs that are
obviously not cointegrated. In a real 2,000-name universe the overwhelming majority are.

**The fix.** Gate on something that costs one BLAS call for the whole universe before
spending an ADF on any pair:

1. Correlation matrix of log-price *differences* — one `(n_tickers × n_bars)` GEMM,
   `O(N²·n)` total but at memory-bandwidth speed rather than `O(N²)` factorizations.
2. Keep pairs above a correlation floor (or within a sector/cluster block).
3. Run §3.2 only on survivors.

**Expected gain.** Entirely dependent on the gate, which is a *modelling* choice, not a
performance one — so this must be **opt-in and off by default**, with the threshold a
documented parameter. A correlation floor of 0.7 typically retains low single-digit
percentages of pairs, i.e. another order of magnitude, but that number is a property of the
universe and must not be asserted in advance.

**Effort:** ~half a day. **Risk:** *this changes results.* It is a screening heuristic, and
must be documented as one — a pair the filter drops is never tested, so the scan is no
longer exhaustive. Default off.

### 3.4 Combined effect

2,000 tickers, 2,000 bars, 16 cores:

| Stage | Estimate |
|---|---:|
| Today | 36.8 min (optimistic) / **9.8 hr serial** |
| + §3.1 nested QR (26×) | ~85 s |
| + §3.2 batch kernel + real 5.1× scaling | **~1.5 min** |
| + §3.3 pre-filter (opt-in) | seconds |

The first two rows are extrapolations from measured per-pair costs; §3.1's 26× is derived
from measured lag counts, not assumed.

---

## 4. P1 — Port the remaining universe-scale Python

### 4.1 Portfolio simulation bar loop — ✅ SHIPPED, and this section's diagnosis was wrong

> **Result: 188.7 → 35.8 ms at 1,000 tickers × 2,000 bars (5.3×); 63.8 → 10.5 ms at
> 500 × 504 (6.1×).** Below the 20–50× predicted below, and for an instructive reason.
>
> **The bar loop was not the bottleneck.** This section derived 124.6 µs/bar by dividing
> total runtime by bar count — which charges the whole function to the loop. Profiling
> after the native kernel was in and only 1.1–2.0× faster:
>
> | | |
> |---|---:|
> | `_matrix` (building the dense price matrices) | 0.780 s of 0.844 s — **92%** |
> | the bar loop and everything else | 0.064 s — 8% |
>
> `_matrix` called `frame.loc[master_index, column]` once per *(ticker, column)* — 2,500
> calls at 500 tickers — each taking pandas' 2-D tuple-key path and each recomputing the
> same row alignment for that ticker. Resolving positions once per **ticker**, and
> skipping alignment entirely when every frame is already on the master calendar, is most
> of the win. The native kernel adds a further 1.7–3.3× on top and is what keeps scaling
> as bars grow.
>
> This is the same finding the existing comment above `_matrix` already records — "the
> bottleneck was never the accounting; it was addressing the data" — one level further
> out. The earlier fix moved from a lookup per *(ticker, bar)* to one per
> *(ticker, column)*; at universe scale the per-column one had become the cost.
>
> **Not bit-identical**, deliberately: Python sums with `np.sum` (pairwise), the kernel
> sequentially. Measured 4.5–20 ULPs on the equity curve, growing like √(accumulation
> count). Tolerance-gated, following `rolling_beta`'s AVX2 precedent.

**Problem.** `run_portfolio_simulation`'s `for bar, date in enumerate(master_index)` loop
(portfolio_engine.py:753) is Python. It has already been carefully optimized — dense
matrices materialized once, positional indexing, a vectorized rebalance fast path — and
still costs **125 µs/bar at 500 tickers**, extrapolating to ~450 µs/bar at 2,000.

The remaining cost is ~8 NumPy dispatches per bar over vectors that are too small to amortize
dispatch, plus per-bar Python float boxing.

**The fix.** Port the loop to C++, exactly as `run_strategy` is the single-asset case. It is
a sequential state machine over `(cash, shares_vec)` with per-bar vector reductions — the
shape C++ wins at most.

```cpp
struct PortfolioSimResult { std::vector<double> equity, cash, gross, net; /* + rebalance log */ };

PortfolioSimResult run_portfolio_simulation_native(
    const double* close, const double* open, const double* hl2,   // (n_bars × n_tickers)
    const double* volume, const double* weights,                   // (n_rebal × n_tickers)
    const int* rebalance_bars, std::size_t n_bars, std::size_t n_tickers,
    std::size_t n_rebal, const PortfolioCosts& costs);
```

**Scope discipline — this is the risk.** The Python function supports `pct` and `per_share`
commission models, an impact model, an ADV participation constraint, borrow fees, margin
interest, three fill-price modes and insolvency detection. Porting all of it in one step is
how this goes wrong.

Port **only the vectorized fast path** the Python already carves out
(`commission_model == "pct" and not use_impact_model and max_adv_participation is None`),
and dispatch to the existing Python loop for everything else. That fast path is the default
configuration and almost certainly the one every backtest actually uses.

**Expected gain.** By analogy with `run_strategy` (58× measured end-to-end), realistically
**20–50×** on the bar loop, i.e. 0.9 s → ~20–45 ms at 2,000 tickers.

**Gate:** bit-identical equity curve, cash, gross/net series and rebalance log against the
Python path across randomized universes, the way `run_strategy_summary` is gated against
`run_strategy` today.

**Effort:** ~3–4 days. **Risk:** medium — large surface, but the fast-path restriction and
a bit-identical gate contain it.

### 4.2 Cross-sectional panel primitives — ⬜ NOT DONE

> Still the largest un-taken item. The measurement stands: 4852 µs/date at 2,000 entities
> for `top_bottom_quantile`, 2.4 s for a 504-date panel, with cost driven by per-date
> Python rather than cross-section size.

**Problem.** `transform_predictions_to_weights` loops per date in Python
(portfolio_eval.py:422), calling `apply_exposure_targets` — which itself contains an
iterative capping loop (`_cap_book`, `for _ in range(n + 1)`). Measured **4852 µs/date** at
2,000 entities for `top_bottom_quantile`; 2.4 s for a 504-date panel.

Note the shape: `top_bottom_quantile` at **100** entities (2077 µs/date) is *slower* than
`cross_sectional_rank` at **2,000** (1555 µs/date). Cost is driven by per-date Python and
pandas overhead, not by the size of the cross-section — which is the signature of a loop
that should not be in Python at all.

`_cap_book` itself is fine and should be ported as-is: it is a water-filling loop that
breaks on the first pass whenever `max_position_weight` is not binding, so it is O(n), not
the O(n²) its `range(n + 1)` bound suggests. It is only expensive because it runs once per
date from Python.

**The fix.** One native call over the whole `(n_dates × n_entities)` panel:

```cpp
void transform_score_panel(const double* scores, const bool* availability,
                           std::size_t n_dates, std::size_t n_entities,
                           const TransformSpec& spec, double* out_weights,
                           double* out_diagnostics);
```

Each date is independent (the function's own docstring stresses this is what makes it
point-in-time), so `#pragma omp parallel for` over dates is trivially correct.

Needs native cross-sectional rank, z-score, quantile selection and the exposure-capping
iteration. All are small, well-specified, and easy to gate.

**Expected gain.** Removes essentially all of the per-date Python. Estimate **10–30×**:
2.4 s → ~80–240 ms for a 504-date × 2,000-entity panel. The wide range reflects the
measurement variance noted in §2.3 — the finding is the order of magnitude, not the factor.

**Effort:** ~2 days. **Risk:** low–medium.

### 4.3 Panel indicator entry points — ✅ SHIPPED

> **Result: 1727.6 → 144.7 ms for five indicators over 500 tickers × 1,000 bars (11.9×).**
> At the binding layer, 2,000 × 2,000: 596.2 → 98.2 ms (6.1×). Bit-identical to calling
> `technical_indicators()` once per ticker.
>
> The decomposition below held up: batching the C++ calls alone was worth ~14%; the win is
> removing the pandas layer and parallelizing. `indicators/panel.py` is the entry point.

**Problem.** §2.4. The feature builder loops per entity
(`modeling/dataset/builder.py:297`), and each iteration pays 318 µs of Python wrapper for
19 µs of kernel.

**The fix.** Panel-shaped entry points taking `(n_tickers × n_bars)` and returning the same,
parallel over tickers — plus a Python API that accepts a DataFrame and returns a DataFrame
without going through per-ticker `Series` round-trips.

```cpp
void rsi_panel(const double* prices, std::size_t n_tickers, std::size_t n_bars,
               int period, double* out);
void technical_indicators_panel(...);
```

**Expected gain, decomposed honestly:**

| Component | Gain |
|---|---|
| Removing the per-ticker Python wrapper | 636 ms → 38 ms (**16.7×**) |
| Removing pybind11 per-call dispatch | 38 ms → 32.7 ms (1.16×) |
| OpenMP across tickers | ~32.7 ms → ~6.5 ms (~5×) |
| **Combined** | **636 ms → ~7 ms (~90×)** |

The middle row is the one to notice: **batching the C++ calls is worth almost nothing.**
Nearly all the win is removing the pandas layer and parallelizing. If effort is limited,
a pure-Python panel wrapper that converts once and loops over `c.rsi()` captures 16.7× of
the 90× for a fraction of the work.

**Effort:** ~2 days for the full native panel API; ~2 hours for the Python-only 16.7×.
**Risk:** low.

---

## 5. P2 — Parallel efficiency and memory

### 5.1 Fix the OpenMP scheduling — ✅ items 1–2 SHIPPED

> **Result: `schedule(guided)` on every parallel loop; scaling is monotonic.**
> `batch_run_strategy` was 44.5 ms on 6 threads and 60.4 ms on 8 — that regression is
> gone (40.9 → 39.9). `rolling_hurst` is now fastest at 16 threads (23.8 ms) rather than
> at 12 (was 28.3 ms at 16), a 16% gain at full width.
>
> No thread count is hardcoded, per item 2. The reasoning lives once in `omp_policy.hpp`.
>
> Items 3 (per-thread allocation churn) and 4 (false sharing) are **not done**. The
> §3.2 finding suggests measuring memory traffic before either: for `batch_engle_granger`
> the scaling ceiling turned out to be bandwidth, and neither allocation nor scheduling.



§2.2 established the ceiling (~9.7× on this hybrid part, not 16×) and the symptom: scaling
is **non-monotonic**, with real regressions at 4, 8 and 14 threads. Realistic headroom is
**1.7–2.3×** across every parallel kernel, for no algorithmic change.

**The fix must be machine-agnostic.** Everything measured above came from one hybrid
laptop, and this library runs on servers, CI containers with cgroup CPU quotas, other
laptops, and ARM. A recommendation like "cap threads at 12 because 12 is this box's P-core
count" would be tuning the library to the machine that happened to profile it — on a
16-core homogeneous server it is a 25% throughput cut for no reason. The changes below are
chosen because they are *right everywhere*, and the hybrid CPU is only the environment that
made the existing problem visible.

In priority order:

1. **Replace `schedule(static)` with `schedule(guided)`.** Every parallel loop in the
   codebase uses `static`, which assigns each thread an equal *count* of iterations up
   front and never rebalances. That is only optimal when every iteration costs the same
   *and* every thread runs at the same speed — and neither holds in general:

   - **Uneven work per iteration.** `rolling_hurst`'s per-window cost genuinely varies,
     because `log_sizes` yields a different box count per window. Nothing about that is
     machine-specific.
   - **Uneven threads.** Hybrid P/E cores (Intel 12th gen+, Apple silicon, ARM
     big.LITTLE), SMT siblings sharing a physical core, a cgroup CPU quota, another
     process on the box, or thermal/power asymmetry. All of these are common; the hybrid
     laptop here is just one instance.

   `guided` hands out shrinking chunks on demand, so a thread that finishes early takes
   more work. It adapts to whatever the machine turns out to be rather than assuming.
   One line per pragma, no effect on results.

2. **Do not hardcode a thread count.** The right default is what the environment already
   says: `SQT_NUM_THREADS` when set, otherwise OpenMP's own default, which honours
   `OMP_NUM_THREADS` and on Linux generally reflects the container's CPU allocation. The
   observation that this box prefers 12 threads for `rolling_hurst` is evidence for item 1
   — load imbalance — not a number to ship. Fixing the scheduling is what removes the
   reason a thread count above the P-core count hurt in the first place.

3. **Per-thread allocation churn.** Each `rolling_hurst` window builds several small
   `std::vector`s and a `HurstResult` holding two `std::string`s. Hoist what can be hoisted
   into the existing `RollingHurstScratch`. Allocator contention across threads is
   machine-independent.

4. **False sharing** on the results arrays. `batch_run_strategy` writes `results[t]` from
   every thread. `BacktestResult` is large enough that this is probably fine — but it is
   unmeasured, so confirm before assuming either way.

**Verification must also be machine-agnostic.** "Faster on this laptop" is not the bar. The
gate is that scaling becomes **monotonic** — adding a thread never makes a kernel slower —
which is a property that should hold on any machine and is exactly what fails today.

**Effort:** ~2 days including measurement. **Risk:** low — items 1–3 have no effect on
results, and are gated by the existing thread-count-independence tests.

### 5.2 `rolling_factor_loadings`: QR update/downdate — ⬜ NOT DONE

> Deliberately not attempted in this pass. It is the one item on this list whose stated
> risk is *high with precedent* — the analogous Cholesky update/downdate was implemented,
> gated and reverted — and it deserves its own focused pass with the adversarial corpus
> ready, not the tail end of a long one.
>
> One input from §3.2 that did not exist when this was written: the column-major layout
> change made the QR inner loops unit-stride, so re-measure `rolling_factor_loadings`
> before starting. Some of the 29 ms may already be gone, and it shares no code with the
> nested path, so it has *not* been re-measured here.

**Problem.** The heaviest per-bar kernel: 29 ms at n=20k/k=3/w=60, **128 ms** at k=10,
**120 ms** at w=252. Per-window QR is `O(n·w·p²)`.

This is a *known, deliberate* regression. `rolling_regression.cpp` documents it: the previous
rank-1 normal-equations update was 15–62× faster and wrong (its pivot test compared every
column against the intercept column's diagonal, returning all-NaN for factor values around
1e-6). The comment names the fix and declines to attempt it:

> the obvious next step … is QR update/downdate across the sliding window (which restores
> O(p²) per bar without giving back the conditioning or the rank policy). Deliberately not
> attempted here — correctness first, then profile.

We have now profiled. **This is the "then".**

**The fix.** Givens-rotation update/downdate of the `R` factor as the window slides:
add the entering row, downdate the leaving row. `O(p²)` per bar instead of `O(w·p²)`.

**Expected gain.** `w×` in principle — 60× at w=60, 252× at w=252. Realistically far less
after per-bar overhead; **10–30×** is a defensible target. 29 ms → ~1–3 ms.

**Risk — high, and there is precedent.** `performance_insights.md` item J records that the
analogous **Cholesky** update/downdate was implemented, gated, and **reverted**: it agreed to
1e-13 on well-conditioned data but broke down by ~30× relative on near-singular input and
5.3% on large-baseline input.

QR update/downdate is substantially better conditioned than Cholesky update/downdate — that
is the entire reason the file moved to QR — so this is not the same experiment. But it must
be held to the same gate: **the same adversarial corpus (near-singular, collinear,
large-baseline, 1e-12 to 1e12 scale sweeps), and the same escape hatch.** Downdating is the
numerically dangerous direction; a periodic full refactorization every `w` bars (the idiom
already used elsewhere in the file) bounds the drift and costs one full QR per window.

**Effort:** ~3 days including the gate. **Risk:** high, mitigated by a documented revert.

### 5.3 Monte Carlo: the output is the bottleneck — ⬜ NOT DONE, and option 1 below was wrong

The core is fine (1.2 G steps/s). But `simulate_forward_paths` at 1 M sims × 252 days needs
a **2.02 GB** output array, which is why only the terminal-only variant was measurable.

> **Correction.** Option 1 as originally written — "compute percentile bands natively and
> never materialize the matrix" — does not work, and it is worth saying why rather than
> quietly dropping it. A per-bar percentile across N paths needs all N values *at that
> bar*. Paths are generated path-major and sequentially in time, so at the moment bar `t`
> of path `i` exists, no other path has reached bar `t` yet. There is no streaming order
> in which exact per-bar quantiles fall out.
>
> What is actually available, and should be chosen deliberately rather than bolted on:
>
> - **Approximate quantiles** (t-digest, P²) — genuinely streaming, bounded memory, at the
>   cost of exactness. Probably the right answer for percentile *bands*, which are a
>   visualization.
> - **Bar-chunked regeneration** — exact, memory bounded by `n_sims × chunk`, paid for
>   with `horizon/chunk` times the compute. Viable only because each path is
>   independently seeded and therefore reproducible.
> - **`float32` output** — an honest 2×, no approximation, no extra compute. 1 M × 252
>   becomes 1.01 GB.
>
> The terminal-only variant already exists and is exact, so the gap is specifically
> *per-bar bands at very large N*, which is narrower than this section originally implied.

Options 2 and 3 stand as written:

2. **Chunked streaming.** Yield the matrix in row blocks so peak memory is bounded.
3. **`float` output.** Halves the footprint; simulated equity paths do not need 15
   significant digits. Opt-in.

**Expected gain:** not speed — *feasibility* above ~500 k simulations.

---

## 6. What NOT to do

Stated explicitly, with the evidence, so it does not get relitigated:

- **Per-bar indicator kernels** (`rsi`, `wilder_atr`, `adx`, `parabolic_sar`, `bollinger`,
  `stochastic`). 1.6–7.9 ms over 200,000 bars, all linear. `performance_insights.md` already
  measured most at ~1× against warm numba. There is nothing here.
- **Batching indicator calls to reduce pybind11 dispatch.** Measured at 2.7 µs/call, 14% of
  the per-ticker cost (§2.4). The Python wrapper is the 16.7× problem; the C++ boundary is not.
- **The Monte Carlo inner loop.** 1.2 G path-steps/s. Its problem is output size, not speed.
- **`SQT_RESTRICT` / LTO / PGO tuning.** All three were measured in the previous pass at
  no detectable difference on this toolchain.
- **`ols2`, `garch`, `kalman`.** Sub-millisecond at every realistic size.

---

## 7. Sequencing

| Phase | Items | Status | Predicted | **Measured** |
|---|---|---|---|---|
| **1** | §3.1 nested QR + §3.2 batch pair scan | ✅ | 2,000-ticker scan 9.8 hr → ~1.5 min | **9.81 hr → 5.31 min (111×)** at 2,000 bars; 61.7 min → 46.7 s at 500 |
| **2** | §4.3 panel indicators | ✅ | ~17× on feature builds | **11.9×** (1727.6 → 144.7 ms, 5 indicators × 500 tickers) |
| **2** | §5.3 Monte Carlo reduction | ⬜ | MC usable at 1 M sims | **specification was wrong** — see §5.3 |
| **3** | §4.1 portfolio simulation | ✅ | 20–50× | **5.3×** — and the bottleneck was `_matrix`, not the loop |
| **4** | §5.1 OpenMP scheduling | ✅ | ~2× across parallel kernels | scaling is **monotonic**; `rolling_hurst` 16% better at full width |
| **4** | §4.2 panel weight transform | ⬜ | 10–30× | not attempted |
| **5** | §5.2 `rolling_factor_loadings` | ⬜ | 10–30× or a revert | not attempted — deserves its own pass, see §5.2 |
| opt-in | §3.3 correlation pre-filter | ⬜ | another order of magnitude | not attempted; *changes results*, so it is a modelling call |

Phase 1 was indeed worth more than everything else combined, and by a wider margin than
predicted.

**Three predictions in this document were wrong, and each is kept next to its result**
rather than edited away: §3.1's speedup was over-estimated 3× by bad arithmetic (the sweep
is `Σₚ(p+1)² = L³/3`, so the factor is `L/3`, not `L`); §4.1 blamed the bar loop when 92%
of the time was in building the price matrices; and §5.3's headline option cannot work at
all, because exact per-bar quantiles have no streaming order. The measurements that
contradicted them are the point of having committed the harnesses.

---

## 8. Verification

Nothing in this plan ships without both of these.

**Correctness gates.** Every item names its gate above. The pattern already established in
this repo is the right one: a bit-identical or tolerance-gated comparison against the
existing implementation, run over an adversarial corpus, with a documented escape hatch for
items that may not survive it (§5.2 explicitly may not).

**Performance regression gates.** The current C++ benchmarks (`bench_backtest`, `bench_hurst`)
cover 2 of 24 kernels and assert only loose upper bounds. Before phase 1:

1. Commit the two harnesses used for this document —
   a per-kernel scaling benchmark and a universe-scale benchmark — under `tests/bench/`.
2. Record this document's §2 tables as the baseline.
3. Add a CI job that fails on a >20% regression in any kernel, so the next correctness fix
   that costs 3× is noticed when it lands rather than a year later.

That third point is not hypothetical. The `rolling_factor_loadings` QR change was a
deliberate, documented 15–62× slowdown — and it was the right call — but nothing in the test
suite would have caught it if it had been accidental.

---

## Appendix: measurement environment

- MSVC 19.44.35228.0, Ninja, `CMAKE_BUILD_TYPE=Release`, `SQT_NATIVE_ARCH=ON`
- Flags: `/O2 /arch:AVX2 /GL` + LTO/IPO, OpenMP 2.0
- **i7-13620H — 6 P-cores (12 threads) + 4 E-cores (4 threads); 10 physical / 16 logical.**
  A hybrid part: results on a homogeneous server CPU will differ, and §5.1's scheduling
  diagnosis is specific to this class of machine (though `schedule(guided)` is not a
  pessimization on a homogeneous one).
- Python 3.12, NumPy 2.x, pandas 2.x, statsmodels 0.14.3
- All timings min-of-N with GC disabled; `SQT_NUM_THREADS=1` for serial figures
