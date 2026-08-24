# Performance

Every number here is **measured, not projected**, on a Windows 11 /
MSVC 19.44 / Python 3.12 development machine with 16 logical cores. Each row
toggles the same module's own `HAS_CPP` flag and times both paths
back-to-back, so it is an apples-to-apples comparison rather than separately
run numbers.

Two things to read this page with in mind. The C++ extension is **optional**
— every kernel has a Python fallback, the API is identical either way, and
the package is fully usable without a compiler. And several entries are
honest disappointments kept beside their predictions, because a performance
document that only records wins is not a record of anything.

For the methodology, the benchmark scripts, and a running log of edge-case
bugs found while building and measuring this, see
[Development/performance_insights.md](../Development/performance_insights.md).
For the modeling layer specifically, see
[Development/modeling_native_plan.md](../Development/modeling_native_plan.md),
which states the arithmetic ceiling on that work before the method.
The optional compiled C++ extension accelerates the highest-impact CPU-bound paths. The API is identical with or without it — pure Python fallback is automatic.

**Measured, not projected**, on a Windows 11 / MSVC 19.44 / Python 3.12 dev machine (16 logical cores) — each row toggles the same module's own `HAS_CPP` flag and times both paths back-to-back, so it's an apples-to-apples comparison, not separately-run numbers:

| Operation | vs. numba (warm)¹ | vs. numba JIT cold-start² | Notes |
|---|---|---|---|
| `hurst_exponent` DFA (n = 500) | **83×** (4.57ms → 0.05ms) | — | No numba path exists for Hurst — this is C++ vs. the pure-Python fallback directly. |
| `hurst_exponent` DFA (n = 2 000) | **131×** (12.3ms → 0.09ms) | — | Same. |
| `rolling_hurst` (n = 2 000, window = 200, step = 1) | **274×** (4.64s → 17ms) | — | Same — the standout number in this table, and it holds up under real measurement. |
| `rsi` (n = 2 000) | **5.3×** (0.47ms → 0.09ms) | 1109ms → 1.2ms first call | |
| `adx` (n = 2 000) | **0.9×** (essentially tied) | 1110ms → 1.2ms first call | Numba's *warm* ADX is already about as fast as C++ on this machine — see the note below. |
| `parabolic_sar` (n = 2 000) | **1.1×** (essentially tied) | ~similar order to ADX | |
| `wilder_atr` (n = 2 000) | **28×** (4.40ms → 0.15ms) | | |
| `bollinger_bands` (n = 2 000) | **1.6×** | | |
| `stochastic_oscillator` (n = 2 000) | **2.6×** | | |
| `cointegration_test` (n = 500, vs. statsmodels) | **23×** (8.3ms → 0.37ms) | — | Compares against statsmodels, not numba — statsmodels has no JIT path at all. |
| `cointegration_test` (n = 2 000, vs. statsmodels) | **86×** (86.9ms → 1.01ms) | — | The ratio grows with n because the kernel is no longer quadratic: the ADF lag sweep used to run one column-pivoted QR per candidate lag, `O(T·L³)` in total, and now reads every candidate's residual off one nested factorization, `O(T·L²)`. |
| `scan_cointegrated_pairs` (2 000 tickers, 2 000 bars) | **111×** (9.81 h → 5.31 min) | — | One native call over the whole pair set instead of ~2 M Python round trips, parallel across pairs. |
| `calculate_beta` (n = 500, vs. `lstsq`) | **1.4×** | — | |
| `half_life` (n = 500, vs. `lstsq`) | **1.1×** | — | |
| `run_strategy` (n = 2 000, `include_trade_log=False`) | **~58×** (26.8ms → 0.46ms) | — | A wrapper-redundancy bug, not a kernel problem — see note below. Was ~1.0× before the fix. |
| `batch_run_strategy` (n = 2 000, num_tests = 2 000) | **~11×** (51.6ms → 4.6ms) | — | Allocation-free summary kernel + OpenMP across parameter combinations (16 cores); ranges ~6–11× depending on grid size — see `Development/performance_insights.md`. |
| `rolling_beta` (n = 2 000, window = 60) | **4.7×**, plus a further ~1.1–1.5× from optional AVX2+FMA dispatch | — | |
| `rolling_factor_loadings` (n = 500, window = 60, k = 3) | **5.5×** (8.9ms → 1.6ms) | — | Was 26× when this used an incremental Cholesky update. That path was removed because it was wrong on small-magnitude factors (all-NaN where NumPy answered correctly); the replacement is a per-window rank-revealing QR. 10.0× at n=2 000/window=60, 2.3× at window=252 — the gap narrows as the window grows, since cost is `O(n·window·p²)`. |
| `technical_indicators_panel` (500 tickers × 1 000 bars, 5 indicators) | **11.9×** (1 727.6ms → 144.7ms) | — | vs. looping the per-ticker Python wrappers. The pybind11 boundary was never the cost (2.7 µs/call, 14%) — the per-ticker pandas round trip was, at 318 µs against 19 µs of kernel. |
| `run_portfolio_simulation` (1 000 tickers × 2 000 bars) | **5.3×** (188.7ms → 35.8ms) | — | Most of it was *not* the bar loop: profiling put 92% in building the dense price matrices, one pandas `.loc` per (ticker, column). The native bar-loop kernel adds a further 1.7–3.3× on top. |
| `fit_preprocess_stats` (per-column winsorize + moments) | **5.5–23.5×** | — | Replaces two `Series.quantile` calls, a `clip` and two moments per column. Must reproduce pandas' *conventions*, not just its arithmetic: linearly interpolated quantiles, ddof=1, NaN skipped but infinities kept. |
| `apply_preprocess_stats` (clip + standardize) | **14.5–53.6×** | — | One fused pass; the Python form allocated two full-panel temporaries per column. |
| `standardize_by_date` (cross-sectional z-score) | **8.6–11.6×** | — | Per-date centre, scale and clip over a counting-sorted panel. |
| `cross_sectional_correlation` (per-date IC) | **3.0–6.2×** | — | spearman 4.9–6.2×, pearson 3.0–4.2×. Counting-sorts rows by date in O(n), replacing an argsort and two gathers. |
| `cross_sectional_correlation` (pooled rank IC) | **1.6–3.0×** | — | Same kernel, one segment. The pooled case has no per-date parallelism to draw on, so the ranking sort splits into per-thread runs and merges above 50 000 rows. |
| `label_uniqueness` (label-overlap weights) | **8–23×** | — | Concurrency by difference array, O(n) where sweeping each label's span is O(n·horizon). Gated below 50 000 rows, where the argument conversion costs more than the Python loop saves. |
| `rolling_hurst` (n = 2 000, window = 200) | **274×** vs. Python, plus a further ~10.5× from OpenMP + a one-pass DFA reformulation on top of the *original* C++ implementation (measured independently, at the same n/window) | — | Combining the two independently-measured ratios gives roughly ~2 900× vs. the pure-Python fallback at this size — not itself a single direct measurement, but both factors are real. |
| `simulate_forward_paths` (n_simulations = 5 000, horizon = 60) | **2.0×** (74.8ms → 37.7ms) | — | No numba path ever existed for this one — was pure uncompiled Python. See OpenMP note below for the parallel path's own measured speedup. |
| `garch11_variance_recursion` (n = 2 000, warm steady-state) | **0.8×** (10.8ms → 12.9ms, i.e. slightly *slower*) | 219ms → 4.8ms first call | The whole point of this port is the cold-start column, not this one — see below. |
| `kalman_filter_*`, `donchian_state_machine`, `vwap_reversion_state_machine` | not separately re-measured | same cold-start pattern as GARCH/ADX above | |

¹ **This is C++ vs. numba, not C++ vs. interpreted Python** — numba is fully functional on this dev machine (NumPy 2.0.2), so the "Python fallback" path for RSI/ADX/PSAR/GARCH/Kalman/signal-state-machines actually means *numba-JIT-compiled*, already close to C speed once warm. On a machine where numba is broken or unavailable (e.g. NumPy 2.4+, which is what originally motivated porting RSI/ADX/PSAR to C++ in the first place), the true comparison is C++ vs. an *interpreted* Python loop, which would show much larger gains than this table — those older, unmeasured "10–30×"-style estimates are directionally right for that scenario, just not what this table reports. Hurst and cointegration have no numba path at all, so their numbers above are already the "real" comparison either way.

² Measured via a genuinely fresh subprocess per number (`time.perf_counter()` around the very first call, nothing warmed up beforehand) — this is the number that actually matters for a single one-off agent-tool call in a new process, which is the primary reason GARCH/Kalman/Donchian/VWAP-reversion were ported at all (see `Development/performance_insights.md`).

**Two honest findings from actually measuring this**, worth calling out rather than hiding:
- **`run_strategy` originally showed only ~1.0× end-to-end**, not the then-documented 3–8×, even though the raw C++ kernel genuinely was faster in isolation (confirmed by `tests/cpp/bench_backtest.cpp`'s native-only numbers below). The gap was never the kernel — it was the Python wrapper: `pct_change`/`shift` computed unconditionally before the C++ dispatch check even though the C++ path never used them, and an unconditional Python trade-log rebuild that overwrote already-correct native stats every call. **Since fixed** (removing both, and only building the Python trade log when a caller actually asks for it via `include_trade_log=True`) — the real, current number is **~58×** (26.8ms → 0.46ms), reflected in the table above. `batch_run_strategy` never had this specific bug (its consumer already read native stats directly), but has since gained its own further ~6–11× from an allocation-free summary kernel plus OpenMP across the parameter grid.
- **OpenMP's measured speedup for `simulate_forward_paths` is ~2.0–2.4×** on this 16-core machine (min-of-7-runs across separate process invocations, `n_simulations=200 000`) — not the near-linear-with-cores scaling the per-path independence would suggest in theory. MSVC's OpenMP support here is version 2.0 (an older spec) — some of that gap was expected going in. A later pass eliminating each path's small per-path RNG/buffer allocations moved this scaling ratio only within noise (~2.4×→~2.1×, both real measurements) — the allocation being eliminated turned out not to be the dominant cost at this problem size, a legitimate change worth keeping regardless (fewer allocations is never worse) but not the win that framing initially suggested.

**A third honest finding, from the modeling kernels.** The plan for that work
opened by stating a *ceiling* rather than a target: feature preprocessing was
47–56% of a walk-forward run and everything else is pandas plumbing no kernel
reaches, so ~2× end-to-end was the arithmetic limit however fast the kernel
got. Measured afterwards: **1.59–2.55×** end-to-end, while the kernels
themselves are 3–53×. The prediction held, and after the first phase the
attribution shifted exactly as it implied — preprocessing fell to 13% of a
run and "everything else" rose to **70%**. That is why the work stopped at
three kernels instead of chasing the remaining 70% with tools that cannot
reach it. Two smaller things went wrong on the way and are recorded in
`Development/modeling_native_plan.md`: the plan missed the pooled rank IC
entirely (41–51% of `regression_metrics`, larger than the per-date IC it did
name, and only visible on re-measuring between phases), and two kernels were
initially *slower* than the Python they replaced at small sizes — fixed with
a cheaper argument conversion and an explicit size gate, because a fast path
that is slower is a bug rather than a trade-off.

Raw C++-only (no Python involved) numbers from `tests/cpp/bench_hurst.cpp` and `tests/cpp/bench_backtest.cpp`, run via `ctest`:

| Operation | Time |
|---|---|
| `hurst_dfa` (n = 2 000) | 0.107 ms |
| `rolling_hurst` DFA (n = 2 000, window = 200, step = 1) | 16.9 ms |
| `rolling_hurst` DFA (n = 5 000, window = 252, step = 1) | 60.8 ms |
| `run_strategy` long-only, all costs (n = 2 000) | 0.017 ms |
| `run_strategy` mixed L/F/S signals, all costs (n = 5 000) | 0.089 ms |

The rolling Hurst gain is the most significant and the most robust to how you measure it: rather than re-entering Python for every bar, the entire sliding-window pass runs in one C++ function, with no numba equivalent to compare against either way.

`rolling_factor_loadings` is the one entry in this table that got **slower on purpose**. It used incremental rank-1 XtX updates — O(k²) per bar instead of a full O(n·k²) `lstsq` — and that was 26×. It was also wrong: the pivot test compared every column against the single largest diagonal of XtX, which belongs to the intercept column and equals the window length, so factors around 1e-6 made every window read as singular and the kernel returned all-NaN where the NumPy fallback returned correct coefficients. It now runs a column-pivoted QR per window, which ranks each column by its own norm and gives a scale-invariant answer, at 2.3–10×. Recovering the speed via QR update/downdate is planned but not attempted — see `Development/optimization_plan.md` §5.2, including why the analogous Cholesky attempt was reverted.

**Deeper native optimization pass** (on top of the module-level wins above): `run_strategy`/`batch_run_strategy` and `rolling_hurst` now parallelize across independent work (parameter combinations, rolling windows) via OpenMP; several kernels' Python/C++ boundary crossings were converted to direct-write into a pre-allocated NumPy buffer instead of allocate-then-copy; `rolling_beta` gained an optional runtime-dispatched AVX2+FMA reduction path (falls back safely to the portable scalar kernel on older CPUs); the build enables LTO/IPO automatically and supports an opt-in, local-only PGO workflow. One optimization (a rank-1 Cholesky *factor* update/downdate, intended to replace `rolling_factor_loadings`'s O(p³) per-step refactor with O(p²)) was implemented, gated against the existing path on real before/after data, found to break down numerically on near-singular inputs, and reverted rather than shipped — documented in `CHANGELOG.md` alongside the items that did ship.

See [Development/performance_insights.md](../Development/performance_insights.md) for the full methodology, every number above with its exact benchmark script, and a running log of real edge-case bugs found and fixed while actually building, running, and benchmarking this codebase — not assumed from reading the code (a degenerate-input NaN in the cointegration ADF test, a half-life NaN-vs-inf gap, an input-validation gap in the Monte Carlo binding, incorrect hand-written C++ test expectations that had never been compiled before, and a Linux-CI-only flake in an audit-trail test caused by an unfiltered directory glob, among others).

---
## Python-Level Optimisations

Confirmed benchmarks on a 2 000-bar series (Python 3.12, NumPy 2.4):

| Optimisation | Before | After | Speedup | Notes |
|---|---|---|---|---|
| ATR true range | 2.8 ms (`pd.concat` + `.max`) | 0.49 ms (`np.maximum`) | **5.6×** | Single-pass; eliminates 3 Series + concat |
| Trade log serialization | 31 ms (`iterrows`, 500 trades) | 3.6 ms (`to_dict`) | **~9×** | Vectorized dict conversion |
| CVaR computation | 0.83 ms (two-pass) | 0.44 ms (one-pass) | **1.9×** | Single `np.percentile` + boolean mask |
| SPY beta screen | N HTTP requests | 1 request per worker | **~N/workers×** | SPY pre-fetched once per batch — 1 total for single-process runs, once per worker for `n_workers > 1` |
| Backtesting equity curve | — | NumPy cumprod | vectorized | `(1 + returns).cumprod()` |
| Portfolio covariance | — | BLAS `pandas.cov` | BLAS-backed | O(n·k²) via LAPACK |
| Screener (50+ tickers) | — | ProcessPoolExecutor | multi-core | Auto async→multiprocess threshold |
| Portfolio simulation (100 tickers × 2 000 bars, monthly) | 1 503 ms (per-ticker `.loc`) | 32 ms (dense matrices) | **47×** | 200 000 pandas label lookups replaced by positional indexing; 500 tickers → **78×** |
| Cross-sectional IC (252 dates × 50 entities) | 91.5 ms (`groupby` + `Series.corr` per date) | 1.26 ms (array passes) | **47.8×** | Was **72%** of a ridge walk-forward run. Balanced panels reshape to `(n_dates, n_entities)`; ragged ones use `np.add.reduceat` over segment bounds. Agreement with the per-date version is 2.2e-16 (spearman) / 5.0e-16 (pearson), including ties and NaN. The multiple shrinks to 1.8× at 2 000 entities as the per-date overhead amortizes. |
| Walk-forward fold masks | `panel["date"].isin(...)` per fold | one `searchsorted` + a per-date gather | — | Also keeps working for splitters whose folds are not contiguous, which purged K-fold needs. |

> **Portfolio simulator note:** `run_portfolio_simulation` holds prices, target weights and liquidity baselines as dense `(n_bars × n_tickers)` matrices and executes the default cost configuration as array arithmetic. The vectorized rebalance is deliberately narrow — `per_share` commission, the impact model and the ADV constraint each need a per-element decision (a per-order minimum, a per-ticker volatility lookup, an error naming one ticker) and keep the explicit loop, selected automatically by cost model. Both routes are held to the same numbers by tests: agreement with the pre-vectorization implementation is within 1.7e-15 relative across every configuration, with `rebalance_log` identical, the residual being pairwise-vs-sequential summation rather than a different formula. The speedup grows with universe size because the removed cost scaled with tickers × bars. See [Documentation/04_backtesting.md](04_backtesting.md).

> **Cross-sectional IC note:** the centered two-pass correlation form is not
> a refinement. The textbook `n·Σxy − ΣxΣy` shortcut differences two nearly
> equal large numbers on return-scale data and loses most of its significant
> digits; switching to the centered form moved pearson agreement from 2.2e-14
> to 5.0e-16, and a test pins the tighter tolerance so it cannot drift back.
> The same trap caught the native preprocessing kernel from the other
> direction: pandas sums *pairwise* via numpy, and a sequential accumulator
> disagreed in the 12th significant digit until the kernel was changed to
> match. Both are recorded in `Development/modeling_analysis.md`.

> **Numba note:** RSI, ADX, Parabolic SAR, GARCH's variance recursion, the Kalman filter, and every backtest-strategy state machine (RSI/Bollinger/Donchian/VWAP-reversion) are decorated with `@njit`. This requires Numba with a compatible NumPy version (≤ 2.0, or wherever Numba's own ABI support currently ends). On an incompatible NumPy version, Numba decorators are a no-op and the code falls back to interpreted Python, where C++ genuinely wins big (the original ~10–30× estimates for RSI/ADX/PSAR describe this scenario). On a machine where Numba *is* working (like the one that produced the measured table below), it's already close to C speed once warm — real measurement shows C++ landing anywhere from a tie to a modest win against it, not a blowout. What C++ reliably wins either way: no per-process JIT compile tax (measured at ~200ms–1.1s on the first call in a fresh process, gone entirely with C++) and no numpy-ABI fragility risk (the exact failure mode that motivated porting RSI/ADX/PSAR to C++ in the first place). Every one of these falls back to pure Python automatically when neither C++ nor Numba is available.

---
