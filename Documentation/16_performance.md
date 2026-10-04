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

For the methodology and a running log of edge-case bugs found while
building and measuring this, see [CHANGELOG.md](../CHANGELOG.md); the
harnesses every figure comes from are described in
[tests/bench/README.md](../tests/bench/README.md). The modeling layer's
kernels are the third honest finding below, which states the arithmetic
ceiling on that work before its method.

Each row is the compiled path against the same module's own fallback, at
the size named:

| Operation | vs. numba (warm)¹ | vs. numba JIT cold-start² | Notes |
|---|---|---|---|
| `hurst_exponent` DFA (n = 500) | **83×** (4.57ms → 0.05ms) | — | No numba path exists for Hurst — this is C++ vs. the pure-Python fallback directly. |
| `hurst_exponent` DFA (n = 2 000) | **131×** (12.3ms → 0.09ms) | — | Same. |
| `rolling_hurst` (n = 2 000, window = 200, step = 1) | **274×** (4.64s → 17ms) | — | Same — the standout number in this table, and it holds up under real measurement. |
| `rsi` (n = 2 115, raw kernel) | **0.97–1.05×** (tied) | 1109ms → 1.2ms first call | Re-measured at the raw kernel, without the wrapper. 0.78–0.99× before the per-bar loop was rewritten; see the fourth finding below. |
| `adx` (n = 2 115, raw kernel) | **0.94–0.99×** (tied) | 1110ms → 1.2ms first call | Was **0.25–0.33×** — a quarter to a third of numba's speed at every size from 2k to 2M bars — until the per-bar loop was rewritten; see the fourth finding below. |
| `parabolic_sar` (raw kernel, 200k–2M bars) | **1.19–1.45×** | ~similar order to ADX | Was 0.82–0.92× from 20k to 2M bars: MSVC compiled the new-extreme update to a branch that went either way from bar to bar. See the fifth finding below. |
| `wilder_atr` (n = 2 115, raw kernel) | **0.92–1.00×** (tied) | | Was **0.24–0.29×** before the same rewrite. |
| `bollinger_bands` (n = 2 000) | **1.6×** | | |
| `stochastic_oscillator` (n = 2 000) | **2.6×** | | |
| `cointegration_test` (n = 500, vs. statsmodels) | **23×** (8.3ms → 0.37ms) | — | Compares against statsmodels, not numba — statsmodels has no JIT path at all. |
| `cointegration_test` (n = 2 000, vs. statsmodels) | **86×** (86.9ms → 1.01ms) | — | The ratio grows with n because the kernel is no longer quadratic: the ADF lag sweep used to run one column-pivoted QR per candidate lag, `O(T·L³)` in total, and now reads every candidate's residual off one nested factorization, `O(T·L²)`. |
| `scan_cointegrated_pairs` (2 000 tickers, 2 000 bars) | **111×** (9.81 h → 5.31 min) | — | One native call over the whole pair set instead of ~2 M Python round trips, parallel across pairs. |
| `calculate_beta` (n = 500, vs. `lstsq`) | **1.4×** | — | |
| `half_life` (n = 500, vs. `lstsq`) | **1.1×** | — | |
| `run_strategy` (n = 2 000, `include_trade_log=False`) | **~58×** (26.8ms → 0.46ms) | — | A wrapper-redundancy bug, not a kernel problem — see note below. Was ~1.0× before the fix. |
| `batch_run_strategy` (n = 2 000, num_tests = 2 000) | **~11×** (51.6ms → 4.6ms) | — | Allocation-free summary kernel + OpenMP across parameter combinations (16 cores); ranges ~6–11× depending on grid size. |
| `rolling_beta` (n = 2 000, window = 60) | **4.7×**, plus a further ~1.1–1.5× from optional AVX2+FMA dispatch | — | |
| `rolling_factor_loadings` (n = 500, window = 60, k = 3) | **5.5×** (8.9ms → 1.6ms) | — | Was 26× when this used an incremental Cholesky update. That path was removed because it was wrong on small-magnitude factors (all-NaN where NumPy answered correctly); the replacement is a per-window rank-revealing QR. 10.0× at n=2 000/window=60, 2.3× at window=252 — the gap narrows as the window grows, since cost is `O(n·window·p²)`. |
| `technical_indicators_panel` (500 tickers × 1 000 bars, 5 indicators) | **11.9×** (1 727.6ms → 144.7ms) | — | vs. looping the per-ticker Python wrappers. The pybind11 boundary was never the cost (2.7 µs/call, 14%) — the per-ticker pandas round trip was, at 318 µs against 19 µs of kernel. |
| `run_portfolio_simulation` (1 000 tickers × 2 000 bars) | **5.3×** (188.7ms → 35.8ms) | — | Most of it was *not* the bar loop: profiling put 92% in building the dense price matrices, one pandas `.loc` per (ticker, column). The native bar-loop kernel adds a further 1.7–3.3× on top. |
| `fit_preprocess_stats` (per-column winsorize + moments) | **5.5–23.5×** | — | Replaces two `Series.quantile` calls, a `clip` and two moments per column. Must reproduce pandas' *conventions*, not just its arithmetic: linearly interpolated quantiles, ddof=1, NaN skipped. An infinite training value is refused on both backends (it has no quantile to winsorize to). |
| `apply_preprocess_stats` (clip + standardize) | **14.5–53.6×** | — | One fused pass; the Python form allocated two full-panel temporaries per column. |
| `standardize_by_date` (cross-sectional z-score) | **8.6–11.6×** | — | Per-date centre, scale and clip over a counting-sorted panel. |
| `cross_sectional_correlation` (per-date IC) | **3.0–6.2×** | — | spearman 4.9–6.2×, pearson 3.0–4.2×. Counting-sorts rows by date in O(n), replacing an argsort and two gathers. |
| `cross_sectional_correlation` (pooled rank IC) | **1.6–3.0×** | — | Same kernel, one segment. The pooled case has no per-date parallelism to draw on, so the ranking sort splits into per-thread runs and merges above 50 000 rows. |
| `label_uniqueness` (label-overlap weights) | **8–23×** | — | Concurrency by difference array, O(n) where sweeping each label's span is O(n·horizon). Gated below 50 000 rows, where the argument conversion costs more than the Python loop saves. |
| `rank_by_date` (per-date average rank) | **4.4–22×** | — | The one modelling operation where VECTORISING LOST: a numpy rewrite measured 395 ms against pandas' `groupby.rank` at 407 ms, and slower at five columns. 4.4× at 50 entities, 16.6–22× at 500. Ranks every column in one call, and an extraction rather than new arithmetic — `average_ranks` already implemented the tie semantics. |
| `permutation_null_ic` (whole permutation loop) | **68–88×** | — | The loop crosses the boundary entire, not the correlation alone: a third of the cost was constructing per-draw pandas Series nobody reads. Ranks once, since shuffling values inside a date permutes their ranks. Two numpy attempts at the same idea measured 0.14× and 0.6×. Seeded reproducibly WITHIN a backend only, the contract `simulate_forward_paths` states. |
| `hierarchical_risk_parity` (2 106 days × 235 assets) | **4.5×** (341ms → 77ms; `frame.corr()` 260ms → 8–11ms) | — | Against pandas' `frame.corr()`, not numba: the correlation matrix was 78–86% of the call. The `pearson_correlation` kernel is pandas' `nancorr` arithmetic per pair, bit for bit, with each complete column's Welford recursion computed once and reused in every pair it is in; a pair with a gap runs pandas' loop as written. Weights, cluster order and risk contributions are identical to the `frame.corr()` path on pandas 2.3.3 (4.46×) and 3.0.5 (4.68×, 295ms → 63ms). |
| `rolling_hurst` (n = 2 000, window = 200) | **274×** vs. Python, plus a further ~10.5× from OpenMP + a one-pass DFA reformulation on top of the *original* C++ implementation (measured independently, at the same n/window) | — | Combining the two independently-measured ratios gives roughly ~2 900× vs. the pure-Python fallback at this size — not itself a single direct measurement, but both factors are real. |
| `simulate_forward_paths` (n_simulations = 5 000, horizon = 60) | **2.0×** (74.8ms → 37.7ms) | — | No numba path ever existed for this one — was pure uncompiled Python. See OpenMP note below for the parallel path's own measured speedup. |
| `cusum_peaks` (200 AR(1) null paths × 2 105 steps) | **20×** (14.4ms → 0.71ms) | — | Against the numpy loop it replaces; there is no numba path. `cusum` gains 2.5× and `detect_basis_dislocation` 1.7× end to end. Bit-identical. The recursion clips at zero, so it has no filter or scan form, and every vectorized closed form measured slower than the loop. Serial on purpose: on 16 threads the kernel ran in a sixth of the time and its callers got 1.7× slower, because OpenMP's idle workers keep spinning beside the Python that runs next. |
| `garch11_variance_recursion` (n = 2 000, warm steady-state) | **0.8×** (10.8ms → 12.9ms, i.e. slightly *slower*) | 219ms → 4.8ms first call | The whole point of this port is the cold-start column, not this one — see below. |
| `kalman_filter_*`, `donchian_state_machine`, `vwap_reversion_state_machine` | not separately re-measured | same cold-start pattern as GARCH/ADX above | |

¹ **This is C++ vs. numba, not C++ vs. interpreted Python** — numba is fully functional on this dev machine (NumPy 2.0.2), so the "Python fallback" path for RSI/ADX/PSAR/GARCH/Kalman/signal-state-machines actually means *numba-JIT-compiled*, already close to C speed once warm. On a machine where numba is broken or unavailable (e.g. NumPy 2.4+, which is what originally motivated porting RSI/ADX/PSAR to C++ in the first place), the true comparison is C++ vs. an *interpreted* Python loop, which would show much larger gains than this table — those older, unmeasured "10–30×"-style estimates are directionally right for that scenario, just not what this table reports. Hurst and cointegration have no numba path at all, so their numbers above are already the "real" comparison either way.

² Measured via a genuinely fresh subprocess per number (`time.perf_counter` around the very first call, nothing warmed up beforehand) — this is the number that actually matters for a single one-off agent-tool call in a new process, which is the primary reason GARCH/Kalman/Donchian/VWAP-reversion were ported at all.

**Two honest findings from actually measuring this**, worth calling out rather than hiding:
- **`run_strategy` originally showed only ~1.0× end-to-end**, not the then-documented 3–8×, even though the raw C++ kernel genuinely was faster in isolation (confirmed by `tests/cpp/bench_backtest.cpp`'s native-only numbers below). The gap was never the kernel — it was the Python wrapper: `pct_change`/`shift` computed unconditionally before the C++ dispatch check even though the C++ path never used them, and an unconditional Python trade-log rebuild that overwrote already-correct native stats every call. **Since fixed** (removing both, and only building the Python trade log when a caller actually asks for it via `include_trade_log=True`) — the real, current number is **~58×** (26.8ms → 0.46ms), reflected in the table above. `batch_run_strategy` never had this specific bug (its consumer already read native stats directly), but has since gained its own further ~6–11× from an allocation-free summary kernel plus OpenMP across the parameter grid.
- **OpenMP's measured speedup for `simulate_forward_paths` is ~2.0–2.4×** on this 16-core machine (min-of-7-runs across separate process invocations, `n_simulations=200 000`) — not the near-linear-with-cores scaling the per-path independence would suggest in theory. MSVC's OpenMP support here is version 2.0 (an older spec) — some of that gap was expected going in, though the spec was not the cause: linking LLVM's newer runtime instead left this kernel where it was on 16 threads (0.99×, [Build variants](#build-variants-openmp-runtime-profile-guided-optimization-and-clang-cl) below). A later pass eliminating each path's small per-path RNG/buffer allocations moved this scaling ratio only within noise (~2.4×→~2.1×, both real measurements) — the allocation being eliminated turned out not to be the dominant cost at this problem size, a legitimate change worth keeping regardless (fewer allocations is never worse) but not the win that framing initially suggested.

**A third honest finding, from the modeling kernels.** That work opened by
stating a *ceiling* rather than a target: feature preprocessing was 47–56%
of a walk-forward run and everything else is pandas plumbing no kernel
reaches, so ~2× end-to-end was the arithmetic limit however fast the kernel
got. Measured afterwards: **1.59–2.55×** end-to-end, while the kernels
themselves are 3–53×. The prediction held, and once the first kernels
landed the attribution shifted exactly as it implied — preprocessing fell to
13% of a run and "everything else" rose to **70%**. That is why the work
stopped at three kernels instead of chasing the remaining 70% with tools
that cannot reach it. Two smaller things went wrong on the way: the first
survey missed the pooled rank IC entirely (41–51% of `regression_metrics`,
larger than the per-date IC it did name, and visible only on re-measuring
after the first kernel landed), and two kernels were initially *slower* than
the Python they replaced at small sizes — fixed with a cheaper argument
conversion and an explicit size gate, because a fast path that is slower is
a bug rather than a trade-off.

**A fourth honest finding: three kernels were slower than the numba code
they replace.** Timed at the raw kernel, `adx` ran at 0.25–0.33× of numba
and `wilder_atr` at 0.24–0.29×, linear in n from 2k to 2M bars, so the cost
was per element. The arithmetic was not the cause: MSVC compiled three ways
of writing the per-bar loop into calls or unpredictable branches.
`std::max({a, b, c})` for the true range became an out-of-line call into the
standard library's `max_element`, whose loop branches on which candidate is
largest; `c ? x : 0.0` on doubles became a conditional jump mispredicted on
about every other bar; and `std::isfinite` became a call into the C runtime
DLL, three per bar. Rewritten without any of them, the kernels are level
with numba and every output is bit-identical to the previous build. The
build guide's section on adding a C++ feature lists the three patterns,
with the fourth from the finding below.

**A fifth honest finding: the same call into the C runtime was still in
seven kernels, and the Parabolic SAR had the branch.** `std::isfinite` and
`std::isnan` are a call to `_dclass` in the CRT DLL under MSVC, once per
value. Replaced with `numerics::is_finite` / `is_nan` in the per-bar and
per-row loops, and in the two numerics helpers Bollinger and
`rolling_beta` call per bar, with every output bit-identical (min of 7
interleaved runs, one thread): `bollinger_bands` 1.35–1.65×,
`stochastic_oscillator` 1.14–1.78×, `rolling_beta` 1.68–2.51×,
`run_portfolio_simulation` with daily rebalancing 1.07–1.42×,
`cross_sectional_correlation` 1.41–1.51× (Pearson) and 1.05–1.09×
(Spearman), `standardize_by_date` 1.03–1.07×, `implied_volatility_batch`
1.01–1.16×. `fit_preprocess_stats`, `rank_by_date`, `permutation_null_ic`
and `rolling_factor_loadings`, where a sort or a QR dominates, measured
within noise and keep the library call. The Parabolic SAR's gap was a
branch: `if (high > ep) { ep = high; af = min(af + step, af_max); }`
compiled to a jump that goes either way unpredictably, where numba's LLVM
emits selects. Written as a max and an add of a masked step, the kernel
runs at 1.19–1.45× of numba from 200k to 2M bars where it ran at
0.82–0.92×, with the previous build's bits. The numba fallback pays for
its new gap test: 0.83× its previous speed at 2k bars, 0.93× at 200k and
0.99× at 2M.

Raw C++-only (no Python involved) numbers from `tests/cpp/bench_hurst.cpp` and `tests/cpp/bench_backtest.cpp`, run via `ctest`:

| Operation | Time |
|---|---|
| `hurst_dfa` (n = 2 000) | 0.107 ms |
| `rolling_hurst` DFA (n = 2 000, window = 200, step = 1) | 16.9 ms |
| `rolling_hurst` DFA (n = 5 000, window = 252, step = 1) | 60.8 ms |
| `run_strategy` long-only, all costs (n = 2 000) | 0.017 ms |
| `run_strategy` mixed L/F/S signals, all costs (n = 5 000) | 0.089 ms |

The rolling Hurst gain is the most significant and the most robust to how you measure it: rather than re-entering Python for every bar, the entire sliding-window pass runs in one C++ function, with no numba equivalent to compare against either way.

`rolling_factor_loadings` is the one entry in this table that got **slower on purpose**. It used incremental rank-1 XtX updates — O(k²) per bar instead of a full O(n·k²) `lstsq` — and that was 26×. It was also wrong: the pivot test compared every column against the single largest diagonal of XtX, which belongs to the intercept column and equals the window length, so factors around 1e-6 made every window read as singular and the kernel returned all-NaN where the NumPy fallback returned correct coefficients. It now runs a column-pivoted QR per window, which ranks each column by its own norm and gives a scale-invariant answer, at 2.3–10×. Recovering the speed via a QR update/downdate is not attempted: the analogous Cholesky attempt was implemented, gated against the existing path on real data, found to break down numerically on near-singular inputs, and reverted rather than shipped.

**Deeper native optimization pass** (on top of the module-level wins above): `run_strategy`/`batch_run_strategy` and `rolling_hurst` now parallelize across independent work (parameter combinations, rolling windows) via OpenMP; several kernels' Python/C++ boundary crossings were converted to direct-write into a pre-allocated NumPy buffer instead of allocate-then-copy; `rolling_beta` gained an optional runtime-dispatched AVX2+FMA reduction path (falls back safely to the portable scalar kernel on older CPUs); the build enables LTO/IPO automatically and supports an opt-in, local-only PGO workflow. One optimization (a rank-1 Cholesky *factor* update/downdate, intended to replace `rolling_factor_loadings`'s O(p³) per-step refactor with O(p²)) was implemented, gated against the existing path on real before/after data, found to break down numerically on near-singular inputs, and reverted rather than shipped — documented in `CHANGELOG.md` alongside the items that did ship.

See the CHANGELOG for the full methodology, every number above with its exact benchmark script, and a running log of real edge-case bugs found and fixed while actually building, running, and benchmarking this codebase — not assumed from reading the code (a degenerate-input NaN in the cointegration ADF test, a half-life NaN-vs-inf gap, an input-validation gap in the Monte Carlo binding, incorrect hand-written C++ test expectations that had never been compiled before, and a Linux-CI-only flake in an audit-trail test caused by an unfiltered directory glob, among others).

---
## Option chains

`analysis.options_batch` prices a chain in one call (see
[12_options.md](12_options.md#whole-chains-in-one-call)). One solve or one
greek is cheap — 30–55 µs for an implied volatility and 3.5 µs for
`black_scholes_greeks` on this machine — so the cost of a chain was never
the formula; it was one Python call, one validation and one dict per
contract, hundreds or tens of thousands of times. Measured with
[`tests/bench/bench_options.py`](../tests/bench/bench_options.py), min of 7
warm runs, three runs on 2026-09-28 (the ranges are run-to-run spread on a
shared workstation, not error bars):

| Workload | Scalar loop | Batch, C++ (16 threads) | Batch, C++ (`SQT_NUM_THREADS=1`) | Batch, numpy fallback |
|---|---|---|---|---|
| Implied vol, 476-contract chain (17 expiries × 14 strikes × call/put, 36 quotes refused) | 14–27 ms (`implied_volatility` per contract) | **0.21–0.28 ms** (60–95×) | 0.39 ms | 3.8–10 ms |
| Greeks, 61 spots × 476 contracts = 29,036 valuations | 100–109 ms (`black_scholes_greeks`); 196–208 ms (`option_greeks`, the full set the batch returns) | **0.32–0.37 ms** (~300×) | 1.7 ms | 10–11 ms |
| `zero_gamma_spot`, 476 legs, 61-spot scan + 5 Brent steps | — | 0.7–1.0 ms | 2.0 ms | 11–14 ms |
| `simulate_delta_hedge`, default 500 paths × 21 rebalances | 79–85 ms | **3.5–4.1 ms** (21×) | — | 9–11.5 ms |
| `option_risk_scenarios`, 61 spot × 21 vol shocks | 3.6–3.8 ms | 0.66–0.90 ms (4–5.5×) | — | 2.0–2.3 ms |
| `option_risk_scenarios`, default 7 × 5 grid | 0.11–0.13 ms | 0.09 ms (1.3–1.5×) | — | 0.23–0.26 ms |

Every batched number is the same result as the loop beside it, not an
approximation of it: the tests hold the kernel, the fallback and the scalar
functions to each other on randomized chains, and the hedge simulation and
the scenario grid to their old per-contract loops double for double.

**Three honest findings.**

- **Threads buy little on one chain's implied volatility.** 476 solves are
  0.39 ms serial and 0.21–0.28 ms on 16 threads: a solve is ~0.8 µs of work,
  so thread start-up and guided scheduling eat most of the split. The greek
  grid, 29,036 valuations, is where OpenMP pays — 1.7 ms to 0.3 ms. Both go
  through the same work gate as every other kernel (`omp_policy.hpp`).
- **The numpy fallback is 1.5–10×, not 50×.** It takes `exp`, `log` and
  `erf` from `math`, one Python-level call per element, because numpy's
  own vectorised transcendentals may round the last bit differently on some
  CPUs and the batch is held to the scalar's doubles. Exactness was chosen
  over speed on the path that only runs without the extension.
- **A small grid does not pay, and one caller was left alone because of
  it.** The default 7 × 5 scenario grid is 35 valuations; the batch's
  validation and array set-up cost about as much as the loop, so it is
  1.3× on the kernel and 2× *slower* on the fallback. `analyze_strategy`
  prices one leg per contract of a structure — two to four, typically — and
  batching them measured 270 µs → 305 µs on a four-leg condor, so it keeps
  its per-leg loop.

**Selected greeks (2026-10-02).** `black_scholes_greeks_batch(...,
greeks=...)` computes only the greeks it names (see
[12_options.md](12_options.md#whole-chains-in-one-call)). On one thread, the
previous full set against the new selection: a 476 × 61 grid is 4.1–4.6×
faster for gamma and 4.0–4.2× for delta, and the full set itself is 1.6×,
because each contract's terms are now formed once per contract rather than
once per cell. A 2,000 × 201 grid is 4.6–5.0× faster for gamma, and the full
set itself 1.5×. Without a grid (20,000 contracts) gamma is 2.4× and delta
1.8×. At 16 threads gamma and delta are 3–5.7×, but the full set on a grid
is only 1.0–1.25×: writing twelve freshly allocated arrays is then the
bottleneck, not the arithmetic. `zero_gamma_spot` (1,000 legs) is 3.4–4×
faster. The implied-volatility kernel is 1.5–1.7× faster from squaring the
volatility by multiplication rather than `pow`, and the public
`implied_volatility_batch` 1.7–2.9× faster, with its discounting check about
40× cheaper: only a contract whose exponent could overflow a double is
checked with `math.exp`, and a real chain has none. These are ratios
measured old against new in one process on a shared workstation, so read
them as indicative.

---
## Build variants: OpenMP runtime, profile-guided optimization and clang-cl

Three build variants measured against the default build: linking LLVM's
OpenMP runtime in place of MSVC's (`SQT_OPENMP_LLVM`), a profile-guided
build (`SQT_PGO_GENERATE` / `SQT_PGO_USE`), and LLVM's clang-cl in place of
cl. None is the default, and this is the evidence. How to build each is in
[30_build_guide.md](30_build_guide.md#9-notes).

**Method** (the OpenMP-runtime and PGO comparisons; clang-cl's is in its own
subsection). Three builds of one commit — the same source digest — identical
except for the option under test: Release, `SQT_NATIVE_ARCH=ON`, LTO, Ninja,
MSVC 19.44.35228, Python 3.12.1, NumPy 2.0.2. The machine is an Intel Core
i7-13620H laptop (6 performance and 4 efficiency cores, 16 threads) on AC
power under its "Silent" power profile, Windows 11 build 26200, with nothing
else heavy running; measured 2026-09-28. The harness is
[`tests/bench/bench_build.py`](../tests/bench/bench_build.py), on the raw
bindings: in each process every kernel is called back to back for 0.25 s and
then timed 15 times, and that process's median is one sample. Each build and
thread setting got ten processes, interleaved across builds (vcomp, libomp,
PGO, then libomp, PGO, vcomp, ...). Every cell is the **median of the ten
per-process medians, in ms, with their range**; a ratio is default ÷
variant, so above 1 means the variant is faster. A *cold* call is a single
call after 0.3 s without work — a kernel called between stretches of Python,
once the runtime's workers have gone to sleep — six per process. Which build
a result came from is in the file: the harness prints `__build_info__`.

### OpenMP runtime: vcomp (default) against LLVM's libomp

16 threads (the OpenMP default here):

| Kernel | vcomp, warm | libomp, warm | ratio | vcomp, cold | libomp, cold | ratio |
|---|---|---|---|---|---|---|
| `batch_run_strategy`, 2,000 bars × 2,000 | 9.90 [9.35–11.09] | 9.65 [9.14–10.74] | 1.03× | 10.46 [7.55–11.19] | 11.65 [9.56–12.87] | 0.90× |
| `batch_run_strategy`, 500 bars × 5,000 | 6.13 [5.67–7.12] | 5.68 [5.51–7.64] | 1.08× | 8.64 [6.36–9.46] | 9.28 [7.87–9.94] | 0.93× |
| `batch_backtest_crossover`, 2,000 × 2,450 pairs | 4.34 [4.02–5.10] | 4.20 [4.05–4.69] | 1.03× | 7.43 [4.76–8.06] | 9.53 [6.86–11.92] | 0.78× |
| `technical_indicators_panel`, 500 × 1,000 | 14.61 [13.96–15.44] | 14.01 [12.50–16.17] | 1.04× | 14.03 [12.17–15.09] | 16.31 [15.07–17.83] | 0.86× |
| `implied_volatility_batch`, 476 contracts | 0.063 [0.057–0.082] | 0.083 [0.077–0.099] | 0.76× | 0.61 [0.32–0.77] | 1.06 [0.50–1.46] | 0.58× |
| `black_scholes_greeks_batch`, 61 × 476 | 0.93 [0.76–1.15] | 0.74 [0.55–0.82] | 1.25× | 2.64 [0.95–3.02] | 3.26 [2.47–4.02] | 0.81× |
| `rolling_hurst`, n = 5,000, window 252 | 6.60 [6.24–10.87] | 6.35 [6.00–6.98] | 1.04× | 9.32 [5.58–9.65] | 13.25 [10.33–16.39] | 0.70× |
| `simulate_forward_paths`, 20,000 × 252 | 7.08 [6.74–7.41] | 7.13 [6.31–7.81] | 0.99× | 10.05 [8.13–10.76] | 11.83 [7.72–13.23] | 0.85× |

8 threads (`SQT_NUM_THREADS=8`, six processes per build):

| Kernel | vcomp, warm | libomp, warm | ratio | vcomp, cold | libomp, cold | ratio |
|---|---|---|---|---|---|---|
| `batch_run_strategy`, 2,000 bars × 2,000 | 15.22 [14.00–16.38] | 13.15 [12.83–13.91] | 1.16× | 13.56 [12.87–14.33] | 14.00 [13.46–15.09] | 0.97× |
| `batch_run_strategy`, 500 bars × 5,000 | 8.50 [7.71–9.13] | 8.17 [7.60–8.76] | 1.04× | 10.40 [10.08–11.07] | 10.74 [10.32–11.61] | 0.97× |
| `batch_backtest_crossover`, 2,000 × 2,450 pairs | 5.98 [5.69–6.67] | 4.81 [4.61–5.30] | 1.24× | 8.83 [8.25–9.34] | 10.79 [9.42–11.94] | 0.82× |
| `technical_indicators_panel`, 500 × 1,000 | 17.89 [16.60–19.42] | 17.18 [16.21–18.94] | 1.04× | 17.78 [17.08–18.20] | 18.58 [18.01–19.22] | 0.96× |
| `implied_volatility_batch`, 476 contracts | 0.072 [0.058–0.076] | 0.067 [0.064–0.072] | 1.07× | 0.56 [0.50–0.67] | 0.83 [0.38–0.90] | 0.67× |
| `black_scholes_greeks_batch`, 61 × 476 | 0.90 [0.75–1.05] | 0.70 [0.61–0.92] | 1.28× | 2.38 [2.00–2.75] | 3.43 [3.33–3.70] | 0.69× |
| `rolling_hurst`, n = 5,000, window 252 | 8.92 [8.34–9.58] | 6.90 [6.34–8.15] | 1.29× | 10.55 [10.12–10.86] | 12.19 [11.08–13.11] | 0.87× |
| `simulate_forward_paths`, 20,000 × 252 | 7.58 [7.13–8.92] | 6.88 [6.63–7.47] | 1.10× | 10.98 [9.83–11.42] | 11.25 [10.27–11.88] | 0.98× |

With `SQT_NUM_THREADS=1` no region goes parallel and the runtime is never
asked to do anything: all seventeen kernels agree within 0.96–1.04×,
including `rolling_beta` and `rolling_factor_loadings` (1.00–1.01×), which
`/openmp:llvm` was expected to help through the `omp simd` hint. It cannot:
the directive is still a compile error under `/openmp:llvm`, and the loop it
annotates only runs on CPUs without AVX2 and FMA.

**Findings.**

- **Warm, it depends on the thread count.** On all 16 threads six of eight
  parallel kernels land within 0.99–1.08× of each other, inside their own
  ranges; the greek grid is 1.25× faster on libomp and the 476-contract
  implied-volatility solve — 60 µs of work — 0.76×, a costlier fork and
  join. Capped at 8 threads, the way a process pool would run it, libomp is
  faster on every parallel kernel, 1.04–1.29×.
- **Cold, libomp is slower on every parallel kernel at both counts**
  (0.58–0.93× on 16 threads, 0.67–0.98× on 8): its sleeping workers take
  longer to wake. For a library whose kernels are called between stretches
  of Python, that is the common case.
- **vcomp's idle workers cost the caller**: they keep spinning for about
  100 ms after a parallel region, and serial work on the calling thread runs
  about a third slower inside that window. Measured with
  `donchian_state_machine` right after a 2,000 × 2,000 grid: 1.16–1.31 ms in
  the first 100 ms against 0.91 ms steady on vcomp, 0.81–0.86 ms in the same
  window on libomp. It is a cost of a serial kernel following a parallel
  one, not of either kernel, and it is why the harness warms each kernel for
  0.25 s: with a single warm-up call the serial kernels measured up to 1.6×
  slower on vcomp purely because of where they sat in the list. These
  measurements predate the package's own `OMP_WAIT_POLICY=PASSIVE` default,
  which puts the workers to sleep after a region instead (see
  [Runtime defaults](#runtime-defaults-openmp-wait-policy-and-blas-threads)).
- **The decision does not rest on the speed.** `libomp140.x86_64.dll` is not
  in the Visual C++ Redistributable, so a build that needs it does not load
  on a machine without Visual Studio. With a gain in one regime and a loss
  in another to weigh against that, `SQT_OPENMP_LLVM` stays an
  off-by-default option for local builds.

### Profile-guided optimization against the plain build

Both builds link vcomp; the PGO build was trained with
[`tests/bench/pgo_training.py`](../tests/bench/pgo_training.py), five rounds
single-threaded, about 5 s. The training calls the same kernels the table
measures but with its own seeds, sizes and parameters, so this is the
favourable case of a profile trained on the workload it is judged on — a
workload the profile never saw gains less.

| Kernel | plain, 1 thread | PGO, 1 thread | ratio | plain, 16 threads | PGO, 16 threads | ratio |
|---|---|---|---|---|---|---|
| `run_strategy`, 5,000 bars | 0.079 [0.072–0.129] | 0.041 [0.033–0.043] | **1.92×** | 0.076 [0.071–0.091] | 0.041 [0.033–0.047] | **1.87×** |
| `donchian_state_machine`, 200,000 bars | 0.94 [0.87–1.11] | 0.62 [0.56–0.73] | **1.52×** | 0.94 [0.82–1.00] | 0.62 [0.57–0.80] | **1.53×** |
| `vwap_reversion_state_machine`, 200,000 bars | 1.03 [0.93–1.09] | 0.84 [0.76–0.91] | **1.23×** | 1.02 [0.90–1.12] | 0.84 [0.74–1.00] | **1.22×** |
| `batch_run_strategy`, 2,000 bars × 2,000 | 64.4 [61.1–68.0] | 53.8 [51.5–55.6] | **1.20×** | 9.90 [9.35–11.09] | 7.82 [7.62–9.07] | **1.27×** |
| `batch_run_strategy`, 500 bars × 5,000 | 40.1 [38.2–44.4] | 33.4 [30.9–34.3] | **1.20×** | 6.13 [5.67–7.12] | 5.00 [4.68–6.31] | **1.23×** |
| `rolling_beta`, 200,000, window 60 | 4.30 [3.78–5.05] | 4.05 [3.77–4.91] | 1.06× | 4.37 [4.15–4.88] | 4.00 [3.74–4.23] | 1.09× |
| `rolling_beta`, 200,000, window 252 | 4.07 [3.76–4.45] | 3.80 [3.64–3.96] | 1.07× | 4.07 [3.66–4.48] | 3.80 [3.48–3.96] | 1.07× |
| `parabolic_sar`, 200,000 | 1.98 [1.79–2.40] | 1.87 [1.69–1.94] | 1.06× | 1.88 [1.76–2.01] | 1.86 [1.77–2.59] | 1.01× |
| `technical_indicators`, 200,000, all five | 31.7 [30.2–32.5] | 30.8 [30.3–32.2] | 1.03× | 31.5 [30.4–33.6] | 30.7 [29.4–32.3] | 1.03× |
| `technical_indicators_panel`, 500 × 1,000 | 68.0 [67.0–112.7] | 66.6 [64.0–68.3] | 1.02× | 14.61 [13.96–15.44] | 14.00 [13.56–16.84] | 1.04× |
| `rolling_factor_loadings`, 5,000, window 252, k = 3 | 28.4 [27.9–29.2] | 28.3 [27.9–29.5] | 1.00× | 28.4 [27.3–29.2] | 28.0 [27.5–29.6] | 1.01× |
| `implied_volatility_batch`, 476 contracts | 0.190 [0.175–0.208] | 0.189 [0.175–0.203] | 1.01× | 0.063 [0.057–0.082] | 0.065 [0.060–0.082] | 0.97× |
| `black_scholes_greeks_batch`, 61 × 476 | 2.57 [2.26–2.77] | 2.60 [2.53–2.85] | 0.99× | 0.93 [0.76–1.15] | 0.90 [0.76–1.36] | 1.03× |
| `run_portfolio_simulation` kernel, 1,000 × 2,000 | 2.43 [2.36–2.63] | 2.52 [2.38–2.85] | 0.96× | 2.49 [2.30–2.79] | 2.57 [2.35–2.77] | 0.97× |
| `rolling_hurst`, n = 5,000, window 252 | 31.2 [30.2–34.9] | 32.5 [31.9–33.7] | 0.96× | 6.60 [6.24–10.87] | 6.75 [6.55–7.86] | 0.98× |
| `batch_backtest_crossover`, 2,000 × 2,450 pairs | 21.5 [20.3–38.9] | 23.4 [23.0–23.7] | **0.92×** | 4.34 [4.02–5.10] | 4.69 [4.46–5.79] | **0.92×** |
| `simulate_forward_paths`, 20,000 × 252 | 24.9 [24.3–30.3] | 27.1 [26.4–27.5] | **0.92×** | 7.08 [6.74–7.41] | 7.20 [6.96–7.54] | 0.98× |

**Findings.**

- **The gain is where branches are, and it is larger than the usual
  5–15% there.** A single backtest 1.9×, the Donchian state machine 1.5×,
  VWAP reversion 1.2×, and the batch grid — the same per-combination
  summary loop, run 2,000 or 5,000 times — 1.2–1.27× at either thread count.
- **Arithmetic kernels gain nothing that clears their spread**: rolling
  regression, the indicators and the option formulas are 0.97–1.09×, most of
  them inside the range of the plain build; the portfolio bar loop and
  rolling Hurst are 0.96–0.98×.
- **Two kernels got slower**, and consistently so — in all ten processes,
  with ranges that do not overlap at one thread: `batch_backtest_crossover`
  and `simulate_forward_paths`, 8% each, serially. Both were in the
  training; the profile moved inlining and layout decisions against them. A
  profile is a trade, not a free speed-up.
- **It stays off by default.** A profile is a statement about one workload
  on one machine, the build it makes cannot be reproduced from the sources
  alone, and it goes stale with every C++ change (the digest then refuses
  the old extension anyway). For a fixed local workload dominated by single
  backtests and signal state machines it is worth the 30 s it costs; the
  workflow is in [30_build_guide.md](30_build_guide.md#9-notes).

### clang-cl against cl

Measured 2026-10-02 on the same machine with the same harness, seven
interleaved processes per build and the median of per-process medians:
clang-cl 23.1.2 with `/clang:-ffp-contract=off` on every unit, ThinLTO and
LLVM's OpenMP runtime, against cl, both with `SQT_NATIVE_ARCH=ON`.

**Findings.**

- **The same bits.** 44 of 44 sampled kernel outputs are bit-identical to
  cl's at default threads, at `SQT_OMP_MIN_WORK=0` and on one thread, and
  the full suite passes.
- **Faster on many kernels:** `rolling_beta` 2.0–2.2×, Bollinger bands
  1.7–2.0×, `run_strategy` 1.7×, the Donchian and VWAP state machines
  1.2–1.7×, Pearson `cross_sectional_correlation` 1.5–1.9×,
  `apply_preprocess_stats` 1.2–1.5×, the indicators 1.2×, `cusum_peaks`
  1.5×.
- **Slower on a few:** `rolling_factor_loadings` 0.88–0.90×,
  `kalman_filter_2state` 0.89–0.93×, and `implied_volatility_batch` 0.73× at
  default threads (the runtime's fork and join on a 0.06 ms call).
- **It stays an opt-in local build.** `libomp.dll` is not redistributed by
  Microsoft. And the contraction rule is what makes it correct: without
  `/clang:-ffp-contract=off`, clang-cl fused multiply-adds even under
  `/fp:precise`, and 27 of the 44 outputs moved.

---
## Runtime defaults: OpenMP wait policy and BLAS threads

**OpenMP workers sleep after a region.** `import standard_quant_tools` sets
`OMP_WAIT_POLICY=PASSIVE` unless you set it, before the extension loads the
OpenMP runtime, which reads the variable only then; a blank value counts as
unset. vcomp's default kept its workers spinning for about 100 ms after
every region, and the Python that followed ran 1.4–2.2× slower beside them
(see the vcomp findings above). With `PASSIVE`, 16 threads land within 5% of
the best thread count on 28 of 35 public calls, against 7 of 35 before, and
the geometric-mean speed-up over one thread is 2.13× instead of 1.79×. The
price is paid by kernels called back to back with no Python between them,
which run 6–24% slower, and by a small region, which costs 0.1–0.6 ms to
wake the workers. No result changes. To keep the old spinning behaviour,
set `OMP_WAIT_POLICY=ACTIVE` before starting Python. If another package
loads the runtime first — scikit-learn imported before this one loads
vcomp — the default cannot take effect in that process, and a debug-level
log line says so rather than a warning; import this package first to avoid
it. The variable is process-wide, so scikit-learn's OpenMP and child
processes inherit it.

**scikit-learn's OpenMP pays for PASSIVE where nothing holds it back.**
hist_gradient_boosting starts a team on every logical CPU for each fit and
prediction, and its work per region is small on a modeling fold: on a
15,000-row, 8-feature fold 16 sleeping-then-waking threads took 1.5–1.7 s
against 0.28–0.36 s on one thread (ACTIVE: about 0.3 s either way). The
modeling engine therefore holds OpenMP estimators to their share of the
budget, one thread under `"auto"` below 2,000,000 training cells, through
`_blas.openmp_thread_limit`, reference-counted across threads where the
runtime's count is process-wide (vcomp) and set per thread where it is kept
per thread (libgomp, libomp) (see
[15_modeling.md](15_modeling.md#max_parallelism-what-the-budget-controls)).
The manifest's environment records `OMP_WAIT_POLICY` beside the other
thread variables.

**The library's own matrix factorizations run on one BLAS thread.** OpenBLAS
defaults to one thread per logical CPU, which is slower for
covariance-sized matrices: `eigh` at 235 assets takes 2.6× as long on 16
threads as on one, and an SVD of 1,260 × 235 takes 2.7× as long (5.4× under
OpenBLAS 0.3.31). The answers also differ in their last bits with the thread
count. The PSD repair (`_repair_psd`), `max_diversification`'s
pseudo-inverse and condition number, `pca_returns`' SVD,
`estimate_covariance`'s eigenvalues, the portfolio optimizers' condition
numbers, the exact mean-variance solve and every fit on a modeling pool run
on one thread, through a reference-counted, process-wide limit so concurrent
callers cannot undo each other. The result is the same bits on any core
count, at 1.3–10× the speed. At 235 assets, before and after (medians of
15–30 repetitions on a shared workstation):

| Call | Python 3.12, NumPy 2.0 | Python 3.11, NumPy 2.4 (OpenBLAS 0.3.31) |
|---|---|---|
| `_repair_psd`, already PSD | 18.2 → 0.39 ms | 24.9 → 0.42 ms |
| `_repair_psd`, repairs | 18.8 → 6.6 ms | 22.1 → 7.8 ms |
| `risk_parity` | 20.4 → 6.2 ms | 28.0 → 7.6 ms |
| `max_diversification` | 49 → 16 ms | 53 → 18 ms |
| `marginal_risk_contribution` | 22 → 2.1 ms | 24 → 6.6 ms |
| `portfolio_scenarios` | 22 → 2.8 ms | 37 → 7.5 ms |
| `estimate_covariance`, Ledoit-Wolf | 75 → 22 ms | 56 → 15 ms |
| `pca_returns` (SVD) | 202 → 70 ms | 1,387 → 169 ms |

The already-PSD row is mostly a second change: a Cholesky factorization now
decides that no repair is needed (see
[05_portfolio.md](05_portfolio.md#an-indefinite-covariance-is-repaired-and-the-repair-is-named)).
The Ledoit-Wolf row includes no longer computing a precision matrix nothing
read. Products that gain from threads keep them — the sample and Ledoit-Wolf
Gram matrices, the EWMA product, PCA's factor-return product — so an EWMA
covariance and PCA's `factor_returns` still vary in their last bits with the
core count. So can a sample or Ledoit-Wolf covariance, including the one
`DataFrame.cov()` computes for the optimizers, on some OpenBLAS builds:
numpy's on CI's Linux and Windows runners gave the 1,260 × 235 product
different last bits at one and four threads, where this machine's did not.
A condition number or eigenvalue reported from such a matrix is still the
one-thread value of that matrix. While any caller is inside the limit, BLAS work on other threads
of the process also runs on one thread. `SQT_BLAS_THREADS` sets another
count; 0 disables the limit. `threadpoolctl`, a declared dependency, applies
it; where it finds no BLAS it can control, the limit does nothing.

---
## When a kernel goes parallel

A parallel region costs the same to start whatever the kernel, so the
question is how long its work would take on one thread. Each parallel call
site states its work as tasks × units per task, together with the measured
serial cost of one unit for that kernel (`cost` in
[`omp_policy.hpp`](../src/standard_quant_tools/_cpp/include/sqt/omp_policy.hpp)).
A region goes parallel when there is more than one task, more than one
thread, and

    tasks × (ns per task + units per task × ns per unit) ≥ 150 µs

The rule used to be a count: tasks × units against `SQT_OMP_MIN_WORK`,
50,000 by default. A unit is about 1 ns of an implied-volatility solve and
140 ns of an Engle-Granger test, so one count suited the option kernels and
held the rest back until they had milliseconds of serial work.

The costs were measured on the raw bindings with `SQT_NUM_THREADS=1`, as the
slope of time against units at 30 µs to 3 ms of serial work (i7-13620H, MSVC
`/arch:AVX2`), so a call's fixed overhead is not counted as work a region
would split. Where the cost grows with the shape, the figure is the low end,
so a mis-estimate delays the parallel path rather than taking it early. The
size at which each goes parallel is 150,000 ns over the cost of one task:

| Kernel | Unit | ns per unit | Goes parallel from | Before |
|---|---|---|---|---|
| `batch_engle_granger` | one bar of one pair | 140 | about 1,070 units: 5 pairs of 252 bars | 50,000 units |
| `rolling_hurst`, DFA / R/S | one bar of one window | 25 / 34 | 6,000 / about 4,410 units: 60 / 45 windows of 100 bars | 50,000 units |
| `simulate_forward_paths` | one day of one path, plus 730 ns a path | 1.9 | 195 paths at 21 days, 125 at 252 | 50,000 units |
| `simulate_forward_paths_terminal` | one day of one path, plus 740 ns a path | 1.0 | 198 paths at 21 days, 152 at 252 | 50,000 units |
| `technical_indicators_panel` | one bar of one ticker for one indicator | the requested indicators' mean: RSI and ATR 6.5, ADX 10, Bollinger 20, stochastic 45 | about 8,500 units for all five (17.6 ns): 7 tickers of 252 bars | 50,000 units |
| `batch_run_strategy` | one bar of one test | 3.8 | about 39,500 units: 20 tests of 2,000 bars | 50,000 units |
| `batch_backtest_crossover` | one bar of one pair | 5.0 | 30,000 units: 120 pairs of 252 bars | 50,000 units |
| `implied_volatility_batch` | 500 a contract | 0.95 | 316 contracts | 100 contracts |
| `black_scholes_greeks_batch` | 50 a valuation | 1.2 | 2,500 valuations, for a selection of greeks too | 1,000 valuations |
| `fit_preprocess_stats` | one value (rows × columns) | 13 | about 11,500 values | 50,000 values |
| `apply_preprocess_stats` | one value (rows × columns) | 2.2 | about 68,200 values | 50,000 values |
| `cross_sectional_correlation`, Spearman | one row of a date | 40 | 3,750 rows | 50,000 rows |
| `cross_sectional_correlation`, Pearson | one row of a date | 3.0 | 50,000 rows | 50,000 rows |
| `pearson_correlation`, complete columns: each column's recursion | one value | 3.0 | 50,000 values: 252 rows × 199 columns | 50,000 values |
| `pearson_correlation`, complete columns: the pair sums | one row of one pair | 0.2 | 750,000 row-pairs: 252 rows × 77 columns | 50,000 row-pairs |
| `pearson_correlation`, pairs with a gap | one row of one pair | 3.0 | 50,000 row-pairs: 252 rows × 20 columns | 50,000 row-pairs |
| `standardize_by_date` | one value (rows × columns) | 4.5 | about 33,300 values | 50,000 values |
| `rank_by_date` | one value (rows × columns) | 15 | 10,000 values | 50,000 values |
| `permutation_null_ic` | one row of one permutation | 6.0 | 25,000 units | 50,000 units |
| `label_uniqueness` | one row of one entity | 28 | about 5,360 rows | 50,000 rows |

Two costs are set rather than measured. Pearson's per-date loop barely
scales: it measured 8 ns a row, but at the 18,750 rows that would put at the
threshold its parallel path was 0.74–0.81× on a call made after the workers
went idle, and only 1.07–1.19× warm at 37,500 rows, so its cost is set to
switch where the unit rule did, at 50,000 rows. And a selection of greeks
costs 13–30 ns a valuation, but the region still paid at the full set's
threshold (a gamma-only grid of 4,000 valuations took 32 µs on the threads
against 53 µs on one), so a selection keeps the full set's cost.

From 150 µs the parallel path measured 1.4–7× faster than the serial one on
back-to-back calls, and about even (0.75–1.2×) on a call made after the
runtime's workers had gone idle, under either wait policy. Against the
previous build, ratios above 1 faster, warm / cold:

| Call | vcomp's own default (workers spin) | `PASSIVE` (the library's default from 2026-10-02) |
|---|---|---|
| Engle-Granger, 8–64 pairs of 252 bars | 2.9–6.7× / 1.5–6.3× | 1.5–3.2× / 1.4–2.4× |
| `rolling_hurst`, n = 300–400, window 100 | 5.3–5.4× / 2.5–3.5× | 2.3–2.9× / 2.3–2.5× |
| `simulate_forward_paths`, 1,000 paths × 21 days (and terminal) | 7.6–7.8× / 3.0–3.2× | 2.7–3.2× / 2.5–2.6× |
| Five-indicator panel, 16 tickers × 252 | 4.5× / 1.9× | 2.2× / 1.7× |
| Spearman IC, 256 dates × 50 | 3.7× / 2.7× | 3.4× / 1.9× |
| `label_uniqueness`, 64 entities | 2.5× / 1.5× | 1.9× / 1.6× |
| `fit_preprocess_stats`, 252 × 128 | 6.3× / 2.9× | 3.6× / 2.2× |

Implied volatility, greeks and preprocessing keep the parallel path they had
at these sizes: a 476-contract chain, a 4,096-valuation greek grid and a
10,000 × 10 `apply_preprocess_stats` take the same path as before. At a
250 µs threshold those three would have run on one thread, 1.3–2.8×
slower. Just above the threshold a cold call can still lose a little: a
252-day Monte Carlo of 132–165 paths measured 0.84–0.92× cold (about
0.1 ms) and 1.4–4.6× warm, and a 127-pair `batch_backtest_crossover`
0.81–0.83× cold under `PASSIVE` and 1.4–5.6× warm.

**`SQT_OMP_MIN_WORK`, when set, is the old unit rule exactly**: a region
goes parallel at tasks × units per task ≥ that many units, and the costs
are ignored. `50000` was the old default. Unset, unparsable or negative
means the time rule. `SQT_NUM_THREADS=1` keeps every region serial, as
before.

**Pooled rank correlations run at once share the cores.** The one region
that divides its threads among concurrent calls is the parallel sort behind
a pooled Spearman correlation (above 50,000 rows): each call inside a
parallel region is counted while it runs, and the sort takes an equal share
of `SQT_NUM_THREADS` (or OpenMP's default) among them. From 8 Python threads
it is 1.2–1.6× faster and uses 1.3–1.7× less CPU. Every other region takes
the configured thread count, as before: shared the same way, the per-date
and per-pair loops measured up to 17% slower, because a share fixed when a
region starts leaves cores idle as the calls around it finish. A call alone
gets exactly what it always did.

Two smaller changes ride along. `permutation_null_ic` hands out its draws
guided rather than in equal static blocks: 1.1–1.8× faster at 12–16
threads, whose cores are not all the same speed. And
`apply_preprocess_stats` tests for NaN inline: MSVC compiled `std::isnan`
to a call into the C runtime DLL for every value, and the inline test is
1.3–1.5× faster on one thread.

None of this changes a result. Every parallel kernel's output is
bit-identical to the previous build at one to sixteen threads, under each
threshold setting and wait policy, and from concurrent callers
(`tests/cpp_bindings/test_thread_count_determinism.py`).

---
## Python-Level Optimisations

Confirmed benchmarks on a 2 000-bar series (Python 3.12, NumPy 2.4):

| Optimisation | Before | After | Speedup | Notes |
|---|---|---|---|---|
| ATR true range | 2.8 ms (`pd.concat` + `.max`) | 0.49 ms (`np.maximum`) | **5.6×** | Single-pass; eliminates 3 Series + concat |
| Trade log serialization | 31 ms (`iterrows`, 500 trades) | 3.6 ms (`to_dict`) | **~9×** | Vectorized dict conversion |
| CVaR computation | 0.83 ms (two-pass) | 0.44 ms (one-pass) | **1.9×** | Single `np.percentile` + boolean mask |
| SPY beta screen | N HTTP requests | 1 request per worker | **~N/workers×** | SPY pre-fetched once per batch — 1 total for single-process runs, once per worker for `n_workers > 1` |
| Backtesting equity curve | — | NumPy cumprod | vectorized | `(1 + returns).cumprod` |
| Portfolio covariance | — | BLAS `pandas.cov` | BLAS-backed | O(n·k²) via LAPACK |
| Screener (50+ tickers) | — | ProcessPoolExecutor | multi-core | Auto async→multiprocess threshold |
| Portfolio simulation (100 tickers × 2 000 bars, monthly) | 1 503 ms (per-ticker `.loc`) | 32 ms (dense matrices) | **47×** | 200 000 pandas label lookups replaced by positional indexing; 500 tickers → **78×** |
| Cross-sectional IC (252 dates × 50 entities) | 91.5 ms (`groupby` + `Series.corr` per date) | 1.26 ms (array passes) | **72.6×** | Was **72%** of a ridge walk-forward run. Balanced panels reshape to `(n_dates, n_entities)`; ragged ones use `np.add.reduceat` over segment bounds. Agreement with the per-date version is 2.2e-16 (spearman) / 5.0e-16 (pearson), including ties and NaN. The multiple shrinks to 1.8× at 2 000 entities as the per-date overhead amortizes. |
| Walk-forward fold masks | `panel["date"].isin(...)` per fold | one `searchsorted` + a per-date gather | — | Also keeps working for splitters whose folds are not contiguous, which purged K-fold needs. |
| Monte Carlo equity bands (`simulate_forward_paths`, 200 000 paths × 60 days) | 796 ms (three `np.percentile` calls along the strided axis) | 272 ms (one call on a transposed copy) | **2.9–3.2×** | 1.9–2.4× at 20 000 × 252, 1.7–1.9× at the default 1 000 paths; the terminal-only variant 1.1–1.4×. The terminal-return 5th percentile is computed once for both VaR and the CVaR threshold. Bit-identical: a band that comes out zero or NaN is recomputed as its own call (note below). |
| Lead-lag p-values (126 names, `max_lag=10`, `min_correlation=0.02`; 99 818 pairs) | 1.3 s (`f_sf` per pair) | 0.10–0.13 s (`f_sf_array`) | **10–13×** | The whole `lead_lag_matrix` call 4.4–5.8×, 1.3–1.6× at the default 0.1 floor. `f_sf`'s double for every input on NumPy 2.0 and 2.4: `math.log`/`math.exp` per element rather than numpy's SIMD versions, and batches under 128 stay on the scalar. |
| Winsorize step fit (2 000 × 200) | 99 ms (two `Series.quantile` per column) | 6.6 ms (one `DataFrame.quantile`) | **15–19×** | 3.3× at 20 000 × 50, 41–56× at 500 × 500. Bit-identical on pandas 2.3 and 3.0. Frames with repeated labels or a non-float64 column keep the per-column calls; the default winsorize + zscore pair runs the fused native path and is unaffected. |

> **Portfolio simulator note:** `run_portfolio_simulation` holds prices, target weights and liquidity baselines as dense `(n_bars × n_tickers)` matrices and executes the default cost configuration as array arithmetic. The vectorized rebalance is deliberately narrow — `per_share` commission, the impact model and the ADV constraint each need a per-element decision (a per-order minimum, a per-ticker volatility lookup, an error naming one ticker) and keep the explicit loop, selected automatically by cost model. Both routes are held to the same numbers by tests: agreement with the pre-vectorization implementation is within 1.7e-15 relative across every configuration, with `rebalance_log` identical, the residual being pairwise-vs-sequential summation rather than a different formula. The speedup grows with universe size because the removed cost scaled with tickers × bars. See [Documentation/04_backtesting.md](04_backtesting.md).

> **Cross-sectional IC note:** the centered two-pass correlation form is not
> a refinement. The textbook `n·Σxy − ΣxΣy` shortcut differences two nearly
> equal large numbers on return-scale data and loses most of its significant
> digits; switching to the centered form moved pearson agreement from 2.2e-14
> to 5.0e-16, and a test pins the tighter tolerance so it cannot drift back.
> The same trap caught the native preprocessing kernel from the other
> direction: pandas sums *pairwise* via numpy, and a sequential accumulator
> disagreed in the 12th significant digit until the kernel was changed to
> match. Both are recorded in the CHANGELOG.

> **One call, several percentiles, and the sign of zero:** `np.percentile`
> with several percentiles partitions once at every position any of them
> needs, which puts the same value at each position as one call per
> percentile. But −0.0 and +0.0, or two NaNs, can land in a different
> order, and the interpolated result then differs in its bits — only ever
> where it is zero or NaN. The Monte Carlo bands and the winsorize bounds
> therefore recompute exactly those results as their own call. Without
> that, the single call disagreed in the sign of a zero on hundreds of
> planted inputs (304 band values across 109 of 120 cases), on NumPy 2.0
> and 2.4 and on pandas 2.3 and 3.0. The kernel reaches −0.0 honestly: a −1
> return zeroes a path, and a later return below −1 flips its sign. A thread
> pool over the band columns measured 5–6× at 200 000 paths but 0.98× at the
> default, and `simulate_forward_paths` has no parallelism budget for it to
> respect, so it is not used.

> **Numba note:** RSI, ADX, Parabolic SAR, GARCH's variance recursion, the Kalman filter, and every backtest-strategy state machine (RSI/Bollinger/Donchian/VWAP-reversion) are decorated with `@njit`. This requires Numba with a compatible NumPy version (≤ 2.0, or wherever Numba's own ABI support currently ends). On an incompatible NumPy version, Numba decorators are a no-op and the code falls back to interpreted Python, where C++ genuinely wins big (the original ~10–30× estimates for RSI/ADX/PSAR describe this scenario). On a machine where Numba *is* working (like the one that produced the measured table below), it's already close to C speed once warm — real measurement shows C++ landing anywhere from a tie to a modest win against it, not a blowout. What C++ reliably wins either way: no per-process JIT compile tax (measured at ~200ms–1.1s on the first call in a fresh process, gone entirely with C++) and no numpy-ABI fragility risk (the exact failure mode that motivated porting RSI/ADX/PSAR to C++ in the first place). Every one of these falls back to pure Python automatically when neither C++ nor Numba is available.

---
