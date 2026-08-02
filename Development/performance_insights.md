# C++ Porting Performance Insights

**Date:** 2026-06-06  
**Scope:** `standard_quant_tools` — analysis of which components benefit from a C++/pybind11 rewrite and by how much.

---

## Executive Summary

Of the ~20 distinct computational modules in this library, several components account for nearly all theoretical speedup. The rest are already running at near-C speed via NumPy/BLAS/Cython and porting them would be wasted effort. **As of 2026-07-02, all originally identified Tier 1 features have been implemented in C++, plus 5 additional functions:** Hurst Exponent (DFA + R/S + rolling), RSI, ADX, Parabolic SAR, Wilder's ATR, Engle-Granger Cointegration, 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`), backtest kernel (`run_strategy`), **batch backtest grid kernel** (`batch_run_strategy`, array-based return: ~1.2× at the binding, ~7× in Python-side `DataFrame` construction, ~1.0× end-to-end at 1,200 combos where kernel compute dominates — see item 6), **rolling factor loadings** (incremental Cholesky, 50–200×), **rolling beta** (incremental sums, 10–40×), **Bollinger Bands** (fused mean+std, 3–8×), and **Stochastic Oscillator** (fused min+max, 5–15×). The most dramatic realised gain — 30–100× — is on `rolling_hurst`.

**Update (this pass):** four modules added after 2026-07-24 (GARCH(1,1) volatility forecasting, Kalman-filter dynamic hedge ratio, Monte Carlo simulation, and 4 new backtest strategies) were evaluated and, where genuinely worth it, ported — see items 11–14 below. Unlike every prior port, most of this new code was already `@njit`-decorated (numba), and numba was directly benchmarked as functional on the current dev machine (numpy 2.0.2, not the numpy 2.4 that broke numba and originally motivated porting RSI/ADX/PSAR) — so the case for these four isn't "numba is broken," it's the same *permanent* argument already made for RSI/ADX/PSAR: eliminating numba's JIT cold-start latency on the first call in a fresh process (measured at 200ms–1.1s, not the earlier ~300–500ms estimate — see below), and immunity to future numpy ABI breakage. `simulate_forward_paths` (Monte Carlo) is the one exception — it had **zero** acceleration applied before this pass (not even numba) and is also embarrassingly parallel, so it got an additional OpenMP-parallelized path on top of the usual compiled-vs-interpreted gain. Three candidates were investigated and explicitly **not** ported: `implied_volatility` (tiny loop, no batch entry point), `risk_parity_weights` (tiny `n_assets`, negligible absolute work), and EVT tail risk / `mean_variance_optimize` / `black_litterman` / `momentum_timeseries` / `adx_trend` (already closed-form, vectorized, or scipy-delegated — no loop to port).

**Update (real build + benchmark pass):** every number in this document up to this point was a *projected* estimate — no C++ toolchain was available in the environment that wrote them. That changed: a Windows SDK gap (missing `rc.exe`/`mt.exe`, `cl.exe` itself was already present) was found and fixed, `_sqt_core` was built for the first time with `cmake -B build -G Ninja -DSQT_BUILD_TESTS=ON` + MSVC 19.44, and every C++ test suite (110 native `ctest` cases across 7 executables, plus the full `tests/test_cpp_*.py` integration suite) was actually run. Two things came out of this:

1. **5 real, previously-undetectable bugs were found and fixed** — none of them visible from code review alone:
   - `simulate_forward_paths`'s pybind11 binding didn't raise for `horizon_days<=0`/`n_simulations<=0` — the "did the C++ function return the expected size" check degenerated to `0 == 0` for exactly these invalid inputs, silently passing them through. Fixed with an explicit upfront check.
   - `adf_test` (cointegration ADF/Engle-Granger) returned `NaN` for a degenerate, (near-)perfectly-collinear input (every regressor has zero variance, so the per-lag OLS solve is singular for every lag candidate) instead of matching statsmodels' own convention for this exact case (`adf_statistic=-inf, p_value≈0`, i.e. "maximal evidence of stationarity," verified empirically against statsmodels). Fixed with an upfront degenerate-input check in `adf_test`.
   - `ar1_halflife` (used for `CointResult.half_life`) returned `NaN` instead of `+inf` for a zero-variance lagged predictor, because `beta >= 0.0` is `false` for `NaN` under IEEE 754 — the exact same "not mean-reverting" case a non-negative beta already gets was falling through a different comparison path. Fixed by testing `!(beta < 0.0)` instead.
   - **4 of `tests/cpp/test_backtest.cpp`'s own hand-written trade-log expectations were wrong** — written without ever compiling or running them, based on a mistaken assumption that a trade's entry/exit price is `prices[i]` (the event bar's own price) rather than the actual, documented, and *correct* `prices[i-1]` (one-bar-lagged reference price) convention `run_strategy`'s return calculation already uses. The 2026-07-24 native trade-log fix (`backtest.cpp`, commit `2242d63`) that this whole document has been describing as "awaiting CI verification" turned out to be **already correct** — verified here by hand against the real `_build_trade_log` Python reference — it was the test file's own numbers that needed fixing, not the implementation. See Item 5/6 below for what this resolves.
   - A native/Python trade-stats parity test (`tests/test_backtest.py::TestNativeTradeStatsCorrectness`) used a tolerance (`abs=1e-9`) tight enough to fail on Python's own intentional `round(..., 4)` display rounding — not a real discrepancy, since `run_strategy()`'s Python wrapper always overwrites these fields with the rounded Python values regardless of backend anyway. Loosened to `abs=5e-5`.
2. **Every documented speedup figure below was a projection, and several turned out to be meaningfully wrong once measured** — most notably RSI/ADX/PSAR/GARCH/Kalman/signal-state-machines showing close to *1×* against warm numba (not 10–30× as originally estimated), because those old estimates implicitly assumed numba was broken (true in whatever environment first wrote this doc) rather than working (true on this machine). See the "Real benchmark results" section below Item 14 for the full, re-measured numbers and what they actually mean.

---

## Implementation Status

| # | Feature | Status | Realized Speedup |
|---|---|---|---|
| 1 | Hurst Exponent — DFA, R/S, `rolling_hurst` | ✅ IMPLEMENTED | Measured: 83× (DFA n=500), 131× (DFA n=2000), 274× (rolling, n=2000/window=200) — projections (20–80×/30–100×) held up or were beaten |
| 2 | RSI, ADX, Parabolic SAR | ✅ IMPLEMENTED, ADX rewritten to O(1) memory | Measured vs. *warm numba* (functional on this machine): RSI 5.3×, ADX 0.9× (tied) at n=2000 (see item 2b below for the O(1)-memory rewrite's own effect), PSAR 1.1× (tied) — see "Real benchmark results" below for why this differs sharply from the original 10–30× projection |
| 2b | ADX rewritten from 4 full n-sized temp arrays to O(1) auxiliary memory (`dm_plus`/`dm_minus`/`tr`/`dx_vals` eliminated) | ✅ IMPLEMENTED | Bit-identical output (pinned exact-equality regression test, verified against the pre-rewrite implementation via git stash both ways). Speed: negligible at n=2000 (~1.02–1.07×, within noise — Python/pybind call overhead dominates at this size), real **~1.21×** at n=50000 (min 3.18ms→2.63ms) where the eliminated arrays are large enough (~1.6MB total) for memory bandwidth/allocation cost to actually matter. Memory: 5 allocations → 1 regardless of n — a correct, worthwhile change even where the speed win doesn't show up. |
| 2b | Wilder's ATR (`wilder_atr`) | ✅ IMPLEMENTED | Measured: 28× vs. warm numba (beat the 4–8× projection) |
| 3 | Engle-Granger Cointegration (OLS + ADF + MacKinnon 2010) | ✅ IMPLEMENTED | Measured: 24× vs statsmodels (within the 5–15× projection's ballpark, on the high side) |
| 4 | 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`) | ✅ IMPLEMENTED | Measured: 1.4× (`calculate_beta`), 1.1× (`half_life`) vs. `lstsq` — well under the 10–20× projection; `lstsq` on a 2-var system is apparently not as slow in practice as the LAPACK-overhead argument suggested |
| 5 | Backtest kernel (`run_strategy` — equity + all metrics in one C++ pass) | ✅ IMPLEMENTED, wrapper redundancy fixed | Measured end-to-end (n=2000): **~58×** (26.8ms → 0.46ms) after removing wrapper-side pandas work the C++ path never needed (redundant `pct_change`/`shift`, and an unconditional Python trade-log rebuild that overwrote already-correct native stats) — see "Real benchmark results" below |
| 6 | Batch backtest grid (`batch_run_strategy` — all combos in one C++ call, array-based return) | ✅ IMPLEMENTED AND VERIFIED — win_rate/profit_factor/num_trades/avg_trade_return_pct previously used the native kernel's own uncorrected trade-log accounting; `backtest.cpp`'s native trade-log logic was rewritten 2026-07-24 to match `_build_trade_log` exactly, closing the divergence at the source. **Verified correct** by actually building `_sqt_core` and running `TestNativeTradeStatsCorrectness` plus the full native `ctest` suite — the fix was already right; 4 of the *test file's own* hand-written expectations were wrong instead (fixed, see the Executive Summary's bug list). Binding changed from `py::list[py::dict]` to a single `(num_tests, 11)` `py::array_t<double>` (item 6, below) | Binding call itself: **~1.21×**. Python-side `DataFrame`-construction step alone: **~7×**. End-to-end at a 1,200-combo grid (n=1,500 bars): within noise either way (~0.26s) — kernel compute dominates wall time at this scale, see item 6 discussion below |
| 7 | Rolling factor loadings (`rolling_factor_loadings` — incremental Cholesky) | ✅ IMPLEMENTED | Measured: 26× (n=500, window=60, k=3) vs per-window `lstsq` — under the 50–200× projection, still large |
| 8 | Rolling beta (`rolling_beta` — incremental sum updates) | ✅ IMPLEMENTED | Measured: 4.7× (n=2000, window=60) vs 2× pandas rolling — under the 10–40× projection |
| 9 | Bollinger Bands (`bollinger_bands` — fused Σx / Σx² pass) | ✅ IMPLEMENTED | Measured: 1.6× (n=2000) vs 2× pandas rolling — well under the 3–8× projection; pandas' own rolling ops are apparently fast enough that the fused pass doesn't win by much |
| 10 | Stochastic Oscillator (`stochastic_oscillator` — fused min+max pass) | ✅ IMPLEMENTED | Measured: 2.6× (n=2000) vs 2× pandas rolling — under the 5–15× projection |
| 11 | Monte Carlo forward simulation (`simulate_forward_paths` — moving-block bootstrap, optional OpenMP), per-path allocation eliminated | ✅ IMPLEMENTED | Measured: 2.0× serial (n_simulations=5000, horizon=60) vs. the previously-uncompiled Python loop. Per-path `std::mt19937_64` construction + `resampled` heap buffer removed (hoisted to one thread-local RNG, values written directly into the output row) — real but modest gain on this benchmark shape: 1-thread min 284.5ms→239.1ms (**~1.19×**), unconstrained min 117.4ms→113.7ms (**~1.03×**) at n_simulations=200000. The per-path allocation being eliminated was small (~480 bytes), so a modern thread-caching allocator was apparently already handling it reasonably well — see "Real benchmark results" below for the honest before/after. |
| 12 | GARCH(1,1) variance recursion (`garch11_variance_recursion`) | ✅ IMPLEMENTED | Measured: 219ms→4.8ms first-call latency in a fresh process (not the ~300–500ms estimate); steady-state (warm numba) is actually 0.8× — i.e. C++ is slightly *slower* once numba is JIT-compiled. The entire value of *this specific kernel* is the cold-start column — see item 12b below for the actual fit-level win. |
| 12b | GARCH fused NLL (`garch11_neg_loglik`/`garch11_neg_loglik_grad`) — the whole `scipy.optimize` objective, not just the recursion | ✅ IMPLEMENTED | Measured end-to-end `garch_volatility_forecast()` (n=1000): **~7.8×** (7.928ms → 1.016ms). Unlike item 12 (recursion only, ~0.8× vs warm numba), this fuses the recursion *and* the NLL reduction into one native call (no per-iteration sigma2 array round-trip) plus an analytic gradient wired via `jac=True` (verified against central differences before being trusted — see "Real benchmark results" below) so L-BFGS-B stops needing 6 extra finite-difference NLL evaluations per iteration for a 3-parameter gradient. This is the review's own point: porting the same recursion loop nets ~1×, but fusing the *work the Python wrapper was doing around it* is where the real win is. |
| 13 | Kalman filter, 1-state and 2-state (`kalman_filter_1state`/`kalman_filter_2state`) | ✅ IMPLEMENTED | Not independently re-measured this pass; same cold-start/ABI-permanence rationale and expected profile as item 12 |
| 14 | Donchian breakout / VWAP-reversion signal hysteresis (`donchian_state_machine`/`vwap_reversion_state_machine`) | ✅ IMPLEMENTED | Not independently re-measured this pass; measured ADX/PSAR cold-start (1109–1110ms→1.2ms) is the closest available proxy, same numba-state-machine shape |
| 15 | Fused `technical_indicators()` — RSI/ADX/ATR/Bollinger/Stochastic in one native call | ✅ IMPLEMENTED, wired into `agent/tools.py` as an additive fast path | Measured at the integration point (`get_technical_analysis`, n=2000, 4 fusable indicators requested): **~4.6×** (1,467µs → 314µs). At the raw C++-binding level alone (no Python-wrapper overhead in the comparison): **~1.0×** (n=2000, ~100µs either way) — see item 6 discussion below for why these differ so much. |
| 22 | Deep native optimization, item K: opt-in local-only PGO build workflow (`SQT_PGO_GENERATE`/`SQT_PGO_USE`, documented 2-step process, not wired into CI) | ✅ IMPLEMENTED (infrastructure only — not run end-to-end to produce a trained profile this pass) | Default-OFF path confirmed unaffected (full ctest+pytest green). `SQT_PGO_GENERATE=ON` confirmed to actually configure and build on this project's MSVC toolchain. No before/after speed number recorded yet — that requires running the full 3-step local workflow with a real training run, left for whoever next wants the extra local speed |
| — | Deep native optimization, item J: rank-1 Cholesky update/downdate for `rolling_factor_loadings` | ❌ ATTEMPTED, NOT SHIPPED — hard numerical-stability gate failed | Implemented, gated against the old full-refactor-per-step path (real git-stash before/after, 8 configs spanning k=3–50). Well-conditioned data: ~1e-13–3e-10 relative agreement (excellent). Near-singular/collinear data: **~30× and ~1.2× relative difference** (real breakdown). Large-baseline data: **~5.3%**. Reverted per the documented escape hatch — see CHANGELOG's "Not shipped" entry for the full writeup. |
| 21 | Deep native optimization, Phase 5: `SQT_RESTRICT` portable qualifier across all 12 `_into` kernels (aliasing audited at every `bindings.cpp` call site first) | ✅ IMPLEMENTED | Measured honestly: **no measurable difference** on this MSVC build (`rsi`/`adx`/`rolling_factor_loadings`/`run_strategy`, n=2000) — consistent with MSVC extracting less benefit from `__restrict` than GCC/Clang; kept as a correctness-neutral hint |
| 20 | Deep native optimization, Phase 4: one-pass DFA reformulation (`dfa_onepass`, exact OLS sufficient-statistics identities, hard tolerance gate) | ✅ IMPLEMENTED — gate passed cleanly, wired into `hurst_exponent_scratch`'s "dfa" branch; public `dfa()` untouched | Measured (min of 7, isolating this item on top of Phase 3b): **n=1000: ~1.15×** (0.90ms→0.78ms); **n=2000: ~1.85×** (2.68ms→1.45ms); **n=5000: ~1.82×** (6.92ms→3.81ms). Combined with Phase 3b vs. the original fully-serial baseline: **~5.2×, ~10.5×, ~10.7×** |
| 19 | Deep native optimization, Phase 3b: OpenMP across `rolling_hurst`'s window loop + scratch-buffer reuse (`dfa_impl` split, `hurst_exponent_scratch`, counted-index loop rewrite) | ✅ IMPLEMENTED, verified against the unchanged public `hurst_exponent()` on the same window slice + exact reproducibility across `OMP_NUM_THREADS=1/2/4/8` | Measured (min of 7 runs, 16 cores): **n=1000/window=100: ~4.5×** (4.05ms→0.90ms); **n=2000/window=200: ~5.7×** (15.28ms→2.68ms); **n=5000/window=200: ~5.9×** (40.77ms→6.92ms) |
| 18 | Deep native optimization, Phase 3: LTO/IPO enabled automatically for Release builds (`CheckIPOSupported`, no opt-in flag needed — link-time only, no cross-CPU portability risk) | ✅ IMPLEMENTED | Measured honestly: **no measurable runtime difference** (~1.0× across `rsi`/`adx`/`rolling_factor_loadings`/`run_strategy`, n=2000) and **no measurable build-time cost** on this small (9-source-file) extension — kept anyway as a free, correctness-neutral toolchain improvement, not because it measured as a win here |
| 17 | Deep native optimization, Phase 2: `run_strategy_summary()` (zero-allocation two-pass metrics kernel) + OpenMP across `batch_run_strategy`'s test-index loop | ✅ IMPLEMENTED, bit-identical verified (40 random trials + edge cases, exact reproducibility across `OMP_NUM_THREADS=1/2/4/8`) | Measured (min of 7 runs, 16 logical cores): **n=500/num_tests=500: ~6.0×** (3.26ms→0.54ms); **n=2000/num_tests=2000: ~11.3×** (51.55ms→4.55ms); **n=2000/num_tests=10000: ~8.6×** (255.25ms→29.81ms) |
| 16 | Deep native optimization, Phase 1 (`rolling_regression.cpp`): dead upper-triangle elimination in the normal-equations build + rank-1 update, `cholesky_solve` scratch-buffer reuse (no per-bar `L`/`z` allocation), `omp simd` reduction hint on `rolling_beta`'s reduction loop (non-MSVC only — MSVC's OpenMP 2.0 hard-errors on `omp simd`, not a silent no-op) | ✅ IMPLEMENTED, bit-identical verified two ways (git-stash exact comparison + new independent Gaussian-elimination reference test) | Measured (`rolling_factor_loadings`, n=2000, window=60, min of 9 runs): **k=3 (this library's own typical factor count): 0.269ms → 0.150ms, ~1.79×** — allocator overhead dominated at this small problem size, more than the O(p²)/O(p³) math itself; **k=10: ~1.30×**; **k=30: ~1.09×** — the win shrinks as p grows since O(p³) Cholesky decomposition increasingly dominates total cost, the opposite of what "matters more at large k" framing would suggest for *this specific pair* of optimizations (the still-pending rank-1 Cholesky *factor* update, item J, is where the large-k payoff is expected to actually show up) |

The compiled extension is `_sqt_core.pyd` (Windows). All Python modules fall back to pure Python if the extension is absent, preserving the library's optional-dependency philosophy.

---

## Real Benchmark Results (measured, not projected)

Methodology: each row toggles the relevant module's own `HAS_CPP` flag and times both paths back-to-back in the same process (warmup + N reps, `time.perf_counter()`), so it's a genuine apples-to-apples comparison — not separately-run numbers that could be skewed by machine load, thermal throttling, or a cold OS file cache. Machine: Windows 11, MSVC 19.44.35228.0, Python 3.12.1, numpy 2.0.2, 16 logical cores. Build: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSQT_BUILD_TESTS=ON`.

| Operation | vs. warm numba / fallback | Notes |
|---|---|---|
| `hurst_exponent` DFA (n=500) | **83×** (4.57ms → 0.05ms) | No numba path for Hurst — this is the real comparison either way. |
| `hurst_exponent` DFA (n=2000) | **131×** (12.3ms → 0.09ms) | |
| `rolling_hurst` (n=2000, window=200, step=1) | **274×** (4.64s → 17ms) | |
| `rsi` (n=2000) | **5.3×** (0.47ms → 0.09ms) | |
| `adx` (n=2000) | **0.9×** (essentially tied) | Warm numba is already about as fast as C++ here. |
| `parabolic_sar` (n=2000) | **1.1×** (essentially tied) | |
| `wilder_atr` (n=2000) | **28×** (4.40ms → 0.15ms) | |
| `bollinger_bands` (n=2000) | **1.6×** | |
| `stochastic_oscillator` (n=2000) | **2.6×** | |
| `cointegration_test` (n=500, vs. statsmodels) | **24×** (29.7ms → 1.24ms) | statsmodels has no JIT path, so this is a clean, unambiguous win. |
| `calculate_beta` (n=500, vs. `lstsq`) | **1.4×** | |
| `half_life` (n=500, vs. `lstsq`) | **1.1×** | |
| `run_strategy` (n=2000, `include_trade_log=False`) | **~58×** (26.8ms → 0.46ms) | Wrapper redundancy fixed (see discussion below) — was ~1.0× (essentially no measured benefit) before this pass. |
| `run_strategy` (n=2000, `include_trade_log=True`) | n/a — still builds the Python trade log | 26.4ms — the trade-log DataFrame itself, not wrapper waste, now dominates when a caller actually asks for it; this is real, requested work, not overhead to eliminate. |
| `rolling_beta` (n=2000, window=60) | **4.7×** | |
| `rsi` (n=100, direct-write binding) | **~1.6×** (0.00429ms → 0.00262ms) | Boundary-copy elimination (item 5, below) — matters proportionally more at small n where the copy is a larger fraction of total call time. |
| `adx` (n=100, direct-write binding) | **~1.9×** (0.00886ms → 0.00477ms) | Same. |
| `rolling_factor_loadings` (n=500, window=60, k=3) | **26×** (32.9ms → 1.27ms) | |
| `simulate_forward_paths` (n_simulations=5000, horizon=60) | **2.0×** (74.8ms → 37.7ms) | No numba path ever existed — pure uncompiled Python before this pass. |
| `garch11_variance_recursion` (n=2000, warm) | **0.8×** (10.8ms → 12.9ms, C++ slightly slower) | The cold-start column below is the entire point of *this specific kernel*. |
| `garch_volatility_forecast` end-to-end fit (n=1000) | **~7.8×** (7.928ms → 1.016ms) | Fused NLL (`garch11_neg_loglik`) + analytic gradient (`garch11_neg_loglik_grad`, `jac=True`) — see discussion below. Same-machine before/after via git stash/pop, not a projection. |
| `batch_run_strategy` binding call (array vs list-of-dict return) | **~1.21×** | Isolated at the C++/pybind11 boundary only. |
| `batch_run_strategy` → `DataFrame` construction (Python side only) | **~7×** | Array→`pd.DataFrame(arr, columns=...)` vs `num_tests` dicts→`pd.DataFrame(rows)`. |
| `backtest_grid` end-to-end (1,200 combos, n=1,500 bars, `sma_crossover`) | **~1.0×** (0.262s → 0.262s, min-of-5) | Kernel compute (1,200 full backtests) dominates wall time at this scale — the marshaling win above is real but a small fraction of the total here. Same-machine git-stash/pop before/after. |
| `get_technical_analysis` (n=2000, rsi+adx+bollinger+stochastic requested) | **~4.6×** (1,467µs → 314µs, median of 9) | Fused `technical_indicators()` fast path vs 4 separate Python-wrapper calls — see discussion below. |
| `technical_indicators` vs 4 individual bindings, raw C++ boundary only (n=2000) | **~1.0×** (~100µs either way) | No Python-wrapper overhead in this comparison — isolates that the win is in the glue, not the kernels. |
| `rolling_factor_loadings` (n=2000, window=60, k=3, min of 9) | **~1.79×** (0.269ms → 0.150ms) | Deep native optimization Phase 1 (dead upper-triangle removal + `cholesky_solve` scratch reuse) — see discussion below. |
| `rolling_factor_loadings` (n=2000, window=60, k=10, min of 9) | **~1.30×** (1.058ms → 0.811ms) | Same change. |
| `rolling_factor_loadings` (n=2000, window=60, k=30, min of 9) | **~1.09×** (7.452ms → 6.833ms, best of 2 runs) | Same change — smaller win at large k since `cholesky_solve`'s O(p³) decomposition increasingly dominates total cost. |
| `batch_run_strategy` (n=500, num_tests=500, min of 7) | **~6.0×** (3.26ms → 0.54ms) | `run_strategy_summary` (zero allocation) + OpenMP across test indices, 16 logical cores. |
| `batch_run_strategy` (n=2000, num_tests=2000, min of 7) | **~11.3×** (51.55ms → 4.55ms) | Same change. |
| `batch_run_strategy` (n=2000, num_tests=10000, min of 7) | **~8.6×** (255.25ms → 29.81ms) | Same change. |
| `rolling_hurst` (n=1000, window=100, min of 7) | **~4.5×** (4.05ms → 0.90ms) | OpenMP + scratch-buffer reuse, 16 logical cores. |
| `rolling_hurst` (n=2000, window=200, min of 7) | **~5.7×** (15.28ms → 2.68ms) | Same change. |
| `rolling_hurst` (n=5000, window=200, min of 7) | **~5.9×** (40.77ms → 6.92ms) | Same change. |
| `rolling_hurst` (n=1000, window=100, one-pass DFA on top of Phase 3b) | **~1.15×** (0.90ms → 0.78ms) | `dfa_onepass` reformulation, tolerance-gated (gate passed). |
| `rolling_hurst` (n=2000, window=200, one-pass DFA on top of Phase 3b) | **~1.85×** (2.68ms → 1.45ms) | Same change. |
| `rolling_hurst` (n=5000, window=200, one-pass DFA on top of Phase 3b) | **~1.82×** (6.92ms → 3.81ms) | Same change. Combined with Phase 3b vs. original serial baseline: ~5.2×/10.5×/10.7× at these three sizes. |

Cold-start latency, measured via a genuinely fresh subprocess per number (nothing warmed up beforehand — this is the number that matters for a single one-off agent-tool call in a new process):

| Function | Numba JIT cold-start | C++ first call | Eliminated |
|---|---|---|---|
| `adx` | 1109.6ms | 1.21ms | ~1.1s |
| `garch_volatility_forecast` | 219.4ms | 4.83ms | ~215ms |

**One surprise from an earlier pass, resolved this pass — kept here rather than deleted, since the "why" is instructive:**

- **`run_strategy` used to measure ~1.0× end-to-end** (68.1ms → 67.7ms), not the documented 3–8×, even though `bench_backtest.cpp`'s native-only numbers (0.017ms at n=2000) confirmed the raw kernel was fast in isolation. The gap was never the kernel — it was the *wrapper*: `returns = prices.pct_change()`/`executed = signals.shift(1)` computed unconditionally before the C++ dispatch check (never used on that path — the kernel recomputes both internally), and an unconditional `_build_trade_log`/`_compute_trade_stats` call after the kernel returned, rebuilding the entire Python trade log purely to overwrite native `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` fields that this session's own CI verification work had already confirmed were correct (`TestNativeTradeStatsCorrectness`, run against a real compiled `_sqt_core` on live CI). Removing both — the redundant pandas calls, and the override (now built only when `include_trade_log=True` actually asks for the DataFrame) — took end-to-end wall time from 26.8ms to 0.46ms, a real ~58× measured on this same benchmark. The batch grid kernel (`batch_run_strategy`) never had this specific bug — its consumer already read native stats directly without a per-combo Python rebuild.
- **OpenMP's measured speedup for `simulate_forward_paths` is ~2.0–2.4×** on this 16-core machine at `n_simulations=200000` (min-of-7-runs methodology, separate process invocations with `OMP_NUM_THREADS` set before each — an in-process env-var toggle does *not* work, since the OpenMP runtime reads it once at thread-pool creation, not on every parallel region). Not the near-linear-with-cores scaling the per-path independence would suggest in theory, and MSVC's OpenMP support here is version 2.0 (an older spec) — some of that gap was expected going in.
  - **Per-path allocation elimination (this pass)**: hoisted `std::mt19937_64`/`std::uniform_int_distribution` construction to one instance per OpenMP thread (reseeded per path, not reconstructed) and removed the intermediate `resampled` heap buffer entirely — 200,000 fewer heap allocations/frees at this problem size, values now written directly into the output row as they're sampled. **Real but modest measured gain**: 1-thread min 284.5ms→239.1ms (~1.19×), unconstrained min 117.4ms→113.7ms (~1.03×) — the OpenMP scaling *ratio* itself barely moved (2.42×→2.10× before/after, both real measurements, well within the noise band of this benchmark). The per-path allocation being eliminated was small (~480 bytes for a 60-day horizon), so it evidently wasn't the dominant cost at this problem size — a legitimate, correct change (worth keeping: fewer allocations is never worse), just not the dramatic win the "no wonder 16 cores only get ~1.4×" framing suggested going in. Run-to-run noise on this benchmark is substantial (up to ~35% across 7 repeats of the same config in separate processes), which is why min-of-N (not a single measurement) is now the reported number.
- **ADX's O(1)-memory rewrite (`dm_plus`/`dm_minus`/`tr`/`dx_vals` arrays eliminated) shows its speed benefit only at scale, not at the n=2000 size this document benchmarks everything else at.** Traced the Wilder recursion by hand first: it only ever needs the immediately-previous smoothed sum plus the *current* bar's raw value, and the DX/ADX seed windows only need a running sum, not individual stored values — so the whole function genuinely reduces to O(1) auxiliary memory (a handful of scalars), not just "smaller." Rewrote as a single fused pass preserving the *exact same order* of floating-point operations as the original 4-pass version (floating-point addition isn't associative, so order matters, not just which values get summed) — verified bit-identical output both ways: existing tests passed unchanged (zero tolerance widening), and a new exact-equality regression pin (`tests/cpp/test_indicators.cpp`) was confirmed to match against *both* the old and new implementation via `git stash`. Measured speed: **negligible at n=2000** (~1.02–1.07×, within noise — fixed Python/pybind call overhead dominates a call this cheap), but a real **~1.21×** at n=50000 (min 3.18ms→2.63ms) once the eliminated arrays are large enough (~1.6MB total at that size) for memory bandwidth/allocation cost to actually show up against the O(n) arithmetic. The memory reduction itself (5 allocations → 1) is unconditional regardless of n — worth keeping even where the wall-clock difference doesn't register.
- **`garch_volatility_forecast` fused NLL is where the real GARCH win lives, not the recursion port itself.** `garch11_variance_recursion` alone measures 0.8× vs warm numba (item 12) — porting the same 2000-iteration loop was never going to beat already-JIT-compiled machine code. But every one of scipy's L-BFGS-B iterations was calling that recursion, copying a full `sigma2` array out of C++, then reducing it to one scalar in NumPy — paying a full array round-trip on every iteration purely to throw the array away. `garch11_neg_loglik` fuses the recursion and the NLL reduction into one native call returning a single `double` (no array ever crosses the boundary), and `garch11_neg_loglik_grad` additionally computes the analytic gradient in the same fused pass, wired via `jac=True` so L-BFGS-B stops needing 6 extra finite-difference NLL evaluations per iteration (2 per parameter, 3 parameters) to numerically estimate what the analytic formula now gives it directly. The analytic gradient was verified against central differences across 5 random `(resid_sq, omega, alpha, beta)` grids before being trusted (`tests/cpp/test_garch.cpp`) — the first attempt at this check used a single absolute step size for all three parameters and failed, not because the gradient was wrong, but because a step size appropriate for alpha/beta (~0.05–0.95) was a ~100%-of-magnitude perturbation for omega (~1e-6), dominated by the numerical reference's own truncation error; per-parameter-scaled step sizes fixed the check, and it now passes cleanly. End-to-end `garch_volatility_forecast()` measured **~7.8×** (7.928ms → 1.016ms, n=1000, same-machine git stash/pop before/after). One expected side effect: `jac=True` changes which gradient L-BFGS-B actually follows, so the C++ and numba/NumPy paths can now converge to a very slightly different point near a flat likelihood surface (real for GARCH persistence/omega) — `TestGarchForecastEndToEndParity` was loosened from `abs=1e-10` (bit-identical, true when both paths used the same finite-difference approach) to `rel=1e-2` on the fitted parameters plus a tight `rel=1e-3` check on the two fits' own log-likelihoods (the actual invariant that matters: both found a comparably good optimum, not that they took the same path there).

- **Direct-write bindings (item 5): eliminated the `std::vector<double> result = sqt::foo(...); py::array_t<double> out(...); std::copy(...)` pattern that ~16 of `bindings.cpp`'s ~21 bindings shared.** Added a buffer-writing `*_into` overload alongside each existing vector-returning `sqt::` function (13 of the ~16 identified — `rsi`, `adx`, `parabolic_sar`, `wilder_atr`, `bollinger_bands`, `stochastic_oscillator`, `rolling_hurst`, `rolling_beta`, `rolling_factor_loadings`, `simulate_forward_paths`, `garch11_variance_recursion`, `donchian_state_machine`, `vwap_reversion_state_machine`), so the binding allocates the NumPy output array *first* and the C++ kernel writes straight into its buffer — one allocation, zero copies, instead of a `std::vector` allocation plus a full-array copy into a second, separately-allocated NumPy array. **Deliberately scoped out**: `run_strategy`'s `equity_curve` field and the two Kalman filters' 3-4 output arrays each — these return multi-field structs (`BacktestResult`, `Kalman1StateResult`/`Kalman2StateResult`), not a single `std::vector`, so the same pattern would need multiple output-buffer parameters per call; lower value (Kalman filters aren't hot-loop calls, and `run_strategy`'s own copy is already dwarfed by item 1's ~58× wrapper fix) for real added complexity, so left as a known, documented gap rather than forced in. Native tests keep calling the unchanged vector-returning `sqt::` API throughout — zero test churn from this item, only `bindings.cpp` changed. Measured on two of the cheapest kernels at small n (where a copy is proportionally largest relative to total call time): `rsi` (n=100) **~1.6×** (0.00429ms→0.00262ms), `adx` (n=100) **~1.9×** (0.00886ms→0.00477ms) — real, same-machine git-stash-verified numbers, not a projection.

- **Item 6, part A (`batch_run_strategy` array return): the marshaling-layer win is real but scale-dependent.** Isolated at each layer, the win looks large — the binding call itself (array-return vs. building `num_tests` `py::dict` objects in C++) is **~1.21×**, and the pure Python-side `DataFrame`-construction step (`pd.DataFrame(arr, columns=...)` vs. `pd.DataFrame(rows)` from `num_tests` dicts) is **~7×**. But at an actual 1,200-combo `backtest_grid()` call (n=1,500 bars, the review's own "1,000+ combos" scale), the two measured within noise of each other (~0.262s either way, min-of-5, same-machine git stash/pop) — because at that grid size, the C++ kernel itself (1,200 full backtests, each simulating 1,500 bars) is the overwhelming majority of wall time, and the marshaling savings are a small fraction of a much bigger number. The lesson isn't that the change was wasted — it's a strict improvement with no downside (fewer allocations, less Python object churn, real wins in the isolated measurements above) — but its end-to-end visibility depends on how cheap the kernel work is relative to the number of combos and result columns; it will matter far more for a grid with many combos over a short series, or a cheaper per-combo computation, than for this particular benchmark shape.
- **Item 6, part B (`technical_indicators()`): another case (like items 2–4) where the review's real point was "fuse the surrounding Python glue," not "make the C++ loop faster.**" At the raw C++/pybind11 boundary, 4 individual bindings (`rsi`, `adx`, `bollinger_bands`, `stochastic_oscillator`) vs. 1 fused `technical_indicators()` call measure **~1.0×** (n=2000, ~100µs either way) — unsurprising, since at this size the pybind11 call overhead itself is negligible next to each kernel's own O(n) work, so cutting 4 calls to 1 barely moves the needle. But at the actual integration point — `agent/tools.py`'s `get_technical_analysis`, which was calling `rsi()`, `adx()`, `bollinger_bands()`, `stochastic_oscillator()` as four separate *Python* wrapper functions, each paying its own `validate_series` decorator, logging calls, `.to_numpy()` conversions, and `pd.Series`/`pd.DataFrame` construction — one fused native call plus one lightweight round of `pd.DataFrame` construction in `tools.py` itself measures **~4.6×** (1,467µs → 314µs, median of 9 runs, n=2000). The plain `atr` indicator was deliberately excluded from the fused fast path: the tool's `atr()` computes a simple rolling-mean ATR, while `technical_indicators()`'s ATR field is Wilder-smoothed (matching `wilder_atr()`) — a genuinely different algorithm, not just a faster route to the same numbers, so fusing it would have silently changed the tool's output. Verified the fused path produces byte-identical `last_values`/`signals` to the per-indicator fallback path (forced via a `HAS_CPP` monkeypatch) before trusting it.

- **Deep native optimization, Phase 1 (`rolling_regression.cpp`): the biggest win came from eliminating allocator overhead, not from the O(p²)/O(p³) math itself — and it showed up strongest at exactly this library's own typical problem size, not the large-k case the reviewing analysis emphasized.** `build_normal_equations()` and the rank-1 XtX update/downdate loop computed all p² entries of the symmetric normal-equations matrix even though `cholesky_solve()`'s decomposition loop only ever reads the lower triangle (`j <= i`) — removed the upper-triangle computation outright (no mirror step needed, since nothing downstream reads it), verified bit-identical via a same-machine `git stash`/`git stash pop` exact-equality comparison of `rolling_factor_loadings()`'s full output array plus a new from-scratch independent-reference test (dense Gaussian elimination on the full normal equations, sharing no code with the production path). Separately, `cholesky_solve()` allocated a fresh `L`/`z` vector on every single call — once per bar in the rolling window — now reusing caller-owned scratch buffers sized once outside the loop; traced the read pattern by hand first and confirmed the old zero-fill was never actually load-bearing (every read of `L` is to an entry the same call already wrote earlier in its own row-by-row iteration order), so the reused buffer needs no re-zeroing either. Measured at n=2000, window=60 (min of 9 runs): **k=3 (this library's own stated typical/tested factor count) 0.269ms → 0.150ms, ~1.79×** — allocator overhead dominated total cost at this small problem size, more than the actual linear-algebra work; **k=10: ~1.30×**; **k=30: ~1.09×**, the win *shrinking* as p grows since `cholesky_solve`'s O(p³) decomposition increasingly dominates total cost and the eliminated O(1)/O(p²) work becomes a smaller fraction of it. This is the inverse of what the original review's "matters more at k=10-50" framing suggested for these two specific changes — that framing turns out to describe the *third*, still-pending change in this file (rank-1 Cholesky *factor* update/downdate, replacing the O(p³) refactor itself, not just its allocation), not these two. Also added `#pragma omp simd reduction(...)` as a vectorization hint on `rolling_beta`'s reduction loop — the first attempt broke the MSVC build outright (`C7660`: MSVC's default `/openmp` only implements OpenMP 2.0, which doesn't recognize `omp simd`; that requires OpenMP 4.0+, only via `/openmp:experimental`), not the silently-ignored no-op initially assumed, so the pragma is now scoped to non-MSVC compilers only — its actual payoff on GCC/Clang is unmeasured on this Windows-only dev machine and will only be confirmed once this lands on the project's Linux CI runner.

- **Deep native optimization, Phase 2 (`backtest.cpp`): the largest single win of this whole pass, and it compounds two independent effects — removing per-call allocation and adding real parallelism across a genuinely independent workload.** `run_strategy_summary()` computes `run_strategy()`'s same 11 scalar fields using zero heap allocation at all, by exploiting a fact discovered during verification (not part of the original review): `strat_ret[i]` has no true loop-carried dependency — `exec_i = signals[i-1]` and the `prev_exec` value needed for `pos_diff` equals `signals[i-2]` (or 0.0 for `i==1`), both directly derivable from array indices with nothing carried across iterations. Only the trade-log open/close bookkeeping is a genuine sequential state machine. This enables a two-pass, allocation-free design: pass 1 fuses that state machine with running equity/peak/drawdown/mean tracking (trade stats as running scalars, no `trade_rets` vector); pass 2 recomputes `strat_ret[i]` on demand — now that the mean is known — to get variance and downside deviation, seeding the accumulator with index 0's implicit `strat_ret[0]=0.0` contribution directly (exploiting `0.0 + x == x` being exact in IEEE 754 to reproduce the original's exact accumulation order). Verified bit-identical against `run_strategy()` across 40 random trials plus edge cases (zero-price bars, leveraged/non-±1 signals, all-short, `n==0`/`n==1`) — this is a genuine from-scratch reimplementation, not a mechanical refactor, so the bit-identical claim needed its own dedicated proof, not an assumption. `batch_run_strategy` then runs every test index in parallel via a plain `#pragma omp parallel for` — simpler than `monte_carlo.cpp`'s nested `#pragma omp parallel { ... #pragma omp for ... }` form, since `run_strategy_summary` needs no per-thread state (no RNG, unlike Monte Carlo) — after switching `results` from `reserve()+push_back()` (not thread-safe across concurrent writers) to `resize()`+indexed writes. Verified exact reproducibility across `OMP_NUM_THREADS=1/2/4/8` — every row is fully independent, so unlike Monte Carlo's per-path-seed reproducibility guarantee, output here must be bit-identical regardless of thread count, not merely per-path-deterministic. Measured on this 16-logical-core machine (min of 7 runs): **~6.0× at n=500/500 tests, ~11.3× at n=2000/2000 tests, ~8.6× at n=2000/10000 tests** — the peak at the middle size likely reflects OpenMP scheduling overhead mattering more at very small per-thread workloads (500 tests) and per-call summary-kernel cost (now tiny) eventually limiting scaling at very high test counts (10,000) as more threads contend for memory bandwidth; not independently isolated into "allocation-elimination share" vs. "parallelism share" this pass, but both are real and this is their honest combined effect.

- **Deep native optimization, Phase 3 (LTO/IPO): a real, honest null result, kept anyway.** `CheckIPOSupported` + automatic `INTERPROCEDURAL_OPTIMIZATION_RELEASE` (not gated behind an opt-in flag, unlike `SQT_NATIVE_ARCH` — LTO carries no cross-CPU portability risk). Measured both build time (clean `_sqt_core`-only build, ~5.7-6.0s either way — noise-level, not a real difference) and runtime speed (`rsi`/`adx`/`rolling_factor_loadings`/`run_strategy` at n=2000, ~1.0× across the board) with the toggle actually flipped off and back on (not assumed). The likely reason: LTO's main value is inlining/constant-propagation *across* translation-unit boundaries, but every kernel in this codebase already has its hot loop living entirely inside one `.cpp` file, called through a single function boundary from `bindings.cpp` — there isn't much cross-TU work for LTO to find and fuse in this specific codebase structure, at this specific (small, 9-source-file) extension size. Kept in the build regardless, since it's free (no measured downside on either build time or runtime) and the review's own framing was "percentages, not multiples, low-effort" — this pass measured closer to "no percentage" than "some percentage," which is worth recording honestly rather than silently rounding a null result up to a categorical win.

**RSI/ADX/PSAR/GARCH/Kalman/signal-state-machines measuring close to 1× against warm numba is not a regression or a wasted port** — it's the expected outcome of the "why C++ wins over Numba (permanently)" argument this document already made for RSI/ADX/PSAR before ever being measured: the win was never claimed to be steady-state throughput on a machine where numba works, it's eliminating the JIT cold-start tax (confirmed above: ~200ms–1.1s per fresh process) and the numpy-ABI fragility that broke numba once already (real: this exact failure originally motivated the RSI/ADX/PSAR port). On a machine where numba is genuinely broken (e.g. an incompatible numpy version), the comparison would instead be C++ vs. an *interpreted* Python loop, which this benchmark pass didn't measure directly but which the original, pre-measurement 10–30×-style estimates describe reasonably well.

---

## Background: Why Some Code Is Already Fast

Before identifying what to port, it is important to understand what is already fast and why — so we do not waste effort.

**NumPy operations (vectorized):** Calls like `np.cumsum`, `np.maximum`, matrix multiply (`@`), `np.linalg.svd`, `np.linalg.lstsq` all drop into compiled BLAS/LAPACK routines. These run at or near theoretical peak for a single core. Rewriting them in C++ would touch the same underlying libraries and gain nothing.

**Pandas rolling/ewm:** `series.rolling(n).mean()`, `.std()`, `.ewm(span=n)` are Cython-compiled inside pandas. They are not Python loops. Rewriting them offers 1–2× at best, which does not justify the maintenance cost.

**I/O-bound code:** The screener and data provider are bottlenecked on HTTP round-trips to yfinance. Rewriting the computation in C++ does not help when 99% of wall time is waiting on the network.

---

## Components: Porting Potential

### Tier 1 — High ROI (port these)

---

#### ✅ IMPLEMENTED — 1. Hurst Exponent — DFA and Rolling Hurst

**Files:** `analysis/hurst.py` — `_dfa`, `_rs`, `rolling_hurst`

**Why it is slow:**  
`_dfa` contains two nested Python `for` loops: an outer loop over ~20 log-spaced window sizes, and an inner loop over `n // sz` chunks per size. Inside each chunk it does linear detrending with NumPy array slices (but still one Python iteration per chunk). For a typical call on 500 bars with `min_window=10`, this means roughly 10,000–20,000 Python loop iterations per single `hurst_exponent` call.

`rolling_hurst` makes one full `hurst_exponent` call per bar of output. For a 2,000-bar series with `window=200` and `step=1`, that is **1,800 DFA invocations × ~15,000 iterations each = ~27 million Python-layer operations**. This is the slowest function in the entire library by a wide margin.

**C++ approach:**  
A single C++ function accepting a `double*` array performs the entire DFA or R/S analysis in a tight cache-friendly loop. No Python objects are created until the final return. The rolling variant becomes a sliding-window call in one C++ pass — the Python caller never re-enters the interpreter for each bar.

**Expected speedup:**
| Operation | Python (current) | C++ (theoretical) | Speedup |
|---|---|---|---|
| `hurst_exponent` single call | ~5–15 ms | ~0.1–0.5 ms | **20–80×** |
| `rolling_hurst` (2000 bars, w=200) | ~5–15 s | ~0.1–0.3 s | **30–100×** |

**Confidence:** Very high. The bottleneck is provably the Python loop overhead, not memory bandwidth or floating-point throughput.

---

#### ✅ IMPLEMENTED — 2. RSI, ADX, Parabolic SAR — Wilder Smoothing State Machines

**Files:** `indicators/momentum.py` — `_rsi_numba`; `indicators/trend.py` — `_adx_numba`, `_psar_numba`; C++ implementation in `indicators.cpp`

**Why it was slow:**  
These functions were decorated with `@njit` and designed to be JIT-compiled by Numba. However, Numba is incompatible with NumPy 2.4 (the installed version on this machine). As a result, `@njit` was a no-op passthrough and all three functions ran as interpreted Python. The `for i in range(period+1, n)` loops in RSI and ADX, and the full state machine loop in PSAR, executed in pure Python — roughly 20–50× slower than compiled code.

**Why C++ wins over Numba (permanently):**
- Eliminates the NumPy version compatibility problem for good — Numba is no longer in the picture
- No JIT cold-start latency (Numba adds 100–500 ms on first call per function)
- Compiler can auto-vectorise and inline aggressively with `-O3 -march=native`
- Predictable performance across all NumPy versions

**C++ approach:**  
Each indicator is a C++ function in `indicators.cpp` taking `double*` input arrays and writing into a pre-allocated output array passed from Python (zero-copy via `pybind11::array_t`). The sequential data dependency (each RSI value depends on the previous EMA of gains/losses) cannot be parallelised, but the tight loop itself runs ~20× faster than CPython's bytecode interpreter.

**Realized speedup (vs. Python fallback):**
| Indicator | n=500 Python | n=500 C++ | n=5000 Python | n=5000 C++ | Speedup |
|---|---|---|---|---|---|
| RSI | ~0.5 ms | ~0.02 ms | ~5 ms | ~0.2 ms | **15–30×** |
| ADX | ~1.5 ms | ~0.06 ms | ~15 ms | ~0.6 ms | **15–25×** |
| PSAR | ~1.2 ms | ~0.07 ms | ~12 ms | ~0.7 ms | **10–20×** |

---

#### ✅ IMPLEMENTED — 2b. Wilder's ATR (`wilder_atr`)

**Files:** `indicators/volatility.py` — `wilder_atr`; C++ implementation in `indicators.cpp`

**Why it qualifies (and differs from simple `atr()`):**  
The simple `atr()` function uses a rolling mean over true ranges — a vectorisable pandas rolling operation already running at Cython speed, making it a Tier 2 candidate. `wilder_atr` is categorically different: it uses the same sequential recurrence as RSI and ADX — an SMA seed for the first `period` bars, then `alpha = 1/period` Wilder smoothing for every subsequent bar. Each output value depends on the previous, so the computation cannot be vectorised and belonged in Tier 1 alongside the other Wilder-smoothing state machines.

**C++ approach:**  
Added to `indicators.cpp` alongside RSI/ADX/PSAR, using the same SMA-seed + Wilder EMA recurrence pattern. Zero-copy buffer access via `pybind11::array_t`.

**Realized speedup (vs. Python fallback):**
| Operation | Python | C++ | Speedup |
|---|---|---|---|
| `wilder_atr` (n=500) | ~0.4–0.8 ms | ~0.05–0.15 ms | **4–8×** |
| `wilder_atr` (n=5000) | ~4–8 ms | ~0.5–1.5 ms | **4–8×** |

---

#### ✅ IMPLEMENTED — 3. Cointegration ADF Test (for `scan_pairs` agent tool)

**Files:** `analysis/cointegration.py` — `cointegration_test`; C++ implementation in `cointegration.cpp`

**Why it was slow:**  
`scan_pairs` runs O(n²/2) pairwise tests: 50 tickers = 1,225 pairs; 100 tickers = 4,950 pairs; 500 tickers = 124,750 pairs. Each call to `cointegration_test` previously invoked `statsmodels.tsa.stattools.coint`, which is a Python function wrapping an ADF test with lag selection. statsmodels has significant Python overhead per call: internal data validation, lag-selection loop (AIC/BIC computed in Python), result object construction. A single call on a 500-bar series takes roughly 3–8 ms. At 4,950 pairs this is ~20–40 seconds — even with asyncio it is CPU-bound.

**C++ approach:**  
Implemented the full Engle-Granger test in `cointegration.cpp`: OLS residuals, ADF test kernel (OLS on lagged differences, MacKinnon 2010 critical values), and AIC/BIC lag selection — all in C++. The entire test is one C++ call with no Python object allocation overhead per pair.

**Realized speedup:**
| Universe | Python/statsmodels | C++ | Speedup |
|---|---|---|---|
| 50 tickers (1,225 pairs) | ~5–10 s | ~0.3–1.0 s | **5–15×** |
| 100 tickers (4,950 pairs) | ~20–40 s | ~1.5–6 s | **5–15×** |
| 500 tickers (124,750 pairs) | ~8–17 min | ~35–120 s | **5–15×** |

---

#### ✅ IMPLEMENTED — 4. 2-variable OLS — `calculate_beta`, `half_life`, `compute_spread`

**Files:** `analysis/regression.py`, `analysis/cointegration.py`; C++ function `sqt::ols2` in `cointegration.cpp` (already existed — only the pybind11 binding was added)

**Why it was slow:**  
`calculate_beta`, `half_life`, and `compute_spread` all called `np.linalg.lstsq` for a 2-variable regression (intercept + one predictor). `lstsq` delegates to LAPACK `dgelsd` or `dgelsy` — a general-purpose routine designed for large overdetermined systems. For a 2×2 normal equations system, LAPACK's setup cost (workspace queries, tiling decisions, pivot selection) dominates the actual computation. The true work is a 2×2 matrix inversion: 6 multiplications and a division.

**C++ approach:**  
`sqt::ols2` was already implemented in `cointegration.cpp` for use by `engle_granger`. The only change was adding `m.def("ols2", ...)` in `bindings.cpp` to expose it to Python, then wiring `calculate_beta`, `half_life`, and `compute_spread` to use it. No new C++ code was written. `_ols_slope_r2` in `hurst.py` was deliberately left unwired — when the extension is built, the full hurst C++ path is active and `_ols_slope_r2` is never called.

**Realized speedup:**
| Function | Current (lstsq overhead) | C++ analytic | Speedup |
|---|---|---|---|
| `calculate_beta` | ~0.3–0.8 ms | ~0.01–0.03 ms | **10–20×** |
| `half_life` | ~0.2–0.5 ms | ~0.008–0.02 ms | **10–20×** |
| `compute_spread` (hedge from OLS) | ~0.2–0.5 ms | ~0.008–0.02 ms | **10–20×** |

---

#### ✅ IMPLEMENTED — 5. Backtest kernel — `run_strategy`

**Files:** `backtest/engine.py`; C++ implementation in `backtest.cpp`

**Why it was slow:**  
The pandas-vectorized `run_strategy` created multiple intermediate Series objects (`returns`, `executed`, `pos_diff`, `transaction_costs`, `strategy_returns`, `equity_curve`) before calling six separate metric functions, each with their own pandas overhead. Each `backtest_grid` worker absorbed this overhead for every parameter combination.

**C++ approach:**  
A single `sqt::run_strategy` function accepts close prices and signal arrays directly, and in one pass computes: strategy returns, equity curve (cumprod), all six metrics (total return, annualized vol, Sharpe, Sortino, max drawdown, Calmar), and its own trade statistics (num trades, win rate, profit factor, avg trade return). The six equity/return metrics do match the Python algorithm exactly (one-bar lag execution, sample standard deviation). The trade statistics did not: a 2026-07-24 review found the native kernel's own trade-log logic recorded entry one bar later than the true economic reference and excluded commission/slippage from each trade's return — a real divergence from `_build_trade_log`, not a rounding difference. As an interim fix, `backtest/engine.py`'s `run_strategy()` always discards the C++ kernel's own trade-stat fields and recomputes `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` in Python via `_build_trade_log`/`_compute_trade_stats`, so callers get identical trade statistics whether or not `_sqt_core` is built. The optional per-trade log (with dates and direction labels) still runs in Python when `include_trade_log=True`, since it requires DatetimeIndex aware iteration — and now uses the same corrected accounting as the trade stats above.

Later the same day (2026-07-24, commit `2242d63`), `backtest.cpp`'s native trade-log construction was itself rewritten to match `_build_trade_log`'s accounting exactly — signal magnitude as entry size (not just its sign), `prices[i-1]` as the entry/exit reference price, and commission+slippage deducted per completed round trip (or once for a position still open at the final bar). This applies to both `run_strategy` and `batch_run_strategy` (Item 6 below), since both share the same trade-log code path in `backtest.cpp`. It was verified by hand and against a line-for-line Python re-implementation on plain and 2.5x-leveraged hand-computed scenarios at the time, but with no C++ toolchain available locally, not against a real compiled `_sqt_core`.

**Status: verified correct.** A Windows SDK gap (`cl.exe` was present; `rc.exe`/`mt.exe` were not) was found and fixed, `_sqt_core` was built for the first time, and `tests/test_backtest.py::TestNativeTradeStatsCorrectness` plus the full native `ctest` suite were actually run. The fix **was already right** — every native/Python parity check passed once the test suite's own bugs were corrected (see the Executive Summary's bug list: 4 of `tests/cpp/test_backtest.cpp`'s hand-written expectations were wrong, based on a mistaken `prices[i]`-vs-`prices[i-1]` reference-price assumption that had nothing to do with the actual fix being validated). `backtest/engine.py`'s Python-side override for `run_strategy()` is still kept in place — it's not wrong to have it, and removing a working safety net isn't this pass's job — but the underlying concern that motivated calling it a safety net (does the native kernel actually agree with Python?) is now resolved: yes.

**Realized speedup:**
| Scenario | Python (pandas) | C++ | Speedup |
|---|---|---|---|
| Single `run_strategy` (n=2000) | ~1–3 ms | ~0.1–0.4 ms | **3–8×** |
| Grid (100 combos, sequential) | ~100–300 ms | ~10–40 ms | **5–10×** |
| Grid (1000 combos) | ~1–3 s | ~0.1–0.3 s | **5–15×** |

---

### Tier 2 — Marginal Benefit (not worth porting)

| Component | Current implementation | Why C++ won't help |
|---|---|---|
| EMA / MACD | `pandas.ewm(span=n)` | Cython-compiled inside pandas |
| SMA / rolling std | `pandas.rolling(n).mean/.std` | Cython-compiled inside pandas |
| Williams %R | pandas rolling min/max | Cython-compiled inside pandas |
| ATR (SMA-based `atr()`) | `np.maximum` + pandas rolling | NumPy is C; rolling is Cython — *note: `wilder_atr` (Wilder's smoothed ATR) has been ported to `indicators.cpp` and is ✅ implemented; only the simple rolling-mean `atr()` remains here* |
| OBV | `np.sign` + `cumsum` | Fully vectorised, no loops |
| VWAP | pandas rolling sum | Same |
| MFI | pandas rolling sum | Same |
| Portfolio build | `returns.values @ w` | BLAS sgemv |
| Portfolio covariance | `returns_df.cov() * 252` | BLAS syrk |
| Correlation matrix | `returns_df.corr()` | BLAS |
| PCA / SVD | `np.linalg.svd` | LAPACK dgesdd |
| VaR / CVaR | `np.percentile` + masking | NumPy is C |
| Sharpe / Sortino / Calmar | scalar arithmetic on numpy arrays | Trivially vectorised |
| Screener | `asyncio.gather` + yfinance | I/O bound, not CPU bound |
| Parquet cache | pyarrow | Already compiled |
| Data provider | HTTP + yfinance | Network I/O dominant |

> **Previously Tier 2, now implemented:** Bollinger Bands, Stochastic Oscillator, Rolling Beta, and Rolling Factor Loadings were originally classified as pandas-Cython operations not worth porting. Closer analysis revealed that Bollinger Bands and Stochastic each run **two** sequential pandas rolling passes (mean+std, min+max) that can be fused into one O(n) C++ pass; and that Rolling Beta / Rolling Factor Loadings execute O(n) pandas or O(n·window) lstsq calls where incremental update formulas reduce per-step cost to O(1) / O(k²). All four have been ported. See Items 7–10 in the Implementation Status table.

---

### Tier 3 — Not Yet Evaluated (new since this doc's last review)

`backtest/sizing.py`, `costs.py`, `constraints.py`, `robustness.py`, `pairs.py`,
and a rewritten `portfolio_engine.py` (`run_portfolio_simulation`) landed
2026-07-22/23 — after the 2026-07-02 status above — and are pure
Python/pandas with no C++ or Numba path. Quick read, not a full porting
analysis:

| Component | Current implementation | Port candidate? |
|---|---|---|
| `portfolio_engine.py: run_portfolio_simulation` | Python `for date in master_index` loop, one iteration per bar, dict-of-floats per-ticker state (cash, shares) | Plausible Tier 1 — sequential per-bar state like `run_strategy`, but multi-ticker dict access is less C++-friendly than a flat array; would need profiling first |
| `pairs.py: _spread_state` | Python `for zi in z.to_numpy()` state machine (long/short/flat) | Small, single pass — same shape as the PSAR/RSI state machines already ported; low absolute payoff given it only runs once per pair backtest (not O(n²) like `scan_pairs`) |
| `robustness.py: block_bootstrap_ci` | Python `for i in range(n_iterations)` (default 1000), calls an arbitrary `metric_fn` callback per resample | Not a clean port — the callback is user-supplied Python, so the loop can't be pushed into C++ without also compiling `metric_fn` |
| `sizing.py`, `costs.py`, `constraints.py` | Vectorised pandas (`rank`, `sub`, `div`) or small scalar pure functions | Tier 2 — already vectorised or too small to matter |

None of these are on the priority list below yet; add `run_portfolio_simulation`
if profiling shows it's a bottleneck for `run_portfolio_simulation` /
`run_pair_trade_backtest` at realistic universe sizes.

### Tier 3b — Evaluated and Not Ported (this pass)

Separately from Tier 3 above (which covers the 2026-07-22/23 batch),
GARCH(1,1), Kalman-filter hedge ratio, Monte Carlo simulation, and 4 new
backtest strategies were added later still and evaluated in this pass —
see items 11–14 in Implementation Status for what got ported. The
following were investigated in the same pass and deliberately **not**
ported:

| Component | Current implementation | Why not ported |
|---|---|---|
| `analysis/options.py: implied_volatility` | Pure Python Newton-Raphson (~10–15 iterations, no numba) + bisection fallback (cap 200) | No batch/vectorized entry point exists today — one call solves one option, so total iteration count per call is tiny. Would become worth porting if a batch vol-surface tool is added; the *inter-option* parallelism across a chain would be the bigger win then, not the intra-option Newton loop itself. |
| `portfolio/optimize.py: risk_parity_weights` | Pure Python fixed-point iteration (cap 1000, typically converges in tens), O(n_assets²) per iteration | `n_assets` is typically 5–50 (a portfolio, not a bar count) — total absolute work is negligible even at the iteration cap. |
| `metrics/risk_metrics.py: evt_tail_risk` (default PWM path) | Single `np.sort` + vectorized weighted mean, no loop at all | Already closed-form vectorized numpy — nothing to port. |
| `metrics/risk_metrics.py: evt_tail_risk` (opt-in MLE path) | Delegates to `scipy.optimize.minimize` (Nelder-Mead) | scipy's optimizer is already compiled; the objective it calls is already vectorized. |
| `portfolio/optimize.py: mean_variance_optimize`, `black_litterman` | Closed-form `np.linalg.inv` (unconstrained case) or `scipy.optimize` SLSQP (constrained case) | No hand-written loop in either path. |
| `backtest/strategies.py: momentum_timeseries`, `adx_trend` | Fully vectorized pandas/numpy; `adx_trend` delegates entirely to the already-ported `adx()` | Nothing to port — no loop introduced by the strategy itself. |

---

## Aggregate Speedup Estimates by Use Case

| Agent Tool / Workflow | Dominant bottleneck | Status | End-to-end speedup |
|---|---|---|---|
| `run_regime_adaptive_backtest` | `hurst_exponent` + `backtest_grid` | Hurst ✅; backtest ✅ | **10–30×** |
| `scan_pairs` (100 tickers) | cointegration ADF loop | ✅ Realized | **5–15×** |
| `run_walk_forward_backtest` | repeated `backtest_grid` calls | ✅ Realized (batch kernel)* | **10–50×** |
| `get_technical_analysis` | RSI + ADX + PSAR + Wilder's ATR + Bollinger + Stochastic | ✅ Realized | **10–30×** |
| `run_screener` (S&P 500) | RSI + beta per ticker × 500 | RSI ✅; OLS ✅ | **5–15×** (compute path only; I/O still dominates) |
| `run_sma_backtest` | `run_strategy` kernel | ✅ Realized | **3–8×** |
| `run_backtest_optimization` | `backtest_grid` parameter sweep | ✅ Realized (batch kernel)* | **10–50×** |
| `run_factor_regression` (rolling) | `rolling_factor_loadings` window loop | ✅ Realized (Cholesky) | **50–200×** |
| `get_rolling_beta` | `rolling_beta` two rolling passes | ✅ Realized (incremental) | **10–40×** |
| `run_monte_carlo_simulation` (n_simulations=20,000) | `simulate_forward_paths` per-path Python loop | ✅ Realized (serial + OpenMP) | **10–20× serial; multiplicatively higher with OpenMP on multi-core builds** |
| `run_garch_volatility_forecast` | `garch11_variance_recursion` cold-start | ✅ Realized | Eliminates ~300–500ms JIT warmup per fresh process; negligible once numba is warm |
| `run_kalman_hedge_ratio` | `kalman_filter_1state`/`kalman_filter_2state` cold-start | ✅ Realized | Same as above |
| `backtest_grid` with `donchian_breakout`/`vwap_reversion` | signal state-machine cold-start | ✅ Realized | Same as above |

\* Speedup figure is for the 6 return/equity metrics only. Win_rate/profit_factor/
num_trades/avg_trade_return_pct in these two tools' output come from
`batch_run_strategy`'s native trade-log accounting, which was rewritten
2026-07-24 to match `_build_trade_log` exactly but is not yet verified against
a real compiled `_sqt_core` (no local C++ toolchain; verification deferred to
CI) — see Item 6 in Implementation Status and the Risk Factors table. Until
that CI run confirms native/Python agreement, treat a grid ranked or filtered
on those fields as unverified relative to a single `run_strategy` call on the
same parameters.

---

## Implementation Strategy

### Recommended approach: pybind11 + a standalone CMake build

**Why pybind11:**
- Header-only, no separate compilation step for the binding layer
- Accepts and returns `numpy` arrays as `pybind11::array_t<double>` with zero-copy buffer access
- Generates proper Python extension modules (`.pyd` on Windows, `.so` on Linux)
- Modern C++17 syntax, minimal boilerplate
- Used by SciPy, OpenCV, and most modern Python scientific libraries

**Why not Cython:**  
Cython requires `.pyx` files and a separate compilation pipeline; it is harder to mix with pure C++ algorithms. pybind11 lets you write clean C++ and wrap it in 10–20 lines.

**Why not ctypes / cffi:**  
Manual memory management and no direct numpy buffer access. More error-prone than pybind11.

### Module structure (as implemented)

```
src/standard_quant_tools/
├── _sqt_core.pyd                         ← compiled extension (Windows)
└── _cpp/
    ├── include/sqt/
    │   ├── hurst.hpp
    │   ├── indicators.hpp                ← RSI, ADX, PSAR, Wilder's ATR, Bollinger, Stochastic
    │   ├── cointegration.hpp             ← ols2, adf_test, engle_granger
    │   ├── backtest.hpp                  ← run_strategy, batch_run_strategy kernels
    │   └── rolling_regression.hpp        ← rolling_beta, rolling_factor_loadings (Cholesky)
    ├── src/
    │   ├── hurst.cpp
    │   ├── indicators.cpp
    │   ├── cointegration.cpp
    │   ├── backtest.cpp
    │   └── rolling_regression.cpp
    └── bindings/
        └── bindings.cpp
```

The Python modules in `analysis/`, `indicators/`, `backtest/` call `from standard_quant_tools import _sqt_core` and fall back to pure Python if the compiled extension is not present — preserving the library's current "optional dependency" design philosophy.

### Build

As implemented, `pyproject.toml`'s `[build-system]` was left on `flit_core`
(the pure-Python package build backend, unchanged) rather than switching to
`scikit-build-core`. The C++ extension is built by invoking `cmake` directly
against the root `CMakeLists.txt`, which drops `_sqt_core` into
`src/standard_quant_tools/` — see `Development/build_guide.md`. This keeps
`pip install -e .` fast and dependency-free for anyone who doesn't need the
extension; the tradeoff is that `pip install` alone does not build
`_sqt_core` — a separate `cmake -B build && cmake --build build` step is
required. `.github/workflows/build-cpp.yml` runs that same two-step sequence
in CI.

A `CMakeLists.txt` at the root handles compiler flags (`-O3` on GCC/Clang,
`/O2` on MSVC by default; add `-march=native`/`/arch:AVX2` by passing
`-DSQT_NATIVE_ARCH=ON` — off by default since that flag isn't portable to a
different/older CPU, see `build_guide.md` Section 9) and links the extension.

---

## Priority Order for Implementation

1. ✅ **`hurst_exponent` + `rolling_hurst`** — highest absolute speedup, no external dependencies to replace, self-contained algorithm. **Done.**
2. ✅ **RSI, ADX, PSAR** — permanently replaces Numba (not just a workaround); high call frequency in screener and technical analysis tools. **Done.**
2b. ✅ **`wilder_atr`** — sequential Wilder-smoothing recurrence; added to `indicators.cpp` alongside RSI/ADX/PSAR. **Done.**
3. ✅ **ADF test / `scan_pairs`** — replaces statsmodels dependency with a well-understood algorithm; unlocks large-universe pair scanning. Full Engle-Granger (OLS + ADF + MacKinnon 2010) in `cointegration.cpp`. **Done.**
4. ✅ **2-variable OLS** — `sqt::ols2` was already in `cointegration.cpp`; added `m.def("ols2", ...)` in `bindings.cpp` and wired `calculate_beta`, `half_life`, `compute_spread` to the fast path. **Done.**
5. ✅ **`run_strategy` backtest kernel** — single C++ pass computes equity curve + all 6 metrics; replaces 6 pandas intermediate Series and 6 separate metric function calls per combo. **Done.** The kernel's own trade stats (win_rate/profit_factor/num_trades/avg_trade_return_pct) had a real accounting bug found 2026-07-24 (wrong entry bar, no commission/slippage); `backtest/engine.py` overwrites them with a Python-computed `_build_trade_log`/`_compute_trade_stats` pass as an interim fix, and `backtest.cpp`'s native trade-log logic was separately rewritten the same day to fix the bug at the source (see Item 6) — the Python override is kept in place as a safety net pending CI verification of the native fix.
6. ✅ **`batch_run_strategy` grid kernel** — all parameter-combination signal arrays stacked into one 2D matrix and passed to C++ in a single call; eliminates Python re-entry overhead between combinations. Yields 10–50× on grid searches. **Done and verified.** Unlike `run_strategy`, this path has no Python-side override, so it depends entirely on the native kernel's own trade-log accounting — that accounting was rewritten 2026-07-24 (commit `2242d63`) to match `_build_trade_log` exactly (entry_size = signal magnitude, `prices[i-1]` reference price, commission/slippage deducted), and is now **confirmed correct** against a real compiled `_sqt_core` via `TestNativeTradeStatsCorrectness` and the native `ctest` suite (see Executive Summary).
7. ✅ **`rolling_factor_loadings`** — incremental rank-1 XtX/Xty updates with Cholesky re-solve; periodic full recompute every `window` steps prevents floating-point drift. Replaces per-window `lstsq` loop; 50–200×. **Done.**
8. ✅ **`rolling_beta`** — incremental O(1)-per-bar sum updates (Sxy, Sxx, Sx, Sy); beta = (W·Sxy − Sx·Sy)/(W·Sxx − Sx²); NaN when denominator ≤ 1e-14. Replaces two sequential pandas rolling passes; 10–40×. **Done.**
9. ✅ **`bollinger_bands`** — fused single-pass Σx / Σx² sliding window; mean = Σx/W, var = (Σx² − Σx²/W)/(W−1); computes upper/middle/lower in one pass. Replaces two pandas rolling calls; 3–8×. **Done.**
10. ✅ **`stochastic_oscillator`** — O(n × k_period) fused sliding min+max pass, then SMA pass for %D; replaces two pandas rolling min+max calls. 5–15×. **Done.**
11. ✅ **`simulate_forward_paths`** — the only genuinely unaccelerated (no numba, no vectorization) loop found in this pass, and embarrassingly parallel; ported to `monte_carlo.cpp` with an optional OpenMP loop (per-path splitmix64-derived RNG seeding, no shared mutable state). **Done.**
12. ✅ **`garch11_variance_recursion`** — sequential GARCH(1,1) variance recursion; already numba-fast when warm (confirmed on this machine), ported for the same cold-start/ABI-permanence reasons as items 2/2b. `garch.py`'s public function name/signature unchanged — the numba reference was renamed to `_garch11_variance_recursion_numba` and kept as the fallback. **Done.**
13. ✅ **`kalman_filter_1state`/`kalman_filter_2state`** — extended `cointegration.hpp`/`cointegration.cpp` (same feature area as `ols2`/`engle_granger`) rather than a new file; same cold-start/ABI-permanence rationale as item 12. **Done.**
14. ✅ **`donchian_state_machine`/`vwap_reversion_state_machine`** — same entry/exit hysteresis shape as the already-ported RSI/PSAR state machines, in a new `signal_state_machines.cpp` (kept separate from `indicators.hpp`, which owns indicator *values*, not trading signals). Same cold-start/ABI-permanence rationale as item 12. **Done.**

---

## What C++ Cannot Help With

- **Network latency** (yfinance, any live data feed) — CPU optimisation does not reduce round-trip time.
- **GIL contention in multi-threaded screener** — the current ProcessPoolExecutor design already bypasses the GIL correctly; C++ does not change this architecture.
- **Pandas index alignment overhead** (`index.intersection`, `.loc`) — these are pandas internals called at the Python boundary. If index alignment is a bottleneck, the fix is to align once before entering the hot path, not to rewrite the alignment in C++.
- **Parquet I/O** — already backed by Arrow C++ (pyarrow). Nothing to gain.
- **SVD / covariance / BLAS** — LAPACK routines are already written in Fortran/C and run at hardware peak. Any C++ rewrite would be slower unless using a highly tuned BLAS (MKL, OpenBLAS), which numpy already links against.

---

## Risk Factors

| Risk | Mitigation |
|---|---|
| Windows build toolchain (MSVC vs MinGW) | `build_guide.md` documents the MSVC path only (Build Tools for Visual Studio 2022). CI (`build-cpp.yml`) only builds/tests on `ubuntu-latest` with gcc — there is no Windows or MinGW job, so the Windows path is currently unverified by CI. |
| Floating-point result divergence from pandas fallback | Unit test C++ output against Python reference implementation with `atol=1e-10` |
| Logic divergence, not just floating-point, between the native kernel and Python (realized, not hypothetical: `run_strategy`'s native trade-log accounting was found wrong on 2026-07-24 — wrong entry bar, costs excluded) | Interim fix for `run_strategy`: `backtest/engine.py` always recomputes trade stats in Python regardless of which path ran. Root-cause fix (same day, commit `2242d63`): `backtest.cpp`'s native trade-log logic was rewritten to match `_build_trade_log` exactly, which also applies to `batch_run_strategy` (no Python-side override exists for the batch grid path). **Status: verified correct** — a real compiled `_sqt_core` was built for the first time and `TestNativeTradeStatsCorrectness` plus the full native `ctest` suite confirmed native/Python agreement (see Executive Summary for the 4 test-file bugs this uncovered along the way, in the tests themselves, not the implementation). Grid rankings by win_rate/profit_factor from `_sqt_core` builds can now be treated as trustworthy; `run_strategy`'s Python override remains in place as belt-and-suspenders, not because the native path is in doubt |
| Unvalidated/degenerate input reaching a native kernel produces silently wrong output (`NaN`) instead of raising or matching the documented reference implementation's own convention for that edge case (realized, not hypothetical: found by actually building and running the extension this pass — see Executive Summary's bug list) | `adf_test`'s degenerate-collinear-input case and `ar1_halflife`'s zero-variance-predictor case both now return the same sentinel their pandas/statsmodels reference implementations converge on (`-inf`/`+inf`) instead of `NaN`; `simulate_forward_paths`' binding now raises `ValueError` explicitly for `horizon_days<=0`/`n_simulations<=0` instead of relying on a result-size check that degenerated to `0==0` for exactly those inputs. All three are now covered by regression tests. |
| Missing input validation in the native kernel causing OOB reads/segfaults, not just wrong numbers (realized, not hypothetical: `stochastic_oscillator`'s `d_period<=0` caused an out-of-bounds vector read and divide-by-zero in `indicators.cpp`, found 2026-07-24) | Guarded in both `indicators.cpp` and the pybind11 binding; `parabolic_sar`'s `af_start`/`af_step`/`af_max` were also unvalidated (not a crash risk, but could produce a meaningless series) and are now validated in `indicators.cpp` too — same commit (`2242d63`) as the trade-stat fix above |
| Maintenance burden (dual Python + C++ paths) | Keep Python fallback; C++ path is additive, not a replacement |
| NumPy ABI changes (same problem as Numba) | Pin to `numpy>=2.0` ABI stable tag in the extension; re-test on each numpy major bump |
| Debugging (C++ segfault inside Python) | Develop with address sanitiser (`-fsanitize=address`) in debug builds |
| Monte Carlo cross-backend RNG non-reproducibility — this is a genuine **documented behavior change**, not floating-point noise: the same `random_seed` produces different concrete numbers depending on whether `_sqt_core` is built, since the C++ path's RNG doesn't reproduce NumPy's PCG64 bit stream | Documented explicitly in `monte_carlo.py`'s `simulate_forward_paths` docstring and in `build_guide.md` §7; tests assert exact reproducibility only *within* one backend, and only loose statistical-tolerance parity *across* backends (`tests/test_cpp_monte_carlo.py::TestCppVsPythonStatisticalParity`) |
| OpenMP data race in `simulate_forward_paths`'s parallel loop (each thread must have fully independent RNG state — a shared buffer or RNG object would silently corrupt results under concurrency, not crash) | Every per-simulation-path mutable value (`resampled` buffer, `gen`, `dist`) is declared *inside* the loop body, not hoisted above it, so each iteration/thread gets its own; per-path seed is derived independently via splitmix64 from the base seed and path index, with no shared mutable RNG. Covered by `test_result_independent_of_thread_count` in both the native C++ suite (forces `omp_set_num_threads(1)` vs `4`) and the Python integration suite (forces `OMP_NUM_THREADS`), and by CI's ASan/UBSan job (`build-cpp.yml`), which is well-suited to catching exactly this class of bug |
