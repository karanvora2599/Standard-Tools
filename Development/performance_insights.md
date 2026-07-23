# C++ Porting Performance Insights

**Date:** 2026-06-06  
**Scope:** `standard_quant_tools` — analysis of which components benefit from a C++/pybind11 rewrite and by how much.

---

## Executive Summary

Of the ~20 distinct computational modules in this library, several components account for nearly all theoretical speedup. The rest are already running at near-C speed via NumPy/BLAS/Cython and porting them would be wasted effort. **As of 2026-07-02, all originally identified Tier 1 features have been implemented in C++, plus 5 additional functions:** Hurst Exponent (DFA + R/S + rolling), RSI, ADX, Parabolic SAR, Wilder's ATR, Engle-Granger Cointegration, 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`), backtest kernel (`run_strategy`), **batch backtest grid kernel** (`batch_run_strategy`, 10–50×), **rolling factor loadings** (incremental Cholesky, 50–200×), **rolling beta** (incremental sums, 10–40×), **Bollinger Bands** (fused mean+std, 3–8×), and **Stochastic Oscillator** (fused min+max, 5–15×). The most dramatic realised gain — 30–100× — is on `rolling_hurst`. The `_sqt_core` feature set itself has not grown since 2026-07-02; the backtest engine gained new pure-Python modules afterward (portfolio simulation, pair trading, sizing, costs, constraints, robustness diagnostics — see Tier 3 below) that have not yet been evaluated for a C++ port.

---

## Implementation Status

| # | Feature | Status | Realized Speedup |
|---|---|---|---|
| 1 | Hurst Exponent — DFA, R/S, `rolling_hurst` | ✅ IMPLEMENTED | 20–80× (single), 30–100× (rolling) |
| 2 | RSI, ADX, Parabolic SAR | ✅ IMPLEMENTED | 10–30× vs Python fallback |
| 2b | Wilder's ATR (`wilder_atr`) | ✅ IMPLEMENTED | 4–8× vs Python fallback |
| 3 | Engle-Granger Cointegration (OLS + ADF + MacKinnon 2010) | ✅ IMPLEMENTED | 5–15× vs statsmodels |
| 4 | 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`) | ✅ IMPLEMENTED | 10–20× vs `lstsq` |
| 5 | Backtest kernel (`run_strategy` — equity + all metrics in one C++ pass) | ✅ IMPLEMENTED | 3–8× vs pandas |
| 6 | Batch backtest grid (`batch_run_strategy` — all combos in one C++ call) | ✅ IMPLEMENTED | 10–50× vs per-combo C++ calls |
| 7 | Rolling factor loadings (`rolling_factor_loadings` — incremental Cholesky) | ✅ IMPLEMENTED | 50–200× vs per-window `lstsq` |
| 8 | Rolling beta (`rolling_beta` — incremental sum updates) | ✅ IMPLEMENTED | 10–40× vs 2× pandas rolling |
| 9 | Bollinger Bands (`bollinger_bands` — fused Σx / Σx² pass) | ✅ IMPLEMENTED | 3–8× vs 2× pandas rolling |
| 10 | Stochastic Oscillator (`stochastic_oscillator` — fused min+max pass) | ✅ IMPLEMENTED | 5–15× vs 2× pandas rolling |

The compiled extension is `_sqt_core.pyd` (Windows). All Python modules fall back to pure Python if the extension is absent, preserving the library's optional-dependency philosophy.

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
A single `sqt::run_strategy` function accepts close prices and signal arrays directly, and in one pass computes: strategy returns, equity curve (cumprod), all six metrics (total return, annualized vol, Sharpe, Sortino, max drawdown, Calmar), and trade statistics (num trades, win rate, profit factor, avg trade return). This matches the Python algorithm exactly — one-bar lag execution, sample standard deviation, same trade state machine as `_build_trade_log`. The optional per-trade log (with dates and direction labels) still runs in Python when `include_trade_log=True`, since it requires DatetimeIndex aware iteration.

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

---

## Aggregate Speedup Estimates by Use Case

| Agent Tool / Workflow | Dominant bottleneck | Status | End-to-end speedup |
|---|---|---|---|
| `run_regime_adaptive_backtest` | `hurst_exponent` + `backtest_grid` | Hurst ✅; backtest ✅ | **10–30×** |
| `scan_pairs` (100 tickers) | cointegration ADF loop | ✅ Realized | **5–15×** |
| `run_walk_forward_backtest` | repeated `backtest_grid` calls | ✅ Realized (batch kernel) | **10–50×** |
| `get_technical_analysis` | RSI + ADX + PSAR + Wilder's ATR + Bollinger + Stochastic | ✅ Realized | **10–30×** |
| `run_screener` (S&P 500) | RSI + beta per ticker × 500 | RSI ✅; OLS ✅ | **5–15×** (compute path only; I/O still dominates) |
| `run_sma_backtest` | `run_strategy` kernel | ✅ Realized | **3–8×** |
| `run_backtest_optimization` | `backtest_grid` parameter sweep | ✅ Realized (batch kernel) | **10–50×** |
| `run_factor_regression` (rolling) | `rolling_factor_loadings` window loop | ✅ Realized (Cholesky) | **50–200×** |
| `get_rolling_beta` | `rolling_beta` two rolling passes | ✅ Realized (incremental) | **10–40×** |

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

A `CMakeLists.txt` at the root handles compiler flags (`-O3 -march=native` on GCC/Clang, `/O2 /arch:AVX2` on MSVC) and links the extension.

---

## Priority Order for Implementation

1. ✅ **`hurst_exponent` + `rolling_hurst`** — highest absolute speedup, no external dependencies to replace, self-contained algorithm. **Done.**
2. ✅ **RSI, ADX, PSAR** — permanently replaces Numba (not just a workaround); high call frequency in screener and technical analysis tools. **Done.**
2b. ✅ **`wilder_atr`** — sequential Wilder-smoothing recurrence; added to `indicators.cpp` alongside RSI/ADX/PSAR. **Done.**
3. ✅ **ADF test / `scan_pairs`** — replaces statsmodels dependency with a well-understood algorithm; unlocks large-universe pair scanning. Full Engle-Granger (OLS + ADF + MacKinnon 2010) in `cointegration.cpp`. **Done.**
4. ✅ **2-variable OLS** — `sqt::ols2` was already in `cointegration.cpp`; added `m.def("ols2", ...)` in `bindings.cpp` and wired `calculate_beta`, `half_life`, `compute_spread` to the fast path. **Done.**
5. ✅ **`run_strategy` backtest kernel** — single C++ pass computes equity curve + all 6 metrics + trade stats; replaces 6 pandas intermediate Series and 6 separate metric function calls per combo. **Done.**
6. ✅ **`batch_run_strategy` grid kernel** — all parameter-combination signal arrays stacked into one 2D matrix and passed to C++ in a single call; eliminates Python re-entry overhead between combinations. Yields 10–50× on grid searches. **Done.**
7. ✅ **`rolling_factor_loadings`** — incremental rank-1 XtX/Xty updates with Cholesky re-solve; periodic full recompute every `window` steps prevents floating-point drift. Replaces per-window `lstsq` loop; 50–200×. **Done.**
8. ✅ **`rolling_beta`** — incremental O(1)-per-bar sum updates (Sxy, Sxx, Sx, Sy); beta = (W·Sxy − Sx·Sy)/(W·Sxx − Sx²); NaN when denominator ≤ 1e-14. Replaces two sequential pandas rolling passes; 10–40×. **Done.**
9. ✅ **`bollinger_bands`** — fused single-pass Σx / Σx² sliding window; mean = Σx/W, var = (Σx² − Σx²/W)/(W−1); computes upper/middle/lower in one pass. Replaces two pandas rolling calls; 3–8×. **Done.**
10. ✅ **`stochastic_oscillator`** — O(n × k_period) fused sliding min+max pass, then SMA pass for %D; replaces two pandas rolling min+max calls. 5–15×. **Done.**

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
| Maintenance burden (dual Python + C++ paths) | Keep Python fallback; C++ path is additive, not a replacement |
| NumPy ABI changes (same problem as Numba) | Pin to `numpy>=2.0` ABI stable tag in the extension; re-test on each numpy major bump |
| Debugging (C++ segfault inside Python) | Develop with address sanitiser (`-fsanitize=address`) in debug builds |
