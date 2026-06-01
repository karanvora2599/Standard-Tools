# C++ Porting Performance Insights

**Date:** 2026-06-01  
**Scope:** `standard_quant_tools` — analysis of which components benefit from a C++/pybind11 rewrite and by how much.

---

## Executive Summary

Of the ~20 distinct computational modules in this library, **5 components account for nearly all theoretical speedup**. The rest are already running at near-C speed via NumPy/BLAS/Cython and porting them would be wasted effort. The most dramatic gain — 30–100× — is available on `rolling_hurst`, which is the only major component still written as a pure Python loop with no vectorisation. The second-highest priority is the family of Wilder-smoothing indicators (RSI, ADX, PSAR) whose Numba JIT paths are currently dead due to the NumPy 2.4 incompatibility, leaving them running as interpreted Python.

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

#### 1. Hurst Exponent — DFA and Rolling Hurst

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

#### 2. RSI, ADX, Parabolic SAR — Wilder Smoothing State Machines

**Files:** `indicators/momentum.py` — `_rsi_numba`; `indicators/trend.py` — `_adx_numba`, `_psar_numba`

**Why it is slow (right now):**  
These functions are decorated with `@njit` and are *designed* to be JIT-compiled by Numba. However, Numba is incompatible with NumPy 2.4 (the installed version on this machine). As a result, `@njit` is a no-op passthrough and all three functions run as interpreted Python. The `for i in range(period+1, n)` loops in RSI and ADX, and the full state machine loop in PSAR, execute in pure Python — roughly 20–50× slower than compiled code.

**Why C++ wins over waiting for Numba:**
- Eliminates the NumPy version compatibility problem permanently
- No JIT cold-start latency (Numba adds 100–500 ms on first call per function)
- Compiler can auto-vectorise and inline aggressively with `-O3 -march=native`
- Predictable performance across all NumPy versions

**C++ approach:**  
Each indicator becomes a C++ function taking `double*` input arrays and writing into a pre-allocated output array passed from Python (zero-copy via `pybind11::array_t`). The sequential data dependency (each RSI value depends on the previous EMA of gains/losses) cannot be parallelised, but the tight loop itself runs ~20× faster than CPython's bytecode interpreter.

**Expected speedup (vs. current Python fallback):**
| Indicator | n=500 Python | n=500 C++ | n=5000 Python | n=5000 C++ | Speedup |
|---|---|---|---|---|---|
| RSI | ~0.5 ms | ~0.02 ms | ~5 ms | ~0.2 ms | **15–30×** |
| ADX | ~1.5 ms | ~0.06 ms | ~15 ms | ~0.6 ms | **15–25×** |
| PSAR | ~1.2 ms | ~0.07 ms | ~12 ms | ~0.7 ms | **10–20×** |

*Note: These would also match what Numba would give when it eventually supports NumPy 2.x — so C++ is a permanent fix, not a workaround.*

---

#### 3. Cointegration ADF Test (for `scan_pairs` agent tool)

**Files:** `analysis/cointegration.py` — `cointegration_test`; uses `statsmodels.tsa.stattools.coint`

**Why it is slow:**  
`scan_pairs` runs O(n²/2) pairwise tests: 50 tickers = 1,225 pairs; 100 tickers = 4,950 pairs; 500 tickers = 124,750 pairs. Each call to `cointegration_test` invokes `statsmodels.tsa.stattools.coint`, which is a Python function wrapping an ADF test with lag selection. statsmodels has significant Python overhead per call: internal data validation, lag-selection loop (AIC/BIC computed in Python), result object construction. A single call on a 500-bar series takes roughly 3–8 ms. At 4,950 pairs this is ~20–40 seconds — even with asyncio it is CPU-bound.

**C++ approach:**  
Implement the Engle-Granger test in C++: OLS residuals (already using numpy lstsq — fine as-is), then implement the ADF test kernel (OLS on lagged differences, critical value lookup). The ADF kernel for a fixed lag is 5–10 lines of C++. The lag-selection loop (AIC/BIC over a handful of lags) is also trivial in C++. The entire test becomes one C++ call with no Python object allocation overhead.

**Expected speedup:**
| Universe | Python/statsmodels | C++ ADF | Speedup |
|---|---|---|---|
| 50 tickers (1,225 pairs) | ~5–10 s | ~0.3–0.7 s | **12–20×** |
| 100 tickers (4,950 pairs) | ~20–40 s | ~1.5–3 s | **12–20×** |
| 500 tickers (124,750 pairs) | ~8–17 min | ~30–75 s | **12–20×** |

**Confidence:** High. statsmodels overhead is well-documented and measurable.

---

#### 4. Tiny OLS (2-variable) — `calculate_beta`, `half_life`, `_ols_slope_r2`

**Files:** `analysis/regression.py`, `analysis/hurst.py`, `analysis/cointegration.py`

**Why it is slow:**  
All three call `np.linalg.lstsq` for a 2-variable regression (intercept + one predictor). `np.linalg.lstsq` delegates to LAPACK `dgelsd` or `dgelsy` — a general-purpose routine designed for large overdetermined systems. For a 2×2 normal equations system, LAPACK's setup cost (workspace queries, tiling decisions, pivot selection) dominates the actual computation. The true work is a 2×2 matrix inversion, which is 6 multiplications and a division.

**C++ approach:**  
A templated `ols2` function that computes the normal equations analytically for the 2-variable case: `beta = (X'X)^{-1} X'y`. This avoids LAPACK entirely. For `rolling_beta`, a rolling incremental OLS update (Woodbury identity) could avoid recomputing from scratch on every window — though the current pandas `cov/var` approach is already optimal for that case.

**Expected speedup:**
| Function | Current (lstsq overhead) | C++ analytic | Speedup |
|---|---|---|---|
| `calculate_beta` | ~0.3–0.8 ms | ~0.01–0.03 ms | **10–20×** |
| `half_life` | ~0.2–0.5 ms | ~0.008–0.02 ms | **10–20×** |
| `_ols_slope_r2` (Hurst) | ~0.1–0.3 ms × 20 sizes | ~0.003–0.01 ms | **15–25×** |

**Note:** The absolute time saved here is small per call. The gain matters most when these functions are called in tight loops — e.g., `_ols_slope_r2` is called inside every `hurst_exponent` invocation; this folds into the Hurst speedup estimate above.

---

#### 5. Backtest Grid — `backtest_grid` / `run_strategy` inner kernel

**Files:** `backtest/engine.py`

**Why it is partially slow:**  
`run_strategy` is largely vectorised via pandas/numpy and already fast for a single call. The bottleneck in `backtest_grid` for large grids (500+ combos on Windows) is the `ProcessPoolExecutor` spawn overhead: each worker process imports Python, loads the module, and starts an event loop before doing any work. For small grids the spawn overhead dominates computation time.

A secondary bottleneck: `_build_trade_log` iterates over **trade events** in Python. For a high-frequency strategy on intraday data this loop could see thousands of iterations. The equity curve `cumprod` is numpy (fast), but the transaction cost calculation and signal-shift are pandas (moderate).

**C++ approach:**  
The primary gain is a C++ `run_strategy` kernel that accepts numpy arrays directly (prices, signals) and returns metrics as a Python dict — eliminating all pandas Series construction overhead. For `backtest_grid`, a C++ kernel that accepts a 2D parameter matrix and runs all combos in a single pass (no subprocess overhead, no GIL, auto-vectorisation across combos) would dramatically reduce grid time.

**Expected speedup:**
| Scenario | Python | C++ | Speedup |
|---|---|---|---|
| Single `run_strategy` (n=2000) | ~1–3 ms | ~0.1–0.4 ms | **3–8×** |
| Grid (100 combos, sequential) | ~100–300 ms | ~10–40 ms | **5–10×** |
| Grid (1000 combos) | ~1–3 s | ~0.1–0.3 s | **5–15×** |

**Confidence:** Moderate. The pandas/numpy internals are already C-backed, so the gain here is from eliminating Python object allocation, Series indexing overhead, and subprocess spawn — not from a fundamentally faster algorithm.

---

### Tier 2 — Marginal Benefit (not worth porting)

| Component | Current implementation | Why C++ won't help |
|---|---|---|
| EMA / MACD | `pandas.ewm(span=n)` | Cython-compiled inside pandas |
| SMA / rolling std | `pandas.rolling(n).mean/.std` | Cython-compiled inside pandas |
| Bollinger Bands | pandas rolling | Same as above |
| Williams %R | pandas rolling min/max | Same |
| ATR | `np.maximum` + pandas rolling | NumPy is C; rolling is Cython |
| OBV | `np.sign` + `cumsum` | Fully vectorised, no loops |
| VWAP | pandas rolling sum | Same |
| MFI | pandas rolling sum | Same |
| Stochastic | pandas rolling min/max | Same |
| Portfolio build | `returns.values @ w` | BLAS sgemv |
| Portfolio covariance | `returns_df.cov() * 252` | BLAS syrk |
| Correlation matrix | `returns_df.corr()` | BLAS |
| PCA / SVD | `np.linalg.svd` | LAPACK dgesdd |
| Rolling beta | pandas `rolling.cov / var` | Pandas incremental algorithm |
| VaR / CVaR | `np.percentile` + masking | NumPy is C |
| Sharpe / Sortino / Calmar | scalar arithmetic on numpy arrays | Trivially vectorised |
| Screener | `asyncio.gather` + yfinance | I/O bound, not CPU bound |
| Parquet cache | pyarrow | Already compiled |
| Data provider | HTTP + yfinance | Network I/O dominant |

---

## Aggregate Speedup Estimates by Use Case

| Agent Tool / Workflow | Dominant bottleneck | Expected end-to-end speedup |
|---|---|---|
| `run_regime_adaptive_backtest` | `hurst_exponent` + `backtest_grid` | **10–30×** |
| `scan_pairs` (100 tickers) | cointegration ADF loop | **12–20×** |
| `run_walk_forward_backtest` | repeated `backtest_grid` calls | **5–15×** |
| `get_technical_analysis` | RSI + ADX + PSAR (Python fallback) | **10–20×** |
| `run_screener` (S&P 500) | RSI + beta per ticker × 500 | **5–15×** (compute path only; I/O still dominates) |
| `run_sma_backtest` | already fast | **3–5×** |

---

## Implementation Strategy

### Recommended approach: pybind11 + scikit-build-core

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

### Suggested module structure

```
src/
  standard_quant_tools/
    _cpp/                      ← C++ source root
      hurst.cpp                ← DFA, R/S, rolling Hurst
      indicators.cpp           ← RSI, ADX, PSAR
      cointegration.cpp        ← ADF test, OLS-2var
      backtest.cpp             ← run_strategy kernel
      bindings.cpp             ← pybind11 PYBIND11_MODULE block
    _sqt_core.pyd              ← compiled extension (generated)
```

The Python modules in `analysis/`, `indicators/`, `backtest/` would call `from standard_quant_tools import _sqt_core` and fall back to pure Python if the compiled extension is not present — preserving the library's current "optional dependency" design philosophy.

### Build

```toml
# pyproject.toml addition
[build-system]
requires = ["scikit-build-core", "pybind11"]
build-backend = "scikit_build_core.build"
```

A `CMakeLists.txt` at the root handles compiler flags (`-O3 -march=native` on GCC/Clang, `/O2 /arch:AVX2` on MSVC) and links the extension.

---

## Priority Order for Implementation

1. **`hurst_exponent` + `rolling_hurst`** — highest absolute speedup, no external dependencies to replace, self-contained algorithm.
2. **RSI, ADX, PSAR** — fixes the Numba-NumPy incompatibility permanently; high call frequency in screener and technical analysis tools.
3. **ADF test / `scan_pairs`** — replaces statsmodels dependency with a well-understood 50-line algorithm; unlocks large-universe pair scanning.
4. **`run_strategy` kernel** — makes backtest_grid faster without subprocess overhead; enables single-process grid search at full speed.
5. **Tiny 2-variable OLS** — small standalone utility; high call frequency when embedded in Hurst and cointegration; gains fold into #1 and #3.

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
| Windows build toolchain (MSVC vs MinGW) | Use `scikit-build-core`; test with both compilers in CI |
| Floating-point result divergence from pandas fallback | Unit test C++ output against Python reference implementation with `atol=1e-10` |
| Maintenance burden (dual Python + C++ paths) | Keep Python fallback; C++ path is additive, not a replacement |
| NumPy ABI changes (same problem as Numba) | Pin to `numpy>=2.0` ABI stable tag in the extension; re-test on each numpy major bump |
| Debugging (C++ segfault inside Python) | Develop with address sanitiser (`-fsanitize=address`) in debug builds |
