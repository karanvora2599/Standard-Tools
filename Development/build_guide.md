# C++ Extension Build Guide

## How it works

The C++ extension (`_sqt_core`) is compiled with **CMake + pybind11**.  
The compiled binary is dropped directly into the Python package directory so
`from standard_quant_tools import _sqt_core` works from an editable install
without any extra install step.

The Python modules automatically fall back to pure Python when the extension
is not built — all existing tests continue to pass either way.

---

## 1. Prerequisites

Root `CMakeLists.txt` requires CMake **>= 3.15**. `pyproject.toml` requires
Python **>= 3.10**. Both are still accurate as of this writing — no change
needed to build against the current `pyproject.toml`.

### Python packages (all platforms)

```
pip install pybind11
```

`cmake` is also needed. Install it via pip if your system doesn't have it:

```
pip install cmake ninja
```

---

### Platform-specific compiler setup

#### Windows

Python on this machine was compiled with MSVC, so extensions must also be
compiled with MSVC.

1. Download **Build Tools for Visual Studio 2022** (free):
   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

2. Run the installer. Select the **"Desktop development with C++"** workload.
   Required components:
   - MSVC v143 build tools (C++ compiler)
   - Windows 11 SDK (or Windows 10 SDK)
   - C++ CMake tools for Windows (optional — you can use your own CMake)

3. After installation, choose **one** of these approaches for every build session:

   **Option A — x64 Native Tools Command Prompt** *(recommended for local development)*
   Open "x64 Native Tools Command Prompt for VS 2022" from the Start Menu.
   MSVC (`cl.exe`) is already in `PATH`.

   **Option B — Developer PowerShell**
   Open "Developer PowerShell for VS 2022" from the Start Menu.
   Same as Option A but in PowerShell.

   **Option C — Visual Studio generator** *(works from any terminal, in theory)*
   Use the VS CMake generator, which finds MSVC without needing it in `PATH`:
   ```
   cmake -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```
   **In practice, this failed** (`No CMAKE_CXX_COMPILER could be found`) on a
   standalone "Build Tools for Visual Studio 2022" install (no full VS IDE) —
   the generator's own compiler-discovery mechanism didn't find `cl.exe` even
   though it was genuinely present and `vcvarsall.bat` found it fine. If you
   hit this, use Option A/B instead, or activate the environment manually and
   use the Ninja generator from Section 2 — that combination is confirmed
   working on exactly this kind of install (see the troubleshooting note
   immediately below for a real gotcha this can also surface).

**Troubleshooting: `cl.exe` found, but linking fails with an RC error**
If `cmake --build` gets past compiling (`.obj` files build fine) but fails at
the link step with something like `RC Pass 1: command "rc /fo ..." failed...
no such file or directory`, the MSVC **compiler** is installed but the
**Windows SDK** (which provides `rc.exe`/`mt.exe`, needed for linking any
Windows binary, not just ones with actual `.rc` resource files) is not —
confirmed by an empty `Windows Kits\10\bin\` directory (or the directory not
existing at all). This is a real gap the "Desktop development with C++"
workload's checkbox list in step 2 doesn't always guarantee gets installed.
Fix: re-run the Visual Studio Installer and add the SDK component explicitly
(swap the version ID for whatever `vswhere.exe`/the installer UI shows as
available):
```
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe" modify ^
  --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" ^
  --add Microsoft.VisualStudio.Component.Windows11SDK.22621 ^
  --quiet --norestart
```
This needs elevation (run from an elevated prompt, or via PowerShell's
`Start-Process -Verb RunAs` if scripting it) — a non-elevated invocation
exits with code 87 and no other explanation. Verify it worked with
`Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin"` — you should see
at least one version-numbered subdirectory containing `rc.exe`.

---

#### macOS

Install the Xcode Command Line Tools (includes `clang++`):

```
xcode-select --install
```

CMake can be installed via Homebrew or pip:

```
brew install cmake        # via Homebrew
# OR
pip install cmake ninja   # via pip (no Homebrew required)
```

No special shell setup needed — build from any terminal.

---

#### Linux (Debian / Ubuntu)

```
sudo apt update
sudo apt install build-essential cmake ninja-build
```

For other distributions:

```
# Fedora / RHEL
sudo dnf install gcc-c++ cmake ninja-build

# Arch
sudo pacman -S base-devel cmake ninja
```

No special shell setup needed.

---

## 2. Build

The cmake commands are **identical on all platforms** once the compiler is
in `PATH` (see platform notes above).

> **Do not pass `-DCMAKE_CXX_FLAGS=...` to add warning flags.** CMake
> *replaces* the variable rather than appending to it, so the project's own
> flag set is discarded — on MSVC that silently drops `/EHsc`, which governs
> exception-unwinding semantics. The build then succeeds, links, and emits
> only warning C4530 ("C++ exception handler used, but unwind semantics are
> not enabled"), which reads like noise. The failure appears much later, as a
> **Windows access violation** the first time a kernel throws across the
> pybind11 boundary — for example `rolling_hurst`'s negative-SSE guard or any
> `ValidationError` raised from native code. This is a real trap, hit while
> auditing this project; nothing about the build output points at the cause.
>
> Two related hazards with the same root:
>
> - **Every configure directory writes to the same output location.**
>   `LIBRARY_OUTPUT_DIRECTORY` is `src/standard_quant_tools/`, so
>   `cmake -B build_something_else` will *overwrite* the extension your main
>   `build/` produced. If you need a second configuration (a warnings audit,
>   a sanitizer build), redirect its output or expect to rebuild `build/`
>   afterwards to restore a good `.pyd`.
> - **If you suspect a stale or bad extension**, delete
>   `src/standard_quant_tools/_sqt_core*.pyd` and run `cmake --build build`
>   again. Ninja tracks its own outputs, so a file replaced by a *different*
>   build directory will not always be detected as dirty.
>
> To add warning flags safely, use a toolchain file or
> `target_compile_options` on the target, both of which compose with the
> existing flags instead of replacing them.

### Standard build (all platforms)

```
cd "path/to/Standard Tools"

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Windows — Visual Studio generator (no developer prompt required)

```
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Ninja everywhere (fastest incremental builds)

Requires `ninja` in `PATH` and MSVC in `PATH` on Windows:

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

The compiled extension is written directly to the package directory:

| Platform | File |
|----------|------|
| Windows  | `src/standard_quant_tools/_sqt_core.pyd` |
| Linux    | `src/standard_quant_tools/_sqt_core.cpython-3XX-x86_64-linux-gnu.so` |
| macOS    | `src/standard_quant_tools/_sqt_core.cpython-3XX-darwin.so` |

No install step is needed.

---

## 3. Verify

```
python -c "from standard_quant_tools import _sqt_core; print('OK:', _sqt_core.__doc__[:50])"
```

---

## 4. Run Tests

### Python tests (always available — uses fallback if extension not built)

```
pytest tests/ -v
```

### Python tests for the C++ bindings

Each C++ feature has a matching Python integration test file. Tests that require
the compiled extension are automatically skipped when it is not built.

```
pytest tests/cpp_bindings/test_cpp_hurst.py -v                  # Hurst + rolling Hurst
pytest tests/cpp_bindings/test_cpp_indicators.py -v             # RSI, ADX, Parabolic SAR, Wilder's ATR
pytest tests/cpp_bindings/test_cpp_new_indicators.py -v         # Bollinger Bands, Stochastic Oscillator, fused technical_indicators()
pytest tests/cpp_bindings/test_cpp_cointegration.py -v          # Engle-Granger cointegration + OLS
pytest tests/cpp_bindings/test_cpp_backtest.py -v               # run_strategy + batch_run_strategy kernels
pytest tests/cpp_bindings/test_cpp_regression.py -v             # rolling_beta (incl. AVX2 dispatch), rolling_factor_loadings
pytest tests/cpp_bindings/test_cpp_monte_carlo.py -v            # simulate_forward_paths
pytest tests/cpp_bindings/test_cpp_garch.py -v                  # garch11_variance_recursion, fused NLL + analytic gradient
pytest tests/cpp_bindings/test_cpp_signals.py -v                # kalman_filter_1state/2state, donchian/vwap-reversion state machines
pytest tests/cpp_bindings/test_cpp_array1d_validation.py -v     # 1-D array validation across every Array1D binding
pytest tests/cpp_bindings/test_cpp_gil_release.py -v            # GIL is actually released around every pure-C++ kernel call
```

Or run all eleven at once:

```
pytest tests/cpp_bindings/test_cpp_hurst.py tests/cpp_bindings/test_cpp_indicators.py tests/cpp_bindings/test_cpp_new_indicators.py tests/cpp_bindings/test_cpp_cointegration.py tests/cpp_bindings/test_cpp_backtest.py tests/cpp_bindings/test_cpp_regression.py tests/cpp_bindings/test_cpp_monte_carlo.py tests/cpp_bindings/test_cpp_garch.py tests/cpp_bindings/test_cpp_signals.py tests/cpp_bindings/test_cpp_array1d_validation.py tests/cpp_bindings/test_cpp_gil_release.py -v
```

Once the extension is built all skipped tests activate — run the suite to
see the current numbers rather than trusting a hardcoded count here; it
grows as tests are added.

A separate gated test class outside these files, `TestNativeTradeStatsCorrectness`
in `tests/backtest/test_backtest.py`, verifies `run_strategy`'s and `batch_run_strategy`'s
native trade-log accounting against hand-computed values once `_sqt_core`
is built — see `Development/performance_insights.md` for the trade-stat
parity background.

### C++ unit tests

```
cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --config Release -V
```

This runs eight test suites: `cpp_hurst`, `cpp_indicators`, `cpp_cointegration`,
`cpp_backtest`, `cpp_monte_carlo`, `cpp_garch`, `cpp_signals`, `cpp_rolling_regression`.

Or run each binary directly:

```
# Windows (VS generator)
build\tests\cpp\Release\test_hurst.exe
build\tests\cpp\Release\test_indicators.exe
build\tests\cpp\Release\test_cointegration.exe
build\tests\cpp\Release\test_backtest.exe
build\tests\cpp\Release\test_monte_carlo.exe
build\tests\cpp\Release\test_garch.exe
build\tests\cpp\Release\test_signals.exe
build\tests\cpp\Release\test_rolling_regression.exe

# Windows (Ninja) / Linux / macOS
./build/tests/cpp/test_hurst
./build/tests/cpp/test_indicators
./build/tests/cpp/test_cointegration
./build/tests/cpp/test_backtest
./build/tests/cpp/test_monte_carlo
./build/tests/cpp/test_garch
./build/tests/cpp/test_signals
./build/tests/cpp/test_rolling_regression
```

Each binary prints its own pass count on exit, e.g.:

```
N / N tests passed.   ← test_hurst
N / N tests passed.   ← test_indicators
N / N tests passed.   ← test_cointegration
N / N tests passed.   ← test_backtest
N / N tests passed.   ← test_monte_carlo
N / N tests passed.   ← test_garch
N / N tests passed.   ← test_signals
N / N tests passed.   ← test_rolling_regression
```

`N` grows as tests are added to `tests/cpp/test_*.cpp` — do not hardcode a
specific count here; run the suite to see the current numbers. A non-`N/N`
result (`M / N` with `M < N`) is a real failure, not a stale-doc mismatch.

### C++ performance benchmarks

Benchmarks are not CTest tests — run them manually to inspect timing output:

```
# Windows (VS generator)
build\tests\cpp\Release\bench_hurst.exe
build\tests\cpp\Release\bench_backtest.exe

# Windows (Ninja) / Linux / macOS
./build/tests/cpp/bench_hurst
./build/tests/cpp/bench_backtest
```

Each benchmark prints a table of median wall-clock times with conservative upper bounds. A failure indicates a debug build or missing optimisation flags — not a correctness problem.

---

## 5. Rebuilding After Code Changes

```
cmake --build build --config Release
```

CMake tracks source timestamps; only changed `.cpp` files are recompiled.

### Full clean rebuild

```
# remove the build directory and start over
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## 6. Project Structure Reference

```
Standard Tools/
├── CMakeLists.txt                           ← Root CMake entry point
├── src/
│   └── standard_quant_tools/
│       ├── _sqt_core.[pyd|so]               ← Compiled output (generated, gitignored)
│       └── _cpp/                            ← All C++ sources
│           ├── CMakeLists.txt               ← Extension build rules (LTO/IPO, PGO options, OpenMP, AVX2 file override)
│           ├── include/sqt/
│           │   ├── platform.hpp             ← SQT_RESTRICT portable qualifier macro
│           │   ├── isa_dispatch.hpp         ← Runtime CPUID feature detection (IsaFeatures{avx2,fma})
│           │   ├── hurst.hpp                ← Hurst exponent / rolling Hurst API
│           │   ├── indicators.hpp           ← RSI / ADX / PSAR / Wilder ATR / Bollinger / Stochastic + fused technical_indicators() API
│           │   ├── cointegration.hpp        ← OLS / ADF / Engle-Granger / Kalman (1-state, 2-state) API
│           │   ├── backtest.hpp             ← run_strategy / run_strategy_summary / batch_run_strategy kernel API
│           │   ├── rolling_regression.hpp   ← rolling_beta / rolling_factor_loadings API
│           │   ├── rolling_beta_avx2.hpp    ← AVX2+FMA rolling_beta reduction kernel API (internal, used only by rolling_regression.cpp)
│           │   ├── monte_carlo.hpp          ← simulate_forward_paths (moving-block bootstrap) API
│           │   ├── garch.hpp                ← GARCH(1,1) variance recursion + fused NLL/gradient API
│           │   └── signal_state_machines.hpp ← Donchian / VWAP-reversion signal hysteresis API
│           ├── src/
│           │   ├── isa_dispatch.cpp         ← CPUID detection + test-only override hook
│           │   ├── hurst.cpp                ← Hurst implementation (OpenMP across rolling windows, one-pass DFA)
│           │   ├── indicators.cpp           ← RSI / ADX / PSAR / Wilder ATR / Bollinger / Stochastic + technical_indicators() implementation
│           │   ├── cointegration.cpp        ← OLS / ADF / cointegration / Kalman filter implementation
│           │   ├── backtest.cpp             ← run_strategy / run_strategy_summary / batch_run_strategy (OpenMP across the grid) implementation
│           │   ├── rolling_regression.cpp   ← incremental rolling beta (+ AVX2 dispatch) / factor loadings implementation
│           │   ├── rolling_beta_avx2.cpp    ← AVX2+FMA kernel, its own translation unit (compiled with /arch:AVX2 unconditionally)
│           │   ├── monte_carlo.cpp          ← moving-block bootstrap, optional OpenMP parallel loop
│           │   ├── garch.cpp                ← GARCH(1,1) variance recursion + fused NLL/analytic-gradient implementation
│           │   └── signal_state_machines.cpp ← Donchian / VWAP-reversion hysteresis implementation
│           └── bindings/
│               └── bindings.cpp             ← pybind11 module definition (all features, direct-write NumPy buffers)
└── tests/
    ├── test_cpp_hurst.py                    ← Python integration tests (Hurst)
    ├── test_cpp_indicators.py               ← Python integration tests (RSI/ADX/PSAR/ATR)
    ├── test_cpp_new_indicators.py           ← Python integration tests (Bollinger/Stochastic, fused technical_indicators())
    ├── test_cpp_cointegration.py            ← Python integration tests (cointegration+OLS+Kalman)
    ├── test_cpp_backtest.py                 ← Python integration tests (backtest + batch kernel, array-based batch return)
    ├── test_cpp_regression.py               ← Python integration tests (rolling beta incl. AVX2 dispatch, rolling factor loadings)
    ├── test_cpp_monte_carlo.py              ← Python integration tests (Monte Carlo, statistical parity only)
    ├── test_cpp_garch.py                    ← Python integration tests (GARCH(1,1) recursion, fused NLL/gradient)
    ├── test_cpp_signals.py                  ← Python integration tests (Donchian/VWAP-reversion signals)
    ├── test_cpp_array1d_validation.py       ← Python integration tests (1-D array validation across every binding)
    ├── test_cpp_gil_release.py              ← Python integration tests (GIL actually released around pure-C++ calls)
    └── cpp/
        ├── CMakeLists.txt                   ← C++ test build rules
        ├── test_hurst.cpp                   ← 23 C++ unit tests (no framework needed)
        ├── test_indicators.cpp              ← 35 C++ unit tests
        ├── test_cointegration.cpp           ← 26 C++ unit tests (incl. Kalman 1-state/2-state)
        ├── test_backtest.cpp                ← 24 C++ unit tests (incl. run_strategy_summary/batch OpenMP reproducibility)
        ├── test_monte_carlo.cpp             ← 10 C++ unit tests (incl. thread-count independence)
        ├── test_garch.cpp                   ← 13 C++ unit tests (incl. analytic-gradient vs. numerical central differences)
        ├── test_signals.cpp                 ← 12 C++ unit tests
        ├── test_rolling_regression.cpp      ← 9 C++ unit tests (incl. AVX2-vs-scalar tolerance gate, forced-scalar-path test)
        ├── bench_hurst.cpp                  ← Hurst timing benchmark (run manually)
        └── bench_backtest.cpp               ← Backtest kernel timing benchmark (run manually)
```

---

## 7. What Is Currently in `_sqt_core`

| Feature | Header | Source | Python caller |
|---|---|---|---|
| Hurst exponent + rolling Hurst (OpenMP across windows, one-pass DFA reformulation) | `hurst.hpp` | `hurst.cpp` | `analysis/hurst.py` |
| RSI (Wilder's smoothing) | `indicators.hpp` | `indicators.cpp` | `indicators/momentum.py` |
| ADX + DI+/DI− (O(1) auxiliary memory) | `indicators.hpp` | `indicators.cpp` | `indicators/trend.py` |
| Parabolic SAR | `indicators.hpp` | `indicators.cpp` | `indicators/trend.py` |
| Wilder's ATR (SMA seed + Wilder's smooth) | `indicators.hpp` | `indicators.cpp` | `indicators/volatility.py` |
| Bollinger Bands (fused Σx/Σx² pass) | `indicators.hpp` | `indicators.cpp` | `indicators/volatility.py` |
| Stochastic Oscillator (fused min+max pass) | `indicators.hpp` | `indicators.cpp` | `indicators/momentum.py` |
| Fused `technical_indicators()` — RSI/ADX/ATR/Bollinger/Stochastic in one native call | `indicators.hpp` | `indicators.cpp` | `agent/tools.py`'s technical-analysis tool (additive fast path when ≥2 fusable indicators requested) |
| 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`) | `cointegration.hpp` | `cointegration.cpp` | `analysis/regression.py`, `analysis/cointegration.py` |
| Engle-Granger cointegration (OLS + ADF + MacKinnon 2010) | `cointegration.hpp` | `cointegration.cpp` | `analysis/cointegration.py` |
| Backtest kernel (`run_strategy` — equity curve + all 11 metrics) | `backtest.hpp` | `backtest.cpp` | `backtest/engine.py` |
| Allocation-free summary kernel (`run_strategy_summary` — same 11 metrics, zero heap allocation, no equity curve) | `backtest.hpp` | `backtest.cpp` | `backtest/engine.py` (internal — used by `batch_run_strategy`, not exposed to Python directly) |
| Batch backtest grid kernel (`batch_run_strategy` — `(num_tests, 11)` NumPy array, OpenMP across parameter combinations) | `backtest.hpp` | `backtest.cpp` | `backtest/engine.py` |
| Rolling beta (incremental sum updates + optional runtime AVX2+FMA dispatch) | `rolling_regression.hpp` | `rolling_regression.cpp`, `rolling_beta_avx2.cpp` | `analysis/regression.py` |
| Rolling factor loadings (incremental Cholesky, dead-triangle elimination, scratch-buffer reuse) | `rolling_regression.hpp` | `rolling_regression.cpp` | `analysis/multi_factor.py` |
| Monte Carlo forward simulation (moving-block bootstrap, optional OpenMP) | `monte_carlo.hpp` | `monte_carlo.cpp` | `backtest/monte_carlo.py` |
| GARCH(1,1) conditional variance recursion + fused NLL/analytic gradient | `garch.hpp` | `garch.cpp` | `analysis/garch.py` |
| Kalman filter, 1-state and 2-state (time-varying hedge ratio) | `cointegration.hpp` | `cointegration.cpp` | `analysis/cointegration.py` |
| Donchian breakout / VWAP-reversion signal hysteresis | `signal_state_machines.hpp` | `signal_state_machines.cpp` | `backtest/strategies.py` |

**`batch_run_strategy`'s return format changed** from a `py::list` of `py::dict` (one dict per grid combination) to a single `(num_tests, 11)` `py::array_t<double>` with a fixed column order (`_BATCH_METRIC_COLUMNS` in `engine.py`) — a direct C++-caller integration, not a public Python API most users touch directly (`backtest_grid()` still returns a `pd.DataFrame` either way).

**Monte Carlo RNG note:** the C++ path's RNG (splitmix64-derived per-path
seeding + `std::mt19937_64`) does **not** reproduce NumPy's PCG64 bit
stream. `random_seed` is only reproducible *within* one backend — the same
seed produces different concrete numbers depending on whether `_sqt_core`
is built, though repeat calls on the same backend are bit-identical.
`tests/cpp_bindings/test_cpp_monte_carlo.py` reflects this: same-backend reproducibility
is asserted exactly, but cross-backend comparisons use loose statistical
tolerance instead of the usual `atol=1e-10`.

**Monte Carlo OpenMP note:** `simulate_forward_paths`'s per-simulation loop
is optionally parallelized via `#pragma omp parallel for`, gated on
`SQT_HAS_OPENMP` (defined only if CMake's `find_package(OpenMP)` succeeds —
not `REQUIRED`, so a build without an OpenMP runtime, e.g. default Apple
Clang, still succeeds and just runs the identical loop serially). Each
simulated path is fully independent — its own per-thread RNG state derived
from the base seed and path index, no shared mutable state, no locking —
so this is safe by construction, not by careful scheduling. Verified by
`test_result_independent_of_thread_count` in both
`tests/cpp/test_monte_carlo.cpp` and `tests/cpp_bindings/test_cpp_monte_carlo.py`
(same seed + inputs must give bit-identical output whether forced to 1
thread or left unconstrained).

**Trade-stat parity (`run_strategy` vs. `batch_run_strategy`) — fix confirmed correct against a real compiled `_sqt_core`:**
`sqt::run_strategy`'s own trade-log logic in `backtest.cpp` used to record entry
one bar later than the true economic reference and exclude commission/slippage
from each trade's return — a real bug in the native kernel itself.
`backtest/engine.py`'s `run_strategy()` worked around it on the Python side: it
always discards the C++ kernel's own `win_rate`/`profit_factor`/`num_trades`/
`avg_trade_return_pct` and recomputes them in Python via
`_build_trade_log`/`_compute_trade_stats` — the same fill-aware, cost-aware
accounting used by the pure-Python path — so a caller gets identical trade
statistics whether or not `_sqt_core` is built. `backtest_grid()`'s C++ batch
path (`batch_run_strategy`) has no such override — rebuilding a Python-side
trade log per grid combination would defeat the point of the batch kernel's
speed — so it depends entirely on the native kernel's own accounting.

On 2026-07-24 (commit `2242d63`), `backtest.cpp`'s native trade-log
construction itself was rewritten to match `_build_trade_log`'s accounting
exactly (entry_size = signal magnitude rather than sign only, `prices[i-1]` as
the entry/exit reference price, commission+slippage deducted per completed
round trip). This applies to both `run_strategy` and `batch_run_strategy`,
since they share the same trade-log code in `backtest.cpp`.

**Status: confirmed correct.** `_sqt_core` has since been built for real and
`tests/backtest/test_backtest.py::TestNativeTradeStatsCorrectness` plus the full native
`ctest` suite were actually run — every native/Python parity check passed.
(Along the way, 4 of `tests/cpp/test_backtest.cpp`'s own hand-written
expectations turned out to be wrong, based on a mistaken `prices[i]`-vs-
`prices[i-1]` reference-price assumption unrelated to the fix being validated
— those were corrected too; see `Development/performance_insights.md`'s
Executive Summary for the full bug list.) `backtest/engine.py`'s Python-side
override for `run_strategy()` is still kept in place — it's a working safety
net, not a sign of remaining doubt — but a `batch_run_strategy` grid search
sorted by `win_rate`/`profit_factor` can now be treated as trustworthy against
a real compiled `_sqt_core`, not merely "unverified but probably fine."

All Python callers follow the same guard pattern:

```python
from typing import Any
_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import _sqt_core as _cpp_core
    HAS_CPP = True
except ImportError:
    pass

# in the function body:
if HAS_CPP and _cpp_core is not None:
    return _cpp_core.feature_name(...)
# fallback:
...
```

---

## 8. Adding the Next C++ Feature

There are two cases:

### Case A — New standalone module (e.g. a new `.hpp`/`.cpp` pair)

*Examples: `backtest.cpp` (the `run_strategy` kernel), `monte_carlo.cpp`,
`garch.cpp`, and `signal_state_machines.cpp` were all added this way.*

1. `_cpp/include/sqt/my_feature.hpp` — C++ API declarations
2. `_cpp/src/my_feature.cpp` — implementation
3. `_cpp/bindings/bindings.cpp` — add `#include "sqt/my_feature.hpp"` and `m.def(...)` inside `PYBIND11_MODULE`
4. `_cpp/CMakeLists.txt` — add `src/my_feature.cpp` to `SQT_SOURCES`
5. `tests/cpp/CMakeLists.txt` — add static lib + test executable + `add_test`
6. `tests/cpp/test_my_feature.cpp` — C++ unit tests (same `CHECK` / `CHECK_NEAR` pattern as existing files)
7. `tests/cpp_bindings/test_cpp_my_feature.py` — Python integration tests (use `requires_cpp` skip marker). Note the two directories: `tests/cpp/` holds the C++ gtest sources CMake compiles, `tests/cpp_bindings/` the Python-side parity tests pytest collects.
8. The relevant Python module — add `_cpp_core: Any = None` guard and fast path

### Case B — Extending an existing module (e.g. adding a function to `indicators.cpp`)

*Examples: Wilder's ATR was added to `indicators.hpp`/`indicators.cpp`, and
the Kalman filter (1-state/2-state) was added to `cointegration.hpp`/
`cointegration.cpp` — both without creating new files.*

1. `_cpp/include/sqt/indicators.hpp` — add declaration
2. `_cpp/src/indicators.cpp` — add implementation
3. `_cpp/bindings/bindings.cpp` — add `m.def(...)` for the new function (no new `#include` needed)
4. `tests/cpp/test_indicators.cpp` — add test functions and call them in `main()`
5. `tests/cpp_bindings/test_cpp_indicators.py` — add `TestCppNew` + `TestNewWrapper` test classes
6. The relevant Python module — add `_cpp_core` guard if not present and add fast path

No changes to `CMakeLists.txt` are needed when extending an existing `.cpp` file.

Rebuild:
```
cmake --build build --config Release
```

---

## 9. Notes

**`-march=native` / `/arch:AVX2` (opt-in via `SQT_NATIVE_ARCH`)**  
Both flags tune the binary for the exact CPU of the build machine — fine for
local development, but the resulting binary can crash with an illegal-
instruction fault on a different/older CPU lacking those ISA extensions. This
is why the default build (`cmake -B build ...` with no extra flags, including
what CI uses) does **not** enable them: `SQT_NATIVE_ARCH` defaults to `OFF`,
so a fresh clone always produces portable codegen. Opt in explicitly for
local speed:
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSQT_NATIVE_ARCH=ON
```
This session's own measured benchmarks in `Development/performance_insights.md`
were built with `SQT_NATIVE_ARCH=ON`. For a distributable wheel (PyPI), leave
it off (the default) rather than substituting a manual baseline flag.

**Extension suffix**  
Python automatically picks up the correct suffix
(`.pyd`, `.so`, `.cpython-*.so`) via the import system. No code changes are
needed across platforms.

**Editable installs**  
`pip install -e .` installs the pure-Python package via flit_core. The C++
extension is built separately with cmake and lands in the same directory, so
both are always importable together after a single `pip install -e .` +
`cmake --build build --config Release`.

**OpenMP (optional)**  
`_cpp/CMakeLists.txt` calls `find_package(OpenMP)` (not `REQUIRED`) to
parallelize `monte_carlo.cpp`'s `simulate_forward_paths` loop and
`backtest.cpp`'s `batch_run_strategy` loop. Linux (`libgomp`, ships with
`build-essential`/`gcc`) and Windows (MSVC's built-in `/openmp` support)
pick this up automatically with no extra install step. Default Apple Clang
on macOS ships no OpenMP support — the build still succeeds either way
(`SQT_HAS_OPENMP` just won't be defined, and the affected loops run their
identical serial fallback). To get the parallel path on macOS, install
LLVM's OpenMP runtime (`brew install libomp`) before configuring.

**PGO (Profile-Guided Optimization, opt-in, local-only, `SQT_PGO_GENERATE`/`SQT_PGO_USE`)**  
Same "opt-in for local max speed, off by default" philosophy as
`SQT_NATIVE_ARCH` — but PGO is a **two-step workflow**, not a single build
flag: you build an instrumented binary, run it against a representative
workload to collect a profile, then rebuild using that profile. Both
options default `OFF` and are mutually exclusive (a `FATAL_ERROR` if both
are set at once). **Deliberately not wired into any CI workflow** — a
simple `cmake -B build && cmake --build build` pipeline has no natural
place for the extra training run between the two builds, and the profile
itself is workload- and machine-specific (a profile trained on one
machine's realistic data isn't guaranteed to transfer cleanly to another).

⚠️ **Every CMake build directory in this repo writes `_sqt_core` to the
same absolute package path** (`src/standard_quant_tools/`), regardless of
which build directory produced it — there's no per-build-dir isolation of
the *output*, only of intermediate object files. Building an instrumented
or PGO-optimized extension **overwrites your normal working extension** in
place. Use a separate build directory for PGO experiments
(`build-pgo` below) and rebuild your normal `build/` directory afterward
to restore it — don't assume the two build dirs are independent just
because their *names* differ.

Step 1 — instrumented build:
```
cmake -B build-pgo -DCMAKE_BUILD_TYPE=Release -DSQT_PGO_GENERATE=ON
cmake --build build-pgo --config Release
```

Step 2 — train it. Run a workload that's representative of real usage
across the functions that matter most — the existing benchmark binaries are
a reasonable starting point, but for a real profile also exercise the
Python-level call paths (`run_strategy`, `batch_run_strategy`,
`rolling_hurst`, `rolling_factor_loadings`, `rolling_beta`,
`simulate_forward_paths`, the technical indicators) across realistic
size/parameter ranges, not just the benchmark binaries' own fixed inputs:
```
./build-pgo/tests/cpp/bench_backtest    # or .exe on Windows
./build-pgo/tests/cpp/bench_hurst
python -c "
from standard_quant_tools import _sqt_core as c
import numpy as np
rng = np.random.default_rng(0)
prices = 100 + np.cumsum(rng.normal(0, 1, 2000))
signals = rng.choice([-1.0, 0.0, 1.0], size=(2000, 2000))
c.batch_run_strategy(prices, signals, 10000.0, 0.001, 0.0005)
c.rolling_hurst(rng.normal(0, 1, 2000), 200, 1, 'dfa', 10)
"
```
MSVC writes profile data to a `.pgd` file next to the `.pyd`/import
library in the build tree (merged automatically across runs by the
`/LTCG:PGInstrument` runtime); GCC/Clang write `.gcda` files next to each
translation unit's object file, merged automatically when the same build
tree is reused for step 3.

Step 3 — optimized rebuild using the collected profile:
```
cmake -B build-pgo -DCMAKE_BUILD_TYPE=Release -DSQT_PGO_USE=ON
cmake --build build-pgo --config Release
```

Then restore your normal working extension:
```
cmake --build build --config Release
```
