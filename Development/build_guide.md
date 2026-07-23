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

   **Option C — Visual Studio generator** *(works from any terminal)*
   Use the VS CMake generator, which finds MSVC without needing it in `PATH`:
   ```
   cmake -B build -G "Visual Studio 17 2022" -A x64
   cmake --build build --config Release
   ```

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
pytest tests/test_cpp_hurst.py -v            # Hurst + rolling Hurst
pytest tests/test_cpp_indicators.py -v       # RSI, ADX, Parabolic SAR, Wilder's ATR
pytest tests/test_cpp_new_indicators.py -v   # Bollinger Bands, Stochastic Oscillator
pytest tests/test_cpp_cointegration.py -v    # Engle-Granger cointegration + OLS
pytest tests/test_cpp_backtest.py -v         # run_strategy + batch_run_strategy kernels
pytest tests/test_cpp_regression.py -v       # rolling_beta, rolling_factor_loadings
```

Or run all six at once:

```
pytest tests/test_cpp_hurst.py tests/test_cpp_indicators.py tests/test_cpp_new_indicators.py tests/test_cpp_cointegration.py tests/test_cpp_backtest.py tests/test_cpp_regression.py -v
```

Once the extension is built all skipped tests activate (145 total across the
six files above — 22 Hurst, 36 indicators, 23 new-indicators, 23 cointegration,
25 backtest, 16 regression — counted from each file's `@requires_cpp` markers).

### C++ unit tests

```
cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --config Release -V
```

This runs four test suites: `cpp_hurst`, `cpp_indicators`, `cpp_cointegration`, `cpp_backtest`.

Or run each binary directly:

```
# Windows (VS generator)
build\tests\cpp\Release\test_hurst.exe
build\tests\cpp\Release\test_indicators.exe
build\tests\cpp\Release\test_cointegration.exe
build\tests\cpp\Release\test_backtest.exe

# Windows (Ninja) / Linux / macOS
./build/tests/cpp/test_hurst
./build/tests/cpp/test_indicators
./build/tests/cpp/test_cointegration
./build/tests/cpp/test_backtest
```

Each binary prints its own pass count on exit, e.g.:

```
N / N tests passed.   ← test_hurst
N / N tests passed.   ← test_indicators
N / N tests passed.   ← test_cointegration
N / N tests passed.   ← test_backtest
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
│           ├── CMakeLists.txt               ← Extension build rules
│           ├── include/sqt/
│           │   ├── hurst.hpp                ← Hurst exponent API
│           │   ├── indicators.hpp           ← RSI / ADX / PSAR / Wilder ATR / Bollinger / Stochastic API
│           │   ├── cointegration.hpp        ← OLS / ADF / Engle-Granger API
│           │   ├── backtest.hpp             ← run_strategy / batch_run_strategy kernel API
│           │   └── rolling_regression.hpp   ← rolling_beta / rolling_factor_loadings API
│           ├── src/
│           │   ├── hurst.cpp                ← Hurst implementation
│           │   ├── indicators.cpp           ← RSI / ADX / PSAR / Wilder ATR / Bollinger / Stochastic implementation
│           │   ├── cointegration.cpp        ← OLS / ADF / cointegration implementation
│           │   ├── backtest.cpp             ← backtest + batch grid kernel implementation
│           │   └── rolling_regression.cpp   ← incremental rolling beta / factor loadings implementation
│           └── bindings/
│               └── bindings.cpp             ← pybind11 module definition (all features)
└── tests/
    ├── test_cpp_hurst.py                    ← Python integration tests (Hurst)
    ├── test_cpp_indicators.py               ← Python integration tests (RSI/ADX/PSAR/ATR)
    ├── test_cpp_new_indicators.py           ← Python integration tests (Bollinger/Stochastic)
    ├── test_cpp_cointegration.py            ← Python integration tests (cointegration+OLS)
    ├── test_cpp_backtest.py                 ← Python integration tests (backtest + batch kernel)
    ├── test_cpp_regression.py               ← Python integration tests (rolling beta/factor loadings)
    └── cpp/
        ├── CMakeLists.txt                   ← C++ test build rules
        ├── test_hurst.cpp                   ← 19 C++ unit tests (no framework needed)
        ├── test_indicators.cpp              ← 24 C++ unit tests
        ├── test_cointegration.cpp           ← 18 C++ unit tests
        ├── test_backtest.cpp                ← 17 C++ unit tests
        ├── bench_hurst.cpp                  ← Hurst timing benchmark (run manually)
        └── bench_backtest.cpp               ← Backtest kernel timing benchmark (run manually)
```

---

## 7. What Is Currently in `_sqt_core`

| Feature | Header | Source | Python caller |
|---|---|---|---|
| Hurst exponent + rolling Hurst | `hurst.hpp` | `hurst.cpp` | `analysis/hurst.py` |
| RSI (Wilder's smoothing) | `indicators.hpp` | `indicators.cpp` | `indicators/momentum.py` |
| ADX + DI+/DI− | `indicators.hpp` | `indicators.cpp` | `indicators/trend.py` |
| Parabolic SAR | `indicators.hpp` | `indicators.cpp` | `indicators/trend.py` |
| Wilder's ATR (SMA seed + Wilder's smooth) | `indicators.hpp` | `indicators.cpp` | `indicators/volatility.py` |
| Bollinger Bands (fused Σx/Σx² pass) | `indicators.hpp` | `indicators.cpp` | `indicators/volatility.py` |
| Stochastic Oscillator (fused min+max pass) | `indicators.hpp` | `indicators.cpp` | `indicators/momentum.py` |
| 2-variable OLS (`calculate_beta`, `half_life`, `compute_spread`) | `cointegration.hpp` | `cointegration.cpp` | `analysis/regression.py`, `analysis/cointegration.py` |
| Engle-Granger cointegration (OLS + ADF + MacKinnon 2010) | `cointegration.hpp` | `cointegration.cpp` | `analysis/cointegration.py` |
| Backtest kernel (`run_strategy` — returns, equity, all metrics, trade stats) | `backtest.hpp` | `backtest.cpp` | `backtest/engine.py` |
| Batch backtest grid kernel (`batch_run_strategy`) | `backtest.hpp` | `backtest.cpp` | `backtest/engine.py` |
| Rolling beta (incremental sum updates) | `rolling_regression.hpp` | `rolling_regression.cpp` | `analysis/regression.py` |
| Rolling factor loadings (incremental Cholesky) | `rolling_regression.hpp` | `rolling_regression.cpp` | `analysis/multi_factor.py` |

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

### Case A — New standalone module (e.g. `kalman_filter`, `garch`)

*Example: `backtest.cpp` — the `run_strategy` kernel was added this way.*

1. `_cpp/include/sqt/my_feature.hpp` — C++ API declarations
2. `_cpp/src/my_feature.cpp` — implementation
3. `_cpp/bindings/bindings.cpp` — add `#include "sqt/my_feature.hpp"` and `m.def(...)` inside `PYBIND11_MODULE`
4. `_cpp/CMakeLists.txt` — add `src/my_feature.cpp` to `SQT_SOURCES`
5. `tests/cpp/CMakeLists.txt` — add static lib + test executable + `add_test`
6. `tests/cpp/test_my_feature.cpp` — C++ unit tests (same `CHECK` / `CHECK_NEAR` pattern as existing files)
7. `tests/test_cpp_my_feature.py` — Python integration tests (use `requires_cpp` skip marker)
8. The relevant Python module — add `_cpp_core: Any = None` guard and fast path

### Case B — Extending an existing module (e.g. adding a function to `indicators.cpp`)

*Example: Wilder's ATR was added to `indicators.hpp` / `indicators.cpp` without creating new files.*

1. `_cpp/include/sqt/indicators.hpp` — add declaration
2. `_cpp/src/indicators.cpp` — add implementation
3. `_cpp/bindings/bindings.cpp` — add `m.def(...)` for the new function (no new `#include` needed)
4. `tests/cpp/test_indicators.cpp` — add test functions and call them in `main()`
5. `tests/test_cpp_indicators.py` — add `TestCppNew` + `TestNewWrapper` test classes
6. The relevant Python module — add `_cpp_core` guard if not present and add fast path

No changes to `CMakeLists.txt` are needed when extending an existing `.cpp` file.

Rebuild:
```
cmake --build build --config Release
```

---

## 9. Notes

**`-march=native` / `/arch:AVX2`**  
Both flags tune the binary for the CPU of the build machine. They are ideal
for local development but produce a binary that may crash on machines without
the same SIMD support. For distributable wheels (PyPI), replace with:
- Linux/macOS: `-march=x86-64-v2` (SSE4.2, widely supported) or omit for baseline
- Windows: `/arch:SSE2` or omit `/arch:AVX2`

**Extension suffix**  
Python automatically picks up the correct suffix
(`.pyd`, `.so`, `.cpython-*.so`) via the import system. No code changes are
needed across platforms.

**Editable installs**  
`pip install -e .` installs the pure-Python package via flit_core. The C++
extension is built separately with cmake and lands in the same directory, so
both are always importable together after a single `pip install -e .` +
`cmake --build build --config Release`.
