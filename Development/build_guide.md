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

```
pytest tests/test_cpp_hurst.py -v
```

The `TestCppBindings` and `TestCppVsPython` groups are automatically skipped
when the extension is not built. Once built, all 48 tests run.

### C++ unit tests

```
cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build --config Release -V
```

Or run the binary directly:

```
# Windows (VS generator)
build\tests\cpp\Release\test_hurst.exe

# Windows (Ninja) / Linux / macOS
./build/tests/cpp/test_hurst
```

Expected:
```
19 / 19 tests passed.
```

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
│           │   └── hurst.hpp                ← Public C++ API
│           ├── src/
│           │   └── hurst.cpp                ← Implementation
│           └── bindings/
│               └── bindings.cpp             ← pybind11 module definition
└── tests/
    ├── test_cpp_hurst.py                    ← Python integration tests
    └── cpp/
        ├── CMakeLists.txt                   ← C++ test build rules
        └── test_hurst.cpp                   ← C++ unit tests (no framework needed)
```

---

## 7. Adding the Next C++ Feature

Each new module follows the same pattern (example: `indicators`):

1. `_cpp/include/sqt/indicators.hpp` — C++ API declarations
2. `_cpp/src/indicators.cpp` — implementation
3. `_cpp/bindings/bindings.cpp` — add `#include "sqt/indicators.hpp"` and `m.def(...)` calls
4. `_cpp/CMakeLists.txt` — add `src/indicators.cpp` to `SQT_SOURCES`
5. `tests/cpp/test_hurst.cpp` or a new `tests/cpp/test_indicators.cpp` — C++ unit tests
6. `tests/test_cpp_indicators.py` — Python integration tests
7. `indicators/momentum.py` (or whichever Python module) — add C++ fast path

Rebuild:
```
cmake --build build --config Release
```

---

## 8. Notes

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
