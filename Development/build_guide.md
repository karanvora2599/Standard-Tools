# C++ Extension Build Guide

## Prerequisites

### 1 — Install Visual Studio Build Tools 2022

The Python on this machine (`MSC v.1937 64-bit`) was compiled with MSVC, so extensions must also be built with MSVC.

1. Download **Build Tools for Visual Studio 2022** (free):  
   https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
2. Run the installer, select the **"Desktop development with C++"** workload.  
   Minimum required components:
   - MSVC v143 build tools
   - Windows 11 SDK (or Windows 10 SDK)

### 2 — Install Python build dependencies

```
pip install pybind11 scikit-build-core
```

Both are already installed on this machine.

---

## Building the Extension

Open **"x64 Native Tools Command Prompt for VS 2022"** (installed with Build Tools — search in Start Menu), then navigate to the project root:

```cmd
cd "C:\Users\karan\Documents\Projects\Standard Tools"
```

### Configure and build

```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The compiled extension (`_sqt_core.pyd`) is placed directly in:

```
src\standard_quant_tools\_sqt_core.pyd
```

No install step is needed — Python finds it there automatically.

### Verify the build

```
python -c "from standard_quant_tools import _sqt_core; print('OK', _sqt_core.__doc__[:40])"
```

---

## Running Tests

### Python tests (always available — uses fallback if extension not built)

```
pytest tests/ -v
```

### Python tests that specifically test the C++ bindings

```
pytest tests/test_cpp_hurst.py -v
```

Tests in `TestCppBindings` and `TestCppVsPython` are automatically skipped if the extension is not built.

### C++ unit tests (requires extension to be built first)

```cmd
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSQT_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --config Release -V
```

Or run the binary directly:

```cmd
build\tests\cpp\Release\test_hurst.exe
```

Expected output:
```
19 / 19 tests passed.
```

---

## Rebuilding After Code Changes

```cmd
cmake --build build --config Release
```

CMake tracks source timestamps; only changed files are recompiled.

---

## Cleaning the Build

```cmd
rmdir /s /q build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Project Structure Reference

```
Standard Tools/
├── CMakeLists.txt                          ← Root CMake (entry point)
├── src/standard_quant_tools/
│   ├── _sqt_core.pyd                       ← Compiled output (generated)
│   └── _cpp/                               ← All C++ sources
│       ├── CMakeLists.txt                  ← Extension build rules
│       ├── include/sqt/
│       │   └── hurst.hpp                   ← Public C++ API headers
│       ├── src/
│       │   └── hurst.cpp                   ← Implementation
│       └── bindings/
│           └── bindings.cpp                ← pybind11 module definition
└── tests/
    ├── test_cpp_hurst.py                   ← Python integration tests
    └── cpp/
        ├── CMakeLists.txt                  ← C++ test build rules
        └── test_hurst.cpp                  ← C++ unit tests (no framework)
```

---

## Adding the Next C++ Feature

When adding a new module (e.g., indicators):

1. Add `include/sqt/indicators.hpp` — declare the C++ API
2. Add `src/indicators.cpp` — implement it
3. In `bindings/bindings.cpp` — add `#include "sqt/indicators.hpp"` and `m.def(...)` calls
4. In `_cpp/CMakeLists.txt` — add `src/indicators.cpp` to `SQT_SOURCES`
5. In `tests/cpp/test_hurst.cpp` — add a new test executable or extend existing one
6. Add `tests/test_cpp_indicators.py` — Python integration tests
7. Update the Python module (`indicators/momentum.py` etc.) with a C++ fast path

Rebuild: `cmake --build build --config Release`
