# Standard Quant Tools for AI Agents

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

A high-performance, modular Python library for quantitative financial analysis. Designed to give AI agents and automated workflows **clean structured data**, **mathematical accuracy**, and **robust error handling**.

Maintained by [Karan Vora](mailto:kv2154@nyu.edu). Source: [github.com/karanvora2599/Standard-Tools](https://github.com/karanvora2599/Standard-Tools).

## Key Features
- **High performance** — an optional C++ extension (`_sqt_core`) accelerates the CPU-bound paths, behind an identical API with an automatic pure-Python fallback. Headline measured figures: rolling Hurst **274×**, Engle-Granger cointegration **23–86×**, a 2 000-name pair scan **111×** (9.81 h → 5.31 min), whole-universe indicator panels **11.9×**, Wilder's ATR **28×**, the backtest kernel **~58×**, the modeling layer's preprocessing kernels **5.5–53.6×**. Several ports are worth less than that and are reported as such — ADX and GARCH measure at or below 1× against *warm* numba, and exist for the ~200 ms–1.1 s JIT cold start they remove, not for steady-state speed. Every number is measured by toggling the same module's own `HAS_CPP` flag and timing both paths back-to-back. Full tables, methodology, and the ports that did not pay off: **[Documentation/16_performance.md](Documentation/16_performance.md)**.
- **Agent-first design** — every tool returns a Pydantic model. **82 LLM-callable tools** with OpenAI/Anthropic function-calling schemas, split across **five parallel runtimes** whose dispatch tables refuse what they do not own, so a tool an agent was never given is unroutable rather than merely unlikely. Bulk values cross between runtimes as typed references, never through the conversation. Unknown arguments are rejected rather than silently ignored, every Sharpe-reporting tool takes the risk-free rate it is measured against, and errors are written to be self-correcting rather than merely accurate.
- **Comprehensive coverage** — 14 indicators, 13 risk/return metrics + 5 backtest diagnostics, 12 analysis functions plus Black-Scholes-Merton pricing/Greeks/implied volatility, portfolio analysis and optimization (Markowitz mean-variance, risk parity, Black-Litterman), a stock screener, 8 backtest strategies with parameter grid search, a shared-cash portfolio simulation engine with pluggable cost and constraint models, a pairs backtest, and walk-forward/robustness diagnostics. Grid search and the signal-panel backtester also accept your own signal-generating callable or matrix, not just the built-in strategies. See the [module map](#module-map).
- **Robust infrastructure** — retry with exponential backoff, TTL + Parquet caching, a custom exception hierarchy, the `@validate_series` decorator, a hash-chained decision-record audit trail with the `sqt` CLI, and graceful degradation when C++/scipy/numba are absent.
- **Audited for correctness** — both tiers have been through line-by-line audits. 41 findings in the core, then two reviews of the modeling runtime: the first found 7 critical issues (two leakage channels, a PCA start-vector degeneracy), the second found 20 more across modeling, the data layer and the numerics — a full-refit model that had seen prices past its own recorded cutoff, an `end_date` that meant different things per provider, a "cross-section" that could mix dates, and aliases that could forge feature provenance. None of it was catchable by the suite as it stood. Every finding was reproduced against a live interpreter before being fixed, and each is pinned by a regression test. The full record, including what each defect broke: **[Documentation/17_correctness.md](Documentation/17_correctness.md)**.

---

## Installation

```bash
pip install .
# or
poetry install
```

**Requirements:** Python 3.10+, `pandas`, `numpy`, `yfinance`, `numba`, `aiohttp`, `cachetools`, `pydantic`, `statsmodels`, `scikit-learn`, `plotly`, `pyarrow`, `python-dotenv`

**Optional:** `pip install standard_quant_tools[bloomberg]` adds `blpapi` (Bloomberg's own SDK) for `BloombergProvider` — requires a running, logged-in Bloomberg Terminal; see [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#bloomberg-provider). `pip install standard_quant_tools[signing]` adds `cryptography` for Ed25519 audit-checkpoint signing — see [Audit Trail & CLI](#audit-trail--cli-standard_quant_toolsaudit-sqt) below. `PolygonProvider` needs no extra install — it's a plain REST API — just an API key (`SQT_POLYGON_API_KEY`); see [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#polygonio-provider). `pip install standard_quant_tools[mcp]` adds the Model Context Protocol SDK and the `sqt-mcp` server, which exposes the whole library to any MCP client — see [Documentation/18_mcp.md](Documentation/18_mcp.md). `pip install standard_quant_tools[polars]` adds optional `polars` interop for a growing subset of functions — pandas remains the default and required backend either way; see [Documentation/14_polars_support.md](Documentation/14_polars_support.md).

> **Note on the C++ extension:** `pip install .` now **builds `_sqt_core` when a C++ toolchain is available** (the backend is scikit-build-core, which drives the project's CMake build). Without a compiler the install still succeeds and you get the pure-Python package — every indicator, backtest and analysis function works through its Numba/pure-Python fallback, and `HAS_CPP` is `False`. That degradation is deliberate: the extension is an optional accelerator, so requiring a compiler to install would turn it into a hard dependency. Pass `-C cmake.define.SQT_REQUIRE_NATIVE=ON` to make a missing toolchain a hard error instead (what CI uses). For the in-place developer build, see [Development/build_guide.md](Development/build_guide.md).

> **Config & secrets:** copy [`.env.example`](.env.example) to `.env` (already `.gitignore`d) for any local provider configuration — currently `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` and `SQT_POLYGON_API_KEY`. `standard_quant_tools.config.load_env()` loads it automatically and is a harmless no-op when `.env` doesn't exist (the normal state in CI). In GitHub Actions / GitLab CI, set the same variable names as encrypted secrets and inject them as job-level environment variables instead — no `.env` file involved, no code changes needed either way.

---

## Quick Start

```python
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators import sma, rsi, bollinger_bands, adx
from standard_quant_tools.metrics import sharpe_ratio, max_drawdown, calmar_ratio, var_historical

# Fetch data
provider = DataFactory.get_provider()
df = provider.get_ohlcv("NVDA", "2023-01-01", "2024-01-01")

# Technical analysis
df['RSI'] = rsi(df['Close'], 14)
df['SMA_50'] = sma(df['Close'], 50)
bb = bollinger_bands(df['Close'], 20, 2.0)
adx_df = adx(df['High'], df['Low'], df['Close'])

# Risk metrics
returns = df['Close'].pct_change().dropna()
equity = (1 + returns).cumprod() * 10_000
print(f"Sharpe: {sharpe_ratio(returns):.2f}")
print(f"Max Drawdown: {max_drawdown(equity):.2%}")
print(f"VaR(95%): {var_historical(returns, 0.95):.4f}")
```

---

## Module map

| Module | What it does | Guide |
|---|---|---|
| `data` | Provider-agnostic OHLCV, fundamentals and metadata, with caching, retries and quality checks | [01](Documentation/01_data_fetching.md), [11](Documentation/11_data_quality.md) |
| `indicators` | RSI, MACD, ADX, Bollinger, ATR, Parabolic SAR, stochastic, volume — plus whole-universe panel computation | [02](Documentation/02_indicators.md) |
| `metrics` | Returns, risk, drawdown, ratios, and five volatility estimators | [03](Documentation/03_metrics.md) |
| `analysis` | Cointegration, Hurst, PCA factors, rolling beta, GARCH, regime detection, rally scoring, Black-Scholes options | [08](Documentation/08_analysis.md), [12](Documentation/12_options.md) |
| `backtest` | Signal backtesting with realistic costs, parameter grids, Monte Carlo, and a shared-cash portfolio simulator | [04](Documentation/04_backtesting.md) |
| `portfolio` | Optimization, risk decomposition, and position sizing | [05](Documentation/05_portfolio.md) |
| `screener` | Multi-ticker screening, parallel across processes above a threshold | [06](Documentation/06_screener.md) |
| `agent` | The 68-tool LLM surface across four runtimes, its dispatch, the handoff interconnect, and orchestration patterns | [07](Documentation/07_agent_tools.md), [09](Documentation/09_advanced_agent_tools.md), [13](Documentation/13_agent_orchestration.md), [19](Documentation/19_runtimes.md) |
| `modeling` | A separate 14-tool runtime for building walk-forward-validated statistical models from this library's own features, plus registry browsing and pre-flight spec validation | [15](Documentation/15_modeling.md) |
| `audit` | Content-addressed decision records, replay verification, and the `sqt` CLI | [10](Documentation/10_auditability.md) |
| `mcp` | The whole library over the Model Context Protocol, with category-gated tool exposure, `sqt://` resources and workflow prompts | [18](Documentation/18_mcp.md) |

Full API surface per module: [Documentation/00_module_reference.md](Documentation/00_module_reference.md).
Polars interop: [14](Documentation/14_polars_support.md).
## Performance
The C++ extension is **optional**. `pip install .` builds it when a toolchain
is available and silently skips it when there isn't one; `HAS_CPP` tells you
which path you are on, and the API is identical either way.

Every published figure is measured rather than projected, by toggling the same
module's own `HAS_CPP` flag and timing both paths back-to-back. The tables are
kept out of this file because they are long, and because they include the ports
that were not worth doing:

| Where | What is there |
|---|---|
| [Documentation/16_performance.md](Documentation/16_performance.md) | Every measured number, the OpenMP and AVX2 paths, the Python-level optimizations, and the honest disappointments kept beside their predictions |
| [Documentation/17_correctness.md](Documentation/17_correctness.md) | The backend-parity contract that makes the two tiers substitutable, and the audit findings behind it |
| [Development/performance_insights.md](Development/performance_insights.md) | Methodology, the benchmark scripts behind each figure, and the edge-case bugs found while measuring |
| [Development/modeling_native_plan.md](Development/modeling_native_plan.md) | Why the modeling layer's native work stopped at three phases — the arithmetic ceiling, stated before the method |
| [Development/build_guide.md](Development/build_guide.md) | Building the extension on Windows / Linux / macOS |

## Error Handling

```python
from standard_quant_tools.error import DataNotFoundError, InvalidSymbolError, ValidationError

try:
    df = provider.get_ohlcv("INVALID", "2023-01-01", "2024-01-01")
except DataNotFoundError as e:
    print(f"No data: {e}")
except InvalidSymbolError as e:
    print(f"Bad symbol: {e}")
```

**Exception hierarchy:** `QuantError` → `DataProviderError` → `DataNotFoundError / InvalidSymbolError / APIError / NonRetryableAPIError`

`ValidationError` (a `QuantError`, not a `DataProviderError`) is raised for
caller-side input problems — a bad period, a non-finite price, mismatched
series lengths — and is **never retried or re-typed** by the `retry`
decorator. See [01_data_fetching.md](Documentation/01_data_fetching.md) for
the full retry classification table.

---

## Audit Trail & CLI (`standard_quant_tools.audit`, `sqt`)

Every call routed through `agent.tools.dispatch()` can produce an immutable JSONL decision record: its inputs, the market data it pulled (with content hashes), which execution path ran, and a hash of its output — enough to tell a stale or tampered cache apart from a genuine code change. Records are hash-chained **across calendar days**, not just within one day's file, so deleting a whole day is detectable and not only editing one record. Nothing runs automatically: `SQT_AUDIT_ENABLED=0` disables record writes and `SQT_AUDIT_DIR` overrides the storage directory.

This is an *engineering control* — tamper detection, not tamper prevention or regulatory certification. See [Documentation/10_auditability.md](Documentation/10_auditability.md#auditability) for what it can and can't certify.

The `sqt` command (installed with the package) inspects and verifies these records by `request_id`:

```bash
sqt replay <request_id>              # re-run the recorded call, report whether data/output still match
sqt compare <request_id_a> <id_b>    # diff two records' status/output/timing/provenance/inputs
sqt report <request_id>              # pretty-print one record in full
sqt verify [--file PATH]             # check hash-chain integrity (full trail, or one file with --file)
sqt hold <date> [--reason TEXT]      # legal/retention hold on a calendar day, protects it from gc
sqt release-hold <date>              # remove a hold
sqt gc [--confirm]                   # delete day files past SQT_AUDIT_RETENTION_DAYS (dry-run by default)
sqt seal <date>                      # chmod a day file read-only (operational safeguard, not WORM)
sqt export --start D --end D --out F # zip a date range + manifest + standalone verifier for an auditor
sqt keygen [--out DIR]                # generate an Ed25519 keypair (local development only)
sqt anchor <date> [--key PATH]        # sign a checkpoint anchoring a day's chain endpoint
sqt verify --checkpoint <date> --pubkey PATH   # verify a checkpoint's signature (public key only)
```

`sqt replay` exits 0 if the output reproduced exactly, 1 on a confirmed mismatch, 2 if the record has no output hash to compare against. `sqt verify` exits 0 if clean, 1 if any problems are found. A dependency-free standalone verifier (`scripts/verify_audit_log.py`) is also available for external auditors who don't want to install the package. `SQT_AUDIT_REDACT_FIELDS` (comma-separated dotted field paths) replaces matching `input` fields — and, best-effort, an `error_message` that echoes one back — with a non-reversible content-hash placeholder before a record is written; set `SQT_AUDIT_REDACT_SALT` to a long random secret so that placeholder isn't brute-forceable offline for a small value space (an unset salt still works but logs a one-time warning).

**Checkpoint signing** (Ed25519, optional `pip install standard_quant_tools[signing]`) closes the one gap the hash chain cannot close on its own: a wholesale, internally-consistent rewrite of an entire day file. `sqt anchor` signs a day's chain endpoint and `sqt verify --checkpoint` verifies it with only the public key. `sqt keygen` is for local development only — a real deployment routes signing through an HSM/KMS via a `signer` callback rather than a bare key file.

---

## Running Tests

```bash
# Unit tests (no network required)
pytest tests/ -m "not integration and not benchmark and not slow"

# Including slow tests (large-data cross-validation)
pytest tests/ -m "not integration"

# Integration tests (requires internet)
pytest tests/ -m integration

# C++ vs Python benchmark tests — prints timing and speedup (requires _sqt_core)
pytest tests/cpp_bindings/test_cpp_hurst.py -m benchmark -s -v

# C++ unit tests (requires _sqt_core built with SQT_BUILD_TESTS=ON)
ctest --test-dir build --config Release -V

# C++ performance benchmark binary (prints a timing table)
# Windows: build\tests\cpp\Release\bench_hurst.exe
# Linux / macOS: ./build/tests/cpp/bench_hurst

# Performance harnesses — minutes to run, so not part of the suite.
# Every figure in Development/optimization_plan.md comes from one of these.
SQT_NUM_THREADS=1 python tests/bench/bench_kernels.py   # per-kernel scaling, serial
python tests/bench/bench_kernels.py                     # ... and parallel
python tests/bench/bench_universe.py                    # 2,000-ticker shapes

# With coverage
pytest tests/ -m "not integration" --cov=src/standard_quant_tools
```

**3335 Python tests total.** With `_sqt_core` built, `-m "not integration"` gives **3317 passing, 1 skipped, 17 deselected**. The single skip is environmental — it exercises a failure path that the input under test does not trigger — and the 7 deselected are the integration tests, which need network. Without the C++ extension the `tests/cpp_bindings/` files skip instead (they are gated on the extension being importable), and the rest still pass: every C++ path has a Python fallback, and both are held to the same contract (see [Documentation/17_correctness.md](Documentation/17_correctness.md)).

`tests/` mirrors `src/standard_quant_tools/` — one directory per package (`agent/`, `analysis/`, `audit/`, `backtest/`, `data/`, `indicators/`, `metrics/`, `modeling/`, `portfolio/`, `screener/`), plus `core/` for cross-cutting suites, `cpp/` for the C++ gtest sources CMake compiles, and `cpp_bindings/` for the Python-side backend-parity tests. Run one group with `pytest tests/backtest`.

**10 C++ test executables** run via `ctest` (Hurst, indicators, cointegration, backtest, Monte Carlo, GARCH, signal state machines, rolling regression, panel statistics, plus a randomized-input cointegration fuzz harness) — **67,731** assertion-level checks between them, 50,234 of which come from the fuzz harness alone.

Note what the fuzz harness does *not* generate: non-finite inputs. Every shape it builds is
finite, which is why a NaN-handling defect once survived it alongside every other suite. The
NaN/Inf data contract is covered separately, in
`tests/cpp_bindings/test_cpp_nan_data_contract.py`.

> **If you build the extension yourself, don't override `CMAKE_CXX_FLAGS`.**
> Passing `-DCMAKE_CXX_FLAGS=...` *replaces* the project's defaults rather
> than appending to them, which silently drops `/EHsc` on MSVC. The result
> builds and links cleanly, emits only a C4530 warning that is easy to
> dismiss, and then takes an access violation the first time a kernel throws
> across the Python boundary. All build output also lands in
> `src/standard_quant_tools/`, so a second configure directory will overwrite
> the extension your main build produced. Use the documented
> `cmake -B build` invocation in
> [Development/build_guide.md](Development/build_guide.md).

---

## Documentation

| File | Module |
|---|---|
| `Documentation/00_module_reference.md` | Every public module's API surface in one place |
| `Documentation/01_data_fetching.md` | Data providers, Parquet cache, error handling |
| `Documentation/02_indicators.md` | All 14 technical indicators with examples |
| `Documentation/03_metrics.md` | Risk/return metrics (VaR, Sharpe, Calmar, …) |
| `Documentation/04_backtesting.md` | Vectorized engine, trade log, custom signals, grid search |
| `Documentation/05_portfolio.md` | Multi-asset metrics, correlation, optimization |
| `Documentation/06_screener.md` | Filter reference, large-universe screening, example screens |
| `Documentation/07_agent_tools.md` | Core 14 LLM tools, full 68-tool registry, Pydantic models, end-to-end agent loop |
| `Documentation/08_analysis.md` | Multi-factor regression, cointegration, PCA, Hurst exponent (incl. C++ acceleration) |
| `Documentation/09_advanced_agent_tools.md` | 31 advanced/supplementary/custom-signal/analytics/options/diagnostic tools: regime-adaptive (full-sample and leakage-free walk-forward), pair scanner, walk-forward, risk attribution, portfolio optimization, position sizer, fundamentals, optimization, advanced indicators, rolling beta, extended risk, backtest diagnostics, true portfolio simulation, pair trade backtest, robustness diagnostics, capacity report, data quality report, compact backtest result, volatility estimators, correlation analysis, Monte Carlo simulation, stress test, liquidity metrics, GARCH volatility forecast, Kalman hedge ratio, EVT tail risk, option pricing/Greeks, implied volatility |
| `Documentation/10_auditability.md` | Decision-record audit trail, replay verification (both tool registries), correlated logging, `sqt` CLI |
| `Documentation/11_data_quality.md` | Dataset provenance metadata, missing-bar/stale-price/price-jump detection |
| `Documentation/12_options.md` | Black-Scholes-Merton option pricing, Greeks, implied volatility (European options only) |
| `Documentation/13_agent_orchestration.md` | Tool-category taxonomy, the lightweight router, and the multi-agent orchestrator-workers architecture |
| `Documentation/14_polars_support.md` | Optional Polars interop (`pip install standard_quant_tools[polars]`): what's supported today, the conversion-boundary design, and the phased roadmap |
| `Documentation/15_modeling.md` | The separate 14-tool modeling runtime: feature catalog and feature report, regression/classification/ranking targets, leakage-purged walk-forward validation, sample weighting, the model adapters, point-in-time joins, the content-addressed model registry, the model→backtest bridge, portfolio evaluation of OOS predictions, and what's explicitly deferred |
| `Documentation/16_performance.md` | Every measured C++ and Python-level performance figure, with methodology and the ports that did not pay off |
| `Documentation/17_correctness.md` | The correctness audits and the backend-parity contract between the Python and C++ tiers |
| `Documentation/18_mcp.md` | The MCP server: install, runtime and category scoping, the context budget, resources, prompts, and the audit trail over the protocol |
| `Documentation/19_runtimes.md` | The five parallel runtimes and why scoping is enforced at dispatch rather than advertised; the typed handoff references that carry bulk values between them; pre-flight tool validation |
| `Development/build_guide.md` | C++ extension build instructions (Windows / Linux / macOS) |
| `Development/performance_insights.md` | Algorithmic analysis: which components benefit from C++ and by how much |
| `Development/optimization_plan.md` | The optimization backlog, each item with its measured outcome |
| `Development/modeling_analysis.md` | Performance and capability analysis of the modeling layer |
| `Development/modeling_native_plan.md` | The native-kernel plan for modeling, and the ceiling that stopped it at three phases |
| `Development/mcp_plan.md` | The MCP server plan: the measured constraints, the design, and what was deferred |

---

## Contributing

Bug reports, doc fixes, and PRs are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md)
for the development workflow, code conventions, and PR expectations.

## Security

Found a security issue? Please don't open a public issue — see
[SECURITY.md](SECURITY.md) for how to report it privately.

## Changelog

Notable changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE) for the full text.

```
Copyright 2026 Karan Vora

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
