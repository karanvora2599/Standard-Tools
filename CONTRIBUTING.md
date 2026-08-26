# Contributing to Standard Quant Tools

Thanks for considering a contribution. This is a small, single-maintainer
project, so the process is intentionally lightweight.

## Getting Set Up

```bash
git clone https://github.com/karanvora2599/Standard-Tools.git
cd Standard-Tools
pip install -e ".[test,dev]"
```

The optional C++ extension (`_sqt_core`) is not required for development —
every code path falls back to Numba/pure-Python automatically when it isn't
built. See [Development/build_guide.md](Development/build_guide.md) if you're
working on the C++ side specifically.

If you're working on the Bloomberg provider (`pip install -e
".[test,dev,bloomberg]"`), copy [`.env.example`](.env.example) to `.env`
(already `.gitignore`d) and fill in `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT`
if you're proxying Desktop API to a non-default address — most setups need
neither, since DAPI talks to a Terminal on `localhost:8194` by default with
no separate credential. Never commit a `.env` file or hardcode a secret in
code; see [SECURITY.md](SECURITY.md) if you're unsure whether something
counts as one.

## Before You Start

- **Open an issue first** for anything beyond a small fix (new agent tool,
  new indicator/metric, a behavior change) so the approach can be agreed on
  before you invest time in an implementation.
- **Bug fixes and doc fixes** can go straight to a PR without a prior issue.

## Development Workflow

1. **Implement → test → document**, in the same PR. Every feature in this
   repo pairs a code change with a `tests/<module>/test_*.py` addition and a
   `Documentation/*.md` (or `README.md`) update — see any recent commit for
   the pattern. A bug fix should add a regression test that would have
   caught it.
   **Where the test goes.** `tests/` mirrors `src/standard_quant_tools/`,
   one directory per package:

   ```
   tests/
     conftest.py        shared fixtures (mock_provider, …) — visible to every subdirectory
     agent/  analysis/  audit/  backtest/  backtesting/  data/
     indicators/  mcp/  metrics/  modeling/  portfolio/  screener/
     core/              cross-cutting: errors, compat shims, regression suites
     surface/           whole-surface layers: invariants, fuzzing, metamorphic, determinism
     docs/              the generated-documentation checks
     bench/             performance harnesses — minutes to run, not part of the suite
     cpp/               C++ gtest sources, compiled by CMake — NOT collected by pytest
     cpp_bindings/      Python-side parity tests for the compiled extension
   ```

   Two directories are easy to confuse: `tests/cpp/` holds `.cpp` gtest
   sources that CMake builds (`.github/workflows/build-cpp.yml` watches that
   path), while `tests/cpp_bindings/` holds the `test_cpp_*.py` files pytest
   collects, which assert the C++ and Python backends against each other.

   Directories holding same-named test modules need an `__init__.py` — the
   suite is a package, which is what lets `tests/agent/test_agent_tools.py`
   and `tests/modeling/test_agent_tools.py` coexist. (`bench/` and `cpp/`
   have none, because pytest does not collect them.) A test needing a path
   outside the suite (a script, a reference implementation) should import
   `REPO_ROOT` from the `tests` package rather than chaining `__file__`
   parents, which encodes how deep the file happens to sit.

2. **Run the test suite** before opening a PR:
   ```bash
   pytest -m "not integration and not benchmark and not slow"
   ```
   A single group runs with `pytest tests/backtest`, one file with
   `pytest tests/backtest/test_costs.py`.
   `integration` tests hit live network (yfinance) and aren't required for
   most PRs; `benchmark`/parts of the C++ test files require `_sqt_core` to
   be built and are skipped automatically otherwise.
3. **Format and lint**:
   ```bash
   black src/ tests/
   isort src/ tests/
   ```
   CI enforces both (`.github/workflows/lint.yml`) with `--check`, so run
   them locally before pushing. `mypy` is listed as an optional dev
   dependency but is not currently enforced in CI — type-annotate new code
   in the existing style, but a `mypy` pass isn't a merge requirement yet.
4. **CI matrix**: tests run against Python 3.10, 3.11, and 3.12
   (`.github/workflows/ci.yml`). A change that only works on one of these
   will fail CI.

## Code Conventions

- **Fallback chains**: performance-sensitive functions (indicators, OLS
  kernels, backtest loops) follow a C++ extension → Numba JIT → pure-Python
  fallback pattern. If you add one, all three paths must produce identical
  output — see `tests/cpp_bindings/test_cpp_*.py` for the parity-test pattern, and note
  the `_sqt_core` C++ build isn't available in most dev environments, so
  your new C++ path should still be exercised via `tests/cpp/*.cpp` native
  tests plus a Python-side parity test that's skipped (not failed) when the
  extension isn't built.
- **Agent tools**: tools live in runtime packages under
  `agent/runtimes/<runtime>/`; `agent/tools.py` is a FACADE that unions
  them, not the place tools are defined. A new one needs (a) the
  computation in a library module (`analysis/`, `portfolio/`,
  `backtesting/`) with its limits in the docstring, (b) an Input model with
  `ConfigDict(extra="forbid")` and a **typed** Result model — an untyped
  return silently drops the MCP output schema, and a test pins that,
  (c) entries in that runtime's `TOOL_DEFS` and `TOOL_DISPATCH`, which are
  built from one list so a tool cannot be advertised without being
  dispatchable, (d) a re-export from `agent/tools.py` and
  `agent/__init__.py`'s `__all__`, which a test also pins, (e) a
  regenerated `Documentation/20_tool_index.md`
  (`python Development/generate_tool_index.py` — `tests/docs/` fails the
  build otherwise), and (f) a matching section in
  `Documentation/07_agent_tools.md` or `09_advanced_agent_tools.md`.
  Adding a whole RUNTIME additionally needs `RUNTIME_CATEGORIES`,
  `RUNTIME_LABELS` and `RUNTIME_DESCRIPTIONS` entries, a category in
  `agent/router.py`, a worker in `Multi_Agent_Implementation/worker_agents.py`,
  and `MOVED_FROM` entries for anything relocated.
- **No look-ahead bias**: this is the single most damaging bug class in a
  backtest library. Any new signal/execution path must not use data that
  wouldn't have been available at decision time — see the walk-forward and
  portfolio-simulation tests for the pattern used to catch this
  (`tests/backtest/test_backtest_walk_forward.py`, `tests/agent/test_new_agent_tools.py`'s
  no-lookahead regression tests).
- **Validate at boundaries, not internally**: match the existing style —
  raise `standard_quant_tools.error.ValidationError` for bad input at
  public function/tool boundaries; don't add defensive checks for states
  that can't occur given those boundary checks.
- Default to **no code comments** unless the *why* is genuinely non-obvious
  (a subtle invariant, a workaround for a specific bug). Well-named
  functions and variables should carry the *what*.

## Submitting a Pull Request

- Keep PRs scoped to one change — a bug fix, one new tool, one doc
  correction. Large unrelated changes bundled together are harder to review
  and revert if something's wrong.
- Describe *why* the change is needed in the PR description, not just what
  changed — the diff already shows what changed.
- Reference the issue number if one exists.

## Reporting Bugs

Open a [GitHub issue](https://github.com/karanvora2599/Standard-Tools/issues)
with:
- A minimal reproduction (ideally a short script against synthetic data,
  not a specific ticker/date range that depends on live market data).
- What you expected vs. what happened.
- Whether it's specific to the C++ path, Numba path, or both.

For security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of
opening a public issue.

## License

By contributing, you agree that your contributions will be licensed under
the project's [Apache License 2.0](LICENSE).
