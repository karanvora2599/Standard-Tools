# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/) —
while the major version is `0`, breaking changes may still land in a minor
bump, consistent with SemVer's pre-1.0 clause.

## [Unreleased]

### Changed

- **Breaking:** `fill_price="midpoint"` renamed to `fill_price="hl2_exploratory"`
  everywhere (`run_strategy`, `run_portfolio_simulation`, `run_pair_backtest`,
  their agent-tool input models, and docs) — it was never a real bid/ask
  midpoint (just `(High+Low)/2`), and the old name implied a market-quote
  guarantee it didn't have. Every reference now carries an explicit
  look-ahead-bias caveat.
- **Breaking:** `CustomSignalBacktestInput`/`SignalPanelBacktestInput`'s
  `signal_type` now defaults to `DIRECTION` (values must be exactly -1/0/1)
  instead of `SCORE` (unrestricted float, multiplied directly into position
  size) — `SCORE` is raw leverage, not a bounded confidence value, and was an
  unsafe default for anyone passing an un-normalized signal.
- `run_pair_trade_backtest`'s `fill_price` now defaults to `"next_open"`
  instead of `"close"` — the z-score signal deciding a transition is computed
  from that same bar's Close, so executing at that same Close was look-ahead
  bias by default. `"close"` is still available for explicit same-bar/
  exploratory analysis.
- `run_backtest_optimization` (the `backtest_grid` agent-tool wrapper) now
  threads `commission_pct`/`slippage_pct` into every grid combination
  instead of silently ignoring them — `backtest_grid` itself already did
  this correctly; the gap was specific to the agent-tool wrapper.

### Added

- `data.bloomberg_provider.BloombergProvider`: a second `DataProvider`
  implementation, backed by a local Bloomberg Terminal via Desktop API
  (`blpapi`, a new optional dependency — `pip install
  standard_quant_tools[bloomberg]`). No API key (DAPI authenticates via the
  Terminal login); `SQT_BLOOMBERG_HOST`/`SQT_BLOOMBERG_PORT` are the only
  configurable, non-secret connection settings. Daily/weekly/monthly bars
  only (intraday raises a clear `ValidationError`, not wrong data). Wired
  into `DataFactory.get_provider("bloomberg")`, replacing the old
  `NotImplementedError` stub. See
  [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#bloomberg-provider).
- `standard_quant_tools.config.load_env()`: a single choke point for
  loading `.env` (via the new `python-dotenv` core dependency) into
  `os.environ`, idempotent per process, never overriding a real environment
  variable — the same mechanism whether config comes from a local `.env`
  file or CI/CD secrets (GitHub Actions / GitLab CI) injected as real env
  vars. `.env.example` documents every variable and both platforms' secrets
  syntax.
- `data/_retry.py`: extracted the retry-with-backoff decorator out of
  `yfinance_provider.py` into a shared module so `BloombergProvider` doesn't
  duplicate it; `yfinance_provider.py`'s behavior is unchanged (verified —
  same tests, same results).
- `audit.py`: a hash-chain (`prev_record_hash`/`record_hash` on every JSONL
  decision record) and `verify_audit_log_integrity()`, so the audit log
  itself is tamper-evident, not just each record's replay. JSONL writes are
  now guarded by a cross-process advisory lock (`msvcrt` on Windows,
  `fcntl.flock` on POSIX; falls back to unlocked with a debug log if neither
  is available, rather than blocking a tool call on a missing OS primitive).
- `verify_replay()` now reports data sources that disappeared between the
  original call and the replay (previously silently dropped from the
  comparison).
- `screener.py` now reports fetch/filter failures via `DataFrame.attrs`
  (`failed_filters`, `failed_tickers`, and `failed_batches` for the
  multi-worker path) instead of returning `None` — previously
  indistinguishable from a ticker that legitimately didn't pass a filter.
- Project governance: Apache 2.0 `LICENSE`/`NOTICE`, `SECURITY.md`,
  `CONTRIBUTING.md`, this `CHANGELOG.md`, license/URL metadata in
  `pyproject.toml`, and a local `v0.1.0` release tag.
- `black`/`isort` now actually pass in CI — added shared `[tool.black]`/
  `[tool.isort]` config (`profile = "black"`) and reformatted the full
  `src/`/`tests/` tree, which had never matched the CI check before.

### Fixed

- `portfolio_engine.py`: `max_gross_leverage`/`max_position_pct` are now
  enforced against realized post-cost state, not just pre-trade intent;
  added insolvency checks (a rebalance that leaves the account with
  zero/negative equity now raises instead of silently continuing); financing
  (borrow fee, margin interest) now accrues on actual elapsed calendar days
  instead of a hardcoded 1-day assumption; added validation for an empty
  universe, duplicate/unsorted rebalance dates, and non-finite weights/prices.
- `sizing.py`: fixed `vol_scaled`'s rolling-window frequency mismatch,
  `equal_weight_top_bottom`'s long/short-only allocation, and
  `dollar_neutral`'s gross-leverage drift.
- `risk_metrics.py`: `var_historical`/`var_parametric`/`cvar` now validate
  `confidence` is a valid probability bound; fixed `var_parametric`'s silent
  fallback when scipy isn't available; fixed `treynor_ratio`'s misaligned
  numerator/denominator index (the excess-return numerator previously used
  the full unaligned series while beta used only the intersected dates).
- `yfinance_provider.py`: path-traversal containment on the Parquet cache
  path (symbol/date/interval), the audit trail now fires on session-cache
  hits (not just misses), cache-hit results are copied so callers can't
  mutate shared cached state, corrupt Parquet files on disk are detected and
  evicted/refetched instead of failing or serving bad data, and atomic-write
  temp filenames are now thread-unique.
- `dispatch()` sanitizes `inf`/`nan` to `None` before returning a result,
  since raw `json.dumps()` would otherwise emit non-standard tokens.
- `run_strategy` (`backtest/engine.py`) now always recomputes
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` in Python
  (`_build_trade_log`/`_compute_trade_stats`) instead of trusting the C++
  kernel's own native trade-log values, which record each entry one bar late
  and exclude commission/slippage. This is a Python-side workaround, not a
  native-code fix — see Known Issues below for the path that's still affected.
- Fixed a day-0 drawdown edge case (see git history for the exact commit).
- The four provider example agent loops (`Implementation/*/_agent_utils.py`)
  fixed duplicate logging handlers on repeated setup, malformed tool-call
  JSON silently becoming `{}`, missing request/tool timeouts, non-strict
  JSON allowing `NaN`/`Infinity` tokens, and narrative text being discarded
  after each tool round.
- CI: dropped the unused `pytest-freezegun` dependency (it imported
  `distutils`, which Python 3.12 removed, and nothing in the suite actually
  used it) and added `anthropic` to the `test` extras, since
  `test_multi_agent_tool_coverage.py` transitively imports it.

### Known Issues

- `backtest_grid`'s `_sqt_core` batch kernel (`batch_run_strategy`) still
  returns the native C++ trade stats uncorrected — the one-bar-late entry
  and missing commission/slippage bug that `run_strategy` now works around
  in Python is still present for grid-search results, including the
  aggregate speedups reported for `run_walk_forward_backtest` and
  `run_backtest_optimization`, which route through the batch path. Not yet
  fixed at the native level.

## [0.1.0] - 2026-07-24

Initial documented release. `main` had no prior tags — this release
consolidates everything built since the first commit into one baseline.

### Added

**Data layer** (`standard_quant_tools.data`)
- `YFinanceProvider`: `get_ohlcv` / `get_ohlcv_async`, `get_ticker_info`,
  `get_financial_ratios`, `get_metadata` (dataset provenance), with retry
  with exponential backoff, an in-process TTL session cache, and a
  persistent Parquet disk cache for historical OHLCV (`SQT_CACHE_DIR`).
- `data.quality`: heuristic data-quality checks — `detect_missing_bars`,
  `detect_stale_prices`, `detect_price_jumps`.

**Indicators** (`standard_quant_tools.indicators`) — 14 functions across
trend (SMA, EMA, MACD, ADX+DI, Parabolic SAR, Williams %R), momentum (RSI,
Stochastic), volatility (Bollinger Bands, ATR, Wilder's ATR), and volume
(OBV, VWAP, MFI), each with a C++ extension → Numba JIT → pure-Python
fallback chain.

**Metrics** (`standard_quant_tools.metrics`) — 18 functions: return metrics
(cumulative return, CAGR, annualized volatility), risk/ratio metrics
(Sharpe, Sortino, Calmar, historical/parametric VaR, CVaR, Information
Ratio, Treynor, max drawdown), and backtest diagnostics (drawdown episodes,
trade expectancy, MAE/MFE excursions, exposure stats).

**Analysis** (`standard_quant_tools.analysis`) — 12 functions: OLS beta /
rolling beta, Engle-Granger cointegration + spread/half-life/z-score,
multi-factor regression + rolling factor loadings, PCA on returns, and
Hurst exponent (DFA / R-S / rolling), several with C++ fast paths.

**Backtesting** (`standard_quant_tools.backtest`)
- Vectorized single-ticker engine (`run_strategy`) with transaction costs,
  trade log, and three execution-timing modes (`close`/`next_open`/a
  same-bar approximate-fill mode, renamed `hl2_exploratory` — see Unreleased).
- Parameter grid search (`backtest_grid`) and walk-forward / regime-adaptive
  (leakage-free) backtesting.
- Multi-ticker signal-panel backtesting (`run_signal_panel_backtest`).
- A shared-cash portfolio simulation engine (`portfolio_engine.py`) with
  pluggable cost models (`costs.py`: percentage/per-share commission,
  spread, square-root market impact, short borrow, margin interest),
  liquidity/capacity constraints (`constraints.py`), and position-sizing
  helpers that turn a score panel into a target-weight panel (`sizing.py`).
- Two-leg pair-trade backtesting (`pairs.py`), reusing the portfolio engine
  so both legs share one cash account and rebalance together.
- Robustness diagnostics (`robustness.py`): block-bootstrap confidence
  intervals, parameter sensitivity, and Deflated Sharpe Ratio.
- A local Parquet artifact store (`artifacts.py`) for equity curves/trade
  logs too large to embed inline in an agent-tool response.
- 4 built-in strategies (SMA crossover, RSI mean-reversion, MACD crossover,
  Bollinger reversion), plus support for bring-your-own signal callables in
  grid search and the signal-panel backtester.

**Portfolio & Screener**
- `standard_quant_tools.portfolio`: multi-asset portfolio metrics, risk
  attribution (marginal contribution to risk, PCA-based, factor-based),
  correlation matrix.
- `standard_quant_tools.screener`: async filter-based stock screener with
  automatic `ProcessPoolExecutor` fan-out for universes over 20 tickers.

**Agent tools** (`standard_quant_tools.agent`) — 34 LLM-callable tools with
Pydantic input/output models and OpenAI/Anthropic function-calling schemas,
covering backtesting, risk/technical/portfolio analysis, screening, factor
regression, cointegration, PCA, Hurst analysis, regime-adaptive and
walk-forward backtests, pair scanning, position sizing, bring-your-own-signal
backtests, portfolio simulation, pair-trade backtests, robustness
diagnostics, capacity reports, and data-quality reports.

**Auditability** (`standard_quant_tools.audit`, `sqt` CLI)
- Every `dispatch()` call can write a tamper-evident JSONL decision record
  (inputs, market-data content hashes, execution path, output hash, latency).
- `verify_replay()` re-runs a recorded call and distinguishes stale/tampered
  cache from a genuine code change.
- The `sqt` CLI (`sqt replay` / `sqt compare` / `sqt report`) inspects and
  verifies decision records by `request_id` from the command line.

**Performance**
- Optional C++ extension (`_sqt_core`, pybind11 + CMake) accelerating Hurst,
  RSI/ADX/Parabolic SAR, Wilder's ATR, Engle-Granger cointegration, 2-variable
  OLS, the backtest kernel and grid-search batch kernel, rolling factor
  loadings, rolling beta, Bollinger Bands, and the Stochastic Oscillator.
  The API is identical with or without it; every path falls back to
  Numba/pure-Python transparently when the extension isn't built.

### Fixed

Notable correctness fixes folded into this baseline (see git history for
full detail):
- Look-ahead bias in the pairs-backtest z-score default (now a rolling
  window by default instead of a full-sample static z-score) and in the
  regime-adaptive walk-forward backtest.
- `run_portfolio_simulation` now rejects `NaN` target weights immediately
  instead of silently propagating them through the equity curve, and surfaces
  an explicit look-ahead-bias warning when using same-bar (`close`) fills.
- `sqt replay` now exits non-zero on a confirmed output mismatch instead of
  always exiting `0`.
- De-annualized the Sharpe ratio fed into the Deflated Sharpe Ratio formula
  in `get_robustness_diagnostics` (previously inflated the statistic).
- `save_artifact` now rejects a reused `(run_id, name)` unless `overwrite=True`,
  validates both against a path-traversal-safe identifier pattern, and writes
  atomically.

[Unreleased]: https://github.com/karanvora2599/Standard-Tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/karanvora2599/Standard-Tools/releases/tag/v0.1.0
