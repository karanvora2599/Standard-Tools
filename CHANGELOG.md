# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and version numbers follow [Semantic Versioning](https://semver.org/) —
while the major version is `0`, breaking changes may still land in a minor
bump, consistent with SemVer's pre-1.0 clause.

## [Unreleased]

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
  trade log, and three execution-timing modes (`close`/`next_open`/`midpoint`).
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
