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
- **Internal:** `src/standard_quant_tools/audit.py` (~1060 lines after
  audit-trail hardening phases 1–2) was split into a package,
  `standard_quant_tools/audit/` (hashing, context, provenance, paths,
  models, storage, writer, verify, redaction, retention, export, signing,
  dispatch, replay), ahead of phase 3 adding more surface area.
  `__init__.py` re-exports the full previous public + semi-private surface,
  so this is a pure internal reorganization — no call site anywhere in the
  codebase or its tests needed to change, and no behavior changed.

### Added

- **3 new agent tools: GARCH volatility forecasting, Kalman dynamic hedge
  ratio, EVT tail risk** (42 → 45 tools). All three model time-varying
  dynamics or fat tails — a gap the analytics layer's existing static/
  point-in-time tools (cointegration, correlation, realized-vol estimators,
  historical VaR/CVaR) didn't cover:
  - `run_garch_volatility_forecast` (`analysis/garch.py`) — fits GARCH(1,1)
    conditional volatility and forecasts it forward, unlike
    `get_volatility_estimators`' backward-looking realized measures. The
    variance recursion is numba-`@njit`'d (inherently sequential, same tool
    `backtest/strategies.py`'s state machines already use); MLE fitting via
    `scipy.optimize` handles millions of bars in well under a second thanks
    to the JIT'd recursion. Requires scipy — no meaningful scipy-free
    fallback for a maximum-likelihood fit.
  - `run_kalman_hedge_ratio` (`analysis/cointegration.py`) — re-estimates a
    pair's hedge ratio every bar via a Kalman filter, a time-varying
    diagnostic companion to `run_cointegration_test`'s static OLS
    `hedge_ratio`. Hand-unrolled 2×2 numba recursion, verified to converge
    to `cointegration_test`'s static hedge ratio as the `delta` tuning
    parameter shrinks toward 0. Deliberately **not** wired into
    `run_pair_trade_backtest`, which still trades a single static hedge
    ratio for the whole window — a real, separate follow-up.
  - `get_tail_risk_metrics` (`metrics/risk_metrics.py`) — Extreme Value
    Theory tail risk via Peaks-Over-Threshold: fits a Generalized Pareto
    Distribution to the worst tail of daily losses and extrapolates
    VaR/CVaR from that fitted tail, reported alongside the naive
    `var_historical` figure for direct contrast. Default fitting method is
    probability-weighted moments (closed-form, pure numpy, zero
    optional-dependency surface); `method="mle"` requires scipy.

  All three follow the established pattern exactly: new Pydantic
  Input/Result models, registration in both `get_agent_tools()` and
  `_TOOL_DISPATCH`, worker assignment + updated system prompt in
  `Multi_Agent_Implementation/worker_agents.py` (verified against
  `test_multi_agent_tool_coverage.py`), and hand-verified pure-function
  tests (GARCH against a simulated known-parameter process; Kalman against
  a hand-computed toy recursion and convergence to static OLS; EVT against
  a known-generating GPD via inverse-CDF sampling) plus structural
  agent-tool tests. See
  [Documentation/09_advanced_agent_tools.md](Documentation/09_advanced_agent_tools.md),
  Tools 26–28.

  Found and fixed a real bug while implementing this: the initial EVT
  probability-weighted-moments estimator had its order-statistic weights
  backwards (weighting by `F(x)` instead of `1-F(x)`), which silently fit
  the wrong tail shape — caught by the known-generating-GPD hand
  verification before it shipped, not by the unit tests alone.

- **4 new backtest strategies** (`backtest/strategies.py`, `STRATEGY_REGISTRY`
  now has 8 entries, up from 4): `donchian_breakout` (Turtle-style channel
  breakout, entry/exit channels use `.shift(1)` so it's a genuine breakout
  past the already-established channel, not a same-bar tautology),
  `momentum_timeseries` (trailing-return threshold, fully vectorized —
  `pandas.Series.pct_change`, no per-bar state at all), `vwap_reversion`
  (mean reversion to a rolling VWAP rather than a plain price mean, aimed
  at intraday/tick data), and `adx_trend` (ADX-strength-filtered
  directional trend, a single vectorized boolean condition on the existing
  `adx()` indicator's output). Every hysteresis-based strategy
  (`donchian_breakout`, `vwap_reversion`, matching the existing
  `rsi_mean_reversion`/`bollinger_reversion` pattern) runs its entry/exit
  tracking through a numba-JIT state machine — verified to complete in
  well under a second on 500k-bar synthetic series in
  `tests/test_strategies.py::TestScalesToLargeSeries`, with no interpreted
  Python loop over the series regardless of length. The other two need no
  state machine at all. All four are immediately usable through every
  entry point that already accepted a `STRATEGY_REGISTRY` name generically
  (`backtest_grid`, `get_backtest_diagnostics`, `run_backtest_compact`,
  `run_backtest_optimization`, `run_walk_forward_backtest`,
  `get_robustness_diagnostics`) — updated their Pydantic field
  descriptions accordingly. They do **not** get dedicated `run_*_backtest`
  tools (only the original 4 do) and are **not** added to
  `compare_strategies`' fixed four-strategy comparison or
  `run_regime_adaptive_backtest`'s curated 3-way regime→strategy map —
  both deliberate scope boundaries, not oversights. See
  [Documentation/04_backtesting.md](Documentation/04_backtesting.md).

  Registering the 4 new strategies surfaced a real, unrelated gap:
  `run_regime_adaptive_walkforward_backtest` (unlike the single-shot
  `run_regime_adaptive_backtest` above) iterates the *entire*
  `STRATEGY_REGISTRY` every window trying all of them, so it immediately
  `KeyError`'d on the first new strategy name via
  `_DEFAULT_PARAM_GRIDS[strat_name]` — that dict only had the original 4
  entries. Fixed by adding default grids for all 4 new strategies and
  changing `grid_overrides[strat_name]` to `grid_overrides.get(strat_name)`
  (the per-strategy override fields on `RegimeAdaptiveWalkForwardInput`
  only exist for the original 4; newer registry entries fall through to
  their default grid, same as any future addition would without a
  matching Pydantic field) — caught by
  `tests/test_new_agent_tools.py::TestRegimeAdaptiveWalkForwardBacktest`,
  not discovered after the fact.

- **Portfolio optimization** (`portfolio/optimize.py`): `mean_variance_optimize`
  (Markowitz mean-variance — `max_sharpe`/`min_volatility`/`target_return`/
  `target_volatility`), `risk_parity_weights`, and `black_litterman` (plus
  `build_bl_views`, a convenience for turning a plain-dict view list into the
  `(P, Q, Omega)` matrices `black_litterman` expects). The unconstrained
  mean-variance case (`allow_short=True`, `max_weight=None`) is solved in
  closed form via the standard Merton (1972) two-fund efficient-frontier
  parametrization — numpy only, no solver dependency, `converged` is always
  `True`. Any long-only and/or weight-capped request uses scipy (SLSQP),
  following the same "scipy optional, clear error if needed and missing"
  convention `metrics.risk_metrics.var_parametric` already established; a
  genuinely infeasible constrained request (e.g. an unreachable
  `target_return` under a `max_weight` cap) reports `converged=False` rather
  than a silently wrong answer. `risk_parity_weights` is a documented
  heuristic (damped multiplicative fixed-point iteration) — not a
  globally-convergence-proven algorithm like the mean-variance closed form —
  and reports its own `converged` flag honestly; verified against a
  diagonal-covariance closed-form case (inverse-volatility weighting) in
  tests. New agent tool `run_portfolio_optimization`
  (`PortfolioOptimizationInput`/`Result`, `BLViewInput`), registered in
  `get_agent_tools()`/`dispatch()` and assigned to the multi-agent
  orchestrator's Portfolio Risk & Sizing worker. This closes the gap
  `backtest/sizing.py`'s own docstring flagged: every other portfolio-facing
  tool only *scored* weights already chosen; nothing *produced* them. See
  [Documentation/05_portfolio.md](Documentation/05_portfolio.md#portfolio-optimization).

- **Options pricing, Greeks & implied volatility** (`analysis/options.py`):
  `black_scholes_price`/`black_scholes_greeks` (Black-Scholes-Merton,
  European options only, `dividend_yield` covers the Merton 1973 continuous-
  dividend extension) and `implied_volatility` (Newton-Raphson with a
  bisection fallback over a practical `[1e-6, 5.0]` bracket, plus a
  no-arbitrage bound check before solving). Dependency-free: the standard
  normal CDF/PDF are computed via `math.erf` (stdlib), not scipy. Every
  Greek is cross-validated in tests against a finite-difference derivative of
  `black_scholes_price` itself (not just checked against the textbook
  formula), and pricing matches Hull's published reference example exactly.
  Two new agent tools, `get_option_pricing` (price + all five Greeks in one
  call) and `get_implied_volatility`
  (`OptionPricingInput`/`Result`/`OptionGreeks`,
  `ImpliedVolatilityInput`/`Result`), registered in
  `get_agent_tools()`/`dispatch()` and assigned to the multi-agent
  orchestrator's Technical & Risk Analysis worker (Greeks are risk
  sensitivities). `get_agent_tools()` now returns 42 tools, up from 39. See
  [Documentation/12_options.md](Documentation/12_options.md) (new file).

- `data.polygon_provider.PolygonProvider`: a third `DataProvider`
  implementation, backed by Polygon.io's plain REST API — no vendor SDK to
  install, just an API key (`SQT_POLYGON_API_KEY`, no default; get a free
  one at https://polygon.io/dashboard/api-keys). Supports `1m`/`5m`/`15m`/
  `30m`/`60m`/`1d`/`1wk`/`1mo`/`3mo` bars via the Aggregates (Bars) endpoint
  (other intervals raise `ValidationError`); `get_financial_ratios` derives
  `trailing_pe`/`price_to_book`/`debt_to_equity`/`return_on_equity`/
  `profit_margins` from the most recent Financials vX filing combined with
  `market_cap` from Ticker Details v3 — `forward_pe` and `dividend_yield`
  are always `None` (no forward estimates or dividend-history aggregation
  in scope). Wired into `DataFactory.get_provider("polygon", api_key=...)`,
  replacing the old `NotImplementedError` stub. See
  [Documentation/01_data_fetching.md](Documentation/01_data_fetching.md#polygonio-provider).
- **Audit trail hardening, phase 3 (Ed25519 checkpoint signing + pluggable
  storage backend):** `audit.generate_keypair()`/`checkpoint_and_sign()`/
  `verify_checkpoint_signature()` add an optional external anchor closing
  the one gap the hash chain can't close on its own — an attacker who
  consistently rewrites an entire day file *and* its chain-index entry to
  stay internally self-consistent. A signed checkpoint
  (`{date, final_record_hash, index_hash, signed_at_utc}`) is verifiable
  with only the public key, no trust in the JSONL files' own consistency
  required. Requires the new optional `cryptography` dependency
  (`pip install standard_quant_tools[signing]`, a new `signing` extra in
  `pyproject.toml`); every other audit-trail feature keeps working without
  it, and calling a signing function without it installed raises a clear
  `ImportError` instead of a confusing traceback (same pattern as the
  `bloomberg` extra). Signing key: pass a `signer` callback (routed through
  an HSM/KMS) for anything beyond local development, or `key_path`/
  `SQT_AUDIT_SIGNING_KEY_PATH` pointing at a raw key file —
  `generate_keypair()`/`sqt keygen` are explicitly labeled local-development
  only, not a production key-custody solution. New `sqt keygen`/
  `sqt anchor <date>`/`sqt verify --checkpoint <date> --pubkey PATH` CLI
  subcommands.

  Also introduces a pluggable `AuditStorageBackend` interface behind
  `AuditWriter`; `LocalFilesystemBackend` (the only implementation shipped)
  is a like-for-like move of the previous direct-filesystem behavior behind
  that interface, not a new capability — it's a seam so a future WORM
  backend (S3 Object Lock, Azure Immutable Blob) could be substituted later
  without touching `AuditWriter`'s chain-hashing/locking logic. Building
  that backend is explicitly out of scope for this round.

  28 new tests across `tests/test_audit_signing.py` (18) and
  `tests/test_audit_storage.py` (5, including a fake in-memory backend that
  proves the interface is a real seam, not a passthrough wrapper) plus 5 new
  `sqt keygen`/`sqt anchor`/`sqt verify --checkpoint` CLI tests in
  `tests/test_cli.py`. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#checkpoint-signing-ed25519).

- **Audit trail hardening, phase 2 (retention, legal hold, sealing,
  redaction, export bundle):** `audit.hold_day()`/`release_hold()`/
  `is_held()` place/remove a legal/retention hold sidecar
  (`<date>.jsonl.hold`) on a calendar day. `gc_candidates()`/`gc()` delete
  day files past `SQT_AUDIT_RETENTION_DAYS` (or an explicit
  `retention_days` param) — held days are always excluded, deletion never
  happens automatically (`dry_run=True` by default, only ever triggered
  explicitly via `sqt gc --confirm`), and an unset retention window means
  never delete. Deleting a day file this way is real and permanent, and —
  by design, not by bug — `verify_audit_trail_integrity()` will correctly
  report it as "likely deleted" afterward, same as it would for tampering;
  the chain has no way to tell the two apart, so treat your own
  gc-invocation log as the record of *why*. `seal_day()` chmod's a day file
  read-only as an operational safeguard against accidental writes —
  explicitly not WORM. `SQT_AUDIT_REDACT_FIELDS` (comma-separated dotted
  field paths) replaces matching `input` fields with a non-reversible
  content-hash placeholder before a record is written, so redacted values
  stay comparable across records without the raw value ever touching disk.
  `export_bundle()` zips a date range of day files, the chain index, a
  manifest (per-file SHA-256, record counts, provenance), a copy of
  `scripts/verify_audit_log.py`, and verification instructions into one
  auditor-ready archive. New `sqt hold`/`sqt release-hold`/`sqt gc`/
  `sqt seal`/`sqt export` CLI subcommands. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#retention-legal-hold-sealing-and-export).
- **Audit trail hardening, phase 1 (cross-day chain continuity, durability,
  `sqt verify`):** decision records were previously hash-chained only
  *within* one day's JSONL file — deleting an entire day's file outright was
  undetectable. `audit.py` now maintains an independent, self-hash-chained
  witness log (`_chain_index.jsonl`) at the audit-dir root, one entry per
  calendar day with any activity; the first record of a new day commits to
  the previous active day's last hash via this index (correctly bridging
  gaps like weekends without a false positive), so an attacker now has to
  rewrite both the day file and the index, consistently, to hide a deletion.
  New `verify_audit_trail_integrity()` checks the full trail (the index's
  own chain, index-vs-on-disk day files in both directions, and each day
  file reseeded with the index's claimed starting hash); the existing
  `verify_audit_log_integrity()` gained an optional `expected_prev_hash`
  param (default unchanged) so it can be seeded that way. Every write — a
  decision record or a chain-index entry — is now followed by `f.flush()` +
  `os.fsync(f.fileno())` before its lock is released, unconditionally, so a
  record isn't lost to a crash immediately after `dispatch()` returns.
  New `sqt verify [--file PATH]` CLI subcommand (full trail by default,
  single file with `--file`; exit 0 clean / 1 problems found). New
  `scripts/verify_audit_log.py`: a deliberate, stdlib-only reimplementation
  of the same hashing/chain-walking logic (no `pydantic`/`pandas`/`numpy`,
  no package install) so an external auditor can verify an exported log
  bundle independently; `tests/test_standalone_verifier.py` is a parity
  test that fails if the two implementations' hash output ever diverges.
  Pre-existing audit directories need no migration — old day files stay
  independently valid, and cross-day linkage begins transparently at the
  next new-day write. See
  [Documentation/10_auditability.md](Documentation/10_auditability.md#what-this-can-and-cannot-certify)
  for what this does and does not certify — it is a tamper-*detection*
  control, not tamper prevention or regulatory certification by itself.
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
  kernel's own native trade-log values, which used to record each entry one
  bar late and exclude commission/slippage. This Python-side override
  remains in place as a safety net even after the underlying native bug was
  also fixed directly (see below) — see Known Issues for the exact pending
  verification status.
- Fixed a day-0 drawdown edge case (see git history for the exact commit).
- `_cpp/src/backtest.cpp`'s `run_strategy` native trade-log construction
  rewritten to match `_build_trade_log`'s accounting exactly (entry size =
  signal magnitude not just sign, `prices[i-1]` as the reference price,
  correct commission/slippage deduction) — this is the fix for the exact bug
  the Python-side override above works around, now applied at the native
  level too, including `backtest_grid`'s batch path (`batch_run_strategy`)
  which had no equivalent Python override. **Not yet verified against a
  real compiled `_sqt_core`** (no C++ toolchain available where this was
  written) — see Known Issues.
- `stochastic_oscillator`: `k_period<=0`/`d_period<=0` now raise
  `ValidationError` in both the C++ kernel and its Python wrapper —
  `d_period<=0` previously reached the native kernel unchecked, causing an
  out-of-bounds vector read (an uncatchable segfault, not a Python
  exception), not just a wrong result.
- `hurst_exponent`/`rolling_hurst`: `method` must now be exactly `"dfa"` or
  `"rs"` (raises `ValidationError` otherwise, in both paths) —
  previously any other string was silently treated as `"rs"` while the
  result's own `"method"` field echoed back the typo, making the mistake
  invisible. `HurstInput.method`, `RegimeAdaptiveInput.hurst_method`, and
  `RegimeAdaptiveWalkForwardInput.hurst_method` are now
  `Literal["dfa", "rs"]` instead of a bare `str` so a bad value is rejected
  by Pydantic before it ever reaches the function.
- `parabolic_sar`: `af_start`/`af_step`/`af_max` are now validated (finite;
  `af_start>0`; `af_step>=0`; `af_max>0`; `af_max>=af_start`) in both the
  C++ kernel and the Python wrapper — a nonsensical combination previously
  produced a silently meaningless SAR series instead of raising.
- `run_strategy`/`backtest_grid`: `initial_capital`, `commission_pct`, and
  `slippage_pct` are now validated (finite, correct sign) before reaching
  the native kernel — a zero/negative/non-finite `initial_capital`
  previously produced silent `inf`/`nan` in `total_return`/`calmar_ratio`
  instead of raising.
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

- **Update:** the native trade-stat parity gap described in earlier drafts
  of this section (`backtest_grid`'s C++ batch kernel returning uncorrected
  trade stats) has a fix implemented at the native level (see Fixed above),
  but it is **unverified** — there is no C++ toolchain available in the
  environment that wrote the fix, so it hasn't been built and run locally.
  Verification is deferred to CI via a new gated test,
  `tests/test_backtest.py::TestNativeTradeStatsCorrectness`, which only runs
  once `_sqt_core` is actually built (e.g. by `build-cpp.yml`). Until that
  CI run confirms the native and Python trade-stat accounting genuinely
  agree, treat `backtest_grid`'s C++-path `win_rate`/`profit_factor`/
  `num_trades`/`avg_trade_return_pct` (and anything built on top of it,
  e.g. `run_walk_forward_backtest`/`run_backtest_optimization`) as
  not yet confirmed trustworthy when `_sqt_core` is built.

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
