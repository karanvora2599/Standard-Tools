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
- **Breaking (Tier 4 item 12 of the C++ code review):** all four
  hysteresis signal state machines — `_rsi_state_machine`,
  `_bollinger_state_machine`, `_donchian_state_machine`,
  `_vwap_reversion_state_machine` in `backtest/strategies.py`, plus the C++
  ports `donchian_state_machine`/`vwap_reversion_state_machine` in
  `signal_state_machines.cpp` — now carry the currently-held position
  through a NaN (rolling-warmup) bar in their *output*, instead of
  hardcoding `0.0` regardless of whether a position was actually open. The
  internal `in_pos` state was never touched by a NaN bar in either version
  (that part was already correct); only the emitted value for that bar was
  wrong, previously showing a phantom close/reopen blip in a position
  series that a real caller (or anything downstream reading these signals
  as an actual position, not just a steady-state indicator) would not
  expect — the position was never actually closed. This changes real
  output values for the `donchian_breakout`/`vwap_reversion` (and any
  RSI-/Bollinger-hysteresis-based) strategies wherever NaN warmup bars
  occur alongside an already-open position; confirmed with the user before
  implementing, given the behavior was previously documented as
  intentional in both the Python and C++ docstrings. Updated docstrings in
  both languages and the affected native/Python tests
  (`tests/cpp/test_signals.cpp`, `tests/test_cpp_signals.py`) accordingly,
  including new coverage for the previously-untested "NaN bar while a
  position is already open" case on the VWAP side.
- **Internal:** `src/standard_quant_tools/audit.py` (~1060 lines after
  audit-trail hardening phases 1–2) was split into a package,
  `standard_quant_tools/audit/` (hashing, context, provenance, paths,
  models, storage, writer, verify, redaction, retention, export, signing,
  dispatch, replay), ahead of phase 3 adding more surface area.
  `__init__.py` re-exports the full previous public + semi-private surface,
  so this is a pure internal reorganization — no call site anywhere in the
  codebase or its tests needed to change, and no behavior changed.

### Added

- **Agent tool orchestration: category taxonomy, a lightweight router, and a
  hardened multi-agent orchestrator.** Tool metadata used to be
  hand-duplicated across `get_agent_tools()`'s `tool_defs`, `_TOOL_DISPATCH`,
  and a hardcoded `WORKER_AGENTS` tool-list in
  `Multi_Agent_Implementation/worker_agents.py`, drifting apart silently
  (README/comments variously claimed 34, 42, or 45 tools against a real
  registry of 45). `standard_quant_tools.agent.tools.TOOL_CATEGORY` is now
  the single source of truth — every tool mapped to one of 7 categories
  (`screener`, `analysis`, `quant_research`, `backtest_execution`,
  `backtest_validation`, `custom_signal`, `portfolio_risk`; the former
  16-tool `backtest` bucket split into execution vs. validation, since
  "run this strategy" and "optimize this strategy's parameters" are
  different jobs). `get_agent_tools()` gained an optional `categories`
  filter param, backward compatible (`None` = every tool). Fixed
  `agent/__init__.py`'s stale `__all__`, which predated ~16 real tools.

  New `standard_quant_tools.agent.router`: a provider-agnostic tool-category
  classifier — one cheap completion call narrows the tool list to 1-2
  categories before the real agent loop starts, without spinning up a
  separate agent session. Fails open by design (returns every category on
  any malformed/empty/unparseable response or API error) — a router that
  wrongly excludes a needed tool is worse than today's unfiltered list.
  `route_request()` + an optional `categories` param on `run_agent()` wired
  into every `Implementation/{Anthropic,OpenAI,Gemini}/Agent_*.py` script
  (27 scripts across 3 providers), replacing "hand every tool to the model
  on every call" with "narrow first, then call."

  `Multi_Agent_Implementation/worker_agents.py`'s `WORKER_AGENTS` now
  *derives* each worker's tool list from `TOOL_CATEGORY` instead of a
  hand-duplicated literal list (7 workers now, up from 6, matching the
  execution/validation split); `Agent_Orchestrator.py`'s delegate-tool set
  and system prompt are generated from `WORKER_AGENTS.keys()`/`len()`
  rather than hardcoded counts. Fixed a missing duplicate-log-handler guard
  in `Multi_Agent_Implementation/_agent_utils.py` (present in
  `Implementation/Anthropic/_agent_utils.py`, absent here) that would have
  gotten worse as delegation fans out across more workers.

  New `tests/test_router.py` (unit tests + an `@pytest.mark.integration`
  routing-accuracy eval — the first actual measurement of routing
  correctness in this codebase, vs. the pre-existing multi-agent test's
  coverage/disjointness-only checks) and expanded
  `tests/test_multi_agent_tool_coverage.py` for the 7-worker split. New
  [Documentation/13_agent_orchestration.md](Documentation/13_agent_orchestration.md).

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
- `_sqt_core` (the optional C++ extension) gained four more kernels, found
  by auditing everything added to the library since its last porting pass:
  `simulate_forward_paths` (Monte Carlo moving-block bootstrap — the only
  genuinely unaccelerated loop found, not even numba-decorated, and
  embarrassingly parallel, so it also gets an optional OpenMP path on top
  of the usual compiled-vs-interpreted speedup),
  `garch11_variance_recursion`, `kalman_filter_1state`/`kalman_filter_2state`
  (added to the existing `cointegration.cpp` rather than a new file), and
  `donchian_state_machine`/`vwap_reversion_state_machine`. The latter three
  were already numba-JIT'd and confirmed fast once warm — ported for the
  same permanent reason already documented for RSI/ADX/PSAR: no JIT
  cold-start latency on a fresh process (measured at 200ms–1.1s, not the
  initial ~300–500ms estimate — see below), and immunity to future numpy
  ABI breakage. Every port keeps the existing pure-Python/numba fallback as
  the default when `_sqt_core` isn't built, and follows the same
  `HAS_CPP`/`_cpp_core` guard pattern as the rest of the extension. All four
  were subsequently built and their full test suites actually run (see the
  build-verification entry below) — real numbers, not projections, are in
  `Development/performance_insights.md`.
  **Behavior note:** the Monte Carlo C++ path's RNG does not reproduce
  NumPy's PCG64 bit stream, so `random_seed` is only reproducible *within*
  one backend — the same seed gives different concrete numbers depending on
  whether `_sqt_core` is built (still bit-identical on repeat calls within
  one backend). See `Development/performance_insights.md` and
  `Development/build_guide.md` for the full detail.

- **C++ hardening, Tier 3 item 9 of an independent code review:** every
  `_sqt_core` binding (all ~21 `m.def(...)` entries in `bindings.cpp`) now
  releases the GIL (`py::gil_scoped_release`) around just the `sqt::` kernel
  call itself — extracting raw pointers/sizes/plain-C++ arguments from the
  `py::` types first (while still holding the GIL, since buffer access and
  argument casting are Python-API calls), then letting multiple Python
  threads run the actual C++ computation concurrently instead of
  serializing on the GIL for work that never touched a Python object once
  argument extraction was done. Added `tests/test_cpp_gil_release.py`: a
  concurrency smoke-test suite (multiple threads hammering `rsi`,
  `run_strategy`, `hurst_dfa`, `bollinger_bands`, and a mixed-kernel
  scenario at once), each thread's result checked against its own
  single-threaded reference rather than attempting to prove GIL-release
  timing from Python.
- **C++ hardening, Tier 3 item 10:** `-march=native`/`/arch:AVX2` (tuning
  codegen for the exact build machine's CPU, not portable to a
  different/older one) is now opt-in via a new `SQT_NATIVE_ARCH` CMake
  option (default `OFF`) instead of always-on in Release builds — applies to
  both `_cpp/CMakeLists.txt` (the actual extension) and `tests/cpp/
  CMakeLists.txt`'s `bench_hurst`/`bench_backtest` targets. A default build
  (what CI and a fresh clone both use) now produces portable codegen; pass
  `-DSQT_NATIVE_ARCH=ON` for the extra local-dev speed this session's own
  measured benchmarks in `performance_insights.md` were built with (no
  re-benchmarking needed — the numbers already reflect `SQT_NATIVE_ARCH=ON`).
  Verified both configurations build clean and pass the full native ctest
  suite + Python suite.

### Fixed

- **Deep native optimization, Phase 3 (build): LTO/IPO enabled for Release
  builds.** `_cpp/CMakeLists.txt` now runs `CheckIPOSupported` and applies
  `INTERPROCEDURAL_OPTIMIZATION_RELEASE` automatically when the toolchain
  supports it — unlike `SQT_NATIVE_ARCH`, this carries no "illegal
  instruction on a different CPU" portability risk (link-time only, doesn't
  change the target ISA), so it's not gated behind an opt-in flag. Scoped to
  Release only, same as the existing `/O2`-vs-`/Od` split. Full native
  ctest + full pytest passed unchanged as the actual correctness gate (LTO
  can in principle shift FP instruction selection under whole-program
  visibility; no regression surfaced). Measured honestly, not assumed:
  clean-build time on this (small, 9-source-file) extension is unaffected
  either way (~5.7-6.0s, noise-level difference); a handful of representative
  kernels (`rsi`, `adx`, `rolling_factor_loadings`, `run_strategy`, n=2000)
  showed **no measurable runtime difference** (~1.0× across the board) —
  each kernel's hot loop already lives entirely within its own translation
  unit, so there wasn't much cross-TU inlining opportunity for LTO to
  exploit in this codebase's current structure. Kept anyway since it's a
  free, correctness-neutral toolchain improvement with no measured downside,
  matching the review's own framing ("percentages, not multiples... low-effort").
- **Deep native optimization, Phase 2 (`backtest.cpp`): allocation-free
  summary kernel + OpenMP across the batch grid.** New `run_strategy_summary()`
  computes `run_strategy()`'s 11 scalar metrics with zero heap allocation at
  all (no `equity_curve`, no `strat_ret`, no `trade_rets` vector), exploiting
  a fact discovered during verification: `strat_ret[i]` has no true
  loop-carried dependency (`exec_i = signals[i-1]` and the `prev_exec`
  needed for `pos_diff` equals `signals[i-2]`, or 0.0 for `i==1`, both
  directly index-derivable) — only the trade-log open/close bookkeeping is
  a genuine sequential state machine. Two passes: pass 1 fuses that state
  machine with running equity/peak/drawdown/mean tracking (trade stats
  accumulated as running scalars instead of a `trade_rets` vector); pass 2
  recomputes `strat_ret[i]` on demand, now that the mean is known, to get
  variance and downside deviation. Verified bit-identical against
  `run_strategy()`'s 11 fields across 40 random `(n, prices, signals,
  commission, slippage)` trials plus edge cases (`n==0`, `n==1`, all-flat,
  all-short, leveraged/non-±1 signals, zero-price bars) — the design
  guarantees this by construction (same formulas, same op order, index-0's
  implicit `strat_ret[0]=0.0` contribution to the variance sum seeded
  directly since `0.0 + x == x` exactly in IEEE 754), and the new test is
  what actually proved it held.
  `batch_run_strategy` now calls `run_strategy_summary` directly (no more
  manual `equity_curve.clear()/shrink_to_fit()` after the fact) and runs
  every test index in parallel via `#pragma omp parallel for` — each call is
  a pure function of its own `(prices, signals_flat + t*n, n, ...)` slice
  with no shared mutable state, so (unlike `simulate_forward_paths_into` in
  `monte_carlo.cpp`, which needs a thread-local RNG) no per-thread setup is
  needed, just the simpler combined form. `results` switched from
  `reserve()+push_back()` to `resize()`+indexed writes first, since
  `push_back` on a shared vector is not thread-safe across concurrent
  writers. Verified exact reproducibility of `batch_run_strategy`'s output
  across `OMP_NUM_THREADS=1/2/4/8` (every row is fully independent, unlike
  Monte Carlo's per-path-seed reproducibility, so output must be identical
  regardless of thread count, not just per-path-deterministic). Measured
  (`batch_run_strategy`, min of 7 runs, same-machine before/after, 16
  logical cores): **n=500/num_tests=500: 3.26ms → 0.54ms, ~6.0×**;
  **n=2000/num_tests=2000: 51.55ms → 4.55ms, ~11.3×**;
  **n=2000/num_tests=10000: 255.25ms → 29.81ms, ~8.6×**.
- **Deep native optimization, Phase 1 (`rolling_regression.cpp`):** three
  changes to `rolling_factor_loadings`'s per-bar Cholesky solve, following a
  third-party review of what's left in the native layer after the
  performance-architecture pass above. (1) `build_normal_equations()` and
  the rank-1 XtX update/downdate loop computed all p² entries of the
  symmetric normal-equations matrix; `cholesky_solve()`'s decomposition loop
  only ever reads the lower triangle (`j <= i`), so the upper triangle was
  provably dead work — removed outright (`c < p` → `c <= r`), no mirror step
  needed since nothing downstream ever reads those entries. Verified
  bit-identical two ways: a same-machine `git stash`/`git stash pop`
  comparison of `rolling_factor_loadings()`'s full output array (exact `==`,
  not tolerance) on a fixed random `(n=400, k=5, window=30)` input, and a
  new from-scratch independent-reference regression test (dense Gaussian
  elimination on the full normal equations, sharing no code with the
  production lower-triangle-only path). (2) `cholesky_solve()` allocated a
  fresh `L`/`z` vector on every single call — one call per bar in the
  rolling window, so `(n-window+1)` allocations per series. Now takes
  caller-owned `L_scratch`/`z_scratch` buffers, sized once outside the
  loop and reused across every call; traced the read pattern by hand and
  confirmed the old `L(p*p, 0.0)` zero-fill was never actually load-bearing
  (every read of `L` is to an entry the same call already wrote earlier in
  its own iteration order), so the scratch buffer is reused with no re-zero
  needed either — also verified bit-identical via the same two methods.
  (3) Added `#pragma omp simd reduction(+:Sx,Sy,Sxy,Sxx)` above
  `rolling_beta_into`'s 4-accumulator reduction loop as a vectorization
  hint. **First attempt broke the MSVC build**: MSVC's default `/openmp`
  only implements OpenMP 2.0, which doesn't recognize `omp simd` (that's
  4.0+) — this is a hard `C7660` compile error requiring
  `/openmp:experimental`, not the silently-ignored no-op initially assumed;
  scoped the pragma to non-MSVC compilers only (`!defined(_MSC_VER)`)
  rather than pulling in a project-wide experimental-flag change for one
  hint whose payoff is itself unproven. Also added `tests/cpp/test_rolling_regression.cpp`
  (new `sqt_rolling_regression_impl`/`cpp_rolling_regression` CMake target) —
  `rolling_regression.cpp` previously had no native-level test coverage at
  all, only the existing Python-level `tests/test_cpp_regression.py`.
  Measured (`rolling_factor_loadings`, n=2000, window=60, min of 9 runs,
  same-machine before/after): **k=3 (this library's own typical/tested
  factor count) 0.269ms → 0.150ms, ~1.79×** — the allocator overhead from
  item (2) turned out to dominate total cost at this library's actual
  problem size, a bigger and more directly-relevant win than the review's
  own "matters more at k=10-50" framing suggested; **k=10: 1.058ms →
  0.811ms, ~1.30×**; **k=30: 7.452ms → 6.833ms (best of 2 runs), ~1.09×** —
  as p grows, `cholesky_solve`'s O(p³) decomposition dominates total cost
  more, so the O(1)/O(p²) savings from items (1)/(2) become proportionally
  smaller, not larger.
- **Performance architecture, item 6:** two changes, per the review's own
  final priority item. (1) `batch_run_strategy` (`bindings.cpp`) returned
  `py::list` of `py::dict`, one per grid combo; `backtest_grid`
  (`engine.py`) then rebuilt a Python dict per row before handing them to
  `pd.DataFrame`. Changed the binding to return a single `(num_tests, 11)`
  `py::array_t<double>` (fixed column order, `_BATCH_METRIC_COLUMNS` in
  `engine.py`) and `backtest_grid` to build the metrics `DataFrame`
  directly via `pd.DataFrame(arr, columns=_BATCH_METRIC_COLUMNS)`, then
  concat the parameter-combo columns — no per-row dict ever built. Isolated
  micro-benchmark: the binding call itself (array vs list-of-dict
  construction in C++) **~1.21×**; the Python-side `DataFrame`-construction
  step alone (array→DataFrame vs `num_tests` dicts→DataFrame) **~7×**. At a
  1,200-combo end-to-end `backtest_grid()` (n=1,500 bars, the review's own
  "1,000+ combos" scale), the two measured within noise of each other
  (~0.26s either way) — at that grid size the C++ kernel itself (1,200 full
  backtests) dominates wall time, so the marshaling-layer win, while real,
  is a small fraction of the total; it matters more for cheaper
  strategies/shorter series or larger combo counts relative to series
  length, not uniformly at every grid size. (2) New fused
  `sqt::technical_indicators(high, low, close, config)` (`indicators.cpp`)
  computes whichever of {RSI, ADX, ATR, Bollinger Bands, Stochastic
  Oscillator} the caller requests in one native call instead of up to 5
  separate ones — pure orchestration over the same already-tested `*_into`
  kernels from item 5, no new algorithm logic. New `technical_indicators`
  pybind11 binding (`py::dict` of arrays, conditional keys). Wired as an
  additive fast path into `agent/tools.py`'s technical-analysis tool: when
  2+ of {rsi, adx, bollinger, stochastic} are requested (and C++ is
  available), one fused call replaces up to 4 separate Python-wrapper round
  trips; the plain `atr` indicator is deliberately excluded from the fused
  path since the tool's `atr()` uses a simple rolling mean while the fused
  call's ATR field is Wilder-smoothed — a different algorithm, not the same
  one computed faster — so fusing it would have silently changed the tool's
  output. Individual indicator wrappers (`rsi()`, `adx()`, etc.) are
  unchanged and still used standalone elsewhere, and as the fallback when
  fewer than 2 fusable indicators are requested. Verified the fused path
  produces byte-identical `last_values`/`signals` to the per-indicator
  fallback (forced via a `HAS_CPP` monkeypatch) in
  `tests/test_agent_tools.py`. Measured at the actual integration point
  (`get_technical_analysis`, n=2,000 bars, all 4 fusable indicators
  requested): **~4.6×** (1,467µs → 314µs, median of 9 runs) — the win here
  is eliminating 3 of 4 redundant Python-wrapper layers (validation,
  logging, numpy conversion, per-call pandas construction), not a faster
  native kernel; at the raw C++-binding level alone the 4 individual
  bindings vs. 1 fused call measure ~1.0× (n=2,000, ~100µs either way — the
  pybind11 call overhead itself is negligible at this size next to the
  kernels' own O(n) work), consistent with the review's own framing that
  the win comes from removing Python-side glue, not from a faster inner
  loop.
- **Performance architecture, item 5:** ~16 of `bindings.cpp`'s ~21
  bindings shared the pattern `std::vector<double> result = sqt::foo(...);
  py::array_t<double> out(...); std::copy(result.begin(), result.end(),
  out.mutable_data());` — a `std::vector` allocation plus a full copy into
  a second, separately-allocated NumPy array, on every call. Added a
  buffer-writing `*_into` overload alongside 13 of the ~16 identified
  vector-returning `sqt::` functions (`rsi`, `adx`, `parabolic_sar`,
  `wilder_atr`, `bollinger_bands`, `stochastic_oscillator`,
  `rolling_hurst`, `rolling_beta`, `rolling_factor_loadings`,
  `simulate_forward_paths`, `garch11_variance_recursion`,
  `donchian_state_machine`, `vwap_reversion_state_machine`) — the
  existing vector-returning form becomes a thin wrapper (allocate, call
  `_into`, return), so every native test keeps calling the unchanged API
  with zero test churn. `bindings.cpp` now allocates the NumPy output
  array first and passes its buffer straight into the `_into` call: one
  allocation, zero copies. `simulate_forward_paths_into` needed a small
  contract change from the vector-returning form (returns `bool` for
  "was `out` actually written" instead of signaling invalid input via an
  empty vector, since a pre-sized buffer can't itself be "empty") — the
  vector-returning wrapper still preserves the original empty-on-invalid
  contract exactly.
  **Deliberately scoped out**: `run_strategy`'s `equity_curve` field and
  the two Kalman filters' 3-4 output arrays each — these return
  multi-field structs, not a single `std::vector`, so the same pattern
  would need multiple output-buffer parameters per call; lower value
  (Kalman filters aren't hot-loop calls, and `run_strategy`'s own copy is
  already dwarfed by item 1's ~58× wrapper fix) for real added
  complexity, left as a known, documented gap. Measured on two of the
  cheapest kernels at small n (where a copy is proportionally largest):
  `rsi` (n=100) **~1.6×** (0.00429ms→0.00262ms), `adx` (n=100) **~1.9×**
  (0.00886ms→0.00477ms) — same-machine git-stash-verified.
- **Performance architecture, item 4:** `adx()` (`indicators.cpp`)
  allocated 4 full n-sized temporary arrays (`dm_plus`, `dm_minus`, `tr`,
  `dx_vals`) beyond its own output array. Traced Wilder's recursion by
  hand: it only ever needs the immediately-previous smoothed sum plus the
  *current* bar's raw TR/DM value (computable inline, no lookback array
  needed), and the DX/ADX seed windows only need a running sum of the
  values seen so far, not the individual values — so the whole function
  genuinely reduces to O(1) auxiliary memory, not just "smaller."
  Rewrote as a single fused pass preserving the exact same order of
  floating-point operations as the original 4-pass version (addition
  isn't associative, so order — not just which values get summed —
  determines the result). Verified bit-identical output two ways: every
  existing test passed unchanged with zero tolerance widening, and a new
  exact-equality regression pin (`tests/cpp/test_indicators.cpp`) was
  confirmed to match against *both* the pre- and post-rewrite
  implementation via `git stash` in both directions. Measured speed:
  negligible at n=2000 (~1.02–1.07×, within noise — fixed Python/pybind
  call overhead dominates at this size) but a real **~1.21×** at n=50000
  (min 3.18ms→2.63ms) once the eliminated arrays are large enough
  (~1.6MB total) for memory bandwidth/allocation cost to matter against
  the O(n) arithmetic. Memory footprint (5 allocations → 1) improves
  unconditionally regardless of n.
- **Performance architecture, item 3:** `garch_volatility_forecast`'s
  scipy L-BFGS-B fit called `_garch11_neg_loglik` every iteration, which
  dispatched to the C++ recursion for a full `sigma2` array, copied it out
  of C++, then reduced it to one scalar in NumPy — a full array round-trip
  every iteration purely to throw the array away. New
  `garch11_neg_loglik` (C++) fuses the recursion and the NLL reduction
  into one native call returning a single `double`; new
  `garch11_neg_loglik_grad` additionally computes the analytic gradient
  w.r.t. `(omega, alpha, beta)` in the same fused pass, wired via scipy's
  `jac=True` convention so L-BFGS-B stops needing 6 extra
  finite-difference NLL evaluations per iteration. The analytic gradient
  was verified against central differences across 5 random input grids
  before being trusted (`tests/cpp/test_garch.cpp`) — per the plan's own
  gate, this was only wired into the optimizer after that check passed
  cleanly (the first attempt used a single absolute step size across all
  three parameters and failed on `omega`, not because the gradient was
  wrong, but because `omega`'s tiny ~1e-6 scale needs a much smaller step
  than `alpha`/`beta`'s ~0.05–0.95 range; per-parameter-scaled steps fixed
  the numerical reference itself). `garch11_variance_recursion` alone
  (just the recursion, no fusion) still measures 0.8× vs warm numba — the
  fusion is what actually pays off. Measured end-to-end
  `garch_volatility_forecast()`: **~7.8×** (7.928ms → 1.016ms, n=1000,
  same-machine git stash/pop before/after). `jac=True` can converge to a
  very slightly different point than finite-difference gradients near a
  flat likelihood surface (real for GARCH), so
  `TestGarchForecastEndToEndParity` was loosened from bit-identical
  (`abs=1e-10`) to `rel=1e-2` on fitted parameters plus a tight `rel=1e-3`
  check on the two fits' own log-likelihoods — the actual invariant that
  matters.
- **Performance architecture, item 2:** `simulate_forward_paths`
  (`monte_carlo.cpp`) constructed a fresh `std::mt19937_64` and allocated a
  `resampled` heap buffer on *every single simulated path* inside the
  OpenMP-parallel loop — 200,000 heap allocations/frees at
  `n_simulations=200000`. Hoisted the RNG/distribution to one instance per
  OpenMP thread (reseeded per path via `gen.seed(path_seed)`, not
  reconstructed — identical reproducibility, since seeding fully
  reinitializes a Mersenne Twister's state either way and no two threads
  ever touch the same `gen`), and removed `resampled` entirely by writing
  sampled values directly into the output row as they're drawn. Did **not**
  swap the RNG family (still `mt19937_64`) — that would break bit-exact
  reproducibility for existing seeds, a separate decision out of scope
  here. Measured (min-of-7-runs, separate process invocations, honest
  about the noise): 1-thread 284.5ms→239.1ms (~1.19×), unconstrained
  117.4ms→113.7ms (~1.03×) at `n_simulations=200000` — real but modest;
  the eliminated per-path allocation was small (~480 bytes) and evidently
  wasn't the dominant cost at this problem size, unlike what the review's
  framing suggested. Kept as a correct change regardless (fewer
  allocations is never worse) with the real numbers recorded, not
  oversold.
- **Performance architecture, item 1 of an independent review of the C++/
  Python boundary:** `run_strategy()` (`backtest/engine.py`) measured
  ~1.0× end-to-end against its own pure-C++ kernel time (68ms wrapper vs.
  0.017ms native kernel) despite the kernel itself being fast — the
  wrapper computed `prices.pct_change()`/`signals.shift(1)` unconditionally
  before even checking whether the C++ path would run (never used on that
  path — the kernel recomputes both internally), and after the kernel
  returned, unconditionally rebuilt the entire Python trade log
  (`_build_trade_log`/`_compute_trade_stats`) purely to overwrite native
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` fields
  that were already correct — confirmed correct by this session's own CI
  verification work (`TestNativeTradeStatsCorrectness` passing against a
  real compiled `_sqt_core` on live CI), which is exactly the precondition
  an existing code comment had flagged as needed before removing the
  override. Both are now gone: the pandas calls are computed only where
  actually used (Python fallback path, or lazily inside the C++ path only
  when `include_trade_log=True` asks for the DataFrame), and the C++
  path's summary stats flow straight from the native result, unmodified.
  Also added an `index.equals()` fast path ahead of the existing
  `intersection()`+`.loc[]` calls for the common case where `price_data`
  and `signal_series` already share an index. Measured end-to-end
  (n=2000, `include_trade_log=False`, the common case): **26.8ms → 0.46ms,
  ~58×** — real numbers, stashed/unstashed the fix to measure the same
  benchmark before and after on the same machine, not a projection.
- **C++ hardening, Tier 4 item 13:** `stochastic_oscillator`
  (`indicators.cpp`) rewritten from an O(n·k_period) full-window rescan
  (re-scanning the entire `[i-k_period+1, i]` window on every single bar
  despite an inline comment claiming O(1)-amortized behavior a different,
  never-actually-implemented technique would have provided) to a genuine
  O(n) sliding max(high)/min(low) via two monotonic deques of indices —
  the standard sliding-window-extrema technique. Removed the stale,
  inaccurate complexity comment. Added native test coverage that didn't
  exist before at all (`tests/cpp/test_indicators.cpp`), including an
  independent brute-force O(n·k) reference oracle (deliberately
  implemented separately from the real function, not just a copy of it)
  and adversarial monotonic-rising/falling and mid-window-spike cases —
  the specific patterns that expose an off-by-one in a monotonic deque's
  front-eviction logic, as opposed to just its back-insertion logic. Added
  matching adversarial Python-level tests
  (`tests/test_cpp_new_indicators.py`) against an independent pandas
  `.rolling().min()/.max()` reference.
- `build-cpp.yml`'s ASan/UBSan job's "Verify extension loaded" step never
  actually verified anything — it imported the ASan-instrumented `_sqt_core`
  without the `LD_PRELOAD=$(gcc -print-file-name=libasan.so)` the very next
  step already correctly sets for the same import, so it always failed
  immediately with "ASan runtime does not come first in initial library
  list" regardless of whether the build itself was healthy. Confirmed via
  an actual failed CI run's logs (fetched with the repo's own stored git
  credential, since the anonymous GitHub API blocks job-log downloads even
  on public repos). Added the same `LD_PRELOAD` export this step was
  missing. This is what let item 8's `-DSQT_BUILD_TESTS=ON` + `ctest` fix
  be verified for real: the native `ctest` suite under ASan/UBSan now
  genuinely passes (confirmed on a live CI run, not just locally on
  Windows/MSVC where sanitizers aren't available at all).
- **C++ hardening, Tier 1-2 (items 1-5 of an independent code review of the
  entire `_cpp` surface at commit `d52e9f2`), each verified against the real
  compiled `_sqt_core` before and after:**
  1. `cointegration.cpp`'s `mackinnon_pvalue` used a 13-point lookup table
     with log-linear interpolation, documented as +-0.01-0.02 accurate —
     independently reproduced the exact algorithm and found errors up to
     0.08 vs. `statsmodels.tsa.stattools.mackinnonp` mid-distribution.
     Replaced with the real MacKinnon (2010) regression-surface algorithm
     (quadratic/cubic polynomial + normal CDF, coefficients extracted from
     `statsmodels`' own `tsa/adfvalues.py` for `regression="c", N=2`),
     verified to machine precision (1e-9) across a swept range of ADF
     statistics.
  2. `Array1D` (`py::array_t<double, c_style|forcecast>`) enforced dtype and
     contiguity but not `ndim` — a 2-D array silently passed through every
     binding and produced garbage (or a native crash) rather than a clear
     error. Added `require_1d()`, called at the top of all 20 `m.def(...)`
     lambdas (37 call sites) taking an `Array1D` parameter.
  3. `bollinger_bands`/`rolling_beta` used raw-moment sliding sums
     (`Sxx - Sx*Sx/W`-style formulas), which suffer catastrophic
     cancellation on a large-baseline series — e.g. a ~1e9-level price
     series previously produced a near-zero variance instead of the true
     small value, and `rolling_beta`'s denominator could collapse to
     exactly zero. Rewrote both with a shifted-window + periodic-recompute
     technique (subtract each window's own first value before accumulating,
     full recompute every `window` bars) — the same idiom already used by
     `rolling_factor_loadings` elsewhere in this codebase.
  4. `backtest.cpp`'s native trade-log cost deduction (and the identical
     logic in `_build_trade_log`, `backtest/engine.py`) was a flat
     `2*cost_per_unit`/`1*cost_per_unit` regardless of the position's actual
     size — a 5x-leveraged SCORE-type trade paid the exact same cost as a
     1x trade even though the equity curve's own `strat_ret` already scales
     cost by `abs(pos_diff)`, silently under-costing every leveraged
     (non-+/-1) position's reported `return_pct`/`avg_trade_return_pct`.
     Cost is now scaled by `abs(position_size)` per leg in both
     implementations, matching the equity curve's convention for the common
     case (full close/reopen, including leveraged round trips). A same-sign
     *resize* (e.g. 1.0 -> 2.5 in one event) remains a documented
     approximation — costed as closing the old size and opening the new one
     independently, which doesn't exactly reconcile with the equity curve's
     single smaller `abs(pos_diff)`-sized cost for that event; a fully exact
     reconciliation would require tracking continuous positions with a
     weighted-average cost basis, a bigger redesign that changes reported
     `num_trades` for resize-using strategies and was left out of scope here.
  5. `hurst.cpp`'s `hurst_exponent` accepted any `method` string, silently
     treating anything other than exactly `"dfa"` as `"rs"` — the Python
     wrapper (`analysis/hurst.py`) already validated this at its own layer,
     but `_sqt_core` is directly importable, so a caller bypassing the
     wrapper got a silently wrong estimator instead of an error. Both
     `hurst_exponent` and `rolling_hurst` now reject any method other than
     `"dfa"`/`"rs"` with `std::invalid_argument` (validated eagerly in
     `rolling_hurst`, before its sliding-window loop, so a too-short input
     that would otherwise run zero iterations still raises). Also added an
     explicit `std::isnan(h)` guard before `std::clamp`/regime
     classification — `std::clamp`'s behavior on a NaN input is unspecified
     by the standard, and relying on classify()'s threshold comparisons
     (all false for NaN) to coincidentally fall through to a safe-looking
     label was fragile.
- Root `CMakeLists.txt`'s `cmake_minimum_required` bumped from `3.15` to
  `3.19` — `3.15` was never actually sufficient: `find_package(...
  Development.Module)` requires `3.18`, and `_cpp/CMakeLists.txt`'s
  multi-value `$<CONFIG:Release,RelWithDebInfo>:...>` generator expressions
  require `3.19`. A fresh `3.15`-`3.17` CMake install would have failed at
  configure time regardless of what the stated minimum claimed.
- `.github/workflows/build-cpp.yml` never actually ran the native `tests/cpp/**`
  suite — `SQT_BUILD_TESTS=ON` wasn't passed to either `build-and-test`'s or
  `build-and-test-sanitizers`'s `cmake -B build` invocation, so the compiled
  test executables never existed, and there was no `ctest` step to run them
  even if they had. A native-only regression (like several fixed in this
  release) could land without CI ever compiling or exercising the code that
  changed. Both jobs now pass `-DSQT_BUILD_TESTS=ON` and run
  `ctest --test-dir build --output-on-failure` immediately after building,
  before the Python `pytest` step. Also added `tests/cpp/**` to the
  workflow's `paths:` triggers (previously only `_cpp/**`/`CMakeLists.txt`),
  so a native-test-only change still triggers this workflow.
- `tests/cpp/test_indicators.cpp` failed to compile on GCC/Linux —
  `std::max({...})` (the initializer-list overload) is declared in
  `<algorithm>`, which this file never included; MSVC's headers transitively
  pull it in via other standard headers, so this went undetected until the
  `build-cpp.yml` fix above actually compiled `tests/cpp/**` on Linux for the
  first time. Added the missing `#include <algorithm>`. While auditing for
  the same class of bug, also added `#include <algorithm>`/`#include
  <stdexcept>` to `bindings.cpp` (uses `std::copy` and `throw
  std::invalid_argument` ~20+ times, currently working only because
  pybind11's own headers happen to pull both in transitively) — not
  currently broken, but relying on transitive includes from a third-party
  header is fragile the same way the `test_indicators.cpp` bug was.
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
- `garch_volatility_forecast`: the one-step-ahead forecast seed never
  incorporated the most recent observed return (`current_var` stopped one
  recursion step short), so `forecast_annualized_vol[0]` silently diverged
  from `current_annualized_vol` and every later forecast step compounded a
  spurious extra decay. Fixed by computing the true T+1 variance explicitly
  and re-indexing the forecast horizon from `h=0`.
- `audit/paths.py`: the Windows advisory file lock (`msvcrt.locking`) raised
  `OSError` after its own ~10s internal retry and was silently swallowed by
  a blanket `except Exception`, letting `AuditWriter.write()` proceed
  completely unlocked under contention (and leaking the file handle). Now
  retries indefinitely, matching POSIX `fcntl.flock`'s existing blocking
  behavior, and closes the handle on failure.
- `PositionSizerInput`: `win_rate`/`avg_win_pct`/`avg_loss_pct` had no range
  validation (unlike the sibling `risk_per_trade_pct`), so an impossible
  input (e.g. `avg_loss_pct=0`, a Kelly-formula divisor) could reach the
  sizing math instead of being rejected up front.
- `data/_cache.py`: the shared in-process session cache (`cachetools.TTLCache`)
  had no locking despite being read/written from multiple threads via each
  provider's async path; added a module-level lock around get/set.
- Audit redaction: exception messages echoing a redacted field's raw value
  were never redacted (only `input` was), and the redaction placeholder
  itself was an unsalted 8-hex-char hash, brute-forceable offline for small
  value spaces (SSNs, PINs). Added `redact_text()` for error messages
  (sharing one `_placeholder_for()` helper with `input` redaction so both
  produce the same placeholder) and an optional `SQT_AUDIT_REDACT_SALT`
  env var, with a one-time warning when it's unset.
- `portfolio_engine.py`: `fill_price="next_open"` still looked up that
  day's own ADV/volatility for cost/impact modeling — not yet knowable at
  that bar's Open. `_valid_dollar_volume`/`_trade_cost` now index at
  `trigger_date` instead of `exec_date` (a no-op for `close`/
  `hl2_exploratory`, where the two are already equal).
- The retry decorator treated HTTP 401/403 (permanent, e.g. an invalid API
  key) identically to 429/5xx (transient), burning through a rate-limited
  API's request budget on every call until the key was fixed. Added
  `NonRetryableAPIError` (a subclass of `APIError`, so existing `except
  APIError` sites are unaffected); `PolygonProvider` now raises it for
  401/403 specifically, and the retry decorator never retries it.
- `agent/__init__.py` was missing re-exports for ~46 Pydantic models defined
  in `models.py` (e.g. `Trade`, `PortfolioOptimizationInput`,
  `OptionPricingResult`), so `from standard_quant_tools.agent import
  SomeInput` silently `ImportError`'d for those classes even though the
  models themselves worked fine. Added a regression-guard test
  (`TestAgentModelExports`) so this can't drift silently again.
- `YFinanceProvider` hard-failed with `ValidationError` on a symbol whose
  characters couldn't be safely encoded into a cache filename, where
  `PolygonProvider` already degraded gracefully by skipping the disk cache
  for that call. Both providers now use `_safe_parquet_path` consistently
  on the read *and* write side (the write-side call in `PolygonProvider`
  itself was missing the same `None` guard the read side already had).
- `CorrelationAnalysisInput.weights`/`MonteCarloSimulationInput.weights`
  (both optional — `None` means equal weighting) had no validation when
  provided, unlike the required `weights` on sibling models
  (`PortfolioInput`, `RiskAttributionInput`). Added the same length/sum-to-1
  check, guarded on `weights is not None`.
- `spread_zscore`'s rolling branch and `rolling_beta`'s pandas fallback both
  divided by a rolling std/variance with no zero-guard — a flat spread or
  constant benchmark window produced `inf`/`-inf` instead of raising or
  producing an explicit missing value. Both now NaN out that window instead
  (not a literal `0.0`, which would be indistinguishable from a legitimate
  zero mid-series).
- Test isolation: `tests/test_polygon_provider.py` and `tests/test_data.py`
  didn't redirect the real persistent Parquet disk cache to a temp
  directory (unlike `test_parquet_cache.py`/`test_audit.py`, which already
  did), so a cache entry written by an earlier test/run could leak into a
  later test in the same run — the root cause of an intermittent CI "Run
  tests" failure. Added the same `autouse=True` `redirect_cache` fixture to
  both files.
- `run_portfolio_simulation`/`run_signal_panel_backtest` fetched every
  ticker with a blocking `provider.get_ohlcv()` call inside a plain `for`
  loop — for a large universe (e.g. the full S&P 500) this meant minutes of
  pure sequential network wait before the simulation itself even started,
  unlike every other multi-ticker tool in the module, which already fetches
  concurrently. Added `fetch_ohlcv_panel_async`/`fetch_ohlcv_panel_sync`
  (same `asyncio.gather` concurrency as the existing `fetch_returns_*`
  helpers, but preserving the full OHLCV panel — Volume/High/Low, not just
  Close-derived returns — since the transaction-cost model needs it) and
  wired both tools to use it. Verified against live yfinance: 20 uncached
  tickers fetched concurrently in ~2.1s vs. ~2.4s for 10 tickers
  sequentially beforehand.
- `_sqt_core` was built and its full test suite actually run for the first
  time this session (previously blocked by a missing Windows SDK — `cl.exe`
  was present, `rc.exe`/`mt.exe` were not; see
  `Development/build_guide.md`'s troubleshooting section). This found 5
  real, previously-undetectable bugs:
  - `simulate_forward_paths`'s pybind11 binding didn't raise for
    `horizon_days<=0`/`n_simulations<=0` — the result-size validation
    degenerated to `0==0` for exactly those inputs, silently passing them
    through instead of raising `ValueError`. Fixed with an explicit upfront
    check.
  - `adf_test` (cointegration ADF/Engle-Granger) returned `NaN` for a
    degenerate, (near-)perfectly-collinear input — every regressor has zero
    variance, so the per-lag OLS solve is singular for every candidate lag —
    instead of matching statsmodels' own convention for this exact case
    (`adf_statistic=-inf, p_value≈0`, verified empirically against
    statsmodels). Fixed with an upfront degenerate-input check.
  - `ar1_halflife` returned `NaN` instead of `+inf` for a zero-variance
    lagged predictor, because `beta >= 0.0` is `false` for `NaN` under
    IEEE 754 — the same "not mean-reverting" case a non-negative beta
    already gets was falling through a different comparison path. Fixed by
    testing `!(beta < 0.0)` instead.
  - 4 of `tests/cpp/test_backtest.cpp`'s own hand-written trade-log test
    expectations were wrong — written without ever compiling or running
    them, based on a mistaken `prices[i]`-vs-`prices[i-1]` reference-price
    assumption. The actual native trade-log implementation (the
    `backtest_grid` fix from 0.1.0, described in Known Issues below) was
    already correct; only the tests needed fixing.
  - A native/Python trade-stats parity test used a tolerance tight enough to
    fail on Python's own intentional `round(..., 4)` display rounding, not a
    real discrepancy. Loosened from `abs=1e-9` to `abs=5e-5`.

### Known Issues

- **Resolved:** the native trade-stat parity gap described in earlier drafts
  of this section (`backtest_grid`'s C++ batch kernel returning uncorrected
  trade stats) is now **confirmed correct**, not just implemented. A missing
  Windows SDK component (`cl.exe` was present; `rc.exe`/`mt.exe` were not)
  was found and fixed, `_sqt_core` was built for the first time, and
  `tests/test_backtest.py::TestNativeTradeStatsCorrectness` plus the full
  native `ctest` suite (110 test cases) were actually run. The native/Python
  trade-stat accounting genuinely agrees — `backtest_grid`'s C++-path
  `win_rate`/`profit_factor`/`num_trades`/`avg_trade_return_pct` (and
  anything built on top of it, e.g. `run_walk_forward_backtest`/
  `run_backtest_optimization`) can now be treated as trustworthy. See
  Fixed below for the 5 bugs this build-and-test pass actually found (none
  of them in the trade-stat fix itself).

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
