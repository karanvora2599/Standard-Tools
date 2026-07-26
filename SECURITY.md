# Security Policy

## Supported Versions

Standard Quant Tools is pre-1.0 (`0.x`). Only the latest released version on
`main` receives security fixes — there are no maintained release branches yet.

| Version | Supported |
|---|---|
| `main` / latest `0.x` tag | :white_check_mark: |
| older `0.x` tags | :x: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report it privately by emailing **kv2154@nyu.edu** with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal code snippet is ideal).
- The version/commit hash you tested against.

You should receive an acknowledgement within **7 business days**. This is a
small, single-maintainer project — there's no formal SLA, but confirmed
vulnerabilities will be prioritized over regular feature work, and a fix or
mitigation timeline will be communicated back to you once triaged.

If the report is accepted, a fix will be released and credited to you (unless
you prefer to remain anonymous) in the [CHANGELOG](CHANGELOG.md). If it's
declined (e.g. not reproducible, out of scope), you'll get an explanation.

## Scope Notes

This library fetches market data from third-party providers (currently
`yfinance`) and executes user-supplied strategy code (custom signal
callables passed to `run_custom_signal_backtest` / `run_signal_panel_backtest`
/ `backtest_grid`). Relevant classes of concern:

- **Data provider trust**: OHLCV/fundamentals data returned by `yfinance` is
  not authenticated or cryptographically verified — this library treats it
  as trusted input from the configured provider, consistent with
  [`Documentation/11_data_quality.md`](Documentation/11_data_quality.md)'s
  explicit "not a new, more reliable data source" scope statement. Report
  data-quality issues via the normal issue tracker, not as a security
  vulnerability, unless they represent an actual code-execution or
  injection risk.
- **User-supplied callables**: `run_custom_signal_backtest` and similar
  tools execute a Python callable you provide directly, with no sandboxing.
  This is by design (the library is a backtest/analysis engine, not a
  sandboxed execution environment) — running untrusted third-party signal
  code through these tools is your own trust boundary to manage, not a
  vulnerability in the library itself.
- **Agent tool dispatch** (`standard_quant_tools.agent.dispatch`): tool
  arguments are validated through Pydantic models before reaching any
  underlying function, and filesystem-path-adjacent inputs are further
  restricted — `backtest.artifacts.save_artifact`'s `run_id`/`name` are
  validated against a plain-slug pattern and the resolved path is confirmed
  to stay inside `SQT_RUNS_DIR`, and `data.yfinance_provider`'s Parquet
  cache path similarly contains the symbol/date/interval used to build the
  cache file path before any read/write. If you find an input that bypasses
  one of these checks and reaches an unintended path or code branch, that's
  a legitimate report — please include the specific tool name and payload.
- **Audit trail integrity**: `standard_quant_tools.audit`'s decision-record
  log is hash-chained (`prev_record_hash`/`record_hash` on every JSONL
  record, checked by `verify_audit_log_integrity()`) **across every
  calendar day**, not just within one day's file — an independent chain
  index (`_chain_index.jsonl`, checked together with every day file by
  `verify_audit_trail_integrity()`) links each new day's first record to the
  previous active day's last hash, so deleting or wholesale-regenerating an
  entire day's file is detectable too, not just editing one record within
  it. This raises the cost of a convincing forgery — an attacker now has to
  rewrite both the day file and the index, consistently, to hide tampering
  — but does **not** eliminate it: an attacker who rewrites both end-to-end
  and keeps them internally self-consistent is still undetected, because
  there is no external anchor (e.g. a cryptographic signature verifiable
  independently of these files) yet. That's tracked as a future
  Ed25519-checkpoint-signing phase, not yet implemented. Until then, the log
  is tamper-evident append-only advisory evidence, not a
  cryptographically-signed ledger. A gap in `verify_audit_log_integrity()`'s
  or `verify_audit_trail_integrity()`'s tamper detection is a legitimate
  report.
