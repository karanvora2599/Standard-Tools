# Auditability

Every call routed through `dispatch()` can produce an immutable **decision
record**: the tool name and inputs, the market data it pulled (with content
hashes), whether the C++ accelerated path was available, a hash of the
output, and how long it took. Records are also **hash-chained** — each one
commits to the previous record's hash — so `verify_audit_log_integrity()`
can detect a line that was edited, removed, reordered, or inserted after
the fact (see [Tamper evidence](#tamper-evidence-hash-chain) below).
`verify_replay()` re-runs a recorded call and reports whether the data and
output still match — enough to tell a stale or tampered cache apart from a
genuine code change. JSONL writes are guarded by a cross-process file lock,
so two callers writing at once can't interleave and corrupt a day's file.

Nothing here runs automatically. The package attaches only a `NullHandler`
by default, and decision records are the only side effect that requires
explicit opt-out (they're on by default once you call `dispatch()`).

---

## Decision records

**Enabled by default.** Every `dispatch()` call writes one JSON line to a
daily file under `SQT_AUDIT_DIR` (default `~/.cache/standard_quant_tools/audit/`),
e.g. `2026-07-19.jsonl`.

```python
from standard_quant_tools.agent import dispatch

result = dispatch("run_sma_backtest", {
    "symbol": "AAPL",
    "start_date": "2022-01-01",
    "end_date": "2022-06-01",
    "strategy_type": "sma_crossover",
    "parameters": {"fast_period": 10, "slow_period": 50},
})
```

produces a record like:

```json
{
  "request_id": "de06f2e1b7db47d7938069970bdc10ab",
  "timestamp_utc": "2026-07-19T16:19:37.695789+00:00",
  "tool_name": "run_sma_backtest",
  "input": {"symbol": "AAPL", "start_date": "2022-01-01", "end_date": "2022-06-01", "...": "..."},
  "data_sources": [
    {"symbol": "AAPL", "start": "2022-01-01", "end": "2022-06-01", "interval": "1d",
     "source": "live_fetch", "content_hash": "1d975f555f10aeb8"}
  ],
  "cpp_available": false,
  "n_workers": null,
  "duration_ms": 6765.8,
  "output_hash": "8a2b0ca80ac84ba1",
  "status": "ok",
  "error_type": null,
  "error_message": null,
  "git_commit_sha": "463b874696913a8ec813c9a789465a443b66a15b",
  "package_version": "0.1.0",
  "random_seed": null,
  "strategy_source_hash": "9f2a7c1e4b8d0356",
  "prev_record_hash": "0000000000000000",
  "record_hash": "7c3a9e21f6b4d805"
}
```

`data_sources` has one entry per OHLCV pull, tagged `disk_cache`,
`live_fetch`, or `session_cache`, with a content hash of the DataFrame
actually used. `session_cache` fires on every in-memory-cache hit inside
`YFinanceProvider.get_ohlcv()`, not just misses — a call that only ever
touches the process's warm cache still produces a complete, auditable data
lineage instead of an empty `data_sources` list. Failed calls still produce
a record — `status: "error"` with `error_type` / `error_message` set, and
`output_hash: null`.

`git_commit_sha` and `package_version` are reproducibility provenance: the
exact commit and library version that produced this record, so a replay
months later can tell "the code changed since this ran" apart from "the
underlying data changed." Both are best-effort — `git_commit_sha` is `null`
outside a git checkout or when git isn't installed; resolving them never
raises or blocks the tool call itself.

`random_seed` is populated whenever the tool's input model has a
`random_seed` field (currently `get_robustness_diagnostics`, whose block
bootstrap needs one for reproducibility) — `null` for every other tool.
`strategy_source_hash` is populated whenever the input model names a
built-in strategy via a `strategy` or `strategy_type` field (e.g.
`run_sma_backtest`, `run_walk_forward_backtest`) — a content hash of that
strategy function's source, so a replay can distinguish "the code for this
strategy changed" from "the market data changed." `null` for tools with no
such field (e.g. `run_custom_signal_backtest`) or if the named strategy
isn't found in the registry — resolving it never raises or blocks the call.

`prev_record_hash`/`record_hash` are the hash-chain link that makes the log
tamper-evident — see [Tamper evidence](#tamper-evidence-hash-chain) below
for what they cover and how to verify them.

### Env vars

| Variable | Default | Purpose |
|---|---|---|
| `SQT_AUDIT_ENABLED` | `1` | Set to `0` to disable decision-record writes entirely |
| `SQT_AUDIT_DIR` | `~/.cache/standard_quant_tools/audit/` | Where JSONL files are written |

### Scope

Decision records are written by `dispatch()` — the documented integration
surface for OpenAI/Anthropic tool calling (see
[07_agent_tools.md](07_agent_tools.md)). Calling a tool function directly
(`run_sma_backtest(BacktestInput(...))`, bypassing `dispatch`) does not
produce a decision record.

---

## Tamper evidence (hash chain)

Each record commits to the record immediately before it in the same day's
file: `record_hash` is a content hash of the record itself (every field
except `record_hash`), and `prev_record_hash` is the *preceding* record's
`record_hash` — `"0" * 16` (the genesis hash) for the first record of a
day's file. Editing a record's content after the fact changes its
`record_hash`; removing, reordering, or inserting a record breaks the
`prev_record_hash` link for every record that follows it.

```python
from standard_quant_tools.audit import verify_audit_log_integrity

problems = verify_audit_log_integrity(
    "~/.cache/standard_quant_tools/audit/2026-07-19.jsonl"
)
if problems:
    for p in problems:
        print(p)
else:
    print("chain intact")
```

`verify_audit_log_integrity(path: str | Path) -> List[str]` walks the file
top to bottom and returns one human-readable problem per broken link —
empty if the file is clean or doesn't exist. Each problem names the
offending `request_id` and line number, and distinguishes:

- **content altered** — `record_hash` no longer matches that line's
  recomputed content hash.
- **chain broken** — `prev_record_hash` doesn't match the preceding line's
  `record_hash` (a record was edited, removed, reordered, or inserted).

**What this does and doesn't guarantee:** an attacker who edits one line
without also rewriting every later line's `prev_record_hash`/`record_hash`
to match is caught. An attacker who consistently rewrites the *entire* file
from the edited point forward is not — there's no external anchor (e.g.
signing each day's final hash into a separate system) to detect a
wholesale rewrite; this function doesn't attempt that. Not wired into the
`sqt` CLI — call it directly from Python.

### Concurrent writes

Each day's JSONL file is protected by a small sidecar lock file (e.g.
`2026-07-19.jsonl.lock`, not the growing JSONL file itself) held for the
duration of one record's read-modify-write — read the current last line's
hash to compute `prev_record_hash`, append the new line, release.
`AuditWriter.write()` acquires it via `msvcrt.locking` on Windows or
`fcntl.flock` on POSIX before touching the file, so two processes (or
threads/async tasks) writing at the same instant can't interleave and
corrupt a line or break the hash chain. If neither locking primitive is
available, writes proceed unlocked rather than blocking a tool call on a
missing OS feature — best-effort, not a hard guarantee.

---

## Replay verification

```python
import json
from standard_quant_tools.agent import verify_replay

with open("path/to/2026-07-19.jsonl") as f:
    record = json.loads(f.readline())

result = verify_replay(record)
print(result.output_match)          # True / False / None
print(result.data_source_matches)   # per-source hash comparison
print(result.notes)                 # human-readable diagnosis, if anything mismatched
```

`verify_replay` re-runs the tool with the stored input, re-fetches (or reads
from the still-immutable Parquet cache) each data source, and compares
hashes:

- **Data matches, output matches** — fully reproducible.
- **Data mismatch, output mismatch** — the data provider likely revised
  historical values since the record was written.
- **Data mismatch, output matches** — the data changed but this particular
  metric happens to be insensitive to it (e.g. a scale-invariant statistic
  under a price rebase) — still worth a closer look.
- **Data matches, output mismatch** — the code/logic changed since the
  record was written.

`data_source_matches` also reports a data source that was in the original
record but **disappeared** from the replay (e.g. the tool changed which
tickers/ranges it fetches, or a symbol was dropped along the way) — that
entry gets `new_hash: null` and `match: false`, the same as any other
mismatch, rather than being silently left out just because the replay
never touched it. Comparing only `set(new_sources)` against the original
would have hidden exactly this case.

Note that `verify_replay` re-executes the tool function directly (not
through `dispatch()`), so it does not itself write a new decision record.

---

## Correlated logging

`configure_logging()` is an opt-in helper — it is never called
automatically, consistent with the library's "consumer configures logging"
default (see `standard_quant_tools/__init__.py`).

```python
from standard_quant_tools.agent import configure_logging
import logging

configure_logging(level=logging.DEBUG)  # or level=logging.INFO for less noise
```

This attaches a formatted handler to the `standard_quant_tools` logger that
includes a `request_id` field:

```
2026-07-19 16:19:31 DEBUG    [de06f2e1b7db47d7938069970bdc10ab] standard_quant_tools.agent.tools: [dispatch] → run_sma_backtest  args=['symbol', 'start_date', ...]
2026-07-19 16:19:37 DEBUG    [de06f2e1b7db47d7938069970bdc10ab] standard_quant_tools.data.yfinance_provider: [fetch] ✓ AAPL  105 rows  6421ms
2026-07-19 16:19:37 DEBUG    [de06f2e1b7db47d7938069970bdc10ab] standard_quant_tools.agent.tools: [dispatch] ✓ run_sma_backtest  completed in 6765ms
```

The `request_id` matches the `request_id` field in that call's decision
record, so a log line and a JSONL record can always be cross-referenced.

If you build your own logging handlers instead of using
`configure_logging()`, you can still get correlation by attaching
`RequestIdFilter` to your handler directly:

```python
from standard_quant_tools.agent import RequestIdFilter

my_handler.addFilter(RequestIdFilter())
```

`RequestIdFilter` must be attached to a **handler**, not a logger — logger-level
filters only run for records originating at that exact logger, and would miss
records from other modules (e.g. `indicators.momentum`) that propagate up
through the `standard_quant_tools` hierarchy.

---

## Data provenance without `dispatch()`

`YFinanceProvider.get_ohlcv()` reports every session-cache hit, disk-cache
hit, and live fetch into whatever decision record is currently open — this
happens automatically inside `dispatch()`, and now covers the in-memory
session cache as well (a call that never leaves the warm cache used to
report nothing at all). Calling the provider directly outside of
`dispatch()` is a no-op for provenance tracking (there's no open decision
record to report into), but the OHLCV data itself is unaffected.

---

## CLI (`sqt`)

A thin command-line wrapper (`cli.py`, stdlib `argparse` only — no new
dependency) around the same JSONL decision records, addressed by
`request_id`. Installed as the `sqt` console script
(`pip install -e .` registers it via `[project.scripts]` in
`pyproject.toml`).

```bash
sqt report <request_id>              # pretty-print one record in full
sqt replay <request_id>              # re-run the call, report data/output match
sqt compare <request_id_a> <id_b>    # diff two records' status/output/inputs
```

```bash
$ sqt replay e88b5d2a17e440ab84914461f1399b9b
request_id   : e88b5d2a17e440ab84914461f1399b9b
tool_name    : run_sma_backtest
output_match : True
  data_source: AAPL 2022-01-01 -> 2023-01-01 (1d)  match=True
```

`sqt replay` is a thin CLI wrapper around `verify_replay()` above — same
re-run, data/output match semantics, and notes, just formatted for a
terminal instead of a `ReplayResult` object.

**Exit codes (`sqt replay` only):** `0` — `output_match` is `True` (the
output reproduced exactly); `1` — `output_match` is `False` (a confirmed
mismatch — code or data changed the result); `2` — `output_match` is `None`
(the stored record has no `output_hash` to compare against, so replay
success is indeterminate, not confirmed). Check the exit code rather than
scraping stdout when scripting `sqt replay` in CI — a prior version of this
CLI always exited `0` regardless of match status, so treat any script
written against that behavior as stale. `sqt report` and `sqt compare` exit
`0` on success and `1` on a lookup error (unknown `request_id`), same as
`sqt replay`'s own error path.

`sqt compare` diffs
`tool_name`, `status`, `output_hash`, `duration_ms`, `git_commit_sha`,
`package_version`, `strategy_source_hash`, `random_seed`, and every key in
`input` that differs between the two records — useful for "why did this
number change between these two runs" without hand-parsing two JSONL lines.

Records are looked up across every `*.jsonl` file in `SQT_AUDIT_DIR`
(default resolution — same env var as everywhere else), so `request_id`
alone is enough regardless of which day's file it landed in.
