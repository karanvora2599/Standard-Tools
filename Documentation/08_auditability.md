# Auditability

Every call routed through `dispatch()` can produce an immutable **decision
record**: the tool name and inputs, the market data it pulled (with content
hashes), whether the C++ accelerated path was available, a hash of the
output, and how long it took. `verify_replay()` re-runs a recorded call and
reports whether the data and output still match — enough to tell a stale or
tampered cache apart from a genuine code change.

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
  "error_message": null
}
```

`data_sources` has one entry per OHLCV pull, tagged `disk_cache` or
`live_fetch`, with a content hash of the DataFrame actually used. Failed
calls still produce a record — `status: "error"` with `error_type` /
`error_message` set, and `output_hash: null`.

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

`YFinanceProvider.get_ohlcv()` reports every disk-cache-hit and live-fetch
into whatever decision record is currently open — this happens automatically
inside `dispatch()`. Calling the provider directly outside of `dispatch()` is
a no-op for provenance tracking (there's no open decision record to report
into), but the OHLCV data itself is unaffected.
