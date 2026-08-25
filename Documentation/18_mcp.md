# MCP server

Standard Tools over the Model Context Protocol, so any MCP client — Claude
Desktop, Claude Code, an IDE, another agent framework — can use the library
without going through the `Implementation/` scripts.

The design and the measurements behind it are in
[Development/mcp_plan.md](../Development/mcp_plan.md). This page is how to
run it.

---

## Install and configure

The SDK is not a dependency of the core package:

```bash
pip install 'standard_quant_tools[mcp]'
```

Then point a client at the `sqt-mcp` entry point:

```jsonc
// Claude Desktop: claude_desktop_config.json
{
  "mcpServers": {
    "standard-tools": {
      "command": "sqt-mcp",
      "args": ["--categories", "screener,analysis,quant_research,discovery"],
      "env": {
        "SQT_RUNS_DIR":  "/absolute/path/to/runs",
        "SQT_AUDIT_DIR": "/absolute/path/to/audit",
        "SQT_CACHE_DIR": "/absolute/path/to/cache"
      }
    }
  }
}
```

**Set those three paths.** A client launches the server with a working
directory nobody chose, so an unset `SQT_RUNS_DIR` does not fail at startup
— it fails three turns into a conversation when a tool tries to persist an
artifact, and resource links stop resolving across restarts. The server
warns on stderr when they are missing and refuses to start if one is set but
not writable.

---

## Choosing categories

The 82 tools cost about **146 KB of schema, ~37,000 tokens**, held for the
whole session. That is the constraint the whole design manages, so category
selection is the first decision, not a tuning knob.

```bash
sqt-mcp --print-budget
```

```
category             tools    bytes   ~tokens
modeling                14   40,309    10,077
backtest_execution      10   26,963     6,740
backtest_validation      9   20,542     5,135
analysis                14   15,832     3,958
portfolio_risk           7   15,752     3,938
quant_research           7    8,583     2,145
custom_signal            2    6,630     1,657
discovery                8    5,396     1,349
microstructure           3    4,017     1,004
provenance               6    3,601       900
screener                 2    2,035       508
all                     82  149,660    37,415
```

**Tool count and cost are barely related**, which is the useful thing to
know when picking. `analysis` carries 14 tools for 15.5 KB; `custom_signal`
carries 2 for 6.5 KB. `modeling` and `backtest_execution` together are
nearly half the surface. Choosing by how many tools a category holds gets
the budget almost exactly backwards.

The default — `screener,analysis,quant_research,discovery`, 31 tools,
~8k tokens — covers screening, risk and technical snapshots, the
factor/cointegration/Hurst research path, and the offline discovery tools.

`discovery` is in the default despite being one of the newest categories,
because it is the only one that makes the OTHERS cheaper to use: it is
5.4 KB, and the questions it answers — which parameters a strategy takes,
which stress windows exist, whether this provider has ticks, whether these
arguments are even valid — were previously answered by a failed call and an
error round trip, which costs more than the category does.

Add what a session needs:

```bash
sqt-mcp --categories screener,analysis,backtest_execution
sqt-mcp --categories modeling
sqt-mcp --categories all          # ~26k tokens, and it says so at startup
```

Categories come from `TOOL_CATEGORY`, the same taxonomy behind
[the router and the twelve workers](13_agent_orchestration.md), plus
`modeling` for the modeling runtime.

### Categories, runtimes, and what the server enforces

A category is a slice of the tool list. A **runtime** is an execution
boundary — see [19_runtimes.md](19_runtimes.md). The server uses both, and
the distinction is what makes its scoping real rather than cosmetic:

- `--categories` decides which tools are **advertised**.
- Each tool is dispatched by its **owning runtime's** dispatcher, so a tool
  served from `research` is executed by a table holding only research
  tools.

That matters because a client can send any tool name it likes. Before
runtimes, a name the server never advertised would still have executed
through the union dispatcher. Now the server rejects it by name, and the
underlying runtime would reject it again.

| Runtime | Categories it owns |
|---|---|
| `research` | `screener`, `analysis`, `quant_research` |
| `backtest` | `backtest_execution`, `backtest_validation`, `custom_signal` |
| `portfolio` | `portfolio_risk`, `microstructure` |
| `meta` | `discovery`, `provenance` |
| `modeling` | `modeling` |

Selecting `--categories microstructure` is therefore not the same as
selecting the `portfolio` runtime: it advertises three tools, and those
three still execute inside `portfolio`.

### Two categories worth knowing about

`microstructure` needs a data provider with a **tick feed**. Every tool in
it refuses by name on a bar-only provider rather than approximating from
OHLCV — call `describe_data_capabilities` (in `discovery`) first.

`provenance` reads and verifies the decision log. It is read-only by
design: retention operations that could destroy evidence (`gc`, `seal`,
`hold`) stay CLI-only, because an agent that can delete the record of its
own decisions is not audited by it.

---

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--categories` | `screener,analysis,quant_research` | Which tools to expose. `all` for everything. |
| `--inline-limit` | `4096` | Results larger than this are stored and returned as a summary plus a `sqt://result/...` link. |
| `--output-schemas` | off | Declare `outputSchema` per tool. **Roughly doubles the context cost.** `structuredContent` is returned either way, so this only helps clients that validate against the schema. |
| `--enable-long-running` | off | Expose `scan_pairs` and `run_backtest_optimization`. |
| `--print-budget` | — | Print the table above and exit. |

### Why `--output-schemas` is off

Every one of the 82 tools has a typed Pydantic return, so the server can
declare an output schema for all of them — and does return
`structuredContent` on every call regardless. Declaring the schemas as well
roughly doubles the surface. The plan assumed that was free; measured, it
is not, so it became a flag rather than a default.

### Why long-running tools are hidden

`scan_pairs` is measured at **5.31 minutes** over a 2,000-ticker universe
and `run_backtest_optimization` grows with the grid. Both can outlast a
default client timeout, and a timeout that fires after most of the work is
done is worse than not offering the tool. `run_screener` is *not* hidden —
its runtime is set by the universe you pass rather than being long by
nature — but it carries a runtime note in its description.

---

## Resources

Large results do not go into the conversation. A result over
`--inline-limit` is stored whole and comes back as a summary naming exactly
what it withheld:

```json
{
  "sharpe_ratio": 1.24,
  "max_drawdown": -0.18,
  "_truncated": {
    "reason": "the full result is 51,204 bytes, over the 4,096-byte inline limit",
    "omitted_fields": ["equity_curve (rows=1250, 49,780 bytes)"],
    "result_uri": "sqt://result/mcpres-9f2c...",
    "note": "The fields listed in omitted_fields are NOT in this response..."
  }
}
```

The rule is all-or-nothing per field. Half a trade log looks like a whole
trade log to a model reading it, and there is no honest way to signal "this
continues" inside the value.

| URI | What it serves |
|---|---|
| `sqt://catalog/categories` | The budget table above, as data |
| `sqt://catalog/features` | The modeling feature catalog |
| `sqt://catalog/capabilities` | What this install's modeling runtime supports |
| `sqt://result/{result_id}` | A stored oversized tool result |
| `sqt://artifact/{run_id}/{name}` | A Parquet artifact as JSON records |
| `sqt://model/{model_id}` | A registered model's manifest |
| `sqt://dataset/{dataset_id}` | A built dataset's metadata |
| `sqt://audit/{request_id}` | The decision record for one tool call |

Separately from these, tools exchange bulk values through **handoff
references** (`sqt://signal_panel/...`, `sqt://predictions/...`). Those are
not MCP resources — they are a value one tool hands another, resolvable
from any runtime and never passing through the conversation. See
[19_runtimes.md](19_runtimes.md#2-why-the-interconnect-exists).

Every path resolves through the same sandbox guard the library uses
(`run_dir()` confirms it is inside `SQT_RUNS_DIR` before any read), because
a URI from a client is untrusted input.

---

## Prompts

Five workflow templates, invoked by the user rather than chosen by the
model:

| Prompt | Arguments | Needs categories |
|---|---|---|
| `screen_and_backtest` | universe, criteria, period | `screener`, `backtest_execution` |
| `factor_research_note` | assets, factors, period | `quant_research` |
| `pair_trade_study` | symbol_a, symbol_b, period | `quant_research`, `backtest_execution` |
| `build_and_validate_model` | universe, horizon, period | `modeling` |
| `risk_review` | holdings, period | `portfolio_risk`, `analysis` |

Invoking one whose categories were not loaded prepends a warning rather than
silently handing the model a workflow it has no tools to run.

---

## The audit trail

Both dispatchers already route through `audit._run_and_record`, so **every
tool call made through this server produces a hash-chained, replayable
decision record** — inputs, the market data pulled with content hashes,
which execution path ran, and a hash of the output.

```bash
sqt report  <request_id>     # what the model actually called
sqt replay  <request_id>     # re-run it; does it still reproduce?
sqt verify                   # chain integrity across the whole trail
```

Records are also readable in-session at `sqt://audit/{request_id}`.

This is tamper *detection*, not prevention or certification — see
[10_auditability.md](10_auditability.md) for what it does and does not
establish. Set `SQT_AUDIT_ENABLED=0` to turn record writing off.

---

## Safety

Every one of the 82 tools declares `readOnlyHint: true` and
`destructiveHint: false`, and a test asserts it. This library does not place
orders, hold positions, or mutate anything outside its own artifact store.

The other two hints are derived from the code rather than maintained by
hand: `openWorldHint` is true when a tool's input schema names a symbol,
ticker or universe anywhere (including nested specs — `build_model_dataset`
hides its universe two levels down), and `idempotentHint` is false for the
four tools that persist a new artifact per call.

---

## Architecture notes

**Five runtimes, one server.** Ten categories come from the 68-tool
analysis surface, spread across four runtimes, and one from the separate
14-tool modeling runtime. They stay apart inside — `dispatch_for(entry)`
returns that tool's own RUNTIME's dispatcher, so schemas and executor are
never chosen separately, and a tool served from `research` is executed by a
table holding only research tools — but a user configures one server, not
five.

**Schemas are dereferenced.** Seven tools carry `$ref`/`$defs` upstream, and
they are the seven most complex tools in the library. The server inlines
them and a test asserts nothing reaching a client still contains a `$ref`.
Inlining turned out to *shrink* the payload by 5.4%, not grow it.

**The server holds no logic.** It converts protocol shapes, routes to a
dispatcher, and converts back. `Implementation/` holds four copies of an
agent loop and what kept them consistent is that none contains logic; this
is the fifth surface onto the same registries and gets the same rule.

---

## Testing

```bash
pytest tests/mcp -m "not integration"   # 48 schema and wiring tests, no subprocess
pytest tests/mcp -m integration         # 10 real stdio sessions
```

The integration file spawns the server as a subprocess and drives it with a
real MCP client. That is the only way to catch the failure most likely to
reach a user: a library module writing to stdout, which corrupts JSON-RPC
and surfaces as an unintelligible protocol error rather than a Python one.
It is also what caught a handler constructing an object the SDK rejects,
which every in-process test had missed.
