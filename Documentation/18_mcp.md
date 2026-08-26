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

## Running as a service (HTTP)

stdio ties one server to one client that launches it. That is right for a
person running an editor and wrong for a team: four analysts, a scheduler
and an internal agent cannot share a configured instance, because there is
no instance, only a command line each of them runs separately.

`--transport http` serves the same tools over streamable HTTP, so the server
becomes something you configure once and point clients at.

```bash
pip install 'standard_quant_tools[mcp-http]'

export SQT_MCP_TOKEN=$(openssl rand -base64 32)
export SQT_RUNS_DIR=/var/lib/sqt/runs
export SQT_AUDIT_DIR=/var/lib/sqt/audit
export SQT_CACHE_DIR=/var/lib/sqt/cache

sqt-mcp --transport http --categories all --port 8765
```

Clients connect to `http://host:8765/mcp` and send the token as a bearer
credential:

```jsonc
{
  "mcpServers": {
    "standard-tools": {
      "type": "http",
      "url": "http://sqt.internal:8765/mcp",
      "headers": { "Authorization": "Bearer <SQT_MCP_TOKEN>" }
    }
  }
}
```

`GET /healthz` answers without a token, so a load balancer does not need the
secret to decide whether the process is alive. It returns the version, the
categories loaded and the tool count, and nothing a caller could not learn
by watching the port answer at all.

### What it refuses to do

All four are startup failures, not runtime ones. A server reachable by more
callers than you meant does not show up in a log line; it shows up as
somebody else's tool call in your audit trail.

| Situation | What happens |
|---|---|
| `--transport http` with no `SQT_MCP_TOKEN` | Refuses to start. Pass `--no-auth` to serve without one deliberately. |
| `SQT_MCP_TOKEN` set **and** `--no-auth` passed | Refuses to start. The two ask for opposite things and guessing fails open. |
| A non-loopback `--host` with no `--allow-host` | Refuses to start. The Host check cannot be derived from a wildcard bind. |
| `--no-auth` on a non-loopback host | Starts, with a warning naming exactly what is now reachable. |

The token is read from the environment and never from a flag, because a
command line is visible in the process table to every user on the box.

### Security notes

**A shared token is not an authorization server.** It is one secret for
every caller, so it cannot say who did what, and revoking it revokes
everyone. That is the right primitive when something in front already
terminates identity (an OAuth proxy, mTLS, a service mesh) and the wrong one
when nothing does. The SDK ships `mcp.server.auth` for the latter; wiring it
needs an issuer and a client registry, which is a decision rather than a
default.

**DNS rebinding protection is on.** Host and Origin headers are validated.
On a loopback bind the allowed hosts are derived from the address; on any
other bind you name them with `--allow-host` (`name:*` matches any port).
Browser clients need `--allow-origin` as well; a client that sends no Origin
header, which is most of them, is unaffected.

**Run it behind TLS.** The server speaks plain HTTP. The token and every
result crosses the network in cleartext without a terminating proxy in
front.

**One instance means one store.** `sqt://` URIs resolve against
`SQT_RUNS_DIR` for every connected client, so a result link handed to one
client is readable by all of them. Over stdio each client had its own
process and therefore its own store. The startup report says so.

### Scaling

`--stateless` handles every request without server-side session state, so
any replica can serve any request and no load balancer needs affinity. The
cost is server-initiated messages: no progress notifications, no resumable
streams. `--json-response` returns a single JSON body instead of an SSE
stream, which some proxies and simple clients prefer.

Streams are not resumable in either mode. Resumability means replaying
events a client missed, which needs durable storage; an in-memory event
store would look like the feature while losing everything on the restart
that most often causes the disconnect.

---

## Choosing what to serve

The 170 tools cost about **260 KB of schema, ~66,000 tokens**, held for the
whole session. That is the constraint the whole design manages, so this is
the first decision, not a tuning knob.

That wall has already been hit and passed. Over the wire a tool averages
1,615 bytes and the session ceiling is 180,000, which buys about 111 tools.
There are 170. **The whole surface has not fitted in one session since the
83rd tool**, and no amount of schema-shrinking brings it back — which is why
scoping stopped being an optimization and became the way the server is
meant to be run. Serving `--runtime all` is a diagnostic, not a deployment.

Two flags, and they are nested rather than alternative:

- **`--runtime`** picks the coarse scope. A runtime owns its categories, so
  this is the boundary a client is served at.
- **`--categories`** narrows *within* the chosen runtime, as it always did.

```bash
sqt-mcp --print-budget
```

```
runtime              tools    bytes   ~tokens
backtest                32   73,057    18,264
modeling                16   46,651    11,662
research                40   44,616    11,154
portfolio               17   29,900     7,475
derivatives             12   17,878     4,469
microstructure          12   15,523     3,880
meta                    19   14,138     3,534
feature_lab              9   11,745     2,936
all                    157  253,508    63,377

  a client is served ONE runtime: backtest is the most
  expensive at 73,057 bytes (29% of the total).

category             tools    bytes   ~tokens
modeling                16   46,651    11,662
backtest_validation     20   35,582     8,895
backtest_execution      10   30,593     7,648
portfolio_risk          17   29,900     7,475
quant_research          26   28,497     7,124
derivatives             12   17,878     4,469
microstructure          12   15,523     3,880
analysis                12   14,084     3,521
feature_lab              9   11,745     2,936
discovery               13   10,537     2,634
custom_signal            2    6,882     1,720
provenance               6    3,601       900
screener                 2    2,035       508
```

**The total is a number nobody pays.** The row that matters is the runtime
a client is actually served, and the most expensive of those is 71 KB at
full detail — 29% of the whole surface. `tests/mcp/test_runtime_scope.py`
pins a 72 KB per-runtime ceiling for exactly that reason, and it is
deliberately tight: `backtest` sits at 73,057 bytes at full detail, 671
bytes under the 73,728 ceiling. That margin is under half a tool, which is
why `--tool-detail auto` is the default rather than an option — the
ceiling test measures what the server actually serves, and under `auto`
`backtest` serves at 34 KB.

**Tool count and cost are barely related**, which is the useful thing to
know when picking. `analysis` carries 12 tools for 13.8 KB; `custom_signal`
carries 2 for 6.7 KB -- a sixth of the tools for half the bytes. `modeling`
and `backtest_validation` are two categories out of thirteen and a third of
the surface between them. Choosing by how many tools a category holds gets
the budget almost exactly backwards.

The default — `screener,analysis,quant_research,discovery`, 53 tools,
~13k tokens — covers screening, risk and technical snapshots, the
factor/cointegration/Hurst research path, the statistical diagnostics
(stationarity, structural breaks, bootstrap intervals, seasonality), and
the offline discovery tools.

`discovery` is in the default despite being one of the newest categories,
because it is the only one that makes the OTHERS cheaper to use: 13 tools
for 10.3 KB, and the questions it answers — which parameters a strategy takes,
which stress windows exist, whether this provider has ticks, whether these
arguments are even valid — were previously answered by a failed call and an
error round trip, which costs more than the category does.

Serve a runtime, or narrow inside one:

```bash
sqt-mcp --runtime research                    # 40 tools, 33 KB served
sqt-mcp --runtime backtest                    # 32 tools, 34 KB served
sqt-mcp --runtime derivatives                 # 12 tools, 17 KB served
sqt-mcp --runtime microstructure              # 12 tools, 15 KB served
sqt-mcp --runtime research+meta               # research plus discovery/provenance
sqt-mcp --runtime research --categories screener
sqt-mcp --runtime all                         # ~63k tokens, and it says so
```

`+` joins runtimes because that is how `combine()` names a joined runtime in
the library itself — the flag and the code spell the same thing the same way.

**`--runtime research` serves all of research**, not research narrowed by
the `--categories` default. That matters more than it sounds: the default
categories belong to research and meta, so inheriting them under
`--runtime backtest` would have served zero tools, and an empty server reads
as a broken install rather than as two flags disagreeing.

### Paying for schemas only when you need them

`--runtime` decides *which* tools; `--tool-detail` decides how much of each
one is sent at connect.

```bash
sqt-mcp --runtime backtest --tool-detail full     # 71 KB, every schema
sqt-mcp --runtime backtest                       # 34 KB, 12 tools thinned (the default)
sqt-mcp --runtime backtest --tool-detail thin    # 14 KB, all thinned
```

A **thinned** tool is still listed and still callable. What it loses is its
argument schema: the listing carries the name, one line of purpose, and an
instruction to call `describe_tool` for the rest. Measured, that is 483
bytes against 1,615 — **70% smaller**.

`auto` is the mode worth using. It thins the **most expensive** tools and
stops as soon as the runtime fits `--detail-budget` (32 KB by default):

| runtime | full | `auto` | thinned by `auto` | thin |
|---|---:|---:|---:|---:|
| `backtest` | 71 KB | 34 KB | 12 | 14 KB |
| `modeling` | 46 KB | 25 KB | 2 | 6 KB |
| `research` | 44 KB | 33 KB | 8 | 18 KB |
| `portfolio` | 29 KB | 29 KB | 0 | 9 KB |
| `derivatives` | 17 KB | 17 KB | 0 | 7 KB |
| `microstructure` | 15 KB | 15 KB | 0 | 8 KB |
| `meta` | 14 KB | 14 KB | 0 | 8 KB |
| `feature_lab` | 11 KB | 11 KB | 0 | 5 KB |

**`auto` is the default now, not an option.** At full detail `backtest`
comes within 671 bytes of the 73,728 per-runtime ceiling, and the ceiling
is not the thing to move — it had been argued up once already. `auto`
thins only what exceeds `--detail-budget`, so the five runtimes already
under it are returned byte-for-byte unchanged and only `backtest`,
`modeling` and `research` differ. `--tool-detail full` still exists and
still does exactly what it did; it simply stopped being implicit.

Five runtimes already fit and pay nothing. That ordering is deliberate: a
runtime's cost is concentrated in a few large schemas — `modeling`'s top
three are 65% of it — so thinning three tools buys what thinning fifteen
cheap ones would not, and **every tool left described is one an agent calls
without a round trip.** Minimising bytes and minimising round trips turn
out to be the same instruction.

**Thinning changes the advertisement and nothing else.** The arguments are
unchanged, validated by the same model, and `extra="forbid"` still rejects a
guessed name rather than defaulting it. What thinning costs is one
`describe_tool` call before the first use of a thinned tool — and what it
would cost to skip that call is a rejected call, which is why the
instruction appears in both the description and the schema. A client may
show a model only one of them, and a model shown an empty `{}` schema
concludes the tool takes no arguments.

Because a thin entry tells an agent to call `describe_tool`, and
`describe_tool` belongs to `meta`, **the server adds it whenever anything is
thinned** and never thins it. Under `--runtime backtest --tool-detail thin`
the instruction would otherwise be unfollowable and every thinned tool
uncallable. It is the one place scope widens automatically, and it is
reported at startup.

`--tool-detail full` is the default, so nothing changes for an existing
invocation.

Naming a category the runtime does not own is refused at startup, by name:

```
$ sqt-mcp --runtime research --categories modeling
sqt-mcp: --categories ['modeling'] not served by --runtime research
('modeling' belongs to 'modeling'). Either drop it, or widen --runtime
deliberately -- serving the intersection would give you a surface neither
flag describes.
```

Categories come from `TOOL_CATEGORY`, the same taxonomy behind
[the router and the fourteen workers](13_agent_orchestration.md), plus
`modeling` for the modeling runtime.

### Categories, runtimes, and what the server enforces

A category is a slice of the tool list. A **runtime** is an execution
boundary — see [19_runtimes.md](19_runtimes.md). The server uses both, and
the distinction is what makes its scoping real rather than cosmetic:

- `--runtime` decides which tools can be **served or executed at all**.
- `--categories` narrows which of those are **advertised**.
- Each tool is dispatched by its **owning runtime's** dispatcher, so a tool
  served from `research` is executed by a table holding only research
  tools.

That matters because a client can send any tool name it likes. Before
runtimes, a name the server never advertised would still have executed
through the union dispatcher. Now there are three independent refusals: the
server does not list it, the server rejects the call by name, and the
underlying runtime would reject it again.

The refusal says which of three situations you are actually in, because
only one of them is fixable by changing a flag:

```
# exists, but in another runtime -> a scope problem, and the fix is named
'run_sma_backtest' exists, but belongs to the 'backtest' runtime and this
server is scoped to research. Restart with `--runtime backtest` to serve
it -- widening scope is a decision, not a fallback.

# exists nowhere -> a hallucination, and no flag will help
unknown tool 'run_sma_backtestt'. No tool by that name exists in this
library. This server serves 40 tools from the research runtime. Read
sqt://catalog/categories for what exists, or call tools/list for what this
server serves -- do not guess another name.

# in this runtime, filtered out by category -> --categories is the fix
'analyze_stock_risk' exists in the 'research' runtime but is not in the
selected categories (screener). It belongs to 'analysis'.
```

The middle case is the one worth the extra code. This error used to name the
loaded categories for *every* unknown tool, which told an agent that had
invented a name to go and widen a scope that could never contain it.

| Runtime | Categories it owns |
|---|---|
| `research` | `screener`, `analysis`, `quant_research` |
| `backtest` | `backtest_execution`, `backtest_validation`, `custom_signal` |
| `portfolio` | `portfolio_risk` |
| `microstructure` | `microstructure` |
| `derivatives` | `derivatives` |
| `meta` | `discovery`, `provenance` |
| `modeling` | `modeling` |
| `feature_lab` | `feature_lab` |

Selecting `--categories microstructure` is therefore not the same as
`--runtime portfolio`: it advertises three tools rather than ten, and those
three still execute inside `portfolio`.

Naming a category outside the chosen runtime is refused at startup rather
than quietly intersected, because serving the intersection would hand back a
surface neither flag describes.

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
| `--runtime` | every runtime | The coarse scope: which runtimes can be served **or executed**. One name, or several joined with `+`, or `all`. Given alone, serves all of that runtime's categories. |
| `--categories` | `screener,analysis,quant_research,discovery` | Narrows which tools are advertised *within* the chosen runtimes. `all` for everything. A category outside the chosen runtime is refused at startup. |
| `--tool-detail` | `full` | How much of each tool to advertise. `auto` thins the most expensive schemas until the runtime fits `--detail-budget`; `thin` thins everything. A thinned tool is still callable — its arguments come from `describe_tool`. |
| `--detail-budget` | `32768` | Byte target for `--tool-detail auto`. |
| `--inline-limit` | `4096` | Results larger than this are stored and returned as a summary plus a `sqt://result/...` link. |
| `--output-schemas` | off | Declare `outputSchema` per tool. **Roughly doubles the context cost.** `structuredContent` is returned either way, so this only helps clients that validate against the schema. |
| `--enable-long-running` | off | Expose `scan_pairs` and `run_backtest_optimization`. |
| `--print-budget` | — | Print the per-runtime and per-category tables above, and exit. |
| `--transport` | `stdio` | `stdio` or `http`. See [Running as a service](#running-as-a-service-http). |
| `--host` | `127.0.0.1` | Bind address. Anything else needs a token and `--allow-host`. |
| `--port` | `8765` | Bind port. |
| `--path` | `/mcp` | Path the endpoint is mounted at. |
| `--stateless` | off | No server-side session state, so any replica serves any request. Costs progress notifications. |
| `--json-response` | off | Single JSON body instead of an SSE stream. |
| `--allow-host` | — | Host header to accept, repeatable. Required on a non-loopback bind. |
| `--allow-origin` | — | Origin header to accept, repeatable. Only browser clients need it. |
| `--no-auth` | off | Serve without a bearer token. Only when something in front authenticates. |

### Why `--output-schemas` is off

Every one of the 170 tools has a typed Pydantic return, so the server can
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

Every one of the 170 tools declares `readOnlyHint: true` and
`destructiveHint: false`, and a test asserts it. This library does not place
orders, hold positions, or mutate anything outside its own artifact store.

The other two hints are derived from the code rather than maintained by
hand: `openWorldHint` is true when a tool's input schema names a symbol,
ticker or universe anywhere (including nested specs — `build_model_dataset`
hides its universe two levels down), and `idempotentHint` is false for the
six tools that persist a new artifact per call.

Read-only is a statement about what the tools do, not about who may call
them. Over stdio the only caller is the process that launched the server;
over HTTP it is anything that can route to the port, and a caller who can
run a 2,000-ticker `scan_pairs` can spend your CPU and your data provider's
rate limit whether or not anything is mutated. See
[the security notes](#security-notes).

---

## Architecture notes

**Eight runtimes, one server.** Eleven of the thirteen categories come from
the 132-tool analysis surface, spread across six runtimes; the other two
are the separate 16-tool `modeling` and 9-tool `feature_lab` runtimes. They
stay apart inside — `dispatch_for(entry)` returns that tool's own RUNTIME's
dispatcher, so schemas and executor are never chosen separately, and a tool
served from `research` is executed by a table holding only research tools —
but a user configures one server, not eight.

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
