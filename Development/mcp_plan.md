# MCP server plan

Exposing Standard Tools over the Model Context Protocol, so any MCP client —
Claude Desktop, Claude Code, an IDE, another agent framework — can use the
library without the `Implementation/` scripts.

This document states the constraints before the design, the same way
`modeling_native_plan.md` stated its ceiling before its method. Three of them
are measured, and they are what make this more than "wrap 54 functions".

> **Status: built, merged, and since outgrown.** Four assumptions in this
> plan did not survive the build; §16 records them, because a plan that is
> quietly edited to match what happened stops being evidence of anything.
> The sections below are left as written.
>
> **Every measurement below is from the 54-tool era and is no longer
> current.** The library now serves **157 tools across eight runtimes** for
> 248 KB, and the exposure policy this document argued for became
> `--runtime` scoping plus `--tool-detail auto`. For the numbers as they
> stand, run `sqt-mcp --print-budget`, or read
> [Documentation/18_mcp.md](../Documentation/18_mcp.md#choosing-what-to-serve).

---

## 1. What this is, and what it is not

**Is:** a server that exposes the library's two existing agent registries —
46 analysis/backtest tools and 8 modeling tools — over MCP, plus the
resource and prompt surfaces MCP provides that the current function-calling
interface has no way to express.

**Is not:** a new tool surface. Every tool it serves already exists and is
already tested. If the MCP layer needs a tool that `dispatch()` does not
have, that tool belongs in the library first. The MCP server is a
**transport**, and the moment it starts holding logic of its own it becomes
a third registry nobody is testing.

**Explicitly out of scope:** order placement, broker connectivity, anything
that moves money. Every tool here is read-only analysis, and that fact is
worth encoding in the protocol rather than only in prose (§5).

---

## 2. The three constraints that decide the design

### 2.1 The tool schemas are 118 KB

Measured, not estimated:

| Registry | Tools | Schema bytes |
|---|---:|---:|
| `standard_quant_tools.agent` | 46 | 90,239 |
| `standard_quant_tools.modeling.agent` | 8 | 30,967 |
| **Total** | **54** | **121,206**  (118.4 KB) |

MCP clients fetch the full tool list at connect and hold it for the session.
At roughly 4 bytes per token that is **~30,000 tokens of every conversation,
before the user asks anything**. On a 200k context that is 15% gone to a
menu.

The modeling tools are the surprise: 8 tools carrying a third of the total,
because `run_model_experiment` (10.3 KB) and `evaluate_model_portfolio`
(9.4 KB) embed deeply nested spec models.

This is the single most important number in this document, and it is the
reason §4 is about *exposure policy* rather than about tools.

It is also not a new problem. This repo already answered it twice —
`TOOL_CATEGORY` + `agent/router.py`, and the 9-worker split in
`Multi_Agent_Implementation/`. The MCP server should reuse that taxonomy,
not invent a third one.

### 2.2 The outputs are unbounded

**83 list- or dict-valued fields across 28 result models.** `BacktestResult`
alone carries `equity_curve` and `trade_log`; `PortfolioResult` carries a
full `correlation_matrix`.

A five-year daily backtest returns ~1,250 equity points plus a trade log. In
the current scripts that is fine — `_agent_utils.py` truncates at 2,000
characters for display and the model sees the JSON once. Over MCP the result
goes into the client's context and *stays* there for the rest of the
session, and a grid search or a panel backtest is far larger.

MCP has the right answer built in: **resource links**. Large payloads get
persisted (the artifact store already exists) and returned as a URI the
client can read on demand. See §6.

### 2.3 Some tools take minutes

`scan_cointegrated_pairs` is measured at **5.31 minutes** for a 2,000-ticker
universe. `backtest_grid` scales with the grid. `run_screener` fans out
across processes. Default MCP client timeouts are far below that.

This needs progress notifications, cancellation, and an honest per-tool
timeout policy (§8) — not a hope that nobody runs the big one.

---

## 3. Architecture

**One server, both registries, category-gated.**

```
                    MCP client (Claude Desktop / Code / IDE)
                                  │  stdio (JSON-RPC)
                    ┌─────────────┴──────────────┐
                    │  standard_quant_tools.mcp  │
                    │                            │
                    │   tools/     resources/    │
                    │   prompts/   progress      │
                    └─────────────┬──────────────┘
                        ┌─────────┴─────────┐
                 agent.dispatch      modeling_dispatch
                  (46 tools)            (8 tools)
                        └─────────┬─────────┘
                          audit._run_and_record
                        (hash-chained decision records)
                                  │
                     data / indicators / analysis /
                     metrics / backtest / portfolio
```

**Why one server rather than two.** The registries stay separate *inside* —
separate dispatchers, separate namespaces, exactly as
`Multi_Agent_Implementation/` keeps them — but a user configuring a client
should add one entry, not two. Two servers would also make cross-registry
workflows ("build a model, then backtest its signal") span two connections
for no benefit; the model→backtest bridge already exists in the library.

**Why not merge the dispatchers.** The two registries share no tool name
(verified: 54 tools, 54 unique names), so a single flat lookup would
*work* — and would be the same mistake the library refused to make. The
server routes by registry the way `run_agent(registry=...)` does, and gets
the same property: a modeling tool name can never reach `agent.dispatch()`.

---

## 4. Tool exposure policy

This is where §2.1 gets paid for.

### 4.1 Launch-time category selection (phase 1)

The server takes a category list, defaulting to a curated core:

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "standard-tools": {
      "command": "sqt-mcp",
      "args": ["--categories", "screener,analysis,quant_research"],
      "env": { "SQT_RUNS_DIR": "...", "SQT_AUDIT_DIR": "..." }
    }
  }
}
```

Categories come from `TOOL_CATEGORY` — the same seven the router and the
workers use — plus `modeling` for the second registry:

Measured per category, sorted by what they actually cost:

| Category | Tools | Schema KB | ~tokens |
|---|---:|---:|---:|
| `modeling` | 8 | 30.2 | 7,700 |
| `backtest_execution` | 9 | 26.6 | 6,800 |
| `backtest_validation` | 7 | 18.7 | 4,800 |
| `analysis` | 13 | 13.7 | 3,500 |
| `portfolio_risk` | 6 | 11.0 | 2,800 |
| `custom_signal` | 2 | 8.4 | 2,200 |
| `quant_research` | 7 | 7.5 | 1,900 |
| `screener` | 2 | 2.2 | 560 |
| **all** | **54** | **118.4** | **30,300** |

Tool count and cost are barely related, and that is the useful finding.
`analysis` carries 13 tools for 13.7 KB; `custom_signal` carries **2 tools
for 8.4 KB**. `backtest_execution` is a quarter of the total on its own,
dominated by `run_portfolio_simulation` (8.1 KB) and
`run_signal_panel_backtest` (4.5 KB). Picking categories by how many tools
they hold would get the budget almost exactly backwards.

**Proposed default: `screener,analysis,quant_research` — 22 tools, 23.4 KB,
~6k tokens.** That covers screening, risk and technical snapshots, and the
factor/cointegration/Hurst research path: the questions people actually open
this library to ask. It deliberately omits the two heaviest categories, both
of which are better switched on for a session that needs them than paid for
in every session that does not.

`--categories all` remains available and documented, with its cost stated at
startup so nobody pays 30k tokens without being told.

### 4.2 Dynamic toolsets (phase 3, only if wanted)

MCP supports `notifications/tools/list_changed`. A `enable_tool_category`
meta-tool could reveal categories mid-session. **Deferred deliberately**:
client support for dynamic tool lists is uneven, and a launch flag solves
90% of it with none of the risk. Revisit once the phase-1 server has real
usage telling us which categories people actually combine.

### 4.3 Names and annotations

Verified: all 54 names match `^[a-zA-Z0-9_-]{1,64}$`, longest is
`run_regime_adaptive_walkforward_backtest` (40 chars). **No renaming
needed**, and no prefixing — prefixed names (`sqt_run_sma_backtest`) would
diverge from the audit trail, the docs, and the worker prompts for no gain.

Every tool gets MCP annotations, derived rather than hand-maintained:

| Annotation | Value | Derivation |
|---|---|---|
| `readOnlyHint` | `true` for all 54 | Nothing here mutates external state |
| `destructiveHint` | `false` | Same |
| `idempotentHint` | `true` for pure compute; `false` where a call persists an artifact | Whether the result model carries a `*_uri` |
| `openWorldHint` | `true` where the tool fetches market data | Presence of a provider call |

`readOnlyHint: true` across the board is the encoded version of "this
library does not place orders", and it lets a client offer these tools
without the confirmation friction a write-capable server needs.

### 4.4 Structured output

**All 54 tools have typed Pydantic return annotations** (verified: 46/46 and
8/8). That means `outputSchema` and `structuredContent` — the 2025-06-18
spec addition — can be generated for **every tool automatically**, from
annotations that already exist and are already tested. Most MCP servers
return untyped text. This one should not, and the work to do it is a
`typing.get_type_hints(fn)["return"].model_json_schema()` call.

### 4.5 The `$ref` problem

**7 of the 54 tools have `$ref`/`$defs` in their input schemas** —
`run_portfolio_optimization`, `run_custom_signal_backtest`,
`run_signal_panel_backtest`, `run_portfolio_simulation`,
`build_model_dataset`, and two more. These are the deeply nested spec
models.

The `$defs` travel with the schema, so they are resolvable in principle. In
practice not every client and not every model handles `$ref` well, and these
are exactly the seven most complex tools — the ones where a
mis-parsed schema costs the most.

**Action:** add a dereferencing step that inlines `$defs` into a flat schema,
and a test that asserts no exposed schema contains `$ref`. Measure the size
cost — inlining duplicates repeated definitions, and if it inflates the
118 KB materially that trade-off needs stating rather than absorbing.

---

## 5. What the server does *not* add

A rule worth writing down before phase 1: the MCP layer performs no
computation. It converts protocol shapes, routes to a dispatcher, and
converts back. Any temptation to "just aggregate these two calls" or "just
reshape this output" is a change to the library, made in the library, with a
library test.

Reason: the `Implementation/` folders have four copies of the agent loop and
the discipline that kept them consistent was that none of them contains
logic. The MCP server is the fifth surface. It gets the same rule.

---

## 6. Resources — the part that is not just a tool list

This is what makes it an MCP *server* rather than a function-calling shim,
and it is the answer to §2.2.

The library already has four resource-shaped things, all content-addressed
and all currently reachable only by knowing an id:

| Resource | URI template | Backed by |
|---|---|---|
| Audit decision record | `sqt://audit/record/{request_id}` | `audit/` JSONL + chain index |
| Run artifact | `sqt://artifact/{path}` | `SQT_RUNS_DIR` Parquet store |
| Registered model | `sqt://model/{model_id}` | `modeling/registry` manifests |
| Modeling dataset | `sqt://dataset/{dataset_id}` | dataset registry |

Plus static ones worth serving directly:

| Resource | URI |
|---|---|
| Feature catalog | `sqt://catalog/features` |
| Modeling capabilities | `sqt://catalog/capabilities` |
| Tool categories and their cost | `sqt://catalog/categories` |

**The output-size fix.** Tools that produce large payloads persist them and
return a resource link plus a summary, instead of inlining thousands of
rows:

```
run_sma_backtest  ->  { sharpe: 1.24, max_drawdown: -0.18, n_trades: 47,
                        equity_curve: "sqt://artifact/runs/ab3f/equity.parquet",
                        trade_log:    "sqt://artifact/runs/ab3f/trades.parquet" }
```

The client reads the artifact only if it needs the detail. This requires a
threshold policy — inline small results, link large ones — and the threshold
must be a stated number, not a feeling. Proposal: inline under 4 KB
serialized, link above, with the boundary configurable and the chosen value
reported in the result so it is never silently truncating.

**Path safety is already handled.** `_resolved_within_runs_dir()` rejects any
path escaping `SQT_RUNS_DIR`. The resource handler must go through it rather
than around it — a URI from a client is untrusted input, and this is the one
place in the server where a traversal bug would be reachable from outside.

---

## 7. Prompts

MCP prompts are user-invoked workflow templates. This repo already has nine
of them, written and refined: the `WORKER_AGENTS` system prompts in
`Multi_Agent_Implementation/worker_agents.py`, plus the eight single-agent
scripts' prompts in `Implementation/*/`.

Phase 2 exposes a starting set, parameterized:

| Prompt | Arguments | From |
|---|---|---|
| `screen_and_backtest` | universe, criteria, period | Screener + Backtest workers |
| `factor_research_note` | assets, factors, period | `Agent_Factor_Researcher.py` |
| `pair_trade_study` | ticker_a, ticker_b, period | `Agent_Pair_Trader.py` |
| `build_and_validate_model` | universe, horizon, period | `Agent_Model_Builder.py` |
| `risk_review` | portfolio weights | `Agent_Risk_Monitor.py` |

These are the same text. If a prompt's wording is worth improving it should
be improved in one place and referenced — a sixth copy of the model-builder
prompt is exactly the duplication §5 exists to prevent.

---

## 8. Long-running tools

For the tools in §2.3:

- **Progress notifications.** When the client sends a `progressToken`, emit
  `notifications/progress`. `scan_cointegrated_pairs`, `backtest_grid` and
  `run_screener` all iterate over a countable work list, so real fractions
  are available, not spinners.
- **Cancellation.** Honor `notifications/cancelled`. The C++ kernels and the
  process pools both need a checked cancellation path; where one cannot be
  interrupted cleanly, say so rather than pretending.
- **Timeout policy, per tool, stated.** A default that kills
  `scan_cointegrated_pairs` at 60s is worse than no tool, because it fails
  after doing most of the work. Either the tool documents its expected
  runtime in its description so the model can warn the user, or the big
  variants are gated behind an explicit `--enable-long-running` flag.
- **The process-pool question.** `backtest_grid` and `run_screener` use
  `ProcessPoolExecutor`. Under a stdio MCP server on Windows (spawn start
  method) the workers re-import the module. `engine.py` already documents
  picklability constraints for exactly this reason, so the machinery is
  aware of it — but this must be **tested under the server**, not assumed to
  carry over. It is the single most likely source of a
  works-in-tests-hangs-in-production failure.

---

## 9. Audit integration — the differentiator

`dispatch()` already routes every call through `audit._run_and_record`,
producing hash-chained, replayable decision records. Nothing needs building;
it needs *not breaking*, plus surfacing:

- The server sets a request-id context per MCP call, so a record ties back
  to the client conversation that caused it.
- `sqt://audit/record/{request_id}` makes records readable in-session.
- The `sqt` CLI (`replay`, `verify`, `compare`) keeps working on records
  produced through MCP, because they are the same records.

The result is an MCP server where every tool call a model made is
independently replayable and tamper-evident. That is unusual, it is already
built, and it should be stated plainly in the README — including its
existing limit, which `10_auditability.md` is careful about: this is tamper
*detection*, not prevention or certification.

**Failure mode to decide:** `SQT_AUDIT_FAIL_CLOSED`. If the audit write
fails, does the tool call fail? For an MCP server used interactively, a hard
failure on an audit-disk problem is probably wrong; for a compliance
deployment it is the point. Default open, document the flag, do not choose
silently.

---

## 10. Configuration and the sandbox

MCP servers launch with an unpredictable working directory and a minimal
environment. Everything the library reads from the environment must be
explicit:

| Variable | Consequence if unset |
|---|---|
| `SQT_RUNS_DIR` | Artifacts land somewhere unintended; resource URIs break across restarts |
| `SQT_AUDIT_DIR` | Same for the audit trail |
| `SQT_CACHE_DIR` | Cold Parquet cache every session |
| `SQT_POLYGON_API_KEY` | Polygon provider unavailable |
| `SQT_BLOOMBERG_HOST/PORT` | Bloomberg provider unavailable |

The server should **resolve these at startup, log them once to stderr, and
fail fast with a clear message** if a required directory is not writable —
rather than failing on the first tool call, three turns into a conversation.

**stdout must stay clean.** JSON-RPC over stdio shares the channel. Checked:
the only `print()` in `src/` is inside a docstring example, and
`audit/context.py`'s `StreamHandler()` defaults to stderr — so the library
is clean today. That is a property to **pin with a test**, not to note and
forget: one stray `print()` in any library module corrupts every MCP session
in a way that looks like a protocol bug.

---

## 11. Testing

The repo's convention is drift-proofing tests that fail when something goes
stale. The MCP layer gets the same:

1. **Coverage** — every name in `_TOOL_DISPATCH` and
   `MODELING_TOOL_DISPATCH` is exposed by exactly one MCP tool, both
   directions. Mirrors `test_multi_agent_tool_coverage.py`.
2. **Schema legality** — every exposed name matches MCP's pattern; every
   schema is valid JSON Schema; **no schema contains `$ref` after
   dereferencing** (§4.5).
3. **Serializability** — every result model round-trips through
   `json.dumps(..., allow_nan=False)`. The `_sanitize_for_json` path exists
   because NaN is a real hazard here, and the manifest-null bug found last
   cycle is precisely what this class of test catches.
4. **Budget** — the measured schema size per category, asserted against a
   ceiling. If someone adds a tool that doubles a category's cost, the test
   says so with a number. This is the §2.1 constraint made permanent.
5. **stdout hygiene** — import every library module and assert nothing was
   written to stdout (§10).
6. **Annotations** — every tool has all four hints set; no tool is missing
   `readOnlyHint`.
7. **Process pools under the server** — at least one grid search and one
   screener run executed through the server process on Windows and Linux
   (§8). This is an integration test and it is the one that matters most.

Where possible these run without network, like the existing agent tests.

---

## 12. Phases

**Phase 1 — a working stdio server.** One registry-routing server, launch-time
category selection, all 54 tools reachable, `outputSchema` for every tool,
dereferenced schemas, annotations, env resolution with fail-fast, tests 1–6.
Packaged as `sqt-mcp` with an optional `[mcp]` dependency group so the core
install is unchanged.
*Done when:* Claude Desktop and Claude Code can both connect, list a chosen
category set, and run a backtest end to end, with the audit record readable
afterward by `sqt report`.

**Phase 2 — resources and prompts.** The four dynamic and three static
resources, the large-output link threshold, five prompts drawn from the
existing worker prompts.
*Done when:* a five-year backtest returns under 4 KB inline with working
artifact links, and the model-builder prompt drives the full pipeline.

**Phase 3 — the long tail.** Progress notifications, cancellation, the
process-pool integration tests, the long-running gate. Then, if usage
justifies it, dynamic toolsets and a streamable-HTTP transport with auth for
shared deployments.

Phases 1 and 2 are the product. Phase 3 is what makes it safe to leave
running.

---

## 13. Open questions

These need answers before phase 1 ships, and I do not think any of them has
an obvious default:

1. **What is the default category set?** "Everything" costs ~30k tokens.
   `screener,analysis,quant_research` is my proposal at 23.4 KB / ~6k
   tokens, but it excludes both backtesting and modeling — and modeling is
   the newest and most differentiated part of the library. The honest
   tension: the cheapest useful default and the most interesting default are
   not the same set.
2. **Inline-vs-link threshold.** 4 KB is a proposal, not a measurement.
   Worth measuring against real backtest outputs first.
3. **Does the modeling registry belong in the same server at all?** It is
   stateful across calls (`dataset_id` → `model_id`), which is a different
   interaction shape from the stateless analysis tools. A separate
   `sqt-modeling-mcp` would be defensible. I lean against — one config entry,
   and the bridge between them already exists — but it is a real choice.
4. **`SQT_AUDIT_FAIL_CLOSED` default** (§9).
5. **Which MCP SDK.** The official `mcp` Python SDK is the obvious answer;
   `FastMCP` versus the low-level `Server` API is a real trade-off, since
   the low-level API gives the control this server needs for dynamic tool
   lists and dereferenced schemas.

---

## 14. Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Context cost makes it unpleasant to use | 30k tokens before any question | §4.1 category gating; §11.4 budget test |
| Process pools deadlock under stdio | Silent hang, no error | §8, §11.7 — test on both platforms |
| A stray `print()` corrupts the protocol | Looks like a client bug | §11.5 |
| `$ref` schemas mis-parsed by a client | The 7 most complex tools fail first | §4.5 dereference + test |
| Long tools hit client timeouts | Fails after doing the work | §8 gate + documented runtimes |
| The server accretes logic | A fifth surface nobody tests | §5, stated as a rule |
| Resource URIs escape the sandbox | Traversal from untrusted client input | Route through `_resolved_within_runs_dir` |

---

## 15. What I would not build

**A write surface.** No order placement, no portfolio mutation, no "execute
this trade". `readOnlyHint: true` on all 54 tools is a property worth
keeping, and the moment one tool breaks it every client has to treat the
whole server as dangerous.

**A second tool taxonomy.** `TOOL_CATEGORY` is already the single source of
truth for the router and the workers. If the MCP grouping wants to differ,
that is an argument for changing `TOOL_CATEGORY`, not for adding a parallel
map that can drift.

**Auto-generated per-tool prompts.** Fifty-four prompts nobody wrote is
worse than five that someone did.

---

## 16. What the build found

Written after phases 1 and 2 shipped. Four things above were wrong, and the
useful part is which kind of wrong each was.

### 16.1 `outputSchema` is not free — it is a 77% surcharge

§4.4 said declaring output schemas for all 54 tools was "a
`typing.get_type_hints(fn)["return"].model_json_schema()` call" and treated
the cost as nil. The call is indeed one line. The schemas are **74 KB**, on
top of a 102 KB input surface — a 77% increase to the thing this entire
design exists to minimise.

The resolution was not in the plan's option space: MCP lets a server return
`structuredContent` **without** declaring `outputSchema`. The declaration
only helps a client that validates against it. So structured results are
sent on every call at zero cost, and the declaration became
`--output-schemas`, default off.

*Kind of wrong:* a cost assumed to be zero because the code was short.

### 16.2 Dereferencing `$ref` made the payload smaller, not larger

§4.5 predicted inlining would duplicate shared definitions and warned the
size cost "needs stating rather than absorbing". Measured across all seven
affected tools: **104,413 → 98,763 bytes, −5.4%**. The `$defs` blocks held
definitions referenced once or not at all, so removing the indirection
removed more than it duplicated.

*Kind of wrong:* a reasonable worry that measurement settled in the opposite
direction.

### 16.3 The default category set had to change

§4.1 proposed `screener,analysis,quant_research` at "23.4 KB". With the
final accounting — compact separators, descriptions counted, output schemas
excluded — it is **20.5 KB / ~5k tokens**, and the full surface is 104,645
bytes rather than 118 KB. Same shape, different arithmetic: the plan's table
mixed pretty-printed and compact measurements. `sqt-mcp --print-budget`
prints the real one, and the budget test pins a ceiling against it.

Also changed: §2.3 named `run_screener` as long-running and the build hid it
by default, which left the `screener` category holding a single tool. Its
runtime is set by the universe the caller passes rather than being long by
nature, so it is now exposed with a runtime note, and only `scan_pairs` and
`run_backtest_optimization` are hidden.

*Kind of wrong:* an accounting inconsistency, and a policy that looked fine
until its consequence was visible.

### 16.4 MCP prompts are not the worker system prompts

§7 proposed serving the nine `WORKER_AGENTS` system prompts verbatim, on the
grounds that rewriting them would be a sixth copy. That conflated two
different artifacts: an MCP prompt is **user-invoked**, takes arguments, and
produces a user message; a worker system prompt **scopes an agent** that
already holds a fixed tool list. Reusing one as the other would have shipped
an agent's private instructions as a user-facing template.

The five prompts are written for this surface and borrow the judgement, not
the text. They also gained something the plan did not anticipate: a prompt
whose categories were not loaded prepends a warning, because a workflow
whose tools are absent is worse than no workflow — the model improvises the
steps it cannot run.

*Kind of wrong:* two things sharing a word treated as one thing.

### 16.5 What the plan got right

The three constraints in §2 were the right three, and the design followed
from them rather than from taste. §11's insistence on an integration test
that spawns a real process paid immediately: it caught a resource handler
constructing an object the SDK rejects, which all 46 in-process tests had
passed over. §5's "the server holds no logic" survived contact — nothing in
`standard_quant_tools/mcp/` computes anything.

Phase 3 (progress notifications, cancellation, the process-pool matrix,
dynamic toolsets, HTTP transport) remains unbuilt and remains deferred.
