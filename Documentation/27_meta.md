# The Meta Runtime

Twenty tools that answer questions about **this library and this
session**, never about a market.

Every other runtime tells you something about the world. This one tells you
what the library can do, what a data source can promise, what a published
value contains, and what a previous call actually did. It is the runtime an
agent reaches for when it does not yet know enough to ask a real question.

## Why it exists as its own boundary

An agent that guesses a tool name gets an error. An agent that guesses a
strategy's parameter bounds gets a failed call. An agent that guesses
whether a provider serves ticks wastes a fetch. Each of those is a round
trip spent learning something the library already knew and had no way to
say.

The alternative to these tools is not "the agent just knows" — it is the
agent finding out from a failure. `describe_tool` costs one call;
discovering the same contract from a rejected argument costs a call, an
error, and a retry that may guess wrong again.

## Pre-flight: ask before you act

**`describe_tool`** returns one tool's full contract — arguments, result
fields, owning runtime, and whether calling it fetches or writes. It
answers for *any* runtime, because describing a tool is not calling it.
That matters when a narrowly-scoped agent has heard of a tool it cannot
run: it can still learn what the tool is before deciding to ask for a
handoff.

**`validate_tool_call`** checks arguments against a schema **without
calling**. Two layers, because the library has two: the JSON schema, and
the strategy parameter contract underneath it. A grid search with an
invalid parameter range fails after the fetch without this, and before it
with.

**`estimate_tool_cost`** reports what each runtime costs a client's
context, in bytes and approximate tokens. There is no fixed ceiling — what
a client can afford depends on its model and its session — so this
measures rather than judges.

**`describe_runtime`** and **`list_reference_kinds`** answer the two
structural questions: what runtimes exist and what each owns, and what
content kinds a reference can carry and what converts to what.

## Registry discovery: never recite from memory

**`list_strategies`** returns every built-in strategy's parameter contract
— names, kinds, defaults, bounds, and the relations that must hold between
them.

This is the tool that makes the generic-registry design work. There are
eight strategies and only four have dedicated tools; `donchian_breakout`,
`momentum_timeseries`, `vwap_reversion` and `adx_trend` are executable by
name through `run_strategy_matrix` and have no tool named after them.
**An agent answering "the library supports SMA, RSI, MACD and Bollinger"
is giving a wrong answer that sounds like a complete one** — and the only
defence is asking rather than remembering.

**`list_stress_scenarios`** does the same for the named historical crash
windows `run_stress_test` accepts.

## What a data source can promise

**`describe_data_capabilities`** answers whether the active provider serves
tick trades, top-of-book quotes or async OHLCV, and which bar intervals it
accepts. Most environments have no tick feed, so asking first is the
difference between a routed request and a refused one.

**`describe_temporal_contract`** asks what a source can say about *when*
its facts became knowable — **before anything is fetched**. A quarterly
filing describes 30 September and is published on 25 October; joining it on
the quarter end is three weeks of hindsight in every row, and the contract
refuses rather than degrading silently.

**`compare_data_sources`** fetches the same fundamentals from two providers
and separates a missed unit conversion from a genuine definition
difference. Those need different fixes, and a percentage gap alone does not
say which you are looking at.

## Reading what other runtimes produced

**`describe_artifact`** reports the shape, date span, per-column statistics
and both ends of a persisted Parquet artifact — so a result can be
inspected instead of the run being repeated.

**`describe_reference`** does the same for a handoff reference: what kind
it carries, its shape, and which runtime produced it.

**`read_reference`** is the other half of it: the actual values at rows
you name, by date or the first/last N. References exist to keep bulk values
out of the conversation, and every reference tool honoured that so
completely that an agent could fetch a price series, confirm its anchor
dates and hand it to an analysis tool while never being able to state a
close. The window is bounded on purpose — ask for the rows you intend to
cite.

**`convert_reference`** turns one kind of published value into another —
raw model predictions into a signal panel, scores into weights. Only
conversions that are genuinely well defined exist; there is deliberately no
best-effort path, because a handoff that guesses is worse than one that
refuses.

**`compare_artifacts`** is a field-by-field diff of two result objects,
ordered by the size of the change.

## The audit trail, read-only

Four of these five READ the decision log; `export_audit_bundle` copies a
range of it out. **None of them mutates it** — holds, sealing, garbage
collection and checkpoint signing stay on the `sqt` CLI, because a surface
an agent can reach should not be able to edit its own record.

| Tool | Answers |
|---|---|
| `explain_decision` | What one recorded call did: inputs, the data it read with content hashes, which execution path ran |
| `replay_decision` | Re-run it and classify: reproduced, `data_changed`, or a genuine code change |
| `compare_decisions` | Diff two recorded calls and say which of the differences explains the outcome |
| `verify_audit_integrity` | Check the tamper-evident hash chain, for one day or the whole trail |
| `export_audit_bundle` | Package a date range plus its chain index and manifest into one zip for an external auditor |

**`replay_decision`'s classification is the point.** "The output changed"
is not useful on its own; "the inputs are identical and the output moved"
is a code change, while "the upstream data moved" is not. Distinguishing
them is what makes the audit trail worth keeping.

`export_audit_bundle` WRITES a file, and the path is CONFINED. A bare or relative name lands under `$SQT_RUNS_DIR/bundles` rather than the working directory; an absolute path is allowed, since a bundle exists to be handed to someone outside this process, but its parent directory must already exist. An existing destination is REFUSED rather than overwritten -- a bundle is evidence, and `out_path` is a string chosen by a model. It is the one
tool in this runtime that is not purely read-only, and it is included
because handing an auditor a bundle is a read operation from the log's
point of view — it copies, never edits.

## The tools

| Tool | Answers |
|---|---|
| `describe_tool` | One tool's full contract, for any runtime |
| `validate_tool_call` | Are these arguments valid — without calling |
| `estimate_tool_cost` | What each runtime costs a client's context |
| `describe_runtime` | What each runtime is for and what it owns |
| `list_reference_kinds` | What a reference can carry, and what converts to what |
| `list_strategies` | Every built-in strategy's parameter contract |
| `list_stress_scenarios` | The named historical crash windows |
| `describe_data_capabilities` | What the active provider can actually serve |
| `describe_temporal_contract` | What a source can say about when facts became knowable |
| `compare_data_sources` | Two providers, one field: unit or definition difference |
| `describe_artifact` | What a persisted Parquet artifact contains |
| `describe_reference` | What a handoff reference points at |
| `read_reference` | The actual values at chosen rows of one |
| `convert_reference` | Turn one published kind into another |
| `compare_artifacts` | Field-by-field diff of two results |
| `explain_decision` | What one recorded call did |
| `replay_decision` | Re-run it, and say what changed |
| `compare_decisions` | Diff two recorded calls |
| `verify_audit_integrity` | Is the hash chain intact |
| `export_audit_bundle` | Package a date range for an auditor |

Full argument lists: [20_tool_index.md](20_tool_index.md#meta--discovery--provenance).

## Related

- [10_auditability.md](10_auditability.md) — the decision record itself, and the `sqt` CLI
- [19_runtimes.md](19_runtimes.md) — runtimes, references and pre-flight
- [26_data.md](26_data.md) — the runtime that fetches, where `meta` only describes
- [18_mcp.md](18_mcp.md) — `describe_tool` is what makes thinned listings work
