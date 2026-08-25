# Runtimes and the Handoff Interconnect

Two mechanisms, and they answer opposite questions.

A **runtime** decides what an agent may *execute*. A **reference** decides
how a value gets from one runtime to another. Keeping them separate is what
makes the boundary strict without making it obstructive: execution is
scoped, data is not.

---

## 1. Why runtimes exist

`get_agent_tools(categories=[...])` has always been able to narrow the
schema list handed to a model. `dispatch()` never honoured it:

```python
>>> exposed = {t["function"]["name"] for t in get_agent_tools(categories=["screener"])}
>>> exposed
{'run_screener', 'get_stock_fundamentals'}

>>> dispatch("list_stress_scenarios", {})     # never advertised to this agent
{'scenarios': [...]}                          # ...and it ran anyway
```

An agent scoped to two screener tools that hallucinated a backtest tool got
a **successful result**. The narrowing was advisory at the schema layer and
absent at the execution layer, so a wrong guess was rewarded — the worst
possible feedback to give a model. The MCP server did enforce its own
selection; nothing else did, which meant every `Implementation/*` script
and the whole multi-agent orchestrator ran with a boundary that was not
there.

A runtime closes that. Each holds its own dispatch table containing only
its own tools, so a name from another runtime is not discouraged — it is
unroutable. This is the guarantee the modeling registry has had since it
shipped, generalized to the rest of the surface.

### The five runtimes

| Runtime | Tools | Categories | What it is for |
|---|---|---|---|
| `research` | 23 | `screener`, `analysis`, `quant_research` | Describe an asset or a universe. Does not run strategies. |
| `backtest` | 21 | `backtest_execution`, `backtest_validation`, `custom_signal` | Run, optimize, validate and diagnose a strategy. Does not build portfolios. |
| `portfolio` | 10 | `portfolio_risk`, `microstructure` | Turn a view into a position and price what it costs. |
| `meta` | 14 | `discovery`, `provenance` | Questions about the library and the session, not about a market. |
| `modeling` | 14 | (one ordered pipeline) | Build, validate and score a model. Lives in `modeling/agent`. |

The grouping is deliberately coarse. A runtime holding two tools is
overhead rather than isolation, so nothing has fewer than eight and a test
pins that. `microstructure` sits with `portfolio_risk` because
`get_liquidity_metrics` and `check_spread_proxy` are the same question at
two data fidelities; `screener` sits with `analysis` because you screen in
order to analyze.

### Runtime is not category

`TOOL_CATEGORY` is unchanged and still drives `agent/router.py`, the MCP
`--categories` flag, and the twelve workers in `Multi_Agent_Implementation`.

- A **category** hints at which tools suit a request.
- A **runtime** states which tools a caller may execute.

Several categories live inside one runtime. Narrowing by category *within*
a runtime works; it can never widen past it.

### Using one

```python
from standard_quant_tools.agent.runtimes import resolve, combine

research = resolve("research")
research.get_tools()                       # 23 schemas, this runtime only
research.dispatch("run_screener", {...})   # fine

research.dispatch("run_sma_backtest", {...})
# ValueError: 'run_sma_backtest' exists but belongs to the 'backtest'
# runtime, not to 'research'. This caller is scoped to 'research', which
# provides: [...]. Either use one of those, or construct the 'backtest'
# runtime deliberately -- widening scope is a decision, not a fallback.
```

The error names the runtime that actually owns the tool. A bare "unknown
tool" cannot be told apart from a hallucinated name, and a model receiving
one guesses again.

A workflow that genuinely spans runtimes composes them explicitly:

```python
wide = combine(["research", "backtest"])   # named "research+backtest"
```

The widening is then visible in the code that asked for it, rather than
being the silent default it used to be.

### The runtimes partition the surface

Every tool belongs to exactly one, and a test enforces it. Duplicating a
convenient tool into a second runtime would dissolve the boundary at
exactly the points where it matters most.

---

## 2. Why the interconnect exists

Runtimes isolate **execution, not data**. Every real workflow spans them —
screen in `research`, backtest in `backtest`, size in `portfolio`, hand a
model's predictions from `modeling` to a backtest — and a boundary that
also blocked results would make the orchestrator impossible.

The first attempt at moving predictions into a backtest was a bespoke tool
that knew about both sides. That does not scale: **N producers and M
consumers need N × M bridges**, each written, tested and kept in step with
both ends, and each a place where the two sides' assumptions can quietly
diverge.

A typed reference makes it **N + M**.

```
sqt://<kind>/<run_id>/<name>
```

A producer publishes a bulk value and gets back a string. Any consumer in
any runtime resolves that string. Neither side knows the other exists, and
nothing is transcribed through a context window to get from one to the
other.

### The kind is the point

The content kind is checked on resolve:

```python
handoff.resolve(ref, expect="equity_curve")
# ValidationError: expected an 'equity_curve' reference but
# 'sqt://trade_log/run1/trades' is a 'trade_log'. ...
```

That is what turns a wrong handoff between two tool calls from plausible
garbage into an error naming both kinds — the difference between an agent
recovering and an agent confidently reporting a drawdown computed from a
trade log.

### The kinds

| Kind | Holds |
|---|---|
| `equity_curve` | Account value per bar |
| `trade_log` | One row per completed trade |
| `signal_panel` | `{ticker: {date: -1/0/+1}}` — what `run_signal_panel_backtest` consumes |
| `weight_panel` | `{ticker: {date: weight}}` — what `run_portfolio_simulation` consumes |
| `score_panel` | `{ticker: {date: score}}` — unrestricted alpha scores |
| `returns_panel` | Wide frame of per-asset returns |
| `price_panel` | Wide price frame or stacked OHLCV |
| `predictions` | Long `(date, entity, prediction)` frame |
| `feature_panel` | Computed features, entity by date |
| `indicator_panel` | Indicator values across a universe |

`list_reference_kinds` returns this table plus what converts to what — the
map of which producer outputs can reach which consumer inputs.

### The path, end to end

```python
experiment = modeling_dispatch("run_model_experiment", {...})
# -> oos_predictions_ref: "sqt://predictions/mdl_ab12/oos_predictions_ref"

converted = resolve("meta").dispatch("convert_reference", {
    "ref": experiment["oos_predictions_ref"],
    "to_kind": "signal_panel",
    "run_id": "study7", "name": "sig",
    "task": "regression",
})

resolve("backtest").dispatch("run_signal_panel_backtest", {
    "tickers": [...], "start_date": ..., "end_date": ...,
    "signal_panel_ref": converted["ref"],
})
```

No code anywhere in that chain knows about more than one side at a time.

### Conversions

Only conversions that are genuinely well defined exist. There is
deliberately **no best-effort path** — a handoff that guesses is worse than
one that refuses.

| From | To | Note |
|---|---|---|
| `predictions` | `signal_panel` | Collapses to −1/0/+1. Magnitude is discarded on purpose: a signal panel's value is read as a leverage multiplier, so a raw 0.02 forward-return prediction would size a 2%-leveraged position. |
| `predictions` | `score_panel` | Raw predictions, unscaled. |
| `score_panel` | `weight_panel` | Through `backtest.sizing` — the same constructor `run_portfolio_simulation` would have used, not a reimplementation. |
| `signal_panel` | `score_panel` | Relabels only. The information magnitude carried was discarded upstream and this does not recover it. |

### Why references beat a shared dispatch table

- **They cross processes.** In the orchestrator two runtimes are two
  agents; a string survives that, a table does not.
- **They are auditable.** The handoff appears in the decision log as an
  input to the second call.
- **They carry no execution rights.** Holding an equity-curve reference
  lets you *describe* that curve from `meta` and still does not let you run
  `get_drawdown_table`, which is `backtest`'s.

### Built for many agents, not one session

Two properties matter once agents publish concurrently, and both are
enforced:

- **One sidecar file per artifact**, never a shared per-`run_id` catalogue.
  A catalogue has to be read, updated and rewritten, so two agents
  publishing different names under one `run_id` race and the loser's entry
  disappears — leaving a live reference that resolves to data of unknown
  kind.
- **`publish()` refuses to overwrite** by default. A reference promises
  that resolving it twice gives the same value; replacing what one agent
  published because another chose the same `(run_id, name)` breaks that for
  every holder, including holders already recorded in an audit log.

`describe_reference` returns a content hash, so a consumer can prove it read
what the producer wrote — across a fleet those are different processes at
different times, and "same reference" is only as good as "same bytes".

### Bulk inputs only

A reference is for data too large to be worth moving through a model's
context. Small results stay inline: a reference to a Sharpe ratio would be
indirection with no benefit and one more thing that can dangle.

---

## 3. Pre-flight

Two `meta` tools exist because of the same concern the runtimes address.

`describe_tool` reports one tool's arguments, result fields, owning runtime,
and whether calling it fetches or writes. The alternative was loading all
82 schemas — which is exactly what the MCP category budget exists to avoid,
so a narrowly-scoped agent could not learn about a tool it had heard of
without paying for every tool it had not. It answers for any runtime,
because describing a tool is not calling it.

`validate_tool_call` checks arguments **without calling**. Two layers,
because the library has two:

1. The Pydantic schema — missing, unknown and out-of-range arguments.
2. The strategy parameter contract — invisible to JSON Schema, which types
   `parameters` as an open dict, so `lookback=-20` passes a schema check
   and is still look-ahead by construction.

### Unknown arguments are rejected

Every tool input sets `extra="forbid"`. Before that:

```python
BacktestInput(..., comission_pct=0.05)   # note the typo
```

...succeeded, dropped the typo, ran at the 0.001 default, and left the
caller believing it had set 5% commission. `backtest/strategy_params.py`
exists because of exactly this failure one layer down; the same hole was
open at the boundary where a *model* chooses the argument names.

Result models stay permissive: the library constructs those from its own
values, and tightening them would only turn a forward-compatible field
addition into a crash.

---

## See also

- [13_agent_orchestration.md](13_agent_orchestration.md) — the router and the twelve workers
- [18_mcp.md](18_mcp.md) — how runtimes reach an MCP client
- [15_modeling.md](15_modeling.md) — the modeling runtime
- [10_auditability.md](10_auditability.md) — what every dispatch records
