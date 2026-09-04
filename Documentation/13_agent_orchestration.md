# Agent Orchestration

`get_agent_tools()` returns 178 LLM-callable tools (see
[07_agent_tools.md](07_agent_tools.md) /
[09_advanced_agent_tools.md](09_advanced_agent_tools.md)). Handing all 178 to
one model on every call — the default behavior of every single-agent script
in `Implementation/{Anthropic,OpenAI,Gemini}/` — is the largest untreated
source of tool-selection error: similarly-named or similarly-scoped tools
(`run_sma_backtest` vs `run_custom_signal_backtest`, `run_backtest_optimization`
vs `run_sma_backtest`) compete for the model's attention on every single
turn, whether or not the request actually needs them.

This page covers three mechanisms, in increasing order of strictness:

1. **The tool-category router** (`standard_quant_tools.agent.router`) — a
   single cheap classification call that narrows the tool list before the
   real completion call, without spinning up a separate agent session.
   Used by every `Implementation/*/Agent_*.py` script that draws on the
   179-tool surface.
2. **The multi-agent orchestrator** (`Multi_Agent_Implementation/`) — a lead
   agent that delegates each sub-task to one of 16 specialist worker agents,
   each with its own independent session and system prompt scoped to a
   small, non-overlapping tool subset.
3. **Runtimes** (`standard_quant_tools.agent.runtimes`) — the only one of
   the three that is ENFORCED. See [19_runtimes.md](19_runtimes.md).

The first two build on the same underlying taxonomy, so a tool's
categorization only ever needs to be correct in one place.

> **Narrowing versus enforcing.** The router and the workers narrow what a
> model is *shown*. Neither stops `dispatch()` from running something else,
> because that function knows every tool. If an agent must be unable to
> reach a tool — rather than merely unlikely to pick it — give it a runtime,
> whose dispatch table holds only its own tools and refuses the rest by
> name.

## Three registries, ten runtimes

Everything above concerns the 179-tool analysis and backtest surface. There
are two more: `standard_quant_tools.modeling.agent`, 20 tools, and the
9-tool `feature_lab` runtime — neither of which the library merges into the
first, see [15_modeling.md](15_modeling.md) for why. 178 + 20 + 9 is the
208-tool whole surface. The example implementations keep the same
separation, and it shows up in three places:

| | Analysis registry | Modeling registry |
|---|---|---|
| Module | `standard_quant_tools.agent` | `standard_quant_tools.modeling.agent` |
| Size | 179 tools, 13 categories, 8 runtimes | 20 tools, one ordered pipeline |
| Narrowing | `route_request()` → `categories=` | nothing to narrow — the pipeline runs in sequence |
| Single-agent script | `Agent_*.py` (eleven of the thirteen; the data runtime has no script of its own yet) | `Agent_Model_Builder.py`; `Agent_Model_Backtester.py` spans both |
| Workers | 13 | 2 (`model_research`, `model_builder`), plus `feature_lab` on its own runtime |

Each `_agent_utils.py` names a registry once and gets that registry's tool
schemas **and** its dispatch function together:

```python
run_agent(..., registry="modeling")     # 20 tools, modeling_dispatch
run_agent(..., registry="analysis")     # 179 tools, dispatch  (the default)
```

`registry=` also accepts a RUNTIME name, and that is the safer choice:

```python
run_agent(..., registry="research")             # 42 tools, scoped dispatch
run_agent(..., registry="research+portfolio")   # joined explicitly
```

The difference is what happens on a wrong guess. `"analysis"` hands back
the union dispatcher, which knows every tool regardless of what was
advertised — so a model that invents a tool it was never shown gets a
RESULT. A runtime hands back a table holding only its own tools, and the
refusal names the runtime that actually owns what was asked for.

Every `Implementation/*/Agent_*.py` script now names a runtime, and each
one's scope is derived from the tools its own prompt mentions rather than
chosen by hand — a prompt that instructs the agent to call a tool the
runtime will refuse is worse than no scoping, because it walks the model
into a wall it was told to walk into. Several of these genuinely span two
or three runtimes; that is what a real workflow looks like, and the value
is that the span is now written down instead of being an accident.

The sixteen workers dispatch through their category's runtime for the same
reason. Each already declared a fixed, non-overlapping tool subset — that
is the architecture — but dispatching through the union made the subset
advisory.

The analysis registry is itself divided into six **runtimes** —
`research`, `backtest`, `portfolio`, `microstructure`, `derivatives`,
`meta` — which are the same idea one level down: a dispatch table that
refuses what it does not own. The modeling registry was the first runtime;
the other seven generalize it. Results still cross freely between all
eight, by value rather than by shared dispatch.
[19_runtimes.md](19_runtimes.md) is the whole story.

That pairing is the whole point. The two registries have identical shapes —
same OpenAI-format schema, same `dispatch(tool_name, arguments)` signature —
so nothing structural stops you from loading one registry's tool list and
calling the other's dispatcher. It would fail at the first tool call with an
"unknown tool" error naming the model's choice, which reads like the model
picked badly rather than like the wiring is wrong. Binding the pair together
in one lookup is what makes that mistake unwriteable.

Passing `categories=` alongside `registry="modeling"` raises rather than
being ignored: a caller who thought they had narrowed the tool list and
silently did not is worse off than one who gets an error.

---

## The category taxonomy (`TOOL_CATEGORY`)

`standard_quant_tools.agent.tools.TOOL_CATEGORY: Dict[str, str]` is the
single source of truth: every one of the 179 tool names mapped to exactly
one of 13 category keys. Each category belongs to exactly one runtime.

| Category | Tools | Runtime | Covers |
|---|---|---|---|
| `screener` | 2 | `research` | Filter a universe, fetch fundamentals |
| `analysis` | 14 | `research` | Single-asset risk/technical/volatility profiling, multi-asset portfolio metrics, panel indicators, data quality |
| `quant_research` | 26 | `research` | Factor regression, cointegration/pairs, Kalman hedge ratio, PCA, Hurst, correlation, and the inference layer — bootstrap intervals, tail index, structural breaks, lead-lag |
| `backtest_execution` | 12 | `backtest` | Run a built-in strategy / portfolio / pair trade / strategy matrix **once**, fixed parameters |
| `backtest_validation` | 21 | `backtest` | Optimize/validate/diagnose — grid search, walk-forward, regime-adaptive, robustness, Monte Carlo, cost sweep, drawdown table, and the overfitting layer (deflated Sharpe, PBO, purged combinatorial CV, reality check) |
| `custom_signal` | 2 | `backtest` | Backtest a signal computed outside this library |
| `portfolio_risk` | 18 | `portfolio` | Risk decomposition, portfolio construction/optimization, sizing, capacity, stress testing, liquidity, trade cost |
| `microstructure` | 17 | `microstructure` | Spreads MEASURED from tick data, the eight bar-based estimators for when there is no tick feed, and a check of the OHLCV proxies against them |
| `delta_one` | 18 | `delta_one` | Carry, basis, futures curves and rolls, hedge sizing, baskets and replication, ETF fair value, swaps and TRFs, and the comparison that normalizes six ways of holding one exposure |
| `derivatives` | 12 | `derivatives` | Option pricing and second-order greeks, multi-leg payoffs, smile/term-structure fitting, put-call parity, delta-hedge simulation |
| `data` | 17 | `data` | Fetch OHLCV, return panels, tick tapes and quotes, build continuous futures, register external datasets too large to copy, and publish them all as `sqt://` references every other runtime reads |
| `discovery` | 13 | `meta` | What the library accepts and what the provider can serve; describe or validate a tool call before making it; the handoff reference map |
| `provenance` | 6 | `meta` | Read and verify the decision log. Read-only by design |

`backtest_execution`/`backtest_validation` is a deliberate split of what
used to be one 16-tool `backtest` bucket: "run SMA on AAPL" and "find the
best SMA parameters" are different jobs, and letting both sets of tools
compete for one model's attention reintroduces exactly the kind of
tool-selection ambiguity this whole page exists to avoid.

```python
from standard_quant_tools.agent.tools import TOOL_CATEGORY, get_agent_tools

TOOL_CATEGORY["run_sma_backtest"]           # "backtest_execution"
TOOL_CATEGORY["run_backtest_optimization"]  # "backtest_validation"

# Filter the registry to just the categories you need:
tools = get_agent_tools(categories=["screener", "analysis"])
```

`get_agent_tools(categories=None)` (the default) returns every tool,
byte-for-byte identical to calling it with no arguments at all — every
existing caller (`dispatch()`, any script that hasn't adopted a router yet)
keeps working unchanged. An unknown category name is silently ignored
rather than raising, since narrowing is a confidence optimization, not a
strict validator.

`tests/agent/test_agent_tools.py::TestToolCategoryCoverage` is the drift-proofing
test: every `_TOOL_DISPATCH` key has exactly one `TOOL_CATEGORY` entry and
vice versa. Add a tool without categorizing it and this test fails
immediately — the same class of drift that used to leave README/comments
variously claiming 34, 42, or 45 tools now fails CI instead of silently
compounding.

---

## The router (`standard_quant_tools.agent.router`)

A single cheap classification call that narrows the tool list to the 1-2
categories a request actually needs, run once before the real agent loop
starts — no separate agent session required. This is the mechanism every
`Implementation/{Anthropic,OpenAI,Gemini}/Agent_*.py` script uses.

```python
from standard_quant_tools.agent.router import (
    TOOL_CATEGORIES, build_router_prompt, parse_router_response,
)
```

- `TOOL_CATEGORIES: Dict[str, Dict[str, str]]` — one `{"label", "description"}`
  entry per category, the prose the classification prompt shows the model.
  Distilled from `Multi_Agent_Implementation/worker_agents.py`'s
  already-tuned worker system prompts, not written from scratch.
- `build_router_prompt(request, categories) -> str` — pure string template,
  no network call. Lists every category and asks for the 1-2 most relevant
  keys.
- `parse_router_response(raw_text, valid_keys) -> List[str]` — parses the
  model's response (JSON array first, falls back to scanning for bare
  category-key tokens in the text).

This module makes **zero network calls**. Provider-specific glue — which
client, which cheap model — lives in each provider's own `_agent_utils.py`:

```python
# Implementation/Anthropic/_agent_utils.py
def route_request(request: str, api_key: str, model: str = "claude-haiku-4-5") -> List[str]:
    ...  # one Anthropic call, build_router_prompt + parse_router_response

def run_agent(..., categories: Optional[List[str]] = None) -> str:
    tools = _to_anthropic_tools(get_agent_tools(categories=categories))
    ...
```

Usage in a script (the pattern every `Agent_*.py` follows):

```python
routed_categories = route_request(USER_REQUEST, api_key=API_KEY, model=MODEL)
result = run_agent(
    system_prompt=SYSTEM_PROMPT,
    user_request=USER_REQUEST,
    api_key=API_KEY,
    model=MODEL,
    categories=routed_categories,
)
```

The same `route_request()`/`run_agent(categories=...)` pair exists in all
three providers' `_agent_utils.py` (`Implementation/Anthropic/`,
`Implementation/OpenAI/`, `Implementation/Gemini/`) — the router itself is
provider-agnostic; only the classification call's client/model differs.

### Fail open, not closed

**Design principle:** `parse_router_response` returns every category (i.e.
no narrowing at all) whenever it can't confidently extract at least one
valid category from the model's response — an empty response, malformed
text, a response naming only unknown category keys, or the classification
API call itself failing. A router that wrongly excludes a tool the caller
actually needed is worse than today's unfiltered list; narrowing is a
confidence optimization, never a hard gate. `route_request()` in each
`_agent_utils.py` follows the same rule: any exception during the
classification call is caught and logged, and the function falls through to
`parse_router_response("", ...)`, which fails open the same way.

### Testing

`tests/agent/test_router.py`:
- `parse_router_response`/`build_router_prompt` unit tests — no network,
  covers valid JSON, prose-wrapped JSON, the bare-token fallback,
  malformed/empty/all-unknown-key inputs (confirms fail-open fires exactly
  when expected), and deduplication.
- `TestRoutingAccuracyEval` — an `@pytest.mark.integration`-gated eval: 10
  labeled representative requests, run through a real `route_request()`
  call, asserting ≥70% top-1 accuracy. Skipped by default (matches this
  repo's `-m "not integration"` CI convention) since it costs real API
  calls; run manually with `pytest -m integration tests/agent/test_router.py`
  (requires `ANTHROPIC_API_KEY`). This is the first actual measurement of
  routing *correctness* in this codebase — the multi-agent coverage test
  below only ever checked tool-set coverage/disjointness, never whether a
  classifier picks the right category for a real request.

---

## The multi-agent orchestrator (`Multi_Agent_Implementation/`)

A heavier but more thorough answer to the same problem: instead of
narrowing one model's tool list, delegate to one of 14 independent worker
agents, each with its own session, system prompt, and fixed tool subset —
the confusable tool is never loaded into the worker's context at all,
not just deprioritized.

```
Multi_Agent_Implementation/
├── Agent_Orchestrator.py   # lead agent: 16 delegate_to_<worker>_agent tools
├── worker_agents.py        # WORKER_AGENTS registry + run_worker_agent()
└── _agent_utils.py         # scoped variant of Implementation/Anthropic's run_agent()
```

Thirteen of the sixteen draw from the analysis registry, two from the
modeling one, and one from `feature_lab`. The orchestrator does not need to
know which is which — a delegate call looks the same either way, and each
worker carries its own `"registry"` field that `run_agent()` reads. That is
what routing through workers buys: a third registry cost one more worker
entry, not a redesign of the delegation loop.

`WORKER_AGENTS` (`worker_agents.py`) has one entry per category above —
for the analysis workers the keys *are* the `TOOL_CATEGORY` values — and
each of those workers' `"tools"` lists is **derived** from `TOOL_CATEGORY`,
not hand-duplicated:

```python
def _tools_for(category: str) -> List[str]:
    return sorted(name for name, cat in TOOL_CATEGORY.items() if cat == category)

WORKER_AGENTS = {
    "screener": {"registry": ANALYSIS_REGISTRY,
                 "tools": _tools_for("screener"), ...},
    "backtest_execution": {"registry": ANALYSIS_REGISTRY,
                           "tools": _tools_for("backtest_execution"), ...},
    ...
}
```

A tool's category only ever needs to be correct in `TOOL_CATEGORY` itself
to show up correctly in both the router's classification prompt and this
worker registry — there is no second list that can drift out of sync.

**The two modeling workers are split differently, because there is nothing
to derive them from.** The modeling runtime has no category taxonomy — it
is sixteen tools in one ordered pipeline — so the split is by pipeline
*stage*, written out explicitly and then checked by the coverage test the
same way `_tools_for()` is:

```python
_MODEL_RESEARCH_TOOLS = ["list_modeling_capabilities", "list_features",
                         "build_model_dataset", "validate_pit_records",
                         "join_point_in_time", "analyze_features",
                         "list_datasets", "check_leakage",
                         "validate_model_spec"]
_MODEL_BUILDER_TOOLS  = ["run_model_experiment", "inspect_model",
                         "score_model", "evaluate_model_portfolio",
                         "list_models", "compare_models",
                         "score_predictions"]
```

The cut is at the dataset. Everything up to "is this dataset worth fitting"
is research; everything after it is construction. That is where a human
would stop and look, and it is the only handoff in the pipeline that
carries a single value — the `dataset_id` — rather than a whole panel,
which is what makes it a viable boundary between two agent sessions that
cannot see each other's context.

It is also a real constraint, not a tidy one. `model_builder` has no tool
that can create a dataset, so the orchestrator must run `model_research`
first and copy the `dataset_id` verbatim into the builder's request. The
orchestrator's system prompt states that ordering explicitly rather than
leaving it to the general "chain specialists when needed" rule.

The orchestrator's own "tools" are 14 hand-authored
`delegate_to_<worker>_agent(request)` tools, **auto-generated from
`WORKER_AGENTS.keys()`** — adding, splitting, or removing a worker in
`worker_agents.py` changes the orchestrator's delegate-tool set and system
prompt automatically, no separate count to keep in sync.

```python
from worker_agents import WORKER_AGENTS, run_worker_agent

run_worker_agent("backtest_execution", "Run SMA crossover on AAPL for 2023",
                  api_key=API_KEY)
```

Each worker only ever sees the sub-task string the orchestrator passes it —
not the original user request, not any other worker's output — unless the
orchestrator explicitly copies that text into the delegate call. See
`Agent_Orchestrator.py`'s system prompt for the exact rules it follows.

### Testing

`tests/agent/test_multi_agent_tool_coverage.py` — pure data validation, no API
key or network required:
- every worker has tools, a system prompt, a label, a description, and a
  registry name the loader actually understands
- **per registry**, the union of that registry's workers' tools equals that
  registry's full tool set, checked both directions (nothing missing,
  nothing referencing a nonexistent tool). Checking the merged union
  instead would pass while a worker quietly listed a modeling tool under
  the analysis registry — which fails at the first tool call, because the
  two dispatch functions do not know each other's names
- the two registries share no tool name, so "which dispatcher runs this"
  can never depend on which worker happened to be asked
- no tool is assigned to more than one worker
- `build_model_dataset`/`analyze_features` sit in `model_research` and
  `run_model_experiment`/`evaluate_model_portfolio` in `model_builder`, so
  the `dataset_id` handoff cannot quietly disappear by the builder gaining
  the ability to build its own dataset
- `run_sma_backtest`/`run_custom_signal_backtest` land in different workers
  (the original confusion-pair regression test)
- `run_sma_backtest`/`run_backtest_optimization` land in different workers
  (`backtest_execution` vs `backtest_validation` — the same guarantee one
  level narrower)
- exactly 16 workers exist, by name — a regression guard for the
  execution/validation and research/builder splits, derived from the actual
  registry rather than a magic number that could itself drift

There is currently no test that runs the orchestrator's delegation loop
itself against a live/mocked API, or asserts it picks the *correct* worker
for a given request (unlike the router's routing-accuracy eval above) — a
known gap, not a hidden one.

---

## Router vs. multi-agent orchestrator: when to use which

| | Router | Multi-agent orchestrator |
|---|---|---|
| Cost | One cheap classification call, then the normal agent loop | A full independent agent session per delegated worker |
| Setup | `route_request()` + `categories=` param on an existing `run_agent()` | A lead agent, 16 worker sessions, its own delegation loop |
| Isolation | The unrouted tools are absent from *this* completion call | The unrouted tools are absent from that *worker's entire session* |
| Best for | Single-agent scripts, cost-sensitive deployments, most requests | Complex multi-step requests that genuinely span several specialist domains in one turn (see `Agent_Orchestrator.py`'s example request) |

They're not mutually exclusive — the router could narrow which worker(s)
an orchestrator delegates to in the first place, though that composition
isn't wired up in this repo today.

---

## What this doesn't solve

Both mechanisms reduce the *size* and *relevance* of what's in front of the
model; neither guarantees correct tool selection. A misrouted request (the
router picked the wrong 1-2 categories, or an orchestrator delegated to the
wrong worker) can still under-serve a request that needed a tool outside
the categories it was routed to — the fail-open design means this shows up
as "the model didn't have the tool it needed" rather than a crash, but it's
still a real failure mode worth watching for via the routing-accuracy eval
above. Neither mechanism validates that the *arguments* passed to a
correctly-selected tool are sensible — that's Pydantic's job at the
`dispatch()` boundary, not this page's.
