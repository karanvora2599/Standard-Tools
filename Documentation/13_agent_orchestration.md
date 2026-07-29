# Agent Orchestration

`get_agent_tools()` now returns 45 LLM-callable tools (see
[07_agent_tools.md](07_agent_tools.md) /
[09_advanced_agent_tools.md](09_advanced_agent_tools.md)). Handing all 45 to
one model on every call — the default behavior of every single-agent script
in `Implementation/{Anthropic,OpenAI,Gemini}/` — is the largest untreated
source of tool-selection error: similarly-named or similarly-scoped tools
(`run_sma_backtest` vs `run_custom_signal_backtest`, `run_backtest_optimization`
vs `run_sma_backtest`) compete for the model's attention on every single
turn, whether or not the request actually needs them.

This page covers the two mechanisms this library provides to narrow that
tool list before it reaches the model, and when to use which:

1. **The tool-category router** (`standard_quant_tools.agent.router`) — a
   single cheap classification call that narrows the tool list before the
   real completion call, without spinning up a separate agent session.
   Used by every `Implementation/*/Agent_*.py` script.
2. **The multi-agent orchestrator** (`Multi_Agent_Implementation/`) — a lead
   agent that delegates each sub-task to one of 7 specialist worker agents,
   each with its own independent session and system prompt scoped to a
   small, non-overlapping tool subset.

Both are built on the same underlying taxonomy, so a tool's categorization
only ever needs to be correct in one place.

---

## The category taxonomy (`TOOL_CATEGORY`)

`standard_quant_tools.agent.tools.TOOL_CATEGORY: Dict[str, str]` is the
single source of truth: every one of the 45 tool names mapped to exactly
one of 7 category keys.

| Category | Tools | Covers |
|---|---|---|
| `screener` | 2 | Filter a universe, fetch fundamentals |
| `analysis` | 12 | Single-asset risk/technical/volatility/option profiling, multi-asset portfolio metrics, data quality |
| `quant_research` | 7 | Factor regression, cointegration/pairs, Kalman hedge ratio, PCA, Hurst, correlation |
| `backtest_execution` | 9 | Run a built-in strategy / portfolio / pair trade **once**, fixed parameters |
| `backtest_validation` | 7 | Optimize/validate/diagnose a built-in strategy — grid search, walk-forward, regime-adaptive, robustness, Monte Carlo |
| `custom_signal` | 2 | Backtest a signal computed outside this library |
| `portfolio_risk` | 6 | Risk decomposition, portfolio construction/optimization, sizing, capacity, stress testing, liquidity |

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

`tests/test_agent_tools.py::TestToolCategoryCoverage` is the drift-proofing
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

`tests/test_router.py`:
- `parse_router_response`/`build_router_prompt` unit tests — no network,
  covers valid JSON, prose-wrapped JSON, the bare-token fallback,
  malformed/empty/all-unknown-key inputs (confirms fail-open fires exactly
  when expected), and deduplication.
- `TestRoutingAccuracyEval` — an `@pytest.mark.integration`-gated eval: 10
  labeled representative requests, run through a real `route_request()`
  call, asserting ≥70% top-1 accuracy. Skipped by default (matches this
  repo's `-m "not integration"` CI convention) since it costs real API
  calls; run manually with `pytest -m integration tests/test_router.py`
  (requires `ANTHROPIC_API_KEY`). This is the first actual measurement of
  routing *correctness* in this codebase — the multi-agent coverage test
  below only ever checked tool-set coverage/disjointness, never whether a
  classifier picks the right category for a real request.

---

## The multi-agent orchestrator (`Multi_Agent_Implementation/`)

A heavier but more thorough answer to the same problem: instead of
narrowing one model's tool list, delegate to one of 7 independent worker
agents, each with its own session, system prompt, and fixed tool subset —
the confusable tool is never loaded into the worker's context at all,
not just deprioritized.

```
Multi_Agent_Implementation/
├── Agent_Orchestrator.py   # lead agent: 7 delegate_to_<worker>_agent tools
├── worker_agents.py        # WORKER_AGENTS registry + run_worker_agent()
└── _agent_utils.py         # scoped variant of Implementation/Anthropic's run_agent()
```

`WORKER_AGENTS` (`worker_agents.py`) has one entry per category above —
the worker keys *are* the `TOOL_CATEGORY` values — and each worker's
`"tools"` list is **derived** from `TOOL_CATEGORY`, not hand-duplicated:

```python
def _tools_for(category: str) -> List[str]:
    return sorted(name for name, cat in TOOL_CATEGORY.items() if cat == category)

WORKER_AGENTS = {
    "screener": {"tools": _tools_for("screener"), ...},
    "backtest_execution": {"tools": _tools_for("backtest_execution"), ...},
    ...
}
```

A tool's category only ever needs to be correct in `TOOL_CATEGORY` itself
to show up correctly in both the router's classification prompt and this
worker registry — there is no second list that can drift out of sync.

The orchestrator's own "tools" are 7 hand-authored
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

`tests/test_multi_agent_tool_coverage.py` — pure data validation, no API
key or network required:
- every worker has tools, a system prompt, a label, a description
- the union of every worker's tools equals the full library tool set,
  checked both directions (nothing missing, nothing referencing a
  nonexistent tool)
- no tool is assigned to more than one worker
- `run_sma_backtest`/`run_custom_signal_backtest` land in different workers
  (the original confusion-pair regression test)
- `run_sma_backtest`/`run_backtest_optimization` land in different workers
  (`backtest_execution` vs `backtest_validation` — the same guarantee one
  level narrower)
- exactly 7 workers exist, by name — a regression guard for the
  execution/validation split derived from the actual registry rather than
  a magic number that could itself drift

There is currently no test that runs the orchestrator's delegation loop
itself against a live/mocked API, or asserts it picks the *correct* worker
for a given request (unlike the router's routing-accuracy eval above) — a
known gap, not a hidden one.

---

## Router vs. multi-agent orchestrator: when to use which

| | Router | Multi-agent orchestrator |
|---|---|---|
| Cost | One cheap classification call, then the normal agent loop | A full independent agent session per delegated worker |
| Setup | `route_request()` + `categories=` param on an existing `run_agent()` | A lead agent, 7 worker sessions, its own delegation loop |
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
