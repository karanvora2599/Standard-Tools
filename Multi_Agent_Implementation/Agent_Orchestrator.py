"""
Multi-Agent Orchestrator — Anthropic / Claude Haiku.

Demonstrates an orchestrator-workers architecture on top of Standard Quant
Tools' 27 agent tools: instead of one agent choosing among all 27 tools
every turn, a top-level orchestrator delegates each sub-task to a specialist
worker agent (see worker_agents.py) that only ever sees the small subset of
tools relevant to its own workflow.

The orchestrator's own "tools" are not the library's 27 — they are six
hand-authored delegate_to_<worker>_agent(request) tools, one per worker.
Calling one spins up a fresh, independently-scoped run_agent() session for
that worker and returns its final answer as the tool result.

Why this helps over a single flat 27-tool agent: a worker that was never
given run_sma_backtest cannot mistakenly call it instead of
run_custom_signal_backtest — the confusable tool simply isn't in front of
the model. Smaller tool lists also mean shorter, more focused system
prompts per turn, which measurably improves function-calling accuracy on
cheaper models like Haiku.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import textwrap
from typing import Any, cast

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam

from worker_agents import WORKER_AGENTS, run_worker_agent
from _agent_utils import setup_logging, _header, _section, _log, _pretty_json

# ── Configuration ──────────────────────────────────────────────────
ANTHROPIC_API_KEY  = ""   # Replace with your key
ORCHESTRATOR_MODEL = "claude-haiku-4-5"
WORKER_MODEL       = "claude-haiku-4-5"

# ── Orchestrator's own tools: one delegate call per worker ──────────

_DELEGATE_TOOLS: list[ToolParam] = [
    ToolParam(
        name=f"delegate_to_{key}_agent",
        description=(
            f"Delegate a sub-task to the {worker['label']}. "
            f"{worker['description']} "
            f"This agent can ONLY do this — it has no other tools."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": (
                        "The complete natural-language sub-task for this agent. "
                        "Include every concrete detail it needs (tickers, dates, "
                        "parameters, signal values, account size, etc.) — the "
                        "worker agent sees only this string, not the original "
                        "user request or any other agent's output unless you "
                        "include it here explicitly."
                    ),
                },
            },
            "required": ["request"],
        },
    )
    for key, worker in WORKER_AGENTS.items()
]

_WORKER_KEY_BY_TOOL = {f"delegate_to_{key}_agent": key for key in WORKER_AGENTS}

ORCHESTRATOR_SYSTEM_PROMPT = f"""You are the lead quantitative analyst coordinating a team of six
specialist agents. You do NOT have direct access to any backtesting, screening,
or analysis tools yourself — you can only delegate to your specialists and
synthesise their answers into one coherent final response for the user.

Your team:
{chr(10).join(f"- delegate_to_{key}_agent — {w['description']}" for key, w in WORKER_AGENTS.items())}

Rules:
- Delegate one focused sub-task per call. Give each specialist everything it
  needs in the `request` string — it cannot see the original user request or
  any other specialist's output unless you copy the relevant numbers into
  your delegate call.
- Chain specialists when the task requires it (e.g. screen -> backtest -> size).
  Use one specialist's reported output as an input to the next specialist's
  request when the workflow requires it.
- Never invent numbers a specialist would need to compute (prices, Sharpe
  ratios, signal values, etc.) — always get them from a real delegate call.
- When you have everything you need, write one final synthesis for the user:
  state the exact numbers each specialist reported and a clear recommendation.
- If a request doesn't need multiple specialists, delegate to just the one
  that's relevant — do not call agents you don't need.
"""


def run_orchestrator(user_request: str, max_iterations: int = 8) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    _header("ORCHESTRATOR SESSION STARTED")
    _log("Orchestrator model", ORCHESTRATOR_MODEL)
    _log("Worker model",       WORKER_MODEL)
    _log("Workers available",  ", ".join(WORKER_AGENTS))
    _section("USER REQUEST")
    print(textwrap.fill(user_request, width=68, initial_indent="  ", subsequent_indent="  "))

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": "user", "content": user_request})
    ]
    session_start = time.perf_counter()
    iteration = 0
    accumulated_text: list[str] = []

    for iteration in range(1, max_iterations + 1):
        _header(f"ORCHESTRATOR ITERATION {iteration}")

        response: Message = cast(
            Message,
            client.messages.create(
                model=ORCHESTRATOR_MODEL,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=_DELEGATE_TOOLS,
                messages=messages,
            ),
        )

        _section(f"ORCHESTRATOR OUTPUT  ({len(response.content)} block(s))")
        for block in response.content:
            if block.type == "text":
                print(textwrap.fill(
                    block.text, width=68,
                    initial_indent="    ", subsequent_indent="    ",
                ))
            elif block.type == "tool_use":
                _log(f"  Delegating → {block.name}", indent=4)
                print(_pretty_json(block.input, indent=6))

        for block in response.content:
            if block.type == "text" and block.text:
                accumulated_text.append(block.text)  # type: ignore[attr-defined]

        messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

        if response.stop_reason == "end_turn":
            _section("ORCHESTRATOR FINISHED  (end_turn)")
            break

        if response.stop_reason != "tool_use":
            _log(f"Unexpected stop reason '{response.stop_reason}' — stopping")
            break

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            worker_key = _WORKER_KEY_BY_TOOL.get(block.name)
            if worker_key is None:
                content, is_error = f"Unknown delegate tool '{block.name}'", True
            else:
                sub_request = block.input.get("request", "")
                try:
                    content, is_error = run_worker_agent(
                        worker_key, sub_request,
                        api_key=ANTHROPIC_API_KEY, model=WORKER_MODEL,
                    ), False
                except Exception as exc:
                    content, is_error = f"Worker '{worker_key}' failed: {exc}", True

            entry = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps({"result": content}),
            }
            if is_error:
                entry["is_error"] = True
            tool_results.append(entry)

        accumulated_text.clear()
        messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

    total_elapsed = time.perf_counter() - session_start
    _header("ORCHESTRATOR SESSION SUMMARY")
    _log("Iterations used", f"{iteration} / {max_iterations}")
    _log("Total wall time", f"{total_elapsed:.2f}s")

    return "".join(accumulated_text) or "Max iterations reached."


if __name__ == "__main__":
    log_file = setup_logging("multi_agent_orchestrator")

    _header("Multi-Agent Orchestrator — Claude Haiku")
    _log("Log file", str(log_file))

    # Deliberately spans three different workers in sequence — screener,
    # backtest, and portfolio_risk — to demonstrate real delegation, not
    # just a single tool call routed to a single worker.
    request = (
        "Screen these mega-cap tech stocks for forward PE under 35 and RSI "
        "under 55: AAPL, MSFT, GOOGL, NVDA, META, AMZN. "
        "For any that pass, run a regime-adaptive backtest from 2022-01-01 "
        "to 2024-01-01 and tell me which one has the best Sharpe ratio. "
        "Then size a position in that best performer for a $100,000 account "
        "risking 1% per trade."
    )

    result = run_orchestrator(request)

    _header("FINAL REPORT")
    print(result)
