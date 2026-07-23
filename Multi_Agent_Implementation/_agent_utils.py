"""
Shared utilities for the multi-agent example (Anthropic).

This is a scoped variant of Implementation/Anthropic/_agent_utils.py: the
core run_agent() loop is identical, but it accepts an optional tool_names
filter so a worker agent can be given only a small, non-overlapping subset
of the library's 29 tools instead of all of them. Shrinking the tool list
per agent is the actual fix for tool-selection confusion between similar
tools (e.g. run_sma_backtest vs run_custom_signal_backtest) — a worker
that was never given the other tool cannot call it.

Provides:
  - setup_logging(name)                    -> configure lib logger + per-run file handler
  - _to_anthropic_tools()                   -> convert OpenAI-format tools to Anthropic format
  - _header / _section / _log / _pretty_json -> console formatting helpers
  - run_agent()                             -> the core agentic loop (Claude + tool dispatch),
                                                optionally scoped to tool_names
"""

import datetime
import json
import logging
import time
import textwrap
from pathlib import Path
from typing import Any, List, Optional, cast

from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam

from standard_quant_tools.agent.tools import get_agent_tools, dispatch

# ── Constants ──────────────────────────────────────────────────────
_LOGS_DIR     = Path(__file__).resolve().parent.parent / "logs"
_DIVIDER      = "═" * 70
_THIN_DIVIDER = "─" * 70

_fmt_console = logging.Formatter("  %(levelname)-7s  %(name)s  %(message)s")
_fmt_file    = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def setup_logging(name: str) -> Path:
    """
    Attach a per-run FileHandler + StreamHandler to the standard_quant_tools
    logger hierarchy. Returns the path of the log file created.
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = _LOGS_DIR / f"{name}_{ts}.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_fmt_file)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(_fmt_console)
    sh.setLevel(logging.DEBUG)

    lib = logging.getLogger("standard_quant_tools")
    lib.setLevel(logging.DEBUG)
    lib.addHandler(fh)
    lib.addHandler(sh)

    return log_file


# ── Console helpers ─────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)

def _section(title: str) -> None:
    print(f"\n{_THIN_DIVIDER}")
    print(f"  {title}")
    print(_THIN_DIVIDER)

def _log(label: str, value: str = "", indent: int = 2) -> None:
    prefix = " " * indent
    print(f"{prefix}{label}: {value}" if value else f"{prefix}{label}")

def _pretty_json(data: Any, indent: int = 4, max_len: int = 2000) -> str:
    raw = json.dumps(data, indent=2, default=str)
    if len(raw) > max_len:
        raw = raw[:max_len] + f"\n  ... [truncated — {len(raw)} chars total]"
    pad = " " * indent
    return "\n".join(pad + line for line in raw.splitlines())


# ── Tool format conversion / filtering ───────────────────────────────

def _to_anthropic_tools(openai_tools: list[dict[str, Any]]) -> list[ToolParam]:
    """Convert get_agent_tools() OpenAI format → Anthropic native format."""
    return [
        ToolParam(
            name=t["function"]["name"],
            description=t["function"]["description"],
            input_schema=t["function"]["parameters"],
        )
        for t in openai_tools
    ]


def _scoped_tools(tool_names: Optional[List[str]]) -> list[ToolParam]:
    """
    Full 29-tool registry by default; filtered to tool_names when given.
    A worker agent should always pass its own fixed subset here.
    """
    all_tools = get_agent_tools()
    if tool_names is not None:
        wanted = set(tool_names)
        all_tools = [t for t in all_tools if t["function"]["name"] in wanted]
        missing = wanted - {t["function"]["name"] for t in all_tools}
        if missing:
            raise ValueError(f"tool_names references unknown tool(s): {sorted(missing)}")
    return _to_anthropic_tools(all_tools)


# ── Core agent loop ─────────────────────────────────────────────────

def run_agent(
    system_prompt: str,
    user_request: str,
    api_key: str,
    model: str = "claude-haiku-4-5",
    max_iterations: int = 15,
    max_tokens: int = 8096,
    tool_names: Optional[List[str]] = None,
) -> str:
    """
    Run the agentic loop: send user_request to Claude, execute any tool calls
    via dispatch(), feed results back, and repeat until end_turn or exhausted.

    tool_names: if given, the model only ever sees this subset of the 29
    registered tools (all of them if omitted). This is how a worker agent
    stays scoped to its own workflow.

    Returns the final text response from the model.
    """
    client = Anthropic(api_key=api_key)
    tools  = _scoped_tools(tool_names)

    _header("AGENT SESSION STARTED")
    _log("Model",          model)
    _log("Max tokens",     str(max_tokens))
    _log("Max iterations", str(max_iterations))
    _log("Tools loaded",   str(len(tools)))
    _log("Tool names",     ", ".join(t["name"] for t in tools))
    _section("USER REQUEST")
    print(textwrap.fill(user_request, width=68, initial_indent="  ", subsequent_indent="  "))

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": "user", "content": user_request})
    ]
    session_start        = time.perf_counter()
    total_input_tokens   = 0
    total_output_tokens  = 0
    iteration            = 0
    accumulated_text: list[str] = []

    for iteration in range(1, max_iterations + 1):
        _header(f"ITERATION {iteration}")
        iter_start = time.perf_counter()

        _log("Sending request to Claude ...")
        response: Message = cast(
            Message,
            client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                tools=tools,
                messages=messages,
            ),
        )

        elapsed              = time.perf_counter() - iter_start
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        _section("API RESPONSE METADATA")
        _log("Stop reason",   response.stop_reason or "—")
        _log("Input tokens",  str(response.usage.input_tokens))
        _log("Output tokens", str(response.usage.output_tokens))
        _log("Latency",       f"{elapsed:.2f}s")

        _section(f"MODEL OUTPUT  ({len(response.content)} block(s))")
        for i, block in enumerate(response.content):
            print(f"\n  [Block {i+1}]  type={block.type}")
            if block.type == "text":
                print(textwrap.fill(
                    block.text, width=68,
                    initial_indent="    ", subsequent_indent="    ",
                ))
            elif block.type == "tool_use":
                _log(f"  Tool call → {block.name}", indent=4)
                _log(f"  Call ID   → {block.id}",   indent=4)
                print(_pretty_json(block.input, indent=6))

        for block in response.content:
            if block.type == "text" and block.text:
                accumulated_text.append(block.text)  # type: ignore[attr-defined]

        messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

        if response.stop_reason == "end_turn":
            _section("AGENT FINISHED  (end_turn)")
            return "".join(accumulated_text) or "Analysis complete."

        if response.stop_reason == "max_tokens":
            has_text = any(b.type == "text" for b in response.content)
            has_tool = any(b.type == "tool_use" for b in response.content)

            if has_text and not has_tool:
                _log("max_tokens hit mid-text — sending continuation prompt ...")
                messages.append(cast(MessageParam, {
                    "role": "user",
                    "content": "Please continue your response from exactly where you left off. Do not repeat anything already written.",
                }))
                continue

            _log("max_tokens with tool_use content — cannot continue cleanly, stopping")
            break

        if response.stop_reason != "tool_use":
            _log(f"Unexpected stop reason '{response.stop_reason}' — breaking loop")
            break

        _section("TOOL EXECUTION")
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\n  ┌─ {block.name}")
            print(f"  │  id : {block.id}")
            print(_pretty_json(block.input, indent=5))

            t0 = time.perf_counter()
            try:
                result = dispatch(block.name, block.input)
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✓  completed in {ms:.0f}ms")
                print(_pretty_json(result, indent=5))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  FAILED in {ms:.0f}ms — {exc}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {exc}",
                    "is_error": True,
                })
            print("  └" + "─" * 50)

        accumulated_text.clear()
        messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

    total_elapsed = time.perf_counter() - session_start
    _header("SESSION SUMMARY")
    _log("Iterations used",     f"{iteration} / {max_iterations}")
    _log("Total input tokens",  str(total_input_tokens))
    _log("Total output tokens", str(total_output_tokens))
    _log("Total tokens",        str(total_input_tokens + total_output_tokens))
    _log("Total wall time",     f"{total_elapsed:.2f}s")

    return "".join(accumulated_text) or "Max iterations reached."
