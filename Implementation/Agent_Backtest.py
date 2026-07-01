"""
Agentic backtest using Claude Haiku.
The agent autonomously selects strategies, runs backtests, and summarizes findings.
"""

import datetime
import json
import logging
import time
import textwrap
from pathlib import Path
from typing import Any, cast
from anthropic import Anthropic
from anthropic.types import Message, MessageParam, ToolParam
from standard_quant_tools.agent.tools import get_agent_tools, dispatch

# ── Logging setup ──────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOGS_DIR.mkdir(exist_ok=True)
_run_ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
_log_file = _LOGS_DIR / f"agent_backtest_{_run_ts}.log"

_fmt_console = logging.Formatter("  %(levelname)-7s  %(name)s  %(message)s")
_fmt_file    = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt_console)
_console_handler.setLevel(logging.DEBUG)

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(_fmt_file)
_file_handler.setLevel(logging.DEBUG)

_lib_logger = logging.getLogger("standard_quant_tools")
_lib_logger.setLevel(logging.DEBUG)
_lib_logger.addHandler(_console_handler)
_lib_logger.addHandler(_file_handler)

ANTHROPIC_API_KEY = ""  # Replace with your key

SYSTEM_PROMPT = """You are a quantitative analyst with access to a suite of backtesting and analysis tools.

Your goal is to:
1. Analyse the requested stock(s) using the available tools
2. Run multiple backtests to compare strategies
3. Evaluate risk metrics
4. Summarize your findings with a clear recommendation

Always start with technical analysis to understand the current market regime, then select and run
the most appropriate strategies. Compare results against buy-and-hold and provide a final recommendation.
"""

# ── Logging helpers ────────────────────────────────────────────────

_DIVIDER      = "═" * 70
_THIN_DIVIDER = "─" * 70

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
    if value:
        print(f"{prefix}{label}: {value}")
    else:
        print(f"{prefix}{label}")

def _pretty_json(data: Any, indent: int = 4, max_len: int = 2000) -> str:
    raw = json.dumps(data, indent=2, default=str)
    if len(raw) > max_len:
        raw = raw[:max_len] + f"\n  ... [truncated — {len(raw)} chars total]"
    lines = raw.splitlines()
    pad = " " * indent
    return "\n".join(pad + line for line in lines)


# ── Tool format conversion ─────────────────────────────────────────

def _to_anthropic_tools(openai_tools: list[dict[str, Any]]) -> list[ToolParam]:
    """Convert OpenAI-format tool defs returned by get_agent_tools() to Anthropic native format."""
    return [
        ToolParam(
            name=t["function"]["name"],
            description=t["function"]["description"],
            input_schema=t["function"]["parameters"],
        )
        for t in openai_tools
    ]


# ── Agent loop ─────────────────────────────────────────────────────

def run_agent_backtest(user_request: str, max_iterations: int = 10) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = _to_anthropic_tools(get_agent_tools())

    _header("AGENT SESSION STARTED")
    _log("Model",          "claude-haiku-4-5")
    _log("Max iterations", str(max_iterations))
    _log("Tools loaded",   str(len(tools)))
    _log("Tool names",     ", ".join(t["name"] for t in tools))
    _section("USER REQUEST")
    print(textwrap.fill(user_request, width=68, initial_indent="  ", subsequent_indent="  "))

    messages: list[MessageParam] = [cast(MessageParam, {"role": "user", "content": user_request})]
    session_start = time.perf_counter()
    total_input_tokens = 0
    total_output_tokens = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        _header(f"ITERATION {iteration}")
        iter_start = time.perf_counter()

        _log("Sending request to API ...")
        response: Message = cast(
            Message,
            client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            ),
        )

        elapsed = time.perf_counter() - iter_start
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        _section("API RESPONSE METADATA")
        _log("Stop reason",    response.stop_reason or "—")
        _log("Input tokens",   str(response.usage.input_tokens))
        _log("Output tokens",  str(response.usage.output_tokens))
        _log("Latency",        f"{elapsed:.2f}s")

        # Log each content block the model returned
        _section(f"MODEL OUTPUT  ({len(response.content)} block(s))")
        for i, block in enumerate(response.content):
            print(f"\n  [Block {i+1}]  type={block.type}")
            if block.type == "text":
                wrapped = textwrap.fill(
                    block.text, width=68,
                    initial_indent="    ", subsequent_indent="    ",
                )
                print(wrapped)
            elif block.type == "tool_use":
                _log(f"  Tool call   → {block.name}",  indent=4)
                _log(f"  Call ID     → {block.id}",    indent=4)
                _log("  Arguments",                    indent=4)
                print(_pretty_json(block.input, indent=6))

        messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

        # ── Done ──────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            _section("AGENT FINISHED  (end_turn)")
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text  # type: ignore[attr-defined]
            return "Analysis complete."

        if response.stop_reason != "tool_use":
            _log(f"Unexpected stop reason '{response.stop_reason}' — breaking loop")
            break

        # ── Execute tool calls ─────────────────────────────────────
        _section("TOOL EXECUTION")
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\n  ┌─ {block.name}")
            print(f"  │  id : {block.id}")
            _log("│  args", indent=3)
            print(_pretty_json(block.input, indent=5))

            t_start = time.perf_counter()
            try:
                result = dispatch(block.name, block.input)
                t_ms = (time.perf_counter() - t_start) * 1000
                print(f"  │  ✓  completed in {t_ms:.0f}ms")
                _log("│  result", indent=3)
                print(_pretty_json(result, indent=5))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
            except Exception as e:
                t_ms = (time.perf_counter() - t_start) * 1000
                print(f"  │  ✗  FAILED in {t_ms:.0f}ms — {e}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                })
            print("  └" + "─" * 50)

        messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

    # ── Session summary ────────────────────────────────────────────
    total_elapsed = time.perf_counter() - session_start
    _header("SESSION SUMMARY")
    _log("Iterations used",    f"{iteration} / {max_iterations}")
    _log("Total input tokens",  str(total_input_tokens))
    _log("Total output tokens", str(total_output_tokens))
    _log("Total tokens",        str(total_input_tokens + total_output_tokens))
    _log("Total wall time",     f"{total_elapsed:.2f}s")

    return "Max iterations reached."


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    request = (
        "Analyse AAPL from 2023-01-01 to 2024-12-31. "
        "Run a regime-adaptive backtest, compare all four strategies, "
        "and provide a risk analysis. Which strategy would you recommend and why?"
    )

    _header("Agentic Backtest — Claude Haiku")
    _log(f"Log file", str(_log_file))
    print(f"  Request: {request}\n")

    result = run_agent_backtest(request)

    _header("FINAL ANALYSIS")
    print(result)
