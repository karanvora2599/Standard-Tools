"""
Shared utilities for all agentic scripts.

Provides:
  - setup_logging(name)          → configure lib logger + per-run file handler
  - _to_anthropic_tools()        → convert OpenAI-format tools to Anthropic format
  - _header / _section / _log / _pretty_json → console formatting helpers
  - run_agent()                  → the core agentic loop (Claude + tool dispatch)

This is demo/example code, not production agent infrastructure. In
particular it prints full user requests and tool call arguments/results to
stdout and a log file by default (`verbose=True` below) — fine for a local
demo run, but do not reuse this as-is anywhere those payloads could contain
data you don't want on disk or on a shared console.
"""

import concurrent.futures
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
from standard_quant_tools.modeling.agent import (
    get_modeling_tools,
    modeling_dispatch,
)

# ── Constants ──────────────────────────────────────────────────────
_LOGS_DIR     = Path(__file__).resolve().parent.parent / "logs"
_DIVIDER      = "═" * 70
_THIN_DIVIDER = "─" * 70

_fmt_console = logging.Formatter("  %(levelname)-7s  %(name)s  %(message)s")
_fmt_file    = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Marker attribute so repeated setup_logging() calls in the same process
# replace the previous run's handlers instead of piling up duplicates that
# would print every log line once per prior call.
_HANDLER_MARKER = "_sqt_example_handler"


def setup_logging(name: str) -> Path:
    """
    Attach a per-run FileHandler + StreamHandler to the standard_quant_tools
    logger hierarchy.  Returns the path of the log file created.

    name: short identifier used in the filename, e.g. "agent_portfolio"
    """
    _LOGS_DIR.mkdir(exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = _LOGS_DIR / f"{name}_{ts}.log"

    lib = logging.getLogger("standard_quant_tools")
    for h in list(lib.handlers):
        if getattr(h, _HANDLER_MARKER, False):
            lib.removeHandler(h)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(_fmt_file)
    fh.setLevel(logging.DEBUG)
    setattr(fh, _HANDLER_MARKER, True)

    sh = logging.StreamHandler()
    sh.setFormatter(_fmt_console)
    sh.setLevel(logging.DEBUG)
    setattr(sh, _HANDLER_MARKER, True)

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


# ── Tool format conversion ──────────────────────────────────────────

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


# ── Core agent loop ─────────────────────────────────────────────────

# ── The two tool registries ──────────────────────────────────────────
#
# This library exposes TWO agent-tool registries and deliberately does not
# merge them (see Documentation/15_modeling.md):
#
#   "analysis"  standard_quant_tools.agent           46 tools, 7 categories
#   "modeling"  standard_quant_tools.modeling.agent   8 tools, one pipeline
#
# Their shapes are identical -- same OpenAI-format schema, same
# dispatch(tool_name, arguments) signature -- which is exactly why keeping
# them apart has to be deliberate rather than incidental. A caller names ONE
# registry and gets that registry's schemas and its dispatch function
# together. Taking the tool list from one and the dispatcher from the other
# would fail at the first tool call, with an "unknown tool" error that
# points at the model's choice rather than at this wiring.
_REGISTRIES = {
    "analysis": (get_agent_tools, dispatch),
    "modeling": (get_modeling_tools, modeling_dispatch),
}


def _registry(registry: str):
    """The (load_tools, dispatch) pair for a registry name."""
    if registry not in _REGISTRIES:
        raise ValueError(
            f"Unknown registry {registry!r}; expected one of {sorted(_REGISTRIES)}."
        )
    return _REGISTRIES[registry]


def _registry_tools(registry: str, categories=None):
    """
    A registry's tool schemas in OpenAI format, optionally narrowed to a set
    of router categories.

    Category routing is an ANALYSIS-registry idea: it exists because 46
    similarly-shaped tools cause selection ambiguity. The modeling runtime is
    eight tools in one ordered pipeline with no taxonomy to route across, so
    a category filter here is a caller mistake rather than a harmless no-op
    -- accepting it silently would hide that the request was never narrowed.
    """
    load_tools, _ = _registry(registry)
    if registry != "analysis":
        if categories:
            raise ValueError(
                f"categories={categories!r} was given for the {registry!r} "
                "registry, which has no category taxonomy to route across. "
                "Only the analysis registry can be narrowed this way."
            )
        return load_tools()
    return load_tools(categories=categories)


def run_agent(
    system_prompt: str,
    user_request: str,
    api_key: str,
    model: str = "claude-haiku-4-5",
    max_iterations: int = 15,
    max_tokens: int = 8096,
    request_timeout_s: float = 60.0,
    tool_timeout_s: float = 120.0,
    verbose: bool = True,
    registry: str = "analysis",
) -> str:
    """
    Run the agentic loop: send user_request to Claude, execute any tool calls
    via dispatch(), feed results back, and repeat until end_turn or exhausted.

    max_tokens applies to each individual API call (not the whole session).
    Haiku 4.5 supports up to 8096 output tokens per call; Opus 4.8 supports 32000.

    request_timeout_s bounds each Anthropic API call; tool_timeout_s bounds
    each individual dispatch() call (run in a worker thread so a hung tool
    can't block the loop forever). verbose=False suppresses printing full
    user/tool payloads (see the module docstring) while keeping status-line
    output.

    registry: which tool registry to load -- "analysis" for the 46-tool
    analysis/backtest surface (the default, and this function's behavior
    before the parameter existed), or "modeling" for the separate 8-tool
    model-construction pipeline. The registry also decides which dispatch
    function executes the calls, so the two never mix.

    Returns the final text response from the model.
    """
    client = Anthropic(api_key=api_key, timeout=request_timeout_s)
    tools  = _to_anthropic_tools(_registry_tools(registry))
    _, tool_dispatch = _registry(registry)

    _header("AGENT SESSION STARTED")
    _log("Model",          model)
    _log("Max tokens",     str(max_tokens))
    _log("Max iterations", str(max_iterations))
    _log("Tools loaded",   str(len(tools)))
    _log("Tool names",     ", ".join(t["name"] for t in tools))
    if verbose:
        _section("USER REQUEST")
        print(textwrap.fill(user_request, width=68, initial_indent="  ", subsequent_indent="  "))

    messages: list[MessageParam] = [
        cast(MessageParam, {"role": "user", "content": user_request})
    ]
    session_start        = time.perf_counter()
    total_input_tokens   = 0
    total_output_tokens  = 0
    iteration            = 0
    # Accumulate text chunks across continuation turns so the final
    # answer is always the complete uninterrupted text.
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
                if verbose:
                    print(textwrap.fill(
                        block.text, width=68,
                        initial_indent="    ", subsequent_indent="    ",
                    ))
            elif block.type == "tool_use":
                _log(f"  Tool call → {block.name}", indent=4)
                _log(f"  Call ID   → {block.id}",   indent=4)
                if verbose:
                    print(_pretty_json(block.input, indent=6))

        # Collect any text produced this turn
        for block in response.content:
            if block.type == "text" and block.text:
                accumulated_text.append(block.text)  # type: ignore[attr-defined]

        messages.append(cast(MessageParam, {"role": "assistant", "content": response.content}))

        # ── Finished normally ────────────────────────────────────────
        if response.stop_reason == "end_turn":
            _section("AGENT FINISHED  (end_turn)")
            return "".join(accumulated_text) or "Analysis complete."

        # ── Mid-text truncation: ask Claude to continue ──────────────
        if response.stop_reason == "max_tokens":
            has_text = any(b.type == "text" for b in response.content)
            has_tool = any(b.type == "tool_use" for b in response.content)

            if has_text and not has_tool:
                # Claude was writing its final answer and ran out of tokens.
                # Append a continuation prompt so it picks up exactly where it stopped.
                _log("max_tokens hit mid-text — sending continuation prompt ...")
                messages.append(cast(MessageParam, {
                    "role": "user",
                    "content": "Please continue your response from exactly where you left off. Do not repeat anything already written.",
                }))
                continue  # next iteration will get the rest

            # If tool_use blocks are present alongside max_tokens the API
            # state is ambiguous — safer to stop than corrupt the tool loop.
            _log("max_tokens with tool_use content — cannot continue cleanly, stopping")
            break

        if response.stop_reason != "tool_use":
            _log(f"Unexpected stop reason '{response.stop_reason}' — breaking loop")
            break

        # ── Execute tool calls ───────────────────────────────────────
        _section("TOOL EXECUTION")
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\n  ┌─ {block.name}")
            print(f"  │  id : {block.id}")
            if verbose:
                print(_pretty_json(block.input, indent=5))

            t0 = time.perf_counter()
            # Not `with ThreadPoolExecutor() as ex:` — that context manager's
            # __exit__ calls shutdown(wait=True), which blocks until the
            # submitted call finishes regardless of the result() timeout
            # below, defeating the entire point of bounding a hung tool
            # call. shutdown(wait=False) in `finally` lets this loop move on
            # immediately while the orphaned thread runs out on its own.
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                result = ex.submit(
                    tool_dispatch, block.name, block.input
                ).result(
                    timeout=tool_timeout_s
                )
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✓  completed in {ms:.0f}ms")
                if verbose:
                    print(_pretty_json(result, indent=5))
                # allow_nan=False: fail loudly here rather than silently
                # emitting non-standard Infinity/NaN JSON tokens to the API
                # if a non-finite float ever slipped past dispatch()'s own
                # sanitization.
                content = json.dumps(result, default=str, allow_nan=False)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
            except concurrent.futures.TimeoutError:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  TIMED OUT after {ms:.0f}ms (limit {tool_timeout_s}s)")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: tool call timed out after {tool_timeout_s}s",
                    "is_error": True,
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
            finally:
                ex.shutdown(wait=False)
            print("  └" + "─" * 50)

        messages.append(cast(MessageParam, {"role": "user", "content": tool_results}))

    # ── Session summary ──────────────────────────────────────────────
    total_elapsed = time.perf_counter() - session_start
    _header("SESSION SUMMARY")
    _log("Iterations used",     f"{iteration} / {max_iterations}")
    _log("Total input tokens",  str(total_input_tokens))
    _log("Total output tokens", str(total_output_tokens))
    _log("Total tokens",        str(total_input_tokens + total_output_tokens))
    _log("Total wall time",     f"{total_elapsed:.2f}s")

    return "".join(accumulated_text) or "Max iterations reached."
