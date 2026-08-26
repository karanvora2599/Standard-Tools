"""
Shared utilities for all agentic scripts — OpenAI provider.

Provides:
  - setup_logging(name)   → configure lib logger + per-run file handler
  - _header / _section / _log / _pretty_json → console formatting helpers
  - run_agent()           → the core agentic loop (GPT + tool dispatch)

Note: get_agent_tools() already returns the OpenAI tool format, so no
conversion step is needed unlike the Anthropic provider.

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
import textwrap
import time
from pathlib import Path
from typing import Any, List, Optional

from openai import OpenAI

from standard_quant_tools.agent.router import (
    TOOL_CATEGORIES,
    build_router_prompt,
    parse_router_response,
)
from standard_quant_tools.agent.tools import dispatch, get_agent_tools
from standard_quant_tools.modeling.agent import (
    get_modeling_tools,
    modeling_dispatch,
)

# ── Constants ──────────────────────────────────────────────────────
_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_DIVIDER = "═" * 70
_THIN_DIVIDER = "─" * 70

_fmt_console = logging.Formatter("  %(levelname)-7s  %(name)s  %(message)s")
_fmt_file = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Marker attribute so repeated setup_logging() calls in the same process
# (e.g. re-running a script's __main__ from a REPL, or importing two of
# these example scripts into one session) replace the previous run's
# handlers instead of piling up duplicates that would print every log
# line once per prior call.
_HANDLER_MARKER = "_sqt_example_handler"


def setup_logging(name: str) -> Path:
    """Attach a per-run FileHandler + StreamHandler to the standard_quant_tools logger."""
    _LOGS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
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


# ── Router glue (provider-specific: which client, which cheap model) ────


def route_request(
    request: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> List[str]:
    """
    One cheap classification call: ask `model` which 1-2 tool categories
    (see agent/router.py) are relevant to `request`, and return that list
    of category keys. Fails open (returns every category, i.e. no
    narrowing) on any API error or unparseable response — see
    agent/router.py's module docstring for why that's the right default.
    """
    raw_text = ""
    try:
        client = OpenAI(api_key=api_key, timeout=15.0)
        response = client.chat.completions.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": build_router_prompt(request, TOOL_CATEGORIES),
                }
            ],
            stream=False,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:
        _log(
            f"route_request: classification call failed ({exc}) — routing to all categories"
        )

    categories = parse_router_response(raw_text, TOOL_CATEGORIES)
    _log("Routed categories", ", ".join(categories))
    return categories


# ── Core agent loop ─────────────────────────────────────────────────


# ── The tool registries ──────────────────────────────────────────────
#
# This library exposes THREE agent-tool registries and deliberately does
# not merge them (see Documentation/15_modeling.md):
#
#   "analysis"     standard_quant_tools.agent          132 tools, 11 categories
#   "modeling"     standard_quant_tools.modeling.agent  16 tools, one pipeline
#   "feature_lab"  standard_quant_tools.modeling.agent   9 tools, exploratory
#
# 132 + 16 + 9 is the 157-tool whole surface. The analysis surface is itself
# divided into six RUNTIMES -- research, backtest, portfolio, microstructure,
# derivatives, meta -- and naming one of those instead gives a dispatch table
# that refuses anything it does not own. That is the
# difference between narrowing what a model is shown and enforcing what it
# can run. See Documentation/19_runtimes.md.
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
    """
    The (load_tools, dispatch) pair for a registry OR a runtime name.

    NAMING A RUNTIME IS THE SAFER CHOICE, and usually the right one. The
    two entries in _REGISTRIES above are the whole-surface views:
    "analysis" hands back `dispatch`, which knows every tool in the library
    regardless of which ones were advertised to the model. That is fine for
    a script that means to offer everything, and wrong for one that
    narrowed the list -- a model that hallucinates a tool it was never
    shown gets a RESULT rather than an error, and a wrong guess that
    succeeds is the worst possible feedback.

    Naming "research", "backtest", "portfolio" or "meta" instead returns
    that runtime's own dispatch table, which holds only its tools and
    refuses the rest by name, saying which runtime actually owns them. See
    Documentation/19_runtimes.md.

    A workflow that genuinely spans runtimes joins them explicitly with a
    "+" -- "research+backtest" -- so the widening is visible in the code
    that asked for it rather than being the silent default.
    """
    if registry in _REGISTRIES:
        return _REGISTRIES[registry]

    from standard_quant_tools.agent.runtimes import all_runtimes, combine, resolve

    names = [part.strip() for part in registry.split("+") if part.strip()]
    known = set(all_runtimes())
    if names and set(names) <= known:
        runtime = resolve(names[0]) if len(names) == 1 else combine(names)
        return (runtime.get_tools, runtime.dispatch)

    raise ValueError(
        f"Unknown registry or runtime {registry!r}; expected one of "
        f"{sorted(_REGISTRIES)} (whole-surface views) or "
        f"{sorted(known)} (scoped runtimes, joinable with '+')."
    )


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
    if registry == "modeling":
        if categories:
            raise ValueError(
                f"categories={categories!r} was given for the {registry!r} "
                "runtime, which has no category taxonomy to route across. "
                "It is one ordered pipeline, so there is nothing to narrow."
            )
        return load_tools()
    # "analysis" and every scoped runtime take a category filter. Inside a
    # runtime the filter can only narrow further -- a category that runtime
    # does not own contributes nothing rather than reaching past it.
    return load_tools(categories=categories)


def run_agent(
    system_prompt: str,
    user_request: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_iterations: int = 15,
    max_tokens: int = 4096,
    request_timeout_s: float = 60.0,
    tool_timeout_s: float = 120.0,
    verbose: bool = True,
    categories: Optional[List[str]] = None,
    registry: str = "analysis",
) -> str:
    """
    Run the agentic loop: send user_request to GPT, execute any tool calls
    via dispatch(), feed results back, and repeat until stop or exhausted.

    finish_reason values:
      "stop"       — model finished normally
      "tool_calls" — model wants to call tools
      "length"     — hit max_tokens mid-response (continuation logic applies)

    request_timeout_s bounds each OpenAI API call; tool_timeout_s bounds
    each individual dispatch() call (run in a worker thread so a hung tool
    can't block the loop forever — the worker itself is left running to
    completion in the background since there's no safe way to force-kill a
    Python thread, but the loop reports the timeout and moves on rather
    than hanging). verbose=False suppresses printing full user/tool
    payloads (see the module docstring) while keeping status-line output.

    categories: optional list of TOOL_CATEGORY values (see agent/router.py)
    to narrow the tool list to — pass the output of route_request() here.
    None (the default) loads every tool, identical to this function's
    behavior before this parameter existed.

    registry: which tool registry to load -- "analysis" for the 46-tool
    analysis/backtest surface (the default, and this function's behavior
    before the parameter existed), or "modeling" for the separate 8-tool
    model-construction pipeline. The registry also decides which dispatch
    function executes the calls, so the two never mix.

    Returns the final text response from the model.
    """
    client = OpenAI(api_key=api_key, timeout=request_timeout_s)
    # Both registries already return OpenAI tool format, so no conversion.
    tools = _registry_tools(registry, categories)
    _, tool_dispatch = _registry(registry)

    _header("AGENT SESSION STARTED  (OpenAI)")
    _log("Model", model)
    _log("Max tokens", str(max_tokens))
    _log("Max iterations", str(max_iterations))
    _log("Tools loaded", str(len(tools)))
    _log("Tool names", ", ".join(t["function"]["name"] for t in tools))
    if verbose:
        _section("USER REQUEST")
        print(
            textwrap.fill(
                user_request, width=68, initial_indent="  ", subsequent_indent="  "
            )
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    session_start = time.perf_counter()
    total_input_tokens = 0
    total_output_tokens = 0
    iteration = 0
    accumulated_text: list[str] = []

    for iteration in range(1, max_iterations + 1):
        _header(f"ITERATION {iteration}")
        iter_start = time.perf_counter()

        _log("Sending request to OpenAI ...")
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            stream=False,
        )

        elapsed = time.perf_counter() - iter_start
        if response.usage:
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        msg = choice.message

        _section("API RESPONSE METADATA")
        _log("Finish reason", finish_reason or "—")
        _log(
            "Input tokens", str(response.usage.prompt_tokens if response.usage else "—")
        )
        _log(
            "Output tokens",
            str(response.usage.completion_tokens if response.usage else "—"),
        )
        _log("Latency", f"{elapsed:.2f}s")

        _section("MODEL OUTPUT")
        if msg.content and verbose:
            print(
                textwrap.fill(
                    msg.content,
                    width=68,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        if msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n  [tool_use]  {tc.function.name}  id={tc.id}")
                if verbose:
                    try:
                        print(_pretty_json(json.loads(tc.function.arguments), indent=4))
                    except json.JSONDecodeError:
                        print(f"    {tc.function.arguments}")

        if msg.content:
            accumulated_text.append(msg.content)

        # Append assistant message preserving tool_calls for API history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # ── Finished normally ────────────────────────────────────────
        if finish_reason == "stop":
            _section("AGENT FINISHED  (stop)")
            return "".join(accumulated_text) or "Analysis complete."

        # ── Mid-text truncation: ask model to continue ───────────────
        if finish_reason == "length":
            if msg.content and not msg.tool_calls:
                _log("length limit hit mid-text — sending continuation prompt ...")
                messages.append(
                    {
                        "role": "user",
                        "content": "Please continue your response from exactly where you left off. Do not repeat anything already written.",
                    }
                )
                continue
            _log("length limit with tool_calls — stopping")
            break

        if finish_reason != "tool_calls":
            _log(f"Unexpected finish reason '{finish_reason}' — breaking loop")
            break

        # ── Execute tool calls ───────────────────────────────────────
        _section("TOOL EXECUTION")
        tool_results: list[dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            print(f"\n  ┌─ {tc.function.name}")
            print(f"  │  id : {tc.id}")

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                # Don't silently substitute {} for malformed arguments — an
                # empty dict can pass validation via field defaults and
                # produce a confusing "successful" result for a call the
                # model never actually intended. Report the parse failure
                # back to the model instead of guessing its intent.
                print(f"  │  ✗  malformed tool-call JSON — {exc}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"Error: could not parse tool call arguments as JSON: {exc}. "
                            "Raw arguments were: " + tc.function.arguments
                        ),
                    }
                )
                print("  └" + "─" * 50)
                continue

            if verbose:
                print(_pretty_json(args, indent=5))

            t0 = time.perf_counter()
            # Not `with ThreadPoolExecutor() as ex:` — that context manager's
            # __exit__ calls shutdown(wait=True), which blocks until the
            # submitted call finishes regardless of the result() timeout
            # below, defeating the entire point of bounding a hung tool
            # call. shutdown(wait=False) in `finally` lets this loop move on
            # immediately while the orphaned thread runs out on its own.
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                result = ex.submit(tool_dispatch, tc.function.name, args).result(
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
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )
            except concurrent.futures.TimeoutError:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  TIMED OUT after {ms:.0f}ms (limit {tool_timeout_s}s)")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: tool call timed out after {tool_timeout_s}s",
                    }
                )
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  FAILED in {ms:.0f}ms — {exc}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Error: {exc}",
                    }
                )
            finally:
                ex.shutdown(wait=False)
            print("  └" + "─" * 50)

        messages.extend(tool_results)

    # ── Session summary ──────────────────────────────────────────────
    total_elapsed = time.perf_counter() - session_start
    _header("SESSION SUMMARY")
    _log("Iterations used", f"{iteration} / {max_iterations}")
    _log("Total input tokens", str(total_input_tokens))
    _log("Total output tokens", str(total_output_tokens))
    _log("Total tokens", str(total_input_tokens + total_output_tokens))
    _log("Total wall time", f"{total_elapsed:.2f}s")

    return "".join(accumulated_text) or "Max iterations reached."
