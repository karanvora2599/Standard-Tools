"""
Shared utilities for all agentic scripts — Google Gemini provider.

Provides:
  - setup_logging(name)   → configure lib logger + per-run file handler
  - _header / _section / _log / _pretty_json → console formatting helpers
  - run_agent()           → the core agentic loop (Gemini + tool dispatch)

Requires: pip install google-genai

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
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Type as GeminiType

from standard_quant_tools.agent.tools import get_agent_tools, dispatch

# ── Constants ──────────────────────────────────────────────────────
_LOGS_DIR     = Path(__file__).resolve().parent.parent.parent / "logs"
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
    """Attach a per-run FileHandler + StreamHandler to the standard_quant_tools logger."""
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

def _schema_to_gemini(schema: dict[str, Any]) -> types.Schema:
    """
    Recursively convert a JSON Schema dict (Pydantic v2 output) to a
    Gemini types.Schema.  Handles anyOf (Pydantic Optional[X] pattern).
    """
    # Unwrap Optional[X] → {"anyOf": [{"type": "X"}, {"type": "null"}]}
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return _schema_to_gemini(non_null[0]) if non_null else types.Schema(type=GeminiType.STRING)  # type: ignore[arg-type]

    _type_map: dict[str, GeminiType] = {
        "string":  GeminiType.STRING,
        "integer": GeminiType.INTEGER,
        "number":  GeminiType.NUMBER,
        "boolean": GeminiType.BOOLEAN,
        "array":   GeminiType.ARRAY,
        "object":  GeminiType.OBJECT,
    }

    kwargs: dict[str, Any] = {
        "type": _type_map.get(schema.get("type", "object"), GeminiType.STRING),
    }

    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "properties" in schema:
        kwargs["properties"] = {
            k: _schema_to_gemini(v) for k, v in schema["properties"].items()
        }
    if "required" in schema:
        kwargs["required"] = list(schema["required"])
    if "items" in schema:
        kwargs["items"] = _schema_to_gemini(schema["items"])
    if "enum" in schema:
        kwargs["enum"] = [str(e) for e in schema["enum"]]

    return types.Schema(**kwargs)


def _to_gemini_tools(openai_tools: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert get_agent_tools() OpenAI format → Gemini FunctionDeclaration list."""
    declarations = [
        types.FunctionDeclaration(
            name=t["function"]["name"],
            description=t["function"]["description"],
            parameters=_schema_to_gemini(t["function"]["parameters"]),
        )
        for t in openai_tools
    ]
    return [types.Tool(function_declarations=declarations)]


# ── Core agent loop ─────────────────────────────────────────────────

def run_agent(
    system_prompt: str,
    user_request: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
    max_iterations: int = 15,
    max_tokens: int = 8192,
    request_timeout_s: float = 60.0,
    tool_timeout_s: float = 120.0,
    verbose: bool = True,
) -> str:
    """
    Run the agentic loop: send user_request to Gemini, execute any function
    calls via dispatch(), feed results back, and repeat until STOP or exhausted.

    finish_reason values (candidate.finish_reason.name):
      "STOP"       — model finished normally
      "MAX_TOKENS" — hit token limit mid-response (continuation logic applies)

    request_timeout_s bounds each Gemini API call; tool_timeout_s bounds each
    individual dispatch() call (run in a worker thread so a hung tool can't
    block the loop forever). verbose=False suppresses printing full
    user/tool payloads (see the module docstring) while keeping status-line
    output.

    Returns the final text response from the model.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(request_timeout_s * 1000)),
    )
    tools  = _to_gemini_tools(get_agent_tools())

    _header("AGENT SESSION STARTED  (Gemini)")
    _log("Model",          model)
    _log("Max tokens",     str(max_tokens))
    _log("Max iterations", str(max_iterations))
    _log("Tools loaded",   str(sum(len(t.function_declarations or []) for t in tools)))
    _log("Tool names",     ", ".join(
        fd.name for t in tools for fd in (t.function_declarations or [])
    ))
    if verbose:
        _section("USER REQUEST")
        print(textwrap.fill(user_request, width=68, initial_indent="  ", subsequent_indent="  "))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools,           # type: ignore[arg-type]
        max_output_tokens=max_tokens,
    )

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_request)])
    ]
    session_start        = time.perf_counter()
    total_input_tokens   = 0
    total_output_tokens  = 0
    iteration            = 0
    accumulated_text: list[str] = []

    for iteration in range(1, max_iterations + 1):
        _header(f"ITERATION {iteration}")
        iter_start = time.perf_counter()

        _log("Sending request to Gemini ...")
        response = client.models.generate_content(
            model=model,
            contents=contents,  # type: ignore[arg-type]
            config=config,
        )

        elapsed = time.perf_counter() - iter_start
        if response.usage_metadata:
            total_input_tokens  += response.usage_metadata.prompt_token_count or 0
            total_output_tokens += response.usage_metadata.candidates_token_count or 0

        # Guard: Gemini response fields can be None on safety blocks
        if not response.candidates:
            _log("No candidates in response — breaking loop")
            break
        candidate = response.candidates[0]
        if candidate is None or candidate.content is None:
            _log("Empty candidate — breaking loop")
            break

        finish_reason = (candidate.finish_reason.name  # "STOP", "MAX_TOKENS", etc.
                         if candidate.finish_reason else "UNKNOWN")
        parts: list[types.Part] = list(candidate.content.parts or [])

        _section("API RESPONSE METADATA")
        _log("Finish reason",  finish_reason)
        _log("Input tokens",   str(response.usage_metadata.prompt_token_count if response.usage_metadata else "—"))
        _log("Output tokens",  str(response.usage_metadata.candidates_token_count if response.usage_metadata else "—"))
        _log("Latency",        f"{elapsed:.2f}s")

        _section(f"MODEL OUTPUT  ({len(parts)} part(s))")
        for i, part in enumerate(parts):
            text = getattr(part, "text", None)
            fn   = getattr(part, "function_call", None)
            if text:
                print(f"\n  [Part {i+1}]  type=text")
                if verbose:
                    print(textwrap.fill(text, width=68, initial_indent="    ", subsequent_indent="    "))
                accumulated_text.append(text)
            elif fn:
                print(f"\n  [Part {i+1}]  type=function_call  name={fn.name}")
                if verbose:
                    print(_pretty_json(dict(fn.args), indent=4))

        # Append model turn to history
        contents.append(candidate.content)  # type: ignore[arg-type]

        has_text = any(getattr(p, "text", None) for p in parts)
        has_fn   = any(getattr(p, "function_call", None) for p in parts)

        # ── Finished normally ────────────────────────────────────────
        if finish_reason == "STOP":
            _section("AGENT FINISHED  (STOP)")
            return "".join(accumulated_text) or "Analysis complete."

        # ── Mid-text truncation: ask model to continue ───────────────
        if finish_reason == "MAX_TOKENS":
            if has_text and not has_fn:
                _log("MAX_TOKENS hit mid-text — sending continuation prompt ...")
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=(
                        "Please continue your response from exactly where you left off. "
                        "Do not repeat anything already written."
                    ))],
                ))
                continue
            _log("MAX_TOKENS with function_calls — stopping")
            break

        if not has_fn:
            _log(f"Unexpected finish reason '{finish_reason}' — breaking loop")
            break

        # ── Execute function calls ───────────────────────────────────
        _section("TOOL EXECUTION")
        fn_response_parts: list[types.Part] = []

        for part in parts:  # type: ignore[union-attr]
            fn = getattr(part, "function_call", None)
            if fn is None:
                continue

            name = fn.name
            args = dict(fn.args)

            print(f"\n  ┌─ {name}")
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
                result = ex.submit(dispatch, name, args).result(timeout=tool_timeout_s)
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✓  completed in {ms:.0f}ms")
                if verbose:
                    print(_pretty_json(result, indent=5))
                fn_response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=name,
                        response=result,
                    ))
                )
            except concurrent.futures.TimeoutError:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  TIMED OUT after {ms:.0f}ms (limit {tool_timeout_s}s)")
                fn_response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=name,
                        response={"error": f"tool call timed out after {tool_timeout_s}s"},
                    ))
                )
            except Exception as exc:
                ms = (time.perf_counter() - t0) * 1000
                print(f"  │  ✗  FAILED in {ms:.0f}ms — {exc}")
                fn_response_parts.append(
                    types.Part(function_response=types.FunctionResponse(
                        name=name,
                        response={"error": str(exc)},
                    ))
                )
            finally:
                ex.shutdown(wait=False)
            print("  └" + "─" * 50)

        contents.append(types.Content(role="user", parts=fn_response_parts))

    # ── Session summary ──────────────────────────────────────────────
    total_elapsed = time.perf_counter() - session_start
    _header("SESSION SUMMARY")
    _log("Iterations used",     f"{iteration} / {max_iterations}")
    _log("Total input tokens",  str(total_input_tokens))
    _log("Total output tokens", str(total_output_tokens))
    _log("Total tokens",        str(total_input_tokens + total_output_tokens))
    _log("Total wall time",     f"{total_elapsed:.2f}s")

    return "".join(accumulated_text) or "Max iterations reached."
