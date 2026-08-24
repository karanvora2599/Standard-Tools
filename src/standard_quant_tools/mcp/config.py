"""
Server configuration, resolved once at startup.

WHY THIS IS FAIL-FAST. An MCP server is launched by a client, with a working
directory nobody chose and an environment stripped down to whatever the
client passes through. Every path this library writes to comes from an
environment variable, so an unset `SQT_RUNS_DIR` does not fail at startup --
it fails three turns into a conversation, when a backtest tries to persist
an artifact somewhere unintended, and the user sees a tool error rather than
a configuration problem.

So everything is resolved and checked here, before the first request is
served, and reported once to stderr.

STDERR, NEVER STDOUT. stdio transport puts JSON-RPC on stdout. A single
stray write corrupts the stream in a way that surfaces as an unintelligible
protocol error, so this module logs to stderr and `tests/mcp/` asserts that
importing the library writes nothing to stdout.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from standard_quant_tools.mcp.catalog import ALL_CATEGORIES, DEFAULT_CATEGORIES

#: Results larger than this are persisted and returned as a resource link
#: instead of inlined. A five-year daily backtest carries ~1,250 equity
#: points plus a trade log; inlined, that sits in the client's context for
#: the rest of the session. 4 KB is the default rather than a law -- it is
#: reported in every truncated result so nothing is ever silently dropped.
DEFAULT_INLINE_LIMIT = 4096

_ENV_DIRS = (
    ("runs_dir", "SQT_RUNS_DIR", "artifact store"),
    ("audit_dir", "SQT_AUDIT_DIR", "audit trail"),
    ("cache_dir", "SQT_CACHE_DIR", "Parquet cache"),
)


@dataclass(frozen=True)
class ServerConfig:
    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    runs_dir: Optional[Path] = None
    audit_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    inline_limit_bytes: int = DEFAULT_INLINE_LIMIT
    include_output_schemas: bool = False
    enable_long_running: bool = False
    warnings: Tuple[str, ...] = field(default=())


def _split_categories(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(ALL_CATEGORIES)
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sqt-mcp",
        description=(
            "Serve Standard Tools over the Model Context Protocol (stdio). "
            "Tools are selected by category because the full set of 54 costs "
            "roughly 30,000 tokens of a client's context at connect."
        ),
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help=(
            "Comma-separated tool categories, or 'all'. "
            f"Available: {', '.join(ALL_CATEGORIES)}. "
            f"Default: {','.join(DEFAULT_CATEGORIES)}."
        ),
    )
    parser.add_argument(
        "--inline-limit",
        type=int,
        default=DEFAULT_INLINE_LIMIT,
        help=(
            "Results larger than this many bytes are persisted and returned "
            f"as a resource link. Default: {DEFAULT_INLINE_LIMIT}."
        ),
    )
    parser.add_argument(
        "--output-schemas",
        action="store_true",
        help=(
            "Declare outputSchema for every tool. Costs about 77%% more "
            "context (74 KB across all 54 tools) and is off by default: "
            "structuredContent is returned either way, and the declaration "
            "only helps clients that validate against it."
        ),
    )
    parser.add_argument(
        "--enable-long-running",
        action="store_true",
        help=(
            "Expose the tools that can run for minutes "
            "(scan_cointegrated_pairs is measured at 5.31 min over a "
            "2,000-ticker universe). Off by default, because a client "
            "timeout that fires after most of the work is done is worse "
            "than not offering the tool."
        ),
    )
    parser.add_argument(
        "--print-budget",
        action="store_true",
        help="Print the per-category context cost and exit.",
    )
    return parser


def _resolve_dir(env_var: str, purpose: str, warnings: List[str]) -> Optional[Path]:
    raw = os.environ.get(env_var)
    if not raw:
        warnings.append(
            f"{env_var} is not set, so the {purpose} will use its default "
            "location relative to this server's working directory -- which "
            "the MCP client chose, not you. Artifacts and resource URIs may "
            "not survive a restart. Set it in the client's server config."
        )
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sqt-mcp-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"sqt-mcp: {env_var} points at {path}, which is not writable "
            f"({exc}). The {purpose} needs it. Fix the path or the "
            "permissions in the client's server configuration."
        )
    return path


def resolve(argv: Optional[Sequence[str]] = None) -> ServerConfig:
    """Parse arguments, resolve the environment, and fail fast if it is wrong."""
    args = build_parser().parse_args(argv)

    categories = _split_categories(args.categories)
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise SystemExit(
            f"sqt-mcp: unknown categor{'y' if len(unknown) == 1 else 'ies'} "
            f"{unknown}. Available: {', '.join(ALL_CATEGORIES)}."
        )
    if not categories:
        raise SystemExit("sqt-mcp: --categories selected nothing")
    if args.inline_limit < 0:
        raise SystemExit("sqt-mcp: --inline-limit must be >= 0")

    warnings: List[str] = []
    dirs = {
        attr: _resolve_dir(env_var, purpose, warnings)
        for attr, env_var, purpose in _ENV_DIRS
    }

    return ServerConfig(
        categories=tuple(categories),
        inline_limit_bytes=args.inline_limit,
        include_output_schemas=bool(args.output_schemas),
        enable_long_running=bool(args.enable_long_running),
        warnings=tuple(warnings),
        **dirs,
    )


def report(config: ServerConfig, tool_count: int, schema_bytes_total: int) -> None:
    """Say what was configured, once, on stderr."""
    lines = [
        "sqt-mcp starting",
        f"  categories        : {', '.join(config.categories)}",
        f"  tools exposed     : {tool_count}",
        f"  context cost      : {schema_bytes_total / 1024:.1f} KB "
        f"(~{schema_bytes_total // 4:,} tokens at connect)",
        f"  output schemas    : {'declared' if config.include_output_schemas else 'omitted (structuredContent still sent)'}",
        f"  inline limit      : {config.inline_limit_bytes} bytes",
        f"  long-running tools: {'enabled' if config.enable_long_running else 'hidden'}",
    ]
    for attr, env_var, _purpose in _ENV_DIRS:
        value = getattr(config, attr)
        lines.append(f"  {env_var:<18}: {value if value else '(unset)'}")
    for warning in config.warnings:
        lines.append(f"  WARNING: {warning}")
    print("\n".join(lines), file=sys.stderr, flush=True)
