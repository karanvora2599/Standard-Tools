"""
Generate `Documentation/20_tool_index.md` from the live registry.

WHY GENERATED RATHER THAN WRITTEN. A hand-maintained index of 157 tools is
stale the day after it is written, and a stale index is worse than none:
it tells a reader a tool does not exist when it does, or describes one that
was renamed. Before this existed, 85 of the 157 tools appeared in no
document at all -- not because anyone decided they should not, but because
adding a tool and remembering to document it are two actions and only one
of them was enforced.

This makes them one action. `tests/docs/test_documentation.py` renders the
index and fails if what is on disk differs, so a tool added without
regenerating breaks the suite in the same commit that added it.

RUN IT WITH:  python scripts/generate_tool_index.py
              python scripts/generate_tool_index.py --check   (writes nothing)

WHAT "DIFFERS" MEANS. Every tool name, description, argument list, count
and heading must match exactly. The one figure that need not is each
runtime's schema cost: it is a whole number of KB from serialized JSON
Schema, which a different pydantic release can lengthen or shorten by a few
bytes without any tool changing, and a runtime that sits near a half
kilobyte then rounds the other way. A cost within `KB_TOLERANCE` of the
live one passes; a larger move still fails, so the figure on disk is never
more than that far from the truth.

The file is written with LF on every platform, as `.gitattributes` stores
it; a CRLF write on Windows showed as a modification in `git status`.

The descriptions are not rewritten here -- they are the same strings the
model sees when it chooses a tool. That is deliberate. If a description
reads badly in this document it reads badly to the model too, and the fix
belongs in the tool definition rather than in a parallel prose layer that
can disagree with it.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "Documentation" / "20_tool_index.md"

#: How far, in whole KB, a runtime's schema cost on disk may sit from the
#: live one. See the module docstring.
KB_TOLERANCE = 1

#: A row of the runtimes table, with the schema cost as its own group.
_COST_ROW = re.compile(
    r"^(?P<head>\| `(?P<runtime>[^`]+)` \| \d+ \| )(?P<kb>\d+)(?P<tail> KB \|)",
    re.MULTILINE,
)

#: Where the deep documentation for each runtime lives, when it has one.
#:
#: A MISSING KEY IS SILENT. The lookup below degrades to an em-dash, so a
#: runtime that HAS a guide but no entry here advertises none, and nothing
#: fails -- `data` sat that way with `26_data.md` on disk the whole time,
#: and `meta` pointed only at the audit guide while `27_meta.md` was its
#: actual deep dive. When a runtime is added, this dict is the easiest
#: thing in the checklist to forget and the only one that stays green.
DEEP_DOCS = {
    "research": "[08_analysis.md](08_analysis.md), [23_inference.md](23_inference.md)",
    "backtest": "[04_backtesting.md](04_backtesting.md), [24_overfitting.md](24_overfitting.md)",
    "portfolio": "[05_portfolio.md](05_portfolio.md)",
    "data": "[26_data.md](26_data.md)",
    "microstructure": "[22_microstructure.md](22_microstructure.md)",
    "derivatives": "[21_derivatives.md](21_derivatives.md)",
    "delta_one": "[28_delta_one.md](28_delta_one.md)",
    "meta": "[27_meta.md](27_meta.md), [10_auditability.md](10_auditability.md)",
    "modeling": "[15_modeling.md](15_modeling.md)",
    "feature_lab": "[15_modeling.md](15_modeling.md)",
}

HEADER = """# Tool index

Every tool in the library, by runtime, with the description the model
actually sees. **Generated from the live registry** by
`scripts/generate_tool_index.py` -- a test renders it and fails if
this file has drifted, so a tool added without regenerating breaks the
suite in the commit that added it.

The descriptions here are not a parallel prose layer. They are the exact
strings a model reads when choosing a tool, which means a description that
reads badly here reads badly to the model too, and the fix belongs in the
tool definition.

## How to read this

A **runtime** is an execution boundary, not a category. Each holds its own
dispatch table, so a tool from another runtime is *unroutable* rather than
discouraged -- and the refusal names the runtime that owns it, so a scoping
mistake is recoverable and a hallucinated name is not mistaken for one.
See [19_runtimes.md](19_runtimes.md) for why, and how results still cross
between them.

A **category** narrows *within* a runtime. `--categories microstructure` is
not the same as `--runtime microstructure`, and the difference matters when
scoping an MCP session -- see [18_mcp.md](18_mcp.md).

Two tools (`run_backtest_optimization`, `scan_pairs`) are long-running and
are served only with `--enable-long-running`. A default MCP session is also
scoped to four categories rather than all of them, so it advertises 58 of the
{total} below; `--runtime all --enable-long-running` serves every one.

"""


def render() -> str:
    """The index as the live registry describes it, LF line endings."""
    from standard_quant_tools.agent.runtimes import all_runtimes
    from standard_quant_tools.mcp.catalog import build_catalog, select_runtimes
    from standard_quant_tools.mcp.server import LONG_RUNNING

    runtimes = all_runtimes()
    catalog = build_catalog()
    total = sum(len(rt) for rt in runtimes.values())

    lines = [HEADER.format(total=total)]

    # Summary table first: the orientation a reader needs before 157 rows.
    lines.append("## The runtimes\n")
    lines.append("| Runtime | Tools | Schema cost | Categories | Deep documentation |")
    lines.append("|---|---:|---:|---|---|")
    for name, rt in sorted(runtimes.items(), key=lambda kv: -len(kv[1])):
        kilobytes = sum(e.cost_bytes() for e in select_runtimes(catalog, [name])) / 1024
        categories = (
            ", ".join(f"`{c}`" for c in rt.categories)
            if rt.categories != (name,)
            else "*(one surface)*"
        )
        lines.append(
            f"| `{name}` | {len(rt)} | {kilobytes:.0f} KB | {categories} | "
            f"{DEEP_DOCS.get(name, '—')} |"
        )
    lines.append(f"| **Total** | **{total}** | | | |")
    lines.append("")

    for name, rt in sorted(runtimes.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"---\n\n## `{name}` — {rt.label}\n")
        lines.append(f"{rt.description}\n")

        # Group by category when the runtime has more than one.
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        by_category: dict[str, list] = {}
        for tool_name, description, model in sorted(rt.tool_defs):
            category = TOOL_CATEGORY.get(tool_name, name)
            by_category.setdefault(category, []).append((tool_name, description, model))

        for category in sorted(by_category):
            if len(by_category) > 1:
                lines.append(f"### `{category}`\n")
            for tool_name, description, model in by_category[category]:
                flag = (
                    "  \n*Long-running: served only with `--enable-long-running`.*"
                    if tool_name in LONG_RUNNING
                    else ""
                )
                fields = list(model.model_fields)
                required = [
                    f for f, info in model.model_fields.items() if info.is_required()
                ]
                signature = (
                    f"**Required:** {', '.join(f'`{f}`' for f in required)}"
                    if required
                    else "*No required arguments.*"
                )
                optional = [f for f in fields if f not in required]
                if optional:
                    signature += (
                        f"  \n**Optional:** {', '.join(f'`{f}`' for f in optional)}"
                    )
                lines.append(f"#### `{tool_name}`\n")
                lines.append(f"{description}{flag}\n")
                lines.append(f"{signature}\n")

    return "\n".join(lines).rstrip() + "\n"


def _costs(text: str) -> Dict[str, int]:
    return {m["runtime"]: int(m["kb"]) for m in _COST_ROW.finditer(text)}


def _masked(text: str) -> str:
    return _COST_ROW.sub(lambda m: f"{m['head']}#{m['tail']}", text)


def drift(on_disk: str, generated: str, tolerance_kb: int = KB_TOLERANCE) -> List[str]:
    """
    Why `on_disk` is not the index `generated` describes; empty when it is.

    Everything but the runtimes' schema costs is compared exactly, line
    endings aside. Each cost may differ from the generated one by at most
    `tolerance_kb`.
    """
    on_disk = on_disk.replace("\r\n", "\n")
    generated = generated.replace("\r\n", "\n")
    problems: List[str] = []
    if _masked(on_disk) != _masked(generated):
        lines = difflib.unified_diff(
            _masked(on_disk).splitlines(),
            _masked(generated).splitlines(),
            "on disk",
            "generated",
            lineterm="",
            n=0,
        )
        problems.append("the text differs:\n" + "\n".join(list(lines)[:40]))
    live = _costs(generated)
    for runtime, kilobytes in _costs(on_disk).items():
        if runtime in live and abs(live[runtime] - kilobytes) > tolerance_kb:
            problems.append(
                f"`{runtime}` schema cost is {kilobytes} KB on disk and "
                f"{live[runtime]} KB live, more than {tolerance_kb} KB apart"
            )
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the index on disk with the live registry, write nothing, "
        "and exit 1 if it is out of date",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="where to write the index, or with --check the index to compare",
    )
    options = parser.parse_args(argv)
    text = render()
    if options.check:
        problems = drift(options.output.read_text(encoding="utf-8"), text)
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    options.output.write_text(text, encoding="utf-8", newline="\n")
    tools, runtimes = text.count("\n#### `"), text.count("\n## `")
    print(f"wrote {options.output} — {tools} tools across {runtimes} runtimes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
