"""The runtime table and the tool-catalog appendix, from the live registry.

    python gen_catalog.py

Writes charts/runtimes-table.html (section 04's table) and
parts/92-catalog.html (Appendix C), so neither is written by hand.
"""

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from standard_quant_tools.agent import runtimes as rt  # noqa: E402
from standard_quant_tools.agent.tools import TOOL_CATEGORY  # noqa: E402
from standard_quant_tools.mcp.catalog import (  # noqa: E402
    build_catalog,
    select_runtimes,
)
from standard_quant_tools.mcp.server import LONG_RUNNING  # noqa: E402

DATE = "27 October 2026"


def first_sentence(text, limit=175):
    text = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    if len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s


runs = rt.all_runtimes()
catalog = build_catalog()
order = sorted(runs, key=lambda n: (-len(runs[n]), n))
total = sum(len(r) for r in runs.values())
kb = {
    n: sum(e.cost_bytes() for e in select_runtimes(catalog, [n])) / 1024 for n in runs
}

rows = []
for name in order:
    question = first_sentence(rt.RUNTIME_DESCRIPTIONS.get(name, ""), 150)
    rows.append(
        f'<tr><td class="k">{name}</td><td class="num">{len(runs[name])}</td>'
        f'<td class="num">{kb[name]:.0f} KB</td><td>{html.escape(question)}</td></tr>'
    )
rows.append(f'<tr><td class="k"><strong>all</strong></td><td class="num"><strong>{total}</strong></td>'
            f'<td class="num">{sum(kb.values()):.0f} KB</td><td>The whole surface, which no client is served at once.</td></tr>')  # fmt: skip
(HERE / "charts" / "runtimes-table.html").write_text(
    '<table class="tight"><thead><tr><th>Runtime</th><th class="num">Tools</th>'
    '<th class="num">Schema</th><th>What it answers</th></tr></thead><tbody>\n'
    + "\n".join(rows)
    + "\n</tbody></table>",
    encoding="utf-8",
    newline="\n",
)

blocks = []
for name in order:
    r = runs[name]
    defs = {tool: desc for tool, desc, _model in r.tool_defs}
    long = ' <span class="pill off">long</span>'
    body = "\n".join(
        f'<tr><td class="k">{html.escape(t)}{long if t in LONG_RUNNING else ""}</td>'
        f"<td>{html.escape(first_sentence(defs[t]))}</td></tr>"
        for t in sorted(r.tool_names)
    )
    label = rt.RUNTIME_LABELS.get(name, name)
    blocks.append(
        f'<h3 class="cat"><code>{name}</code> · {html.escape(label)} '
        f'<span class="muted">· {len(r)} tools</span></h3>\n'
        f'<p class="small">{html.escape(first_sentence(rt.RUNTIME_DESCRIPTIONS.get(name, ""), 400))}</p>\n'
        f'<table class="tight cat"><tbody>\n{body}\n</tbody></table>'
    )

page = f"""<h2 class="sec" id="catalog" data-toc="Appendix C · The complete tool catalog" data-n="C"><span class="chip c-amber">C</span>Appendix C · The complete tool catalog</h2>
<div class="sec-rule"></div>
<p class="lede">All {total} tools on {DATE}, generated from the live registry rather than written by hand,
grouped by the runtime that owns them, largest first. Each description is the first sentence of the
text the model reads when it chooses the tool.</p>
<p class="small">A runtime is an execution boundary: each holds its own dispatch table, so a tool from
another runtime is unroutable, and the refusal names the runtime that owns it. {len(TOOL_CATEGORY)} of the
{total} also sit in the analysis facade, <code>get_agent_tools()</code>. The {len(LONG_RUNNING)} marked <span class="pill off">long</span> are
long-running, and an MCP session serves them only with <code>--enable-long-running</code>.</p>
{chr(10).join(blocks)}
"""
(HERE / "parts" / "92-catalog.html").write_text(page, encoding="utf-8", newline="\n")
print(f"catalog: {total} tools in {len(runs)} runtimes, {len(TOOL_CATEGORY)} in the facade; "
      f"schema {sum(kb.values()):.0f} KB")  # fmt: skip
