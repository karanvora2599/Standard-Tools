"""Assemble parts/*.html into one document, render it with headless Chrome,
and fill the contents page with real page numbers (two passes).

usage: build.py OUT.pdf [dark]          (build_all.py builds both editions)

`dark` builds the dark edition: dark.css over style.css, the dark lattices,
and every colour baked into a chart remapped through DARK_SVG. A blank page
goes in before the back cover when the count would be odd, so the back cover
prints on the back of the last sheet.
"""

import re
import subprocess
import sys
from pathlib import Path

import pypdfium2 as pdfium
from chrome import CHROME, PROFILE
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).resolve().parent
PARTS = HERE / "parts"
CHARTS = HERE / "charts"
DOCS = HERE.parent.parent
OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else DOCS / "Standard-Quant-Tools-Briefing.pdf"
)
FLAGS = set(sys.argv[2:])
assert FLAGS <= {"dark"}, f"unknown flags: {FLAGS - {'dark'}}"
DARK = "dark" in FLAGS

# Every colour a chart carries, light -> dark: paper #0b0b0b/#131313/#1a1a1a,
# ink #efefed/#a4a4a4/#8c8c8c, gridline #2a2a2a/#444, accent #75fb6d/#14301a,
# critical #f25144, and the categorical slots' dark counterparts. A colour
# missing here fails the build, so the dark edition never shows a light one.
DARK_SVG = {
    "#fff": "#131313", "#1a1a1a": "#efefed", "#3d3d3d": "#a4a4a4", "#333": "#c8c8c6",
    "#444": "#a4a4a4", "#555": "#a4a4a4", "#6b6b6b": "#8c8c8c", "#999": "#6b6b6b",
    "#bbb": "#444", "#f4f3ef": "#1a1a1a", "#e4e1da": "#262626", "#dedad2": "#2e2e2e",
    "#2a78d6": "#3987e5", "#eb6834": "#d95926",
    "#5bbff4": "#5ac5fa", "#e9f6fe": "#0c1b24", "#3fd9b7": "#75fbe1", "#f0fdf9": "#0b1f1b",
    "#79ee5f": "#75fb6d", "#e8fbe2": "#14301a", "#f2b43c": "#f5be41", "#fdf3dd": "#1f190b",
    "#fdf0d3": "#2a2108", "#ecc777": "#4a3a12", "#7a5000": "#f5be41",
    "#d6352a": "#dc2f21", "#fdf3f2": "#221010", "#a8231b": "#f25144",
}  # fmt: skip

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Standard Quant Tools: October 2026 Briefing</title>
<link rel="stylesheet" href="style.css">
<style>.tm{font-size:1px;color:#fff;letter-spacing:0;line-height:0;}</style>
</head><body>
"""


def darken_svgs(body):
    """Remap every fill, stroke and stop colour in the charts. The lattices
    are skipped: they have dark versions of their own."""
    unmapped = set()

    def one(svg_match):
        svg = svg_match.group(0)
        if 'class="hex-lattice"' in svg:
            return svg

        def sub(m):
            attr, value = m.group(1), m.group(2).strip().lower()
            if value == "none" or value.startswith("url("):
                return m.group(0)
            if value not in DARK_SVG:
                unmapped.add(value)
                return m.group(0)
            return f'{attr}="{DARK_SVG[value]}"'

        return re.sub(r'\b(fill|stroke|stop-color)="([^"]+)"', sub, svg)

    body = re.sub(r"<svg\b.*?</svg>", one, body, flags=re.S)
    assert not unmapped, f"chart colours with no dark mapping: {sorted(unmapped)}"
    return body


def assemble(pages=None, filler=False):
    files = sorted(PARTS.glob("*.html"))
    back = [p for p in files if p.name.startswith("99-")]
    assert len(back) == 1, "parts/ needs exactly one 99- back cover"
    body = "".join(p.read_text(encoding="utf-8") for p in files if p not in back)
    if filler:
        body += '<div class="filler"><p>THIS PAGE IS INTENTIONALLY BLANK</p></div>'
    body += back[0].read_text(encoding="utf-8")
    if DARK:
        for lattice in ("cover-hex", "back-hex"):
            include = f"<!--INCLUDE:{lattice}.svg-->"
            assert body.count(include) == 1, include
            body = body.replace(include, f"<!--INCLUDE:{lattice}-dark.svg-->")
    body = re.sub(
        r"<!--INCLUDE:([\w.-]+)-->",
        lambda m: (CHARTS / m.group(1)).read_text(encoding="utf-8"),
        body,
    )
    if DARK:
        body = darken_svgs(body)
    entries = []
    for m in re.finditer(
        r'<(?:h2 class="sec[^"]*"|section class="part") id="([^"]+)" '
        r'data-toc="([^"]+)"(?: data-n="([^"]*)")?',
        body,
    ):
        kind = "part" if m.group(0).startswith("<section") else "sec"
        entries.append((kind, m.group(1), m.group(2), m.group(3) or ""))
    rows = []
    for kind, ident, title, n in entries:
        page = (pages or {}).get(ident, "00")
        cls = "row part-row" if kind == "part" else "row"
        rows.append(
            f'<a class="{cls}" href="#{ident}"><span class="n">{n}</span>'
            f'<span class="t">{title}</span><span class="p">{page}</span></a>'
        )
    body = body.replace("<!--TOC-->", "\n".join(rows))
    # An invisible marker inside every contents target, found again in the PDF text.
    body = re.sub(
        r'(<(?:h2 class="sec[^"]*"|section class="part") id="([^"]+)"[^>]*>)',
        lambda m: m.group(1)
        + f'<span class="tm">QQ{m.group(2).replace("-", "")}QQ</span>',
        body,
    )
    head = HEAD
    if DARK:
        head = HEAD.replace(
            '<link rel="stylesheet" href="style.css">',
            '<link rel="stylesheet" href="style.css">\n<link rel="stylesheet" href="dark.css">',
        )
    return head + body + "\n</body></html>", entries


def render(html, pdf):
    src = HERE / "index.html"
    src.write_text(html, encoding="utf-8")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--generate-pdf-document-outline", f"--user-data-dir={PROFILE}",
         f"--print-to-pdf={pdf}", src.as_uri()],
        check=True, capture_output=True, timeout=240,
    )  # fmt: skip


def locate(pdf, entries):
    doc = pdfium.PdfDocument(str(pdf))
    found = {}
    for i in range(len(doc)):
        text = doc[i].get_textpage().get_text_range()
        for _, ident, _, _ in entries:
            if ident not in found and f"QQ{ident.replace('-', '')}QQ" in text:
                found[ident] = str(i + 1)
    missing = [e[1] for e in entries if e[1] not in found]
    return found, missing, len(doc)


def main():
    html, entries = assemble()
    draft = HERE / "draft.pdf"
    render(html, draft)
    pages, missing, n = locate(draft, entries)
    if missing:
        sys.exit(f"contents: no page found for {missing}")
    html, _ = assemble(pages, filler=n % 2 == 1)
    final = HERE / "final.pdf"
    render(html, final)
    pages2, _, n2 = locate(final, entries)
    if pages2 != pages:
        # The real page numbers moved the layout, so the contents would be wrong.
        moved = {k: (pages.get(k), v) for k, v in pages2.items() if pages.get(k) != v}
        sys.exit(f"contents: pagination moved between passes: {moved}")
    w = PdfWriter(clone_from=PdfReader(str(final)))
    w.add_metadata(
        {
            "/Title": "Standard Quant Tools: Quantitative Finance for AI Agents. October 2026 Briefing"
            + (" (dark)" if DARK else ""),
            "/Author": "Karan Vora",
            "/Subject": "What Standard Quant Tools is, how it is built, what it computes, and the evidence behind it",
            "/Keywords": "Standard Quant Tools, quantitative finance, AI agents, tools, MCP, audit, backtesting",
        }
    )
    with open(OUT, "wb") as f:
        w.write(f)
    print(f"pages={n2} -> {OUT}")
    for _, ident, title, _ in entries:
        print(f"  p{pages2.get(ident, '?'):>3}  {title}")


if __name__ == "__main__":
    main()
