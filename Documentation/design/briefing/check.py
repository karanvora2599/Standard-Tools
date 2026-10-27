"""Layout and readability checks for one built edition. Exits 1 on any problem.

usage: check.py index.html doc.pdf [sheet]
  index.html  the assembled page build.py leaves behind for that edition
  sheet       also write contact sheets (sheet-NN.png) beside the PDF

Checks: code lines that would wrap; fonts other than IBM Plex embedded; SVG
labels that overrun their box or canvas (measured with the real font
metrics); elements wider than the text column; covers that clip their
content; WCAG text contrast; PDF text past the page margins."""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pypdfium2 as pdfium
from chrome import CHROME, PROFILE
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
FONTS = HERE.parent / "fonts"  # shared with the README images
_fonts = {}


def font(family, weight):
    mono = "Mono" in family
    w = int(weight)
    choices = [400, 500, 600] if mono else [300, 400, 500, 600, 700]
    w = min(choices, key=lambda c: abs(c - w))
    style = {300: "Light", 400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"}[
        w
    ]
    name = f"IBMPlex{'Mono' if mono else 'Sans'}-{style}.woff2"
    if name not in _fonts:
        f = TTFont(FONTS / name)
        _fonts[name] = (f["hmtx"].metrics, f.getBestCmap(), f["head"].unitsPerEm)
    return _fonts[name]


def text_width(s, family, weight, size, spacing):
    metrics, cmap, upm = font(family, weight)
    total = 0
    for ch in s:
        g = cmap.get(ord(ch))
        adv = metrics[g][0] if g in metrics else upm * 0.6
        total += adv
    return total / upm * size + spacing * max(len(s) - 1, 0)


def num(v, default=0.0):
    try:
        return float(str(v).replace("px", ""))
    except (TypeError, ValueError):
        return default


def check_svgs(html):
    problems = []
    for k, m in enumerate(re.finditer(r"<svg\b.*?</svg>", html, re.S)):
        src = m.group(0)
        try:
            root = ET.fromstring(src)
        except ET.ParseError as e:
            problems.append(f"svg#{k}: not well-formed XML ({e})")
            continue
        vb = [float(x) for x in root.get("viewBox").split()]
        W, H = vb[2], vb[3]
        X0, Y0 = vb[0], vb[1]
        rects = []
        for r in root.iter("{http://www.w3.org/2000/svg}rect"):
            rx, ry, rw, rh = (
                num(r.get("x")),
                num(r.get("y")),
                num(r.get("width")),
                num(r.get("height")),
            )
            rects.append((rx, ry, rw, rh))
            sw = (
                num(r.get("stroke-width"), 1.0)
                if r.get("stroke") not in (None, "none")
                else 0
            )
            if sw and (
                rx - sw / 2 < X0
                or ry - sw / 2 < Y0
                or rx + rw + sw / 2 > X0 + W
                or ry + rh + sw / 2 > Y0 + H
            ):
                problems.append(
                    f"svg#{k}: stroked rect at x={rx:.0f} y={ry:.0f} w={rw:.0f} h={rh:.0f} has its border clipped by the canvas"
                )

        def walk(node, inh):
            st = dict(inh)
            for a in (
                "font-family",
                "font-size",
                "font-weight",
                "text-anchor",
                "letter-spacing",
            ):
                if node.get(a) is not None:
                    st[a] = node.get(a)
            if node.tag.endswith("text"):
                s = "".join(node.itertext())
                x, y = num(node.get("x")), num(node.get("y"))
                w = text_width(
                    s,
                    st["font-family"],
                    num(st.get("font-weight", 400), 400),
                    num(st["font-size"], 9),
                    num(st.get("letter-spacing", 0)),
                )
                anchor = st.get("text-anchor", "start")
                x0 = (
                    x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
                )
                x1 = x0 + w
                label = f"svg#{k} '{s[:40]}'"
                if x0 < X0 - 0.5 or x1 > X0 + W + 0.5:
                    problems.append(
                        f"{label}: outside canvas ({x0:.0f}..{x1:.0f} of {W:.0f})"
                    )
                if y > Y0 + H + 0.5 or y - num(st["font-size"], 9) < Y0 - 0.5:
                    problems.append(
                        f"{label}: outside canvas vertically (y={y:.0f}, H={H:.0f})"
                    )
                inside = [
                    r
                    for r in rects
                    if r[0] - 0.5 <= x0 + 0.5
                    and x0 + 0.5 <= r[0] + r[2]
                    and r[1] <= y - 2 <= r[1] + r[3]
                ]
                if inside:
                    r = min(inside, key=lambda r: r[2] * r[3])
                    if x1 > r[0] + r[2] - 1.5:
                        problems.append(
                            f"{label}: overruns its box by {x1 - (r[0] + r[2]):.1f} (box x={r[0]:.0f} w={r[2]:.0f})"
                        )
            for c in node:
                walk(c, st)

        walk(
            root,
            {
                "font-family": root.get("font-family", "Plex"),
                "font-size": root.get("font-size", "9"),
            },
        )
    return problems


PROBE = """<style>@media screen { body { width: 6.94in; margin: 0; } }</style>
<script>
document.fonts.ready.then(() => {
  const right = document.body.getBoundingClientRect().right;
  const bad = [];
  // A cover is a fixed sheet with overflow hidden, so too much content is
  // clipped silently rather than spilling onto a new page.
  for (const c of document.querySelectorAll('.cover')) {
    if (c.scrollHeight > c.clientHeight + 1 || c.scrollWidth > c.clientWidth + 1)
      bad.push(`cover.${c.className} clips its content: ${c.scrollWidth}x${c.scrollHeight} in ${c.clientWidth}x${c.clientHeight}`);
  }
  for (const el of document.querySelectorAll('body *')) {
    if (el.closest('.cover') || el.closest('svg') && el.tagName !== 'svg') continue;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.right > right + 0.5)
      bad.push(`${el.tagName.toLowerCase()}.${el.className} "${(el.textContent || '').trim().slice(0, 50)}" +${(r.right - right).toFixed(1)}px`);
    const cs = getComputedStyle(el);
    if (cs.display !== 'inline' && el.scrollWidth > el.clientWidth + 1 && el.clientWidth > 0 && el.tagName !== 'BODY')
      bad.push(`${el.tagName.toLowerCase()}.${el.className} "${(el.textContent || '').trim().slice(0, 50)}" scroll ${el.scrollWidth}>${el.clientWidth}`);
  }
  const pre = document.createElement('pre'); pre.id = 'overflow-report';
  pre.textContent = JSON.stringify(bad); document.body.appendChild(pre);
});
</script>"""


def check_dom(index_html):
    import json
    import subprocess

    src = Path(index_html).resolve()
    probe = src.with_name("_probe.html")
    probe.write_text(
        src.read_text(encoding="utf-8").replace("</body>", PROBE + "</body>"),
        encoding="utf-8",
    )
    out = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={PROFILE}",
            "--virtual-time-budget=10000",
            "--window-size=1200,900",
            "--dump-dom",
            probe.as_uri(),
        ],
        capture_output=True,
        timeout=240,
        encoding="utf-8",
        errors="replace",
    ).stdout
    m = re.search(r'<pre id="overflow-report">(.*?)</pre>', out, re.S)
    if not m:
        return ["DOM probe produced no report"]
    import html as h

    return json.loads(h.unescape(m.group(1)))


CONTRAST_PROBE = r"""<script>
document.fonts.ready.then(() => {
  const PAGE = document.querySelector('link[href="dark.css"]') ? [11, 11, 11] : [255, 255, 255];
  const rgba = s => { const m = s.match(/[\d.]+/g) || []; return [+m[0], +m[1], +m[2], m[3] === undefined ? 1 : +m[3]]; };
  const lum = ([r, g, b]) => { const f = c => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
                               return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const bgOf = el => {
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      if (n.tagName === 'figure' || n.tagName === 'FIGURE' || n.tagName !== 'svg' && n.namespaceURI === 'http://www.w3.org/1999/xhtml') {
        const c = rgba(getComputedStyle(n).backgroundColor);
        if (c[3] > 0.5) return c;
      }
    }
    return PAGE;
  };
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.closest('.tm') || el.closest('.hex-lattice')) continue;
    const own = [...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim());
    if (!own) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const isSvg = el.namespaceURI === 'http://www.w3.org/2000/svg';
    const fg = rgba(isSvg ? cs.fill : cs.color);
    if (fg[3] === 0) continue;
    const bg = bgOf(isSvg ? el.closest('figure') || el : el);
    const px = parseFloat(cs.fontSize), weight = parseInt(cs.fontWeight) || 400;
    const large = px >= 24 || (px >= 18.66 && weight >= 600);
    const need = large ? 3.0 : 4.5;
    const r = ratio(fg, bg);
    if (r < need) bad.push(`${r.toFixed(2)}<${need} ${el.tagName.toLowerCase()}.${(el.className && el.className.baseVal !== undefined) ? el.className.baseVal : el.className} "${el.textContent.trim().slice(0, 40)}"`);
  }
  const pre = document.createElement('pre'); pre.id = 'contrast-report';
  pre.textContent = JSON.stringify(bad); document.body.appendChild(pre);
});
</script>"""


def check_contrast(index_html):
    """WCAG text contrast for every element with its own text, against the
    nearest opaque background (the page colour when none): 4.5:1, or 3:1 for
    large text. SVG labels are measured by fill against their figure."""
    import html as h
    import json
    import subprocess

    src = Path(index_html).resolve()
    probe = src.with_name("_contrast.html")
    probe.write_text(
        src.read_text(encoding="utf-8").replace("</body>", CONTRAST_PROBE + "</body>"),
        encoding="utf-8",
    )
    out = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={PROFILE}",
            "--virtual-time-budget=10000",
            "--window-size=1200,900",
            "--dump-dom",
            probe.as_uri(),
        ],
        capture_output=True,
        timeout=240,
        encoding="utf-8",
        errors="replace",
    ).stdout
    m = re.search(r'<pre id="contrast-report">(.*?)</pre>', out, re.S)
    if not m:
        return ["contrast probe produced no report"]
    return json.loads(h.unescape(m.group(1)))


def check_pdf(path):
    doc = pdfium.PdfDocument(str(path))
    problems = []
    for i in range(len(doc)):
        page = doc[i]
        pw, ph = page.get_size()
        if i == 0:
            continue  # the full-bleed cover is checked by eye
        tp = page.get_textpage()
        n = tp.count_chars()
        worst = None
        for c in range(n):
            ch = tp.get_text_range(c, 1)
            if not ch.strip():
                continue
            l, b, r, t = tp.get_charbox(c)
            if r > pw - 0.70 * 72 or l < 0.70 * 72 - 1:
                worst = (ch, l, r, b)
                break
        if worst:
            problems.append(
                f"page {i + 1}: text at x={worst[1]:.0f}..{worst[2]:.0f}pt beyond the margin ('{worst[0]}')"
            )
    if len(doc) % 2:
        problems.append(
            f"{len(doc)} pages: an odd count puts the back cover on a front, not the back of the last sheet"
        )
    return problems, len(doc)


def sheet(pdf, out_dir, per=6, scale=0.55):
    from PIL import Image

    doc = pdfium.PdfDocument(str(pdf))
    imgs = [doc[i].render(scale=scale).to_pil() for i in range(len(doc))]
    w, h = imgs[0].size
    outs = []
    for s in range(0, len(imgs), per):
        batch = imgs[s : s + per]
        cols = 3
        rows = (len(batch) + cols - 1) // cols
        canvas = Image.new(
            "RGB",
            (cols * w + (cols + 1) * 8, rows * h + (rows + 1) * 8),
            (120, 120, 120),
        )
        for j, im in enumerate(batch):
            canvas.paste(im, (8 + (j % cols) * (w + 8), 8 + (j // cols) * (h + 8)))
        p = Path(out_dir) / f"sheet-{s // per + 1:02d}.png"
        canvas.save(p)
        outs.append(p.name)
    return outs


if __name__ == "__main__":
    html = Path(sys.argv[1]).read_text(encoding="utf-8")
    svg = check_svgs(html)
    import html as _h

    pre_long = []
    for m in re.finditer(r"<pre><code>(.*?)</code></pre>", html, re.S):
        for line in _h.unescape(re.sub(r"<[^>]+>", "", m.group(1))).splitlines():
            if len(line) > 96:
                pre_long.append(f"{len(line)} chars: {line[:60]}...")
    print(f"Code lines over 96 chars (would wrap): {len(pre_long)}")
    for p in pre_long:
        print("  ", p)
    pdfp, n = check_pdf(sys.argv[2])
    dom = check_dom(sys.argv[1])
    from pypdf import PdfReader

    foreign = set()
    for pg in PdfReader(sys.argv[2]).pages:
        res = pg.get("/Resources") or {}
        fonts = res.get("/Font") or {}
        for f in fonts.values():
            bf = str(f.get_object().get("/BaseFont"))
            if "IBMPlex" not in bf:
                foreign.add(bf)
        for xo in (res.get("/XObject") or {}).values():
            xr = xo.get_object().get("/Resources") or {}
            for f in (xr.get("/Font") or {}).values():
                bf = str(f.get_object().get("/BaseFont"))
                if "IBMPlex" not in bf:
                    foreign.add(bf)
    print(f"Non-Plex fonts embedded: {len(foreign)} {sorted(foreign)}")
    print(f"SVG problems: {len(svg)}")
    for p in svg:
        print("  ", p)
    print(f"DOM overflow problems: {len(dom)}")
    for p in dom:
        print("  ", p)
    low = check_contrast(sys.argv[1])
    print(f"Contrast problems: {len(low)}")
    for p in low[:40]:
        print("  ", p)
    print(f"PDF pages: {n}, edge problems: {len(pdfp)}")
    for p in pdfp:
        print("  ", p)
    if len(sys.argv) > 3:
        print("sheets:", sheet(sys.argv[2], Path(sys.argv[2]).parent))
    total = len(pre_long) + len(foreign) + len(svg) + len(dom) + len(low) + len(pdfp)
    print("OK" if total == 0 else f"FAILED: {total} problem(s)")
    sys.exit(1 if total else 0)
