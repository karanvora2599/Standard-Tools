"""Generate the README and documentation images.

    python Documentation/design/gen_assets.py

Writes to Documentation/assets/, each image twice (light and dark) except
the numbered chips, which read on both:

  banner        the README's title image
  docs-banner   the title image of Documentation/README.md
  header        the strip at the top of every guide
  pipeline      how a call runs, in the README
  glance        the README's at-a-glance tiles
  chip-NN       the numbered chips beside headings, 00 to 40

An SVG shown as an <img> (GitHub) cannot load fonts of its own, so each
image carries IBM Plex subset to the characters it uses. Every label is
measured with the real font metrics, and the script stops if one overruns
its box. Each image's <title> is its alt text; a test holds the README's
and the guides' alt text to it.
"""

import base64
import io
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

import hexlattice as hx

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
FONTS = HERE / "fonts"

LIGHT = dict(
    paper="#efefed", panel="#ffffff", ink="#1a1a1a", ink2="#3d3d3d", muted="#555555",
    rule="#d8d5ce", red="#b3261e", on_colour="#1a1a1a", on_red="#ffffff",
    blue="#5bbff4", mint="#74f2d6", green="#79ee5f", amber="#f2b43c",
    signal_red="#d6352a", red_deep="#a8231b", box_rule="#d8d5ce", inset="#efefed",
)  # fmt: skip
DARK = dict(
    paper="#0b0b0b", panel="#131313", ink="#efefed", ink2="#a4a4a4", muted="#a4a4a4",
    rule="#2a2a2a", red="#f25144", on_colour="#0b0b0b", on_red="#0b0b0b",
    blue="#5ac5fa", mint="#75fbe1", green="#75fb6d", amber="#f5be41",
    signal_red="#dc2f21", red_deep="#a62116", box_rule="#444444", inset="#1a1a1a",
)  # fmt: skip

FACES = {  # (family, weight) -> file
    ("Plex", 300): "IBMPlexSans-Light.woff2",
    ("Plex", 400): "IBMPlexSans-Regular.woff2",
    ("Plex", 500): "IBMPlexSans-Medium.woff2",
    ("Plex", 600): "IBMPlexSans-SemiBold.woff2",
    ("PlexMono", 400): "IBMPlexMono-Regular.woff2",
    ("PlexMono", 500): "IBMPlexMono-Medium.woff2",
    ("PlexMono", 600): "IBMPlexMono-SemiBold.woff2",
}
_metrics = {}

NAME = "Standard Quant Tools"
SOURCE = "github.com/karanvora2599/Standard-Tools"


def width(s, family, weight, size, spacing=0.0):
    """Advance width of `s` in px, as a browser lays it out (no kerning)."""
    name = FACES[(family, weight)]
    if name not in _metrics:
        f = TTFont(FONTS / name)
        _metrics[name] = (f["hmtx"].metrics, f.getBestCmap(), f["head"].unitsPerEm)
    hmtx, cmap, upm = _metrics[name]
    missing = [c for c in s if ord(c) not in cmap]
    assert not missing, f"{name} has no glyph for {missing}"
    advance = sum(hmtx[cmap[ord(c)]][0] for c in s) / upm * size
    return advance + spacing * size * max(len(s) - 1, 0)


def esc(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class Canvas:
    """An SVG being drawn. Records every (family, weight) and character used,
    so each image embeds exactly the glyphs it needs."""

    def __init__(self, w, h, p):
        self.w, self.h, self.p = w, h, p
        self.parts = []
        self.used = {}

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, sw=1.0, rx=0):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}"{st}/>'
        )

    def text(
        self, x, y, s, *, family="Plex", weight=400, size=14, fill,
        anchor="start", spacing=0.0, box=None,
    ):  # fmt: skip
        """`box` = (x0, x1): fail unless the text fits inside it."""
        w = width(s, family, weight, size, spacing)
        x0 = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        if box:
            assert box[0] - 0.5 <= x0 and x0 + w <= box[1] + 0.5, (
                f"label {s!r} ({x0:.0f}..{x0 + w:.0f}) overruns its box "
                f"{box[0]:.0f}..{box[1]:.0f}"
            )
        assert 0 <= x0 and x0 + w <= self.w, f"label {s!r} outside the canvas"
        self.used.setdefault((family, weight), set()).update(s)
        ls = f' letter-spacing="{spacing * size:.2f}"' if spacing else ""
        an = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-weight="{weight}" font-size="{size}"{ls}{an} '
            f'fill="{fill}">{esc(s)}</text>'
        )
        return w

    def plate(self, x, y, s, size, family, weight, spacing=0.0, anchor="start", pad=6):
        """A rectangle in the paper colour behind a label that sits on the
        lattice, so the lines stop at the letters."""
        w = width(s, family, weight, size, spacing)
        x0 = x - w if anchor == "end" else x
        self.rect(x0 - pad, y - size * 0.95, w + 2 * pad, size * 1.3, self.p["paper"])

    def svg(self, *, label):
        faces = []
        for (family, weight), chars in sorted(self.used.items()):
            data = subset_font(FACES[(family, weight)], chars)
            faces.append(
                f'@font-face{{font-family:"{family}";font-weight:{weight};'
                f'src:url(data:font/woff2;base64,{data}) format("woff2");}}'
            )
        style = "<style>" + "".join(faces) + "</style>" if faces else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" role="img" aria-label="{esc(label)}" '
            f'font-family="Plex" font-size="14"><title>{esc(label)}</title>{style}\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


def subset_font(name, chars):
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["kern", "liga"]
    opts.name_IDs = []
    opts.notdef_outline = True
    f = TTFont(FONTS / name)
    s = subset.Subsetter(options=opts)
    s.populate(text="".join(sorted(chars)) + " ")
    s.subset(f)
    buf = io.BytesIO()
    f.flavor = "woff2"
    f.save(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def bars(c, x, y, h, w=10, gap=11, step=9):
    """The mark: blue, mint, green, then amber, red, deep red, each of the
    last three a step further apart. Returns the x where the wordmark
    starts."""
    p = c.p
    colours = [p["blue"], p["mint"], p["green"], p["amber"], p["signal_red"], p["red_deep"]]
    for i, col in enumerate(colours):
        c.rect(x + i * (w + gap) + step * max(0, i - 2), y, w, h, col, rx=1.5)
    return x + 6 * (w + gap) + 3 * step + 2 * gap


def wordmark(c, x, baseline, size):
    """`Standard` in the regular face and `Quant Tools` in the light one.
    Returns the x where it ends."""
    p = c.p
    first = c.text(x, baseline, "Standard", weight=400, size=size, spacing=-0.03, fill=p["ink"])
    gap = size * 0.24
    rest = c.text(
        x + first + gap, baseline, "Quant Tools", weight=300, size=size,
        spacing=-0.03, fill=p["ink2"],
    )  # fmt: skip
    return x + first + gap + rest


# The five things that happen to every call, left to right. The banner's
# band and the pipeline figure name them in the same order and colours.
STAGES = [
    ("SERVE", "blue", ["one runtime per client", "MCP or function calling",
                       "thinned schemas under auto", "describe_tool on demand"]),
    ("DISPATCH", "mint", ["Pydantic-validated input", "unknown arguments refused",
                          "foreign tools unroutable", "errors that name the fix"]),
    ("COMPUTE", "green", ["C++ kernels when built", "Python fallback, same API",
                          "point-in-time contracts", "a stated risk-free rate"]),
    ("HAND OFF", "signal_red", ["bulk values as sqt:// refs", "never through the chat",
                                "kind checked on resolve", "read in any runtime"]),
    ("RECORD", "amber", ["a decision record per call", "content hashes of the data",
                         "hash-chained across days", "sqt replay and verify"]),
]  # fmt: skip


def layer_band(c, x, y, w, h, size=13):
    p = c.p
    cw = w / len(STAGES)
    c.rect(x, y, w, h, "none", stroke=p["box_rule"], sw=1)
    for i, (name, key, _items) in enumerate(STAGES):
        cx = x + i * cw
        c.rect(cx, y, cw, h, p[key])
        fg = p["on_red"] if key == "signal_red" else p["on_colour"]
        c.text(
            cx + 14, y + h / 2 + size * 0.36, name, family="PlexMono", weight=500,
            size=size, fill=fg, spacing=0.1, box=(cx, cx + cw),
        )  # fmt: skip


def lattice(c, name, dark, *, r=22, peaks=(0.95, 0.95), dark_peaks=(0.85, 0.8),
            line=("#dadad5", "#1f1f1f"), line_w=1.0, colour_w=1.4,
            sky=(20, -40, 760), amber=(-40, 30, 640)):  # fmt: skip
    """The lattice over the whole canvas; `sky` and `amber` place their
    gradients relative to the top-right and bottom-left corners."""
    W, H = c.w, c.h
    s, a = hx.dark_stops(*dark_peaks) if dark else hx.light_stops(*peaks)
    c.add(
        hx.group(
            name + ("-dark" if dark else ""), W, H, r=r, line=line[dark],
            line_w=line_w, colour_w=colour_w,
            sky=((W + sky[0], sky[1], sky[2]), s),
            amber=((amber[0], H + amber[1], amber[2]), a),
        )
    )  # fmt: skip


# ---------------------------------------------------------------- the banners


def banner(p, dark, *, meta, lines):
    W, H = 1280, 640
    c = Canvas(W, H, p)
    c.rect(0, 0, W, H, p["paper"])
    lattice(c, "banner", dark)
    for txt, x, an in ((meta[0], 80, "start"), (meta[1], W - 80, "end")):
        c.plate(x, 64, txt, 13, "PlexMono", 500, 0.14, an)
        c.text(x, 64, txt, family="PlexMono", weight=500, size=13, spacing=0.14,
               fill=p["muted"], anchor=an)  # fmt: skip

    # The lattice runs behind the wordmark and the large lines; only the
    # small meta and footer text sit on plates.
    x = bars(c, 80, 150, 94, w=8, gap=9, step=7)
    end = wordmark(c, x, 228, 104)
    assert end <= W - 80, f"the wordmark ends at {end:.0f}, past the margin"

    sizes = [("Plex", 300, 32, p["ink"], 314), ("Plex", 400, 38, p["red"], 370),
             ("PlexMono", 400, 14, p["ink2"], 420)]  # fmt: skip
    for txt, (fam, wt, size, fill, y) in zip(lines, sizes):
        c.text(80, y, txt, family=fam, weight=wt, size=size, fill=fill, box=(80, W - 80))

    layer_band(c, 80, 496, W - 160, 46)
    for txt, x, an, wt in (
        (f"{NAME}  ·  by Karan Vora  ·  Apache 2.0", 80, "start", 400),
        (SOURCE, W - 80, "end", 600),
    ):
        c.plate(x, 590, txt, 15, "Plex", wt, anchor=an)
        c.text(x, 590, txt, weight=wt, size=15,
               fill=p["ink2"] if wt == 400 else p["ink"], anchor=an)  # fmt: skip
    return c


README_LINES = (
    "Quantitative finance for AI agents, in Python with a C++ core",
    "Measured, Not Projected",
    "Typed tools  ·  One runtime per agent  ·  Point-in-time by declaration"
    "  ·  Every call on the record",
)
README_LABEL = (
    "Standard Quant Tools: quantitative finance for AI agents, in Python "
    "with a C++ core. Measured, not projected."
)
DOCS_LINES = (
    "The documentation: a guide for every part of the library",
    "Measured, Not Projected",
    "Generated references  ·  Counts checked by the suite  ·  What did not pay"
    " off, kept beside what did",
)
DOCS_LABEL = (
    "Standard Quant Tools documentation: a guide for every part of the "
    "library. Measured, not projected."
)


# ------------------------------------------------------- the strip on a guide

HEADER_LABEL = "Standard Quant Tools documentation"


def header(p, dark):
    W, H = 1280, 112
    c = Canvas(W, H, p)
    c.rect(0, 0, W, H, p["paper"])
    lattice(c, "header", dark, r=18, peaks=(0.8, 0.7), dark_peaks=(0.7, 0.6),
            line=("#e2e2dd", "#191919"), line_w=0.8, colour_w=1.2,
            sky=(10, -60, 520), amber=(-30, 60, 460))  # fmt: skip
    size, baseline = 30, 66
    pad = 10
    x0 = 48
    # The wordmark is small here, so it sits on a plate as the meta does.
    mark_w = 6 * (5 + 6) + 3 * 4 + 2 * 6
    words = width("Standard", "Plex", 400, size, -0.03) + size * 0.24
    words += width("Quant Tools", "Plex", 300, size, -0.03)
    c.rect(x0 - pad, 30, mark_w + words + 2 * pad, 52, p["paper"])
    x = bars(c, x0, 36, 40, w=5, gap=6, step=4)
    wordmark(c, x, baseline, size)
    meta = "DOCUMENTATION"
    c.plate(W - 48, 62, meta, 13, "PlexMono", 500, 0.14, "end")
    c.text(W - 48, 62, meta, family="PlexMono", weight=500, size=13, spacing=0.14,
           fill=p["muted"], anchor="end")  # fmt: skip
    return c


# -------------------------------------------------------------- the pipeline

PIPELINE_LABEL = (
    "How a call runs: serve, dispatch, compute, hand off, record. From "
    "Python one tool at a time; over MCP one runtime per session."
)


def pipeline(p, dark):
    W, H = 1200, 372
    c = Canvas(W, H, p)
    c.rect(0, 0, W, H, p["panel"])
    m, gap = 20, 30
    cw = (W - 2 * m - 4 * gap) / 5
    top, ch = 20, 236
    for i, (name, key, items) in enumerate(STAGES):
        x = m + i * (cw + gap)
        c.rect(x, top, cw, ch, p["panel"], stroke=p["box_rule"], sw=1)
        c.rect(x, top, cw, 6, p[key])
        c.text(x + 14, top + 34, f"{i + 1:02d}  {name}", family="PlexMono", weight=600,
               size=13, spacing=0.12, fill=p["ink"], box=(x, x + cw))  # fmt: skip
        for j, item in enumerate(items):
            c.text(x + 14, top + 72 + j * 40, item, size=14, fill=p["ink"],
                   box=(x + 4, x + cw - 4))  # fmt: skip
        if i < 4:  # a chevron in the gap
            ax, ay = x + cw + gap / 2, top + ch / 2
            c.add(
                f'<path d="M{ax - 5:.1f} {ay - 8:.1f}L{ax + 4:.1f} {ay:.1f}'
                f'L{ax - 5:.1f} {ay + 8:.1f}" fill="none" stroke="{p["muted"]}" '
                f'stroke-width="2"/>'
            )
    y = top + ch + 20
    half = (W - 2 * m - gap) / 2
    for k, (label, txt) in enumerate(
        (
            ("PYTHON", "import a tool: one Pydantic model in, one Pydantic model out"),
            ("MCP", "sqt-mcp --runtime research serves one runtime to any client"),
        )
    ):
        x = m + k * (half + gap)
        c.rect(x, y, half, 76, p["inset"])
        c.text(x + 16, y + 30, label, family="PlexMono", weight=600, size=13,
               spacing=0.12, fill=p["red"], box=(x, x + half))  # fmt: skip
        c.text(x + 16, y + 56, txt, size=15, fill=p["ink"], box=(x, x + half))
    return c


# ------------------------------------------------------------ at a glance

#: (figure, colour, what it is). Every figure is one the README and the
#: guides state, measured where they say: the speedups in
#: 16_performance.md, the test layers and mutations in 25_testing.md.
GLANCE = [
    ("274×", "blue", "rolling Hurst with the C++ extension, against the same module's Python path"),
    ("1.7 min", "green", "to test 2,000 names for cointegration at 500 bars: both orders of all 1,999,000 pairs"),
    ("~58×", "signal_red", "the backtest kernel, timed by toggling HAS_CPP on the same call"),
    ("12", "mint", "runtimes, each refusing every tool it does not own"),
    ("9", "amber", "test layers, each catching a class of defect the others cannot"),
    ("21 of 21", "ink", "planted mutations killed: the suite notices a bug, not only runs green"),
]  # fmt: skip
GLANCE_LABEL = (
    "At a glance: 274× for rolling Hurst with the C++ extension; 1.7 min to "
    "test all 1,999,000 pairs of 2,000 names for cointegration at 500 bars; "
    "~58× for the backtest kernel; 12 runtimes; 9 test layers; 21 of 21 "
    "planted mutations killed"
)


def wrap(s, family, weight, size, w):
    words, lines, cur = s.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if width(trial, family, weight, size) <= w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def glance(p, dark):
    W, H = 1200, 360
    c = Canvas(W, H, p)
    cols, gx, gy = 3, 36, 30
    tw = (W - (cols - 1) * gx) / cols
    th = (H - gy) / 2
    for i, (n, key, desc) in enumerate(GLANCE):
        x, y = (i % cols) * (tw + gx), (i // cols) * (th + gy)
        c.rect(x, y, tw, 5, p[key])
        c.text(x, y + 56, n, family="PlexMono", weight=500, size=42, spacing=-0.02,
               fill=p["ink"], box=(x, x + tw))  # fmt: skip
        lines = wrap(desc, "Plex", 400, 16, tw)
        assert len(lines) <= 2, f"tile {n!r}: description wraps to {len(lines)} lines"
        for j, line in enumerate(lines):
            c.text(x, y + 92 + j * 23, line, size=16, fill=p["ink2"], box=(x, x + tw))
    return c


# -------------------------------------------------------------------- chips

CHIP_COLOURS = ["blue", "mint", "green", "amber", "signal_red"]
CHIPS = range(0, 41)


def chip(n, p):
    c = Canvas(30, 26, p)
    key = CHIP_COLOURS[(n - 1) % len(CHIP_COLOURS)]
    c.rect(0, 0, 30, 26, p[key])
    c.text(15, 17.6, f"{n:02d}", family="PlexMono", weight=600, size=12, anchor="middle",
           fill=p["on_red"] if key == "signal_red" else p["on_colour"])  # fmt: skip
    return c


def main():
    ASSETS.mkdir(exist_ok=True)
    for p, dark, suffix in ((LIGHT, False, "light"), (DARK, True, "dark")):
        images = {
            "banner": (
                banner(p, dark, meta=("OPEN SOURCE  ·  APACHE 2.0", "PYTHON  ·  C++17  ·  MCP"),
                       lines=README_LINES),
                README_LABEL,
            ),
            "docs-banner": (
                banner(p, dark, meta=("OPEN SOURCE  ·  APACHE 2.0", "DOCUMENTATION"),
                       lines=DOCS_LINES),
                DOCS_LABEL,
            ),
            "header": (header(p, dark), HEADER_LABEL),
            "pipeline": (pipeline(p, dark), PIPELINE_LABEL),
            "glance": (glance(p, dark), GLANCE_LABEL),
        }  # fmt: skip
        for name, (canvas, label) in images.items():
            (ASSETS / f"{name}-{suffix}.svg").write_text(
                canvas.svg(label=label), encoding="utf-8", newline="\n"
            )
    for n in CHIPS:
        (ASSETS / f"chip-{n:02d}.svg").write_text(
            chip(n, LIGHT).svg(label=f"{n:02d}"), encoding="utf-8", newline="\n"
        )
    total = 0
    for f in sorted(ASSETS.glob("*.svg")):
        total += f.stat().st_size
        if not f.name.startswith("chip-"):
            print(f"{f.relative_to(HERE.parent)}: {f.stat().st_size // 1024} KB")
    print(f"{len(CHIPS)} chips; {total // 1024} KB in all")


if __name__ == "__main__":
    main()
