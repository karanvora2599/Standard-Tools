"""The briefing's lattices and charts, written to charts/.

    python gen_charts.py

Lattices: cover, back cover and page, light and dark, US Letter. Charts are
print-only, so values are labelled directly. The tools-per-runtime chart is
read from the live registry; the speedups are the measured figures of
Documentation/16_performance.md.
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CHARTS = HERE / "charts"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "src"))

import hexlattice as hx  # noqa: E402

INK, INK2, MUTED, GRID = "#1a1a1a", "#3d3d3d", "#6b6b6b", "#e4e1da"
FONT, MONO = 'font-family="Plex"', 'font-family="PlexMono"'


def lattice(name, w, h, **kw):
    """A whole-sheet lattice. Its class tells the dark build to leave it alone."""
    return (
        f'<svg class="hex-lattice" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'preserveAspectRatio="xMidYMid slice" aria-hidden="true">\n  {hx.group(name, w, h, **kw)}\n</svg>\n'
    )


def lattices():
    W, H = 816, 1056  # US Letter in CSS px
    for dark in (False, True):
        sfx = "-dark" if dark else ""
        s, a = hx.dark_stops(0.85, 0.8) if dark else hx.light_stops(0.95, 0.95)
        line = "#1f1f1f" if dark else "#dadad5"
        for name in ("cover-hex", "back-hex"):
            (CHARTS / f"{name}{sfx}.svg").write_text(
                lattice(name + sfx, W, H, r=19, line=line, line_w=0.8, colour_w=1.15,
                        sky=((W + 10, -30, 600), s), amber=((-30, H + 20, 540), a)),
                encoding="utf-8", newline="\n",
            )  # fmt: skip
        s, a = hx.dark_stops(0.4, 0.34) if dark else hx.light_stops(0.62, 0.52)
        (CHARTS / f"page-hex{sfx}.svg").write_text(
            lattice("page-hex" + sfx, W, H, r=19, line="#161616" if dark else "#e9e9e4",
                    line_w=0.55, colour_w=0.85, sky=((W + 110, -110, 480), s),
                    amber=((-120, H + 120, 460), a)),
            encoding="utf-8", newline="\n",
        )  # fmt: skip


def bar(x, y, w, h, fill, r=3):
    """A bar with its far end rounded, drawn from x."""
    r = min(r, w / 2, h / 2)
    return (
        f'<path d="M{x:.1f} {y:.1f} H{x + w - r:.1f} Q{x + w:.1f} {y:.1f} {x + w:.1f} {y + r:.1f} '
        f'V{y + h - r:.1f} Q{x + w:.1f} {y + h:.1f} {x + w - r:.1f} {y + h:.1f} H{x:.1f} Z" fill="{fill}"/>'
    )


def runtimes():
    from standard_quant_tools.agent.runtimes import all_runtimes

    data = sorted(
        ((n, len(r)) for n, r in all_runtimes().items()), key=lambda kv: -kv[1]
    )
    W, row, left, right = 680, 19, 118, 40
    H = len(data) * row + 8
    top = max(n for _, n in data)
    scale = (W - left - right) / (math.ceil(top / 5) * 5)
    out = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" {FONT} font-size="9">'
    ]
    for i, (name, n) in enumerate(data):
        y = 4 + i * row
        out.append(
            f'<text x="{left - 8}" y="{y + 11}" text-anchor="end" {MONO} font-size="8.4" fill="{INK2}">{name}</text>'
        )
        out.append(bar(left, y + 2, n * scale, 12, "#f2b43c"))
        out.append(
            f'<text x="{left + n * scale + 5:.1f}" y="{y + 11.5}" {MONO} font-size="8.4" fill="{INK}">{n}</text>'
        )
    out.append(
        f'<line x1="{left}" y1="2" x2="{left}" y2="{H - 2}" stroke="{INK}" stroke-width="0.8"/>'
    )
    out.append("</svg>")
    (CHARTS / "runtimes.svg").write_text("\n".join(out), encoding="utf-8", newline="\n")
    return sum(n for _, n in data), len(data)


# (operation, low, high) from 16_performance.md: the C++ path against the
# Python path it replaces, the same call on the same machine. A range is the
# spread over the sizes measured; the bar runs to its low end.
SPEEDUPS = [
    ("rolling_hurst", 274, 274),
    ("scan_cointegrated_pairs", 111, 111),
    ("permutation_null_ic", 68, 88),
    ("run_strategy", 58, 58),
    ("cointegration_test", 37, 41),
    ("binomial_lattice", 28, 28),
    ("cusum_peaks", 20, 20),
    ("order_event_metrics", 17, 17),
    ("technical_indicators_panel", 11.9, 11.9),
    ("batch_run_strategy", 11, 11),
    ("run_portfolio_simulation", 5.3, 5.3),
    ("rolling_beta", 4.7, 4.7),
    ("hierarchical_risk_parity", 4.5, 4.5),
    ("stochastic_oscillator", 2.6, 2.6),
    ("simulate_forward_paths", 2.0, 2.0),
    ("bollinger_bands", 1.6, 1.6),
    ("rsi", 0.97, 1.05),
    ("adx", 0.94, 0.99),
    ("wilder_atr", 0.92, 1.00),
    ("garch11_variance_recursion", 0.8, 0.8),
]


def fmt(v):
    return f"{v:g}" if v < 10 else f"{v:.0f}" if v == int(v) else f"{v:.1f}"


def speedups():
    W, row, left, right, top = 680, 17, 168, 56, 18
    H = top + len(SPEEDUPS) * row + 26
    lo, hi = math.log10(0.5), math.log10(500)
    x = lambda v: left + (math.log10(v) - lo) / (hi - lo) * (
        W - left - right
    )  # noqa: E731
    out = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" {FONT} font-size="9">'
    ]
    for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        out.append(
            f'<line x1="{x(t):.1f}" y1="{top - 4}" x2="{x(t):.1f}" y2="{top + len(SPEEDUPS) * row}" stroke="{GRID}" stroke-width="0.7"/>'
        )
        out.append(
            f'<text x="{x(t):.1f}" y="{top + len(SPEEDUPS) * row + 12}" text-anchor="middle" {MONO} font-size="7.6" fill="{MUTED}">{fmt(t)}×</text>'
        )
    for i, (name, a, b) in enumerate(SPEEDUPS):
        y = top + i * row
        tie = b <= 1.05
        out.append(
            f'<text x="{left - 8}" y="{y + 10}" text-anchor="end" {MONO} font-size="7.8" fill="{INK2}">{name}</text>'
        )
        if tie:
            # A tie or a slowdown: the span from its low end to 1x, in grey.
            x0, x1 = x(min(a, 1)), x(max(b, 1))
            out.append(
                f'<rect x="{x0:.1f}" y="{y + 3}" width="{max(x1 - x0, 2):.1f}" height="9" rx="2" fill="#dedad2"/>'
            )
        else:
            out.append(bar(x(1), y + 3, x(a) - x(1), 9, "#2a78d6", r=2))
        label = f"{fmt(a)}–{fmt(b)}×" if a != b else f"{fmt(a)}×"
        verdict = "" if not tie else "  tied" if b >= 0.9 else "  slower"
        out.append(
            f'<text x="{max(x(b), x(1)) + 5:.1f}" y="{y + 10.5}" {MONO} font-size="7.6" fill="{INK}">{label}{verdict}</text>'
        )
    out.append(
        f'<line x1="{x(1):.1f}" y1="{top - 8}" x2="{x(1):.1f}" y2="{top + len(SPEEDUPS) * row}" stroke="{INK}" stroke-width="0.9" stroke-dasharray="4 3"/>'
    )
    out.append(
        f'<text x="{x(1) + 4:.1f}" y="{top - 8}" {FONT} font-size="7.6" fill="{MUTED}">1× = the Python path</text>'
    )
    out.append("</svg>")
    (CHARTS / "speedups.svg").write_text("\n".join(out), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    CHARTS.mkdir(exist_ok=True)
    lattices()
    total, n = runtimes()
    speedups()
    print(f"{total} tools in {n} runtimes;", sorted(p.name for p in CHARTS.iterdir()))
