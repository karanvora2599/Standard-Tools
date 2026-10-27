"""The hexagon lattice, with the colour in the strokes and nowhere else.

Flat-top hexagons, tile 3r x sqrt(3) r, every edge drawn once in the
theme's gridline colour, then stroked again with corner-anchored gradients
that fade to transparent. The edges are one path, drawn three times by
reference, so a lattice costs its geometry once.
"""

import math

SKY, AQUA, GREEN, AMBER = "#5ac5fa", "#75fbe1", "#75fb6d", "#f5be41"


def edges(w, h, r):
    seen = set()
    dx, dy = 1.5 * r, math.sqrt(3) * r
    for col in range(-1, int(w / dx) + 2):
        for row in range(-1, int(h / dy) + 2):
            cx = col * dx
            cy = row * dy + (dy / 2 if col % 2 else 0)
            pts = [
                (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
                for a in range(0, 360, 60)
            ]
            for i in range(6):
                a, b = pts[i], pts[(i + 1) % 6]
                key = tuple(
                    sorted(
                        (
                            (round(a[0], 1), round(a[1], 1)),
                            (round(b[0], 1), round(b[1], 1)),
                        )
                    )
                )
                if key not in seen:
                    seen.add(key)
                    yield key


def _stops(stops):
    return "\n".join(
        f'      <stop offset="{o}" stop-color="{c}" stop-opacity="{a:.3f}"/>'
        for o, c, a in stops
    )


def light_stops(peak_sky, peak_amber):
    return (
        [(0, SKY, peak_sky), (0.38, AQUA, peak_sky * 0.58), (0.8, AQUA, 0)],
        [(0, AMBER, peak_amber), (0.42, AMBER, peak_amber * 0.47), (0.82, AMBER, 0)],
    )


def dark_stops(peak_sky, peak_amber):
    # In the dark palette the glow is the accent green; it joins the sky corner.
    return (
        [
            (0, SKY, peak_sky),
            (0.34, AQUA, peak_sky * 0.62),
            (0.58, GREEN, peak_sky * 0.3),
            (0.82, GREEN, 0),
        ],
        [(0, AMBER, peak_amber), (0.42, AMBER, peak_amber * 0.45), (0.82, AMBER, 0)],
    )


def _num(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def group(name, w, h, *, r, line, line_w, colour_w, sky, amber):
    """The lattice as <defs> + <g>, to place inside a larger SVG.
    `sky`/`amber` are ((cx, cy, radius), stops)."""
    # Edges that lie wholly outside the canvas are clipped anyway; dropping
    # them keeps the file small.
    d = " ".join(
        f"M{_num(a[0])} {_num(a[1])}L{_num(b[0])} {_num(b[1])}"
        for a, b in edges(w, h, r)
        if not (max(a[0], b[0]) < 0 or min(a[0], b[0]) > w)
        and not (max(a[1], b[1]) < 0 or min(a[1], b[1]) > h)
    )
    (sx, sy, sr), sky_stops = sky
    (ax, ay, ar), amber_stops = amber
    return f"""<defs>
    <radialGradient id="{name}-sky" gradientUnits="userSpaceOnUse" cx="{sx}" cy="{sy}" r="{sr}">
{_stops(sky_stops)}
    </radialGradient>
    <radialGradient id="{name}-amber" gradientUnits="userSpaceOnUse" cx="{ax}" cy="{ay}" r="{ar}">
{_stops(amber_stops)}
    </radialGradient>
    <path id="{name}-edges" d="{d}"/>
  </defs>
  <g fill="none" stroke-linecap="round">
    <use href="#{name}-edges" stroke="{line}" stroke-width="{line_w}"/>
    <use href="#{name}-edges" stroke="url(#{name}-sky)" stroke-width="{colour_w}"/>
    <use href="#{name}-edges" stroke="url(#{name}-amber)" stroke-width="{colour_w}"/>
  </g>"""
