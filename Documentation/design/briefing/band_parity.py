"""Header/footer parity: the text in the top and bottom 0.6in of every page
must be identical between a light edition and its dark twin (same pagination),
and the front and back covers must carry none. Exits 1 otherwise.
usage: band_parity.py light.pdf dark.pdf"""

import sys

import pypdfium2 as p


def bands(path):
    d = p.PdfDocument(path)
    out = []
    for i in range(len(d)):
        pg = d[i]
        tp = pg.get_textpage()
        W, H = pg.get_size()
        top, bottom = [], []
        for c in range(tp.count_chars()):
            ch = tp.get_text_range(c, 1)
            if not ch.strip():
                continue
            l, b, r, t = tp.get_charbox(c)
            if b > H - 0.6 * 72:
                top.append(ch)
            elif t < 0.6 * 72:
                bottom.append(ch)
        out.append(("".join(top), "".join(bottom)))
    return out


light, dark = bands(sys.argv[1]), bands(sys.argv[2])
assert len(light) == len(dark), (len(light), len(dark))
diff = [(i + 1, l, d) for i, (l, d) in enumerate(zip(light, dark)) if l != d]
print(f"pages {len(light)}, header/footer mismatches: {len(diff)}")
for n, l, d in diff[:12]:
    print(f"  page {n}: light={l!r} dark={d!r}")
covers = {1: light[0], len(light): light[-1]}
print("cover bands (light):", {k: v for k, v in covers.items()})
cover_text = any(any(band) for band in covers.values())
if cover_text:
    print("FAILED: a cover page carries header or footer text")
sys.exit(1 if diff or cover_text else 0)
