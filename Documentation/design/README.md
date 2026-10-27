# Documentation/design

The sources of the images in `Documentation/assets/`: the README's banner,
pipeline and at-a-glance tiles, the documentation page's banner, the strip at
the top of every guide, and the numbered chips beside headings. `briefing/`
holds the sources of the briefing PDFs in `Documentation/` (see
[The briefing](#the-briefing) below).

## Rebuild

```bash
pip install -r Documentation/design/requirements.txt   # once
python Documentation/design/gen_assets.py
```

This rewrites every file in `Documentation/assets/`. Commit the images with
the change that made them, so no page shows a stale one. Rebuild after
changing any label, figure or stage in `gen_assets.py`.

## What is where

| path | holds |
|---|---|
| `gen_assets.py` | every image, light and dark, and the labels and figures they carry |
| `hexlattice.py` | the hexagon lattice behind the banners and the guide strip |
| `fonts/` | IBM Plex Sans and Mono, unmodified, under the SIL Open Font License (`fonts/OFL.txt`) |
| `requirements.txt` | fontTools and brotli, which read the fonts and write the subsets, and the two PDF libraries the briefing build uses |
| `briefing/` | the briefing: its pages, stylesheets, build and checks |

## How the pages use them

- **The README** opens with `banner`, then `glance` under "At a glance" and
  `pipeline` under "How it works". Its sections carry `chip-01` onwards.
- **The documentation page** (`Documentation/README.md`) opens with
  `docs-banner` and groups the guides under chips of its own.
- **Every guide** opens with `header` and two links, to the README and the
  documentation page, and its title carries the chip of its own number:
  `05_portfolio.md` has `chip-05`. It closes with the same links. The two
  generated guides get theirs from their generators.

Each image but the chips exists twice, `-light` and `-dark`, and a page
switches between them with `<picture>` and `prefers-color-scheme`. The chips
read on both. An SVG shown as an `<img>`, as GitHub shows it, cannot load
fonts of its own, so each image carries IBM Plex subset to the characters it
uses. Every label is measured with the real font metrics, and the script
stops if one overruns its box.

## What the tests hold

`tests/docs/test_documentation.py` checks that:

- each image's alt text on a page is the `<title>` of the file it shows, in
  both editions, so the words a screen reader hears are the words drawn;
- every image a page names exists;
- every guide opens with the strip and the chip of its own number;
- the runtime count on the glance tiles and the README's badges is the live
  registry's.

## The briefing

A 40-page technical briefing on the library, in the same theme, in two
editions:

| file in `Documentation/` | edition |
|---|---|
| `Standard-Quant-Tools-Briefing.pdf` | light |
| `Standard-Quant-Tools-Briefing-Dark.pdf` | dark |

The pages are HTML and CSS printed to PDF by headless Chrome. Rebuild both
editions, and check each one, with:

```bash
pip install -r Documentation/design/requirements.txt   # once
python Documentation/design/briefing/build_all.py
```

It needs Chrome; set `CHROME` if it is not at the Windows default (see
`briefing/chrome.py`). It stops with a non-zero exit at the first failure, so
a red run never leaves a stale PDF looking current. Commit the two PDFs with
the change that made them.

| path in `briefing/` | holds |
|---|---|
| `parts/` | one HTML file per part, joined in file-name order: `00` cover and contents, `05` executive summary, `10` to `50` Parts I to V, `90` Appendices A and B, `99` back cover |
| `gen_catalog.py` | Appendix C (`parts/92-catalog.html`) and section 04's runtime table, from the live registry |
| `gen_charts.py` | the page lattices, Figure 1 (tools per runtime, from the registry) and Figure 2 (the measured speedups of `16_performance.md`) |
| `build.py` | joins the parts and prints them twice, so the contents page carries real page numbers; adds a blank page when the count would be odd |
| `style.css`, `dark.css` | the print stylesheet and the dark edition's colour overrides (colour only, so both editions paginate alike) |
| `check.py`, `band_parity.py` | the checks below |

`check.py` fails an edition on a code line that would wrap, a font other than
IBM Plex embedded in the PDF, a chart label that overruns its box, an element
wider than the text column, a cover that clips its content, text below WCAG
contrast (4.5:1, or 3:1 for large text), text past the margins, or an odd page
count. `band_parity.py` fails when the two editions differ in any page's
header or footer, or when a cover carries either.

Every figure in the briefing was measured from the code on the date its cover
gives. Appendix A lists each one and where it comes from: re-measure rather
than edit, and rebuild after a tool count, a test count or a measurement in it
moves.

## Design rules

- The colour lives in the hexagon strokes, never as a wash over the page.
- The lattice is quiet: it must not compete with the text. Small text that
  sits on it gets a plate in the paper colour, so the lines stop at the
  letters.
- Dark mode uses its own palette, not an inverted one: `#0b0b0b` paper,
  `#efefed` ink, `#2a2a2a` lines, `#75fb6d` accent, `#f25144` critical.
- A figure on an image is one the pages state, measured where they say.
  The glance tiles name their sources in the README beneath them.
