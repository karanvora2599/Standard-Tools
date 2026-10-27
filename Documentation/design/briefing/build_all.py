"""Regenerate the charts and the catalog, build both editions of the briefing
into Documentation/, and check each one.

usage: python build_all.py

Each edition is built, then checked straight away (check.py reads the
index.html that build.py left behind), then the light and dark pair is
compared. Stops at the first failure with a non-zero exit, so a red run
never leaves a stale PDF looking current.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
LIGHT, DARK = (
    "Standard-Quant-Tools-Briefing.pdf",
    "Standard-Quant-Tools-Briefing-Dark.pdf",
)


def run(*args):
    print("\n$ python", " ".join(args), flush=True)
    if subprocess.run([sys.executable, *args], cwd=HERE).returncode:
        sys.exit(f"stopped: {args[0]} failed")


run("gen_charts.py")
run("gen_catalog.py")
for name, flags in ((LIGHT, ()), (DARK, ("dark",))):
    run("build.py", str(DOCS / name), *flags)
    run("check.py", "index.html", str(DOCS / name))
run("band_parity.py", str(DOCS / LIGHT), str(DOCS / DARK))
print("\nBoth editions built and checked.")
