#!/usr/bin/env python3
"""
Run the standalone audit verifier from a source checkout.

The verifier is `src/standard_quant_tools/audit/verify_audit_log.py`. It
lives inside the package so that every exported bundle carries it, however
the package was installed: a wheel or a git install has no `scripts/`
directory, and a bundle exported from one used to ship without the verifier
its own README tells the reader to run. This file runs that one by path and
imports nothing from the package, so it keeps the verifier's promise of the
standard library only:

    python scripts/verify_audit_log.py <audit_dir>
    python scripts/verify_audit_log.py --file <path.jsonl>
"""

import runpy
from pathlib import Path

VERIFIER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "standard_quant_tools"
    / "audit"
    / "verify_audit_log.py"
)

if __name__ == "__main__":
    runpy.run_path(str(VERIFIER), run_name="__main__")
