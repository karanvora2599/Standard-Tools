"""Test package root.

`REPO_ROOT` is defined here, not recomputed per test file. Several tests
reach outside the package for files that are not importable modules — the
standalone audit verifier script, the reference agent implementations —
and each did it with its own `Path(__file__).parent.parent`, which encodes
how deep that particular file happens to sit. Grouping the suite into
per-module subdirectories moved all of them one level down and broke every
one of those chains at once. Anchoring the path here means a future move
updates one line instead of N, and the N is not discoverable until the
tests fail.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["REPO_ROOT"]
