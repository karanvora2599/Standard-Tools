"""
The surface layers run against their own audit log, not the machine's.

WHY THIS EXISTS. Every `dispatch()` call appends a decision record to
`SQT_AUDIT_DIR`, which defaults to `~/.cache/standard_quant_tools/audit`.
The surface suite dispatches every tool in the library several times over,
so it is one of the largest writers to that directory -- and three `meta`
tools then READ it back: `verify_audit_integrity` walks the hash chain,
and `explain_decision` and `replay_decision` scan for a request_id.

That makes the suite's runtime a function of how much the developer has
run it before. Measured here after a day of repeated runs: a 375 MB audit
directory, 355 MB of it written that same day, and three tools costing
15.4s, 7.8s and 7.2s for ONE call each. Against a fresh directory the same
three are milliseconds. `25_testing.md` records this layer at ~90 seconds,
which is what it costs on a clean machine and nowhere near what it had
grown to.

The failure mode is worse than slow. A suite that reads accumulated state
is not reproducible: `export_audit_bundle` called twice in one session
legitimately returns different bytes as records land between the calls,
which is a real non-determinism that has nothing to do with the code under
test. Isolating the directory removes both problems at once, and it is the
correct scope anyway -- these tests are about the tool surface, not about
whatever happens to be in a developer's cache.

The fixture is session-scoped and autouse: no test in this package should
have to remember to ask for it, because forgetting is silent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_audit_log(tmp_path_factory: pytest.TempPathFactory):
    """
    Point the audit trail at a throwaway directory for this session.

    `audit.paths._audit_dir()` reads the environment on every call rather
    than caching it, so setting the variable here is enough -- nothing has
    to be imported in a particular order for it to take effect.
    """
    previous = os.environ.get("SQT_AUDIT_DIR")
    scratch: Path = tmp_path_factory.mktemp("sqt-surface-audit")
    os.environ["SQT_AUDIT_DIR"] = str(scratch)
    try:
        yield scratch
    finally:
        if previous is None:
            os.environ.pop("SQT_AUDIT_DIR", None)
        else:
            os.environ["SQT_AUDIT_DIR"] = previous
