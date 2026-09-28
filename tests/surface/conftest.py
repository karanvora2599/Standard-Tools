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
three are milliseconds. `25_testing.md` records what each layer costs on a
clean machine, which is nowhere near what an accumulated directory had
grown it to.

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
        # Every call this layer made -- thousands of them, hostile inputs
        # included -- wrote a decision record here. A NaN in an input once
        # made its record fail its own hash, so the day read as tampered
        # forever; one verification over the whole trail pins that for
        # every tool at the cost of one call.
        from standard_quant_tools.audit.verify import verify_audit_trail_integrity

        problems = verify_audit_trail_integrity(scratch)
        assert not problems, (
            "the audit trail the surface layer wrote does not verify: "
            f"{problems[:5]}"
        )
    finally:
        if previous is None:
            os.environ.pop("SQT_AUDIT_DIR", None)
        else:
            os.environ["SQT_AUDIT_DIR"] = previous


@pytest.fixture(scope="session", autouse=True)
def _hermetic_market():
    """
    Every fetch in this layer is answered by `hermetic.FakeTicker`.

    Session-wide, so the determinism and invariant layers see the same bars
    the adversarial one does, and none of them depends on a connection or on
    what the market did today. The provider's own code still runs; only the
    call to yfinance is replaced. See `hermetic.install` for the caches.
    """
    from . import hermetic

    patch = pytest.MonkeyPatch()
    hermetic.install(patch)
    try:
        yield
    finally:
        patch.undo()
        hermetic.uninstall()


@pytest.fixture(scope="session")
def published(tmp_path_factory: pytest.TempPathFactory):
    """The references, dataset, models and record ids a baseline can name,
    built once per session (about twenty seconds)."""
    from . import hermetic

    return hermetic.publish_fixtures(
        str(tmp_path_factory.mktemp("sqt-surface-external"))
    )
