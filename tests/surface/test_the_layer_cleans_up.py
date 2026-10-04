"""
Nothing this package installs outlives its tests (see the CHANGELOG entry of
2026-10-04).

`conftest.py` fakes yfinance, switches the Parquet tier off, removes the
retry backoff and moves the audit directory for the surface tests. Those
fixtures were session-scoped, so they stayed in place for every test pytest
ran after this package in the same session: `tests/surface` followed by
`tests/audit/test_audit.py::TestVerifyReplay` and
`tests/data/test_parquet_cache.py` failed 15 tests that pass on their own.

The check needs a session in which a test outside the package runs after
one inside it, whatever order collection would choose, so it starts one: the
cheapest test here, then `tests/test_surface_fixtures_end_with_the_package.py`,
which asserts the provider, the retry module and the audit directory are
the process's own again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

_INSIDE = (
    "tests/surface/test_the_layer_cleans_up.py::"
    "test_inside_the_package_the_market_is_fake"
)
_AFTER = "tests/test_surface_fixtures_end_with_the_package.py"


def test_inside_the_package_the_market_is_fake():
    """The other side of the check: inside the package the fixtures are in
    place without being asked for."""
    import standard_quant_tools.data.yfinance_provider as provider

    from . import hermetic

    assert isinstance(provider.yf, hermetic.FakeYFinance)
    assert provider._is_historical("2000-01-03") is False
    assert "sqt-surface-audit" in os.environ["SQT_AUDIT_DIR"]


def test_a_test_after_the_package_sees_the_real_provider():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            _INSIDE,
            _AFTER,
            "-p",
            "no:cacheprovider",
            "-q",
            "--no-header",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr
    assert "2 passed" in completed.stdout, completed.stdout[-3000:]
