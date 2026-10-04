"""
Nothing `tests/surface` installs is still installed outside it (see the
CHANGELOG entry of 2026-10-04).

`tests/surface/conftest.py` fakes yfinance, switches the Parquet tier off,
removes the retry backoff and moves the audit directory for the surface
tests. Those fixtures were session-scoped, so they stayed in place for every
test pytest ran after the package in the same session, and tests that read
the Parquet cache failed after it and passed on their own.

Run alone, this holds trivially. It is a check when it runs after a surface
test in the same session: in the whole suite, where `tests/surface` is
collected before the files at the top of `tests/`, and in
`tests/surface/test_the_layer_cleans_up.py`, which starts such a session
explicitly.
"""

from __future__ import annotations

import os
import time

import yfinance

import standard_quant_tools.data._cache as cache
import standard_quant_tools.data._retry as retry
import standard_quant_tools.data.yfinance_provider as provider


def test_the_provider_retry_and_audit_directory_are_the_real_ones():
    assert provider.yf is yfinance, "yfinance is still the surface tests' fake"
    assert (
        provider._is_historical is cache._is_historical
    ), "the Parquet tier is still switched off"
    assert retry.time is time, "provider retries still skip their backoff"
    assert "sqt-surface-audit" not in os.environ.get(
        "SQT_AUDIT_DIR", ""
    ), "the audit directory is still the surface tests' own"
