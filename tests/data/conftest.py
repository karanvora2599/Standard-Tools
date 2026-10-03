"""Shared isolation for the data-layer tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _forget_databento_publication_edges():
    """The Databento daily feed's publication edge is remembered for the
    process, keyed by dataset. Every test's stub vendor publishes to its own
    edge, so none may inherit another's."""
    from standard_quant_tools.data.databento_provider import forget_publication_edges

    forget_publication_edges()
    yield
    forget_publication_edges()
