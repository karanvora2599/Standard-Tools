"""
The reference layer could move a price across runtimes but not state one.

References exist so bulk values cross runtimes WITHOUT passing through the
conversation, and every tool honoured that literally: `describe_reference`
returns shape, `convert_reference` returns another reference, and
`FetchResult` is ref/kind/rows/columns/start/end. So an agent could fetch
NVDA's daily OHLCV, confirm every anchor date was present, hand the frame to
an analysis tool -- and never be able to say what the close was.

A deep-research worker hit exactly that wall and submitted a typed null:
"the data exists but cannot be materialized into a returns table in this
interface ... This is an API boundary, not a data gap." Its whole task was
lost. The findings that did survive had back-solved prices from scalars,
which an adversarial verifier correctly graded overstated.

`read_reference` is the missing half of `describe_reference`: what is IN it,
bounded so the reason references exist still holds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from standard_quant_tools.agent.models import ReadReferenceInput
from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.meta.tools import (
    _READ_REFERENCE_MAX_ROWS,
    read_reference,
)


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    return tmp_path


@pytest.fixture
def closes():
    """Ten sessions of closes, the shape the lost task was working in."""
    index = pd.date_range("2026-06-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {"close": [205.16 + i for i in range(10)], "volume": [1000 + i for i in range(10)]},
        index=index,
    )
    return handoff.publish(frame, "returns_panel", "run_read", "closes", producer="test")


def test_an_anchor_date_yields_its_actual_close(closes):
    """The question the whole failure was about: the close on a given day."""
    result = read_reference(ReadReferenceInput(ref=closes, dates=["2026-06-04"]))

    assert result.returned == 1
    assert result.rows[0]["close"] == pytest.approx(208.16)


def test_a_date_index_is_addressable_by_its_date(closes):
    """The index renders as '2026-06-04 00:00:00'; nobody asks that way."""
    result = read_reference(ReadReferenceInput(ref=closes, dates=["2026-06-04"]))
    assert result.rows[0]["index"].startswith("2026-06-04")


def test_an_absent_date_is_reported_not_raised(closes):
    """A holiday in an anchor list must not cost the caller the other rows."""
    result = read_reference(
        ReadReferenceInput(ref=closes, dates=["2026-06-04", "2026-01-01"])
    )

    assert result.missing == ["2026-01-01"]
    assert result.returned == 1


def test_naming_nothing_returns_both_ends(closes):
    result = read_reference(ReadReferenceInput(ref=closes))

    assert result.total_rows == 10
    assert result.rows[0]["index"].startswith("2026-06-01")
    assert result.rows[-1]["index"].startswith("2026-06-10")


def test_an_unknown_column_is_reported_not_raised(closes):
    result = read_reference(
        ReadReferenceInput(ref=closes, head=2, columns=["close", "nope"])
    )

    assert result.columns == ["close"]
    assert result.missing_columns == ["nope"]
    assert "volume" not in result.rows[0]


def test_the_window_is_bounded_and_says_when_it_capped():
    """The cap is the design: a reader that returns a frame undoes references."""
    index = pd.date_range("2020-01-01", periods=_READ_REFERENCE_MAX_ROWS + 40, freq="D")
    frame = pd.DataFrame({"close": range(len(index))}, index=index)
    ref = handoff.publish(frame, "returns_panel", "run_big", "big", producer="test")

    result = read_reference(ReadReferenceInput(ref=ref, head=len(index)))

    assert result.returned == _READ_REFERENCE_MAX_ROWS
    assert result.truncated is True
    assert result.total_rows == len(index)


def test_a_non_tabular_kind_is_refused_by_name(tmp_path):
    """An equity_curve resolves as a Series — say so, don't fail on .columns."""
    curve = pd.Series([1.0, 1.01, 1.02], index=pd.date_range("2026-01-01", periods=3))
    ref = handoff.publish(curve, "equity_curve", "run_curve", "curve", producer="test")

    with pytest.raises(ValueError, match="not tabular"):
        read_reference(ReadReferenceInput(ref=ref))


def test_an_unknown_argument_is_rejected_not_ignored():
    with pytest.raises(Exception):
        ReadReferenceInput(ref="sqt://returns_panel/r/n", rows=5)
