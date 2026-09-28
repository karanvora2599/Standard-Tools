"""
The calendar allowlist cannot be extended by whoever is handed it.

`calendar_names()` is cached, and `validate_calendar_name` uses what it
returns as the allowlist a `DatasetSpec.calendar` is checked against. It
returned the cached LIST, so any caller that appended to what it was given
extended the allowlist for the rest of the process -- a made-up venue then
validated. It returns a tuple.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import calendar as calendar_module

pytestmark = pytest.mark.skipif(
    not calendar_module.calendar_available(),
    reason="exchange_calendars is not installed",
)


def test_appending_to_the_names_does_not_admit_a_venue():
    """Planted: this append used to make FAKE_EXCHANGE a valid calendar."""
    names = calendar_module.calendar_names()
    try:
        with pytest.raises(AttributeError):
            names.append("FAKE_EXCHANGE")  # type: ignore[attr-defined]
        with pytest.raises(ValidationError, match="not an exchange_calendars name"):
            calendar_module.validate_calendar_name("FAKE_EXCHANGE")
    finally:
        calendar_module.calendar_names.cache_clear()


def test_a_real_calendar_still_validates():
    """Null."""
    assert calendar_module.validate_calendar_name("XNYS") == "XNYS"
    names = calendar_module.calendar_names()
    assert isinstance(names, tuple) and list(names) == sorted(names)
    assert "XNYS" in names
