"""
A lag or an age with a part under a second reads the same seconds on every
pandas.

`Series.dt.total_seconds()` divided a duration's integer by the units in a
second up to pandas 3.0 and, from 3.1, adds the whole seconds to the
fraction left over. The two round differently: 3 days and 0.847301324 s
read 259200.847301324 one way and 259200.84730132401 the other. Three
places read a duration that way -- the time-decay sample weights, the
publication lag `validate_pit_records` reports and the one an event panel's
validation accumulates -- and each takes timestamps as given, so an external
panel's or a record's nanoseconds reach it. They now count from the integer,
as `analysis.order_events` counts an order's lifetime: the whole seconds
plus the exact fraction, added in floating point. The planted answer is
that sum, formed here from the duration's own parts; it is what pandas 3.1
returns, and pandas 2.3 and 3.0 returned the division.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools._durations import duration_seconds
from standard_quant_tools.data.external_validation import _Checker
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import PitRecordsInput
from standard_quant_tools.modeling.agent.tools import validate_pit_records
from standard_quant_tools.modeling.validation.weights import time_decay_weights

START = pd.Timestamp("2024-01-01")

#: 3 days and 0.847301324 s, whose division and sum differ in the last bit.
SHORT_NS = 259_200_847_301_324
SHORT_SECONDS = 259_200 + 847_301_324 / 1e9

#: Past 2**53 ns (104 days), where the division first rounds the integer.
LONG_NS = 10_538_617_267_068_453
LONG_SECONDS = 10_538_617 + 267_068_453 / 1e9


def _stamp(ns: int) -> str:
    return (START + pd.Timedelta(ns, unit="ns")).isoformat()


class TestTheCount:
    def test_the_whole_seconds_plus_the_fraction(self):
        deltas = pd.Series(pd.to_timedelta([SHORT_NS, LONG_NS, 500, 0], unit="ns"))
        got = duration_seconds(deltas)
        assert got.tolist() == [SHORT_SECONDS, LONG_SECONDS, 5e-07, 0.0]
        # The division the older pandas used reads other bits for both.
        assert SHORT_NS / 1e9 != SHORT_SECONDS
        assert LONG_NS / 1e9 != LONG_SECONDS

    def test_a_missing_duration_is_nan(self):
        deltas = pd.Series(pd.to_timedelta([SHORT_NS, None], unit="ns"))
        got = duration_seconds(deltas)
        assert got[0] == SHORT_SECONDS and math.isnan(got[1])

    @pytest.mark.parametrize("unit", ["us", "ns"])
    def test_the_unit_the_duration_is_stored_in_does_not_matter(self, unit):
        """pandas 2 parses a stamp to nanoseconds and pandas 3 to
        microseconds when nothing finer is written: 3 days and 0.847301 s
        is the same bits in either unit, since 847301 / 1e6 and
        847301000 / 1e9 are the same quotient rounded once."""
        stored = np.array([259_200_847_301], dtype="m8[us]").astype(f"m8[{unit}]")
        assert duration_seconds(stored).tolist() == [259_200 + 847_301 / 1e6]

    def test_what_is_not_a_duration_is_refused(self):
        with pytest.raises(ValidationError, match="timedelta64"):
            duration_seconds(np.array([1.5, 2.5]))


class TestTheThreeReaders:
    def test_time_decay_weights_age_a_row_by_the_count(self):
        """Two rows 3 days and 0.847301324 s apart at a one-day half-life:
        the older row's weight is 0.5 to the power of its age in days."""
        dates = pd.DatetimeIndex([START, START + pd.Timedelta(SHORT_NS, unit="ns")])
        dates = dates.as_unit("ns").to_numpy()
        weights = time_decay_weights(dates, half_life=1.0)
        raw = np.power(0.5, np.array([SHORT_SECONDS / 86400.0, 0.0]) / 1.0)
        assert weights.tolist() == (raw / raw.mean()).tolist()

    def test_validate_pit_records_reports_the_lag_by_the_count(self):
        records = [
            {
                "event_time": "2024-01-01",
                "available_time": _stamp(SHORT_NS),
                "entity": "A",
                "eps": 1.0,
            },
            {
                "event_time": "2024-01-01",
                "available_time": _stamp(LONG_NS),
                "entity": "B",
                "eps": 2.0,
            },
        ]
        result = validate_pit_records(PitRecordsInput(records=records))
        lags = np.array([SHORT_SECONDS, LONG_SECONDS]) / 86400.0
        assert result.median_publication_lag_days == float(np.median(lags))

    def test_an_event_panel_s_validation_accumulates_the_count(self):
        frame = pd.DataFrame(
            {
                "event_time": [START, START],
                "available_time": pd.to_datetime([_stamp(SHORT_NS), _stamp(LONG_NS)]),
                "entity": ["A", "B"],
                "eps": [1.0, 2.0],
            }
        )
        checker = _Checker("event_panel", list(frame.columns))
        checker.feed(frame)
        lags = pd.Series([SHORT_SECONDS, LONG_SECONDS]) / 86400.0
        assert checker.lag_min == float(lags.min())
        assert checker.lag_max == float(lags.max())
        assert checker.lag_sum == float(lags.sum())
        assert checker.lag_n == 2
