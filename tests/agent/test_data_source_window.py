"""
A `DataSource` fetches what the caller asked for, and resolves only returns.

The symbol path fetched 1900-01-01..2100-01-01 from the default provider
whatever was asked, so the same call answered differently on different
days and for providers with different history depths; and the ref path
resolved any kind of reference, so a weight panel or a cash curve arrived
as if it were returns (see the CHANGELOG entry of 2026-10-22). A symbol now
takes its window and provider, and a reference is resolved with
`expect="returns_panel"`.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.data.models import DataSource, resolve_source
from standard_quant_tools.agent.runtimes.research.reference_tools import (
    SeriesMetricsInput,
    calculate_series_metrics,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError

START, END = "2024-01-02", "2024-12-31"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def asked(monkeypatch) -> List[Dict[str, Any]]:
    """Every provider request: which provider, which window."""
    calls: List[Dict[str, Any]] = []

    class _Provider:
        def __init__(self, source):
            self.source = source

        def get_ohlcv(self, symbol, start, end, interval="1d"):
            calls.append(
                {"source": self.source, "symbol": symbol, "start": start, "end": end}
            )
            index = pd.bdate_range(start, end)
            rng = np.random.default_rng(5)
            close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(index)))
            return pd.DataFrame({"Close": close}, index=index)

    monkeypatch.setattr(
        DataFactory, "get_provider", lambda source=None, *a, **k: _Provider(source)
    )
    return calls


class TestASymbolIsFetchedOverTheCallersWindow:
    def test_the_dates_and_provider_reach_the_fetch(self, asked):
        series = resolve_source(
            DataSource(symbol="AAA", start_date=START, end_date=END, source="polygon")
        )
        assert asked == [
            {"source": "polygon", "symbol": "AAA", "start": START, "end": END}
        ]
        assert series.index[0] > pd.Timestamp(START)
        assert series.index[-1] == pd.Timestamp(END)

    def test_no_source_asks_the_default(self, asked):
        resolve_source(DataSource(symbol="AAA", start_date=START, end_date=END))
        assert asked[0]["source"] is None

    def test_a_symbol_without_a_window_is_refused(self):
        with pytest.raises(
            PydanticValidationError, match="needs start_date and end_date"
        ):
            DataSource(symbol="AAA")

    def test_a_window_without_a_symbol_is_refused(self):
        with pytest.raises(PydanticValidationError, match="only apply to a `symbol`"):
            DataSource(values=[0.01, -0.01], start_date=START)

    def test_the_metrics_tool_fetches_the_window_it_was_given(self, asked):
        calculate_series_metrics(
            SeriesMetricsInput(
                series={"symbol": "AAA", "start_date": START, "end_date": END},
                metrics=["sharpe_ratio"],
            )
        )
        assert (asked[0]["start"], asked[0]["end"]) == (START, END)


class TestAReferenceMustBeReturns:
    def _series(self):
        index = pd.bdate_range(START, periods=60)
        return pd.Series(np.random.default_rng(1).normal(0, 0.01, 60), index=index)

    def test_a_returns_panel_is_resolved(self, runs_dir):
        ref = handoff.publish(self._series().to_frame("r"), "returns_panel", "w", "r")
        assert len(resolve_source(DataSource(ref=ref))) == 60

    @pytest.mark.parametrize("kind", ["analytic_series", "weight_panel"])
    def test_another_kind_is_refused_by_name(self, runs_dir, kind):
        series = self._series()
        data = series if kind == "analytic_series" else series.to_frame("AAA")
        ref = handoff.publish(data, kind, "w", kind)
        with pytest.raises(ValidationError, match="expected a 'returns_panel'"):
            resolve_source(DataSource(ref=ref))

    def test_an_equity_curve_is_refused_with_its_conversion(self, runs_dir):
        curve = (1 + self._series()).cumprod() * 10_000.0
        ref = handoff.publish(curve, "equity_curve", "w", "eq")
        with pytest.raises(ValidationError, match="convert_reference"):
            resolve_source(DataSource(ref=ref))
