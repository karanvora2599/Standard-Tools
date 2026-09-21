"""
Every provider fetch reaches the open decision record.

Only bar fetches did. A tool that read Databento bars, any provider's
ticks, or Polygon's point-in-time filings produced a record with an
empty `data_sources`, so it could never replay as `data_changed`. Each
method is called inside an open record here and must leave exactly one
line saying what it read and a digest of what came back.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.context import _data_sources_var
from standard_quant_tools.data import databento_provider, polygon_provider
from standard_quant_tools.data.polygon_provider import PolygonProvider

from .test_databento_provider import (
    BASIC,
    CONSOLIDATED,
    DEPTH,
    SINCE_2023,
    WIDE,
    StubClient,
    _bars,
    _provider,
)
from .test_point_in_time_records import EPS, PAGE_ONE, PAGE_TWO, REVENUES


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    from standard_quant_tools.data import _cache as cache_module

    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)


@pytest.fixture
def open_record():
    token = _data_sources_var.set([])
    try:
        yield lambda: list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token)


class TestDatabento:
    def test_bars_name_the_dataset_that_answered(self, open_record):
        client = StubClient({CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE})
        _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        (entry,) = open_record()
        assert entry["symbol"] == "NVDA" and entry["interval"] == "1d"
        assert entry["source"] == f"databento:{CONSOLIDATED}"
        assert len(entry["content_hash"]) >= 16

    def test_ticks_depth_and_events_are_recorded_by_kind(
        self, open_record, monkeypatch
    ):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=3, freq="s", tz="UTC"),
                "x": [1.0, 2.0, 3.0],
            }
        )
        provider = _provider(StubClient({DEPTH: WIDE}, default=_bars()))
        monkeypatch.setattr(provider, "_fetch", lambda *a, **k: (frame.copy(), DEPTH))
        for name in ("normalize_trades", "normalize_quotes", "normalize_mbo"):
            monkeypatch.setattr(databento_provider, name, lambda f, **kw: (f, []))
        monkeypatch.setattr(
            databento_provider, "normalize_book", lambda f, levels=10, **kw: (f, [])
        )
        provider.get_trades("NVDA", "2024-01-02", "2024-01-03")
        provider.get_quotes("NVDA", "2024-01-02", "2024-01-03")
        provider.get_order_book("NVDA", "2024-01-02", "2024-01-03", levels=5)
        provider.get_order_events("NVDA", "2024-01-02", "2024-01-03")
        kinds = [entry["interval"] for entry in open_record()]
        assert kinds == ["trades", "quotes", "mbp-10:5", "mbo"]
        assert {entry["source"] for entry in open_record()} == {f"databento:{DEPTH}"}


class TestPolygon:
    def test_trades_and_quotes_are_recorded(self, open_record, monkeypatch):
        def fake_get(path, params, api_key):
            row = {
                "sip_timestamp": 1_704_200_000_000_000_000,
                "price": 10.0,
                "size": 5,
                "exchange": 4,
                "bid_price": 9.9,
                "bid_size": 1,
                "ask_price": 10.1,
                "ask_size": 1,
            }
            return {
                "results": [row, {**row, "sip_timestamp": row["sip_timestamp"] + 1}]
            }

        monkeypatch.setattr(polygon_provider, "_polygon_get", fake_get)
        provider = PolygonProvider(api_key="k")
        provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
        provider.get_quotes("AAPL", "2024-01-02", "2024-01-03")
        entries = open_record()
        assert [e["interval"] for e in entries] == ["trades", "quotes"]
        assert {e["source"] for e in entries} == {"polygon"}
        assert entries[0]["content_hash"] != entries[1]["content_hash"]

    def test_point_in_time_records_are_recorded_for_the_universe(
        self, open_record, monkeypatch
    ):
        def fake_get(path, params, api_key):
            return PAGE_TWO if params.get("cursor") == "abc" else PAGE_ONE

        monkeypatch.setattr(polygon_provider, "_polygon_get", fake_get)
        PolygonProvider(api_key="k").get_point_in_time_records(
            ["aapl", "msft"],
            "fundamentals",
            [EPS, REVENUES],
            "2022-01-01",
            "2023-12-31",
        )
        (entry,) = open_record()
        assert entry["interval"] == "pit:fundamentals"
        assert entry["symbol"] == "aapl,msft"
        assert entry["start"] == "2022-01-01" and entry["end"] == "2023-12-31"


class TestOutsideARecord:
    def test_a_direct_call_records_nothing_and_still_works(self):
        client = StubClient({CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE})
        frame = _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert len(frame) == 7
        assert _data_sources_var.get() is None or isinstance(
            _data_sources_var.get(), list
        )
        assert np.isfinite(frame["Close"]).all()
