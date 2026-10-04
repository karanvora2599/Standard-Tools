"""
A provider digests a frame for the decision record only when one is open.

Every fetch used to hash the frame it served and hand the hash to
`record_data_access`, which drops it at once when no decision record is
open -- the case for every direct call and every library user outside
`dispatch()`. The digest was most of the cost of a session-cache hit. Now
each provider asks `audit.recording_data_access()` first (see the
CHANGELOG entry of 2026-10-01).

Two halves: outside a record no frame is hashed, on any path of any
provider; inside one the line written is the line an unconditional digest
writes, byte for byte. Since the CHANGELOG entry of 2026-10-04 that digest is
`canonical_frame_hash`, recorded with `content_hash_version` 2, where it was
`hash_dataframe`.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import BaseModel

from standard_quant_tools import audit
from standard_quant_tools.audit import hashing
from standard_quant_tools.audit.context import _data_sources_var
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.data import (
    bloomberg_provider,
    databento_provider,
    polygon_provider,
    yfinance_provider,
)
from standard_quant_tools.data.bloomberg_provider import BloombergProvider
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider

from ..surface.hermetic import FakeYFinance
from .test_databento_provider import (
    BASIC,
    CONSOLIDATED,
    DEPTH,
    SINCE_2023,
    WIDE,
    StubClient,
    _provider,
)
from .test_point_in_time_records import EPS, PAGE_ONE, PAGE_TWO


def _reference_record(symbol, start_date, end_date, what, dataset, frame) -> None:
    """Databento's `_record` without the question: the digest taken first,
    unconditionally, then handed to a report that may discard it."""
    audit.record_data_access(
        symbol,
        str(start_date),
        str(end_date),
        what,
        source=f"databento:{dataset}",
        content_hash=audit.canonical_frame_hash(frame),
        content_hash_version=audit.DATA_SOURCE_HASH_VERSION,
    )


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from standard_quant_tools.data import _cache as cache_module

    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()
    yield
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


@pytest.fixture
def digests(monkeypatch) -> List[int]:
    """Every frame the providers hash, counted by row: the digest they
    record, and the earlier one, which only a replay of an earlier record
    takes."""
    seen: List[int] = []

    def counting(real):
        def digest(frame: Any) -> str:
            seen.append(len(frame))
            return real(frame)

        return digest

    for name in ("canonical_frame_hash", "hash_dataframe"):
        monkeypatch.setattr(hashing, name, counting(getattr(hashing, name)))
        monkeypatch.setattr(audit, name, counting(getattr(audit, name)))
    return seen


@pytest.fixture
def open_record():
    token = _data_sources_var.set([])
    try:
        yield lambda: list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token)


# ── the question ─────────────────────────────────────────────────────────


class _Probe(BaseModel):
    payload: Dict[str, Any] = {}


class TestTheQuestion:
    def test_no_record_is_open_outside_dispatch(self):
        assert audit.recording_data_access() is False

    def test_a_record_is_open_inside_dispatch_and_closed_after(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        seen: List[bool] = []

        def tool(_model):
            seen.append(audit.recording_data_access())
            return _Probe()

        _run_and_record("probe_tool", tool, _Probe())
        assert seen == [True]
        assert audit.recording_data_access() is False

    def test_a_record_is_open_while_audit_writes_are_disabled(self, monkeypatch):
        """A disabled audit still collects data sources for the call, so
        the answer must not depend on the write setting."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        seen: List[bool] = []

        def tool(_model):
            seen.append(audit.recording_data_access())
            return _Probe()

        _run_and_record("probe_tool", tool, _Probe())
        assert seen == [True]


# ── outside a record: nothing is hashed ──────────────────────────────────


def _databento_three_ways(provider) -> None:
    """A live fetch, a session-cache hit, and a disk-cache hit on a fresh
    instance (whose session cache is its own)."""
    provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
    provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")


def _databento_client() -> StubClient:
    return StubClient({CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE})


class TestOutsideARecordNothingIsHashed:
    def test_databento_bars_live_session_and_disk(self, digests):
        _databento_three_ways(_provider(_databento_client()))
        _provider(_databento_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert digests == []

    def test_databento_ticks_depth_and_events(self, digests, monkeypatch):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=3, freq="s", tz="UTC"),
                "x": [1.0, 2.0, 3.0],
            }
        )
        provider = _provider(_databento_client())
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
        assert digests == []

    def test_yfinance_live_session_and_disk(self, digests, monkeypatch):
        monkeypatch.setattr(yfinance_provider, "yf", FakeYFinance())
        YFinanceProvider().get_ohlcv("AAPL", "2024-01-02", "2024-03-28")
        provider = YFinanceProvider()
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-03-28")  # disk
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-03-28")  # session
        assert digests == []

    def test_polygon_bars_ticks_and_filings(self, digests, monkeypatch):
        monkeypatch.setattr(polygon_provider, "_polygon_get", _polygon_answers)
        provider = PolygonProvider(api_key="k")
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        PolygonProvider(api_key="k").get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
        provider.get_quotes("AAPL", "2024-01-02", "2024-01-03")
        provider.get_point_in_time_records(
            ["aapl"], "fundamentals", [EPS], "2022-01-01", "2023-12-31"
        )
        assert digests == []

    def test_bloomberg_bars(self, digests, monkeypatch):
        _bloomberg(monkeypatch).get_ohlcv("AAPL", "2023-01-03", "2023-01-04")
        assert digests == []


# ── inside a record: the same line as before ─────────────────────────────


class TestInsideARecordTheLineIsUnchanged:
    def test_databento_every_path_is_hashed_once(self, digests, open_record):
        _databento_three_ways(_provider(_databento_client()))
        _provider(_databento_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        sources = [entry["source"] for entry in open_record()]
        assert sources == [
            f"databento:{CONSOLIDATED}",
            f"databento:{CONSOLIDATED}:session_cache",
            f"databento:{CONSOLIDATED}:disk_cache",
        ]
        assert len(digests) == 3

    def test_databento_lines_match_the_reference_byte_for_byte(
        self, open_record, monkeypatch
    ):
        def run(record_fn) -> List[str]:
            monkeypatch.setattr(databento_provider, "_record", record_fn)
            token = _data_sources_var.set([])
            try:
                provider = _provider(_databento_client())
                _databento_three_ways(provider)
                _provider(_databento_client()).get_ohlcv(
                    "NVDA", "2024-01-02", "2024-01-10"
                )
                return [json.dumps(e) for e in _data_sources_var.get()]
            finally:
                _data_sources_var.reset(token)

        from standard_quant_tools.data import _cache as cache_module

        new_record = databento_provider._record
        current = run(new_record)
        with cache_module._session_cache_lock:
            cache_module._session_cache.clear()
        for path in Path(cache_module.cache_root()).rglob("*.parquet"):
            path.unlink()
        before = run(_reference_record)
        assert len(current) == 3
        assert current == before

    def test_a_written_decision_record_carries_the_same_data_sources(
        self, tmp_path, monkeypatch
    ):
        """Through `dispatch`'s core, onto disk: the data sources the
        record carries are the ones an unconditional digest writes, apart
        from the measured `fetch_ms`, which is a timing and differs run to
        run."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")

        from standard_quant_tools.data import _cache as cache_module

        def written(record_fn, directory: Path) -> List[Dict[str, Any]]:
            monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
            # Each run fetches live: its own disk cache, as a first run has.
            monkeypatch.setattr(cache_module, "_CACHE_ROOT", directory / "cache")
            monkeypatch.setattr(databento_provider, "_record", record_fn)
            client = _databento_client()

            def tool(_model):
                frame = _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
                return _Probe(payload={"rows": len(frame)})

            _run_and_record("probe_tool", tool, _Probe())
            day = audit._iter_day_files(directory)[-1]
            record = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
            return record["data_sources"]

        new_record = databento_provider._record
        current = written(new_record, tmp_path / "new")
        before = written(_reference_record, tmp_path / "old")
        assert [list(e) for e in current] == [list(e) for e in before]
        for entry in current + before:
            entry.pop("fetch_ms")
        assert json.dumps(current) == json.dumps(before)
        assert current[0]["source"] == f"databento:{CONSOLIDATED}"

    def test_yfinance_every_path_is_hashed_once(
        self, digests, open_record, monkeypatch
    ):
        monkeypatch.setattr(yfinance_provider, "yf", FakeYFinance())
        YFinanceProvider().get_ohlcv("AAPL", "2024-01-02", "2024-03-28")
        provider = YFinanceProvider()
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-03-28")
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-03-28")
        entries = open_record()
        assert [e["source"] for e in entries] == [
            "live_fetch",
            "disk_cache",
            "session_cache",
        ]
        assert len(digests) == 3
        assert len({e["content_hash"] for e in entries}) == 1

    def test_polygon_every_path_is_hashed_once(self, digests, open_record, monkeypatch):
        monkeypatch.setattr(polygon_provider, "_polygon_get", _polygon_answers)
        provider = PolygonProvider(api_key="k")
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        provider.get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        PolygonProvider(api_key="k").get_ohlcv("AAPL", "2024-01-02", "2024-01-05")
        provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
        provider.get_quotes("AAPL", "2024-01-02", "2024-01-03")
        provider.get_point_in_time_records(
            ["aapl"], "fundamentals", [EPS], "2022-01-01", "2023-12-31"
        )
        assert [(e["source"], e["interval"]) for e in open_record()] == [
            ("live_fetch", "1d"),
            ("session_cache", "1d"),
            ("disk_cache", "1d"),
            ("polygon", "trades"),
            ("polygon", "quotes"),
            ("polygon", "pit:fundamentals"),
        ]
        assert len(digests) == 6

    def test_bloomberg_bars_are_hashed_once(self, digests, open_record, monkeypatch):
        _bloomberg(monkeypatch).get_ohlcv("AAPL", "2023-01-03", "2023-01-04")
        (entry,) = open_record()
        assert entry["source"] == "live_fetch" and len(digests) == 1


# ── fakes ────────────────────────────────────────────────────────────────


def _polygon_answers(path: str, params: Dict[str, Any], api_key: str):
    if path.startswith("/v2/aggs/"):
        days = pd.bdate_range("2024-01-02", "2024-01-05")
        return {
            "results": [
                {
                    "t": int(day.value // 1_000_000),
                    "o": 10.0 + i,
                    "h": 11.0 + i,
                    "l": 9.0 + i,
                    "c": 10.5 + i,
                    "v": 1000 + i,
                }
                for i, day in enumerate(days)
            ]
        }
    if path.startswith("/v3/trades/") or path.startswith("/v3/quotes/"):
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
        return {"results": [row, {**row, "sip_timestamp": row["sip_timestamp"] + 1}]}
    return PAGE_TWO if params.get("cursor") == "abc" else PAGE_ONE


def _bloomberg(monkeypatch) -> BloombergProvider:
    """A provider whose Desktop API session answers two daily bars."""
    provider = BloombergProvider.__new__(BloombergProvider)
    monkeypatch.setattr(
        provider, "_open_session", lambda: MagicMock(name="session"), raising=False
    )
    bars = [
        {
            "date": dt.date(2023, 1, 3),
            "PX_OPEN": 100.0,
            "PX_HIGH": 105.0,
            "PX_LOW": 99.0,
            "PX_LAST": 103.0,
            "PX_VOLUME": 1_000_000.0,
        },
        {
            "date": dt.date(2023, 1, 4),
            "PX_OPEN": 103.0,
            "PX_HIGH": 108.0,
            "PX_LOW": 102.0,
            "PX_LAST": 107.0,
            "PX_VOLUME": 1_200_000.0,
        },
    ]
    monkeypatch.setattr(
        bloomberg_provider, "_drain_historical_response", lambda s, t: (bars, None)
    )
    return provider
