"""
A request gate sees every billable Databento request before it is sent.

An application that budgets the vendor -- a spend ledger with a per-request
cap and a daily budget -- could not see the requests this provider made, so
a fetch through it was outside the budget entirely. `set_request_gate`
registers one for the process: asked before every `timeseries.get_range`
with what is about to be sent, able to refuse, and told afterwards what
came back. There is no default, and with none registered nothing changes.
See the CHANGELOG entry of 2026-10-01.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd
import pytest

from standard_quant_tools.data import databento_provider
from standard_quant_tools.data.databento_provider import (
    BillableFetch,
    BillableRequest,
    RequestRefusedError,
    request_gate,
    set_request_gate,
)
from standard_quant_tools.error import NonRetryableAPIError, ValidationError

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


@dataclass
class Verdict:
    allowed: bool
    reason: str = ""


class RecordingGate:
    """Allows by default; `refuse(request)` decides otherwise."""

    def __init__(self, refuse=None) -> None:
        self.refuse = refuse
        self.asked: List[BillableRequest] = []
        self.told: List[Tuple[BillableRequest, Any, BillableFetch]] = []
        self._lock = threading.Lock()

    def before(self, request: BillableRequest) -> Verdict:
        with self._lock:
            self.asked.append(request)
        reason = self.refuse(request) if self.refuse else None
        return Verdict(reason is None, reason or "within budget")

    def after(self, request, verdict, fetched) -> None:
        with self._lock:
            self.told.append((request, verdict, fetched))


class SizedStore:
    """A vendor store that reports its size, as the real one does."""

    def __init__(self, frame: pd.DataFrame, nbytes: int) -> None:
        self._frame = frame
        self.nbytes = nbytes

    def to_df(self) -> pd.DataFrame:
        return self._frame


class SizedClient(StubClient):
    """The stub client, with every answer reporting 4,096 bytes per row."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        inner = self.timeseries

        class _Timeseries:
            def get_range(self, **kwargs):
                frame = inner.get_range(**kwargs).to_df()
                return SizedStore(frame, 4096 * len(frame))

        self.timeseries = _Timeseries()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """No gate, no cached bars and no retry sleeps carried in or out."""
    from standard_quant_tools.data import _cache as cache_module

    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()
    previous = set_request_gate(None)
    yield
    set_request_gate(previous)


def _client(cls=StubClient, **kwargs):
    return cls({CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE}, **kwargs)


class TestNoGate:
    def test_there_is_no_default_gate(self):
        assert request_gate() is None

    def test_without_a_gate_the_requests_are_unchanged(self):
        """The null case: the same calls, with the same arguments, as a
        provider that has never heard of a gate."""
        client = _client()
        frame = _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert len(frame) == 7
        assert client.calls == [
            {
                "dataset": CONSOLIDATED,
                "schema": "ohlcv-1d",
                "symbols": ["NVDA"],
                "stype_in": "raw_symbol",
                "start": "2024-01-02",
                "end": "2024-01-11",
            }
        ]


class TestTheGateIsAsked:
    def test_before_sees_the_request_exactly_as_it_is_sent(self):
        gate = RecordingGate()
        set_request_gate(gate)
        client = _client()
        _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        (asked,) = gate.asked
        (sent,) = client.calls
        assert asked == BillableRequest(
            dataset=sent["dataset"],
            schema=sent["schema"],
            symbols=tuple(sent["symbols"]),
            stype_in=sent["stype_in"],
            start=sent["start"],
            end=sent["end"],
        )
        assert asked.client is client

    def test_after_is_told_what_came_back_with_the_verdict_before_gave(self):
        gate = RecordingGate()
        set_request_gate(gate)
        frame = _provider(_client(SizedClient)).get_ohlcv(
            "NVDA", "2024-01-02", "2024-01-10"
        )
        ((request, verdict, fetched),) = gate.told
        assert request is gate.asked[0]
        assert verdict == Verdict(True, "within budget")
        assert fetched == BillableFetch(records=7, nbytes=7 * 4096)
        assert len(frame) == 7

    def test_a_store_without_a_size_reports_none(self):
        gate = RecordingGate()
        set_request_gate(gate)
        _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        ((_, _, fetched),) = gate.told
        assert fetched == BillableFetch(records=7, nbytes=None)

    def test_a_cache_hit_is_not_a_request(self):
        gate = RecordingGate()
        set_request_gate(gate)
        provider = _provider(_client())
        provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")  # session cache
        _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")  # disk
        assert len(gate.asked) == 1 and len(gate.told) == 1

    def test_ticks_depth_and_events_are_gated_too(self, monkeypatch):
        # The decoding is not under test; the requests are.
        for name in ("normalize_trades", "normalize_quotes", "normalize_mbo"):
            monkeypatch.setattr(databento_provider, name, lambda f, **kw: (f, []))
        monkeypatch.setattr(
            databento_provider, "normalize_book", lambda f, levels=10, **kw: (f, [])
        )
        gate = RecordingGate()
        set_request_gate(gate)
        client = _client(default=_bars())
        provider = _provider(client)
        provider.get_trades("NVDA", "2024-03-01", "2024-03-03")
        provider.get_quotes("NVDA", "2024-03-01", "2024-03-03")
        provider.get_order_book("NVDA", "2024-03-01", "2024-03-03")
        provider.get_order_events("NVDA", "2024-03-01", "2024-03-03")
        assert [r.schema for r in gate.asked] == ["trades", "mbp-1", "mbp-10", "mbo"]
        assert [c["schema"] for c in client.calls] == [r.schema for r in gate.asked]
        assert [f.records for _, _, f in gate.told] == [3, 3, 3, 3]

    def test_none_allows(self):
        class Observer:
            told: List[Any] = []

            def before(self, request):
                return None

            def after(self, request, verdict, fetched):
                self.told.append(verdict)

        observer = Observer()
        set_request_gate(observer)
        _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert observer.told == [None]


class TestARefusal:
    def test_is_raised_before_anything_is_sent(self):
        gate = RecordingGate(refuse=lambda r: "over the $0.50 per-request cap")
        set_request_gate(gate)
        client = _client()
        with pytest.raises(RequestRefusedError) as caught:
            _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert client.calls == []
        assert gate.told == []
        assert caught.value.reason == "over the $0.50 per-request cap"
        assert caught.value.request is gate.asked[0]
        assert "over the $0.50 per-request cap" in str(caught.value)
        assert CONSOLIDATED in str(caught.value)

    def test_is_not_retried_and_no_other_dataset_is_asked(self):
        gate = RecordingGate(refuse=lambda r: "daily budget spent")
        set_request_gate(gate)
        with pytest.raises(NonRetryableAPIError):
            _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert [r.dataset for r in gate.asked] == [CONSOLIDATED]

    def test_is_deterministic(self):
        set_request_gate(RecordingGate(refuse=lambda r: "daily budget spent"))
        messages = set()
        for _ in range(3):
            with pytest.raises(RequestRefusedError) as caught:
                _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
            messages.add(str(caught.value))
        assert len(messages) == 1

    def test_a_gate_that_fails_refuses(self):
        class Broken:
            def before(self, request):
                raise ConnectionError("ledger unreachable")

            def after(self, request, verdict, fetched):
                raise AssertionError("never told")

        set_request_gate(Broken())
        client = _client()
        with pytest.raises(RequestRefusedError, match="ledger unreachable"):
            _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert client.calls == []

    def test_an_unreadable_verdict_refuses(self):
        class Vague:
            def before(self, request):
                return "yes"

            def after(self, request, verdict, fetched):
                return None

        set_request_gate(Vague())
        with pytest.raises(RequestRefusedError, match="no `allowed`"):
            _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")

    def test_a_preflight_that_met_the_unfinalized_tail_walks_back(self):
        """The vendor refuses a daily end in its unfinalized tail, and the
        provider walks the end back a day on that error. A gate whose
        preflight met the same error says so in its reason, and the walk
        back applies to it the same way."""

        def refuse(request: BillableRequest) -> Optional[str]:
            if request.end > "2024-01-08":
                return "cost preflight failed (422 data_end_after_available_end)"
            return None

        gate = RecordingGate(refuse=refuse)
        set_request_gate(gate)
        client = _client()
        frame = _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert [r.end for r in gate.asked] == [
            "2024-01-11",
            "2024-01-10",
            "2024-01-09",
            "2024-01-08",
        ]
        assert [c["end"] for c in client.calls] == ["2024-01-08"]
        assert frame.index[-1] == pd.Timestamp("2024-01-05")


class TestAfterFailing:
    def test_the_fetch_still_returns_and_the_failure_is_logged(self, caplog):
        class Forgetful(RecordingGate):
            def after(self, request, verdict, fetched):
                raise OSError("ledger file locked")

        set_request_gate(Forgetful())
        with caplog.at_level(logging.WARNING, logger=databento_provider.__name__):
            frame = _provider(_client()).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert len(frame) == 7
        assert "ledger file locked" in caplog.text


class TestRegistration:
    def test_set_returns_the_gate_it_replaced(self):
        first, second = RecordingGate(), RecordingGate()
        assert set_request_gate(first) is None
        assert set_request_gate(second) is first
        assert request_gate() is second
        assert set_request_gate(None) is second
        assert request_gate() is None

    def test_an_object_without_both_methods_is_refused(self):
        class Half:
            def before(self, request):
                return None

        with pytest.raises(ValidationError, match="before.*after"):
            set_request_gate(Half())
        assert request_gate() is None

    def test_concurrent_fetches_are_each_asked_and_told_once(self):
        gate = RecordingGate()
        set_request_gate(gate)
        provider = _provider(_client())
        symbols = [f"Q{chr(65 + i)}" for i in range(16)]
        errors: List[BaseException] = []

        def fetch(symbol: str) -> None:
            try:
                provider.get_ohlcv(symbol, "2024-01-02", "2024-01-10")
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=fetch, args=(s,)) for s in symbols]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert sorted(r.symbols[0] for r in gate.asked) == sorted(symbols)
        assert sorted(r.symbols[0] for r, _, _ in gate.told) == sorted(symbols)
