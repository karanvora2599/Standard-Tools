"""
A vendor that fails, a daily edge that lags, and a window that ends today.

Three fetch-path faults, each reproduced against a stub vendor; no request
leaves the process and no key is read. See the CHANGELOG entry of 2026-10-02.

  - A 504 was reported as "Databento returned no bars", after six requests
    and three seconds of sleeps: the retry layer re-ran a walk that passed
    every failure to the next dataset. A transient failure is now asked
    once more on the same dataset and then named; an empty walk is a
    `DataNotFoundError`; nothing is repeated by the retry layer.
  - The daily feed's publication edge was learned again on every call, one
    or two refused requests each time, because each tool call builds a new
    provider. The refused end is remembered for the process until the next
    UTC hour.
  - A window ending today was never stored, so every call downloaded its
    whole history. Its settled part is stored, and a later call asks only
    for the bars after it. The answer is compared below with one request
    for the whole window: same rows, dtypes and attrs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import audit
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data import databento_provider as dbp
from standard_quant_tools.data.databento import (
    DATASET_CONSOLIDATED,
    DATASET_DEPTH,
    DATASET_FUTURES,
    DATASET_NASDAQ_BASIC,
    DATASET_SUMMARY,
)
from standard_quant_tools.data.databento_provider import (
    DatabentoProvider,
    RequestRefusedError,
    _failure_kind,
    forget_publication_edges,
    set_request_gate,
)
from standard_quant_tools.error import (
    DataNotFoundError,
    NonRetryableAPIError,
    VendorUnavailableError,
)

SUMMARY, MINI = DATASET_SUMMARY, DATASET_CONSOLIDATED
BASIC, ITCH, GLOBEX = DATASET_NASDAQ_BASIC, DATASET_DEPTH, DATASET_FUTURES

#: The day the tests stand in for today: a Wednesday.
TODAY = date(2026, 3, 11)
#: The live edge every dataset reports: mid-session today.
EDGE = "2026-03-11T15:00:00+00:00"
RANGES = {
    SUMMARY: ("2024-07-01T00:00:00+00:00", EDGE),
    MINI: ("2023-03-28T00:00:00+00:00", EDGE),
    BASIC: ("2018-05-01T00:00:00+00:00", EDGE),
    ITCH: ("2018-05-01T00:00:00+00:00", EDGE),
    GLOBEX: ("2017-05-21T00:00:00+00:00", EDGE),
}


# ── the stub vendor ───────────────────────────────────────────────────────


class HttpError(Exception):
    """The shape of the vendor client's HTTP errors: a status, the vendor's
    message, and the response headers."""

    def __init__(self, status: int, message: str, headers=None) -> None:
        super().__init__(f"{status} {message}")
        self.http_status = status
        self.message = message
        self.headers = dict(headers or {})


class Store:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.nbytes = 64 * len(frame)
        self.symbology = {"not_found": []}

    def to_df(self) -> pd.DataFrame:
        return self._frame.copy()


def _close(stamp: pd.Timestamp) -> float:
    return round(
        100.0 + (stamp - pd.Timestamp("2026-01-01", tz="UTC")).total_seconds() / 36e4, 6
    )


def _session_hours(dataset: str, stamps: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if dataset == GLOBEX:
        day, hour = stamps.weekday, stamps.hour
        closed = (day == 5) | ((day == 6) & (hour < 22)) | ((day == 4) & (hour >= 21))
        return stamps[~closed & (hour != 21)]
    return stamps[(stamps.weekday < 5) & (stamps.hour >= 14) & (stamps.hour <= 20)]


class Vendor:
    """
    Answers exactly the window it is asked for, the same bars for the same
    instants whatever the window: the property a stored part relies on.
    `finalized_through` is the last finalized daily bar; a daily request
    ending after it is refused as the vendor refuses one. `nan_closes`
    are bars listed with no Close. `fail(kw)` may return an exception for
    a data request, and `fail_range(dataset)` for a coverage lookup.
    """

    def __init__(
        self,
        ranges=None,
        *,
        finalized_through: Optional[str] = "2026-03-10",
        nan_closes=(),
        fail: Optional[Callable[[dict], Optional[Exception]]] = None,
        fail_range: Optional[Callable[[str], Optional[Exception]]] = None,
        empty_from: Optional[str] = None,
    ) -> None:
        self.ranges = dict(RANGES if ranges is None else ranges)
        self.finalized_through = finalized_through
        self.nan_closes = {pd.Timestamp(x, tz="UTC") for x in nan_closes}
        self.fail = fail
        self.fail_range = fail_range
        self.empty_from = (
            None if empty_from is None else pd.Timestamp(empty_from, tz="UTC")
        )
        self.calls: List[dict] = []
        self.range_calls: List[str] = []
        self.priced: List[dict] = []
        vendor = self

        class Metadata:
            def get_dataset_range(self, dataset):
                vendor.range_calls.append(dataset)
                if vendor.fail_range is not None:
                    failure = vendor.fail_range(dataset)
                    if failure is not None:
                        raise failure
                if dataset not in vendor.ranges:
                    raise HttpError(403, f"not_entitled for {dataset}")
                first, last = vendor.ranges[dataset]
                return {"start": first, "end": last}

            def get_billable_size(self, **kw):
                vendor.priced.append(kw)
                if vendor.fail is not None:
                    failure = vendor.fail({**kw, "endpoint": "billable"})
                    if failure is not None:
                        raise failure
                return 4096

        class Timeseries:
            def get_range(self, **kw):
                vendor.calls.append(kw)
                if vendor.fail is not None:
                    failure = vendor.fail(kw)
                    if failure is not None:
                        raise failure
                return Store(vendor.bars(kw))

        self.metadata = Metadata()
        self.timeseries = Timeseries()

    def bars(self, kw: dict) -> pd.DataFrame:
        start = pd.Timestamp(kw["start"], tz="UTC")
        end = pd.Timestamp(kw["end"], tz="UTC")
        if kw["schema"] == "ohlcv-1d":
            if self.finalized_through is not None and end > pd.Timestamp(
                self.finalized_through, tz="UTC"
            ) + pd.Timedelta(days=1):
                raise HttpError(
                    422,
                    "data_end_after_available_end: the daily schema is not "
                    "finalized past this end",
                )
            stamps = pd.bdate_range(start.normalize(), end, tz="UTC", inclusive="left")
        else:
            step = {"ohlcv-1h": "h", "ohlcv-1m": "min"}[kw["schema"]]
            stamps = _session_hours(
                kw["dataset"],
                pd.date_range(
                    start.ceil(step), end, freq=step, tz="UTC", inclusive="left"
                ),
            )
        stamps = stamps[(stamps >= start) & (stamps < end)]
        if self.empty_from is not None:
            stamps = stamps[stamps < self.empty_from]
        close = np.array([_close(s) for s in stamps], dtype="float64")
        nan = np.array([s in self.nan_closes for s in stamps], dtype=bool)
        close[nan] = np.nan
        # No `freq` on the index, as on the client's decoded frames: a range
        # built here would carry one, which neither a Parquet read nor a
        # concatenation keeps.
        index = pd.DatetimeIndex(
            pd.to_datetime(stamps.as_unit("ns").asi8, utc=True), name="ts_event"
        )
        frame = pd.DataFrame(
            {
                "open": close - 0.25,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.array(
                    [1000 + s.hour + s.day for s in stamps], dtype="uint64"
                ),
            },
            index=index,
        )
        frame.loc[nan, ["open", "high", "low"]] = np.nan
        return frame

    def datasets_called(self) -> List[str]:
        return [c["dataset"] for c in self.calls]


class RecordingGate:
    def __init__(self, refuse: Optional[Callable[[Any], Optional[str]]] = None) -> None:
        self.refuse = refuse
        self.asked: List[Any] = []
        self.told: List[Any] = []

    def before(self, request):
        self.asked.append(request)
        reason = self.refuse(request) if self.refuse else None
        return type(
            "Verdict", (), {"allowed": reason is None, "reason": reason or ""}
        )()

    def after(self, request, verdict, fetched):
        self.told.append((request, fetched))


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    taken: List[float] = []
    monkeypatch.setattr("time.sleep", lambda seconds: taken.append(seconds))
    return taken


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(cache_module, "_utc_today", lambda: TODAY)
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
        "DATABENTO_FUTURES_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()
    previous = set_request_gate(None)
    forget_publication_edges()
    yield
    forget_publication_edges()
    set_request_gate(previous)


def _provider(vendor: Vendor) -> DatabentoProvider:
    return DatabentoProvider(api_key="not-used", client=vendor)


# ── 1. a vendor-side failure is named, retried once, never passed on ─────


class TestATransientFailureIsTheVendors:
    WINDOW = ("2025-09-01", "2025-09-10")

    def test_a_504_is_asked_once_more_on_the_same_feed_and_named(self, sleeps):
        """The reproduction: an hourly request answered 504. It used to be
        six requests over both venue feeds, three seconds of sleeps, and
        "Databento returned no bars"."""
        vendor = Vendor(fail=lambda kw: HttpError(504, "The remote gateway timed out."))
        with pytest.raises(VendorUnavailableError) as caught:
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW, interval="1h")
        assert vendor.datasets_called() == [BASIC, BASIC]
        assert vendor.calls[0] == vendor.calls[1], "the same request, not another"
        assert len(sleeps) == 1 and 0.25 <= sleeps[0] <= 0.75
        error = caught.value
        assert isinstance(error, NonRetryableAPIError)
        assert (error.status, error.dataset) == (504, BASIC)
        text = str(error)
        assert "HTTP 504" in text and BASIC in text and "vendor-side" in text
        assert "returned no" not in text and "Datasets tried" not in text

    def test_an_empty_walk_is_not_found_and_costs_one_request_per_feed(self, sleeps):
        vendor = Vendor(empty_from="2000-01-01")
        with pytest.raises(DataNotFoundError, match="Datasets tried"):
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW, interval="1h")
        assert vendor.datasets_called() == [BASIC, ITCH]
        assert sleeps == []

    def test_a_failure_that_clears_is_answered_by_the_same_feed(self, sleeps):
        state = {"left": 1}

        def once(kw):
            if state["left"]:
                state["left"] -= 1
                return HttpError(503, "Service Unavailable")
            return None

        vendor = Vendor(fail=once)
        frame = _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert frame.attrs["dataset"] == SUMMARY
        assert vendor.datasets_called() == [SUMMARY, SUMMARY]
        clean = _provider(Vendor()).get_ohlcv("AAPL", *self.WINDOW)
        pd.testing.assert_frame_equal(frame, clean)

    @pytest.mark.parametrize(
        "failure",
        [
            HttpError(408, "The request transmission timed out."),
            HttpError(500, "internal error"),
            HttpError(502, "Bad gateway from auth proxy"),
            TimeoutError("timed out"),
            ConnectionResetError("connection reset by peer"),
        ],
        ids=["408", "500", "502-mentioning-auth", "timeout", "reset"],
    )
    def test_each_transient_kind_is_retried_then_named(self, failure):
        vendor = Vendor(fail=lambda kw: failure)
        with pytest.raises(VendorUnavailableError):
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert vendor.datasets_called() == [SUMMARY, SUMMARY]

    def test_a_429_waits_the_time_the_vendor_asks(self, sleeps):
        state = {"left": 1}

        def limited(kw):
            if state["left"]:
                state["left"] -= 1
                return HttpError(429, "Too many requests", {"Retry-After": "2"})
            return None

        vendor = Vendor(fail=limited)
        frame = _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert sleeps == [2.0]
        assert len(vendor.calls) == 2 and len(frame) == 8

    def test_a_429_asking_for_longer_than_the_cap_is_not_asked_again(self, sleeps):
        vendor = Vendor(
            fail=lambda kw: HttpError(429, "Too many requests", {"retry-after": "30"})
        )
        with pytest.raises(VendorUnavailableError, match="30 s") as caught:
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert len(vendor.calls) == 1 and sleeps == []
        assert caught.value.retry_after == 30.0 and caught.value.status == 429

    def test_a_request_level_failure_still_asks_the_next_feed(self, sleeps):
        """A 422 that is not the daily lag is about this request on this
        dataset: not retried, and the next feed is asked, as before."""
        vendor = Vendor(
            fail=lambda kw: (
                HttpError(422, "symbology_invalid_request")
                if kw["dataset"] == SUMMARY
                else None
            )
        )
        frame = _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert vendor.datasets_called() == [SUMMARY, MINI]
        assert frame.attrs["dataset"] == MINI and sleeps == []

    def test_request_level_failures_everywhere_are_named_not_called_empty(self):
        vendor = Vendor(fail=lambda kw: HttpError(400, "bad_request"))
        with pytest.raises(NonRetryableAPIError) as caught:
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert not isinstance(caught.value, VendorUnavailableError)
        text = str(caught.value)
        assert "HTTP 400" in text and "returned no" not in text
        assert vendor.datasets_called() == [SUMMARY, MINI, BASIC, ITCH]

    def test_a_failed_coverage_lookup_is_named_and_nothing_lesser_is_asked(self):
        vendor = Vendor(
            fail_range=lambda dataset: (
                HttpError(503, "Service Unavailable") if dataset == SUMMARY else None
            )
        )
        with pytest.raises(VendorUnavailableError, match="coverage lookup") as caught:
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert vendor.range_calls == [SUMMARY, SUMMARY] and vendor.calls == []
        assert "No dataset covers" not in str(caught.value)

    def test_each_retry_passes_through_the_gate(self):
        gate = RecordingGate()
        set_request_gate(gate)
        state = {"left": 1}

        def once(kw):
            if state["left"]:
                state["left"] -= 1
                return HttpError(504, "The remote gateway timed out.")
            return None

        _provider(Vendor(fail=once)).get_ohlcv("AAPL", *self.WINDOW)
        assert len(gate.asked) == 2 and gate.asked[0] == gate.asked[1]
        assert len(gate.told) == 1, "only a request that returned is reported"

    def test_a_retry_the_gate_refuses_is_a_refusal(self):
        gate = RecordingGate(refuse=lambda r: "over budget" if gate.asked[1:] else None)
        set_request_gate(gate)
        vendor = Vendor(fail=lambda kw: HttpError(504, "The remote gateway timed out."))
        with pytest.raises(RequestRefusedError, match="over budget"):
            _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert len(vendor.calls) == 1

    def test_trades_are_named_the_same_way(self):
        vendor = Vendor(fail=lambda kw: HttpError(504, "The remote gateway timed out."))
        with pytest.raises(VendorUnavailableError, match="trades"):
            _provider(vendor).get_trades(
                "AAPL", "2025-09-02T14:30:00", "2025-09-02T14:31:00"
            )
        assert vendor.datasets_called() == [BASIC, BASIC]


class TestTheFreeLookupsNameAVendorFailureToo:
    def test_coverage_does_not_leave_a_failing_dataset_out(self):
        """Leaving it out read as "unentitled, unknown, or declined"."""
        vendor = Vendor(fail_range=lambda d: HttpError(502, "Bad Gateway"))
        with pytest.raises(
            VendorUnavailableError, match="coverage lookup for EQUS.SUMMARY"
        ):
            _provider(vendor).get_dataset_coverage([SUMMARY, MINI])

    def test_null_a_dataset_the_subscription_declines_is_still_left_out(self):
        vendor = Vendor(ranges={MINI: RANGES[MINI]})
        assert set(_provider(vendor).get_dataset_coverage([SUMMARY, MINI])) == {MINI}

    def test_pricing_does_not_move_on_to_another_feed(self):
        vendor = Vendor(
            fail=lambda kw: (
                HttpError(504, "The remote gateway timed out.")
                if kw.get("endpoint") == "billable"
                else None
            )
        )
        with pytest.raises(VendorUnavailableError, match="billable-size lookup"):
            _provider(vendor).get_billable_size(
                "AAPL", "2025-09-02", "2025-09-03", "ohlcv-1d"
            )
        assert [p["dataset"] for p in vendor.priced] == [SUMMARY, SUMMARY]


class TestAFailureIsReadByItsKind:
    @pytest.mark.parametrize(
        "failure, kind",
        [
            (HttpError(504, "The remote gateway timed out."), "transient"),
            (HttpError(429, "Too many requests"), "transient"),
            (HttpError(408, "timed out"), "transient"),
            (HttpError(503, "maintenance"), "transient"),
            (HttpError(401, "Unauthorized"), "auth"),
            (HttpError(403, "not_entitled"), "denied"),
            (HttpError(422, "available up to 2026-10-02T12:50:00.503118000Z"), "other"),
            (HttpError(404, "dataset_not_found"), "other"),
            (RuntimeError("500 gateway error"), "transient"),
            (RuntimeError("403 Forbidden: not_entitled"), "denied"),
            (RuntimeError("422 symbology author mismatch"), "other"),
            (TimeoutError("read timed out"), "transient"),
            (ConnectionAbortedError("aborted"), "transient"),
            (ValueError("bad input"), "other"),
        ],
    )
    def test_the_kind(self, failure, kind):
        assert _failure_kind(failure) == kind

    def test_the_clients_own_errors(self):
        requests = pytest.importorskip("requests")
        error = pytest.importorskip("databento.common.error")
        assert (
            _failure_kind(requests.exceptions.ReadTimeout("read timed out"))
            == "transient"
        )
        assert (
            _failure_kind(requests.exceptions.ConnectionError("refused")) == "transient"
        )
        assert (
            _failure_kind(requests.exceptions.ChunkedEncodingError("reset"))
            == "transient"
        )
        streamed = error.BentoError(
            "Error streaming response: ('Connection broken: IncompleteRead')"
        )
        assert _failure_kind(streamed) == "transient"
        gateway = error.BentoServerError(
            http_status=504, http_body=b"", message="The remote gateway timed out."
        )
        assert _failure_kind(gateway) == "transient"
        assert (
            _failure_kind(error.BentoClientError(http_status=401, message="x"))
            == "auth"
        )


# ── 2. the daily publication edge is learned once per process ────────────


class TestThePublicationEdgeIsRemembered:
    WINDOW = ("2026-02-02", "2026-02-27")

    def _vendor(self, **kw) -> Vendor:
        # Finalized through the 25th: an end of the 28th or the 27th is
        # refused, so a walk from the window's end costs two refusals.
        return Vendor(finalized_through="2026-02-25", **kw)

    def _clear_disk(self, tmp_path) -> None:
        for path in (tmp_path / "cache").glob("*.parquet"):
            path.unlink()

    def test_a_new_provider_does_not_pay_the_refusals_again(self):
        first = self._vendor()
        frame = _provider(first).get_ohlcv("AAPL", *self.WINDOW)
        assert [c["end"] for c in first.calls] == [
            "2026-02-28",
            "2026-02-27",
            "2026-02-26",
        ]
        second = self._vendor()
        again = _provider(second).get_ohlcv("AAPL", "2026-02-09", "2026-02-27")
        assert [c["end"] for c in second.calls] == ["2026-02-26"]
        assert (
            str(frame.index[-1].date()) == str(again.index[-1].date()) == "2026-02-25"
        )

    def test_the_answer_is_the_one_the_full_walk_gives(self, tmp_path):
        _provider(self._vendor()).get_ohlcv("AAPL", "2026-02-09", "2026-02-27")
        vendor = self._vendor()
        remembered = _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert len(vendor.calls) == 1
        self._clear_disk(tmp_path)
        forget_publication_edges()
        walking = self._vendor()
        walked = _provider(walking).get_ohlcv("AAPL", *self.WINDOW)
        assert len(walking.calls) == 3
        pd.testing.assert_frame_equal(remembered, walked)

    def test_it_is_learned_again_after_the_hour(self, monkeypatch):
        _provider(self._vendor()).get_ohlcv("AAPL", *self.WINDOW)
        later = datetime.now(timezone.utc) + timedelta(hours=1, minutes=1)
        monkeypatch.setattr(dbp, "_utc_now", lambda: later)
        vendor = self._vendor()
        _provider(vendor).get_ohlcv("AAPL", "2026-02-16", "2026-02-27")
        assert len(vendor.calls) == 3

    def test_the_skipped_days_count_against_the_walk(self, tmp_path):
        """Six attempts reach back five days. A remembered refusal starts
        the walk lower, and the days it skips count against those six, so
        it ends where the full walk ends: here, with nothing finalized in
        reach. Without the count the walk would reach the 25th and answer."""
        _provider(self._vendor()).get_ohlcv("AAPL", *self.WINDOW)
        vendor = Vendor(finalized_through="2026-02-24")
        with pytest.raises(DataNotFoundError):
            _provider(vendor).get_ohlcv("AAPL", "2026-02-24", "2026-03-02")
        assert [c["end"] for c in vendor.calls if c["dataset"] == SUMMARY] == [
            "2026-02-26"
        ]
        self._clear_disk(tmp_path)
        forget_publication_edges()
        walked = Vendor(finalized_through="2026-02-24")
        with pytest.raises(DataNotFoundError):
            _provider(walked).get_ohlcv("AAPL", "2026-02-24", "2026-03-02")
        assert len([c for c in walked.calls if c["dataset"] == SUMMARY]) == 6

    def test_null_a_dataset_never_refused_is_asked_at_its_end(self):
        vendor = Vendor(finalized_through=None)
        _provider(vendor).get_ohlcv("AAPL", *self.WINDOW)
        assert [c["end"] for c in vendor.calls] == ["2026-02-28"]


# ── 3. a window ending today keeps its settled part ──────────────────────


def _same(served: pd.DataFrame, reference: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(served, reference)
    assert served.attrs == reference.attrs
    assert audit.hash_dataframe(served) == audit.hash_dataframe(reference)


def _settled_files(root) -> List[str]:
    return sorted(p.name for p in root.glob("*.parquet"))


class TestAWindowEndingTodayKeepsItsSettledPart:
    START = "2026-02-02"

    @pytest.fixture
    def reference(self, tmp_path, monkeypatch):
        """One request for the whole window, from a fresh cache and with
        nothing remembered: the answer a stored part must reproduce."""

        def fetch(vendor: Vendor, *window, interval="1d", symbol="AAPL"):
            root = cache_module._CACHE_ROOT
            monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "reference")
            forget_publication_edges()
            try:
                frame = _provider(vendor).get_ohlcv(symbol, *window, interval=interval)
            finally:
                monkeypatch.setattr(cache_module, "_CACHE_ROOT", root)
                forget_publication_edges()
            return frame

        return fetch

    def _today(self, monkeypatch, day: date) -> None:
        monkeypatch.setattr(cache_module, "_utc_today", lambda: day)

    def test_the_second_call_asks_only_for_the_bars_after_the_part(
        self, reference, tmp_path
    ):
        cold = Vendor()
        first = _provider(cold).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert [c["start"] for c in cold.calls] == [self.START, self.START]
        assert any(
            "2026-02-02_2026-03-09" in f for f in _settled_files(tmp_path / "cache")
        )

        warm = Vendor()
        second = _provider(warm).get_ohlcv("AAPL", self.START, "2026-03-11")
        # The refused end is remembered (see section 2), so one request.
        assert [(c["start"], c["end"]) for c in warm.calls] == [
            ("2026-03-10", "2026-03-11")
        ]
        _same(second, first)
        _same(second, reference(Vendor(), self.START, "2026-03-11"))
        assert str(second.index[-1].date()) == "2026-03-10"

    def test_the_next_day_reads_the_older_part_and_stores_a_newer_one(
        self, reference, tmp_path, monkeypatch
    ):
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        self._today(monkeypatch, date(2026, 3, 12))
        # A day later the remembered edge has long expired; here the real
        # clock has not moved, so it is dropped by hand.
        forget_publication_edges()
        edge = {d: (s, "2026-03-12T15:00:00+00:00") for d, (s, _e) in RANGES.items()}
        vendor = Vendor(edge, finalized_through="2026-03-11")
        served = _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-12")
        assert all(c["start"] == "2026-03-10" for c in vendor.calls)
        _same(
            served,
            reference(
                Vendor(edge, finalized_through="2026-03-11"), self.START, "2026-03-12"
            ),
        )
        # The newer part replaces the older: one copy of the history, not
        # one a day.
        files = _settled_files(tmp_path / "cache")
        assert [f for f in files if "2026-02-02_2026-03-" in f] == [
            f"{cache_module.cache_generation('databento')}_databento-{SUMMARY}_AAPL_"
            "2026-02-02_2026-03-10_1d.parquet"
        ]
        # And the day after that reads it.
        self._today(monkeypatch, date(2026, 3, 13))
        forget_publication_edges()
        edge = {d: (s, "2026-03-13T15:00:00+00:00") for d, (s, _e) in RANGES.items()}
        vendor = Vendor(edge, finalized_through="2026-03-12")
        served = _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-13")
        assert all(c["start"] == "2026-03-11" for c in vendor.calls)
        _same(
            served,
            reference(
                Vendor(edge, finalized_through="2026-03-12"), self.START, "2026-03-13"
            ),
        )

    def test_a_request_for_the_part_alone_is_served_from_it(self, reference):
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        vendor = Vendor()
        part = _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-09")
        assert vendor.calls == []
        _same(part, reference(Vendor(), self.START, "2026-03-09"))

    @pytest.mark.parametrize(
        "nan_closes",
        [
            ["2026-03-09"],
            ["2026-03-09", "2026-03-10"],
            ["2026-02-10", "2026-03-10"],
            ["2026-02-02", "2026-03-06"],
        ],
        ids=["part-tail", "part-tail-and-rest", "inside-and-rest", "first-and-inside"],
    )
    def test_bars_with_no_close_are_disclosed_as_one_request_would(
        self, reference, nan_closes
    ):
        """A part's trailing placeholder is a missing bar of the longer
        window: the two lists are recomputed for the window served."""
        _provider(Vendor(nan_closes=nan_closes)).get_ohlcv(
            "AAPL", self.START, "2026-03-11"
        )
        warm = Vendor(nan_closes=nan_closes)
        served = _provider(warm).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert len(warm.calls) == 1
        _same(
            served, reference(Vendor(nan_closes=nan_closes), self.START, "2026-03-11")
        )

    def test_a_future_s_trade_dates_join_at_the_part(self, reference):
        _provider(Vendor()).get_ohlcv("ESH6", self.START, "2026-03-11")
        warm = Vendor()
        served = _provider(warm).get_ohlcv("ESH6", self.START, "2026-03-11")
        assert [(c["schema"], c["start"]) for c in warm.calls] == [
            ("ohlcv-1h", "2026-03-09T00:00:00")
        ]
        _same(served, reference(Vendor(), self.START, "2026-03-11", symbol="ESH6"))
        assert served.attrs["session"].startswith("CME trade date")

    def test_intraday_bars_join_at_the_part(self, reference):
        _provider(Vendor()).get_ohlcv("AAPL", "2026-03-02", "2026-03-11", interval="1h")
        warm = Vendor()
        served = _provider(warm).get_ohlcv(
            "AAPL", "2026-03-02", "2026-03-11", interval="1h"
        )
        assert [c["start"] for c in warm.calls] == ["2026-03-10T00:00:00"]
        _same(served, reference(Vendor(), "2026-03-02", "2026-03-11", interval="1h"))

    def test_a_coverage_downgrade_is_disclosed_for_the_whole_window(self, reference):
        window = ("2024-06-03", "2026-03-11")
        _provider(Vendor()).get_ohlcv("AAPL", *window)
        warm = Vendor()
        served = _provider(warm).get_ohlcv("AAPL", *window)
        assert warm.datasets_called() == [MINI]
        assert served.attrs["coverage_downgrade"]["preferred"] == SUMMARY
        _same(served, reference(Vendor(), *window))

    def test_nothing_after_the_part_serves_the_part(self, reference):
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        served = _provider(Vendor(empty_from="2026-03-10")).get_ohlcv(
            "AAPL", self.START, "2026-03-11"
        )
        _same(
            served, reference(Vendor(empty_from="2026-03-10"), self.START, "2026-03-11")
        )

    def test_the_rest_goes_through_the_gate_and_the_part_does_not(self):
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        gate = RecordingGate()
        set_request_gate(gate)
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert [(r.start, r.end) for r in gate.asked] == [("2026-03-10", "2026-03-11")]

    def test_a_vendor_failure_after_the_part_is_named_not_served_short(self):
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        vendor = Vendor(fail=lambda kw: HttpError(504, "The remote gateway timed out."))
        with pytest.raises(VendorUnavailableError):
            _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-11")

    def test_a_part_the_vendor_had_not_finalized_is_not_stored(self, tmp_path):
        """Finalized only through the 6th: the part through the 9th would
        be short, so it is not written, and nothing is marked settled."""
        vendor = Vendor(finalized_through="2026-03-06")
        _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert _settled_files(tmp_path / "cache") == []

    def test_an_unmarked_entry_is_never_used_as_a_part(self, reference, tmp_path):
        """An entry the historical path wrote before it required the whole
        window may be short -- written while its last day was unfinalized --
        and nothing on it says so. One is planted here as it was written."""
        short = _provider(Vendor(finalized_through="2026-03-05")).get_ohlcv(
            "AAPL", self.START, "2026-03-09"
        )
        assert str(short.index[-1].date()) == "2026-03-05"
        assert _settled_files(tmp_path / "cache") == [], "short: not written"
        entry = DatabentoProvider._bar_cache_path(
            DatabentoProvider.resolve_symbol("AAPL"),
            SUMMARY,
            self.START,
            "2026-03-09",
            "1d",
        )
        assert cache_module._write_cached_ohlcv(
            entry, short, "1d", self.START, "2026-03-09"
        )
        forget_publication_edges()
        vendor = Vendor()
        served = _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert vendor.calls[-1]["start"] == self.START, "fetched whole"
        _same(served, reference(Vendor(), self.START, "2026-03-11"))

    def test_a_tampered_part_is_read_again_by_a_new_provider(self, tmp_path):
        """What a replay relies on: a fresh provider re-reads the disk, so
        an edited part changes the answer and its digest."""
        _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        honest = _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        (path,) = [p for p in (tmp_path / "cache").glob("*2026-03-09*.parquet")]
        stored = pd.read_parquet(path)
        stored.iloc[0, stored.columns.get_loc("Close")] += 1.0
        stored.to_parquet(path)
        edited = _provider(Vendor()).get_ohlcv("AAPL", self.START, "2026-03-11")
        assert audit.hash_dataframe(edited) != audit.hash_dataframe(honest)

    def test_null_a_historical_window_is_cached_whole_as_before(self, tmp_path):
        vendor = Vendor()
        _provider(vendor).get_ohlcv("AAPL", self.START, "2026-03-06")
        assert _settled_files(tmp_path / "cache") == [
            f"{cache_module.cache_generation('databento')}_databento-{SUMMARY}_AAPL_"
            "2026-02-02_2026-03-06_1d.parquet"
        ]
        frame = pd.read_parquet(
            tmp_path / "cache" / _settled_files(tmp_path / "cache")[0]
        )
        assert cache_module.SETTLED_ATTR not in frame.attrs


# ── 4. only an answer to the whole window is cached ──────────────────────


class TestOnlyTheWholeWindowIsCached:
    """
    A window ending yesterday is historical, but the daily feed finalizes a
    day or two late: asked early, the walk-back served it to the day before,
    and that short answer was cached for good. It is now served and not
    written, and the next call asks again until the vendor serves the whole
    window (see the CHANGELOG entry of 2026-10-02).
    """

    WINDOW = ("2026-02-02", "2026-03-10")

    def test_a_short_answer_is_served_not_written_and_written_once_whole(
        self, tmp_path
    ):
        early = Vendor(finalized_through="2026-03-09")
        short = _provider(early).get_ohlcv("AAPL", *self.WINDOW)
        assert str(short.index[-1].date()) == "2026-03-09"
        assert _settled_files(tmp_path / "cache") == []

        forget_publication_edges()
        later = Vendor(finalized_through="2026-03-10")
        whole = _provider(later).get_ohlcv("AAPL", *self.WINDOW)
        assert [c["end"] for c in later.calls] == ["2026-03-11"], "asked again"
        assert str(whole.index[-1].date()) == "2026-03-10"
        assert len(_settled_files(tmp_path / "cache")) == 1

        reread = Vendor(finalized_through="2026-03-10")
        cached = _provider(reread).get_ohlcv("AAPL", *self.WINDOW)
        assert reread.calls == []
        pd.testing.assert_frame_equal(cached, whole, check_freq=False)

    def test_an_end_the_published_edge_clamped_is_not_written(self, tmp_path):
        """Intraday: the dataset had published only to 18:00 of the last
        day, so the bars after it were missing from the answer."""
        window = ("2026-03-09", "2026-03-10")
        lagging = {BASIC: (RANGES[BASIC][0], "2026-03-10T18:00:00+00:00")}
        short = _provider(Vendor(lagging)).get_ohlcv("AAPL", *window, interval="1h")
        assert short.index[-1] == pd.Timestamp("2026-03-10 17:00")
        assert _settled_files(tmp_path / "cache") == []
        whole = _provider(Vendor()).get_ohlcv("AAPL", *window, interval="1h")
        assert whole.index[-1] == pd.Timestamp("2026-03-10 20:00")
        assert len(_settled_files(tmp_path / "cache")) == 1

    def test_the_same_provider_keeps_a_short_answer_a_minute(self):
        provider = _provider(Vendor(finalized_through="2026-03-09"))
        provider.get_ohlcv("AAPL", *self.WINDOW)
        with cache_module._session_cache_lock:
            entries = list(cache_module._session_cache.values())
        assert len(entries) == 1 and entries[0][1] is not None

    def test_null_a_whole_answer_is_kept_for_the_hour(self):
        provider = _provider(Vendor())
        provider.get_ohlcv("AAPL", *self.WINDOW)
        with cache_module._session_cache_lock:
            entries = list(cache_module._session_cache.values())
        assert len(entries) == 1 and entries[0][1] is None
