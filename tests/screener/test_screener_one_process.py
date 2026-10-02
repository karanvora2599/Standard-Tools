"""
`screen_stocks` screens in one process unless asked otherwise.

Above 20 tickers it used to split the universe across up to cpu_count
worker processes, and that was the slow path: every worker is a fresh
interpreter that imports the library, builds the exchange calendar and
re-reads its bars before screening anything, and none of them can see the
caller's warm session cache. What the workers did buy -- more requests in
flight -- is kept by fetching on a wider thread pool in the one process.
A worker also cannot see the caller's open decision record or a Databento
request gate, so the pool is declined while either would be escaped. See
the CHANGELOG entry of 2026-10-01.

The batch path still exists for a caller who asks, and it must give the
same answer: here it runs on an in-process stand-in for the pool, so the
split-and-merge is compared with the one-process screen exactly.
"""

from __future__ import annotations

import logging
import os
import threading
import zlib
from concurrent.futures import Future
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.context import _data_sources_var
from standard_quant_tools.data import databento_provider
from standard_quant_tools.data.base import FinancialRatios
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.screener import screener
from standard_quant_tools.screener.screener import screen_stocks

FILTERS = {"pe_ratio_max": 30, "rsi_max": 65, "price_above_sma": 20}


def _reference_default_workers(n: int) -> int:
    """The worker count `screen_stocks` chose before, for n tickers."""
    return 1 if n <= 20 else min(os.cpu_count() or 4, max(n // 10, 2))


class _Market:
    """A provider answering from the ticker's name: ratios, then bars."""

    def __init__(self, barrier: Optional[threading.Barrier] = None) -> None:
        self.barrier = barrier
        self.threads: set = set()
        self._lock = threading.Lock()

    def get_financial_ratios(self, ticker: str) -> FinancialRatios:
        with self._lock:
            self.threads.add(threading.current_thread().name)
        if self.barrier is not None:
            self.barrier.wait()
        if ticker.startswith("BAD"):
            raise RuntimeError(f"no fundamentals for {ticker}")
        h = zlib.crc32(ticker.encode())
        return FinancialRatios(forward_pe=5.0 + h % 40, market_cap=1e9 * (1 + h % 50))

    async def get_ohlcv_async(self, ticker, start, end, interval="1d"):
        days = pd.bdate_range(start, end)
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(days))))
        return pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": 1e6,
            },
            index=days,
        )


class _InlinePool:
    """`ProcessPoolExecutor`, run in this process: the batch path's split
    and merge, without the spawn. Records that it was asked for."""

    created: List[int] = []

    def __init__(self, max_workers: int) -> None:
        _InlinePool.created.append(max_workers)

    def __enter__(self) -> "_InlinePool":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def submit(self, fn, *args) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001 - delivered through the future
            future.set_exception(exc)
        return future


class _NoPool:
    def __init__(self, *a: Any, **k: Any) -> None:
        raise AssertionError("a worker process was started")


@pytest.fixture
def market(monkeypatch) -> _Market:
    provider = _Market()
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
    return provider


@pytest.fixture
def inline_pool(monkeypatch):
    _InlinePool.created = []
    monkeypatch.setattr(screener, "ProcessPoolExecutor", _InlinePool)
    return _InlinePool


def _universe(n: int) -> List[str]:
    return [f"T{i:03d}" for i in range(n - 2)] + ["BAD1", "BAD2"]


def _same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.attrs == b.attrs


class TestTheDefault:
    def test_a_large_universe_is_screened_in_this_process(self, market, monkeypatch):
        monkeypatch.setattr(screener, "ProcessPoolExecutor", _NoPool)
        out = screen_stocks(_universe(235), FILTERS, "2024-01-02", "2024-12-31")
        assert _reference_default_workers(235) > 1, "the old default split it"
        assert (
            len(out)
            + len(out.attrs["failed_filters"])
            + len(out.attrs["failed_tickers"])
            == 235
        )
        assert set(out.attrs["failed_tickers"]) == {"BAD1", "BAD2"}
        assert out.attrs["failed_batches"] == []

    def test_the_old_default_split_gives_the_same_answer(self, market, inline_pool):
        """The pre-change default -- the universe in batches across
        `_reference_default_workers` processes -- against the new one."""
        tickers = _universe(235)
        before = screen_stocks(
            tickers,
            FILTERS,
            "2024-01-02",
            "2024-12-31",
            sort_by="rsi_14",
            n_workers=_reference_default_workers(len(tickers)),
        )
        assert inline_pool.created == [_reference_default_workers(len(tickers))]
        now = screen_stocks(
            tickers, FILTERS, "2024-01-02", "2024-12-31", sort_by="rsi_14"
        )
        assert inline_pool.created == [_reference_default_workers(len(tickers))]
        _same(before, now)
        assert len(now) > 0 and now.attrs["failed_filters"]

    @pytest.mark.parametrize("workers", [2, 3, 7, 16])
    def test_every_worker_count_gives_the_same_answer(
        self, market, inline_pool, workers
    ):
        tickers = _universe(61)
        one = screen_stocks(tickers, FILTERS, "2024-01-02", "2024-12-31", n_workers=1)
        many = screen_stocks(
            tickers, FILTERS, "2024-01-02", "2024-12-31", n_workers=workers
        )
        assert inline_pool.created == [workers]
        many.attrs.pop("failed_batches")
        one.attrs.pop("failed_batches")
        _same(one, many)

    def test_an_explicit_pool_is_still_used(self, market, inline_pool):
        screen_stocks(_universe(30), FILTERS, "2024-01-02", "2024-12-31", n_workers=4)
        assert inline_pool.created == [4]


class TestFetchConcurrency:
    def test_forty_fetches_are_in_flight_at_once(self, monkeypatch):
        """Planted: every ratios fetch waits until forty are waiting. The
        event loop's default executor holds twenty threads on a machine
        with sixteen cores, and twenty would never release the barrier."""
        barrier = threading.Barrier(40, timeout=20)
        provider = _Market(barrier)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
        tickers = [f"T{i:03d}" for i in range(40)]
        out = screen_stocks(tickers, {"pe_ratio_max": 100}, n_workers=1)
        assert out.attrs["failed_tickers"] == {}
        assert len(out) == 40
        assert len(provider.threads) == 40
        assert all(name.startswith("screener") for name in provider.threads)

    def test_the_threads_are_capped(self, market):
        tickers = [f"T{i:03d}" for i in range(150)]
        screen_stocks(tickers, {"pe_ratio_max": 100})
        assert 1 < len(market.threads) <= screener._FETCH_THREADS

    def test_one_ticker_needs_one_thread(self, market):
        screen_stocks(["T000"], {"pe_ratio_max": 100})
        assert len(market.threads) == 1


class TestThePoolIsDeclinedWhenItWouldEscapeTheCaller:
    def test_an_open_decision_record_keeps_the_screen_here(
        self, market, monkeypatch, caplog
    ):
        monkeypatch.setattr(screener, "ProcessPoolExecutor", _NoPool)
        token = _data_sources_var.set([])
        try:
            with caplog.at_level(logging.WARNING, logger=screener.__name__):
                out = screen_stocks(
                    _universe(30), FILTERS, "2024-01-02", "2024-12-31", n_workers=4
                )
        finally:
            _data_sources_var.reset(token)
        assert len(out) > 0
        assert "decision record is open" in caplog.text

    def test_a_databento_request_gate_keeps_a_databento_screen_here(
        self, market, monkeypatch, caplog
    ):
        class Gate:
            def before(self, request):
                return None

            def after(self, request, verdict, fetched):
                return None

        monkeypatch.setattr(screener, "ProcessPoolExecutor", _NoPool)
        previous = databento_provider.set_request_gate(Gate())
        try:
            with caplog.at_level(logging.WARNING, logger=screener.__name__):
                screen_stocks(
                    _universe(30),
                    FILTERS,
                    "2024-01-02",
                    "2024-12-31",
                    n_workers=4,
                    source="databento",
                )
        finally:
            databento_provider.set_request_gate(previous)
        assert "request gate is registered" in caplog.text

    def test_null_case_a_gate_does_not_hold_another_source(self, market, inline_pool):
        class Gate:
            def before(self, request):
                return None

            def after(self, request, verdict, fetched):
                return None

        previous = databento_provider.set_request_gate(Gate())
        try:
            screen_stocks(
                _universe(30),
                FILTERS,
                "2024-01-02",
                "2024-12-31",
                n_workers=4,
                source="yfinance",
            )
        finally:
            databento_provider.set_request_gate(previous)
        assert inline_pool.created == [4]

    def test_null_case_no_gate_and_no_record_uses_the_pool(self, market, inline_pool):
        screen_stocks(
            _universe(30),
            FILTERS,
            "2024-01-02",
            "2024-12-31",
            n_workers=4,
            source="databento",
        )
        assert inline_pool.created == [4]
