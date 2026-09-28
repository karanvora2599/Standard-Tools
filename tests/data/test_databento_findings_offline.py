"""
The live findings, reproduced offline: phase 1 of
the Databento live fix plan (CHANGELOG, 2026-09-20).

Each test here is a finding from the live pass planted into
the stub client, which now answers the window it is asked for. The
questions are the ones the live pass asked: does a daily request return
tomorrow (D1); does the index reach the modeling runtime tz-naive (D2);
which feed answers a daily window and does the frame say so (D3, D12);
does a futures root resolve to an equity (D5); do bars and ticks come
from one tape (D11); does a repeated request cost a second fetch.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.audit.context import _data_sources_var
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data.databento import (
    DATASET_CONSOLIDATED,
    DATASET_DEPTH,
    DATASET_FUTURES,
    DATASET_NASDAQ_BASIC,
    DATASET_OPTIONS,
    DATASET_SUMMARY,
)
from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.data.quality import (
    detect_missing_bars,
    detect_volume_anomalies,
)
from standard_quant_tools.error import ValidationError

from .test_databento_provider import (
    BASIC,
    CONSOLIDATED,
    DEPTH,
    SINCE_2023,
    WIDE,
    StubClient,
    _provider,
)

SUMMARY = DATASET_SUMMARY
SINCE_2024_07 = ("2024-07-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00")
ALL = {SUMMARY: SINCE_2024_07, CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE}


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)


class TestD1TheDailyEndIsInclusiveAndNothingMore:
    def test_a_single_day_returns_that_day_and_no_tomorrow(self):
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2025-09-16", "2025-09-16")
        assert [str(d.date()) for d in frame.index] == ["2025-09-16"]
        # The request itself stops at the exclusive boundary the inclusive
        # end implies, not one day past it.
        assert client.calls[0]["end"] == "2025-09-17"

    def test_a_window_ends_on_the_day_asked(self):
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2025-09-10", "2025-09-16")
        assert str(frame.index[-1].date()) == "2025-09-16"
        assert len(frame) == 5  # the 10th, 11th, 14th, 15th, 16th
        assert len(client.calls) == 1  # no guaranteed-to-fail first attempt

    def test_an_unfinalized_tail_still_walks_back(self):
        client = StubClient(ALL, finalized_through="2025-09-15")
        frame = _provider(client).get_ohlcv("AAPL", "2025-09-10", "2025-09-17")
        assert str(frame.index[-1].date()) == "2025-09-15"
        assert [c["end"] for c in client.calls] == [
            "2025-09-18",
            "2025-09-17",
            "2025-09-16",
        ]


class TestD2TheIndexReachesEveryConsumerNaive:
    def test_bars_are_naive_dated_and_integer_volume(self):
        frame = _provider(StubClient(ALL)).get_ohlcv("AAPL", "2025-09-10", "2025-09-16")
        assert frame.index.tz is None
        assert frame.index.dtype == "datetime64[ns]"
        assert frame["Volume"].dtype == "int64"
        assert (frame["Volume"].diff().dropna() == 0).all()  # no 1.8e19 from uint64

    def test_a_modeling_dataset_builds_on_this_provider(self, monkeypatch):
        from standard_quant_tools.data.factory import DataFactory
        from standard_quant_tools.modeling.dataset.builder import build_dataset
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            FeatureSpec,
            TargetSpec,
        )

        provider = _provider(StubClient(ALL))
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
        built = build_dataset(
            DatasetSpec(
                universe=["AAPL", "MSFT"],
                start="2025-01-02",
                end="2025-12-31",
                features=[
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(id="market.momentum"),
                ],
                target=TargetSpec(horizon=5),
                provider="databento",
            )
        )
        panel = built["panel"]
        assert len(panel) > 100
        assert pd.DatetimeIndex(panel["date"]).tz is None
        assert panel["label_end_date"].notna().any()
        assert built["data_sources"] == {
            "AAPL": f"databento:{SUMMARY}",
            "MSFT": f"databento:{SUMMARY}",
        }


class TestD3TheDailyFeedIsTheTapeWhereItCanBe:
    def test_the_summary_feed_answers_a_daily_window_it_covers(self):
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert client.datasets_called()[0] == SUMMARY
        assert frame.attrs["dataset"] == SUMMARY

    def test_a_window_before_the_summary_feed_never_asks_it(self):
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2023-06-01", "2023-06-09")
        assert SUMMARY not in client.datasets_called()
        assert client.datasets_called()[0] == CONSOLIDATED
        assert frame.attrs["dataset"] == CONSOLIDATED

    def test_intraday_never_asks_the_summary_or_sample_feed(self):
        client = StubClient(ALL)
        provider = _provider(client)
        frame = provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-04", interval="1h")
        assert client.datasets_called() == [DATASET_NASDAQ_BASIC]
        assert frame.attrs["dataset"] == DATASET_NASDAQ_BASIC
        notes = " ".join(provider.get_metadata("AAPL", "1h").notes)
        assert "single-venue" in notes and "No consolidated intraday" in notes
        daily_notes = " ".join(provider.get_metadata("AAPL", "1d").notes)
        assert SUMMARY in daily_notes and "SAMPLE" in daily_notes


class TestD5AFuturesRootIsNotAnEquity:
    @pytest.mark.parametrize(
        "symbol,raw,stype,family",
        [
            ("ES.c.0", "ES.c.0", "continuous", "future"),
            ("es.C.1", "ES.c.1", "continuous", "future"),
            ("ES.FUT", "ES.FUT", "parent", "future"),
            ("ES.OPT", "ES.OPT", "parent", "option"),
            ("ESZ6", "ESZ6", "raw_symbol", "future"),
            ("CLF27", "CLF27", "raw_symbol", "future"),
            ("AAPL  240119C00190000", "AAPL  240119C00190000", "raw_symbol", "option"),
            ("AAPL240119C00190000", "AAPL  240119C00190000", "raw_symbol", "option"),
            ("ES~equity", "ES", "raw_symbol", "equity"),
            ("BRK.B", "BRK.B", "raw_symbol", "equity"),
            ("NVDA", "NVDA", "raw_symbol", "equity"),
        ],
    )
    def test_the_grammar_routes_each_spelling(self, symbol, raw, stype, family):
        """A share class keeps its dot: Databento's Historical symbology
        resolves `BRK.B` and not `BRKB`, which this row used to expect."""
        route = DatabentoProvider.resolve_symbol(symbol)
        assert (route.raw, route.stype_in, route.family) == (raw, stype, family)

    @pytest.mark.parametrize("root", ["ES", "CL", "GC", "NQ", "SI"])
    def test_a_bare_root_that_is_also_a_ticker_is_refused(self, root):
        client = StubClient(ALL)
        with pytest.raises(ValidationError, match="ambiguous"):
            _provider(client).get_ohlcv(root, "2025-03-03", "2025-03-07")
        assert client.calls == []

    def test_a_future_goes_to_the_futures_dataset_with_its_symbology(self):
        client = StubClient({**ALL, DATASET_FUTURES: WIDE})
        frame = _provider(client).get_ohlcv("ES.c.0", "2025-03-03", "2025-03-07")
        assert client.datasets_called() == [DATASET_FUTURES]
        assert client.calls[0]["stype_in"] == "continuous"
        assert client.calls[0]["symbols"] == ["ES.c.0"]
        assert frame.attrs["dataset"] == DATASET_FUTURES

    def test_an_option_goes_to_the_options_dataset(self):
        client = StubClient({**ALL, DATASET_OPTIONS: WIDE})
        _provider(client).get_ohlcv("AAPL  240119C00190000", "2024-01-02", "2024-01-05")
        assert client.datasets_called() == [DATASET_OPTIONS]
        assert client.calls[0]["stype_in"] == "raw_symbol"


class TestD11OneTapeForBarsAndTicks:
    def test_intraday_bars_and_trades_prefer_the_same_dataset(self, monkeypatch):
        from standard_quant_tools.data import databento_provider as module

        client = StubClient(ALL)
        provider = _provider(client)
        provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-04", interval="1m")
        monkeypatch.setattr(module, "normalize_trades", lambda f, **kw: (f, []))
        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2025-03-03 14:30", periods=3, freq="s", tz="UTC"
                ),
                "price": [1.0, 2.0, 3.0],
            }
        )
        client.default = trades
        provider.get_trades("AAPL", "2025-03-03", "2025-03-04")
        assert client.datasets_called() == [DATASET_NASDAQ_BASIC, DATASET_NASDAQ_BASIC]
        assert provider._tick_datasets("AAPL") == provider._bar_datasets("ohlcv-1m")


class TestTheSeams:
    def test_a_repeated_request_is_one_fetch_and_still_recorded(self):
        client = StubClient(ALL)
        provider = _provider(client)
        token = _data_sources_var.set([])
        try:
            first = provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
            second = provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
            sources = list(_data_sources_var.get() or [])
        finally:
            _data_sources_var.reset(token)
        assert len(client.calls) == 1
        assert first.equals(second) and second.attrs["dataset"] == SUMMARY
        assert [s["source"] for s in sources] == [
            f"databento:{SUMMARY}",
            f"databento:{SUMMARY}:session_cache",
        ]

    def test_the_disk_cache_is_keyed_by_the_feed_that_answered(self, tmp_path):
        client = StubClient(ALL)
        provider = _provider(client)
        provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        files = list((cache_module._CACHE_ROOT).rglob("*.parquet"))
        assert len(files) == 1 and f"databento-{SUMMARY}" in files[0].name
        # A fresh instance reads the file and knows which feed it came from.
        fresh_client = StubClient(ALL)
        again = _provider(fresh_client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert fresh_client.calls == []
        assert again.attrs["dataset"] == SUMMARY
        assert again.index.dtype == "datetime64[ns]"

    def test_a_file_written_before_the_databento_generation_is_not_read(self):
        """Databento's previous generation may hold a file named for
        `GOOG.L` with Alphabet class A's bars, so none of its files is read
        again: a planted one here is passed over for a fetch, and the
        collector lists it, while another provider's file at the shared
        version stays current."""
        root = cache_module._CACHE_ROOT
        root.mkdir(parents=True, exist_ok=True)
        poisoned = f"v3_databento-{SUMMARY}_AAPL_2025-03-03_2025-03-07_1d.parquet"
        _daily_file(root / poisoned, volume=1)
        (root / "v3_yfinance_AAPL_2025-03-03_2025-03-07_1d.parquet").write_bytes(b"x")
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert client.datasets_called() == [SUMMARY]
        assert int(frame["Volume"].iloc[0]) != 1
        dead = [p.name for p in cache_module.dead_generations(dry_run=True)]
        assert dead == [poisoned]
        written = [p.name for p in root.glob(f"*databento-{SUMMARY}*")]
        assert sorted(written) == sorted(
            [poisoned, f"v4_databento-{SUMMARY}_AAPL_2025-03-03_2025-03-07_1d.parquet"]
        )

    def test_the_temporal_contract_agrees_with_the_metadata(self):
        provider = _provider(StubClient(ALL))
        assert provider.get_temporal_contract("bars").revisions == "unknown"
        assert provider.get_metadata("AAPL").point_in_time is False


def _daily_file(path, volume):
    """A cache file as this provider writes one: five sessions, naive
    dates, the five columns."""
    index = pd.bdate_range("2025-03-03", "2025-03-07")
    close = 200.0 + np.arange(len(index))
    pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(index), volume, dtype="int64"),
        },
        index=index,
    ).to_parquet(path)


def _volumes(frame):
    return int(frame["Volume"].iloc[0])


class TestTheDiskCacheAnswersWithTheFeedThatWouldAnswer:
    """
    Each feed's answer is its own disk entry, and the lookup used to serve
    the first entry it found among every candidate. One failure on the
    summary feed, or one run with `DATABENTO_OHLCV_DATASET` set, left a
    sample-feed file that every later call served -- 5% of the true volume
    -- for a window the summary feed covered all along.
    """

    WINDOW = ("2025-03-03", "2025-03-07")
    VOLUME = {SUMMARY: 36_000_000, CONSOLIDATED: 1_700_000}

    def _client(self, rules=None, ranges=None):
        """Bars whose volume says which feed answered."""
        client = StubClient(ranges or ALL, rules=rules)
        volume = self.VOLUME

        def by_feed(kw):
            if kw["dataset"] not in volume:
                return None
            index = pd.bdate_range(
                pd.Timestamp(kw["start"], tz="UTC"),
                pd.Timestamp(kw["end"], tz="UTC") - pd.Timedelta(days=1),
                tz="UTC",
            )
            close = 200.0 + np.arange(len(index))
            return pd.DataFrame(
                {
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": np.full(len(index), volume[kw["dataset"]], "uint64"),
                },
                index=index,
            )

        client.rules.append(by_feed)
        return client

    def _fresh_healthy_read(self):
        client = self._client()
        frame = _provider(client).get_ohlcv("AAPL", *self.WINDOW)
        return frame, client

    def _assert_the_summary_feed_answers(self):
        frame, client = self._fresh_healthy_read()
        assert client.datasets_called() == [SUMMARY]
        assert frame.attrs["dataset"] == SUMMARY
        assert _volumes(frame) == self.VOLUME[SUMMARY]

    def test_a_failed_data_request_does_not_pin_the_window(self):
        failing = self._client(
            rules=[
                lambda kw: (
                    RuntimeError("500 gateway error")
                    if kw["dataset"] == SUMMARY
                    else None
                )
            ]
        )
        first = _provider(failing).get_ohlcv("AAPL", *self.WINDOW)
        assert first.attrs["dataset"] == CONSOLIDATED
        self._assert_the_summary_feed_answers()

    def test_a_failed_range_lookup_does_not_pin_the_window(self):
        client = self._client()
        original = client.metadata.get_dataset_range

        def flaky(dataset):
            if dataset == SUMMARY:
                raise RuntimeError("503 service unavailable")
            return original(dataset)

        client.metadata.get_dataset_range = flaky
        first = _provider(client).get_ohlcv("AAPL", *self.WINDOW)
        assert first.attrs["dataset"] == CONSOLIDATED
        assert SUMMARY not in client.datasets_called()
        self._assert_the_summary_feed_answers()

    def test_a_dataset_override_used_once_does_not_pin_the_window(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_OHLCV_DATASET", CONSOLIDATED)
        first = _provider(self._client()).get_ohlcv("AAPL", *self.WINDOW)
        assert first.attrs["dataset"] == CONSOLIDATED
        monkeypatch.delenv("DATABENTO_OHLCV_DATASET")
        self._assert_the_summary_feed_answers()

    def test_the_same_instance_asks_the_preferred_feed_again(self):
        """The degraded answer is not kept for the session either: the next
        call on the instance that saw the failure asks the summary feed."""
        state = {"failures": 1}

        def once(kw):
            if kw["dataset"] == SUMMARY and state["failures"]:
                state["failures"] -= 1
                return RuntimeError("500 gateway error")
            return None

        client = self._client(rules=[once])
        provider = _provider(client)
        assert provider.get_ohlcv("AAPL", *self.WINDOW).attrs["dataset"] == (
            CONSOLIDATED
        )
        again = provider.get_ohlcv("AAPL", *self.WINDOW)
        assert again.attrs["dataset"] == SUMMARY
        assert _volumes(again) == self.VOLUME[SUMMARY]

    def test_a_denied_preferred_feed_still_reads_the_fallback_entry(self):
        """
        A subscription without the summary feed gets the sample feed's
        answer from disk on the next fresh provider, not a metered refetch:
        once the denial is learned, the sample feed is the one that would
        answer.
        """
        without_summary = {k: v for k, v in ALL.items() if k != SUMMARY}
        first = _provider(self._client(ranges=without_summary)).get_ohlcv(
            "AAPL", *self.WINDOW
        )
        assert first.attrs["dataset"] == CONSOLIDATED
        client = self._client(ranges=without_summary)
        again = _provider(client).get_ohlcv("AAPL", *self.WINDOW)
        assert client.calls == []
        assert again.attrs["dataset"] == CONSOLIDATED
        assert _volumes(again) == self.VOLUME[CONSOLIDATED]

    def test_a_request_level_denial_still_reads_the_fallback_entry(self):
        """The same when the denial arrives on the data request rather than
        the range lookup: the one refused request, and no data refetched."""
        denied = [
            lambda kw: (
                RuntimeError("403 Forbidden: not_entitled")
                if kw["dataset"] == SUMMARY
                else None
            )
        ]
        _provider(self._client(rules=denied)).get_ohlcv("AAPL", *self.WINDOW)
        client = self._client(rules=denied)
        again = _provider(client).get_ohlcv("AAPL", *self.WINDOW)
        assert client.datasets_called() == [SUMMARY]
        assert again.attrs["dataset"] == CONSOLIDATED

    def test_null_a_clean_summary_answer_is_served_warm(self):
        self._assert_the_summary_feed_answers()
        frame, client = self._fresh_healthy_read()
        assert client.calls == []
        assert frame.attrs["dataset"] == SUMMARY
        assert _volumes(frame) == self.VOLUME[SUMMARY]

    def test_null_a_window_before_the_summary_feed_reads_the_sample_entry(self):
        """Policy, not a pin: the summary feed does not reach 2024-06, so
        the sample feed is the preferred answer there, warm as well as
        cold."""
        window = ("2024-06-03", "2024-06-07")
        cold = self._client()
        first = _provider(cold).get_ohlcv("AAPL", *window)
        assert cold.datasets_called() == [CONSOLIDATED]
        warm = self._client()
        again = _provider(warm).get_ohlcv("AAPL", *window)
        assert warm.calls == []
        assert again.attrs["dataset"] == CONSOLIDATED
        assert again.equals(first)


class TestQualityReadsTheCalendarAndTheVolume:
    def test_a_holiday_is_not_a_gap_and_a_missing_session_is(self):
        pytest.importorskip("exchange_calendars")
        # 2024-07-04 (Independence Day) is closed; 2024-07-08 is a session.
        index = pd.DatetimeIndex(
            [
                "2024-07-01",
                "2024-07-02",
                "2024-07-03",
                "2024-07-05",
                "2024-07-09",
                "2024-07-10",
            ]
        )
        frame = pd.DataFrame({"Close": np.arange(6.0), "Volume": 1.0}, index=index)
        gaps = detect_missing_bars(frame)
        assert [g["date"] for g in gaps] == ["2024-07-08"]
        assert gaps[0]["basis"] == "calendar"

    def test_zero_and_thin_volume_are_findings(self):
        index = pd.bdate_range("2024-01-02", periods=40)
        volume = np.full(40, 1_000_000.0)
        volume[30] = 0.0
        volume[35] = 20_000.0
        frame = pd.DataFrame({"Close": 100.0, "Volume": volume}, index=index)
        found = detect_volume_anomalies(frame)
        assert [(f["kind"], f["date"]) for f in found] == [
            ("zero", str(index[30].date())),
            ("thin", str(index[35].date())),
        ]
        assert found[1]["trailing_median"] == 1_000_000.0
