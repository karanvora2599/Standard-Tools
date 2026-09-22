"""
Depth and order-by-order data, from a vendor request to a metric.

WHAT THIS CLOSES. The library has served L2 depth and market-by-order
since it grew a Databento provider, and nothing could ask for either: the
two tools that read a book accepted only an `sqt://` reference, and the
only way to mint one was `register_external_dataset`, which wants a file
the caller captured somewhere else. So the depth analytics were reachable
exactly for someone who already had depth. The three tools under test here
are the door (CHANGELOG entry of 2026-09-22), and the assertion that
matters is not "the fetch worked" but that what it registers is read
END TO END by the tools that were waiting for it.

EVERY TEST HERE IS OFFLINE. The vendor client is injected -- there is no
key in this environment and there should not need to be one, because
dataset routing, the metered-feed preflight and the cap that makes a
prefix visible are exactly the parts that are expensive to get wrong and
impossible to exercise against a live, billed API in a suite.

THE OTHER HALF is what these tools must not do. A metered feed cannot be
allowed to look free, a capped pull cannot be allowed to look like a short
session, and a coverage window nobody reported cannot be allowed to arrive
as a plausible-looking pair of dates. Each is asserted as a field or a
warning STRING rather than left to prose.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.data.databento import FIXED_PRICE_SCALE, SCHEMA_KINDS
from standard_quant_tools.data.factory import DataFactory
from tests.agent.test_data_runtime import _Provider
from tests.data.test_databento_provider import WIDE, StubClient, _mbp10, _provider

DEPTH = "XNAS.ITCH"
SUMMARY = "EQUS.SUMMARY"

#: Measured on one active name for five minutes: depth is the expensive
#: schema and market-by-order, despite being the deeper feed, is not.
DEPTH_BYTES = 41_900_000
EVENT_BYTES = 14_400_000


@pytest.fixture(autouse=True)
def _isolated_runs(tmp_path, monkeypatch):
    """Artifacts land in this test's own directory, and the dataset
    environment overrides are cleared so the routing under test is the
    library's own preference order rather than a developer's."""
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)


def _mbo(rows: int = 60) -> pd.DataFrame:
    """A raw market-by-order frame in the vendor's own spelling.

    Two events per order id, alternating adds with a cancel or a fill, so
    the lifetime and cancel-to-add statistics have something real to
    compute rather than a single censored population.
    """
    rng = np.random.default_rng(11)
    base = pd.Timestamp("2026-03-02 14:30", tz="UTC").value
    actions = np.array(["A", "C", "A", "F"] * (rows // 4 + 1))[:rows]
    return pd.DataFrame(
        {
            "ts_recv": (base + np.arange(rows) * 1_000_000).astype("int64"),
            "ts_event": (base + np.arange(rows) * 1_000_000 - 100_000).astype("int64"),
            "order_id": np.repeat(np.arange(rows // 2 + 1), 2)[:rows].astype("int64"),
            "action": actions,
            "side": np.where(np.arange(rows) % 2 == 0, "B", "A"),
            "price": np.round(
                (250.0 + rng.normal(0, 0.05, rows)) * FIXED_PRICE_SCALE
            ).astype("int64"),
            "size": rng.integers(100, 900, rows).astype("int64"),
            "flags": np.zeros(rows, dtype="int64"),
        }
    )


def _depth_provider(frame=None, ranges=None, billable=DEPTH_BYTES):
    """A Databento provider answering from an injected client."""
    client = StubClient(
        {DEPTH: WIDE} if ranges is None else ranges,
        default=_mbp10(rows=40) if frame is None else frame,
        billable=billable,
    )
    return _provider(client), client


def _data(tool, arguments, provider):
    with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
        return resolve_runtime("data").dispatch(tool, arguments)


def _micro(tool, arguments):
    """The consumer side, with no provider in sight -- a registered
    reference is read by whoever holds the string."""
    return resolve_runtime("microstructure").dispatch(tool, arguments)


class TestAFetchedBookIsReadByTheToolsThatWaitedForIt:
    def test_a_served_book_becomes_a_reference_the_metrics_read(self):
        """
        THE TEST THIS INCREMENT EXISTS FOR. The fetch happens against a
        provider; the read happens through the reference alone, in another
        runtime, with nothing resolving a provider at all.
        """
        provider, _client = _depth_provider()
        fetched = _data(
            "fetch_order_book",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "levels": 10,
                "limit": 5_000,
                "run_id": "depth_round_trip",
                "name": "aapl_book",
            },
            provider,
        )
        assert fetched["kind"] == "order_book_panel"
        assert fetched["ref"].startswith("sqt://order_book_panel/")
        assert fetched["rows"] == 40
        assert fetched["entities"] == ["AAPL"]
        # `timestamp` is a COLUMN on a book, not the index, and reading the
        # index instead would report the window as 0 to 39.
        assert fetched["start"].startswith("2026-03-02T14:30:00")
        assert fetched["end"] > fetched["start"]
        assert fetched["dataset"] == DEPTH
        assert fetched["provider"] == "databento"
        assert fetched["levels"] == 4

        metrics = _micro(
            "get_order_book_metrics", {"ref": fetched["ref"], "include_dynamics": True}
        )
        assert metrics["n_snapshots"] == 40
        assert metrics["mean_spread"] == pytest.approx(0.02, abs=1e-6)
        assert metrics["ofi"] is not None

    def test_a_served_order_feed_becomes_a_reference_the_metrics_read(self):
        provider, _client = _depth_provider(frame=_mbo(), billable=EVENT_BYTES)
        fetched = _data(
            "fetch_order_events",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "limit": 5_000,
                "run_id": "order_feed_round_trip",
                "name": "aapl_events",
            },
            provider,
        )
        assert fetched["kind"] == "order_event_panel"
        assert fetched["rows"] == 60
        # A book has levels; an order feed aggregates nothing and has none,
        # which is null rather than zero.
        assert fetched["levels"] is None

        metrics = _micro("get_order_event_metrics", {"ref": fetched["ref"]})
        assert metrics["n_events"] == 60
        assert metrics["counts_by_action"]["A"] == 30
        assert metrics["cancel_to_add"] == pytest.approx(0.5)

    def test_the_book_says_it_is_one_venues(self):
        """Depth is published per venue and never consolidated, so an
        imbalance computed from it is that venue's imbalance."""
        provider, _client = _depth_provider()
        fetched = _data(
            "fetch_order_book",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "limit": 5_000,
                "run_id": "one_venue_only",
                "name": "aapl_book",
            },
            provider,
        )
        assert any("ONE VENUE" in w for w in fetched["warnings"])


class TestACappedPullIsAPrefixRatherThanAShortSession:
    @pytest.mark.parametrize(
        "tool,frame,name",
        [
            ("fetch_order_book", None, "book"),
            ("fetch_order_events", "events", "events"),
        ],
    )
    def test_a_cap_below_the_served_rows_truncates_and_says_so(self, tool, frame, name):
        provider, _client = _depth_provider(frame=_mbo() if frame else None)
        fetched = _data(
            tool,
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "limit": 12,
                "run_id": f"cap_binds_{name}",
                "name": name,
            },
            provider,
        )
        assert fetched["truncated"] is True
        assert fetched["rows"] == 12
        assert any("PREFIX" in w for w in fetched["warnings"])

    @pytest.mark.parametrize(
        "tool,frame,name",
        [
            ("fetch_order_book", None, "book"),
            ("fetch_order_events", "events", "events"),
        ],
    )
    def test_a_cap_above_them_leaves_the_window_whole(self, tool, frame, name):
        provider, _client = _depth_provider(frame=_mbo() if frame else None)
        fetched = _data(
            tool,
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "limit": 100_000,
                "run_id": f"cap_is_slack_{name}",
                "name": name,
            },
            provider,
        )
        assert fetched["truncated"] is False
        assert not any("PREFIX" in w for w in fetched["warnings"])


class TestAProviderWithoutTheFeedRefusesByName:
    """A bars-only provider is the normal case, and the refusal has to say
    which provider serves depth and where to ask what this one can reach --
    a bare NotImplementedError reads to a caller as a library bug."""

    @pytest.fixture
    def bars_only(self):
        return _Provider()

    @pytest.mark.parametrize(
        "tool,extra",
        [("fetch_order_book", {"levels": 5}), ("fetch_order_events", {})],
    )
    def test_the_depth_fetches_point_at_the_capability_door(
        self, bars_only, tool, extra
    ):
        with pytest.raises(Exception) as caught:
            _data(
                tool,
                {
                    "symbol": "AAPL",
                    "start_date": "2026-03-02",
                    "end_date": "2026-03-02",
                    "run_id": f"bars_only_{tool}",
                    "name": "unreachable",
                    **extra,
                },
                bars_only,
            )
        message = str(caught.value)
        assert tool in message
        assert "describe_data_capabilities" in message
        assert "databento" in message

    def test_the_preflight_names_the_provider_that_can_answer(self, bars_only):
        with pytest.raises(Exception) as caught:
            _data(
                "preflight_vendor_request",
                {
                    "symbol": "AAPL",
                    "start_date": "2026-03-02",
                    "end_date": "2026-03-02",
                    "vendor_schema": "mbp-10",
                },
                bars_only,
            )
        assert "source='databento'" in str(caught.value)


class TestThePreflightAnswersBeforeTheRequestIsMade:
    def test_it_returns_the_routed_dataset_the_window_and_the_bytes(self):
        provider, client = _depth_provider()
        report = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "vendor_schema": "mbp-10",
            },
            provider,
        )
        assert report["dataset"] == DEPTH
        assert report["datasets_considered"] == [DEPTH]
        assert report["coverage_start"].startswith("2018-01-01")
        assert report["coverage_end"].startswith("2026-09-02")
        assert report["covers_request"] is True
        assert report["billable_bytes"] == DEPTH_BYTES
        assert report["kind"] == "order_book_panel"
        # The quote is a metadata call; nothing was transferred.
        assert client.calls == []
        assert client.metadata.billable_calls[0]["schema"] == "mbp-10"

    def test_bytes_are_reported_because_a_subscription_prices_at_zero(self):
        provider, _client = _depth_provider()
        report = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "vendor_schema": "mbo",
            },
            provider,
        )
        assert any("BYTES, NOT MONEY" in w for w in report["warnings"])
        assert report["kind"] == "order_event_panel"

    def test_the_routing_follows_the_window_rather_than_a_fixed_feed(self):
        """The daily summary feed does not exist before its start date, so
        a window below it is answered by a different dataset entirely -- a
        sample feed carrying a few percent of consolidated volume. Which one
        answered is the whole reason the preflight reports a dataset."""
        provider, _client = _depth_provider(
            ranges={SUMMARY: WIDE, "XNAS.BASIC": WIDE, DEPTH: WIDE}
        )
        recent = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2025-01-02",
                "end_date": "2025-01-31",
                "vendor_schema": "ohlcv-1d",
            },
            provider,
        )
        early = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2019-01-02",
                "end_date": "2019-01-31",
                "vendor_schema": "ohlcv-1d",
            },
            provider,
        )
        assert recent["dataset"] == SUMMARY
        assert SUMMARY not in early["datasets_considered"]
        assert early["dataset"] != SUMMARY

    def test_a_provider_that_reports_no_coverage_yields_nulls(self):
        """A window nobody reported is null. Inventing a plausible pair of
        dates here would be inventing the one number a caller plans the
        request around."""
        provider, _client = _depth_provider(ranges={}, billable=None)
        report = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2026-03-02",
                "end_date": "2026-03-02",
                "vendor_schema": "mbp-10",
            },
            provider,
        )
        assert report["coverage_start"] is None
        assert report["coverage_end"] is None
        assert report["covers_request"] is None
        assert report["billable_bytes"] is None
        assert any("null rather than zero" in w for w in report["warnings"])

    def test_a_window_the_dataset_does_not_cover_is_said_so(self):
        provider, _client = _depth_provider(
            ranges={DEPTH: ("2024-01-01T00:00:00+00:00", "2024-06-30T00:00:00+00:00")}
        )
        report = _data(
            "preflight_vendor_request",
            {
                "symbol": "AAPL",
                "start_date": "2024-06-01",
                "end_date": "2024-12-31",
                "vendor_schema": "mbp-10",
            },
            provider,
        )
        assert report["covers_request"] is False
        assert any("does not contain the window" in w for w in report["warnings"])

    def test_an_inverted_window_is_refused_by_name(self):
        provider, _client = _depth_provider()
        with pytest.raises(Exception) as caught:
            _data(
                "preflight_vendor_request",
                {
                    "symbol": "AAPL",
                    "start_date": "2026-03-05",
                    "end_date": "2026-03-02",
                    "vendor_schema": "mbp-10",
                },
                provider,
            )
        assert "end_date" in str(caught.value)


class TestTheSchemaMapNamesWhatTheSchemaProduces:
    def test_market_by_order_is_an_order_event_panel(self):
        """It said `order_book_panel` and had no reader until the depth
        fetch tools acquired one. An order tape registered under the book's
        kind passes registration and then fails inside the book statistics
        on a `bid_price_0` market-by-order does not have."""
        assert SCHEMA_KINDS["mbo"] == "order_event_panel"
        assert SCHEMA_KINDS["mbp-10"] == "order_book_panel"


class TestTheProviderContractAnswersForEveryProvider:
    def test_a_bars_only_provider_declines_coverage_and_pricing_by_name(self):
        provider = _Provider()
        for call in (
            lambda: provider.get_dataset_coverage(),
            lambda: provider.get_billable_size(
                "AAPL", "2026-03-02", "2026-03-02", "trades"
            ),
        ):
            with pytest.raises(NotImplementedError) as caught:
                call()
            assert "databento" in str(caught.value)

    def test_coverage_leaves_out_what_the_vendor_would_not_report(self):
        provider, _client = _depth_provider(ranges={DEPTH: WIDE})
        coverage = provider.get_dataset_coverage([DEPTH, "EQUS.MINI"])
        assert set(coverage) == {DEPTH}
        assert coverage[DEPTH][0].startswith("2018-01-01")


class TestTheThreeToolsAreReachableFromEverySurface:
    NAMES = ("fetch_order_book", "fetch_order_events", "preflight_vendor_request")

    def test_the_runtime_advertises_and_dispatches_each_of_them(self):
        from standard_quant_tools.agent.runtimes import data as data_runtime

        advertised = {name for name, _description, _model in data_runtime.TOOL_DEFS}
        for name in self.NAMES:
            assert name in advertised
            assert name in data_runtime.TOOL_DISPATCH
            assert name in data_runtime.__all__
            assert data_runtime.TOOL_CATEGORY[name] == "data"

    def test_the_facade_dispatches_and_exports_each_of_them(self):
        import standard_quant_tools.agent as package
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        for name in self.NAMES:
            assert name in _TOOL_DISPATCH
            assert name in package.__all__
            assert callable(getattr(package, name))

    def test_each_description_carries_the_measured_cost(self):
        """A metered feed whose schema does not say so is a bill nobody
        agreed to. The byte figures are measured, not estimated."""
        from standard_quant_tools.agent.runtimes import data as data_runtime

        described = {
            name: description for name, description, _model in data_runtime.TOOL_DEFS
        }
        assert "42 MB" in described["fetch_order_book"]
        assert "20,000" in described["fetch_order_book"]
        assert "14 MB" in described["fetch_order_events"]
        assert "100,000" in described["fetch_order_events"]
        assert "get_order_book_metrics" in described["fetch_order_book"]
        assert "get_order_event_metrics" in described["fetch_order_events"]


class TestAnArtifactWithoutARunIsNotWritten:
    def test_a_fetch_with_no_run_id_is_refused_and_leaves_nothing_behind(
        self, tmp_path
    ):
        """`run_id` is what groups a workflow's artifacts and what makes a
        reference resolvable, so it is required rather than defaulted --
        and the refusal happens before anything reaches the disk."""
        provider, client = _depth_provider()
        with pytest.raises(Exception) as caught:
            _data(
                "fetch_order_book",
                {
                    "symbol": "AAPL",
                    "start_date": "2026-03-02",
                    "end_date": "2026-03-02",
                    "name": "no_run",
                },
                provider,
            )
        assert "run_id" in str(caught.value)
        assert client.calls == []
        runs = tmp_path / "runs"
        assert not runs.exists() or not list(runs.rglob("*.parquet"))

    def test_a_second_fetch_under_one_name_is_refused_rather_than_replacing(self):
        """A reference promises that resolving it twice gives the same
        value, so the collision fails loudly and names the remedy."""
        provider, _client = _depth_provider()
        arguments = {
            "symbol": "AAPL",
            "start_date": "2026-03-02",
            "end_date": "2026-03-02",
            "limit": 5_000,
            "run_id": "one_name_one_dataset",
            "name": "aapl_book",
        }
        _data("fetch_order_book", arguments, provider)
        with pytest.raises(Exception) as caught:
            _data("fetch_order_book", dict(arguments), provider)
        assert "run_id" in str(caught.value)
