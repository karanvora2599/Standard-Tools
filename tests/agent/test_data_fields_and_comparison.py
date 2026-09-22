"""
What the data layer knew and did not say.

FOUR SILENCES, all of them about data that already existed somewhere in the
process and stopped at a result model.

1. WHICH VENDOR DATASET ANSWERED. A provider that chooses between feeds by
   date writes its choice onto the frame, and the choice changes the
   numbers -- one feed is the consolidated tape and another is a
   single-venue sample carrying a few percent of volume whose daily close
   is often an after-hours print. The rows look identical either way. The
   attribute survived publication, a process boundary and a resolve in
   another runtime, and was read by nothing.

2. WHAT SEPARATES THE PROVIDERS. `describe_data_capabilities` is the tool an
   agent is told to consult before choosing a source, and it reported none
   of the four capabilities on which the shipped providers actually differ.

3. WHAT A LABEL BUYS. `build_data_bundle` takes a `frame_kind` per frame,
   and that label is not a comment: it chooses the temporal contract the
   bundle is validated under. A returns panel labelled `fundamentals` was
   accepted and then answered a point-in-time question confidently about
   something it did not hold.

4. WHAT "NO DISAGREEMENT" MEANS. `compare_ratio_frames` read keys the
   library never emitted and counted "neither source reported this" as a
   disagreement, so the most natural composition on the surface -- fetch
   ratios from two providers, compare them -- reported eight of eight
   fields in conflict on two IDENTICAL inputs.

Each test below plants the answer, so what is asserted is the number the
library computed rather than the shape of the result. See the CHANGELOG
entry of 2026-09-22.
"""

from __future__ import annotations

from unittest.mock import patch

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.agent.runtimes.data.tools import _BUNDLE_KINDS
from standard_quant_tools.agent.runtimes.handoff import KINDS
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data.base import (
    DataProvider,
    DataSetMetadata,
    FinancialRatios,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError

#: The dataset the stub provider says answered. A real one of these carries
#: a few percent of consolidated volume, which is the reason the name has
#: to reach the caller rather than stay on the frame.
SAMPLE_DATASET = "EQUS.MINI"

_SESSIONS = pd.DatetimeIndex(
    xcals.get_calendar("XNYS").sessions_in_range(
        pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31")
    )
).tz_localize(None)


def _bars(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    index = _SESSIONS if index is None else index
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(index))))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=index,
    )


class _Bars(DataProvider):
    """Bars and nothing else, recording nothing about where they came from
    -- the provider most environments actually have."""

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        return _bars()

    async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
        return _bars()

    def get_ticker_info(self, symbol):
        raise NotImplementedError

    def get_financial_ratios(self, symbol):
        raise NotImplementedError

    def get_metadata(self, symbol, interval="1d"):
        return DataSetMetadata(
            provider="stub",
            adjusted=True,
            survivorship_free=False,
            point_in_time=False,
            frequency=interval,
            timezone="America/New_York",
        )


class _Stamped(_Bars):
    """A provider that stamps its bars the way a multi-feed vendor does."""

    NOTES = [
        "Served by a SAMPLE feed below 2024-07-01: a few percent of "
        "consolidated volume, and a daily close that is often an "
        "after-hours print.",
        "The index is naive session dates, like every provider's.",
    ]

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        frame = _bars()
        frame.attrs.update(
            {
                "dataset": SAMPLE_DATASET,
                "provider": "databento",
                "adjusted": False,
            }
        )
        return frame

    def get_metadata(self, symbol, interval="1d"):
        return DataSetMetadata(
            provider="databento",
            adjusted=False,
            survivorship_free=True,
            point_in_time=False,
            frequency=interval,
            timezone="UTC",
            notes=list(self.NOTES),
        )


def _data(tool, arguments):
    return resolve_runtime("data").dispatch(tool, arguments)


def _serving(provider):
    return patch.object(DataFactory, "get_provider", lambda *a, **k: provider)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _fetch(run_id="r", name="bars"):
    return _data(
        "fetch_ohlcv",
        {
            "symbol": "AAPL",
            "start_date": "2024-01-02",
            "end_date": "2024-12-31",
            "run_id": run_id,
            "name": name,
        },
    )


# ── 1. which vendor dataset answered ─────────────────────────────────────


class TestTheDatasetThatAnsweredReachesTheCaller:
    def test_a_fetch_names_the_dataset_the_provider_chose(self, runs):
        with _serving(_Stamped()):
            result = _fetch()
        assert result["dataset"] == SAMPLE_DATASET
        assert result["provider"] == "databento"
        assert result["adjusted"] is False
        assert result["source"] == f"databento:{SAMPLE_DATASET}"

    def test_it_survives_publication_and_is_readable_from_the_reference(self, runs):
        """
        THE ONE THAT MATTERS. The agent that resolves a reference is
        usually not the one that fetched it -- different call, often a
        different process -- so the answer has to come off the stored
        frame rather than out of the fetching call's memory. The provider
        is patched away entirely before the reference is described.
        """
        with _serving(_Stamped()):
            ref = _fetch()["ref"]

        def _no_provider(*_a, **_k):
            raise AssertionError("describing a reference must not fetch")

        with patch.object(DataFactory, "get_provider", _no_provider):
            described = dispatch("describe_reference", {"ref": ref})

        assert described["dataset"] == SAMPLE_DATASET
        assert described["provider"] == "databento"
        assert described["adjusted"] is False
        assert described["source"] == f"databento:{SAMPLE_DATASET}"

    def test_a_provider_that_says_nothing_reports_null_not_blank(self, runs):
        """
        The null case, and the distinction it protects. An empty string
        would read as "no dataset", which is a claim; null reads as "this
        provider does not record one", which is the truth. `adjusted` is
        the sharp one -- null is not False.
        """
        with _serving(_Bars()):
            result = _fetch()
            ref = result["ref"]
        described = dispatch("describe_reference", {"ref": ref})
        for field in ("dataset", "provider", "adjusted", "source"):
            assert result[field] is None, field
            assert described[field] is None, field


# ── 2. what separates the providers ──────────────────────────────────────


class TestTheCapabilitiesThatDecideTheChoice:
    #: source -> (order_book, order_events, point_in_time_records,
    #: temporal_contract). Depth and order events are served by exactly one
    #: shipped provider; point-in-time records by exactly one other. Those
    #: are the facts that decide which source a piece of work can use, and
    #: this tool reported none of them.
    EXPECTED = {
        "databento": (True, True, False, True),
        "polygon": (False, False, True, True),
        "yfinance": (False, False, False, False),
        "bloomberg": (False, False, False, False),
    }

    @pytest.mark.parametrize("source", sorted(EXPECTED))
    def test_each_shipped_provider_reports_its_own_four(self, source):
        result = dispatch("describe_data_capabilities", {"source": source})
        got = (
            result["order_book"],
            result["order_events"],
            result["point_in_time_records"],
            result["temporal_contract"],
        )
        assert got == self.EXPECTED[source]

    def test_exactly_one_provider_serves_depth_and_events(self):
        """Stated as a property rather than per provider: if a second one
        ever serves depth, the sentence this tool prints about databento
        being the only source of it has gone stale."""
        depth = [
            s
            for s in self.EXPECTED
            if dispatch("describe_data_capabilities", {"source": s})["order_book"]
        ]
        assert depth == ["databento"]

    def test_the_class_answer_does_not_depend_on_a_credential(self, monkeypatch):
        """
        A provider with no key still has to answer "could I use depth if I
        configured this?". The capability is a fact about the class; only
        `available` is a fact about the environment. Naming the class was
        also what stopped this path raising KeyError for the one provider
        that was missing from the table.
        """
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        result = dispatch("describe_data_capabilities", {"source": "databento"})
        assert result["available"] is False
        assert result["unavailable_reason"]
        assert (result["order_book"], result["order_events"]) == (True, True)

    def test_a_recorded_call_says_which_dataset_it_read(self, monkeypatch):
        """
        The third place this was dropped. The decision record has written
        `source` and `interval` on every provider fetch since fetches
        started reporting themselves, and `explain_decision` -- the tool
        whose whole job is to say what a recorded call did -- returned
        neither. It is also the only place the answer survives, because
        the frame it was read from is long gone.
        """
        from standard_quant_tools.agent.runtimes.meta import tools as meta_tools

        record = {
            "request_id": "req-1",
            "timestamp_utc": "2026-09-22T00:00:00Z",
            "tool_name": "fetch_ohlcv",
            "status": "ok",
            "input": {"symbol": "AAPL"},
            "duration_ms": 1.0,
            "data_sources": [
                {
                    "symbol": "AAPL",
                    "start": "2019-01-02",
                    "end": "2019-12-31",
                    "interval": "1d",
                    "source": f"databento:{SAMPLE_DATASET}",
                    "content_hash": "abc123",
                }
            ],
        }
        monkeypatch.setattr(meta_tools, "_find_audit_record", lambda request_id: record)
        result = dispatch("explain_decision", {"request_id": "req-1"})
        (read,) = result["data_sources"]
        assert read["source"] == f"databento:{SAMPLE_DATASET}"
        assert read["interval"] == "1d"

    def test_databento_is_a_legal_value_of_the_field_that_reports_it(self):
        """The provider that serves depth was absent from the description
        of the argument that selects it, so the tool that exists to report
        coverage did not list the only source with the coverage."""
        from standard_quant_tools.agent.models import DataCapabilitiesInput

        described = DataCapabilitiesInput.model_fields["source"].description
        assert "databento" in described


class TestTheCacheIsCountedAndNotTouched:
    def _populate(self, root):
        root.mkdir(parents=True)
        for name in (
            "v2_yfinance_AAPL_a_b_1d.parquet",
            "v2_polygon_MSFT_a_b_1d.parquet",
            "v3_yfinance_AAPL_a_b_1d.parquet",
        ):
            (root / name).write_bytes(b"x" * 10)
        return root

    def test_dead_files_are_counted_and_left_where_they_are(
        self, tmp_path, monkeypatch
    ):
        """
        COUNTED, NEVER DELETED. `sqt cache gc` owns the deletion; a tool
        asked to describe something must not change it, and an agent that
        called a describe tool and lost cached data would have no way to
        know which call did it.
        """
        root = self._populate(tmp_path / "ohlcv")
        monkeypatch.setattr(cache_module, "_CACHE_ROOT", root)
        result = dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert result["cache_files"] == 3
        assert result["cache_dead_files"] == 2
        assert result["cache_bytes"] == 30
        assert result["cache_generations"] == ["v2", "v3"]
        assert any("sqt cache gc" in note for note in result["notes"])
        assert len(list(root.iterdir())) == 3

    def test_a_cold_cache_is_zeros_rather_than_a_refusal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "never_written")
        result = dispatch("describe_data_capabilities", {"source": "yfinance"})
        assert result["cache_files"] == 0
        assert result["cache_bytes"] == 0
        assert result["cache_dead_files"] == 0
        assert result["cache_generations"] == []


class TestTheProvidersOwnProse:
    def test_dataset_metadata_carries_the_notes(self):
        """The four booleans have no slot for "this feed is a sample", so
        a provider that names its own sampling problem could say it and
        have it dropped one layer above."""
        with _serving(_Stamped()):
            result = _data("get_dataset_metadata", {"symbol": "AAPL"})
        assert result["notes"] == _Stamped.NOTES

    def test_a_provider_with_nothing_to_add_reports_an_empty_list(self):
        with _serving(_Bars()):
            result = _data("get_dataset_metadata", {"symbol": "AAPL"})
        assert result["notes"] == []


class TestTheTickFeedsStampTheirTapeToo:
    """
    Only the bars path recorded which feed answered. The tick, quote, book
    and order-event feeds are SINGLE-VENUE tapes, so a volume there is one
    venue's share rather than the market's -- which makes "which venue" a
    harder question on those than on bars, and it was the one path that
    said nothing.
    """

    _WINDOW = pd.date_range("2026-03-02 14:30", periods=4, freq="s", tz="UTC")

    def _answered_with(self, monkeypatch, frame, dataset):
        from standard_quant_tools.data.databento_provider import DatabentoProvider

        monkeypatch.setattr(
            DatabentoProvider,
            "_fetch",
            lambda self, *a, **k: (frame.copy(), dataset),
        )
        return DatabentoProvider(api_key="not-used", client=object())

    def _assert_stamped(self, frame, dataset):
        assert frame.attrs["dataset"] == dataset
        assert frame.attrs["provider"] == "databento"
        assert frame.attrs["adjusted"] is False

    def test_a_trade_tape_names_its_venue(self, monkeypatch):
        raw = pd.DataFrame(
            {"price": [250.01, 250.02, 250.0, 249.99], "size": [100, 200, 150, 50]},
            index=self._WINDOW,
        )
        provider = self._answered_with(monkeypatch, raw, "XNAS.ITCH")
        tape = provider.get_trades("NVDA", "2026-03-02", "2026-03-02")
        self._assert_stamped(tape, "XNAS.ITCH")

    def test_a_quote_panel_names_its_venue(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "bid_px_00": [250.0] * 4,
                "ask_px_00": [250.02] * 4,
                "bid_sz_00": [100] * 4,
                "ask_sz_00": [120] * 4,
            },
            index=self._WINDOW,
        )
        provider = self._answered_with(monkeypatch, raw, "XNAS.BASIC")
        quotes = provider.get_quotes("NVDA", "2026-03-02", "2026-03-02")
        self._assert_stamped(quotes, "XNAS.BASIC")

    def test_a_book_names_its_venue(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "bid_px_00": [250.0] * 4,
                "ask_px_00": [250.02] * 4,
                "bid_sz_00": [100] * 4,
                "ask_sz_00": [120] * 4,
                "bid_px_01": [249.99] * 4,
                "ask_px_01": [250.03] * 4,
                "bid_sz_01": [200] * 4,
                "ask_sz_01": [210] * 4,
            },
            index=self._WINDOW,
        )
        provider = self._answered_with(monkeypatch, raw, "XNAS.ITCH")
        book = provider.get_order_book("NVDA", "2026-03-02", "2026-03-02", levels=2)
        self._assert_stamped(book, "XNAS.ITCH")

    def test_an_order_event_stream_names_its_venue(self, monkeypatch):
        raw = pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4],
                "action": ["A", "C", "T", "A"],
                "side": ["B", "A", "B", "A"],
                "price": [250.0, 250.02, 250.01, 250.03],
                "size": [100, 120, 50, 80],
            },
            index=self._WINDOW,
        )
        provider = self._answered_with(monkeypatch, raw, "XNAS.ITCH")
        events = provider.get_order_events("NVDA", "2026-03-02", "2026-03-02")
        self._assert_stamped(events, "XNAS.ITCH")


# ── 3. the quality report's own parameters ───────────────────────────────


def _quality(**overrides):
    arguments = {
        "symbol": "AAPL",
        "start_date": "2024-01-02",
        "end_date": "2024-12-31",
    }
    arguments.update(overrides)
    return dispatch("get_data_quality_report", arguments)


class TestWhichCalendarDecidesWhatIsMissing:
    def test_two_exchanges_disagree_about_the_same_frame(self):
        """
        A frame of US equity sessions has no gaps against the US calendar
        and several against Tokyo's, because the two exchanges do not
        trade on the same days. Whichever one is in force is the answer,
        so it has to be the caller's to choose -- and every entry says
        `calendar` so the verdict is not the weekday fallback in disguise.
        """
        with _serving(_Bars()):
            home = _quality(calendar="XNYS")
            away = _quality(calendar="XTKS")
        assert home["missing_bars"] == []
        assert len(away["missing_bars"]) > 0
        assert {m["basis"] for m in away["missing_bars"]} == {"calendar"}

    def test_a_span_with_no_holiday_in_it_is_clean_under_both(self):
        """The null case. A detector that fires on a fortnight containing
        no holiday is measuring something other than holidays."""
        quiet = _SESSIONS[(_SESSIONS >= "2024-03-04") & (_SESSIONS <= "2024-03-15")]

        class _Quiet(_Bars):
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                return _bars(quiet)

        with _serving(_Quiet()):
            assert _quality(calendar="XNYS")["missing_bars"] == []
            assert _quality(calendar="XTKS")["missing_bars"] == []


class TestWhatCountsAsAThinBar:
    def test_a_planted_thin_bar_is_found_at_the_threshold_that_looks(self):
        """
        The default is severe on purpose -- a feed that is thin all the way
        through is thin CONSISTENTLY and nothing flags it -- so the
        question "is this bar thin against its own recent past" needs the
        knobs. Same frame, same planted bar, two answers.
        """
        frame = _bars()
        frame.iloc[60, frame.columns.get_loc("Volume")] = 200_000.0

        class _Thin(_Bars):
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                return frame

        planted = str(frame.index[60].date())
        with _serving(_Thin()):
            loose = _quality(thin_fraction=0.5, volume_window=5)
            strict = _quality()
        assert planted in {a["date"] for a in loose["volume_anomalies"]}
        assert all(a["kind"] == "thin" for a in loose["volume_anomalies"])
        assert strict["volume_anomalies"] == []


class TestWhichProviderIsBeingChecked:
    def test_the_named_provider_is_the_one_that_is_fetched(self):
        seen = {}

        def _factory(source=None, *a, **k):
            seen["source"] = source
            return _Bars()

        with patch.object(DataFactory, "get_provider", _factory):
            _quality(source="polygon")
        assert seen["source"] == "polygon"

    def test_an_unknown_provider_is_refused_by_name(self):
        """Hard-wiring one provider meant the feeds most worth checking
        could never be checked; taking the argument means an unknown one
        has to refuse with the list rather than raise from the factory."""
        with pytest.raises(ValidationError) as exc:
            _quality(source="nosuch")
        message = str(exc.value)
        for provider in ("yfinance", "polygon", "bloomberg", "databento"):
            assert provider in message


# ── 4. a bundle label is checked against what it labels ──────────────────


class TestALabelIsNotAComment:
    def _panel(self):
        return _fetch(run_id="bundle", name="panel")["ref"]

    def test_a_price_panel_labelled_fundamentals_is_refused(self, runs):
        """
        The label chooses the temporal contract the bundle is validated
        under, so accepting a wrong one buys a confident point-in-time
        verdict about data the bundle does not hold. The refusal names
        both kinds because knowing only that something was wrong does not
        say which of the two to change.
        """
        with _serving(_Bars()):
            ref = self._panel()
        with pytest.raises(ValidationError) as exc:
            _data(
                "build_data_bundle",
                {
                    "frames": [
                        {
                            "frame_kind": "fundamentals",
                            "ref": ref,
                            "source": "stub",
                        }
                    ],
                    "run_id": "bundle",
                    "name": "mislabelled",
                },
            )
        message = str(exc.value)
        assert "price_panel" in message and "fundamentals" in message
        assert "bars" in message

    def test_the_same_panel_labelled_bars_builds(self, runs):
        with _serving(_Bars()):
            ref = self._panel()
            result = _data(
                "build_data_bundle",
                {
                    "frames": [{"frame_kind": "bars", "ref": ref, "source": "stub"}],
                    "run_id": "bundle",
                    "name": "correct",
                },
            )
        assert result["kinds"] == ["bars"]
        assert result["n_frames"] == 1

    def test_a_raw_artifact_path_builds_but_says_it_was_not_checked(self, runs):
        """
        A raw path carries no kind -- that is the documented trade-off of
        accepting one at all -- so there is nothing to check the label
        against. Refusing it would break the older tools that return bare
        paths; saying nothing would let the check LOOK like it ran.
        """
        from standard_quant_tools.backtest.artifacts import save_artifact

        path = save_artifact(_bars(), "bundle", "raw_bars")
        result = _data(
            "build_data_bundle",
            {
                "frames": [{"frame_kind": "bars", "ref": str(path), "source": "stub"}],
                "run_id": "bundle",
                "name": "unchecked",
            },
        )
        assert result["n_frames"] == 1
        assert any("TAKEN ON TRUST" in w for w in result["warnings"])

    def test_every_reference_kind_has_a_ruling(self):
        """The drift guard. A kind added to the interconnect with no entry
        here would pass the check by not being in the table, which is the
        silence this whole change is about."""
        assert set(_BUNDLE_KINDS) == set(KINDS)


# ── 5. two sources, the same question ────────────────────────────────────


def _ratios(**overrides):
    values = {
        "forward_pe": 21.4,
        "trailing_pe": 27.9,
        "price_to_book": 8.1,
        "debt_to_equity": 1.47,
        "return_on_equity": 0.152,
        "profit_margins": 0.248,
        "dividend_yield": 0.0052,
        "market_cap": 3_100_000_000_000,
    }
    values.update(overrides)
    return FinancialRatios(**values)


class _Ratios(_Bars):
    def __init__(self, ratios) -> None:
        self._ratios = ratios

    def get_financial_ratios(self, symbol):
        return self._ratios


def _fetched_ratios(ratios):
    with _serving(_Ratios(ratios)):
        return _data("fetch_financial_ratios", {"symbol": "AAPL"})


def _compare(left, right, **overrides):
    arguments = {"left": left, "right": right}
    arguments.update(overrides)
    return _data("compare_ratio_frames", arguments)


class TestTheSiblingToolsOutputComparesWithItself:
    def test_identical_inputs_disagree_about_nothing(self):
        """
        THE COMPOSITION THIS EXISTS FOR, and the one that was broken:
        fetch ratios from two providers and compare them. The fetch tool
        returns ONE company's flat field map and the comparison reads a
        map of ticker -> ratios, so every field name was read as a
        company with no ratios on it -- and `no_overlap` was counted as a
        disagreement. Two identical payloads reported eight of eight
        fields in conflict.
        """
        payload = _fetched_ratios(_ratios())
        result = _compare(payload, payload, left_name="a", right_name="b")
        assert result["n_compared"] == 8
        assert result["n_disagreeing"] == 0
        assert result["n_no_overlap"] == 0
        assert {f["classification"] for f in result["fields"]} == {"agree"}
        assert all(f["n_compared"] == 1 for f in result["fields"])

    def test_the_two_values_are_reported_for_a_single_company(self):
        """With one company there is nothing to sample from, so the pair
        of numbers IS the comparison and the result carries it."""
        left = _fetched_ratios(_ratios())
        right = _fetched_ratios(_ratios(forward_pe=23.9))
        result = _compare(left, right)
        row = next(f for f in result["fields"] if f["field_name"] == "forward_pe")
        assert row["left"] == pytest.approx(21.4)
        assert row["right"] == pytest.approx(23.9)
        assert row["relative_difference"] == pytest.approx(
            (23.9 - 21.4) / 23.9, rel=1e-6
        )

    def test_one_company_cannot_prove_a_unit_error_and_says_so(self):
        """A constant ratio is what distinguishes a unit conversion from a
        definition difference, and a single pair has nothing to be
        constant across. The verdict is honest about which it is."""
        left = _fetched_ratios(_ratios())
        right = _fetched_ratios(_ratios(dividend_yield=0.52))
        result = _compare(left, right)
        row = next(f for f in result["fields"] if f["field_name"] == "dividend_yield")
        assert row["classification"] == "definition"
        assert any("ONE entity" in w for w in result["warnings"])


class TestTheThreeCasesReachTheResult:
    def test_a_hundredfold_unit_error_is_scale_and_names_the_factor(self):
        """
        `ratio` IS the conversion, and it was computed and discarded: the
        difference between "these disagree" and "multiply by 100" was
        already known and never crossed.
        """
        left = {
            "AAPL": {"dividend_yield": 0.0052},
            "MSFT": {"dividend_yield": 0.0074},
            "XOM": {"dividend_yield": 0.0331},
        }
        right = {
            entity: {"dividend_yield": values["dividend_yield"] * 100}
            for entity, values in left.items()
        }
        result = _compare(left, right)
        row = next(f for f in result["fields"] if f["field_name"] == "dividend_yield")
        assert row["classification"] == "scale"
        assert row["ratio"] == pytest.approx(100.0)
        assert row["ratio_spread"] == pytest.approx(0.0, abs=1e-9)
        assert row["n_compared"] == 3
        assert result["n_disagreeing"] == 1
        assert row["left"] is None and row["right"] is None

    def test_a_wandering_ratio_reports_the_spread_that_proves_it(self):
        left = {
            "AAPL": {"debt_to_equity": 1.47},
            "MSFT": {"debt_to_equity": 0.42},
            "XOM": {"debt_to_equity": 0.21},
        }
        right = {
            "AAPL": {"debt_to_equity": 2.98},
            "MSFT": {"debt_to_equity": 0.51},
            "XOM": {"debt_to_equity": 0.77},
        }
        result = _compare(left, right)
        row = next(f for f in result["fields"] if f["field_name"] == "debt_to_equity")
        assert row["classification"] == "definition"
        assert row["ratio_spread"] > 0.05
        assert row["ratio"] is None


class TestSilenceIsNotAgreement:
    def test_no_field_in_common_is_counted_separately(self):
        """
        Eight fields, nothing checked, and the old answer was "eight
        disagreements" -- which is the same number an agent would get from
        two genuinely incompatible providers. The count moves to its own
        field and the result says plainly that nothing was compared.
        """
        result = _compare({"forward_pe": 21.4}, {"market_cap": 3_100_000_000_000})
        assert result["n_compared"] == 8
        assert result["n_no_overlap"] == 8
        assert result["n_disagreeing"] == 0
        assert {f["classification"] for f in result["fields"]} == {"no_overlap"}
        assert any("NOTHING WAS COMPARED" in w for w in result["warnings"])

    def test_mismatched_nesting_is_refused_by_name(self):
        """One company against a universe cannot be reconciled without
        guessing which name the company is, and a wrong guess reports
        every field as a divergence -- which is how this was found."""
        with pytest.raises(ValidationError) as exc:
            _compare(
                {"forward_pe": 21.4},
                {"AAPL": {"forward_pe": 21.4}, "MSFT": {"forward_pe": 33.0}},
            )
        message = str(exc.value)
        assert "left" in message and "right" in message
        assert "ticker" in message

    def test_two_different_companies_are_refused(self):
        """The tool asks whether two SOURCES disagree about one company.
        Two companies disagree for reasons that have nothing to do with
        the providers."""
        with pytest.raises(ValidationError) as exc:
            _compare(
                _fetched_ratios(_ratios()),
                {"symbol": "MSFT", "ratios": _ratios().model_dump()},
            )
        assert "AAPL" in str(exc.value) and "MSFT" in str(exc.value)
