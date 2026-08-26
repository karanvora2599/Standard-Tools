"""
The `data` runtime: fetch once, publish a reference, read it from anywhere.

WHAT THIS RUNTIME IS BUYING, and therefore what these tests pin. Before it,
a panel built inside one tool died inside that tool: two agents asking about
the same universe fetched it twice, and an intermediate frame could not be
handed to the next runtime without being recomputed. The tools here return
an `sqt://` reference instead of the rows, so the interesting property is
not "does the fetch work" -- it is that the SECOND reader needs no provider
at all.

That is the assertion the whole increment rests on, and it is the one below
that patches the provider away before resolving.

THE OTHER HALF is what the data cannot support. A universe fetch drops
tickers that returned nothing, a capped tape is truncated rather than short,
and a bundle validated without `require_pit` says nothing about whether a
leakage-free join is possible. Each of those is a silence that used to be
invisible, and each is asserted here as a warning STRING rather than left to
prose, because a warning nobody checks is a warning that quietly stops
firing.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.agent.runtimes.handoff import KINDS
from standard_quant_tools.agent.runtimes.handoff import resolve as resolve_ref
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError

_INDEX = pd.bdate_range("2023-01-02", periods=260)


def _bars(symbol: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % 1_000)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(_INDEX))))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=_INDEX,
    )


class _Provider(DataProvider):
    """
    A provider that serves bars and nothing else -- the normal case.

    SUBCLASSES DataProvider deliberately. A duck-typed stand-in would
    simply lack `get_trades`, and the tool would fail with an
    AttributeError that proves nothing; the real base raises
    NotImplementedError with a message, which is the behaviour the
    refusal tests below are actually about.
    """

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        return _bars(symbol)

    async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
        return _bars(symbol)

    def get_ticker_info(self, symbol):
        raise NotImplementedError

    def get_financial_ratios(self, symbol):
        raise NotImplementedError

    def get_metadata(self, symbol, interval="1d"):
        raise NotImplementedError


@pytest.fixture
def bars_provider():
    with patch.object(DataFactory, "get_provider", lambda *a, **k: _Provider()):
        yield


def _data(tool, arguments):
    return resolve_runtime("data").dispatch(tool, arguments)


class TestAReferenceOutlivesTheFetch:
    def test_a_published_panel_resolves_without_any_provider(
        self, bars_provider, tmp_path, monkeypatch
    ):
        """
        THE TEST THIS RUNTIME EXISTS FOR. The fetch happens once, under a
        provider; the read happens with the provider patched away entirely.
        If this ever needs the provider back, references have stopped being
        references and every consumer is refetching.
        """
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        result = _data(
            "fetch_ohlcv_panel",
            {
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2023-01-02",
                "end_date": "2023-12-29",
                "run_id": "t_ref",
                "name": "bars",
            },
        )
        ref = result["ref"]

        # No provider at all from here on.
        def _no_provider(*_a, **_k):
            raise AssertionError("resolving a reference must not reach a data provider")

        with patch.object(DataFactory, "get_provider", _no_provider):
            frame = resolve_ref(ref)

        assert len(frame) == result["rows"] > 0
        assert set(result["entities"]) == {"AAPL", "MSFT"}

    def test_the_panel_shape_matches_what_the_consumer_needs(
        self, bars_provider, tmp_path, monkeypatch
    ):
        """Long-stacked bars carry `entity`; a returns panel is wide. The
        two are not interchangeable, and fetching the wrong one means the
        consumer rebuilds it."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        args = {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2023-01-02",
            "end_date": "2023-12-29",
            "run_id": "t_shape",
        }
        stacked = _data("fetch_ohlcv_panel", {**args, "name": "bars"})
        wide = _data("fetch_returns_panel", {**args, "name": "rets"})

        assert "entity" in stacked["columns"]
        assert stacked["kind"] == "price_panel"
        assert set(wide["columns"]) == {"AAPL", "MSFT"}
        assert wide["kind"] == "returns_panel"


class TestTheNewKindsAreDeclared:
    @pytest.mark.parametrize("kind", ["tick_tape", "quote_panel", "data_bundle"])
    def test_the_kind_exists_and_says_what_it_holds(self, kind):
        assert kind in KINDS, f"{kind} is not a declared reference kind"
        assert KINDS[kind]["description"].strip()


class TestWhatTheDataCannotSupportIsReturned:
    def test_a_ticker_that_returned_EMPTY_is_dropped_and_named(
        self, tmp_path, monkeypatch
    ):
        """
        A dropped ticker is ABSENT from the panel, not NaN. Downstream a
        complete-case join cannot tell an excluded name from one that was
        never requested, so the fetch has to say which happened.

        This is the reachable half: the fetch SUCCEEDED and returned
        nothing, which is a different event from the fetch raising.
        """
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))

        class _Empty(_Provider):
            async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
                return pd.DataFrame() if symbol == "GONE" else _bars(symbol)

        with patch.object(DataFactory, "get_provider", lambda *a, **k: _Empty()):
            result = _data(
                "fetch_ohlcv_panel",
                {
                    "tickers": ["AAPL", "GONE"],
                    "start_date": "2023-01-02",
                    "end_date": "2023-12-29",
                    "run_id": "t_missing",
                    "name": "bars",
                },
            )
        assert "GONE" not in result["entities"]
        assert any("GONE" in w for w in result["warnings"]), (
            "a dropped ticker must be named; silence makes it look like a "
            "universe the caller chose"
        )

    def test_a_ticker_that_RAISES_fails_the_whole_batch_and_says_so(
        self, tmp_path, monkeypatch
    ):
        """
        The other half, and it is a real limitation rather than a choice
        made here: `fetch_ohlcv_panel_async` gathers without
        `return_exceptions`, so one failure propagates and there is no
        partial panel. The refusal has to say that, because the raw error
        names only the symbol that happened to raise first and a caller
        cannot tell from it whether the other names were fine.

        Pinned so that if the helper ever becomes resilient, this test
        fails and the tool's promise gets revisited deliberately rather
        than drifting.
        """
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))

        class _Broken(_Provider):
            async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
                if symbol == "GONE":
                    raise RuntimeError("upstream refused")
                return _bars(symbol)

        with patch.object(DataFactory, "get_provider", lambda *a, **k: _Broken()):
            with pytest.raises(ValidationError) as excinfo:
                _data(
                    "fetch_ohlcv_panel",
                    {
                        "tickers": ["AAPL", "GONE"],
                        "start_date": "2023-01-02",
                        "end_date": "2023-12-29",
                        "run_id": "t_broken",
                        "name": "bars",
                    },
                )
        assert "whole batch fails together" in str(excinfo.value)

    def test_a_bundle_validated_without_require_pit_says_so(
        self, bars_provider, tmp_path, monkeypatch
    ):
        """
        `usable` at the default does NOT mean a leakage-free join is
        possible, and a caller who reads it that way has been misled by the
        field name alone.
        """
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        panel = _data(
            "fetch_ohlcv_panel",
            {
                "tickers": ["AAPL"],
                "start_date": "2023-01-02",
                "end_date": "2023-12-29",
                "run_id": "t_pit",
                "name": "bars",
            },
        )
        bundle = _data(
            "build_data_bundle",
            {
                "frames": [
                    {"frame_kind": "bars", "ref": panel["ref"], "source": "test"}
                ],
                "run_id": "t_pit",
                "name": "inputs",
            },
        )
        verdict = _data("validate_data_bundle", {"ref": bundle["ref"]})
        assert verdict["usable"] is True
        assert any("require_pit" in w for w in verdict["warnings"])

    def test_requiring_pit_actually_blocks(self, bars_provider, tmp_path, monkeypatch):
        """The other direction. If require_pit=True never blocks anything,
        the flag is decorative and the default is not a trade-off."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        panel = _data(
            "fetch_ohlcv_panel",
            {
                "tickers": ["AAPL"],
                "start_date": "2023-01-02",
                "end_date": "2023-12-29",
                "run_id": "t_pit2",
                "name": "bars",
            },
        )
        bundle = _data(
            "build_data_bundle",
            {
                "frames": [
                    {"frame_kind": "bars", "ref": panel["ref"], "source": "test"}
                ],
                "run_id": "t_pit2",
                "name": "inputs",
            },
        )
        strict = _data(
            "validate_data_bundle", {"ref": bundle["ref"], "require_pit": True}
        )
        assert strict["usable"] is False
        assert strict["blocking"], "require_pit must name what blocked it"


class TestAMissingCapabilityIsARefusalNotACrash:
    @pytest.mark.parametrize("tool", ["fetch_tick_tape", "fetch_quote_panel"])
    def test_no_tick_feed_refuses_by_name(self, bars_provider, tool):
        """
        Most environments have no tick feed, so this is the NORMAL path
        rather than an edge. `DataProvider` raises NotImplementedError,
        which the surface contract does not accept -- a caller cannot tell
        it from a bug. It has to arrive as a refusal naming the remedy.
        """
        with pytest.raises(ValidationError) as excinfo:
            _data(
                tool,
                {
                    "symbol": "AAPL",
                    "start_date": "2023-01-02",
                    "end_date": "2023-01-05",
                    "run_id": "t_tick",
                    "name": "tape",
                },
            )
        assert "describe_data_capabilities" in str(excinfo.value)

    def test_a_malformed_reference_refuses_by_name(self):
        """`handoff.resolve` raises from inside its loader for anything that
        is not a well-formed reference, and an AttributeError three frames
        down names no argument and no remedy."""
        with pytest.raises(ValidationError) as excinfo:
            _data("infer_temporal_contract", {"ref": "not-a-reference"})
        assert "not-a-reference" in str(excinfo.value)

    def test_publishing_an_empty_panel_is_refused(self, tmp_path, monkeypatch):
        """An empty panel behind a reference is indistinguishable
        downstream from one whose data has not arrived yet."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))

        class _Empty(_Provider):
            def get_ohlcv(self, symbol, start, end, interval="1d"):
                return pd.DataFrame()

        with patch.object(DataFactory, "get_provider", lambda *a, **k: _Empty()):
            with pytest.raises(ValidationError, match="no rows"):
                _data(
                    "fetch_ohlcv",
                    {
                        "symbol": "AAPL",
                        "start_date": "2023-01-02",
                        "end_date": "2023-12-29",
                        "run_id": "t_empty",
                        "name": "bars",
                    },
                )


class TestTheProviderContractIsReportedHonestly:
    def test_a_non_point_in_time_provider_is_called_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))

        class _Meta(_Provider):
            def get_metadata(self, symbol, interval="1d"):
                from standard_quant_tools.data.metadata import DataSetMetadata

                return DataSetMetadata(
                    provider="test",
                    adjusted=True,
                    survivorship_free=False,
                    point_in_time=False,
                    timezone="UTC",
                    frequency=interval,
                )

        with patch.object(DataFactory, "get_provider", lambda *a, **k: _Meta()):
            result = _data("get_dataset_metadata", {"symbol": "AAPL"})

        assert result["point_in_time"] is False
        joined = " ".join(result["warnings"])
        assert "point-in-time" in joined
        assert "survivorship" in joined
