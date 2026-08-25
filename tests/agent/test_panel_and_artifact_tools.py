"""
get_technical_panel, describe_artifact and get_drawdown_table.

Two contracts run through all three. The first is that the panel path is
the SAME arithmetic as the per-ticker path -- the tool exists to save round
trips, not to trade accuracy for them, so the test compares it against
indicators.rsi() called directly rather than against a recorded number.

The second is that reading a persisted run must beat re-running it. These
tests build an equity curve with drawdowns whose depths are known by
construction, persist it, and assert the tool reports those episodes from
the artifact -- including the unrecovered final one, whose duration is a
floor rather than a measurement and must not be reported as if it were
complete.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import rsi


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _ohlcv(seed: int, n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(rng.normal(0.0005, 0.012, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2023-01-02", periods=n),
    )


@pytest.fixture
def universe(monkeypatch):
    panel = {"AAA": _ohlcv(1), "BBB": _ohlcv(2), "CCC": _ohlcv(3)}
    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.research.tools.fetch_ohlcv_panel_sync",
        lambda tickers, start, end, interval="1d": {t: panel[t] for t in tickers},
    )
    return panel


@pytest.fixture
def equity_curve_uri():
    """A curve whose drawdowns are known by construction: a -20.8% episode
    that recovers, then a -15.4% one that never does."""
    legs = [
        np.linspace(10_000, 12_000, 100),
        np.linspace(12_000, 9_500, 60),
        np.linspace(9_500, 13_000, 80),
        np.linspace(13_000, 11_000, 60),
    ]
    equity = pd.Series(
        np.concatenate(legs),
        index=pd.bdate_range("2022-01-03", periods=300),
        name="equity",
    )
    return save_artifact(equity, "testrun", "equity_curve", overwrite=True)


class TestTechnicalPanel:
    def test_panel_matches_the_per_ticker_function(self, universe):
        """The whole justification for the panel path: identical
        arithmetic, fewer round trips. A tolerance here would be hiding a
        second implementation."""
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA", "BBB", "CCC"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["rsi"],
            },
        )
        for ticker, frame in universe.items():
            expected = float(rsi(frame["Close"], period=14).iloc[-1])
            assert result["latest"][ticker]["RSI"] == pytest.approx(expected, abs=1e-6)

    def test_multi_field_indicators_keep_the_per_ticker_field_names(self, universe):
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA", "BBB"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["bollinger_bands"],
            },
        )
        assert set(result["latest"]["AAA"]) == {"BB_Upper", "BB_Middle", "BB_Lower"}

    def test_a_young_ticker_truncates_the_whole_panel_and_says_so(
        self, monkeypatch, universe
    ):
        """The trap this tool has to not walk into silently. The panel is
        computed on the bars every ticker SHARES, so one recent listing
        collapses the window for all of them and every indicator comes back
        NaN. Naming every ticker as incomplete would be true and useless --
        the result has to say the calendar was the cause."""
        # Overlapping, but only just: DDD's history covers the last eight
        # bars of everyone else's, so there IS an intersection and it is
        # far too short for a 14-period RSI.
        young = _ohlcv(4, n=8)
        young.index = universe["AAA"].index[-8:]
        short = {**universe, "DDD": young}
        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.research.tools.fetch_ohlcv_panel_sync",
            lambda tickers, start, end, interval="1d": {t: short[t] for t in tickers},
        )
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA", "DDD"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["rsi"],
            },
        )
        assert result["calendar_limited_by"] == ["DDD"]
        assert result["n_bars"] == 8
        assert any("shorter histories" in note for note in result["notes"])
        assert any("warmed up" in note for note in result["notes"])

    def test_no_overlap_at_all_is_rejected_outright(self, monkeypatch, universe):
        """Not the same case: with no shared bars there is no panel to
        compute, and the library refuses rather than returning an empty
        one."""
        disjoint = _ohlcv(5, n=20)
        disjoint.index = pd.bdate_range("2025-01-02", periods=20)
        short = {**universe, "ZZZ": disjoint}
        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.research.tools.fetch_ohlcv_panel_sync",
            lambda tickers, start, end, interval="1d": {t: short[t] for t in tickers},
        )
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "get_technical_panel",
                {
                    "tickers": ["AAA", "ZZZ"],
                    "start_date": "2023-01-01",
                    "end_date": "2025-07-01",
                    "indicators": ["rsi"],
                },
            )
        assert "common bars" in str(exc.value)

    def test_a_full_history_universe_reports_no_calendar_limit(self, universe):
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA", "BBB", "CCC"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["rsi"],
            },
        )
        assert result["calendar_limited_by"] == []
        assert result["notes"] == []
        assert result["incomplete_tickers"] == []

    def test_a_missing_ticker_is_an_error_not_a_smaller_universe(self, monkeypatch):
        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.research.tools.fetch_ohlcv_panel_sync",
            lambda tickers, start, end, interval="1d": {"AAA": _ohlcv(1)},
        )
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "get_technical_panel",
                {
                    "tickers": ["AAA", "GONE"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-07-01",
                    "indicators": ["rsi"],
                },
            )
        assert "GONE" in str(exc.value)

    def test_duplicate_tickers_are_rejected(self, universe):
        with pytest.raises(Exception) as exc:
            dispatch(
                "get_technical_panel",
                {
                    "tickers": ["AAA", "AAA"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-07-01",
                    "indicators": ["rsi"],
                },
            )
        assert "duplicate" in str(exc.value).lower()

    def test_persisting_round_trips_a_multiindex_panel(self, universe):
        """Parquet has no MultiIndex column concept, so the flattening has
        to be lossless -- otherwise the artifact silently loses which field
        belonged to which ticker."""
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA", "BBB"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["bollinger_bands"],
                "persist_run_id": "panelrun",
            },
        )
        uri = result["artifact_uris"]["bollinger_bands"]
        described = dispatch("describe_artifact", {"uri": uri, "preview_rows": 1})
        assert "AAA::BB_Upper" in described["columns"]
        assert described["rows"] == result["n_bars"]

    def test_no_artifacts_without_a_run_id(self, universe):
        result = dispatch(
            "get_technical_panel",
            {
                "tickers": ["AAA"],
                "start_date": "2023-01-01",
                "end_date": "2023-07-01",
                "indicators": ["rsi"],
            },
        )
        assert result["artifact_uris"] == {}


class TestDescribeArtifact:
    def test_reports_shape_span_and_hash(self, equity_curve_uri):
        result = dispatch("describe_artifact", {"uri": equity_curve_uri})
        assert result["rows"] == 300
        assert result["columns"] == ["equity"]
        assert result["index_start"].startswith("2022-01-03")
        assert len(result["content_hash"]) == 64

    def test_the_hash_is_over_the_file_not_the_uri(self, equity_curve_uri):
        """Two reads of one artifact must agree, and a different artifact
        must not collide -- that is the whole point of reporting it."""
        first = dispatch("describe_artifact", {"uri": equity_curve_uri})
        second = dispatch("describe_artifact", {"uri": equity_curve_uri})
        assert first["content_hash"] == second["content_hash"]

        other = save_artifact(
            pd.Series([1.0, 2.0, 3.0], name="equity"), "testrun", "other"
        )
        assert dispatch("describe_artifact", {"uri": other})["content_hash"] != (
            first["content_hash"]
        )

    def test_the_middle_is_never_returned(self, equity_curve_uri):
        result = dispatch(
            "describe_artifact", {"uri": equity_curve_uri, "preview_rows": 3}
        )
        assert len(result["head"]) == 3
        assert len(result["tail"]) == 3
        assert result["head"][0] != result["tail"][0]

    def test_preview_rows_zero_returns_only_the_summary(self, equity_curve_uri):
        result = dispatch(
            "describe_artifact", {"uri": equity_curve_uri, "preview_rows": 0}
        )
        assert result["head"] == [] and result["tail"] == []
        assert result["column_summary"]["equity"]["max"] == pytest.approx(13_000.0)

    def test_timestamps_survive_as_strings(self, equity_curve_uri):
        """Parquet round trips Timestamps, which json.dumps cannot encode.
        A preview that crashed the serializer would be worse than no
        preview."""
        import json

        result = dispatch("describe_artifact", {"uri": equity_curve_uri})
        json.dumps(result, allow_nan=False)

    def test_a_path_outside_the_runs_dir_is_refused(self, tmp_path):
        outside = tmp_path / "elsewhere.parquet"
        pd.DataFrame({"a": [1]}).to_parquet(outside)
        with pytest.raises(ValidationError):
            dispatch("describe_artifact", {"uri": str(outside)})


class TestDrawdownTable:
    def test_finds_the_episodes_the_curve_was_built_with(self, equity_curve_uri):
        result = dispatch("get_drawdown_table", {"equity_curve_uri": equity_curve_uri})
        assert result["n_episodes_total"] == 2
        depths = [e["depth"] for e in result["episodes"]]
        assert depths[0] == pytest.approx(-0.208333, abs=1e-5)
        assert depths[1] == pytest.approx(-0.153846, abs=1e-5)

    def test_episodes_come_back_deepest_first(self, equity_curve_uri):
        result = dispatch("get_drawdown_table", {"equity_curve_uri": equity_curve_uri})
        depths = [e["depth"] for e in result["episodes"]]
        assert depths == sorted(depths)

    def test_an_unrecovered_episode_is_flagged_not_completed(self, equity_curve_uri):
        """Its duration is a floor, not a measurement. Reporting a recovery
        that has not happened would overstate how quickly the strategy
        came back."""
        result = dispatch("get_drawdown_table", {"equity_curve_uri": equity_curve_uri})
        assert result["currently_underwater"] is True
        last = result["episodes"][-1]
        assert last["end"] is None and last["recovery_bars"] is None

    def test_max_drawdown_agrees_with_the_deepest_episode(self, equity_curve_uri):
        result = dispatch("get_drawdown_table", {"equity_curve_uri": equity_curve_uri})
        assert result["max_drawdown"] == pytest.approx(
            min(e["depth"] for e in result["episodes"]), abs=1e-6
        )

    def test_min_depth_filters_without_hiding_the_count(self, equity_curve_uri):
        """The cap must never be silent: n_episodes_total still reports
        what was there before filtering."""
        result = dispatch(
            "get_drawdown_table",
            {"equity_curve_uri": equity_curve_uri, "min_depth": 0.18},
        )
        assert result["n_episodes_returned"] == 1
        assert result["n_episodes_total"] == 2

    def test_a_multi_column_artifact_is_rejected(self):
        uri = save_artifact(
            pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), "testrun", "twocol"
        )
        with pytest.raises(ValidationError) as exc:
            dispatch("get_drawdown_table", {"equity_curve_uri": uri})
        assert "single series" in str(exc.value)

    def test_time_underwater_is_a_fraction_of_bars(self, equity_curve_uri):
        result = dispatch("get_drawdown_table", {"equity_curve_uri": equity_curve_uri})
        assert 0.0 < result["time_underwater_pct"] < 1.0
