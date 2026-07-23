"""Tests for backtest/artifacts.py — local Parquet artifact store."""

import pandas as pd
import pytest

from standard_quant_tools.backtest.artifacts import save_artifact, load_artifact
from standard_quant_tools.error import ValidationError


@pytest.fixture(autouse=True)
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


class TestSaveArtifact:
    def test_empty_series_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            save_artifact(pd.Series(dtype=float), run_id="run1", name="equity_curve")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            save_artifact(pd.DataFrame(), run_id="run1", name="trades")

    def test_returns_path_string(self):
        series = pd.Series([1.0, 2.0, 3.0])
        uri = save_artifact(series, run_id="run1", name="equity_curve")
        assert isinstance(uri, str)
        assert uri.endswith("equity_curve.parquet")


class TestRoundTrip:
    def test_series_round_trip(self):
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        series = pd.Series([100.0, 101.0, 102.5, 103.0, 99.5], index=dates, name="equity")
        uri = save_artifact(series, run_id="run1", name="equity_curve")
        loaded = load_artifact(uri).squeeze("columns")
        # Parquet doesn't preserve a DatetimeIndex's `freq` attribute — a
        # round-trip storage-format quirk, not a data-loss bug.
        pd.testing.assert_series_equal(loaded, series, check_names=False, check_freq=False)

    def test_unnamed_series_defaults_to_value_column(self):
        series = pd.Series([1.0, 2.0, 3.0])
        uri = save_artifact(series, run_id="run1", name="equity_curve")
        loaded = load_artifact(uri)
        assert list(loaded.columns) == ["value"]

    def test_dataframe_round_trip(self):
        df = pd.DataFrame({
            "entry_date": ["2023-01-02", "2023-01-05"],
            "exit_date": ["2023-01-04", "2023-01-10"],
            "return_pct": [1.5, -2.0],
        })
        uri = save_artifact(df, run_id="run1", name="trades")
        loaded = load_artifact(uri)
        pd.testing.assert_frame_equal(loaded, df)

    def test_different_names_produce_different_files(self):
        series = pd.Series([1.0, 2.0])
        uri1 = save_artifact(series, run_id="run1", name="equity_curve")
        uri2 = save_artifact(series, run_id="run1", name="trades")
        assert uri1 != uri2


class TestLoadArtifact:
    def test_missing_uri_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="not found"):
            load_artifact(str(tmp_path / "nonexistent.parquet"))
