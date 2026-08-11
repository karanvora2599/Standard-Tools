"""Tests for standard_quant_tools._compat — optional Polars interop helpers."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools._compat import (
    HAS_POLARS,
    is_dataframe_like,
    is_empty,
    is_series_like,
    require_polars,
    to_clean_numpy,
)
from standard_quant_tools.error import ValidationError

if HAS_POLARS:
    import polars as pl


class TestIsSeriesLike:
    def test_pandas_series_is_series_like(self):
        assert is_series_like(pd.Series([1, 2, 3])) is True

    def test_plain_list_is_not_series_like(self):
        assert is_series_like([1, 2, 3]) is False

    def test_pandas_dataframe_is_not_series_like(self):
        assert is_series_like(pd.DataFrame({"a": [1]})) is False

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_polars_series_is_series_like(self):
        assert is_series_like(pl.Series([1, 2, 3])) is True


class TestIsDataframeLike:
    def test_pandas_dataframe_is_dataframe_like(self):
        assert is_dataframe_like(pd.DataFrame({"a": [1]})) is True

    def test_pandas_series_is_not_dataframe_like(self):
        assert is_dataframe_like(pd.Series([1, 2, 3])) is False

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_polars_dataframe_is_dataframe_like(self):
        assert is_dataframe_like(pl.DataFrame({"a": [1]})) is True


class TestIsEmpty:
    def test_empty_pandas_series(self):
        assert is_empty(pd.Series([], dtype=float)) is True

    def test_nonempty_pandas_series(self):
        assert is_empty(pd.Series([1.0])) is False

    def test_empty_pandas_dataframe(self):
        assert is_empty(pd.DataFrame()) is True

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_empty_polars_series(self):
        assert is_empty(pl.Series([], dtype=pl.Float64)) is True

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_nonempty_polars_series(self):
        assert is_empty(pl.Series([1.0])) is False


class TestRequirePolars:
    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars installed")
    def test_no_raise_when_installed(self):
        require_polars("some feature")  # must not raise

    def test_raises_actionable_error_when_missing(self, monkeypatch):
        monkeypatch.setattr("standard_quant_tools._compat.HAS_POLARS", False)
        with pytest.raises(ValidationError, match="requires polars"):
            require_polars("some feature")


class TestToCleanNumpy:
    def test_pandas_drops_nan_and_none(self):
        s = pd.Series([1.0, np.nan, 3.0, None])
        result = to_clean_numpy(s)
        assert list(result) == [1.0, 3.0]

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_polars_drops_both_null_and_nan(self):
        """Regression coverage: polars treats null and NaN as distinct
        concepts (unlike pandas' single .dropna()) — .drop_nulls() alone
        would leave NaN values behind. to_clean_numpy must drop both."""
        s = pl.Series([1.0, None, 3.0, np.nan])
        result = to_clean_numpy(s)
        assert list(result) == [1.0, 3.0]

    @pytest.mark.skipif(not HAS_POLARS, reason="requires polars")
    def test_pandas_and_polars_agree_on_identical_data(self):
        data = [1.0, np.nan, 2.5, None, -3.0]
        pd_result = to_clean_numpy(pd.Series(data))
        pl_result = to_clean_numpy(pl.Series(data))
        assert list(pd_result) == list(pl_result)

    def test_returns_requested_dtype(self):
        s = pd.Series([1, 2, 3])
        result = to_clean_numpy(s, dtype=float)
        assert result.dtype == np.float64
