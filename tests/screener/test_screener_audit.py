"""
Regression tests for the screener half of the portfolio/screener/agent-tools
audit: unestimable beta reported as a real value, and filter VALUES going
unvalidated while only their keys were checked.
"""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.screener.screener import (
    _MIN_BETA_OBS,
    _fetch_ticker_data,
    _validate_filter_values,
)


def _ohlcv(n: int, start: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def _provider(asset_df: pd.DataFrame, spy_df: pd.DataFrame) -> MagicMock:
    provider = MagicMock()

    async def _get(ticker, start, end, *args, **kwargs):
        return spy_df if ticker == "SPY" else asset_df

    provider.get_ohlcv_async = _get
    return provider


class TestBetaEstimability:
    """
    calculate_beta returns alpha/beta/r_squared all 0.0 when fewer than two
    points overlap. That is a sentinel, but an indistinguishable one, because
    0.0 is also a perfectly legitimate beta -- and the screener FILTERED on
    it. A ticker whose history did not overlap the benchmark reported beta
    0.0 and PASSED beta_max=0.5, so "could not be estimated" was read as
    "very low beta": backwards for the defensive screen that bound exists to
    express.
    """

    def _run(self, filters, asset_bars=3, asset_start="2025-06-02"):
        spy = _ohlcv(300, "2023-01-02")
        asset = _ohlcv(asset_bars, asset_start)
        return asyncio.run(
            _fetch_ticker_data(
                _provider(asset, spy), "NEWCO", "2023-01-02", "2025-06-06", filters, spy
            )
        )

    def test_non_overlapping_history_is_an_error_not_a_beta_of_zero(self):
        status, ticker, payload = self._run({"beta_max": 0.5})
        assert status == "error", "previously 'passed' with beta 0.0"
        assert "not estimable" in payload

    def test_error_message_reports_the_actual_overlap(self):
        status, _, payload = self._run({"beta_max": 0.5})
        assert status == "error"
        assert str(_MIN_BETA_OBS) in payload

    def test_a_beta_max_screen_no_longer_admits_an_unestimable_ticker(self):
        """The direction of the failure is the point: beta_max is a ceiling,
        so a spurious 0.0 passes it while beta_min correctly rejected. Only
        one side of the filter was visibly broken."""
        assert self._run({"beta_max": 0.5})[0] == "error"
        assert self._run({"beta_min": 0.8})[0] == "error"

    def test_sufficient_overlap_still_computes_a_beta(self):
        spy = _ohlcv(300, "2023-01-02")
        status, _, payload = asyncio.run(
            _fetch_ticker_data(
                _provider(spy, spy),
                "AAA",
                "2023-01-02",
                "2024-01-01",
                {"beta_max": 5.0},
                spy,
            )
        )
        assert status == "passed"
        assert "beta" in payload


class TestFilterValueValidation:
    """
    Only filter KEYS were validated. Every consequence of a bad VALUE
    surfaced per ticker rather than once -- and the NaN case did not surface
    at all.
    """

    def test_nan_bound_is_rejected_because_it_disables_the_screen(self):
        """
        NaN fails every comparison, so `last_rsi > rsi_max` is False for every
        ticker and an oversold screen silently became a no-op that admitted
        RSI 100. A filter that rejects nothing looks exactly like a filter
        nothing failed.
        """
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_filter_values({"rsi_max": float("nan")})

    def test_nan_rejection_explains_the_direction_of_the_failure(self):
        with pytest.raises(ValidationError, match="every ticker passes"):
            _validate_filter_values({"beta_max": float("nan")})

    @pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
    def test_infinite_bound_rejected(self, bad):
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_filter_values({"rsi_min": bad})

    def test_non_numeric_bound_rejected_once_not_per_ticker(self):
        with pytest.raises(ValidationError, match="must be a number"):
            _validate_filter_values({"rsi_max": "fifty"})

    def test_bool_is_not_accepted_as_a_number(self):
        """bool subclasses int, so True would otherwise sail through as 1."""
        with pytest.raises(ValidationError, match="must be a number"):
            _validate_filter_values({"rsi_max": True})

    @pytest.mark.parametrize("bad", [-5, 0, 1.5])
    def test_window_filters_must_be_positive_whole_numbers(self, bad):
        with pytest.raises(ValidationError, match="positive whole number"):
            _validate_filter_values({"price_above_sma": bad})

    def test_valid_filters_still_accepted(self):
        _validate_filter_values(
            {"rsi_max": 50, "price_above_sma": 200, "beta_max": 1.2, "roe_min": 0.15}
        )
