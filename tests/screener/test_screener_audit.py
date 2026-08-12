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
    DEFAULT_MIN_BETA_OBS,
    _fetch_ticker_data,
    _screen_batch,
    _validate_filter_values,
    _validate_min_beta_obs,
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


class TestConfigurableBetaFloor:
    """
    20 is a judgment call, not a mathematical bound: weekly bars, or a
    deliberate recent-listing screen, are legitimate reasons to lower it. So
    it is a default rather than a rule -- but bounded, because the value that
    reopens the original bug is not a matter of taste.
    """

    def _run(self, filters, floor, overlap_bars=8):
        spy = _ohlcv(300, "2023-01-02")
        asset = _ohlcv(overlap_bars, "2023-01-02")
        return asyncio.run(
            _fetch_ticker_data(
                _provider(asset, spy),
                "NEWCO",
                "2023-01-02",
                "2024-01-01",
                filters,
                spy,
                floor,
            )
        )

    def test_default_matches_the_documented_constant(self):
        assert DEFAULT_MIN_BETA_OBS == 20
        assert _MIN_BETA_OBS == DEFAULT_MIN_BETA_OBS, "legacy alias must not drift"

    def test_default_floor_rejects_a_short_overlap(self):
        status, _, payload = self._run({"beta_max": 5.0}, DEFAULT_MIN_BETA_OBS)
        assert status == "error"
        assert "not estimable" in payload

    def test_lowering_the_floor_lets_the_same_ticker_be_evaluated(self):
        """Same ticker, same data — only the caller's stated tolerance
        differs, and now it actually gets a beta computed."""
        status, _, _ = self._run({"beta_max": 5.0}, 5)
        assert status != "error"

    def test_error_message_quotes_the_floor_actually_in_force(self):
        """The message interpolated the module constant. Left that way it
        would have reported 20 while enforcing whatever the caller passed —
        an error message lying about its own threshold."""
        status, _, payload = self._run({"beta_max": 5.0}, 15)
        assert status == "error"
        assert "need 15" in payload
        assert "need 20" not in payload

    @pytest.mark.parametrize("bad", [1, 0, -3])
    def test_floor_below_two_is_rejected(self, bad):
        """
        Below two overlapping points calculate_beta returns its all-zero
        sentinel, which is indistinguishable from a real beta of 0.0 — so a
        floor under 2 would reopen exactly the bug the floor exists to close.
        Configurable does not mean unbounded.
        """
        with pytest.raises(ValidationError, match="must be >= 2"):
            _validate_min_beta_obs(bad)

    def test_rejection_explains_the_sentinel_collision(self):
        with pytest.raises(ValidationError, match="sentinel"):
            _validate_min_beta_obs(1)

    @pytest.mark.parametrize("bad", [2.5, "20", True])
    def test_non_integer_floor_rejected(self, bad):
        with pytest.raises(ValidationError, match="must be an int"):
            _validate_min_beta_obs(bad)

    def test_absolute_minimum_is_accepted(self):
        assert _validate_min_beta_obs(2) == 2


class TestWorkerTupleCarriesEveryTunable:
    """
    The ProcessPoolExecutor worker rebuilds its own call from a plain tuple.
    A parameter left out of that tuple does not fail -- it silently reverts
    to its default inside the child, so the same request would screen
    differently at n_workers=1 than at n_workers=8. That is the quiet kind of
    divergence this codebase keeps finding, so the unpack is strict.
    """

    def test_batch_worker_requires_the_full_tuple(self):
        with pytest.raises(ValueError):
            _screen_batch((["AAA"], {"rsi_max": 50}, "2023-01-02", "2024-01-01"))

    def test_screen_stocks_forwards_the_floor_to_every_worker(self, monkeypatch):
        """
        The executor is faked rather than the worker: ProcessPoolExecutor
        runs _screen_batch in a CHILD process that re-imports the real
        module, so monkeypatching the worker in the parent intercepts
        nothing. Capturing at submit() is what actually observes the tuple
        the child will be handed.
        """
        import standard_quant_tools.screener.screener as screener_mod

        captured = []

        class _FakeFuture:
            def result(self):
                return pd.DataFrame()

        class _FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, args):
                captured.append(args)
                return _FakeFuture()

        monkeypatch.setattr(screener_mod, "ProcessPoolExecutor", _FakeExecutor)
        screener_mod.screen_stocks(
            [f"T{i}" for i in range(40)],
            {"beta_max": 1.5},
            start_date="2023-01-02",
            end_date="2024-01-01",
            n_workers=4,
            min_beta_obs=7,
        )
        assert captured, "expected the multi-worker path to be taken"
        assert all(
            args[-1] == 7 for args in captured
        ), "every batch must carry the caller's floor, not the module default"

    def test_worker_tuple_shape_matches_what_the_worker_unpacks(self, monkeypatch):
        """Submit-side and unpack-side arity are asserted against each other,
        since a mismatch is the whole failure mode."""
        import standard_quant_tools.screener.screener as screener_mod

        captured = []

        class _FakeFuture:
            def result(self):
                return pd.DataFrame()

        class _FakeExecutor:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, args):
                captured.append(args)
                return _FakeFuture()

        monkeypatch.setattr(screener_mod, "ProcessPoolExecutor", _FakeExecutor)
        screener_mod.screen_stocks(
            [f"T{i}" for i in range(40)],
            {"rsi_max": 50},
            start_date="2023-01-02",
            end_date="2024-01-01",
            n_workers=2,
        )
        # _screen_batch unpacks exactly this many names.
        assert all(len(args) == 5 for args in captured)
