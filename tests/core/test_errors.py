"""Tests for the custom exception hierarchy and input validation decorators."""

import pandas as pd
import pytest

from standard_quant_tools.data._retry import retry
from standard_quant_tools.error import (
    APIError,
    BacktestError,
    CalculationError,
    DataNotFoundError,
    DataProviderError,
    InvalidSymbolError,
    NonRetryableAPIError,
    QuantError,
    ValidationError,
)
from standard_quant_tools.indicators.momentum import rsi
from standard_quant_tools.metrics.risk_metrics import sharpe_ratio, sortino_ratio


class TestExceptionHierarchy:
    def test_quant_error_is_base(self):
        assert issubclass(DataProviderError, QuantError)
        assert issubclass(CalculationError, QuantError)
        assert issubclass(ValidationError, QuantError)
        assert issubclass(BacktestError, QuantError)

    def test_data_provider_errors_inherit_correctly(self):
        assert issubclass(DataNotFoundError, DataProviderError)
        assert issubclass(InvalidSymbolError, DataProviderError)
        assert issubclass(APIError, DataProviderError)

    def test_non_retryable_api_error_is_an_api_error(self):
        # Deliberately a subclass, not a sibling -- every existing
        # `except APIError` call site must keep working unchanged.
        assert issubclass(NonRetryableAPIError, APIError)

    def test_all_errors_are_exceptions(self):
        for exc in (
            QuantError,
            DataProviderError,
            DataNotFoundError,
            InvalidSymbolError,
            APIError,
            CalculationError,
            ValidationError,
            BacktestError,
        ):
            assert issubclass(exc, Exception)

    def test_quant_error_stores_original_exception(self):
        original = ValueError("original")
        wrapped = QuantError("wrapped", original_exception=original)
        assert wrapped.original_exception is original

    def test_quant_error_message_preserved(self):
        err = QuantError("test message")
        assert str(err) == "test message"

    def test_data_not_found_is_catchable_as_provider_error(self):
        with pytest.raises(DataProviderError):
            raise DataNotFoundError("no data")

    def test_invalid_symbol_is_catchable_as_quant_error(self):
        with pytest.raises(QuantError):
            raise InvalidSymbolError("bad symbol")


class TestValidateSeriesDecorator:
    def test_empty_series_raises_validation_error(self):
        with pytest.raises(ValidationError):
            sharpe_ratio(pd.Series(dtype=float))

    def test_empty_series_raises_for_sortino(self):
        with pytest.raises(ValidationError):
            sortino_ratio(pd.Series(dtype=float))

    def test_empty_series_raises_for_rsi(self):
        with pytest.raises(ValidationError):
            rsi(pd.Series(dtype=float))

    def test_non_empty_series_does_not_raise(self):
        import numpy as np

        s = pd.Series(np.random.normal(0, 0.01, 50))
        sharpe_ratio(s)  # should not raise


class TestDataProviderErrors:
    def test_invalid_symbol_raises_invalid_symbol_error(
        self, patched_factory, mock_provider
    ):
        from standard_quant_tools.data.factory import DataFactory
        from standard_quant_tools.error import InvalidSymbolError

        mock_provider.get_ohlcv.side_effect = InvalidSymbolError("bad ticker")
        provider = DataFactory.get_provider()
        with pytest.raises(InvalidSymbolError):
            provider.get_ohlcv("", "2023-01-01", "2024-01-01")

    def test_data_not_found_error_on_missing_data(self, patched_factory, mock_provider):
        from standard_quant_tools.data.factory import DataFactory
        from standard_quant_tools.error import DataNotFoundError

        mock_provider.get_ohlcv.side_effect = DataNotFoundError("no data")
        provider = DataFactory.get_provider()
        with pytest.raises(DataNotFoundError):
            provider.get_ohlcv("FAKE", "2023-01-01", "2024-01-01")


class TestRetryDecorator:
    """Regression coverage: the retry decorator used to retry HTTP 401/403
    (permanent, e.g. an invalid API key) identically to 429/5xx (genuinely
    transient), burning through a rate-limited API's request budget on
    every single call until the key was fixed."""

    def test_non_retryable_api_error_is_not_retried(self):
        calls = []

        @retry(times=3, delay=0)
        def always_fails_permanently():
            calls.append(1)
            raise NonRetryableAPIError("invalid key")

        with pytest.raises(NonRetryableAPIError):
            always_fails_permanently()
        assert len(calls) == 1

    def test_plain_api_error_is_retried_up_to_the_configured_times(self):
        calls = []

        @retry(times=3, delay=0)
        def always_fails_transiently():
            calls.append(1)
            raise APIError("rate limited")

        with pytest.raises(APIError):
            always_fails_transiently()
        assert len(calls) == 3

    def test_non_retryable_api_error_succeeds_immediately_if_no_error(self):
        calls = []

        @retry(times=3, delay=0)
        def succeeds():
            calls.append(1)
            return "ok"

        assert succeeds() == "ok"
        assert len(calls) == 1
