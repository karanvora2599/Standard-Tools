import sys
import os

# Add src to path BEFORE importing local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import pytest
from standard_quant_tools.error import DataNotFoundError, InvalidSymbolError, ValidationError, APIError
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.metrics.risk_metrics import sharpe_ratio
from standard_quant_tools.indicators.momentum import rsi

def test_data_provider_errors():
    print("\n--- Testing Data Provider Errors ---")
    provider = DataFactory.get_provider("yfinance")

    # 1. Invalid Symbol
    print("1. Testing Invalid Symbol (Should raise InvalidSymbolError or DataNotFoundError)...")
    try:
        provider.get_ohlcv("INVALID_SYMBOL_XYZ_123", "2023-01-01", "2023-01-10")
        print("FAIL: Did not raise error for invalid symbol")
    except (DataNotFoundError, InvalidSymbolError) as e:
        print(f"SUCCESS: Caught expected error: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error type: {type(e).__name__}: {e}")

    # 2. Empty Request
    print("\n2. Testing Empty Symbol (Should raise InvalidSymbolError)...")
    try:
        provider.get_ohlcv("", "2023-01-01", "2023-01-10")
        print("FAIL: Did not raise error for empty symbol")
    except InvalidSymbolError as e:
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {type(e).__name__}: {e}")

def test_validation_errors():
    print("\n--- Testing Input Validation ---")
    
    # 1. Empty Series for Sharpe
    print("1. Testing Sharpe Ratio with Empty Series...")
    empty_series = pd.Series(dtype=float)
    try:
        sharpe_ratio(empty_series)
        print("FAIL: Did not raise ValidationError")
    except ValidationError as e:
        print(f"SUCCESS: Caught expected ValidationError: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {type(e).__name__}: {e}")

    # 2. Empty Series for RSI
    print("\n2. Testing RSI with Empty Series...")
    try:
        rsi(empty_series)
        print("SUCCESS: RSI handled empty series gracefully (returned empty)")
    except ValidationError as e:
        print(f"SUCCESS: Caught expected ValidationError: {e}")
    except Exception as e:
        print(f"FAIL: Caught unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_data_provider_errors()
    test_validation_errors()
