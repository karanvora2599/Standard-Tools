import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.factory import DataFactory
from datetime import datetime, timedelta

def test_yfinance():
    print("Testing YFinanceProvider...")
    factory = DataFactory()
    provider = factory.get_provider("yfinance")
    
    symbol = "AAPL"
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        print(f"Fetching OHLCV for {symbol}...")
        df = provider.get_ohlcv(symbol, start_date, end_date)
        print("Success! Head of DataFrame:")
        print(df.head())
        
    except Exception as e:
        print(f"Error fetching OHLCV: {e}")

    try:
        print(f"\nFetching info for {symbol}...")
        info = provider.get_ticker_info(symbol)
        print(f"Success! Info: {info}")
    except Exception as e:
        print(f"Error fetching info: {e}")

if __name__ == "__main__":
    test_yfinance()
