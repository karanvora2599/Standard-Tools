# Data Fetching

The data layer wraps yfinance with caching, retry logic, and Pydantic-validated outputs. All providers implement the same `DataProvider` ABC so swapping sources requires zero changes to downstream code.

---

## Basic OHLCV Fetch

```python
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider()  # defaults to yfinance

df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
print(df.columns)  # ['Open', 'High', 'Low', 'Close', 'Volume']
print(df.head())
```

Supported intervals: `"1d"` (default), `"1wk"`, `"1mo"`, `"1h"`, `"15m"`, etc.

---

## Async Batch Fetching

Fetch multiple tickers concurrently. All tasks run in parallel; total wall time ≈ single-ticker time.

```python
import asyncio
from standard_quant_tools.data.factory import DataFactory

async def fetch_universe(tickers):
    provider = DataFactory.get_provider()
    tasks = [
        provider.get_ohlcv_async(t, "2023-01-01", "2024-01-01")
        for t in tickers
    ]
    return await asyncio.gather(*tasks)

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
dfs = asyncio.run(fetch_universe(tickers))
# dfs[0] = AAPL, dfs[1] = MSFT, ...
```

---

## Company Metadata

```python
info = provider.get_ticker_info("TSLA")
print(info.name)     # "Tesla, Inc."
print(info.sector)   # "Consumer Cyclical"
print(info.industry) # "Auto Manufacturers"
print(info.model_dump())
# {symbol, name, sector, industry, full_time_employees, city, country, website}
```

`TickerInfo` is a Pydantic model — call `.model_dump_json()` to pass it directly to an LLM.

---

## Financial Ratios

```python
ratios = provider.get_financial_ratios("MSFT")
print(f"Forward P/E : {ratios.forward_pe}")
print(f"P/B         : {ratios.price_to_book}")
print(f"D/E         : {ratios.debt_to_equity}")
print(f"ROE         : {ratios.return_on_equity:.1%}")
print(f"Profit Margin: {ratios.profit_margins:.1%}")
```

All ratio fields are `Optional[float]` — missing data returns `None` rather than crashing.

---

## Caching & Retry

- **TTL cache**: identical calls within 1 hour return the cached DataFrame (no network round-trip)
- **Retry**: up to 3 attempts with exponential backoff (1s, 2s, 4s) on transient failures
- **Thread-safe**: the cache key includes `(symbol, start_date, end_date, interval)`

To force a fresh fetch, create a new provider instance (cache is per-instance):

```python
fresh_provider = DataFactory.get_provider("yfinance")
df = fresh_provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
```

---

## Error Handling

```python
from standard_quant_tools.error import DataNotFoundError, InvalidSymbolError, APIError

try:
    df = provider.get_ohlcv("INVALID_XYZ", "2023-01-01", "2024-01-01")
except DataNotFoundError:
    print("Symbol not found or no data in date range.")
except InvalidSymbolError:
    print("Symbol string is malformed or empty.")
except APIError as e:
    print(f"Network/API error: {e}")
```

Errors are designed to be descriptive enough for LLM self-correction — the message always includes the symbol and the reason for failure.

---

## Portfolio-Level Async Fetch

For multi-asset workflows, use the portfolio module's built-in async fetch:

```python
from standard_quant_tools.portfolio import fetch_returns_sync

# Returns a DataFrame of daily returns, one column per ticker
returns_df = fetch_returns_sync(
    ["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2024-01-01",
)
print(returns_df.shape)  # (252, 3)
```
