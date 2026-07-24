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

- **TTL cache**: identical calls within 1 hour return a `.copy()` of the cached DataFrame (no network round-trip); holds up to 100 entries, LRU-evicted beyond that
- **Retry**: up to 3 attempts, waiting 1s then 2s between attempts (exponential backoff, factor 2) on transient failures
- **Cache key**: `(id(self), symbol, start_date, end_date, interval)` — `get_ohlcv` checks the session cache itself rather than via a `@cached()` decorator wrapping the whole method, so an audit record is written on every call, including a session-cache hit, not just on a live fetch. `id(self)` (not just the call args) keeps a fresh provider instance from transparently reusing another instance's cached result. It's not lock-guarded, so it isn't safe against races between concurrent threads hitting the same instance/args at once
- **Copy-on-return**: every `get_ohlcv` call — session-cache hit, disk-cache hit, or live fetch — returns a fresh copy, so a caller mutating the result in place can't corrupt the cached object shared with the next caller

To force a fresh fetch, create a new provider instance (cache is per-instance):

```python
fresh_provider = DataFactory.get_provider("yfinance")
df = fresh_provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
```

---

## Persistent Parquet Cache

Every `get_ohlcv` call for a **historical date range** (end date before today) is automatically saved as a Parquet file. Subsequent calls — even from a completely new Python process — skip the network entirely and load from disk.

```
~/.cache/standard_quant_tools/ohlcv/AAPL_2020-01-01_2024-01-01_1d.parquet
```

**Why only historical ranges?** "Historical" here means the bar is no longer forming — it does *not* mean the cached values can never change. Data is fetched with `auto_adjust=True`, so a later corporate action (split, special dividend) can retroactively revise the adjusted Close/Open/High/Low for dates already on disk. The cache trades that small staleness risk for avoiding repeated network calls; a symbol with a recent corporate action needs the cache cleared or bypassed (`SQT_CACHE_DIR`) rather than assuming it self-heals. Today's still-forming bar always goes through the in-process TTL cache (1 hour) instead, never the disk cache.

```python
import time

provider = DataFactory.get_provider()

# First call: fetches from yfinance, writes Parquet (~300ms)
t0 = time.perf_counter()
df = provider.get_ohlcv("NVDA", "2020-01-01", "2024-01-01")
print(f"First call: {time.perf_counter() - t0:.2f}s")

# Exit Python, restart, call again
# Second call: reads from Parquet (~5ms)
t0 = time.perf_counter()
df = provider.get_ohlcv("NVDA", "2020-01-01", "2024-01-01")
print(f"Cached call: {time.perf_counter() - t0:.3f}s")
```

**Corrupt cache files evict themselves**: if a Parquet file on disk fails to read (truncated write, disk corruption, etc.), it's logged, deleted, and the data is transparently refetched from yfinance and rewritten — callers never see the corrupt file or an exception because of it.

**Cache path safety**: `symbol`, `start_date`/`end_date`, and `interval` are all validated (allow-listed characters, `..` rejected) before being used to build the Parquet filename, and the resolved path is checked to still resolve inside the cache root — a malformed or adversarial symbol string (these are LLM-reachable via `get_ohlcv`'s own parameters) can't write outside `SQT_CACHE_DIR`.

**Override the cache directory** via the `SQT_CACHE_DIR` environment variable:

```bash
export SQT_CACHE_DIR=/data/market_cache   # Linux/Mac
set SQT_CACHE_DIR=D:\market_cache         # Windows
```

The cache is safe for concurrent access — each write goes to a temp file unique to the process, the thread, and a random suffix, then is atomically renamed into place, so races between workers (e.g. parallel screener) — including multiple threads writing the same symbol/range within one process — are handled correctly.

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

---

## Dataset Provenance and Data Quality

Every `DataProvider` also implements `get_metadata(symbol, interval)`,
reporting what guarantees the fetched data actually carries (adjusted?
survivorship-free? point-in-time?), plus standalone checks for missing
bars, stale prices, and large single-bar jumps on data you've already
fetched. See [11_data_quality.md](11_data_quality.md) for the full
reference — this is the credibility-of-the-data-itself counterpart to the
backtesting engine's own trustworthiness work in
[04_backtesting.md](04_backtesting.md).
