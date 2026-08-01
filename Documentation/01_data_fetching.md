# Data Fetching

The data layer wraps yfinance (and, optionally, a Bloomberg Terminal via Desktop API, or Polygon.io's REST API) with caching, retry logic, and Pydantic-validated outputs. All providers implement the same `DataProvider` ABC so swapping sources requires zero changes to downstream code.

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
- **Cache key**: `(provider_name, instance_token, symbol, start_date, end_date, interval)` — `get_ohlcv` checks the session cache itself rather than via a `@cached()` decorator wrapping the whole method, so an audit record is written on every call, including a session-cache hit, not just on a live fetch. The per-instance token (a UUID, not `id(self)` — CPython can reuse a freed object's `id()`) keeps a fresh provider instance from transparently reusing another instance's cached result. The cache dict itself is guarded by a module-level lock (`data/_cache.py`), so concurrent threads hitting the same or different instances/args at once are safe — the lock only wraps the get/set, not the network fetch, so calls to different keys still run concurrently
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

**Cache path safety**: `symbol`, `start_date`/`end_date`, and `interval` are all validated (allow-listed characters, `..` rejected) before being used to build the Parquet filename, and the resolved path is checked to still resolve inside the cache root — a malformed or adversarial symbol string (these are LLM-reachable via `get_ohlcv`'s own parameters) can't write outside `SQT_CACHE_DIR`. A symbol that fails this check doesn't cause `get_ohlcv` itself to fail, though: caching is an optimization, not a correctness requirement, so every provider degrades gracefully by skipping the disk cache for that one call (still served live/from the session cache) rather than raising `ValidationError` for a symbol its own live-fetch path can otherwise handle fine.

**Override the cache directory** via the `SQT_CACHE_DIR` environment variable:

```bash
export SQT_CACHE_DIR=/data/market_cache   # Linux/Mac
set SQT_CACHE_DIR=D:\market_cache         # Windows
```

The cache is safe for concurrent access — each write goes to a temp file unique to the process, the thread, and a random suffix, then is atomically renamed into place, so races between workers (e.g. parallel screener) — including multiple threads writing the same symbol/range within one process — are handled correctly.

---

## Error Handling

```python
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
)

try:
    df = provider.get_ohlcv("INVALID_XYZ", "2023-01-01", "2024-01-01")
except DataNotFoundError:
    print("Symbol not found or no data in date range.")
except InvalidSymbolError:
    print("Symbol string is malformed or empty.")
except NonRetryableAPIError as e:
    print(f"Permanent API failure (e.g. a bad key) — won't succeed on retry: {e}")
except APIError as e:
    print(f"Network/API error: {e}")
```

Errors are designed to be descriptive enough for LLM self-correction — the message always includes the symbol and the reason for failure.

`NonRetryableAPIError` is a subclass of `APIError` (so an existing `except APIError` still catches it — it's a narrowing, not a new branch you have to add), used for failures the shared `retry` decorator knows will never succeed no matter how many times it's retried — currently just `PolygonProvider`'s HTTP 401/403 (an invalid/expired API key). Everything else `APIError`-shaped (429 rate limits, 5xx, network errors) is retried with the usual exponential backoff; `DataNotFoundError`/`InvalidSymbolError` are also never retried, for the same reason (retrying "the symbol doesn't exist" can't change the answer).

---

## Portfolio-Level Async Fetch

For multi-asset workflows, use the portfolio module's built-in async fetch —
one `asyncio.gather` round-trip per ticker instead of a blocking loop, which
matters once you're past a handful of tickers (a few seconds for dozens of
tickers rather than one network round-trip's latency multiplied by the
ticker count):

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

If you need the full OHLCV panel (Volume/High/Low, not just Close-derived
returns — e.g. to feed your own ADV or volatility calculation) use
`fetch_ohlcv_panel_sync` instead, same concurrency, different return shape:

```python
from standard_quant_tools.portfolio import fetch_ohlcv_panel_sync

# Dict[ticker, DataFrame] — each DataFrame has the usual Open/High/Low/Close/Volume columns
panel = fetch_ohlcv_panel_sync(
    ["AAPL", "MSFT", "GOOGL"],
    start_date="2023-01-01",
    end_date="2024-01-01",
)
print(panel["AAPL"].columns.tolist())  # ['Open', 'High', 'Low', 'Close', 'Volume']
```

Both of the agent tools that operate over a full ticker universe with
rebalancing (`run_portfolio_simulation`, `run_signal_panel_backtest`) use
`fetch_ohlcv_panel_sync` internally — every multi-ticker tool in the
package fetches concurrently this way, so a large universe (e.g. the full
S&P 500) is bounded by the default executor's thread pool (~32 requests in
flight), not by ticker count times per-request latency.

---

## Bloomberg Provider

`standard_quant_tools.data.bloomberg_provider.BloombergProvider` implements
the same `DataProvider` ABC against a locally running, **logged-in Bloomberg
Terminal** via Desktop API (DAPI) — same `get_ohlcv`/`get_ticker_info`/
`get_financial_ratios`/`get_metadata` interface as `YFinanceProvider`, so
switching providers is a one-line change:

```python
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider("bloomberg")
df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
```

**No API key.** Desktop API authenticates via the Terminal login itself —
there is no separate secret this library holds. What *is* configurable is
purely connection-level (only relevant if you proxy DAPI to a non-default
address), and is read from the environment rather than hardcoded, the same
`SQT_*`-prefixed convention every other provider config in this package
uses:

| Variable | Default | Meaning |
|---|---|---|
| `SQT_BLOOMBERG_HOST` | `localhost` | DAPI server host |
| `SQT_BLOOMBERG_PORT` | `8194` | DAPI server port |

**Where these live:** copy [`.env.example`](../.env.example) (repo root) to
`.env` — already `.gitignore`d — for local development;
`standard_quant_tools.config.load_env()` loads it into `os.environ` once per
process automatically (and is a no-op, harmlessly, if `.env` doesn't
exist — the normal state in CI). In GitHub Actions / GitLab CI, set the same
variable names as encrypted repo/org secrets and inject them as job-level
environment variables instead of using a `.env` file at all — see the
comments at the bottom of `.env.example` for exact syntax on both platforms.
Real environment variables set any other way always win over a stale
`.env` value (`load_env()` never calls `override=True`).

```python
# Explicit args override SQT_BLOOMBERG_HOST/PORT for one instance:
provider = DataFactory.get_provider("bloomberg", host="10.0.0.5", port=8194)
```

**Ticker convention:** a bare symbol (`"AAPL"`) is normalized to a
fully-qualified Bloomberg ticker (`"AAPL US Equity"`) automatically. A
symbol that already ends in a recognized market-sector keyword (`Equity`,
`Govt`, `Corp`, `Curncy`, `Comdty`, `Index`, `Mtge`, `Muni`, `Pfd`) is passed
through unchanged — pass the fully-qualified ticker yourself for anything
non-US or non-equity (e.g. `"VOD LN Equity"`, `"EURUSD Curncy"`).

**Scope, stated explicitly:**
- Only daily/weekly/monthly bars are supported (`HistoricalDataRequest`).
  Intraday intervals raise a clear `ValidationError` rather than silently
  returning wrong data — proper intraday support needs a structurally
  different request (`IntradayBarRequest`, with its own history-depth
  limits) that isn't implemented.
- `get_metadata()` honestly reports `survivorship_free=False` and
  `point_in_time=False` — plain Desktop API makes neither guarantee; a real
  point-in-time/survivorship-free feed needs Bloomberg's enterprise data
  products (e.g. PORT), not DAPI.
- No caching layer (session TTL cache or persistent Parquet disk cache) yet —
  unlike `YFinanceProvider`, every call reaches the Terminal. Worth adding
  if Bloomberg becomes a hot path; not built preemptively.
- `blpapi` (Bloomberg's own SDK) is an **optional** dependency —
  `pip install standard_quant_tools[bloomberg]` (or `pip install blpapi`
  directly; if that doesn't resolve, use Bloomberg's own package index,
  `pip install --index-url
  https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`).
  Constructing `BloombergProvider()` (directly or via
  `DataFactory.get_provider("bloomberg")`) without it installed raises a
  clear `APIError` explaining how to install it, rather than an opaque
  `ImportError` — the rest of the package works normally either way.

---

## Polygon.io Provider

`standard_quant_tools.data.polygon_provider.PolygonProvider` implements the
same `DataProvider` ABC against Polygon.io's plain REST API — no vendor SDK
to install, just an API key:

```python
from standard_quant_tools.data.factory import DataFactory

provider = DataFactory.get_provider("polygon")  # or api_key="..." explicitly
df = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
```

**API key required, no default.** Read from `SQT_POLYGON_API_KEY` (via a
local `.env` — copy [`.env.example`](../.env.example) — or a real
environment variable / CI secret), or pass `api_key=` explicitly to
`DataFactory.get_provider("polygon", api_key=...)`. Get a free key at
[polygon.io/dashboard/api-keys](https://polygon.io/dashboard/api-keys).
Constructing `PolygonProvider()` (directly or via the factory) with no key
resolvable anywhere raises a clear `APIError` rather than an opaque
failure deep inside the first network call.

**Supported intervals:** `"1m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"`/`"1h"`,
`"1d"`, `"1wk"`, `"1mo"`, `"3mo"` — the subset Polygon's Aggregates (Bars)
endpoint supports natively. Anything else raises `ValidationError` rather
than silently guessing a mapping.

**Scope, stated explicitly:**
- Only plain equity tickers are exercised end-to-end; crypto (`X:BTCUSD`)
  and forex (`C:EURUSD`) prefixes may work against the same aggs endpoint
  but are untested here.
- `get_ohlcv` fetches a single page (`limit=50000`). A request whose true
  result set exceeds one page — mostly a risk for long intraday ranges — is
  **not** paginated; a logged warning fires when Polygon's response
  indicates more pages exist (`next_url` present), so truncation is visible
  rather than silent, but the remaining pages aren't fetched.
- `get_financial_ratios` has no direct analogue to yfinance's `.info`
  ratios in Polygon's reference data. `market_cap` comes straight from
  Ticker Details v3. `trailing_pe`, `price_to_book`, `debt_to_equity`,
  `return_on_equity`, and `profit_margins` are derived from the most recent
  filing on the Financials vX endpoint combined with `market_cap` (e.g.
  `trailing_pe ~= market_cap / net_income`). `forward_pe` (no forward
  estimates in this data) and `dividend_yield` (would need a separate
  dividends-history aggregation) are always `None` — missing, not wrong.
- `get_ticker_info`'s `sector`/`industry` both fall back to Polygon's single
  `sic_description` classification field — a coarser taxonomy than
  yfinance's separate sector/industry fields.
- `get_metadata()` honestly reports `survivorship_free=False` and
  `point_in_time=False` — this provider makes neither guarantee.
- Shares the same two-tier cache as `YFinanceProvider` (in-memory session
  TTL cache + persistent Parquet disk cache, both in `data/_cache.py`), so
  a repeated call for the same symbol/date-range/interval doesn't reach
  Polygon at all. The free tier is rate-limited (5 requests/minute at the
  time of writing); a 429 is retried like any other transient `APIError`
  via the shared `retry` decorator, with no Polygon-specific backoff
  tuning. A 401/403 (invalid/expired API key) is raised as
  `NonRetryableAPIError` instead and is never retried, since retrying a
  bad key can't make it valid.

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
