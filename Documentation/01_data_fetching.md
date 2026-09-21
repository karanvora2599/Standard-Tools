# Data Fetching

> **Fetching is also a RUNTIME.** This page is about the providers
> themselves — how they are configured, what they guarantee, how retries
> and caching behave. The agent-facing side, where a fetch publishes an
> `sqt://` reference that every other runtime reads instead of refetching,
> is [26_data.md](26_data.md).

The data layer wraps yfinance (and, optionally, a Bloomberg Terminal via Desktop API, Polygon.io's REST API, or Databento Historical — the only one that serves L2 depth) with caching, retry logic, and Pydantic-validated outputs. All providers implement the same `DataProvider` ABC so swapping sources requires zero changes to downstream code.

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

### `end_date` is inclusive, on every provider

`get_ohlcv("AAPL", "2023-01-01", "2024-01-01")` returns bars **through**
2024-01-01, not up to it. A bare date means "through the end of that day" at
every interval; an explicit intraday timestamp means exactly that instant.

This is stated by the `DataProvider` ABC and is not optional for
implementations. It had to be stated because the three shipped providers had
already drifted apart on the answer:

| Provider | Underlying call | Native semantics |
|---|---|---|
| yfinance | `ticker.history(end=...)` | **exclusive** |
| Polygon | `/v2/aggs/.../range/{from}/{to}` | inclusive |
| Bloomberg | `request.set("endDate", ...)` | inclusive |

So the same call returned a different window depending only on who served
it — and on the *default* provider it silently dropped the final bar. That
is the mechanism behind `score_model(as_of=X)` excluding X's own data while
still reporting X as the as-of date.

Inclusive won because it matches two of the three, matches what a caller
passing a date means, and is the only reading under which an as-of date can
be reported honestly. `YFinanceProvider` requests an exclusive bound past
the whole inclusive window (over-fetching at most one day, which is
harmless — under-fetching silently changes the answer) and trims back.

**All three trim through the shared `trim_to_inclusive_end`, including the
two that were already inclusive.** Deliberate: the contract then holds by
*construction* rather than by trusting each vendor's documented boundary, so
a vendor changing or mis-documenting its own semantics cannot quietly move
the window. It costs one boolean mask on an already-materialized frame.

> **Cache format is `v3`.** Two bumps, both for the same reason — a stale
> file would answer a request differently than a live fetch, which is exactly
> the cache/live parity failure this layer exists to prevent. `v1` files were
> written under the exclusive-end behaviour and are missing their final bar;
> `v2` intraday files hold local wall-clock timestamps rather than the
> canonical UTC described below. Superseded files are never looked up again
> rather than migrated — they age out with the directory.

### Intraday timestamps survive the round trip

Index normalization is interval-aware. It used to call `idx.normalize()`
unconditionally, setting every timestamp to midnight — four hourly bars
became four copies of one date, losing time-series identity outright. It ran
on yfinance's live fetch *and* on both providers' Parquet cache reads, so it
also made the same request answer differently live vs cached (Polygon's live
parse preserved intraday timestamps; the cache read did not).

**Intraday timestamps are canonical UTC.** Stripping the timezone without
converting first keeps the *local wall clock*, which makes bars from different
exchanges look simultaneous:

| Venue | Local | True instant (UTC) | Old naive index |
|---|---|---|---|
| London | 15:00 BST | 14:00 UTC | `15:00` |
| New York | 15:00 EDT | 19:00 UTC | `15:00` |

Those two bars are **five hours apart**, and their normalized indexes were
equal — so a join, a correlation, a PCA or a cross-sectional panel silently
paired a London afternoon with a New York afternoon as one instant. Nothing
raised; the numbers simply described a market state that never existed.
Intraday data is now converted to UTC before the timezone is dropped, in the
Polygon parser as well, which had been emitting naive New York time.

**Daily and coarser deliberately do not convert.** A daily bar is identified
by its *local trading date*, and converting first would shift it: Tokyo
2024-06-03 00:00 JST is 2024-06-02 15:00 UTC, which normalizes to the wrong
day. An intraday bar is an instant; a daily bar is a session. The two are
handled differently on purpose.

Cache identity gained the same awareness: cache-key date bounds used to be
truncated to 10 characters, so `09:30→12:00` and `13:00→16:00` on the same
day resolved to **one** file and the second silently served the first's
bars. Intraday bounds now carry `HHMMSS`; daily tokens are unchanged, so
existing daily cache files stay addressable.

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

### One canonical unit and definition, whichever provider served it

The shared field names used to imply an interchangeability that did not exist:

| Field | yfinance | Polygon |
|---|---|---|
| `debt_to_equity` | `150.5` (a **percentage**) | `1.505` (a plain **ratio**) |

A screen written as `debt_equity_max=2.0` therefore admitted nearly every
company on one provider and nearly none on the other, with nothing in either
result saying which convention was in force. Every field now has one canonical
unit, and each provider converts to it:

| Field | Canonical unit |
|---|---|
| `forward_pe`, `trailing_pe`, `price_to_book`, `debt_to_equity` | plain ratio |
| `return_on_equity`, `profit_margins`, `dividend_yield` | decimal fraction (`0.15` == 15%) |
| `market_cap` | absolute units of the reporting currency |

The screener's filters (`roe_min`, `debt_equity_max`, …) are stated in these
same units, so a filter now means the same thing on every provider.

**Units are converted; definitions are declared.** These are different
problems. Bloomberg's `TOT_DEBT_TO_TOT_EQY` is total *debt* over equity, while
Polygon derives its ratio from total *liabilities* — which include payables,
deferred revenue and lease obligations. That figure is systematically higher
for the same company, not by a scale factor that could be corrected but
because it answers a different question. `FinancialRatios.definition_notes`
names any field whose basis departs from the canonical one:

```python
ratios = polygon.get_financial_ratios("AAPL")
ratios.definition_notes.get("debt_to_equity")
# 'Computed as total LIABILITIES / equity, not total DEBT / equity: ...'
```

The value is still returned rather than discarded — a liabilities-to-equity
ratio is useful when you know that is what it is. Shipping it silently under a
debt-based name was the actual problem.

**Suspicious values are reported, not auto-corrected.** Inferring "15.0 must
be a percentage" would silently rewrite a genuine 1500% return on equity,
which small-equity companies really do post. Providers declare their own
vendor's units, and a value that looks like an unconverted percentage produces
a logged warning instead — so a vendor changing convention (as yfinance did
with `dividendYield` between releases) surfaces as a warning rather than as a
wrong number that every downstream screen silently inherits.

---

## Caching & Retry

- **TTL cache**: identical calls within 1 hour return a `.copy()` of the cached DataFrame (no network round-trip); holds up to 100 entries, LRU-evicted beyond that. A window whose end is not yet historical (its last bar is still forming) is kept for **60 seconds** only — the hour-long TTL used to serve an unsettled bar as final for up to an hour, and a minute still turns three identical requests in one run into one metered fetch. Every provider goes through this cache, Databento included (it used to bypass both caches and the retry layer, so three identical live requests were three metered fetches)
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

Every `get_ohlcv` call for a **historical date range** (end date before today, on the **UTC** date — the guard compared against the local date, so east of UTC+5:30 a session still trading was already "yesterday" and its mid-session bar was written to disk permanently) is automatically saved as a Parquet file. Subsequent calls — even from a completely new Python process — skip the network entirely and load from disk.

```
~/.cache/standard_quant_tools/ohlcv/v3_yfinance_AAPL_2020-01-01_2024-01-01_1d.parquet
~/.cache/standard_quant_tools/ohlcv/v3_databento-EQUS.SUMMARY_AAPL_2024-07-01_2025-06-30_1d.parquet
```

The filename carries the format generation, the provider and — for a provider that chooses a dataset per window — the dataset that answered, so two feeds 30x apart in volume never share one file.

**Why only historical ranges?** "Historical" here means the bar is no longer forming — it does *not* mean the cached values can never change. Data is fetched with `auto_adjust=True`, so a later corporate action (split, special dividend) can retroactively revise the adjusted Close/Open/High/Low for dates already on disk. The cache trades that small staleness risk for avoiding repeated network calls; a symbol with a recent corporate action needs the cache cleared or bypassed (`SQT_CACHE_DIR`) rather than assuming it self-heals. Today's still-forming bar always goes through the in-process session cache instead (for 60 seconds, not the hour a settled window gets), never the disk cache.

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

**Dead generations are collected, not read.** A format bump (see the `v3` note above) leaves the previous generation's files on disk, never looked up again; a live cache held 1,574 files, 501 of them dead. `sqt cache gc` lists them and `sqt cache gc --confirm` deletes them — only files carrying an old generation prefix, never the current generation and never a file without one.

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

### What `retry` retries, precisely

| Exception | Retried? | Final type seen by the caller |
|---|---|---|
| `APIError` (429, 5xx) | yes | `APIError` |
| `ValueError` | yes | `APIError` |
| Raw network/stdlib errors (`ConnectionError`, `TimeoutError`, `socket.gaierror`, `aiohttp`/`requests` client errors) | yes | `APIError`, chained from the original |
| `NonRetryableAPIError` (401/403) | no | `NonRetryableAPIError` |
| `InvalidSymbolError`, `DataNotFoundError` | no | unchanged |
| `ValidationError` and every other non-`APIError` `QuantError` | no | **unchanged** |

Two of those rows describe behavior that was fixed rather than merely
documented, and are worth knowing if you have code depending on the old
shape:

- **Raw network exceptions are now genuinely retried.** They are neither
  `ValueError` nor `APIError`, so a catch-all previously wrapped them and
  re-raised on the *first* attempt — the single most common transient
  failure mode was never actually retried. (Providers wrap most of their own
  network errors as `APIError` internally, which masked this in practice, but
  the decorator's own contract was wrong.)
- **`ValidationError` keeps its type.** It used to be caught by the same
  catch-all and re-raised as `APIError`, so `except ValidationError` around a
  decorated provider call never fired.

`retry(times=...)` requires `times >= 1` and raises `ValueError` at
decoration time otherwise. `times=0` previously returned `None` without ever
calling the wrapped function.

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

## Databento Provider

`standard_quant_tools.data.databento_provider.DatabentoProvider` implements
the same `DataProvider` ABC against Databento Historical, and is the one
provider that serves every tier: bars, ticks, top-of-book quotes, L2 depth
(`get_order_book`) and order-by-order events (`get_order_events`).

```python
provider = DataFactory.get_provider("databento")   # reads DATABENTO_API_KEY
df = provider.get_ohlcv("AAPL", "2024-07-01", "2025-06-30")
df.attrs["dataset"]                                 # "EQUS.SUMMARY"
```

**The key comes from the environment.** `DATABENTO_API_KEY`, never an
argument. The provider constructs without one and fails on its first fetch,
naming the variable — which is why `describe_data_capabilities` reports an
unconfigured Databento as `available=False` rather than taking construction
for availability.

**Which feed answers a daily request, and why it matters.** Databento
publishes several equity datasets and they are not the same tape:

| Dataset | What it is | Used for |
|---|---|---|
| `EQUS.SUMMARY` | the consolidated tape exactly (`ohlcv-1d`, from 2024-07-01) | any daily window it covers, first |
| `EQUS.MINI` | a **sample** feed — 2-4% of consolidated volume, a UTC-day close | the daily fallback before 2024-07-01 |
| `XNAS.BASIC`, `XNAS.ITCH` | one venue's feeds (`XNAS.ITCH` also carries depth) | intraday bars, ticks, quotes, depth |

`EQUS.MINI` used to be the default and was documented as the consolidated
tape, so daily closes and volumes read a few percent low with no warning.
`EQUS.SUMMARY` answers any daily window from 2024-07-01 first; the sample
feed is the fallback before it; intraday asks the venue feeds only, in the
same order the tick methods use, so bars and ticks for one window come from
one tape. The dataset that answered travels on `frame.attrs["dataset"]` and
in `get_metadata(...).notes`, and the disk cache is keyed by it. Override
the daily choice with `DATABENTO_OHLCV_DATASET` (or `DATABENTO_DATASET` /
`DATABENTO_DEPTH_DATASET` for the venue and depth feeds).

**A daily request no longer returns tomorrow.** The daily request ended one
day past the inclusive end and nothing trimmed, so every as-of query on this
provider read the next session's close (D1). `end_date` is inclusive here as
on every other provider, and Databento now shares the session cache, the
disk cache, the retry layer and the index normaliser the others use — it
used to bypass all of them, which is why the bars skipped normalisation, a
cached frame did not round-trip, and three identical live requests were
three metered fetches (D2). The frame carries `attrs["adjusted"] = False`
(the venue publishes unadjusted, which the backtest split screen reads), and
`get_temporal_contract("bars")` reports `revisions="unknown"` to agree with
`point_in_time=False`.

**A futures root is not an equity.** `ES`, `CL` and `GC` are equity tickers
as well as roots, and the provider used to resolve them to the equity —
`get_ohlcv("CL")` returned Colgate-Palmolive (D5). A bare ambiguous root is
now refused with the spellings for each reading: `ES.c.0` (front
continuous), `ESZ6` (a contract), `ES.FUT` (the parent) and OSI option
strings route to the futures and options datasets (`GLBX.MDP3`,
`OPRA.PILLAR`; override with `DATABENTO_FUTURES_DATASET` /
`DATABENTO_OPTIONS_DATASET`), while `ES~equity` names the ticker.

**`DataSetMetadata` gained a `notes` list** — the served dataset, the index
normalisation and any ambiguity travel in it — and its `timezone` is now
documented as a label for a normalised, naive index rather than a live
zone.

**Two things the live pass left open.** Databento marks some sessions
`degraded`, and this provider does not read that flag, so a session the
vendor marks is served unmarked; reading it needs the `statistics`
schema and an entitlement check the fix could not run. And
`compare_ratio_sources` compares fundamentals only: run against a
provider that serves bars, it reports zero entities compared with no
warnings, which means nothing was compared, not that nothing was found.

---

## Tick data (`get_trades` / `get_quotes`)

The first optional capability on the provider contract. **Two providers
implement it**: `PolygonProvider`, on a plan tier that includes tick data
(the free tier serves bars and fundamentals but returns 403 here), and
`DatabentoProvider`, from the venue tape. `DataFactory.get_provider("polygon")`
or `("databento")` selects one; the agent data runtime's fetch tools take the
same choice as `source`.

```python
trades = provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
# price  size  exchange, indexed by nanosecond SIP timestamp

quotes = provider.get_quotes("AAPL", "2024-01-02", "2024-01-03")
# bid_price  bid_size  ask_price  ask_size
```

Four things worth knowing before you use them:

**They are not abstract methods.** yfinance and Bloomberg inherit a base
implementation that raises `NotImplementedError` naming the provider, naming
the two that do work (the message said only Polygon did, which was false
once Databento served both), and refusing to offer bars as a substitute. Making
them abstract would break those two providers at *import* time to express
something better said at the point of use — and the substitution is the real
hazard: a "trade" derived from an OHLCV row is a fiction every
microstructure measure downstream would treat as fact.

**Timestamps are nanoseconds.** The aggregates endpoint used by `get_ohlcv`
returns milliseconds. Parsing one with the other's unit dates every tick to
1970 while leaving the frame structurally plausible, so the two paths are
kept deliberately separate.

**The range is half-open, `[start, end)`.** A closed range on a nanosecond
clock either double-counts the boundary tick when two windows are
concatenated or drops it, and which one is invisible until someone
concatenates.

**One page per call.** Polygon paginates ticks by cursor and a liquid name
produces millions of trades a day, so following `next_url` automatically
would turn one call into an unbounded download. `limit` caps the page
(50,000 is Polygon's own maximum); narrow the time range for more.

**Depth, from exactly one provider.** `get_quotes` is top of book
everywhere. `get_order_book` is a different call, and `DatabentoProvider` is
its only implementation — the others refuse by name rather than returning
top-of-book twice and calling it depth, which would produce a one-level book
whose imbalance is zero by construction.

That is why `get_liquidity_metrics` still estimates spread with
Corwin-Schultz and Amihud and still says plainly that they are proxies:
those exist for the case where depth is absent, which is every provider but
one and every symbol outside a Databento subscription.

**Queue position is in neither.** It needs an order-level feed (MBO), not
aggregated size at a level, and inferring it from depth would be a guess
wearing a measurement's clothes.

---

## Point-in-time records (`get_point_in_time_records`)

`get_financial_ratios` returns the latest filing with no `as_of`: a fact
about the present. A model trained on history needs the fact **as it was
known on each date**, and the difference is the leak that makes a backtest
look prescient — a quarterly figure describes 30 June, is filed on 29 July
and may be restated in August, and a join on the quarter end reads it a
month early.

```python
records = provider.get_point_in_time_records(
    ["AAPL", "MSFT"],
    "fundamentals",
    ["income_statement.revenues", "income_statement.diluted_earnings_per_share"],
    "2021-01-01",
    "2024-12-31",
)
```

The frame is `modeling.dataset.point_in_time`'s schema: one row per
**version** of a fact with `entity`, `event_time` (the period's end),
`available_time` (the filing date) and one column per field, plus
`fiscal_year`, `fiscal_period` and `timeframe`. A restatement is a second
row with a later `available_time`, never an overwrite. A filing with no
filing date is left out and counted in `records.attrs`, rather than dated
to its period end.

- **`PolygonProvider`** serves `frame_kind="fundamentals"` only, from the
  same `/vX/reference/financials` endpoint, quarterly, walked through every
  page; fields are `<statement>.<key>` paths into a filing's `financials`.
  Its `get_temporal_contract("fundamentals")` declares both timestamps and
  `revisions="unknown"`: Polygon documents amended filings as separate
  results, which would make it `versioned`, but that has not been measured
  on a pulled history and a contract is a claim about the source.
  `modeling.dataset.point_in_time.observed_revisions(records)` is the
  measurement; the dataset builder reports what it observed on every pull.
- **Every other provider refuses by name**, and its contract for
  `fundamentals` says `has_available_time=False` first, so the modeling
  builder never calls this on a provider that cannot serve it.

The consumer is `FeatureScope.POINT_IN_TIME` in the modeling runtime — see
[15_modeling.md](15_modeling.md#a-point-in-time-fundamentals-source).

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
